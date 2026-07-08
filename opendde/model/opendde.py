# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import os
import time
from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from opendde.config.schema import OpenDDEConfig
from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.metrics import (
    FoldCPBenchmarkRecorder,
    measure_foldcp_stage,
)
from opendde.distributed.foldcp.pair_sharding import shard_pair_tensor
from opendde.distributed.foldcp.real_pairformer import (
    distributed_pairformer_stack_single_bridge_update,
)
from opendde.distributed.foldcp.trunk_init import (
    apply_trunk_z_cycle_local,
    build_trunk_z_init_local,
)
from opendde.model import sample_confidence
from opendde.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
)
from opendde.model.modules.confidence import ConfidenceHead
from opendde.model.modules.diffusion import DiffusionModule
from opendde.model.modules.embedders import (
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from opendde.model.modules.head import DistogramHead
from opendde.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from opendde.model.modules.primitives import LinearNoBias
from opendde.model.shape_complementarity import (
    build_shape_comp_pred_outputs,
    compute_shape_complementarity_fields,
    get_shape_comp_atom_mask,
)
from opendde.model.modules.structural_tokens import StructuralTokenExpander
from opendde.model.triangular.layers import LayerNorm
from opendde.model.utils import simple_merge_dict_list
from opendde.utils.logger import get_logger
from opendde.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from opendde.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict



class OpenDDE(nn.Module):
    """
    Implements the OpenDDE prediction loop.
    """

    def __init__(self, configs: OpenDDEConfig) -> None:
        super(OpenDDE, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        torch.backends.cudnn.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed

        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        # Model
        self.input_embedder = InputFeatureEmbedder(**configs.model.input_embedder)
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data["msa"],
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        diffusion_module_configs = copy.deepcopy(configs.model.diffusion_module.to_dict())

        self.diffusion_module = DiffusionModule(**diffusion_module_configs)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)
        structural_token_expansion_configs = configs.model.structural_token_expansion
        self.enable_structural_token_expansion = (
            structural_token_expansion_configs.enable
        )
        self.pair_output_space = structural_token_expansion_configs.pair_output_space
        if self.pair_output_space not in {"residue", "structural"}:
            raise ValueError(
                "model.structural_token_expansion.pair_output_space must be "
                f"'residue' or 'structural'; got {self.pair_output_space!r}"
            )
        structural_refiner_configs = (
            structural_token_expansion_configs.structural_refiner
        )
        self.enable_structural_token_refiner = (
            self.enable_structural_token_expansion
            and structural_refiner_configs.enable
        )
        if self.enable_structural_token_expansion:
            required_n_roles = max(STRUCTURAL_TOKEN_ROLES.values()) + 1
            configured_n_roles = structural_token_expansion_configs.n_roles
            if configured_n_roles < required_n_roles:
                raise ValueError(
                    "model.structural_token_expansion.n_roles="
                    f"{configured_n_roles} is too small; need at least "
                    f"{required_n_roles} for structural token roles "
                    f"{STRUCTURAL_TOKEN_ROLES}"
                )
            self.structural_token_expander = StructuralTokenExpander(
                c_s=self.c_s,
                c_z=self.c_z,
                c_s_inputs=self.c_s_inputs,
                n_roles=configured_n_roles,
                init_mode=structural_token_expansion_configs.init_mode,
                role_init_std=structural_token_expansion_configs.role_init_std,
                pair_feature_init_std=(
                    structural_token_expansion_configs.pair_feature_init_std
                ),
                attention_bias_init=(
                    structural_token_expansion_configs.attention_bias_init
                ),
                pair_projection_mode=(
                    structural_token_expansion_configs.pair_projection_mode
                ),
                pair_chunk_size=structural_token_expansion_configs.pair_chunk_size,
            )
            if self.enable_structural_token_refiner:
                self.structural_token_refiner = PairformerStack(
                    n_blocks=structural_refiner_configs.n_blocks,
                    n_heads=structural_refiner_configs.n_heads,
                    c_z=self.c_z,
                    c_s=self.c_s,
                    num_intermediate_factor=(
                        structural_refiner_configs.num_intermediate_factor
                    ),
                    blocks_per_ckpt=structural_refiner_configs.blocks_per_ckpt,
                    hidden_scale_up=structural_refiner_configs.hidden_scale_up,
                )

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    @staticmethod
    def _maybe_foldcp_mesh() -> Optional[FoldCPProcessMesh]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        if torch.is_grad_enabled():
            return None
        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        return FoldCPProcessMesh.create(foldcp)

    @staticmethod
    def _maybe_foldcp_config() -> Optional[FoldCPConfig]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        if not foldcp.metrics_jsonl:
            return None
        return foldcp

    def _foldcp_stage_context(self, stage_name: str, n_token: int):
        foldcp = self._maybe_foldcp_config()
        if foldcp is None:
            return nullcontext()
        rank = dist.get_rank() if dist.is_initialized() else 0
        recorder = FoldCPBenchmarkRecorder(
            foldcp.metrics_jsonl,
            rank=rank,
            write_rank_sidecar=stage_name == "opendde_pre_sample" + "_p2p_warmup",
        )
        return measure_foldcp_stage(
            task_id="i69",
            stage_name=stage_name,
            foldcp_config=foldcp,
            recorder=recorder,
            sample_name="opendde_model",
            n_token=n_token,
            reset_peak=False,
            record_start=True,
        )

    @staticmethod
    def _foldcp_cleanup_before_p2p_warmup() -> None:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def expand_to_structural_tokens(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.enable_structural_token_expansion:
            return input_feature_dict, s_inputs, s, z

        required_features = [
            "parent_residue_idx",
            "subtoken_role_id",
            "structural_token_index",
            "atom_to_structural_token_idx",
            "atom_to_structural_tokatom_idx",
            "structural_distogram_rep_atom_mask",
            "structural_pae_rep_atom_mask",
            "structural_has_frame",
            "structural_frame_atom_index",
        ]
        missing_features = [
            key for key in required_features if key not in input_feature_dict
        ]
        if missing_features:
            raise KeyError(
                "Structural token expansion is enabled, but input_feature_dict is "
                "missing required structural feature(s): "
                + ", ".join(missing_features)
            )

        parent = input_feature_dict["parent_residue_idx"].long()
        structural_feature_dict = dict(input_feature_dict)
        for residue_feature in [
            "token_index",
            "asym_id",
            "residue_index",
            "entity_id",
            "sym_id",
            "atom_to_token_idx",
            "atom_to_tokatom_idx",
            "has_frame",
            "frame_atom_index",
            "pae_rep_atom_mask",
            "distogram_rep_atom_mask",
        ]:
            structural_feature_dict[f"residue_level_{residue_feature}"] = (
                input_feature_dict[residue_feature]
            )
        foldcp_result = self._maybe_expand_to_structural_tokens_foldcp_local(
            input_feature_dict=input_feature_dict,
            structural_feature_dict=structural_feature_dict,
            s_inputs=s_inputs,
            s=s,
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if foldcp_result is not None:
            return foldcp_result

        s_inputs, s, z, structural_pair_features = self.structural_token_expander(
            input_feature_dict=input_feature_dict,
            s_inputs_res=s_inputs,
            s_res=s,
            z_res=z,
        )

        structural_feature_dict["token_index"] = input_feature_dict[
            "structural_token_index"
        ].long()
        structural_feature_dict["atom_to_token_idx"] = input_feature_dict[
            "atom_to_structural_token_idx"
        ].long()
        structural_feature_dict["atom_to_tokatom_idx"] = input_feature_dict[
            "atom_to_structural_tokatom_idx"
        ].long()
        for token_feature in ["asym_id", "residue_index", "entity_id", "sym_id"]:
            structural_feature_dict[token_feature] = input_feature_dict[
                token_feature
            ].index_select(dim=-1, index=parent)

        structural_feature_dict["has_frame"] = input_feature_dict[
            "structural_has_frame"
        ]
        structural_feature_dict["frame_atom_index"] = input_feature_dict[
            "structural_frame_atom_index"
        ]
        structural_feature_dict["pae_rep_atom_mask"] = input_feature_dict[
            "structural_pae_rep_atom_mask"
        ].long()
        structural_feature_dict["distogram_rep_atom_mask"] = input_feature_dict[
            "structural_distogram_rep_atom_mask"
        ].long()
        for feature_name, feature_value in structural_pair_features.items():
            structural_feature_dict[feature_name] = feature_value
        lazy_structural_relp = os.environ.get("OPENDDE_FOLDCP_MODE") == "distributed"
        structural_feature_dict = self.relative_position_encoding.generate_relp(
            structural_feature_dict,
            lazy=lazy_structural_relp,
        )
        if self.enable_structural_token_refiner:
            s, z = self.structural_token_refiner(
                s=s,
                z=z,
                pair_mask=None,
                triangle_multiplicative=self.configs.triangle_multiplicative,
                triangle_attention=self.configs.triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                extra_attn_bias=structural_feature_dict.get(
                    "structural_pair_attn_bias", None
                ),
            )
        self.drop_residue_only_features_for_structural_branch(structural_feature_dict)
        return structural_feature_dict, s_inputs, s, z

    def _maybe_expand_to_structural_tokens_foldcp_local(
        self,
        input_feature_dict: dict[str, Any],
        structural_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Optional[tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]]:
        mesh = self._maybe_foldcp_mesh()
        if mesh is None:
            return None

        parent = input_feature_dict["parent_residue_idx"].long()

        (
            s_inputs,
            s,
            z_local,
            structural_pair_features_local,
            structural_pair_specs,
            z_spec,
        ) = self.structural_token_expander.forward_foldcp_local_pair(
            input_feature_dict=input_feature_dict,
            s_inputs_res=s_inputs,
            s_res=s,
            z_res=z,
            mesh=mesh,
            z_res_spec=input_feature_dict.get("_foldcp_pair_z_spec"),
        )

        structural_feature_dict["token_index"] = input_feature_dict[
            "structural_token_index"
        ].long()
        structural_feature_dict["atom_to_token_idx"] = input_feature_dict[
            "atom_to_structural_token_idx"
        ].long()
        structural_feature_dict["atom_to_tokatom_idx"] = input_feature_dict[
            "atom_to_structural_tokatom_idx"
        ].long()
        for token_feature in ["asym_id", "residue_index", "entity_id", "sym_id"]:
            structural_feature_dict[token_feature] = input_feature_dict[
                token_feature
            ].index_select(dim=-1, index=parent)

        structural_feature_dict["has_frame"] = input_feature_dict[
            "structural_has_frame"
        ]
        structural_feature_dict["frame_atom_index"] = input_feature_dict[
            "structural_frame_atom_index"
        ]
        for feature_name, feature_value in structural_pair_features_local.items():
            structural_feature_dict[feature_name] = feature_value.contiguous()
        structural_feature_dict["_foldcp_pair_feature_specs"] = structural_pair_specs
        structural_feature_dict["_foldcp_pair_features_are_local"] = True

        structural_feature_dict = self.relative_position_encoding.generate_relp(
            structural_feature_dict,
            lazy=True,
        )
        if self.enable_structural_token_refiner:
            s, z_local, z_spec = distributed_pairformer_stack_single_bridge_update(
                self.structural_token_refiner,
                s,
                z_local,
                mesh,
                pair_mask=None,
                extra_attn_bias=structural_pair_features_local.get(
                    "structural_pair_attn_bias",
                    None,
                ),
                extra_attn_bias_is_local=True,
                return_local_pair=True,
                z_spec=z_spec,
            )
        structural_feature_dict["_foldcp_pair_z_spec"] = z_spec
        self.drop_residue_only_features_for_structural_branch(structural_feature_dict)
        return structural_feature_dict, s_inputs, s, z_local.contiguous()

    def select_pair_output_branch(
        self,
        residue_feature_dict: dict[str, Any],
        residue_s_inputs: torch.Tensor,
        residue_s: torch.Tensor,
        residue_z: torch.Tensor,
        structural_feature_dict: dict[str, Any],
        structural_s_inputs: torch.Tensor,
        structural_s: torch.Tensor,
        structural_z: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.enable_structural_token_expansion and self.pair_output_space == "residue":
            return residue_feature_dict, residue_s_inputs, residue_s, residue_z
        return structural_feature_dict, structural_s_inputs, structural_s, structural_z

    @staticmethod
    def drop_residue_only_features_for_structural_branch(
        input_feature_dict: dict[str, Any]
    ) -> None:
        """
        Keep MSA/template strictly residue-level.

        After trunk, structural-token branches consume expanded s/z plus structural
        metadata. Residue-column features are removed from this local branch dict so
        downstream code cannot accidentally treat them as structural-token features.
        """
        residue_only_keys = {
            "msa",
            "has_deletion",
            "deletion_value",
            "msa_mask",
            "profile",
            "deletion_mean",
            "token_bonds",
        }
        for key in list(input_feature_dict.keys()):
            if key in residue_only_keys or key.startswith("template_"):
                input_feature_dict.pop(key, None)

    @staticmethod
    def pool_pair_matrix_to_residue(
        values: torch.Tensor, parent: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        n_struct = parent.numel()
        pair_index = (
            parent[:, None] * n_residue + parent[None, :]
        ).reshape(-1).to(device=values.device)
        flat_values = values.reshape(*values.shape[:-2], n_struct * n_struct)
        prefix_shape = flat_values.shape[:-1]
        out = values.new_zeros(*prefix_shape, n_residue * n_residue)
        index = pair_index.reshape((1,) * len(prefix_shape) + (-1,)).expand(
            *prefix_shape, n_struct * n_struct
        )
        out.scatter_add_(dim=-1, index=index, src=flat_values)
        counts = values.new_zeros(n_residue * n_residue)
        counts.scatter_add_(
            dim=0,
            index=pair_index,
            src=values.new_ones(n_struct * n_struct),
        )
        out = out / counts.clamp_min(1).reshape((1,) * len(prefix_shape) + (-1,))
        return out.reshape(*prefix_shape, n_residue, n_residue)

    @staticmethod
    def pool_pair_matrix_to_residue_max(
        values: torch.Tensor, parent: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        n_struct = parent.numel()
        pair_index = (
            parent[:, None] * n_residue + parent[None, :]
        ).reshape(-1).to(device=values.device)
        flat_values = values.reshape(*values.shape[:-2], n_struct * n_struct)
        prefix_shape = flat_values.shape[:-1]
        out = values.new_full(
            (*prefix_shape, n_residue * n_residue),
            torch.finfo(values.dtype).min,
        )
        index = pair_index.reshape((1,) * len(prefix_shape) + (-1,)).expand(
            *prefix_shape, n_struct * n_struct
        )
        out.scatter_reduce_(
            dim=-1,
            index=index,
            src=flat_values,
            reduce="amax",
            include_self=True,
        )
        return out.reshape(*prefix_shape, n_residue, n_residue)

    @staticmethod
    def pool_pair_distribution_to_residue(
        probs: torch.Tensor, parent: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        n_struct = parent.numel()
        n_bins = probs.shape[-1]
        pair_index = (
            parent[:, None] * n_residue + parent[None, :]
        ).reshape(-1).to(device=probs.device)
        flat_probs = probs.reshape(*probs.shape[:-3], n_struct * n_struct, n_bins)
        prefix_shape = flat_probs.shape[:-2]
        out = probs.new_zeros(*prefix_shape, n_residue * n_residue, n_bins)
        index = pair_index.reshape((1,) * len(prefix_shape) + (-1, 1)).expand(
            *prefix_shape, n_struct * n_struct, n_bins
        )
        out.scatter_add_(dim=-2, index=index, src=flat_probs)
        counts = probs.new_zeros(n_residue * n_residue)
        counts.scatter_add_(
            dim=0,
            index=pair_index,
            src=probs.new_ones(n_struct * n_struct),
        )
        out = out / counts.clamp_min(1).reshape(
            (1,) * len(prefix_shape) + (-1, 1)
        )
        return out.reshape(*prefix_shape, n_residue, n_residue, n_bins)

    @staticmethod
    def get_parent_representative_token_idx(
        parent: torch.Tensor, role: torch.Tensor, n_residue: int
    ) -> torch.Tensor:
        """
        Pick the public residue representative in structural-token space.

        Polymer residues prefer their BB token so public PAE/PDE preserve
        backbone-frame residue semantics. Non-polymer/atom-token parents fall
        back to their first structural token.
        """
        backbone_roles = {
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_bb"],
            STRUCTURAL_TOKEN_ROLES["rna_bb"],
        }
        parent_list = parent.detach().cpu().tolist()
        role_list = role.detach().cpu().tolist()
        representative_idx = [-1] * n_residue
        for token_idx, parent_idx in enumerate(parent_list):
            if representative_idx[parent_idx] < 0:
                representative_idx[parent_idx] = token_idx
        for token_idx, (parent_idx, role_id) in enumerate(zip(parent_list, role_list)):
            if role_id in backbone_roles:
                representative_idx[parent_idx] = token_idx
        if any(token_idx < 0 for token_idx in representative_idx):
            raise ValueError(
                "Could not find a structural representative token for every parent "
                f"residue: {representative_idx}"
            )
        return torch.tensor(
            representative_idx, dtype=torch.long, device=parent.device
        )

    def get_residue_level_confidence_inputs(
        self,
        input_feature_dict: dict[str, Any],
        pae_logits: torch.Tensor,
        pde_logits: torch.Tensor,
        contact_probs: torch.Tensor,
        target_device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        target_device = target_device or pae_logits.device
        parent = input_feature_dict.get("parent_residue_idx", None)
        role = input_feature_dict.get("subtoken_role_id", None)
        residue_level_keys = (
            "residue_level_asym_id",
            "residue_level_has_frame",
            "residue_level_atom_to_token_idx",
        )
        has_residue_level_features = all(
            key in input_feature_dict for key in residue_level_keys
        )
        is_structural_pair_space = (
            parent is not None
            and role is not None
            and has_residue_level_features
            and pae_logits.shape[-3] == parent.numel()
            and pae_logits.shape[-2] == parent.numel()
            and pde_logits.shape[-3] == parent.numel()
            and pde_logits.shape[-2] == parent.numel()
            and contact_probs.shape[-2] == parent.numel()
            and contact_probs.shape[-1] == parent.numel()
        )
        if not is_structural_pair_space:
            return {
                "pae_logits": pae_logits.to(device=target_device),
                "pde_logits": pde_logits.to(device=target_device),
                "contact_probs": contact_probs.to(device=target_device),
                "token_asym_id": input_feature_dict["asym_id"].to(
                    device=target_device
                ),
                "token_has_frame": input_feature_dict["has_frame"].to(
                    device=target_device
                ),
                "atom_to_token_idx": input_feature_dict["atom_to_token_idx"].to(
                    device=target_device
                ),
            }

        parent = parent.long().to(device=pae_logits.device)
        n_residue = int(parent.max().item()) + 1
        representative_idx = self.get_parent_representative_token_idx(
            parent=parent,
            role=role.long().to(device=pae_logits.device),
            n_residue=n_residue,
        )
        residue_pae_logits = pae_logits.index_select(
            dim=-3, index=representative_idx
        ).index_select(dim=-2, index=representative_idx)
        residue_pde_logits = pde_logits.index_select(
            dim=-3, index=representative_idx
        ).index_select(dim=-2, index=representative_idx)
        pooled_contact_probs = self.pool_pair_matrix_to_residue_max(
            values=contact_probs.to(device=pae_logits.device, dtype=torch.float32),
            parent=parent,
            n_residue=n_residue,
        )
        return {
            "pae_logits": residue_pae_logits.to(
                device=target_device, dtype=pae_logits.dtype
            ),
            "pde_logits": residue_pde_logits.to(
                device=target_device, dtype=pde_logits.dtype
            ),
            "contact_probs": pooled_contact_probs.to(
                device=target_device, dtype=contact_probs.dtype
            ),
            "token_asym_id": input_feature_dict["residue_level_asym_id"].to(
                device=target_device
            ),
            "token_has_frame": input_feature_dict["residue_level_has_frame"].to(
                device=target_device
            ),
            "atom_to_token_idx": input_feature_dict[
                "residue_level_atom_to_token_idx"
            ].to(
                device=target_device
            ),
        }

    def _shape_comp_effective_weight(self, weight_name: str) -> float:
        shape_comp_configs = self.configs.confidence.shape_comp
        return float(self.configs.confidence.weight.alpha_shape_comp) * float(
            getattr(shape_comp_configs, weight_name)
        )

    def _should_store_shape_comp_pair_map(self) -> bool:
        shape_comp_configs = self.configs.confidence.shape_comp
        # Pair maps are O(N_sample * N_token^2), so keep them only for debug output.
        return bool(shape_comp_configs.debug_pair_map)

    def _should_compute_shape_comp(self) -> bool:
        shape_comp_configs = self.configs.confidence.shape_comp
        return bool(
            self._shape_comp_effective_weight("pair_weight") > 0
            or self._shape_comp_effective_weight("token_weight") > 0
            or self._shape_comp_effective_weight("global_weight") > 0
            or shape_comp_configs.debug_pair_map
        )

    def add_shape_complementarity_predictions(
        self,
        pred_dict: dict[str, torch.Tensor],
        input_feature_dict: dict[str, Any],
        coordinate: torch.Tensor,
        label_dict: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self._should_compute_shape_comp():
            return

        keep_pair_map = self._should_store_shape_comp_pair_map()
        pred_dict["shape_comp_uses_structural_tokens"] = torch.tensor(
            [int("residue_level_token_index" in input_feature_dict)],
            dtype=torch.long,
            device=coordinate.device,
        )
        shape_comp = autocasting_disable_decorator(True)(
            compute_shape_complementarity_fields
        )(
            coordinate=coordinate,
            feat_dict=input_feature_dict,
            atom_mask=get_shape_comp_atom_mask(
                feat_dict=input_feature_dict,
                label_dict=label_dict,
            ),
            return_pair_map=keep_pair_map,
            **self.configs.confidence.shape_comp,
        )
        pred_dict.update(
            build_shape_comp_pred_outputs(
                shape_comp=shape_comp,
                keep_pair_map=keep_pair_map,
            )
        )

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        foldcp_result = self._maybe_get_pairformer_output_foldcp_local(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            s_inputs=s_inputs,
            s_init=s_init,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        if foldcp_result is not None:
            return foldcp_result

        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(False):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return s_inputs, s, z

    def _maybe_get_pairformer_output_foldcp_local(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        s_inputs: torch.Tensor,
        s_init: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        mesh = self._maybe_foldcp_mesh()
        if mesh is None:
            return None

        z_init_local, z_spec = build_trunk_z_init_local(
            s_init=s_init,
            linear_zinit1=self.linear_no_bias_zinit1,
            linear_zinit2=self.linear_no_bias_zinit2,
            relative_position_encoding=self.relative_position_encoding,
            linear_token_bond=self.linear_no_bias_token_bond,
            relp_feature=input_feature_dict["relp"],
            token_bonds=input_feature_dict["token_bonds"],
            mesh=mesh,
        )
        z_local = torch.zeros_like(z_init_local)
        s = torch.zeros_like(s_init)

        for _ in range(N_cycle):
            with torch.set_grad_enabled(False):
                z_local = apply_trunk_z_cycle_local(
                    z_init_local=z_init_local,
                    z_local=z_local,
                    layernorm_z_cycle=self.layernorm_z_cycle,
                    linear_z_cycle=self.linear_no_bias_z_cycle,
                    z_spec=z_spec,
                )
                if self.template_embedder.n_blocks > 0:
                    template_update_local, z_spec = (
                        self.template_embedder.forward_foldcp_local_pair(
                            input_feature_dict=input_feature_dict,
                            z_local=z_local,
                            z_spec=z_spec,
                            mesh=mesh,
                            pair_mask=None,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    )
                    if template_update_local is not None:
                        z_local = z_local + template_update_local
                z_local, z_spec = self.msa_module.forward_foldcp_local_pair(
                    input_feature_dict=input_feature_dict,
                    z_local=z_local,
                    z_spec=z_spec,
                    s_inputs=s_inputs,
                    pair_mask=None,
                    mesh=mesh,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z_local, z_spec = distributed_pairformer_stack_single_bridge_update(
                    self.pairformer_stack,
                    s,
                    z_local,
                    mesh,
                    pair_mask=None,
                    return_local_pair=True,
                    z_spec=z_spec,
                )

        input_feature_dict["_foldcp_pair_z_spec"] = z_spec
        input_feature_dict["_foldcp_pair_is_local"] = True
        return s_inputs, s, z_local.contiguous()

    def run_sample_diffusion_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_z_spec: Optional[Any],
        cache: dict[str, Any],
        N_sample: int,
        noise_schedule: Any,
        chunk_size: Optional[int],
        inplace_safe: bool,
    ) -> torch.Tensor:
        sample_pair_z_spec = cache["pair_z_spec"]
        if sample_pair_z_spec is None and cache["pair_z"] is None:
            sample_pair_z_spec = pair_z_spec
        sample_input_feature_dict = input_feature_dict
        sample_pair_z = cache["pair_z"]
        sample_z_trunk = None if sample_pair_z is not None else z
        rollout_seed = input_feature_dict.get("inference_seed")
        if isinstance(rollout_seed, torch.Tensor):
            rollout_seed = int(rollout_seed.detach().cpu().item())
        elif rollout_seed is not None:
            rollout_seed = int(rollout_seed)

        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=sample_input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=sample_z_trunk,
            pair_z=sample_pair_z,
            pair_z_spec=sample_pair_z_spec,
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            attn_chunk_size=chunk_size,
            diffusion_chunk_size=self.configs.infer_setting.sample_diffusion_chunk_size,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
            rollout_seed=rollout_seed,
        )
        return pred_dict["coordinate"]

    def sample_diffusion(
        self,
        attn_chunk_size: Optional[int] = None,
        diffusion_chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Samples diffusion process based on the provided configurations.

        Args:
            attn_chunk_size (Optional[int]): Token chunk size used inside attention-style blocks.
            diffusion_chunk_size (Optional[int]): Chunk size used to split diffusion samples.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        sample_diffusion_configs = self.configs.sample_diffusion.to_dict()
        _configs.update(
            {
                "attn_chunk_size": attn_chunk_size,
                "diffusion_chunk_size": diffusion_chunk_size,
                "guidance_configs": sample_diffusion_configs.get("guidance"),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)


    @staticmethod
    def _foldcp_is_non_output_rank() -> bool:
        return (
            os.environ.get("OPENDDE_FOLDCP_MODE", "single") == "distributed"
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        )

    def prepare_diffusion_cache_for_sampling(
        self,
        *,
        input_feature_dict: dict[str, Any],
        z: torch.Tensor,
        foldcp_mesh: Optional[Any],
        diffusion_z_spec: Optional[Any],
    ) -> dict[str, Any]:
        cache: dict[str, Any] = {"pair_z_spec": None}
        if self.enable_diffusion_shared_vars_cache:
            if foldcp_mesh is not None and diffusion_z_spec is not None:
                cache["pair_z"], cache["pair_z_spec"] = autocasting_disable_decorator(
                    self.configs.skip_amp.sample_diffusion
                )(
                    self.diffusion_module.diffusion_conditioning.prepare_cache_foldcp_local
                )(
                    input_feature_dict["relp"],
                    z,
                    diffusion_z_spec,
                    foldcp_mesh,
                    False,
                )
            else:
                cache["pair_z"] = autocasting_disable_decorator(
                    self.configs.skip_amp.sample_diffusion
                )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                    input_feature_dict["relp"], z, False
                )
            if os.environ.get("OPENDDE_FOLDCP_MODE", "single") == "distributed":
                cache["p_lm/c_l"] = [None, None]
            else:
                cache["p_lm/c_l"] = autocasting_disable_decorator(
                    self.configs.skip_amp.sample_diffusion
                )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                    ref_pos=input_feature_dict["ref_pos"],
                    ref_charge=input_feature_dict["ref_charge"],
                    ref_mask=input_feature_dict["ref_mask"],
                    ref_element=input_feature_dict["ref_element"],
                    ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                    atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                    d_lm=input_feature_dict["d_lm"],
                    v_lm=input_feature_dict["v_lm"],
                    pad_info=input_feature_dict["pad_info"],
                    r_l=True,
                    z=cache["pair_z"],
                    inplace_safe=False,
                )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        return cache

    def compute_distogram_contact_probs(
        self,
        pair_z: torch.Tensor,
        pair_z_spec: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        bin_params = sample_confidence.get_bin_params(self.configs.confidence.distogram)
        mesh = self._maybe_foldcp_mesh()
        if mesh is not None:
            if pair_z_spec is not None:
                return self.distogram_head.contact_probs_foldcp_local(
                    z_pair_local=pair_z,
                    z_pair_spec=pair_z_spec,
                    mesh=mesh,
                    gather_to_rank0_only=True,
                    **bin_params,
                )
            return self.distogram_head.contact_probs_foldcp_from_full_pair(
                z_pair=pair_z,
                mesh=mesh,
                gather_to_rank0_only=True,
                **bin_params,
            )
        return sample_confidence.compute_contact_prob(
            distogram_logits=self.distogram_head(pair_z),
            **bin_params,
        )

    def run_distogram_contact_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        pair_z: torch.Tensor,
        pair_z_spec: Optional[Any],
    ) -> Optional[torch.Tensor]:
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            self.compute_distogram_contact_probs
        )(
            pair_z,
            pair_z_spec=pair_z_spec,
        )
        return pred_dict["contact_probs"]

    def run_post_confidence_outputs_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        pair_input_feature_dict: dict[str, Any],
        pair_z: torch.Tensor,
        N_cycle: int,
    ) -> dict[str, Any]:
        del pair_z
        torch.cuda.empty_cache()

        self.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=pair_input_feature_dict,
            coordinate=pred_dict["coordinate"],
            label_dict=None,
        )

        residue_confidence_inputs = self.get_residue_level_confidence_inputs(
            input_feature_dict=pair_input_feature_dict,
            pae_logits=pred_dict["pae"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            target_device=pred_dict["plddt"].device,
        )
        self.replace_public_pair_logits_with_residue_level(
            pred_dict=pred_dict,
            residue_confidence_inputs=residue_confidence_inputs,
        )
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=residue_confidence_inputs["pae_logits"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=residue_confidence_inputs["pde_logits"],
            contact_probs=residue_confidence_inputs["contact_probs"],
            token_asym_id=residue_confidence_inputs["token_asym_id"],
            token_has_frame=residue_confidence_inputs["token_has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=residue_confidence_inputs["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=None,
            return_full_data=bool(getattr(self.configs, "need_atom_confidence", False)),
            mol_id=None,
            elements_one_hot=None,
        )
        return pred_dict

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def run_confidence_head_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        z_trunk_spec: Optional[Any],
        pair_mask: Optional[torch.Tensor],
        x_pred_coords: torch.Tensor,
        triangle_multiplicative: str,
        triangle_attention: str,
        inplace_safe: bool,
        chunk_size: Optional[int],
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        plddt, pae, pde, resolved = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            z_trunk_spec=z_trunk_spec,
            pair_mask=pair_mask,
            x_pred_coords=x_pred_coords,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        self.update_confidence_predictions(
            pred_dict=pred_dict,
            plddt=plddt,
            pae=pae,
            pde=pde,
            resolved=resolved,
        )
        return plddt, pae, pde, resolved

    @staticmethod
    def update_confidence_predictions(
        pred_dict: dict[str, Any],
        plddt: Optional[torch.Tensor],
        pae: Optional[torch.Tensor],
        pde: Optional[torch.Tensor],
        resolved: Optional[torch.Tensor],
    ) -> None:
        for key, value in {
            "plddt": plddt,
            "pae": pae,
            "pde": pde,
            "resolved": resolved,
        }.items():
            if value is not None:
                pred_dict[key] = value

    @staticmethod
    def replace_public_pair_logits_with_residue_level(
        pred_dict: dict[str, Any],
        residue_confidence_inputs: dict[str, torch.Tensor],
    ) -> None:
        if (
            pred_dict["pae"].shape[-3]
            == residue_confidence_inputs["pae_logits"].shape[-3]
            and pred_dict["pae"].shape[-2]
            == residue_confidence_inputs["pae_logits"].shape[-2]
        ):
            pred_dict["pae"] = residue_confidence_inputs["pae_logits"]
            pred_dict["pde"] = residue_confidence_inputs["pde_logits"]
        else:
            pred_dict["structural_pae"] = pred_dict["pae"]
            pred_dict["structural_pde"] = pred_dict["pde"]
            pred_dict["pae"] = residue_confidence_inputs["pae_logits"]
            pred_dict["pde"] = residue_confidence_inputs["pde_logits"]
        pred_dict["contact_probs"] = residue_confidence_inputs["contact_probs"]
        pred_dict.pop("per_sample_contact_probs", None)

    def run_post_confidence_outputs_stage(
        self,
        *,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        pair_input_feature_dict: dict[str, Any],
        pair_z: torch.Tensor,
        N_cycle: int,
    ) -> dict[str, Any]:
        del pair_z
        torch.cuda.empty_cache()

        self.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=pair_input_feature_dict,
            coordinate=pred_dict["coordinate"],
            label_dict=None,
        )

        residue_confidence_inputs = self.get_residue_level_confidence_inputs(
            input_feature_dict=pair_input_feature_dict,
            pae_logits=pred_dict["pae"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            target_device=pred_dict["plddt"].device,
        )
        self.replace_public_pair_logits_with_residue_level(
            pred_dict=pred_dict,
            residue_confidence_inputs=residue_confidence_inputs,
        )
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=residue_confidence_inputs["pae_logits"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=residue_confidence_inputs["pde_logits"],
            contact_probs=residue_confidence_inputs["contact_probs"],
            token_asym_id=residue_confidence_inputs["token_asym_id"],
            token_has_frame=residue_confidence_inputs["token_has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=residue_confidence_inputs["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=None,
            return_full_data=bool(
                getattr(self.configs, "need_atom_confidence", False)
            ),
            mol_id=None,
            elements_one_hot=None,
        )
        return pred_dict

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop, optionally evaluating multiple model seeds.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            N_cycle (int): Number of cycles.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        if N_model_seed > 1:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=copy.deepcopy(input_feature_dict),
                    N_cycle=N_cycle,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            if self._foldcp_is_non_output_rank():
                return (
                    pred_dicts[-1],
                    simple_merge_dict_list(log_dicts),
                    simple_merge_dict_list(time_trackers),
                )

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        dynamic_chunk_size = (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        )
        if dynamic_chunk_size:
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        log_dict = {}
        pred_dict = {}
        time_tracker = {}
        with self._foldcp_stage_context("opendde_pairformer_trunk", N_token):
            s_inputs, s, z = self.get_pairformer_output(
                input_feature_dict=input_feature_dict,
                N_cycle=N_cycle,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        if self._maybe_foldcp_mesh() is not None and s_inputs.is_cuda:
            torch.cuda.empty_cache()
        residue_input_feature_dict = input_feature_dict
        residue_s_inputs, residue_s, residue_z = s_inputs, s, z
        structural_chunk_size = chunk_size
        if dynamic_chunk_size and self.enable_structural_token_expansion:
            structural_chunk_size = self._get_dynamic_chunk_size(
                input_feature_dict["parent_residue_idx"].shape[-1]
            )
        with self._foldcp_stage_context("opendde_structural_token_expansion", N_token):
            input_feature_dict, s_inputs, s, z = self.expand_to_structural_tokens(
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s=s,
                z=z,
                inplace_safe=inplace_safe,
                chunk_size=structural_chunk_size,
            )
        with self._foldcp_stage_context("opendde_pair_output_branch", N_token):
            (
                pair_input_feature_dict,
                pair_s_inputs,
                pair_s,
                pair_z,
            ) = self.select_pair_output_branch(
                residue_feature_dict=residue_input_feature_dict,
                residue_s_inputs=residue_s_inputs,
                residue_s=residue_s,
                residue_z=residue_z,
                structural_feature_dict=input_feature_dict,
                structural_s_inputs=s_inputs,
                structural_s=s,
                structural_z=z,
            )
        foldcp_mesh = self._maybe_foldcp_mesh()
        pair_z_spec = pair_input_feature_dict.get("_foldcp_pair_z_spec")
        diffusion_z_spec = input_feature_dict.get("_foldcp_pair_z_spec")
        if foldcp_mesh is None:
            pair_z_spec = None
            diffusion_z_spec = None
        del residue_input_feature_dict, residue_s_inputs, residue_s, residue_z
        if dynamic_chunk_size:
            chunk_size = structural_chunk_size

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        with self._foldcp_stage_context("opendde_pre_sample_p2p_warmup", N_token):
            self._foldcp_cleanup_before_p2p_warmup()
            if (
                foldcp_mesh is not None
                and (diffusion_z_spec is not None or pair_z_spec is not None)
                and s_inputs.is_cuda
            ):
                self.diffusion_module.atom_attention_encoder._warmup_foldcp_atom_window_p2p(
                    mesh=foldcp_mesh,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                )
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        with self._foldcp_stage_context("opendde_diffusion_cache", N_token):
            cache = self.prepare_diffusion_cache_for_sampling(
                input_feature_dict=input_feature_dict,
                z=z,
                foldcp_mesh=foldcp_mesh,
                diffusion_z_spec=diffusion_z_spec,
            )
        if foldcp_mesh is not None and s_inputs.is_cuda:
            torch.cuda.empty_cache()
        with self._foldcp_stage_context("opendde_sample_diffusion", N_token):
            self.run_sample_diffusion_stage(
                pred_dict=pred_dict,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s=s,
                z=z,
                pair_z_spec=pair_z_spec,
                cache=cache,
                N_sample=N_sample,
                noise_schedule=noise_schedule,
                chunk_size=chunk_size,
                inplace_safe=inplace_safe,
            )

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        # Distogram contact probabilities are the only public output needed here.
        # In distributed mode, avoid materializing full [N, N, bins] logits.
        with self._foldcp_stage_context("opendde_distogram_contact", N_token):
            self.run_distogram_contact_stage(
                pred_dict=pred_dict,
                pair_z=pair_z,
                pair_z_spec=pair_z_spec,
            )  # [N_token, N_token] on output rank; None on non-output CP ranks.

        # Confidence logits
        with self._foldcp_stage_context("opendde_confidence_head", N_token):
            self.run_confidence_head_stage(
                pred_dict=pred_dict,
                input_feature_dict=pair_input_feature_dict,
                s_inputs=pair_s_inputs,
                s_trunk=pair_s,
                z_trunk=pair_z,
                z_trunk_spec=pair_z_spec,
                pair_mask=None,
                x_pred_coords=pred_dict["coordinate"],
                triangle_multiplicative=self.configs.triangle_multiplicative,
                triangle_attention=self.configs.triangle_attention,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})
        if self._foldcp_is_non_output_rank():
            return pred_dict, log_dict, time_tracker

        with self._foldcp_stage_context("opendde_post_confidence_outputs", N_token):
            self.run_post_confidence_outputs_stage(
                pred_dict=pred_dict,
                input_feature_dict=input_feature_dict,
                pair_input_feature_dict=pair_input_feature_dict,
                pair_z=pair_z,
                N_cycle=N_cycle,
            )

        return pred_dict, log_dict, time_tracker

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: Optional[dict[str, Any]] = None,
        label_dict: Optional[dict[str, Any]] = None,
        mode: str = "inference",
        disable_inplace: bool = False,
    ) -> tuple[dict[str, torch.Tensor], Optional[dict[str, Any]], dict[str, Any]]:
        """
        Forward pass for structure prediction.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict: Kept for checkpoint/API compatibility; ignored.
            label_dict: Kept for checkpoint/API compatibility; ignored.
            mode: Only "inference" is supported.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, ignored label dictionary, and log dictionary.
        """

        if mode != "inference":
            raise ValueError("OpenDDE only supports mode='inference'.")

        not_use_gradient = not torch.is_grad_enabled()
        inplace_safe = not_use_gradient and (not disable_inplace)

        lazy_relp = os.environ.get("OPENDDE_FOLDCP_MODE") == "distributed"
        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict,
            lazy=lazy_relp,
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        pred_dict, log_dict, time_tracker = self.main_inference_loop(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=self.configs.infer_setting.chunk_size,
            N_model_seed=self.N_model_seed,
        )
        log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
