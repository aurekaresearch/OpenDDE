# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
from typing import Optional, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn

from opendde.distributed.foldcp.confidence import (
    add_confidence_distance_embedding_local,
    distributed_confidence_pair_logits,
)
from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    shard_pair_tensor,
)
from opendde.distributed.foldcp.real_pairformer import (
    distributed_pairformer_stack_single_bridge_update,
)
from opendde.model.modules.pairformer import PairformerStack
from opendde.model.modules.primitives import LinearNoBias
from opendde.model.triangular.layers import LayerNorm
from opendde.model.utils import broadcast_token_to_atom, one_hot


class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3

    Args:
        n_blocks (int, optional): number of blocks for ConfidenceHead. Defaults to 4.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional): hidden dim [for single embedding from InputFeatureEmbedder]. Defaults to 449.
        b_pae (int, optional): the bin number for pae. Defaults to 64.
        b_pde (int, optional): the bin numer for pde. Defaults to 64.
        b_plddt (int, optional): the bin number for plddt. Defaults to 50.
        b_resolved (int, optional): the bin number for resolved. Defaults to 2.
        max_atoms_per_token (int, optional): max atoms in a token. Defaults to 20.
        blocks_per_ckpt: number of Pairformer blocks in each activation checkpoint
        distance_bin_start (float, optional): Start of the distance bin range. Defaults to 3.25.
        distance_bin_end (float, optional): End of the distance bin range. Defaults to 52.0.
        distance_bin_step (float, optional): Step size for the distance bins. Defaults to 1.25.
        hidden_scale_up (bool, optional): Whether to scale up hidden dimension. Defaults to False.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        b_pae: int = 64,
        b_pde: int = 64,
        b_plddt: int = 50,
        b_resolved: int = 2,
        max_atoms_per_token: int = 20,
        blocks_per_ckpt: Optional[int] = None,
        distance_bin_start: float = 3.25,
        distance_bin_end: float = 52.0,
        distance_bin_step: float = 1.25,
        hidden_scale_up: bool = False,
    ) -> None:
        super(ConfidenceHead, self).__init__()
        self.n_blocks = n_blocks
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.b_pae = b_pae
        self.b_pde = b_pde
        self.b_plddt = b_plddt
        self.b_resolved = b_resolved
        self.max_atoms_per_token = max_atoms_per_token
        self.linear_no_bias_s1 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        self.linear_no_bias_s2 = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_z
        )
        lower_bins = torch.arange(
            distance_bin_start, distance_bin_end, distance_bin_step
        )
        upper_bins = torch.cat([lower_bins[1:], lower_bins.new_tensor([1e6])], dim=-1)
        self.lower_bins = nn.Parameter(lower_bins, requires_grad=False)
        self.upper_bins = nn.Parameter(upper_bins, requires_grad=False)
        self.num_bins = len(lower_bins)  # + 1

        self.linear_no_bias_d = LinearNoBias(
            in_features=self.num_bins, out_features=self.c_z
        )
        self.linear_no_bias_d_wo_onehot = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.pairformer_stack = PairformerStack(
            c_z=self.c_z,
            c_s=self.c_s,
            n_blocks=n_blocks,
            blocks_per_ckpt=blocks_per_ckpt,
            hidden_scale_up=hidden_scale_up,
        )
        self.linear_no_bias_pae = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pae
        )
        self.linear_no_bias_pde = LinearNoBias(
            in_features=self.c_z, out_features=self.b_pde
        )
        self.plddt_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_plddt))
        )
        self.resolved_weight = nn.Parameter(
            data=torch.empty(size=(self.max_atoms_per_token, self.c_s, self.b_resolved))
        )

        self.input_strunk_ln = LayerNorm(self.c_s)
        self.pae_ln = LayerNorm(self.c_z)
        self.pde_ln = LayerNorm(self.c_z)
        self.plddt_ln = LayerNorm(self.c_s)
        self.resolved_ln = LayerNorm(self.c_s)

        with torch.no_grad():
            # Zero init for output layer (before softmax) to zero
            nn.init.zeros_(self.linear_no_bias_pae.weight)
            nn.init.zeros_(self.linear_no_bias_pde.weight)
            nn.init.zeros_(self.plddt_weight)
            nn.init.zeros_(self.resolved_weight)
        self._foldcp_mesh: Optional[FoldCPProcessMesh] = None

    @staticmethod
    def _select_distogram_rep_atom_mask(
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        n_token: int,
    ) -> torch.Tensor:
        structural_mask = cast(
            Optional[torch.Tensor],
            input_feature_dict.get("structural_distogram_rep_atom_mask"),
        )
        if (
            structural_mask is not None
            and int(structural_mask.long().sum().item()) == n_token
        ):
            return structural_mask.bool()
        distogram_rep_atom_mask = cast(
            torch.Tensor, input_feature_dict["distogram_rep_atom_mask"]
        )
        return distogram_rep_atom_mask.bool()

    def _maybe_create_foldcp_mesh(self) -> Optional[FoldCPProcessMesh]:
        if os.environ.get("OPENDDE_FOLDCP_MODE", "single") != "distributed":
            return None
        if not dist.is_available() or not dist.is_initialized():
            return None
        if self._foldcp_mesh is not None:
            return self._foldcp_mesh
        foldcp = FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP", "1")),
            size_cp=int(os.environ.get("OPENDDE_FOLDCP_SIZE_CP", "4")),
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )
        self._foldcp_mesh = FoldCPProcessMesh.create(foldcp)
        return self._foldcp_mesh

    @staticmethod
    def _foldcp_is_non_output_rank() -> bool:
        return (
            os.environ.get("OPENDDE_FOLDCP_MODE", "single") == "distributed"
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() != 0
        )

    def _build_confidence_z_init_local(
        self,
        s_inputs: torch.Tensor,
        z_pair_spec: FoldCPPairShardSpec,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        row_dim, col_dim = z_pair_spec.pair_dims
        if row_dim != 0 or col_dim != 1 or reference.ndim != 3:
            raise ValueError("confidence z_init local path expects z=[T,T,C].")

        s1 = self.linear_no_bias_s1(s_inputs)
        s2 = self.linear_no_bias_s2(s_inputs)
        row_start, row_end = z_pair_spec.row_range
        col_start, col_end = z_pair_spec.col_range
        n_token = z_pair_spec.original_shape[row_dim]
        valid_row_end = min(row_end, n_token)
        valid_col_end = min(col_end, n_token)
        z_init_local = reference.new_zeros(z_pair_spec.local_shape)
        if row_start >= valid_row_end or col_start >= valid_col_end:
            return z_init_local
        valid_rows = valid_row_end - row_start
        valid_cols = valid_col_end - col_start
        z_init_local[:valid_rows, :valid_cols, :] = (
            s2[row_start:valid_row_end, None, :]
            + s1[None, col_start:valid_col_end, :]
        )
        return z_init_local.contiguous()

    def forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_coords: torch.Tensor,
        z_trunk_spec: Optional[FoldCPPairShardSpec] = None,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        compute_plddt: bool = True,
        compute_pae: bool = True,
        compute_pde: bool = True,
        compute_resolved: bool = True,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Args:
            input_feature_dict: Dictionary containing input features.
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_sample, N_atoms, 3]
            triangle_multiplicative: Triangle multiplicative implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": Cuequivariance implementation
            triangle_attention: Triangle attention implementation type.
                - "torch" (default): PyTorch native implementation
                - "cuequivariance": cuEquivariance implementation
            inplace_safe (bool, optional): Whether to use inplace operations. Defaults to False.
            chunk_size (Optional[int], optional): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - plddt_preds: Predicted pLDDT scores [..., N_sample, N_atom, plddt_bins].
                - pae_preds: Predicted PAE scores [..., N_sample, N_token, N_token, pae_bins].
                - pde_preds: Predicted PDE scores [..., N_sample, N_token, N_token, pde_bins].
                - resolved_preds: Predicted resolved scores [..., N_sample, N_atom, 2].
        """

        s_inputs = s_inputs.detach()
        s_trunk = s_trunk.detach()
        z_trunk = z_trunk.detach()

        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))

        x_rep_atom_mask = self._select_distogram_rep_atom_mask(
            input_feature_dict=input_feature_dict,
            n_token=s_trunk.shape[-2],
        )
        x_pred_rep_coords = x_pred_coords[..., x_rep_atom_mask, :]
        N_sample = x_pred_rep_coords.size(-3)

        foldcp_mesh = self._maybe_create_foldcp_mesh()
        use_foldcp_confidence = (
            foldcp_mesh is not None
            and self.pairformer_stack.blocks
            and getattr(self.pairformer_stack.blocks[0], "c_s", 0) > 0
        )
        z_pair_spec = None
        if use_foldcp_confidence:
            if z_trunk_spec is None:
                z_trunk_local, z_pair_spec = shard_pair_tensor(
                    z_trunk,
                    foldcp_mesh,
                    pair_dims=(-3, -2),
                )
            else:
                z_trunk_local = z_trunk
                z_pair_spec = z_trunk_spec
            z_init_local = self._build_confidence_z_init_local(
                s_inputs=s_inputs,
                z_pair_spec=z_pair_spec,
                reference=z_trunk_local,
            )
            z_trunk = z_trunk_local + z_init_local
            del z_trunk_local, z_init_local
            torch.cuda.empty_cache()
        else:
            z_init = (
                self.linear_no_bias_s1(s_inputs)[..., None, :, :]
                + self.linear_no_bias_s2(s_inputs)[..., None, :]
            )
            z_trunk = z_init + z_trunk
            del z_init
            torch.cuda.empty_cache()

        plddt_preds = [] if compute_plddt else None
        pae_preds = [] if compute_pae else None
        pde_preds = [] if compute_pde else None
        resolved_preds = [] if compute_resolved else None
        single_sample = N_sample == 1
        foldcp_non_output_rank = self._foldcp_is_non_output_rank()
        for i in range(N_sample):
            if use_foldcp_confidence:
                assert foldcp_mesh is not None
                assert z_pair_spec is not None
                plddt_pred, pae_pred, pde_pred, resolved_pred = (
                    self.memory_efficient_forward_foldcp_local(
                        input_feature_dict=input_feature_dict,
                        s_trunk=(
                            s_trunk
                            if single_sample
                            else (s_trunk.clone() if inplace_safe else s_trunk)
                        ),
                        z_pair_local=z_trunk if single_sample else z_trunk.clone(),
                        z_pair_spec=z_pair_spec,
                        foldcp_mesh=foldcp_mesh,
                        pair_mask=pair_mask,
                        x_pred_rep_coords=x_pred_rep_coords[..., i, :, :],
                        compute_plddt=compute_plddt,
                        compute_pae=compute_pae,
                        compute_pde=compute_pde,
                        compute_resolved=compute_resolved,
                    )
                )
            else:
                plddt_pred, pae_pred, pde_pred, resolved_pred = (
                    self.memory_efficient_forward(
                        input_feature_dict=input_feature_dict,
                        s_trunk=s_trunk.clone() if inplace_safe else s_trunk,
                        z_pair=z_trunk.clone() if inplace_safe else z_trunk,
                        pair_mask=pair_mask,
                        x_pred_rep_coords=x_pred_rep_coords[..., i, :, :],
                        triangle_multiplicative=triangle_multiplicative,
                        triangle_attention=triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                        compute_plddt=compute_plddt,
                        compute_pae=compute_pae,
                        compute_pde=compute_pde,
                        compute_resolved=compute_resolved,
                    )
                )
            if foldcp_non_output_rank and all(
                pred is None
                for pred in (plddt_pred, pae_pred, pde_pred, resolved_pred)
            ):
                continue
            if plddt_preds is not None:
                plddt_preds.append(plddt_pred)
            if pae_preds is not None:
                pae_preds.append(pae_pred)
            if pde_preds is not None:
                pde_preds.append(pde_pred)
            if resolved_preds is not None:
                resolved_preds.append(resolved_pred)
        if foldcp_non_output_rank:
            return None, None, None, None
        plddt_preds = (
            torch.stack(plddt_preds, dim=-3) if plddt_preds is not None else None
        )  # [..., N_sample, N_atom, plddt_bins]
        # Pae_preds/pde_preds single tensor will occupy 11.6G[BF16]/23.2G[FP32]
        pae_preds = (
            torch.stack(pae_preds, dim=-4) if pae_preds is not None else None
        )  # [..., N_sample, N_token, N_token, pae_bins]
        pde_preds = (
            torch.stack(pde_preds, dim=-4) if pde_preds is not None else None
        )  # [..., N_sample, N_token, N_token, pde_bins]
        resolved_preds = (
            torch.stack(resolved_preds, dim=-3) if resolved_preds is not None else None
        )  # [..., N_sample, N_atom, 2]
        return (
            plddt_preds,
            pae_preds,
            pde_preds,
            resolved_preds,
        )

    def memory_efficient_forward_foldcp_local(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_trunk: torch.Tensor,
        z_pair_local: torch.Tensor,
        z_pair_spec: FoldCPPairShardSpec,
        foldcp_mesh: FoldCPProcessMesh,
        pair_mask: Optional[torch.Tensor],
        x_pred_rep_coords: torch.Tensor,
        compute_plddt: bool = True,
        compute_pae: bool = True,
        compute_pde: bool = True,
        compute_resolved: bool = True,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
        z_pair_local = add_confidence_distance_embedding_local(
            z_pair_local=z_pair_local,
            z_pair_spec=z_pair_spec,
            x_pred_rep_coords=x_pred_rep_coords,
            lower_bins=self.lower_bins,
            upper_bins=self.upper_bins,
            linear_onehot=self.linear_no_bias_d,
            linear_distance=self.linear_no_bias_d_wo_onehot,
        )
        s_single, z_pair_local, z_pair_spec = (
            distributed_pairformer_stack_single_bridge_update(
                self.pairformer_stack,
                s_trunk,
                z_pair_local,
                foldcp_mesh,
                pair_mask,
                extra_attn_bias=input_feature_dict.get(
                    "structural_pair_attn_bias", None
                ),
                extra_attn_bias_is_local=bool(
                    input_feature_dict.get("_foldcp_pair_features_are_local", False)
                ),
                return_local_pair=True,
                z_spec=z_pair_spec,
            )
        )
        if compute_pae or compute_pde:
            z_pair_local = z_pair_local.to(torch.float32)
            torch.cuda.empty_cache()
        if compute_plddt or compute_resolved:
            s_single = s_single.to(torch.float32)

        atom_to_token_idx = cast(
            torch.Tensor, input_feature_dict["atom_to_token_idx"]
        )
        atom_to_tokatom_idx = cast(
            torch.Tensor, input_feature_dict["atom_to_tokatom_idx"]
        )

        with torch.amp.autocast("cuda", enabled=False):
            pae_pred, pde_pred = distributed_confidence_pair_logits(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=foldcp_mesh,
                pae_ln=self.pae_ln,
                pae_linear=self.linear_no_bias_pae,
                pde_ln=self.pde_ln,
                pde_linear=self.linear_no_bias_pde,
                compute_pae=compute_pae,
                compute_pde=compute_pde,
                gather_to_rank0_only=True,
            )
            is_output_rank = dist.get_rank(foldcp_mesh.group_2d) == 0
            if not is_output_rank:
                return None, None, None, None
            if compute_plddt or compute_resolved:
                a = broadcast_token_to_atom(
                    x_token=s_single, atom_to_token_idx=atom_to_token_idx
                )
                plddt_pred = (
                    torch.einsum(
                        "...nc,ncb->...nb",
                        self.plddt_ln(a),
                        self.plddt_weight[atom_to_tokatom_idx],
                    )
                    if compute_plddt
                    else None
                )
                resolved_pred = (
                    torch.einsum(
                        "...nc,ncb->...nb",
                        self.resolved_ln(a),
                        self.resolved_weight[atom_to_tokatom_idx],
                    )
                    if compute_resolved
                    else None
                )
            else:
                plddt_pred = None
                resolved_pred = None
        del z_pair_local
        torch.cuda.empty_cache()
        return plddt_pred, pae_pred, pde_pred, resolved_pred

    def memory_efficient_forward(
        self,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_trunk: torch.Tensor,
        z_pair: torch.Tensor,
        pair_mask: torch.Tensor,
        x_pred_rep_coords: torch.Tensor,
        triangle_multiplicative: str = "torch",
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        compute_plddt: bool = True,
        compute_pae: bool = True,
        compute_pde: bool = True,
        compute_resolved: bool = True,
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Args:
            ...
            x_pred_coords (torch.Tensor): predicted coordinates
                [..., N_atoms, 3] # Note: N_sample = 1 for avoiding CUDA OOM
        """
        foldcp_mesh = self._maybe_create_foldcp_mesh()
        use_foldcp_confidence = (
            foldcp_mesh is not None
            and self.pairformer_stack.blocks
            and getattr(self.pairformer_stack.blocks[0], "c_s", 0) > 0
        )
        if use_foldcp_confidence:
            z_pair_local, z_pair_spec = shard_pair_tensor(
                z_pair,
                foldcp_mesh,
                pair_dims=(-3, -2),
            )
            del z_pair
            torch.cuda.empty_cache()
            return self.memory_efficient_forward_foldcp_local(
                input_feature_dict=input_feature_dict,
                s_trunk=s_trunk,
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                foldcp_mesh=foldcp_mesh,
                pair_mask=pair_mask,
                x_pred_rep_coords=x_pred_rep_coords,
                compute_plddt=compute_plddt,
                compute_pae=compute_pae,
                compute_pde=compute_pde,
                compute_resolved=compute_resolved,
            )

        # Embed pair distances of representative atoms:
        with torch.amp.autocast("cuda", enabled=False):
            x_pred_rep_coords = x_pred_rep_coords.to(torch.float32)
            distance_pred = torch.cdist(
                x_pred_rep_coords, x_pred_rep_coords
            )  # [..., N_tokens, N_tokens]
        if inplace_safe:
            z_pair += self.linear_no_bias_d(
                one_hot(
                    x=distance_pred,
                    lower_bins=self.lower_bins,
                    upper_bins=self.upper_bins,
                ).to(dtype=self.linear_no_bias_d.weight.dtype)
            )  # [..., N_tokens, N_tokens, c_z]
            z_pair += self.linear_no_bias_d_wo_onehot(
                distance_pred.unsqueeze(dim=-1).to(
                    dtype=self.linear_no_bias_d_wo_onehot.weight.dtype
                ),
            )  # [..., N_tokens, N_tokens, c_z]
        else:
            z_pair = z_pair + self.linear_no_bias_d(
                one_hot(
                    x=distance_pred,
                    lower_bins=self.lower_bins,
                    upper_bins=self.upper_bins,
                ).to(dtype=self.linear_no_bias_d.weight.dtype)
            )  # [..., N_tokens, N_tokens, c_z]

            z_pair = z_pair + self.linear_no_bias_d_wo_onehot(
                distance_pred.unsqueeze(dim=-1).to(
                    dtype=self.linear_no_bias_d_wo_onehot.weight.dtype
                )
            )  # [..., N_tokens, N_tokens, c_z]

        # Line 4
        s_single, z_pair = self.pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            triangle_multiplicative=triangle_multiplicative,
            triangle_attention=triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias", None),
        )
        # Upcast only the representations needed by enabled confidence heads.
        if compute_pae or compute_pde:
            z_pair = z_pair.to(torch.float32)
        if compute_plddt or compute_resolved:
            s_single = s_single.to(torch.float32)
        atom_to_token_idx = cast(
            torch.Tensor, input_feature_dict["atom_to_token_idx"]
        )  # in range [0, N_token-1] shape: [N_atom]
        atom_to_tokatom_idx = cast(
            torch.Tensor, input_feature_dict["atom_to_tokatom_idx"]
        )  # in range [0, max_atoms_per_token-1] shape: [N_atom] # influenced by crop

        with torch.amp.autocast("cuda", enabled=False):
            pae_pred = (
                self.linear_no_bias_pae(self.pae_ln(z_pair)) if compute_pae else None
            )
            pde_pred = (
                self.linear_no_bias_pde(self.pde_ln(z_pair + z_pair.transpose(-2, -3)))
                if compute_pde
                else None
            )
            if compute_plddt or compute_resolved:
                # Broadcast s_single: [N_tokens, c_s] -> [N_atoms, c_s]
                a = broadcast_token_to_atom(
                    x_token=s_single, atom_to_token_idx=atom_to_token_idx
                )
                plddt_pred = (
                    torch.einsum(
                        "...nc,ncb->...nb",
                        self.plddt_ln(a),
                        self.plddt_weight[atom_to_tokatom_idx],
                    )
                    if compute_plddt
                    else None
                )
                resolved_pred = (
                    torch.einsum(
                        "...nc,ncb->...nb",
                        self.resolved_ln(a),
                        self.resolved_weight[atom_to_tokatom_idx],
                    )
                    if compute_resolved
                    else None
                )
            else:
                plddt_pred = None
                resolved_pred = None
        if z_pair.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return plddt_pred, pae_pred, pde_pred, resolved_pred
