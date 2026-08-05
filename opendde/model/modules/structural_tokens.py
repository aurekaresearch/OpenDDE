# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from typing import Any, Optional

import torch
import torch.nn as nn

from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.distributed.foldcp.atom_window import (
    _gather_pair_rows_one_by_p,
    gather_pair_embedding_in_dense_trunk_from_foldcp_local,
)
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    make_pair_shard_spec,
)
from opendde.model.modules.primitives import LinearNoBias
from opendde.model.triangular.layers import LayerNorm


class StructuralTokenExpander(nn.Module):
    """
    Expand residue-level trunk activations into structural token activations.

    The residue trunk remains unchanged. This module gathers each structural
    token's parent residue representation, adds role-specific single/pair
    conditioning, and injects explicit pair biases for same-residue structure.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_s_inputs: int,
        n_roles: int = max(STRUCTURAL_TOKEN_ROLES.values()) + 1,
        init_mode: str = "zero",
        role_init_std: float = 0.02,
        pair_feature_init_std: float = 0.02,
        attention_bias_init: float = 0.1,
        pair_projection_mode: str = "factorized",
        pair_chunk_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        required_n_roles = max(STRUCTURAL_TOKEN_ROLES.values()) + 1
        if n_roles < required_n_roles:
            raise ValueError(
                f"n_roles={n_roles} is too small for structural token roles; "
                f"need at least {required_n_roles}"
            )
        self.c_s = c_s
        self.c_z = c_z
        self.c_s_inputs = c_s_inputs
        self.n_roles = n_roles
        self.init_mode = init_mode
        if pair_chunk_size is not None and int(pair_chunk_size) <= 0:
            raise ValueError(
                "StructuralTokenExpander pair_chunk_size must be positive or None; "
                f"got {pair_chunk_size}"
            )
        self.pair_chunk_size = None if pair_chunk_size is None else int(pair_chunk_size)
        if pair_projection_mode not in {"full", "factorized", "none"}:
            raise ValueError(
                "StructuralTokenExpander pair_projection_mode must be one of "
                f"'full', 'factorized', or 'none'; got {pair_projection_mode!r}"
            )
        self.pair_projection_mode = pair_projection_mode

        self.single_split_mlp = nn.Sequential(
            LayerNorm(c_s),
            LinearNoBias(c_s, 2 * c_s, initializer="relu"),
            nn.SiLU(),
            LinearNoBias(2 * c_s, c_s, initializer="zeros"),
        )
        self.single_input_role_embedding = nn.Embedding(n_roles, c_s_inputs)
        self.single_role_embedding = nn.Embedding(n_roles, c_s)

        if pair_projection_mode == "full":
            self.pair_block_proj = nn.ModuleList(
                [
                    LinearNoBias(c_z, c_z, initializer="zeros")
                    for _ in range(n_roles * n_roles)
                ]
            )
        elif pair_projection_mode == "factorized":
            self.shared_pair_proj = LinearNoBias(c_z, c_z, initializer="zeros")
            self.role_pair_gate = nn.Embedding(8, c_z)
            self.role_pair_delta_bias = nn.Embedding(8, c_z)
        self.same_parent_embedding = nn.Embedding(2, c_z)
        self.same_residue_twin_embedding = nn.Embedding(2, c_z)
        self.prev_bb_chain_embedding = nn.Embedding(2, c_z)
        self.next_bb_chain_embedding = nn.Embedding(2, c_z)
        self.role_pair_type_embedding = nn.Embedding(8, c_z)
        for embedding in [
            self.same_parent_embedding,
            self.same_residue_twin_embedding,
            self.prev_bb_chain_embedding,
            self.next_bb_chain_embedding,
            self.role_pair_type_embedding,
        ]:
            nn.init.zeros_(embedding.weight)
        self.attn_bias_same_parent = nn.Parameter(torch.zeros(()))
        self.attn_bias_same_residue_twin = nn.Parameter(torch.zeros(()))
        self.attn_bias_prev_bb_chain = nn.Parameter(torch.zeros(()))
        self.attn_bias_next_bb_chain = nn.Parameter(torch.zeros(()))
        self.attn_bias_role_pair_type = nn.Parameter(torch.zeros(8))
        self._init_role_conditioning(
            init_mode=init_mode,
            role_init_std=role_init_std,
            pair_feature_init_std=pair_feature_init_std,
            attention_bias_init=attention_bias_init,
        )

        self.backbone_role_ids = (
            STRUCTURAL_TOKEN_ROLES["protein_bb"],
            STRUCTURAL_TOKEN_ROLES["dna_bb"],
            STRUCTURAL_TOKEN_ROLES["rna_bb"],
        )
        self.sidechain_role_id = STRUCTURAL_TOKEN_ROLES["protein_sc"]
        self.base_role_ids = (
            STRUCTURAL_TOKEN_ROLES["dna_base"],
            STRUCTURAL_TOKEN_ROLES["rna_base"],
        )

    @staticmethod
    def _trunc_normal_small_(weight: torch.Tensor, std: float) -> None:
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)

    def _init_boolean_embedding(self, embedding: nn.Embedding, std: float) -> None:
        nn.init.zeros_(embedding.weight)
        self._trunc_normal_small_(embedding.weight[1:], std)

    def _init_role_conditioning(
        self,
        init_mode: str,
        role_init_std: float,
        pair_feature_init_std: float,
        attention_bias_init: float,
    ) -> None:
        if init_mode not in {"zero", "scratch"}:
            raise ValueError(
                "StructuralTokenExpander init_mode must be 'zero' or 'scratch'; "
                f"got {init_mode!r}"
            )

        nn.init.zeros_(self.single_input_role_embedding.weight)
        nn.init.zeros_(self.single_role_embedding.weight)
        if self.pair_projection_mode == "factorized":
            nn.init.ones_(self.role_pair_gate.weight)
            nn.init.zeros_(self.role_pair_delta_bias.weight)
        for embedding in [
            self.same_parent_embedding,
            self.same_residue_twin_embedding,
            self.prev_bb_chain_embedding,
            self.next_bb_chain_embedding,
            self.role_pair_type_embedding,
        ]:
            nn.init.zeros_(embedding.weight)

        if init_mode == "zero":
            return

        self._trunc_normal_small_(
            self.single_input_role_embedding.weight, role_init_std
        )
        self._trunc_normal_small_(self.single_role_embedding.weight, role_init_std)
        for embedding in [
            self.same_parent_embedding,
            self.same_residue_twin_embedding,
            self.prev_bb_chain_embedding,
            self.next_bb_chain_embedding,
        ]:
            self._init_boolean_embedding(embedding, pair_feature_init_std)
        self._trunc_normal_small_(
            self.role_pair_type_embedding.weight, pair_feature_init_std
        )
        if self.pair_projection_mode == "factorized":
            with torch.no_grad():
                nn.init.trunc_normal_(
                    self.role_pair_gate.weight,
                    mean=1.0,
                    std=pair_feature_init_std,
                    a=1.0 - 2.0 * pair_feature_init_std,
                    b=1.0 + 2.0 * pair_feature_init_std,
                )
            self._trunc_normal_small_(
                self.role_pair_delta_bias.weight, pair_feature_init_std
            )
        with torch.no_grad():
            self.attn_bias_same_parent.fill_(attention_bias_init)
            self.attn_bias_same_residue_twin.fill_(attention_bias_init)
            self.attn_bias_prev_bb_chain.fill_(attention_bias_init)
            self.attn_bias_next_bb_chain.fill_(attention_bias_init)

    @staticmethod
    def _gather_parent_single(x: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        return x.index_select(dim=-2, index=parent)

    @staticmethod
    def _gather_parent_pair(z: torch.Tensor, parent: torch.Tensor) -> torch.Tensor:
        return z.index_select(dim=-3, index=parent).index_select(dim=-2, index=parent)

    @staticmethod
    def _gather_parent_pair_rows(
        z: torch.Tensor, parent: torch.Tensor, row_index: torch.Tensor
    ) -> torch.Tensor:
        row_parent = parent.index_select(dim=0, index=row_index)
        return z.index_select(dim=-3, index=row_parent).index_select(
            dim=-2, index=parent
        )

    @staticmethod
    def _gather_parent_pair_tile(
        z: torch.Tensor,
        parent: torch.Tensor,
        row_index: torch.Tensor,
        col_index: torch.Tensor,
    ) -> torch.Tensor:
        row_parent = parent.index_select(dim=0, index=row_index)
        col_parent = parent.index_select(dim=0, index=col_index)
        return z.index_select(dim=-3, index=row_parent).index_select(
            dim=-2, index=col_parent
        )

    @staticmethod
    def _gather_parent_pair_tile_from_foldcp_local(
        z: torch.Tensor,
        z_spec: FoldCPPairShardSpec,
        parent: torch.Tensor,
        row_index: torch.Tensor,
        col_index: torch.Tensor,
        mesh: FoldCPProcessMesh,
    ) -> torch.Tensor:
        row_parent = parent.index_select(dim=0, index=row_index)
        col_parent = parent.index_select(dim=0, index=col_index)
        if mesh.layout.shape[0] == 1:
            return _gather_pair_rows_one_by_p(
                z_local=z,
                z_spec=z_spec,
                idx_q=row_parent.unsqueeze(0),
                idx_k=col_parent.unsqueeze(0),
                mesh=mesh,
            ).squeeze(0)
        return gather_pair_embedding_in_dense_trunk_from_foldcp_local(
            z_local=z,
            z_spec=z_spec,
            idx_q=row_parent.unsqueeze(0),
            idx_k=col_parent.unsqueeze(0),
            mesh=mesh,
        ).squeeze(0)

    def _use_foldcp_full_projection_source_order(self, z: torch.Tensor) -> bool:
        return (
            self.pair_projection_mode == "full"
            and z.is_cuda
            and torch.are_deterministic_algorithms_enabled()
        )

    def _pair_project_by_role_full(
        self, z: torch.Tensor, role: torch.Tensor
    ) -> torch.Tensor:
        n_struct = role.shape[-1]
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, n_struct * n_struct, self.c_z)
        flat_delta = torch.zeros_like(flat_z)
        role_i = role[:, None].expand(n_struct, n_struct).reshape(-1)
        role_j = role[None, :].expand(n_struct, n_struct).reshape(-1)
        dummy_input = flat_z[..., :1, :]
        dummy_use = flat_z.new_zeros(())

        for role_i_value in range(self.n_roles):
            for role_j_value in range(self.n_roles):
                flat_mask = (role_i == role_i_value) & (role_j == role_j_value)
                proj_idx = role_i_value * self.n_roles + role_j_value
                projection = self.pair_block_proj[proj_idx]
                if torch.any(flat_mask):
                    flat_delta[..., flat_mask, :] = projection(
                        flat_z[..., flat_mask, :]
                    )
                else:
                    # Keep every role-pair projection in the autograd graph so
                    # DDP static_graph does not see role-dependent unused params.
                    dummy_use = dummy_use + projection(dummy_input).sum() * 0.0
        return (
            flat_delta.reshape(*batch_shape, n_struct, n_struct, self.c_z) + dummy_use
        )

    def _pair_project_by_role_full_chunk(
        self,
        z: torch.Tensor,
        role: torch.Tensor,
        row_index: torch.Tensor,
    ) -> torch.Tensor:
        n_struct = role.shape[-1]
        chunk_len = row_index.numel()
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, chunk_len * n_struct, self.c_z)
        flat_delta = torch.zeros_like(flat_z)
        row_role = role.index_select(dim=0, index=row_index)
        role_i = row_role[:, None].expand(chunk_len, n_struct).reshape(-1)
        role_j = role[None, :].expand(chunk_len, n_struct).reshape(-1)
        dummy_input = flat_z[..., :1, :]
        dummy_use = flat_z.new_zeros(())

        for role_i_value in range(self.n_roles):
            for role_j_value in range(self.n_roles):
                flat_mask = (role_i == role_i_value) & (role_j == role_j_value)
                proj_idx = role_i_value * self.n_roles + role_j_value
                projection = self.pair_block_proj[proj_idx]
                if torch.any(flat_mask):
                    flat_delta[..., flat_mask, :] = projection(
                        flat_z[..., flat_mask, :]
                    )
                else:
                    dummy_use = dummy_use + projection(dummy_input).sum() * 0.0
        return (
            flat_delta.reshape(*batch_shape, chunk_len, n_struct, self.c_z) + dummy_use
        )

    def _pair_project_by_role_full_tile(
        self,
        z: torch.Tensor,
        role: torch.Tensor,
        row_index: torch.Tensor,
        col_index: torch.Tensor,
    ) -> torch.Tensor:
        row_len = row_index.numel()
        col_len = col_index.numel()
        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, row_len * col_len, self.c_z)
        flat_delta = torch.zeros_like(flat_z)
        row_role = role.index_select(dim=0, index=row_index)
        col_role = role.index_select(dim=0, index=col_index)
        role_i = row_role[:, None].expand(row_len, col_len).reshape(-1)
        role_j = col_role[None, :].expand(row_len, col_len).reshape(-1)
        dummy_input = flat_z[..., :1, :]
        dummy_use = flat_z.new_zeros(())

        for role_i_value in range(self.n_roles):
            for role_j_value in range(self.n_roles):
                flat_mask = (role_i == role_i_value) & (role_j == role_j_value)
                proj_idx = role_i_value * self.n_roles + role_j_value
                projection = self.pair_block_proj[proj_idx]
                if torch.any(flat_mask):
                    flat_delta[..., flat_mask, :] = projection(
                        flat_z[..., flat_mask, :]
                    )
                else:
                    dummy_use = dummy_use + projection(dummy_input).sum() * 0.0
        return flat_delta.reshape(*batch_shape, row_len, col_len, self.c_z) + dummy_use

    def _pair_project_by_role(
        self,
        z: torch.Tensor,
        role: torch.Tensor,
        pair_features: dict[str, torch.Tensor],
        row_index: Optional[torch.Tensor] = None,
        col_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.pair_projection_mode == "none":
            return None
        if self.pair_projection_mode == "full":
            if row_index is not None and col_index is not None:
                return self._pair_project_by_role_full_tile(
                    z,
                    role,
                    row_index,
                    col_index,
                )
            if row_index is not None:
                return self._pair_project_by_role_full_chunk(z, role, row_index)
            return self._pair_project_by_role_full(z, role)

        role_pair_type = pair_features["role_pair_type"]
        base_delta = self.shared_pair_proj(z)
        gate = self.role_pair_gate(role_pair_type).to(dtype=z.dtype)
        bias = self.role_pair_delta_bias(role_pair_type).to(dtype=z.dtype)
        return base_delta * gate + bias

    def _add_pair_project_by_role_inplace(
        self,
        z: torch.Tensor,
        role: torch.Tensor,
        pair_features: dict[str, torch.Tensor],
        row_index: Optional[torch.Tensor] = None,
        col_index: Optional[torch.Tensor] = None,
        flat_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if self.pair_projection_mode == "none":
            return z
        if self.pair_projection_mode == "full":
            delta = self._pair_project_by_role(
                z=z,
                role=role,
                pair_features=pair_features,
                row_index=row_index,
                col_index=col_index,
            )
            if delta is not None:
                z.add_(delta)
            return z

        role_pair_type = pair_features["role_pair_type"]
        flat_count = role_pair_type.numel()
        if flat_count == 0:
            return z
        if flat_chunk_size is None:
            flat_chunk_size = flat_count
        flat_chunk_size = int(flat_chunk_size)
        if flat_chunk_size <= 0:
            raise ValueError(
                f"flat_chunk_size must be positive or None, got {flat_chunk_size}"
            )

        batch_shape = z.shape[:-3]
        flat_z = z.reshape(*batch_shape, flat_count, self.c_z)
        flat_role_pair_type = role_pair_type.reshape(flat_count)
        for flat_start in range(0, flat_count, flat_chunk_size):
            flat_end = min(flat_start + flat_chunk_size, flat_count)
            z_chunk = flat_z[..., flat_start:flat_end, :]
            role_chunk = flat_role_pair_type[flat_start:flat_end]
            base_delta = self.shared_pair_proj(z_chunk)
            gate = self.role_pair_gate(role_chunk).to(dtype=z.dtype)
            bias = self.role_pair_delta_bias(role_chunk).to(dtype=z.dtype)
            base_delta.mul_(gate).add_(bias)
            z_chunk.add_(base_delta)
        return z

    def _build_structural_pair_context(
        self,
        input_feature_dict: dict[str, Any],
        role: torch.Tensor,
        parent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        n_struct = role.shape[-1]
        residue_index = input_feature_dict["residue_index"].index_select(
            dim=-1, index=parent
        )
        asym_id = input_feature_dict["asym_id"].index_select(dim=-1, index=parent)
        polymer_type = input_feature_dict.get("structural_polymer_type")
        if polymer_type is None:
            polymer_type = torch.zeros_like(role)
        polymer_type = polymer_type.long()
        is_backbone = (
            (role == self.backbone_role_ids[0])
            | (role == self.backbone_role_ids[1])
            | (role == self.backbone_role_ids[2])
        )
        is_sidechain = role == self.sidechain_role_id
        is_base = (role == self.base_role_ids[0]) | (role == self.base_role_ids[1])

        prev_parent = input_feature_dict.get("prev_parent_residue_idx")
        next_parent = input_feature_dict.get("next_parent_residue_idx")
        if prev_parent is None:
            prev_parent = parent.new_full((n_struct,), -1)
        if next_parent is None:
            next_parent = parent.new_full((n_struct,), -1)

        return {
            "parent": parent,
            "role": role,
            "residue_index": residue_index,
            "asym_id": asym_id,
            "polymer_type": polymer_type,
            "is_backbone": is_backbone,
            "is_sidechain": is_sidechain,
            "is_base": is_base,
            "prev_parent": prev_parent,
            "next_parent": next_parent,
        }

    def _build_structural_pair_features_for_rows(
        self,
        context: dict[str, torch.Tensor],
        row_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        parent = context["parent"]
        residue_index = context["residue_index"]
        asym_id = context["asym_id"]
        polymer_type = context["polymer_type"]
        is_backbone = context["is_backbone"]
        is_sidechain = context["is_sidechain"]
        is_base = context["is_base"]
        prev_parent = context["prev_parent"]
        next_parent = context["next_parent"]
        n_struct = parent.shape[-1]

        row_parent = parent.index_select(dim=0, index=row_index)
        row_asym_id = asym_id.index_select(dim=0, index=row_index)
        row_polymer_type = polymer_type.index_select(dim=0, index=row_index)
        row_is_backbone = is_backbone.index_select(dim=0, index=row_index)
        row_is_sidechain = is_sidechain.index_select(dim=0, index=row_index)
        row_is_base = is_base.index_select(dim=0, index=row_index)
        row_prev_parent = prev_parent.index_select(dim=0, index=row_index)
        row_next_parent = next_parent.index_select(dim=0, index=row_index)

        same_parent_residue = row_parent[:, None] == parent[None, :]
        same_chain = row_asym_id[:, None] == asym_id[None, :]
        same_polymer_type = (row_polymer_type[:, None] == polymer_type[None, :]) & (
            row_polymer_type[:, None] > 0
        )
        same_residue_twin = same_parent_residue & (
            (row_is_backbone[:, None] & (is_sidechain[None, :] | is_base[None, :]))
            | (
                is_backbone[None, :]
                & (row_is_sidechain[:, None] | row_is_base[:, None])
            )
        )
        prev_bb_chain = (
            row_is_backbone[:, None]
            & is_backbone[None, :]
            & same_chain
            & (row_prev_parent[:, None] == parent[None, :])
        )
        next_bb_chain = (
            row_is_backbone[:, None]
            & is_backbone[None, :]
            & same_chain
            & (row_next_parent[:, None] == parent[None, :])
        )

        chunk_len = row_index.numel()
        role_pair_type = torch.full(
            (chunk_len, n_struct), 7, dtype=torch.long, device=parent.device
        )
        role_pair_type[row_is_backbone[:, None] & is_backbone[None, :]] = 0
        role_pair_type[row_is_backbone[:, None] & is_sidechain[None, :]] = 1
        role_pair_type[row_is_sidechain[:, None] & is_backbone[None, :]] = 2
        role_pair_type[row_is_sidechain[:, None] & is_sidechain[None, :]] = 3
        role_pair_type[row_is_backbone[:, None] & is_base[None, :]] = 4
        role_pair_type[row_is_base[:, None] & is_backbone[None, :]] = 5
        role_pair_type[row_is_base[:, None] & is_base[None, :]] = 6

        return {
            "same_parent_residue": same_parent_residue,
            "same_residue_twin": same_residue_twin,
            "prev_bb_chain": prev_bb_chain,
            "next_bb_chain": next_bb_chain,
            "role_pair_type": role_pair_type,
            "same_chain": same_chain,
            "same_polymer_type": same_polymer_type,
            "residue_index": residue_index,
        }

    def _build_structural_pair_features_for_tile(
        self,
        context: dict[str, torch.Tensor],
        row_index: torch.Tensor,
        col_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        parent = context["parent"]
        residue_index = context["residue_index"]
        asym_id = context["asym_id"]
        polymer_type = context["polymer_type"]
        is_backbone = context["is_backbone"]
        is_sidechain = context["is_sidechain"]
        is_base = context["is_base"]
        prev_parent = context["prev_parent"]
        next_parent = context["next_parent"]

        row_parent = parent.index_select(dim=0, index=row_index)
        col_parent = parent.index_select(dim=0, index=col_index)
        row_asym_id = asym_id.index_select(dim=0, index=row_index)
        col_asym_id = asym_id.index_select(dim=0, index=col_index)
        row_polymer_type = polymer_type.index_select(dim=0, index=row_index)
        col_polymer_type = polymer_type.index_select(dim=0, index=col_index)
        row_is_backbone = is_backbone.index_select(dim=0, index=row_index)
        col_is_backbone = is_backbone.index_select(dim=0, index=col_index)
        row_is_sidechain = is_sidechain.index_select(dim=0, index=row_index)
        col_is_sidechain = is_sidechain.index_select(dim=0, index=col_index)
        row_is_base = is_base.index_select(dim=0, index=row_index)
        col_is_base = is_base.index_select(dim=0, index=col_index)
        row_prev_parent = prev_parent.index_select(dim=0, index=row_index)
        row_next_parent = next_parent.index_select(dim=0, index=row_index)

        same_parent_residue = row_parent[:, None] == col_parent[None, :]
        same_chain = row_asym_id[:, None] == col_asym_id[None, :]
        same_polymer_type = (row_polymer_type[:, None] == col_polymer_type[None, :]) & (
            row_polymer_type[:, None] > 0
        )
        same_residue_twin = same_parent_residue & (
            (
                row_is_backbone[:, None]
                & (col_is_sidechain[None, :] | col_is_base[None, :])
            )
            | (
                col_is_backbone[None, :]
                & (row_is_sidechain[:, None] | row_is_base[:, None])
            )
        )
        prev_bb_chain = (
            row_is_backbone[:, None]
            & col_is_backbone[None, :]
            & same_chain
            & (row_prev_parent[:, None] == col_parent[None, :])
        )
        next_bb_chain = (
            row_is_backbone[:, None]
            & col_is_backbone[None, :]
            & same_chain
            & (row_next_parent[:, None] == col_parent[None, :])
        )

        row_len = row_index.numel()
        col_len = col_index.numel()
        role_pair_type = torch.full(
            (row_len, col_len), 7, dtype=torch.long, device=parent.device
        )
        role_pair_type[row_is_backbone[:, None] & col_is_backbone[None, :]] = 0
        role_pair_type[row_is_backbone[:, None] & col_is_sidechain[None, :]] = 1
        role_pair_type[row_is_sidechain[:, None] & col_is_backbone[None, :]] = 2
        role_pair_type[row_is_sidechain[:, None] & col_is_sidechain[None, :]] = 3
        role_pair_type[row_is_backbone[:, None] & col_is_base[None, :]] = 4
        role_pair_type[row_is_base[:, None] & col_is_backbone[None, :]] = 5
        role_pair_type[row_is_base[:, None] & col_is_base[None, :]] = 6

        return {
            "same_parent_residue": same_parent_residue,
            "same_residue_twin": same_residue_twin,
            "prev_bb_chain": prev_bb_chain,
            "next_bb_chain": next_bb_chain,
            "role_pair_type": role_pair_type,
            "same_chain": same_chain,
            "same_polymer_type": same_polymer_type,
            "residue_index": residue_index,
        }

    def build_structural_pair_features(
        self,
        input_feature_dict: dict[str, Any],
        role: torch.Tensor,
        parent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context = self._build_structural_pair_context(
            input_feature_dict=input_feature_dict,
            role=role,
            parent=parent,
        )
        row_index = torch.arange(parent.shape[-1], device=parent.device)
        return self._build_structural_pair_features_for_rows(
            context=context,
            row_index=row_index,
        )

    def _make_pair_init_bias(
        self, pair_features: dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        row_count = pair_features["role_pair_type"].shape[0]
        return self._make_pair_init_bias_for_rows(
            pair_features=pair_features,
            dtype=dtype,
            row_start=0,
            row_end=row_count,
        )

    def _make_pair_init_bias_for_rows(
        self,
        pair_features: dict[str, torch.Tensor],
        dtype: torch.dtype,
        row_start: int,
        row_end: int,
    ) -> torch.Tensor:
        pair_bias = self.same_parent_embedding(
            pair_features["same_parent_residue"][row_start:row_end].long()
        ).to(dtype=dtype)
        pair_bias = pair_bias + self.same_residue_twin_embedding(
            pair_features["same_residue_twin"][row_start:row_end].long()
        ).to(dtype=dtype)
        pair_bias = pair_bias + self.prev_bb_chain_embedding(
            pair_features["prev_bb_chain"][row_start:row_end].long()
        ).to(dtype=dtype)
        pair_bias = pair_bias + self.next_bb_chain_embedding(
            pair_features["next_bb_chain"][row_start:row_end].long()
        ).to(dtype=dtype)
        return pair_bias + self.role_pair_type_embedding(
            pair_features["role_pair_type"][row_start:row_end]
        ).to(dtype=dtype)

    @staticmethod
    def _reshape_pair_term_for_target(
        term: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if term.dim() >= target.dim():
            return term
        leading_ones = (1,) * (target.dim() - term.dim())
        return term.reshape(*leading_ones, *term.shape)

    def _add_pair_init_bias_chunked_inplace(
        self,
        z: torch.Tensor,
        pair_features: dict[str, torch.Tensor],
        dtype: torch.dtype,
        row_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        row_count = pair_features["role_pair_type"].shape[0]
        if row_count == 0:
            return z
        if row_chunk_size is None or row_chunk_size >= row_count:
            z.add_(
                self._reshape_pair_term_for_target(
                    self._make_pair_init_bias(pair_features, dtype=dtype),
                    z,
                )
            )
            return z
        if row_chunk_size <= 0:
            raise ValueError(f"row_chunk_size must be positive, got {row_chunk_size}")
        for row_start in range(0, row_count, row_chunk_size):
            row_end = min(row_start + row_chunk_size, row_count)
            target = z[..., row_start:row_end, :, :]
            target.add_(
                self._reshape_pair_term_for_target(
                    self._make_pair_init_bias_for_rows(
                        pair_features=pair_features,
                        dtype=dtype,
                        row_start=row_start,
                        row_end=row_end,
                    ),
                    target,
                )
            )
        return z

    def _make_structural_pair_activations_chunked(
        self,
        input_feature_dict: dict[str, Any],
        z_res: torch.Tensor,
        role: torch.Tensor,
        parent: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n_struct = role.shape[-1]
        chunk_size = min(self.pair_chunk_size or n_struct, n_struct)
        context = self._build_structural_pair_context(
            input_feature_dict=input_feature_dict,
            role=role,
            parent=parent,
        )
        z_chunks = []
        attn_bias_chunks = []
        for start in range(0, n_struct, chunk_size):
            end = min(start + chunk_size, n_struct)
            row_index = torch.arange(start, end, device=parent.device)
            pair_features = self._build_structural_pair_features_for_rows(
                context=context,
                row_index=row_index,
            )
            z_chunk = self._gather_parent_pair_rows(
                z=z_res,
                parent=parent,
                row_index=row_index,
            )
            delta = self._pair_project_by_role(
                z=z_chunk,
                role=role,
                pair_features=pair_features,
                row_index=row_index,
            )
            if delta is not None:
                z_chunk = z_chunk + delta
            z_chunk = z_chunk + self._make_pair_init_bias(
                pair_features,
                dtype=z_chunk.dtype,
            )
            z_chunks.append(z_chunk)
            attn_bias_chunks.append(
                self._make_attention_bias(pair_features, dtype=z_chunk.dtype)
            )

        return torch.cat(z_chunks, dim=-3), {
            "structural_pair_attn_bias": torch.cat(attn_bias_chunks, dim=0)
        }

    def _make_attention_bias(
        self, pair_features: dict[str, torch.Tensor], dtype: torch.dtype
    ) -> torch.Tensor:
        role_pair_bias = self.attn_bias_role_pair_type[
            pair_features["role_pair_type"]
        ].to(dtype=dtype)
        return (
            self.attn_bias_same_parent.to(dtype=dtype)
            * pair_features["same_parent_residue"].to(dtype)
            + self.attn_bias_same_residue_twin.to(dtype=dtype)
            * pair_features["same_residue_twin"].to(dtype)
            + self.attn_bias_prev_bb_chain.to(dtype=dtype)
            * pair_features["prev_bb_chain"].to(dtype)
            + self.attn_bias_next_bb_chain.to(dtype=dtype)
            * pair_features["next_bb_chain"].to(dtype)
            + role_pair_bias
        )

    def _write_attention_bias_inplace(
        self,
        out: torch.Tensor,
        pair_features: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        feature_terms = (
            (self.attn_bias_same_parent, pair_features["same_parent_residue"]),
            (self.attn_bias_same_residue_twin, pair_features["same_residue_twin"]),
            (self.attn_bias_prev_bb_chain, pair_features["prev_bb_chain"]),
            (self.attn_bias_next_bb_chain, pair_features["next_bb_chain"]),
        )
        for term_index, (weight, feature) in enumerate(feature_terms):
            term = feature.to(dtype) * weight.to(dtype=dtype)
            if term_index == 0:
                out.copy_(self._reshape_pair_term_for_target(term, out))
            else:
                out.add_(self._reshape_pair_term_for_target(term, out))
            del term
        role_pair_bias = self.attn_bias_role_pair_type[
            pair_features["role_pair_type"]
        ].to(dtype=dtype)
        out.add_(self._reshape_pair_term_for_target(role_pair_bias, out))
        del role_pair_bias
        return out

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs_res: torch.Tensor,
        s_res: torch.Tensor,
        z_res: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        parent = input_feature_dict["parent_residue_idx"].long()
        role = input_feature_dict["subtoken_role_id"].long()
        s_inputs_struct = self._gather_parent_single(
            s_inputs_res, parent
        ) + self.single_input_role_embedding(role).to(dtype=s_inputs_res.dtype)
        s_parent = self._gather_parent_single(s_res, parent)
        s_struct = (
            s_parent
            + self.single_split_mlp(s_parent)
            + self.single_role_embedding(role).to(dtype=s_parent.dtype)
        )

        if self.pair_chunk_size is None:
            pair_features = self.build_structural_pair_features(
                input_feature_dict=input_feature_dict,
                role=role,
                parent=parent,
            )
            z_parent = self._gather_parent_pair(z_res, parent)
            delta = self._pair_project_by_role(z_parent, role, pair_features)
            z_struct = z_parent if delta is None else z_parent + delta
            z_struct = z_struct + self._make_pair_init_bias(
                pair_features, dtype=z_parent.dtype
            )
            pair_features["structural_pair_attn_bias"] = self._make_attention_bias(
                pair_features, dtype=z_parent.dtype
            )
        else:
            z_struct, pair_features = self._make_structural_pair_activations_chunked(
                input_feature_dict=input_feature_dict,
                z_res=z_res,
                role=role,
                parent=parent,
            )
        return s_inputs_struct, s_struct, z_struct, pair_features

    def forward_foldcp_local_pair(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs_res: torch.Tensor,
        s_res: torch.Tensor,
        z_res: torch.Tensor,
        mesh: FoldCPProcessMesh,
        z_res_spec: Optional[FoldCPPairShardSpec] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, FoldCPPairShardSpec],
        FoldCPPairShardSpec,
    ]:
        parent = input_feature_dict["parent_residue_idx"].long()
        role = input_feature_dict["subtoken_role_id"].long()
        s_inputs_struct = self._gather_parent_single(
            s_inputs_res, parent
        ) + self.single_input_role_embedding(role).to(dtype=s_inputs_res.dtype)
        s_parent = self._gather_parent_single(s_res, parent)
        s_struct = (
            s_parent
            + self.single_split_mlp(s_parent)
            + self.single_role_embedding(role).to(dtype=s_parent.dtype)
        )

        n_struct = role.shape[-1]
        residue_batch_shape = (
            z_res.shape[:-3] if z_res_spec is None else z_res_spec.original_shape[:-3]
        )
        z_shape = (*residue_batch_shape, n_struct, n_struct, self.c_z)
        z_spec = make_pair_shard_spec(z_shape, mesh, pair_dims=(-3, -2))
        bias_spec = make_pair_shard_spec(
            (*residue_batch_shape, n_struct, n_struct),
            mesh,
            pair_dims=(-2, -1),
        )

        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_row_end = min(row_end, n_struct)
        valid_col_end = min(col_end, n_struct)
        valid_rows = max(0, valid_row_end - row_start)
        valid_cols = max(0, valid_col_end - col_start)
        z_local = z_res.new_zeros(z_spec.local_shape)
        attn_bias_local = z_res.new_zeros(bias_spec.local_shape)

        if valid_rows > 0 and valid_cols > 0:
            col_index = torch.arange(col_start, valid_col_end, device=parent.device)
            context = self._build_structural_pair_context(
                input_feature_dict=input_feature_dict,
                role=role,
                parent=parent,
            )
            if self._use_foldcp_full_projection_source_order(z_res):
                source_chunk = min(self.pair_chunk_size or n_struct, n_struct)
                source_col_index = torch.arange(n_struct, device=parent.device)
                for source_row_start in range(0, n_struct, source_chunk):
                    source_row_end = min(source_row_start + source_chunk, n_struct)
                    local_row_start = max(row_start, source_row_start)
                    local_row_end = min(valid_row_end, source_row_end)
                    source_row_index = torch.arange(
                        source_row_start,
                        source_row_end,
                        device=parent.device,
                    )
                    pair_features = self._build_structural_pair_features_for_rows(
                        context=context,
                        row_index=source_row_index,
                    )
                    if z_res_spec is None:
                        z_source = self._gather_parent_pair_rows(
                            z=z_res,
                            parent=parent,
                            row_index=source_row_index,
                        )
                    else:
                        z_source = self._gather_parent_pair_tile_from_foldcp_local(
                            z=z_res,
                            z_spec=z_res_spec,
                            parent=parent,
                            row_index=source_row_index,
                            col_index=source_col_index,
                            mesh=mesh,
                        )
                    if local_row_start >= local_row_end:
                        continue
                    delta = self._pair_project_by_role(
                        z=z_source,
                        role=role,
                        pair_features=pair_features,
                        row_index=source_row_index,
                    )
                    if delta is not None:
                        z_source = z_source + delta
                    z_source = z_source + self._make_pair_init_bias(
                        pair_features,
                        dtype=z_source.dtype,
                    )
                    attn_bias_source = self._make_attention_bias(
                        pair_features,
                        dtype=z_source.dtype,
                    )
                    local_source_rows = slice(
                        local_row_start - source_row_start,
                        local_row_end - source_row_start,
                    )
                    local_target_rows = slice(
                        local_row_start - row_start,
                        local_row_end - row_start,
                    )
                    local_source_cols = slice(col_start, valid_col_end)
                    z_local[
                        ...,
                        local_target_rows,
                        :valid_cols,
                        :,
                    ].copy_(
                        z_source[
                            ...,
                            local_source_rows,
                            local_source_cols,
                            :,
                        ]
                    )
                    attn_bias_local[
                        ...,
                        local_target_rows,
                        :valid_cols,
                    ].copy_(
                        self._reshape_pair_term_for_target(
                            attn_bias_source[
                                local_source_rows,
                                local_source_cols,
                            ],
                            attn_bias_local[..., local_target_rows, :valid_cols],
                        )
                    )
            else:
                pair_row_chunk = min(self.pair_chunk_size or 256, valid_rows)
                for row_offset in range(0, valid_rows, pair_row_chunk):
                    row_chunk_end = min(row_offset + pair_row_chunk, valid_rows)
                    row_index = torch.arange(
                        row_start + row_offset,
                        row_start + row_chunk_end,
                        device=parent.device,
                    )
                    pair_features = self._build_structural_pair_features_for_tile(
                        context=context,
                        row_index=row_index,
                        col_index=col_index,
                    )
                    if z_res_spec is None:
                        z_valid = self._gather_parent_pair_tile(
                            z=z_res,
                            parent=parent,
                            row_index=row_index,
                            col_index=col_index,
                        )
                    else:
                        z_valid = self._gather_parent_pair_tile_from_foldcp_local(
                            z=z_res,
                            z_spec=z_res_spec,
                            parent=parent,
                            row_index=row_index,
                            col_index=col_index,
                            mesh=mesh,
                        )
                    target_z = z_local[..., row_offset:row_chunk_end, :valid_cols, :]
                    if target_z.is_contiguous() and not torch.is_grad_enabled():
                        target_z.copy_(z_valid)
                        del z_valid
                        z_work = target_z
                        writeback_z_work = False
                    else:
                        z_work = z_valid
                        writeback_z_work = True
                    if torch.is_grad_enabled():
                        delta = self._pair_project_by_role(
                            z=z_work,
                            role=role,
                            pair_features=pair_features,
                            row_index=row_index,
                            col_index=col_index,
                        )
                        if delta is not None:
                            z_work = z_work + delta
                        z_work = z_work + self._make_pair_init_bias(
                            pair_features,
                            dtype=z_work.dtype,
                        )
                    else:
                        z_work = self._add_pair_project_by_role_inplace(
                            z=z_work,
                            role=role,
                            pair_features=pair_features,
                            row_index=row_index,
                            col_index=col_index,
                            flat_chunk_size=min(
                                65536,
                                pair_features["role_pair_type"].numel(),
                            ),
                        )
                        z_work = self._add_pair_init_bias_chunked_inplace(
                            z_work,
                            pair_features,
                            dtype=z_work.dtype,
                            row_chunk_size=min(64, row_chunk_end - row_offset),
                        )
                    if writeback_z_work:
                        target_z.copy_(z_work)
                    self._write_attention_bias_inplace(
                        attn_bias_local[..., row_offset:row_chunk_end, :valid_cols],
                        pair_features=pair_features,
                        dtype=z_work.dtype,
                    )

        return (
            s_inputs_struct,
            s_struct,
            z_local.contiguous(),
            {"structural_pair_attn_bias": attn_bias_local.contiguous()},
            {"structural_pair_attn_bias": bias_spec},
            z_spec,
        )
