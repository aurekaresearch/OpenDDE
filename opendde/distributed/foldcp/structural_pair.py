# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for structural-token pair context."""

from __future__ import annotations

import torch

from opendde.distributed.foldcp.comm import run_group_rank_action_synchronized
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    make_pair_shard_spec,
)


def structural_role_pair_type(role: torch.Tensor, n_types: int = 8) -> torch.Tensor:
    """Small deterministic role-pair typing helper for CP validation."""

    return (role[:, None] * 3 + role[None, :]).remainder(n_types).long()


def serial_structural_pair_context(
    z_res: torch.Tensor,
    parent: torch.Tensor,
    role: torch.Tensor,
    role_pair_embedding: torch.Tensor,
) -> torch.Tensor:
    """Serial parent-pair gather plus role-pair bias.

    z_res: [B, N_res, N_res, C]
    parent/role: [N_struct]
    role_pair_embedding: [n_role_pair_types, C]
    """

    z_parent = z_res.index_select(1, parent).index_select(2, parent)
    role_pair_type = structural_role_pair_type(role, role_pair_embedding.shape[0])
    return z_parent + role_pair_embedding.index_select(
        0, role_pair_type.reshape(-1)
    ).reshape(
        role.shape[0],
        role.shape[0],
        z_res.shape[-1],
    )


def distributed_structural_pair_context(
    z_res: torch.Tensor,
    parent: torch.Tensor,
    role: torch.Tensor,
    role_pair_embedding: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> tuple[torch.Tensor, FoldCPPairShardSpec]:
    """Build only the current rank's structural pair tile."""

    def _compute_local_structural_pair():
        if z_res.ndim != 4:
            raise ValueError("z_res must be [B, N_res, N_res, C].")
        if parent.ndim != 1 or role.ndim != 1:
            raise ValueError("parent and role must be one-dimensional.")
        if parent.shape != role.shape:
            raise ValueError("parent and role must have the same length.")
        if role_pair_embedding.ndim != 2:
            raise ValueError("role_pair_embedding must be [N_type, C].")
        if role_pair_embedding.shape[-1] != z_res.shape[-1]:
            raise ValueError("role-pair and residue-pair channels must match.")

        n_struct = int(parent.shape[0])
        spec = make_pair_shard_spec(
            (z_res.shape[0], n_struct, n_struct, z_res.shape[-1]),
            mesh,
            pair_dims=(1, 2),
        )
        row_start, row_end = spec.row_range
        col_start, col_end = spec.col_range
        valid_row_end = min(row_end, n_struct)
        valid_col_end = min(col_end, n_struct)
        valid_rows = max(0, valid_row_end - row_start)
        valid_cols = max(0, valid_col_end - col_start)
        local = z_res.new_zeros(spec.local_shape)
        if valid_rows > 0 and valid_cols > 0:
            row_parent = parent[row_start:valid_row_end]
            col_parent = parent[col_start:valid_col_end]
            z_tile = z_res.index_select(1, row_parent).index_select(2, col_parent)

            row_role = role[row_start:valid_row_end]
            col_role = role[col_start:valid_col_end]
            role_pair_type = (row_role[:, None] * 3 + col_role[None, :]).remainder(
                role_pair_embedding.shape[0]
            )
            bias_tile = role_pair_embedding.index_select(
                0, role_pair_type.reshape(-1)
            ).reshape(
                valid_rows,
                valid_cols,
                z_res.shape[-1],
            )
            local[:, :valid_rows, :valid_cols, :] = z_tile + bias_tile
        return local.contiguous(), spec

    if int(mesh.layout.numel) <= 1:
        return _compute_local_structural_pair()
    result = run_group_rank_action_synchronized(
        _compute_local_structural_pair,
        group=mesh.group_2d,
        description="structural-pair local context computation",
    )
    if result is None:  # pragma: no cover
        raise RuntimeError("structural-pair local context returned no result.")
    return result
