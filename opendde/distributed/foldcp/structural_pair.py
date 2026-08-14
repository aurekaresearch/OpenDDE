# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for structural-token pair context."""

from __future__ import annotations

import torch

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec


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

    n_struct = parent.shape[0]
    if n_struct % mesh.layout.shape[0] != 0:
        raise ValueError(
            "Task8 structural pair context expects n_struct divisible by mesh side."
        )
    tile = n_struct // mesh.layout.shape[0]
    row_start = mesh.coord[0] * tile
    col_start = mesh.coord[1] * tile
    row_range = (row_start, row_start + tile)
    col_range = (col_start, col_start + tile)

    row_parent = parent[row_range[0] : row_range[1]]
    col_parent = parent[col_range[0] : col_range[1]]
    z_tile = z_res.index_select(1, row_parent).index_select(2, col_parent)

    row_role = role[row_range[0] : row_range[1]]
    col_role = role[col_range[0] : col_range[1]]
    role_pair_type = (row_role[:, None] * 3 + col_role[None, :]).remainder(
        role_pair_embedding.shape[0]
    )
    bias_tile = role_pair_embedding.index_select(0, role_pair_type.reshape(-1)).reshape(
        tile,
        tile,
        z_res.shape[-1],
    )
    local = (z_tile + bias_tile).contiguous()
    spec = FoldCPPairShardSpec(
        original_shape=(z_res.shape[0], n_struct, n_struct, z_res.shape[-1]),
        padded_shape=(z_res.shape[0], n_struct, n_struct, z_res.shape[-1]),
        pair_dims=(1, 2),
        row_range=row_range,
        col_range=col_range,
        mesh_shape=mesh.layout.shape,
        mesh_coord=mesh.coord,
    )
    return local, spec
