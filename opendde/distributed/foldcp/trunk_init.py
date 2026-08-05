# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP local construction for trunk pair initialization."""

from __future__ import annotations

from typing import Any

import os
import torch

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    make_pair_shard_spec,
)
from opendde.distributed.foldcp.launch import (
    foldcp_pair_row_slab_linear_with_source_grid_launch,
    foldcp_pair_row_slab_linear_with_source_launch_policy,
)


def _valid_ranges(
    spec: FoldCPPairShardSpec,
) -> tuple[int, int, int, int, int, int]:
    row_start, row_end = spec.row_range
    col_start, col_end = spec.col_range
    n_token = spec.original_shape[spec.pair_dims[0]]
    valid_row_end = min(row_end, n_token)
    valid_col_end = min(col_end, n_token)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    return row_start, valid_row_end, col_start, valid_col_end, valid_rows, valid_cols


def _local_pair_zeros(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return torch.zeros(
        shape,
        dtype=reference.dtype if dtype is None else dtype,
        device=reference.device,
    )


def _slice_tensor_pair_to_local(
    pair_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    row_start, row_end, col_start, col_end, valid_rows, valid_cols = _valid_ranges(spec)
    local_shape = list(pair_tensor.shape)
    local_shape[spec.pair_dims[0]] = spec.local_shape[spec.pair_dims[0]]
    local_shape[spec.pair_dims[1]] = spec.local_shape[spec.pair_dims[1]]
    local = _local_pair_zeros(pair_tensor, tuple(local_shape))
    if valid_rows == 0 or valid_cols == 0:
        return local

    source = [slice(None)] * pair_tensor.ndim
    source[spec.pair_dims[0]] = slice(row_start, row_end)
    source[spec.pair_dims[1]] = slice(col_start, col_end)
    target = [slice(None)] * pair_tensor.ndim
    target[spec.pair_dims[0]] = slice(0, valid_rows)
    target[spec.pair_dims[1]] = slice(0, valid_cols)
    local[tuple(target)] = pair_tensor[tuple(source)]
    return local.contiguous()


def _materialize_relp_local(
    relp_feature: Any,
    spec: FoldCPPairShardSpec,
    reference: torch.Tensor,
) -> torch.Tensor:
    row_start, row_end, col_start, col_end, valid_rows, valid_cols = _valid_ranges(spec)
    if hasattr(relp_feature, "feature_dim"):
        local_shape = list(spec.local_shape)
        local_shape[-1] = int(relp_feature.feature_dim)
        local = _local_pair_zeros(reference, tuple(local_shape))
        if valid_rows == 0 or valid_cols == 0:
            return local
        relp_valid = relp_feature.materialize(
            row_slice=slice(row_start, row_end),
            col_slice=slice(col_start, col_end),
        ).to(device=reference.device, dtype=reference.dtype)
        local[..., :valid_rows, :valid_cols, :] = relp_valid
        return local.contiguous()
    return _slice_tensor_pair_to_local(relp_feature, spec)


def apply_trunk_z_cycle_local(
    *,
    z_init_local: torch.Tensor,
    z_local: torch.Tensor,
    layernorm_z_cycle: torch.nn.Module,
    linear_z_cycle: torch.nn.Module,
    z_spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    """Apply the trunk recycling z update on a local pair tile."""

    row_start, _, col_start, _, valid_rows, valid_cols = _valid_ranges(z_spec)
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    z_cycle_update = foldcp_pair_row_slab_linear_with_source_grid_launch(
        linear_z_cycle,
        layernorm_z_cycle(z_local),
        original_n=original_n,
        row_start=row_start,
        col_start=col_start,
        valid_rows=valid_rows,
        valid_cols=valid_cols,
    )
    return z_init_local + z_cycle_update


def build_trunk_z_init_local(
    *,
    s_init: torch.Tensor,
    linear_zinit1: torch.nn.Module,
    linear_zinit2: torch.nn.Module,
    relative_position_encoding: torch.nn.Module,
    linear_token_bond: torch.nn.Module,
    relp_feature: Any,
    token_bonds: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> tuple[torch.Tensor, FoldCPPairShardSpec]:
    """Build local trunk z_init tile with the same math as serial OpenDDE."""

    n_token = int(s_init.shape[-2])
    source_pair_rows = n_token * n_token
    for prefix_dim in s_init.shape[:-2]:
        source_pair_rows *= int(prefix_dim)
    c_z = int(linear_zinit1.weight.shape[0])
    full_shape = (*s_init.shape[:-2], n_token, n_token, c_z)
    spec = make_pair_shard_spec(full_shape, mesh, pair_dims=(-3, -2))

    z_local = _local_pair_zeros(s_init, spec.local_shape)

    row_start, row_end, col_start, col_end, valid_rows, valid_cols = _valid_ranges(spec)
    if valid_rows > 0 and valid_cols > 0:
        z1_full = linear_zinit1(s_init)
        z2_full = linear_zinit2(s_init)
        z_valid = (
            z1_full[..., row_start:row_end, None, :]
            + z2_full[..., None, col_start:col_end, :]
        )
        z_local[..., :valid_rows, :valid_cols, :] = z_valid

    relp_shape = (
        *s_init.shape[:-2],
        n_token,
        n_token,
        relative_position_encoding.linear_no_bias.in_features,
    )
    relp_spec = make_pair_shard_spec(relp_shape, mesh, pair_dims=(-3, -2))
    relp_local = _materialize_relp_local(relp_feature, relp_spec, s_init)
    relp_update = foldcp_pair_row_slab_linear_with_source_launch_policy(
        relative_position_encoding.linear_no_bias,
        relp_local,
        original_n=n_token,
        row_start=row_start,
        col_start=col_start,
        valid_rows=valid_rows,
        valid_cols=valid_cols,
    )
    z_local = z_local + relp_update

    bond_shape = (*s_init.shape[:-2], n_token, n_token)
    bond_spec = make_pair_shard_spec(bond_shape, mesh, pair_dims=(-2, -1))
    token_bonds_local = _slice_tensor_pair_to_local(token_bonds, bond_spec)
    token_bond_update = foldcp_pair_row_slab_linear_with_source_launch_policy(
        linear_token_bond,
        token_bonds_local.unsqueeze(dim=-1),
        original_n=n_token,
        row_start=row_start,
        col_start=col_start,
        valid_rows=valid_rows,
        valid_cols=valid_cols,
    )
    z_local = z_local + token_bond_update
    return z_local.contiguous(), spec
