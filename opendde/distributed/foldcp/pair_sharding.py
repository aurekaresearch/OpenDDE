# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""2D pair-tensor sharding helpers for Fold-CP."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.layout import FoldCP2DLayout
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh


@dataclass(frozen=True)
class FoldCPPairShardSpec:
    """Metadata needed to gather a local 2D pair shard back to serial layout."""

    original_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    pair_dims: tuple[int, int]
    row_range: tuple[int, int]
    col_range: tuple[int, int]
    mesh_shape: tuple[int, int]
    mesh_coord: tuple[int, int]

    @property
    def local_shape(self) -> tuple[int, ...]:
        shape = list(self.padded_shape)
        row_dim, col_dim = self.pair_dims
        shape[row_dim] = self.row_range[1] - self.row_range[0]
        shape[col_dim] = self.col_range[1] - self.col_range[0]
        return tuple(shape)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_pair_dims(ndim: int, pair_dims: tuple[int, int]) -> tuple[int, int]:
    row_dim, col_dim = pair_dims
    if row_dim < 0:
        row_dim += ndim
    if col_dim < 0:
        col_dim += ndim
    if row_dim == col_dim:
        raise ValueError("pair_dims must point to two different dimensions.")
    if not (0 <= row_dim < ndim and 0 <= col_dim < ndim):
        raise ValueError(f"pair_dims {pair_dims} are outside tensor ndim={ndim}.")
    return row_dim, col_dim


def infer_pair_dims(tensor: torch.Tensor, n_token: int) -> Optional[tuple[int, int]]:
    """Infer common OpenDDE pair dimensions for a token pair tensor."""

    if tensor.ndim >= 2 and tensor.shape[-2:] == (n_token, n_token):
        return (tensor.ndim - 2, tensor.ndim - 1)
    if tensor.ndim >= 3 and tensor.shape[-3:-1] == (n_token, n_token):
        return (tensor.ndim - 3, tensor.ndim - 2)
    return None


def _padded_pair_size(n_pair: int, partitions: int) -> int:
    return int(math.ceil(n_pair / partitions) * partitions)


def _slice_for_pair(
    ndim: int,
    pair_dims: tuple[int, int],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> tuple[slice, ...]:
    slices = [slice(None)] * ndim
    slices[pair_dims[0]] = slice(*row_range)
    slices[pair_dims[1]] = slice(*col_range)
    return tuple(slices)


def _pair_crop_is_noop(
    shape: tuple[int, ...],
    pair_dims: tuple[int, int],
    original_shape: tuple[int, ...],
) -> bool:
    row_dim, col_dim = pair_dims
    return (
        shape[row_dim] == original_shape[row_dim]
        and shape[col_dim] == original_shape[col_dim]
    )


def _pair_output_shape_like(
    local_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
) -> tuple[int, ...]:
    shape = list(local_tensor.shape)
    row_dim, col_dim = spec.pair_dims
    shape[row_dim] = spec.original_shape[row_dim]
    shape[col_dim] = spec.original_shape[col_dim]
    return tuple(shape)


def _copy_pair_shard_into_output(
    output: torch.Tensor,
    shard: torch.Tensor,
    pair_dims: tuple[int, int],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> None:
    row_dim, col_dim = pair_dims
    row_start, row_end = row_range
    col_start, col_end = col_range
    valid_row_end = min(row_end, output.shape[row_dim])
    valid_col_end = min(col_end, output.shape[col_dim])
    if row_start >= valid_row_end or col_start >= valid_col_end:
        return

    valid_rows = valid_row_end - row_start
    valid_cols = valid_col_end - col_start
    target = _slice_for_pair(
        output.ndim,
        pair_dims,
        (row_start, valid_row_end),
        (col_start, valid_col_end),
    )
    source = _slice_for_pair(
        shard.ndim,
        pair_dims,
        (0, valid_rows),
        (0, valid_cols),
    )
    output[target] = shard[source]


def shard_pair_tensor(
    tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_dims: tuple[int, int],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, FoldCPPairShardSpec]:
    """Shard a full pair tensor into the current rank's local 2D tile."""

    pair_dims = _normalize_pair_dims(tensor.ndim, pair_dims)
    row_dim, col_dim = pair_dims
    n_row = tensor.shape[row_dim]
    n_col = tensor.shape[col_dim]
    if n_row != n_col:
        raise ValueError(f"pair tensor must be square, got {n_row} x {n_col}.")
    mesh_rows, mesh_cols = mesh.layout.shape
    padded_rows = _padded_pair_size(n_row, mesh_rows)
    padded_cols = _padded_pair_size(n_col, mesh_cols)
    padded_shape = list(tensor.shape)
    padded_shape[row_dim] = padded_rows
    padded_shape[col_dim] = padded_cols

    row_tile = padded_rows // mesh_rows
    col_tile = padded_cols // mesh_cols
    row_start = mesh.coord[0] * row_tile
    col_start = mesh.coord[1] * col_tile
    row_range = (row_start, row_start + row_tile)
    col_range = (col_start, col_start + col_tile)
    local_shape = list(tensor.shape)
    local_shape[row_dim] = row_tile
    local_shape[col_dim] = col_tile
    local = tensor.new_full(tuple(local_shape), pad_value)

    valid_row_end = min(row_range[1], n_row)
    valid_col_end = min(col_range[1], n_col)
    if row_start < valid_row_end and col_start < valid_col_end:
        valid_rows = valid_row_end - row_start
        valid_cols = valid_col_end - col_start
        source = _slice_for_pair(
            tensor.ndim,
            pair_dims,
            (row_start, valid_row_end),
            (col_start, valid_col_end),
        )
        target = _slice_for_pair(
            tensor.ndim,
            pair_dims,
            (0, valid_rows),
            (0, valid_cols),
        )
        local[target] = tensor[source]
    spec = FoldCPPairShardSpec(
        original_shape=tuple(tensor.shape),
        padded_shape=tuple(padded_shape),
        pair_dims=pair_dims,
        row_range=row_range,
        col_range=col_range,
        mesh_shape=mesh.layout.shape,
        mesh_coord=mesh.coord,
    )
    return local, spec


def make_pair_shard_spec(
    original_shape: tuple[int, ...],
    mesh: FoldCPProcessMesh,
    pair_dims: tuple[int, int],
) -> FoldCPPairShardSpec:
    """Create pair-shard metadata without materializing the full pair tensor."""

    pair_dims = _normalize_pair_dims(len(original_shape), pair_dims)
    row_dim, col_dim = pair_dims
    n_row = original_shape[row_dim]
    n_col = original_shape[col_dim]
    if n_row != n_col:
        raise ValueError(f"pair tensor must be square, got {n_row} x {n_col}.")
    mesh_rows, mesh_cols = mesh.layout.shape
    padded_rows = _padded_pair_size(n_row, mesh_rows)
    padded_cols = _padded_pair_size(n_col, mesh_cols)
    padded_shape = list(original_shape)
    padded_shape[row_dim] = padded_rows
    padded_shape[col_dim] = padded_cols

    row_tile = padded_rows // mesh_rows
    col_tile = padded_cols // mesh_cols
    row_start = mesh.coord[0] * row_tile
    col_start = mesh.coord[1] * col_tile
    return FoldCPPairShardSpec(
        original_shape=tuple(original_shape),
        padded_shape=tuple(padded_shape),
        pair_dims=pair_dims,
        row_range=(row_start, row_start + row_tile),
        col_range=(col_start, col_start + col_tile),
        mesh_shape=mesh.layout.shape,
        mesh_coord=mesh.coord,
    )


def gather_pair_tensor(
    local_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather local pair shards from all CP ranks and crop padding."""

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before gather.")
    layout = FoldCP2DLayout(spec.mesh_shape)
    gathered = [torch.empty_like(local_tensor) for _ in range(layout.numel)]
    dist.all_gather(gathered, local_tensor.contiguous(), group=group)

    full = local_tensor.new_empty(spec.original_shape)
    tile_row = spec.local_shape[spec.pair_dims[0]]
    tile_col = spec.local_shape[spec.pair_dims[1]]
    for cp_rank, shard in enumerate(gathered):
        row, col = layout.to_coord(cp_rank)
        row_range = (row * tile_row, (row + 1) * tile_row)
        col_range = (col * tile_col, (col + 1) * tile_col)
        _copy_pair_shard_into_output(full, shard, spec.pair_dims, row_range, col_range)
    return full


def gather_pair_tensor_like(
    local_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather a pair-like shard whose non-pair dimensions differ from ``spec``.

    This is used after local CP pair transforms such as confidence logits:
    the row/column tile layout is inherited from the original pair shard, while
    the trailing channel/bin dimension may be smaller than the source ``z``.
    """

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before gather.")
    row_dim, col_dim = spec.pair_dims
    if local_tensor.ndim != len(spec.original_shape):
        raise ValueError(
            "local pair-like tensor must keep the same rank as the source pair tensor."
        )

    layout = FoldCP2DLayout(spec.mesh_shape)
    gathered = [torch.empty_like(local_tensor) for _ in range(layout.numel)]
    dist.all_gather(gathered, local_tensor.contiguous(), group=group)

    full = local_tensor.new_empty(_pair_output_shape_like(local_tensor, spec))
    tile_row = local_tensor.shape[row_dim]
    tile_col = local_tensor.shape[col_dim]
    for cp_rank, shard in enumerate(gathered):
        row, col = layout.to_coord(cp_rank)
        row_range = (row * tile_row, (row + 1) * tile_row)
        col_range = (col * tile_col, (col + 1) * tile_col)
        _copy_pair_shard_into_output(full, shard, spec.pair_dims, row_range, col_range)
    return full


def gather_pair_tensor_like_to_rank(
    local_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
    group: dist.ProcessGroup,
    dst_group_rank: int = 0,
) -> torch.Tensor | None:
    """Gather a pair-like shard only on one group rank.

    All ranks participate, but only ``dst_group_rank`` materializes the full
    pair-like tensor. This avoids replicating final PAE/PDE logits on every CP
    rank after the model has already finished the distributed pair computation.
    """

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before gather.")
    row_dim, col_dim = spec.pair_dims
    if local_tensor.ndim != len(spec.original_shape):
        raise ValueError(
            "local pair-like tensor must keep the same rank as the source pair tensor."
        )

    layout = FoldCP2DLayout(spec.mesh_shape)
    group_rank = dist.get_rank(group)
    dst_global_rank = dist.get_global_rank(group, dst_group_rank)
    local_tensor = local_tensor.contiguous()

    if group_rank != dst_group_rank:
        dist.send(local_tensor, dst=dst_global_rank, group=group)
        return None

    full = local_tensor.new_empty(_pair_output_shape_like(local_tensor, spec))
    tile_row = local_tensor.shape[row_dim]
    tile_col = local_tensor.shape[col_dim]

    for cp_rank in range(layout.numel):
        row, col = layout.to_coord(cp_rank)
        row_range = (row * tile_row, (row + 1) * tile_row)
        col_range = (col * tile_col, (col + 1) * tile_col)
        if cp_rank == dst_group_rank:
            _copy_pair_shard_into_output(
                full, local_tensor, spec.pair_dims, row_range, col_range
            )
        else:
            shard = torch.empty_like(local_tensor)
            src_global_rank = dist.get_global_rank(group, cp_rank)
            dist.recv(shard, src=src_global_rank, group=group)
            _copy_pair_shard_into_output(
                full, shard, spec.pair_dims, row_range, col_range
            )
    return full


def shard_pair_feature_dict(
    feature_dict: dict[str, object],
    mesh: FoldCPProcessMesh,
    n_token: int,
) -> tuple[dict[str, object], dict[str, FoldCPPairShardSpec]]:
    """Shard every tensor in a feature dict that has token-pair dimensions."""

    sharded: dict[str, object] = {}
    specs: dict[str, FoldCPPairShardSpec] = {}
    for key, value in feature_dict.items():
        if isinstance(value, torch.Tensor):
            pair_dims = infer_pair_dims(value, n_token)
            if pair_dims is not None:
                local, spec = shard_pair_tensor(value, mesh, pair_dims=pair_dims)
                sharded[key] = local
                specs[key] = spec
                continue
        sharded[key] = value
    return sharded, specs
