# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for confidence-head pair logits."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    _copy_pair_shard_into_output,
    gather_pair_tensor_like,
)
from opendde.model.utils import one_hot
from opendde.utils.torch_utils import cdist


def _transpose_pair_tile_collective(
    z_pair_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Collect the reciprocal pair tile without materializing every tile at once."""

    z_pair_t_send = z_pair_local.transpose(-2, -3).contiguous()
    return mesh.ring_comm().comm_2d_trans.exchange(z_pair_t_send).contiguous()


def _collect_pair_row_slab(
    z_pair_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Collect this rank row block across column tiles with row-ring P2P."""

    side = mesh.layout.shape[1]
    if side == 1:
        return z_pair_local.contiguous()

    ring = mesh.ring_comm()
    row_tiles: list[torch.Tensor | None] = [None for _ in range(side)]
    row_tiles[mesh.coord[1]] = z_pair_local.contiguous()
    ready = row_tiles[mesh.coord[1]]
    for step in range(1, side):
        ready = ring.comm_row.exchange(ready.contiguous())
        source_col = (mesh.coord[1] + step) % side
        row_tiles[source_col] = ready
    if any(item is None for item in row_tiles):
        raise RuntimeError("failed to collect confidence row slab.")
    row_slab = torch.cat([item for item in row_tiles if item is not None], dim=-2)
    del row_tiles
    return row_slab.contiguous()


def _linear_pair_row_slab_with_source_grid_launch(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    valid_rows: int,
) -> torch.Tensor:
    if valid_rows <= 0:
        return linear(x)
    flat = x.contiguous().reshape(-1, x.shape[-1])
    source_rows = int(original_n) * int(original_n)
    launch = flat.new_zeros(source_rows, flat.shape[-1])
    row_offsets = (torch.arange(valid_rows, device=x.device) + int(row_start)) * int(
        original_n
    )
    source_index = (
        row_offsets[:, None] + torch.arange(int(original_n), device=x.device)[None, :]
    ).reshape(-1)
    launch.index_copy_(0, source_index, flat[: source_index.numel()])
    projected = linear(launch).index_select(0, source_index)
    return projected.reshape(valid_rows, int(original_n), -1)


def add_confidence_distance_embedding_local(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    x_pred_rep_coords: torch.Tensor,
    lower_bins: torch.Tensor,
    upper_bins: torch.Tensor,
    linear_onehot: torch.nn.Module,
    linear_distance: torch.nn.Module,
) -> torch.Tensor:
    """Add confidence distance embeddings to a CP local pair tile.

    This is the local-tile equivalent of:

    ``z_pair += linear_onehot(one_hot(cdist(coords, coords)))``
    ``z_pair += linear_distance(cdist(coords, coords)[..., None])``

    Padding rows/columns in the local tile are left unchanged so padded tokens
    cannot feed fake distance information into later pair operations.
    """

    row_dim, col_dim = z_pair_spec.pair_dims
    if row_dim != 0 or col_dim != 1 or z_pair_local.ndim != 3:
        raise ValueError("confidence distance embedding expects z_pair_local=[T,T,C].")

    row_start, row_end = z_pair_spec.row_range
    col_start, col_end = z_pair_spec.col_range
    n_token = z_pair_spec.original_shape[row_dim]
    valid_row_end = min(row_end, n_token)
    valid_col_end = min(col_end, n_token)
    if row_start >= valid_row_end or col_start >= valid_col_end:
        return z_pair_local

    row_chunk_size = int(
        os.environ.get("OPENDDE_FOLDCP_CONFIDENCE_DISTANCE_ROW_CHUNK", "0")
    )
    if row_chunk_size <= 0:
        row_chunk_size = valid_row_end - row_start

    out = z_pair_local
    coords = x_pred_rep_coords.to(torch.float32)
    col_coords = coords[col_start:valid_col_end]
    local_col_count = valid_col_end - col_start

    for global_row_start in range(row_start, valid_row_end, row_chunk_size):
        global_row_end = min(global_row_start + row_chunk_size, valid_row_end)
        local_row_start = global_row_start - row_start
        local_row_end = global_row_end - row_start
        row_coords = coords[global_row_start:global_row_end]
        with torch.amp.autocast("cuda", enabled=False):
            distance_pred = cdist(row_coords, col_coords)
        local_target = out[local_row_start:local_row_end, :local_col_count, :]
        onehot_input = one_hot(
            x=distance_pred,
            lower_bins=lower_bins,
            upper_bins=upper_bins,
        ).to(dtype=linear_onehot.weight.dtype)
        onehot_update = linear_onehot(onehot_input)
        local_target = local_target + onehot_update
        distance_update = linear_distance(
            distance_pred.unsqueeze(dim=-1).to(dtype=linear_distance.weight.dtype)
        )
        out[local_row_start:local_row_end, :local_col_count, :] = (
            local_target + distance_update
        )
        del distance_pred, onehot_input, onehot_update, distance_update, local_target
        torch.cuda.empty_cache()
    return out


def _confidence_pair_logits_local_rowslab(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    layer_norm: torch.nn.Module,
    linear: torch.nn.Module,
    add_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project local confidence logits from a source-layout row slab."""

    z_row_slab = _collect_pair_row_slab(z_pair_local, mesh)
    row_start, row_end = z_pair_spec.row_range
    n_token = z_pair_spec.original_shape[z_pair_spec.pair_dims[0]]
    valid_row_end = min(row_end, n_token)
    valid_rows = max(0, valid_row_end - row_start)
    if add_local is not None:
        add_row_slab = _collect_pair_row_slab(add_local, mesh)
        z_row_slab = z_row_slab + add_row_slab
        del add_row_slab

    tile_rows = z_row_slab.shape[-3]
    if n_token <= 3072 and valid_rows > 0:
        z_norm = layer_norm(z_row_slab[:valid_rows, :n_token])
        logits_row_slab = _linear_pair_row_slab_with_source_grid_launch(
            linear,
            z_norm,
            original_n=n_token,
            row_start=row_start,
            valid_rows=valid_rows,
        )
        del z_norm
    else:
        # Small row-slab GEMM launches drift from the serial confidence logits path.
        # Pad the launch rows to at least 128 while keeping the Fold-CP tile payload
        # consistent across ranks; final gather/crop ignores the extra zero rows.
        row_launch = min(n_token, max(tile_rows + 64, 128))
        if row_launch != tile_rows:
            z_launch = z_row_slab.new_zeros(
                (row_launch, z_row_slab.shape[-2], z_row_slab.shape[-1])
            )
            z_launch[:tile_rows] = z_row_slab
        else:
            z_launch = z_row_slab
        logits_row_slab = linear(layer_norm(z_launch))

    if (
        logits_row_slab.shape[-3] != tile_rows
        or logits_row_slab.shape[-2] != z_row_slab.shape[-2]
    ):
        logits_padded = logits_row_slab.new_zeros(
            (tile_rows, z_row_slab.shape[-2], logits_row_slab.shape[-1])
        )
        copy_rows = min(tile_rows, logits_row_slab.shape[-3])
        copy_cols = min(z_row_slab.shape[-2], logits_row_slab.shape[-2])
        logits_padded[
            :copy_rows,
            :copy_cols,
        ] = logits_row_slab[:copy_rows, :copy_cols]
        logits_row_slab = logits_padded
    del z_row_slab

    tile_col = z_pair_local.shape[-2]
    col_start = mesh.coord[1] * tile_col
    col_end = col_start + tile_col
    logits_local = logits_row_slab[
        : z_pair_local.shape[-3], col_start:col_end, :
    ].contiguous()
    del logits_row_slab
    return logits_local


def _gather_pair_logit_chunk_to_rank0(
    *,
    full_output: torch.Tensor | None,
    local_chunk: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    row_start: int,
    row_end: int,
    dst_group_rank: int = 0,
) -> None:
    group = mesh.group_2d
    group_rank = dist.get_rank(group)
    local_chunk = local_chunk.contiguous()
    gathered = (
        [torch.empty_like(local_chunk) for _ in range(mesh.layout.numel)]
        if group_rank == dst_group_rank
        else None
    )
    dist.gather(
        local_chunk,
        gather_list=gathered,
        dst=mesh.cp_global_ranks[dst_group_rank],
        group=group,
    )

    if group_rank != dst_group_rank:
        return

    if full_output is None:
        raise ValueError("full_output must be allocated on the destination rank.")
    if gathered is None:
        raise ValueError("gathered shards must be available on the destination rank.")

    row_dim, col_dim = z_pair_spec.pair_dims
    tile_row = z_pair_spec.local_shape[row_dim]
    tile_col = z_pair_spec.local_shape[col_dim]
    for cp_rank in range(mesh.layout.numel):
        row, col = mesh.layout.to_coord(cp_rank)
        target_row_range = (row * tile_row + row_start, row * tile_row + row_end)
        target_col_range = (col * tile_col, (col + 1) * tile_col)
        _copy_pair_shard_into_output(
            full_output,
            gathered[cp_rank],
            z_pair_spec.pair_dims,
            target_row_range,
            target_col_range,
        )


def _stream_pair_logits_to_rank0(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    layer_norm: torch.nn.Module,
    linear: torch.nn.Module,
    add_local: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute confidence pair logits with source row-slab projection layout.

    Serial confidence applies layer norm and the output linear to a row-major
    ``[N, N, C]`` pair tensor. Applying the same modules to a local
    ``[tile, tile, C]`` tensor can choose a different CUDA GEMM shape and drift
    by a few ulps. Fold-CP keeps ownership local, but reconstructs only this
    rank row block across all columns before projection, then slices the local
    column tile and gathers the public logits on rank 0.
    """

    row_dim, col_dim = z_pair_spec.pair_dims
    if row_dim != 0 or col_dim != 1 or z_pair_local.ndim != 3:
        raise ValueError("confidence pair logits currently expect local z=[T,T,C].")

    logits_local = _confidence_pair_logits_local_rowslab(
        z_pair_local=z_pair_local,
        z_pair_spec=z_pair_spec,
        mesh=mesh,
        layer_norm=layer_norm,
        linear=linear,
        add_local=add_local,
    )

    output_dim = int(linear.weight.shape[0])
    full_output = None
    if dist.get_rank(mesh.group_2d) == 0:
        output_shape = list(z_pair_spec.original_shape)
        output_shape[-1] = output_dim
        full_output = z_pair_local.new_empty(tuple(output_shape))

    gather_row_chunk = min(128, max(1, z_pair_local.shape[0]))
    for row_start in range(0, z_pair_local.shape[0], gather_row_chunk):
        row_end = min(row_start + gather_row_chunk, z_pair_local.shape[0])
        _gather_pair_logit_chunk_to_rank0(
            full_output=full_output,
            local_chunk=logits_local[row_start:row_end],
            z_pair_spec=z_pair_spec,
            mesh=mesh,
            row_start=row_start,
            row_end=row_end,
        )
    del logits_local

    if full_output is None:
        return None
    return full_output


def distributed_confidence_pair_logits(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    pae_ln: torch.nn.Module,
    pae_linear: torch.nn.Module,
    pde_ln: torch.nn.Module,
    pde_linear: torch.nn.Module,
    compute_pae: bool = True,
    compute_pde: bool = True,
    gather_to_rank0_only: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute confidence pair logits from CP-sharded pair activations.

    PAE is pointwise on each pair tile. PDE uses ``z[i, j] + z[j, i]``; the
    reciprocal tile is obtained with the Fold-CP 2D transpose exchange, matching
    the serial ``z + z.transpose(-2, -3)`` formula without gathering full ``z``.
    """

    pae_pred = None
    pde_pred = None

    if compute_pae:
        if gather_to_rank0_only:
            pae_pred = _stream_pair_logits_to_rank0(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=mesh,
                layer_norm=pae_ln,
                linear=pae_linear,
            )
        else:
            pae_local = _confidence_pair_logits_local_rowslab(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=mesh,
                layer_norm=pae_ln,
                linear=pae_linear,
            )
            pae_pred = gather_pair_tensor_like(pae_local, z_pair_spec, mesh.group_2d)

    if compute_pde:
        z_pair_t_local = _transpose_pair_tile_collective(z_pair_local, mesh)
        if gather_to_rank0_only:
            pde_pred = _stream_pair_logits_to_rank0(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=mesh,
                layer_norm=pde_ln,
                linear=pde_linear,
                add_local=z_pair_t_local,
            )
        else:
            pde_local = _confidence_pair_logits_local_rowslab(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=mesh,
                layer_norm=pde_ln,
                linear=pde_linear,
                add_local=z_pair_t_local,
            )
            pde_pred = gather_pair_tensor_like(pde_local, z_pair_spec, mesh.group_2d)
        del z_pair_t_local

    return pae_pred, pde_pred
