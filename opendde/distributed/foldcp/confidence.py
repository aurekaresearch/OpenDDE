# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for confidence-head pair logits."""

from __future__ import annotations

import math
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


_CONFIDENCE_TRANSPOSE_CPU_OFFLOAD_MIN_BYTES = 2 * 1024**3
_CONFIDENCE_CPU_ADD_CHUNK_MAX_BYTES = 256 * 1024**2


def _confidence_should_offload_transpose_source(
    z_pair_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> bool:
    threshold = int(
        os.environ.get(
            "OPENDDE_FOLDCP_CONFIDENCE_TRANSPOSE_CPU_OFFLOAD_MIN_BYTES",
            _CONFIDENCE_TRANSPOSE_CPU_OFFLOAD_MIN_BYTES,
        )
    )
    return (
        not torch.is_grad_enabled()
        and z_pair_local.is_cuda
        and mesh.layout.shape[0] == 1
        and mesh.layout.shape[1] > 1
        and z_pair_local.numel() * z_pair_local.element_size() > threshold
    )


def _transpose_pair_tile_collective(
    z_pair_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    """Collect the reciprocal pair tile without materializing every tile at once."""

    if mesh.layout.shape[0] == 1 and mesh.layout.shape[1] > 1:
        from opendde.distributed.foldcp.distogram import (
            _transpose_pair_tile_collective as transpose_column_shards,
        )

        return transpose_column_shards(
            z_pair_local,
            mesh,
            output_device=output_device,
        )

    if output_device is not None and torch.device(output_device) != z_pair_local.device:
        raise ValueError("CPU-source transpose is only supported for 1xP Fold-CP.")
    z_pair_t_send = z_pair_local.transpose(-2, -3).contiguous()
    return mesh.ring_comm().comm_2d_trans.exchange(z_pair_t_send).contiguous()


def _add_cpu_pair_source_in_place(
    z_pair_t_local: torch.Tensor,
    z_pair_cpu: torch.Tensor,
) -> None:
    bytes_per_row = z_pair_cpu[..., :1, :, :].numel() * z_pair_cpu.element_size()
    row_chunk = max(
        1,
        _CONFIDENCE_CPU_ADD_CHUNK_MAX_BYTES // max(1, bytes_per_row),
    )
    for row_start in range(0, z_pair_cpu.shape[-3], row_chunk):
        row_end = min(row_start + row_chunk, z_pair_cpu.shape[-3])
        source_chunk = z_pair_cpu[..., row_start:row_end, :, :].to(
            z_pair_t_local.device
        )
        z_pair_t_local[..., row_start:row_end, :, :].add_(source_chunk)
        del source_chunk


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


def _confidence_should_stream_projection(
    *,
    mesh_rows: int,
    mesh_cols: int,
    source_slab_bytes: int,
    source_slab_budget: int,
    deterministic: bool,
) -> bool:
    """Whether confidence projection should stay on local column shards."""

    if mesh_rows != 1 or mesh_cols <= 1:
        return False
    if not deterministic:
        return True
    return source_slab_budget >= 0 and source_slab_bytes > source_slab_budget


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

    row_start, row_end = z_pair_spec.row_range
    col_start, col_end = z_pair_spec.col_range
    n_token = z_pair_spec.original_shape[z_pair_spec.pair_dims[0]]
    valid_row_end = min(row_end, n_token)
    valid_rows = max(0, valid_row_end - row_start)
    tile_cols = int(z_pair_local.shape[-2])
    valid_cols = max(0, min(col_end, n_token) - col_start)
    source_slab_bytes = (
        int(n_token)
        * int(n_token)
        * int(z_pair_local.shape[-1])
        * int(z_pair_local.element_size())
    )
    source_slab_budget = int(
        os.environ.get(
            "OPENDDE_FOLDCP_CONFIDENCE_SOURCE_SLAB_MAX_BYTES",
            str(16 * 1024**3),
        )
    )
    if _confidence_should_stream_projection(
        mesh_rows=mesh.layout.shape[0],
        mesh_cols=mesh.layout.shape[1],
        source_slab_bytes=source_slab_bytes,
        source_slab_budget=source_slab_budget,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    ):
        chunk_cols = int(os.environ.get("OPENDDE_FOLDCP_CONFIDENCE_COL_CHUNK", "256"))
        if chunk_cols <= 0:
            raise ValueError("Fold-CP confidence column chunk must be positive.")
        out_features = int(linear.weight.shape[0])
        logits_local = z_pair_local.new_zeros(
            (z_pair_local.shape[-3], tile_cols, out_features)
        )
        local_col_end = col_start + valid_cols
        for chunk_start in range(0, int(n_token), chunk_cols):
            chunk_end = min(chunk_start + chunk_cols, int(n_token))
            overlap_start = max(col_start, chunk_start)
            overlap_end = min(local_col_end, chunk_end)
            if overlap_start >= overlap_end:
                continue
            local_offset = overlap_start - col_start
            chunk_offset = overlap_start - chunk_start
            overlap_cols = overlap_end - overlap_start
            source_chunk = z_pair_local.new_zeros(
                (int(n_token), chunk_cols, z_pair_local.shape[-1])
            )
            source_values = z_pair_local[
                :valid_rows,
                local_offset : local_offset + overlap_cols,
            ]
            if add_local is not None:
                source_values = (
                    source_values
                    + add_local[
                        :valid_rows,
                        local_offset : local_offset + overlap_cols,
                    ]
                )
            source_chunk[
                :valid_rows,
                chunk_offset : chunk_offset + overlap_cols,
            ] = source_values
            projected_chunk = linear(layer_norm(source_chunk))
            logits_local[
                :valid_rows,
                local_offset : local_offset + overlap_cols,
            ] = projected_chunk[
                :valid_rows,
                chunk_offset : chunk_offset + overlap_cols,
            ]
            del projected_chunk, source_chunk, source_values
        return logits_local.contiguous()

    z_row_slab = _collect_pair_row_slab(z_pair_local, mesh)
    if add_local is not None:
        add_row_slab = _collect_pair_row_slab(add_local, mesh)
        z_row_slab = z_row_slab + add_row_slab
        del add_row_slab

    tile_rows = z_row_slab.shape[-3]
    row_slab_cols = z_row_slab.shape[-2]
    uses_source_grid_launch = n_token <= 3072 and valid_rows > 0
    if uses_source_grid_launch:
        z_norm = layer_norm(z_row_slab[:valid_rows, :n_token])
        del z_row_slab
        if z_norm.is_cuda:
            torch.cuda.empty_cache()
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
        or logits_row_slab.shape[-2] != row_slab_cols
    ):
        logits_padded = logits_row_slab.new_zeros(
            (tile_rows, row_slab_cols, logits_row_slab.shape[-1])
        )
        copy_rows = min(tile_rows, logits_row_slab.shape[-3])
        copy_cols = min(row_slab_cols, logits_row_slab.shape[-2])
        logits_padded[
            :copy_rows,
            :copy_cols,
        ] = logits_row_slab[:copy_rows, :copy_cols]
        logits_row_slab = logits_padded
    if not uses_source_grid_launch:
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
        output_shard = gathered[cp_rank]
        if output_shard.device != full_output.device:
            output_shard = output_shard.to(device=full_output.device)
        _copy_pair_shard_into_output(
            full_output,
            output_shard,
            z_pair_spec.pair_dims,
            target_row_range,
            target_col_range,
        )
        del output_shard


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
        output_bytes = math.prod(output_shape) * int(z_pair_local.element_size())
        gpu_output_max_bytes = int(
            os.environ.get(
                "OPENDDE_FOLDCP_CONFIDENCE_GPU_OUTPUT_MAX_BYTES",
                str(2 * 1024**3),
            )
        )
        spill_to_cpu = (
            z_pair_local.is_cuda
            and gpu_output_max_bytes >= 0
            and output_bytes > gpu_output_max_bytes
        )
        full_output = torch.empty(
            tuple(output_shape),
            dtype=z_pair_local.dtype,
            device="cpu" if spill_to_cpu else z_pair_local.device,
        )

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
        reuse_pair_storage = not torch.is_grad_enabled()
        offload_transpose_source = _confidence_should_offload_transpose_source(
            z_pair_local,
            mesh,
        )
        if offload_transpose_source:
            output_device = z_pair_local.device
            z_pair_cpu = z_pair_local.cpu()
            del z_pair_local
            torch.cuda.empty_cache()
            z_pair_t_local = _transpose_pair_tile_collective(
                z_pair_cpu,
                mesh,
                output_device=output_device,
            )
            _add_cpu_pair_source_in_place(z_pair_t_local, z_pair_cpu)
            del z_pair_cpu
            z_pair_local = z_pair_t_local
            del z_pair_t_local
            torch.cuda.empty_cache()
        else:
            z_pair_t_local = _transpose_pair_tile_collective(z_pair_local, mesh)
        reuse_pair_storage = reuse_pair_storage and not offload_transpose_source
        if reuse_pair_storage:
            z_pair_local.add_(z_pair_t_local)
            del z_pair_t_local
            if z_pair_local.is_cuda:
                torch.cuda.empty_cache()
        if gather_to_rank0_only:
            pde_pred = _stream_pair_logits_to_rank0(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=mesh,
                layer_norm=pde_ln,
                linear=pde_linear,
                add_local=(
                    None
                    if reuse_pair_storage or offload_transpose_source
                    else z_pair_t_local
                ),
            )
        else:
            pde_local = _confidence_pair_logits_local_rowslab(
                z_pair_local=z_pair_local,
                z_pair_spec=z_pair_spec,
                mesh=mesh,
                layer_norm=pde_ln,
                linear=pde_linear,
                add_local=(
                    None
                    if reuse_pair_storage or offload_transpose_source
                    else z_pair_t_local
                ),
            )
            pde_pred = gather_pair_tensor_like(pde_local, z_pair_spec, mesh.group_2d)
        if not reuse_pair_storage and not offload_transpose_source:
            del z_pair_t_local

    return pae_pred, pde_pred
