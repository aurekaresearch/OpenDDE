# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for distogram/contact probabilities."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.launch import (
    foldcp_pair_row_slab_linear_with_source_launch_policy,
)
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    _copy_pair_shard_into_output,
    gather_pair_tensor_like,
    shard_pair_tensor,
)


def _one_by_p_transpose_peer_rounds(
    *,
    cp_rank: int,
    mesh_cols: int,
) -> tuple[tuple[int, int], ...]:
    """Return rank-invariant send/receive peers for a pairwise all-to-all."""

    return tuple(
        (
            (int(cp_rank) + step) % int(mesh_cols),
            (int(cp_rank) - step) % int(mesh_cols),
        )
        for step in range(1, int(mesh_cols))
    )


def _transpose_pair_tile_collective(
    z_pair_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    *,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    """Exchange the reciprocal pair tile without a full all-gather buffer."""

    mesh_rows, mesh_cols = mesh.layout.shape
    if mesh_rows == 1 and mesh_cols > 1:
        result_device = (
            z_pair_local.device
            if output_device is None
            else torch.device(output_device)
        )
        n_token = z_pair_local.shape[-3]
        tile = z_pair_local.shape[-2]
        if result_device == z_pair_local.device:
            padded_n = int(mesh_cols) * int(tile)
            if n_token > padded_n:
                raise ValueError(
                    f"pair length {n_token} exceeds padded length {padded_n}."
                )
            if n_token == padded_n:
                padded = z_pair_local.contiguous()
            else:
                padded = z_pair_local.new_zeros(
                    z_pair_local.shape[:-3] + (padded_n, tile, z_pair_local.shape[-1])
                )
                padded[..., :n_token, :, :].copy_(z_pair_local)
            prefix_dims = padded.ndim - 3
            send = (
                padded.reshape(
                    padded.shape[:-3]
                    + (
                        int(mesh_cols),
                        int(tile),
                        int(tile),
                        padded.shape[-1],
                    )
                )
                .movedim(prefix_dims, 0)
                .contiguous()
            )
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=mesh.group_row)
            transposed = (
                recv.movedim(0, prefix_dims)
                .transpose(-3, -2)
                .contiguous()
                .reshape(
                    z_pair_local.shape[:-3] + (padded_n, tile, z_pair_local.shape[-1])
                )
            )
            return transposed[..., :n_token, :, :].contiguous()

        block_shape = list(z_pair_local.shape)
        block_shape[-3] = tile
        block_shape[-2] = tile
        z_pair_t_recv = torch.zeros(
            tuple(z_pair_local.shape),
            dtype=z_pair_local.dtype,
            device=result_device,
        )

        def _send_block(
            destination: int,
            channel_start: int,
            channel_end: int,
        ) -> torch.Tensor:
            row_start = int(destination) * tile
            valid_rows = max(0, min(tile, n_token - row_start))
            chunk_shape = list(block_shape)
            chunk_shape[-1] = channel_end - channel_start
            block = torch.zeros(
                tuple(chunk_shape),
                dtype=z_pair_local.dtype,
                device=result_device,
            )
            if valid_rows:
                block[..., :valid_rows, :, :] = z_pair_local[
                    ...,
                    row_start : row_start + valid_rows,
                    :,
                    channel_start:channel_end,
                ]
            return block.contiguous()

        self_row_start = int(mesh.cp_rank) * tile
        self_valid_rows = max(
            0,
            min(tile, n_token - self_row_start),
        )
        block_row_bytes = tile * tile * int(z_pair_local.element_size())
        channel_chunk = max(
            1,
            (256 * 1024**2) // max(1, block_row_bytes),
        )
        for channel_start in range(
            0,
            int(z_pair_local.shape[-1]),
            channel_chunk,
        ):
            channel_end = min(
                channel_start + channel_chunk,
                int(z_pair_local.shape[-1]),
            )
            if self_valid_rows:
                z_pair_t_recv[
                    ...,
                    self_row_start : self_row_start + self_valid_rows,
                    :self_valid_rows,
                    channel_start:channel_end,
                ].copy_(
                    z_pair_local[
                        ...,
                        self_row_start : self_row_start + self_valid_rows,
                        :self_valid_rows,
                        channel_start:channel_end,
                    ].transpose(-3, -2)
                )
        for send_peer, recv_peer in _one_by_p_transpose_peer_rounds(
            cp_rank=mesh.cp_rank,
            mesh_cols=mesh_cols,
        ):
            recv_start = int(recv_peer) * tile
            recv_valid_rows = max(
                0,
                min(tile, n_token - recv_start),
            )
            for channel_start in range(
                0,
                int(z_pair_local.shape[-1]),
                channel_chunk,
            ):
                channel_end = min(
                    channel_start + channel_chunk,
                    int(z_pair_local.shape[-1]),
                )
                send_block = _send_block(
                    send_peer,
                    channel_start,
                    channel_end,
                )
                recv_block = torch.empty_like(send_block)
                operations = [
                    dist.P2POp(
                        dist.irecv,
                        recv_block,
                        mesh.cp_global_ranks[recv_peer],
                        group=mesh.group_2d,
                    ),
                    dist.P2POp(
                        dist.isend,
                        send_block,
                        mesh.cp_global_ranks[send_peer],
                        group=mesh.group_2d,
                    ),
                ]
                for work in dist.batch_isend_irecv(operations):
                    work.wait()
                if recv_valid_rows and self_valid_rows:
                    z_pair_t_recv[
                        ...,
                        recv_start : recv_start + recv_valid_rows,
                        :self_valid_rows,
                        channel_start:channel_end,
                    ].copy_(
                        recv_block[
                            ...,
                            :self_valid_rows,
                            :recv_valid_rows,
                            :,
                        ].transpose(-3, -2)
                    )
                del send_block, recv_block
        return z_pair_t_recv

    if output_device is not None and torch.device(output_device) != z_pair_local.device:
        raise ValueError("CPU-source transpose is only supported for 1xP Fold-CP.")
    z_pair_t_send = z_pair_local.transpose(-2, -3).contiguous()
    transposed_rank = mesh.layout.transpose_rank(mesh.coord)
    z_pair_t_recv = torch.empty_like(z_pair_t_send)

    # Broadcast one source tile at a time. Every rank participates in the same
    # collective order, but each rank only keeps the reciprocal tile it needs.
    for source_rank in range(mesh.layout.numel):
        buffer = (
            z_pair_t_send
            if mesh.cp_rank == source_rank
            else torch.empty_like(z_pair_t_send)
        )
        dist.broadcast(
            buffer,
            src=mesh.cp_global_ranks[source_rank],
            group=mesh.group_2d,
        )
        if source_rank == transposed_rank:
            z_pair_t_recv.copy_(buffer)
        if mesh.cp_rank != source_rank:
            del buffer
    return z_pair_t_recv.contiguous()


def _project_pair_row_slab_local(
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    keep_full_one_by_p: bool = False,
) -> torch.Tensor:
    """Project this rank's pair tile using the source row-slab layout.

    The serial distogram head applies ``linear`` to the contiguous
    ``[N, N, C]`` pair tensor. Calling the same module on a local
    ``[tile, tile, C]`` slice can select a different CUDA GEMM shape and drift
    by a few ulps. Fold-CP keeps pair ownership local, but reconstructs only the
    current row block across all column tiles before projection, then slices the
    current column tile back out. This preserves the source row-major projection
    layout without materializing full pair logits on any rank.
    """

    if z_pair_local.ndim != 3:
        raise ValueError(
            "Fold-CP distogram row-slab projection expects [tile, tile, c_z]."
        )

    row_start, row_end = z_pair_spec.row_range
    col_start, col_end = z_pair_spec.col_range
    n_token = z_pair_spec.original_shape[z_pair_spec.pair_dims[0]]
    valid_rows = max(0, min(row_end, n_token) - row_start)
    tile_col = int(z_pair_local.shape[-2])
    valid_cols = max(0, min(col_end, n_token) - col_start)
    source_slab_bytes = (
        int(n_token)
        * int(n_token)
        * int(z_pair_local.shape[-1])
        * int(z_pair_local.element_size())
    )
    source_slab_budget = int(
        os.environ.get(
            "OPENDDE_FOLDCP_DISTOGRAM_SOURCE_SLAB_MAX_BYTES",
            str(16 * 1024**3),
        )
    )
    if (
        mesh.layout.shape[0] == 1
        and source_slab_budget >= 0
        and source_slab_bytes > source_slab_budget
    ):
        chunk_cols = int(os.environ.get("OPENDDE_FOLDCP_DISTOGRAM_COL_CHUNK", "256"))
        if chunk_cols <= 0:
            raise ValueError("Fold-CP distogram column chunk must be positive.")
        out_features = int(linear.weight.shape[0])
        logits_local = z_pair_local.new_zeros(
            (z_pair_local.shape[-3], tile_col, out_features)
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
            source_chunk[
                :valid_rows,
                chunk_offset : chunk_offset + overlap_cols,
            ] = z_pair_local[
                :valid_rows,
                local_offset : local_offset + overlap_cols,
            ]
            projected_chunk = linear(source_chunk)
            logits_local[
                :valid_rows,
                local_offset : local_offset + overlap_cols,
            ] = projected_chunk[
                :valid_rows,
                chunk_offset : chunk_offset + overlap_cols,
            ]
            del projected_chunk, source_chunk
        return logits_local.contiguous()

    side = mesh.layout.shape[1]
    if side == 1:
        z_row_slab = z_pair_local.contiguous()
    else:
        ring = mesh.ring_comm()
        row_tiles: list[torch.Tensor | None] = [None for _ in range(side)]
        row_tiles[mesh.coord[1]] = z_pair_local.contiguous()
        ready = row_tiles[mesh.coord[1]]
        for step in range(1, side):
            ready = ring.comm_row.exchange(ready.contiguous())
            source_col = (mesh.coord[1] + step) % side
            row_tiles[source_col] = ready
        if any(item is None for item in row_tiles):
            raise RuntimeError("failed to collect distogram row slab.")
        z_row_slab = torch.cat([item for item in row_tiles if item is not None], dim=-2)
        z_row_slab = z_row_slab.contiguous()
        del row_tiles

    if mesh.layout.shape[0] == 1:
        # A 1 x P mesh owns every source row on every rank after the row-ring
        # collection above.  Crop CP padding before the projection and launch
        # the exact [N, N, C] operation used by the serial head.  The generic
        # source-grid launcher preserves logical indexing, but its segmented
        # GEMM shape can differ from the serial GEMM at P >= 4 by a few ulps.
        source_pair = z_row_slab[:n_token, :n_token, :].contiguous()
        logits_row_slab = linear(source_pair)
        del source_pair
    else:
        logits_row_slab = foldcp_pair_row_slab_linear_with_source_launch_policy(
            linear,
            z_row_slab,
            original_n=n_token,
            row_start=row_start,
            col_start=0,
            valid_rows=valid_rows,
            valid_cols=n_token,
        )
    del z_row_slab
    if keep_full_one_by_p and int(mesh.layout.shape[0]) == 1:
        return logits_row_slab.contiguous()

    logits_local = logits_row_slab.new_zeros(
        (z_pair_local.shape[-3], tile_col, logits_row_slab.shape[-1])
    )
    if valid_cols:
        logits_local[:, :valid_cols, :] = logits_row_slab[
            :, col_start : col_start + valid_cols, :
        ]
    logits_local = logits_local.contiguous()
    del logits_row_slab
    return logits_local


def _distogram_bin_tops(
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=device,
        dtype=dtype,
    )
    return torch.cat([boundaries, boundaries.new_tensor([1e8])], dim=0)


def _contact_probs_with_serial_launch_shape(
    logits_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    bin_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Use the serial softmax row count where its temporary is capacity-safe."""

    n_token = int(z_pair_spec.original_shape[z_pair_spec.pair_dims[0]])
    if mesh.layout.shape[0] != 1 or n_token > 1536:
        return None
    col_start, col_end = z_pair_spec.col_range
    valid_cols = max(0, min(col_end, n_token) - col_start)
    source_logits = logits_local.new_zeros((n_token, n_token, logits_local.shape[-1]))
    source_logits[:, col_start : col_start + valid_cols, :] = logits_local[
        :n_token, :valid_cols, :
    ]
    source_probs = torch.nn.functional.softmax(source_logits, dim=-1)
    source_contact = source_probs[..., bin_mask].sum(dim=-1)
    contact_local = logits_local.new_zeros(logits_local.shape[:-1])
    contact_local[:n_token, :valid_cols] = source_contact[
        :, col_start : col_start + valid_cols
    ]
    return contact_local.contiguous()


def distogram_contact_probs_local(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
) -> torch.Tensor:
    """Compute contact probabilities for one CP local pair tile.

    The serial DistogramHead computes ``linear(z_ij) + linear(z_ji)``. The
    reciprocal ``z_ji`` tile is obtained via the Fold-CP 2D transpose exchange,
    so no rank needs to materialize full ``[N, N, bins]`` logits.
    """

    n_token = int(z_pair_spec.original_shape[z_pair_spec.pair_dims[0]])
    source_slab_bytes = (
        n_token
        * n_token
        * int(z_pair_local.shape[-1])
        * int(z_pair_local.element_size())
    )
    source_slab_budget = int(
        os.environ.get(
            "OPENDDE_FOLDCP_DISTOGRAM_SOURCE_SLAB_MAX_BYTES",
            str(16 * 1024**3),
        )
    )
    keep_full_one_by_p = (
        int(mesh.layout.shape[0]) == 1
        and int(mesh.layout.shape[1]) > 1
        and source_slab_budget >= 0
        and source_slab_bytes <= source_slab_budget
    )
    logits_direct_local = _project_pair_row_slab_local(
        z_pair_local,
        z_pair_spec,
        mesh,
        linear,
        keep_full_one_by_p=keep_full_one_by_p,
    )
    if keep_full_one_by_p:
        col_start, col_end = z_pair_spec.col_range
        valid_cols = max(0, min(col_end, n_token) - col_start)
        tile_cols = int(z_pair_local.shape[-2])
        bin_tops = _distogram_bin_tops(
            min_bin=min_bin,
            max_bin=max_bin,
            no_bins=no_bins,
            device=logits_direct_local.device,
            dtype=logits_direct_local.dtype,
        )
        if n_token <= 1536:
            logits = logits_direct_local + logits_direct_local.transpose(-2, -3)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            contact_full = probs[..., bin_tops <= thres].sum(dim=-1)
            contact_local = logits.new_zeros((n_token, tile_cols))
            contact_local[:, :valid_cols] = contact_full[
                :, col_start : col_start + valid_cols
            ]
            del logits, probs, contact_full
        else:
            logits = logits_direct_local[
                :, col_start : col_start + valid_cols, :
            ] + logits_direct_local[col_start : col_start + valid_cols, :, :].transpose(
                -2, -3
            )
            probs = torch.nn.functional.softmax(logits, dim=-1)
            contact_local = logits.new_zeros((n_token, tile_cols))
            contact_local[:, :valid_cols] = probs[..., bin_tops <= thres].sum(dim=-1)
            del logits, probs
        del logits_direct_local
        return contact_local.contiguous()

    logits_t_local = _transpose_pair_tile_collective(logits_direct_local, mesh)
    logits_local = logits_direct_local + logits_t_local
    del logits_direct_local, logits_t_local
    bin_tops = _distogram_bin_tops(
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        device=logits_local.device,
        dtype=logits_local.dtype,
    )
    bin_mask = bin_tops <= thres
    contact_local = _contact_probs_with_serial_launch_shape(
        logits_local,
        z_pair_spec,
        mesh,
        bin_mask,
    )
    if contact_local is None:
        probs_local = torch.nn.functional.softmax(logits_local, dim=-1)
        contact_local = probs_local[..., bin_mask].sum(dim=-1)
        del probs_local
    del logits_local
    return contact_local.contiguous()


def distributed_distogram_contact_probs(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
    gather_to_rank0_only: bool = False,
) -> torch.Tensor | None:
    """Compute contact probabilities from CP-sharded pair activations."""

    contact_local = distogram_contact_probs_local(
        z_pair_local=z_pair_local,
        z_pair_spec=z_pair_spec,
        mesh=mesh,
        linear=linear,
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        thres=thres,
    )
    contact_local_with_channel = contact_local.unsqueeze(dim=-1)
    if gather_to_rank0_only:
        contact = _gather_pair_like_collective_to_rank0(
            contact_local_with_channel,
            z_pair_spec,
            mesh,
        )
    else:
        contact = gather_pair_tensor_like(
            contact_local_with_channel,
            z_pair_spec,
            mesh.group_2d,
        )
    if contact is None:
        return None
    return contact.squeeze(dim=-1).contiguous()


def _gather_pair_like_collective_to_rank0(
    local_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor | None:
    row_dim, col_dim = spec.pair_dims
    group_rank = torch.distributed.get_rank(mesh.group_2d)
    gathered = (
        [torch.empty_like(local_tensor) for _ in range(mesh.layout.numel)]
        if group_rank == 0
        else None
    )
    torch.distributed.gather(
        local_tensor.contiguous(),
        gather_list=gathered,
        dst=mesh.cp_global_ranks[0],
        group=mesh.group_2d,
    )
    if group_rank != 0:
        return None
    if gathered is None:
        raise ValueError("gathered shards must be available on the destination rank.")

    output_shape = list(local_tensor.shape)
    output_shape[row_dim] = spec.original_shape[row_dim]
    output_shape[col_dim] = spec.original_shape[col_dim]
    full = local_tensor.new_empty(tuple(output_shape))
    tile_row = local_tensor.shape[row_dim]
    tile_col = local_tensor.shape[col_dim]
    for cp_rank, shard in enumerate(gathered):
        row, col = mesh.layout.to_coord(cp_rank)
        row_range = (row * tile_row, (row + 1) * tile_row)
        col_range = (col * tile_col, (col + 1) * tile_col)
        _copy_pair_shard_into_output(
            full,
            shard,
            spec.pair_dims,
            row_range,
            col_range,
        )
    return full


def _reciprocal_tile_from_full_pair(
    z_pair: torch.Tensor,
    spec: FoldCPPairShardSpec,
    reference: torch.Tensor,
) -> torch.Tensor:
    row_start, row_end = spec.row_range
    col_start, col_end = spec.col_range
    n_token = spec.original_shape[spec.pair_dims[0]]
    valid_row_end = min(row_end, n_token)
    valid_col_end = min(col_end, n_token)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    reciprocal = reference.new_zeros(reference.shape)
    if valid_rows == 0 or valid_cols == 0:
        return reciprocal
    reciprocal_valid = z_pair[
        ...,
        col_start:valid_col_end,
        row_start:valid_row_end,
        :,
    ].transpose(-2, -3)
    reciprocal[..., :valid_rows, :valid_cols, :] = reciprocal_valid
    return reciprocal.contiguous()


def distributed_distogram_contact_probs_from_full_pair(
    *,
    z_pair: torch.Tensor,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
    gather_to_rank0_only: bool = False,
) -> torch.Tensor | None:
    """Compute contact probabilities from full pair input without full logits.

    This is the NCCL-safe main-path bridge while earlier stages still expose a
    full ``pair_z``. It slices only this rank's pair tile and reciprocal tile,
    computes local distogram logits/contact, and gathers contact probabilities.
    """

    z_pair_local, z_pair_spec = shard_pair_tensor(z_pair, mesh, pair_dims=(-3, -2))
    z_pair_t_local = _reciprocal_tile_from_full_pair(
        z_pair=z_pair,
        spec=z_pair_spec,
        reference=z_pair_local,
    )
    logits_local = linear(z_pair_local) + linear(z_pair_t_local)
    del z_pair_t_local
    probs_local = torch.nn.functional.softmax(logits_local, dim=-1)
    bin_tops = _distogram_bin_tops(
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        device=logits_local.device,
        dtype=logits_local.dtype,
    )
    contact_local = probs_local[..., bin_tops <= thres].sum(dim=-1).contiguous()
    del logits_local, probs_local
    contact_local_with_channel = contact_local.unsqueeze(dim=-1)
    if gather_to_rank0_only:
        contact = _gather_pair_like_collective_to_rank0(
            contact_local_with_channel,
            z_pair_spec,
            mesh,
        )
    else:
        contact = gather_pair_tensor_like(
            contact_local_with_channel,
            z_pair_spec,
            mesh.group_2d,
        )
    if contact is None:
        return None
    return contact.squeeze(dim=-1).contiguous()
