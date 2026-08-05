# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for atom local-window attention."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Optional

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec
from opendde.model.modules.primitives import (
    gather_pair_embedding_in_dense_trunk,
    rearrange_qk_to_dense_trunk,
)


@dataclass(frozen=True)
class FoldCPWindowShardSpec:
    """Ownership metadata for local atom windows.

    The atom path is already local-window attention rather than full N_atom^2
    attention. Fold-CP owns this stage by splitting the query windows across
    the CP ranks while preserving the same per-window key neighborhood.
    """

    n_atom: int
    n_windows: int
    n_queries: int
    n_keys: int
    q_pad: int
    block_range: tuple[int, int]
    size_cp: int
    padded_n_windows: Optional[int] = None


def _flat_cp_rank(mesh: FoldCPProcessMesh) -> int:
    return mesh.layout.to_linear(mesh.coord)


def window_block_range(n_windows: int, mesh: FoldCPProcessMesh) -> tuple[int, int]:
    """Return the contiguous window-block range owned by the local CP rank."""

    blocks_per_rank = int(math.ceil(n_windows / mesh.config.size_cp))
    start = _flat_cp_rank(mesh) * blocks_per_rank
    return start, start + blocks_per_rank


def atom_window_token_indices(
    atom_to_token_idx: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build dense local-window token indices without gathering token pairs."""

    query_idx, key_idx, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        dim_q=-1,
        dim_k=-1,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )
    return query_idx.long(), key_idx.long(), pad_info


def serial_atom_window_pair_context(
    z_token: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Serial token-pair broadcast to full atom local-window blocks."""

    query_idx, key_idx, pad_info = atom_window_token_indices(
        atom_to_token_idx,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )
    return gather_pair_embedding_in_dense_trunk(z_token, query_idx, key_idx), pad_info


def distributed_atom_window_pair_context(
    z_token: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    mesh: FoldCPProcessMesh,
    compute_mask: bool = True,
) -> tuple[torch.Tensor, FoldCPWindowShardSpec, dict[str, object]]:
    """Build only the local CP rank's atom local-window token-pair blocks."""

    query_idx, key_idx, pad_info = atom_window_token_indices(
        atom_to_token_idx,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )
    n_windows = query_idx.shape[-2]
    block_range = window_block_range(n_windows, mesh)
    blocks_per_rank = block_range[1] - block_range[0]
    valid_end = min(block_range[1], n_windows)
    local = z_token.new_zeros(blocks_per_rank, n_queries, n_keys, z_token.shape[-1])
    if block_range[0] < valid_end:
        valid_local = gather_pair_embedding_in_dense_trunk(
            z_token,
            query_idx[block_range[0] : valid_end],
            key_idx[block_range[0] : valid_end],
        )
        local[: valid_end - block_range[0]] = valid_local
    local = local.contiguous()
    spec = FoldCPWindowShardSpec(
        n_atom=int(atom_to_token_idx.shape[-1]),
        n_windows=int(n_windows),
        n_queries=int(n_queries),
        n_keys=int(n_keys),
        q_pad=int(pad_info["q_pad"]),
        block_range=block_range,
        size_cp=mesh.config.size_cp,
        padded_n_windows=int(blocks_per_rank * mesh.config.size_cp),
    )
    return local, spec, pad_info


def gather_pair_embedding_in_dense_trunk_from_foldcp_local(
    z_local: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Gather atom-window token-pair blocks from Fold-CP local pair tiles.

    This is the local-pair equivalent of ``gather_pair_embedding_in_dense_trunk``.
    It streams one CP tile at a time with broadcast, fills the requested
    atom-window entries owned by the current rank, and never materializes the
    full ``[N, N, C]`` pair tensor on any rank.
    """

    if z_local.ndim != 3:
        raise ValueError("Fold-CP atom-window pair lookup expects z_local=[T,T,C].")
    if idx_q.ndim != 2 or idx_k.ndim != 2:
        raise ValueError("idx_q and idx_k must be [N_block, N_query/key].")

    idx_q = idx_q.long()
    idx_k = idx_k.long()
    out = z_local.new_zeros(
        (idx_q.shape[0], idx_q.shape[1], idx_k.shape[-1], z_local.shape[-1])
    )
    tile_rows = z_spec.local_shape[z_spec.pair_dims[0]]
    tile_cols = z_spec.local_shape[z_spec.pair_dims[1]]
    n_token = z_spec.original_shape[z_spec.pair_dims[0]]
    group_rank = dist.get_rank(mesh.group_2d)
    row_chunk_size = int(
        os.environ.get("OPENDDE_FOLDCP_ATOM_WINDOW_PAIR_ROW_CHUNK", "16")
    )
    row_chunk_size = max(1, row_chunk_size)
    recv_buffer = (
        z_local.new_empty((row_chunk_size, tile_cols, z_local.shape[-1]))
        if mesh.layout.numel > 1
        else None
    )

    for cp_rank in range(mesh.layout.numel):
        row_coord, col_coord = mesh.layout.to_coord(cp_rank)
        row_start = row_coord * tile_rows
        col_start = col_coord * tile_cols
        col_end = min(col_start + tile_cols, n_token)
        src_global_rank = dist.get_global_rank(mesh.group_2d, cp_rank)
        k_in_tile = (idx_k >= col_start) & (idx_k < col_end)
        local_has_k = bool(k_in_tile.any())

        for row_offset in range(0, tile_rows, row_chunk_size):
            chunk_rows = min(row_chunk_size, tile_rows - row_offset)
            row_chunk_start = row_start + row_offset
            row_chunk_end = min(row_chunk_start + chunk_rows, n_token)
            q_in_chunk = (idx_q >= row_chunk_start) & (idx_q < row_chunk_end)
            local_needs_chunk = bool(q_in_chunk.any()) and local_has_k

            if group_rank == cp_rank:
                shard = z_local[row_offset : row_offset + chunk_rows].contiguous()
            else:
                if recv_buffer is None:
                    raise RuntimeError("missing Fold-CP atom-window receive buffer.")
                shard = recv_buffer[:chunk_rows]
            dist.broadcast(shard, src=src_global_rank, group=mesh.group_2d)

            if local_needs_chunk:
                for block_index in range(idx_q.shape[0]):
                    q_pos = torch.nonzero(
                        q_in_chunk[block_index], as_tuple=False
                    ).flatten()
                    k_pos = torch.nonzero(
                        k_in_tile[block_index], as_tuple=False
                    ).flatten()
                    if q_pos.numel() == 0 or k_pos.numel() == 0:
                        continue
                    q_local = idx_q[block_index, q_pos] - row_chunk_start
                    k_local = idx_k[block_index, k_pos] - col_start
                    out[block_index, q_pos[:, None], k_pos[None, :], :] = shard[
                        q_local[:, None],
                        k_local[None, :],
                        :,
                    ]
            del shard
    return out.contiguous()


def _gather_pair_rows_one_by_p(
    z_local: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Gather shared query rows from a 1xP column-sharded pair tensor."""

    side = int(mesh.layout.shape[1])
    tile_cols = int(z_spec.local_shape[z_spec.pair_dims[1]])
    n_token = int(z_spec.original_shape[z_spec.pair_dims[1]])
    batch = int(idx_q.shape[0])
    n_query = int(idx_q.shape[1])
    channels = int(z_local.shape[-1])

    local_rows = z_local.index_select(0, idx_q.reshape(-1)).contiguous()
    gathered_rows = z_local.new_empty((side * batch * n_query, tile_cols, channels))
    dist.all_gather_into_tensor(
        gathered_rows,
        local_rows,
        group=mesh.group_2d,
    )
    dense_rows = (
        gathered_rows.reshape(side, batch, n_query, tile_cols, channels)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
        .reshape(batch, n_query, side * tile_cols, channels)[..., :n_token, :]
    )
    gather_index = idx_k[:, None, :, None].expand(
        batch,
        n_query,
        idx_k.shape[1],
        channels,
    )
    return torch.gather(dense_rows, dim=2, index=gather_index).contiguous()


def gather_window_blocks(
    local: torch.Tensor,
    spec: FoldCPWindowShardSpec,
    group: dist.ProcessGroup,
    *,
    block_dim: int,
) -> torch.Tensor:
    """Gather equal-sized local window blocks back to serial window order."""

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before gather.")
    if local.shape[block_dim] != spec.block_range[1] - spec.block_range[0]:
        raise ValueError("local block dimension does not match FoldCPWindowShardSpec.")

    group_size = dist.get_world_size(group)
    if group_size != spec.size_cp:
        raise ValueError("window shard spec size does not match process group size.")
    blocks_per_rank = spec.block_range[1] - spec.block_range[0]
    group_rank = dist.get_rank(group)
    expected_start = group_rank * blocks_per_rank
    if spec.block_range[0] != expected_start:
        raise ValueError("window block range must follow flat CP rank order.")

    padded_n_windows = spec.padded_n_windows or spec.n_windows
    if padded_n_windows != blocks_per_rank * group_size:
        raise RuntimeError("gathered window block count does not match spec.")

    block_dim = block_dim % local.dim()
    local_front = local.movedim(block_dim, 0).contiguous()
    full_front = local_front.new_empty((padded_n_windows, *local_front.shape[1:]))
    full_front[spec.block_range[0] : spec.block_range[1]].copy_(local_front)

    send_chunk = local_front
    for step in range(1, group_size):
        recv_chunk = torch.empty_like(local_front)
        send_rank = (group_rank - 1) % group_size
        recv_rank = (group_rank + 1) % group_size
        work = dist.batch_isend_irecv(
            [
                dist.P2POp(
                    dist.isend,
                    send_chunk,
                    dist.get_global_rank(group, send_rank),
                    group,
                ),
                dist.P2POp(
                    dist.irecv,
                    recv_chunk,
                    dist.get_global_rank(group, recv_rank),
                    group,
                ),
            ]
        )
        for item in work:
            item.wait()
        source_rank = (group_rank + step) % group_size
        start = source_rank * blocks_per_rank
        full_front[start : start + blocks_per_rank].copy_(recv_chunk)
        send_chunk = recv_chunk

    full_front = full_front[: spec.n_windows]
    full = full_front.movedim(0, block_dim)
    return full


def _window_attention_blocks(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    *,
    mask: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = torch.einsum("...bqd,...bkd->...bqk", q_blocks, k_blocks)
    if attn_bias is not None:
        logits = logits + attn_bias
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(0)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("...bqk,...bkd->...bqd", weights, v_blocks).contiguous()


def _qkv_window_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    q_blocks, kv_blocks, pad_info = rearrange_qk_to_dense_trunk(
        q=q,
        k=[k, v],
        dim_q=-2,
        dim_k=[-2, -2],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=True,
    )
    return q_blocks, kv_blocks[0], kv_blocks[1], pad_info


def serial_atom_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Serial local-window attention reference used by Task9 validation."""

    q_blocks, k_blocks, v_blocks, pad_info = _qkv_window_blocks(
        q,
        k,
        v,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    out_blocks = _window_attention_blocks(
        q_blocks,
        k_blocks,
        v_blocks,
        mask=pad_info["mask_trunked"],
        attn_bias=attn_bias,
    )
    out = out_blocks.reshape(*out_blocks.shape[:-3], -1, out_blocks.shape[-1])
    q_pad = int(pad_info["q_pad"])
    if q_pad > 0:
        out = out[..., :-q_pad, :]
    return out.contiguous()


def distributed_atom_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_queries: int,
    n_keys: int,
    mesh: FoldCPProcessMesh,
    local_attn_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, FoldCPWindowShardSpec]:
    """Compute only the local CP rank's atom local-window attention blocks."""

    q_blocks, k_blocks, v_blocks, pad_info = _qkv_window_blocks(
        q,
        k,
        v,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    n_windows = q_blocks.shape[-3]
    block_range = window_block_range(n_windows, mesh)
    blocks_per_rank = block_range[1] - block_range[0]
    valid_end = min(block_range[1], n_windows)
    local_out = q_blocks.new_zeros(
        *q_blocks.shape[:-3],
        blocks_per_rank,
        n_queries,
        q_blocks.shape[-1],
    )
    if block_range[0] < valid_end:
        valid_blocks = valid_end - block_range[0]
        local_slice = slice(block_range[0], valid_end)
        local_bias = None
        if local_attn_bias is not None:
            local_bias = local_attn_bias[..., :valid_blocks, :, :]
        local_out[..., :valid_blocks, :, :] = _window_attention_blocks(
            q_blocks[..., local_slice, :, :],
            k_blocks[..., local_slice, :, :],
            v_blocks[..., local_slice, :, :],
            mask=pad_info["mask_trunked"][local_slice],
            attn_bias=local_bias,
        )
    spec = FoldCPWindowShardSpec(
        n_atom=int(q.shape[-2]),
        n_windows=int(n_windows),
        n_queries=int(n_queries),
        n_keys=int(n_keys),
        q_pad=int(pad_info["q_pad"]),
        block_range=block_range,
        size_cp=mesh.config.size_cp,
        padded_n_windows=int(blocks_per_rank * mesh.config.size_cp),
    )
    return local_out.contiguous(), spec


def gather_window_attention_output(
    local_out: torch.Tensor,
    spec: FoldCPWindowShardSpec,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Gather local window attention output blocks and remove query padding."""

    out_blocks = gather_window_blocks(local_out, spec, group, block_dim=-3)
    out = out_blocks.reshape(*out_blocks.shape[:-3], -1, out_blocks.shape[-1])
    if spec.q_pad > 0:
        out = out[..., : -spec.q_pad, :]
    return out.contiguous()
