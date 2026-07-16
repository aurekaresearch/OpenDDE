# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP implementation of MSA pair weighted averaging core."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh


def serial_msa_pair_weighted_average(
    pair_logits: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Serial reference matching OpenDDE MSAPairWeightedAveraging softmax axis.

    pair_logits: [B, N, N, H]
    value: [B, S, N, H, C]
    output: [B, S, N, H, C]
    """

    weights = torch.softmax(pair_logits, dim=-2)
    return torch.einsum("bijh,bsjhc->bsihc", weights, value)


def shard_msa_value_by_token(value: torch.Tensor, mesh: FoldCPProcessMesh) -> torch.Tensor:
    """Shard value tensor on the token/source axis using the CP column coordinate."""

    token_dim = 2
    n_token = value.shape[token_dim]
    mesh_side = mesh.layout.shape[1]
    padded_n = int(math.ceil(n_token / mesh_side) * mesh_side)
    tile = padded_n // mesh_side
    start = mesh.coord[1] * tile
    end = start + tile
    local_shape = list(value.shape)
    local_shape[token_dim] = tile
    local = value.new_zeros(tuple(local_shape))
    valid_end = min(end, n_token)
    if start < valid_end:
        local[:, :, : valid_end - start, :, :] = value[:, :, start:valid_end, :, :]
    return local.contiguous()


def gather_msa_rows_from_cp(
    local_output: torch.Tensor,
    mesh: FoldCPProcessMesh,
    token_dim: int = 2,
    original_tokens: int | None = None,
) -> torch.Tensor:
    """Gather row-sharded MSA weighted-average output on every CP rank.

    After the source-token all-reduce, every rank in the same mesh row owns the
    same output-token chunk. Gathering over the mesh column group reconstructs
    the full token axis without involving any full pair tensor.
    """

    gathered = [torch.empty_like(local_output) for _ in range(mesh.layout.shape[0])]
    dist.all_gather(gathered, local_output.contiguous(), group=mesh.group_col)
    output = torch.cat(gathered, dim=token_dim)
    if original_tokens is not None:
        target = [slice(None)] * output.dim()
        target[token_dim] = slice(0, original_tokens)
        output = output[tuple(target)]
    return output.contiguous()


def collect_msa_pair_row_slab(
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_tokens: int,
) -> torch.Tensor:
    """Collect this rank's local output rows over all source-token columns.

    This is a row-ring all-gather specialized for pair tiles. It materializes
    only the current rank's row slab, not a full pair tensor.
    """

    n_cols = mesh.layout.shape[1]
    col_tile = z_local.shape[-2]
    padded_cols = n_cols * col_tile
    row_slab_shape = z_local.shape[:-2] + (padded_cols, z_local.shape[-1])
    row_slab = z_local.new_empty(row_slab_shape)

    local_col = mesh.coord[1]
    target = [slice(None)] * row_slab.dim()
    target[-2] = slice(local_col * col_tile, (local_col + 1) * col_tile)
    row_slab[tuple(target)] = z_local

    if n_cols == 1:
        return row_slab[..., :original_tokens, :].contiguous()

    row = mesh.coord[0]
    col = mesh.coord[1]
    layout = mesh.layout
    send_tensor = z_local.contiguous()
    for offset in range(1, n_cols):
        dest_col = (col + offset) % n_cols
        src_col = (col - offset) % n_cols
        dest_rank = mesh.cp_global_ranks[layout.to_linear((row, dest_col))]
        src_rank = mesh.cp_global_ranks[layout.to_linear((row, src_col))]
        recv_tensor = torch.empty_like(send_tensor)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, send_tensor, dest_rank),
                dist.P2POp(dist.irecv, recv_tensor, src_rank),
            ]
        )
        for request in requests:
            request.wait()
        target[-2] = slice(src_col * col_tile, (src_col + 1) * col_tile)
        row_slab[tuple(target)] = recv_tensor

    return row_slab[..., :original_tokens, :].contiguous()




def distributed_msa_pair_weighted_average_with_full_value(
    pair_logits_local: torch.Tensor,
    value: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_tokens: int | None = None,
) -> torch.Tensor:
    """Distributed MSA weighted average when every rank already owns full value.

    Production MSAStack computes value projections from full MSA on every rank.
    In deterministic mode, preserve exact serial softmax order by gathering only
    the much smaller pair-logit source shards, then multiplying by the existing
    full value tensor. Non-deterministic mode reuses the generic sharded-value
    reduction core.
    """

    if value.ndim != 5:
        raise ValueError("value must be [B, S, N, H, C].")
    if pair_logits_local.ndim != 4:
        raise ValueError("pair_logits_local must be [B, I_local, J_local, H].")
    if pair_logits_local.shape[0] != value.shape[0]:
        raise ValueError("batch dimensions must match.")
    if pair_logits_local.shape[3] != value.shape[3]:
        raise ValueError("head dimensions must match.")

    if torch.are_deterministic_algorithms_enabled():
        logits_parts = [
            torch.empty_like(pair_logits_local) for _ in range(mesh.layout.shape[1])
        ]
        dist.all_gather(
            logits_parts,
            pair_logits_local.contiguous(),
            group=mesh.group_row,
        )
        pair_logits = torch.cat(logits_parts, dim=2)
        if original_tokens is not None:
            pair_logits = pair_logits[:, :, :original_tokens, :]
            value = value[:, :, :original_tokens, :, :]
        return serial_msa_pair_weighted_average(pair_logits, value).contiguous()

    value_local = shard_msa_value_by_token(value, mesh)
    return distributed_msa_pair_weighted_average(
        pair_logits_local,
        value_local,
        mesh,
        original_tokens=original_tokens,
    )

def distributed_msa_pair_weighted_average(
    pair_logits_local: torch.Tensor,
    value_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_tokens: int | None = None,
) -> torch.Tensor:
    """Distributed exact core for MSAPairWeightedAveraging.

    pair_logits_local is sharded on [output-token, source-token] over the 2D CP
    mesh. The OpenDDE serial module normalizes logits over the source-token
    axis, so the softmax max/sum reductions happen over mesh.group_row. The
    weighted value sum over source-token shards also reduces over mesh.group_row.
    """

    if pair_logits_local.ndim != 4:
        raise ValueError("pair_logits_local must be [B, I_local, J_local, H].")
    if value_local.ndim != 5:
        raise ValueError("value_local must be [B, S, J_local, H, C].")
    if pair_logits_local.shape[0] != value_local.shape[0]:
        raise ValueError("batch dimensions must match.")
    if pair_logits_local.shape[2] != value_local.shape[2]:
        raise ValueError("source-token local dimensions must match.")
    if pair_logits_local.shape[3] != value_local.shape[3]:
        raise ValueError("head dimensions must match.")

    use_gather_exact = torch.are_deterministic_algorithms_enabled()
    if use_gather_exact:
        logits_parts = [
            torch.empty_like(pair_logits_local) for _ in range(mesh.layout.shape[1])
        ]
        value_parts = [
            torch.empty_like(value_local) for _ in range(mesh.layout.shape[1])
        ]
        dist.all_gather(
            logits_parts,
            pair_logits_local.contiguous(),
            group=mesh.group_row,
        )
        dist.all_gather(
            value_parts,
            value_local.contiguous(),
            group=mesh.group_row,
        )
        pair_logits = torch.cat(logits_parts, dim=2)
        value = torch.cat(value_parts, dim=2)
        if original_tokens is not None:
            pair_logits = pair_logits[:, :, :original_tokens, :]
            value = value[:, :, :original_tokens, :, :]
        return serial_msa_pair_weighted_average(pair_logits, value).contiguous()

    if original_tokens is not None:
        source_tile = pair_logits_local.shape[2]
        source_start = mesh.coord[1] * source_tile
        valid_sources = max(0, min(source_tile, int(original_tokens) - source_start))
        if valid_sources < source_tile:
            pair_logits_local = pair_logits_local.clone()
            pair_logits_local[:, :, valid_sources:, :] = -torch.inf

    local_amax = pair_logits_local.amax(dim=2)
    global_amax = local_amax.clone()
    dist.all_reduce(global_amax, op=dist.ReduceOp.MAX, group=mesh.group_row)

    exp_logits = torch.exp(pair_logits_local - global_amax.unsqueeze(2))
    local_denom = exp_logits.sum(dim=2)
    global_denom = local_denom.clone()
    dist.all_reduce(global_denom, op=dist.ReduceOp.SUM, group=mesh.group_row)
    weights_local = exp_logits / global_denom.unsqueeze(2)

    partial = torch.einsum("bijh,bsjhc->bsihc", weights_local, value_local)
    out = partial.contiguous()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=mesh.group_row)
    return out.contiguous()
