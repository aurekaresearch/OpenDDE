# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP implementation of MSA pair weighted averaging core."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.comm import (
    detach_rank_local_error_traceback,
    dispatch_p2p_batch_and_wait,
    gather_tensor_by_ring,
    run_group_rank_action_synchronized,
)
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


def shard_msa_value_by_token(
    value: torch.Tensor, mesh: FoldCPProcessMesh
) -> torch.Tensor:
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

    # The maintained topology is 1 x P: the source-token reduction runs over
    # all P columns while the single mesh row already owns every output-token
    # row. Its column group therefore contains only this rank. Going through
    # all_gather + cat would duplicate the complete MSA output for no data
    # movement and can create a large, avoidable capacity spike.
    if int(mesh.layout.shape[0]) == 1:
        normalized_dim = token_dim if token_dim >= 0 else local_output.ndim + token_dim
        if normalized_dim < 0 or normalized_dim >= local_output.ndim:
            raise IndexError(f"MSA token dim {normalized_dim} is out of range.")
        if original_tokens is None or int(original_tokens) >= int(
            local_output.shape[normalized_dim]
        ):
            return local_output
        return local_output.narrow(
            normalized_dim,
            0,
            max(0, int(original_tokens)),
        ).contiguous()

    def _allocate_gather_buffers() -> tuple[torch.Tensor, list[torch.Tensor]]:
        send = local_output.contiguous()
        return send, [torch.empty_like(send) for _ in range(mesh.layout.shape[0])]

    buffers = run_group_rank_action_synchronized(
        _allocate_gather_buffers,
        group=mesh.group_col,
        description="MSA row gather allocation",
    )
    if buffers is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("MSA row gather allocation returned no buffers.")
    send, gathered = buffers
    dist.all_gather(gathered, send, group=mesh.group_col)

    def _assemble_rows() -> torch.Tensor:
        output = torch.cat(gathered, dim=token_dim)
        if original_tokens is not None:
            target = [slice(None)] * output.dim()
            target[token_dim] = slice(0, original_tokens)
            output = output[tuple(target)]
        return output.contiguous()

    result = run_group_rank_action_synchronized(
        _assemble_rows,
        group=mesh.group_col,
        description="MSA row gather assembly",
    )
    if result is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("MSA row gather assembly returned no result.")
    return result


def collect_msa_pair_row_slab(
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_tokens: int,
) -> torch.Tensor:
    """Collect this rank's local output rows over all source-token columns.

    This is a row-ring all-gather specialized for pair tiles. It materializes
    only the current rank's row slab, not a full pair tensor.
    """

    def _allocate_ring_buffers() -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        int,
        int,
        tuple[tuple[int, int, int], ...],
    ]:
        n_cols = int(mesh.layout.shape[1])
        if n_cols <= 0:
            raise ValueError("MSA pair-row ring must have at least one column.")
        row = int(mesh.coord[0])
        local_col = int(mesh.coord[1])
        if local_col < 0 or local_col >= n_cols:
            raise ValueError(
                f"MSA pair-row column must be in [0, {n_cols}), got {local_col}."
            )
        col_tile = int(z_local.shape[-2])
        padded_cols = n_cols * col_tile
        row_slab_shape = z_local.shape[:-2] + (
            padded_cols,
            z_local.shape[-1],
        )
        row_slab = z_local.new_empty(row_slab_shape)
        send_tensor = z_local.contiguous()
        recv_tensor = torch.empty_like(send_tensor) if n_cols > 1 else None
        local_target = [slice(None)] * row_slab.dim()
        local_target[-2] = slice(local_col * col_tile, (local_col + 1) * col_tile)
        row_slab[tuple(local_target)] = send_tensor
        peer_rounds = []
        for offset in range(1, n_cols):
            dest_col = (local_col + offset) % n_cols
            src_col = (local_col - offset) % n_cols
            peer_rounds.append(
                (
                    src_col,
                    mesh.cp_global_ranks[mesh.layout.to_linear((row, dest_col))],
                    mesh.cp_global_ranks[mesh.layout.to_linear((row, src_col))],
                )
            )
        return (
            row_slab,
            send_tensor,
            recv_tensor,
            n_cols,
            col_tile,
            tuple(peer_rounds),
        )

    buffers = run_group_rank_action_synchronized(
        _allocate_ring_buffers,
        group=mesh.group_row,
        description="MSA pair-row ring allocation",
    )
    if buffers is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("MSA pair-row ring allocation returned no buffers.")
    row_slab, send_tensor, recv_tensor, n_cols, col_tile, peer_rounds = buffers
    assembly_error: Exception | None = None
    for src_col, dest_rank, src_rank in peer_rounds:
        if recv_tensor is None:  # pragma: no cover - guarded by n_cols > 1
            raise RuntimeError("MSA pair-row ring receive buffer is missing.")
        operations = [
            dist.P2POp(
                dist.isend,
                send_tensor,
                dest_rank,
                group=mesh.group_row,
            ),
            dist.P2POp(
                dist.irecv,
                recv_tensor,
                src_rank,
                group=mesh.group_row,
            ),
        ]
        dispatch_p2p_batch_and_wait(operations)
        if assembly_error is None:
            try:
                target = [slice(None)] * row_slab.dim()
                target[-2] = slice(src_col * col_tile, (src_col + 1) * col_tile)
                row_slab[tuple(target)] = recv_tensor
            except Exception as exc:
                # Receive every scheduled peer tile before reporting a local
                # assembly failure, otherwise a sender can remain blocked.
                assembly_error = detach_rank_local_error_traceback(exc)

    def _finish_ring() -> torch.Tensor:
        if assembly_error is not None:
            raise assembly_error
        return row_slab[..., :original_tokens, :].contiguous()

    result = run_group_rank_action_synchronized(
        _finish_ring,
        group=mesh.group_row,
        description="MSA pair-row ring assembly",
    )
    if result is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("MSA pair-row ring assembly returned no result.")
    return result


def _gather_msa_source_by_ring(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    *,
    original_tokens: int | None,
    description: str,
) -> torch.Tensor:
    """Reconstruct the source-token axis without list[P] plus ``cat``."""

    side = int(mesh.layout.shape[1])
    source_dim = 2
    output_tokens = (
        side * int(local_tensor.shape[source_dim])
        if original_tokens is None
        else int(original_tokens)
    )
    return gather_tensor_by_ring(
        local_tensor,
        comm=mesh.ring_comm().comm_row,
        group=mesh.group_row,
        local_index=int(mesh.coord[1]),
        side=side,
        dim=source_dim,
        length=output_tokens,
        description=description,
    )


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

    if torch.are_deterministic_algorithms_enabled():

        def _validate_exact_gather() -> None:
            if value.ndim != 5:
                raise ValueError("value must be [B, S, N, H, C].")
            if pair_logits_local.ndim != 4:
                raise ValueError("pair_logits_local must be [B, I_local, J_local, H].")
            if pair_logits_local.shape[0] != value.shape[0]:
                raise ValueError("batch dimensions must match.")
            if pair_logits_local.shape[3] != value.shape[3]:
                raise ValueError("head dimensions must match.")

        run_group_rank_action_synchronized(
            _validate_exact_gather,
            group=mesh.group_row,
            description="deterministic MSA logits gather validation",
        )
        pair_logits = _gather_msa_source_by_ring(
            pair_logits_local,
            mesh,
            original_tokens=original_tokens,
            description="deterministic MSA logits Ring gather",
        )

        def _compute_exact_output() -> torch.Tensor:
            exact_value = value
            if original_tokens is not None:
                exact_value = exact_value[:, :, :original_tokens, :, :]
            # Preserve the serial full-row launch geometry for bitwise parity.
            return serial_msa_pair_weighted_average(
                pair_logits, exact_value
            ).contiguous()

        result = run_group_rank_action_synchronized(
            _compute_exact_output,
            group=mesh.group_row,
            description="deterministic MSA weighted-average assembly",
        )
        if result is None:  # pragma: no cover - action always runs on every rank
            raise RuntimeError("deterministic MSA assembly returned no result.")
        return result

    return distributed_msa_pair_weighted_average(
        pair_logits_local,
        value,
        mesh,
        original_tokens=original_tokens,
        _value_is_full=True,
    )


def distributed_msa_pair_weighted_average(
    pair_logits_local: torch.Tensor,
    value_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_tokens: int | None = None,
    *,
    _value_is_full: bool = False,
) -> torch.Tensor:
    """Distributed exact core for MSAPairWeightedAveraging.

    pair_logits_local is sharded on [output-token, source-token] over the 2D CP
    mesh. The OpenDDE serial module normalizes logits over the source-token
    axis, so the softmax max/sum reductions happen over mesh.group_row. The
    weighted value sum over source-token shards also reduces over mesh.group_row.
    """

    use_gather_exact = torch.are_deterministic_algorithms_enabled()
    if use_gather_exact:

        def _validate_exact_gathers() -> None:
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

        run_group_rank_action_synchronized(
            _validate_exact_gathers,
            group=mesh.group_row,
            description="deterministic sharded-MSA gather validation",
        )
        pair_logits = _gather_msa_source_by_ring(
            pair_logits_local,
            mesh,
            original_tokens=original_tokens,
            description="deterministic sharded-MSA logits Ring gather",
        )
        value = _gather_msa_source_by_ring(
            value_local,
            mesh,
            original_tokens=original_tokens,
            description="deterministic sharded-MSA value Ring gather",
        )

        def _compute_exact_output() -> torch.Tensor:
            return serial_msa_pair_weighted_average(pair_logits, value).contiguous()

        result = run_group_rank_action_synchronized(
            _compute_exact_output,
            group=mesh.group_row,
            description="deterministic sharded-MSA weighted-average assembly",
        )
        if result is None:  # pragma: no cover - action always runs on every rank
            raise RuntimeError("deterministic sharded-MSA assembly returned no result.")
        return result

    def _prepare_amax() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pair_logits_local.ndim != 4:
            raise ValueError("pair_logits_local must be [B, I_local, J_local, H].")
        if value_local.ndim != 5:
            expected = "N" if _value_is_full else "J_local"
            raise ValueError(f"value_local must be [B, S, {expected}, H, C].")
        if pair_logits_local.shape[0] != value_local.shape[0]:
            raise ValueError("batch dimensions must match.")
        if pair_logits_local.shape[3] != value_local.shape[3]:
            raise ValueError("head dimensions must match.")
        prepared_value = value_local
        if _value_is_full:
            prepared_value = shard_msa_value_by_token(prepared_value, mesh)
        elif pair_logits_local.shape[2] != prepared_value.shape[2]:
            raise ValueError("source-token local dimensions must match.")
        prepared_logits = pair_logits_local
        if original_tokens is not None:
            source_tile = prepared_logits.shape[2]
            source_start = mesh.coord[1] * source_tile
            valid_sources = max(
                0, min(source_tile, int(original_tokens) - source_start)
            )
            if valid_sources < source_tile:
                prepared_logits = prepared_logits.clone()
                prepared_logits[:, :, valid_sources:, :] = -torch.inf
        return (
            prepared_logits,
            prepared_logits.amax(dim=2).clone(),
            prepared_value,
        )

    prepared = run_group_rank_action_synchronized(
        _prepare_amax,
        group=mesh.group_row,
        description="distributed MSA max-reduction allocation",
    )
    if prepared is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("distributed MSA max stage returned no tensors.")
    pair_logits_local, global_amax, value_local = prepared
    dist.all_reduce(global_amax, op=dist.ReduceOp.MAX, group=mesh.group_row)

    def _prepare_denom() -> tuple[torch.Tensor, torch.Tensor]:
        exp_logits = torch.exp(pair_logits_local - global_amax.unsqueeze(2))
        return exp_logits, exp_logits.sum(dim=2).clone()

    prepared = run_group_rank_action_synchronized(
        _prepare_denom,
        group=mesh.group_row,
        description="distributed MSA denominator-reduction allocation",
    )
    if prepared is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("distributed MSA denominator stage returned no tensors.")
    exp_logits, global_denom = prepared
    dist.all_reduce(global_denom, op=dist.ReduceOp.SUM, group=mesh.group_row)

    def _prepare_output() -> torch.Tensor:
        weights_local = exp_logits / global_denom.unsqueeze(2)
        partial = torch.einsum("bijh,bsjhc->bsihc", weights_local, value_local)
        return partial.contiguous()

    out = run_group_rank_action_synchronized(
        _prepare_output,
        group=mesh.group_row,
        description="distributed MSA output-reduction allocation",
    )
    if out is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("distributed MSA output stage returned no tensor.")
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=mesh.group_row)
    result = run_group_rank_action_synchronized(
        out.contiguous,
        group=mesh.group_row,
        description="distributed MSA output finalization",
    )
    if result is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("distributed MSA output finalization returned no tensor.")
    return result
