# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP distributed triangle multiplication core."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import torch

from opendde.distributed.foldcp.comm import (
    Ring2DComm,
    detach_rank_local_error_traceback,
    run_group_rank_action_synchronized,
)


class TriangleMultiplicationDirection(str, Enum):
    OUTGOING = "outgoing"
    INCOMING = "incoming"


class _TransposeArg(str, Enum):
    LHS = "lhs"
    RHS = "rhs"


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def _bmm_block_sizes(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[int, int]:
    block_size = _positive_int_env("OPENDDE_FOLDCP_TRIMUL_BMM_BLOCK_SIZE", 0)
    row_block_size = _positive_int_env(
        "OPENDDE_FOLDCP_TRIMUL_BMM_ROW_BLOCK_SIZE",
        block_size,
    )
    col_block_size = _positive_int_env(
        "OPENDDE_FOLDCP_TRIMUL_BMM_COL_BLOCK_SIZE",
        block_size,
    )
    if row_block_size <= 0 or row_block_size >= lhs.shape[-2]:
        row_block_size = lhs.shape[-2]
    if col_block_size <= 0 or col_block_size >= rhs.shape[-1]:
        col_block_size = rhs.shape[-1]
    return row_block_size, col_block_size


def _distributed_bmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    ring: Ring2DComm,
    *,
    permute_lhs: Optional[tuple[int, ...]],
    permute_rhs: Optional[tuple[int, ...]],
    permute_out: Optional[tuple[int, ...]],
    transpose_arg: Optional[_TransposeArg],
) -> torch.Tensor:
    if permute_lhs is not None:
        lhs = lhs.permute(permute_lhs)
    if permute_rhs is not None:
        rhs = rhs.permute(permute_rhs)

    row_block_size, col_block_size = _bmm_block_sizes(lhs, rhs)
    if row_block_size < lhs.shape[-2] or col_block_size < rhs.shape[-1]:
        return _distributed_bmm_streamed(
            lhs,
            rhs,
            ring,
            row_block_size=row_block_size,
            col_block_size=col_block_size,
            permute_out=permute_out,
            transpose_arg=transpose_arg,
        )

    out = _distributed_bmm_double_buffered(
        lhs,
        rhs,
        ring,
        transpose_arg=transpose_arg,
        permute_out=permute_out,
    )
    return out


def _distributed_bmm_double_buffered(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    ring: Ring2DComm,
    *,
    transpose_arg: Optional[_TransposeArg],
    permute_out: Optional[tuple[int, ...]] = None,
) -> torch.Tensor:
    """Fold-CP Cannon/ring BMM with boltz-cp style double buffering."""

    def _allocate_bmm_buffers() -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        lhs_contiguous = lhs.contiguous()
        rhs_contiguous = rhs.contiguous()
        if transpose_arg == _TransposeArg.LHS:
            lhs_recv = torch.empty_like(lhs_contiguous)
            rhs_recv = rhs_contiguous
        elif transpose_arg == _TransposeArg.RHS:
            lhs_recv = lhs_contiguous
            rhs_recv = torch.empty_like(rhs_contiguous)
        elif transpose_arg is None:
            lhs_recv = lhs_contiguous
            rhs_recv = rhs_contiguous
        else:
            raise ValueError(f"invalid transpose_arg={transpose_arg}")
        lhs_next = torch.empty_like(lhs_recv)
        rhs_next = torch.empty_like(rhs_recv)
        out = lhs_recv.new_zeros((*lhs_recv.shape[:-1], rhs_recv.shape[-1]))
        final_out = (
            torch.empty(
                tuple(out.shape[index] for index in permute_out),
                dtype=out.dtype,
                device=out.device,
            )
            if permute_out is not None
            else out
        )
        return (
            lhs_contiguous,
            rhs_contiguous,
            lhs_recv,
            rhs_recv,
            lhs_next,
            rhs_next,
            out,
            final_out,
        )

    buffers = run_group_rank_action_synchronized(
        _allocate_bmm_buffers,
        group=ring.group_2d,
        description="triangle-multiplication ring-buffer allocation",
    )
    if buffers is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("triangle-multiplication allocation returned no buffers.")
    lhs, rhs, lhs_recv, rhs_recv, lhs_next, rhs_next, out, final_out = buffers

    if transpose_arg == _TransposeArg.LHS:
        lhs_recv = ring.comm_2d_trans.enqueue_to_dispatch(lhs, lhs_recv)
        ring.comm_2d_trans.wait_until_finished()
    elif transpose_arg == _TransposeArg.RHS:
        rhs_recv = ring.comm_2d_trans.enqueue_to_dispatch(rhs, rhs_recv)
        ring.comm_2d_trans.wait_until_finished()

    lhs_buffer = [lhs_recv, lhs_next]
    rhs_buffer = [rhs_recv, rhs_next]
    ready_index = 0
    recv_index = 1

    lhs_buffer[recv_index] = ring.comm_row_init.enqueue_to_dispatch(
        lhs_buffer[ready_index],
        lhs_buffer[recv_index],
    )
    rhs_buffer[recv_index] = ring.comm_col_init.enqueue_to_dispatch(
        rhs_buffer[ready_index],
        rhs_buffer[recv_index],
    )
    ready_index ^= 1
    recv_index ^= 1

    ring.comm_row_init.wait_until_finished()
    ring.comm_col_init.wait_until_finished()

    compute_error: Exception | None = None
    for step in range(ring.layout.shape[1]):
        lhs_ready = lhs_buffer[ready_index]
        rhs_ready = rhs_buffer[ready_index]
        if step < ring.layout.shape[1] - 1:
            lhs_buffer[recv_index] = ring.comm_row.enqueue_to_dispatch(
                lhs_ready,
                lhs_buffer[recv_index],
            )
            rhs_buffer[recv_index] = ring.comm_col.enqueue_to_dispatch(
                rhs_ready,
                rhs_buffer[recv_index],
            )
        if compute_error is None:
            try:
                out.add_(torch.matmul(lhs_ready, rhs_ready))
            except Exception as exc:
                # The matching async sends/receives have already been queued.
                # Drain this and every remaining ring step before reporting the
                # local compute/allocation failure to the full CP group.
                compute_error = detach_rank_local_error_traceback(exc)
        if step < ring.layout.shape[1] - 1:
            ring.comm_row.wait_until_finished()
            ring.comm_col.wait_until_finished()
            ready_index ^= 1
            recv_index ^= 1

    def _finish_bmm() -> torch.Tensor:
        if compute_error is not None:
            raise compute_error
        if permute_out is not None:
            final_out.copy_(out.permute(permute_out))
        return final_out

    result = run_group_rank_action_synchronized(
        _finish_bmm,
        group=ring.group_2d,
        description="triangle-multiplication ring compute",
    )
    if result is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("triangle-multiplication ring returned no result.")
    return result


def _distributed_bmm_streamed(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    ring: Ring2DComm,
    *,
    row_block_size: int,
    col_block_size: int,
    permute_out: Optional[tuple[int, ...]],
    transpose_arg: Optional[_TransposeArg],
) -> torch.Tensor:
    output_buffers = run_group_rank_action_synchronized(
        lambda: (
            lhs.new_zeros((*lhs.shape[:-2], lhs.shape[-2], rhs.shape[-1])),
            (
                lhs.new_empty(
                    tuple(
                        (*lhs.shape[:-2], lhs.shape[-2], rhs.shape[-1])[index]
                        for index in permute_out
                    )
                )
                if permute_out is not None
                else None
            ),
        ),
        group=ring.group_2d,
        description="streamed triangle-multiplication output allocation",
    )
    if output_buffers is None:  # pragma: no cover - action always runs on every rank
        raise RuntimeError("streamed triangle-multiplication returned no output.")
    out, final_out = output_buffers
    n_row = lhs.shape[-2]
    n_col = rhs.shape[-1]
    completed_out: torch.Tensor | None = None

    for row_start in range(0, n_row, row_block_size):
        row_end = min(row_start + row_block_size, n_row)
        for col_start in range(0, n_col, col_block_size):
            col_end = min(col_start + col_block_size, n_col)

            def _allocate_streamed_block_buffers():
                lhs_block = lhs[..., row_start:row_end, :].contiguous()
                rhs_block = rhs[..., :, col_start:col_end].contiguous()
                lhs_transpose_recv = (
                    torch.empty_like(lhs_block)
                    if transpose_arg == _TransposeArg.LHS
                    else lhs_block
                )
                rhs_transpose_recv = (
                    torch.empty_like(rhs_block)
                    if transpose_arg == _TransposeArg.RHS
                    else rhs_block
                )
                return (
                    lhs_block,
                    rhs_block,
                    lhs_transpose_recv,
                    rhs_transpose_recv,
                    torch.empty_like(lhs_block),
                    torch.empty_like(rhs_block),
                )

            buffers = run_group_rank_action_synchronized(
                _allocate_streamed_block_buffers,
                group=ring.group_2d,
                description="streamed triangle-multiplication block allocation",
            )
            if buffers is None:  # pragma: no cover
                raise RuntimeError(
                    "streamed triangle-multiplication returned no block buffers."
                )
            (
                lhs_block_base,
                rhs_block_base,
                lhs_transpose_recv,
                rhs_transpose_recv,
                lhs_next,
                rhs_next,
            ) = buffers

            if transpose_arg == _TransposeArg.LHS:
                lhs_ready = ring.comm_2d_trans.exchange(
                    lhs_block_base, to_recv=lhs_transpose_recv
                )
                rhs_ready = rhs_block_base
            elif transpose_arg == _TransposeArg.RHS:
                lhs_ready = lhs_block_base
                rhs_ready = ring.comm_2d_trans.exchange(
                    rhs_block_base, to_recv=rhs_transpose_recv
                )
            elif transpose_arg is None:
                lhs_ready = lhs_block_base
                rhs_ready = rhs_block_base
            else:
                raise ValueError(f"invalid transpose_arg={transpose_arg}")

            lhs_buffer = [lhs_ready, lhs_next]
            rhs_buffer = [rhs_ready, rhs_next]
            ready_index = 0
            recv_index = 1
            lhs_buffer[recv_index] = ring.comm_row_init.exchange(
                lhs_buffer[ready_index], to_recv=lhs_buffer[recv_index]
            )
            rhs_buffer[recv_index] = ring.comm_col_init.exchange(
                rhs_buffer[ready_index], to_recv=rhs_buffer[recv_index]
            )
            ready_index ^= 1
            recv_index ^= 1
            out_block = out[..., row_start:row_end, col_start:col_end]

            compute_error: Exception | None = None
            for step in range(ring.layout.shape[1]):
                lhs_ready = lhs_buffer[ready_index]
                rhs_ready = rhs_buffer[ready_index]
                if step < ring.layout.shape[1] - 1:
                    lhs_buffer[recv_index] = ring.comm_row.exchange(
                        lhs_ready, to_recv=lhs_buffer[recv_index]
                    )
                    rhs_buffer[recv_index] = ring.comm_col.exchange(
                        rhs_ready, to_recv=rhs_buffer[recv_index]
                    )
                if compute_error is None:
                    try:
                        out_block.add_(torch.matmul(lhs_ready, rhs_ready))
                    except Exception as exc:
                        compute_error = detach_rank_local_error_traceback(exc)
                if step < ring.layout.shape[1] - 1:
                    ready_index ^= 1
                    recv_index ^= 1

            is_last_block = row_end == n_row and col_end == n_col

            def _finish_streamed_block() -> torch.Tensor | None:
                if compute_error is not None:
                    raise compute_error
                if not is_last_block:
                    return None
                if permute_out is not None:
                    if final_out is None:  # pragma: no cover
                        raise RuntimeError(
                            "streamed triangle-multiplication final output is missing."
                        )
                    final_out.copy_(out.permute(permute_out))
                    return final_out
                return out

            completed_out = run_group_rank_action_synchronized(
                _finish_streamed_block,
                group=ring.group_2d,
                description="streamed triangle-multiplication block compute",
            )
            # Every object below is block-local.  In particular, `buffers` and
            # the two ping-pong lists otherwise keep the completed block's
            # dense operands alive while the next block is allocated, which
            # defeats the capacity purpose of streamed BMM.
            del (
                _finish_streamed_block,
                buffers,
                lhs_block_base,
                rhs_block_base,
                lhs_transpose_recv,
                rhs_transpose_recv,
                lhs_next,
                rhs_next,
                lhs_ready,
                rhs_ready,
                lhs_buffer,
                rhs_buffer,
                out_block,
            )

    if completed_out is None:  # pragma: no cover - the final block returns it
        raise RuntimeError("streamed triangle-multiplication returned no result.")
    return completed_out


def distributed_triangle_multiplication(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    ring: Ring2DComm,
    direction: TriangleMultiplicationDirection | str,
) -> torch.Tensor:
    """Compute local triangle multiplication tile from local projected a/b tiles."""

    def _validate_inputs() -> TriangleMultiplicationDirection:
        normalized_direction = TriangleMultiplicationDirection(direction)
        if a_local.shape != b_local.shape:
            raise ValueError("a_local and b_local must have the same shape.")
        if a_local.ndim != 4:
            raise ValueError(
                "triangle multiplication expects [B, N_local, N_local, C]."
            )
        return normalized_direction

    validated_direction = run_group_rank_action_synchronized(
        _validate_inputs,
        group=ring.group_2d,
        description="triangle-multiplication input validation",
    )
    if validated_direction is None:  # pragma: no cover
        raise RuntimeError(
            "triangle-multiplication input validation returned no result."
        )
    direction = validated_direction

    channel_chunk_size = int(os.environ.get("OPENDDE_FOLDCP_TRIMUL_CHANNEL_CHUNK", "8"))
    if 0 < channel_chunk_size < a_local.shape[-1]:
        out = run_group_rank_action_synchronized(
            lambda: torch.empty_like(a_local),
            group=ring.group_2d,
            description="triangle-multiplication channel output allocation",
        )
        if out is None:  # pragma: no cover
            raise RuntimeError(
                "triangle-multiplication channel output allocation returned no result."
            )
        for channel_start in range(0, a_local.shape[-1], channel_chunk_size):
            channel_end = min(channel_start + channel_chunk_size, a_local.shape[-1])
            channel_slice = slice(channel_start, channel_end)
            channel_update = _distributed_triangle_multiplication_no_chunk(
                a_local[..., channel_slice],
                b_local[..., channel_slice],
                ring,
                direction,
            )

            def _store_channel_update() -> None:
                out[..., channel_slice].copy_(channel_update)

            run_group_rank_action_synchronized(
                _store_channel_update,
                group=ring.group_2d,
                description="triangle-multiplication channel output copy",
            )
            del channel_update

        result = run_group_rank_action_synchronized(
            out.contiguous,
            group=ring.group_2d,
            description="triangle-multiplication channel output finalization",
        )
        if result is None:  # pragma: no cover - action always runs on every rank
            raise RuntimeError(
                "triangle-multiplication channel output finalization returned no result."
            )
        return result

    return _distributed_triangle_multiplication_no_chunk(
        a_local, b_local, ring, direction
    )


def _distributed_triangle_multiplication_no_chunk(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    ring: Ring2DComm,
    direction: TriangleMultiplicationDirection,
) -> torch.Tensor:
    if direction == TriangleMultiplicationDirection.OUTGOING:
        return _distributed_bmm(
            a_local,
            b_local,
            ring,
            permute_lhs=(0, 3, 1, 2),
            permute_rhs=(0, 3, 2, 1),
            permute_out=(0, 2, 3, 1),
            transpose_arg=_TransposeArg.RHS,
        )
    if direction == TriangleMultiplicationDirection.INCOMING:
        return _distributed_bmm(
            a_local,
            b_local,
            ring,
            permute_lhs=(0, 3, 2, 1),
            permute_rhs=(0, 3, 1, 2),
            permute_out=(0, 2, 3, 1),
            transpose_arg=_TransposeArg.LHS,
        )
    raise ValueError(f"unsupported direction={direction}")


def serial_triangle_multiplication(
    a: torch.Tensor,
    b: torch.Tensor,
    direction: TriangleMultiplicationDirection | str,
) -> torch.Tensor:
    direction = TriangleMultiplicationDirection(direction)
    if direction == TriangleMultiplicationDirection.OUTGOING:
        return torch.einsum("bnkd,bmkd->bnmd", a, b)
    if direction == TriangleMultiplicationDirection.INCOMING:
        return torch.einsum("bknd,bkmd->bnmd", a, b)
    raise ValueError(f"unsupported direction={direction}")
