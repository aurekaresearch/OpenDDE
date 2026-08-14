# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP distributed triangle multiplication core."""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import torch

from opendde.distributed.foldcp.comm import Ring2DComm


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
    )

    if permute_out is not None:
        out = out.permute(permute_out)
    return out.contiguous()


def _distributed_bmm_double_buffered(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    ring: Ring2DComm,
    *,
    transpose_arg: Optional[_TransposeArg],
) -> torch.Tensor:
    """Fold-CP Cannon/ring BMM with boltz-cp style double buffering."""

    lhs = lhs.contiguous()
    rhs = rhs.contiguous()

    if transpose_arg == _TransposeArg.LHS:
        lhs_recv = ring.comm_2d_trans.enqueue_to_dispatch(lhs)
        rhs_recv = rhs
        lhs_next = torch.empty_like(lhs_recv)
        rhs_next = torch.empty_like(rhs_recv)
        ring.comm_2d_trans.wait_until_finished()
    elif transpose_arg == _TransposeArg.RHS:
        lhs_recv = lhs
        rhs_recv = ring.comm_2d_trans.enqueue_to_dispatch(rhs)
        lhs_next = torch.empty_like(lhs_recv)
        rhs_next = torch.empty_like(rhs_recv)
        ring.comm_2d_trans.wait_until_finished()
    elif transpose_arg is None:
        lhs_recv = lhs
        rhs_recv = rhs
        lhs_next = torch.empty_like(lhs_recv)
        rhs_next = torch.empty_like(rhs_recv)
    else:
        raise ValueError(f"invalid transpose_arg={transpose_arg}")

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

    lhs_ready = lhs_buffer[ready_index]
    rhs_ready = rhs_buffer[ready_index]
    out = lhs_ready.new_zeros((*lhs_ready.shape[:-1], rhs_ready.shape[-1]))

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
        out.add_(torch.matmul(lhs_ready, rhs_ready))
        if step < ring.layout.shape[1] - 1:
            ring.comm_row.wait_until_finished()
            ring.comm_col.wait_until_finished()
            ready_index ^= 1
            recv_index ^= 1

    return out


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
    out = lhs.new_zeros((*lhs.shape[:-2], lhs.shape[-2], rhs.shape[-1]))
    n_row = lhs.shape[-2]
    n_col = rhs.shape[-1]

    for row_start in range(0, n_row, row_block_size):
        row_end = min(row_start + row_block_size, n_row)
        lhs_block_base = lhs[..., row_start:row_end, :].contiguous()
        for col_start in range(0, n_col, col_block_size):
            col_end = min(col_start + col_block_size, n_col)
            rhs_block_base = rhs[..., :, col_start:col_end].contiguous()

            if transpose_arg == _TransposeArg.LHS:
                lhs_ready = ring.comm_2d_trans.exchange(lhs_block_base)
                rhs_ready = rhs_block_base
            elif transpose_arg == _TransposeArg.RHS:
                lhs_ready = lhs_block_base
                rhs_ready = ring.comm_2d_trans.exchange(rhs_block_base)
            elif transpose_arg is None:
                lhs_ready = lhs_block_base
                rhs_ready = rhs_block_base
            else:
                raise ValueError(f"invalid transpose_arg={transpose_arg}")

            lhs_ready = ring.comm_row_init.exchange(lhs_ready)
            rhs_ready = ring.comm_col_init.exchange(rhs_ready)
            out_block = out[..., row_start:row_end, col_start:col_end]

            for step in range(ring.layout.shape[1]):
                out_block.add_(torch.matmul(lhs_ready, rhs_ready))
                if step < ring.layout.shape[1] - 1:
                    lhs_ready = ring.comm_row.exchange(lhs_ready)
                    rhs_ready = ring.comm_col.exchange(rhs_ready)

    if permute_out is not None:
        out = out.permute(permute_out)
    return out.contiguous()


def distributed_triangle_multiplication(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    ring: Ring2DComm,
    direction: TriangleMultiplicationDirection | str,
) -> torch.Tensor:
    """Compute local triangle multiplication tile from local projected a/b tiles."""

    direction = TriangleMultiplicationDirection(direction)
    if a_local.shape != b_local.shape:
        raise ValueError("a_local and b_local must have the same shape.")
    if a_local.ndim != 4:
        raise ValueError("triangle multiplication expects [B, N_local, N_local, C].")

    channel_chunk_size = int(os.environ.get("OPENDDE_FOLDCP_TRIMUL_CHANNEL_CHUNK", "8"))
    if 0 < channel_chunk_size < a_local.shape[-1]:
        out = torch.empty_like(a_local)
        for channel_start in range(0, a_local.shape[-1], channel_chunk_size):
            channel_end = min(channel_start + channel_chunk_size, a_local.shape[-1])
            channel_slice = slice(channel_start, channel_end)
            out[..., channel_slice] = _distributed_triangle_multiplication_no_chunk(
                a_local[..., channel_slice],
                b_local[..., channel_slice],
                ring,
                direction,
            )
        return out.contiguous()

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
