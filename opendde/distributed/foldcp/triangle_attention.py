# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP ring attention reference implementation."""

from __future__ import annotations

import torch

from opendde.distributed.foldcp.comm import (
    Ring2DComm,
    exchange_tensor_synchronized,
    run_group_rank_action_synchronized,
)
from opendde.distributed.foldcp.online_softmax import (
    attention_block,
    online_softmax_update,
    serial_attention,
)


def distributed_ring_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    bias_local: torch.Tensor | None,
    ring: Ring2DComm,
) -> torch.Tensor:
    """Compute local query-tile attention while rotating key/value tiles."""

    synchronize_failures = int(ring.layout.numel) > 1
    k_ready = k_local
    v_ready = v_local
    bias_ready = bias_local

    out = None
    lse = None
    amax = None
    for step in range(ring.layout.shape[1]):

        def _compute_attention_block():
            block_out, block_lse, block_amax = attention_block(
                q_local,
                k_ready,
                v_ready,
                bias_ready,
            )
            return online_softmax_update(
                block_out, block_lse, block_amax, out, lse, amax
            )

        completed = (
            run_group_rank_action_synchronized(
                _compute_attention_block,
                group=ring.group_2d,
                description=f"reference ring attention step {step} computation",
            )
            if synchronize_failures
            else _compute_attention_block()
        )
        if completed is None:  # pragma: no cover - every rank runs the action
            raise RuntimeError("reference ring attention returned no step output.")
        out, lse, amax = completed
        if step < ring.layout.shape[1] - 1:
            k_ready = exchange_tensor_synchronized(
                k_ready,
                comm=ring.comm_row,
                group=ring.group_2d,
                description=f"reference ring attention step {step} K exchange",
            )
            v_ready = exchange_tensor_synchronized(
                v_ready,
                comm=ring.comm_row,
                group=ring.group_2d,
                description=f"reference ring attention step {step} V exchange",
            )
            if bias_ready is not None:
                bias_ready = exchange_tensor_synchronized(
                    bias_ready,
                    comm=ring.comm_row,
                    group=ring.group_2d,
                    description=f"reference ring attention step {step} bias exchange",
                )
    if out is None:
        raise RuntimeError("ring attention did not process any blocks.")

    finish = out.contiguous
    if not synchronize_failures:
        return finish()
    result = run_group_rank_action_synchronized(
        finish,
        group=ring.group_2d,
        description="reference ring attention completion",
    )
    if result is None:  # pragma: no cover
        raise RuntimeError("reference ring attention returned no result.")
    return result


def _triangle_attention_block(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.einsum("bijc,bikc->bijk", q_local, k_local)
    block_amax = logits.amax(dim=-1)
    weights_num = torch.exp(logits - block_amax.unsqueeze(-1))
    denom = weights_num.sum(dim=-1)
    block_lse = torch.log(denom)
    weights = weights_num / denom.unsqueeze(-1)
    block_out = torch.einsum("bijk,bikc->bijc", weights, v_local)
    return block_out, block_lse, block_amax


def distributed_triangle_attention_starting(
    z_local: torch.Tensor,
    ring: Ring2DComm,
) -> torch.Tensor:
    """Triangle attention over each row while keeping a local pair tile output."""

    synchronize_failures = int(ring.layout.numel) > 1
    q_local = z_local
    k_ready = z_local
    v_ready = z_local
    out = None
    lse = None
    amax = None
    for step in range(ring.layout.shape[1]):

        def _compute_triangle_block():
            block_out, block_lse, block_amax = _triangle_attention_block(
                q_local,
                k_ready,
                v_ready,
            )
            return online_softmax_update(
                block_out,
                block_lse,
                block_amax,
                out,
                lse,
                amax,
            )

        completed = (
            run_group_rank_action_synchronized(
                _compute_triangle_block,
                group=ring.group_2d,
                description=(
                    f"reference starting triangle attention step {step} computation"
                ),
            )
            if synchronize_failures
            else _compute_triangle_block()
        )
        if completed is None:  # pragma: no cover
            raise RuntimeError("reference triangle attention returned no step output.")
        out, lse, amax = completed
        if step < ring.layout.shape[1] - 1:
            k_ready = exchange_tensor_synchronized(
                k_ready,
                comm=ring.comm_row,
                group=ring.group_2d,
                description=(
                    f"reference starting triangle attention step {step} K exchange"
                ),
            )
            v_ready = exchange_tensor_synchronized(
                v_ready,
                comm=ring.comm_row,
                group=ring.group_2d,
                description=(
                    f"reference starting triangle attention step {step} V exchange"
                ),
            )
    if out is None:
        raise RuntimeError("triangle attention did not process any blocks.")

    finish = out.contiguous
    if not synchronize_failures:
        return finish()
    result = run_group_rank_action_synchronized(
        finish,
        group=ring.group_2d,
        description="reference starting triangle attention completion",
    )
    if result is None:  # pragma: no cover
        raise RuntimeError("reference triangle attention returned no result.")
    return result


def distributed_triangle_attention_ending(
    z_local: torch.Tensor,
    ring: Ring2DComm,
) -> torch.Tensor:
    """Triangle attention over each column via transpose-starting-transpose."""

    z_t_local = exchange_tensor_synchronized(
        z_local,
        comm=ring.comm_2d_trans,
        group=ring.group_2d,
        description="reference ending triangle attention input transpose",
        prepare=lambda tensor: tensor.transpose(1, 2),
    )
    out_t_local = distributed_triangle_attention_starting(z_t_local, ring)
    return exchange_tensor_synchronized(
        out_t_local,
        comm=ring.comm_2d_trans,
        group=ring.group_2d,
        description="reference ending triangle attention output transpose",
        prepare=lambda tensor: tensor.transpose(1, 2),
    )


def serial_triangle_attention_starting(z: torch.Tensor) -> torch.Tensor:
    logits = torch.einsum("bijc,bikc->bijk", z, z)
    weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bijk,bikc->bijc", weights, z)


def serial_triangle_attention_ending(z: torch.Tensor) -> torch.Tensor:
    out_t = serial_triangle_attention_starting(z.transpose(1, 2).contiguous())
    return out_t.transpose(1, 2).contiguous()


__all__ = [
    "distributed_ring_attention",
    "distributed_triangle_attention_ending",
    "distributed_triangle_attention_starting",
    "serial_attention",
    "serial_triangle_attention_ending",
    "serial_triangle_attention_starting",
]
