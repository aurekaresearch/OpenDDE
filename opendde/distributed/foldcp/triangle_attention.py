# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP ring attention reference implementation."""

from __future__ import annotations

import torch

from opendde.distributed.foldcp.comm import Ring2DComm
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

    k_ready = k_local
    v_ready = v_local
    bias_ready = bias_local

    out = None
    lse = None
    amax = None
    for step in range(ring.layout.shape[1]):
        block_out, block_lse, block_amax = attention_block(
            q_local,
            k_ready,
            v_ready,
            bias_ready,
        )
        out, lse, amax = online_softmax_update(
            block_out, block_lse, block_amax, out, lse, amax
        )
        if step < ring.layout.shape[1] - 1:
            k_ready = ring.comm_row.exchange(k_ready)
            v_ready = ring.comm_row.exchange(v_ready)
            if bias_ready is not None:
                bias_ready = ring.comm_row.exchange(bias_ready)
    if out is None:
        raise RuntimeError("ring attention did not process any blocks.")
    return out.contiguous()


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

    q_local = z_local
    k_ready = z_local
    v_ready = z_local
    out = None
    lse = None
    amax = None
    for step in range(ring.layout.shape[1]):
        block_out, block_lse, block_amax = _triangle_attention_block(
            q_local,
            k_ready,
            v_ready,
        )
        out, lse, amax = online_softmax_update(
            block_out,
            block_lse,
            block_amax,
            out,
            lse,
            amax,
        )
        if step < ring.layout.shape[1] - 1:
            k_ready = ring.comm_row.exchange(k_ready)
            v_ready = ring.comm_row.exchange(v_ready)
    if out is None:
        raise RuntimeError("triangle attention did not process any blocks.")
    return out.contiguous()


def distributed_triangle_attention_ending(
    z_local: torch.Tensor,
    ring: Ring2DComm,
) -> torch.Tensor:
    """Triangle attention over each column via transpose-starting-transpose."""

    z_t_local = ring.comm_2d_trans.exchange(z_local.transpose(1, 2).contiguous())
    out_t_local = distributed_triangle_attention_starting(z_t_local, ring)
    return ring.comm_2d_trans.exchange(out_t_local.transpose(1, 2).contiguous())


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
