# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Numerically stable online softmax accumulation used by Fold-CP attention."""

from __future__ import annotations

import torch


def online_softmax_update(
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    block_amax: torch.Tensor,
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    amax: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge one attention block into the running online-softmax state.

    `block_out` must already be weighted by the block-local softmax. `block_lse`
    is log(sum(exp(logits - block_amax))) for the block.
    """

    if out is None or lse is None or amax is None:
        return block_out, block_lse, block_amax

    new_amax = torch.maximum(amax, block_amax)
    prev_scale = torch.exp(amax + lse - new_amax)
    block_scale = torch.exp(block_amax + block_lse - new_amax)
    denom = prev_scale + block_scale
    new_out = (
        out * prev_scale.unsqueeze(-1) + block_out * block_scale.unsqueeze(-1)
    ) / denom.unsqueeze(-1)
    new_lse = torch.log(denom)
    return new_out, new_lse, new_amax


def attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute local block attention and return online-softmax summary tensors."""

    logits = torch.matmul(q, k.transpose(-1, -2))
    if bias is not None:
        if isinstance(bias, (list, tuple)):
            for item in bias:
                logits.add_(item)
        else:
            logits.add_(bias)
    block_amax = logits.amax(dim=-1)
    logits.sub_(block_amax.unsqueeze(-1))
    logits.exp_()
    denom = logits.sum(dim=-1)
    block_lse = torch.log(denom)
    logits.div_(denom.unsqueeze(-1))
    block_out = torch.matmul(logits, v)
    return block_out, block_lse, block_amax


def serial_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = torch.matmul(q, k.transpose(-1, -2))
    if bias is not None:
        logits = logits + bias
    weights = torch.softmax(logits, dim=-1)
    return torch.matmul(weights, v)
