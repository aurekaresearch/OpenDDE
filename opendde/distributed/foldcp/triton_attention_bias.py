# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Bitwise-compatible BF16 bias fusion for Fold-CP triangle attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_attention_biases_kernel(
    scores,
    mask_bias,
    triangle_bias,
    n_elements,
    no_heads,
    n_query,
    n_key,
    key_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    program = tl.program_id(0)
    key_block = program % key_blocks
    outer = program // key_blocks
    key_index = key_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = key_index < n_key
    query_index = outer % n_query
    outer = outer // n_query
    head_index = outer % no_heads
    row_index = outer // no_heads
    score_offsets = (
        (row_index * no_heads + head_index) * n_query + query_index
    ) * n_key + key_index

    score = tl.load(scores + score_offsets, mask=active, other=0.0)
    row_bias = tl.load(
        mask_bias + row_index * n_key + key_index,
        mask=active,
        other=0.0,
    )
    head_bias = tl.load(
        triangle_bias + (head_index * n_query + query_index) * n_key + key_index,
        mask=active,
        other=0.0,
    )
    # Eager torch rounds the BF16 score after each in-place addition. Keep the
    # same two rounding points while eliminating the intermediate global write.
    first_sum = tl.cast(score + row_bias, tl.bfloat16)
    final_sum = tl.cast(first_sum.to(tl.float32) + head_bias, tl.bfloat16)
    tl.store(scores + score_offsets, final_sum, mask=active)


def add_attention_biases_inplace(
    scores: torch.Tensor,
    mask_bias: torch.Tensor,
    triangle_bias: torch.Tensor,
) -> torch.Tensor:
    """Apply the two standard triangle-attention biases with eager BF16 parity."""

    if (
        scores.device.type != "cuda"
        or scores.dtype != torch.bfloat16
        or scores.ndim != 4
        or not scores.is_contiguous()
    ):
        raise ValueError("fused attention bias requires contiguous CUDA BF16 scores")
    rows, no_heads, n_query, n_key = scores.shape
    if mask_bias.shape != (rows, 1, 1, n_key):
        raise ValueError("mask bias must have shape [rows, 1, 1, key]")
    if triangle_bias.shape != (1, no_heads, n_query, n_key):
        raise ValueError("triangle bias must have shape [1, heads, query, key]")
    if not mask_bias.is_contiguous() or not triangle_bias.is_contiguous():
        raise ValueError("fused attention biases must be contiguous")

    elements = scores.numel()
    key_blocks = triton.cdiv(n_key, 512)
    outer_rows = rows * no_heads * n_query
    _add_attention_biases_kernel[(outer_rows * key_blocks,)](
        scores,
        mask_bias,
        triangle_bias,
        elements,
        no_heads,
        n_query,
        n_key,
        key_blocks,
        BLOCK_SIZE=512,
    )
    return scores
