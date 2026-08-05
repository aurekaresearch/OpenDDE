# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP adapters for real OpenDDE Pairformer modules."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Iterator

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.layout import FoldCP2DLayout
from opendde.distributed.foldcp.launch import (
    foldcp_linear_with_source_launch_shape,
    foldcp_pair_tile_linear_with_source_chunk_launch,
    foldcp_pair_row_slab_linear_with_source_grid_launch,
)
from opendde.distributed.foldcp.triangular_mult import (
    TriangleMultiplicationDirection,
    distributed_triangle_multiplication,
)
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    gather_pair_tensor,
    gather_pair_tensor_like,
    make_pair_shard_spec,
    shard_pair_tensor,
)
from opendde.model.modules import primitives as _primitives
from opendde.model.modules.primitives import _attention as _single_feature_attention
from opendde.model.triangular.layers import _attention, softmax_no_cast
from opendde.model.utils import permute_final_dims


_TRIATT_BIAS_SOURCE_LAUNCH_MIN_ROWS = 1_048_576
_TRIATT_BIAS_SOURCE_LAUNCH_MAX_ROWS = 1_054_729
_TRIATT_MAX_ATTENTION_HEAD_CHUNK = 2
_TRIATT_ATTENTION_SCORE_BUDGET_BYTES = 3 * 1024**3
_TRIATT_STRUCTURAL_ATTENTION_SCORE_BUDGET_BYTES = 2 * 1024**3
_ONE_BY_P_TRIMUL_FULL_B_MAX_BYTES = 6 * 1024**3
_ONE_BY_P_TRIMUL_REPLICATED_SERIAL_MAX_BYTES = 1024**3
_ONE_BY_P_TRIMUL_LHS_RING_MAX_BYTES = 2 * 1024**3
_ONE_BY_P_TRIMUL_LHS_RING_ROW_BUFFER_BYTES = 256 * 1024**2
_ONE_BY_P_TRIMUL_B_OFFLOAD_MIN_BYTES = 2 * 1024**3
_ONE_BY_P_ENDING_TRANSPOSE_BUFFER_BYTES = 512 * 1024**2
_TRIATT_CANONICAL_BATCH_SCOPE_ENABLED: ContextVar[bool] = ContextVar(
    "foldcp_triatt_canonical_batch_scope_enabled",
    default=True,
)


def _foldcp_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    biases: list[torch.Tensor],
) -> torch.Tensor:
    """Run torch attention with an exact BF16 two-bias CUDA fast path."""

    can_fuse_biases = (
        query.device.type == "cuda"
        and query.dtype == torch.bfloat16
        and len(biases) == 2
        and query.ndim == 4
        and key.ndim == 4
        and value.ndim == 4
    )
    if not can_fuse_biases:
        return _attention(query, key, value, biases)

    key_transposed = permute_final_dims(key, (1, 0))
    scores = torch.matmul(query, key_transposed)
    mask_bias, triangle_bias = biases
    expected_mask_shape = (scores.shape[0], 1, 1, scores.shape[-1])
    expected_triangle_shape = (
        1,
        scores.shape[1],
        scores.shape[-2],
        scores.shape[-1],
    )
    if (
        mask_bias.shape != expected_mask_shape
        or triangle_bias.shape != expected_triangle_shape
        or not mask_bias.is_contiguous()
        or not triangle_bias.is_contiguous()
    ):
        return _attention(query, key, value, biases)

    from opendde.distributed.foldcp.triton_attention_bias import (
        add_attention_biases_inplace,
    )

    add_attention_biases_inplace(scores, mask_bias, triangle_bias)
    scores = softmax_no_cast(scores, -1)
    return torch.matmul(scores, value)


def _triatt_attention_score_budget_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIATT_ATTENTION_SCORE_BUDGET_BYTES")
    budget = (
        _TRIATT_ATTENTION_SCORE_BUDGET_BYTES if value is None else max(0, int(value))
    )
    if not _TRIATT_CANONICAL_BATCH_SCOPE_ENABLED.get():
        budget = min(
            budget,
            _TRIATT_STRUCTURAL_ATTENTION_SCORE_BUDGET_BYTES,
        )
    return budget


def _triatt_attention_head_ranges(
    no_heads: int,
    *,
    score_bytes_per_head: int | None = None,
) -> tuple[tuple[int, int], ...]:
    no_heads = int(no_heads)
    if no_heads <= 0:
        return ()
    head_chunk = int(_TRIATT_MAX_ATTENTION_HEAD_CHUNK)
    if score_bytes_per_head is not None and int(score_bytes_per_head) > 0:
        budget_heads = max(
            1,
            _triatt_attention_score_budget_bytes() // int(score_bytes_per_head),
        )
        head_chunk = min(head_chunk, budget_heads)
    if head_chunk <= 0 or head_chunk >= no_heads:
        return ((0, no_heads),)
    return tuple(
        (head_start, min(head_start + head_chunk, no_heads))
        for head_start in range(0, no_heads, head_chunk)
    )


def _trimul_replicated_serial_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIMUL_REPLICATED_SERIAL_MAX_BYTES")
    if value is None:
        return _ONE_BY_P_TRIMUL_REPLICATED_SERIAL_MAX_BYTES
    return max(0, int(value))


def _trimul_should_use_replicated_serial(
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
) -> bool:
    """Use the source TriMul when its complete input has a bounded footprint."""

    if int(mesh.layout.shape[0]) != 1 or int(mesh.layout.shape[1]) <= 1:
        return False
    max_bytes = _trimul_replicated_serial_max_bytes()
    if max_bytes <= 0:
        return False
    full_shape = list(z_local.shape)
    row_dim, col_dim = z_spec.pair_dims
    full_shape[row_dim] = z_spec.original_shape[row_dim]
    full_shape[col_dim] = z_spec.original_shape[col_dim]
    full_elements = 1
    for size in full_shape:
        full_elements *= int(size)
    return full_elements * int(z_local.element_size()) <= max_bytes


def _triatt_bias_source_launch_boundary(source_rows: int) -> bool:
    source_rows = int(source_rows)
    return (
        _TRIATT_BIAS_SOURCE_LAUNCH_MIN_ROWS
        <= source_rows
        <= _TRIATT_BIAS_SOURCE_LAUNCH_MAX_ROWS
    )


def _triatt_query_pad_size(valid_query: int) -> int:
    min_query = 128 if valid_query <= 64 else 192
    return ((max(valid_query, min_query) + 15) // 16) * 16


def _triatt_qkv_row_pad_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 512:
        return original_n
    return ((valid_rows + 15) // 16) * 16


def _triatt_attention_row_chunk_size(
    valid_rows: int,
    original_n: int,
    serial_chunk_size: int | None = None,
    *,
    valid_query: int | None = None,
    element_size: int = 2,
) -> int:
    if original_n <= 1024:
        row_chunk_size = valid_rows
    elif valid_query is not None and int(valid_query) > 0:
        query_pad = _triatt_query_pad_size(int(valid_query))
        score_bytes_per_row = (
            int(query_pad)
            * int(original_n)
            * int(element_size)
            * 2
            * int(_TRIATT_MAX_ATTENTION_HEAD_CHUNK)
        )
        budget_rows = max(
            1,
            _triatt_attention_score_budget_bytes() // int(score_bytes_per_row),
        )
        row_chunk_size = min(valid_rows, budget_rows)
    elif original_n < 3072:
        row_chunk_size = min(valid_rows, 128)
    else:
        row_chunk_size = min(valid_rows, 24)
    if serial_chunk_size is not None and serial_chunk_size > 0:
        row_chunk_size = min(row_chunk_size, serial_chunk_size)
    return row_chunk_size


def _triatt_collective_query_width(
    *,
    valid_query: int,
    physical_query_width: int,
    mesh_cols: int,
) -> int:
    """Choose a rank-invariant query width for collective-bearing row loops."""

    if int(mesh_cols) > 1:
        return int(physical_query_width)
    return int(valid_query)


def _triatt_source_chunk_geometry(
    global_row_start: int,
    original_n: int,
    serial_chunk_size: int,
) -> tuple[int, int, int]:
    """Return serial chunk rows, row offset, and remaining rows at a global row."""

    chunk_start = (int(global_row_start) // int(serial_chunk_size)) * int(
        serial_chunk_size
    )
    source_rows = min(int(serial_chunk_size), int(original_n) - chunk_start)
    row_offset = int(global_row_start) - chunk_start
    return source_rows, row_offset, source_rows - row_offset


def _triatt_wrap_row_chunk_size(
    out_by_row_head_query: torch.Tensor,
    x_local: torch.Tensor,
) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIATT_WRAP_ROW_CHUNK")
    row_chunk_size = int("4" if value is None else value)
    if row_chunk_size <= 0:
        return out_by_row_head_query.shape[0]
    return row_chunk_size


def _pair_transition_flat_chunk_size(z_local: torch.Tensor) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_PAIR_TRANSITION_FLAT_CHUNK")
    return int("262144" if value is None else value)


def _pair_transition_row_pad_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 256:
        return original_n
    return min(original_n, max(valid_rows, 128))


def _pair_transition_source_flat_chunk_size(z_local: torch.Tensor) -> int:
    flat_chunk_size = _pair_transition_flat_chunk_size(z_local)
    if flat_chunk_size <= 0:
        return flat_chunk_size
    transition_chunk_rows = int(
        getattr(_primitives, "_TRANSITION_FLAT_CHUNK_ROWS", flat_chunk_size)
    )
    if transition_chunk_rows <= 0:
        return flat_chunk_size
    return min(flat_chunk_size, transition_chunk_rows)


def _pair_transition_global_flat_chunk_segments(
    *,
    original_n: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    global_flat_start: int,
    global_flat_end: int,
) -> list[tuple[int, int, int, int]]:
    original_n = int(original_n)
    row_start = int(row_start)
    row_end = int(row_end)
    col_start = int(col_start)
    col_end = int(col_end)
    global_flat_start = int(global_flat_start)
    global_flat_end = int(global_flat_end)
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    if (
        row_start >= valid_row_end
        or col_start >= valid_col_end
        or global_flat_start >= global_flat_end
    ):
        return []

    chunk_row_start = global_flat_start // original_n
    chunk_row_end = (global_flat_end - 1) // original_n + 1
    overlap_row_start = max(row_start, chunk_row_start)
    overlap_row_end = min(valid_row_end, chunk_row_end)
    segments: list[tuple[int, int, int, int]] = []
    for global_row in range(overlap_row_start, overlap_row_end):
        segment_col_start = 0
        segment_col_end = original_n
        if global_row == chunk_row_start:
            segment_col_start = max(segment_col_start, global_flat_start % original_n)
        if global_row == chunk_row_end - 1:
            segment_col_end = min(
                segment_col_end, (global_flat_end - 1) % original_n + 1
            )
        segment_col_start = max(segment_col_start, col_start)
        segment_col_end = min(segment_col_end, valid_col_end)
        if segment_col_start >= segment_col_end:
            continue
        chunk_offset = global_row * original_n + segment_col_start - global_flat_start
        segments.append((global_row, segment_col_start, segment_col_end, chunk_offset))
    return segments


def _pair_transition_intersecting_global_flat_chunks(
    *,
    original_n: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    flat_chunk_size: int,
) -> list[tuple[int, int]]:
    if flat_chunk_size <= 0:
        return []
    original_n = int(original_n)
    full_flat_rows = original_n * original_n
    chunks: list[tuple[int, int]] = []
    for global_flat_start in range(0, full_flat_rows, int(flat_chunk_size)):
        global_flat_end = min(global_flat_start + int(flat_chunk_size), full_flat_rows)
        if _pair_transition_global_flat_chunk_segments(
            original_n=original_n,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            global_flat_start=global_flat_start,
            global_flat_end=global_flat_end,
        ):
            chunks.append((global_flat_start, global_flat_end))
    return chunks


def _pair_transition_compact_source_launch_plan(
    *,
    original_n: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    flat_chunk_size: int,
) -> list[tuple[int, int, int]]:
    """Pack owned values while preserving serial full/tail launch families."""

    original_n = int(original_n)
    flat_chunk_size = int(flat_chunk_size)
    valid_rows = max(0, min(int(row_end), original_n) - int(row_start))
    valid_cols = max(0, min(int(col_end), original_n) - int(col_start))
    local_rows = valid_rows * valid_cols
    if local_rows == 0 or flat_chunk_size <= 0:
        return []

    source_rows = original_n * original_n
    tail_rows = source_rows % flat_chunk_size
    full_family_local_rows = local_rows
    if tail_rows:
        tail_start = source_rows - tail_rows
        tail_row, tail_col = divmod(tail_start, original_n)
        rows_before_tail = min(
            valid_rows,
            max(0, tail_row - int(row_start)),
        )
        full_family_local_rows = rows_before_tail * valid_cols
        if int(row_start) <= tail_row < int(row_start) + valid_rows:
            full_family_local_rows += min(
                valid_cols,
                max(0, tail_col - int(col_start)),
            )

    plan = [
        (
            local_start,
            min(local_start + flat_chunk_size, full_family_local_rows),
            flat_chunk_size,
        )
        for local_start in range(
            0,
            full_family_local_rows,
            flat_chunk_size,
        )
    ]
    if full_family_local_rows < local_rows:
        plan.append((full_family_local_rows, local_rows, tail_rows))
    return plan


def _pair_transition_should_use_compact_source(
    *,
    original_n: int,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    flat_chunk_size: int,
) -> bool:
    """Use compact launches only when the shard geometry amortizes packing."""

    source_plan = _pair_transition_intersecting_global_flat_chunks(
        original_n=original_n,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        flat_chunk_size=flat_chunk_size,
    )
    compact_plan = _pair_transition_compact_source_launch_plan(
        original_n=original_n,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        flat_chunk_size=flat_chunk_size,
    )
    launches_saved = len(source_plan) - len(compact_plan)
    source_launch_rows = sum(end - start for start, end in source_plan)
    compact_launch_rows = sum(launch_rows for _, _, launch_rows in compact_plan)
    return launches_saved >= 2 and compact_launch_rows * 4 <= source_launch_rows * 3


def _linear_output_slice(
    linear: torch.nn.Module,
    x: torch.Tensor,
    output_slice: slice,
) -> torch.Tensor:
    weight = linear.weight[output_slice]
    bias = None if linear.bias is None else linear.bias[output_slice]
    if getattr(linear, "precision", None) is not None:
        precision = linear.precision
        with torch.amp.autocast("cuda", enabled=False):
            x_precision = x.to(dtype=precision)
            weight_precision = weight.to(dtype=precision)
            bias = None if bias is None else bias.to(dtype=precision)
            return torch.nn.functional.linear(
                x_precision,
                weight_precision,
                bias,
            ).to(dtype=x.dtype)
    if x.dtype is torch.bfloat16:
        with torch.amp.autocast("cuda", enabled=False):
            bias = None if bias is None else bias.to(dtype=x.dtype)
            return torch.nn.functional.linear(x, weight.to(dtype=x.dtype), bias)
    return torch.nn.functional.linear(x, weight, bias)


def _source_grid_linear_output_chunk_size(out_features: int) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_SOURCE_GRID_LINEAR_OUTPUT_CHUNK")
    chunk_size = int("80" if value is None else value)
    if chunk_size <= 0:
        return int(out_features)
    return min(int(out_features), chunk_size)


def _linear_with_exact_source_launch_shape(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    source_rows: int,
) -> torch.Tensor:
    """Run a Linear with the exact source flat-row launch shape.

    This is only used at CUDA launch-family boundaries where the regular
    Fold-CP bucket is insufficient for bitwise parity. It pads owned rows with
    zeros and discards them after projection; it never gathers remote pair data.
    """

    local_rows = int(x.numel() // x.shape[-1]) if x.shape[-1] else 0
    source_rows = int(source_rows)
    if source_rows <= local_rows:
        return linear(x)
    flat = x.contiguous().reshape(local_rows, x.shape[-1])
    launch = flat.new_zeros(source_rows, flat.shape[-1])
    launch[:local_rows].copy_(flat)
    projected = linear(launch)[:local_rows]
    return projected.reshape(*x.shape[:-1], -1)


def _linear_pair_tile_with_source_grid_launch(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    col_start: int,
    valid_rows: int,
    valid_cols: int,
    output_chunk_size: int | None = None,
) -> torch.Tensor:
    """Project a local pair tile with the source full-pair flat launch layout."""

    if valid_rows <= 0 or valid_cols <= 0:
        return linear(x)
    source_rows = int(original_n) * int(original_n)
    flat = x.contiguous().reshape(-1, x.shape[-1])
    tile_rows = int(x.shape[-3])
    tile_cols = int(x.shape[-2])
    tile = flat[: tile_rows * tile_cols].reshape(
        tile_rows,
        tile_cols,
        flat.shape[-1],
    )
    out_features = int(linear.weight.shape[0])
    if output_chunk_size is None:
        output_chunk_size = _source_grid_linear_output_chunk_size(out_features)
    elif output_chunk_size <= 0:
        output_chunk_size = out_features
    else:
        output_chunk_size = min(out_features, int(output_chunk_size))

    compact_bf16_384 = (
        os.environ.get(
            "OPENDDE_FOLDCP_COMPACT_BF16_384_SOURCE_GRID",
            "1",
        )
        != "0"
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and tuple(linear.weight.shape) == (384, 384)
        and source_rows >= 90_000
        and output_chunk_size >= out_features
    )
    if compact_bf16_384:
        compact = (
            tile[:valid_rows, :valid_cols, :]
            .contiguous()
            .reshape(valid_rows * valid_cols, flat.shape[-1])
        )
        out = flat.new_zeros(x.shape[:-1] + (out_features,))
        out_tile = out.reshape(-1, out_features)[: tile_rows * tile_cols].reshape(
            tile_rows,
            tile_cols,
            out_features,
        )
        out_tile[:valid_rows, :valid_cols, :].copy_(
            linear(compact).reshape(valid_rows, valid_cols, out_features)
        )
        return out

    launch = flat.new_zeros(source_rows, flat.shape[-1])
    launch_grid = launch.reshape(original_n, original_n, flat.shape[-1])
    launch_grid[
        row_start : row_start + valid_rows,
        col_start : col_start + valid_cols,
        :,
    ].copy_(tile[:valid_rows, :valid_cols, :])
    if output_chunk_size >= out_features:
        projected = linear(launch).reshape(original_n, original_n, out_features)
        out = projected.new_zeros(x.shape[:-1] + (out_features,))
        out_tile = out.reshape(-1, out_features)[: tile_rows * tile_cols].reshape(
            tile_rows,
            tile_cols,
            out_features,
        )
        out_tile[:valid_rows, :valid_cols, :].copy_(
            projected[
                row_start : row_start + valid_rows,
                col_start : col_start + valid_cols,
                :,
            ]
        )
        return out

    out = flat.new_zeros(x.shape[:-1] + (out_features,))
    out_tile = out.reshape(-1, out_features)[: tile_rows * tile_cols].reshape(
        tile_rows,
        tile_cols,
        out_features,
    )
    for channel_start in range(0, out_features, output_chunk_size):
        channel_end = min(channel_start + output_chunk_size, out_features)
        projected = _linear_output_slice(
            linear,
            launch,
            slice(channel_start, channel_end),
        ).reshape(original_n, original_n, channel_end - channel_start)
        out_tile[:valid_rows, :valid_cols, channel_start:channel_end].copy_(
            projected[
                row_start : row_start + valid_rows,
                col_start : col_start + valid_cols,
                :,
            ]
        )
        del projected
    return out


def _ring_gather_should_preallocate(
    local_tensor: torch.Tensor,
    side: int,
    dim: int,
    length: int | None,
) -> bool:
    """Use direct placement only when it removes a material gather buffer."""

    dim = dim if dim >= 0 else local_tensor.ndim + dim
    output_length = (
        int(side) * int(local_tensor.shape[dim]) if length is None else int(length)
    )
    output_elements = output_length
    for axis, size in enumerate(local_tensor.shape):
        if axis != dim:
            output_elements *= int(size)
    return output_elements * local_tensor.element_size() >= 512 * 1024**2


def _ring_gather_by_row(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    dim: int,
    length: int | None = None,
) -> torch.Tensor:
    side = mesh.layout.shape[1]
    if side == 1:
        out = local_tensor
    elif mesh.layout.shape[0] == 1 and side >= 3:
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        local_width = int(local_tensor.shape[dim])
        padded_length = side * local_width
        output_length = padded_length if length is None else int(length)
        if output_length < 0 or output_length > padded_length:
            raise ValueError(
                f"row gather length must be in [0, {padded_length}], "
                f"got {output_length}"
            )
        source = local_tensor.movedim(dim, 0).contiguous()
        gathered = source.new_empty((padded_length,) + tuple(source.shape[1:]))
        dist.all_gather_into_tensor(
            gathered,
            source,
            group=mesh.group_row,
        )
        out = gathered.narrow(0, 0, output_length).movedim(0, dim)
        return out.contiguous()
    else:
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        local_width = int(local_tensor.shape[dim])
        padded_length = side * local_width
        output_length = padded_length if length is None else int(length)
        if output_length < 0 or output_length > padded_length:
            raise ValueError(
                f"row gather length must be in [0, {padded_length}], "
                f"got {output_length}"
            )
        if not _ring_gather_should_preallocate(
            local_tensor,
            side,
            dim,
            length,
        ):
            ring = mesh.ring_comm()
            gathered: list[torch.Tensor | None] = [None for _ in range(side)]
            gathered[mesh.coord[1]] = local_tensor.contiguous()
            ready = gathered[mesh.coord[1]]
            for step in range(1, side):
                ready = ring.comm_row.exchange(ready.contiguous())
                source_col = (mesh.coord[1] + step) % side
                gathered[source_col] = ready
            if any(item is None for item in gathered):
                raise RuntimeError("failed to gather row ring blocks.")
            out = torch.cat(
                [item for item in gathered if item is not None],
                dim=dim,
            )
            if length is not None:
                out = out.narrow(dim, 0, length)
            return out.contiguous()
        out_shape = list(local_tensor.shape)
        out_shape[dim] = output_length
        out = local_tensor.new_empty(out_shape)
        ring = mesh.ring_comm()
        ready = local_tensor.contiguous()
        for step in range(side):
            source_col = (mesh.coord[1] + step) % side
            output_start = source_col * local_width
            output_end = min(output_start + local_width, output_length)
            if output_start < output_end:
                out.narrow(dim, output_start, output_end - output_start).copy_(
                    ready.narrow(dim, 0, output_end - output_start)
                )
            if step + 1 < side:
                ready = ring.comm_row.exchange(ready.contiguous())
        return out
    if length is not None:
        dim = dim if dim >= 0 else out.ndim + dim
        out = out.narrow(dim, 0, length)
    return out.contiguous()


def _ring_gather_pair_matmul_lhs(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    *,
    length: int,
    incoming: bool,
) -> torch.Tensor:
    """Gather a 1xP pair shard directly into the contiguous BMM-LHS layout."""

    if local_tensor.ndim != 4 or local_tensor.shape[0] != 1:
        raise ValueError("direct pair-matmul gather expects [1, N, N_local, C].")
    side = int(mesh.layout.shape[1])
    local_width = int(local_tensor.shape[-2])
    padded_length = side * local_width
    length = int(length)
    if length < 0 or length > padded_length:
        raise ValueError(
            f"pair-matmul gather length must be in [0, {padded_length}], got {length}"
        )
    n_rows = int(local_tensor.shape[-3])
    channels = int(local_tensor.shape[-1])
    out_shape = (channels, length, n_rows) if incoming else (channels, n_rows, length)
    out = local_tensor.new_empty(out_shape)
    ring = mesh.ring_comm()
    local_tensor_bytes = int(local_tensor.numel()) * int(local_tensor.element_size())
    if local_tensor_bytes > _ONE_BY_P_TRIMUL_LHS_RING_MAX_BYTES:
        row_bytes = local_width * channels * int(local_tensor.element_size())
        row_chunk = max(
            1,
            _ONE_BY_P_TRIMUL_LHS_RING_ROW_BUFFER_BYTES // max(1, row_bytes),
        )
        for row_start in range(0, n_rows, row_chunk):
            row_end = min(row_start + row_chunk, n_rows)
            ready = local_tensor[..., row_start:row_end, :, :].contiguous()
            for step in range(side):
                source_col = (int(mesh.coord[1]) + step) % side
                output_start = source_col * local_width
                output_end = min(output_start + local_width, length)
                if output_start < output_end:
                    width = output_end - output_start
                    if incoming:
                        out[
                            :,
                            output_start:output_end,
                            row_start:row_end,
                        ].copy_(ready.squeeze(0).permute(2, 1, 0)[:, :width, :])
                    else:
                        out[
                            :,
                            row_start:row_end,
                            output_start:output_end,
                        ].copy_(ready.squeeze(0).permute(2, 0, 1)[:, :, :width])
                if step + 1 < side:
                    ready = ring.comm_row.exchange(ready.contiguous())
        return out

    ready = local_tensor.contiguous()
    for step in range(side):
        source_col = (int(mesh.coord[1]) + step) % side
        output_start = source_col * local_width
        output_end = min(output_start + local_width, length)
        if output_start < output_end:
            width = output_end - output_start
            if incoming:
                out[:, output_start:output_end, :].copy_(
                    ready.squeeze(0).permute(2, 1, 0)[:, :width, :]
                )
            else:
                out[:, :, output_start:output_end].copy_(
                    ready.squeeze(0).permute(2, 0, 1)[:, :, :width]
                )
        if step + 1 < side:
            ready = ring.comm_row.exchange(ready.contiguous())
    return out


def _one_by_p_local_source_column_chunk(
    local_tensor: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
    source_start: int,
    source_end: int,
) -> torch.Tensor:
    """Place this rank's owned columns in one global source-column chunk."""

    original_n = int(z_spec.original_shape[z_spec.pair_dims[1]])
    source_start = max(0, int(source_start))
    source_end = min(int(source_end), original_n)
    if source_start >= source_end:
        raise ValueError(
            f"source column chunk must be non-empty, got [{source_start}, {source_end})"
        )

    col_start, col_end = z_spec.col_range
    valid_col_end = min(int(col_end), original_n)
    overlap_start = max(source_start, int(col_start))
    overlap_end = min(source_end, valid_col_end)
    chunk = local_tensor.new_zeros(
        local_tensor.shape[:-2] + (source_end - source_start, local_tensor.shape[-1])
    )
    if overlap_start < overlap_end:
        local_slice = slice(
            overlap_start - int(col_start), overlap_end - int(col_start)
        )
        chunk_slice = slice(overlap_start - source_start, overlap_end - source_start)
        chunk[..., chunk_slice, :] = local_tensor[..., local_slice, :]
    return chunk


def _one_by_p_gather_source_columns_by_row(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
    source_start: int,
    source_end: int,
) -> torch.Tensor:
    """Reconstruct one P-independent source-column chunk on every row rank."""

    chunk = _one_by_p_local_source_column_chunk(
        local_tensor,
        z_spec,
        source_start,
        source_end,
    )
    if mesh.layout.shape[1] > 1:
        dist.all_reduce(chunk, op=dist.ReduceOp.SUM, group=mesh.group_row)
    return chunk


def _one_by_p_transpose_columns_to_owned_rows(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    """Redistribute a column shard into this rank's rows with one all-to-all."""

    if local_tensor.ndim != 4 or local_tensor.shape[0] != 1:
        raise ValueError(
            "1xP pair transpose expects an unbatched [1, N, N_local, C] tensor."
        )
    side = int(mesh.layout.shape[1])
    local_width = int(local_tensor.shape[-2])
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    padded_n = side * local_width
    if original_n > padded_n:
        raise ValueError(
            f"original pair length {original_n} exceeds padded length {padded_n}."
        )
    if side == 1:
        return local_tensor.transpose(-3, -2).contiguous()

    padded = local_tensor.new_zeros((padded_n, local_width, local_tensor.shape[-1]))
    padded[:original_n].copy_(local_tensor.squeeze(0)[:original_n])
    send = padded.reshape(
        side,
        local_width,
        local_width,
        local_tensor.shape[-1],
    ).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=mesh.group_row)
    owned_rows = (
        recv.permute(1, 0, 2, 3)
        .contiguous()
        .reshape(local_width, padded_n, local_tensor.shape[-1])
    )
    return owned_rows[:, :original_n].unsqueeze(0).contiguous()


_ONE_BY_P_OWNED_ROWS_MAX_TENSOR_BYTES = 2 * 1024**3


def _one_by_p_should_materialize_owned_rows(
    local_tensor: torch.Tensor,
) -> bool:
    """Bound the four simultaneous buffers used by the all-to-all transpose."""

    tensor_bytes = int(local_tensor.numel()) * int(local_tensor.element_size())
    return tensor_bytes <= _ONE_BY_P_OWNED_ROWS_MAX_TENSOR_BYTES


def _one_by_p_should_project_b_owned_rows(
    z_norm: torch.Tensor,
    mesh: FoldCPProcessMesh,
    *,
    original_n: int,
    c_hidden: int,
    direction: TriangleMultiplicationDirection,
) -> bool:
    """Keep outgoing B in its final row-owned layout when capacity permits."""

    if (
        os.environ.get(
            "OPENDDE_FOLDCP_TRIMUL_DIRECT_B_OWNED_ROWS",
            "1",
        )
        == "0"
        or tuple(mesh.layout.shape)[0] != 1
        or int(mesh.layout.shape[1]) <= 1
        or direction != TriangleMultiplicationDirection.OUTGOING
        or not z_norm.is_cuda
        or z_norm.dtype != torch.bfloat16
        or torch.is_grad_enabled()
    ):
        return False
    prefix_elements = int(z_norm.numel()) // (
        int(z_norm.shape[-3]) * int(z_norm.shape[-2]) * int(z_norm.shape[-1])
    )
    owned_rows_bytes = (
        prefix_elements
        * int(z_norm.shape[-2])
        * int(original_n)
        * int(c_hidden)
        * int(z_norm.element_size())
    )
    return owned_rows_bytes <= _ONE_BY_P_OWNED_ROWS_MAX_TENSOR_BYTES


def _one_by_p_should_offload_trimul_b(
    b_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> bool:
    """Offload only inference-scale CUDA shards that exceed the safe peak."""

    tensor_bytes = int(b_local.numel()) * int(b_local.element_size())
    return (
        not torch.is_grad_enabled()
        and b_local.device.type == "cuda"
        and int(mesh.layout.shape[0]) == 1
        and int(mesh.layout.shape[1]) > 1
        and tensor_bytes > _ONE_BY_P_TRIMUL_B_OFFLOAD_MIN_BYTES
    )


def _one_by_p_gather_row_chunk_by_row(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    row_start: int,
    row_end: int,
    length: int,
) -> torch.Tensor:
    """Gather one row chunk without materializing the complete pair slab."""

    return _ring_gather_by_row(
        local_tensor[..., row_start:row_end, :, :],
        mesh,
        dim=-2,
        length=length,
    )


def _ring_gather_output_rows_by_row(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    *,
    row_start: int,
    valid_rows: int,
    padded_rows: int,
    length: int,
) -> torch.Tensor:
    """Collect K blocks after selecting rows owned by the receiving rank.

    Each peer owns a different K-column block but the receiver needs the same
    output-row interval from every peer.  Therefore selection must happen
    after each full local tile reaches the receiver, not before the ring.
    """

    side = mesh.layout.shape[1]
    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    ready = local_tensor.contiguous()
    for step in range(side):
        source_col = (mesh.coord[1] + step) % side
        rows = local_tensor.new_zeros(
            local_tensor.shape[:-3]
            + (padded_rows, local_tensor.shape[-2], local_tensor.shape[-1])
        )
        rows[..., :valid_rows, :, :] = ready[
            ..., row_start : row_start + valid_rows, :, :
        ]
        gathered[source_col] = rows
        if step + 1 < side:
            ready = ring.comm_row.exchange(ready.contiguous())
    out = torch.cat([item for item in gathered if item is not None], dim=-2)
    return out.narrow(-2, 0, length).contiguous()


def _ring_gather_by_col(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    dim: int,
    length: int | None = None,
) -> torch.Tensor:
    side = mesh.layout.shape[0]
    if side == 1:
        out = local_tensor
    else:
        ring = mesh.ring_comm()
        gathered: list[torch.Tensor | None] = [None for _ in range(side)]
        gathered[mesh.coord[0]] = local_tensor.contiguous()
        ready = gathered[mesh.coord[0]]
        for step in range(1, side):
            ready = ring.comm_col.exchange(ready.contiguous())
            source_row = (mesh.coord[0] + step) % side
            gathered[source_row] = ready
        if any(item is None for item in gathered):
            raise RuntimeError("failed to gather column ring blocks.")
        out = torch.cat([item for item in gathered if item is not None], dim=dim)
    if length is not None:
        dim = dim if dim >= 0 else out.ndim + dim
        out = out.narrow(dim, 0, length)
    return out.contiguous()


def _transpose_source_pair_tile(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Return the reciprocal source tile for triangle multiplication.

    A square mesh owns the reciprocal tile on its transposed rank. In the
    runtime 1 x P layout, each rank already owns every row for its column
    shard, so the reciprocal tile is a local swap of the two pair axes.
    """

    if mesh.layout.shape[0] == 1:
        return local_tensor.transpose(-3, -2).contiguous()
    return mesh.ring_comm().comm_2d_trans.exchange(local_tensor.contiguous())


def _select_query_row_stack(
    row_stack: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Select the row block that owns this rank's query block."""

    stack_index = mesh.coord[1] if mesh.layout.shape[0] > 1 else 0
    return row_stack[stack_index]


def _transpose_pair_shard_spec_for_local_attention(
    spec: FoldCPPairShardSpec,
) -> FoldCPPairShardSpec:
    """Describe a locally transposed 1 x P pair shard."""

    row_dim, col_dim = spec.pair_dims
    original_n = spec.original_shape[row_dim]
    padded_shape = list(spec.original_shape)
    padded_shape[row_dim] = spec.padded_shape[col_dim]
    padded_shape[col_dim] = original_n
    return replace(
        spec,
        padded_shape=tuple(padded_shape),
        row_range=spec.col_range,
        col_range=(0, original_n),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )


def _local_attention_mesh(mesh: FoldCPProcessMesh) -> FoldCPProcessMesh:
    """Create a no-communication view for a transposed 1 x P shard."""

    return replace(
        mesh,
        layout=FoldCP2DLayout((1, 1)),
        cp_global_ranks=(mesh.cp_global_ranks[mesh.cp_rank],),
        cp_rank=0,
        coord=(0, 0),
    )


def _triangle_source_column_chunks(
    n_token: int,
    chunk_size: int = 256,
) -> list[tuple[int, int]]:
    half_n = n_token // 2 + n_token % 2
    chunks: list[tuple[int, int]] = []
    starts = list(range(0, half_n, chunk_size))
    for start, next_start in zip(starts, starts[1:] + [half_n]):
        chunks.append((start, next_start))
    for start in range(half_n, n_token, chunk_size):
        chunks.append((start, min(start + chunk_size, n_token)))
    return chunks


def _one_by_p_trimul_full_b_bytes(
    b_local: torch.Tensor,
    original_n: int,
) -> int:
    if b_local.ndim < 2:
        raise ValueError("triangle multiplication input must have at least two axes.")
    full_elements = 1
    global_col_dim = b_local.ndim - 2
    for dim, size in enumerate(b_local.shape):
        full_elements *= int(original_n) if dim == global_col_dim else int(size)
    return full_elements * b_local.element_size()


def _one_by_p_trimul_full_b_max_bytes() -> int:
    """Return the audited, P-independent full-B gather limit."""

    return _ONE_BY_P_TRIMUL_FULL_B_MAX_BYTES


def _one_by_p_trimul_should_gather_full_b(
    b_local: torch.Tensor,
    original_n: int,
) -> bool:
    return (
        _one_by_p_trimul_full_b_bytes(
            b_local,
            original_n,
        )
        <= _one_by_p_trimul_full_b_max_bytes()
    )


def _triangle_source_matmul_row_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 1024:
        return original_n
    return valid_rows


def _trimul_layer_norm_source_grid_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIMUL_LAYERNORM_SOURCE_GRID_MAX_BYTES")
    if value is None:
        return 0
    return int(value)


def _trimul_layer_norm_row_slab_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIMUL_LAYERNORM_ROW_SLAB_MAX_BYTES")
    if value is None:
        return 0
    return int(value)


def _trimul_can_layer_norm_source_grid(
    z_in: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
) -> bool:
    max_bytes = _trimul_layer_norm_source_grid_max_bytes()
    if max_bytes <= 0:
        return False
    row_dim, col_dim = z_spec.pair_dims
    source_rows = int(z_spec.original_shape[row_dim]) * int(
        z_spec.original_shape[col_dim]
    )
    launch_bytes = source_rows * int(z_in.shape[-1]) * z_in.element_size()
    return launch_bytes <= max_bytes


def _trimul_projection_source_grid_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIMUL_PROJECTION_SOURCE_GRID_MAX_BYTES")
    if value is None:
        return 3 * 1024 * 1024 * 1024
    return int(value)


def _trimul_projection_launch_bytes(
    x: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
) -> int:
    row_dim, col_dim = z_spec.pair_dims
    source_rows = int(z_spec.original_shape[row_dim]) * int(
        z_spec.original_shape[col_dim]
    )
    return source_rows * int(x.shape[-1]) * x.element_size()


def _trimul_can_projection_source_grid(
    x: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
) -> bool:
    max_bytes = _trimul_projection_source_grid_max_bytes()
    if max_bytes <= 0:
        return False
    return _trimul_projection_launch_bytes(x, z_spec) <= max_bytes


def _trimul_project_channel_chunk_size(
    z_norm: torch.Tensor,
    z_spec: FoldCPPairShardSpec | None,
    c_hidden: int,
) -> int:
    explicit = os.environ.get("OPENDDE_FOLDCP_TRIMUL_PROJECT_CHANNEL_CHUNK")
    if explicit is None:
        explicit = os.environ.get("OPENDDE_FOLDCP_TRIMUL_CHANNEL_CHUNK")
    if explicit is not None:
        return int(explicit)
    if z_spec is None:
        return 0
    launch_bytes = _trimul_projection_launch_bytes(z_norm, z_spec)
    large_launch = launch_bytes >= 128 * 1024 * 1024
    if large_launch or not _trimul_can_projection_source_grid(z_norm, z_spec):
        return min(16, c_hidden)
    return 0


def _triangle_layer_norm_source_row_slab(
    layer_norm: torch.nn.Module,
    z_in: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    if valid_rows == 0 or valid_cols == 0:
        return z_in.new_zeros(z_in.shape)
    if _trimul_can_layer_norm_source_grid(z_in, z_spec):
        source_grid = z_in.new_zeros(
            z_in.shape[:-3] + (original_n, original_n, z_in.shape[-1])
        )
        source_grid[
            ...,
            row_start:valid_row_end,
            col_start:valid_col_end,
            :,
        ] = z_in[..., :valid_rows, :valid_cols, :]
        normed = layer_norm(source_grid)
        del source_grid
        out = z_in.new_zeros(z_in.shape)
        out[..., :valid_rows, :valid_cols, :] = normed[
            ...,
            row_start:valid_row_end,
            col_start:valid_col_end,
            :,
        ]
        del normed
        return out.contiguous()
    row_slab_bytes = (
        int(valid_rows)
        * int(original_n)
        * int(z_in.shape[-1])
        * int(z_in.element_size())
    )
    if row_slab_bytes > _trimul_layer_norm_row_slab_max_bytes():
        # LayerNorm reduces only over C_z.  For very large 1xP structural-token
        # grids, materializing zero-filled remote columns can exceed device
        # capacity even though those columns are immediately discarded.  The
        # local launch computes exactly the owned rows/columns and bounds the
        # temporary to the sharded pair tensor.
        return layer_norm(z_in).contiguous()
    row_slab = z_in.new_zeros(
        z_in.shape[:-3] + (valid_rows, original_n, z_in.shape[-1])
    )
    row_slab[..., :, col_start:valid_col_end, :] = z_in[
        ..., :valid_rows, :valid_cols, :
    ]
    normed = layer_norm(row_slab)
    del row_slab
    out = z_in.new_zeros(z_in.shape)
    out[..., :valid_rows, :valid_cols, :] = normed[..., :, col_start:valid_col_end, :]
    del normed
    return out.contiguous()


def _triangle_project_source_launch(
    linear_g: torch.nn.Module,
    linear_p: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    source_rows: int,
    source_unbatched: bool = False,
    layer_norm: torch.nn.Module | None = None,
    original_n: int | None = None,
    row_start: int = 0,
    col_start: int = 0,
    source_chunk_rows: int | None = None,
    source_chunk_cols: int | None = None,
) -> torch.Tensor:
    if source_unbatched and x.ndim == 4 and x.shape[0] == 1:
        projected = _triangle_project_source_launch(
            linear_g,
            linear_p,
            x.squeeze(0),
            None if mask is None else mask.squeeze(0),
            source_rows=source_rows,
            source_unbatched=False,
            layer_norm=layer_norm,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
            source_chunk_rows=source_chunk_rows,
            source_chunk_cols=source_chunk_cols,
        )
        return projected.unsqueeze(0)
    if layer_norm is not None:
        x = layer_norm(x)
    if x.dtype == torch.bfloat16 and not torch.is_grad_enabled():
        gate = linear_g(x)
        proj = linear_p(x)
        out = torch.sigmoid(gate)
        out *= proj
        if mask is not None:
            out *= mask
        return out
    if (
        source_chunk_rows is not None
        and source_chunk_cols is not None
        and x.ndim in (3, 4)
    ):
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError("triangle source-chunk launch expects batch size 1.")
            x_for_linear = x.squeeze(0)
            unsqueeze_batch = True
        else:
            x_for_linear = x
            unsqueeze_batch = False
        gate = foldcp_pair_tile_linear_with_source_chunk_launch(
            linear_g,
            x_for_linear,
            source_rows=source_chunk_rows,
            source_cols=source_chunk_cols,
            row_start=row_start,
            col_start=col_start,
        )
        proj = foldcp_pair_tile_linear_with_source_chunk_launch(
            linear_p,
            x_for_linear,
            source_rows=source_chunk_rows,
            source_cols=source_chunk_cols,
            row_start=row_start,
            col_start=col_start,
        )
        if unsqueeze_batch:
            gate = gate.unsqueeze(0)
            proj = proj.unsqueeze(0)
    elif original_n is not None and x.ndim in (3, 4):
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError("triangle source-grid launch expects batch size 1.")
            x_for_linear = x.squeeze(0)
            unsqueeze_batch = True
        else:
            x_for_linear = x
            unsqueeze_batch = False
        gate = foldcp_pair_row_slab_linear_with_source_grid_launch(
            linear_g,
            x_for_linear,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
        )
        proj = foldcp_pair_row_slab_linear_with_source_grid_launch(
            linear_p,
            x_for_linear,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
        )
        if unsqueeze_batch:
            gate = gate.unsqueeze(0)
            proj = proj.unsqueeze(0)
    else:
        gate = foldcp_linear_with_source_launch_shape(
            linear_g,
            x,
            source_rows=source_rows,
        )
        proj = foldcp_linear_with_source_launch_shape(
            linear_p,
            x,
            source_rows=source_rows,
        )
    out = torch.sigmoid(gate)
    out *= proj
    if mask is not None:
        out *= mask
    return out


def _triangle_a_projection_source_chunks(
    module: torch.nn.Module,
    z_norm: torch.Tensor,
    mask: torch.Tensor | None,
    z_spec: FoldCPPairShardSpec,
    *,
    source_unbatched: bool = False,
) -> torch.Tensor:
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    out = z_norm.new_zeros(z_norm.shape[:-1] + (int(module.c_hidden),))
    if valid_rows == 0 or valid_cols == 0:
        return out

    source_row_chunk = 256
    for global_start in range(0, original_n, source_row_chunk):
        global_end = min(global_start + source_row_chunk, original_n)
        overlap_start = max(global_start, row_start)
        overlap_end = min(global_end, valid_row_end)
        if overlap_start >= overlap_end:
            continue
        local_row_slice = slice(overlap_start - row_start, overlap_end - row_start)
        source_rows = int(global_end - global_start) * int(original_n)
        projected = _triangle_project_source_launch(
            module.linear_a_g,
            module.linear_a_p,
            z_norm[..., local_row_slice, :valid_cols, :],
            None if mask is None else mask[..., local_row_slice, :valid_cols, :],
            source_rows=source_rows,
            source_unbatched=source_unbatched,
            row_start=overlap_start - global_start,
            col_start=col_start,
            source_chunk_rows=global_end - global_start,
            source_chunk_cols=original_n,
        )
        out[..., local_row_slice, :valid_cols, :] = projected
    return out.contiguous()


def _triangle_b_projection_source_chunk(
    module: torch.nn.Module,
    z_norm: torch.Tensor,
    mask: torch.Tensor | None,
    mesh: FoldCPProcessMesh,
    direction: TriangleMultiplicationDirection,
    z_spec: FoldCPPairShardSpec,
    *,
    z_source: torch.Tensor | None = None,
    source_unbatched: bool = False,
    return_owned_rows: bool = False,
) -> torch.Tensor:
    original_n = z_spec.original_shape[z_spec.pair_dims[0]]
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    if return_owned_rows and (
        direction != TriangleMultiplicationDirection.OUTGOING
        or tuple(mesh.layout.shape)[0] != 1
    ):
        raise ValueError(
            "owned-row B projection requires outgoing triangle multiplication "
            "on a 1xP mesh."
        )
    if return_owned_rows:
        out = z_norm.new_zeros(
            z_norm.shape[:-3]
            + (
                int(z_norm.shape[-2]),
                int(original_n),
                int(module.c_hidden),
            )
        )
    else:
        out = z_norm.new_zeros(z_norm.shape[:-1] + (int(module.c_hidden),))
    # A padding-only column rank must still enter row-group collectives so its
    # peers do not wait forever on non-divisible 1 x P layouts.
    if valid_rows == 0:
        return out
    use_source_grid = z_source is None and _trimul_can_projection_source_grid(
        z_norm,
        z_spec,
    )
    z_project_source = z_norm if z_source is None else z_source

    if direction == TriangleMultiplicationDirection.OUTGOING:
        for global_start, global_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(global_start, row_start)
            overlap_end = min(global_end, valid_row_end)
            if overlap_start >= overlap_end:
                continue
            local_row_slice = slice(overlap_start - row_start, overlap_end - row_start)
            z_slab = _one_by_p_gather_row_chunk_by_row(
                z_project_source,
                mesh,
                local_row_slice.start,
                local_row_slice.stop,
                original_n,
            )
            mask_slab = (
                None
                if mask is None
                else _one_by_p_gather_row_chunk_by_row(
                    mask,
                    mesh,
                    local_row_slice.start,
                    local_row_slice.stop,
                    original_n,
                )
            )
            if use_source_grid:
                projected = _triangle_project_source_launch(
                    module.linear_b_g,
                    module.linear_b_p,
                    z_slab,
                    mask_slab,
                    source_rows=int(global_end - global_start) * int(original_n),
                    source_unbatched=source_unbatched,
                    original_n=original_n,
                    row_start=overlap_start,
                    col_start=0,
                )
            else:
                projected = _triangle_project_source_launch(
                    module.linear_b_g,
                    module.linear_b_p,
                    z_slab,
                    mask_slab,
                    source_rows=int(global_end - global_start) * int(original_n),
                    source_unbatched=source_unbatched,
                    layer_norm=module.layer_norm_in if z_source is not None else None,
                    row_start=overlap_start - global_start,
                    col_start=0,
                    source_chunk_rows=global_end - global_start,
                    source_chunk_cols=original_n,
                )
            if return_owned_rows:
                owned_start = max(overlap_start, col_start)
                owned_end = min(overlap_end, valid_col_end)
                if owned_start < owned_end:
                    out[
                        ...,
                        owned_start - col_start : owned_end - col_start,
                        :original_n,
                        :,
                    ] = projected[
                        ...,
                        owned_start - overlap_start : owned_end - overlap_start,
                        :original_n,
                        :,
                    ]
            else:
                out[..., local_row_slice, :valid_cols, :] = projected[
                    ..., :, col_start:valid_col_end, :
                ]
            del projected, z_slab, mask_slab
        return out.contiguous()

    if direction == TriangleMultiplicationDirection.INCOMING:
        z_slab = _ring_gather_by_col(z_project_source, mesh, dim=-3, length=original_n)
        mask_slab = (
            None
            if mask is None
            else _ring_gather_by_col(mask, mesh, dim=-3, length=original_n)
        )
        z_slab = z_slab[..., :, :valid_cols, :]
        if mask_slab is not None:
            mask_slab = mask_slab[..., :, :valid_cols, :]
        for global_start, global_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(global_start, col_start)
            overlap_end = min(global_end, valid_col_end)
            if overlap_start >= overlap_end:
                continue
            local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
            if use_source_grid:
                projected = _triangle_project_source_launch(
                    module.linear_b_g,
                    module.linear_b_p,
                    z_slab[..., :, local_col_slice, :],
                    None
                    if mask_slab is None
                    else mask_slab[..., :, local_col_slice, :],
                    source_rows=int(original_n) * int(global_end - global_start),
                    source_unbatched=source_unbatched,
                    original_n=original_n,
                    row_start=0,
                    col_start=overlap_start,
                )
            else:
                projected = _triangle_project_source_launch(
                    module.linear_b_g,
                    module.linear_b_p,
                    z_slab[..., :, local_col_slice, :],
                    None
                    if mask_slab is None
                    else mask_slab[..., :, local_col_slice, :],
                    source_rows=int(original_n) * int(global_end - global_start),
                    source_unbatched=source_unbatched,
                    layer_norm=module.layer_norm_in if z_source is not None else None,
                    row_start=0,
                    col_start=overlap_start - global_start,
                    source_chunk_rows=original_n,
                    source_chunk_cols=global_end - global_start,
                )
            out[..., :valid_rows, local_col_slice, :] = projected[
                ..., row_start:valid_row_end, :, :
            ]
        return out.contiguous()

    raise ValueError(f"unsupported direction={direction}")


def _distributed_triangle_multiplication_source_matmul(
    a_local: torch.Tensor,
    b_local: torch.Tensor | None,
    mesh: FoldCPProcessMesh,
    direction: TriangleMultiplicationDirection,
    z_spec: FoldCPPairShardSpec,
    *,
    b_owned_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a local triangle-multiplication tile with source-like matmuls.

    The ring implementation is mathematically correct but sums the K dimension
    as separate per-rank matmuls.  The OpenDDE inference path computes each
    output-column chunk with one full-K matmul.  This variant keeps the output
    sharded while gathering only the full-K projected inputs needed for the
    current local tile, preserving that source matmul shape.
    """

    if a_local.ndim != 4 or (b_local is not None and b_local.ndim != 4):
        raise ValueError("triangle multiplication expects [B, N, N, C] inputs.")
    if b_local is not None and a_local.shape != b_local.shape:
        raise ValueError("a_local and b_local must have the same shape.")
    if b_local is None and b_owned_rows is None:
        raise ValueError("triangle multiplication requires B projection data.")

    local_shape = a_local.shape
    original_n = z_spec.original_shape[z_spec.pair_dims[0]]
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range

    def _source_unbatched_outgoing_matmul(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        lhs_prepared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lhs.shape[0] != 1 or rhs.shape[0] != 1:
            return torch.matmul(
                lhs.permute(0, 3, 1, 2).contiguous(),
                rhs.permute(0, 3, 2, 1).contiguous(),
            ).permute(0, 2, 3, 1)
        lhs_for_matmul = (
            lhs.squeeze(0).permute(2, 0, 1).contiguous()
            if lhs_prepared is None
            else lhs_prepared
        )
        chunk = torch.matmul(
            lhs_for_matmul,
            rhs.squeeze(0).permute(2, 1, 0),
        ).permute(1, 2, 0)
        return chunk.unsqueeze(0)

    def _source_unbatched_incoming_matmul(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        lhs_prepared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lhs.shape[0] != 1 or rhs.shape[0] != 1:
            return torch.matmul(
                lhs.permute(0, 3, 2, 1).contiguous(),
                rhs.permute(0, 3, 1, 2).contiguous(),
            ).permute(0, 2, 3, 1)
        lhs_for_matmul = (
            lhs.squeeze(0).permute(2, 1, 0).contiguous()
            if lhs_prepared is None
            else lhs_prepared
        )
        chunk = torch.matmul(
            lhs_for_matmul,
            rhs.squeeze(0).permute(2, 0, 1),
        ).permute(1, 2, 0)
        return chunk.unsqueeze(0)

    if mesh.layout.shape[0] == 1:
        valid_col_end = min(col_end, original_n)
        side = int(mesh.layout.shape[1])

        # A 1 x P mesh owns every pair row and only shards pair columns.  The
        # square-mesh transpose-rank exchange below is therefore not valid:
        # a local axis swap cannot manufacture the remote K blocks.  Gather
        # the sharded K axis on the row ring.  The source inference kernel emits
        # columns in its 256-wide, half-boundary-aware chunks.  Reproduce those
        # exact GEMM launch widths and retain only each rank's intersection;
        # using the rank-local width can select a different BF16 GEMM kernel.
        direct_matmul_lhs = int(mesh.layout.shape[1]) > 1 and (
            _ring_gather_should_preallocate(
                a_local,
                int(mesh.layout.shape[1]),
                -2,
                original_n,
            )
        )
        offload_b = (
            b_local is not None
            and direct_matmul_lhs
            and a_local.device.type == "cuda"
            and b_local.device.type == "cpu"
        )
        b_local_host = None
        if offload_b:
            b_local_host = b_local
            del b_local
        a_full_k_prepared = None
        if direct_matmul_lhs:
            incoming = direction == TriangleMultiplicationDirection.INCOMING
            a_full_k_prepared = _ring_gather_pair_matmul_lhs(
                a_local,
                mesh,
                length=original_n,
                incoming=incoming,
            )
            a_full_k = (
                a_full_k_prepared.permute(2, 1, 0).unsqueeze(0)
                if incoming
                else a_full_k_prepared.permute(1, 2, 0).unsqueeze(0)
            )
        else:
            a_full_k = _ring_gather_by_row(
                a_local,
                mesh,
                dim=-2,
                length=original_n,
            )
        del a_local
        gather_full_b = (
            b_local is not None
            and not offload_b
            and side == 1
            and _one_by_p_trimul_should_gather_full_b(
                b_local,
                original_n,
            )
        )
        b_full_k = None
        if gather_full_b:
            b_full_k = _ring_gather_by_row(
                b_local,
                mesh,
                dim=-2,
                length=original_n,
            )
        if (
            b_owned_rows is None
            and b_local is not None
            and side > 1
            and direction == TriangleMultiplicationDirection.OUTGOING
            and not offload_b
            and _one_by_p_should_materialize_owned_rows(b_local)
        ):
            b_owned_rows = _one_by_p_transpose_columns_to_owned_rows(
                b_local,
                mesh,
                z_spec,
            )
        if b_full_k is not None:
            del b_local
        update = a_full_k.new_zeros(local_shape)
        for source_start, source_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(source_start, col_start)
            overlap_end = min(source_end, valid_col_end)
            collective_outgoing_fallback = (
                side > 1
                and direction == TriangleMultiplicationDirection.OUTGOING
                and b_owned_rows is None
            )
            if overlap_start >= overlap_end and not collective_outgoing_fallback:
                continue
            if b_full_k is not None:
                if direction == TriangleMultiplicationDirection.OUTGOING:
                    b_source_chunk = b_full_k[..., source_start:source_end, :, :]
                elif direction == TriangleMultiplicationDirection.INCOMING:
                    b_source_chunk = b_full_k[..., source_start:source_end, :]
                else:
                    raise ValueError(f"unsupported direction={direction}")
            elif (
                side > 1
                and direction == TriangleMultiplicationDirection.OUTGOING
                and b_owned_rows is not None
            ):
                b_source_chunk = a_full_k.new_zeros(
                    a_full_k.shape[:-3]
                    + (
                        source_end - source_start,
                        original_n,
                        b_owned_rows.shape[-1],
                    )
                )
                local_start = overlap_start - col_start
                local_end = overlap_end - col_start
                chunk_start = overlap_start - source_start
                chunk_end = overlap_end - source_start
                b_source_chunk[..., chunk_start:chunk_end, :, :].copy_(
                    b_owned_rows[..., local_start:local_end, :, :]
                )
            elif side > 1 and direction == TriangleMultiplicationDirection.INCOMING:
                b_source_chunk = _one_by_p_local_source_column_chunk(
                    b_local_host if offload_b else b_local,
                    z_spec,
                    source_start,
                    source_end,
                )
                if offload_b:
                    b_source_chunk = b_source_chunk.to(device=a_full_k.device)
            elif direction == TriangleMultiplicationDirection.OUTGOING:
                b_chunk_local = (
                    b_local_host[..., source_start:source_end, :, :].to(
                        device=a_full_k.device
                    )
                    if offload_b
                    else b_local[..., source_start:source_end, :, :]
                )
                b_source_chunk = _ring_gather_by_row(
                    b_chunk_local,
                    mesh,
                    dim=-2,
                    length=original_n,
                )
                if offload_b:
                    del b_chunk_local
            elif direction == TriangleMultiplicationDirection.INCOMING:
                b_source_chunk = _one_by_p_gather_source_columns_by_row(
                    b_local,
                    mesh,
                    z_spec,
                    source_start,
                    source_end,
                )
            else:
                raise ValueError(f"unsupported direction={direction}")
            if overlap_start >= overlap_end:
                del b_source_chunk
                continue
            if direction == TriangleMultiplicationDirection.OUTGOING:
                source_chunk = _source_unbatched_outgoing_matmul(
                    a_full_k,
                    b_source_chunk,
                    a_full_k_prepared,
                )
            elif direction == TriangleMultiplicationDirection.INCOMING:
                source_chunk = _source_unbatched_incoming_matmul(
                    a_full_k,
                    b_source_chunk,
                    a_full_k_prepared,
                )
            local_slice = slice(overlap_start - col_start, overlap_end - col_start)
            source_slice = slice(
                overlap_start - source_start, overlap_end - source_start
            )
            update[..., :original_n, local_slice, :] = source_chunk[
                ..., :original_n, source_slice, :
            ]
            del source_chunk, b_source_chunk
        del a_full_k, a_full_k_prepared
        del b_owned_rows
        if offload_b:
            del b_local_host
        elif b_full_k is None and b_local is not None:
            del b_local
        else:
            del b_full_k
        return update.contiguous()

    if direction == TriangleMultiplicationDirection.OUTGOING:
        a_full_k = _ring_gather_by_row(a_local, mesh, dim=-2, length=original_n)
        del a_local
        b_trans = _transpose_source_pair_tile(b_local, mesh)
        del b_local
        valid_row_end = min(row_end, original_n)
        valid_col_end = min(col_end, original_n)
        valid_rows = max(0, valid_row_end - row_start)
        update = a_full_k.new_zeros(local_shape)
        matmul_rows = _triangle_source_matmul_row_size(valid_rows, original_n)
        a_mat_input = a_full_k[..., :valid_rows, :, :]
        row_slice = slice(0, valid_rows)
        if matmul_rows != valid_rows:
            a_padded = a_full_k.new_zeros(
                a_full_k.shape[:-3]
                + (matmul_rows, a_full_k.shape[-2], a_full_k.shape[-1])
            )
            row_slice = (
                slice(row_start, valid_row_end)
                if matmul_rows == original_n
                else slice(0, valid_rows)
            )
            a_padded[..., row_slice, :, :] = a_mat_input
            a_mat_input = a_padded
        a_mat_input_prepared = (
            a_mat_input.squeeze(0).permute(2, 0, 1).contiguous()
            if a_mat_input.shape[0] == 1
            else None
        )
        for global_start, global_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(global_start, col_start)
            overlap_end = min(global_end, col_end, original_n)
            if overlap_start >= overlap_end:
                continue
            local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
            b_full_k = _ring_gather_by_col(
                b_trans[..., local_col_slice, :, :],
                mesh,
                dim=-2,
                length=original_n,
            )
            chunk = _source_unbatched_outgoing_matmul(
                a_mat_input,
                b_full_k,
                a_mat_input_prepared,
            )
            if matmul_rows != valid_rows:
                chunk = chunk[..., row_slice, :, :]
            update[..., :valid_rows, local_col_slice, :] = chunk
            del chunk, b_full_k
        if matmul_rows != valid_rows:
            del a_padded
        del a_mat_input, a_mat_input_prepared
        del a_full_k, b_trans
        return update.contiguous()

    if direction == TriangleMultiplicationDirection.INCOMING:
        a_trans = _transpose_source_pair_tile(a_local, mesh)
        del a_local
        valid_row_end = min(row_end, original_n)
        valid_col_end = min(col_end, original_n)
        if row_start < valid_row_end:
            a_full_k = _ring_gather_by_row(
                a_trans[..., : valid_row_end - row_start, :],
                mesh,
                dim=-3,
                length=original_n,
            )
            del a_trans
            update = a_full_k.new_zeros(local_shape)
            valid_rows = valid_row_end - row_start
            matmul_rows = _triangle_source_matmul_row_size(valid_rows, original_n)
            a_mat_input = a_full_k
            row_slice = slice(0, valid_rows)
            if matmul_rows != valid_rows:
                a_padded = a_full_k.new_zeros(
                    a_full_k.shape[:-2] + (matmul_rows, a_full_k.shape[-1])
                )
                row_slice = (
                    slice(row_start, valid_row_end)
                    if matmul_rows == original_n
                    else slice(0, valid_rows)
                )
                a_padded[..., row_slice, :] = a_full_k
                a_mat_input = a_padded
            a_mat_input_prepared = (
                a_mat_input.squeeze(0).permute(2, 1, 0).contiguous()
                if a_mat_input.shape[0] == 1
                else None
            )
            for global_start, global_end in _triangle_source_column_chunks(original_n):
                overlap_start = max(global_start, col_start)
                overlap_end = min(global_end, col_end, original_n)
                if overlap_start >= overlap_end:
                    continue
                local_col_slice = slice(
                    overlap_start - col_start, overlap_end - col_start
                )
                b_full_k = _ring_gather_by_col(
                    b_local[..., local_col_slice, :],
                    mesh,
                    dim=-3,
                    length=original_n,
                )
                chunk = _source_unbatched_incoming_matmul(
                    a_mat_input,
                    b_full_k,
                    a_mat_input_prepared,
                )
                if matmul_rows != valid_rows:
                    chunk = chunk[..., row_slice, :, :]
                update[..., :valid_rows, local_col_slice, :] = chunk
                del chunk, b_full_k
            if matmul_rows != valid_rows:
                del a_padded
            del a_mat_input, a_mat_input_prepared
            del a_full_k
        else:
            update = b_local.new_zeros(local_shape)
            del a_trans
        del b_local
        return update.contiguous()

    raise ValueError(f"unsupported direction={direction}")


def _triangle_multiplication_output_norm_gate(
    module: torch.nn.Module,
    update: torch.Tensor,
    z_norm: torch.Tensor,
    mesh: FoldCPProcessMesh | None = None,
) -> torch.Tensor:
    flat_chunk_size = int(
        os.environ.get("OPENDDE_FOLDCP_TRIMUL_OUTPUT_GATE_FLAT_CHUNK", "262144")
    )
    if flat_chunk_size <= 0:
        update = module.layer_norm_out(update)
        update = module.linear_z(update)
        update = update * torch.sigmoid(module.linear_g(z_norm))
        return update

    flat_update = update.reshape(-1, update.shape[-1])
    flat_z_norm = z_norm.reshape(-1, z_norm.shape[-1])
    c_z = int(module.c_z)
    write_inplace = (not torch.is_grad_enabled()) and update.shape[-1] == c_z
    out = (
        flat_update
        if write_inplace
        else flat_update.new_empty((flat_update.shape[0], c_z))
    )
    for start in range(0, flat_update.shape[0], flat_chunk_size):
        end = min(start + flat_chunk_size, flat_update.shape[0])
        norm_chunk = module.layer_norm_out(flat_update[start:end])
        out_chunk = module.linear_z(norm_chunk)
        gate_chunk = torch.sigmoid(module.linear_g(flat_z_norm[start:end]))
        out[start:end] = out_chunk * gate_chunk
    if write_inplace:
        return update
    return out.reshape(update.shape[:-1] + (c_z,))


def _trimul_output_source_grid_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIMUL_OUTPUT_SOURCE_GRID_MAX_BYTES")
    if value is None:
        return 0
    return int(value)


def _trimul_output_can_use_source_grid(
    update: torch.Tensor,
    z_source: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
) -> bool:
    max_bytes = _trimul_output_source_grid_max_bytes()
    if max_bytes <= 0:
        return False
    row_dim, col_dim = z_spec.pair_dims
    source_rows = int(z_spec.original_shape[row_dim]) * int(
        z_spec.original_shape[col_dim]
    )
    update_bytes = source_rows * int(update.shape[-1]) * update.element_size()
    z_bytes = source_rows * int(z_source.shape[-1]) * z_source.element_size()
    return max(update_bytes, z_bytes) <= max_bytes


def _copy_pair_tile_to_source_grid(
    tile: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    col_start: int,
    valid_rows: int,
    valid_cols: int,
) -> torch.Tensor:
    launch = tile.new_zeros((original_n, original_n, tile.shape[-1]))
    launch[
        row_start : row_start + valid_rows,
        col_start : col_start + valid_cols,
        :,
    ] = tile[:valid_rows, :valid_cols, :]
    return launch


def _triangle_multiplication_output_norm_gate_source_grid(
    module: torch.nn.Module,
    update: torch.Tensor,
    z_source: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
    *,
    source_unbatched: bool,
) -> torch.Tensor | None:
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    if valid_rows == 0 or valid_cols == 0:
        return update.new_zeros(update.shape[:-1] + (int(module.c_z),))
    if not _trimul_output_can_use_source_grid(update, z_source, z_spec):
        return None
    if update.ndim == 4:
        if not source_unbatched or update.shape[0] != 1 or z_source.shape[0] != 1:
            return None
        update_tile = update.squeeze(0)
        z_tile = z_source.squeeze(0)
        unsqueeze_batch = True
    else:
        update_tile = update
        z_tile = z_source
        unsqueeze_batch = False

    update_launch = _copy_pair_tile_to_source_grid(
        update_tile,
        original_n=original_n,
        row_start=row_start,
        col_start=col_start,
        valid_rows=valid_rows,
        valid_cols=valid_cols,
    )
    slab = module.linear_z(module.layer_norm_out(update_launch))
    del update_launch

    z_launch = _copy_pair_tile_to_source_grid(
        z_tile,
        original_n=original_n,
        row_start=row_start,
        col_start=col_start,
        valid_rows=valid_rows,
        valid_cols=valid_cols,
    )
    gate = module.linear_g(module.layer_norm_in(z_launch))
    del z_launch

    slab *= torch.sigmoid(gate)
    out_tile = slab[
        row_start:valid_row_end,
        col_start:valid_col_end,
        :,
    ].contiguous()
    del slab, gate
    if unsqueeze_batch:
        out_tile = out_tile.unsqueeze(0)

    out = update.new_zeros(update.shape[:-1] + (int(module.c_z),))
    out[..., :valid_rows, :valid_cols, :] = out_tile
    return out.contiguous()


def _paired_pair_tile_linears_with_source_chunk_launch(
    first_linear: torch.nn.Module,
    second_linear: torch.nn.Module,
    first_x: torch.Tensor,
    second_x: torch.Tensor,
    *,
    source_rows: int,
    source_cols: int,
    row_start: int = 0,
    col_start: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run two equal-shape source launches as one strided-batched GEMM."""

    if first_x.ndim != 3 or second_x.shape != first_x.shape:
        raise ValueError(
            "paired source-chunk linears expect equal [rows, cols, C] inputs."
        )
    if first_linear.bias is not None or second_linear.bias is not None:
        raise ValueError("paired source-chunk linears require bias-free modules.")
    if first_linear.weight.shape != second_linear.weight.shape:
        raise ValueError("paired source-chunk linears require equal weight shapes.")

    source_rows = int(source_rows)
    source_cols = int(source_cols)
    row_start = int(row_start)
    col_start = int(col_start)
    valid_rows = max(0, min(int(first_x.shape[0]), source_rows - row_start))
    valid_cols = max(0, min(int(first_x.shape[1]), source_cols - col_start))
    out_features = int(first_linear.weight.shape[0])
    out_shape = first_x.shape[:-1] + (out_features,)
    if valid_rows == 0 or valid_cols == 0:
        return first_x.new_zeros(out_shape), second_x.new_zeros(out_shape)

    launch_rows = source_rows * source_cols
    compact_bf16_384 = (
        os.environ.get(
            "OPENDDE_FOLDCP_COMPACT_TRIMUL_OUTPUT_GATE",
            "1",
        )
        != "0"
        and first_x.is_cuda
        and first_x.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and tuple(first_linear.weight.shape) == (384, 384)
        and launch_rows >= 90_000
        and valid_rows * valid_cols * 8 <= launch_rows * 5
    )
    if compact_bf16_384:
        first_compact = (
            first_x[:valid_rows, :valid_cols, :]
            .contiguous()
            .reshape(valid_rows * valid_cols, first_x.shape[-1])
        )
        second_compact = (
            second_x[:valid_rows, :valid_cols, :]
            .contiguous()
            .reshape(valid_rows * valid_cols, second_x.shape[-1])
        )
        first_projected = first_linear(first_compact).reshape(
            valid_rows,
            valid_cols,
            out_features,
        )
        second_projected = second_linear(second_compact).reshape(
            valid_rows,
            valid_cols,
            out_features,
        )
        if valid_rows == int(first_x.shape[0]) and valid_cols == int(first_x.shape[1]):
            return first_projected, second_projected
        first_out = first_projected.new_zeros(out_shape)
        second_out = second_projected.new_zeros(out_shape)
        first_out[:valid_rows, :valid_cols, :].copy_(first_projected)
        second_out[:valid_rows, :valid_cols, :].copy_(second_projected)
        return first_out, second_out

    launch = first_x.new_zeros((2, launch_rows, first_x.shape[-1]))
    source_index = (
        (torch.arange(valid_rows, device=first_x.device) + row_start)[:, None]
        * source_cols
        + col_start
        + torch.arange(valid_cols, device=first_x.device)[None, :]
    ).reshape(-1)
    tile_index = (
        torch.arange(valid_rows, device=first_x.device)[:, None] * int(first_x.shape[1])
        + torch.arange(valid_cols, device=first_x.device)[None, :]
    ).reshape(-1)
    launch[0].index_copy_(
        0,
        source_index,
        first_x.reshape(-1, first_x.shape[-1]).index_select(0, tile_index),
    )
    launch[1].index_copy_(
        0,
        source_index,
        second_x.reshape(-1, second_x.shape[-1]).index_select(0, tile_index),
    )
    weights = torch.stack(
        (first_linear.weight.transpose(0, 1), second_linear.weight.transpose(0, 1)),
        dim=0,
    )
    projected = torch.bmm(launch, weights).index_select(1, source_index)
    first_out = projected.new_zeros(out_shape)
    second_out = projected.new_zeros(out_shape)
    first_out.reshape(-1, out_features).index_copy_(0, tile_index, projected[0])
    second_out.reshape(-1, out_features).index_copy_(0, tile_index, projected[1])
    return first_out, second_out


def _pair_tile_linears_are_batch_compatible(
    first_linear: torch.nn.Module,
    second_linear: torch.nn.Module,
    first_x: torch.Tensor,
    second_x: torch.Tensor,
) -> bool:
    """Return whether two projections can share an unchanged batched GEMM geometry."""

    return (
        first_x.ndim == 3
        and second_x.shape == first_x.shape
        and first_linear.bias is None
        and second_linear.bias is None
        and first_linear.weight.shape == second_linear.weight.shape
    )


def _triangle_multiplication_output_norm_gate_source_slab(
    module: torch.nn.Module,
    update: torch.Tensor,
    z_source: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
    *,
    source_unbatched: bool = False,
) -> torch.Tensor:
    original_n = z_spec.original_shape[z_spec.pair_dims[0]]
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    out = update.new_zeros(update.shape[:-1] + (int(module.c_z),))
    if row_start >= valid_row_end:
        return out
    source_grid = _triangle_multiplication_output_norm_gate_source_grid(
        module,
        update,
        z_source,
        z_spec,
        source_unbatched=source_unbatched,
    )
    if source_grid is not None:
        return source_grid

    for global_start, global_end in _triangle_source_column_chunks(original_n):
        overlap_start = max(global_start, col_start)
        overlap_end = min(global_end, col_end, original_n)
        if overlap_start >= overlap_end:
            continue
        local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
        update_slab = _ring_gather_by_col(
            update[..., :, local_col_slice, :],
            mesh,
            dim=-3,
            length=original_n,
        )
        z_slab = _ring_gather_by_col(
            z_source[..., :, local_col_slice, :],
            mesh,
            dim=-3,
            length=original_n,
        )
        if source_unbatched and update_slab.ndim == 4 and update_slab.shape[0] == 1:
            slab_3d = update_slab.squeeze(0)
            z_slab_3d = z_slab.squeeze(0)
            slab_norm = module.layer_norm_out(slab_3d)
            gate_norm = module.layer_norm_in(z_slab_3d)
            if _pair_tile_linears_are_batch_compatible(
                module.linear_z,
                module.linear_g,
                slab_norm,
                gate_norm,
            ):
                slab, gate = _paired_pair_tile_linears_with_source_chunk_launch(
                    module.linear_z,
                    module.linear_g,
                    slab_norm,
                    gate_norm,
                    source_rows=original_n,
                    source_cols=global_end - global_start,
                    row_start=0,
                    col_start=overlap_start - global_start,
                )
            else:
                slab = foldcp_pair_tile_linear_with_source_chunk_launch(
                    module.linear_z,
                    slab_norm,
                    source_rows=original_n,
                    source_cols=global_end - global_start,
                    row_start=0,
                    col_start=overlap_start - global_start,
                )
                gate = foldcp_pair_tile_linear_with_source_chunk_launch(
                    module.linear_g,
                    gate_norm,
                    source_rows=original_n,
                    source_cols=global_end - global_start,
                    row_start=0,
                    col_start=overlap_start - global_start,
                )
            slab = (slab * torch.sigmoid(gate)).unsqueeze(0)
            del slab_3d, z_slab_3d
        else:
            slab_norm = module.layer_norm_out(update_slab)
            slab = foldcp_pair_tile_linear_with_source_chunk_launch(
                module.linear_z,
                slab_norm,
                source_rows=original_n,
                source_cols=global_end - global_start,
                row_start=0,
                col_start=overlap_start - global_start,
            )
            gate_norm = module.layer_norm_in(z_slab)
            gate = foldcp_pair_tile_linear_with_source_chunk_launch(
                module.linear_g,
                gate_norm,
                source_rows=original_n,
                source_cols=global_end - global_start,
                row_start=0,
                col_start=overlap_start - global_start,
            )
            slab = slab * torch.sigmoid(gate)
        out[..., : valid_row_end - row_start, local_col_slice, :] = slab[
            ..., row_start:valid_row_end, :, :
        ]
        del update_slab, z_slab, slab_norm, gate_norm, gate, slab
    return out.contiguous()


def _transpose_pair_spec(z_spec: FoldCPPairShardSpec) -> FoldCPPairShardSpec:
    row_dim, col_dim = z_spec.pair_dims
    original_shape = list(z_spec.original_shape)
    padded_shape = list(z_spec.padded_shape)
    original_shape[row_dim], original_shape[col_dim] = (
        original_shape[col_dim],
        original_shape[row_dim],
    )
    padded_shape[row_dim], padded_shape[col_dim] = (
        padded_shape[col_dim],
        padded_shape[row_dim],
    )
    return FoldCPPairShardSpec(
        original_shape=tuple(original_shape),
        padded_shape=tuple(padded_shape),
        pair_dims=z_spec.pair_dims,
        row_range=z_spec.col_range,
        col_range=z_spec.row_range,
        mesh_shape=z_spec.mesh_shape,
        mesh_coord=(z_spec.mesh_coord[1], z_spec.mesh_coord[0]),
    )


def _replicated_serial_triangle_multiplication_update(
    module: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None,
    z_spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    """Run the unchanged source inference kernel and restore local ownership."""

    z_full = gather_pair_tensor_like(z_local, z_spec, mesh.group_2d)
    mask_full = None
    if mask_local is not None:
        mask_spec = make_pair_shard_spec(
            tuple(z_spec.original_shape[:-1]),
            mesh,
            pair_dims=z_spec.pair_dims,
        )
        mask_full = gather_pair_tensor(
            mask_local,
            mask_spec,
            mesh.group_2d,
        )
    source_result = module(
        z_full,
        mask=mask_full,
        inplace_safe=True,
        _add_with_inplace=True,
        triangle_multiplicative="torch",
    )
    local_result, local_spec = shard_pair_tensor(
        source_result,
        mesh,
        pair_dims=z_spec.pair_dims,
    )
    if (
        local_spec.row_range != z_spec.row_range
        or local_spec.col_range != z_spec.col_range
    ):
        raise RuntimeError("replicated TriMul changed Fold-CP shard ownership.")
    return local_result.contiguous()


def distributed_triangle_multiplication_update(
    module: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
) -> torch.Tensor:
    """Run a real TriangleMultiplication module on a Fold-CP local pair tile.

    The OpenDDE module owns layernorm, projections, gates, and output projection.
    Fold-CP only replaces the full `sum_k` triangular BMM with the 2D ring
    contraction over sharded projected pair tiles.
    """

    if (
        z_spec is not None
        and residual_local is z_local
        and z_local.dtype == torch.bfloat16
        and not torch.is_grad_enabled()
        and torch.are_deterministic_algorithms_enabled()
        and _trimul_should_use_replicated_serial(z_local, mesh, z_spec)
    ):
        return _replicated_serial_triangle_multiplication_update(
            module,
            z_local,
            mesh,
            mask_local,
            z_spec,
        )

    if z_local.ndim == 3:
        z_in = z_local.unsqueeze(0)
        squeeze_batch = True
    elif z_local.ndim == 4:
        z_in = z_local
        squeeze_batch = False
    else:
        raise ValueError("z_local must be [N, N, C] or [B, N, N, C].")

    if mask_local is None:
        mask = None
    elif mask_local.ndim == 2:
        mask = mask_local.unsqueeze(0)
    elif mask_local.ndim == 3:
        mask = mask_local
    else:
        raise ValueError("mask_local must be [N, N] or [B, N, N].")
    if mask is not None:
        mask = mask.unsqueeze(-1)

    if z_spec is None:
        z_norm = module.layer_norm_in(z_in)
    else:
        z_norm = _triangle_layer_norm_source_row_slab(
            module.layer_norm_in,
            z_in,
            mesh,
            z_spec,
        )

    direction = (
        TriangleMultiplicationDirection.OUTGOING
        if bool(module._outgoing)
        else TriangleMultiplicationDirection.INCOMING
    )
    project_chunk_size = _trimul_project_channel_chunk_size(
        z_norm,
        z_spec,
        int(module.c_hidden),
    )
    if (
        z_spec is not None
        and residual_local is not None
        and torch.are_deterministic_algorithms_enabled()
    ):
        project_chunk_size = 0
    source_pair_rows = None
    if z_spec is not None:
        row_dim, col_dim = z_spec.pair_dims
        source_pair_rows = int(z_spec.original_shape[row_dim]) * int(
            z_spec.original_shape[col_dim]
        )

    def project_linear(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if source_pair_rows is None:
            return linear(x)
        return foldcp_linear_with_source_launch_shape(
            linear,
            x,
            source_rows=source_pair_rows,
        )

    if 0 < project_chunk_size < int(module.c_hidden):
        update = z_in.new_empty(z_in.shape[:-1] + (int(module.c_hidden),))
        for channel_start in range(0, int(module.c_hidden), project_chunk_size):
            channel_end = min(channel_start + project_chunk_size, int(module.c_hidden))
            channel_slice = slice(channel_start, channel_end)
            a_local = torch.sigmoid(
                _linear_output_slice(module.linear_a_g, z_norm, channel_slice)
            )
            a_local *= _linear_output_slice(module.linear_a_p, z_norm, channel_slice)
            if mask is not None:
                a_local *= mask
            b_local = torch.sigmoid(
                _linear_output_slice(module.linear_b_g, z_norm, channel_slice)
            )
            b_local *= _linear_output_slice(module.linear_b_p, z_norm, channel_slice)
            if mask is not None:
                b_local *= mask
            if z_spec is None:
                update[..., channel_slice] = distributed_triangle_multiplication(
                    a_local,
                    b_local,
                    mesh.ring_comm(),
                    direction,
                )
            else:
                b_matmul = (
                    b_local.cpu()
                    if _one_by_p_should_offload_trimul_b(b_local, mesh)
                    else b_local
                )
                b_was_offloaded = b_matmul is not b_local
                if b_was_offloaded:
                    del b_local
                    torch.cuda.empty_cache()
                update[..., channel_slice] = (
                    _distributed_triangle_multiplication_source_matmul(
                        a_local,
                        b_matmul,
                        mesh,
                        direction,
                        z_spec,
                    )
                )
                del b_matmul
                if not b_was_offloaded:
                    del b_local
            del a_local
            if z_spec is None:
                del b_local
        if z_spec is not None:
            del z_norm, mask
    else:
        if z_spec is None:
            a_local = torch.sigmoid(project_linear(module.linear_a_g, z_norm))
            a_local *= project_linear(module.linear_a_p, z_norm)
            if mask is not None:
                a_local *= mask
        else:
            a_local = _triangle_a_projection_source_chunks(
                module,
                z_norm,
                mask,
                z_spec,
                source_unbatched=squeeze_batch,
            )
        if z_spec is None:
            b_local = torch.sigmoid(project_linear(module.linear_b_g, z_norm))
            b_local *= project_linear(module.linear_b_p, z_norm)
            if mask is not None:
                b_local *= mask
            b_owned_rows = None
        else:
            original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
            direct_b_owned_rows = _one_by_p_should_project_b_owned_rows(
                z_norm,
                mesh,
                original_n=original_n,
                c_hidden=int(module.c_hidden),
                direction=direction,
            )
            b_projection = _triangle_b_projection_source_chunk(
                module,
                z_norm,
                mask,
                mesh,
                direction,
                z_spec,
                z_source=z_in if residual_local is not None else None,
                source_unbatched=squeeze_batch,
                return_owned_rows=direct_b_owned_rows,
            )
            b_owned_rows = b_projection if direct_b_owned_rows else None
            b_local = None if direct_b_owned_rows else b_projection
        if z_spec is None:
            update = distributed_triangle_multiplication(
                a_local,
                b_local,
                mesh.ring_comm(),
                direction,
            )
        else:
            del z_norm, mask
            if b_owned_rows is not None:
                update = _distributed_triangle_multiplication_source_matmul(
                    a_local,
                    None,
                    mesh,
                    direction,
                    z_spec,
                    b_owned_rows=b_owned_rows,
                )
                del b_owned_rows
            else:
                b_matmul = (
                    b_local.cpu()
                    if _one_by_p_should_offload_trimul_b(b_local, mesh)
                    else b_local
                )
                b_was_offloaded = b_matmul is not b_local
                if b_was_offloaded:
                    del b_local
                    torch.cuda.empty_cache()
                update = _distributed_triangle_multiplication_source_matmul(
                    a_local,
                    b_matmul,
                    mesh,
                    direction,
                    z_spec,
                )
                del b_matmul
                if not b_was_offloaded:
                    del b_local
        del a_local
        if z_spec is None:
            del b_local
    if z_spec is None:
        update = _triangle_multiplication_output_norm_gate(module, update, z_norm, mesh)
    else:
        update = _triangle_multiplication_output_norm_gate_source_slab(
            module,
            update,
            z_in,
            mesh,
            z_spec,
            source_unbatched=squeeze_batch,
        )

    if squeeze_batch:
        update = update.squeeze(0)
    if residual_local is not None:
        residual_local += update
        return residual_local.contiguous()
    return update.contiguous()


def distributed_pair_transition_update(
    transition: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
) -> torch.Tensor:
    """Run the real pair transition on a local Fold-CP pair tile.

    Pair transition is pointwise over the two token axes, so no CP communication
    is required. The memory win comes from never materializing the full
    `[N, N, C] -> [N, N, n*C]` transition workspace on one rank.
    """

    if z_spec is not None and mesh is not None and not torch.is_grad_enabled():
        original_n = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_rows = max(0, min(row_end, original_n) - row_start)
        valid_cols = max(0, min(col_end, original_n) - col_start)
        if valid_rows == 0 or valid_cols == 0:
            update = z_local.new_zeros(z_local.shape)
            if residual_local is not None:
                residual_local += update
                return residual_local.contiguous()
            return update

        flat_chunk_size = _pair_transition_source_flat_chunk_size(z_local)
        if flat_chunk_size <= 0:
            z_row_slab = _ring_gather_by_row(z_local, mesh, dim=-2, length=original_n)
            row_pad = _pair_transition_row_pad_size(valid_rows, original_n)
            source_launch_sensitive = valid_rows * original_n >= 2_097_152
            if source_launch_sensitive and (
                row_start > 0 or col_start > 0 or valid_rows > 1024
            ):
                row_pad = max(row_pad, original_n)
            launch_row_start = row_start if row_pad == original_n else 0
            if row_pad != z_row_slab.shape[-3]:
                z_padded = z_row_slab.new_zeros(
                    z_row_slab.shape[:-3]
                    + (row_pad, z_row_slab.shape[-2], z_row_slab.shape[-1])
                )
                z_padded[
                    ...,
                    launch_row_start : launch_row_start + valid_rows,
                    :,
                    :,
                ] = z_row_slab[..., :valid_rows, :, :]
            else:
                z_padded = z_row_slab
            update_row_slab = transition(z_padded)
            update = z_local.new_zeros(z_local.shape)
            update[..., :valid_rows, :valid_cols, :] = update_row_slab[
                ...,
                launch_row_start : launch_row_start + valid_rows,
                col_start : col_start + valid_cols,
                :,
            ]
            del z_row_slab, z_padded, update_row_slab
        elif all(
            int(size) == 1 for size in z_local.shape[:-3]
        ) and _pair_transition_should_use_compact_source(
            original_n=original_n,
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            flat_chunk_size=flat_chunk_size,
        ):
            update = z_local.new_zeros(z_local.shape)
            z_valid = z_local[..., :valid_rows, :valid_cols, :]
            z_valid_flat = z_valid.reshape(-1, z_local.shape[-1])
            update_valid = update[..., :valid_rows, :valid_cols, :]
            update_valid_flat = update_valid.reshape(-1, update.shape[-1])
            update_valid_copy_back = not update_valid.is_contiguous()
            if update_valid_copy_back:
                update_valid_flat = update_valid.contiguous().reshape(
                    -1,
                    update.shape[-1],
                )

            for (
                local_start,
                local_end,
                launch_rows,
            ) in _pair_transition_compact_source_launch_plan(
                original_n=original_n,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                flat_chunk_size=flat_chunk_size,
            ):
                local_count = local_end - local_start
                launch = z_local.new_zeros(
                    z_local.shape[:-3] + (launch_rows, 1, z_local.shape[-1])
                )
                launch_flat = launch.reshape(-1, launch.shape[-1])
                launch_flat[:local_count].copy_(z_valid_flat[local_start:local_end])
                projected = transition(launch).reshape(
                    -1,
                    update.shape[-1],
                )
                update_valid_flat[local_start:local_end].copy_(projected[:local_count])
                del launch, launch_flat, projected
            if update_valid_copy_back:
                update[..., :valid_rows, :valid_cols, :].copy_(
                    update_valid_flat.reshape(
                        update.shape[:-3]
                        + (
                            valid_rows,
                            valid_cols,
                            update.shape[-1],
                        )
                    )
                )
            del z_valid, z_valid_flat, update_valid, update_valid_flat
        else:
            update = z_local.new_zeros(z_local.shape)
            for (
                global_flat_start,
                global_flat_end,
            ) in _pair_transition_intersecting_global_flat_chunks(
                original_n=original_n,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                flat_chunk_size=flat_chunk_size,
            ):
                segments = _pair_transition_global_flat_chunk_segments(
                    original_n=original_n,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    global_flat_start=global_flat_start,
                    global_flat_end=global_flat_end,
                )
                if not segments:
                    continue
                chunk_rows = int(global_flat_end) - int(global_flat_start)
                launch = z_local.new_zeros(
                    z_local.shape[:-3] + (chunk_rows, 1, z_local.shape[-1])
                )
                for (
                    global_row,
                    segment_col_start,
                    segment_col_end,
                    chunk_offset,
                ) in segments:
                    local_row = int(global_row) - int(row_start)
                    local_col_start = int(segment_col_start) - int(col_start)
                    local_col_end = int(segment_col_end) - int(col_start)
                    segment_len = int(segment_col_end) - int(segment_col_start)
                    launch[..., chunk_offset : chunk_offset + segment_len, 0, :] = (
                        z_local[..., local_row, local_col_start:local_col_end, :]
                    )
                projected = transition(launch).squeeze(-2)
                for (
                    global_row,
                    segment_col_start,
                    segment_col_end,
                    chunk_offset,
                ) in segments:
                    local_row = int(global_row) - int(row_start)
                    local_col_start = int(segment_col_start) - int(col_start)
                    local_col_end = int(segment_col_end) - int(col_start)
                    segment_len = int(segment_col_end) - int(segment_col_start)
                    update[..., local_row, local_col_start:local_col_end, :] = (
                        projected[
                            ...,
                            chunk_offset : chunk_offset + segment_len,
                            :,
                        ]
                    )
                del launch, projected
        if residual_local is not None:
            residual_local += update
            return residual_local.contiguous()
        return update

    flat_chunk_size = _pair_transition_flat_chunk_size(z_local)
    if flat_chunk_size <= 0:
        update = transition(z_local)
        if residual_local is not None:
            if torch.is_grad_enabled():
                return (residual_local + update).contiguous()
            residual_local += update
            return residual_local.contiguous()
        return update.contiguous()

    flat = z_local.reshape(-1, z_local.shape[-1])
    if residual_local is not None:
        if torch.is_grad_enabled():
            out = flat.new_empty((flat.shape[0], z_local.shape[-1]))
            for start in range(0, flat.shape[0], flat_chunk_size):
                end = min(start + flat_chunk_size, flat.shape[0])
                out[start:end] = transition(flat[start:end])
            return (residual_local + out.reshape_as(z_local)).contiguous()

        residual_flat = residual_local.reshape(-1, residual_local.shape[-1])
        for start in range(0, flat.shape[0], flat_chunk_size):
            end = min(start + flat_chunk_size, flat.shape[0])
            residual_flat[start:end] += transition(flat[start:end])
        return residual_local.contiguous()

    out = flat.new_empty((flat.shape[0], z_local.shape[-1]))
    for start in range(0, flat.shape[0], flat_chunk_size):
        end = min(start + flat_chunk_size, flat.shape[0])
        out[start:end] = transition(flat[start:end])
    return out.reshape_as(z_local).contiguous()


def _gather_single_update_by_col_ring(
    local_update: torch.Tensor,
    n_token: int,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Gather row-sharded single updates without a column all-gather collective."""

    local_update = local_update.contiguous()
    side = mesh.layout.shape[0]
    if side == 1:
        return local_update[:n_token].contiguous()

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[0]] = local_update

    ready = local_update
    for step in range(1, side):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side
        gathered[source_row] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect AttentionPairBias single update.")
    full_update = torch.cat([item for item in gathered if item is not None], dim=-2)
    return full_update[:n_token].contiguous()


def _gather_single_update_by_2d_ring(
    local_update: torch.Tensor,
    n_token: int,
    mesh: FoldCPProcessMesh,
    pair_row_tile: int,
) -> torch.Tensor:
    """Gather rank-sharded single updates with Fold-CP row then column rings."""

    local_update = local_update.contiguous()
    ring = mesh.ring_comm()
    side_rows, side_cols = mesh.layout.shape

    row_chunks: list[torch.Tensor | None] = [None for _ in range(side_cols)]
    row_chunks[mesh.coord[1]] = local_update
    ready = local_update
    for step in range(1, side_cols):
        ready = ring.comm_row.exchange(ready.contiguous())
        source_col = (mesh.coord[1] + step) % side_cols
        row_chunks[source_col] = ready
    if any(item is None for item in row_chunks):
        raise RuntimeError("failed to collect single update row chunks.")
    row_block = torch.cat([item for item in row_chunks if item is not None], dim=-2)

    col_blocks: list[torch.Tensor | None] = [None for _ in range(side_rows)]
    col_blocks[mesh.coord[0]] = row_block.contiguous()
    ready = row_block
    for step in range(1, side_rows):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side_rows
        col_blocks[source_row] = ready
    if any(item is None for item in col_blocks):
        raise RuntimeError("failed to collect single update column blocks.")
    trimmed_blocks: list[torch.Tensor] = []
    for row_index, item in enumerate(col_blocks):
        if item is None:
            raise RuntimeError("failed to collect single update column block.")
        row_start = row_index * pair_row_tile
        valid_rows = max(0, min(pair_row_tile, n_token - row_start))
        trimmed_blocks.append(item[:valid_rows])
    full_update = torch.cat(trimmed_blocks, dim=-2)
    return full_update[:n_token].contiguous()


def _gather_single_rows_by_col_ring(
    local_rows: torch.Tensor,
    n_token: int,
    mesh: FoldCPProcessMesh,
    row_dim: int,
) -> torch.Tensor:
    """Gather row-sharded single-token tensors in global row order."""

    local_rows = local_rows.contiguous()
    side = mesh.layout.shape[0]
    if side == 1:
        row_dim = row_dim if row_dim >= 0 else local_rows.ndim + row_dim
        return local_rows.narrow(row_dim, 0, n_token).contiguous()

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[0]] = local_rows

    ready = local_rows
    for step in range(1, side):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side
        gathered[source_row] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect row-sharded single-token tensors.")
    full_rows = torch.cat([item for item in gathered if item is not None], dim=row_dim)
    row_dim = row_dim if row_dim >= 0 else full_rows.ndim + row_dim
    return full_rows.narrow(row_dim, 0, n_token).contiguous()


def _gather_row_blocks_by_col_ring(
    local_block: torch.Tensor,
    mesh: FoldCPProcessMesh,
    cat_dim: int,
) -> torch.Tensor:
    """Collect column-sharded row blocks in global column order."""

    local_block = local_block.contiguous()
    side = mesh.layout.shape[1]
    if side == 1:
        return local_block

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[1]] = local_block

    ready = local_block
    for step in range(1, side):
        ready = ring.comm_row.exchange(ready.contiguous())
        source_col = (mesh.coord[1] + step) % side
        gathered[source_col] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect AttentionPairBias row blocks.")
    return torch.cat(
        [item for item in gathered if item is not None], dim=cat_dim
    ).contiguous()


def _attention_pair_bias_extra_rows(
    extra_attn_bias_local: torch.Tensor | None,
    mesh: FoldCPProcessMesh,
    local_row_offset: int,
    valid_rows: int,
    n_token: int,
) -> torch.Tensor | None:
    if extra_attn_bias_local is None:
        return None
    if extra_attn_bias_local.ndim < 2:
        raise ValueError("extra attention bias must carry pair row/column dimensions.")
    extra_rows = _ring_gather_by_row(
        extra_attn_bias_local,
        mesh,
        dim=-1,
        length=n_token,
    )
    return extra_rows[
        ..., local_row_offset : local_row_offset + valid_rows, :
    ].contiguous()


def _single_update_rank_range(
    n_token: int,
    mesh: FoldCPProcessMesh,
    pair_row_tile: int,
) -> tuple[int, int, int]:
    tile = (pair_row_tile + mesh.layout.shape[1] - 1) // mesh.layout.shape[1]
    pair_row_start = mesh.coord[0] * pair_row_tile
    pair_row_end = min(pair_row_start + pair_row_tile, n_token)
    start = pair_row_start + mesh.coord[1] * tile
    end = min(start + tile, pair_row_end)
    return start, end, tile


def _attention_pair_bias_row_launch_size(valid_rows: int, original_n: int) -> int:
    """Match the serial query launch shape for BF16 bitwise parity."""

    if original_n < 3072:
        return int(original_n)
    del valid_rows
    return 112


def _attention_pair_bias_owned_chunk_segments(
    *,
    owner_start: int,
    owner_end: int,
    n_token: int,
    row_chunk_size: int,
) -> list[tuple[int, int, int, int]]:
    """Intersect one rank's rows with global, P-independent row chunks."""

    owner_start = max(0, int(owner_start))
    owner_end = min(int(owner_end), int(n_token))
    row_chunk_size = int(row_chunk_size)
    if owner_start >= owner_end or row_chunk_size <= 0:
        return []

    segments: list[tuple[int, int, int, int]] = []
    first_chunk = (owner_start // row_chunk_size) * row_chunk_size
    for chunk_start in range(first_chunk, owner_end, row_chunk_size):
        chunk_rows = min(row_chunk_size, int(n_token) - chunk_start)
        segment_start = max(owner_start, chunk_start)
        segment_end = min(owner_end, chunk_start + chunk_rows)
        if segment_start < segment_end:
            segments.append(
                (
                    chunk_start,
                    chunk_rows,
                    segment_start - chunk_start,
                    segment_end - segment_start,
                )
            )
    return segments


def distributed_attention_pair_bias_update(
    attention_pair_bias: torch.nn.Module,
    a: torch.Tensor,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec | None = None,
    extra_attn_bias_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Pairformer single attention while keeping pair bias as a CP tile."""

    if getattr(attention_pair_bias, "has_s", False):
        raise ValueError("Fold-CP AttentionPairBias currently supports has_s=False.")
    if getattr(attention_pair_bias, "cross_attention_mode", False):
        raise ValueError(
            "Fold-CP AttentionPairBias currently supports self-attention only."
        )
    if a.ndim != 2 or z_local.ndim != 3:
        raise ValueError(
            "Fold-CP AttentionPairBias expects a=[N,C] and z_local=[T,T,C]."
        )

    n_token = a.shape[-2]
    tile = z_local.shape[-3]

    a_norm = attention_pair_bias.layernorm_a(a)
    row_start, valid_row_end, single_tile = _single_update_rank_range(
        n_token,
        mesh,
        tile,
    )
    valid_rows = max(0, valid_row_end - row_start)
    local_update = a.new_zeros((single_tile, a.shape[-1]))
    pair_row_start = mesh.coord[0] * tile
    local_row_offset = row_start - pair_row_start
    extra_bias_rows = _attention_pair_bias_extra_rows(
        extra_attn_bias_local,
        mesh,
        local_row_offset,
        valid_rows,
        n_token,
    )
    q, k, v = attention_pair_bias.attention._prep_qkv(
        q_x=a_norm,
        kv_x=a_norm,
        apply_scale=True,
    )
    source_shape_out = q.new_zeros((n_token, q.shape[0], q.shape[-1]))
    row_chunk_size = _attention_pair_bias_row_launch_size(valid_rows, n_token)
    pair_valid_row_end = min(pair_row_start + tile, n_token)
    communication_segments = _attention_pair_bias_owned_chunk_segments(
        owner_start=pair_row_start,
        owner_end=pair_valid_row_end,
        n_token=n_token,
        row_chunk_size=row_chunk_size,
    )

    source_launch_bytes = (
        int(n_token)
        * int(n_token)
        * int(z_local.shape[-1])
        * int(z_local.element_size())
    )
    source_launch_budget = int(
        os.environ.get(
            "OPENDDE_FOLDCP_ATTN_PAIR_BIAS_SOURCE_GRID_MAX_BYTES",
            str(32 * 1024**3),
        )
    )
    use_chunked_source_launch = (
        source_launch_budget >= 0 and source_launch_bytes > source_launch_budget
    )
    bias_source_chunks: dict[int, torch.Tensor] = {}

    # Keep the z -> attention-bias GEMM launch identical to the ordinary
    # single-device path while the source grid fits the configured budget. For
    # larger grids, use globally aligned row chunks. Their launch geometry is
    # independent of P and avoids materializing an otherwise unsharded N^2 C_z
    # temporary on every rank.
    z_source_launch = (
        None
        if use_chunked_source_launch
        else z_local.new_zeros((n_token, n_token, z_local.shape[-1]))
    )
    for (
        chunk_start,
        chunk_rows,
        pair_chunk_offset,
        pair_segment_rows,
    ) in communication_segments:
        pair_segment_start = chunk_start + pair_chunk_offset
        slab_offset = pair_segment_start - pair_row_start
        z_rows = _ring_gather_by_row(
            z_local[slab_offset : slab_offset + pair_segment_rows],
            mesh,
            dim=-2,
            length=n_token,
        )
        segment_start = max(row_start, pair_segment_start)
        segment_end = min(valid_row_end, pair_segment_start + pair_segment_rows)
        if segment_start < segment_end:
            z_row_offset = segment_start - pair_segment_start
            z_normalized = attention_pair_bias.layernorm_z(
                z_rows[z_row_offset : z_row_offset + segment_end - segment_start]
            )
            if use_chunked_source_launch:
                chunk_offset = segment_start - chunk_start
                z_chunk_launch = z_local.new_zeros(
                    (chunk_rows, n_token, z_local.shape[-1])
                )
                z_chunk_launch[
                    chunk_offset : chunk_offset + segment_end - segment_start
                ] = z_normalized
                bias_source_chunks[chunk_start] = permute_final_dims(
                    attention_pair_bias.linear_nobias_z(z_chunk_launch),
                    [2, 0, 1],
                ).contiguous()
                del z_chunk_launch
            else:
                z_source_launch[segment_start:segment_end] = z_normalized
            del z_normalized
        del z_rows
    # Normalizing only populated rows avoids a second full [N, N, C_z]
    # temporary. LayerNorm is row-local; the following Linear is the operation
    # whose complete source shape is required for BF16 bitwise parity.
    if use_chunked_source_launch:
        bias_source = None
    else:
        bias_source = attention_pair_bias.linear_nobias_z(z_source_launch)
        del z_source_launch
        bias_source = permute_final_dims(bias_source, [2, 0, 1]).contiguous()

    for (
        chunk_start,
        chunk_rows,
        pair_chunk_offset,
        pair_segment_rows,
    ) in communication_segments:
        pair_segment_start = chunk_start + pair_chunk_offset
        pair_segment_end = pair_segment_start + pair_segment_rows
        segment_start = max(row_start, pair_segment_start)
        segment_end = min(valid_row_end, pair_segment_end)
        if segment_start >= segment_end:
            continue
        segment_rows = segment_end - segment_start
        chunk_offset = segment_start - chunk_start
        if use_chunked_source_launch:
            bias_launch = bias_source_chunks[chunk_start]
        else:
            bias_launch = bias_source[
                :, chunk_start : chunk_start + chunk_rows, :
            ].contiguous()
        if extra_bias_rows is not None:
            owner_offset = segment_start - row_start
            extra_bias = extra_bias_rows[
                ..., owner_offset : owner_offset + segment_rows, :
            ]
            if extra_bias.ndim == 2:
                extra_bias = extra_bias.unsqueeze(0)
            bias_launch[:, chunk_offset : chunk_offset + segment_rows, :] += (
                extra_bias.to(dtype=bias_launch.dtype, device=bias_launch.device)
            )

        q_launch = q.new_zeros((q.shape[0], chunk_rows, q.shape[-1]))
        q_launch[:, chunk_offset : chunk_offset + segment_rows, :] = q[
            :, segment_start:segment_end, :
        ]
        row_out_launch = _single_feature_attention(
            q=q_launch.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            attn_bias=bias_launch.contiguous(),
            use_efficient_implementation=attention_pair_bias.attention.use_efficient_implementation,
            inplace_safe=False,
        )
        row_out = row_out_launch[:, chunk_offset : chunk_offset + segment_rows, :]
        source_shape_out[segment_start:segment_end] = row_out.to(
            dtype=q.dtype
        ).transpose(-2, -3)
    del bias_source, bias_source_chunks
    source_shape_update = attention_pair_bias.attention._wrap_up(
        source_shape_out,
        a_norm,
    )
    if valid_rows > 0:
        local_update[:valid_rows] = source_shape_update[row_start:valid_row_end]
    return _gather_single_update_by_2d_ring(local_update, a.shape[-2], mesh, tile)


def _local_triangle_bias(
    triangle_attention: torch.nn.Module, x_local: torch.Tensor
) -> torch.Tensor:
    """Compute the real TriangleAttention pair bias for this local pair tile."""

    triangle_bias = permute_final_dims(triangle_attention.linear(x_local), (2, 0, 1))
    return triangle_bias.unsqueeze(-4).contiguous()


def _starting_triangle_bias_stack(
    local_triangle_bias: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Collect query-block bias tiles for this local key block via Fold-CP ring.

    In OpenDDE starting-node triangle attention, the pair bias used for output
    tile `(row_block=r, query_block=c)` comes from `z[query_block=c, key_block]`,
    not from `z[row_block=r, key_block]`. Each mesh column owns one key block, so
    all ranks in a column need the row-indexed bias stack.

    The previous implementation used a column `all_gather`, which became the M5
    2304 failure point under NCCL.  A Fold-CP column ring gathers the same stack
    with point-to-point exchanges and avoids the failing collective.
    """

    local_triangle_bias = local_triangle_bias.contiguous()
    side = mesh.layout.shape[0]
    if side == 1:
        return local_triangle_bias.unsqueeze(0).contiguous()

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[0]] = local_triangle_bias

    ready = local_triangle_bias
    for step in range(1, side):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side
        gathered[source_row] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect starting triangle bias stack.")
    return torch.stack(
        [item for item in gathered if item is not None], dim=0
    ).contiguous()


def _starting_triangle_bias_full_key_from_source_slab(
    triangle_attention: torch.nn.Module,
    x_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_n: int,
    query_start: int,
    valid_query: int,
    *,
    source_grid_launch: bool,
) -> torch.Tensor:
    """Project starting triangle bias from source-layout query-row slabs."""

    if source_grid_launch:
        row_side = mesh.layout.shape[0]
        if row_side == 1:
            x_stack = x_local.unsqueeze(0).contiguous()
        else:
            ring = mesh.ring_comm()
            gathered_x: list[torch.Tensor | None] = [None for _ in range(row_side)]
            gathered_x[mesh.coord[0]] = x_local.contiguous()
            ready_x = gathered_x[mesh.coord[0]]
            for step in range(1, row_side):
                ready_x = ring.comm_col.exchange(ready_x.contiguous())
                source_row = (mesh.coord[0] + step) % row_side
                gathered_x[source_row] = ready_x
            if any(item is None for item in gathered_x):
                raise RuntimeError(
                    "failed to collect starting triangle bias source rows."
                )
            x_stack = torch.stack(
                [item for item in gathered_x if item is not None],
                dim=0,
            ).contiguous()

        x_source_row_slab = _select_query_row_stack(
            _ring_gather_by_row(
                x_stack,
                mesh,
                dim=-2,
                length=original_n,
            ),
            mesh,
        )
        exact_source_launch = _triatt_exact_source_launch(original_n)
        source_rows = int(original_n) * int(original_n)
        source_launch_boundary = _triatt_bias_source_launch_boundary(source_rows)
        query_offset = query_start if x_source_row_slab.shape[-3] == original_n else 0
        if x_source_row_slab.shape[-3] != original_n and (
            exact_source_launch or source_launch_boundary
        ):
            x_padded = x_source_row_slab.new_zeros(
                x_source_row_slab.shape[:-3]
                + (original_n, x_source_row_slab.shape[-2], x_source_row_slab.shape[-1])
            )
            x_padded[..., query_start : query_start + valid_query, :, :] = (
                x_source_row_slab[..., :valid_query, :, :]
            )
            x_source_row_slab = x_padded
            query_offset = query_start
        linear_bias = (
            _linear_with_exact_source_launch_shape(
                triangle_attention.linear,
                x_source_row_slab,
                source_rows=source_rows,
            )
            if exact_source_launch or source_launch_boundary
            else triangle_attention.linear(x_source_row_slab)
        )
        triangle_bias = permute_final_dims(linear_bias, (2, 0, 1))
        return triangle_bias[
            :, query_offset : query_offset + valid_query, :
        ].contiguous()

    local_triangle_bias = _project_starting_triangle_bias_local_tile(
        triangle_attention,
        x_local,
        mesh,
        original_n,
    )

    row_side = mesh.layout.shape[0]
    if row_side == 1:
        triangle_bias_stack = local_triangle_bias.unsqueeze(0).contiguous()
    else:
        ring = mesh.ring_comm()
        gathered: list[torch.Tensor | None] = [None for _ in range(row_side)]
        gathered[mesh.coord[0]] = local_triangle_bias.contiguous()
        ready = gathered[mesh.coord[0]]
        for step in range(1, row_side):
            ready = ring.comm_col.exchange(ready.contiguous())
            source_row = (mesh.coord[0] + step) % row_side
            gathered[source_row] = ready
        if any(item is None for item in gathered):
            raise RuntimeError("failed to collect starting triangle bias source rows.")
        triangle_bias_stack = torch.stack(
            [item for item in gathered if item is not None],
            dim=0,
        ).contiguous()

    triangle_bias_source_row_slab = _select_query_row_stack(
        _ring_gather_by_row(
            triangle_bias_stack,
            mesh,
            dim=-1,
            length=original_n,
        ),
        mesh,
    )
    # A 1 x P shard retains all global source rows; a 2D shard's selected
    # row slab already begins at the query owner's row block.
    query_offset = (
        query_start if triangle_bias_source_row_slab.shape[-2] == original_n else 0
    )
    return triangle_bias_source_row_slab[
        :, query_offset : query_offset + valid_query, :
    ].contiguous()


def _project_starting_triangle_bias_local_tile(
    triangle_attention: torch.nn.Module,
    x_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_n: int,
) -> torch.Tensor:
    """Project local starting-triangle bias before cross-rank gathering.

    The serial TriangleAttention path projects the full normalized pair tensor
    with ``TriangleAttention.linear`` and then permutes the result to
    ``[heads, rows, cols]``.  For Fold-CP we keep the same source launch family
    for every owned pair entry, but gather only the projected head-bias channels.
    This avoids moving a full ``C_z`` source row slab through both ring gathers.
    """

    tile_rows = int(x_local.shape[-3])
    tile_cols = int(x_local.shape[-2])
    owner_row_start = int(mesh.coord[0]) * tile_rows
    owner_col_start = int(mesh.coord[1]) * tile_cols
    valid_rows = max(0, min(tile_rows, int(original_n) - owner_row_start))
    valid_cols = max(0, min(tile_cols, int(original_n) - owner_col_start))
    exact_source_launch = _triatt_exact_source_launch(original_n)
    source_rows = int(original_n) * int(original_n)
    source_launch_boundary = _triatt_bias_source_launch_boundary(source_rows)

    if exact_source_launch or source_launch_boundary:
        linear_bias = _linear_pair_tile_with_source_grid_launch(
            triangle_attention.linear,
            x_local,
            original_n=original_n,
            row_start=owner_row_start,
            col_start=owner_col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
    elif x_local.dtype == torch.bfloat16 and not torch.is_grad_enabled():
        # For large inference-only source grids, padding a local tile back to
        # [valid_rows, original_n, C] recreates an O(N^2 C) buffer on every
        # rank. The projection is row-local, so project only owned valid values
        # and retain the padded local output layout. This branch is deliberately
        # limited to BF16 inference; exact audited source-launch boundaries and
        # training/autograd continue through the established launch policies.
        out_features = int(triangle_attention.linear.weight.shape[0])
        linear_bias = x_local.new_zeros((tile_rows, tile_cols, out_features))
        if valid_rows > 0 and valid_cols > 0:
            projected = triangle_attention.linear(
                x_local[:valid_rows, :valid_cols].contiguous()
            )
            linear_bias[:valid_rows, :valid_cols].copy_(projected)
    else:
        linear_bias = foldcp_pair_tile_linear_with_source_chunk_launch(
            triangle_attention.linear,
            x_local,
            source_rows=valid_rows,
            source_cols=original_n,
            row_start=0,
            col_start=owner_col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
    return permute_final_dims(linear_bias, (2, 0, 1)).contiguous()


def _wrap_up_triangle_attention_output(
    mha: torch.nn.Module,
    out_by_row_head_query: torch.Tensor,
    x_local: torch.Tensor,
    residual_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run TriangleAttention MHA wrap-up in row chunks.

    If ``residual_local`` is provided, add each wrapped row chunk directly into
    that tensor. This avoids materializing a second full local ``N^2 x C`` tile
    just to immediately add it back to the Pairformer residual.
    """

    row_chunk_size = _triatt_wrap_row_chunk_size(out_by_row_head_query, x_local)
    if row_chunk_size <= 0 or out_by_row_head_query.shape[0] <= row_chunk_size:
        wrapped = mha._wrap_up(out_by_row_head_query.transpose(-2, -3), x_local)
        if residual_local is not None:
            residual_local += wrapped
            return residual_local.contiguous()
        return wrapped.contiguous()

    wrapped = (
        residual_local
        if residual_local is not None
        else x_local.new_empty(x_local.shape)
    )
    for row_start in range(0, out_by_row_head_query.shape[0], row_chunk_size):
        row_end = min(row_start + row_chunk_size, out_by_row_head_query.shape[0])
        row_slice = slice(row_start, row_end)
        o_chunk = out_by_row_head_query[row_slice].transpose(-2, -3)
        x_chunk = x_local[row_slice]
        if mha.linear_g is not None:
            g = mha.sigmoid(mha.linear_g(x_chunk))
            g = g.view(g.shape[:-1] + (mha.no_heads, -1))
            o_chunk = o_chunk * g
        o_chunk = o_chunk.reshape(o_chunk.shape[:-2] + (-1,))
        update_chunk = mha.linear_o(o_chunk)
        if residual_local is not None:
            wrapped[row_slice] += update_chunk
        else:
            wrapped[row_slice] = update_chunk
    return wrapped.contiguous()


def _triatt_qkv_source_rows(original_n: int) -> int | None:
    del original_n
    return None


def _triatt_exact_source_launch(original_n: int) -> bool:
    source_rows = int(original_n) * int(original_n)
    return _triatt_bias_source_launch_boundary(source_rows)


def _triatt_wrap_source_grid_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIATT_WRAP_SOURCE_GRID_MAX_BYTES")
    if value is None:
        return 16 * 1024 * 1024 * 1024
    return int(value)


def _triatt_projection_source_grid_max_bytes() -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIATT_PROJECTION_SOURCE_GRID_MAX_BYTES")
    if value is None:
        return 2 * 1024 * 1024 * 1024
    return int(value)


def _triatt_projection_source_grid_launch(
    original_n: int,
    c_in: int,
    element_size: int,
) -> bool:
    if _triatt_exact_source_launch(original_n):
        return True
    max_bytes = _triatt_projection_source_grid_max_bytes()
    if max_bytes <= 0:
        return False
    source_bytes = int(original_n) * int(original_n) * int(c_in) * int(element_size)
    return source_bytes <= max_bytes


def _triatt_projection_source_grid_for_callsite(
    original_n: int,
    c_in: int,
    element_size: int,
    serial_chunk_size: int | None,
) -> bool:
    """Select one projection launch family for the complete attention callsite."""

    if serial_chunk_size is not None:
        return False
    return _triatt_projection_source_grid_launch(original_n, c_in, element_size)


def _triatt_wrap_source_grid_launch(
    original_n: int,
    c_in: int,
    element_size: int,
) -> bool:
    source_rows = int(original_n) * int(original_n)
    if _triatt_exact_source_launch(original_n) or _triatt_bias_source_launch_boundary(
        source_rows
    ):
        return True
    max_bytes = _triatt_wrap_source_grid_max_bytes()
    if max_bytes <= 0:
        return False
    launch_bytes = source_rows * int(c_in) * int(element_size)
    return launch_bytes <= max_bytes


def _prep_triangle_attention_qkv_chunks(
    mha: torch.nn.Module,
    q_x: torch.Tensor,
    kv_x: torch.Tensor,
    apply_scale: bool = True,
    source_rows: int | None = None,
    exact_source_launch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a TriangleAttention row/query chunk without full-tile QKV."""

    if source_rows is None:
        q = mha.linear_q(q_x)
        k = mha.linear_k(kv_x)
        v = mha.linear_v(kv_x)
    elif exact_source_launch:
        q = _linear_with_exact_source_launch_shape(
            mha.linear_q,
            q_x,
            source_rows=source_rows,
        )
        k = _linear_with_exact_source_launch_shape(
            mha.linear_k,
            kv_x,
            source_rows=source_rows,
        )
        v = _linear_with_exact_source_launch_shape(
            mha.linear_v,
            kv_x,
            source_rows=source_rows,
        )
    else:
        q = foldcp_linear_with_source_launch_shape(
            mha.linear_q,
            q_x,
            source_rows=source_rows,
        )
        k = foldcp_linear_with_source_launch_shape(
            mha.linear_k,
            kv_x,
            source_rows=source_rows,
        )
        v = foldcp_linear_with_source_launch_shape(
            mha.linear_v,
            kv_x,
            source_rows=source_rows,
        )

    q = q.view(q.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)

    if apply_scale:
        q = q / (float(mha.c_hidden) ** 0.5)

    return q.contiguous(), k.contiguous(), v.contiguous()


def _prep_triangle_attention_qkv_source_chunk_chunks(
    mha: torch.nn.Module,
    x_row_chunk: torch.Tensor,
    *,
    original_n: int,
    source_rows: int,
    row_start: int,
    apply_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_rows = x_row_chunk.shape[-3]
    q = foldcp_pair_tile_linear_with_source_chunk_launch(
        mha.linear_q,
        x_row_chunk,
        source_rows=source_rows,
        source_cols=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
    )
    k = foldcp_pair_tile_linear_with_source_chunk_launch(
        mha.linear_k,
        x_row_chunk,
        source_rows=source_rows,
        source_cols=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
    )
    v = foldcp_pair_tile_linear_with_source_chunk_launch(
        mha.linear_v,
        x_row_chunk,
        source_rows=source_rows,
        source_cols=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
    )
    q = q.view(q.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    if apply_scale:
        q = q / (float(mha.c_hidden) ** 0.5)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _triatt_batched_no_bias_linear(
    linear: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run equal-shape independent Linear calls as one strided batched GEMM."""

    if linear.bias is not None:
        raise ValueError("canonical TriangleAttention batch requires bias-free Linear.")
    batch = int(x.shape[0])
    flat_rows = int(x.numel() // (batch * x.shape[-1]))
    flat = x.contiguous().reshape(batch, flat_rows, x.shape[-1])
    weight = linear.weight.transpose(0, 1).unsqueeze(0).expand(batch, -1, -1)
    projected = torch.bmm(flat, weight)
    return projected.reshape(*x.shape[:-1], projected.shape[-1])


def _triatt_batched_qkv_linears(
    mha: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project Q/K/V for equal inputs with one strided-batched GEMM launch."""

    linears = (mha.linear_q, mha.linear_k, mha.linear_v)
    if any(linear.bias is not None for linear in linears):
        raise ValueError("batched TriangleAttention QKV requires bias-free linears.")
    weight_shape = linears[0].weight.shape
    if any(linear.weight.shape != weight_shape for linear in linears[1:]):
        raise ValueError("batched TriangleAttention QKV requires equal weight shapes.")

    batch = int(x.shape[0])
    flat_rows = int(x.numel() // (batch * x.shape[-1]))
    flat = x.contiguous().reshape(batch, flat_rows, x.shape[-1])
    inputs = (
        flat[:, None, :, :]
        .expand(batch, 3, flat_rows, flat.shape[-1])
        .reshape(batch * 3, flat_rows, flat.shape[-1])
    )
    weights = (
        torch.stack(
            tuple(linear.weight.transpose(0, 1) for linear in linears),
            dim=0,
        )[None, :, :, :]
        .expand(batch, 3, -1, -1)
        .reshape(batch * 3, weight_shape[1], weight_shape[0])
    )
    projected = torch.bmm(inputs, weights).reshape(
        batch,
        3,
        flat_rows,
        weight_shape[0],
    )
    output_shape = x.shape[:-1] + (weight_shape[0],)
    return tuple(projected[:, index].reshape(output_shape) for index in range(3))


def _prep_triangle_attention_qkv_canonical_batch(
    mha: torch.nn.Module,
    x_row_batch: torch.Tensor,
    *,
    canonical_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(x_row_batch.shape[0] // int(canonical_rows))
    x = x_row_batch.reshape(
        batch,
        int(canonical_rows),
        x_row_batch.shape[-2],
        x_row_batch.shape[-1],
    )
    q, k, v = _triatt_batched_qkv_linears(mha, x)
    output_shape = (
        x_row_batch.shape[0],
        x_row_batch.shape[-2],
        mha.no_heads,
        -1,
    )
    q = q.reshape(output_shape).transpose(-2, -3)
    k = k.reshape(output_shape).transpose(-2, -3)
    v = v.reshape(output_shape).transpose(-2, -3)
    q = q / (float(mha.c_hidden) ** 0.5)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _wrap_up_triangle_attention_canonical_batch(
    mha: torch.nn.Module,
    out_by_row_head_query: torch.Tensor,
    x_query: torch.Tensor,
    *,
    canonical_rows: int,
) -> torch.Tensor:
    batch = int(out_by_row_head_query.shape[0] // int(canonical_rows))
    rows = int(canonical_rows)
    query_rows = int(out_by_row_head_query.shape[-2])
    out = out_by_row_head_query.reshape(
        batch,
        rows,
        out_by_row_head_query.shape[1],
        query_rows,
        out_by_row_head_query.shape[-1],
    ).transpose(-2, -3)
    x = x_query.reshape(
        batch,
        rows,
        query_rows,
        x_query.shape[-1],
    )
    if mha.linear_g is not None:
        gate = mha.sigmoid(_triatt_batched_no_bias_linear(mha.linear_g, x)).reshape(
            batch,
            rows,
            query_rows,
            mha.no_heads,
            -1,
        )
        out = out * gate
    out = out.reshape(batch, rows, query_rows, -1)
    out = _triatt_batched_no_bias_linear(mha.linear_o, out)
    return out.reshape(
        batch * rows,
        query_rows,
        out.shape[-1],
    )


@contextmanager
def foldcp_triatt_canonical_batch_scope(enabled: bool) -> Iterator[None]:
    """Temporarily allow or suppress the canonical-row batching fast path."""

    token = _TRIATT_CANONICAL_BATCH_SCOPE_ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _TRIATT_CANONICAL_BATCH_SCOPE_ENABLED.reset(token)


def _triatt_canonical_batch_enabled() -> bool:
    return _TRIATT_CANONICAL_BATCH_SCOPE_ENABLED.get()


def _prep_triangle_attention_qkv_source_grid_chunks(
    mha: torch.nn.Module,
    x_row_chunk: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    apply_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project QKV with the pre-streaming source-grid launch geometry."""

    valid_rows = x_row_chunk.shape[-3]
    q = _linear_pair_tile_with_source_grid_launch(
        mha.linear_q,
        x_row_chunk,
        original_n=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
        output_chunk_size=0,
    )
    k = _linear_pair_tile_with_source_grid_launch(
        mha.linear_k,
        x_row_chunk,
        original_n=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
        output_chunk_size=0,
    )
    v = _linear_pair_tile_with_source_grid_launch(
        mha.linear_v,
        x_row_chunk,
        original_n=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
        output_chunk_size=0,
    )
    q = q.view(q.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    if apply_scale:
        q = q / (float(mha.c_hidden) ** 0.5)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _wrap_up_triangle_attention_output_chunk(
    mha: torch.nn.Module,
    out_by_row_head_query: torch.Tensor,
    x_chunk: torch.Tensor,
    *,
    source_grid_wrap_launch: bool = False,
    source_chunk_wrap_launch: bool = False,
    preserve_source_projection: bool = False,
    original_n: int | None = None,
    row_start: int = 0,
    col_start: int = 0,
    valid_rows: int | None = None,
    valid_query: int | None = None,
    source_chunk_rows: int | None = None,
    source_chunk_row_start: int = 0,
) -> torch.Tensor:
    """Run TriangleAttention MHA wrap-up for one row/query chunk."""

    o_chunk = out_by_row_head_query.transpose(-2, -3)
    if mha.linear_g is not None:
        if source_grid_wrap_launch:
            if original_n is None or valid_rows is None or valid_query is None:
                raise ValueError(
                    "source-grid TriangleAttention wrap requires shape metadata."
                )
            g = mha.sigmoid(
                _linear_pair_tile_with_source_grid_launch(
                    mha.linear_g,
                    x_chunk,
                    original_n=original_n,
                    row_start=row_start,
                    col_start=col_start,
                    valid_rows=valid_rows,
                    valid_cols=valid_query,
                    output_chunk_size=0 if preserve_source_projection else None,
                )
            )
        elif source_chunk_wrap_launch:
            if (
                original_n is None
                or valid_rows is None
                or valid_query is None
                or source_chunk_rows is None
            ):
                raise ValueError(
                    "source-chunk TriangleAttention wrap requires shape metadata."
                )
            g = mha.sigmoid(
                foldcp_pair_tile_linear_with_source_chunk_launch(
                    mha.linear_g,
                    x_chunk,
                    source_rows=source_chunk_rows,
                    source_cols=original_n,
                    row_start=source_chunk_row_start,
                    col_start=col_start,
                    valid_rows=valid_rows,
                    valid_cols=valid_query,
                )
            )
        else:
            g = mha.sigmoid(mha.linear_g(x_chunk))
        g = g.view(g.shape[:-1] + (mha.no_heads, -1))
        o_chunk = o_chunk * g
    o_chunk = o_chunk.reshape(o_chunk.shape[:-2] + (-1,))
    if source_grid_wrap_launch:
        if original_n is None or valid_rows is None or valid_query is None:
            raise ValueError(
                "source-grid TriangleAttention wrap requires shape metadata."
            )
        return _linear_pair_tile_with_source_grid_launch(
            mha.linear_o,
            o_chunk,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_query,
            output_chunk_size=0 if preserve_source_projection else None,
        )
    if source_chunk_wrap_launch:
        if (
            original_n is None
            or valid_rows is None
            or valid_query is None
            or source_chunk_rows is None
        ):
            raise ValueError(
                "source-chunk TriangleAttention wrap requires shape metadata."
            )
        return foldcp_pair_tile_linear_with_source_chunk_launch(
            mha.linear_o,
            o_chunk,
            source_rows=source_chunk_rows,
            source_cols=original_n,
            row_start=source_chunk_row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_query,
        )
    return mha.linear_o(o_chunk)


def _distributed_triangle_attention_starting_update(
    triangle_attention: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
    triangle_bias_full_key_override: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask_local is None:
        mask = z_local.new_ones(z_local.shape[:-1])
    else:
        mask = mask_local

    if z_spec is None:
        x_local = triangle_attention.layer_norm(z_local)
        original_n = _ring_gather_by_row(x_local, mesh, dim=-2).shape[-2]
        row_start = mesh.coord[0] * x_local.shape[-3]
        col_start = mesh.coord[1] * x_local.shape[-2]
        valid_rows = x_local.shape[-3]
        valid_query = x_local.shape[-2]
    else:
        x_local = _triangle_layer_norm_source_row_slab(
            triangle_attention.layer_norm,
            z_local,
            mesh,
            z_spec,
        )
        original_n = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_rows = max(0, min(row_end, original_n) - row_start)
        valid_query = max(0, min(col_end, original_n) - col_start)

    out_local = (
        residual_local
        if residual_local is not None
        else x_local.new_zeros(x_local.shape)
    )
    if valid_rows == 0 or valid_query == 0:
        return out_local.contiguous()

    bias_source_grid_launch = _triatt_projection_source_grid_launch(
        original_n,
        x_local.shape[-1],
        x_local.element_size(),
    )
    projection_source_grid_launch = _triatt_projection_source_grid_for_callsite(
        original_n,
        x_local.shape[-1],
        x_local.element_size(),
        chunk_size,
    )

    triangle_bias_full_key = triangle_bias_full_key_override
    if triangle_bias_full_key is None:
        triangle_bias_full_key = _starting_triangle_bias_full_key_from_source_slab(
            triangle_attention,
            x_local,
            mesh,
            original_n,
            col_start,
            valid_query,
            source_grid_launch=bias_source_grid_launch,
        )

    exact_source_launch = projection_source_grid_launch and _triatt_exact_source_launch(
        original_n
    )
    query_pad = (
        original_n if exact_source_launch else _triatt_query_pad_size(valid_query)
    )
    local_query = slice(col_start, col_start + valid_query)
    triangle_bias = triangle_bias_full_key.new_zeros(
        (triangle_bias_full_key.shape[0], query_pad, original_n)
    )
    if exact_source_launch:
        triangle_bias[:, local_query, :] = triangle_bias_full_key[:, :valid_query, :]
    else:
        triangle_bias[:, :valid_query, :] = triangle_bias_full_key[:, :valid_query, :]
    del triangle_bias_full_key

    collective_query_width = _triatt_collective_query_width(
        valid_query=valid_query,
        physical_query_width=x_local.shape[-2],
        mesh_cols=mesh.layout.shape[1],
    )
    row_chunk_size = _triatt_attention_row_chunk_size(
        valid_rows,
        original_n,
        chunk_size,
        valid_query=collective_query_width,
        element_size=x_local.element_size(),
    )
    valid_row_start = 0
    while valid_row_start < valid_rows:
        global_row_start = row_start + valid_row_start
        source_chunk_rows = row_chunk_size
        source_chunk_row_start = 0
        rows_until_source_chunk_end = row_chunk_size
        if chunk_size is not None:
            (
                source_chunk_rows,
                source_chunk_row_start,
                rows_until_source_chunk_end,
            ) = _triatt_source_chunk_geometry(
                global_row_start,
                original_n,
                chunk_size,
            )
        canonical_batch_rows = (
            2 * int(chunk_size) if chunk_size is not None and int(chunk_size) > 0 else 0
        )
        can_run_canonical_batch = (
            _triatt_canonical_batch_enabled()
            and not exact_source_launch
            and not torch.is_grad_enabled()
            and x_local.dtype == torch.bfloat16
            and mesh.layout.shape[0] == 1
            and mesh.layout.shape[1] > 1
            and chunk_size is not None
            and int(chunk_size) > 0
            and row_chunk_size == int(chunk_size)
            and source_chunk_row_start == 0
            and rows_until_source_chunk_end == int(chunk_size)
            and valid_rows - valid_row_start >= canonical_batch_rows
        )
        if can_run_canonical_batch:
            valid_row_end = valid_row_start + canonical_batch_rows
            x_row_batch = _ring_gather_by_row(
                x_local[valid_row_start:valid_row_end],
                mesh,
                dim=-2,
                length=original_n,
            )
            mask_row_batch = _ring_gather_by_row(
                mask[valid_row_start:valid_row_end],
                mesh,
                dim=-1,
                length=original_n,
            )
            q_batch, k_batch, v_batch = _prep_triangle_attention_qkv_canonical_batch(
                triangle_attention.mha,
                x_row_batch,
                canonical_rows=int(chunk_size),
            )
            q_chunk = q_batch.new_zeros(
                (
                    canonical_batch_rows,
                    q_batch.shape[1],
                    query_pad,
                    q_batch.shape[-1],
                )
            )
            q_chunk[:, :, :valid_query, :] = q_batch[
                :,
                :,
                local_query,
                :,
            ]
            mask_bias = (triangle_attention.inf * (mask_row_batch - 1))[
                :, None, None, :
            ]
            # Keep the score/softmax workspace at the original canonical-row
            # size. QKV and wrap-up still use batched GEMMs, while the largest
            # attention temporary never doubles.
            attention_parts = []
            for canonical_start in range(
                0,
                canonical_batch_rows,
                int(chunk_size),
            ):
                canonical_end = canonical_start + int(chunk_size)
                score_bytes_per_head = (
                    int(chunk_size)
                    * int(q_chunk.shape[-2])
                    * int(k_batch.shape[-2])
                    * int(q_chunk.element_size())
                    * 2
                )
                head_ranges = _triatt_attention_head_ranges(
                    q_chunk.shape[1],
                    score_bytes_per_head=score_bytes_per_head,
                )
                if len(head_ranges) > 1:
                    out_heads = []
                    for head_start, head_end in head_ranges:
                        out_heads.append(
                            _foldcp_attention(
                                q_chunk[
                                    canonical_start:canonical_end,
                                    head_start:head_end,
                                ].contiguous(),
                                k_batch[
                                    canonical_start:canonical_end,
                                    head_start:head_end,
                                ].contiguous(),
                                v_batch[
                                    canonical_start:canonical_end,
                                    head_start:head_end,
                                ].contiguous(),
                                [
                                    mask_bias[
                                        canonical_start:canonical_end
                                    ].contiguous(),
                                    triangle_bias[
                                        None,
                                        head_start:head_end,
                                    ].contiguous(),
                                ],
                            )
                        )
                    attention_parts.append(torch.cat(out_heads, dim=1))
                    del out_heads
                else:
                    attention_parts.append(
                        _foldcp_attention(
                            q_chunk[canonical_start:canonical_end].contiguous(),
                            k_batch[canonical_start:canonical_end].contiguous(),
                            v_batch[canonical_start:canonical_end].contiguous(),
                            [
                                mask_bias[canonical_start:canonical_end].contiguous(),
                                triangle_bias.unsqueeze(0).contiguous(),
                            ],
                        )
                    )
            out_chunk = torch.cat(attention_parts, dim=0)
            del attention_parts
            update = _wrap_up_triangle_attention_canonical_batch(
                triangle_attention.mha,
                out_chunk[:, :, :valid_query, :].to(dtype=x_local.dtype),
                x_row_batch[:, local_query, :].contiguous(),
                canonical_rows=int(chunk_size),
            )[:, :valid_query]
            if residual_local is not None:
                out_local[
                    valid_row_start:valid_row_end,
                    :valid_query,
                ] += update
            else:
                out_local[
                    valid_row_start:valid_row_end,
                    :valid_query,
                ] = update
            del q_batch, k_batch, v_batch, q_chunk, mask_bias
            del x_row_batch, mask_row_batch, out_chunk, update
            valid_row_start = valid_row_end
            continue
        valid_row_end = min(
            valid_row_start + row_chunk_size,
            valid_row_start + rows_until_source_chunk_end,
            valid_rows,
        )
        current_rows = valid_row_end - valid_row_start
        x_row_chunk = _ring_gather_by_row(
            x_local[valid_row_start:valid_row_end],
            mesh,
            dim=-2,
            length=original_n,
        )
        mask_row_chunk = _ring_gather_by_row(
            mask[valid_row_start:valid_row_end],
            mesh,
            dim=-1,
            length=original_n,
        )
        source_grid_qkv = not exact_source_launch and projection_source_grid_launch
        source_chunk_qkv = not exact_source_launch and not source_grid_qkv
        source_grid_wrap_launch = (
            chunk_size is None
            and _triatt_wrap_source_grid_launch(
                original_n,
                x_local.shape[-1],
                x_local.element_size(),
            )
        )
        source_chunk_wrap_launch = chunk_size is not None
        qkv_row_pad = (
            original_n
            if exact_source_launch
            else _triatt_qkv_row_pad_size(current_rows, original_n)
        )
        streamed_row_launch = row_chunk_size < valid_rows
        row_pad = (
            current_rows
            if source_chunk_qkv or (source_grid_qkv and streamed_row_launch)
            else qkv_row_pad
        )
        launch_row_start = row_start + valid_row_start if exact_source_launch else 0
        if source_grid_qkv:
            x_row_source = x_row_chunk
            q_row, k_row, v_row = _prep_triangle_attention_qkv_source_grid_chunks(
                triangle_attention.mha,
                x_row_chunk,
                original_n=original_n,
                row_start=row_start + valid_row_start,
            )
        elif source_chunk_qkv:
            x_row_source = x_row_chunk
            q_row, k_row, v_row = _prep_triangle_attention_qkv_source_chunk_chunks(
                triangle_attention.mha,
                x_row_chunk,
                original_n=original_n,
                source_rows=source_chunk_rows,
                row_start=source_chunk_row_start,
            )
        else:
            if qkv_row_pad != current_rows:
                x_row_source = x_row_chunk.new_zeros(
                    (qkv_row_pad, x_row_chunk.shape[-2], x_row_chunk.shape[-1])
                )
                x_row_source[
                    launch_row_start : launch_row_start + current_rows,
                    :,
                    :,
                ] = x_row_chunk
            else:
                x_row_source = x_row_chunk
            q_row, k_row, v_row = _prep_triangle_attention_qkv_chunks(
                triangle_attention.mha,
                x_row_source,
                x_row_source,
                apply_scale=True,
                source_rows=_triatt_qkv_source_rows(original_n),
                exact_source_launch=exact_source_launch,
            )

        q_chunk = q_row.new_zeros((row_pad, q_row.shape[1], query_pad, q_row.shape[3]))
        k_chunk = k_row.new_zeros((row_pad, k_row.shape[1], original_n, k_row.shape[3]))
        v_chunk = v_row.new_zeros((row_pad, v_row.shape[1], original_n, v_row.shape[3]))
        x_chunk = x_row_chunk.new_zeros((row_pad, query_pad, x_row_chunk.shape[-1]))
        mask_bias = mask_row_chunk.new_zeros((row_pad, 1, 1, original_n))

        if exact_source_launch:
            row_slice = slice(launch_row_start, launch_row_start + current_rows)
            q_chunk[row_slice, :, local_query, :] = q_row[row_slice, :, local_query, :]
            k_chunk[row_slice] = k_row[row_slice]
            v_chunk[row_slice] = v_row[row_slice]
            x_chunk[row_slice, local_query, :] = x_row_source[row_slice, local_query, :]
            mask_bias[row_slice] = (triangle_attention.inf * (mask_row_chunk - 1))[
                :, None, None, :
            ]
        else:
            q_chunk[:current_rows, :, :valid_query, :] = q_row[
                launch_row_start : launch_row_start + current_rows,
                :,
                local_query,
                :,
            ]
            k_chunk[:current_rows] = k_row[
                launch_row_start : launch_row_start + current_rows
            ]
            v_chunk[:current_rows] = v_row[
                launch_row_start : launch_row_start + current_rows
            ]
            x_chunk[:current_rows, :valid_query, :] = x_row_source[
                launch_row_start : launch_row_start + current_rows,
                local_query,
                :,
            ]
            mask_bias[:current_rows] = (triangle_attention.inf * (mask_row_chunk - 1))[
                :, None, None, :
            ]

        score_bytes_per_head = (
            int(q_chunk.shape[0])
            * int(q_chunk.shape[-2])
            * int(k_chunk.shape[-2])
            * int(q_chunk.element_size())
            * 2
        )
        head_ranges = _triatt_attention_head_ranges(
            q_chunk.shape[1],
            score_bytes_per_head=score_bytes_per_head,
        )
        if len(head_ranges) > 1:
            out_heads = []
            for head_start, head_end in head_ranges:
                out_heads.append(
                    _foldcp_attention(
                        q_chunk[:, head_start:head_end].contiguous(),
                        k_chunk[:, head_start:head_end].contiguous(),
                        v_chunk[:, head_start:head_end].contiguous(),
                        [
                            mask_bias.contiguous(),
                            triangle_bias[None, head_start:head_end].contiguous(),
                        ],
                    )
                )
            out_chunk = torch.cat(out_heads, dim=1)
            del out_heads
        else:
            out_chunk = _foldcp_attention(
                q_chunk.contiguous(),
                k_chunk.contiguous(),
                v_chunk.contiguous(),
                [
                    mask_bias.contiguous(),
                    triangle_bias.unsqueeze(0).contiguous(),
                ],
            )
        if exact_source_launch:
            out_for_wrap = out_chunk[
                launch_row_start : launch_row_start + current_rows,
                :,
                local_query,
                :,
            ]
            x_for_wrap = x_chunk[
                launch_row_start : launch_row_start + current_rows,
                local_query,
                :,
            ]
        else:
            out_for_wrap = out_chunk
            x_for_wrap = x_chunk
        update = _wrap_up_triangle_attention_output_chunk(
            triangle_attention.mha,
            out_for_wrap.to(dtype=x_local.dtype),
            x_for_wrap.contiguous(),
            source_grid_wrap_launch=source_grid_wrap_launch,
            source_chunk_wrap_launch=source_chunk_wrap_launch,
            preserve_source_projection=projection_source_grid_launch,
            original_n=original_n,
            row_start=row_start + valid_row_start,
            col_start=col_start,
            valid_rows=current_rows,
            valid_query=valid_query,
            source_chunk_rows=source_chunk_rows,
            source_chunk_row_start=source_chunk_row_start,
        )[:current_rows, :valid_query]
        if residual_local is not None:
            out_local[valid_row_start:valid_row_end, :valid_query] += update
        else:
            out_local[valid_row_start:valid_row_end, :valid_query] = update
        del q_row, k_row, v_row, q_chunk, k_chunk, v_chunk
        del (
            x_row_chunk,
            mask_row_chunk,
            x_row_source,
            x_chunk,
            mask_bias,
            out_chunk,
            update,
        )
        valid_row_start = valid_row_end
    return out_local.contiguous()


def _project_one_by_p_ending_triangle_bias_local(
    triangle_attention: torch.nn.Module,
    x_transposed: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    valid_rows: int,
    chunk_size: int | None,
) -> torch.Tensor:
    """Project ending-attention bias without a full source grid for large N."""

    use_source_grid = _triatt_projection_source_grid_for_callsite(
        original_n,
        x_transposed.shape[-1],
        x_transposed.element_size(),
        chunk_size,
    )
    if use_source_grid:
        projected = _linear_pair_tile_with_source_grid_launch(
            triangle_attention.linear,
            x_transposed,
            original_n=original_n,
            row_start=row_start,
            col_start=0,
            valid_rows=valid_rows,
            valid_cols=original_n,
        )
        return permute_final_dims(projected, (2, 0, 1)).contiguous()

    out_features = int(triangle_attention.linear.weight.shape[0])
    projected = x_transposed.new_zeros(x_transposed.shape[:-1] + (out_features,))
    if valid_rows <= 0:
        return permute_final_dims(projected, (2, 0, 1)).contiguous()

    source_chunk_rows = (
        int(chunk_size)
        if chunk_size is not None and chunk_size > 0
        else _triatt_attention_row_chunk_size(valid_rows, original_n)
    )
    local_row_start = 0
    while local_row_start < valid_rows:
        global_row_start = row_start + local_row_start
        (
            canonical_rows,
            canonical_row_start,
            rows_until_chunk_end,
        ) = _triatt_source_chunk_geometry(
            global_row_start,
            original_n,
            source_chunk_rows,
        )
        current_rows = min(
            rows_until_chunk_end,
            valid_rows - local_row_start,
        )
        local_row_end = local_row_start + current_rows
        projected[local_row_start:local_row_end] = (
            foldcp_pair_tile_linear_with_source_chunk_launch(
                triangle_attention.linear,
                x_transposed[local_row_start:local_row_end],
                source_rows=canonical_rows,
                source_cols=original_n,
                row_start=canonical_row_start,
                col_start=0,
                valid_rows=current_rows,
                valid_cols=original_n,
            )
        )
        local_row_start = local_row_end
    return permute_final_dims(projected, (2, 0, 1)).contiguous()


def _one_by_p_ending_transpose_row_chunk(
    z_local: torch.Tensor,
    row_start: int,
    row_end: int,
) -> torch.Tensor:
    """Materialize only selected rows of the locally transposed 1 x P tile."""

    return (
        z_local[..., :, int(row_start) : int(row_end), :].transpose(-3, -2).contiguous()
    )


def _one_by_p_ending_transpose_row_chunk_size(
    *,
    valid_rows: int,
    bytes_per_row: int,
    alignment: int,
) -> int:
    """Bound the ending-attention transpose while preserving launch boundaries."""

    valid_rows = int(valid_rows)
    if valid_rows <= 0:
        return 0
    bytes_per_row = max(1, int(bytes_per_row))
    alignment = max(1, int(alignment))
    budget_rows = max(
        1,
        int(_ONE_BY_P_ENDING_TRANSPOSE_BUFFER_BYTES) // bytes_per_row,
    )
    if budget_rows >= valid_rows:
        return valid_rows
    aligned_rows = (budget_rows // alignment) * alignment
    if aligned_rows <= 0:
        aligned_rows = alignment
    return min(valid_rows, aligned_rows)


def _one_by_p_triangle_attention_ending_update(
    triangle_attention: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None,
    z_spec: FoldCPPairShardSpec,
    chunk_size: int | None,
) -> torch.Tensor:
    """Run ending-node attention locally for a column-sharded pair tensor.

    Transposing a canonical ``[N, N/P, C]`` shard produces
    ``[N/P, N, C]``. Every attention key for those local source columns is
    already present, so the starting-node implementation can run through a
    no-communication mesh view while retaining the source-grid launch policy.
    """

    transposed_spec = _transpose_pair_shard_spec_for_local_attention(z_spec)
    local_mesh = _local_attention_mesh(mesh)
    original_n = transposed_spec.original_shape[transposed_spec.pair_dims[0]]
    row_start, row_end = transposed_spec.row_range
    valid_rows = max(0, min(row_end, original_n) - row_start)
    physical_rows = int(z_local.shape[-2])
    bytes_per_row = int(z_local[..., :, :1, :].numel()) * int(z_local.element_size())
    launch_alignment = (
        2 * int(chunk_size)
        if chunk_size is not None and int(chunk_size) > 0
        else _triatt_attention_row_chunk_size(
            physical_rows,
            original_n,
            valid_query=original_n,
            element_size=z_local.element_size(),
        )
    )
    transpose_row_chunk = _one_by_p_ending_transpose_row_chunk_size(
        valid_rows=physical_rows,
        bytes_per_row=bytes_per_row,
        alignment=launch_alignment,
    )

    local_bias_chunks = []
    for local_row_start in range(0, physical_rows, transpose_row_chunk):
        local_row_end = min(
            local_row_start + transpose_row_chunk,
            physical_rows,
        )
        global_row_start = row_start + local_row_start
        current_valid_rows = max(
            0,
            min(row_start + valid_rows, row_start + local_row_end) - global_row_start,
        )
        z_transposed_chunk = _one_by_p_ending_transpose_row_chunk(
            z_local,
            local_row_start,
            local_row_end,
        )
        chunk_spec = replace(
            transposed_spec,
            row_range=(
                global_row_start,
                row_start + local_row_end,
            ),
        )
        x_transposed_chunk = _triangle_layer_norm_source_row_slab(
            triangle_attention.layer_norm,
            z_transposed_chunk,
            local_mesh,
            chunk_spec,
        )
        local_bias_chunks.append(
            _project_one_by_p_ending_triangle_bias_local(
                triangle_attention,
                x_transposed_chunk,
                original_n=original_n,
                row_start=global_row_start,
                valid_rows=current_valid_rows,
                chunk_size=chunk_size,
            )
        )
        del z_transposed_chunk, x_transposed_chunk

    local_triangle_bias = torch.cat(local_bias_chunks, dim=-2)
    del local_bias_chunks
    triangle_bias_full_key = _ring_gather_by_row(
        local_triangle_bias,
        mesh,
        dim=-2,
        length=original_n,
    )
    del local_triangle_bias

    for local_row_start in range(0, physical_rows, transpose_row_chunk):
        local_row_end = min(
            local_row_start + transpose_row_chunk,
            physical_rows,
        )
        global_row_start = row_start + local_row_start
        z_transposed_chunk = _one_by_p_ending_transpose_row_chunk(
            z_local,
            local_row_start,
            local_row_end,
        )
        mask_transposed_chunk = (
            None
            if mask_local is None
            else mask_local[..., :, local_row_start:local_row_end]
            .transpose(-2, -1)
            .contiguous()
        )
        chunk_spec = replace(
            transposed_spec,
            row_range=(
                global_row_start,
                row_start + local_row_end,
            ),
        )
        updated_transposed_chunk = _distributed_triangle_attention_starting_update(
            triangle_attention,
            z_transposed_chunk,
            local_mesh,
            mask_transposed_chunk,
            residual_local=z_transposed_chunk,
            z_spec=chunk_spec,
            chunk_size=chunk_size,
            triangle_bias_full_key_override=triangle_bias_full_key,
        )
        z_local[..., :, local_row_start:local_row_end, :].copy_(
            updated_transposed_chunk.transpose(-3, -2)
        )
        del z_transposed_chunk, updated_transposed_chunk
        if mask_transposed_chunk is not None:
            del mask_transposed_chunk

    del triangle_bias_full_key
    return z_local.contiguous()


def distributed_triangle_attention_update(
    triangle_attention: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Run a real TriangleAttention module on a Fold-CP local pair tile.

    The implementation matches the real torch TriangleAttention path: layernorm,
    Q/K/V projections, triangle bias, gate, and output projection come from the
    OpenDDE module. Fold-CP only replaces the full `[..., H, Q, K]` softmax with
    row-ring K/V/bias rotation plus online softmax accumulation.
    """

    if getattr(triangle_attention, "starting", True):
        return _distributed_triangle_attention_starting_update(
            triangle_attention,
            z_local,
            mesh,
            mask_local,
            residual_local=residual_local,
            z_spec=z_spec,
            chunk_size=chunk_size,
        )

    ring = mesh.ring_comm()
    z_t_local = ring.comm_2d_trans.exchange(z_local.transpose(-2, -3).contiguous())
    mask_t_local = (
        None
        if mask_local is None
        else ring.comm_2d_trans.exchange(mask_local.transpose(-1, -2).contiguous())
    )
    out_t_local = _distributed_triangle_attention_starting_update(
        triangle_attention,
        z_t_local,
        mesh,
        mask_t_local,
        z_spec=z_spec,
        chunk_size=chunk_size,
    )
    out_local = ring.comm_2d_trans.exchange(out_t_local.transpose(-2, -3).contiguous())
    if residual_local is not None:
        residual_local += out_local
        return residual_local.contiguous()
    return out_local


def _distributed_pairformer_block_pair_ops(
    block: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    z_local = distributed_triangle_multiplication_update(
        block.tri_mul_out,
        z_local,
        mesh,
        pair_mask_local,
        residual_local=z_local,
        z_spec=z_spec,
    )
    z_local = distributed_triangle_multiplication_update(
        block.tri_mul_in,
        z_local,
        mesh,
        pair_mask_local,
        residual_local=z_local,
        z_spec=z_spec,
    )
    z_local = distributed_triangle_attention_update(
        block.tri_att_start,
        z_local,
        mesh,
        pair_mask_local,
        residual_local=z_local,
        z_spec=z_spec,
        chunk_size=chunk_size,
    )

    if mesh.layout.shape[0] == 1 and z_spec is not None:
        z_local = _one_by_p_triangle_attention_ending_update(
            block.tri_att_end,
            z_local,
            mesh,
            pair_mask_local,
            z_spec=z_spec,
            chunk_size=chunk_size,
        )
    else:
        ring = mesh.ring_comm()
        z_t_local = ring.comm_2d_trans.exchange(z_local.transpose(-2, -3).contiguous())
        mask_t_local = (
            None
            if pair_mask_local is None
            else ring.comm_2d_trans.exchange(
                pair_mask_local.transpose(-1, -2).contiguous()
            )
        )
        del z_local
        if torch.is_grad_enabled():
            z_t_local = z_t_local + distributed_triangle_attention_update(
                block.tri_att_end,
                z_t_local,
                mesh,
                mask_t_local,
                z_spec=z_spec,
                chunk_size=chunk_size,
            )
        else:
            z_t_local = distributed_triangle_attention_update(
                block.tri_att_end,
                z_t_local,
                mesh,
                mask_t_local,
                residual_local=z_t_local,
                z_spec=z_spec,
                chunk_size=chunk_size,
            )
        z_local = ring.comm_2d_trans.exchange(z_t_local.transpose(-2, -3).contiguous())
        del z_t_local, mask_t_local

    z_local = distributed_pair_transition_update(
        block.pair_transition,
        z_local,
        mesh,
        residual_local=z_local,
        z_spec=z_spec,
    )
    return z_local.contiguous()


def distributed_pairformer_block_pair_update(
    block: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Run the pair-only part of a real PairformerBlock on a Fold-CP pair tile."""

    if getattr(block, "c_s", 0) != 0:
        raise ValueError(
            "distributed_pairformer_block_pair_update currently requires c_s=0."
        )

    return _distributed_pairformer_block_pair_ops(
        block,
        z_local,
        mesh,
        pair_mask_local,
        z_spec,
        chunk_size,
    )


def _foldcp_release_pairformer_block_cache() -> None:
    if torch.is_grad_enabled() or not torch.cuda.is_available():
        return
    policy = os.environ.get(
        "OPENDDE_FOLDCP_PAIRFORMER_BLOCK_CACHE_RELEASE",
        "auto",
    ).lower()
    if policy in {"0", "false", "never", "off"}:
        return
    if policy in {"1", "true", "always", "on"}:
        torch.cuda.empty_cache()
        return

    min_allocated_mib = int(
        os.environ.get(
            "OPENDDE_FOLDCP_PAIRFORMER_BLOCK_CACHE_RELEASE_MIN_ALLOCATED_MIB",
            "8192",
        )
    )
    if (
        min_allocated_mib > 0
        and torch.cuda.memory_allocated() < min_allocated_mib * 1024 * 1024
    ):
        return

    min_free_mib = int(
        os.environ.get(
            "OPENDDE_FOLDCP_PAIRFORMER_BLOCK_CACHE_RELEASE_MIN_FREE_MIB",
            "8192",
        )
    )
    if min_free_mib <= 0:
        return
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < min_free_mib * 1024 * 1024:
        torch.cuda.empty_cache()


def distributed_pairformer_stack_pair_update(
    stack: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Run a real c_s=0 PairformerStack while keeping pair activations sharded."""

    for block in stack.blocks:
        z_local = distributed_pairformer_block_pair_update(
            block,
            z_local,
            mesh,
            pair_mask_local,
            z_spec,
            chunk_size,
        )
        _foldcp_release_pairformer_block_cache()
    return z_local.contiguous()


def distributed_pairformer_stack_single_bridge_update(
    stack: torch.nn.Module,
    s: torch.Tensor,
    z: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask: torch.Tensor | None = None,
    extra_attn_bias: torch.Tensor | None = None,
    extra_attn_bias_is_local: bool = False,
    return_local_pair: bool = False,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, FoldCPPairShardSpec]
):
    """Run a real c_s>0 PairformerStack with CP pair ops and local pair bias.

    Pair operations stay on Fold-CP local tiles. The single update consumes the
    local pair-bias tile via row-ring attention and gathers only the N-sized
    single update, not the full pair tensor, at each block.
    """

    if not stack.blocks:
        return s, z
    if getattr(stack.blocks[0], "c_s", 0) <= 0:
        raise ValueError("single bridge requires a PairformerStack with c_s > 0.")

    if z_spec is None:
        z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        del z
        torch.cuda.empty_cache()
    else:
        z_local = z.contiguous()
    if pair_mask is None:
        mask_local = None
    else:
        mask_local, _ = shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))
    if extra_attn_bias is None:
        extra_attn_bias_local = None
    elif extra_attn_bias_is_local:
        extra_attn_bias_local = extra_attn_bias.contiguous()
    else:
        extra_attn_bias_local, _ = shard_pair_tensor(
            extra_attn_bias,
            mesh,
            pair_dims=(-2, -1),
        )

    for block_index, block in enumerate(stack.blocks):
        z_local = _distributed_pairformer_block_pair_ops(
            block,
            z_local,
            mesh,
            mask_local,
            z_spec,
            chunk_size,
        )
        s = s + distributed_attention_pair_bias_update(
            block.attention_pair_bias,
            s,
            z_local,
            mesh,
            z_spec=z_spec,
            extra_attn_bias_local=extra_attn_bias_local,
        )
        s = s + block.single_transition(s)
        _foldcp_release_pairformer_block_cache()

    if return_local_pair:
        return s, z_local.contiguous(), z_spec

    z = gather_pair_tensor(z_local, z_spec, mesh.group_2d)
    return s, z.contiguous()
