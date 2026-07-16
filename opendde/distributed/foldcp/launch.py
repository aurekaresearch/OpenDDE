# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP CUDA launch-shape helpers.

These helpers keep distributed local-shard projections on the same deterministic
CUDA launch family as the serial source path without gathering full pair tensors.
"""

from __future__ import annotations

import torch


_SMALL_LINEAR_SOURCE_ROWS = 262_144


def foldcp_linear_launch_rows(*, local_rows: int, source_rows: int) -> int:
    """Return the launch row count for a local Fold-CP Linear projection.

    The policy is intentionally shape based rather than module based. Small
    local tiles can select a different deterministic CUDA GEMM path from the
    source full-shape projection. Padding the local rows to the source bucket
    preserves the launch family while only materializing local shard data.
    """

    if local_rows <= 0 or source_rows <= 0:
        return max(local_rows, 0)
    if local_rows >= source_rows:
        return local_rows
    if source_rows <= _SMALL_LINEAR_SOURCE_ROWS:
        return source_rows
    if local_rows < _SMALL_LINEAR_SOURCE_ROWS:
        return _SMALL_LINEAR_SOURCE_ROWS
    return local_rows


def foldcp_linear_with_source_launch_shape(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    source_rows: int,
) -> torch.Tensor:
    """Run ``linear`` on a local tensor using the Fold-CP launch policy.

    Only the local rows are populated; padded rows are zeros and discarded. This
    does not gather remote pair data and does not change the mathematical
    function of any owned row.
    """

    local_rows = int(x.numel() // x.shape[-1]) if x.shape[-1] else 0
    launch_rows = foldcp_linear_launch_rows(
        local_rows=local_rows,
        source_rows=int(source_rows),
    )
    flat = x.contiguous().reshape(local_rows, x.shape[-1])
    if launch_rows <= local_rows:
        projected = linear(flat)
        return projected.reshape(*x.shape[:-1], -1)

    launch = flat.new_zeros(launch_rows, flat.shape[-1])
    launch[:local_rows].copy_(flat)
    projected = linear(launch)[:local_rows]
    return projected.reshape(*x.shape[:-1], -1)


def foldcp_pair_row_slab_linear_with_source_grid_launch(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    col_start: int = 0,
    valid_rows: int | None = None,
    valid_cols: int | None = None,
) -> torch.Tensor:
    """Run a pair row-slab Linear with source full-pair flat indices.

    The input contains only this CP rank's output-row slab and all source
    columns, i.e. ``[local_rows, source_cols, C]``. Some deterministic CUDA
    GEMM paths are sensitive not just to row count, but to the full source
    ``[N, N, C]`` flat launch family and the owned rows' global offset. This
    helper places only owned valid entries at their source flat indices, runs
    the projection once, and slices the same owned entries back. It does not
    gather remote pair data and it does not materialize a full pair output.
    """

    if x.ndim != 3:
        raise ValueError("source-grid row-slab launch expects [rows, cols, channels].")
    original_n = int(original_n)
    row_start = int(row_start)
    col_start = int(col_start)
    tile_rows = int(x.shape[0])
    tile_cols = int(x.shape[1])
    if valid_rows is None:
        valid_rows = max(0, min(tile_rows, original_n - row_start))
    else:
        valid_rows = max(0, min(int(valid_rows), tile_rows, original_n - row_start))
    if valid_cols is None:
        valid_cols = min(tile_cols, original_n)
    else:
        valid_cols = max(0, min(int(valid_cols), tile_cols, original_n))

    out_features = int(linear.weight.shape[0])
    if valid_rows <= 0 or valid_cols <= 0:
        return x.new_zeros((tile_rows, tile_cols, out_features))

    flat = x.contiguous().reshape(-1, x.shape[-1])
    source_rows = original_n * original_n
    launch = flat.new_zeros(source_rows, flat.shape[-1])
    row_offsets = (
        (torch.arange(valid_rows, device=x.device) + row_start) * original_n
    )
    source_index = (
        row_offsets[:, None]
        + col_start
        + torch.arange(valid_cols, device=x.device)[None, :]
    ).reshape(-1)
    tile_index = (
        torch.arange(valid_rows, device=x.device)[:, None] * tile_cols
        + torch.arange(valid_cols, device=x.device)[None, :]
    ).reshape(-1)
    launch.index_copy_(0, source_index, flat.index_select(0, tile_index))
    projected = linear(launch).index_select(0, source_index)
    out = projected.new_zeros((tile_rows, tile_cols, out_features))
    out.reshape(-1, out_features).index_copy_(0, tile_index, projected)
    return out


def foldcp_pair_tile_linear_with_source_chunk_launch(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    source_rows: int,
    source_cols: int,
    row_start: int = 0,
    col_start: int = 0,
    valid_rows: int | None = None,
    valid_cols: int | None = None,
) -> torch.Tensor:
    """Project a local pair tile at its source chunk-local flat indices.

    Serial triangle multiplication projects b chunks and chunked a rows on the
    current chunk tensor, not on the full global ``[N, N]`` pair grid. This
    helper preserves that chunk-local launch shape and chunk-local row/column
    offsets while still materializing only this rank's local tile entries.
    """

    if x.ndim != 3:
        raise ValueError("source-chunk tile launch expects [rows, cols, channels].")
    source_rows = int(source_rows)
    source_cols = int(source_cols)
    row_start = int(row_start)
    col_start = int(col_start)
    tile_rows = int(x.shape[0])
    tile_cols = int(x.shape[1])
    if valid_rows is None:
        valid_rows = max(0, min(tile_rows, source_rows - row_start))
    else:
        valid_rows = max(0, min(int(valid_rows), tile_rows, source_rows - row_start))
    if valid_cols is None:
        valid_cols = max(0, min(tile_cols, source_cols - col_start))
    else:
        valid_cols = max(0, min(int(valid_cols), tile_cols, source_cols - col_start))

    out_features = int(linear.weight.shape[0])
    if valid_rows <= 0 or valid_cols <= 0:
        return x.new_zeros((tile_rows, tile_cols, out_features))

    flat = x.contiguous().reshape(-1, x.shape[-1])
    launch_rows = source_rows * source_cols
    launch = flat.new_zeros(launch_rows, flat.shape[-1])
    row_offsets = (
        (torch.arange(valid_rows, device=x.device) + row_start) * source_cols
    )
    source_index = (
        row_offsets[:, None]
        + col_start
        + torch.arange(valid_cols, device=x.device)[None, :]
    ).reshape(-1)
    tile_index = (
        torch.arange(valid_rows, device=x.device)[:, None] * tile_cols
        + torch.arange(valid_cols, device=x.device)[None, :]
    ).reshape(-1)
    launch.index_copy_(0, source_index, flat.index_select(0, tile_index))
    projected = linear(launch).index_select(0, source_index)
    out = projected.new_zeros((tile_rows, tile_cols, out_features))
    out.reshape(-1, out_features).index_copy_(0, tile_index, projected)
    return out


_PAIR_ROW_SLAB_SOURCE_GRID_MIN_LOCAL_ROWS = 1_048_576
_PAIR_ROW_SLAB_SOURCE_GRID_BOUNDARY_MIN_ROWS = 1_048_576
_PAIR_ROW_SLAB_SOURCE_GRID_BOUNDARY_MAX_ROWS = 1_054_729


def _pair_row_slab_source_grid_boundary(source_rows: int) -> bool:
    source_rows = int(source_rows)
    return (
        _PAIR_ROW_SLAB_SOURCE_GRID_BOUNDARY_MIN_ROWS
        <= source_rows
        <= _PAIR_ROW_SLAB_SOURCE_GRID_BOUNDARY_MAX_ROWS
    )


def foldcp_pair_row_slab_linear_with_source_launch_policy(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    col_start: int = 0,
    valid_rows: int | None = None,
    valid_cols: int | None = None,
) -> torch.Tensor:
    """Run a row-slab pair Linear with the cheapest source launch that is safe.

    Most row slabs only need the standard source-row launch bucket. Very large
    local row slabs can cross a CUDA launch-family boundary where the owned
    rows' global pair offset affects deterministic FP32 results. For that
    continuous large-slab region, fall back to source-grid indexing while still
    computing only this rank's local output tile.
    """
    if x.ndim != 3:
        raise ValueError("source row-slab launch policy expects [rows, cols, channels].")
    original_n = int(original_n)
    row_start = int(row_start)
    col_start = int(col_start)
    tile_rows = int(x.shape[0])
    tile_cols = int(x.shape[1])
    if valid_rows is None:
        valid_rows = max(0, min(tile_rows, original_n - row_start))
    else:
        valid_rows = max(0, min(int(valid_rows), tile_rows, original_n - row_start))
    if valid_cols is None:
        valid_cols = min(tile_cols, original_n)
    else:
        valid_cols = max(0, min(int(valid_cols), tile_cols, original_n))

    source_rows = original_n * original_n
    local_valid_rows = valid_rows * valid_cols
    if (
        local_valid_rows >= _PAIR_ROW_SLAB_SOURCE_GRID_MIN_LOCAL_ROWS
        or _pair_row_slab_source_grid_boundary(source_rows)
    ):
        return foldcp_pair_row_slab_linear_with_source_grid_launch(
            linear,
            x,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_cols,
        )
    return foldcp_linear_with_source_launch_shape(
        linear,
        x,
        source_rows=source_rows,
    )


def foldcp_module_with_source_launch_shape(
    module: torch.nn.Module,
    x: torch.Tensor,
    *,
    source_rows: int,
) -> torch.Tensor:
    """Run a row-wise module on local rows with the source launch policy.

    The module must preserve the leading row count and operate independently over
    the last dimension, such as LayerNorm or Linear. Only owned local rows are
    populated; padded rows are discarded.
    """

    local_rows = int(x.numel() // x.shape[-1]) if x.shape[-1] else 0
    launch_rows = foldcp_linear_launch_rows(
        local_rows=local_rows,
        source_rows=int(source_rows),
    )
    if launch_rows <= local_rows:
        return module(x)

    flat = x.contiguous().reshape(local_rows, x.shape[-1])
    launch = flat.new_zeros(launch_rows, flat.shape[-1])
    launch[:local_rows].copy_(flat)
    out = module(launch)[:local_rows]
    return out.reshape(*x.shape[:-1], -1)
