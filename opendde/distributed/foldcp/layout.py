# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Rank layout helpers for Fold-CP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoldCP2DLayout:
    """Map between a CP coordinate and rank inside a CP group.

    The maintained Fold-CP topology is strictly 1 x P.  Keeping the two-axis
    coordinate type avoids a broad representation change in pair-sharding
    metadata, but the row coordinate is always zero at runtime.
    """

    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("FoldCP2DLayout expects a 2D shape.")
        rows, cols = self.shape
        if rows < 1 or cols < 1:
            raise ValueError("Fold-CP mesh dimensions must be positive.")
        if rows != 1:
            raise ValueError("Only the maintained 1 x P Fold-CP layout is supported.")

    @property
    def numel(self) -> int:
        return self.shape[0] * self.shape[1]

    def to_linear(self, coord: tuple[int, int]) -> int:
        row, col = coord
        rows, cols = self.shape
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"Coordinate {coord} is outside mesh shape {self.shape}.")
        return row * cols + col

    def to_coord(self, linear_rank: int) -> tuple[int, int]:
        if not (0 <= linear_rank < self.numel):
            raise ValueError(
                f"Rank {linear_rank} is outside mesh with {self.numel} ranks."
            )
        return divmod(linear_rank, self.shape[1])

    def shifted_rank(self, coord: tuple[int, int], axis: int, shift: int) -> int:
        if axis not in (0, 1):
            raise ValueError("axis must be 0 for rows or 1 for columns.")
        row, col = coord
        if axis == 0:
            row = (row + shift) % self.shape[0]
        else:
            col = (col + shift) % self.shape[1]
        return self.to_linear((row, col))

    def transpose_rank(self, coord: tuple[int, int]) -> int:
        if self.shape[0] == 1:
            return self.to_linear(coord)
        return self.to_linear((coord[1], coord[0]))
