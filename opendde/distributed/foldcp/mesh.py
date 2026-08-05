# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP process mesh creation."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist

from opendde.distributed.foldcp.comm import Ring2DComm
from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.layout import FoldCP2DLayout


@dataclass(frozen=True)
class FoldCPProcessMesh:
    """Process groups and local coordinates for one Fold-CP 2D mesh."""

    config: FoldCPConfig
    layout: FoldCP2DLayout
    group_2d: dist.ProcessGroup
    group_row: dist.ProcessGroup
    group_col: dist.ProcessGroup
    cp_global_ranks: tuple[int, ...]
    cp_rank: int
    coord: tuple[int, int]

    @classmethod
    def create(cls, config: FoldCPConfig) -> "FoldCPProcessMesh":
        if not config.enabled:
            raise ValueError("FoldCPProcessMesh.create requires distributed mode.")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first.")

        world_size = dist.get_world_size()
        if world_size != config.size_dp * config.size_cp:
            raise ValueError(
                "WORLD_SIZE must equal foldcp_size_dp * foldcp_size_cp; "
                f"got {world_size} vs {config.size_dp} * {config.size_cp}."
            )

        layout = FoldCP2DLayout(config.cp_mesh_shape)
        rows, cols = layout.shape
        world_rank = dist.get_rank()
        dp_index = world_rank // config.size_cp
        cp_offset = dp_index * config.size_cp
        cp_global_ranks = tuple(range(cp_offset, cp_offset + config.size_cp))

        selected_group_2d = None
        selected_row_group = None
        selected_col_group = None
        cp_rank = world_rank - cp_offset
        coord = layout.to_coord(cp_rank)

        # All ranks create groups in the same order to avoid distributed hangs.
        for dp in range(config.size_dp):
            block_offset = dp * config.size_cp
            block_ranks = tuple(range(block_offset, block_offset + config.size_cp))
            group = dist.new_group(list(block_ranks))
            if dp == dp_index:
                selected_group_2d = group
            for row in range(rows):
                row_ranks = [
                    block_offset + layout.to_linear((row, col)) for col in range(cols)
                ]
                group = dist.new_group(row_ranks)
                if dp == dp_index and row == coord[0]:
                    selected_row_group = group
            for col in range(cols):
                col_ranks = [
                    block_offset + layout.to_linear((row, col)) for row in range(rows)
                ]
                group = dist.new_group(col_ranks)
                if dp == dp_index and col == coord[1]:
                    selected_col_group = group

        if (
            selected_group_2d is None
            or selected_row_group is None
            or selected_col_group is None
        ):
            raise RuntimeError("failed to create Fold-CP row/column groups.")

        return cls(
            config=config,
            layout=layout,
            group_2d=selected_group_2d,
            group_row=selected_row_group,
            group_col=selected_col_group,
            cp_global_ranks=cp_global_ranks,
            cp_rank=cp_rank,
            coord=coord,
        )

    def ring_comm(self) -> Ring2DComm:
        return Ring2DComm(
            group_2d=self.group_2d,
            group_col=self.group_col,
            layout=self.layout,
        )
