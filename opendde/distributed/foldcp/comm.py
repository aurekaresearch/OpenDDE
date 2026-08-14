# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP point-to-point and 2D ring communication primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.layout import FoldCP2DLayout


def _require_dist(group: Optional[dist.ProcessGroup]) -> dist.ProcessGroup:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before Fold-CP comms."
        )
    return dist.group.WORLD if group is None else group


def _ternary_parity(my_rank: int, send_rank: int, recv_rank: int) -> bool:
    return my_rank < min(send_rank, recv_rank)


class One2OneComm:
    """Small wrapper around async send/recv with deterministic ordering."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        rank_send_to: int,
        rank_recv_from: int,
        parity: Optional[bool] = None,
    ) -> None:
        self.group = _require_dist(group)
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        if rank_send_to >= self.world_size or rank_recv_from >= self.world_size:
            raise ValueError("send/recv ranks must be ranks inside the process group.")
        self.rank_send_to = rank_send_to
        self.rank_recv_from = rank_recv_from
        self.is_self_comm = rank_send_to == self.rank and rank_recv_from == self.rank
        if (rank_send_to == self.rank) != (rank_recv_from == self.rank):
            raise ValueError("asymmetric self send/recv is not supported.")
        self.global_send_to = (
            rank_send_to
            if self.is_self_comm
            else dist.get_global_rank(self.group, rank_send_to)
        )
        self.global_recv_from = (
            rank_recv_from
            if self.is_self_comm
            else dist.get_global_rank(self.group, rank_recv_from)
        )
        self.parity = bool(self.rank % 2) if parity is None else bool(parity)
        self._queue: list[dist.P2POp] = []
        self._work: Optional[list[dist.Work]] = None

    def enqueue_to_dispatch(
        self,
        to_send: torch.Tensor,
        to_recv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        recv = self._prepare(to_send=to_send, to_recv=to_recv)
        self.dispatch()
        return recv

    def prepare_to_dispatch(
        self,
        to_send: torch.Tensor,
        to_recv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Queue one send/recv pair without dispatching it yet.

        Triangle attention rotates K, V, mask, and triangle-bias tensors to the
        same peer in each ring step.  Queuing them together lets one
        `batch_isend_irecv` launch match boltz-cp's multi-buffer communication
        pattern while keeping the existing `exchange` API unchanged.
        """

        return self._prepare(to_send=to_send, to_recv=to_recv)

    def exchange(
        self, to_send: torch.Tensor, to_recv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        recv = self.enqueue_to_dispatch(to_send=to_send, to_recv=to_recv)
        self.wait_until_finished()
        return recv

    def dispatch(self) -> None:
        if self.is_self_comm:
            return
        if self._work is not None:
            raise RuntimeError("cannot dispatch with unfinished communication.")
        self._work = dist.batch_isend_irecv(self._queue)

    def wait_until_finished(self) -> None:
        if self.is_self_comm:
            return
        if self._work is None:
            raise RuntimeError("cannot wait without dispatched communication.")
        for work in self._work:
            work.wait()
        self._queue = []
        self._work = None

    def _prepare(
        self,
        to_send: torch.Tensor,
        to_recv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.is_self_comm:
            recv = to_send.detach().clone() if to_recv is None else to_recv
            if to_recv is not None:
                recv.copy_(to_send)
            return recv
        recv = torch.empty_like(to_send) if to_recv is None else to_recv
        send_op = dist.P2POp(dist.isend, to_send, self.global_send_to, group=self.group)
        recv_op = dist.P2POp(dist.irecv, recv, self.global_recv_from, group=self.group)
        self._queue.extend([send_op, recv_op] if self.parity else [recv_op, send_op])
        return recv


class TransposeComm(One2OneComm):
    """Exchange each 2D rank tile with its transposed coordinate."""

    def __init__(self, group: dist.ProcessGroup, layout: FoldCP2DLayout) -> None:
        group = _require_dist(group)
        group_rank = dist.get_rank(group)
        coord = layout.to_coord(group_rank)
        transposed_rank = layout.transpose_rank(coord)
        parity = coord[0] < coord[1]
        super().__init__(group, transposed_rank, transposed_rank, parity=parity)


@dataclass
class Ring2DComm:
    """2D ring communication handles used by Fold-CP distributed kernels."""

    group_2d: dist.ProcessGroup
    group_col: dist.ProcessGroup
    layout: FoldCP2DLayout

    def __post_init__(self) -> None:
        self.group_2d = _require_dist(self.group_2d)
        self.group_col = _require_dist(self.group_col)
        if dist.get_world_size(self.group_2d) != self.layout.numel:
            raise ValueError("group_2d size must match the 2D CP layout.")
        self.rank_2d = dist.get_rank(self.group_2d)
        self.coord_2d = self.layout.to_coord(self.rank_2d)

        self.comm_2d_trans = TransposeComm(self.group_2d, self.layout)

        row = self.coord_2d[0]
        col = self.coord_2d[1]
        row_init_send = self.layout.shifted_rank(self.coord_2d, axis=1, shift=-row)
        row_init_recv = self.layout.shifted_rank(self.coord_2d, axis=1, shift=row)
        row_send = self.layout.shifted_rank(self.coord_2d, axis=1, shift=-1)
        row_recv = self.layout.shifted_rank(self.coord_2d, axis=1, shift=1)

        col_init_send = self.layout.shifted_rank(self.coord_2d, axis=0, shift=-col)
        col_init_recv = self.layout.shifted_rank(self.coord_2d, axis=0, shift=col)
        col_send = self.layout.shifted_rank(self.coord_2d, axis=0, shift=-1)
        col_recv = self.layout.shifted_rank(self.coord_2d, axis=0, shift=1)

        self.comm_row_init = One2OneComm(
            self.group_2d,
            row_init_send,
            row_init_recv,
            parity=_ternary_parity(self.rank_2d, row_init_send, row_init_recv),
        )
        self.comm_row = One2OneComm(
            self.group_2d,
            row_send,
            row_recv,
            parity=_ternary_parity(self.rank_2d, row_send, row_recv),
        )
        self.comm_col_init = One2OneComm(
            self.group_2d,
            col_init_send,
            col_init_recv,
            parity=_ternary_parity(self.rank_2d, col_init_send, col_init_recv),
        )
        self.comm_col = One2OneComm(
            self.group_2d,
            col_send,
            col_recv,
            parity=_ternary_parity(self.rank_2d, col_send, col_recv),
        )
