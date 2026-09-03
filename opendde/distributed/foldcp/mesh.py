# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP process mesh creation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.comm import (
    One2OneComm,
    Ring2DComm,
    clear_foldcp_communication_cache,
    get_foldcp_cpu_control_group,
    run_group_rank_action_synchronized,
)
from opendde.distributed.foldcp.config import FoldCPConfig
from opendde.distributed.foldcp.layout import FoldCP2DLayout

_PROCESS_MESH_CACHE: dict[tuple[int, int, int], "FoldCPProcessMesh"] = {}
_PROCESS_MESH_TOPOLOGY: dict[int, tuple[int, int]] = {}


def clear_foldcp_process_mesh_cache() -> None:
    """Drop cached mesh handles before the owning default group is destroyed."""

    _PROCESS_MESH_CACHE.clear()
    _PROCESS_MESH_TOPOLOGY.clear()
    clear_foldcp_communication_cache()


@dataclass(frozen=True)
class FoldCPProcessMesh:
    """Process groups and local coordinates for the maintained 1 x P mesh."""

    config: FoldCPConfig
    layout: FoldCP2DLayout
    group_2d: dist.ProcessGroup
    group_row: dist.ProcessGroup
    cp_global_ranks: tuple[int, ...]
    cp_rank: int
    coord: tuple[int, int]

    @property
    def group_col(self) -> dist.ProcessGroup:
        """Reject access to the removed multi-row column communicator."""

        raise RuntimeError(
            "Fold-CP column communication is unavailable in the maintained "
            "1xP topology."
        )

    @classmethod
    def create(cls, config: FoldCPConfig) -> "FoldCPProcessMesh":
        # Callers can instantiate the frozen dataclass directly instead of using
        # ``from_runtime_args``.  Validate again at the library boundary so the
        # removed 2 x 2 topology cannot bypass the CLI/Runner guard.
        config = config.validate()
        if not config.enabled:
            raise ValueError("FoldCPProcessMesh.create requires distributed mode.")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized first.")

        world_identity = id(dist.group.WORLD)
        cache_key = (world_identity, config.size_dp, config.size_cp)
        cached = _PROCESS_MESH_CACHE.get(cache_key)
        if cached is not None:
            return cached

        requested_topology = (config.size_dp, config.size_cp)
        established_topology = _PROCESS_MESH_TOPOLOGY.get(world_identity)
        if established_topology is not None:
            raise RuntimeError(
                "Fold-CP topology is immutable for an initialized default process "
                f"group; established={established_topology}, "
                f"requested={requested_topology}."
            )

        world_size = dist.get_world_size()
        local_topology = {
            "size_dp": config.size_dp,
            "size_cp": config.size_cp,
            "mesh_shape": config.cp_mesh_shape,
        }
        gathered_topologies: list[dict[str, object] | None] = [None] * world_size
        control_group = get_foldcp_cpu_control_group()
        if control_group is None:
            dist.all_gather_object(gathered_topologies, local_topology)
        else:
            if dist.get_world_size(control_group) != world_size:
                raise RuntimeError(
                    "Fold-CP CPU control group does not span the 1xP world."
                )
            dist.all_gather_object(
                gathered_topologies,
                local_topology,
                group=control_group,
            )
        if any(topology != local_topology for topology in gathered_topologies):
            summary = ", ".join(
                f"rank {rank}: {topology}"
                for rank, topology in enumerate(gathered_topologies)
            )
            raise RuntimeError(
                "Distributed ranks configured different Fold-CP mesh topologies "
                f"before process-group creation: {summary}"
            )

        if world_size != config.size_dp * config.size_cp:
            raise ValueError(
                "WORLD_SIZE must equal foldcp_size_dp * foldcp_size_cp; "
                f"got {world_size} vs {config.size_dp} * {config.size_cp}."
            )

        layout = FoldCP2DLayout(config.cp_mesh_shape)
        rows, cols = layout.shape
        if rows != 1 or cols != world_size:  # pragma: no cover - validate() above
            raise RuntimeError("Fold-CP process mesh must be the complete 1xP world.")
        world_rank = dist.get_rank()
        cp_global_ranks = tuple(range(world_size))

        # In the maintained 1xP topology, the complete mesh is exactly NCCL
        # WORLD: all ranks, in the same order. Creating another all-rank group
        # duplicates NCCL communicator/channel state and defers hundreds of MiB
        # of internal allocations until the first model collective. Reuse WORLD
        # for both the mesh and its only row.
        selected_group_2d = dist.group.WORLD
        selected_row_group = selected_group_2d
        cp_rank = world_rank
        coord = layout.to_coord(cp_rank)

        mesh = cls(
            config=config,
            layout=layout,
            group_2d=selected_group_2d,
            group_row=selected_row_group,
            cp_global_ranks=cp_global_ranks,
            cp_rank=cp_rank,
            coord=coord,
        )
        _PROCESS_MESH_CACHE[cache_key] = mesh
        _PROCESS_MESH_TOPOLOGY[world_identity] = requested_topology
        return mesh

    def prewarm_communications(self) -> None:
        """Initialize every collective/P2P route before model allocation.

        NCCL process-group construction is lazy. A plain ``new_group`` (and now
        the reused WORLD handle) does not reserve collective or peer-to-peer
        channel memory. The maintained 1xP implementation uses a full-group
        reduction plus direct peer rounds in the Pairformer/Diffusion paths, so
        exercise those exact rank pairs with one scalar while GPU memory is
        still otherwise empty. No model arithmetic or launch shape is changed.
        """

        side = int(self.layout.shape[1])
        if side <= 1:
            return

        device = torch.device("cuda", torch.cuda.current_device())
        buffers = run_group_rank_action_synchronized(
            lambda: (
                torch.zeros((), device=device),
                torch.empty((), device=device),
            ),
            group=self.group_2d,
            description="Fold-CP NCCL communication warmup allocation",
        )
        if buffers is None:  # pragma: no cover - every rank runs the action
            raise RuntimeError("Fold-CP NCCL warmup returned no buffers.")
        send, receive = buffers

        # Pairformer uses the neighbouring ring; pair-row collection also uses
        # every other peer. Covering offsets 1..P-1 initializes all routes that
        # can otherwise allocate NCCL P2P buffers late in a near-capacity run.
        for offset in range(1, side):
            send_rank = (int(self.cp_rank) + offset) % side
            receive_rank = (int(self.cp_rank) - offset) % side
            peer = One2OneComm(
                self.group_2d,
                send_rank,
                receive_rank,
            )
            peer.exchange(send, to_recv=receive)

    def ring_comm(self) -> Ring2DComm:
        return Ring2DComm(
            group_2d=self.group_2d,
            layout=self.layout,
        )
