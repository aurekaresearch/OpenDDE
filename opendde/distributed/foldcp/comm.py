# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP point-to-point and 2D ring communication primitives."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Callable, Optional, TypeVar, cast

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.layout import FoldCP2DLayout


_T = TypeVar("_T")
_NCCL_STATUS_TENSORS: dict[tuple[int, int], torch.Tensor] = {}
_CPU_CONTROL_GROUP: dist.ProcessGroup | None = None


def register_foldcp_cpu_control_group(group: dist.ProcessGroup) -> None:
    """Publish the Runner-owned Gloo group for post-OOM error reporting."""

    global _CPU_CONTROL_GROUP
    try:
        backend = dist.get_backend(group)
    except Exception as exc:
        raise RuntimeError("Could not inspect the Fold-CP CPU control group.") from exc
    if backend != dist.Backend.GLOO:
        raise ValueError("Fold-CP CPU control group must use the Gloo backend.")
    if _CPU_CONTROL_GROUP is not None and _CPU_CONTROL_GROUP is not group:
        raise RuntimeError("A different Fold-CP CPU control group is already active.")
    _CPU_CONTROL_GROUP = group


def unregister_foldcp_cpu_control_group(
    group: dist.ProcessGroup | None = None,
) -> None:
    """Forget a destroyed Runner-owned Gloo group without touching NCCL state."""

    global _CPU_CONTROL_GROUP
    if group is None or _CPU_CONTROL_GROUP is group:
        _CPU_CONTROL_GROUP = None


def get_foldcp_cpu_control_group() -> dist.ProcessGroup | None:
    """Return the active Runner-owned Gloo group, if one is registered."""

    return _CPU_CONTROL_GROUP


def foldcp_control_barrier(data_group: dist.ProcessGroup) -> None:
    """Synchronize 1xP control flow without a late CUDA/NCCL allocation.

    Runner-backed inference registers a same-rank Gloo world for status and
    lifecycle coordination.  Prefer it for barriers that carry no model data;
    embedded callers without that group retain their original data-group
    behavior.
    """

    barrier_group = data_group
    control_group = _CPU_CONTROL_GROUP
    if control_group is not None:
        try:
            if dist.get_world_size(control_group) == dist.get_world_size(data_group):
                barrier_group = control_group
        except Exception:
            # A direct caller may tear down its optional control group without
            # unregistering it. Preserve the historical data-group fallback.
            barrier_group = data_group
    dist.barrier(group=barrier_group)


def dispatch_p2p_batch_and_wait(operations: list[dist.P2POp]) -> None:
    """Launch one P2P batch, drain every Work, and release request payloads.

    Python loop targets, the operation list, and Work handles can each retain
    CUDA tensors after a completed transfer. Streamed kernels allocate their
    next block before those locals naturally leave scope, so centralize the
    cleanup and make the failure path drain every already-launched request.
    """

    work_items: list[dist.Work] | None = None
    current_work: dist.Work | None = None
    first_error: Exception | None = None
    try:
        try:
            work_items = dist.batch_isend_irecv(operations)
        except Exception as exc:
            first_error = detach_rank_local_error_traceback(exc)
        if work_items is not None:
            for current_work in work_items:
                try:
                    current_work.wait()
                except Exception as exc:
                    detached = detach_rank_local_error_traceback(exc)
                    if first_error is None:
                        first_error = detached
    finally:
        operations.clear()
        if work_items is not None:
            work_items.clear()
        current_work = None
        work_items = None
    if first_error is not None:
        raise first_error


def _gather_rank_errors(
    local_error: str,
    *,
    data_group: dist.ProcessGroup,
) -> list[str | None]:
    """Gather rare-path diagnostics without allocating on CUDA after an OOM."""

    data_size = dist.get_world_size(data_group)
    if data_size <= 1:
        return [local_error]

    error_group = data_group
    if _CPU_CONTROL_GROUP is not None:
        control_size = dist.get_world_size(_CPU_CONTROL_GROUP)
        if control_size == data_size:
            # Only 1xP is supported, so every non-singleton model group spans
            # the same ranks in the same order as the Runner-owned world group.
            error_group = _CPU_CONTROL_GROUP

    errors: list[str | None] = [None] * data_size
    dist.all_gather_object(errors, local_error, group=error_group)
    return errors


def clear_foldcp_communication_cache() -> None:
    """Release CUDA status tensors owned by Fold-CP process groups."""

    _NCCL_STATUS_TENSORS.clear()


def detach_rank_local_error_traceback(exc: Exception) -> Exception:
    """Retain an error for post-collective reporting without retaining tensors.

    A traceback owns every completed frame in the failing CUDA operation. Those
    frames can own multi-gigabyte attention/matmul temporaries, defeating the
    retained-error pattern whose purpose is to leave enough memory to drain the
    remaining communication schedule. Preserve the exception type/message, but
    clear and detach its old traceback before storing it.
    """

    pending: list[BaseException] = [exc]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        old_traceback = current.__traceback__
        if old_traceback is not None:
            traceback.clear_frames(old_traceback)
            current.__traceback__ = None
    return exc


def _append_cuda_cache_cleanup_error(
    error: str,
    *,
    attempt: bool,
) -> str:
    """Best-effort cache cleanup that preserves the following rank handshake."""

    if not attempt or not torch.cuda.is_available():
        return error
    try:
        torch.cuda.empty_cache()
    except Exception as cleanup_error:
        return (
            f"{error}\nCUDA cache cleanup also failed: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    return error


def _require_dist(group: Optional[dist.ProcessGroup]) -> dist.ProcessGroup:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before Fold-CP comms."
        )
    return dist.group.WORLD if group is None else group


def _nccl_group_has_failure(
    _local_error: str,
    group: dist.ProcessGroup,
) -> bool | None:
    """Use a cached device scalar for the common all-success NCCL path.

    Object collectives serialize strings and stage tensors on CUDA even when
    every rank succeeded. The scalar reduction avoids that fixed overhead;
    returning ``None`` keeps the original object-collective path for non-NCCL
    groups and test doubles.
    """

    try:
        backend = dist.get_backend(group)
    except Exception:
        return None
    if backend != dist.Backend.NCCL or not torch.cuda.is_available():
        return None

    device_index = torch.cuda.current_device()
    cache_key = (id(group), device_index)
    status = _NCCL_STATUS_TENSORS.get(cache_key)
    if status is None:
        # The synchronized action entry primes NCCL groups before the protected
        # allocation. Falling back here avoids an uncoordinated late allocation
        # if a caller clears the cache concurrently or calls this private helper
        # directly.
        return None
    dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    return bool(status.item())


def _arm_nccl_group_status(group: dist.ProcessGroup) -> bool:
    """Pre-arm the cached failure scalar before protected CUDA work starts."""

    try:
        backend = dist.get_backend(group)
    except Exception:
        return False
    if backend != dist.Backend.NCCL or not torch.cuda.is_available():
        return False
    cache_key = (id(group), torch.cuda.current_device())
    status = _NCCL_STATUS_TENSORS.get(cache_key)
    if status is None:
        return False
    status.fill_(1)
    return True


def _clear_nccl_group_status(group: dist.ProcessGroup) -> None:
    """Mark a pre-armed protected action successful before its reduction."""

    cache_key = (id(group), torch.cuda.current_device())
    status = _NCCL_STATUS_TENSORS.get(cache_key)
    if status is None:  # pragma: no cover - the caller just armed this key
        raise RuntimeError("NCCL status scalar disappeared during a rank action.")
    status.zero_()


def _prime_nccl_group_status(group: dist.ProcessGroup) -> None:
    """Reserve the NCCL status scalar before a protected action consumes memory."""

    try:
        backend = dist.get_backend(group)
    except Exception:
        return
    if backend != dist.Backend.NCCL or not torch.cuda.is_available():
        return

    device_index = torch.cuda.current_device()
    cache_key = (id(group), device_index)
    if cache_key in _NCCL_STATUS_TENSORS:
        return

    status: torch.Tensor | None = None
    local_error = ""
    try:
        status = torch.empty(
            (),
            dtype=torch.int32,
            device=torch.device("cuda", device_index),
        )
    except Exception as exc:
        local_error = (
            f"group rank {dist.get_rank(group)} NCCL status reservation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        detach_rank_local_error_traceback(exc)
        local_error = _append_cuda_cache_cleanup_error(local_error, attempt=True)

    errors = _gather_rank_errors(local_error, data_group=group)
    failures = [error for error in errors if error]
    if failures:
        raise RuntimeError("\n".join(cast(list[str], failures)))
    if status is None:  # pragma: no cover - a missing status must report above
        raise RuntimeError("NCCL status reservation returned no tensor.")
    _NCCL_STATUS_TENSORS[cache_key] = status


def run_group_rank_action_synchronized(
    action: Callable[[], _T] | None,
    *,
    group: dist.ProcessGroup,
    description: str,
) -> _T | None:
    """Run an optional rank-local allocation/action before peers communicate.

    Destination-only buffers are a common source of distributed OOM hangs: the
    destination raises while peers enter the following gather/send.  Every rank
    calls this helper, which exchanges the tiny completion status first.
    """

    _prime_nccl_group_status(group)
    status_prearmed = _arm_nccl_group_status(group)

    result: _T | None = None
    local_error = ""
    if action is not None:
        try:
            result = action()
        except Exception as exc:
            # ``action`` may be a closure over multi-GiB CUDA inputs or
            # workspaces.  Drop it before cache cleanup and before a later
            # peer-failure exception retains this helper frame.
            action = None
            local_error = (
                f"group rank {dist.get_rank(group)} {description} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            # Release tensor-owning frames before empty_cache().  Calling cache
            # cleanup while the exception traceback still owns the failed
            # action's temporaries cannot return those blocks to the allocator.
            detach_rank_local_error_traceback(exc)
            local_error = _append_cuda_cache_cleanup_error(
                local_error,
                attempt=True,
            )
        else:
            # The callable is no longer needed once its result has been
            # materialized.  Keeping it until the status handshake completes
            # can retain closure-owned CUDA tensors on a successful rank while
            # another rank reports an allocation failure.
            action = None

    if status_prearmed and not local_error:
        try:
            _clear_nccl_group_status(group)
        except Exception as exc:
            # The status was set to failure before the protected action. If
            # clearing it surfaces a deferred CUDA/OOM error, leave it armed so
            # every rank observes the failure in the already-required reduce.
            local_error = (
                f"group rank {dist.get_rank(group)} {description} completion "
                f"failed: {type(exc).__name__}: {exc}"
            )
            detach_rank_local_error_traceback(exc)
            local_error = _append_cuda_cache_cleanup_error(
                local_error,
                attempt=True,
            )

    has_failure = _nccl_group_has_failure(local_error, group)
    if has_failure is False:
        return result

    errors = _gather_rank_errors(local_error, data_group=group)
    failures = [error for error in errors if error]
    if failures:
        # A peer can fail while this rank successfully allocated a multi-GiB
        # result.  Do not let the exception traceback retain this helper frame
        # together with that result until a much later outer cleanup.  Dropping
        # the reference here makes the storage immediately reusable while every
        # rank reports the same root cause.
        result = None
        raise RuntimeError("\n".join(cast(list[str], failures)))
    return result


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
        if not (
            0 <= rank_send_to < self.world_size
            and 0 <= rank_recv_from < self.world_size
        ):
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
        try:
            recv = self._prepare(to_send=to_send, to_recv=to_recv)
            self.dispatch()
            return recv
        except Exception:
            # This wrapper is part of a propagated traceback. Do not let its
            # argument locals undo `_prepare()`/`dispatch()` cleanup.
            to_send = None  # type: ignore[assignment]
            to_recv = None
            recv = None  # type: ignore[assignment]
            raise

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

        try:
            return self._prepare(to_send=to_send, to_recv=to_recv)
        except Exception:
            to_send = None  # type: ignore[assignment]
            to_recv = None
            raise

    def exchange(
        self, to_send: torch.Tensor, to_recv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        try:
            recv = self.enqueue_to_dispatch(to_send=to_send, to_recv=to_recv)
            self.wait_until_finished()
            return recv
        except Exception:
            to_send = None  # type: ignore[assignment]
            to_recv = None
            recv = None  # type: ignore[assignment]
            raise

    def dispatch(self) -> None:
        if self.is_self_comm:
            return
        if self._work is not None:
            raise RuntimeError("cannot dispatch with unfinished communication.")
        try:
            self._work = dist.batch_isend_irecv(self._queue)
        except Exception as exc:
            # P2POp owns its send/receive tensor.  A failed launch must not keep
            # those potentially large Ring blocks alive until this communication
            # object is garbage-collected by a much later Runner cleanup.  The
            # backend exception traceback can also retain the operation-list
            # argument, so detach it before propagating the failure.
            self._queue = []
            self._work = None
            raise detach_rank_local_error_traceback(exc)

    def wait_until_finished(self) -> None:
        if self.is_self_comm:
            return
        if self._work is None:
            raise RuntimeError("cannot wait without dispatched communication.")
        first_error: Exception | None = None
        pending_work: dist.Work | None = None
        try:
            # A batched send/receive returns one Work per operation.  Retain the
            # first error but still observe the remaining handles so a failed
            # wait cannot silently abandon the peer operation from the same
            # already-dispatched Ring step.
            for pending_work in self._work:
                try:
                    pending_work.wait()
                except Exception as exc:
                    detached = detach_rank_local_error_traceback(exc)
                    if first_error is None:
                        first_error = detached
        finally:
            # Both P2POp and Work may retain CUDA tensors.  Release them before
            # propagating the communication error so post-failure cleanup has a
            # chance to return their storage to the allocator.
            self._queue = []
            self._work = None
            # Python keeps a for-loop target alive after the loop.  The Work
            # object may itself own CUDA request tensors, and the new traceback
            # created below retains this frame, so clear the local explicitly.
            pending_work = None
        if first_error is not None:
            raise first_error

    def _prepare(
        self,
        to_send: torch.Tensor,
        to_recv: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self._work is not None:
            raise RuntimeError(
                "cannot queue communication while a dispatched batch is unfinished."
            )
        try:
            if self.is_self_comm:
                recv = to_send.detach().clone() if to_recv is None else to_recv
                if to_recv is not None:
                    recv.copy_(to_send)
                return recv
            recv = torch.empty_like(to_send) if to_recv is None else to_recv
            send_op = dist.P2POp(
                dist.isend,
                to_send,
                self.global_send_to,
                group=self.group,
            )
            recv_op = dist.P2POp(
                dist.irecv,
                recv,
                self.global_recv_from,
                group=self.group,
            )
            self._queue.extend(
                [send_op, recv_op] if self.parity else [recv_op, send_op]
            )
            return recv
        except Exception as exc:
            # `prepare_to_dispatch()` supports building one multi-buffer batch.
            # If a later receive/P2POp preparation fails, earlier queued ops
            # still own their payloads and the propagated traceback owns the
            # new payload. The batch can no longer be dispatched safely, so
            # cancel it transactionally and detach all tensor-owning frames.
            self._queue = []
            detached_error = detach_rank_local_error_traceback(exc)
            to_send = None  # type: ignore[assignment]
            to_recv = None
            recv = None  # type: ignore[assignment]
            send_op = None  # type: ignore[assignment]
            recv_op = None  # type: ignore[assignment]
            raise detached_error


def gather_tensor_by_ring(
    local_tensor: torch.Tensor,
    *,
    comm: One2OneComm,
    group: dist.ProcessGroup,
    local_index: int,
    side: int,
    dim: int,
    length: int | None = None,
    description: str,
) -> torch.Tensor:
    """Gather equal-width ring blocks into one tensor without transient shards.

    Every rank reserves its output and reusable receive buffers before P2P starts.
    A rank-local copy failure is retained until all scheduled exchanges have been
    drained, so peers cannot be stranded in the ring.
    """

    if side <= 0:
        raise ValueError(f"ring side must be positive, got {side}.")
    if side == 1:
        normalized_dim = dim if dim >= 0 else local_tensor.ndim + dim
        if normalized_dim < 0 or normalized_dim >= local_tensor.ndim:
            raise IndexError(f"ring gather dim {normalized_dim} is out of range.")
        local_width = int(local_tensor.shape[normalized_dim])
        output_length = local_width if length is None else int(length)
        if output_length < 0 or output_length > local_width:
            raise ValueError(
                f"ring gather length must be in [0, {local_width}], "
                f"got {output_length}."
            )
        return local_tensor.narrow(normalized_dim, 0, output_length).contiguous()

    def _allocate_ring() -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        int,
        int,
        int,
        int,
    ]:
        normalized_dim = dim if dim >= 0 else local_tensor.ndim + dim
        if normalized_dim < 0 or normalized_dim >= local_tensor.ndim:
            raise IndexError(f"ring gather dim {normalized_dim} is out of range.")
        normalized_index = int(local_index)
        if normalized_index < 0 or normalized_index >= int(side):
            raise ValueError(
                f"ring local index must be in [0, {side}), got {normalized_index}."
            )
        local_width = int(local_tensor.shape[normalized_dim])
        padded_length = int(side) * local_width
        output_length = padded_length if length is None else int(length)
        if output_length < 0 or output_length > padded_length:
            raise ValueError(
                f"ring gather length must be in [0, {padded_length}], "
                f"got {output_length}."
            )
        out_shape = list(local_tensor.shape)
        out_shape[normalized_dim] = output_length
        local_block = local_tensor.contiguous()
        return (
            local_tensor.new_empty(out_shape),
            local_block,
            tuple(torch.empty_like(local_block) for _ in range(min(2, side - 1))),
            normalized_dim,
            local_width,
            output_length,
            normalized_index,
        )

    buffers = run_group_rank_action_synchronized(
        _allocate_ring,
        group=group,
        description=f"{description} allocation",
    )
    if buffers is None:  # pragma: no cover - every rank runs the action
        raise RuntimeError(f"{description} returned no buffers.")
    out, ready, recv_buffers, dim, local_width, output_length, local_index = buffers
    assembly_error: Exception | None = None
    for step in range(side):
        source_index = (int(local_index) + step) % int(side)
        output_start = source_index * local_width
        output_end = min(output_start + local_width, output_length)
        if output_start < output_end and assembly_error is None:
            try:
                out.narrow(dim, output_start, output_end - output_start).copy_(
                    ready.narrow(dim, 0, output_end - output_start)
                )
            except Exception as exc:
                assembly_error = detach_rank_local_error_traceback(exc)
        if step + 1 < side:
            ready = comm.exchange(
                ready,
                to_recv=recv_buffers[step % len(recv_buffers)],
            )

    def _finish_ring() -> torch.Tensor:
        if assembly_error is not None:
            raise assembly_error
        return out

    result = run_group_rank_action_synchronized(
        _finish_ring,
        group=group,
        description=f"{description} assembly",
    )
    if result is None:  # pragma: no cover - every rank runs the action
        raise RuntimeError(f"{description} returned no result.")
    return result


def exchange_tensor_synchronized(
    local_tensor: torch.Tensor,
    *,
    comm: One2OneComm,
    group: dist.ProcessGroup,
    description: str,
    prepare: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Preflight one P2P exchange's source and destination on every rank."""

    def _allocate_exchange() -> tuple[torch.Tensor, torch.Tensor]:
        source = (
            local_tensor.contiguous()
            if prepare is None
            else prepare(local_tensor).contiguous()
        )
        return source, torch.empty_like(source)

    buffers = run_group_rank_action_synchronized(
        _allocate_exchange,
        group=group,
        description=f"{description} allocation",
    )
    if buffers is None:  # pragma: no cover - every rank runs the action
        raise RuntimeError(f"{description} returned no buffers.")
    source, destination = buffers
    return comm.exchange(source, to_recv=destination)


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
    """Ring communication handles for the maintained 1xP Fold-CP mesh."""

    group_2d: dist.ProcessGroup
    layout: FoldCP2DLayout

    def __post_init__(self) -> None:
        self.group_2d = _require_dist(self.group_2d)
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
