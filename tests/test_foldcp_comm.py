# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

from contextlib import nullcontext
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from opendde.distributed.foldcp.pair_sharding import FoldCPPairShardSpec


def test_canonical_launch_releases_completed_chunks():
    from opendde.distributed.foldcp.launch import (
        foldcp_module_with_canonical_launch_chunks,
    )

    prior_refs = []
    calls = 0

    def module(value):
        nonlocal calls, prior_refs
        if calls:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        projected = value + 1
        prior_refs = [weakref.ref(value), weakref.ref(projected)]
        calls += 1
        return projected

    result = foldcp_module_with_canonical_launch_chunks(
        module,
        torch.zeros(4, 1),
        launch_rows=2,
    )

    assert torch.equal(result, torch.ones_like(result))
    assert calls == 2


def test_ring_dispatch_failure_releases_queued_tensor_references(monkeypatch):
    from opendde.distributed.foldcp import comm

    ring = comm.One2OneComm.__new__(comm.One2OneComm)
    ring.is_self_comm = False
    queued_tensor = torch.empty(1)
    queued_ref = weakref.ref(queued_tensor)
    ring._queue = [queued_tensor]
    ring._work = None

    monkeypatch.setattr(
        comm.dist,
        "batch_isend_irecv",
        lambda _queue: (_ for _ in ()).throw(RuntimeError("P2P launch failed")),
    )

    with pytest.raises(RuntimeError, match="P2P launch failed") as captured:
        ring.dispatch()

    assert ring._queue == []
    assert ring._work is None
    del queued_tensor
    gc.collect()
    assert queued_ref() is None
    assert captured.value is not None


def test_ring_wait_failure_drains_batch_and_releases_request_state():
    from opendde.distributed.foldcp import comm

    events = []

    class Work:
        def __init__(self, name, payload, error=None):
            self.name = name
            self.payload = payload
            self.error = error

        def wait(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    ring = comm.One2OneComm.__new__(comm.One2OneComm)
    ring.is_self_comm = False
    queued_tensor = torch.empty(1)
    queued_ref = weakref.ref(queued_tensor)
    ring._queue = [queued_tensor]
    ring._work = [
        Work("send", queued_tensor, RuntimeError("send wait failed")),
        Work("receive", queued_tensor),
    ]

    with pytest.raises(RuntimeError, match="send wait failed") as captured:
        ring.wait_until_finished()

    assert events == ["send", "receive"]
    assert ring._queue == []
    assert ring._work is None
    del queued_tensor
    gc.collect()
    assert queued_ref() is None
    assert captured.value is not None


def test_direct_p2p_batch_failure_drains_and_releases_operation_payloads(monkeypatch):
    from opendde.distributed.foldcp import comm

    events = []

    class Work:
        def __init__(self, name, payload, error=None):
            self.name = name
            self.payload = payload
            self.error = error

        def wait(self):
            events.append(self.name)
            if self.error is not None:
                raise self.error

    payload = torch.empty(1)
    payload_ref = weakref.ref(payload)
    operations = [payload]
    work_items = [
        Work("send", payload, RuntimeError("direct P2P wait failed")),
        Work("receive", payload),
    ]
    monkeypatch.setattr(comm.dist, "batch_isend_irecv", lambda _ops: work_items)

    retained_error = None
    try:
        comm.dispatch_p2p_batch_and_wait(operations)  # type: ignore[arg-type]
    except RuntimeError as exc:
        retained_error = exc

    assert retained_error is not None
    assert "direct P2P wait failed" in str(retained_error)
    assert events == ["send", "receive"]
    assert operations == []
    assert work_items == []
    del payload
    gc.collect()
    assert payload_ref() is None


def test_ring_rejects_queueing_while_dispatched_batch_is_unfinished():
    from opendde.distributed.foldcp import comm

    ring = comm.One2OneComm.__new__(comm.One2OneComm)
    ring.is_self_comm = False
    ring._queue = []
    ring._work = [object()]

    with pytest.raises(RuntimeError, match="dispatched batch is unfinished"):
        ring.prepare_to_dispatch(torch.empty(1))

    assert ring._queue == []
    assert ring._work == [ring._work[0]]


def test_ring_prepare_failure_cancels_queued_batch_and_releases_payloads(monkeypatch):
    from opendde.distributed.foldcp import comm

    ring = comm.One2OneComm.__new__(comm.One2OneComm)
    ring.is_self_comm = False
    ring._work = None
    queued_payload = torch.empty(1)
    next_payload = torch.empty(1)
    queued_ref = weakref.ref(queued_payload)
    next_ref = weakref.ref(next_payload)
    ring._queue = [queued_payload]

    monkeypatch.setattr(
        comm.torch,
        "empty_like",
        lambda _tensor: (_ for _ in ()).throw(RuntimeError("receive allocation OOM")),
    )

    retained_error = None
    try:
        ring.prepare_to_dispatch(next_payload)
    except RuntimeError as exc:
        retained_error = exc

    assert retained_error is not None
    assert "receive allocation OOM" in str(retained_error)
    assert ring._queue == []
    del queued_payload, next_payload
    gc.collect()
    assert queued_ref() is None
    assert next_ref() is None


@pytest.mark.parametrize("send_rank,recv_rank", [(-1, 0), (0, -1), (2, 0), (0, 2)])
def test_ring_rejects_peer_rank_outside_group(monkeypatch, send_rank, recv_rank):
    from opendde.distributed.foldcp import comm

    group = object()
    monkeypatch.setattr(comm, "_require_dist", lambda selected: selected)
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)

    with pytest.raises(ValueError, match="ranks inside the process group"):
        comm.One2OneComm(group, send_rank, recv_rank)


def test_nccl_rank_action_success_uses_scalar_fast_path(monkeypatch):
    from opendde.distributed.foldcp import comm

    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: False)
    monkeypatch.setattr(
        comm.dist,
        "all_gather_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("success must not use object gather")
        ),
    )

    result = comm.run_group_rank_action_synchronized(
        lambda: "ready",
        group=object(),
        description="fast success",
    )

    assert result == "ready"


def test_nccl_rank_action_failure_still_gathers_error_strings(monkeypatch):
    from opendde.distributed.foldcp import comm

    gathered_errors = []

    def gather(output, local_error, *, group):
        gathered_errors.append(local_error)
        output[:] = [local_error, ""]

    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: True)
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(comm.dist, "all_gather_object", gather)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="local fast-path OOM"):
        comm.run_group_rank_action_synchronized(
            lambda: (_ for _ in ()).throw(RuntimeError("local fast-path OOM")),
            group=object(),
            description="fast failure",
        )

    assert len(gathered_errors) == 1
    assert "local fast-path OOM" in gathered_errors[0]


def test_peer_rank_action_failure_releases_successful_local_result(monkeypatch):
    from opendde.distributed.foldcp import comm

    local_result_ref = None

    class LargeLocalResult:
        pass

    def allocate_local_result():
        nonlocal local_result_ref
        result = LargeLocalResult()
        local_result_ref = weakref.ref(result)
        return result

    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: True)
    monkeypatch.setattr(
        comm,
        "_gather_rank_errors",
        lambda *_args, **_kwargs: ["", "group rank 1 peer allocation OOM"],
    )

    with pytest.raises(RuntimeError, match="peer allocation OOM"):
        comm.run_group_rank_action_synchronized(
            allocate_local_result,
            group=object(),
            description="large peer-synchronized allocation",
        )

    assert local_result_ref is not None
    assert local_result_ref() is None


def test_peer_rank_action_failure_releases_action_closure(monkeypatch):
    from opendde.distributed.foldcp import comm

    action_payload_ref = None

    class LargeActionPayload:
        pass

    def make_action():
        nonlocal action_payload_ref
        payload = LargeActionPayload()
        action_payload_ref = weakref.ref(payload)
        return lambda payload=payload: None

    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: True)
    monkeypatch.setattr(
        comm,
        "_gather_rank_errors",
        lambda *_args, **_kwargs: ["", "group rank 1 peer allocation OOM"],
    )

    retained_error = None
    try:
        comm.run_group_rank_action_synchronized(
            make_action(),
            group=object(),
            description="closure-owning peer-synchronized allocation",
        )
    except RuntimeError as exc:
        retained_error = exc

    gc.collect()
    assert retained_error is not None
    assert action_payload_ref is not None
    assert action_payload_ref() is None


def test_nccl_rank_action_failure_reports_over_registered_gloo_group(monkeypatch):
    from opendde.distributed.foldcp import comm

    data_group = object()
    control_group = object()
    gather_groups = []
    comm.unregister_foldcp_cpu_control_group()

    def backend(group):
        return (
            comm.dist.Backend.GLOO if group is control_group else comm.dist.Backend.NCCL
        )

    def gather(output, local_error, *, group):
        gather_groups.append(group)
        output[:] = [local_error, ""]

    monkeypatch.setattr(comm.dist, "get_backend", backend)
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(comm.dist, "all_gather_object", gather)
    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: True)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: False)

    try:
        comm.register_foldcp_cpu_control_group(control_group)
        with pytest.raises(RuntimeError, match="local protected OOM"):
            comm.run_group_rank_action_synchronized(
                lambda: (_ for _ in ()).throw(RuntimeError("local protected OOM")),
                group=data_group,
                description="protected allocation",
            )
    finally:
        comm.unregister_foldcp_cpu_control_group(control_group)

    assert gather_groups == [control_group]


def test_nccl_status_is_reserved_before_protected_action(monkeypatch):
    from opendde.distributed.foldcp import comm

    comm.clear_foldcp_communication_cache()
    group = object()
    events = []
    status = torch.empty((), dtype=torch.int32)

    monkeypatch.setattr(comm.dist, "get_backend", lambda _group: comm.dist.Backend.NCCL)
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(comm.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        comm.torch,
        "empty",
        lambda *args, **kwargs: events.append("reserve") or status,
    )
    monkeypatch.setattr(
        comm.dist,
        "all_gather_object",
        lambda output, value, **_kwargs: (
            events.append("reservation-handshake"),
            output.__setitem__(0, value),
        ),
    )
    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: False)

    result = comm.run_group_rank_action_synchronized(
        lambda: events.append("action") or "ready",
        group=group,
        description="large protected allocation",
    )

    assert result == "ready"
    assert events == ["reserve", "action"]


def test_nccl_status_clear_oom_keeps_prearmed_failure(monkeypatch):
    from opendde.distributed.foldcp import comm

    comm.clear_foldcp_communication_cache()
    group = object()
    status = torch.zeros((), dtype=torch.int32)
    gathered_errors = []
    comm._NCCL_STATUS_TENSORS[(id(group), 0)] = status

    monkeypatch.setattr(comm, "_prime_nccl_group_status", lambda _group: None)
    monkeypatch.setattr(comm.dist, "get_backend", lambda _group: comm.dist.Backend.NCCL)
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(comm.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(comm.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(comm.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        comm,
        "_clear_nccl_group_status",
        lambda _group: (_ for _ in ()).throw(
            torch.OutOfMemoryError("status clear OOM")
        ),
    )

    def gather(output, local_error, **_kwargs):
        gathered_errors.append(local_error)
        output[0] = local_error

    monkeypatch.setattr(comm.dist, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="status clear OOM"):
        comm.run_group_rank_action_synchronized(
            lambda: "ready",
            group=group,
            description="protected action",
        )

    assert status.item() == 1
    assert gathered_errors == []
    comm.clear_foldcp_communication_cache()


def test_nccl_status_reservation_failure_stops_protected_action(monkeypatch):
    from opendde.distributed.foldcp import comm

    comm.clear_foldcp_communication_cache()
    group = object()
    actions = []

    monkeypatch.setattr(comm.dist, "get_backend", lambda _group: comm.dist.Backend.NCCL)
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(comm.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(comm.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        comm.torch,
        "empty",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory while reserving status")
        ),
    )

    def gather(output, local_error, **_kwargs):
        output[:] = [local_error, ""]

    monkeypatch.setattr(comm.dist, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="reserving status"):
        comm.run_group_rank_action_synchronized(
            lambda: actions.append("action"),
            group=group,
            description="large protected allocation",
        )

    assert actions == []


def test_rank_local_allocation_failure_is_synchronized(monkeypatch):
    from opendde.distributed.foldcp import comm

    group = object()

    def gather(output, local_error, *, group):
        output[:] = [local_error, ""]

    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(comm.dist, "all_gather_object", gather)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        comm.run_group_rank_action_synchronized(
            lambda: (_ for _ in ()).throw(
                RuntimeError("CUDA out of memory while allocating output")
            ),
            group=group,
            description="destination allocation",
        )


def test_rank_local_failure_still_synchronizes_when_cache_cleanup_fails(monkeypatch):
    from opendde.distributed.foldcp import comm

    group = object()
    gathered_errors = []

    def gather(output, local_error, *, group):
        gathered_errors.append(local_error)
        output[:] = [local_error, ""]

    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(comm.dist, "all_gather_object", gather)
    monkeypatch.setattr(comm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        comm.torch.cuda,
        "empty_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("deferred CUDA error")),
    )

    with pytest.raises(RuntimeError, match="cache cleanup also failed"):
        comm.run_group_rank_action_synchronized(
            lambda: (_ for _ in ()).throw(RuntimeError("original allocation OOM")),
            group=group,
            description="destination allocation",
        )

    assert len(gathered_errors) == 1
    assert "original allocation OOM" in gathered_errors[0]
    assert "deferred CUDA error" in gathered_errors[0]


def test_internal_metric_entry_failure_stops_stage_body(monkeypatch):
    from opendde.model import opendde

    body_calls = []
    exits = []

    class MetricContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            exits.append(args)
            return None

    def synchronize(action, *, description, **_kwargs):
        if description.endswith("metric initialization"):
            action()
            raise RuntimeError("remote metric JSONL failure")
        return action()

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="remote metric JSONL failure"):
        with opendde._synchronized_foldcp_stage_context(
            MetricContext(),
            group=object(),
            stage_name="pairformer",
        ):
            body_calls.append("model")

    assert body_calls == []
    assert len(exits) == 1
    assert exits[0][0] is RuntimeError


def test_internal_metric_exit_propagates_stage_body_failure(monkeypatch):
    from opendde.model import opendde

    exits = []

    class MetricContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, _traceback):
            exits.append((exc_type, exc))
            return None

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    with pytest.raises(RuntimeError, match="stage allocation OOM"):
        with opendde._synchronized_foldcp_stage_context(
            MetricContext(),
            group=object(),
            stage_name="confidence",
        ):
            raise RuntimeError("stage allocation OOM")

    assert len(exits) == 1
    assert exits[0][0] is RuntimeError
    assert str(exits[0][1]) == "stage allocation OOM"


def test_internal_stage_boundary_remains_synchronized_without_metrics(monkeypatch):
    from opendde.model import opendde

    group = object()
    descriptions = []
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: SimpleNamespace(group_2d=group),
    )
    monkeypatch.setattr(
        opendde.FoldCPConfig,
        "from_environment",
        lambda: opendde.FoldCPConfig(
            mode="distributed",
            size_dp=1,
            size_cp=3,
            metrics_jsonl="",
        ),
    )

    def synchronize(action, *, group: object, description: str):
        descriptions.append(description)
        return action()

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with opendde.OpenDDE._foldcp_stage_context(module, "confidence", 401):
        pass

    assert descriptions == [
        "Fold-CP confidence metric initialization",
        "Fold-CP confidence metric finalization",
    ]


def test_internal_metric_failure_releases_traceback_tensors_before_handshake(
    monkeypatch,
):
    from opendde.model import opendde

    tensor_refs = []
    handshake_observations = []

    class MetricContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return None

    def fail_with_tensor_in_frame():
        temporary = torch.empty(1)
        tensor_refs.append(weakref.ref(temporary))
        raise RuntimeError("stage allocation OOM")

    def synchronize(action, *, description, **_kwargs):
        if description.endswith("metric/error finalization"):
            gc.collect()
            handshake_observations.append(tensor_refs[0]() is None)
        return action()

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="stage allocation OOM"):
        with opendde._synchronized_foldcp_stage_context(
            MetricContext(),
            group=object(),
            stage_name="confidence",
        ):
            fail_with_tensor_in_frame()

    assert handshake_observations == [True]


def test_internal_metric_double_failure_releases_both_tracebacks_before_handshake(
    monkeypatch,
):
    from opendde.model import opendde

    tensor_refs = []
    handshake_observations = []

    class MetricContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            metric_temporary = torch.empty(1)
            tensor_refs.append(weakref.ref(metric_temporary))
            raise RuntimeError("metric finalization OOM")

    def fail_with_tensor_in_frame():
        stage_temporary = torch.empty(1)
        tensor_refs.append(weakref.ref(stage_temporary))
        raise RuntimeError("stage allocation OOM")

    def synchronize(action, *, description, **_kwargs):
        if description.endswith("metric/error finalization"):
            gc.collect()
            handshake_observations.append([ref() is None for ref in tensor_refs])
        return action()

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(
        RuntimeError,
        match="stage allocation OOM; metric finalization also failed",
    ):
        with opendde._synchronized_foldcp_stage_context(
            MetricContext(),
            group=object(),
            stage_name="confidence",
        ):
            fail_with_tensor_in_frame()

    assert handshake_observations == [[True, True]]


def test_stage_metric_releases_failed_payload_before_cuda_finalization(monkeypatch):
    from opendde.distributed.foldcp import metrics
    from opendde.distributed.foldcp.config import FoldCPConfig

    tensor_refs = []
    finalization_observations = []
    sync_calls = 0

    def synchronize(_device=None):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            gc.collect()
            finalization_observations.append(tensor_refs[0]() is None)

    def fail_with_tensor_in_frame():
        temporary = torch.empty(1)
        tensor_refs.append(weakref.ref(temporary))
        raise RuntimeError("CUDA out of memory in stage allocation")

    monkeypatch.setattr(metrics, "_sync_device", synchronize)
    monkeypatch.setattr(metrics, "_cuda_available", lambda _device=None: False)
    recorder = metrics.FoldCPBenchmarkRecorder()

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        with metrics.measure_foldcp_stage(
            task_id="traceback-release",
            stage_name="model_forward",
            foldcp_config=FoldCPConfig(),
            recorder=recorder,
        ):
            fail_with_tensor_in_frame()

    assert finalization_observations == [True]
    assert recorder.records[-1].status == "oom"


def test_confidence_assembly_releases_failed_payload_before_cuda_cleanup(
    monkeypatch,
):
    from opendde.distributed.foldcp import confidence

    class Payload:
        pass

    payload_reference = None
    cleanup_observations = []
    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        cp_global_ranks=(0,),
        layout=SimpleNamespace(numel=1, to_coord=lambda _rank: (0, 0)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )

    def fail_copy(*_args, **_kwargs):
        nonlocal payload_reference
        payload = Payload()
        payload_reference = weakref.ref(payload)
        raise RuntimeError("confidence assembly OOM")

    def observe_cleanup(error, *, attempt):
        gc.collect()
        cleanup_observations.append((attempt, payload_reference() is None))
        return error

    monkeypatch.setattr(confidence.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(confidence.dist, "gather", lambda *args, **kwargs: None)
    monkeypatch.setattr(confidence, "_copy_pair_shard_into_output", fail_copy)
    monkeypatch.setattr(
        confidence,
        "_append_cuda_cache_cleanup_error",
        observe_cleanup,
    )

    error = confidence._gather_pair_logit_chunk_to_rank0(
        full_output=torch.zeros(2, 2, 1),
        local_chunk=torch.zeros(2, 2, 1),
        z_pair_spec=spec,
        mesh=mesh,
        row_start=0,
        row_end=2,
        gathered=[torch.zeros(2, 2, 1)],
    )

    assert "confidence assembly OOM" in error
    assert cleanup_observations == [(False, True)]


def test_remote_rank_allocation_failure_reaches_local_rank(monkeypatch):
    from opendde.distributed.foldcp import comm

    group = object()

    def gather(output, _local_error, *, group):
        output[:] = ["", "group rank 1 allocation failed: OOM"]

    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(comm.dist, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="rank 1 allocation failed"):
        comm.run_group_rank_action_synchronized(
            None,
            group=group,
            description="destination allocation",
        )


def test_synchronized_rank_action_returns_local_result(monkeypatch):
    from opendde.distributed.foldcp import comm

    group = object()

    def gather(output, local_error, *, group):
        output[:] = [local_error]

    monkeypatch.setattr(comm.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(comm.dist, "all_gather_object", gather)

    result = comm.run_group_rank_action_synchronized(
        lambda: "allocated",
        group=group,
        description="destination allocation",
    )

    assert result == "allocated"


def test_ring_gather_does_not_exchange_after_remote_allocation_oom(monkeypatch):
    from opendde.distributed.foldcp import comm

    exchanges = []
    ring_comm = SimpleNamespace(
        exchange=lambda *args, **kwargs: exchanges.append("exchange")
    )
    monkeypatch.setattr(
        comm,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote row-slab allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote row-slab allocation OOM"):
        comm.gather_tensor_by_ring(
            torch.zeros(1, 1, 1),
            comm=ring_comm,
            group=object(),
            local_index=0,
            side=3,
            dim=1,
            description="test row-slab ring",
        )

    assert exchanges == []


@pytest.mark.parametrize(
    ("dim", "local_index", "message"),
    [
        (3, 0, "ring gather dim"),
        (1, 3, "ring local index"),
    ],
)
def test_ring_gather_validation_runs_inside_rank_action(
    monkeypatch,
    dim,
    local_index,
    message,
):
    from opendde.distributed.foldcp import comm

    actions = []

    def synchronize(action, *, description, **_kwargs):
        actions.append(description)
        return action()

    monkeypatch.setattr(comm, "run_group_rank_action_synchronized", synchronize)

    with pytest.raises((IndexError, ValueError), match=message):
        comm.gather_tensor_by_ring(
            torch.zeros(1, 1, 1),
            comm=SimpleNamespace(exchange=lambda *args, **kwargs: None),
            group=object(),
            local_index=local_index,
            side=3,
            dim=dim,
            description="test row-slab ring",
        )

    assert actions == ["test row-slab ring allocation"]


def test_ring_gather_reuses_receive_buffers_and_orders_blocks(monkeypatch):
    from opendde.distributed.foldcp import comm

    receive_ids = []
    next_value = iter((1.0, 2.0, 3.0))

    def exchange(_send, *, to_recv):
        receive_ids.append(id(to_recv))
        to_recv.fill_(next(next_value))
        return to_recv

    monkeypatch.setattr(
        comm,
        "run_group_rank_action_synchronized",
        lambda action, **kwargs: action(),
    )
    result = comm.gather_tensor_by_ring(
        torch.zeros(1, 1, 1),
        comm=SimpleNamespace(exchange=exchange),
        group=object(),
        local_index=0,
        side=4,
        dim=1,
        description="test row-slab ring",
    )

    assert result.flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert len(receive_ids) == 3
    assert len(set(receive_ids)) == 2
    assert receive_ids[0] == receive_ids[2]


def test_synchronized_exchange_does_not_dispatch_after_remote_oom(monkeypatch):
    from opendde.distributed.foldcp import comm

    exchanges = []
    monkeypatch.setattr(
        comm,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote transpose allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote transpose allocation OOM"):
        comm.exchange_tensor_synchronized(
            torch.zeros(2, 3),
            comm=SimpleNamespace(
                exchange=lambda *args, **kwargs: exchanges.append("exchange")
            ),
            group=object(),
            description="test transpose",
            prepare=lambda tensor: tensor.transpose(0, 1),
        )

    assert exchanges == []


def test_synchronized_exchange_prepares_source_and_reuses_destination(monkeypatch):
    from opendde.distributed.foldcp import comm

    monkeypatch.setattr(
        comm,
        "run_group_rank_action_synchronized",
        lambda action, **kwargs: action(),
    )

    def exchange(source, *, to_recv):
        assert source.shape == (3, 2)
        assert to_recv.shape == source.shape
        to_recv.copy_(source + 1)
        return to_recv

    result = comm.exchange_tensor_synchronized(
        torch.arange(6).reshape(2, 3),
        comm=SimpleNamespace(exchange=exchange),
        group=object(),
        description="test transpose",
        prepare=lambda tensor: tensor.transpose(0, 1),
    )

    assert torch.equal(result, torch.arange(6).reshape(2, 3).transpose(0, 1) + 1)


def test_diffusion_bias_transpose_stops_before_collective_after_remote_oom(
    monkeypatch,
):
    from opendde.model.modules import transformer

    collectives = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote diffusion transpose allocation OOM")
        ),
    )
    monkeypatch.setattr(
        transformer.dist,
        "all_to_all_single",
        lambda *args, **kwargs: collectives.append("all_to_all"),
    )

    with pytest.raises(RuntimeError, match="remote diffusion transpose allocation OOM"):
        transformer.AttentionPairBias._foldcp_transpose_bias_to_query_rows(
            bias_tile=torch.zeros(1, 4, 2),
            n_token=4,
            mesh=mesh,
        )

    assert collectives == []


def test_diffusion_bias_workspace_reset_oom_drains_all_to_all(monkeypatch):
    from opendde.model.modules import transformer

    collectives = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        transformer,
        "_zero_foldcp_drain_buffer",
        lambda _tensor: (_ for _ in ()).throw(
            torch.OutOfMemoryError("local diffusion bias reset OOM")
        ),
    )

    def all_to_all(output, source, **_kwargs):
        collectives.append("all_to_all")
        output.copy_(source)

    monkeypatch.setattr(transformer.dist, "all_to_all_single", all_to_all)

    with pytest.raises(RuntimeError, match="bias preparation failed"):
        transformer.AttentionPairBias._foldcp_transpose_bias_to_query_rows(
            bias_tile=torch.zeros(1, 4, 2),
            n_token=4,
            mesh=mesh,
            workspace=transformer._FoldCPAttentionWorkspace(),
        )

    assert collectives == ["all_to_all"]


def test_diffusion_bias_failure_flag_clear_oom_drains_all_to_all(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from opendde.model.modules import transformer

    gathered_flags = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    def all_to_all(output, source, **_kwargs):
        query_tile_rows = 2
        gathered_flags.append(source[..., query_tile_rows, :].detach().clone())
        output.copy_(source)

    monkeypatch.setattr(transformer.dist, "all_to_all_single", all_to_all)

    class FailFlagRowZero(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            # P=2, N=4 and the fixture bias shape produce four elements in the
            # source-local failure row. The preceding payload reset has eight.
            if func is torch.ops.aten.zero_.default and args[0].numel() == 4:
                raise torch.OutOfMemoryError("bias failure-flag clear OOM")
            return func(*args, **kwargs)

    with (
        FailFlagRowZero(),
        pytest.raises(RuntimeError, match="bias preparation failed"),
    ):
        transformer.AttentionPairBias._foldcp_transpose_bias_to_query_rows(
            bias_tile=torch.zeros(1, 4, 2),
            n_token=4,
            mesh=mesh,
            workspace=transformer._FoldCPAttentionWorkspace(),
        )

    assert len(gathered_flags) == 1
    assert torch.all(gathered_flags[0] == 1)


def test_diffusion_bias_workspace_excludes_failure_flag_from_payload(monkeypatch):
    from opendde.model.modules import transformer

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )

    def all_to_all(output, source, **_kwargs):
        # Simulate two source ranks. Payload rows are distinct, while the final
        # row is the transport-only failure flag and must never enter row_bias.
        output[0, ..., :2, :].copy_(source[0, ..., :2, :])
        output[1, ..., :2, :].copy_(source[0, ..., :2, :] + 10)
        output[..., 2, :].zero_()

    monkeypatch.setattr(transformer.dist, "all_to_all_single", all_to_all)
    bias_tile = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])

    row_bias = transformer.AttentionPairBias._foldcp_transpose_bias_to_query_rows(
        bias_tile=bias_tile,
        n_token=4,
        mesh=mesh,
        workspace=transformer._FoldCPAttentionWorkspace(),
    )

    assert row_bias.shape == (1, 2, 4)
    assert torch.equal(
        row_bias,
        torch.tensor([[[1.0, 2.0, 11.0, 12.0], [3.0, 4.0, 13.0, 14.0]]]),
    )


def test_diffusion_output_allocation_failure_stops_before_gather(monkeypatch):
    from opendde.model.modules import transformer

    collectives = []
    module = SimpleNamespace(
        attention=SimpleNamespace(
            _prep_qkv=lambda **_kwargs: (
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
            ),
        ),
        _foldcp_valid_ranges=lambda _spec: (0, 4, 0, 2, 4, 2),
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    z_spec = SimpleNamespace(local_shape=(4, 2, 1), pair_dims=(0, 1))
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote diffusion output allocation OOM")
        ),
    )
    monkeypatch.setattr(
        transformer.dist,
        "all_to_all_single",
        lambda *args, **kwargs: collectives.append("all_to_all"),
    )

    with pytest.raises(RuntimeError, match="remote diffusion output allocation OOM"):
        transformer.AttentionPairBias.standard_multihead_attention_foldcp_local_z(
            module,
            q=torch.zeros(1, 4, 2),
            kv=torch.zeros(1, 4, 2),
            z_local=torch.zeros(4, 2, 1),
            z_spec=z_spec,
            mesh=mesh,
            projected_bias_local=transformer.FoldCPQueryOwnedAttentionBias(
                torch.zeros(1, 4, 4)
            ),
        )

    assert collectives == []


def test_diffusion_local_attention_oom_drains_gather_before_raising(monkeypatch):
    from opendde.model.modules import transformer

    gather_calls = []
    module = SimpleNamespace(
        attention=SimpleNamespace(
            _prep_qkv=lambda **_kwargs: (
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
            ),
            use_efficient_implementation=False,
            _wrap_up=lambda raw, _q: raw,
        ),
        _foldcp_valid_ranges=lambda _spec: (0, 4, 0, 2, 4, 2),
        _align_bias_to_query=lambda bias, _q, **_kwargs: bias,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    z_spec = SimpleNamespace(local_shape=(4, 2, 1), pair_dims=(0, 1))

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        transformer,
        "_attention",
        lambda **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("local attention OOM")
        ),
    )

    def _all_gather(output, source, **_kwargs):
        gather_calls.append("all_gather")
        output[: source.shape[0]].copy_(source)
        output[source.shape[0] :].copy_(source)

    monkeypatch.setattr(transformer.dist, "all_gather_into_tensor", _all_gather)

    with pytest.raises(RuntimeError, match="local attention failed"):
        transformer.AttentionPairBias.standard_multihead_attention_foldcp_local_z(
            module,
            q=torch.zeros(1, 4, 2),
            kv=torch.zeros(1, 4, 2),
            z_local=torch.zeros(4, 2, 1),
            z_spec=z_spec,
            mesh=mesh,
            projected_bias_local=transformer.FoldCPQueryOwnedAttentionBias(
                torch.zeros(1, 2, 4)
            ),
        )

    assert gather_calls == ["all_gather"]


def test_diffusion_workspace_reset_oom_drains_gather_before_raising(monkeypatch):
    from opendde.model.modules import transformer

    gather_calls = []
    module = SimpleNamespace(
        attention=SimpleNamespace(
            _prep_qkv=lambda **_kwargs: (
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
            ),
            use_efficient_implementation=False,
            _wrap_up=lambda raw, _q: raw,
        ),
        _foldcp_valid_ranges=lambda _spec: (0, 4, 0, 2, 4, 2),
        _align_bias_to_query=lambda bias, _q, **_kwargs: bias,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    z_spec = SimpleNamespace(local_shape=(4, 2, 1), pair_dims=(0, 1))

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        transformer,
        "_zero_foldcp_drain_buffer",
        lambda _tensor: (_ for _ in ()).throw(
            torch.OutOfMemoryError("local diffusion workspace reset OOM")
        ),
    )

    def _all_gather(output, source, **_kwargs):
        gather_calls.append("all_gather")
        output[: source.shape[0]].copy_(source)
        output[source.shape[0] :].copy_(source)

    monkeypatch.setattr(transformer.dist, "all_gather_into_tensor", _all_gather)

    with pytest.raises(RuntimeError, match="local attention failed"):
        transformer.AttentionPairBias.standard_multihead_attention_foldcp_local_z(
            module,
            q=torch.zeros(1, 4, 2),
            kv=torch.zeros(1, 4, 2),
            z_local=torch.zeros(4, 2, 1),
            z_spec=z_spec,
            mesh=mesh,
            projected_bias_local=transformer.FoldCPQueryOwnedAttentionBias(
                torch.zeros(1, 2, 4)
            ),
            workspace=transformer._FoldCPAttentionWorkspace(),
        )

    assert gather_calls == ["all_gather"]


def test_diffusion_failure_flag_clear_oom_drains_gather_before_raising(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    from opendde.model.modules import transformer

    gather_flags = []
    module = SimpleNamespace(
        attention=SimpleNamespace(
            _prep_qkv=lambda **_kwargs: (
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
                torch.zeros(1, 4, 1, 2),
            ),
            use_efficient_implementation=False,
            _wrap_up=lambda raw, _q: raw,
        ),
        _foldcp_valid_ranges=lambda _spec: (0, 4, 0, 2, 4, 2),
        _align_bias_to_query=lambda bias, _q, **_kwargs: bias,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    z_spec = SimpleNamespace(local_shape=(4, 2, 1), pair_dims=(0, 1))
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        transformer,
        "_attention",
        lambda **_kwargs: torch.zeros(1, 2, 1, 2),
    )

    def _all_gather(output, source, **_kwargs):
        gather_flags.append(float(source[-1].reshape(-1)[0]))
        output[: source.shape[0]].copy_(source)
        output[source.shape[0] :].copy_(source)

    monkeypatch.setattr(transformer.dist, "all_gather_into_tensor", _all_gather)

    class FailScalarZero(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            if func is torch.ops.aten.zero_.default and args[0].numel() == 1:
                raise torch.OutOfMemoryError("failure-flag clear OOM")
            return func(*args, **kwargs)

    with (
        FailScalarZero(),
        pytest.raises(RuntimeError, match="local attention failed"),
    ):
        transformer.AttentionPairBias.standard_multihead_attention_foldcp_local_z(
            module,
            q=torch.zeros(1, 4, 2),
            kv=torch.zeros(1, 4, 2),
            z_local=torch.zeros(4, 2, 1),
            z_spec=z_spec,
            mesh=mesh,
            projected_bias_local=transformer.FoldCPQueryOwnedAttentionBias(
                torch.zeros(1, 2, 4)
            ),
            workspace=transformer._FoldCPAttentionWorkspace(),
        )

    assert gather_flags == [1.0]


def test_diffusion_block_tail_failure_is_synchronized(monkeypatch):
    from opendde.model.modules import transformer

    completion_calls = []

    class FailingBlock:
        def forward_foldcp_local_z(self, **_kwargs):
            raise torch.OutOfMemoryError("diffusion final gate OOM")

    module = SimpleNamespace(
        blocks=[FailingBlock()],
        _foldcp_attention_workspace=None,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )

    def _synchronize(action, **kwargs):
        completion_calls.append(kwargs["description"])
        with pytest.raises(torch.OutOfMemoryError, match="final gate OOM"):
            action()
        raise RuntimeError("remote diffusion block completion failed")

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        _synchronize,
    )

    with pytest.raises(RuntimeError, match="remote diffusion block completion failed"):
        transformer.DiffusionTransformer.forward_foldcp_local_z(
            module,
            a=torch.zeros(1, 4, 2),
            s=torch.zeros(1, 4, 2),
            z_local=torch.zeros(4, 2, 1),
            z_spec=SimpleNamespace(),
            mesh=mesh,
        )

    assert completion_calls == ["Fold-CP diffusion block 0 completion"]


def test_remote_diffusion_block_tail_failure_stops_before_next_block(monkeypatch):
    from opendde.model.modules import transformer

    block_calls = []

    class HealthyBlock:
        def __init__(self, block_index):
            self.block_index = block_index

        def forward_foldcp_local_z(self, *, a, s, z_local, **_kwargs):
            block_calls.append(self.block_index)
            return a, s, z_local

    module = SimpleNamespace(
        blocks=[HealthyBlock(0), HealthyBlock(1)],
        _foldcp_attention_workspace=None,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )

    def _synchronize(_action, **kwargs):
        raise RuntimeError(f"{kwargs['description']} failed on a peer")

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        _synchronize,
    )

    with pytest.raises(RuntimeError, match="block 0 completion failed on a peer"):
        transformer.DiffusionTransformer.forward_foldcp_local_z(
            module,
            a=torch.zeros(1, 4, 2),
            s=torch.zeros(1, 4, 2),
            z_local=torch.zeros(4, 2, 1),
            z_spec=SimpleNamespace(),
            mesh=mesh,
        )

    assert block_calls == [0]


def test_one_by_p_atom_window_attention_state_failure_is_synchronized(monkeypatch):
    from opendde.model.modules import transformer

    synchronization = []
    attention_calls = []
    module = SimpleNamespace(
        has_s=False,
        layernorm_a=lambda _value: (_ for _ in ()).throw(
            torch.OutOfMemoryError("atom attention-state OOM")
        ),
        cross_attention_mode=False,
        local_multihead_attention_foldcp_window=lambda **_kwargs: (
            attention_calls.append("attention")
        ),
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )

    def _synchronize(action, **kwargs):
        synchronization.append(kwargs["description"])
        return action()

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        _synchronize,
    )

    with pytest.raises(torch.OutOfMemoryError, match="attention-state OOM"):
        transformer.AttentionPairBias.forward_foldcp_window(
            module,
            a=torch.zeros(1, 4, 2),
            s=torch.zeros(1, 4, 2),
            z_local=torch.zeros(1, 2, 4, 1),
            window_spec=SimpleNamespace(),
            mesh=mesh,
        )

    assert synchronization == ["Fold-CP atom-window attention state preparation"]
    assert attention_calls == []


def test_one_by_p_atom_window_attention_oom_drains_window_ring(monkeypatch):
    from opendde.model.modules import transformer

    synchronization = []
    gather_calls = []
    module = SimpleNamespace(
        attention=SimpleNamespace(
            _prep_qkv=lambda **_kwargs: (
                torch.zeros(4, 1, 2),
                torch.zeros(4, 1, 2),
                torch.zeros(4, 1, 2),
            ),
            num_heads=1,
            c_hidden=2,
            use_efficient_implementation=False,
            _wrap_up=lambda raw, _q: raw,
        ),
        layernorm_z=lambda _value: (_ for _ in ()).throw(
            torch.OutOfMemoryError("atom local attention OOM")
        ),
        linear_nobias_z=lambda value: value,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )
    window_spec = SimpleNamespace(
        n_queries=2,
        n_keys=4,
        n_windows=2,
        block_range=(0, 1),
        q_pad=0,
    )

    def _rearrange(*, q, compute_mask, **_kwargs):
        if compute_mask:
            return (
                q.new_zeros((2, 2, q.shape[-1])),
                None,
                {"mask_trunked": torch.ones(2, 2, 4, dtype=torch.bool)},
            )
        return (
            q.new_zeros((2, 2, 1, 2)),
            [q.new_zeros((2, 4, 1, 2)), q.new_zeros((2, 4, 1, 2))],
            {},
        )

    def _synchronize(action, **kwargs):
        synchronization.append(kwargs["description"])
        return action()

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        _synchronize,
    )
    monkeypatch.setattr(transformer, "rearrange_qk_to_dense_trunk", _rearrange)
    monkeypatch.setattr(
        transformer,
        "gather_window_blocks",
        lambda local, *_args, **_kwargs: gather_calls.append("window-ring") or local,
    )

    with pytest.raises(torch.OutOfMemoryError, match="atom local attention OOM"):
        transformer.AttentionPairBias.local_multihead_attention_foldcp_window(
            module,
            q=torch.zeros(4, 2),
            kv=torch.zeros(4, 2),
            z_local=torch.zeros(1, 2, 4, 1),
            window_spec=window_spec,
            mesh=mesh,
        )

    assert gather_calls == ["window-ring"]
    assert synchronization == [
        "Fold-CP atom-window attention preparation",
        "Fold-CP atom-window attention completion",
    ]


def test_one_by_p_atom_window_block_tail_failure_is_synchronized(monkeypatch):
    from opendde.model.modules import transformer

    completion_calls = []

    class FailingBlock:
        def forward_foldcp_window(self, **_kwargs):
            raise torch.OutOfMemoryError("atom transition OOM")

    module = SimpleNamespace(blocks=[FailingBlock()])
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )

    def _synchronize(action, **kwargs):
        completion_calls.append(kwargs["description"])
        with pytest.raises(torch.OutOfMemoryError, match="atom transition OOM"):
            action()
        raise RuntimeError("remote atom block completion failed")

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        _synchronize,
    )

    with pytest.raises(RuntimeError, match="remote atom block completion failed"):
        transformer.DiffusionTransformer.forward_foldcp_window(
            module,
            a=torch.zeros(1, 4, 2),
            s=torch.zeros(1, 4, 2),
            z_local=torch.zeros(1, 2, 4, 1),
            window_spec=SimpleNamespace(),
            mesh=mesh,
        )

    assert completion_calls == ["Fold-CP atom-window transformer block 0 completion"]


def test_one_by_p_atom_encoder_preparation_failure_stops_before_transformer(
    monkeypatch,
):
    from opendde.model.modules import transformer

    transformer_calls = []
    module = SimpleNamespace(
        has_coords=False,
        n_queries=2,
        n_keys=4,
        _add_atom_single_context_and_mlp_foldcp_local=lambda **_kwargs: (
            _ for _ in ()
        ).throw(torch.OutOfMemoryError("atom encoder input OOM")),
        atom_transformer=SimpleNamespace(
            forward_foldcp_window=lambda **_kwargs: transformer_calls.append("run")
        ),
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )
    window_spec = SimpleNamespace(block_range=(0, 1), n_windows=2)

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        transformer,
        "rearrange_qk_to_dense_trunk",
        lambda **_kwargs: (
            torch.zeros(2, 2, 1),
            torch.zeros(2, 4, 1),
            {},
        ),
    )

    with pytest.raises(torch.OutOfMemoryError, match="atom encoder input OOM"):
        transformer.AtomAttentionEncoder.forward_foldcp_window(
            module,
            atom_to_token_idx=torch.zeros(4, dtype=torch.long),
            ref_pos=torch.zeros(4, 3),
            ref_charge=torch.zeros(4),
            ref_mask=torch.ones(4),
            ref_atom_name_chars=torch.zeros(4, 4, 64),
            ref_element=torch.zeros(4, 128),
            d_lm=torch.zeros(1),
            v_lm=torch.zeros(1),
            pad_info={},
            mesh=mesh,
            p_lm=torch.zeros(1, 2, 4, 1),
            c_l=torch.zeros(4, 1),
            window_spec=window_spec,
        )

    assert transformer_calls == []


def test_atom_encoder_releases_completed_context_launch_chunks():
    from opendde.model.modules import transformer

    prior_refs = []
    calls = 0

    def add_context(*, p_lm, c_l_q, c_l_k, **_kwargs):
        nonlocal calls, prior_refs
        if calls:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        updated = p_lm + 1
        prior_refs = [
            weakref.ref(p_lm),
            weakref.ref(c_l_q),
            weakref.ref(c_l_k),
            weakref.ref(updated),
        ]
        calls += 1
        return updated

    module = SimpleNamespace(
        _add_atom_single_context_and_mlp=add_context,
    )
    result = (
        transformer.AtomAttentionEncoder._add_atom_single_context_and_mlp_foldcp_local(
            module,
            p_lm=torch.zeros(128, 1, 1, 1),
            c_l_q=torch.zeros(128, 1, 1),
            c_l_k=torch.zeros(128, 1, 1),
            block_start=0,
            n_windows=128,
        )
    )

    assert result.shape == (128, 1, 1, 1)
    assert calls == 2


def test_one_by_p_atom_decoder_tail_failure_is_synchronized(monkeypatch):
    from opendde.model.modules import transformer

    synchronization = []
    module = SimpleNamespace(
        linear_no_bias_a=lambda value: value,
        atom_transformer=SimpleNamespace(
            forward_foldcp_window=lambda **kwargs: kwargs["q"]
        ),
        layernorm_q=lambda _value: (_ for _ in ()).throw(
            torch.OutOfMemoryError("atom decoder tail OOM")
        ),
        linear_no_bias_out=lambda value: value,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )

    def _synchronize(action, **kwargs):
        synchronization.append(kwargs["description"])
        return action()

    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        _synchronize,
    )
    monkeypatch.setattr(
        transformer,
        "broadcast_token_to_atom",
        lambda x_token, atom_to_token_idx: x_token[atom_to_token_idx],
    )

    with pytest.raises(torch.OutOfMemoryError, match="atom decoder tail OOM"):
        transformer.AtomAttentionDecoder.forward_foldcp_window(
            module,
            atom_to_token_idx=torch.tensor([0, 1]),
            a=torch.zeros(2, 2),
            q_skip=torch.zeros(2, 2),
            c_skip=torch.zeros(2, 2),
            p_skip_local=torch.zeros(1, 2, 4, 1),
            window_spec=SimpleNamespace(),
            mesh=mesh,
        )

    assert synchronization == [
        "Fold-CP atom decoder transformer-input preparation",
        "Fold-CP atom decoder completion",
    ]


def test_diffusion_attention_workspace_reuses_collective_buffers(monkeypatch):
    from opendde.model.modules import transformer

    allocations = []

    def run(action, **kwargs):
        allocations.append(kwargs["description"])
        return action()

    monkeypatch.setattr(transformer, "run_group_rank_action_synchronized", run)
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 3)),
    )
    workspace = transformer._FoldCPAttentionWorkspace()
    q_proj = torch.zeros(2, 4, 5, 3)
    bias_tile = torch.zeros(1, 4, 5, 2)

    output_first = workspace.output_buffers(q_proj, n_token=5, mesh=mesh)
    output_second = workspace.output_buffers(q_proj, n_token=5, mesh=mesh)
    bias_first = workspace.bias_buffers(bias_tile, n_token=5, mesh=mesh)
    bias_second = workspace.bias_buffers(bias_tile, n_token=5, mesh=mesh)

    assert [id(tensor) for tensor in output_first] == [
        id(tensor) for tensor in output_second
    ]
    assert [id(tensor) for tensor in bias_first] == [
        id(tensor) for tensor in bias_second
    ]
    # One extra row carries a rank-local failure flag in the existing gather.
    assert output_first[0].shape == (3, 2, 4, 3)
    assert output_first[1].shape == (9, 2, 4, 3)
    assert output_first[2].shape == (2, 5, 4, 3)
    # One extra row carries a source-local preparation failure flag in the
    # existing all-to-all, avoiding a hot-loop status collective.
    assert bias_first[0].shape == (3, 1, 4, 3, 2)
    assert bias_first[2].shape == (1, 4, 2, 5)
    assert allocations == [
        "Fold-CP diffusion output-workspace allocation",
        "Fold-CP diffusion bias-workspace allocation",
    ]


def test_diffusion_output_rows_use_safe_column_ring(monkeypatch):
    from opendde.model.modules import transformer

    calls = []

    def gather(local_tensor, **kwargs):
        calls.append(kwargs)
        shape = list(local_tensor.shape)
        shape[kwargs["dim"]] = kwargs["length"]
        return local_tensor.new_zeros(shape)

    monkeypatch.setattr(transformer, "gather_tensor_by_ring", gather)
    mesh = SimpleNamespace(
        group_col=object(),
        coord=(0, 1),
        layout=SimpleNamespace(shape=(2, 2)),
        ring_comm=lambda: SimpleNamespace(comm_col=object()),
    )
    result = transformer.AttentionPairBias._foldcp_gather_rows_by_col_ring(
        torch.zeros(1, 2, 3),
        n_token=3,
        mesh=mesh,
        row_dim=-2,
    )

    assert result.shape == (1, 3, 3)
    assert len(calls) == 1
    assert calls[0]["description"] == "diffusion attention output-row ring"
    assert calls[0]["group"] is mesh.group_col


def test_atom_window_stream_finishes_p2p_before_reporting_compute_error(monkeypatch):
    from opendde.model.modules import transformer

    sends = []
    layout = SimpleNamespace(
        numel=2,
        to_coord=lambda rank: (0, rank),
    )
    mesh = SimpleNamespace(group_2d=object(), layout=layout)
    z_spec = SimpleNamespace(
        local_shape=(4, 2, 1),
        original_shape=(4, 4, 1),
        pair_dims=(0, 1),
    )
    module = SimpleNamespace(
        n_queries=1,
        n_keys=1,
        c_atompair=1,
        layernorm_z=lambda _tensor: (_ for _ in ()).throw(
            RuntimeError("local atom-window projection OOM")
        ),
        linear_no_bias_z=lambda tensor: tensor,
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **kwargs: action() if action is not None else None,
    )
    monkeypatch.setattr(transformer.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(transformer.dist, "get_global_rank", lambda _group, rank: rank)

    def all_gather(outputs, source, *, group):
        for output in outputs:
            output.copy_(source)

    monkeypatch.setattr(transformer.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        transformer.dist,
        "send",
        lambda tensor, **kwargs: sends.append(tensor.detach().clone()),
    )
    monkeypatch.setattr(
        transformer.dist,
        "recv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rank 0 should not receive in this fixture")
        ),
    )

    with pytest.raises(RuntimeError, match="local atom-window projection OOM"):
        transformer.AtomAttentionEncoder._project_pair_embedding_in_dense_trunk_from_foldcp_local(
            module,
            z_local=torch.arange(8.0).reshape(4, 2, 1),
            z_spec=z_spec,
            idx_q=torch.zeros(1, 1, dtype=torch.long),
            idx_k=torch.zeros(1, 1, dtype=torch.long),
            mesh=mesh,
            out=torch.zeros(1, 1, 1, 1),
            block_start=0,
            n_windows=2,
            window_chunk_size=2,
        )

    assert len(sends) == 1


def test_atom_window_stream_drains_p2p_after_sender_index_failure(monkeypatch):
    from opendde.model.modules import transformer

    sends = []
    layout = SimpleNamespace(
        numel=2,
        to_coord=lambda rank: (0, rank),
    )
    mesh = SimpleNamespace(group_2d=object(), layout=layout)
    z_spec = SimpleNamespace(
        local_shape=(4, 2, 1),
        original_shape=(4, 4, 1),
        pair_dims=(0, 1),
    )
    module = SimpleNamespace(
        n_queries=1,
        n_keys=1,
        c_atompair=1,
        layernorm_z=lambda tensor: tensor,
        linear_no_bias_z=lambda tensor: tensor,
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **kwargs: action() if action is not None else None,
    )
    monkeypatch.setattr(transformer.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(transformer.dist, "get_global_rank", lambda _group, rank: rank)

    def all_gather(outputs, source, *, group):
        for output in outputs:
            output.copy_(source)

    monkeypatch.setattr(transformer.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        transformer.dist,
        "send",
        lambda tensor, **kwargs: sends.append(tensor.detach().clone()),
    )
    monkeypatch.setattr(
        transformer.dist,
        "recv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rank 0 should not receive in this fixture")
        ),
    )

    with pytest.raises(IndexError, match="out of bounds"):
        transformer.AtomAttentionEncoder._project_pair_embedding_in_dense_trunk_from_foldcp_local(
            module,
            # The gathered metadata still schedules rank 0 as the sender, but
            # its malformed local tile makes the sender-side index operation
            # fail. The remaining scheduled transfer must still be drained.
            z_local=torch.empty(0, 2, 1),
            z_spec=z_spec,
            idx_q=torch.zeros(1, 1, dtype=torch.long),
            idx_k=torch.zeros(1, 1, dtype=torch.long),
            mesh=mesh,
            out=torch.zeros(1, 1, 1, 1),
            block_start=0,
            n_windows=2,
            window_chunk_size=2,
        )

    assert len(sends) == 1


def test_atom_window_stream_drains_p2p_after_buffer_reset_failure(monkeypatch):
    from opendde.model.modules import transformer

    sends = []
    layout = SimpleNamespace(
        numel=2,
        to_coord=lambda rank: (0, rank),
    )
    mesh = SimpleNamespace(group_2d=object(), layout=layout)
    z_spec = SimpleNamespace(
        local_shape=(4, 2, 1),
        original_shape=(4, 4, 1),
        pair_dims=(0, 1),
    )
    module = SimpleNamespace(
        n_queries=1,
        n_keys=1,
        c_atompair=1,
        layernorm_z=lambda tensor: tensor,
        linear_no_bias_z=lambda tensor: tensor,
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **kwargs: action() if action is not None else None,
    )
    monkeypatch.setattr(transformer.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(transformer.dist, "get_global_rank", lambda _group, rank: rank)

    def all_gather(outputs, source, *, group):
        for output in outputs:
            output.copy_(source)

    monkeypatch.setattr(transformer.dist, "all_gather", all_gather)
    monkeypatch.setattr(
        transformer.dist,
        "send",
        lambda tensor, **kwargs: sends.append(tensor.detach().clone()),
    )
    monkeypatch.setattr(
        transformer.dist,
        "recv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("rank 0 should not receive in this fixture")
        ),
    )
    reset_calls = []

    def fail_first_reset(tensor):
        reset_calls.append(tuple(tensor.shape))
        if len(reset_calls) == 1:
            raise torch.OutOfMemoryError("local atom-window reset OOM")
        tensor.zero_()

    monkeypatch.setattr(transformer, "_zero_foldcp_drain_buffer", fail_first_reset)

    with pytest.raises(torch.OutOfMemoryError, match="atom-window reset OOM"):
        transformer.AtomAttentionEncoder._project_pair_embedding_in_dense_trunk_from_foldcp_local(
            module,
            z_local=torch.arange(8.0).reshape(4, 2, 1),
            z_spec=z_spec,
            idx_q=torch.zeros(1, 1, dtype=torch.long),
            idx_k=torch.zeros(1, 1, dtype=torch.long),
            mesh=mesh,
            out=torch.zeros(1, 1, 1, 1),
            block_start=0,
            n_windows=2,
            window_chunk_size=2,
        )

    assert reset_calls
    assert len(sends) == 1


def test_diffusion_bias_cache_stops_before_gather_after_remote_oom(monkeypatch):
    from opendde.model.modules import transformer

    collectives = []
    attention_pair_bias = SimpleNamespace(
        n_heads=1,
        _foldcp_valid_ranges=lambda _spec: (0, 4, 0, 2, 4, 2),
    )
    module = SimpleNamespace(
        blocks=(SimpleNamespace(attention_pair_bias=attention_pair_bias),)
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 2),
        padded_shape=(4, 4, 2),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    monkeypatch.setattr(
        transformer,
        "foldcp_diffusion_bias_cache_is_safe",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote diffusion cache allocation OOM")
        ),
    )
    monkeypatch.setattr(
        transformer.dist,
        "all_to_all_single",
        lambda *args, **kwargs: collectives.append("all_to_all"),
    )

    with pytest.raises(RuntimeError, match="remote diffusion cache allocation OOM"):
        transformer.DiffusionTransformer.prepare_foldcp_attention_bias_cache(
            module,
            torch.zeros(4, 2, 2),
            spec,
            mesh,
        )

    assert collectives == []


def test_diffusion_bias_cache_padding_oom_occurs_inside_source_boundary(monkeypatch):
    from opendde.model.modules import transformer

    collectives = []
    attention_pair_bias = SimpleNamespace(
        n_heads=1,
        _foldcp_valid_ranges=lambda _spec: (0, 5, 0, 2, 5, 2),
    )
    module = SimpleNamespace(
        blocks=(SimpleNamespace(attention_pair_bias=attention_pair_bias),)
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 3)),
        coord=(0, 0),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(5, 5, 2),
        padded_shape=(5, 6, 2),
        pair_dims=(0, 1),
        row_range=(0, 5),
        col_range=(0, 2),
        mesh_shape=(1, 3),
        mesh_coord=(0, 0),
    )
    monkeypatch.setattr(
        transformer,
        "foldcp_diffusion_bias_cache_is_safe",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        transformer,
        "_prepare_foldcp_diffusion_bias_cache_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("padding allocation OOM")
        ),
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        transformer.dist,
        "all_to_all_single",
        lambda *args, **kwargs: collectives.append("all_to_all"),
    )

    with pytest.raises(torch.OutOfMemoryError, match="padding allocation OOM"):
        transformer.DiffusionTransformer.prepare_foldcp_attention_bias_cache(
            module,
            torch.zeros(5, 2, 2),
            spec,
            mesh,
        )

    assert collectives == []


def test_diffusion_bias_cache_releases_result_containers_before_projection(
    monkeypatch,
):
    from opendde.model.modules import transformer

    class Owners:
        def __init__(self, *values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

    attention_pair_bias = SimpleNamespace(
        n_heads=1,
        _foldcp_valid_ranges=lambda _spec: (0, 4, 0, 2, 4, 2),
        _project_attention_bias=lambda **kwargs: torch.zeros(1, 4, 2),
    )
    module = SimpleNamespace(
        blocks=(SimpleNamespace(attention_pair_bias=attention_pair_bias),)
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 2),
        padded_shape=(4, 4, 2),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    owner_refs = []
    calls = 0

    def synchronized(action, **_kwargs):
        nonlocal calls
        calls += 1
        gc.collect()
        if calls >= 2:
            assert owner_refs[0]() is None
        if calls >= 3:
            assert owner_refs[1]() is None
        result = action()
        if calls in (1, 2):
            owner = Owners(*result)
            owner_refs.append(weakref.ref(owner))
            return owner
        return result

    monkeypatch.setattr(
        transformer,
        "foldcp_diffusion_bias_cache_is_safe",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        synchronized,
    )
    monkeypatch.setattr(
        transformer.dist,
        "all_to_all_single",
        lambda output, source, **_kwargs: output.copy_(source),
    )

    result = transformer.DiffusionTransformer.prepare_foldcp_attention_bias_cache(
        module,
        torch.zeros(4, 2, 2),
        spec,
        mesh,
    )

    assert len(result) == 1
    assert tuple(result[0].tensor.shape) == (1, 2, 4)
    assert calls == 3


def test_diffusion_pair_cache_failure_is_synchronized_before_atom_window(monkeypatch):
    from opendde.model.modules import diffusion

    local_preparations = []
    module = SimpleNamespace(
        _prepare_cache_foldcp_local_impl=lambda *args, **kwargs: (
            local_preparations.append("prepare")
        )
    )
    mesh = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        diffusion,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote diffusion pair-cache OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote diffusion pair-cache OOM"):
        diffusion.DiffusionConditioning.prepare_cache_foldcp_local(
            module,
            relp_feature=torch.zeros(1, 1, 1),
            z_trunk_local=torch.zeros(1, 1, 1),
            z_spec=SimpleNamespace(),
            mesh=mesh,
        )

    assert local_preparations == []


@pytest.mark.parametrize(
    ("failure_stage", "expected_calls"),
    [
        ("Fold-CP diffusion conditioning completion", []),
        ("Fold-CP token-transformer input preparation", ["atom-encoder"]),
        (
            "Fold-CP atom-decoder input normalization",
            ["atom-encoder", "token-transformer"],
        ),
    ],
)
def test_one_by_p_diffusion_stage_failure_stops_before_next_collective(
    monkeypatch,
    failure_stage,
    expected_calls,
):
    from opendde.model.modules import diffusion

    calls = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )
    s_single = torch.zeros(1, 2, 2)
    pair_z = torch.zeros(2, 1, 2)
    a_token = torch.zeros(1, 2, 2)
    window_spec = SimpleNamespace()

    def _atom_encoder(*_args, **_kwargs):
        calls.append("atom-encoder")
        return (
            a_token,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            window_spec,
        )

    def _token_transformer(**kwargs):
        calls.append("token-transformer")
        return kwargs["a"]

    def _atom_decoder(**_kwargs):
        calls.append("atom-decoder")
        return torch.zeros(1, 4, 3)

    module = SimpleNamespace(
        blocks_per_ckpt=None,
        _maybe_foldcp_mesh=lambda: mesh,
        diffusion_conditioning=lambda *_args, **_kwargs: (s_single, pair_z),
        atom_attention_encoder=SimpleNamespace(
            _warmup_foldcp_atom_window_p2p=lambda **_kwargs: None,
            forward_foldcp_window=_atom_encoder,
        ),
        linear_no_bias_s=lambda value: value,
        layernorm_s=lambda value: value,
        normalize=lambda value: value,
        diffusion_transformer=SimpleNamespace(
            forward_foldcp_local_z=_token_transformer
        ),
        layernorm_a=lambda value: value,
        atom_attention_decoder=SimpleNamespace(
            forward_foldcp_window=_atom_decoder,
        ),
    )

    def _synchronize(action, **kwargs):
        if kwargs["description"] == failure_stage:
            raise RuntimeError(f"remote {failure_stage} OOM")
        return action()

    monkeypatch.setattr(
        diffusion,
        "run_group_rank_action_synchronized",
        _synchronize,
    )
    features = {
        "relp": torch.zeros(1),
        "atom_to_token_idx": torch.zeros(4, dtype=torch.long),
        "ref_pos": torch.zeros(4, 3),
        "ref_charge": torch.zeros(4),
        "ref_mask": torch.ones(4),
        "ref_atom_name_chars": torch.zeros(4, 4, 64),
        "ref_element": torch.zeros(4, 128),
        "d_lm": torch.zeros(1),
        "v_lm": torch.zeros(1),
        "pad_info": {},
    }

    with pytest.raises(RuntimeError, match=f"remote {failure_stage} OOM"):
        diffusion.DiffusionModule.f_forward(
            module,
            x_noisy=torch.zeros(1, 4, 3),
            r_noisy=torch.zeros(1, 4, 3),
            t_hat_noise_level=torch.zeros(1),
            input_feature_dict=features,
            s_inputs=torch.zeros(2, 2),
            s_trunk=torch.zeros(2, 2),
            z_trunk=torch.zeros(2, 2, 2),
            pair_z=pair_z,
            p_lm=torch.zeros(1),
            c_l=torch.zeros(1),
            pair_z_spec=SimpleNamespace(),
            atom_window_spec=window_spec,
        )

    assert calls == expected_calls


def test_one_by_p_diffusion_releases_fp32_pair_before_atom_decoder(monkeypatch):
    from opendde.model.modules import diffusion

    pair_references = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
    )
    s_single = torch.zeros(1, 2, 2)
    pair_z = torch.zeros(2, 1, 2, dtype=torch.bfloat16)
    a_token = torch.zeros(1, 2, 2)
    window_spec = SimpleNamespace()

    def _token_transformer(**kwargs):
        pair_references.append(weakref.ref(kwargs["z_local"]))
        return kwargs["a"]

    def _atom_decoder(**_kwargs):
        gc.collect()
        assert pair_references[0]() is None
        return torch.zeros(1, 4, 3)

    module = SimpleNamespace(
        blocks_per_ckpt=None,
        _maybe_foldcp_mesh=lambda: mesh,
        diffusion_conditioning=lambda *_args, **_kwargs: (s_single, pair_z),
        atom_attention_encoder=SimpleNamespace(
            _warmup_foldcp_atom_window_p2p=lambda **_kwargs: None,
            forward_foldcp_window=lambda *_args, **_kwargs: (
                a_token,
                torch.zeros(1),
                torch.zeros(1),
                torch.zeros(1),
                window_spec,
            ),
        ),
        linear_no_bias_s=lambda value: value,
        layernorm_s=lambda value: value,
        normalize=lambda value: value,
        diffusion_transformer=SimpleNamespace(
            forward_foldcp_local_z=_token_transformer
        ),
        layernorm_a=lambda value: value,
        atom_attention_decoder=SimpleNamespace(
            forward_foldcp_window=_atom_decoder,
        ),
    )
    monkeypatch.setattr(
        diffusion,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    features = {
        "relp": torch.zeros(1),
        "atom_to_token_idx": torch.zeros(4, dtype=torch.long),
        "ref_pos": torch.zeros(4, 3),
        "ref_charge": torch.zeros(4),
        "ref_mask": torch.ones(4),
        "ref_atom_name_chars": torch.zeros(4, 4, 64),
        "ref_element": torch.zeros(4, 128),
        "d_lm": torch.zeros(1),
        "v_lm": torch.zeros(1),
        "pad_info": {},
    }

    diffusion.DiffusionModule.f_forward(
        module,
        x_noisy=torch.zeros(1, 4, 3),
        r_noisy=torch.zeros(1, 4, 3),
        t_hat_noise_level=torch.zeros(1),
        input_feature_dict=features,
        s_inputs=torch.zeros(2, 2),
        s_trunk=torch.zeros(2, 2),
        z_trunk=torch.zeros(2, 2, 2),
        pair_z=pair_z,
        p_lm=torch.zeros(1),
        c_l=torch.zeros(1),
        pair_z_spec=SimpleNamespace(),
        atom_window_spec=window_spec,
    )

    assert len(pair_references) == 1


def test_diffusion_bias_source_failure_stops_before_bias_cache_collective(monkeypatch):
    from opendde.model import opendde

    bias_cache_calls = []
    tensor = torch.zeros(1, 1, 1)
    module = SimpleNamespace(
        enable_diffusion_shared_vars_cache=True,
        enable_efficient_fusion=True,
        configs=SimpleNamespace(skip_amp=SimpleNamespace(sample_diffusion=False)),
        diffusion_module=SimpleNamespace(
            diffusion_conditioning=SimpleNamespace(
                prepare_cache_foldcp_local=lambda *args, **kwargs: (
                    tensor,
                    SimpleNamespace(),
                )
            ),
            atom_attention_encoder=SimpleNamespace(
                prepare_cache_foldcp_window=lambda *args, **kwargs: (
                    tensor,
                    tensor,
                    SimpleNamespace(),
                )
            ),
            normalize=lambda value: value,
            diffusion_transformer=SimpleNamespace(
                prepare_foldcp_attention_bias_cache=lambda *args, **kwargs: (
                    bias_cache_calls.append("bias-cache")
                )
            ),
        ),
    )
    mesh = SimpleNamespace(group_2d=object())
    input_features = {
        key: tensor
        for key in (
            "relp",
            "ref_pos",
            "ref_charge",
            "ref_mask",
            "ref_element",
            "ref_atom_name_chars",
            "atom_to_token_idx",
            "d_lm",
            "v_lm",
        )
    }
    input_features["pad_info"] = {}
    monkeypatch.setattr(
        opendde,
        "autocasting_disable_decorator",
        lambda _skip_amp: lambda action: action,
    )
    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote diffusion bias-source OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote diffusion bias-source OOM"):
        opendde.OpenDDE.prepare_diffusion_cache_for_sampling(
            module,
            input_feature_dict=input_features,
            z=tensor,
            foldcp_mesh=mesh,
            diffusion_z_spec=SimpleNamespace(),
        )

    assert bias_cache_calls == []


def test_structural_parent_failure_stops_before_expander_collectives(monkeypatch):
    from opendde.model import opendde

    expander_calls = []
    mesh = SimpleNamespace(group_2d=object())
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        structural_token_expander=SimpleNamespace(
            forward_foldcp_local_pair=lambda *args, **kwargs: expander_calls.append(
                "expand"
            )
        ),
    )
    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote structural parent OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote structural parent OOM"):
        opendde.OpenDDE._maybe_expand_to_structural_tokens_foldcp_local(
            module,
            input_feature_dict={"parent_residue_idx": torch.tensor([0])},
            structural_feature_dict={},
            s_inputs=torch.zeros(1, 1),
            s=torch.zeros(1, 1),
            z=torch.zeros(1, 1, 1),
        )

    assert expander_calls == []


def test_structural_feature_failure_stops_before_refiner_collectives(monkeypatch):
    from opendde.model import opendde

    tensor = torch.zeros(1, 1)
    spec = SimpleNamespace()
    refiner_calls = []
    mesh = SimpleNamespace(group_2d=object())
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        structural_token_expander=SimpleNamespace(
            forward_foldcp_local_pair=lambda *args, **kwargs: (
                tensor,
                tensor,
                tensor,
                {"structural_pair_attn_bias": tensor},
                {"structural_pair_attn_bias": spec},
                spec,
            )
        ),
        enable_structural_token_refiner=True,
        structural_token_refiner=lambda *args, **kwargs: refiner_calls.append("refine"),
    )
    sync_calls = []

    def synchronize(action, **kwargs):
        sync_calls.append(kwargs["description"])
        if len(sync_calls) == 1:
            return action()
        raise RuntimeError("remote structural feature OOM")

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="remote structural feature OOM"):
        opendde.OpenDDE._maybe_expand_to_structural_tokens_foldcp_local(
            module,
            input_feature_dict={"parent_residue_idx": torch.tensor([0])},
            structural_feature_dict={},
            s_inputs=tensor,
            s=tensor,
            z=tensor,
        )

    assert sync_calls == [
        "Fold-CP structural parent-index preparation",
        "Fold-CP structural feature preparation",
    ]
    assert refiner_calls == []


def test_structural_finalization_failure_is_group_synchronized(monkeypatch):
    from opendde.model import opendde

    tensor = torch.zeros(1, 1)
    spec = SimpleNamespace()
    mesh = SimpleNamespace(group_2d=object())
    dropped = []
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        structural_token_expander=SimpleNamespace(
            forward_foldcp_local_pair=lambda *args, **kwargs: (
                tensor,
                tensor,
                tensor,
                {"structural_pair_attn_bias": tensor},
                {"structural_pair_attn_bias": spec},
                spec,
            )
        ),
        relative_position_encoding=SimpleNamespace(
            generate_relp=lambda features, lazy: features
        ),
        enable_structural_token_refiner=False,
        drop_residue_only_features_for_structural_branch=lambda features: (
            dropped.append(features)
        ),
    )
    sync_calls = []

    def synchronize(action, **kwargs):
        sync_calls.append(kwargs["description"])
        if kwargs["description"] == "Fold-CP structural branch finalization":
            raise RuntimeError("remote structural finalization OOM")
        return action()

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    input_features = {
        key: torch.tensor([0])
        for key in (
            "parent_residue_idx",
            "structural_token_index",
            "atom_to_structural_token_idx",
            "atom_to_structural_tokatom_idx",
            "asym_id",
            "residue_index",
            "entity_id",
            "sym_id",
            "structural_has_frame",
            "structural_frame_atom_index",
        )
    }

    with pytest.raises(RuntimeError, match="remote structural finalization OOM"):
        opendde.OpenDDE._maybe_expand_to_structural_tokens_foldcp_local(
            module,
            input_feature_dict=input_features,
            structural_feature_dict={},
            s_inputs=tensor,
            s=tensor,
            z=tensor,
        )

    assert sync_calls[-1] == "Fold-CP structural branch finalization"
    assert dropped == []


def test_forward_input_preparation_failure_stops_before_model_collectives():
    from opendde.model import opendde

    model_calls = []
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: SimpleNamespace(group_2d=object()),
        relative_position_encoding=SimpleNamespace(
            generate_relp=lambda features, lazy: features
        ),
        _run_foldcp_local_action_synchronized=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(RuntimeError("remote input preparation OOM")),
        main_inference_loop=lambda **_kwargs: model_calls.append("model"),
        N_cycle=1,
        N_model_seed=1,
        configs=SimpleNamespace(infer_setting=SimpleNamespace(chunk_size=4)),
    )

    with pytest.raises(RuntimeError, match="remote input preparation OOM"):
        opendde.OpenDDE.forward(module, input_feature_dict={})

    assert model_calls == []


def test_post_trunk_cleanup_failure_stops_before_structural_collectives():
    from opendde.model import opendde

    structural_calls = []
    fake_cuda_tensor = SimpleNamespace(is_cuda=True)
    module = SimpleNamespace(
        configs=SimpleNamespace(
            infer_setting=SimpleNamespace(dynamic_chunk_size=False),
        ),
        _bound_pairformer_chunk_size=lambda _n_token, chunk_size: chunk_size,
        _foldcp_stage_context=lambda *_args: nullcontext(),
        get_pairformer_output=lambda **_kwargs: (
            fake_cuda_tensor,
            fake_cuda_tensor,
            fake_cuda_tensor,
        ),
        _maybe_foldcp_mesh=lambda: SimpleNamespace(group_2d=object()),
        _run_foldcp_local_action_synchronized=lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(RuntimeError("remote post-trunk cleanup failed")),
        enable_structural_token_expansion=False,
        expand_to_structural_tokens=lambda **_kwargs: structural_calls.append(
            "structural"
        ),
    )

    with pytest.raises(RuntimeError, match="remote post-trunk cleanup failed"):
        opendde.OpenDDE._main_inference_loop(
            module,
            input_feature_dict={"residue_index": torch.zeros(1)},
            N_cycle=1,
        )

    assert structural_calls == []


def test_structural_expander_preparation_failure_stops_before_pair_gather(monkeypatch):
    from opendde.model.modules import structural_tokens

    gather_calls = []
    module = SimpleNamespace(
        _gather_parent_pair_tile_from_foldcp_local=lambda *args, **kwargs: (
            gather_calls.append("gather")
        )
    )
    monkeypatch.setattr(
        structural_tokens,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote structural expander OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote structural expander OOM"):
        structural_tokens.StructuralTokenExpander.forward_foldcp_local_pair(
            module,
            input_feature_dict={},
            s_inputs_res=torch.zeros(1, 1),
            s_res=torch.zeros(1, 1),
            z_res=torch.zeros(1, 1, 1),
            mesh=SimpleNamespace(group_2d=object()),
            z_res_spec=SimpleNamespace(),
        )

    assert gather_calls == []


def test_structural_source_chunk_failure_stops_before_next_gather(monkeypatch):
    from opendde.model.modules import structural_tokens

    gather_calls = []
    mesh = SimpleNamespace(group_2d=object())
    pair_spec = SimpleNamespace(
        original_shape=(2, 2, 1),
        local_shape=(2, 2, 1),
        row_range=(0, 2),
        col_range=(0, 2),
    )
    bias_spec = SimpleNamespace(
        original_shape=(2, 2),
        local_shape=(2, 2),
        row_range=(0, 2),
        col_range=(0, 2),
    )
    specs = iter((pair_spec, bias_spec))
    monkeypatch.setattr(
        structural_tokens,
        "make_pair_shard_spec",
        lambda *args, **kwargs: next(specs),
    )

    def synchronize(action, **kwargs):
        if kwargs["description"] == "Fold-CP structural source-chunk completion":
            raise RuntimeError("remote structural source projection OOM")
        return action()

    monkeypatch.setattr(
        structural_tokens,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    module = SimpleNamespace(
        c_z=1,
        pair_chunk_size=1,
        _gather_parent_single=lambda value, parent: value.index_select(-2, parent),
        single_input_role_embedding=lambda role: torch.zeros(role.shape[0], 1),
        single_split_mlp=lambda value: torch.zeros_like(value),
        single_role_embedding=lambda role: torch.zeros(role.shape[0], 1),
        _build_structural_pair_context=lambda **kwargs: {},
        _use_foldcp_full_projection_source_order=lambda _z: True,
        _build_structural_pair_features_for_rows=lambda **kwargs: {},
        _gather_parent_pair_tile_from_foldcp_local=lambda **kwargs: (
            gather_calls.append("gather") or torch.zeros(1, 2, 1)
        ),
    )

    with pytest.raises(RuntimeError, match="remote structural source projection OOM"):
        structural_tokens.StructuralTokenExpander.forward_foldcp_local_pair(
            module,
            input_feature_dict={
                "parent_residue_idx": torch.tensor([0, 1]),
                "subtoken_role_id": torch.tensor([0, 0]),
            },
            s_inputs_res=torch.zeros(2, 1),
            s_res=torch.zeros(2, 1),
            z_res=torch.zeros(2, 2, 1),
            mesh=mesh,
            z_res_spec=SimpleNamespace(original_shape=(2, 2, 1)),
        )

    assert gather_calls == ["gather"]


def test_structural_expander_releases_chunk_owners_before_next_chunk(monkeypatch):
    from opendde.model.modules import structural_tokens

    mesh = SimpleNamespace(group_2d=object())
    pair_spec = SimpleNamespace(
        original_shape=(2, 2, 1),
        local_shape=(2, 2, 1),
        row_range=(0, 2),
        col_range=(0, 2),
    )
    bias_spec = SimpleNamespace(
        original_shape=(2, 2),
        local_shape=(2, 2),
        row_range=(0, 2),
        col_range=(0, 2),
    )
    specs = iter((pair_spec, bias_spec))
    monkeypatch.setattr(
        structural_tokens,
        "make_pair_shard_spec",
        lambda *args, **kwargs: next(specs),
    )

    prior_refs = []
    preparation_count = 0

    def synchronize(action, *, description, **_kwargs):
        nonlocal preparation_count
        if description == "Fold-CP structural source-chunk preparation":
            if preparation_count:
                gc.collect()
                assert all(reference() is None for reference in prior_refs)
            preparation_count += 1
        return action()

    monkeypatch.setattr(
        structural_tokens,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    def build_pair_features(**_kwargs):
        owner = torch.zeros(1)
        prior_refs.append(weakref.ref(owner))
        return {"owner": owner}

    def gather_pair_tile(**_kwargs):
        owner = torch.zeros(1, 2, 1)
        prior_refs.append(weakref.ref(owner))
        return owner

    module = SimpleNamespace(
        c_z=1,
        pair_chunk_size=1,
        _gather_parent_single=lambda value, parent: value.index_select(-2, parent),
        single_input_role_embedding=lambda role: torch.zeros(role.shape[0], 1),
        single_split_mlp=lambda value: torch.zeros_like(value),
        single_role_embedding=lambda role: torch.zeros(role.shape[0], 1),
        _build_structural_pair_context=lambda **_kwargs: {},
        _use_foldcp_full_projection_source_order=lambda _z: True,
        _build_structural_pair_features_for_rows=build_pair_features,
        _gather_parent_pair_tile_from_foldcp_local=gather_pair_tile,
        _pair_project_by_role=lambda **_kwargs: None,
        _make_pair_init_bias=lambda *_args, **_kwargs: torch.zeros(1, 2, 1),
        _make_attention_bias=lambda *_args, **_kwargs: torch.zeros(1, 2),
        _reshape_pair_term_for_target=lambda value, _target: value,
    )

    result = structural_tokens.StructuralTokenExpander.forward_foldcp_local_pair(
        module,
        input_feature_dict={
            "parent_residue_idx": torch.tensor([0, 1]),
            "subtoken_role_id": torch.tensor([0, 0]),
        },
        s_inputs_res=torch.zeros(2, 1),
        s_res=torch.zeros(2, 1),
        z_res=torch.zeros(2, 2, 1),
        mesh=mesh,
        z_res_spec=SimpleNamespace(original_shape=(2, 2, 1)),
    )

    assert result[2].shape == (2, 2, 1)
    assert preparation_count == 2


def test_model_local_boundary_action_uses_foldcp_failure_handshake(monkeypatch):
    from opendde.model import opendde

    actions = []
    mesh = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote noise-schedule OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote noise-schedule OOM"):
        opendde.OpenDDE._run_foldcp_local_action_synchronized(
            mesh,
            lambda: actions.append("distributed"),
            description="noise schedule",
        )

    assert actions == []
    result = opendde.OpenDDE._run_foldcp_local_action_synchronized(
        None,
        lambda: "single-path",
        description="single GPU",
    )
    assert result == "single-path"


def test_diffusion_rollout_input_failure_stops_before_sampling():
    from opendde.model import opendde

    sample_calls = []
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: SimpleNamespace(group_2d=object()),
        _run_foldcp_local_action_synchronized=lambda *args, **kwargs: (
            _ for _ in ()
        ).throw(RuntimeError("remote rollout seed copy OOM")),
        sample_diffusion=lambda *args, **kwargs: sample_calls.append("sample"),
    )

    with pytest.raises(RuntimeError, match="remote rollout seed copy OOM"):
        opendde.OpenDDE.run_sample_diffusion_stage(
            module,
            pred_dict={},
            input_feature_dict={"inference_seed": torch.tensor(1)},
            s_inputs=torch.zeros(1, 1),
            s=torch.zeros(1, 1),
            z=torch.zeros(1, 1, 1),
            pair_z_spec=SimpleNamespace(),
            cache={"pair_z_spec": SimpleNamespace(), "pair_z": torch.zeros(1)},
            N_sample=1,
            noise_schedule=torch.zeros(1),
            chunk_size=None,
            inplace_safe=False,
        )

    assert sample_calls == []


def test_diffusion_rollout_error_wins_over_workspace_cleanup_error():
    from opendde.model import opendde

    events = []
    mesh = SimpleNamespace(group_2d=object(), layout=SimpleNamespace(shape=(1, 2)))

    def synchronize(_mesh, action, *, description):
        events.append(description)
        return action() if action is not None else None

    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        _run_foldcp_local_action_synchronized=synchronize,
        sample_diffusion=lambda **_kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("primary diffusion OOM")
        ),
        diffusion_module=SimpleNamespace(
            diffusion_transformer=SimpleNamespace(
                clear_foldcp_attention_workspace=lambda: (_ for _ in ()).throw(
                    RuntimeError("secondary workspace cleanup failure")
                )
            )
        ),
        configs=SimpleNamespace(
            infer_setting=SimpleNamespace(sample_diffusion_chunk_size=None)
        ),
        enable_efficient_fusion=False,
    )

    with pytest.raises(torch.OutOfMemoryError, match="primary diffusion OOM"):
        opendde.OpenDDE.run_sample_diffusion_stage(
            module,
            pred_dict={},
            input_feature_dict={"inference_seed": 1},
            s_inputs=torch.zeros(1, 1),
            s=torch.zeros(1, 1),
            z=torch.zeros(1, 1, 1),
            pair_z_spec=SimpleNamespace(),
            cache={
                "pair_z_spec": SimpleNamespace(),
                "pair_z": torch.zeros(1),
                "p_lm/c_l": (torch.zeros(1), torch.zeros(1)),
            },
            N_sample=1,
            noise_schedule=torch.zeros(1),
            chunk_size=None,
            inplace_safe=False,
        )

    assert events == [
        "Fold-CP diffusion rollout-input preparation",
        "Fold-CP diffusion workspace cleanup",
        "Fold-CP diffusion rollout failure propagation",
    ]


def test_single_gpu_diffusion_success_skips_distributed_error_handshake():
    from opendde.model import opendde

    events = []
    coordinate = torch.zeros(1, 2, 3)

    def local_action(_mesh, action, *, description):
        events.append(description)
        return action()

    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: None,
        _run_foldcp_local_action_synchronized=local_action,
        sample_diffusion=lambda **_kwargs: coordinate,
        diffusion_module=SimpleNamespace(
            diffusion_transformer=SimpleNamespace(
                clear_foldcp_attention_workspace=lambda: None
            )
        ),
        configs=SimpleNamespace(
            infer_setting=SimpleNamespace(sample_diffusion_chunk_size=None)
        ),
        enable_efficient_fusion=False,
    )

    pred_dict = {}
    result = opendde.OpenDDE.run_sample_diffusion_stage(
        module,
        pred_dict=pred_dict,
        input_feature_dict={"inference_seed": 1},
        s_inputs=torch.zeros(1, 1),
        s=torch.zeros(1, 1),
        z=torch.zeros(1, 1, 1),
        pair_z_spec=None,
        cache={
            "pair_z_spec": None,
            "pair_z": torch.zeros(1),
            "p_lm/c_l": (torch.zeros(1), torch.zeros(1)),
        },
        N_sample=1,
        noise_schedule=torch.zeros(1),
        chunk_size=None,
        inplace_safe=False,
    )

    assert result is coordinate
    assert pred_dict["coordinate"] is coordinate
    assert events == [
        "Fold-CP diffusion rollout-input preparation",
        "Fold-CP diffusion workspace cleanup",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "expected_forward_calls"),
    [
        ("Fold-CP diffusion scaled-coordinate preparation", 0),
        ("Fold-CP diffusion denoised-coordinate rescaling", 1),
    ],
)
def test_one_by_p_diffusion_coordinate_boundary_failure_is_synchronized(
    monkeypatch,
    failure_stage,
    expected_forward_calls,
):
    from opendde.model.modules import diffusion

    forward_calls = []
    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        layout=SimpleNamespace(shape=(1, 3)),
    )
    module = SimpleNamespace(
        sigma_data=16.0,
        _maybe_foldcp_mesh=lambda: mesh,
        f_forward=lambda **kwargs: (
            forward_calls.append(kwargs) or torch.zeros_like(kwargs["r_noisy"])
        ),
    )

    def synchronize(action, *, group, description):
        if description == failure_stage:
            raise RuntimeError(f"remote {failure_stage} OOM")
        return action()

    monkeypatch.setattr(
        diffusion,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match=f"remote {failure_stage} OOM"):
        diffusion.DiffusionModule.forward(
            module,
            x_noisy=torch.zeros(1, 4, 3),
            t_hat_noise_level=torch.ones(1),
            input_feature_dict={},
            s_inputs=torch.zeros(2, 2),
            s_trunk=torch.zeros(2, 2),
            z_trunk=torch.zeros(2, 2, 2),
            pair_z=torch.zeros(2, 2, 2),
            p_lm=torch.zeros(1),
            c_l=torch.zeros(1),
        )

    assert len(forward_calls) == expected_forward_calls


def test_distogram_assembly_failure_is_synchronized_after_gather(monkeypatch):
    from opendde.distributed.foldcp import distogram

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        cp_global_ranks=(0,),
        layout=SimpleNamespace(numel=1, to_coord=lambda _rank: (0, 0)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    synchronized_stages = []

    monkeypatch.setattr(distogram.torch.distributed, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        distogram.torch.distributed,
        "gather",
        lambda local, *, gather_list, dst, group: gather_list[0].copy_(local),
    )

    def run_synchronized(action, *, group, description):
        synchronized_stages.append(description)
        return action() if action is not None else None

    monkeypatch.setattr(
        distogram,
        "run_group_rank_action_synchronized",
        run_synchronized,
    )
    monkeypatch.setattr(
        distogram,
        "_copy_pair_shard_into_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("copy destination unavailable")
        ),
    )

    with pytest.raises(RuntimeError, match="copy destination unavailable"):
        distogram._gather_pair_like_collective_to_rank0(
            torch.zeros(2, 2, 1),
            spec,
            mesh,
        )

    assert synchronized_stages == [
        "distogram destination-buffer allocation",
        "distogram output assembly",
    ]


def test_pair_gather_does_not_enter_collective_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import pair_sharding

    group = object()
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    collective_calls = []

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pair_sharding.dist,
        "all_gather",
        lambda *args, **kwargs: collective_calls.append("all_gather"),
    )
    monkeypatch.setattr(
        pair_sharding,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote pair allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote pair allocation OOM"):
        pair_sharding.gather_pair_tensor(
            torch.zeros(2, 2, 1),
            spec,
            group,
        )

    assert collective_calls == []


def test_pair_gather_contiguous_send_failure_stops_before_collective(monkeypatch):
    from opendde.distributed.foldcp import pair_sharding

    class BrokenShard:
        ndim = 3
        shape = (2, 2, 1)

        def narrow(self, *_args, **_kwargs):
            return self

        def contiguous(self):
            raise RuntimeError("contiguous send allocation OOM")

    group = object()
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    collective_calls = []

    def gather_errors(output, local_error, *, group):
        output[:] = [local_error]

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pair_sharding.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(pair_sharding.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(pair_sharding.dist, "all_gather_object", gather_errors)
    monkeypatch.setattr(
        pair_sharding.dist,
        "all_gather",
        lambda *args, **kwargs: collective_calls.append("all_gather"),
    )
    monkeypatch.setattr(pair_sharding.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="contiguous send allocation OOM"):
        pair_sharding.gather_pair_tensor(BrokenShard(), spec, group)

    assert collective_calls == []


def test_pair_gather_uses_arbitrary_1xp_streaming_ring(monkeypatch):
    from opendde.distributed.foldcp import pair_sharding

    group = object()
    local = torch.zeros(3, 3, 2)
    spec = FoldCPPairShardSpec(
        original_shape=(3, 11, 2),
        padded_shape=(3, 15, 2),
        pair_dims=(0, 1),
        row_range=(0, 3),
        col_range=(6, 9),
        mesh_shape=(1, 5),
        mesh_coord=(0, 2),
    )
    calls = []
    result = object()

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pair_sharding.dist, "get_world_size", lambda _group: 5)
    monkeypatch.setattr(pair_sharding.dist, "get_rank", lambda _group: 2)
    monkeypatch.setattr(
        pair_sharding.dist,
        "get_global_rank",
        lambda _group, rank: rank,
    )
    monkeypatch.setattr(
        pair_sharding,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    def gather(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return result

    monkeypatch.setattr(pair_sharding, "gather_tensor_by_ring", gather)

    assert pair_sharding.gather_pair_tensor(local, spec, group) is result
    assert len(calls) == 1
    tensor, kwargs = calls[0]
    assert tensor is local
    assert kwargs["group"] is group
    assert kwargs["local_index"] == 2
    assert kwargs["side"] == 5
    assert kwargs["dim"] == 1
    assert kwargs["length"] == 11


def test_pair_like_rank_validation_stops_before_collective(monkeypatch):
    from opendde.distributed.foldcp import pair_sharding

    group = object()
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    collective_calls = []

    def gather_errors(output, local_error, *, group):
        output[:] = [local_error]

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pair_sharding.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(pair_sharding.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(pair_sharding.dist, "all_gather_object", gather_errors)
    monkeypatch.setattr(
        pair_sharding.dist,
        "all_gather",
        lambda *args, **kwargs: collective_calls.append("all_gather"),
    )
    monkeypatch.setattr(pair_sharding.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="same rank as the source pair tensor"):
        pair_sharding.gather_pair_tensor_like(
            torch.zeros(2, 2),
            spec,
            group,
        )

    assert collective_calls == []


def test_pair_gather_mesh_group_mismatch_stops_before_collective(monkeypatch):
    from opendde.distributed.foldcp import pair_sharding

    group = object()
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 1),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    collective_calls = []

    def gather_errors(output, local_error, *, group):
        output[:] = [local_error]

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pair_sharding.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(pair_sharding.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(pair_sharding.dist, "all_gather_object", gather_errors)
    monkeypatch.setattr(
        pair_sharding.dist,
        "all_gather",
        lambda *args, **kwargs: collective_calls.append("all_gather"),
    )
    monkeypatch.setattr(pair_sharding.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="group size must match the shard mesh"):
        pair_sharding.gather_pair_tensor(
            torch.zeros(2, 1, 1),
            spec,
            group,
        )

    assert collective_calls == []


def test_pair_to_rank_invalid_destination_stops_before_p2p(monkeypatch):
    from opendde.distributed.foldcp import pair_sharding

    group = object()
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    p2p_calls = []

    def gather_errors(output, local_error, *, group):
        output[:] = [local_error]

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pair_sharding.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(pair_sharding.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(pair_sharding.dist, "all_gather_object", gather_errors)
    monkeypatch.setattr(
        pair_sharding.dist,
        "get_global_rank",
        lambda *_args, **_kwargs: p2p_calls.append("global rank"),
    )
    monkeypatch.setattr(
        pair_sharding.dist,
        "send",
        lambda *_args, **_kwargs: p2p_calls.append("send"),
    )
    monkeypatch.setattr(
        pair_sharding.dist,
        "recv",
        lambda *_args, **_kwargs: p2p_calls.append("recv"),
    )
    monkeypatch.setattr(pair_sharding.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="destination group rank must be in"):
        pair_sharding.gather_pair_tensor_like_to_rank(
            torch.zeros(2, 2, 1),
            spec,
            group,
            dst_group_rank=1,
        )

    assert p2p_calls == []


def test_pair_to_rank_does_not_send_after_destination_allocation_oom(monkeypatch):
    from opendde.distributed.foldcp import pair_sharding

    group = object()
    spec = FoldCPPairShardSpec(
        original_shape=(2, 2, 1),
        padded_shape=(2, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    sends = []

    monkeypatch.setattr(pair_sharding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(pair_sharding.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        pair_sharding.dist, "get_global_rank", lambda _group, rank: rank
    )
    monkeypatch.setattr(
        pair_sharding.dist,
        "send",
        lambda *args, **kwargs: sends.append("send"),
    )
    monkeypatch.setattr(
        pair_sharding,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("destination allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="destination allocation OOM"):
        pair_sharding.gather_pair_tensor_like_to_rank(
            torch.zeros(2, 2, 1),
            spec,
            group,
        )

    assert sends == []


def test_pair_to_rank_uses_cpu_control_group_for_transfer_barriers(monkeypatch):
    from opendde.distributed.foldcp import comm, pair_sharding

    data_group = object()
    control_group = object()
    barriers = []
    monkeypatch.setattr(comm, "_CPU_CONTROL_GROUP", control_group)
    monkeypatch.setattr(
        comm.dist,
        "get_world_size",
        lambda group: 2 if group in {data_group, control_group} else 0,
    )
    monkeypatch.setattr(
        comm.dist,
        "barrier",
        lambda *, group: barriers.append(group),
    )

    # Exercise the helper through the binding used by pair sharding. The
    # payload transfers remain on `data_group`; only the ordering barrier moves
    # to the same-rank CPU control plane.
    pair_sharding.foldcp_control_barrier(data_group)

    assert barriers == [control_group]


@pytest.mark.parametrize(
    ("function_name", "failure_stage"),
    [
        (
            "distributed_atom_window_pair_context",
            "atom-window local pair-context computation",
        ),
        (
            "distributed_atom_window_attention",
            "atom-window local attention computation",
        ),
    ],
)
def test_public_atom_window_local_failure_is_synchronized(
    monkeypatch,
    function_name,
    failure_stage,
):
    from opendde.distributed.foldcp import atom_window

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(numel=2),
    )

    def synchronize(action, *, description, **_kwargs):
        if description == failure_stage:
            raise RuntimeError(f"remote {failure_stage} OOM")
        return action()

    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    function = getattr(atom_window, function_name)

    with pytest.raises(RuntimeError, match=f"remote {failure_stage} OOM"):
        if function_name.endswith("pair_context"):
            function(
                torch.zeros(2, 2, 1),
                torch.zeros(4, dtype=torch.long),
                n_queries=2,
                n_keys=4,
                mesh=mesh,
            )
        else:
            function(
                torch.zeros(4, 1),
                torch.zeros(4, 1),
                torch.zeros(4, 1),
                n_queries=2,
                n_keys=4,
                mesh=mesh,
            )


def test_atom_window_gathered_output_failure_is_synchronized(monkeypatch):
    from opendde.distributed.foldcp import atom_window

    group = object()
    spec = atom_window.FoldCPWindowShardSpec(
        n_atom=4,
        n_windows=2,
        n_queries=2,
        n_keys=4,
        q_pad=0,
        block_range=(0, 2),
        size_cp=1,
    )
    monkeypatch.setattr(
        atom_window,
        "gather_window_blocks",
        lambda *_args, **_kwargs: torch.zeros(2, 2, 1),
    )

    def synchronize(action, *, description, **_kwargs):
        if description == "atom-window gathered-attention finalization":
            raise RuntimeError("remote atom-window output reshape OOM")
        return action()

    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="remote atom-window output reshape OOM"):
        atom_window.gather_window_attention_output(
            torch.zeros(2, 2, 1),
            spec,
            group,
        )


def test_structural_pair_context_supports_nondivisible_one_by_p(monkeypatch):
    from opendde.distributed.foldcp import structural_pair
    from opendde.distributed.foldcp.layout import FoldCP2DLayout

    monkeypatch.setattr(
        structural_pair,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    z_res = torch.arange(18.0).reshape(1, 3, 3, 2)
    parent = torch.tensor([2, 0, 1])
    role = torch.tensor([0, 1, 2])
    role_embedding = torch.arange(16.0).reshape(8, 2)
    expected = structural_pair.serial_structural_pair_context(
        z_res,
        parent,
        role,
        role_embedding,
    )
    assembled = torch.zeros_like(expected)

    for col in range(2):
        mesh = SimpleNamespace(
            group_2d=object(),
            layout=FoldCP2DLayout((1, 2)),
            coord=(0, col),
        )
        local, spec = structural_pair.distributed_structural_pair_context(
            z_res,
            parent,
            role,
            role_embedding,
            mesh,
        )
        row_start, row_end = spec.row_range
        col_start, col_end = spec.col_range
        valid_row_end = min(row_end, parent.shape[0])
        valid_col_end = min(col_end, parent.shape[0])
        assembled[
            :,
            row_start:valid_row_end,
            col_start:valid_col_end,
            :,
        ] = local[
            :,
            : valid_row_end - row_start,
            : valid_col_end - col_start,
            :,
        ]

    assert torch.equal(assembled, expected)


def test_structural_pair_remote_local_failure_is_synchronized(monkeypatch):
    from opendde.distributed.foldcp import structural_pair

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(numel=2),
    )

    def synchronize(action, *, description, **_kwargs):
        if description == "structural-pair local context computation":
            raise RuntimeError("remote structural-pair allocation OOM")
        return action()

    monkeypatch.setattr(
        structural_pair,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="remote structural-pair allocation OOM"):
        structural_pair.distributed_structural_pair_context(
            torch.zeros(1, 2, 2, 1),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            torch.zeros(8, 1),
            mesh,
        )


def test_atom_window_ring_does_not_send_after_remote_allocation_oom(monkeypatch):
    from opendde.distributed.foldcp import atom_window

    group = object()
    spec = atom_window.FoldCPWindowShardSpec(
        n_atom=4,
        n_windows=4,
        n_queries=1,
        n_keys=1,
        q_pad=0,
        block_range=(0, 2),
        size_cp=2,
        padded_n_windows=4,
    )
    sends = []

    monkeypatch.setattr(atom_window.dist, "is_available", lambda: True)
    monkeypatch.setattr(atom_window.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(atom_window.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(atom_window.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        atom_window.dist,
        "batch_isend_irecv",
        lambda *args, **kwargs: sends.append("send"),
    )
    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote atom-window allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote atom-window allocation OOM"):
        atom_window.gather_window_blocks(
            torch.zeros(2, 1),
            spec,
            group,
            block_dim=0,
        )

    assert sends == []


def test_atom_window_ring_validation_runs_inside_rank_action(monkeypatch):
    from opendde.distributed.foldcp import atom_window

    actions = []
    group = object()
    spec = atom_window.FoldCPWindowShardSpec(
        n_atom=4,
        n_windows=4,
        n_queries=1,
        n_keys=1,
        q_pad=0,
        block_range=(1, 3),
        size_cp=2,
        padded_n_windows=4,
    )
    monkeypatch.setattr(atom_window.dist, "is_available", lambda: True)
    monkeypatch.setattr(atom_window.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(atom_window.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(atom_window.dist, "get_rank", lambda _group: 0)

    def synchronize(action, *, description, **_kwargs):
        actions.append(description)
        return action()

    monkeypatch.setattr(atom_window, "run_group_rank_action_synchronized", synchronize)

    with pytest.raises(ValueError, match="window block range must follow"):
        atom_window.gather_window_blocks(
            torch.zeros(2, 1),
            spec,
            group,
            block_dim=0,
        )

    assert actions == ["atom-window ring gather allocation"]


def test_atom_window_pair_rows_do_not_gather_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import atom_window

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(2, 4, 1),
        padded_shape=(2, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    gathers = []

    monkeypatch.setattr(
        atom_window.dist,
        "all_gather_into_tensor",
        lambda *args, **kwargs: gathers.append("all_gather"),
    )
    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote pair-row allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote pair-row allocation OOM"):
        atom_window._gather_pair_rows_one_by_p(
            torch.zeros(2, 2, 1),
            spec,
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            mesh,
        )

    assert gathers == []


def test_atom_window_pair_rows_release_source_before_assembly(monkeypatch):
    from opendde.distributed.foldcp import atom_window

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(2, 4, 1),
        padded_shape=(2, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    source_refs = []
    released_before_assembly = []

    def all_gather(gathered, source, **_kwargs):
        source_refs.append(weakref.ref(source))
        gathered.zero_()

    def synchronize(action, *, description, **_kwargs):
        if description == "atom-window pair-row gather assembly":
            gc.collect()
            released_before_assembly.append(source_refs[0]() is None)
        return action()

    monkeypatch.setattr(atom_window.dist, "all_gather_into_tensor", all_gather)
    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = atom_window._gather_pair_rows_one_by_p(
        torch.ones(2, 2, 1),
        spec,
        torch.tensor([[0]]),
        torch.tensor([[0]]),
        mesh,
    )

    assert result.shape == (1, 1, 1, 1)
    assert released_before_assembly == [True]


def test_2d_atom_window_broadcasts_drain_after_local_index_failure(monkeypatch):
    from opendde.distributed.foldcp import atom_window

    broadcasts = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(
            numel=2,
            to_coord=lambda rank: (rank, 0),
        ),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 2, 1),
        padded_shape=(4, 2, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(2, 1),
        mesh_coord=(0, 0),
    )
    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(atom_window.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(atom_window.dist, "get_global_rank", lambda _group, rank: rank)
    monkeypatch.setattr(
        atom_window.dist,
        "broadcast",
        lambda *args, **kwargs: broadcasts.append("broadcast"),
    )
    monkeypatch.setattr(
        atom_window.torch,
        "nonzero",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local atom-window index OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="local atom-window index OOM"):
        atom_window.gather_pair_embedding_in_dense_trunk_from_foldcp_local(
            torch.zeros(2, 2, 1),
            spec,
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            mesh,
        )

    assert broadcasts == ["broadcast", "broadcast"]


def test_2d_atom_window_preparation_failure_stops_before_broadcast(monkeypatch):
    from opendde.distributed.foldcp import atom_window

    broadcasts = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(numel=2),
    )
    monkeypatch.setattr(
        atom_window,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote 2D atom-window allocation OOM")
        ),
    )
    monkeypatch.setattr(
        atom_window.dist,
        "broadcast",
        lambda *args, **kwargs: broadcasts.append("broadcast"),
    )

    with pytest.raises(RuntimeError, match="remote 2D atom-window allocation OOM"):
        atom_window.gather_pair_embedding_in_dense_trunk_from_foldcp_local(
            torch.zeros(2, 2, 1),
            SimpleNamespace(),
            torch.tensor([[0]]),
            torch.tensor([[0]]),
            mesh,
        )

    assert broadcasts == []


def test_msa_exact_path_does_not_gather_after_remote_allocation_oom(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    gathers = []

    monkeypatch.setattr(
        msa_pair_weighted.torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        msa_pair_weighted.dist,
        "all_gather",
        lambda *args, **kwargs: gathers.append("all_gather"),
    )
    monkeypatch.setattr(
        msa_pair_weighted,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote MSA allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote MSA allocation OOM"):
        msa_pair_weighted.distributed_msa_pair_weighted_average_with_full_value(
            torch.zeros(1, 2, 1, 1),
            torch.zeros(1, 1, 2, 1, 1),
            mesh,
            original_tokens=2,
        )

    assert gathers == []


def test_msa_exact_path_streams_source_tokens_by_arbitrary_1xp_ring(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    group = object()
    ring = object()
    full_logits = torch.tensor([0.5, -0.25, 0.75, 1.0, 0.0, -0.5]).reshape(1, 2, 3, 1)
    local_logits = full_logits[:, :, 1:2, :].clone()
    value = torch.tensor([2.0, 4.0, 8.0]).reshape(1, 1, 3, 1, 1)
    calls = []
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 3)),
        coord=(0, 1),
        ring_comm=lambda: SimpleNamespace(comm_row=ring),
    )

    monkeypatch.setattr(
        msa_pair_weighted.torch,
        "are_deterministic_algorithms_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        msa_pair_weighted,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    def gather(tensor, **kwargs):
        calls.append((tensor, kwargs))
        return full_logits

    monkeypatch.setattr(msa_pair_weighted, "gather_tensor_by_ring", gather)

    result = msa_pair_weighted.distributed_msa_pair_weighted_average_with_full_value(
        local_logits,
        value,
        mesh,
        original_tokens=3,
    )
    expected = msa_pair_weighted.serial_msa_pair_weighted_average(
        full_logits,
        value,
    )

    assert torch.equal(result, expected)
    assert len(calls) == 1
    tensor, kwargs = calls[0]
    assert tensor is local_logits
    assert kwargs["comm"] is ring
    assert kwargs["group"] is group
    assert kwargs["local_index"] == 1
    assert kwargs["side"] == 3
    assert kwargs["dim"] == 2
    assert kwargs["length"] == 3


def test_msa_full_value_sharding_failure_stops_before_reduction(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    reductions = []
    mesh = SimpleNamespace(
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    monkeypatch.setattr(
        msa_pair_weighted.torch,
        "are_deterministic_algorithms_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        msa_pair_weighted,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        msa_pair_weighted,
        "shard_msa_value_by_token",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local MSA value-sharding OOM")
        ),
    )
    monkeypatch.setattr(
        msa_pair_weighted.dist,
        "all_reduce",
        lambda *args, **kwargs: reductions.append("all_reduce"),
    )

    with pytest.raises(RuntimeError, match="local MSA value-sharding OOM"):
        msa_pair_weighted.distributed_msa_pair_weighted_average_with_full_value(
            torch.zeros(1, 2, 1, 1),
            torch.zeros(1, 1, 2, 1, 1),
            mesh,
            original_tokens=2,
        )

    assert reductions == []


def test_msa_full_value_nondeterministic_single_rank_matches_serial(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    mesh = SimpleNamespace(
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 1)),
        coord=(0, 0),
    )
    pair_logits = torch.tensor([[[[0.5], [-0.25]], [[1.0], [0.0]]]])
    value = torch.tensor([[[[[2.0]], [[4.0]]]]])
    monkeypatch.setattr(
        msa_pair_weighted.torch,
        "are_deterministic_algorithms_enabled",
        lambda: False,
    )
    monkeypatch.setattr(
        msa_pair_weighted,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        msa_pair_weighted.dist,
        "all_reduce",
        lambda tensor, **_kwargs: tensor,
    )

    result = msa_pair_weighted.distributed_msa_pair_weighted_average_with_full_value(
        pair_logits,
        value,
        mesh,
        original_tokens=2,
    )
    expected = msa_pair_weighted.serial_msa_pair_weighted_average(
        pair_logits,
        value,
    )

    assert torch.equal(result, expected)


def test_msa_row_result_is_zero_copy_for_maintained_1xp(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    local_output = torch.arange(24).reshape(1, 2, 3, 2, 2)
    mesh = SimpleNamespace(
        group_col=object(),
        layout=SimpleNamespace(shape=(1, 4)),
    )
    monkeypatch.setattr(
        msa_pair_weighted.dist,
        "all_gather",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("1xP must not gather its already-complete MSA rows")
        ),
    )

    result = msa_pair_weighted.gather_msa_rows_from_cp(
        local_output,
        mesh,
        token_dim=2,
        original_tokens=3,
    )

    assert result is local_output


def test_msa_pair_row_ring_does_not_send_after_remote_allocation_oom(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    group = object()
    layout = SimpleNamespace(shape=(1, 2), to_linear=lambda coord: coord[1])
    mesh = SimpleNamespace(
        group_row=group,
        layout=layout,
        coord=(0, 0),
        cp_global_ranks=(0, 1),
    )
    sends = []

    monkeypatch.setattr(
        msa_pair_weighted.dist,
        "batch_isend_irecv",
        lambda *args, **kwargs: sends.append("send"),
    )
    monkeypatch.setattr(
        msa_pair_weighted,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote MSA ring allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote MSA ring allocation OOM"):
        msa_pair_weighted.collect_msa_pair_row_slab(
            torch.zeros(1, 1, 2, 1),
            mesh,
            original_tokens=4,
        )

    assert sends == []


def test_msa_pair_row_ring_preflights_schedule_and_uses_row_group(monkeypatch):
    from opendde.distributed.foldcp import msa_pair_weighted

    group = object()
    inside_action = False
    p2p_groups = []

    def _to_linear(coord):
        assert inside_action
        return coord[1]

    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2), to_linear=_to_linear),
        coord=(0, 0),
        cp_global_ranks=(4, 5),
    )

    def _run_action(action, **_kwargs):
        nonlocal inside_action
        inside_action = True
        try:
            return action()
        finally:
            inside_action = False

    def _p2p_op(_op, _tensor, _peer, *, group):
        p2p_groups.append(group)
        return object()

    monkeypatch.setattr(
        msa_pair_weighted,
        "run_group_rank_action_synchronized",
        _run_action,
    )
    monkeypatch.setattr(msa_pair_weighted.dist, "P2POp", _p2p_op)
    monkeypatch.setattr(
        msa_pair_weighted.dist,
        "batch_isend_irecv",
        lambda _ops: [SimpleNamespace(wait=lambda: None)],
    )

    result = msa_pair_weighted.collect_msa_pair_row_slab(
        torch.zeros(1, 1, 2, 1),
        mesh,
        original_tokens=4,
    )

    assert result.shape == (1, 1, 4, 1)
    assert p2p_groups == [group, group]


def test_pairformer_row_gather_does_not_collect_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 3)),
    )
    gathers = []

    monkeypatch.setattr(
        real_pairformer.dist,
        "all_gather_into_tensor",
        lambda *args, **kwargs: gathers.append("all_gather"),
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote Pairformer gather allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote Pairformer gather allocation OOM"):
        real_pairformer._ring_gather_by_row(
            torch.zeros(1, 2),
            mesh,
            dim=-1,
        )

    assert gathers == []


@pytest.mark.parametrize("side", [2, 3])
def test_pairformer_row_gather_validation_runs_inside_rank_action(
    monkeypatch,
    side,
):
    from opendde.distributed.foldcp import real_pairformer

    actions = []
    mesh = SimpleNamespace(
        group_row=object(),
        layout=SimpleNamespace(shape=(1, side)),
    )

    def synchronize(action, *, description, **_kwargs):
        actions.append(description)
        return action()

    monkeypatch.setattr(
        real_pairformer, "run_group_rank_action_synchronized", synchronize
    )

    with pytest.raises(ValueError, match="row gather length must be"):
        real_pairformer._ring_gather_by_row(
            torch.zeros(1, 2),
            mesh,
            dim=-1,
            length=side * 2 + 1,
        )

    assert actions == [
        "Pairformer row-ring preparation"
        if side == 2
        else "Pairformer row all-gather allocation"
    ]


def test_pairformer_transpose_does_not_collect_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    collectives = []

    monkeypatch.setattr(
        real_pairformer.dist,
        "all_to_all_single",
        lambda *args, **kwargs: collectives.append("all_to_all"),
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote Pairformer transpose allocation OOM")
        ),
    )

    with pytest.raises(
        RuntimeError, match="remote Pairformer transpose allocation OOM"
    ):
        real_pairformer._one_by_p_transpose_columns_to_owned_rows(
            torch.zeros(1, 4, 2, 1),
            mesh,
            spec,
        )

    assert collectives == []


def test_pairformer_transpose_validation_runs_inside_rank_action(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    actions = []
    mesh = SimpleNamespace(
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )

    def synchronize(action, *, description, **_kwargs):
        actions.append(description)
        return action()

    monkeypatch.setattr(
        real_pairformer, "run_group_rank_action_synchronized", synchronize
    )

    with pytest.raises(ValueError, match="1xP pair transpose expects"):
        real_pairformer._one_by_p_transpose_columns_to_owned_rows(
            torch.zeros(4, 2, 1),
            mesh,
            SimpleNamespace(),
        )

    assert actions == ["Pairformer row-transpose all-to-all allocation"]


def test_pairformer_transpose_releases_send_before_result_assembly(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = SimpleNamespace(original_shape=(1, 4, 4, 1), pair_dims=(1, 2))
    send_refs = []
    released_before_assembly = []

    def all_to_all(recv, send, **_kwargs):
        send_refs.append(weakref.ref(send))
        recv.copy_(send)

    def synchronize(action, *, description, **_kwargs):
        if description == "Pairformer row-transpose all-to-all assembly":
            gc.collect()
            released_before_assembly.append(send_refs[0]() is None)
        return action()

    monkeypatch.setattr(real_pairformer.dist, "all_to_all_single", all_to_all)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = real_pairformer._one_by_p_transpose_columns_to_owned_rows(
        torch.arange(8, dtype=torch.float32).reshape(1, 4, 2, 1),
        mesh,
        spec,
    )

    assert result.shape == (1, 2, 4, 1)
    assert released_before_assembly == [True]


def test_pairformer_row_all_gather_releases_source_before_assembly(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 3)),
    )
    source_refs = []
    released_before_assembly = []

    def all_gather(gathered, source, **_kwargs):
        source_refs.append(weakref.ref(source))
        gathered.zero_()

    def synchronize(action, *, description, **_kwargs):
        if description == "Pairformer row all-gather assembly":
            gc.collect()
            released_before_assembly.append(source_refs[0]() is None)
        return action()

    monkeypatch.setattr(real_pairformer.dist, "all_gather_into_tensor", all_gather)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = real_pairformer._ring_gather_by_row(
        torch.ones(2, 1),
        mesh,
        dim=1,
        length=3,
    )

    assert result.shape == (2, 3)
    assert released_before_assembly == [True]


def test_distogram_transpose_does_not_collect_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import distogram

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        group_2d=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    collectives = []

    monkeypatch.setattr(
        distogram.dist,
        "all_to_all_single",
        lambda *args, **kwargs: collectives.append("all_to_all"),
    )
    monkeypatch.setattr(
        distogram,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote distogram transpose allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote distogram transpose allocation OOM"):
        distogram._transpose_pair_tile_collective(
            torch.zeros(4, 2, 1),
            mesh,
        )

    assert collectives == []


def test_distogram_transpose_releases_send_before_result_assembly(monkeypatch):
    from opendde.distributed.foldcp import distogram

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        group_2d=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    send_refs = []
    released_before_assembly = []

    def all_to_all(recv, send, **_kwargs):
        send_refs.append(weakref.ref(send))
        recv.copy_(send)

    def synchronize(action, *, description, **_kwargs):
        if description == "distogram 1xP transpose all-to-all assembly":
            gc.collect()
            released_before_assembly.append(send_refs[0]() is None)
        return action()

    monkeypatch.setattr(distogram.dist, "all_to_all_single", all_to_all)
    monkeypatch.setattr(
        distogram,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = distogram._transpose_pair_tile_collective(
        torch.arange(8, dtype=torch.float32).reshape(4, 2, 1),
        mesh,
    )

    assert result.shape == (4, 2, 1)
    assert released_before_assembly == [True]


def test_confidence_rowslab_projection_releases_gathered_sources(monkeypatch):
    from opendde.distributed.foldcp import confidence

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_row=group,
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    source_refs = []

    def gather(_value, _mesh):
        source = torch.zeros(4, 4, 1)
        source_refs.append(weakref.ref(source))
        return source

    def layer_norm(value):
        gc.collect()
        assert len(source_refs) == 2
        assert all(reference() is None for reference in source_refs)
        return value

    monkeypatch.setattr(confidence, "_collect_pair_row_slab", gather)
    monkeypatch.setattr(
        confidence,
        "_confidence_should_stream_projection",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    result = confidence._confidence_pair_logits_local_rowslab_synchronized(
        z_pair_local=torch.zeros(4, 2, 1),
        z_pair_spec=spec,
        mesh=mesh,
        layer_norm=layer_norm,
        linear=torch.nn.Linear(1, 1, bias=False),
        add_local=torch.zeros(4, 2, 1),
    )

    assert result.shape == (4, 2, 1)


def test_distogram_rowslab_projection_releases_gathered_source(monkeypatch):
    from opendde.distributed.foldcp import distogram

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_row=group,
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
        ring_comm=lambda: SimpleNamespace(comm_row=object()),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(3, 3, 1),
        padded_shape=(3, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 3),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    source_ref = None

    def gather(*_args, **_kwargs):
        nonlocal source_ref
        source = torch.zeros(3, 4, 1)
        source_ref = weakref.ref(source)
        return source

    def linear(value):
        gc.collect()
        assert source_ref is not None and source_ref() is None
        return value

    monkeypatch.setattr(distogram, "gather_tensor_by_ring", gather)
    monkeypatch.setattr(
        distogram,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    result = distogram._project_pair_row_slab_local(
        torch.zeros(3, 2, 1),
        spec,
        mesh,
        linear,
    )

    assert result.shape == (3, 2, 1)


def test_distogram_projection_failure_is_synchronized_after_row_gather(monkeypatch):
    from opendde.distributed.foldcp import distogram

    actions = []
    mesh = SimpleNamespace(
        group_2d=object(),
        group_row=object(),
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
        ring_comm=lambda: SimpleNamespace(comm_row=object()),
    )
    spec = SimpleNamespace(
        row_range=(0, 2),
        col_range=(0, 2),
        original_shape=(4, 4, 1),
        pair_dims=(0, 1),
    )

    def synchronize(action, *, description, **_kwargs):
        actions.append(description)
        return action()

    monkeypatch.setattr(distogram, "run_group_rank_action_synchronized", synchronize)
    monkeypatch.setattr(
        distogram,
        "gather_tensor_by_ring",
        lambda tensor, **_kwargs: tensor.new_zeros(2, 4, 1),
    )

    with pytest.raises(RuntimeError, match="local distogram projection OOM"):
        distogram._project_pair_row_slab_local(
            torch.zeros(2, 2, 1),
            spec,
            mesh,
            lambda _tensor: (_ for _ in ()).throw(
                RuntimeError("local distogram projection OOM")
            ),
        )

    assert actions == [
        "distogram projection metadata preparation",
        "distogram local row-slab projection",
    ]


def test_distogram_local_contact_failure_stops_before_public_gather(monkeypatch):
    from opendde.distributed.foldcp import distogram

    gathers = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(numel=2),
    )
    monkeypatch.setattr(
        distogram,
        "distogram_contact_probs_local",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("local distogram softmax OOM")
        ),
    )
    monkeypatch.setattr(
        distogram,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        distogram,
        "gather_pair_tensor_like",
        lambda *args, **kwargs: gathers.append("gather"),
    )

    with pytest.raises(RuntimeError, match="local distogram softmax OOM"):
        distogram.distributed_distogram_contact_probs(
            z_pair_local=torch.zeros(2, 2, 1),
            z_pair_spec=SimpleNamespace(),
            mesh=mesh,
            linear=torch.nn.Identity(),
            min_bin=2.0,
            max_bin=22.0,
            no_bins=64,
        )

    assert gathers == []


def test_distogram_full_pair_failure_stops_before_public_gather(monkeypatch):
    from opendde.distributed.foldcp import distogram

    gathers = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(numel=2),
    )
    monkeypatch.setattr(
        distogram,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote full-pair distogram OOM")
        ),
    )
    monkeypatch.setattr(
        distogram,
        "gather_pair_tensor_like",
        lambda *args, **kwargs: gathers.append("gather"),
    )

    with pytest.raises(RuntimeError, match="remote full-pair distogram OOM"):
        distogram.distributed_distogram_contact_probs_from_full_pair(
            z_pair=torch.zeros(4, 4, 1),
            mesh=mesh,
            linear=torch.nn.Identity(),
            min_bin=2.0,
            max_bin=22.0,
            no_bins=64,
        )

    assert gathers == []


def test_triangle_bmm_does_not_dispatch_after_remote_allocation_oom(monkeypatch):
    from opendde.distributed.foldcp import triangular_mult

    ring = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote triangle BMM allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote triangle BMM allocation OOM"):
        triangular_mult._distributed_bmm_double_buffered(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            ring,
            transpose_arg=None,
        )


def test_triangle_bmm_drains_ring_after_local_compute_oom(monkeypatch):
    from opendde.distributed.foldcp import triangular_mult

    events = []

    class FakeComm:
        def __init__(self, name):
            self.name = name

        def enqueue_to_dispatch(self, send, recv=None):
            events.append((self.name, "enqueue"))
            if recv is None:
                recv = torch.empty_like(send)
            recv.copy_(send)
            return recv

        def wait_until_finished(self):
            events.append((self.name, "wait"))

    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        comm_2d_trans=FakeComm("transpose"),
        comm_row_init=FakeComm("row_init"),
        comm_col_init=FakeComm("col_init"),
        comm_row=FakeComm("row"),
        comm_col=FakeComm("col"),
    )

    def run_synchronized(action, **_kwargs):
        return action() if action is not None else None

    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        run_synchronized,
    )
    monkeypatch.setattr(
        triangular_mult.torch,
        "matmul",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local matmul workspace OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="local matmul workspace OOM"):
        triangular_mult._distributed_bmm_double_buffered(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            ring,
            transpose_arg=None,
        )

    assert events.count(("row", "enqueue")) == 1
    assert events.count(("row", "wait")) == 1
    assert events.count(("col", "enqueue")) == 1
    assert events.count(("col", "wait")) == 1


def test_triangle_bmm_materializes_final_permutation_inside_synchronized_stage(
    monkeypatch,
):
    from opendde.distributed.foldcp import triangular_mult

    class FakeComm:
        def enqueue_to_dispatch(self, send, recv=None):
            if recv is None:
                recv = torch.empty_like(send)
            recv.copy_(send)
            return recv

        def wait_until_finished(self):
            return None

    comm = FakeComm()
    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 1)),
        comm_2d_trans=comm,
        comm_row_init=comm,
        comm_col_init=comm,
        comm_row=comm,
        comm_col=comm,
    )
    synchronized_stages = []

    def run_synchronized(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        run_synchronized,
    )
    lhs = torch.arange(6.0).reshape(1, 2, 3)
    rhs = torch.arange(12.0).reshape(1, 3, 4)
    result = triangular_mult._distributed_bmm_double_buffered(
        lhs,
        rhs,
        ring,
        transpose_arg=None,
        permute_out=(0, 2, 1),
    )

    assert torch.equal(result, torch.matmul(lhs, rhs).permute(0, 2, 1))
    assert result.is_contiguous()
    assert synchronized_stages[-1] == "triangle-multiplication ring compute"


def test_triangle_channel_output_oom_stops_before_ring_dispatch(monkeypatch):
    from opendde.distributed.foldcp import triangular_mult

    dispatches = []
    comm = SimpleNamespace(
        enqueue_to_dispatch=lambda *args, **kwargs: dispatches.append("dispatch")
    )
    ring = SimpleNamespace(group_2d=object(), comm_2d_trans=comm)
    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote channel output allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote channel output allocation OOM"):
        triangular_mult.distributed_triangle_multiplication(
            torch.zeros(1, 2, 2, 16),
            torch.zeros(1, 2, 2, 16),
            ring,
            triangular_mult.TriangleMultiplicationDirection.OUTGOING,
        )

    assert dispatches == []


def test_triangle_input_failure_stops_before_ring_dispatch(monkeypatch):
    from opendde.distributed.foldcp import triangular_mult

    dispatches = []
    comm = SimpleNamespace(
        enqueue_to_dispatch=lambda *args, **kwargs: dispatches.append("dispatch"),
        exchange=lambda *args, **kwargs: dispatches.append("dispatch"),
    )
    ring = SimpleNamespace(group_2d=object(), comm_2d_trans=comm)
    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote triangle input validation failure")
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="remote triangle input validation failure",
    ):
        triangular_mult.distributed_triangle_multiplication(
            torch.zeros(1, 2, 2, 16),
            torch.zeros(1, 2, 2, 16),
            ring,
            triangular_mult.TriangleMultiplicationDirection.OUTGOING,
        )

    assert dispatches == []


def test_triangle_channel_copy_failure_stops_before_next_ring(monkeypatch):
    from opendde.distributed.foldcp import triangular_mult

    class FailingChannelView:
        def copy_(self, _source):
            raise RuntimeError("local channel output copy OOM")

    class FailingChannelOutput:
        def __getitem__(self, _index):
            return FailingChannelView()

    synchronized_descriptions = []
    ring_calls = []

    def run_synchronized(action, *, description, **_kwargs):
        synchronized_descriptions.append(description)
        if description == "triangle-multiplication channel output allocation":
            return FailingChannelOutput()
        return action()

    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        run_synchronized,
    )
    monkeypatch.setattr(
        triangular_mult,
        "_distributed_triangle_multiplication_no_chunk",
        lambda *args, **kwargs: ring_calls.append("ring") or torch.zeros(1, 2, 2, 8),
    )
    ring = SimpleNamespace(group_2d=object())

    with pytest.raises(RuntimeError, match="local channel output copy OOM"):
        triangular_mult.distributed_triangle_multiplication(
            torch.zeros(1, 2, 2, 16),
            torch.zeros(1, 2, 2, 16),
            ring,
            triangular_mult.TriangleMultiplicationDirection.OUTGOING,
        )

    assert ring_calls == ["ring"]
    assert synchronized_descriptions[-1] == (
        "triangle-multiplication channel output copy"
    )


def test_streamed_triangle_bmm_does_not_dispatch_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import triangular_mult

    exchanges = []
    comm = SimpleNamespace(
        exchange=lambda *args, **kwargs: exchanges.append("exchange")
    )
    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        comm_2d_trans=comm,
        comm_row_init=comm,
        comm_col_init=comm,
        comm_row=comm,
        comm_col=comm,
    )
    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote streamed BMM allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote streamed BMM allocation OOM"):
        triangular_mult._distributed_bmm_streamed(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            ring,
            row_block_size=1,
            col_block_size=1,
            permute_out=None,
            transpose_arg=None,
        )

    assert exchanges == []


def test_streamed_triangle_bmm_drains_ring_after_local_compute_oom(monkeypatch):
    from opendde.distributed.foldcp import triangular_mult

    events = []

    class FakeComm:
        def __init__(self, name):
            self.name = name

        def exchange(self, send, to_recv=None):
            events.append(self.name)
            if to_recv is None:
                to_recv = torch.empty_like(send)
            to_recv.copy_(send)
            return to_recv

    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        comm_2d_trans=FakeComm("transpose"),
        comm_row_init=FakeComm("row_init"),
        comm_col_init=FakeComm("col_init"),
        comm_row=FakeComm("row"),
        comm_col=FakeComm("col"),
    )
    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        triangular_mult.torch,
        "matmul",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local streamed matmul workspace OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="local streamed matmul workspace OOM"):
        triangular_mult._distributed_bmm_streamed(
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 2),
            ring,
            row_block_size=2,
            col_block_size=2,
            permute_out=None,
            transpose_arg=None,
        )

    assert events.count("row_init") == 1
    assert events.count("col_init") == 1
    assert events.count("row") == 1
    assert events.count("col") == 1


def test_streamed_triangle_bmm_releases_completed_block_before_next_allocation(
    monkeypatch,
):
    from opendde.distributed.foldcp import triangular_mult

    class FakeComm:
        @staticmethod
        def exchange(send, to_recv=None):
            if to_recv is None:
                to_recv = torch.empty_like(send)
            to_recv.copy_(send)
            return to_recv

    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 1)),
        comm_2d_trans=FakeComm(),
        comm_row_init=FakeComm(),
        comm_col_init=FakeComm(),
        comm_row=FakeComm(),
        comm_col=FakeComm(),
    )
    prior_refs = []
    allocation_count = 0

    def synchronize(action, *, description, **_kwargs):
        nonlocal allocation_count, prior_refs
        if description != "streamed triangle-multiplication block allocation":
            return action()
        if allocation_count:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        result = action()
        prior_refs = [weakref.ref(value) for value in result]
        allocation_count += 1
        return result

    monkeypatch.setattr(
        triangular_mult,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = triangular_mult._distributed_bmm_streamed(
        torch.ones(1, 2, 2),
        torch.ones(1, 2, 2),
        ring,
        row_block_size=1,
        col_block_size=1,
        permute_out=None,
        transpose_arg=None,
    )

    assert result.shape == (1, 2, 2)
    assert allocation_count == 4


def test_local_opm_failure_stops_before_pairformer_collectives(monkeypatch):
    from opendde.model.modules import pairformer

    pairformer_calls = []
    module = SimpleNamespace(
        _foldcp_outer_product_mean_local_update=lambda *args, **kwargs: torch.zeros(
            1, 2, 1, 1
        ),
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    monkeypatch.setattr(pairformer.torch, "is_grad_enabled", lambda: False)
    monkeypatch.setattr(
        pairformer.torch, "are_deterministic_algorithms_enabled", lambda: True
    )
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote local OPM allocation OOM")
        ),
    )
    monkeypatch.setattr(
        pairformer,
        "distributed_pairformer_block_pair_update",
        lambda *args, **kwargs: pairformer_calls.append("pairformer"),
    )

    with pytest.raises(RuntimeError, match="remote local OPM allocation OOM"):
        pairformer.MSABlock._forward_foldcp_local_pair_update(
            module,
            m=torch.zeros(1, 1, 2, 1),
            z_local=torch.zeros(1, 2, 1, 1),
            z_spec=SimpleNamespace(),
            mesh=mesh,
            mask_local=None,
        )

    assert pairformer_calls == []


def test_trunk_initialization_failure_stops_before_model_collectives(monkeypatch):
    from opendde.model import opendde

    mesh = SimpleNamespace(group_2d=object())
    model = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        linear_no_bias_zinit1=None,
        linear_no_bias_zinit2=None,
        relative_position_encoding=None,
        linear_no_bias_token_bond=None,
    )
    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote trunk initialization OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote trunk initialization OOM"):
        opendde.OpenDDE._maybe_get_pairformer_output_foldcp_local(
            model,
            input_feature_dict={},
            N_cycle=1,
            s_inputs=torch.zeros(2, 1),
            s_init=torch.zeros(2, 1),
        )


def test_trunk_output_failure_is_synchronized_before_next_stage(monkeypatch):
    from opendde.model import opendde

    events = []
    mesh = SimpleNamespace(group_2d=object())
    model = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        linear_no_bias_zinit1=None,
        linear_no_bias_zinit2=None,
        relative_position_encoding=None,
        linear_no_bias_token_bond=None,
    )
    z_local = torch.zeros(2, 1, 1)
    z_spec = SimpleNamespace()

    monkeypatch.setattr(
        opendde,
        "build_trunk_z_init_local",
        lambda **_kwargs: (z_local, z_spec),
    )

    def synchronize(action, *, description, **_kwargs):
        events.append(description)
        if description == "Fold-CP trunk output finalization":
            raise RuntimeError("remote trunk output OOM")
        return action()

    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="remote trunk output OOM"):
        opendde.OpenDDE._maybe_get_pairformer_output_foldcp_local(
            model,
            input_feature_dict={"relp": torch.zeros(1), "token_bonds": torch.zeros(1)},
            N_cycle=0,
            s_inputs=torch.zeros(2, 1),
            s_init=torch.zeros(2, 1),
        )

    assert events == [
        "Fold-CP trunk initialization",
        "Fold-CP trunk output finalization",
    ]


def test_foldcp_trunk_releases_completed_pair_stage_owners(monkeypatch):
    from opendde.model import opendde

    references = {}
    mesh = SimpleNamespace(group_2d=object())
    z_init = torch.ones(2, 1, 1)
    z_spec = SimpleNamespace()

    def synchronize(action, *, description, **_kwargs):
        result = action()
        if description == "Fold-CP trunk initialization":
            references["initial_zero"] = weakref.ref(result[2])
        elif description == "Fold-CP trunk cycle update":
            references["cycled"] = weakref.ref(result)
        elif description == "Fold-CP template residual update":
            references["template_residual"] = weakref.ref(result)
        return result

    def template_forward(**_kwargs):
        update = torch.full_like(z_init, 2)
        references["template_update"] = weakref.ref(update)
        return update, z_spec

    def msa_forward(*, z_local, **_kwargs):
        gc.collect()
        assert references["initial_zero"]() is None
        assert references["cycled"]() is None
        return z_local + 3, z_spec

    def pairformer_forward(_stack, s, z_local, _mesh, **_kwargs):
        gc.collect()
        assert references["template_residual"]() is None
        assert references["template_update"]() is None
        return s + 1, z_local + 1, z_spec

    model = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        linear_no_bias_zinit1=None,
        linear_no_bias_zinit2=None,
        relative_position_encoding=None,
        linear_no_bias_token_bond=None,
        layernorm_z_cycle=None,
        linear_no_bias_z_cycle=None,
        template_embedder=SimpleNamespace(
            n_blocks=1,
            forward_foldcp_local_pair=template_forward,
        ),
        msa_module=SimpleNamespace(forward_foldcp_local_pair=msa_forward),
        configs=SimpleNamespace(
            triangle_multiplicative="torch",
            triangle_attention="torch",
        ),
        linear_no_bias_s=lambda value: value,
        layernorm_s=lambda value: value,
        pairformer_stack=object(),
    )
    monkeypatch.setattr(
        opendde,
        "build_trunk_z_init_local",
        lambda **_kwargs: (z_init, z_spec),
    )
    monkeypatch.setattr(
        opendde,
        "apply_trunk_z_cycle_local",
        lambda **_kwargs: torch.full_like(z_init, 4),
    )
    monkeypatch.setattr(
        opendde,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        opendde,
        "distributed_pairformer_stack_single_bridge_update",
        pairformer_forward,
    )

    result = opendde.OpenDDE._maybe_get_pairformer_output_foldcp_local(
        model,
        input_feature_dict={"relp": torch.zeros(1), "token_bonds": torch.zeros(1)},
        N_cycle=1,
        s_inputs=torch.zeros(2, 1),
        s_init=torch.zeros(2, 1),
    )

    assert result is not None


def test_foldcp_template_releases_completed_output_before_next_template(monkeypatch):
    from opendde.model.modules import pairformer

    references = {}
    template_calls = 0
    mesh = SimpleNamespace(group_2d=object())
    z_spec = SimpleNamespace(local_shape=(2, 2, 1))

    def synchronize(action, *, description, **_kwargs):
        result = action()
        if description == "Fold-CP template initialization":
            references["initial_zero"] = weakref.ref(result[3])
        elif description == "Fold-CP template output projection":
            gc.collect()
            assert references["last_template"]() is None
        return result

    def single_template_forward(**_kwargs):
        nonlocal template_calls
        if template_calls:
            gc.collect()
            assert references["initial_zero"]() is None
            assert references["last_template"]() is None
        value = torch.full((2, 2, 1), template_calls + 1.0)
        references["last_template"] = weakref.ref(value)
        template_calls += 1
        return value

    module = SimpleNamespace(
        n_blocks=1,
        c=1,
        _local_valid_pair_mask=lambda **_kwargs: torch.ones(2, 2),
        _local_pair_mask_from_asym_id=lambda **_kwargs: torch.ones(2, 2),
        layernorm_z=lambda value: value,
        single_template_forward_foldcp_local=single_template_forward,
        relu=lambda value: value,
        linear_no_bias_u=lambda value: value,
        _linear_no_bias_source_stride_tile=lambda _linear, value, *_args, **_kwargs: (
            value
        ),
    )
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result, returned_spec = pairformer.TemplateEmbedder.forward_foldcp_local_pair(
        module,
        input_feature_dict={
            "template_aatype": torch.zeros(2, 1),
            "asym_id": torch.zeros(2, dtype=torch.long),
        },
        z_local=torch.zeros(2, 2, 1),
        z_spec=z_spec,
        mesh=mesh,
    )

    assert result.shape == (2, 2, 1)
    assert returned_spec is z_spec
    assert template_calls == 2


def test_template_pair_preparation_failure_stops_before_pairformer(monkeypatch):
    from opendde.model.modules import pairformer

    pairformer_calls = []
    module = SimpleNamespace(
        pairformer_stack=SimpleNamespace(
            forward_source=lambda *args, **kwargs: pairformer_calls.append("source")
        )
    )
    mesh = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote template feature OOM")
        ),
    )
    monkeypatch.setattr(
        pairformer,
        "distributed_pairformer_stack_pair_update",
        lambda *args, **kwargs: pairformer_calls.append("distributed"),
    )

    with pytest.raises(RuntimeError, match="remote template feature OOM"):
        pairformer.TemplateEmbedder.single_template_forward_foldcp_local(
            module,
            template_id=0,
            input_feature_dict={},
            z_local=torch.zeros(1, 1, 1),
            z_spec=SimpleNamespace(),
            mesh=mesh,
            pair_mask_local=torch.ones(1, 1),
            multichain_mask_local=torch.ones(1, 1),
        )

    assert pairformer_calls == []


def test_msa_sample_preparation_failure_stops_before_blocks(monkeypatch):
    from opendde.model.modules import pairformer

    block_calls = []
    module = SimpleNamespace(
        _prepare_msa_sample=lambda *args, **kwargs: torch.zeros(1),
        _maybe_forward_foldcp_blocks=lambda *args, **kwargs: block_calls.append(
            "blocks"
        ),
    )
    mesh = SimpleNamespace(group_2d=object())
    z_spec = SimpleNamespace(original_shape=(2, 2, 1), pair_dims=(0, 1))
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote MSA sample OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote MSA sample OOM"):
        pairformer.MSAModule.forward_foldcp_local_pair(
            module,
            input_feature_dict={},
            z_local=torch.zeros(1, 1, 1),
            z_spec=z_spec,
            s_inputs=torch.zeros(2, 1),
            pair_mask=None,
            mesh=mesh,
        )

    assert block_calls == []


def test_msa_projection_failure_stops_before_weighted_average(monkeypatch):
    from opendde.model.modules import pairformer

    collective_calls = []
    mesh = SimpleNamespace(group_2d=object())
    z_spec = SimpleNamespace(
        pair_dims=(0, 1),
        original_shape=(2, 2, 1),
        row_range=(0, 1),
        col_range=(0, 1),
    )
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote MSA projection OOM")
        ),
    )
    monkeypatch.setattr(
        pairformer,
        "distributed_msa_pair_weighted_average_with_full_value",
        lambda *args, **kwargs: collective_calls.append("weighted-average"),
    )

    with pytest.raises(RuntimeError, match="remote MSA projection OOM"):
        pairformer.MSAPairWeightedAveraging._maybe_forward_foldcp(
            SimpleNamespace(),
            m=torch.zeros(1, 2, 1),
            z_local=torch.zeros(1, 1, 1),
            z_pair_spec=z_spec,
            mesh=mesh,
        )

    assert collective_calls == []


def test_2d_opm_projection_failure_stops_before_ring(monkeypatch):
    from opendde.model.modules import pairformer

    ring_calls = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(2, 2)),
    )
    module = SimpleNamespace(
        _foldcp_add_opm_to_local_pair_no_grad=lambda *args, **kwargs: ring_calls.append(
            "opm-ring"
        )
    )
    monkeypatch.setattr(pairformer.torch, "is_grad_enabled", lambda: False)
    monkeypatch.setattr(
        pairformer.torch, "are_deterministic_algorithms_enabled", lambda: False
    )
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote 2D OPM projection OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote 2D OPM projection OOM"):
        pairformer.MSABlock._forward_foldcp_local_pair_update(
            module,
            m=torch.zeros(1, 1, 2, 1),
            z_local=torch.zeros(1, 1, 1, 1),
            z_spec=SimpleNamespace(),
            mesh=mesh,
            mask_local=None,
        )

    assert ring_calls == []


def test_atom_window_warmup_allocation_failure_stops_before_p2p(monkeypatch):
    from opendde.model.modules import transformer

    p2p_calls = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(
            numel=3,
            to_coord=lambda rank: (0, rank),
            shifted_rank=lambda coord, axis, shift: (coord[1] + shift) % 3,
        ),
    )
    monkeypatch.setattr(transformer.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        transformer.dist,
        "batch_isend_irecv",
        lambda *args, **kwargs: p2p_calls.append("p2p"),
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote warmup allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote warmup allocation OOM"):
        transformer.AtomAttentionEncoder._warmup_foldcp_atom_window_p2p(
            SimpleNamespace(_foldcp_atom_window_p2p_warmed=False),
            mesh=mesh,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert p2p_calls == []


def test_atom_window_index_preparation_failure_stops_before_gather(monkeypatch):
    from opendde.model.modules import transformer

    gathers = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(numel=3),
    )
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote atom-window index allocation OOM")
        ),
    )
    monkeypatch.setattr(
        transformer.dist,
        "all_gather",
        lambda *args, **kwargs: gathers.append("all_gather"),
    )

    with pytest.raises(RuntimeError, match="remote atom-window index allocation OOM"):
        transformer.AtomAttentionEncoder._project_pair_embedding_in_dense_trunk_from_foldcp_local(
            SimpleNamespace(),
            z_local=torch.zeros(4, 2, 1),
            z_spec=SimpleNamespace(
                local_shape=(4, 2, 1),
                original_shape=(4, 4, 1),
                pair_dims=(0, 1),
            ),
            idx_q=torch.zeros(1, 1),
            idx_k=torch.zeros(1, 1),
            mesh=mesh,
            out=torch.zeros(1, 1, 1, 1),
        )

    assert gathers == []


def test_legacy_atom_window_allocation_failure_stops_before_p2p(monkeypatch):
    from opendde.model.modules import transformer

    p2p_calls = []
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(
            numel=2,
            to_coord=lambda rank: (0, rank),
        ),
    )

    def synchronize(action, *, description, **_kwargs):
        if description == "legacy atom-window transfer allocation":
            raise RuntimeError("remote legacy atom-window allocation OOM")
        return action()

    monkeypatch.setattr(transformer, "run_group_rank_action_synchronized", synchronize)
    monkeypatch.setattr(transformer.dist, "get_rank", lambda _group: 0)
    monkeypatch.setattr(
        transformer.dist,
        "all_gather",
        lambda outputs, source, **_kwargs: [output.copy_(source) for output in outputs],
    )
    monkeypatch.setattr(
        transformer.dist,
        "send",
        lambda *args, **kwargs: p2p_calls.append("send"),
    )
    monkeypatch.setattr(
        transformer.dist,
        "recv",
        lambda *args, **kwargs: p2p_calls.append("recv"),
    )

    with pytest.raises(RuntimeError, match="remote legacy atom-window allocation OOM"):
        transformer.AtomAttentionEncoder._project_pair_embedding_in_dense_trunk_from_foldcp_local(
            SimpleNamespace(),
            z_local=torch.zeros(4, 2, 1),
            z_spec=SimpleNamespace(
                local_shape=(4, 2, 1),
                original_shape=(4, 4, 1),
                pair_dims=(0, 1),
            ),
            idx_q=torch.zeros(1, 1),
            idx_k=torch.zeros(1, 1),
            mesh=mesh,
            out=torch.zeros(1, 1, 1, 1),
        )

    assert p2p_calls == []


def test_atom_window_local_feature_failure_stops_before_index_gather(monkeypatch):
    from opendde.model.modules import transformer

    projection_calls = []
    module = SimpleNamespace(
        _project_pair_embedding_in_dense_trunk_from_foldcp_local=lambda *args, **kwargs: (
            projection_calls.append("index-gather")
        )
    )
    mesh = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        transformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote atom feature OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote atom feature OOM"):
        transformer.AtomAttentionEncoder.prepare_cache_foldcp_window(
            module,
            ref_pos=torch.zeros(1, 3),
            ref_charge=torch.zeros(1),
            ref_mask=torch.ones(1),
            ref_element=torch.zeros(1, 128),
            ref_atom_name_chars=torch.zeros(1, 4, 64),
            atom_to_token_idx=torch.zeros(1, dtype=torch.long),
            d_lm=torch.zeros(1, 1, 1, 3),
            v_lm=torch.zeros(1, 1, 1),
            pad_info={},
            mesh=mesh,
            r_l=True,
            z=torch.zeros(1, 1, 1),
            z_spec=SimpleNamespace(),
        )

    assert projection_calls == []


def test_diffusion_pair_row_slab_uses_hardened_ring_helper(monkeypatch):
    from opendde.model.modules import diffusion

    source = torch.zeros(1, 2, 1)
    expected = torch.ones(1, 3, 1)
    calls = []
    monkeypatch.setattr(
        diffusion,
        "collect_msa_pair_row_slab",
        lambda tensor, mesh, original_tokens: (
            calls.append((tensor, mesh, original_tokens)) or expected
        ),
    )
    mesh = SimpleNamespace()

    result = diffusion.DiffusionConditioning._collect_pair_row_slab(
        source,
        mesh,
        n_token=3,
    )

    assert result is expected
    assert calls == [(source, mesh, 3)]


def test_confidence_projection_failure_stops_before_public_gather(monkeypatch):
    from opendde.distributed.foldcp import confidence

    gather_calls = []
    mesh = SimpleNamespace(group_2d=object())
    spec = SimpleNamespace(pair_dims=(0, 1))
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote confidence projection OOM")
        ),
    )
    monkeypatch.setattr(
        confidence,
        "gather_pair_tensor_like",
        lambda *args, **kwargs: gather_calls.append("gather"),
    )

    with pytest.raises(RuntimeError, match="remote confidence projection OOM"):
        confidence.distributed_confidence_pair_logits(
            z_pair_local=torch.zeros(1, 1, 1),
            z_pair_spec=spec,
            mesh=mesh,
            pae_ln=SimpleNamespace(),
            pae_linear=SimpleNamespace(),
            pde_ln=SimpleNamespace(),
            pde_linear=SimpleNamespace(),
            compute_pae=True,
            compute_pde=False,
            gather_to_rank0_only=False,
        )

    assert gather_calls == []


def test_confidence_row_slab_collectives_run_outside_rank_action(monkeypatch):
    from opendde.distributed.foldcp import confidence

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        layout=SimpleNamespace(shape=(1, 3)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(6, 6, 1),
        padded_shape=(6, 6, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 3),
        mesh_coord=(0, 0),
    )
    source = torch.zeros(2, 2, 1)
    collected = torch.ones(2, 6, 1)
    inside_action = False
    collect_calls = []

    def _run_action(action, **_kwargs):
        nonlocal inside_action
        assert not inside_action
        inside_action = True
        try:
            return action() if action is not None else None
        finally:
            inside_action = False

    def _collect(tensor, selected_mesh):
        assert not inside_action
        collect_calls.append((tensor, selected_mesh))
        return collected

    def _project(**kwargs):
        assert inside_action
        assert kwargs["z_row_slab"] is collected
        assert kwargs["stream_projection"] is False
        return source

    monkeypatch.setattr(
        confidence,
        "_confidence_should_stream_projection",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(confidence, "run_group_rank_action_synchronized", _run_action)
    monkeypatch.setattr(confidence, "_collect_pair_row_slab", _collect)
    monkeypatch.setattr(confidence, "_confidence_pair_logits_local_rowslab", _project)

    result = confidence._confidence_pair_logits_local_rowslab_synchronized(
        z_pair_local=source,
        z_pair_spec=spec,
        mesh=mesh,
        layer_norm=torch.nn.Identity(),
        linear=torch.nn.Identity(),
    )

    assert result is source
    assert collect_calls == [(source, mesh)]


def test_confidence_offload_failure_stops_before_transpose(monkeypatch):
    from opendde.distributed.foldcp import confidence

    transpose_calls = []
    mesh = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        confidence,
        "_confidence_should_offload_transpose_source",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote confidence CPU offload OOM")
        ),
    )
    monkeypatch.setattr(
        confidence,
        "_transpose_pair_tile_collective",
        lambda *args, **kwargs: transpose_calls.append("transpose"),
    )

    with pytest.raises(RuntimeError, match="remote confidence CPU offload OOM"):
        confidence.distributed_confidence_pair_logits(
            z_pair_local=torch.zeros(1, 1, 1),
            z_pair_spec=SimpleNamespace(),
            mesh=mesh,
            pae_ln=SimpleNamespace(),
            pae_linear=SimpleNamespace(),
            pde_ln=SimpleNamespace(),
            pde_linear=SimpleNamespace(),
            compute_pae=False,
            compute_pde=True,
        )

    assert transpose_calls == []


def test_confidence_post_offload_cleanup_failure_stops_before_transpose(monkeypatch):
    from opendde.distributed.foldcp import confidence

    transpose_calls = []
    mesh = SimpleNamespace(group_2d=object())
    monkeypatch.setattr(
        confidence,
        "_confidence_should_offload_transpose_source",
        lambda *args, **kwargs: True,
    )

    def synchronize(action, *, description, **_kwargs):
        if description == "confidence post-offload allocator cleanup":
            raise RuntimeError("remote confidence allocator failure")
        return action()

    monkeypatch.setattr(
        confidence,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        confidence,
        "_transpose_pair_tile_collective",
        lambda *args, **kwargs: transpose_calls.append("transpose"),
    )

    with pytest.raises(RuntimeError, match="remote confidence allocator failure"):
        confidence.distributed_confidence_pair_logits(
            z_pair_local=torch.zeros(1, 1, 1),
            z_pair_spec=SimpleNamespace(),
            mesh=mesh,
            pae_ln=SimpleNamespace(),
            pae_linear=SimpleNamespace(),
            pde_ln=SimpleNamespace(),
            pde_linear=SimpleNamespace(),
            compute_pae=False,
            compute_pde=True,
        )

    assert transpose_calls == []


def test_single_bridge_input_failure_stops_before_pairformer_block(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    block_calls = []
    tensor = torch.zeros(1, 1, 1)
    spec = SimpleNamespace()
    mesh = SimpleNamespace(group_2d=object())
    stack = SimpleNamespace(blocks=[SimpleNamespace(c_s=1)])
    monkeypatch.setattr(
        real_pairformer,
        "shard_pair_tensor",
        lambda *args, **kwargs: (tensor, spec),
    )

    def synchronize(action, *, description, **_kwargs):
        if description == "Fold-CP single-bridge input preparation":
            raise RuntimeError("remote single-bridge input OOM")
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_pairformer_block_pair_ops",
        lambda *args, **kwargs: block_calls.append("block"),
    )

    with pytest.raises(RuntimeError, match="remote single-bridge input OOM"):
        real_pairformer.distributed_pairformer_stack_single_bridge_update(
            stack,
            s=torch.zeros(1, 1),
            z=tensor,
            mesh=mesh,
        )

    assert block_calls == []


def test_pair_only_input_sharding_is_group_synchronized(monkeypatch):
    from opendde.model.modules import pairformer

    group = object()
    mesh = SimpleNamespace(group_2d=group)
    shard_calls = []

    def synchronize(action, *, group, description):
        assert group is mesh.group_2d
        assert description == "Fold-CP pair-only input preparation"
        raise RuntimeError("remote pair-only input OOM")

    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        pairformer,
        "shard_pair_tensor",
        lambda *args, **kwargs: shard_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="remote pair-only input OOM"):
        pairformer._prepare_foldcp_pair_only_inputs(
            torch.zeros(1, 2, 2, 1),
            torch.ones(2, 2),
            mesh,
        )

    assert shard_calls == []


@pytest.mark.parametrize("entrypoint", ["block", "stack"])
def test_pair_only_dispatch_stops_before_cp_compute_on_input_failure(
    monkeypatch,
    entrypoint,
):
    from opendde.model.modules import pairformer

    mesh = SimpleNamespace(group_2d=object())
    compute_calls = []
    monkeypatch.setenv("OPENDDE_FOLDCP_MODE", "distributed")
    monkeypatch.setattr(pairformer.dist, "is_available", lambda: True)
    monkeypatch.setattr(pairformer.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pairformer.FoldCPConfig,
        "from_environment",
        lambda: object(),
    )
    monkeypatch.setattr(
        pairformer.FoldCPProcessMesh,
        "create",
        lambda _config: mesh,
    )
    monkeypatch.setattr(
        pairformer,
        "_prepare_foldcp_pair_only_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote pair-only sharding OOM")
        ),
    )
    monkeypatch.setattr(
        pairformer,
        "distributed_pairformer_block_pair_update",
        lambda *args, **kwargs: compute_calls.append("block"),
    )
    monkeypatch.setattr(
        pairformer,
        "distributed_pairformer_stack_pair_update",
        lambda *args, **kwargs: compute_calls.append("stack"),
    )

    z = torch.zeros(1, 2, 2, 1)
    pair_mask = torch.ones(2, 2)
    if entrypoint == "block":
        module = SimpleNamespace(c_s=0)
        call = pairformer.PairformerBlock._maybe_forward_foldcp_pair_only
        args = (module, None, z, pair_mask)
    else:
        module = SimpleNamespace(blocks=[SimpleNamespace(c_s=0)])
        call = pairformer.PairformerStack._maybe_forward_foldcp_pair_only
        args = (module, None, z, pair_mask, None)

    with pytest.raises(RuntimeError, match="remote pair-only sharding OOM"):
        call(*args)

    assert compute_calls == []


def test_msa_block_dispatch_stops_before_cp_compute_on_input_failure(monkeypatch):
    from opendde.model.modules import pairformer

    mesh = SimpleNamespace(group_2d=object())
    compute_calls = []
    module = SimpleNamespace(
        _maybe_foldcp_mesh=lambda: mesh,
        forward_foldcp_local=lambda **_kwargs: compute_calls.append("compute"),
    )
    monkeypatch.setattr(
        pairformer,
        "_prepare_foldcp_pair_only_inputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote MSA pair input OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote MSA pair input OOM"):
        pairformer.MSABlock.forward(
            module,
            m=torch.zeros(1, 1, 2, 1),
            z=torch.zeros(1, 2, 2, 1),
            pair_mask=torch.ones(2, 2),
        )

    assert compute_calls == []


def test_msa_local_pair_finalization_is_group_synchronized(monkeypatch):
    from opendde.model.modules import pairformer

    mesh = SimpleNamespace(group_2d=object())

    def synchronize(action, *, group, description):
        assert group is mesh.group_2d
        if description == "Fold-CP MSA block input preparation":
            return action()
        assert description == "Fold-CP MSA local pair finalization"
        raise RuntimeError("remote MSA final contiguous OOM")

    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    module = SimpleNamespace(blocks=[])

    with pytest.raises(RuntimeError, match="remote MSA final contiguous OOM"):
        pairformer.MSAModule._maybe_forward_foldcp_blocks(
            module,
            msa_sample=torch.zeros(1, 1, 2, 1),
            z=torch.zeros(1, 2, 2, 1),
            pair_mask=None,
            z_spec=SimpleNamespace(),
            mesh=mesh,
            z_is_local=True,
            return_local_pair=True,
        )


def test_empty_msa_local_pair_finalization_is_group_synchronized(monkeypatch):
    from opendde.model.modules import pairformer

    mesh = SimpleNamespace(group_2d=object())
    descriptions = []

    def synchronize(action, *, group, description):
        assert group is mesh.group_2d
        descriptions.append(description)
        if description == "Fold-CP MSA sample preparation":
            return None
        assert description == "Fold-CP empty MSA local pair finalization"
        raise RuntimeError("remote empty MSA contiguous OOM")

    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    module = SimpleNamespace(_prepare_msa_sample=lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="remote empty MSA contiguous OOM"):
        pairformer.MSAModule.forward_foldcp_local_pair(
            module,
            input_feature_dict={},
            z_local=torch.zeros(1, 2, 2, 1),
            z_spec=SimpleNamespace(original_shape=(1, 2, 2, 1), pair_dims=(1, 2)),
            s_inputs=torch.zeros(1, 2, 1),
            pair_mask=None,
            mesh=mesh,
        )

    assert descriptions == [
        "Fold-CP MSA sample preparation",
        "Fold-CP empty MSA local pair finalization",
    ]


def test_replicated_template_resharding_is_group_synchronized(monkeypatch):
    from opendde.model.modules import pairformer

    mesh = SimpleNamespace(group_2d=object())
    z_spec = SimpleNamespace(pair_dims=(0, 1))
    shard_calls = []

    def synchronize(action, *, group, description):
        assert group is mesh.group_2d
        assert description == "replicated Fold-CP template pair resharding"
        raise RuntimeError("remote template reshard OOM")

    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        pairformer,
        "shard_pair_tensor",
        lambda *args, **kwargs: shard_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="remote template reshard OOM"):
        pairformer._reshard_replicated_template_pair(
            torch.zeros(2, 2, 1),
            z_spec,
            mesh,
        )

    assert shard_calls == []


def test_replicated_template_mask_spec_is_group_synchronized(monkeypatch):
    from opendde.model.modules import pairformer

    mesh = SimpleNamespace(group_2d=object())
    z_spec = SimpleNamespace(original_shape=(1, 5, 5, 128), pair_dims=(1, 2))
    make_calls = []

    def synchronize(action, *, group, description):
        assert group is mesh.group_2d
        assert description == "replicated Fold-CP template mask-spec preparation"
        raise RuntimeError("remote template mask-spec failure")

    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        pairformer,
        "make_pair_shard_spec",
        lambda *args, **kwargs: make_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="remote template mask-spec failure"):
        pairformer._prepare_replicated_template_mask_spec(z_spec, mesh)

    assert make_calls == []


def test_trimul_b_offload_failure_is_group_synchronized(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote triangle B host OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote triangle B host OOM"):
        real_pairformer._one_by_p_offload_trimul_b_synchronized(
            torch.zeros(1),
            SimpleNamespace(group_2d=object()),
        )


def test_streamed_source_trimul_drains_remaining_gathers_after_local_oom(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    ring_calls = []
    synchronized_stages = []

    def gather(tensor, _mesh, dim, length=None):
        ring_calls.append((tuple(tensor.shape), dim, length))
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_should_preallocate",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_should_materialize_owned_rows",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 2), (2, 4)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer.torch,
        "matmul",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local source TriMul workspace OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="local source TriMul workspace OOM"):
        real_pairformer._distributed_triangle_multiplication_source_matmul(
            torch.zeros(1, 4, 2, 1),
            torch.zeros(1, 4, 2, 1),
            mesh,
            real_pairformer.TriangleMultiplicationDirection.OUTGOING,
            spec,
        )

    # One gather reconstructs A, then both scheduled B chunks are drained even
    # though the first local GEMM failed.
    assert len(ring_calls) == 3
    assert synchronized_stages[-1] == "Pairformer streamed TriMul completion"


def test_owned_row_source_trimul_propagates_local_compute_oom(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    synchronized_stages = []

    def gather(tensor, _mesh, dim, length=None):
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_should_preallocate",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 2), (2, 4)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer.torch,
        "matmul",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local owned-row TriMul workspace OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="local owned-row TriMul workspace OOM"):
        real_pairformer._distributed_triangle_multiplication_source_matmul(
            torch.zeros(1, 4, 2, 1),
            None,
            mesh,
            real_pairformer.TriangleMultiplicationDirection.OUTGOING,
            spec,
            b_owned_rows=torch.zeros(1, 2, 4, 1),
        )

    assert synchronized_stages[-1] == "Pairformer source TriMul completion"


def test_owned_row_source_trimul_stops_after_remote_output_oom(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    matmul_calls = []

    def gather(tensor, _mesh, dim, length=None):
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        if description == "Pairformer source TriMul output allocation":
            raise RuntimeError("remote source TriMul output OOM")
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_should_preallocate",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer.torch,
        "matmul",
        lambda *args, **kwargs: matmul_calls.append("matmul"),
    )

    with pytest.raises(RuntimeError, match="remote source TriMul output OOM"):
        real_pairformer._distributed_triangle_multiplication_source_matmul(
            torch.zeros(1, 4, 2, 1),
            None,
            mesh,
            real_pairformer.TriangleMultiplicationDirection.OUTGOING,
            spec,
            b_owned_rows=torch.zeros(1, 2, 4, 1),
        )

    assert matmul_calls == []


def test_incoming_source_trimul_propagates_local_chunk_oom(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    synchronized_stages = []

    def gather(tensor, _mesh, dim, length=None):
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_should_preallocate",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 2), (2, 4)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_local_source_column_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local incoming source chunk OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="local incoming source chunk OOM"):
        real_pairformer._distributed_triangle_multiplication_source_matmul(
            torch.zeros(1, 4, 2, 1),
            torch.zeros(1, 4, 2, 1),
            mesh,
            real_pairformer.TriangleMultiplicationDirection.INCOMING,
            spec,
        )

    assert synchronized_stages[-1] == "Pairformer source TriMul completion"


def test_source_trimul_output_drains_col_gathers_after_local_oom(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_col=object(),
        layout=SimpleNamespace(shape=(2, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(2, 2),
        mesh_coord=(0, 0),
    )
    col_gathers = []
    synchronized_stages = []

    def col_gather(tensor, _mesh, dim, length=None):
        col_gathers.append((tuple(tensor.shape), dim, length))
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    class FailingLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, 1))
            self.bias = None

        def forward(self, _x):
            raise RuntimeError("local source output projection OOM")

    module = SimpleNamespace(
        c_z=1,
        layer_norm_out=torch.nn.Identity(),
        layer_norm_in=torch.nn.Identity(),
        linear_z=FailingLinear(),
        linear_g=torch.nn.Linear(1, 1, bias=True),
    )
    monkeypatch.setattr(real_pairformer, "_ring_gather_by_col", col_gather)
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 1), (1, 2)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="local source output projection OOM"):
        real_pairformer._triangle_multiplication_output_norm_gate_source_slab(
            module,
            torch.zeros(1, 2, 2, 1),
            torch.zeros(1, 2, 2, 1),
            mesh,
            spec,
            source_unbatched=True,
        )

    assert len(col_gathers) == 4
    assert synchronized_stages[-1] == "Pairformer source TriMul output completion"


def test_source_trimul_output_linear_fusion_respects_deterministic_mode():
    from opendde.distributed.foldcp import real_pairformer

    first = torch.nn.Linear(4, 4, bias=False)
    second = torch.nn.Linear(4, 4, bias=False)
    first_x = torch.zeros(3, 5, 4)
    second_x = torch.zeros_like(first_x)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        assert real_pairformer._pair_tile_linears_are_batch_compatible(
            first,
            second,
            first_x,
            second_x,
        )
        torch.use_deterministic_algorithms(True)
        assert not real_pairformer._pair_tile_linears_are_batch_compatible(
            first,
            second,
            first_x,
            second_x,
        )
    finally:
        torch.use_deterministic_algorithms(previous)


def test_source_trimul_output_padding_row_still_gathers(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_col=object(),
        layout=SimpleNamespace(shape=(2, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 6, 4, 1),
        pair_dims=(1, 2),
        row_range=(4, 6),
        col_range=(0, 2),
        mesh_shape=(2, 2),
        mesh_coord=(1, 0),
    )
    col_gathers = []
    synchronized_stages = []

    def col_gather(tensor, _mesh, dim, length=None):
        col_gathers.append((tuple(tensor.shape), dim, length))
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    module = SimpleNamespace(
        c_z=1,
        layer_norm_out=torch.nn.Identity(),
        layer_norm_in=torch.nn.Identity(),
        linear_z=torch.nn.Identity(),
        linear_g=torch.nn.Identity(),
    )
    monkeypatch.setattr(real_pairformer, "_ring_gather_by_col", col_gather)
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 1), (1, 2)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = real_pairformer._triangle_multiplication_output_norm_gate_source_slab(
        module,
        torch.zeros(1, 2, 2, 1),
        torch.zeros(1, 2, 2, 1),
        mesh,
        spec,
        source_unbatched=True,
    )

    assert torch.equal(result, torch.zeros_like(result))
    assert len(col_gathers) == 4
    assert synchronized_stages[-1] == "Pairformer source TriMul output completion"


@pytest.mark.parametrize("canonical_batch", [False, True])
def test_starting_triangle_attention_drains_row_gathers_after_local_oom(
    monkeypatch,
    canonical_batch,
):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    row_gathers = []
    synchronized_stages = []

    def row_gather(tensor, _mesh, dim, length=None):
        row_gathers.append((tuple(tensor.shape), dim, length))
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    attention = SimpleNamespace(
        layer_norm=torch.nn.Identity(),
        linear=torch.nn.Linear(1, 1, bias=False),
        mha=SimpleNamespace(),
        inf=1e9,
    )
    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", row_gather)
    monkeypatch.setattr(
        real_pairformer,
        "_starting_triangle_bias_full_key_from_source_slab",
        lambda *_args, **_kwargs: torch.zeros(1, 2, 4),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triatt_projection_source_grid_launch",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triatt_projection_source_grid_for_callsite",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triatt_exact_source_launch",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triatt_attention_row_chunk_size",
        lambda *_args, **_kwargs: 1 if canonical_batch else 2,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triatt_canonical_batch_enabled",
        lambda: canonical_batch,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_prep_triangle_attention_qkv_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("local triangle-attention QKV OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_prep_triangle_attention_qkv_source_chunk_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("local triangle-attention QKV OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_prep_triangle_attention_qkv_canonical_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("local triangle-attention QKV OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with (
        torch.no_grad(),
        pytest.raises(RuntimeError, match="local triangle-attention QKV OOM"),
    ):
        real_pairformer._distributed_triangle_attention_starting_update(
            attention,
            torch.zeros(4, 2, 1, dtype=torch.bfloat16),
            mesh,
            z_spec=spec,
            chunk_size=1 if canonical_batch else None,
        )

    assert len(row_gathers) == 4
    assert synchronized_stages[-1] == "starting triangle-attention completion"


def test_starting_triangle_attention_remote_preparation_failure_stops_before_bias(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    bias_calls = []

    def synchronize(action, *, description, **_kwargs):
        if description == "starting triangle-attention input preparation":
            raise RuntimeError("remote starting-attention LayerNorm OOM")
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_starting_triangle_bias_full_key_from_source_slab",
        lambda *_args, **_kwargs: bias_calls.append("bias"),
    )
    attention = SimpleNamespace(layer_norm=torch.nn.Identity())

    with pytest.raises(RuntimeError, match="remote starting-attention LayerNorm OOM"):
        real_pairformer._distributed_triangle_attention_starting_update(
            attention,
            torch.zeros(4, 2, 1),
            mesh,
            z_spec=spec,
        )

    assert bias_calls == []


def test_legacy_starting_triangle_attention_remote_failure_stops_before_row_gather(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(2, 2)),
    )
    row_gathers = []

    def synchronize(action, *, description, **_kwargs):
        if description == "legacy starting triangle-attention input preparation":
            raise RuntimeError("remote legacy starting-attention LayerNorm OOM")
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_by_row",
        lambda *_args, **_kwargs: row_gathers.append("row gather"),
    )
    attention = SimpleNamespace(layer_norm=torch.nn.Identity())

    with pytest.raises(
        RuntimeError,
        match="remote legacy starting-attention LayerNorm OOM",
    ):
        real_pairformer._distributed_triangle_attention_starting_update(
            attention,
            torch.zeros(2, 2, 1),
            mesh,
        )

    assert row_gathers == []


def test_replicated_trimul_remote_mask_spec_failure_stops_before_mask_gather(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    mask_gathers = []

    def synchronize(action, *, description, **_kwargs):
        if description == "replicated TriMul mask-spec preparation":
            raise RuntimeError("remote replicated TriMul mask-spec OOM")
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "gather_pair_tensor_like",
        lambda *_args, **_kwargs: torch.zeros(4, 4, 1),
    )
    monkeypatch.setattr(
        real_pairformer,
        "gather_pair_tensor",
        lambda *_args, **_kwargs: mask_gathers.append("mask gather"),
    )

    with pytest.raises(RuntimeError, match="remote replicated TriMul mask-spec OOM"):
        real_pairformer._replicated_serial_triangle_multiplication_update(
            object(),
            torch.zeros(4, 2, 1),
            mesh,
            torch.ones(4, 2),
            spec,
        )

    assert mask_gathers == []


def test_replicated_trimul_local_compute_failure_is_synchronized(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    synchronized_stages = []

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    class FailingTriMul:
        def __call__(self, *_args, **_kwargs):
            raise RuntimeError("local replicated TriMul compute OOM")

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "gather_pair_tensor_like",
        lambda *_args, **_kwargs: torch.zeros(4, 4, 1),
    )

    with pytest.raises(RuntimeError, match="local replicated TriMul compute OOM"):
        real_pairformer._replicated_serial_triangle_multiplication_update(
            FailingTriMul(),
            torch.zeros(4, 2, 1),
            mesh,
            None,
            spec,
        )

    assert synchronized_stages == ["replicated TriMul local computation"]


def test_legacy_ending_triangle_attention_remote_completion_failure_propagates(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(2, 2)),
        ring_comm=lambda: SimpleNamespace(comm_2d_trans=object()),
    )
    exchanges = []

    def exchange(source, *, description, prepare, **_kwargs):
        exchanges.append(description)
        return prepare(source).contiguous()

    def synchronize(action, *, description, **_kwargs):
        if description == "legacy ending triangle-attention completion":
            raise RuntimeError("remote legacy ending residual OOM")
        return action()

    monkeypatch.setattr(real_pairformer, "exchange_tensor_synchronized", exchange)
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_triangle_attention_starting_update",
        lambda _attention, z_local, *_args, **_kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="remote legacy ending residual OOM"):
        real_pairformer.distributed_triangle_attention_update(
            SimpleNamespace(starting=False),
            torch.zeros(2, 2, 1),
            mesh,
            residual_local=torch.zeros(2, 2, 1),
        )

    assert exchanges == [
        "ending triangle-attention input transpose",
        "ending triangle-attention output transpose",
    ]


def test_reference_triangle_attention_remote_compute_failure_stops_before_ring(
    monkeypatch,
):
    from opendde.distributed.foldcp import triangle_attention

    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
        comm_row=object(),
    )
    exchanges = []

    def synchronize(action, *, description, **_kwargs):
        if description == "reference starting triangle attention step 0 computation":
            raise RuntimeError("remote reference triangle attention OOM")
        return action()

    monkeypatch.setattr(
        triangle_attention,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        triangle_attention,
        "exchange_tensor_synchronized",
        lambda *_args, **kwargs: exchanges.append(kwargs["description"]),
    )

    with pytest.raises(RuntimeError, match="remote reference triangle attention OOM"):
        triangle_attention.distributed_triangle_attention_starting(
            torch.zeros(1, 2, 1),
            ring,
        )

    assert exchanges == []


def test_reference_ring_attention_remote_compute_failure_stops_before_exchange(
    monkeypatch,
):
    from opendde.distributed.foldcp import triangle_attention

    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
        comm_row=object(),
    )
    exchanges = []

    def synchronize(action, *, description, **_kwargs):
        if description == "reference ring attention step 0 computation":
            raise RuntimeError("remote reference ring attention OOM")
        return action()

    monkeypatch.setattr(
        triangle_attention,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        triangle_attention,
        "exchange_tensor_synchronized",
        lambda *_args, **kwargs: exchanges.append(kwargs["description"]),
    )

    with pytest.raises(RuntimeError, match="remote reference ring attention OOM"):
        triangle_attention.distributed_ring_attention(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 1),
            None,
            ring,
        )

    assert exchanges == []


def test_reference_triangle_attention_single_rank_matches_serial():
    from opendde.distributed.foldcp import triangle_attention

    torch.manual_seed(7)
    z = torch.randn(1, 3, 3, 2)
    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 1), numel=1),
    )

    distributed = triangle_attention.distributed_triangle_attention_starting(z, ring)
    serial = triangle_attention.serial_triangle_attention_starting(z)

    torch.testing.assert_close(distributed, serial)


def test_reference_ring_attention_single_rank_matches_serial():
    from opendde.distributed.foldcp import triangle_attention

    torch.manual_seed(11)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 5, 4)
    v = torch.randn(2, 5, 6)
    bias = torch.randn(2, 3, 5)
    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 1), numel=1),
    )

    distributed = triangle_attention.distributed_ring_attention(q, k, v, bias, ring)
    serial = triangle_attention.serial_attention(q, k, v, bias)

    torch.testing.assert_close(distributed, serial)


def test_reference_ending_triangle_attention_uses_safe_transpose_exchanges(
    monkeypatch,
):
    from opendde.distributed.foldcp import triangle_attention

    ring = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2), numel=2),
        comm_2d_trans=object(),
    )
    descriptions = []

    def exchange(source, *, description, prepare, **_kwargs):
        descriptions.append(description)
        return prepare(source).contiguous()

    monkeypatch.setattr(triangle_attention, "exchange_tensor_synchronized", exchange)
    monkeypatch.setattr(
        triangle_attention,
        "distributed_triangle_attention_starting",
        lambda z_local, _ring: z_local,
    )
    z = torch.arange(12).reshape(1, 3, 2, 2)

    result = triangle_attention.distributed_triangle_attention_ending(z, ring)

    assert torch.equal(result, z)
    assert descriptions == [
        "reference ending triangle attention input transpose",
        "reference ending triangle attention output transpose",
    ]


@pytest.mark.parametrize(
    "failure_stage",
    [
        "ending triangle-attention bias preparation",
        "ending triangle-attention local update completion",
    ],
)
def test_ending_triangle_attention_local_failures_are_synchronized(
    monkeypatch,
    failure_stage,
):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    transposed_spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 4),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    row_gathers = []

    def synchronize(action, *, description, **_kwargs):
        if description == failure_stage:
            raise RuntimeError(f"remote {failure_stage} OOM")
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_transpose_pair_shard_spec_for_local_attention",
        lambda _spec: transposed_spec,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_local_attention_mesh",
        lambda _mesh: SimpleNamespace(layout=SimpleNamespace(shape=(1, 1))),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_project_one_by_p_ending_triangle_bias_local",
        lambda *_args, **_kwargs: torch.zeros(1, 2, 4),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_by_row",
        lambda tensor, *_args, **_kwargs: row_gathers.append("gather") or tensor,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_triangle_attention_starting_update",
        lambda _module, tensor, *_args, **_kwargs: tensor,
    )
    attention = SimpleNamespace(layer_norm=torch.nn.Identity())

    with pytest.raises(RuntimeError, match=f"remote {failure_stage} OOM"):
        real_pairformer._one_by_p_triangle_attention_ending_update(
            attention,
            torch.zeros(4, 2, 1),
            mesh,
            None,
            spec,
            None,
        )

    assert row_gathers == ([] if "bias preparation" in failure_stage else ["gather"])


def test_ending_triangle_attention_releases_completed_transpose_chunks(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    transposed_spec = FoldCPPairShardSpec(
        original_shape=(4, 4, 1),
        padded_shape=(4, 4, 1),
        pair_dims=(0, 1),
        row_range=(0, 2),
        col_range=(0, 4),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    prior_refs = []
    transpose_calls = 0

    def transpose_chunk(_z_local, row_start, row_end):
        nonlocal transpose_calls, prior_refs
        if transpose_calls:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        value = torch.zeros(row_end - row_start, 4, 1)
        prior_refs = [weakref.ref(value)]
        transpose_calls += 1
        return value

    def layer_norm(value):
        normalized = value + 1
        prior_refs.append(weakref.ref(normalized))
        return normalized

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_transpose_pair_shard_spec_for_local_attention",
        lambda _spec: transposed_spec,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_local_attention_mesh",
        lambda _mesh: SimpleNamespace(layout=SimpleNamespace(shape=(1, 1))),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_ending_transpose_row_chunk_size",
        lambda **_kwargs: 1,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_ending_transpose_row_chunk",
        transpose_chunk,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_project_one_by_p_ending_triangle_bias_local",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 4),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_by_row",
        lambda tensor, *_args, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_triangle_attention_starting_update",
        lambda _module, tensor, *_args, **_kwargs: tensor,
    )

    result = real_pairformer._one_by_p_triangle_attention_ending_update(
        SimpleNamespace(layer_norm=layer_norm),
        torch.zeros(4, 2, 1),
        mesh,
        None,
        spec,
        None,
    )

    assert result.shape == (4, 2, 1)
    assert transpose_calls == 4


def test_outgoing_b_projection_drains_remaining_gathers_after_local_oom(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_row=group,
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 4),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )
    module = SimpleNamespace(
        c_hidden=1,
        linear_b_g=object(),
        linear_b_p=object(),
        layer_norm_in=object(),
    )
    gather_calls = []
    synchronized_stages = []

    def gather(tensor, _mesh, start, end, length):
        gather_calls.append((start, end, length))
        return tensor.new_zeros(1, end - start, length, 1)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_gather_row_chunk_by_row",
        gather,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 2), (2, 4)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_project_source_launch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local outgoing B projection OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="local outgoing B projection OOM"):
        real_pairformer._triangle_b_projection_source_chunk(
            module,
            torch.zeros(1, 4, 2, 1),
            None,
            mesh,
            real_pairformer.TriangleMultiplicationDirection.OUTGOING,
            spec,
            return_owned_rows=True,
        )

    assert gather_calls == [(0, 2, 4), (2, 4, 4)]
    assert synchronized_stages[-1] == ("Pairformer outgoing B-projection completion")


def test_triangle_a_projection_releases_completed_source_chunks(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    spec = FoldCPPairShardSpec(
        original_shape=(1, 300, 300, 1),
        padded_shape=(1, 300, 300, 1),
        pair_dims=(1, 2),
        row_range=(0, 300),
        col_range=(0, 300),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    prior_ref = None
    calls = 0

    def project(_gate, _projection, source, *_args, **_kwargs):
        nonlocal prior_ref, calls
        if calls:
            gc.collect()
            assert prior_ref is not None and prior_ref() is None
        result = torch.zeros(*source.shape[:-1], 1)
        prior_ref = weakref.ref(result)
        calls += 1
        return result

    monkeypatch.setattr(real_pairformer, "_triangle_project_source_launch", project)

    result = real_pairformer._triangle_a_projection_source_chunks(
        SimpleNamespace(c_hidden=1, linear_a_g=object(), linear_a_p=object()),
        torch.zeros(1, 300, 300, 1),
        None,
        spec,
    )

    assert result.shape == (1, 300, 300, 1)
    assert calls == 2


def test_incoming_b_projection_releases_completed_source_chunks(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_col=group,
        layout=SimpleNamespace(shape=(1, 1)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 4),
        col_range=(0, 4),
        mesh_shape=(1, 1),
        mesh_coord=(0, 0),
    )
    prior_ref = None
    calls = 0

    def project(_gate, _projection, source, *_args, **_kwargs):
        nonlocal prior_ref, calls
        if calls:
            gc.collect()
            assert prior_ref is not None and prior_ref() is None
        result = torch.zeros(*source.shape[:-1], 1)
        prior_ref = weakref.ref(result)
        calls += 1
        return result

    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 2), (2, 4)],
    )
    monkeypatch.setattr(
        real_pairformer, "_ring_gather_by_col", lambda value, *_a, **_k: value
    )
    monkeypatch.setattr(
        real_pairformer, "_trimul_can_projection_source_grid", lambda *_a: True
    )
    monkeypatch.setattr(real_pairformer, "_triangle_project_source_launch", project)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    result = real_pairformer._triangle_b_projection_source_chunk(
        SimpleNamespace(
            c_hidden=1,
            linear_b_g=object(),
            linear_b_p=object(),
            layer_norm_in=object(),
        ),
        torch.zeros(1, 4, 4, 1),
        None,
        mesh,
        real_pairformer.TriangleMultiplicationDirection.INCOMING,
        spec,
    )

    assert result.shape == (1, 4, 4, 1)
    assert calls == 2


def test_incoming_b_projection_padding_rank_still_enters_col_gather(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    group = object()
    mesh = SimpleNamespace(
        group_2d=group,
        group_col=group,
        layout=SimpleNamespace(shape=(2, 1)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 2, 2, 1),
        padded_shape=(1, 4, 2, 1),
        pair_dims=(1, 2),
        row_range=(2, 4),
        col_range=(0, 2),
        mesh_shape=(2, 1),
        mesh_coord=(1, 0),
    )
    module = SimpleNamespace(
        c_hidden=1,
        linear_b_g=object(),
        linear_b_p=object(),
        layer_norm_in=object(),
    )
    col_gathers = []
    synchronized_stages = []

    def gather(tensor, *_args, **_kwargs):
        col_gathers.append("gather")
        return tensor

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_col", gather)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    result = real_pairformer._triangle_b_projection_source_chunk(
        module,
        torch.zeros(1, 2, 2, 1),
        None,
        mesh,
        real_pairformer.TriangleMultiplicationDirection.INCOMING,
        spec,
    )

    assert torch.equal(result, torch.zeros_like(result))
    assert col_gathers == ["gather"]
    assert synchronized_stages[-1] == "Pairformer incoming B-projection completion"


@pytest.mark.parametrize("direction_name", ["OUTGOING", "INCOMING"])
def test_2d_source_trimul_drains_remaining_col_gathers_after_local_oom(
    monkeypatch,
    direction_name,
):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        group_row=object(),
        group_col=object(),
        layout=SimpleNamespace(shape=(2, 2)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(2, 2),
        mesh_coord=(0, 0),
    )
    col_gathers = []
    synchronized_stages = []

    def gather(tensor, _mesh, dim, length=None):
        shape = list(tensor.shape)
        normalized_dim = dim if dim >= 0 else tensor.ndim + dim
        shape[normalized_dim] = int(length)
        return tensor.new_zeros(shape)

    def col_gather(tensor, mesh, dim, length=None):
        col_gathers.append((tuple(tensor.shape), dim, length))
        return gather(tensor, mesh, dim, length)

    def synchronize(action, *, description, **_kwargs):
        synchronized_stages.append(description)
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(real_pairformer, "_ring_gather_by_col", col_gather)
    monkeypatch.setattr(
        real_pairformer,
        "_transpose_source_pair_tile",
        lambda tensor, _mesh: tensor,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_column_chunks",
        lambda _n: [(0, 1), (1, 2)],
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_source_matmul_row_size",
        lambda rows, _n: rows,
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer.torch,
        "matmul",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("local 2D source TriMul OOM")
        ),
    )

    direction = getattr(
        real_pairformer.TriangleMultiplicationDirection,
        direction_name,
    )
    with pytest.raises(RuntimeError, match="local 2D source TriMul OOM"):
        real_pairformer._distributed_triangle_multiplication_source_matmul(
            torch.zeros(1, 2, 2, 1),
            torch.zeros(1, 2, 2, 1),
            mesh,
            direction,
            spec,
        )

    assert len(col_gathers) == 2
    assert synchronized_stages[-1] == (
        f"Pairformer 2D {direction_name.lower()} TriMul completion"
    )


def test_pairformer_block_finalization_failure_stops_block_return(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    tensor = torch.zeros(1, 1, 1)
    block = SimpleNamespace(
        tri_mul_out=object(),
        tri_mul_in=object(),
        tri_att_start=object(),
        tri_att_end=object(),
        pair_transition=object(),
    )
    mesh = SimpleNamespace(layout=SimpleNamespace(shape=(1, 2)), group_2d=object())
    monkeypatch.setattr(
        real_pairformer,
        "distributed_triangle_multiplication_update",
        lambda _module, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "distributed_triangle_attention_update",
        lambda _module, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_triangle_attention_ending_update",
        lambda _module, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda _action, *, description, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("remote pair transition OOM")
        ),
    )

    with (
        torch.no_grad(),
        pytest.raises(RuntimeError, match="remote pair transition OOM"),
    ):
        real_pairformer._distributed_pairformer_block_pair_ops(
            block,
            tensor,
            mesh,
            z_spec=SimpleNamespace(),
        )


def test_triangle_output_failure_is_propagated_before_next_pairformer_op(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    stages = []
    local_pair = torch.zeros(2, 2, 1)
    projected = torch.zeros(1, 2, 2, 1)
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = SimpleNamespace(
        original_shape=(2, 2, 1),
        pair_dims=(-3, -2),
        row_range=(0, 2),
        col_range=(0, 1),
    )
    module = SimpleNamespace(c_hidden=1, _outgoing=True, layer_norm_in=object())

    monkeypatch.setattr(
        real_pairformer,
        "_triangle_layer_norm_source_row_slab",
        lambda *_args, **_kwargs: projected,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_trimul_project_channel_chunk_size",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_a_projection_source_chunks",
        lambda *_args, **_kwargs: projected,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_should_project_b_owned_rows",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_b_projection_source_chunk",
        lambda *_args, **_kwargs: projected,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_triangle_multiplication_source_matmul",
        lambda *_args, **_kwargs: projected,
    )
    inside_synchronized_action = False

    def synchronize(action, *, description, **_kwargs):
        nonlocal inside_synchronized_action
        stages.append(description)
        assert not inside_synchronized_action
        inside_synchronized_action = True
        try:
            return action()
        finally:
            inside_synchronized_action = False

    def source_output(*_args, **_kwargs):
        assert not inside_synchronized_action
        return synchronize(
            lambda: (_ for _ in ()).throw(
                RuntimeError("local triangle output-gate OOM")
            ),
            description="Pairformer source TriMul output completion",
        )

    monkeypatch.setattr(
        real_pairformer,
        "_triangle_multiplication_output_norm_gate_source_slab",
        source_output,
    )

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with pytest.raises(RuntimeError, match="local triangle output-gate OOM"):
        real_pairformer.distributed_triangle_multiplication_update(
            module,
            local_pair,
            mesh,
            residual_local=local_pair,
            z_spec=spec,
        )

    assert stages[-1] == "Pairformer source TriMul output completion"


def test_triangle_input_preparation_failure_stops_before_projection(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    projections = []
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote triangle input LayerNorm OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_a_projection_source_chunks",
        lambda *args, **kwargs: projections.append("projection"),
    )

    with pytest.raises(RuntimeError, match="remote triangle input LayerNorm OOM"):
        real_pairformer.distributed_triangle_multiplication_update(
            SimpleNamespace(),
            torch.zeros(2, 2, 1),
            SimpleNamespace(group_2d=object()),
            z_spec=SimpleNamespace(),
        )

    assert projections == []


def test_triangle_a_projection_failure_stops_before_b_projection(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    b_projections = []
    module = SimpleNamespace(c_hidden=1, _outgoing=True, layer_norm_in=object())
    mesh = SimpleNamespace(group_2d=object())
    spec = SimpleNamespace(
        original_shape=(2, 2, 1),
        pair_dims=(-3, -2),
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_layer_norm_source_row_slab",
        lambda *_args, **_kwargs: torch.zeros(1, 2, 2, 1),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_trimul_project_channel_chunk_size",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_a_projection_source_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("local triangle A projection OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_b_projection_source_chunk",
        lambda *_args, **_kwargs: b_projections.append("projection"),
    )

    with pytest.raises(RuntimeError, match="local triangle A projection OOM"):
        real_pairformer.distributed_triangle_multiplication_update(
            module,
            torch.zeros(2, 2, 1),
            mesh,
            z_spec=spec,
        )

    assert b_projections == []


def test_triangle_b_offload_releases_projection_aliases_before_matmul(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    references = {}
    cleanup_observations = []
    module = SimpleNamespace(c_hidden=1, _outgoing=True, layer_norm_in=object())
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
    )
    spec = SimpleNamespace(
        original_shape=(1, 2, 2, 1),
        pair_dims=(1, 2),
    )

    def normalized_pair(*_args, **_kwargs):
        value = torch.ones(1, 2, 2, 1)
        references["normalized"] = weakref.ref(value)
        return value

    def projected_b(*_args, **_kwargs):
        value = torch.ones(1, 2, 2, 1)
        references["b"] = weakref.ref(value)
        return value

    def synchronize(action, *, description, **_kwargs):
        if description == "Fold-CP triangle-multiplication B post-offload cleanup":
            gc.collect()
            cleanup_observations.append(
                (references["normalized"]() is None, references["b"]() is None)
            )
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_layer_norm_source_row_slab",
        normalized_pair,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_trimul_project_channel_chunk_size",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_a_projection_source_chunks",
        lambda *_args, **_kwargs: torch.ones(1, 2, 2, 1),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_should_project_b_owned_rows",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_b_projection_source_chunk",
        projected_b,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_should_offload_trimul_b",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_offload_trimul_b_synchronized",
        lambda value, _mesh: value.clone(),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_triangle_multiplication_source_matmul",
        lambda a, *_args, **_kwargs: torch.zeros_like(a),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_triangle_multiplication_output_norm_gate_source_slab",
        lambda _module, update, *_args, **_kwargs: update,
    )

    result = real_pairformer.distributed_triangle_multiplication_update(
        module,
        torch.zeros(2, 2, 1),
        mesh,
        z_spec=spec,
    )

    assert result.shape == (1, 2, 2, 1)
    assert cleanup_observations == [(True, True)]


def test_row_slab_pair_transition_gathers_outside_rank_action(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    inside_action = False
    stages = []
    mesh = SimpleNamespace(group_2d=object(), group_row=object())
    spec = FoldCPPairShardSpec(
        original_shape=(1, 4, 4, 1),
        padded_shape=(1, 4, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 2),
        col_range=(0, 2),
        mesh_shape=(1, 2),
        mesh_coord=(0, 0),
    )

    def synchronize(action, *, description, **_kwargs):
        nonlocal inside_action
        stages.append(description)
        inside_action = True
        try:
            return action()
        finally:
            inside_action = False

    def gather(tensor, _mesh, dim, length=None):
        assert not inside_action
        shape = list(tensor.shape)
        shape[dim if dim >= 0 else tensor.ndim + dim] = int(length)
        return tensor.new_zeros(shape)

    def transition(_tensor):
        assert inside_action
        raise RuntimeError("local row-slab transition OOM")

    monkeypatch.setattr(
        real_pairformer,
        "_pair_transition_source_flat_chunk_size",
        lambda _tensor: 0,
    )
    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with (
        torch.no_grad(),
        pytest.raises(RuntimeError, match="local row-slab transition OOM"),
    ):
        real_pairformer.distributed_pair_transition_update(
            transition,
            torch.zeros(1, 2, 2, 1),
            mesh,
            residual_local=torch.zeros(1, 2, 2, 1),
            z_spec=spec,
        )

    assert stages == ["Pairformer row-slab transition completion"]


def test_one_by_p_padding_rank_still_joins_pair_transition_row_gather(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    gathers = []
    transition_calls = []
    stages = []
    mesh = SimpleNamespace(
        group_2d=object(),
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 4)),
    )
    spec = FoldCPPairShardSpec(
        original_shape=(1, 2, 2, 1),
        padded_shape=(1, 2, 4, 1),
        pair_dims=(1, 2),
        row_range=(0, 2),
        col_range=(2, 3),
        mesh_shape=(1, 4),
        mesh_coord=(0, 2),
    )
    residual = torch.ones(1, 2, 1, 1)

    monkeypatch.setattr(
        real_pairformer,
        "_pair_transition_source_flat_chunk_size",
        lambda _tensor: 0,
    )

    def gather(tensor, _mesh, dim, length=None):
        gathers.append((dim, length))
        shape = list(tensor.shape)
        shape[dim if dim >= 0 else tensor.ndim + dim] = int(length)
        return tensor.new_zeros(shape)

    def synchronize(action, *, description, **_kwargs):
        stages.append(description)
        return action()

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with torch.no_grad():
        result = real_pairformer.distributed_pair_transition_update(
            lambda tensor: transition_calls.append(tensor) or tensor,
            torch.zeros_like(residual),
            mesh,
            residual_local=residual,
            z_spec=spec,
        )

    assert gathers == [(-2, 2)]
    assert transition_calls == []
    assert stages == ["Pairformer row-slab transition completion"]
    assert torch.equal(result, torch.ones_like(result))


def test_pairformer_row_gather_transition_runs_outside_block_rank_action(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    inside_action = False
    transition_calls = []
    stages = []
    tensor = torch.zeros(1, 1, 1)
    block = SimpleNamespace(
        tri_mul_out=object(),
        tri_mul_in=object(),
        tri_att_start=object(),
        tri_att_end=object(),
        pair_transition=object(),
    )
    mesh = SimpleNamespace(layout=SimpleNamespace(shape=(1, 2)), group_2d=object())

    monkeypatch.setattr(
        real_pairformer,
        "distributed_triangle_multiplication_update",
        lambda _module, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "distributed_triangle_attention_update",
        lambda _module, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_one_by_p_triangle_attention_ending_update",
        lambda _module, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_pair_transition_source_flat_chunk_size",
        lambda _tensor: 0,
    )

    def transition(_module, z_local, *args, **kwargs):
        assert not inside_action
        transition_calls.append("transition")
        return z_local

    def synchronize(action, *, description, **_kwargs):
        nonlocal inside_action
        stages.append(description)
        inside_action = True
        try:
            return action()
        finally:
            inside_action = False

    monkeypatch.setattr(
        real_pairformer,
        "distributed_pair_transition_update",
        transition,
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )

    with torch.no_grad():
        result = real_pairformer._distributed_pairformer_block_pair_ops(
            block,
            tensor,
            mesh,
            z_spec=SimpleNamespace(),
        )

    assert result is tensor
    assert transition_calls == ["transition"]
    assert stages[-1] == "Fold-CP Pairformer row-gather block finalization"


def test_single_finalization_failure_stops_before_next_block(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    tensor = torch.zeros(1, 1, 1)
    transition_calls = []
    block = SimpleNamespace(
        c_s=1,
        attention_pair_bias=object(),
        single_transition=lambda value: transition_calls.append(value),
    )
    mesh = SimpleNamespace(group_2d=object())

    def synchronize(action, *, description, **_kwargs):
        if description == "Fold-CP Pairformer block 0 single finalization":
            raise RuntimeError("remote single transition OOM")
        return action()

    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        synchronize,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_distributed_pairformer_block_pair_ops",
        lambda _block, z_local, *args, **kwargs: z_local,
    )
    monkeypatch.setattr(
        real_pairformer,
        "distributed_attention_pair_bias_update",
        lambda *args, **kwargs: torch.zeros_like(tensor),
    )

    with (
        torch.no_grad(),
        pytest.raises(RuntimeError, match="remote single transition OOM"),
    ):
        real_pairformer.distributed_pairformer_stack_single_bridge_update(
            SimpleNamespace(blocks=[block]),
            s=tensor,
            z=tensor,
            mesh=mesh,
            z_spec=SimpleNamespace(),
        )

    assert transition_calls == []


def test_attention_pair_bias_preparation_failure_stops_before_row_ring(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    row_gathers = []
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote AttentionPairBias preparation OOM")
        ),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_by_row",
        lambda *args, **kwargs: row_gathers.append("gather"),
    )

    with pytest.raises(RuntimeError, match="remote AttentionPairBias preparation OOM"):
        real_pairformer.distributed_attention_pair_bias_update(
            SimpleNamespace(),
            a=torch.zeros(4, 1),
            z_local=torch.zeros(4, 2, 1),
            mesh=SimpleNamespace(group_2d=object()),
        )

    assert row_gathers == []


def test_attention_pair_bias_drains_row_rings_after_local_projection_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    row_gathers = []
    final_gathers = []

    class FailingLayerNorm:
        def __call__(self, _value):
            raise RuntimeError("local AttentionPairBias projection OOM")

    attention = SimpleNamespace(
        _prep_qkv=lambda **_kwargs: (
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 1),
        ),
        use_efficient_implementation=False,
    )
    module = SimpleNamespace(
        has_s=False,
        cross_attention_mode=False,
        layernorm_a=torch.nn.Identity(),
        layernorm_z=FailingLayerNorm(),
        attention=attention,
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_attention_pair_bias_row_launch_size",
        lambda *_args: 2,
    )

    def gather_row(local_tensor, *_args, **_kwargs):
        row_gathers.append("gather")
        return local_tensor

    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather_row)
    monkeypatch.setattr(
        real_pairformer,
        "_gather_single_update_by_2d_ring",
        lambda *args, **kwargs: final_gathers.append("gather"),
    )

    with pytest.raises(RuntimeError, match="local AttentionPairBias projection OOM"):
        real_pairformer.distributed_attention_pair_bias_update(
            module,
            a=torch.zeros(4, 1),
            z_local=torch.zeros(4, 2, 1),
            mesh=mesh,
        )

    assert row_gathers == ["gather", "gather"]
    assert final_gathers == []


def test_attention_pair_bias_releases_completed_compute_chunks(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    mesh = SimpleNamespace(
        group_2d=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
    )
    attention = SimpleNamespace(
        _prep_qkv=lambda **_kwargs: (
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 1),
            torch.zeros(1, 4, 1),
        ),
        use_efficient_implementation=False,
        _wrap_up=lambda source, _a: source[:, 0, :],
    )
    module = SimpleNamespace(
        has_s=False,
        cross_attention_mode=False,
        layernorm_a=torch.nn.Identity(),
        layernorm_z=torch.nn.Identity(),
        linear_nobias_z=torch.nn.Identity(),
        attention=attention,
    )
    prior_refs = []
    attention_calls = 0

    def run_attention(*, q, attn_bias, **_kwargs):
        nonlocal attention_calls, prior_refs
        if attention_calls:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        output = q + 1
        prior_refs = [weakref.ref(q), weakref.ref(attn_bias)]
        prior_refs.append(weakref.ref(output))
        attention_calls += 1
        return output

    monkeypatch.setenv(
        "OPENDDE_FOLDCP_ATTN_PAIR_BIAS_SOURCE_GRID_MAX_BYTES",
        "0",
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_attention_pair_bias_row_launch_size",
        lambda *_args: 1,
    )
    monkeypatch.setattr(
        real_pairformer,
        "_ring_gather_by_row",
        lambda tensor, *_args, **_kwargs: tensor.repeat(1, 2, 1),
    )
    monkeypatch.setattr(real_pairformer, "_single_feature_attention", run_attention)
    monkeypatch.setattr(
        real_pairformer,
        "_gather_single_update_by_2d_ring",
        lambda update, *_args, **_kwargs: update,
    )

    result = real_pairformer.distributed_attention_pair_bias_update(
        module,
        a=torch.zeros(4, 1),
        z_local=torch.zeros(4, 2, 1),
        mesh=mesh,
    )

    assert result.shape == (2, 1)
    assert attention_calls == 2


def test_trimul_output_gate_releases_completed_flat_chunks(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    prior_refs = []
    calls = 0

    def layer_norm(value):
        nonlocal calls, prior_refs
        if calls:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        normalized = value + 1
        prior_refs = [weakref.ref(normalized)]
        calls += 1
        return normalized

    def linear_z(value):
        projected = value + 1
        prior_refs.append(weakref.ref(projected))
        return projected

    def linear_g(value):
        gate = value + 1
        prior_refs.append(weakref.ref(gate))
        return gate

    monkeypatch.setenv("OPENDDE_FOLDCP_TRIMUL_OUTPUT_GATE_FLAT_CHUNK", "2")
    result = real_pairformer._triangle_multiplication_output_norm_gate(
        SimpleNamespace(
            c_z=1,
            layer_norm_out=layer_norm,
            linear_z=linear_z,
            linear_g=linear_g,
        ),
        torch.zeros(4, 1),
        torch.zeros(4, 1),
    )

    assert result.shape == (4, 1)
    assert calls == 2


def test_triangle_attention_wrap_releases_completed_row_chunks(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    prior_refs = []
    calls = 0

    def linear_g(value):
        nonlocal calls, prior_refs
        if calls:
            gc.collect()
            assert all(reference() is None for reference in prior_refs)
        gate = torch.ones(*value.shape[:-1], 1)
        prior_refs = [weakref.ref(value), weakref.ref(gate)]
        calls += 1
        return gate

    def linear_o(value):
        update = value + 1
        prior_refs.append(weakref.ref(update))
        return update

    monkeypatch.setattr(
        real_pairformer,
        "_triatt_wrap_row_chunk_size",
        lambda *_args: 1,
    )
    result = real_pairformer._wrap_up_triangle_attention_output(
        SimpleNamespace(
            linear_g=linear_g,
            sigmoid=lambda value: value,
            no_heads=1,
            linear_o=linear_o,
        ),
        torch.zeros(2, 1, 1, 1),
        torch.zeros(2, 1, 1),
    )

    assert result.shape == (2, 1, 1)
    assert calls == 2


def test_pairformer_bmm_lhs_does_not_exchange_after_remote_allocation_oom(
    monkeypatch,
):
    from opendde.distributed.foldcp import real_pairformer

    exchanges = []
    ring = SimpleNamespace(
        comm_row=SimpleNamespace(
            exchange=lambda *args, **kwargs: exchanges.append("exchange")
        )
    )
    mesh = SimpleNamespace(
        group_row=object(),
        layout=SimpleNamespace(shape=(1, 2)),
        coord=(0, 0),
        ring_comm=lambda: ring,
    )
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("remote BMM-LHS gather allocation OOM")
        ),
    )

    with pytest.raises(RuntimeError, match="remote BMM-LHS gather allocation OOM"):
        real_pairformer._ring_gather_pair_matmul_lhs(
            torch.zeros(1, 4, 2, 1),
            mesh,
            length=4,
            incoming=False,
        )

    assert exchanges == []


def test_confidence_single_update_uses_only_maintained_1xp_row_ring(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    calls = []

    def gather(local_tensor, **kwargs):
        calls.append(
            (
                kwargs["description"],
                kwargs["side"],
                kwargs["dim"],
                kwargs["length"],
            )
        )
        shape = list(local_tensor.shape)
        shape[kwargs["dim"]] = kwargs["length"]
        return local_tensor.new_zeros(shape)

    monkeypatch.setattr(real_pairformer, "gather_tensor_by_ring", gather)
    mesh = SimpleNamespace(
        group_row=object(),
        coord=(0, 0),
        layout=SimpleNamespace(shape=(1, 2)),
        ring_comm=lambda: SimpleNamespace(comm_row=object()),
    )
    result = real_pairformer._gather_single_update_by_2d_ring(
        torch.zeros(1, 2, 1),
        n_token=3,
        mesh=mesh,
        pair_row_tile=3,
    )

    assert result.shape == (1, 3, 1)
    assert calls == [
        ("AttentionPairBias single-update row ring", 2, -2, 3),
    ]


def test_starting_triangle_bias_uses_safe_source_row_rings(monkeypatch):
    from opendde.distributed.foldcp import real_pairformer

    calls = []

    def gather(local_tensor, **kwargs):
        calls.append(kwargs["description"])
        return torch.cat([local_tensor, local_tensor], dim=kwargs["dim"])

    def gather_by_row(local_tensor, _mesh, dim, length=None):
        gathered = torch.cat([local_tensor, local_tensor], dim=dim)
        if length is not None:
            dim = dim if dim >= 0 else gathered.ndim + dim
            gathered = gathered.narrow(dim, 0, length)
        return gathered.contiguous()

    monkeypatch.setattr(real_pairformer, "gather_tensor_by_ring", gather)
    monkeypatch.setattr(real_pairformer, "_ring_gather_by_row", gather_by_row)
    monkeypatch.setattr(
        real_pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )
    monkeypatch.setattr(
        real_pairformer,
        "_project_starting_triangle_bias_local_tile",
        lambda *_args, **_kwargs: torch.zeros(1, 2, 2),
    )
    mesh = SimpleNamespace(
        group_2d=object(),
        group_col=object(),
        coord=(0, 0),
        layout=SimpleNamespace(shape=(2, 2)),
        ring_comm=lambda: SimpleNamespace(comm_col=object()),
    )
    attention = SimpleNamespace(linear=torch.nn.Linear(1, 1, bias=False))
    x_local = torch.zeros(2, 2, 1)

    source_grid_result = (
        real_pairformer._starting_triangle_bias_full_key_from_source_slab(
            attention,
            x_local,
            mesh,
            original_n=4,
            query_start=0,
            valid_query=2,
            source_grid_launch=True,
        )
    )
    projected_result = (
        real_pairformer._starting_triangle_bias_full_key_from_source_slab(
            attention,
            x_local,
            mesh,
            original_n=4,
            query_start=0,
            valid_query=2,
            source_grid_launch=False,
        )
    )

    assert source_grid_result.shape == (1, 2, 4)
    assert projected_result.shape == (1, 2, 4)
    assert calls == [
        "starting triangle-bias source-row ring",
        "starting triangle-bias projected-row ring",
    ]


def test_opm_norm_drains_ring_after_local_compute_failure(monkeypatch):
    from opendde.model.modules import pairformer

    events = []
    payload_reference = None

    class Payload:
        pass

    def exchange(name):
        def _exchange(source, *, to_recv):
            events.append(name)
            to_recv.copy_(source)
            return to_recv

        return _exchange

    ring = SimpleNamespace(
        layout=SimpleNamespace(shape=(3, 3)),
        comm_2d_trans=SimpleNamespace(exchange=exchange("transpose")),
        comm_row_init=SimpleNamespace(exchange=exchange("row_init")),
        comm_col_init=SimpleNamespace(exchange=exchange("col_init")),
        comm_row=SimpleNamespace(exchange=exchange("row")),
        comm_col=SimpleNamespace(exchange=exchange("col")),
    )
    mesh = SimpleNamespace(group_2d=object(), ring_comm=lambda: ring)
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **kwargs: action(),
    )

    def fail_einsum(*_args, **_kwargs):
        nonlocal payload_reference
        payload = Payload()
        payload_reference = weakref.ref(payload)
        raise RuntimeError("local OPM compute OOM")

    monkeypatch.setattr(pairformer.torch, "einsum", fail_einsum)

    with pytest.raises(RuntimeError, match="local OPM compute OOM") as captured:
        pairformer.MSABlock._foldcp_opm_norm(
            object(),
            torch.zeros(1, 2, 2),
            mesh,
        )

    assert events == [
        "transpose",
        "row_init",
        "col_init",
        "row",
        "col",
        "row",
        "col",
    ]
    assert captured.value is not None
    gc.collect()
    assert payload_reference is not None
    assert payload_reference() is None


def test_opm_ring_releases_norm_and_completed_channel_chunks(monkeypatch):
    from opendde.model.modules import pairformer

    norm_reference = None
    prior_channel_reference = None
    linear_calls = 0
    original_linear = pairformer.F.linear

    def opm_norm(_mask, _mesh):
        nonlocal norm_reference
        norm = torch.ones(1, 1, 1)
        norm_reference = weakref.ref(norm)
        return norm

    def linear(input, weight, bias=None):
        nonlocal prior_channel_reference, linear_calls
        gc.collect()
        if linear_calls == 0:
            assert norm_reference is not None and norm_reference() is None
        else:
            assert (
                prior_channel_reference is not None
                and prior_channel_reference() is None
            )
        prior_channel_reference = weakref.ref(input)
        linear_calls += 1
        return original_linear(input, weight, bias)

    def exchange(source, *, to_recv):
        to_recv.copy_(source)
        return to_recv

    ring = SimpleNamespace(
        layout=SimpleNamespace(shape=(1, 1)),
        comm_2d_trans=SimpleNamespace(exchange=exchange),
        comm_row_init=SimpleNamespace(exchange=exchange),
        comm_col_init=SimpleNamespace(exchange=exchange),
    )
    mesh = SimpleNamespace(group_2d=object(), ring_comm=lambda: ring)
    module = SimpleNamespace(
        outer_product_mean_msa=SimpleNamespace(
            eps=1e-3,
            linear_out=SimpleNamespace(
                weight=torch.ones(1, 4),
                bias=None,
            ),
        ),
        _foldcp_opm_norm=opm_norm,
    )
    monkeypatch.setenv("OPENDDE_FOLDCP_OPM_CHANNEL_CHUNK", "1")
    monkeypatch.setattr(pairformer.F, "linear", linear)
    monkeypatch.setattr(
        pairformer,
        "run_group_rank_action_synchronized",
        lambda action, **_kwargs: action(),
    )

    result = pairformer.MSABlock._foldcp_add_opm_to_local_pair_no_grad(
        module,
        torch.ones(1, 1, 1, 2),
        torch.ones(1, 1, 1, 2),
        torch.ones(1, 1, 1),
        torch.zeros(1, 1, 1),
        mesh,
    )

    assert result.shape == (1, 1, 1)
    assert linear_calls == 2


def test_foldcp_communication_cache_releases_status_tensors():
    from opendde.distributed.foldcp import comm

    status = torch.empty((), dtype=torch.int32)
    comm._NCCL_STATUS_TENSORS[(123, 0)] = status

    comm.clear_foldcp_communication_cache()

    assert comm._NCCL_STATUS_TENSORS == {}


def test_detach_rank_local_error_traceback_releases_failed_frame_payload():
    from opendde.distributed.foldcp import comm

    class Payload:
        pass

    payload_reference = None

    def fail_with_payload():
        nonlocal payload_reference
        payload = Payload()
        payload_reference = weakref.ref(payload)
        raise RuntimeError("local CUDA OOM")

    retained_error = None
    try:
        fail_with_payload()
    except RuntimeError as exc:
        retained_error = comm.detach_rank_local_error_traceback(exc)

    gc.collect()
    assert retained_error is not None
    assert retained_error.__traceback__ is None
    assert str(retained_error) == "local CUDA OOM"
    assert payload_reference is not None
    assert payload_reference() is None


def test_rank_action_releases_failed_payload_before_cuda_cache_cleanup(monkeypatch):
    from opendde.distributed.foldcp import comm

    class Payload:
        pass

    payload_reference = None
    cleanup_observations = []

    def fail_with_payload():
        nonlocal payload_reference
        payload = Payload()
        payload_reference = weakref.ref(payload)
        raise RuntimeError("rank-local OOM")

    def observe_cleanup(error, *, attempt):
        gc.collect()
        cleanup_observations.append((attempt, payload_reference() is None))
        return error

    monkeypatch.setattr(comm, "_prime_nccl_group_status", lambda _group: None)
    monkeypatch.setattr(comm, "_arm_nccl_group_status", lambda _group: False)
    monkeypatch.setattr(comm, "_nccl_group_has_failure", lambda *_args: None)
    monkeypatch.setattr(comm, "_append_cuda_cache_cleanup_error", observe_cleanup)
    monkeypatch.setattr(
        comm,
        "_gather_rank_errors",
        lambda error, **_kwargs: [error],
    )
    monkeypatch.setattr(comm.dist, "get_rank", lambda _group: 0)

    with pytest.raises(RuntimeError, match="rank-local OOM"):
        comm.run_group_rank_action_synchronized(
            fail_with_payload,
            group=object(),
            description="payload test",
        )

    assert cleanup_observations == [(True, True)]


def test_detach_rank_local_error_traceback_releases_chained_error_payload():
    from opendde.distributed.foldcp import comm

    class Payload:
        pass

    payload_reference = None

    def fail_inner():
        nonlocal payload_reference
        payload = Payload()
        payload_reference = weakref.ref(payload)
        raise RuntimeError("inner CUDA OOM")

    def fail_wrapped():
        try:
            fail_inner()
        except RuntimeError as cause:
            raise ValueError("outer distributed failure") from cause

    retained_error = None
    try:
        fail_wrapped()
    except ValueError as exc:
        retained_error = comm.detach_rank_local_error_traceback(exc)

    gc.collect()
    assert retained_error is not None
    assert retained_error.__traceback__ is None
    assert retained_error.__cause__ is not None
    assert retained_error.__cause__.__traceback__ is None
    assert str(retained_error) == "outer distributed failure"
    assert str(retained_error.__cause__) == "inner CUDA OOM"
    assert payload_reference is not None
    assert payload_reference() is None
