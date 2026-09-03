# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from types import SimpleNamespace

import pytest
import torch

from opendde.data.inference.infer_dataloader import (
    InferenceJobSampler,
    _data_parallel_coordinates,
)
from opendde.distributed.foldcp.config import FoldCPConfig


def test_foldcp_sampler_replicates_every_job_across_1xp(monkeypatch):
    from opendde.data.inference import infer_dataloader

    configs = SimpleNamespace(
        foldcp_mode="distributed",
        foldcp_size_dp=1,
        foldcp_size_cp=4,
    )
    assignments = []
    monkeypatch.setattr(infer_dataloader.DIST_WRAPPER, "world_size", 4)
    for world_rank in range(4):
        monkeypatch.setattr(infer_dataloader.DIST_WRAPPER, "rank", world_rank)
        size_dp, dp_rank = _data_parallel_coordinates(configs)
        assignments.append(
            list(
                InferenceJobSampler(
                    range(5),
                    num_replicas=size_dp,
                    rank=dp_rank,
                )
            )
        )

    assert assignments == [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]


def test_inference_sampler_marks_padded_jobs_as_not_owned():
    samplers = [
        InferenceJobSampler(range(3), num_replicas=4, rank=rank) for rank in range(4)
    ]

    assert [list(sampler) for sampler in samplers] == [[0], [1], [2], [0]]
    assert [sampler.owns(0) for sampler in samplers] == [True, False, False, False]


def test_inference_sampler_can_select_jobs_for_one_seed():
    sampler = InferenceJobSampler(range(5), num_replicas=2, rank=1)
    sampler.set_sample_indices([0, 2, 3])

    assert list(sampler) == [2, 0]
    assert len(sampler) == 2
    assert sampler.owns(2)
    assert not sampler.owns(0)


def test_padded_data_parallel_sample_does_not_own_persistent_outputs():
    from runner import inference

    real_owner = InferenceJobSampler(range(3), num_replicas=4, rank=0)
    padded_worker = InferenceJobSampler(range(3), num_replicas=4, rank=3)
    sample = {"sample_index": 0, "sample_name": "job0"}

    assert inference._sampler_owns_inference_sample(real_owner, sample)
    assert not inference._sampler_owns_inference_sample(padded_worker, sample)


def test_data_parallel_coordinates_use_initialized_process_group(monkeypatch):
    from opendde.data.inference import infer_dataloader

    monkeypatch.setattr(
        infer_dataloader.torch.distributed, "is_available", lambda: True
    )
    monkeypatch.setattr(
        infer_dataloader.torch.distributed, "is_initialized", lambda: True
    )
    monkeypatch.setattr(infer_dataloader.torch.distributed, "get_world_size", lambda: 6)
    monkeypatch.setattr(infer_dataloader.torch.distributed, "get_rank", lambda: 5)

    assert _data_parallel_coordinates(SimpleNamespace(foldcp_mode="single")) == (6, 5)
    assert _data_parallel_coordinates(
        SimpleNamespace(
            foldcp_mode="distributed",
            foldcp_size_dp=1,
            foldcp_size_cp=6,
        )
    ) == (1, 0)


def test_data_parallel_coordinates_reject_removed_multirow_topology(monkeypatch):
    from opendde.data.inference import infer_dataloader

    monkeypatch.setattr(infer_dataloader.DIST_WRAPPER, "world_size", 4)
    monkeypatch.setattr(infer_dataloader.DIST_WRAPPER, "rank", 0)

    with pytest.raises(ValueError, match="foldcp_size_dp must be 1"):
        _data_parallel_coordinates(
            SimpleNamespace(
                foldcp_mode="distributed",
                foldcp_size_dp=2,
                foldcp_size_cp=2,
            )
        )


def test_predict_input_transfer_failure_stops_before_distributed_model(monkeypatch):
    from runner import inference

    model_calls = []
    runner = inference.InferenceRunner.__new__(inference.InferenceRunner)
    runner.configs = SimpleNamespace(dtype="fp32")
    runner.use_cuda = False
    runner.device = torch.device("cpu")
    runner.foldcp_config = SimpleNamespace(enabled=True)
    runner.foldcp_recorder = SimpleNamespace()
    runner.model = lambda **_kwargs: model_calls.append("model")
    monkeypatch.setattr(
        inference,
        "_run_rank_stage_synchronized",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            inference.FoldCPJobCoordinationError("remote input transfer OOM")
        ),
    )

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="remote input transfer OOM",
    ):
        runner.predict({"sample_name": "sample", "input_feature_dict": {}})

    assert model_calls == []


def test_predict_metric_initialization_failure_stops_before_model(monkeypatch):
    from runner import inference

    control_group = object()
    model_calls = []
    runner = inference.InferenceRunner.__new__(inference.InferenceRunner)
    runner.configs = SimpleNamespace(dtype="fp32")
    runner.use_cuda = False
    runner.device = torch.device("cpu")
    runner.foldcp_config = SimpleNamespace(enabled=True)
    runner.foldcp_world_control_group = control_group
    runner.foldcp_recorder = SimpleNamespace()
    runner.model = lambda **_kwargs: model_calls.append("model")
    monkeypatch.setattr(inference, "to_device", lambda data, _device: data)

    def synchronize(action, *, stage, world_control_group, **_kwargs):
        assert world_control_group is control_group
        if stage == "model-forward metric initialization":
            raise inference.FoldCPJobCoordinationError("remote metric init failure")
        action()

    monkeypatch.setattr(inference, "_run_rank_stage_synchronized", synchronize)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="remote metric init failure",
    ):
        runner.predict({"sample_name": "sample", "input_feature_dict": {}})

    assert model_calls == []


def test_resolve_job_seed_schedule_is_per_job(monkeypatch):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: False)
    jobs = [{"modelSeeds": [11]}, {"modelSeeds": [22, 23]}, {}]
    monkeypatch.setattr(inference.random, "randint", lambda _low, _high: 31)

    assert inference._resolve_job_seed_schedule(jobs, None) == [
        [11],
        [22, 23],
        [31],
    ]
    assert inference._resolve_job_seed_schedule(jobs, [7, 8]) == [
        [7, 8],
        [7, 8],
        [7, 8],
    ]


def test_resolve_job_seed_schedule_broadcasts_rank0_failure(monkeypatch):
    from runner import inference

    broadcasts = []
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        inference,
        "validate_inference_seed",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("invalid seed")),
    )
    monkeypatch.setattr(
        inference.dist,
        "broadcast_object_list",
        lambda payload, src: broadcasts.append((payload[0], src)),
    )

    with pytest.raises(ValueError, match="Invalid inference seed schedule"):
        inference._resolve_job_seed_schedule([{"modelSeeds": [1]}], None)

    assert len(broadcasts) == 1
    assert broadcasts[0][0][0] is False
    assert broadcasts[0][1] == 0


def test_resolve_job_seed_schedule_nonzero_rank_joins_broadcast(monkeypatch):
    from runner import inference

    control_group = object()
    broadcasts = []
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 1)

    def broadcast(payload, *, src, group):
        broadcasts.append((src, group))
        payload[0] = (True, [[301]])

    monkeypatch.setattr(inference.dist, "broadcast_object_list", broadcast)

    result = inference._resolve_job_seed_schedule(
        [{"modelSeeds": [999]}],
        None,
        control_group,
    )

    assert result == [[301]]
    assert broadcasts == [(0, control_group)]


def test_foldcp_batch_guard_rejects_different_jobs(monkeypatch):
    from runner import inference

    def gather(output, _descriptor, *, group):
        assert group is control_group
        output[:] = [
            {
                "sample_index": 0,
                "sample_name": "first",
                "seed": 1,
                "N_token": 10,
                "N_atom": 20,
                "N_msa": 1,
                "error": "",
            },
            {
                "sample_index": 1,
                "sample_name": "second",
                "seed": 1,
                "N_token": 12,
                "N_atom": 24,
                "N_msa": 1,
                "error": "",
            },
        ]

    control_group = object()
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    error = inference._synchronize_foldcp_batch(
        data={
            "sample_index": 0,
            "sample_name": "first",
            "N_token": torch.tensor(10),
            "N_atom": torch.tensor(20),
            "N_msa": torch.tensor(1),
        },
        data_error_message="",
        seed=1,
        group=control_group,
        size_cp=2,
    )

    assert "different inference jobs" in error


def test_foldcp_batch_guard_propagates_preprocessing_error(monkeypatch):
    from runner import inference

    def gather(output, descriptor, *, group):
        assert group is control_group
        failed = dict(descriptor)
        failed["error"] = "bad MSA"
        output[:] = [descriptor, failed]

    control_group = object()
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)
    error = inference._synchronize_foldcp_batch(
        data={"sample_index": 0, "sample_name": "same"},
        data_error_message="",
        seed=1,
        group=control_group,
        size_cp=2,
    )

    assert error == "CP rank 1: bad MSA"


def test_foldcp_batch_guard_rejects_different_feature_contents(monkeypatch):
    from runner import inference

    def gather(output, descriptor, *, group):
        assert group is control_group
        different = dict(descriptor)
        different["feature_fingerprint"] = "different"
        output[:] = [descriptor, different]

    control_group = object()
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)
    error = inference._synchronize_foldcp_batch(
        data={
            "sample_index": 0,
            "sample_name": "same",
            "N_token": torch.tensor(2),
            "N_atom": torch.tensor(2),
            "N_msa": torch.tensor(1),
            "input_feature_dict": {"token_index": torch.tensor([0, 1])},
        },
        data_error_message="",
        seed=1,
        group=control_group,
        size_cp=2,
    )

    assert "different feature contents" in error


def test_foldcp_batch_guard_synchronizes_local_fingerprint_failure(monkeypatch):
    from runner import inference

    gathered_descriptors = []

    def gather(output, descriptor, *, group):
        assert group is control_group
        gathered_descriptors.append(descriptor)
        output[:] = [descriptor, dict(descriptor)]

    control_group = object()
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        inference,
        "_feature_fingerprint",
        lambda _data: (_ for _ in ()).throw(RuntimeError("host copy OOM")),
    )

    error = inference._synchronize_foldcp_batch(
        data={
            "sample_index": 0,
            "sample_name": "same",
            "N_token": torch.tensor(2),
            "N_atom": torch.tensor(2),
            "N_msa": torch.tensor(1),
            "input_feature_dict": {"token_index": torch.tensor([0, 1])},
        },
        data_error_message="",
        seed=1,
        group=control_group,
        size_cp=2,
    )

    assert len(gathered_descriptors) == 1
    assert "input descriptor preparation failed" in error
    assert "host copy OOM" in error


def test_dataloader_none_result_is_synchronized(monkeypatch):
    from runner import inference

    synchronized_errors = []
    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **kwargs: None,
    )

    def synchronize(local_error, _foldcp_config):
        synchronized_errors.append(local_error)
        return local_error

    monkeypatch.setattr(inference, "_synchronize_foldcp_world_error", synchronize)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="returned no dataloader",
    ):
        inference._create_inference_dataloader_synchronized(
            SimpleNamespace(),
            [],
            SimpleNamespace(enabled=True),
        )

    assert len(synchronized_errors) == 1
    assert "returned no dataloader" in synchronized_errors[0]


def test_dataloader_size_failure_is_synchronized(monkeypatch):
    from runner import inference

    synchronized_errors = []

    class BrokenDataset:
        def __len__(self):
            raise RuntimeError("bad length")

    dataloader = SimpleNamespace(dataset=BrokenDataset())

    def synchronize(local_error, _foldcp_config):
        synchronized_errors.append(local_error)
        return local_error

    monkeypatch.setattr(inference, "_synchronize_foldcp_world_error", synchronize)

    with pytest.raises(inference.FoldCPJobCoordinationError, match="bad length"):
        inference._get_dataloader_size_synchronized(
            dataloader,
            SimpleNamespace(enabled=True),
        )

    assert len(synchronized_errors) == 1
    assert "dataset-size inspection" in synchronized_errors[0]


def test_dataloader_size_mismatch_is_rejected_before_seed_loop(monkeypatch):
    from runner import inference

    dataloader = SimpleNamespace(dataset=[1, 2])
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_world_error",
        lambda local_error, _foldcp_config: local_error,
    )
    monkeypatch.setattr(inference.dist, "get_world_size", lambda _group=None: 2)

    def gather(output, _local_size):
        output[:] = [2, 3]

    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="different dataset sizes",
    ):
        inference._get_dataloader_size_synchronized(
            dataloader,
            SimpleNamespace(enabled=True),
        )


def test_foldcp_world_error_is_synchronized_for_cp_only_launch(monkeypatch):
    from runner import inference

    def gather(output, local_error):
        output[:] = [local_error, "rank 1 failed"]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)
    foldcp = SimpleNamespace(enabled=True, size_dp=1)

    error = inference._synchronize_foldcp_world_error("", foldcp)

    assert error == "Global rank 1: rank 1 failed"


def test_foldcp_batch_error_is_not_gathered_twice_for_1xp_world(monkeypatch):
    from runner import inference

    control_group = object()
    world_gathers = []
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_world_error",
        lambda *_args, **_kwargs: world_gathers.append(True) or "unexpected",
    )

    error = inference._finalize_foldcp_batch_error(
        "CP rank 1: malformed feature",
        SimpleNamespace(enabled=True),
        control_group,
        control_group,
    )

    assert error == "CP rank 1: malformed feature"
    assert world_gathers == []


def test_non_foldcp_batch_error_uses_normal_world_error_path(monkeypatch):
    from runner import inference

    world_calls = []
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_world_error",
        lambda error, config, group: (
            world_calls.append((error, config, group)) or error
        ),
    )
    config = SimpleNamespace(enabled=False)

    assert (
        inference._finalize_foldcp_batch_error("bad input", config, None, None)
        == "bad input"
    )
    assert world_calls == [("bad input", config, None)]


def test_foldcp_output_error_is_synchronized_within_cp_group(monkeypatch):
    from runner import inference

    control_group = object()

    def gather(output, local_error, *, group):
        assert group is control_group
        output[:] = ["rank 0 disk full", local_error]

    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    error = inference._synchronize_foldcp_group_error(
        "", group=control_group, size_cp=2
    )

    assert error == "CP rank 0: rank 0 disk full"


def test_foldcp_output_error_is_propagated_across_1xp_world(monkeypatch):
    from runner import inference

    control_group = object()
    cp_gathers = []
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_group_error",
        lambda local_error, **_kwargs: cp_gathers.append(local_error) or local_error,
    )
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_world_error",
        lambda local_error, _config, _group=None: (
            local_error or "remote rank disk full"
        ),
    )

    error = inference._synchronize_foldcp_output_error(
        "",
        group=control_group,
        size_cp=2,
        foldcp_config=SimpleNamespace(enabled=True),
        world_control_group=control_group,
    )

    assert error == "remote rank disk full"
    assert cp_gathers == []


def test_foldcp_output_error_without_world_group_uses_one_cp_gather(monkeypatch):
    from runner import inference

    control_group = object()
    world_gathers = []
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_group_error",
        lambda local_error, **_kwargs: f"CP rank 0: {local_error}",
    )
    monkeypatch.setattr(
        inference,
        "_synchronize_foldcp_world_error",
        lambda *_args, **_kwargs: world_gathers.append(True) or "unexpected",
    )

    error = inference._synchronize_foldcp_output_error(
        "disk full",
        group=control_group,
        size_cp=2,
        foldcp_config=SimpleNamespace(enabled=True),
    )

    assert error == "CP rank 0: disk full"
    assert world_gathers == []


def test_infer_predict_reuses_runner_owned_control_group(monkeypatch):
    from runner import inference

    control_group = object()
    runner = SimpleNamespace(
        foldcp_config=SimpleNamespace(enabled=True),
        foldcp_control_group=control_group,
        foldcp_world_control_group=control_group,
        foldcp_cp_rank=2,
    )
    calls = []
    monkeypatch.setattr(
        inference,
        "_create_foldcp_control_groups",
        lambda _config: pytest.fail("prediction recreated the Runner-owned group"),
    )
    monkeypatch.setattr(
        inference,
        "_infer_predict_impl",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        inference,
        "_destroy_foldcp_control_groups",
        lambda *_args: pytest.fail("prediction destroyed the Runner-owned group"),
    )

    configs = object()
    inference.infer_predict(runner, configs)

    assert calls == [
        (
            (runner, configs),
            {
                "control_group": control_group,
                "world_control_group": control_group,
                "cp_rank": 2,
            },
        )
    ]


def test_control_group_rejects_rank_topology_mismatch_before_new_group(monkeypatch):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)

    def gather(output, local_topology):
        output[:] = [
            local_topology,
            {
                "mode": "distributed",
                "enabled": True,
                "size_dp": 2,
                "size_cp": 2,
            },
            local_topology,
            local_topology,
        ]

    new_group_calls = []
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        inference.dist,
        "new_group",
        lambda ranks, **kwargs: new_group_calls.append((ranks, kwargs)),
    )

    config = SimpleNamespace(enabled=True, size_dp=1, size_cp=4)
    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="different Fold-CP topologies",
    ):
        inference._create_foldcp_control_groups(config)

    assert new_group_calls == []


def test_control_group_checks_world_size_after_every_rank_preflights_topology(
    monkeypatch,
):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    gathered = []

    def gather(output, local_topology):
        gathered.append(local_topology)
        output[:] = [local_topology] * 4

    new_group_calls = []
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        inference.dist,
        "new_group",
        lambda *args, **kwargs: new_group_calls.append((args, kwargs)),
    )

    with pytest.raises(RuntimeError, match="WORLD_SIZE.*4 vs 3"):
        inference._create_foldcp_control_groups(
            SimpleNamespace(enabled=True, size_dp=1, size_cp=3)
        )

    assert gathered == [
        {
            "mode": "distributed",
            "enabled": True,
            "size_dp": 1,
            "size_cp": 3,
        }
    ]
    assert new_group_calls == []


def test_disabled_rank_still_joins_control_topology_preflight(monkeypatch):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)

    gather_calls = []

    def gather(output, local_topology):
        gather_calls.append(local_topology)
        output[:] = [local_topology, local_topology]

    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    assert inference._create_foldcp_control_groups(
        SimpleNamespace(enabled=False, size_dp=1, size_cp=1)
    ) == (None, None, 0)
    assert gather_calls == [
        {"mode": "single", "enabled": False, "size_dp": 1, "size_cp": 1}
    ]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            FoldCPConfig(mode="single", size_dp=1, size_cp=2),
            id="single-with-multiple-ranks",
        ),
        pytest.param(
            FoldCPConfig(mode="distributed", size_dp=1, size_cp=1),
            id="distributed-with-one-rank",
        ),
    ],
)
def test_control_group_validates_direct_config_after_topology_preflight(
    monkeypatch,
    config,
):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    gathered = []

    def gather(output, topology):
        gathered.append(topology)
        output[:] = [topology, topology]

    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(ValueError):
        inference._create_foldcp_control_groups(config)

    assert len(gathered) == 1


def test_1xp_control_group_uses_one_world_gloo_group(monkeypatch):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "is_gloo_available", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(
        inference.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value] * 4),
    )
    control_group = object()
    new_group_calls = []

    def new_group(ranks, *, backend):
        new_group_calls.append((ranks, backend))
        return control_group

    monkeypatch.setattr(inference.dist, "new_group", new_group)

    cp_group, world_group, cp_rank = inference._create_foldcp_control_groups(
        SimpleNamespace(enabled=True, size_dp=1, size_cp=4)
    )

    assert cp_group is control_group
    assert world_group is control_group
    assert cp_rank == 2
    assert new_group_calls == [([0, 1, 2, 3], "gloo")]


def test_control_group_rejects_non_1xp_topology(monkeypatch):
    from runner import inference

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        inference.dist,
        "all_gather_object",
        lambda output, value: output.__setitem__(slice(None), [value] * 4),
    )
    monkeypatch.setattr(inference.dist, "new_group", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="foldcp_size_dp must be 1"):
        inference._create_foldcp_control_groups(
            SimpleNamespace(enabled=True, size_dp=2, size_cp=2)
        )


def test_model_error_cleanup_failure_keeps_primary_error_for_rank_sync(monkeypatch):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "cleanup_device_memory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    message = inference._cleanup_after_model_error(
        torch.device("cpu"),
        "primary model failure",
    )

    assert "primary model failure" in message
    assert "cleanup after model failure also failed" in message
    assert "cleanup failed" in message


def test_batch_preparation_failure_becomes_a_synchronizable_error():
    from runner import inference

    data, atom_array, error = inference._prepare_inference_batch([object()])

    assert data == {"sample_index": -1, "sample_name": "unknown"}
    assert atom_array is None
    assert "Batch preparation failed" in error
    assert "cannot unpack" in error


def test_dataloader_iteration_failure_is_synchronized(monkeypatch):
    from runner import inference

    def gather(output, local_status):
        output[:] = [
            local_status,
            {"status": "batch", "error": ""},
        ]

    def failing_iterator():
        raise OSError("feature cache unreadable")
        yield

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="feature cache unreadable",
    ):
        inference._next_inference_batch_synchronized(
            iter(failing_iterator()),
            SimpleNamespace(enabled=True),
        )


def test_peer_dataloader_failure_releases_healthy_rank_batch_before_raise(
    monkeypatch,
):
    import gc
    import weakref

    from runner import inference

    class Batch:
        pass

    class OneBatchIterator:
        def __init__(self, value):
            self.value = value

        def __next__(self):
            value = self.value
            self.value = None
            return value

    batch = Batch()
    batch_ref = weakref.ref(batch)
    iterator = OneBatchIterator(batch)
    del batch

    def gather(output, local_status):
        output[:] = [
            local_status,
            {"status": "error", "error": "rank 1 feature OOM"},
        ]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    retained_error = None
    try:
        inference._next_inference_batch_synchronized(
            iterator,
            SimpleNamespace(enabled=True),
        )
    except inference.FoldCPJobCoordinationError as exc:
        retained_error = exc

    assert retained_error is not None
    assert "feature OOM" in str(retained_error)
    gc.collect()
    assert batch_ref() is None


def test_dataloader_uneven_end_is_rejected(monkeypatch):
    from runner import inference

    def gather(output, local_status):
        output[:] = [
            local_status,
            {"status": "batch", "error": ""},
        ]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="end at different steps",
    ):
        inference._next_inference_batch_synchronized(
            iter(()),
            SimpleNamespace(enabled=True),
        )


def test_dataloader_iterator_initialization_failure_is_synchronized(monkeypatch):
    from runner import inference

    class BrokenLoader:
        def __iter__(self):
            raise OSError("worker process unavailable")

    def gather(output, local_error):
        output[:] = [local_error, ""]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="worker process unavailable",
    ):
        inference._create_dataloader_iterator_synchronized(
            BrokenLoader(),
            SimpleNamespace(enabled=True),
        )


def test_rank_lifecycle_failure_is_synchronized(monkeypatch):
    from runner import inference

    def gather(output, local_error):
        output[:] = [local_error, "[Rank 1] seed cleanup failed: CUDA error"]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="seed cleanup failed",
    ):
        inference._run_rank_stage_synchronized(
            lambda: None,
            stage="seed cleanup",
            foldcp_config=SimpleNamespace(enabled=True),
        )


def test_foldcp_runtime_config_guard_accepts_matching_compute_configs(monkeypatch):
    from runner import inference

    control_group = object()
    configs = SimpleNamespace(
        model_name="opendde_v1",
        dtype="bf16",
        triangle_multiplicative="torch",
        triangle_attention="torch",
        deterministic=True,
        enable_tf32=False,
    )

    def gather(output, signature, *, group):
        assert group is control_group
        output[:] = [signature, dict(signature)]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    inference._validate_foldcp_runtime_config_consistency(
        configs,
        SimpleNamespace(enabled=True),
        control_group,
    )


def test_foldcp_runtime_config_guard_rejects_rank_local_dtype_resolution(
    monkeypatch,
):
    from runner import inference

    control_group = object()
    configs = SimpleNamespace(
        model_name="opendde_v1",
        dtype="bf16",
        triangle_multiplicative="torch",
        triangle_attention="torch",
        deterministic=True,
        enable_tf32=False,
    )

    def gather(output, signature, *, group):
        assert group is control_group
        incompatible = dict(signature)
        incompatible["dtype"] = "fp32"
        incompatible["digest"] = "different-runtime-config"
        output[:] = [signature, incompatible]

    monkeypatch.setattr(inference.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="different inference compute configurations",
    ) as error:
        inference._validate_foldcp_runtime_config_consistency(
            configs,
            SimpleNamespace(enabled=True),
            control_group,
        )

    assert "rank 0" in str(error.value)
    assert "rank 1" in str(error.value)
    assert "'dtype': 'fp32'" in str(error.value)


def test_foldcp_runtime_config_signature_includes_ring_tuning_environment(
    monkeypatch,
):
    from runner import inference

    configs = SimpleNamespace(dtype="bf16")
    monkeypatch.delenv("OPENDDE_FOLDCP_TRIATT_WRAP_ROW_CHUNK", raising=False)
    default_signature = inference._foldcp_runtime_config_signature(configs)

    monkeypatch.setenv("OPENDDE_FOLDCP_TRIATT_WRAP_ROW_CHUNK", "17")
    tuned_signature = inference._foldcp_runtime_config_signature(configs)

    assert tuned_signature["digest"] != default_signature["digest"]
    assert tuned_signature["compute_environment"] == {
        "OPENDDE_FOLDCP_TRIATT_WRAP_ROW_CHUNK": "17"
    } | {
        key: value
        for key, value in default_signature["compute_environment"].items()
        if key != "OPENDDE_FOLDCP_TRIATT_WRAP_ROW_CHUNK"
    }


def test_runner_initialization_failure_is_synchronized_after_process_group(
    monkeypatch,
):
    from runner import inference

    control_group = object()

    def gather(output, local_error, *, group):
        assert group is control_group
        output[:] = [
            local_error,
            "[Rank 1] checkpoint loading failed: CUDA out of memory",
        ]

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="checkpoint loading failed: CUDA out of memory",
    ):
        inference._run_runner_initialization_stage(
            lambda: None,
            stage="checkpoint loading",
            foldcp_config=SimpleNamespace(enabled=True),
            world_control_group=control_group,
        )


def test_prediction_batch_preparation_failure_is_synchronized(monkeypatch):
    from runner import inference

    def gather(output, local_error):
        output[:] = [local_error, ""]

    runner = SimpleNamespace(
        update_model_configs=lambda _configs: pytest.fail(
            "invalid metadata reached model config update"
        )
    )
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    with pytest.raises(
        inference.FoldCPJobCoordinationError,
        match="Missing required inference dimensions",
    ):
        inference._run_rank_stage_synchronized(
            lambda: inference._prepare_prediction_batch(
                runner,
                SimpleNamespace(),
                {
                    "sample_index": 0,
                    "N_token": torch.tensor(4),
                    "N_atom": torch.tensor(8),
                    "N_msa": torch.tensor(1),
                    "input_feature_dict": {},
                },
                seed=7,
            ),
            stage="model batch preparation",
            foldcp_config=SimpleNamespace(enabled=True),
        )


def test_batch_cleanup_does_not_replace_active_error(monkeypatch, caplog):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "cleanup_device_memory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    inference._cleanup_batch_synchronized(
        torch.device("cpu"),
        SimpleNamespace(enabled=False),
        active_error=ValueError("model failed"),
    )

    assert "preserving active ValueError" in caplog.text


def test_batch_cleanup_uses_remote_active_error_to_avoid_rank_split(
    monkeypatch, caplog
):
    from runner import inference

    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        inference, "cleanup_device_memory", lambda *args, **kwargs: None
    )

    def gather(output, local_status):
        output[:] = [
            {"has_active_error": True, "cleanup_error": "rank 0 cleanup failed"},
            local_status,
        ]

    monkeypatch.setattr(inference.dist, "all_gather_object", gather)

    inference._cleanup_batch_synchronized(
        torch.device("cpu"),
        SimpleNamespace(enabled=True),
        active_error=None,
    )

    assert "preserving active error from another rank" in caplog.text


def test_batch_cleanup_failure_propagates_without_active_error(monkeypatch):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "cleanup_device_memory",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        inference._cleanup_batch_synchronized(
            torch.device("cpu"),
            SimpleNamespace(enabled=False),
            active_error=None,
        )


@pytest.mark.parametrize(
    ("world_rank", "is_non_output_rank"),
    [(0, False), (1, True)],
)
def test_only_rank_zero_is_the_1xp_output_rank(
    monkeypatch, world_rank, is_non_output_rank
):
    from opendde.model.modules import confidence
    from opendde.model import opendde

    monkeypatch.setenv("OPENDDE_FOLDCP_MODE", "distributed")
    monkeypatch.setenv("OPENDDE_FOLDCP_SIZE_CP", "2")
    monkeypatch.setattr(confidence.dist, "is_available", lambda: True)
    monkeypatch.setattr(confidence.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(confidence.dist, "get_rank", lambda: world_rank)

    assert confidence.ConfidenceHead._foldcp_is_non_output_rank() is is_non_output_rank
    assert opendde.OpenDDE._foldcp_is_non_output_rank() is is_non_output_rank


def test_foldcp_mesh_topology_preflight_uses_registered_gloo_group(monkeypatch):
    from opendde.distributed.foldcp import comm
    from opendde.distributed.foldcp import mesh as mesh_module
    from opendde.distributed.foldcp.config import FoldCPConfig

    control_group = object()
    gather_groups = []
    mesh_module.clear_foldcp_process_mesh_cache()
    comm.unregister_foldcp_cpu_control_group()
    monkeypatch.setattr(mesh_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        mesh_module.dist,
        "get_backend",
        lambda group: (
            mesh_module.dist.Backend.GLOO
            if group is control_group
            else mesh_module.dist.Backend.NCCL
        ),
    )
    monkeypatch.setattr(
        mesh_module.dist,
        "get_world_size",
        lambda _group=None: 2,
    )
    monkeypatch.setattr(mesh_module.dist, "get_rank", lambda: 0)

    def gather(output, topology, *, group):
        gather_groups.append(group)
        output[:] = [topology, topology]

    monkeypatch.setattr(mesh_module.dist, "all_gather_object", gather)
    monkeypatch.setattr(mesh_module.dist, "new_group", lambda _ranks: object())

    try:
        comm.register_foldcp_cpu_control_group(control_group)
        mesh = mesh_module.FoldCPProcessMesh.create(
            FoldCPConfig.from_runtime_args(
                mode="distributed",
                size_dp=1,
                size_cp=2,
            )
        )
    finally:
        mesh_module.clear_foldcp_process_mesh_cache()
        comm.unregister_foldcp_cpu_control_group(control_group)

    assert mesh.layout.shape == (1, 2)
    assert gather_groups == [control_group]


def test_foldcp_process_mesh_rejects_direct_legacy_2x2_config():
    from opendde.distributed.foldcp.config import FoldCPConfig
    from opendde.distributed.foldcp.mesh import FoldCPProcessMesh

    # Direct dataclass construction deliberately bypasses from_runtime_args().
    # The mesh library boundary must still reject the removed topology before
    # inspecting or creating any distributed process groups.
    config = FoldCPConfig(mode="distributed", size_dp=2, size_cp=2)

    with pytest.raises(ValueError, match="foldcp_size_dp must be 1"):
        FoldCPProcessMesh.create(config)


def test_foldcp_process_mesh_is_cached(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from opendde.distributed.foldcp import mesh as mesh_module

    mesh_module.clear_foldcp_process_mesh_cache()
    groups = []

    monkeypatch.setattr(mesh_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(mesh_module.dist, "get_rank", lambda: 0)

    topology_gathers = []

    def gather_topology(output, topology):
        topology_gathers.append(topology)
        output[:] = [topology] * 2

    monkeypatch.setattr(mesh_module.dist, "all_gather_object", gather_topology)

    def new_group(ranks):
        group = tuple(ranks)
        groups.append(group)
        return group

    monkeypatch.setattr(mesh_module.dist, "new_group", new_group)
    config = FoldCPConfig.from_runtime_args(
        mode="distributed",
        size_dp=1,
        size_cp=2,
    )
    first = mesh_module.FoldCPProcessMesh.create(config)
    second = mesh_module.FoldCPProcessMesh.create(config)

    assert second is first
    assert groups == []
    assert first.group_2d is mesh_module.dist.group.WORLD
    assert first.group_row is first.group_2d
    with pytest.raises(RuntimeError, match="column communication is unavailable"):
        _ = first.group_col
    assert topology_gathers == [{"size_dp": 1, "size_cp": 2, "mesh_shape": (1, 2)}]
    mesh_module.clear_foldcp_process_mesh_cache()


def test_foldcp_mesh_teardown_also_clears_communication_cache():
    from opendde.distributed.foldcp import comm
    from opendde.distributed.foldcp import mesh as mesh_module

    comm._NCCL_STATUS_TENSORS[(123, 0)] = torch.empty((), dtype=torch.int32)
    mesh_module._PROCESS_MESH_TOPOLOGY[123] = (1, 2)

    mesh_module.clear_foldcp_process_mesh_cache()

    assert comm._NCCL_STATUS_TENSORS == {}
    assert mesh_module._PROCESS_MESH_TOPOLOGY == {}


def test_foldcp_process_mesh_cache_ignores_non_topology_config(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from opendde.distributed.foldcp import mesh as mesh_module

    mesh_module.clear_foldcp_process_mesh_cache()
    groups = []
    monkeypatch.setattr(mesh_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(mesh_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        mesh_module.dist,
        "all_gather_object",
        lambda output, topology: output.__setitem__(slice(None), [topology] * 2),
    )
    monkeypatch.setattr(
        mesh_module.dist,
        "new_group",
        lambda ranks: groups.append(tuple(ranks)) or object(),
    )

    first = mesh_module.FoldCPProcessMesh.create(
        FoldCPConfig.from_runtime_args(
            mode="distributed", size_dp=1, size_cp=2, metrics_jsonl="first.jsonl"
        )
    )
    second = mesh_module.FoldCPProcessMesh.create(
        FoldCPConfig.from_runtime_args(
            mode="distributed", size_dp=1, size_cp=2, metrics_jsonl="second.jsonl"
        )
    )

    assert second is first
    assert groups == []
    assert first.group_2d is mesh_module.dist.group.WORLD
    assert first.group_row is first.group_2d
    mesh_module.clear_foldcp_process_mesh_cache()


def test_foldcp_process_mesh_prewarms_every_1xp_peer_route(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from opendde.distributed.foldcp.layout import FoldCP2DLayout
    from opendde.distributed.foldcp import mesh as mesh_module

    group = object()
    events = []
    send = torch.zeros(())
    receive = torch.empty(())

    def run_action(_action, *, group, description):
        events.append(("collective", group, description))
        return send, receive

    class Peer:
        def __init__(self, selected_group, send_rank, receive_rank):
            events.append(("peer", selected_group, send_rank, receive_rank))

        def exchange(self, source, *, to_recv):
            assert source is send
            assert to_recv is receive
            return to_recv

    monkeypatch.setattr(mesh_module, "run_group_rank_action_synchronized", run_action)
    monkeypatch.setattr(mesh_module, "One2OneComm", Peer)
    monkeypatch.setattr(mesh_module.torch.cuda, "current_device", lambda: 0)
    mesh = mesh_module.FoldCPProcessMesh(
        config=FoldCPConfig.from_runtime_args(mode="distributed", size_dp=1, size_cp=4),
        layout=FoldCP2DLayout((1, 4)),
        group_2d=group,
        group_row=group,
        cp_global_ranks=(0, 1, 2, 3),
        cp_rank=1,
        coord=(0, 1),
    )

    mesh.prewarm_communications()

    assert events == [
        (
            "collective",
            group,
            "Fold-CP NCCL communication warmup allocation",
        ),
        ("peer", group, 2, 0),
        ("peer", group, 3, 3),
        ("peer", group, 0, 2),
    ]


def test_foldcp_process_mesh_rejects_topology_change_without_collective(
    monkeypatch,
):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from opendde.distributed.foldcp import mesh as mesh_module

    mesh_module.clear_foldcp_process_mesh_cache()
    world_identity = id(mesh_module.dist.group.WORLD)
    mesh_module._PROCESS_MESH_TOPOLOGY[world_identity] = (1, 4)
    collective_calls = []
    new_group_calls = []
    monkeypatch.setattr(mesh_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        mesh_module.dist,
        "all_gather_object",
        lambda *args, **kwargs: collective_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        mesh_module.dist,
        "new_group",
        lambda ranks: new_group_calls.append(ranks),
    )

    changed = FoldCPConfig.from_runtime_args(
        mode="distributed",
        size_dp=1,
        size_cp=2,
    )
    with pytest.raises(RuntimeError, match="topology is immutable"):
        mesh_module.FoldCPProcessMesh.create(changed)

    assert collective_calls == []
    assert new_group_calls == []
    mesh_module.clear_foldcp_process_mesh_cache()


def test_foldcp_process_mesh_rejects_topology_mismatch_before_new_group(
    monkeypatch,
):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from opendde.distributed.foldcp import mesh as mesh_module

    mesh_module.clear_foldcp_process_mesh_cache()
    monkeypatch.setattr(mesh_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(mesh_module.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(mesh_module.dist, "get_rank", lambda: 0)

    def gather(output, local_topology):
        output[:] = [
            local_topology,
            {"size_dp": 2, "size_cp": 2, "mesh_shape": (1, 2)},
            local_topology,
            local_topology,
        ]

    new_group_calls = []
    monkeypatch.setattr(mesh_module.dist, "all_gather_object", gather)
    monkeypatch.setattr(
        mesh_module.dist,
        "new_group",
        lambda ranks: new_group_calls.append(ranks),
    )

    config = FoldCPConfig.from_runtime_args(
        mode="distributed",
        size_dp=1,
        size_cp=4,
    )
    with pytest.raises(RuntimeError, match="different Fold-CP mesh topologies"):
        mesh_module.FoldCPProcessMesh.create(config)

    assert new_group_calls == []
