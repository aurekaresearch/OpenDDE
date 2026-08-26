# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import inspect
import os
import random
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opendde.config.inference import (
    apply_runtime_compatibility,
    build_inference_config,
    update_gpu_compatible_configs,
    validate_triangle_kernel_runtime,
    validate_triangle_kernels,
)
from opendde.config.model_base import configs as configs_base
from opendde.config.schema import OpenDDEConfig
from opendde.utils.environment import CuEquivarianceRuntimeStatus


def _runtime_status(
    reason: str | None = "torch fallback",
    *,
    requires_cc7_fallback: bool = False,
) -> CuEquivarianceRuntimeStatus:
    return CuEquivarianceRuntimeStatus(
        unavailable_reason=reason,
        requires_cc7_fallback=requires_cc7_fallback,
    )


def test_build_inference_config_applies_model_specific_defaults():
    cfg = build_inference_config(fill_required_with_null=True)

    assert cfg.model_name == "opendde_v1"
    assert cfg.device == "auto"
    assert cfg.c_z == 384
    assert cfg.no_bins == 96
    assert cfg.model.N_cycle == 10
    assert cfg.model.msa_module.c_m == 128
    assert cfg.model.template_embedder.n_blocks == 2
    assert cfg.sample_diffusion.N_step == 200
    assert cfg.confidence.distogram.no_bins == 96
    assert cfg.need_atom_confidence is True
    assert cfg.seed_batch_size == 1


def test_legacy_config_dict_uses_new_schema_defaults():
    legacy_config = build_inference_config(fill_required_with_null=True).model_dump()
    legacy_config.pop("device")
    legacy_config.pop("seed_batch_size")

    config = OpenDDEConfig.model_validate(legacy_config)
    assert config.device == "auto"
    assert config.seed_batch_size == 1


def test_build_inference_config_keeps_cli_overrides_highest_priority():
    cfg = build_inference_config(
        arg_str=(
            "--model_name opendde_v1 "
            "--model.N_cycle 3 "
            "--sample_diffusion.N_step 7 "
            "--triangle_attention torch"
        ),
        fill_required_with_null=True,
    )

    assert cfg.model.N_cycle == 3
    assert cfg.sample_diffusion.N_step == 7
    assert cfg.triangle_attention == "torch"
    assert cfg.c_z == 384


def test_build_inference_config_does_not_mutate_base_defaults():
    build_inference_config(fill_required_with_null=True)

    assert configs_base["c_z"] == 384
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["model"]["msa_module"]["c_m"] == 128
    assert configs_base["model"]["template_embedder"]["n_blocks"] == 2
    assert configs_base["confidence"]["distogram"]["min_bin"] == 2.25
    assert configs_base["confidence"]["distogram"]["max_bin"] == 25.75
    assert configs_base["confidence"]["distogram"]["no_bins"] == 96


def test_get_default_runner_config_build_does_not_mutate_base_defaults(monkeypatch):
    from runner import batch_inference

    class DummyRunner:
        def __init__(self, cfg, *, foldcp_config=None):
            self.configs = cfg
            self.foldcp_config = foldcp_config

    monkeypatch.setattr(batch_inference, "InferenceRunner", DummyRunner)

    runner = batch_inference.get_default_runner(
        seeds=[101],
        seed_batch_size=2,
        n_cycle=2,
        n_step=3,
        n_sample=1,
        dtype="fp32",
        use_msa=False,
        trimul_kernel="torch",
        triatt_kernel="torch",
        enable_cache=False,
        enable_fusion=False,
    )

    assert runner.configs.c_z == 384
    assert runner.configs.model.N_cycle == 2
    assert runner.configs.sample_diffusion.N_step == 3
    assert runner.configs.need_atom_confidence is True
    assert runner.configs.seed_batch_size == 2
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["confidence"]["distogram"]["no_bins"] == 96


def test_batch_device_parameters_are_keyword_only_and_last():
    from runner import batch_inference

    for function in (
        batch_inference.get_default_runner,
        batch_inference.inference_jsons,
    ):
        parameter = list(inspect.signature(function).parameters.values())[-1]
        assert parameter.name == "device"
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_runner_run_config_build_does_not_mutate_base_defaults(monkeypatch):
    from runner import inference

    captured_configs = []
    monkeypatch.setattr(inference, "parse_sys_args", lambda: "")
    monkeypatch.setattr(inference, "main", lambda cfg: captured_configs.append(cfg))

    inference.run()

    assert captured_configs[0].c_z == 384
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["model"]["msa_module"]["c_m"] == 128


def test_validate_triangle_kernels_rejects_unknown_values():
    validate_triangle_kernels("auto", "cuequivariance")
    validate_triangle_kernels("torch", "cuequivariance")
    with pytest.raises(ValueError):
        validate_triangle_kernels("unsupported", "torch")
    with pytest.raises(ValueError):
        validate_triangle_kernels("torch", "unsupported")


def test_apply_runtime_compatibility_consumes_resolved_device(monkeypatch):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(
        arg_str="--triangle_attention auto --triangle_multiplicative auto",
        fill_required_with_null=True,
    )
    device = torch.device("cpu")
    seen_devices = []

    def inspect_runtime(resolved_device, *, probe_packages):
        seen_devices.append((resolved_device, probe_packages))
        return _runtime_status()

    monkeypatch.setattr(
        inference_config,
        "get_cuequivariance_runtime_status",
        inspect_runtime,
    )
    monkeypatch.setattr(
        inference_config,
        "select_torch_device",
        lambda *args, **kwargs: pytest.fail("device must not be resolved twice"),
    )

    result = apply_runtime_compatibility(cfg, device)

    assert seen_devices == [(device, True)]
    assert result.triangle_attention == "torch"
    assert result.triangle_multiplicative == "torch"


def test_explicit_torch_kernels_skip_optional_package_probe(monkeypatch):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(
        arg_str="--triangle_attention torch --triangle_multiplicative torch",
        fill_required_with_null=True,
    )
    probe_values = []

    def inspect_runtime(device, *, probe_packages):
        probe_values.append(probe_packages)
        return _runtime_status()

    monkeypatch.setattr(
        inference_config,
        "get_cuequivariance_runtime_status",
        inspect_runtime,
    )

    apply_runtime_compatibility(cfg, torch.device("cpu"))

    assert probe_values == [False]


def test_update_gpu_compatible_configs_resolves_device_once(monkeypatch):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(fill_required_with_null=True)
    device = torch.device("cpu")
    select_calls = []
    apply_calls = []

    def select(requested_device, *, local_rank):
        select_calls.append((requested_device, local_rank))
        return device

    def apply(configs, resolved_device):
        apply_calls.append((configs, resolved_device))
        return configs

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(inference_config, "select_torch_device", select)
    monkeypatch.setattr(inference_config, "apply_runtime_compatibility", apply)

    assert update_gpu_compatible_configs(cfg) is cfg
    assert select_calls == [("auto", 3)]
    assert apply_calls == [(cfg, device)]


def test_cc7_runtime_enforces_fp32_and_torch_kernels(monkeypatch):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(
        arg_str=(
            "--dtype bf16 --triangle_attention cuequivariance "
            "--triangle_multiplicative cuequivariance"
        ),
        fill_required_with_null=True,
    )
    monkeypatch.setattr(
        inference_config,
        "get_cuequivariance_runtime_status",
        lambda device, **kwargs: _runtime_status(requires_cc7_fallback=True),
    )

    result = apply_runtime_compatibility(cfg, torch.device("cuda:0"))

    assert result.dtype == "fp32"
    assert result.triangle_attention == "torch"
    assert result.triangle_multiplicative == "torch"


def test_explicit_cuequivariance_reports_runtime_reason():
    cfg = build_inference_config(
        arg_str=(
            "--triangle_attention cuequivariance "
            "--triangle_multiplicative cuequivariance"
        ),
        fill_required_with_null=True,
    )
    status = _runtime_status("supported only on Linux x86_64")

    with pytest.raises(RuntimeError, match="supported only on Linux x86_64"):
        validate_triangle_kernel_runtime(cfg, status)


def test_get_default_runner_passes_foldcp_config_once(monkeypatch):
    from runner import batch_inference

    captured = {}

    class DummyRunner:
        def __init__(self, cfg, *, foldcp_config=None):
            captured["foldcp"] = foldcp_config
            self.configs = cfg
            self.foldcp_config = foldcp_config

    monkeypatch.setattr(batch_inference, "InferenceRunner", DummyRunner)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    runner = batch_inference.get_default_runner(
        use_tfg_guidance=True,
        foldcp_mode="distributed",
        foldcp_size_dp=1,
        foldcp_size_cp=4,
        foldcp_devices="0,1,2,3",
        foldcp_metrics_jsonl="metrics.jsonl",
    )

    assert runner.configs.sample_diffusion.guidance["enable"] is True
    assert captured["foldcp"] is runner.foldcp_config
    assert runner.foldcp_config.mode == "distributed"
    assert runner.foldcp_config.size_cp == 4


def test_foldcp_config_validation_is_independent_of_process_environment(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig

    monkeypatch.setenv("WORLD_SIZE", "8")

    assert FoldCPConfig().validate().mode == "single"


@pytest.mark.parametrize(
    ("mode", "size_dp", "size_cp", "world_size"),
    [
        ("single", 1, 1, 1),
        ("single", 2, 1, 2),
        ("single", 3, 1, 3),
        ("single", 4, 1, 4),
        ("distributed", 1, 2, 2),
        ("distributed", 1, 3, 3),
        ("distributed", 1, 4, 4),
    ],
)
def test_matching_world_size_reaches_device_selection(
    monkeypatch, mode, size_dp, size_cp, world_size
):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    class DeviceSelectionReached(Exception):
        pass

    runner = object.__new__(inference.InferenceRunner)
    runner.foldcp_config = FoldCPConfig.from_runtime_args(
        mode=mode,
        size_dp=size_dp,
        size_cp=size_cp,
    )
    runner.configs = SimpleNamespace(device="auto")
    runner.print = lambda _message: None
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=world_size, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DeviceSelectionReached),
    )

    with pytest.raises(DeviceSelectionReached):
        runner.init_env()


def test_world_size_mismatch_fails_before_device_or_model_selection(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner.foldcp_config = FoldCPConfig.from_runtime_args(
        mode="single", size_dp=4, size_cp=1
    )
    runner.configs = SimpleNamespace(device="auto")
    runner.print = lambda _message: None
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda *_args, **_kwargs: pytest.fail(
            "world-size mismatch must fail before device or model selection"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=r"WORLD_SIZE=4.*foldcp_size_dp=4.*foldcp_size_cp=1.*WORLD_SIZE=2",
    ):
        runner.init_env()


@pytest.mark.parametrize(
    ("seed_batch_size", "num_workers", "size_cp", "n_model_seed", "message"),
    [
        (0, 0, 1, 1, "seed_batch_size must be >= 1"),
        (2, 0, 2, 1, "normal P=1 model path only"),
        (2, 1, 1, 1, "requires num_workers=0"),
        (2, 0, 1, 2, "requires model.N_model_seed=1"),
    ],
)
def test_invalid_seed_batch_fails_before_device_selection(
    monkeypatch, seed_batch_size, num_workers, size_cp, n_model_seed, message
):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner.foldcp_config = FoldCPConfig.from_runtime_args(
        mode="distributed" if size_cp > 1 else "single",
        size_dp=1,
        size_cp=size_cp,
    )
    runner.configs = SimpleNamespace(
        device="auto",
        seed_batch_size=seed_batch_size,
        num_workers=num_workers,
        model=SimpleNamespace(N_model_seed=n_model_seed),
    )
    runner.print = lambda _message: None
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=size_cp, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid seed batching must fail before device selection"
        ),
    )

    with pytest.raises(ValueError, match=message):
        runner.init_env()


def test_seed_batch_rejects_tfg_before_device_selection(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner.foldcp_config = FoldCPConfig.from_runtime_args(
        mode="single", size_dp=1, size_cp=1
    )
    runner.configs = SimpleNamespace(
        device="auto",
        seed_batch_size=2,
        num_workers=0,
        sample_diffusion=SimpleNamespace(guidance={"enable": True}),
    )
    runner.print = lambda _message: None
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid seed batching must fail before device selection"
        ),
    )

    with pytest.raises(ValueError, match="Training-Free Guidance"):
        runner.init_env()


def test_predict_help_describes_seed_batching_and_sharding():
    from click.testing import CliRunner

    from runner import batch_inference

    result = CliRunner().invoke(
        batch_inference.predict,
        ["--help"],
        terminal_width=160,
    )

    assert result.exit_code == 0
    assert "--seed_batch_size" in result.output
    assert "Maximum seeds per rank-local model batch" in result.output
    assert "--foldcp_size_dp" in result.output
    assert "Number of seed-sharding ranks" in result.output
    assert "normal model path (including seed batching/sharding)" in result.output


def test_seed_assignment_is_rank_strided_without_loss_or_duplication():
    from runner.inference import _seed_batches_for_rank, _seeds_for_rank

    seeds = [101, 102, 103, 104, 105, 106]
    assignments = [_seeds_for_rank(seeds, 4, rank) for rank in range(4)]

    assert assignments == [[101, 105], [102, 106], [103], [104]]
    assert sorted(seed for lane in assignments for seed in lane) == seeds
    assert _seeds_for_rank(seeds, 1, 3) == seeds

    seeds = [101, 102, 103, 104, 105, 106, 107]
    batches = [
        _seed_batches_for_rank(seeds, 3, rank, seed_batch_size=2) for rank in range(3)
    ]
    assert batches == [
        [[101, 104], [107]],
        [[102, 105]],
        [[103, 106]],
    ]
    assert sorted(seed for rank in batches for batch in rank for seed in batch) == seeds


def test_local_seed_resolution_preserves_precedence(monkeypatch, tmp_path):
    from runner import inference

    input_path = tmp_path / "input.json"
    input_path.write_text(
        '[{"modelSeeds": [11, 12]}, {"modelSeeds": [31, 32]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(inference.random, "randint", lambda *_args: 41)

    configs = SimpleNamespace(input_json_path=str(input_path), seeds=[21, 22])
    assert inference._resolve_local_inference_seeds(configs) == [21, 22]

    configs.seeds = []
    assert inference._resolve_local_inference_seeds(configs) == [11, 12]

    input_path.write_text('[{"name": "no-seeds"}]', encoding="utf-8")
    assert inference._resolve_local_inference_seeds(configs) == [41]


def test_rank_zero_resolves_and_broadcasts_seeds_once(monkeypatch, tmp_path):
    from runner import inference

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"modelSeeds": [101, 102]}]', encoding="utf-8")
    broadcasts = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((list(status), src)),
    )

    seeds = inference._resolve_inference_seeds(
        SimpleNamespace(input_json_path=str(input_path), seeds=[]),
        size_dp=2,
    )

    assert seeds == [101, 102]
    assert broadcasts == [([(True, [101, 102], "")], 0)]


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ([101, 101], "requires unique seeds"),
        ([101], "requires at least one seed per lane"),
    ],
)
def test_rank_zero_broadcasts_seed_validation_failure(
    monkeypatch, tmp_path, seeds, message
):
    from runner import inference

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample"}]', encoding="utf-8")
    broadcasts = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((list(status), src)),
    )

    with pytest.raises(RuntimeError, match=message):
        inference._resolve_inference_seeds(
            SimpleNamespace(input_json_path=str(input_path), seeds=seeds),
            size_dp=2,
        )

    assert broadcasts[0][0][0][0] is False
    assert message in broadcasts[0][0][0][2]


def test_nonzero_rank_receives_seeds_without_opening_input(monkeypatch):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=1, local_rank=1),
    )
    monkeypatch.setattr(
        inference,
        "_resolve_local_inference_seeds",
        lambda _configs: pytest.fail("nonzero rank must not resolve seeds"),
    )

    def receive(status, src):
        assert src == 0
        status[0] = (True, [101, 102], "")

    monkeypatch.setattr(inference.dist, "broadcast_object_list", receive)

    assert inference._resolve_inference_seeds(object(), size_dp=2) == [101, 102]


def test_rank_zero_preprocesses_once_and_broadcasts_path(monkeypatch):
    from runner import batch_inference

    calls = []
    broadcasts = []
    monkeypatch.setattr(
        batch_inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        batch_inference,
        "preprocess_input",
        lambda input_json, **_kwargs: calls.append(input_json) or "updated.json",
    )
    monkeypatch.setattr(
        batch_inference.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((list(status), src)),
    )

    result = batch_inference._preprocess_input_for_runner("input.json")

    assert result == "updated.json"
    assert calls == ["input.json"]
    assert broadcasts == [([(True, "updated.json", "")], 0)]


def test_rank_zero_broadcasts_preprocessing_failure(monkeypatch):
    from runner import batch_inference

    broadcasts = []
    monkeypatch.setattr(
        batch_inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        batch_inference,
        "preprocess_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("preprocessing unavailable")
        ),
    )
    monkeypatch.setattr(
        batch_inference.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((list(status), src)),
    )

    with pytest.raises(RuntimeError, match="preprocessing unavailable"):
        batch_inference._preprocess_input_for_runner("input.json")

    assert broadcasts == [([(False, "", "OSError: preprocessing unavailable")], 0)]


def test_nonzero_rank_receives_preprocessing_failure(monkeypatch):
    from runner import batch_inference

    monkeypatch.setattr(
        batch_inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=1, local_rank=1),
    )
    monkeypatch.setattr(
        batch_inference,
        "preprocess_input",
        lambda *_args, **_kwargs: pytest.fail("nonzero rank must not preprocess"),
    )

    def receive(status, src):
        assert src == 0
        status[0] = (False, "", "OSError: preprocessing unavailable")

    monkeypatch.setattr(batch_inference.dist, "broadcast_object_list", receive)

    with pytest.raises(RuntimeError, match="preprocessing unavailable"):
        batch_inference._preprocess_input_for_runner("input.json")


def test_directory_inputs_are_processed_in_sorted_order(monkeypatch, tmp_path):
    from runner import batch_inference

    (tmp_path / "b.json").write_text("[]", encoding="utf-8")
    (tmp_path / "a.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")
    processed = []
    inferred = []
    closed = []
    runner = SimpleNamespace(configs={}, close=lambda: closed.append(True))
    monkeypatch.setattr(
        batch_inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: runner,
    )
    monkeypatch.setattr(
        batch_inference,
        "_preprocess_input_for_runner",
        lambda input_json, **_kwargs: processed.append(input_json) or input_json,
    )
    monkeypatch.setattr(
        batch_inference,
        "infer_predict",
        lambda _runner, configs: inferred.append(configs["input_json_path"]),
    )
    monkeypatch.setattr(batch_inference.tqdm, "tqdm", lambda values: values)

    batch_inference.inference_jsons(str(tmp_path))

    expected = [str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    assert processed == expected
    assert inferred == expected
    assert closed == [True]


def test_distributed_preprocessing_failure_escapes_and_closes_runner(
    monkeypatch, tmp_path
):
    from runner import batch_inference

    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    closed = []
    runner = SimpleNamespace(configs={}, close=lambda: closed.append(True))
    monkeypatch.setattr(
        batch_inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: runner,
    )
    monkeypatch.setattr(
        batch_inference,
        "_preprocess_input_for_runner",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("preprocessing failed")
        ),
    )
    monkeypatch.setattr(batch_inference.tqdm, "tqdm", lambda values: values)

    with pytest.raises(RuntimeError, match="preprocessing failed"):
        batch_inference.inference_jsons(str(input_path))

    assert closed == [True]


def test_inference_dataloader_keeps_all_records_in_order(monkeypatch):
    from torch.utils.data import SequentialSampler

    from opendde.data.inference import infer_dataloader

    records = ["first", "second"]
    monkeypatch.setattr(
        infer_dataloader,
        "InferenceDataset",
        lambda configs: records,
    )
    configs = SimpleNamespace(num_workers=0)

    loaders = [infer_dataloader.get_inference_dataloader(configs) for _ in range(2)]

    assert all(isinstance(loader.sampler, SequentialSampler) for loader in loaders)
    assert [list(loader.sampler) for loader in loaders] == [[0, 1], [0, 1]]


def test_get_default_runner_uses_shared_kalign_resolver(monkeypatch):
    from runner import batch_inference

    calls = []

    class DummyRunner:
        def __init__(self, cfg, *, foldcp_config=None):
            self.configs = cfg

    monkeypatch.setattr(batch_inference, "InferenceRunner", DummyRunner)
    monkeypatch.setattr(
        batch_inference.kalign,
        "resolve_kalign_binary",
        lambda binary_path: calls.append(binary_path) or "/tools/kalign",
    )

    runner = batch_inference.get_default_runner(
        use_template=True,
        kalign_binary_path="custom-kalign",
    )

    assert calls == ["custom-kalign"]
    assert runner.configs.data.template.kalign_binary_path == "/tools/kalign"


def test_download_inference_assets_single_process_downloads_directly(monkeypatch):
    from runner import inference

    downloads = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference, "download_inference_cache", lambda configs: downloads.append(configs)
    )
    monkeypatch.setattr(
        inference.dist,
        "broadcast_object_list",
        lambda *args, **kwargs: pytest.fail(
            "single-process download must not broadcast"
        ),
    )
    configs = object()

    inference._download_inference_assets(configs)

    assert downloads == [configs]


def test_download_inference_assets_rank_zero_broadcasts_success(monkeypatch):
    from runner import inference

    downloads = []
    broadcasts = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        inference, "download_inference_cache", lambda configs: downloads.append(configs)
    )
    monkeypatch.setattr(
        inference.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((list(status), src)),
    )
    configs = object()

    inference._download_inference_assets(configs)

    assert downloads == [configs]
    assert broadcasts == [([(True, "")], 0)]


def test_download_inference_assets_nonzero_rank_waits_for_success(monkeypatch):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=1, local_rank=1),
    )
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        inference,
        "download_inference_cache",
        lambda configs: pytest.fail("nonzero rank must not download assets"),
    )

    def receive_success(status, src):
        assert src == 0
        status[0] = (True, "")

    monkeypatch.setattr(inference.dist, "broadcast_object_list", receive_success)

    inference._download_inference_assets(object())


def test_download_inference_assets_broadcasts_rank_zero_failure(monkeypatch):
    from runner import inference

    broadcasts = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)

    def fail_download(configs):
        raise OSError("cache unavailable")

    monkeypatch.setattr(inference, "download_inference_cache", fail_download)
    monkeypatch.setattr(
        inference.dist,
        "broadcast_object_list",
        lambda status, src: broadcasts.append((list(status), src)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        inference._download_inference_assets(object())

    expected_message = (
        "Inference asset preparation failed on rank 0: OSError: cache unavailable"
    )
    assert str(exc_info.value) == expected_message
    assert broadcasts == [([(False, "OSError: cache unavailable")], 0)]


def test_download_inference_assets_nonzero_rank_raises_rank_zero_failure(monkeypatch):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=1, local_rank=1),
    )
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(
        inference,
        "download_inference_cache",
        lambda configs: pytest.fail("nonzero rank must not download assets"),
    )

    def receive_failure(status, src):
        assert src == 0
        status[0] = (False, "OSError: cache unavailable")

    monkeypatch.setattr(inference.dist, "broadcast_object_list", receive_failure)

    with pytest.raises(RuntimeError) as exc_info:
        inference._download_inference_assets(object())

    assert str(exc_info.value) == (
        "Inference asset preparation failed on rank 0: OSError: cache unavailable"
    )


def test_inference_runner_applies_foldcp_and_runtime_once(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    cfg = build_inference_config(fill_required_with_null=True)
    foldcp = FoldCPConfig.from_runtime_args()
    device = torch.device("cpu")
    events = []

    def apply_foldcp(configs, supplied_foldcp):
        assert supplied_foldcp is foldcp
        events.append("foldcp")
        return configs

    def select_device(requested_device, *, local_rank):
        assert requested_device == "auto"
        assert local_rank == 2
        events.append("select")
        return device

    def apply_runtime(configs, resolved_device):
        assert resolved_device is device
        events.append("runtime")
        return configs

    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=2),
    )
    monkeypatch.setattr(inference, "apply_foldcp_config", apply_foldcp)
    monkeypatch.setattr(inference, "select_torch_device", select_device)
    monkeypatch.setattr(inference, "apply_runtime_compatibility", apply_runtime)
    monkeypatch.setattr(
        inference,
        "download_inference_cache",
        lambda configs: events.append("download"),
    )
    monkeypatch.setattr(
        inference, "FoldCPBenchmarkRecorder", lambda *args, **kwargs: object()
    )
    for method_name in ("init_basics", "init_model", "load_checkpoint"):
        monkeypatch.setattr(
            inference.InferenceRunner,
            method_name,
            lambda self, name=method_name: events.append(name),
        )
    monkeypatch.setattr(
        inference.InferenceRunner,
        "init_dumper",
        lambda self, **kwargs: events.append("init_dumper"),
    )

    runner = inference.InferenceRunner(cfg, foldcp_config=foldcp)

    assert runner.device is device
    assert events.count("foldcp") == 1
    assert events.count("select") == 1
    assert events.count("runtime") == 1
    assert events.index("runtime") < events.index("download")
    assert events.index("download") < events.index("init_model")
    assert events[-1] == "foldcp"


def test_failed_runner_does_not_publish_foldcp_environment(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    cfg = build_inference_config(fill_required_with_null=True)
    foldcp = FoldCPConfig.from_runtime_args()
    monkeypatch.setattr(
        inference, "FoldCPBenchmarkRecorder", lambda *args, **kwargs: object()
    )

    def fail_init(self):
        raise RuntimeError("invalid runtime")

    monkeypatch.setattr(inference.InferenceRunner, "init_env", fail_init)
    monkeypatch.setattr(
        inference,
        "apply_foldcp_config",
        lambda *args, **kwargs: pytest.fail("failed runner published Fold-CP state"),
    )

    with pytest.raises(RuntimeError, match="invalid runtime"):
        inference.InferenceRunner(cfg, foldcp_config=foldcp)


def _patch_cuda_distributed_runner(monkeypatch, inference, *, initialized):
    state = {"initialized": initialized}
    calls = {"init": [], "destroy": 0}

    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference, "FoldCPBenchmarkRecorder", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda requested_device, *, local_rank: torch.device("cuda:0"),
    )
    monkeypatch.setattr(
        inference,
        "apply_runtime_compatibility",
        lambda configs, device: configs,
    )
    monkeypatch.setattr(inference, "_download_inference_assets", lambda configs: None)
    monkeypatch.setattr(inference.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(inference.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(inference.dist, "get_backend", lambda: "nccl")

    def init_process_group(*, backend, timeout):
        calls["init"].append((backend, timeout))
        state["initialized"] = True

    def destroy_process_group():
        calls["destroy"] += 1
        state["initialized"] = False

    monkeypatch.setattr(inference.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(inference.dist, "destroy_process_group", destroy_process_group)
    return calls


def test_failed_runner_destroys_process_group_it_created(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    configs = build_inference_config(fill_required_with_null=True)
    foldcp = FoldCPConfig.from_runtime_args(mode="distributed", size_dp=1, size_cp=2)
    calls = _patch_cuda_distributed_runner(monkeypatch, inference, initialized=False)

    def fail_init_basics(self):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(inference.InferenceRunner, "init_basics", fail_init_basics)

    with pytest.raises(RuntimeError, match="initialization failed"):
        inference.InferenceRunner(configs, foldcp_config=foldcp)

    assert calls == {
        "init": [("nccl", inference._DISTRIBUTED_STARTUP_TIMEOUT)],
        "destroy": 1,
    }


def test_failed_runner_preserves_preinitialized_process_group(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    configs = build_inference_config(fill_required_with_null=True)
    foldcp = FoldCPConfig.from_runtime_args(mode="distributed", size_dp=1, size_cp=2)
    calls = _patch_cuda_distributed_runner(monkeypatch, inference, initialized=True)

    def fail_init_basics(self):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(inference.InferenceRunner, "init_basics", fail_init_basics)

    with pytest.raises(RuntimeError, match="initialization failed"):
        inference.InferenceRunner(configs, foldcp_config=foldcp)

    assert calls == {"init": [], "destroy": 0}


def test_runner_close_restores_foldcp_environment(monkeypatch):
    from opendde.distributed.foldcp.config import (
        FOLDCP_ENVIRONMENT_KEYS,
        FoldCPConfig,
    )
    from runner import inference

    configs = build_inference_config(fill_required_with_null=True)
    foldcp = FoldCPConfig.from_runtime_args()
    previous_value = "previous-mode"
    monkeypatch.setenv(FOLDCP_ENVIRONMENT_KEYS[0], previous_value)
    for key in FOLDCP_ENVIRONMENT_KEYS[1:]:
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(
        inference, "FoldCPBenchmarkRecorder", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(inference, "_download_inference_assets", lambda configs: None)
    for method_name in (
        "init_env",
        "init_basics",
        "init_model",
        "load_checkpoint",
        "init_dumper",
    ):
        monkeypatch.setattr(
            inference.InferenceRunner,
            method_name,
            lambda self, *args, **kwargs: None,
        )

    runner = inference.InferenceRunner(configs, foldcp_config=foldcp)

    assert os.environ[FOLDCP_ENVIRONMENT_KEYS[0]] == "single"
    runner.close()
    assert os.environ[FOLDCP_ENVIRONMENT_KEYS[0]] == previous_value
    assert all(key not in os.environ for key in FOLDCP_ENVIRONMENT_KEYS[1:])

    runner.close()
    assert os.environ[FOLDCP_ENVIRONMENT_KEYS[0]] == previous_value


def test_runner_close_retries_failed_process_group_destruction(monkeypatch):
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = True
    attempts = []

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)

    def destroy_process_group():
        attempts.append(None)
        if len(attempts) == 1:
            raise RuntimeError("destroy failed")

    monkeypatch.setattr(inference.dist, "destroy_process_group", destroy_process_group)

    runner.close()
    assert runner._owns_process_group

    runner.close()
    assert not runner._owns_process_group
    assert len(attempts) == 2


def test_main_passes_runner_canonical_config(monkeypatch):
    from runner import inference

    input_config = build_inference_config(fill_required_with_null=True)
    canonical_config = input_config.model_copy(deep=True)
    calls = []
    closed = []

    class DummyRunner:
        def __init__(self, configs):
            self.configs = canonical_config

        def close(self):
            closed.append(self)

    monkeypatch.setattr(inference, "InferenceRunner", DummyRunner)
    monkeypatch.setattr(
        inference,
        "infer_predict",
        lambda runner, configs: calls.append((runner, configs)),
    )

    inference.main(input_config)

    assert calls[0][1] is canonical_config
    assert closed == [calls[0][0]]


def test_main_closes_runner_when_inference_fails(monkeypatch):
    from runner import inference

    input_config = build_inference_config(fill_required_with_null=True)
    closed = []

    class DummyRunner:
        def __init__(self, configs):
            self.configs = configs

        def close(self):
            closed.append(self)

    monkeypatch.setattr(inference, "InferenceRunner", DummyRunner)
    monkeypatch.setattr(
        inference,
        "infer_predict",
        lambda runner, configs: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )

    with pytest.raises(RuntimeError, match="inference failed"):
        inference.main(input_config)

    assert len(closed) == 1


@pytest.mark.parametrize("seeds", [[101, 101], [101]])
def test_invalid_seed_parallel_request_fails_before_dataloader_or_forward(
    monkeypatch, tmp_path, seeds
):
    from runner import inference

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample"}]', encoding="utf-8")
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: pytest.fail("invalid seeds must fail before data loading"),
    )
    runner = SimpleNamespace(
        foldcp_config=SimpleNamespace(size_dp=2),
        predict=lambda _data: pytest.fail("invalid seeds must fail before forward"),
    )
    configs = SimpleNamespace(input_json_path=str(input_path), seeds=seeds)

    with pytest.raises(ValueError, match="Seed-parallel inference requires"):
        inference.infer_predict(runner, configs)


def test_seed_parallel_and_foldcp_output_ownership(monkeypatch, tmp_path):
    from runner import inference

    class OrderedLoader:
        dataset = [object(), object()]

        def __iter__(self):
            for sample_index, sample_name in enumerate(("job-a", "job-b")):
                data = {
                    "sample_name": sample_name,
                    "sample_index": sample_index,
                    "N_asym": torch.tensor(1),
                    "N_token": torch.tensor(1),
                    "N_atom": torch.tensor(1),
                    "N_msa": torch.tensor(1),
                    "entity_poly_type": {},
                    "input_feature_dict": {},
                }
                yield [(data, None, "")]

    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: OrderedLoader(),
    )
    monkeypatch.setattr(
        inference, "cleanup_device_memory", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(inference, "seed_everything", lambda **_kwargs: None)

    def run_rank(rank, *, size_dp, foldcp_enabled, seeds, seed_batch_size=1):
        predictions = []
        forward_batches = []
        dumps = []
        monkeypatch.setattr(
            inference,
            "DIST_WRAPPER",
            SimpleNamespace(world_size=2, rank=rank, local_rank=rank),
        )
        monkeypatch.setattr(
            inference,
            "_resolve_inference_seeds",
            lambda _configs, _size_dp, _seed_batch_size=1: list(seeds),
        )

        def predict_seed_batch(data_batch, batch_seeds, **_kwargs):
            forward_batches.append((data_batch[0]["sample_name"], tuple(batch_seeds)))
            predictions.extend(
                (data_batch[0]["sample_name"], seed) for seed in batch_seeds
            )
            return [{} for _ in batch_seeds]

        runner = SimpleNamespace(
            device=torch.device("cpu"),
            error_dir=str(tmp_path / f"errors-{rank}"),
            foldcp_config=SimpleNamespace(
                enabled=foldcp_enabled,
                size_dp=size_dp,
            ),
            update_model_configs=lambda _configs: None,
            predict_seed_batch=predict_seed_batch,
            dumper=SimpleNamespace(
                dump=lambda **kwargs: dumps.append((kwargs["pdb_id"], kwargs["seed"]))
            ),
        )
        configs = SimpleNamespace(
            input_json_path="input.json",
            seed_batch_size=seed_batch_size,
            deterministic=False,
            dump_dir=str(tmp_path),
            skip_amp=SimpleNamespace(
                confidence_head=False,
                sample_diffusion=False,
            ),
        )

        inference.infer_predict(runner, configs)
        return predictions, dumps, forward_batches

    dp_results = [
        run_rank(
            rank,
            size_dp=2,
            foldcp_enabled=False,
            seeds=[101, 102, 103, 104],
        )
        for rank in range(2)
    ]
    assert dp_results[0][0] == [
        ("job-a", 101),
        ("job-b", 101),
        ("job-a", 103),
        ("job-b", 103),
    ]
    assert dp_results[1][0] == [
        ("job-a", 102),
        ("job-b", 102),
        ("job-a", 104),
        ("job-b", 104),
    ]
    assert dp_results[0][1] == dp_results[0][0]
    assert dp_results[1][1] == dp_results[1][0]

    batched_results = [
        run_rank(
            rank,
            size_dp=2,
            foldcp_enabled=False,
            seeds=[101, 102, 103, 104, 105, 106, 107],
            seed_batch_size=2,
        )
        for rank in range(2)
    ]
    assert batched_results[0][2] == [
        ("job-a", (101, 103)),
        ("job-b", (101, 103)),
        ("job-a", (105, 107)),
        ("job-b", (105, 107)),
    ]
    assert batched_results[1][2] == [
        ("job-a", (102, 104)),
        ("job-b", (102, 104)),
        ("job-a", (106,)),
        ("job-b", (106,)),
    ]
    assert batched_results[0][1] == batched_results[0][0]
    assert batched_results[1][1] == batched_results[1][0]

    cp_results = [
        run_rank(rank, size_dp=1, foldcp_enabled=True, seeds=[101]) for rank in range(2)
    ]
    assert (
        cp_results[0][0]
        == cp_results[1][0]
        == [
            ("job-a", 101),
            ("job-b", 101),
        ]
    )
    assert cp_results[0][1] == cp_results[0][0]
    assert cp_results[1][1] == []


def test_infer_predict_releases_batch_before_seed_cleanup(monkeypatch, tmp_path):
    from runner import inference

    class WeakrefableData(dict):
        pass

    references = []

    class OneBatchIterator:
        def __init__(self):
            self.done = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.done:
                raise StopIteration
            self.done = True
            data = WeakrefableData(
                sample_name="sample",
                sample_index=0,
                N_asym=torch.tensor(1),
                N_token=torch.tensor(1),
                N_atom=torch.tensor(1),
                N_msa=torch.tensor(1),
                entity_poly_type={},
                input_feature_dict={},
            )
            references.append(weakref.ref(data))
            return [(data, None, "")]

    class OneBatchLoader:
        dataset = [object()]

        def __iter__(self):
            return OneBatchIterator()

    cleanup_states = []
    cleanup_kwargs = []

    def record_cleanup(_device, **kwargs):
        cleanup_kwargs.append(kwargs)
        if references:
            cleanup_states.append(all(reference() is None for reference in references))

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"modelSeeds": [1, 2]}]', encoding="utf-8")
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(tmp_path / "errors"),
        foldcp_config=SimpleNamespace(enabled=False, size_dp=1),
        update_model_configs=lambda _configs: None,
        predict_seed_batch=lambda _data, seeds, **_kwargs: [{} for _ in seeds],
        dumper=SimpleNamespace(dump=lambda **_kwargs: None),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=[1, 2],
        seed_batch_size=1,
        deterministic=False,
        dump_dir=str(tmp_path),
        skip_amp=SimpleNamespace(confidence_head=False, sample_diffusion=False),
    )

    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: OneBatchLoader(),
    )
    monkeypatch.setattr(inference, "cleanup_device_memory", record_cleanup)
    monkeypatch.setattr(inference, "seed_everything", lambda **_kwargs: None)

    inference.infer_predict(runner, configs)

    assert cleanup_states
    assert all(cleanup_states)
    # Per seed: full cleanup before the loop, per-batch cleanup without garbage
    # collection, then a synchronizing cleanup at the seed boundary.
    assert cleanup_kwargs == [
        {},
        {"collect_garbage": False},
        {"synchronize": True},
        {},
        {"collect_garbage": False},
        {"synchronize": True},
    ]


def test_seed_batch_preserves_each_seed_rng_stream_across_records():
    from runner import inference

    class RandomIterator:
        def __init__(self):
            self.index = 0
            self.setup_draw = torch.rand(1)

        def __iter__(self):
            return self

        def __next__(self):
            if self.index == 2:
                raise StopIteration
            self.index += 1
            return (
                self.setup_draw.clone(),
                random.random(),
                np.random.random(),
                torch.rand(2),
            )

    class RandomLoader:
        def __iter__(self):
            return RandomIterator()

    seeds = [101, 202]
    expected = {}
    for seed in seeds:
        inference.seed_everything(seed=seed, deterministic=False)
        iterator = iter(RandomLoader())
        expected[seed] = [
            (next(iterator), torch.rand(3)),
            (next(iterator), torch.rand(3)),
        ]

    lanes = inference._seed_data_lanes(RandomLoader(), seeds, deterministic=False)
    generators = inference._seed_batch_msa_generators(lanes, seeds, torch.device("cpu"))
    actual = {seed: [] for seed in seeds}
    for _ in range(2):
        features = [inference._next_seed_data(lane) for lane in lanes]
        inference._sync_cpu_msa_generators(
            lanes,
            generators,
            before_model=True,
            device=torch.device("cpu"),
        )
        model_draws = [torch.rand(3, generator=generator) for generator in generators]
        inference._sync_cpu_msa_generators(
            lanes,
            generators,
            before_model=False,
            device=torch.device("cpu"),
        )
        for seed, feature, model_draw in zip(seeds, features, model_draws):
            actual[seed].append((feature, model_draw))

    for seed in seeds:
        for actual_record, expected_record in zip(actual[seed], expected[seed]):
            actual_features, actual_model_draw = actual_record
            expected_features, expected_model_draw = expected_record
            torch.testing.assert_close(
                actual_features[0], expected_features[0], rtol=0, atol=0
            )
            assert actual_features[1:3] == expected_features[1:3]
            torch.testing.assert_close(
                actual_features[3], expected_features[3], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual_model_draw, expected_model_draw, rtol=0, atol=0
            )


@pytest.mark.parametrize(
    ("seeds", "seed_batch_size", "error_lanes", "forward_seeds", "fatal"),
    [
        ([101, 202], 2, 0, (101, 202), True),
        ([101], 1, 0, (101,), False),
        ([101, 202], 2, 1, (202,), False),
    ],
)
def test_only_seed_batch_oom_is_fatal_and_not_retried(
    monkeypatch,
    tmp_path,
    seeds,
    seed_batch_size,
    error_lanes,
    forward_seeds,
    fatal,
):
    from runner import inference

    class OneRecordLoader:
        dataset = [object()]

        def __init__(self):
            self.lane = 0

        def __iter__(self):
            data = {
                "sample_name": "sample",
                "sample_index": 0,
                "N_asym": torch.tensor(1),
                "N_token": torch.tensor(1),
                "N_atom": torch.tensor(1),
                "N_msa": torch.tensor(1),
                "entity_poly_type": {},
                "input_feature_dict": {},
            }
            data_error = "test data error" if self.lane < error_lanes else ""
            self.lane += 1
            return iter([[(data, None, data_error)]])

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample"}]', encoding="utf-8")
    calls = []
    dumps = []
    cleanup_calls = []

    def predict_seed_batch(_data, seeds, **_kwargs):
        calls.append(tuple(seeds))
        raise torch.cuda.OutOfMemoryError("test OOM")

    error_dir = tmp_path / "errors"
    error_dir.mkdir()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(error_dir),
        foldcp_config=SimpleNamespace(enabled=False, size_dp=1),
        update_model_configs=lambda _configs: None,
        predict_seed_batch=predict_seed_batch,
        dumper=SimpleNamespace(dump=lambda **kwargs: dumps.append(kwargs)),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=seeds,
        seed_batch_size=seed_batch_size,
        deterministic=False,
        dump_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        inference, "get_inference_dataloader", lambda **_kwargs: OneRecordLoader()
    )
    monkeypatch.setattr(
        inference,
        "cleanup_device_memory",
        lambda *_args, **kwargs: cleanup_calls.append(kwargs),
    )
    monkeypatch.setattr(
        inference, "update_inference_configs", lambda configs, _n_token: configs
    )
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )

    if fatal:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="test OOM"):
            inference.infer_predict(runner, configs)
    else:
        inference.infer_predict(runner, configs)

    assert calls == [forward_seeds]
    assert dumps == []
    assert cleanup_calls == (
        [{}, {"collect_garbage": False}]
        if fatal
        else [{}, {"collect_garbage": False}, {"synchronize": True}]
    )
    assert (error_dir / "sample.txt").exists() is not fatal


def test_single_process_batch_oom_escapes_and_closes_runner(monkeypatch, tmp_path):
    from runner import batch_inference

    input_path = tmp_path / "input.json"
    input_path.write_text("[]", encoding="utf-8")
    closed = []
    runner = SimpleNamespace(configs={}, close=lambda: closed.append(True))
    monkeypatch.setattr(
        batch_inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )
    monkeypatch.setattr(batch_inference, "get_default_runner", lambda **_kwargs: runner)
    monkeypatch.setattr(
        batch_inference,
        "_preprocess_input_for_runner",
        lambda input_json, **_kwargs: input_json,
    )
    monkeypatch.setattr(
        batch_inference,
        "infer_predict",
        lambda *_args: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("test OOM")),
    )
    monkeypatch.setattr(batch_inference.tqdm, "tqdm", lambda values: values)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="test OOM"):
        batch_inference.inference_jsons(str(input_path), seed_batch_size=2)

    assert closed == [True]
