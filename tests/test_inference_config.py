# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gc
import inspect
import os
import weakref
from types import SimpleNamespace

import pytest
import torch

from opendde.config.inference import (
    apply_runtime_compatibility,
    build_inference_config,
    update_gpu_compatible_configs,
    validate_inference_schedule,
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


def test_legacy_config_dict_without_device_uses_schema_default():
    legacy_config = build_inference_config(fill_required_with_null=True).model_dump()
    legacy_config.pop("device")

    assert OpenDDEConfig.model_validate(legacy_config).device == "auto"


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


def test_build_inference_config_rejects_unknown_model_with_available_names():
    with pytest.raises(ValueError, match="Unsupported model_name.*opendde_v1"):
        build_inference_config(model_name="missing_model")


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
    assert configs_base["model"]["N_cycle"] == 10
    assert configs_base["confidence"]["distogram"]["no_bins"] == 96


def test_get_default_runner_rejects_rna_msa_when_global_msa_is_disabled():
    from runner import batch_inference

    with pytest.raises(ValueError, match="requires --use_msa true"):
        batch_inference.get_default_runner(
            use_msa=False,
            use_rna_msa=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dtype": "fp16"}, "dtype must be one of"),
        ({"device": "tpu"}, "device must be one of"),
        ({"trimul_kernel": "missing"}, "Invalid triangle_multiplicative"),
        ({"triatt_kernel": "missing"}, "Invalid triangle_attention"),
        ({"seeds": [True]}, "seeds must be an integer, not a boolean"),
    ],
)
def test_get_default_runner_rejects_invalid_python_api_arguments(kwargs, message):
    from runner import batch_inference

    with pytest.raises(ValueError, match=message):
        batch_inference.get_default_runner(**kwargs)


@pytest.mark.parametrize(
    ("argument", "kwargs"),
    [
        ("--cycle", {"n_cycle": 0}),
        ("--step", {"n_step": -1}),
        ("--sample", {"n_sample": 0}),
    ],
)
def test_get_default_runner_rejects_nonpositive_inference_counts(argument, kwargs):
    from runner import batch_inference

    with pytest.raises(ValueError, match=rf"{argument} must be at least 1"):
        batch_inference.get_default_runner(**kwargs)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("N_cycle", 0),
        ("N_model_seed", 0),
        ("N_step", -1),
        ("N_sample", 0),
    ],
)
def test_raw_inference_config_rejects_nonpositive_schedule(path, value):
    cfg = build_inference_config()
    if path in {"N_cycle", "N_model_seed"}:
        setattr(cfg.model, path, value)
    else:
        setattr(cfg.sample_diffusion, path, value)

    with pytest.raises(ValueError, match=rf"{path} must be at least 1"):
        validate_inference_schedule(cfg)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("chunk_size", 0),
        ("sample_diffusion_chunk_size", -1),
    ],
)
def test_raw_inference_config_rejects_nonpositive_chunk_sizes(path, value):
    cfg = build_inference_config()
    setattr(cfg.infer_setting, path, value)

    with pytest.raises(ValueError, match=rf"{path} must be at least 1 or null"):
        validate_inference_schedule(cfg)


@pytest.mark.parametrize(
    ("thresholds", "message"),
    [
        ({"invalid": 32}, "keys must be positive integers"),
        ({"0": 32}, "keys must be positive integers"),
        ({"1024": 0}, "values must be -1 or at least 1"),
        ({"1024": 32, "01024": 16}, "duplicate numeric threshold 1024"),
    ],
)
def test_raw_inference_config_rejects_invalid_chunk_thresholds(thresholds, message):
    cfg = build_inference_config()
    cfg.infer_setting.chunk_size_thresholds = thresholds

    with pytest.raises(ValueError, match=message):
        validate_inference_schedule(cfg)


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


def test_mps_device_uses_torch_kernels_and_keeps_supported_bf16(monkeypatch):
    monkeypatch.setattr(
        torch.backends.mps, "is_macos_or_newer", lambda major, minor: True
    )
    cfg = build_inference_config(
        arg_str=(
            "--dtype bf16 --triangle_attention auto --triangle_multiplicative auto"
        ),
        fill_required_with_null=True,
    )

    result = apply_runtime_compatibility(cfg, torch.device("mps"))

    assert result.dtype == "bf16"
    assert result.triangle_attention == "torch"
    assert result.triangle_multiplicative == "torch"


def test_mps_device_downgrades_bf16_below_macos_14(monkeypatch):
    monkeypatch.setattr(
        torch.backends.mps, "is_macos_or_newer", lambda major, minor: False
    )
    cfg = build_inference_config(arg_str="--dtype bf16", fill_required_with_null=True)

    assert apply_runtime_compatibility(cfg, torch.device("mps")).dtype == "fp32"


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


def test_distributed_auto_triangle_kernels_resolve_to_torch(monkeypatch):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(fill_required_with_null=True)
    cfg.foldcp_mode = "distributed"
    cfg.foldcp_size_dp = 1
    cfg.foldcp_size_cp = 4
    cfg.triangle_attention = "auto"
    cfg.triangle_multiplicative = "auto"
    probes = []

    def runtime_status(device, *, probe_packages):
        probes.append((device, probe_packages))
        return _runtime_status("package probing was intentionally skipped")

    monkeypatch.setattr(
        inference_config,
        "get_cuequivariance_runtime_status",
        runtime_status,
    )

    result = apply_runtime_compatibility(cfg, torch.device("cuda:0"))

    assert result.triangle_attention == "torch"
    assert result.triangle_multiplicative == "torch"
    assert probes == [(torch.device("cuda:0"), False)]


@pytest.mark.parametrize(
    "triangle_attention,triangle_multiplicative",
    [("cuequivariance", "torch"), ("torch", "cuequivariance")],
)
def test_distributed_explicit_cuequivariance_is_rejected(
    monkeypatch,
    triangle_attention,
    triangle_multiplicative,
):
    import opendde.config.inference as inference_config

    cfg = build_inference_config(fill_required_with_null=True)
    cfg.foldcp_mode = "distributed"
    cfg.foldcp_size_dp = 1
    cfg.foldcp_size_cp = 4
    cfg.triangle_attention = triangle_attention
    cfg.triangle_multiplicative = triangle_multiplicative
    monkeypatch.setattr(
        inference_config,
        "get_cuequivariance_runtime_status",
        lambda *args, **kwargs: pytest.fail(
            "unsupported distributed cueq must fail before package probing"
        ),
    )

    with pytest.raises(ValueError, match="does not support cuEquivariance"):
        apply_runtime_compatibility(cfg, torch.device("cuda:0"))


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
    monkeypatch.setattr(inference.dist, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
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
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
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
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)

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
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
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


def test_download_inference_assets_uses_runner_cpu_control_group(monkeypatch):
    from runner import inference

    control_group = object()
    broadcasts = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda _group=None: 2)
    monkeypatch.setattr(inference, "download_inference_cache", lambda _configs: None)

    def broadcast(status, *, src, group):
        broadcasts.append((list(status), src, group))

    monkeypatch.setattr(inference.dist, "broadcast_object_list", broadcast)

    inference._download_inference_assets(object(), control_group)

    assert broadcasts == [([(True, "")], 0, control_group)]


def test_rank0_preprocessing_broadcast_uses_runner_cpu_group(monkeypatch):
    from runner import batch_inference

    control_group = object()
    broadcasts = []
    monkeypatch.setattr(batch_inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(batch_inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(batch_inference.dist, "get_rank", lambda: 1)

    def broadcast(payload, *, src, group):
        broadcasts.append((src, group))
        payload[0] = (True, ["input.json"])

    monkeypatch.setattr(batch_inference.dist, "broadcast_object_list", broadcast)

    result = batch_inference._run_on_rank0_and_broadcast(
        lambda: pytest.fail("nonzero rank must not preprocess"),
        description="discovering inputs",
        world_control_group=control_group,
    )

    assert result == ["input.json"]
    assert broadcasts == [(0, control_group)]


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


def test_foldcp_rejects_legacy_2x2_topology():
    from opendde.distributed.foldcp.config import FoldCPConfig

    with pytest.raises(ValueError, match="foldcp_size_dp must be 1"):
        FoldCPConfig.from_runtime_args(
            mode="distributed",
            size_dp=2,
            size_cp=2,
            metrics_jsonl="metrics.jsonl",
        )


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
    monkeypatch.setattr(inference.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(inference.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        inference,
        "_create_foldcp_control_groups",
        lambda _config: (None, None, 0),
    )

    def init_process_group(*, backend, timeout):
        calls["init"].append((backend, timeout))
        state["initialized"] = True

    def destroy_process_group():
        calls["destroy"] += 1
        state["initialized"] = False

    monkeypatch.setattr(inference.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(inference.dist, "destroy_process_group", destroy_process_group)
    return calls


def test_distributed_runner_prewarms_nccl_mesh_before_model_allocation(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner.configs = SimpleNamespace(device="cuda")
    runner.foldcp_config = FoldCPConfig.from_runtime_args(
        mode="distributed",
        size_dp=1,
        size_cp=2,
    )
    runner._owns_process_group = False
    runner.foldcp_control_group = None
    runner.foldcp_world_control_group = None
    runner.foldcp_cp_rank = 0
    runner.print = lambda *_args, **_kwargs: None

    control_group = object()
    mesh = SimpleNamespace(
        prewarm_communications=lambda: events.append(("prewarm_nccl_routes",))
    )
    events = []
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(inference, "_refresh_dist_wrapper", lambda: None)
    monkeypatch.setattr(inference, "_distributed_world_size", lambda: 2)
    monkeypatch.setattr(inference, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda _device, *, local_rank: torch.device(f"cuda:{local_rank}"),
    )
    monkeypatch.setattr(inference.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(inference.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(inference.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        inference,
        "_create_foldcp_control_groups",
        lambda _config: (control_group, control_group, 0),
    )
    monkeypatch.setattr(
        inference,
        "register_foldcp_cpu_control_group",
        lambda group: events.append(("register_gloo", group)),
    )
    monkeypatch.setattr(
        inference.FoldCPProcessMesh,
        "create",
        lambda config: events.append(("create_nccl_mesh", config)) or mesh,
    )

    def run_stage(action, *, stage, foldcp_config, world_control_group):
        events.append(("stage", stage, foldcp_config, world_control_group))
        action()

    monkeypatch.setattr(inference, "_run_runner_initialization_stage", run_stage)
    monkeypatch.setattr(
        inference,
        "apply_runtime_compatibility",
        lambda configs, _device: configs,
    )
    monkeypatch.setattr(
        inference,
        "_validate_foldcp_runtime_config_consistency",
        lambda configs, foldcp_config, group: events.append(
            ("validate_runtime_config", configs, foldcp_config, group)
        ),
    )

    runner.init_env()

    assert events[:4] == [
        ("register_gloo", control_group),
        (
            "stage",
            "Fold-CP NCCL mesh and route initialization",
            runner.foldcp_config,
            control_group,
        ),
        ("create_nccl_mesh", runner.foldcp_config),
        ("prewarm_nccl_routes",),
    ]


def test_multirank_runner_preflights_foldcp_mode_before_local_branch(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner.configs = SimpleNamespace(device="cuda")
    runner.foldcp_config = FoldCPConfig.from_runtime_args(mode="single")
    runner._owns_process_group = False
    runner.foldcp_control_group = None
    runner.foldcp_world_control_group = None
    runner.foldcp_cp_rank = 0
    runner.print = lambda *_args, **_kwargs: None
    calls = []

    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=2, rank=0, local_rank=0),
    )
    monkeypatch.setattr(inference, "_refresh_dist_wrapper", lambda: None)
    monkeypatch.setattr(inference, "_distributed_world_size", lambda: 2)
    monkeypatch.setattr(inference, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(
        inference,
        "select_torch_device",
        lambda _device, *, local_rank: torch.device(f"cuda:{local_rank}"),
    )
    monkeypatch.setattr(inference.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(inference.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(inference.dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(inference.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        inference,
        "_create_foldcp_control_groups",
        lambda config: calls.append(config) or (None, None, 0),
    )
    monkeypatch.setattr(
        inference,
        "apply_runtime_compatibility",
        lambda configs, _device: configs,
    )

    runner.init_env()

    assert calls == [runner.foldcp_config]


def test_failed_runner_destroys_process_group_it_created(monkeypatch):
    from runner import inference

    configs = build_inference_config(fill_required_with_null=True)
    calls = _patch_cuda_distributed_runner(monkeypatch, inference, initialized=False)

    def fail_init_basics(self):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(inference.InferenceRunner, "init_basics", fail_init_basics)

    with pytest.raises(RuntimeError, match="initialization failed"):
        inference.InferenceRunner(configs)

    assert calls == {
        "init": [("nccl", inference._DISTRIBUTED_STARTUP_TIMEOUT)],
        "destroy": 1,
    }


def test_failed_runner_preserves_preinitialized_process_group(monkeypatch):
    from runner import inference

    configs = build_inference_config(fill_required_with_null=True)
    calls = _patch_cuda_distributed_runner(monkeypatch, inference, initialized=True)

    def fail_init_basics(self):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(inference.InferenceRunner, "init_basics", fail_init_basics)

    with pytest.raises(RuntimeError, match="initialization failed"):
        inference.InferenceRunner(configs)

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


def test_runner_close_releases_model_before_allocator_cleanup(monkeypatch):
    from runner import inference

    class Payload:
        pass

    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = False
    runner.foldcp_control_group = None
    runner.foldcp_world_control_group = None
    runner.device = torch.device("cpu")
    payload = Payload()
    payload_reference = weakref.ref(payload)
    runner.model = payload
    runner.dumper = object()
    del payload
    cleanup_observations = []

    def observe_cleanup(device, *, collect_garbage):
        gc.collect()
        cleanup_observations.append(
            (device, collect_garbage, payload_reference() is None)
        )

    monkeypatch.setattr(inference, "cleanup_device_memory", observe_cleanup)
    monkeypatch.setattr(inference.dist, "is_available", lambda: False)
    monkeypatch.setattr(inference, "clear_foldcp_process_mesh_cache", lambda: None)

    runner.close()

    assert runner.model is None
    assert runner.dumper is None
    assert payload_reference() is None
    assert cleanup_observations == [(torch.device("cpu"), True, True)]


def test_failed_checkpoint_initialization_releases_half_built_model(monkeypatch):
    from opendde.distributed.foldcp.config import FoldCPConfig
    from runner import inference

    class Payload:
        pass

    configs = build_inference_config(fill_required_with_null=True)
    payload_references = []
    cleanup_observations = []

    def init_env(self):
        self.device = torch.device("cpu")
        self.use_cuda = False

    def init_model(self):
        payload = Payload()
        payload_references.append(weakref.ref(payload))
        self.model = payload

    def load_checkpoint(_self):
        raise RuntimeError("checkpoint initialization failed")

    def observe_cleanup(_device, *, collect_garbage):
        gc.collect()
        cleanup_observations.append((collect_garbage, payload_references[0]() is None))

    monkeypatch.setattr(inference.InferenceRunner, "init_env", init_env)
    monkeypatch.setattr(inference.InferenceRunner, "init_basics", lambda self: None)
    monkeypatch.setattr(inference.InferenceRunner, "init_model", init_model)
    monkeypatch.setattr(
        inference.InferenceRunner,
        "load_checkpoint",
        load_checkpoint,
    )
    monkeypatch.setattr(inference, "_download_inference_assets", lambda *_args: None)
    monkeypatch.setattr(inference, "cleanup_device_memory", observe_cleanup)
    monkeypatch.setattr(inference.dist, "is_available", lambda: False)
    monkeypatch.setattr(inference, "clear_foldcp_process_mesh_cache", lambda: None)

    with pytest.raises(RuntimeError, match="checkpoint initialization failed"):
        inference.InferenceRunner(
            configs,
            foldcp_config=FoldCPConfig.from_runtime_args(),
        )

    gc.collect()
    assert payload_references[0]() is None
    assert cleanup_observations == [(True, True)]


def test_padding_only_cp_ranks_use_serial_model_and_restore_environment(monkeypatch):
    from opendde.distributed.foldcp.config import (
        FOLDCP_ENVIRONMENT_KEYS,
        FoldCPConfig,
        apply_foldcp_config,
        use_serial_model_when_cp_has_padding_only_ranks,
    )

    configs = SimpleNamespace()
    foldcp = FoldCPConfig.from_runtime_args(mode="distributed", size_dp=1, size_cp=4)
    previous = {key: os.environ.get(key) for key in FOLDCP_ENVIRONMENT_KEYS}
    apply_foldcp_config(configs, foldcp)

    try:
        with use_serial_model_when_cp_has_padding_only_ranks(foldcp, 2) as active:
            assert active is True
            assert os.environ["OPENDDE_FOLDCP_MODE"] == "single"
            assert os.environ["OPENDDE_FOLDCP_SIZE_DP"] == "1"
            assert os.environ["OPENDDE_FOLDCP_SIZE_CP"] == "1"

        assert os.environ["OPENDDE_FOLDCP_MODE"] == "distributed"
        assert os.environ["OPENDDE_FOLDCP_SIZE_DP"] == "1"
        assert os.environ["OPENDDE_FOLDCP_SIZE_CP"] == "4"

        with use_serial_model_when_cp_has_padding_only_ranks(foldcp, 4) as active:
            assert active is False
            assert os.environ["OPENDDE_FOLDCP_MODE"] == "distributed"

        with pytest.raises(RuntimeError, match="model failed"):
            with use_serial_model_when_cp_has_padding_only_ranks(foldcp, 2):
                raise RuntimeError("model failed")
        assert os.environ["OPENDDE_FOLDCP_MODE"] == "distributed"
        assert os.environ["OPENDDE_FOLDCP_SIZE_CP"] == "4"
    finally:
        for key, value in previous.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)


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


def test_runner_close_destroys_cpu_control_group_for_external_nccl(monkeypatch):
    from runner import inference

    control_group = object()
    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = False
    runner.foldcp_control_group = control_group
    runner.foldcp_world_control_group = control_group
    destroyed = []
    unregistered = []
    cache_cleared = []

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        inference,
        "_destroy_foldcp_control_groups",
        lambda group, world_group: destroyed.append((group, world_group)),
    )
    monkeypatch.setattr(
        inference,
        "unregister_foldcp_cpu_control_group",
        lambda group: unregistered.append(group),
    )
    monkeypatch.setattr(
        inference,
        "clear_foldcp_process_mesh_cache",
        lambda: cache_cleared.append(True),
    )
    monkeypatch.setattr(
        inference.dist,
        "destroy_process_group",
        lambda *_args: pytest.fail("external NCCL group must remain initialized"),
    )

    runner.close()

    assert destroyed == [(control_group, control_group)]
    assert unregistered == [control_group]
    assert cache_cleared == [True]
    assert runner.foldcp_control_group is None
    assert runner.foldcp_world_control_group is None


def test_runner_close_unregisters_failed_external_cpu_control_group(monkeypatch):
    from runner import inference

    control_group = object()
    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = False
    runner.foldcp_control_group = control_group
    runner.foldcp_world_control_group = control_group
    unregistered = []
    cache_cleared = []

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        inference,
        "_destroy_foldcp_control_groups",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("Gloo destroy failed")),
    )
    monkeypatch.setattr(
        inference,
        "unregister_foldcp_cpu_control_group",
        lambda group: unregistered.append(group),
    )
    monkeypatch.setattr(
        inference,
        "clear_foldcp_process_mesh_cache",
        lambda: cache_cleared.append(True),
    )
    monkeypatch.setattr(
        inference.dist,
        "destroy_process_group",
        lambda *_args: pytest.fail("external NCCL group must remain initialized"),
    )

    runner.close()

    assert unregistered == [control_group]
    assert cache_cleared == [True]
    # Preserve the handles so a repeated close can retry destruction.
    assert runner.foldcp_control_group is control_group
    assert runner.foldcp_world_control_group is control_group


def test_runner_close_destroys_owned_nccl_even_if_cache_cleanup_fails(monkeypatch):
    from runner import inference

    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = True
    runner.foldcp_control_group = None
    runner.foldcp_world_control_group = None
    destroyed = []

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        inference,
        "clear_foldcp_process_mesh_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("cache cleanup failed")),
    )
    monkeypatch.setattr(
        inference.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )

    runner.close()

    assert destroyed == [True]
    assert not runner._owns_process_group


def test_runner_close_clears_stale_foldcp_state_after_external_world_is_gone(
    monkeypatch,
):
    from runner import inference

    control_group = object()
    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = False
    runner.foldcp_control_group = control_group
    runner.foldcp_world_control_group = control_group
    unregistered = []
    cache_cleared = []

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        inference,
        "unregister_foldcp_cpu_control_group",
        lambda group: unregistered.append(group),
    )
    monkeypatch.setattr(
        inference,
        "clear_foldcp_process_mesh_cache",
        lambda: cache_cleared.append(True),
    )

    runner.close()

    assert unregistered == [control_group]
    assert cache_cleared == [True]
    assert runner.foldcp_control_group is None
    assert runner.foldcp_world_control_group is None


def test_runner_close_still_destroys_owned_nccl_when_gloo_destroy_fails(
    monkeypatch,
):
    from runner import inference

    control_group = object()
    runner = object.__new__(inference.InferenceRunner)
    runner._foldcp_environment_before_publish = None
    runner._owns_process_group = True
    runner.foldcp_control_group = control_group
    runner.foldcp_world_control_group = control_group
    nccl_destroyed = []
    unregistered = []
    cache_cleared = []

    monkeypatch.setattr(inference.dist, "is_available", lambda: True)
    monkeypatch.setattr(inference.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        inference,
        "_destroy_foldcp_control_groups",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("Gloo destroy failed")),
    )
    monkeypatch.setattr(
        inference,
        "clear_foldcp_process_mesh_cache",
        lambda: cache_cleared.append(True),
    )
    monkeypatch.setattr(
        inference.dist,
        "destroy_process_group",
        lambda: nccl_destroyed.append(True),
    )
    monkeypatch.setattr(
        inference,
        "unregister_foldcp_cpu_control_group",
        lambda group: unregistered.append(group),
    )

    runner.close()

    assert cache_cleared == [True]
    assert nccl_destroyed == [True]
    assert unregistered == [control_group]
    assert not runner._owns_process_group
    assert runner.foldcp_control_group is None
    assert runner.foldcp_world_control_group is None


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
    seed_calls = []

    def record_cleanup(_device, **kwargs):
        cleanup_kwargs.append(kwargs)
        if references:
            cleanup_states.append(all(reference() is None for reference in references))

    input_path = tmp_path / "input.json"
    input_path.write_text(
        '[{"name": "sample", "modelSeeds": [1, 2]}]', encoding="utf-8"
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(tmp_path / "errors"),
        foldcp_config=SimpleNamespace(enabled=False),
        update_model_configs=lambda _configs: None,
        predict=lambda _data: {},
        dumper=SimpleNamespace(dump=lambda **_kwargs: None),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=[1, 2],
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
    monkeypatch.setattr(
        inference,
        "seed_everything",
        lambda **kwargs: seed_calls.append(kwargs["seed"]),
    )

    inference.infer_predict(runner, configs)

    assert cleanup_states
    assert all(cleanup_states)
    # Each outer seed is reset once at seed setup, once before the batch is
    # fetched, and once before the synchronized end-of-iterator check.  A later
    # job can therefore never inherit RNG consumed by an earlier job.
    assert seed_calls == [1, 1, 1, 2, 2, 2]
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


def test_infer_predict_releases_iterator_before_seed_cleanup(monkeypatch, tmp_path):
    from runner import inference

    class IteratorPayload:
        pass

    iterator_references = []

    class EmptyIterator:
        def __init__(self):
            self.payload = IteratorPayload()
            iterator_references.append(weakref.ref(self.payload))

        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

    class EmptyLoader:
        dataset = [object()]

        def __iter__(self):
            return EmptyIterator()

    final_cleanup_observations = []

    def observe_cleanup(_device, **kwargs):
        if kwargs.get("synchronize"):
            gc.collect()
            final_cleanup_observations.append(
                all(reference() is None for reference in iterator_references)
            )

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample", "modelSeeds": [1]}]')
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(tmp_path / "errors"),
        foldcp_config=SimpleNamespace(enabled=False, size_dp=1, size_cp=1),
        update_model_configs=lambda _configs: None,
        predict=lambda _data: {},
        dumper=SimpleNamespace(dump=lambda **_kwargs: None),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=[1],
        deterministic=False,
        dump_dir=str(tmp_path),
        skip_amp=SimpleNamespace(confidence_head=False, sample_diffusion=False),
    )

    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: EmptyLoader(),
    )
    monkeypatch.setattr(inference, "cleanup_device_memory", observe_cleanup)
    monkeypatch.setattr(inference, "seed_everything", lambda **_kwargs: None)

    inference.infer_predict(runner, configs)

    assert final_cleanup_observations == [True]


def test_infer_predict_releases_failed_model_payload_before_error_cleanup(
    monkeypatch,
    tmp_path,
):
    from runner import inference

    class Payload:
        pass

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

    class OneBatchLoader:
        dataset = [object()]

        def __iter__(self):
            return iter([[(data, None, "")]])

    payload_reference = None
    cleanup_observations = []

    def fail_prediction(_data):
        nonlocal payload_reference
        payload = Payload()
        payload_reference = weakref.ref(payload)
        raise RuntimeError("CUDA out of memory in model")

    def observe_cleanup(_device, **_kwargs):
        if payload_reference is not None:
            gc.collect()
            cleanup_observations.append(payload_reference() is None)

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample", "modelSeeds": [1]}]')
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(tmp_path / "errors"),
        foldcp_config=SimpleNamespace(enabled=False, size_dp=1, size_cp=1),
        update_model_configs=lambda _configs: None,
        predict=fail_prediction,
        dumper=SimpleNamespace(dump=lambda **_kwargs: None),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=[1],
        deterministic=False,
        dump_dir=str(tmp_path),
        skip_amp=SimpleNamespace(confidence_head=False, sample_diffusion=False),
    )

    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: OneBatchLoader(),
    )
    monkeypatch.setattr(inference, "cleanup_device_memory", observe_cleanup)
    monkeypatch.setattr(inference, "seed_everything", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="CUDA out of memory in model"):
        inference.infer_predict(runner, configs)

    assert cleanup_observations
    assert all(cleanup_observations)


def test_single_process_output_failure_returns_a_failure(monkeypatch, tmp_path):
    from runner import inference

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

    class OneBatchLoader:
        dataset = [object()]

        def __iter__(self):
            return iter([[(data, None, "")]])

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample", "modelSeeds": [1]}]')
    error_dir = tmp_path / "ERR"
    error_dir.mkdir()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(error_dir),
        foldcp_config=SimpleNamespace(enabled=False, size_dp=1, size_cp=1),
        update_model_configs=lambda _configs: None,
        predict=lambda _data: {},
        dumper=SimpleNamespace(
            dump=lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full"))
        ),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=[1],
        deterministic=False,
        dump_dir=str(tmp_path),
        skip_amp=SimpleNamespace(confidence_head=False, sample_diffusion=False),
    )
    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: OneBatchLoader(),
    )
    monkeypatch.setattr(inference, "cleanup_device_memory", lambda *a, **k: None)
    monkeypatch.setattr(inference, "seed_everything", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="disk full"):
        inference.infer_predict(runner, configs)


def test_batch_cleanup_failure_does_not_replace_handled_sample_error(
    monkeypatch, tmp_path, caplog
):
    from runner import inference

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

    class OneBatchLoader:
        dataset = [object()]

        def __iter__(self):
            return iter([[(data, None, "")]])

    input_path = tmp_path / "input.json"
    input_path.write_text('[{"name": "sample", "modelSeeds": [1]}]')
    error_dir = tmp_path / "ERR"
    error_dir.mkdir()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        error_dir=str(error_dir),
        foldcp_config=SimpleNamespace(enabled=False, size_dp=1, size_cp=1),
        update_model_configs=lambda _configs: None,
        predict=lambda _data: {},
        dumper=SimpleNamespace(
            dump=lambda **_kwargs: (_ for _ in ()).throw(OSError("disk full"))
        ),
    )
    configs = SimpleNamespace(
        input_json_path=str(input_path),
        seeds=[1],
        deterministic=False,
        dump_dir=str(tmp_path),
        skip_amp=SimpleNamespace(confidence_head=False, sample_diffusion=False),
    )

    def cleanup(_device, **kwargs):
        if kwargs.get("collect_garbage") is False:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        inference,
        "get_inference_dataloader",
        lambda **_kwargs: OneBatchLoader(),
    )
    monkeypatch.setattr(inference, "cleanup_device_memory", cleanup)
    monkeypatch.setattr(inference, "seed_everything", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="disk full"):
        inference.infer_predict(runner, configs)

    assert "cleanup failed" in caplog.text
    assert "preserving active RuntimeError: " in caplog.text


def test_error_report_failure_does_not_replace_original_error(
    monkeypatch, tmp_path, caplog
):
    from runner import inference

    monkeypatch.setattr(
        inference,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read only")),
        raising=False,
    )

    inference._append_error_report(str(tmp_path), "sample.txt", "original error")

    assert "Could not write inference error report" in caplog.text
    assert "read only" in caplog.text


def test_error_report_recreates_directory_removed_by_previous_input(tmp_path):
    from runner import inference

    error_dir = tmp_path / "ERR"

    inference._append_error_report(str(error_dir), "later.txt", "later failure")

    assert (error_dir / "later.txt").read_text() == "later failure"


def test_runner_start_removes_stale_error_reports_from_previous_run(
    tmp_path, monkeypatch
):
    from runner import inference

    error_dir = tmp_path / "ERR"
    error_dir.mkdir()
    (error_dir / "old-oom.txt").write_text("previous run failed")
    runner = object.__new__(inference.InferenceRunner)
    runner.configs = SimpleNamespace(dump_dir=str(tmp_path))
    monkeypatch.setattr(inference, "_distributed_rank", lambda: 0)

    runner.init_basics()

    assert error_dir.is_dir()
    assert list(error_dir.iterdir()) == []
