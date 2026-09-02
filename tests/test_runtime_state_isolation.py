# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
from types import SimpleNamespace

import torch


def test_seed_non_deterministic_does_not_inherit_previous_global_mode():
    from opendde.utils.seed import seed_everything

    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True

        seed_everything(7, deterministic=False)

        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
    finally:
        torch.backends.cudnn.deterministic = previous_cudnn
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )


def test_runner_determinism_policy_restores_host_process_state(monkeypatch):
    from runner.inference import (
        _apply_determinism_runtime,
        _capture_determinism_runtime,
        _restore_determinism_runtime,
    )

    original = _capture_determinism_runtime()
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        host_state = _capture_determinism_runtime()

        _apply_determinism_runtime(True)

        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

        _restore_determinism_runtime(host_state)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        _restore_determinism_runtime(original)


def test_tf32_policy_is_scoped_to_one_model_call():
    from opendde.model.opendde import _tf32_runtime_scope

    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        with _tf32_runtime_scope(True):
            assert torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32

        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


def test_reusing_same_ccd_paths_keeps_warm_caches(monkeypatch):
    from opendde.data.core import ccd

    components, rdkit = ccd.get_ccd_cache_paths()
    clears = []
    cached_mols = {"ATP": object()}
    monkeypatch.setattr(ccd, "_ccd_rdkit_mols", cached_mols)
    for name in (
        "biotite_load_ccd_cif",
        "get_component_atom_array",
        "get_one_letter_code",
        "get_mol_type",
        "get_ccd_ref_info",
    ):
        monkeypatch.setattr(
            getattr(ccd, name),
            "cache_clear",
            lambda name=name: clears.append(name),
        )

    ccd.set_ccd_cache_paths(components, rdkit)

    assert clears == []
    assert ccd._ccd_rdkit_mols is cached_mols


def test_to_device_does_not_mutate_the_callers_feature_tree():
    from opendde.utils.torch_utils import to_device

    tensor = torch.tensor([1.0])
    nested = {"input_feature_dict": {"feature": tensor}, "name": "sample"}

    moved = to_device(nested, torch.device("cpu"))

    assert moved is not nested
    assert moved["input_feature_dict"] is not nested["input_feature_dict"]
    assert nested == {"input_feature_dict": {"feature": tensor}, "name": "sample"}


def test_runner_context_manager_always_closes(monkeypatch):
    from runner.inference import InferenceRunner

    runner = InferenceRunner.__new__(InferenceRunner)
    closed = []
    monkeypatch.setattr(runner, "close", lambda: closed.append(True))

    try:
        with runner as entered:
            assert entered is runner
            raise RuntimeError("prediction failed")
    except RuntimeError:
        pass

    assert closed == [True]


def test_update_model_configs_synchronizes_runtime_switches():
    from runner.inference import InferenceRunner

    runner = InferenceRunner.__new__(InferenceRunner)
    runner.configs = object()
    runner.model = SimpleNamespace(
        configs=object(),
        enable_diffusion_shared_vars_cache=False,
        enable_efficient_fusion=False,
        N_cycle=1,
        N_model_seed=1,
    )
    new_configs = SimpleNamespace(
        enable_diffusion_shared_vars_cache=True,
        enable_efficient_fusion=True,
        model=SimpleNamespace(N_cycle=7, N_model_seed=3),
    )

    runner.update_model_configs(new_configs)

    assert runner.configs is new_configs
    assert runner.model.configs is new_configs
    assert runner.model.enable_diffusion_shared_vars_cache is True
    assert runner.model.enable_efficient_fusion is True
    assert runner.model.N_cycle == 7
    assert runner.model.N_model_seed == 3
