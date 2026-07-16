# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

from types import SimpleNamespace

import pytest
import torch

from opendde.utils import environment


def test_auto_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert environment.select_torch_device("auto", local_rank=1) == torch.device(
        "cuda:1"
    )


def test_auto_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert environment.select_torch_device("auto") == torch.device("cpu")


def test_explicit_unavailable_cuda_has_actionable_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        environment.select_torch_device("cuda")


def test_cpu_triangle_kernel_does_not_probe_optional_packages(monkeypatch):
    def unexpected_probe(module_name):
        raise AssertionError(f"CPU selection must not probe {module_name}")

    monkeypatch.setattr(environment, "probe_optional_module", unexpected_probe)

    status = environment.get_cuequivariance_runtime_status(torch.device("cpu"))
    assert status.auto_triangle_kernel == "torch"


def test_cuequivariance_rejects_unsupported_platform_before_package_probe(
    monkeypatch,
):
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    def unexpected_probe(module_name):
        raise AssertionError(f"Unsupported platforms must not probe {module_name}")

    monkeypatch.setattr(environment, "probe_optional_module", unexpected_probe)

    status = environment.get_cuequivariance_runtime_status(
        torch.device("cuda:0"),
    )

    assert not status.usable
    assert "supported only on Linux x86_64" in (status.unavailable_reason or "")


def test_cc7_fallback_applies_before_platform_policy(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    status = environment.get_cuequivariance_runtime_status(
        torch.device("cuda:0"),
        probe_packages=False,
    )

    assert status.requires_cc7_fallback


def test_cuequivariance_reports_unusable_packages(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    statuses = (
        environment.OptionalModuleStatus(
            name="cuequivariance",
            installed=False,
        ),
        environment.OptionalModuleStatus(
            name="cuequivariance_torch",
            installed=True,
            import_error="OSError: ABI mismatch",
        ),
        environment.OptionalModuleStatus(
            name="cuequivariance_ops_torch",
            installed=True,
        ),
    )
    status = environment.get_cuequivariance_runtime_status(
        torch.device("cuda:0"),
        optional_modules=statuses,
    )

    assert not status.usable
    assert "cuequivariance (missing)" in (status.unavailable_reason or "")
    assert "ABI mismatch" in (status.unavailable_reason or "")


def test_cc7_requires_torch_fallback(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))

    status = environment.get_cuequivariance_runtime_status(
        torch.device("cuda:0"),
    )

    assert status.requires_cc7_fallback
    assert status.auto_triangle_kernel == "torch"


def test_cc6_is_unavailable_without_cc7_precision_policy(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (6, 1))

    status = environment.get_cuequivariance_runtime_status(
        torch.device("cuda:0"),
    )

    assert not status.usable
    assert not status.requires_cc7_fallback
    assert status.auto_triangle_kernel == "torch"
    assert "Compute Capability 8.0 or newer" in (status.unavailable_reason or "")


def test_broken_optional_import_is_installed_but_unusable(monkeypatch):
    monkeypatch.setattr(environment, "module_available", lambda module_name: True)
    monkeypatch.setattr(
        environment, "distribution_version", lambda module_name: "0.8.0"
    )

    def broken_import(module_name):
        raise OSError(f"ABI mismatch while importing {module_name}")

    monkeypatch.setattr(environment.importlib, "import_module", broken_import)

    status = environment.probe_optional_module("cuequivariance_torch")

    assert status.installed
    assert not status.usable
    assert "ABI mismatch" in (status.import_error or "")


def test_cuda_probe_failure_keeps_cpu_torch_usable(monkeypatch):
    monkeypatch.setattr(environment, "module_available", lambda module_name: True)
    monkeypatch.setattr(
        environment, "distribution_version", lambda module_name: "2.7.1"
    )

    def fail_cuda_probe():
        raise OSError("CUDA driver mismatch")

    torch_module = SimpleNamespace(
        __version__="2.7.1",
        version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=fail_cuda_probe),
    )
    monkeypatch.setattr(
        environment.importlib, "import_module", lambda module_name: torch_module
    )

    info = environment.get_torch_runtime_info()

    assert info.usable
    assert info.import_error is None
    assert info.cuda_probe_error == "OSError: CUDA driver mismatch"
    assert "CPU inference remains available" in environment._runtime_recommendation(
        info, None, None
    )


def test_doctor_probes_each_optional_package_once(monkeypatch):
    probes = []

    def probe(module_name):
        probes.append(module_name)
        return environment.OptionalModuleStatus(module_name, installed=False)

    monkeypatch.setattr(
        environment,
        "get_torch_runtime_info",
        lambda: environment.TorchRuntimeInfo(installed=True, version="2.7.1"),
    )
    monkeypatch.setattr(
        environment, "select_torch_device", lambda requested: torch.device("cpu")
    )
    monkeypatch.setattr(environment, "probe_optional_module", probe)
    monkeypatch.setattr(environment, "nvidia_smi_summary", lambda: (False, None))

    report = environment.format_doctor_report()

    assert probes == list(environment.GPU_OPTIONAL_MODULES)
    assert "Selected inference device for auto mode: cpu" in report
    assert "Selected triangle kernel for auto mode: torch" in report
    assert "python -m pip install opendde" in report
    assert 'python -m pip install "opendde[gpu]"' in report
    assert "uv pip install --torch-backend auto opendde" in report
    assert "uv pip install --torch-backend cpu opendde" in report
    assert 'uv pip install --torch-backend cu126 "opendde[gpu]"' in report


def test_torch_summary_distinguishes_missing_from_broken():
    assert (
        environment._torch_summary(environment.TorchRuntimeInfo(installed=False))
        == "missing"
    )
    assert "installed but unusable" in environment._torch_summary(
        environment.TorchRuntimeInfo(
            installed=True,
            version="2.7.1",
            import_error="OSError: incompatible shared library",
        )
    )
