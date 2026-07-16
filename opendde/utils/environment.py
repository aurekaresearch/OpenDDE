# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from opendde.config.schema import (
    INFERENCE_DEVICE_CHOICES,
    InferenceDevice,
    TriangleKernel,
)

if TYPE_CHECKING:
    import torch


GPU_OPTIONAL_MODULES = (
    "cuequivariance",
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
)

GPU_OPTIONAL_IMPORT_TARGETS = {
    "cuequivariance": "cuequivariance",
    "cuequivariance_torch": "cuequivariance_torch.primitives.triangle",
    "cuequivariance_ops_torch": "cuequivariance_ops_torch",
}

GPU_OPTIONAL_DISTRIBUTIONS = {
    "cuequivariance": "cuequivariance",
    "cuequivariance_torch": "cuequivariance-torch",
    "cuequivariance_ops_torch": "cuequivariance-ops-torch-cu12",
}


@dataclass(frozen=True)
class OptionalModuleStatus:
    """Installation and import state for one optional module."""

    name: str
    installed: bool
    version: Optional[str] = None
    import_error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.installed and self.import_error is None

    @property
    def summary(self) -> str:
        if self.usable:
            status = "usable"
        elif not self.installed:
            status = "missing"
        else:
            status = f"installed but unusable ({self.import_error or 'unknown error'})"
        return f"{status}; version {self.version}" if self.version else status


@dataclass(frozen=True)
class TorchRuntimeInfo:
    installed: bool
    version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    device_count: int = 0
    device_name: Optional[str] = None
    import_error: Optional[str] = None
    cuda_probe_error: Optional[str] = None

    @property
    def usable(self) -> bool:
        return self.installed and self.import_error is None


@dataclass(frozen=True)
class CuEquivarianceRuntimeStatus:
    """Whether cuEquivariance can be used for one resolved device."""

    unavailable_reason: Optional[str] = None
    requires_cc7_fallback: bool = False

    @property
    def usable(self) -> bool:
        return self.unavailable_reason is None

    @property
    def auto_triangle_kernel(self) -> TriangleKernel:
        return "cuequivariance" if self.usable else "torch"


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 160:
        message = f"{message[:157]}..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def distribution_version(module_name: str) -> Optional[str]:
    distribution = GPU_OPTIONAL_DISTRIBUTIONS.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_optional_module(module_name: str) -> OptionalModuleStatus:
    """Inspect an optional module without letting import/ABI failures escape."""
    version = distribution_version(module_name)
    installed = module_available(module_name) or version is not None
    if not installed:
        return OptionalModuleStatus(name=module_name, installed=False)

    import_target = GPU_OPTIONAL_IMPORT_TARGETS.get(module_name, module_name)
    try:
        importlib.import_module(import_target)
    except Exception as exc:
        return OptionalModuleStatus(
            name=module_name,
            installed=True,
            version=version,
            import_error=_format_exception(exc),
        )
    return OptionalModuleStatus(name=module_name, installed=True, version=version)


def get_torch_runtime_info() -> TorchRuntimeInfo:
    if not module_available("torch"):
        return TorchRuntimeInfo(installed=False)

    version = distribution_version("torch")
    try:
        torch_module = importlib.import_module("torch")
    except Exception as exc:
        return TorchRuntimeInfo(
            installed=True,
            version=version,
            import_error=_format_exception(exc),
        )

    runtime_version = (
        str(getattr(torch_module, "__version__", version or "")) or version
    )
    cuda_available = False
    device_count = 0
    device_name = None
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch_module.cuda.is_available())
        device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
        device_name = torch_module.cuda.get_device_name(0) if device_count else None
    except Exception as exc:
        return TorchRuntimeInfo(
            installed=True,
            version=runtime_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            device_count=device_count,
            device_name=device_name,
            cuda_probe_error=_format_exception(exc),
        )

    return TorchRuntimeInfo(
        installed=True,
        version=runtime_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        device_count=device_count,
        device_name=device_name,
    )


def select_torch_device(
    requested_device: InferenceDevice = "auto", local_rank: int = 0
) -> torch.device:
    """Resolve an inference device, preferring CUDA and otherwise using CPU."""
    if requested_device not in INFERENCE_DEVICE_CHOICES:
        choices = ", ".join(INFERENCE_DEVICE_CHOICES)
        raise ValueError(
            f"Invalid device {requested_device!r}. Choose from: {choices}."
        )

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"PyTorch is required for inference but could not be imported: {_format_exception(exc)}"
        ) from exc

    cuda_available = torch.cuda.is_available()
    if requested_device == "auto":
        requested_device = "cuda" if cuda_available else "cpu"

    if requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch build or use '--device cpu'."
            )
        device_count = torch.cuda.device_count()
        if local_rank < 0 or local_rank >= device_count:
            raise RuntimeError(
                f"CUDA device index {local_rank} is unavailable; detected "
                f"{device_count} CUDA device(s)."
            )
        return torch.device(f"cuda:{local_rank}")

    return torch.device("cpu")


def get_cuequivariance_runtime_status(
    device: torch.device,
    *,
    optional_modules: Optional[tuple[OptionalModuleStatus, ...]] = None,
    probe_packages: bool = True,
) -> CuEquivarianceRuntimeStatus:
    """Check cuEquivariance support for an already-resolved device."""
    if device.type != "cuda":
        return CuEquivarianceRuntimeStatus(
            "cuEquivariance kernels require an NVIDIA CUDA device. Use a CUDA "
            "device or select torch triangle kernels."
        )

    import torch

    try:
        major, _ = torch.cuda.get_device_capability(device)
    except Exception as exc:
        return CuEquivarianceRuntimeStatus(
            "Could not determine CUDA compute capability: " + _format_exception(exc)
        )

    if major < 8:
        if major < 7:
            return CuEquivarianceRuntimeStatus(
                "cuEquivariance kernels require CUDA Compute Capability 8.0 or "
                "newer. Use torch triangle kernels on this device."
            )
        return CuEquivarianceRuntimeStatus(
            "CUDA Compute Capability 7.x requires FP32 and torch triangle kernels "
            "for OpenDDE compatibility.",
            requires_cc7_fallback=True,
        )
    if platform.system() != "Linux" or platform.machine().lower() != "x86_64":
        return CuEquivarianceRuntimeStatus(
            "cuEquivariance kernels are supported only on Linux x86_64. "
            "Use auto/torch kernels on this platform."
        )
    if not probe_packages:
        return CuEquivarianceRuntimeStatus(
            "cuEquivariance package availability was not requested."
        )

    statuses = (
        optional_modules
        if optional_modules is not None
        else tuple(
            probe_optional_module(module_name) for module_name in GPU_OPTIONAL_MODULES
        )
    )
    status_by_name = {status.name: status for status in statuses}
    unusable = [
        status_by_name.get(name, OptionalModuleStatus(name=name, installed=False))
        for name in GPU_OPTIONAL_MODULES
        if name not in status_by_name or not status_by_name[name].usable
    ]
    if unusable:
        details = ", ".join(f"{status.name} ({status.summary})" for status in unusable)
        return CuEquivarianceRuntimeStatus(
            "cuEquivariance optional packages are missing or unusable: "
            f"{details}. Install matching packages with: python -m pip install "
            '"opendde[gpu]", or select the validated CUDA 12.6 backend with: '
            'uv pip install --torch-backend cu126 "opendde[gpu]".'
        )

    return CuEquivarianceRuntimeStatus()


def nvidia_smi_summary() -> tuple[bool, Optional[str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False, None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True, None

    summary = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
    return True, summary


def _torch_summary(torch_info: TorchRuntimeInfo) -> str:
    if not torch_info.installed:
        return "missing"
    if not torch_info.usable:
        version = f"; version {torch_info.version}" if torch_info.version else ""
        return (
            f"installed but unusable ({torch_info.import_error or 'unknown error'})"
            f"{version}"
        )
    return torch_info.version or "installed"


def _runtime_recommendation(
    torch_info: TorchRuntimeInfo,
    selected_device: Optional[torch.device],
    cueq_status: Optional[CuEquivarianceRuntimeStatus],
) -> str:
    if not torch_info.installed:
        return "PyTorch is missing; install a supported PyTorch build before inference."
    if not torch_info.usable:
        return (
            "PyTorch is installed but unusable. Reinstall a build compatible with "
            f"this Python/platform ({torch_info.import_error})."
        )
    if torch_info.cuda_probe_error:
        return (
            "PyTorch imports successfully, but CUDA probing failed "
            f"({torch_info.cuda_probe_error}). CPU inference remains available; "
            "use --device cpu."
        )
    if selected_device is not None and selected_device.type == "cuda":
        if platform.system() == "Windows":
            return (
                "PyTorch CUDA inference is available with torch triangle kernels. "
                "Windows support is experimental; the Linux-only cuEquivariance "
                "extra is not used."
            )
        if cueq_status is not None and cueq_status.usable:
            return "Linux CUDA inference and cuEquivariance triangle kernels are ready."
        if cueq_status is not None and cueq_status.requires_cc7_fallback:
            return (
                "CUDA Compute Capability 7.x was detected; OpenDDE will use FP32 "
                "and torch triangle kernels."
            )
        return (
            "PyTorch CUDA inference is available with torch triangle kernels. "
            "For optional Linux x86_64 cuEquivariance kernels, run: python -m "
            'pip install "opendde[gpu]", or select the validated CUDA 12.6 '
            'backend with: uv pip install --torch-backend cu126 "opendde[gpu]".'
        )
    return "No supported CUDA runtime is visible; OpenDDE will use CPU inference."


def format_doctor_report() -> str:
    torch_info = get_torch_runtime_info()
    optional_modules = tuple(
        probe_optional_module(module_name) for module_name in GPU_OPTIONAL_MODULES
    )
    selected_device: Optional[torch.device] = None
    device_error: Optional[str] = None
    cueq_status: Optional[CuEquivarianceRuntimeStatus] = None
    if torch_info.usable:
        if torch_info.cuda_probe_error:
            device_error = torch_info.cuda_probe_error
        else:
            try:
                selected_device = select_torch_device("auto")
            except Exception as exc:
                device_error = _format_exception(exc)
            else:
                cueq_status = get_cuequivariance_runtime_status(
                    selected_device,
                    optional_modules=optional_modules,
                )

    nvidia_smi_found, nvidia_smi_info = nvidia_smi_summary()
    selected_kernel = (
        cueq_status.auto_triangle_kernel if cueq_status is not None else "torch"
    )
    lines = [
        "OpenDDE environment",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- PyTorch: {_torch_summary(torch_info)}",
        f"- torch.cuda.is_available: {torch_info.cuda_available}",
        f"- torch CUDA version: {torch_info.cuda_version or 'none'}",
        f"- CUDA device count: {torch_info.device_count}",
        f"- CUDA device 0: {torch_info.device_name or 'none'}",
        f"- CUDA probe error: {torch_info.cuda_probe_error or 'none'}",
        f"- nvidia-smi: {nvidia_smi_info or ('found' if nvidia_smi_found else 'not found')}",
        "- GPU optional packages:",
        *(f"- {status.name}: {status.summary}" for status in optional_modules),
        "- Selected inference device for auto mode: "
        f"{selected_device if selected_device is not None else 'unavailable'}",
        f"- Selected triangle kernel for auto mode: {selected_kernel}",
    ]
    if device_error:
        lines.append(f"- Auto device error: {device_error}")
    lines.extend(
        [
            "",
            "Install recommendation",
            "- Standard PyPI install: python -m pip install opendde",
            '- Standard PyPI GPU extra: python -m pip install "opendde[gpu]"',
            "- Validated CPU backend: uv pip install --torch-backend cpu opendde",
            "- Validated Linux CUDA 12.6 with cuEquivariance: uv pip install "
            '--torch-backend cu126 "opendde[gpu]"',
            "- Optional automatic PyTorch backend: uv pip install "
            "--torch-backend auto opendde",
            "- PyPI publishes one OpenDDE package. uv can select the PyTorch "
            "backend automatically, but the optional [gpu] extra remains explicit.",
            f"- {_runtime_recommendation(torch_info, selected_device, cueq_status)}",
        ]
    )
    return "\n".join(lines)
