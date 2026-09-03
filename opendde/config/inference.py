# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

import torch

from opendde.config.config import parse_configs
from opendde.config.data import data_configs
from opendde.config.inference_defaults import inference_configs
from opendde.config.model_base import configs as configs_base
from opendde.config.model_registry import model_configs
from opendde.config.schema import OpenDDEConfig
from opendde.utils.environment import (
    CuEquivarianceRuntimeStatus,
    get_cuequivariance_runtime_status,
    select_torch_device,
)

TRIANGLE_KERNELS = ("auto", "cuequivariance", "torch")
logger = logging.getLogger(__name__)


def deep_update(configs: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update nested config dictionaries in place."""
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and key in configs
            and isinstance(configs[key], Mapping)
        ):
            deep_update(configs[key], value)
        else:
            configs[key] = copy.deepcopy(value)
    return configs


def make_base_inference_config(model_name: Optional[str] = None) -> dict[str, Any]:
    """Return an isolated base inference config tree."""
    configs = {
        **copy.deepcopy(configs_base),
        "data": copy.deepcopy(data_configs),
        **copy.deepcopy(inference_configs),
    }
    if model_name is not None:
        configs["model_name"] = model_name
    return configs


def build_inference_config(
    arg_str: Optional[str] = None,
    model_name: Optional[str] = None,
    fill_required_with_null: bool = True,
) -> OpenDDEConfig:
    """
    Build inference configs with model-specific defaults and CLI overrides.

    The selected model is parsed first, model-specific defaults are merged into
    a fresh base tree, then the same arguments are parsed again so user-provided
    values keep highest priority. The fully resolved tree is wrapped in the typed
    :class:`OpenDDEConfig` view (the merge/CLI engine itself is untouched).
    """
    first_pass = parse_configs(
        configs=make_base_inference_config(model_name=model_name),
        arg_str=arg_str,
        fill_required_with_null=fill_required_with_null,
    )
    selected_model_name = first_pass.model_name
    if selected_model_name not in model_configs:
        supported = ", ".join(sorted(model_configs))
        raise ValueError(
            f"Unsupported model_name {selected_model_name!r}. "
            f"Available models: {supported}."
        )

    base_configs = make_base_inference_config(model_name=model_name)
    deep_update(base_configs, model_configs[selected_model_name])
    merged = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=fill_required_with_null,
    )
    return OpenDDEConfig.model_validate(merged.to_dict())


def validate_triangle_kernels(
    triangle_multiplicative: str, triangle_attention: str
) -> None:
    """Validate triangle kernel names used by inference."""
    if triangle_multiplicative not in TRIANGLE_KERNELS:
        raise ValueError(
            "Invalid triangle_multiplicative. Options: 'auto', 'cuequivariance', 'torch'."
        )
    if triangle_attention not in TRIANGLE_KERNELS:
        raise ValueError(
            "Invalid triangle_attention. Options: 'auto', 'cuequivariance', 'torch'."
        )


def validate_config_triangle_kernels(configs: OpenDDEConfig) -> None:
    validate_triangle_kernels(
        configs.triangle_multiplicative,
        configs.triangle_attention,
    )


def validate_inference_schedule(configs: OpenDDEConfig) -> None:
    """Reject empty/negative inference schedules before model initialization."""

    for option, value in (
        ("model.N_cycle", configs.model.N_cycle),
        ("model.N_model_seed", configs.model.N_model_seed),
        ("sample_diffusion.N_step", configs.sample_diffusion.N_step),
        ("sample_diffusion.N_sample", configs.sample_diffusion.N_sample),
    ):
        if value < 1:
            raise ValueError(f"{option} must be at least 1, got {value}.")

    for option, value in (
        ("infer_setting.chunk_size", configs.infer_setting.chunk_size),
        (
            "infer_setting.sample_diffusion_chunk_size",
            configs.infer_setting.sample_diffusion_chunk_size,
        ),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option} must be at least 1 or null, got {value}.")

    seen_thresholds: set[int] = set()
    for (
        raw_threshold,
        chunk_size,
    ) in configs.infer_setting.chunk_size_thresholds.items():
        try:
            threshold = int(raw_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "infer_setting.chunk_size_thresholds keys must be positive integers; "
                f"got {raw_threshold!r}."
            ) from exc
        if threshold < 1:
            raise ValueError(
                "infer_setting.chunk_size_thresholds keys must be positive integers; "
                f"got {raw_threshold!r}."
            )
        if threshold in seen_thresholds:
            raise ValueError(
                "infer_setting.chunk_size_thresholds contains duplicate numeric "
                f"threshold {threshold}."
            )
        seen_thresholds.add(threshold)
        if chunk_size != -1 and chunk_size < 1:
            raise ValueError(
                "infer_setting.chunk_size_thresholds values must be -1 or at "
                f"least 1; got {chunk_size} for threshold {raw_threshold!r}."
            )


def resolve_auto_triangle_kernels(
    configs: OpenDDEConfig,
    runtime_status: CuEquivarianceRuntimeStatus,
) -> OpenDDEConfig:
    requested_multiplicative = configs.triangle_multiplicative
    requested_attention = configs.triangle_attention
    if "auto" not in {requested_multiplicative, requested_attention}:
        return configs

    auto_kernel = runtime_status.auto_triangle_kernel
    if requested_multiplicative == "auto":
        configs.triangle_multiplicative = auto_kernel
    if requested_attention == "auto":
        configs.triangle_attention = auto_kernel

    logger.info(
        "Resolved triangle kernels from auto to multiplicative=%s, attention=%s.",
        configs.triangle_multiplicative,
        configs.triangle_attention,
    )
    return configs


def validate_triangle_kernel_runtime(
    configs: OpenDDEConfig,
    runtime_status: CuEquivarianceRuntimeStatus,
) -> None:
    if "cuequivariance" not in {
        configs.triangle_multiplicative,
        configs.triangle_attention,
    }:
        return
    if runtime_status.unavailable_reason is not None:
        raise RuntimeError(runtime_status.unavailable_reason)


def apply_runtime_compatibility(
    configs: OpenDDEConfig,
    device: torch.device,
) -> OpenDDEConfig:
    """Apply runtime policy using one already-resolved inference device."""
    if not isinstance(device, torch.device):
        raise TypeError("device must be an already-resolved torch.device")

    validate_inference_schedule(configs)
    validate_config_triangle_kernels(configs)

    # The maintained 1 x P implementation owns the distributed triangle
    # schedule and is validated only with its PyTorch kernels.  Leaving the CLI
    # default at ``auto`` previously selected cuEquivariance whenever its
    # packages imported successfully, despite the Fold-CP documentation saying
    # that backend is unsupported.  That silently changed P=1/P>1 numerics and
    # could enter an unsupported collective/kernel combination.
    if getattr(configs, "foldcp_mode", "single") == "distributed":
        explicit_cueq = {
            configs.triangle_multiplicative,
            configs.triangle_attention,
        } & {"cuequivariance"}
        if explicit_cueq:
            raise ValueError(
                "Distributed 1 x P Fold-CP does not support cuEquivariance "
                "triangle kernels. Use --trimul_kernel torch "
                "--triatt_kernel torch."
            )
        if configs.triangle_multiplicative == "auto":
            configs.triangle_multiplicative = "torch"
        if configs.triangle_attention == "auto":
            configs.triangle_attention = "torch"

    requested_kernels = {
        configs.triangle_multiplicative,
        configs.triangle_attention,
    }
    runtime_status = get_cuequivariance_runtime_status(
        device,
        probe_packages=requested_kernels != {"torch"},
    )

    if device.type == "mps" and configs.dtype == "bf16":
        # MPS autocast rejects BF16 below macOS 14 and silently disables itself.
        # Resolve that here so the run reports the precision it actually uses.
        if not torch.backends.mps.is_macos_or_newer(14, 0):
            logger.info(
                "Enforcing FP32 on the Apple MPS device; BF16 autocast needs "
                "macOS 14 or newer."
            )
            configs.dtype = "fp32"

    if runtime_status.requires_cc7_fallback:
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info(
            "Enforcing FP32 and torch kernels for compatibility with detected "
            "GPU (Compute Capability 7.x)."
        )
    else:
        configs = resolve_auto_triangle_kernels(configs, runtime_status)

    validate_triangle_kernel_runtime(configs, runtime_status)
    return configs


def update_gpu_compatible_configs(configs: OpenDDEConfig) -> OpenDDEConfig:
    """Compatibility wrapper that resolves a device once before applying policy."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = select_torch_device(configs.device, local_rank=local_rank)
    return apply_runtime_compatibility(configs, device)
