# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gc
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial

import numpy as np
import torch


def _mps_autocast_available() -> bool:
    """Whether an MPS autocast state exists that a guard would have to clear."""
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


@contextmanager
def disabled_autocast():
    """Force FP32 inside the block on every accelerator that can autocast.

    Regions that must stay FP32 historically guarded only CUDA autocast. The
    Apple MPS backend keeps a separate autocast state, so a CUDA-only guard
    silently lets BF16 through there.
    """
    with ExitStack() as stack:
        stack.enter_context(torch.amp.autocast("cuda", enabled=False))
        if _mps_autocast_available():
            stack.enter_context(torch.amp.autocast("mps", enabled=False))
        yield


def to_device(obj, device, non_blocking: bool = False):
    """Move tensor or dict of tensors to device"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device, non_blocking=non_blocking)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device=device, non_blocking=non_blocking)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device=device, non_blocking=non_blocking)
    else:
        raise Exception(f"type {type(obj)} not supported")
    return obj


def _clear_accelerator_cache(
    *,
    synchronize: Callable[[], None] | None,
    empty_cache: Callable[[], None],
    suppress_errors: bool,
) -> None:
    """Run cache cleanup without letting synchronization skip cache release."""
    synchronize_error: RuntimeError | None = None
    if synchronize is not None:
        try:
            synchronize()
        except RuntimeError as exc:
            if not suppress_errors:
                synchronize_error = exc
    try:
        empty_cache()
    except RuntimeError:
        if not suppress_errors and synchronize_error is None:
            raise
    if synchronize_error is not None:
        raise synchronize_error


def cleanup_device_memory(
    device: torch.device | str,
    collect_garbage: bool = True,
    synchronize: bool = False,
    suppress_errors: bool = False,
) -> None:
    """Collect garbage, optionally synchronize, and clear an accelerator cache.

    ``suppress_errors`` is reserved for best-effort cleanup after an operation
    has already failed. Normal synchronization must surface asynchronous device
    errors instead of allowing inference to report success.
    """
    selected_device = torch.device(device)
    if collect_garbage:
        gc.collect()

    if selected_device.type == "mps" and torch.backends.mps.is_available():
        _clear_accelerator_cache(
            synchronize=torch.mps.synchronize if synchronize else None,
            empty_cache=torch.mps.empty_cache,
            suppress_errors=suppress_errors,
        )
        return

    if selected_device.type == "cuda" and torch.cuda.is_available():
        # Cleanup may run after a failed CUDA operation, which leaves the
        # context in a sticky error state where every CUDA call re-reports it.
        # Keep the calls on separate error boundaries so a failed synchronize
        # still lets the allocator release its blocks, but suppress failures
        # only when the caller is already recovering from another error.
        _clear_accelerator_cache(
            synchronize=(
                partial(torch.cuda.synchronize, device=selected_device)
                if synchronize
                else None
            ),
            empty_cache=torch.cuda.empty_cache,
            suppress_errors=suppress_errors,
        )


@contextmanager
def disable_cudnn_benchmark(device: torch.device | str | None = None):
    """Temporarily disable cuDNN benchmark for the selected CUDA device."""
    device_type = torch.device(device).type if device is not None else None
    if device_type not in {None, "cuda"} or not torch.cuda.is_available():
        yield
        return

    benchmark_enabled = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark_enabled


def cdist(a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """Use PyTorch's default distance compute-mode selection."""
    return torch.cdist(a, b if b is not None else a)


def map_values_to_list(data, recursive=True):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = v.cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, dict) and recursive:
            data[k] = map_values_to_list(v, recursive)
    return data


def round_values(data, recursive=True):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = np.round(v.cpu().numpy(), 2)
        elif isinstance(v, np.ndarray):
            data[k] = np.round(v, 2)
        elif isinstance(v, list):
            data[k] = list(np.round(np.array(v), 2))
        elif isinstance(v, dict) and recursive:
            data[k] = round_values(v, recursive)
    return data


def autocasting_disable_decorator(disable_casting):
    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = disabled_autocast() if disable_casting else nullcontext()

            # Helper function to conditionally cast tensors
            def conditioned_cast(tensor):
                if (
                    disable_casting
                    and isinstance(tensor, torch.Tensor)
                    and torch.is_floating_point(tensor)
                ):
                    return tensor.to(dtype=torch.float32)
                return tensor

            with _amp_context:
                return func(
                    *(conditioned_cast(v) for v in args),
                    **{k: conditioned_cast(v) for k, v in kwargs.items()},
                )

        return new_func

    return func_wrapper


def dict_to_tensor(feature_dict):
    for k, v in feature_dict.items():
        if not isinstance(v, torch.Tensor):
            dtype = feature_dict[k].dtype
            feature_dict[k] = torch.tensor(v)

            if dtype in [np.int64, np.int32]:
                feature_dict[k] = feature_dict[k].to(torch.int64)
            elif dtype in [np.float32, np.float64]:
                feature_dict[k] = feature_dict[k].to(torch.float32)

    return feature_dict


def collate_fn_identity(x):
    return x
