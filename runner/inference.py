# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sized
from contextlib import ExitStack, nullcontext
from datetime import timedelta
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, cast

import torch
import torch.distributed as dist

from opendde.config.config import parse_sys_args
from opendde.config.inference import (
    apply_runtime_compatibility,
    build_inference_config,
)
from opendde.config.schema import OpenDDEConfig
from opendde.data.core import ccd
from opendde.data.inference.input_validation import (
    validate_inference_jobs,
    validate_inference_seed,
)
from opendde.data.inference.infer_dataloader import (
    InferenceJobSampler,
    get_inference_dataloader,
)
from opendde.distributed.foldcp.config import (
    FOLDCP_ENVIRONMENT_KEYS,
    FoldCPConfig,
    apply_foldcp_config,
    use_serial_model_when_cp_has_padding_only_ranks,
)
from opendde.distributed.foldcp.comm import (
    detach_rank_local_error_traceback,
    register_foldcp_cpu_control_group,
    unregister_foldcp_cpu_control_group,
)
from opendde.distributed.foldcp.metrics import (
    FoldCPBenchmarkRecorder,
    infer_n_token,
    measure_foldcp_stage,
)
from opendde.distributed.foldcp.mesh import (
    FoldCPProcessMesh,
    clear_foldcp_process_mesh_cache,
)
from opendde.model.opendde import OpenDDE
from opendde.model.triangular.layers import skip_random_init
from opendde.utils.distributed import DIST_WRAPPER
from opendde.utils.download import (
    download_inference_cache,
    resolve_checkpoint_path,
)
from opendde.utils.environment import select_torch_device
from opendde.utils.logging_config import init_logging
from opendde.utils.seed import seed_everything
from opendde.utils.torch_utils import (
    cleanup_device_memory,
    disable_cudnn_benchmark,
    to_device,
)
from runner.dumper import DataDumper

logger = logging.getLogger(__name__)

_DISTRIBUTED_STARTUP_TIMEOUT = timedelta(hours=2)


def _capture_determinism_runtime() -> dict[str, Any]:
    """Capture process-wide PyTorch determinism state for Runner restoration."""

    return {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _apply_determinism_runtime(enabled: bool) -> None:
    """Publish one Runner's deterministic policy before CUDA initialization."""

    torch.backends.cudnn.deterministic = bool(enabled)
    torch.use_deterministic_algorithms(bool(enabled))
    if enabled:
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _restore_determinism_runtime(state: dict[str, Any]) -> None:
    torch.backends.cudnn.deterministic = bool(state["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    torch.use_deterministic_algorithms(
        bool(state["algorithms"]),
        warn_only=bool(state["warn_only"]),
    )
    workspace = state["cublas_workspace"]
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(workspace)


class FoldCPJobCoordinationError(RuntimeError):
    """Raised on every rank when a 1xP inference step cannot proceed safely."""


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return DIST_WRAPPER.rank


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return DIST_WRAPPER.world_size


def _group_world_size(group: dist.ProcessGroup | None) -> int:
    """Query a group's size without changing legacy/default-group call sites."""

    if group is None:
        return dist.get_world_size()
    return dist.get_world_size(group)


def _all_gather_object(
    output: list[Any],
    value: Any,
    group: dist.ProcessGroup | None,
) -> None:
    """Gather Python control data on the selected CPU group when provided."""

    if group is None:
        dist.all_gather_object(output, value)
    else:
        dist.all_gather_object(output, value, group=group)


def _broadcast_object_list(
    payload: list[Any],
    *,
    src: int,
    group: dist.ProcessGroup | None,
) -> None:
    """Broadcast Python control data on the selected CPU group when provided."""

    if group is None:
        dist.broadcast_object_list(payload, src=src)
    else:
        dist.broadcast_object_list(payload, src=src, group=group)


def _refresh_dist_wrapper() -> None:
    refresh = getattr(DIST_WRAPPER, "refresh", None)
    if refresh is not None:
        refresh()


def _append_error_report(error_dir: str, filename: str, message: str) -> None:
    """Write a diagnostic without replacing the failure being diagnosed."""

    path = opjoin(error_dir, filename)
    try:
        os.makedirs(error_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(message)
    except Exception as exc:
        logger.error("Could not write inference error report %s: %s", path, exc)


def _cleanup_after_model_error(device: torch.device, model_error: str) -> str:
    """Attempt allocator cleanup without skipping the following rank sync."""

    try:
        cleanup_device_memory(device, collect_garbage=False)
    except Exception as exc:
        return (
            f"{model_error}\n[Rank {_distributed_rank()}] cleanup after model "
            f"failure also failed: {type(exc).__name__}: {exc}"
        )
    return model_error


def _prepare_inference_batch(
    batch: Any,
) -> tuple[Mapping[str, Any], Any, str]:
    """Turn rank-local batch preparation failures into synchronizable errors."""

    try:
        data, atom_array, data_error_message = batch[0]
        # Access this before the first collective as it may itself expose a
        # malformed rank-local batch. The caller will synchronize the error.
        str(data["sample_name"])
        return data, atom_array, str(data_error_message)
    except Exception as exc:
        return (
            {"sample_index": -1, "sample_name": "unknown"},
            None,
            f"Batch preparation failed: {type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}",
        )


def _next_inference_batch_synchronized(
    iterator: Any,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> tuple[bool, Any]:
    """Advance every rank together and expose iterator failures before model work."""

    batch = None
    try:
        batch = next(iterator)
        status = "batch"
        error = ""
    except StopIteration:
        status = "done"
        error = ""
    except Exception as exc:
        status = "error"
        error = (
            f"[Rank {_distributed_rank()}] dataloader iteration failed: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )

    if not foldcp_config.enabled:
        if error:
            raise RuntimeError(error)
        return status == "batch", batch

    gathered: list[dict[str, str] | None] = [None] * _group_world_size(
        world_control_group
    )
    _all_gather_object(
        gathered,
        {"status": status, "error": error},
        world_control_group,
    )
    rank_statuses = cast(list[dict[str, str]], gathered)
    errors = [item["error"] for item in rank_statuses if item["status"] == "error"]
    if errors:
        # A healthy rank may already own a very large CPU batch while another
        # rank failed in its dataloader.  The coordination exception retains
        # this function frame; clear the local batch first so a retained
        # traceback cannot pin all MSA/features until Runner teardown.
        batch = None
        raise FoldCPJobCoordinationError("\n".join(errors))

    statuses = {item["status"] for item in rank_statuses}
    if len(statuses) != 1:
        summary = ", ".join(
            f"rank {rank}: {item['status']}" for rank, item in enumerate(rank_statuses)
        )
        batch = None
        raise FoldCPJobCoordinationError(
            "Dataloader ranks reached the end at different steps: " + summary
        )
    return status == "batch", batch


def _create_dataloader_iterator_synchronized(
    dataloader: Any,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> Any:
    """Construct an iterator without letting one rank enter it alone."""

    iterator_holder: list[Any] = []
    _run_rank_stage_synchronized(
        lambda: iterator_holder.append(iter(dataloader)),
        stage="dataloader iterator initialization",
        foldcp_config=foldcp_config,
        world_control_group=world_control_group,
    )
    return iterator_holder[0]


def _create_inference_dataloader_synchronized(
    configs: Any,
    inputs: list[dict[str, Any]],
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> Any:
    """Construct the dataloader and synchronize exceptions or an empty result."""

    dataloader_holder: list[Any] = []

    def _create() -> None:
        dataloader = get_inference_dataloader(configs=configs, inputs=inputs)
        if dataloader is None:
            raise RuntimeError("Dataloader initialization returned no dataloader.")
        dataloader_holder.append(dataloader)

    _run_rank_stage_synchronized(
        _create,
        stage="dataloader initialization",
        foldcp_config=foldcp_config,
        world_control_group=world_control_group,
    )
    return dataloader_holder[0]


def _get_dataloader_size_synchronized(
    dataloader: Any,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> int:
    """Read and compare dataset sizes before ranks enter the seed loop."""

    size_holder: list[int] = []
    _run_rank_stage_synchronized(
        lambda: size_holder.append(len(cast(Sized, dataloader.dataset))),
        stage="dataloader dataset-size inspection",
        foldcp_config=foldcp_config,
        world_control_group=world_control_group,
    )
    local_size = size_holder[0]
    if not foldcp_config.enabled:
        return local_size

    gathered_sizes: list[int | None] = [None] * _group_world_size(world_control_group)
    _all_gather_object(
        gathered_sizes,
        local_size,
        world_control_group,
    )
    if any(size != local_size for size in gathered_sizes):
        raise FoldCPJobCoordinationError(
            "Dataloader ranks reported different dataset sizes: "
            + ", ".join(
                f"rank {rank}: {size}" for rank, size in enumerate(gathered_sizes)
            )
        )
    return local_size


def _load_inference_jobs_synchronized(
    path: str,
    world_control_group: dist.ProcessGroup | None = None,
) -> list[dict[str, Any]]:
    """Read and validate jobs once so every rank featurizes identical input data."""

    is_distributed = dist.is_available() and dist.is_initialized()
    is_source = not is_distributed or dist.get_rank() == 0
    payload: list[tuple[bool, object] | None] = [None]
    if is_source:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                jobs = validate_inference_jobs(json.load(handle))
            payload[0] = (True, jobs)
        except Exception as exc:
            payload[0] = (False, f"{type(exc).__name__}: {exc}")

    if is_distributed:
        _broadcast_object_list(payload, src=0, group=world_control_group)
    result = payload[0]
    if result is None:
        raise RuntimeError("Rank 0 returned no input JSON status.")
    succeeded, value = result
    if not succeeded:
        raise ValueError(f"Invalid inference input {path}: {value}")
    return cast(list[dict[str, Any]], value)


def _create_foldcp_control_groups(
    foldcp_config: FoldCPConfig,
) -> tuple[dist.ProcessGroup | None, dist.ProcessGroup | None, int]:
    """Create the CPU control group for the maintained 1xP CP world."""
    is_distributed = dist.is_available() and dist.is_initialized()
    if not is_distributed:
        if not foldcp_config.enabled:
            return None, None, 0
        raise RuntimeError(
            "Fold-CP control synchronization requires torch.distributed."
        )

    # Every world rank must call ``new_group`` with the same groups in the same
    # order.  A rank-local configuration mismatch would otherwise make ranks
    # enter different group-creation collectives and hang without a useful
    # diagnostic.
    local_topology = {
        "mode": getattr(
            foldcp_config,
            "mode",
            "distributed" if foldcp_config.enabled else "single",
        ),
        "enabled": bool(foldcp_config.enabled),
        "size_dp": int(foldcp_config.size_dp),
        "size_cp": int(foldcp_config.size_cp),
    }
    world_size = dist.get_world_size()
    gathered_topologies: list[dict[str, int | bool] | None] = [None] * world_size
    dist.all_gather_object(gathered_topologies, local_topology)
    if any(topology != local_topology for topology in gathered_topologies):
        summary = ", ".join(
            f"rank {rank}: {topology}"
            for rank, topology in enumerate(gathered_topologies)
        )
        raise FoldCPJobCoordinationError(
            "Distributed ranks configured different Fold-CP topologies before "
            f"control-group creation: {summary}"
        )
    # InferenceRunner accepts an explicitly supplied FoldCPConfig. A caller can
    # instantiate that dataclass directly and bypass from_runtime_args(); do the
    # full validation only after every rank has joined the topology preflight so
    # an invalid rank cannot exit alone and strand its peers.
    if isinstance(foldcp_config, FoldCPConfig):
        foldcp_config = foldcp_config.validate()
    if not foldcp_config.enabled:
        return None, None, 0
    if int(foldcp_config.size_dp) != 1:
        raise ValueError(
            "Only the maintained 1 x P Fold-CP topology is supported; "
            "foldcp_size_dp must be 1."
        )
    if world_size != int(foldcp_config.size_cp):
        raise RuntimeError(
            "Distributed 1 x P Fold-CP requires WORLD_SIZE to equal "
            f"foldcp_size_cp; got {world_size} vs {foldcp_config.size_cp}."
        )
    if not dist.is_gloo_available():
        raise RuntimeError(
            "Distributed Fold-CP inference requires the Gloo backend for its "
            "CPU control plane."
        )

    world_rank = dist.get_rank()
    world_control_group = dist.new_group(
        list(range(world_size)),
        backend="gloo",
    )
    return world_control_group, world_control_group, world_rank


_FOLDCP_RUNTIME_CONFIG_EXCLUDES = {
    # These fields control local I/O or diagnostics, not model computation.
    "dump_dir",
    "input_json_path",
    "load_checkpoint_dir",
    "foldcp_metrics_jsonl",
    # The validated FoldCPConfig/topology is checked separately before process
    # group creation.  Its runtime copy also contains the metrics path above.
    "foldcp",
}


def _foldcp_compute_environment() -> dict[str, str]:
    """Return environment switches that can change rank-local model execution."""

    explicit_keys = {
        "CUBLAS_WORKSPACE_CONFIG",
        "LAYERNORM_TYPE",
        "OPENDDE_FORCE_CONFIDENCE_AMP",
        "OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP",
        "OPENDDE_PAIR_BIAS_ROW_CHUNK",
    }
    keys = explicit_keys | {
        key
        for key in os.environ
        if key.startswith("OPENDDE_FOLDCP_") and key not in FOLDCP_ENVIRONMENT_KEYS
    }
    return {key: os.environ[key] for key in sorted(keys) if key in os.environ}


def _foldcp_runtime_config_signature(configs: Any) -> dict[str, Any]:
    """Build a stable summary of one rank's resolved compute configuration."""

    if hasattr(configs, "model_dump"):
        payload = configs.model_dump(
            mode="json",
            exclude=_FOLDCP_RUNTIME_CONFIG_EXCLUDES,
        )
    elif hasattr(configs, "to_dict"):
        payload = configs.to_dict()
    else:
        payload = vars(configs)
    payload = {
        key: value
        for key, value in payload.items()
        if key not in _FOLDCP_RUNTIME_CONFIG_EXCLUDES
    }
    compute_environment = _foldcp_compute_environment()
    payload["_compute_environment"] = compute_environment
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        "digest": hashlib.sha256(encoded).hexdigest(),
        "model_name": getattr(configs, "model_name", None),
        "dtype": getattr(configs, "dtype", None),
        "triangle_multiplicative": getattr(configs, "triangle_multiplicative", None),
        "triangle_attention": getattr(configs, "triangle_attention", None),
        "deterministic": getattr(configs, "deterministic", None),
        "enable_tf32": getattr(configs, "enable_tf32", None),
        "compute_environment": compute_environment,
    }


def _validate_foldcp_runtime_config_consistency(
    configs: Any,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None,
) -> None:
    """Reject rank-local compute policies before model collectives can diverge."""

    if not foldcp_config.enabled:
        return
    local_signature = _foldcp_runtime_config_signature(configs)
    gathered: list[dict[str, Any] | None] = [None] * _group_world_size(
        world_control_group
    )
    _all_gather_object(gathered, local_signature, world_control_group)
    if any(signature != local_signature for signature in gathered):
        summary = ", ".join(
            f"rank {rank}: {signature}" for rank, signature in enumerate(gathered)
        )
        raise FoldCPJobCoordinationError(
            "Distributed ranks resolved different inference compute "
            "configurations before model loading. This can cause collective "
            f"dtype/shape mismatches or silent numerical corruption: {summary}"
        )


def _destroy_foldcp_control_groups(
    group: dist.ProcessGroup | None,
    world_group: dist.ProcessGroup | None,
) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if group is not None and group is not world_group:
        dist.destroy_process_group(group)
    if world_group is not None:
        dist.destroy_process_group(world_group)


def _tensor_scalar(data: Mapping[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _update_feature_fingerprint(
    digest: Any,
    value: Any,
) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping{")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            _update_feature_fingerprint(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        if tensor.numel() > 0:
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
        return
    if isinstance(value, (list, tuple)):
        digest.update(f"sequence:{len(value)}[".encode("ascii"))
        for item in value:
            _update_feature_fingerprint(digest, item)
        digest.update(b"]")
        return
    digest.update(f"{type(value).__name__}:{value!r}".encode("utf-8"))


def _feature_fingerprint(data: Mapping[str, Any]) -> str:
    """Hash model inputs without materializing an additional byte copy."""

    digest = hashlib.blake2b(digest_size=16)
    _update_feature_fingerprint(digest, data.get("input_feature_dict", {}))
    return digest.hexdigest()


def _synchronize_foldcp_batch(
    *,
    data: Mapping[str, Any],
    data_error_message: str,
    seed: int,
    group: dist.ProcessGroup | None,
    size_cp: int,
) -> str:
    """Fail fast if CP ranks are about to enter a model with different jobs."""
    if group is None:
        return data_error_message

    try:
        descriptor = {
            "sample_index": int(data.get("sample_index", -1)),
            "sample_name": str(data.get("sample_name", "unknown")),
            "seed": int(seed),
            "N_token": _tensor_scalar(data, "N_token"),
            "N_atom": _tensor_scalar(data, "N_atom"),
            "N_msa": _tensor_scalar(data, "N_msa"),
            "feature_fingerprint": (
                "" if data_error_message else _feature_fingerprint(data)
            ),
            "error": data_error_message,
        }
    except Exception as exc:
        descriptor = {
            "sample_index": -1,
            "sample_name": "unknown",
            "seed": int(seed),
            "N_token": None,
            "N_atom": None,
            "N_msa": None,
            "feature_fingerprint": "",
            "error": (
                f"[Rank {_distributed_rank()}] input descriptor preparation failed: "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            ),
        }
    gathered: list[dict[str, Any] | None] = [None] * size_cp
    dist.all_gather_object(gathered, descriptor, group=group)
    descriptors = cast(list[dict[str, Any]], gathered)

    errors = [
        f"CP rank {rank}: {item['error']}"
        for rank, item in enumerate(descriptors)
        if item["error"]
    ]
    if errors:
        return "\n".join(errors)

    identities = {
        (item["sample_index"], item["sample_name"], item["seed"])
        for item in descriptors
    }
    if len(identities) != 1:
        return (
            "Fold-CP ranks received different inference jobs before model forward: "
            f"{sorted(identities)}"
        )

    shapes = {(item["N_token"], item["N_atom"], item["N_msa"]) for item in descriptors}
    if len(shapes) != 1:
        return (
            "Fold-CP ranks produced different feature shapes for the same job: "
            f"{sorted(shapes)}"
        )
    fingerprints = {item.get("feature_fingerprint") for item in descriptors}
    if len(fingerprints) != 1:
        return (
            "Fold-CP ranks produced different feature contents for the same job: "
            f"{sorted(str(value) for value in fingerprints)}"
        )
    return ""


def _finalize_foldcp_batch_error(
    batch_error: str,
    foldcp_config: FoldCPConfig,
    control_group: dist.ProcessGroup | None,
    world_control_group: dist.ProcessGroup | None,
) -> str:
    """Return the already-world-synchronized 1xP batch status once."""

    if foldcp_config.enabled and control_group is not None:
        # `_synchronize_foldcp_batch` gathered descriptors/errors across the
        # complete maintained 1xP world. Every rank already holds the same
        # result, so gathering that combined string again is both redundant and
        # quadratic in the size of a traceback.
        return batch_error
    return _synchronize_foldcp_world_error(
        batch_error,
        foldcp_config,
        world_control_group,
    )


def _synchronize_foldcp_world_error(
    local_error: str,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> str:
    """Make every CP/DP rank observe the same stage failure."""
    if not foldcp_config.enabled:
        return local_error
    gathered: list[str | None] = [None] * _group_world_size(world_control_group)
    _all_gather_object(
        gathered,
        local_error,
        world_control_group,
    )
    return "\n".join(
        f"Global rank {rank}: {error}" for rank, error in enumerate(gathered) if error
    )


def _run_rank_stage_synchronized(
    action: Callable[[], None],
    *,
    stage: str,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> None:
    """Run a rank-local lifecycle action and synchronize its completion status."""

    local_error = ""
    try:
        action()
    except Exception as exc:
        local_error = (
            f"[Rank {_distributed_rank()}] {stage} failed: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
    if world_control_group is None:
        world_error = _synchronize_foldcp_world_error(local_error, foldcp_config)
    else:
        world_error = _synchronize_foldcp_world_error(
            local_error,
            foldcp_config,
            world_control_group,
        )
    if world_error:
        if foldcp_config.enabled:
            raise FoldCPJobCoordinationError(world_error)
        raise RuntimeError(world_error)


def _run_runner_initialization_stage(
    action: Callable[[], None],
    *,
    stage: str,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> None:
    """Synchronize a local startup stage once the process group is available."""

    if foldcp_config.enabled and dist.is_available() and dist.is_initialized():
        _run_rank_stage_synchronized(
            action,
            stage=stage,
            foldcp_config=foldcp_config,
            world_control_group=world_control_group,
        )
        return
    action()


def _cleanup_batch_synchronized(
    device: torch.device,
    foldcp_config: FoldCPConfig,
    *,
    active_error: BaseException | str | None,
    world_control_group: dist.ProcessGroup | None = None,
) -> None:
    """Synchronize per-batch cleanup without replacing an active root cause."""

    cleanup_error = ""
    try:
        cleanup_device_memory(device, collect_garbage=False)
    except Exception as exc:
        cleanup_error = (
            f"[Rank {_distributed_rank()}] batch memory cleanup failed: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )

    local_status = {
        "has_active_error": active_error is not None,
        "cleanup_error": cleanup_error,
    }
    if foldcp_config.enabled:
        gathered: list[dict[str, object] | None] = [None] * _group_world_size(
            world_control_group
        )
        _all_gather_object(
            gathered,
            local_status,
            world_control_group,
        )
        statuses = cast(list[dict[str, object]], gathered)
    else:
        statuses = [local_status]

    cleanup_errors = [
        str(status["cleanup_error"]) for status in statuses if status["cleanup_error"]
    ]
    if cleanup_errors:
        if not any(bool(status["has_active_error"]) for status in statuses):
            message = "\n".join(cleanup_errors)
            if foldcp_config.enabled:
                raise FoldCPJobCoordinationError(message)
            raise RuntimeError(message)
        active_error_name = (
            active_error
            if isinstance(active_error, str)
            else (
                type(active_error).__name__
                if active_error is not None
                else "error from another rank"
            )
        )
        logger.error(
            "Batch cleanup also failed while preserving active %s: %s",
            active_error_name,
            "\n".join(cleanup_errors),
        )


def _synchronize_foldcp_group_error(
    local_error: str,
    *,
    group: dist.ProcessGroup | None,
    size_cp: int,
) -> str:
    """Make one CP replica observe its output-stage failure as a unit."""

    if group is None:
        return local_error
    gathered: list[str | None] = [None] * size_cp
    dist.all_gather_object(gathered, local_error, group=group)
    return "\n".join(
        f"CP rank {rank}: {error}" for rank, error in enumerate(gathered) if error
    )


def _synchronize_foldcp_output_error(
    local_error: str,
    *,
    group: dist.ProcessGroup | None,
    size_cp: int,
    foldcp_config: FoldCPConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> str:
    """Propagate an output failure to every rank in the 1xP world."""

    # In the maintained 1xP topology the CP group spans the complete world.
    # The Runner-owned Gloo group has those same ranks in the same order, so a
    # CP gather followed by a world gather would duplicate every traceback P
    # times and add a second collective on an already-failing path.
    if world_control_group is not None:
        return _synchronize_foldcp_world_error(
            local_error,
            foldcp_config,
            world_control_group,
        )
    # Preserve one synchronization for embedded callers that provide only the
    # CP control group and not the Runner's world-control handle.
    return _synchronize_foldcp_group_error(
        local_error,
        group=group,
        size_cp=size_cp,
    )


def _resolve_job_seed_schedule(
    json_data: list[dict[str, Any]],
    cli_seeds: list[int] | None,
    world_control_group: dist.ProcessGroup | None = None,
) -> list[list[int]]:
    """Resolve seeds per JSON job and synchronize generated defaults."""

    def _build_schedule() -> list[list[int]]:
        schedule = []
        for job in json_data:
            configured = cli_seeds if cli_seeds else job.get("modelSeeds")
            seeds = (
                [
                    validate_inference_seed(
                        seed,
                        location=f"seed for job {job.get('name', '<unnamed>')!r}",
                    )
                    for seed in configured
                ]
                if configured
                else []
            )
            schedule.append(seeds or [random.randint(1, 65536)])
        return schedule

    is_distributed = dist.is_available() and dist.is_initialized()
    if not is_distributed:
        return _build_schedule()

    payload: list[tuple[bool, object] | None] = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = (True, _build_schedule())
        except Exception as exc:
            payload[0] = (False, f"{type(exc).__name__}: {exc}")
    _broadcast_object_list(payload, src=0, group=world_control_group)
    result = payload[0]
    if result is None:
        raise RuntimeError("Rank 0 returned no inference seed schedule status.")
    succeeded, value = result
    if not succeeded:
        raise ValueError(f"Invalid inference seed schedule: {value}")
    return cast(list[list[int]], value)


def _download_inference_assets(
    configs: OpenDDEConfig,
    world_control_group: dist.ProcessGroup | None = None,
) -> None:
    """Prepare shared inference assets once and synchronize all ranks."""
    if _distributed_world_size() <= 1:
        download_inference_cache(configs)
        return

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed inference asset preparation requires an initialized "
            "process group."
        )

    download_error: Exception | None = None
    download_status: list[tuple[bool, str] | None] = [None]
    if dist.get_rank() == 0:
        try:
            download_inference_cache(configs)
        except Exception as exc:
            download_error = exc
            download_status[0] = (False, f"{type(exc).__name__}: {exc}")
        else:
            download_status[0] = (True, "")

    _broadcast_object_list(
        download_status,
        src=0,
        group=world_control_group,
    )
    result = download_status[0]
    if result is None:
        raise RuntimeError("Rank 0 broadcast an invalid inference asset status.")

    succeeded, error_message = result
    if not succeeded:
        error = RuntimeError(
            f"Inference asset preparation failed on rank 0: {error_message}"
        )
        if download_error is not None:
            raise error from download_error
        raise error


class InferenceRunner(object):
    """
    Runner class for AlphaFold3 model inference.
    Handles environment setup, model initialization, and running predictions.

    Args:
        configs (OpenDDEConfig): Configuration object for inference.
        foldcp_config (FoldCPConfig | None): Pre-validated Fold-CP settings.
    """

    def __init__(
        self,
        configs: OpenDDEConfig,
        *,
        foldcp_config: FoldCPConfig | None = None,
    ) -> None:
        self._owns_process_group = False
        self._determinism_runtime_before_run = _capture_determinism_runtime()
        self._ccd_cache_paths_before_run = ccd.get_ccd_cache_paths()
        self._foldcp_environment_before_publish: dict[str, str | None] | None = None
        self.foldcp_control_group: dist.ProcessGroup | None = None
        self.foldcp_world_control_group: dist.ProcessGroup | None = None
        self.foldcp_cp_rank = 0
        try:
            self.foldcp_config = (
                foldcp_config
                if foldcp_config is not None
                else FoldCPConfig.from_config(configs)
            )
            self.configs = configs
            # Runtime compatibility (including triangle-kernel resolution)
            # runs during ``init_env``.  Make the already validated request
            # visible on this Runner-owned config before that stage; publishing
            # process-global Fold-CP environment variables remains deferred
            # until all initialization succeeds.
            self.configs.foldcp_mode = self.foldcp_config.mode
            self.configs.foldcp_size_dp = self.foldcp_config.size_dp
            self.configs.foldcp_size_cp = self.foldcp_config.size_cp
            self.configs.foldcp_devices = self.foldcp_config.devices
            self.configs.foldcp_metrics_jsonl = self.foldcp_config.metrics_jsonl
            self.configs.foldcp = self.foldcp_config.to_dict()
            _apply_determinism_runtime(self.configs.deterministic)
            self.init_env()
            metric_rank = _distributed_rank()
            recorder_holder: list[FoldCPBenchmarkRecorder] = []
            _run_runner_initialization_stage(
                lambda: recorder_holder.append(
                    FoldCPBenchmarkRecorder(
                        self.foldcp_config.metrics_jsonl,
                        rank=metric_rank,
                        write_rank_sidecar=False,
                    )
                ),
                stage="benchmark recorder initialization",
                foldcp_config=self.foldcp_config,
                world_control_group=self.foldcp_world_control_group,
            )
            self.foldcp_recorder = recorder_holder[0]
            if self.foldcp_world_control_group is None:
                _download_inference_assets(self.configs)
            else:
                _download_inference_assets(
                    self.configs,
                    self.foldcp_world_control_group,
                )
            _run_runner_initialization_stage(
                self.init_basics,
                stage="output directory initialization",
                foldcp_config=self.foldcp_config,
                world_control_group=self.foldcp_world_control_group,
            )

            def _initialize_model() -> None:
                with skip_random_init() if self.configs.load_strict else nullcontext():
                    self.init_model()

            _run_runner_initialization_stage(
                _initialize_model,
                stage="model construction and device transfer",
                foldcp_config=self.foldcp_config,
                world_control_group=self.foldcp_world_control_group,
            )
            _run_runner_initialization_stage(
                self.load_checkpoint,
                stage="checkpoint loading",
                foldcp_config=self.foldcp_config,
                world_control_group=self.foldcp_world_control_group,
            )
            _run_runner_initialization_stage(
                lambda: self.init_dumper(
                    need_atom_confidence=self.configs.need_atom_confidence,
                    sorted_by_ranking_score=self.configs.sorted_by_ranking_score,
                ),
                stage="output dumper initialization",
                foldcp_config=self.foldcp_config,
                world_control_group=self.foldcp_world_control_group,
            )

            # Fold-CP is process-global today; publish it only after initialization
            # succeeds and retain the previous state for close().
            def _publish_foldcp_config() -> None:
                self._foldcp_environment_before_publish = {
                    key: os.environ.get(key) for key in FOLDCP_ENVIRONMENT_KEYS
                }
                self.configs = apply_foldcp_config(
                    self.configs,
                    self.foldcp_config,
                )

            _run_runner_initialization_stage(
                _publish_foldcp_config,
                stage="Fold-CP runtime publication",
                foldcp_config=self.foldcp_config,
                world_control_group=self.foldcp_world_control_group,
            )
        except BaseException:
            self.close()
            raise

    def init_env(self) -> None:
        """
        Initialize the execution environment, including CUDA and distributed setup.
        """
        _refresh_dist_wrapper()
        world_size = _distributed_world_size()
        self.print(
            f"Distributed environment: world size: {world_size}, "
            f"global rank: {_distributed_rank()}, local rank: {DIST_WRAPPER.local_rank}"
        )
        self.device = select_torch_device(
            self.configs.device, local_rank=DIST_WRAPPER.local_rank
        )
        self.use_cuda = self.device.type == "cuda"
        if self.use_cuda:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            all_gpu_ids = ",".join(str(x) for x in range(torch.cuda.device_count()))
            devices = os.getenv("CUDA_VISIBLE_DEVICES", all_gpu_ids)
            logging.info(
                f"LOCAL_RANK: {DIST_WRAPPER.local_rank} - CUDA_VISIBLE_DEVICES: [{devices}]"
            )
            torch.cuda.set_device(self.device)

        if world_size > 1:
            if not self.use_cuda:
                raise RuntimeError(
                    "Distributed Fold-CP inference requires NVIDIA CUDA; CPU "
                    "and Apple MPS support single-process inference only."
                )
            if not dist.is_nccl_available():
                raise RuntimeError(
                    "Distributed Fold-CP inference requires the NCCL backend, "
                    "which is unavailable in this PyTorch build. Windows "
                    "distributed inference is not currently supported."
                )
            if dist.is_initialized():
                if dist.get_backend() != "nccl":
                    raise RuntimeError(
                        "Distributed Fold-CP requires an NCCL process group."
                    )
            else:
                dist.init_process_group(
                    backend="nccl", timeout=_DISTRIBUTED_STARTUP_TIMEOUT
                )
                self._owns_process_group = True
            _refresh_dist_wrapper()

            # Every world rank must enter the topology preflight even if its
            # local configuration says Fold-CP is disabled. Otherwise a
            # rank-local mode mismatch lets enabled ranks wait forever in the
            # first object collective while disabled ranks skip this block.
            control_group, world_control_group, cp_rank = _create_foldcp_control_groups(
                self.foldcp_config
            )
            if self.foldcp_config.enabled:
                self.foldcp_control_group = control_group
                self.foldcp_world_control_group = world_control_group
                self.foldcp_cp_rank = cp_rank
                register_foldcp_cpu_control_group(self.foldcp_world_control_group)

                def _initialize_foldcp_mesh() -> None:
                    mesh = FoldCPProcessMesh.create(self.foldcp_config)
                    mesh.prewarm_communications()

                _run_runner_initialization_stage(
                    _initialize_foldcp_mesh,
                    stage="Fold-CP NCCL mesh and route initialization",
                    foldcp_config=self.foldcp_config,
                    world_control_group=self.foldcp_world_control_group,
                )

        compatible_configs: list[OpenDDEConfig] = []
        _run_runner_initialization_stage(
            lambda: compatible_configs.append(
                apply_runtime_compatibility(self.configs, self.device)
            ),
            stage="runtime compatibility initialization",
            foldcp_config=self.foldcp_config,
            world_control_group=self.foldcp_world_control_group,
        )
        self.configs = compatible_configs[0]
        if self.device.type == "mps":
            logging.info(
                "Apple MPS backend selected; dtype=%s, triangle kernels: "
                "multiplicative=%s, attention=%s.",
                self.configs.dtype,
                self.configs.triangle_multiplicative,
                self.configs.triangle_attention,
            )
        _validate_foldcp_runtime_config_consistency(
            self.configs,
            self.foldcp_config,
            self.foldcp_world_control_group,
        )

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "torch")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Selected inference device: %s", self.device)
        logging.info("Finished environment initialization.")

    def __enter__(self) -> "InferenceRunner":
        """Support deterministic resource cleanup for embedded Python callers."""
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def close(self) -> None:
        """Restore process-global state and release Runner-owned resources."""
        determinism_state = getattr(self, "_determinism_runtime_before_run", None)
        self._determinism_runtime_before_run = None
        if determinism_state is not None:
            _restore_determinism_runtime(determinism_state)

        ccd_paths = getattr(self, "_ccd_cache_paths_before_run", None)
        self._ccd_cache_paths_before_run = None
        if ccd_paths is not None:
            ccd.set_ccd_cache_paths(
                components_file=ccd_paths[0],
                rdkit_mol_pkl=ccd_paths[1],
            )

        previous_environment = self._foldcp_environment_before_publish
        self._foldcp_environment_before_publish = None
        if previous_environment is not None:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        # ``close()`` previously tore down process groups but kept the complete
        # model reachable from the Runner.  That is especially harmful when
        # constructor/checkpoint initialization fails: callers never receive
        # the half-built Runner, while its exception/cleanup lifetime can keep
        # all model parameters resident and make the next attempt OOM.  Drop
        # accelerator-owning state before communicator teardown.  Cleanup is
        # best-effort so a secondary allocator error cannot skip group teardown.
        try:
            if hasattr(self, "model"):
                self.model = None
            if hasattr(self, "dumper"):
                self.dumper = None
            device = getattr(self, "device", None)
            if device is not None:
                cleanup_device_memory(device, collect_garbage=True)
        except Exception:
            logger.exception("Failed to release InferenceRunner model resources.")

        control_group = getattr(self, "foldcp_control_group", None)
        world_control_group = getattr(self, "foldcp_world_control_group", None)
        if not dist.is_available() or not dist.is_initialized():
            if world_control_group is not None:
                unregister_foldcp_cpu_control_group(world_control_group)
            try:
                clear_foldcp_process_mesh_cache()
            except Exception:
                logger.exception("Failed to clear Fold-CP communication caches.")
            self.foldcp_control_group = None
            self.foldcp_world_control_group = None
            self._owns_process_group = False
            return

        control_group_destroyed = world_control_group is None
        if world_control_group is not None:
            try:
                _destroy_foldcp_control_groups(
                    control_group,
                    world_control_group,
                )
            except Exception:
                logger.exception("Failed to destroy the Fold-CP CPU control group.")
                # The group handle is no longer safe for post-OOM reporting even
                # if PyTorch could not confirm its destruction.  In particular,
                # an externally owned NCCL world returns below and a subsequent
                # Runner must be able to register its own control group.  Keep
                # the instance attributes so a second close() can retry the
                # actual destroy, but never leave the process-global registry
                # pointing at the failed handle.
                unregister_foldcp_cpu_control_group(world_control_group)
            else:
                control_group_destroyed = True
                unregister_foldcp_cpu_control_group(world_control_group)
                self.foldcp_control_group = None
                self.foldcp_world_control_group = None

        try:
            clear_foldcp_process_mesh_cache()
        except Exception:
            # Cache cleanup is local and best-effort.  It must never prevent a
            # Runner-owned default process group from being torn down below.
            logger.exception("Failed to clear Fold-CP communication caches.")

        if not self._owns_process_group:
            return

        try:
            dist.destroy_process_group()
        except Exception:
            logger.exception("Failed to destroy the Runner-owned process group.")
        else:
            self._owns_process_group = False
            # destroy_process_group() without an explicit group tears down the
            # complete Runner-owned distributed world.  If the earlier Gloo-
            # only destroy failed, its handle is no longer usable either; do
            # not retain the instance handle. The failure branch above already
            # removed it from the global error-control registry.
            if not control_group_destroyed and world_control_group is not None:
                self.foldcp_control_group = None
                self.foldcp_world_control_group = None

    def init_basics(self) -> None:
        """
        Initialize basic directory structures for dumping results and errors.
        """
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
        # ERR is the status of this Runner invocation, not an append-only log.
        # A successful retry in the same output directory must not retain a
        # stale OOM/disk/feature error from an earlier process. Only rank 0
        # publishes diagnostics; the synchronized initialization stage makes
        # every Fold-CP peer wait until this reset has completed.
        if _distributed_rank() == 0:
            if os.path.islink(self.error_dir) or os.path.isfile(self.error_dir):
                os.unlink(self.error_dir)
            elif os.path.isdir(self.error_dir):
                shutil.rmtree(self.error_dir)
            os.makedirs(self.error_dir, exist_ok=True)

    def init_model(self) -> None:
        """
        Initialize the OpenDDE model and move it to the appropriate device.
        """
        self.model = OpenDDE(self.configs).to(self.device)

    def load_checkpoint(self) -> None:
        """
        Load model weights from a checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint path does not exist.
        """
        checkpoint_path = resolve_checkpoint_path(self.configs)
        if not opexists(checkpoint_path):
            raise FileNotFoundError(
                f"Given checkpoint path not exist [{checkpoint_path}]"
            )

        self.print(
            f"Loading from {checkpoint_path}, strict: {self.configs.load_strict}"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        sample_key = list(checkpoint["model"].keys())[0]
        self.print(f"Sampled key: {sample_key}")
        if sample_key.startswith("module."):  # DDP checkpoint has module. prefix
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }
        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=self.configs.load_strict,
        )
        self.model.eval()
        self.print("Finish loading checkpoint.")

        def count_parameters(model: torch.nn.Module) -> float:
            """Count total parameters in millions."""
            total_params = sum(p.numel() for p in model.parameters())
            return total_params / 1e6

        self.print(f"Model parameters: {count_parameters(self.model):.2f}M")

    def init_dumper(
        self, need_atom_confidence: bool = False, sorted_by_ranking_score: bool = True
    ) -> None:
        """
        Initialize the data dumper for saving predictions.

        Args:
            need_atom_confidence (bool): Whether to dump atom-level confidence.
            sorted_by_ranking_score (bool): Whether to sort results by ranking score.
        """
        self.dumper = DataDumper(
            base_dir=self.dump_dir,
            need_atom_confidence=need_atom_confidence,
            sorted_by_ranking_score=sorted_by_ranking_score,
        )

    @torch.no_grad()
    def predict(self, data: Mapping[str, Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Run model prediction on the provided data.

        Args:
            data (Mapping[str, Mapping[str, Any]]): Input data dictionary.

        Returns:
            dict[str, torch.Tensor]: Prediction results.
        """
        eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }[self.configs.dtype]

        if eval_precision == torch.float32 or self.device.type not in {"cuda", "mps"}:
            enable_amp = nullcontext()
        else:
            enable_amp = torch.autocast(
                device_type=self.device.type, dtype=eval_precision
            )
        world_control_group = getattr(self, "foldcp_world_control_group", None)

        prepared_input: list[tuple[str, int | None, Any]] = []

        def _prepare_model_input() -> None:
            sample_name = "unknown"
            if isinstance(data, Mapping):
                sample_name = str(data.get("sample_name", "unknown"))
            prepared_input.append(
                (
                    sample_name,
                    infer_n_token(data),
                    to_device(data, self.device),
                )
            )

        _run_rank_stage_synchronized(
            _prepare_model_input,
            stage="model input device transfer",
            foldcp_config=self.foldcp_config,
            world_control_group=world_control_group,
        )
        sample_name, n_token, data = prepared_input[0]

        with ExitStack() as metric_stack:

            def _enter_model_contexts() -> None:
                # Preserve the original context ordering: autocast first, then
                # CUDA peak/timing measurement.
                metric_stack.enter_context(enable_amp)
                metric_stack.enter_context(
                    use_serial_model_when_cp_has_padding_only_ranks(
                        self.foldcp_config,
                        n_token,
                    )
                )
                metric_stack.enter_context(
                    measure_foldcp_stage(
                        task_id="task0",
                        stage_name="model_forward",
                        foldcp_config=self.foldcp_config,
                        recorder=self.foldcp_recorder,
                        sample_name=sample_name,
                        n_token=n_token,
                        device=self.device,
                    )
                )

            _run_rank_stage_synchronized(
                _enter_model_contexts,
                stage="model-forward metric initialization",
                foldcp_config=self.foldcp_config,
                world_control_group=world_control_group,
            )
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        return prediction

    def print(self, msg: str) -> None:
        """
        Print message only on the master rank (rank 0).

        Args:
            msg (str): Message to print.
        """
        if _distributed_rank() == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: OpenDDEConfig) -> None:
        """
        Update per-inference runtime configuration without rebuilding the model.

        Args:
            new_configs (OpenDDEConfig): New configuration object.
        """
        self.configs = new_configs
        self.model.configs = new_configs
        # OpenDDE snapshots these hot-path switches at construction time. Keep
        # them coherent when an embedded caller supplies a fresh config object.
        self.model.enable_diffusion_shared_vars_cache = (
            new_configs.enable_diffusion_shared_vars_cache
        )
        self.model.enable_efficient_fusion = new_configs.enable_efficient_fusion
        self.model.N_cycle = new_configs.model.N_cycle
        self.model.N_model_seed = new_configs.model.N_model_seed


def update_inference_configs(configs: OpenDDEConfig, n_token: int) -> OpenDDEConfig:
    """
    Adjust inference configurations based on the number of tokens to avoid OOM.

    Args:
        configs (OpenDDEConfig): Original configurations.
        n_token (int): Number of tokens in the sample.

    Returns:
        OpenDDEConfig: Updated configurations.
    """
    # Adjust configurations based on sequence length to manage memory usage
    if n_token > 3840:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True

    if os.getenv("OPENDDE_FORCE_SAMPLE_DIFFUSION_AMP") == "1":
        configs.skip_amp.sample_diffusion = False
    if os.getenv("OPENDDE_FORCE_CONFIDENCE_AMP") == "1":
        configs.skip_amp.confidence_head = False

    return configs


def _prepare_prediction_batch(
    runner: InferenceRunner,
    configs: OpenDDEConfig,
    data: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[int, int, int, int, int]:
    """Prepare rank-local model metadata before any model collective."""

    sample_index = int(data["sample_index"])
    dimensions = {
        key: _tensor_scalar(data, key)
        for key in ("N_asym", "N_token", "N_atom", "N_msa")
    }
    missing = [key for key, value in dimensions.items() if value is None]
    if missing:
        raise KeyError(f"Missing required inference dimensions: {missing}")
    n_asym = cast(int, dimensions["N_asym"])
    n_token = cast(int, dimensions["N_token"])
    n_atom = cast(int, dimensions["N_atom"])
    n_msa = cast(int, dimensions["N_msa"])
    input_features = cast(dict[str, Any], data["input_feature_dict"])
    input_features["inference_seed"] = torch.tensor(
        int(seed),
        dtype=torch.long,
    )
    runner.update_model_configs(update_inference_configs(configs, n_token))
    return sample_index, n_asym, n_token, n_atom, n_msa


def _sampler_owns_inference_sample(sampler: Any, data: Mapping[str, Any]) -> bool:
    """Return whether this process owns the sample's persistent outputs.

    ``InferenceJobSampler`` may pad a multi-process data-parallel assignment so
    every worker executes the same number of forwards.  Padded workers must
    participate in compute but must never overwrite the real owner's files.
    Every rank in maintained 1 x P Fold-CP owns the same job, with ``cp_rank``
    selecting its sole output writer separately.
    """

    return not isinstance(sampler, InferenceJobSampler) or sampler.owns(
        data.get("sample_index", -1)
    )


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """Run inference with CPU control planes for job/error coordination."""
    control_group = getattr(runner, "foldcp_control_group", None)
    world_control_group = getattr(runner, "foldcp_world_control_group", None)
    cp_rank = int(getattr(runner, "foldcp_cp_rank", 0))
    if runner.foldcp_config.enabled and world_control_group is None:
        raise RuntimeError(
            "Distributed 1xP inference requires the Runner-owned Gloo control "
            "group to be initialized before prediction."
        )
    _infer_predict_impl(
        runner,
        configs,
        control_group=control_group,
        world_control_group=world_control_group,
        cp_rank=cp_rank,
    )


def _infer_predict_impl(
    runner: InferenceRunner,
    configs: Any,
    *,
    control_group: dist.ProcessGroup | None,
    world_control_group: dist.ProcessGroup | None,
    cp_rank: int,
) -> None:
    """
    Run the full inference process for the given runner and configurations.
    Processes all samples in the dataloader for each specified seed.

    Args:
        runner (InferenceRunner): The initialized runner instance.
        configs (Any): Inference configurations.
    """
    logger.info(f"Loading data from {configs.input_json_path}")
    json_data = _load_inference_jobs_synchronized(
        configs.input_json_path,
        world_control_group,
    )

    # Seed precedence is resolved independently for every JSON job:
    # command line > that job's modelSeeds > synchronized random seed.
    cli_seeds = [int(seed) for seed in configs.seeds] if configs.seeds else None
    job_seed_schedule = _resolve_job_seed_schedule(
        json_data,
        cli_seeds,
        world_control_group,
    )
    seeds = list(
        dict.fromkeys(seed for job_seeds in job_seed_schedule for seed in job_seeds)
    )
    if cli_seeds:
        logger.info(f"Using seeds from command line: {seeds}")
    else:
        logger.info(f"Using per-job JSON/default seed schedule: {job_seed_schedule}")

    try:
        dataloader = _create_inference_dataloader_synchronized(
            configs,
            json_data,
            runner.foldcp_config,
            world_control_group,
        )
    except Exception as exc:
        logger.error("Dataloader initialization failed: %s", exc)
        if _distributed_rank() == 0:
            _append_error_report(runner.error_dir, "error.txt", str(exc))
        raise

    num_data = _get_dataloader_size_synchronized(
        dataloader,
        runner.foldcp_config,
        world_control_group,
    )
    inference_errors: list[str] = []
    t0_start = time.time()
    with disable_cudnn_benchmark(runner.device):
        for seed in seeds:
            _run_rank_stage_synchronized(
                lambda: seed_everything(
                    seed=seed,
                    deterministic=configs.deterministic,
                ),
                stage=f"seed {seed} initialization",
                foldcp_config=runner.foldcp_config,
                world_control_group=world_control_group,
            )
            _run_rank_stage_synchronized(
                lambda: cleanup_device_memory(runner.device),
                stage=f"seed {seed} initial memory cleanup",
                foldcp_config=runner.foldcp_config,
                world_control_group=world_control_group,
            )
            t1_start = time.time()
            sampler = getattr(dataloader, "sampler", None)
            if isinstance(sampler, InferenceJobSampler):
                _run_rank_stage_synchronized(
                    lambda: sampler.set_sample_indices(
                        index
                        for index, job_seeds in enumerate(job_seed_schedule)
                        if seed in job_seeds
                    ),
                    stage=f"seed {seed} sampler configuration",
                    foldcp_config=runner.foldcp_config,
                    world_control_group=world_control_group,
                )
            dataloader_iterator = _create_dataloader_iterator_synchronized(
                dataloader,
                runner.foldcp_config,
                world_control_group,
            )
            batch_ordinal = 0
            while True:
                # A model seed belongs to each inference job, not to the process's
                # cumulative job stream.  Reset before ``next`` so featurization
                # and model sampling are independent of directory order and DP
                # assignment while every CP rank still consumes identical RNG.
                _run_rank_stage_synchronized(
                    lambda: seed_everything(
                        seed=seed,
                        deterministic=configs.deterministic,
                    ),
                    stage=f"seed {seed} batch {batch_ordinal} RNG reset",
                    foldcp_config=runner.foldcp_config,
                    world_control_group=world_control_group,
                )
                has_batch, batch = _next_inference_batch_synchronized(
                    dataloader_iterator,
                    runner.foldcp_config,
                    world_control_group,
                )
                if not has_batch:
                    break
                batch_ordinal += 1
                sample_name = "unknown"
                data = None
                atom_array = None
                prediction = None
                handled_error_label: str | None = None
                try:
                    t2_start = time.time()
                    data, atom_array, data_error_message = _prepare_inference_batch(
                        batch
                    )
                    sample_name = str(data["sample_name"])

                    data_error_message = _synchronize_foldcp_batch(
                        data=data,
                        data_error_message=data_error_message,
                        seed=seed,
                        group=control_group,
                        size_cp=getattr(runner.foldcp_config, "size_cp", 1),
                    )
                    world_error_message = _finalize_foldcp_batch_error(
                        data_error_message,
                        runner.foldcp_config,
                        control_group,
                        world_control_group,
                    )
                    if world_error_message:
                        logger.error(
                            f"Data error for {sample_name}: {world_error_message}"
                        )
                        if data_error_message and cp_rank == 0:
                            _append_error_report(
                                runner.error_dir,
                                f"{sample_name}.txt",
                                data_error_message,
                            )
                        inference_errors.append(
                            f"{sample_name} [seed:{seed}]: {world_error_message}"
                        )
                        handled_error_label = "data preparation error"
                        continue

                    prediction_error = ""
                    try:
                        prepared_batch: list[tuple[int, int, int, int, int]] = []
                        _run_rank_stage_synchronized(
                            lambda: prepared_batch.append(
                                _prepare_prediction_batch(
                                    runner,
                                    configs,
                                    data,
                                    seed=seed,
                                )
                            ),
                            stage=f"model batch preparation for {sample_name}",
                            foldcp_config=runner.foldcp_config,
                            world_control_group=world_control_group,
                        )
                        sample_index, n_asym, n_token, n_atom, n_msa = prepared_batch[0]
                        logger.info(
                            f"[Rank {_distributed_rank()} ({sample_index + 1}/{num_data})] "
                            f"{sample_name} [seed:{seed}]: N_asym {n_asym}, "
                            f"N_token {n_token}, N_atom {n_atom}, N_msa {n_msa}"
                        )
                        prediction = runner.predict(data)
                    except Exception as exc:
                        prediction_error = (
                            f"[Rank {_distributed_rank()}] model stage failed for "
                            f"{sample_name}: {exc}\n{traceback.format_exc()}"
                        )
                        # The failed model frame can retain its largest CUDA
                        # intermediates.  Detach it before allocator cleanup;
                        # otherwise empty_cache() runs while those tensors are
                        # still live and the next sample/seed inherits the
                        # failed allocation's high-water footprint.
                        detach_rank_local_error_traceback(exc)
                        prediction_error = _cleanup_after_model_error(
                            runner.device,
                            prediction_error,
                        )

                    world_prediction_error = _synchronize_foldcp_world_error(
                        prediction_error,
                        runner.foldcp_config,
                        world_control_group,
                    )
                    if world_prediction_error:
                        if runner.foldcp_config.enabled:
                            raise FoldCPJobCoordinationError(world_prediction_error)
                        raise RuntimeError(world_prediction_error)
                    dump_error = ""
                    owns_sample = _sampler_owns_inference_sample(sampler, data)
                    if cp_rank == 0 and owns_sample:
                        try:
                            runner.dumper.dump(
                                group_name="",
                                pdb_id=sample_name,
                                seed=seed,
                                pred_dict=prediction,
                                atom_array=atom_array,
                                entity_poly_type={
                                    k: v
                                    for k, v in data["entity_poly_type"].items()
                                    if v != "non-polymer"
                                },
                            )
                        except Exception as exc:
                            dump_error = (
                                f"[Rank {_distributed_rank()}] output stage failed for "
                                f"{sample_name}: {exc}\n{traceback.format_exc()}"
                            )
                    dump_error = _synchronize_foldcp_output_error(
                        dump_error,
                        group=control_group,
                        size_cp=getattr(runner.foldcp_config, "size_cp", 1),
                        foldcp_config=runner.foldcp_config,
                        world_control_group=world_control_group,
                    )
                    if dump_error:
                        if runner.foldcp_config.enabled:
                            raise FoldCPJobCoordinationError(dump_error)
                        raise RuntimeError(dump_error)
                    t2_end = time.time()
                    logger.info(
                        f"[Rank {_distributed_rank()}] {sample_name} [seed:{seed}] succeeded. "
                        f"Model forward time: {t2_end - t2_start:.2f}s. "
                        f"Results saved to {configs.dump_dir}"
                    )
                except Exception as e:
                    handled_error_label = f"{type(e).__name__}: {e}"
                    error_message = (
                        f"[Rank {_distributed_rank()}] {sample_name} failed: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    logger.error(error_message)
                    owns_sample = data is None or _sampler_owns_inference_sample(
                        sampler, data
                    )
                    if cp_rank == 0 and owns_sample:
                        _append_error_report(
                            runner.error_dir,
                            f"{sample_name}.txt",
                            error_message,
                        )
                    if isinstance(e, FoldCPJobCoordinationError):
                        raise
                    inference_errors.append(error_message)
                finally:
                    active_error = sys.exc_info()[1]
                    cleanup_error_label = handled_error_label
                    if active_error is not None:
                        cleanup_error_label = (
                            f"{type(active_error).__name__}: {active_error}"
                        )
                        if isinstance(active_error, Exception):
                            detach_rank_local_error_traceback(active_error)
                    # Drop both the CPU dataloader batch and the detached
                    # device-side feature tree on every path, including batch
                    # unpacking or model execution failing.
                    del batch, data, atom_array, prediction
                    _cleanup_batch_synchronized(
                        runner.device,
                        runner.foldcp_config,
                        active_error=cleanup_error_label,
                        world_control_group=world_control_group,
                    )
            # A multiprocessing DataLoader iterator owns worker processes,
            # prefetched batches, queues, and potentially pinned feature
            # buffers.  Do not keep the exhausted iterator alive through the
            # seed-boundary cleanup (or until the next seed overwrites it).
            del dataloader_iterator
            _run_rank_stage_synchronized(
                lambda: cleanup_device_memory(runner.device, synchronize=True),
                stage=f"seed {seed} final memory cleanup",
                foldcp_config=runner.foldcp_config,
                world_control_group=world_control_group,
            )
            t1_end = time.time()
            logger.info(
                f"[Rank {_distributed_rank()}] Seed {seed} completed in {t1_end - t1_start:.2f}s."
            )
    # Do not let rank 0 remove ERR while another 1xP rank may still write it.
    if world_control_group is not None:
        dist.barrier(group=world_control_group)
    if _distributed_rank() == 0 and opexists(runner.error_dir):
        try:
            if not os.listdir(runner.error_dir):
                os.rmdir(runner.error_dir)
        except Exception:
            pass

    t0_end = time.time()
    logger.info(
        f"[Rank {_distributed_rank()}] Job completed in {t0_end - t0_start:.2f}s."
    )
    if inference_errors:
        raise RuntimeError(
            f"{len(inference_errors)} inference sample(s) failed. "
            f"First error:\n{inference_errors[0]}"
        )


def main(configs: OpenDDEConfig) -> None:
    """
    Inference entry point.

    Args:
        configs (OpenDDEConfig): Inference configurations.
    """
    runner = InferenceRunner(configs)
    try:
        infer_predict(runner, runner.configs)
    finally:
        runner.close()


def run() -> None:
    """
    Initialize and execute the inference pipeline.
    """
    init_logging()

    try:
        arg_str = parse_sys_args()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    configs = build_inference_config(
        arg_str=arg_str,
        fill_required_with_null=True,
    )
    model_name = configs.model_name
    logger.info(
        f"Using params for model {model_name}: "
        f"cycle={configs.model.N_cycle}, step={configs.sample_diffusion.N_step}"
    )
    logger.info(
        f"Inference by OpenDDE: model_name: {model_name}, dtype: {configs.dtype}"
    )
    logger.info(
        f"Optimization: shared_vars_cache={configs.enable_diffusion_shared_vars_cache}, "
        f"efficient_fusion={configs.enable_efficient_fusion}, tf32={configs.enable_tf32}"
    )
    main(configs)


if __name__ == "__main__":
    run()
