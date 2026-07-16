# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Benchmark and validation metrics for Fold-CP tasks."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from opendde.distributed.foldcp.config import FoldCPConfig

_MIB = 1024 * 1024


def bytes_to_mib(value: int | float | None) -> Optional[float]:
    if value is None:
        return None
    return float(value) / _MIB


def _cuda_available(device: Optional[torch.device] = None) -> bool:
    if device is not None and device.type != "cuda":
        return False
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _sync_device(device: Optional[torch.device] = None) -> None:
    if _cuda_available(device):
        torch.cuda.synchronize(device=device)


def _cuda_mem_info_mib(
    device: Optional[torch.device] = None,
) -> tuple[Optional[float], Optional[float]]:
    if not _cuda_available(device):
        return None, None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
    return bytes_to_mib(free_bytes), bytes_to_mib(total_bytes)


def _scalar_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "item"):
        return int(value.item())
    if isinstance(value, (list, tuple)) and value:
        return _scalar_int(value[0])
    return int(value)


def infer_n_token(data: Any) -> Optional[int]:
    if not isinstance(data, dict):
        return None
    if "N_token" in data:
        return _scalar_int(data["N_token"])
    input_feature_dict = data.get("input_feature_dict", {})
    token_index = (
        input_feature_dict.get("token_index")
        if isinstance(input_feature_dict, dict)
        else None
    )
    if token_index is not None and hasattr(token_index, "shape"):
        return int(token_index.shape[-1])
    return None


@dataclass(frozen=True)
class FoldCPStageMetric:
    task_id: str
    stage_name: str
    mode: str
    size_dp: int
    size_cp: int
    cp_mesh_shape: tuple[int, int]
    sample_name: str
    n_token: Optional[int]
    elapsed_ms: float
    stage_peak_mib: Optional[float]
    total_peak_mib: Optional[float]
    allocated_after_mib: Optional[float]
    rank: int = 0
    device_index: Optional[int] = None
    cuda_free_mib: Optional[float] = None
    cuda_total_mib: Optional[float] = None
    precision_kind: str = "not_checked"
    bitwise_equal: Optional[bool] = None
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    status: str = "ok"
    oom_stage: str = ""
    error: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cp_mesh_shape"] = list(self.cp_mesh_shape)
        return data


class FoldCPBenchmarkRecorder:
    """Append-only JSONL writer for Fold-CP benchmark records."""

    def __init__(
        self,
        jsonl_path: str = "",
        rank: int = 0,
        write_rank_sidecar: bool = False,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.rank = rank
        self.write_rank_sidecar = write_rank_sidecar
        self.records: list[FoldCPStageMetric] = []

    def record(self, metric: FoldCPStageMetric) -> None:
        self.records.append(metric)
        if not self.jsonl_path:
            return
        path = Path(self.jsonl_path)
        if self.rank == 0:
            self._write_jsonl(path, metric)
        if self.write_rank_sidecar:
            self._write_jsonl(Path(f"{self.jsonl_path}.rank{self.rank}"), metric)

    @staticmethod
    def _write_jsonl(path: Path, metric: FoldCPStageMetric) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric.to_json_dict(), sort_keys=True) + "\n")


@contextmanager
def measure_foldcp_stage(
    *,
    task_id: str,
    stage_name: str,
    foldcp_config: FoldCPConfig,
    recorder: FoldCPBenchmarkRecorder,
    sample_name: str = "unknown",
    n_token: Optional[int] = None,
    reset_peak: bool = True,
    record_start: bool = False,
    device: Optional[torch.device] = None,
) -> Iterator[None]:
    """Measure wall time and CUDA peak memory for one Fold-CP task stage."""

    if reset_peak and _cuda_available(device):
        torch.cuda.reset_peak_memory_stats(device=device)
    _sync_device(device)
    rank = recorder.rank
    device_index = (
        device.index if device is not None and device.type == "cuda" else None
    )
    if device_index is None and _cuda_available(device):
        device_index = torch.cuda.current_device()
    if record_start:
        if _cuda_available(device):
            start_peak = bytes_to_mib(torch.cuda.max_memory_allocated(device=device))
            start_allocated = bytes_to_mib(torch.cuda.memory_allocated(device=device))
        else:
            start_peak = None
            start_allocated = None
        start_free, start_total = _cuda_mem_info_mib(device)
        recorder.record(
            FoldCPStageMetric(
                task_id=task_id,
                stage_name=stage_name,
                mode=foldcp_config.mode,
                size_dp=foldcp_config.size_dp,
                size_cp=foldcp_config.size_cp,
                cp_mesh_shape=foldcp_config.cp_mesh_shape,
                sample_name=sample_name,
                rank=rank,
                device_index=device_index,
                n_token=n_token,
                elapsed_ms=0.0,
                stage_peak_mib=start_peak,
                total_peak_mib=start_peak,
                allocated_after_mib=start_allocated,
                cuda_free_mib=start_free,
                cuda_total_mib=start_total,
                status="started",
            )
        )
    start = time.perf_counter()
    status = "ok"
    error = ""
    try:
        yield
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        error = str(exc).splitlines()[0]
        raise
    except Exception as exc:
        status = "error"
        error = str(exc).splitlines()[0]
        raise
    finally:
        _sync_device(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if _cuda_available(device):
            peak = bytes_to_mib(torch.cuda.max_memory_allocated(device=device))
            allocated_after = bytes_to_mib(torch.cuda.memory_allocated(device=device))
        else:
            peak = None
            allocated_after = None
        cuda_free, cuda_total = _cuda_mem_info_mib(device)
        recorder.record(
            FoldCPStageMetric(
                task_id=task_id,
                stage_name=stage_name,
                mode=foldcp_config.mode,
                size_dp=foldcp_config.size_dp,
                size_cp=foldcp_config.size_cp,
                cp_mesh_shape=foldcp_config.cp_mesh_shape,
                sample_name=sample_name,
                rank=rank,
                device_index=device_index,
                n_token=n_token,
                elapsed_ms=elapsed_ms,
                stage_peak_mib=peak,
                total_peak_mib=peak,
                allocated_after_mib=allocated_after,
                cuda_free_mib=cuda_free,
                cuda_total_mib=cuda_total,
                status=status,
                oom_stage=stage_name if status == "oom" else "",
                error=error,
            )
        )
