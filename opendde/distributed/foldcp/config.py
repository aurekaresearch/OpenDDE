# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Runtime flags for single-process or multi-GPU 1 x P Fold-CP inference."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Literal

import torch.distributed as dist

FoldCPMode = Literal["single", "distributed"]
FOLDCP_ENVIRONMENT_KEYS = (
    "OPENDDE_FOLDCP_MODE",
    "OPENDDE_FOLDCP_SIZE_DP",
    "OPENDDE_FOLDCP_SIZE_CP",
    "OPENDDE_FOLDCP_DEVICES",
    "OPENDDE_FOLDCP_METRICS_JSONL",
)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class FoldCPConfig:
    """Validated Fold-CP launch configuration."""

    mode: FoldCPMode = "single"
    size_dp: int = 1
    size_cp: int = 1
    devices: str = ""
    metrics_jsonl: str = ""

    @classmethod
    def from_runtime_args(
        cls,
        *,
        mode: str = "single",
        size_dp: int = 1,
        size_cp: int = 1,
        devices: str = "",
        metrics_jsonl: str = "",
    ) -> "FoldCPConfig":
        return cls(
            mode=mode,  # type: ignore[arg-type]
            size_dp=int(size_dp),
            size_cp=int(size_cp),
            devices=devices,
            metrics_jsonl=metrics_jsonl,
        ).validate()

    @classmethod
    def from_config(cls, configs: Any) -> "FoldCPConfig":
        return cls.from_runtime_args(
            mode=getattr(configs, "foldcp_mode", "single"),
            size_dp=_as_int(getattr(configs, "foldcp_size_dp", 1), 1),
            size_cp=_as_int(getattr(configs, "foldcp_size_cp", 1), 1),
            devices=getattr(configs, "foldcp_devices", "") or "",
            metrics_jsonl=getattr(configs, "foldcp_metrics_jsonl", "") or "",
        )

    @classmethod
    def from_environment(cls) -> "FoldCPConfig":
        """Resolve process-global Fold-CP settings without a fixed-P fallback.

        The Runner publishes every field explicitly. Library users may instead
        enable distributed execution after initializing the default process
        group; in that case the maintained topology is 1xWORLD_SIZE, so infer
        P from the group rather than silently assuming the historical CP=4.
        """

        mode = os.environ.get("OPENDDE_FOLDCP_MODE", "single")
        size_dp = _as_int(os.environ.get("OPENDDE_FOLDCP_SIZE_DP"), 1)
        raw_size_cp = os.environ.get("OPENDDE_FOLDCP_SIZE_CP")
        if raw_size_cp in {None, ""}:
            size_cp = (
                dist.get_world_size()
                if mode == "distributed"
                and dist.is_available()
                and dist.is_initialized()
                else 1
            )
        else:
            size_cp = int(raw_size_cp)
        return cls.from_runtime_args(
            mode=mode,
            size_dp=size_dp,
            size_cp=size_cp,
            devices=os.environ.get("OPENDDE_FOLDCP_DEVICES", ""),
            metrics_jsonl=os.environ.get("OPENDDE_FOLDCP_METRICS_JSONL", ""),
        )

    def validate(self) -> "FoldCPConfig":
        if self.mode not in {"single", "distributed"}:
            raise ValueError("foldcp_mode must be 'single' or 'distributed'.")
        if self.size_dp != 1:
            raise ValueError(
                "Only the maintained 1 x P Fold-CP topology is supported; "
                "foldcp_size_dp must be 1."
            )
        if self.size_cp < 1:
            raise ValueError("foldcp_size_cp must be >= 1.")
        if self.mode == "single" and self.size_cp != 1:
            raise ValueError("foldcp_mode='single' requires foldcp_size_cp=1.")
        if self.mode == "distributed":
            if self.size_cp == 1:
                raise ValueError(
                    "foldcp_mode='distributed' requires foldcp_size_cp > 1."
                )
        return self

    @property
    def enabled(self) -> bool:
        return self.mode == "distributed"

    @property
    def cp_mesh_shape(self) -> tuple[int, int]:
        if not self.enabled:
            return (1, 1)
        return (1, self.size_cp)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["enabled"] = self.enabled
        data["cp_mesh_shape"] = self.cp_mesh_shape
        return data

    def launch_hint(self) -> str:
        if self.enabled:
            nproc = self.size_cp
            return (
                f"torchrun --nproc_per_node {nproc} -m runner.batch_inference pred "
                f"--foldcp_mode distributed --foldcp_size_dp {self.size_dp} "
                f"--foldcp_size_cp {self.size_cp}"
            )
        return (
            "python -m runner.batch_inference pred "
            "--foldcp_mode single --foldcp_size_cp 1"
        )


def apply_foldcp_config(configs: Any, foldcp: FoldCPConfig) -> Any:
    """Attach validated Fold-CP settings to the mutable OpenDDE config object."""

    values = (
        foldcp.mode,
        str(foldcp.size_dp),
        str(foldcp.size_cp),
        foldcp.devices,
        foldcp.metrics_jsonl,
    )
    for key, value in zip(FOLDCP_ENVIRONMENT_KEYS, values, strict=True):
        os.environ[key] = value

    configs.foldcp_mode = foldcp.mode
    configs.foldcp_size_dp = foldcp.size_dp
    configs.foldcp_size_cp = foldcp.size_cp
    configs.foldcp_devices = foldcp.devices
    configs.foldcp_metrics_jsonl = foldcp.metrics_jsonl
    configs.foldcp = foldcp.to_dict()
    return configs


@contextmanager
def use_serial_model_when_cp_has_padding_only_ranks(
    foldcp: FoldCPConfig,
    n_token: int | None,
) -> Iterator[bool]:
    """Run the model's original serial path when a 1 x P launch has P > N.

    Context parallelism cannot reduce an input below one token column per
    rank. When ``P > N``, the additional ranks contain padding only, so the
    distributed kernels add communication and can select different tiny GEMM
    launch families without saving useful compute or memory. All ranks instead
    execute the unchanged serial model for this input; the existing distributed
    runner still coordinates errors and retains only its normal output rank.
    """

    use_serial = bool(
        foldcp.enabled and n_token is not None and int(n_token) < int(foldcp.size_cp)
    )
    if not use_serial:
        yield False
        return

    previous = {key: os.environ.get(key) for key in FOLDCP_ENVIRONMENT_KEYS}
    os.environ["OPENDDE_FOLDCP_MODE"] = "single"
    os.environ["OPENDDE_FOLDCP_SIZE_DP"] = "1"
    os.environ["OPENDDE_FOLDCP_SIZE_CP"] = "1"
    try:
        yield True
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
