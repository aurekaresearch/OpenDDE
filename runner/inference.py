# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
import logging
import os
import random
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence, Sized
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from os.path import exists as opexists
from os.path import join as opjoin
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.distributed as dist

from opendde.config.config import parse_sys_args
from opendde.config.inference import (
    apply_runtime_compatibility,
    build_inference_config,
)
from opendde.config.schema import OpenDDEConfig
from opendde.data.inference.infer_dataloader import get_inference_dataloader
from opendde.distributed.foldcp.config import (
    FOLDCP_ENVIRONMENT_KEYS,
    FoldCPConfig,
    apply_foldcp_config,
)
from opendde.distributed.foldcp.metrics import (
    FoldCPBenchmarkRecorder,
    infer_n_token,
    measure_foldcp_stage,
)
from opendde.model.opendde import OpenDDE
from opendde.model.seed_batch import stack_seed_batch_features
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


def _default_foldcp_cuda_memory_fraction(size_cp: int) -> float:
    """Do not impose an artificial allocator limit by default."""

    return 0.0


def _download_inference_assets(configs: OpenDDEConfig) -> None:
    """Prepare shared inference assets once and synchronize all ranks."""
    if DIST_WRAPPER.world_size <= 1:
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

    dist.broadcast_object_list(download_status, src=0)
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
        self._foldcp_environment_before_publish: dict[str, str | None] | None = None
        try:
            self.foldcp_config = (
                foldcp_config
                if foldcp_config is not None
                else FoldCPConfig.from_config(configs)
            )
            self.configs = configs
            self.foldcp_recorder = FoldCPBenchmarkRecorder(
                self.foldcp_config.metrics_jsonl,
                rank=DIST_WRAPPER.rank,
            )
            self.init_env()
            _download_inference_assets(self.configs)
            self.init_basics()
            with skip_random_init() if self.configs.load_strict else nullcontext():
                self.init_model()
            self.load_checkpoint()
            self.init_dumper(
                need_atom_confidence=self.configs.need_atom_confidence,
                sorted_by_ranking_score=self.configs.sorted_by_ranking_score,
            )
            # Fold-CP is process-global today; publish it only after initialization
            # succeeds and retain the previous state for close().
            self._foldcp_environment_before_publish = {
                key: os.environ.get(key) for key in FOLDCP_ENVIRONMENT_KEYS
            }
            self.configs = apply_foldcp_config(self.configs, self.foldcp_config)
        except BaseException:
            self.close()
            raise

    def init_env(self) -> None:
        """
        Initialize the execution environment, including CUDA and distributed setup.
        """
        self.print(
            f"Distributed environment: world size: {DIST_WRAPPER.world_size}, "
            f"global rank: {DIST_WRAPPER.rank}, local rank: {DIST_WRAPPER.local_rank}"
        )
        expected_world_size = self.foldcp_config.size_dp * self.foldcp_config.size_cp
        seed_batch_size = getattr(self.configs, "seed_batch_size", 1)
        if seed_batch_size < 1:
            raise ValueError(f"seed_batch_size must be >= 1; got {seed_batch_size}.")
        if self.foldcp_config.size_cp > 1 and seed_batch_size > 1:
            raise ValueError(
                "Seed batching is supported by the normal P=1 model path only; "
                f"got foldcp_size_cp={self.foldcp_config.size_cp} and "
                f"seed_batch_size={seed_batch_size}."
            )
        if getattr(self.configs, "num_workers", 0) > 0 and seed_batch_size > 1:
            raise ValueError(
                "Seed batching requires num_workers=0 so each seed's featurization "
                "RNG stream can be preserved across input records."
            )
        model = getattr(self.configs, "model", None)
        n_model_seed = getattr(model, "N_model_seed", 1)
        if seed_batch_size > 1 and n_model_seed > 1:
            raise ValueError(
                "Seed batching requires model.N_model_seed=1; "
                f"got model.N_model_seed={n_model_seed}."
            )
        sample_diffusion = getattr(self.configs, "sample_diffusion", None)
        guidance = getattr(sample_diffusion, "guidance", {})
        if seed_batch_size > 1 and bool(guidance.get("enable", False)):
            raise ValueError(
                "Seed batching cannot be combined with Training-Free Guidance."
            )
        if DIST_WRAPPER.world_size != expected_world_size:
            raise RuntimeError(
                "Inference topology requires "
                f"WORLD_SIZE={expected_world_size} for "
                f"foldcp_size_dp={self.foldcp_config.size_dp}, "
                f"foldcp_size_cp={self.foldcp_config.size_cp}; got "
                f"WORLD_SIZE={DIST_WRAPPER.world_size}. Example: "
                f"{self.foldcp_config.launch_hint()}"
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
            self._configure_foldcp_cuda_memory_fraction()

        self.configs = apply_runtime_compatibility(self.configs, self.device)

        if DIST_WRAPPER.world_size > 1:
            if not self.use_cuda:
                raise RuntimeError(
                    "Distributed inference requires NVIDIA CUDA; CPU "
                    "supports single-process inference only."
                )
            if not dist.is_nccl_available():
                raise RuntimeError(
                    "Distributed inference requires the NCCL backend, "
                    "which is unavailable in this PyTorch build. Windows "
                    "distributed inference is not currently supported."
                )
            if dist.is_initialized():
                if dist.get_backend() != "nccl":
                    raise RuntimeError(
                        "Distributed inference requires an NCCL process group."
                    )
            else:
                dist.init_process_group(
                    backend="nccl", timeout=_DISTRIBUTED_STARTUP_TIMEOUT
                )
                self._owns_process_group = True

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "torch")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Selected inference device: %s", self.device)
        logging.info("Finished environment initialization.")

    def _configure_foldcp_cuda_memory_fraction(self) -> None:
        if not self.foldcp_config.enabled:
            return
        default_fraction = _default_foldcp_cuda_memory_fraction(
            self.foldcp_config.size_cp
        )
        value = os.environ.get(
            "OPENDDE_FOLDCP_CUDA_MEMORY_FRACTION",
            str(default_fraction),
        )
        fraction = float(value)
        if fraction <= 0:
            return
        if fraction > 1:
            raise ValueError(
                "OPENDDE_FOLDCP_CUDA_MEMORY_FRACTION must be in (0, 1] or "
                f"non-positive to disable; got {fraction}"
            )
        torch.cuda.set_per_process_memory_fraction(
            fraction,
            device=DIST_WRAPPER.local_rank,
        )
        logging.info(
            "Fold-CP CUDA allocator memory fraction: %.3f",
            fraction,
        )

    def close(self) -> None:
        """Restore process-global state and release Runner-owned resources."""
        previous_environment = self._foldcp_environment_before_publish
        self._foldcp_environment_before_publish = None
        if previous_environment is not None:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        if not self._owns_process_group:
            return
        if not dist.is_available() or not dist.is_initialized():
            self._owns_process_group = False
            return

        try:
            dist.destroy_process_group()
        except Exception:
            logger.exception("Failed to destroy the Runner-owned process group.")
        else:
            self._owns_process_group = False

    def init_basics(self) -> None:
        """
        Initialize basic directory structures for dumping results and errors.
        """
        self.dump_dir = self.configs.dump_dir
        self.error_dir = opjoin(self.dump_dir, "ERR")
        os.makedirs(self.dump_dir, exist_ok=True)
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
    def predict(
        self,
        data: Mapping[str, Mapping[str, Any]],
        *,
        msa_generators: Optional[Sequence[torch.Generator]] = None,
    ) -> Any:
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

        enable_amp = (
            torch.autocast(device_type="cuda", dtype=eval_precision)
            if self.use_cuda and eval_precision != torch.float32
            else nullcontext()
        )

        sample_name = "unknown"
        if isinstance(data, Mapping):
            sample_name = str(data.get("sample_name", "unknown"))
        n_token = infer_n_token(data)

        data = to_device(data, self.device)
        with (
            enable_amp,
            measure_foldcp_stage(
                task_id="task0",
                stage_name="model_forward",
                foldcp_config=self.foldcp_config,
                recorder=self.foldcp_recorder,
                sample_name=sample_name,
                n_token=n_token,
                device=self.device,
            ),
        ):
            prediction, _, _ = self.model(
                input_feature_dict=data["input_feature_dict"],
                label_full_dict=None,
                label_dict=None,
                mode="inference",
                msa_generators=msa_generators,
            )

        return prediction

    @torch.no_grad()
    def predict_seed_batch(
        self,
        data_batch: Sequence[Mapping[str, Any]],
        seeds: Sequence[int],
        *,
        msa_generators: Optional[Sequence[torch.Generator]] = None,
    ) -> list[dict[str, Any]]:
        """Run one model forward for a rank-local batch of independent seeds."""
        if not data_batch or len(data_batch) != len(seeds):
            raise ValueError(
                "Seed-batched prediction requires one data record per seed; "
                f"got {len(data_batch)} records and {len(seeds)} seeds."
            )

        sample_names = {str(data.get("sample_name", "unknown")) for data in data_batch}
        if len(sample_names) != 1:
            raise ValueError(
                f"Seed-batched data must describe one sample; got {sorted(sample_names)}."
            )

        data = dict(data_batch[0])
        data["input_feature_dict"] = stack_seed_batch_features(
            [data["input_feature_dict"] for data in data_batch],
            seeds,
        )
        predictions = self.predict(data, msa_generators=msa_generators)

        if len(seeds) == 1 and isinstance(predictions, dict):
            return [predictions]

        if not isinstance(predictions, list) or len(predictions) != len(seeds):
            raise RuntimeError(
                "Seed-batched model output does not match the requested seeds: "
                f"seeds={list(seeds)}, output_type={type(predictions).__name__}, "
                f"output_count={len(predictions) if isinstance(predictions, list) else 'n/a'}."
            )
        return predictions

    def print(self, msg: str) -> None:
        """
        Print message only on the master rank (rank 0).

        Args:
            msg (str): Message to print.
        """
        if DIST_WRAPPER.rank == 0:
            logger.info(msg)

    def update_model_configs(self, new_configs: OpenDDEConfig) -> None:
        """
        Update the model's configuration.

        Args:
            new_configs (OpenDDEConfig): New configuration object.
        """
        self.model.configs = new_configs


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


def _resolve_local_inference_seeds(configs: Any) -> list[int]:
    with open(configs.input_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list) or len(json_data) == 0:
        raise ValueError(
            f"Input JSON must be a non-empty top-level list, got {type(json_data).__name__} "
            f"from {configs.input_json_path}"
        )

    # Seed precedence: command line (configs.seeds) > JSON modelSeeds > random.
    cli_seeds = configs.seeds
    json_seeds = json_data[0].get("modelSeeds")
    if cli_seeds:
        seeds = [int(i) for i in cli_seeds]
        logger.info(f"Using seeds from command line: {seeds}")
    elif json_seeds:
        seeds = [int(i) for i in json_seeds]
        logger.info(f"Using modelSeeds from JSON: {seeds}")
    else:
        seeds = [random.randint(1, 65536)]
        logger.info(f"No seeds provided; sampled random seed: {seeds}")
    return seeds


def _validate_seed_parallel_seeds(
    seeds: list[int], size_dp: int, seed_batch_size: int = 1
) -> None:
    if seed_batch_size < 1:
        raise ValueError(f"seed_batch_size must be >= 1; got {seed_batch_size}.")
    if (size_dp > 1 or seed_batch_size > 1) and len(set(seeds)) != len(seeds):
        raise ValueError(
            "Seed-parallel inference requires unique seeds when sharding or batching; "
            f"got {seeds} for foldcp_size_dp={size_dp}, "
            f"seed_batch_size={seed_batch_size}."
        )
    if size_dp > 1 and len(seeds) < size_dp:
        raise ValueError(
            "Seed-parallel inference requires at least one seed per lane; "
            f"got {len(seeds)} seeds for foldcp_size_dp={size_dp}."
        )


def _seeds_for_rank(seeds: list[int], size_dp: int, rank: int) -> list[int]:
    if size_dp <= 1:
        return seeds
    return seeds[rank::size_dp]


def _seed_batches_for_rank(
    seeds: list[int], size_dp: int, rank: int, seed_batch_size: int
) -> list[list[int]]:
    local_seeds = _seeds_for_rank(seeds, size_dp, rank)
    return [
        local_seeds[start : start + seed_batch_size]
        for start in range(0, len(local_seeds), seed_batch_size)
    ]


def _resolve_inference_seeds(
    configs: Any, size_dp: int, seed_batch_size: int = 1
) -> list[int]:
    """Resolve and validate seeds on rank 0, then share them with peers."""
    if DIST_WRAPPER.world_size <= 1:
        seeds = _resolve_local_inference_seeds(configs)
        _validate_seed_parallel_seeds(seeds, size_dp, seed_batch_size)
        return seeds

    local_error: Exception | None = None
    status: list[tuple[bool, list[int] | None, str] | None] = [None]
    if DIST_WRAPPER.rank == 0:
        try:
            seeds = _resolve_local_inference_seeds(configs)
            _validate_seed_parallel_seeds(seeds, size_dp, seed_batch_size)
            status[0] = (True, seeds, "")
        except Exception as exc:
            local_error = exc
            status[0] = (False, None, f"{type(exc).__name__}: {exc}")

    dist.broadcast_object_list(status, src=0)
    result = status[0]
    if result is None:
        raise RuntimeError("Rank 0 broadcast an invalid seed-resolution status.")

    succeeded, seeds, error_message = result
    if not succeeded:
        error = RuntimeError(f"Seed resolution failed on rank 0: {error_message}")
        if local_error is not None:
            raise error from local_error
        raise error
    if not isinstance(seeds, list):
        raise RuntimeError("Rank 0 broadcast invalid inference seeds.")
    return seeds


@dataclass
class _SeedDataLane:
    iterator: Iterator[Any]
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_state: torch.Tensor


def _capture_featurization_random_state() -> tuple[
    object, tuple[Any, ...], torch.Tensor
]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _restore_featurization_random_state(lane: _SeedDataLane) -> None:
    random.setstate(lane.python_state)
    np.random.set_state(lane.numpy_state)
    torch.random.set_rng_state(lane.torch_state)


def _seed_data_lanes(
    dataloader: Any,
    seeds: Sequence[int],
    deterministic: bool,
) -> list[_SeedDataLane]:
    lanes = []
    for seed in seeds:
        seed_everything(seed=int(seed), deterministic=deterministic)
        iterator = iter(dataloader)
        python_state, numpy_state, torch_state = _capture_featurization_random_state()
        lanes.append(_SeedDataLane(iterator, python_state, numpy_state, torch_state))
    return lanes


def _next_seed_data(lane: _SeedDataLane) -> Any:
    _restore_featurization_random_state(lane)
    batch = next(lane.iterator)
    lane.python_state, lane.numpy_state, lane.torch_state = (
        _capture_featurization_random_state()
    )
    return batch


def _seed_batch_msa_generators(
    lanes: Sequence[_SeedDataLane],
    seeds: Sequence[int],
    device: torch.device,
) -> tuple[torch.Generator, ...]:
    generators = []
    for lane, seed in zip(lanes, seeds):
        generator = torch.Generator(device=device)
        if device.type == "cpu":
            generator.set_state(lane.torch_state)
        else:
            generator.manual_seed(int(seed))
        generators.append(generator)
    return tuple(generators)


def _sync_cpu_msa_generators(
    lanes: Sequence[_SeedDataLane],
    generators: Sequence[torch.Generator],
    *,
    before_model: bool,
    device: torch.device,
) -> None:
    if device.type != "cpu":
        return
    for lane, generator in zip(lanes, generators):
        if before_model:
            generator.set_state(lane.torch_state)
        else:
            lane.torch_state = generator.get_state()


def _infer_seed_batch_record(
    runner: InferenceRunner,
    configs: Any,
    lanes: Sequence[_SeedDataLane],
    seed_batch: Sequence[int],
    msa_generators: Sequence[torch.Generator],
    num_data: int,
) -> None:
    sample_name = "unknown"
    model_batch_size = 0
    try:
        seed_records = [_next_seed_data(lane)[0] for lane in lanes]
        sample_names = {record[0]["sample_name"] for record in seed_records}
        if len(sample_names) != 1:
            raise RuntimeError(
                f"Seed data lanes produced different samples: {sorted(sample_names)}."
            )
        sample_name = next(iter(sample_names))

        valid_indices = []
        for index, (seed, record) in enumerate(zip(seed_batch, seed_records)):
            _, _, data_error_message = record
            if data_error_message:
                logger.error(
                    f"Data error for {sample_name} [seed:{seed}]: {data_error_message}"
                )
                with open(
                    opjoin(runner.error_dir, f"{sample_name}.txt"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(data_error_message)
            else:
                valid_indices.append(index)
        if not valid_indices:
            return

        valid_seeds = [seed_batch[index] for index in valid_indices]
        model_batch_size = len(valid_seeds)
        valid_records = [seed_records[index] for index in valid_indices]
        valid_data = [record[0] for record in valid_records]
        valid_lanes = [lanes[index] for index in valid_indices]
        valid_generators = [msa_generators[index] for index in valid_indices]
        data = valid_data[0]
        start_time = time.time()

        logger.info(
            f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] "
            f"{sample_name} [seeds:{valid_seeds}]: "
            f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
            f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
        )
        new_configs = update_inference_configs(configs, data["N_token"].item())
        runner.update_model_configs(new_configs)
        _sync_cpu_msa_generators(
            valid_lanes,
            valid_generators,
            before_model=True,
            device=runner.device,
        )
        try:
            predictions = runner.predict_seed_batch(
                valid_data,
                valid_seeds,
                msa_generators=valid_generators,
            )
        finally:
            _sync_cpu_msa_generators(
                valid_lanes,
                valid_generators,
                before_model=False,
                device=runner.device,
            )

        if not (runner.foldcp_config.enabled and DIST_WRAPPER.rank != 0):
            for seed, record, prediction in zip(
                valid_seeds, valid_records, predictions
            ):
                seed_data, atom_array, _ = record
                runner.dumper.dump(
                    group_name="",
                    pdb_id=sample_name,
                    seed=seed,
                    pred_dict=prediction,
                    atom_array=atom_array,
                    entity_poly_type={
                        key: value
                        for key, value in seed_data["entity_poly_type"].items()
                        if value != "non-polymer"
                    },
                )
        logger.info(
            f"[Rank {DIST_WRAPPER.rank}] {sample_name} "
            f"[seeds:{valid_seeds}] succeeded. "
            f"Model forward time: {time.time() - start_time:.2f}s. "
            f"Results saved to {configs.dump_dir}"
        )
    except Exception as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) and model_batch_size > 1:
            logger.exception(
                "[Rank %s] %s seed batch %s ran out of CUDA memory.",
                DIST_WRAPPER.rank,
                sample_name,
                list(seed_batch),
            )
            raise
        error_message = (
            f"[Rank {DIST_WRAPPER.rank}] {sample_name} failed: {exc}\n"
            f"{traceback.format_exc()}"
        )
        logger.error(error_message)
        with open(
            opjoin(runner.error_dir, f"{sample_name}.txt"),
            "a",
            encoding="utf-8",
        ) as f:
            f.write(error_message)


def infer_predict(runner: InferenceRunner, configs: Any) -> None:
    """
    Run the full inference process for the given runner and configurations.
    Processes all samples in the dataloader for each specified seed.

    Args:
        runner (InferenceRunner): The initialized runner instance.
        configs (Any): Inference configurations.
    """
    # Data loading
    logger.info(f"Loading data from {configs.input_json_path}")
    size_dp = runner.foldcp_config.size_dp
    seed_batch_size = getattr(configs, "seed_batch_size", 1)
    seeds = _resolve_inference_seeds(configs, size_dp, seed_batch_size)
    seed_batches = _seed_batches_for_rank(
        seeds, size_dp, DIST_WRAPPER.rank, seed_batch_size
    )
    logger.info(f"[Rank {DIST_WRAPPER.rank}] Assigned seed batches: {seed_batches}")

    try:
        dataloader = get_inference_dataloader(configs=configs)
    except Exception as e:
        error_message = (
            f"Dataloader initialization failed: {e}\n{traceback.format_exc()}"
        )
        logger.error(error_message)
        with open(opjoin(runner.error_dir, "error.txt"), "a", encoding="utf-8") as f:
            f.write(error_message)
        return

    num_data = len(cast(Sized, dataloader.dataset))
    t0_start = time.time()
    with disable_cudnn_benchmark(runner.device):
        for seed_batch in seed_batches:
            cleanup_device_memory(runner.device)
            t1_start = time.time()
            lanes = _seed_data_lanes(
                dataloader, seed_batch, deterministic=configs.deterministic
            )
            msa_generators = _seed_batch_msa_generators(
                lanes, seed_batch, runner.device
            )
            for _ in range(num_data):
                try:
                    _infer_seed_batch_record(
                        runner=runner,
                        configs=configs,
                        lanes=lanes,
                        seed_batch=seed_batch,
                        msa_generators=msa_generators,
                        num_data=num_data,
                    )
                finally:
                    cleanup_device_memory(runner.device, collect_garbage=False)
            cleanup_device_memory(runner.device, synchronize=True)
            t1_end = time.time()
            logger.info(
                f"[Rank {DIST_WRAPPER.rank}] Seed batch {seed_batch} completed in "
                f"{t1_end - t1_start:.2f}s."
            )
    # Remove the error directory if it's empty
    if opexists(runner.error_dir):
        try:
            if not os.listdir(runner.error_dir):
                os.rmdir(runner.error_dir)
        except Exception:
            pass

    t0_end = time.time()
    logger.info(
        f"[Rank {DIST_WRAPPER.rank}] Job completed in {t0_end - t0_start:.2f}s."
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
