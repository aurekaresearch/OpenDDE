# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
import logging
import os
import random
import time
import traceback
from collections.abc import Mapping, Sized
from contextlib import nullcontext
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
        if self.foldcp_config.enabled:
            expected_world_size = (
                self.foldcp_config.size_dp * self.foldcp_config.size_cp
            )
            if DIST_WRAPPER.world_size != expected_world_size:
                raise RuntimeError(
                    "Distributed Fold-CP must be launched with torchrun using "
                    f"{expected_world_size} processes. Example: "
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

        self.configs = apply_runtime_compatibility(self.configs, self.device)

        if DIST_WRAPPER.world_size > 1:
            if not self.use_cuda:
                raise RuntimeError(
                    "Distributed Fold-CP inference requires NVIDIA CUDA; CPU "
                    "supports single-process inference only."
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

        use_fastlayernorm = os.getenv("LAYERNORM_TYPE", "torch")
        if use_fastlayernorm == "fast_layernorm":
            logging.info(
                "Kernels will be compiled when fast_layernorm is first called."
            )

        logging.info("Selected inference device: %s", self.device)
        logging.info("Finished environment initialization.")

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
            )

        return prediction

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
        for seed in seeds:
            seed_everything(seed=seed, deterministic=configs.deterministic)
            cleanup_device_memory(runner.device)
            t1_start = time.time()
            for batch in dataloader:
                sample_name = "unknown"
                data = None
                atom_array = None
                prediction = None
                try:
                    t2_start = time.time()
                    data, atom_array, data_error_message = batch[0]
                    sample_name = data["sample_name"]

                    if len(data_error_message) > 0:
                        logger.error(
                            f"Data error for {sample_name}: {data_error_message}"
                        )
                        with open(
                            opjoin(runner.error_dir, f"{sample_name}.txt"),
                            "a",
                            encoding="utf-8",
                        ) as f:
                            f.write(data_error_message)
                        continue

                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank} ({data['sample_index'] + 1}/{num_data})] "
                        f"{sample_name} [seed:{seed}]: "
                        f"N_asym {data['N_asym'].item()}, N_token {data['N_token'].item()}, "
                        f"N_atom {data['N_atom'].item()}, N_msa {data['N_msa'].item()}"
                    )
                    new_configs = update_inference_configs(
                        configs, data["N_token"].item()
                    )
                    data["input_feature_dict"]["inference_seed"] = torch.tensor(
                        int(seed),
                        dtype=torch.long,
                    )
                    runner.update_model_configs(new_configs)
                    prediction = runner.predict(data)
                    if not (runner.foldcp_config.enabled and DIST_WRAPPER.rank != 0):
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
                    t2_end = time.time()
                    logger.info(
                        f"[Rank {DIST_WRAPPER.rank}] {sample_name} [seed:{seed}] succeeded. "
                        f"Model forward time: {t2_end - t2_start:.2f}s. "
                        f"Results saved to {configs.dump_dir}"
                    )
                except Exception as e:
                    error_message = (
                        f"[Rank {DIST_WRAPPER.rank}] {sample_name} failed: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    logger.error(error_message)
                    with open(
                        opjoin(runner.error_dir, f"{sample_name}.txt"),
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(error_message)
                finally:
                    del data, atom_array, prediction
                    cleanup_device_memory(runner.device, collect_garbage=False)
            cleanup_device_memory(runner.device)
            t1_end = time.time()
            logger.info(
                f"[Rank {DIST_WRAPPER.rank}] Seed {seed} completed in {t1_end - t1_start:.2f}s."
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
