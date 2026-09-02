# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist

from opendde.distributed.foldcp.comm import run_group_rank_action_synchronized
from opendde.model.utils import centre_random_augmentation
from opendde.tfg import TFGEngine, parse_tfg_config
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps).

    Args:
        s_max (float, optional): maximal noise level. Defaults to 160.0.
        s_min (float, optional): minimal noise level. Defaults to 4e-4.
        rho (float, optional): the exponent numerical part. Defaults to 7.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    rollout_seed: Optional[int] = None,
    guidance_configs: Optional[dict[str, Any]] = None,
    pair_z_spec: Any = None,
    atom_window_spec: Any = None,
    foldcp_attention_bias: Any = None,
    foldcp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.
        guidance_configs (Optional[dict[str, Any]]): Training-free guidance configs. Defaults to None.
        pair_z_spec (Any): Optional Fold-CP pair shard metadata forwarded to denoise_net.
        foldcp_group (Optional[dist.ProcessGroup]): Process group for the maintained
            1xP Fold-CP topology. When provided, rank-local sampler work is
            completed on every rank before the next denoiser collective starts.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3].
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    torch_generator = None
    numpy_rng = None
    if rollout_seed is not None:
        torch_generator = torch.Generator(device=device)
        torch_generator.manual_seed(int(rollout_seed))
        numpy_rng = np.random.default_rng(int(rollout_seed))
    tfg_cfg = parse_tfg_config(guidance_configs)
    if tfg_cfg.enable:
        logger.info("Training-free guidance is enabled.")
        tfg = TFGEngine(tfg_cfg, device=device, dtype=dtype)

    num_diffusion_steps = len(noise_schedule) - 1

    def _run_sampler_action(action: Callable[[], Any], description: str) -> Any:
        if foldcp_group is None:
            return action()
        result = run_group_rank_action_synchronized(
            action,
            group=foldcp_group,
            description=description,
        )
        if result is None:  # pragma: no cover - actions below always return tensors
            raise RuntimeError(f"{description} returned no result.")
        return result

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # init noise
        # [..., chunk_n_sample, N_atom, 3]
        x_l = _run_sampler_action(
            lambda: (
                noise_schedule[0]
                * torch.randn(
                    size=(*batch_shape, chunk_n_sample, N_atom, 3),
                    device=device,
                    dtype=dtype,
                    generator=torch_generator,
                )
            ),
            "Fold-CP diffusion initial-noise preparation",
        )

        for step_i, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):

            def _prepare_diffusion_step():
                # [..., chunk_n_sample, N_atom, 3]
                augmented_x_l = (
                    centre_random_augmentation(
                        x_input_coords=x_l,
                        N_sample=1,
                        torch_generator=torch_generator,
                        numpy_rng=numpy_rng,
                    )
                    .squeeze(dim=-3)
                    .to(dtype)
                )
                # Denoise with a predictor-corrector sampler
                # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
                gamma = float(gamma0) if c_tau > gamma_min else 0
                prepared_t_hat = c_tau_last * (gamma + 1)

                delta_noise_level = torch.sqrt(prepared_t_hat**2 - c_tau_last**2)
                prepared_x_noisy = (
                    augmented_x_l
                    + noise_scale_lambda
                    * delta_noise_level
                    * torch.randn(
                        size=augmented_x_l.shape,
                        device=device,
                        dtype=dtype,
                        generator=torch_generator,
                    )
                )
                # 2. Denoise from x_{t_hat} to x_{c_tau}
                # Euler step only
                prepared_t_hat = (
                    prepared_t_hat.reshape((1,) * (len(batch_shape) + 1))
                    .expand(*batch_shape, chunk_n_sample)
                    .to(dtype)
                )
                return prepared_x_noisy, prepared_t_hat

            prepared_step = _run_sampler_action(
                _prepare_diffusion_step,
                f"Fold-CP diffusion sampler step {step_i} preparation",
            )
            x_noisy, t_hat = prepared_step

            if tfg_cfg.enable:
                x_l = tfg.step(
                    denoise_net,
                    x=x_noisy,
                    t_hat=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    pair_z_spec=pair_z_spec,
                    atom_window_spec=atom_window_spec,
                    foldcp_attention_bias=foldcp_attention_bias,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                    c_tau=c_tau,
                    step_i=step_i,
                    num_diffusion_steps=num_diffusion_steps,
                    step_scale_eta=step_scale_eta,
                    torch_generator=torch_generator,
                    rank_action_synchronizer=(
                        _run_sampler_action if foldcp_group is not None else None
                    ),
                )
            else:
                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    pair_z_spec=pair_z_spec,
                    p_lm=p_lm,
                    c_l=c_l,
                    atom_window_spec=atom_window_spec,
                    foldcp_attention_bias=foldcp_attention_bias,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                )

                def _complete_diffusion_step():
                    delta = (x_noisy - x_denoised) / t_hat[
                        ..., None, None
                    ]  # Line 9 of AF3 uses x_l_hat instead, which we believe is a typo.
                    dt = c_tau - t_hat
                    return x_noisy + step_scale_eta * dt[..., None, None] * delta

                x_l = _run_sampler_action(
                    _complete_diffusion_step,
                    f"Fold-CP diffusion sampler step {step_i} completion",
                )

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        chunk_size = max(1, diffusion_chunk_size)
        x_l = []
        no_chunks = N_sample // chunk_size + (N_sample % chunk_size != 0)
        for i in range(no_chunks):
            chunk_n_sample = (
                chunk_size if i < no_chunks - 1 else N_sample - i * chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = _run_sampler_action(
            lambda: torch.cat(x_l, -3),
            "Fold-CP diffusion sample-chunk assembly",
        )  # [..., N_sample, N_atom, 3]

    return x_l
