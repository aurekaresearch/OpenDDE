# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generator-level Training-Free Guidance regression tests."""

import numpy as np
import torch

from opendde.model.generator import InferenceNoiseScheduler, sample_diffusion
from opendde.tfg.potentials import CLASS_REGISTRY, Potential, register

if "_MockGeoPotential" not in CLASS_REGISTRY:

    @register
    class _MockGeoPotential(Potential):
        """Coordinate-only test potential: energy = scale * ||coords||^2."""

        def _eval(self, coords, feats, params, need_grad):
            scale = float(params.get("scale", 1.0))
            energy = scale * (coords**2).flatten(start_dim=-2).sum(dim=-1)
            if need_grad:
                return energy, 2.0 * scale * coords
            return energy


def _seed(s=0):
    # centre_random_augmentation rotates via scipy/numpy RNG, so seed both
    # torch and numpy to get a reproducible sampler.
    torch.manual_seed(s)
    np.random.seed(s)


def _mock_denoiser(*, x_noisy, t_hat_noise_level, **kwargs):
    return x_noisy * 0.9


def _inputs(n_token=2, c=8, n_atom=6, n_step=4):
    # Inputs are built from a private generator so they never consume the global
    # RNG; this lets two runs share identical model inputs under the same seed.
    g = torch.Generator().manual_seed(123)
    s_inputs = torch.randn(n_token, c, generator=g)
    z_trunk = torch.randn(n_token, n_token, c, generator=g)
    return dict(
        denoise_net=_mock_denoiser,
        input_feature_dict={"atom_to_token_idx": torch.zeros(n_atom, dtype=torch.long)},
        s_inputs=s_inputs,
        s_trunk=s_inputs,
        z_trunk=z_trunk,
        pair_z=None,
        p_lm=None,
        c_l=None,
        noise_schedule=InferenceNoiseScheduler()(N_step=n_step),
    )


def _tfg_cfg():
    return {
        "enable": True,
        "rho": 0.0,
        "mu": 0.1,
        "mc": {"std": 0.0, "batch": 1},
        "steps": {
            "tfg_outer": 1,
            "tfg_inner": 2,
            "projection_outer": 0,
            "projection_inner": 0,
        },
        "terms": {"_MockGeoPotential": {"interval": 1, "weight": 0.01}},
    }


def test_sample_diffusion_shape_and_determinism_without_guidance():
    _seed(0)
    out1 = sample_diffusion(N_sample=3, **_inputs())
    _seed(0)
    out2 = sample_diffusion(N_sample=3, **_inputs())
    assert out1.shape == (3, 6, 3)
    torch.testing.assert_close(out1, out2)


def test_tfg_guidance_runs_and_preserves_n_sample():
    _seed(0)
    out = sample_diffusion(N_sample=2, guidance_configs=_tfg_cfg(), **_inputs())
    assert out.shape == (2, 6, 3)
    assert torch.isfinite(out).all()


def test_tfg_guidance_chunked_preserves_shape():
    _seed(0)
    out = sample_diffusion(
        N_sample=3,
        diffusion_chunk_size=2,
        guidance_configs=_tfg_cfg(),
        **_inputs(),
    )
    assert out.shape == (3, 6, 3)
    assert torch.isfinite(out).all()


def test_sample_diffusion_forwards_pair_z_spec_without_guidance():
    _seed(0)
    sentinel = object()
    seen_specs = []

    def denoiser(*, x_noisy, t_hat_noise_level, **kwargs):
        seen_specs.append(kwargs.get("pair_z_spec"))
        return x_noisy * 0.9

    inputs = _inputs(n_step=1)
    inputs["denoise_net"] = denoiser
    out = sample_diffusion(N_sample=1, pair_z_spec=sentinel, **inputs)

    assert out.shape == (1, 6, 3)
    assert seen_specs
    assert all(spec is sentinel for spec in seen_specs)


def test_tfg_guidance_forwards_pair_z_spec_to_denoiser():
    _seed(0)
    sentinel = object()
    seen_specs = []

    def denoiser(*, x_noisy, t_hat_noise_level, **kwargs):
        seen_specs.append(kwargs.get("pair_z_spec"))
        return x_noisy * 0.9

    cfg = _tfg_cfg()
    cfg["rho"] = 0.0
    cfg["mu"] = 0.0
    inputs = _inputs(n_step=1)
    inputs["denoise_net"] = denoiser
    out = sample_diffusion(
        N_sample=1,
        guidance_configs=cfg,
        pair_z_spec=sentinel,
        **inputs,
    )

    assert out.shape == (1, 6, 3)
    assert seen_specs
    assert all(spec is sentinel for spec in seen_specs)
