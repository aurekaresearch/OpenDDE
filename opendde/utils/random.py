# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Random helpers that preserve one independent stream per seed batch lane."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, TypeAlias

import torch


TorchGenerator: TypeAlias = Optional[torch.Generator | Sequence[torch.Generator]]


def randn_with_generators(
    size: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: TorchGenerator = None,
    seed_batch_dim: int = 0,
) -> torch.Tensor:
    """Draw Gaussian noise independently for each seed batch lane."""
    shape = tuple(size)
    if generator is None or isinstance(generator, torch.Generator):
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    generators = tuple(generator)
    batch_dim = seed_batch_dim % len(shape)
    if shape[batch_dim] != len(generators):
        raise ValueError(
            "Random seed batch dimension does not match the generator count: "
            f"shape={shape}, seed_batch_dim={seed_batch_dim}, "
            f"generators={len(generators)}."
        )
    lane_shape = shape[:batch_dim] + shape[batch_dim + 1 :]
    return torch.stack(
        [
            torch.randn(
                lane_shape,
                device=device,
                dtype=dtype,
                generator=lane_generator,
            )
            for lane_generator in generators
        ],
        dim=batch_dim,
    )
