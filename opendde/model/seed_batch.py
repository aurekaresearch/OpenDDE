# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Feature collation helpers for seed-batched inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


SEED_BATCH_FEATURES = frozenset(
    {
        "deletion_mean",
        "deletion_value",
        "has_deletion",
        "msa",
        "msa_mask",
        "profile",
        "ref_atom_name_chars",
        "ref_charge",
        "ref_element",
        "ref_mask",
        "ref_pos",
        "ref_space_uid",
        "restype",
    }
)


def _stack_tensors(name: str, values: Sequence[Any]) -> torch.Tensor:
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"Seed-batched feature {name!r} must be a tensor.")
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise ValueError(
            f"Seed-batched feature {name!r} must have one shape; got {sorted(shapes)}."
        )
    return torch.stack(tuple(values), dim=0)


def stack_seed_batch_features(
    feature_dicts: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Stack seed-varying model features and share invariant topology features."""
    if not feature_dicts or len(feature_dicts) != len(seeds):
        raise ValueError(
            "Seed-batched inference requires one feature dictionary per seed; "
            f"got {len(feature_dicts)} feature dictionaries and {len(seeds)} seeds."
        )

    reference = feature_dicts[0]
    reference_keys = set(reference)
    for feature_dict in feature_dicts[1:]:
        if set(feature_dict) != reference_keys:
            raise ValueError(
                "Seed-batched feature dictionaries must have identical keys."
            )

    batched = dict(reference)
    if len(feature_dicts) == 1:
        batched["inference_seed"] = torch.tensor(
            int(seeds[0]),
            dtype=torch.long,
            device=next(
                (
                    value.device
                    for value in reference.values()
                    if isinstance(value, torch.Tensor)
                ),
                None,
            ),
        )
        return batched

    for name in SEED_BATCH_FEATURES:
        if name not in reference:
            continue
        values = [feature_dict[name] for feature_dict in feature_dicts]
        batched[name] = _stack_tensors(name, values)

    template_features = {
        name for name in reference_keys if name.startswith("template_")
    }
    for name in template_features:
        values = [feature_dict[name] for feature_dict in feature_dicts]
        if not all(isinstance(value, torch.Tensor) for value in values):
            continue
        if not all(torch.equal(values[0], value) for value in values[1:]):
            batched[name] = _stack_tensors(name, values)

    for name in reference_keys - SEED_BATCH_FEATURES - template_features:
        reference_value = reference[name]
        if not isinstance(reference_value, torch.Tensor):
            continue
        for value in (feature_dict[name] for feature_dict in feature_dicts[1:]):
            if not isinstance(value, torch.Tensor) or not torch.equal(
                reference_value, value
            ):
                raise ValueError(
                    "Seed batching requires identical topology features, but "
                    f"{name!r} differs between seeds."
                )

    batched["inference_seed"] = torch.tensor(
        [int(seed) for seed in seeds],
        dtype=torch.long,
        device=next(
            (
                value.device
                for value in reference.values()
                if isinstance(value, torch.Tensor)
            ),
            None,
        ),
    )
    return batched


def select_seed_batch_features(
    feature_dict: Mapping[str, Any],
    seed_index: int,
    batch_size: int,
) -> dict[str, Any]:
    """Select one seed lane while retaining shared topology features."""
    selected = dict(feature_dict)
    for name in SEED_BATCH_FEATURES | {"inference_seed", "d_lm", "v_lm"}:
        value = selected.get(name)
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0 or value.shape[0] != batch_size:
            raise ValueError(
                f"Seed-batched feature {name!r} has invalid shape {tuple(value.shape)} "
                f"for batch size {batch_size}."
            )
        selected[name] = value[seed_index]

    pad_info = selected.get("pad_info")
    if isinstance(pad_info, Mapping):
        selected["pad_info"] = {
            key: (
                value[seed_index]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
            )
            for key, value in pad_info.items()
        }
    return selected
