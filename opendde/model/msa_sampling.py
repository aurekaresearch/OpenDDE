# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import math
from collections.abc import Sequence
from typing import Optional

import torch


def _collapse_msa_row_mask(row_mask: torch.Tensor) -> torch.Tensor:
    """Collapse optional batch dims and keep one boolean per MSA row."""
    if row_mask.ndim == 1:
        return row_mask
    return row_mask.reshape(-1, row_mask.shape[-1]).any(dim=0)


def subsample_msa_feature_dict_valid_first(
    feat_dict: dict[str, torch.Tensor],
    dim_dict: dict[str, int],
    num_msa: int = 1024,
    msa_mask: Optional[torch.Tensor] = None,
    gap_token: Optional[int] = None,
    generators: Optional[Sequence[torch.Generator]] = None,
) -> dict[str, torch.Tensor]:
    """Subsample MSA rows with AF3/OpenFold3-style valid-first priority.

    Rows with at least one valid token are shuffled ahead of fully padded/all-gap
    rows, then truncated to ``num_msa``. Each call re-samples the order, which
    lets recycle iterations see different MSA subsets.
    """

    msa = feat_dict["msa"]
    msa_dim = dim_dict["msa"]
    msa_len = msa.size(dim=msa_dim)
    device = msa.device
    num_msa = max(0, min(num_msa, msa_len))

    if generators is not None:
        batch_shape = msa.shape[: msa_dim % msa.ndim]
        batch_size = math.prod(batch_shape)
        if batch_size != len(generators):
            raise ValueError(
                "MSA seed batch shape does not match the generator count: "
                f"batch_shape={tuple(batch_shape)}, generators={len(generators)}."
            )
        if msa_dim % msa.ndim != msa.ndim - 2:
            raise ValueError(
                "Seed-batched MSA sampling requires the MSA row dim at -2."
            )

        row_valid = None
        if msa_mask is not None:
            row_valid = msa_mask.bool().any(dim=-1).reshape(batch_size, msa_len)
        if gap_token is not None:
            token_row_valid = (
                (msa != gap_token).any(dim=-1).reshape(batch_size, msa_len)
            )
            if row_valid is None:
                row_valid = token_row_valid
            else:
                mask_has_no_signal = row_valid.all(dim=-1, keepdim=True)
                row_valid = torch.where(mask_has_no_signal, token_row_valid, row_valid)
        if row_valid is None:
            row_valid = torch.ones(batch_size, msa_len, dtype=torch.bool, device=device)

        batch_indices = []
        for lane, generator in enumerate(generators):
            valid_idx = row_valid[lane].nonzero(as_tuple=False).squeeze(-1)
            invalid_idx = (~row_valid[lane]).nonzero(as_tuple=False).squeeze(-1)
            selected = []
            take_valid = min(valid_idx.numel(), num_msa)
            if take_valid > 0:
                valid_perm = valid_idx[
                    torch.randperm(
                        valid_idx.numel(), device=device, generator=generator
                    )
                ]
                selected.append(valid_perm[:take_valid])
            take_invalid = num_msa - take_valid
            if take_invalid > 0 and invalid_idx.numel() > 0:
                invalid_perm = invalid_idx[
                    torch.randperm(
                        invalid_idx.numel(), device=device, generator=generator
                    )
                ]
                selected.append(invalid_perm[:take_invalid])
            batch_indices.append(
                torch.cat(selected, dim=0)
                if selected
                else torch.empty(0, dtype=torch.long, device=device)
            )
        indices = torch.stack(batch_indices, dim=0)

        def gather_seed_batch(value: torch.Tensor, dim: int) -> torch.Tensor:
            normalized_dim = dim % value.ndim
            flat_value = value.reshape(batch_size, *value.shape[len(batch_shape) :])
            flat_dim = normalized_dim - len(batch_shape) + 1
            index_shape = [batch_size] + [1] * (flat_value.ndim - 1)
            index_shape[flat_dim] = num_msa
            gather_index = indices.reshape(index_shape)
            expand_shape = list(flat_value.shape)
            expand_shape[flat_dim] = num_msa
            gathered = torch.gather(
                flat_value,
                dim=flat_dim,
                index=gather_index.expand(expand_shape),
            )
            return gathered.reshape(*batch_shape, *gathered.shape[1:])

        return {
            feat_name: gather_seed_batch(feat_dict[feat_name], dim)
            for feat_name, dim in dim_dict.items()
        }

    if num_msa == 0:
        indices = torch.empty(0, dtype=torch.long, device=device)
    else:
        row_valid = None
        if msa_mask is not None:
            row_valid = _collapse_msa_row_mask(msa_mask.bool().any(dim=-1))

        # OpenDDE currently stores msa_mask as all-ones, so fall back to the
        # MSA tokens when the mask carries no row-validity signal.
        if gap_token is not None and (row_valid is None or torch.all(row_valid)):
            row_valid = _collapse_msa_row_mask((msa != gap_token).any(dim=-1))

        if row_valid is None:
            row_valid = torch.ones(msa_len, dtype=torch.bool, device=device)

        valid_idx = row_valid.nonzero(as_tuple=False).squeeze(-1)
        invalid_idx = (~row_valid).nonzero(as_tuple=False).squeeze(-1)

        selected = []
        take_valid = min(valid_idx.numel(), num_msa)
        if take_valid > 0:
            valid_perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
            selected.append(valid_perm[:take_valid])

        take_invalid = num_msa - take_valid
        if take_invalid > 0 and invalid_idx.numel() > 0:
            invalid_perm = invalid_idx[
                torch.randperm(invalid_idx.numel(), device=device)
            ]
            selected.append(invalid_perm[:take_invalid])

        indices = (
            torch.cat(selected, dim=0)
            if selected
            else torch.empty(0, dtype=torch.long, device=device)
        )

    return {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
