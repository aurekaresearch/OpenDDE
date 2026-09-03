# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
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
