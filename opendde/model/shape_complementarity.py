# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from typing import Any, Optional, TypedDict

import torch
from torch.utils.checkpoint import checkpoint

from opendde.data.tokenizer import STRUCTURAL_TOKEN_ROLES
from opendde.model.utils import aggregate_atom_to_token
from opendde.utils.scatter_utils import scatter


def _to_bool_mask(
    mask: Optional[torch.Tensor],
    n_atom: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(n_atom, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


def get_shape_comp_atom_mask(
    feat_dict: dict[str, Any],
    label_dict: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    atom_to_token_idx = feat_dict.get("atom_to_token_idx")
    if atom_to_token_idx is None:
        raise KeyError("shape complementarity requires atom_to_token_idx")
    n_atom = atom_to_token_idx.shape[0]
    device = atom_to_token_idx.device

    if label_dict is None:
        return _to_bool_mask(feat_dict.get("ref_mask"), n_atom=n_atom, device=device)

    atom_mask = _to_bool_mask(
        label_dict.get("coordinate_mask"), n_atom=n_atom, device=device
    )
    if "is_known_chain_condition_case" in feat_dict and bool(
        feat_dict["is_known_chain_condition_case"].reshape(-1)[0].item()
    ):
        return atom_mask

    if "atom_supervision_mask" in label_dict:
        return _to_bool_mask(
            label_dict["atom_supervision_mask"], n_atom=n_atom, device=device
        )

    return atom_mask


def _select_rep_atom_mask(
    feat_dict: dict[str, Any],
    atom_to_token_idx: torch.Tensor,
    n_token: int,
    is_structural: bool,
) -> torch.Tensor:
    candidate_keys = (
        ("structural_distogram_rep_atom_mask", "distogram_rep_atom_mask")
        if is_structural
        else ("distogram_rep_atom_mask", "structural_distogram_rep_atom_mask")
    )
    expected = torch.arange(n_token, device=atom_to_token_idx.device, dtype=torch.long)
    checked = []
    for key in candidate_keys:
        if key not in feat_dict:
            continue
        rep_atom_mask = feat_dict[key].to(device=atom_to_token_idx.device).bool()
        checked.append(key)
        if (
            rep_atom_mask.ndim != 1
            or rep_atom_mask.shape[0] != atom_to_token_idx.shape[0]
        ):
            continue
        rep_atom_idx = torch.nonzero(rep_atom_mask, as_tuple=False).squeeze(dim=-1)
        if rep_atom_idx.numel() != n_token:
            continue
        rep_token_idx = atom_to_token_idx.index_select(dim=0, index=rep_atom_idx)
        sorted_token_idx = torch.sort(rep_token_idx).values
        if torch.equal(sorted_token_idx, expected):
            return rep_atom_mask
    raise ValueError(
        "Could not resolve representative atom mask for shape complementarity "
        f"with n_token={n_token}; checked={checked}"
    )


def _resolve_residue_protein_token_mask(
    feat_dict: dict[str, Any],
    atom_to_token_idx: torch.Tensor,
    n_token: int,
) -> torch.Tensor:
    if (
        "is_protein_token" in feat_dict
        and feat_dict["is_protein_token"].shape[0] == n_token
    ):
        return feat_dict["is_protein_token"].bool()

    def _token_any(atom_feature_name: str) -> torch.Tensor:
        atom_flag = feat_dict[atom_feature_name].to(
            device=atom_to_token_idx.device,
            dtype=torch.float32,
        )
        return (
            scatter(
                src=atom_flag,
                index=atom_to_token_idx,
                dim=-1,
                dim_size=n_token,
                reduce="sum",
            )
            > 0.5
        )

    atom_count = scatter(
        src=torch.ones_like(atom_to_token_idx, dtype=torch.float32),
        index=atom_to_token_idx,
        dim=-1,
        dim_size=n_token,
        reduce="sum",
    )
    token_is_protein = _token_any("is_protein")
    token_is_ligand = _token_any("is_ligand")
    token_is_dna = _token_any("is_dna")
    token_is_rna = _token_any("is_rna")
    return (
        token_is_protein
        & (~token_is_ligand)
        & (~token_is_dna)
        & (~token_is_rna)
        & (atom_count > 1)
    )


_STRUCTURAL_SHAPE_COMP_REQUIRED = (
    "parent_residue_idx",
    "structural_token_index",
    "atom_to_structural_token_idx",
    "structural_distogram_rep_atom_mask",
    "subtoken_role_id",
    "asym_id",
    "token_index",
)
_STRUCTURAL_SHAPE_COMP_ALIASES = (
    ("token_index", "structural_token_index", True),
    ("atom_to_token_idx", "atom_to_structural_token_idx", True),
    ("atom_to_tokatom_idx", "atom_to_structural_tokatom_idx", True),
    ("distogram_rep_atom_mask", "structural_distogram_rep_atom_mask", False),
    ("pae_rep_atom_mask", "structural_pae_rep_atom_mask", False),
)
_PARENT_INDEXED_TOKEN_FEATURES = ("asym_id", "residue_index", "entity_id", "sym_id")


class _ShapeCompTokenFeatures(TypedDict):
    atom_to_token_idx: torch.Tensor
    rep_atom_mask: torch.Tensor
    token_asym_id: torch.Tensor
    token_role_id: torch.Tensor
    is_structural: bool
    is_protein_token: torch.Tensor


def shape_comp_pred_uses_structural_tokens(
    feat_dict: dict[str, Any],
    pred_dict: dict[str, torch.Tensor],
) -> bool:
    marker = pred_dict.get("shape_comp_uses_structural_tokens")
    if marker is not None:
        return bool(marker.reshape(-1)[0].item())

    structural_token_index = feat_dict.get("structural_token_index")
    if structural_token_index is None:
        return False

    pred_n_token = int(pred_dict["shape_comp_token_pred"].shape[-1])
    residue_n_token = int(feat_dict["token_index"].shape[-1])
    structural_n_token = int(structural_token_index.shape[-1])
    return pred_n_token == structural_n_token and pred_n_token != residue_n_token


def _validate_structural_parent_mapping(
    parent: torch.Tensor,
    structural_token_index: torch.Tensor,
    residue_n_token: int,
) -> None:
    if parent.ndim != 1 or structural_token_index.ndim != 1:
        raise ValueError(
            "Structural-token shape-complementarity features must be unbatched "
            f"with shapes parent_residue_idx=[N_struct], "
            f"structural_token_index=[N_struct]; got "
            f"{tuple(parent.shape)} and {tuple(structural_token_index.shape)}"
        )
    if parent.shape[0] != structural_token_index.shape[0]:
        raise ValueError(
            "Structural-token shape-complementarity parent mapping must match "
            f"structural token count: parent={tuple(parent.shape)}, "
            f"structural_token_index={tuple(structural_token_index.shape)}"
        )
    if parent.numel() == 0:
        return

    min_parent = int(parent.min().item())
    max_parent = int(parent.max().item())
    if min_parent < 0 or max_parent >= residue_n_token:
        raise ValueError(
            "Structural-token shape-complementarity parent_residue_idx points "
            f"outside residue-token range: min={min_parent}, "
            f"max={max_parent}, N_residue={residue_n_token}"
        )


def structural_shape_comp_feature_dict(feat_dict: dict[str, Any]) -> dict[str, Any]:
    missing = [
        name for name in _STRUCTURAL_SHAPE_COMP_REQUIRED if name not in feat_dict
    ]
    if missing:
        raise KeyError(
            "Structural-token shape-complementarity prediction requires feature(s): "
            + ", ".join(missing)
        )

    parent = feat_dict["parent_residue_idx"].long()
    structural_token_index = feat_dict["structural_token_index"].long()
    _validate_structural_parent_mapping(
        parent=parent,
        structural_token_index=structural_token_index,
        residue_n_token=int(feat_dict["token_index"].shape[-1]),
    )

    structural_feat_dict = dict(feat_dict)
    for target_key, source_key, cast_long in _STRUCTURAL_SHAPE_COMP_ALIASES:
        if source_key in feat_dict:
            value = feat_dict[source_key]
            structural_feat_dict[target_key] = value.long() if cast_long else value

    for token_feature in _PARENT_INDEXED_TOKEN_FEATURES:
        if token_feature not in feat_dict:
            continue
        parent_on_feature_device = parent.to(device=feat_dict[token_feature].device)
        structural_feat_dict[token_feature] = feat_dict[token_feature].index_select(
            dim=-1,
            index=parent_on_feature_device,
        )

    return structural_feat_dict


def resolve_shape_comp_feature_dict_for_pred(
    feat_dict: dict[str, Any],
    pred_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    if shape_comp_pred_uses_structural_tokens(
        feat_dict=feat_dict,
        pred_dict=pred_dict,
    ):
        return structural_shape_comp_feature_dict(feat_dict)
    return feat_dict


def resolve_shape_comp_token_features(
    feat_dict: dict[str, Any],
    n_token: Optional[int] = None,
) -> _ShapeCompTokenFeatures:
    if n_token is None:
        n_token = int(feat_dict["token_index"].shape[-1])

    atom_to_token_idx = feat_dict["atom_to_token_idx"].long()
    if atom_to_token_idx.numel() == 0:
        raise ValueError("shape complementarity requires at least one atom")
    max_token_idx = int(atom_to_token_idx.max().item())
    if max_token_idx + 1 != n_token:
        raise ValueError(
            "atom_to_token_idx does not match active token space: "
            f"max+1={max_token_idx + 1}, n_token={n_token}"
        )

    token_asym_id = feat_dict["asym_id"].long()
    if token_asym_id.shape[0] != n_token:
        raise ValueError(
            "asym_id does not match active token space: "
            f"{tuple(token_asym_id.shape)} vs ({n_token},)"
        )

    token_role_id = feat_dict.get("subtoken_role_id")
    is_structural = (
        token_role_id is not None
        and token_role_id.ndim == 1
        and token_role_id.shape[0] == n_token
    )
    rep_atom_mask = _select_rep_atom_mask(
        feat_dict=feat_dict,
        atom_to_token_idx=atom_to_token_idx,
        n_token=n_token,
        is_structural=is_structural,
    )
    if is_structural:
        token_role_id = token_role_id.long()
        if (
            "structural_is_protein_token" in feat_dict
            and feat_dict["structural_is_protein_token"].shape[0] == n_token
        ):
            is_protein_token = feat_dict["structural_is_protein_token"].bool()
        else:
            is_protein_token = (
                token_role_id == STRUCTURAL_TOKEN_ROLES["protein_bb"]
            ) | (token_role_id == STRUCTURAL_TOKEN_ROLES["protein_sc"])
    else:
        token_role_id = torch.full(
            (n_token,),
            -1,
            dtype=torch.long,
            device=atom_to_token_idx.device,
        )
        is_protein_token = _resolve_residue_protein_token_mask(
            feat_dict=feat_dict,
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
        )

    return {
        "atom_to_token_idx": atom_to_token_idx,
        "rep_atom_mask": rep_atom_mask,
        "token_asym_id": token_asym_id,
        "token_role_id": token_role_id,
        "is_structural": is_structural,
        "is_protein_token": is_protein_token,
    }


def _sorted_rep_atom_indices(
    atom_to_token_idx: torch.Tensor,
    rep_atom_mask: torch.Tensor,
    n_token: int,
) -> torch.Tensor:
    rep_atom_idx = torch.nonzero(rep_atom_mask, as_tuple=False).squeeze(dim=-1)
    rep_token_idx = atom_to_token_idx.index_select(dim=0, index=rep_atom_idx)
    sorted_token_idx, order = torch.sort(rep_token_idx)
    expected = torch.arange(n_token, device=atom_to_token_idx.device, dtype=torch.long)
    if not torch.equal(sorted_token_idx, expected):
        raise ValueError(
            "Representative atoms do not cover the active token space exactly"
        )
    return rep_atom_idx.index_select(dim=0, index=order)


def _masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    eps: float,
) -> torch.Tensor:
    large_negative = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask, logits, torch.full_like(logits, large_negative))
    max_logits = masked_logits.amax(dim=dim, keepdim=True)
    max_logits = torch.where(
        mask.any(dim=dim, keepdim=True), max_logits, torch.zeros_like(max_logits)
    )
    exp_logits = torch.where(
        mask, torch.exp(masked_logits - max_logits), torch.zeros_like(logits)
    )
    denom = exp_logits.sum(dim=dim, keepdim=True).clamp_min(eps)
    return torch.where(mask, exp_logits / denom, torch.zeros_like(logits))


def summarize_shape_comp_pair(
    pair_score: torch.Tensor,
    pair_mask: torch.Tensor,
    topk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_pair = pair_score.reshape(-1, pair_score.shape[-2] * pair_score.shape[-1])
    flat_mask = pair_mask.reshape(-1, pair_mask.shape[-2] * pair_mask.shape[-1])
    pair_mean = flat_pair.new_zeros(flat_pair.shape[0])
    pair_topk_mean = flat_pair.new_zeros(flat_pair.shape[0])
    valid_pair_frac = flat_mask.to(dtype=flat_pair.dtype).mean(dim=-1)

    for batch_idx in range(flat_pair.shape[0]):
        valid_values = flat_pair[batch_idx][flat_mask[batch_idx]]
        if valid_values.numel() == 0:
            continue
        pair_mean[batch_idx] = valid_values.mean()
        cur_topk = min(topk, valid_values.numel())
        pair_topk_mean[batch_idx] = torch.topk(valid_values, k=cur_topk).values.mean()

    prefix_shape = pair_score.shape[:-2]
    if len(prefix_shape) == 0:
        return pair_mean[0], pair_topk_mean[0], valid_pair_frac[0]
    return (
        pair_mean.reshape(prefix_shape),
        pair_topk_mean.reshape(prefix_shape),
        valid_pair_frac.reshape(prefix_shape),
    )


def _shape_comp_pair_summary(
    shape_comp: dict[str, torch.Tensor],
    pair_summary_topk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        "shape_comp_pair_mean" in shape_comp
        and "shape_comp_pair_topk_mean" in shape_comp
        and "shape_comp_valid_pair_frac" in shape_comp
    ):
        return (
            shape_comp["shape_comp_pair_mean"],
            shape_comp["shape_comp_pair_topk_mean"],
            shape_comp["shape_comp_valid_pair_frac"],
        )
    return summarize_shape_comp_pair(
        pair_score=shape_comp["shape_comp_pair"],
        pair_mask=shape_comp["shape_comp_pair_mask"],
        topk=pair_summary_topk,
    )


def build_shape_comp_pred_outputs(
    shape_comp: dict[str, torch.Tensor],
    keep_pair_map: bool,
    pair_summary_topk: int = 32,
) -> dict[str, torch.Tensor]:
    pair_mean, pair_topk_mean, valid_pair_frac = _shape_comp_pair_summary(
        shape_comp=shape_comp,
        pair_summary_topk=pair_summary_topk,
    )
    outputs = {
        "shape_comp_token_pred": shape_comp["shape_comp_token"],
        "shape_comp_global_pred": shape_comp["shape_comp_global"],
        "shape_comp_token_mask": shape_comp["shape_comp_token_mask"],
        "shape_comp_pair_mean_pred": pair_mean.detach(),
        "shape_comp_pair_topk_mean_pred": pair_topk_mean.detach(),
        "shape_comp_valid_pair_frac_pred": valid_pair_frac.detach(),
    }
    if keep_pair_map:
        if (
            "shape_comp_pair" not in shape_comp
            or "shape_comp_pair_mask" not in shape_comp
        ):
            raise KeyError("keep_pair_map=True requires shape_comp_pair and mask")
        outputs["shape_comp_pair_pred"] = shape_comp["shape_comp_pair"]
        outputs["shape_comp_pair_mask"] = shape_comp["shape_comp_pair_mask"]
    return outputs


def _resolve_pair_chunk_size(pair_chunk_size: Optional[int], n_token: int) -> int:
    if pair_chunk_size is None:
        return n_token
    pair_chunk_size = int(pair_chunk_size)
    if pair_chunk_size <= 0:
        return n_token
    return min(pair_chunk_size, n_token)


def compute_shape_complementarity_fields(
    coordinate: torch.Tensor,
    feat_dict: dict[str, Any],
    atom_mask: Optional[torch.Tensor] = None,
    density_sigma: float = 1.5,
    interface_cutoff: float = 12.0,
    gap_mean: float = 4.0,
    gap_scale: float = 2.0,
    clash_distance: float = 2.0,
    clash_scale: float = 0.5,
    pool_temperature: float = 25.0,
    normal_strength_min: float = 1e-3,
    pair_chunk_size: Optional[int] = 128,
    checkpoint_chunks: bool = True,
    return_pair_map: bool = True,
    eps: float = 1e-6,
    **_: Any,
) -> dict[str, torch.Tensor]:
    if coordinate.ndim < 2 or coordinate.shape[-1] != 3:
        raise ValueError(
            "coordinate must have shape [..., N_atom, 3]; got "
            f"{tuple(coordinate.shape)}"
        )

    n_token = int(feat_dict["token_index"].shape[-1])
    resolved = resolve_shape_comp_token_features(feat_dict=feat_dict, n_token=n_token)
    atom_to_token_idx = resolved["atom_to_token_idx"]
    rep_atom_mask = resolved["rep_atom_mask"]
    token_asym_id = resolved["token_asym_id"]
    token_role_id = resolved["token_role_id"]
    is_structural = bool(resolved["is_structural"])
    is_protein_token = resolved["is_protein_token"].bool()

    coord = coordinate.to(dtype=torch.float32)
    prefix_ndim = coord.ndim - 2
    atom_mask = _to_bool_mask(
        mask=atom_mask,
        n_atom=coord.shape[-2],
        device=coord.device,
    )
    atom_mask_float = atom_mask.to(dtype=coord.dtype)

    rep_atom_indices = _sorted_rep_atom_indices(
        atom_to_token_idx=atom_to_token_idx,
        rep_atom_mask=rep_atom_mask,
        n_token=n_token,
    )
    rep_center = coord.index_select(dim=-2, index=rep_atom_indices)
    rep_valid = atom_mask.index_select(dim=0, index=rep_atom_indices)

    supervised_count = scatter(
        src=atom_mask_float,
        index=atom_to_token_idx,
        dim=-1,
        dim_size=n_token,
        reduce="sum",
    )
    supervised_coord_sum = aggregate_atom_to_token(
        x_atom=coord * atom_mask_float.unsqueeze(dim=-1),
        atom_to_token_idx=atom_to_token_idx,
        n_token=n_token,
        reduce="sum",
    )
    supervised_center = supervised_coord_sum / supervised_count.clamp_min(
        1.0
    ).unsqueeze(dim=-1)

    if is_structural:
        protein_sc_mask = token_role_id == STRUCTURAL_TOKEN_ROLES["protein_sc"]
        token_center = torch.where(
            protein_sc_mask.reshape((1,) * prefix_ndim + (n_token, 1)),
            supervised_center,
            rep_center,
        )
        center_valid = torch.where(protein_sc_mask, supervised_count > 0, rep_valid)
    else:
        token_center = rep_center
        center_valid = rep_valid

    chunk_size = _resolve_pair_chunk_size(
        pair_chunk_size=pair_chunk_size,
        n_token=n_token,
    )

    atom_asym_id = token_asym_id.index_select(dim=0, index=atom_to_token_idx)
    same_chain_atom_mask = (
        token_asym_id[:, None] == atom_asym_id[None, :]
    ) & atom_mask[None, :]

    use_checkpoint = bool(checkpoint_chunks and coord.requires_grad)

    def _density_gradient_chunk(
        center_chunk: torch.Tensor,
        all_coord: torch.Tensor,
        same_chain_mask_chunk: torch.Tensor,
    ) -> torch.Tensor:
        chunk_len = center_chunk.shape[-2]
        token_atom_delta = center_chunk.unsqueeze(dim=-2) - all_coord.unsqueeze(dim=-3)
        token_atom_sq_dist = torch.sum(token_atom_delta * token_atom_delta, dim=-1)
        gaussian_weight = torch.exp(
            -token_atom_sq_dist / (2.0 * density_sigma * density_sigma)
        )
        gaussian_weight = gaussian_weight * same_chain_mask_chunk.reshape(
            (1,) * prefix_ndim + (chunk_len, all_coord.shape[-2])
        ).to(dtype=all_coord.dtype)
        return (gaussian_weight.unsqueeze(dim=-1) * token_atom_delta).sum(dim=-2) / (
            density_sigma * density_sigma
        )

    token_gradient_chunks = []
    for start in range(0, n_token, chunk_size):
        end = min(start + chunk_size, n_token)
        center_chunk = token_center[..., start:end, :]
        same_chain_mask_chunk = same_chain_atom_mask[start:end]
        if use_checkpoint:
            token_gradient_chunks.append(
                checkpoint(
                    _density_gradient_chunk,
                    center_chunk,
                    coord,
                    same_chain_mask_chunk,
                    use_reentrant=False,
                )
            )
        else:
            token_gradient_chunks.append(
                _density_gradient_chunk(
                    center_chunk=center_chunk,
                    all_coord=coord,
                    same_chain_mask_chunk=same_chain_mask_chunk,
                )
            )
    token_gradient = torch.cat(token_gradient_chunks, dim=-2)
    normal_strength = torch.linalg.vector_norm(token_gradient, dim=-1)
    token_normal = token_gradient / normal_strength.clamp_min(eps).unsqueeze(dim=-1)

    static_valid = center_valid & is_protein_token
    token_valid = static_valid.reshape((1,) * prefix_ndim + (n_token,)) & (
        normal_strength > normal_strength_min
    )

    pair_score_chunks = []
    pair_mask_chunks = []
    token_score_chunks = []
    token_mask_chunks = []
    prefix_shape = coord.shape[:-2]
    pair_count = coord.new_zeros(prefix_shape)
    pair_sum = coord.new_zeros(prefix_shape)
    flat_prefix_size = 1
    for dim_size in prefix_shape:
        flat_prefix_size *= int(dim_size)
    topk = 32
    topk_values = coord.new_full((flat_prefix_size, topk), -torch.inf)
    total_pair_count = max(n_token * n_token, 1)
    checkpoint_pair_chunks = bool(use_checkpoint and not return_pair_map)

    def _pair_score_and_mask_chunk(
        all_center: torch.Tensor,
        center_chunk: torch.Tensor,
        all_normal: torch.Tensor,
        normal_chunk: torch.Tensor,
        all_valid: torch.Tensor,
        valid_chunk: torch.Tensor,
        cross_chain_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_len = center_chunk.shape[-2]
        token_pair_delta = all_center.unsqueeze(dim=-3) - center_chunk.unsqueeze(dim=-2)
        token_pair_dist = torch.linalg.vector_norm(token_pair_delta, dim=-1)
        token_pair_unit = token_pair_delta / token_pair_dist.clamp_min(eps).unsqueeze(
            dim=-1
        )

        normal_i = normal_chunk.unsqueeze(dim=-2)
        normal_j = all_normal.unsqueeze(dim=-3)
        facing = torch.relu(torch.sum(normal_i * token_pair_unit, dim=-1)) * torch.relu(
            torch.sum(normal_j * (-token_pair_unit), dim=-1)
        )
        opposite = 0.5 * (1.0 - torch.sum(normal_i * normal_j, dim=-1))
        gap = torch.exp(-(((token_pair_dist - gap_mean) / gap_scale) ** 2))
        anti_clash = 1.0 - torch.sigmoid(
            (clash_distance - token_pair_dist) / clash_scale
        )
        pair_score_chunk = facing * opposite * gap * anti_clash

        pair_mask_chunk = (
            valid_chunk.unsqueeze(dim=-1)
            & all_valid.unsqueeze(dim=-2)
            & cross_chain_chunk.reshape((1,) * prefix_ndim + (chunk_len, n_token))
            & (token_pair_dist <= interface_cutoff)
        )
        pair_score_chunk = torch.where(
            pair_mask_chunk,
            pair_score_chunk,
            torch.zeros_like(pair_score_chunk),
        )
        return pair_score_chunk, pair_mask_chunk, token_pair_dist

    def _token_score_chunk(
        all_center: torch.Tensor,
        center_chunk: torch.Tensor,
        all_normal: torch.Tensor,
        normal_chunk: torch.Tensor,
        all_valid: torch.Tensor,
        valid_chunk: torch.Tensor,
        cross_chain_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pair_score_chunk, pair_mask_chunk, token_pair_dist = _pair_score_and_mask_chunk(
            all_center=all_center,
            center_chunk=center_chunk,
            all_normal=all_normal,
            normal_chunk=normal_chunk,
            all_valid=all_valid,
            valid_chunk=valid_chunk,
            cross_chain_chunk=cross_chain_chunk,
        )
        partner_logits = -(token_pair_dist * token_pair_dist) / pool_temperature
        partner_weight = _masked_softmax(
            logits=partner_logits,
            mask=pair_mask_chunk,
            dim=-1,
            eps=eps,
        )
        token_score_chunk = torch.sum(partner_weight * pair_score_chunk, dim=-1)
        token_mask_chunk = pair_mask_chunk.any(dim=-1)
        return torch.where(
            token_mask_chunk,
            token_score_chunk,
            torch.zeros_like(token_score_chunk),
        )

    for start in range(0, n_token, chunk_size):
        end = min(start + chunk_size, n_token)
        chunk_len = end - start
        center_chunk = token_center[..., start:end, :]
        normal_chunk = token_normal[..., start:end, :]
        valid_chunk = token_valid[..., start:end]
        cross_chain_mask = token_asym_id[start:end, None] != token_asym_id[None, :]

        if checkpoint_pair_chunks:
            token_score_chunk = checkpoint(
                _token_score_chunk,
                token_center,
                center_chunk,
                token_normal,
                normal_chunk,
                token_valid,
                valid_chunk,
                cross_chain_mask,
                use_reentrant=False,
            )
            with torch.no_grad():
                pair_score_chunk, pair_mask_chunk, _dist = _pair_score_and_mask_chunk(
                    all_center=token_center,
                    center_chunk=center_chunk,
                    all_normal=token_normal,
                    normal_chunk=normal_chunk,
                    all_valid=token_valid,
                    valid_chunk=valid_chunk,
                    cross_chain_chunk=cross_chain_mask,
                )
            token_mask_chunk = pair_mask_chunk.any(dim=-1)
        else:
            (
                pair_score_chunk,
                pair_mask_chunk,
                token_pair_dist,
            ) = _pair_score_and_mask_chunk(
                all_center=token_center,
                center_chunk=center_chunk,
                all_normal=token_normal,
                normal_chunk=normal_chunk,
                all_valid=token_valid,
                valid_chunk=valid_chunk,
                cross_chain_chunk=cross_chain_mask,
            )
            partner_logits = -(token_pair_dist * token_pair_dist) / pool_temperature
            partner_weight = _masked_softmax(
                logits=partner_logits,
                mask=pair_mask_chunk,
                dim=-1,
                eps=eps,
            )
            token_score_chunk = torch.sum(partner_weight * pair_score_chunk, dim=-1)
            token_mask_chunk = pair_mask_chunk.any(dim=-1)
            token_score_chunk = torch.where(
                token_mask_chunk,
                token_score_chunk,
                torch.zeros_like(token_score_chunk),
            )
        token_score_chunks.append(token_score_chunk)
        token_mask_chunks.append(token_mask_chunk)

        with torch.no_grad():
            detached_score = pair_score_chunk.detach()
            detached_mask = pair_mask_chunk.detach()
            detached_mask_float = detached_mask.to(dtype=coord.dtype)
            pair_sum = pair_sum + (detached_score * detached_mask_float).sum(
                dim=(-2, -1)
            )
            pair_count = pair_count + detached_mask_float.sum(dim=(-2, -1))
            flat_score = detached_score.reshape(flat_prefix_size, chunk_len * n_token)
            flat_mask = detached_mask.reshape(flat_prefix_size, chunk_len * n_token)
            for batch_idx in range(flat_prefix_size):
                valid_values = flat_score[batch_idx][flat_mask[batch_idx]]
                if valid_values.numel() == 0:
                    continue
                cur_topk = min(topk, valid_values.numel())
                chunk_topk = torch.topk(valid_values, k=cur_topk).values
                combined = torch.cat([topk_values[batch_idx], chunk_topk], dim=0)
                topk_values[batch_idx] = torch.topk(
                    combined,
                    k=topk,
                ).values

        if return_pair_map:
            pair_score_chunks.append(pair_score_chunk)
            pair_mask_chunks.append(pair_mask_chunk)

    token_score = torch.cat(token_score_chunks, dim=-1)
    token_mask = torch.cat(token_mask_chunks, dim=-1)

    global_denom = token_mask.to(dtype=coord.dtype).sum(dim=-1).clamp_min(1.0)
    global_score = torch.sum(token_score, dim=-1) / global_denom
    global_score = torch.where(
        token_mask.any(dim=-1),
        global_score,
        torch.zeros_like(global_score),
    )

    pair_mean = pair_sum / pair_count.clamp_min(1.0)
    pair_mean = torch.where(pair_count > 0, pair_mean, torch.zeros_like(pair_mean))
    valid_pair_frac = pair_count / float(total_pair_count)
    topk_finite = torch.isfinite(topk_values)
    pair_topk_mean = torch.where(
        topk_finite.any(dim=-1),
        torch.where(topk_finite, topk_values, torch.zeros_like(topk_values)).sum(dim=-1)
        / topk_finite.to(dtype=coord.dtype).sum(dim=-1).clamp_min(1.0),
        torch.zeros(flat_prefix_size, dtype=coord.dtype, device=coord.device),
    ).reshape(prefix_shape)

    outputs = {
        "shape_comp_token": token_score,
        "shape_comp_token_mask": token_mask,
        "shape_comp_global": global_score,
        "shape_comp_pair_mean": pair_mean.detach(),
        "shape_comp_pair_topk_mean": pair_topk_mean.detach(),
        "shape_comp_valid_pair_frac": valid_pair_frac.detach(),
        "normal_strength": normal_strength,
    }
    if return_pair_map:
        outputs["shape_comp_pair"] = torch.cat(pair_score_chunks, dim=-2)
        outputs["shape_comp_pair_mask"] = torch.cat(pair_mask_chunks, dim=-2)
    return outputs
