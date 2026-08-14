# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import unittest

import torch

from opendde.config.config import ConfigManager
from opendde.config.model_base import configs as configs_base
from opendde.model.opendde import OpenDDE
from opendde.model.shape_complementarity import (
    build_shape_comp_pred_outputs,
    compute_shape_complementarity_fields,
    get_shape_comp_atom_mask,
)


def _base_feat_dict():
    return {
        "token_index": torch.arange(2, dtype=torch.long),
        "atom_to_token_idx": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "distogram_rep_atom_mask": torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.long),
        "asym_id": torch.tensor([0, 1], dtype=torch.long),
        "is_protein": torch.ones(6, dtype=torch.long),
        "is_ligand": torch.zeros(6, dtype=torch.long),
        "is_dna": torch.zeros(6, dtype=torch.long),
        "is_rna": torch.zeros(6, dtype=torch.long),
        "bond_mask": torch.zeros(6, 6),
        "resolution": torch.tensor([1.5]),
    }


def _good_interface_coords():
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.6, 0.2, 0.0],
            [-0.6, -0.2, 0.0],
            [4.0, 0.0, 0.0],
            [4.6, 0.2, 0.0],
            [4.6, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def _bad_interface_coords():
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.6, 0.2, 0.0],
            [-0.6, -0.2, 0.0],
            [4.0, 0.0, 0.0],
            [3.4, 0.2, 0.0],
            [3.4, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def _same_chain_feat_dict():
    feat = _base_feat_dict()
    feat["token_index"] = torch.arange(3, dtype=torch.long)
    feat["atom_to_token_idx"] = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long
    )
    feat["distogram_rep_atom_mask"] = torch.tensor(
        [1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.long
    )
    feat["asym_id"] = torch.tensor([0, 0, 1], dtype=torch.long)
    feat["is_protein"] = torch.ones(9, dtype=torch.long)
    feat["is_ligand"] = torch.zeros(9, dtype=torch.long)
    feat["is_dna"] = torch.zeros(9, dtype=torch.long)
    feat["is_rna"] = torch.zeros(9, dtype=torch.long)
    feat["bond_mask"] = torch.zeros(9, 9)
    return feat


def _same_chain_coords():
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.5, 0.2, 0.0],
            [-0.5, -0.2, 0.0],
            [2.0, 0.0, 0.0],
            [1.4, 0.2, 0.0],
            [1.4, -0.2, 0.0],
            [5.0, 0.0, 0.0],
            [5.6, 0.2, 0.0],
            [5.6, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def _weak_normal_coords():
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.6, 0.2, 0.0],
            [4.6, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def _structural_feat_dict():
    return {
        "token_index": torch.arange(4, dtype=torch.long),
        "atom_to_token_idx": torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.long),
        "distogram_rep_atom_mask": torch.tensor([1, 1, 0, 1, 1, 0], dtype=torch.long),
        "structural_distogram_rep_atom_mask": torch.tensor(
            [1, 0, 1, 1, 0, 1], dtype=torch.long
        ),
        "asym_id": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "subtoken_role_id": torch.tensor([1, 2, 1, 2], dtype=torch.long),
        "is_protein": torch.ones(6, dtype=torch.long),
        "is_ligand": torch.zeros(6, dtype=torch.long),
        "is_dna": torch.zeros(6, dtype=torch.long),
        "is_rna": torch.zeros(6, dtype=torch.long),
        "bond_mask": torch.zeros(6, 6),
        "resolution": torch.tensor([1.5]),
    }


def _structural_coords():
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [-0.8, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.4, 0.0, 0.0],
            [4.8, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )


def _rotate_translate(coords: torch.Tensor) -> torch.Tensor:
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )
    return coords @ rotation.T + torch.tensor([3.0, -2.0, 1.5], dtype=torch.float32)


def _build_shape_comp_config(
    alpha_shape_comp: float, pair_weight: float, debug_pair_map: bool = False
):
    manager = ConfigManager(copy.deepcopy(configs_base), fill_required_with_null=True)
    return manager.merge_configs(
        {
            "confidence.weight.alpha_shape_comp": alpha_shape_comp,
            "confidence.shape_comp.pair_weight": pair_weight,
            "confidence.shape_comp.debug_pair_map": str(debug_pair_map).lower(),
        }
    )


class TestShapeComplementarity(unittest.TestCase):
    def test_known_chain_condition_uses_full_shape_comp_mask(self):
        feat_dict = _base_feat_dict()
        feat_dict.update(
            {
                "is_known_chain_condition_case": torch.tensor([1], dtype=torch.long),
                "known_atom_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.long),
                "target_atom_mask": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            }
        )
        label_dict = {
            "coordinate_mask": torch.ones(6, dtype=torch.float32),
            "atom_supervision_mask": torch.tensor(
                [0, 0, 0, 1, 1, 1], dtype=torch.float32
            ),
        }

        atom_mask = get_shape_comp_atom_mask(
            feat_dict=feat_dict,
            label_dict=label_dict,
        )

        self.assertTrue(torch.equal(atom_mask, torch.ones(6, dtype=torch.bool)))

    def test_forward_guard_skips_shape_comp_when_disabled(self):
        model = OpenDDE.__new__(OpenDDE)
        model.configs = _build_shape_comp_config(alpha_shape_comp=0.0, pair_weight=0.0)

        pred_dict = {}
        model.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=_base_feat_dict(),
            coordinate=_good_interface_coords().unsqueeze(dim=0),
            label_dict={
                "coordinate_mask": torch.ones(6, dtype=torch.float32),
            },
        )
        self.assertFalse(any(key.startswith("shape_comp_") for key in pred_dict))

    def test_forward_guard_skips_shape_comp_when_alpha_disabled(self):
        model = OpenDDE.__new__(OpenDDE)
        model.configs = _build_shape_comp_config(alpha_shape_comp=0.0, pair_weight=1.0)

        pred_dict = {}
        model.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=_base_feat_dict(),
            coordinate=_good_interface_coords().unsqueeze(dim=0),
            label_dict={
                "coordinate_mask": torch.ones(6, dtype=torch.float32),
            },
        )
        self.assertFalse(any(key.startswith("shape_comp_") for key in pred_dict))

    def test_forward_guard_enables_shape_comp_when_weighted(self):
        model = OpenDDE.__new__(OpenDDE)
        model.configs = _build_shape_comp_config(alpha_shape_comp=1.0, pair_weight=0.0)

        pred_dict = {}
        model.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=_base_feat_dict(),
            coordinate=_good_interface_coords().unsqueeze(dim=0),
            label_dict={
                "coordinate_mask": torch.ones(6, dtype=torch.float32),
            },
        )
        self.assertIn("shape_comp_token_pred", pred_dict)
        self.assertIn("shape_comp_global_pred", pred_dict)
        self.assertNotIn("shape_comp_pair_pred", pred_dict)

    def test_forward_guard_pair_score_does_not_store_pair_map(self):
        model = OpenDDE.__new__(OpenDDE)
        model.configs = _build_shape_comp_config(alpha_shape_comp=1.0, pair_weight=1.0)

        pred_dict = {}
        model.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=_base_feat_dict(),
            coordinate=_good_interface_coords().unsqueeze(dim=0),
            label_dict={
                "coordinate_mask": torch.ones(6, dtype=torch.float32),
            },
        )
        self.assertIn("shape_comp_token_pred", pred_dict)
        self.assertIn("shape_comp_global_pred", pred_dict)
        self.assertNotIn("shape_comp_pair_pred", pred_dict)
        self.assertNotIn("shape_comp_pair_mask", pred_dict)

    def test_forward_guard_debug_pair_map_overrides_disabled_shape_comp(self):
        model = OpenDDE.__new__(OpenDDE)
        model.configs = _build_shape_comp_config(
            alpha_shape_comp=0.0,
            pair_weight=0.0,
            debug_pair_map=True,
        )

        pred_dict = {}
        model.add_shape_complementarity_predictions(
            pred_dict=pred_dict,
            input_feature_dict=_base_feat_dict(),
            coordinate=_good_interface_coords().unsqueeze(dim=0),
            label_dict={
                "coordinate_mask": torch.ones(6, dtype=torch.float32),
            },
        )
        self.assertIn("shape_comp_token_pred", pred_dict)
        self.assertIn("shape_comp_global_pred", pred_dict)
        self.assertIn("shape_comp_pair_pred", pred_dict)
        self.assertIn("shape_comp_pair_mask", pred_dict)

    def test_pred_eq_gt_and_rigid_invariant(self):
        feat = _base_feat_dict()
        coords = _good_interface_coords()
        shape_comp = compute_shape_complementarity_fields(
            coordinate=coords,
            feat_dict=feat,
            atom_mask=torch.ones(coords.shape[0], dtype=torch.bool),
        )
        rigid_shape_comp = compute_shape_complementarity_fields(
            coordinate=_rotate_translate(coords),
            feat_dict=feat,
            atom_mask=torch.ones(coords.shape[0], dtype=torch.bool),
        )

        self.assertTrue(
            torch.allclose(
                shape_comp["shape_comp_pair"],
                rigid_shape_comp["shape_comp_pair"],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                shape_comp["shape_comp_token"],
                rigid_shape_comp["shape_comp_token"],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                shape_comp["shape_comp_global"],
                rigid_shape_comp["shape_comp_global"],
                atol=1e-5,
            )
        )

    def test_chunked_token_only_matches_full_pair_mode(self):
        feat = _base_feat_dict()
        coords = _good_interface_coords().unsqueeze(dim=0)
        full = compute_shape_complementarity_fields(
            coordinate=coords,
            feat_dict=feat,
            atom_mask=torch.ones(coords.shape[-2], dtype=torch.bool),
            pair_chunk_size=1,
            return_pair_map=True,
        )
        token_only = compute_shape_complementarity_fields(
            coordinate=coords,
            feat_dict=feat,
            atom_mask=torch.ones(coords.shape[-2], dtype=torch.bool),
            pair_chunk_size=1,
            return_pair_map=False,
        )
        self.assertNotIn("shape_comp_pair", token_only)
        self.assertNotIn("shape_comp_pair_mask", token_only)
        self.assertTrue(
            torch.allclose(
                full["shape_comp_token"],
                token_only["shape_comp_token"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                full["shape_comp_global"],
                token_only["shape_comp_global"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                full["shape_comp_pair_mean"],
                token_only["shape_comp_pair_mean"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                full["shape_comp_pair_topk_mean"],
                token_only["shape_comp_pair_topk_mean"],
                atol=1e-6,
            )
        )

    def test_same_chain_zero_and_good_interface_beats_bad_interface(self):
        same_chain = compute_shape_complementarity_fields(
            coordinate=_same_chain_coords(),
            feat_dict=_same_chain_feat_dict(),
            atom_mask=torch.ones(9, dtype=torch.bool),
        )
        self.assertFalse(same_chain["shape_comp_pair_mask"][0, 1].item())
        self.assertEqual(same_chain["shape_comp_pair"][0, 1].item(), 0.0)

        good = compute_shape_complementarity_fields(
            coordinate=_good_interface_coords(),
            feat_dict=_base_feat_dict(),
            atom_mask=torch.ones(6, dtype=torch.bool),
        )
        bad = compute_shape_complementarity_fields(
            coordinate=_bad_interface_coords(),
            feat_dict=_base_feat_dict(),
            atom_mask=torch.ones(6, dtype=torch.bool),
        )
        self.assertGreater(
            good["shape_comp_pair"][0, 1].item(),
            bad["shape_comp_pair"][0, 1].item(),
        )

    def test_normal_strength_and_protein_only_mask(self):
        weak = compute_shape_complementarity_fields(
            coordinate=_weak_normal_coords(),
            feat_dict=_base_feat_dict(),
            atom_mask=torch.ones(6, dtype=torch.bool),
            normal_strength_min=1e-3,
        )
        self.assertLess(weak["normal_strength"][0].item(), 1e-6)
        self.assertFalse(weak["shape_comp_token_mask"][0].item())

        feat = _base_feat_dict()
        feat["token_index"] = torch.arange(3, dtype=torch.long)
        feat["atom_to_token_idx"] = torch.tensor(
            [0, 0, 0, 1, 2, 2, 2], dtype=torch.long
        )
        feat["distogram_rep_atom_mask"] = torch.tensor(
            [1, 0, 0, 1, 1, 0, 0], dtype=torch.long
        )
        feat["asym_id"] = torch.tensor([0, 1, 2], dtype=torch.long)
        feat["is_protein"] = torch.tensor([1, 1, 1, 0, 1, 1, 1], dtype=torch.long)
        feat["is_ligand"] = torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype=torch.long)
        feat["is_dna"] = torch.zeros(7, dtype=torch.long)
        feat["is_rna"] = torch.zeros(7, dtype=torch.long)
        feat["bond_mask"] = torch.zeros(7, 7)
        coords = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [-0.6, 0.2, 0.0],
                [-0.6, -0.2, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.6, 0.2, 0.0],
                [4.6, -0.2, 0.0],
            ],
            dtype=torch.float32,
        )
        shape_comp = compute_shape_complementarity_fields(
            coordinate=coords,
            feat_dict=feat,
            atom_mask=torch.ones(7, dtype=torch.bool),
        )
        self.assertFalse(shape_comp["shape_comp_token_mask"][1].item())

    def test_structural_space_shapes_and_default_outputs_skip_pair_map(self):
        shape_comp = compute_shape_complementarity_fields(
            coordinate=_structural_coords(),
            feat_dict=_structural_feat_dict(),
            atom_mask=torch.ones(6, dtype=torch.bool),
        )
        self.assertEqual(shape_comp["shape_comp_token"].shape, (4,))
        self.assertEqual(shape_comp["shape_comp_pair"].shape, (4, 4))
        outputs = build_shape_comp_pred_outputs(
            shape_comp=shape_comp, keep_pair_map=False
        )
        self.assertNotIn("shape_comp_pair_pred", outputs)
        self.assertNotIn("shape_comp_pair_mask", outputs)
        self.assertIn("shape_comp_token_pred", outputs)
        self.assertIn("shape_comp_global_pred", outputs)
        self.assertIn("shape_comp_token_mask", outputs)


if __name__ == "__main__":
    unittest.main()
