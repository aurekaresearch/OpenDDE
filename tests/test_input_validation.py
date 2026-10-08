# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
import pytest
import torch
from biotite.structure import Atom, AtomArray

from opendde.data.inference.json_parser import build_polymer
from opendde.data.inference import json_to_feature
from opendde.data.msa.msa_utils import map_to_standard


def test_build_polymer_rejects_out_of_range_ptm_position():
    with pytest.raises(ValueError, match="ptmPosition 5 is out of range"):
        build_polymer(
            {
                "proteinChain": {
                    "sequence": "AC",
                    "modifications": [{"ptmPosition": 5, "ptmType": "CCD_MSE"}],
                }
            }
        )


def test_map_to_standard_rejects_unknown_chain():
    meta = {0: {"sequence": "AAA"}, 1: {"sequence": "BB"}}
    with pytest.raises(ValueError, match="could not map residues"):
        map_to_standard(np.array([9]), np.array([1]), meta)


def test_unknown_constraint_fields_are_rejected(monkeypatch):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )

    with pytest.raises(ValueError, match="Unsupported constraint field.*pocket"):
        json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": {"pocket": []}}
        )


def test_epitope_constraint_requires_tfg_guidance(monkeypatch):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )
    sample = {
        "sequences": [],
        "constraint": {"movable_chains": ["H"], "epitope": {"residues": []}},
    }

    with pytest.raises(ValueError, match="use_tfg_guidance"):
        json_to_feature.SampleDictToFeatures(sample, extract_features_for_tfg=False)


def test_empty_contact_constraint_does_not_wholesale_ignore(monkeypatch, caplog):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )

    with caplog.at_level("WARNING"):
        sample = json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": {"contact": []}}
        )

    assert sample.contact_restraints == []
    # Empty contact list alone should not claim the whole constraint field is ignored.
    assert "Only covalent_bonds are supported" not in caplog.text


def test_contact_without_tfg_logs_enable_hint(monkeypatch, caplog):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )
    contact = [
        {
            "entity1": "1",
            "position1": "1",
            "atom1": "CA",
            "entity2": "2",
            "position2": "2",
            "atom2": "CA",
            "min_distance": 3.5,
            "max_distance": 8.0,
        }
    ]
    with caplog.at_level("WARNING"):
        sample = json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": {"contact": contact}},
            extract_features_for_tfg=False,
        )

    assert len(sample.contact_restraints) == 1
    assert "--use_tfg_guidance" in caplog.text


def test_parse_contact_restraints_entity_and_left_right_aliases(monkeypatch):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )
    sample = json_to_feature.SampleDictToFeatures(
        {
            "sequences": [],
            "constraint": {
                "contact": [
                    {
                        "entity1": "1",
                        "copy1": 1,
                        "position1": "20",
                        "atom1": "CA",
                        "entity2": "2",
                        "copy2": 1,
                        "position2": "102",
                        "atom2": "CA",
                        "min_distance": 3.5,
                        "max_distance": 8.0,
                    },
                    {
                        "left_entity": "1",
                        "left_copy": 1,
                        "left_position": "25",
                        "left_atom": "CB",
                        "right_entity": "3",
                        "right_copy": 1,
                        "right_position": "55",
                        "right_atom": "CB",
                        "min_distance": 4.0,
                        "max_distance": 10.0,
                    },
                ]
            },
        },
        extract_features_for_tfg=True,
    )
    assert len(sample.contact_restraints) == 2
    first = sample.contact_restraints[0]
    assert first["left"]["entity_id"] == 1
    assert first["left"]["position"] == 20
    assert first["left"]["atom_name"] == "CA"
    assert first["right"]["entity_id"] == 2
    assert first["min_distance"] == 3.5
    assert first["max_distance"] == 8.0
    second = sample.contact_restraints[1]
    assert second["left"]["atom_name"] == "CB"
    assert second["right"]["entity_id"] == 3
    assert second["max_distance"] == 10.0


def _synthetic_two_chain_atom_array() -> AtomArray:
    """Minimal AtomArray with two protein chains (entity 1 and 2), one CA each."""
    atoms = [
        Atom(
            [0.0, 0.0, 0.0],
            atom_name="N",
            res_name="ALA",
            res_id=1,
            element="N",
        ),
        Atom(
            [1.0, 0.0, 0.0],
            atom_name="CA",
            res_name="ALA",
            res_id=1,
            element="C",
        ),
        Atom(
            [2.0, 0.0, 0.0],
            atom_name="C",
            res_name="ALA",
            res_id=1,
            element="C",
        ),
        Atom(
            [10.0, 0.0, 0.0],
            atom_name="N",
            res_name="GLY",
            res_id=5,
            element="N",
        ),
        Atom(
            [11.0, 0.0, 0.0],
            atom_name="CA",
            res_name="GLY",
            res_id=5,
            element="C",
        ),
        Atom(
            [12.0, 0.0, 0.0],
            atom_name="C",
            res_name="GLY",
            res_id=5,
            element="C",
        ),
    ]
    arr = AtomArray(len(atoms))
    for i, atom in enumerate(atoms):
        arr[i] = atom
    arr.set_annotation(
        "label_entity_id",
        np.array(["1", "1", "1", "2", "2", "2"], dtype=object),
    )
    arr.set_annotation("copy_id", np.array([1, 1, 1, 1, 1, 1], dtype=int))
    return arr


def test_build_user_distance_restraint_features_parse_to_index(monkeypatch):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )
    sample = json_to_feature.SampleDictToFeatures(
        {
            "sequences": [],
            "constraint": {
                "contact": [
                    {
                        "entity1": "1",
                        "copy1": 1,
                        "position1": "1",
                        # atom1 omitted -> defaults to CA for polypeptide
                        "entity2": "2",
                        "copy2": 1,
                        "position2": "5",
                        "atom2": "CA",
                        "min_distance": 3.5,
                        "max_distance": 8.0,
                    }
                ]
            },
        },
        extract_features_for_tfg=True,
    )
    sample.entity_poly_type = {"1": "polypeptide(L)", "2": "polypeptide(L)"}
    atom_array = _synthetic_two_chain_atom_array()
    feats = sample.build_user_distance_restraint_features(atom_array)

    assert feats["user_distance_restraint_index"].shape == (2, 1)
    assert feats["user_distance_restraint_index"].dtype == torch.int64
    # entity1 res1 CA is index 1; entity2 res5 CA is index 4
    assert feats["user_distance_restraint_index"][:, 0].tolist() == [1, 4]
    assert feats["user_distance_restraint_lower_bound"].tolist() == pytest.approx([3.5])
    assert feats["user_distance_restraint_upper_bound"].tolist() == pytest.approx([8.0])


def test_empty_contact_emits_stable_empty_shapes(monkeypatch):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )
    sample = json_to_feature.SampleDictToFeatures(
        {"sequences": [], "constraint": {"contact": []}},
        extract_features_for_tfg=True,
    )
    atom_array = _synthetic_two_chain_atom_array()
    feats = sample.build_user_distance_restraint_features(atom_array)
    assert feats["user_distance_restraint_index"].shape == (2, 0)
    assert feats["user_distance_restraint_lower_bound"].shape == (0,)
    assert feats["user_distance_restraint_upper_bound"].shape == (0,)


def test_user_distance_restraint_potential_energy_and_grad():
    from opendde.tfg.potentials import UserDistanceRestraintPotential

    pot = UserDistanceRestraintPotential()
    # Two atoms 10 Å apart; restraint wants [3.5, 8.0] -> violation
    coords = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=torch.float32)
    feats = {
        "user_distance_restraint_index": torch.tensor([[0], [1]], dtype=torch.int64),
        "user_distance_restraint_lower_bound": torch.tensor([3.5], dtype=torch.float32),
        "user_distance_restraint_upper_bound": torch.tensor([8.0], dtype=torch.float32),
    }
    energy, grad = pot.energy_and_grad(coords, feats)
    assert energy.ndim == 1
    assert float(energy.item()) > 0.0
    assert grad.shape == coords.shape
    # Gradient should pull atoms closer (atom0 +x, atom1 -x)
    assert float(grad[0, 0, 0]) < 0.0  # pull atom0 toward atom1
    assert float(grad[0, 1, 0]) > 0.0  # pull atom1 toward atom0

    # Empty restraints -> zero energy
    empty = {
        "user_distance_restraint_index": torch.empty((2, 0), dtype=torch.int64),
        "user_distance_restraint_lower_bound": torch.empty((0,), dtype=torch.float32),
        "user_distance_restraint_upper_bound": torch.empty((0,), dtype=torch.float32),
    }
    e0 = pot.energy(coords, empty)
    assert float(e0.item()) == 0.0


def test_user_distance_restraint_registered_in_tfg_config():
    from opendde.tfg.config import _REQUIRED_FEATURES
    from opendde.tfg.potentials import CLASS_REGISTRY

    assert "UserDistanceRestraintPotential" in CLASS_REGISTRY
    keys = _REQUIRED_FEATURES["UserDistanceRestraintPotential"]
    assert "user_distance_restraint_index" in keys
    assert "user_distance_restraint_lower_bound" in keys
    assert "user_distance_restraint_upper_bound" in keys


def _tfg_sample(monkeypatch, constraint):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )
    sample = json_to_feature.SampleDictToFeatures(
        {"sequences": [], "constraint": constraint}, extract_features_for_tfg=True
    )
    sample.entity_poly_type = {"1": "polypeptide(L)", "2": "polypeptide(L)"}
    return sample


def test_contact_window_starts_at_3_5_angstrom_by_default(monkeypatch):
    sample = _tfg_sample(
        monkeypatch,
        {
            "contact": [
                {"entity1": "1", "position1": "1", "entity2": "2", "position2": "5"}
            ]
        },
    )
    feats = sample.build_user_distance_restraint_features(
        _synthetic_two_chain_atom_array()
    )
    assert feats["user_distance_restraint_lower_bound"].tolist() == pytest.approx([3.5])
    assert feats["user_distance_restraint_upper_bound"].tolist() == pytest.approx([8.0])


def test_epitope_defaults_are_the_whole_antibody_and_half_of_the_residues(
    monkeypatch, caplog
):
    sample = _tfg_sample(
        monkeypatch,
        {
            "movable_chains": ["B"],
            "epitope": {"residues": [{"entity": "1", "copy": 1, "position": "1"}]},
        },
    )
    atom_array = _synthetic_two_chain_atom_array()
    atom_array.set_annotation(
        "chain_id", np.array(["A", "A", "A", "B", "B", "B"], dtype=object)
    )
    movable = torch.tensor([False, False, False, True, True, True])
    with caplog.at_level("INFO"):
        feats = sample.build_epitope_features(atom_array, movable)
    # the paratope is every atom of the movable chain (no CDR window selects residue 5)
    assert torch.equal(feats["user_epitope_paratope_atom"], movable)
    assert "min_fraction=0.5" in caplog.text


def _patch_entity_array(monkeypatch):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )


def _contact(**overrides):
    pair = {
        "entity1": "1",
        "position1": "1",
        "atom1": "CA",
        "entity2": "2",
        "position2": "2",
        "atom2": "CA",
        "min_distance": 3.5,
        "max_distance": 8.0,
    }
    pair.update(overrides)
    return pair


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"min_distance": float("nan")}, "min_distance must be a finite number"),
        ({"max_distance": float("inf")}, "max_distance must be a finite number"),
        ({"min_distance": True}, "min_distance must be a finite number"),
        ({"max_distance": "8.0"}, "max_distance must be a finite number"),
        ({"min_distance": None}, "min_distance must be a finite number"),
        ({"min_distance": -5.0, "max_distance": -1.0}, "must be >= 0"),
        ({"min_distance": 9.0, "max_distance": 8.0}, "max_distance"),
        ({"entity1": True}, "entity1 must be an integer"),
        ({"position1": 9.9}, "position1 must be an integer"),
        ({"copy1": "x"}, "copy1 must be an integer"),
    ],
)
def test_contact_values_are_validated(monkeypatch, overrides, match):
    _patch_entity_array(monkeypatch)
    with pytest.raises(ValueError, match=match):
        json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": {"contact": [_contact(**overrides)]}}
        )


@pytest.mark.parametrize(
    "constraint, match",
    [
        ([1, 2], "'constraint' field must be an object"),
        ("contact", "'constraint' field must be an object"),
        ({"contact": {"a": 1}}, "constraint.contact must be a list"),
        ({"contact": "x"}, "constraint.contact must be a list"),
    ],
)
def test_malformed_constraint_is_an_error(monkeypatch, constraint, match):
    _patch_entity_array(monkeypatch)
    with pytest.raises(ValueError, match=match):
        json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": constraint}
        )


def test_null_constraint_is_treated_as_absent(monkeypatch):
    _patch_entity_array(monkeypatch)
    sample = json_to_feature.SampleDictToFeatures({"sequences": [], "constraint": None})
    assert sample.contact_restraints == []


@pytest.mark.parametrize("value", [True, "0.5", None, float("nan")])
def test_epitope_min_fraction_must_be_a_finite_number(monkeypatch, value):
    _patch_entity_array(monkeypatch)
    sample = json_to_feature.SampleDictToFeatures(
        {"sequences": [], "constraint": {"movable_chains": ["H"]}}
    )
    sample.single_sample_dict["constraint"]["epitope"] = {
        "residues": [{"entity": "1", "copy": 1, "position": "1"}],
        "min_fraction": value,
    }
    with pytest.raises(ValueError, match="min_fraction must be a finite number"):
        sample.build_epitope_features(None, torch.ones(1, dtype=torch.bool))
