# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, BondList

from opendde.data import utils as data_utils


def _make_atom_array() -> AtomArray:
    atom_array = AtomArray(2)
    atom_array.coord = np.zeros((2, 3), dtype=np.float32)
    atom_array.set_annotation("is_resolved", np.array([True, False]))
    return atom_array


def _make_terminal_oxt_atom_array(
    *, externally_bonded_atom_name: str | None = None
) -> AtomArray:
    atom_names = ["N", "CA", "C", "O", "OXT"]
    has_external_atom = externally_bonded_atom_name is not None
    atom_array = AtomArray(len(atom_names) + int(has_external_atom))
    atom_array.atom_name[:5] = atom_names
    atom_array.res_name[:5] = "ALA"
    atom_array.chain_id[:5] = "A"
    atom_array.res_id[:5] = 1
    atom_array.element[:5] = ["N", "C", "C", "O", "O"]
    atom_array.set_annotation(
        "mol_type",
        np.array(["protein"] * 5 + (["ligand"] if has_external_atom else [])),
    )
    atom_array.set_annotation(
        "label_asym_id",
        np.array(["A"] * 5 + (["B"] if has_external_atom else [])),
    )

    ref_pos = np.array(
        [
            [-2.0, 0.5, 0.0],
            [-1.52, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 1.15, 0.0],
            [0.5, -1.15, 0.0],
        ]
        + ([[0.0, 0.0, 0.0]] if has_external_atom else []),
        dtype=np.float32,
    )
    atom_array.coord = ref_pos.copy()
    atom_array.set_annotation("ref_pos", ref_pos)
    atom_array.set_annotation("ref_mask", np.ones(len(atom_array), dtype=int))

    bonds = [[1, 2, 1], [2, 3, 2], [2, 4, 1]]
    if externally_bonded_atom_name is not None:
        atom_array.atom_name[5] = "C1"
        atom_array.res_name[5] = "UNL"
        atom_array.chain_id[5] = "B"
        atom_array.res_id[5] = 1
        atom_array.element[5] = "C"
        bonds.append([atom_names.index(externally_bonded_atom_name), 5, 1])
    atom_array.bonds = BondList(len(atom_array), np.array(bonds, dtype=np.uint32))
    return atom_array


def test_save_structure_cif_does_not_save_wounresol_by_default(monkeypatch, tmp_path):
    saved_paths = []

    def fake_save_atoms_to_cif(output_cif_file, atom_array, entity_poly_type, pdb_id):
        saved_paths.append(output_cif_file)

    monkeypatch.setattr(data_utils, "save_atoms_to_cif", fake_save_atoms_to_cif)

    output_path = str(tmp_path / "model.cif")
    data_utils.save_structure_cif(
        atom_array=_make_atom_array(),
        pred_coordinate=torch.zeros((2, 3)),
        output_fpath=output_path,
        entity_poly_type={},
        pdb_id="test",
    )

    assert saved_paths == [output_path]


def test_save_structure_cif_can_save_wounresol_when_requested(monkeypatch, tmp_path):
    saved = []

    def fake_save_atoms_to_cif(output_cif_file, atom_array, entity_poly_type, pdb_id):
        saved.append((output_cif_file, len(atom_array)))

    monkeypatch.setattr(data_utils, "save_atoms_to_cif", fake_save_atoms_to_cif)

    output_path = str(tmp_path / "model.cif")
    data_utils.save_structure_cif(
        atom_array=_make_atom_array(),
        pred_coordinate=torch.zeros((2, 3)),
        output_fpath=output_path,
        entity_poly_type={},
        pdb_id="test",
        save_wo_unresolved=True,
    )

    assert saved == [
        (output_path, 2),
        (str(tmp_path / "model_wounresol.cif"), 1),
    ]


def test_save_structure_cif_repairs_detached_terminal_oxt(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    transformed_ref_pos = atom_array.ref_pos @ rotation.T + [10.0, 20.0, 30.0]
    pred_coordinate = torch.tensor(transformed_ref_pos)
    pred_coordinate[4] = torch.tensor([100.0, 100.0, 100.0])
    original_pred_coordinate = pred_coordinate.clone()
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    assert len(saved) == 1
    np.testing.assert_allclose(
        saved[0].coord[4],
        transformed_ref_pos[4],
        atol=1e-6,
    )
    torch.testing.assert_close(pred_coordinate, original_pred_coordinate)


def test_save_structure_cif_keeps_valid_terminal_oxt(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    np.testing.assert_allclose(saved[0].coord, pred_coordinate.numpy(), atol=1e-6)


def test_save_structure_cif_repairs_coincident_terminal_oxygens(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    pred_coordinate[4] = pred_coordinate[3]
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    assert np.linalg.norm(saved[0].coord[3] - saved[0].coord[4]) >= 1.8


def test_save_structure_cif_repairs_stretched_terminal_oxt(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    pred_coordinate[4] = pred_coordinate[2] + pred_coordinate.new_tensor(
        [-1.9, 0.0, 0.0]
    )
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    c_oxt_distance = np.linalg.norm(saved[0].coord[2] - saved[0].coord[4])
    assert c_oxt_distance <= 1.7


def test_save_structure_cif_repairs_compressed_terminal_oxt(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    pred_coordinate[4] = pred_coordinate[2] + pred_coordinate.new_tensor(
        [-0.9, 0.0, 0.0]
    )
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    c_oxt_distance = np.linalg.norm(saved[0].coord[2] - saved[0].coord[4])
    assert c_oxt_distance >= 1.0


def test_save_structure_cif_skips_invalid_terminal_anchor_geometry(
    monkeypatch, tmp_path
):
    atom_array = _make_terminal_oxt_atom_array()
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    pred_coordinate[3] = torch.tensor([50.0, 50.0, 50.0])
    pred_coordinate[4] = torch.tensor([100.0, 100.0, 100.0])
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    np.testing.assert_allclose(saved[0].coord, pred_coordinate.numpy(), atol=1e-6)


def test_save_structure_cif_repairs_nan_terminal_oxt(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    pred_coordinate[4] = torch.nan
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    assert np.all(np.isfinite(saved[0].coord[4]))


def test_save_structure_cif_repairs_each_protein_chain(monkeypatch, tmp_path):
    atom_array = _make_terminal_oxt_atom_array()
    second_chain = _make_terminal_oxt_atom_array()
    second_chain.chain_id[:] = "B"
    second_chain.label_asym_id[:] = "B"
    atom_array += second_chain
    pred_coordinate = torch.tensor(atom_array.ref_pos + [3.0, 2.0, 1.0])
    pred_coordinate[[4, 9]] = pred_coordinate.new_tensor([100.0, 100.0, 100.0])
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    for carbon_index, oxt_index in ((2, 4), (7, 9)):
        c_oxt_distance = np.linalg.norm(
            saved[0].coord[carbon_index] - saved[0].coord[oxt_index]
        )
        assert 1.0 <= c_oxt_distance <= 1.7


@pytest.mark.parametrize("externally_bonded_atom_name", ["C", "O", "OXT"])
def test_save_structure_cif_skips_externally_bonded_terminal_carboxyl(
    monkeypatch, tmp_path, externally_bonded_atom_name
):
    atom_array = _make_terminal_oxt_atom_array(
        externally_bonded_atom_name=externally_bonded_atom_name
    )
    pred_coordinate = torch.tensor(atom_array.ref_pos + [10.0, 20.0, 30.0])
    pred_coordinate[4] = torch.tensor([100.0, 100.0, 100.0])
    saved = []

    monkeypatch.setattr(
        data_utils,
        "save_atoms_to_cif",
        lambda _path, output, _types, _id: saved.append(output.copy()),
    )

    data_utils.save_structure_cif(
        atom_array=atom_array,
        pred_coordinate=pred_coordinate,
        output_fpath=str(tmp_path / "model.cif"),
        entity_poly_type={"1": "polypeptide(L)"},
        pdb_id="test",
    )

    np.testing.assert_allclose(saved[0].coord, pred_coordinate.numpy(), atol=1e-6)
