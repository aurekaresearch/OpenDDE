# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
from biotite.structure import AtomArray

from opendde.data.tokenizer import (
    NO_TWIN_TOKEN_IDX,
    STRUCTURAL_TOKEN_ROLES,
    AtomArrayTokenizer,
)


def _make_atom_array(
    atom_names: list[str],
    res_names: list[str],
    mol_types: list[str],
    elements: list[str],
    centre_atom_mask: list[int],
) -> AtomArray:
    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = np.array(atom_names)
    atom_array.res_name = np.array(res_names)
    atom_array.chain_id = np.array(["A"] * len(atom_names))
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.element = np.array(elements)
    atom_array.coord = np.zeros((len(atom_names), 3), dtype=np.float32)
    atom_array.set_annotation("mol_type", np.array(mol_types))
    atom_array.set_annotation("centre_atom_mask", np.array(centre_atom_mask))
    return atom_array


def test_standard_protein_residue_splits_backbone_and_sidechain_tokens():
    atom_array = _make_atom_array(
        atom_names=["N", "CA", "C", "O", "CB", "CG"],
        res_names=["ALA"] * 6,
        mol_types=["protein"] * 6,
        elements=["N", "C", "C", "O", "C", "C"],
        centre_atom_mask=[0, 1, 0, 0, 0, 0],
    )

    residue_tokens = AtomArrayTokenizer(atom_array).get_token_array()
    structural_tokens = AtomArrayTokenizer(atom_array).get_structural_token_array(
        residue_tokens
    )

    assert len(residue_tokens) == 1
    assert len(structural_tokens) == 2
    assert structural_tokens.get_annotation("subtoken_role") == [
        "protein_bb",
        "protein_sc",
    ]
    assert structural_tokens.get_annotation("subtoken_role_id") == [
        STRUCTURAL_TOKEN_ROLES["protein_bb"],
        STRUCTURAL_TOKEN_ROLES["protein_sc"],
    ]
    assert structural_tokens.get_annotation("atom_indices") == [[0, 1, 2, 3], [4, 5]]
    assert structural_tokens.get_annotation("centre_atom_index") == [1, 4]
    assert structural_tokens.get_annotation("parent_residue_idx") == [0, 0]
    assert structural_tokens.get_annotation("residue_token_group_id") == [0, 0]
    assert structural_tokens.get_annotation("twin_token_idx") == [1, 0]


def test_glycine_stays_single_backbone_token_without_empty_sidechain_token():
    atom_array = _make_atom_array(
        atom_names=["N", "CA", "C", "O"],
        res_names=["GLY"] * 4,
        mol_types=["protein"] * 4,
        elements=["N", "C", "C", "O"],
        centre_atom_mask=[0, 1, 0, 0],
    )

    structural_tokens = AtomArrayTokenizer(atom_array).get_structural_token_array()

    assert len(structural_tokens) == 1
    assert structural_tokens.get_annotation("subtoken_role") == ["protein_bb"]
    assert structural_tokens.get_annotation("atom_indices") == [[0, 1, 2, 3]]
    assert structural_tokens.get_annotation("twin_token_idx") == [NO_TWIN_TOKEN_IDX]


def test_standard_dna_residue_splits_backbone_and_base_tokens():
    atom_names = [
        "P",
        "OP1",
        "OP2",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "C1'",
        "N9",
        "C8",
        "N7",
        "C5",
        "C4",
    ]
    atom_array = _make_atom_array(
        atom_names=atom_names,
        res_names=["DA"] * len(atom_names),
        mol_types=["dna"] * len(atom_names),
        elements=[
            "P",
            "O",
            "O",
            "O",
            "C",
            "C",
            "O",
            "C",
            "O",
            "C",
            "C",
            "N",
            "C",
            "N",
            "C",
            "C",
        ],
        centre_atom_mask=[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
        ],
    )

    structural_tokens = AtomArrayTokenizer(atom_array).get_structural_token_array()

    assert len(structural_tokens) == 2
    assert structural_tokens.get_annotation("subtoken_role") == ["dna_bb", "dna_base"]
    assert structural_tokens.get_annotation("atom_indices") == [
        list(range(11)),
        list(range(11, 16)),
    ]
    assert structural_tokens.get_annotation("centre_atom_index") == [5, 11]
    assert structural_tokens.get_annotation("twin_token_idx") == [1, 0]


def test_ligand_and_modified_residues_remain_atom_tokens():
    ligand_atom_array = _make_atom_array(
        atom_names=["C1", "O1"],
        res_names=["LIG", "LIG"],
        mol_types=["ligand", "ligand"],
        elements=["C", "O"],
        centre_atom_mask=[1, 1],
    )
    modified_atom_array = _make_atom_array(
        atom_names=["N", "CA", "P"],
        res_names=["SEP", "SEP", "SEP"],
        mol_types=["protein", "protein", "protein"],
        elements=["N", "C", "P"],
        centre_atom_mask=[1, 1, 1],
    )

    ligand_tokens = AtomArrayTokenizer(ligand_atom_array).get_structural_token_array()
    modified_tokens = AtomArrayTokenizer(
        modified_atom_array
    ).get_structural_token_array()

    assert len(ligand_tokens) == 2
    assert ligand_tokens.get_annotation("subtoken_role") == ["atom", "atom"]
    assert ligand_tokens.get_annotation("atom_indices") == [[0], [1]]
    assert ligand_tokens.get_annotation("parent_residue_idx") == [0, 1]

    assert len(modified_tokens) == 3
    assert modified_tokens.get_annotation("subtoken_role") == ["atom", "atom", "atom"]
    assert modified_tokens.get_annotation("atom_indices") == [[0], [1], [2]]
    assert modified_tokens.get_annotation("parent_residue_idx") == [0, 1, 2]
