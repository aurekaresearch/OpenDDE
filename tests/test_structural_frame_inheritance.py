# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.bonds import BondList

from opendde.data.core.featurizer import Featurizer
from opendde.data.tokenizer import Token, TokenArray


def _token(
    role: str, parent: int, has_frame: int, frame_atom_index: list[int]
) -> Token:
    token = Token(0)
    token.subtoken_role = role
    token.parent_residue_idx = parent
    token.has_frame = has_frame
    token.frame_atom_index = frame_atom_index
    return token


def test_sidechain_and_base_inherit_parent_backbone_frame():
    token_array = TokenArray(
        [
            _token("protein_bb", 0, 1, [0, 1, 2]),
            _token("protein_sc", 0, 0, [-1, -1, -1]),
            _token("dna_bb", 1, 1, [3, 4, 5]),
            _token("dna_base", 1, 0, [-1, -1, -1]),
        ]
    )

    updated = Featurizer.inherit_parent_backbone_frames(token_array)

    assert updated[1].has_frame == 1
    assert updated[1].frame_atom_index == [0, 1, 2]
    assert updated[3].has_frame == 1
    assert updated[3].frame_atom_index == [3, 4, 5]


def test_child_token_without_backbone_twin_keeps_invalid_frame():
    token_array = TokenArray(
        [
            _token("protein_sc", 0, 1, [9, 9, 9]),
        ]
    )

    updated = Featurizer.inherit_parent_backbone_frames(token_array)

    assert updated[0].has_frame == 0
    assert updated[0].frame_atom_index == [-1, -1, -1]


def _residue_token(atom_indices: list[int]) -> Token:
    token = Token(0)
    token.atom_indices = atom_indices
    token.atom_names = ["N", "CA", "C", "O"]
    token.centre_atom_index = atom_indices[1]
    return token


def test_partial_polymer_bond_graph_uses_fallback_for_missing_adjacent_edges():
    atom_names = ["N", "CA", "C", "O"] * 4
    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = np.array(atom_names)
    atom_array.res_name = np.array(["ALA"] * len(atom_names))
    atom_array.chain_id = np.array(["A"] * len(atom_names))
    atom_array.res_id = np.repeat(np.arange(1, 5), 4)
    atom_array.coord = np.zeros((len(atom_names), 3), dtype=np.float32)
    atom_array.set_annotation("mol_type", np.array(["protein"] * len(atom_names)))
    atom_array.set_annotation("asym_id_int", np.zeros(len(atom_names), dtype=int))
    # Only res1 C -- res2 N is explicit; res2-res3 and res3-res4 must use fallback.
    atom_array.bonds = BondList(
        len(atom_array),
        np.array([[2, 4, 1]], dtype=np.uint32),
    )
    token_array = TokenArray(
        [_residue_token(list(range(offset, offset + 4))) for offset in range(0, 16, 4)]
    )

    featurizer = Featurizer(
        cropped_token_array=token_array,
        cropped_atom_array=atom_array,
    )

    prev_parent, next_parent = featurizer.get_polymer_residue_graph()

    assert prev_parent.tolist() == [-1, 0, 1, 2]
    assert next_parent.tolist() == [1, 2, 3, -1]
