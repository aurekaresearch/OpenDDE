# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
from biotite.structure import AtomArray

from opendde.data.core.parser import AddAtomArrayAnnot, MMCIFParser


def _make_ala_atom_array(chain_ids: list[str]) -> AtomArray:
    atom_names = ["N", "CA", "C", "O", "CB"] * len(chain_ids)
    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = np.array(atom_names)
    atom_array.res_name = np.array(["ALA"] * len(atom_names))
    atom_array.chain_id = np.repeat(chain_ids, 5)
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.element = np.array(["N", "C", "C", "O", "C"] * len(chain_ids))
    atom_array.coord = np.zeros((len(atom_names), 3), dtype=np.float32)
    atom_array.set_annotation(
        "label_entity_id",
        np.repeat([str(i + 1) for i in range(len(chain_ids))], 5),
    )
    return atom_array


def _patch_standard_ccd(monkeypatch):
    from opendde.data.core import parser as parser_module

    monkeypatch.setattr(parser_module.ccd, "get_mol_type", lambda res_name: "protein")
    monkeypatch.setattr(
        parser_module.ccd,
        "get_one_letter_code",
        lambda res_name: {"ALA": "A"}.get(res_name),
    )


def test_mmcif_parser_mse_to_met_uses_shared_normalization():
    atom_array = AtomArray(1)
    atom_array.atom_name = np.array(["SE"])
    atom_array.res_name = np.array(["MSE"])
    atom_array.element = np.array(["Se"])
    atom_array.hetero = np.ones(1, dtype=bool)

    MMCIFParser.mse_to_met(atom_array)

    assert atom_array.atom_name.tolist() == ["SD"]
    assert atom_array.res_name.tolist() == ["MET"]
    assert atom_array.element.tolist() == ["S"]
    assert atom_array.hetero.tolist() == [False]


def test_add_token_annotations_preserves_standard_mask_relationships(monkeypatch):
    _patch_standard_ccd(monkeypatch)
    atom_array = _make_ala_atom_array(["A"])

    atom_array = AddAtomArrayAnnot.add_token_annotations(
        atom_array,
        {"1": "polypeptide(L)"},
    )

    assert atom_array.mol_type.tolist() == ["protein"] * 5
    assert atom_array.centre_atom_mask.tolist() == [0, 1, 0, 0, 0]
    assert atom_array.distogram_rep_atom_mask.tolist() == [0, 0, 0, 0, 1]
    assert atom_array.plddt_m_rep_atom_mask.tolist() == [0, 1, 0, 0, 0]
    assert atom_array.modified_res_mask.tolist() == [0, 0, 0, 0, 0]
    assert atom_array.centre_atom_mask.sum() == atom_array.distogram_rep_atom_mask.sum()
