# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

import numpy as np
import pytest
from biotite.structure import AtomArray

from opendde.data.msa.msa_featurizer import InferenceMSAFeaturizer
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer


PROTEIN_SEQUENCE = "ACDEF"
EntitySpec = tuple[str, int]


def _make_input(entity_specs: tuple[EntitySpec, ...]) -> tuple[list[dict], AtomArray]:
    bioassembly = []
    asym_ids = []
    res_ids = []
    chain_ids = []
    curr_asym_id = 0

    for entity_type, count in entity_specs:
        if entity_type == "protein":
            bioassembly.append(
                {
                    "proteinChain": {
                        "sequence": PROTEIN_SEQUENCE,
                        "count": count,
                    }
                }
            )
            entity_size = len(PROTEIN_SEQUENCE)
        elif entity_type == "ion":
            bioassembly.append({"ion": {"ion": "ZN", "count": count}})
            entity_size = 1
        elif entity_type == "ligand":
            bioassembly.append({"ligand": {"ligand": "CCD_ATP", "count": count}})
            entity_size = 2
        else:
            raise ValueError(f"Unsupported test entity type: {entity_type}")

        for _ in range(count):
            chain_id = chr(ord("A") + curr_asym_id)
            asym_ids.extend([curr_asym_id] * entity_size)
            res_ids.extend(range(1, entity_size + 1))
            chain_ids.extend([chain_id] * entity_size)
            curr_asym_id += 1

    atom_array = AtomArray(len(asym_ids))
    atom_array.res_id[:] = res_ids
    atom_array.chain_id[:] = chain_ids
    atom_array.set_annotation("asym_id_int", np.asarray(asym_ids, dtype=np.int64))
    atom_array.set_annotation("centre_atom_mask", np.ones(len(asym_ids), dtype=np.int8))
    return bioassembly, atom_array


@pytest.mark.parametrize(
    "entity_specs",
    [
        (("protein", 1), ("ion", 1)),
        (("ion", 1), ("protein", 1)),
        (("protein", 1), ("ion", 2)),
        (("protein", 1), ("ligand", 1), ("ion", 1)),
        (("ion", 1), ("ion", 1), ("protein", 1)),
        (("protein", 1), ("ion", 1), ("protein", 1)),
    ],
)
def test_msa_feature_metadata_includes_ion(
    entity_specs: tuple[EntitySpec, ...],
) -> None:
    bioassembly, atom_array = _make_input(entity_specs)

    features = InferenceMSAFeaturizer.make_msa_feature(
        bioassembly=bioassembly,
        atom_array=atom_array,
        msa_pair_as_unpair=False,
        use_rna_msa=False,
    )

    assert features["msa"].shape[1] == len(atom_array)
    assert features["profile"].shape[0] == len(atom_array)


@pytest.mark.parametrize(
    "entity_specs",
    [
        (("protein", 1), ("ion", 1)),
        (("ion", 1), ("protein", 1)),
        (("protein", 1), ("ion", 2)),
        (("protein", 1), ("ligand", 1), ("ion", 1)),
        (("ion", 1), ("ion", 1), ("protein", 1)),
        (("protein", 1), ("ion", 1), ("protein", 1)),
    ],
)
def test_template_feature_metadata_includes_ion(
    entity_specs: tuple[EntitySpec, ...],
) -> None:
    bioassembly, atom_array = _make_input(entity_specs)

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=bioassembly,
        atom_array=atom_array,
        use_template=False,
        online_template_featurizer=None,
    )

    assert features["template_aatype"].shape[1] == len(atom_array)
    assert features["template_atom_mask"].shape[1] == len(atom_array)
