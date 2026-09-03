# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import numpy as np
import pytest

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


def test_constraint_field_warns_that_it_is_ignored(monkeypatch, caplog):
    monkeypatch.setattr(
        json_to_feature,
        "add_entity_atom_array",
        lambda sample: {"sequences": sample["sequences"]},
    )

    with caplog.at_level("WARNING"):
        sample = json_to_feature.SampleDictToFeatures(
            {"sequences": [], "constraint": {"contact": []}}
        )

    assert sample.entity_poly_type == {}
    assert "constraint" in caplog.text
    assert "ignored" in caplog.text
