# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import pytest

from opendde.data.template.template_parser import TemplateParser


def _polymer_data(polymer_type):
    return {
        "_entity_poly_seq.entity_id": ["1"],
        "_entity_poly_seq.mon_id": ["ALA"],
        "_entity_poly_seq.num": ["1"],
        "_chem_comp.id": ["ALA"],
        "_chem_comp.type": ["."],
        "_entity_poly.entity_id": ["1"],
        "_entity_poly.type": [polymer_type],
        "_struct_asym.id": ["A"],
        "_struct_asym.entity_id": ["1"],
    }


@pytest.mark.parametrize("polymer_type", ["polypeptide(L)", "polypeptide(D)"])
def test_parser_uses_entity_poly_type_when_component_types_are_unknown(polymer_type):
    chains = TemplateParser._get_protein_chains(_polymer_data(polymer_type))

    assert list(chains) == ["A"]
    assert chains["A"][0].id == "ALA"


@pytest.mark.parametrize(
    "polymer_type", ["peptide nucleic acid", "cyclic-pseudo-peptide"]
)
def test_parser_rejects_non_polypeptide_entity_types(polymer_type):
    assert TemplateParser._get_protein_chains(_polymer_data(polymer_type)) == {}
