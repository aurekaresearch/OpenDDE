# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from biotite.structure import AtomArray

import opendde.data.template.template_featurizer as template_featurizer
from opendde.data.constants import ATOM37_NUM
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer
from opendde.data.template.template_input import (
    get_explicit_templates,
    needs_template_search,
)
from opendde.data.template.template_parser import (
    TemplateAtomMaskAllZerosError,
    TemplateParser,
)
from opendde.data.template.template_utils import TemplateHitProcessor


QUERY_SEQUENCE = "GHCIPTTSGPICLRD"
MMCIF_FIXTURE = Path(__file__).parents[1] / "examples" / "2lwu.cif"
EXAMPLE_INPUT = MMCIF_FIXTURE.parent / "example_explicit_template.json"


def _template_entry(source, query_indices=None, template_indices=None):
    if query_indices is None:
        query_indices = list(range(len(QUERY_SEQUENCE)))
    if template_indices is None:
        template_indices = list(range(len(QUERY_SEQUENCE)))
    return {
        "mmcifPath": str(source),
        "queryIndices": query_indices,
        "templateIndices": template_indices,
    }


def _atom_array_for_sequences(*sequences):
    lengths = [len(sequence) for sequence in sequences]
    asym_ids = np.concatenate(
        [np.full(length, asym_id) for asym_id, length in enumerate(lengths)]
    )
    atom_array = AtomArray(int(sum(lengths)))
    atom_array.set_annotation("asym_id_int", asym_ids)
    atom_array.set_annotation(
        "res_id", np.concatenate([np.arange(1, length + 1) for length in lengths])
    )
    atom_array.set_annotation(
        "chain_id",
        np.concatenate(
            [
                np.full(length, chr(ord("A") + asym_id))
                for asym_id, length in enumerate(lengths)
            ]
        ),
    )
    atom_array.set_annotation(
        "centre_atom_mask", np.ones(len(atom_array), dtype=np.int8)
    )
    return atom_array


def _protein_bioassembly(*, sequence=QUERY_SEQUENCE, count=1, **fields):
    return [
        {
            "proteinChain": {
                "sequence": sequence,
                "count": count,
                **fields,
            }
        }
    ]


@pytest.mark.parametrize(
    ("protein_chain", "expected"),
    [
        ({"sequence": "ACDE", "count": 1}, None),
        (
            {"sequence": "ACDE", "count": 1, "templatesPath": "hits.a3m"},
            None,
        ),
        ({"sequence": "ACDE", "count": 1, "templates": []}, []),
        (
            {
                "sequence": "ACDE",
                "count": 1,
                "templates": [
                    {
                        "mmcif": "data_demo\n#\n",
                        "queryIndices": [0],
                        "templateIndices": [0],
                    }
                ],
            },
            [
                {
                    "mmcif": "data_demo\n#\n",
                    "queryIndices": [0],
                    "templateIndices": [0],
                }
            ],
        ),
    ],
)
def test_get_explicit_templates(protein_chain, expected):
    assert get_explicit_templates(protein_chain) == expected


def test_template_input_rejects_explicit_and_search_hit_modes_together():
    with pytest.raises(ValueError, match="either explicit templates or templatesPath"):
        get_explicit_templates({"templates": [], "templatesPath": "hits.a3m"})


def test_template_input_rejects_null_explicit_templates():
    with pytest.raises(TypeError, match="templates must be a list"):
        get_explicit_templates({"templates": None})


def test_needs_template_search_preserves_mixed_chain_modes():
    mixed = [
        {
            "sequences": [
                {"proteinChain": {"sequence": "AAAAA", "count": 1}},
                {
                    "proteinChain": {
                        "sequence": "CCCCC",
                        "count": 1,
                        "templates": [],
                    }
                },
            ]
        }
    ]
    explicit_only = [
        {
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "DDDDD",
                        "count": 1,
                        "templates": [
                            {
                                "mmcif": "data_demo\n#\n",
                                "queryIndices": [0],
                                "templateIndices": [0],
                            }
                        ],
                    }
                },
            ]
        }
    ]

    assert needs_template_search(mixed)
    assert not needs_template_search(explicit_only)


@pytest.mark.parametrize("raw_templates", ["not-a-list", [42]])
def test_explicit_template_rejects_invalid_container_or_entry(raw_templates):
    with pytest.raises(TypeError, match="must be a list|must be an object"):
        InferenceTemplateFeaturizer._build_explicit_template_features(
            query_seq=QUERY_SEQUENCE,
            raw_templates=raw_templates,
            json_path=None,
        )


@pytest.mark.parametrize(
    ("template_info", "error"),
    [
        ({}, "requires either mmcif or mmcifPath"),
        (
            {"mmcif": "data_demo\n#\n", "mmcifPath": "demo.cif"},
            "accepts only one of mmcif or mmcifPath",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [],
                "templateIndices": [],
            },
            "non-empty",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [0],
                "templateIndices": [0, 1],
            },
            "same length",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [True],
                "templateIndices": [0],
            },
            "JSON integers",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [0.0],
                "templateIndices": [0],
            },
            "JSON integers",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [0, 0],
                "templateIndices": [0, 1],
            },
            "queryIndices must be unique",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [-1],
                "templateIndices": [0],
            },
            "non-negative",
        ),
        (
            {
                "mmcif": "data_demo\n#\n",
                "queryIndices": [len(QUERY_SEQUENCE)],
                "templateIndices": [0],
            },
            "queryIndices exceed",
        ),
    ],
)
def test_explicit_template_rejects_invalid_source_and_indices(template_info, error):
    with pytest.raises((TypeError, ValueError), match=error):
        InferenceTemplateFeaturizer._build_explicit_template_features(
            query_seq=QUERY_SEQUENCE,
            raw_templates=[template_info],
            json_path=None,
        )


@pytest.mark.parametrize(
    ("mmcif_object", "template_indices", "error"),
    [
        (None, [0], "Failed to parse"),
        (
            SimpleNamespace(
                chain_to_seqres={"A": "AC", "B": "DE"},
                header={"release_date": "2024-01-01"},
            ),
            [0],
            "exactly one protein chain",
        ),
        (
            SimpleNamespace(chain_to_seqres={"A": "AC"}, header={}),
            [0],
            "missing .*revision_date",
        ),
        (
            SimpleNamespace(
                chain_to_seqres={"A": "AC"},
                header={"release_date": "2024-01-01"},
            ),
            [2],
            "templateIndices exceed",
        ),
    ],
)
def test_explicit_template_rejects_invalid_parsed_structure(
    monkeypatch, mmcif_object, template_indices, error
):
    monkeypatch.setattr(
        TemplateParser,
        "parse",
        lambda **_kwargs: SimpleNamespace(mmcif_object=mmcif_object),
    )

    with pytest.raises(ValueError, match=error):
        InferenceTemplateFeaturizer._build_explicit_template_features(
            query_seq="AC",
            raw_templates=[
                {
                    "mmcif": "data_demo\n#\n",
                    "queryIndices": [0],
                    "templateIndices": template_indices,
                }
            ],
            json_path=None,
        )


def test_runnable_example_maps_only_selected_query_residues():
    payload = json.loads(EXAMPLE_INPUT.read_text(encoding="utf-8"))
    query_indices = payload[0]["sequences"][0]["proteinChain"]["templates"][0][
        "queryIndices"
    ]
    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=payload[0]["sequences"],
        atom_array=_atom_array_for_sequences(QUERY_SEQUENCE),
        use_template=True,
        online_template_featurizer=None,
        json_path=str(EXAMPLE_INPUT),
    )

    assert features["template_aatype"].shape == (4, 15)
    residue_mask = features["template_atom_mask"][0].any(axis=-1)
    assert residue_mask[query_indices].all()
    unmapped = sorted(set(range(15)) - set(query_indices))
    if unmapped:
        assert not residue_mask[unmapped].any()
    assert not features["template_atom_mask"][1:].any()


def test_nonidentity_index_mapping_places_exact_template_residues_and_coordinates():
    query_sequence = "A" * 18
    query_indices = [16, 1, 6]
    template_indices = [14, 7, 7]
    processor = TemplateHitProcessor(mmcif_dir="")

    features = InferenceTemplateFeaturizer._build_explicit_template_features(
        query_seq=query_sequence,
        raw_templates=[
            _template_entry(
                MMCIF_FIXTURE,
                query_indices=query_indices,
                template_indices=template_indices,
            )
        ],
        json_path=None,
    )[0]

    parsed = TemplateParser.parse(
        file_id="coordinate_reference",
        mmcif_string=MMCIF_FIXTURE.read_text(encoding="utf-8"),
    )
    assert parsed.mmcif_object is not None
    chain_id, template_sequence = next(
        iter(parsed.mmcif_object.chain_to_seqres.items())
    )
    source_positions, source_mask = processor._get_atom_positions(
        parsed.mmcif_object, chain_id, 150.0, _zero_center=True
    )

    expected_sequence = ["-"] * len(query_sequence)
    for query_index, template_index in zip(query_indices, template_indices):
        expected_sequence[query_index] = template_sequence[template_index]
        np.testing.assert_array_equal(
            features["template_all_atom_positions"][query_index],
            source_positions[template_index],
        )
        np.testing.assert_array_equal(
            features["template_all_atom_masks"][query_index],
            source_mask[template_index],
        )

    assert features["template_sequence"] == "".join(expected_sequence).encode()
    unmapped = sorted(set(range(len(query_sequence))) - set(query_indices))
    assert not features["template_all_atom_masks"][unmapped].any()


def test_explicit_mapping_accepts_one_nonzero_atom(monkeypatch):
    positions = np.zeros((len(QUERY_SEQUENCE), ATOM37_NUM, 3), dtype=np.float32)
    masks = np.zeros((len(QUERY_SEQUENCE), ATOM37_NUM), dtype=np.float32)
    positions[0, 0] = [1.0, 2.0, 3.0]
    masks[0, 0] = 1.0
    monkeypatch.setattr(
        TemplateHitProcessor,
        "_get_atom_positions",
        lambda *_args, **_kwargs: (positions, masks),
    )

    features = InferenceTemplateFeaturizer._build_explicit_template_features(
        query_seq=QUERY_SEQUENCE,
        raw_templates=[
            _template_entry(
                MMCIF_FIXTURE,
                query_indices=[0],
                template_indices=[0],
            )
        ],
        json_path=None,
    )[0]

    assert features["template_all_atom_masks"].sum() == 1


def test_legacy_hit_extraction_retains_five_atom_minimum(monkeypatch):
    processor = TemplateHitProcessor(mmcif_dir="unused")
    positions = np.zeros((1, ATOM37_NUM, 3), dtype=np.float32)
    masks = np.zeros((1, ATOM37_NUM), dtype=np.float32)
    masks[0, 0] = 1.0
    monkeypatch.setattr(
        processor,
        "_get_atom_positions",
        lambda *_args, **_kwargs: (positions, masks),
    )

    with pytest.raises(TemplateAtomMaskAllZerosError, match="Empty atom mask"):
        processor._extract_template_features(
            SimpleNamespace(chain_to_seqres={"A": "A"}),
            "legacy",
            {0: 0},
            "A",
            "A",
            "A",
        )


def test_explicit_template_rejects_all_zero_selected_atom_mask(monkeypatch):
    monkeypatch.setattr(
        TemplateHitProcessor,
        "_get_atom_positions",
        lambda *_args, **_kwargs: (
            np.zeros((len(QUERY_SEQUENCE), ATOM37_NUM, 3), dtype=np.float32),
            np.zeros((len(QUERY_SEQUENCE), ATOM37_NUM), dtype=np.float32),
        ),
    )

    with pytest.raises(TemplateAtomMaskAllZerosError, match="Empty atom mask"):
        InferenceTemplateFeaturizer._build_explicit_template_features(
            query_seq=QUERY_SEQUENCE,
            raw_templates=[
                _template_entry(
                    MMCIF_FIXTURE,
                    query_indices=[0],
                    template_indices=[0],
                )
            ],
            json_path=None,
        )


def test_explicit_templates_keep_input_order_and_do_not_parse_after_four():
    entries = [
        _template_entry(
            MMCIF_FIXTURE,
            query_indices=[index],
            template_indices=[index],
        )
        for index in range(1, 5)
    ]
    entries.append(
        _template_entry(
            Path("unused-missing-template.cif"),
            query_indices=[5],
            template_indices=[5],
        )
    )

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=_protein_bioassembly(templates=entries),
        atom_array=_atom_array_for_sequences(QUERY_SEQUENCE),
        use_template=True,
        online_template_featurizer=None,
    )

    assert features["template_atom_mask"].shape[0] == 4
    for template_index, query_index in enumerate(range(1, 5)):
        residue_mask = features["template_atom_mask"][template_index].any(axis=-1)
        assert residue_mask[query_index]
        assert residue_mask.sum() == 1


def test_inline_explicit_mmcif_uses_the_same_feature_path():
    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=_protein_bioassembly(
            templates=[
                {
                    "mmcif": MMCIF_FIXTURE.read_text(encoding="utf-8"),
                    "queryIndices": list(range(15)),
                    "templateIndices": list(range(15)),
                }
            ]
        ),
        atom_array=_atom_array_for_sequences(QUERY_SEQUENCE),
        use_template=True,
        online_template_featurizer=None,
    )

    assert features["template_atom_mask"][0].any()


def test_global_template_disable_does_not_parse_explicit_entries(monkeypatch):
    monkeypatch.setattr(
        TemplateParser,
        "parse",
        lambda **_kwargs: pytest.fail("templates must not be parsed when disabled"),
    )

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=_protein_bioassembly(
            sequence="ACDEF", templates=[{"invalid": "ignored"}]
        ),
        atom_array=_atom_array_for_sequences("ACDEF"),
        use_template=False,
        online_template_featurizer=None,
    )

    assert not features["template_atom_mask"].any()


def test_explicit_template_missing_path_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        InferenceTemplateFeaturizer._load_explicit_template_mmcif(
            {"mmcifPath": "missing.cif"},
            json_path=str(tmp_path / "input.json"),
            template_index=0,
        )


def test_explicit_template_list_is_copied_to_each_chain_count():
    atom_array = _atom_array_for_sequences(QUERY_SEQUENCE, QUERY_SEQUENCE)

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=_protein_bioassembly(
            count=2, templates=[_template_entry(MMCIF_FIXTURE)]
        ),
        atom_array=atom_array,
        use_template=True,
        online_template_featurizer=None,
    )

    first = features["template_atom_mask"][0, :15]
    second = features["template_atom_mask"][0, 15:]
    assert first.any()
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("suffix", "parser_name"),
    [(".a3m", "HmmsearchA3MParser"), (".hhr", "HHRParser")],
)
def test_legacy_search_hit_path_still_uses_existing_featurizer(
    tmp_path, monkeypatch, suffix, parser_name
):
    hit_path = tmp_path / f"hits{suffix}"
    hit_path.write_text(">query\nACDEF\n", encoding="utf-8")
    captured = []

    class _SearchFeaturizer:
        def get_templates(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(features=[]), {}

    monkeypatch.setattr(
        getattr(template_featurizer, parser_name), "parse", lambda **_kwargs: []
    )

    features = InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=_protein_bioassembly(sequence="ACDEF", templatesPath=str(hit_path)),
        atom_array=_atom_array_for_sequences("ACDEF"),
        use_template=True,
        online_template_featurizer=_SearchFeaturizer(),
    )

    assert len(captured) == 1
    assert features["template_aatype"].shape == (4, 5)
