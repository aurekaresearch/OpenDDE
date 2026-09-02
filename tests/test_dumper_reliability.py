# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray

from runner.dumper import DataDumper, get_clean_full_confidence


def _patch_minimal_writers(monkeypatch, dumper, *, fail_confidence=False):
    def save_structure(**kwargs):
        Path(kwargs["prediction_save_dir"], "new.cif").write_text("new")

    def save_confidence(**kwargs):
        if fail_confidence:
            raise OSError("simulated output failure")
        Path(kwargs["prediction_save_dir"], "new.json").write_text("new")

    monkeypatch.setattr(dumper, "_save_structure", save_structure)
    monkeypatch.setattr(dumper, "_save_confidence", save_confidence)


def _minimal_prediction():
    return {
        "coordinate": torch.zeros(1, 1, 3),
        "summary_confidence": [{"ranking_score": 1.0}],
        "full_data": [{}],
    }


def test_dump_predictions_replaces_prior_output_set(tmp_path, monkeypatch):
    dumper = DataDumper(str(tmp_path))
    dump_dir = tmp_path / "job" / "seed_1"
    old_dir = dump_dir / "predictions"
    old_dir.mkdir(parents=True)
    (old_dir / "stale_sample_4.cif").write_text("stale")
    (old_dir / "stale_full_data_sample_4.json").write_text("stale")
    _patch_minimal_writers(monkeypatch, dumper)

    dumper.dump_predictions(
        pred_dict=_minimal_prediction(),
        dump_dir=str(dump_dir),
        pdb_id="job",
        atom_array=None,
        entity_poly_type={},
        seed=1,
    )

    assert {path.name for path in old_dir.iterdir()} == {"new.cif", "new.json"}
    assert not list(dump_dir.glob(".predictions-*"))


def test_dump_predictions_preserves_prior_set_when_staging_fails(tmp_path, monkeypatch):
    dumper = DataDumper(str(tmp_path))
    dump_dir = tmp_path / "job" / "seed_1"
    old_dir = dump_dir / "predictions"
    old_dir.mkdir(parents=True)
    (old_dir / "previous.cif").write_text("previous")
    _patch_minimal_writers(monkeypatch, dumper, fail_confidence=True)

    with pytest.raises(OSError, match="simulated output failure"):
        dumper.dump_predictions(
            pred_dict=_minimal_prediction(),
            dump_dir=str(dump_dir),
            pdb_id="job",
            atom_array=None,
            entity_poly_type={},
            seed=1,
        )

    assert {path.name for path in old_dir.iterdir()} == {"previous.cif"}
    assert not list(dump_dir.glob(".predictions-*"))


def test_confidence_serialization_does_not_mutate_prediction_tree():
    nested = torch.tensor([1.234], dtype=torch.float32)
    confidence = {
        "atom_coordinate": torch.ones(1, 3),
        "atom_is_polymer": torch.ones(1, dtype=torch.bool),
        "nested": {"score": nested},
        "values": np.array([2.345]),
    }

    cleaned = get_clean_full_confidence(confidence)

    assert "atom_coordinate" in confidence
    assert "atom_is_polymer" in confidence
    assert confidence["nested"]["score"] is nested
    assert "atom_coordinate" not in cleaned
    assert "atom_is_polymer" not in cleaned
    np.testing.assert_allclose(cleaned["nested"]["score"], np.array([1.23]))
    np.testing.assert_allclose(cleaned["values"], np.array([2.35]))


@pytest.mark.parametrize(
    ("group_name", "seed", "message"),
    [
        ("../escape", 1, "safe path component"),
        ("group", "../../escape", "output seed must be an integer"),
        ("group", True, "output seed must be an integer, not a boolean"),
    ],
)
def test_dumper_rejects_unsafe_library_output_coordinates(
    tmp_path, group_name, seed, message
):
    dumper = DataDumper(str(tmp_path))

    with pytest.raises(ValueError, match=message):
        dumper._get_dump_dir(group_name, "sample", seed)


def test_structure_serialization_does_not_annotate_caller_atom_array(
    tmp_path, monkeypatch
):
    dumper = DataDumper(str(tmp_path))
    atom_array = AtomArray(1)
    written = []
    monkeypatch.setattr(
        "runner.dumper.save_structure_cif",
        lambda **kwargs: written.append(kwargs["atom_array"]),
    )

    dumper._save_structure(
        pred_coordinates=torch.zeros(1, 1, 3),
        prediction_save_dir=str(tmp_path),
        sample_name="sample",
        atom_array=atom_array,
        entity_poly_type={},
        seed=1,
        sorted_indices=[0],
        b_factor=[np.array([87.65])],
    )

    assert "b_factor" not in atom_array.get_annotation_categories()
    assert written[0] is not atom_array
    np.testing.assert_array_equal(written[0].b_factor, np.array([87.65]))
