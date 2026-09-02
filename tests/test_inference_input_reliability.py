# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
from pathlib import Path

import pytest

from opendde.data.inference.input_validation import (
    validate_inference_jobs,
    validate_inference_seed,
    validate_sample_name,
)


@pytest.mark.parametrize(
    "name", ["", None, ".", "..", "../escape", "/tmp/out", "a/b", "ERR"]
)
def test_validate_sample_name_rejects_unsafe_output_components(name):
    with pytest.raises(ValueError):
        validate_sample_name(name)


def test_validate_inference_jobs_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicated"):
        validate_inference_jobs([{"name": "same"}, {"name": "same"}])


@pytest.mark.parametrize(
    "model_seeds",
    [7, [True], ["not-an-int"], [1.5], [-1], [2**32]],
)
def test_validate_inference_jobs_rejects_invalid_model_seeds(model_seeds):
    with pytest.raises(ValueError, match="modelSeeds"):
        validate_inference_jobs([{"name": "job", "modelSeeds": model_seeds}])


def test_validate_inference_jobs_allows_empty_model_seed_fallback():
    assert validate_inference_jobs([{"name": "job", "modelSeeds": []}]) == [
        {"name": "job", "modelSeeds": []}
    ]


def test_validate_inference_seed_preserves_quoted_integer_compatibility():
    assert validate_inference_seed("42") == 42


def test_discover_inference_jsons_is_sorted_and_excludes_generated_files(tmp_path):
    from runner.batch_inference import _discover_inference_jsons

    input_dir = tmp_path / "inputs"
    output_dir = input_dir / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()
    for relative_path in [
        "z.json",
        "a.json",
        "a-update-msa.json",
        "z-final-updated.json",
    ]:
        (input_dir / relative_path).write_text("[]", encoding="utf-8")
    (output_dir / "confidence.json").write_text("{}", encoding="utf-8")

    assert _discover_inference_jsons(str(input_dir), str(output_dir)) == [
        str(input_dir / "a.json"),
        str(input_dir / "z.json"),
    ]


def test_input_collection_rejects_cross_file_output_collisions(tmp_path):
    from runner.batch_inference import _validate_input_collection

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('[{"name": "same"}]', encoding="utf-8")
    second.write_text('[{"name": "same"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="outputs would collide"):
        _validate_input_collection([str(first), str(second)])


def test_inference_jsons_rejects_bad_input_before_runner_initialization(
    tmp_path, monkeypatch
):
    from runner import batch_inference

    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: pytest.fail("invalid input must not initialize the model"),
    )

    with pytest.raises(RuntimeError, match="Can not read input"):
        batch_inference.inference_jsons(str(tmp_path / "missing.json"))


def test_inference_jsons_rejects_output_collision_before_runner_initialization(
    tmp_path, monkeypatch
):
    from runner import batch_inference

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('[{"name": "same"}]', encoding="utf-8")
    second.write_text('[{"name": "same"}]', encoding="utf-8")
    monkeypatch.setattr(
        batch_inference,
        "get_default_runner",
        lambda **_kwargs: pytest.fail("invalid input must not initialize the model"),
    )

    with pytest.raises(ValueError, match="outputs would collide"):
        batch_inference.inference_jsons(str(tmp_path))


def test_load_inference_jobs_validates_before_featurization(tmp_path, monkeypatch):
    from runner import inference

    path = tmp_path / "input.json"
    path.write_text(json.dumps([{"name": "job", "modelSeeds": [7]}]))
    monkeypatch.setattr(inference.dist, "is_available", lambda: False)

    assert inference._load_inference_jobs_synchronized(str(path)) == [
        {"name": "job", "modelSeeds": [7]}
    ]


def test_load_inference_jobs_reports_missing_name_without_secondary_error(
    tmp_path, monkeypatch
):
    from runner import inference

    path = tmp_path / "input.json"
    path.write_text("[{}]", encoding="utf-8")
    monkeypatch.setattr(inference.dist, "is_available", lambda: False)

    with pytest.raises(ValueError, match="must be a non-empty string"):
        inference._load_inference_jobs_synchronized(str(path))


def test_preprocess_skips_msa_conversion_when_msa_is_disabled(tmp_path, monkeypatch):
    from runner import batch_inference

    input_path = tmp_path / "readonly-input.json"
    input_path.write_text('[{"name": "job"}]', encoding="utf-8")
    monkeypatch.setattr(
        batch_inference,
        "update_infer_json",
        lambda *args, **kwargs: pytest.fail("disabled MSA must not rewrite input"),
    )

    result = batch_inference.preprocess_input(
        str(input_path),
        str(tmp_path / "output"),
        use_msa=False,
        use_template=False,
        use_rna_msa=False,
    )

    assert result == str(input_path)


def test_preprocess_writes_derived_json_under_output_root(tmp_path, monkeypatch):
    from runner import batch_inference

    input_path = tmp_path / "input.json"
    output_root = tmp_path / "output"
    input_path.write_text('[{"name": "job"}]', encoding="utf-8")
    monkeypatch.setattr(
        batch_inference,
        "update_template_info",
        lambda *args, **kwargs: True,
    )

    result = batch_inference.preprocess_input(
        str(input_path),
        str(output_root),
        use_msa=False,
        use_template=True,
        use_rna_msa=False,
    )

    assert output_root.resolve() in Path(result).resolve().parents
    assert Path(result).is_file()


def test_precomputed_msa_does_not_skip_requested_template_search(tmp_path, monkeypatch):
    from runner import batch_inference

    input_path = tmp_path / "input.json"
    output_root = tmp_path / "output"
    input_path.write_text('[{"name": "job"}]', encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        batch_inference,
        "update_infer_json",
        lambda *args, **kwargs: (str(input_path), False),
    )

    def update_template_info(*args, **kwargs):
        calls.append((args, kwargs))
        return True

    monkeypatch.setattr(
        batch_inference,
        "update_template_info",
        update_template_info,
    )

    result = batch_inference.preprocess_input(
        str(input_path),
        str(output_root),
        use_msa=True,
        use_template=True,
        use_rna_msa=False,
    )

    assert len(calls) == 1
    assert Path(result).is_file()
