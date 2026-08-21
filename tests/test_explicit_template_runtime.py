# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import json
from types import SimpleNamespace

import pytest
from ml_collections.config_dict import ConfigDict

from opendde.config.inference import build_inference_config
from opendde.data.inference import infer_dataloader
from runner import batch_inference, inference, template_search


def _protein_chain(template_mode, tmp_path):
    chain = {"sequence": "ACDEFG", "count": 1}
    if template_mode == "explicit":
        chain["templates"] = [
            {
                "mmcif": "data_template\n#\n",
                "queryIndices": [0],
                "templateIndices": [0],
            }
        ]
    elif template_mode == "disabled":
        chain["templates"] = []
    elif template_mode == "search_hits":
        hit_path = tmp_path / "hits.a3m"
        hit_path.write_text(">query\nACDEFG\n", encoding="utf-8")
        chain["templatesPath"] = str(hit_path)
    return chain


def _write_input(tmp_path, template_mode):
    input_path = tmp_path / f"{template_mode}.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "name": template_mode,
                    "sequences": [
                        {"proteinChain": _protein_chain(template_mode, tmp_path)}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    return input_path


def _dataset_configs(tmp_path, template_mode):
    input_path = _write_input(tmp_path, template_mode)
    template_dir = tmp_path / "mmcif"
    if template_mode in {"auto", "search_hits"}:
        template_dir.mkdir()

    configs = ConfigDict()
    configs.input_json_path = str(input_path)
    configs.dump_dir = str(tmp_path / "output")
    configs.use_msa = False
    configs.use_rna_msa = False
    configs.use_template = True
    configs.data = ConfigDict()
    configs.data.ccd_components_file = str(tmp_path / "components.cif")
    configs.data.ccd_components_rdkit_mol_file = str(tmp_path / "components.pkl")
    configs.data.template = ConfigDict()
    configs.data.template.prot_template_mmcif_dir = str(template_dir)
    configs.data.template.prot_template_cache_dir = str(tmp_path / "template-cache")
    configs.data.template.kalign_binary_path = "kalign"
    configs.data.template.release_dates_path = ""
    configs.data.template.obsolete_pdbs_path = ""
    configs.data.template.fetch_remote = False
    return configs, template_dir


@pytest.mark.parametrize(
    ("template_mode", "expects_search"),
    [
        ("explicit", False),
        ("auto", True),
    ],
)
def test_dataset_constructs_search_processor_only_when_needed(
    tmp_path, monkeypatch, template_mode, expects_search
):
    configs, template_dir = _dataset_configs(tmp_path, template_mode)
    search_processor = object()
    monkeypatch.setattr(
        infer_dataloader,
        "TemplateHitFeaturizer",
        lambda **_kwargs: search_processor,
    )

    dataset = infer_dataloader.InferenceDataset(configs)

    assert (dataset.online_template_featurizer is search_processor) is expects_search
    if not expects_search:
        assert not template_dir.exists()


def test_template_preprocessing_skips_explicit_and_disabled_but_regenerates_missing_hit(
    tmp_path, monkeypatch
):
    msa_dir = tmp_path / "msa"
    msa_dir.mkdir()
    paired_path = msa_dir / "pairing.a3m"
    paired_path.write_text(">query\nACDEFG\n", encoding="utf-8")
    existing_hits = tmp_path / "existing-hits.a3m"
    existing_hits.write_text(">query\nACDEFG\n", encoding="utf-8")
    searches = []

    def run_search(**kwargs):
        searches.append(kwargs)
        (msa_dir / "hmmsearch.a3m").write_text(">query\nACDEFG\n", encoding="utf-8")

    monkeypatch.setattr(template_search, "run_template_search", run_search)
    explicit_templates = [
        {
            "mmcif": "data_template\n#\n",
            "queryIndices": [0],
            "templateIndices": [0],
        }
    ]
    payload = [
        {
            "name": "mixed",
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "ACDEFG",
                        "count": 1,
                        "pairedMsaPath": str(paired_path),
                        "templates": explicit_templates,
                    }
                },
                {
                    "proteinChain": {
                        "sequence": "ACDEFG",
                        "count": 1,
                        "pairedMsaPath": str(paired_path),
                        "templates": [],
                    }
                },
                {
                    "proteinChain": {
                        "sequence": "ACDEFG",
                        "count": 1,
                        "pairedMsaPath": str(paired_path),
                        "templatesPath": str(tmp_path / "missing-hits.a3m"),
                    }
                },
                {
                    "proteinChain": {
                        "sequence": "ACDEFG",
                        "count": 1,
                        "pairedMsaPath": str(paired_path),
                        "templatesPath": str(existing_hits),
                    }
                },
            ],
        }
    ]

    assert template_search.update_template_info(payload)

    explicit, disabled, missing_hit, existing_hit = [
        item["proteinChain"] for item in payload[0]["sequences"]
    ]
    assert explicit["templates"] == explicit_templates
    assert "templatesPath" not in explicit
    assert disabled["templates"] == []
    assert "templatesPath" not in disabled
    assert missing_hit["templatesPath"] == str(msa_dir / "hmmsearch.a3m")
    assert existing_hit["templatesPath"] == str(existing_hits)
    assert len(searches) == 1


def _patch_runner_startup(monkeypatch, metadata_requests):
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(
            world_size=1,
            rank=0,
            local_rank=0,
        ),
    )
    monkeypatch.setattr(
        inference, "FoldCPBenchmarkRecorder", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(inference.InferenceRunner, "init_env", lambda self: None)
    for method_name in ("init_basics", "init_model", "load_checkpoint"):
        monkeypatch.setattr(
            inference.InferenceRunner,
            method_name,
            lambda self: None,
        )
    monkeypatch.setattr(
        inference.InferenceRunner,
        "init_dumper",
        lambda self, **_kwargs: None,
    )
    monkeypatch.setattr(
        inference,
        "_download_inference_assets",
        lambda configs: metadata_requests.append(configs.use_template),
    )


@pytest.mark.parametrize(
    ("template_mode", "needs_search"),
    [
        ("explicit", False),
        ("auto", True),
    ],
)
def test_public_batch_boundary_gates_actual_kalign_resolver_and_search_metadata(
    tmp_path, monkeypatch, template_mode, needs_search
):
    input_path = _write_input(tmp_path, template_mode)
    metadata_requests = []
    kalign_requests = []
    _patch_runner_startup(monkeypatch, metadata_requests)
    monkeypatch.setattr(
        inference.kalign,
        "resolve_kalign_binary",
        lambda path: kalign_requests.append(path) or "/resolved/kalign",
    )
    monkeypatch.setattr(
        batch_inference, "preprocess_input", lambda path, **_kwargs: path
    )
    monkeypatch.setattr(batch_inference, "infer_predict", lambda *_args: None)

    batch_inference.inference_jsons(
        str(input_path),
        use_msa=False,
        use_template=True,
        kalign_binary_path="configured-kalign",
        device="cpu",
    )

    assert kalign_requests == (["configured-kalign"] if needs_search else [])
    assert metadata_requests == [needs_search]


def test_asset_preparation_handles_direct_mixed_and_unreadable_inputs(
    tmp_path, monkeypatch
):
    downloads = []
    kalign_requests = []
    configs = build_inference_config(fill_required_with_null=True)
    configs.use_template = True
    explicit_path = _write_input(tmp_path, "disabled")
    automatic_path = _write_input(tmp_path, "auto")
    configs.input_json_path = str(explicit_path)
    monkeypatch.setattr(
        inference,
        "DIST_WRAPPER",
        SimpleNamespace(world_size=1, rank=0, local_rank=0),
    )
    monkeypatch.setattr(
        inference,
        "download_inference_cache",
        lambda asset_configs: downloads.append(asset_configs.use_template),
    )
    monkeypatch.setattr(
        inference.kalign,
        "resolve_kalign_binary",
        lambda path: kalign_requests.append(path) or "/resolved/kalign",
    )

    inference._prepare_template_dependencies(configs, None)
    inference._prepare_template_dependencies(
        configs, [str(explicit_path), str(automatic_path)]
    )
    inference._prepare_template_dependencies(configs, [str(tmp_path / "missing.json")])

    assert downloads == [False, True, True]
    assert len(kalign_requests) == 2
    assert configs.use_template is True
