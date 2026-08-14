# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import importlib
import json
import shutil
from pathlib import Path


def test_generate_infer_jsons_keeps_non_sdf_ligand_file_intact(tmp_path, monkeypatch):
    module = importlib.import_module("runner.batch_inference")
    ligand_file = tmp_path / "lig.mol"
    ligand_file.write_text("mock mol")
    monkeypatch.setattr(module, "lig_file_to_atom_info", lambda path: {"path": path})

    outputs = module.generate_infer_jsons(
        {"ACDE": {"count": 1}},
        str(ligand_file),
    )

    try:
        assert len(outputs) == 1
        with open(outputs[0], "r") as f:
            payload = json.load(f)

        ligand_entries = [
            entity for entity in payload[0]["sequences"] if "ligand" in entity
        ]
        assert len(ligand_entries) == 1
        assert ligand_entries[0]["ligand"]["ligand"] == f"FILE_{ligand_file}"
        assert payload[0]["name"] == ligand_file.stem
    finally:
        json_dir = Path(outputs[0]).parent if outputs else None
        if json_dir is not None:
            scratch_dir = json_dir.with_name(json_dir.name.removesuffix("_jsons"))
            shutil.rmtree(json_dir, ignore_errors=True)
            shutil.rmtree(scratch_dir, ignore_errors=True)
