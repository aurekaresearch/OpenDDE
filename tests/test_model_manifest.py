# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from opendde import __version__
from opendde.config.dependency_url import (
    DEFAULT_ASSET_REVISION,
    MANAGED_ASSETS,
)
from opendde.config.model_manifest import (
    checkpoint_url,
    find_checkpoint_manifest,
    get_checkpoint_manifest,
    get_default_model_manifest,
    get_model_manifest,
    load_model_manifest,
    verify_checkpoint_file,
)
from opendde.config.model_registry import DEFAULT_MODEL_NAME

ROOT = Path(__file__).resolve().parents[1]


def _checkpoint_manifest(
    filename: str, contents: bytes, model_name: str = "test_model"
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": {"name": "opendde", "version": __version__},
        "source": {
            "repository": "https://example.test/opendde",
            "revision": "a" * 40,
            "verified_at": "2026-07-16",
        },
        "default_model": model_name,
        "models": [
            {
                "name": model_name,
                "status": "supported",
                "parameter_count": 1,
                "data_cutoff": "2026-01-01",
                "compatible_package_versions": ">=1.0.0,<2.0.0",
                "default_checkpoint": filename,
                "checkpoints": [
                    {
                        "filename": filename,
                        "variant": "test",
                        "size_bytes": len(contents),
                        "sha256": hashlib.sha256(contents).hexdigest(),
                    }
                ],
            }
        ],
    }


def _write_checkpoint_manifest(
    tmp_path: Path, filename: str, contents: bytes, model_name: str = "test_model"
) -> Path:
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(
        json.dumps(_checkpoint_manifest(filename, contents, model_name)),
        encoding="utf-8",
    )
    return manifest_path


def test_model_manifest_matches_package_and_runtime_assets():
    manifest = load_model_manifest()
    model = get_model_manifest(DEFAULT_MODEL_NAME, manifest)
    checkpoint = get_checkpoint_manifest(model, model["default_checkpoint"])

    assert manifest["package"] == {"name": "opendde", "version": __version__}
    assert manifest["default_model"] == DEFAULT_MODEL_NAME
    assert get_default_model_manifest(manifest) is model
    assert manifest["source"]["revision"] == DEFAULT_ASSET_REVISION
    assert {item["filename"] for item in model["checkpoints"]} == {
        "opendde.pt",
        "opendde_abag.pt",
    }
    assert model["default_checkpoint"] == "opendde.pt"
    assert MANAGED_ASSETS[DEFAULT_MODEL_NAME].size == checkpoint["size_bytes"]
    assert MANAGED_ASSETS[DEFAULT_MODEL_NAME].sha256 == checkpoint["sha256"]
    assert checkpoint_url("opendde.pt", manifest) == (
        f"{manifest['source']['repository']}/resolve/"
        f"{manifest['source']['revision']}/opendde.pt"
    )


def test_model_manifest_accessors_reject_unknown_records():
    manifest = load_model_manifest()
    model = get_model_manifest(DEFAULT_MODEL_NAME, manifest)

    with pytest.raises(KeyError, match="Unknown released model"):
        get_model_manifest("unknown_model", manifest)
    with pytest.raises(KeyError, match="Unknown checkpoint"):
        get_checkpoint_manifest(model, "unknown_checkpoint.pt")
    with pytest.raises(KeyError, match="Unknown released checkpoint"):
        find_checkpoint_manifest("unknown_checkpoint.pt", manifest)


def test_checkpoint_url_enforces_model_ownership():
    manifest = _checkpoint_manifest("first.pt", b"first", "first_model")
    second_manifest = _checkpoint_manifest("second.pt", b"second", "second_model")
    manifest["models"].append(second_manifest["models"][0])

    assert checkpoint_url("first.pt", manifest, model_name="first_model").endswith(
        "/first.pt"
    )
    with pytest.raises(KeyError, match="Unknown checkpoint"):
        checkpoint_url("second.pt", manifest, model_name="first_model")


def test_checkpoint_verifier_checks_size_and_sha256(tmp_path):
    checkpoint = tmp_path / "download.tmp"
    checkpoint.write_bytes(b"released checkpoint")
    manifest = _checkpoint_manifest("released.pt", checkpoint.read_bytes())
    expected = manifest["models"][0]["checkpoints"][0]

    assert (
        verify_checkpoint_file(checkpoint, "released.pt", manifest)["sha256"]
        == expected["sha256"]
    )

    checkpoint.write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_checkpoint_file(checkpoint, "released.pt", manifest)

    checkpoint.write_bytes(b"x" * expected["size_bytes"])
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint_file(checkpoint, "released.pt", manifest)


def test_checkpoint_verifier_cli_accepts_an_explicit_manifest(tmp_path):
    checkpoint = tmp_path / "checkpoint.tmp"
    checkpoint.write_bytes(b"small test checkpoint")
    manifest_path = _write_checkpoint_manifest(
        tmp_path, "released.pt", checkpoint.read_bytes()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opendde.config.model_manifest",
            "verify-checkpoint",
            str(checkpoint),
            "--checkpoint",
            "released.pt",
            "--manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Verified released.pt" in result.stdout


def test_manifest_cli_resolves_custom_default_without_shell_constants(tmp_path):
    manifest_path = _write_checkpoint_manifest(
        tmp_path, "future_release.pt", b"future checkpoint", "future_model"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "opendde" / "config" / "model_manifest.py"),
            "resolve-model",
            "--manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\t")[:2] == ["future_model", "future_release.pt"]
    assert result.stdout.rstrip().endswith("/resolve/" + "a" * 40)


def test_download_script_verifies_and_repairs_released_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "future_release.pt"
    source.write_bytes(b"small released checkpoint")
    manifest_path = _write_checkpoint_manifest(
        tmp_path, source.name, source.read_bytes(), model_name="opendde_v1"
    )
    root = tmp_path / "data"
    command = [
        "bash",
        str(ROOT / "scripts" / "download_opendde_data.sh"),
        "--root",
        str(root),
        "--skip-common",
        "--skip-search-database",
        "--dependency-url",
        source_dir.as_uri(),
        "--checkpoint",
        "future_release.pt",
        "--model-manifest",
        str(manifest_path),
    ]
    env = {"PATH": os.environ["PATH"]}

    first = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout.count("Verified future_release.pt") == 1
    installed = root / "checkpoint" / "future_release.pt"
    assert installed.read_bytes() == source.read_bytes()

    installed.write_bytes(b"x" * installed.stat().st_size)
    repaired = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repaired.returncode == 0, repaired.stderr
    assert repaired.stdout.count("Verified future_release.pt") == 1
    assert "failed verification" in repaired.stdout
    assert "SHA-256 mismatch" in repaired.stdout
    assert "replacing it" in repaired.stdout
    assert installed.read_bytes() == source.read_bytes()


def test_download_script_all_skip_does_not_require_manifest_or_python(tmp_path):
    root = tmp_path / "data"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "download_opendde_data.sh"),
            "--root",
            str(root),
            "--skip-common",
            "--skip-search-database",
            "--skip-model",
            "--model-manifest",
            str(tmp_path / "missing-manifest.json"),
        ],
        cwd=ROOT,
        env={
            "PATH": os.environ["PATH"],
            "OPENDDE_PYTHON": str(tmp_path / "missing-python"),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "not changed; model download skipped" in result.stdout


def test_download_script_preserves_target_when_copy_fails(tmp_path):
    source = tmp_path / "future_release.pt"
    source.write_bytes(b"new checkpoint")
    manifest_path = _write_checkpoint_manifest(
        tmp_path, source.name, source.read_bytes(), model_name="opendde_v1"
    )
    root = tmp_path / "data"
    target = root / "checkpoint" / source.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing checkpoint")

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_cp = fake_bin / "cp"
    fake_cp.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_cp.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "download_opendde_data.sh"),
            "--root",
            str(root),
            "--skip-common",
            "--skip-search-database",
            "--model-source",
            source.as_uri(),
            "--checkpoint",
            source.name,
            "--model-manifest",
            str(manifest_path),
            "--force",
        ],
        cwd=ROOT,
        env={
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "OPENDDE_PYTHON": sys.executable,
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Failed to copy/download" in result.stderr
    assert target.read_bytes() == b"existing checkpoint"
    assert not list(root.rglob("*.tmp.*"))
    assert not list(root.rglob("*.download.*"))


def test_download_script_verifies_official_filename_for_custom_model(tmp_path):
    released_contents = b"released checkpoint"
    source = tmp_path / "future_release.pt"
    source.write_bytes(b"tampered")
    manifest_path = _write_checkpoint_manifest(
        tmp_path,
        source.name,
        released_contents,
        model_name="opendde_v1",
    )
    root = tmp_path / "data"

    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "download_opendde_data.sh"),
            "--root",
            str(root),
            "--skip-common",
            "--skip-search-database",
            "--model-name",
            "custom_model",
            "--model-source",
            source.as_uri(),
            "--model-manifest",
            str(manifest_path),
        ],
        cwd=ROOT,
        env={"PATH": os.environ["PATH"], "OPENDDE_PYTHON": sys.executable},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "size mismatch" in result.stderr
    assert not (root / "checkpoint" / source.name).exists()


def test_concurrent_zst_downloads_use_unique_temporary_files(tmp_path):
    filenames = (
        "pdb_seqres_2022_09_28.fasta.zst",
        "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        "rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst",
    )
    source_dir = tmp_path / "search-source"
    source_dir.mkdir()
    for filename in filenames:
        (source_dir / filename).write_bytes(filename.encode())

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_zstd = fake_bin / "zstd"
    fake_zstd.write_text(
        """#!/bin/sh
output=""
input=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -o) output="$2"; shift 2 ;;
        -d|-f) shift ;;
        *) input="$1"; shift ;;
    esac
done
sleep 0.1
/bin/cp "$input" "$output"
""",
        encoding="utf-8",
    )
    fake_zstd.chmod(0o755)

    root = tmp_path / "data"
    command = [
        "bash",
        str(ROOT / "scripts" / "download_opendde_data.sh"),
        "--root",
        str(root),
        "--skip-common",
        "--skip-model",
        "--search-database-url",
        source_dir.as_uri(),
        "--force",
    ]
    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "OPENDDE_PYTHON": str(tmp_path / "missing-python"),
    }
    processes = [
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]

    results = [process.communicate(timeout=15) for process in processes]

    for process, (stdout, stderr) in zip(processes, results, strict=True):
        assert process.returncode == 0, f"{stdout}\n{stderr}"
    for filename in filenames:
        target = root / "search_database" / filename.removesuffix(".zst")
        assert target.read_bytes() == filename.encode()
    assert not list(root.rglob("*.tmp.*"))
    assert not list(root.rglob("*.download.*"))
