# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import hashlib
import importlib
from pathlib import Path

import pytest

from opendde.config.model_registry import DEFAULT_MODEL_NAME

DEFAULT_ASSET_REVISION = "eddd563ce96571f784012edd8f045181c8f8627d"
DEFAULT_DEPENDENCY_ROOT = (
    f"https://huggingface.co/aurekaresearch/OpenDDE/resolve/{DEFAULT_ASSET_REVISION}"
)
DEFAULT_COMMON_ROOT = f"{DEFAULT_DEPENDENCY_ROOT}/common"


class Config(dict):
    def __getattr__(self, key):
        return self[key]


def _inference_cache_config(
    tmp_path,
    *,
    load_checkpoint_path="",
    use_template=False,
):
    common_dir = tmp_path / "common"
    common_dir.mkdir()
    components_file = common_dir / "components.cif"
    rdkit_file = common_dir / "components.cif.rdkit_mol.pkl"
    components_file.write_text("data_components\n")
    rdkit_file.write_bytes(b"existing-rdkit-cache")

    return Config(
        data={
            "ccd_components_file": str(components_file),
            "ccd_components_rdkit_mol_file": str(rdkit_file),
            "template": {
                "obsolete_pdbs_path": str(common_dir / "obsolete_to_successor.json"),
                "release_dates_path": str(common_dir / "release_date_cache.json"),
            },
        },
        load_checkpoint_path=load_checkpoint_path,
        load_checkpoint_dir=str(tmp_path / "checkpoint"),
        model_name=DEFAULT_MODEL_NAME,
        use_template=use_template,
    )


def _reload_dependency_url(monkeypatch, dependency_root, common_root=None):
    import opendde.config.dependency_url as dependency_url_module

    if dependency_root is None:
        monkeypatch.delenv("OPENDDE_DEPENDENCY_URL", raising=False)
    else:
        monkeypatch.setenv("OPENDDE_DEPENDENCY_URL", dependency_root)

    if common_root is None:
        monkeypatch.delenv("OPENDDE_COMMON_URL", raising=False)
    else:
        monkeypatch.setenv("OPENDDE_COMMON_URL", common_root)

    return importlib.reload(dependency_url_module)


def test_dependency_urls_default_to_public_https(monkeypatch):
    dependency_url_module = _reload_dependency_url(monkeypatch, None)

    assert dependency_url_module.URL["ccd_components_file"] == (
        f"{DEFAULT_COMMON_ROOT}/components.cif"
    )
    assert dependency_url_module.URL["ccd_components_rdkit_mol_file"] == (
        f"{DEFAULT_COMMON_ROOT}/components.cif.rdkit_mol.pkl"
    )
    assert dependency_url_module.URL[DEFAULT_MODEL_NAME] == (
        f"{DEFAULT_DEPENDENCY_ROOT}/opendde.pt"
    )
    assert dependency_url_module.dependency_url("opendde_abag.pt") == (
        f"{DEFAULT_DEPENDENCY_ROOT}/opendde_abag.pt"
    )
    assert dependency_url_module.CHECKPOINT_FILES[DEFAULT_MODEL_NAME] == "opendde.pt"
    assert dependency_url_module.DEFAULT_ASSET_REVISION == DEFAULT_ASSET_REVISION


def test_default_checkpoint_path_uses_released_filename(tmp_path):
    from opendde.utils.download import resolve_checkpoint_path

    cfg = Config(
        load_checkpoint_path="",
        load_checkpoint_dir=str(tmp_path),
        model_name=DEFAULT_MODEL_NAME,
    )

    assert resolve_checkpoint_path(cfg) == str(tmp_path / "opendde.pt")


def test_dependency_url_supports_custom_https_root(monkeypatch):
    dependency_url_module = _reload_dependency_url(
        monkeypatch, "https://example.com/opendde/dependency/"
    )

    assert dependency_url_module.dependency_url("opendde.pt") == (
        "https://example.com/opendde/dependency/opendde.pt"
    )
    assert dependency_url_module.common_url("components.cif") == (
        "https://example.com/opendde/dependency/components.cif"
    )


def test_common_url_supports_independent_custom_root(monkeypatch):
    dependency_url_module = _reload_dependency_url(
        monkeypatch,
        "https://example.com/opendde/dependency/",
        "https://example.com/opendde/common/",
    )

    assert dependency_url_module.URL[DEFAULT_MODEL_NAME] == (
        "https://example.com/opendde/dependency/opendde.pt"
    )
    assert dependency_url_module.URL["ccd_components_file"] == (
        "https://example.com/opendde/common/components.cif"
    )


ALPHAFOLD_DB_ROOT = "https://storage.googleapis.com/alphafold-databases/v3.0"

_SEARCH_DB_ENV_VARS = (
    "OPENDDE_SEARCH_DATABASE_URL",
    "OPENDDE_PDB_SEQRES_URL",
    "OPENDDE_RFAM_DB_URL",
    "OPENDDE_NT_RNA_DB_URL",
    "OPENDDE_RNACENTRAL_DB_URL",
)


def _reload_with_search_db_env(monkeypatch, env):
    import opendde.config.dependency_url as dependency_url_module

    for key in _SEARCH_DB_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(dependency_url_module)


def test_search_database_urls_default_to_alphafold_v3_archives(monkeypatch):
    module = _reload_with_search_db_env(monkeypatch, {})

    assert module.SEARCH_DATABASE_URL == {
        "pdb_seqres": f"{ALPHAFOLD_DB_ROOT}/pdb_seqres_2022_09_28.fasta.zst",
        "rfam": (
            f"{ALPHAFOLD_DB_ROOT}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
        ),
        "nt_rna": (
            f"{ALPHAFOLD_DB_ROOT}/"
            "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
        ),
        "rnacentral": (
            f"{ALPHAFOLD_DB_ROOT}/rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst"
        ),
    }


def test_search_database_root_override_applies_to_all(monkeypatch):
    module = _reload_with_search_db_env(
        monkeypatch, {"OPENDDE_SEARCH_DATABASE_URL": "https://mirror.example.com/db/"}
    )

    assert module.SEARCH_DATABASE_URL["rfam"] == (
        "https://mirror.example.com/db/"
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst"
    )


def test_search_database_per_database_override_takes_priority(monkeypatch):
    module = _reload_with_search_db_env(
        monkeypatch,
        {
            "OPENDDE_SEARCH_DATABASE_URL": "https://mirror.example.com/db",
            "OPENDDE_PDB_SEQRES_URL": "https://my-s3.example.com/pdb_seqres.fasta",
        },
    )

    assert module.SEARCH_DATABASE_URL["pdb_seqres"] == (
        "https://my-s3.example.com/pdb_seqres.fasta"
    )
    assert module.SEARCH_DATABASE_URL["nt_rna"].startswith(
        "https://mirror.example.com/db/"
    )


def test_download_from_url_stages_before_atomic_replace(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    calls = []

    def fake_retrieve(url, filename):
        calls.append((url, filename))
        Path(filename).write_bytes(b"demo")

    monkeypatch.setattr(download_module, "_retrieve_url", fake_retrieve)

    output_path = tmp_path / "components.cif"
    download_module.download_from_url(
        "https://example.com/components.cif",
        str(output_path),
        check_weight=False,
    )

    assert output_path.read_bytes() == b"demo"
    assert len(calls) == 1
    assert calls[0][0] == "https://example.com/components.cif"
    staged_path = Path(calls[0][1])
    assert staged_path.parent == tmp_path
    assert staged_path != output_path
    assert not staged_path.exists()


def test_retrieve_url_retries_interrupted_transfer(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    calls = []

    class FakeResponse:
        headers = {"content-length": "8"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield b"complete"

    def flaky_get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise download_module.requests.ConnectionError("interrupted transfer")
        return FakeResponse()

    monkeypatch.setattr(download_module.requests, "get", flaky_get)

    output_path = tmp_path / "asset.bin"
    download_module._retrieve_url("https://example.com/asset.bin", str(output_path))

    assert len(calls) == 2
    assert calls[0][1] == {
        "stream": True,
        "timeout": download_module._DOWNLOAD_TIMEOUT,
    }
    assert output_path.read_bytes() == b"complete"


def test_download_from_url_decompresses_zst_to_requested_path(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    url = "https://example.com/db.fasta.zst"
    calls = []
    decompression_paths = []

    def fake_retrieve(url, filename):
        calls.append((url, filename))
        Path(filename).write_bytes(b"compressed")

    def fake_decompress_zst(zst_path, output_path, source_url):
        assert Path(zst_path).read_bytes() == b"compressed"
        assert source_url == url
        decompression_paths.append((zst_path, output_path))
        Path(output_path).write_text(">seq\nACGU\n")

    monkeypatch.setattr(download_module, "_retrieve_url", fake_retrieve)
    monkeypatch.setattr(download_module, "_decompress_zst", fake_decompress_zst)

    output_path = tmp_path / "db.fasta"
    download_module.download_from_url(url, str(output_path), check_weight=False)

    assert output_path.read_text() == ">seq\nACGU\n"
    assert len(calls) == 1
    assert calls[0][0] == url
    compressed_path = Path(calls[0][1])
    staged_path = Path(decompression_paths[0][1])
    assert compressed_path.parent == tmp_path
    assert staged_path.parent == tmp_path
    assert compressed_path != output_path
    assert staged_path != output_path
    assert compressed_path.suffix == ".zst"
    assert not compressed_path.exists()
    assert not staged_path.exists()


def test_download_validation_failure_preserves_existing_file(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    output_path = tmp_path / "opendde.pt"
    output_path.write_bytes(b"existing-checkpoint")
    staged_paths = []

    def fake_retrieve(url, filename):
        staged_paths.append(Path(filename))
        Path(filename).write_bytes(b"invalid-checkpoint")

    def fail_validation(path, **kwargs):
        assert Path(path) != output_path
        raise ValueError("invalid checkpoint")

    monkeypatch.setattr(download_module, "_retrieve_url", fake_retrieve)
    monkeypatch.setattr(download_module.torch, "load", fail_validation)

    with pytest.raises(RuntimeError, match="invalid checkpoint"):
        download_module.download_from_url(
            "https://example.com/opendde.pt", str(output_path)
        )

    assert output_path.read_bytes() == b"existing-checkpoint"
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()
    assert set(tmp_path.iterdir()) == {output_path}


def _asset_metadata(content):
    from opendde.config.dependency_url import ManagedAsset

    return ManagedAsset(
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def test_ensure_managed_asset_keeps_matching_published_size(monkeypatch, tmp_path):
    import opendde.utils.download as download_module

    content = b"published-asset"
    destination = tmp_path / "asset.bin"
    destination.write_bytes(content)
    monkeypatch.setitem(
        download_module.MANAGED_ASSETS,
        "test_asset",
        _asset_metadata(content),
    )
    monkeypatch.setitem(
        download_module.URL,
        "test_asset",
        "https://example.com/asset.bin",
    )
    monkeypatch.setattr(
        download_module,
        "_retrieve_url",
        lambda *args, **kwargs: pytest.fail("matching asset must not download"),
    )

    download_module._ensure_managed_asset("test_asset", str(destination))

    assert destination.read_bytes() == content


@pytest.mark.parametrize(
    "asset_name",
    [
        DEFAULT_MODEL_NAME,
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
        "obsolete_pdbs_path",
        "release_dates_path",
    ],
)
def test_ensure_managed_asset_replaces_truncated_release_asset(
    monkeypatch,
    tmp_path,
    asset_name,
):
    import opendde.utils.download as download_module

    complete = f"complete-{asset_name}".encode()
    source_url = download_module.URL[asset_name]
    destination = tmp_path / Path(source_url).name
    destination.write_bytes(complete[:-1])
    monkeypatch.setitem(
        download_module.MANAGED_ASSETS,
        asset_name,
        _asset_metadata(complete),
    )

    def fake_retrieve(url, filename):
        assert url == source_url
        Path(filename).write_bytes(complete)

    monkeypatch.setattr(download_module, "_retrieve_url", fake_retrieve)

    download_module._ensure_managed_asset(asset_name, str(destination))

    assert destination.read_bytes() == complete


def test_ensure_managed_asset_preserves_existing_file_when_repair_fails(
    monkeypatch,
    tmp_path,
):
    import opendde.utils.download as download_module

    destination = tmp_path / "asset.bin"
    destination.write_bytes(b"old")
    expected = b"expected"
    monkeypatch.setitem(
        download_module.MANAGED_ASSETS,
        "test_asset",
        _asset_metadata(expected),
    )
    monkeypatch.setitem(
        download_module.URL,
        "test_asset",
        "https://example.com/asset.bin",
    )

    def fake_retrieve(url, filename):
        Path(filename).write_bytes(b"incorrect")

    monkeypatch.setattr(download_module, "_retrieve_url", fake_retrieve)

    with pytest.raises(RuntimeError, match="failed validation"):
        download_module._ensure_managed_asset("test_asset", str(destination))

    assert destination.read_bytes() == b"old"
    assert set(tmp_path.iterdir()) == {destination}


def test_download_inference_cache_routes_all_release_assets_through_one_ensurer(
    monkeypatch,
    tmp_path,
):
    import opendde.utils.download as download_module

    configs = _inference_cache_config(tmp_path, use_template=True)
    calls = []

    monkeypatch.setattr(
        download_module,
        "_ensure_managed_asset",
        lambda name, path: calls.append((name, path)),
    )

    download_module.download_inference_cache(configs)

    assert calls == [
        ("ccd_components_file", configs.data["ccd_components_file"]),
        (
            "ccd_components_rdkit_mol_file",
            configs.data["ccd_components_rdkit_mol_file"],
        ),
        ("obsolete_pdbs_path", configs.data["template"]["obsolete_pdbs_path"]),
        ("release_dates_path", configs.data["template"]["release_dates_path"]),
        (
            DEFAULT_MODEL_NAME,
            str(Path(configs.load_checkpoint_dir) / "opendde.pt"),
        ),
    ]


def test_download_inference_cache_does_not_replace_explicit_checkpoint(
    monkeypatch, tmp_path
):
    import opendde.utils.download as download_module

    checkpoint_path = tmp_path / "custom.pt"
    checkpoint_path.write_bytes(b"PK\x03\x04truncated-custom-checkpoint")
    original_checkpoint = checkpoint_path.read_bytes()
    configs = _inference_cache_config(
        tmp_path, load_checkpoint_path=str(checkpoint_path)
    )
    monkeypatch.setattr(
        download_module,
        "_ensure_managed_asset",
        lambda *args, **kwargs: None,
    )

    download_module.download_inference_cache(configs)

    assert checkpoint_path.read_bytes() == original_checkpoint


def test_download_inference_cache_rejects_missing_explicit_checkpoint(
    monkeypatch, tmp_path
):
    import opendde.utils.download as download_module

    checkpoint_path = tmp_path / "missing-custom.pt"
    configs = _inference_cache_config(
        tmp_path, load_checkpoint_path=str(checkpoint_path)
    )
    monkeypatch.setattr(
        download_module,
        "_ensure_managed_asset",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(FileNotFoundError, match="Given checkpoint path not exist"):
        download_module.download_inference_cache(configs)
