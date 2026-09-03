# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
import posixpath
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from opendde.config.model_registry import DEFAULT_MODEL_NAME
from opendde.config.model_manifest import (
    get_checkpoint_manifest,
    get_model_manifest,
    load_model_manifest,
    source_root,
)

MODEL_MANIFEST = load_model_manifest()
DEFAULT_MODEL_MANIFEST = get_model_manifest(DEFAULT_MODEL_NAME, MODEL_MANIFEST)
DEFAULT_CHECKPOINT_FILE = DEFAULT_MODEL_MANIFEST["default_checkpoint"]
DEFAULT_CHECKPOINT_MANIFEST = get_checkpoint_manifest(
    DEFAULT_MODEL_MANIFEST, DEFAULT_CHECKPOINT_FILE
)
CHECKPOINT_FILES = {
    model["name"]: model["default_checkpoint"] for model in MODEL_MANIFEST["models"]
}
DEFAULT_ASSET_REVISION = MODEL_MANIFEST["source"]["revision"]
DEFAULT_DEPENDENCY_URL_ROOT = source_root(MODEL_MANIFEST)


@dataclass(frozen=True)
class ManagedAsset:
    """Published runtime asset identity used for repair and download validation."""

    size: int
    sha256: str


MANAGED_ASSETS = {
    DEFAULT_MODEL_NAME: ManagedAsset(
        size=DEFAULT_CHECKPOINT_MANIFEST["size_bytes"],
        sha256=DEFAULT_CHECKPOINT_MANIFEST["sha256"],
    ),
    "ccd_components_file": ManagedAsset(
        size=490_777_362,
        sha256="bb31ae5cf6c8bc669924313077cb4231ee5ffefd3a20118cd14f3ec89f8bb6a5",
    ),
    "ccd_components_rdkit_mol_file": ManagedAsset(
        size=142_498_117,
        sha256="d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35",
    ),
    "obsolete_pdbs_path": ManagedAsset(
        size=86_882,
        sha256="2bc08348d0efba438c109bb27be6fa25b611d371c60b8a8da3de387a4a0698ad",
    ),
    "release_dates_path": ManagedAsset(
        size=12_754_898,
        sha256="8b1ef12ddc01a0d5eb2d388c77ded91aa906eebce7440726c57b6f8d1a3ec142",
    ),
}

DEPENDENCY_URL_ROOT = os.environ.get(
    "OPENDDE_DEPENDENCY_URL",
    DEFAULT_DEPENDENCY_URL_ROOT,
).rstrip("/")

COMMON_URL_ROOT = os.environ.get(
    "OPENDDE_COMMON_URL",
    os.environ.get(
        "OPENDDE_DEPENDENCY_URL",
        f"{DEFAULT_DEPENDENCY_URL_ROOT}/common",
    ),
).rstrip("/")


def _root_url(root: str, *parts: str) -> str:
    parsed = urlsplit(root)
    clean_parts = [part.strip("/") for part in parts if part]
    if parsed.scheme and parsed.netloc:
        path = posixpath.join(parsed.path.rstrip("/"), *clean_parts)
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return os.path.join(root, *clean_parts)


def dependency_url(*parts: str) -> str:
    return _root_url(DEPENDENCY_URL_ROOT, *parts)


def common_url(*parts: str) -> str:
    return _root_url(COMMON_URL_ROOT, *parts)


URL = {
    DEFAULT_MODEL_NAME: dependency_url(CHECKPOINT_FILES[DEFAULT_MODEL_NAME]),
    "ccd_components_file": common_url("components.cif"),
    "ccd_components_rdkit_mol_file": common_url("components.cif.rdkit_mol.pkl"),
    # the following files will be used if enable_template is True
    "obsolete_pdbs_path": common_url("obsolete_to_successor.json"),
    "release_dates_path": common_url("release_date_cache.json"),
}


# Sequence databases for local MSA/template search (hmmsearch / nhmmer).
#
# These default to the AlphaFold database v3.0 archives so that local
# protein-template and RNA-MSA search work out of the box. The small databases
# (pdb_seqres ~220MB and rfam ~220MB) can be re-hosted by setting the
# per-database env vars below; the large databases (nt_rna ~75GB,
# rnacentral ~13GB) are best left on a reliable mirror.
# Set OPENDDE_SEARCH_DATABASE_URL to relocate all four under a single root.
SEARCH_DATABASE_URL_ROOT = os.environ.get(
    "OPENDDE_SEARCH_DATABASE_URL",
    "https://storage.googleapis.com/alphafold-databases/v3.0",
).rstrip("/")


def search_database_url(filename: str, env_var: str = "") -> str:
    """Resolve a search-database download URL, honoring a per-database override."""
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override
    return f"{SEARCH_DATABASE_URL_ROOT}/{filename}"


SEARCH_DATABASE_URL = {
    "pdb_seqres": search_database_url(
        "pdb_seqres_2022_09_28.fasta.zst", "OPENDDE_PDB_SEQRES_URL"
    ),
    "rfam": search_database_url(
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        "OPENDDE_RFAM_DB_URL",
    ),
    "nt_rna": search_database_url(
        "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        "OPENDDE_NT_RNA_DB_URL",
    ),
    "rnacentral": search_database_url(
        "rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst",
        "OPENDDE_RNACENTRAL_DB_URL",
    ),
}
