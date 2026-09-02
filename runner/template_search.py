# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import hashlib
import os
import pathlib
import shutil
import time
from typing import Any, Optional, Sequence

from opendde.config.data import default_root_dir
from opendde.config.dependency_url import SEARCH_DATABASE_URL
from opendde.data.tools.search import HmmsearchConfig, run_hmmsearch_with_a3m
from opendde.utils.download import download_from_url
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

TEMPLATE_SEARCH_DATABASE_URL = SEARCH_DATABASE_URL["pdb_seqres"]


def ensure_ends_with_newline(s: str) -> str:
    """
    Ensure the string ends with a newline character.

    Args:
        s: The input string.

    Returns:
        The string with a trailing newline if it wasn't empty.
    """
    if not s.endswith("\n") and (len(s) > 0):
        s += "\n"
    return s


def _resolve_executable(requested: Optional[str], default_name: str) -> str:
    """Resolve an executable path or PATH-visible command name."""

    candidate = requested or default_name
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find {candidate!r} as an executable path or on PATH. "
            "Install HMMER with `apt install hmmer` or "
            "`conda install -c bioconda hmmer`."
        )
    return resolved


def run_template_search(
    msa_for_template_search_dir: Optional[str] = None,
    msa_for_template_search_name: Optional[str] = None,
    msa_for_template_search_paths: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
) -> None:
    """
    Run template search using hmmsearch with a3m files.

    Args:
        msa_for_template_search_dir: Directory containing MSA files.
            Templates will be saved in the same directory.
        msa_for_template_search_name: Comma-separated names of MSA files to search.
        msa_for_template_search_paths: Exact MSA paths. When supplied, filenames
            do not need to follow the pairing/non_pairing convention.
        output_path: Exact template output path. Defaults to hmmsearch.a3m in
            ``msa_for_template_search_dir`` for backward compatibility.
        hmmsearch_binary_path: Path to hmmsearch binary.
        hmmbuild_binary_path: Path to hmmbuild binary.
        seqres_database_path: Path to sequence database.
    """
    # msa_for_template_search_dir contains the paired/unpaired MSA files, used for template search
    assert msa_for_template_search_dir is not None, "input msa dir should not be None"

    if msa_for_template_search_paths is None:
        # Legacy entry point: names are stems relative to the output directory.
        assert msa_for_template_search_name is not None, (
            "input msa name should not be None"
        )
        msa_paths = [
            os.path.join(msa_for_template_search_dir, f"{name}.a3m")
            for name in msa_for_template_search_name.split(",")
        ]
    else:
        msa_paths = list(
            dict.fromkeys(os.fspath(path) for path in msa_for_template_search_paths)
        )
        if not msa_paths:
            raise ValueError("At least one MSA path is required for template search.")
        missing_paths = [path for path in msa_paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(
                "Template-search MSA files do not exist: " + ", ".join(missing_paths)
            )

    hmmsearch_binary_path = _resolve_executable(
        hmmsearch_binary_path,
        "hmmsearch",
    )
    hmmbuild_binary_path = _resolve_executable(
        hmmbuild_binary_path,
        "hmmbuild",
    )

    if seqres_database_path is None:
        _HOME_DIR = pathlib.Path(os.environ.get("OPENDDE_ROOT_DIR", default_root_dir()))
        _SEQRES_DATABASE_PATH = (
            _HOME_DIR / "search_database" / "pdb_seqres_2022_09_28.fasta"
        )
        seqres_database_path = _SEQRES_DATABASE_PATH.as_posix()
    if not os.path.exists(seqres_database_path):
        os.makedirs(os.path.dirname(seqres_database_path), exist_ok=True)
        logger.info(
            f"Downloading template search database from {TEMPLATE_SEARCH_DATABASE_URL} to {seqres_database_path}"
        )
        download_from_url(
            TEMPLATE_SEARCH_DATABASE_URL, seqres_database_path, check_weight=False
        )

    logger.info("Template search start!")
    template_start_time = time.time()
    hmmsearch_config = HmmsearchConfig(
        hmmsearch_binary_path=hmmsearch_binary_path,
        hmmbuild_binary_path=hmmbuild_binary_path,
        filter_f1=0.1,
        filter_f2=0.1,
        filter_f3=0.1,
        e_value=100,
        inc_e=100,
        dom_e=100,
        incdom_e=100,
        alphabet="amino",
    )
    max_a3m_query_sequences = 300
    msa_a3m = ""
    for unpaired_msa_path in msa_paths:
        logger.info(f"msa path: {unpaired_msa_path}")
        if os.path.exists(unpaired_msa_path):
            with open(unpaired_msa_path, "r") as f:
                unpaired_msa_a3m = f.read()
        else:
            unpaired_msa_a3m = ""

        unpaired_msa_a3m = ensure_ends_with_newline(unpaired_msa_a3m)
        msa_a3m = msa_a3m + unpaired_msa_a3m
    msa_a3m = ensure_ends_with_newline(msa_a3m)
    hmmsearch_a3m = run_hmmsearch_with_a3m(
        database_path=seqres_database_path,
        hmmsearch_config=hmmsearch_config,
        max_a3m_query_sequences=max_a3m_query_sequences,
        a3m=msa_a3m,
    )

    if output_path is None:
        output_path = os.path.join(msa_for_template_search_dir, "hmmsearch.a3m")
    with open(output_path, "w") as f:
        f.write(hmmsearch_a3m)
    template_end_time = time.time()
    logger.info(
        f"Template search done!, using {template_end_time - template_start_time}"
    )
    logger.info(f"Template result is saved at: {output_path}")


def update_template_info(
    json_data: list[dict[str, Any]],
    hmmsearch_binary_path: Optional[str] = None,
    hmmbuild_binary_path: Optional[str] = None,
    seqres_database_path: Optional[str] = None,
) -> bool:
    """
    Update template information in the JSON data.
    If templatesPath is missing, it performs a template search.

    Args:
        json_data (list[dict[str, Any]]): The input JSON data.
        hmmsearch_binary_path (Optional[str]): Path to hmmsearch binary.
        hmmbuild_binary_path (Optional[str]): Path to hmmbuild binary.
        seqres_database_path (Optional[str]): Path to sequence database.

    Returns:
        bool: True if any template information was updated.
    """
    actual_updated = False
    for task_idx, infer_data in enumerate(json_data):
        task_name = infer_data.get("name", f"task_{task_idx}")
        for sequence in infer_data["sequences"]:
            if "proteinChain" in sequence:
                protein_chain = sequence["proteinChain"]
                # Skip if templatesPath already exists and is valid
                if "templatesPath" in protein_chain and os.path.exists(
                    protein_chain["templatesPath"]
                ):
                    continue

                # Get MSA path to perform template search
                paired_msa_path = protein_chain.get("pairedMsaPath")
                unpaired_msa_path = protein_chain.get("unpairedMsaPath")
                msa_paths = list(
                    dict.fromkeys(
                        path
                        for path in (paired_msa_path, unpaired_msa_path)
                        if path and os.path.isfile(path)
                    )
                )
                if not msa_paths:
                    raise FileNotFoundError(
                        f"Template search for task {task_name!r} requires an "
                        "existing pairedMsaPath or unpairedMsaPath."
                    )

                msa_dir = os.path.dirname(msa_paths[0])
                standard_names = {"pairing.a3m", "non_pairing.a3m"}
                basenames = {os.path.basename(path) for path in msa_paths}
                if basenames <= standard_names:
                    template_name = "hmmsearch.a3m"
                else:
                    source_identity = "\0".join(
                        os.path.abspath(path) for path in msa_paths
                    ).encode("utf-8")
                    suffix = hashlib.blake2b(
                        source_identity,
                        digest_size=8,
                    ).hexdigest()
                    template_name = f"hmmsearch-{suffix}.a3m"
                template_path = os.path.join(msa_dir, template_name)
                if not os.path.exists(template_path):
                    logger.info(
                        f"Running template search for task {task_name}, "
                        f"sequence: {protein_chain.get('sequence', '')}"
                    )
                    run_template_search(
                        msa_for_template_search_dir=msa_dir,
                        msa_for_template_search_paths=msa_paths,
                        output_path=template_path,
                        hmmsearch_binary_path=hmmsearch_binary_path,
                        hmmbuild_binary_path=hmmbuild_binary_path,
                        seqres_database_path=seqres_database_path,
                    )
                protein_chain["templatesPath"] = template_path
                actual_updated = True
    return actual_updated


if __name__ == "__main__":
    run_template_search(
        msa_for_template_search_dir="examples/5sak/1",
        msa_for_template_search_name="pairing,non_pairing",
    )
