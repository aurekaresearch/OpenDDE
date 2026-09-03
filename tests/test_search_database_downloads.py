# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
from pathlib import Path


def test_template_search_downloads_missing_database(monkeypatch, tmp_path):
    from runner import template_search

    calls = []

    def fake_download(url, path, check_weight=True):
        calls.append((url, path, check_weight))
        Path(path).write_text(">seq\nACDE\n")

    monkeypatch.setattr(template_search, "download_from_url", fake_download)
    monkeypatch.setattr(template_search, "run_hmmsearch_with_a3m", lambda **_: "")

    template_search.run_template_search(
        msa_for_template_search_dir=str(tmp_path),
        msa_for_template_search_name="missing",
        hmmsearch_binary_path="/bin/echo",
        hmmbuild_binary_path="/bin/echo",
        seqres_database_path=str(tmp_path / "db" / "pdb_seqres.fasta"),
    )

    assert calls == [
        (
            template_search.TEMPLATE_SEARCH_DATABASE_URL,
            str(tmp_path / "db" / "pdb_seqres.fasta"),
            False,
        )
    ]
    assert (tmp_path / "hmmsearch.a3m").exists()


def test_template_search_resolves_bare_hmmer_commands(monkeypatch, tmp_path):
    from runner import template_search

    msa_path = tmp_path / "custom_input.a3m"
    database_path = tmp_path / "pdb_seqres.fasta"
    msa_path.write_text(">query\nACDE\n")
    database_path.write_text(">template\nACDE\n")
    resolved = []

    def which(command):
        resolved.append(command)
        return "/bin/echo"

    monkeypatch.setattr(template_search.shutil, "which", which)
    monkeypatch.setattr(template_search, "run_hmmsearch_with_a3m", lambda **_: "")

    template_search.run_template_search(
        msa_for_template_search_dir=str(tmp_path),
        msa_for_template_search_paths=[str(msa_path)],
        hmmsearch_binary_path="hmmsearch",
        hmmbuild_binary_path="hmmbuild",
        seqres_database_path=str(database_path),
    )

    assert resolved == ["hmmsearch", "hmmbuild"]


def test_update_template_info_accepts_arbitrary_msa_filenames(monkeypatch, tmp_path):
    from runner import template_search

    paired = tmp_path / "protein-hash.paired.custom.a3m"
    unpaired = tmp_path / "protein-hash.unpaired.custom.a3m"
    paired.write_text(">query\nACDE\n")
    unpaired.write_text(">query\nACDE\n")
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_text(">hit\nACDE\n")

    monkeypatch.setattr(template_search, "run_template_search", run)
    jobs = [
        {
            "name": "job",
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "ACDE",
                        "pairedMsaPath": str(paired),
                        "unpairedMsaPath": str(unpaired),
                    }
                }
            ],
        }
    ]

    assert template_search.update_template_info(jobs)
    assert calls[0]["msa_for_template_search_paths"] == [
        str(paired),
        str(unpaired),
    ]
    template_path = Path(jobs[0]["sequences"][0]["proteinChain"]["templatesPath"])
    assert template_path.parent == tmp_path
    assert template_path.name.startswith("hmmsearch-")
    assert template_path.suffix == ".a3m"


def test_custom_msa_sets_in_one_directory_do_not_share_template_results(
    monkeypatch, tmp_path
):
    from runner import template_search

    chains = []
    for index, sequence in enumerate(("ACDE", "FGHI")):
        paired = tmp_path / f"chain{index}.paired.a3m"
        unpaired = tmp_path / f"chain{index}.unpaired.a3m"
        paired.write_text(f">query\n{sequence}\n")
        unpaired.write_text(f">query\n{sequence}\n")
        chains.append(
            {
                "proteinChain": {
                    "sequence": sequence,
                    "pairedMsaPath": str(paired),
                    "unpairedMsaPath": str(unpaired),
                }
            }
        )

    def run(**kwargs):
        Path(kwargs["output_path"]).write_text(">hit\nACDE\n")

    monkeypatch.setattr(template_search, "run_template_search", run)
    jobs = [{"name": "complex", "sequences": chains}]

    assert template_search.update_template_info(jobs)
    template_paths = [chain["proteinChain"]["templatesPath"] for chain in chains]
    assert len(set(template_paths)) == 2


def test_requested_template_search_rejects_missing_msa_files(tmp_path):
    import pytest

    from runner import template_search

    jobs = [
        {
            "name": "job",
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": "ACDE",
                        "pairedMsaPath": str(tmp_path / "missing-paired.a3m"),
                        "unpairedMsaPath": str(tmp_path / "missing-unpaired.a3m"),
                    }
                }
            ],
        }
    ]

    with pytest.raises(FileNotFoundError, match="existing pairedMsaPath"):
        template_search.update_template_info(jobs)


def test_rna_msa_search_downloads_missing_databases(monkeypatch, tmp_path):
    from runner import rna_msa_search

    calls = []

    def fake_download(url, path, check_weight=True):
        calls.append((url, path, check_weight))
        Path(path).write_text(">seq\nACGU\n")

    class DummyMsa:
        def to_a3m(self):
            return ">query\nACGU\n"

    monkeypatch.setattr(rna_msa_search, "download_from_url", fake_download)
    monkeypatch.setattr(rna_msa_search, "_get_rna_msa", lambda **_: DummyMsa())

    db_paths = {
        "ntrna_database_path": tmp_path / "db" / "nt.fasta",
        "rfam_database_path": tmp_path / "db" / "rfam.fasta",
        "rna_central_database_path": tmp_path / "db" / "rnacentral.fasta",
    }

    rna_msa_search.run_rna_msa_search(
        rna_seq_for_msa_search="ACGU",
        rna_result_path=str(tmp_path / "out"),
        rna_seq_id="rna",
        nhmmer_binary_path="/bin/echo",
        hmmalign_binary_path="/bin/echo",
        hmmbuild_binary_path="/bin/echo",
        ntrna_database_path=str(db_paths["ntrna_database_path"]),
        rfam_database_path=str(db_paths["rfam_database_path"]),
        rna_central_database_path=str(db_paths["rna_central_database_path"]),
    )

    assert calls == [
        (
            rna_msa_search.NT_SEARCH_DATABASE_URL,
            str(db_paths["ntrna_database_path"]),
            False,
        ),
        (
            rna_msa_search.RFAM_SEARCH_DATABASE_URL,
            str(db_paths["rfam_database_path"]),
            False,
        ),
        (
            rna_msa_search.RNACENTRAL_SEARCH_DATABASE_URL,
            str(db_paths["rna_central_database_path"]),
            False,
        ),
    ]
    assert (tmp_path / "out" / "rna" / "rna_msa.a3m").read_text() == ">query\nACGU\n"
