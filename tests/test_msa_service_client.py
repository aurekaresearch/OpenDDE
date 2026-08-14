# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import io
import tarfile

import pytest

import opendde.data.msa.msa_service_client as client
from opendde.data.msa.msa_service_client import (
    _safe_extract_tar,
    _write_query_leading_a3m,
    gather_a3m_lines,
    search_and_build_msa,
)


def test_gather_a3m_lines_splits_by_numeric_header(tmp_path):
    """ColabFold packs results keyed by numeric header, separated by \\x00."""
    a3m = tmp_path / "uniref.a3m"
    a3m.write_text(">101\nACDE\n>hitA\tmeta\nA-DE\n\x00>102\nFGHI\n>hitB\tmeta\nF-HI\n")

    merged = gather_a3m_lines([str(a3m)], [101, 102])

    assert merged[0].startswith(">101\nACDE\n")
    assert "hitA" in merged[0] and "hitB" not in merged[0]
    assert merged[1].startswith(">102\nFGHI\n")
    assert "hitB" in merged[1] and "hitA" not in merged[1]


def test_gather_a3m_lines_follows_requested_order(tmp_path):
    a3m = tmp_path / "uniref.a3m"
    a3m.write_text(">101\nACDE\n\x00>102\nFGHI\n")

    merged = gather_a3m_lines([str(a3m)], [102, 101])

    assert merged[0].startswith(">102\nFGHI")
    assert merged[1].startswith(">101\nACDE")


def test_search_and_build_msa_writes_query_leading_files(tmp_path, monkeypatch):
    seqs = ["ACDE", "FGHI"]

    def fake_run(query_seqs, prefix, use_env=True, use_pairing=False, **kwargs):
        tag = "p" if use_pairing else "u"
        return [
            f">{101 + i}\n{seq}\n>{tag}_{i}\n{seq[::-1]}\n"
            for i, seq in enumerate(query_seqs)
        ]

    monkeypatch.setattr(client, "run_mmseqs2", fake_run)

    dirs = search_and_build_msa(seqs, str(tmp_path))

    assert len(dirs) == 2
    for i, seq in enumerate(seqs):
        non_pairing = (tmp_path / str(i) / "non_pairing.a3m").read_text()
        pairing = (tmp_path / str(i) / "pairing.a3m").read_text()
        # First record header is normalized to >query, sequence preserved.
        assert non_pairing.startswith(f">query\n{seq}\n")
        assert f"u_{i}" in non_pairing
        assert pairing.startswith(f">query\n{seq}\n")
        assert f"p_{i}" in pairing


def test_search_and_build_msa_single_sequence_skips_pairing(tmp_path, monkeypatch):
    def fake_run(query_seqs, prefix, use_env=True, use_pairing=False, **kwargs):
        assert not use_pairing, "single sequence should not trigger paired search"
        return [f">101\n{query_seqs[0]}\n>u\n{query_seqs[0]}\n"]

    monkeypatch.setattr(client, "run_mmseqs2", fake_run)

    search_and_build_msa(["ACDE"], str(tmp_path))

    # No cross-chain pairing for a single chain: pairing is query-only.
    assert (tmp_path / "0" / "pairing.a3m").read_text() == ">query\nACDE\n"
    assert (tmp_path / "0" / "non_pairing.a3m").read_text().startswith(">query\nACDE\n")


def test_search_and_build_msa_falls_back_to_query_only_on_failure(
    tmp_path, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("MSA service unreachable")

    monkeypatch.setattr(client, "run_mmseqs2", boom)

    dirs = search_and_build_msa(["ACDE", "FGHI"], str(tmp_path))

    assert len(dirs) == 2
    for i, seq in enumerate(["ACDE", "FGHI"]):
        assert (tmp_path / str(i) / "non_pairing.a3m").read_text() == f">query\n{seq}\n"
        assert (tmp_path / str(i) / "pairing.a3m").read_text() == f">query\n{seq}\n"


def test_write_query_leading_a3m_falls_back_on_query_mismatch(tmp_path):
    output = tmp_path / "mismatch.a3m"

    _write_query_leading_a3m(str(output), ">101\nWRONG\n>hit\nWRONG\n", "ACDE")

    assert output.read_text() == ">query\nACDE\n"


def test_safe_extract_tar_rejects_path_traversal(tmp_path):
    malicious = tmp_path / "evil.tar"
    payload = b"owned"
    with tarfile.open(malicious, "w") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    destination = tmp_path / "out"
    destination.mkdir()
    with tarfile.open(malicious) as tar:
        with pytest.raises(RuntimeError):
            _safe_extract_tar(tar, str(destination))

    assert not (tmp_path / "escape.txt").exists()
