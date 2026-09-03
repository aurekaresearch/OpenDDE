# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import subprocess
from unittest.mock import Mock

import pytest

from opendde.data.tools import kalign


def test_resolve_kalign_binary_prefers_explicit_binary(monkeypatch):
    lookups = []

    def fake_which(command):
        lookups.append(command)
        return "/opt/kalign/bin/kalign" if command == "custom-kalign" else None

    monkeypatch.setattr(kalign.shutil, "which", fake_which)

    assert kalign.resolve_kalign_binary("custom-kalign") == "/opt/kalign/bin/kalign"
    assert lookups == ["custom-kalign"]


def test_resolve_kalign_binary_falls_back_for_none(monkeypatch):
    lookups = []

    def fake_which(command):
        lookups.append(command)
        return "/usr/local/bin/kalign" if command == "kalign" else None

    monkeypatch.setattr(kalign.shutil, "which", fake_which)

    assert kalign.resolve_kalign_binary(None) == "/usr/local/bin/kalign"
    assert lookups == ["kalign"]


def test_resolve_kalign_binary_falls_back_after_explicit_miss(monkeypatch):
    lookups = []

    def fake_which(command):
        lookups.append(command)
        return "/usr/local/bin/kalign" if command == "kalign" else None

    monkeypatch.setattr(kalign.shutil, "which", fake_which)

    assert kalign.resolve_kalign_binary("missing-kalign") == "/usr/local/bin/kalign"
    assert lookups == ["missing-kalign", "kalign"]


def test_align_nonzero_exit_preserves_stdout_and_stderr(monkeypatch):
    resolver = Mock(return_value="/usr/local/bin/kalign")
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="partial alignment output",
            stderr="invalid residue in input",
        )
    )
    monkeypatch.setattr(kalign, "resolve_kalign_binary", resolver)
    monkeypatch.setattr(kalign.subprocess, "run", run)

    aligner = kalign.Kalign(binary_path="configured-kalign")
    with pytest.raises(RuntimeError) as exc_info:
        aligner.align(["ACDEFG", "ACDFFG"])

    resolver.assert_called_once_with("configured-kalign")
    run.assert_called_once()
    assert run.call_args.kwargs == {
        "capture_output": True,
        "text": True,
        "errors": "replace",
        "check": False,
    }
    error_message = str(exc_info.value)
    assert "partial alignment output" in error_message
    assert "invalid residue in input" in error_message
