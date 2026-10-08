# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Guidance must stay usable on installs without Triton (CPU and MPS wheels do not ship it)."""

import sys

import pytest
import torch

import opendde.tfg as tfg
from opendde.tfg import contact_core, epitope_guidance, rigid_contact

_TRITON_MODULES = (
    "rigid_core",
    "group_contact",
    "clash_field",
    "pose_search_kernel",
    "vina_steric_kernel",
)


@pytest.fixture
def no_triton(monkeypatch):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)
    for name in _TRITON_MODULES:
        monkeypatch.delitem(sys.modules, f"opendde.tfg.{name}", raising=False)
        monkeypatch.delattr(tfg, name, raising=False)


def test_core_mode_is_read_without_triton(no_triton, monkeypatch):
    monkeypatch.delenv("OPENDDE_RIGID_CORE", raising=False)
    assert rigid_contact.rigid_core_mode() == "auto"
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "off")
    assert rigid_contact.rigid_core_mode() == "off"
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "check")
    assert rigid_contact.rigid_core_mode() == "check"
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "maybe")
    with pytest.raises(ValueError, match="OPENDDE_RIGID_CORE"):
        rigid_contact.rigid_core_mode()


def test_epitope_dispatch_with_the_core_off_never_imports_triton(
    no_triton, monkeypatch
):
    monkeypatch.delenv("OPENDDE_RIGID_CORE", raising=False)

    def core(*_args):
        raise AssertionError("the core must not run")

    out = epitope_guidance._core_dispatch(
        "refine", core, lambda coords, *_args: "dense", torch.zeros(1)
    )
    assert out == "dense"
    assert "opendde.tfg.rigid_core" not in sys.modules


def test_epitope_dispatch_falls_back_when_the_core_cannot_be_imported(
    no_triton, monkeypatch
):
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "on")

    def core(*_args):
        from opendde.tfg import rigid_core  # noqa: F401

    out = epitope_guidance._core_dispatch(
        "refine", core, lambda coords, *_args: "dense", torch.zeros(1)
    )
    assert out == "dense"


def test_contact_dispatch_with_the_core_off_never_imports_triton(
    no_triton, monkeypatch
):
    monkeypatch.delenv("OPENDDE_RIGID_CORE", raising=False)
    out = contact_core.run(
        "refine", torch.zeros(1), {}, rigid_contact, lambda coords, feats, **kw: "dense"
    )
    assert out == "dense"
    assert "opendde.tfg.rigid_core" not in sys.modules


def test_contact_dispatch_falls_back_to_dense_on_cpu_with_the_core_on(
    no_triton, monkeypatch
):
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "on")
    out = contact_core.run(
        "refine", torch.zeros(1), {}, rigid_contact, lambda coords, feats, **kw: "dense"
    )
    assert out == "dense"
