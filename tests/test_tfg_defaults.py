"""The defaults of the rigid-body guidance are the evaluated settings, and they resolve per input and per machine."""

import logging

import pytest
import torch

from opendde.tfg import contact_core, epitope_guidance as eg, rigid_contact as rc

VARIABLES = (
    "OPENDDE_RIGID_CONTACT",
    "OPENDDE_RIGID_CORE",
    "OPENDDE_RIGID_START",
    "OPENDDE_RIGID_EVERY",
    "OPENDDE_RIGID_X0_START",
    "OPENDDE_RIGID_X0_EVERY",
    "OPENDDE_RIGID_X0_LAST",
    "OPENDDE_VINA_FAST",
)


@pytest.fixture(autouse=True)
def no_overrides(monkeypatch):
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)


def contact_feats(chains, movable=None):
    """Two contact pairs between chain 0 and the last chain; one atom per token."""
    atoms = len(chains)
    feats = {
        "asym_id": torch.tensor(chains),
        "atom_to_token_idx": torch.arange(atoms),
        "user_distance_restraint_index": torch.tensor([[0, 0], [atoms - 1, atoms - 2]]),
    }
    if movable is not None:
        feats["user_rigid_movable_atom"] = torch.tensor(movable)
    return feats


def test_the_early_pass_defaults_to_the_evaluated_schedule():
    assert eg.x0_schedule() == (100, 3, 189)
    steps = [s for s in range(200) if eg.x0_step_active(s)]
    assert steps == list(range(100, 188, 3)) and len(steps) == 30


def test_the_early_pass_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("OPENDDE_RIGID_X0_START", "off")
    assert eg.x0_schedule() is None


def test_the_late_pass_defaults_to_the_evaluated_schedule():
    coarse, refine = rc.intervention_schedule(200)
    assert coarse == [190, 199]
    assert refine == [190, 192, 194, 196, 198, 199]


def test_the_default_late_pass_fits_a_short_run():
    coarse, refine = rc.intervention_schedule(100)
    assert coarse == [99] and refine == [99]


def test_an_explicit_late_start_must_lie_inside_the_run(monkeypatch):
    monkeypatch.setenv("OPENDDE_RIGID_START", "150")
    with pytest.raises(ValueError):
        rc.intervention_schedule(100)


def test_rigid_guidance_is_chosen_from_the_input():
    epitope = {"user_epitope_atom_index": torch.zeros(1, 1, dtype=torch.long)}
    assert eg.rigid_mode(epitope) == "on"
    assert (
        eg.rigid_mode(contact_feats([0, 0, 1, 1], [False, False, True, True])) == "on"
    )
    assert eg.rigid_mode(contact_feats([0, 0, 1, 1])) == "on"  # exactly two chains
    assert (
        eg.rigid_mode(contact_feats([0, 1, 2, 2])) == "off"
    )  # no movable_chains, three chains
    assert eg.rigid_mode({}) == "off"  # no constraint
    assert eg.rigid_mode(epitope, tfg_enabled=False) == "off"


@pytest.mark.parametrize("value", ["off", "control", "on"])
def test_an_explicit_rigid_mode_is_taken_as_given(monkeypatch, value):
    monkeypatch.setenv("OPENDDE_RIGID_CONTACT", value)
    assert eg.rigid_mode({}) == value


def test_an_unknown_rigid_mode_is_refused(monkeypatch):
    monkeypatch.setenv("OPENDDE_RIGID_CONTACT", "maybe")
    with pytest.raises(ValueError, match="OPENDDE_RIGID_CONTACT"):
        eg.rigid_mode({})


def test_the_default_core_falls_back_quietly_on_cpu(caplog):
    caplog.set_level(logging.DEBUG)
    out = contact_core.run(
        "refine", torch.zeros(1), {}, rc, lambda coords, feats, **kw: "dense"
    )
    assert out == "dense"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("value", ["false", "0", "yes", ""])
def test_rigid_late_rejects_values_other_than_on_and_off(monkeypatch, value):
    from opendde.tfg.rigid_contact import late_pass_enabled

    monkeypatch.setenv("OPENDDE_RIGID_LATE", value)
    with pytest.raises(ValueError, match="OPENDDE_RIGID_LATE must be on or off"):
        late_pass_enabled()


def test_rigid_late_accepts_on_and_off_case_insensitively(monkeypatch):
    from opendde.tfg.rigid_contact import late_pass_enabled

    monkeypatch.delenv("OPENDDE_RIGID_LATE", raising=False)
    assert late_pass_enabled() is True
    monkeypatch.setenv("OPENDDE_RIGID_LATE", "OFF")
    assert late_pass_enabled() is False
    monkeypatch.setenv("OPENDDE_RIGID_LATE", " On ")
    assert late_pass_enabled() is True


@pytest.mark.parametrize("value", ["OFF", "off", " off ", ""])
def test_x0_start_off_is_case_insensitive(monkeypatch, value):
    from opendde.tfg import epitope_guidance

    monkeypatch.setenv("OPENDDE_RIGID_X0_START", value)
    assert epitope_guidance.x0_schedule() is None


def test_x0_start_garbage_is_a_named_error(monkeypatch):
    from opendde.tfg import epitope_guidance

    monkeypatch.setenv("OPENDDE_RIGID_X0_START", "soon")
    with pytest.raises(ValueError, match="OPENDDE_RIGID_X0_START must be a step"):
        epitope_guidance.x0_schedule()
