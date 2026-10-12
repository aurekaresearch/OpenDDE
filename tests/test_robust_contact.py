# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Opt-in robust contact objective (opendde.tfg.robust): switch parsing, the penalty, equivalence with the stock term where they must
agree, gradients, the refusal to fall back to the stock objective, and runs on the accelerated core.

The switches, the penalty and the dispatch need only torch and run everywhere. The term-level tests need Triton to be importable
(they run on CPU tensors); the solver runs need CUDA as well."""

import logging

import pytest
import torch

from opendde.tfg import contact_core, epitope_guidance, robust, rigid_contact as rc

CARBON = 6
TRITON = rc.triton_available()
CUDA = torch.cuda.is_available() and TRITON
needs_triton = pytest.mark.skipif(not TRITON, reason="needs Triton")
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA and Triton")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("OPENDDE_RIGID_PENALTY", "OPENDDE_RIGID_INIT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "on")
    robust._noted.clear()


def pair_problem(n_pairs=5, samples=3, seed=0, outlier=None):
    g = torch.Generator().manual_seed(seed)
    n_fixed, n_moving = 40, 40
    fixed = torch.randn(samples, n_fixed, 3, generator=g) * 6
    moving = torch.randn(samples, n_moving, 3, generator=g) * 5 + torch.tensor(
        [0.0, 0.0, 12.0]
    )
    fb = torch.randperm(n_fixed, generator=g)[:n_pairs]
    ma = torch.randperm(n_moving, generator=g)[:n_pairs]
    lower = torch.full((n_pairs,), 3.5)
    upper = torch.full((n_pairs,), 8.0)
    if outlier is not None:
        moving[:, ma[outlier]] += 40.0
    return fixed, moving, ma, fb, lower, upper


# ---- switches and penalty (torch only) -------------------------------------------------------------------------------------
def test_nothing_is_requested_by_default():
    assert robust.penalty_from_env() is None
    assert robust.init_subsets_enabled() is False
    assert robust.variant_requested() is False


def test_switch_values_are_parsed(monkeypatch):
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "welsch")
    assert robust.penalty_from_env().scale == 4.0
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", " Welsch:2.5 ")
    assert robust.penalty_from_env().scale == 2.5
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "quad")
    assert robust.penalty_from_env() is None and not robust.variant_requested()
    monkeypatch.setenv("OPENDDE_RIGID_INIT", "subsets")
    assert robust.init_subsets_enabled() and robust.variant_requested()
    monkeypatch.setenv("OPENDDE_RIGID_INIT", "all")
    assert not robust.init_subsets_enabled()


@pytest.mark.parametrize(
    "name,value",
    [
        ("OPENDDE_RIGID_PENALTY", "huber"),
        ("OPENDDE_RIGID_PENALTY", "welsch:abc"),
        ("OPENDDE_RIGID_PENALTY", "welsch:0"),
        ("OPENDDE_RIGID_PENALTY", "welsch:-3"),
        ("OPENDDE_RIGID_PENALTY", "welsch:nan"),
        ("OPENDDE_RIGID_INIT", "random"),
    ],
)
def test_invalid_switch_values_raise(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        robust.variant_requested()


def test_the_welsch_force_returns_to_zero_and_the_penalty_saturates():
    w = robust.Welsch(4.0)
    f = w.force(torch.tensor([1.0, 2.0, 3.0, 8.0, 30.0]))
    assert f[2] > f[3] > f[4] and float(f[4]) < 1e-6
    assert float(w.rho(torch.tensor([100.0]))) == pytest.approx(16.0, abs=1e-3)
    e = torch.tensor([0.0, 0.01, 0.1])
    assert torch.allclose(
        w.rho(e), e * e, rtol=1e-3, atol=1e-9
    )  # quadratic near zero, in float32 too


def test_the_penalty_bound_covers_the_change_of_the_energy():
    """`bound` feeds the provable shortlist of the pose search: it must cover |0.5 rho(e') - 0.5 rho(e)| whenever |e' - e| <= dl."""
    g = torch.Generator().manual_seed(0)

    def uniform(scale, shift=0.0):
        return (torch.rand(4000, generator=g, dtype=torch.float64) + shift) * scale

    e, dl = uniform(30.0), uniform(3.0)
    e2 = (e + uniform(2.0, -0.5) * dl).clamp_min(0)  # a violation is never negative
    for penalty in (robust.Penalty(), robust.Welsch(4.0), robust.Welsch(1.5)):
        change = (0.5 * penalty.rho(e2) - 0.5 * penalty.rho(e)).abs()
        assert bool((change <= penalty.bound(e, dl) + 1e-12).all()), penalty.name


# ---- dispatch (torch only) -------------------------------------------------------------------------------------------------
CONTACT_FEATS = {"user_distance_restraint_index": torch.zeros(2, 4, dtype=torch.long)}


@pytest.mark.parametrize(
    "switch", [("OPENDDE_RIGID_PENALTY", "welsch:4"), ("OPENDDE_RIGID_INIT", "subsets")]
)
@pytest.mark.parametrize("core", ["on", "off"])
def test_a_requested_switch_refuses_the_dense_fallback(switch, core, monkeypatch):
    monkeypatch.setenv("OPENDDE_RIGID_CORE", core)
    monkeypatch.setenv(*switch)
    x = torch.zeros(1, 10, 3)  # CPU coordinates: the accelerated core is unusable
    with pytest.raises(RuntimeError, match="accelerated core"):
        contact_core.run(
            "refine", x, CONTACT_FEATS, rc, lambda *a, **k: x, iterations=1
        )


def test_a_request_without_contact_pairs_is_not_refused(monkeypatch):
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "welsch:4")
    x = torch.zeros(1, 10, 3)
    out = contact_core.run("refine", x, {}, rc, lambda *a, **k: "dense", iterations=1)
    assert out == "dense"


def test_epitope_requests_say_once_that_the_contact_switches_do_not_apply(
    monkeypatch, caplog
):
    monkeypatch.setenv("OPENDDE_RIGID_CORE", "off")
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "welsch:4")
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            out = epitope_guidance._core_dispatch(
                "refine", None, lambda coords, *_args: "dense", torch.zeros(1)
            )
    assert out == "dense"
    messages = [r.getMessage() for r in caplog.records]
    assert sum("constraint.contact only" in m for m in messages) == 1


def test_epitope_requests_stay_quiet_without_a_switch(caplog):
    with caplog.at_level(logging.WARNING):
        epitope_guidance._core_dispatch(
            "refine", None, lambda coords, *_args: "dense", torch.zeros(1)
        )
    assert "constraint.contact only" not in caplog.text


# ---- terms (Triton importable, CPU tensors) --------------------------------------------------------------------------------
@needs_triton
def test_the_default_environment_builds_the_stock_term(monkeypatch):
    from opendde.tfg.rigid_core import PairTerm
    from opendde.tfg.robust_term import RobustPairTerm

    fixed, _, ma, fb, lo, up = pair_problem()
    assert type(robust.make_pair_term(fixed, ma, fb, lo, up)) is PairTerm
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "welsch:4")
    assert type(robust.make_pair_term(fixed, ma, fb, lo, up)) is RobustPairTerm


@needs_triton
def test_the_quadratic_robust_term_equals_the_stock_term():
    from opendde.tfg.rigid_core import PairTerm
    from opendde.tfg.robust_term import RobustPairTerm

    fixed, moving, ma, fb, lo, up = pair_problem()
    stock = PairTerm(fixed, ma, fb, lo, up)
    rob = RobustPairTerm(fixed, ma, fb, lo, up, robust.Penalty())
    e0, d0 = stock.energy(moving)
    e1, d1 = rob.energy(moving)
    assert torch.allclose(e0, e1) and torch.allclose(d0, d1)
    e0, g0, _ = stock.energy_grad(moving)
    e1, g1, _ = rob.energy_grad(moving)
    assert torch.allclose(e0, e1) and torch.allclose(g0, g1, atol=1e-6)
    assert torch.equal(stock.satisfied(d0), rob.satisfied(d1))
    assert torch.allclose(stock.search_exact(moving), rob.search_exact(moving))


@needs_triton
@pytest.mark.parametrize("scale", [1.5, 4.0])
def test_the_welsch_gradient_matches_finite_differences(scale):
    from opendde.tfg.robust_term import RobustPairTerm

    fixed, moving, ma, fb, lo, up = pair_problem(outlier=1)
    moving = moving.double()
    term = RobustPairTerm(
        fixed.double(), ma, fb, lo.double(), up.double(), robust.Welsch(scale)
    )
    _, grad, _ = term.energy_grad(moving)  # gradient of 0.5 * energy
    eps = 1e-6
    checked = 0
    for atom in ma.tolist():
        for axis in range(3):
            plus, minus = moving.clone(), moving.clone()
            plus[0, atom, axis] += eps
            minus[0, atom, axis] -= eps
            numeric = (
                0.5
                * float(term.energy(plus)[0][0] - term.energy(minus)[0][0])
                / (2 * eps)
            )
            assert numeric == pytest.approx(
                float(grad[0, atom, axis]), abs=1e-5, rel=1e-4
            )
            checked += 1
    assert checked == 3 * len(ma)


@needs_triton
def test_the_welsch_term_ignores_a_pair_that_cannot_be_met():
    from opendde.tfg.rigid_core import PairTerm
    from opendde.tfg.robust_term import RobustPairTerm

    fixed, moving, ma, fb, lo, up = pair_problem(n_pairs=4, outlier=2)
    stock = PairTerm(fixed, ma, fb, lo, up)
    welsch = RobustPairTerm(fixed, ma, fb, lo, up, robust.Welsch(4.0))
    _, g_stock, _ = stock.energy_grad(moving)
    _, g_welsch, _ = welsch.energy_grad(moving)
    far = ma[2]  # moved 40 A away from its partner
    assert float(g_welsch[:, far].norm(dim=-1).max()) < 0.01 * float(
        g_stock[:, far].norm(dim=-1).min()
    )


# ---- end to end on the accelerated core ------------------------------------------------------------------------------------
def slab_scene(height=12.0, wrong_pair=True):
    xs = torch.arange(-13.5, 14.0, 3.0)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    antigen = torch.cat([grid, torch.zeros(len(grid), 1)], -1)
    bx = torch.arange(-8.75, 9.0, 3.5)
    plate = torch.stack(torch.meshgrid(bx, bx, indexing="ij"), -1).reshape(-1, 2)
    body = torch.cat(
        [torch.cat([plate, torch.full((len(plate), 1), 3.5 * i)], -1) for i in range(3)]
    )
    body = body + torch.tensor([0.0, 0.0, height])
    n_fixed, n_moving = len(antigen), len(body)
    coords = torch.cat([antigen, body]).unsqueeze(0).repeat(2, 1, 1)
    atoms = n_fixed + n_moving
    movable = torch.zeros(atoms, dtype=torch.bool)
    movable[n_fixed:] = True
    centre = [
        i
        for i in range(n_fixed)
        if abs(float(antigen[i, 0])) < 5 and abs(float(antigen[i, 1])) < 5
    ]
    mov_centre = [i for i in range(len(plate)) if abs(float(plate[i, 0])) < 5]
    fixed_side = torch.tensor(centre[:3])
    moving_side = n_fixed + torch.tensor(mov_centre[:3])
    if wrong_pair:  # a far antigen corner paired with a moving atom on the opposite corner of the block
        far = int(torch.argmax(antigen[:, 0] + antigen[:, 1]))
        opp = int(torch.argmin(plate[:, 0] + plate[:, 1]))
        fixed_side = torch.cat([fixed_side, torch.tensor([far])])
        moving_side = torch.cat([moving_side, torch.tensor([n_fixed + opp])])
    n = len(fixed_side)
    element = torch.zeros(atoms, 128)
    element[:, CARBON] = 1
    feats = {
        "user_distance_restraint_index": torch.stack([fixed_side, moving_side]),
        "user_distance_restraint_lower_bound": torch.full((n,), 3.5),
        "user_distance_restraint_upper_bound": torch.full((n,), 8.0),
        "user_rigid_movable_atom": movable,
        "ref_element": element,
        "asym_id": movable.long(),
        "atom_to_token_idx": torch.arange(atoms),
    }
    return coords, feats, n_fixed


def pair_d(x, feats):
    idx = feats["user_distance_restraint_index"]
    return torch.linalg.vector_norm(x[:, idx[0]] - x[:, idx[1]], dim=-1)


def solve(x, feats):
    found = rc.search_rigid_contact(x, feats)
    return rc.refine_rigid_contact(found, feats, iterations=120)


@needs_cuda
def test_welsch_with_subset_starts_runs_on_the_core_and_never_creates_a_severe_overlap(
    monkeypatch,
):
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "welsch:4")
    monkeypatch.setenv("OPENDDE_RIGID_INIT", "subsets")
    x, feats, n_fixed = slab_scene(height=30.0, wrong_pair=False)
    x, feats = x.cuda(), {k: v.cuda() for k, v in feats.items()}
    out = solve(x, feats)
    assert torch.equal(out[:, :n_fixed], x[:, :n_fixed])  # the antigen never moves
    assert bool((pair_d(out, feats) <= 8.0 + 0.5).all())


@needs_cuda
def test_welsch_with_subset_starts_ends_closer_to_the_correct_pairs_than_stock_with_one_wrong_pair(
    monkeypatch,
):
    """The fourth pair pairs far corners and cannot be met together with the other three. This synthetic scene does not recover the
    full pose (the benchmark numbers are in docs/tfg_constraint_guidance.md); it only checks the direction of the effect."""
    x, feats, _ = slab_scene(height=12.0, wrong_pair=True)
    x, feats = x.cuda(), {k: v.cuda() for k, v in feats.items()}
    stock = pair_d(solve(x, feats), feats)
    monkeypatch.setenv("OPENDDE_RIGID_PENALTY", "welsch:4")
    monkeypatch.setenv("OPENDDE_RIGID_INIT", "subsets")
    robust_d = pair_d(solve(x, feats), feats)
    assert float(robust_d[:, :3].max()) < float(stock[:, :3].max())
    assert float(robust_d[:, :3].mean()) < float(stock[:, :3].mean())
