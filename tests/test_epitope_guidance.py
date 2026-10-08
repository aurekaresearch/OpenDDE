"""Unit tests of the epitope guidance on synthetic geometry (CPU, no model, no data files).

System: an antigen slab (two layers of a 3 A grid, chain 0, fixed) and an antibody block (three layers of a 3.5 A grid, chain 1,
moving) whose lowest layer is the paratope. Six epitope residues of three atoms each sit on the slab.
"""

import math
import os

import pytest
import torch

DEV = "cuda" if os.environ.get("EPI_TEST_DEVICE") == "cuda" else "cpu"

from opendde.tfg import epitope_guidance as eg
from opendde.tfg import rigid_contact as rc

CARBON = 6


def build(height=20.0, samples=1, k=2, seed=0, rotate=False, layers=3, para_layers=1):
    g = torch.Generator().manual_seed(seed)
    xs = torch.arange(-13.5, 14.0, 3.0)
    slab = torch.stack(
        [
            torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2),
        ],
        0,
    )[0]
    top = torch.cat([slab, torch.zeros(len(slab), 1)], -1)
    bottom = torch.cat([slab, torch.full((len(slab), 1), -3.0)], -1)
    antigen = torch.cat([top, bottom])
    bx = torch.arange(-8.75, 9.0, 3.5)
    plate = torch.stack(torch.meshgrid(bx, bx, indexing="ij"), -1).reshape(-1, 2)
    body = torch.cat(
        [
            torch.cat([plate, torch.full((len(plate), 1), 3.5 * i)], -1)
            for i in range(layers)
        ]
    )
    n_fixed, n_moving = len(antigen), len(body)
    x = (
        torch.cat([antigen, body + torch.tensor([0.0, 0.0, height])])
        .unsqueeze(0)
        .repeat(samples, 1, 1)
    )
    x = x + 0.01 * torch.randn(x.shape, generator=g)
    if rotate:
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        x = x @ q.T + torch.tensor([5.0, -7.0, 11.0])
    atoms = n_fixed + n_moving
    movable = torch.zeros(atoms, dtype=torch.bool)
    movable[n_fixed:] = True
    para = torch.zeros(atoms, dtype=torch.bool)
    para[n_fixed : n_fixed + para_layers * len(plate)] = (
        True  # the lowest para_layers layers of the block
    )
    centre = [
        i
        for i in range(len(top))
        if abs(float(top[i, 0])) < 5 and abs(float(top[i, 1])) < 5
    ]
    epi = torch.full((6, 3), -1, dtype=torch.int64)
    for r in range(6):
        epi[r] = torch.tensor(centre[3 * r : 3 * r + 3])
    element = torch.zeros(atoms, 128)
    element[:, CARBON] = 1
    chain = torch.zeros(atoms, dtype=torch.long)
    chain[n_fixed:] = 1
    feats = {
        "user_rigid_movable_atom": movable,
        "user_epitope_atom_index": epi,
        "user_epitope_paratope_atom": para,
        "user_epitope_k": torch.tensor([k]),
        "ref_element": element,
        "asym_id": chain,
        "atom_to_token_idx": torch.arange(atoms),
    }
    return x.to(DEV), {key: v.to(DEV) for key, v in feats.items()}, n_fixed


def severe(x, feats):
    fixed_ids, moving_ids, *_ = eg.groups(x, feats)
    rsum = eg._radii(x, feats, moving_ids, fixed_ids)
    d = torch.cdist(
        x[:, moving_ids], x[:, fixed_ids], compute_mode="donot_use_mm_for_euclid_dist"
    )
    return (d < 0.75 * rsum).flatten(1).any(-1)


def reached(x, feats):
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x, feats)
    d, *_ = eg.residue_distances(
        x[:, moving_ids], x[:, fixed_ids], epi_local, valid, para_local
    )
    return eg.reached_count(d), k


def test_gradient_matches_finite_differences():
    x, feats, _ = build(height=8.0, k=3)
    x = x.double()
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x, feats)
    fixed, moving = x[:, fixed_ids], x[:, moving_ids].clone()
    energy, grad, _ = eg.contact_energy_and_gradient(
        moving, fixed, epi_local, valid, para_local, k
    )
    assert float(energy) > 0
    eps = 1e-5
    checked = 0
    for atom in para_local.tolist():
        for axis in range(3):
            plus, minus = moving.clone(), moving.clone()
            plus[0, atom, axis] += eps
            minus[0, atom, axis] -= eps
            e1 = eg.contact_energy_and_gradient(
                plus, fixed, epi_local, valid, para_local, k
            )[0]
            e2 = eg.contact_energy_and_gradient(
                minus, fixed, epi_local, valid, para_local, k
            )[0]
            numeric = float(e1 - e2) / (2 * eps)
            assert numeric == pytest.approx(
                float(grad[0, atom, axis]), abs=1e-6, rel=1e-4
            )
            checked += 1
    assert checked > 50 and float(grad.abs().sum()) > 0


def test_forces_from_residues_sharing_a_paratope_atom_add_up():
    x, feats, _ = build(height=10.0, k=2)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x, feats)
    fixed, moving = x[:, fixed_ids].clone(), x[:, moving_ids].clone()
    # one paratope atom is moved next to two epitope residues so that it is the nearest atom of both
    target = int(para_local[0])
    moving[0, target] = fixed[0, epi_local[0, 0]] + torch.tensor(
        [0.0, 0.0, 6.0], device=x.device
    )
    d, a_slot, p_slot, _ = eg.residue_distances(
        moving, fixed, epi_local, valid, para_local
    )
    energy, grad, _ = eg.contact_energy_and_gradient(
        moving, fixed, epi_local, valid, para_local, 6
    )
    shared = (para_local[p_slot[0]] == target).sum().item()
    assert shared >= 2, (
        "the test geometry must make one atom the nearest paratope atom of two residues"
    )
    single = eg.contact_energy_and_gradient(
        moving, fixed, epi_local[:1], valid[:1], para_local, 1
    )[1][0, target]
    assert float(grad[0, target].norm()) > float(single.norm()) * 1.5


def test_far_start_is_moved_into_contact_rigidly_and_without_severe_clash():
    x, feats, _ = build(height=25.0, k=2)
    before, k = reached(x, feats)
    assert int(before) == 0
    out = eg.search_epitope(x, feats)
    out = eg.refine_epitope(out, feats, iterations=120)
    after, _ = reached(out, feats)
    assert int(after) >= k
    assert not bool(severe(out, feats).any())
    mov = feats["user_rigid_movable_atom"]
    exact = "donot_use_mm_for_euclid_dist"  # the default matrix-product mode is only accurate to about 1e-2 in float32
    d0 = torch.cdist(x[0, mov], x[0, mov], compute_mode=exact)
    d1 = torch.cdist(out[0, mov], out[0, mov], compute_mode=exact)
    assert float((d0 - d1).abs().max()) < 1e-3, (
        "the moving group must stay a rigid body"
    )
    assert torch.equal(out[0, ~mov], x[0, ~mov]), "the fixed group must not move"


def test_already_satisfied_sample_is_untouched_in_a_mixed_batch():
    near, feats, _ = build(height=4.5, k=2)
    far, _, _ = build(height=25.0, k=2)
    x = torch.cat([near, far])
    ok, k = reached(x, feats)
    assert int(ok[0]) >= k and int(ok[1]) == 0
    out = eg.refine_epitope(eg.search_epitope(x, feats), feats, iterations=120)
    assert torch.equal(out[0], x[0])
    assert int(reached(out, feats)[0][1]) >= k


@pytest.mark.parametrize("k", [1, 6])
def test_extreme_k_values_run_and_never_worsen_the_energy(k):
    x, feats, _ = build(height=25.0, k=k)
    out = eg.refine_epitope(eg.search_epitope(x, feats), feats, iterations=60)
    fixed_ids, moving_ids, epi_local, valid, para_local, kk = eg.groups(x, feats)
    e0 = eg.contact_energy_and_gradient(
        x[:, moving_ids], x[:, fixed_ids], epi_local, valid, para_local, kk
    )[0]
    e1 = eg.contact_energy_and_gradient(
        out[:, moving_ids], out[:, fixed_ids], epi_local, valid, para_local, kk
    )[0]
    assert float(e1) <= float(e0) + 1e-6
    assert not bool(severe(out, feats).any())


def test_refinement_commutes_with_a_rigid_motion_of_the_whole_system():
    x, feats, _ = build(height=9.0, k=3, seed=3)
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=torch.Generator().manual_seed(9))
    )
    if torch.det(q) < 0:
        q[:, 0] *= -1
    q = q.to(x.device)
    t = torch.tensor([3.0, -2.0, 6.0], device=x.device)
    a = eg.refine_epitope(x, feats, iterations=40)
    b = eg.refine_epitope(x @ q.T + t, feats, iterations=40)
    assert float((a @ q.T + t - b).abs().max()) < 5e-3


def test_permuting_the_epitope_list_changes_nothing():
    x, feats, _ = build(height=9.0, k=3, seed=5)
    a = eg.refine_epitope(x, feats, iterations=40)
    perm = torch.randperm(6, generator=torch.Generator().manual_seed(1))
    feats2 = dict(feats)
    feats2["user_epitope_atom_index"] = feats["user_epitope_atom_index"][perm]
    b = eg.refine_epitope(x, feats2, iterations=40)
    assert float((a - b).abs().max()) < 1e-4


def test_degenerate_and_cross_chain_normals_fall_back_without_nan():
    x, feats, n_fixed = build(height=20.0, k=2)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x, feats)
    fixed = x[:, fixed_ids].clone()
    line = torch.zeros_like(fixed)
    line[..., 0] = (
        torch.arange(fixed.shape[1]).float() * 1.5
    )  # every antigen atom on one line
    chain = feats["asym_id"][fixed_ids]
    axes, note = eg.approach_axes(
        line, chain, epi_local, valid, line[:, epi_local[valid]].mean(1)
    )
    assert (
        "fallback_normal_for_samples=[0]" in note
        and len(axes) == 1 + eg.TILT_AZIMUTHS + eg.FIBONACCI_AXES
    )
    assert all(bool(torch.isfinite(a).all()) for a in axes)
    two_chains = chain.clone()
    two_chains[epi_local[3, 0]] = 5
    axes, note = eg.approach_axes(
        fixed, two_chains, epi_local, valid, fixed[:, epi_local[valid]].mean(1)
    )
    assert "cross_chain" in note and len(axes) == eg.FIBONACCI_AXES
    axes, note = eg.approach_axes(
        fixed, chain, epi_local, valid, fixed[:, epi_local[valid]].mean(1)
    )
    assert (
        note == "normal+tilts+fibonacci"
        and len(axes) == 1 + eg.TILT_AZIMUTHS + eg.FIBONACCI_AXES
    )
    # the outward normal of the slab points up (+z) and the antibody starts above it
    assert float(axes[0][0, 2]) > 0.9


def test_antiparallel_alignment_is_a_proper_rotation():
    a = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    r = eg._rotation_aligning(a, -a)
    assert torch.allclose(
        torch.bmm(r, r.transpose(1, 2)), torch.eye(3).expand(2, 3, 3), atol=1e-5
    )
    assert torch.allclose(torch.det(r), torch.ones(2), atol=1e-5)
    assert torch.allclose(torch.bmm(r, a[..., None]).squeeze(-1), -a, atol=1e-5)


def test_bad_groups_are_refused():
    x, feats, _ = build()
    bad = dict(feats)
    epi = feats["user_epitope_atom_index"].clone()
    epi[0, 0] = int(torch.where(feats["user_rigid_movable_atom"])[0][0])
    bad["user_epitope_atom_index"] = epi
    with pytest.raises(ValueError):
        eg.groups(x, bad)
    bad = dict(feats)
    bad["user_epitope_k"] = torch.tensor([99])
    with pytest.raises(ValueError):
        eg.groups(x, bad)
    bad = dict(feats)
    bad["user_rigid_movable_atom"] = torch.empty(0, dtype=torch.bool)
    with pytest.raises(ValueError):
        eg.groups(x, bad)


def test_a_paratope_spread_deep_into_the_body_still_finds_a_clash_free_pose():
    """Regression for the first smoke test (8aon): the paratope centroid lies far behind the surface that must touch the antigen.
    Anchoring the placement on the centroid buried the antibody in the antigen for every stand-off; the paratope tip is the anchor now."""
    x, feats, _ = build(height=40.0, k=2, layers=10, para_layers=9)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x, feats)
    body = x[0, moving_ids]
    para = body[para_local]
    assert float((para.mean(0)[2] - para[:, 2].min())) > 12.0, (
        "the test geometry needs a deep centroid"
    )
    out = eg.search_epitope(x, feats)
    assert not bool(severe(out, feats).any())
    assert not torch.equal(out, x), "the search must accept a candidate"
    out = eg.refine_epitope(out, feats, iterations=120)
    assert int(reached(out, feats)[0]) >= k
    assert not bool(severe(out, feats).any())


def test_coarse_search_returns_the_input_untouched_when_every_sample_already_meets_the_request():
    x, feats, _ = build(height=4.5, k=2, samples=2)
    assert bool((reached(x, feats)[0] >= 2).all())
    out = eg.search_epitope(x, feats)
    assert out is x or torch.equal(out, x)


# ---- edge cases of the geometry --------------------------------------------------------------------------------------------------------


def test_alignment_rotation_is_exactly_a_rotation_near_the_antiparallel_branch():
    """The closed form 1/(1+c) would lose orthogonality (determinant 0.948) about 0.1 degree away from antiparallel; the construction used must stay an exact rotation."""
    for angle in [0.0, 1e-9, 1e-7, 1e-5, 1e-4, 2e-3, 1e-2, 1e-1, 0.5, 3.0]:
        a = torch.tensor([[0.0, 0.0, 1.0]])
        b = torch.tensor(
            [[math.sin(angle), 0.0, -math.cos(angle)]]
        )  # angle away from exactly antiparallel
        r = eg._rotation_aligning(a, b)
        assert float((r @ r.transpose(1, 2) - torch.eye(3)).abs().max()) < 1e-5, angle
        assert float(torch.det(r)[0]) == pytest.approx(1.0, abs=1e-5), angle
        assert float((torch.bmm(r, a[..., None]).squeeze(-1) - b).abs().max()) < 1e-4, (
            angle
        )
    r = eg._rotation_aligning(a, a)  # parallel: identity
    assert torch.allclose(r, torch.eye(3).expand(1, 3, 3), atol=1e-6)


def test_the_full_search_never_changes_the_distances_inside_the_moving_group():
    """Whatever candidate the search accepts is a rigid placement: the distances inside the moving group do not change."""
    z = 1 - 1 / eg.FIBONACCI_AXES
    u = torch.tensor([math.sqrt(1 - z * z), 0.0, z])
    angle = 0.002
    q = torch.tensor(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    a = q @ u
    tip = torch.tensor([0.0, 0.0, 20.0])
    moving = torch.stack(
        [
            tip,
            tip - 10 * a + torch.tensor([0.0, 1.0, 0.0]),
            tip - 10 * a - torch.tensor([0.0, 1.0, 0.0]),
        ]
    )
    coords = torch.cat([torch.zeros(1, 3), moving])[None]
    elements = torch.zeros(4, 128)
    elements[:, 6] = 1
    feats = {
        "user_rigid_movable_atom": torch.tensor([False, True, True, True]),
        "user_epitope_atom_index": torch.tensor([[0]]),
        "user_epitope_paratope_atom": torch.tensor([False, True, False, False]),
        "user_epitope_k": torch.tensor([1]),
        "ref_element": elements,
        "asym_id": torch.tensor([0, 1, 1, 1]),
        "atom_to_token_idx": torch.arange(4),
    }
    out = eg.search_epitope(coords, feats)
    mode = "donot_use_mm_for_euclid_dist"
    d0 = torch.cdist(coords[:, 1:], coords[:, 1:], compute_mode=mode)
    d1 = torch.cdist(out[:, 1:], out[:, 1:], compute_mode=mode)
    assert float((d1 - d0).abs().max()) < 1e-4


def test_the_search_objective_weights_the_clash_term_by_ten():
    """The search objective is 0.5 * contact + 10 * clash. Check the coefficients on a system with a known overlap."""
    x, feats, _ = build(height=2.0, k=2)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x, feats)
    fixed, moving = x[:, fixed_ids], x[:, moving_ids]
    rsum = eg._radii(x, feats, moving_ids, fixed_ids)
    energy, _ = eg.candidate_energy(
        moving, fixed, rsum, epi_local, valid, para_local, k
    )
    d, *_ = eg.residue_distances(moving, fixed, epi_local, valid, para_local)
    contact = eg.contact_terms(d, k)[0]
    clash = rc.clash_terms(moving, fixed, rsum, eg.OVERLAP_FRACTION)[0]
    assert float(clash) > 0, "the geometry must overlap for this test to mean anything"
    assert float(energy) == pytest.approx(float(contact + 10.0 * clash), rel=1e-5)


def test_only_float32_and_finite_coordinates_are_accepted():
    x, feats, _ = build(height=25.0, k=2)
    for bad in (x.double(), x.bfloat16()):
        with pytest.raises(ValueError):
            eg.refine_epitope(bad, feats)
        with pytest.raises(ValueError):
            eg.search_epitope(bad, feats)
    nan = x.clone()
    moving_only = torch.where(
        feats["user_rigid_movable_atom"] & ~feats["user_epitope_paratope_atom"]
    )[0][0]
    nan[0, moving_only, 0] = float("nan")
    with pytest.raises(ValueError):
        eg.refine_epitope(nan, feats)


def test_a_degenerate_sample_does_not_change_what_another_sample_of_the_batch_does():
    """One degenerate normal in a batch must not remove the normal-based candidates of the other samples."""
    x1, feats, _ = build(height=25.0, k=2, seed=1)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = eg.groups(x1, feats)
    alone = eg.search_epitope(x1, feats)
    line = x1.clone()
    line[0, fixed_ids, :] = 0.0
    line[0, fixed_ids, 0] = (
        torch.arange(len(fixed_ids), device=x1.device).float() * 1.5
    )  # a second sample whose antigen atoms sit on one line
    mixed = eg.search_epitope(torch.cat([x1, line]), feats)
    assert float((mixed[0] - alone[0]).abs().max()) < 1e-4


def test_ties_between_epitope_residues_are_broken_by_row_order_only():
    """Two residues at the same distance: the tensor code breaks the tie by row order, and the parser makes that order canonical."""
    fixed = torch.tensor([[[6.0, 0, 0], [-6.0, 0, 0]]])
    moving = torch.tensor([[[0.0, 0, 0]]])
    valid = torch.tensor([[True], [True]])
    d, *_ = eg.residue_distances(
        moving, fixed, torch.tensor([[0], [1]]), valid, torch.tensor([0])
    )
    a = eg.contact_terms(d, 1)
    d2, *_ = eg.residue_distances(
        moving, fixed, torch.tensor([[1], [0]]), valid, torch.tensor([0])
    )
    b = eg.contact_terms(d2, 1)
    assert (
        float(a[0]) == float(b[0]) and int(a[2][0, 0]) == 0 and int(b[2][0, 0]) == 0
    )  # the FIRST ROW wins in both orders


def test_alignment_of_a_batch_in_which_only_some_rows_are_close_to_the_x_axis():
    """The GPU failure of the first final-code verification (7ua2): boolean-mask assignment of the helper axis with a few true rows."""
    a = torch.tensor(
        [
            [0.95, 0.1, 0.29],
            [0.1, 0.9, 0.42],
            [0.97, -0.2, 0.1],
            [0.2, 0.3, 0.93],
            [0.93, 0.3, 0.2],
        ],
        device=DEV,
    )
    a = a / a.norm(dim=-1, keepdim=True)
    b = -torch.tensor([[0.0, 0.0, 1.0]], device=DEV).expand_as(a)
    other = torch.tensor([[0.3, 0.2, -0.9]], device=DEV)
    for target in (b, other.expand_as(a) / other.norm()):
        r = eg._rotation_aligning(a, target)
        assert (
            float((r @ r.transpose(1, 2) - torch.eye(3, device=DEV)).abs().max()) < 1e-5
        )
        assert (
            float((torch.bmm(r, a[..., None]).squeeze(-1) - target).abs().max()) < 1e-4
        )


def _with_env(env, fn):
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return fn()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_x0_hook_is_off_unless_scheduled_and_outside_its_window():
    x, feats, _ = build(height=25.0, k=2)
    on = {"OPENDDE_RIGID_CONTACT": "on"}
    assert (
        _with_env(
            on | {"OPENDDE_RIGID_X0_START": "off"}, lambda: eg.guide_x0(x, feats, 100)
        )
        is x
    )  # turned off
    env = on | {
        "OPENDDE_RIGID_X0_START": "100",
        "OPENDDE_RIGID_X0_EVERY": "10",
        "OPENDDE_RIGID_X0_LAST": "150",
    }
    assert (
        _with_env(env, lambda: eg.guide_x0(x, feats, 105)) is x
    )  # not a scheduled step
    assert _with_env(env, lambda: eg.guide_x0(x, feats, 160)) is x  # after LAST
    assert _with_env(env, lambda: eg.guide_x0(x, feats, 90)) is x  # before START
    assert (
        _with_env(
            env | {"OPENDDE_RIGID_CONTACT": "control"},
            lambda: eg.guide_x0(x, feats, 100),
        )
        is x
    )  # only in mode 'on'
    assert (
        _with_env({"OPENDDE_RIGID_X0_START": "off"}, lambda: eg.x0_step_active(100))
        is False
    )
    assert (
        _with_env(env, lambda: eg.x0_step_active(120)) is True
        and _with_env(env, lambda: eg.x0_step_active(150)) is True
    )


def test_x0_hook_moves_the_antibody_group_rigidly_into_contact():
    x, feats, _ = build(height=25.0, k=2, samples=2)
    before, k = reached(x, feats)
    assert int(before.max()) == 0
    env = {
        "OPENDDE_RIGID_CONTACT": "on",
        "OPENDDE_RIGID_X0_START": "100",
        "OPENDDE_RIGID_X0_EVERY": "10",
        "OPENDDE_RIGID_X0_LAST": "150",
    }
    out = _with_env(env, lambda: eg.guide_x0(x, feats, 110))
    after, _ = reached(out, feats)
    assert bool((after >= k).all())
    assert not bool(severe(out, feats).any())
    mov = feats["user_rigid_movable_atom"]
    exact = "donot_use_mm_for_euclid_dist"
    for s in range(2):
        d0 = torch.cdist(x[s, mov], x[s, mov], compute_mode=exact)
        d1 = torch.cdist(out[s, mov], out[s, mov], compute_mode=exact)
        assert float((d0 - d1).abs().max()) < 1e-3
    assert torch.equal(out[:, ~mov], x[:, ~mov])
    assert out.dtype == x.dtype


def _all_paratope(x, feats):
    f = dict(feats)
    f["user_epitope_paratope_atom"] = feats["user_rigid_movable_atom"].clone()
    return f


def test_whole_antibody_mode_finds_a_pose_without_any_window():
    x, feats, _ = build(height=25.0, k=2)
    fa = _all_paratope(x, feats)
    before, k = reached(x, fa)
    assert int(before) == 0
    out = eg.search_epitope(x, fa)
    out = eg.refine_epitope(out, fa, iterations=120)
    after, _ = reached(out, fa)
    assert int(after) >= k
    assert not bool(severe(out, fa).any())
    mov = fa["user_rigid_movable_atom"]
    exact = "donot_use_mm_for_euclid_dist"
    assert (
        float(
            (
                torch.cdist(x[0, mov], x[0, mov], compute_mode=exact)
                - torch.cdist(out[0, mov], out[0, mov], compute_mode=exact)
            )
            .abs()
            .max()
        )
        < 1e-3
    )
    assert torch.equal(out[0, ~mov], x[0, ~mov])


def test_whole_antibody_mode_does_not_depend_on_the_storage_order_of_the_moving_atoms():
    x, feats, n_fixed = build(height=25.0, k=2, samples=1, rotate=True)
    fa = _all_paratope(x, feats)
    out1 = eg.refine_epitope(eg.search_epitope(x, fa), fa, iterations=60)
    m = int(feats["user_rigid_movable_atom"].sum())
    perm = torch.randperm(m, generator=torch.Generator().manual_seed(1)).to(x.device)
    order = torch.arange(x.shape[1], device=x.device)
    order[n_fixed:] = n_fixed + perm  # reverse-map: new atom i is old atom order[i]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(len(order), device=x.device)
    xp = x[:, order]
    fp = {}
    for key, v in fa.items():
        if key in (
            "user_rigid_movable_atom",
            "user_epitope_paratope_atom",
            "ref_element",
            "asym_id",
            "atom_to_token_idx",
        ):
            fp[key] = v[order]
        elif key == "user_epitope_atom_index":
            fp[key] = torch.where(v >= 0, inv[v.clamp_min(0)], v)
        else:
            fp[key] = v
    out2 = eg.refine_epitope(eg.search_epitope(xp, fp), fp, iterations=60)[:, inv]
    assert float((out1 - out2).abs().max()) < 1e-3, float((out1 - out2).abs().max())


def test_x0_hook_nudges_a_sample_that_misses_the_request_by_a_little_instead_of_re_docking_it():
    # a system that just fails the request: the block is placed so that few epitope residues are reached
    x, feats, _ = build(height=5.6, k=3)
    before, k = reached(x, feats)
    assert int(before) < k, (int(before), k)
    env = {
        "OPENDDE_RIGID_CONTACT": "on",
        "OPENDDE_RIGID_X0_START": "100",
        "OPENDDE_RIGID_X0_EVERY": "10",
        "OPENDDE_RIGID_X0_LAST": "150",
    }
    out = _with_env(env, lambda: eg.guide_x0(x, feats, 110))
    mov = feats["user_rigid_movable_atom"]
    moved = float((out[0, mov] - x[0, mov]).norm(dim=-1).mean())
    assert int(reached(out, feats)[0]) >= k
    assert moved < 3.0, moved  # a nudge, not a re-placement
    far, ffeats, _ = build(height=25.0, k=3)
    out = _with_env(env, lambda: eg.guide_x0(far, ffeats, 110))
    assert (
        int(reached(out, ffeats)[0]) >= k
    )  # a sample that is far away still gets the coarse placement
