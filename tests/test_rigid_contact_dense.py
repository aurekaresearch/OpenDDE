"""Rigid-body contact guidance through the dense (pure torch) implementation, on CPU.

This is the path a machine without Triton or CUDA takes: ``OPENDDE_RIGID_CORE`` unset. The synthetic system is
a fixed antigen slab (chain 0) and a movable antibody block (chain 1) joined by three contact pairs.
"""

import pytest
import torch

from opendde.tfg import rigid_contact as rc

CARBON = 6
LOWER, UPPER = 3.5, 8.0
TOLERANCE = 0.25  # Å left over after the 40 refinement iterations


@pytest.fixture(autouse=True)
def dense_path(monkeypatch):
    monkeypatch.delenv("OPENDDE_RIGID_CORE", raising=False)


def system(height, tilt=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = torch.arange(-13.5, 14.0, 3.0)
    grid = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), -1).reshape(-1, 2)
    antigen = torch.cat([grid, torch.zeros(len(grid), 1)], -1)
    bx = torch.arange(-8.75, 9.0, 3.5)
    plate = torch.stack(torch.meshgrid(bx, bx, indexing="ij"), -1).reshape(-1, 2)
    body = torch.cat(
        [torch.cat([plate, torch.full((len(plate), 1), 3.5 * i)], -1) for i in range(3)]
    )
    body = body + torch.tensor([0.0, 0.0, height])
    if tilt:
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
        if torch.det(q) < 0:
            q[:, 0] *= -1
        body = (body - body.mean(0)) @ q.T + body.mean(0) + torch.tensor([6.0, -4.0, 0])
    n_fixed, n_moving = len(antigen), len(body)
    coords = torch.cat([antigen, body]).unsqueeze(0)
    atoms = n_fixed + n_moving
    movable = torch.zeros(atoms, dtype=torch.bool)
    movable[n_fixed:] = True
    centre = [
        i
        for i in range(n_fixed)
        if abs(float(antigen[i, 0])) < 5 and abs(float(antigen[i, 1])) < 5
    ]
    fixed_side = torch.tensor(centre[:3])
    moving_side = n_fixed + torch.tensor(
        [i for i in range(len(plate)) if abs(float(plate[i, 0])) < 5][:3]
    )
    element = torch.zeros(atoms, 128)
    element[:, CARBON] = 1
    feats = {
        "user_distance_restraint_index": torch.stack([fixed_side, moving_side]),
        "user_distance_restraint_lower_bound": torch.full((3,), LOWER),
        "user_distance_restraint_upper_bound": torch.full((3,), UPPER),
        "user_rigid_movable_atom": movable,
        "ref_element": element,
        "asym_id": movable.long(),
        "atom_to_token_idx": torch.arange(atoms),
    }
    return coords, feats, n_fixed


def internal_distances(x):
    return torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")


def pair_distances(x, feats):
    idx = feats["user_distance_restraint_index"]
    return torch.linalg.vector_norm(x[:, idx[0]] - x[:, idx[1]], dim=-1)


def test_refinement_pulls_the_body_into_the_window_rigidly():
    x, feats, n_fixed = system(height=12.0)
    assert bool((pair_distances(x, feats) > UPPER).all())
    out = rc.refine_rigid_contact(x, feats)
    assert bool((pair_distances(out, feats) <= UPPER + TOLERANCE).all())
    assert bool((pair_distances(out, feats) >= LOWER - TOLERANCE).all())
    assert torch.equal(out[:, :n_fixed], x[:, :n_fixed])
    assert torch.allclose(
        internal_distances(x[:, n_fixed:]),
        internal_distances(out[:, n_fixed:]),
        atol=1e-3,
    )


def test_a_satisfied_sample_is_left_untouched():
    x, feats, _ = system(height=3.6)
    assert bool((pair_distances(x, feats) <= UPPER).all())
    assert torch.equal(rc.refine_rigid_contact(x, feats), x)


def test_search_then_refinement_docks_a_far_tilted_body():
    x, feats, n_fixed = system(height=30.0, tilt=True)
    found = rc.search_rigid_contact(x, feats)
    assert torch.equal(found[:, :n_fixed], x[:, :n_fixed])
    out = rc.refine_rigid_contact(found, feats)
    assert bool((pair_distances(out, feats) <= UPPER + TOLERANCE).all())
    assert torch.allclose(
        internal_distances(x[:, n_fixed:]),
        internal_distances(out[:, n_fixed:]),
        atol=1e-3,
    )


def test_the_dense_path_logs_no_fallback_and_needs_no_triton(caplog):
    x, feats, _ = system(height=12.0)
    rc.refine_rigid_contact(x, feats)
    assert "fallback" not in caplog.text.lower()
