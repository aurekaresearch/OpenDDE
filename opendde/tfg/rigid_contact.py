"""Rigid-body contact guidance: the dense reference implementation and the helpers shared with the accelerated core.

All updates are proper rigid transformations of the moving chain group. No contact atom receives an independent
displacement. A backtracking gate rejects new severe interchain overlaps; existing overlaps may only decrease, and a
sample whose contacts all lie inside their windows is left untouched.

refine_rigid_contact and search_rigid_contact run the accelerated core (rigid_core, through contact_core) when
OPENDDE_RIGID_CORE selects it and the dense implementation below otherwise.
"""

import torch

from opendde.data.constants import rdkit_vdws
from opendde.utils.logger import get_logger

logger = get_logger(__name__)


def rigid_core_mode():
    """OPENDDE_RIGID_CORE: auto (default: the accelerated core when a CUDA device and Triton are available, the dense implementation
    otherwise, without a warning), on (the same, but a warning says why the core was not used; the dense implementation is also the
    fallback on any exception), off (dense implementation), check (both; the dense result is returned and the differences are logged).

    It is read here, not in rigid_core, because rigid_core imports Triton at module level and a CPU-only install has none.
    """
    import os

    value = os.environ.get("OPENDDE_RIGID_CORE", "auto")
    if value not in ("auto", "off", "on", "check"):
        raise ValueError("OPENDDE_RIGID_CORE must be auto, off, on or check")
    return value


def triton_available():
    """True when Triton can be imported (it is absent on CPU-only and MPS installs)."""
    import importlib.util

    try:
        return importlib.util.find_spec("triton") is not None
    except (ImportError, ValueError):
        return False


SOFT_OVERLAP = 0.85
# Pair-element budget above which the clash test is evaluated in chunks.
EXACT_CLASH_BUDGET = 200_000_000


def contact_groups(coords, feats):
    """Atom indices of the group that moves, of everything held fixed, and the
    per-contact atom pair split across the two.

    An explicit ``user_rigid_movable_atom`` mask names the moving chains, which
    is how a Fab moves as one body: heavy and light chain travel together and
    every other chain, ligand or copy stays put. Without that mask exactly two
    chains are supported and the one holding the second atom of each contact moves.
    """
    idx = feats["user_distance_restraint_index"]
    mask = feats.get("user_rigid_movable_atom")
    if mask is not None and mask.numel() != 0:
        # Only an absent or empty mask may fall back; a malformed one is an error,
        # because falling back would silently move the wrong chain.
        if mask.ndim != 1 or mask.numel() != coords.shape[-2]:
            raise ValueError(
                "user_rigid_movable_atom must be a 1-D mask over %d atoms, got shape %s"
                % (coords.shape[-2], tuple(mask.shape))
            )
        mask = mask.to(torch.bool)
        if not mask.any() or mask.all():
            raise ValueError("movable_chains must name some but not all atoms.")
        moving_ids = torch.where(mask)[0]
        fixed_ids = torch.where(~mask)[0]
        left_moves = mask[idx[0]]
        right_moves = mask[idx[1]]
        if bool((left_moves == right_moves).any()):
            raise ValueError(
                "Every contact must join one atom of the moving group to one fixed atom."
            )
        mov_atoms = torch.where(left_moves, idx[0], idx[1])
        fix_atoms = torch.where(left_moves, idx[1], idx[0])
        return fixed_ids, moving_ids, fix_atoms, mov_atoms
    atom_chain = feats["asym_id"][feats["atom_to_token_idx"]]
    left = torch.unique(atom_chain[idx[0]])
    right = torch.unique(atom_chain[idx[1]])
    if left.numel() != 1 or right.numel() != 1 or left.item() == right.item():
        raise ValueError("Rigid contact requires contacts between two distinct chains.")
    if torch.unique(atom_chain).numel() != 2:
        raise ValueError(
            "Rigid contact without movable_chains supports exactly two chains."
        )
    fixed_ids = torch.where(atom_chain == left.item())[0]
    moving_ids = torch.where(atom_chain == right.item())[0]
    return fixed_ids, moving_ids, idx[0], idx[1]


def clash_terms(x, fixed, rsum, fraction, want_gradient=False):
    """Repulsion energy, severe-overlap summary and gradient against the fixed
    group, evaluated in chunks when the full pair matrix would not fit.

    The exact branch keeps the full boolean matrix. The chunked branch reduces
    per chunk and reports severe overlaps as one count per chunk plus a depth. Those counts are
    for reporting only: acceptance in the chunked regime goes through
    severe_transition, which compares the boolean sets directly.

    The energy is returned as ``sum(overlap**2)``, unweighted, and each caller
    applies its own weight. The gradient is the gradient of the energy as it
    enters ``refine_rigid_contact``, namely ``0.5 * 20 * sum(overlap**2)``; the
    coarse search uses the energy only and asks for no gradient.
    """
    samples, moving_n, _ = x.shape
    fixed_n = fixed.shape[-2]
    budget = EXACT_CLASH_BUDGET
    if samples * moving_n * fixed_n <= budget:
        dist = torch.cdist(
            x, fixed, compute_mode="donot_use_mm_for_euclid_dist"
        ).clamp_min(1e-6)
        overlap = torch.relu(fraction * rsum - dist)
        severe = dist < 0.75 * rsum
        depth = torch.relu(0.75 * rsum - dist).amax(dim=(-2, -1))
        energy = overlap.square().sum((-2, -1))
        if not want_gradient:
            return energy, severe, depth, None
        coeff = torch.where(dist > 1e-6, -20.0 * overlap / dist, torch.zeros_like(dist))
        grad = coeff.sum(-1, keepdim=True) * x - torch.bmm(coeff, fixed)
        return energy, severe, depth, grad
    step = max(1, budget // max(1, samples * moving_n))
    energy = torch.zeros(samples, device=x.device, dtype=x.dtype)
    counts = []
    depth = torch.zeros(samples, device=x.device, dtype=x.dtype)
    grad = torch.zeros_like(x) if want_gradient else None
    for start in range(0, fixed_n, step):
        stop = min(start + step, fixed_n)
        block = fixed[:, start:stop]
        radii = rsum[:, start:stop]
        dist = torch.cdist(
            x, block, compute_mode="donot_use_mm_for_euclid_dist"
        ).clamp_min(1e-6)
        overlap = torch.relu(fraction * radii - dist)
        energy = energy + overlap.square().sum((-2, -1))
        severe = dist < 0.75 * radii
        counts.append(severe.sum((-2, -1)))
        depth = torch.maximum(depth, torch.relu(0.75 * radii - dist).amax(dim=(-2, -1)))
        if want_gradient:
            coeff = torch.where(
                dist > 1e-6, -20.0 * overlap / dist, torch.zeros_like(dist)
            )
            grad = grad + coeff.sum(-1, keepdim=True) * x - torch.bmm(coeff, block)
    return energy, torch.stack(counts, dim=-1), depth, grad


def severe_transition(previous_x, current_x, fixed, rsum):
    """Per sample: did any severe overlap appear that was not there before.

    The two boolean sets are compared inside each chunk, so a pair appearing
    cannot be offset by another disappearing, whether in the same chunk or a
    different one. Used when the clash test runs chunked; the exact branch
    compares the full matrices directly.
    """
    samples, moving_n, _ = previous_x.shape
    fixed_n = fixed.shape[-2]
    step = max(1, EXACT_CLASH_BUDGET // max(1, samples * moving_n))
    ok = torch.ones(samples, device=previous_x.device, dtype=torch.bool)
    for start in range(0, fixed_n, step):
        stop = min(start + step, fixed_n)
        block = fixed[:, start:stop]
        threshold = 0.75 * rsum[:, start:stop]
        was = (
            torch.cdist(previous_x, block, compute_mode="donot_use_mm_for_euclid_dist")
            < threshold
        )
        now = (
            torch.cdist(current_x, block, compute_mode="donot_use_mm_for_euclid_dist")
            < threshold
        )
        ok &= ~(now & ~was).any(dim=(-2, -1))
    return ok


def severe_count(summary):
    """Number of severe overlaps per sample, from either summary form."""
    return summary.sum((-2, -1)) if summary.dtype == torch.bool else summary.sum(-1)


def no_new_severe(previous, current):
    """Whether no severe overlap appeared that was not there before.

    With the full matrix this is exact. The chunked summary holds one count per
    chunk of the fixed group, and every chunk must be non-increasing, so a clash
    appearing in one region cannot be hidden by one resolving in another. Within
    a single chunk the counts can still cancel, which is the residual looseness
    of the chunked branch.
    """
    if previous.dtype == torch.bool:
        return ~(current & ~previous).any(dim=(-2, -1))
    return (current <= previous).all(-1)


def late_pass_enabled():
    """OPENDDE_RIGID_LATE: "on" (default) or "off"; anything else is an error instead of a silent "on"."""
    import os

    value = os.environ.get("OPENDDE_RIGID_LATE", "on").strip().lower()
    if value not in ("on", "off"):
        raise ValueError("OPENDDE_RIGID_LATE must be on or off")
    return value == "on"


def intervention_schedule(num_steps):
    """Steps at which the coarse search and the refinement run."""
    import os

    # The default start is step 190 of 200 (scaled down when there are fewer than 191 steps); an explicit value must lie inside the run.
    try:
        start = int(os.environ.get("OPENDDE_RIGID_START", min(190, num_steps - 1)))
        every = int(os.environ.get("OPENDDE_RIGID_EVERY", "2"))
    except ValueError as exc:
        raise ValueError(
            "OPENDDE_RIGID_START / OPENDDE_RIGID_EVERY must be integers"
        ) from exc
    if not 0 <= start < num_steps or every < 1:
        raise ValueError("Invalid OPENDDE_RIGID_START / OPENDDE_RIGID_EVERY")
    last = num_steps - 1
    coarse = sorted({start, last})
    refine = sorted({i for i in range(start, num_steps) if i % every == 0} | {last})
    return coarse, refine


@torch.no_grad()
def _refine_rigid_contact_dense(coords, feats, iterations=40):
    if coords.ndim != 3:
        raise ValueError("Rigid contact pilot requires [sample, atom, xyz].")
    idx = feats["user_distance_restraint_index"]
    if idx.numel() == 0:
        return coords
    fixed_ids, moving_ids, fix_atoms, mov_atoms = contact_groups(coords, feats)
    fixed_lookup = torch.full(
        (coords.shape[-2],), -1, device=coords.device, dtype=torch.long
    )
    fixed_lookup[fixed_ids] = torch.arange(len(fixed_ids), device=coords.device)
    li = fixed_lookup[fix_atoms]
    moving_lookup = torch.full(
        (coords.shape[-2],), -1, device=coords.device, dtype=torch.long
    )
    moving_lookup[moving_ids] = torch.arange(len(moving_ids), device=coords.device)
    ri = moving_lookup[mov_atoms]
    original_dtype = coords.dtype
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float().clone()
    radii = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    atomic_number_index = feats["ref_element"].argmax(-1)
    if (atomic_number_index == 0).any():
        raise ValueError("Pilot expects heavy-atom protein input; hydrogen found.")
    rsum = (
        radii[atomic_number_index[moving_ids]][:, None]
        + radii[atomic_number_index[fixed_ids]][None, :]
    )
    lower = feats["user_distance_restraint_lower_bound"].float()
    upper = feats["user_distance_restraint_upper_bound"].float()
    entry = moving.clone()

    def evaluate(x, gradient=False):
        pair_vec = x[:, ri] - fixed[:, li]
        pair_d = torch.linalg.vector_norm(pair_vec, dim=-1).clamp_min(1e-6)
        violation = torch.relu(pair_d - upper) - torch.relu(lower - pair_d)
        clash_energy, severe, severe_depth, clash_grad = clash_terms(
            x, fixed, rsum, SOFT_OVERLAP, want_gradient=gradient
        )
        energy = 0.5 * (violation.square().sum(-1) + 20.0 * clash_energy)
        if not gradient:
            return energy, severe, severe_depth, pair_d
        grad = clash_grad
        # Below the clamp the energy is constant in the distance, so its gradient
        # is zero there rather than a direction read off a degenerate vector.
        contact_grad = violation[..., None] * pair_vec / pair_d[..., None]
        contact_grad = torch.where(
            (pair_d > 1e-6)[..., None], contact_grad, torch.zeros_like(contact_grad)
        )
        grad.index_add_(1, ri, contact_grad)
        return energy, severe, severe_depth, pair_d, grad

    first_energy, first_severe, _, first_d = evaluate(moving)
    accepted_count = torch.zeros(
        coords.shape[0], device=coords.device, dtype=torch.long
    )
    identity = torch.eye(3, device=coords.device).expand(coords.shape[0], 3, 3)
    for _ in range(iterations):
        energy, severe, depth, pair_d, grad = evaluate(moving, True)
        satisfied = ((pair_d >= lower - 1e-6) & (pair_d <= upper + 1e-6)).all(-1)
        center = moving.mean(1, keepdim=True)
        centered = moving - center
        translation = -0.2 * grad.sum(1) / idx.shape[1]
        translation *= 0.5 / torch.linalg.vector_norm(
            translation, dim=-1, keepdim=True
        ).clamp_min(0.5)
        torque = torch.linalg.cross(centered, grad, dim=-1).sum(1)
        inertia = centered.square().sum((1, 2))[:, None, None] * identity - torch.bmm(
            centered.transpose(1, 2), centered
        )
        rotation = (
            -0.15
            * len(moving_ids)
            / idx.shape[1]
            * torch.linalg.solve(inertia + 1e-3 * identity, torque[..., None]).squeeze(
                -1
            )
        )
        rotation *= 0.03 / torch.linalg.vector_norm(
            rotation, dim=-1, keepdim=True
        ).clamp_min(0.03)
        found = torch.zeros(coords.shape[0], device=coords.device, dtype=torch.bool)
        next_coords = moving.clone()
        for backtrack in range(10):
            scale = 0.5**backtrack
            w = rotation * scale
            skew = torch.zeros_like(identity)
            skew[:, 0, 1], skew[:, 0, 2] = -w[:, 2], w[:, 1]
            skew[:, 1, 0], skew[:, 1, 2] = w[:, 2], -w[:, 0]
            skew[:, 2, 0], skew[:, 2, 1] = -w[:, 1], w[:, 0]
            matrix = torch.linalg.matrix_exp(skew)
            proposal = (
                torch.bmm(centered, matrix.transpose(1, 2))
                + center
                + translation[:, None] * scale
            )
            new_energy, new_severe, new_depth, _ = evaluate(proposal)
            if severe.dtype == torch.bool:
                no_new_clash = no_new_severe(severe, new_severe)
            else:
                no_new_clash = severe_transition(moving, proposal, fixed, rsum)
            accept = (
                (~found)
                & (~satisfied)
                & no_new_clash
                & (new_depth <= depth + 1e-6)
                & (new_energy < energy - 1e-7)
            )
            next_coords[accept] = proposal[accept]
            found |= accept
            if found.all():
                break
        if not found.any():
            break
        moving = next_coords
        accepted_count += found.long()
    final_energy, final_severe, _, final_d = evaluate(moving)
    intact = (
        no_new_severe(first_severe, final_severe)
        if first_severe.dtype == torch.bool
        else severe_transition(entry, moving, fixed, rsum)
    )
    if not bool(intact.all()):
        raise RuntimeError("Rigid contact introduced a severe interchain clash.")
    result = coords.clone()
    result[:, moving_ids] = moving.to(original_dtype)
    logger.info(
        "RIGID_CONTACT energy %s -> %s; mean_distance %s -> %s; severe_pairs %s -> %s; accepted %s",
        first_energy.tolist(),
        final_energy.tolist(),
        first_d.mean(-1).tolist(),
        final_d.mean(-1).tolist(),
        severe_count(first_severe).tolist(),
        severe_count(final_severe).tolist(),
        accepted_count.tolist(),
    )
    return result


@torch.no_grad()
def _search_rigid_contact_dense(coords, feats):
    """Deterministic coarse pose search using only the supplied contact pairs.

    Tests whole-chain poses on a sphere around the requested epitope. Candidates
    with any severe interchain overlap are rejected. No native structure or
    confidence score is used to choose a pose.
    """
    import math

    fi, mi, fix_atoms, mov_atoms = contact_groups(coords, feats)
    fixed = coords[:, fi].float()
    moving = coords[:, mi].float()
    source = coords[:, mov_atoms].float()
    target = coords[:, fix_atoms].float()
    sc = source.mean(1, keepdim=True)
    tc = target.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(torch.bmm((source - sc).transpose(1, 2), target - tc))
    diag = torch.eye(3, device=coords.device).expand(len(coords), 3, 3).clone()
    diag[:, 2, 2] = torch.det(torch.bmm(u, vh))
    fit = torch.bmm(torch.bmm(u, diag), vh)
    fitted = torch.bmm(moving - sc, fit)
    fitted_contacts = torch.bmm(source - sc, fit)
    outward = (tc - fixed.mean(1, keepdim=True)).squeeze(1)
    outward /= torch.linalg.vector_norm(outward, dim=-1, keepdim=True).clamp_min(1e-6)
    radii = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    r = radii[feats["ref_element"].argmax(-1)]
    rsum = r[mi][:, None] + r[fi][None, :]
    lower = feats["user_distance_restraint_lower_bound"]
    upper = feats["user_distance_restraint_upper_bound"]

    def score(x, contact):
        distance = torch.linalg.vector_norm(contact - target, dim=-1)
        error = torch.relu(distance - upper) + torch.relu(lower - distance)
        clash_energy, summary, _, _ = clash_terms(x, fixed, rsum, SOFT_OVERLAP)
        severe = severe_count(summary) > 0
        energy = 0.5 * error.square().sum(-1) + 10 * clash_energy
        return energy, severe

    best = moving.clone()
    best_energy, bad = score(moving, source)
    best_energy = torch.where(
        bad, torch.full_like(best_energy, float("inf")), best_energy
    )
    # A pose already inside every requested window is left untouched.
    entry_distance = torch.linalg.vector_norm(
        coords[:, mov_atoms].float() - target, dim=-1
    )
    satisfied = (
        (entry_distance >= lower - 1e-6) & (entry_distance <= upper + 1e-6)
    ).all(-1)
    normals = [outward]
    for i in range(24):
        z = 1 - 2 * (i + 0.5) / 24
        phi = i * math.pi * (3 - math.sqrt(5))
        rad = math.sqrt(1 - z * z)
        normals.append(
            torch.tensor(
                [rad * math.cos(phi), rad * math.sin(phi), z], device=coords.device
            ).expand(len(coords), 3)
        )
    accepted = torch.zeros(len(coords), device=coords.device, dtype=torch.long)
    feasible = torch.zeros(len(coords), device=coords.device, dtype=torch.long)
    improving = torch.zeros(len(coords), device=coords.device, dtype=torch.long)
    tested = 0
    for normal in normals:
        for turn in [i * math.pi / 6 for i in range(12)]:
            w = normal * turn
            skew = torch.zeros_like(fit)
            skew[:, 0, 1], skew[:, 0, 2] = -w[:, 2], w[:, 1]
            skew[:, 1, 0], skew[:, 1, 2] = w[:, 2], -w[:, 0]
            skew[:, 2, 0], skew[:, 2, 1] = -w[:, 1], w[:, 0]
            rotation = torch.linalg.matrix_exp(skew)
            rotated = torch.bmm(fitted, rotation.transpose(1, 2))
            rot_contacts = torch.bmm(fitted_contacts, rotation.transpose(1, 2))
            for radius in [5.5, 7.5, 10.0, 15.0, 20.0]:
                shift = tc + radius * normal[:, None]
                proposed = rotated + shift
                proposed_contacts = rot_contacts + shift
                energy, bad = score(proposed, proposed_contacts)
                tested += 1
                feasible += (~bad).long()
                improving += (energy < best_energy).long()
                take = (~bad) & (~satisfied) & (energy < best_energy)
                best[take] = proposed[take]
                best_energy = torch.where(take, energy, best_energy)
                accepted += take.long()
    result = coords.clone()
    result[:, mi] = best.to(coords.dtype)
    # accepted counts refreshes of the best pose. feasible and improving separate
    # the two reasons a candidate is turned down, so a zero cannot be read as
    # evidence that the alternative site was occupied.
    logger.info(
        "RIGID_CONTACT coarse search satisfied=%s candidates=%d feasible=%s improving=%s accepted=%s final_energy=%s",
        satisfied.tolist(),
        tested,
        feasible.tolist(),
        improving.tolist(),
        accepted.tolist(),
        best_energy.tolist(),
    )
    return result


@torch.no_grad()
def refine_rigid_contact(coords, feats, iterations=40):
    """Rigid refinement; OPENDDE_RIGID_CORE selects the accelerated core (see contact_core.py)."""
    import sys
    from opendde.tfg import contact_core

    if coords.ndim != 3:
        raise ValueError("Rigid contact pilot requires [sample, atom, xyz].")
    if feats["user_distance_restraint_index"].numel() == 0:
        return coords
    return contact_core.run(
        "refine",
        coords,
        feats,
        sys.modules[__name__],
        _refine_rigid_contact_dense,
        iterations=iterations,
    )


@torch.no_grad()
def search_rigid_contact(coords, feats):
    """Coarse pose search; OPENDDE_RIGID_CORE selects the accelerated core (see contact_core.py)."""
    import sys
    from opendde.tfg import contact_core

    return contact_core.run(
        "search", coords, feats, sys.modules[__name__], _search_rigid_contact_dense
    )
