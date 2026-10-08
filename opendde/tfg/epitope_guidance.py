"""Epitope-only rigid guidance: move the antibody as one body until at least K of the requested
antigen residues are in heavy-atom contact with its binding face.

The request is what a user has when only the antigen epitope is known (typically from a patent):
antigen residues, no antibody-side residue and no pairing. Nothing is read from a deposited structure and
no antibody pose is assumed; every geometric quantity comes from the coordinates the sampler holds.

  reached   an epitope residue is reached when a heavy atom of it is within GATE of a heavy atom of the paratope
            (the paratope is the set of heavy atoms of the binding-face residues of the moving chains)
  request   at least K of the n distinct epitope residues are reached, K = max(1, ceil(min_fraction * n))
  energy    0.5 * sum over the K epitope residues currently closest to the paratope of relu(d_e - TARGET)^2,
            plus the soft clash term of rigid_contact weighted 10 in both the refinement and the coarse search;
            TARGET sits inside the gate so that a pose that just satisfies the energy stays inside it after small changes
  updates   proper rigid transformations of the moving group only (the same force and torque reduction, step clipping and
            backtracking as rigid_contact); a move is refused if it adds a severe clash or does not lower the energy
  skip      a sample that already meets the request is not touched

The grids and constants below are fixed by design.
Status per sample: already_satisfied_skipped, delivered, partial, search_failed.
"""

import math
import os

import torch

from opendde.data.constants import rdkit_vdws
from opendde.tfg import contact_core, rigid_contact as rc
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

GATE = 5.0
TARGET = 4.5
CLASH_WEIGHT_REFINE = 20.0
CLASH_WEIGHT_SEARCH = 10.0
OVERLAP_FRACTION = 0.85
NORMAL_RADIUS = 12.0
TILT_DEGREES = 15.0
TILT_AZIMUTHS = 12
FIBONACCI_AXES = 24
SPINS = 24
STANDOFFS = (2.0, 4.0, 6.0, 9.0, 12.0)
TIP_FRACTION = 0.1  # the anchor is the mean of the TIP_FRACTION of paratope atoms that reach furthest along the binding-face direction


def _traced(fn):
    """Log the full traceback of a failure inside the guidance before it propagates: the sampler wrapper keeps only the message."""
    import functools, traceback

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.error(
                "EPITOPE_GUIDANCE failure in %s:\n%s",
                fn.__name__,
                traceback.format_exc(),
            )
            raise

    return wrapper


def active(feats):
    """True when the request carries an epitope."""
    return feats.get("user_epitope_atom_index") is not None


def rigid_mode(feats, tfg_enabled=True):
    """The OPENDDE_RIGID_CONTACT mode in force for this input: "on", "off" or "control".

    Unset or "auto" (the default): "on" when TFG is enabled and the input carries an epitope request, or contact pairs that rigid guidance
    can apply (``movable_chains`` is given, or the complex has exactly two chains); "off" otherwise, so a contact request on any other
    complex keeps the atom-level restraint and an input without a constraint is not touched. "on", "off" and "control" are taken as given.
    """
    value = os.environ.get("OPENDDE_RIGID_CONTACT", "auto")
    if value not in {"auto", "off", "control", "on"}:
        raise ValueError("Invalid OPENDDE_RIGID_CONTACT mode")
    if value != "auto":
        return value
    if not tfg_enabled:
        return "off"
    if active(feats):
        return "on"
    pairs = feats.get("user_distance_restraint_index")
    if pairs is None or pairs.numel() == 0:
        return "off"
    movable = feats.get("user_rigid_movable_atom")
    if movable is not None and movable.numel() != 0:
        return "on"
    chains = feats["asym_id"][feats["atom_to_token_idx"]]
    return "on" if torch.unique(chains).numel() == 2 else "off"


def _guard(coords):
    if coords.dtype != torch.float32:
        raise ValueError(
            "epitope guidance requires float32 coordinates, got %s: the request is checked on the returned coordinates and a cast can move an atom across the gate."
            % coords.dtype
        )
    if not bool(torch.isfinite(coords).all()):
        raise ValueError(
            "non-finite coordinates reached the epitope guidance; this is a failure of the sampler, not an unreachable request."
        )


def groups(coords, feats):
    """Atom-index bookkeeping shared by search and refinement."""
    mask = feats.get("user_rigid_movable_atom")
    if mask is None or mask.numel() != coords.shape[-2] or mask.ndim != 1:
        raise ValueError(
            "epitope guidance needs user_rigid_movable_atom over all %d atoms."
            % coords.shape[-2]
        )
    mask = mask.to(torch.bool).to(coords.device)
    if not mask.any() or mask.all():
        raise ValueError("movable_chains must name some but not all atoms.")
    moving_ids = torch.where(mask)[0]
    fixed_ids = torch.where(~mask)[0]
    n_atom = coords.shape[-2]
    fixed_lookup = torch.full((n_atom,), -1, device=coords.device, dtype=torch.long)
    fixed_lookup[fixed_ids] = torch.arange(len(fixed_ids), device=coords.device)
    moving_lookup = torch.full((n_atom,), -1, device=coords.device, dtype=torch.long)
    moving_lookup[moving_ids] = torch.arange(len(moving_ids), device=coords.device)
    epi = feats["user_epitope_atom_index"].to(coords.device)
    valid = epi >= 0
    epi_local = torch.where(
        valid, fixed_lookup[epi.clamp_min(0)], torch.full_like(epi, -1)
    )
    if bool((epi_local[valid] < 0).any()):
        raise ValueError("an epitope atom lies in the movable group.")
    para_ids = torch.where(feats["user_epitope_paratope_atom"].to(coords.device))[0]
    para_local = moving_lookup[para_ids]
    if para_local.numel() == 0 or bool((para_local < 0).any()):
        raise ValueError(
            "the paratope must be a non-empty subset of the movable group."
        )
    k = int(feats["user_epitope_k"].reshape(-1)[0])
    if not 1 <= k <= epi.shape[0]:
        raise ValueError("K=%d is outside 1..%d." % (k, epi.shape[0]))
    return fixed_ids, moving_ids, epi_local, valid, para_local, k


def residue_distances(moving, fixed, epi_local, valid, para_local):
    """Per sample and epitope residue: the smallest heavy-atom distance to the paratope, and the pair that gives it.

    Returns (d [S, n], fixed slot [S, n] in 0..m-1, paratope slot [S, n] in 0..P-1)."""
    s, n, m = moving.shape[0], epi_local.shape[0], epi_local.shape[1]
    fixed_epi = fixed[:, epi_local.clamp_min(0).reshape(-1)].reshape(s, n, m, 3)
    para = moving[:, para_local]
    dist = torch.cdist(
        fixed_epi.reshape(s, n * m, 3),
        para,
        compute_mode="donot_use_mm_for_euclid_dist",
    ).reshape(s, n, m, -1)
    dist = dist.masked_fill(~valid[None, :, :, None], float("inf"))
    dmin, arg = dist.reshape(s, n, -1).min(-1)
    width = para.shape[1]
    return dmin, arg // width, arg % width, fixed_epi


def reached_count(d):
    return (d <= GATE).sum(-1)


def contact_terms(d, k):
    """Energy over the K closest epitope residues and the violation each contributes."""
    ordered, order = torch.sort(
        d, dim=-1, stable=True
    )  # ties keep the (canonical) row order, so the list order never matters
    vals, idx = ordered[:, :k], order[:, :k]
    violation = torch.relu(vals - TARGET)
    return 0.5 * violation.square().sum(-1), violation, idx


def contact_energy_and_gradient(x, fixed, epi_local, valid, para_local, k):
    """Contact energy, its gradient with respect to the moving atoms [S, M, 3] and the residue distances [S, n].

    Every counted epitope residue pulls its nearest paratope atom towards its own nearest atom with the force
    relu(d - TARGET) along the pair; several residues that share a paratope atom add up (index_add, never assignment)."""
    samples, n_moving = x.shape[0], x.shape[1]
    d, a_slot, p_slot, fixed_epi = residue_distances(
        x, fixed, epi_local, valid, para_local
    )
    energy, violation, top = contact_terms(d, k)
    rows = torch.arange(samples, device=x.device)[:, None].expand(-1, k)
    fa = fixed_epi[rows, top, a_slot.gather(1, top)]
    mp = para_local[p_slot.gather(1, top)]
    vec = x[rows, mp] - fa
    dist = torch.linalg.vector_norm(vec, dim=-1).clamp_min(1e-6)
    contrib = violation[..., None] * vec / dist[..., None]
    grad = torch.zeros_like(x)
    grad.view(-1, 3).index_add_(
        0, (rows * n_moving + mp).reshape(-1), contrib.reshape(-1, 3)
    )
    return energy, grad, d


def _status(entry_ok, final_ok, moved):
    out = []
    for e, f, m in zip(entry_ok.tolist(), final_ok.tolist(), moved.tolist()):
        out.append(
            "already_satisfied_skipped"
            if e
            else ("delivered" if f else ("partial" if m else "search_failed"))
        )
    return out


def _radii(coords, feats, moving_ids, fixed_ids):
    table = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    element = feats["ref_element"].argmax(-1)
    if (element == 0).any():
        raise ValueError(
            "epitope guidance expects heavy-atom input with known elements; an atom with element index 0 (unknown or placeholder) was found."
        )
    return table[element[moving_ids]][:, None] + table[element[fixed_ids]][None, :]


def _refine_epitope_dense(coords, feats, iterations=40):
    """Dense implementation of the epitope refinement: torch.cdist evaluation of every term, on any device."""
    if coords.ndim != 3:
        raise ValueError("epitope guidance requires [sample, atom, xyz].")
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    original_dtype = coords.dtype
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float().clone()
    rsum = _radii(coords, feats, moving_ids, fixed_ids)
    samples = coords.shape[0]
    n_moving = len(moving_ids)

    def evaluate(x, gradient=False):
        contact_energy, contact_grad, d = contact_energy_and_gradient(
            x, fixed, epi_local, valid, para_local, k
        )
        clash_energy, severe, depth, clash_grad = rc.clash_terms(
            x, fixed, rsum, OVERLAP_FRACTION, want_gradient=gradient
        )
        energy = contact_energy + 0.5 * CLASH_WEIGHT_REFINE * clash_energy
        if not gradient:
            return energy, severe, depth, d
        return energy, severe, depth, d, clash_grad + contact_grad

    first_energy, first_severe, _, first_d = evaluate(moving)
    entry_ok = reached_count(first_d) >= k
    accepted = torch.zeros(samples, device=coords.device, dtype=torch.long)
    identity = torch.eye(3, device=coords.device).expand(samples, 3, 3)
    entry = moving.clone()
    for _ in range(iterations):
        energy, severe, depth, d, grad = evaluate(moving, True)
        satisfied = reached_count(d) >= k
        center = moving.mean(1, keepdim=True)
        centered = moving - center
        translation = -0.2 * grad.sum(1) / k
        translation *= 0.5 / torch.linalg.vector_norm(
            translation, dim=-1, keepdim=True
        ).clamp_min(0.5)
        torque = torch.linalg.cross(centered, grad, dim=-1).sum(1)
        inertia = centered.square().sum((1, 2))[:, None, None] * identity - torch.bmm(
            centered.transpose(1, 2), centered
        )
        rotation = (
            -0.15
            * n_moving
            / k
            * torch.linalg.solve(inertia + 1e-3 * identity, torque[..., None]).squeeze(
                -1
            )
        )
        rotation *= 0.03 / torch.linalg.vector_norm(
            rotation, dim=-1, keepdim=True
        ).clamp_min(0.03)
        found = torch.zeros(samples, device=coords.device, dtype=torch.bool)
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
                no_new_clash = rc.no_new_severe(severe, new_severe)
            else:
                no_new_clash = rc.severe_transition(moving, proposal, fixed, rsum)
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
        accepted += found.long()
    final_energy, final_severe, _, final_d = evaluate(moving)
    intact = (
        rc.no_new_severe(first_severe, final_severe)
        if first_severe.dtype == torch.bool
        else rc.severe_transition(entry, moving, fixed, rsum)
    )
    if not bool(intact.all()):
        raise RuntimeError("Epitope guidance introduced a severe interchain clash.")
    result = coords.clone()
    result[:, moving_ids] = moving.to(original_dtype)
    # the status is read off the coordinates that are returned, not the internal state
    returned_d, _, _, _ = residue_distances(
        result[:, moving_ids].float(),
        result[:, fixed_ids].float(),
        epi_local,
        valid,
        para_local,
    )
    final_d = returned_d
    final_ok = reached_count(final_d) >= k
    logger.info(
        "EPITOPE_GUIDANCE refine K=%d n=%d reached_before=%s reached_after=%s energy %s -> %s status=%s accepted=%s",
        k,
        epi_local.shape[0],
        reached_count(first_d).tolist(),
        reached_count(final_d).tolist(),
        first_energy.tolist(),
        final_energy.tolist(),
        _status(entry_ok, final_ok, accepted > 0),
        accepted.tolist(),
    )
    return result


def _helper_axis(v):
    """A coordinate axis that is not (nearly) parallel to each row of v [S,3]: x unless |v_x| > 0.9, then y. Selected with torch.where
    rather than boolean-mask assignment."""
    x = torch.tensor([1.0, 0.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    y = torch.tensor([0.0, 1.0, 0.0], device=v.device, dtype=v.dtype).expand_as(v)
    return torch.where((v[:, 0].abs() > 0.9)[:, None], y, x)


def _rotation_aligning(a, b):
    """Rotation matrices [S,3,3] taking unit vectors a to unit vectors b, exactly orthogonal with determinant 1.

    Built as a rotation about the unit axis a x b by atan2(|a x b|, a . b), in double precision. The closed form with 1/(1 + a.b)
    loses orthogonality when a.b is close to -1 (determinant 0.948 at a 0.1 degree deviation from antiparallel), so it is not
    used; parallel and antiparallel vectors, where the axis is undefined, are handled explicitly."""
    a64, b64 = a.double(), b.double()
    a64 = a64 / torch.linalg.vector_norm(a64, dim=-1, keepdim=True).clamp_min(1e-12)
    b64 = b64 / torch.linalg.vector_norm(b64, dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.linalg.cross(a64, b64, dim=-1)
    s_ = torch.linalg.vector_norm(v, dim=-1)
    c = (a64 * b64).sum(-1)
    theta = torch.atan2(s_, c)
    helper = _helper_axis(a64)
    perp = torch.linalg.cross(a64, helper, dim=-1)
    perp = perp / torch.linalg.vector_norm(perp, dim=-1, keepdim=True).clamp_min(1e-12)
    degenerate = s_ < 1e-9  # parallel (theta = 0) or antiparallel (theta = pi)
    axis = torch.where(degenerate[:, None], perp, v / s_.clamp_min(1e-300)[:, None])
    theta = torch.where(
        degenerate,
        torch.where(c < 0, torch.full_like(theta, math.pi), torch.zeros_like(theta)),
        theta,
    )
    return _rotation_about(axis, theta).to(a.dtype)


def _rotation_about(axis, angle):
    """Rotation matrices [S,3,3] about unit axes [S,3] by angles [S] (or scalar); computed in double precision."""
    dtype_out = axis.dtype
    axis = axis.double()
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    s = axis.shape[0]
    angle = torch.as_tensor(angle, device=axis.device, dtype=torch.float64).expand(s)
    skew = torch.zeros(s, 3, 3, device=axis.device, dtype=torch.float64)
    skew[:, 0, 1], skew[:, 0, 2] = -axis[:, 2], axis[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = axis[:, 2], -axis[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -axis[:, 1], axis[:, 0]
    eye = torch.eye(3, device=axis.device, dtype=torch.float64).expand(s, 3, 3)
    rot = (
        eye
        + torch.sin(angle)[:, None, None] * skew
        + (1 - torch.cos(angle))[:, None, None] * torch.bmm(skew, skew)
    )
    return rot.to(dtype_out)


def approach_axes(fixed, fixed_chain, epi_local, valid, centroid):
    """Unit approach directions [A, S, 3] pointing away from the antigen at the epitope, and a note on what was used.

    For an epitope on ONE chain: the outward local normal (smallest principal axis of the heavy atoms of that chain within NORMAL_RADIUS of the
    epitope centroid, oriented away from the chain centroid) plus TILT_AZIMUTHS tilts of TILT_DEGREES around it; a sample whose normal is degenerate
    uses the chain-centroid-to-epitope direction (or +z) in its place, per sample, so a sample never depends on the others in its batch.
    FIBONACCI_AXES Fibonacci-sphere axes are always added. An epitope on several chains gets only the Fibonacci axes."""
    s = fixed.shape[0]
    device, dtype = fixed.device, fixed.dtype
    axes = []
    note = "fibonacci_only"
    slot_atoms = epi_local[valid]
    chains = torch.unique(fixed_chain[slot_atoms])
    if chains.numel() == 1:
        pool = fixed[:, fixed_chain == chains[0]]
        near = (
            torch.linalg.vector_norm(pool - centroid[:, None], dim=-1) <= NORMAL_RADIUS
        )
        ok = near.sum(-1) >= 10
        w = near.to(dtype)
        cnt = w.sum(-1, keepdim=True).clamp_min(1.0)
        mean = (pool * w[..., None]).sum(1) / cnt
        diff = (pool - mean[:, None]) * w[..., None]
        cov = torch.bmm(diff.transpose(1, 2), diff) / cnt[..., None]
        evals, evecs = torch.linalg.eigh(cov)
        normal = evecs[:, :, 0]
        outward = centroid - pool.mean(1)
        sign = torch.sign((normal * outward).sum(-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        normal = normal * sign[:, None]
        finite = torch.isfinite(normal).all(-1)
        ok = ok & finite & (evals[:, 1] > 1e-6)
        # per-sample fallback: a sample whose local normal is degenerate uses the direction from the chain centroid to the epitope,
        # (or +z if that fails too); the other samples of the batch keep their own normals, so a sample never depends on the others
        fallback = outward / torch.linalg.vector_norm(
            outward, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        bad_fb = ~torch.isfinite(fallback).all(-1) | (
            torch.linalg.vector_norm(outward, dim=-1) < 1e-6
        )
        fallback = torch.where(
            bad_fb[:, None],
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(
                fallback
            ),
            fallback,
        )
        base = torch.where(
            ok[:, None],
            normal
            / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-9),
            fallback,
        )
        helper = _helper_axis(base)
        e1 = torch.linalg.cross(base, helper, dim=-1)
        e1 = e1 / torch.linalg.vector_norm(e1, dim=-1, keepdim=True).clamp_min(1e-9)
        e2 = torch.linalg.cross(base, e1, dim=-1)
        axes.append(base)
        theta = math.radians(TILT_DEGREES)
        for i in range(TILT_AZIMUTHS):
            phi = 2.0 * math.pi * i / TILT_AZIMUTHS
            axes.append(
                math.cos(theta) * base
                + math.sin(theta) * (math.cos(phi) * e1 + math.sin(phi) * e2)
            )
        note = (
            "normal+tilts+fibonacci"
            if bool(ok.all())
            else "normal+tilts+fibonacci(fallback_normal_for_samples=%s)"
            % torch.where(~ok)[0].tolist()
        )
    else:
        note = "fibonacci_only(cross_chain_epitope)"
    for i in range(FIBONACCI_AXES):
        z = 1 - 2 * (i + 0.5) / FIBONACCI_AXES
        phi = i * math.pi * (3 - math.sqrt(5))
        r = math.sqrt(1 - z * z)
        axes.append(
            torch.tensor(
                [r * math.cos(phi), r * math.sin(phi), z], device=device, dtype=dtype
            ).expand(s, 3)
        )
    return axes, note


def candidate_energy(x, fixed, rsum, epi_local, valid, para_local, k):
    """Objective of the coarse search: contact energy (0.5 * sum) + CLASH_WEIGHT_SEARCH * clash energy;
    returns (energy [S], has_severe_clash [S])."""
    d, _, _, _ = residue_distances(x, fixed, epi_local, valid, para_local)
    contact_energy, _, _ = contact_terms(d, k)
    clash_energy, summary, _, _ = rc.clash_terms(x, fixed, rsum, OVERLAP_FRACTION)
    return contact_energy + CLASH_WEIGHT_SEARCH * clash_energy, rc.severe_count(
        summary
    ) > 0


PATCH_FRACTION = 0.10
PATCH_MINIMUM = 100


def _interface_patch(moving, fixed, fraction=PATCH_FRACTION, minimum=PATCH_MINIMUM):
    """The antibody atoms that face the antigen in the current pose: per sample, the `n` moving atoms closest to any fixed atom (n = max(minimum, fraction * atoms), the same for every sample).
    With the paratope defined as the whole antibody this is what defines the binding face and the tip; it needs no numbering scheme. Returns [S, n, 3]."""
    s_, m_ = moving.shape[0], moving.shape[1]
    n = min(m_, max(minimum, int(math.ceil(fraction * m_))))
    out = []
    for i in range(s_):
        d = (
            torch.cdist(
                moving[i], fixed[i], compute_mode="donot_use_mm_for_euclid_dist"
            )
            .min(-1)
            .values
        )  # [M]
        idx = torch.topk(-d, n, dim=-1).indices
        out.append(moving[i, idx])
    return torch.stack(out)


def _interface_patch_fast(
    moving, fixed, fraction=PATCH_FRACTION, minimum=PATCH_MINIMUM
):
    """_interface_patch with the nearest-fixed-atom distances from a fused kernel (values bit-identical to the cdist(...).min of
    _interface_patch, so the same topk selection); the topk is taken per sample."""
    from opendde.tfg import group_contact as _fg

    s_, m_ = moving.shape[0], moving.shape[1]
    n = min(m_, max(minimum, int(math.ceil(fraction * m_))))
    d_all = _fg.nearest_fixed_distance(moving, fixed)  # [S, M]
    out = []
    for i in range(s_):
        idx = torch.topk(-d_all[i], n, dim=-1).indices
        out.append(moving[i, idx])
    return torch.stack(out)


def _radii_split(coords, feats, moving_ids, fixed_ids):
    """(ra [M], rb [N]) with rsum == ra[:, None] + rb[None, :] exactly as in _radii."""
    table = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    element = feats["ref_element"].argmax(-1)
    return table[element[moving_ids]], table[element[fixed_ids]]


@torch.no_grad()
def _search_epitope_dense(coords, feats):
    """Coarse pose search from the epitope geometry alone (no crystal pose, no pairing).

    Each candidate turns the binding face of the moving group (the direction from the group centre to the paratope centre) towards the epitope,
    spins it about the approach axis and places the paratope TIP (the mean of the TIP_FRACTION of paratope atoms that reach furthest along the
    binding-face direction, a point on the antibody surface) at STANDOFFS from the epitope centre along that axis. The paratope centre itself
    is not used as the anchor: for a paratope spread over both chains it lies inside the antibody, and a placement 2 to 12 A from the epitope
    would bury the whole antibody in the antigen.
    A candidate with a severe clash is dropped; the best remaining one replaces the current pose only if it has a lower energy."""
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float()
    samples = coords.shape[0]
    rsum = _radii(coords, feats, moving_ids, fixed_ids)
    asym = feats["asym_id"][feats["atom_to_token_idx"]].to(coords.device)
    fixed_chain = asym[fixed_ids]
    # entry state
    d0, _, _, fixed_epi = residue_distances(moving, fixed, epi_local, valid, para_local)
    entry_ok = reached_count(d0) >= k
    if bool(entry_ok.all()):
        # every sample already meets the request: nothing may move, so the candidate loop is skipped (the result is the input)
        logger.info(
            "EPITOPE_GUIDANCE coarse skipped: every sample already meets the request K=%d n=%d reached=%s",
            k,
            epi_local.shape[0],
            reached_count(d0).tolist(),
        )
        return coords
    resid_centres = (fixed_epi * valid[None, :, :, None]).sum(2) / valid.sum(1)[
        None, :, None
    ].clamp_min(1)
    centroid = resid_centres.mean(1)
    auto_face = (
        int(para_local.numel()) == int(moving.shape[1])
    )  # whole-antibody mode (paratope = every movable atom): no CDR-window heuristic, the binding face comes from the pose itself
    para_pos = _interface_patch(moving, fixed) if auto_face else moving[:, para_local]
    para_centre = para_pos.mean(1)
    group_centre = moving.mean(1)
    face = para_centre - group_centre
    face = face / torch.linalg.vector_norm(face, dim=-1, keepdim=True).clamp_min(1e-6)
    reach = (para_pos * face[:, None]).sum(-1)  # [S, P] extent along the face direction
    n_tip = max(1, int(math.ceil(TIP_FRACTION * para_pos.shape[1])))
    tip_idx = torch.topk(reach, n_tip, dim=-1).indices
    tip = torch.gather(para_pos, 1, tip_idx[..., None].expand(-1, -1, 3)).mean(
        1
    )  # [S, 3] the paratope tip
    axes, note = approach_axes(fixed, fixed_chain, epi_local, valid, centroid)

    def score(x):
        return candidate_energy(x, fixed, rsum, epi_local, valid, para_local, k)

    best = moving.clone()
    cur_energy, cur_bad = score(moving)
    best_energy = torch.where(
        cur_bad, torch.full_like(cur_energy, float("inf")), cur_energy
    )
    tested = 0
    feasible = torch.zeros(samples, device=coords.device, dtype=torch.long)
    improving = torch.zeros(samples, device=coords.device, dtype=torch.long)
    accepted = torch.zeros(samples, device=coords.device, dtype=torch.long)
    for u in axes:
        align = _rotation_aligning(face, -u)
        aligned = torch.bmm(moving - tip[:, None], align.transpose(1, 2))
        for i in range(SPINS):
            spin = _rotation_about(u, 2.0 * math.pi * i / SPINS)
            rotated = torch.bmm(aligned, spin.transpose(1, 2))
            for radius in STANDOFFS:
                proposed = rotated + (centroid + radius * u)[:, None]
                energy, bad = score(proposed)
                tested += 1
                feasible += (~bad).long()
                improving += (energy < best_energy).long()
                take = (~bad) & (~entry_ok) & (energy < best_energy)
                best = torch.where(take[:, None, None], proposed, best)
                best_energy = torch.where(take, energy, best_energy)
                accepted += take.long()
    result = coords.clone()
    result[:, moving_ids] = best.to(coords.dtype)
    d1, _, _, _ = residue_distances(
        result[:, moving_ids].float(),
        result[:, fixed_ids].float(),
        epi_local,
        valid,
        para_local,
    )
    logger.info(
        "EPITOPE_GUIDANCE coarse axes=%s candidates=%d(per axis x spins x standoffs) feasible_per_sample=%s improving_per_sample=%s accepted=%s K=%d n=%d reached_before=%s reached_after=%s energy_before=%s energy_after=%s skipped_already_satisfied=%s",
        note,
        tested,
        feasible.tolist(),
        improving.tolist(),
        accepted.tolist(),
        k,
        epi_local.shape[0],
        reached_count(d0).tolist(),
        reached_count(d1).tolist(),
        cur_energy.tolist(),
        best_energy.tolist(),
        entry_ok.tolist(),
    )
    return result


# ---------------------------------------------------------------------------------------------------------------------------------
# Guidance on the denoiser's clean estimate x0 at intermediate steps.
# At an intermediate step the noisy state is a cloud of tens of Angstrom on which the request is trivially met, but the denoiser's prediction x0 is a
# compact structure on which the request is meaningful. guide_x0 moves the antibody group of x0 rigidly (refinement, then a coarse search for the samples that
# still miss the request, then a second refinement) and the sampler's Euler update uses the guided x0, so the trajectory is pulled towards a pose that
# meets the request while the network can still adapt the interface in the remaining steps. It is on by default and OPENDDE_RIGID_X0_START=off turns
# it off; the late pass on the sampled state (generator.py) is independent of it and can run after it.
# ---------------------------------------------------------------------------------------------------------------------------------
_X0_CALLS = {"n": 0}


def x0_schedule():
    """(start, every, last) or None. START defaults to 100, EVERY to 3 and LAST to 189 (the late pass owns the last steps);
    OPENDDE_RIGID_X0_START=off (or empty) turns the early pass off."""

    start = os.environ.get("OPENDDE_RIGID_X0_START", "100").strip()
    if start.lower() in ("", "off"):
        return None
    try:
        every = int(os.environ.get("OPENDDE_RIGID_X0_EVERY", "3"))
        last = int(os.environ.get("OPENDDE_RIGID_X0_LAST", "189"))
        start = int(start)
    except ValueError as exc:
        raise ValueError(
            "OPENDDE_RIGID_X0_START must be a step number or off, and OPENDDE_RIGID_X0_EVERY / OPENDDE_RIGID_X0_LAST step numbers"
        ) from exc
    if start < 0 or every < 1 or last < start:
        raise ValueError("Invalid OPENDDE_RIGID_X0_START / EVERY / LAST")
    return start, every, last


def x0_step_active(step_i):
    sch = x0_schedule()
    if sch is None:
        return False
    start, every, last = sch
    return start <= step_i <= last and (step_i - start) % every == 0


@torch.no_grad()
def _guide_x0_contact(x0, feats, step_i):
    """Rigidly move the antibody group of x0 so that the requested contacts are met: local refinement first (smallest move), then, for the samples that still
    violate a contact, the coarse pose search and a second refinement; the same order as guide_x0 uses for an epitope request. It uses the contact functions of
    the late schedule (rc.refine_rigid_contact, rc.search_rigid_contact), so the objective and the clash gate are the same."""
    orig_dtype = x0.dtype
    y = x0.float()
    if y.ndim != 3:
        raise ValueError("contact x0 guidance expects [sample, atom, xyz]")
    y = rc.refine_rigid_contact(y, feats, iterations=40)
    y = rc.search_rigid_contact(y, feats)
    y = rc.refine_rigid_contact(y, feats, iterations=40)
    _X0_CALLS["n"] += 1
    logger.info("RIGID_CONTACT x0 hook step=%d call=%d", step_i, _X0_CALLS["n"])
    return y.to(orig_dtype)


@torch.no_grad()
def guide_x0(x0, feats, step_i):
    """Rigidly move the antibody group of the denoiser's estimate x0 [.., S, N_atom, 3] so that the request is met. Returns x0 unchanged when the hook is not scheduled at this step."""

    mode = rigid_mode(feats)
    # A contact request gets the same early intervention on the clean estimate; samples that already satisfy it are left alone.
    if (
        mode == "on"
        and not active(feats)
        and x0_step_active(step_i)
        and feats.get("user_distance_restraint_index") is not None
        and feats["user_distance_restraint_index"].numel() > 0
    ):
        return _guide_x0_contact(x0, feats, step_i)
    if mode != "on" or not (active(feats) and x0_step_active(step_i)):
        return x0
    orig_dtype = x0.dtype
    y = x0.float()
    batch_shape = y.shape[:-3]
    if len(batch_shape) != 0:
        raise ValueError("guide_x0 expects coordinates without extra batch dimensions")
    # Smallest move first: a sample that misses the request by one residue is nudged, not re-docked. Only the samples that still miss it afterwards get the
    # coarse re-placement (search_epitope leaves every sample that already meets the request untouched).
    y = refine_epitope(y, feats, iterations=40)
    y = search_epitope(y, feats)
    y = refine_epitope(y, feats, iterations=40)
    _X0_CALLS["n"] += 1
    logger.info("EPITOPE_GUIDANCE x0 hook step=%d call=%d", step_i, _X0_CALLS["n"])
    return y.to(orig_dtype)


# ---------------------------------------------------------------------------------------------------------------------------------
# Accelerated core (rigid_core): the same clash kernels, acceptance rule and shortlist search as the contact mode; the epitope term is the
# fused group-minimum-distance kernel (group_contact). OPENDDE_RIGID_CORE=off runs the dense implementation above, =auto (default) runs the core
# when a CUDA device and Triton are available and the dense implementation otherwise, =on does the same but warns when the core cannot be used and
# falls back to the dense implementation on any exception, =check runs both on the same input, returns the dense result and logs
# "EPITOPE_GUIDANCE core parity".
# ---------------------------------------------------------------------------------------------------------------------------------
def _core_dispatch(kind, core_fn, dense_fn, coords, *args):
    import time

    core = rc.rigid_core_mode()  # not rigid_core.mode(): importing rigid_core needs Triton, which a CPU-only install does not have
    if core == "off":
        return dense_fn(coords, *args)
    reason = contact_core.supported(
        coords, rc
    )  # the same gate as the contact mode, TF32 included
    if reason is not None:
        contact_core.note_unsupported(core, kind, reason)
        return dense_fn(coords, *args)
    try:
        t0 = time.time()
        out = core_fn(coords, *args)
        torch.cuda.synchronize()
        t_fast = time.time() - t0
    except Exception as exc:  # noqa: BLE001  the dense implementation is always available
        logger.warning(
            "EPITOPE_GUIDANCE core %s failed (%r); dense fallback", kind, exc
        )
        return dense_fn(coords, *args)
    if core in ("auto", "on"):
        return out
    t0 = time.time()
    ref = dense_fn(coords, *args)
    torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs().flatten(1).amax(-1)
    moved_a = (out.float() - coords.float()).abs().flatten(1).amax(-1) > 0
    moved_b = (ref.float() - coords.float()).abs().flatten(1).amax(-1) > 0
    logger.info(
        "EPITOPE_GUIDANCE core parity %s max_abs_dX=%s moved_fast=%s moved_dense=%s t_fast=%.3fs t_dense=%.3fs",
        kind,
        diff.tolist(),
        moved_a.tolist(),
        moved_b.tolist(),
        t_fast,
        time.time() - t0,
    )
    return ref


@_traced
@torch.no_grad()
def refine_epitope(coords, feats, iterations=40):
    return _core_dispatch(
        "refine",
        _refine_epitope_core,
        _refine_epitope_dense,
        coords,
        feats,
        iterations,
    )


@_traced
@torch.no_grad()
def search_epitope(coords, feats):
    return _core_dispatch(
        "search", _search_epitope_core, _search_epitope_dense, coords, feats
    )


@torch.no_grad()
def _refine_epitope_core(coords, feats, iterations=40):
    from opendde.tfg import rigid_core

    if coords.ndim != 3:
        raise ValueError("epitope guidance requires [sample, atom, xyz].")
    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float()
    ra, rb = _radii_split(coords, feats, moving_ids, fixed_ids)
    term = rigid_core.GroupMinTerm(fixed, epi_local, para_local, k)
    new, accepted, info = rigid_core.refine(
        moving, ra, fixed, rb, term, iterations=iterations, gate=True
    )
    result = coords.clone()
    result[:, moving_ids] = new.to(coords.dtype)
    first_d = info["aux_first"]
    if info["early_out"]:
        logger.info(
            "EPITOPE_GUIDANCE refine K=%d n=%d reached_before=%s reached_after=%s status=all_satisfied_early_out accepted=%s",
            k,
            epi_local.shape[0],
            reached_count(first_d).tolist(),
            reached_count(first_d).tolist(),
            [0] * coords.shape[0],
        )
        return result
    # the status is read off the coordinates that are returned, not the internal state
    final_d, _, _ = term.distances(result[:, moving_ids].float())
    entry_ok = reached_count(first_d) >= k
    final_ok = reached_count(final_d) >= k
    logger.info(
        "EPITOPE_GUIDANCE refine K=%d n=%d reached_before=%s reached_after=%s energy %s -> %s status=%s accepted=%s",
        k,
        epi_local.shape[0],
        reached_count(first_d).tolist(),
        reached_count(final_d).tolist(),
        info["first_energy"].tolist(),
        info["final_energy"].tolist(),
        _status(entry_ok, final_ok, accepted > 0),
        accepted.tolist(),
    )
    return result


@torch.no_grad()
def _search_epitope_core(coords, feats):
    """search_epitope on the accelerated core: the same axes / spins / standoffs / tip anchor and the same selection rule as the dense loop;
    the shortlist (clash kernel + fused group term with provable error windows) is re-evaluated exactly with `place(p)`, the dense loop's
    own construction of candidate p."""
    from opendde.tfg import rigid_core

    _guard(coords)
    fixed_ids, moving_ids, epi_local, valid, para_local, k = groups(coords, feats)
    fixed = coords[:, fixed_ids].float()
    moving = coords[:, moving_ids].float()
    samples = coords.shape[0]
    ra, rb = _radii_split(coords, feats, moving_ids, fixed_ids)
    asym = feats["asym_id"][feats["atom_to_token_idx"]].to(coords.device)
    fixed_chain = asym[fixed_ids]
    term = rigid_core.GroupMinTerm(fixed, epi_local, para_local, k)
    d0, _, _ = term.distances(moving)
    entry_ok = reached_count(d0) >= k
    if bool(entry_ok.all()):
        logger.info(
            "EPITOPE_GUIDANCE coarse skipped: every sample already meets the request K=%d n=%d reached=%s",
            k,
            epi_local.shape[0],
            reached_count(d0).tolist(),
        )
        return coords
    s_, n_, m_ = samples, epi_local.shape[0], epi_local.shape[1]
    fixed_epi = fixed[:, epi_local.clamp_min(0).reshape(-1)].reshape(s_, n_, m_, 3)
    resid_centres = (fixed_epi * valid[None, :, :, None]).sum(2) / valid.sum(1)[
        None, :, None
    ].clamp_min(1)
    centroid = resid_centres.mean(1)
    auto_face = int(para_local.numel()) == int(moving.shape[1])
    para_pos = (
        _interface_patch_fast(moving, fixed) if auto_face else moving[:, para_local]
    )
    para_centre = para_pos.mean(1)
    group_centre = moving.mean(1)
    face = para_centre - group_centre
    face = face / torch.linalg.vector_norm(face, dim=-1, keepdim=True).clamp_min(1e-6)
    reach = (para_pos * face[:, None]).sum(-1)
    n_tip = max(1, int(math.ceil(TIP_FRACTION * para_pos.shape[1])))
    tip_idx = torch.topk(reach, n_tip, dim=-1).indices
    tip = torch.gather(para_pos, 1, tip_idx[..., None].expand(-1, -1, 3)).mean(1)
    axes, note = approach_axes(fixed, fixed_chain, epi_local, valid, centroid)
    n_spin, n_stand = SPINS, len(STANDOFFS)
    # candidate transforms for the shortlist kernel (bounds only; the exact placement below uses the dense loop's per-candidate ops):
    # all spins of an axis at once.  p = (axis * SPINS + spin) * len(STANDOFFS) + standoff, the dense loop order.
    angles = torch.tensor(
        [2.0 * math.pi * i / n_spin for i in range(n_spin)],
        device=coords.device,
        dtype=torch.float64,
    )
    stand = torch.tensor(STANDOFFS, device=coords.device, dtype=torch.float32)
    Rs, ts, aligns = [], [], []
    for u in axes:
        align = _rotation_aligning(face, -u)
        aligns.append(align)
        spins = _rotation_about(
            u[:, None].expand(-1, n_spin, -1).reshape(-1, 3), angles.repeat(samples)
        ).view(samples, n_spin, 3, 3)
        Rm = torch.matmul(spins, align[:, None])  # [S,spin,3,3]
        shift = torch.matmul(tip[:, None, None, :], Rm.transpose(-1, -2))[
            :, :, 0
        ]  # [S,spin,3]
        tt = (centroid[:, None, :] + stand[None, :, None] * u[:, None, :])[
            :, None, :, :
        ] - shift[:, :, None, :]  # [S,spin,stand,3]
        Rs.append(Rm[:, :, None].expand(-1, -1, n_stand, -1, -1))
        ts.append(tt)
    R = torch.cat(Rs, 1).reshape(samples, -1, 3, 3).contiguous()
    t = torch.cat(ts, 1).reshape(samples, -1, 3).contiguous()
    P = R.shape[1]
    cache = {}

    def place(p):
        a, rem = divmod(p, n_spin * n_stand)
        i, r = divmod(rem, n_stand)
        u = axes[a]
        if a not in cache:
            cache.clear()
            cache[a] = torch.bmm(moving - tip[:, None], aligns[a].transpose(1, 2))
        spin = _rotation_about(u, 2.0 * math.pi * i / n_spin)
        rotated = torch.bmm(cache[a], spin.transpose(1, 2))
        return rotated + (centroid + STANDOFFS[r] * u)[:, None]

    out = rigid_core.search(moving, ra, fixed, rb, term, R, t, place, entry_ok)
    best_idx = out["best"]
    best = moving.clone()
    for p in sorted(set(best_idx[best_idx >= 0].tolist())):
        X = place(p)
        best = torch.where((best_idx == p)[:, None, None], X, best)
    best_energy = out["best_energy"]
    result = coords.clone()
    result[:, moving_ids] = best.to(coords.dtype)
    d1, _, _ = term.distances(result[:, moving_ids].float())
    logger.info(
        "EPITOPE_GUIDANCE coarse axes=%s candidates=%d(per axis x spins x standoffs) feasible_per_sample=%s improving_per_sample=%s accepted=%s K=%d n=%d reached_before=%s reached_after=%s energy_before=%s energy_after=%s skipped_already_satisfied=%s shortlist=%s reeval=%d",
        note,
        P,
        out["n_feasible"].tolist(),
        out["n_improving"].tolist(),
        (best_idx >= 0).long().tolist(),
        k,
        epi_local.shape[0],
        reached_count(d0).tolist(),
        reached_count(d1).tolist(),
        out["entry_energy"].tolist(),
        best_energy.tolist(),
        entry_ok.tolist(),
        out["stats"]["shortlist"],
        out["stats"]["reeval"],
    )
    return result
