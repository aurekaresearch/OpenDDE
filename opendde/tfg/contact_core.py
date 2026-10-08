"""Contact-mode adapter of the accelerated rigid-body core.

Maps the contact tensors onto rigid_core (PairTerm) and back: the restraint tables, the 1,500 candidate poses of the
coarse search (25 approach axes x 12 rotations x 5 stand-offs around a Kabsch fit of the constrained atoms) and the
placement of the winning candidates with the dense implementation's own operations.

    OPENDDE_RIGID_CORE=auto   rigid_core when the setting is supported, otherwise the dense implementation (default)
    OPENDDE_RIGID_CORE=off    dense implementation only
    OPENDDE_RIGID_CORE=on     as auto, but a warning names the reason when the core cannot be used; the dense implementation on any exception
    OPENDDE_RIGID_CORE=check  rigid_core AND the dense implementation on the same input; the dense result is returned and
                              the differences are logged ("RIGID_CONTACT core parity")

Supported: CUDA coordinates in float32, float16 or bfloat16 with TF32 matmul disabled. Anything else uses the dense
implementation, so a result never silently changes its objective.
"""

import math
import time

import torch

from opendde.data.constants import rdkit_vdws
from opendde.utils.logger import get_logger

logger = get_logger(__name__)

_RADII = (5.5, 7.5, 10.0, 15.0, 20.0)
_TURNS = tuple(i * math.pi / 6 for i in range(12))
_NORMALS = 25


def supported(coords, rc):
    """None when the accelerated core may be used, otherwise the reason it may not."""
    if not coords.is_cuda:
        return "coordinates are not on a CUDA device"
    if not rc.triton_available():
        return "Triton is not installed"
    if coords.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return "unsupported dtype %s" % coords.dtype
    if torch.backends.cuda.matmul.allow_tf32:
        return "TF32 matmul is enabled"
    return None


_REASONS_SEEN = set()


def note_unsupported(core, kind, reason):
    """Say once per reason why the accelerated core was skipped: at INFO under ``auto``, as a warning under ``on`` and ``check``."""
    if reason in _REASONS_SEEN:
        return
    _REASONS_SEEN.add(reason)
    log = logger.info if core == "auto" else logger.warning
    log("RIGID_CONTACT core not used (%s); dense %s", reason, kind)


def _tables(coords, feats, rc):
    fixed_ids, moving_ids, fix_atoms, mov_atoms = rc.contact_groups(coords, feats)
    fixed_lookup = torch.full(
        (coords.shape[-2],), -1, device=coords.device, dtype=torch.long
    )
    fixed_lookup[fixed_ids] = torch.arange(len(fixed_ids), device=coords.device)
    moving_lookup = torch.full(
        (coords.shape[-2],), -1, device=coords.device, dtype=torch.long
    )
    moving_lookup[moving_ids] = torch.arange(len(moving_ids), device=coords.device)
    radii = torch.as_tensor(rdkit_vdws, device=coords.device, dtype=torch.float32)
    element = feats["ref_element"].argmax(-1)
    if (element == 0).any():
        raise ValueError("Pilot expects heavy-atom protein input; hydrogen found.")
    return dict(
        fixed_ids=fixed_ids,
        moving_ids=moving_ids,
        fix_atoms=fix_atoms,
        mov_atoms=mov_atoms,
        ma=moving_lookup[mov_atoms],
        fb=fixed_lookup[fix_atoms],
        ra=radii[element[moving_ids]],
        rb=radii[element[fixed_ids]],
        lower=feats["user_distance_restraint_lower_bound"].float(),
        upper=feats["user_distance_restraint_upper_bound"].float(),
    )


def _skew(w):
    skew = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    skew[..., 0, 1], skew[..., 0, 2] = -w[..., 2], w[..., 1]
    skew[..., 1, 0], skew[..., 1, 2] = w[..., 2], -w[..., 0]
    skew[..., 2, 0], skew[..., 2, 1] = -w[..., 1], w[..., 0]
    return skew


def _candidate_frames(coords, tab, rc):
    """Kabsch fit, normals, the 1500 (R, t) transforms (candidate p = (normal * 12 + turn) * 5 + radius, the dense loop order) and
    the exact placement function place(p) built with the dense path's own operations."""
    dev = coords.device
    S = coords.shape[0]
    fixed = coords[:, tab["fixed_ids"]].float()
    moving = coords[:, tab["moving_ids"]].float()
    source = coords[:, tab["mov_atoms"]].float()
    target = coords[:, tab["fix_atoms"]].float()
    sc = source.mean(1, keepdim=True)
    tc = target.mean(1, keepdim=True)
    u, _, vh = torch.linalg.svd(torch.bmm((source - sc).transpose(1, 2), target - tc))
    diag = torch.eye(3, device=dev).expand(S, 3, 3).clone()
    diag[:, 2, 2] = torch.det(torch.bmm(u, vh))
    fit = torch.bmm(torch.bmm(u, diag), vh)
    fitted = torch.bmm(moving - sc, fit)
    outward = (tc - fixed.mean(1, keepdim=True)).squeeze(1)
    outward /= torch.linalg.vector_norm(outward, dim=-1, keepdim=True).clamp_min(1e-6)
    normals = [outward]
    for i in range(24):
        z = 1 - 2 * (i + 0.5) / 24
        phi = i * math.pi * (3 - math.sqrt(5))
        rad = math.sqrt(1 - z * z)
        normals.append(
            torch.tensor(
                [rad * math.cos(phi), rad * math.sin(phi), z], device=dev
            ).expand(S, 3)
        )
    normals = torch.stack(normals, dim=1)  # [S, 25, 3]
    rots = torch.stack(
        [torch.linalg.matrix_exp(_skew(normals * turn)) for turn in _TURNS], dim=2
    )  # [S,25,12,3,3]
    total = torch.matmul(fit[:, None, None], rots.transpose(-1, -2))
    R = (
        total.transpose(-1, -2)[:, :, :, None]
        .expand(S, _NORMALS, 12, 5, 3, 3)
        .reshape(S, 1500, 3, 3)
        .contiguous()
    )
    offset = torch.matmul(sc[:, None, None], total).squeeze(-2)
    radius = torch.tensor(_RADII, device=dev, dtype=torch.float32)
    shift = tc[:, None] + radius[None, None, :, None] * normals[:, :, None, :]
    t = (shift[:, :, None] - offset[:, :, :, None]).reshape(S, 1500, 3).contiguous()
    lower, upper = tab["lower"], tab["upper"]
    entry_distance = torch.linalg.vector_norm(source - target, dim=-1)
    satisfied = (
        (entry_distance >= lower - 1e-6) & (entry_distance <= upper + 1e-6)
    ).all(-1)
    turns = torch.tensor(_TURNS, device=dev, dtype=torch.float32)

    def place(p):
        """Candidate p for every sample, with the dense loop's operations (rotation = matrix_exp(skew(normal * turn)), bmm, add)."""
        n_idx, k_idx, r_idx = p // 60, (p // 5) % 12, p % 5
        normal = normals[:, n_idx]
        rotation = torch.linalg.matrix_exp(_skew(normal * turns[k_idx]))
        return torch.bmm(fitted, rotation.transpose(1, 2)) + (
            tc + radius[r_idx] * normal[:, None]
        )

    return dict(
        fixed=fixed,
        moving=moving,
        fitted=fitted,
        tc=tc,
        normals=normals,
        radius=radius,
        turns=turns,
        R=R,
        t=t,
        satisfied=satisfied,
        place=place,
    )


def _place_winners(fr_, best):
    """The winning candidate of every sample (per-sample index), dense-path operations; samples with best < 0 keep their pose."""
    S = best.shape[0]
    dev = best.device
    won = best >= 0
    p = best.clamp_min(0)
    n_idx, k_idx, r_idx = p // 60, (p // 5) % 12, p % 5
    ar = torch.arange(S, device=dev)
    normal = fr_["normals"][ar, n_idx]
    turn = fr_["turns"][k_idx]
    rotation = torch.linalg.matrix_exp(_skew(normal * turn[:, None]))
    proposed = torch.bmm(fr_["fitted"], rotation.transpose(1, 2)) + (
        fr_["tc"] + fr_["radius"][r_idx][:, None, None] * normal[:, None]
    )
    return torch.where(won[:, None, None], proposed, fr_["moving"])


@torch.no_grad()
def refine_core(coords, feats, iterations, rc):
    """rigid_core.refine with the contact term (same contract as rigid_contact.refine_rigid_contact)."""
    from opendde.tfg import rigid_core

    tab = _tables(coords, feats, rc)
    fixed = coords[:, tab["fixed_ids"]].float()
    moving = coords[:, tab["moving_ids"]].float()
    term = rigid_core.PairTerm(fixed, tab["ma"], tab["fb"], tab["lower"], tab["upper"])
    new, accepted, info = rigid_core.refine(
        moving,
        tab["ra"],
        fixed,
        tab["rb"],
        term,
        iterations=iterations,
        gate=True,
    )
    result = coords.clone()
    result[:, tab["moving_ids"]] = new.to(coords.dtype)
    logger.info(
        "RIGID_CONTACT core refine energy %s -> %s; mean_distance %s -> %s; iterations=%d accepted=%s early_out=%s",
        info["first_energy"].tolist(),
        info["final_energy"].tolist(),
        info["aux_first"].mean(-1).tolist(),
        info["aux_final"].mean(-1).tolist(),
        iterations,
        accepted.tolist(),
        info["early_out"],
    )
    return result


@torch.no_grad()
def search_core(coords, feats, rc):
    """rigid_core.search with the contact term: same candidates, order and selection rule as the dense search."""
    from opendde.tfg import rigid_core

    tab = _tables(coords, feats, rc)
    fr_ = _candidate_frames(coords, tab, rc)
    term = rigid_core.PairTerm(
        fr_["fixed"], tab["ma"], tab["fb"], tab["lower"], tab["upper"]
    )
    out = rigid_core.search(
        fr_["moving"],
        tab["ra"],
        fr_["fixed"],
        tab["rb"],
        term,
        fr_["R"],
        fr_["t"],
        fr_["place"],
        fr_["satisfied"],
    )
    chosen = _place_winners(fr_, out["best"])
    result = coords.clone()
    result[:, tab["moving_ids"]] = chosen.to(coords.dtype)
    logger.info(
        "RIGID_CONTACT core coarse search satisfied=%s candidates=1500 feasible=%s improving=%s winner=%s final_energy=%s shortlist=%s reeval=%d",
        fr_["satisfied"].tolist(),
        out["n_feasible"].tolist(),
        out["n_improving"].tolist(),
        out["best"].tolist(),
        out["best_energy"].tolist(),
        out["stats"]["shortlist"],
        out["stats"]["reeval"],
    )
    return result


def _parity(tag, kind, out, ref, coords, t_a, t_b):
    diff = (out.float() - ref.float()).abs().flatten(1).amax(-1)
    changed_a = (out.float() - coords.float()).abs().flatten(1).amax(-1) > 0
    changed_b = (ref.float() - coords.float()).abs().flatten(1).amax(-1) > 0
    logger.info(
        "RIGID_CONTACT %s %s max_abs_dX=%s moved_fast=%s moved_dense=%s t_fast=%.3fs t_dense=%.3fs",
        tag,
        kind,
        diff.tolist(),
        changed_a.tolist(),
        changed_b.tolist(),
        t_a,
        t_b,
    )


def run(kind, coords, feats, rc, dense, **kw):
    """Dispatch one call. ``dense`` is the dense implementation, called as dense(coords, feats, **kw).
    OPENDDE_RIGID_CORE (auto/on/check) selects the accelerated core; the dense implementation is its fallback and its comparison."""
    core = rc.rigid_core_mode()  # not rigid_core.mode(): importing rigid_core needs Triton, which a CPU-only install does not have
    if core == "off":
        return dense(coords, feats, **kw)
    reason = supported(coords, rc)
    if reason is not None:
        note_unsupported(core, kind, reason)
        return dense(coords, feats, **kw)
    fast = search_core if kind == "search" else refine_core
    args = (
        (coords, feats, kw["iterations"], rc)
        if kind == "refine"
        else (coords, feats, rc)
    )
    try:
        t0 = time.time()
        out = fast(*args)
        torch.cuda.synchronize()
        t_fast = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        logger.warning("RIGID_CONTACT core %s failed (%r); dense fallback", kind, exc)
        return dense(coords, feats, **kw)
    if core in ("auto", "on"):
        return out
    t0 = time.time()
    ref = dense(coords, feats, **kw)
    torch.cuda.synchronize()
    _parity("core parity", kind, out, ref, coords, t_fast, time.time() - t0)
    return ref
