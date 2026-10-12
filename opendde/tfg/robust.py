# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Opt-in robust objective for contact constraints. With no switch set the stock objective runs unchanged.

The stock rigid-body solver sums the squared violation of every contact pair and starts its coarse search from a Kabsch fit to all
pairs. The force of a violated pair grows with its distance and one wrong pair spoils every starting pose, so a single wrong pair
can pull the antibody away from the interface that the other pairs agree on. Two switches change the objective and the search
start, and nothing else (not the clash field, the pose-search kernels or the acceptance rule):

    OPENDDE_RIGID_PENALTY   quad (stock) | welsch:<L>
                            Penalty on a violation e (in A). `welsch:<L>` is quadratic near zero and saturates at L^2; its force
                            e exp(-e^2 / L^2) returns to zero for large violations, so a pair that cannot be met stops pulling
                            (Rosetta TOPOUT, a Welsch loss). L defaults to 4.
    OPENDDE_RIGID_INIT      all (stock) | subsets
                            Coarse search: also start from the Kabsch fit of every leave-one-out subset of the pairs (three pairs
                            or more), each time from the best pose so far, so a pose is replaced only by one of lower energy.

Both apply to `constraint.contact` and run on the accelerated core only. A requested switch never falls back to the dense solver
(that would silently run the stock objective): it raises instead.

This module needs only torch, so it can be imported on installs without Triton. The solver term that uses the penalty is in
`robust_term`, which imports `rigid_core` and with it Triton.
"""

import math
import os

import torch

from opendde.utils.logger import get_logger

logger = get_logger(__name__)


class Penalty:
    """The stock penalty: the squared violation."""

    name = "quad"

    def rho(self, e):
        return e * e

    def force(self, e):
        """rho'(e) / 2 for e >= 0."""
        return e

    def bound(self, e, dl):
        """Upper bound on |0.5 rho(e') - 0.5 rho(e)| for |e' - e| <= dl (e >= 0)."""
        return e * dl + 0.5 * dl * dl


class Welsch(Penalty):
    """rho(e) = L^2 (1 - exp(-e^2 / L^2)): about e^2 near zero, saturating at L^2, force e exp(-e^2 / L^2) -> 0 (Rosetta TOPOUT)."""

    def __init__(self, scale):
        self.scale = float(scale)
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError(
                f"OPENDDE_RIGID_PENALTY welsch:<L> needs a positive scale, got {scale!r}"
            )
        self.name = f"welsch:{scale}"

    def rho(self, e):
        l2 = self.scale * self.scale
        # l2 (1 - exp(-e^2 / l2)), without the float32 cancellation near zero
        return -l2 * torch.expm1(-(e * e) / l2)

    def force(self, e):
        return e * torch.exp(-(e * e) / (self.scale * self.scale))

    def bound(self, e, dl):
        return (e + dl) * dl  # force(e) <= e: loose but valid


def penalty_from_env():
    """None for the stock quadratic penalty."""
    spec = os.environ.get("OPENDDE_RIGID_PENALTY", "quad").strip().lower()
    if spec in ("", "quad"):
        return None
    kind, _, arg = spec.partition(":")
    if kind == "welsch":
        try:
            return Welsch(arg or 4.0)
        except ValueError as exc:
            raise ValueError(
                f"OPENDDE_RIGID_PENALTY={spec!r} is not valid: {exc}"
            ) from exc
    raise ValueError("OPENDDE_RIGID_PENALTY must be quad or welsch:<L>")


def init_subsets_enabled():
    spec = os.environ.get("OPENDDE_RIGID_INIT", "all").strip().lower()
    if spec not in ("", "all", "subsets"):
        raise ValueError("OPENDDE_RIGID_INIT must be all or subsets")
    return spec == "subsets"


def variant_requested():
    """True when a switch is set: such a request must not run on the dense solver."""
    return penalty_from_env() is not None or init_subsets_enabled()


def refuse_dense(kind):
    raise RuntimeError(
        f"a robust constraint switch is requested for {kind} but the accelerated core is not usable here "
        "(needs CUDA, Triton and TF32 off); the dense fallback would silently run the stock objective"
    )


def make_pair_term(fixed, ma, fb, lower, upper):
    """The contact term of the rigid solver: the stock PairTerm unless a penalty is requested."""
    # rigid_core needs Triton: this is reached on the accelerated core only
    from opendde.tfg import rigid_core

    penalty = penalty_from_env()
    if penalty is None:
        return rigid_core.PairTerm(fixed, ma, fb, lower, upper)
    from opendde.tfg.robust_term import RobustPairTerm

    return RobustPairTerm(fixed, ma, fb, lower, upper, penalty)


_noted = set()


def note_contact_only(kind):
    """The switches change the contact objective only. Say so once for another kind of request instead of silently running stock."""
    if kind not in _noted and variant_requested():
        _noted.add(kind)
        logger.warning(
            "OPENDDE_RIGID_PENALTY and OPENDDE_RIGID_INIT apply to constraint.contact only; this %s request uses the stock objective",
            kind,
        )
