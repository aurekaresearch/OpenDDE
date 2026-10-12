# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""The contact term of the rigid-body solver with a penalty other than the squared violation (see `robust`).

This module imports `rigid_core` and with it Triton: import it on the accelerated core only (`robust.make_pair_term` does).
"""

import torch

from opendde.tfg.rigid_core import PairTerm


class RobustPairTerm(PairTerm):
    """PairTerm whose cost per pair is `penalty.rho(violation)` instead of `violation^2`.

    With the quadratic `robust.Penalty` it equals the stock term; satisfaction (every pair inside its window), the clash handling and the
    pose search are the stock ones."""

    def __init__(self, fixed, ma, fb, lower, upper, penalty, tol=1e-6):
        super().__init__(fixed, ma, fb, lower, upper, tol)
        self.pen = penalty

    def energy(self, X):
        _, pair_d, v = self._pair(X)
        return self.pen.rho(v.abs()).sum(-1), pair_d

    def energy_grad(self, X):
        pair_vec, pair_d, v = self._pair(X)
        a = v.abs()
        force = torch.sign(v) * self.pen.force(a)
        g = force[..., None] * pair_vec / pair_d[..., None]
        g = torch.where((pair_d > 1e-6)[..., None], g, torch.zeros_like(g))
        grad = torch.zeros_like(X)
        grad.index_add_(1, self.ma, g)
        return self.pen.rho(a).sum(-1), grad, pair_d

    def search_exact(self, X):
        d = torch.linalg.vector_norm(X[:, self.ma] - self.fixed[:, self.fb], dim=-1)
        err = torch.relu(d - self.upper) + torch.relu(self.lower - d)
        return 0.5 * self.pen.rho(err).sum(-1)

    def fast_candidates(self, A, R_all, t_all, model):
        Am = A[:, self.ma].double()
        xm = (
            torch.einsum("scij,spj->scpi", R_all.double(), Am)
            + t_all.double()[:, :, None]
        )
        dpair = torch.linalg.vector_norm(
            xm - self.fixed[:, self.fb].double()[:, None], dim=-1
        )
        lo64, up64 = self.lower.double(), self.upper.double()
        err = torch.relu(dpair - up64) + torch.relu(lo64 - dpair)
        E = 0.5 * self.pen.rho(err).sum(-1)
        dl = model.delta_pair(dpair, lo64, up64)
        W = self.pen.bound(err, dl).sum(-1)
        zero = ((dpair >= lo64 + dl) & (dpair <= up64 - dl)).all(-1)
        return E, W, zero
