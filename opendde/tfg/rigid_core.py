"""rigid_core: ONE rigid-body refinement and ONE provable-shortlist search for both guidance modes.

The two modes (contact restraints, epitope "pocket") differ only in their contact term and in how candidate poses are
generated; everything else - clash / severe-overlap evaluation (clash_field.ClashField), the backtracking acceptance rule,
the search's error windows, shortlist and exact re-evaluation - is this file.

    refine(X0, ra, B, rb, term, iterations, gate=True) -> (X, accepted, info)
    search(A, ra, B, rb, term, R, t, place, satisfied)  -> dict(best, best_energy, n_feasible, n_improving, entry_energy, entry_severe, stats)

A `term` provides (X is [S,M,3] or [S,K,M,3], K proposals per sample):
    D                    normaliser of the rigid step (P restraint pairs / K epitope groups)
    energy(X)            -> (sum of squared violations [S,K], aux)
    energy_grad(X)       -> (sum of squared violations [S], grad [S,M,3] (zeros + index_add of the pair forces), aux)
    satisfied(aux)       -> bool [S]   (gate: a satisfied sample is never moved)
    search_exact(X)      -> 0.5 * sum of squared violations [S], the dense implementation's own expression (exact re-evaluation)
    fast_candidates(A, R_all, t_all, model) -> (E_term [S,NC] float64, W_term [S,NC] float64, zero [S,NC] bool)
                         bound of the term for every candidate pose (NC = P + 1, the last one is the entry pose, R = I, t = 0)
Two implementations: PairTerm (contact mode) and GroupMinTerm (pocket mode; fused kernel group_contact).

Semantics (both modes, identical to the dense implementations in rigid_contact and epitope_guidance):
    overlap = relu(0.85 (ra+rb) - d), severe: d < 0.75 (ra+rb);   refine energy 0.5 * (sum viol^2 + 20 sum overlap^2),
    search energy 0.5 * sum viol^2 + 10 * sum overlap^2;  rigid step: translation -0.2 sum(grad)/D clipped to 0.5 A,
    rotation -0.15 M/D solve(inertia + 1e-3 I, torque) clipped to 0.03 rad, 10 halvings 0.5^b evaluated in one launch, the first
    accepted wins; accept iff not satisfied, no new severe pair, max depth <= depth + 1e-6, energy < energy - 1e-7; stop when no
    sample moved; the final pose must not have a severe pair that the entry pose did not have (RuntimeError otherwise).
    Search: a candidate with a severe pair is dropped; the running best starts at the exact entry energy (inf if the entry pose is
    severe); a candidate wins iff energy < running best (strict; equal energies keep the first index); satisfied samples are skipped.

Numerical tier: the clash terms come from the cell-list kernel (float64 accumulation, per-pair arithmetic bit-matched to
torch.cdist), the contact terms are bit-identical to the dense implementation's (fused group kernel / plain torch for pairs);
decisions equal the dense implementation's unless its own fp32 rounding is within ~1e-7 relative of a decision margin.
The search shortlist is provable for any fp32 program within the error model (_Model); the exact re-evaluation of the shortlist
uses the mode's own candidate construction (`place(p)`) and these kernels.
"""

import math
import torch
import triton

from opendde.tfg import clash_field as fr
from opendde.tfg import pose_search_kernel as psk
from opendde.tfg import group_contact as fg

SOFT = 0.85
HARD = 0.75
W_REFINE = 20.0
W_SEARCH = 10.0
TARGET = 4.5
GATE = 5.0
_U = 2.0**-24
_STATS = {}
_CDIST_ORDER = {"checked": False, "matches": None}


def cdist_order_ok():
    """True iff this torch build's cdist(donot_use_mm) reduces squares in the order the kernels reproduce (checked once)."""
    if not _CDIST_ORDER["checked"]:
        _CDIST_ORDER["matches"] = bool(fr.cdist_order_matches())
        _CDIST_ORDER["checked"] = True
    return _CDIST_ORDER["matches"]


# ======================================================================================================================
# terms
# ======================================================================================================================
class PairTerm:
    """Contact mode: P atom pairs (moving atom ma[i], fixed atom fb[i]) with distance windows [lower_i, upper_i]."""

    def __init__(self, fixed, ma, fb, lower, upper, tol=1e-6):
        self.fixed = fixed.float()
        self.ma = ma.long()
        self.fb = fb.long()
        self.lower = lower.float()
        self.upper = upper.float()
        self.tol = tol
        self.D = int(self.ma.numel())

    def _pair(self, X):
        pair_vec = X[..., self.ma, :] - (
            self.fixed[:, self.fb] if X.dim() == 3 else self.fixed[:, None, self.fb]
        )
        pair_d = torch.linalg.vector_norm(pair_vec, dim=-1).clamp_min(1e-6)
        violation = torch.relu(pair_d - self.upper) - torch.relu(self.lower - pair_d)
        return pair_vec, pair_d, violation

    def energy(self, X):
        _, pair_d, violation = self._pair(X)
        return violation.square().sum(-1), pair_d

    def energy_grad(self, X):
        pair_vec, pair_d, violation = self._pair(X)
        contact_grad = violation[..., None] * pair_vec / pair_d[..., None]
        contact_grad = torch.where(
            (pair_d > 1e-6)[..., None], contact_grad, torch.zeros_like(contact_grad)
        )
        grad = torch.zeros_like(X)
        grad.index_add_(1, self.ma, contact_grad)
        return violation.square().sum(-1), grad, pair_d

    def satisfied(self, pair_d):
        return (
            (pair_d >= self.lower - self.tol) & (pair_d <= self.upper + self.tol)
        ).all(-1)

    def search_exact(self, X):
        d = torch.linalg.vector_norm(X[:, self.ma] - self.fixed[:, self.fb], dim=-1)
        err = torch.relu(d - self.upper) + torch.relu(self.lower - d)
        return 0.5 * err.square().sum(-1)

    def fast_candidates(self, A, R_all, t_all, model):
        Am = A[:, self.ma].double()  # our side in float64: error ~0
        xm = (
            torch.einsum("scij,spj->scpi", R_all.double(), Am)
            + t_all.double()[:, :, None]
        )  # [S,NC,P,3]
        dpair = torch.linalg.vector_norm(
            xm - self.fixed[:, self.fb].double()[:, None], dim=-1
        )  # [S,NC,P]
        lo64, up64 = self.lower.double(), self.upper.double()
        err = torch.relu(dpair - up64) + torch.relu(lo64 - dpair)
        E = 0.5 * err.square().sum(-1)
        dl = model.delta_pair(dpair, lo64, up64)
        W = (err * dl + 0.5 * dl * dl).sum(-1)
        zero = ((dpair >= lo64 + dl) & (dpair <= up64 - dl)).all(-1)
        return E, W, zero

    def describe(self, pair_d):
        return pair_d.mean(-1)


class GroupMinTerm:
    """Pocket mode: n groups of fixed atoms (grp [n,m] long, -1 padded); d_g = min distance from any atom of group g to any probe
    atom A[para]; energy over the k closest groups: 0.5 * sum relu(d_g - TARGET)^2; satisfied iff #(d_g <= GATE) >= k."""

    def __init__(self, fixed, grp, para, k, dense=False):
        self.fixed = fixed.float().contiguous()
        self.grp = grp.long()
        self.para = para.long()
        self.k = int(k)
        self.valid = self.grp >= 0
        self.gflat = self.grp[self.valid]
        self.gslot = torch.nonzero(self.valid)
        self.D = self.k
        self.dense = dense
        self._inst = {}

    def _inst_of_pose(self, S, K):
        key = (S, K)
        if key not in self._inst:
            self._inst[key] = torch.arange(
                S, device=self.fixed.device
            ).repeat_interleave(K)
        return self._inst[key]

    def distances(self, X):
        """X [S,M,3] or [S,K,M,3] -> d [S(,K),n], a_slot, p_slot (bit-identical to the dense implementation)."""
        K = 1 if X.dim() == 3 else X.shape[1]
        S = X.shape[0]
        M = X.shape[-2]
        fn = fg.group_min_distance_dense if self.dense else fg.group_min_distance_fast
        d, a, p = fn(
            X.reshape(S * K, M, 3),
            self.fixed,
            self._inst_of_pose(S, K),
            self.grp,
            self.para,
        )
        if X.dim() == 3:
            return d, a, p
        return d.view(S, K, -1), a.view(S, K, -1), p.view(S, K, -1)

    def energy(self, X):
        d, _, _ = self.distances(X)
        ordered, _ = torch.sort(d, dim=-1, stable=True)
        violation = torch.relu(ordered[..., : self.k] - TARGET)
        return violation.square().sum(-1), d

    def energy_grad(self, X):
        S, M, _ = X.shape
        d, a_slot, p_slot = self.distances(X)
        _, violation, top = fg.contact_terms(d, self.k)
        rows = torch.arange(S, device=X.device)[:, None].expand(-1, self.k)
        fa = self.fixed[rows, self.grp.clamp_min(0)[top, a_slot.gather(1, top)]]
        mp = self.para[p_slot.gather(1, top)]
        vec = X[rows, mp] - fa
        dist = torch.linalg.vector_norm(vec, dim=-1).clamp_min(1e-6)
        contrib = violation[..., None] * vec / dist[..., None]
        grad = torch.zeros_like(X)
        grad.view(-1, 3).index_add_(
            0, (rows * M + mp).reshape(-1), contrib.reshape(-1, 3)
        )
        return violation.square().sum(-1), grad, d

    def satisfied(self, d):
        return (d <= GATE).sum(-1) >= self.k

    def reached(self, d):
        return (d <= GATE).sum(-1)

    def search_exact(self, X):
        d, _, _ = self.distances(X)
        return fg.contact_terms(d, self.k)[0]

    def fast_candidates(self, A, R_all, t_all, model):
        S, NC = R_all.shape[0], R_all.shape[1]
        n, m = self.grp.shape
        G = int(self.gflat.numel())
        dev = A.device
        inf = float("inf")
        A_sub = A[:, self.para].contiguous()
        Qpts = (
            self.fixed[:, self.gflat].contiguous()
            if G > 0
            else torch.zeros(S, 1, 3, device=dev)
        )
        dG = fg.candidate_group_nn(
            A_sub, R_all.reshape(S * NC, 9), t_all.reshape(S * NC, 3), Qpts
        )  # [S*NC, G]
        dgm = torch.full((S * NC, n, m), inf, device=dev)
        if G > 0:
            dgm[:, self.gslot[:, 0], self.gslot[:, 1]] = dG[:, :G]
        d_g = dgm.min(-1).values.double()  # [S*NC, n]
        d_sorted = torch.sort(d_g, dim=1, stable=True).values
        viol_k = torch.relu(d_sorted[:, : self.k] - TARGET)
        E = 0.5 * viol_k.square().sum(1)
        dl = torch.where(
            torch.isfinite(d_g), model.delta_group(d_g), torch.zeros_like(d_g)
        )
        W = (torch.relu(d_g - TARGET) * dl + 0.5 * dl * dl).sum(1) + model.rel_round * E
        dk = d_sorted[:, self.k - 1]
        zero = dk < TARGET - model.delta_group(dk)
        zero = zero | torch.isinf(
            E
        )  # inf (a fully padded group among the k smallest) is exact
        return E.view(S, NC), W.view(S, NC), zero.view(S, NC)

    def describe(self, d):
        return self.reached(d)


# ======================================================================================================================
# error model of the search (two-sided: the kernel's in-register placement vs the mode's placement by two bmm + add)
# ======================================================================================================================
class _Model:
    def __init__(self, A, R, t, ra, rb, h):
        amax = float(A.norm(dim=-1).max())
        rmax = float(R.norm(dim=-1).max()) * (1.0 + 1e-6)
        tmax = float(t.norm(dim=-1).max())
        self.L = amax * rmax + tmax
        u_mm = _U
        try:
            if (
                torch.backends.cuda.matmul.allow_tf32
                or torch.get_float32_matmul_precision() != "highest"
            ):
                u_mm = 2.0**-11
        except Exception:
            pass
        # consumer: (A - c) [u L_v] -> bmm(., M1^T) -> bmm(., M2^T) -> + shift: <= 22.4 u amax rmax + 2 u L  (L_v <= 2 amax);
        # ours: R = fl(M2 M1) (entry error 3u -> 3 sqrt(3) u amax), fl(R a) [3u], t = fl(shift - fl(c R^T)), + t: <= 11.2 u amax rmax + 2 u L
        self.c_coord = 34.0 * u_mm * amax * rmax + 4.0 * _U * self.L
        self.rsum_max = float(ra.max() + rb.max())
        self.delta_d = (
            math.sqrt(3.0) * self.c_coord
            + 16.0 * _U * h
            + 4.0 * _U * SOFT * self.rsum_max
        )
        self.rel_round = 1024.0 * _U
        self.rel_comb = 16.0 * _U
        self.h = h

    def set_depth(self, max_cell_count):
        self.rel_round = _U * (512.0 + 27.0 * float(max_cell_count) + 8.0) * 1.1

    def delta_pair(self, d, lo, up):
        return (
            math.sqrt(3.0) * self.c_coord
            + 12.0 * _U * d
            + 2.0 * _U * torch.maximum(torch.maximum(d, up), lo)
        )

    def delta_group(self, d):
        return (
            math.sqrt(3.0) * self.c_coord
            + 16.0 * _U * d
            + 4.0 * _U * torch.clamp_min(d, TARGET)
        )


# ======================================================================================================================
# refinement
# ======================================================================================================================
@torch.no_grad()
def refine(
    X0,
    ra,
    B,
    rb,
    term,
    iterations=40,
    gate=True,
    n_backtrack=10,
    block=32,
    warps=1,
):
    """X0 [S,M,3], ra [M], B [S,N,3], rb [N].  Returns (X [S,M,3] in X0's dtype, accepted [S] long, info dict with
    first_energy / final_energy [S] fp32, aux_first / aux_final (term aux: pair distances or group distances), early_out bool)."""
    if not cdist_order_ok():
        raise RuntimeError(
            "torch.cdist reduction order differs from the one the kernels reproduce; dense path"
        )
    S, M, _ = X0.shape
    dev = X0.device
    D = term.D
    identity = torch.eye(3, device=dev).expand(S, 3, 3)
    moving = X0.clone().float()
    fixed = B.float()
    ra = ra.float()
    rb = rb.float()
    field = fr.ClashField(fixed, rb, ra, moving, block=block, warps=warps)
    K = n_backtrack
    scales = torch.tensor([0.5**b for b in range(K)], device=dev, dtype=torch.float32)

    def evaluate(x):
        vsum, aux = term.energy(x[:, None])
        ce, depth, _, _ = field.run(x[:, None], None, with_grad=False)
        return 0.5 * (vsum[:, 0] + W_REFINE * ce[:, 0]), aux[:, 0] if torch.is_tensor(
            aux
        ) else aux

    first_energy, aux_first = evaluate(moving)
    accepted = torch.zeros(S, dtype=torch.long, device=dev)
    if gate and bool(term.satisfied(aux_first).all()):
        # exact early-out: no sample can move (accept requires ~satisfied), the loop would return X0 unchanged
        return (
            moving.to(X0.dtype),
            accepted,
            dict(
                first_energy=first_energy,
                final_energy=first_energy,
                aux_first=aux_first,
                aux_final=aux_first,
                early_out=True,
            ),
        )
    for _ in range(iterations):
        vsum, grad_t, aux = term.energy_grad(moving)
        ce, depth, _, grad = field.run(moving[:, None], None, with_grad=True)
        ce = ce[:, 0]
        depth = depth[:, 0]
        grad = grad[:, 0] + grad_t
        energy = 0.5 * (vsum + W_REFINE * ce)
        satisfied = (
            term.satisfied(aux)
            if gate
            else torch.zeros(S, dtype=torch.bool, device=dev)
        )
        center = moving.mean(1, keepdim=True)
        centered = moving - center
        translation = -0.2 * grad.sum(1) / D
        translation *= 0.5 / torch.linalg.vector_norm(
            translation, dim=-1, keepdim=True
        ).clamp_min(0.5)
        torque = torch.linalg.cross(centered, grad, dim=-1).sum(1)
        inertia = centered.square().sum((1, 2))[:, None, None] * identity - torch.bmm(
            centered.transpose(1, 2), centered
        )
        rotation = (
            -0.15
            * M
            / D
            * torch.linalg.solve(inertia + 1e-3 * identity, torque[..., None]).squeeze(
                -1
            )
        )
        rotation *= 0.03 / torch.linalg.vector_norm(
            rotation, dim=-1, keepdim=True
        ).clamp_min(0.03)
        matrix = torch.linalg.matrix_exp(
            fr.skew_matrix(rotation[:, None] * scales[None, :, None])
        )  # [S,K,3,3]
        proposal = torch.bmm(
            centered[:, None].expand(-1, K, -1, -1).reshape(S * K, M, 3),
            matrix.transpose(-1, -2).reshape(S * K, 3, 3),
        ).view(S, K, M, 3)
        proposal = (
            proposal
            + center[:, None]
            + (translation[:, None, None] * scales[None, :, None, None])
        )
        vsum_k, _ = term.energy(proposal)
        ce_k, depth_k, new_k, _ = field.run(proposal, moving, with_grad=False)
        new_energy = 0.5 * (vsum_k + W_REFINE * ce_k)  # [S,K]
        accept_k = (
            (~satisfied)[:, None]
            & ~new_k
            & (depth_k <= depth[:, None] + 1e-6)
            & (new_energy < energy[:, None] - 1e-7)
        )
        found = accept_k.any(1)
        if not bool(found.any()):
            break
        first = torch.argmax(accept_k.to(torch.int8), dim=1)
        chosen = proposal[torch.arange(S, device=dev), first]
        moving = torch.where(found[:, None, None], chosen, moving)
        accepted += found.long()
    _, _, new_final, _ = field.run(moving[:, None], X0.float(), with_grad=False)
    if bool(new_final.any()):
        raise RuntimeError("rigid refinement introduced a severe interchain clash")
    final_energy, aux_final = evaluate(moving)
    return (
        moving.to(X0.dtype),
        accepted,
        dict(
            first_energy=first_energy,
            final_energy=final_energy,
            aux_first=aux_first,
            aux_final=aux_final,
            early_out=False,
        ),
    )


# ======================================================================================================================
# search
# ======================================================================================================================
@torch.no_grad()
def search(A, ra, B, rb, term, R, t, place, satisfied, block=128, warps=1):
    """A [S,M,3] entry pose, R [S,P,3,3], t [S,P,3] the candidate transforms (X_p = R_p a + t_p up to rounding), place(p) -> X [S,M,3]
    the mode's own construction of candidate p for every sample (used for the exact re-evaluation and by the caller for the winner),
    satisfied [S] bool: samples that are left untouched (best = -1).
    Returns dict(best Long[S] (-1: keep the entry pose), best_energy fp32[S] (exact; the entry energy, inf if severe, when best = -1),
    entry_energy fp32[S] (exact, raw), entry_severe bool[S], n_feasible, n_improving Long[S] (fast-table counts, logging), stats)."""
    if not cdist_order_ok():
        raise RuntimeError(
            "torch.cdist reduction order differs from the one the kernels reproduce; dense path"
        )
    A = A.contiguous().float()
    B = B.contiguous().float()
    ra = ra.contiguous().float()
    rb = rb.contiguous().float()
    R = R.contiguous().float()
    t = t.contiguous().float()
    dev = A.device
    S, M, _ = A.shape
    P = R.shape[1]
    NC = P + 1
    inf = float("inf")
    field = fr.ClashField(
        B, rb, ra, A, block=32, warps=1
    )  # cells of B (shared by the shortlist kernel) + exact clash evaluator
    cells = field.cells
    h = field.h
    cutoff = SOFT * float(ra.max() + rb.max())
    model = _Model(A, R, t, ra, rb, h)
    model.set_depth(int(cells["cnt"].max()))
    assert h > cutoff + model.delta_d, "coordinate scale too large for the cell margin"
    A_p = field.gather(A)
    ra_p = field.ras
    eye = torch.eye(3, device=dev, dtype=torch.float32).expand(S, 1, 3, 3)
    R_all = torch.cat([R, eye], 1).contiguous()
    t_all = torch.cat([t, torch.zeros(S, 1, 3, device=dev)], 1).contiguous()
    nblk = triton.cdiv(M, block)
    part = torch.empty(S * NC, nblk, 5, dtype=torch.float32, device=dev)
    psk.search_kernel[(S * NC, nblk)](
        A_p,
        ra_p,
        R_all.view(S * NC, 9),
        t_all.view(S * NC, 3),
        cells["Bs"],
        cells["rbs"],
        cells["start"],
        cells["cnt"],
        cells["lo"],
        cells["dims"],
        cells["coff"],
        part,
        M,
        NC,
        nblk,
        1.0 / h,
        float(model.delta_d),
        SOFT_C=SOFT,
        HARD_C=HARD,
        BLOCK=block,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    pd = part.double()
    clash = pd[:, :, 0].sum(1)
    sum_o = pd[:, :, 1].sum(1)
    n_near = pd[:, :, 2].sum(1)
    m_hard = part[:, :, 3].min(1).values
    m_soft = part[:, :, 4].min(1).values
    m_hard = torch.where(
        m_hard > 1.0e29, torch.full_like(m_hard, h - HARD * model.rsum_max), m_hard
    )
    m_soft = torch.where(
        m_soft > 1.0e29, torch.full_like(m_soft, h - SOFT * model.rsum_max), m_soft
    )
    E_term, W_term, zero_term = term.fast_candidates(A, R_all, t_all, model)
    dd = model.delta_d
    E = E_term + W_SEARCH * clash.view(S, NC)
    w_clash = 2.0 * sum_o * dd + n_near * dd * dd + model.rel_round * clash
    W = W_term + W_SEARCH * w_clash.view(S, NC) + model.rel_comb * E + 1e-9
    exact_zero = (
        (clash.view(S, NC) == 0) & (m_soft.view(S, NC).double() > dd) & zero_term
    )
    W = torch.where(exact_zero, torch.zeros_like(W), W)

    # ---- exact evaluation (all samples of one candidate at once; the entry pose always) ------------------------------
    cache = {}

    def exact_all(p):
        if p not in cache:
            X = A if p == P else place(p).contiguous().float()
            ce, depth, _, _ = field.run(X[:, None], None, with_grad=False)
            e = term.search_exact(X) + W_SEARCH * ce[:, 0]
            cache[p] = (e.cpu(), (depth[:, 0] > 0).cpu())
        return cache[p]

    entry_e, entry_sev = exact_all(P)
    # ---- host decision (one transfer of the tables) -------------------------------------------------------------------
    Eh = E.cpu()
    Wh = W.cpu()
    mh = m_hard.view(S, NC).cpu()
    sat = satisfied.bool().cpu()
    cert_feas = mh > dd
    cert_sev = mh < -dd
    exact_e = Wh == 0
    exact_f = cert_feas | cert_sev
    for s in range(S):
        Eh[s, P] = float(entry_e[s])
        Wh[s, P] = 0.0
        exact_e[s, P] = True
        cert_sev[s, P] = bool(entry_sev[s])
        cert_feas[s, P] = not bool(entry_sev[s])
        exact_f[s, P] = True
    best = torch.full((S,), -1, dtype=torch.long)
    best_e = torch.zeros(S, dtype=torch.float32)
    nfeas = torch.zeros(S, dtype=torch.long)
    nimp = torch.zeros(S, dtype=torch.long)
    stats = dict(
        reeval=0, shortlist=[], ambiguous_margin=int((~exact_f).sum()), max_rel_dev=0.0
    )

    def reeval(s, p):
        e_all, sv_all = exact_all(p)
        ef = float(e_all[s])
        sv = bool(sv_all[s])
        if math.isfinite(ef) and ef > 0:
            stats["max_rel_dev"] = max(
                stats["max_rel_dev"], abs(float(Eh[s, p]) - ef) / ef
            )
        stats["reeval"] += 1
        Eh[s, p] = ef
        Wh[s, p] = 0.0
        exact_e[s, p] = True
        cert_sev[s, p] = sv
        cert_feas[s, p] = not sv
        exact_f[s, p] = True

    for s in range(S):
        Es = Eh[s]
        Ws = Wh[s]
        cur = inf if bool(cert_sev[s, P]) else float(Es[P])
        if bool(sat[s]):
            best[s] = -1
            best_e[s] = cur
        else:
            while True:
                E_lo = Es[:P] - Ws[:P]
                E_hi = Es[:P] + Ws[:P]
                pool = ~cert_sev[s, :P] & (E_lo < cur)
                feas_pool = pool & cert_feas[s, :P]
                m_star = float(E_hi[feas_pool].min()) if bool(feas_pool.any()) else inf
                short = pool & (E_lo <= m_star)
                need = short & ~(exact_e[s, :P] & exact_f[s, :P])
                if not bool(need.any()):
                    break
                for p in torch.nonzero(need).flatten().tolist():
                    reeval(s, p)
            stats["shortlist"].append(int(short.sum()))
            fin = short & cert_feas[s, :P]
            if bool(fin.any()):
                vals = torch.where(fin, Es[:P], torch.full_like(Es[:P], inf))
                emin = float(vals.min())
                pbest = int(torch.nonzero(vals == emin)[0])
                if emin < cur:
                    best[s] = pbest
                    best_e[s] = emin
                else:
                    best[s] = -1
                    best_e[s] = cur
            else:
                best[s] = -1
                best_e[s] = cur
        feas = ~cert_sev[s, :P]
        nfeas[s] = int(feas.sum())
        elig = feas if not bool(sat[s]) else torch.zeros_like(feas)
        run = torch.where(elig, Es[:P], torch.full_like(Es[:P], inf)).cummin(0).values
        before = torch.cat([torch.tensor([cur], dtype=run.dtype), run[:-1]])
        before = torch.minimum(before, torch.full_like(before, cur))
        nimp[s] = int((Es[:P] < before).sum())
    stats["cache"] = len(cache) - 1
    _STATS.clear()
    _STATS.update(stats)
    return {
        "best": best.to(dev),
        "best_energy": best_e.to(dev),
        "entry_energy": entry_e.to(dev),
        "entry_severe": entry_sev.to(dev),
        "n_feasible": nfeas.to(dev),
        "n_improving": nimp.to(dev),
        "stats": stats,
    }


def mode():
    """OPENDDE_RIGID_CORE: off (dense implementation), on (this core, dense fallback on any exception), check (both; the dense
    result is returned and the differences are logged). The dispatchers read it through rigid_contact.rigid_core_mode, which does not
    import Triton."""
    from opendde.tfg.rigid_contact import rigid_core_mode

    return rigid_core_mode()
