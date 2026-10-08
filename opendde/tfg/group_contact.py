"""Fused group-minimum-distance term of the epitope (pocket) mode.

For every pose q and every epitope group g (the heavy atoms of one epitope residue, padded with -1) the term needs the
smallest distance from any atom of the group to any probe atom (the paratope atoms of the moving group) and the atom
pair that attains it:
    dist[q, g, j, p] = ||B[s(q), grp[g, j]] - A[q, para[p]]||      (torch.cdist, donot_use_mm, fp32)
    dist[.., j, ..] = +inf where grp[g, j] < 0
    d[q, g], arg = min over the flat index (j * P + p)  -> a_slot = arg // P, p_slot = arg % P   (first index among equal values)
The energy is 0.5 * sum over the k smallest d (stable sort) of relu(d - 4.5)^2, and each of the k selected groups pulls its
probe atom towards its fixed atom with the force relu(d - 4.5) * vec / max(|vec|, 1e-6).

The kernel reproduces every distance bit for bit: the fp32 formula of torch.cdist(donot_use_mm) (squares rounded
separately, summed as (dx^2 + dz^2) + dy^2, IEEE sqrt; clash_field.cdist_order_matches() checks that the installed torch
reduces in that order), the per-group minimum and the first-index tie rule on the flat index j * P + p, and +inf for padded
slots. One program per (pose, group) runs a fixed tiled loop over the probe points, with no atomics and no
data-dependent reduction order. Energy and gradient are ordinary torch expressions on the returned (d, a_slot, p_slot).

Entry points
    group_min_distance_fast(A [Q,M,3], B [S,N,3], inst_of_pose [Q] long, grp [n,m] long, para [P] long)
        -> d [Q,n] fp32, a_slot [Q,n] long, p_slot [Q,n] long
    group_min_distance_dense(...)   the same values from torch.cdist
    contact_terms(d, k)             energy, violations and indices of the k closest groups
    candidate_group_nn(A_sub [S,Pm,3], R_all [S*NC,9], t_all [S*NC,3], Q [S,G,3]) -> [S*NC, G] nearest-probe distance of every
        valid group point for every candidate pose, placed in-kernel (the bound used by the shortlist stage of rigid_core.search)
    nearest_fixed_distance(X, B)    nearest fixed atom of every moving atom
"""

import torch
import triton
import triton.language as tl

TARGET = 4.5


# ----------------------------------------------------------------------------------------------------------------------
# kernel: per (pose, group) minimum distance + first-index argmin over (slot, probe)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _group_min_kernel(
    A_ptr,
    para_ptr,
    B_ptr,
    grp_ptr,
    inst_ptr,
    d_ptr,
    i_ptr,
    M,
    N,
    P,
    n,
    m,
    MS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    q = tl.program_id(0)
    g = tl.program_id(1)
    s = tl.load(inst_ptr + q).to(tl.int64)
    moffs = tl.arange(0, MS)
    mm = moffs < m
    gidx = tl.load(grp_ptr + g * m + moffs, mask=mm, other=-1).to(tl.int64)
    valid = mm & (gidx >= 0)
    bb = (s * N + tl.where(valid, gidx, 0)) * 3
    bx = tl.load(B_ptr + bb + 0, mask=valid, other=0.0)
    by = tl.load(B_ptr + bb + 1, mask=valid, other=0.0)
    bz = tl.load(B_ptr + bb + 2, mask=valid, other=0.0)
    best = tl.full([], float("inf"), dtype=tl.float32)
    bidx = tl.full([], 0, dtype=tl.int32)
    BIG: tl.constexpr = 2147483647
    q64 = q.to(tl.int64)
    for p0 in range(0, P, BLOCK_P):
        poffs = p0 + tl.arange(0, BLOCK_P)
        pm = poffs < P
        pa = tl.load(para_ptr + poffs, mask=pm, other=0).to(tl.int64)
        ab = (q64 * M + pa) * 3
        x = tl.load(A_ptr + ab + 0, mask=pm, other=0.0)
        y = tl.load(A_ptr + ab + 1, mask=pm, other=0.0)
        z = tl.load(A_ptr + ab + 2, mask=pm, other=0.0)
        dx = bx[:, None] - x[None, :]
        dy = by[:, None] - y[None, :]
        dz = bz[:, None] - z[None, :]
        acc = (
            (dx * dx + dz * dz) + dy * dy
        )  # torch.cdist's reduction order (x, z, y), squares rounded separately
        d = tl.sqrt_rn(acc)
        ok = valid[:, None] & pm[None, :]
        d = tl.where(ok, d, float("inf"))
        v = tl.min(tl.min(d, axis=1), axis=0)
        flat = moffs[:, None] * P + poffs[None, :]
        cand = tl.min(tl.min(tl.where(d == v, flat, BIG), axis=1), axis=0)
        take = v < best
        eq = v == best
        bidx = tl.where(take, cand, tl.where(eq, tl.minimum(bidx, cand), bidx))
        best = tl.where(take, v, best)
    tl.store(d_ptr + q64 * n + g, best)
    tl.store(i_ptr + q64 * n + g, bidx)


def _pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


@torch.no_grad()
def group_min_distance_fast(A, B, inst_of_pose, grp, para, block_p=128, warps=4):
    """A [Q,M,3] fp32, B [S,N,3] fp32, inst_of_pose [Q], grp [n,m] long (-1 padded), para [P] long -> (d [Q,n], a_slot [Q,n], p_slot [Q,n])."""
    A = A.contiguous()
    B = B.contiguous()
    Q, M, _ = A.shape
    S, N, _ = B.shape
    n, m = grp.shape
    P = para.numel()
    dev = A.device
    if A.dtype != torch.float32 or B.dtype != torch.float32:
        raise TypeError("group_min_distance_fast expects float32 coordinates")
    d = torch.empty(Q, n, dtype=torch.float32, device=dev)
    idx = torch.empty(Q, n, dtype=torch.int32, device=dev)
    if Q == 0 or n == 0:
        return d, idx.long(), idx.long()
    grp_c = grp.to(dev).contiguous()
    para_c = para.to(dev).contiguous()
    inst_c = inst_of_pose.to(dev).to(torch.int32).contiguous()
    MS = _pow2(max(m, 2))
    _group_min_kernel[(Q, n)](
        A,
        para_c,
        B,
        grp_c,
        inst_c,
        d,
        idx,
        M,
        N,
        P,
        n,
        m,
        MS=MS,
        BLOCK_P=block_p,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    idx = idx.long()
    return d, idx // P, idx % P


def group_min_distance_dense(A, B, inst_of_pose, grp, para):
    """The same values from torch.cdist. B is gathered per pose through inst_of_pose."""
    Q, n, m = A.shape[0], grp.shape[0], grp.shape[1]
    valid = grp >= 0
    B_inst = B[inst_of_pose.long()]
    gp = B_inst[:, grp.clamp_min(0).reshape(-1)].reshape(Q, n, m, 3)
    dist = torch.cdist(
        gp.reshape(Q, n * m, 3), A[:, para], compute_mode="donot_use_mm_for_euclid_dist"
    ).reshape(Q, n, m, -1)
    dist = dist.masked_fill(~valid[None, :, :, None], float("inf"))
    dmin, arg = dist.reshape(Q, n, -1).min(-1)
    P = para.numel()
    return dmin, arg // P, arg % P


def contact_terms(d, k):
    """0.5 * sum over the k smallest d of relu(d - TARGET)^2 (stable sort: ties keep the group order)."""
    ordered, order = torch.sort(d, dim=-1, stable=True)
    vals, idx = ordered[:, :k], order[:, :k]
    violation = torch.relu(vals - TARGET)
    return 0.5 * violation.square().sum(-1), violation, idx


# ----------------------------------------------------------------------------------------------------------------------
# shortlist stage: nearest-probe distance of every valid group point for every candidate pose (bound, not bit-matched)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _candidate_nn_kernel(
    A_ptr,
    R_ptr,
    t_ptr,
    Q_ptr,
    out_ptr,
    Pm,
    NC,
    G,
    BLOCK_G: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    gc = tl.program_id(0)  # global candidate id = s * NC + c
    gb = tl.program_id(1)
    s = (gc // NC).to(tl.int64)
    gc64 = gc.to(tl.int64)
    goffs = gb * BLOCK_G + tl.arange(0, BLOCK_G)
    gm = goffs < G
    qb = (s * G + goffs) * 3
    qx = tl.load(Q_ptr + qb + 0, mask=gm, other=0.0)
    qy = tl.load(Q_ptr + qb + 1, mask=gm, other=0.0)
    qz = tl.load(Q_ptr + qb + 2, mask=gm, other=0.0)
    rp = R_ptr + gc64 * 9
    r00 = tl.load(rp + 0)
    r01 = tl.load(rp + 1)
    r02 = tl.load(rp + 2)
    r10 = tl.load(rp + 3)
    r11 = tl.load(rp + 4)
    r12 = tl.load(rp + 5)
    r20 = tl.load(rp + 6)
    r21 = tl.load(rp + 7)
    r22 = tl.load(rp + 8)
    t0 = tl.load(t_ptr + gc64 * 3 + 0)
    t1 = tl.load(t_ptr + gc64 * 3 + 1)
    t2 = tl.load(t_ptr + gc64 * 3 + 2)
    best = tl.full([BLOCK_G, BLOCK_M], 1.0e30, dtype=tl.float32)
    for m0 in range(0, Pm, BLOCK_M):
        moffs = m0 + tl.arange(0, BLOCK_M)
        mm = moffs < Pm
        ab = (s * Pm + moffs) * 3
        a0 = tl.load(A_ptr + ab + 0, mask=mm, other=0.0)
        a1 = tl.load(A_ptr + ab + 1, mask=mm, other=0.0)
        a2 = tl.load(A_ptr + ab + 2, mask=mm, other=0.0)
        x = r00 * a0 + r01 * a1 + r02 * a2 + t0
        y = r10 * a0 + r11 * a1 + r12 * a2 + t1
        z = r20 * a0 + r21 * a1 + r22 * a2 + t2
        x = tl.where(mm, x, 1.0e15)
        dx = qx[:, None] - x[None, :]
        dy = qy[:, None] - y[None, :]
        dz = qz[:, None] - z[None, :]
        best = tl.minimum(best, dx * dx + dy * dy + dz * dz)
    tl.store(out_ptr + gc64 * G + goffs, tl.sqrt(tl.min(best, axis=1)), mask=gm)


@torch.no_grad()
def candidate_group_nn(A_sub, R_all, t_all, Qpts, block_g=32, block_m=64, warps=1):
    """A_sub [S,Pm,3] (probe points of every instance), R_all [S*NC,9], t_all [S*NC,3], Qpts [S,G,3] -> [S*NC, G] fp32."""
    S, Pm, _ = A_sub.shape
    G = Qpts.shape[1]
    SNC = R_all.shape[0]
    NC = SNC // S
    out = torch.empty(SNC, max(G, 1), dtype=torch.float32, device=A_sub.device)
    if G == 0 or SNC == 0:
        return out
    _candidate_nn_kernel[(SNC, triton.cdiv(G, block_g))](
        A_sub.contiguous(),
        R_all.contiguous(),
        t_all.contiguous(),
        Qpts.contiguous(),
        out,
        Pm,
        NC,
        G,
        BLOCK_G=block_g,
        BLOCK_M=block_m,
        num_warps=warps,
    )
    return out


# ----------------------------------------------------------------------------------------------------------------------
# nearest fixed atom of every moving atom (bit-identical to torch.cdist(X, B, donot_use_mm).min(-1).values)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _nn_min_kernel(
    X_ptr, B_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    s = tl.program_id(0).to(tl.int64)
    mb = tl.program_id(1)
    moffs = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = moffs < M
    xb = (s * M + moffs) * 3
    x = tl.load(X_ptr + xb + 0, mask=mm, other=0.0)
    y = tl.load(X_ptr + xb + 1, mask=mm, other=0.0)
    z = tl.load(X_ptr + xb + 2, mask=mm, other=0.0)
    best = tl.full([BLOCK_M], float("inf"), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        noffs = n0 + tl.arange(0, BLOCK_N)
        nm = noffs < N
        bb = (s * N + noffs) * 3
        bx = tl.load(B_ptr + bb + 0, mask=nm, other=0.0)
        by = tl.load(B_ptr + bb + 1, mask=nm, other=0.0)
        bz = tl.load(B_ptr + bb + 2, mask=nm, other=0.0)
        dx = x[:, None] - bx[None, :]
        dy = y[:, None] - by[None, :]
        dz = z[:, None] - bz[None, :]
        d = tl.sqrt_rn((dx * dx + dz * dz) + dy * dy)
        d = tl.where(nm[None, :], d, float("inf"))
        best = tl.minimum(best, tl.min(d, axis=1))
    tl.store(out_ptr + s * M + moffs, best, mask=mm)


@torch.no_grad()
def nearest_fixed_distance(X, B, block_m=64, block_n=64, warps=4):
    """X [S,M,3], B [S,N,3] fp32 -> [S,M] distance of every moving atom to its nearest fixed atom (cdist-identical values)."""
    X = X.contiguous().float()
    B = B.contiguous().float()
    S, M, _ = X.shape
    N = B.shape[1]
    out = torch.empty(S, M, dtype=torch.float32, device=X.device)
    _nn_min_kernel[(S, triton.cdiv(M, block_m))](
        X,
        B,
        out,
        M,
        N,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out
