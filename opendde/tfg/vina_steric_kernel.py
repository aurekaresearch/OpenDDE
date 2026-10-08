"""Fused short-range pair potential between atom groups (the Vina steric term of the TFG engine).

PairPotential(group, r, allowed, buffer=0.225)            built once per run (static group / radius / filter)
  .energy_and_grad(X [S,N,3] float32 cuda) -> (E [S] float32, G [S,N,3] float32)

Semantics (identical to VinaStericPotential's dense evaluation): over all unordered cross-group pairs (i, j) with both
groups valid (> 1 point) and allowed[g_i, g_j], the pair is active iff d < (r_i + r_j) * (1 - buffer) with d the fp32
||X_i - X_j|| (torch.linalg.norm of the fp32 difference, clamped at 1e-8); for active pairs, with delta = d - r_eq,
  g1 = -0.0356 exp(-(delta/0.5)^2),  g2 = -0.00516 exp(-((delta-3)/2)^2),  e = g1 + g2 + 0.840 [delta<0] delta^2,
  dE/dd = -8 delta g1 - 0.5 (delta-3) g2 + 1.68 [delta<0] delta,   G_i += dE (X_i - X_j)/d,  G_j -= the same.

Algorithm
---------
* Points of every cloud are sorted once (and again every `reorder_every` calls) by (group, Morton code of a 2 A cell):
  blocks of B consecutive sorted points are spatially compact and mostly single-group. The order only affects speed,
  never the result.
* Per call (3 Triton launches + 3 small torch ops, no host synchronisation):
  1. gather X into the sorted order; torch.aminmax -> bounding box of every block;
  2. `_tiles_kernel`: one program per (cloud, i-block), vectorised over all j-blocks: a tile survives iff some
     (group_i, group_j) pair of the two blocks is allowed (block bitmasks) and the distance between the two bounding
     boxes is <= cutoff_max + 1e-3 A (cutoff_max = (1-buffer) * 2 r_max). Surviving j-blocks are compacted with a
     prefix sum into a per-row list (CSR with fixed row stride; deterministic, no atomics);
  3. `_pair_kernel`: one program per (cloud, i-block), loops over its surviving tiles only. Per pair: the fp32
     distance (same operation order as torch.linalg.norm), the threshold (r_i + r_j) * fp32(1-buffer), the group
     filter; the energy / force terms are computed in float64 only for tiles that contain an active pair
     (tile-uniform branch). Each pair is visited from both sides (row i accumulates its own force over all j: no
     atomics, fixed order, bit-identical run to run); the energy is counted from the side with the smaller sorted
     index. Per-row float64 accumulators, per-program partial energies, torch float64 sum over programs, one rounding
     to fp32.
  No active pair is missed: every pair (i, j) belongs to exactly one tile (i-block, j-block) and the tile is tested
  unless (a) no group pair of the two blocks is allowed - then (i, j) is not a candidate - or (b) the box distance
  exceeds cutoff_max + 1e-3: then d_true >= box distance > cutoff_max + 1e-3 - (fp32 rounding of the box arithmetic,
  < 1e-5 A) > cutoff_max >= (r_i + r_j)(1 - buffer), and d >= d_true (1 - 4u) is still above the threshold, so the pair
  is inactive. The tile kernel is re-run on the current coordinates every call, so nothing can go stale.

Exactness of the active set: identical to the dense evaluation whenever this torch build reduces linalg.norm over 3
elements in the order (dx^2 + dz^2) + dy^2 with separately rounded squares (`norm_order_matches()` checks it in 10 ms).
On a build that differs, pairs whose d is within 1 fp32 ulp of the threshold could be classified differently; the
energy jumps by ~0.5 at the threshold, so this would be visible - hence the check.
"""

import torch
import triton
import triton.language as tl

_U = 2.0**-24


# ----------------------------------------------------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _tiles_kernel(
    lo_ptr, hi_ptr, bg_ptr, bm_ptr, cnt_ptr, cols_ptr, nblk, CUT2, BLOCKJ: tl.constexpr
):
    """row = s * nblk + bi -> cnt[row] surviving j-blocks, their indices (ascending) in cols[row * nblk : ...]."""
    row = tl.program_id(0)
    s = row // nblk
    rb = row.to(tl.int64)
    lo_ix = tl.load(lo_ptr + rb * 3 + 0)
    lo_iy = tl.load(lo_ptr + rb * 3 + 1)
    lo_iz = tl.load(lo_ptr + rb * 3 + 2)
    hi_ix = tl.load(hi_ptr + rb * 3 + 0)
    hi_iy = tl.load(hi_ptr + rb * 3 + 1)
    hi_iz = tl.load(hi_ptr + rb * 3 + 2)
    um_i = tl.load(bm_ptr + rb)
    offs = tl.arange(0, BLOCKJ)
    m = offs < nblk
    jb = (s * nblk + offs).to(tl.int64)
    lo_jx = tl.load(lo_ptr + jb * 3 + 0, mask=m, other=1.0e30)
    lo_jy = tl.load(lo_ptr + jb * 3 + 1, mask=m, other=1.0e30)
    lo_jz = tl.load(lo_ptr + jb * 3 + 2, mask=m, other=1.0e30)
    hi_jx = tl.load(hi_ptr + jb * 3 + 0, mask=m, other=-1.0e30)
    hi_jy = tl.load(hi_ptr + jb * 3 + 1, mask=m, other=-1.0e30)
    hi_jz = tl.load(hi_ptr + jb * 3 + 2, mask=m, other=-1.0e30)
    gbits = tl.load(bg_ptr + jb, mask=m, other=0)
    dgx = tl.maximum(tl.maximum(lo_jx - hi_ix, lo_ix - hi_jx), 0.0)
    dgy = tl.maximum(tl.maximum(lo_jy - hi_iy, lo_iy - hi_jy), 0.0)
    dgz = tl.maximum(tl.maximum(lo_jz - hi_iz, lo_iz - hi_jz), 0.0)
    box2 = dgx * dgx + dgy * dgy + dgz * dgz
    ok = m & ((um_i & gbits) != 0) & (box2 <= CUT2)
    oki = ok.to(tl.int32)
    pos = tl.cumsum(oki, axis=0) - oki
    tl.store(cols_ptr + rb * nblk + pos, offs.to(tl.int32), mask=ok)
    tl.store(cnt_ptr + row, tl.sum(oki, axis=0))


@triton.jit
def _pair_kernel(
    X_ptr,
    r_ptr,
    g_ptr,
    m_ptr,
    perm_ptr,
    cnt_ptr,
    cols_ptr,
    e_ptr,
    G_ptr,
    N,
    Npad,
    nblk,
    SCALE,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid // nblk
    bi = pid % nblk
    rb = pid.to(tl.int64)
    offs_i = bi * B + tl.arange(0, B)
    mi_valid = offs_i < N
    base_i = s.to(tl.int64) * Npad + offs_i
    xi = tl.load(X_ptr + base_i * 3 + 0, mask=mi_valid, other=0.0)
    yi = tl.load(X_ptr + base_i * 3 + 1, mask=mi_valid, other=0.0)
    zi = tl.load(X_ptr + base_i * 3 + 2, mask=mi_valid, other=0.0)
    ri = tl.load(r_ptr + base_i, mask=mi_valid, other=0.0)
    gi = tl.load(g_ptr + base_i, mask=mi_valid, other=31)
    mi = tl.load(m_ptr + base_i, mask=mi_valid, other=0)
    gx = tl.zeros([B], dtype=tl.float64)
    gy = tl.zeros([B], dtype=tl.float64)
    gz = tl.zeros([B], dtype=tl.float64)
    ea = tl.zeros([B], dtype=tl.float64)
    cnt = tl.load(cnt_ptr + pid)
    for t in range(0, cnt):
        bj = tl.load(cols_ptr + rb * nblk + t)
        offs_j = bj * B + tl.arange(0, B)
        mj_valid = offs_j < N
        base_j = s.to(tl.int64) * Npad + offs_j
        xj = tl.load(X_ptr + base_j * 3 + 0, mask=mj_valid, other=0.0)
        yj = tl.load(X_ptr + base_j * 3 + 1, mask=mj_valid, other=0.0)
        zj = tl.load(X_ptr + base_j * 3 + 2, mask=mj_valid, other=0.0)
        rj = tl.load(r_ptr + base_j, mask=mj_valid, other=0.0)
        gj = tl.load(g_ptr + base_j, mask=mj_valid, other=31)
        dx = xi[:, None] - xj[None, :]
        dy = yi[:, None] - yj[None, :]
        dz = zi[:, None] - zj[None, :]
        # fp32 distance as torch.linalg.norm computes it: squares rounded separately, summed as (dx^2 + dz^2) + dy^2, IEEE sqrt
        d32 = tl.sqrt_rn((dx * dx + dz * dz) + dy * dy)
        thr = (ri[:, None] + rj[None, :]) * SCALE
        allowed = ((mi[:, None] >> gj[None, :]) & 1) != 0
        active = (
            (d32 < thr)
            & allowed
            & (gi[:, None] != gj[None, :])
            & mi_valid[:, None]
            & mj_valid[None, :]
        )
        if tl.max(active.to(tl.int32)) > 0:
            dxd = dx.to(tl.float64)
            dyd = dy.to(tl.float64)
            dzd = dz.to(tl.float64)
            d64 = tl.sqrt(dxd * dxd + dyd * dyd + dzd * dzd)
            d64 = tl.maximum(d64, 1.0e-8)
            req = ri[:, None].to(tl.float64) + rj[None, :].to(tl.float64)
            delta = d64 - req
            g1 = -0.0356 * tl.exp(-(delta / 0.5) * (delta / 0.5))
            g2 = -0.00516 * tl.exp(-((delta - 3.0) / 2.0) * ((delta - 3.0) / 2.0))
            neg = delta < 0.0
            e = g1 + g2 + 0.840 * tl.where(neg, delta * delta, 0.0)
            dE = (
                -8.0 * delta * g1
                - 0.5 * (delta - 3.0) * g2
                + tl.where(neg, 1.68 * delta, 0.0)
            )
            coef = tl.where(active, dE / d64, 0.0)
            gx += tl.sum(coef * dxd, axis=1)
            gy += tl.sum(coef * dyd, axis=1)
            gz += tl.sum(coef * dzd, axis=1)
            once = active & (offs_i[:, None] < offs_j[None, :])
            ea += tl.sum(tl.where(once, e, 0.0), axis=1)
    tl.store(e_ptr + pid, tl.sum(ea, axis=0))
    orig = tl.load(perm_ptr + base_i, mask=mi_valid, other=0)
    gb = (s.to(tl.int64) * N + orig) * 3
    tl.store(G_ptr + gb + 0, gx.to(tl.float32), mask=mi_valid)
    tl.store(G_ptr + gb + 1, gy.to(tl.float32), mask=mi_valid)
    tl.store(G_ptr + gb + 2, gz.to(tl.float32), mask=mi_valid)


# ----------------------------------------------------------------------------------------------------------------------
def _morton(c):
    """c [..., 3] int64 in 0..1023 -> 30-bit Morton code."""

    def spread(v):
        v = v & 0x3FF
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return spread(c[..., 0]) | (spread(c[..., 1]) << 1) | (spread(c[..., 2]) << 2)


def norm_order_matches(device="cuda", n=200000):
    """True iff torch.linalg.norm(v, dim=-1) on fp32 3-vectors equals sqrt((vx^2 + vz^2) + vy^2) with separately
    rounded squares on this torch build (the order the kernel reproduces)."""
    g = torch.Generator(device="cpu").manual_seed(0)
    v = ((torch.rand(n, 3, generator=g) - 0.5) * 8).to(device)
    ref = torch.linalg.norm(v, dim=-1)
    sq = v * v
    return bool(torch.equal(torch.sqrt((sq[:, 0] + sq[:, 2]) + sq[:, 1]), ref))


class PairPotential:
    def __init__(
        self,
        group,
        r,
        allowed,
        buffer=0.225,
        block=16,
        reorder_every=20,
        warps=4,
    ):
        assert block in (16, 32, 64, 128)
        self.dev = r.device
        self.N = int(r.shape[0])
        self.B = block
        self.reorder_every = reorder_every
        self.warps = warps
        group = group.long().to(self.dev)
        r = r.float().contiguous().to(self.dev)
        allowed = allowed.bool().to(self.dev)
        self.group, self.r, self.allowed = group, r, allowed
        G = int(allowed.shape[0])
        assert G <= 30
        self.G = G
        size = torch.bincount(group, minlength=G)
        valid_g = size > 1
        al = allowed.clone()
        al.fill_diagonal_(False)
        al &= valid_g[:, None] & valid_g[None, :]
        self.allowed_eff = al
        gid = torch.where(
            valid_g[group], group, torch.full_like(group, 31)
        )  # 31 = never interacts
        bits = (al.long() << torch.arange(G, device=self.dev)[None, :]).sum(
            1
        )  # [G] mask of allowed partners
        self.gid = gid.to(torch.int32)
        self.pmask = torch.where(
            valid_g[group], bits[group], torch.zeros_like(group)
        ).to(torch.int32)
        self.scale = float(
            torch.tensor(1.0 - buffer, dtype=torch.float32)
        )  # fp32(1 - buffer), as the dense evaluation multiplies by it
        self.buffer = buffer
        rmax = float(r.max())
        self.cutoff_max = self.scale * 2.0 * rmax
        self.cut2 = (self.cutoff_max + 1e-3) ** 2
        self.nblk = triton.cdiv(self.N, block)
        self.Npad = self.nblk * block
        self.blockj = max(16, triton.next_power_of_2(self.nblk))
        assert self.blockj <= 2048, (
            "N / block too large for the tile kernel (use a larger block)"
        )
        self.calls = 0
        self.perm = None
        self._S = None

    # ------------------------------------------------------------------------------------------------------------------
    def reorder(self, X):
        """Sort every cloud by (group rank, Morton code of 2 A cells); build the padded static tables."""
        S = X.shape[0]
        N, B, nblk, G = self.N, self.B, self.nblk, self.G
        dev = self.dev
        lo = X.amin(1, keepdim=True)
        c = torch.floor((X - lo) / 2.0).long().clamp_(0, 1023)
        key = (self.gid.long()[None, :] << 31) | _morton(
            c
        )  # invalid group (31) sorts last
        perm = torch.argsort(key, dim=1, stable=True)  # [S,N]
        pad = torch.arange(self.Npad, device=dev)
        src = torch.where(
            pad < N, pad, (pad // B) * B
        )  # pads repeat the block's first point (boxes exact)
        perm_pad = perm[:, src]  # [S,Npad] original index
        self.perm = perm_pad.to(torch.int32).contiguous()
        self._gather_idx = perm_pad[:, :, None].expand(-1, -1, 3)
        is_pad = (pad >= N)[None, :].expand(S, -1)
        g = self.gid[perm_pad]
        m = self.pmask[perm_pad]
        self.g_pad = torch.where(
            is_pad, torch.full_like(g, 31), g
        ).contiguous()  # pads never interact
        self.m_pad = torch.where(is_pad, torch.zeros_like(m), m).contiguous()
        self.r_pad = self.r[perm_pad].contiguous()
        ar = torch.arange(G, device=dev)
        onehot = (self.g_pad.view(S, nblk, B, 1) == ar).any(
            2
        )  # [S,nblk,G] groups present in the block
        self.bgbits = (onehot.long() << ar).sum(-1).to(torch.int32).contiguous()
        mbits = (
            ((self.m_pad.view(S, nblk, B, 1) >> ar) & 1).bool().any(2)
        )  # partner groups wanted by the block
        self.bmask = (mbits.long() << ar).sum(-1).to(torch.int32).contiguous()
        if self._S != S:
            self.cnt = torch.empty(S * nblk, dtype=torch.int32, device=dev)
            self.cols = torch.empty(S * nblk * nblk, dtype=torch.int32, device=dev)
            self._S = S

    def _boxes(self, Xp):
        lo, hi = torch.aminmax(Xp.view(Xp.shape[0], self.nblk, self.B, 3), dim=2)
        return lo.contiguous(), hi.contiguous()

    def _tiles(self, lo, hi, cut2):
        S = lo.shape[0]
        _tiles_kernel[(S * self.nblk,)](
            lo,
            hi,
            self.bgbits,
            self.bmask,
            self.cnt,
            self.cols,
            self.nblk,
            float(cut2),
            BLOCKJ=self.blockj,
            num_warps=4,
        )

    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def energy_and_grad(self, X):
        X = X.contiguous().float()
        S, N, _ = X.shape
        assert N == self.N
        if (
            self.perm is None
            or self._S != S
            or (self.reorder_every and self.calls % self.reorder_every == 0)
        ):
            self.reorder(X)
        self.calls += 1
        Xp = torch.gather(X, 1, self._gather_idx).contiguous()  # [S,Npad,3]
        lo, hi = self._boxes(Xp)
        self._tiles(lo, hi, self.cut2)
        e_part = torch.empty(S * self.nblk, dtype=torch.float64, device=self.dev)
        G = torch.empty(S, N, 3, dtype=torch.float32, device=self.dev)
        _pair_kernel[(S * self.nblk,)](
            Xp,
            self.r_pad,
            self.g_pad,
            self.m_pad,
            self.perm,
            self.cnt,
            self.cols,
            e_part,
            G,
            N,
            self.Npad,
            self.nblk,
            self.scale,
            B=self.B,
            num_warps=self.warps,
            enable_fp_fusion=False,
        )
        E = e_part.view(S, self.nblk).sum(1).float()
        return E, G
