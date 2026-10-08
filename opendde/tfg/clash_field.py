"""Cell-list clash kernel for rigid-body guidance.

For K candidate placements of a moving cloud against a fixed cloud, the kernel computes the soft-overlap energy
sum relu(0.85 (ra + rb) - d)^2 (accumulated in float64, rounded to fp32 once), the deepest severe overlap
max relu(0.75 (ra + rb) - d), the gradient in the dense form, and, given a reference pose, whether a pair is severe
at a placement that was not severe at the reference pose.

Only pairs closer than the overlap cutoff contribute, so the fixed cloud is bucketed into a uniform grid and every
moving point visits the 27 cells around it. Per-pair arithmetic reproduces torch.cdist(compute_mode=
"donot_use_mm_for_euclid_dist") bit for bit: squares are rounded separately and added as (dx^2 + dz^2) + dy^2, the
square root is IEEE, and fused multiply-add is disabled. cdist_order_matches() checks this for the installed torch.
The accumulation order is fixed and uses no atomics, so equal inputs give equal outputs.
"""

import torch
import triton
import triton.language as tl

SOFT = 0.85
HARD = 0.75
MAX_CELLS = 2**24


def skew_matrix(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack(
        [
            torch.stack([z, -w[..., 2], w[..., 1]], -1),
            torch.stack([w[..., 2], z, -w[..., 0]], -1),
            torch.stack([-w[..., 1], w[..., 0], z], -1),
        ],
        -2,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Triton kernel: energy / depth / (gradient | new-severe-pair test) for K placements per instance
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def _field_kernel(
    Xs_ptr,
    Xo_ptr,
    ras_ptr,
    perm_ptr,
    Bs_ptr,
    rbs_ptr,
    cstart_ptr,
    ccnt_ptr,
    lo_ptr,
    dims_ptr,
    coff_ptr,
    e_ptr,
    dep_ptr,
    new_ptr,
    g_ptr,
    M,
    K,
    nblk,
    inv_h,
    SOFT_C: tl.constexpr,
    HARD_C: tl.constexpr,
    WITH_GRAD: tl.constexpr,
    CHECK_NEW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    item = tl.program_id(0)  # = s * K + k
    blk = tl.program_id(1)
    s = item // K
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < M
    xb = item.to(tl.int64) * M + offs
    x = tl.load(Xs_ptr + xb * 3 + 0, mask=m, other=-1.0e9)
    y = tl.load(Xs_ptr + xb * 3 + 1, mask=m, other=-1.0e9)
    z = tl.load(Xs_ptr + xb * 3 + 2, mask=m, other=-1.0e9)
    sb = s.to(tl.int64) * M + offs
    ra = tl.load(ras_ptr + sb, mask=m, other=0.0)
    if CHECK_NEW:
        xo = tl.load(Xo_ptr + sb * 3 + 0, mask=m, other=-1.0e9)
        yo = tl.load(Xo_ptr + sb * 3 + 1, mask=m, other=-1.0e9)
        zo = tl.load(Xo_ptr + sb * 3 + 2, mask=m, other=-1.0e9)
    ox = tl.load(lo_ptr + s * 3 + 0)
    oy = tl.load(lo_ptr + s * 3 + 1)
    oz = tl.load(lo_ptr + s * 3 + 2)
    dimx = tl.load(dims_ptr + s * 3 + 0)
    dimy = tl.load(dims_ptr + s * 3 + 1)
    dimz = tl.load(dims_ptr + s * 3 + 2)
    coff = tl.load(coff_ptr + s)
    fx = tl.minimum(tl.maximum((x - ox) * inv_h, -4.0), dimx + 4.0)
    fy = tl.minimum(tl.maximum((y - oy) * inv_h, -4.0), dimy + 4.0)
    fz = tl.minimum(tl.maximum((z - oz) * inv_h, -4.0), dimz + 4.0)
    cx = tl.floor(fx).to(tl.int32)
    cy = tl.floor(fy).to(tl.int32)
    cz = tl.floor(fz).to(tl.int32)
    esq = tl.zeros([BLOCK], dtype=tl.float64)
    dep = tl.zeros([BLOCK], dtype=tl.float32)
    anynew = tl.zeros([BLOCK], dtype=tl.int32)
    sc = tl.zeros([BLOCK], dtype=tl.float32)
    sbx = tl.zeros([BLOCK], dtype=tl.float32)
    sby = tl.zeros([BLOCK], dtype=tl.float32)
    sbz = tl.zeros([BLOCK], dtype=tl.float32)
    for k in range(0, 27):
        nx = cx + (k // 9 - 1)
        ny = cy + ((k // 3) % 3 - 1)
        nz = cz + (k % 3 - 1)
        valid = (
            (nx >= 0) & (nx < dimx) & (ny >= 0) & (ny < dimy) & (nz >= 0) & (nz < dimz)
        )
        cid = coff + (nx.to(tl.int64) * dimy + ny) * dimz + nz
        start = tl.load(cstart_ptr + cid, mask=valid, other=0)
        cnt = tl.load(ccnt_ptr + cid, mask=valid, other=0)
        cmax = tl.max(cnt, axis=0)
        for jj in range(0, cmax):
            ok = jj < cnt
            idx = start + jj
            bx = tl.load(Bs_ptr + idx * 3 + 0, mask=ok, other=0.0)
            by = tl.load(Bs_ptr + idx * 3 + 1, mask=ok, other=0.0)
            bz = tl.load(Bs_ptr + idx * 3 + 2, mask=ok, other=0.0)
            rb = tl.load(rbs_ptr + idx, mask=ok, other=0.0)
            # per-pair arithmetic identical to torch.cdist (squares rounded separately, summed as (dx^2 + dz^2) + dy^2, IEEE sqrt)
            # and to the dense formulas (the kernel is launched with enable_fp_fusion=False so that a*b - c is not fused)
            ddx = x - bx
            ddy = y - by
            ddz = z - bz
            acc = (
                (ddx * ddx + ddz * ddz) + ddy * ddy
            )  # torch.cdist's block reduction adds the squares in the order x, z, y (no fma)
            d = tl.maximum(tl.sqrt_rn(acc), 1.0e-6)
            rs = ra + rb
            o = tl.where(ok, tl.maximum(SOFT_C * rs - d, 0.0), 0.0)
            esq += (o * o).to(tl.float64)
            dep = tl.maximum(dep, tl.where(ok, tl.maximum(HARD_C * rs - d, 0.0), 0.0))
            if CHECK_NEW:
                sev = ok & (d < HARD_C * rs)
                dox = xo - bx
                doy = yo - by
                doz = zo - bz
                acco = (dox * dox + doz * doz) + doy * doy
                do = tl.maximum(tl.sqrt_rn(acco), 1.0e-6)
                anynew += (sev & (do >= HARD_C * rs)).to(tl.int32)
            if WITH_GRAD:
                coeff = tl.where(
                    d > 1.0e-6, tl.fdiv(-20.0 * o, d, ieee_rounding=True), 0.0
                )
                # gradient in the dense form: (sum_j coeff) * x - sum_j coeff * b_j, fp32 with fma accumulation
                sc += coeff
                sbx = tl.fma(coeff, bx, sbx)
                sby = tl.fma(coeff, by, sby)
                sbz = tl.fma(coeff, bz, sbz)
    tl.store(e_ptr + item * nblk + blk, tl.sum(esq, axis=0))
    tl.store(dep_ptr + item * nblk + blk, tl.max(dep, axis=0))
    tl.store(new_ptr + item * nblk + blk, tl.sum(anynew, axis=0))
    if WITH_GRAD:
        iorig = tl.load(perm_ptr + sb, mask=m, other=0)
        gb = (item.to(tl.int64) * M + iorig) * 3
        tl.store(g_ptr + gb + 0, sc * x - sbx, mask=m)
        tl.store(g_ptr + gb + 1, sc * y - sby, mask=m)
        tl.store(g_ptr + gb + 2, sc * z - sbz, mask=m)


# ----------------------------------------------------------------------------------------------------------------------
# spatial index
# ----------------------------------------------------------------------------------------------------------------------
def build_cells(B, rb, h):
    S, N, _ = B.shape
    while True:
        lo = B.min(1).values - 0.5 * h
        hi = B.max(1).values + 0.5 * h
        dims = torch.ceil((hi - lo) / h).long() + 1
        ncell_s = dims.prod(1)
        if int(ncell_s.max()) <= MAX_CELLS:
            break
        h *= 2.0
    c = torch.floor((B - lo[:, None]) / h).long()
    c = torch.minimum(torch.clamp_min(c, 0), dims[:, None] - 1)
    coff = torch.cumsum(ncell_s, 0) - ncell_s
    cid = (
        (c[..., 0] * dims[:, None, 1] + c[..., 1]) * dims[:, None, 2]
        + c[..., 2]
        + coff[:, None]
    )
    flat = cid.flatten()
    order = torch.argsort(flat, stable=True)
    ntot = int(ncell_s.sum())
    cnt = torch.bincount(flat, minlength=ntot).to(torch.int32)
    start = torch.cumsum(cnt, 0, dtype=torch.int32) - cnt
    return dict(
        Bs=B.reshape(-1, 3)[order].contiguous(),
        rbs=rb.repeat(S)[order].contiguous(),
        start=start.contiguous(),
        cnt=cnt.contiguous(),
        lo=lo.contiguous().float(),
        dims=dims.to(torch.int32).contiguous(),
        coff=coff.to(torch.int64).contiguous(),
        h=h,
    )


def spatial_order(A, h):
    lo = A.min(1).values
    c = torch.floor((A - lo[:, None]) / h).long()
    dm = c.max(1).values + 1
    key = (c[..., 0] * dm[:, None, 1] + c[..., 1]) * dm[:, None, 2] + c[..., 2]
    return torch.argsort(key, dim=1, stable=True)


class ClashField:
    """Cell grid of the fixed cloud B (built once) and the moving cloud's point order (built once from X0)."""

    def __init__(self, B, rb, ra, X0, block=32, warps=1):
        self.block, self.warps = block, warps
        self.S, self.N, _ = B.shape
        self.M = ra.shape[0]
        self.dev = B.device
        cutoff = SOFT * float(ra.max() + rb.max())
        self.cells = build_cells(B, rb, 1.02 * cutoff + 1e-3)
        self.h = self.cells["h"]
        self.perm = spatial_order(X0, self.h)  # [S,M] long
        self.perm32 = self.perm.to(torch.int32).contiguous()
        self.ras = ra[self.perm].contiguous()  # [S,M]
        self.nblk = triton.cdiv(self.M, block)
        self._pexp = self.perm[:, :, None].expand(-1, -1, 3)

    def gather(self, X):
        """X [S,M,3] or [S,K,M,3] -> same, points in cell order."""
        if X.dim() == 3:
            return torch.gather(X, 1, self._pexp).contiguous()
        K = X.shape[1]
        return torch.gather(
            X, 2, self._pexp[:, None].expand(-1, K, -1, -1)
        ).contiguous()

    def run(self, X, X_old=None, with_grad=False):
        """X [S,K,M,3] (original point order).  Returns clash [S,K] fp32 (sum overlap^2), depth [S,K] fp32,
        new_severe [S,K] bool (only if X_old [S,M,3] is given: a pair severe at X but not at X_old exists),
        grad [S,K,M,3] fp32 (only with_grad)."""
        S, K, M = X.shape[0], X.shape[1], self.M
        Xs = self.gather(X)
        Xo = self.gather(X_old) if X_old is not None else Xs
        items = S * K
        c = self.cells
        dev = self.dev
        e_part = torch.empty(items, self.nblk, dtype=torch.float64, device=dev)
        d_part = torch.empty(items, self.nblk, dtype=torch.float32, device=dev)
        n_part = torch.empty(items, self.nblk, dtype=torch.int32, device=dev)
        grad = (
            torch.empty(items, M, 3, dtype=torch.float32, device=dev)
            if with_grad
            else e_part
        )
        _field_kernel[(items, self.nblk)](
            Xs,
            Xo,
            self.ras,
            self.perm32,
            c["Bs"],
            c["rbs"],
            c["start"],
            c["cnt"],
            c["lo"],
            c["dims"],
            c["coff"],
            e_part,
            d_part,
            n_part,
            grad,
            M,
            K,
            self.nblk,
            1.0 / self.h,
            SOFT_C=SOFT,
            HARD_C=HARD,
            WITH_GRAD=with_grad,
            CHECK_NEW=X_old is not None,
            BLOCK=self.block,
            num_warps=self.warps,
            enable_fp_fusion=False,
        )
        clash = e_part.sum(1).float().view(S, K)
        depth = d_part.amax(1).view(S, K)
        new = (n_part.sum(1) > 0).view(S, K) if X_old is not None else None
        return clash, depth, new, (grad.view(S, K, M, 3) if with_grad else None)


def cdist_order_matches(device="cuda", n=2000):
    """True iff torch.cdist(donot_use_mm) on this torch/CUDA build equals sqrt((dx^2 + dz^2) + dy^2) with separately
    rounded fp32 squares, the order the kernels reproduce. When it is False the kernels' per-pair distances can differ
    from cdist by one fp32 ulp, and the callers use the dense path instead."""
    g = torch.Generator(device="cpu").manual_seed(0)
    X = (torch.rand(n, 3, generator=g) * 50).to(device)
    B = (torch.rand(n, 3, generator=g) * 50).to(device)
    d_ref = torch.cdist(X, B, compute_mode="donot_use_mm_for_euclid_dist")
    diff = X[:, None, :] - B[None, :, :]
    sq = diff * diff
    d = torch.sqrt((sq[..., 0] + sq[..., 2]) + sq[..., 1])
    return bool(torch.equal(d, d_ref))
