"""Triton kernel for the candidate-pose search: clash partials of many rigid placements against a fixed cloud.

For every (candidate pose, block of moving points) one program places the points in registers (x = R a + t), visits the
27 grid cells around each point and writes five partial results: the sum of squared soft overlaps, the sum of soft
overlaps, the number of pairs closer than the soft cutoff plus DELTA, the minimum of d - 0.75 (ra + rb) and the minimum
of d - 0.85 (ra + rb). rigid_core.search reduces the partials, turns them into energy bounds with an explicit
floating-point error window, and re-evaluates only the few candidates that can still win with the dense arithmetic.
Per-pair arithmetic follows torch.cdist (squares rounded separately and added as (dx^2 + dz^2) + dy^2, IEEE sqrt); the
kernel is launched with fused multiply-add disabled.
"""

import triton
import triton.language as tl


# ----------------------------------------------------------------------------------------------------------------------
# Triton kernel: 5 clash partials per (candidate, block of moving points)
# ----------------------------------------------------------------------------------------------------------------------
@triton.jit
def search_kernel(
    A_ptr,
    ra_ptr,
    R_ptr,
    t_ptr,
    Bs_ptr,
    rbs_ptr,
    cstart_ptr,
    ccnt_ptr,
    lo_ptr,
    dims_ptr,
    coff_ptr,
    out_ptr,
    M,
    NC,
    nblk,
    inv_h,
    DELTA,
    SOFT_C: tl.constexpr,
    HARD_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(
        0
    )  # global candidate id = s * NC + c  (c = NC-1 is the entry pose)
    blk = tl.program_id(1)
    s = g // NC
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < M
    ab = s.to(tl.int64) * M + offs
    a0 = tl.load(A_ptr + ab * 3 + 0, mask=m, other=0.0)
    a1 = tl.load(A_ptr + ab * 3 + 1, mask=m, other=0.0)
    a2 = tl.load(A_ptr + ab * 3 + 2, mask=m, other=0.0)
    ra = tl.load(ra_ptr + ab, mask=m, other=0.0)
    g64 = g.to(tl.int64)
    rp = R_ptr + g64 * 9
    r00 = tl.load(rp + 0)
    r01 = tl.load(rp + 1)
    r02 = tl.load(rp + 2)
    r10 = tl.load(rp + 3)
    r11 = tl.load(rp + 4)
    r12 = tl.load(rp + 5)
    r20 = tl.load(rp + 6)
    r21 = tl.load(rp + 7)
    r22 = tl.load(rp + 8)
    t0 = tl.load(t_ptr + g64 * 3 + 0)
    t1 = tl.load(t_ptr + g64 * 3 + 1)
    t2 = tl.load(t_ptr + g64 * 3 + 2)
    x = r00 * a0 + r01 * a1 + r02 * a2 + t0
    y = r10 * a0 + r11 * a1 + r12 * a2 + t1
    z = r20 * a0 + r21 * a1 + r22 * a2 + t2
    x = tl.where(m, x, -1.0e9)
    y = tl.where(m, y, -1.0e9)
    z = tl.where(m, z, -1.0e9)  # masked lanes: no neighbours
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
    acc_sq = tl.zeros([BLOCK], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK], dtype=tl.float32)
    acc_n = tl.zeros([BLOCK], dtype=tl.int32)
    mhard = tl.full([BLOCK], 1.0e30, dtype=tl.float32)
    msoft = tl.full([BLOCK], 1.0e30, dtype=tl.float32)
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
            ddx = x - bx
            ddy = y - by
            ddz = z - bz
            acc = (
                (ddx * ddx + ddz * ddz) + ddy * ddy
            )  # torch.cdist's block reduction adds the squares in the order x, z, y (no fma)
            d = tl.sqrt_rn(acc)  # IEEE sqrt, same operation order as torch.cdist
            rs = ra + rb
            sm = d - SOFT_C * rs
            o = tl.where(ok, tl.maximum(-sm, 0.0), 0.0)
            acc_sq += o * o
            acc_o += o
            acc_n += (ok & (sm < DELTA)).to(tl.int32)
            mhard = tl.minimum(mhard, tl.where(ok, d - HARD_C * rs, 1.0e30))
            msoft = tl.minimum(msoft, tl.where(ok, sm, 1.0e30))
    ob = (g64 * nblk + blk) * 5
    tl.store(out_ptr + ob + 0, tl.sum(acc_sq, axis=0))
    tl.store(out_ptr + ob + 1, tl.sum(acc_o, axis=0))
    tl.store(out_ptr + ob + 2, tl.sum(acc_n, axis=0).to(tl.float32))
    tl.store(out_ptr + ob + 3, tl.min(mhard, axis=0))
    tl.store(out_ptr + ob + 4, tl.min(msoft, axis=0))
