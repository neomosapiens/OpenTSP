"""expand: per-head probabilities to per-lane vectors . The head probabilities (f32, one vector per
head of `dk` lanes) are narrowed to f16 on eight VXM ALUs per LW cycle and installed straight from the VXM into west
plane 0 as the matrix W' whose row l is the probability vector of head l div dk (8 LW cycles for 64 rows); then one ABC
per slot with the f16 one-hot vector e_s (1.0 in lane s) and one ACC give P_s, whose lane l is p[l div dk][s]: the
product is p * 1.0 with every other product zero, exact."""
import numpy as np

from opentsp_tools.programs import geometry
from opentsp_tools.programs.kernels import install as inst

PLANE = 0
ABC_SPACING = 2      # generator schedule spacing; shipped profiles permit initiation every cycle


def one_hot(slots, vlanes):
    """slots f16 vectors of vlanes lanes: e_s has 1.0 in lane s."""
    e = np.zeros((slots, vlanes), np.float16)
    e[np.arange(slots), np.arange(slots)] = 1.0
    return list(e)


def earliest(f, heads, dk, per_lw=8):
    """The first t0 for `expand` at which every LW finds the head vectors it converts readable: LW i takes the heads
    (8 i .. 8 i + 7) div dk, and its Read is at t0 + i - d_conv - transit + lw skew."""
    lw = inst.plane_queues(PLANE)[2]
    d_conv, transit = f.p.timing("conv_f32_f16.f32")[0], f.transit(lw, inst.plane_queues(PLANE)[1], 0)
    first = {}
    for i in range(len(heads) * dk // per_lw):
        for r in range(per_lw):
            first.setdefault((per_lw * i + r) // dk, i)
    return max(f.bound(heads[h], "conv_f32_f16.f32", 88) - i + d_conv + transit - f.p.timing("lw")[1] for h, i in first.items())


def expand(f, heads, one_hots, dk, t0, out_quad, hemi="W"):
    """Install W' from the f32 Vecs `heads` (4 planes each) with LW from cycle t0, then expand with the
    f16 Vecs `one_hots` (west). The P_s are written to quad `out_quad` of `hemi` (bank 1) as they are produced; a tuple of quads
    takes them in turn (P_s on quad s mod len), which spaces the reads of one slice, so that a paged program can fetch text
    between them (C = 128 reads one P_s per cycle for 128 cycles). Returns the list of Vecs P_s."""
    b, p = f.b, f.p
    base, row, lw = inst.plane_queues(PLANE)
    abc, acc_q = lw + 1, lw + 2
    per_lw = 8
    lws = len(heads) * dk // per_lw             # one LW cycle per 8 rows of W': the rows are the lanes, not the slots
    geometry.check_install(p, lws, f"the expansion matrix of {len(heads)} heads of {dk} lanes")
    d_conv, transit = p.timing("conv_f32_f16.f32")[0], f.transit(lw, row, 0)
    for i in range(lws):
        t_vxm = t0 + i - d_conv - transit + p.timing("lw")[1]
        heads_i = {h: f.read(heads[h], t_vxm, "conv_f32_f16.f32", 88) for h in {(per_lw * i + r) // dk for r in range(per_lw)}}
        pairs = [f.vxm(r, "conv_f32_f16.f32", [heads_i[(per_lw * i + r) // dk]], to=row, width=2, stream=2 * r)
                 for r in range(per_lw)]
        b.unit(lw, "lw", f"lw.f16 {row}:0", pairs)
    t_iw = t0 + lws - 1 + p.timing("lw")[0]
    b.unit(lw, "iw", "iw.f16", cycle=t_iw)
    t_abc = t_iw + p.timing("iw")[0]
    out = []
    for s, e in enumerate(one_hots):
        v = f.read(e, t_abc + ABC_SPACING * s, "abc", abc, row)
        t, _ = b.unit(abc, "abc", f"abc.f16 {row}:{v.stream}", [v])
        st = f.stream("E", 4)
        _, acc = b.unit(acc_q, "acc", f"acc.f32 E:{st}, replace, 0", cycle=t + p.timing("abc")[0], dst=("E", st, 4))
        quad = out_quad[s % len(out_quad)] if isinstance(out_quad, (tuple, list)) else out_quad
        out.append(f.store(acc, quad, hemi=hemi))
    return out

