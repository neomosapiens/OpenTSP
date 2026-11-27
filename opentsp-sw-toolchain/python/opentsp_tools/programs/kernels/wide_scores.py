"""wide_scores: the scores of the heads of one 256-lane tile (GPT-2 small: four heads of 64 lanes per tile) against an installed K tile
(`scores.scores` for a width above one vector). The ops are those of `scores.scores`; the result of each head's scale and mask
selection is given a stream of its own (`RESULT_STREAMS`, below the Flow's rotation, which `f.low` reserves) so that the four heads of a
tile, whose ALU chains commit in the same cycles, never share one."""
from opentsp_tools.programs.kernels import install as inst
from opentsp_tools.programs.kernels.head_mask import head_mask
from opentsp_tools.programs.kernels.scores import ABC_SPACING, PLANE

RESULT_STREAMS = 16        # the streams 0..15 of row E: four f32 results of four planes each (the installs use them between tiles)


def tile_scores(f, slots, q, masks, scale, add_mask, t0=30, addr=inst.ADDR_W, text=None):
    """Install the tile's `slots`-row K matrix (plane PLANE, word `addr`; `text(j, i)` names its words), then for every head of
    the tile (the f16 Vecs `masks`): ABC and ACC of the query masked to the head, scale then select -inf for masked slots on two chained ALUs.
    `q` is the tile's f32 query Vec; returns one f32 Value per head (lane s = score of slot s). Call with `f.low["E"]` >=
    RESULT_STREAMS."""
    b, p = f.b, f.p
    assert f.low["E"] >= RESULT_STREAMS and len(masks) * 4 <= RESULT_STREAMS
    t_abc = inst.install(f.k, "f16", slots // 8, True, addr, t0, PLANE, "k", text)
    lw = inst.plane_queues(PLANE)[2]
    abc, acc_q = lw + 1, lw + 2
    q16 = f.store(f.vxm(0, "conv_f32_f16.f32", [q], to="W", width=2), 3)
    d_mask = p.timing("mask.f32")[0]
    transit = f.transit(abc, "E", 0)
    out, t_mask = [], t_abc - d_mask - transit + p.timing("abc")[1]
    for h, mask in enumerate(masks):
        m = head_mask(f, q16, mask, 1 + h % 4, at=t_mask)
        t_mask = m.cycle - d_mask + ABC_SPACING
        t, _ = b.unit(abc, "abc", f"abc.f16 E:{m.stream}", [m])
        s = f.stream("W", 4)
        _, acc = b.unit(acc_q, "acc", f"acc.f32 W:{s}, replace, 0", cycle=t + p.timing("abc")[0], dst=("W", s, 4))
        alu = 5 + 2 * (h % 5)
        scaled = f.vxm(alu, "mul.f32", [acc, scale])
        # add_mask is 0 for visible slots and -inf otherwise. Selection also
        # excludes NaN/Inf from stale masked K rows; addition would propagate it.
        out.append(f.vxm(alu + 1, "select.f32", [add_mask, scaled, add_mask], chain=(1,), stream=4 * h))
    return out
