"""context: the attention context, sum over slots of P_s * V_s in f32 . The products (one VXM Multiply
per slot) are accumulated in four interleaved chains over the slots s mod 4, each addition taking its two operands chained
from the cycle's results; the four chains are combined as (c0 + c1) + (c2 + c3) and the sum is narrowed to f16 for the output
projection. The summation order is the specification of the result (`ref.context`), so the four chains stay four on every
profile.

The timing comes from the profile: a chained add follows the add it chains from by `recurrence(p)` = d_func + 1 - d_skew of
`add` cycles (4 on `ref`), so a chain takes an element every `recurrence` cycles and the four chains together four slots per
`recurrence` cycles: slot s is multiplied at t0 + `slot_cycle(p, s)` (= s on `ref`, where the slots go one per cycle). On a
profile with a longer add the V rows are read in groups of four slots, `recurrence` cycles apart (an AGT loop with a countdown).
A profile whose recurrence is below 4 cannot keep four chains in step and is refused."""
from opentsp_tools.stations import ScheduleError

ADDERS = 8           # ALUs 0..7 multiply, ALUs 8..15 add
BLOCK = 16           # slots of a pool block
CHAINS = 4           # interleaved chains: the summation order of `ref.context`


def recurrence(p):
    """Cycles from the dispatch of a chained add to the dispatch of the add that chains from its result."""
    d_func, d_skew = p.timing("add.f32")
    r = d_func + 1 - d_skew
    if r < CHAINS:
        raise ScheduleError(f"context: a chained add follows its predecessor after {r} cycles on profile {p.name} "
                            f"(d_func {d_func}, d_skew {d_skew}); the four interleaved chains of the summation order need at least {CHAINS}")
    return r


def slot_cycle(p, s):
    """The cycle, from the start of the kernel, at which slot s is multiplied: the four chains advance one slot each per
    recurrence, in consecutive cycles."""
    return s % CHAINS + (s // CHAINS) * recurrence(p)


def v_loop(p):
    """The AGT records of one block's V loop (16 references): one word and one cycle apart where the recurrence is
    `CHAINS`, else groups of `CHAINS` words one cycle apart, the groups `recurrence` cycles apart."""
    r = recurrence(p)
    if r == CHAINS:
        return [("step", 1), ("iterations", BLOCK)]
    return [("step", 1), ("iterations", CHAINS), ("countdown", r - CHAINS), ("step", 1), ("iterations", BLOCK // CHAINS)]


def reserve_v_streams(f, row="E"):
    """Take the first group of 8 streams of `row` out of the Flow's rotation for the V loops of the whole program (one
    group serves every layer and lane: their loops are never live together) and return its first stream. Call it before
    the Flow allocates anything on `row`."""
    if f.streams[row] > f.low[row]:
        raise ScheduleError(f"read_v_blocks: row {row} has streams allocated up to {f.streams[row]}; call it before the "
                            "Flow allocates on that row (its loops take the first group of 8 out of the rotation)")
    stream = f.low[row]
    f.low[row] = stream + 8             # a multiple of 8 keeps the number of rotating groups even, as context needs
    return stream


def read_v_blocks(f, t0, table, addr_of, slices, vrange, row="E", prefix="vb", stream=None):
    """The V rows of the blocks `table` (placeholders: block ids in block-table order) as the Values the Multiplies of
    `context` take, one per slot: slot 16 j + o is the word `addr_of(block j) + o` of the 4 `slices` of the layer's V
    quad (one byte plane each), met by the Multiply at cycle t0 + slot. Block j is one Read per slice, patch site
    `<prefix>.b<j>.s<slice>` with the range `vrange` = (lo, hi, stride, span) of block addresses (all in one bank), and an AGT loop
    of 16 references. The loops' streams (the first group of the 8 streams it reserves on `row`, for all blocks) are taken out of the
    Flow's rotation, so call this before the Flow allocates other streams on `row`; a program with several calls reserves
    once (`reserve_v_streams`) and passes the stream."""
    b = f.b
    if stream is None:
        stream = reserve_v_streams(f, row)
    out = []
    for j, blk in enumerate(table):
        c = b.read_cycle_for(t0 + slot_cycle(f.p, BLOCK * j), "mul.f32", 88, slices[0], row)
        out += b.read_loop(c, slices, addr_of(blk), row, stream, v_loop(f.p),
                           patch=[(f"{prefix}.b{j}.s{s}", *vrange) for s in slices])
    return out


def context(f, ps, vs, zero, t0, quads, to="E"):
    """P_s and V_s are the Vecs `ps` and `vs` (f32, one word per slot; ps and vs on slices that are read in the same
    cycle, so on different quads; the V_s may be Values from read_v_blocks); `zero` is an f32 zero Vec the first addition of every chain takes. Slot s multiplies
    at cycle t0 + `slot_cycle(p, s)` (t0 + s on `ref`). The four chain totals are kept on the east quads `quads`. Returns (the f32 sum, its f16
    narrowing), Values on row `to`."""
    acc = []
    for s, (pv, vv) in enumerate(zip(ps, vs)):
        prod = f.vxm(s % ADDERS, "mul.f32", [pv, vv], at=t0 + slot_cycle(f.p, s))
        if s < 4:
            acc.append(f.vxm(ADDERS + s % ADDERS, "add.f32", [prod, zero], chain=(0,)))
        else:
            acc[s % 4] = f.vxm(ADDERS + s % ADDERS, "add.f32", [prod, acc[s % 4]], chain=(0, 1))
    kept = [f.store(a, q) for a, q in zip(acc, quads)]
    both = max(f.bound(v, "add.f32", 88) for v in kept)      # x and y commit in one cycle, for z to chain both
    x = f.vxm(0, "add.f32", [kept[0], kept[1]], at=both)
    y = f.vxm(1, "add.f32", [kept[2], kept[3]], at=both)
    z = f.vxm(2, "add.f32", [x, y], chain=(0, 1), to=to)
    return z, f.vxm(3, "conv_f32_f16.f32", [z], chain=(0,), to=to, width=2)
