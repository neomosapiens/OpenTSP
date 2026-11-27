"""Category B of the design's sec. 2.2: MEM addressing and the AGT (the families
multi_mem_step, countdown_iterations, mem_compound_*, compound_step_write_46_roundtrip,
mem_readwrite and parallel_gather_scatter families). Fused operations become one reference plus AGT configuration
records on the following cycles; same-slice round trips move to a downstream slice; ReadWrite becomes a Read +
Write parcel in different banks (microarchitecture.md sec. 4). Loop timing follows opentsp_tools.programs.kernels.agt_refs."""
import random

from opentsp_tools.programs.case import Kit
from basic import PROFILES, rand_word
from opentsp_tools import emit
from opentsp_tools import stations


def words(rng, p, n):
    return [rand_word(rng, p) for _ in range(n)]


def block(k, name, slice_, addr, ws):
    k.b.prog.extern(name, slice_, addr, len(ws))
    k.weights[name] = ("i8", b"".join(ws))


def gather_partial(p, seed=1):
    """Per-tile effects of Gather and Scatter (microarchitecture.md sec. 4 and 8). A tile whose map
    address is outside the slice acts alone: a Gather tile drives zero on its lane, a Scatter tile writes nothing,
    and the instruction reports one OTS-E-OPR-001. Odd tiles carry a bad address (every tile on a one-tile profile).
    MEM[0] gathers lane 16g + k of a word of its table, on the map from MEM[45] (row W): k = 0 and k = 5 with the
    mixed map (one diagnostic each), then k = 0 with an all-good map; MEM[44] writes the three results. MEM[44]
    then scatters byte 5 of a vector (MEM[1]) on the mixed map (MEM[2]) into its own words. Only quad 0 of each
    hemisphere is used, so the program also runs on a profile with one quad per hemisphere."""
    k = Kit(p, "mem/gather_partial", "Gather and Scatter with bad addresses on some tiles", seed, module="mem")
    rng = random.Random(seed)
    b = k.b
    d_mem, d_skew = p.timing("gather")
    n_table, base_t, base_o = 8, 0x40, 0x40
    table, old = words(rng, p, n_table), words(rng, p, n_table)
    block(k, "tbl", 0, base_t, table)
    block(k, "old", 44, base_o, old)

    def tile_ok(g):
        return g % 2 == 0 and p.superlanes > 1

    def map_word(base, all_good):
        w, addrs = bytearray(rand_word(rng, p)), []
        for g in range(p.superlanes):
            a = base + rng.randrange(n_table) if all_good or tile_ok(g) else p.words + 4 + g
            addrs.append(a)
            w[16 * g], w[16 * g + 1] = a & 0xFF, a >> 8
        return bytes(w), addrs

    mixed, a_mixed = map_word(base_t, False)
    good, a_good = map_word(base_t, True)
    k.extern_word("mixed", 45, 0x10, mixed)
    k.extern_word("good", 45, 0x11, good)
    exact_ref = stations.need_gap(p, 0, "E", "write")
    t_g = 12
    for i, (kk, name, addrs) in enumerate(((0, "mixed", a_mixed), (5, "mixed", a_mixed), (0, "good", a_good))):
        t = t_g + 4 * i
        vm = b.read_for(t, "gather", 0, [45], [0x10 + (name == "good")], "W", 0, addr_text=[name])
        b._read(vm, 0, "gather", t)
        b.raw(t, 0, f"gather E:{4 + 2 * i}, W:0, {kk:#x}, 0x0", slot="R")
        res = b._write("E", 4 + 2 * i, 1, exact_ref, t + d_mem)
        k.last = max(k.last, b.write(res, [44], [0x30 + i]))
        out = bytearray(p.vlanes)
        for g, a in enumerate(addrs):
            if a < p.words:
                out[16 * g + kk] = table[a - base_t][16 * g + kk]
        k.expect(44, 0x30 + i, bytes(out))
        if name == "mixed":
            k.diags.append((t + d_skew, 0, "OTS-E-OPR-001"))
    v = rand_word(rng, p)
    m2, a_m2 = map_word(base_o, False)
    k.extern_word("v", 1, 0x10, v)
    k.extern_word("m2", 2, 0x10, m2)
    t_s = t_g + 16
    vv = b.read_for(t_s, "scatter", 44, [1], [0x10], "E", 8, addr_text=["v"])
    vm2 = b.read_for(t_s, "scatter", 44, [2], [0x10], "E", 9, addr_text=["m2"])
    b._read(vv, 44, "scatter", t_s)
    b._read(vm2, 44, "scatter", t_s)
    b.raw(t_s, 44, "scatter E:8, E:9, 0x5, 0x0", slot="W")
    new = [bytearray(w) for w in old]
    for g, a in enumerate(a_m2):
        if a < p.words:
            new[a - base_o][16 * g + 5] = v[16 * g + 5]
    for i, w in enumerate(new):
        k.expect(44, base_o + i, bytes(w))
    k.diags.append((t_s + d_skew, 44, "OTS-E-OPR-001"))
    k.last = max(k.last, t_s)
    return k.done("gather_partial")


def agt_step(p, seed=1):
    """(a) MEM[4] reads xs[0], xs[2], xs[4], xs[6] (Step 2, Iterations 4) on consecutive cycles; MEM[44] writes them
    to four consecutive words (Step 1, Iterations 4). (b) MEM[8] reads ws[3] down to ws[0] (Countdown 1, Step -1,
    Iterations 4: period 2); MEM[45] writes them upwards with the same period."""
    k = Kit(p, "mem/agt_step", "strided, descending and periodic AGT loops", seed, module="mem")
    rng = random.Random(seed)
    xs, ws = words(rng, p, 7), words(rng, p, 4)
    block(k, "xs", 4, 0x10, xs)
    block(k, "ws", 8, 0x10, ws)
    b = k.b
    va = b.read_loop(0, 4, 0x10, "E", 0, [("step", 2), ("iterations", 4)], addr_text=["xs"])
    refs = b.write_loop(va, 44, 0x30, [("step", 1), ("iterations", 4)])
    vb = b.read_loop(0, 8, 0x13, "E", 1, [("countdown", 1), ("step", -1), ("iterations", 4)], addr_text=["ws+0x3"])
    refs += b.write_loop(vb, 45, 0x30, [("countdown", 1), ("step", 1), ("iterations", 4)])
    for i in range(4):
        k.expect(44, 0x30 + i, xs[2 * i])
        k.expect(45, 0x30 + i, ws[3 - i])
    k.last = max(c for c, _, _ in refs)
    return k.done("agt_step")


def agt_levels(p, seed=1):
    """(a) Two levels, a worked example: MEM[4] Iterations 3, Countdown 2, Step 1 reads A, A, A, A+1, A+1, A+1
    at p, p+1, p+2, p+5, p+6, p+7; MEM[44] writes them to six words with Step 1, Iterations 3, Countdown 2, Step 1
    (the same cadence). (b) Four levels of Step 1 (the FIG. 10 traversal extended): 16 consecutive words read by
    MEM[8] and written by MEM[45]. (c) StreamStep: MEM[12] Step 1, StreamStep 1, Iterations 4 puts its references on
    streams that advance once the record has arrived; MEM[46] writes them back with the same configuration."""
    k = Kit(p, "mem/agt_levels", "two- and four-level AGT loops and StreamStep", seed, module="mem")
    rng = random.Random(seed)
    xa, xb, xc = words(rng, p, 2), words(rng, p, 16), words(rng, p, 4)
    block(k, "xa", 4, 0x10, xa)
    block(k, "xb", 8, 0x20, xb)
    block(k, "xc", 12, 0x10, xc)
    b = k.b
    refs = []
    va = b.read_loop(0, 4, 0x10, "E", 0, [("iterations", 3), ("countdown", 2), ("step", 1)], addr_text=["xa"])
    refs += b.write_loop(va, 44, 0x30, [("step", 1), ("iterations", 3), ("countdown", 2), ("step", 1)])
    vb = b.read_loop(0, 8, 0x20, "E", 1, [("step", 1)] * 4, addr_text=["xb"])
    refs += b.write_loop(vb, 45, 0x40, [("step", 1)] * 4)
    vc = b.read_loop(0, 12, 0x10, "E", 8, [("step", 1), ("streamstep", 1), ("iterations", 4)], addr_text=["xc"])
    refs += b.write_loop(vc, 46, 0x50, [("step", 1), ("streamstep", 1), ("iterations", 4)])
    for i in range(6):
        k.expect(44, 0x30 + i, xa[i // 3])
    for i in range(16):
        k.expect(45, 0x40 + i, xb[i])
    for i in range(4):
        k.expect(46, 0x50 + i, xc[i])
    k.last = max(c for c, _, _ in refs)
    return k.done("agt_levels")


def agt_long_loop(p, seed=1, n=46):
    """A loop longer than the 45-register stream window: MEM[4] reads n = 46 consecutive words (Step 1,
    Iterations 46) onto one stream, one per cycle, and MEM[44] on the other hemisphere writes them with a second
    loop of the same shape."""
    k = Kit(p, "mem/agt_long_loop", f"{n} references read by one loop and written by another", seed, module="mem")
    rng = random.Random(seed)
    xs = words(rng, p, n)
    block(k, "xs", 4, 0x10, xs)
    b = k.b
    v = b.read_loop(0, 4, 0x10, "E", 0, [("step", 1), ("iterations", n)], addr_text=["xs"])
    refs = b.write_loop(v, 44, 0x40, [("step", 1), ("iterations", n)])
    for i in range(n):
        k.expect(44, 0x40 + i, xs[i])
    k.last = max(c for c, _, _ in refs)
    return k.done("agt_long_loop")


def rw_pair(p, seed=1):
    """A Read and a Write of one thread in one parcel of MEM[4]. (a) Different banks: MEM[8] (upstream on row E)
    reads y so that it passes slice 4's gap at t; MEM[4] reads x (bank 0) and writes y to a bank-1 word in that
    parcel; MEM[44] writes x downstream. (b) Same bank: the assemblers refuse such a parcel statically, so the pair
    is formed at run time by an AGT reference (reference 1 of a Read loop, Iterations 2) and a Write in one cycle:
    both are OTS-E-MEM-001 and neither runs, so the target word keeps its old contents."""
    k = Kit(p, "mem/rw_pair", "Read + Write parcels in different banks and in one bank (MEM-001)", seed,
            module="mem")
    rng = random.Random(seed)
    x, y, old, x2, y2 = words(rng, p, 5)
    hi = 1 << p.bank_bit
    k.extern_word("x", 4, 0x10, x)
    k.extern_word("old", 4, 0x11, old)
    k.extern_word("x2", 4, 0x12, x2)
    k.extern_word("y", 8, 0x10, y)
    k.extern_word("y2", 8, 0x11, y2)
    b = k.b
    t = 8
    vy = b.read_for(t, "write", 4, [8], [0x10], "E", 4, addr_text=["y"])
    vx = b.read(t, [4], [0x10], "E", 0, addr_text=["x"])
    b.write(vy, [4], [hi + 0x20], cycle=t)
    k.last = b.write(vx, [44], [0x30])
    k.expect(4, hi + 0x20, y)
    k.expect(44, 0x30, x)
    t2 = k.last + 4
    b.read_for(t2, "write", 4, [8], [0x11], "E", 5, addr_text=["y2"])
    b.raw(t2 - 1, 4, "read x2, E:6, 0x0", slot="R")
    b.raw(t2, 4, "iterations 0x2", slot="config")
    b.raw(t2, 4, "write 0x11, E:5, 0x0", slot="W")
    k.diags += [(t2, 4, "OTS-E-MEM-001"), (t2, 4, "OTS-E-MEM-001")]
    k.expect(4, 0x11, old)
    k.last = t2
    return k.done("rw_pair")


def gather_scatter(p, seed=1, n_table=8):
    """Gather and Scatter move one element per tile (microarchitecture.md sec. 4): tile g takes its
    address from bytes 0..1 of superlane g of the map stream, and moves lane 16g + k. (a) MEM[8] reads the map m
    once and repeats it 15 times, so it passes slice 4's gap on 16 consecutive cycles; MEM[4] issues 16 Gathers,
    k = 0..15, from its table; MEM[44] writes the 16 results with one Step 1 loop: result k holds T[m_g][16g + k] on
    lane 16g + k and zero elsewhere. (b) MEM[44] scatters lane 16g + 5 of a vector v (from slice 5) to word m'_g
    of its own old words (map m' from slice 6); every other byte keeps its old value. Lanes 2..15 of each map
    superlane hold random bytes, which must be ignored."""
    k = Kit(p, "mem/gather_scatter", "16 Gathers k = 0..15 and a Scatter, one element per tile", seed, module="mem")
    rng = random.Random(seed)
    b = k.b
    d_mem, _ = p.timing("gather")
    table = words(rng, p, n_table)
    base_t, base_o = 0x40, 0x40
    block(k, "tbl", 4, base_t, table)

    def map_word(base):
        w = bytearray(rand_word(rng, p))
        addrs = [base + rng.randrange(n_table) for _ in range(p.superlanes)]
        for g, a in enumerate(addrs):
            w[16 * g], w[16 * g + 1] = a & 0xFF, a >> 8
        return bytes(w), addrs

    m, addrs = map_word(base_t)
    k.extern_word("m", 8, 0x10, m)
    t_g = 8                                            # first Gather
    g4 = stations.need_gap(p, 4, "E", "write")
    vm = [b.read_for(t_g + i, "gather", 4, [8], [0x10], "E", 0, addr_text=["m"], emit=(i == 0)) for i in range(16)]
    b.raw(vm[0].cycle - p.timing("read")[0] + 1, 8, "repeat 0xf, 0x0", slot=None)
    res = []
    for i in range(16):
        b._read(vm[i], 4, "gather", t_g + i)
        b.raw(t_g + i, 4, f"gather E:4, E:0, {i:#x}, 0x0", slot="R")
        res.append(b._write("E", 4, 1, g4, t_g + i + d_mem))
        out = bytearray(p.vlanes)
        for g, a in enumerate(addrs):
            out[16 * g + i] = table[a - base_t][16 * g + i]
        k.expect(44, 0x30 + i, bytes(out))
    refs = b.write_loop(res, 44, 0x30, [("step", 1), ("iterations", 16)])
    # (b) Scatter
    old = words(rng, p, n_table)
    block(k, "old", 44, base_o, old)
    v = rand_word(rng, p)
    k.extern_word("v", 5, 0x10, v)
    m2, addrs2 = map_word(base_o)
    k.extern_word("m2", 6, 0x10, m2)
    t_s = refs[-1][0] + 2
    vv = b.read_for(t_s, "scatter", 44, [5], [0x10], "E", 8, addr_text=["v"])
    vm2 = b.read_for(t_s, "scatter", 44, [6], [0x10], "E", 9, addr_text=["m2"])
    b._read(vv, 44, "scatter", t_s)
    b._read(vm2, 44, "scatter", t_s)
    b.raw(t_s, 44, "scatter E:8, E:9, 0x5, 0x0", slot="W")
    new = [bytearray(w) for w in old]
    for g, a in enumerate(addrs2):
        new[a - base_o][16 * g + 5] = v[16 * g + 5]
    for i, w in enumerate(new):
        k.expect(44, base_o + i, bytes(w))
    k.last = t_s
    return k.done("gather_scatter")


def rw_pair_static(p, seed=1):
    """The static half of rw_pair: a Read and a Write of one thread in one bundle of MEM[4], both in bank 0. Both
    assemblers refuse the program with OTS-E-MEM-001 (assembly.md), so nothing runs."""
    k = Kit(p, "mem/rw_pair_static", "a same-bank Read + Write bundle is refused by both assemblers", seed,
            module="mem")
    k.b.raw(8, 4, "read 0x10, E:0, 0x0", slot="R")
    k.b.raw(8, 4, "write 0x11, E:1, 0x0", slot="W")
    text = k.b.finish()
    return text, emit.case(k.name, p, golden={"source": "hand", "generator": "gen/mem.py:rw_pair_static"},
                           cycles=16, tags=(), asm_error="OTS-E-MEM-001")


CASES = [
    ("mem/rw_pair_static", rw_pair_static, PROFILES, {}),
    ("mem/gather_scatter", gather_scatter, PROFILES, {}),
    ("mem/gather_partial", gather_partial, ["mini", "test/mid_min"], {}),
    ("mem/agt_step", agt_step, PROFILES, {}),
    ("mem/agt_levels", agt_levels, PROFILES, {}),
    ("mem/agt_long_loop", agt_long_loop, PROFILES, {}),
    ("mem/rw_pair", rw_pair, PROFILES, {}),
]
