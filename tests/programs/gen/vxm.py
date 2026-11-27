"""Category C of the design's sec. 2.2, VXM kernels (the families
cmp_select_mask, gelu_chain, vxm_transcendental, strided_reduce and conv_* families). Every cycle follows from the
profile through the toolchain; every expectation is the numpy twin opentsp_tools.programs.numerics, which test_numerics.py checks against
the Python oracle (tier bit-exact reference). Operands are externs on west quad 1 read onto row E, results are written to
east quads."""
import numpy as np

from opentsp_tools.programs.case import Kit
from basic import PROFILES
from opentsp_tools.programs import data
from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs.kernels.gelu import gelu_chain
from opentsp_tools.programs.kernels import reduce_bcast as rb

ADDR = 0x10          # word of group j: ADDR + j (below 0x80, bank 0 on every profile)

SPECIAL = {          # zeros, subnormals, the extremes, one, infinities and NaNs (bit patterns)
    "f16": [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
            0x7E00, 0x7C01, 0xFE00, 0x3555, 0x4248],
    "f32": [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000,
            0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00000, 0x4B800000,
            0x33800000],
}


def kit(p, name, doc, seed):
    return Kit(p, name, doc, seed, module="vxm", source="numpy-twin")


def src_quad(p):
    """The west quad of the operands: 1, or 0 on a profile with one quad per hemisphere (the operand and result
    slices then differ by hemisphere; programs that need several quads are not generated for it)."""
    return 1 if p.slices_per_hemisphere > 4 else 0


def out_quad(p, i):
    return i if p.slices_per_hemisphere > 4 else 0


def vectors(k, t, specials, extra, lo=-12, hi=12):
    """Whole vlanes-lane vectors of type t: first `specials` (bit patterns, or None for every value of an 8-bit
    integer type), then `extra` vectors of random values (floats: finite, exponents in [lo, hi))."""
    n = k.p.vlanes
    if specials is None:
        specials = list(range(256))
    count = -(-len(specials) // n) + extra
    a = data.rand_float(k.rng, t, count * n, lo, hi) if t[0] == "f" else data.rand_int(k.rng, t, count * n)
    a[:len(specials)] = nx.from_bits(np.array(specials, dtype=np.uint64), t)
    return a.reshape(count, n)


def pack4(vecs):
    """Four 8-bit vectors as the four byte planes of one 32-bit-element array (an SG4 group)."""
    return np.stack([np.ascontiguousarray(v).view(np.uint8) for v in vecs], axis=1).copy().view("<u4").reshape(-1)


def alu(j):
    return j % 16


def lane_op(k, j, mnem, t, x, y, dst_type=None, text=None, start=0):
    """Group j: one VXM instruction `text` (default `mnem.t`) on x (type t), its result y (type dst_type) stored on
    east quad 0 at word ADDR + j. Groups are one cycle apart (from `start`) and use distinct ALU queues."""
    out_w = data.width(dst_type or t)
    op = text or f"{mnem}.{t}"
    tv = k.lead(op) + start + j
    vx = k.feed(k.place(f"x{j}", x, src_quad(k.p), ADDR + j), ADDR + j, "E", 0, tv, op)
    v = k.b.vxm(alu(j), op, "E", 8, [vx], width=out_w)
    k.store(v, 0, ADDR + j, y)


def transcendental(p, seed=1):
    """tanh, exp, rsqrt and log on i8 (all 256 values), f16 and f32 (special values: zeros, subnormals, extremes,
    infinities, NaNs; then random values), plus the i8 SG4 form, which runs four i8 vectors in one instruction
    (microarchitecture.md sec. 5)."""
    k = kit(p, "vxm/transcendental", "tanh, exp, rsqrt, log on i8, f16, f32", seed)
    j = 0
    for t in ("i8", "f16", "f32"):
        for op in nx.TR_NAMES:
            for x in vectors(k, t, None if t == "i8" else SPECIAL[t], 0 if t == "i8" else 1, -14, 6):
                lane_op(k, j, op, t, x, nx.transcendental(op, t, x))
                j += 1
    for op in nx.TR_NAMES:
        xs = vectors(k, "i8", None, 0)
        for first in range(0, len(xs) - len(xs) % 4 or 4, 4):
            grp = [xs[(first + i) % len(xs)] for i in range(4)]
            ys = [nx.transcendental(op, "i8", g) for g in grp]
            lane_op(k, j, op, "i8", pack4(grp), pack4(ys), dst_type="i32", text=f"{op}.i8.sg4")
            j += 1
    return k.done("transcendental")


def gelu_tanh(p, seed=1):
    """GELU, tanh form, on f16 and on f32: nine chained ALU operations each (x*x, *x, *0.044715, +x, *sqrt(2/pi),
    tanh, +1, *0.5, *x); x holds random values and the special values 0, -0 and the f16/f32 extremes."""
    k = kit(p, "vxm/gelu_tanh", "GELU in its tanh form, f16 and f32, nine chained ALUs", seed)
    t0 = max(k.lead(f"mul.{t}") for t in ("f16", "f32"))
    for slot, t in enumerate(("f16", "f32")):
        x = vectors(k, t, SPECIAL[t][:6] + SPECIAL[t][8:10], 1, -6, 4)[0]
        res, want = gelu_chain(k, t, t0 + 60 * slot, x, slot, src_quad(p), ADDR)
        k.store(res, out_quad(p, slot), ADDR + 40 * slot, want)
    return k.done("gelu_tanh")


def cmp_select_mask(p, seed=1):
    """For every type in i8, f16, f32 and every compare mode: m = cmp(x, y) on one ALU, and in the next cycle two ALUs
    read m through `.chain`: select(x, y, m) and mask(x, m). The inputs hold ties, signed zeros, infinities and NaNs
    (a NaN compares false except `ne`; a mask element is true when any bit is set). The three results go to east
    quads 0, 1 and 2."""
    k = kit(p, "vxm/cmp_select_mask", "cmp, select and mask with the compare result chained", seed)
    n, j = p.vlanes, 0
    for t in ("i8", "f16", "f32"):
        for mode in ("lt", "le", "gt", "ge", "eq", "ne"):
            for x in vectors(k, t, [] if t == "i8" else SPECIAL[t], 1, -3, 3):
                y = np.where(k.rng.random(n) < 0.4, x, x[k.rng.permutation(n)])
                m = nx.cmp(mode, t, x, y)
                w = data.width(t)
                # a group reads its operands twice, at tv (the compare) and delta cycles later (select and mask); groups `step` apart never
                # read one slice in one cycle when `step` does not divide delta (5 on the reference profile, unless that divides delta)
                delta = p.timing(f"cmp.{mode}.{t}")[0] + 1 - p.timing(f"select.{t}")[1]
                step = next(n for n in range(5, 40) if delta % n)
                tv, base = k.lead(f"cmp.{mode}.{t}") + step * j, ADDR + j
                at_x, at_y = k.place(f"x{j}", x, 1, base), k.place(f"y{j}", y, 2, base)
                at_z = k.place(f"z{j}", x, 3, base)      # a second copy: one read per slice and cycle
                vm = k.b.vxm(3 * j % 16, f"cmp.{mode}.{t}", "E", 20, [k.feed(at_x, base, "E", 0, tv, f"cmp.{mode}.{t}"),
                                                                       k.feed(at_y, base, "E", 4, tv, f"cmp.{mode}.{t}")],
                             width=w)
                t1 = vm.cycle + 1 - p.timing(f"select.{t}")[1]
                vs = k.b.vxm((3 * j + 1) % 16, f"select.{t}", "E", 24,
                             [k.feed(at_x, base, "E", 8, t1, f"select.{t}"), k.feed(at_y, base, "E", 12, t1, f"select.{t}"), vm],
                             chain=(2,), width=w)
                vk = k.b.vxm((3 * j + 2) % 16, f"mask.{t}", "E", 28, [k.feed(at_z, base, "E", 16, t1, f"mask.{t}"), vm],
                             chain=(1,), width=w)
                k.store(vm, 0, base, m)
                k.store(vs, 1, base, nx.select(t, x, y, m))
                k.store(vk, 2, base, nx.mask(t, x, m))
                j += 1
    return k.done("cmp_select_mask")


def reduce(p, seed=1):
    """reduce.add, .max and .min per superlane on f16, f32 and i8, and reduce.add on i32: lane 16g of the result
    holds the fold of superlane g in ascending lane order, one rounding per step; the other lanes are zero. The
    f16 and f32 add vectors hold the order-sensitive pattern 1, 1, 0 ... 0, 2^e (a descending fold gives 2^e) in
    superlane 0; the max and min vectors hold a NaN and both signed zeros there."""
    k = kit(p, "vxm/reduce", "per-superlane reduce, add max min, in the element type", seed)
    j = 0
    for t, modes in (("f16", ("add", "max", "min")), ("f32", ("add", "max", "min")), ("i8", ("add", "max", "min")),
                     ("i32", ("add",))):
        for mode in modes:
            x = vectors(k, t, [], 1, -6, 6)[0]
            if mode == "add" and t[0] == "f":
                x[:16] = 0.0
                x[0], x[1], x[15] = 1.0, 1.0, 2.0 ** (11 if t == "f16" else 24)
            elif t[0] == "f":
                x[3], x[5], x[6] = np.nan, -0.0, 0.0
            lane_op(k, j, "reduce", t, x, nx.reduce(mode, t, x), text=f"reduce.{mode}.{t}")
            j += 1
    return k.done("reduce")


TREES = (("f32", "add"), ("f32", "max"), ("f16", "add"), ("i8", "add"), ("i32", "min"))


def reduce_tree(p, seed=1):
    """The cross-superlane tree of the design (sec. 4.4) on segments of four superlanes: Reduce per superlane
    (VXM), then two rounds of Permute by a rotation of 2 and of 1 superlanes (maps loaded once, by LoadMap or by
    LoadMapWide above 256 lanes) and a combine, staging the planes of a value through MEM because the one permuter
    of a hemisphere moves one stream per cycle (tree_round). Every superlane of a segment ends with the same total in
    lane 16g, the other lanes are zero. add.f32, max.f32, add.f16, add.i8 and min.i32, one after the other."""
    k = kit(p, "vxm/reduce_tree", "reduce, then permute-and-combine rounds across the superlanes of a segment", seed)
    t_ready, t0 = rb.load_maps(k, 60), 30
    for j, (t, op) in enumerate(TREES):
        a0 = ADDR + 8 * j
        x = vectors(k, t, [], 1, -8, 8)[0]
        vx = k.feed(k.place(f"x{j}", x, 0, a0), a0, "E", 0, t0, f"reduce.{op}.{t}")
        t_w, a2 = rb.reduce_tree(k, t, op, vx, a0, t_ready)
        for i, plane in enumerate(data.to_planes(nx.reduce_bcast(op, t, x))):
            k.expect(44 + i, a2, plane)
        k.last = t_w
        t0 = t_w + 20
    return k.done("reduce_tree")


# the eight fixed conversions of the catalogue: (mnemonic, source type, destination type)
FIXED = [("conv_i8_f32", "i8", "f32"), ("conv_f32_i8", "f32", "i8"), ("conv_i8_f16", "i8", "f16"),
         ("conv_f16_i8", "f16", "i8"), ("conv_i8_i32", "i8", "i32"), ("conv_i32_i8", "i32", "i8"),
         ("conv_f16_f32", "f16", "f32"), ("conv_f32_f16", "f32", "f16")]
# Convert between other types: integers saturate, floats round to nearest even, a boolean is `value != 0`
GENERIC = [("f32", "u8"), ("f16", "i32"), ("i32", "f32"), ("u32", "f16"), ("f32", "b8"), ("i16", "f16"), ("f32", "i16")]


def conversion_inputs(k, t):
    """Vectors of type t that exercise the conversions: floats from tiny to beyond the integer ranges with
    ties at .5 and the special values; integers over the whole range and over a small one."""
    n = k.p.vlanes
    if t == "i8":
        return vectors(k, t, list(range(0, 256, 17)), 1)
    if t[0] == "f":
        xs = vectors(k, t, SPECIAL[t], 1, -20, 12 if t == "f32" else 14)
        xs[:, ::5] = (k.rng.integers(-200, 200, size=xs[:, ::5].shape) + 0.5).astype(data.DTYPES[t])
        return xs
    xs = vectors(k, t, [], 2)
    xs[:, ::3] = k.rng.integers(-300, 300, size=xs[:, ::3].shape).astype(data.DTYPES[t]) if t[0] == "i" else \
        k.rng.integers(0, 600, size=xs[:, ::3].shape).astype(data.DTYPES[t])
    return xs[:, :n]


def convert_chain(p, seed=1):
    """Chained: i8 -> f32 -> (* 0.75) -> f16 -> i8 on four ALUs, over every i8 value (halves round to even in
    the f16 step, and the integer step rounds to nearest even and saturates). Single: the eight fixed conversions
    and seven Convert forms on inputs from the helper above."""
    k = kit(p, "vxm/convert_chain", "chained and single conversions between i8, i16, i32, f16, f32", seed)
    n = p.vlanes
    steps = ("conv_i8_f32.i8", "mul.f32", "conv_f32_f16.f32", "conv_f16_i8.f16")
    j = 0
    for x in vectors(k, "i8", None, 0):
        c = np.full(n, 0.75, dtype=np.float32)
        want = nx.convert("f16", "i8", nx.convert("f32", "f16", nx.binary("mul", "f32", nx.convert("i8", "f32", x), c)))
        tv, base, a = k.lead(steps[0]) + j, ADDR + j, 4 * j
        v0 = k.b.vxm(a % 16, steps[0], "E", 8, [k.feed(k.place(f"x{j}", x, 1, base), base, "E", 0, tv,
                                                        steps[0])], width=4)
        t1 = v0.cycle + 1 - p.timing("mul.f32")[1]
        vc = k.feed(k.place(f"c{j}", c, 2, base), base, "E", 24, t1, "mul.f32")
        v1 = k.b.vxm((a + 1) % 16, steps[1], "E", 12, [v0, vc], chain=(0,), width=4)
        v2 = k.b.vxm((a + 2) % 16, steps[2], "E", 16, [v1], chain=(0,), width=2)
        v3 = k.b.vxm((a + 3) % 16, steps[3], "E", 20, [v2], chain=(0,), width=1)
        k.store(v3, 0, base, want)
        j += 1
    start = j + 24
    for i, (mnem, st, dt) in enumerate(FIXED + [("convert", st, dt) for st, dt in GENERIC]):
        text = f"{mnem}.{st}" if mnem != "convert" else f"convert.{st}.{dt}"
        for x in conversion_inputs(k, st):
            lane_op(k, j, mnem, st, x, nx.convert(st, dt, x), dst_type=dt, text=text, start=start)
            j += 1
    return k.done("convert_chain")


# the programs that need one source quad run on test/mid_min too (4 superlanes: Reduce crosses lanes of several)
SMALL = PROFILES + ["test/mid_min"]
CASES = [
    ("vxm/transcendental", transcendental, SMALL, {}),
    ("vxm/gelu_tanh", gelu_tanh, SMALL, {}),
    ("vxm/cmp_select_mask", cmp_select_mask, PROFILES, {}),
    ("vxm/reduce", reduce, SMALL, {}),
    ("vxm/reduce_tree", reduce_tree, ["ref", "test/ref_slow", "test/mid_min"], {}),
    ("vxm/convert_chain", convert_chain, PROFILES, {}),
]
