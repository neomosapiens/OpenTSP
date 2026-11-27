"""Category E of the design's sec. 2.2, MXM: a matrix-vector product through LW, IW, ABC and ACC (the J1 migration of
tests/e2e/mxm_dot.sh and weighted_mxm.sh). Every cycle follows from the profile through the toolchain; the expected words
are the exact integer products computed with numpy."""
import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs.kernels.install import install

# One matrix of mxm.rows rows is large (102 KB on ref): the cases run on ref and mini only.
PROFILES = ["ref", "mini"]

ADDR_W = 0x10        # first word of a matrix slice: row 16 * k + j is word ADDR_W + k of slice j
ADDR_X = 0x30        # activation word on slice 16 (and 17)
ADDR_Y = 0x30        # result words on west slices 4..7 (the four byte planes)


def planes_i32(y):
    return data.to_planes(y.astype(np.int32))


def matmul(p, name, doc, seed, weighted, accumulate, abc_type, wtype="i8"):
    """y = W x with W an i8 (`wtype`) matrix of mxm.rows rows by vlanes columns (LW of 16 rows per cycle, one per
    stream, from 16 west slices; IW; ABC; ACC). `weighted`: matrix and activations are externs bound from an `.safetensors`
    package, else they are loaded into SRAM through the harness's S lines and read by address. `abc_type` is the
    activation type:
    `u8` pairs the i8 weights with unsigned activations, and `i8` activations with u8 weights. With
    `accumulate` a second activation x2 is multiplied one cycle later and accumulated: its word holds y1 + y2 (i32,
    wrapping), next to y1."""
    k = Kit(p, name, doc, seed, module="mxm", tags=("sim", "e2e"))
    rows, n = p.mxm_rows, p.vlanes
    nlw = rows // 16
    W = data.rand_int(k.rng, wtype, rows * n).reshape(rows, n)
    acts = [data.rand_int(k.rng, abc_type, n) for _ in range(2 if accumulate else 1)]
    words = {(j, ADDR_W + i): bytes(W[16 * i + j].view(np.uint8)) for j in range(16) for i in range(nlw)}
    xwords = {(16 + i, ADDR_X): bytes(a.view(np.uint8)) for i, a in enumerate(acts)}
    if weighted:
        for j in range(16):
            nm = f"w{j}"
            k.b.prog.extern(nm, j, ADDR_W, nlw)
            k.weights[nm] = (wtype, b"".join(words[(j, ADDR_W + i)] for i in range(nlw)))
        for i in range(len(acts)):
            nm = f"x{i}"
            k.b.prog.extern(nm, 16 + i, ADDR_X, 1)
            k.weights[nm] = (abc_type, xwords[(16 + i, ADDR_X)])
    else:
        k.preload = {**words, **xwords}
    b = k.b
    t_abc = install(k, wtype, nlw, weighted, ADDR_W)
    for i, a in enumerate(acts):
        vx = b.read_for(t_abc + i, "abc", 105, [16 + i], [ADDR_X], "W", 16 + i,
                        addr_text=[f"x{i}"] if weighted else None)
        b.unit(105, "abc", f"abc.{abc_type} W:{16 + i}", [vx])
    t_acc = t_abc + p.timing("abc")[0]
    prod = [(W.astype(np.int64) @ a.astype(np.int64)) for a in acts]
    _, v1 = b.unit(106, "acc", "acc.i32 E:8, replace, 0", cycle=t_acc, dst=("E", 8, 4))
    k.store(v1, 1, ADDR_Y, prod[0].astype(np.int32), hemi="W")
    if accumulate:
        _, v2 = b.unit(106, "acc", "acc.i32 E:12, accumulate, 0", cycle=t_acc + 1, dst=("E", 12, 4))
        k.store(v2, 1, ADDR_Y + 1, (prod[0] + prod[1]).astype(np.int32), hemi="W")
    return k.done(name.split("/")[1])


def matmul_i8(p, seed=1):
    """The J1 mxm_dot: SRAM preloaded through S lines; ABC and ACC twice (replace, then accumulate)."""
    return matmul(p, "mxm/matmul_i8", "y = W x, then y + W x2, i8 weights and activations, SRAM preloaded", seed,
                  weighted=False, accumulate=True, abc_type="i8")


def matmul_mixed(p, seed=3):
    """u8 weights with signed i8 activations (the unsigned-weight, signed-activation mode), then ACC
    replace and accumulate."""
    return matmul(p, "mxm/matmul_mixed", "y = W x, u8 weights from a package, i8 activations", seed,
                  weighted=True, accumulate=True, abc_type="i8", wtype="u8")


def weighted_matmul(p, seed=2):
    """The J1 weighted_mxm: the matrix and the activations come from an `.safetensors` package bound to externs; the
    activations are unsigned (abc.u8 on i8 weights)."""
    return matmul(p, "mxm/weighted_matmul", "y = W x, i8 weights from a package, u8 activations", seed,
                  weighted=True, accumulate=False, abc_type="u8")


def f16_matrix(k, rows):
    """A rows x vlanes f16 matrix and two activation vectors with rows that exercise the exact sum: rows 1 mod 4
    cancel exactly against the periodic first activation, rows 2 mod 4 cancel except for one ulp, rows 3 mod 4 mix
    products of 2^20 and 2^-28, row 0 holds an infinity and row 4 a NaN."""
    n = k.p.vlanes
    h = n // 2
    W = data.rand_float(k.rng, "f16", rows * n, -4, 4).reshape(rows, n)
    x1, x2, bad = (data.rand_float(k.rng, "f16", n, -4, 4) for _ in range(3))
    x1[0], x2[0] = np.abs(x1[0]) + np.float16(1), np.abs(x2[0]) + np.float16(1)
    x1[h:] = x1[:h]
    for r in range(rows):
        if r % 4 in (1, 2):
            W[r, h:] = -W[r, :h]
        if r % 4 == 2:
            W[r, h] = np.nextafter(W[r, h], np.float16(np.inf))
        if r % 4 == 3:
            W[r, ::2] = data.rand_float(k.rng, "f16", len(W[r, ::2]), 8, 11)
            W[r, 1::2] = data.rand_float(k.rng, "f16", len(W[r, 1::2]), -14, -10)
    W[0, 0], W[4, 3] = np.inf, np.nan
    return W, x1, x2, bad


def matmul_f16(p, seed=4):
    """y = W x with f16 weights on the even plane 0 (mxm.rows / 2 rows: LW takes 8 rows per cycle, a row being a
    stream pair, low byte then high byte) and f16 activations (a stream pair): the exact sum of the exact products
    rounded once to f32. ABC x1 at T, an intermediate ABC at T + 1, and ABC x2 at T + 2.
    The intermediate ABC is rejected only on a profile with occupancy greater than one. Three ACCs read the newest ABC committed at or before their
    dispatch: replace -> x1 (accumulator 0), accumulate -> x1 + x2 (one f32 add), replace -> x2 (accumulator 1);
    the VXM narrows the first result to f16 as it passes (MXM -> VXM -> MEM). An IW of f16 on the odd plane 1 is
    OTS-E-RES-001 too."""
    k = Kit(p, "mxm/matmul_f16", "f16 matrix-vector products, configured ABC occupancy, odd-plane f16", seed, module="mxm",
            tags=("sim", "e2e"), source="numpy-twin")
    rows, n = p.mxm_rows // 2, p.vlanes
    W, x1, x2, bad = f16_matrix(k, rows)
    nlw = rows // 8
    planes = [data.to_planes(W[r]) for r in range(rows)]
    for j in range(16):
        for i in range(nlw):
            k.preload[(j, ADDR_W + i)] = planes[8 * i + j // 2][j % 2]
    for slice_, v in ((16, x1), (18, bad), (20, x2)):
        for s, pl in zip((slice_, slice_ + 1), data.to_planes(v)):
            k.preload[(s, ADDR_X)] = pl
    b = k.b
    t_abc = install(k, "f16", nlw, False, ADDR_W)
    for dt, (slice_, name) in enumerate(((16, "x1"), (18, "bad"), (20, "x2"))):
        vx = b.read_for(t_abc + dt, "abc", 105, [slice_, slice_ + 1], [ADDR_X] * 2, "W", slice_)
        cycle, _ = b.unit(105, "abc", f"abc.f16 W:{slice_}", [vx])
        if name == "bad" and p.cfg["timing"]["mxm_abc"]["occupancy_f16"] > 1:
            k.diags.append((cycle + p.timing("abc")[1], 105, "OTS-E-RES-001"))
    y = [np.zeros(n, np.float32) for _ in range(2)]
    for out, x in zip(y, (x1, x2)):
        out[:rows] = nx.dot_f16(W, x)
    t_acc = t_abc + p.timing("abc")[0]
    for quad, (cycle, text, want) in enumerate((
            (t_acc, "acc.f32 E:8, replace, 0", y[0]),
            (t_acc + 2, "acc.f32 E:12, accumulate, 0", nx.binary("add", "f32", y[0], y[1])),
            (t_acc + 3, "acc.f32 E:16, replace, 1", y[1]))):
        _, v = b.unit(106, "acc", text, cycle=cycle, dst=("E", int(text.split("E:")[1].split(",")[0]), 4))
        k.store(v, 1 + quad, ADDR_Y, want, hemi="W")
        if quad == 0:      # the accumulator's stream also reaches the VXM, which narrows it to f16
            vn = b.vxm(0, "conv_f32_f16.f32", "E", 24, [v], width=2)
            k.store(vn, 0, ADDR_Y, nx.convert("f32", "f16", want), hemi="E")
    t_odd = t_acc + 10
    b.unit(107, "iw", "iw.f16", cycle=t_odd)
    k.diags.append((t_odd + p.timing("iw")[1], 107, "OTS-E-RES-001"))
    return k.done("matmul_f16")


CASES = [
    ("mxm/matmul_f16", matmul_f16, PROFILES, {}),
    ("mxm/matmul_i8", matmul_i8, PROFILES, {}),
    ("mxm/weighted_matmul", weighted_matmul, PROFILES, {}),
    ("mxm/matmul_mixed", matmul_mixed, PROFILES, {}),
]
