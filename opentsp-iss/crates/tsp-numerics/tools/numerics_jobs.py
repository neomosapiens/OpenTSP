"""Job lists and oracle evaluation for the tsp-numerics differential checks.

Shared by gen_reference.py (mpmath oracle, opentsp-arch-spec/test/oracle/numerics.py) and
gen_jobs.py (job files for the Sail evaluator, tools/sail_dump.sh).

Job line format (also the format of the checked-in reference files, "<job> = <result hex>"):
  tr <op 21..24> <width 16|32> <x>      transcendental of an f16 / f32 element
  i8 <op 21..24> <x>                    i8 transcendental
  bin <op> <type code> <x> <y>          k_binary
  un <op> <type code> <x>               k_unary (transcendentals included)
  cmp <mode> <type code> <x> <y>        k_cmp
  cvt <src code> <dst code> <x>         k_convert_any
  conv <op 25..32> <x>                  k_convert
  nz <type code> <x>                    k_nonzero (1 / 0)
Values are hex without prefix, everything else decimal.

The "hash sets" are streams of results over input sets that are too large to check in. Each is
defined by a spec line (see SET_SPECS) that the Rust test re-reads from the reference file; the
result stream is the results as little-endian bytes of the result width, in input order, hashed
with SHA-256. Input sets (the generators below are mirrored in tests/common/mod.rs):
  all8        unary: x = 0..255; binary: x = 0..255 outer, y = 0..255 inner
  all16       unary: x = 0..65535
  rand S N    the edge values first (unary: each; binary: every pair, x outer), then N values or
              pairs from the type's generator seeded with splitmix64(S)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE_DIR = os.path.normpath(os.path.join(HERE, "../../../../opentsp-arch-spec/test/oracle"))
sys.path.insert(0, os.path.dirname(ORACLE_DIR))
from oracle import numerics as O  # noqa: E402

M64 = (1 << 64) - 1
TYPES = O.TYPES  # code order: i8 f16 i32 f32 u8 i16 u16 u32 b8 b16 b32
TNAME = {c: n for c, n in enumerate(TYPES)}


class Rng:
    """splitmix64"""

    def __init__(self, seed):
        self.s = seed & M64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


# ------------------------------------------------------------------ value generators

F16_EDGES = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x0401, 0x3BFF, 0x3C00, 0xBC00,
             0x3C01, 0x4000, 0x4400, 0x5640, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFE00,
             0x7FFF, 0x0200, 0x3555, 0x6800, 0x0100]
F32_EDGES = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x00800001,
             0x3F7FFFFF, 0x3F800000, 0xBF800000, 0x3F800001, 0x40000000, 0x4B800000, 0x7F7FFFFF,
             0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7F800001, 0x7FC00000, 0xFFC00000, 0x7FFFFFFF,
             0x33800000, 0x3EAAAAAB, 0x5F000000, 0x00400000]


def edges(t):
    n = O.nbits(TNAME[t]) if isinstance(t, int) else O.nbits(t)
    name = TNAME[t] if isinstance(t, int) else t
    if name == "f16":
        return list(F16_EDGES)
    if name == "f32":
        return list(F32_EDGES)
    mask = (1 << n) - 1
    h = mask >> 1
    vals = [0, 1, 2, 3, h - 1, h, h + 1, h + 2, mask - 2, mask - 1, mask, mask >> 2, (mask >> 2) + 1]
    out = []
    for v in vals:
        v &= mask
        if v not in out:
            out.append(v)
    return out


def rand_float(rng, name):
    n = O.nbits(name)
    mask = (1 << n) - 1
    f = O.fmt(name)
    r = rng.next()
    k = r & 7
    r >>= 3
    if k < 3:
        return rng.next() & mask
    sign = r & 1
    e = f.bias + ((r >> 1) % 17) - 8
    m = rng.next() & ((1 << f.mbits) - 1)
    return (sign << (n - 1)) | (e << f.mbits) | m


def rand_int(rng, name):
    n = O.nbits(name)
    mask = (1 << n) - 1
    r = rng.next()
    k = r & 3
    if k == 1:
        return rng.next() & ((1 << (n // 2)) - 1)
    if k == 2:
        base = [0, mask >> 1, (mask >> 1) + 1, mask][(r >> 2) & 3]
        return (base + ((r >> 4) & 3) - 1) & mask
    return rng.next() & mask


def rand_val(rng, t):
    name = TNAME[t]
    return rand_float(rng, name) if O.is_float(name) else rand_int(rng, name)


def rand_pair(rng, t):
    name = TNAME[t]
    if O.is_float(name):
        n = O.nbits(name)
        sign = 1 << (n - 1)
        mask = (1 << n) - 1
        x = rand_float(rng, name)
        k = rng.next() & 3
        if k == 1:
            y = x ^ sign
        elif k == 2:
            y = (x ^ sign) ^ (rng.next() & 0xF)
        else:
            y = rand_float(rng, name)
        return x, y & mask
    return rand_val(rng, t), rand_val(rng, t)


def unary_inputs(t, spec):
    """spec: ('all8',), ('all16',) or ('rand', seed, n)"""
    if spec[0] == "all8":
        return list(range(256))
    if spec[0] == "all16":
        return list(range(65536))
    _, seed, n = spec
    rng = Rng(seed)
    return edges(t) + [rand_val(rng, t) for _ in range(n)]


def binary_inputs(t, spec):
    if spec[0] == "all8":
        return [(x, y) for x in range(256) for y in range(256)]
    _, seed, n = spec
    rng = Rng(seed)
    e = edges(t)
    return [(x, y) for x in e for y in e] + [rand_pair(rng, t) for _ in range(n)]


# ------------------------------------------------------------------ f16 dot cases and reduce groups

def f16_finite(rng, lo, hi):
    """a finite f16 with biased exponent field in lo..hi"""
    r = rng.next()
    return ((r & 1) << 15) | ((lo + (r >> 1) % (hi - lo + 1)) << 10) | (rng.next() & 0x3FF)


def dot_case(rng, idx, lanes):
    """A case idx of the f16 dot product: (ws, xs) lists of f16 patterns."""
    k = idx % 6
    ws, xs = [], []
    if k == 0:
        for _ in range(lanes):
            ws.append(f16_finite(rng, 0, 30))
            xs.append(f16_finite(rng, 0, 30))
    elif k == 1:
        for _ in range(lanes):
            ws.append(f16_finite(rng, 10, 20))
            xs.append(f16_finite(rng, 10, 20))
    elif k == 2:  # exact cancellation, with a one-bit residual in odd cases
        for _ in range(lanes // 2):
            w = f16_finite(rng, 8, 22)
            x = f16_finite(rng, 8, 22)
            ws += [w, w]
            xs += [x, x ^ 0x8000]
        while len(ws) < lanes:
            ws.append(0)
            xs.append(0)
        if idx & 1:
            xs[0] ^= 1
    elif k == 3:  # wide dynamic range
        for _ in range(lanes):
            ws.append(f16_finite(rng, 1, 3) if rng.next() & 1 else f16_finite(rng, 27, 30))
            xs.append(f16_finite(rng, 1, 3) if rng.next() & 1 else f16_finite(rng, 27, 30))
    elif k == 4:  # zeros
        if idx % 4 == 0:  # every product is -0
            for _ in range(lanes):
                s = rng.next() & 1
                ws.append(s << 15)
                xs.append((1 - s) << 15)
        else:
            for _ in range(lanes):
                if rng.next() & 1:
                    ws.append((rng.next() & 1) << 15)
                    xs.append(f16_finite(rng, 0, 30))
                else:
                    ws.append(f16_finite(rng, 8, 22))
                    xs.append(f16_finite(rng, 8, 22))
    else:  # one special lane among moderate ones
        for _ in range(lanes):
            ws.append(f16_finite(rng, 8, 22))
            xs.append(f16_finite(rng, 8, 22))
        pos = rng.next() % lanes
        sel = (idx // 6) % 6
        sg = (rng.next() & 1) << 15
        if sel == 0:
            ws[pos] = 0x7C00 | sg
        elif sel == 1:
            ws[pos] = 0x7C00 | sg
            xs[pos] = (rng.next() & 1) << 15       # Inf x 0
        elif sel == 2:
            xs[pos] = 0x7E00 | (rng.next() & 0x1FF)  # NaN
        elif sel == 3:
            ws[pos] = 0x7C00
            xs[pos] = 0x3C00
            ws[(pos + 1) % lanes] = 0x7C00
            xs[(pos + 1) % lanes] = 0xBC00           # +Inf and -Inf
        elif sel == 4:
            ws[pos] = 0x7BFF
            xs[pos] = 0x7BFF                         # overflow to Inf in f32? 65504^2 fits f32
        else:
            ws[pos] = 0x7C00 | sg
            xs[pos] = 0x3C00
    return ws, xs


def reduce_groups(t, seed, n):
    rng = Rng(seed)
    return [[rand_val(rng, t) for _ in range(16)] for _ in range(n)]


# ------------------------------------------------------------------ oracle evaluation

def oracle_job(job):
    """Result of a job line as an int (no ' = ...'), by the mpmath oracle."""
    f = job.split()
    kind = f[0]
    if kind == "tr":
        op, w, x = int(f[1]), int(f[2]), int(f[3], 16)
        return O.fp_transcendental(op, O.F16 if w == 16 else O.F32, x)
    if kind == "i8":
        return O.i8_transcendental(int(f[1]), int(f[2], 16))
    if kind == "bin":
        return O.k_binary(int(f[1]), TNAME[int(f[2])], int(f[3], 16), int(f[4], 16))
    if kind == "un":
        return O.k_unary(int(f[1]), TNAME[int(f[2])], int(f[3], 16))
    if kind == "cmp":
        return O.k_cmp(int(f[1]), TNAME[int(f[2])], int(f[3], 16), int(f[4], 16))
    if kind == "cvt":
        return O.k_convert_any(TNAME[int(f[1])], TNAME[int(f[2])], int(f[3], 16))
    if kind == "conv":
        return O.k_convert(int(f[1]), int(f[2], 16))
    if kind == "nz":
        return 1 if O.nonzero(TNAME[int(f[1])], int(f[2], 16)) else 0
    raise ValueError(job)


def fmt_result(r):
    return "%08x" % (r & 0xFFFFFFFF)


def job_line(kind, *args):
    return " ".join([kind] + [a if isinstance(a, str) else str(a) for a in args])


def hx(v):
    return "%x" % v


# ------------------------------------------------------------------ the sample (checked in)

def sample_jobs():
    """~3000 jobs: i8 transcendentals exhaustive; strided f16 and random f32 transcendentals;
    random and edge jobs for every kernel and type."""
    jobs = []
    for op in (21, 22, 23, 24):
        for x in range(256):
            jobs.append(job_line("i8", op, hx(x)))
    for op in (21, 22, 23, 24):
        for x in range(0, 0x10000, 211):
            jobs.append(job_line("tr", op, 16, hx(x)))
        for x in F16_EDGES:
            jobs.append(job_line("tr", op, 16, hx(x)))
        rng = Rng(1000 + op)
        for _ in range(60):
            jobs.append(job_line("tr", op, 32, hx(rand_float(rng, "f32"))))
        for x in F32_EDGES:
            jobs.append(job_line("tr", op, 32, hx(x)))
    rng = Rng(7)
    for t in range(11):
        for op in range(14):
            for _ in range(6):
                x, y = rand_pair(rng, t)
                jobs.append(job_line("bin", op, t, hx(x), hx(y)))
        for op in (14, 15, 16, 20):
            for _ in range(5):
                jobs.append(job_line("un", op, t, hx(rand_val(rng, t))))
        for mode in range(6):
            for _ in range(5):
                x, y = rand_pair(rng, t)
                jobs.append(job_line("cmp", mode, t, hx(x), hx(y)))
        for _ in range(3):
            jobs.append(job_line("nz", t, hx(rand_val(rng, t))))
        for dt in range(11):
            for _ in range(3):
                jobs.append(job_line("cvt", t, dt, hx(rand_val(rng, t))))
    for x in F16_EDGES[:12]:
        for y in F16_EDGES[:12]:
            for op in (0, 3, 6, 9, 10):
                jobs.append(job_line("bin", op, 1, hx(x), hx(y)))
    for x in F32_EDGES[:12]:
        for y in F32_EDGES[:12]:
            for op in (0, 3, 6, 9, 10):
                jobs.append(job_line("bin", op, 3, hx(x), hx(y)))
    for op in range(25, 33):
        src = {25: 0, 26: 3, 27: 0, 28: 1, 29: 0, 30: 2, 31: 1, 32: 3}[op]
        for _ in range(12):
            jobs.append(job_line("conv", op, hx(rand_val(rng, src))))
    return jobs


# ------------------------------------------------------------------ the hash sets

def set_specs():
    """[(name, kind, args, input spec)]: the streams hashed by gen_reference.py and re-computed
    by tests/oracle_hashes.rs."""
    specs = []
    for t in (0, 4, 8):  # i8 u8 b8, all pairs
        for op in range(14):
            specs.append(("bin_%s_%d" % (TNAME[t], op), "bin", [op, t], ("all8",)))
        for mode in range(6):
            specs.append(("cmp_%s_%d" % (TNAME[t], mode), "cmp", [mode, t], ("all8",)))
    for t in range(11):
        for op in (14, 15, 16, 20):
            w = O.nbits(TNAME[t])
            if w == 8:
                specs.append(("un_%s_%d" % (TNAME[t], op), "un", [op, t], ("all8",)))
            elif w == 16:
                specs.append(("un_%s_%d" % (TNAME[t], op), "un", [op, t], ("all16",)))
            else:
                specs.append(("un_%s_%d" % (TNAME[t], op), "un", [op, t], ("rand", 100 + t, 20000)))
    for t in (1, 5, 6, 2, 3, 7, 9, 10):  # f16 i16 u16 i32 f32 u32 b16 b32
        ops = (0, 3, 6, 9, 10) if O.is_float(TNAME[t]) else range(14)
        n = {1: 100000, 3: 60000}.get(t, 20000)
        for op in ops:
            specs.append(("bin_%s_%d" % (TNAME[t], op), "bin", [op, t], ("rand", 200 + t * 20 + op, n)))
        for mode in range(6):
            specs.append(("cmp_%s_%d" % (TNAME[t], mode), "cmp", [mode, t], ("rand", 400 + t * 20 + mode, 20000)))
    for st in range(11):
        w = O.nbits(TNAME[st])
        for dt in range(11):
            if w == 8:
                inp = ("all8",)
            elif w == 16:
                inp = ("all16",)
            else:
                inp = ("rand", 600 + st * 11 + dt, 30000)
            specs.append(("cvt_%s_%s" % (TNAME[st], TNAME[dt]), "cvt", [st, dt], inp))
    for op in range(25, 33):
        st = {25: 0, 26: 3, 27: 0, 28: 1, 29: 0, 30: 2, 31: 1, 32: 3}[op]
        w = O.nbits(TNAME[st])
        inp = ("all8",) if w == 8 else ("all16",) if w == 16 else ("rand", 800 + op, 30000)
        specs.append(("conv_%d" % op, "conv", [op], inp))
    for op in (21, 22, 23, 24):
        specs.append(("i8tr_%d" % op, "i8", [op], ("all8",)))
        specs.append(("tr_f16_%d" % op, "tr", [op, 16], ("all16",)))
    for t in (0, 1, 2, 3, 4, 5, 6, 7):
        for rop in (0, 9, 10):
            specs.append(("reduce_%s_%d" % (TNAME[t], rop), "reduce", [rop, t], ("rand", 900 + t * 3 + rop, 3000)))
    specs.append(("dot_f16_320", "dot16", [320], ("rand", 1001, 600)))
    specs.append(("dot_f16_16", "dot16", [16], ("rand", 1002, 3000)))
    specs.append(("dot_f16_7", "dot16", [7], ("rand", 1003, 3000)))
    return specs


def spec_results(kind, args, inp):
    """Oracle results (list of (value, width_bytes)) for a set spec, in input order."""
    out = []
    if kind in ("bin", "cmp"):
        op, t = args
        fn = O.k_binary if kind == "bin" else O.k_cmp
        name = TNAME[t]
        w = O.WIDTH[name]
        for x, y in binary_inputs(t, inp):
            out.append((fn(op, name, x, y), w))
    elif kind == "un":
        op, t = args
        name = TNAME[t]
        w = O.WIDTH[name]
        for x in unary_inputs(t, inp):
            out.append((O.k_unary(op, name, x), w))
    elif kind == "cvt":
        st, dt = args
        sn, dn = TNAME[st], TNAME[dt]
        for x in unary_inputs(st, inp):
            out.append((O.k_convert_any(sn, dn, x), O.WIDTH[dn]))
    elif kind == "conv":
        op = args[0]
        src = {25: 0, 26: 3, 27: 0, 28: 1, 29: 0, 30: 2, 31: 1, 32: 3}[op]
        dst = {25: 3, 26: 0, 27: 1, 28: 0, 29: 2, 30: 0, 31: 3, 32: 1}[op]
        for x in unary_inputs(src, inp):
            out.append((O.k_convert(op, x), O.WIDTH[TNAME[dst]]))
    elif kind == "i8":
        for x in range(256):
            out.append((O.i8_transcendental(args[0], x), 1))
    elif kind == "tr":
        op, w = args
        for x in range(65536):
            out.append((O.fp_transcendental(op, O.F16, x), 2))
    elif kind == "reduce":
        rop, t = args
        name = TNAME[t]
        kop = {0: 0, 9: 9, 10: 10}[rop]
        _, seed, n = inp
        for g in reduce_groups(t, seed, n):
            acc = g[0]
            for e in g[1:]:
                acc = O.k_binary(kop, name, acc, e)
            out.append((acc, O.WIDTH[name]))
    elif kind == "dot16":
        lanes = args[0]
        _, seed, n = inp
        rng = Rng(seed)
        for i in range(n):
            ws, xs = dot_case(rng, i, lanes)
            out.append((O.dot_f16(ws, xs), 4))
    else:
        raise ValueError(kind)
    return out
