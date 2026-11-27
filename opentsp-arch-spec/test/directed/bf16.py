"""Directed end-to-end tests of the profile extension bf16 (bfloat16) on the Sail emulator (tsp_sim) or, with the ISS
as the simulator (tools/iss_as_sim.sh), on the ISS (microarchitecture.md sec. 5 "Extension types" and sec. 6).

A profile that does not declare the extension must refuse every bf16 word (OTS-E-DEC-003) and the reserved type codes
17..31 on every profile. A profile that declares it (config/test/ext_bf16.json) runs every program through the whole dataflow
(MEM read -> stream -> VXM or MXM -> stream -> MEM write) and the results are compared with the oracle
(test/oracle/numerics.py: exact fractions, no Sail):
  * Add, Sub, Mul, Max, Min and the six comparisons on a grid of bf16 values (every class: zeros, subnormals, normals,
    infinities, NaNs) and on random pairs, and the SG4 form;
  * Neg, Abs and ReLU on all 65536 values, Mask and Select, the reduce;
  * conversions: all 65536 bf16 values to f32, f16, i8, u8, i32, b16, e4m3; all f16 values, random and halfway binary32
    values and i8 to bf16;
  * the matrix product: the bf16 matrix-vector product W (64 x 320) . x with IW from streams, ABC, ACC f32 against the exact
    sum rounded once (the proof-of-concept case), with the cancellation a float32 accumulation would lose, the
    special values, and the type rules of the matrix unit.
usage: bf16.py <tsp_sim> <config.json>"""
import json, os, random, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "tools"))
sys.path.insert(0, os.path.join(here, "..", "oracle"))
from tsp_as import lower_records, q_lines
import numerics as nx

SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]
D_ABC, D_ACC = TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"]
VL = 16 * CJ["geometry"]["superlanes"]


def run(recs, sram={}, dumps=[]):
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    lines += q_lines(lower_records(recs)) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    r = subprocess.run([SIM, CFG, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=900)
    if r.returncode or r.stderr.strip():       # a refused input (for example an IQ image over icu.iq_depth) is a test error
        raise SystemExit(f"tsp_sim failed ({r.returncode}): {r.stderr.strip()[:300]}")
    o = r.stdout
    words = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:]) for t in (l.split() for l in o.splitlines()) if t and t[0] == 'W'}
    return o, words


def diags(o): return [l for l in o.splitlines() if "diag" in l]


ok = True
def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= cond


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    west = slice_ < 44
    up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row)
    sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


TY = dict(nx.CODE)          # the 5-bit type codes: bf16 is 16
def VXM(op, ty, d, a, b=0, m=0, mode=0, tx=0, bits=0):
    return (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22) | bits
def vxm_op(op, t, d, a, b=0, m=0, mode=0, g=0, dz=0):
    """A VXM word of element type t: the code is bit 4 (word bit 17) @ bits 3..2 (23..22) @ bits 1..0 (53..52); dz is bit 16, the
    type-extension bit of a Convert's destination type."""
    c = TY[t]
    return VXM(op, c & 3, d, a, b, m, mode, (c >> 2) & 3, (g << 18) | ((c >> 4) << 17) | (dz << 16))
def mxm(op, t, s=0, mode=0, acc=0, h=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | ((acc | h) << 12)


BATCH = 96    # vectors per program: every MEM slice queue holds one read or write per vector, and its image is bounded by icu.iq_depth


def vxm_vectors(make, operands, wout, nvec, name="vxm", sg4=False):
    """Like vxm_batch, in programs of at most BATCH vectors."""
    out, d = [], []
    for lo in range(0, nvec, BATCH):
        hi = min(nvec, lo + BATCH)
        o, dd = vxm_batch(make, [(w, vecs[lo:hi]) for w, vecs in operands], wout, hi - lo)
        out += o; d += dd
    return out, d


def vxm_batch(make, operands, wout, nvec):
    """Run `nvec` vectors through one VXM instruction each (one per cycle on queues 88 + k mod 16).
    operands: lists (per operand) of nvec vectors, each a list of VL raw element values of width w; the width is
    the operand's. wout: bytes of the result element. make(k, srcs, dst) builds the 64-bit word.
    Returns the list of nvec result vectors (lists of raw element values)."""
    sram = {}; recs = []; T = 60
    srcs = []
    for j, (w, vecs) in enumerate(operands):
        base = 8 * j
        srcs.append(base)
        for k in range(nvec):
            for b in range(w):
                plane = bytes((v >> (8 * b)) & 0xFF for v in vecs[k])
                feed(recs, sram, base + b, k, plane, base + b, "E", 21, T + k + VXM_DS)
    dst = 24
    for k in range(nvec):
        recs.append((T + k, 88 + (k % 16), 1, make(k, srcs, dst), 0))
        for b in range(wout):
            drain(recs, 44 + 4 * b, k, dst + b, "E", 21, T + k + DV)
    dumps = [(44 + 4 * b, k) for k in range(nvec) for b in range(wout)]
    o, wd = run(recs, sram, dumps)
    out = []
    for k in range(nvec):
        planes = [wd.get((44 + 4 * b, k), bytes(VL)) for b in range(wout)]
        out.append([sum(planes[b][l] << (8 * b) for b in range(wout)) for l in range(VL)])
    return out, diags(o)


def chunks(vals, pad=0):
    vals = list(vals)
    vals += [pad] * (-len(vals) % VL)
    return [vals[i:i + VL] for i in range(0, len(vals), VL)]


def flat(vs, n): return [x for v in vs for x in v][:n]



EXT = "bf16" in CJ.get("extensions", [])
BF = "bf16"
BFV = lambda lo, hi: lo | (hi << 8)
mxm_bf = lambda op, s=0, mode=0: mxm(op, 0, s, mode, 0, 1)      # the weight type 4: bits 21..20 zero, bit 12 set


def illegal(word, queue=88):
    o, _ = run([(10, queue, 1, word, 0)])
    return any("OTS-E-DEC-003" in l for l in diags(o)), diags(o)


# the 16-bit values of every class and a random remainder
def bf_values(rnd, n):
    special = [0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0, 0x7F81, 0x4000, 0x3F81, 0x3B80, 0x3FC0]
    return special + [rnd.getrandbits(16) for _ in range(n)]


def matvec(rows, x, wtype_h=1, abc_word=None):
    """The PoC: rows (len R <= 160) x VL bf16 values; installs them with IW from streams in groups of 8 rows (16 stream
    values), one ABC and one ACC f32. Returns the R products (f32 bits)."""
    R = len(rows)
    assert R % 8 == 0
    sram = {}; recs = []; T = 200
    for g in range(R // 8):
        tg = T + 2 * g
        for i in range(16):
            r, hi = 8 * g + i // 2, i % 2
            vals = bytes(((rows[r][k] >> (8 * hi)) & 0xFF) for k in range(VL))
            feed(recs, sram, 44 + 4 * (i % 11) + (i // 11), 3 + 16 * g + i, vals, i, "W", 43, tg)
        recs.append((tg, 104, 1, mxm(1, 0, 32 | 0, 1, 0, wtype_h), 0))
    T2 = T + 2 * (R // 8) + 2
    feed(recs, sram, 60, 5, bytes(v & 0xFF for v in x), 24, "W", 43, T2)
    feed(recs, sram, 61, 5, bytes(v >> 8 for v in x), 25, "W", 43, T2)
    recs.append((T2, 105, 1, abc_word if abc_word is not None else mxm_bf(2, 32 | 24), 0))
    T3 = T2 + D_ABC
    recs.append((T3, 106, 1, mxm(3, 3, 8, 0, 1), 0))                          # acc.f32 E:8, replace, accumulator 1
    for b in range(4):
        drain(recs, 4 * b, 9, 8 + b, "E", 2, T3 + D_ACC)
    o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
    words = [wd.get((4 * b, 9), bytes(VL)) for b in range(4)]
    return [int.from_bytes(bytes(words[b][l] for b in range(4)), "little") for l in range(R)], diags(o)


def conv(st, dt, vals, name=""):
    vs = chunks(vals)
    wi, wo = nx.WIDTH[st], nx.WIDTH[dt]
    out, d = vxm_vectors(lambda k, s, dst: vxm_op(34, st, dst, s[0], mode=TY[dt] & 15, dz=TY[dt] >> 4), [(wi, vs)], wo, len(vs))
    got = flat(out, len(vals)); exp = [nx.k_convert_any(st, dt, x) for x in vals]
    bad = [(i, hex(vals[i]), hex(got[i]), hex(exp[i])) for i in range(len(vals)) if got[i] != exp[i]]
    check(f"convert.{st}.{dt}: {name}", not bad and not d, (bad[:3], d[:2]))


rnd = random.Random(1633)
if not EXT:
    # 1. the gate: every bf16 word is refused, the base words still decode
    for nm, w in (("add.bf16", vxm_op(0, BF, 24, 0, 8)), ("mul.bf16", vxm_op(6, BF, 24, 0, 8)), ("neg.bf16", vxm_op(15, BF, 24, 0)),
                  ("cmp.lt.bf16", vxm_op(19, BF, 24, 0, 8, mode=2)), ("reduce.add.bf16", vxm_op(33, BF, 24, 0)),
                  ("mask.bf16", vxm_op(17, BF, 24, 0, 8)), ("convert.f32.bf16", vxm_op(34, "f32", 24, 0, mode=0, dz=1)),
                  ("convert.bf16.f32", vxm_op(34, BF, 24, 0, mode=3))):
        ill, d = illegal(w)
        check(f"{nm} is OTS-E-DEC-003 without the extension", ill, d)
    for nm, w in (("lw.bf16", mxm_bf(0, 32 | 0)), ("iw.bf16", mxm_bf(1)), ("iws.bf16", mxm_bf(1, 32 | 0, 1)), ("abc.bf16", mxm_bf(2, 32 | 24))):
        ill, d = illegal(w, 105)
        check(f"{nm} is OTS-E-DEC-003 without the extension", ill, d)
    for nm, w in (("add.f16", vxm_op(0, "f16", 24, 0, 8)), ("convert.f32.f16", vxm_op(34, "f32", 24, 0, mode=1))):
        ill, d = illegal(w)
        check(f"{nm} still decodes", not ill, d)
else:
    # 1. binary operations and comparisons on a grid and on random pairs
    V = bf_values(rnd, 700)
    PR = [(a, b) for a in V[:19] for b in V[:19]] + [(rnd.getrandbits(16), rnd.getrandbits(16)) for _ in range(2000)] + [(a, b) for a, b in zip(V[19:], V[:0:-1])]
    A_V = chunks([a for a, _ in PR]); B_V = chunks([b for _, b in PR])
    for nm, op in (("add", 0), ("sub", 3), ("mul", 6), ("max", 9), ("min", 10)):
        out, d = vxm_vectors(lambda k, s, dst, op=op: vxm_op(op, BF, dst, s[0], s[1]), [(2, A_V), (2, B_V)], 2, len(A_V))
        got = flat(out, len(PR)); exp = [nx.k_binary(op, BF, a, b) for a, b in PR]
        bad = [(i, hex(PR[i][0]), hex(PR[i][1]), hex(got[i]), hex(exp[i])) for i in range(len(PR)) if got[i] != exp[i]]
        check(f"{nm}.bf16: {len(PR)} operand pairs", not bad and not d, (bad[:3], d[:2]))
    for mode, mn in enumerate(nx.CMP_MODES):
        out, d = vxm_vectors(lambda k, s, dst, mode=mode: vxm_op(19, BF, dst, s[0], s[1], mode=mode), [(2, A_V), (2, B_V)], 2, len(A_V))
        got = flat(out, len(PR)); exp = [nx.k_cmp(mode, BF, a, b) for a, b in PR]
        bad = [(i, hex(got[i]), hex(exp[i])) for i in range(len(PR)) if got[i] != exp[i]]
        check(f"cmp.{mn}.bf16: {len(PR)} operand pairs", not bad and not d, (bad[:3], d[:2]))
    # 2. unary operations on every value, Mask and Select, SG4, reduce
    V64 = chunks(list(range(65536)))
    for nm, op in (("neg", 15), ("abs", 16), ("relu", 20)):
        out, d = vxm_vectors(lambda k, s, dst, op=op: vxm_op(op, BF, dst, s[0]), [(2, V64)], 2, len(V64))
        got = flat(out, 65536); exp = [nx.k_unary(op, BF, a) for a in range(65536)]
        check(f"{nm}.bf16: all 65536 values", got == exp and not d, ([(i, hex(got[i]), hex(exp[i])) for i in range(65536) if got[i] != exp[i]][:3], d[:2]))
    n3 = 12 * VL
    X, Y, M = chunks([rnd.getrandbits(16) for _ in range(n3)]), chunks([rnd.getrandbits(16) for _ in range(n3)]), chunks([rnd.choice([0, 0xFFFF, 1]) for _ in range(n3)])
    out, d = vxm_vectors(lambda k, s, dst: vxm_op(17, BF, dst, s[0], b=s[1]), [(2, X), (2, M)], 2, len(X))
    got = flat(out, n3); xs, ms = flat(X, n3), flat(M, n3)
    check("mask.bf16", got == [x if m else 0 for x, m in zip(xs, ms)] and not d, d[:2])
    out, d = vxm_vectors(lambda k, s, dst: vxm_op(18, BF, dst, s[0], b=s[1], m=s[2]), [(2, X), (2, Y), (2, M)], 2, len(X))
    got = flat(out, n3); ys = flat(Y, n3)
    check("select.bf16", got == [x if m else y for x, y, m in zip(xs, ys, ms)] and not d, d[:2])
    for nm, op in (("add", 0), ("mul", 6)):
        a2 = [[rnd.getrandbits(32) for _ in range(VL)]]; b2 = [[rnd.getrandbits(32) for _ in range(VL)]]
        out, d = vxm_vectors(lambda k, s, dst, op=op: vxm_op(op, BF, dst, s[0], s[1], g=1), [(4, a2), (4, b2)], 4, 1)
        exp = [sum(nx.k_binary(op, BF, (a2[0][l] >> (16 * i)) & 0xFFFF, (b2[0][l] >> (16 * i)) & 0xFFFF) << (16 * i) for i in range(2)) for l in range(VL)]
        check(f"{nm}.bf16.sg4: two bf16 operations per lane", out[0] == exp and not d, (d[:2], hex(out[0][0]), hex(exp[0])))
    vals = [rnd.choice([0x0001, 0x3F80, 0x3B80, 0x4000, 0xBF80, 0x7F7F, 0x3F00 | rnd.randrange(128)]) for _ in range(VL)]
    out, d = vxm_vectors(lambda k, s, dst: vxm_op(33, BF, dst, s[0], mode=0), [(2, [vals])], 2, 1)
    exp = [0] * VL
    for g in range(VL // 16):
        acc = vals[16 * g]
        for j in range(1, 16):
            acc = nx.k_binary(0, BF, acc, vals[16 * g + j])
        exp[16 * g] = acc
    check("reduce.add.bf16: ascending lane order, one rounding per step", out[0] == exp and not d, (d[:2], out[0][:2], exp[:2]))
    # 3. conversions
    r32 = random.Random(16)
    F32_RND = [r32.getrandbits(32) for _ in range(3000)] + [(h << 16) | lo for h in range(0, 65536, 97) for lo in (0, 0x8000, 0x8001, 0x7FFF)]
    conv("f32", BF, F32_RND, "random and halfway binary32 values")
    conv("f16", BF, list(range(65536)), "all 2^16 binary16 values")
    for dt in ("f32", "f16", "i8", "u8", "i32", "b16", "e4m3", "e5m2"):
        conv(BF, dt, list(range(65536)), "all 2^16 values")
    conv("i8", BF, list(range(256)), "all 256 values")
    conv("i32", BF, [r32.getrandbits(32) for _ in range(640)] + [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 255, 256, 257], "random int32")
    # 4. the matrix product: the proof-of-concept W (64 x 320) . x
    rm = random.Random(633)
    def rand_bf(lo=-30, hi=30):
        e = rm.randint(lo, hi)
        return (rm.getrandbits(1) << 15) | ((127 + e) << 7) | rm.getrandbits(7)
    rows = [[rand_bf() for _ in range(VL)] for _ in range(64)]
    x = [rand_bf() for _ in range(VL)]
    got, d = matvec(rows, x)
    exp = [nx.dot_bf16(rows[r], x) for r in range(64)]
    check("bf16 matvec W (64 x 320) . x: the exact sum rounded once to f32 (the whole range of exponents -30..30)", got == exp and not d,
          (d[:2], [(r, hex(a), hex(b)) for r, (a, b) in enumerate(zip(got, exp)) if a != b][:3]))
    rows = [[rand_bf(-126, 127) & 0xFFFF for _ in range(VL)] for _ in range(16)]
    x = [rand_bf(-126, 127) for _ in range(VL)]
    got, d = matvec(rows, x)
    exp = [nx.dot_bf16(rows[r], x) for r in range(16)]
    check("bf16 matvec over the whole exponent range (overflow to infinity, underflow)", got == exp and not d, (d[:2], [(r, hex(a), hex(b)) for r, (a, b) in enumerate(zip(got, exp)) if a != b][:3]))
    rows = [[0] * VL for _ in range(16)]
    for r in range(16):
        rows[r][0], rows[r][1], rows[r][2] = 0x7F00, 0x3F80, 0xFF00        # 2^127, 1, -2^127
    x = [0] * VL
    x[0], x[1], x[2] = 0x3F80, 0x3F80, 0x3F80
    got, d = matvec(rows, x)
    check("bf16 matvec: cancellation 2^127 + 1 - 2^127 = 1 (a float32 accumulation gives 0)", got == [0x3F800000] * 16 and not d, (d[:2], [hex(v) for v in got]))
    rows = [[rand_bf() for _ in range(VL)] for _ in range(16)]
    x = [rand_bf() for _ in range(VL)]
    rows[0][5], rows[1][5], rows[2][5], x[5] = 0x7FC0, 0xFF80, 0x7F80, 0x0000
    for r in range(3, 8):
        rows[r] = [0x8000 if k % 2 else 0 for k in range(VL)]
    got, d = matvec(rows, x)
    exp = [nx.dot_bf16(rows[r], x) for r in range(16)]
    check("bf16 matvec: NaN, infinity times zero, zeros of both signs", got == exp and not d, (d[:2], [(r, hex(a), hex(b)) for r, (a, b) in enumerate(zip(got, exp)) if a != b][:3]))
    # the type rules of the matrix unit
    got, d = matvec([[0] * VL] * 16, [0] * VL, abc_word=mxm(2, 1, 32 | 24))
    check("abc.f16 on bf16 weights is OTS-E-RES-001", any("OTS-E-RES-001" in l for l in d), d)
    got, d = matvec([[0] * VL] * 16, [0] * VL, abc_word=mxm(2, 0, 32 | 24))
    check("abc.i8 on bf16 weights is OTS-E-RES-001", any("OTS-E-RES-001" in l for l in d), d)
    got, d = matvec([[0] * VL] * 16, [0] * VL, wtype_h=0, abc_word=mxm_bf(2, 32 | 24))
    check("abc.bf16 on i8 weights is OTS-E-RES-001", any("OTS-E-RES-001" in l for l in d), d)
    # 5. what stays illegal with the extension
    base = vxm_op(0, BF, 24, 0, 8)
    for nm, w in (("code 17", base | (1 << 52)), ("code 18", base | (2 << 52)), ("code 19", base | (3 << 52)), ("code 20", base | (1 << 22)), ("code 31", base | (3 << 52) | (3 << 22))):
        ill, d = illegal(w)
        check(f"add with type {nm} is OTS-E-DEC-003", ill, d)
    for nm, w in (("and.bf16", vxm_op(11, BF, 24, 0, 8)), ("or.bf16", vxm_op(12, BF, 24, 0, 8)), ("not.bf16", vxm_op(14, BF, 24, 0)),
                  ("tanh.bf16", vxm_op(21, BF, 24, 0)), ("exp.bf16", vxm_op(22, BF, 24, 0)), ("add_sat.bf16", vxm_op(1, BF, 24, 0, 8)),
                  ("mul_mod.bf16", vxm_op(8, BF, 24, 0, 8)), ("cmp mode 6 on bf16", vxm_op(19, BF, 24, 0, 8, mode=6)),
                  ("convert.i8 to code 17", vxm_op(34, "i8", 24, 0, mode=1, dz=1)), ("conv_i8_f32 with type bf16", vxm_op(25, BF, 24, 0))):
        ill, d = illegal(w)
        check(f"{nm} is OTS-E-DEC-003", ill, d)
    for nm, w in (("lw type 5", mxm(0, 1, 32 | 0, 0, 0, 1)), ("iw type 6", mxm(1, 2, 0, 0, 0, 1)), ("abc.bf16 with mode bit 13", mxm_bf(2, 32 | 24, 1)), ("lw.bf16 with a stream off a 16 boundary", mxm_bf(0, 32 | 1))):
        ill, d = illegal(w, 105)
        check(f"{nm} is OTS-E-DEC-003", ill, d)
    for nm, w in (("lw.bf16", mxm_bf(0, 32 | 0)), ("iw.bf16", mxm_bf(1)), ("iws.bf16", mxm_bf(1, 32 | 0, 1)), ("abc.bf16", mxm_bf(2, 32 | 24))):
        ill, d = illegal(w, 105)
        check(f"{nm} decodes (no OTS-E-DEC-003)", not ill, d)
    for nm, w in (("add.bf16", vxm_op(0, BF, 24, 0, 8)), ("convert.f32.bf16", vxm_op(34, "f32", 24, 0, mode=0, dz=1)), ("convert.bf16.f32", vxm_op(34, BF, 24, 0, mode=3))):
        ill, d = illegal(w)
        check(f"{nm} decodes (no OTS-E-DEC-003)", not ill, d)
print("bf16 directed (" + ("extension declared" if EXT else "extension not declared") + "): " + ("all passed" if ok else "FAILED"))
sys.exit(0 if ok else 1)
