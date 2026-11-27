"""Directed end-to-end tests of the FP8 element types e4m3 and e5m2 on the Sail emulator (tsp_sim)
(, microarchitecture.md sec. 5 "FP8 element types" and sec. 6 "FP8 matrix products").

Every program goes through the whole dataflow (MEM read -> stream -> VXM or MXM -> stream -> MEM write)
and the results are compared with the oracle (test/oracle/numerics.py: exact fractions, no Sail):
  * every one of the 256 x 256 operand pairs of both formats through Add, Sub, Mul, Max, Min and the six
    comparisons (one program per operation: 205 vectors of 320 lanes), and the SG4 form;
  * all 2^16 binary16 values converted to both formats, every fp8 value converted to f16, f32, i8, u8,
    i32, b8 and the other format, i8 and u8 converted to fp8, random and halfway binary32 values converted to fp8;
  * Neg, Abs and ReLU on every value, Mask and Select;
  * an fp8 matrix product (IW from streams, ABC, ACC f32) against the exact sum, in both formats,
    with the cancellation that a float32 accumulation would lose, and the special values;
  * the codes and modes that stay illegal: type codes 13..15, and.e4m3, tanh.e4m3, add_sat.e5m2, ABC type 0 with
    mode bit 13, and the type rules of the matrix unit.
usage: fp8.py <tsp_sim> <config.json>"""
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


TY = {t: i for i, t in enumerate(nx.TYPES)}
def VXM(op, ty, d, a, b=0, m=0, mode=0, tx=0, bits=0):
    return (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22) | bits
def vxm_op(op, t, d, a, b=0, m=0, mode=0, g=0):
    c = TY[t]
    return VXM(op, c & 3, d, a, b, m, mode, c >> 2, g << 18)
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)


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


FMTS = ("e4m3", "e5m2")
PAIRS = [(a, b) for a in range(256) for b in range(256)]
A_V = chunks([a for a, _ in PAIRS]); B_V = chunks([b for _, b in PAIRS])

# helpers of sections 6 and 7 (also used by tools/gen_model_vectors.py::fp8_programs, which runs the definitions above section 1)
def mxm_run(wfmt, rows, x, abc_word=None, wtype=3, drain_acc=True):
    """16 rows of weights (lists of VL bytes), activation x (VL bytes); returns the 16 products (f32 bits) or the output."""
    sram = {}; recs = []; T = 200
    for r in range(16):
        feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes(rows[r]), r, "W", 43, T)
    recs.append((T, 104, 1, mxm(1, wtype, 32 | 0, 1), 0))                     # iw.<type> W:0 (from streams)
    T2 = T + 2
    feed(recs, sram, 60, 5, bytes(x), 24, "W", 43, T2)
    recs.append((T2, 105, 1, abc_word if abc_word is not None else mxm(2, 3, 32 | 24, wfmt), 0))
    T3 = T2 + D_ABC
    recs.append((T3, 106, 1, mxm(3, 3, 8, 0, 1), 0))                          # acc.f32 E:8, replace, accumulator 1
    for b in range(4):
        drain(recs, 4 * b, 9, 8 + b, "E", 2, T3 + D_ACC)
    o, wd = run(recs, sram, [(4 * b, 9) for b in range(4)])
    words = [wd.get((4 * b, 9), bytes(VL)) for b in range(4)]
    return [int.from_bytes(bytes(words[b][l] for b in range(4)), "little") for l in range(16)], diags(o)

def illegal(word, queue=88):
    o, _ = run([(10, queue, 1, word, 0)])
    return any("OTS-E-DEC-003" in l for l in diags(o)), diags(o)
# 1. binary operations and comparisons: every operand pair
BIN = [("add", 0), ("sub", 3), ("mul", 6), ("max", 9), ("min", 10)]
for t in FMTS:
    for nm, op in BIN:
        out, d = vxm_vectors(lambda k, s, dst, op=op, t=t: vxm_op(op, t, dst, s[0], s[1]), [(1, A_V), (1, B_V)], 1, len(A_V))
        exp = [nx.k_binary(op, t, a, b) for a, b in PAIRS]
        got = flat(out, len(PAIRS))
        bad = [(i, hex(got[i]), hex(exp[i])) for i in range(len(PAIRS)) if got[i] != exp[i]]
        check(f"{nm}.{t}: all 65536 operand pairs", not bad and not d, (bad[:3], d[:2]))
    for mode, mn in enumerate(nx.CMP_MODES):
        out, d = vxm_vectors(lambda k, s, dst, mode=mode, t=t: vxm_op(19, t, dst, s[0], s[1], mode=mode), [(1, A_V), (1, B_V)], 1, len(A_V))
        exp = [nx.k_cmp(mode, t, a, b) for a, b in PAIRS]
        got = flat(out, len(PAIRS))
        bad = [(i, hex(got[i]), hex(exp[i])) for i in range(len(PAIRS)) if got[i] != exp[i]]
        check(f"cmp.{mn}.{t}: all 65536 operand pairs", not bad and not d, (bad[:3], d[:2]))

# 2. unary operations on every value, Mask and Select
V256 = chunks(list(range(256)))
assert 1280 % VL == 0
NV = 1280 // VL          # vectors that hold 1280 operand pairs (4 of 320 lanes)
for t in FMTS:
    for nm, op in (("neg", 15), ("abs", 16), ("relu", 20)):
        out, d = vxm_vectors(lambda k, s, dst, op=op, t=t: vxm_op(op, t, dst, s[0]), [(1, V256)], 1, len(V256))
        got = flat(out, 256)
        exp = [nx.k_unary(op, t, a) for a in range(256)]
        check(f"{nm}.{t}: all 256 values", got == exp and not d, ([(i, hex(got[i]), hex(exp[i])) for i in range(256) if got[i] != exp[i]][:3], d[:2]))
    out, d = vxm_vectors(lambda k, s, dst, t=t: vxm_op(17, t, dst, s[0], b=s[1]), [(1, A_V[:NV]), (1, B_V[:NV])], 1, NV)
    got = flat(out, 1280); exp = [a if b & 0xFF else 0 for a, b in PAIRS[:1280]]
    check(f"mask.{t}", got == exp and not d, d[:2])
    out, d = vxm_vectors(lambda k, s, dst, t=t: vxm_op(18, t, dst, s[0], b=s[1], m=s[2]), [(1, A_V[:NV]), (1, B_V[:NV]), (1, A_V[NV:2 * NV])], 1, NV)
    got = flat(out, 1280); exp = [PAIRS[i][0] if PAIRS[1280 + i][0] else PAIRS[i][1] for i in range(1280)]
    check(f"select.{t}", got == exp and not d, d[:2])

# 3. SG4: four independent operations per lane on a group of four streams
rnd = random.Random(86)
for t in FMTS:
    for nm, op in (("add", 0), ("mul", 6)):
        a4 = [[rnd.getrandbits(32) for _ in range(VL)]]; b4 = [[rnd.getrandbits(32) for _ in range(VL)]]
        out, d = vxm_vectors(lambda k, s, dst, op=op, t=t: vxm_op(op, t, dst, s[0], s[1], g=1), [(4, a4), (4, b4)], 4, 1)
        exp = [sum(nx.k_binary(op, t, (a4[0][l] >> (8 * i)) & 255, (b4[0][l] >> (8 * i)) & 255) << (8 * i) for i in range(4)) for l in range(VL)]
        check(f"{nm}.{t}.sg4: four fp8 operations per lane", out[0] == exp and not d, (d[:2], hex(out[0][0]), hex(exp[0])))

# 4. conversions
def conv(st, dt, vals, name):
    vs = chunks(vals)
    wi, wo = nx.WIDTH[st], nx.WIDTH[dt]
    out, d = vxm_vectors(lambda k, s, dst: vxm_op(34, st, dst, s[0], mode=TY[dt]), [(wi, vs)], wo, len(vs))
    got = flat(out, len(vals)); exp = [nx.k_convert_any(st, dt, x) for x in vals]
    bad = [(i, hex(vals[i]), hex(got[i]), hex(exp[i])) for i in range(len(vals)) if got[i] != exp[i]]
    check(f"convert.{st}.{dt}: {name}", not bad and not d, (bad[:3], d[:2]))
r32 = random.Random(1862)
F32_RND = [r32.getrandbits(32) for _ in range(3000)] + [(h << 16) | lo for h in range(0, 65536, 97) for lo in (0, 0x8000, 0x8001, 0x7FFF)]
for t in FMTS:
    conv("f16", t, list(range(65536)), "all 2^16 binary16 values")
    conv("f32", t, F32_RND, "random and halfway binary32 values")
    for dt in ("f16", "f32", "i8", "u8", "i32", "b8", "e5m2" if t == "e4m3" else "e4m3"):
        conv(t, dt, list(range(256)), "all 256 values")
    for st in ("i8", "u8"):
        conv(st, t, list(range(256)), "all 256 values")
    conv("i32", t, [r32.getrandbits(32) for _ in range(640)] + [0, 1, 0xFFFFFFFF, 448, 449, 464, 465, 57344, 61440, 0x7FFFFFFF, 0x80000000], "random int32")

# 5. the reduce of FP8 values: ascending lanes, rounded in the format at each step
for t in FMTS:
    vals = [rnd.choice([0x01, 0x08, 0x38, 0x30, 0x7B if t == "e5m2" else 0x7E, 0x80 | 0x38, 0xC0 | rnd.randrange(8)]) for _ in range(VL)]
    out, d = vxm_vectors(lambda k, s, dst, t=t: vxm_op(33, t, dst, s[0], mode=0), [(1, [vals])], 1, 1)
    exp = [0] * VL
    for g in range(VL // 16):
        acc = vals[16 * g]
        for j in range(1, 16):
            acc = nx.k_binary(0, t, acc, vals[16 * g + j])
        exp[16 * g] = acc
    check(f"reduce.add.{t}: ascending lane order, one rounding per step", out[0] == exp and not d, (d[:2], out[0][:2], exp[:2]))

# The assembly guide's tie at 2049, and the reassociated control.
half_inputs = [[0x6800, 0x3c00, 0xe800] + [0] * (VL - 3),
               [0x6800, 0xe800, 0x3c00] + [0] * (VL - 3)]
out, d = vxm_vectors(lambda k, s, dst: vxm_op(33, "f16", dst, s[0]), [(2, half_inputs)], 2, 2)
check("documented f16 cancellation orders yield +0 and 1",
      not d and out[0] == [0] * VL and out[1] == [0x3c00] + [0] * (VL-1), (out[0][:2], out[1][:2], d))

# 6. the matrix unit: fp8 weights (IW from streams), fp8 ABC, ACC f32
def finite_codes(t):
    f = nx.fmt(t)
    return [c for c in range(256) if nx.fclass(f, c)[0] in ("fin", "zero")]

rm = random.Random(8686)
for wfmt, t in ((0, "e4m3"), (1, "e5m2")):
    f = nx.fmt(t)
    fin = finite_codes(t)
    for case in range(3):
        if case == 0:       # random finite values
            rows = [[rm.choice(fin) for _ in range(VL)] for _ in range(16)]
            x = [rm.choice(fin) for _ in range(VL)]
        elif case == 1:     # cancellation: float32 accumulation of 448^2 - 448^2 + 2^-9 would lose the last term
            big, one = (0x7E, 0x38) if t == "e4m3" else (0x7B, 0x3C)
            rows = [[0] * VL for _ in range(16)]
            for r in range(16):
                rows[r][0], rows[r][1], rows[r][2] = big, big, 0x01
            x = [0] * VL
            x[0], x[1], x[2] = big, big | 0x80, one
        else:               # specials: NaN, an infinity (e5m2), zeros of both signs
            rows = [[rm.choice(fin) for _ in range(VL)] for _ in range(16)]
            x = [rm.choice(fin) for _ in range(VL)]
            rows[0][5] = 0x7F if t == "e4m3" else 0x7C
            rows[1][5] = 0xFF if t == "e4m3" else 0xFC
            rows[2][5], x[5] = (0x7C, 0) if t == "e5m2" else (0x7F, 0)
            for r in range(3, 8):
                rows[r] = [0x80 if k % 2 else 0 for k in range(VL)]
            x[5] = 0 if t == "e5m2" else x[5]
        got, d = mxm_run(wfmt, rows, x)
        exp = [nx.dot_fp8(f, rows[r], x) for r in range(16)]
        check(f"fp8 matrix product {t} case {case}: the exact sum rounded once to f32 (16 rows)", got == exp and not d, (d[:2], [(hex(a), hex(b)) for a, b in zip(got, exp) if a != b][:3]))
    # a row beyond the installed ones and the weights' type rule
    got, d = mxm_run(wfmt, [[0] * VL] * 16, [0] * VL, abc_word=mxm(2, 0, 32 | 24))
    check(f"abc.i8 on fp8 weights is OTS-E-RES-001 ({t} installed)", any("OTS-E-RES-001" in l for l in d), d)
    got, d = mxm_run(wfmt, [[0] * VL] * 16, [0] * VL, wtype=0)
    check(f"abc.{t} on i8 weights is OTS-E-RES-001", any("OTS-E-RES-001" in l for l in d), d)

# 7. what stays illegal
base = vxm_op(0, "i8", 24, 0, 8)
for code in (13, 14, 15):
    ill, d = illegal((base & ~((3 << 52) | (3 << 22))) | ((code & 3) << 52) | ((code >> 2) << 22))
    check(f"add with type code {code} is OTS-E-DEC-003", ill, d)
for nm, w in (("and.e4m3", vxm_op(11, "e4m3", 24, 0, 8)), ("or.e5m2", vxm_op(12, "e5m2", 24, 0, 8)), ("not.e4m3", vxm_op(14, "e4m3", 24, 0)),
              ("tanh.e4m3", vxm_op(21, "e4m3", 24, 0)), ("exp.e5m2", vxm_op(22, "e5m2", 24, 0)), ("add_sat.e5m2", vxm_op(1, "e5m2", 24, 0, 8)),
              ("mul_mod.e4m3", vxm_op(8, "e4m3", 24, 0, 8)), ("cmp mode 6 on e4m3", vxm_op(19, "e4m3", 24, 0, 8, mode=6)),
              ("reduce mode 3 on e4m3", vxm_op(33, "e4m3", 24, 0, mode=3)),
              ("convert.i8 to code 13", vxm_op(34, "i8", 24, 0, mode=13)), ("convert.i8 to code 15", vxm_op(34, "i8", 24, 0, mode=15)),
              ("conv_i8_f32 with type e4m3 (the type must equal the source)", vxm_op(25, "e4m3", 24, 0))):
    ill, d = illegal(w)
    check(f"{nm} is OTS-E-DEC-003", ill, d)
for nm, w in (("abc type 0 with mode bit 13", mxm(2, 0, 32 | 24, 1)), ("abc type 1 with mode bit 13", mxm(2, 1, 32 | 24, 1)), ("abc type 2 with mode bit 13", mxm(2, 2, 32 | 24, 1)),
              ("lw.fp8 with mode bit 13", mxm(0, 3, 32 | 0, 1)), ("iw.fp8 (buffer form) with a stream", mxm(1, 3, 32 | 1, 0))):
    ill, d = illegal(w, 105)
    check(f"{nm} is OTS-E-DEC-003", ill, d)
for nm, w in (("lw.fp8", mxm(0, 3, 32 | 0)), ("iw.fp8", mxm(1, 3)), ("abc.e4m3", mxm(2, 3, 32 | 24, 0)), ("abc.e5m2", mxm(2, 3, 32 | 24, 1))):
    ill, d = illegal(w, 105)
    check(f"{nm} decodes (no OTS-E-DEC-003)", not ill, d)
print("fp8 directed: " + ("all passed" if ok else "FAILED"))
sys.exit(0 if ok else 1)
