"""Directed end-to-end tests of the optional extensions wide_acc (ACC type x128) and block_fp8 (the MXM type bfp8)
(microarchitecture.md sec. 6.1) on the Sail emulator or the ISS: usage ext_types.py <tsp_sim|tsp-iss> <config.json>.

On a profile that declares the extensions (config/test/mini_ext.json) every program goes through the whole dataflow
(MEM read -> stream -> MXM -> stream -> MEM write) and the results are compared with the oracle (test/oracle: exact
fractions, no Sail):
  * ACC x128 after an f16 ABC: replace and three accumulations of 2^-25 on 1.0 give 1 + 2^-23 (one rounding at the read-out)
    where the f32 accumulation of the same products gives 1.0;
  * bfp8 matrix products (IWS, ABC, ACC f32 and ACC x128) against the exact scaled sum, on random data and scales;
  * ACC x128 after an integer ABC is OTS-E-OPR-001.
On a profile that does not declare them (ref, mini) every encoding of an extension is OTS-E-DEC-003."""
import json, os, random, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "tools"))
sys.path.insert(0, os.path.join(here, "..", "oracle"))
import tsp_as
from tsp_as import lower_records, q_lines
import numerics as nx
sys.path.insert(0, os.path.join(here, ".."))
from oracle import machine as mc

SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
EXT = {k: k in CJ["extensions"] for k in ("wide_acc", "block_fp8")}
DM = TM["mem"]["d_func"]
D_ABC, D_ACC = TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"]
VL = 16 * CJ["geometry"]["superlanes"]
SLICES = {106: [0, 4, 8, 12], 105: [16, 20, 24, 28]}      # the MEM slices that receive the four byte planes of the ACCs of a queue
ROWS = 8      # rows of a pair-type matrix that one IW from streams installs (16 stream values)


def run(recs, sram={}, dumps=[]):
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    lines += q_lines(lower_records(recs)) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    r = subprocess.run([SIM, CFG, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=900)
    if r.returncode or r.stderr.strip():
        raise SystemExit(f"simulator failed ({r.returncode}): {r.stderr.strip()[:300]}")
    o = r.stdout
    words = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:]) for t in (l.split() for l in o.splitlines()) if t and t[0] == 'W'}
    return o, words


def diags(o): return [l for l in o.splitlines() if "diag" in l]


def oracle_words(recs, sram):
    """The SRAM words the oracle machine (test/oracle, written from the documents, not from the model) writes for the program."""
    images = tsp_as.lower([(c, 0xffff if q == 144 else q, w0) for (c, q, wc, w0, w1) in recs])
    ev = mc.Machine(CJ).run(dict(sorted(images.items(), key=lambda kv: (kv[0] == 0xffff, kv[0]))), dict(sram))
    return {(e[2], e[3]): bytes(e[4]) for e in ev if e[0] == 'sram'}, [e for e in ev if e[0] == 'diag']


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


def mxm(op, t, s=0, mode=0, acc=0, tx=0):
    """An MXM word: type bits 21..20, type extension bit 12 (LW, IW, IWS, ABC) or the accumulator (ACC)."""
    return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | ((acc if op == 3 else tx) << 12)


def illegal(word, queue=104):
    o, _ = run([(10, queue, 1, word, 0)])
    return any("OTS-E-DEC-003" in l for l in diags(o)), diags(o)


def stripes(vals, n=4):
    return [bytes((v >> (8 * b)) & 0xff for v in vals) for b in range(n)]


def words_at(wd, queue, addr):
    planes = [wd.get((s, addr), bytes(VL)) for s in SLICES[queue]]
    return [int.from_bytes(bytes(planes[b][l] for b in range(4)), "little") for l in range(ROWS)]


# ---------------------------------------------------------------------------------------------------------------
# The decoder gate
WORDS = {
    "acc.x128": mxm(3, 0, 8, 0, 0),
    "lw.bfp8": mxm(0, 1, 32 | 0, 0, tx=1),
    "iw.bfp8": mxm(1, 1, 0, 0, tx=1),
    "iw.bfp8 from streams": mxm(1, 1, 32 | 0, 1, tx=1),
    "abc.bfp8": mxm(2, 1, 32 | 24, 0, tx=1),
}
for nm, w in WORDS.items():
    ill, d = illegal(w)
    declared = EXT["wide_acc"] if nm == "acc.x128" else EXT["block_fp8"]
    check(f"{nm} is {'accepted' if declared else 'OTS-E-DEC-003'} on a profile that {'declares' if declared else 'does not declare'} its extension",
          ill != declared, d)
# Reserved codes, plus type 4 whose extension-bit form is bf16 when declared.
for nm, w in (("acc type 1", mxm(3, 1, 8, 0, 0)), ("abc with the extension bit and type i8", mxm(2, 0, 32 | 24, 0, tx=1)),
              ("abc with the extension bit and type u8", mxm(2, 2, 32 | 24, 0, tx=1)), ("abc8 with the extension bit", mxm(2, 3, 32 | 24, 0, tx=1))):
    ill, d = illegal(w)
    # ABC type 0 plus its extension bit is bf16 (type 4), legal on the combined profile.
    want_illegal = not (nm == "abc with the extension bit and type i8" and "bf16" in CJ["extensions"])
    check(f"{nm} is {'OTS-E-DEC-003' if want_illegal else 'accepted as bf16'}", ill == want_illegal, d)

if not (EXT["wide_acc"] and EXT["block_fp8"]):
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------------------------------------------
# One matrix program: ROWS rows of a pair type installed from streams (the stream pair of row r is 2r and 2r + 1), then
# a list of activation pairs (one ABC each, back to back on queue 105), then a list of ACCs (queue, type, mode, accumulator,
# cycle offset after the first ABC's commit), each draining its result to slices 4b at its own address.
def matrix_program(wtype, tx, weight_rows, abcs, accs, abc_type=None):
    """weight_rows: ROWS pairs (first bytes, second bytes) of VL bytes; abcs: pairs of VL bytes; accs: (queue, acc type, mode, acc,
    offset, address)."""
    sram = {}; recs = []; T = 200
    for r in range(ROWS):
        feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + 2 * r, bytes(weight_rows[r][0]), 2 * r, "W", 43, T)
        feed(recs, sram, 44 + 4 * ((r + 8) % 11) + ((r + 8) // 11), 3 + 2 * r + 1, bytes(weight_rows[r][1]), 2 * r + 1, "W", 43, T)
    recs.append((T, 104, 1, mxm(1, wtype, 32 | 0, 1, tx=tx), 0))
    t0 = T + 2
    for i, (d0, d1) in enumerate(abcs):
        feed(recs, sram, 60 + 4 * i, 5, bytes(d0), 24 + 2 * i, "W", 43, t0 + i)
        feed(recs, sram, 61 + 4 * i, 6, bytes(d1), 25 + 2 * i, "W", 43, t0 + i)
        recs.append((t0 + i, 105, 1, mxm(2, abc_type if abc_type is not None else wtype, 32 | (24 + 2 * i), 0, tx=tx if abc_type is None else 0), 0))
    base = t0 + D_ABC
    dumps = []
    for queue, at, mode, acc, off, addr in accs:
        dst = 8 if queue == 106 else 16              # two ACCs of one cycle write different streams, to different slices
        recs.append((base + off, queue, 1, mxm(3, at, dst, mode, acc), 0))
        for b in range(4):
            drain(recs, SLICES[queue][b], addr, dst + b, "E", 2, base + off + D_ACC)
            dumps.append((SLICES[queue][b], addr))
    o, wd = run(recs, sram, dumps)
    ow, od = oracle_words(recs, sram)
    same = all(ow.get(k, bytes(VL))[:VL] == wd.get(k, bytes(VL))[:VL] for k in dumps)      # the dump of a word is 320 bytes whatever the lane count
    bad = [(k, wd.get(k, bytes(VL))[:4].hex(), ow.get(k, bytes(VL))[:4].hex()) for k in dumps if ow.get(k, bytes(VL))[:VL] != wd.get(k, bytes(VL))[:VL]]
    check("the oracle machine writes the same result words as the simulator", same and bool(od) == bool(diags(o)), (len(od), len(diags(o)), bad[:3]))
    return o, wd


def vec(vals): return vals + [0] * (VL - len(vals))
def f16_pair(vals):
    return (bytes(v & 0xff for v in vals), bytes(v >> 8 for v in vals))


# 1. ACC x128 after an f16 ABC: the exact sum is kept across accumulations.
ONE, TINY_W, TINY_X = 0x3C00, 0x0400, 0x1000          # 1.0, 2^-14, 2^-11: 2^-14 * 2^-11 = 2^-25
W = [f16_pair(vec([ONE, TINY_W])) for _ in range(ROWS)]
XA = f16_pair(vec([ONE, 0]))
XB = f16_pair(vec([0, TINY_X]))
accs = [(106, 0, 0, 0, 0, 9), (105, 3, 0, 1, 0, 10)]                 # x128 replace on accumulator 0, f32 replace on 1, with ABC a
for i in (1, 2, 3):
    accs += [(106, 0, 1, 0, 2 * i, 9 + 2 * i), (105, 3, 1, 1, 2 * i, 10 + 2 * i)]      # accumulate with ABC b (the newest committed)
o, wd = matrix_program(1, 0, W, [XA, XB], accs)
d = diags(o)
lane = nx.wide_of(nx.exact_f16([ONE, TINY_W] + [0] * (VL - 2), [ONE, 0] + [0] * (VL - 2)))
step = nx.wide_of(nx.exact_f16([ONE, TINY_W] + [0] * (VL - 2), [0, TINY_X] + [0] * (VL - 2)))
seq_x, seq_f = [nx.wide_read(lane)], [nx.round_exact((None, lane[1], False))]
facc = nx.round_exact((None, lane[1], False))
for i in range(3):
    lane = nx.wide_add(lane, step)
    seq_x.append(nx.wide_read(lane))
    facc = nx.fp_add(nx.F32, facc, nx.wide_read(step))
    seq_f.append(facc)
for i in range(4):
    gx = words_at(wd, 106, 9 + 2 * i)
    gf = words_at(wd, 105, 10 + 2 * i)
    check(f"acc.x128 step {i}: the read-out of the exact sum, rounded once", gx == [seq_x[i]] * ROWS and not d, (hex(gx[0]), hex(seq_x[i]), d[:2]))
    check(f"acc.f32 step {i}: the f32 accumulation of the same products", gf == [seq_f[i]] * ROWS and not d, (hex(gf[0]), hex(seq_f[i]), d[:2]))
check("the wide accumulator keeps what every f32 step loses (1 + 3 * 2^-25 rounds to 1 + 2^-23, the f32 chain stays 1.0)",
      seq_x[3] == 0x3F800001 and seq_f[3] == 0x3F800000, (hex(seq_x[3]), hex(seq_f[3])))

# 2. bfp8 matrix products against the exact scaled sum, in ACC f32 and in ACC x128
rm = random.Random(3434)
fin = [c for c in range(256) if nx.fclass(nx.E4M3, c)[0] in ("fin", "zero")]
for case in range(4):
    rows = []
    for r in range(ROWS):
        data = [rm.choice(fin) if rm.random() < 0.9 else 0 for _ in range(VL)]
        scales = [rm.randrange(256) for _ in range(VL)]           # only the low nibble of the first lane of a block counts
        rows.append((data, scales))
    xd = [rm.choice(fin) for _ in range(VL)]
    xs = [rm.randrange(256) for _ in range(VL)]
    if case == 3:            # a NaN element and a zero activation
        rows[0][0][3] = 0x7F; xd[7] = 0
    if case == 2:            # extreme scales
        for r in range(ROWS):
            rows[r][1][0] = 0x07; xs[0] = 0x07
            rows[r][1][16 % VL] = 0x08
    exp_d = [nx.exact_bfp8(rows[r][0], rows[r][1], xd, xs) for r in range(ROWS)]
    o, wd = matrix_program(1, 1, rows, [(xd, xs)], [(106, 3, 0, 0, 0, 9), (105, 0, 0, 1, 0, 10)])
    d = diags(o)
    gf = words_at(wd, 106, 9)
    gx = words_at(wd, 105, 10)
    ef = [nx.round_exact(e) for e in exp_d]
    check(f"bfp8 matrix product case {case}: acc.f32 is the exact scaled sum rounded once", gf == ef and not d, (d[:2], [(hex(a), hex(b)) for a, b in zip(gf, ef) if a != b][:3]))
    check(f"bfp8 matrix product case {case}: acc.x128 replace gives the same read-out", gx == ef and not d, (d[:2], [(hex(a), hex(b)) for a, b in zip(gx, ef) if a != b][:3]))

# 3. the type rules
o, wd = matrix_program(1, 1, [(vec([0]), vec([0]))] * ROWS, [(vec([0]), vec([0]))], [(106, 3, 0, 0, 0, 9)], abc_type=1)
check("abc.f16 on bfp8 weights is OTS-E-RES-001", any("OTS-E-RES-001" in l for l in diags(o)), diags(o))
o, wd = matrix_program(1, 0, [f16_pair(vec([0]))] * ROWS, [(vec([0]), vec([0]))], [(106, 3, 0, 0, 0, 9)], abc_type=1)
check("abc.f16 on f16 weights is accepted (control)", not any("OTS-E-RES-001" in l for l in diags(o)), diags(o))
sram = {}; recs = []; T = 200
for r in range(16):
    feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes(VL), r, "W", 43, T)
recs.append((T, 104, 1, mxm(1, 0, 32 | 0, 1), 0))                    # iw.i8 from streams
feed(recs, sram, 60, 5, bytes(VL), 24, "W", 43, T + 2)
recs.append((T + 2, 105, 1, mxm(2, 0, 32 | 24, 0), 0))              # abc.i8
recs.append((T + 2 + D_ABC, 106, 1, mxm(3, 0, 8, 0, 0), 0))         # acc.x128 after an integer product
o, _ = run(recs, sram)
check("acc.x128 after an integer ABC is OTS-E-OPR-001", any("OTS-E-OPR-001" in l for l in diags(o)), diags(o))

print("ALL PASS" if ok else "SOME FAIL")
sys.exit(0 if ok else 1)
