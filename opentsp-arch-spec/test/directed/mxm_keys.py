"""Directed tests of the MXM structure keys (microarchitecture.md sec. 6, topology.md "Profile parameters"): `mxm.acc_depth`,
`mxm.f16_rows`, `mxm.fp8_rate` and `mxm.planes_per_hemisphere`. Every expected value is computed from the configuration, so the
same script runs on every profile (and on the model and on the ISS):

 - accumulators: an ACC replaces and accumulates into any index below `mxm.acc_depth`, the accumulators are independent, and an
   index at or above the depth does not decode (OTS-E-DEC-003);
 - f16 capacity: a weight buffer takes 2 * `mxm.f16_rows` f16 stream values and then OTS-E-RES-002, while an i8 buffer takes
   `mxm.rows`; an installed f16 matrix of `mxm.f16_rows` rows gives a product with that many non-zero lanes;
 - FP8 rate: a plane accepts `mxm.fp8_rate` FP8 ABCs in one cycle (one more is OTS-E-RES-003 for every ABC of the cycle); the two
   products of the rate-2 plane go to the even and the odd accumulator by queue order;
 - planes: with one plane per hemisphere the queues of the second plane of each hemisphere report OTS-E-QUE-002, an f16 load is
   OTS-E-RES-001 (no pair), and the east plane is plane 1 (a matrix-check event names it).

usage: mxm_keys.py <tsp_sim|tsp-iss> <config.json>
"""
import json, os, struct, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines

SIM, CFG = sys.argv[1], sys.argv[2]
CJ = json.load(open(CFG))
TM, MX = CJ["timing"], CJ["mxm"]
VL = 16 * CJ["geometry"]["superlanes"]
ROWS, F16ROWS, DEPTH, RATE, PLANES = MX["rows"], MX["f16_rows"], MX["acc_depth"], MX["fp8_rate"], MX["planes_per_hemisphere"]
DM = TM["mem"]["d_func"]
D_LW, D_ABC, D_ACC, D_IW = TM["mxm_lw"]["d_func"], TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"], TM["mxm_iw"]["d_func"]
ok = True
if CJ["geometry"]["mem_slices_per_hemisphere"] < 44:
    print("PASS all (skipped: the 16 weight streams are read from distinct slices up to 44; this profile has fewer)")
    sys.exit(0)


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | ((acc & 1) << 12) | ((acc >> 1) << 8)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    up = (row == "E") == (slice_ < 44)
    return 18 - q if up else 24 + q


def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row)
    sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))


def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def run(recs, sram, dumps=(), extra=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(extra)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-300:])
    return p.stdout


def lines_of(out, kind): return [l.split() for l in out.splitlines() if l.startswith(f"T {kind} ")]
def diags(out): return [(int(t[2]), int(t[4]), t[5][6:]) for t in lines_of(out, "diag") if t[5][6:] != "TIM-001"]
def words(out): return {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in out.splitlines()) if t and t[0] == "W"}


def drains_of(recs, base_cycle, dst_streams, slices=(0, 4, 8, 12), addr=9):
    """The four byte-plane streams of an ACC result (commit cycle base_cycle) written to MEM east slices on row E."""
    dumps = []
    for b in range(4):
        drain(recs, slices[b], addr, dst_streams + b, "E", 2, base_cycle)
        dumps.append((slices[b], addr))
    return dumps


def word_lane(w, dumps, lane):
    return int.from_bytes(bytes(w.get(d, bytes(VL))[lane] for d in dumps), "little")


def i8w(v): return bytes(x & 0xFF for x in v)


# the west plane 0: queues 104 (IW), 105 (ABC), 106 (ACC)
QI, QA, QC = 104, 105, 106
T = 200


def install_i8(recs, sram, rows16, q=QI, t=0, row="W", slices_west=False):
    """iw.<t> from the 16 streams 0..15 (row W: MEM east slices; row E: MEM west slices) at cycle T."""
    for r in range(16):
        sl = 4 * (r % 11) + (r // 11) + (0 if slices_west else 44)
        feed(recs, sram, sl, 3 + r, rows16[r], r, row, 43, T)
    recs.append((T, q, 1, mxm(1, t, (32 if row == "W" else 0) | 0, 1), 0))


# ---------------------------------------------------------------------------------------------------------------------
# 1. accumulators
ROW_BYTES = [i8w([(3 * r + k) % 7 - 3 for k in range(VL)]) for r in range(16)]
XV = [(k % 5) - 2 for k in range(VL)]
EXP = [sum((w if w < 128 else w - 256) * x for w, x in zip(ROW_BYTES[r], XV)) & 0xFFFFFFFF for r in range(16)]


def acc_prog(i, j):
    sram, recs = {}, []
    install_i8(recs, sram, ROW_BYTES)
    c = T + 2
    feed(recs, sram, 60, 5, i8w(XV), 24, "W", 43, c)
    recs.append((c, QA, 1, mxm(2, 0, 32 | 24), 0))
    c3 = c + D_ABC
    recs.append((c3, QC, 1, mxm(3, 2, 8, 0, i), 0))              # acc.i32 E:8, replace, index i
    recs.append((c3 + 1, QC, 1, mxm(3, 2, 12, 1, i), 0))         # accumulate into i: 2 x product
    recs.append((c3 + 2, QC, 1, mxm(3, 2, 16, 1, j), 0))         # accumulate into j: the product (j was zero)
    d = {}
    for k, st in enumerate((8, 12, 16)):
        d[st] = drains_of(recs, c3 + k + D_ACC, st, addr=9 + k)
    return sram, recs, d


PAIRS = sorted({(i, (i + 1) % DEPTH) for i in (0, 1, DEPTH // 2, DEPTH - 1)} | ({(0, 2), (0, DEPTH // 2), (1, DEPTH - 1)} if DEPTH >= 4 else set()))
for i, j in PAIRS:
    sram, recs, d = acc_prog(i, j)
    dumps = [x for v in d.values() for x in v]
    out = run(recs, sram, dumps)
    w = words(out)
    got = [[word_lane(w, d[st], r) for r in range(16)] for st in (8, 12, 16)]
    two = [(2 * x) & 0xFFFFFFFF for x in EXP]
    check(f"accumulator {i} of {DEPTH}: replace gives the product, accumulate gives twice it, and accumulator {j} (untouched) accumulates just the product",
          not diags(out) and got[0] == EXP and got[1] == two and got[2] == EXP, (diags(out), got[0][:3], got[1][:3], got[2][:3], EXP[:3]))

# an index at or above the depth does not decode; the last legal index and (when it exists) the first illegal one
if DEPTH < 32:
    sram, recs, d = acc_prog(0, 1)
    recs = [r for r in recs if r[3] >> 28 != 3 or (r[3] >> 22) & 63 != 3]
    recs.append((T + 2 + D_ABC, QC, 1, mxm(3, 2, 8, 0, DEPTH), 0))
    out = run(recs, sram)
    check(f"accumulator index {DEPTH} (the depth) is OTS-E-DEC-003 and does not execute", [x for x in diags(out) if x[2] == "DEC-003"] == [(T + 2 + D_ABC, QC, "DEC-003")], diags(out))
bad = [(T + 2 + D_ABC, QC, 1, mxm(3, 2, 8, 0, 0) | (1 << 7), 0)]
sram, recs, d = acc_prog(0, 1)
recs = [r for r in recs if r[3] >> 28 != 3 or (r[3] >> 22) & 63 != 3] + bad
out = run(recs, sram)
check("a reserved bit of the ACC word (bit 7) is OTS-E-DEC-003", [x for x in diags(out) if x[2] == "DEC-003"] == [(T + 2 + D_ABC, QC, "DEC-003")], diags(out))

# ---------------------------------------------------------------------------------------------------------------------
# 2. f16 capacity: LW f16 up to 2 * f16_rows stream values, i8 up to rows; the matrix of f16_rows rows
def lw_prog(t, n_lw, data_lo, data_hi=None):
    sram, recs = {}, []
    for k in range(n_lw):
        c = 100 + 4 * k
        for r in range(16):
            val = (data_lo if (t != 1 or r % 2 == 0) else data_hi)
            feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, val, r, "W", 43, c)
        recs.append((c, QI, 1, mxm(0, t, 32 | 0), 0))
    return sram, recs, 100 + 4 * (n_lw - 1)


for tn, t, cap in (("i8", 0, ROWS), ("f16", 1, 2 * F16ROWS)):
    if t == 1 and PLANES != 2:
        continue                      # no plane runs f16 (section 4 checks that)
    n = cap // 16
    sram, recs, last = lw_prog(t, n + 1, bytes([1] * VL), bytes([2] * VL))
    out = run(recs, sram)
    d = [x for x in diags(out)]
    check(f"lw.{tn}: {n} loads of 16 stream values fill the buffer ({cap} stream values); load {n + 1} is OTS-E-RES-002",
          d == [(last, QI, "RES-002")], d)

if PLANES == 2:
    # the f16 matrix: LW f16 until full, IW, ABC, ACC.f32: the product has f16_rows non-zero lanes
    n = 2 * F16ROWS // 16
    one = struct.pack("<e", 1.0)
    sram, recs, last = lw_prog(1, n, bytes([one[0]] * VL), bytes([one[1]] * VL))
    c_iw = last + D_LW
    recs.append((c_iw, QI, 1, mxm(1, 1, 0, 0), 0))                              # iw.f16 (the buffer)
    c_abc = c_iw + 1
    feed(recs, sram, 60, 5, bytes([one[0]] * VL), 24, "W", 43, c_abc)
    feed(recs, sram, 61, 6, bytes([one[1]] * VL), 25, "W", 43, c_abc)
    recs.append((c_abc, QA, 1, mxm(2, 1, 32 | 24), 0))
    c_acc = c_abc + D_ABC
    recs.append((c_acc, QC, 1, mxm(3, 3, 8, 0, 0), 0))
    dumps = drains_of(recs, c_acc + D_ACC, 8)
    out = run(recs, sram, dumps)
    w = words(out)
    want = struct.unpack("<I", struct.pack("<f", float(VL)))[0]
    lanes = min(F16ROWS, VL)
    last_lane, next_lane = word_lane(w, dumps, lanes - 1), (word_lane(w, dumps, lanes) if lanes < VL else 0)
    check(f"an f16 matrix of {F16ROWS} rows: lane {lanes - 1} of the product is {VL}.0 and the lane above the rows is zero",
          not diags(out) and last_lane == want and next_lane == 0, (diags(out), hex(last_lane), hex(next_lane)))

# ---------------------------------------------------------------------------------------------------------------------
# 3. the FP8 rate
E4M3_ONE, E4M3_TWO, E4M3_FOUR = 0x38, 0x40, 0x48
if PLANES == 2:
    def fp8_prog(queues):
        sram, recs = {}, []
        for r in range(16):
            feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes([E4M3_ONE] * VL), r, "W", 43, T)
        recs.append((T, QI, 1, mxm(1, 3, 32 | 0, 1), 0))                        # iw.fp8 W:0
        c = T + 2
        for n, q in enumerate(queues):
            feed(recs, sram, 60 + n, 5, bytes([E4M3_ONE if n % 2 == 0 else E4M3_TWO] * VL), 24 + n, "W", 43, c)
            recs.append((c, q, 1, mxm(2, 3, 32 | (24 + n), 0), 0))               # abc.e4m3 W:24+n
        return sram, recs, c

    f32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    sram, recs, c = fp8_prog((QA, QC))
    out = run(recs, sram)
    d = diags(out)
    if RATE == 2:
        c_acc = c + D_ABC
        recs.append((c_acc, QI, 1, mxm(3, 3, 8, 0, 0), 0))
        recs.append((c_acc, QC, 1, mxm(3, 3, 12, 0, 1), 0))
        dumps = drains_of(recs, c_acc + D_ACC, 8) + drains_of(recs, c_acc + D_ACC, 12, slices=(16, 20, 24, 28))
        out = run(recs, sram, dumps)
        w = words(out)
        a0, a1 = word_lane(w, dumps[:4], 0), word_lane(w, dumps[4:], 0)
        check("fp8_rate 2: two ABCs in one cycle give no diagnostic; the even accumulator reads the lower queue's product, the odd one the higher queue's",
              not diags(out) and a0 == f32(float(VL)) and a1 == f32(2.0 * VL), (diags(out), hex(a0), hex(a1)))
        # a third ABC accepted in the cycle the pair commits does not cost the pair a product: an ACC of the next cycle (the new
        # ABC has not committed) still reads the lower queue's product on accumulator 0 and the higher queue's on accumulator 1
        sram, recs, c = fp8_prog((QA, QC))
        c_commit = c + D_ABC
        feed(recs, sram, 62, 5, bytes([E4M3_FOUR] * VL), 26, "W", 43, c_commit)
        recs.append((c_commit, QI, 1, mxm(2, 3, 32 | 26, 0), 0))                 # abc.e4m3 W:26 in the commit cycle of the pair
        c_acc = c_commit + 1
        recs.append((c_acc, QA, 1, mxm(3, 3, 8, 0, 0), 0))
        recs.append((c_acc, QC, 1, mxm(3, 3, 12, 0, 1), 0))
        dumps = drains_of(recs, c_acc + D_ACC, 8) + drains_of(recs, c_acc + D_ACC, 12, slices=(16, 20, 24, 28))
        out = run(recs, sram, dumps)
        w = words(out)
        a0, a1 = word_lane(w, dumps[:4], 0), word_lane(w, dumps[4:], 0)
        check("fp8_rate 2: an ABC in the pair's commit cycle leaves both products of the pair to the ACCs of the next cycle",
              not diags(out) and a0 == f32(float(VL)) and a1 == f32(2.0 * VL), (diags(out), hex(a0), hex(a1)))
    else:
        check("fp8_rate 1: two ABCs of one plane in one cycle are OTS-E-RES-003 on both queues",
              sorted(x for x in d if x[2] == "RES-003") == [(c, QA, "RES-003"), (c, QC, "RES-003")], d)
    sram, recs, c = fp8_prog((QI, QA, QC))
    out = run(recs, sram)
    d = diags(out)
    check(f"fp8_rate {RATE}: three ABCs of one plane in one cycle are OTS-E-RES-003 on every queue",
          sorted(x for x in d if x[2] == "RES-003") == [(c, QI, "RES-003"), (c, QA, "RES-003"), (c, QC, "RES-003")], d)

# ---------------------------------------------------------------------------------------------------------------------
# 4. planes per hemisphere
sram, recs = {}, []
for r in range(16):
    feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes([1] * VL), r, "W", 43, 50)
recs.append((50, 107, 1, mxm(0, 0, 32 | 0), 0))                                    # lw.i8 on the first queue of the second west plane
out = run(recs, sram)
check(f"planes_per_hemisphere {PLANES}: an lw.i8 on queue 107 (the second west plane) is "
      + ("accepted" if PLANES == 2 else "OTS-E-QUE-002"), diags(out) == ([] if PLANES == 2 else [(50, 107, "QUE-002")]), diags(out))
sram, recs = {}, []
for r in range(16):
    feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes([1] * VL), r, "W", 43, 50)
recs.append((50, QI, 1, mxm(0, 1, 32 | 0), 0))                                      # lw.f16 on plane 0
out = run(recs, sram)
check(f"planes_per_hemisphere {PLANES}: an lw.f16 on plane 0 is " + ("accepted (the pair 0, 1 exists)" if PLANES == 2 else "OTS-E-RES-001 (no pair)"),
      diags(out) == ([] if PLANES == 2 else [(50, QI, "RES-001")]), diags(out))
# the east plane is plane PLANES: a matrix-check event on it names that plane
QE = 110
sram, recs = {}, []
for r in range(16):
    feed(recs, sram, 4 * (r % 11) + (r // 11), 3 + r, ROW_BYTES[r], r, "E", 43, T)
recs.append((T, QE, 1, mxm(1, 0, 0 | 0, 1), 0))                                     # iw.i8 E:0
c = T + 2
feed(recs, sram, 16, 5, i8w(XV), 24, "E", 43, c)
recs.append((c, QE + 1, 1, mxm(2, 0, 24), 0))
exec_c = c + TM["mxm_abc"]["d_skew"]
out = run(recs, sram, (), [f"F mxm {exec_c:x} {PLANES:x} 3 5"])
ev = [" ".join(t) for t in lines_of(out, "ecc")]
check(f"the east plane is plane {PLANES}: a flipped bit in its ABC is corrected and reported on plane {PLANES}",
      ev == [f"T ecc {exec_c} queue {QE + 1} mxm corr plane {PLANES} lane 3"] and not diags(out), (ev, diags(out)))
if PLANES == 1:
    p = subprocess.run([SIM, CFG], input=f"F mxm {exec_c:x} 2 0 0\n", capture_output=True, text=True, timeout=60)
    check("a fault on plane 2 of a profile with one plane per hemisphere is an input error naming the key",
          p.returncode == 2 and "plane out of range (>= 2 * mxm.planes_per_hemisphere)" in p.stderr, (p.returncode, p.stderr))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
