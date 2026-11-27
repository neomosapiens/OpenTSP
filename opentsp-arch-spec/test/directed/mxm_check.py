"""Directed end-to-end tests of exact per-row matrix checking (microarchitecture.md sec.18).
Primary row products are compared with an independently recomputed nominal row vector.
Every final multirow error is uncorrectable; one changed row is located and corrected.
The fault-free fast path and ABC timing are unchanged. usage: mxm_check.py <tsp_sim> <config.json>"""
import json, os, random, struct, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
VL = 16 * CJ["geometry"]["superlanes"]
DM = TM["mem"]["d_func"]
D_ABC, D_ACC = TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"]
ABC_DS = TM["mxm_abc"]["d_skew"]
ROWS = 16                                        # installed rows of i8 (f16 rows take two streams: 8 rows)
ok = True
_PROGS = []
if CJ["geometry"]["mem_slices_per_hemisphere"] < 44:
    print("PASS all (skipped: the 16 weight streams are read from distinct slices up to 44; this profile has fewer)"); sys.exit(0)


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)
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


def text(recs, sram, dumps, extra=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    return ls + q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(extra)


def run(lines, *args, want=0):
    if not args and want == 0:
        _PROGS.append(list(lines))     # recorded by tools/gen_model_vectors.py (the programs of the vectors)
    p = subprocess.run([SIM, CFG, "--trace", *args], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == want, (p.returncode, p.stderr)
    return p.stdout


def lines_of(out, kind): return [l for l in out.splitlines() if l.startswith(f"T {kind} ")]
def words(out): return {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in out.splitlines()) if t and t[0] == "W"}
def diags(out): return [l for l in lines_of(out, "diag") if "OTS-E-TIM-001" not in l]     # the drain writes of a missing product read no value (TIM-001): not the point here
def eccs(out): return lines_of(out, "ecc")
def fm(cycle, plane, lane, bit): return f"F mxm {cycle:x} {plane:x} {lane:x} {bit:x}"


T = 200                                          # the IW from streams
T2 = T + 2                                       # the ABC (dispatch); it executes at T2 + d_skew
EXEC = T2 + ABC_DS
T3 = T2 + D_ABC                                  # the ACC (dispatch)
QI, QA, QC = 104, 105, 106                       # IW, ABC and ACC queues of plane 0


def i8s(v): return bytes(x & 0xFF for x in v)
def f16s(vals): return b"".join(struct.pack("<e", v) for v in vals)
def f16_planes(vals):
    raw = f16s(vals)
    return raw[0::2], raw[1::2]


def prog(t, rows, x, abc_count=1, acc=True, gap=0):
    """16 byte-plane streams of weights (i8: 16 rows; f16: 8 rows as (low, high) planes), the activation (i8: one plane; f16: low and
    high planes), `abc_count` ABCs (the k-th `gap + k * ABC_GAP` cycles after the first) and an ACC after the last. Returns the input lines."""
    sram, recs = {}, []
    for r in range(16):
        feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, rows[r], r, "W", 43, T)
    recs.append((T, QI, 1, mxm(1, t, 32 | 0, 1), 0))                        # iw.<t> W:0 (from streams)
    for k in range(abc_count):
        c = T2 + k * 40
        feed(recs, sram, 60, 5 + k, x[0], 24, "W", 43, c)
        if t == 1:
            feed(recs, sram, 61, 6 + k, x[1], 25, "W", 43, c)
        recs.append((c, QA, 1, mxm(2, t, 32 | 24), 0))
    c3 = T2 + (abc_count - 1) * 40 + D_ABC
    if acc:
        recs.append((c3, QC, 1, mxm(3, 2 if t == 0 else 3, 8, 0, 1), 0))        # acc.i32 or acc.f32 E:8, replace, accumulator 1
        for b in range(4):
            drain(recs, 4 * b, 9, 8 + b, "E", 2, c3 + D_ACC)
    return sram, recs, [(4 * b, 9) for b in range(4)] if acc else []


def product(out):
    w = words(out)
    got = [w.get((4 * b, 9), bytes(VL)) for b in range(4)]
    return [int.from_bytes(bytes(got[b][l] for b in range(4)), "little") for l in range(ROWS)]


rnd = random.Random(2026)
def i8_case():
    rows = [i8s([rnd.randrange(-128, 128) for _ in range(VL)]) for _ in range(16)]
    xs = [rnd.randrange(-128, 128) for _ in range(VL)]
    exp = [sum((w if w < 128 else w - 256) * x for w, x in zip(rows[r], xs)) & 0xFFFFFFFF for r in range(ROWS)]
    return rows, [i8s(xs)], exp
def f16_case():
    vals = [[rnd.choice([0, 1, -1, 0.5, -0.25, 3, -2, 0.125, 1.5, 7, -6.5]) for _ in range(VL)] for _ in range(8)]
    pl = [f16_planes(v) for v in vals]
    rows = [p[i] for p in pl for i in (0, 1)]                              # stream 2i low, 2i + 1 high
    xv = [rnd.choice([0, 1, -1, 0.5, 2, -3, 0.25, 1.75]) for _ in range(VL)]
    return rows, list(f16_planes(xv)), None


CASES = {"i8": (0, i8_case())}
if CJ["mxm"]["planes_per_hemisphere"] == 2:                    # an f16 product needs the pair of planes
    CASES["f16"] = (1, f16_case())
for tn, (t, (rows, x, exp)) in CASES.items():
    sram, recs, dumps = prog(t, rows, x)
    base = run(text(recs, sram, dumps))
    ref = product(base)
    check(f"{tn}: fault-free product, no ecc line, no diagnostic" + (", equals the exact i32 sums" if exp else ""),
          not eccs(base) and not diags(base) and (exp is None or ref == exp) and any(ref), (eccs(base), diags(base), ref[:3], (exp or [])[:3]))
    # 1. one wrong lane is located and corrected: the product is the fault-free one
    NL = ROWS if tn == "i8" else ROWS // 2                                # installed rows: f16 rows take two byte planes
    for lane, bit in ((0, 0), (1, 5), (5, 40), (NL - 1, 63), (NL - 6, 90), (NL - 3, 95), (NL - 7, 12)):
        p = run(text(recs, sram, dumps, [fm(EXEC, 0, lane, bit)]))
        check(f"{tn}: bit {bit} of lane {lane} flips: corrected at lane {lane}, product unchanged, no diagnostic",
              eccs(p) == [f"T ecc {EXEC} queue {QA} mxm corr plane 0 lane {lane}"] and not diags(p) and product(p) == ref, (eccs(p), diags(p), product(p)[:4]))
    # 2. two bits of one lane are one wrong lane
    p = run(text(recs, sram, dumps, [fm(EXEC, 0, 6, 3), fm(EXEC, 0, 6, 44)]))
    check(f"{tn}: two flips in one lane: still one wrong lane, corrected", eccs(p) == [f"T ecc {EXEC} queue {QA} mxm corr plane 0 lane 6"] and not diags(p) and product(p) == ref, (eccs(p), diags(p)))
    # 3. two wrong lanes are uncorrectable: ECC-003 on the ABC, no product, RES-001 at the ACC
    for a, b in ((3, NL - 4), (0, 1), (5, NL - 6)):
        p = run(text(recs, sram, dumps, [fm(EXEC, 0, a, 9), fm(EXEC, 0, b, 17)]))
        check(f"{tn}: lanes {a} and {b} wrong: OTS-E-ECC-003 on the ABC, the ABC has no effect (RES-001 at the ACC), no product",
              diags(p) == [f"T diag {EXEC} queue {QA} OTS-E-ECC-003", f"T diag {T3} queue {QC} OTS-E-RES-001"] and not eccs(p) and not any(product(p)), (diags(p), eccs(p)))
    # 4. two lanes whose errors cancel in the total are still found by their row residuals
    p = run(text(recs, sram, dumps, [fm(EXEC, 0, 2, 20), fm(EXEC, 0, NL - 5, 20)]))
    check(f"{tn}: two lanes wrong by the same bit: uncorrectable (OTS-E-ECC-003)", diags(p)[:1] == [f"T diag {EXEC} queue {QA} OTS-E-ECC-003"] and not any(product(p)), (diags(p), eccs(p)))
    # 5. a lane the installed weights do not occupy carries no product: no effect
    lane = NL
    if lane + 3 < CJ["mxm"]["rows"]:                                       # a lane the profile has but the installed weights do not occupy
        p = run(text(recs, sram, dumps, [fm(EXEC, 0, lane + 3, 7)]))
        check(f"{tn}: a flip of lane {lane + 3}, beyond the installed rows: no effect", not eccs(p) and not diags(p) and product(p) == ref, (eccs(p), diags(p)))
    # 6. re-execution: the ABC that found an error has no effect; the same ABC again, without a fault, gives the product
    sram2, recs2, dumps2 = prog(t, rows, x, abc_count=2)
    p = run(text(recs2, sram2, dumps2, [fm(EXEC, 0, 4, 1), fm(EXEC, 0, NL - 1, 2)]))
    check(f"{tn}: the first ABC is uncorrectable (ECC-003), the second, re-executed without a fault, gives the product",
          diags(p) == [f"T diag {EXEC} queue {QA} OTS-E-ECC-003"] and product(p) == ref, (diags(p), product(p)[:3]))

# 8. The matrix-unit events in the error CSRs and the error interrupt (microarchitecture.md sec. 16 and 18): a corrected lane is a kind 5
#    entry (ERR_STATUS bit 4, SBE_COUNT), an uncorrectable ABC a kind 6 entry (bit 5, MBE_COUNT); log word 0 carries the ABC's
#    execute cycle and queue, log word 1 is plane << 16 | lane (the lane the check located; 0xFFFF when the check located none). The
#    reset ERR_MASK 0x0AA raises MSI-X vector 0 for an uncorrectable ABC and not for a corrected one.
DEPTH = CJ.get("ras", {}).get("log_depth", 8)
def w0(cycle, kind, q): return cycle | (kind << 48) | (q << 52)
def w1(plane, lane): return (plane << 16) | lane
def hw(cycle, addr, val): return f"H csr {cycle:x} w {addr:x} {val:x}"
ADDRS = (0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x021, 0x022, 0x0F0, 0x0F1, 0x030, 0x032)
RR = [f"R {a:x}" for a in ADDRS]
def xs(out): return {int(t[1], 16): int(t[2], 16) for t in (l.split() for l in out.splitlines()) if t and t[0] == "X"}
def reg(out):
    x = xs(out)
    return tuple(x[a] for a in ADDRS)
def irqs(out): return lines_of(out, "irq")
for tn, (t, (rows, x, exp)) in CASES.items():
    sram, recs, dumps = prog(t, rows, x)
    NL = ROWS if tn == "i8" else ROWS // 2
    p = run(text(recs, sram, dumps, RR))
    check(f"{tn}: events: no fault, every error register reads 0, no interrupt", reg(p) == (0,) * 12 and not irqs(p), (reg(p), irqs(p)))
    for lane, bit in ((0, 0), (5, 40), (NL - 1, 63)):
        p = run(text(recs, sram, dumps, [fm(EXEC, 0, lane, bit)] + RR))
        check(f"{tn}: events: a corrected lane {lane}: ERR_STATUS bit 4, severity 1, SBE_COUNT 1, one kind 5 entry (cycle {EXEC}, queue {QA}, plane 0 << 16 | lane), FIRST_ERR the same, no interrupt",
              reg(p) == (0x10, 1, 1, 0, 1, w0(EXEC, 5, QA), w1(0, lane), 0, w0(EXEC, 5, QA), w1(0, lane), 0, 0) and not irqs(p), (reg(p), irqs(p)))
    for a, b in ((3, NL - 4), (0, 1)):
        p = run(text(recs, sram, dumps, [fm(EXEC, 0, a, 9), fm(EXEC, 0, b, 17)] + RR))
        check(f"{tn}: events: lanes {a} and {b} wrong: ERR_STATUS bit 5, severity 2, MBE_COUNT 1, one kind 6 entry (lane field 0xFFFF: none located), `T irq {EXEC} msix 0`, IRQ_COUNT 1",
              reg(p) == (0x20, 2, 0, 1, 1, w0(EXEC, 6, QA), w1(0, 0xFFFF), 0, w0(EXEC, 6, QA), w1(0, 0xFFFF), 1, 1) and irqs(p) == [f"T irq {EXEC} msix 0"], (reg(p), irqs(p)))
    # ERR_MASK decides the interrupt like for every other error bit
    p = run(text(recs, sram, dumps, [fm(EXEC, 0, 5, 40), hw(1, 0x016, 0x10)] + RR))
    check(f"{tn}: events: with ERR_MASK bit 4 set a corrected lane raises the interrupt too", irqs(p) == [f"T irq {EXEC} msix 0"] and reg(p)[10:] == (1, 1), (irqs(p), reg(p)))
    p = run(text(recs, sram, dumps, [fm(EXEC, 0, 3, 9), fm(EXEC, 0, NL - 4, 17), hw(1, 0x016, 0)] + RR))
    check(f"{tn}: events: with ERR_MASK cleared an uncorrectable ABC raises no interrupt, the registers still show it", not irqs(p) and reg(p)[:4] == (0x20, 2, 0, 1) and reg(p)[10:] == (0, 0), (irqs(p), reg(p)))
    # two ABCs, the first with one wrong lane (corrected), the second with two (uncorrectable): two entries in execution order
    sram2, recs2, dumps2 = prog(t, rows, x, abc_count=2)
    EX2 = EXEC + 40
    p = run(text(recs2, sram2, dumps2, [fm(EXEC, 0, 4, 1), fm(EX2, 0, 2, 3), fm(EX2, 0, NL - 2, 5)] + RR + ["R 023", "R 024"]))
    xx = xs(p)
    check(f"{tn}: events: a corrected ABC then an uncorrectable one: bits 4 and 5, SBE 1, MBE 1, entries in execution order, FIRST_ERR the first, one interrupt at {EX2}",
          (xx[0x010], xx[0x011], xx[0x012], xx[0x013], xx[0x014]) == (0x30 | (512 if DEPTH < 2 else 0), 2, 1, 1, min(2, DEPTH))
          and (xx[0x020], xx[0x021]) == (w0(EXEC, 5, QA), w1(0, 4)) and (DEPTH < 2 or (xx[0x022], xx[0x023]) == (w0(EX2, 6, QA), w1(0, 0xFFFF)))
          and (xx[0x0F0], xx[0x0F1]) == (w0(EXEC, 5, QA), w1(0, 4)) and irqs(p) == [f"T irq {EX2} msix 0"], ({k: hex(v) for k, v in xx.items()}, irqs(p)))

# the plane number is in log word 1: the same program on plane 1 (its three queues follow those of plane 0), a corrected lane and an uncorrectable ABC
if CJ["mxm"]["planes_per_hemisphere"] == 2:                    # with one plane per hemisphere the queues of a second west plane are absent
    QI0, QA0, QC0 = QI, QA, QC
    QI, QA, QC = QI0 + 3, QA0 + 3, QC0 + 3
    sram, recs, dumps = prog(0, *CASES["i8"][1][:2])
    p = run(text(recs, sram, dumps, [fm(EXEC, 1, 5, 40)] + RR))
    check("i8: events, plane 1: a corrected lane 5 is a kind 5 entry with plane 1 << 16 | lane (0x10005) and the queue of the ABC of plane 1",
          reg(p) == (0x10, 1, 1, 0, 1, w0(EXEC, 5, QA), 0x10005, 0, w0(EXEC, 5, QA), 0x10005, 0, 0) and eccs(p) == [f"T ecc {EXEC} queue {QA} mxm corr plane 1 lane 5"], (reg(p), eccs(p)))
    p = run(text(recs, sram, dumps, [fm(EXEC, 1, 3, 9), fm(EXEC, 1, 12, 17)] + RR))
    check("i8: events, plane 1: an uncorrectable ABC is a kind 6 entry with plane 1 << 16 | 0xFFFF (0x1FFFF), the interrupt is raised",
          reg(p) == (0x20, 2, 0, 1, 1, w0(EXEC, 6, QA), 0x1FFFF, 0, w0(EXEC, 6, QA), 0x1FFFF, 1, 1) and irqs(p) == [f"T irq {EXEC} msix 0"], (reg(p), irqs(p)))
    QI, QA, QC = QI0, QA0, QC0

# : the aggregate subset scheme aliased both of these admitted four-row errors.
for nominal in ([0, 0, 0, 1], [0, 1, 1, 0]):
    alias_rows = [bytes([value]) + bytes(VL - 1) for value in nominal] + [bytes(VL)] * 12
    alias_x = [bytes([1]) + bytes(VL - 1)]
    alias_sram, alias_recs, alias_dumps = prog(0, alias_rows, alias_x)
    alias_out = run(text(alias_recs, alias_sram, alias_dumps, [fm(EXEC, 0, r, 0) for r in range(4)]))
    check(f" four changed rows from {nominal}: ECC003, no corrected alias, no ABC result",
          any("OTS-E-ECC-003" in d for d in diags(alias_out))
          and any("OTS-E-RES-001" in d for d in diags(alias_out))
          and not eccs(alias_out) and product(alias_out) == [0] * ROWS,
          (eccs(alias_out), diags(alias_out), product(alias_out)))

# 7. the input errors
sram, recs, dumps = prog(0, *CASES["i8"][1][:2])
base_lines = text(recs, sram, dumps)
def bad(name, extra, msg, args=(), lines=None):
    p = subprocess.run([SIM, CFG, *args], input="\n".join((lines or base_lines) + list(extra)) + "\n", capture_output=True, text=True, timeout=600)
    check(f"bad input, {name}: exit 2 and `{msg}`", p.returncode == 2 and msg in p.stderr, (p.returncode, p.stderr))
bad("plane out of range", [f"F mxm {EXEC:x} 4 0 0"], "plane out of range")
bad("lane out of range", [f"F mxm {EXEC:x} 0 {CJ['mxm']['rows']:x} 0"], "lane out of range")
bad("bit out of range", [f"F mxm {EXEC:x} 0 0 60"], "bit out of range")
bad("missing field", [f"F mxm {EXEC:x} 0 0"], "missing bit")
bad("the same bit twice", [fm(EXEC, 0, 1, 2), fm(EXEC, 0, 1, 2)], "two lines flip the same bit")
bad("bad t|p marker", [f"F mxm {EXEC:x} 0 0 1 x"], "expected t or p")
bad("unknown F word", ["F mx 1 0 0 0"], "expected sram, srf, iq, link or mxm")
bad("no ABC executes in that cycle", [fm(EXEC + 1, 0, 1, 2)], f"F mxm: no ABC executes on plane 0 in cycle {EXEC + 1:x}")
bad("a cycle before the ABC of the plane: the later ABC does not take it", [fm(EXEC - 100, 0, 1, 2)], f"F mxm: no ABC executes on plane 0 in cycle {EXEC - 100:x}")
bad("an ABC on another plane", [fm(EXEC, 1, 1, 2)], f"F mxm: no ABC executes on plane 1 in cycle {EXEC:x}")
bad("a cycle with no program at all", [fm(5, 0, 1, 2)], "no ABC executes on plane 0 in cycle 5", lines=["R 0"])

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
