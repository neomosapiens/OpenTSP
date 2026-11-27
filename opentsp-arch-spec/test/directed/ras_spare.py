"""Directed end-to-end tests of the error report in the spare bits of the stream registers (microarchitecture.md sec. 15.5): a Gather or
Scatter tile whose address is outside the segment sets, in the commit cycle of the instruction at its write gap, the spare bit of its
superlane on S[0] (Gather) or S[1] (Scatter), the even parity on S[3] and the address on S[4..16]; the first consumer of such a stream
logs it (`T ecc <c> queue <q> spare <row>:<stream> sl <x> pos <S[0..3]>`, a log entry of kind 9, `ERR_STATUS` bit 8). The data and the
diagnostics (OTS-E-OPR-001) are the same as without the report.
Runs against the Sail emulator or the ISS. Profile-portable: ref, mini and the test profiles.
usage: ras_spare.py <tsp_sim|tsp-iss|oracle> <config.json>   (oracle: the Python oracle; the error-register checks are skipped)"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
ok = True
if CJ["geometry"]["mem_slices_per_hemisphere"] < 44:
    print("PASS all (skipped: the programs use MEM slices up to 44; this profile has fewer)"); sys.exit(0)


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)
def gather(d, m, k=0, t=0): return (1 << 60) | (3 << 54) | (t << 53) | (k << 48) | (d << 42) | (m << 36)
def scatter(s, m, k=0, t=0): return (1 << 60) | (4 << 54) | (t << 53) | (k << 48) | (s << 42) | (m << 36)
def setsegment(n): return (1 << 60) | (8 << 54) | (n << 16)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap) - MEM_DS, slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


X = SL // 2                                  # the superlane of the bad tile
BAD = 0x25                                   # its address; the segment is 4 words, the good tiles use address 1
GAP = gap_mem(0, "E")                        # the write gap of slice 0 on row E (the report's register)
RD = 21                                      # the consumer's gap (the VXM)
TG = 100                                     # dispatch cycle of the Gather or Scatter
TV = TG + DM + (RD - GAP)                    # the cycle the report reaches the consumer (and its operands)
VXMQ = 88


def map_word(bad):
    m = bytearray(VL)
    for g in range(SL): m[16 * g] = 1
    if bad: m[16 * X] = BAD
    return bytes(m)


A = bytes((13 * l + 5) & 0xFF for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
SUM = bytes((x + y) & 0xFF for x, y in zip(A, B))
WORD = bytes((3 * l + 17) & 0xFF for l in range(VL))
DATA = bytes((11 * l + 29) & 0xFF for l in range(VL))


def program(op, bad, reads):
    """Phase 1: SetSegment 4 on slice 0; Gather E:2 <- map E:4 (or Scatter E:2 -> map E:4) at TG, the map word fed from slice 4.
    Phase 2: a VXM `add.i8 E:8 = E:<a> + E:<b>` at the cycle the report reaches gap 21, its operands fed from slices 8 and 12."""
    recs, sram = [], {}
    sram[(0, 1)] = WORD
    recs.append((10, 0, 1, setsegment(4), 0))
    feed(recs, sram, 4, 0, map_word(bad), 4, "E", GAP, TG)
    if op == "scatter": feed(recs, sram, 16, 3, DATA, 2, "E", GAP, TG)
    recs.append((TG, 0, 1, gather(E(2), E(4)) if op == "gather" else scatter(E(2), E(4)), 0))
    a, b = reads
    feed(recs, sram, 8, 1, A, a, "E", RD, TV)
    feed(recs, sram, 12, 2, B, b, "E", RD, TV)
    recs.append((TV - VXM_DS, VXMQ, 1, add_i8(E(8), E(a), E(b)), 0))
    drain(recs, 44, 9, 8, "E", RD, TV - VXM_DS + DV)
    return recs, sram


_ORACLE = []
def oracle_go(recs, sram):
    import types
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, ".."))
    from oracle import machine as mc, diff_oracle as do
    if not _ORACLE:
        _ORACLE.append(mc.Machine(CJ))
    prog = types.SimpleNamespace(recs=list(recs), units={}, pre=dict(sram), faults=[], dump={(44, 9)})
    return types.SimpleNamespace(returncode=0, stderr="", stdout="\n".join(do.oracle_output(_ORACLE[0], prog)) + "\n")


def go(op, bad, reads, opts=()):
    recs, sram = program(op, bad, reads)
    if SIM == "oracle":
        return oracle_go(recs, sram)
    dumps = [(44, 9), (0, 1)]
    extra = ["R 10", "R 14", "R 20", "R 21", "R 22", "R 23"]          # ERR_STATUS, the counter, the first two log entries
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + extra
    p = subprocess.run([SIM, CFG, "--trace", *opts], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (op, bad, reads, p.returncode, p.stderr, p.stdout[-600:])
    return p


def ln(p, prefix): return [l for l in p.stdout.splitlines() if l.startswith(prefix)]
def hexs(b): return " ".join(f"{x:02x}" for x in b)
def csr(p, a):
    for l in p.stdout.splitlines():
        t = l.split()
        if len(t) == 3 and t[0] == "X" and int(t[1], 16) == a: return int(t[2], 16)
    return None
SPARE = lambda p: [l for l in ln(p, "T ecc") if " spare " in l]
GOOD_SUM = f"W 2c 9 {hexs(SUM)}"
SIG = f"sl {X} pos "

# 0. No error: no report, nothing logged.
p = go("gather", False, (0, 3))
check("spare: a Gather with every tile in range reports nothing", not SPARE(p) and not ln(p, "T diag") and (SIM == "oracle" or csr(p, 0x10) == 0) and GOOD_SUM in p.stdout, (SPARE(p), ln(p, "T diag"), csr(p, 0x10)))

# 1. Gather with one bad tile: S[0] and the parity S[3] of superlane X; the consumer of E:0 and E:3 logs both.
p = go("gather", True, (0, 3))
sp = SPARE(p)
check("spare: Gather, one bad tile: the diagnostic OTS-E-OPR-001 is reported as before", any("OTS-E-OPR-001" in l for l in ln(p, "T diag")), ln(p, "T diag"))
check("spare: Gather, one bad tile: the consumer of E:0 and E:3 logs the report, S[0..3] = 1,0,0,1 (pos 9), once per stream",
      sp == [f"T ecc {TV} queue {VXMQ} spare E:0 {SIG}9", f"T ecc {TV} queue {VXMQ} spare E:3 {SIG}9"], sp)
check("spare: Gather, one bad tile: the data of the consumer is untouched by the report", GOOD_SUM in p.stdout, "")
check("spare: Gather, one bad tile: a report raises no error interrupt (ERR_MASK resets with bit 8 clear)", not ln(p, "T irq"), ln(p, "T irq"))
st, w0, w1 = csr(p, 0x10), csr(p, 0x20), csr(p, 0x21)
if SIM != "oracle": check("spare: the error registers: ERR_STATUS bit 8, the first log entry is kind 9 with the cycle and the queue",
      st is not None and st & 0x100 and w0 == (VXMQ << 52) | (9 << 48) | TV, (st, w0))
if SIM != "oracle": check("spare: the log entry's location: row E, stream 0, the consumer's gap; superlane and S[0..3]",
      w1 is not None and (w1 & 0x7FFFFFFFFFF) == (RD << 6) and (w1 >> 43) & 31 == X and (w1 >> 48) & 511 == 9, w1)
if SIM != "oracle": check("spare: the second log entry is the stream E:3 of the same cycle", len(sp) == 2 and csr(p, 0x22) == (VXMQ << 52) | (9 << 48) | TV and (csr(p, 0x23) & 0x7FFFFFFFFFF) == (RD << 6) | (3 << 1), (csr(p, 0x22), csr(p, 0x23)))

# 2. The address of the bad tile is on S[4..16] (bit i on stream 4 + i): 0x25 = bits 0, 2, 5 -> streams 4, 6, 9; a consumer of E:4 and E:5.
p = go("gather", True, (4, 6))
check("spare: Gather: the address bits on S[4] and S[6] (0x25 = bits 0, 2, 5) are logged by their consumer, with S[0..3] = 9",
      SPARE(p) == [f"T ecc {TV} queue {VXMQ} spare E:4 {SIG}9", f"T ecc {TV} queue {VXMQ} spare E:6 {SIG}9"], SPARE(p))
p = go("gather", True, (5, 7))
check("spare: Gather: the streams whose address bit is clear (0x25: bits 1 and 3) carry no spare bit: no log", not SPARE(p), SPARE(p))

# 3. Scatter with one bad tile: S[1] and the parity S[3] (pos 10).
p = go("scatter", True, (1, 3))
sp = SPARE(p)
check("spare: Scatter, one bad tile: S[1] and S[3] (pos 10) of superlane X, logged by the consumer of E:1 and E:3",
      sp == [f"T ecc {TV} queue {VXMQ} spare E:1 {SIG}10", f"T ecc {TV} queue {VXMQ} spare E:3 {SIG}10"], sp)
check("spare: Scatter, one bad tile: OTS-E-OPR-001 as before, the good tiles' bytes are written", any("OTS-E-OPR-001" in l for l in ln(p, "T diag")) and GOOD_SUM in p.stdout, ln(p, "T diag"))
p = go("scatter", True, (0, 7))
check("spare: Scatter: S[0] is clear (no Gather error): a consumer of E:0 and E:7 logs nothing", not SPARE(p), SPARE(p))

# 4. A consumer that reads no spare stream sees nothing; the superlanes other than X carry no report.
p = go("gather", True, (9, 10))
check("spare: Gather: a consumer of E:9 and E:10 (address bit 5 is on E:9: logged; E:10 is not)", len(SPARE(p)) == 1 and "E:9" in SPARE(p)[0], SPARE(p))

# 5. Report retention: the model keeps the 256 latest reports. Queue 40 writes one bank-conflict report at
# cycle 100 (E:2, read of the pair; slice 41 reads a value onto E:2 for it to ride with; the observer is `add.i8 E:8 = E:2 + E:2` of the VXM queue at cycle 116). Every other MEM queue of a
# hemisphere (21 of the 22 queues that are multiples of 4) writes one report per cycle on the W row from cycle 101: n cycles of such
# traffic make 21 n + n new registers after the first. With n = 11 the array still holds the cycle-100 report and the VXM observes it
# (two spare reads, E:2 twice); with n = 12 the 257th new report overwrote it while it was still in its row: the VXM observes nothing
# (only the bank conflicts are reported) and the lint OTS-W-RAS-001 says so (--lint; the Python oracle prints the same line).
def evict_program(n):
    rd = lambda s, a=0: mem(0, a, s)
    wr = lambda s, a=0: mem(1, a, s)
    recs = []
    recs += [(100, 40, 1, rd(2), 0), (100, 40, 1, wr(6), 0)]                       # the report that is observed (E:2, E:6)
    recs.append((100, 41, 1, rd(2), 0))                                           # a value on E:2 for the report to ride with (a report on an empty register is seen by nobody)
    for k in range(n - 1): recs += [(101 + k, 40, 1, rd(34), 0), (101 + k, 40, 1, wr(38), 0)]
    for q in range(0, 88, 4):
        if q == 40: continue
        for k in range(n): recs += [(101 + k, q, 1, rd(34), 0), (101 + k, q, 1, wr(38), 0)]
    recs.append((116, VXMQ, 1, add_i8(E(8), E(2), E(2)), 0))
    return recs


def evict_run(n):
    recs = evict_program(n)
    if SIM == "oracle":
        out = oracle_go(recs, {})
        return out.stdout.splitlines()
    ls = q_lines(lower_records(list(recs))) + ["D 54 9"]
    p = subprocess.run([SIM, CFG, "--trace", "--lint"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (n, p.returncode, p.stderr, p.stdout[-600:])
    return p.stdout.splitlines()


def retention(out):
    spare = [l for l in out if l.startswith("T ecc") and " spare " in l]
    lint = [l for l in out if l.startswith("T lint")]
    return len(spare), lint


# the cycles of the program are those of the ref timing (MEM d_func 5, VXM d_skew 2, d_func 5, MEM d_skew 0): a profile with other rows skips the section
if (DM, VXM_DS, DV, MEM_DS) == (5, 2, 5, 0):
    n11, l11 = retention(evict_run(11))
    n12, l12 = retention(evict_run(12))
    check("retention n = 11 (242 new reports): the report of cycle 100 is still there: the VXM observes two spare reads, no lint", n11 == 2 and not l11, (n11, l11))
    check("retention n = 12 (265 new reports): the report of cycle 100 was evicted while live: no spare read, the lint OTS-W-RAS-001",
          n12 == 0 and len(l12) >= 1 and all("OTS-W-RAS-001" in l for l in l12), (n12, l12[:3]))

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
