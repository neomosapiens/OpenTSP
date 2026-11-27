"""Directed end-to-end tests of the chip diagnostic log (microarchitecture.md sec. 16.3, platform.md sec. 9.1): the
diagnostic codes of class `data` (the data-dependent column of sec. 8: SXM-001, DEC-003, IQ-002, IQ-004, OPR-001 of a MEM or `*_SEL` queue,
TIM-001 of a C2C receive queue) are fed, in the reporting order of sec. 8, into a log of `ras.diag_depth` entries that the host reads through
the CSR space: DIAG_COUNT (0x400), DIAG_POP (0x401, write), DIAG_STATUS (0x402, bit 0 overflow, W1C) and DIAG[i] (0x410 + i, word: cycle 47..0,
code 55..48, queue 63..56). A code of another class is not logged. The registers are read with `R` lines after the run and written
by the host's `H csr` lines. Runs against the Sail emulator or the ISS. Profile-portable: ref, mini and the test profiles.
usage: ras_diag.py <tsp_sim|tsp-iss> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG))
DEPTH = CJ["ras"].get("diag_depth", 16)
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def setsegment(n): return (1 << 60) | (8 << 54) | (n << 16)


# the code numbers of DIAG[i] bits 55..48: the position of the code in the diag type of core/diagnostics.sail
CODE = {c: i for i, c in enumerate(["DEC-003", "QUE-001", "QUE-002", "TIM-001", "DIR-001", "SRF-002", "MEM-001", "OPR-001", "OPR-002", "RES-001", "RES-002",
                                    "RES-003", "SXM-001", "LNK-001", "LNK-002", "LNK-003", "IQ-002", "IQ-003", "IQ-004", "HIO-001", "ECC-001", "LNK-004",
                                    "ECC-003", "ECC-002"])}
CU = 144                                                  # the control channel's queue field


def entry(cycle, code, queue): return cycle | (CODE[code] << 48) | (queue << 56)


def run(recs, extra=(), units=()):
    sram, dumps = {}, []
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(units) + list(extra)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-400:])
    return p


def run_units(extra, units):
    ls = list(units) + list(extra)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-400:])
    return p


def xs(p): return {int(t[1], 16): int(t[2], 16) for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "X"}
def diags(p): return [(int(t[2]), int(t[4]), t[5][6:]) for t in (l.split() for l in p.stdout.splitlines()) if t[:2] == ["T", "diag"]]
def rl(*addrs): return [f"R {a:x}" for a in addrs]
def hw(cycle, addr, val): return f"H csr {cycle:x} w {addr:x} {val:x}"


COUNT, POP, STATUS, LOG = 0x400, 0x401, 0x402, 0x410

# 0. A run that reports nothing: the log is empty.
p = run([(10, 0, 1, mem(0, 1, 0), 0)], rl(COUNT, STATUS, LOG))
x = xs(p)
check("a clean run: DIAG_COUNT 0, DIAG_STATUS 0, DIAG[0] 0", x == {COUNT: 0, STATUS: 0, LOG: 0}, x)

# 1. OPR-001 of a MEM queue (a Read outside the segment) is class data: logged, in cycle order, with its queue; QUE-002 (a Read on a VXM queue,
# class schedule) and the like are reported (T diag) but not logged.
N = DEPTH + 2
recs = [(10, 0, 1, setsegment(4), 0)] + [(11 + k, 0, 1, mem(0, 0x20 + k, 1), 0) for k in range(N)] + [(12, 88, 1, mem(0, 1, 2), 0)]
p = run(recs, rl(COUNT, STATUS, *(LOG + i for i in range(DEPTH))))
x, d = xs(p), diags(p)
opr = [(c, q, code) for c, q, code in d if code == "OPR-001"]
check("the program raises N OPR-001 on queue 0 and one QUE-002 on queue 88", len(opr) == N and (12, 88, "QUE-002") in d, d[:4])
check(f"DIAG_COUNT is the depth {DEPTH} (the log is full) and DIAG_STATUS bit 0 (overflow) is set: {N - DEPTH} records were dropped",
      x[COUNT] == DEPTH and x[STATUS] & 1 == 1, (x[COUNT], x[STATUS]))
want = [entry(c, "OPR-001", 0) for c, _, _ in opr[:DEPTH]]
check("DIAG[i] holds the first records in reporting order: cycle 47..0, code 55..48, queue 63..56; the QUE-002 of queue 88 is not in the log",
      [x[LOG + i] for i in range(DEPTH)] == want, ([hex(x[LOG + i]) for i in range(DEPTH)], [hex(w) for w in want]))

# 2. The host pops the oldest entry and clears the overflow flag (W1C); the later entries move down. Reads at cycles after the program.
T0 = 11 + N + 20
extra = [hw(T0, POP, 1), f"H csr {T0 + 1:x} r {COUNT:x}", f"H csr {T0 + 1:x} r {LOG:x}", hw(T0 + 2, STATUS, 1), f"H csr {T0 + 3:x} r {STATUS:x}"] + rl(COUNT, STATUS, LOG)
p = run(recs, extra)
x = xs(p)
check("DIAG_POP removes the oldest entry: DIAG_COUNT falls by one and DIAG[0] is the second record", x[COUNT] == DEPTH - 1 and x[LOG] == (want[1] if DEPTH > 1 else 0), (x[COUNT], hex(x[LOG])))
check("a write of 1 to DIAG_STATUS clears the overflow flag", x[STATUS] == 0, x[STATUS])
csr_reads = [l for l in p.stdout.splitlines() if l.startswith("T csr")]
check("the host's timed reads see the same values", len(csr_reads) == 3, csr_reads)

# 3. DEC-003 on the control channel (an illegal word, tag 7) is class data too: queue field 144 (the control channel).
units = ["Q ffff 00000005 70000000"]
p = run_units(rl(COUNT, LOG), units)
x, d = xs(p), diags(p)
check("DEC-003 of the control channel is logged with the queue field 144", d == [(5, 65535, "DEC-003")] and x[COUNT] == 1 and x[LOG] == entry(5, "DEC-003", CU), (d, x))

# 4. DIAG_COUNT and DIAG[i] are read-only; an unmapped address above the log reads 0.
p = run(recs, [hw(T0, COUNT, 0), hw(T0, LOG, 0)] + rl(COUNT, LOG, LOG + DEPTH))
x = xs(p)
check("writes to DIAG_COUNT and DIAG[0] are ignored; the address after the last entry reads 0", x[COUNT] == DEPTH and x[LOG] == want[0] and x[LOG + DEPTH] == 0, (x, hex(want[0])))

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
