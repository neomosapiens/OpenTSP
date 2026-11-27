""" mut_bank kill: two AGT loop references in one bank with no new records → MEM-001 x2.

Only run_loop reports those diagnostics; disabling `if blocked then report(MEM_BANK)`
(the mut_bank mutant) drops them. Directed Q20 of directed.py.
usage: bank_agt_clash.py <tsp_sim> <config.json>
"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines

SIM, CFG = sys.argv[1], sys.argv[2]
CJ = json.load(open(CFG))
if CJ["geometry"]["mem_slices_per_hemisphere"] < 44:
    print("PASS bank_agt_clash (skipped: needs 44 MEM slices)")
    sys.exit(0)
TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]
VL = 16 * SL
BANK1 = 1 << CJ["sram"]["bank_bit"]
DM = TM["mem"]["d_func"]


def mem(op, addr=0, s=0):
    return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)


def agt(op, v):
    return (1 << 60) | (op << 54) | ((v & 0xFFFF) << 16)


def E(i):
    return i


def W(i):
    return 32 + i


def at(c, q, code, n=1):
    return [f"T diag {c} queue {q} OTS-E-{code}"] * n


def diags(o):
    return [l for l in o.splitlines() if "diag" in l]


FEED_W16 = mem(0, 0, W(16))
SH = max(0, DM + 10 - 15)


def fw(a, b):
    return [(c - DM - 10, 1, 1, FEED_W16, 0) for c in range(a, b)]


def sh(recs):
    return [(c + SH, *r) for c, *r in recs]


def run(recs, sram=None):
    sram = sram or {}
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v[:VL]) for (sl, a), v in sram.items()]
    lines += q_lines(lower_records(recs))
    o = subprocess.run(
        [SIM, CFG, "--trace"],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    return o


H6Z = {(1, 0): bytes([9] * VL), (43, 0): bytes([5] * VL)}
recs = fw(16, 30)
recs += [
    (10, 43, 1, mem(0, 0, E(0)), 0),
    (11, 43, 1, agt(5, 3), 0),
    (12, 43, 1, agt(7, 6), 0),
    (16, 43, 1, mem(1, 0xff & (BANK1 - 1), W(16)), 0),
    (17, 43, 1, agt(5, 1), 0),
    (18, 43, 1, agt(7, 4), 0),
]
o = run(sh(recs), H6Z)
want = at(18 + SH, 43, "MEM-001", 2) + at(22 + SH, 43, "MEM-001", 2)
got = diags(o)
ok = got == want
if ok:
    print("PASS two AGT loops in one bank: MEM-001 x2 at 18 and 22")
else:
    print("Assertion failed: two AGT loops in one bank: MEM-001 x2 at 18 and 22: got %r want %r" % (got, want))
    raise SystemExit(1)
sys.exit(0)
