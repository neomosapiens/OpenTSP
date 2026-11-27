"""Directed test of the weight-install time against its published bound .

The published TSP sources (ISCA20 section V; the Hot Chips 34 and the trade-press summaries of it) say that the MEM slices read all 409,600 weights
and install them into the MXM arrays "in less than 40 cycles", counted by an undefined convention (first read to last install, or first arrival to
last arrival). This test fixes a convention and measures it on the model: the first cycle at which an ABC sees every installed weight when the first
SRAM read is in cycle 0.

  schedule: every MXM plane installs `mxm.rows` rows through 16 streams (IW from streams, one row per stream, `mxm.rows / 16` issues per plane). The
    readers are the `min(32, MEM slices per hemisphere)` slices of each hemisphere nearest the MXM (quads Q - readers / 4 .. Q - 1, Q = MEM slices per
    hemisphere / 4), one stream each. With 32 readers the two planes of a hemisphere install at once (16 streams each); with 16 to 31 readers one plane
    after the other, plane 0 last; fewer than 16 readers cannot feed one IW and the profile is skipped. The farthest used quad is q0 = Q - readers / 4,
    whose stream reaches the MXM input after `mxm_in_gap - (mem_down_gap_base + q0)` gaps (16 on ref).
  derived bound: the cycle of the first ABC = (issues - 1) + mxm_iw.d_func + mem.d_func + transit, with
    issues = ceil(bytes / (2 x readers x vlanes)), bytes = 4 x rows x vlanes   (20 issues on ref: 409,600 / (2 x 32 x 320))
    = 19 + 1 + 5 + 16 = 41 on ref, one cycle above the published "less than 40" because MEM d_func is 5 (an implementation choice). The bound is
    below 40 exactly when the derived bound (tools/perf/derive_peaks.py, `install_first_abc_cycle`) is. The test measures the first ABC cycle, with plane
    0's last install carrying the only non-zero weights (an ABC that runs before that install lands reads a zero product) and asserts it equals the derived
    bound, on every profile with 16 or 32 readers (the shipped ref and mini, the test profiles ref_slow, ref_hw and sl8, a profile with IW 3 and MEM 3).
usage: published_bounds.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
cfg = json.load(open(CFG)); TM = cfg["timing"]; LAY = cfg["layout"]
SL = cfg["geometry"]["superlanes"]; VL = 16 * SL
STREAMS = cfg["geometry"]["streams_per_row"]; ROWS = cfg["mxm"]["rows"]
Q = cfg["geometry"]["mem_slices_per_hemisphere"] // 4
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]; D_IW = TM["mxm_iw"]["d_func"]
D_ABC = TM["mxm_abc"]["d_func"]; D_ACC = TM["mxm_acc"]["d_func"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


READERS = min(STREAMS, 4 * Q)                         # the slices of a hemisphere that read weights, one stream each
if READERS < 16:
    print(f"PASS all (skipped: an IW takes 16 streams from 16 slices and this profile has {4 * Q} MEM slices per hemisphere)"); sys.exit(0)
PLANES_AT_ONCE = 2 if READERS >= 32 else 1            # the planes of a hemisphere that install in the same cycle
USED = 16 * PLANES_AT_ONCE                            # the readers the schedule uses


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def mxm(op, t, s=0, mode=0, acc=0): return (3 << 28) | (op << 22) | (t << 20) | (s << 14) | (mode << 13) | (acc << 12)
def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q


RG = ROWS // 16                                       # IW issues of one plane
ISSUES = 2 * RG // PLANES_AT_ONCE                     # cycles with an IW issue (equal to ceil(4 x rows x vlanes / (2 x USED x vlanes)))
Q0 = Q - USED // 4
TRANSIT = LAY["mxm_in_gap"] - (LAY["mem_down_gap_base"] + Q0)
T = DM + TRANSIT                                      # the first install issue: its operands are read in cycle 0
BOUND = T + ISSUES - 1 + D_IW                         # the first ABC
SHIFT = 64                                            # every cycle of the program is moved by this much: a profile with few issues has reads before cycle 0


def measure(a):
    """The program with the ABC of plane 0 at cycle a; the product of plane 0 (lanes = rows) read back. Returns (diagnostics, product row 319 or the last row)."""
    sram, recs = {}, []
    for hemi, base, row in ((0, 0, "W"), (1, 44, "E")):
        flag = 32 if row == "W" else 0
        slices = [base + 4 * q + j for q in range(Q0, Q) for j in range(4)]
        for i, sl in enumerate(slices):
            g = gap_mem(sl, row)
            lat = DM + (LAY["mxm_in_gap"] - g)
            for k in range(ISSUES):
                last_plane0 = hemi == 0 and i < 16 and k == ISSUES - 1
                sram[(sl, k)] = bytes([1] * VL) if last_plane0 else bytes(VL)
                recs.append((T + k - lat, sl, 1, mem(0, k, flag + i), 0))
        for k in range(ISSUES):
            if PLANES_AT_ONCE == 2:                                                # both planes of the hemisphere, 16 streams each
                issues = ((2 * hemi, 0), (2 * hemi + 1, 16))
            else:                                                                  # one plane after the other, plane 2 h last
                issues = (((2 * hemi + 1, 2 * hemi)[k // RG], 0),)
            for p, grp in issues:
                recs.append((T + k, 104 + 3 * p, 1, mxm(1, 0, flag | grp, 1), 0))
    # the activation of plane 0: all ones on W:24, from east slice 44 (not a weight slice), at the MXM input in cycle a
    g = gap_mem(44, "W")
    sram[(44, 31)] = bytes([1] * VL)
    recs.append((a - DM - (LAY["mxm_in_gap"] - g), 44, 1, mem(0, 31, 32 + 24), 0))
    recs.append((a, 105, 1, mxm(2, 0, 32 | 24), 0))                                # abc.i8 W:24 on plane 0
    c3 = a + D_ABC
    recs.append((c3, 106, 1, mxm(3, 2, 8, 0, 1), 0))                               # acc.i32 E:8, replace, accumulator 1
    for b in range(4):                                                             # the four byte planes of the i32 product on E:8..11, stored by west slices 4b
        g = gap_mem(4 * b, "E")
        recs.append((c3 + D_ACC + (g - LAY["mxm_acc_gap"]) - MEM_DS, 4 * b, 1, mem(1, 9, 8 + b), 0))
    ls = [f"S {sl:x} {ad:x} " + " ".join(f"{x:x}" for x in v) for (sl, ad), v in sram.items()]
    ls += q_lines(lower_records([(c + SHIFT, q, wc, w0, w1) for (c, q, wc, w0, w1) in recs])) + [f"D {4 * b:x} 9" for b in range(4)]
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    words = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "W"}
    prod = [int.from_bytes(bytes(words[(4 * b, 9)][r] for b in range(4)), "little") for r in range(min(VL, ROWS))]
    return [l for l in p.stdout.splitlines() if l.startswith("T diag")], prod


rows_last = range(ROWS - 16, ROWS)                    # the rows installed by the last issue of plane 0
def sees_all(a):
    d, prod = measure(a)
    return d == [] and all(prod[r] == VL for r in rows_last if r < len(prod))
first = next((a for a in range(BOUND - 6, BOUND + 3) if sees_all(a)), None)
check(f"the first ABC that sees every installed weight is at cycle {BOUND} (= {ISSUES - 1} + {D_IW} + {DM} + {TRANSIT}; first read at cycle 0), measured {first}", first == BOUND, (first, BOUND))
check(f"an ABC one cycle earlier ({BOUND - 1}) does not see the last install", not sees_all(BOUND - 1), BOUND - 1)
published = 40
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "perf"))
import derive_peaks
derived = derive_peaks.derive(derive_peaks.load_profile(CFG))["install_first_abc_cycle"].value
if READERS in (16, 32):
    check(f"the measured bound {BOUND} equals the derived bound {derived} of tools/perf/derive_peaks.py", BOUND == derived, (BOUND, derived))
    check(f"the bound is {BOUND}: it meets the published 'less than {published} cycles' exactly when the derived bound {derived} does; recorded, not a defect",
          (BOUND < published) == (derived < published), (BOUND, derived))
else:
    print(f"note: {READERS} readers: derive_peaks counts {READERS} readers at once, the schedule 16 (one IW takes 16 streams): the bound {BOUND} is measured, not compared")
if DM == 5 and not os.environ.get("PUBLISHED_BOUNDS_VARIANT"):
    # the claim above, on a variant of this profile with MEM d_func 3 (a profile the schema admits): the same schedule, re-timed, meets the published bound
    import tempfile
    v = json.loads(json.dumps(cfg)); v["timing"]["mem"]["d_func"] = 3
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem3.json")
        json.dump(v, open(path, "w"))
        r = subprocess.run([sys.executable, os.path.abspath(__file__), SIM, path], capture_output=True, text=True, timeout=900,
                           env={**os.environ, "PUBLISHED_BOUNDS_VARIANT": "1"})
    check(f"a variant with MEM d_func 3 measures {BOUND - 2} < {published}: the published bound is met", r.returncode == 0 and f"measured {BOUND - 2}" in r.stdout, r.stdout[-300:] + r.stderr[-300:])
print("published_bounds:", "all pass" if ok else "FAILURES")
sys.exit(0 if ok else 1)
