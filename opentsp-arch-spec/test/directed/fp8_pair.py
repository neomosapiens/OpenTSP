"""Directed test of the FP8 pair rule on a profile with `mxm.fp8_rate` 2 (microarchitecture.md sec. 6, the FP8 rate 2 rule): an FP8
pair (two ABCs of one plane in one cycle), then optionally one more ABC on the same plane in the cycle the pair commits, then ACC 0
and ACC 1 one cycle later. The newest ABC result committed by the ACC's cycle is the pair (the new ABC has not committed), so
accumulator 0 takes the lower queue's product (VL x 1.0) and accumulator 1 the higher queue's (VL x 2.0). The Python oracle and the
simulator under test must both give the documented values, with and without the third ABC.

usage: fp8_pair.py <tsp_sim|tsp-iss> <config.json>
"""
import json, os, struct, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..", "tools")); sys.path.insert(0, os.path.join(here, ".."))
from tsp_as import lower_records, q_lines
from oracle import machine as mc
from oracle.diff_oracle import images_of

SIM, CFG = sys.argv[1], sys.argv[2]
CJ = json.load(open(CFG))
TM = CJ["timing"]; VL = 16 * CJ["geometry"]["superlanes"]
DM, D_ABC, D_ACC = TM["mem"]["d_func"], TM["mxm_abc"]["d_func"], TM["mxm_acc"]["d_func"]
assert CJ["mxm"]["fp8_rate"] == 2, "a profile with mxm.fp8_rate 2"
ok = True


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


def drains_of(recs, base_cycle, dst_streams, slices=(0, 4, 8, 12), addr=9):
    dumps = []
    for b in range(4):
        drain(recs, slices[b], addr, dst_streams + b, "E", 2, base_cycle)
        dumps.append((slices[b], addr))
    return dumps


QI, QA, QC = 104, 105, 106
T = 200
E4M3_ONE, E4M3_TWO, E4M3_FOUR = 0x38, 0x40, 0x48
f32 = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]


def lane0(w, ds): return int.from_bytes(bytes(w.get(d, bytes(VL))[0] for d in ds), "little")


def program(third_abc):
    sram, recs = {}, []
    for r in range(16):
        feed(recs, sram, 44 + 4 * (r % 11) + (r // 11), 3 + r, bytes([E4M3_ONE] * VL), r, "W", 43, T)
    recs.append((T, QI, 1, mxm(1, 3, 32 | 0, 1), 0))                                  # iw.fp8 W:0
    c = T + 2
    for n, q in enumerate((QA, QC)):                                                   # the pair: QA x 1.0, QC x 2.0
        feed(recs, sram, 60 + n, 5, bytes([E4M3_ONE if n == 0 else E4M3_TWO] * VL), 24 + n, "W", 43, c)
        recs.append((c, q, 1, mxm(2, 3, 32 | (24 + n), 0), 0))                         # abc.e4m3 W:24+n
    c_commit = c + D_ABC
    if third_abc:                                                                      # one more ABC on the plane, in the pair's commit cycle
        feed(recs, sram, 62, 5, bytes([E4M3_FOUR] * VL), 26, "W", 43, c_commit)
        recs.append((c_commit, QI, 1, mxm(2, 3, 32 | 26, 0), 0))
    c_acc = c_commit + 1                                                               # before the new ABC commits
    recs.append((c_acc, QA, 1, mxm(3, 3, 8, 0, 0), 0))                                 # acc.f32 E:8, accumulator 0, replace
    recs.append((c_acc, QC, 1, mxm(3, 3, 12, 0, 1), 0))                                # acc.f32 E:12, accumulator 1, replace
    dumps = drains_of(recs, c_acc + D_ACC, 8) + drains_of(recs, c_acc + D_ACC, 12, slices=(16, 20, 24, 28))
    return sram, recs, dumps


want = (f32(1.0 * VL), f32(2.0 * VL))
for third in (False, True):
    sram, recs, dumps = program(third)
    what = "with" if third else "without"
    m = mc.Machine(CJ)
    ev = m.run(images_of(recs), dict(sram))
    w = {(e[2], e[3]): e[4][:VL] for e in ev if e[0] == "sram"}
    d = [e for e in ev if e[0] == "diag" and e[3] != "OTS-E-TIM-001"]
    got = (lane0(w, dumps[:4]), lane0(w, dumps[4:]))
    check(f"oracle {what} an ABC in the pair's commit cycle: ACC 0 is the lower queue's product, ACC 1 the higher queue's",
          not d and got == want, (d, [hex(x) for x in got]))
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    lines += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, (p.returncode, p.stderr, p.stdout[-300:])
    ws = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "W"}
    sd = [l for l in p.stdout.splitlines() if l.startswith("T diag") and "TIM-001" not in l]
    sgot = (lane0(ws, dumps[:4]), lane0(ws, dumps[4:]))
    check(f"{os.path.basename(SIM)} {what} an ABC in the pair's commit cycle: the oracle's products",
          not sd and sgot == want and sgot == got, (sd, [hex(x) for x in sgot], [hex(x) for x in got]))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
