"""Directed end-to-end tests of the software SRAM scrub (microarchitecture.md sec. 15.4): the sequence MEM Read a,
VXM `max` of the stream with itself, MEM Write a removes a stored single-bit error, because the MEM slice does not check
the ECC on a Read (it forwards the stored check bits) and the VXM checks the operand, corrects a single-bit error in its
copy and generates fresh check bits. Without the scrub a second upset of the same word makes it uncorrectable.
Runs against the Sail emulator or the ISS (both take the same command line). Profile-portable: ref, mini and the test profiles.
usage: ras_scrub.py <tsp_sim|tsp-iss> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
_rows = {o["entry"]: (o["d_func"], o["d_skew"]) for o in TM["overrides"]}
MDF, MDS = _rows.get("Max.i8") or _rows.get("Max") or (DV, VXM_DS)      # max.i8: its row keyed by the type, else the entry's, else the class row
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
def W(i): return 32 + i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)
def max_i8(d, a, b): return VXM(9, 0, d, a, b)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap) - MEM_DS, slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


A = bytes((13 * l + 5) & 0xFF for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
SUM = bytes((x + y) & 0xFF for x, y in zip(A, B))
VXMQ = 88
T1 = 60                 # the scrub
T2 = T1 + 120           # the consumer of the scrubbed word
SLICE, ADDR = 0, 1      # the word: A


def program(scrub):
    """Phase 1 (T1, only with scrub): Read (0,1) -> E:0, max.i8 W:8 = E:0, E:0, Write W:8 -> (0,1). Phase 2 (T2): A (the word) and B
    on E:0 and E:4, summed by the VXM on E:8 into slice 44 addr 9."""
    recs, sram = [], {}
    if scrub:
        feed(recs, sram, SLICE, ADDR, A, 0, "E", 21, T1 + MDS)
        recs.append((T1, VXMQ, 1, max_i8(W(8), E(0), E(0)), 0))
        drain(recs, SLICE, ADDR, 8, "W", 21, T1 + MDF)
    feed(recs, sram, SLICE, ADDR, A, 0, "E", 21, T2 + VXM_DS)
    feed(recs, sram, 1, 2, B, 4, "E", 21, T2 + VXM_DS)
    recs.append((T2, VXMQ, 1, add_i8(E(8), E(0), E(4)), 0))
    drain(recs, 44, 9, 8, "E", 21, T2 + DV)
    return recs, sram


def fsram(cycle, slice_, addr, sl, bit): return f"F sram {cycle:x} {slice_:x} {addr:x} {sl:x} {bit:x}"


def go(scrub, flips, dumps=((44, 9), (SLICE, ADDR)), opts=(), expect=0):
    recs, sram = program(scrub)
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(flips)
    p = subprocess.run([SIM, CFG, "--trace", *opts], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == expect, (scrub, flips, p.returncode, p.stderr, p.stdout[-600:])
    return p


def ln(p, prefix): return [l for l in p.stdout.splitlines() if l.startswith(prefix)]
def hexs(b): return " ".join(f"{x:02x}" for x in b)
GOOD_SUM = f"W 2c 9 {hexs(SUM)}"
GOOD_WORD = f"W 0 1 {hexs(A)}"
F1 = fsram(0, SLICE, ADDR, 0, 5)              # the first upset, before the scrub (data bit 5 of superlane 0)
F2 = fsram(T1 + 60, SLICE, ADDR, 0, 9)        # the second upset of the same word, after the scrub and before its consumer

# 0. Fault-free: the scrub sequence is silent and changes nothing.
p = go(True, [])
check("scrub: fault-free: no ECC event, no diagnostic, the word and the sum are unchanged",
      not ln(p, "T ecc") and not ln(p, "T diag") and GOOD_SUM in p.stdout and GOOD_WORD in p.stdout, p.stdout[-500:])

# 1. One upset, no scrub: the consumer corrects its copy, the stored word stays flipped.
p = go(False, [F1])
check("scrub: one upset, no scrub: the consumer corrects (one corrected event) and the sum is right",
      len(ln(p, "T ecc")) == 1 and "corr" in ln(p, "T ecc")[0] and GOOD_SUM in p.stdout and not ln(p, "T irq"), ln(p, "T ecc"))
check("scrub: one upset, no scrub: the stored word is still flipped at the end", GOOD_WORD not in p.stdout, "")

# 2. Two upsets of the word, no scrub: uncorrectable at the consumer.
p = go(False, [F1, F2])
check("scrub: two upsets, no scrub: uncorrectable at the consumer (error interrupt) and the sum is wrong",
      len(ln(p, "T irq")) == 1 and any("unc" in l for l in ln(p, "T ecc")) and GOOD_SUM not in p.stdout, (ln(p, "T ecc"), ln(p, "T irq")))

# 3. Upset, scrub, second upset: the scrub removed the first, so the second is a single error and the sum is right.
p = go(True, [F1, F2])
ecc = ln(p, "T ecc")
c0 = [int(l.split()[2]) for l in ecc]
at_scrub = [c for c in c0 if c < T1 + 60]; at_consumer = [c for c in c0 if c >= T2]
check("scrub: upset, scrub, second upset: the scrub's VXM reads its operand twice (two corrected events in its cycle), the consumer finds one, none uncorrectable, no interrupt",
      len(at_scrub) == 2 and len(at_consumer) == 1 and len(c0) == 3 and all("corr" in l and "unc" not in l for l in ecc) and not ln(p, "T irq"), ecc)
check("scrub: upset, scrub, second upset: the sum is right", GOOD_SUM in p.stdout, p.stdout[-400:])

# 4. Upset and scrub, nothing after: the word is clean again at the end (the data bit is restored).
p = go(True, [F1])
check("scrub: upset then scrub: the scrub's two operand reads correct, and the stored word equals the original at the end",
      len(ln(p, "T ecc")) == 2 and GOOD_WORD in p.stdout and GOOD_SUM in p.stdout and not ln(p, "T irq"), ln(p, "T ecc"))

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
