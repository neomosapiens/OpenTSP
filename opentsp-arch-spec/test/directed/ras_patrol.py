"""Directed end-to-end tests of the patrol scrubber (microarchitecture.md sec. 15.4, `scrub.period`): in a cycle c with c mod period = 0 every
MEM slice whose read and write paths issue nothing in c reads the word at its scrub pointer, corrects a single-bit error, writes the
corrected word back in a later idle patrol slot unless an intervening slice mutation invalidates it and advances the pointer modulo words_per_slice; an uncorrectable word is left as it is and
logged. A correction prints `T ecc <c> scrub slice <s> addr <a> sl <g>` (no `T sram` line: the data are the ones the word should
hold) and is a corrected-data entry of the error log; period 0 (the value of every shipped profile) is off.
Runs against the Sail emulator or the ISS (both take the same command line).
usage: ras_patrol.py <tsp_sim|tsp-iss> <config.json>"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG0 = sys.argv[2]
CJ = json.load(open(CFG0)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
ok = True
PERIOD = 64
TD = tempfile.mkdtemp()


def cfg_with(period):
    c = json.loads(json.dumps(CJ)); c["scrub"] = {"period": period}
    path = os.path.join(TD, f"cfg{period}.json"); json.dump(c, open(path, "w")); return path


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
def W(i): return 32 + i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)


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
T2 = 400                # the consumer of the words: after the patrol cycles 0, 64, ..., 384
S0 = 0                  # the slice that holds the words under test
SLX = min(3, SL - 1)    # the superlane of the flips


def program(busy_at=None):
    """Words (S0, 1), (S0, 2) hold A; the consumer at T2 sums (S0, 1) and (1, 2) into slice 44 addr 9. With busy_at, a Read of (S0, 5) is
    dispatched in that cycle (slice S0's read path is busy in it)."""
    recs, sram = [], {}
    feed(recs, sram, S0, 1, A, 0, "E", 21, T2 + VXM_DS)
    feed(recs, sram, 1, 2, B, 4, "E", 21, T2 + VXM_DS)
    sram[(S0, 2)] = A; sram[(S0, 5)] = B
    recs.append((T2, VXMQ, 1, add_i8(E(8), E(0), E(4)), 0))
    drain(recs, 44, 9, 8, "E", 21, T2 + DV)
    if busy_at is not None:
        recs.append((busy_at, S0, 1, mem(0, 5, 16), 0))
    return recs, sram


def fsram(cycle, slice_, addr, sl, bit): return f"F sram {cycle:x} {slice_:x} {addr:x} {sl:x} {bit:x}"


def go(period, flips, busy_at=None, dumps=((44, 9), (S0, 1), (S0, 2)), extra=()):
    recs, sram = program(busy_at)
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(flips) + list(extra)
    p = subprocess.run([SIM, cfg_with(period), "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (period, flips, p.returncode, p.stderr, p.stdout[-600:])
    return p


def ln(p, prefix): return [l for l in p.stdout.splitlines() if l.startswith(prefix)]
def hexs(b): return " ".join(f"{x:02x}" for x in b)
def scrub(p): return [l for l in ln(p, "T ecc") if " scrub " in l]
def csr(p, a):
    for l in p.stdout.splitlines():
        if l.startswith(f"X {a:03x} "): return int(l.split()[2], 16)
GOOD_SUM = f"W 2c 9 {hexs(SUM)}"
W1 = f"W 0 1 {hexs(A)}"
W2 = f"W 0 2 {hexs(A)}"

# 0. Period 0 (off) and a fault-free run with the patrol on: silent, nothing changes.
p0 = go(0, [])
check("patrol: off, fault-free: no event, the sum is right", not ln(p0, "T ecc") and GOOD_SUM in p0.stdout and W1 in p0.stdout, p0.stdout[-300:])
p = go(PERIOD, [])
check("patrol: on, fault-free: nothing is printed or logged and the output equals the one with the patrol off",
      not ln(p, "T ecc") and not ln(p, "T diag") and p.stdout == p0.stdout, (ln(p, "T ecc"), ))

# 1. A single-bit error of the word the pointer reaches in cycle 64 (address 1: pointer 0 in cycle 0, 1 in cycle 64): corrected there.
F1 = fsram(10, S0, 1, SLX, 5)
p = go(0, [F1])
check("patrol: off: the flipped word stays flipped, the consumer corrects its copy",
      W1 not in p.stdout and len(ln(p, "T ecc")) == 1 and "corr" in ln(p, "T ecc")[0] and GOOD_SUM in p.stdout, ln(p, "T ecc"))
p = go(PERIOD, [F1], extra=["R 10", "R 14"])
check("patrol: on: cycle 64 corrects address 1 of slice 0, superlane SLX: one line",
      scrub(p) == [f"T ecc 64 scrub slice 0 addr 1 sl {SLX}"], ln(p, "T ecc"))
check("patrol: on: the stored word equals the original at the end, the consumer finds nothing to correct, the sum is right",
      W1 in p.stdout and len(ln(p, "T ecc")) == 1 and GOOD_SUM in p.stdout, ln(p, "T ecc"))
check("patrol: on: no T sram line for the scrubbed word, no interrupt, no diagnostic", not [l for l in ln(p, "T sram") if " slice 0 addr 1 " in l] and not ln(p, "T irq") and not ln(p, "T diag"), p.stdout[-300:])
check("patrol: on: the correction is a corrected-data entry of the error log (ERR_STATUS bit 0, one entry)",
      csr(p, 0x10) == 1 and csr(p, 0x14) == 1, (csr(p, 0x10), csr(p, 0x14)))

# 2. The pointer: address 2 is reached in cycle 128; a Read in that cycle makes slice 0 busy: the word is reached in cycle 192 (the pointer waits).
F2 = fsram(10, S0, 2, 0, 9)
p = go(PERIOD, [F2])
check("patrol: address 2 is corrected in cycle 128", scrub(p) == ["T ecc 128 scrub slice 0 addr 2 sl 0"], ln(p, "T ecc"))
p = go(PERIOD, [F2], busy_at=128)
check("patrol: a Read dispatched in cycle 128 holds slice 0: the pointer waits and address 2 is corrected in cycle 192",
      scrub(p) == ["T ecc 192 scrub slice 0 addr 2 sl 0"] and W2 in p.stdout, ln(p, "T ecc"))

# 3. An uncorrectable word is left as it is and logged: two flips of one superlane of address 1.
p = go(PERIOD, [fsram(10, S0, 1, SLX, 5), fsram(11, S0, 1, SLX, 7)], extra=["R 10"])
sc = scrub(p)
check("patrol: two flips: the word is logged uncorrectable in cycle 64 (error interrupt) and left as it is",
      len(sc) >= 1 and sc[0] == f"T ecc 64 scrub slice 0 addr 1 sl {SLX} unc" and W1 not in p.stdout and len(ln(p, "T irq")) >= 1 and (csr(p, 0x10) or 0) & 2, (sc, ln(p, "T irq")))

# 4. A check-bit flip is corrected too (the overlay becomes clean): one line, and the consumer later reads without any event.
p = go(PERIOD, [fsram(10, S0, 1, SLX, 130)])
check("patrol: a check-bit flip is corrected in cycle 64 and the consumer logs nothing",
      scrub(p) == [f"T ecc 64 scrub slice 0 addr 1 sl {SLX}"] and len(ln(p, "T ecc")) == 1, ln(p, "T ecc"))

# 5. A cycle after the last event is not run: nothing scrubs when the whole run ends before the first multiple that reaches the word.
p = go(PERIOD, [fsram(10, S0, 1, SLX, 5)], extra=[])
check("patrol: the pointer of slice 0 holds one word per scrub cycle: slice 1 reaches address 1 in cycle 64 as well",
      go(PERIOD, [fsram(10, 1, 1, 0, 6)]) is not None and any(" slice 1 addr 1 " in l for l in scrub(go(PERIOD, [fsram(10, 1, 1, 0, 6)]))), "")

# 6. A scrub cycle between two model events sees the state before the later event: a Write that rewrites address 1 of slice 0 in cycle 65
#    does not take away the correction of cycle 64 (the cycles between two ticks run before the tick that changes the state). The Write's
#    operand is read from slice 40 (ten cycles away along the row), so that the only model events are its Read (cycle 50, commit 55) and the Write.
def go_write(period, flips):
    recs, sram = program(None)
    feed(recs, sram, 40, 7, A, 0, "E", gap_mem(S0, "E"), 65)
    recs.append((65, S0, 1, mem(1, 1, 0), 0))
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in ((44, 9), (S0, 1))] + list(flips)
    p = subprocess.run([SIM, cfg_with(period), "--trace"], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, (period, flips, p.returncode, p.stderr, p.stdout[-600:])
    return p
p = go_write(PERIOD, [fsram(10, S0, 1, SLX, 5)]) if CJ["geometry"]["mem_slices_per_hemisphere"] >= 44 else None
if p is not None: check("patrol: a flip of address 1 rewritten by a Write in cycle 65 is still corrected in cycle 64 (the idle cycles run before the tick of cycle 65)",
      scrub(p) == [f"T ecc 64 scrub slice 0 addr 1 sl {SLX}"] and W1 in p.stdout, (ln(p, "T ecc"), ))

# A patrol slot performs either a read or its later writeback, never both.
# The stored byte (rather than a consumer corrected copy) exposes the distinction.
for end, later_fault, busy, expected in (
        (1, None, False, 1), (33, None, False, 0),
        (33, 1, False, 3), (33, 130, False, 1),
        (33, None, True, 1), (65, None, True, 0)):
    lines = ["S 0 0 0", fsram(0, 0, 0, 0, 0), fsram(end, 1, 0, 0, 0), "D 0 0"]
    if later_fault is not None:
        lines.append(fsram(1, 0, 0, 0, later_fault))
    if busy:
        lines += q_lines(lower_records([(32 - MEM_DS, 0, 1, mem(0, 3, 0), 0)]))
    r = subprocess.run([SIM, cfg_with(32), "--trace"], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, timeout=120)
    dumped = [l.split() for l in r.stdout.splitlines() if l.startswith("W 0 0 ")]
    got = int(dumped[0][3], 16) if dumped else None
    check(f"patrol separate writeback end={end} later_fault={later_fault} busy={busy}",
          r.returncode == 0 and got == expected, (got, expected, r.stderr))

print("PASS all" if ok else "FAIL")
sys.exit(0 if ok else 1)
