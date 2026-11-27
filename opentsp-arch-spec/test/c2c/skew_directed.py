"""Directed tests of multi-chip start skew : what a program that is scheduled at fixed cycles gets when the chips do not
start together, and the pattern that does not depend on the start. The model starts every chip at cycle 0 unless the host script says
`H start <cycle>` (tsp_sim.c); the real host starts the chips together (platform.md sec. 1) and the specification does not claim more.

1. Fixed offsets. Chip 0 sends at cycle 19 on ref (west slice 0 reads W:0 at 0, the C2C TX queue sends it on link 0 to chip 1); chip 1 receives at
   19 + link_latency_max (the generator rule of the toolchain: the worst case of the link) and stores the vector 14 cycles later.
   With aligned starts, and with chip 0 starting up to link_latency_max - link_latency cycles late (the margin), the vector is stored. One more
   cycle of start skew and the Receive finds no vector: TIM-001 at the Receive (ref: 247) and at the Write that stores it (ref: 261).
   A late receiver is harmless (the vector waits in the FIFO).
2. The Deskew pattern (ISCA22 sec. 3.2, Fig. 7(b); the polling loop of DeskewRecv): the parent runs Deskew, then Transmit at the epoch boundary
   after it; the child runs DeskewRecv, which takes the vector at the first epoch boundary at or after its arrival. Both are right for every
   start skew below the epoch; the stored vector and the cycle of the store follow from the boundaries only.
usage: skew_directed.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
cfg = json.load(open(CFG))
L = cfg["c2c"]["link_latency"]; LMAX = cfg["c2c"]["link_latency_max"]; P = cfg["c2c"]["hac_period"]
TX_GAP = cfg["layout"]["c2c_tx_gap"]; RX_GAP = cfg["layout"]["c2c_rx_gap"]
D_C2C = cfg["timing"]["c2c"]["d_func"]; DM = cfg["timing"]["mem"]["d_func"]
VL = 16 * cfg["geometry"]["superlanes"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def c2c(op, chip=0, s=0, link=0): return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)
SEND, TRANSMIT, DESKEW = 0, 3, c2c(2)
def deskewrecv(chip, link, s): return c2c(9, chip, s, link)
def W(i): return 32 + i
WTX, WRX = 126, 128                   # a west C2C transmit and receive queue
VAL = bytes((7 * l + 3) & 0xFF for l in range(VL))
NOTE = bytes([1]) + bytes(VL - 1)


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def chip_text(recs, sram, dumps, start):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    return ls + ([f"H start {start:x}"] if start else [])


def run2(chips):
    """chips: [(recs, sram, dumps, start)] x 2. One (diagnostics, words) per chip."""
    t = []
    for k, c in enumerate(chips):
        t += [f"C {k:x}"] + chip_text(*c)
    p = subprocess.run([SIM, CFG, "--trace", "--chips", "2"], input="\n".join(t) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr
    parts, cur = [], None
    for l in p.stdout.splitlines():
        if l.startswith("C "): cur = []; parts.append(cur)
        else: cur.append(l)
    out = []
    for part in parts:
        words = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in part) if t and t[0] == "W"}
        out.append(([l for l in part if l.startswith("T diag")], words))
    return out


# 1. Fixed offsets.
T = DM + TX_GAP - gap_mem(0, "W")                    # the earliest Send whose operand the Read at cycle 0 delivers (19 on ref)
RX = T + LMAX
def fixed(s0, s1):
    r0, sr0 = [], {}
    feed(r0, sr0, 0, 1, VAL, 0, "W", TX_GAP, T)
    r0.append((T, WTX, 1, c2c(SEND, 1, W(0)), 0))
    r1 = [(RX, WRX, 1, c2c(1, 0, 0), 0)]                                  # Receive chip 0, E:0
    drain(r1, 0, 5, 0, "E", RX_GAP, RX + D_C2C)
    store = max(c for c, *_ in r1)
    (d0, _), (d1, w1) = run2([(r0, sr0, [], s0), (r1, {}, [(0, 5)], s1)])
    return d0, d1, w1.get((0, 5), b""), store
margin = LMAX - L
for s0, s1 in ((0, 0), (margin, 0), (0, 30), (0, 1000)):
    d0, d1, w, store = fixed(s0, s1)
    check(f"fixed offsets, starts ({s0}, {s1}): the vector is received and stored", d0 == [] and d1 == [] and w == VAL, (d0, d1, w[:4].hex()))
for s0 in (margin + 1, 30, 100):
    d0, d1, w, store = fixed(s0, 0)
    check(f"fixed offsets, chip 0 starts {s0} cycles late (margin {margin}): TIM-001 at the Receive ({RX}) and at the Write ({store}), nothing stored",
          d0 == [] and d1 == [f"T diag {RX} queue {WRX} OTS-E-TIM-001", f"T diag {store} queue 0 OTS-E-TIM-001"] and w != VAL, (d1, w[:4].hex()))
d0, d1, w, store = fixed(30, 0)
check("the stamps with send 19 and link_latency_max 228 (ref) are 247 and 261", (T, LMAX) != (19, 228) or (RX, store) == (247, 261), (RX, store))


# 2. The Deskew pattern: independent of the start skew.
def boundary_after(c):
    """The first epoch boundary after cycle c (a Deskew dispatched at c is released there)."""
    return (c // P + 1) * P
def deskewed(s0, s1):
    t3 = boundary_after(s0 + 10)                                           # the parent's Deskew is dispatched at s0 + 10
    arrive = t3 + L
    exit_at = boundary_after(s1 + 10)                                      # the child's DeskewRecv polls at the first boundary after its dispatch ...
    while exit_at < arrive:                                                # ... and at each one after, until the vector is there
        exit_at += P
    parent = [(10, WTX, 1, DESKEW, 0), (t3 - s0, WTX, 1, c2c(TRANSMIT, 1, 0, 0), 0)]
    child = [(10, WRX, 1, deskewrecv(0, 0, 0), 0)]
    drain(child, 0, 5, 0, "E", RX_GAP, exit_at - s1 + D_C2C)                 # text cycles are local to the chip: global cycle - its start
    (d0, _), (d1, w1) = run2([(parent, {}, [], s0), (child, {(0, 5): bytes([0x5A]) * VL}, [(0, 5)], s1)])
    return d0, d1, w1.get((0, 5), b""), exit_at
for s0, s1 in ((0, 0), (30, 0), (0, 30), (100, 7), (1000, 0), (0, 1000)):
    d0, d1, w, exit_at = deskewed(s0, s1)
    check(f"Deskew pattern, starts ({s0}, {s1}): the child takes the notification at the boundary {exit_at}, no diagnostic", d0 == [] and d1 == [] and w == NOTE, (d0, d1, w[:4].hex()))

print("skew:", "all pass" if ok else "FAILURES")
sys.exit(0 if ok else 1)
