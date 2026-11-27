"""Directed test of a relay over three chips : chip A sends a vector to chip B, B forwards it to chip C, C stores it.
`--chips K` (K <= 8) is fully connected, so a relay is the way a pair of chips that a larger wiring (the 64-chip system) does not join directly
talk through a third one. topology.md sec. 5: local link l of chip i reaches chip (i + l + 1) mod K on its link K - 2 - l (K = 3: link 0 of
chip i reaches chip i + 1 on its link 1, link 1 reaches chip i + 2 on its link 0).

The program (no MEM slice on B: the vector passes through the stream registers). A: a west slice reads the word onto E:0, the east TX queue sends it on
link 0 to B at cycle T. B: the RX queue receives it on link 1 at T + L (the link latency: the cycle it arrives) onto E:1, which is written at the RX gap
and reaches the TX gap `c2c_tx_gap - c2c_rx_gap` cycles later, where the east TX queue sends it on link 0 to C. C: receives on link 1 at the arrival and stores it.
The cost of a hop is the link latency; the relay adds `d_func(c2c) + (c2c_tx_gap - c2c_rx_gap)` cycles at B (ref: 1 + 33 = 34), so the end-to-end
latency is 2 L + 34 cycles on ref from A's Send to C's Receive (L = 216: 466), and a vector costs `c2c.cycles_per_vector` cycles of each link it crosses
(LNK-001: a second Send on a link fewer than cycles_per_vector cycles after the first sends nothing; OTS-E-LNK-003 is the same for a link a csr or ctrl packet occupies).
usage: relay_directed.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
cfg = json.load(open(CFG))
L = cfg["c2c"]["link_latency"]; CPV = cfg["c2c"]["cycles_per_vector"]
TX_GAP = cfg["layout"]["c2c_tx_gap"]; RX_GAP = cfg["layout"]["c2c_rx_gap"]
D_C2C = cfg["timing"]["c2c"]["d_func"]; DM = cfg["timing"]["mem"]["d_func"]
VL = 16 * cfg["geometry"]["superlanes"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def c2c(op, chip=0, s=0, link=0): return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)
SEND, RECEIVE = 0, 1
def E(i): return i
WRX, ETX = 128, 140
VALS = [bytes((7 * l + 3 + 50 * k) & 0xFF for l in range(VL)) for k in range(3)]


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def chip_text(recs, sram, dumps, host=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    return ls + q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(host)


def run3(chips):
    """chips: [(recs, sram, dumps, host)] x 3. One (diagnostics, words, T pkt lines) per chip."""
    t = []
    for k, c in enumerate(chips):
        t += [f"C {k:x}"] + chip_text(*c)
    p = subprocess.run([SIM, CFG, "--trace", "--chips", "3"], input="\n".join(t) + "\n", capture_output=True, text=True, timeout=600)
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


HOP = D_C2C + (TX_GAP - RX_GAP)               # B: the Receive's commit and the transit from the RX gap to the TX gap
T0 = DM + TX_GAP - gap_mem(0, "E") + 1        # A's first Send (its Read is at cycle >= 1)
def program(sends, relay_gap=None, ctrl=None):
    """sends: the cycles of A's Sends (vector k at sends[k]). B relays each one HOP cycles after its arrival; `relay_gap` overrides the spacing of B's relays."""
    a, sa = [], {}
    for k, c in enumerate(sends):
        feed(a, sa, 0, 1 + k, VALS[k], k, "E", TX_GAP, c)
        a.append((c, ETX, 1, c2c(SEND, 1, E(k), 0), 0))
    b, c_ = [], []
    for k, c in enumerate(sends):
        rb = c + L                                                         # arrival of vector k at B
        b.append((rb, WRX, 1, c2c(RECEIVE, 0, E(k), 1), 0))
        sb = rb + HOP if relay_gap is None else sends[0] + L + HOP + k * relay_gap
        b.append((sb, ETX, 1, c2c(SEND, 2, E(k), 0), 0))
        rc = sb + L
        c_.append((rc, WRX, 1, c2c(RECEIVE, 1, E(k), 1), 0))
        drain(c_, 0, 20 + k, k, "E", RX_GAP, rc + D_C2C)
    return [(a, sa, [], ()), (b, {}, [], ctrl or ()), (c_, {}, [(0, 20 + k) for k in range(len(sends))], ())]


# 1. One vector through the relay: A's Send at T0, B's Receive at T0 + L, B's Send HOP cycles later, C's Receive L later.
(da, _), (db, _), (dc, wc) = run3(program([T0]))
rc = T0 + L + HOP + L
check(f"A -> B -> C: the vector arrives at C and is stored; no diagnostic on any chip (A's Send at {T0}, C's Receive at {rc} = {rc - T0} cycles later)",
      da == db == dc == [] and wc[(0, 20)] == VALS[0], (da, db, dc, wc[(0, 20)][:4].hex()))
check(f"the end-to-end latency from A's Send to C's Receive is 2 x link_latency + {HOP} = {2 * L + HOP}", rc - T0 == 2 * L + HOP, (rc, T0))

# 2. B's Receive one cycle early is TIM-001 on B and nothing reaches C (the hop cost is exactly the link latency).
prog = program([T0])
prog[1][0][0] = (T0 + L - 1, WRX, 1, c2c(RECEIVE, 0, E(0), 1), 0)
(da, _), (db, _), (dc, wc) = run3(prog)
check("a Receive on B one cycle before the arrival is TIM-001; the relay Send then forwards nothing and C's Receive is TIM-001 as well",
      db[:1] == [f"T diag {T0 + L - 1} queue {WRX} OTS-E-TIM-001"] and any("TIM-001" in d for d in dc) and wc[(0, 20)] != VALS[0], (db, dc))

# 3. Two vectors: A sends them cycles_per_vector apart (the link's rate), B relays at the same spacing: both arrive at C, in order.
(da, _), (db, _), (dc, wc) = run3(program([T0, T0 + CPV]))
check(f"two vectors {CPV} cycles apart (the link rate, LNK-001 boundary) are relayed in order: no LNK-001 on A or B",
      da == db == dc == [] and wc[(0, 20)] == VALS[0] and wc[(0, 21)] == VALS[1], (da, db, dc, wc[(0, 21)][:4].hex()))

# 4. LNK-001 on the relay's own link: A sends at the link rate, B forwards the second vector one cycle too soon after the first.
(da, _), (db, _), (dc, wc) = run3(program([T0, T0 + CPV], relay_gap=CPV - 1))
send2 = T0 + L + HOP + CPV - 1
check(f"B's second relay Send {CPV - 1} cycles after its first is LNK-001 on B (cycle {send2}); the vector is not sent and C's second Receive is TIM-001",
      db == [f"T diag {send2} queue {ETX} OTS-E-LNK-001"] and wc[(0, 21)] != VALS[1] and any("TIM-001" in d for d in dc), (db, dc))

# 5. LNK-003 on B's outgoing link: a ctrl packet that left B's link 0 at cycle h still occupies it, so the relay Send released into it sends nothing.
FV = 1 + (VL + 7) // 8                        # flits of a vector (a header and vlanes / 8 payload ones); a ctrl packet is 2 flits
KX = -(-CPV * 2 // FV)                        # the cycles a ctrl packet occupies a link that carries a vector in CPV cycles
h = T0 + L + HOP - (KX - 1)                   # asked so that its last occupied cycle is the relay Send's
(da, _), (db, _), (dc, wc) = run3(program([T0], ctrl=[f"P ctrl {h:x} 0"]))
check(f"a ctrl packet leaving B's link 0 at cycle {h} (it occupies {KX} cycle(s)) makes the relay Send at {T0 + L + HOP} OTS-E-LNK-003; nothing reaches C",
      db == [f"T diag {T0 + L + HOP} queue {ETX} OTS-E-LNK-003"] and wc[(0, 20)] != VALS[0], (db, dc))

print("relay:", "all pass" if ok else "FAILURES")
sys.exit(0 if ok else 1)
