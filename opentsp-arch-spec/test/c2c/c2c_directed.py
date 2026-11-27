"""Directed end-to-end tests of chip-to-chip links (slices/c2c/c2c_insts.sail) and of the
harness contract (c_emulator/tsp_sim.c, postlude/harness_api.sail) on the Sail emulator.

C2C: Send/Receive over a looped-back link with the configured latency (ISCA 2022 Table 2),
Receive before arrival, Transmit's notification vector, the link field (c2c.links links,
ISCA 2020 sec. II), chip_id checks, Deskew to the HAC epoch boundary (252 cycles, ISCA 2022
footnote 4) on any queue, RuntimeDeskew t, per-link latency in [209, 228] with a Deskew-
synchronised program that is deterministic for every latency, and the node topology
(--chips K: 8 fully connected chips, ISCA 2022 sec. 2.2; nodes joined by global links).
Harness: input validation (Q lines), effects after a long AGT countdown, idle-cycle skipping,
--max-cycles, many dumps. Profile-portable: vectors are
vlanes = 16 x superlanes bytes and only those bytes are compared.
usage: c2c_directed.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines     # the normative lowering (assembly.md sec. 4)
SIM = sys.argv[1]; CFG = sys.argv[2]
cfg = json.load(open(CFG))
L = cfg["c2c"]["link_latency"]; P = cfg["c2c"]["hac_period"]
LMIN = cfg["c2c"]["link_latency_min"]; LMAX = cfg["c2c"]["link_latency_max"]
LINKS = cfg["c2c"]["links"]
TX_GAP = cfg["layout"]["c2c_tx_gap"]; RX_GAP = cfg["layout"]["c2c_rx_gap"]
D_C2C = cfg["timing"]["c2c"]["d_func"]
VL = 16 * cfg["geometry"]["superlanes"]
assert LMAX < P, "Deskew synchronisation needs every link latency below the HAC period"


def lines(recs, sram={}, dumps=[]):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    return ls + q_lines(lower_records(recs)) + [f"D {sl:x} {a:x}" for sl, a in dumps]


def sim(text, *args, timeout=600, cfg_path=None):
    return subprocess.run([SIM, cfg_path or CFG, "--trace", *args], input=text, capture_output=True, text=True, timeout=timeout)


def parse(o):
    words = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in o.splitlines()) if t and t[0] == "W"}
    return [l for l in o.splitlines() if l.startswith("T diag")], words


def run(recs, sram={}, dumps=[], args=(), cfg_path=None):
    p = sim("\n".join(lines(recs, sram, dumps)) + "\n", *args, cfg_path=cfg_path)
    assert p.returncode == 0, p.stderr
    return (p.stdout,) + parse(p.stdout)


def run_chips(progs, args=()):
    """progs: one (recs, sram, dumps) per chip. Returns one (diags, words) per chip."""
    text = []
    for k, prog in enumerate(progs):
        text += [f"C {k:x}"] + lines(*prog)
    p = sim("\n".join(text) + "\n", "--chips", str(len(progs)), *args)
    assert p.returncode == 0, p.stderr
    parts, cur = [], None
    for l in p.stdout.splitlines():
        if l.startswith("C "): cur = []; parts.append(cur)
        else: cur.append(l)
    return [parse("\n".join(x)) for x in parts]


ok = True
def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def agt(op, v): return (1 << 60) | (op << 54) | ((v & 0xFFFF) << 16)
# C2C 32-bit layout: slice 5, opcode 27..22, chip_id 21..16, stream 15..10, link 9..6.
def c2c(op, chip=0, s=0, link=0): return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)
SEND, RECEIVE, TRANSMIT = 0, 1, 3
DESKEW = c2c(2)
def rtdeskew(t): return (5 << 28) | (4 << 22) | t
def nop(n): return n
def E(i): return i
def W(i): return 32 + i
# C2C queues (topology.md sec. 5): 2 TX and 2 RX in each 14-queue SXM block.
WTX, WRX, ETX, ERX, VXMQ = 126, 128, 140, 142, 88
WTX1, WRX1, ETX1, ERX1 = 127, 129, 141, 143
NOTE = bytes([1]) + bytes(VL - 1)


# Station gaps (topology.md sec. 4): west MEM quad q: E at 18-q, W at 24+q; east MEM quad q:
# W at 18-q, E at 24+q. Transit = reader gap - writer gap; MEM d_func 5.
def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - 5 - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


VAL = bytes((7 * l + 3) & 0xFF for l in range(VL))

# 1. West Send on link 0 -> loop back -> east Receive on link 0, L cycles later, intact. West
#    TX reads row W at TX_GAP; east slice 44 (quad 0) writes W at 18. East RX writes row W.
T = 100
recs, sram = [], {}
feed(recs, sram, 44, 1, VAL, 0, "W", TX_GAP, T)
recs.append((T, WTX, 1, c2c(SEND, 0, W(0)), 0))
recs.append((T + L, ERX, 1, c2c(RECEIVE, 0, W(1)), 0))
drain(recs, 44, 50, 1, "W", RX_GAP, T + L + D_C2C)
o, d, wd = run(recs, sram, [(44, 50)])
check(f"Send/Receive loop back west->east after link_latency {L}", d == [] and wd[(44, 50)] == VAL, (d, wd[(44, 50)][:8].hex()))

# 1b. each SXM block has two C2C TX and two RX queues; the second ones work alike.
recs1b = [(r[0], {WTX: WTX1, ERX: ERX1}.get(r[1], r[1])) + r[2:] for r in recs]
o, d, wd = run(recs1b, sram, [(44, 50)])
check(f"the second TX ({WTX1}) and RX ({ERX1}) queues send and receive like the first", d == [] and wd[(44, 50)] == VAL, (d, wd[(44, 50)][:8].hex()))

# 2. Receive one cycle before arrival: OTS-E-TIM-001, nothing written; the vector stays queued.
recs2 = recs[:2] + [(T + L - 1, ERX, 1, c2c(RECEIVE, 0, W(1)), 0), (T + L, ERX, 1, c2c(RECEIVE, 0, W(1)), 0)] + recs[3:]
o, d, wd = run(recs2, sram, [(44, 50)])
check("Receive before arrival is TIM-001; the next Receive gets the vector", d == [f"T diag {T + L - 1} queue {ERX} OTS-E-TIM-001"] and wd[(44, 50)] == VAL, d)

# 2b. The link field: a vector sent on link 9 is received on link 9 (one chip: every link
#     loops back to itself), not on link 8.
recs3 = recs[:1] + [(T, WTX, 1, c2c(SEND, 0, W(0), 9), 0), (T + L, ERX, 1, c2c(RECEIVE, 0, W(1), 8), 0),
                    (T + L + 1, ERX, 1, c2c(RECEIVE, 0, W(1), 9), 0)]
drain(recs3, 44, 50, 1, "W", RX_GAP, T + L + 1 + D_C2C)
o, d, wd = run(recs3, sram, [(44, 50)])
check("links are separate: send on link 9, nothing on link 8, the vector on link 9", d == [f"T diag {T + L} queue {ERX} OTS-E-TIM-001"] and wd[(44, 50)] == VAL, d)

# 2c. A link index >= c2c.links is OPR-001 (a profile with the ISCA 2022 count of 11 links).
cfg11 = json.loads(json.dumps(cfg)); cfg11["c2c"]["links"] = 11
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(cfg11, f); p11 = f.name
o, d, _ = run([(10, WTX, 1, c2c(TRANSMIT, 0, 0, 10), 0), (11, WTX, 1, c2c(TRANSMIT, 0, 0, 11), 0), (12, ERX, 1, c2c(RECEIVE, 0, W(0), 15), 0)], cfg_path=p11)
os.unlink(p11)
check("c2c.links = 11: link 10 is valid, links 11 and 15 are OPR-001", d == [f"T diag 11 queue {WTX} OTS-E-OPR-001", f"T diag 12 queue {ERX} OTS-E-OPR-001"], d)

# 3. East Transmit -> west Receive: the notification vector (lane 0 = 1, other lanes 0).
#    West RX writes row E at RX_GAP; west slice 0 reads E at 18.
recs = [(10, ETX, 1, c2c(TRANSMIT, 0, 0, 3), 0), (10 + L, WRX, 1, c2c(RECEIVE, 0, E(2), 3), 0)]
drain(recs, 0, 60, 2, "E", RX_GAP, 10 + L + D_C2C)
o, d, wd = run(recs, {(0, 60): bytes([0xEE] * VL)}, [(0, 60)])
check("Transmit sends the notification vector", d == [] and wd[(0, 60)] == NOTE, (d, wd[(0, 60)][:4].hex()))

# 4. chip_id must name the chip at the far end of the link (one chip: itself, 0).
o, d, _ = run([(10, WTX, 1, c2c(SEND, 1, W(0)), 0), (11, ETX, 1, c2c(TRANSMIT, 2), 0), (12, WRX, 1, c2c(RECEIVE, 3, E(0)), 0)])
check("wrong chip_id is OPR-001", d == [f"T diag {c} queue {q} OTS-E-OPR-001" for c, q in ((10, WTX), (11, ETX), (12, WRX))], d)
o, d, _ = run([(10, VXMQ, 1, c2c(SEND, 0, W(0)), 0), (11, WTX, 1, c2c(RECEIVE, 0, W(0)), 0)])
check("Send off a TX queue / Receive off an RX queue is QUE-002", d == [f"T diag 10 queue {VXMQ} OTS-E-QUE-002", f"T diag 11 queue {WTX} OTS-E-QUE-002"], d)

# 5. Deskew holds its queue (any queue: VXM, MEM, C2C) until the next HAC epoch boundary. The probe is an
#    instruction without its data: its TIM-001 is reported at dispatch + the unit's skew, so the report
#    cycle shows when the held instruction dispatched.
PROBE = {VXMQ: (2 << 60) | (15 << 54) | (8 << 46), VXMQ + 1: (2 << 60) | (15 << 54) | (8 << 46),
         0: mem(1, 0, E(0)), WTX: c2c(SEND, 0, W(0))}
def probe_at(q, c):
    """The cycle at which a probe dispatched at c on queue q reports TIM-001, and the diagnostics line."""
    o, d, _ = run([(10, q, 1, PROBE[q], 0)])
    skew = int(d[0].split()[2]) - 10
    return [f"T diag {c + skew} queue {q} OTS-E-TIM-001"]
check("the HAC epoch is 252 cycles (ISCA 2022 footnote 4)", P == 252, P)
for q in (VXMQ, 0, WTX):
    c0 = P + 10
    o, d, _ = run([(c0, q, 1, DESKEW, 0), (2 * P - 1, q, 1, PROBE[q], 0)])
    check(f"Deskew on queue {q} holds it to the epoch boundary {2 * P}: the probe due at {2 * P - 1} dispatches at {2 * P}", d == probe_at(q, 2 * P), d)
o, d, _ = run([(2 * P, VXMQ, 1, DESKEW, 0), (3 * P - 1, VXMQ, 1, PROBE[VXMQ], 0)])
check("Deskew exactly at a boundary waits a whole period", d == probe_at(VXMQ, 3 * P), d)
o, d, _ = run([(50, VXMQ, 1, DESKEW, 0), (60, VXMQ + 1, 1, PROBE[VXMQ + 1], 0)])
check("Deskew holds only its own queue", d == probe_at(VXMQ + 1, 60), d)

# 6. RuntimeDeskew t holds its queue t cycles (one chip: no drift).
o, d, _ = run([(300, VXMQ, 1, rtdeskew(0x20), 0), (331, VXMQ, 1, PROBE[VXMQ], 0), (340, 0, 1, rtdeskew(0), 0), (341, 0, 1, PROBE[0], 0)])
check("RuntimeDeskew t holds its queue t cycles (t = 0: one)", d == probe_at(VXMQ, 332) + probe_at(0, 341), d)

# 6b. Per-link latency (ISCA 2022 Table 2: 209..228). --link-latency N fixes every link.
for lat in (LMIN, LMAX):
    o, d, _ = run([(10, ETX, 1, c2c(TRANSMIT), 0), (10 + lat - 1, WRX, 1, c2c(RECEIVE, 0, E(0)), 0), (10 + lat, WRX, 1, c2c(RECEIVE, 0, E(0)), 0)], args=("--link-latency", str(lat)))
    check(f"--link-latency {lat}: the vector arrives {lat} cycles after the send", d == [f"T diag {9 + lat} queue {WRX} OTS-E-TIM-001"], d)
p = sim("", "--link-latency", str(LMAX + 1))
check("--link-latency outside [min, max] is rejected", p.returncode == 2, (p.returncode, p.stderr))

# 6c. A program scheduled at the nominal latency (Receive at send + L) is wrong when the link
#     is slower, while the Deskew-synchronised one (ISCA 2022 Fig. 7(b)) is right for every
#     latency in [min, max]: the sender issues Deskew then Send at an epoch boundary b; the
#     receiver issues Deskew twice and receives at b + P, after every possible arrival.
def naive(): # Send at T, Receive at T + L
    r, s = [], {}
    feed(r, s, 44, 1, VAL, 0, "W", TX_GAP, 3 * P)
    r += [(3 * P, WTX, 1, c2c(SEND, 0, W(0), 2), 0), (3 * P + L, ERX, 1, c2c(RECEIVE, 0, W(1), 2), 0)]
    drain(r, 44, 50, 1, "W", RX_GAP, 3 * P + L + D_C2C)
    return r, s
def synced(): # sender: Deskew at 2P + 7, Send at 3P; receiver: Deskew at 2P + 30, Deskew at 3P, Receive at 4P
    r, s = [], {}
    feed(r, s, 44, 1, VAL, 0, "W", TX_GAP, 3 * P)
    r += [(2 * P + 7, WTX, 1, DESKEW, 0), (3 * P, WTX, 1, c2c(SEND, 0, W(0), 2), 0),
          (2 * P + 30, ERX, 1, DESKEW, 0), (3 * P, ERX, 1, DESKEW, 0), (4 * P, ERX, 1, c2c(RECEIVE, 0, W(1), 2), 0)]
    drain(r, 44, 50, 1, "W", RX_GAP, 4 * P + D_C2C)
    return r, s
o, d, wd = run(*naive(), [(44, 50)], args=("--link-latency", str(LMAX)))
check(f"without Deskew: Receive at send + {L} fails on a {LMAX}-cycle link", d[:1] == [f"T diag {3 * P + L} queue {ERX} OTS-E-TIM-001"] and wd[(44, 50)] != VAL, d)
outs = []
for args in [("--link-latency", str(LMIN)), (), ("--link-latency", str(LMAX))] + [("--link-latency-seed", str(k)) for k in range(1, 6)]:
    o, d, wd = run(*synced(), [(44, 50)], args=args)
    outs.append((d, wd[(44, 50)]))
check("with Deskew: the same result for latencies min, nominal, max and 5 random per-link draws", all(x == ([], VAL) for x in outs), outs[:2])

# 7. Topology. Two chips (one node): local link 0 of chip 0 is local link 0 of chip 1.
T = 40
r0, s0 = [], {}
feed(r0, s0, 0, 1, VAL, 0, "E", TX_GAP, T)                    # west slice 0 writes E at 18; east TX reads E
r0.append((T, ETX, 1, c2c(SEND, 1, E(0)), 0))
r1 = [(T + L - 1, WRX, 1, c2c(RECEIVE, 0, E(3)), 0), (T + L, WRX, 1, c2c(RECEIVE, 0, E(3)), 0)]
drain(r1, 0, 70, 3, "E", RX_GAP, T + L + D_C2C)
r1.append((T + 2 * L, ETX, 1, c2c(TRANSMIT, 0), 0))           # chip 1 link 0 -> chip 0 link 0
r0.append((T + 3 * L, WRX, 1, c2c(RECEIVE, 1, E(4)), 0))
drain(r0, 0, 71, 4, "E", RX_GAP, T + 3 * L + D_C2C)
(d0, w0), (d1, w1) = run_chips([(r0, s0, [(0, 71)]), (r1, {}, [(0, 70)])])
check("two chips: Send chip 0 -> Receive chip 1 after the link latency", d1 == [f"T diag {T + L - 1} queue {WRX} OTS-E-TIM-001"] and w1[(0, 70)] == VAL, (d1, w1[(0, 70)][:4].hex()))
check("two chips: Transmit chip 1 -> Receive chip 0", d0 == [] and w0[(0, 71)] == NOTE, (d0, w0[(0, 71)][:4].hex()))
(d0, _), (d1, _) = run_chips([([(10, ETX, 1, c2c(SEND, 0, E(0)), 0), (11, ETX, 1, c2c(TRANSMIT, 1, 0, 1), 0)], {}, []), ([(10, ETX, 1, c2c(TRANSMIT, 0), 0)], {}, [])])
check("two chips: chip_id names the peer; link 1 is not connected (OPR-001)", d0 == [f"T diag 10 queue {ETX} OTS-E-OPR-001", f"T diag 11 queue {ETX} OTS-E-OPR-001"] and d1 == [], (d0, d1))


# 7b. Eight chips, one node (ISCA 2022 sec. 2.2): local link l of chip i reaches chip
#     (i + l + 1) mod 8 on its link 6 - l. Every chip transmits on all 7 links (cycles 10..16)
#     and receives on all 7 links from the expected sender just after arrival; link 7 is open.
def node_progs(K, lat):
    progs = []
    for i in range(K):
        r = [(10 + l, ETX if l % 2 else WTX, 1, c2c(TRANSMIT, (i + l + 1) % K, 0, l), 0) for l in range(K - 1)]
        for m in range(K - 1):               # link m is paired with link K - 2 - m of chip (i + m + 1) mod K
            l = K - 2 - m
            r.append((10 + l + lat + 20 * m, WRX, 1, c2c(RECEIVE, (i + m + 1) % K, E(m % 8), m), 0))
        r.append((5, WTX, 1, c2c(TRANSMIT, 0, 0, K - 1), 0))   # link K - 1 is not connected
        progs.append((r, {}, []))
    return progs
res = run_chips(node_progs(8, L))
check("eight chips: the 7 local links fully connect the node (every chip hears every other once)",
      all(d == [f"T diag 5 queue {WTX} OTS-E-OPR-001"] for d, _ in res), [d for d, _ in res][:3])
res = run_chips(node_progs(5, LMAX), ("--link-latency", str(LMAX)))
check("five chips: one node, local links 4..6 open", all(d == [f"T diag 5 queue {WTX} OTS-E-OPR-001"] for d, _ in res), [d for d, _ in res][:3])


# 7c. Sixteen chips (two nodes): global link 7 of chip 0 (node 0, port 0) reaches chip 8
#     (node 1, port 0) on its link 7; chip 1's link 7 (port 4) is open with two nodes.
progs = [([], {}, []) for _ in range(16)]
progs[0] = ([(10, ETX, 1, c2c(TRANSMIT, 8, 0, 7), 0), (11, ETX, 1, c2c(TRANSMIT, 1, 0, 0), 0)], {}, [])
progs[1] = ([(10, ETX, 1, c2c(TRANSMIT, 9, 0, 7), 0), (11 + L, WRX, 1, c2c(RECEIVE, 0, E(0), 6), 0)], {}, [])
progs[8] = ([(10 + L, WRX, 1, c2c(RECEIVE, 0, E(0), 7), 0), (10 + L + 1, WRX, 1, c2c(RECEIVE, 0, E(0), 7), 0)], {}, [])
res = run_chips(progs)
check("sixteen chips: global link 7 joins chip 0 and chip 8; local links stay in the node",
      res[0][0] == [] and res[1][0] == [f"T diag 10 queue {ETX} OTS-E-OPR-001"] and res[8][0] == [f"T diag {11 + L} queue {WRX} OTS-E-TIM-001"]
      and all(res[k][0] == [] for k in range(16) if k not in (1, 8)), [r[0] for r in res[:9]])
p = sim("", "--chips", "265")
check("--chips 265 (more than the 33 nodes of the wiring) is rejected", p.returncode == 2, (p.returncode, p.stderr))
p = sim("", "--chips", "12")
check("--chips 12 (a node of 8 and a node of 4 chips) is accepted", p.returncode == 0, (p.returncode, p.stderr))

# ---------------------------------------------------------------------------------------
# Harness input and output contracts.
def srfs(o): return [int(l.split()[2]) for l in o.splitlines() if l.startswith("T srf")]
Z = {(0, 1): bytes([1] * VL)}

# An effect that commits long after the last record is not lost.
for cd in (59, 60, 200):
    o, d, _ = run([(0, 0, 1, mem(0, 1, E(0)), 0), (1, 0, 1, agt(5, cd), 0), (2, 0, 1, agt(7, 2), 0)], Z)
    check(f"T2 AGT countdown {cd}: the late reference's stream write is applied", srfs(o) == [5, cd + 6], srfs(o))
o, d, _ = run([(0, 0, 1, mem(0, 1, E(0)), 0), (1, 0, 1, agt(5, 200), 0), (2, 0, 1, agt(7, 2), 0), (1, 1, 1, mem(0, 1, E(0)), 0), (2, 1, 1, agt(5, 199), 0), (3, 1, 1, agt(7, 2), 0)], {(0, 1): bytes(VL), (1, 1): bytes(VL)})
check("T2 a collision at commit after a long countdown is reported", d != [] and all(l.startswith("T diag 206 ") and l.endswith("OTS-E-SRF-002") for l in d), d)

# Idle cycles are skipped; an over-long run stops with exit 3. A gap is lowered to NOP units of
# at most 65535 cycles each, and a queue's initial text is at most icu.iq_depth / 4 units, so the
# last cycle of an initial image is about 65535 x icu.iq_depth / 4.
t0 = time.time()
o, d, _ = run([(0x1312D00, 0, 1, mem(0, 1, E(0)), 0)], Z)
check("T3 a record at cycle 2 x 10^7 (305 NOP units) runs in well under a second", srfs(o) == [0x1312D05] and time.time() - t0 < 5, (srfs(o), time.time() - t0))
huge = lines([(0, 0, 1, mem(0, 1, E(0)), 0)] + [(1 + i, 0, 1, agt(7, 0xFFFF), 0) for i in range(4)], Z)
p = sim("\n".join(huge) + "\n", "--max-cycles", "300")
check("T3 --max-cycles: exit 3 with a message", p.returncode == 3 and "max-cycles" in p.stderr, (p.returncode, p.stderr))
p = sim("Q 0 0000000a 10000000 00000000\n", "--max-cycles", "9")
check("T3 --max-cycles below a parcel's cycle: exit 3", p.returncode == 3, (p.returncode, p.stderr))

# Input validation (exit 2, message naming the line).
CAP = cfg["icu"]["iq_depth"] // 4
for bad in ("S 58 0 1", "S 0 2000 1", "D 1000 0", "D 0 2000", "Q 10000 0", "Q 90 0", "Q 100 0", "Q ffffffff 0", "Q 0 100000000",
            "Q -1 0", "Q 0 zz", "Q 0 -1", "S 0 0 100", "X 1 2", "P 5 0 1 0 0", "P 5 0 1 1000 0", "Q 0 " + " ".join(["0"] * (CAP + 1)),
            "Q 0 " + " ".join(["0"] * CAP) + "\nQ 0 0"):
    p = sim(bad + "\n")
    check(f"Malformed harness input: {bad[:60]!r}", p.returncode == 2 and "line " in p.stderr and p.stdout == "", (p.returncode, p.stderr, p.stdout[:80]))
for good in ("Q 5", "Q 0 " + " ".join(["0"] * CAP), "Q 0 ffffffff ffffffff\nQ 0 " + " ".join(["0"] * CAP), "Q ffff 0", "Q 8f 0"):
    p = sim(good + "\n")
    check(f"T4 accepted input: {good[:60]!r}", p.returncode == 0, (p.returncode, p.stderr))
p = sim("\n".join(["D 0 0"] * 5000) + "\n")
check("T4 5000 dumps (no fixed limit)", p.returncode == 0 and p.stdout.count("W 0 0 ") == 5000, (p.returncode, p.stderr))
# Every queue (the control channel too) at the full initial image: 145 x icu.iq_depth / 4 units of NOP 1
# (a timing prefix: no parcel ever completes).
big = "\n".join(f"Q {q:x} " + " ".join(["00000001"] * CAP) for q in list(range(144)) + [0xFFFF]) + "\n"
p = sim(big)
check("T4 every queue full (no fixed limit)", p.returncode == 0 and p.stdout.startswith("E - -1 0"), (p.returncode, p.stderr[:200], p.stdout[:80]))

# A text unit for a queue outside 0..143 and 0xFFFF is an input error (above); a word that fits no
# layout is lowered to the undecodable unit (tag 6) and is OTS-E-DEC-003 at its cycle, nothing else runs.
o, d, _ = run([(5, 0, 1, (1 << 32) | (1 << 22), 0), (6, 0, 1, mem(0, 1, E(0)), 0)], Z)
check("an undecodable word is DEC-003 at its cycle and the next instruction runs", d == ["T diag 5 queue 0 OTS-E-DEC-003"] and srfs(o) == [11], (d, srfs(o)))

# Duplicate records give the same output in either order.
a, b = (10, 0, 1, mem(0, 1, E(0)), 0), (10, 0, 1, mem(0, 2, E(0)), 0)
oa, ob = run([a, b], Z)[0], run([b, a], Z)[0]
check("T5 duplicates: output independent of record order", oa == ob and "OTS-E-QUE-001" in oa, (oa[:200], ob[:200]))


# =======================================================================================
# A rejected Receive has no effect: a Receive to a stream the RX unit cannot write
# (W:0 from the east RX, DIR-001) leaves the vector in the link FIFO for the next Receive.
o, d, _ = run([(0, WTX, 1, 0x50C00000, 0), (0x12C, WRX, 1, 0x50408000, 0), (0x136, WRX, 1, 0x50400000, 0)])
check(" a DIR-001 Receive keeps the FIFO head; the next Receive takes it", d == [f"T diag 300 queue {WRX} OTS-E-DIR-001"] and "T srf 311 E:0 gap 5 " in o, (d, srfs(o)))
# --link-jitter-seed S draws every transfer's latency in [min, max] (in order); the
# Deskew-synchronised program is right under it; the naive one sees varying arrivals.
outs = [run(*synced(), [(44, 50)], args=("--link-jitter-seed", str(k)))[1:] for k in range(1, 9)]
check("with Deskew: correct under per-transfer jitter (8 seeds)", all(dd == [] and w[(44, 50)] == VAL for dd, w in outs), [dd for dd, _ in outs][:3])
nv = [tuple(run(*naive(), [(44, 50)], args=("--link-jitter-seed", str(k)))[1]) for k in range(1, 17)]
check("without Deskew: the jittered arrival differs between seeds", len(set(nv)) > 1, nv[:4])
check("same seed, same run", run(*naive(), [(44, 50)], args=("--link-jitter-seed", "3"))[0] == run(*naive(), [(44, 50)], args=("--link-jitter-seed", "3"))[0])
p = sim("", "--link-jitter-seed", "1", "--link-latency", str(L))
check("--link-jitter-seed with --link-latency is rejected", p.returncode == 2, (p.returncode, p.stderr))

# : the harness keeps vectors in order under --link-jitter-seed: a later send never arrives before an earlier one (it
# arrives at least one cycle after it). The rule can fire only when two sends are closer than the jitter window is wide:
# the draws differ by at most LMAX - LMIN, the sends are at least cycles_per_vector cycles apart, so it is reachable on a
# profile with cycles_per_vector <= LMAX - LMIN (the test profile link_tight: 3 against 19) and not on `ref` (20 against 19).
# The test reproduces the harness's generator (splitmix64 per link, the state seeded with S xor (link index + 1) times a
# constant) to find the seeds whose two draws would invert the arrivals, sends two different vectors CPV cycles apart on
# link 0 and receives them at the earliest legal cycles: the first vector first, and the second exactly one cycle after
# it. Without the rule the second vector overtakes the first in the receive FIFO.
CPV13 = cfg["c2c"]["cycles_per_vector"]; W13 = LMAX - LMIN + 1; SK13 = cfg["timing"]["c2c"]["d_skew"]; M64 = (1 << 64) - 1
def link_latency_draws(seed):
    x = (seed ^ ((1 * 0xD1B54A32D192ED03) & M64)) & M64
    out = []
    for _ in range(2):
        x = (x + 0x9E3779B97F4A7C15) & M64
        z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out.append((z ^ (z >> 31)) % W13)
    return out
def run_link_latency_case(seed, late):
    """Two Sends CPV cycles apart on link 0; the Receives at the earliest legal cycles (late = False: the first at its arrival,
    the second one cycle later) or well after both arrivals. Returns (diagnostics, word 50, word 51)."""
    T = 100; r1, r2 = link_latency_draws(seed)
    v1, v2 = VAL, bytes((11 * l + 5) & 0xFF for l in range(VL))
    recs, sram = [], {}
    feed(recs, sram, 44, 1, v1, 0, "W", TX_GAP, T)
    feed(recs, sram, 45, 2, v2, 2, "W", TX_GAP, T + CPV13)
    recs.append((T, WTX, 1, c2c(SEND, 0, W(0)), 0)); recs.append((T + CPV13, WTX, 1, c2c(SEND, 0, W(2)), 0))
    d1 = (T + SK13 + LMIN + r1 - SK13) if not late else (T + CPV13 + LMAX + 10)
    recs.append((d1, ERX, 1, c2c(RECEIVE, 0, W(1)), 0)); recs.append((d1 + 1, ERX, 1, c2c(RECEIVE, 0, W(1)), 0))
    drain(recs, 44, 50, 1, "W", RX_GAP, d1 + D_C2C); drain(recs, 44, 51, 1, "W", RX_GAP, d1 + 1 + D_C2C)
    o, d, wd = run(recs, sram, [(44, 50), (44, 51)], args=("--link-jitter-seed", str(seed)))
    return d, wd.get((44, 50)), wd.get((44, 51)), v1, v2
INVERTING = [sd for sd in range(1, 400) if (lambda r: r[0] - r[1] >= CPV13)(link_latency_draws(sd))]
check(": the in-order rule is reachable exactly when cycles_per_vector <= link_latency_max - link_latency_min",
      bool(INVERTING) == (CPV13 <= LMAX - LMIN), (CPV13, LMAX - LMIN, INVERTING[:3]))
for sd in INVERTING[:4]:
    d, w50, w51, v1, v2 = run_link_latency_case(sd, False)
    check(f": seed {sd} draws {link_latency_draws(sd)} (the second would arrive first): the first vector is received first, "
          "at its arrival, and the second exactly one cycle later", d == [] and w50 == v1 and w51 == v2, (d, (w50 or b"")[:4].hex(), (w51 or b"")[:4].hex()))
# strictly later: a second Receive queue asks for the next vector in the cycle the first arrives; it is not there yet
for sd in INVERTING[:2]:
    T = 100; r1, _ = link_latency_draws(sd); d1 = T + LMIN + r1
    recs, sram = [], {}
    feed(recs, sram, 44, 1, VAL, 0, "W", TX_GAP, T)
    feed(recs, sram, 45, 2, bytes(VL), 2, "W", TX_GAP, T + CPV13)
    recs.append((T, WTX, 1, c2c(SEND, 0, W(0)), 0)); recs.append((T + CPV13, WTX, 1, c2c(SEND, 0, W(2)), 0))
    recs.append((d1, ERX, 1, c2c(RECEIVE, 0, W(1)), 0)); recs.append((d1, ERX1, 1, c2c(RECEIVE, 0, W(3)), 0))
    o, d, _w = run(recs, sram, [], args=("--link-jitter-seed", str(sd)))
    check(f": seed {sd}: the second vector arrives strictly after the first (a Receive of the next vector in the first's arrival cycle is TIM-001)",
          d == [f"T diag {d1 + SK13} queue {ERX1} OTS-E-TIM-001"], d)
for sd in [1, 2, 3, 4]:
    d, w50, w51, v1, v2 = run_link_latency_case(sd, True)
    check(f": seed {sd}: the vectors arrive in the order sent", d == [] and w50 == v1 and w51 == v2, (d, (w50 or b"")[:4].hex(), (w51 or b"")[:4].hex()))

# =======================================================================================
# Per-link latency checked
# per link, the seeded draw checked against the published range, and global links with
# port p != q.
def h6_arrival(K, tx, rx, link, plink, args, lo=LMIN - 3, hi=LMAX + 3):
    """Chip tx Transmits on link at 16; chip rx Receives on plink every cycle 16+lo..16+hi.
    Returns the first latency at which the Receive succeeds (None: never)."""
    progs = [([], {}, []) for _ in range(K)]
    progs[tx] = ([(16, ETX, 1, c2c(TRANSMIT, rx, 0, link), 0)], {}, [])
    rr = [(16 + k, WRX, 1, c2c(RECEIVE, tx, E(0), plink), 0) for k in range(lo, hi + 1)]
    if rx == tx: progs[tx] = (progs[tx][0] + rr, {}, [])
    else: progs[rx] = (rr, {}, [])
    res = run_chips(progs, args)
    tim = {int(l.split()[2]) for l in res[rx][0] if "TIM-001" in l}
    ok = [k for k in range(lo, hi + 1) if 16 + k not in tim]
    return ok[0] if ok else None
# The documented draw (microarchitecture.md sec. 10; c_emulator/tsp_sim.c links_setup):
# splitmix64 seeded with S, one draw per (chip, link) in chip-major, link-minor order over
# 16 link slots, latency = min + draw mod (max - min + 1). Recomputed here independently.
M64 = (1 << 64) - 1
def h6_draws(seed, K):
    x, out = seed, {}
    for c in range(K):
        for l in range(16):
            x = (x + 0x9E3779B97F4A7C15) & M64
            z = x
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
            out[(c, l)] = LMIN + (z ^ (z >> 31)) % (LMAX - LMIN + 1)
    return out
# Three chips (one node): chip 0 link 0 -> chip 1 link 1, chip 0 link 1 -> chip 2 link 0,
# chip 1 link 0 -> chip 2 link 1; each link has its own drawn latency.
PAIRS3 = [(0, 1, 0, 1), (0, 2, 1, 0), (1, 2, 0, 1), (2, 0, 0, 1)]
seen = set()
for seed in range(1, 6):
    want = h6_draws(seed, 3)
    got = {(tx, l): h6_arrival(3, tx, rx, l, pl, ("--link-latency-seed", str(seed))) for tx, rx, l, pl in PAIRS3}
    seen |= set(got.values())
    check(f"seed {seed}: every link's arrival is its own documented draw in [{LMIN}, {LMAX}]",
          all(got[k] == want[k] and LMIN <= got[k] <= LMAX for k in got), (got, {k: want[k] for k in got}))
check("the seed changes the latencies (more than one value over 5 seeds x 4 links)", len(seen) > 1, seen)
# Global links with p != q (topology.md sec. 8; tsp_sim.c link_peer): with N nodes, global
# port p = 4k + (l - 7) of chip k of node n reaches node (n + p + 1) mod N at port q =
# N - 2 - p, i.e. chip q / 4 of that node, on its link 7 + q mod 4.
def h6_global(N, k, l):
    p = 4 * k + (l - 7)
    if p >= N - 1: return None
    m, q = (0 + p + 1) % N, N - 2 - p
    return m * 8 + q // 4, 7 + q % 4
for K, k, l in ((24, 0, 7), (24, 0, 8), (64, 0, 7), (64, 1, 8), (64, 1, 9)):
    rx, pl = h6_global(K // 8, k, l)
    lat = h6_arrival(K, k, rx, l, pl, (), L - 1, L + 1)
    check(f"{K} chips: chip {k} global link {l} reaches chip {rx} link {pl} in {L} cycles", lat == L, lat)
# F4-V2ONLY: the latency range is the literal minimum and maximum of ISCA22 Table 2, not the
# profile's values: --link-latency takes 209 and 228 and refuses 208 and 229.
for lat, good in ((209, True), (228, True), (208, False), (229, False)):
    p = sim("", "--link-latency", str(lat))
    check(f"--link-latency {lat} is {'accepted' if good else 'refused'} (ISCA22 Table 2: 209..228)", (p.returncode == 0) == good, p.returncode)

# =======================================================================================
# Link bandwidth (OTS-E-LNK-001, model) and receive FIFO (OTS-E-LNK-002, harness).
CPV = cfg["c2c"]["cycles_per_vector"]; RXD = cfg["c2c"]["rx_depth"]
SK = cfg["timing"]["c2c"]["d_skew"]            # a C2C instruction executes d_skew cycles after its dispatch
def lnk1(c, q): return f"T diag {c + SK} queue {q} OTS-E-LNK-001"
def lnk2(c): return f"T diag {c} queue 65535 OTS-E-LNK-002"
def tim1(c, q): return f"T diag {c + SK} queue {q} OTS-E-TIM-001"
T = 100
# A Transmit on a link that accepted a vector fewer than cycles_per_vector cycles ago is LNK-001 and
# sends nothing: of the Transmits at T, T + 1 and T + CPV - 1 only the first is accepted, and exactly one
# vector arrives.
o, d, _ = run([(T, WTX, 1, c2c(TRANSMIT), 0), (T + 1, WTX, 1, c2c(TRANSMIT), 0), (T + CPV - 1, WTX, 1, c2c(TRANSMIT), 0),
               (T + L, ERX, 1, c2c(RECEIVE, 0, W(0)), 0), (T + L + 1, ERX, 1, c2c(RECEIVE, 0, W(0)), 0)])
check("LNK-001: a Transmit fewer than cycles_per_vector cycles after an accepted one is busy and sends nothing",
      d == [lnk1(T + 1, WTX), lnk1(T + CPV - 1, WTX), tim1(T + L + 1, ERX)], d)
# One vector per cycles_per_vector cycles is accepted; the window is measured from the last accepted
# vector (not from a refused one). The three accepted vectors are not received: with rx_depth < 3 the
# third is dropped at its arrival (LNK-002).
recs = [(T + i * CPV, WTX, 1, c2c(TRANSMIT), 0) for i in range(3)] + [(T + CPV + 5, WTX, 1, c2c(TRANSMIT), 0)]
o, d, _ = run(recs)
check("LNK-001: a Transmit exactly cycles_per_vector cycles after the last accepted one is accepted",
      d == [lnk1(T + CPV + 5, WTX)] + ([lnk2(T + 2 * CPV + L)] if RXD < 3 else []), d)
# Literal values and not the profile's: the shipped profiles give a link one vector per 24 cycles
# (ISCA22 sec. 2.3 footnote 2: four lanes of 25 Gb/s give a 2560-bit vector in 24 cycles at the 900 MHz clock).
if os.path.basename(CFG) in ("ref.json", "mini.json"):
    o, d, _ = run([(T, WTX, 1, c2c(TRANSMIT), 0), (T + 23, WTX, 1, c2c(TRANSMIT), 0), (T + 24, WTX, 1, c2c(TRANSMIT), 0)])
    check("a shipped profile's link is busy 23 cycles after an accepted vector and accepts one at 24",
          CPV == 24 and d == [lnk1(T + 23, WTX)] + ([lnk2(T + 24 + L)] if RXD < 2 else []), (CPV, d))
# Send is busy alike (and nothing is read): the second Send is G < cycles_per_vector cycles after the first.
G = min(3, CPV - 1)
recs, sram = [], {}
feed(recs, sram, 44, 1, VAL, 0, "W", TX_GAP, T)
feed(recs, sram, 44, 2, VAL, 1, "W", TX_GAP, T + G)
recs += [(T, WTX, 1, c2c(SEND, 0, W(0)), 0), (T + G, WTX, 1, c2c(SEND, 0, W(1)), 0),
         (T + L, ERX, 1, c2c(RECEIVE, 0, W(2)), 0), (T + L + G, ERX, 1, c2c(RECEIVE, 0, W(2)), 0)]
drain(recs, 44, 50, 2, "W", RX_GAP, T + L + D_C2C)
o, d, wd = run(recs, sram, [(44, 50)])
check("LNK-001: a Send on a busy link is refused (the first vector arrives, the second does not)",
      d == [lnk1(T + G, WTX), tim1(T + L + G, ERX)] and wd[(44, 50)] == VAL, (d, wd[(44, 50)][:4].hex()))
# Links are independent.
o, d, _ = run([(T, WTX, 1, c2c(TRANSMIT, 0, 0, 0), 0), (T + 1, WTX, 1, c2c(TRANSMIT, 0, 0, 1), 0), (T + 2, WTX, 1, c2c(TRANSMIT, 0, 0, 2), 0)])
check("LNK-001: the links of a chip are independent", d == [], d)
# Two in one cycle: the second in execution order (ascending queue) is busy; here the east TX queue.
o, d, _ = run([(T, WTX, 1, c2c(TRANSMIT), 0), (T, ETX, 1, c2c(TRANSMIT), 0), (T, WTX1, 1, c2c(TRANSMIT, 0, 0, 5), 0), (T, ETX1, 1, c2c(TRANSMIT, 0, 0, 5), 0)])
check("LNK-001: two sends in one cycle: the second in execution order is busy", d == [lnk1(T, ETX), lnk1(T, ETX1)], d)
# A rejected instruction occupies nothing: a Send with no data (TIM-001) and a wrong chip_id (OPR-001).
o, d, _ = run([(T, WTX, 1, c2c(SEND, 0, W(0)), 0), (T + 1, WTX, 1, c2c(TRANSMIT, 1), 0), (T + 2, WTX, 1, c2c(TRANSMIT), 0)])
check("LNK-001: a Send that finds no data and a wrong chip_id leave the link free",
      d == [tim1(T, WTX), f"T diag {T + 1 + SK} queue {WTX} OTS-E-OPR-001"], d)

# LNK-002: RXD vectors fill the FIFO of link 0 (one chip: loop back; no Receive), the next one is dropped
# at its arrival cycle on queue 65535; each further one too.
def flood(n, extra=()):
    return [(T + i * CPV, WTX, 1, c2c(TRANSMIT), 0) for i in range(n)] + list(extra)
arr = lambda i: T + i * CPV + L                  # arrival cycle of vector i (0-based)
o, d, _ = run(flood(RXD))
check(f"LNK-002: {RXD} vectors fit the receive FIFO", d == [], d)
o, d, _ = run(flood(RXD + 2))
check("LNK-002: a vector that arrives at a full FIFO is dropped; reported at its arrival cycle on queue 65535, one per vector",
      d == [lnk2(arr(RXD)), lnk2(arr(RXD + 1))], d)
check("LNK-002: the E line names the first code and its cycle", f"E OTS-E-LNK-002 {arr(RXD)} 2" in o, o.splitlines()[0] if o else o)
# A Receive in an earlier cycle than the arrival has freed a slot; a Receive in the arrival cycle takes after it.
o, d, _ = run(flood(RXD + 1, [(arr(RXD) - 1, ERX, 1, c2c(RECEIVE, 0, W(0)), 0)]))
check("LNK-002: a Receive one cycle before the arrival frees a slot (no drop)", d == [], d)
o, d, _ = run(flood(RXD + 1, [(arr(RXD), ERX, 1, c2c(RECEIVE, 0, W(0)), 0)]))
check("LNK-002: a Receive in the arrival cycle takes after the arrival (dropped)", d == [lnk2(arr(RXD))], d)
# The dropped vector is gone: RXD + 1 sent, the Receives at RXD + 1 cycles after the arrivals get RXD vectors then TIM-001.
rx = [(arr(RXD) + 1 + i, ERX, 1, c2c(RECEIVE, 0, W(0)), 0) for i in range(RXD + 1)]
o, d, _ = run(flood(RXD + 1, rx))
check("LNK-002: the dropped vector is lost (RXD Receives succeed, the next is TIM-001)", d == [lnk2(arr(RXD)), tim1(arr(RXD) + 1 + RXD, ERX)], d)
# Reported by the harness before the tick of its cycle: a TIM-001 of the same cycle comes after.
o, d, _ = run(flood(RXD + 1, [(arr(RXD), ERX, 1, c2c(RECEIVE, 0, W(0), 3), 0)]))
check("LNK-002: reported before the cycle's tick (before the diagnostics of the cycle's instructions)",
      d == [lnk2(arr(RXD)), tim1(arr(RXD), ERX)] if SK == 0 else d[0] == lnk2(arr(RXD)), d)
# Other links have their own FIFO.
o, d, _ = run(flood(RXD) + [(T + 1 + i * CPV, WTX, 1, c2c(TRANSMIT, 0, 0, 1), 0) for i in range(RXD)])
check("LNK-002: each link has its own receive FIFO", d == [], d)
# Two chips: the receiving chip reports (chip 1 here), the sender does not.
r0 = [(T + i * CPV, ETX, 1, c2c(TRANSMIT, 1), 0) for i in range(RXD + 1)]
(d0, _), (d1, _) = run_chips([(r0, {}, []), ([], {}, [])])
check("LNK-002: with two chips the receiving chip reports at the arrival cycle", d0 == [] and d1 == [lnk2(arr(RXD))], (d0, d1))
# A link of rx_depth 1: the second vector in flight is dropped (a profile with a small FIFO).
cfg1 = json.loads(json.dumps(cfg)); cfg1["c2c"]["rx_depth"] = 1; cfg1["c2c"]["cycles_per_vector"] = 1
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(cfg1, f); p1 = f.name
o, d, _ = run([(T, WTX, 1, c2c(TRANSMIT), 0), (T + 1, WTX, 1, c2c(TRANSMIT), 0), (T + 2, WTX, 1, c2c(TRANSMIT), 0),
               (T + L + 1, ERX, 1, c2c(RECEIVE, 0, W(0)), 0), (T + L + 2, ERX, 1, c2c(RECEIVE, 0, W(0)), 0)], cfg_path=p1)
os.unlink(p1)
check("rx_depth 1, cycles_per_vector 1: a vector that arrives while the FIFO holds one is dropped; the third arrives after the Receive took the first",
      d == [lnk2(T + 1 + L)], d)

print("c2c/harness directed:", "all pass" if ok else "FAILURES")
sys.exit(0 if ok else 1)
