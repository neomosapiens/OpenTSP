"""Directed end-to-end tests of DeskewRecv (, C2C opcode 9): the data-dependent polling loop at program
start (ISCA22 sec. 3.2, Fig. 7(b)), on the Sail emulator or any binary with its command line (the ISS).

The published loop: the child device is placed in a polling synchronisation loop; at each epoch boundary
(HAC = 0) it waits to receive a vector from the parent; if the vector has not been transmitted yet the loop
continues; the parent runs DESKEW then TRANSMIT; the vector is consumed by the RECV that issues after the next
epoch boundary after it arrives, so the child exits the loop floor(L / period) + 1 epochs after the TRANSMIT.
DeskewRecv chip, link, s is that loop as one instruction (the ISA has no branch: an implementation choice).
Profile-portable. usage: deskewrecv_directed.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
cfg = json.load(open(CFG))
L = cfg["c2c"]["link_latency"]; P = cfg["c2c"]["hac_period"]
LMIN = cfg["c2c"]["link_latency_min"]; LMAX = cfg["c2c"]["link_latency_max"]
LINKS = cfg["c2c"]["links"]
RX_GAP = cfg["layout"]["c2c_rx_gap"]
D_C2C = cfg["timing"]["c2c"]["d_func"]
BAR = cfg["barrier"]["release_latency"]
VL = 16 * cfg["geometry"]["superlanes"]
assert LMAX < P

ok = True
def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def c2c(op, chip=0, s=0, link=0): return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)
def deskewrecv(chip, link, s): return c2c(9, chip, s, link)
def transmit(chip, link): return c2c(3, chip, 0, link)
DESKEW = c2c(2)
SYNC, NOTIFY = 1 << 22, 2 << 22
def E(i): return i
WTX, WRX, ETX, ERX, VXMQ = 126, 128, 140, 142, 88
NOTE = bytes([1]) + bytes(VL - 1)
ZW = bytes([0x5A]) * VL


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def text(prog, start=0):
    recs, sram, dumps = prog
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(recs)) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    if start: ls.append(f"H start {start:x}")
    return ls


def sim(t, *args):
    return subprocess.run([SIM, CFG, "--trace", *args], input=t, capture_output=True, text=True, timeout=600)


def parse(lines_):
    words = {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in lines_) if t and t[0] == "W"}
    return ([l for l in lines_ if l.startswith("T diag")], words, [int(l.split()[2]) for l in lines_ if l.startswith("T srf")])


def run1(prog, args=(), start=0):
    p = sim("\n".join(text(prog, start)) + "\n", *args)
    assert p.returncode == 0, p.stderr
    return parse(p.stdout.splitlines())


def runk(progs, args=()):
    """progs: one (prog, start) per chip. One (diags, words, srf cycles) per chip."""
    t = []
    for k, (prog, start) in enumerate(progs): t += [f"C {k:x}"] + text(prog, start)
    p = sim("\n".join(t) + "\n", "--chips", str(len(progs)), *args)
    assert p.returncode == 0, p.stderr
    parts, cur = [], None
    for l in p.stdout.splitlines():
        if l.startswith("C "): cur = []; parts.append(cur)
        else: cur.append(l)
    return [parse(x) for x in parts]


def stored(boundary):
    """Records that store a vector committed by a Receive dispatched at `boundary` (stream E:3, written at RX_GAP) in slice 0 addr 60."""
    r = []
    drain(r, 0, 60, 3, "E", RX_GAP, boundary + D_C2C)
    return r


# 1. One chip (every link loops back): Transmit at 10 arrives at 10 + L; DeskewRecv at 5 polls at 252: the vector
#    is there, so the Receive dispatches at 252.
recs = [(5, WRX, 1, deskewrecv(0, 0, E(3)), 0), (10, ETX, 1, transmit(0, 0), 0)] + stored(P)
d, w, _ = run1((recs, {(0, 60): ZW}, [(0, 60)]))
check("DeskewRecv polls at the first epoch boundary and takes the vector that has arrived: Receive at 252", d == [] and w[(0, 60)] == NOTE, (d, w[(0, 60)][:4].hex()))

# 2. The vector arrives after the first boundary: Transmit at 250 arrives at 250 + L > 252, so the loop goes on and
#    the Receive dispatches at the second boundary, 504, not at the arrival cycle (a poll every cycle would
#    take it at 250 + L); nothing is reported while waiting.
recs = [(5, WRX, 1, deskewrecv(0, 0, E(3)), 0), (250, ETX, 1, transmit(0, 0), 0)] + stored(2 * P)
d, w, _ = run1((recs, {(0, 60): ZW}, [(0, 60)]))
check("a vector that arrives after a boundary is taken at the next boundary (504), with no diagnostic while waiting", d == [] and w[(0, 60)] == NOTE, (d, w[(0, 60)][:4].hex()))
recs = [(5, WRX, 1, deskewrecv(0, 0, E(3)), 0), (250, ETX, 1, transmit(0, 0), 0)] + stored(250 + L)
d, w, _ = run1((recs, {(0, 60): ZW}, [(0, 60)]))
check(f"a Receive at the arrival cycle {250 + L} does not happen (the test is at the boundary only)", w[(0, 60)] != NOTE, w[(0, 60)][:4].hex())

# 3. The boundary is the next one AFTER the dispatch, as Deskew: DeskewRecv dispatched at 252 with the vector long
#    there polls at 504, not at 252.
recs = [(P, WRX, 1, deskewrecv(0, 0, E(3)), 0), (10, ETX, 1, transmit(0, 0), 0)] + stored(2 * P)
d, w, _ = run1((recs, {(0, 60): ZW}, [(0, 60)]))
check("DeskewRecv dispatched exactly at a boundary first polls at the next one (a whole period later)", d == [] and w[(0, 60)] == NOTE, (d, w[(0, 60)][:4].hex()))

# 4. A vector that never comes leaves the queue held for the rest of the run: no diagnostic, and the record behind it
#    never dispatches (a Sync on MEM queue 1 is released by a Notify behind the DeskewRecv: it never is).
recs = [(5, WRX, 1, deskewrecv(0, 0, E(3)), 0), (6, WRX, 1, NOTIFY, 0), (3, 1, 1, SYNC, 0), (7, 1, 1, mem(0, 0, E(0)), 0)]
d, w, s = run1((recs, {(1, 0): ZW}, []))
check("no vector ever: no diagnostic, the queue stays held, the Notify behind it never dispatches (the Sync is never released)", d == [] and s == [], (d, s))

# 5. Bad operands are the diagnostics of Receive, at the dispatch of the DeskewRecv, which then holds nothing.
d, _, _ = run1(([(10, WRX, 1, deskewrecv(1, 0, E(0)), 0), (12, VXMQ, 1, deskewrecv(0, 0, E(0)), 0), (13, WTX, 1, deskewrecv(0, 0, E(0)), 0)], {}, []))
check("DeskewRecv: a wrong chip_id is OPR-001 at the dispatch; off an RX queue it is QUE-002",
      d == [f"T diag 10 queue {WRX} OTS-E-OPR-001", f"T diag 12 queue {VXMQ} OTS-E-QUE-002", f"T diag 13 queue {WTX} OTS-E-QUE-002"], d)
# a profile with c2c.links = 11 (ISCA22 sec. 2.2): link 11 does not exist
cfg11 = json.loads(json.dumps(cfg)); cfg11["c2c"]["links"] = 11
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(cfg11, f); P11 = f.name
p = subprocess.run([SIM, P11, "--trace"], input="\n".join(text(([(10, WRX, 1, deskewrecv(0, 11, E(0)), 0), (11, WRX, 1, deskewrecv(0, 10, E(0)), 0), (12, WRX, 1, NOTIFY, 0), (3, 1, 1, SYNC, 0), (4, 1, 1, mem(0, 0, E(0)), 0)], {(1, 0): ZW}, []))) + "\n", capture_output=True, text=True)
os.unlink(P11)
check("DeskewRecv: link 11 of an 11-link profile is OPR-001 and holds nothing; link 10 exists and holds the queue (the Notify behind it never runs)",
      p.returncode == 0 and [l for l in p.stdout.splitlines() if l.startswith("T diag")] == [f"T diag 10 queue {WRX} OTS-E-OPR-001"] and not [l for l in p.stdout.splitlines() if l.startswith("T srf")], (p.returncode, p.stdout[:300], p.stderr))

# 6. Aligned Notify after a skewed start (ISCA22 §3.2): the parent (chip 0) starts 1000 cycles after the child
#    (chip 1). The parent: Deskew, Transmit at the boundary t3, Deskew, Notify one cycle after the next boundary. The
#    child: DeskewRecv, Notify. The child leaves the loop floor(L / 252) + 1 epochs after the Transmit and both
#    Notify at the same global time; a probe on MEM queue 1 (Sync, then a Read) shows the Notify cycle through the
#    barrier release (Notify + the release latency).
START = 1000
T3 = ((START + 10) // P + 1) * P          # the boundary the parent's Deskew (dispatched at START + 10) ends at
for lat in sorted({LMIN, L, LMAX}):
    n_ep = lat // P + 1
    exit_at = T3 + n_ep * P               # the child's Receive dispatches here
    child = ([(10, WRX, 1, deskewrecv(0, 0, E(3)), 0), (11, WRX, 1, NOTIFY, 0), (5, 1, 1, SYNC, 0), (6, 1, 1, mem(0, 0, E(0)), 0)]
             + stored(exit_at), {(0, 60): ZW, (1, 0): ZW}, [(0, 60)])
    # the parent's Notify is one cycle after the child's Receive, as the child's Notify behind the Receive
    parent = ([(10, ETX, 1, DESKEW, 0), (T3 - START, ETX, 1, transmit(1, 0), 0), (T3 - START + 1, ETX, 1, DESKEW, 0),
               (exit_at - START + 1, ETX, 1, NOTIFY, 0), (5, 1, 1, SYNC, 0), (6, 1, 1, mem(0, 0, E(0)), 0)], {(1, 0): ZW}, [])
    (dp, wp, sp), (dc, wc, sc) = runk([(parent, START), (child, 0)], ("--link-latency", str(lat)))
    # the cycle of the probe's read after a Notify at exit_at + 1, measured on one chip with the same probe
    ref = run1(([(5, 1, 1, SYNC, 0), (6, 1, 1, mem(0, 0, E(0)), 0), (exit_at + 1, VXMQ, 1, NOTIFY, 0)], {(1, 0): ZW}, []))[2]
    check(f"L = {lat}: the child exits the polling loop {n_ep} epoch(s) after the Transmit (stores the notification at {exit_at})",
          dc == [] and wc[(0, 60)] == NOTE, (dc, wc[(0, 60)][:4].hex()))
    check(f"L = {lat}: both chips Notify at the same global time ({exit_at + 1})", dp == [] and sp == [sc[-1]] == ref and len(ref) == 1, (dp, sp, sc, ref))

print("deskewrecv:", "all pass" if ok else "FAILURES")
sys.exit(0 if ok else 1)
