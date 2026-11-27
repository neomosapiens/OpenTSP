"""Directed end-to-end tests of the host path (the host path): the CSR space,
the host's DMA engine and doorbells (H lines), the host I/O instructions PciReceive, PciSend,
HostWait and HostNotify on the C2C queues, MSI-X delivery, the start cycle, text injection
and the diagnostics OTS-E-HIO-001/002, on the Sail emulator. Profile-portable: a vector is
vlanes = 16 x superlanes bytes and only those bytes are compared.
usage: host_directed.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
cfg = json.load(open(CFG))
VL = 16 * cfg["geometry"]["superlanes"]
H = cfg["host"]; CPV = H["cycles_per_vector"]; DEPTH = H["ingress_depth"]
TX_GAP = cfg["layout"]["c2c_tx_gap"]; RX_GAP = cfg["layout"]["c2c_rx_gap"]
D_C2C = cfg["timing"]["c2c"]["d_func"]; D_MEM = cfg["timing"]["mem"]["d_func"]; D_CU = cfg["timing"]["cu"]["d_skew"]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def c2c(op, s=0, low=0): return (5 << 28) | (op << 22) | (s << 10) | low
def pci_receive(s): return c2c(5, s)
def pci_send(s): return c2c(6, s)
def host_wait(i): return c2c(7, 0, i)
def host_notify(n): return c2c(8, 0, n)
def E(i): return i
def W(i): return 32 + i
WTX, WRX, ETX, ERX, VXMQ = 126, 128, 140, 142, 88
VAL = lambda k: bytes((7 * l + 3 + 11 * k) & 0xFF for l in range(VL))


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - D_MEM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def sim(text, *args, cfg_path=None):
    return subprocess.run([SIM, cfg_path or CFG, "--trace", *args], input=text, capture_output=True, text=True, timeout=600)


def run(recs=(), sram={}, dumps=(), host=(), units=(), args=(), cfg_path=None, expect_rc=0):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(host)
    p = sim("\n".join(ls) + "\n", *args, cfg_path=cfg_path)
    assert p.returncode == expect_rc, (p.returncode, p.stderr)
    return p


def diags(p): return [l for l in p.stdout.splitlines() if l.startswith("T diag")]
def tl(p, kind): return [l for l in p.stdout.splitlines() if l.startswith(f"T {kind} ")]
def words(p): return {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "W"}
def ys(p): return [(int(t[1]), bytes(int(x, 16) for x in t[2:])) for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "Y"]
def xs(p): return [(t[1], t[2]) for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "X"]
def dma(cycle, v): return f"H dma {cycle:x} " + " ".join(f"{b:x}" for b in v)


# 1. Emplacement (ISCA20 sec. II-6): the host's DMA engine pushes 4 vectors, 4 PciReceive on an RX queue
#    write them on S24..S27 (west row), 4 MEM writes commit them to slice 44.
T = 100
recs = []
for k in range(4):
    recs.append((T + k, ERX, 1, pci_receive(W(24 + k)), 0))
    drain(recs, 44, 10 + k, 24 + k, "W", RX_GAP, T + k + D_C2C)
p = run(recs, dumps=[(44, 10 + k) for k in range(4)], host=[dma(0, VAL(k)) for k in range(4)])
w = words(p)
check("emplacement: 4 vectors pushed by the host reach slice 44 through PciReceive + Write",
      diags(p) == [] and all(w[(44, 10 + k)] == VAL(k) for k in range(4)), (diags(p), p.stderr))
check(f"arrivals are {CPV} cycles apart: T dma 0, {CPV}, {2 * CPV}, {3 * CPV} in",
      tl(p, "dma") == [f"T dma {k * CPV} in" for k in range(4)], tl(p, "dma"))

# 2. A vector requested at 0 and one at 3 arrive at 0 and CPV.
p = run(host=[dma(0, VAL(0)), dma(3, VAL(1))])
check(f"a vector requested at 0 and one at 3 arrive at 0 and {CPV}", tl(p, "dma") == ["T dma 0 in", f"T dma {CPV} in"], tl(p, "dma"))

# 3. (the ingress FIFO overflow of the earlier version is gone: the DMA engine waits, section 20)

# 4. PciReceive before anything arrived is TIM-001 (as Receive).
p = run([(T, ERX, 1, pci_receive(W(24)), 0)])
check("PciReceive with an empty ingress FIFO is OTS-E-TIM-001", diags(p) == [f"T diag {T} queue {ERX} OTS-E-TIM-001"], diags(p))

# 5. PciReceive of a stream outside S24..S31 does not decode; neither does HostNotify 0.
p = run([(T, ERX, 1, pci_receive(E(3)), 0), (T + 1, ERX, 1, pci_receive(W(23)), 0), (T + 2, VXMQ, 1, host_notify(0), 0)])
check("PciReceive of S3 / S23 and HostNotify 0 are OTS-E-DEC-003",
      diags(p) == [f"T diag {T} queue {ERX} OTS-E-DEC-003", f"T diag {T + 1} queue {ERX} OTS-E-DEC-003", f"T diag {T + 2} queue {VXMQ} OTS-E-DEC-003"], diags(p))

# 6. PciSend of a stream appears as a Y line at its cycle; two within CPV cycles: OTS-E-HIO-001.
recs, sram = [], {}
feed(recs, sram, 44, 1, VAL(5), 0, "W", TX_GAP, T)
recs.append((T, WTX, 1, pci_send(W(0)), 0))
p = run(recs, sram)
check("PciSend: the vector read from the stream is the Y line of its cycle", diags(p) == [] and ys(p) == [(T, VAL(5))], (diags(p), ys(p)[:1]))
check("the trace has T dma out", tl(p, "dma") == [f"T dma {T} out"], tl(p, "dma"))
recs, sram = [], {}
feed(recs, sram, 44, 1, VAL(5), 0, "W", TX_GAP, T)
feed(recs, sram, 44, 2, VAL(6), 1, "W", TX_GAP, T + CPV // 2)
feed(recs, sram, 44, 3, VAL(7), 2, "W", TX_GAP, T + CPV)
recs += [(T, WTX, 1, pci_send(W(0)), 0), (T + CPV // 2, WTX, 1, pci_send(W(1)), 0), (T + CPV, WTX, 1, pci_send(W(2)), 0)]
p = run(recs, sram)
check(f"two PciSend within {CPV} cycles: OTS-E-HIO-001 and no effect; the third after {CPV} is accepted",
      diags(p) == [f"T diag {T + CPV // 2} queue {WTX} OTS-E-HIO-001"] and [c for c, _ in ys(p)] == [T, T + CPV], (diags(p), [c for c, _ in ys(p)]))
p = run([(T, ERX, 1, pci_send(W(0)), 0)])
check("PciSend on an RX queue is OTS-E-QUE-002", diags(p) == [f"T diag {T} queue {ERX} OTS-E-QUE-002"], diags(p))

# 7. HostNotify n: MSI-X vector n; IRQ_CAUSE (0x30) bit n, IRQ_COUNT (0x32); a masked vector gives no interrupt.
p = run([(T, VXMQ, 1, host_notify(3), 0)], host=["R 30", "R 32"])
check("HostNotify 3: T irq msix 3, IRQ_CAUSE = 8, IRQ_COUNT = 1", tl(p, "irq") == [f"T irq {T + D_CU} msix 3"] and xs(p) == [("030", "8"), ("032", "1")], (tl(p, "irq"), xs(p)))
p = run([(T, VXMQ, 1, host_notify(3), 0)], host=["H csr 0 w 31 0", "R 30", "R 32", "R 31"])
check("a vector masked in IRQ_MASK sets its cause bit but gives no interrupt", tl(p, "irq") == [] and xs(p) == [("030", "8"), ("032", "0"), ("031", "0")], (tl(p, "irq"), xs(p)))
p = run([(T, VXMQ, 1, host_notify(3), 0)], host=[f"H csr {T + 5:x} w 30 8", "R 30"])
check("writing 1 to a cause bit clears it (W1C)", xs(p) == [("030", "0")], xs(p))

# 8. CSR reads and the register file: CHIP_ID, doorbells, a read of an unmapped address.
p = run(host=["R 0", "R 300", "R 7f0", "H csr 5 r 0", "H csr 6 w 301 1", "H csr 7 r 301"])
check("CHIP_ID reads 0; a doorbell and an unmapped register read 0", xs(p) == [("000", "0"), ("300", "0"), ("7f0", "0")], xs(p))
check("H csr r prints T csr; a doorbell written 1 reads 1", tl(p, "csr") == ["T csr 5 000 0", "T csr 7 301 1"], tl(p, "csr"))

# 9. HostWait / doorbell: HostWait 3 on a queue, then HostNotify 5. The host rings at 50: the next
#    parcel dispatches at 51 and its HostNotify gives T irq 51. A doorbell rung before the HostWait is taken by it.
recs = [(10, VXMQ, 1, host_wait(3), 0), (11, VXMQ, 1, host_notify(5), 0)]
p = run(recs, host=["H db 32 3"])
check("HostWait parks its queue until the doorbell is rung at 50: the next instruction dispatches at 51",
      diags(p) == [] and tl(p, "irq") == [f"T irq {51 + D_CU} msix 5"], (diags(p), tl(p, "irq")))
p = run(recs, host=["H db 5 3"])
check("a doorbell rung before HostWait is taken by it: the next instruction dispatches at 11", tl(p, "irq") == [f"T irq {11 + D_CU} msix 5"], tl(p, "irq"))
p = run(recs, host=["R 303"])
check("with no ring the parked queue never resumes: no T irq, no diagnostic", diags(p) == [] and tl(p, "irq") == [], (diags(p), tl(p, "irq")))
p = run(recs + [(12, VXMQ, 1, host_wait(3), 0), (13, VXMQ, 1, host_notify(6), 0)], host=["H db 32 3", "H db 40 3"])
check("each HostWait takes one ring", [l.split()[4] for l in tl(p, "irq")] == ["5", "6"] and tl(p, "irq")[1].split()[2] == str(0x40 + 1 + D_CU), tl(p, "irq"))
nd = H["doorbells"]
p = run([(10, VXMQ, 1, host_wait(nd), 0)])
check("HostWait of a doorbell >= host.doorbells is OTS-E-OPR-001", diags(p) == [f"T diag {10 + D_CU} queue {VXMQ} OTS-E-OPR-001"], diags(p))

# 10. H start: the queues begin dispatching at the start cycle; a program without host traffic is
#     the same run shifted by it.
recs, sram = [], {}
feed(recs, sram, 44, 1, VAL(1), 0, "W", TX_GAP, 30)
drain(recs, 0, 9, 0, "E", RX_GAP, 60)
recs.append((30, WTX, 1, c2c(3, 0, 0), 0))
p0 = run(recs, sram, dumps=[(44, 1)])
p1 = run(recs, sram, dumps=[(44, 1)], host=["H start 64"])
def shifted(p, d):
    out = []
    for l in p.stdout.splitlines():
        t = l.split()
        if l.startswith("T srf") or l.startswith("T sram"): t[2] = str(int(t[2]) + d)
        elif l.startswith("T diag"): t[2] = str(int(t[2]) + d)
        out.append(" ".join(t))
    return out
check("H start 64 (hex) shifts every trace line by 100 cycles",
      [l for l in shifted(p0, 100) if l.startswith("T")] == [l for l in p1.stdout.splitlines() if l.startswith("T")] and
      len([l for l in p1.stdout.splitlines() if l.startswith("T")]) > 0, (p0.stdout[:300], p1.stdout[:300]))

# 11. H inject: the host pushes text into a queue's IQ: HostNotify 9 (one 32-bit unit) injected into the
#     VXM queue at cycle 20 is dispatched on arrival.
unit = host_notify(9)
p = run(host=[f"H inject 14 {VXMQ:x} {unit:x}"])
check("H inject delivers host text to the IQ: HostNotify 9 runs", tl(p, "irq") != [] and tl(p, "irq")[0].endswith("msix 9"), (tl(p, "irq"), p.stderr))

# A timed control-channel unit enters the decoder even with no initial image.
p = run(host=["H inject 14 ffff 70000000"])
check("timed control IQ input reaches decode", diags(p) == ["T diag 20 queue 65535 OTS-E-DEC-003"], diags(p))

# 12. Bad input: each is an input error, exit 2, with a message.
for name, line in [("H with an unknown word", "H frob 1"), ("H db beyond host.doorbells", f"H db 1 {nd:x}"), ("H csr beyond 0xfff", "H csr 1 r 1000"),
                   ("R beyond 0xfff", "R 1000"), ("H csr without r/w", "H csr 1 x 10"), ("H inject on a missing queue", "H inject 1 90 1"),
                   ("H dma with too many bytes", "H dma 1 " + " ".join(["1"] * (VL + 1)))]:
    p = sim(line + "\n")
    check(f"bad input: {name} is exit 2 with a message", p.returncode == 2 and "input line 1" in p.stderr, (p.returncode, p.stderr))

# 13. A smaller ingress FIFO (host.ingress_depth 2) and slower DMA (host.cycles_per_vector 3): the third vector waits, nothing is dropped.
c2 = json.loads(json.dumps(cfg)); c2["host"]["ingress_depth"] = 2; c2["host"]["cycles_per_vector"] = 3
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(c2, f); pc = f.name
p = run(host=[dma(0, VAL(0)), dma(0, VAL(1)), dma(0, VAL(2))], cfg_path=pc)
os.unlink(pc)
check("host.ingress_depth 2, cycles_per_vector 3: arrivals at 0 and 3; the third waits at 6 (no PciReceive frees a slot), no diagnostic",
      tl(p, "dma") == ["T dma 0 in", "T dma 3 in", "T dma 6 wait"] and diags(p) == [], (tl(p, "dma"), diags(p)))

# 14. The patch-free path (ISCA20 sec. II-6): the host pushes text as two DMA vectors, two RX queues write them as the
#     stream pair S24/S25 (PciReceive, one cycle), and an Ifetch on a VXM queue brings them into its IQ, where they
#     run: HostNotify 9 then HostNotify 10. No MEM slice, no patching of an object. (Ifetch has one stream operand and reads two
#     consecutive vectors of it, at t and t + 1: both PciReceive write the same stream, one cycle apart.)
def text_vec(units):
    return b"".join(u.to_bytes(4, "little") for u in units).ljust(VL, b"\xff")
FL = cfg["icu"]["fetch_latency"]; VG = cfg["layout"]["vxm_gap"]
ex = T + D_C2C + (VG - RX_GAP)                    # the stream pair is at the VXM read gap in this cycle; Ifetch has d_skew 0 (CU)
recs = [(T, ERX, 1, pci_receive(W(24)), 0), (T + 1, ERX, 1, pci_receive(W(24)), 0), (ex - D_CU, VXMQ, 1, (1 << 24) | (W(24) << 10), 0)]
p = run(recs, host=[dma(0, text_vec([host_notify(9)])), dma(0, text_vec([host_notify(10)]))])
irq = tl(p, "irq")
check("host text through PciReceive + Ifetch runs: HostNotify 9 at the fetch, HostNotify 10 after the padding of the first vector",
      diags(p) == [] and len(irq) == 2 and irq[0] == f"T irq {ex + 1 + FL} msix 9" and irq[1].endswith("msix 10"), (diags(p), irq))
recs[2] = (ex - D_CU + 1, VXMQ, 1, (1 << 24) | (W(24) << 10), 0)
p = run(recs, host=[dma(0, text_vec([host_notify(9)])), dma(0, text_vec([host_notify(10)]))])
check("a late Ifetch samples the final vector then reports TIM-001 on the next edge", [l for l in diags(p)] == [f"T diag {ex + 2} queue {VXMQ} OTS-E-TIM-001"], diags(p))

# 15. (the routed remote CSR access of the earlier version is gone: a csr packet is one hop, section 22)
def sections(p):
    sec, cur = {}, None
    for l in p.stdout.splitlines():
        if l.startswith("C "): cur = int(l.split()[1]); sec[cur] = []
        elif cur is not None: sec[cur].append(l)
    return sec
L = cfg["c2c"]["link_latency"]

# 16. Link packet types and packets sharing a link (HC34 p.45: the link header carries "Type (data, CSR, Sync, Ctrl)").
#     Send carries data, Transmit sync, a remote CSR access (`P csr`) csr, the control unit's own packets ctrl. A csr or ctrl packet
#     is consumed by the receiving chip's control unit (T pkt <cycle> csr|ctrl <link>): it never enters a receive FIFO, so no
#     Receive sees it and OTS-E-LNK-002 counts data and sync only. It occupies ceil(CC * flits / (1 + VL / 8)) cycles of a link
#     (flits: 3 for csr, 2 for ctrl, 1 + VL / 8 for data and sync) and leaves at the first cycle at which the link is free (the
#     packets before it and the vectors accepted before it are gone from the link). A Send released into its cycles is OTS-E-LNK-003.
CC = cfg["c2c"]["cycles_per_vector"]
FV = 1 + (VL + 7) // 8
KC = -(-CC * 3 // FV); KX = -(-CC * 2 // FV)
def send_op(chip, link, s): return (5 << 28) | (chip << 16) | (s << 10) | (link << 6)
def chip_text(chip, sends=(), extra=()):
    """The program text of one chip: Send to `peer` on `link` at each (cycle, peer, link) of `sends`."""
    recs, sram = [], {}
    for j, (cyc, peer, link) in enumerate(sends):
        feed(recs, sram, 44, 1 + j, VAL(j), j % 8, "W", TX_GAP, cyc)
        recs.append((cyc, WTX, 1, send_op(peer, link, W(j % 8)), 0))
    ls = [f"C {chip}"] + [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    return ls + q_lines(lower_records(recs)) + list(extra)
def pkts(sec, kind=None): return [l for l in sec if l.startswith("T pkt ") and (kind is None or l.split()[3] == kind)]
def csrs(sec): return [l for l in sec if l.startswith("T csr ")]
def lnk3(sec): return [l for l in sec if l.startswith("T diag") and "OTS-E-LNK-003" in l]
S = 400

h = S - KC                                        # the packet has left the link when the Send goes
p = sim("\n".join(chip_text(0, [(S, 1, 0)], [f"P csr {h:x} 0 r 0"]) + ["C 1"]) + "\n", "--chips", "2")
sc = sections(p)
check(f"a remote read that is out of the link before a Send goes at once: arrives at h + L (T pkt {h + L} csr 0), answered at h + 2L = {h + 2 * L}; no diagnostic",
      p.returncode == 0 and pkts(sc[1], "csr") == [f"T pkt {h + L} csr 0"] and csrs(sc[0]) == [f"T csr {h + 2 * L} 1 000 1"] and not [l for l in sc[0] if l.startswith("T diag")], (p.returncode, p.stderr, sc))
h = S - KC + 1                                    # the Send is released into the packet's cycles: the packet is not delayed, the Send is the error
p = sim("\n".join(chip_text(0, [(S, 1, 0)], [f"P csr {h:x} 0 r 0"]) + ["C 1"]) + "\n", "--chips", "2")
sc = sections(p)
check(f"a Send released into the cycles of a csr packet in flight is OTS-E-LNK-003 and the packet is not delayed: answered at h + 2L = {h + 2 * L}",
      p.returncode == 0 and lnk3(sc[0]) == [f"T diag {S} queue {WTX} OTS-E-LNK-003"] and pkts(sc[1], "csr") == [f"T pkt {h + L} csr 0"] and csrs(sc[0]) == [f"T csr {h + 2 * L} 1 000 1"], (p.returncode, p.stderr, sc))
c = S + CC                                        # asked after the Send went: it waits for the vector (the control unit knows it)
p = sim("\n".join(chip_text(0, [(S, 1, 0)], [f"P csr {S + 1:x} 0 r 0"]) + ["C 1"]) + "\n", "--chips", "2")
sc = sections(p)
check(f"a remote read asked while the link carries a vector waits for it: sent at S + CC = {c}, answered at {c + 2 * L}; no diagnostic",
      p.returncode == 0 and pkts(sc[1], "csr") == [f"T pkt {c + L} csr 0"] and csrs(sc[0]) == [f"T csr {c + 2 * L} 1 000 1"] and not [l for l in sc[0] if l.startswith("T diag")], (p.returncode, p.stderr, sc))
p = sim("\n".join(["C 0", "P csr a 0 r 0", "P csr b 0 r 1", "P csr b 0 r 2"]) + "\n", "--chips", "2")
sc = sections(p)
check(f"three packets on one link (asked at 10, 11, 11) occupy {KC} cycles each and leave in order: sent at 10, 10 + {KC}, 10 + {2 * KC}",
      p.returncode == 0 and pkts(sc[1], "csr") == [f"T pkt {10 + L + j * KC} csr 0" for j in range(3)], (p.returncode, p.stderr, sc))
# the reply: it leaves in the cycle the request arrives (5 + L) unless chip 1's link holds a vector accepted before
Sb = 5 + L - 3                                    # chip 1's own Send, 3 cycles before the request arrives: the reply waits for it
p = sim("\n".join(["C 0", "P csr 5 0 r 0"] + chip_text(1, [(Sb, 0, 0)])) + "\n", "--chips", "2")
sc = sections(p); c = Sb + CC
check(f"the reply waits for a vector chip 1's link accepted before: Send at {Sb}, reply sent at {c}, answered at {c + L}; no OTS-E-LNK-001/003",
      p.returncode == 0 and csrs(sc[0]) == [f"T csr {c + L} 1 000 1"] and pkts(sc[0], "csr") == [f"T pkt {c + L} csr 0"] and not [l for l in sc[1] if l.startswith("T diag")], (p.returncode, p.stderr, sc))
Sb = 5 + L + KC - 1                               # chip 1's Send is released into the reply's cycles: LNK-003 on chip 1, the reply goes at once
p = sim("\n".join(["C 0", "P csr 5 0 r 0"] + chip_text(1, [(Sb, 0, 0)])) + "\n", "--chips", "2")
sc = sections(p)
check(f"a Send of chip 1 released into the reply's cycles is OTS-E-LNK-003 (T diag {Sb}); the reply is not delayed: answered at 5 + 2L = {5 + 2 * L}",
      p.returncode == 0 and lnk3(sc[1]) == [f"T diag {Sb} queue {WTX} OTS-E-LNK-003"] and csrs(sc[0]) == [f"T csr {5 + 2 * L} 1 000 1"], (p.returncode, p.stderr, sc))
# a doorbell write reaches the chip next to it and releases a HostWait there
recs = [(10, VXMQ, 1, host_wait(3), 0), (11, VXMQ, 1, host_notify(5), 0)]
p = sim("\n".join(["C 1"] + q_lines(lower_records(recs)) + ["C 0", "P csr 0 0 w 303 1"]) + "\n", "--chips", "2")
sc = sections(p)
check(f"a remote doorbell write reaches chip 1 at L = {L}: its HostWait resumes at L + 1 (T irq {L + 1 + D_CU} msix 5 in chip 1's section)",
      p.returncode == 0 and f"T irq {L + 1 + D_CU} msix 5" in sc[1], (p.returncode, p.stderr, sc.get(1)))

# control packets: P ctrl <cycle> <link>; consumed by the control unit, never in a FIFO
p = sim("C 0\nP ctrl a 0\nP csr b 0 r 0\n", "--chips", "2")
sc = sections(p); c = 10 + KX
check(f"a ctrl packet sent at 10 occupies {KX} cycles: it arrives at 10 + L (T pkt {10 + L} ctrl 0) and a csr packet asked at 11 follows at {c}",
      p.returncode == 0 and pkts(sc[1], "ctrl") == [f"T pkt {10 + L} ctrl 0"] and pkts(sc[1], "csr") == [f"T pkt {c + L} csr 0"], (p.returncode, p.stderr, sc))
recs = [(10 + L + 5, ERX, 1, (5 << 28) | (1 << 22) | (0 << 16) | (W(0) << 10) | (0 << 6), 0)]
txt = ["C 0", "P ctrl a 0", "P csr 100 0 r 0"] + ["C 1"] + q_lines(lower_records(recs))
p = sim("\n".join(txt) + "\n", "--chips", "2")
sc = sections(p)
check("a Receive on the link that carried a ctrl packet and a csr packet finds no vector (OTS-E-TIM-001): they never enter the FIFO",
      p.returncode == 0 and pkts(sc[1], "ctrl") and pkts(sc[1], "csr") and [l for l in sc[1] if l.startswith("T diag")] == [f"T diag {10 + L + 5} queue {ERX} OTS-E-TIM-001"], (p.returncode, p.stderr, sc.get(1)))
n = cfg["c2c"]["rx_depth"] + 4
p = sim("\n".join(["C 0"] + ["P ctrl 0 0"] * n) + "\n", "--chips", "2")
sc = sections(p)
check(f"{n} ctrl packets on one link (more than c2c.rx_depth = {n - 4}) with no Receive: no OTS-E-LNK-002", p.returncode == 0 and len(pkts(sc[1], "ctrl")) == n and not [l for l in sc[1] if "LNK-002" in l] and [l for l in sc[1] if l.startswith("E ")] == ["E - -1 0"], (p.returncode, p.stderr, sc.get(1, [])[:4]))
sends = [(S + j * CC, 1, 0) for j in range(n)]
p = sim("\n".join(chip_text(0, sends) + ["C 1"]) + "\n", "--chips", "2")
sc = sections(p)
check(f"{n} data vectors on one link with no Receive: the last 4 are dropped, OTS-E-LNK-002 each (data and sync count)", p.returncode == 0 and len([l for l in sc[1] if "OTS-E-LNK-002" in l and l.startswith("T diag")]) == 4, (p.returncode, p.stderr, sc.get(1, [])[:6]))
p = sim("C 0\nP ctrl 0 1\n", "--chips", "2")
check("a ctrl packet on a link that is not connected is exit 2 with a message", p.returncode == 2 and "not connected" in p.stderr, (p.returncode, p.stderr))
p = sim("C 0\nP sync 0 0\n", "--chips", "2")
check("P with an unknown packet type is exit 2", p.returncode == 2 and "unknown packet type" in p.stderr, (p.returncode, p.stderr))

# 17. One chip, every link loops back to the chip itself: a Transmit (sync) and a ctrl packet on link 0. The Receive takes the
#     sync vector; the ctrl packet, which arrives after it, is consumed by the control unit (T pkt) and never reaches a
#     Receive (OTS-E-TIM-001). It leaves in a gap: asked at S + 2 inside the Transmit's window, it goes at S + CC.
def transmit_op(chip, link): return (5 << 28) | (3 << 22) | (chip << 16) | (link << 6)
def receive_op(chip, link, s): return (5 << 28) | (1 << 22) | (chip << 16) | (s << 10) | (link << 6)
S1 = 200
recs = [(S1, WTX, 1, transmit_op(0, 0), 0), (S1 + L + 1, ERX, 1, receive_op(0, 0, W(0)), 0), (S1 + CC + L + 4, ERX, 1, receive_op(0, 0, W(1)), 0)]
p = run(recs, host=[f"P ctrl {S1 + 2:x} 0"])
check(f"loopback: the Transmit's sync vector is taken by a Receive; the ctrl packet (asked at {S1 + 2}, sent at {S1 + CC}, T pkt {S1 + CC + L} ctrl 0) is not: the second Receive is OTS-E-TIM-001",
      tl(p, "pkt") == [f"T pkt {S1 + CC + L} ctrl 0"] and diags(p) == [f"T diag {S1 + CC + L + 4} queue {ERX} OTS-E-TIM-001"], (tl(p, "pkt"), diags(p)))
n = cfg["c2c"]["rx_depth"] + 3
p = run(host=["P ctrl 0 0"] * n)
check(f"{n} ctrl packets on a looped link with no Receive: {n} T pkt lines, no OTS-E-LNK-002", len(tl(p, "pkt")) == n and diags(p) == [], (len(tl(p, "pkt")), diags(p)))

# control traffic does not change what a program sees: with per-transfer latencies (--link-latency-seed jitter), the cycle at
# which a Transmit's vector is first received is the same with and without ctrl packets on the link (they have a generator of their own)
S2j = 300
scan = [(S2j + 200 + j, ERX, 1, receive_op(0, 0, W(j % 8)), 0) for j in range(40)]
def first_ok(extra_host):
    p = run([(S2j, WTX, 1, transmit_op(0, 0), 0)] + scan, host=extra_host, args=("--link-jitter-seed", "5"))
    bad = {int(l.split()[2]) for l in diags(p)}
    return next(c for c in range(S2j + 200, S2j + 240) if c not in bad)
a0, a1 = first_ok([]), first_ok(["P ctrl 0 0", "P ctrl 1 0", "P ctrl 100 0"])
check(f"jitter: the Transmit's vector is first received at {a0} with or without ctrl packets on the link", a0 == a1 and S2j + 209 <= a0 <= S2j + 228, (a0, a1))

# a link is one serial channel, so under per-transfer jitter no packet of any type passes an earlier one on the same link (RV2-B-06):
# the in-order clamp is one per link over data, sync, csr and ctrl packets. Loopback, one Transmit (sync vector) and one ctrl packet
# on link 0, in both orders; the data arrival is the first Receive cycle that finds a vector, the ctrl arrival is its `T pkt` line.
def jitter_order(seed, tx_at, ctrl_at):
    scan = [(tx_at + 150 + j, ERX, 1, receive_op(0, 0, W(j % 8)), 0) for j in range(130)]
    p = run([(tx_at, WTX, 1, transmit_op(0, 0), 0)] + scan, host=[f"P ctrl {ctrl_at:x} 0"], args=("--link-jitter-seed", str(seed)))
    bad = {int(l.split()[2]) for l in diags(p)}
    data = next((c for c in range(tx_at + 150, tx_at + 280) if c not in bad), -1)       # -1: no Receive found the vector (a profile whose Transmit is timed otherwise: the test is not run on it, only captured)
    return data, int(tl(p, "pkt")[0].split()[2])
TXJ = 300
for ctrl_at, first in ((TXJ - KX, "ctrl"), (TXJ + 2, "data")):     # the ctrl packet fits in the KX cycles just before the Transmit / follows it
    out = [jitter_order(sd, TXJ, ctrl_at) for sd in range(1, 13)]
    swapped = [(sd, d, c) for sd, (d, c) in enumerate(out, 1) if not (d > c if first == "ctrl" else c > d)]
    check(f"jitter, 12 seeds: with the {first} packet sent first, the other (sent after it) never arrives before it", not swapped,
          f"(seed, data arrival, ctrl arrival) reordered: {swapped}")

# the departure cycle of a packet does not depend on how the harness schedules the chips: the same output with every
# lock-step window (--window), and each link latency source is used by the packet (fixed, per link, per transfer)
mix = "\n".join(chip_text(0, [(S, 1, 0), (S + CC + 1, 1, 0)], ["P ctrl 8 0", f"P csr {S + 5:x} 0 r 0", f"P csr {S + 6:x} 0 w 303 1", f"P csr {S + 7:x} 0 r 303"]) + chip_text(1, [(S + 3, 0, 0)])) + "\n"
quad = "\n".join(chip_text(0, [], ["P ctrl 3 0", "P csr 5 1 r 0", "P csr 6 2 r 0", "P csr 7 2 w 303 1", "H csr 9 r 0"]) + chip_text(1, [(5 + L - 3, 3, 1)], ["P csr 20 0 r 0"])
                 + chip_text(2, [(6 + L + 1, 0, 1)], ["H db 30 1"]) + ["C 3", "H dma 0 1 2 3"]) + "\n"
for name, txt, chips in (("two chips", mix, "2"), ("four chips", quad, "4")):
    for extra in [(), ("--link-latency", "209"), ("--link-latency", "228"), ("--link-jitter-seed", "5")] + [("--link-latency-seed", str(k)) for k in range(1, 9)]:
        base = sim(txt, "--chips", chips, *extra)
        outs = [sim(txt, "--chips", chips, "--window", str(w), *extra) for w in (1, CC - 1, CC, CC + 1, 37 + CC, 100 + CC)]
        check(f"{name} {' '.join(extra) or 'default latency'}: the output is the same for every lock-step window (--window 1, {CC}, ...)",
              base.returncode == 0 and all(o.returncode == 0 and o.stdout == base.stdout for o in outs) and any(l.startswith("T csr") for l in base.stdout.splitlines()),
              (base.returncode, base.stderr[-200:], [o.returncode for o in outs]))

# ---------------------------------------------------------------------------------------------------------------------------
# The host interface on every chip and the causal link rules. Marker: RED-BEGIN
# 18. Every chip has its own host interface (the published sources give every chip its PCIe host interface in its C2C module:
#     ISCA20 sec. II-6; ISCA22 sec. 5.2): an H line after a `C <chip>` line acts on that chip, which has its own CHIP_ID,
#     ingress FIFO, DMA engine, doorbells and interrupt delivery. The profile has no host.chip key.
p = sim("\n".join(["C 0", dma(0, VAL(0)), "H csr 5 r 0", "C 1", dma(0, VAL(1)), "H csr 5 r 0", "H db 7 2", "R 0"]) + "\n", "--chips", "2")
sc = sections(p)
check("every chip has its own host interface: the H lines after C 1 act on chip 1 (its own CHIP_ID, its own DMA engine)",
      p.returncode == 0 and "T csr 5 000 0" in sc[0] and "T csr 5 000 1" in sc[1] and "T dma 0 in" in sc[0] and "T dma 0 in" in sc[1] and "X 000 1" in sc[1],
      (p.returncode, p.stderr, sc))
c4 = json.loads(json.dumps(cfg)); c4["host"]["chip"] = 0
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(c4, f); pc = f.name
p = sim("", cfg_path=pc)
os.unlink(pc)
check("a profile that still has host.chip is rejected (exit 2, the key is named)", p.returncode == 2 and "host.chip" in p.stderr, (p.returncode, p.stderr))

# 18b. The pipeline shape of the published serving system (SHIP paper sec. 2.1 and 7.1.1: input at the first chip, output at the last): the
#      host feeds chip 0 by DMA, chip 0 PciReceives the vector into MEM and Sends it to chip 1, chip 1 Receives it into MEM and
#      PciSends it to its own host (a Y line of chip 1). No host instruction is routed over a link.
L = cfg["c2c"]["link_latency"]
TP, SP = 100, 300
r0, r1 = [], []
r0.append((TP, ERX, 1, pci_receive(W(24)), 0)); drain(r0, 44, 10, 24, "W", RX_GAP, TP + D_C2C)
feed(r0, {}, 44, 10, VAL(9), 0, "W", TX_GAP, SP); r0.append((SP, WTX, 1, send_op(1, 0, W(0)), 0))
r1.append((SP + L, ERX, 1, receive_op(0, 0, W(0)), 0)); drain(r1, 44, 20, 0, "W", RX_GAP, SP + L + D_C2C)
SO = SP + L + 60
feed(r1, {}, 44, 20, VAL(9), 1, "W", TX_GAP, SO); r1.append((SO, WTX, 1, pci_send(W(1)), 0))
ship = ["C 0"] + q_lines(lower_records(r0)) + [dma(0, VAL(9))] + ["C 1"] + q_lines(lower_records(r1))
p = sim("\n".join(ship) + "\n", "--chips", "2")
sc = sections(p)
y1 = [l for l in sc[1] if l.startswith("Y ")]
check(f"input to chip 0 and output from chip 1 on one system: chip 1's host receives the vector chip 0's host sent (Y at {SO}), no diagnostic, chip 0 has no Y",
      p.returncode == 0 and len(y1) == 1 and y1[0].split()[1] == str(SO) and bytes(int(x, 16) for x in y1[0].split()[2:])[:VL] == VAL(9)
      and not [l for l in sc[0] + sc[1] if l.startswith("T diag")] and not [l for l in sc[0] if l.startswith("Y ")], (p.returncode, p.stderr, sc[0][-4:], sc[1][-4:]))

# 19. MSI-X pending (RV2-B-04). An interrupt is delivered whenever a cause bit is set and unmasked: HostNotify delivers at once when
#     its vector is unmasked, and unmasking a vector whose cause bit is set delivers it (one interrupt) at the cycle of the mask write.
#     IRQ_CAUSE bits stay set until the host clears them (W1C). A masked vector is not lost.
p = run([(T, VXMQ, 1, host_notify(3), 0)], host=["H csr 0 w 31 0", f"H csr {T + 20:x} w 31 ffff", "R 30", "R 32"])
check("a masked vector's cause bit stays set and unmasking delivers it: T irq at the mask write, IRQ_CAUSE 8 (kept), IRQ_COUNT 1",
      tl(p, "irq") == [f"T irq {T + 20} msix 3"] and xs(p) == [("030", "8"), ("032", "1")], (tl(p, "irq"), xs(p)))
p = run([(T, VXMQ, 1, host_notify(3), 0)], host=["H csr 0 w 31 0", f"H csr {T + 20:x} w 31 ffff", f"H csr {T + 30:x} w 31 0", f"H csr {T + 40:x} w 31 ffff"])
check("unmasking again with the cause bit still set delivers again (the bit stays set until the host clears it)",
      tl(p, "irq") == [f"T irq {T + 20} msix 3", f"T irq {T + 40} msix 3"], tl(p, "irq"))
p = run([(T, VXMQ, 1, host_notify(3), 0)], host=["H csr 0 w 31 0", f"H csr {T + 10:x} w 30 8", f"H csr {T + 20:x} w 31 ffff", "R 30"])
check("a cause bit the host cleared while masked is not delivered on unmasking", tl(p, "irq") == [] and xs(p) == [("030", "0")], (tl(p, "irq"), xs(p)))
p = run(host=["H csr 0 w 31 0", "H csr 10 w 31 ffff"])
check("unmasking a vector with no cause bit set delivers nothing", tl(p, "irq") == [], tl(p, "irq"))

# 20. The ingress FIFO and back pressure (RV2-B-07). The DMA engine does not drop data: a vector that finds host.ingress_depth vectors in
#     the FIFO waits and enters in the cycle after a PciReceive has freed a slot.
n = DEPTH + 1
TQ = DEPTH * CPV + 100
recs = []
for k in range(n):
    recs.append((TQ + k, ERX, 1, pci_receive(W(24 + k % 8)), 0))
    drain(recs, 44, 10 + k, 24 + k % 8, "W", RX_GAP, TQ + k + D_C2C)
p = run(recs, dumps=[(44, 10 + k) for k in range(n)], host=[dma(0, VAL(k)) for k in range(n)])
w = words(p)
check(f"back pressure: {n} vectors at host.ingress_depth {DEPTH}: the last waits (T dma {DEPTH * CPV} wait), enters the cycle after the first PciReceive ({TQ + 1}), no diagnostic, all {n} vectors arrive in order",
      diags(p) == [] and tl(p, "dma") == [f"T dma {k * CPV} in" for k in range(DEPTH)] + [f"T dma {DEPTH * CPV} wait", f"T dma {TQ + 1} in"]
      and all(w[(44, 10 + k)] == VAL(k) for k in range(n)), (diags(p), tl(p, "dma")[-3:]))
eline = [l for l in p.stdout.splitlines() if l.startswith("E ")][0]
check("the E line has no diagnostic", eline == "E - -1 0", eline)

# 21. A packet in flight on a link (microarchitecture.md sec. 10). A csr or ctrl packet leaves in arrival order, as soon as the link is
#     free: no packet of the link occupies it and no vector the link accepted before is still in it (the control unit knows what happened
#     before, never what comes after). A Send or Transmit released into the cycles of a packet in flight is a program error
#     (OTS-E-LNK-003, no effect).
CC = cfg["c2c"]["cycles_per_vector"]
FV = 1 + (VL + 7) // 8
KX = -(-CC * 2 // FV)
def transmit_op(chip, link): return (5 << 28) | (3 << 22) | (chip << 16) | (link << 6)
S1 = 200
r = S1 - KX + 1                                  # the ctrl packet (KX cycles) is still on the link in cycle S1
p = run([(S1, WTX, 1, transmit_op(0, 0), 0)], host=[f"P ctrl {r:x} 0"])
check(f"a Transmit released into the cycles of a ctrl packet in flight (sent at {r}, {KX} cycles) is OTS-E-LNK-003; the packet is not delayed (T pkt {r + L})",
      diags(p) == [f"T diag {S1} queue {WTX} OTS-E-LNK-003"] and tl(p, "pkt") == [f"T pkt {r + L} ctrl 0"], (diags(p), tl(p, "pkt")))
r = S1 - KX                                      # the packet has left the link when the Transmit goes
p = run([(S1, WTX, 1, transmit_op(0, 0), 0)], host=[f"P ctrl {r:x} 0"])
check(f"a Transmit in the cycle after a ctrl packet left the link is accepted (the packet sent at {r} is gone at {S1})", diags(p) == [] and tl(p, "pkt") == [f"T pkt {r + L} ctrl 0"], (diags(p), tl(p, "pkt")))
p = run([(S1, WTX, 1, transmit_op(0, 0), 0)], host=[f"P ctrl {S1 + 2:x} 0"])
check(f"a ctrl packet asked while the link carries a vector waits for it (sent at {S1 + CC}), as before", tl(p, "pkt") == [f"T pkt {S1 + CC + L} ctrl 0"] and diags(p) == [], (diags(p), tl(p, "pkt")))
p = run(host=["P ctrl 5 0", "P ctrl 5 0", "P ctrl 5 0"])
check(f"packets of one link leave in arrival order, each when the one before has left the link: asked together at 5 they are sent at 5, {5 + KX}, {5 + 2 * KX}",
      tl(p, "pkt") == [f"T pkt {5 + j * KX + L} ctrl 0" for j in range(3)], tl(p, "pkt"))
# RV2-B-13: the departure does not depend on a later arrival. Chip 0's doorbell write reaches chip 1 at a and releases a Transmit
# there (dispatch a + 1); a ctrl packet of chip 1 asked at a - 18 leaves at once.
def release_case(h):
    a = h + L; rq = a - 18
    recs1 = [(10, ETX, 1, host_wait(3), 0), (11, ETX, 1, transmit_op(0, 0), 0)]
    txt = ["C 0", f"P csr {h:x} 0 w {0x303:x} 1", "C 1"] + q_lines(lower_records(recs1)) + [f"P ctrl {rq:x} 0"]
    p = sim("\n".join(txt) + "\n", "--chips", "2")
    return a, rq, p, sections(p)
for h in (0, 2):
    a, rq, p, sc = release_case(h)
    check(f"a ctrl packet asked at {rq} leaves at once although a doorbell that arrives later (at {a}) releases a Transmit (h = {h}): T pkt {rq + L} on chip 0, no diagnostic",
          p.returncode == 0 and [l for l in sc[0] if l.startswith("T pkt") and "ctrl" in l] == [f"T pkt {rq + L} ctrl 0"] and not [l for l in sc[0] + sc[1] if l.startswith("T diag")],
          (p.returncode, p.stderr, sc))

# 22. Remote CSR access without routing: the control unit of a chip sends a csr packet on one of its links (`P csr <cycle> <link> r|w <addr>
#     [<value>]`); the control unit at the other end reads or writes the register in the arrival cycle and answers a read with a csr packet on the
#     link the request came in. No host instruction is routed: the host of a chip acts on that chip's own registers (`H csr`).
p = sim("C 0\nP csr 5 0 r 0\nP csr 6 0 r 0\n", "--chips", "2")
sc = sections(p)
check(f"a one-hop remote read of CHIP_ID: asked at 5 on link 0, it reaches chip 1 at 5 + L and the reply reaches chip 0 at 5 + 2L = {5 + 2 * L} (T csr .. 1 000 1)",
      p.returncode == 0 and f"T csr {5 + 2 * L} 1 000 1" in sc[0] and f"T pkt {5 + L} csr 0" in sc[1], (p.returncode, p.stderr, sc))
c3 = json.loads(json.dumps(cfg)); c3["host"]["remote_csr"] = False
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(c3, f); pc = f.name
p = sim("P csr 5 0 r 0\n", "--chips", "2", cfg_path=pc)
os.unlink(pc)
check("with host.remote_csr false a P csr line is exit 2 with a message", p.returncode == 2 and "remote CSR access is off" in p.stderr, (p.returncode, p.stderr))
p = sim("C 0\nH csr 5 r 1:0\n", "--chips", "2")
check("the routed form `H csr <cycle> r <chip>:<addr>` no longer exists (exit 2)", p.returncode == 2 and "input line 2" in p.stderr, (p.returncode, p.stderr))
# RED-END

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
