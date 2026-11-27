"""The scenarios of the `System` tests and the oracle they are held to: the model simulator `tsp_sim --chips N`.

A scenario is one program per chip, written as cycle records the way `opentsp-arch-spec/test/c2c/c2c_directed.py` and
`test/host/host_directed.py` author them (the helpers `feed`, `drain`, `c2c` are theirs), the SRAM words the chips start with, the
words to dump and the host's DMA requests. It becomes (1) the input of `tsp_sim --chips N --trace` and (2) the objects and
host calls of a `System`; `Oracle` parses the simulator's output per chip, `System` results are read in the same terms
(`seen_by_system`) so that a test compares two dicts.
"""

import json
import os
import pathlib
import subprocess
import sys
import types

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-arch-spec" / "tools"))

from opentsp_tools import refasm
tsp_as = refasm.load()

CONFIGS = REPO / "opentsp-arch-spec" / "config"

# C2C 32-bit layout: slice 5, opcode 27..22, chip_id 21..16, stream 15..10, link 9..6 (the host instructions put their operand in 5..0).
SEND, RECEIVE, TRANSMIT = 0, 1, 3
PCI_RECEIVE, PCI_SEND, HOST_WAIT, HOST_NOTIFY = 5, 6, 7, 8
WTX, WRX, ETX, ERX, VXMQ = 126, 128, 140, 142, 88


def c2c(op, chip=0, s=0, link=0):
    return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)


def hio(op, s=0, low=0):
    return (5 << 28) | (op << 22) | (s << 10) | low


def mem(op, addr=0, s=0):
    return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)


def E(i):
    return i


def W(i):
    return 32 + i


class Profile:
    def __init__(self, name):
        self.name = name
        self.path = str(CONFIGS / f"{name}.json")
        cfg = json.load(open(self.path))
        self.cfg = cfg
        self.vl = 16 * cfg["geometry"]["superlanes"]
        self.L = cfg["c2c"]["link_latency"]
        self.cpv = cfg["host"]["cycles_per_vector"]
        self.depth = cfg["host"]["ingress_depth"]
        self.tx_gap = cfg["layout"]["c2c_tx_gap"]
        self.rx_gap = cfg["layout"]["c2c_rx_gap"]
        self.d_c2c = cfg["timing"]["c2c"]["d_func"]
        self.d_mem = cfg["timing"]["mem"]["d_func"]

    def val(self, k):
        return bytes((7 * lane + 3 + 11 * k) & 0xFF for lane in range(self.vl))


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4
    west = slice_ < 44
    up = (row == "E") == west
    return 18 - q if up else 24 + q


class Chip:
    """The program, the start-up SRAM, the dumps and the host's DMA requests of one chip."""

    def __init__(self, p):
        self.p, self.recs, self.sram, self.dumps, self.dma = p, [], {}, [], []

    def feed(self, slice_, addr, val, stream, row, reader_gap, read_cycle):
        """A MEM Read that puts the word on the stream so that a reader at `reader_gap` finds it at `read_cycle`; `val` is the
        word the SRAM starts with (None: the word was written by the program)."""
        g = gap_mem(slice_, row)
        if val is not None:
            self.sram[(slice_, addr)] = val
        self.recs.append((read_cycle - self.p.d_mem - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))

    def drain(self, slice_, addr, stream, row, writer_gap, commit_cycle):
        g = gap_mem(slice_, row)
        self.recs.append((commit_cycle + (g - writer_gap), slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))

    def at(self, cycle, queue, word):
        self.recs.append((cycle, queue, 1, word, 0))

    def object(self):
        prog = types.SimpleNamespace(profile=self.p.name, images=tsp_as.lower_records(list(self.recs)), externs={})
        return tsp_as.write_object(prog, self.p.path)


def hexs(b):
    return " ".join(f"{x:x}" for x in b)


def oracle_text(chips):
    out = []
    for c, ch in enumerate(chips):
        out.append(f"C {c:x}")
        out += [f"S {s:x} {a:x} {hexs(v)}" for (s, a), v in ch.sram.items()]
        out += tsp_as.q_lines(tsp_as.lower_records(list(ch.recs)))
        out += [f"D {s:x} {a:x}" for s, a in ch.dumps]
        out += [f"H dma {cycle:x} {hexs(v)}" for cycle, v in ch.dma]
    return "\n".join(out) + "\n"


def sim_path():
    # An explicit oracle is caller-owned and may be deliberately pinned. Its
    # source identity is the caller's responsibility; never rebuild that path.
    selected = os.environ.get("TSP_SIM")
    sim = pathlib.Path(selected) if selected else REPO / "target" / "spec" / "tsp_sim"
    if not selected:
        subprocess.run(["bash", str(REPO / "tools" / "ensure_sim.sh"), str(sim)], check=True)
    label = f"TSP_SIM={sim}" if selected else str(sim)
    if not sim.is_file():
        raise FileNotFoundError(f"{label}: oracle is not a regular executable file")
    if not os.access(sim, os.X_OK):
        raise PermissionError(f"{label}: oracle is not executable")
    return str(sim.resolve())


def run_oracle(sim, p, chips, *args):
    r = subprocess.run(
        [sim, p.path, "--trace", "--chips", str(len(chips)), *args],
        input=oracle_text(chips), capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, r.stderr
    return parse_oracle(r.stdout, p.vl, len(chips))


def parse_oracle(text, vl, n):
    """Per chip: the `T srf`, `T sram`, `T dma`, `T irq`, `T diag` lists, the `E` line, the `W` words and the `Y` vectors."""
    new = lambda: {"srf": [], "sram": [], "dma": [], "irq": [], "diag": [], "E": None, "W": {}, "Y": []}  # noqa: E731
    chips = [new()] if n == 1 else []  # one chip prints no `C` line
    for line in text.splitlines():
        t = line.split()
        if t[0] == "C":
            chips.append(new())
            continue
        c = chips[-1]
        if t[:2] == ["T", "srf"]:
            row, stream = t[3].split(":")
            c["srf"].append((int(t[2]), row, int(stream), int(t[5]), bytes.fromhex(t[6]) if len(t) > 6 else b""))
        elif t[:2] == ["T", "sram"]:
            c["sram"].append((int(t[2]), int(t[4]), int(t[6]), bytes.fromhex(t[7]) if len(t) > 7 else b""))
        elif t[:2] == ["T", "dma"]:
            c["dma"].append((int(t[2]), t[3]))
        elif t[:2] == ["T", "irq"]:
            c["irq"].append((int(t[2]), int(t[4])))
        elif t[:2] == ["T", "diag"]:
            c["diag"].append((int(t[2]), int(t[4]), t[5]))
        elif t[0] == "E":
            c["E"] = (None if t[1] == "-" else t[1], int(t[2]), int(t[3]))
        elif t[0] == "W":
            c["W"][(int(t[1], 16), int(t[2], 16))] = bytes(int(x, 16) for x in t[3:])[:vl]
        elif t[0] == "Y":
            c["Y"].append((int(t[1]), bytes(int(x, 16) for x in t[2:])[:vl]))
    assert len(chips) == n, (len(chips), n)
    return chips


def seen_by_system(s, chips):
    """What a `System` that has run to `done` and `finish`ed shows, in the oracle's terms."""
    out = []
    for c, ch in enumerate(chips):
        srf, sram = s.trace(c)
        diags = s.diagnostics(c)
        out.append({
            "srf": [(cy, row, st, gap, bytes(v)) for cy, row, st, gap, v in srf],
            "sram": [(cy, sl, a, bytes(v)) for cy, sl, a, v in sram],
            "dma": list(s.dma_events(c)),
            "irq": list(s.irqs(c)),
            "diag": list(diags),
            "E": s.error_line(c),
            "W": {(sl, a): bytes(s.read_words(c, sl, a, 1)) for sl, a in ch.dumps},
            "Y": [(cy, bytes(v)) for cy, v in s.egress(c)],
        })
    return out


# ---- the scenarios -------------------------------------------------------------------------------------------------------

T0 = 100


def send_receive(p):
    """Chip 0 reads a word of its slice 44 and Sends it to chip 1 on link 0 (the west TX queue); chip 1 Receives it L cycles later
    (the east RX queue) and writes it to slice 44 address 50 (dumped)."""
    c0, c1 = Chip(p), Chip(p)
    val = p.val(1)
    c0.feed(44, 1, val, 0, "W", p.tx_gap, T0)
    c0.at(T0, WTX, c2c(SEND, 1, W(0)))
    c1.at(T0 + p.L, ERX, c2c(RECEIVE, 0, W(1)))
    c1.drain(44, 50, 1, "W", p.rx_gap, T0 + p.L + p.d_c2c)
    c1.dumps = [(44, 50)]
    return [c0, c1], val


def tight_transmit(p):
    """Chip 0's first event after the empty window `[0, L)` is its Transmit at cycle `t` on link 0 (the window of the run starts at it);
    chip 1 Receives the notification vector at exactly `t + L`, the earliest cycle it can be there: the window `[t, t + L)` must end
    before the Receive."""
    t = p.L + 84
    c0, c1 = Chip(p), Chip(p)
    c0.at(t, ETX, c2c(TRANSMIT, 1))
    c1.at(t + p.L, WRX, c2c(RECEIVE, 0, E(3)))
    c1.drain(0, 70, 3, "E", p.rx_gap, t + p.L + p.d_c2c)
    c1.dumps = [(0, 70)]
    return [c0, c1]


def far_send(p, chips, src, link, dst, far=None):
    """In a node of `chips` chips (2..8) the local link l of chip i reaches chip (i + l + 1) mod `chips` on its link `chips - 2 - l`
    (`far` names the far end's link where the wiring is another: the global links of 16 chips): chip `src` Sends on `link` to chip
    `dst`, which Receives on the far end of the link."""
    if far is None:
        assert dst == (src + link + 1) % chips
        far = chips - 2 - link
    cs = [Chip(p) for _ in range(chips)]
    val = p.val(9)
    cs[src].feed(44, 1, val, 0, "W", p.tx_gap, T0)
    cs[src].at(T0, WTX, c2c(SEND, dst, W(0), link))
    cs[dst].at(T0 + p.L, ERX, c2c(RECEIVE, src, W(1), far))
    cs[dst].drain(44, 50, 1, "W", p.rx_gap, T0 + p.L + p.d_c2c)
    cs[dst].dumps = [(44, 50)]
    return cs, val


def echo_one_chip(p, vectors=1):
    """The host's DMA vectors enter chip 0 by PciReceive (the east RX queue), go through slice 44 and leave by PciSend (the west
    TX queue): the host gets them back, `vectors` of them, paced by `host.cycles_per_vector`."""
    c = Chip(p)
    gap = 4 * p.cpv + 8  # the PciSend of vector k reads its word well after the PciReceive wrote it
    for k in range(vectors):
        c.dma.append((0, p.val(k)))
        recv = T0 + k * p.cpv
        c.at(recv, ERX, hio(PCI_RECEIVE, W(24 + k % 8)))
        c.drain(44, 10 + k, 24 + k % 8, "W", p.rx_gap, recv + p.d_c2c)
        send = recv + gap + k * 0
        c.feed(44, 10 + k, None, k % 8, "W", p.tx_gap, send)
        c.at(send, WTX, hio(PCI_SEND, W(k % 8)))
    c.dumps = [(44, 10 + k) for k in range(vectors)]
    return [c]


def echo_two_chips(p):
    """Chip 0 takes the host's vector by PciReceive and Sends it over link 0 to chip 1; chip 1 Receives it and gives it to the host
    by PciSend."""
    c0, c1 = Chip(p), Chip(p)
    c0.dma.append((0, p.val(3)))
    c0.at(T0, ERX, hio(PCI_RECEIVE, W(24)))
    c0.drain(44, 10, 24, "W", p.rx_gap, T0 + p.d_c2c)
    send = T0 + 40
    c0.feed(44, 10, None, 0, "W", p.tx_gap, send)
    c0.at(send, WTX, c2c(SEND, 1, W(0)))
    recv = send + p.L
    c1.at(recv, ERX, c2c(RECEIVE, 0, W(1)))
    c1.drain(44, 20, 1, "W", p.rx_gap, recv + p.d_c2c)
    out = recv + 40
    c1.feed(44, 20, None, 2, "W", p.tx_gap, out)
    c1.at(out, WTX, hio(PCI_SEND, W(2)))
    c1.dumps = [(44, 10)]
    c0.dumps = [(44, 10)]
    return [c0, c1]


def echo_three_chips(p, vectors=1):
    """The pipeline: the host in at chip 0, over link 0 to chip 1 (arriving on its link 1), over its link 0 to chip 2 (arriving on
    its link 1), and out to the host at chip 2. In three chips (one node) the link l of chip i reaches chip (i + l + 1) mod 3 on
    its link 1 - l."""
    c0, c1, c2 = Chip(p), Chip(p), Chip(p)
    for k in range(vectors):
        c0.dma.append((0, p.val(5 + k)))
        r0 = T0 + k * 30
        c0.at(r0, ERX, hio(PCI_RECEIVE, W(24 + k % 4)))
        c0.drain(44, 10 + k, 24 + k % 4, "W", p.rx_gap, r0 + p.d_c2c)
        s0 = r0 + 40
        c0.feed(44, 10 + k, None, k % 4, "W", p.tx_gap, s0)
        c0.at(s0, WTX, c2c(SEND, 1, W(k % 4), 0))
        r1 = s0 + p.L
        c1.at(r1, ERX, c2c(RECEIVE, 0, W(8 + k % 4), 1))
        c1.drain(44, 20 + k, 8 + k % 4, "W", p.rx_gap, r1 + p.d_c2c)
        s1 = r1 + 40
        c1.feed(44, 20 + k, None, 4 + k % 4, "W", p.tx_gap, s1)
        c1.at(s1, WTX, c2c(SEND, 2, W(4 + k % 4), 0))
        r2 = s1 + p.L
        c2.at(r2, ERX, c2c(RECEIVE, 1, W(12 + k % 4), 1))
        c2.drain(44, 30 + k, 12 + k % 4, "W", p.rx_gap, r2 + p.d_c2c)
        out = r2 + 40
        c2.feed(44, 30 + k, None, 8 + k % 4, "W", p.tx_gap, out)
        c2.at(out, WTX, hio(PCI_SEND, W(8 + k % 4)))
    c0.dumps = [(44, 10 + k) for k in range(vectors)]
    c1.dumps = [(44, 20 + k) for k in range(vectors)]
    c2.dumps = [(44, 30 + k) for k in range(vectors)]
    return [c0, c1, c2]


def new_system(_System, p, chips, **kw):
    s = _System(p.name, len(chips), trace=True, **kw)
    for c, ch in enumerate(chips):
        s.load_object(c, ch.object())
    for c, ch in enumerate(chips):
        for (sl, a), v in ch.sram.items():
            s.write_words(c, sl, a, v)
        for cycle, v in ch.dma:
            s.dma_write(c, v, cycle)
    return s
