"""Directed end-to-end tests of the C2C link FEC and the link error reporting (microarchitecture.md sec. 17), on the Sail
emulator. A vector on a link is a header flit, vlanes / 8 body flits of 8 bytes and a tail flit (the published packet format
of HC34 p.45). An `F link <cycle> <link> <flit> corr|uncorr` line marks one flit of the vector the chip sends on that link in
that cycle: `corr` is a simple error the FEC corrects in flight (the vector arrives intact, `T ecc <arrival> link <l> corr flit
<f>`, the receiving chip's CSR LINK_CORR[l] counts it), `uncorr` a burst the FEC only detects (`T ecc ... unc flit <f>`,
LINK_UNCORR[l] counts it): the vector is delivered with the superlanes of the bad flit poisoned, the Receive that takes it
reports OTS-E-LNK-004 and still writes the vector, and the first consumer reports OTS-E-ECC-001. Every latency and gap is
read from the profile. Profile-portable: ref, mini and the test profiles.
usage: link_fec.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
L = CJ["c2c"]["link_latency"]; LINKS = CJ["c2c"]["links"]
TX_GAP = CJ["layout"]["c2c_tx_gap"]; RX_GAP = CJ["layout"]["c2c_rx_gap"]
D_C2C = TM["c2c"]["d_func"]; C2C_DS = TM["c2c"]["d_skew"]
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
FLITS = 1 + (VL + 7) // 8                       # header, body flits, tail
ok = True
_PROGS = []


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def c2c(op, chip=0, s=0, link=0): return (5 << 28) | (op << 22) | (chip << 16) | (s << 10) | (link << 6)
SEND, RECEIVE, TRANSMIT = 0, 1, 3
def E(i): return i
def W(i): return 32 + i
VXM = lambda op, ty, d, a, b=0, m=0, mode=0, tx=0: (2 << 60) | (op << 54) | ((ty & 3) << 52) | (d << 46) | (a << 40) | (b << 34) | (m << 28) | (mode << 24) | (tx << 22)
def add_i8(d, a, b): return VXM(0, 0, d, a, b)
WTX, ERX, VXMQ = 126, 142, 88


def gap_mem(slice_, row):
    q = (slice_ % 44) // 4; west = slice_ < 44; up = (row == "E") == west
    return 18 - q if up else 24 + q
def feed(recs, sram, slice_, addr, val, stream, row, reader_gap, read_cycle):
    g = gap_mem(slice_, row); sram[(slice_, addr)] = val
    recs.append((read_cycle - DM - (reader_gap - g), slice_, 1, mem(0, addr, (32 if row == "W" else 0) + stream), 0))
def drain(recs, slice_, addr, stream, row, writer_gap, commit_cycle):
    g = gap_mem(slice_, row)
    recs.append((commit_cycle + (g - writer_gap) - MEM_DS, slice_, 1, mem(1, addr, (32 if row == "W" else 0) + stream), 0))


def text(recs, sram, dumps, extra=()):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    return ls + q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(extra)


def run(lines, *args, want=0):
    if not args and want == 0:
        _PROGS.append(list(lines))     # recorded by tools/gen_model_vectors.py (link_fec_programs)
    p = subprocess.run([SIM, CFG, "--trace", *args], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == want, (p.returncode, p.stderr)
    return p


def lines_of(out, kind): return [l for l in out.splitlines() if l.startswith(f"T {kind} ")]
def words(out): return {(int(t[1], 16), int(t[2], 16)): bytes(int(x, 16) for x in t[3:])[:VL] for t in (l.split() for l in out.splitlines()) if t and t[0] == "W"}
def xs(out): return {int(t[1], 16): int(t[2], 16) for t in (l.split() for l in out.splitlines()) if t and t[0] == "X"}
def flit_mask(f):                         # the superlanes a flit covers: a body flit its 8 lanes, header and tail the whole vector
    if f == 0 or f == FLITS - 1: return set(range(SL))
    return {byte // 16 for byte in range(5 + 8 * (f - 1), 13 + 8 * (f - 1))}
def fl(cycle, link, flit, kind): return f"F link {cycle:x} {link:x} {flit:x} {kind}"


VAL = bytes((7 * l + 3) & 0xFF for l in range(VL))
ZERO = bytes(VL)
T = 100                                    # Send
TR = T + L                                 # Receive (the first cycle the vector can be taken)
T2 = TR + 80                               # the consumer
VXMQ_EXEC = T2 + VXM_DS


def one_chip(consume=True, link=0):
    """west Send on `link` (loop back), east Receive, a Write to slice 44 addr 50; optionally an add.i8 of that word with zero."""
    recs, sram = [], {}
    feed(recs, sram, 44, 1, VAL, 0, "W", TX_GAP, T)
    recs.append((T, WTX, 1, c2c(SEND, 0, W(0), link), 0))
    recs.append((TR, ERX, 1, c2c(RECEIVE, 0, W(1), link), 0))
    drain(recs, 44, 50, 1, "W", RX_GAP, TR + D_C2C)
    dumps = [(44, 50)]
    if consume:
        feed(recs, sram, 44, 50, ZERO, 8, "W", 21, VXMQ_EXEC)
        feed(recs, sram, 45, 3, ZERO, 9, "W", 21, VXMQ_EXEC)
        recs.append((T2, VXMQ, 1, add_i8(W(16), W(8), W(9)), 0))
        drain(recs, 0, 20, 16, "W", 21, T2 + DV)
        dumps.append((0, 20))
    return text(recs, sram, dumps)


R = ["R 200", "R 210", "R 201", "R 211"]
RECV_EXEC = TR + C2C_DS

# 1. No fault: the vector arrives intact, no FEC line, no diagnostic, both counters 0.
p = run(one_chip() + R)
check("no fault: intact vector, no T ecc, no diagnostic, counters 0",
      words(p.stdout)[(44, 50)] == VAL and lines_of(p.stdout, "ecc") == [] and lines_of(p.stdout, "diag") == []
      and xs(p.stdout) == {0x200: 0, 0x210: 0, 0x201: 0, 0x211: 0}, (p.stdout[-300:]))

# 2. A simple error (one corrected flit): the vector arrives intact, one `T ecc link corr` at the arrival cycle, LINK_CORR[0] = 1.
for f in sorted({1, 2, FLITS // 2, FLITS - 2, 0, FLITS - 1}):
    p = run(one_chip() + [fl(T, 0, f, "corr")] + R)
    check(f"corr on flit {f}: intact vector, `T ecc {TR} link 0 corr flit {f}`, no diagnostic, LINK_CORR[0] = 1",
          words(p.stdout)[(44, 50)] == VAL and lines_of(p.stdout, "ecc") == [f"T ecc {TR} link 0 corr flit {f}"] and lines_of(p.stdout, "diag") == []
          and xs(p.stdout)[0x200] == 1 and xs(p.stdout)[0x210] == 0, (lines_of(p.stdout, "ecc"), lines_of(p.stdout, "diag"), xs(p.stdout)))

# 3. An uncorrectable burst on a body flit: the Receive reports OTS-E-LNK-004 and still writes the vector (the data bits are
#    the sent ones), the superlanes of the flit are poisoned, the add.i8 that reads the stored word reports OTS-E-ECC-001 once and
#    one `T ecc unc` per poisoned superlane; LINK_UNCORR[0] = 1.
for f in sorted({1, 2, min(3, FLITS - 1), FLITS // 2}):
    p = run(one_chip() + [fl(T, 0, f, "uncorr")] + R)
    bad = flit_mask(f)
    eccs = lines_of(p.stdout, "ecc")
    check(f"uncorr on flit {f}: `T ecc {TR} link 0 unc flit {f}`, LNK-004 at the Receive, the vector still written, ECC-001 at the consumer, LINK_UNCORR[0] = 1",
          eccs[0] == f"T ecc {TR} link 0 unc flit {f}"
          and lines_of(p.stdout, "diag") == [f"T diag {RECV_EXEC} queue {ERX} OTS-E-LNK-004", f"T diag {VXMQ_EXEC} queue {VXMQ} OTS-E-ECC-001"]
          and words(p.stdout)[(44, 50)] == VAL and xs(p.stdout)[0x210] == 1 and xs(p.stdout)[0x200] == 0
          and {int(l.split()[8]) for l in eccs[1:]} == bad and len(eccs) == 1 + len(bad), (eccs, lines_of(p.stdout, "diag"), xs(p.stdout)))

# 4. Header and tail flit: the whole vector is bad (every superlane poisoned).
for f in sorted({0, FLITS - 1}):
    p = run(one_chip() + [fl(T, 0, f, "uncorr")] + R)
    eccs = lines_of(p.stdout, "ecc")
    check(f"uncorr on the {'header' if f == 0 else 'tail'} flit: every superlane poisoned",
          lines_of(p.stdout, "diag")[0] == f"T diag {RECV_EXEC} queue {ERX} OTS-E-LNK-004" and {int(l.split()[8]) for l in eccs[1:]} == set(range(SL)), (eccs, lines_of(p.stdout, "diag")))

# 5. A burst over several flits (the detected error of two flits): one LNK-004, both flits traced, superlanes of both poisoned; a
#    corrected flit beside it is counted apart.
if FLITS >= 6:
    fa, fb, fc = 1, 4, 2
    p = run(one_chip() + [fl(T, 0, fa, "uncorr"), fl(T, 0, fb, "uncorr"), fl(T, 0, fc, "corr")] + R)
    eccs = lines_of(p.stdout, "ecc")
    check("two uncorr flits and one corr flit in one vector: three `T ecc` lines in flit order, one LNK-004, counters 1 and 2",
          eccs[:3] == [f"T ecc {TR} link 0 unc flit {fa}", f"T ecc {TR} link 0 corr flit {fc}", f"T ecc {TR} link 0 unc flit {fb}"]
          and [l for l in lines_of(p.stdout, "diag") if "LNK" in l] == [f"T diag {RECV_EXEC} queue {ERX} OTS-E-LNK-004"]
          and {int(l.split()[8]) for l in eccs[3:]} == flit_mask(fa) | flit_mask(fb)
          and xs(p.stdout)[0x200] == 1 and xs(p.stdout)[0x210] == 2, (eccs, xs(p.stdout)))

# 6. The counters belong to the receiving link: a vector on link 1 counts in LINK_CORR[1] (one chip: every link loops back).
if LINKS > 1:
    p = run(one_chip(link=1) + [fl(T, 1, 1, "corr")] + R)
    check("a corrected flit on link 1 counts in LINK_CORR[1] only",
          xs(p.stdout)[0x201] == 1 and xs(p.stdout)[0x200] == 0 and lines_of(p.stdout, "ecc") == [f"T ecc {TR} link 1 corr flit 1"], (xs(p.stdout), lines_of(p.stdout, "ecc")))

# 7. Replay marker: `t` and `p` are accepted and change nothing in a run without `--on-irq replay` (ras_replay_marks.py has the replay).
p = run(one_chip() + [fl(T, 0, 1, "corr") + " t"] + R)
q = run(one_chip() + [fl(T, 0, 1, "corr") + " p"] + R)
check("the t and p markers are accepted and change nothing", p.stdout == q.stdout and xs(p.stdout)[0x200] == 1)

# 8. Two chips: chip 0 sends on link 0 to chip 1, chip 1 receives. One corrected flit, one burst.
def two_chips(extra0=(), extra1=()):
    r0, s0 = [], {}
    feed(r0, s0, 44, 1, VAL, 0, "W", TX_GAP, T)
    r0.append((T, WTX, 1, c2c(SEND, 1, W(0), 0), 0))
    r1, s1 = [], {}
    r1.append((TR, ERX, 1, c2c(RECEIVE, 0, W(1), 0), 0))
    drain(r1, 44, 50, 1, "W", RX_GAP, TR + D_C2C)
    return ["C 0"] + text(r0, s0, [], extra0) + ["C 1"] + text(r1, s1, [(44, 50)], list(extra1) + R)


p = run(two_chips([fl(T, 0, 1, "corr")]), "--chips", "2")
c1 = p.stdout.split("C 1\n")[1]
check("2 chips, corr: chip 1 gets the vector intact, `T ecc` in its section, its LINK_CORR[0] = 1, chip 0 unchanged",
      words(c1)[(44, 50)] == VAL and lines_of(c1, "ecc") == [f"T ecc {TR} link 0 corr flit 1"] and xs(c1)[0x200] == 1 and lines_of(p.stdout.split("C 1\n")[0], "ecc") == [], (c1[-300:]))
p = run(two_chips([fl(T, 0, 1, "uncorr")]), "--chips", "2")
c1 = p.stdout.split("C 1\n")[1]
check("2 chips, uncorr: chip 1's Receive reports OTS-E-LNK-004, the vector is still written, LINK_UNCORR[0] = 1",
      f"T diag {RECV_EXEC} queue {ERX} OTS-E-LNK-004" in c1 and words(c1)[(44, 50)] == VAL and xs(c1)[0x210] == 1, (c1[-300:]))

# 10. The link FEC events in the error CSRs and the error interrupt (microarchitecture.md sec. 16 and 17): a corrected flit is a kind 7
#     entry (ERR_STATUS bit 6, SBE_COUNT), an uncorrectable one a kind 8 entry (bit 7, MBE_COUNT); log word 0 carries the arrival
#     cycle and no queue (0xFFF), log word 1 is link << 16 | flit; the reset ERR_MASK 0x0AA raises MSI-X vector 0 for an
#     uncorrectable flit (in the arrival cycle, after the `T ecc` line) and not for a corrected one.
DEPTH = CJ.get("ras", {}).get("log_depth", 8)
def w0(cycle, kind): return cycle | (kind << 48) | (0xFFF << 52)
def w1(link, flit): return (link << 16) | flit
def hw(cycle, addr, val): return f"H csr {cycle:x} w {addr:x} {val:x}"
RR = [f"R {a:x}" for a in (0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x021, 0x022, 0x0F0, 0x0F1, 0x030, 0x032)]
OVF = 512
def reg(p):
    x = xs(p.stdout)
    return tuple(x[a] for a in (0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x021, 0x022, 0x0F0, 0x0F1, 0x030, 0x032))

p = run(one_chip(consume=False) + RR)
check("events: no fault, every error register reads 0 and no interrupt", reg(p) == (0,) * 12 and lines_of(p.stdout, "irq") == [], (reg(p), lines_of(p.stdout, "irq")))

for f in sorted({1, min(3, FLITS - 1), FLITS - 1}):
    p = run(one_chip(consume=False) + [fl(T, 0, f, "corr")] + RR)
    check(f"events: a corrected flit {f}: ERR_STATUS bit 6, severity 1, SBE_COUNT 1, one kind 7 entry (cycle {TR}, no queue, link 0 << 16 | flit), FIRST_ERR the same, no interrupt",
          reg(p) == (0x40, 1, 1, 0, 1, w0(TR, 7), w1(0, f), 0, w0(TR, 7), w1(0, f), 0, 0) and lines_of(p.stdout, "irq") == [], (reg(p), lines_of(p.stdout, "irq")))

for f in sorted({1, 2, FLITS // 2, 0}):
    p = run(one_chip(consume=False) + [fl(T, 0, f, "uncorr")] + RR)
    check(f"events: an uncorrectable flit {f}: ERR_STATUS bit 7, severity 2, MBE_COUNT 1, one kind 8 entry, `T irq {TR} msix 0` after the `T ecc` line, IRQ_CAUSE 1, IRQ_COUNT 1",
          reg(p) == (0x80, 2, 0, 1, 1, w0(TR, 8), w1(0, f), 0, w0(TR, 8), w1(0, f), 1, 1)
          and lines_of(p.stdout, "irq") == [f"T irq {TR} msix 0"]
          and [l for l in p.stdout.splitlines() if l.startswith(("T ecc", "T irq"))][:2] == [f"T ecc {TR} link 0 unc flit {f}", f"T irq {TR} msix 0"],
          (reg(p), lines_of(p.stdout, "irq")))

if LINKS > 1:
    f = min(3, FLITS - 1)
    p = run(one_chip(consume=False, link=1) + [fl(T, 1, f, "uncorr")] + RR)
    check("events: the link and flit numbers are in log word 1", reg(p)[5:7] == (w0(TR, 8), w1(1, f)) and lines_of(p.stdout, "ecc") == [f"T ecc {TR} link 1 unc flit {f}"], (reg(p), lines_of(p.stdout, "ecc")))

# several flits of one vector: one event per flit in flit order, one interrupt (the 0 to 1 change of bit 7)
if FLITS >= 6:
    p = run(one_chip(consume=False) + [fl(T, 0, 1, "uncorr"), fl(T, 0, 4, "uncorr"), fl(T, 0, 2, "corr")] + RR + ["R 023", "R 024", "R 025"])
    evs = [(8, 1), (7, 2), (8, 4)][:DEPTH]
    x = xs(p.stdout)
    check("events: two uncorrectable and one corrected flit: ERR_STATUS bits 6 and 7, SBE 1, MBE 2, entries in flit order, FIRST_ERR the first, one interrupt",
          (x[0x010], x[0x011], x[0x012], x[0x013], x[0x014]) == (0xC0 | (OVF if DEPTH < 3 else 0), 2, 1, 2, min(3, DEPTH))
          and [(x[0x020 + 2 * i], x[0x021 + 2 * i]) for i in range(len(evs))] == [(w0(TR, k), w1(0, f)) for k, f in evs]
          and (x[0x0F0], x[0x0F1]) == (w0(TR, 8), w1(0, 1)) and lines_of(p.stdout, "irq") == [f"T irq {TR} msix 0"] and x[0x032] == 1, ({k: hex(v) for k, v in x.items()}, lines_of(p.stdout, "irq")))
    LATE = TR + 20
    p = run(one_chip(consume=False) + [fl(T, 0, 1, "uncorr"), fl(T, 0, 4, "corr")] + [hw(LATE, 0x015, 0)] + RR)
    x = reg(p)
    check("events: LOG_POP removes the oldest link entry, FIRST_ERR keeps the first one",
          DEPTH < 2 or (x[4] == 1 and (x[5], x[6]) == (w0(TR, 7), w1(0, 4)) and (x[8], x[9]) == (w0(TR, 8), w1(0, 1))), x)

# ERR_MASK decides the interrupt like for every other error bit
p = run(one_chip(consume=False) + [fl(T, 0, 1, "corr"), hw(1, 0x016, 0x40)] + RR)
check("events: with ERR_MASK bit 6 set a corrected link flit raises the interrupt too", lines_of(p.stdout, "irq") == [f"T irq {TR} msix 0"] and reg(p)[10:] == (1, 1), (lines_of(p.stdout, "irq"), reg(p)))
p = run(one_chip(consume=False) + [fl(T, 0, 1, "uncorr"), hw(1, 0x016, 0)] + RR)
check("events: with ERR_MASK cleared an uncorrectable link flit raises no interrupt, ERR_STATUS and the counters still show it",
      lines_of(p.stdout, "irq") == [] and reg(p)[:4] == (0x80, 2, 0, 1) and reg(p)[10:] == (0, 0), (lines_of(p.stdout, "irq"), reg(p)))

# the link counters 0x200 / 0x210 are unchanged by the reporting
p = run(one_chip(consume=False) + [fl(T, 0, 1, "uncorr"), fl(T, 0, 2, "corr")] + R)
check("events: LINK_CORR and LINK_UNCORR still count the flits", xs(p.stdout)[0x200] == 1 and xs(p.stdout)[0x210] == 1, xs(p.stdout))

# two chips: the receiving chip (1) records the event, the sending chip (0) none
p = run(two_chips([fl(T, 0, 1, "uncorr")], ["R 10", "R 13", "R 20", "R 21"]), "--chips", "2")
c0, c1 = p.stdout.split("C 1\n")[0], p.stdout.split("C 1\n")[1]
check("events, 2 chips: chip 1 reads ERR_STATUS 0x80, MBE_COUNT 1 and the entry (arrival cycle, link 0, flit 1) and raised the interrupt; chip 0 recorded nothing",
      (xs(c1)[0x10], xs(c1)[0x13], xs(c1)[0x20], xs(c1)[0x21]) == (0x80, 1, w0(TR, 8), w1(0, 1)) and lines_of(c1, "irq") == [f"T irq {TR} msix 0"] and lines_of(c0, "irq") == [], (xs(c1), lines_of(c1, "irq")))

# 9. Bad inputs: each is an input error (exit 2) with its message.
def bad(name, extra, msg, args=(), prog=None):
    p = run((prog if prog is not None else one_chip()) + extra, *args, want=2)
    check(f"bad input, {name}: exit 2 and `{msg}`", msg in p.stderr, p.stderr)
bad("flit out of range", [fl(T, 0, FLITS, "corr")], "flit out of range")
bad("link that does not exist", [fl(T, LINKS, 1, "corr")], "link out of range")
bad("unknown outcome word", [f"F link {T:x} 0 1 maybe"], "expected corr or uncorr")
bad("a vector that is not sent: wrong cycle", [fl(T + 1, 0, 1, "corr")], "no vector is sent on link 0 in cycle")
bad("a vector that is not sent: other link", [fl(T, 2, 1, "corr")], "no vector is sent on link 2 in cycle") if LINKS > 2 else None
bad("the same flit twice in one cycle", [fl(T, 0, 1, "corr"), fl(T, 0, 1, "uncorr")], "two lines mark the same flit")
bad("missing field", [f"F link {T:x} 0 1"], "expected corr or uncorr")
bad("a link of a chip that does not exist (unconnected link)", [fl(T, 0, 1, "corr")], "that link is not connected", args=("--chips", "2"),
    prog=["C 1", fl(T, 3, 1, "corr")]) if LINKS > 3 else None

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
