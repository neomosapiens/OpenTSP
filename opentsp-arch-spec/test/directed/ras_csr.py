"""Directed end-to-end tests of the error CSRs (error register, counters, log, first-error copy, the ICU CSR
space), of the uncorrectable-error interrupt and replay and of ECC on instruction text, on the Sail
emulator (microarchitecture.md sec. 15 and 16). The faults are injected with `F` lines; the registers are read with
`R` lines after the run (`X <csr> <value>`) and written by the host's `H csr` lines. Every latency and gap is read
from the profile. Profile-portable: ref, mini and the test profiles.
usage: ras_csr.py <tsp_sim> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
DEPTH = CJ.get("ras", {}).get("log_depth", 8)
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def mem(op, addr=0, s=0): return (1 << 60) | (op << 54) | (addr << 32) | (s << 26)
def E(i): return i
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


def run(recs, sram, dumps=(), extra=(), opts=(), expect=0):
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(extra)
    if os.environ.get("RAS_CSR_SAVE") and any(l.startswith("F iq") for l in extra):   # debugging: keep the input of the last text-fault run
        run.n = getattr(run, "n", 0) + 1
        open(f'{os.environ["RAS_CSR_SAVE"]}.{run.n}', "w").write("\n".join(ls) + "\n")
    p = subprocess.run([SIM, CFG, "--trace", *opts], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == expect, (p.returncode, p.stderr, p.stdout[-400:])
    return p


def lines(p, kind): return [l for l in p.stdout.splitlines() if l.startswith(f"T {kind} ")]
def diags(p): return lines(p, "diag")
def xs(p): return {int(t[1], 16): int(t[2], 16) for t in (l.split() for l in p.stdout.splitlines()) if t and t[0] == "X"}


POS = [p for p in range(3, 137) if p & (p - 1)]
def pos_of(bit): return POS[bit] if bit < 128 else (1 << (bit - 128) if bit < 136 else 0)
def pos_of2(a, b): return pos_of(a) ^ pos_of(b)


A = bytes((13 * l + 5) & 0xFF for l in range(VL))
B = bytes((7 * l + 91) & 0xFF for l in range(VL))
T = 60
VXMQ = 88
FETCH_LAT = CJ["icu"]["fetch_latency"]


def vxm_program(t=T):
    """A on E:0 (slice 0 addr 1) and B on E:4 (slice 1 addr 2) read by the VXM at gap 21; the sum on E:8 to slice 44 addr 9."""
    recs, sram = [], {}
    feed(recs, sram, 0, 1, A, 0, "E", 21, t + VXM_DS)
    feed(recs, sram, 1, 2, B, 4, "E", 21, t + VXM_DS)
    recs.append((t, VXMQ, 1, add_i8(E(8), E(0), E(4)), 0))
    drain(recs, 44, 9, 8, "E", 21, t + DV)
    return recs, sram


def fsram(cycle, slice_, addr, sl, bit, tp=""): return f"F sram {cycle:x} {slice_:x} {addr:x} {sl:x} {bit:x}" + (" " + tp if tp else "")
def rlines(*addrs): return [f"R {a:x}" for a in addrs]
def hw(cycle, addr, val): return f"H csr {cycle:x} w {addr:x} {val:x}"


OVF = 512 if DEPTH < 2 else 0      # two events fill a log of one entry: ERR_STATUS bit 9
EV = T + VXM_DS                 # the cycle of the VXM's operand check


def w0(cycle, kind, q): return cycle | (kind << 48) | (q << 52)
def w1_reg(row_w, stream, gap, sl, pos): return (1 if row_w else 0) | (stream << 1) | (gap << 6) | (sl << 43) | (pos << 48)
def w1_sram(slice_, addr, sl, pos): return slice_ | (addr << 7) | (sl << 43) | (pos << 48)


# 1. A run with no fault leaves the error registers clear.
recs, sram = vxm_program()
p = run(recs, sram, [], rlines(0x010, 0x011, 0x012, 0x013, 0x014, 0x0F0, 0x0F1, 0x030, 0x032))
check("error CSRs: no fault, every error register reads 0 and no interrupt was raised",
      all(v == 0 for v in xs(p).values()) and len(xs(p)) == 9 and lines(p, "irq") == [], (xs(p), lines(p, "irq")))

# 2. A corrected error: ERR_STATUS bit 0, severity 1, SBE_COUNT 1, one log entry and FIRST_ERR the same entry; no interrupt.
recs, sram = vxm_program()
p = run(recs, sram, [], [fsram(0, 0, 1, 0, 5)] + rlines(0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x021, 0x022, 0x0F0, 0x0F1, 0x030, 0x032))
x = xs(p)
want0, want1 = w0(EV, 1, VXMQ), w1_reg(False, 0, 21, 0, pos_of(5))
check("error CSRs: a corrected error sets ERR_STATUS bit 0, severity 1, SBE_COUNT 1, MBE_COUNT 0, LOG_COUNT 1",
      (x.get(0x010), x.get(0x011), x.get(0x012), x.get(0x013), x.get(0x014)) == (1, 1, 1, 0, 1), x)
check("error CSRs: log entry 0 holds the cycle, kind 1 (SBE data), the queue, the register location, superlane and position; entry 1 is empty",
      (x.get(0x020), x.get(0x021), x.get(0x022)) == (want0, want1, 0), (hex(x.get(0x020, -1)), hex(x.get(0x021, -1)), hex(want0), hex(want1)))
check("error CSRs: FIRST_ERR0/1 hold a copy of the first log entry", (x.get(0x0F0), x.get(0x0F1)) == (want0, want1), x)
check("error CSRs: a corrected error raises no interrupt (cause and count read 0)", (x.get(0x030), x.get(0x032)) == (0, 0) and lines(p, "irq") == [], (x, lines(p, "irq")))

# 3. An SRAM word checked by a MEM unit (Gather: the tile's superlane of the stored word) is logged with its SRAM location.
if CJ["geometry"]["mem_slices_per_hemisphere"] >= 8:
    m = bytearray(VL)
    for g in range(SL):
        m[16 * g] = 40 + g; m[16 * g + 1] = 0
    for kind, fl in [("corrected", [0]), ("uncorrectable", [0, 1])]:
        sram2 = {(4, 1): bytes(m)}
        for g in range(SL): sram2[(0, 40 + g)] = bytes([g] * VL)
        TG = 30
        recs = [(TG + MEM_DS - DM - 1, 4, 1, mem(0, 1, E(1)), 0), (TG, 0, 1, (1 << 60) | (3 << 54) | (E(2) << 42) | (E(1) << 36), 0)]
        recs.append((TG + DM + 6, 44, 1, mem(1, 7, E(2)), 0))
        p = run(recs, sram2, [], [fsram(0, 0, 40, 0, b) for b in fl] + rlines(0x010, 0x020, 0x021))
        x = xs(p)
        unc = len(fl) == 2
        pos = pos_of2(*fl) if unc else pos_of(fl[0])
        check(f"error CSRs: a {kind} SRAM word read by the Gather is logged with kind {2 if unc else 1}, queue 0 and its slice and address",
              x.get(0x010) == (2 if unc else 1) and x.get(0x020) == w0(TG + MEM_DS, 2 if unc else 1, 0) and x.get(0x021) == w1_sram(0, 40, 0, pos),
              {k: hex(v) for k, v in x.items()})

# 4. Both operands corrected: two entries in order, SBE_COUNT 2; an uncorrectable error: bit 1, severity 2, MBE_COUNT,
#    the interrupt of MSI-X vector 0 (cause bit 0, count 1) raised in the cycle of the error.
recs, sram = vxm_program()
p = run(recs, sram, [], [fsram(0, 0, 1, 0, 5), fsram(0, 1, 2, 0, 9)] + rlines(0x010, 0x012, 0x014, 0x020, 0x022, 0x021, 0x023))
x = xs(p)
check("error CSRs: two corrected operands are two entries in the order of the trace, SBE_COUNT 2",
      (x.get(0x012), x.get(0x014), x.get(0x010)) == (2, min(2, DEPTH), 1 | OVF) and x.get(0x020) == w0(EV, 1, VXMQ)
      and x.get(0x021) == w1_reg(False, 0, 21, 0, pos_of(5))
      and (DEPTH < 2 or (x.get(0x022) == w0(EV, 1, VXMQ) and x.get(0x023) == w1_reg(False, 4, 21, 0, pos_of(9)))), {k: hex(v) for k, v in x.items()})
bits2 = [5, 77]
recs, sram = vxm_program()
p = run(recs, sram, [], [fsram(0, 0, 1, 0, b) for b in bits2] + rlines(0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x021, 0x030, 0x032))
x = xs(p)
check("error CSRs: an uncorrectable error sets ERR_STATUS bit 1, severity 2, MBE_COUNT 1, SBE_COUNT 0 and a kind 2 entry with the raw syndrome",
      (x.get(0x010), x.get(0x011), x.get(0x012), x.get(0x013), x.get(0x014)) == (2, 2, 0, 1, 1)
      and x.get(0x020) == w0(EV, 2, VXMQ) and x.get(0x021) == w1_reg(False, 0, 21, 0, pos_of2(*bits2)), {k: hex(v) for k, v in x.items()})
check("error interrupt: the uncorrectable error raises the error interrupt (MSI-X vector 0) in the cycle of the error: `T irq`, cause bit 0, count 1",
      lines(p, "irq") == [f"T irq {EV} msix 0"] and (x.get(0x030), x.get(0x032)) == (1, 1), (lines(p, "irq"), x))

# 5. ERR_MASK (0x016) chooses which error bits raise the interrupt: reset 0x0aa (the uncorrectable bits 1, 3, 5, 7).
recs, sram = vxm_program()
p = run(recs, sram, [], [fsram(0, 0, 1, 0, b) for b in bits2] + [hw(1, 0x016, 0)] + rlines(0x016, 0x032))
check("error interrupt: with ERR_MASK cleared (host write) the uncorrectable error raises no interrupt",
      lines(p, "irq") == [] and xs(p) == {0x016: 0, 0x032: 0}, (lines(p, "irq"), xs(p)))
p = run(recs, sram, [], [fsram(0, 0, 1, 0, 5)] + [hw(1, 0x016, 1)] + rlines(0x016, 0x032))
check("error interrupt: with bit 0 of ERR_MASK set a corrected error raises the interrupt too",
      lines(p, "irq") == [f"T irq {EV} msix 0"] and xs(p) == {0x016: 1, 0x032: 1}, (lines(p, "irq"), xs(p)))
p = run(recs, sram, [], [fsram(0, 0, 1, 0, 5)] + [hw(1, 0x016, 1), hw(1, 0x031, 0xFFFE)] + rlines(0x030, 0x031, 0x032))
check("error interrupt: a masked MSI-X vector 0 keeps the cause bit and gives no interrupt (IRQ_MASK bit 0 clear)",
      lines(p, "irq") == [] and xs(p) == {0x030: 1, 0x031: 0xFFFE, 0x032: 0}, (lines(p, "irq"), xs(p)))

# 6. The host clears: ERR_STATUS is write-one-to-clear, the counters clear on any write, LOG_POP removes the oldest entry,
#    FIRST_ERR is re-armed when ERR_STATUS has been cleared.
LATE = EV + 40
recs, sram = vxm_program()
fl = [fsram(0, 0, 1, 0, 5), fsram(0, 1, 2, 0, 9)]
p = run(recs, sram, [], fl + [hw(LATE, 0x015, 0)] + rlines(0x014, 0x020, 0x021, 0x022))
x = xs(p)
check("error CSRs: LOG_POP removes the oldest entry (the second becomes entry 0)",
      x.get(0x014) == min(2, DEPTH) - 1 and x.get(0x022) == 0
      and (DEPTH < 2 or (x.get(0x020) == w0(EV, 1, VXMQ) and x.get(0x021) == w1_reg(False, 4, 21, 0, pos_of(9)))), {k: hex(v) for k, v in x.items()})
p = run(recs, sram, [], fl + [hw(LATE, 0x010, 1), hw(LATE, 0x012, 7)] + rlines(0x010, 0x011, 0x012, 0x0F0, 0x014))
x = xs(p)
check("error CSRs: writing 1 to ERR_STATUS clears bit 0, severity falls to 0, any write clears SBE_COUNT, FIRST_ERR is re-armed once ERR_STATUS is 0; the log stays",
      (x.get(0x010), x.get(0x011), x.get(0x012), x.get(0x014)) == (OVF, 0, 0, min(2, DEPTH))
      and x.get(0x0F0) == (w0(EV, 1, VXMQ) if OVF else 0), {k: hex(v) for k, v in x.items()})
p = run(recs, sram, [], fl + [hw(LATE, 0x010, 2)] + rlines(0x010, 0x011))
check("error CSRs: writing a bit that is not set clears nothing", xs(p) == {0x010: 1 | OVF, 0x011: 1}, xs(p))

# 7. The log holds log_depth entries; a further event sets ERR_STATUS bit 9 and is dropped, the counters still count.
if DEPTH + 1 <= 2 * SL:
    n_ev = DEPTH + 1
    flips = []
    for i in range(n_ev):
        flips.append(fsram(0, 0 if i % 2 == 0 else 1, 1 if i % 2 == 0 else 2, i // 2, 5 + i))
    recs, sram = vxm_program()
    p = run(recs, sram, [], flips + rlines(0x010, 0x012, 0x014, 0x0F0))
    x = xs(p)
    check(f"error CSRs: {n_ev} corrected events with log_depth {DEPTH}: LOG_COUNT {DEPTH}, overflow bit 9 set, SBE_COUNT {n_ev}, FIRST_ERR is the first event",
          (x.get(0x010), x.get(0x012), x.get(0x014)) == (1 | (1 << 9), n_ev, DEPTH) and x.get(0x0F0) == w0(EV, 1, VXMQ), {k: hex(v) for k, v in x.items()})

# 8. The ICU CSR space: FETCH_RT[q] (0x100 + q, q = 0..143) holds the queue's fetch latency, read only; the rest is unmapped (reads 0).
recs, sram = vxm_program()
p = run(recs, sram, [], [hw(1, 0x100, 99), hw(1, 0x110, 99)] + rlines(0x100, 0x110, 0x18f, 0x190, 0x017, 0x1ff))
check("error CSRs: FETCH_RT[q] reads icu.fetch_latency, ignores writes; q > 143 and unmapped addresses read 0",
      xs(p) == {0x100: FETCH_LAT, 0x110: FETCH_LAT, 0x18f: FETCH_LAT, 0x190: 0, 0x017: 0, 0x1ff: 0}, xs(p))

# 9. Bad inputs: a CSR address beyond 0xfff is an input error (exit 2).
p = subprocess.run([SIM, CFG], input="R 1000\n", capture_output=True, text=True, timeout=60)
check("R 1000 is an input error with a message", p.returncode == 2 and "csr address out of range" in p.stderr, (p.returncode, p.stderr))

# 10. ECC on instruction text (microarchitecture.md section 16): two adds on queue 88 in consecutive cycles. The text of the queue
# is [NOP, add1.hi, add1.lo, add2.hi | add2.lo, ...]: add2 straddles the text words 0 and 1. A flip is `F iq <cycle> <queue>
# <word> <bit>`; a text error is found when the parcel dispatches, before decode.
def fiq(cycle, queue, word, bit, tp=""): return f"F iq {cycle:x} {queue:x} {word:x} {bit:x}" + (" " + tp if tp else "")
def tl(p, kind): return lines(p, kind)
O1 = (44, 9, 8, "E"); O2 = (45, 10, 9, "E")      # the two results: slice, address, stream, row


def vxm_two(t=T):
    """add1 at t (A on E:0, B on E:4 -> E:8 to slice 44 addr 9) and add2 at t + 1 (E:1 + E:5 -> E:9 to slice 45 addr 10)."""
    recs, sram = [], {}
    feed(recs, sram, 0, 1, A, 0, "E", 21, t + VXM_DS)
    feed(recs, sram, 1, 2, B, 4, "E", 21, t + VXM_DS)
    recs.append((t, VXMQ, 1, add_i8(E(8), E(0), E(4)), 0))
    drain(recs, 44, 9, 8, "E", 21, t + DV)
    feed(recs, sram, 2, 3, B, 1, "E", 21, t + 1 + VXM_DS)
    feed(recs, sram, 3, 4, A, 5, "E", 21, t + 1 + VXM_DS)
    recs.append((t + 1, VXMQ, 1, add_i8(E(9), E(1), E(5)), 0))
    drain(recs, 45, 10, 9, "E", 21, t + 1 + DV)
    return recs, sram


def dump_lines(p):
    return [l for l in p.stdout.splitlines() if l.startswith("W ")]


recs, sram = vxm_two()
base = run(recs, sram, [(44, 9), (45, 10)], rlines(0x010, 0x012, 0x013, 0x014))
check("text ECC: with no text fault two adds run, no T ecc, no diagnostic, the error registers read 0",
      tl(base, "ecc") == [] and diags(base) == [] and set(xs(base).values()) == {0} and len(dump_lines(base)) == 2, (tl(base, "ecc"), diags(base), xs(base)))

# 10a. A single flip in the slice-tag bit of add1: corrected before decode, both adds run as without the fault; the word is
# corrected once (the correction is written back), a corrected text error raises no interrupt.
TAGBIT = 32 + 28
p = run(recs, sram, [(44, 9), (45, 10)], [fiq(0, VXMQ, 0, TAGBIT)] + rlines(0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x021, 0x0F0))
x = xs(p)
check("text ECC: a single flip in the slice-tag bit of a queued add is corrected before decode; both adds give the results of the run with no fault",
      dump_lines(p) == dump_lines(base) and diags(p) == [], (dump_lines(p), dump_lines(base), diags(p)))
check(f"text ECC: the correction is one 'T ecc ... corr text word 0 pos {pos_of(TAGBIT)}' line at the dispatch cycle (the word is corrected once)",
      tl(p, "ecc") == [f"T ecc {T} queue {VXMQ} corr text word 0 pos {pos_of(TAGBIT)}"], tl(p, "ecc"))
check("text ECC: ERR_STATUS bit 2, severity 1, SBE_COUNT 1, log entry kind 3 with the queue and the text word, no interrupt",
      (x.get(0x010), x.get(0x011), x.get(0x012), x.get(0x013), x.get(0x014)) == (4, 1, 1, 0, 1)
      and x.get(0x020) == w0(T, 3, VXMQ) and x.get(0x021) == ((0 << 8) | VXMQ) | (pos_of(TAGBIT) << 48) and x.get(0x0F0) == w0(T, 3, VXMQ)
      and tl(p, "irq") == [], {k: hex(v) for k, v in x.items()})

# 10b. A double flip in the text word 0 [NOP, add1.hi, add1.lo, add2.hi]: every instruction of the word is OTS-E-ECC-002 (a text word that is
# uncorrectable is checked by each instruction that has a unit in it, the timing NOP included) and has no effect; the results
# are not written (their readers find no value: OTS-E-TIM-001 downstream). The error interrupt is raised once: the first
# 0 -> 1 change of ERR_STATUS bit 3.
def vxm_queue_diagnostics(p): return [l for l in diags(p) if f" queue {VXMQ} " in l]


dbl = [fiq(0, VXMQ, 0, 32), fiq(0, VXMQ, 0, 33)]
p = run(recs, sram, [(44, 9), (45, 10)], dbl + rlines(0x010, 0x011, 0x012, 0x013, 0x014, 0x020, 0x022, 0x0F0, 0x030, 0x032))
x = xs(p)
check("text ECC: a double flip in text word 0: the NOP, add1 and add2 are OTS-E-ECC-002 (cycles 0, 1, 2) and the adds have no effect",
      vxm_queue_diagnostics(p) == [f"T diag {c} queue {VXMQ} OTS-E-ECC-002" for c in (0, 1, 2)]
      and len(dump_lines(p)) == 2 and all(set(l.split()[3:]) <= {"0", "00"} for l in dump_lines(p)) and dump_lines(p) != dump_lines(base), (diags(p), dump_lines(p)[:1]))
unc = [f"T ecc {c} queue {VXMQ} unc text word 0 pos {pos_of2(32, 33)}" for c in (0, 1, 2)]
check("text ECC: each uncorrectable instruction prints a 'T ecc ... unc text word 0' line, before its diagnostic", tl(p, "ecc") == unc, tl(p, "ecc"))
check("text ECC: ERR_STATUS bit 3 (plus 9 if the log is full), severity 2, MBE_COUNT 3, log entries of kind 4, FIRST_ERR the first",
      (x.get(0x010), x.get(0x011), x.get(0x012), x.get(0x013), x.get(0x014)) == (8 | (512 if DEPTH < 3 else 0), 2, 0, 3, min(3, DEPTH))
      and x.get(0x020) == w0(0, 4, VXMQ) and x.get(0x0F0) == w0(0, 4, VXMQ), {k: hex(v) for k, v in x.items()})
check("text ECC: the uncorrectable text error raises the host interrupt once (T irq, cause bit 0, count 1)",
      tl(p, "irq") == ["T irq 0 msix 0"] and (x.get(0x030), x.get(0x032)) == (1, 1), (tl(p, "irq"), x))

# 10c. The pair rule: a double flip in text word 1, in a unit add2 does not use, still makes add2 ECC-002 (both words of the instruction are
# checked); add1 lies in word 0 only and runs.
p = run(recs, sram, [(44, 9), (45, 10)], [fiq(0, VXMQ, 1, 64), fiq(0, VXMQ, 1, 65)] + rlines(0x010, 0x013))
check("text ECC: pair rule: a double flip in text word 1 (a unit add2 does not use) makes add2 OTS-E-ECC-002; add1, which lies in word 0, runs",
      vxm_queue_diagnostics(p) == [f"T diag {T + 1} queue {VXMQ} OTS-E-ECC-002"] and tl(p, "ecc") == [f"T ecc {T + 1} queue {VXMQ} unc text word 1 pos {pos_of2(64, 65)}"]
      and len(dump_lines(p)) == len(dump_lines(base)) and dump_lines(p)[0] == dump_lines(base)[0] and xs(p) == {0x010: 8, 0x013: 1}, (vxm_queue_diagnostics(p), tl(p, "ecc"), xs(p)))

# 10d. A flip of a check bit and of the spare bit: a check-bit flip is a correctable error (syndrome names a check bit), the spare bit alone is no error.
p = run(recs, sram, [(44, 9), (45, 10)], [fiq(0, VXMQ, 0, 0x80 + 3)] + rlines(0x010, 0x012))
check("text ECC: a flipped check bit is corrected (T ecc corr, SBE_COUNT 1) and the adds run", len(tl(p, "ecc")) == 1 and " corr text word 0 " in tl(p, "ecc")[0]
      and xs(p) == {0x010: 4, 0x012: 1} and dump_lines(p) == dump_lines(base) and diags(p) == [], (tl(p, "ecc"), xs(p)))
p = run(recs, sram, [(44, 9), (45, 10)], [fiq(0, VXMQ, 0, 0x89)] + rlines(0x010, 0x012))
check("text ECC: a flipped spare bit alone is not an error", tl(p, "ecc") == [] and xs(p) == {0x010: 0, 0x012: 0} and dump_lines(p) == dump_lines(base), (tl(p, "ecc"), xs(p)))

# 10e. A flip applied after the instruction left the IQ (word 0 is gone at cycle T + 20): an input error, exit 2.
p = run(recs, sram, [], [fiq(T + 20, VXMQ, 0, 3)], expect=2)
check("text ECC: F iq of a word that is not in the IQ at that cycle is an input error (exit 2) with a message",
      f"F iq at cycle {T + 20}: text word 0 of queue {VXMQ:x} is not in the instruction queue" in p.stderr, p.stderr)
p = subprocess.run([SIM, CFG], input="F iq 0 91 0 0\n", capture_output=True, text=True, timeout=60)
check("text ECC: a bad queue, a bad bit and a missing field are input errors",
      p.returncode == 2 and "queue out of range" in p.stderr, (p.returncode, p.stderr))
p = subprocess.run([SIM, CFG], input="F iq 0 58 0 8a\n", capture_output=True, text=True, timeout=60)
check("text ECC: a bit above 0x89 is an input error", p.returncode == 2 and "bit out of range" in p.stderr, (p.returncode, p.stderr))
p = subprocess.run([SIM, CFG], input="F iq 0 58 0\n", capture_output=True, text=True, timeout=60)
check("text ECC: F iq without its bit is an input error", p.returncode == 2 and "bit" in p.stderr, (p.returncode, p.stderr))

# 11. Fetched text (IfetchM): the check bits of a text word are those of the SRAM word it was read from; a fault of the SRAM word is
# found when the fetched instruction dispatches, not when it is fetched. Queue 0 (slice 0) runs `ifetchm A_ADDR` at cycle 0 (2 units: the
# fetched payload is text word 1 on, the image being word 0) and the fetched text holds a `read` of stream E:3.
import tsp_as as A
PROFILE = os.path.splitext(os.path.basename(CFG))[0]
ISA = A.ISA(PROFILE, CFG)
WU = VL // 4
def U(text): return A.word_units(ISA.parse(text)[1])
def text_word(units): return b"".join(u.to_bytes(4, "little") for u in (list(units) + [A.PADDING] * WU)[:WU])
FETCH_READY = 1 + FETCH_LAT + next((o["d_skew"] for o in CJ["timing"]["overrides"] if o["entry"] == "IfetchM"), CJ["timing"]["cu"]["d_skew"])
A_ADDR = 10
RD = U("read 0x40, E:3, 0x0")
words = {(0, A_ADDR): text_word(RD), (0, A_ADDR + 1): text_word([])}
def run_fetchm(flips, rl=(), expect=0):
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in words.items()]
    lines += A.q_lines({0: U(f"ifetchm {A_ADDR:#x}")}) + list(flips) + list(rl)
    p = subprocess.run([SIM, CFG, "--trace"], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == expect, (p.returncode, p.stderr, p.stdout[-300:])
    return p
def disp3(p): return [int(t[2]) - DM for t in (l.split() for l in p.stdout.splitlines()) if len(t) > 3 and t[:2] == ["T", "srf"] and t[3] == "E:3"]
p = run_fetchm([], rlines(0x010))
check("text ECC: IfetchM without a fault: the fetched read dispatches after the second read plus fetch_latency, no T ecc", disp3(p) == [FETCH_READY] and tl(p, "ecc") == [] and xs(p) == {0x010: 0}, (disp3(p), tl(p, "ecc")))
p = run_fetchm([f"F sram 0 0 {A_ADDR:x} 0 {28:x}"], rlines(0x010, 0x012, 0x020))
check("text ECC: IfetchM: a flip of the slice tag in the SRAM word is corrected when the fetched instruction dispatches (the read runs), one T ecc line for text word 1",
      disp3(p) == [FETCH_READY] and diags(p) == [] and tl(p, "ecc") == [f"T ecc {FETCH_READY} queue 0 corr text word 1 pos {pos_of(28)}"]
      and (xs(p).get(0x010), xs(p).get(0x012)) == (4, 1) and xs(p).get(0x020) == w0(FETCH_READY, 3, 0) | 0, (disp3(p), diags(p), tl(p, "ecc"), xs(p)))
p = run_fetchm([f"F sram 0 0 {A_ADDR:x} 0 0", f"F sram 0 0 {A_ADDR:x} 0 1"], rlines(0x010, 0x013, 0x030))
check("text ECC: IfetchM: a double flip in the SRAM word makes the fetched read OTS-E-ECC-002 at its dispatch; the read does not run; the interrupt is raised",
      disp3(p) == [] and [d for d in diags(p) if " queue 0 " in d] == [f"T diag {FETCH_READY} queue 0 OTS-E-ECC-002"]
      and tl(p, "ecc") == [f"T ecc {FETCH_READY} queue 0 unc text word 1 pos {pos_of2(0, 1)}"] and tl(p, "irq") == [f"T irq {FETCH_READY} msix 0"]
      and xs(p) == {0x010: 8, 0x013: 1, 0x030: 1}, (disp3(p), diags(p), tl(p, "ecc"), tl(p, "irq"), xs(p)))

print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
