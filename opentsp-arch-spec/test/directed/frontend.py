"""Directed tests of the instruction front end (microarchitecture.md sec. 3 "Instruction
front end"): the per-queue IQ of 32-bit
text units, the Parcel and Decode stages (bundle bit, NOP timing prefix, tags 6 and 7),
waiting (a held parcel dispatches when nothing holds it, and shifts every later parcel),
Ifetch and IfetchM text arriving at icu.fetch_latency and icu.fetch_bytes_per_cycle, padding,
the lint OTS-W-IQ-001 (underflow; a simulator lint, printed with --lint and not counted in E), the diagnostics
OTS-E-IQ-002 (IQ overflow), OTS-E-IQ-003 (fetch staging overflow) and OTS-E-IQ-004 (text left in an IQ at the end of the run),
the control channel's text, the input limits, and the lowering theorem of tools/tsp_as.py (lower).

Every expected cycle is computed here from the configuration, independently of the model.
Observation: a MEM Read onto stream E:s commits at its dispatch cycle + timing.mem.d_func.

usage: frontend.py <tsp_sim> <config.json>
"""
import json, os, random, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import tsp_as as A

SIM, CFG = sys.argv[1], sys.argv[2]
cfg = json.load(open(CFG))
PROFILE = os.path.splitext(os.path.basename(CFG))[0]
isa = A.ISA(PROFILE, CFG)
VL = 16 * cfg["geometry"]["superlanes"]
DM = cfg["timing"]["mem"]["d_func"]
LAT = cfg["icu"]["fetch_latency"]
LOCAL_LAT = LAT + 1  # payload completes after two successive local SRAM reads
RATE = cfg["icu"]["fetch_bytes_per_cycle"] // 4          # units per cycle
CAP = cfg["icu"]["iq_depth"] // 4                        # units
REL = cfg["barrier"]["release_latency"]
BANK = 1 << cfg["sram"]["bank_bit"]
WORD_UNITS = VL // 4                                     # units of one SRAM word
PAD = A.PADDING
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)
    ok &= bool(cond)


def U(text, bundle=False):
    """The units of one instruction (bit 31 of the first unit: the bundle bit)."""
    us = A.word_units(isa.parse(text)[1])
    if bundle:
        us[0] |= A.BUNDLE_BIT
    return us


def run(images, sram=None, config=CFG, dumps=(), tmo=600, flags=("--trace", "--lint"), events=()):
    lines = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in (sram or {}).items()]
    lines += A.q_lines(images) + [f"D {sl:x} {a:x}" for sl, a in dumps]
    lines.extend(events)
    r = subprocess.run([SIM, config, *flags], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=tmo)
    return r


def disp(out, stream, row="E"):
    """Dispatch cycles of the Reads whose result is on stream row:stream (srf commit - d_func), in order."""
    key = f"{row}:{stream}"
    return [int(t[2]) - DM for t in (ln.split() for ln in out.splitlines()) if len(t) > 3 and t[:2] == ["T", "srf"] and t[3] == key]


def diags(out, code=None, queue=None):
    r = []
    for ln in out.splitlines():
        t = ln.split()
        if t[:2] == ["T", "diag"] and (code is None or t[5] == code) and (queue is None or int(t[4]) == queue):
            r.append((int(t[2]), int(t[4]), t[5]))
    return r


def lints(out, code=None, queue=None):
    """The T lint lines (--lint): (cycle, queue, code)."""
    r = []
    for ln in out.splitlines():
        t = ln.split()
        if t[:2] == ["T", "lint"] and (code is None or t[5] == code) and (queue is None or int(t[4]) == queue):
            r.append((int(t[2]), int(t[4]), t[5]))
    return r


def e_line(out):
    return [ln for ln in out.splitlines() if ln.startswith("E ")]


def read(addr, stream, bundle=False):
    return U(f"read {addr:#x}, E:{stream}, 0x0", bundle)


def text_word(units):
    """An SRAM word (VL bytes) holding the units (little endian), padded with 0xFF."""
    us = (list(units) + [PAD] * WORD_UNITS)[:WORD_UNITS]
    return b"".join(u.to_bytes(4, "little") for u in us)


def with_config(**icu):
    """A temporary copy of the configuration with other icu values."""
    c = json.loads(json.dumps(cfg))
    c["icu"].update(icu)
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(c, f)
    f.close()
    return f.name


# ---------------------------------------------------------------------------------------
# 1. Bundle dispatch: the instructions of one parcel dispatch in one cycle.
two = U("read 0x1, E:2, 0x1")                                       # the other liquid thread: it fits the parcel
r = run({0: read(0x1, 1, bundle=True) + two})
out = r.stdout
check("bundle: two Reads (threads 0 and 1) dispatch together", disp(out, 1) == [0] and disp(out, 2) == [0] and not diags(out), out[:300])
r = run({0: read(0x1, 1) + two})
check("no bundle bit: the same two instructions are two parcels, one cycle apart",
      disp(r.stdout, 1) == [0] and disp(r.stdout, 2) == [1] and not diags(r.stdout), r.stdout[:300])
r = run({0: read(0x1, 1, bundle=True) + read(0x2, 2)})
check("bundle: two Reads of one thread do not fit a parcel: QUE-001 on each, none dispatches",
      diags(r.stdout) == [(0, 0, "OTS-E-QUE-001")] * 2 and not disp(r.stdout, 1), (diags(r.stdout), disp(r.stdout, 1)))

# ---------------------------------------------------------------------------------------
# 2. NOP is a timing prefix: earliest = last dispatch + 1 + the sum of the NOPs before the parcel.
r = run({0: [5] + read(0, 1) + [3] + read(0, 2) + [2, 3] + read(0, 3) + read(0, 4)})
check("NOP prefix: dispatch at 5, then 5+1+3, then +1+2+3, then +1 (no prefix)",
      (disp(r.stdout, 1), disp(r.stdout, 2), disp(r.stdout, 3), disp(r.stdout, 4)) == ([5], [9], [15], [16]) and not diags(r.stdout), r.stdout[:300])
r = run({0: [65535, 65535, 7] + read(0, 1)})
check("NOP prefix: units add (65535 + 65535 + 7)", disp(r.stdout, 1) == [131077], r.stdout[:300])

# ---------------------------------------------------------------------------------------
# 3. A parcel waits behind a parked queue (no diagnostic) and shifts the parcels after it.
sync, notify = U("sync"), U("notify")
r = run({0: [2] + sync + read(0, 1) + [3] + read(0, 2), 1: [10] + notify})
out = r.stdout
rel = 10 + REL
check("wait behind a Sync: the Read dispatches at the release (Notify 10 + release latency), no diagnostic",
      disp(out, 1) == [rel] and not diags(out), (disp(out, 1), diags(out)))
check("a held parcel shifts every later parcel: the next one is NOP 3 after the actual dispatch",
      disp(out, 2) == [rel + 1 + 3], disp(out, 2))
# a control-channel Sync parks every data queue; its Notify releases them
r = run({0: [5] + read(0, 1), 1: [1] + read(0, 2), 0xFFFF: [4] + sync + [2] + notify})
out = r.stdout
check("control Sync at 4, Notify at 7: queue 0 (earliest 5) waits to 7 + release latency; queue 1 (earliest 1) is not held",
      disp(out, 1) == [7 + REL] and disp(out, 2) == [1] and not diags(out), (disp(out, 1), disp(out, 2), diags(out)))
check("a Sync no Notify releases ends the run (the parked queue is never released)",
      run({0: [5] + read(0, 1), 0xFFFF: [4] + sync}).stdout.count("T srf") == 0)

# ---------------------------------------------------------------------------------------
# 4. A parcel waits behind a Repeat.
r = run({0: read(0, 0) + U("repeat 0x3, 0x2") + read(0, 5)})
out = r.stdout
check("wait behind a Repeat: Read at 0, iterations at 3, 6, 9, the next Read at 10",
      disp(out, 0) == [0, 3, 6, 9] and disp(out, 5) == [10] and not diags(out), (disp(out, 0), disp(out, 5), diags(out)))

# ---------------------------------------------------------------------------------------
# 5. NOP and a parcel while an AGT loop runs: the loop holds the path until its tail ends.
loop = read(0, 0) + U("countdown 0x1") + U("iterations 0x3")
r = run({0: loop + [1] + read(0, 9)})
out = r.stdout
check("AGT loop (references at 0, 2, 4, tail to 6): the Read with NOP 1 waits for the path, to 6",
      disp(out, 0) == [0, 2, 4] and disp(out, 9) == [6] and not diags(out), (disp(out, 0), disp(out, 9), diags(out)))
r = run({0: loop + [20] + read(0, 9)})
check("AGT loop: a NOP longer than the loop sets the dispatch (2 + 1 + 20)", disp(r.stdout, 9) == [23] and not diags(r.stdout), disp(r.stdout, 9))
r = run({0: read(0, 0) + [4, 4] + U("countdown 0x1")})
check("a configuration record after a gap is stray (OPR-001, at its cycle); the Read ran once (no level was configured)",
      disp(r.stdout, 0) == [0] and diags(r.stdout) == [(9, 0, "OTS-E-OPR-001")], (disp(r.stdout, 0), diags(r.stdout)))

# ---------------------------------------------------------------------------------------
# 6. IfetchM: text arrives icu.fetch_latency after its second SRAM read, at icu.fetch_bytes_per_cycle.
A_ADDR = 10
words = {(0, A_ADDR): text_word(read(0x40, 3)), (0, A_ADDR + 1): text_word([])}
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words)
out = r.stdout
check("IfetchM at 0: the fetched Read dispatches at second-read cycle + fetch_latency; the queue ran dry: one lint IQ-001 then, no diagnostic",
      disp(out, 3) == [LOCAL_LAT] and lints(out) == [(LOCAL_LAT, 0, "OTS-W-IQ-001")] and not diags(out) and e_line(out) == ["E - -1 0"],
      (disp(out, 3), lints(out), diags(out), e_line(out)))
r0 = run({0: U(f"ifetchm {A_ADDR:#x}")}, words, flags=("--trace",))
check("without --lint no T lint line is printed (the trace is unchanged)", disp(r0.stdout, 3) == [LOCAL_LAT] and "T lint" not in r0.stdout and r0.stdout.count("T lint") == 0, r0.stdout[:200])
r0 = run({0: U(f"ifetchm {A_ADDR:#x}")}, words, flags=("--lint",))
check("--lint without --trace prints the lint line alone", "T lint %d queue 0 OTS-W-IQ-001" % LOCAL_LAT in r0.stdout and "T srf" not in r0.stdout, r0.stdout[:200])
words = {(0, A_ADDR): text_word([30] + read(0x40, 3)), (0, A_ADDR + 1): text_word([])}
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words)
check("IfetchM with a NOP 30 prefix in the fetched text: dispatch at 0 + 1 + 30, no lint IQ-001",
      disp(r.stdout, 3) == [31] and not diags(r.stdout) and not lints(r.stdout), (disp(r.stdout, 3), diags(r.stdout), lints(r.stdout)))
c9 = with_config(fetch_latency=9)
words = {(0, A_ADDR): text_word(read(0x40, 3)), (0, A_ADDR + 1): text_word([])}
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words, config=c9)
check("fetch_latency 9 after the second read: dispatch at 10", disp(r.stdout, 3) == [10], disp(r.stdout, 3))
os.unlink(c9)
# fill rate: four two-unit parcels back to back
quad = [u for s in (3, 4, 5, 6) for u in read(0x40, s)]
words = {(0, A_ADDR): text_word(quad[:WORD_UNITS]), (0, A_ADDR + 1): text_word(quad[WORD_UNITS:])}
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words)
cyc = [disp(r.stdout, s)[0] for s in (3, 4, 5, 6)]
unit_cycle = lambda i: LOCAL_LAT + i // RATE                    # the cycle unit i of the payload enters the IQ
exp, last = [], -1
for k in range(4):
    complete = unit_cycle(2 * k + 1)
    last = max(complete, last + 1)
    exp.append(last)
check(f"fill rate {RATE} units a cycle: four Reads dispatch at {exp}", cyc == exp, (cyc, exp))
c1 = with_config(fetch_bytes_per_cycle=4)
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words, config=c1)
cyc = [disp(r.stdout, s)[0] for s in (3, 4, 5, 6)]
check("fetch_bytes_per_cycle 4 (one unit a cycle): the parcels are two cycles apart",
      cyc == [LOCAL_LAT + 1, LOCAL_LAT + 3, LOCAL_LAT + 5, LOCAL_LAT + 7], cyc)
os.unlink(c1)
# padding is dropped (and costs its place in the rate): a Read in word a and one in word a+1
words = {(0, A_ADDR): text_word(read(0x40, 3)), (0, A_ADDR + 1): text_word(read(0x41, 4))}
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words)
out = r.stdout
second = LOCAL_LAT + WORD_UNITS // RATE
check("padding in the fetched text is dropped, no DEC-003; the second word's Read enters after the padding",
      disp(out, 3) == [LOCAL_LAT] and disp(out, 4) == [max(second, LOCAL_LAT + 1)] and not diags(out, "OTS-E-DEC-003"),
      (disp(out, 3), disp(out, 4), diags(out)))
# two fetches queue behind each other: the second payload enters after the first one
words = {(0, A_ADDR): text_word(read(0x40, 3)), (0, A_ADDR + 1): text_word([]),
         (0, A_ADDR + 2): text_word(read(0x41, 4)), (0, A_ADDR + 3): text_word([])}
r = run({0: U(f"ifetchm {A_ADDR:#x}") + U(f"ifetchm {A_ADDR + 2:#x}")}, words)
check("two IfetchM: the second payload queues behind the first (units at the fetch rate)",
      disp(r.stdout, 3) == [LOCAL_LAT] and disp(r.stdout, 4) == [max(LOCAL_LAT + 2 * WORD_UNITS // RATE, LOCAL_LAT + 2)] and not diags(r.stdout, "OTS-E-DEC-003"),
      (disp(r.stdout, 3), disp(r.stdout, 4), diags(r.stdout)[:3]))

# ---------------------------------------------------------------------------------------
# 7. Ifetch (the stream form): ONE stream operand carries the two vectors on consecutive cycles (ISCA20 sec. III-A3:
# "a single stream operand ... 640-bytes (a pair of 320-byte vectors)"); the first is read at the fetch's execution cycle t,
# the second at t + 1, and the text reaches the IQ icu.fetch_latency cycles after the second one (any stream, no even rule).
GAP = lambda sl, row: (18 - (sl % 44) // 4) if ((row == "E") == (sl < 44)) else 24 + (sl % 44) // 4
T0 = 20
# The feeders must be upstream of the fetching slice on the E row. On a profile with few slices
# The fetching slice is the first one of the east hemisphere (gap 24), fed from the west.
if cfg['geometry']['mem_slices_per_hemisphere'] > 12:
    RQ, SL6 = 0, 8
else:
    RQ, SL6 = 44, 1


def ifetch_case(st, first=True, second=True, t0=T0):
    """Ifetch E:st at t0; the text vectors (two Reads, 0x40 and 0x41, onto stream st) are read from SL6 so that they reach the
    fetching queue at t0 and t0 + 1 (a vector left out is not fed)."""
    feeders, sram = [], {}
    for k, (addr, present, dispatch_read) in enumerate(((1, first, 11), (2, second, 12))):
        if not present:
            continue
        sram[(SL6, addr)] = text_word(read(0x40 + k, dispatch_read))
        feeders.append((t0 + k - DM - (GAP(RQ, "E") - GAP(SL6, "E")), SL6, isa.parse(f"read {addr:#x}, E:{st}, 0x0")[1]))
    images = A.lower(feeders)
    images[RQ] = [t0] + U(f"ifetch E:{st}")
    return run(images, sram).stdout


for st in (6, 7):
    out = ifetch_case(st)
    check(f"Ifetch E:{st} at {T0} (one stream, two vectors on consecutive cycles): the fetched Reads dispatch at {T0} + 1 + fetch_latency and one word later"
          + (" (an odd stream: no even rule)" if st % 2 else ""),
          disp(out, 11) == [T0 + 1 + LAT] and disp(out, 12) == [max(T0 + 1 + LAT + WORD_UNITS // RATE, T0 + 2 + LAT)]
          and not diags(out, "OTS-E-DEC-003") and not diags(out, "OTS-E-OPR-002"),
          (disp(out, 11), disp(out, 12), diags(out)[:3]))
out = ifetch_case(6, second=False)
check("Ifetch with the first vector only (nothing on its stream the next cycle): OTS-E-TIM-001 at the second sample, nothing is fetched",
      (T0 + 1, RQ, "OTS-E-TIM-001") in diags(out, None, RQ) and disp(out, 11) == [] and disp(out, 12) == [], (diags(out)[:3], disp(out, 11)))
out = ifetch_case(6, first=False)
check("Ifetch with the second vector only (nothing on its stream at the execution cycle): OTS-E-TIM-001, nothing is fetched",
      (T0, RQ, "OTS-E-TIM-001") in diags(out, None, RQ) and disp(out, 11) == [] and disp(out, 12) == [], (diags(out)[:3], disp(out, 11)))

# ---------------------------------------------------------------------------------------
# Two flips of check bits leave the opcode bytes intact. A later clean word still executes.
if PROFILE == "ref":
    images = {0: read(0, 0) + read(1, 0), 88: [8] + U("ifetch E:0")}
    words = {(0, 0): text_word([]), (0, 1): text_word([0x52000001] + [PAD] * 3 + [0x52000002])}
    ev = ["F srf 9 e 0 15 0 80", "F srf 9 e 0 15 0 81"]
    out = run(images, words, events=ev).stdout
    irqs = [int(t[-1]) for t in map(str.split, out.splitlines()) if t[:2] == ["T", "irq"]]
    check("second-beat ECC: suppress vector 1, signal the error, execute the clean instruction",
          (9, 88, "OTS-E-ECC-001") in diags(out) and 1 not in irqs and 0 in irqs and 2 in irqs,
          (diags(out), irqs))

# 8. Diagnostic IQ-002: units that do not fit are lost, and the loss is an architectural diagnostic (once per queue and cycle).
PAYLOAD = 2 * WORD_UNITS
full = U(f"ifetchm {A_ADDR:#x}") + [65535] * (CAP - 2)           # after the IfetchM dispatches, 2 units of room
words = {(0, A_ADDR): text_word([1] * WORD_UNITS), (0, A_ADDR + 1): text_word([1] * WORD_UNITS)}
r = run({0: full}, words)
got = [c for c, q, code in diags(r.stdout, "OTS-E-IQ-002") if q == 0]
steps = -(-PAYLOAD // RATE)
check(f"IQ-002: a full IQ loses the payload: one diagnostic per cycle for {steps} cycles from fetch_latency, counted in E, no lint",
      got == list(range(LOCAL_LAT, LOCAL_LAT + steps)) and len(diags(r.stdout)) == len(got) and not lints(r.stdout)
      and e_line(r.stdout) == [f"E OTS-E-IQ-002 {LOCAL_LAT} {steps}"],
      (got, diags(r.stdout)[:4], lints(r.stdout)[:4], e_line(r.stdout)))
# The drop rule: units that arrive at a full IQ are lost, and the diagnostic reports them (same with or without --lint).
words_drop = {(0, A_ADDR): text_word([1, 1] + read(0x40, 3)), (0, A_ADDR + 1): text_word([])}
r = run({0: full}, words_drop)
check("IQ-002: the units that find the IQ full are lost (the Read in the payload never dispatches); the diagnostic reports them",
      disp(r.stdout, 3) == [] and bool(diags(r.stdout, "OTS-E-IQ-002")), (disp(r.stdout, 3), diags(r.stdout)[:3]))
r = run({0: full}, words_drop, flags=("--trace",))
check("IQ-002: the diagnostic does not depend on --lint", disp(r.stdout, 3) == [] and bool(diags(r.stdout, "OTS-E-IQ-002")) and not lints(r.stdout), (disp(r.stdout, 3), diags(r.stdout)[:3]))
fits = U(f"ifetchm {A_ADDR:#x}") + [65535] * (CAP - 2 - PAYLOAD)
r = run({0: fits}, words)
check("IQ-002: a payload that exactly fits is not an overflow", not diags(r.stdout, "OTS-E-IQ-002"), diags(r.stdout)[:3])
words = {(0, A_ADDR): text_word([PAD] * WORD_UNITS), (0, A_ADDR + 1): text_word([])}
r = run({0: full}, words)
check("IQ-002: padding takes no room in the IQ", not diags(r.stdout) and not lints(r.stdout), (diags(r.stdout)[:3], lints(r.stdout)[:3]))

# ---------------------------------------------------------------------------------------
# 9. Lint IQ-001 and the end of a program.
r = run({0: read(0, 1)})
check("no IQ-001 at the end of the program (nothing is fetched, nothing is missing)", not diags(r.stdout) and not lints(r.stdout) and disp(r.stdout, 1) == [0], r.stdout[:200])
r = run({0: read(0, 1) + [50] + read(0, 2)})
check("a long NOP prefix is not an underflow", not diags(r.stdout) and not lints(r.stdout) and disp(r.stdout, 2) == [51], r.stdout[:200])

# ---------------------------------------------------------------------------------------
# 10. The control channel's text.
r = run({0xFFFF: U("sync", True) + U("notify")})
check("control channel: two instructions in one parcel are QUE-001 each",
      diags(r.stdout) == [(0, 0xFFFF, "OTS-E-QUE-001")] * 2, diags(r.stdout))
r = run({0xFFFF: [3] + read(0, 1)})
check("control channel: a Read is QUE-002 at its cycle", diags(r.stdout) == [(3, 0xFFFF, "OTS-E-QUE-002")], diags(r.stdout))
r = run({0xFFFF: [3] + U("config 0x1")})
check("control channel: Config is accepted", not diags(r.stdout), diags(r.stdout))

# ---------------------------------------------------------------------------------------
# 11. An image longer than the IQ is an input error; padding does not count.
over = [1] * (CAP + 1)
r = run({0: over})
check("image of iq_depth/4 + 1 units: input error (exit 2)", r.returncode == 2 and "line" in r.stderr, (r.returncode, r.stderr[:200]))
r = run({0: [1] * CAP})
check("image of exactly iq_depth/4 units is accepted", r.returncode == 0, (r.returncode, r.stderr[:200]))
r = run({0: [1] * CAP + [PAD] * 50})
check("padding units do not count against the capacity", r.returncode == 0, (r.returncode, r.stderr[:200]))
for line in ("Q 90 00000001", "Q ffff0 00000001", "Q 0 100000000", "Q 0 zz", "Q 0 -1"):
    r = subprocess.run([SIM, CFG], input=line + "\n", capture_output=True, text=True)
    check(f"input line {line!r} is rejected (exit 2)", r.returncode == 2, (r.returncode, r.stderr[:100]))
r = subprocess.run([SIM, CFG], input="P 0 0 1 400000 0\n", capture_output=True, text=True)
check("the old P record line is rejected (exit 2)", r.returncode == 2, (r.returncode, r.stderr[:100]))

# : an S line longer than the lane count is an input error naming the line and the lane count
# Include a 320-byte ref image in the smaller-profile rejection tests.
def sline(n, line=1):
    return ("# comment\n" * (line - 1)) + "S 0 1 " + " ".join(f"{(i % 255) + 1:x}" for i in range(n)) + "\n"
r = subprocess.run([SIM, CFG], input=sline(VL + 1, 3), capture_output=True, text=True)
check(f" an S line of {VL + 1} bytes (lanes {VL}): exit 2, the message names line 3 and the lane count {VL}",
      r.returncode == 2 and "input line 3" in r.stderr and str(VL) in r.stderr and "lane" in r.stderr and r.stdout == "", (r.returncode, r.stderr[:200], r.stdout[:80]))
r = subprocess.run([SIM, CFG], input="S 0 1 " + " ".join(["1"] * 320) + "\n", capture_output=True, text=True)
check(" an S line of 320 bytes (a ref-sized line): accepted on ref and on every wider profile, exit 2 with the lane count on every smaller profile",
      (r.returncode == 0) if VL >= 320 else (r.returncode == 2 and str(VL) in r.stderr and "line 1" in r.stderr), (r.returncode, r.stderr[:200]))
r = run({}, sram={(0, 1): [(i % 255) + 1 for i in range(VL)]}, dumps=[(0, 1)])
check(f" an S line of exactly {VL} bytes is accepted and dumps back whole",
      r.returncode == 0 and r.stdout.split("W 0 1 ")[1].split("\n")[0].split()[:VL] == [f"{(i % 255) + 1:02x}" for i in range(VL)], (r.returncode, r.stderr[:100]))
r = run({}, sram={(0, 1): [7] * (VL - 3)}, dumps=[(0, 1)])
check(" a shorter S line is zero-filled (unchanged)", r.returncode == 0 and r.stdout.split("W 0 1 ")[1].split("\n")[0].split()[VL - 3:VL] == ["00"] * 3, r.stdout[:120])

# ---------------------------------------------------------------------------------------
# 12. Tags 6 and 7 do not decode: DEC-003, one unit consumed.
for tag in (6, 7):
    r = run({0: [(tag << 28) | 5] + read(0, 1)})
    check(f"tag {tag}: DEC-003 at 0 consuming one unit; the next parcel dispatches at 1",
          diags(r.stdout) == [(0, 0, "OTS-E-DEC-003")] and disp(r.stdout, 1) == [1], (diags(r.stdout), disp(r.stdout, 1)))
r = run({0: [A.BUNDLE_BIT | (7 << 28)] + read(0, 1)})
check("an undecodable instruction bundled with another: QUE-001 on each, none dispatches",
      diags(r.stdout) == [(0, 0, "OTS-E-QUE-001")] * 2 and not disp(r.stdout, 1), (diags(r.stdout), disp(r.stdout, 1)))
r = run({0xFFFF: [(6 << 28) | 1]})
check("control channel: tag 6 is DEC-003", diags(r.stdout) == [(0, 0xFFFF, "OTS-E-DEC-003")], diags(r.stdout))

# ---------------------------------------------------------------------------------------
# 12b. A whole-queue instruction meets an AGT path whose configuration is still open
# (design 7.6: a reference issued in cycle c also holds the queue for it). The instruction
# right after the Read, or after a configuration record, finds the path in state Config, with
# a reference due in that cycle or not; it must wait for the loop and its tail, with no
# diagnostic and no model fault (tsp_sim exits with 4 on an internal error).
def run_fast(images):
    try:
        return run(images, tmo=60)
    except subprocess.TimeoutExpired:
        return None


def whole_queue(name, body, loop_refs, tail_end):
    """body: the queue text up to the SetVL (its first unit is at cycle 0); the loop's references
    are on stream 1, the Read after the SetVL on stream 2 (a lone parcel: one cycle after it)."""
    r = run_fast({0: body + U("setvl 0x4") + read(0x9, 2)})
    good = r is not None and r.returncode == 0 and not diags(r.stdout)
    got = (disp(r.stdout, 1), disp(r.stdout, 2)) if good else None
    check(f"whole-queue instruction after open configuration, {name}: waits to the loop's end (cycle {tail_end}), the next Read follows it",
          good and got == (loop_refs, [tail_end + 1]),
          ("timeout" if r is None else (r.returncode, r.stderr[:200], got)))


# R path: Read, Countdown cd: references at 0 and 1 + cd (default 2 iterations); the SetVL is due
# at cycle 2, when the configuration ends (a reference is due then iff cd = 1: 1 + cd = 2).
whole_queue("reference due in the cycle (Countdown 1)", read(0x1, 1) + U("countdown 0x1"), [0, 2], 4)
whole_queue("no reference due in the cycle (Countdown 3)", read(0x1, 1) + U("countdown 0x3"), [0, 4], 8)
whole_queue("three references (Countdown 1, Iterations 3)", read(0x1, 1) + U("countdown 0x1") + U("iterations 0x3"), [0, 2, 4], 6)
# the Write path: the loop is on the W path, the instruction needs both paths
r = run_fast({0: U("write 0x1, E:3, 0x0") + U("countdown 0x1") + U("setvl 0x4") + read(0x9, 2)})
check("whole-queue instruction after an open W-path configuration (Countdown 1): waits to cycle 4, the next Read follows",
      r is not None and r.returncode == 0 and {d[2] for d in diags(r.stdout)} <= {"OTS-E-TIM-001"} and disp(r.stdout, 2) == [5],   # TIM-001: the Write reads an empty stream
      "timeout" if r is None else (r.returncode, r.stderr[:200], disp(r.stdout, 2), diags(r.stdout)))
# no configuration record: the lone Read leaves its path free, so the instruction is not delayed
r = run_fast({0: read(0x1, 1) + U("setvl 0x4") + read(0x9, 2)})
check("a Read without configuration records: the SetVL after it dispatches in the next cycle (unchanged)",
      r is not None and r.returncode == 0 and not diags(r.stdout) and disp(r.stdout, 1) == [0] and disp(r.stdout, 2) == [2],
      "timeout" if r is None else (r.returncode, r.stderr[:200], disp(r.stdout, 1), disp(r.stdout, 2)))

# ---------------------------------------------------------------------------------------
# 13. The lowering theorem: a program whose entries never wait runs every entry at its cycle.
rnd = random.Random(1)
for trial in range(12):
    entries, want = [], {}
    # MEM queues that exist on the profile (all 88 on ref and mini)
    qs = rnd.sample([s for s in range(88) if s % 44 < cfg['geometry']['mem_slices_per_hemisphere']], 6)
    for q in qs:
        c = rnd.choice([0, 1, 2, 7])
        for k in range(rnd.randint(1, 5)):
            stream = qs.index(q) * 5 + k
            w = isa.parse(f"read 0x{rnd.randrange(64):x}, E:{stream}, 0x0")[1]
            entries.append((c, q, w))
            want[(q, stream)] = want.get((q, stream), []) + [c]
            c += rnd.choice([1, 1, 2, 3, 70000 if trial == 11 and k == 2 else 5])
            if rnd.random() < 0.3:                                   # a lone NOP entry: a timing instruction
                entries.append((c, q, isa.parse("nop 0x4")[1]))
                c += 5
    images = A.lower(entries)
    out = run(images).stdout
    got = {}
    for q, stream in want:
        got[(q, stream)] = disp(out, stream)
    # one stream may carry several queues' Reads: compare per stream the merged, sorted dispatch cycles
    by_stream = {}
    for (q, stream), cs in want.items():
        by_stream.setdefault(stream, []).extend(cs)
    good = all(sorted(by_stream[s]) == sorted(disp(out, s)) for s in by_stream) and not diags(out)
    check(f"lowering theorem, trial {trial}: {len(entries)} entries on 6 queues dispatch at their cycles", good,
          {s: (sorted(by_stream[s]), sorted(disp(out, s))) for s in by_stream if sorted(by_stream[s]) != sorted(disp(out, s))})

# ---------------------------------------------------------------------------------------
# 14. : text left in an IQ at the end of the run is OTS-E-IQ-004 (a truncated or mis-patched image used to end "done" with no report).
# One report per queue, after the last cycle, at the cycle of the queue's last IQ activity (a unit entering or a dispatch).
STR = "OTS-E-IQ-004"
r = run({0: read(0, 1, bundle=True)})
check(": the last instruction of the text has the bundle bit: IQ-004 on queue 0 at cycle 0; the Read never dispatches",
      diags(r.stdout) == [(0, 0, STR)] and not disp(r.stdout, 1) and e_line(r.stdout) == [f"E {STR} 0 1"], (diags(r.stdout), e_line(r.stdout)))
r = run({0: read(0, 1)[:1]})
check(": the text ends in the first unit of a 64-bit instruction: IQ-004", diags(r.stdout) == [(0, 0, STR)], diags(r.stdout))
r = run({0: read(0, 1) + [3] + read(0, 2)[:1]})
check(": a Read, then a NOP prefix and half an instruction: the Read dispatches at 0, IQ-004 at cycle 0 (the last dispatch)",
      disp(r.stdout, 1) == [0] and diags(r.stdout) == [(0, 0, STR)], (disp(r.stdout, 1), diags(r.stdout)))
r = run({0: [4] + read(0, 1) + [5] + read(0, 2, bundle=True)})
check(": the second Read has the bundle bit and nothing follows: the first dispatches at 4, IQ-004 at cycle 4",
      disp(r.stdout, 1) == [4] and not disp(r.stdout, 2) and diags(r.stdout) == [(4, 0, STR)], (disp(r.stdout, 1), disp(r.stdout, 2), diags(r.stdout)))
r = run({0xFFFF: [3] + U("notify", True)})
check(": the control channel's text ends with a bundle bit: IQ-004 on queue 65535", diags(r.stdout) == [(0, 0xFFFF, STR)], diags(r.stdout))
r = run({0: read(0, 1)[:1], 1: read(0, 2, bundle=True), 0xFFFF: [3] + U("notify", True)})
check(": one report per queue with leftover text, in ascending queue order, the control channel (65535) last",
      diags(r.stdout) == [(0, 0, STR), (0, 1, STR), (0, 0xFFFF, STR)] and e_line(r.stdout) == [f"E {STR} 0 3"], (diags(r.stdout), e_line(r.stdout)))
words = {(0, A_ADDR): text_word(read(0x40, 3)[:1]), (0, A_ADDR + 1): text_word([])}
r = run({0: U(f"ifetchm {A_ADDR:#x}")}, words)
check(": fetched text that ends in half an instruction: IQ-004 at the cycle its unit entered the IQ (fetch_latency)",
      diags(r.stdout) == [(LOCAL_LAT, 0, STR)] and not disp(r.stdout, 3), (diags(r.stdout), disp(r.stdout, 3)))
# not a truncation:
r = run({0: read(0, 1) + [5]})
check(": a trailing NOP prefix (a complete NOP with no successor) is not a truncation: no diagnostic", not diags(r.stdout) and disp(r.stdout, 1) == [0], (diags(r.stdout), disp(r.stdout, 1)))
r = run({0: [5] + read(0, 1)})
check(": a complete parcel is not a truncation: no diagnostic", not diags(r.stdout) and disp(r.stdout, 1) == [5], (diags(r.stdout), disp(r.stdout, 1)))
r = run({0: [2] + sync + read(0, 1) + read(0, 2)[:1]})
check(": a queue parked at a Sync no Notify releases keeps its text without a report (it never resumes, platform.md section 1)", not diags(r.stdout), diags(r.stdout))
r = run({0: [2] + sync + read(0, 1, bundle=True), 1: [10] + notify})
check(": a Sync that is released, then a bundle bit on the last instruction: IQ-004 after the release",
      [(q, code) for c, q, code in diags(r.stdout)] == [(0, STR)], diags(r.stdout))
r = run({0: read(0, 1, bundle=True)}, flags=("--lint",))
check(": the report is a diagnostic, not a lint (printed without --lint, no T lint line)", not lints(r.stdout) and bool(diags(run({0: read(0, 1, bundle=True)}, flags=("--trace",)).stdout)), r.stdout[:200])

# ---------------------------------------------------------------------------------------
# 15. : the fetch staging FIFO is bounded by icu.fetch_staging_depth payloads; a fetch that finds it full is OTS-E-IQ-003 and has no effect.
D0 = cfg["icu"].get("fetch_staging_depth")
check(": the profile sets icu.fetch_staging_depth (an integer of at least 1)", isinstance(D0, int) and D0 >= 1, D0)
three = {(0, A_ADDR): text_word(read(0x40, 3)), (0, A_ADDR + 1): text_word([]),
         (0, A_ADDR + 2): text_word(read(0x41, 4)), (0, A_ADDR + 3): text_word([]),
         (0, A_ADDR + 4): text_word(read(0x42, 5)), (0, A_ADDR + 5): text_word([])}
fetches = U(f"ifetchm {A_ADDR:#x}") + U(f"ifetchm {A_ADDR + 2:#x}") + U(f"ifetchm {A_ADDR + 4:#x}")
c2 = with_config(fetch_staging_depth=2, fetch_latency=4)
r = run({0: fetches}, three, config=c2)
check(" depth 2: the third IfetchM (executes at cycle 4, two payloads still staged) is OTS-E-IQ-003 at cycle 4 and delivers nothing; the first two Reads dispatch",
      diags(r.stdout) == [(4, 0, "OTS-E-IQ-003")] and bool(disp(r.stdout, 3)) and bool(disp(r.stdout, 4)) and not disp(r.stdout, 5),
      (diags(r.stdout), disp(r.stdout, 3), disp(r.stdout, 4), disp(r.stdout, 5)))
os.unlink(c2)
c3 = with_config(fetch_staging_depth=3, fetch_latency=4)
r = run({0: fetches}, three, config=c3)
check(" depth 3: the same three fetches are accepted and all three Reads dispatch",
      not diags(r.stdout) and bool(disp(r.stdout, 3)) and bool(disp(r.stdout, 4)) and bool(disp(r.stdout, 5)), (diags(r.stdout), disp(r.stdout, 5)))
os.unlink(c3)
c1 = with_config(fetch_staging_depth=1, fetch_latency=4)
r = run({0: fetches}, three, config=c1)
check(" depth 1: the second and third IfetchM are OTS-E-IQ-003 (cycles 2 and 4); only the first Read dispatches",
      diags(r.stdout) == [(2, 0, "OTS-E-IQ-003"), (4, 0, "OTS-E-IQ-003")] and bool(disp(r.stdout, 3)) and not disp(r.stdout, 4) and not disp(r.stdout, 5),
      (diags(r.stdout), disp(r.stdout, 3), disp(r.stdout, 4), disp(r.stdout, 5)))
wait = LAT_STAGE = 4 + -(-PAYLOAD // RATE) + 2                    # fetch_latency 4 here: payload 1 is fully in the IQ after 4 + steps cycles
spaced = U(f"ifetchm {A_ADDR:#x}") + [wait] + U(f"ifetchm {A_ADDR + 2:#x}")
r = run({0: spaced}, three, config=c1)
check(" depth 1: a payload that has entered the IQ no longer counts; a second fetch after it is accepted",
      not diags(r.stdout) and bool(disp(r.stdout, 3)) and bool(disp(r.stdout, 4)), (diags(r.stdout), disp(r.stdout, 3), disp(r.stdout, 4)))
os.unlink(c1)

# ---------------------------------------------------------------------------------------
# 16. : a repeated local fetch reserves its second beat exactly as an explicit second IfetchM does.
# "ifetchm a; nop 1; ifetchm a" and "ifetchm a; nop 1; repeat 1, 1" are two encodings of one schedule (IfetchM at 0 and at 2): the
# second fetch's second read is at cycle 3, so the R path is held through cycle 3 (a following R-type instruction dispatches at 4) and
# the bank of word a + 1 is taken in cycle 3 (a Write of that bank dispatched at 3 is OTS-E-MEM-001, not a port overlap TIM-001).
FW = {(0, A_ADDR): text_word([]), (0, A_ADDR + 1): text_word([])}
IFM = U(f"ifetchm {A_ADDR:#x}")
WR = U(f"write {A_ADDR + 1:#x}, E:5, 0x0")
plain = run({0: IFM + [1] + IFM + read(0x20, 7)}, FW).stdout
rep = run({0: IFM + [1] + U("repeat 0x1, 0x1") + read(0x20, 7)}, FW).stdout
check(" read: explicit second IfetchM holds the R path through its second beat: the Read dispatches at 4",
      disp(plain, 7) == [4] and not diags(plain), (disp(plain, 7), diags(plain)))
check(" read: the repeated IfetchM holds the R path exactly as the explicit one: the Read dispatches at 4 (was 3)",
      disp(rep, 7) == disp(plain, 7) and not diags(rep), (disp(rep, 7), disp(plain, 7), diags(rep)))
plain = run({0: IFM + [1] + IFM + WR}, FW).stdout
rep = run({0: IFM + [1] + U("repeat 0x1, 0x1") + WR}, FW).stdout
check(" write: a Write of the second beat's bank at cycle 3 is OTS-E-MEM-001 after the explicit second IfetchM",
      diags(plain) == [(3, 0, "OTS-E-MEM-001")], diags(plain))
check(" write: the same Write after the repeated IfetchM is OTS-E-MEM-001 at cycle 3 too (was OTS-E-TIM-001)",
      diags(rep) == diags(plain), (diags(rep), diags(plain)))
plain = run({0: IFM + [1] + IFM + [1] + IFM + read(0x20, 7)}, FW).stdout
rep = run({0: IFM + [1] + U("repeat 0x2, 0x1") + read(0x20, 7)}, FW).stdout
check(" two iterations (spacing 2): the Read after them dispatches at 6 in both encodings",
      disp(plain, 7) == [6] and disp(rep, 7) == [6] and not diags(plain) and not diags(rep), (disp(plain, 7), disp(rep, 7), diags(plain), diags(rep)))
# Repeat 2, 0: iterations at 2 and 3 (spacing 1). The second overlaps the first one's second read: OTS-E-QUE-001 at its execute cycle 3, and,
# as for an explicit IfetchM that fails its own checks, it still holds the R path through the cycle after it (the Read dispatches at 5).
rep = run({0: IFM + U("repeat 0x2, 0x0") + read(0x20, 7)}, FW).stdout
check(" spacing 1: the overlapping iteration is OTS-E-QUE-001 at 3 and the Read waits for its second beat (5)",
      diags(rep) == [(3, 0, "OTS-E-QUE-001")] and disp(rep, 7) == [5], (diags(rep), disp(rep, 7)))
# a repeated Read is unchanged: it holds its path through its own cycle only.
rep = run({0: read(0x20, 6) + U("repeat 0x1, 0x1") + read(0x20, 7)}, FW).stdout
check(" control: a repeated Read holds its path through its own cycle only: Reads at 0 and 2, then 3",
      disp(rep, 6) == [0, 2] and disp(rep, 7) == [3] and not diags(rep), (disp(rep, 6), disp(rep, 7), diags(rep)))

print("frontend:", "all passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
