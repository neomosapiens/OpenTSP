"""Directed end-to-end tests of the replay policy of the error interrupt: `--on-irq stop|continue|replay`, the
transient (`t`) and persistent (`p`) marks of the `F` lines, `T replay start|resolved|persistent` and exit status 5
(microarchitecture.md sec. 16.2). Runs against the Sail emulator or the ISS (both take the same command line).
Profile-portable: ref, mini and the test profiles.
usage: ras_replay.py <tsp_sim|tsp-iss> <config.json>"""
import json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from tsp_as import lower_records, q_lines
SIM = sys.argv[1]; CFG = sys.argv[2]
CJ = json.load(open(CFG)); TM = CJ["timing"]
SL = CJ["geometry"]["superlanes"]; VL = 16 * SL
VXM_DS = TM["vxm"]["d_skew"]; DV = TM["vxm"]["d_func"]
DM = TM["mem"]["d_func"]; MEM_DS = TM["mem"]["d_skew"]
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


def vxm_program(t=T):
    """A on E:0 (slice 0 addr 1) and B on E:4 (slice 1 addr 2) read by the VXM at gap 21; the sum on E:8 to slice 44 addr 9."""
    recs, sram = [], {}
    feed(recs, sram, 0, 1, A, 0, "E", 21, t + VXM_DS)
    feed(recs, sram, 1, 2, B, 4, "E", 21, t + VXM_DS)
    recs.append((t, VXMQ, 1, add_i8(E(8), E(0), E(4)), 0))
    drain(recs, 44, 9, 8, "E", 21, t + DV)
    return recs, sram


def fsram(cycle, slice_, addr, sl, bit, tp=""): return f"F sram {cycle:x} {slice_:x} {addr:x} {sl:x} {bit:x}" + (" " + tp if tp else "")


def go(extra=(), opts=(), expect=0, dumps=((44, 9),)):
    recs, sram = vxm_program()
    ls = [f"S {sl:x} {a:x} " + " ".join(f"{b:x}" for b in v) for (sl, a), v in sram.items()]
    ls += q_lines(lower_records(list(recs))) + [f"D {sl:x} {a:x}" for sl, a in dumps] + list(extra)
    p = subprocess.run([SIM, CFG, "--trace", *opts], input="\n".join(ls) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == expect, (opts, extra, p.returncode, p.stderr, p.stdout[-600:])
    return p


def ln(p, prefix): return [l for l in p.stdout.splitlines() if l.startswith(prefix)]
def sections(p):
    """The output split at `T replay start`: (first run, second run or None)."""
    a, sep, b = p.stdout.partition("T replay start\n")
    return a, (b if sep else None)
SUM = bytes((x + y) & 0xFF for x, y in zip(A, B))
def wline(text): return [l for l in text.splitlines() if l.startswith("W ")]
GOOD_W = "W 2c 9 " + " ".join(f"{b:02x}" for b in SUM)


DBL = [fsram(0, 0, 1, 0, 5), fsram(0, 0, 1, 0, 9)]       # two bits of one stored word: an uncorrectable error at the VXM
DBL_T = [fsram(0, 0, 1, 0, 5, "t"), fsram(0, 0, 1, 0, 9, "t")]
DBL_P = [fsram(0, 0, 1, 0, 5, "p"), fsram(0, 0, 1, 0, 9, "p")]

# 0. Reference runs.
clean = go()
dbl = go(DBL)
check("replay: reference: a clean run has no error interrupt and the sum is stored", not ln(clean, "T irq") and GOOD_W in clean.stdout, clean.stdout[-300:])
check("replay: reference: a double flip raises the error interrupt (T irq ... msix 0) and the sum is not the clean sum",
      len(ln(dbl, "T irq")) == 1 and "msix 0" in ln(dbl, "T irq")[0] and GOOD_W not in dbl.stdout, ln(dbl, "T irq"))

# 1. The default is continue: same output with and without the option, exit 0.
p = go(DBL, ["--on-irq", "continue"])
check("replay: --on-irq continue gives the output of a run without the option", p.stdout == dbl.stdout and p.stderr == dbl.stderr)

# 2. stop: the run ends at the end of the cycle of the first error interrupt; exit 5; the results are printed as they stand.
p = go(DBL, ["--on-irq", "stop"], expect=5)
irq_c = int(ln(dbl, "T irq")[0].split()[2])
last_t = max(int(l.split()[2]) for l in p.stdout.splitlines() if l.startswith("T ") and l.split()[1] in ("srf", "sram", "diag", "ecc", "irq"))
check("replay: --on-irq stop: exit 5, the output ends at the cycle of the interrupt and has an E line and the W line",
      len(ln(p, "T irq")) == 1 and last_t <= irq_c and len(ln(p, "E ")) == 1 and len(ln(p, "W ")) == 1 and "T replay" not in p.stdout,
      (irq_c, last_t, p.stdout[-300:]))
tl = lambda q: [l for l in q.stdout.splitlines() if l.startswith("T ")]
check("replay: --on-irq stop: its trace lines are the first lines of the continue run's trace, and the sum was not stored",
      tl(p) == tl(dbl)[:len(tl(p))] and 0 < len(tl(p)) < len(tl(dbl)) and GOOD_W not in p.stdout, (len(tl(p)), len(tl(dbl))))
p = go([], ["--on-irq", "stop"])
check("replay: --on-irq stop with no interrupt: identical to the plain run, exit 0", p.stdout == clean.stdout)

# 3. replay of a transient fault: resolved.
p = go(DBL_T, ["--on-irq", "replay"])
a, b = sections(p)
check("replay: transient double flip: the first run raised the interrupt, `T replay start`, the second run has none, `T replay resolved`, exit 0",
      b is not None and len(ln(p, "T irq")) == 1 and "T replay resolved" in b and not [l for l in b.splitlines() if l.startswith(("T irq", "T diag", "T ecc"))],
      p.stdout[-500:])
check("replay: the second run stores the clean sum, the first run does not", GOOD_W in b and GOOD_W not in a, (a[-200:], b[-300:]))
check("replay: the first run is the run of --on-irq continue", a == go(DBL_T).stdout)
# 4. persistent: the second run raises it again.
p = go(DBL_P, ["--on-irq", "replay"], expect=5)
a, b = sections(p)
check("replay: persistent double flip: the second run raises the interrupt again, `T replay persistent`, exit 5",
      b is not None and len(ln(p, "T irq")) == 2 and p.stdout.rstrip().endswith("T replay persistent") and GOOD_W not in b, p.stdout[-400:])
check("replay: the two runs of a persistent fault are identical", a == b.rsplit("T replay persistent\n", 1)[0])
# 5. mixed: one persistent flip alone is correctable, so the replay resolves.
p = go([fsram(0, 0, 1, 0, 5, "p"), fsram(0, 0, 1, 0, 9, "t")], ["--on-irq", "replay"])
a, b = sections(p)
check("replay: one flip persistent and one transient: the replay has a single (corrected) error, no interrupt, resolved, exit 0",
      b is not None and "T replay resolved" in b and len(ln(p, "T irq")) == 1 and len([l for l in b.splitlines() if l.startswith("T ecc") and " corr " in l]) >= 1 and GOOD_W in b, p.stdout[-500:])
# 6. a run with no interrupt is unchanged by replay.
p = go([], ["--on-irq", "replay"])
check("replay: no interrupt: no replay, the output of the plain run, exit 0", p.stdout == clean.stdout and "T replay" not in p.stdout)
p = go([fsram(0, 0, 1, 0, 5)], ["--on-irq", "replay"])
check("replay: a corrected error raises no interrupt and starts no replay", "T replay" not in p.stdout and not ln(p, "T irq"))

# 7. Bad input.
for opts, what in [(["--on-irq", "foo"], "a bad policy"), (["--on-irq"], "a missing policy"), (["--on-irq", "stop", "--chips", "2"], "stop with --chips 2"), (["--on-irq", "replay", "--chips", "2"], "replay with --chips 2")]:
    p = subprocess.run([SIM, CFG, *opts], input="", capture_output=True, text=True, timeout=60)
    check(f"replay: {what} is refused with exit 2 and a message", p.returncode == 2 and p.stderr.strip() != "" and p.stdout == "", (p.returncode, p.stderr))
for bad in ["F sram 0 0 1 0 5 x", "F sram 0 0 1 0 5 t t"]:
    p = subprocess.run([SIM, CFG, "--on-irq", "replay"], input=bad + "\n", capture_output=True, text=True, timeout=60)
    check(f"replay: the bad F line {bad!r} is refused with exit 2 and a message", p.returncode == 2 and p.stderr.strip() != "" and p.stdout == "", (p.returncode, p.stderr))

print("ALL PASS" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
