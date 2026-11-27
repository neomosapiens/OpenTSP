"""Directed end-to-end tests of the transient (`t`) and persistent (`p`) marks on the `F mxm` and `F link` lines under
`--on-irq replay` (microarchitecture.md sec. 16.2, 17, 18): the replay run applies the lines marked `p` again and drops the
others, like the `F sram`, `F srf` and `F iq` lines. The programs are those of mxm_check.py and link_fec.py (one chip, every link
loops back). Runs against the Sail emulator or the ISS. Profiles with 44 MEM slices per hemisphere (the matrix programs need them).
usage: ras_replay_marks.py <tsp_sim|tsp-iss> <config.json>"""
import ast, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
SIM, CFG = sys.argv[1], sys.argv[2]
ok = True


def check(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info); ok &= bool(cond)


def load(name):
    """The definitions and constants of a directed test, without its checks: top-level imports, functions and assignments that do not run the program."""
    path = os.path.join(HERE, name)
    src = open(path).read()
    tree = ast.parse(src)
    saved = sys.argv
    sys.argv = [name, SIM, CFG]
    ns = {"__name__": name, "__file__": path}
    try:
        for n in tree.body:
            seg = ast.get_source_segment(src, n) or ""
            one = ast.Module(body=[n], type_ignores=[])
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                exec(compile(one, path, "exec"), ns)
            elif isinstance(n, ast.Assign) and "run(" not in seg and "check(" not in seg:
                try:
                    exec(compile(one, path, "exec"), ns)
                except NameError:           # a value computed from the result of a run: not needed here
                    pass
    finally:
        sys.argv = saved
    return ns


def go(lines, opts=(), expect=0):
    p = subprocess.run([SIM, CFG, "--trace", *opts], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    assert p.returncode == expect, (opts, p.returncode, p.stderr, p.stdout[-400:])
    return p


def ln(p, prefix): return [l for l in p.stdout.splitlines() if l.startswith(prefix)]


def fsram(cycle, slice_, addr, bit, tp=""): return f"F sram {cycle:x} {slice_:x} {addr:x} 0 {bit:x}" + (" " + tp if tp else "")
def dbl(cycle, slice_, addr, tp="t"): return [fsram(cycle, slice_, addr, 5, tp), fsram(cycle, slice_, addr, 9, tp)]    # two bits of a stored word: uncorrectable at its consumer
def second(p): return p.stdout.split("T replay start\n", 1)[1]


MX = load("mxm_check.py")
sram, recs, dumps = MX["prog"](0, *MX["i8_case"]()[:2])
EXEC = MX["EXEC"]
mx = lambda extra: MX["text"](recs, sram, dumps, extra)
fm = MX["fm"]
TRIG_MX = dbl(0, 44 + 4 * (15 % 11) + 15 // 11, 3 + 15)        # weight row 15 read uncorrectable by the IW: the error interrupt
LANE = lambda tp: fm(EXEC, 0, 1, 5) + tp                       # one wrong lane: corrected by the sum check, `T ecc ... mxm corr plane 0 lane 1`

p = go(mx(TRIG_MX + [LANE("")]))
check("replay marks: F mxm: reference run: the interrupt is raised and the lane is corrected", len(ln(p, "T irq")) == 1 and "mxm corr plane 0 lane 1" in p.stdout, (ln(p, "T irq"), ln(p, "T ecc")))
p = go(mx(TRIG_MX + [LANE(" p")]), ["--on-irq", "replay"])
check("replay marks: F mxm p: the lane fault comes again in the replay (its corrected line is in the second run)", "mxm corr plane 0 lane 1" in second(p), second(p)[:300])
p = go(mx(TRIG_MX + [LANE(" t")]), ["--on-irq", "replay"])
check("replay marks: F mxm t: the lane fault is gone in the replay: no corrected line in the second run, resolved", "mxm corr" not in second(p) and p.stdout.endswith("T replay resolved\n"), second(p)[:300])
p = go(mx(TRIG_MX + [LANE("")]), ["--on-irq", "replay"])
check("replay marks: F mxm with no mark is transient (the default)", "mxm corr" not in second(p), second(p)[:300])

LK = load("link_fec.py")
T = LK["T"]
TRIG_LK = dbl(LK["TR"] + 40, 44, 50)                              # the received word, stored by the Write, read uncorrectable by the consumer
lk = lambda tp: LK["one_chip"]() + TRIG_LK + [LK["fl"](T, 0, 1, "corr") + tp]
p = go(lk(""))
check("replay marks: F link: reference run: the interrupt is raised and the flit is corrected in flight", len(ln(p, "T irq")) == 1 and any("link 0 corr flit 1" in l for l in ln(p, "T ecc")), (ln(p, "T irq"), ln(p, "T ecc")))
p = go(lk(" p"), ["--on-irq", "replay"])
check("replay marks: F link p: the flit fault comes again in the replay", any("link 0 corr flit 1" in l for l in second(p).splitlines()), second(p)[:300])
p = go(lk(" t"), ["--on-irq", "replay"])
check("replay marks: F link t: the flit fault is gone in the replay, resolved", "link 0 corr" not in second(p) and p.stdout.endswith("T replay resolved\n"), second(p)[:300])
p = go(lk(""), ["--on-irq", "replay"])
check("replay marks: F link with no mark is transient (the default)", "link 0 corr" not in second(p), second(p)[:300])

print("ALL PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
