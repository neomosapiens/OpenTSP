#!/usr/bin/env python3
"""Run the conformance program cases with both assemblers and simulators.

usage: run.py [--profile P ...] [--sim PATH] [--iss PATH] [--asm PATH] [--tag T] [--dir DIR ...] [--jobs N]
       [--keep DIR] [CASE ...]

For each case (a name such as basic/vec_add, or a .case.json path; default: every case under tests/programs/):
  1. assemble with ots-asm and with tsp_as.py; the objects must be byte-identical and the assemblers' warnings
     exactly the case's `expect.asm_warnings`;
  2. pack the case's weights with both assemblers' wpack (byte-identical packages), build the harness input with
     both objinput commands (identical), and append the case's S (preload) and D (dump) lines;
  3. run on tsp_sim (--sim, default $TSP_SIM; cases tagged `sim`) and on the ISS in its tsp_sim-compatible mode
     (--iss, default $TSP_ISS or target/release/tsp-iss when built); a case without preload also runs on the ISS
     natively (`tsp-iss run --config C --weights in.safetensors --dump S:A prog.elf`: the object and the package, the
     path of the end-to-end programs, no harness text);
  4. compare the E line, the T diag list (exact, in order) and the dumped W words (byte for byte); a case with
     `expect.lints` (every run is made with --lint) also the T lint list, [] meaning no lint; with two or more runs
     also the lint lists against each other and the full canonical trace (each cycle's run of T srf lines sorted,
     values included).
A case with `expect.asm_error` instead requires both assemblers to refuse the program with that code (no run).
One line per (case, simulator); exit 1 on any mismatch. The only skip is the ISS run when no ISS is available.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

from opentsp_tools.build import CaseBuilder, Fail, config_of, sh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
TSP_AS = os.path.join(SPEC, "tools", "tsp_as.py")


def vlanes_of(profile):
    with open(config_of(profile)) as f:
        return 16 * json.load(f)["geometry"]["superlanes"]


def parse_out(text):
    lines = text.split("\n")
    trace = [ln for ln in lines if ln.startswith("T ") and not ln.startswith("T lint")]
    e = [ln for ln in lines if ln.startswith("E ")]
    if len(e) != 1:
        raise Fail(f"expected one E line, got {len(e)}")
    lints = [(int(f[2]), int(f[4]), f[5]) for f in (ln.split() for ln in lines if ln.startswith("T lint"))]
    diags = []
    for ln in trace:
        f = ln.split()
        if f[1] == "diag":
            diags.append({"cycle": int(f[2]), "queue": int(f[4]), "code": f[5]})
    words = {}
    locations = []
    egress = []
    for ln in lines:
        if ln.startswith("W "):
            f = ln.split()
            location = f"{int(f[1], 16)}:{int(f[2], 16)}"
            locations.append(location)
            words[location] = bytes(int(x, 16) for x in f[3:])
        elif ln.startswith("Y "):
            f = ln.split()
            egress.append((int(f[1]), bytes(int(x, 16) for x in f[2:])))
    return {"trace": trace, "E": e[0], "diags": diags, "words": words, "locations": locations, "lints": lints, "egress": egress}


def canonical(trace):
    """Each cycle's run of T srf lines sorted into canonical order (microarchitecture.md sec. 12)."""
    out, run = [], []
    for ln in trace + [None]:
        if ln is not None and ln.startswith("T srf"):
            if run and run[0].split()[2] != ln.split()[2]:
                out += sorted(run, key=srf_key)
                run = []
            run.append(ln)
            continue
        out += sorted(run, key=srf_key)
        run = []
        if ln is not None:
            out.append(ln)
    return out


def srf_key(ln):
    f = ln.split()
    row, stream = f[3].split(":")
    return (row, int(stream), int(f[5]))


class Runner(CaseBuilder):
    def __init__(self, a):
        self.a = a

    def refused(self, case, cdir, work):
        """A case with `expect.asm_error`: both assemblers must refuse the program with that code."""
        cfg = ["--config", config_of(case["profile"])]
        prog = os.path.join(cdir, case["program"])
        code = case["expect"]["asm_error"]
        for who, cmd in (("ots-asm", [self.a.asm, "asm", *cfg, "-o", os.path.join(work, "a.elf"), prog]),
                         ("tsp_as.py", [sys.executable, TSP_AS, "asm", prog, "-o", os.path.join(work, "b.elf"), *cfg])):
            r = sh(cmd, ok=(0, 1))
            if r.returncode != 1 or f": {code}: " not in r.stderr.decode():
                raise Fail(f"{who} did not refuse the program with {code} (exit {r.returncode}): "
                           f"{r.stderr.decode().strip()[:200]}")


    def simulate(self, exe, case, text, work, label):
        budget = case["budget"]["cycles"]
        r = sh([exe, config_of(case["profile"]), "--trace", "--lint", "--max-cycles", str(budget)], stdin=text.encode(),
               timeout=self.a.timeout, ok=(0, 3))
        open(os.path.join(work, f"out.{label}.txt"), "wb").write(r.stdout)
        if r.returncode == 3:
            raise Fail(f"--max-cycles {budget} exceeded")
        return parse_out(r.stdout.decode())

    def simulate_native(self, exe, case, native, work):
        """The ISS on the object and the weight package directly (tsp-iss run), with the case's dumps."""
        obj, pkg = native
        cmd = [exe, "run", "--config", config_of(case["profile"]), "--trace", "--lint", "--max-cycles", str(case["budget"]["cycles"])]
        if pkg:
            cmd += ["--weights", pkg]
        for k in case["dump"]:
            s, addr = (int(x) for x in k.split(":"))
            cmd += ["--dump", f"{s:x}:{addr:x}"]
        r = sh(cmd + [obj], timeout=self.a.timeout, ok=(0, 3))
        open(os.path.join(work, "out.iss-native.txt"), "wb").write(r.stdout)
        if r.returncode == 3:
            raise Fail(f"--max-cycles {case['budget']['cycles']} exceeded")
        return parse_out(r.stdout.decode())

    def compare(self, case, out):
        exp = case["expect"]
        if out["diags"] != exp["diagnostics"]:
            raise Fail(f"diagnostics {out['diags'][:6]} != {exp['diagnostics'][:6]}")
        if "lints" in exp and out["lints"] != [tuple(x) for x in exp["lints"]]:
            raise Fail(f"lints {out['lints'][:6]} != {exp['lints'][:6]}")
        d = exp["diagnostics"]
        e_want = f"E {d[0]['code']} {d[0]['cycle']} {len(d)}" if d else "E - -1 0"
        if out["E"] != e_want:
            raise Fail(f"{out['E']!r} != {e_want!r}")
        vl = vlanes_of(case["profile"])
        if out["locations"] != case["dump"]:
            raise Fail(f"dump locations {out['locations']!r} != {case['dump']!r}")
        width = max(320, vl)
        for k in case["dump"]:
            word = out["words"][k]
            if len(word) != width:
                raise Fail(f"W {k}: expected {width} bytes, got {len(word)}")
            if any(word[vl:]):
                raise Fail(f"W {k}: bytes beyond vlanes {vl} are not zero")
        if "egress" in exp:
            got = [v[:vl] for _, v in out["egress"]]
            want = [bytes.fromhex(h) for h in exp["egress"]]
            if got != want:
                bad = next((i for i, (g, w) in enumerate(zip(got, want)) if g != w), min(len(got), len(want)))
                raise Fail(f"PciSend vectors: {len(got)} given, {len(want)} expected, the first difference at {bad}")
        for k, h in exp["words"].items():
            got = out["words"].get(k, b"")[:vl]
            if got != bytes.fromhex(h):
                bad = next(i for i in range(vl) if got[i:i + 1] != bytes.fromhex(h)[i:i + 1])
                raise Fail(f"word {k} lane {bad}: {got[bad:bad + 1].hex()} != {h[2 * bad:2 * bad + 2]}")

    def run_case(self, path):
        case = json.load(open(path))
        cdir = os.path.dirname(path)
        label = f"{case['name']} [{case['profile']}]"
        lines, ok = [], True
        work = tempfile.mkdtemp(prefix="tsp-programs-")
        try:
            if case["expect"].get("asm_error"):
                try:
                    self.refused(case, cdir, work)
                    return [f"PASS {label} assemble: both assemblers refuse it with {case['expect']['asm_error']}"], True
                except (Fail, subprocess.TimeoutExpired) as e:
                    return [f"FAIL {label} assemble: {e}"], False
            try:
                text, native = self.build_input(case, cdir, work)
            except (Fail, subprocess.TimeoutExpired) as e:
                return [f"FAIL {label} assemble: {e}"], False
            outs = {}
            sims = []
            if "sim" in case["tags"]:
                sims.append(("tsp_sim", self.a.sim))
            if self.a.iss:
                sims.append(("iss", self.a.iss))
            else:
                lines.append(f"SKIP {label} iss: ISS unavailable ({self.a.iss_why})")
            runs = [(name, lambda exe=exe, name=name: self.simulate(exe, case, text, work, name))
                    for name, exe in sims]
            if self.a.iss and case.get("host"):
                lines.append(f"SKIP {label} iss-native: the CLI takes no host vectors (the text-input ISS ran them)")
            elif self.a.iss and not case["preload"]:
                runs.append(("iss-native", lambda: self.simulate_native(self.a.iss, case, native, work)))
            for name, run in runs:
                try:
                    outs[name] = run()
                    self.compare(case, outs[name])
                    lines.append(f"PASS {label} {name}")
                except (Fail, subprocess.TimeoutExpired) as e:
                    lines.append(f"FAIL {label} {name}: {e}")
                    ok = False
            if len(outs) >= 2:
                first = next(iter(outs))
                ta = canonical(outs[first]["trace"])
                for other in list(outs)[1:]:
                    tb = canonical(outs[other]["trace"])
                    if outs[first]["lints"] != outs[other]["lints"]:
                        lines.append(f"FAIL {label} lints: {first} {outs[first]['lints'][:4]} and {other} "
                                     f"{outs[other]['lints'][:4]} differ")
                        ok = False
                    if ta != tb:
                        i = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                        lines.append(f"FAIL {label} trace: {first} and {other} differ at line {i}: "
                                     f"{(ta[i:i + 1] or ['<end>'])[0][:80]!r} vs {(tb[i:i + 1] or ['<end>'])[0][:80]!r}")
                        ok = False
                    else:
                        lines.append(f"PASS {label} trace ({first} = {other}, {len(ta)} lines)")
            if not sims:
                lines.append(f"FAIL {label}: no simulator ran (case not tagged sim and no ISS)")
                ok = False
        finally:
            if self.a.keep:
                dst = os.path.join(self.a.keep, case["name"].replace("/", "_") + "." + os.path.basename(case["profile"]))
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(work, dst)
            shutil.rmtree(work, ignore_errors=True)
        return lines, ok


def find_cases(names, dirs=()):
    """The cases under tests/programs/ and under each of `dirs` (a build.py --out directory of on-demand cases)."""
    roots = [HERE] + [os.path.abspath(d) for d in dirs]
    every = sorted(p for r in roots for p in glob.glob(os.path.join(r, "*", "*.case.json")))
    if not names:
        return every
    out = []
    for n in names:
        if n.endswith(".case.json"):
            out.append(os.path.abspath(n))
            continue
        hit = [p for r in roots for p in every
               if p.startswith(r + os.sep) and os.path.relpath(p, r).startswith(n + ".")]
        if not hit:
            sys.exit(f"run.py: no case named {n}")
        out += hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="append")
    ap.add_argument("--sim", default=os.environ.get("TSP_SIM"))
    ap.add_argument("--iss", default=os.environ.get("TSP_ISS"))
    ap.add_argument("--asm", default=os.environ.get("OTS_ASM", os.path.join(ROOT, "target", "release", "ots-asm")))
    ap.add_argument("--tag", action="append")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--dir", action="append", default=[],
                    help="also the cases under DIR/<category>/ (a build.py --out directory of on-demand cases)")
    ap.add_argument("--keep")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per simulator run")
    ap.add_argument("cases", nargs="*")
    a = ap.parse_args()
    if not a.sim or not os.access(a.sim, os.X_OK):
        sys.exit(f"run.py: tsp_sim not found (--sim PATH or $TSP_SIM; got {a.sim!r})")
    if not os.access(a.asm, os.X_OK):
        sys.exit(f"run.py: ots-asm not found at {a.asm} (cargo build --release -p ots-asm, or --asm PATH)")
    default_iss = os.path.join(ROOT, "target", "release", "tsp-iss")
    a.iss_why = ""
    if a.iss and not (os.path.isfile(a.iss) and os.access(a.iss, os.X_OK)):
        sys.exit(f"run.py: requested ISS is not executable: {a.iss}")
    if not a.iss:
        if os.access(default_iss, os.X_OK):
            a.iss = default_iss
        else:
            a.iss_why = "no --iss, $TSP_ISS unset, target/release/tsp-iss not built"
    if a.keep:
        os.makedirs(a.keep, exist_ok=True)
    paths = []
    for p in find_cases(a.cases, a.dir):
        c = json.load(open(p))
        if a.profile and c["profile"] not in a.profile:
            continue
        if a.tag and not set(a.tag) & set(c["tags"]):
            continue
        paths.append(p)
    selection = " ".join([*(f"--profile {p}" for p in a.profile or []), *(f"--tag {t}" for t in a.tag or []), *a.cases]) or "all cases"
    if not paths:
        sys.exit(f"run.py: the selection ({selection}) matched no case")
    print(f"run: {len(paths)} cases selected ({selection})", flush=True)
    r = Runner(a)
    ok_all = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for lines, ok in ex.map(r.run_case, paths):
            for ln in lines:
                print(ln, flush=True)
            ok_all &= ok
    print(f"run: {len(paths)} cases, {'all passed' if ok_all else 'FAILURES'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
