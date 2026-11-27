#!/usr/bin/env python3
"""A chain of stage programs on `tsp_sim --chips N` and on the ISS in its tsp_sim-compatible mode: one program per chip (`gen/ship_layer.py: chain`), the
host's vectors on chip 0, the output tiles of each chip Sent over a link to the next, the last chip's output given back to the host.

usage: chain_run.py [--sim PATH] [--iss PATH] [--chips N] [--layers T] [--chunks C] [--keep DIR] [--timeout S]

The programs are built in memory with `chain`, assembled and packed exactly as run.py does (both assemblers, identical objects and packages), joined into
one harness input (`C <chip>` sections) and run once per simulator. Every chip's run is compared as run.py compares a case (E line, diagnostics, expected
words, the PciSend vectors), and the two simulators' traces chip by chip. The twin is the oracle: the expected words of every stage and the host's output
are the stage twins in sequence. Exit 1 on any mismatch."""
import argparse
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "gen"))
import run as R  # noqa: E402
from opentsp_tools import profile as tprofile
from opentsp_vllm.models.gpt2 import layer as ship_layer


def split_chips(text, n):
    """The output of a run per chip (`C <chip>` lines precede each chip's lines)."""
    parts, cur = [], None
    for ln in text.split("\n"):
        if ln.startswith("C "):
            cur = []
            parts.append(cur)
        elif cur is not None:
            cur.append(ln)
    assert len(parts) == n, (len(parts), n)
    return ["\n".join(x) for x in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=os.environ.get("TSP_SIM"))
    ap.add_argument("--iss", default=os.environ.get("TSP_ISS", os.path.join(R.ROOT, "target", "release", "tsp-iss")))
    ap.add_argument("--asm", default=os.environ.get("OTS_ASM", os.path.join(R.ROOT, "target", "release", "ots-asm")))
    ap.add_argument("--chips", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2, help="tiles of 128 lanes: the hidden width is 128 x LAYERS")
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--keep")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    for label, path in (("tsp_sim", a.sim), ("tsp-iss", a.iss), ("ots-asm", a.asm)):
        if not (path and os.path.isfile(path) and os.access(path, os.X_OK)):
            sys.exit(f"chain_run.py: {label} is required and must be executable (got {path!r})")
    a.tag, a.jobs = None, 1
    runner = R.Runner(a)
    p = tprofile.load("ref")
    stages, infos = ship_layer.chain(p, a.chips, a.layers, a.chunks)
    work = tempfile.mkdtemp(prefix="tsp-chain-")
    ok = True
    try:
        texts, cases = [], []
        for i, (ots, case) in enumerate(stages):
            cdir = os.path.join(work, f"c{i}")
            os.makedirs(cdir)
            open(os.path.join(cdir, case["program"]), "w").write(ots)
            text, _ = runner.build_input(case, cdir, cdir)
            texts.append(f"C {i:x}\n" + text)
            cases.append(case)
        budget = max(c["budget"]["cycles"] for c in cases)
        print(f"chain: {a.chips} chips, stage cycles {[inf['end'] for inf in infos]}, budget {budget}", flush=True)
        outs = {}
        for name, exe in (("tsp_sim", a.sim), ("iss", a.iss)):
            r = R.sh([exe, R.config_of("ref"), "--trace", "--lint", "--chips", str(a.chips), "--max-cycles", str(budget)],
                     stdin="".join(texts).encode(), timeout=a.timeout, ok=(0, 3))
            if r.returncode == 3:
                print(f"FAIL chain {name}: --max-cycles {budget} exceeded")
                ok = False
                continue
            open(os.path.join(work, f"out.{name}.txt"), "wb").write(r.stdout)
            per = [R.parse_out(t) for t in split_chips(r.stdout.decode(), a.chips)]
            outs[name] = per
            for i, (c, out) in enumerate(zip(cases, per)):
                try:
                    runner.compare(c, out)
                    print(f"PASS chain {name} chip {i}: {len(c['expect']['words'])} words" + (f", {len(out['egress'])} vectors to the host" if "egress" in c["expect"] else ""))
                except R.Fail as e:
                    print(f"FAIL chain {name} chip {i}: {e}")
                    ok = False
        if len(outs) == 2:
            for i in range(a.chips):
                ta, tb = (R.canonical(outs[n][i]["trace"]) for n in ("tsp_sim", "iss"))
                if ta != tb:
                    k = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                    print(f"FAIL chain trace chip {i}: tsp_sim and iss differ at line {k}")
                    ok = False
                else:
                    print(f"PASS chain trace chip {i} (tsp_sim = iss, {len(ta)} lines)")
    finally:
        if a.keep:
            shutil.rmtree(a.keep, ignore_errors=True)
            shutil.copytree(work, a.keep)
        shutil.rmtree(work, ignore_errors=True)
    print("chain: " + ("all passed" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
