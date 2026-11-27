#!/usr/bin/env python3
"""Set a typed timing override. `derive_profile.py --set timing.overrides.<Entry>.<field>=V` finds the row of an entry with a dot in its name
(`Max.i8`); a row the template does not have is a refusal, not a traceback."""
import json, os, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
DP = os.path.join(SPEC, "tools", "perf", "derive_profile.py")
TPL = os.path.join("config", "test", "ref_hw.json")
fails = []
with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "out.json")
    p = subprocess.run([sys.executable, DP, "--template", TPL, "--set", "timing.overrides.Max.i8.d_func=4", "--out", out],
                       cwd=SPEC, capture_output=True, text=True)
    if p.returncode != 0 or "Traceback" in p.stderr:
        fails.append(f"typed row: exit {p.returncode}: {p.stderr.strip()[-200:]}")
    else:
        rows = {r["entry"]: r for r in json.load(open(out))["timing"]["overrides"]}
        if rows["Max.i8"].get("d_func") != 4:
            fails.append(f"typed row: Max.i8.d_func is {rows['Max.i8'].get('d_func')}, want 4")
    p = subprocess.run([sys.executable, DP, "--template", TPL, "--set", "timing.overrides.Nope.d_func=4", "--out", out],
                       cwd=SPEC, capture_output=True, text=True)
    if p.returncode == 0:
        fails.append("missing row: exit 0, want a refusal")
    if "Traceback" in p.stderr:
        fails.append("missing row: a traceback, want a refusal message")
if fails:
    print("FAIL:\n  " + "\n  ".join(fails)); sys.exit(1)
print("derive_profile --set on typed rows: all pass")
