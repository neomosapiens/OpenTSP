#!/usr/bin/env python3
"""the documents say which tool prints the `mxm.rows` over-vlanes advisory, and that tool does. derive_profile.py does not
print it (it is printed by schema_invariants.py --check only), so no document may name derive_profile.py as a printer."""
import os, re, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
DOCS = ["opentsp-arch-spec/docs/assembly.md", "opentsp-arch-spec/docs/porting-guide.md"]
fails = []
PRINTER = re.compile(r"(?:printed by|advisory from)(?:(?!not |\. |;)[^\n])*?derive_profile")  # a sentence that names it as a printer
for d in DOCS:
    text = open(os.path.join(ROOT, d), encoding="utf-8").read()
    for m in PRINTER.finditer(text):
        fails.append(f"{d}: names derive_profile.py as a printer of the advisory: {text[m.start():m.end() + 20]!r}")
# behaviour: a profile with mxm.rows above vlanes gets no advisory from derive_profile.py
with tempfile.TemporaryDirectory() as t:
    out = os.path.join(t, "o.json")
    p = subprocess.run([sys.executable, os.path.join(SPEC, "tools", "perf", "derive_profile.py"), "--superlanes", "1", "--rows", "320",
                        "--slices-per-hemisphere", "4", "--out", out], cwd=SPEC, capture_output=True, text=True)
    if "advisory" in (p.stdout + p.stderr).lower():
        fails.append("derive_profile.py prints an advisory, the documents say it does not")
if fails:
    print("FAIL:\n  " + "\n  ".join(fails)); sys.exit(1)
print("mxm.rows advisory: the documents name the printer correctly: all pass")
