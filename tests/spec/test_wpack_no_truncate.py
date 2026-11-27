#!/usr/bin/env python3
"""`tsp_as.py wpack` validates the whole package before it opens the output. A refused pack leaves an existing
output file as it was (not truncated to zero bytes) and creates no output file."""
import os, subprocess, sys, tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
TSP_AS = os.path.join(SPEC, "tools", "tsp_as.py")
fails = []
with tempfile.TemporaryDirectory() as d:
    w = os.path.join(d, "w.bin")
    open(w, "wb").write(bytes(320))                      # one ref word (320 vlanes-bytes)
    good = os.path.join(d, "good.safetensors")
    open(good, "wb").write(b"\x01\x02\x03\x04")          # a previous package, to be kept
    cases = {
        "duplicate name": ["t=u8:" + w, "t=u8:" + w],
        "bad dtype": ["t=zz:" + w],
    }
    for label, specs in cases.items():
        p = subprocess.run([sys.executable, TSP_AS, "wpack", "--profile", "ref", "-o", good] + specs, cwd=SPEC, capture_output=True, text=True)
        if p.returncode == 0:
            fails.append(f"{label}: exit 0, want a refusal")
        if open(good, "rb").read() != b"\x01\x02\x03\x04":
            fails.append(f"{label}: the existing output was changed or truncated")
    fresh = os.path.join(d, "new.safetensors")
    p = subprocess.run([sys.executable, TSP_AS, "wpack", "--profile", "ref", "-o", fresh, "t=zz:" + w], cwd=SPEC, capture_output=True, text=True)
    if os.path.exists(fresh):
        fails.append("a refused pack created its output file")
if fails:
    print("FAIL:\n  " + "\n  ".join(fails)); sys.exit(1)
print("wpack refusals keep the output: all pass")

#  structural pin (also covered by the refusal runs above)
import re as _re
_src = open(os.path.join(ROOT, "opentsp-arch-spec", "tools", "tsp_as.py"), encoding="utf-8").read()
_m = _re.search(r'elif cmd == "wpack":(.*?)(?:elif cmd|else:)', _src, _re.S)
assert _m, "wpack branch missing"
_body = _m.group(1)
assert _body.find("write_weights(") < _body.find('open(opts["-o"], "wb")'), "write_weights must precede open -o"
print("wpack source order: ok")
