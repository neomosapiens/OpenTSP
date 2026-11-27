#!/usr/bin/env python3
"""under `--config <file>` the extension refusal of the reference assembler names the configuration in force (the file),
not the default profile `ref`. The refusal of ots-asm is checked by its own tests (ots-asm is built by the gates)."""
import os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
TSP_AS = os.path.join(SPEC, "tools", "tsp_as.py")
CFG = os.path.join("config", "test", "ref_ext.json")  # declares no bf16: the bf16 operand is refused

p = subprocess.run([sys.executable, TSP_AS, "insn", "--config", CFG, "abc.bf16 W:24"], cwd=SPEC, capture_output=True, text=True)
fails = []
if p.returncode != 1:
    fails.append(f"exit {p.returncode}, want 1")
if "does not declare" not in p.stderr:
    fails.append("no 'does not declare' refusal in stderr")
if "ref_ext.json" not in p.stderr:
    fails.append("the refusal does not name the configuration file")
if "(profile ref)" in p.stderr:
    fails.append("the refusal names the default profile ref: " + p.stderr.strip()[:200])
if fails:
    print("FAIL:\n  " + "\n  ".join(fails))
    sys.exit(1)
print("refusal names the configuration: all pass")
