#!/usr/bin/env python3
"""User-facing RSqrt and cycle-limit text of specification 0.1.

1. The RSqrt catalogue text names the two measured division idioms (the ISA has no divide), their domains and the kernel cookbook,
   in the catalogue file, in the generator that writes it and in the generated `isa.md`.
2. The topology chapter states the unsigned 64-bit cycle range and the exit status for overflow.

usage: check_public_contract_text.py"""
import pathlib
import sys

spec = pathlib.Path(__file__).resolve().parents[2]
docs = spec / "docs"
failures = []

for label, path in (("isa-db/instructions/vxm_rsqrt.yaml", spec / "isa-db/instructions/vxm_rsqrt.yaml"), ("isa.md", docs / "isa.md")):
    text = " ".join(path.read_text().split())
    for needle in ("rsqrt(x * x)", "rsqrt(x) * rsqrt(x)", "kernel-cookbook.md", "[2^-63, 2^64)"):
        # isa.md folds the text of one entry into a table row: the needle may sit next to markup
        if needle not in text:
            failures.append(f"{label}: the RSqrt text does not contain {needle!r}")
if "kernel-cookbook.md" not in (spec / "tools/catalogue/gen_catalogue.py").read_text():
    failures.append("tools/catalogue/gen_catalogue.py: the RSqrt note is not in the generator (the catalogue would lose it at the next regeneration)")

for path in sorted(spec.rglob("*")):
    if path.is_file() and path.suffix in {".c", ".h", ".py", ".sh", ".sail", ".json", ".yaml", ".md"} and "CYCLE_LIMIT" in path.read_text(errors="ignore"):
        if path != pathlib.Path(__file__).resolve():
            failures.append(f"{path.relative_to(spec)}: names CYCLE_LIMIT")
row = next((ln for ln in (docs / "topology.md").read_text().splitlines() if ln.startswith("| cycle number in a run")), "")
if "CYCLE_LIMIT" in row or "exit 4" not in row or "2⁶⁴" not in row:
    failures.append("topology.md: the cycle-number row does not state the 64-bit range and exit 4")

for f in failures:
    print("FAIL", f)
print(f"public contract text: {len(failures)} failures")
sys.exit(1 if failures else 0)
