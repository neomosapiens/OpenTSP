"""assembly.md says an ACC index at or above the profile's depth is OTS-E-OPR-001, as the assemblers refuse it.

The text must say "at or above" (the index is legal below `mxm.acc_depth`), and the reference assembler must refuse exactly the index equal to the
depth of `ref` (2) and of `mini` and accept the one below it.

Run: python3 tests/docs/test_assembly_acc_index.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

root = Path(os.environ.get("OPENTSP_DOCS_ROOT") or Path(__file__).resolve().parents[2])
text = (root / "opentsp-arch-spec/docs/assembly.md").read_text()
row = next((l for l in text.splitlines() if l.startswith("| `acc_mode`, `acc_index`")), None)
if row is None:
    print("FAIL: assembly.md has no acc_mode/acc_index row")
    sys.exit(1)
if not re.search(r"an index at or above the profile's depth is OTS-E-OPR-001", row):
    print("FAIL: assembly.md does not say that an index at or above the profile's depth is OTS-E-OPR-001: " + row)
    sys.exit(1)

tool = root / "opentsp-arch-spec/tools/tsp_as.py"
for profile in ("ref", "mini"):
    depth = json.loads((root / f"opentsp-arch-spec/config/{profile}.json").read_text())["mxm"]["acc_depth"]
    for index, ok in ((depth - 1, True), (depth, False)):
        p = subprocess.run([sys.executable, str(tool), "insn", "--profile", profile, f"acc.i32 W:0, replace, {index}"], capture_output=True, text=True)
        refused = p.returncode != 0 and "OTS-E-OPR-001" in p.stderr
        if refused == ok:
            print(f"FAIL: {profile} (acc_depth {depth}): index {index} {'refused' if refused else 'accepted'}: {p.stdout!r} {p.stderr!r}")
            sys.exit(1)
print("acc index text: all pass")
