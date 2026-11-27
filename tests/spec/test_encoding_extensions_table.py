"""encoding.json extensions table names every extension the catalogue uses.

Every key of any instruction variable's `extension_codes`, and every name the model
accepts in a profile's `extensions` list, must appear in encoding.json's top-level
`extensions` table.
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
enc = json.loads((root / "opentsp-arch-spec/isa-db/generated/encoding.json").read_text())
table = set(enc["extensions"])

used = set()
for ins in enc["instructions"]:
    for var in ins.get("variables", []):
        used.update(var.get("extension_codes", {}))

missing = sorted(used - table)
assert not missing, (
    "encoding.json extensions table omits extensions that extension_codes refer to: "
    + ", ".join(missing)
)

tsp_as = (root / "opentsp-arch-spec/tools/tsp_as.py").read_text()
m = re.search(r'_EXTENSIONS\s*=\s*\(([^)]*)\)', tsp_as)
assert m, "tsp_as.py _EXTENSIONS not found"
known = {s.strip().strip('"').strip("'") for s in m.group(1).split(",") if s.strip().strip('"').strip("'")}
assert known, "_EXTENSIONS empty"
missing_known = sorted(known - table)
assert not missing_known, (
    "encoding.json extensions table omits known extensions of this version: "
    + ", ".join(missing_known)
)
extra = sorted(table - known)
assert not extra, (
    "encoding.json extensions table lists names that are not extensions of this version: "
    + ", ".join(extra)
)
