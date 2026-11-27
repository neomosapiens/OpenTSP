"""Check the generated VXM type legend against the tool-facing five-bit code table."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
reference = (root / "opentsp-arch-spec/docs/encoding.md").read_text()
table = json.loads((root / "opentsp-arch-spec/isa-db/generated/encoding.json").read_text())
legend = next(line for line in reference.splitlines() if line.startswith("- VXM element type:"))
assigned = {name: int(code) for name, code in re.findall(r"(\w+)=(\d+)", legend)}
assert assigned == table["eltype_codes"], "VXM type assignments disagree with encoding.json"
sentence = re.search(r"codes ([\d.\sand]+) are reserved", legend)
reserved_codes = set()
for lo, hi in re.findall(r"(\d+)\.\.(\d+)", sentence[1] if sentence else ""):
    reserved_codes |= set(range(int(lo), int(hi) + 1))
assert reserved_codes == set(range(32)) - set(assigned.values()), "VXM reserved range disagrees with assigned codes"
