#!/usr/bin/env python3
"""The model has no 128-bit instruction.

No catalogue entry is 128 bits, and the text decode builds only 32-bit and 64-bit words.
A mapping, a constructor or a query for a 128-bit word is dead and must not come back.
"""
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[2]
needles = ("encdec128", "ILLEGAL128", "isa_disasm128", "isa_roundtrip128", "illegal128")
bad = []
for path in (root / "opentsp-arch-spec/model").rglob("*.sail"):
    text = path.read_text()
    for needle in needles:
        if needle in text:
            bad.append(f"{path.relative_to(root)} contains {needle}")
encoding = (root / "opentsp-arch-spec/docs/encoding.md").read_text()
if "encdec128" in encoding or "isa_disasm128" in encoding:
    bad.append("encoding.md still names a 128-bit mapping")
if "there are no 128-bit instructions" not in encoding:
    bad.append("encoding.md does not say there are no 128-bit instructions")
check = (root / "opentsp-arch-spec/tools/isa_db_check.sh").read_text()
if "disasm128" in check or "roundtrip128" in check:
    bad.append("isa_db_check.sh still preserves a 128-bit query")
if bad:
    print("\n".join(bad))
    sys.exit(1)
print("no 128-bit instruction mapping")
