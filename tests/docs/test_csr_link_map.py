"""the link counters of the CSR table in microarchitecture.md section 14 do not overlap one another or the other rows.

`LINK_CORR[l]` and `LINK_UNCORR[l]` live at 0x200 + l and 0x210 + l for l below 16 and at 0x500 + (l - 16) and 0x600 + (l - 16) from link 16 on
(`core/csr.sail`, at most 128 links). The check reads the register table, expands every row whose address is `0xHHH + l`, `0xHHH + (l - 16)` or
`0xHHH + i` over the bound its Content cell gives (`l < min(16, c2c.links)`, `16 <= l < c2c.links`, with `c2c.links` taken at its maximum of 128), and
fails when two registers share an address, when a link register is missing, or when a row's address differs from the one the model's header comment gives.

Run: python3 tests/docs/test_csr_link_map.py
"""
import os
import re
import sys
from pathlib import Path

root = Path(os.environ.get("OPENTSP_DOCS_ROOT") or Path(__file__).resolve().parents[2])
micro = (root / "opentsp-arch-spec/docs/microarchitecture.md").read_text()
csr = (root / "opentsp-arch-spec/model/core/csr.sail").read_text()
MAX_LINKS = 128

section = micro[micro.index("## 14. The CSR space"):]
section = section[: section.index("\n## 15.")] if "\n## 15." in section else section
rows = [r for r in section.splitlines() if re.match(r"\| 0x[0-9A-Fa-f]+", r)]
assert rows, "no register rows found in section 14"

addresses = {}  # address -> register name
found = set()
for r in rows:
    cells = [c.strip() for c in r.strip().strip("|").split("|")]
    addr, name, content = cells[0], cells[1], cells[3] if len(cells) > 3 else ""
    m = re.fullmatch(r"(0x[0-9A-Fa-f]+) \+ (l|\(l - 16\))", addr)
    if m and name.startswith("`LINK_"):
        base = int(m.group(1), 16)
        reg = name.strip("`").split("[")[0]
        shifted = m.group(2) != "l"
        lo = 16 if shifted else 0
        hi = MAX_LINKS if shifted else 16
        if not shifted and not re.search(r"min\(16|min\(`c2c\.links`, 16\)", content):
            hi = MAX_LINKS  # the row says l < c2c.links: read at its maximum, it runs into the next block
        for l in range(lo, hi):
            a = base + (l - 16 if shifted else l)
            if a in addresses:
                print(f"FAIL: {reg}[{l}] at {a:#x} is also {addresses[a]}")
                sys.exit(1)
            addresses[a] = f"{reg}[{l}]"
        found.add((reg, shifted))
    elif addr.startswith("0x") and "+" not in addr and " to " not in addr:
        a = int(addr, 16)
        if a in addresses:
            print(f"FAIL: {name} at {a:#x} is also {addresses[a]}")
            sys.exit(1)
        addresses[a] = name

for reg in ("LINK_CORR", "LINK_UNCORR"):
    for shifted in (False, True):
        if (reg, shifted) not in found:
            print(f"FAIL: section 14 has no row for {reg}{'[l >= 16]' if shifted else '[l < 16]'}")
            sys.exit(1)

# the model's header comment gives the same bases
for want in ("0x200 + l LINK_CORR[l]", "0x210 + l LINK_UNCORR[l]", "0x500 + (l - 16) LINK_CORR[l], 0x600 + (l - 16) LINK_UNCORR[l]"):
    flat = re.sub(r"\s+", " ", re.sub(r"//\s*", " ", csr))
    if want not in flat:
        print(f"FAIL: core/csr.sail does not say {want!r}")
        sys.exit(1)
print(f"csr link map: all pass ({len(addresses)} addresses, no overlap)")
