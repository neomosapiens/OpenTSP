"""Architecture prose agrees with the implemented CSR, timing and diagnostic contracts."""
import json
import os
import re
import sys
from pathlib import Path

root = Path(os.environ.get("OPENTSP_DOCS_ROOT") or Path(__file__).resolve().parents[2])
spec = root / "opentsp-arch-spec/docs"
micro = (spec / "microarchitecture.md").read_text()
platform = (spec / "platform.md").read_text()
bring = (spec / "bring-up.md").read_text()
encoding = (spec / "encoding.md").read_text()
validation = (spec / "validation.md").read_text()
csr = (root / "opentsp-arch-spec/model/core/csr.sail").read_text()
ref = json.loads((root / "opentsp-arch-spec/config/ref.json").read_text())

failures = []



def check(item, ok, message):
    if not ok:
        failures.append(f"{item}: {message}")


def section(text, heading):
    """The text of the section that starts at the line `heading`, up to the next heading of the same or a higher level."""
    m = re.search(r"^(#+) " + re.escape(heading) + r".*$", text, re.M)
    if not m:
        return ""
    level = len(m.group(1))
    rest = text[m.end():]
    n = re.search(r"^#{1,%d} " % level, rest, re.M)
    return rest[: n.start()] if n else rest


table = [l for l in micro.splitlines() if l.startswith("| 0x")]
check("link CSR addresses", any(l.startswith("| 0x500 + (l - 16) |") and "`LINK_CORR[l]`" in l for l in table), "no CSR row for LINK_CORR above link 15 at 0x500")
check("link CSR addresses", any(l.startswith("| 0x600 + (l - 16) |") and "`LINK_UNCORR[l]`" in l for l in table), "no CSR row for LINK_UNCORR above link 15 at 0x600")
for row, name in (("0x200 + l", "LINK_CORR"), ("0x210 + l", "LINK_UNCORR")):
    r = [l for l in table if l.startswith(f"| {row} |")]
    check("link CSR addresses", len(r) == 1 and "min(`c2c.links`, 16)" in r[0], f"the {name} row {row} must stop at link 15 (it would overlap the next block)")
check("link CSR addresses", "link 22..16" in csr and "link 19..16" not in csr, "core/csr.sail: the link field of LOG word 1 is bits 22..16")
check("link CSR addresses", "bits 22..16, 7 bits" in micro, "microarchitecture.md: the link field of LOG word 1 has no stated width")
check("link CSR addresses", "0x500 + (l - 16)" in micro.split("**Outcome per flit**")[1], "section 17 names only the 0x200 + l address")

check("encoding round trip", 'validation.md`, section "Encoding round trip"' in encoding, "encoding.md has no pointer to the argument of validation.md")
check("encoding round trip", "**Encoding round trip:**" in validation, "validation.md lost the Encoding round trip argument")

entries = [o["entry"] for o in ref["timing"]["overrides"]]
placeholders = micro.split("**`ref` is a scheduling reference.**")[1].split("`config/test/ref_hw.json`")[0]
check("reference timing overrides", "class rows of `ref` for Exp" not in micro, "microarchitecture.md calls the override rows of ref 'class rows'")
for e in entries:
    check("reference timing overrides", e in placeholders, f"the override entry {e} of ref.json is not named among the placeholders")
check("reference timing overrides", "config/test/ref_hw.json" in platform and "config/test/ref_hw.json" in micro, "ref_hw.json is named in no specification document")


check("reported address width", "**The address is the low 13 bits** on every profile" in micro, "section 15.5 does not say how many address bits an RTL report carries")

fm = [l for l in micro.splitlines() if l.startswith("| `F mxm ")]
check("matrix fault timing", len(fm) == 1 and "fp8_rate" in fm[0] and "first of them" in fm[0], "the F mxm line does not say which of two FP8 ABCs of one cycle it binds")

check("reset state", "| Spare-bit report array (§15.5) |" in micro and "256 slots" in micro, "the reset table has no spare-report array row")
heads = [m.group(1) for m in re.finditer(r"^### (16\.\d) ", micro, re.M)]
check("harness CLI", heads == ["16.1", "16.2", "16.3"], f"sections of 16 are in the order {heads}")
usage = [l for l in micro.splitlines() if l.startswith("`tsp_sim <config.json>")]
check("harness CLI", len(usage) == 1 and "--observe" in usage[0] and "--lint" in usage[0], "the tsp_sim usage line omits --observe or --lint")
s163 = section(micro, "16.3 The diagnostic log")
check("diagnostic conformance", "A code of class `schedule` or `fault` is not logged" not in s163, "section 16.3 says that no schedule or fault code is logged")
check("diagnostic conformance", "admits a superset of the codes of class `data`" in s163, "section 16.3 lost the superset statement")
check("diagnostic conformance", "no `schedule` code and no `data` code but the end-of-run record IQ-004" in platform, "platform.md 7.1(a) well-formed definition")
check("diagnostic cascade", "After a `data` record, though, the cascade rule makes an RTL reproduce the effect of a later `schedule` code" in micro, "the DIR-001 note does not state the cascade rule")
check("host input timing", "a testbench applies it at the real cycle c" in platform and "real cycle c + 1 + K at superlane 0" in platform, "the host-input mapping lacks the real cycle of a write or of the doorbell release")
check("type-specific timing", "type-granular on request" in micro, "no per-type timing rows")
s91 = section(platform, "9.1 Chip CSR registers and the platform registers")
check("platform registers", "`ERR_FIRST` (a read-only register" in s91 and "(`ID`, `SPEC_VERSION`, `PROFILE_DIGEST`" in s91, "section 9.1 does not map the identification registers and ERR_FIRST")
check("platform registers", "of the wrapper" not in bring.split("\n## 2")[0], "bring-up.md still calls the identification registers 'of the wrapper'")
check("bring-up coverage", "## 3. " in bring and "## 4. Tap-to-trace map for co-simulation" in bring, "bring-up.md lacks the coverage plan or the tap-to-trace map")

if failures:
    print("FAIL:\n  " + "\n  ".join(failures))
    sys.exit(1)
print("ok: specification contract checks")
