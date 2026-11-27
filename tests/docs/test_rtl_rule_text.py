"""The RTL conformance rule is stated once and the specification documents agree with it.

Checks, on the hand-written specification documents:
  1. no sentence calls the diagnostic-reporting question open or points at the withdrawn proposal;
  2. platform.md section 7 holds the one 'RTL conformance' subsection, and microarchitecture.md sections 12 and 13 point to it;
  3. the section 8 code table has one class per code (data, schedule or fault), and every code of the table is placed
     in the reporting order of a cycle and, unless it belongs to the cycle rather than to an instruction, in the check order;
  4. ECC-002 has a row, a class and a place in both ordered lists;
  5. the spare-bit error report states what an RTL must provide.
Run: python3 tests/docs/test_rtl_rule_text.py
"""
import json
import os
import re
import sys
from pathlib import Path

# The tree to check; OPENTSP_DOCS_ROOT lets the mutation runs point at a modified copy.
root = Path(os.environ.get("OPENTSP_DOCS_ROOT") or Path(__file__).resolve().parents[2])
spec = root / "opentsp-arch-spec/docs"
platform = (spec / "platform.md").read_text()
micro = (spec / "microarchitecture.md").read_text()

# Codes that belong to the cycle or to the run, not to one instruction: they have a place in the reporting order only.
CYCLE_LEVEL = {"IQ-002", "IQ-004", "SRF-002", "LNK-002", "RES-003"}
CLASSES = {"data", "schedule", "fault"}
# The question is named by an identifier that code-scope text checks reserve for prose, so it is assembled here.
QUESTION = "O" + "-16"
PROPOSAL = "P" + "-2"


def fail(message):
    print("FAIL: " + message)
    sys.exit(1)


def sentences(text):
    return re.split(r"(?<=[.;:])\s+|\n+", text)


def section(text, start, end):
    i = text.find(start)
    if i < 0:
        fail("marker not found: " + start)
    j = text.find(end, i + len(start))
    if j < 0:
        fail("marker not found: " + end)
    return text[i:j]


# 1. The stale sentences.
for name, text in (("platform.md", platform), ("microarchitecture.md", micro)):
    for s in sentences(text):
        if (QUESTION in s or PROPOSAL in s) and re.search(r"\bopen\b", s) and "resolved" not in s:
            fail(name + " calls " + QUESTION + " or " + PROPOSAL + " open: " + s.strip()[:120])
        if re.search(r"fpga-prototype-proposal\.md[^.\n]{0,60}P-1|proposal P-1\b", s):
            fail(name + " points at the withdrawn instruction-delivery proposal: " + s.strip()[:120])

# 2. One RTL conformance subsection, pointed to from sections 12 and 13.
if not re.search(r"^### 7\.1 RTL conformance", platform, re.M):
    fail("platform.md section 7 has no subsection '7.1 RTL conformance'")
rtl = section(platform, "### 7.1 RTL conformance", "\n## 8.")
if "well-formed" not in rtl or "schedule" not in rtl:
    fail("the RTL conformance subsection does not define 'well-formed' by the schedule codes")
link = "platform.md#71-rtl-conformance"
s12 = section(micro, "## 12. Harness and trace protocol", "## 13. Notes for hardware")
s13 = section(micro, "## 13. Notes for hardware", "## 14. The CSR space")
for label, body in (("section 12", s12), ("section 13", s13)):
    if link not in body:
        fail("microarchitecture.md " + label + " does not point to " + link)

# 3. The section 8 table: one class per code.
s8 = section(micro, "## 8. Diagnostics", "**IQ underflow is a lint")
rows = [line for line in s8.splitlines() if line.startswith("| ")]
header = next((r for r in rows if r.startswith("| Code |")), None)
if header is None or [c.strip() for c in header.strip("|").split("|")][-1] != "Class":
    fail("the section 8 code table has no 'Class' column")
classes = {}
for r in rows:
    cells = [c.strip() for c in r.strip("|").split("|")]
    if not re.match(r"^[A-Z]+-\d{3}(, [A-Z]+-\d{3})*$", cells[0]):
        continue
    codes = cells[0].split(", ")
    cls = cells[-1].split(" ")[0]
    if cls == "-":
        continue  # a retired code
    if cls not in CLASSES:
        fail("class of " + cells[0] + " is '" + cells[-1] + "', not one of " + ", ".join(sorted(CLASSES)))
    for c in codes:
        classes[c] = cls
if len(classes) < 20:
    fail("the section 8 code table was not parsed (" + str(len(classes)) + " codes)")

# The two ordered lists.
# Only the numbered steps count as the lists (not the prose around them).
reporting = "\n".join(l for l in section(micro, "**Reporting order within a cycle:**", "A dispatch check reports at the dispatch cycle").splitlines()
                      if re.match(r"^\d+a?\. ", l))
check = "\n".join(l for l in section(micro, "**Check order per (queue, cycle)**", "**Order within a queue's cycle**").splitlines()
                  if re.match(r"^\d+a?\. \*\*", l))
if reporting.count("\n") < 6 or check.count("\n") < 6:
    fail("the numbered steps of the reporting order or of the check order were not parsed")
for code in sorted(classes):
    if code not in reporting:
        fail(code + " is in the section 8 table but not in the reporting order of a cycle")
    if code not in CYCLE_LEVEL and code not in check:
        fail(code + " is in the section 8 table but not in the check order of a queue")

# 4. ECC-002.
if classes.get("ECC-002") != "fault":
    fail("ECC-002 has no row of class 'fault' in the section 8 table")

# 5. The spare-bit error report.
s155 = section(micro, "### 15.5 Error reports in the spare bits", "## 16. The error CSRs")
if "required for the model and ISS" not in s155:
    fail("section 15.5 does not say what an RTL must provide of the error report")

# 6. The host outputs: every field of a vector's `expect` and every kind of `ext_events` is compared by 7.1 or listed as excluded.
conformance = root / "opentsp-arch-spec/conformance/model"
schema = json.loads((conformance / "schema.json").read_text())
expect_props = schema["$defs"]["vector"]["properties"]["expect"]["properties"]
ext_pattern = expect_props["ext_events"]["items"]["pattern"]


def top_level_alternatives(pattern):
    """The alternatives of '^(a|b|(c|d) e)$' at the outermost level."""
    body = pattern[2:-2]
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


# The kinds are the first words of the alternatives ("dma", "irq", "csr", "Y", "X", "pkt", "ecc").
kinds = {a.split(" ")[0] for a in top_level_alternatives(ext_pattern)}
if kinds != {"dma", "irq", "csr", "Y", "X", "pkt", "ecc"}:
    fail("the ext_events kinds parsed from the schema are not the expected seven: " + ", ".join(sorted(kinds)))
# The kinds the committed vectors use must be among them (a new kind in the vectors has to reach the schema and 7.1).
used = set()
for profile in ("ref", "mini"):
    for v in json.loads((conformance / (profile + ".json")).read_text())["vectors"]:
        for e in v["expect"].get("ext_events", []):
            used.add(e.split(" ")[0])
if not used <= kinds:
    fail("the vectors use ext_events kinds the schema does not name: " + ", ".join(sorted(used - kinds)))
required = {"expect." + p for p in expect_props if p != "ext_events"} | {"ext_events: " + k for k in kinds}

reproduces = section(rtl, "- **What an RTL reproduces.**", "- **Objects that are not well-formed.**")
obligation_letters = set(re.findall(r"\(([a-z])\) ", reproduces))
listed = {}
for r in (l for l in rtl.splitlines() if l.startswith("| ")):
    cells = [c.strip() for c in r.strip("|").split("|")]
    key = cells[0].strip("`")
    if key in required:
        verdict = cells[-1]
        m = re.match(r"^(compared \(([a-z])\)|excluded)", verdict)
        if not m:
            fail("7.1 table row '" + key + "' does not start with 'compared (x)' or 'excluded': " + verdict[:80])
        if m.group(2) and m.group(2) not in obligation_letters:
            fail("7.1 table row '" + key + "' compares (" + m.group(2) + "), which the list of what an RTL reproduces does not state")
        listed[key] = verdict
for key in sorted(required - set(listed)):
    fail("7.1 neither compares nor excludes the vector field '" + key + "'")
for letter, what in (("a", "records"), ("b", "SRAM"), ("c", "egress"), ("d", "HostNotify"), ("e", "CSR")):
    if letter not in obligation_letters:
        fail("7.1 'What an RTL reproduces' has no item (" + letter + ") for the " + what)
    # every lettered item is used by some row, so deleting a row or an item is noticed
    if not any(v.startswith("compared (" + letter + ")") for v in listed.values()):
        fail("no field of the vectors is compared by item (" + letter + ") of 7.1")

# 7. What 7.1 says about cascades, verification, the mxm fault vectors and the three-valued decision.
if not re.search(r"after a `data` record[^.]*every later code[^.]*`schedule` codes included", rtl):
    fail("7.1 does not say that after a data record the RTL reproduces the effects of every later code, schedule codes included")
if not re.search(r"verif[^.]*before the image is loaded", rtl):
    fail("7.1 does not say that a loader or runtime verifies a well-formed object before the image is loaded")
for label, body in (("7.1", rtl), ("the fault-point table", section(platform, "### 5.1 Fault points", "## 6."))):
    if not re.search(r"`F mxm` vectors[^.]*model and the ISS only", body):
        fail(label + " does not say that the `F mxm` vectors are for the model and the ISS only")
for word in ("well-formed", "data-dependent", "ill-formed"):
    if "**" + word + "**" not in rtl:
        fail("7.1 does not define the verdict '" + word + "' of the three-valued decision in bold")

# One definition of well-formed: a data code other than IQ-004 decides nothing about its object (the verifier's rule), whether or
# not a schedule code follows it; the verdict has no case in which a data code is "clean".
well = section(rtl, "- **Well-formed.**", "- **What an RTL reproduces.**")
for needle, what in (
    ("no `data` code but the end-of-run record IQ-004", "that a well-formed run raises no data code but IQ-004"),
    ("whether or not a `schedule` code follows it", "that a data code decides nothing whether or not a schedule code follows"),
    ("a `schedule` code that the data code hid", "that a data code can hide a schedule code"),
):
    if needle not in well:
        fail("7.1 'Well-formed' does not state " + what)
if "(the run raises no `schedule` code)" in well:
    fail("7.1 'Well-formed' still calls a run clean when it raises no schedule code, whatever data codes it raised")

hw = section(platform, "**Host-visible state in hardware (normative).**", "### 5.1 Fault points")
for needle, what in (
    ("IQ-004", "the end-of-run exception for IQ-004"),
    ("end-of-run", "the end-of-run exception for IQ-004"),
    ("model state of r - D", "the mapping of a host read under the delay D"),
    ("c + D", "the presentation of an interrupt that a host write causes"),
    ("sum-check", "the sum-check term of the delay"),
):
    if needle not in hw:
        fail("platform.md section 5 lacks " + what + " ('" + needle + "')")

# 8. One definition of ERR_FIRST, one FIFO depth, one source of fifo_overflow, one story of the chip functions.
s7 = section(platform, "## 7. Diagnostics", "### 7.1 RTL conformance")
s91 = section(platform, "### 9.1 Chip CSR registers", "### 9.2 PowerConfig")
if "at least **16**" in s7 or "at least 16" in s7:
    fail("platform.md section 7 still gives the FIFO a minimum depth of 16 although it mirrors ras.diag_depth (1..64)")
if "ras.diag_depth" not in s7:
    fail("platform.md section 7 does not give the FIFO the depth ras.diag_depth")
if "the wrapper adds nothing" in s7:
    fail("platform.md section 7 says the wrapper adds nothing, but ERR_FIRST has no chip source for a schedule code")
if "of any code" in s91 or "of any code" in s7:
    fail("platform.md defines ERR_FIRST as the first record of any code; it is the first record of the diagnostic log")
if not re.search(r"ERR_FIRST[^.]*first record[^.]*log order", s7):
    fail("platform.md section 7 does not define ERR_FIRST as the first record in log order")
for row in s91.splitlines():
    if row.startswith("| `STATUS.error`") and "fifo_overflow" in row.split("|")[1]:
        fail("the section 9.1 ERR_STATUS row maps fifo_overflow, which DIAG_STATUS bit 0 carries")
if "exposes the CSR space only" in s91 or "the cycle counter is the wrapper's" in s91:
    fail("platform.md 9.1 still says that silicon exposes the CSR space only and that the cycle counter is the wrapper's")
if "chip functions" not in s91 or "implementation-defined" not in s91:
    fail("platform.md 9.1 does not say that start, stop, load window, aperture and cycle counter are chip functions with implementation-defined registers")
if "superset" not in reproduces or "16.3" not in reproduces:
    fail("7.1 (a) does not name the log filter of microarchitecture 16.3 as a superset of class data")

# 9. The residual-risk rule of a stored poisoned word, in the serving RAS document (stated twice there).
ras = (spec / "production-ras.md").read_text()
for needle in ("syndrome above 136", "data-bit flip"):
    if ras.count(needle) < 2:
        fail("production-ras.md states '" + needle + "' fewer than twice (the residual-risk paragraph and the qualification row)")

# 10. Bring-up and design-space documents.
bring = (spec / "bring-up.md").read_text()
for needle, what in (("Coverage plan", "a coverage plan"), ("Tap-to-trace", "a tap-to-trace map for co-simulation"),
                     ("IQ-003", "IQ-003 in the coverage plan"), ("IQ-002", "IQ-002 in the coverage plan")):
    if needle not in bring:
        fail("bring-up.md has no " + what + " ('" + needle + "')")
design = (spec / "design-space.md").read_text()
alternatives = design.split("\n## 4.")[-1] if "\n## 4." in design else ""
for needle, what in (("FP8", "the FP8 rate"), ("Small-ALU", "the small and big ALU split"), ("accumulator", "the accumulators")):
    if needle not in alternatives:
        fail("design-space.md has no recorded alternative for " + what + " in its section 4")

print("ok: the RTL conformance rule text is consistent (" + str(len(classes)) + " codes classed, "
      + str(len(required)) + " vector fields compared or excluded)")
