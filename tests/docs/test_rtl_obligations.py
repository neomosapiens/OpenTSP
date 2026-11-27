"""Every golden vector carries its RTL obligation, and bring-up.md says how to use it.

Checks:
  1. conformance/rtl/<profile>.json names every vector of conformance/model/<profile>.json, once, with one of the three values
     (pass, none, undecided), a reason for every value but pass, and the digest of the vector file it was generated from (a
     vector change without a regeneration fails here; the regeneration itself, which needs tsp-iss, is `just iss-rtl-obligations`);
  2. the counts of the file agree with its entries, and with the table of bring-up.md section 2;
  3. a vector that carries an F line (a fault run) is never `pass`, and an `F mxm` vector is `none`;
  4. platform.md 7.1 defines the three values, and bring-up.md says what to do with each.
Run: python3 tests/docs/test_rtl_obligations.py
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

root = Path(os.environ.get("OPENTSP_DOCS_ROOT") or Path(__file__).resolve().parents[2])
spec = root / "opentsp-arch-spec/docs"
conf = root / "opentsp-arch-spec/conformance"
VALUES = ("pass", "none", "undecided")


def fail(message):
    print("FAIL: " + message)
    sys.exit(1)


counts = {}
WITNESS = {"ref": ["directed-239", "directed-262"], "mini": ["directed-256", "directed-313"]}
for profile in ("ref", "mini"):
    raw = (conf / "model" / f"{profile}.json").read_bytes()
    vectors = json.loads(raw)["vectors"]
    side_path = conf / "rtl" / f"{profile}.json"
    if not side_path.exists():
        fail(f"{side_path.relative_to(root)} does not exist: run opentsp-arch-spec/tools/gen_rtl_obligations.py")
    side = json.loads(side_path.read_text())
    if side.get("vectors_sha256") != hashlib.sha256(raw).hexdigest():
        fail(f"{profile}: the obligation file was generated from other vectors than conformance/model/{profile}.json")
    entries = side["vectors"]
    names = [v["name"] for v in vectors]
    if sorted(entries) != sorted(names):
        missing = sorted(set(names) - set(entries))[:3]
        extra = sorted(set(entries) - set(names))[:3]
        fail(f"{profile}: the obligation file and the vectors name different vectors (missing {missing}, extra {extra})")
    for v in vectors:
        e = entries[v["name"]]
        if e["obligation"] not in VALUES:
            fail(f"{v['name']}: obligation '{e['obligation']}' is not one of {VALUES}")
        if e["obligation"] != "pass" and not e["why"]:
            fail(f"{v['name']}: obligation '{e['obligation']}' has no reason")
        faults = [h for h in v.get("host", []) if h.startswith("F ")]
        if faults and e["obligation"] == "pass":
            fail(f"{v['name']}: a fault run cannot be `pass`")
        if any(h.startswith("F mxm ") for h in faults) and e["obligation"] != "none":
            fail(f"{v['name']}: an `F mxm` vector is for the model and the ISS only, so `none`")
    for name, e in entries.items():
        if "witness" in e and e["obligation"] != "undecided":
            fail(f"{profile} {name}: only an `undecided` vector has a witness")
    if sorted(n for n, e in entries.items() if "witness" in e) != sorted(WITNESS[profile]):
        fail(f"{profile}: the SXM-001 cascade vectors {WITNESS[profile]} must carry a witness, and no other vector")
    got = {k: sum(1 for e in entries.values() if e["obligation"] == k) for k in VALUES}
    if got != side["counts"]:
        fail(f"{profile}: the counts of the file {side['counts']} differ from its entries {got}")
    counts[profile] = got
    counts[profile]["vectors"] = len(vectors)

bring = (spec / "bring-up.md").read_text()
for profile, c in counts.items():
    row = re.search(r"^\| `%s` \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|" % profile, bring, re.M)
    if not row:
        fail(f"bring-up.md has no row of obligation counts for `{profile}`")
    doc = tuple(int(x) for x in row.groups())
    want = (c["pass"], c["none"], c["undecided"], c["vectors"])
    if doc != want:
        fail(f"bring-up.md counts for `{profile}` are {doc}, the obligation file has {want}")

platform = (spec / "platform.md").read_text()
for value in VALUES:
    if f"`{value}`" not in platform[platform.index("### 7.1 RTL conformance"):]:
        fail(f"platform.md 7.1 does not define the obligation `{value}`")
order = bring[bring.index("## 2. A suggested order"):bring.index("### The obligation of each vector")]
use = next((l for l in order.splitlines() if l.startswith("4. ")), "")
for word in ("`pass`", "`none`", "`undecided`"):
    if word not in use:
        fail(f"bring-up.md section 2, step 4, does not say what to do with a {word} vector")
print("ok: every vector has its RTL obligation (%s)" % ", ".join(f"{p} {c['pass']}/{c['none']}/{c['undecided']}" for p, c in counts.items()))
