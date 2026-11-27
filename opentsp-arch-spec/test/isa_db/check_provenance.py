"""Provenance and source-coverage checks (no model build needed).

1. Every config key in config/ref.json has a record in isa-db/provenance.yaml, with a
   basis and a citation; every catalogue instruction's slice (or slice.name) has a record.
2. The queue ranges in provenance agree with core/queues.sail and cover 0..143 exactly.
3. Source coverage: every instruction listed in ISCA20 Table I and HC34 p.25 is either in
   the catalogue or in the documented abstraction list below. A new source instruction
   therefore cannot be forgotten silently.

usage: check_provenance.py
"""
import glob, json, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
BASES = {"paper", "figure", "slides", "decision", "implementation-choice"}

# ISCA 2020 Table I and HC34 p.25 instruction lists (slice, mnemonic as published), plus
# instructions named elsewhere in the sources (ISCA 2022).
SOURCE_INSTRUCTIONS = [
    ("CU", "NOP"), ("CU", "Ifetch"), ("CU", "Sync"), ("CU", "Notify"), ("CU", "Config"), ("CU", "Repeat"),
    ("MEM", "Read"), ("MEM", "Write"), ("MEM", "Gather"), ("MEM", "Scatter"),
    ("MEM", "Countdown"), ("MEM", "Step"), ("MEM", "Iterations"),
    ("VXM", "unary operation"), ("VXM", "binary operation"), ("VXM", "type conversions"),
    ("VXM", "ReLU"), ("VXM", "TanH"), ("VXM", "Exp"), ("VXM", "RSqrt"),
    ("MXM", "LW"), ("MXM", "IW"), ("MXM", "ABC"), ("MXM", "ACC"),
    ("SXM", "Shift"), ("SXM", "Select"), ("SXM", "Permute"), ("SXM", "Distribute"), ("SXM", "Rotate"), ("SXM", "Transpose"),
    ("C2C", "Deskew"), ("C2C", "Send"), ("C2C", "Receive"),
    # Further instructions named elsewhere in the sources and in ISCA 2022.
    ("C2C", "Transmit"), ("C2C", "RuntimeDeskew"),
    ("C2C", "PciReceive"),     # the host interface receive (ISCA20 sec. II-6); the two-step form is the model's choice
]
# Source entries that are families, mapped to the catalogue entries that implement them.
FAMILIES = {
    ("VXM", "unary operation"): ["Not", "Neg", "Abs", "Mask"],
    ("VXM", "binary operation"): ["Add", "Sub", "Mul", "Max", "Min", "And", "Or", "Xor"],
    ("VXM", "type conversions"): ["Convert"],
    ("SXM", "Select"): ["SelectS"],
}
# Known disagreements between the sources; each must have a conflicts: record.
REQUIRED_CONFLICTS = ["sram.ports", "c2c.links", "timing.encoding", "queues.split", "mxm.f16_capacity"]
OPENTSP_SPLIT = {"mem": 88, "vxm": 16, "mxm": 12, "sxm_w": 14, "sxm_e": 14}
# Source instructions deliberately not in the catalogue, with the reason (none at present).
ABSTRACTED = {}


def config_keys(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from config_keys(v, p + k + ".")
        else:
            yield p + k


WORDS = {"published", "derived", "implementation choice"}
DOC = os.path.join(ROOT, "docs", "microarchitecture.md")


def effective_row(cfg, entry, cls):
    """The (d_func, d_skew) of a catalogue entry in a profile: its timing.overrides row, else its class row."""
    for o in cfg["timing"]["overrides"]:
        if o["entry"] == entry:
            return [o["d_func"], o["d_skew"]]
    return [cfg["timing"][cls]["d_func"], cfg["timing"][cls]["d_skew"]]


def check_timing_rows(prov, entries, errors):
    """Every catalogue entry has a timing row with one of the three words, its row is the
    effective row of the shipped profiles, and microarchitecture.md section 1 shows the same table."""
    rows = prov.get("timing_rows", {})
    want = {f"{e['slice']}.{e['name']}": e for e in entries}
    for k in sorted(set(want) - set(rows)):
        errors.append(f"timing_rows: catalogue entry {k} has no row")
    for k in sorted(set(rows) - set(want)):
        errors.append(f"timing_rows: {k} is not a catalogue entry")
    for k, r in rows.items():
        if r.get("word") not in WORDS:
            errors.append(f"timing_rows.{k}: word must be one of {sorted(WORDS)}")
        if not r.get("cite") or len(r["cite"]) < 40:
            errors.append(f"timing_rows.{k}: needs a cite (the evidence or its stated absence)")
        if ("rule" in r) != ("rule_word" in r) or ("rule_word" in r and r["rule_word"] not in WORDS):
            errors.append(f"timing_rows.{k}: rule and rule_word go together, the word one of {sorted(WORDS)}")
        if k in want and r.get("class") != want[k]["timing"].lower():
            errors.append(f"timing_rows.{k}: class {r.get('class')} but the catalogue says {want[k]['timing']}")
        if r.get("word") == "implementation choice" and not r.get("why"):
            errors.append(f"timing_rows.{k}: an implementation choice needs its engineering rationale")
    for prof in ("ref", "mini"):
        cfg = json.load(open(os.path.join(ROOT, "config", prof + ".json")))
        for k, r in rows.items():
            if k in want and effective_row(cfg, want[k]["name"], r.get("class", "")) != r.get("row"):
                errors.append(f"timing_rows.{k}: row {r.get('row')} but {prof}.json gives {effective_row(cfg, want[k]['name'], r.get('class', ''))}")
    if rows.get("MXM.ABC", {}).get("word") != "published":
        errors.append("timing_rows.MXM.ABC: the matrix execution time is published (HC34 p.28)")
    if rows.get("MXM.ACC", {}).get("word") != "implementation choice":
        errors.append("timing_rows.MXM.ACC: no source gives ACC a latency")
    doc = open(DOC).read()
    # the table of section 1: one line per entry with its class, row, word and rule word
    lines = {}
    for line in doc.splitlines():
        m = re.match(r"\| `(\w+)` \| `(\w+)` \| (\d+) \| (\d+) \| ([a-z ]+) \| (.*) \| ([a-z -]+) \|$", line)
        if m:
            lines[m.group(1)] = m.groups()
    for k, r in rows.items():
        name = k.split(".", 1)[1]
        g = lines.get(name)
        if g is None:
            errors.append(f"microarchitecture.md section 1: no table line for {name}")
        elif (g[1], int(g[2]), int(g[3]), g[4], g[6]) != (r["class"], r["row"][0], r["row"][1], r["word"], r.get("rule_word", "-")):
            errors.append(f"microarchitecture.md section 1: the table line of {name} differs from timing_rows.{k}")


TIMING_BOUND_FIELDS = ("source", "quote", "convention", "value", "model_value", "status")
STATUS_RE = re.compile(r"^(met|exceeded by \d+ \(recorded\)|not reproduced \(.+\))$")
SPECS = os.path.join(ROOT, "docs")


def check_timing_bounds(prov, errors):
    """The `timing_bounds` section parses: every entry has its six fields and a status of the allowed forms."""
    bounds = prov.get("timing_bounds")
    if not isinstance(bounds, dict) or not bounds:
        errors.append("timing_bounds: the section is missing or empty")
        return
    for k, r in bounds.items():
        for f in TIMING_BOUND_FIELDS:
            if f not in r or r[f] in (None, ""):
                errors.append(f"timing_bounds.{k}: missing field {f}")
        if "status" in r and not STATUS_RE.match(str(r["status"])):
            errors.append(f"timing_bounds.{k}: status {r['status']!r} is not 'met', 'exceeded by n (recorded)' or 'not reproduced (reason)'")


def check_iq_capacity(errors):
    """The public topology chapter agrees with the shipped profiles about IQ capacity."""
    with open(os.path.join(SPECS, "topology.md"), encoding="utf-8") as doc:
        row = re.search(r"^- `icu\.iq_depth`: .*?\b(\d+) bytes in `ref` and `mini`", doc.read(), re.M)
    if row is None:
        errors.append("topology.md: missing shipped-profile IQ capacity")
        return
    for profile in ("ref", "mini"):
        with open(os.path.join(ROOT, "config", profile + ".json")) as config:
            depth = json.load(config)["icu"]["iq_depth"]
        if int(row.group(1)) != depth:
            errors.append(f"topology.md: icu.iq_depth is {depth} in {profile}.json; the text says {row.group(1)}")


def check_design_space(prov, errors):
    """design-space.md lists every implementation-choice timing row of the provenance, with its d_func and d_skew."""
    text = open(os.path.join(SPECS, "design-space.md"), encoding="utf-8").read()
    lines = {}
    for m in re.finditer(r"^\| `(\w+)` \| (\d+) \| (\d+) \|", text, re.M):
        lines[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    choices = {k.split(".", 1)[1]: r["row"] for k, r in prov.get("timing_rows", {}).items() if r.get("word") == "implementation choice"}
    for name, row in sorted(choices.items()):
        if name not in lines:
            errors.append(f"design-space.md: the implementation-choice timing row {name} is not listed")
        elif list(lines[name]) != row:
            errors.append(f"design-space.md: {name} has row {list(lines[name])}, the provenance has {row}")
    for name in sorted(set(lines) - set(choices)):
        errors.append(f"design-space.md: {name} is listed but is not an implementation-choice timing row")


def main():
    prov = yaml.safe_load(open(os.path.join(ROOT, "isa-db", "provenance.yaml")))
    errors = []
    for section in ("config", "queues", "instructions"):
        for k, r in prov[section].items():
            if r.get("basis") not in BASES or not r.get("cite"):
                errors.append(f"provenance {section}.{k}: needs basis in {sorted(BASES)} and a cite")
    # implementation-choice (the interface a source does not publish, the host path): the record says why.
    for section in ("config", "queues", "instructions", "host", "ecc", "ras"):
        for k, r in prov.get(section, {}).items():
            if r.get("basis") == "implementation-choice" and not r.get("why"):
                errors.append(f"provenance {section}.{k}: basis implementation-choice needs a why")
    for sec in ("host", "ecc", "ras"):
        for k, r in prov.get(sec, {}).items():
            if r.get("basis") not in BASES or not r.get("cite"):
                errors.append(f"provenance {sec}.{k}: needs basis in {sorted(BASES)} and a cite")
    cfg = json.load(open(os.path.join(ROOT, "config", "ref.json")))
    for k in config_keys(cfg):
        if k not in prov["config"]:
            errors.append(f"config key {k} has no provenance record")
    for k in prov["config"]:
        if k not in set(config_keys(cfg)):
            errors.append(f"provenance record {k} names no config key")
    # Queue ranges: contiguous, 0..143, and as in core/queues.sail.
    rs = sorted(tuple(r["range"]) for r in prov["queues"].values() if "range" in r)
    nxt = 0
    for lo, hi in rs:
        if lo != nxt:
            errors.append(f"queue ranges: gap or overlap at {lo}")
        nxt = hi + 1
    if nxt != 144:
        errors.append(f"queue ranges end at {nxt - 1}, not 143")
    # The published split (ScaledML 2020 and ISC 2020 floorplans; MPR 2020 Fig. 2): MEM 44 + 44, VXM 16, and per
    # hemisphere MXM 6 and SXM 14, the C2C queues counted in the SXM block.
    sizes = {n: r["range"][1] - r["range"][0] + 1 for n, r in prov["queues"].items() if "range" in r}
    for name, want in OPENTSP_SPLIT.items():
        if sizes.get(name) != want:
            errors.append(f"queue range {name}: {sizes.get(name)} queues, the published split has {want}")
    # The ranges must match the header of core/queues.sail ("lo .. hi" lines); the
    # model's issuer_of is checked against the same table by $[test] test_queue_units.
    qs = open(os.path.join(ROOT, "model", "core", "queues.sail")).read()
    starts = {int(a) for a in re.findall(r"//\s+(\d+) \.\.", qs)}
    for name, r in prov["queues"].items():
        if "range" in r and r["range"][0] not in starts:
            errors.append(f"queue range {name} {r['range']} not in core/queues.sail")
    # Catalogue entries have records; source instructions are covered.
    entries = [yaml.safe_load(open(f)) for f in glob.glob(os.path.join(ROOT, "isa-db", "instructions", "*.yaml"))]
    names = {(e["slice"], e["name"]) for e in entries}
    for s, n in sorted(names):
        if s not in prov["instructions"] and f"{s}.{n}" not in prov["instructions"]:
            errors.append(f"catalogue {s}/{n} has no provenance record")
    check_timing_rows(prov, entries, errors)
    check_timing_bounds(prov, errors)
    check_iq_capacity(errors)
    check_design_space(prov, errors)
    for s, n in SOURCE_INSTRUCTIONS:
        if (s, n) in ABSTRACTED:
            continue
        impl = FAMILIES.get((s, n), [n])
        missing = [x for x in impl if (s, x) not in names]
        if missing:
            errors.append(f"source instruction {s} {n!r} not in the catalogue (missing {missing})")
    # Source conflicts: where the sources disagree, the record names every side and the
    # winner, and the decision it leads to is labelled as such.
    conflicts = prov.get("conflicts", {})
    for k in REQUIRED_CONFLICTS:
        if k not in conflicts:
            errors.append(f"source conflict {k} is not recorded under conflicts:")
    for k, r in conflicts.items():
        if not isinstance(r.get("sides"), list) or len(r["sides"]) < 2 or not r.get("chosen") or not r.get("reason"):
            errors.append(f"conflict {k}: needs sides (2 or more), chosen and reason")
        elif r["chosen"] not in [x.get("source") for x in r["sides"]]:
            errors.append(f"conflict {k}: chosen {r['chosen']!r} is not one of its sides")
    # Labels that would overstate a source: a record whose citation calls the
    # value an opentsp mapping or decision cannot have a source basis.
    for section in ("config", "queues", "instructions"):
        for k, r in prov[section].items():
            c = r.get("cite", "")
            if r.get("basis") != "decision" and re.search(r"opentsp (mapping|decision)|is an opentsp", c) and not re.search(r"are opentsp decisions", c):
                errors.append(f"provenance {section}.{k}: basis {r['basis']} but the citation says it is an opentsp mapping/decision")
    print(f"provenance: {len(prov['config'])} config, {len(prov['queues'])} queue, {len(prov['instructions'])} instruction records; "
          f"{len(SOURCE_INSTRUCTIONS)} source instructions checked, {len(prov['timing_rows'])} timing rows; {len(errors)} errors")
    for e in errors:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
