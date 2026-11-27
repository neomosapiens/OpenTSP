"""Check the conformance vectors against conformance/model/schema.json (readiness R13c).

The venv has no jsonschema package, so this is a small validator for exactly the JSON Schema
(2020-12) keywords schema.json uses: type, const, enum, minimum, maximum, pattern, required,
properties, additionalProperties, propertyNames, items, prefixItems, minItems, maxItems, $ref.
A keyword it does not know is an error, so the schema cannot silently outgrow it.

Beyond the schema it checks what JSON Schema cannot express: the profile's geometry (slice
< 88, address < sram.words_per_slice, a word is vlanes bytes), expect.words keys = dump,
every unit list fits icu.iq_depth, every code is in encoding.json diagnostics, config_sha256 is the digest of the
profile's configuration, every trace event carries a vlanes-byte value, the events of a vector are in canonical order and
hash to expect.trace_sha256. It also checks that the
validator rejects mutated vectors (so a broken validator cannot pass).

usage: check_conformance_schema.py [vectors.json ...]   (default: conformance/model/*.json)
"""
import copy, glob, hashlib, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "..", "..")
SCHEMA = os.path.join(SPEC, "conformance", "model", "schema.json")
ENC = os.path.join(SPEC, "isa-db", "generated", "encoding.json")
sys.path.insert(0, os.path.join(SPEC, "tools"))
from opentsp_reference.schema import with_defaults
KNOWN = {"$schema", "$id", "title", "description", "$defs", "type", "const", "enum", "minimum", "maximum", "pattern",
         "required", "properties", "additionalProperties", "propertyNames", "items", "prefixItems", "minItems", "maxItems", "$ref"}
TYPES = {"object": dict, "array": list, "string": str}


def validate(x, s, root, path="$", errs=None):
    errs = [] if errs is None else errs
    unknown = set(s) - KNOWN
    if unknown:
        raise SystemExit(f"check_conformance_schema: schema keyword(s) not supported: {sorted(unknown)} at {path}")
    if "$ref" in s:
        node = root
        for part in s["$ref"].lstrip("#/").split("/"):
            node = node[part]
        validate(x, node, root, path, errs)
    t = s.get("type")
    if t == "integer":
        if not isinstance(x, int) or isinstance(x, bool):
            return errs.append(f"{path}: not an integer") or errs
    elif t and not isinstance(x, TYPES[t]):
        return errs.append(f"{path}: not {t}") or errs
    if "const" in s and x != s["const"]:
        errs.append(f"{path}: {x!r} != {s['const']!r}")
    if "enum" in s and x not in s["enum"]:
        errs.append(f"{path}: {x!r} not in {s['enum']}")
    if "minimum" in s and x < s["minimum"]:
        errs.append(f"{path}: {x} < {s['minimum']}")
    if "maximum" in s and x > s["maximum"]:
        errs.append(f"{path}: {x} > {s['maximum']}")
    if "pattern" in s and not re.search(s["pattern"], x):
        errs.append(f"{path}: {x[:40]!r} does not match {s['pattern']}")
    if isinstance(x, dict):
        for k in s.get("required", []):
            if k not in x:
                errs.append(f"{path}: missing {k}")
        props = s.get("properties", {})
        for k, v in x.items():
            if "propertyNames" in s:
                validate(k, s["propertyNames"], root, f"{path}[key {k!r}]", errs)
            if k in props:
                validate(v, props[k], root, f"{path}.{k}", errs)
            elif s.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k}")
            elif isinstance(s.get("additionalProperties"), dict):
                validate(v, s["additionalProperties"], root, f"{path}[{k!r}]", errs)
    if isinstance(x, list):
        if len(x) < s.get("minItems", 0) or len(x) > s.get("maxItems", len(x)):
            errs.append(f"{path}: {len(x)} items")
        pre = s.get("prefixItems", [])
        for i, v in enumerate(x):
            if i < len(pre):
                validate(v, pre[i], root, f"{path}[{i}]", errs)
            elif "items" in s:
                validate(v, s["items"], root, f"{path}[{i}]", errs)
    return errs


def canonical_events(evs):
    """Each cycle's run of consecutive `srf` events sorted by (row, stream, gap); the rest keep their order."""
    out, run = [], []
    def key(e):
        t = e.split()
        row, st = t[2].split(":")
        return (row, int(st), int(t[4]))
    for e in evs + ["end"]:
        if e.startswith("srf ") and (not run or run[0].split()[1] == e.split()[1]):
            run.append(e)
            continue
        out.extend(sorted(run, key=key))
        run = [e] if e.startswith("srf ") else []
        if not e.startswith("srf ") and e != "end":
            out.append(e)
    return out


def semantic(doc):
    errs = []
    cfg = json.load(open(os.path.join(SPEC, "config", f"{doc.get('profile')}.json")))
    vbytes = 16 * cfg["geometry"]["superlanes"]
    words = cfg["sram"]["words_per_slice"]
    codes = set(json.load(open(ENC))["diagnostics"])
    cap = cfg["icu"]["iq_depth"] // 4
    if doc.get("config_sha256") != hashlib.sha256(json.dumps(with_defaults(cfg), sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        errs.append("config_sha256 is not the digest of config/%s.json" % doc.get("profile"))
    for v in doc.get("vectors", []):
        n = v.get("name")
        for q, us in v["text"].items():
            us = us.split()
            if not (int(q) <= 143 or int(q) == 65535):
                errs.append(f"{n}: text of queue {q}: not a queue")
            if len(us) > cap:
                errs.append(f"{n}: text of queue {q} has {len(us)} units, more than icu.iq_depth / 4 = {cap}")
            if "ffffffff" in us:
                errs.append(f"{n}: text of queue {q} holds the padding unit")
        keys = list(v["sram"]) + v["dump"] + list(v["expect"]["words"])
        for k in keys:
            s, a = map(int, k.split(":"))
            if s >= 88 or a >= words:
                errs.append(f"{n}: {k} outside the profile (slice < 88, address < {words})")
        for k, w in list(v["sram"].items()) + list(v["expect"]["words"].items()):
            if len(w) != 2 * vbytes:
                errs.append(f"{n}: word {k} has {len(w) // 2} bytes, not vlanes = {vbytes}")
        if sorted(v["expect"]["words"]) != sorted(v["dump"]):
            errs.append(f"{n}: expect.words keys differ from dump")
        for d in v["expect"]["diagnostics"]:
            if d["code"] not in codes:
                errs.append(f"{n}: {d['code']} is not in encoding.json diagnostics")
        evs = v["expect"].get("events")
        if evs is not None:
            for e in evs:
                if len(e.split()[-1]) != 2 * vbytes:
                    errs.append(f"{n}: event value of {e[:40]!r} has {len(e.split()[-1]) // 2} bytes, not vlanes = {vbytes}")
            if canonical_events(evs) != evs:
                errs.append(f"{n}: events are not in canonical order")
            if hashlib.sha256("".join("T " + e + "\n" for e in evs).encode()).hexdigest() != v["expect"]["trace_sha256"]:
                errs.append(f"{n}: trace_sha256 is not the digest of events")
    return errs


def check(doc, schema):
    errs = validate(doc, schema, schema)
    return errs or semantic(doc)


def mutations(doc):
    """(what, mutated document) pairs the check must reject."""
    v0 = doc["vectors"][0]
    def mut(f):
        d = copy.deepcopy(doc)
        f(d, d["vectors"][0])
        return d
    k0 = v0["dump"][0] if v0["dump"] else "0:0"
    return [
        ("spec version", mut(lambda d, v: d.update(spec_version="0.2"))),
        ("extra top-level key", mut(lambda d, v: d.update(extra=1))),
        ("queue 144", mut(lambda d, v: v["text"].update({"144": "00000001"}))),
        ("queue as a name", mut(lambda d, v: v["text"].update({"MEM[0]": "00000001"}))),
        ("negative unit", mut(lambda d, v: v["text"].update({"0": "-0000001"}))),
        ("unit of 9 digits", mut(lambda d, v: v["text"].update({"0": "000000001"}))),
        ("upper-case unit", mut(lambda d, v: v["text"].update({"0": "0000000A"}))),
        ("two spaces between units", mut(lambda d, v: v["text"].update({"0": "00000001  00000001"}))),
        ("empty text", mut(lambda d, v: v["text"].update({"0": ""}))),
        ("padding unit", mut(lambda d, v: v["text"].update({"0": "ffffffff"}))),
        ("text longer than the instruction queue", mut(lambda d, v: v["text"].update({"0": " ".join(["00000001"] * 4097)}))),
        ("records instead of text", mut(lambda d, v: v.update(records=[[1, 0, 1, 0, 0]]))),
        ("hex slice key", mut(lambda d, v: v["sram"].update({"0x1:2": "00" * 320}))),
        ("upper-case word", mut(lambda d, v: v["sram"].update({"1:2": "AB" * 320}))),
        ("short word", mut(lambda d, v: v["sram"].update({"1:2": "00" * 15}))),
        ("slice 88", mut(lambda d, v: v["sram"].update({"88:0": "00" * 320}))),
        ("unknown code", mut(lambda d, v: v["expect"]["diagnostics"].append({"cycle": 1, "queue": 0, "code": "OTS-E-XYZ-999"}))),
        ("diagnostic without queue", mut(lambda d, v: v["expect"]["diagnostics"].append({"cycle": 1, "code": "OTS-E-DEC-003"}))),
        ("digest of another trace", mut(lambda d, v: v["expect"].update(trace_sha256="0" * 64))),
        ("no trace digest", mut(lambda d, v: v["expect"].pop("trace_sha256"))),
        ("short event value", mut(lambda d, v: v["expect"].update(events=["sram 1 slice 0 addr 0 00"], trace_sha256=hashlib.sha256(b"T sram 1 slice 0 addr 0 00\n").hexdigest()))),
        ("event with a 0x value", mut(lambda d, v: v["expect"].update(events=["sram 1 slice 0 addr 0 0x00"]))),
        ("another configuration", mut(lambda d, v: d.update(config_sha256="1" * 64))),
        ("words without dump", mut(lambda d, v: v["expect"]["words"].pop(k0, None) if v["dump"] else v["dump"].append(k0))),
]


def main(argv):
    schema = json.load(open(SCHEMA))
    files = argv[1:] or sorted(f for f in glob.glob(os.path.join(SPEC, "conformance", "model", "*.json")) if not f.endswith("schema.json"))
    bad = 0
    for f in files:
        doc = json.load(open(f))
        errs = check(doc, schema)
        missed = [what for what, d in mutations(doc) if not check(d, schema)]
        if missed:
            errs.append(f"the check accepts mutated vectors: {missed}")
        print(f"conformance schema ({os.path.relpath(f, SPEC)}): {len(doc.get('vectors', []))} vectors, {len(errs)} errors")
        for e in errs[:20]:
            print("  " + e)
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
