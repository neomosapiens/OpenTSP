"""Cross-check the ISA catalogue (isa-db/instructions/*.yaml) against the Sail model.

Sail is normative; the YAML catalogue is derived and must agree.
For every instruction file, and for random fills of its variable fields, this asks the
model (through test/isa_db/isa_query, built by tools/build_isa_query.sh) to decode the
word, and checks:
  (a) the word decodes (it is not ILLEGAL) and its assembly starts with the mnemonic;
  (b) encode(decode(w)) == w;
  (c) field positions: the assembly equals the YAML template with the field values
      substituted, so every variable sits exactly where the YAML says;
  (d) values the YAML excludes (values / not_values / max) do not decode, and flipping
      any bit the YAML marks fixed makes the word decode to something else;
  (e) coverage: the catalogue has exactly one file per instruction encoding (EXPECTED);
  (f) no two entries' encodings can match the same word;
  (i) text format (microarchitecture.md sec. 3a): bit 31 / 63 of every instruction is a fixed 0 (bit 31 of
      the first unit is the bundle bit); a 64-bit instruction has a fixed 0 bit in 31..0 (its second
      unit is never the padding unit 0xFFFFFFFF); the text decode (isa_query `text`) of the units of
      a random legal word of every entry, with the bundle bit 0 or 1, equals the width decode, and
      tags 6 and 7 are OTS-E-DEC-003;
  (h) stations: encoding.json `stations`, evaluated with the profile's layout, gives the
      model's read and write gap (isa_station) on every queue 0..143, row and direction;
  (g) timing: every word of an entry belongs to that entry's row of the timing table
      (the model's entry_of is the YAML name) and has the YAML timing class (timing_of), and
      the model's table of catalogue entries (postlude/timing_check.sail) lists exactly the
      catalogue's names with those classes.

usage: check_isa_db.py <isa_query> <config.json> [samples-per-instruction] [seed]
"""
import glob, os, random, subprocess, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "..", "isa-db", "instructions")
TYPES = ["i8", "f16", "i32", "f32", "u8", "i16", "u16", "u32", "b8", "b16", "b32", "e4m3", "e5m2", None, None, None, "bf16"]
BOOLS = ("b8", "b16", "b32")
CMP = ["eq", "ne", "lt", "le", "gt", "ge"]
RED = ["add", "max", "min"]
ACC_DEPTH = 2     # mxm.acc_depth of the profile under test (set by main)
EXPECTED = 78     # instruction encodings in the model (72 + the four host instructions + ABC8, the fp8 form of ABC + DeskewRecv)


class Model:
    def __init__(self, exe, cfg):
        self.p = subprocess.Popen([exe, cfg], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def query(self, width, w):
        self.p.stdin.write(f"{width} {w:x}\n")
        ok, text = self.p.stdout.readline().rstrip("\n").split("\t", 1)
        return ok == "1", text

    def entry(self, width, w):
        """(catalogue entry, timing class, element type its timing row may be keyed by) of the word, from the model's entry_of,
        timing_of and timing_type_of ("" for an instruction without an element type)."""
        self.p.stdin.write(f"e{width} {w:x}\n")
        return tuple(self.p.stdout.readline().rstrip("\n").split("\t", 2))

    def takes_type(self, name, ty):
        """Whether the model accepts a timing row "<name>.<ty>" as naming a type the entry takes."""
        self.p.stdin.write(f"takes {name} {ty}\n")
        return self.p.stdout.readline().strip() == "1"

    def station(self, q, row, write):
        self.p.stdin.write(f"gap {q} {row} {1 if write else 0}\n")
        return int(self.p.stdout.readline())

    def entry_class(self, name):
        self.p.stdin.write(f"class {name}\n")
        return self.p.stdout.readline().rstrip("\n")

    def text(self, u0, u1=0):
        """(decode-level code or None, units, text) by the text-level decode rule."""
        self.p.stdin.write(f"text {u0:x} {u1:x}\n")
        code, units, text = self.p.stdout.readline().rstrip("\n").split("\t", 2)
        return (None if code == "-" else code), int(units), text


def span(location):
    a, _, b = location.partition("-")
    return int(a), int(b or a)


def loc(v):
    return span(v["location"])


PARTS = ("location", "hi_location", "top_location")


def nbits(v):
    """Logical width of a variable (a split field spans location, hi_location and top_location)."""
    return sum(a - b + 1 for a, b in (span(v[k]) for k in PARTS if k in v))


def bits_of(v):
    """All bit positions a variable occupies."""
    out = []
    for k in PARTS:
        if k in v:
            msb, lsb = span(v[k])
            out += list(range(lsb, msb + 1))
    return out


def place(v, x):
    """The word bits of logical value x of variable v."""
    w, shift = 0, 0
    for k in PARTS:
        if k in v:
            msb, lsb = span(v[k])
            n = msb - lsb + 1
            w |= ((x >> shift) & ((1 << n) - 1)) << lsb
            shift += n
    return w


def stream_text(x):
    return ("W:" if x & 32 else "E:") + str(x & 31)


def render(kind, x, width_bits):
    """The assembly text of field value x, as the Sail assembly mappings print it."""
    if kind == "stream":
        return stream_text(x)
    if kind == "stream_group16":
        return stream_text(x << 4)
    if kind in ("eltype", "eltype_code", "eltype4"):
        return TYPES[x]
    if kind == "mxm_wtype":
        return {0: "i8", 1: "f16", 2: "u8", 3: "fp8", 4: "bf16", 5: "bfp8"}[x]
    if kind == "fp8_format":
        return ["e4m3", "e5m2"][x]
    if kind == "simm16":
        return hex(x)
    if kind == "mxm_acctype":
        return {0: "x128", 2: "i32", 3: "f32"}[x]
    if kind == "cmp_mode":
        return CMP[x]
    if kind == "reduce_op":
        return RED[x]
    if kind == "acc_index":
        return str(x)
    if kind == "acc_mode":
        return ["replace", "accumulate"][x]
    if kind == "chain":          # a VXM source chain bit
        return ["", ".chain"][x]
    if kind == "group4":         # the VXM group bit
        return ["", ".sg4"][x]
    if kind == "shift_dir":
        return ["up", "down"][x]
    return hex(x)   # immediates, addresses, strides, counts, map ids, chip ids, n


def profile_vars(cfg):
    """The profile variables of encoding.json that bound a field (`max_expr`) for a configuration: the lane count and the words a slice addresses."""
    return {"vlanes": 16 * cfg["geometry"]["superlanes"], "sram_words": 2 ** cfg["sram"]["addr_bits"], "acc_depth": cfg["mxm"]["acc_depth"]}


def expr_bound(v, pv):
    """The value of the field's `max_expr` (`<variable>` or `<variable>-<int>`); `pv` is the profile variables, or the lane count alone (then
    the SRAM address operands have the reference 13 bits)."""
    if not isinstance(pv, dict):
        pv = {"vlanes": pv, "sram_words": 8192, "acc_depth": ACC_DEPTH}
    name, _, off = v["max_expr"].partition("-")
    return pv[name.strip()] - (int(off) if off else 0)


def corner_values(v, pv):
    """The values tried when a field is too wide to list (legal_values is None): 0, 1 and the largest legal value."""
    hi = (1 << nbits(v)) - 1
    if "max_expr" in v:
        hi = min(hi, expr_bound(v, pv))
    return [x for x in (0, 1, hi) if x not in v.get("not_values", []) and x <= v.get("max", x) and x <= hi]


def undeclared(v, declared):
    """The codes of v that belong to an extension the profile does not declare."""
    return {c for ext, cs in v.get("extension_codes", {}).items() if ext not in declared for c in cs}


def legal_values(v, pv, declared=()):
    n = nbits(v)
    if "values" in v:
        return [x for x in v["values"] if x not in undeclared(v, declared)]
    if "codes" in v:
        return [x for x in v["codes"] if x not in undeclared(v, declared)]
    if v["kind"] == "simm16":
        lo, hi = v.get("min", -32768), v.get("max", 32767)
        if "max_expr" in v:           # a signed field: the expression bounds the magnitude
            hi = min(hi, expr_bound(v, pv))
            lo = max(lo, -hi)
        return [x & 0xFFFF for x in (lo, -1, 0, 1, hi)]
    hi = (1 << n) - 1
    if "max_expr" in v:
        hi = min(hi, expr_bound(v, pv))
    elif v["kind"] == "lane_count":
        hi = min(hi, pv["vlanes"] - 1 if isinstance(pv, dict) else pv - 1)
    if v["kind"] == "acc_index":
        hi = min(hi, ACC_DEPTH - 1)
    if "max" in v:
        hi = min(hi, v["max"])
    return [x for x in range(0, hi + 1) if x not in v.get("not_values", [])] if n <= 10 else None


def pick(v, rnd, pv, declared=()):
    vals = legal_values(v, pv, declared)
    if vals is not None:
        return rnd.choice(vals)
    if v["kind"] == "simm16":
        hi = v.get("max", 32767)
        lo = v.get("min", -32768)
        if "max_expr" in v:
            hi = min(hi, expr_bound(v, pv))
            lo = max(lo, -hi)
        return rnd.randint(lo, hi) & 0xFFFF
    while True:
        x = rnd.getrandbits(nbits(v))
        if "max" in v and x > v["max"]:
            continue
        if "max_expr" in v and x > expr_bound(v, pv):
            continue
        if x not in v.get("not_values", []):
            return x


def build(ins, fields):
    w = int(ins["encoding"]["match"].replace("-", "0"), 2)
    for v in ins["encoding"]["variables"]:
        w |= place(v, fields[v["name"]])
    return w


def expected_asm(ins, fields):
    width = ins["encoding"]["width"]
    text = ins["assembly"]
    for v in ins["encoding"]["variables"]:
        text = text.replace("{" + v["name"] + "}", render(v["kind"], fields[v["name"]], width))
    return text


def main(argv):
    exe, cfg = argv[1], argv[2]
    samples = int(argv[3]) if len(argv) > 3 else 40
    rnd = random.Random(int(argv[4]) if len(argv) > 4 else 1)
    vlanes = profile_vars(yaml.safe_load(open(cfg)))   # JSON is YAML; the profile variables that bound a field
    global ACC_DEPTH
    ACC_DEPTH = yaml.safe_load(open(cfg))["mxm"]["acc_depth"]
    declared = tuple(yaml.safe_load(open(cfg)).get("extensions", []))
    m = Model(exe, cfg)
    files = sorted(glob.glob(os.path.join(DB, "*.yaml")))
    errors, checked, seen = [], 0, {}

    def fail(ins, msg):
        errors.append(f"{ins['slice']}/{ins['name']}: {msg}")

    for f in files:
        ins = yaml.safe_load(open(f))
        enc = ins["encoding"]
        width, pat = enc["width"], enc["match"]
        if len(pat) != width:
            fail(ins, f"match has {len(pat)} bits, width {width}")
            continue
        # Variables cover exactly the '-' positions of match, without overlap.
        cover = ["0"] * width
        for v in enc["variables"]:
            for i in bits_of(v):
                if cover[i] == "1":
                    fail(ins, f"variable {v['name']} overlaps")
                cover[i] = "1"
        if "".join(reversed(cover)) != "".join("1" if c == "-" else "0" for c in pat):
            fail(ins, "variables do not cover exactly the '-' positions of match")
        key = (width, pat)
        if key in seen:
            fail(ins, f"same match as {seen[key]}")
        seen[key] = ins["name"]
        if m.entry_class(ins["name"]) != ins["timing"]:
            fail(ins, f"model timing table class {m.entry_class(ins['name'])!r} != catalogue {ins['timing']!r}")
        # a timing row keyed by an element type is accepted for exactly the types of the entry's `type` operand
        codes = {c for v in enc["variables"] if v["name"] == "type" and v["kind"] == "eltype_code" for c in v["codes"]}
        for code, tn in enumerate(TYPES):
            if m.takes_type(ins["name"], tn) != (code in codes):
                fail(ins, f"model: a timing row {ins['name']}.{tn} is {'refused' if code in codes else 'accepted'}, the catalogue says the entry "
                          f"{'takes' if code in codes else 'does not take'} {tn}")
        # (a)-(c) on the minimum, maximum and random legal field values.
        trials = []
        for corner in ("lo", "hi"):
            fv = {}
            for v in enc["variables"]:
                vals = legal_values(v, vlanes, declared)
                if vals is None:
                    vals = corner_values(v, vlanes)
                fv[v["name"]] = min(vals) if corner == "lo" else max(vals)
            trials.append(fv)
        trials += [{v["name"]: pick(v, rnd, vlanes, declared) for v in enc["variables"]} for _ in range(samples)]
        # Joint constraint: a field with bool_values only allows those when the type is boolean.
        def joint_ok(fv):
            for v in enc["variables"]:
                if "bool_values" in v and TYPES[fv.get("type", 0)] in BOOLS and fv[v["name"]] not in v["bool_values"]:
                    return False
            return True
        trials = [fv for fv in trials if joint_ok(fv)] or trials[:1]
        for fv in trials:
            w = build(ins, fv)
            ok, text = m.query(width, w)
            checked += 1
            if text.startswith("illegal"):
                fail(ins, f"{w:#x} with {fv} does not decode")
                break
            if not ok:
                fail(ins, f"{w:#x}: encode(decode(w)) != w ({text})")
                break
            if text.split(" ")[0].split(".")[0] != ins["mnemonic"]:
                fail(ins, f"{w:#x} decodes as {text!r}, expected mnemonic {ins['mnemonic']}")
                break
            exp = expected_asm(ins, fv)
            if text != exp:
                fail(ins, f"{w:#x}: model assembly {text!r} != catalogue {exp!r}")
                break
            # (g) the word's timing-table row and class.
            got = m.entry(width, w)
            want_type = next((TYPES[fv["type"]] for v in enc["variables"] if v["name"] == "type" and v["kind"] == "eltype_code"), "")
            if got != (ins["name"], ins["timing"], want_type):
                fail(ins, f"{w:#x}: model entry/timing/type {got} != catalogue {(ins['name'], ins['timing'], want_type)}")
                break
        # (d) excluded field values do not decode as this instruction.
        base = trials[0]
        for v in enc["variables"]:
            n = nbits(v)
            vals = legal_values(v, vlanes, declared)
            bad = [x for x in v.get("not_values", [])]
            if v["kind"] == "simm16":
                vals = None
                bad += [x & 0xFFFF for x in (v.get("min", -32768) - 1, v.get("max", 32767) + 1) if -32768 <= x <= 32767]
                if "max_expr" in v:
                    e = expr_bound(v, vlanes)
                    bad += [x & 0xFFFF for x in (-e - 1, e + 1) if -32768 <= x <= 32767]
            elif "max" in v and n <= 16:
                bad += [v["max"] + 1] if v["max"] + 1 < (1 << n) else []
            elif "max_expr" in v and n <= 16:
                bad += [expr_bound(v, vlanes) + 1] if expr_bound(v, vlanes) + 1 < (1 << n) else []
            if vals is not None and n <= 10:
                bad += [x for x in range(1 << n) if x not in vals][:8]
            for x in bad:
                w = build(ins, dict(base, **{v["name"]: x}))
                _, text = m.query(width, w)
                checked += 1
                if not text.startswith("illegal") and expected_asm(ins, dict(base, **{v["name"]: x})) == text:
                    fail(ins, f"excluded {v['name']}={x} decodes as {text!r}")
        # (d) every fixed bit matters.
        w0 = build(ins, base)
        exp0 = expected_asm(ins, base)
        for i, c in enumerate(reversed(pat)):
            if c == "-":
                continue
            _, text = m.query(width, w0 ^ (1 << i))
            checked += 1
            if text == exp0:
                fail(ins, f"fixed bit {i} is ignored by the model")
    # (f) distinct encodings: for every pair of entries of one width whose fixed bits are
    # compatible, a word built from one entry's legal fields with the other entry's fixed
    # bits forced must not decode as both. Compatible pairs are reported with the word that
    # the model decodes, so decode never depends on the order of the Sail clauses.
    entries = [yaml.safe_load(open(f)) for f in files]
    overlaps = 0
    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            ea, eb = a["encoding"], b["encoding"]
            if ea["width"] != eb["width"]:
                continue
            if any(x != "-" and y != "-" and x != y for x, y in zip(ea["match"], eb["match"])):
                continue
            # Patterns compatible: try to build a word legal for both.
            for _ in range(20):
                fa = {v["name"]: pick(v, rnd, vlanes, declared) for v in ea["variables"]}
                w = build(a, fa)
                wb = int(eb["match"].replace("-", "0"), 2)
                mb = int("".join("0" if c == "-" else "1" for c in eb["match"]), 2)
                if w & mb == wb:
                    _, text = m.query(ea["width"], w)
                    checked += 1
                    overlaps += 1
                    errors.append(f"{a['name']} and {b['name']} both match {w:#x} (model: {text!r})")
                    break
    # (i) text format.
    for ins in entries:
        enc = ins["encoding"]
        width, pat = enc["width"], enc["match"]
        if pat[0] != "0":
            fail(ins, f"bit {width - 1} (the bundle bit of the first unit) is not a fixed 0")
        if width == 64 and "0" not in pat[32:]:
            fail(ins, "no fixed 0 bit in 31..0: the second unit could be the padding unit 0xFFFFFFFF")
        if width == 32 and pat[1:4] == "111":
            fail(ins, "tag 7")
        for _ in range(4):
            base = {v["name"]: pick(v, rnd, vlanes, declared) for v in enc["variables"]}
            w = build(ins, base)
            _, want = m.query(width, w)
            if want.startswith("illegal"):
                continue
            units = [w] if width == 32 else [w >> 32, w & 0xFFFFFFFF]
            for bundle in (0, 1):
                u = [units[0] | (bundle << 31)] + units[1:]
                code, nu, text = m.text(u[0], u[1] if len(u) > 1 else 0)
                checked += 1
                if code or text != want or nu != len(units):
                    fail(ins, f"text decode of {[hex(x) for x in u]} is ({code}, {nu}, {text!r}), the width decode is {want!r}")
    for tag in (6, 7):
        for b in (0, 1):
            code, nu, text = m.text((b << 31) | (tag << 28) | 0x0123456, 0)
            checked += 1
            if code != "OTS-E-DEC-003" or nu != 1:
                errors.append(f"text unit with tag {tag} decodes as ({code}, {nu}, {text!r}), not OTS-E-DEC-003")
    # (e) coverage: one entry per instruction clause the model defines.
    names = {(yaml.safe_load(open(f))["slice"], yaml.safe_load(open(f))["name"]) for f in files}
    if len(names) != EXPECTED:
        errors.append(f"catalogue has {len(names)} instructions, expected {EXPECTED}")
    # (h) stations: encoding.json against the model's read_gap/write_gap.
    import json
    enc = json.load(open(os.path.join(HERE, "..", "..", "isa-db", "generated", "encoding.json")))
    layout = json.load(open(cfg))["layout"]
    unit_of = {q: u for u, qs in enc["units"].items() if u not in ("text", "any") for q in qs}

    def evaluate(expr, q):
        if expr is None:
            return -1
        key, *rest = expr.split()
        v = layout[key.split(".", 1)[1]]
        if rest:
            quad = (q % 44) // 4
            v = v - quad if rest[0] == "-" else v + quad
        return v if 0 <= v <= 43 else -1

    for q in range(144):
        st = enc["stations"][unit_of[q]]
        for r, row in enumerate("EW"):
            for write in (False, True):
                want = evaluate(st["write" if write else "read"][row], q)
                got = m.station(q, r, write)
                checked += 1
                if got != want:
                    errors.append(f"stations: queue {q} ({unit_of[q]}) {'writes' if write else 'reads'} {row} at "
                                  f"{got} in the model, {want} in encoding.json")
    m.p.stdin.close()
    m.p.wait()
    print(f"{len(files)} catalogue entries, {checked} model queries, {len(errors)} errors")
    for e in errors[:40]:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
