"""A text decoder built from isa-db/generated/encoding.json alone, compared
with the Sail model.

This file deliberately imports nothing from tools/tsp_as.py or check_isa_db.py: everything
it knows comes from the JSON (decode_rule, field_rule, stream_encoding, boolean_codes,
profile_variables, control_channel, units, queues, operand_kinds, instructions) and the
profile configuration. It is the decoder a second implementation would write from
porting-guide.md sec. 2.

For random queue text (random words; words built from each entry with random field bits;
one fixed bit flipped; random tags and queues, including the control channel) it computes
the decode-level diagnostic (OTS-E-DEC-003), the placement diagnostic (OTS-E-QUE-002) and
the canonical text, and compares:
  - the text with isa_query (`text <u0> <u1>`);
  - the diagnostics with tsp_sim --trace (one lone instruction per parcel).

usage: json_decoder.py <isa_query> <tsp_sim> <config.json> [items] [seed]
"""
import json, os, random, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENC = os.path.join(HERE, "..", "..", "isa-db", "generated", "encoding.json")


def span(loc):
    a, _, b = loc.partition("-")
    return int(a), int(b or a)


class JsonDecoder:
    def __init__(self, db, cfg):
        self.db, self.cfg = db, cfg
        self.rule = db["decode_rule"]
        self.ctl = db["control_channel"]
        self.signed = set(db["field_rule"]["signed_kinds"])
        self.pvars = {}
        for n, d in db["profile_variables"].items():
            val = 1
            for f in d["expr"].split("*"):
                f = f.strip()
                base, _, key = f.partition("^")
                if f.isdigit():
                    val *= int(f)
                else:
                    x = cfg
                    for k in (key if key else f).strip().split("."):
                        x = x[k]
                    val *= int(base) ** x if key else x
            self.pvars[n] = val
        self.units = {u: set(q) for u, q in db["units"].items() if u != "text"}
        self.qname = {q: n for n, q in db["queues"]["names"].items()}

    def field(self, v, w):
        hi, lo = span(v["location"])
        n = hi - lo + 1
        x = (w >> lo) & ((1 << n) - 1)
        for key in ("hi_location", "top_location"):
            if key in v:
                a, b = span(v[key])
                x |= ((w >> b) & ((1 << (a - b + 1)) - 1)) << n
                n += a - b + 1
        return x, n

    def legal(self, e, fields):
        for v in e["variables"]:
            x, n = fields[v["name"]]
            k = self.db["operand_kinds"][v["kind"]]
            val = x - (1 << n) if v["kind"] in self.signed and x >> (n - 1) else x
            if "codes" in v and x not in v["codes"]:
                return False
            if "values" in v and x not in v["values"]:
                return False
            for ext, ext_codes in v.get("extension_codes", {}).items():
                if x in ext_codes and ext not in self.cfg.get("extensions", []):
                    return False
            if x in v.get("not_values", []):
                return False
            if "min" in v and val < v["min"]:
                return False
            if "max" in v and val > v["max"]:
                return False
            if "max_expr" in v:
                name, _, off = v["max_expr"].partition("-")
                bound = self.pvars[name.strip()] - (int(off) if off else 0)
                if v["kind"] in self.signed:
                    if abs(val) > bound:
                        return False
                elif x > bound:
                    return False
            if "bool_values" in v and "type" in fields and fields["type"][0] in self.db["boolean_codes"] \
                    and x not in v["bool_values"]:
                return False
            if k["syntax"] == "name" and x not in k["names"].values():
                return False
            if k["syntax"] == "stream":
                g = k.get("group", 1)
                if (x * g) >> (self.db["stream_encoding"]["row_bit"] + 1):
                    return False
        return True

    def text(self, e, fields):
        out = e["assembly"]
        se = self.db["stream_encoding"]
        rows = {c: r for r, c in se["row_codes"].items()}
        for v in e["variables"]:
            x, _ = fields[v["name"]]
            k = self.db["operand_kinds"][v["kind"]]
            if k["syntax"] == "name":
                t = {c: n for n, c in k["names"].items()}[x]
            elif k["syntax"] == "stream":
                x *= k.get("group", 1)
                t = f"{rows[(x >> se['row_bit']) & 1]}:{x & ((1 << se['row_bit']) - 1)}"
            elif k["syntax"] == "dec":
                t = str(x)
            else:
                t = hex(x)
            out = out.replace("{" + v["name"] + "}", t)
        return out

    def decode_word(self, width, w):
        hits = []
        for e in self.db["instructions"]:
            if e["width"] != width:
                continue
            m = e["match"]
            fixed = int(m.replace("-", "0"), 2)
            mask = int("".join("0" if c == "-" else "1" for c in m), 2)
            if w & mask != fixed:
                continue
            fields = {v["name"]: self.field(v, w) for v in e["variables"]}
            if self.legal(e, fields):
                hits.append((e, fields))
        assert len(hits) <= 1, [h[0]["name"] for h in hits]
        return hits[0] if hits else None

    def text_decode(self, q, units):
        """(diagnostic or None, units consumed, text) of the instruction that starts a queue's text
        at units[0] (units[1] for a two-unit layout): decode_rule of encoding.json."""
        r = self.rule
        lo, hi = span(r["tag_bits"])
        u0 = units[0] & ~(1 << r["bundle_bit"]) & 0xFFFFFFFF
        tag = (u0 >> hi) & ((1 << (lo - hi + 1)) - 1)
        if tag in r["narrow_tags"]:
            width, w, n = 32, u0, 1
        elif tag in r["wide_tags"]:
            width, w, n = 64, (u0 << 32) | units[1], 2
        else:
            return r["no_match"], 1, None
        d = self.decode_word(width, w)
        if d is None:
            return r["no_match"], n, None
        e, f = d
        if q == self.ctl["queue"]:
            bad = e["name"] not in self.ctl["instructions"]
        else:
            bad = not any(q in self.units[u] for u in e["units"])
        return ("OTS-E-QUE-002" if bad else None), n, self.text(e, f)


def main(argv):
    iq, sim, cfgp = argv[1], argv[2], argv[3]
    n = int(argv[4]) if len(argv) > 4 else 3000
    rnd = random.Random(int(argv[5]) if len(argv) > 5 else 1)
    db = json.load(open(ENC))
    dec = JsonDecoder(db, json.load(open(cfgp)))
    entries = db["instructions"]
    queues = list(range(144)) + [db["control_channel"]["queue"]]
    items = []                 # (queue, units)
    for i in range(n):
        q = rnd.choice(queues) if i % 4 else db["control_channel"]["queue"] if i % 8 == 0 else rnd.randrange(144)
        k = i % 5
        e = rnd.choice(entries)
        w = int(e["match"].replace("-", "0"), 2) | (rnd.getrandbits(e["width"]) & int(e["match"].replace("1", "0").replace("-", "1"), 2))
        if k == 1:
            w ^= 1 << rnd.choice([j for j, c in enumerate(reversed(e["match"])) if c != "-"])
        elif k == 2:
            w = rnd.getrandbits(64) if rnd.random() < 0.5 else rnd.getrandbits(32)
        elif k == 3 and e["width"] == 32:
            w |= rnd.getrandbits(32) << 32
        # The text of the word: a 32-bit layout is one unit; a 64-bit layout two, high half first
        # (bit 31 of the first unit is the bundle bit: always 0 here, so the instruction is alone).
        units = [w & 0xFFFFFFFF] if w >> 32 == 0 and i % 7 else [(w >> 32) & 0x7FFFFFFF, w & 0xFFFFFFFF]
        if units[-1] == 0xFFFFFFFF:        # the padding unit is never enqueued
            units[-1] = 0
        units[0] &= 0x7FFFFFFF
        items.append((q, units))
    # Text: isa_query's `text` decode.
    p = subprocess.Popen([iq, cfgp], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    mine = [dec.text_decode(q, u + [0]) for q, u in items]
    items = [(q, (u + [0])[:m[1]]) for (q, u), m in zip(items, mine)]   # the units the instruction consumes (a wide tag takes the next unit: 0 here)
    out, _ = p.communicate("".join(f"text {u[0]:x} {(u + [0])[1]:x}\n" for q, u in items))
    errors = []
    for (q, u), m, ln in zip(items, mine, out.splitlines()):
        code, nu, t = ln.split("\t", 2)
        if (None if code == "-" else code) != (m[0] if m[0] == "OTS-E-DEC-003" else None) or nu != str(m[1]) or \
                (t != m[2] and m[2] is not None):
            errors.append(f"text {q:x} {[hex(x) for x in u]}: json decoder {m}, isa_query ({code}, {nu}, {t!r})")
    # Diagnostics: tsp_sim. An instruction the decoder rejects changes no state, so the rejects of
    # a queue run in one image, one parcel each (parcel k dispatches at cycle k); an instruction
    # that dispatches can (a control Sync parks every queue, a NOP keeps its queue busy), so each
    # runs alone (4 processes at a time).
    sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
    from tsp_as import q_lines

    def diags(imgs):
        out = subprocess.run([sim, cfgp, "--trace"], input="\n".join(q_lines(imgs)) + "\n",
                             capture_output=True, text=True, timeout=1200).stdout
        got = {}
        for ln in out.splitlines():
            t = ln.split()
            if ln.startswith("T diag") and t[5].startswith(("OTS-E-DEC", "OTS-E-QUE-002")):
                got[(int(t[2]), int(t[4]))] = t[5]
        return got
    rejects, expect, imgs = {}, {}, {}
    for (q, u), m in zip(items, mine):
        if m[0] is None:
            continue
        c = len(rejects.get(q, []))
        if len(imgs.get(q, [])) + len(u) > 300:       # icu.iq_depth / 4 >= 320 units
            continue
        rejects.setdefault(q, []).append((q, u))
        imgs.setdefault(q, []).extend(u)
        expect[(c, q)] = m[0]
    model = diags({q: imgs[q] for q in sorted(imgs, key=lambda q: (q == 0xFFFF, q))})
    for key, code in expect.items():
        if model.get(key) != code:
            errors.append(f"cycle {key[0]} queue {key[1]:x}: json decoder {code}, model {model.get(key)}")
    for extra in set(model) - set(expect):
        errors.append(f"cycle {extra[0]} queue {extra[1]:x}: model {model[extra]}, json decoder none")
    alone = [(q, u, m) for (q, u), m in zip(items, mine) if m[0] is None]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        for (q, u, m), got in zip(alone, ex.map(lambda r: diags({r[0]: r[1]}), alone)):
            got = {k: v for k, v in got.items() if k[0] == 0}      # the instruction's own dispatch cycle (a fetch delivers more text later)
            if got:
                errors.append(f"text {q:x} {[hex(x) for x in u]}: json decoder {m[0]}, model {got}")
    mine = [m for m in mine if m[0] in (None, "OTS-E-QUE-002")]
    ndec = len(mine)
    print(f"json-only decoder ({os.path.basename(cfgp)}): {len(items)} text items ({ndec} decoded), {len(errors)} differences from the model")
    for e in errors[:20]:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
