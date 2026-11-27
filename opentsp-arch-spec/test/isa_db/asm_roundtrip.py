"""Assembly round trip through the reference assembler (tools/tsp_as.py), both directions.

The Sail `assembly` mapping prints; Sail 0.20.3 cannot execute its string -> instruction
direction (assembly.md sec. 1), so acceptance is defined by the written rules of
assembly.md, which tools/tsp_as.py implements from isa-db/generated/encoding.json.

Accept direction: for every catalogue entry, at its corners and on random legal field
values, the model prints the word; tsp_as must parse that text with exactly one entry,
encode it to the same word, and disassemble the word to the same text.

Reject direction: for every field, values outside its range (a stream id
above 31, a group base other than 0/16, a value wider than the field, an excluded or
not-listed value, a step outside -8191..8191, a shift amount at or above the configured
lane count) must be rejected by tsp_as, never wrapped; and the word that a wrapping
encoder would build for an excluded value must not decode in the model.

usage: asm_roundtrip.py <isa_query> <config.json> [samples] [seed]
"""
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import check_isa_db as C
import tsp_as as A


def bad_tokens(isa, e, v):
    """(token, why) pairs that are out of range for field v of entry e."""
    k = isa.kinds[v["kind"]]
    n = A.field_bits(v)
    out = []
    if k["syntax"] == "stream":
        out += [("E:32", "stream > 31"), ("W:99", "stream > 31"), ("E:007" if k.get("group") else "E:40", "range")]
        if k.get("group"):
            out += [("E:5", "not a group base"), ("W:31", "not a group base"), ("E:32", "> 31")]
            out = [o for o in out if o[0] != "E:007"]
        return out
    if k["syntax"] == "name":
        legal = v.get("values", v.get("codes"))
        if legal is not None:
            inv = {c: t for t, c in k["names"].items()}
            out += [(inv[c], "code not legal here") for c in inv if c not in legal]
        return out
    if k["syntax"] == "dec":
        return [(str(1 << n), "wider than field"), ("10", "wider than field")]
    if k.get("signed"):
        return [(hex((v["min"] - 1) & 0xFFFF), "below min"), (hex(v["max"] + 1), "above max"), (hex(1 << n), "wider than field")]
    out.append((hex(1 << n), "wider than field"))
    out.append(("0x" + "0" * 3 + hex((1 << n) + 1)[2:], "wider than field, leading zeros"))
    out += [(hex(x), "excluded") for x in v.get("not_values", [])]
    if "max" in v and v["max"] + 1 < (1 << n):
        out.append((hex(v["max"] + 1), "above max"))
    if "values" in v:
        out += [(hex(x), "not listed") for x in range(min(1 << n, 64)) if x not in v["values"]][:6]
    if "max_expr" in v and isa.max_of(v) + 1 < (1 << n):
        out += [(hex(isa.max_of(v) + 1), "above max_expr (the configured lane count)"), (hex((1 << n) - 1), "above max_expr")]
    return out


def main(argv):
    exe, cfg = argv[1], argv[2]
    samples = int(argv[3]) if len(argv) > 3 else 20
    rnd = random.Random(int(argv[4]) if len(argv) > 4 else 1)
    profile = os.path.splitext(os.path.basename(cfg))[0]
    isa = A.ISA(profile, cfg)
    vlanes = isa.pvars              # the profile variables that bound a field
    m = C.Model(exe, cfg)
    errors, n, nrej = [], 0, 0
    for e in isa.entries:
        ins = {"encoding": {"match": e["match"], "variables": e["variables"]}}
        width = e["width"]
        trials = []
        for corner in (min, max):
            fv = {}
            for v in e["variables"]:
                vals = C.legal_values(v, vlanes)
                if vals is None:
                    vals = C.corner_values(v, vlanes)
                fv[v["name"]] = corner(vals)
            trials.append(fv)
        trials += [{v["name"]: C.pick(v, rnd, vlanes) for v in e["variables"]} for _ in range(samples)]
        base = None
        for fv in trials:
            w = C.build(ins, fv)
            ok, text = m.query(width, w)
            if text.startswith("illegal"):
                continue          # a joint constraint (boolean cmp modes); covered by check_isa_db.py
            base = base or (fv, text)
            n += 1
            try:
                got, gw = isa.parse(text)
            except A.AsmError as err:
                errors.append(f"{e['name']}: model text {text!r} rejected: {err}")
                break
            if got["name"] != e["name"] or gw != w:
                errors.append(f"{e['name']}: {text!r} assembles to {got['name']} {gw:#x}, model word {w:#x}")
                break
            if isa.disasm(width, w) != text:
                errors.append(f"{e['name']}: disassembly {isa.disasm(width, w)!r} != model {text!r}")
                break
        if base is None:
            errors.append(f"{e['name']}: no decodable sample")
            continue
        # Reject direction: substitute one bad operand into the template of a legal text.
        fv, _ = base
        for v in e["variables"]:
            for tok, why in bad_tokens(isa, e, v):
                text = e["assembly"]
                for u in e["variables"]:
                    t = tok if u is v else C.render(u["kind"], fv[u["name"]], width)
                    text = text.replace("{" + u["name"] + "}", t)
                nrej += 1
                try:
                    got, gw = isa.parse(text)
                    errors.append(f"{e['name']}: out-of-range {text!r} ({why}) accepted as {gw:#x}")
                except A.AsmError:
                    pass
            # A wrapping encoder's word for an excluded value must not decode in the model either.
            for x in v.get("not_values", []):
                w = C.build(ins, dict(fv, **{v["name"]: x}))
                _, t = m.query(width, w)
                if not t.startswith("illegal") and t == C.expected_asm({"assembly": e["assembly"], **ins}, dict(fv, **{v["name"]: x})):
                    errors.append(f"{e['name']}: excluded {v['name']}={x} decodes in the model as {t!r}")
                if isa.decode(width, w) is not None and isa.decode(width, w)[0]["name"] == e["name"]:
                    errors.append(f"{e['name']}: excluded {v['name']}={x} decodes in tsp_as")
    # Decode direction: words with random field bits (not only legal values) and with one
    # fixed bit flipped; the disassembler and the model must print the same text, including
    # "illegal<width>" for every word that does not decode.
    ndis = 0
    for e in isa.entries:
        width = e["width"]
        fixed = int(e["match"].replace("-", "0"), 2)
        free = int(e["match"].replace("1", "0").replace("-", "1"), 2)
        for k in range(samples * 4):
            w = fixed | (rnd.getrandbits(width) & free)
            if k % 2:
                w ^= 1 << rnd.choice([i for i, c in enumerate(reversed(e["match"])) if c != "-"])
            if width == 64 and isa.width_of(w) == 32:
                continue           # the record decoder would treat it as a 32-bit word
            _, t = m.query(width, w)
            ndis += 1
            if isa.disasm(width, w) != t:
                errors.append(f"{e['name']}: word {w:#x}: tsp_as prints {isa.disasm(width, w)!r}, model {t!r}")
                break
    m.p.stdin.close()
    m.p.wait()
    print(f"assembly round trip ({profile}): {len(isa.entries)} entries, {n} texts accepted, {nrej} out-of-range texts rejected, "
          f"{ndis} words disassembled as the model does, {len(errors)} errors")
    for e in errors[:30]:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
