"""Operand stream footprints in isa-db/generated/encoding.json.

Every stream operand of every catalogue entry carries `streams` (how many consecutive streams
it uses from its base) and `align` (the alignment of the base). This check evaluates them for
every legal value of the fields they depend on (element types, the group bit, the tile size)
and compares the streams covered with the Python oracle's operand model (test/oracle/diff_oracle.py
`operands`), which the differential suite compares with the Sail model. Each operand is placed at an aligned base
on a row its unit can read (or write), using the `stations` table of encoding.json.

usage: check_footprints.py <config.json>
"""
import itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "oracle"))
sys.path.insert(0, os.path.join(HERE, ".."))
import diff_oracle as d                      # noqa: E402
from oracle import machine as mc, numerics as nx   # noqa: E402

cfg = json.load(open(sys.argv[1]))
enc = json.load(open(os.path.join(HERE, "..", "..", "isa-db", "generated", "encoding.json")))
m = mc.Machine(cfg)
WIDTH = dict(nx.WIDTH)


def evaluate(expr, f):
    if isinstance(expr, int):
        return expr
    if expr.startswith("sg4 ? 4 : "):
        return 4 if f.get("sg4") else evaluate(expr[len("sg4 ? 4 : "):], f)
    if expr == "n * n":
        return f["n"] * f["n"]
    if expr.startswith("width(") and expr.endswith(")"):
        v = f[expr[6:-1]]
        return WIDTH[v if isinstance(v, str) else nx.TYPE_NAME[v]]
    if expr.startswith("mxm_width(") and expr.endswith(")"):
        return 2 if mc.MXM_WT[f[expr[10:-1]]] in mc.PAIR else 1     # f16, bf16 and bfp8: a pair of planes
    raise SystemExit(f"check_footprints: unknown expression {expr!r}")


def queue_of(units):
    u = "VXM" if "any" in units else units[0]
    return u, enc["units"][u][0]


errors, checked = [], 0
for e in enc["instructions"]:
    svars = [v for v in e["variables"] if "streams" in v]
    if not svars:
        continue
    unit, q = queue_of(e["units"])
    st = enc["stations"][unit]
    # the discrete fields the footprints depend on, with every legal value
    choices = {}
    for v in e["variables"]:
        if v["kind"] == "eltype_code":
            choices[v["name"]] = [nx.TYPE_NAME[c] for c in v["codes"]]
        elif v["kind"] == "eltype4":
            choices[v["name"]] = v["codes"] if "codes" in v else list(range(13)) + [16]
        elif v["kind"] == "group4":
            choices[v["name"]] = [0, 1]
        elif v["kind"] == "tile_size":
            choices[v["name"]] = v["values"]
        elif v["kind"] in ("mxm_wtype", "mxm_acctype"):
            choices[v["name"]] = v.get("values", [0, 1, 2, 3, 4] if v["kind"] == "mxm_wtype" else [2, 3])
    names = list(choices)
    reads = {"lhs", "rhs", "mask", "src", "src2", "map", "a", "b", "c", "sel"}
    for combo in itertools.product(*[choices[n] for n in names]):
        f = dict(zip(names, combo))
        # place each stream operand at an aligned base on a usable row, 8 streams apart per row
        nxt = {"E": 0, "W": 0}
        bases = {}
        ok = True
        for v in svars:
            is_read = v["name"] in reads or (v["name"] == "stream" and e["name"] in ("Ifetch", "Send", "PciSend"))
            rows = [r for r in "EW" if st["read" if is_read else "write"][r]]
            if not rows:
                ok = False
                break
            r = min(rows, key=lambda x: nxt[x])
            n, a = evaluate(v["streams"], f), evaluate(v["align"], f)
            base = -(-nxt[r] // max(a, 1)) * max(a, 1)
            if v["kind"] == "stream_group16":
                base = -(-nxt[r] // 16) * 16
            if base + n > 32:
                ok = False
                break
            bases[v["name"]] = (r, base, n, is_read)
            nxt[r] = base + max(n, 8)
            # a 16-stream group field holds the stream reference / 16 (row bit and half)
            f[v["name"]] = d.sref(r, base) >> 4 if v["kind"] == "stream_group16" else d.sref(r, base)
        if not ok:
            errors.append(f"{e['name']} {f}: cannot place the operands on {unit}'s rows")
            continue
        for v in e["variables"]:
            if v["kind"] in ("chain",):
                f.setdefault(v["name"], 0)
            elif v["name"] not in f:
                f[v["name"]] = (v.get("values") or [v.get("min", 0) or 0])[0] if v["kind"] not in ("imm16", "imm14", "imm12", "imm6", "imm5", "imm4", "imm1") else 1
        if e["name"] == "Rotate":
            f["n"] = combo[names.index("n")]
        R, W = d.operands(m, q, e["name"], f, 100, 0)
        covered = {}
        for (s0, w, *_rest) in R:
            for b in range(w):
                covered.setdefault(("r", s0 & 32, (s0 & 31) + b), True)
        for (s0, w, *_rest) in W:
            for b in range(w):
                covered.setdefault(("w", s0 & 32, (s0 & 31) + b), True)
        for name, (r, base, n, is_read) in bases.items():
            row = 32 if r == "W" else 0
            got = sorted(i for (k, rb, i) in covered if k == ("r" if is_read else "w") and rb == row and base <= i < base + 16
                         and all(not (bb != base and bb <= i and rr == r) for nm, (rr, bb, _n, _ir) in bases.items() if nm != name and bb > base))
            want = list(range(base, base + n))
            checked += 1
            if got != want:
                errors.append(f"{e['name']} {dict((k, f[k]) for k in names)}: {name} covers {got}, footprint says {want}")

print(f"operand footprints ({os.path.basename(sys.argv[1])}): {checked} operands checked, {len(errors)} errors")
for x in errors[:30]:
    print("  " + x)
sys.exit(1 if errors else 0)
