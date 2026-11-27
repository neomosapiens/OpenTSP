#!/usr/bin/env python3
"""Measure the domains of the reciprocal idioms of division.py by running them on the ISS.

usage: division_domains.py [--iss PATH] [--asm PATH] [--check] [--out FILE]

The measurement is one vector of `vlanes` positive f32 values per binade 2^e (e = -149 .. 127: the subnormal binades
included; lane 0 and 1 are the two ends of the binade, the others random with a fixed seed), run through every idiom on
the ISS (ots-asm and tsp-iss from target/release). Per binade it records the largest error in ulp of the exact quotient
(in float64) over the lanes, and whether any result is not finite or is zero. The numpy twin must give the same bits
(any difference is an error: the golden programs rely on it). The edges where the result becomes inf or 0 are then
found to the last bit by a search that runs batches of candidate values on the ISS.

Output (tests/programs/gen/division_domains.json): per idiom the error bound, the accurate binades, the largest error
inside them, the first and last finite non-zero value and the table of errors per binade. `--check` compares a new
measurement with the file (exit 1 on a difference) instead of writing it."""
import argparse
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import division  # noqa: E402
import run  # noqa: E402
from opentsp_tools import profile as tprofile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RESULT = os.path.join(HERE, "division_domains.json")
BOUND = {"rsqrt_sq": 1.0, "rsqrt_pair": 2.0, "newton_sq": 1.0, "newton_pair": 2.0}   # claimed error, in ulp
EXPONENTS = list(range(-149, 128))
GROUPS_PER_PROGRAM = 40


def exact(idiom, x):
    """1/|x| for rsqrt_sq (the sign is lost), 1/x otherwise, in float64 (relative error 2^-53)."""
    x64 = x.astype(np.float64)
    with np.errstate(all="ignore"):
        return 1.0 / (np.abs(x64) if idiom == "rsqrt_sq" else x64)


def ulp_error(idiom, got, x):
    """|got - exact| in ulps of the exact value rounded to f32 (2^-149 in the subnormal range); inf where got is not
    finite or exact overflows."""
    ex = exact(idiom, x)
    g = got.astype(np.float64)
    sp = np.spacing(np.abs(np.minimum(ex, 3.0e38)).astype(np.float32)).astype(np.float64)
    sp = np.maximum(sp, 2.0 ** -149)
    with np.errstate(all="ignore"):
        err = np.abs(g - ex) / sp
    return np.where(np.isfinite(g) & np.isfinite(ex), err, np.inf)


def binade_vectors(n, seed=7):
    """One vector of n values (bit patterns) per binade of EXPONENTS, the first two lanes the ends of the binade."""
    rng = np.random.default_rng(seed)
    out = []
    for e in EXPONENTS:
        if e >= -126:
            lo, width = (e + 127) << 23, 1 << 23
        else:
            lo, width = 1 << (e + 149), 1 << (e + 149)
        b = lo + rng.integers(0, width, size=n, dtype=np.int64)
        b[0], b[1] = lo, lo + width - 1
        out.append(b.astype(np.uint32))
    return out


def as_f32(bits):
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


class Iss:
    """Runs idiom groups on the ISS in programs of at most GROUPS_PER_PROGRAM groups."""

    def __init__(self, iss, asm, profile="ref", sim=None):
        self.p = tprofile.load(profile)
        self.iss, self.sim = iss, sim          # `sim`: tsp_sim (the Sail model); every program then runs there too and must agree
        self.runner = run.Runner(types.SimpleNamespace(asm=asm, timeout=600))

    def run(self, groups):
        """The results (f32 vectors) of groups [(idiom, x[, seed])], in order; raises on any diagnostic."""
        results = []
        for first in range(0, len(groups), GROUPS_PER_PROGRAM):
            chunk = groups[first:first + GROUPS_PER_PROGRAM]
            k, wants = division.program(self.p, "division/measure", chunk, "measurement")
            text, case = k.done("measure")
            work = tempfile.mkdtemp(prefix="div-", dir=os.environ.get("TMPDIR", "/tmp"))
            open(os.path.join(work, case["program"]), "w").write(text)
            text_input, native = self.runner.build_input(case, work, work)
            out = self.runner.simulate_native(self.iss, case, native, work)
            if out["diags"]:
                raise RuntimeError(f"diagnostics {out['diags'][:3]}")
            if self.sim:
                ref = self.runner.simulate(self.sim, case, text_input, work, "tsp_sim")
                if ref["diags"] or ref["words"] != {k: v for k, v in out["words"].items() if k in ref["words"]} \
                        or set(ref["words"]) != set(out["words"]):
                    raise RuntimeError(f"tsp_sim and the ISS differ in the programs of groups {first}..{first + len(chunk) - 1}")
            for j, want in enumerate(wants):
                planes = [np.frombuffer(out["words"][f"{44 + i}:{division.ADDR + j}"][:self.p.vlanes], dtype=np.uint8)
                          for i in range(4)]
                got = np.stack(planes, axis=1).copy().view("<u4").reshape(-1)
                if not (got == want.view(np.uint32)).all():
                    raise RuntimeError(f"{chunk[j][0]}: the ISS and the numpy twin differ in group {first + j}")
                results.append(got.view(np.float32))
            for f in os.listdir(work):
                os.remove(os.path.join(work, f))
            os.rmdir(work)
        return results


def evaluate(iss, idiom, vecs):
    """The results of idiom on the vectors `vecs` (bit patterns), the seed of a Newton step taken from the ISS too."""
    xs = [as_f32(v) for v in vecs]
    if idiom in division.SEED:
        seeds = iss.run([(division.SEED[idiom], x) for x in xs])
        return iss.run([(idiom, x, y) for x, y in zip(xs, seeds)])
    return iss.run([(idiom, x) for x in xs])


def usable(r):
    return np.isfinite(r) & (r != 0)


def first_flip(iss, idiom, lo, hi, to):
    """The smallest bit pattern b in (lo, hi] whose result has usable(r) == to, given that it is not `to` at lo and is
    `to` at hi (a monotone edge). Batches of 8 vectors run on the ISS until the interval is one pattern wide."""
    n = iss.p.vlanes
    while hi - lo > 1:
        cand = np.unique(np.linspace(lo + 1, hi, min(8 * n, hi - lo), dtype=np.int64))
        pad = np.concatenate([cand, np.full(-len(cand) % n, cand[-1])]).astype(np.uint32)
        res = np.concatenate(evaluate(iss, idiom, pad.reshape(-1, n)))[:len(cand)]
        i = int(np.flatnonzero(usable(res) == to)[0])
        hi, lo = int(cand[i]), (int(cand[i - 1]) if i else lo)
    return hi


def binade_bits(e):
    """(first, last) bit pattern of the binade 2^e of positive f32 values."""
    if e >= -126:
        return (e + 127) << 23, ((e + 128) << 23) - 1
    return 1 << (e + 149), (1 << (e + 150)) - 1


def usable_edges(iss, idiom, ok_binades):
    """The first and the last bit pattern whose result is finite and non-zero, to the last bit; the binades in
    `ok_binades` are those whose every measured result is."""
    e0, e1 = ok_binades[0], ok_binades[-1]
    first = binade_bits(e0)[0]
    if e0 > EXPONENTS[0]:
        below = binade_bits(e0 - 1)[0]
        assert not usable(evaluate(iss, idiom, [np.full(iss.p.vlanes, below, dtype=np.uint32)])[0])[0]
        first = first_flip(iss, idiom, below, first, True)
    last = binade_bits(e1)[1]
    if e1 < EXPONENTS[-1]:
        above = binade_bits(e1 + 1)[1]
        assert not usable(evaluate(iss, idiom, [np.full(iss.p.vlanes, above, dtype=np.uint32)])[0])[0]
        last = first_flip(iss, idiom, last, above, False) - 1
    return first, last


def measure(iss):
    n = iss.p.vlanes
    xs = [as_f32(b) for b in binade_vectors(n)]
    first = [("rsqrt_sq", x) for x in xs] + [("rsqrt_pair", x) for x in xs]
    got = iss.run(first)
    res = {"rsqrt_sq": got[:len(xs)], "rsqrt_pair": got[len(xs):]}
    seeded = [("newton_sq", x, y) for x, y in zip(xs, res["rsqrt_sq"])] + \
             [("newton_pair", x, y) for x, y in zip(xs, res["rsqrt_pair"])]
    got = iss.run(seeded)
    res["newton_sq"], res["newton_pair"] = got[:len(xs)], got[len(xs):]
    out = {"vlanes": n, "binades": [EXPONENTS[0], EXPONENTS[-1]], "idioms": {}}
    for idiom in division.IDIOMS:
        errs = [float(ulp_error(idiom, r, x).max()) for r, x in zip(res[idiom], xs)]
        good = [e for e, err in zip(EXPONENTS, errs) if err <= BOUND[idiom]]
        lo_e, hi_e = good[0], good[-1]
        assert good == list(range(lo_e, hi_e + 1)), f"{idiom}: the accurate binades are not contiguous"
        inside = [err for e, err in zip(EXPONENTS, errs) if lo_e <= e <= hi_e]
        u = [e for e, r in zip(EXPONENTS, res[idiom]) if usable(r).all()]
        assert u == list(range(u[0], u[-1] + 1)), f"{idiom}: the usable binades are not contiguous"
        first, last = usable_edges(iss, idiom, u)
        out["idioms"][idiom] = {"bound_ulp": BOUND[idiom], "accurate_binades": [lo_e, hi_e],
                                "max_ulp_accurate": round(max(inside), 3), "usable_binades": [u[0], u[-1]],
                                "first_usable_bits": f"{first:#010x}", "last_usable_bits": f"{last:#010x}",
                                "max_ulp_by_binade": [round(e, 3) if np.isfinite(e) else None for e in errs]}
    return out


def render_table(domains):
    """The Markdown table of  ('Division and square root'), from the measurement."""
    def num(bits):
        return f"{float(as_f32([bits])[0]):.6g}"

    def pow2(e):
        return f"2^{e}"
    rows = ["| idiom | accurate for x in | error bound | worst error measured | result finite and non-zero for x in |",
            "|---|---|---|---|---|"]
    for idiom in division.IDIOMS:
        d = domains["idioms"][idiom]
        lo_e, hi_e = d["accurate_binades"]
        top = hi_e == EXPONENTS[-1]
        lo = binade_bits(lo_e)[0]
        hi = 0x7F7FFFFF if top else binade_bits(hi_e + 1)[0]
        acc = f"[{pow2(lo_e)}, {'largest f32]' if top else pow2(hi_e + 1) + ')'} = [{num(lo)}, {num(hi)}{']' if top else ')'}"
        use = f"[{num(int(d['first_usable_bits'], 16))}, {num(int(d['last_usable_bits'], 16))}] (`{d['first_usable_bits']}` .. `{d['last_usable_bits']}`)"
        rows.append(f"| `{idiom}` | {acc} | {d['bound_ulp']:g} ulp | {d['max_ulp_accurate']:.3f} ulp | {use} |")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iss", default=os.path.join(ROOT, "target", "release", "tsp-iss"))
    ap.add_argument("--asm", default=os.path.join(ROOT, "target", "release", "ots-asm"))
    ap.add_argument("--sim", help="also run every program on this tsp_sim and require the same words (slow: minutes)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=RESULT)
    ap.add_argument("--table", action="store_true", help="print the Markdown table of the checked-in measurement")
    a = ap.parse_args()
    if a.table:
        print(render_table(json.load(open(a.out))), end="")
        return 0
    got = measure(Iss(a.iss, a.asm, sim=a.sim))
    text = json.dumps(got, indent=1) + "\n"
    if a.check:
        if open(a.out).read() != text:
            print("division_domains: the measurement differs from", a.out)
            return 1
        print("division_domains: measurement equals", a.out)
        return 0
    open(a.out, "w").write(text)
    for idiom, d in got["idioms"].items():
        print(idiom, {k: v for k, v in d.items() if k != "max_ulp_by_binade"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
