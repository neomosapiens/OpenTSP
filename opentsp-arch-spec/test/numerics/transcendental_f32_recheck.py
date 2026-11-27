#!/usr/bin/env python3
"""Reference side of the exhaustive f32 transcendental check.

Two sub-commands, driven by tools/check_transcendental_f32.sh:

  recheck.py jobs  --hard hard.txt --stride S --seed N --samples K --out DIR [--procs P]
      Builds the list of inputs the Sail evaluator must be compared on:
        H  every "hard" input found by the double-precision sweep,
        E  every special / edge class (NaN payloads, +-Inf, +-0, subnormals, powers of two and
           their neighbours, rounding and shortcut thresholds of the model, small integers),
        R  a seeded random sample (uniform over all bit patterns, and stratified over the
           exponent range where the result is neither trivial nor saturated).
      For each input the correctly rounded f32 result is computed with mpmath (200 bits, re-done
      at doubled precision whenever the value is too close to a rounding boundary to decide), and
      the relative distance of the exact value to the nearest rounding boundary is recorded.
      Writes DIR/jobs.txt (the Sail driver's job list), DIR/recheck.json.

  recheck.py report --dir DIR --sail OUT1 [OUT2 ...] [--sweep-stats stats.txt]
      Merges the Sail driver outputs and the sweep statistics with recheck.json, prints the
      per-function table and exits non-zero on any failure.

The expected value of a special follows the model's rules (opentsp-arch-spec/docs/
microarchitecture.md sec. 5, core/numerics/transcendental.sail fp_tr_value):
  NaN -> canonical NaN 0x7FC00000.  +Inf: TanH 1, Exp +Inf, RSqrt +0, Log +Inf.
  -Inf: TanH -1, Exp +0, RSqrt NaN, Log NaN.  +-0: TanH +-0, Exp 1, RSqrt +-Inf, Log -Inf.
  A negative finite input to RSqrt or Log -> NaN.  Log(1) = +0.
Finite results are rounded once, to nearest even, with gradual underflow (no flush) and
overflow to +-Inf at 2^128 - 2^103.
"""
import argparse
import json
import math
import multiprocessing as mp_
import os
import random
import struct
import sys

import mpmath
from mpmath import mp

TANH, EXP, RSQRT, LOG = 21, 22, 23, 24
OPS = (TANH, EXP, RSQRT, LOG)
NAME = {TANH: "TanH", EXP: "Exp", RSQRT: "RSqrt", LOG: "Log"}
CANON_NAN = 0x7FC00000
INF = 0x7F800000
E_MAX = 900            # accepted closeness: relative distance >= 2^-(24 + E_MAX)
FILTER_K = 41          # a non-hard input is farther than 2^-41 from every boundary
START_PREC = 200
MAX_PREC = 3300        # resolves k up to about 3280 bits


def f32_of_bits(x):
    return struct.unpack("<f", struct.pack("<I", x))[0]


class NeedMore(Exception):
    pass


def eval_exact(op, xf, prec):
    """The function value (an mpf with prec bits) for the f32 value xf (a Python float)."""
    mp.prec = prec
    x = mpmath.mpf(xf)
    if op == TANH:
        return mpmath.tanh(x)
    if op == EXP:
        return mpmath.exp(x)
    if op == RSQRT:
        return 1 / mpmath.sqrt(x)
    return mpmath.log(x)


def round_f32(v, prec):
    """Round the mpf v (relative error about 2^-prec) to f32 RNE. Returns (bits, k) where k is
    -log2 of the relative distance of v to the nearest rounding boundary (k = 0 means 'about
    half a ulp away or more'). Raises NeedMore if v is too close to a boundary to decide."""
    if v == 0:
        raise NeedMore("zero")
    neg = v < 0
    man, exp = mpmath.libmp.to_man_exp(v._mpf_)[0:2]
    man = abs(man)
    L = man.bit_length() - 1 + exp                   # a = man 2^exp in [2^L, 2^(L+1))
    q = max(L - 23, -149)                            # quantum of the f32 grid at a
    sgn = 0x80000000 if neg else 0
    if L >= 128:                                     # a >= 2^128 > 2^128 - 2^103: rounds to Inf
        return INF | sgn, 0
    if exp < q and q - exp > man.bit_length() + 1:   # a < 1/4 of the smallest subnormal: rounds to 0
        return sgn, 0
    if exp >= q:
        n, rem, den = man << (exp - q), 0, 1
    else:
        s = q - exp
        n, rem, den = man >> s, man & ((1 << s) - 1), 1 << s
    # distance to the midpoint, in units of 2^q, as the rational |2 rem - den| / (2 den) of a unit
    twice = abs(2 * rem - den)                       # |rem/den - 1/2| * 2 den
    if twice == 0:
        raise NeedMore("tie")
    # relative distance = (twice / (2 den)) * 2^q / a = twice * 2^q / (2 den * man * 2^exp)
    # k = -log2(rel) computed from integers: log2(2 den man 2^exp / (twice 2^q))
    num_bits = (2 * den * man).bit_length() + exp - q
    k = num_bits - twice.bit_length()                # within +-1 of -log2(rel)
    if k > prec - 16:
        raise NeedMore("close")
    if 2 * rem > den:
        n += 1
    # n in [0, 2^24]; carry into the next binade is handled by the packing below
    if n >= (1 << 24):
        n >>= 1
        q += 1
    if n == 0:
        bits = 0
    elif n < (1 << 23):                              # subnormal (q == -149)
        bits = n
    else:
        biased = q + 23 + 127
        bits = INF if biased >= 255 else (biased << 23) | (n - (1 << 23))
    if neg:
        bits |= 0x80000000
    return bits, max(k, 0)


def expected_finite(op, x):
    """(expected bits, k) for a finite input x on which the function is a finite non-exact
    value; k as in round_f32, at the lowest working precision that decides."""
    xf = f32_of_bits(x)
    prec = START_PREC
    while True:
        v = eval_exact(op, xf, prec)
        try:
            return round_f32(v, prec)
        except NeedMore:
            prec *= 2
            if prec > MAX_PREC:
                return None, MAX_PREC


def expected(op, x):
    """(expected result bits, k or None) per the model's rules; k is None for specials."""
    ex = (x >> 23) & 0xFF
    frac = x & 0x7FFFFF
    neg = x >> 31
    if ex == 255:
        if frac:
            return CANON_NAN, None
        if op == TANH:
            return (0xBF800000 if neg else 0x3F800000), None
        if op == EXP:
            return (0 if neg else INF), None
        if op == RSQRT:
            return (CANON_NAN if neg else 0), None
        return (CANON_NAN if neg else INF), None
    if ex == 0 and frac == 0:
        if op == TANH:
            return x, None
        if op == EXP:
            return 0x3F800000, None
        if op == RSQRT:
            return (0xFF800000 if neg else INF), None
        return 0xFF800000, None
    if op in (RSQRT, LOG) and neg:
        return CANON_NAN, None
    if op == LOG and x == 0x3F800000:
        return 0, None
    return expected_finite(op, x)


def _work(job):
    op, x = job
    return expected(op, x)


# --------------------------------------------------------------------------------------
# input classes
# --------------------------------------------------------------------------------------
def neighbours(x, n):
    """Bit patterns x-n .. x+n (as a magnitude walk, staying within the same sign)."""
    sign = x & 0x80000000
    m = x & 0x7FFFFFFF
    return [sign | k for k in range(max(m - n, 0), min(m + n, 0x7F800000) + 1)]


def f32_near(value, n):
    """The f32 patterns within n ulps of the double `value` (and its negation)."""
    b = struct.unpack("<I", struct.pack("<f", value))[0] if abs(value) < 3.4e38 else 0x7F7FFFFF
    return neighbours(b, n) + neighbours(b | 0x80000000, n)


def edge_inputs(op, rng):
    s = set()
    # specials: zeros, infinities, NaN payload classes
    s.update([0, 0x80000000, INF, 0xFF800000])
    nan_payloads = [1, 2, 0x400000, 0x400001, 0x7FFFFF, 0x3FFFFF, 0x555555, 0x2AAAAA, 0x200000, 0x000100]
    nan_payloads += [rng.randrange(1, 1 << 23) for _ in range(256)]
    for p in nan_payloads:
        s.add(0x7F800000 | p)
        s.add(0xFF800000 | p)
    # subnormals: smallest, largest, powers of two inside, random
    subs = [1, 2, 3, 0x7FFFFF, 0x7FFFFE, 0x400000, 0x3FFFFF, 0x400001]
    subs += [1 << k for k in range(23)]
    subs += [rng.randrange(1, 1 << 23) for _ in range(300)]
    for b in subs:
        s.add(b)
        s.add(b | 0x80000000)
    # smallest and largest normals and their neighbours
    for b in (0x00800000, 0x7F7FFFFF, 0x00800001, 0x7F7FFFFE):
        for t in neighbours(b, 3):
            s.add(t)
            s.add(t | 0x80000000)
    # every power of two, +-, with neighbours (RSqrt powers of 4 are the exact results)
    for e in range(1, 255):
        for t in neighbours(e << 23, 2):
            s.add(t)
            s.add(t | 0x80000000)
    # small integers and halves, and values near 1
    for v in list(range(-300, 301)) + [k / 2 for k in range(-40, 41)] + [0.1, 0.2, 0.3, 0.7, 1.5, 2.5, math.e, math.pi]:
        for t in neighbours(struct.unpack("<I", struct.pack("<f", v))[0], 2):
            s.add(t)
    for t in neighbours(0x3F800000, 64) + neighbours(0xBF800000, 64):
        s.add(t)
    for t in neighbours(0x3F000000, 16) + neighbours(0x40000000, 16):
        s.add(t)
    # thresholds of the model and of the exact functions
    mp.prec = 100
    th = []
    th += [512.0, 511.99, 64.0, 63.99, 16.0, 8.0, 4.0]        # model shortcuts (|x| >= 512 Exp, >= 64 TanH)
    if op == EXP:
        th += [float(mpmath.log(mpmath.mpf(2) ** 128 - mpmath.mpf(2) ** 103)),      # overflow boundary
               float(-mpmath.log(mpmath.mpf(2) ** 150)),                              # underflow to 0
               float(-mpmath.log(mpmath.mpf(2) ** 149)), float(-mpmath.log(mpmath.mpf(2) ** 126)),
               float(mpmath.log(mpmath.mpf(2) ** 128)), 2.0 ** -24, 2.0 ** -25, 2.0 ** -23, 2.0 ** -26,
               88.0, 89.0, -87.0, -88.0, -103.0, -104.0, -100.0, 128.0, 127.0]
    if op == TANH:
        th += [float(mpmath.log(mpmath.mpf(2) ** 26 - 1) / 2), 9.0, 9.1, 2.0 ** -12, 2.0 ** -13, 2.0 ** -11,
               2.0 ** -25, 2.0 ** -24, 2.0 ** -20, 0.5, 1.0, 2.0, 3.0, 19.0, 20.0, 18.0, 17.0]
    if op == LOG:
        th += [2.0 ** -126, 2.0 ** -149, 3.4028235e38, 10.0, 100.0, 1e-5, 0.999, 1.001]
    if op == RSQRT:
        th += [2.0 ** -126, 2.0 ** -149, 3.4028235e38, 10.0, 100.0, 1e-5, 0.999, 1.001, 2.0, 3.0]
    for t in th:
        for u in f32_near(t, 24):
            s.add(u)
    return sorted(s)


def sample_inputs(op, rng, count):
    s = set()
    for _ in range(count):                      # uniform over all 2^32 patterns
        s.add(rng.getrandbits(32))
    lo, hi = {TANH: (100, 131), EXP: (100, 135), RSQRT: (0, 254), LOG: (0, 254)}[op]
    for _ in range(count):                      # stratified over the biased exponent range
        e = rng.randint(lo, hi)
        x = (e << 23) | rng.getrandbits(23)
        if op in (TANH, EXP) and rng.getrandbits(1):
            x |= 0x80000000
        s.add(x)
    return sorted(s)


def cmd_jobs(a):
    rng = random.Random(a.seed)
    hard = {op: [] for op in OPS}
    with open(a.hard) as f:
        for line in f:
            if line.strip():
                o, x = line.split()
                hard[int(o)].append(int(x, 16))
    classes = {}                                 # (op, x) -> class letter
    for op in OPS:
        for x in sample_inputs(op, rng, a.samples):
            classes[(op, x)] = "R"
        for x in edge_inputs(op, rng):
            classes[(op, x)] = "E"
        for x in hard[op]:
            classes[(op, x)] = "H"
    jobs = sorted(classes)
    with mp_.Pool(a.procs) as pool:
        results = pool.map(_work, jobs, chunksize=256)
    os.makedirs(a.out, exist_ok=True)
    per = {op: dict(H=0, E=0, R=0, max_k=-1, max_k_x=None, k_hist={}, unresolved=[], filter_miss=[], filter_checked=0,
                    hard_k_min=10**9) for op in OPS}
    with open(os.path.join(a.out, "jobs.txt"), "w") as out:
        for (op, x), (exp_bits, k) in zip(jobs, results):
            cl = classes[(op, x)]
            p = per[op]
            p[cl] += 1
            if exp_bits is None:
                p["unresolved"].append("%08x" % x)
                continue
            out.write("E %d %08x %08x\n" % (op, x, exp_bits))
            if k is not None:
                if cl == "H":
                    p["hard_k_min"] = min(p["hard_k_min"], k)
                    if k > p["max_k"]:
                        p["max_k"], p["max_k_x"] = k, "%08x" % x
                    b = k // 8 * 8
                    p["k_hist"][b] = p["k_hist"].get(b, 0) + 1
                elif x % a.stride == 0:
                    # a swept input that the double filter did not flag: must be farther than 2^-41
                    p["filter_checked"] += 1
                    if k > FILTER_K:
                        p["filter_miss"].append("%08x k=%d" % (x, k))
        # evaluator-accuracy probes: the precision lost against p at the hardest-conditioned inputs
        acc_inputs = {
            TANH: [1, 2, 0x007FFFFF, 0x00800000, 0x2F800000, 0x3F800000, 0x41100000, 0x427FFFFF, 0x33800000],
            EXP: [0x42B17218, 0xC2CFF1B4, 0xC2D00000, 0x43000000, 0x43FFFFFF, 0xC3FFFFFF, 0x33800000, 0xB3000000, 0x3F800000, 1],
            RSQRT: [1, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0x40800000],
            LOG: [1, 0x007FFFFF, 0x3F800001, 0x3F7FFFFF, 0x7F7FFFFF, 0x3F800000 + 7, 0x40000000, 0x3E800000],
        }
        for op in OPS:
            for x in acc_inputs[op]:
                for p_ in (64, 2048):
                    out.write("A %d %08x %x\n" % (op, x, p_))
    meta = dict(seed=a.seed, samples=a.samples, stride=a.stride, per={})
    for op in OPS:
        meta["per"][str(op)] = per[op]
    with open(os.path.join(a.out, "recheck.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    print("jobs: %d (%s)" % (len(jobs), ", ".join("%s H=%d E=%d R=%d" % (NAME[o], per[o]["H"], per[o]["E"], per[o]["R"]) for o in OPS)))


def cmd_report(a):
    meta = json.load(open(os.path.join(a.dir, "recheck.json")))
    summ = {op: dict(evals=0, mism=0, nonconv=0, levels=[0] * 7, seconds=0.0) for op in OPS}
    acc = {op: {} for op in OPS}
    problems = []
    for path in a.sail:
        for line in open(path):
            t = line.split()
            if not t:
                continue
            kv = dict(w.split("=", 1) for w in t[1:] if "=" in w)
            if t[0] == "SUMMARY":
                s = summ[int(kv["op"])]
                s["evals"] += int(kv["evals"])
                s["mism"] += int(kv["mismatches"])
                s["nonconv"] += int(kv["nonconv"])
                s["seconds"] += float(kv["seconds"])
                for i, c in enumerate(kv["levels"].split(",")):
                    s["levels"][i] += int(c)
            elif t[0] in ("MISMATCH", "NONCONV"):
                problems.append(line.strip())
            elif t[0] == "ACC":
                op, p_, ac = int(kv["op"]), int(kv["p"]), int(kv["acc"])
                acc[op][p_] = min(acc[op].get(p_, 10**9), ac)
    sweep = {}
    if a.sweep_stats:
        for line in open(a.sweep_stats):
            kv = dict(w.split("=", 1) for w in line.split())
            sweep[int(kv["op"])] = kv
    fails = []
    audit = json.load(open(os.path.join(a.dir, "filter_audit.json")))
    if audit.get("precision_bits") != [200, 400]:
        fails.append("missing independent high-precision filter audit")
    for op in OPS:
        row = audit.get("per", {}).get(str(op), {})
        if row.get("count", 0) < 1000 or not 0 <= row.get("max_relative_error", math.inf) <= 2.0 ** -48:
            fails.append("%s: independent filter audit missing or failed" % NAME[op])
    print("Coverage: exhaustive double screen in full mode; selected Sail evaluations; sampled libm audit.")
    print("%-6s %12s %10s %8s %12s %7s %10s %9s %11s" %
          ("func", "swept", "hard", "closest", "closest 2^-k", "k-24", "sail evals", "mismatch", "max p"))
    for op in OPS:
        m = meta["per"][str(op)]
        s = summ[op]
        maxp = 0
        for i, c in enumerate(s["levels"]):
            if c:
                maxp = 64 << i
        k = m["max_k"]
        e = k - 24
        sw = sweep.get(op, {})
        print("%-6s %12s %10s %8s %12s %7s %10d %9d %11d" %
              (NAME[op], sw.get("swept", "?"), sw.get("hard", m["H"]), m["max_k_x"], "2^-%d" % k if k >= 0 else "-", e if k >= 0 else "-",
               s["evals"], s["mism"] + s["nonconv"], maxp))
        if m["unresolved"]:
            fails.append("%s: %d inputs closer than 2^-%d to a rounding boundary (first %s)" %
                         (NAME[op], len(m["unresolved"]), MAX_PREC - 20, m["unresolved"][0]))
        if k >= 0 and e > E_MAX:
            fails.append("%s: closest input %s is 2^-%d from a boundary: e = %d > %d" % (NAME[op], m["max_k_x"], k, e, E_MAX))
        if s["mism"] or s["nonconv"]:
            fails.append("%s: %d mismatches, %d non-converged" % (NAME[op], s["mism"], s["nonconv"]))
        if m["filter_miss"]:
            fails.append("%s: the double filter missed %d inputs closer than 2^-%d: %s" %
                         (NAME[op], len(m["filter_miss"]), FILTER_K, ", ".join(m["filter_miss"][:5])))
        expected_evals = m["H"] + m["E"] + m["R"] - len(m["unresolved"])
        if s["evals"] != expected_evals:
            fails.append("%s: Sail evaluated %d inputs, expected %d" % (NAME[op], s["evals"], expected_evals))
        if sw:
            if float(sw["audit_max_rel_err"]) > 2.0 ** -48:
                fails.append("%s: libm audit error %s exceeds 2^-48" % (NAME[op], sw["audit_max_rel_err"]))
            if int(sw["hard"]) != m["H"]:
                fails.append("%s: sweep reported %s hard inputs, recheck saw %d" % (NAME[op], sw["hard"], m["H"]))
    print()
    for op in OPS:
        m = meta["per"][str(op)]
        s = summ[op]
        sw = sweep.get(op, {})
        lv = ", ".join("p=%d:%d" % (64 << i, c) for i, c in enumerate(s["levels"]) if c)
        print("%s: jobs H=%d E=%d R=%d (random inputs checked against the double filter: %d); Ziv stops at %s; "
              "long-double comparison max rel err %s over %s inputs (zero count means unavailable); sail time %.1f s" %
              (NAME[op], m["H"], m["E"], m["R"], m["filter_checked"], lv or "-",
               sw.get("audit_max_rel_err", "?"), sw.get("audit_n", "?"), s["seconds"]))
        hist = sorted((int(b), c) for b, c in m["k_hist"].items())
        print("   hard inputs by closeness 2^-k, k in [b, b+8): " + ", ".join("%d:%d" % bc for bc in hist))
    # evaluator limit: bits lost at p = 2048 (acc probe), per function and overall
    worst = 0
    for op in OPS:
        a2048 = acc[op].get(2048)
        if a2048 is not None:
            lost = 2048 - a2048
            worst = max(worst, lost)
            print("%s: enclosure accuracy at p=2048 is >= %d bits (loses %d); at p=64 >= %s bits" %
                  (NAME[op], a2048, lost, acc[op].get(64, "?")))
    if worst:
        limit = 2048 - worst
        print("evaluator limit (p = 2048, fully accounted levels): relative distance 2^-%d, i.e. e = %d >= %d" %
              (limit, limit - 24, E_MAX))
        if limit - 24 < E_MAX:
            fails.append("the evaluator's limit e = %d is below the accepted E_MAX = %d" % (limit - 24, E_MAX))
    for line in problems[:20]:
        print(line)
    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  - " + f)
        return 1
    print("\nPASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--hard", required=True)
    j.add_argument("--stride", type=int, default=1)
    j.add_argument("--seed", type=int, default=20261112)
    j.add_argument("--samples", type=int, default=20000)
    j.add_argument("--out", required=True)
    j.add_argument("--procs", type=int, default=12)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--sail", nargs="+", required=True)
    r.add_argument("--sweep-stats")
    a = ap.parse_args()
    if a.cmd == "jobs":
        cmd_jobs(a)
    else:
        sys.exit(cmd_report(a))


if __name__ == "__main__":
    main()
