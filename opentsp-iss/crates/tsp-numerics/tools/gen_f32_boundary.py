#!/usr/bin/env python3
"""tests/data/f32_boundary.bin: the f32 inputs of TanH, Exp, RSqrt and Log whose function value lies
closest to a rounding boundary of binary32, with the correctly rounded results.

  gen_f32_boundary.py [--per-op N] [--procs P] [--out FILE]

The fast path of tsp_numerics::transcendental evaluates the function in f64 with the host libm and
accepts the result when it is more than 2^-40 (relative) away from every rounding boundary. These
are the inputs that decide whether that filter is wide enough on a libm: the ones below the filter
are declined (the Ziv port answers: the f32 screening "hard" list of opentsp-arch-spec/tools/
check_transcendental_f32.sh), the ones just beyond it are accepted on the strength of the host's
f64 value alone, and those are the ones a libm that is off by more than the margin would get wrong.
Per function the file holds the N/2 closest inputs overall (all declined when the hard list is
longer than N/2) and the N/2 closest beyond the filter; the two sets overlap when the closest
inputs are already beyond it, so the file holds up to N inputs per function. Ties in the distance
are broken by the input bits, so two runs write the same bytes.

  1. sweep   every f32 bit pattern in the domain of the model's evaluator (finite, non-zero, not
             caught by the special-value tables of fp_tr_value) in float64 (numpy; the sweep only
             ranks inputs, so numpy's own accuracy of ~1e-16 does not matter) and keep the N with
             the smallest relative distance d = |frac(s) - 1/2| / s of the value to a boundary,
             s = |value| / 2^q with q the quantum of the result's binade (subnormals and the
             overflow threshold included);
  2. oracle  correctly rounded results by the mpmath oracle of the specification
             (opentsp-arch-spec/test/oracle/numerics.py, the independent half of the checks).

File format (little endian): u32 magic 0x42323346 ("F32B"), u32 N21 N22 N23 N24 (the number of
records per function, in the order TanH, Exp, RSqrt, Log), then per function N records of
(u32 input bits, u32 result bits), by increasing input bits.

Needs the specification toolchain's python (numpy, mpmath):
  source "${OPENTSP_TOOLCHAIN:-$HOME/.local/opt/opentsp-toolchain}/env.sh"
"""
import argparse
import multiprocessing as mp
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../../../opentsp-arch-spec/test")))
from oracle import numerics as O  # noqa: E402

OPS = (21, 22, 23, 24)
MAGIC = 0x42323346
CHUNK = 1 << 22
FILTER = 2.0**-40  # tsp_numerics::transcendental::FILTER


def distances(op, bits):
    """Relative distance of the f64 value of op(x) to the nearest binary32 rounding boundary, per input;
    inf where the evaluator's special-value tables answer or the value is not a usable f64."""
    x = bits.view(np.float32).astype(np.float64)
    with np.errstate(all="ignore"):
        dom = np.isfinite(x) & (x != 0.0)
        if op == 21:
            dom &= np.abs(x) < 64.0
            y = np.tanh(x)
        elif op == 22:
            dom &= np.abs(x) < 512.0
            y = np.exp(x)
        elif op == 23:
            dom &= x > 0.0
            y = 1.0 / np.sqrt(x)
        else:
            dom &= (x > 0.0) & (x != 1.0)
            y = np.log(x)
        ay = np.abs(y)
        dom &= np.isfinite(ay) & (ay != 0.0)
        ay = np.where(dom, ay, 1.0)
        _, e = np.frexp(ay)  # ay = m 2^e, m in [0.5, 1): floor(log2 ay) = e - 1
        qe = np.maximum(e - 1, -126) - 23
        s = np.ldexp(ay, -qe)
        fr = s - np.floor(s)
        d = np.abs(fr - 0.5) / s
    return np.where(dom, d, np.inf)


def sweep_chunk(args):
    """The n smallest distances of one chunk of bit patterns, and the n smallest beyond the filter:
    (distance, bits) per function and per kind."""
    lo, n = args
    bits = np.arange(lo, lo + CHUNK, dtype=np.uint64).astype(np.uint32)
    out = {}
    for op in OPS:
        d = distances(op, bits)
        out[op, "all"] = smallest(d, bits, n)
        out[op, "accepted"] = smallest(np.where(d > FILTER, d, np.inf), bits, n)
    return out


def smallest(d, bits, n):
    """The n smallest (distance, bits) pairs, ties in the distance broken by the bits, so the result
    does not depend on the order in which chunks complete."""
    k = min(n, d.size)
    cut = np.partition(d, k - 1)[k - 1]
    if np.isinf(cut):
        sel = np.flatnonzero(np.isfinite(d))  # fewer than n usable inputs: the rest are never kept
    else:
        sel = np.flatnonzero(d <= cut)
    order = np.lexsort((bits[sel], d[sel]))[:k]
    return d[sel][order], bits[sel][order]


def oracle_one(job):
    op, x = job
    return O.fp_transcendental(op, O.F32, x) & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-op", type=int, default=50000)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "..", "tests", "data", "f32_boundary.bin"))
    a = ap.parse_args()
    half = a.per_op // 2
    with mp.Pool(a.procs) as pool:
        best = {(op, kind): (np.empty(0), np.empty(0, dtype=np.uint32)) for op in OPS for kind in ("all", "accepted")}
        for part in pool.imap_unordered(sweep_chunk, [(lo, half) for lo in range(0, 1 << 32, CHUNK)]):
            for key, (pd, pb) in part.items():
                d, b = best[key]
                best[key] = smallest(np.concatenate([d, pd]), np.concatenate([b, pb]), half)
        picked = {}
        for op in OPS:
            xs = set()
            for kind in ("all", "accepted"):
                d, b = best[op, kind]
                keep = np.isfinite(d)
                xs.update(int(v) for v in b[keep])
                print("op %d %-8s: %d inputs, d from %.3g to %.3g (filter 2^-40 = %.3g)" % (op, kind, keep.sum(), d[keep].min(), d[keep].max(), FILTER))
            picked[op] = sorted(xs)
        jobs = [(op, x) for op in OPS for x in picked[op]]
        results = pool.map(oracle_one, jobs, chunksize=256)
    data = bytearray(struct.pack("<I4I", MAGIC, *[len(picked[op]) for op in OPS]))
    for (op, x), r in zip(jobs, results):
        data += struct.pack("<II", x, r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(data)
    print("wrote %s: %d bytes" % (os.path.normpath(a.out), len(data)))


if __name__ == "__main__":
    main()
