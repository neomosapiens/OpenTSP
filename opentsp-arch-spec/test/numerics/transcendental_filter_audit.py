#!/usr/bin/env python3
"""Sample the actual C screening functions against independent 200/400-bit values.

This is empirical evidence about libm, not a proof of its error over all inputs.
The audit never treats an equal-width long double comparison as higher precision.
"""
import json
import math
from pathlib import Path
import random
import subprocess
import sys

from mpmath import mp
from transcendental_f32_recheck import OPS, edge_inputs, eval_exact, f32_of_bits, sample_inputs


def relative_error(op, bits, value):
    x = f32_of_bits(bits)
    low = eval_exact(op, x, 200)
    high = eval_exact(op, x, 400)
    if abs((low - high) / high) > mp.mpf(2) ** -180:
        raise ValueError("high-precision reference did not stabilize")
    return abs((mp.mpf(value) - high) / high)


def audit(executable, out):
    rng = random.Random(20261117)
    jobs = []
    for op in OPS:
        for bits in sorted(set(edge_inputs(op, rng)) | set(sample_inputs(op, rng, 4096))):
            x = f32_of_bits(bits)
            if not math.isfinite(x) or x == 0 or (op in (23, 24) and x <= 0) or (op == 24 and x == 1):
                continue
            # Beyond these limits Exp rounds to zero/infinity even in double;
            # those paths do not participate in the screening distance test.
            if op == 22 and not -690 < x < 700:
                continue
            jobs.append((op, bits))
    proc = subprocess.run([str(executable), "--audit"],
                          input="".join("%d %08x\n" % j for j in jobs),
                          text=True, capture_output=True, check=True, timeout=60)
    lines = proc.stdout.splitlines()
    if len(lines) != len(jobs):
        raise ValueError("C audit result count mismatch")
    results = {str(op): {"count": 0, "max_relative_error": 0.0} for op in OPS}
    for (op, bits), line in zip(jobs, lines):
        o, b, y = line.split()
        if (int(o), int(b, 16)) != (op, bits):
            raise ValueError("C audit result identity mismatch")
        value = float.fromhex(y)
        if not math.isfinite(value) or value == 0:
            raise ValueError("unexpected nonfinite/zero C audit result")
        error = relative_error(op, bits, value)
        row = results[str(op)]
        row["count"] += 1
        if error > row["max_relative_error"]:
            row["max_relative_error"] = float(error)
            row["worst_input"] = "%08x" % bits
        if error > mp.mpf(2) ** -48:
            raise ValueError("libm audit exceeds 2^-48: op=%d input=%08x error=%s" % (op, bits, error))
    receipt = {"precision_bits": [200, 400], "seed": 20261117,
               "scope": "sampled libm error; no exhaustive error-bound proof", "per": results}
    Path(out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    audit(Path(sys.argv[1]).resolve(), sys.argv[2])
