#!/usr/bin/env python3
"""Profile of one bucket program: the scheduled cycles by stage (the emitter's stage marks) and by layer, and the instruction
mix by unit (counts and the first-to-last cycle window of each), from the emitted program before it runs.

    python3 profile_bucket.py --layers 2|8 [--bucket P1_1_64]

Not a generator source (changing it does not change a cache digest)."""
import argparse
import collections
import sys


from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import program_cache as PC


def unit_of(text):
    op = text.split()[0].split(".")[0]
    return op


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=8, choices=(2, 8))
    ap.add_argument("--bucket", default="P1_1_64")
    a = ap.parse_args(argv)
    bucket = PC.parse_bucket(a.bucket)
    model = PC.Model(a.layers, 1, None)
    batch = E.emit_step_batch(*bucket, layers=model.layers, w=model.w)
    print(f"{a.bucket} layers {a.layers}: scheduled last cycle {batch.kit.last}")
    prev, rows = 0, []
    for l, st, c in batch.marks:
        rows.append((l, st, c, c - prev))
        prev = c
    by_stage = collections.OrderedDict()
    for l, st, c, d in rows:
        by_stage.setdefault(st, 0)
        by_stage[st] += d
    print("stage totals (cycles, share of the scheduled last cycle):")
    for st, d in by_stage.items():
        print(f"  {st:10s} {d:7d} {100 * d / batch.kit.last:5.1f}%")
    print("per layer, per stage deltas:")
    for l in sorted({r[0] for r in rows}):
        print(f"  L{l}: " + " ".join(f"{st}={d}" for ll, st, c, d in rows if ll == l))
    mix = collections.defaultdict(list)
    for c, q, t in batch.kit.view2.entries:
        mix[unit_of(t)].append(c)
    print("instruction mix (count, first cycle, last cycle):")
    for op, cs in sorted(mix.items(), key=lambda kv: -len(kv[1])):
        print(f"  {op:12s} {len(cs):7d} {min(cs):7d} {max(cs):7d}")


if __name__ == "__main__":
    sys.exit(main())
