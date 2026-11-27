#!/usr/bin/env python3
"""Diagnostics of one bucket program by code: emit the bucket (no cache entry is kept), run it once on the ISS on a padding-only
plan (the program's own text, nothing masked or filtered) and count what the ISS reports.

    python3 diag_bucket.py --layers 2|8 [--bucket P1_1_64] [--seed N]

Prints `emit: <ScheduleError>` when the generator itself refuses the program (a generation-time violation), else the cycle count and
one line per diagnostic code with its count and its first three (cycle, queue). Exit 0 when there is no diagnostic, 1 otherwise."""
import argparse
import collections
import os
import sys
import tempfile


from opentsp_vllm.models.tinystories import build_programs as BP
from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_tools.stations import ScheduleError


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=2, choices=(2, 8))
    ap.add_argument("--bucket", default="P1_1_64")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    bucket = PC.parse_bucket(a.bucket)
    E.check_bucket(*bucket)
    model = PC.Model(a.layers, a.seed, None)
    try:
        batch = E.emit_step_batch(*bucket, layers=model.layers, w=model.w)
    except ScheduleError as e:
        print(f"emit: {e}")
        return 1
    asm, iss = BP.ots_asm(), BP.tsp_iss()
    with tempfile.TemporaryDirectory(prefix="diag-bucket-") as tmp:
        sizes = BP.write_artifacts(model, batch, tmp, asm)
        PC.write_json(os.path.join(tmp, "program.json"), {**sizes, "outputs": PC.output_regions(batch.plan, bucket[1])})
        art = BP.Artifacts(tmp, model.profile)
        run = BP.native_run(iss, art, BP.padding_values(batch), model.package, 2 * batch.kit.last)
    by = collections.defaultdict(list)
    for d in run["diags"]:
        by[d[0]].append(tuple(d[1:]))
    print(f"{a.bucket} layers {a.layers}: scheduled {batch.kit.last} cycles, last cycle {run['last_cycle']}, "
          f"{len(run['diags'])} diagnostics, exit {run['returncode']}")
    for code, rows in sorted(by.items()):
        print(f"  {code}: {len(rows)}  first {rows[:3]}")
    return 1 if run["diags"] else 0


if __name__ == "__main__":
    sys.exit(main())
