#!/usr/bin/env python3
"""Materialise the ISS performance benchmark inputs: for each named bucket of the program cache, the object and the one
weight package `tsp-iss run` takes, with the padding-only lane plan `cycle_report.py --time` uses.

    python3 tools/iss_bench_prep.py [--out DIR] [--layers 2|8] [BUCKET ...]

writes `DIR/<bucket>.elf` and `DIR/<bucket>.safetensors`; the benchmark run is then

    target/release/tsp-iss run --profile ref --weights DIR/<bucket>.safetensors --stats=S DIR/<bucket>.elf

whose `wall_seconds` is the ISS's own time (machine construction, image load and the cycles), as in `cycles.json`.
The lane plan is padding only, so the cycle count is the bucket's and does not depend on the host."""
import argparse
import os
import shutil
import sys
import tempfile

from opentsp_vllm import step_plan
from opentsp_vllm.models.tinystories import build_programs as BP
from opentsp_vllm.models.tinystories import program_cache as PC


def prep(name, layers, out, cache=None):
    model = PC.Model(layers)
    bucket = PC.parse_bucket(name)
    BP.build_bucket(model, bucket, cache=cache)
    art = BP.Artifacts(PC.bucket_dir(cache, model, bucket), model.profile)
    names = [str(n) for n in art.paged.table.names]
    values = PC.plan_site_values(art.record["geometry"], names, step_plan.padding_plan(bucket, vocab=50257))
    with tempfile.TemporaryDirectory(prefix="iss-bench-") as tmp:
        obj, pkg = BP.write_run_files(art, values, tmp, model.package)
        for src, ext in ((obj, ".elf"), (pkg, ".safetensors")):
            shutil.copyfile(src, os.path.join(out, PC.bucket_name(bucket) + ext))
    return art.profile_name, art.record["cycles"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/tmp/iss-bench")
    p.add_argument("--layers", type=int, default=8, choices=(2, 8))
    p.add_argument("--cache", default=None)
    p.add_argument("buckets", nargs="*", default=["P1_1_64", "P8_4_128"])
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for b in a.buckets:
        profile, cycles = prep(b, a.layers, a.out, a.cache)
        print(f"{b}: {a.out}/{b}.elf + {b}.safetensors (profile {profile}, {cycles} cycles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
