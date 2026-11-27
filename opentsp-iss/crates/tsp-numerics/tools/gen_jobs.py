#!/usr/bin/env python3
"""Write a job file for tools/sail_dump.sh (format: numerics_jobs.py).

  gen_jobs.py SET OUT.txt [--screen-work DIR]

Sets (the large ones are for the optional full differential run, tools/run_differential.sh):
  sample    the checked-in sample (reference/sample.txt)
  i8        i8 transcendentals, all inputs
  f16tr     f16 transcendentals, all inputs of the four functions
  f32tr     f32 transcendentals: edge classes, a random sample, and (--screen-work DIR) every job of the
            exhaustive check's work directory (opentsp-arch-spec/tools/check_transcendental_f32.sh
            --work DIR: its hard inputs and recheck sample)
  kernels   every set_specs() input of the kernels (binary, unary, cmp, cvt, conv), expanded
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numerics_jobs as J  # noqa: E402


def f32tr_jobs(screen_work):
    jobs = []
    for op in (21, 22, 23, 24):
        for x in J.F32_EDGES:
            jobs.append(J.job_line("tr", op, 32, J.hx(x)))
        rng = J.Rng(5000 + op)
        for _ in range(100000):
            jobs.append(J.job_line("tr", op, 32, J.hx(J.rand_float(rng, "f32"))))
    if screen_work:
        seen = set()
        for name in ("jobs_eval.txt", "jobs.txt"):
            path = os.path.join(screen_work, name)
            if os.path.exists(path):
                for line in open(path):
                    f = line.split()
                    if len(f) == 4 and f[0] == "E":
                        j = J.job_line("tr", int(f[1]), 32, "%x" % int(f[2], 16))
                        if j not in seen:
                            seen.add(j)
                            jobs.append(j)
                break
        else:
            sys.exit("no jobs file in " + screen_work)
    return jobs


def kernel_jobs():
    jobs = []
    for name, kind, args, inp in J.set_specs():
        if kind in ("bin", "cmp"):
            for x, y in J.binary_inputs(args[1], inp):
                jobs.append(J.job_line(kind, args[0], args[1], J.hx(x), J.hx(y)))
        elif kind == "un":
            for x in J.unary_inputs(args[1], inp):
                jobs.append(J.job_line("un", args[0], args[1], J.hx(x)))
        elif kind == "cvt":
            for x in J.unary_inputs(args[0], inp):
                jobs.append(J.job_line("cvt", args[0], args[1], J.hx(x)))
        elif kind == "conv":
            src = {25: 0, 26: 3, 27: 0, 28: 1, 29: 0, 30: 2, 31: 1, 32: 3}[args[0]]
            for x in J.unary_inputs(src, inp):
                jobs.append(J.job_line("conv", args[0], J.hx(x)))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["sample", "i8", "f16tr", "f32tr", "kernels"])
    ap.add_argument("out")
    ap.add_argument("--screen-work")
    a = ap.parse_args()
    if a.set == "sample":
        jobs = J.sample_jobs()
    elif a.set == "i8":
        jobs = [J.job_line("i8", op, J.hx(x)) for op in (21, 22, 23, 24) for x in range(256)]
    elif a.set == "f16tr":
        jobs = [J.job_line("tr", op, 16, J.hx(x)) for op in (21, 22, 23, 24) for x in range(65536)]
    elif a.set == "f32tr":
        jobs = f32tr_jobs(a.screen_work)
    else:
        jobs = kernel_jobs()
    with open(a.out, "w") as f:
        f.write("\n".join(jobs) + "\n")
    print("%d jobs" % len(jobs), file=sys.stderr)


if __name__ == "__main__":
    main()
