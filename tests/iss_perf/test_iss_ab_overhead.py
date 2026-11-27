#!/usr/bin/env python3
"""The verdict of the performance gate (`tools/iss_ab_overhead.py --gate`), driven with synthetic timings: no binary runs.

The tree is SLOW times slower than the base on the programs in SLOWED and equal elsewhere. The gate must fail when any one
program is slower than its threshold, whatever the other programs do, and must stay quiet when nothing moved.

    python3 tests/iss_perf/test_iss_ab_overhead.py
"""
import argparse
import contextlib
import io
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import iss_ab_overhead as G  # noqa: E402

REAL_BUCKET_WORK = G.bucket_work  # the other tests replace it with a stub
BENCH = ["b1", "b3", "b4", "dense_f16", "dense_i8", "vxm_valid", "transc", "syncnotify"]
BUCKETS = ["P1_1_64", "P8_4_128"]
# base seconds of each program: the short ones fall under the 5 ms line of the looser threshold
BASE_S = {"b1": 0.9, "b3": 0.7, "b4": 6.0, "dense_f16": 0.1, "dense_i8": 0.08, "vxm_valid": 0.4, "transc": 0.02, "syncnotify": 0.004,
          "P1_1_64": 0.03, "P8_4_128": 0.13}


def trial(slowed, slow, noise=0.0, seed=1, buckets=True):
    """(exit code, output) of the gate over every program with the given synthetic timings."""
    rng = random.Random(seed)

    def secs(bin_, name):
        t = BASE_S[name] * (slow if bin_ == "tree" and name in slowed else 1.0)
        return t * (1.0 + rng.uniform(-noise, noise))

    G.one_bench = lambda bin_, prog: secs(bin_, os.path.basename(prog).removesuffix(".txt"))
    G.one_bucket = lambda bin_, extra, d, prog, tmp: secs(bin_, prog)
    G.bucket_work = lambda bin_, extra, d, prog, tmp: (100, 50)
    a = argparse.Namespace(
        bench=[f"x/{p}.txt" for p in BENCH], progs=BUCKETS if buckets else None, d="/nonexistent", n=7, gate=1.15, gate_small=1.30
    )
    out, code = io.StringIO(), 0
    with contextlib.redirect_stdout(out):
        try:
            G.bench_mode(a, [("base", "base", []), ("tree", "tree", [])])
        except SystemExit as e:
            code = e.code or 0
    return code, out.getvalue()


class GateVerdict(unittest.TestCase):
    def test_nothing_moved_passes(self):
        code, out = trial([], 1.0)
        self.assertEqual(code, 0, out)

    def test_noise_passes(self):
        code, out = trial([], 1.0, noise=0.05)
        self.assertEqual(code, 0, out)

    def test_one_program_three_times_slower_fails(self):
        code, out = trial(["b4"], 3.0)
        self.assertEqual(code, 1, out)
        self.assertIn("b4", out.splitlines()[-1])

    def test_one_program_hundred_times_slower_fails(self):
        code, out = trial(["b4"], 100.0)
        self.assertEqual(code, 1, out)

    def test_three_programs_three_times_slower_fails(self):
        code, out = trial(["b4", "dense_f16", "vxm_valid"], 3.0)
        self.assertEqual(code, 1, out)
        last = out.splitlines()[-1]
        for p in ("b4", "dense_f16", "vxm_valid"):
            self.assertIn(p, last)

    def test_serving_step_is_timed(self):
        code, out = trial(["P8_4_128"], 1.5)
        self.assertEqual(code, 1, out)
        self.assertIn("P8_4_128", out.splitlines()[-1])
        code, out = trial([], 1.0, buckets=False)
        self.assertNotIn("P8_4_128", out)

    def test_short_program_has_the_looser_threshold(self):
        # syncnotify runs 4 ms: 1.25x is inside the looser threshold, 1.6x is not
        self.assertEqual(trial(["syncnotify"], 1.25)[0], 0)
        self.assertEqual(trial(["syncnotify"], 1.6)[0], 1)
        # a long program at 1.25x is above the plain threshold
        self.assertEqual(trial(["b3"], 1.25)[0], 1)


def run_gate(bench_fn=None, bucket_fn=None, insn_fn=None, insn_gate=None, small=None, work=None, insn_only=False, bucket_work=None, gate=1.15):
    """The gate with stubbed runs: (exit code, stdout, stderr). `work` maps (program, label) to the cycles that side reports."""
    G.WORK.clear()

    def one_bench(bin_, prog):
        name = os.path.basename(prog).removesuffix(".txt")
        if bench_fn:
            bench_fn(bin_, name)
        G.WORK[(prog, bin_)] = (work or {}).get((name, bin_), 100)
        return BASE_S[name]

    def one_bucket(bin_, extra, d, prog, tmp):
        if bucket_fn:
            bucket_fn(bin_, prog)
        return BASE_S[prog]

    def stub_work(bin_, extra, d, prog, tmp):
        return ((work or {}).get((prog, bin_), 100), 50)

    G.one_bench, G.one_bucket, G.bucket_work = one_bench, one_bucket, bucket_work or stub_work
    G.one_insn = insn_fn or (lambda bin_, extra, d, prog, bucket, tmp: 1000)
    a = argparse.Namespace(
        bench=[f"x/{p}.txt" for p in BENCH], progs=BUCKETS, d="/nonexistent", n=2, gate=gate, gate_small=1.30,
        insn_gate=insn_gate, small=small, dirs={}, insn_only=insn_only,
    )
    out, err, code = io.StringIO(), io.StringIO(), 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            G.bench_mode(a, [("base", "base", []), ("tree", "tree", [])])
        except SystemExit as e:
            code = e.code or 0
    return code, out.getvalue(), err.getvalue()


class ExitCodes(unittest.TestCase):
    def test_a_base_side_error_is_not_a_regression(self):
        def fail(bin_, prog):
            if bin_ == "base" and prog == "P8_4_128":
                raise G.RunFailed("OTS-E-ASM-003: object was built against another encoding.json")

        code, _, err = run_gate(bucket_fn=fail)
        self.assertEqual(code, G.EXIT_BASE, err)
        self.assertNotEqual(code, G.EXIT_REGRESSION)
        self.assertIn("not a regression", err)

    def test_a_tree_side_error_has_its_own_code(self):
        def fail(bin_, prog):
            if bin_ == "tree" and prog == "P1_1_64":
                raise G.RunFailed("rc 2")

        code, _, err = run_gate(bucket_fn=fail)
        self.assertEqual(code, G.EXIT_TREE, err)

    def test_a_bench_error_of_the_base_is_a_base_error(self):
        def fail(bin_, prog):
            if bin_ == "base" and prog == "b4":
                raise G.RunFailed("rc 1")

        self.assertEqual(run_gate(bench_fn=fail)[0], G.EXIT_BASE)

    def test_different_work_is_not_comparable_not_fast(self):
        # the tree stops early on P8: the same time, half the cycles
        code, _, err = run_gate(work={("P8_4_128", "base"): 8000, ("P8_4_128", "tree"): 4000})
        self.assertEqual(code, G.EXIT_NOT_COMPARABLE, err)
        self.assertIn("P8_4_128", err)

    def test_equal_work_passes(self):
        self.assertEqual(run_gate(work={("P8_4_128", "base"): 8000, ("P8_4_128", "tree"): 8000})[0], 0)

    def test_different_ticks_alone_are_not_comparable_either(self):
        G.WORK.clear()

        def bucket_work(bin_, extra, d, prog, tmp):
            return (8000, 4000 if bin_ == "tree" else 5000)

        code, _, err = run_gate(bucket_work=bucket_work)
        self.assertEqual(code, G.EXIT_NOT_COMPARABLE, err)

    def test_the_instruction_leg_alone_refuses_different_work(self):
        # the leg that every gate runs compared no work at all
        code, out, err = run_gate(
            insn_fn=InstructionLeg.counts({}), insn_gate=1.01, insn_only=True,
            work={("P1_1_64", "base"): 26398, ("P1_1_64", "tree"): 26399},
        )
        self.assertEqual(code, G.EXIT_NOT_COMPARABLE, (out, err))
        self.assertIn("P1_1_64", err)

    def test_the_codes_are_distinct(self):
        codes = [G.EXIT_REGRESSION, G.EXIT_SETUP, G.EXIT_BASE, G.EXIT_TREE, G.EXIT_NOT_COMPARABLE]
        self.assertEqual(len(set(codes)), 5)


FAKE_ISS = """#!/bin/sh
# a stand-in for `tsp-iss run`: prints the E line of a clean run and writes the --stats file with the cycles of $FAKE_CYCLES
for a in "$@"; do case "$a" in --stats=*) f="${a#--stats=}";; esac; done
echo "E - -1 0"
printf '{"wall_seconds": 0.5, "last_cycle": %s, "ticks": 7}' "$FAKE_CYCLES" > "$f"
"""


class RealParse(unittest.TestCase):
    """ with a real subprocess and the real `E - -1 0` line: the work comes from `--stats`, not from the E line."""

    def setUp(self):
        import tempfile
        G.bucket_work = REAL_BUCKET_WORK
        self.tmp = tempfile.mkdtemp(prefix="iss-fake-")
        self.bin = os.path.join(self.tmp, "fake-iss")
        with open(self.bin, "w") as f:
            f.write(FAKE_ISS)
        os.chmod(self.bin, 0o755)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def work(self, cycles):
        os.environ["FAKE_CYCLES"] = str(cycles)
        return G.bucket_work(self.bin, [], self.tmp, "P1_1_64", self.tmp)

    def test_the_work_is_the_cycle_count_not_the_first_diagnostic_field(self):
        self.assertEqual(self.work(26398), (26398, 7))
        self.assertNotEqual(self.work(26398), self.work(26399))

    def test_a_different_bucket_fires_exit_5_with_real_runs(self):
        a = argparse.Namespace(d=self.tmp, dirs={})
        err, code = io.StringIO(), 0
        os.environ["FAKE_CYCLES"] = "26398"
        same = (("base", self.bin, []), ("tree", self.bin, []))
        G.check_buckets_work(a, same, ["P1_1_64"], self.tmp)  # equal work: no exit
        other = os.path.join(self.tmp, "fake-iss-other")
        with open(other, "w") as f:
            f.write(FAKE_ISS.replace('"$FAKE_CYCLES"', "27000"))
        os.chmod(other, 0o755)
        with contextlib.redirect_stderr(err):
            try:
                G.check_buckets_work(a, (("base", self.bin, []), ("tree", other, [])), ["P1_1_64"], self.tmp)
            except SystemExit as e:
                code = e.code
        self.assertEqual(code, G.EXIT_NOT_COMPARABLE, err.getvalue())
        self.assertIn("NOT COMPARABLE", err.getvalue())


class InstructionLeg(unittest.TestCase):
    @staticmethod
    def counts(extra):
        """Instructions per run: 1000 on every program and side, `extra` more on the tree for the programs named."""

        def f(bin_, _extra, d, prog, bucket, tmp):
            name = prog if bucket else os.path.basename(prog).removesuffix(".txt")
            return 1000 + (extra.get(name, 0) if bin_ == "tree" else 0)

        return f

    def test_equal_counts_pass(self):
        self.assertEqual(run_gate(insn_fn=self.counts({}), insn_gate=1.01)[0], 0)

    def test_two_percent_more_work_on_one_program_fails_whatever_the_time(self):
        code, out, _ = run_gate(insn_fn=self.counts({"b4": 20}), insn_gate=1.01)
        self.assertEqual(code, G.EXIT_REGRESSION, out)
        self.assertIn("b4", out)
        self.assertIn("instructions", out)

    def test_instruction_limit_is_enforced_without_wall_limit(self):
        result = run_gate(insn_fn=self.counts({'b4': 20}), insn_gate=1.01, gate=None)
        self.assertEqual(result[0], G.EXIT_REGRESSION, result)

    def test_under_the_limit_passes(self):
        self.assertEqual(run_gate(insn_fn=self.counts({"b4": 5}), insn_gate=1.01)[0], 0)

    def test_a_short_program_takes_the_least_of_three_runs(self):
        seen = {}

        def noisy(bin_, _extra, d, prog, bucket, tmp):
            n = seen[bin_] = seen.get(bin_, 0) + 1
            return 1000 + (500 if (bin_ == "tree" and n == 1) else 0)

        code, out, _ = run_gate(insn_fn=noisy, insn_gate=1.01)
        self.assertEqual(code, 0, out)

    def test_a_long_program_also_takes_the_least_of_three_runs(self):
        # one run of a billion-instruction program is a coin flip near the limit
        seen = {}

        def noisy(bin_, _extra, d, prog, bucket, tmp):
            n = seen[bin_] = seen.get(bin_, 0) + 1
            return 5_000_000_000 + (200_000_000 if (bin_ == "tree" and n % 3 == 1) else 0)

        code, out, _ = run_gate(insn_fn=noisy, insn_gate=1.01)
        self.assertEqual(code, 0, out)
        self.assertEqual(seen["tree"], seen["base"])
        self.assertEqual(seen["tree"] % 3, 0)

    def test_the_leg_alone_runs_no_timed_pair_and_still_judges(self):
        def no_time(bin_, prog):
            raise AssertionError("a timed run in the instruction-only mode")

        ok = run_gate(bench_fn=no_time, bucket_fn=no_time, insn_fn=self.counts({}), insn_gate=1.01, insn_only=True)
        self.assertEqual(ok[0], 0, ok)
        bad = run_gate(bench_fn=no_time, bucket_fn=no_time, insn_fn=self.counts({"vxm_valid": 20}), insn_gate=1.01, insn_only=True)
        self.assertEqual(bad[0], G.EXIT_REGRESSION, bad)

    def test_the_leg_is_off_without_a_limit(self):
        self.assertEqual(run_gate(insn_fn=self.counts({"b4": 500}), insn_gate=None)[0], 0)

    def test_an_error_of_the_base_in_the_leg_is_a_base_error(self):
        def f(bin_, _extra, d, prog, bucket, tmp):
            if bin_ == "base":
                raise G.RunFailed("no instruction counter")
            return 1000

        self.assertEqual(run_gate(insn_fn=f, insn_gate=1.01)[0], G.EXIT_BASE)


class ThresholdClass(unittest.TestCase):
    def test_the_class_comes_from_the_program_list_not_from_a_time(self):
        # b1 is on the list: 1.25x passes; b3 is not: 1.25x fails, though both are measured at any speed
        self.assertEqual(trial(["b1"], 1.25)[0], 0)
        self.assertEqual(trial(["b3"], 1.25)[0], 1)

    def test_a_fast_program_off_the_list_has_the_tight_threshold(self):
        # transc runs 20 ms in the table; a quick base on a busy host would not move it onto the loose threshold
        self.assertEqual(trial(["transc"], 1.25)[0], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
