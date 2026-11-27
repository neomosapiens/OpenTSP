#!/usr/bin/env python3
"""`just verify-all` runs every formal gate, times each step and hashes the verdict lines of all of them."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

SPEC = Path(__file__).resolve().parents[2]
ROOT = SPEC.parent
FORMAL = (ROOT / "tools/just/formal.just").read_text()

# The formal checks that the default run must contain.
REQUIRED = ["model-prove", "model-exhaustive", "model-test", "export-check", "prove-extra", "lean-check",
            "srf-ring-test", "lanes-test", "encoding-exact", "encoding-model"]
# The recipes that only the nightly run executes.
NIGHTLY = ["prove-mutants", "prove-profiles", "prove-extra-wide", "prove-nightly"]


def default_steps():
    match = re.search(r"VERIFY_ALL_STEPS-([^}]*)\}", FORMAL)
    assert match, "verify-all has no default step list"
    return match.group(1).split()


def verdict_pattern():
    match = re.search(r"grep -E '\^\(([^']*)\)' \"\$log\"", FORMAL)
    assert match, "verify-all has no verdict pattern"
    return re.compile(r"^(?:" + match.group(1) + ")", re.M)


def recipes():
    out = subprocess.run(["just", "--summary"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return set(out.split())


class Steps(unittest.TestCase):
    def test_every_formal_recipe_is_in_the_default_run(self):
        steps = default_steps()
        for recipe in REQUIRED:
            self.assertIn(recipe, steps)

    def test_no_step_is_listed_twice_and_each_is_a_recipe(self):
        steps = default_steps()
        self.assertEqual(len(steps), len(set(steps)))
        known = recipes()
        for step in steps:
            self.assertIn(step, known)

    def test_the_nightly_recipes_exist(self):
        known = recipes()
        for recipe in NIGHTLY:
            self.assertIn(recipe, known)


class Verdicts(unittest.TestCase):
    """The hash covers the verdict line of every step: a step whose verdict is not in the pattern could change silently."""

    LINES = {
        "prove-extra": "prove_extra: all suites proved (ties frontend wide hold numerics fptheory)",
        "lean-check": "lean_check: ok profile=ref theorems=10",
        "srf-ring-test": "c_test ok: srf_ring.sail profile=mini",
        "lanes-test": "c_test ok: lanes_exhaustive.sail profile=mini",
        "model-test": "model-test: all passed",
        "export-check": "export_check: all checks passed (mini)",
        "export-check-while": "ok: 29 while loops in the generated Lean, as recorded",
        "encoding-model": "encoding_model: 78 entries, 1710 probe words, 95040 bit flips, 0 problem(s)",
        "fptheory": "fround_fp_theory: binary16 rounding agrees with the solver's floating-point theory for 14 exponents",
        "encoding-exact": "encoding_exact: 78 entries, 1707 same-width pairs checked exactly, 0 overlapping pair(s) separated by field legality",
    }

    def test_each_new_step_has_a_verdict_line_in_the_pattern(self):
        pattern = verdict_pattern()
        for step, line in self.LINES.items():
            with self.subTest(step=step):
                self.assertTrue(pattern.search(line + "\n"), line)

    def test_the_scripts_print_those_verdict_lines(self):
        needles = {
            "test/formal/prove_extra.sh": 'echo "prove_extra: all suites proved',
            "test/formal/lean/lean_check.sh": 'echo "lean_check: ok profile=',
            "test/formal/c_test.sh": 'echo "c_test ok: ',
            "test/formal/encoding_exact.py": '"encoding_exact: %d entries',
            "test/formal/encoding_from_model.py": '"encoding_model: %d entries',
            "test/formal/fround_fp_theory.py": '"fround_fp_theory: binary16 rounding agrees',
        }
        for path, needle in needles.items():
            with self.subTest(path=path):
                self.assertIn(needle, (SPEC / path).read_text())

    def test_a_timing_or_path_is_not_a_verdict(self):
        pattern = verdict_pattern()
        for line in ("lean_check: lake build Proofs 178 s: 8 theorems checked", "-- verify-all: lean-check took 180 s",
                     "c_test: the $[test] functions passed (/tmp/x/config/mini.json)"):
            self.assertFalse(pattern.search(line + "\n"), line)


class Run(unittest.TestCase):
    """One cheap step end to end: the header, the timing line, the summary and the verdict line."""

    def run_all(self, steps, recipe="verify-all"):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "verification.log"
            prefix = recipe.upper().replace("-", "_")
            env = os.environ | {f"{prefix}_STEPS": steps, f"{prefix}_LOG": str(log)}
            result = subprocess.run(["just", recipe], cwd=ROOT, env=env, capture_output=True, text=True)
            return result, log.read_text() if log.exists() else ""

    def test_a_step_is_timed_and_its_verdict_is_hashed(self):
        result, log = self.run_all("encoding-exact")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, r"-- verify-all: encoding-exact took \d+ s")
        self.assertRegex(result.stdout, rf"verify-all: PASS source-directory={re.escape(str(ROOT))} verdict-sha256=[0-9a-f]{{64}}")
        self.assertIn("encoding_exact: ", log)

    def test_the_hash_changes_with_the_verdict_of_a_step(self):
        """Two stand-in steps with different verdict lines (a justfile that imports the real recipe) give different hashes."""
        with tempfile.TemporaryDirectory() as tmp:
            just = Path(tmp) / "justfile"
            just.write_text("import '%s'\nfake-a:\n    @echo 'PASS p_a'\nfake-b:\n    @echo 'PASS p_b'\n" % (ROOT / "tools/just/formal.just"))
            digests = []
            for step in ("fake-a", "fake-b", "fake-a"):
                env = dict(os.environ, VERIFY_ALL_STEPS=step, VERIFY_ALL_LOG=str(Path(tmp) / "log"))
                result = subprocess.run(["just", "--justfile", str(just), "--working-directory", str(ROOT), "verify-all"],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                digests.append(re.search(r"verdict-sha256=([0-9a-f]{64})", result.stdout).group(1))
        self.assertNotEqual(digests[0], digests[1])
        self.assertEqual(digests[0], digests[2])

    def test_an_empty_selection_fails_without_a_verdict(self):
        result, _ = self.run_all("   ")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no checks selected", result.stderr)
        self.assertNotIn("PASS", result.stdout)

    def test_a_partial_run_identifies_the_selected_checks(self):
        result, _ = self.run_all("encoding-exact")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("scope=selected checks=encoding-exact", result.stdout)

    def test_an_empty_nightly_selection_fails_without_a_verdict(self):
        result, _ = self.run_all("   ", recipe="prove-nightly")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no checks selected", result.stderr)
        self.assertNotIn("PASS", result.stdout)

    def test_a_failing_step_fails_the_run(self):
        result, _ = self.run_all("no-such-recipe")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
