#!/usr/bin/env python3
"""Reject infrastructure/signal errors in both resource and native proof verdicts."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from resource_mutations import proof_rejection

BASE = ("error: Proofs/Resources.lean:60:68: unsolved goals\n"
        "error: Lean exited with code 1\nerror: build failed\n")
CASES = [
    ("unwrapped_recursion", 1, BASE + "maximum recursion depth has been reached\n", False),
    ("unwrapped_heartbeat", 1, BASE + "(deterministic) timeout at whnf, maximum number of heartbeats (200000) has been reached\n", False),
    ("nested_recursion", 1, BASE.replace("unsolved goals", "Tactic simp failed with a nested error:").replace("error: Lean exited", "maximum recursion depth has been reached\nerror: Lean exited"), False),
    ("nested_unknown", 1, BASE.replace("unsolved goals", "Tactic simp failed with a nested error:").replace("error: Lean exited", "dependency loader failed\nerror: Lean exited"), False),
    ("plain_tactic", 1, BASE.replace("unsolved goals", "Tactic simp failed"), True),
    ("nested_heartbeat", 1, BASE.replace("unsolved goals", "Tactic simp failed with a nested error:").replace("error: Lean exited", "(deterministic) timeout at whnf, maximum number of heartbeats (200000) has been reached\nerror: Lean exited"), False),
    ("resource_words_in_goal", 1, BASE + '⊢ "maximum recursion depth has been reached" = "literal"\n', True),
    ("theorem", 1, BASE, True),
    ("signal", -9, BASE, False),
    ("signal_exit", 137, BASE, False),
    ("other_exit", 2, BASE, False),
    ("successful_exit", 0, BASE, False),
    ("mixed_dependency", 1, BASE + "error: build dependency missing\n", False),
    ("mixed_fatal", 1, BASE + "fatal: dependency corrupt\n", False),
    ("parser", 1, BASE.replace("unsolved goals", "unexpected token"), False),
    ("omega", 1, BASE.replace("unsolved goals", "omega could not prove the goal:"), True),
    ("empty", 1, "", False),
]


class ResourceVerdict(unittest.TestCase):
    def test_resource_classifier(self):
        for name, code, text, expected in CASES:
            with self.subTest(name=name):
                self.assertEqual(proof_rejection(code, text), expected)
        self.assertFalse(proof_rejection(1, BASE + "Permission denied\n"))

    def test_actual_native_driver_branch(self):
        # Execute the actual production driver's verdict branch, not a copied regex.
        driver = Path(__file__).with_name("lean_check.sh").read_text()
        start = driver.index('if [ "$proofs_status" -ne 0 ]; then')
        end = driver.index('/usr/bin/grep -qEi "error:|fatal:|exception:"', start)
        branch = driver[start:end]
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env["work"] = directory
            for name, code, text, expected in CASES:
                if code == 0:
                    continue  # successful-path checks are outside this failure branch
                with self.subTest(name=name):
                    Path(directory, "proofs.log").write_text(text)
                    env["proofs_status"] = str(code)
                    result = subprocess.run(["bash", "-c", branch], env=env,
                                            capture_output=True, text=True, check=False)
                    self.assertEqual("lean_check: a proof does not check\n" in result.stdout,
                                     expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
