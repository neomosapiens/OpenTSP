"""Exercise the production Lean proof-build stage with bounded tool outcomes.

The stage is read verbatim from lean_check.sh, under its actual errexit/pipefail
settings. Tool output is controlled to distinguish theorem rejection from a build
or infrastructure failure without invoking a complete Sail/Lean build.
"""
from pathlib import Path
import subprocess
import tempfile
import unittest

DRIVER = Path(__file__).with_name("lean") / "lean_check.sh"


class ProofBuildOutcome(unittest.TestCase):
    def stage(self, transcript, exit_code, broken_consumer=None):
        source = DRIVER.read_text()
        start = source.index("# The proofs about the generated model")
        end = source.index('/usr/bin/grep -qE "sorryAx|ofReduceBool', start)
        fragment = source[start:end]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            fixture = work / "tool-output"
            fixture.write_text(transcript)
            # Function lookup precedes PATH, without altering HOME or a toolchain.
            script = """set -euo pipefail
work="$1"
fixture_log="$2"
fixture_exit="$3"
here="$4"
proj="$work/gen/out"
mkdir -p "$proj"
lake() {
  [ "$1" = build ] && [ "$2" = Proofs ] || return 99
  cat "$fixture_log"
  return "$fixture_exit"
}
""" + fragment + "\n: # the successful stage continues to axiom checking\n"
            if broken_consumer is not None:
                # A real filter command runs, but its wrapper reports a tool failure.
                # This is a controlled pipeline failure, not a fabricated disk-full run.
                script = script.replace(
                    'lake() {',
                    f'{broken_consumer}() {{ command {broken_consumer} "$@"; return 1; }}\nlake() {{',
                    1,
                )
            return subprocess.run(
                ["bash", "-c", script, "lean-stage", str(work), str(fixture), str(exit_code), str(DRIVER.resolve().parents[3])],
                capture_output=True, text=True, timeout=10,
            )

    def test_theorem_rejection_reaches_the_mutant_detector(self):
        # Exact rejection family observed from the real m_l1 generated model.
        result = self.stage(
            "error: Proofs/IqPush.lean:46:93: Application type mismatch: The argument\n"
            "  rfl\nerror: Lean exited with code 1\nerror: build failed\n", 1,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("lean_check: a proof does not check", result.stdout)

    def test_unsolved_goals_are_a_theorem_rejection(self):
        result = self.stage("error: Proofs/Barrier.lean:22:10: unsolved goals\ncase some\nFalse\n", 1)
        self.assertEqual(result.returncode, 1)
        self.assertIn("lean_check: a proof does not check", result.stdout)

    def test_kernel_decide_false_is_a_theorem_rejection(self):
        # Exact family from the actual source-mutated ln2 production export.
        result = self.stage(
            "error: Proofs/Ln2Certificates.lean:61:2: Tactic `decide` proved that the proposition\n"
            "  Certificate 1024\nis false\nerror: Lean exited with code 1\nerror: build failed\n", 1,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("lean_check: a proof does not check", result.stdout)

    def test_infrastructure_failure_is_not_a_theorem_rejection(self):
        for output in (
            "error: failed to fetch package\n",
            "error: Proofs/IqPush.lean:1:0: unknown module prefix Sail\n",
            "error: Proofs/IqPush.lean:5:0: unexpected token\n",
            "error: a dependency has an Application type mismatch\n",
        ):
            with self.subTest(output=output):
                result = self.stage(output, 1)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("lean_check: a proof does not check", result.stdout)

    def test_mixed_infrastructure_failure_is_not_a_theorem_rejection(self):
        for other in ("error: failed to fetch package", "error: Proofs/Common.lean:1:0: unknown module prefix Sail", "fatal: missing build state", "exception: dependency failure", "prefix: error: compiler failed"):
            result = self.stage(
                "error: Proofs/IqPush.lean:46:93: Application type mismatch\n" + other + "\nerror: build failed\n", 1,
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("lean_check: a proof does not check", result.stdout)

    def test_nested_child_failure_is_not_a_theorem_rejection(self):
        for child_code in (2, 137, 143):
            result = self.stage(
                "error: Proofs/IqPush.lean:46:93: Application type mismatch\n"
                + f"error: Lean exited with code {child_code}\nerror: build failed\n", 1,
            )
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("lean_check: a proof does not check", result.stdout)

    def test_outer_and_nested_status_matrix_fails_closed(self):
        outer_statuses = (0, 1, 2, 3, 126, 127, 128, 130, 137, 143, 255)
        child_statuses = (0, 1, 2, 126, 127, 137, 143, 255)
        for outer in outer_statuses:
            for child in child_statuses:
                with self.subTest(outer=outer, child=child):
                    result = self.stage(
                        "error: Proofs/IqPush.lean:46:93: Application type mismatch\n"
                        + f"error: Lean exited with code {child}\nerror: build failed\n", outer,
                    )
                    self.assertEqual(result.returncode, 1 if outer == 0 else outer)
                    self.assertEqual(
                        "lean_check: a proof does not check" in result.stdout,
                        outer == 1 and child == 1,
                    )

    def test_output_consumer_failure_is_not_a_theorem_rejection(self):
        for consumer in ("tee", "tail"):
            with self.subTest(consumer=consumer):
                result = self.stage(
                    "error: Proofs/IqPush.lean:46:93: Application type mismatch\n"
                    "error: Lean exited with code 1\nerror: build failed\n", 1, consumer,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("lean_check: a proof does not check", result.stdout)

    def test_a_signal_is_not_a_theorem_rejection(self):
        result = self.stage("error: Proofs/IqPush.lean:46:93: Application type mismatch\n", 137)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("lean_check: a proof does not check", result.stdout)

    def test_axiom_info_with_error_in_theorem_name_proceeds(self):
        # Actual successful kernel output from the integrated enclosure proofs.
        result = self.stage(
            "info: Proofs/Axioms.lean:91:0: 'EnclosureProofs.term_error_decomposition' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
            "Build completed successfully (74 jobs).\n", 0,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_successful_build_proceeds(self):
        result = self.stage("Build completed successfully.\n", 0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("lean_check: a proof does not check", result.stdout)

    def test_success_status_with_error_output_does_not_pass(self):
        result = self.stage("error: backend inconsistency\n", 0)
        self.assertEqual(result.returncode, 1)
        self.assertNotIn("lean_check: a proof does not check", result.stdout)


if __name__ == "__main__":
    unittest.main()
