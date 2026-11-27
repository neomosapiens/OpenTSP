"""Actual SMT producer paths preserve clean counterexamples and reject protocol failures."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import runpy
from contextlib import redirect_stderr
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC / "tools"))
import check_smt_obligations as checker
import fround_fp_theory as theory

SOURCE = "$property\nfunction p_example() -> bool = false\n"
CASES = [
    ("valid_counterexample", "sat\n", "", 0, "FAIL p_example:"),
    ("valid_proof", "unsat\n", "", 0, "PASS p_example:"),
    ("unknown", "unknown\n", "", 0, "FAIL p_example:"),
    ("timeout", "timeout\n", "", 0, "FAIL p_example:"),
    ("parse_error", '(error "bad query")\nsat\n', "", 1, "ERROR SMT p_example:"),
    ("nonzero_sat", "sat\n", "", 7, "ERROR SMT p_example:"),
    ("stderr_unsat", "unsat\n", "solver warning/error", 0, "ERROR SMT p_example:"),
    ("extra_stdout", "sat\nunsat\n", "", 0, "ERROR SMT p_example:"),
]


def outer_cli_error_output(checker_path=None):
    """Actual model-prove CLI: one clean counterexample, then a mocked spawn/IO error."""
    checker_path = Path(checker_path) if checker_path else SPEC / "tools/check_smt_obligations.py"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        script = root / "tools/check_smt_obligations.py"
        script.parent.mkdir()
        script.write_bytes(checker_path.read_bytes())
        inventory = (SPEC / "test/formal/obligations.txt").read_text()
        target = root / "test/formal/obligations.txt"
        target.parent.mkdir(parents=True)
        target.write_text(inventory)
        names = sorted(inventory.split(), key=lambda n: n.startswith("c_"))
        source = root / "source.sail"
        source.write_text("\n".join(("$counterexample" if n.startswith("c_") else "$property") +
                                    "\nfunction " + n + "(x : bool) -> bool = x" for n in names))
        queries = root / "queries"
        queries.mkdir()
        for name in names:
            (queries / (name + ".smt2")).write_text("(check-sat)\n")
        out, err = io.StringIO(), io.StringIO()
        responses = [subprocess.CompletedProcess(["z3"], 0, "sat\n", ""),
                     OSError("producer IO broke")]
        with patch.object(sys, "argv", [str(script), str(source), str(queries)]), \
             patch.object(subprocess, "run", side_effect=responses), \
             redirect_stdout(out), redirect_stderr(err):
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exit:
                if exit.code != 1:
                    raise AssertionError("CLI did not fail closed") from exit
        return out.getvalue() + err.getvalue()


def outer_extra_error_output():
    """Execute the actual embedded prove_extra producer in its all-query mode."""
    text = (SPEC / "test/formal/prove_extra.sh").read_text()
    start = text.index("import sys\nsys.path.insert(0, sys.argv[1] + \"/tools\")", text.index('  "$py" - "$spec" "$src"'))
    end = text.index("\nPY", start)
    chunk = text[start:end]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, inventory = root / "source.sail", root / "inventory.txt"
        source.write_text("$property\nfunction p_first(x : bool) -> bool = x\n"
                          "$property\nfunction p_second(x : bool) -> bool = x\n")
        inventory.write_text("p_first p_second\n")
        for name in ("p_first", "p_second"):
            (root / (name + ".smt2")).write_text("(check-sat)\n")
        out, err = io.StringIO(), io.StringIO()
        responses = [subprocess.CompletedProcess(["z3"], 0, "sat\n", ""), OSError("producer IO broke")]
        with patch.object(sys, "argv", ["-", str(SPEC), str(source), str(root), str(inventory), "0"]), \
             patch.dict(__import__("os").environ, {"PROVE_FAIL_FAST": "0"}), \
             patch.object(subprocess, "run", side_effect=responses), \
             redirect_stdout(out), redirect_stderr(err):
            try:
                exec(compile(chunk, "prove_extra.sh:actual-producer", "exec"), {"__name__": "__main__"})
            except SystemExit as exit:
                if exit.code != 1:
                    raise AssertionError("extra producer did not fail closed") from exit
        return out.getvalue() + err.getvalue()


class SolverProtocol(unittest.TestCase):
    def exercise(self, producer):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "p_example.smt2").write_text("(check-sat)\n")
            for name, stdout, stderr, code, expected in CASES:
                with self.subTest(name=name):
                    result = subprocess.CompletedProcess(["z3"], code, stdout, stderr)
                    capture = io.StringIO()
                    with patch.object(subprocess, "run", return_value=result), redirect_stdout(capture):
                        try:
                            producer(SOURCE, directory)
                        except ValueError:
                            pass
                    self.assertTrue(capture.getvalue().startswith(expected), capture.getvalue())
                    if expected.startswith("ERROR"):
                        self.assertNotIn("FAIL ", capture.getvalue())

    def test_actual_extra_outer_error_after_counterexample(self):
        output = outer_extra_error_output()
        self.assertIn("FAIL ", output)
        self.assertIn("ERROR SMT prove_extra: producer IO broke", output)

    def test_actual_cli_outer_error_after_counterexample(self):
        output = outer_cli_error_output()
        self.assertIn("FAIL ", output)
        self.assertIn("ERROR SMT model-prove: producer IO broke", output)

    def test_model_prove_producer(self):
        self.exercise(checker.check)

    def test_fptheory_solver_protocol(self):
        for name, stdout, stderr, code, expected in CASES:
            with self.subTest(name=name):
                result = subprocess.CompletedProcess(["z3"], code, stdout, stderr)
                with patch.object(subprocess, "run", return_value=result):
                    if expected.startswith("ERROR"):
                        with self.assertRaises(ValueError):
                            theory.solve("(check-sat)")
                    else:
                        self.assertEqual(theory.solve("(check-sat)"), stdout.strip())

    def test_fptheory_mismatch_is_distinct_from_infrastructure(self):
        source = str(SPEC / "test/formal/props/fptheory.sail")
        inventory = str(SPEC / "test/formal/props/fptheory.txt")
        for error, prefix in [(theory.TheoryUnproved("clean mismatch"), "fround_fp_theory:"),
                              (ValueError("bad solver"), "ERROR SMT fround_fp_theory:")]:
            capture = io.StringIO()
            with patch.object(theory, "run", side_effect=error), redirect_stderr(capture):
                self.assertEqual(theory.main([source, ".", inventory]), 1)
            self.assertTrue(capture.getvalue().startswith(prefix), capture.getvalue())

    def test_actual_fail_fast_producer(self):
        # Load the exact function from the shell's embedded Python producer.
        text = Path(__file__).with_name("prove_extra.sh").read_text()
        body = text[text.index("def check_fail_fast("):text.index("\n\ntry:", text.index("def check_fail_fast("))]
        namespace = {"c": checker, "Path": Path, "re": __import__("re"), "subprocess": subprocess}
        exec(compile(body, "prove_extra.sh:check_fail_fast", "exec"), namespace)
        self.exercise(namespace["check_fail_fast"])


if __name__ == "__main__":
    unittest.main()
