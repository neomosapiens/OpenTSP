"""The proof gate distinguishes names, not just global sat/unsat counts."""
from pathlib import Path
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from check_smt_obligations import check, check_inventory, check_vacuity, obligations, solver_info, twin_source

SOURCE = "$property\nfunction p(x : bool) -> bool = true\n" \
         "$counterexample\nfunction c(x : bool) -> bool = false\n"


class ProofGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.write("p", "false")
        self.write("c", "true")

    def write(self, name, expression):
        (self.root / (name + ".smt2")).write_text("(assert %s)\n(check-sat)\n" % expression)

    def run_check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            check(SOURCE, self.root)

    def test_valid(self):
        self.run_check()

    def test_paired_opposite_mutations(self):
        self.write("p", "true")
        self.write("c", "false")
        with self.assertRaisesRegex(ValueError, "unproved obligations: p, c"):
            self.run_check()

    def test_missing(self):
        (self.root / "p.smt2").unlink()
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.run_check()

    def test_unexpected(self):
        self.write("extra", "false")
        with self.assertRaisesRegex(ValueError, "inventory mismatch"):
            self.run_check()

    def test_deleting_a_theorem_and_its_generated_query_is_detected(self):
        with self.assertRaisesRegex(ValueError, "proof contract mismatch"):
            check_inventory(SOURCE.split("$counterexample")[0], "p c")

    def test_duplicate_declaration(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            obligations(SOURCE + SOURCE)

    def test_solver_error_after_valid_result(self):
        with (self.root / "p.smt2").open("a") as f:
            f.write("(not-a-command)\n")
        with self.assertRaisesRegex(ValueError, "unproved"):
            self.run_check()

    def test_no_result(self):
        (self.root / "p.smt2").write_text("(assert false)\n")
        with self.assertRaisesRegex(ValueError, "unproved"):
            self.run_check()


class SolverInfo(unittest.TestCase):
    """The gate names the solver, its version against the pin and its effective timeouts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(os.environ.__setitem__, "PATH", os.environ["PATH"])
        os.environ["PATH"] = str(self.root) + os.pathsep + os.environ["PATH"]

    def solver(self, version, body=""):
        path = self.root / "z3"
        path.write_text("#!/bin/sh\n%s\necho 'Z3 version %s - 64 bit'\n" % (body, version))
        path.chmod(0o755)

    def pin(self, version):
        (self.root / "pin.txt").write_text(version + "\n")
        return self.root / "pin.txt"

    def test_pinned_version_is_reported(self):
        self.solver("4.15.4")
        line = solver_info(self.pin("4.15.4"))
        self.assertIn("Z3 version 4.15.4", line)
        self.assertIn("pinned 4.15.4", line)
        self.assertIn("requested 60 s", line)

    def test_other_version_is_rejected(self):
        self.solver("4.13.0")
        with self.assertRaisesRegex(ValueError, "4.13.0 is not the pinned 4.15.4"):
            solver_info(self.pin("4.15.4"))

    def test_other_version_can_be_allowed_with_a_warning(self):
        self.solver("4.13.0")
        self.assertIn("warning: z3 4.13.0 is not the pinned 4.15.4", solver_info(self.pin("4.15.4"), allow=True))

    def test_wrapper_timeout_rewrite_is_shown(self):
        # The wrapper is recognised by its rewrite clause, as the toolchain's wrapper writes it.
        path = self.root / "z3"
        path.write_text("#!/bin/bash\nargs=(); for a in \"$@\"; do case \"$a\" in -T:*) args+=(\"-T:300\");; "
                        "*) args+=(\"$a\");; esac; done\necho 'Z3 version 4.15.4 - 64 bit'\n")
        path.chmod(0o755)
        line = solver_info(self.pin("4.15.4"))
        self.assertIn("rewriting -T to 300 s", line)
        self.assertIn("effective 300 s", line)

    def test_the_limits_of_a_caller_that_overrides_them_are_the_ones_printed(self):
        self.solver("4.15.4")
        line = solver_info(self.pin("4.15.4"), timeout=240, kill_after=250)
        self.assertIn("timeout requested 240 s", line)
        self.assertIn("kills a query after 250 s", line)
        self.assertIn("timeout requested 60 s", solver_info(self.pin("4.15.4")))

    def test_missing_solver(self):
        os.environ["PATH"] = str(self.root)
        with self.assertRaisesRegex(ValueError, "not found"):
            solver_info(self.pin("4.15.4"), solver="no-such-solver")


def prop(body, name="p_x"):
    return "$property\nfunction %s(x : bits(4)) -> bool = %s\n" % (name, body)


class Vacuity(unittest.TestCase):
    """A property body that can raise counts as proved (the path is unsat), so it is rejected."""

    def reject(self, source):
        with self.assertRaisesRegex(ValueError, "vacuous"):
            check_vacuity(source)

    def test_each_keyword_in_a_property_body(self):
        for word, call in (("throw", "throw Error_not_implemented(\"x\")"),
                           ("internal_error", "internal_error(__FILE__, __LINE__, \"x\")"),
                           ("not_implemented", "not_implemented(\"x\")"),
                           ("exit", "exit()"),
                           ("assert", "assert(x == x, \"m\")")):
            with self.subTest(word=word):
                self.reject(prop("{\n  let y = %s;\n  true\n}" % call))

    def test_keyword_inside_an_if_branch(self):
        self.reject(prop("{\n  if x == 0b0000 then throw Error_not_implemented(\"x\") else ();\n  true\n}"))

    def test_keyword_in_the_second_of_several_properties(self):
        good = prop("true", "p_a")
        bad = prop("{\n  assert(false);\n  true\n}", "p_b")
        self.reject(good + bad)

    def test_message_names_the_property_and_the_word(self):
        with self.assertRaisesRegex(ValueError, "p_b.*throw"):
            check_vacuity(prop("true", "p_a") + prop("{\n  throw Error_not_implemented(\"x\")\n}", "p_b"))

    def test_comments_do_not_count(self):
        check_vacuity(prop("{\n  // never throw here, assert nothing\n  /* internal_error */\n  true\n}"))

    def test_word_boundaries(self):
        check_vacuity(prop("{\n  let exit_code = 1;\n  let assertion = 2;\n  let rethrown = 3;\n  true\n}"))

    def test_counterexamples_and_helpers_are_not_checked(self):
        source = ("function helper(x : bool) -> bool = { assert(x); true }\n" +
                  "$counterexample\nfunction c_x(x : bits(4)) -> bool = { assert(x == x); false }\n" +
                  prop("true"))
        check_vacuity(source)

    def test_property_ends_at_the_next_declaration(self):
        source = prop("{\n  true\n}") + "function helper(x : bool) -> bool = { assert(x); true }\n"
        check_vacuity(source)

    def test_main_rejects_before_solving(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "obligations.txt"
            source.write_text(prop("{\n  throw Error_not_implemented(\"x\")\n}"))
            result = subprocess.run([sys.executable, str(TOOLS / "check_smt_obligations.py"), str(source), tmp],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("vacuous", result.stderr)


THROWING = "function %s(x : bits(8)) -> bool = { if unsigned(x) < 300 then throw E_boom(); false }\n"


class VacuityLayouts(unittest.TestCase):
    """What stands between `$property` and its `function` must not hide the property from the check: the layouts of a
    property that always raises, each of which Sail accepts and turns into a query the solver answers `unsat`."""

    def reject(self, source):
        with self.assertRaisesRegex(ValueError, "vacuous"):
            check_vacuity(source)

    def test_plain_layout(self):
        self.reject("$property\n" + THROWING % "p_plain")

    def test_a_comment_line_between_the_pragma_and_the_function(self):
        self.reject("$property\n// a comment between the pragma and the function\n" + THROWING % "p_commented")

    def test_a_blank_line_between_the_pragma_and_the_function(self):
        self.reject("$property\n\n" + THROWING % "p_blank")

    def test_a_block_comment_and_blank_lines(self):
        self.reject("$property\n\n/* why\n   this */\n\n" + THROWING % "p_block")

    def test_the_function_on_the_line_of_the_pragma(self):
        self.reject("$property " + THROWING % "p_inline")

    def test_blanks_before_a_property_that_follows_a_commented_one(self):
        self.reject(prop("true", "p_a") + "\n$property\n// note\n\n" + THROWING % "p_b")

    def test_a_declaration_other_than_a_function_after_the_pragma_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not followed directly by a function"):
            check_vacuity("$property\nlet unrelated = 1\n" + prop("true", "p_a"))

    def test_a_pragma_at_the_end_of_the_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not followed directly by a function"):
            check_vacuity(prop("true") + "$property\n")

    def test_a_clean_property_in_any_of_those_layouts_is_accepted(self):
        for layout in ("$property\n", "$property\n// note\n", "$property\n\n", "$property ", "$property\n/* note */\n\n"):
            with self.subTest(layout=layout):
                check_vacuity(layout + "function p_x(x : bits(4)) -> bool = true\n")


class Twins(unittest.TestCase):
    """The twin of a property is a counterexample that is satisfiable when the property can complete and return true."""

    def test_one_twin_per_property_with_the_same_parameters(self):
        source = (prop("true", "p_a") + "$counterexample\nfunction c_b(x : bits(4)) -> bool = false\n" +
                  "$property\nfunction p_b(h : bits(10), r : range(0, 144), o : bits(3)) -> bool = true\n")
        twins = twin_source(source)
        self.assertEqual(twins.count("$counterexample"), 2)
        self.assertIn("function twin_p_a(x : bits(4)) -> bool = {\n  let claim : bool = {\ntrue\n  };\n  not(claim)\n}", twins)
        self.assertIn("function twin_p_b(h : bits(10), r : range(0, 144), o : bits(3)) -> bool = {\n  let claim : bool = {\ntrue\n  };", twins)
        self.assertEqual(obligations(source + twins)["twin_p_a"], "sat")

    def test_the_body_is_copied_up_to_the_next_declaration(self):
        source = prop("{\n  let y = x;\n  y == x\n}", "p_a") + "function helper(x : bool) -> bool = x\n"
        twins = twin_source(source)
        self.assertIn("let claim : bool = {\n{\n  let y = x;\n  y == x\n}\n  };", twins)
        self.assertNotIn("helper", twins)

    def test_a_parameter_list_over_several_lines(self):
        source = "$property\nfunction p_m(a : bits(5), ks : bits(5),\n                u : bits(2)) -> bool = true\n"
        self.assertIn("function twin_p_m(a : bits(5), ks : bits(5), u : bits(2)) -> bool = {\n  let claim", twin_source(source))

    def test_no_properties_no_twins(self):
        self.assertEqual(twin_source("function f(x : bool) -> bool = x\n$counterexample\nfunction c(x : bool) -> bool = x\n"), "")

    def test_a_property_that_returns_something_else_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not return bool"):
            twin_source("$property\nfunction p_n(x : bits(4)) -> int = 1\n")

    def test_a_name_clash_is_refused(self):
        with self.assertRaisesRegex(ValueError, "twin_p_x already exists"):
            twin_source(prop("true") + "function twin_p_x(x : bits(4)) -> bool = true\n")

    def test_comments_are_ignored(self):
        self.assertEqual(twin_source("// $property\nfunction p(x : bool) -> bool = x\n"), "")


@unittest.skipUnless(shutil.which("sail") and shutil.which("z3"), "needs sail and z3 (the specification toolchain)")
class TwinsAgainstTheSolver(unittest.TestCase):
    """End to end with Sail and Z3: the false property that raises is `unsat` (proved) and its twin is `unsat` too, so the
    gate flags it; a real property has a satisfiable twin."""

    HEAD = ("default Order dec\n$include <prelude.sail>\nunion exception = { E_boom : unit }\n"
            "val not : bool -> bool\nfunction not(b) = if b then false else true\n")

    def queries(self, body):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "t.sail"
        path.write_text(body + "\n" + twin_source(body))
        result = subprocess.run(["sail", "-smt", str(path)], cwd=tmp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return tmp.name, obligations(path.read_text())

    def verdicts(self, body):
        directory, wanted = self.queries(body)
        got = {}
        for name in wanted:
            query = re.sub(r"(?m)^\(get-model\)\s*$", "", (Path(directory) / (name + ".smt2")).read_text())
            got[name] = subprocess.run(["z3", "-in", "-T:60"], input=query, text=True, capture_output=True).stdout.strip()
        return wanted, got

    def test_a_property_that_always_raises_has_an_unsatisfiable_twin(self):
        for layout in ("$property\n", "$property\n// c\n", "$property\n\n"):
            with self.subTest(layout=layout):
                wanted, got = self.verdicts(self.HEAD + layout + THROWING % "p_t")
                self.assertEqual(got["p_t"], "unsat")             # "proved"
                self.assertEqual(got["twin_p_t"], "unsat")        # but the twin is not satisfiable: the obligation is vacuous
                self.assertNotEqual(got["twin_p_t"], wanted["twin_p_t"])

    def test_a_property_that_completes_has_a_satisfiable_twin(self):
        wanted, got = self.verdicts(self.HEAD + "$property\nfunction p_ok(x : bits(8)) -> bool = unsigned(x) < 300\n")
        self.assertEqual((got["p_ok"], got["twin_p_ok"]), ("unsat", "sat"))
        self.assertEqual(got, wanted)


if __name__ == "__main__":
    unittest.main()
