"""reach.py builds the obligation call graph from Sail's type-checked AST dump."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reach

AST = """val helper : int -> int

function helper x : int = leaf(x) + 1

function leaf x : int = x

function unused x : int = x

$[complete]
function p_good b : bits(4) = $[overloaded { "name" = "==", "is_infix" = true }] eq_bits(map_forwards(b), b) & helper(1)

$[complete]
function p_other b : bits(4) = true

function map_forwards x : bits(4) = x
"""

MODEL = {
    "core/a.sail": "function helper(x : int) -> int = leaf(x) + 1\n\n// function commented(x) = x\n"
                   "function leaf(x : int) -> int = x\nprivate function unused(x : int) -> int = x\n"
                   "mapping map : bits(4) <-> bits(4) = {\n  x <-> x\n}\n",
    "props/props.sail": "$property\nfunction p_good(b : bits(4)) -> bool = true\n",
    "unit_tests/t.sail": "function in_a_test() -> unit = ()\n",
}


class Reach(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = Path(self.tmp.name)
        for rel, text in MODEL.items():
            (self.model / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.model / rel).write_text(text)

    def run_analysis(self, ast=AST, obligations=("p_good", "p_other"), copies=()):
        return reach.analyse(ast, self.model, list(obligations), copies)

    def test_model_functions_exclude_props_unit_tests_and_comments(self):
        self.assertEqual(sorted(reach.source_functions(self.model)), ["helper", "leaf", "map", "unused"])

    def test_calls_are_followed_transitively(self):
        _, called_by, per, _ = self.run_analysis()
        self.assertEqual(per["p_good"], ["helper", "leaf", "map"])
        self.assertEqual(per["p_other"], [])
        self.assertEqual(called_by["leaf"], ["p_good"])
        self.assertEqual(called_by["unused"], [])

    def test_a_mapping_is_its_generated_functions(self):
        self.assertEqual(reach.base("encdec_eltype_forwards"), "encdec_eltype")
        self.assertEqual(reach.base("x_backwards_matches"), "x")
        self.assertEqual(reach.base("forwards_x"), "forwards_x")

    def test_an_obligation_missing_from_the_dump_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "p_gone"):
            self.run_analysis(obligations=("p_good", "p_gone"))

    def test_report_gives_counts_and_percentages(self):
        text = reach.report(*self.run_analysis(), per=True)
        self.assertIn("3 of 4 model functions", text)
        self.assertIn("p_good", text)
        self.assertIn("calls   3 model functions", text)

    def test_functions_proved_only_by_a_copy_are_named(self):
        _, _, _, copied = self.run_analysis(copies=["unused", "helper", "not_a_model_function"])
        self.assertEqual(copied, ["unused"])

    def test_comment_mentions_are_not_calls(self):
        ast = AST.replace("& helper(1)", "")
        _, called_by, _, _ = self.run_analysis(ast=ast)
        self.assertEqual(called_by["helper"], [])

    def test_lean_files_reach_what_the_functions_they_name_call(self):
        lean = self.model / "lean"
        lean.mkdir()
        (lean / "T.lean").write_text("theorem t : helper 1 = 2 := by simp [helper]\n")
        sizes = reach.source_functions(self.model)
        self.assertEqual(reach.lean_roots(lean, sizes), ["helper"])
        self.assertEqual(sorted(reach.lean_reached(AST, lean, sizes)), ["helper", "leaf"])

    def test_one_figure_per_source_and_the_union(self):
        sizes = reach.source_functions(self.model)
        text = reach.source_report({"model-prove": {"helper"}, "extra:t": {"leaf"}, "lean": {"helper", "leaf"}}, sizes)
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("model-prove", lines[0])
        self.assertIn("1 of 4 model functions", lines[0])
        self.assertIn("2 of 4 model functions", lines[-1])
        self.assertIn("union of all sources", lines[-1])

    def test_main_prints_a_figure_per_source(self):
        import contextlib, io
        ast = Path(self.tmp.name) / "ast.txt"
        ast.write_text(AST)
        inventory = Path(self.tmp.name) / "inv.txt"
        inventory.write_text("p_other\n")
        obligations = Path(self.tmp.name) / "obl.txt"
        obligations.write_text("p_good\n")
        lean = self.model / "lean"
        lean.mkdir()
        (lean / "T.lean").write_text("theorem t : leaf 1 = 1 := rfl\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = reach.main([str(ast), "--model", str(self.model), "--obligations", str(obligations),
                               "--source", "t=%s:%s" % (ast, inventory), "--lean", str(lean)])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("model-prove", text)
        self.assertIn("extra:t", text)
        self.assertIn("lean", text)
        self.assertIn("union of all sources", text)

    def test_model_prove_passes_every_extra_suite_to_the_reach_report(self):
        # : the line "union of all sources" is a union only when the extra SMT suites are sources of it.
        text = (reach.SPEC / "tools/model_prove.sh").read_text()
        for suite in ("ties", "frontend", "numerics"):
            self.assertIn("--source", text)
            self.assertIn("extra_ast/$suite.ast:$here/test/formal/props/$suite.txt", text)
        self.assertIn("PROVE_EXTRA_AST_DIR", text)

    def test_the_shipped_obligations_are_all_found_in_a_dump(self):
        names = (reach.SPEC / "test/formal/obligations.txt").read_text().split()
        self.assertEqual(len(names), 33)


if __name__ == "__main__":
    unittest.main()
