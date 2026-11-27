#!/usr/bin/env python3
"""check_axioms.py accepts the three axioms of Lean's core and refuses any other (a user axiom, sorryAx, ofReduceBool)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / "lean"))
import check_axioms as ca  # noqa: E402

AXIOMS = "import Proofs.IqPop\nopen Proofs\n\n#print axioms Proofs.iq_cap_pos\n#print axioms iq_pop_conservation\n#print axioms barrier_window\n"
GOOD = ("'Proofs.iq_cap_pos' does not depend on any axioms\n"
        "'iq_pop_conservation' depends on axioms: [propext, Classical.choice, Quot.sound]\n"
        "'barrier_window' depends on axioms: [propext]\n")


class Axioms(unittest.TestCase):
    def test_the_allowed_axioms_pass(self):
        found, count = ca.problems(GOOD, AXIOMS)
        self.assertEqual((found, count), ([], 3))

    def test_a_user_axiom_is_refused(self):
        log = GOOD.replace("[propext]", "[propext, Proofs.my_axiom]")
        found, _ = ca.problems(log, AXIOMS)
        self.assertEqual(len(found), 1)
        self.assertIn("my_axiom", found[0])

    def test_sorry_and_native_decide_are_refused(self):
        for axiom in ("sorryAx", "Lean.ofReduceBool"):
            with self.subTest(axiom=axiom):
                found, _ = ca.problems(GOOD.replace("[propext]", "[propext, %s]" % axiom), AXIOMS)
                self.assertEqual(len(found), 1)

    def test_a_missing_theorem_is_refused(self):
        found, _ = ca.problems(GOOD.replace("'barrier_window' depends on axioms: [propext]\n", ""), AXIOMS)
        self.assertEqual(found, ["barrier_window: no `#print axioms` line in the log"])

    def test_a_theorem_the_axioms_file_does_not_name_is_refused(self):
        found, _ = ca.problems(GOOD + "'extra' does not depend on any axioms\n", AXIOMS)
        self.assertEqual(len(found), 1)

    def test_a_long_list_wrapped_over_lines_is_read(self):
        log = GOOD.replace("[propext, Classical.choice, Quot.sound]", "[propext,\n  Classical.choice,\n  Quot.sound]")
        self.assertEqual(ca.problems(log, AXIOMS)[0], [])

    def test_an_empty_axioms_file_is_refused(self):
        self.assertEqual(ca.problems(GOOD, "import X\n")[0][0], "Axioms.lean names no theorem")

    def test_namespace_collisions_cannot_hide_a_forbidden_axiom(self):
        wanted = "#print axioms Left.same\n#print axioms Right.same\n"
        bad = "'Left.same' depends on axioms: [sorryAx]\n"
        good = "'Right.same' depends on axioms: [propext]\n"
        for log in (bad + good, good + bad):
            with self.subTest(log=log):
                found, _ = ca.problems(log, wanted)
                self.assertTrue(any("Left.same" in problem and "sorryAx" in problem
                                    for problem in found), found)
                self.assertFalse(any("Right.same" in problem and "outside" in problem
                                     for problem in found), found)

    def test_same_name_duplicates_cannot_erase_a_forbidden_axiom(self):
        wanted = "#print axioms Proofs.same\n"
        bad = "'Proofs.same' depends on axioms: [sorryAx]\n"
        for good in ("'Proofs.same' depends on axioms: [propext]\n",
                     "'Proofs.same' does not depend on any axioms\n"):
            for log in (bad + good, good + bad):
                with self.subTest(log=log):
                    found, _ = ca.problems(log, wanted)
                    self.assertTrue(any("sorryAx" in problem for problem in found), found)

    def test_allowed_duplicates_still_pass(self):
        wanted = "#print axioms Proofs.same\n"
        log = ("'Proofs.same' depends on axioms: [propext]\n"
               "'Proofs.same' does not depend on any axioms\n"
               "'Proofs.same' depends on axioms: [Quot.sound]\n")
        self.assertEqual(ca.problems(log, wanted), ([], 1))

    def test_another_namespace_cannot_satisfy_a_missing_theorem(self):
        found, _ = ca.problems("'Right.same' does not depend on any axioms\n",
                               "#print axioms Left.same\n")
        self.assertIn("Left.same: no `#print axioms` line in the log", found)

    def test_shared_common_theorems_are_explicitly_registered(self):
        import re
        proofdir = Path(__file__).resolve().parent / "lean" / "proofs"
        common = (proofdir / "Common.lean").read_text()
        registered = ca.theorems((proofdir / "Axioms.lean").read_text())
        for name in re.findall(r"^theorem ([A-Za-z_][A-Za-z_0-9]*)", common, re.M):
            with self.subTest(theorem=name):
                self.assertIn("Proofs." + name, registered)

    def test_attributed_numerical_theorems_are_explicitly_registered(self):
        import re
        proofdir = Path(__file__).resolve().parent / "lean" / "proofs"
        source = (proofdir / "Numerics.lean").read_text()
        registered = ca.theorems((proofdir / "Axioms.lean").read_text())
        attributed = re.findall(r"^@\[[^\]]+\]\s+theorem\s+(\w+)", source, re.M)
        self.assertEqual(len(attributed), 16)
        for name in attributed:
            with self.subTest(theorem=name):
                self.assertIn("NumericalProofs." + name, registered)

    def test_declaration_inventory_handles_attributes_modifiers_and_scopes(self):
        source = """namespace Outer
/- theorem omitted : True := by trivial /- theorem nested : True := by trivial -/ -/
section localProofs
@[simp] theorem first : True := by trivial
@[simp]
@[inline] lemma second : True := by trivial
private theorem hidden : True := by trivial
end localProofs
namespace Inner
protected theorem third : True := by trivial
def text := "theorem fake : True := by trivial"
end Inner
end Outer
"""
        self.assertEqual(ca.declared_theorems(source),
                         ["Outer.first", "Outer.second", "Outer.Inner.third"])

    def test_complete_current_public_declaration_inventory_is_registered(self):
        proofdir = Path(__file__).resolve().parent / "lean" / "proofs"
        axioms = (proofdir / "Axioms.lean").read_text()
        self.assertEqual(ca.inventory_problems(proofdir, axioms), [])
        # A real attributed declaration remains a coverage failure even if all
        # remaining registered axiom lists would be permitted by the kernel audit.
        mutant = axioms.replace("#print axioms NumericalProofs.cls_FZero_FZero\n", "")
        self.assertEqual(ca.inventory_problems(proofdir, mutant),
                         ["NumericalProofs.cls_FZero_FZero: public declaration missing explicit axiom registration"])

    def test_inventory_refuses_unclosed_comment_or_scope(self):
        for source in ("/- theorem hidden : True := by trivial", "namespace Unclosed\n"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    ca.declared_theorems(source)

    def test_inventory_includes_registry_and_nested_source_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Nested").mkdir()
            (root / "Nested" / "One.lean").write_text(
                "namespace A\n@[simp] theorem x : True := by trivial\nend A\n")
            registry = "#print axioms A.x\ntheorem unregistered_in_registry : True := by trivial\n"
            (root / "Axioms.lean").write_text(registry)
            self.assertEqual(ca.inventory_problems(root, registry),
                             ["unregistered_in_registry: public declaration missing explicit axiom registration"])

    def test_main_prints_the_count_and_the_exit_status(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            log, lean = Path(tmp, "log"), Path(tmp, "Axioms.lean")
            lean.write_text(AXIOMS)
            log.write_text(GOOD)
            self.assertEqual(ca.main([str(log), str(lean)]), 0)
            log.write_text(GOOD.replace("[propext]", "[sorryAx]"))
            self.assertEqual(ca.main([str(log), str(lean)]), 1)


if __name__ == "__main__":
    unittest.main()
