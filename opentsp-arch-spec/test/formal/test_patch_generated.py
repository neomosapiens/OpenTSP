#!/usr/bin/env python3
"""Tests of lean/patch_generated.py: the exact rules that make the generated Lean elaborate."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lean"))
import patch_generated as pg  # noqa: E402

ASSERT_SRC = "def c2c_links : Int :=\n  unwrapValue (let n : Int := 16\n  assert ((1 ≤b n) && (n ≤b 16)) \"msg\"\n  (pure n))\n"
LIST_SRC = ("  let cands ← (( do\n    [(pure base), (pure (GetElem?.getElem! (← readReg queue_busy_until) q)), (pure (f\n"
            "        (← readReg a) (2 *i q)))] ) : SailM (List Nat) )\n")


class Rules(unittest.TestCase):
    def test_assert_rule(self):
        import re
        out, count = re.subn(pg.RULES[0][1], pg.RULES[0][2], ASSERT_SRC)
        self.assertEqual(count, 1)
        self.assertIn("unwrapValue (do\n  let n : Int := 16", out)

    def test_pure_list_is_lifted(self):
        out, count = pg.lift_pure_list(LIST_SRC)
        self.assertEqual(count, 1)
        self.assertIn("(pure [base, (GetElem?.getElem! (← readReg queue_busy_until) q), (f (← readReg a) (2 *i q))])", out)
        self.assertNotIn("(pure base)", out)

    def test_a_list_with_an_impure_element_is_refused(self):
        with self.assertRaises(ValueError):
            pg.lift_pure_list(LIST_SRC.replace("(pure (f", "((f"))

    def test_keywords_renamed_except_open_statements(self):
        text = "open fclass\n  let open := (a && b)\n  if (open : Bool)\n/-- open the door -/\n  let matches : Bool :=\n"
        out, count = pg.rename_keywords(text)
        self.assertEqual(count, 3)
        self.assertEqual(out.split("\n")[0], "open fclass")
        self.assertIn("let open_ := (a && b)", out)
        self.assertIn("let matches_ : Bool", out)
        self.assertIn("/-- open the door -/", out)

    def test_a_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("C2cInsts.lean", "MxmInsts.lean", "InstsEnd.lean", "Tick.lean"):
                Path(tmp, name).write_text("-- nothing to patch\n")
            with self.assertRaises(ValueError):
                pg.patch(tmp)


if __name__ == "__main__":
    unittest.main()
