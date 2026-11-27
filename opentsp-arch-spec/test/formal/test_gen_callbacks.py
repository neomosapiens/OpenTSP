#!/usr/bin/env python3
"""Tests of lean/gen_callbacks.py: the Lean declarations of the model's 32 externally implemented functions."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lean"))
import gen_callbacks as gc  # noqa: E402

MODEL = HERE.parents[1] / "model"


class Declarations(unittest.TestCase):
    def setUp(self):
        self.found = gc.declarations(MODEL)

    def test_all_32_are_found(self):
        self.assertEqual(len(self.found), 32)

    def test_every_extern_of_the_model_is_declared(self):
        import re
        names = set()
        for path in MODEL.rglob("*.sail"):
            if path.parts[-2] == "prelude":
                continue          # sub_vec and sub_vec_int: C names of arithmetic the Sail library defines
            names.update(re.findall(r"^val\s+(\w+)\s*=\s*(?:pure|impure)\s*\{c:", path.read_text(), re.M))
        self.assertEqual(set(self.found), names)

    def test_kinds(self):
        pure = sorted(n for n, v in self.found.items() if v[0] == "pure")
        self.assertEqual(pure, ["diag_callback", "ecc_event_callback", "lint_callback", "mxm_corr_callback",
                                "scrub_event_callback", "spare_event_callback", "sram_write_callback", "srf_write_callback"])
        self.assertTrue(all(v[2] == "Unit" for v in self.found.values() if v[0] == "pure"))

    def test_signatures(self):
        self.assertEqual(self.found["host_recv"], ("impure", ["Nat"], "vec"))
        self.assertEqual(self.found["c2c_peer"], ("impure", ["Nat"], "Int"))
        self.assertEqual(self.found["host_chip_id"], ("impure", ["Unit"], "Nat"))
        self.assertEqual(self.found["read_sram_word"], ("impure", ["mem_slice", "Nat"], "vec"))
        self.assertEqual(self.found["iq_unit_write"], ("impure", ["Nat", "Nat", "(BitVec 32)"], "Unit"))

    def test_rendering(self):
        text = gc.render(self.found)
        self.assertIn("opaque host_recv : Nat → SailM vec", text)
        self.assertIn("def srf_write_callback (_ : Nat) (_ : Bool) (_ : Nat) (_ : Nat) (_ : vec) : Unit := ()", text)
        self.assertEqual(text.count("\nopaque "), 24)
        self.assertEqual(text.count("\ndef "), 10)      # eight trace sinks and the two shims
        self.assertIn("def tdiv_int {α : Type} [SailTrunc α]", text)
        self.assertEqual(text, gc.render(gc.declarations(MODEL)), "the output is deterministic")


class Types(unittest.TestCase):
    def test_range_sign(self):
        self.assertEqual(gc.lean_type("range(0, 87)"), "Nat")
        self.assertEqual(gc.lean_type("range(-1, 63)"), "Int")

    def test_unknown_type_is_an_error(self):
        with self.assertRaises(ValueError):
            gc.lean_type("vector(3, int)")

    def test_a_pure_callback_with_a_result_is_refused(self):
        with self.assertRaises(ValueError):
            gc.render({"f": ("pure", ["Nat"], "Nat")})

    def test_a_missing_declaration_fails_the_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rel in gc.FILES:
                target = Path(tmp) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                text = (MODEL / rel).read_text().replace("val c2c_peer = impure", "val c2c_peer_gone = impure", 1)
                target.write_text(text)
            self.assertEqual(len(gc.declarations(tmp)), 32)      # renamed, still 32
            out = Path(tmp) / "Env.lean"
            (Path(tmp) / "core/callbacks.sail").write_text(
                (MODEL / "core/callbacks.sail").read_text().replace("val host_send = impure", "val host_send_x = impure", 1)
                .replace("val host_send_x = impure {c: \"host_send\", cpp: \"host_send\"} : (/* cycle */ nat, vec) -> unit", ""))
            self.assertEqual(gc.main([tmp, str(out)]), 1)


if __name__ == "__main__":
    unittest.main()
