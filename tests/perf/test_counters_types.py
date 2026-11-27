#!/usr/bin/env python3
"""The Python pricing of tests/perf/test_counters_xcheck.py treats bf16 as the float type it is: the float scale of
vxm.by_class_type applies to bf16 and not to a mask type that starts with the same letter, as the ISS does (`EType::is_float`).

Run: python3 tests/perf/test_counters_types.py   (no simulator needed)."""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("xcheck", os.path.join(HERE, "test_counters_xcheck.py"))
xc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xc)

class FloatScale(unittest.TestCase):
    def test_bf16_is_a_float_type(self):
        self.assertIn("bf16", xc.TYPES)
        self.assertEqual(xc.FLOAT_TYPES, {"f16", "f32", "e4m3", "e5m2", "bf16"})

    def test_mask_and_integer_types_are_not(self):
        for ty in ("b8", "b16", "b32", "i8", "u16"):
            self.assertNotIn(ty, xc.FLOAT_TYPES)


if __name__ == "__main__":
    unittest.main()
