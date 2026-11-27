#!/usr/bin/env python3
"""test/export/lean_while.py counts `while` loops in generated Lean and holds the count as a ratchet."""
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "export"))
import lean_while as lw  # noqa: E402


class Count(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "Out").mkdir()
        (self.root / "Out/Tick.lean").write_text("def a := do\n  while x do\n    pure ()\n  while y do\n    pure ()\n  let w := 1\n  -- a while in a comment? no: this line starts with a dash\n")
        (self.root / "Out/Iq.lean").write_text("def b := do\n    while z do\n      pure ()\n")
        (self.root / "Out/Plain.lean").write_text("def c := pure ()  -- meanwhile\n")

    def test_loops_are_counted_per_file(self):
        self.assertEqual(lw.count(self.root), {"Tick.lean": 2, "Iq.lean": 1})

    def test_the_verdict_is_a_ratchet(self):
        self.assertTrue(lw.verdict(29, 29)[0])
        ok, message = lw.verdict(30, 29)
        self.assertFalse(ok)
        self.assertIn("a loop was added", message)
        ok, message = lw.verdict(28, 29)
        self.assertFalse(ok)
        self.assertIn("lower the record to 28", message)

    def test_main_reads_the_record(self):
        record = self.root / "count.txt"
        record.write_text("3\n")
        self.assertEqual(lw.main([str(self.root), str(record)]), 0)
        record.write_text("2\n")
        self.assertEqual(lw.main([str(self.root), str(record)]), 1)
        (self.root / "Out/Extra.lean").write_text("def d := do\n  while q do\n    pure ()\n")
        record.write_text("3\n")
        self.assertEqual(lw.main([str(self.root), str(record)]), 1)


class Recorded(unittest.TestCase):
    def test_the_record_is_one_number(self):
        text = (HERE.parent / "export/lean_while_count.txt").read_text().split()
        self.assertEqual(len(text), 1)
        self.assertTrue(text[0].isdigit(), text[0])       # zero is a count: the bounded loops left no while loop in the Lean


if __name__ == "__main__":
    unittest.main()
