"""coverage_report.py merges recorder output against the static file and ignores library and test files."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_report as cr

STATIC = """F 0, "initialize_registers", "", 0, 0, 0, 0
F 1, "slot_of", "core/streams.sail", 38, 1, 38, 60
F 2, "never", "core/streams.sail", 50, 1, 52, 3
F 3, "lanes", "core/numerics/lanes.sail", 10, 1, 12, 3
F 4, "mem", "slices/mem/mem_insts.sail", 5, 1, 9, 1
F 5, "lib_fn", "/opt/sail/lib/vector.sail", 1, 1, 2, 2
F 6, "prop", "props/props.sail", 1, 1, 2, 2
F 7, "test", "unit_tests/t.sail", 1, 1, 2, 2
B 0, "core/streams.sail", 38, 1, 38, 60
T 0, 0, "core/streams.sail", 38, 5, 38, 9
T 0, 1, "core/streams.sail", 38, 20, 38, 30
"""
RUN_A = 'F 9, "slot_of", "core/streams.sail", 38, 1, 38, 60\nT 4, 0, "core/streams.sail", 38, 5, 38, 9\n'
RUN_B = 'F 1, "lanes", "core/numerics/lanes.sail", 10, 1, 12, 3\nT 0, 1, "core/streams.sail", 38, 20, 38, 30\n'


class Report(unittest.TestCase):
    def test_runs_merge_by_source_span_not_by_id(self):
        totals, covered, missed = cr.merge(STATIC, [RUN_A, RUN_B])
        self.assertEqual(sum(v for (a, k), v in covered.items() if k == "F"), 2)
        self.assertEqual(sum(v for (a, k), v in covered.items() if k == "T"), 2)

    def test_library_props_and_test_files_are_not_counted(self):
        totals, _, _ = cr.merge(STATIC, [])
        self.assertEqual(sum(v for (a, k), v in totals.items() if k == "F"), 4)
        self.assertEqual(set(a for a, _ in totals), {"core", "slices/mem"})

    def test_missed_functions_are_listed_with_their_line(self):
        _, _, missed = cr.merge(STATIC, [RUN_A])
        self.assertIn("never\tcore/streams.sail:50", missed)
        self.assertNotIn("slot_of\tcore/streams.sail:38", missed)

    def test_report_percentages(self):
        totals, covered, _ = cr.merge(STATIC, [RUN_A])
        text = cr.report(totals, covered)
        self.assertIn("functions entered 1 of 4 (25.0%)", text)
        self.assertIn("branch targets taken 1 of 2 (50.0%)", text)

    def test_unknown_line_is_an_error(self):
        with self.assertRaises(ValueError):
            list(cr.parse("X 1, 2\n"))

    def test_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            static, run, missed = (Path(tmp) / n for n in ("static", "run", "missed"))
            static.write_text(STATIC)
            run.write_text(RUN_A)
            self.assertEqual(cr.main([str(static), str(run), "--missed", str(missed)]), 0)
            self.assertIn("never", missed.read_text())
            self.assertEqual(cr.main([str(static)]), 2)


if __name__ == "__main__":
    unittest.main()
