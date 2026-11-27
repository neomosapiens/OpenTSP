#!/usr/bin/env python3
"""ACC x128 uses the 32-bit lane energy counter.

Run: python3 tests/perf/test_counters_acc_price.py
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1].parent
ENERGY = ROOT / "opentsp-iss" / "perf" / "energy.yaml"
ACTIVITY = ROOT / "opentsp-iss" / "crates" / "tsp-iss" / "src" / "slices" / "activity.rs"
PERF_README = ROOT / "opentsp-iss" / "perf" / "README.md"


class AccX128StatedLimit(unittest.TestCase):
    def test_energy_table_states_x128_is_one_32bit_lane_unit(self):
        text = ENERGY.read_text()
        self.assertIn("mxm_acc_pj_per_lane", text)
        # The limit must be next to the key, not only elsewhere in the file.
        line = next(l for l in text.splitlines() if "mxm_acc_pj_per_lane" in l)
        self.assertRegex(
            line,
            r"(?=.*32-bit)(?=.*x128)",
            msg="energy.yaml must state on the mxm_acc_pj_per_lane line that ACC x128 is priced as one 32-bit lane unit",
        )

    def test_activity_comment_states_the_same_limit(self):
        text = ACTIVITY.read_text()
        self.assertIn("acc_lanes", text)
        self.assertRegex(
            text,
            r"x128.*32-bit|32-bit.*x128|priced as a 32-bit lane",
            msg="activity.rs must state that ACC x128 is counted/priced as one 32-bit lane unit",
        )

    def test_perf_readme_states_the_limit(self):
        text = PERF_README.read_text()
        self.assertRegex(
            text,
            r"ACC x128|x128.*32-bit|priced per 32-bit lane",
            msg="opentsp-iss/perf/README.md must state the ACC x128 pricing limit",
        )


if __name__ == "__main__":
    unittest.main()
