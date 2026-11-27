#!/usr/bin/env python3
"""Synthetic fault controls for certificate validation, never native evidence."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numeric_cap_certificate as cap


def synthetic_records():
    records = []
    for fmt, width in enumerate(cap.WIDTHS):
        for op in range(21, 25):
            result = dict(format=fmt, op=op, start=0, end=1 << width,
                          checked=1 << width, status="COMPLETE",
                          seconds=0, digest="0000000000000000")
            records.append(dict(identity="SYNTHETIC_UNIT_CONTROL_ONLY", exit=0,
                                stderr="", stdout=json.dumps(result), result=result))
    return records


class CoverageFaultControls(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_records()
        self.identity = "SYNTHETIC_UNIT_CONTROL_ONLY"

    def verify(self):
        return cap.validate(self.rows, self.identity, complete=True)

    def test_synthetic_complete_partition(self):
        self.assertEqual(self.verify(), cap.EXPECTED)

    def test_missing_domain(self):
        self.rows.pop()
        with self.assertRaises(ValueError):
            self.verify()

    def test_duplicate_domain(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaises(ValueError):
            self.verify()

    def test_omitted_last_binary32_pattern(self):
        self.rows[0]["result"]["end"] -= 1
        self.rows[0]["result"]["checked"] -= 1
        self.rows[0]["stdout"] = json.dumps(self.rows[0]["result"])
        with self.assertRaises(ValueError):
            self.verify()

    def test_invented_out_of_range_endpoint(self):
        self.rows[0]["result"]["end"] += 1
        with self.assertRaises(ValueError):
            self.verify()

    def test_hidden_stderr_on_resume(self):
        self.rows[0]["stderr"] = "runtime warning"
        with self.assertRaises(ValueError):
            self.verify()

    def test_hidden_native_failure(self):
        self.rows[0]["exit"] = 1
        with self.assertRaises(ValueError):
            self.verify()

    def test_tampered_result_disagrees_with_stdout(self):
        self.rows[0]["result"]["digest"] = "1111111111111111"
        with self.assertRaises(ValueError):
            self.verify()

    def test_boolean_format_rejected(self):
        self.rows[4]["result"]["format"] = True
        self.rows[4]["stdout"] = json.dumps(self.rows[4]["result"])
        with self.assertRaises(ValueError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
