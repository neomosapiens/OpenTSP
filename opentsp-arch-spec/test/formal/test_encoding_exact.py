"""encoding_exact.py accepts the shipped catalogue and rejects each kind of structural fault."""
from pathlib import Path
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoding_exact as ee

CATALOGUE = json.loads(ee.DEFAULT.read_text())


def entry(name):
    return next(e for e in CATALOGUE["instructions"] if e["name"] == name)


class Exact(unittest.TestCase):
    def mutated(self, name, change):
        catalogue = copy.deepcopy(CATALOGUE)
        change(next(e for e in catalogue["instructions"] if e["name"] == name))
        return ee.check(catalogue)

    def test_shipped_catalogue_is_exact(self):
        problems, resolved = ee.check(CATALOGUE)
        self.assertEqual(problems, [])
        self.assertEqual(len(resolved), 1)
        self.assertIn("ABC", resolved[0])

    def test_every_entry_is_checked(self):
        self.assertEqual(len(CATALOGUE["instructions"]), 78)

    def test_short_match_pattern(self):
        problems, _ = self.mutated("Read", lambda e: e.update(match=e["match"][1:]))
        self.assertTrue(any("Read: match pattern has length" in p for p in problems), problems)

    def test_reversed_field_range(self):
        problems, _ = self.mutated("Read", lambda e: e["variables"][0].update(location="32-44"))
        self.assertTrue(any("Read.addr" in p and "contiguous" in p for p in problems), problems)

    def test_field_outside_the_word(self):
        problems, _ = self.mutated("Read", lambda e: e["variables"][0].update(location="70-58"))
        self.assertTrue(any("Read.addr" in p for p in problems), problems)

    def test_overlapping_fields(self):
        problems, _ = self.mutated("Read", lambda e: e["variables"][1].update(location="33-28"))
        self.assertTrue(any("belongs to both addr and dst" in p for p in problems), problems)

    def test_wildcard_with_no_field(self):
        def drop(e):
            e["variables"].pop()
        problems, _ = self.mutated("Read", drop)
        self.assertTrue(any("belong to no field" in p for p in problems), problems)

    def test_field_on_a_fixed_bit(self):
        problems, _ = self.mutated("Read", lambda e: e["variables"][2].update(location="52"))
        self.assertTrue(any("are fixed in the match pattern" in p for p in problems), problems)

    def test_two_entries_with_overlapping_patterns(self):
        problems, _ = self.mutated("Write", lambda e: e.update(match=entry("Read")["match"]))
        self.assertTrue(any("Read and Write" in p and "no field separates" in p for p in problems), problems)

    def test_abc_overlap_needs_the_type_restriction(self):
        problems, _ = self.mutated("ABC", lambda e: e["variables"][0].update(values=[0, 1, 2, 3]))
        self.assertTrue(any("ABC and ABC8" in p for p in problems), problems)

    def test_smt_query_agrees_with_the_pair_check(self):
        def solve(a, b):
            result = subprocess.run(["z3", "-in", "-T:60"], input=ee.smt_pair_query(entry(a), entry(b)), text=True,
                                    capture_output=True)
            return result.stdout.strip()
        self.assertEqual(solve("Read", "Write"), "unsat")
        self.assertEqual(solve("Read", "Read"), "sat")

    def test_split_field_has_two_contiguous_ranges(self):
        split = next((e, v) for e in CATALOGUE["instructions"] for v in e["variables"] if "hi_location" in v)
        name = split[0]["name"]
        problems, _ = self.mutated(name, lambda e: next(v for v in e["variables"] if "hi_location" in v).update(
            hi_location="1-5"))
        self.assertTrue(problems)


class CommandLine(unittest.TestCase):
    def run_main(self, text=None, name="catalogue.json"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            if text is not None:
                path.write_text(text)
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = ee.main([str(path)])
        return code, err.getvalue()

    def test_not_json(self):
        code, err = self.run_main("not json")
        self.assertEqual(code, 2)
        self.assertIn("cannot check", err)

    def test_missing_file(self):
        code, err = self.run_main(None)
        self.assertEqual(code, 2)
        self.assertIn("FileNotFoundError", err)

    def test_entry_without_fields(self):
        code, err = self.run_main('{"instructions": [{"name": "X"}]}')
        self.assertEqual(code, 2)
        self.assertIn("KeyError", err)

    def test_a_fault_is_exit_one(self):
        bad = {"instructions": [{"name": "X", "width": 32, "match": "0101", "variables": []}]}
        self.assertEqual(self.run_main(json.dumps(bad))[0], 1)

    def test_the_shipped_catalogue_is_exit_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ee.main([]), 0)


if __name__ == "__main__":
    unittest.main()
