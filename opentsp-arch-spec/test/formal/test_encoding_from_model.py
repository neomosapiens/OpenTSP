#!/usr/bin/env python3
"""encoding_from_model.py compares the catalogue's fixed bits with the bits the model treats as fixed (a fake model stands in)."""
from pathlib import Path
import contextlib
import io
import json
import re
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoding_from_model as efm  # noqa: E402

FAKE = """#!/usr/bin/env python3
import json, sys
table = json.load(open(sys.argv[1] + ".table"))
for line in sys.stdin:
    mode, word = line.split()
    width, word = int(mode[1:]), int(word, 16)
    name = "-"
    for entry in table:
        if entry["width"] == width and word & entry["mask"] == entry["value"] and (entry.get("also") is None or entry["also"](word)):
            name = entry["name"]
            break
    print(name + "\\tCU", flush=True)
"""

# 8 bits would do, but the catalogue is 32-bit: A fixes the top 8 bits (0x01) and has a 4-bit field at 7-4 with legal values 0..3 (a values list) and
# a free field at 3-0; B fixes the top 8 bits to 0x02 and has a free 8-bit field.
CATALOGUE = {"instructions": [
    {"name": "A", "width": 32, "match": "00000001" + "-" * 8 + "0" * 8 + "-" * 8, "variables": [
        {"name": "x", "location": "7-0", "kind": "imm8"}, {"name": "y", "location": "23-16", "kind": "imm8", "values": [0, 1, 2, 3]}]},
    {"name": "B", "width": 32, "match": "00000010" + "0" * 8 + "-" * 16, "variables": [{"name": "z", "location": "15-0", "kind": "imm16"}]},
]}


class ExportedQueryInterface(unittest.TestCase):
    def test_native_query_external_functions_survive_sail_dce(self):
        spec = Path(__file__).resolve().parents[2]
        query = (spec / 'test/isa_db/isa_query.c').read_text()
        build = (spec / 'test/formal/encoding_model.sh').read_text()
        external = set(re.findall(r'^extern .*?\b(zisa_\w+)\s*\(', query, re.M))
        self.assertTrue(external, 'must inspect the actual C query interface')
        roots = set(re.findall(r'--c-preserve\s+(isa_\w+)', build))
        missing = {name[1:] for name in external} - roots
        self.assertEqual(missing, set(), 'C query functions removed by Sail dead-code elimination')


class Probe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exe = self.root / "fake"
        self.exe.write_text(FAKE)
        self.exe.chmod(self.exe.stat().st_mode | stat.S_IEXEC)

    def table(self, entries):
        (self.root / "fake.table").write_text(json.dumps(entries))

    def run_probe(self, catalogue=CATALOGUE):
        model = efm.Model(str(self.exe), str(self.root / "fake"))
        try:
            return efm.run(model, catalogue)
        finally:
            model.close()

    # The model that agrees with the catalogue: A is 0x01 in 31-24, 0x00 in 15-8, any 23-16 below 4 ... (the legality is out of the table's mask).
    def agreeing(self):
        return [{"name": "A", "width": 32, "mask": 0xFFFFFF00 & ~0x00FF0000 | 0, "value": 0x01000000},
                {"name": "B", "width": 32, "mask": 0xFFFF0000, "value": 0x02000000}]

    def test_a_model_that_agrees_has_no_problems(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF00FF00      # bits 31-24 and 15-8 fixed, the fields at 23-16 and 7-0 free
        self.table(entries)
        problems, probes, flips, widths = self.run_probe()
        self.assertEqual(problems, [])
        self.assertEqual(widths, {32: [2, 64]})
        self.assertGreater(probes, 2)

    def test_a_bit_fixed_in_the_catalogue_but_free_in_the_model_is_an_error(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF000000      # the model does not fix bits 15-8 at all
        self.table(entries)
        problems = self.run_probe()[0]
        self.assertTrue(any("A: bit 8 is fixed in the catalogue but the model keeps the word" in p for p in problems), problems)

    def test_a_bit_free_in_the_catalogue_but_fixed_in_the_model_is_an_error_without_a_restriction(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF00FF01      # the model also fixes bit 0 of the unrestricted field x
        entries[0]["value"] = 0x01000000
        self.table(entries)
        problems = self.run_probe()[0]
        self.assertTrue(any("A: bit 0 is free in the catalogue (field x, no restriction listed)" in p for p in problems), problems)

    def test_the_same_bit_in_a_field_with_a_values_list_is_legal(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFFFCFF00      # bits 23-18 are fixed to 0: y keeps 0..3, as its `values` list says
        self.table(entries)
        self.assertEqual(self.run_probe()[0], [])

    def test_a_restricted_top_piece_is_owned_by_its_declared_field(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF00FF01  # model rejects top-piece bit0 for this profile
        self.table(entries)
        catalogue = json.loads(json.dumps(CATALOGUE))
        catalogue["instructions"][0]["variables"][0]["location"] = "7-1"
        catalogue["instructions"][0]["variables"][1]["top_location"] = "0"
        self.assertEqual(self.run_probe(catalogue)[0], [])

    def test_an_unrestricted_top_piece_still_reports_a_fixed_model_bit(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF00FF01
        self.table(entries)
        catalogue = json.loads(json.dumps(CATALOGUE))
        catalogue["instructions"][0]["variables"][0]["location"] = "7-1"
        catalogue["instructions"][0]["variables"][0]["top_location"] = "0"
        problems = self.run_probe(catalogue)[0]
        self.assertTrue(any("bit 0 is free in the catalogue (field x, no restriction listed)" in x for x in problems), problems)

    def test_an_entry_that_no_word_reaches_is_an_error(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF00FF00
        entries[1]["name"] = "C"             # the model has no entry B
        self.table(entries)
        problems = self.run_probe()[0]
        self.assertTrue(any(p.startswith("B: no word") for p in problems), problems)

    def test_main_prints_a_summary_and_the_exit_status(self):
        entries = self.agreeing()
        entries[0]["mask"] = 0xFF00FF00
        self.table(entries)
        catalogue = self.root / "encoding.json"
        catalogue.write_text(json.dumps(CATALOGUE))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = efm.main([str(self.exe), str(self.root / "fake"), str(catalogue)])
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("2 entries", out.getvalue())
        entries[0]["mask"] = 0xFF000000
        self.table(entries)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(efm.main([str(self.exe), str(self.root / "fake"), str(catalogue)]), 1)


if __name__ == "__main__":
    unittest.main()
