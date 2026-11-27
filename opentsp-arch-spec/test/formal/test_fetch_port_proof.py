"""The port proofs check production text rebound by a substitution table with exact counts."""
from pathlib import Path
import contextlib
import io
import shutil
import sys
import tempfile
import unittest

SPEC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SPEC / "tools"))
import fetch_port_proof as fpp


class PortProof(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(setattr, fpp, "SPEC", fpp.SPEC)
        self.root = Path(self.tmp.name)
        for path, _functions, _table in fpp.BINDINGS:
            (self.root / "model" / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(SPEC / "model" / path, self.root / "model" / path)
        (self.root / "test/formal").mkdir(parents=True)
        shutil.copy(SPEC / "test/formal/fetch_ports.sail", self.root / "test/formal/fetch_ports.sail")
        fpp.SPEC = self.root

    def edit(self, path, function, old, new):
        """Replace `old` (once) by `new` inside one function of a copied model file."""
        file = self.root / "model" / path
        text = file.read_text()
        start = text.index("function %s(" % function)
        end = text.find("\nfunction ", start + 1)
        end = len(text) if end < 0 else end
        body = text[start:end]
        self.assertEqual(body.count(old), 1, old)
        file.write_text(text[:start] + body.replace(old, new) + text[end:])

    def test_production_text_passes(self):
        text = fpp.build()
        self.assertNotIn("INSERT", text)
        self.assertIn("function probe_retire(", text)

    def test_every_substitution_has_an_exact_count(self):
        for _path, _functions, table in fpp.BINDINGS:
            for old, _new, expected in table:
                self.assertGreaterEqual(expected, 1, old)

    def test_an_added_use_of_a_port_fails(self):
        self.edit("slices/cu/cu_insts.sail", "stream_fetch_service", "ecc_begin();\n      match read_planes(f.source, 1, c) {",
                  "ecc_begin();\n      let _ = read_planes(f.source, 1, c);\n      match read_planes(f.source, 1, c) {")
        with self.assertRaisesRegex(ValueError, "'read_planes' made 3 substitutions, expected exactly 2"):
            fpp.build()

    def test_a_removed_use_of_a_port_fails(self):
        self.edit("postlude/tick.sail", "barrier_retire", "foreach (k from 0 to 144)", "foreach (k from 0 to 143)")
        with self.assertRaisesRegex(ValueError, "'foreach \\(k from 0 to 144\\)' made 0 substitutions, expected exactly 1"):
            fpp.build()

    def test_a_proof_name_that_clashes_with_production_text_fails(self):
        file = self.root / "model/core/csr.sail"
        text = file.read_text()
        start = text.index("function csr_write(")
        file.write_text(text[:start] + text[start:].replace("{", "{ let write_value = 0;", 1))
        with self.assertRaisesRegex(ValueError, "'write_value' occurs 8 times, expected 7"):
            fpp.build()

    def test_a_queue_dependent_retirement_fails(self):
        """One observed queue stands for all of them only while the loop body is uniform in the queue."""
        self.edit("postlude/tick.sail", "barrier_retire", "Some(r) => if c >= r then { parked_since[k] = None(); release_at[k] = None() },",
                  "Some(r) => if c >= r & k != 5 then { parked_since[k] = None(); release_at[k] = None() },")
        with self.assertRaisesRegex(ValueError, "uses its loop variable outside the index"):
            fpp.build()

    def test_the_uniformity_check_accepts_the_production_loop(self):
        production = fpp.extract((fpp.SPEC / "model/postlude/tick.sail").read_text(), ["barrier_retire"])
        fpp.check_retire_uniform(production)
        with self.assertRaisesRegex(ValueError, "loop over every queue was not found"):
            fpp.check_retire_uniform("function barrier_retire(c : nat) -> unit = ()")

    def test_diff_is_printed(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fpp.build(show_diff=True)
        self.assertIn("-function stream_fetch_start(", out.getvalue())
        self.assertIn("+function probe_start(", out.getvalue())

    def test_the_proof_body_is_the_production_body_modulo_the_table(self):
        for path, functions, table in fpp.BINDINGS:
            production = fpp.extract((fpp.SPEC / "model" / path).read_text(), functions)
            self.assertEqual(fpp.reverse(fpp.substitute(production, table), table), production)


if __name__ == "__main__":
    unittest.main()
