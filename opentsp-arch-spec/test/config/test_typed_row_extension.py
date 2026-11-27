"""A timing row keyed by an element type that only an extension defines (`Max.bf16`) is refused by `schema_invariants.py --check` when the
profile does not declare that extension. The simulators read such a row without a message (the type has no instruction in that profile, so
the row is never used); the check tells a profile author, and it runs on every shipped profile (`model_check.sh`)."""
import copy, json, subprocess, sys, tempfile, unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2]
from opentsp_reference import schema as si

def load(rel):
    with open(SPEC / rel) as fh:
        return json.load(fh)


REF = load("config/ref.json")
EXT = load("config/test/ext_bf16.json")


def with_row(cfg, entry):
    c = copy.deepcopy(cfg)
    c["timing"]["overrides"].append({"entry": entry, "d_func": 6, "d_skew": 2})
    return c


class TypedRowNeedsItsExtension(unittest.TestCase):
    def test_bf16_row_without_the_extension_is_refused(self):
        errs = si.check(with_row(REF, "Max.bf16"))
        self.assertEqual(len(errs), 1, errs)
        self.assertIn("Max.bf16", errs[0])
        self.assertIn("extension bf16", errs[0])

    def test_bf16_row_with_the_extension_is_accepted(self):
        self.assertEqual(si.check(with_row(EXT, "Max.bf16")), [])

    def test_rows_of_always_defined_types_and_untyped_rows_are_accepted(self):
        for entry in ("Max.i8", "Max.f32", "Exp.f16", "Add.e4m3", "Max"):
            self.assertEqual(si.check(with_row(REF, entry)), [], entry)

    def test_the_command_line_check_exits_1_and_names_the_row(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "p.json"
            with open(f, "w") as fh:
                json.dump(with_row(REF, "Exp.bf16"), fh)
            r = subprocess.run([sys.executable, str(SPEC / "tools/schema_invariants.py"), "--check", str(f)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("Exp.bf16", r.stdout)

    def test_shipped_profiles_have_no_such_row(self):
        for p in sorted((SPEC / "config").glob("*.json")) + sorted((SPEC / "config/test").glob("*.json")):
            if p.name == "schema.json":
                continue
            self.assertEqual(si.check(load(p.relative_to(SPEC))), [], p.name)


if __name__ == "__main__":
    unittest.main()
