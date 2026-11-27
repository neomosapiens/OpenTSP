"""The schema admits machines whose extra MXM rows are not observable; schema_invariants.py warns (advisory, not a schema relation)."""
import copy, json, subprocess, sys, tempfile, unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2]
from opentsp_reference import schema as si

MINI = json.loads((SPEC / "config/mini.json").read_text())


def variant(rows, superlanes):
    c = copy.deepcopy(MINI)
    c["mxm"]["rows"] = rows
    c["mxm"]["f16_rows"] = rows // 2
    c["geometry"]["superlanes"] = superlanes
    return c


class Advisories(unittest.TestCase):
    def test_rows_beyond_vlanes_warn(self):
        adv = si.advisories(variant(320, 1))
        self.assertEqual(len(adv), 1, adv)
        self.assertIn("mxm.rows", adv[0])
        self.assertIn("RES-002", adv[0])

    def test_rows_up_to_vlanes_do_not_warn(self):
        for rows, sl in ((16, 1), (320, 20), (160, 20), (32, 4)):
            self.assertEqual(si.advisories(variant(rows, sl)), [], (rows, sl))

    def test_shipped_profiles(self):
        """ref, mini and every test profile are clean except mini_small, which keeps 48 rows on one superlane on purpose (it tests RES-002 with a deep buffer)."""
        for p in sorted((SPEC / "config").glob("*.json")) + sorted((SPEC / "config/test").glob("*.json")):
            if p.name == "schema.json":
                continue
            adv = si.advisories(json.loads(p.read_text()))
            self.assertEqual(len(adv), 1 if p.name == "mini_small.json" else 0, (p.name, adv))

    def test_advisory_is_not_an_error(self):
        """--check prints the warning and still exits 0 (it is not a schema relation); the schema itself is unchanged."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "wide.json"
            f.write_text(json.dumps(variant(320, 1)))
            r = subprocess.run([sys.executable, str(SPEC / "tools/schema_invariants.py"), "--check", str(f)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("advisory", r.stdout + r.stderr)
        self.assertEqual(si.check(variant(320, 1)), [])


if __name__ == "__main__":
    unittest.main()
