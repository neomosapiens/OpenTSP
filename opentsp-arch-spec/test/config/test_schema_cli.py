"""Configuration tools reject incomplete commands and preserve files on errors."""
from pathlib import Path
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

SPEC = Path(__file__).resolve().parents[2]
SCRIPT = SPEC / "tools/schema_invariants.py"


class SchemaCommandLine(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def test_help_and_empty_check(self):
        self.assertEqual(self.invoke("--help").returncode, 0)
        self.assertEqual(self.invoke("--check").returncode, 2)
        self.assertEqual(self.invoke("--unknown-option").returncode, 2)

    def test_invalid_input_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.json", Path(tmp) / "out.json"
            source.write_text("invalid-json")
            target.write_text("preserve-this")
            result = self.invoke("--normalise", str(source), str(target))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(target.read_text(), "preserve-this")
            self.assertNotIn("Traceback", result.stderr)

    def test_check_reports_invalid_json_shapes_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bad.json"
            for value in [[], {}, {"geometry": None}]:
                source.write_text(json.dumps(value))
                result = self.invoke("--check", str(source))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_normalise_preserves_existing_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target, link = root / "in.json", root / "out.json", root / "linked.json"
            source.write_text((SPEC / "config/mini.json").read_text())
            target.write_text("replace-this")
            link.symlink_to(target.name)
            result = self.invoke("--normalise", str(source), str(link))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(link.is_symlink())
            self.assertEqual(json.loads(target.read_text())["spec_version"], "0.1")

    def test_normalise_and_validation_preserve_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.json", Path(tmp) / "out.json"
            profile = json.loads((SPEC / "config/mini.json").read_text())
            profile["ras"].pop("diag_depth", None)
            source.write_text(json.dumps(profile))
            target.write_text("replace-this")
            os.chmod(target, 0o640)
            result = self.invoke("--normalise", str(source), str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(target.read_text())["ras"]["diag_depth"], 16)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
            self.assertEqual(self.invoke("--check", str(target)).returncode, 0)


if __name__ == "__main__":
    unittest.main()
