"""Generator help and invalid options never modify generated artifacts."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC = Path(__file__).resolve().parents[2]


class GeneratorCommandLines(unittest.TestCase):
    def test_isa_help_and_unknown_flag_leave_outputs_unchanged(self):
        files = [SPEC / "docs/isa.md", SPEC / "docs/encoding.md", SPEC / "isa-db/generated/encoding.json"]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        script = SPEC / "tools/gen_isa_doc.py"
        with tempfile.TemporaryDirectory() as tmp:
            for args, status in [(["--help"], 0), (["--check", "--unknown-option"], 2)]:
                result = subprocess.run([sys.executable, str(script), *args], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, status, result.stderr)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in files])

    def test_catalogue_help_and_unknown_flag_create_nothing(self):
        script = SPEC / "tools/catalogue/gen_catalogue.py"
        with tempfile.TemporaryDirectory() as tmp:
            for args, status in [(["--help"], 0), (["--unknown-option"], 2)]:
                result = subprocess.run([sys.executable, str(script), *args], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
