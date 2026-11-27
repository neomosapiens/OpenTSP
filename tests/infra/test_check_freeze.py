"""The public specification baseline fails closed on changed or uncovered files."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / 'tools/check_freeze.py'
FILES = ['model/core/example.sail', 'c_emulator/example.c', 'config/ref.json', 'config/mini.json', 'config/schema.json', 'isa-db/instructions/example.yaml', 'isa-db/generated/encoding.json', 'conformance/model/ref.json', 'tools/tsp_as.py', 'docs/isa.md', 'docs/encoding.md', 'reference/opentsp_reference/schema.py', 'pyproject.toml', 'tools/schema_invariants.py']


class SpecificationBaseline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = self.root / 'opentsp-arch-spec'
        self.rows = []
        for name in FILES:
            path = self.spec / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('baseline\n')
            self.rows.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  opentsp-arch-spec/' + name)
        self.manifest = self.spec / 'SHA256SUMS'
        self.manifest.write_text('\n'.join(self.rows) + '\n')

    def run_check(self):
        return subprocess.run([sys.executable, str(CHECK), '--root', str(self.root)], capture_output=True, text=True)

    def test_clean_directory_without_git(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generated_compiler_and_editable_install_files_do_not_enter_the_baseline(self):
        (self.spec / 'model/sail_smt_cache').touch()
        metadata = self.spec / 'reference/opentsp_reference.egg-info'
        metadata.mkdir()
        (metadata / 'SOURCES.txt').touch()
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_file(self):
        (self.spec / FILES[0]).write_text('different\n')
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_missing_manifest_entry(self):
        self.manifest.write_text('\n'.join(self.rows[1:]) + '\n')
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_new_normative_file(self):
        (self.spec / 'model/core/new.sail').write_text('new\n')
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_missing_file(self):
        (self.spec / FILES[0]).unlink()
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_duplicate_manifest_entry(self):
        self.manifest.write_text('\n'.join(self.rows + self.rows[:1]) + '\n')
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_path_escape(self):
        self.manifest.write_text(self.manifest.read_text() + '0' * 64 + '  ../outside\n')
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_no_update_route(self):
        result = subprocess.run([sys.executable, str(CHECK), '--update'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
