"""Simulator cache identity follows contents and preserves a good binary on failure."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class SimulatorFreshness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'tools').mkdir()
        for name in ('ensure_sim.sh', 'ensure_sim.py', 'source_files.py'):
            source = ROOT / 'tools' / name
            if source.exists():
                shutil.copyfile(source, self.root / 'tools' / name)
        for directory in ('model', 'c_emulator', 'config', 'tools'):
            (self.root / 'opentsp-arch-spec' / directory).mkdir(parents=True, exist_ok=True)
        self.input = self.root / 'opentsp-arch-spec/model/model.sail'
        self.input.write_text('first\n')
        (self.root / 'opentsp-arch-spec/config/schema.json').write_text('{}\n')
        (self.root / 'opentsp-arch-spec/tools/schema_header.py').write_text('# fixture\n')
        self.builder = self.root / 'opentsp-arch-spec/tools/build_sim.sh'
        self.builder.write_text('''#!/bin/sh
set -eu
here="$(dirname "$0")/.."
echo build >> "$here/builds"
[ ! -f "$here/fail" ] || exit 7
printf '#!/bin/sh\\n# ' > "$1"
cat "$here/model/model.sail" >> "$1"
chmod +x "$1"
''')
        fake = self.root / 'bin'
        fake.mkdir()
        for name in ('sail', 'cc'):
            path = fake / name
            path.write_text('#!/bin/sh\necho fixture-tool-1\n')
            path.chmod(0o755)
        self.runtime = self.root / 'share/sail/lib/rts.c'
        self.runtime.parent.mkdir(parents=True)
        self.runtime.write_text('first runtime\n')
        self.env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'])
        self.binary = self.root / 'output/tsp_sim'

    def run_check(self):
        return subprocess.run(['bash', str(self.root / 'tools/ensure_sim.sh'), str(self.binary)], env=self.env, capture_output=True, text=True)

    def build_count(self):
        return len((self.root / 'opentsp-arch-spec/builds').read_text().splitlines())

    def test_unchanged_content_reuses_binary(self):
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 1)

    def test_compiler_cache_created_during_build_is_not_a_source_change(self):
        with self.builder.open('a') as out:
            out.write('echo cache >> "$here/model/sail_smt_cache"\n')
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 1)

    def test_changed_content_with_preserved_timestamp_rebuilds(self):
        self.assertEqual(self.run_check().returncode, 0)
        old = self.input.stat()
        self.input.write_text('second\n')
        os.utime(self.input, ns=(old.st_atime_ns, old.st_mtime_ns))
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 2)
        self.assertIn('second', self.binary.read_text())

    def test_changed_builder_rebuilds(self):
        self.assertEqual(self.run_check().returncode, 0)
        with self.builder.open('a') as out:
            out.write('# changed build flags\n')
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 2)

    def test_corrupted_binary_rebuilds(self):
        self.assertEqual(self.run_check().returncode, 0)
        self.binary.write_text('broken')
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 2)

    def test_compiled_sail_runtime_change_invalidates_the_receipt(self):
        self.assertEqual(self.run_check().returncode, 0)
        self.runtime.write_text('changed runtime\n')
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 2)

    def test_tool_changed_without_version_change_rebuilds(self):
        self.assertEqual(self.run_check().returncode, 0)
        with (self.root / 'bin/sail').open('a') as out:
            out.write('# different executable with same version\n')
        self.assertEqual(self.run_check().returncode, 0)
        self.assertEqual(self.build_count(), 2)

    def test_concurrent_callers_share_one_build(self):
        command = ['bash', str(self.root / 'tools/ensure_sim.sh'), str(self.binary)]
        first = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        second = subprocess.Popen(command, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        first.communicate(timeout=10)
        second.communicate(timeout=10)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0)
        self.assertEqual(self.build_count(), 1)

    def test_failure_preserves_existing_binary(self):
        self.assertEqual(self.run_check().returncode, 0)
        old = self.binary.read_bytes()
        self.input.write_text('second\n')
        (self.root / 'opentsp-arch-spec/fail').touch()
        self.assertNotEqual(self.run_check().returncode, 0)
        self.assertEqual(self.binary.read_bytes(), old)


if __name__ == '__main__':
    unittest.main()
