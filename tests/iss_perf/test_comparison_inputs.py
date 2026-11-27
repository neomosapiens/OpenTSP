"""Comparison inputs and per-side fixtures cannot silently change the verdict."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import iss_ab_overhead as comparison


class ComparisonInputs(unittest.TestCase):
    def test_a_missing_new_stats_file_cannot_reuse_another_runs_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / 'stats.json').write_text(json.dumps(dict(wall_seconds=0.5, last_cycle=100, ticks=50)))
            result = SimpleNamespace(returncode=0, stderr=b'')
            with mock.patch.object(comparison.subprocess, 'run', return_value=result):
                with self.assertRaises(comparison.RunFailed):
                    comparison.one('/fake', [], temporary, 'bucket', temporary)

    def test_invalid_limits_and_binary_specs_are_usage_errors(self):
        cases = [['-n', '0'], ['--gate', 'nan', '--bench', 'b.txt'], ['--gate-small', '-1'], ['--insn-gate', 'inf', '--bench', 'b.txt'], ['--dir', 'missing-equals'], ['--dir', 'unknown=/tmp']]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run([sys.executable, str(ROOT / 'tools/iss_ab_overhead.py'), *arguments, 'base=/missing-base', 'tree=/missing-tree'], capture_output=True)
                self.assertEqual(result.returncode, 2, result.stderr)
        for binaries in [['base='], ['same=/a', 'same=/b'], ['invalid']]:
            result = subprocess.run([sys.executable, str(ROOT / 'tools/iss_ab_overhead.py'), *binaries], capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_nonfinite_or_nonpositive_benchmark_time_is_rejected(self):
        for seconds in ('nan', 'inf', '0', '-1'):
            result = SimpleNamespace(returncode=0, stdout=f'bench cycles=10 seconds={seconds}', stderr='')
            with self.subTest(seconds=seconds), mock.patch.object(comparison.subprocess, 'run', return_value=result):
                with self.assertRaises(comparison.RunFailed):
                    comparison.one_bench('/fake', 'b.txt')

    def test_bucket_mode_uses_each_sides_fixture(self):
        calls = []
        def measure(binary, extra, directory, program, temporary):
            calls.append((binary, directory))
            return dict(whole=1., iss=1., cps=100., ust=1., load=0.)
        args = argparse.Namespace(bench=None, dirs={'base': '/base', 'tree': '/tree'}, d='/wrong', n=1)
        with mock.patch.object(comparison, 'one', side_effect=measure), mock.patch.object(comparison, 'bucket_work', return_value=(100, 50)), contextlib.redirect_stdout(io.StringIO()):
            comparison.run_modes(args, [('base', 'base', []), ('tree', 'tree', [])], ['bucket'])
        self.assertEqual(calls, [('base', '/base'), ('tree', '/tree')])

    def test_bucket_mode_refuses_different_work(self):
        args = argparse.Namespace(bench=None, dirs={}, d='/fixture', n=1)
        def work(binary, *_):
            return (100 if binary == 'base' else 200, 50)
        with mock.patch.object(comparison, 'bucket_work', side_effect=work), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                comparison.run_modes(args, [('base', 'base', []), ('tree', 'tree', [])], ['bucket'])
        self.assertEqual(result.exception.code, comparison.EXIT_NOT_COMPARABLE)


if __name__ == '__main__':
    unittest.main()
