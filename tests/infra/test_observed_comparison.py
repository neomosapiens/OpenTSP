"""Observed conformance cannot pass through matching setup failures or uncovered timeouts."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import iss_compare_sites as comparison

VECTOR = dict(name='fixture', sram={}, text={}, dump=[])


class ObservedComparison(unittest.TestCase):
    def test_successful_matching_runs_pass(self):
        with mock.patch.object(comparison, 'run', return_value=(0, b'E - -1 0\n', b'')):
            self.assertTrue(comparison.compare('mini', VECTOR, ('model', 'iss')))

    def test_matching_setup_errors_do_not_pass(self):
        with mock.patch.object(comparison, 'run', return_value=(2, b'', b'invalid configuration')):
            with self.assertRaises(ValueError):
                comparison.compare('mini', VECTOR, ('model', 'iss'))

    def test_one_timeout_among_two_hundred_cases_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = Path(temporary)
            model = spec / 'conformance/model'
            model.mkdir(parents=True)
            for profile in ('mini', 'ref'):
                (model / (profile + '.json')).write_text(json.dumps({'vectors': [VECTOR] * 200}))
            with mock.patch.object(comparison, 'SPEC', spec), mock.patch.object(comparison, 'compare', side_effect=[False] + [True] * 399), mock.patch.object(sys, 'argv', ['iss_compare_sites.py', '--jobs', '1']), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    comparison.main()


if __name__ == '__main__':
    unittest.main()
