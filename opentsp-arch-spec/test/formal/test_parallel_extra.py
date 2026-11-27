#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import parallel_extra as p


class PortableGuard(unittest.TestCase):
    def test_non_linux_rejected_before_launch(self):
        with mock.patch.object(p.sys, 'platform', 'darwin'), mock.patch.object(p.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'only on Linux'):
                p.run(Path('unused'), Path('.'), 2, ['domains'])
            popen.assert_not_called()


@unittest.skipUnless(sys.platform == 'linux', 'parallel proof groups require Linux subreaper support')
class ParallelExtra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.props = self.root / 'test/formal/props'
        self.props.mkdir(parents=True)
        for suite in p.SUITES:
            for suffix in ('.sail', '.txt'):
                (self.props / (suite + suffix)).write_text('stub input\n')
        self.script = self.root / 'child.sh'
        self.script.write_text('test "$PROVE_EXTRA_JOBS" = 1\nsleep .05\n')
        self.env = mock.patch.dict(os.environ, {'PROVE_KEEP': str(self.root / 'keep'), 'PROVE_FAIL_FAST': '0'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_all_suites_once_and_isolated(self):
        self.assertEqual(p.run(self.script, self.root, 8, list(p.SUITES)), 0)
        results = list((self.root / 'keep').glob('*/results.json'))
        self.assertEqual(len(results), 1)
        receipt = json.loads(results[0].read_text())
        self.assertEqual(set(receipt['results']), set(p.SUITES))
        self.assertTrue(all(code == 0 for code in receipt['results'].values()))
        self.assertEqual(len(list(results[0].parent.glob('*/'))), 8)

    def test_bad_inventory_fails_before_launch(self):
        for suites in (['domains', 'domains'], ['not-a-suite']):
            with mock.patch.object(p.subprocess, 'Popen') as popen:
                with self.assertRaises(ValueError):
                    p.run(self.script, self.root, 8, suites)
                popen.assert_not_called()
        (self.props / 'domains.txt').unlink()
        with self.assertRaises(ValueError):
            p.run(self.script, self.root, 8, ['domains'])

    def test_failure_is_not_pass(self):
        self.script.write_text('if [ "$1" = domains ]; then exit 7; fi\n')
        self.assertEqual(p.run(self.script, self.root, 2, ['domains', 'ties']), 1)

    def test_launch_failure_cleans_prior_owned_group(self):
        self.script.write_text('sleep 60\n')
        real = subprocess.Popen
        launched = []

        def popen(*args, **kwargs):
            if launched:
                raise OSError('injected launch failure')
            process = real(*args, **kwargs)
            launched.append(process)
            return process

        with mock.patch.object(p.subprocess, 'Popen', side_effect=popen):
            with self.assertRaises(OSError):
                p.run(self.script, self.root, 2, ['domains', 'ties'])
        with self.assertRaises(ProcessLookupError):
            os.killpg(launched[0].pid, 0)

    def test_fail_fast_reaps_term_ignoring_grandchild_and_preserves_foreign(self):
        if os.name != 'posix':
            self.skipTest('process groups require POSIX')
        self.script.write_text('''if [ "$1" = domains ]; then sleep .3; exit 7; fi
python3 - "$PROVE_KEEP/grandchild" <<'PY'
import os,signal,time,sys
child=os.fork()
if child==0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 while True: time.sleep(1)
open(sys.argv[1],'w').write(str(child))
while True: time.sleep(1)
PY
''')
        foreign = subprocess.Popen(['sleep', '60'], start_new_session=True)
        try:
            with mock.patch.dict(os.environ, {'PROVE_FAIL_FAST': '1'}):
                self.assertEqual(p.run(self.script, self.root, 2, ['domains', 'ties']), 1)
            grandchild = int(next((self.root / 'keep').glob('*/ties/grandchild')).read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(grandchild, 0)
            self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait()

    def test_external_interrupt_reaps_owned_grandchild(self):
        self.script.write_text('''python3 - "$PROVE_KEEP/grandchild" <<'PY'
import os,signal,time,sys
child=os.fork()
if child==0:
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 while True: time.sleep(1)
open(sys.argv[1],'w').write(str(child))
while True: time.sleep(1)
PY
''')
        process = subprocess.Popen([sys.executable, p.__file__, str(self.script), str(self.root), '2', 'ties'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   start_new_session=True)
        try:
            until = time.monotonic() + 5
            records = []
            while not records and time.monotonic() < until:
                records = list((self.root / 'keep').glob('*/ties/grandchild'))
                time.sleep(.02)
            self.assertTrue(records)
            child = int(records[0].read_text())
            suite_group = os.getpgid(child)
            os.killpg(process.pid, signal.SIGTERM)
            # The real outer mutation runner sends KILL after five seconds.
            self.assertEqual(process.wait(timeout=5), 143)
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
            with self.assertRaises(ProcessLookupError):
                os.killpg(suite_group, 0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


if __name__ == '__main__':
    unittest.main()
