#!/usr/bin/env python3
"""Exercise real shell profile isolation, child statuses and bounded concurrency."""
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "opentsp-arch-spec/tools/model_test.sh"
STUB = '''import fcntl,json,os,pathlib,sys,time
args=sys.argv[1:];phase='check' if args[0]=='--check' else 'gen'
if phase=='check':args=args[1:]
name=pathlib.Path(args[1]).stem;out=pathlib.Path(args[2])
def event(kind):
 with open(os.environ['EVENTS'],'a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps([time.monotonic(),kind,phase,name])+'\\n');f.flush()
event('start');time.sleep(.08)
if phase=='gen':out.write_text(json.dumps({'profile':name}))
else:assert json.loads(out.read_text())=={'profile':name}
for i in range(5):print(f'{phase}:{name}:{i}')
fail=os.environ.get('FAIL_MODE')==phase and os.environ.get('FAIL_PROFILE')==name
event('end')
if fail:print(f'controlled {phase} failure {name}',file=sys.stderr);sys.exit(7)
'''


def function():
    text = SOURCE.read_text()
    return text[text.index("vectors_variants() {"):text.index('\nif [ "${1:-}"')]


def run(jobs, failure="", mutate=""):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "tools").mkdir()
        (root / "tools/gen_model_vectors.py").write_text(STUB)
        code = function()
        if mutate == "wait":
            code = code.replace("|| rc=1", "|| true")
        if mutate == "pipe":
            code = code.replace("set -euo pipefail", "set -eu; set +o pipefail")
        env = {**os.environ, "MODEL_VECTOR_JOBS": str(jobs), "EVENTS": str(root / "events"),
               "FAIL_MODE": failure, "FAIL_PROFILE": "b"}
        script = ('set -euo pipefail\nhere="$1";scratch="$1";py="$2";sim=unused;'
                  'VARIANTS="a b c d"\n' + code + "\nvectors_variants\n")
        result = subprocess.run(["bash", "-c", script, "vectors-test", tmp, sys.executable],
                                env=env, capture_output=True, text=True, timeout=20)
        events = [json.loads(line) for line in (root / "events").read_text().splitlines()] if (root / "events").exists() else []
        return result, events


class ProfileParallel(unittest.TestCase):
    def test_serial_and_parallel_preserve_every_call_and_order(self):
        expected = [f"{phase}:{name}:{i}" for name in "abcd" for phase, indices in
                    [("gen", [3, 4]), ("check", [2, 3, 4])] for i in indices]
        for jobs in ((1, 2, 16) if sys.platform.startswith("linux") else (1,)):
            result, events = run(jobs)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), expected)
            active = peak = 0
            for _, kind, _, _ in sorted(events):
                active += 1 if kind == "start" else -1
                peak = max(peak, active)
            self.assertEqual(active, 0)
            self.assertLessEqual(peak, jobs)
            if jobs > 1:
                self.assertGreaterEqual(peak, 2)
            self.assertEqual(sorted((phase, name) for _, kind, phase, name in events if kind == "end"),
                             sorted((phase, name) for phase in ("gen", "check") for name in "abcd"))

    @unittest.skipUnless(sys.platform.startswith("linux"), "Optional parallel runner requires Linux setsid; serial remains covered")
    def test_generation_failure_is_retained_even_after_writing_output(self):
        result, events = run(2, "gen")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("controlled gen failure b", result.stdout)
        self.assertEqual(sum(kind == "end" and phase == "gen" for _, kind, phase, _ in events), 4)
        self.assertEqual(sum(kind == "end" and phase == "check" for _, kind, phase, _ in events), 3)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Optional parallel runner requires Linux setsid; serial remains covered")
    def test_check_failure_is_retained_and_other_profiles_finish(self):
        result, events = run(2, "check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("controlled check failure b", result.stdout)
        self.assertEqual(sum(kind == "end" for _, kind, _, _ in events), 8)

    def test_invalid_worker_count_is_rejected_before_any_child(self):
        for jobs in (0, 33, "01", "word"):
            result, events = run(jobs)
            self.assertEqual(result.returncode, 2)
            self.assertIn("MODEL_VECTOR_JOBS must be 1..32", result.stderr)
            self.assertEqual(events, [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Optional parallel runner requires Linux setsid; serial remains covered")
    def test_interrupt_stops_descendants_before_scratch_cleanup(self):
        # The generator forks a simulator-like child. Signal only the shell PID,
        # so success depends on the gate's own process-group cleanup.
        for sig in (signal.SIGTERM, signal.SIGINT):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "tools").mkdir()
                stub = ("import json,os,pathlib,subprocess,sys,time\n"
                        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
                        "pathlib.Path(os.environ['PIDS']).write_text(json.dumps([os.getpid(),child.pid,os.getpgrp()]))\n"
                        "time.sleep(30)\n")
                (root / "tools/gen_model_vectors.py").write_text(stub)
                env = {**os.environ, "MODEL_VECTOR_JOBS": "2", "PIDS": str(root / "pids")}
                script = ('set -euo pipefail\nhere="$1";scratch="$1";py="$2";sim=unused;VARIANTS="a"\n'
                          + function() + "\ntrap 'touch \"$scratch/cleaned\";rm -rf \"$scratch/vectors\"' EXIT\nvectors_variants\n")
                proc = subprocess.Popen(["bash", "-c", script, "interrupt-test", tmp, sys.executable],
                                        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, start_new_session=True)
                pids = []
                try:
                    deadline = time.monotonic() + 5
                    while not (root / "pids").exists() and time.monotonic() < deadline:
                        time.sleep(.02)
                    pids = json.loads((root / "pids").read_text())
                    proc.send_signal(sig)
                    proc.communicate(timeout=5)
                    self.assertEqual(proc.returncode, 128 + sig)
                    self.assertTrue((root / "cleaned").exists())
                    for pid in pids[:2]:
                        stat = Path(f"/proc/{pid}/stat")
                        self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z",
                                        f"Owned descendant {pid} survived cleanup")
                    # The owned session leader has also been reaped by wait.
                    self.assertFalse(Path(f"/proc/{pids[2]}/stat").exists())
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    if pids:
                        try:
                            os.killpg(pids[2], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    proc.communicate(timeout=5)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Optional parallel runner requires Linux setsid; serial remains covered")
    def test_wait_failure_and_pipeline_status_mutants_are_rejected(self):
        for mutation in ("wait", "pipe"):
            result, _ = run(2, "gen", mutation)
            self.assertEqual(result.returncode, 0, "The causal mutant should expose a false green")
            # The acceptance rule rejects this mutant: a controlled child failure must fail the gate.
            self.assertFalse(result.returncode != 0)


if __name__ == "__main__":
    unittest.main()
