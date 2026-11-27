#!/usr/bin/env python3
"""The program runner's case selection: an explicit selection that matches no case fails, and the runner reports how many cases it resolved.

usage: python3 tests/programs/gen/test_run_selection.py
"""
import importlib.util
import os
import subprocess
import sys
import unittest

RUN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.py")
STUB = "/bin/echo"  # the selection is resolved before any tool runs

spec = importlib.util.spec_from_file_location("program_test_runner", RUN)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def run(*args):
    return subprocess.run([sys.executable, RUN, "--sim", STUB, "--asm", STUB, "--iss", STUB, *args],
                          capture_output=True, text=True, timeout=60)


class Selection(unittest.TestCase):
    def test_an_unknown_tag_fails(self):
        r = run("--tag", "not-a-real-tag")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("matched no case", r.stderr)
        self.assertIn("--tag not-a-real-tag", r.stderr)

    def test_an_unknown_profile_fails(self):
        r = run("--profile", "no-such-profile")
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("matched no case", r.stderr)

    def test_an_explicit_missing_iss_is_an_error(self):
        r = run("--iss", "/nonexistent-opentsp-iss", "basic/f16_chain.mini")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("requested ISS is not executable", r.stderr)
        self.assertNotIn("cases selected", r.stdout)

    def test_the_resolved_case_count_is_reported_before_the_run(self):
        r = run("basic/f16_chain.mini")
        self.assertIn("run: 1 cases selected", r.stdout)

class HarnessOutput(unittest.TestCase):
    case = {
        "profile": "mini",
        "dump": ["2:3", "2:4"],
        "expect": {"diagnostics": [], "words": {}},
    }

    def compare(self, first, second):
        parsed = runner.parse_out(f"E - -1 0\n{first}\n{second}\n")
        runner.Runner(None).compare(self.case, parsed)

    def test_dump_shape_matches_the_request_and_the_profile(self):
        first = "W 2 3" + " 00" * 320
        second = "W 2 4" + " 00" * 320
        self.compare(first, second)
        with self.assertRaisesRegex(runner.Fail, "dump locations"):
            self.compare(second, first)
        with self.assertRaisesRegex(runner.Fail, "dump locations"):
            self.compare(first, first)
        with self.assertRaisesRegex(runner.Fail, "expected 320 bytes"):
            self.compare(first[:-3], second)
        with self.assertRaisesRegex(runner.Fail, "bytes beyond vlanes"):
            self.compare(first[:-3] + " 01", second)


if __name__ == "__main__":
    unittest.main()
