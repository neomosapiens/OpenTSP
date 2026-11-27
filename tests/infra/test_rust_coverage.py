"""Exercise the Rust coverage recipes with a deterministic llvm-cov summary."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER = ("ots-profile", "ots-isa", "ots-format", "ots-asm")
ISS = ("tsp-numerics", "tsp-iss")


class CoverageRecipes(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.log = self.root / "cargo.jsonl"
        rustup = self.root / "rustup"
        rustup.write_text("#!/bin/sh\nexit 0\n")
        cargo = self.root / "cargo"
        cargo.write_text(f"#!{sys.executable}\n" + """\
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["CARGO_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if "--output-path" in args and os.environ["COVERAGE_FIXTURE"]:
    pathlib.Path(args[args.index("--output-path") + 1]).write_text(os.environ["COVERAGE_FIXTURE"])
""")
        for program in (rustup, cargo):
            program.chmod(0o755)

    def invoke(self, component, crates, floor, covered=90, write_report=True):
        counter = {"count": 100, "covered": covered}
        files = [
            {"filename": f"/workspace/crates/{name}/src/lib.rs",
             "summary": {"lines": counter, "branches": counter}}
            for name in crates
        ]
        env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                   PYTHON=sys.executable, COVERAGE_MIN=floor, COVERAGE_TOOLCHAIN="test-nightly",
                   COVERAGE_JSON=str(self.root / "report.json"), CARGO_LOG=str(self.log),
                   COVERAGE_FIXTURE=json.dumps({"data": [{"files": files}]}) if write_report else "")
        return subprocess.run(
            ["bash", str(ROOT / "tools" / f"coverage_{component}.sh")],
            env=env, capture_output=True, text=True, check=False,
        )

    def test_non_finite_and_out_of_range_floors_are_rejected_before_launch(self):
        for invalid in ("nan", "inf", "-1", "101"):
            with self.subTest(floor=invalid):
                result = self.invoke("asm", ASSEMBLER, invalid, covered=0)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("TOTAL", result.stdout)
                self.assertFalse(self.log.exists())

    def test_both_recipes_preserve_scope_threshold_and_report_path(self):
        for component, crates in (("asm", ASSEMBLER), ("iss", ISS)):
            with self.subTest(component=component):
                result = self.invoke(component, crates, "90")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("TOTAL", result.stdout)
                self.assertTrue((self.root / "report.json").is_file())
                commands = [json.loads(line) for line in self.log.read_text().splitlines()]
                self.assertEqual(commands[-2], ["+test-nightly", "llvm-cov", "clean", "--workspace"])
                args = commands[-1]
                self.assertEqual(tuple(args[i + 1] for i, arg in enumerate(args) if arg == "-p"), crates)
                self.assertIn("--branch", args)

    def test_missing_new_report_cannot_reuse_published_report(self):
        published = self.root / "report.json"
        self.assertEqual(self.invoke("iss", ISS, "90").returncode, 0)
        previous = published.read_bytes()
        result = self.invoke("iss", ISS, "90", write_report=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid coverage report", result.stderr)
        self.assertEqual(published.read_bytes(), previous)

    def test_missing_or_below_threshold_crate_is_reported(self):
        result = self.invoke("iss", ISS[:1], "90", covered=89)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("coverage-iss: below 90.0%", result.stderr)
        self.assertIn("tsp-iss", result.stderr)


if __name__ == "__main__":
    unittest.main()
