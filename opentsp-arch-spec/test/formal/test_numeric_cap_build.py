#!/usr/bin/env python3
"""Exercise model-path handling before generation with a deliberate fake tool.
No result in this unit test is native model or certificate evidence.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

SPEC = Path(__file__).resolve().parents[2]


class ModelPathHandling(unittest.TestCase):
    def check_model_path(self, absolute):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "input/model"
            model.mkdir(parents=True)
            (model / "tsp.sail_project").write_text("\n".join(
                name + " {\n  files unit.sail\n}\n" for name in ("prelude", "core", "numerics", "main")))
            (model / "unit.sail").write_text("// unit control only\n")
            bindir = root / "bin"
            bindir.mkdir()
            (root / "share/sail/lib").mkdir(parents=True)
            sail = bindir / "sail"
            sail.write_text("#!/bin/sh\nif test -f model/tsp.sail_project; then echo MODEL_RESOLVED; else echo MODEL_BROKEN; fi\nexit 17\n")
            sail.chmod(0o755)
            work = root / "output"
            selected = str(model) if absolute else "input/model"
            env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"])
            result = subprocess.run(["bash", str(SPEC / "test/formal/build_numeric_cap.sh"), str(work), selected],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 17, result.stderr)
            self.assertEqual((work / "sail.log").read_text().strip(), "MODEL_RESOLVED")
            self.assertEqual((work / "model").resolve(), model.resolve())

    def test_relative_model_from_different_work_directory(self):
        self.check_model_path(False)

    def test_absolute_model_from_different_work_directory(self):
        self.check_model_path(True)


if __name__ == "__main__":
    unittest.main()
