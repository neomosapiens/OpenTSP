#!/usr/bin/env python3
"""Reject a serving environment whose package roots disagree."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CHECK = Path(__file__).resolve().parents[2] / "tools/check_python_checkout.py"


class Checkout(unittest.TestCase):
    def test_package_checks_include_every_editable_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "current checkout"
            other = Path(temporary) / "different checkout"
            roots = root, other
            modules = {
                "opentsp_reference": "opentsp-arch-spec/reference",
                "opentsp_tools": "opentsp-sw-toolchain/python",
                "opentsp_vllm": "opentsp-vllm/src",
            }
            for location in roots:
                for name, directory in modules.items():
                    source = location / directory / name
                    source.mkdir(parents=True)
                    (source / "__init__.py").touch()

            def check(locations):
                paths = [str(location / directory) for location, directory in zip(locations, modules.values())]
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths))
                return subprocess.run([sys.executable, str(CHECK), str(root)], env=env, capture_output=True, text=True, check=False)

            current = check([root] * len(modules))
            self.assertEqual(current.returncode, 0, current.stderr)
            for index, name in enumerate(modules):
                with self.subTest(name=name):
                    selected = [root] * len(modules)
                    selected[index] = other
                    result = check(selected)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"{name} is imported from {other.resolve()}", result.stderr)


if __name__ == "__main__":
    unittest.main()
