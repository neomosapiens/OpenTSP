"""Installed Python package boundaries and source-cache preservation."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

def child(code, **changes):
    env = dict(os.environ, **changes)
    with tempfile.TemporaryDirectory() as directory:
        return subprocess.run([sys.executable, "-c", code], cwd=directory, env=env, capture_output=True, text=True)

class PythonPackageBoundaries(unittest.TestCase):
    def test_production_does_not_mutate_global_search_paths_or_import_tests(self):
        for directory in ("opentsp-sw-toolchain/python", "opentsp-vllm/src"):
            paths=list((ROOT / directory).rglob("*.py"))
            self.assertGreater(len(paths),10,directory)
            for path in paths:
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.Call):
                        self.assertNotIn(ast.unparse(node.func), ("sys.path.insert", "sys.path.append", "sys.path.extend"), str(path))
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith(("tests.",)), (path,node.lineno))
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            self.assertFalse(name.name.startswith(("tests.",)), (path,node.lineno))

    def test_models_ignore_foreign_flat_modules_and_import_in_both_orders(self):
        modules = ("opentsp_vllm.models.tinystories.fixed_package", "opentsp_vllm.models.gpt2.pipeline")
        for order in (modules, modules[::-1]):
            code = f"""import sys,types,importlib
for name in ("config","weights","export","layout","run","basic","tsp_as","schema_invariants"):
    sys.modules[name]=types.ModuleType(name)
before=list(sys.path)
for name in {order!r}: importlib.import_module(name)
assert sys.path==before
from opentsp_vllm.models.tinystories import config as tiny
from opentsp_vllm.models.gpt2 import config as gpt
assert tiny.TINYSTORIES.hidden==64 and gpt.GPT2_SMALL.hidden==768
assert tiny is not gpt
assert "vllm" not in sys.modules and "torch" not in sys.modules
from opentsp_tools import refasm
a=refasm.load()
a.assemble(".arch opentsp-tsa\\n.profile mini\\n.section text\\n")
assert sys.modules["schema_invariants"].__dict__.keys()=={{"__name__","__doc__","__package__","__loader__","__spec__"}}
"""
            result=child(code)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_generic_layer_has_no_model_dependency(self):
        result=child("""import sys,importlib.abc
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.startswith(("opentsp_vllm","tests")):raise RuntimeError(fullname)
sys.meta_path.insert(0,NoModels())
from opentsp_tools import pager,profile
from opentsp_tools.programs import kernels,numerics
assert profile.load("mini").vlanes==16
assert numerics.oracle().fp_add(numerics.oracle().F32,0x3f800000,0x3f800000)==0x40000000
""")
        self.assertEqual(result.returncode,0,result.stderr)

    def test_canonical_cache_inputs_change_both_model_keys(self):
        import shutil
        from opentsp_vllm.models.tinystories import program_cache as tiny
        from opentsp_vllm.models.gpt2 import pipeline as gpt
        required = (
            "opentsp-sw-toolchain/python/opentsp_tools/programs/data.py",
            "opentsp-sw-toolchain/python/opentsp_tools/pager.py",
            "opentsp-arch-spec/reference/opentsp_reference/numerics.py",
            "opentsp-arch-spec/reference/opentsp_reference/schema.py",
            "opentsp-arch-spec/tools/tsp_as.py",
            "opentsp-arch-spec/config/ref.json",
            "opentsp-arch-spec/isa-db/generated/encoding.json",
        )
        inventories = (set(str(Path(x).relative_to(ROOT)) for x in tiny.generator_files()),
                       set(gpt.digest_files(ROOT)))
        for files in inventories:
            self.assertTrue(set(required) <= files)
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory)
            for name in set.union(*inventories):
                dst = clone / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dst)
            class Weights:
                def digest(self): return "fixed-preservation-weights"
            def keys():
                return (tiny.generator_digest(tiny.generator_files(clone)),
                        gpt.source_digest(Weights(), 2, [0, 1], 2, root=clone))
            baseline = keys()
            for name in required:
                target = clone / name
                saved = target.read_bytes()
                target.write_bytes(saved + b"\n")
                changed = keys()
                self.assertNotEqual(changed[0], baseline[0], name)
                self.assertNotEqual(changed[1], baseline[1], name)
                target.write_bytes(saved)
                self.assertEqual(keys(), baseline)

    def test_shared_contract_identity_survives_pickling(self):
        import pickle
        from opentsp_vllm import chip, contracts, ras
        self.assertIs(chip.ChipInfo, contracts.ChipInfo)
        self.assertIs(chip.TspChipError, contracts.TspChipError)
        self.assertIs(ras.DomainFault, contracts.DomainFault)
        value = chip.ChipInfo(2, 97, 64, 16, 4, (64,))
        self.assertEqual(pickle.loads(pickle.dumps(value)), value)

    def test_missing_serving_extra_has_an_actionable_error(self):
        result = child("""import tempfile,pathlib
from unittest.mock import patch
from opentsp_vllm import serve
with tempfile.TemporaryDirectory() as d:
    pathlib.Path(d,"weights.safetensors").touch()
    pathlib.Path(d,"model").mkdir()
    pathlib.Path(d,"model/config.json").touch()
    with patch("opentsp_vllm.serve.importlib.util.find_spec",return_value=None):
        raise SystemExit(serve.main(["--tsp-programs",d]))
""")
        self.assertEqual(result.returncode,2,result.stderr)
        self.assertIn("run just vllm-env",result.stderr)
        self.assertNotIn("Traceback",result.stderr)

    def test_invalid_explicit_root_fails_without_fallback(self):
        result=child("from opentsp_tools import paths", OPENTSP_ROOT="/not-an-opentsp-checkout")
        self.assertNotEqual(result.returncode,0)
        self.assertIn("OPENTSP_ROOT",result.stderr)
        self.assertIn("not an OpenTSP source checkout",result.stderr)

if __name__=="__main__":
    unittest.main()
