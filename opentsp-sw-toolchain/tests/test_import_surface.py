"""The toolchain's dependency on the reference assembler (`opentsp-arch-spec/tools/tsp_as.py`) is one function in one module.

Run: python3 opentsp-sw-toolchain/tests/test_import_surface.py
  1. `refasm.load()` returns the module and every symbol of `refasm.SYMBOLS` exists in it.
  2. Every `tsp_as.<name>` that a toolchain or model adapter uses is in `SYMBOLS` (no symbol is taken without being listed).
  3. No module but `refasm` puts a path on `sys.path` or imports `tsp_as` itself.
  4. `refasm.load()` reports every missing symbol when the file lacks some (a stand-in module)."""
import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))

from opentsp_tools import refasm  # noqa: E402


def modules():
    root = Path(HERE).parents[1]
    packages = ("opentsp-sw-toolchain/python/opentsp_tools", "opentsp-vllm/src/opentsp_vllm")
    for package in packages:
        for path in sorted((root / package).rglob("*.py")):
            yield path.relative_to(root), ast.parse(path.read_text(), filename=str(path))


def used(tree):
    return {n.attr for n in ast.walk(tree)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "tsp_as"}


def test_symbols_exist():
    mod = refasm.load()
    assert mod is refasm.load(), "load() imports once"
    missing = [s for s in refasm.SYMBOLS if not hasattr(mod, s)]
    assert not missing, "tsp_as lacks %s" % missing


def test_every_use_is_listed():
    seen = set()
    for name, tree in modules():
        seen |= used(tree)
    unlisted = sorted(seen - set(refasm.SYMBOLS))
    assert not unlisted, "tsp_as symbols used but not in refasm.SYMBOLS: %s" % unlisted
    # Model adapters also use the explicitly declared serialization/input symbols.
    assert {"assemble", "word_units", "queue_id"} <= seen


def test_one_import_point():
    for name, tree in modules():
        if name.name == "refasm.py":
            continue
        for n in ast.walk(tree):
            direct_import = isinstance(n, ast.Import) and any(alias.name == "tsp_as" for alias in n.names)
            direct_from = isinstance(n, ast.ImportFrom) and n.module == "tsp_as"
            if direct_import or direct_from:
                raise AssertionError(f"{name} imports the reference assembler directly")
            if (isinstance(n, ast.Attribute) and n.attr == "path" and isinstance(n.value, ast.Name)
                    and n.value.id == "sys"):
                raise AssertionError("%s changes sys.path" % name)


def test_missing_symbols_are_reported():
    saved_mod, saved_symbols = refasm.load(), refasm.SYMBOLS
    refasm._module = None
    refasm.SYMBOLS = saved_symbols + ("no_such_symbol", "also_missing")
    try:
        refasm.load()
    except ImportError as e:
        assert "no_such_symbol" in str(e) and "also_missing" in str(e), str(e)
        assert sys.modules["opentsp_reference.assembler"] is saved_mod
    else:
        raise AssertionError("a missing symbol was not reported")
    finally:
        refasm._module, refasm.SYMBOLS = saved_mod, saved_symbols


def test_concurrent_callers_receive_the_same_module():
    saved_module = refasm.load()
    first_loading = Event()
    second_calling = Event()
    duplicate_load = Event()
    release = Event()
    original_spec = refasm.importlib.util.spec_from_file_location

    def controlled_spec(*args):
        if first_loading.is_set():
            duplicate_load.set()
        else:
            first_loading.set()
            if not release.wait(5):
                raise RuntimeError("test did not release the first load")
        return original_spec(*args)

    def second():
        second_calling.set()
        return refasm.load()

    name = saved_module.__name__
    with patch.object(refasm, "_module", None), patch.object(refasm.importlib.util, "spec_from_file_location", controlled_spec):
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first_result = pool.submit(refasm.load)
                try:
                    assert first_loading.wait(5)
                    second_result = pool.submit(second)
                    assert second_calling.wait(5)
                    raced = duplicate_load.wait(0.25)
                finally:
                    release.set()
                assert not raced, "two callers started loading the same reference module"
                assert first_result.result() is second_result.result()
        finally:
            sys.modules[name] = saved_module


def test_load_leaves_sys_path_alone():
    """Importing the reference assembler does not put the specification's tools directory on `sys.path` (a path entry there would
    let any later `import` of that directory's other files, or a file of the same name, succeed by accident)."""
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); before = list(sys.path); from opentsp_tools import refasm; refasm.load(); "
            "assert sys.path == before, [p for p in sys.path if p not in before]; print('unchanged')" % os.path.dirname(HERE))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "unchanged" in r.stdout, "refasm.load() changed sys.path: " + (r.stderr.strip().splitlines() or [""])[-1]



def test_loaded_assembler_can_validate_and_assemble_without_a_tools_path():
    import subprocess
    code = ("import sys; sys.path.insert(0, %r); before = list(sys.path); "
            "from opentsp_tools import refasm; a = refasm.load(); "
            "a.assemble('.arch opentsp-tsa\\n.profile mini\\n.section text\\n'); assert sys.path == before; print('assembled')" % os.path.dirname(HERE))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0 and "assembled" in result.stdout, result.stderr


def test_the_reference_assembler_dependency_is_documented():
    """The source-checkout prerequisite and in-process calls must be visible to library users."""
    readme = open(os.path.join(os.path.dirname(HERE), "tools", "README.md")).read()
    for term in ("tools/tsp_as.py", "source checkout", "in process", "assemble", "word_units", "queue_id"):
        assert term in readme, f"Python toolchain README is missing {term!r}"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, v in tests:
        v()
        print("ok:", k)
    print("ok: %d import-surface tests" % len(tests))
