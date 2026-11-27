"""The one import point of the reference assembler: `opentsp-arch-spec/tools/tsp_as.py`, a normative Python file that is not a package.

The pager and the patch tool lower and assemble text with it. They depend on this single function, `load`, and not on
a `sys.path` entry of their own, so the dependency on that file is in one place and `SYMBOLS` lists exactly what the toolchain takes from it.
`test_import_surface.py` holds the list to the tools' use and to the file.

The in-process boundary supplies frequent parser and encoder calls without spawning a process for each one.
The module is loaded by path from this installation's source checkout.
"""
import importlib.util
import os
import sys
from threading import Lock

from . import profile as tprofile

# The toolchain and serving adapters share this reference assembler symbol contract.
SYMBOLS = (
    "AsmError", "BUNDLE_BIT", "CONTROL", "ISA", "NOP_MAX", "PADDING",
    "assemble", "is_nop_word", "queue_id", "queue_name", "word_units",
    "write_object", "write_weights", "lower_records", "q_lines",
    "bind_weights", "input_text", "parse_images", "preload_text", "raise_images",
    "_read_elf", "elf_identity", "object_preloads", "read_object", "read_weights",
)

_module = None
_load_lock = Lock()


def load():
    """Load the reference assembler once and validate the toolchain's required symbols."""
    global _module
    with _load_lock:
        if _module is None:
            _module = _load_reference()
        return _module


def _load_reference():
    path = os.path.join(tprofile.SPEC, "tools", "tsp_as.py")
    name = "opentsp_reference.assembler"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the reference assembler from {path}")
    module = importlib.util.module_from_spec(spec)
    had_previous = name in sys.modules
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        missing = [symbol for symbol in SYMBOLS if not hasattr(module, symbol)]
        if missing:
            raise ImportError(f"the reference assembler ({path}) lacks {', '.join(missing)}")
    except BaseException:
        if had_previous:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
        raise
    return module
