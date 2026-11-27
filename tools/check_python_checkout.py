#!/usr/bin/env python3
"""Verify that all editable OpenTSP packages resolve to the requested checkout."""

import argparse
import importlib
import os
from pathlib import Path
import sys

SOURCES = {
    "opentsp_reference": "opentsp-arch-spec/reference",
    "opentsp_tools": "opentsp-sw-toolchain/python",
    "opentsp_vllm": "opentsp-vllm/src",
}


def package_source(root, name, directory):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    expected = root / directory / name
    if not path.is_relative_to(expected):
        raise ValueError(f"{name} is imported from {path}, expected a file under {expected}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        selected = os.environ.get("OPENTSP_ROOT")
        if selected and Path(selected).expanduser().resolve() != root:
            raise ValueError(f"OPENTSP_ROOT selects {selected!r}; unset it for checks against {root}")
        paths = {name: package_source(root, name, directory) for name, directory in SOURCES.items()}
    except (ImportError, TypeError, ValueError) as error:
        print(f"OpenTSP package check: {error}. Run 'just vllm-env' or 'just python-setup' in {root}.", file=sys.stderr)
        return 1
    print(paths["opentsp_vllm"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
