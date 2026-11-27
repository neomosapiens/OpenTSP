#!/usr/bin/env python3
"""List source-directory files without requiring repository metadata."""
import argparse
import os
from pathlib import Path
import sys

BUILD_DIRECTORIES = {'.git', 'target', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache', '.mypy_cache'}


GENERATED_FILES = {'.DS_Store', '.agent-owner', 'sail_smt_cache', '.props.sail_project', '.enc.sail_project'}


def source_paths(root):
    """Yield maintained files, excluding tool output and editable-install metadata."""
    for directory, children, names in os.walk(root):
        children[:] = [name for name in children if name not in BUILD_DIRECTORIES and not name.endswith('.egg-info')]
        for name in names:
            if name in GENERATED_FILES or name.startswith('._') or name.endswith('.pyc'):
                continue
            path = Path(directory) / name
            if path.is_file():
                yield path


def inventory(root):
    root = Path(root)
    return sorted(path.relative_to(root).as_posix() for path in source_paths(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--suffix', action='append', default=[])
    parser.add_argument('--null', action='store_true')
    args = parser.parse_args()
    paths = [p for p in inventory(args.root) if not args.suffix or Path(p).suffix in args.suffix]
    separator = '\0' if args.null else '\n'
    if paths:
        sys.stdout.write(separator.join(paths) + separator)


if __name__ == '__main__':
    main()
