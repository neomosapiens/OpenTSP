#!/usr/bin/env python3
"""Verify the complete specification 0.1 public SHA-256 baseline."""
import argparse
import hashlib
from pathlib import Path
import re
import sys

from source_files import source_paths

DIRECTORIES = ('model', 'c_emulator', 'isa-db/instructions', 'isa-db/generated', 'conformance/model', 'reference')
FILES = ('config/ref.json', 'config/mini.json', 'config/schema.json', 'tools/tsp_as.py', 'docs/isa.md', 'docs/encoding.md', 'pyproject.toml', 'tools/schema_invariants.py')


def normative_paths(root):
    spec = root / 'opentsp-arch-spec'
    actual = set()
    for name in DIRECTORIES:
        directory = spec / name
        if not directory.is_dir():
            raise ValueError('missing normative directory: ' + name)
        actual.update(path.relative_to(root).as_posix() for path in source_paths(directory))
    for name in FILES:
        path = spec / name
        if not path.is_file():
            raise ValueError('missing normative file: ' + name)
        actual.add(path.relative_to(root).as_posix())
    return actual


def check(root, manifest):
    actual = normative_paths(root)
    expected = {}
    for number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        if not match:
            raise ValueError(f'malformed SHA256SUMS line {number}')
        digest, name = match.groups()
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or name not in actual:
            raise ValueError('uncovered or invalid normative path: ' + name)
        if name in expected:
            raise ValueError('duplicate normative path: ' + name)
        expected[name] = digest
    if set(expected) != actual:
        raise ValueError('normative inventory mismatch; missing entries: ' + ', '.join(sorted(actual - set(expected))))
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('normative digest mismatch: ' + name)
    return len(actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--list', action='store_true', help='list the normative source inventory without changing it')
    args = parser.parse_args()
    try:
        if args.list:
            print('\n'.join(sorted(normative_paths(args.root.resolve()))))
            return 0
        count = check(args.root.resolve(), args.manifest or args.root / 'opentsp-arch-spec/SHA256SUMS')
    except (OSError, ValueError) as error:
        print('check_freeze: ' + str(error), file=sys.stderr)
        return 1
    print(f'specification 0.1: {count} normative files match SHA256SUMS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
