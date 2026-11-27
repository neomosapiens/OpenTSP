#!/usr/bin/env python3
"""Validate an explicit filesystem baseline and identify its build and fixture inputs."""
import argparse
import hashlib
from pathlib import Path
import sys
from source_files import inventory

NEEDED = ('Cargo.toml', 'Cargo.lock', 'opentsp-iss/crates/tsp-iss/Cargo.toml',
          'opentsp-sw-toolchain/crates/ots-asm/Cargo.toml', 'tools/iss_bench_prep.py',
          'opentsp-vllm/src/opentsp_vllm/models/tinystories/build_programs.py')
ENGINE_PREFIXES = ('opentsp-iss/crates/', 'opentsp-sw-toolchain/crates/',
                   'opentsp-arch-spec/isa-db/', 'opentsp-arch-spec/config/')


def fingerprint(root, paths):
    result = hashlib.sha256()
    for name in sorted(set(paths)):
        result.update(name.encode() + b'\0')
        result.update(hashlib.sha256((root / name).read_bytes()).digest())
    return result.hexdigest()


def source_key(root):
    return fingerprint(root, [p for p in inventory(root) if p in NEEDED or p in ('rust-toolchain.toml', 'rust-toolchain') or p.startswith(ENGINE_PREFIXES + ('.cargo/',))])



def check(root, baseline):
    problems = []
    if root.resolve() == baseline.resolve():
        problems.append('the candidate and baseline must be separate source directories')
    for label, directory in (('candidate', root), ('baseline', baseline)):
        for name in NEEDED:
            if not (directory / name).is_file():
                problems.append(f'{label} is missing {name}')
    if not problems and source_key(root) == source_key(baseline):
        problems.append('baseline build inputs are identical to the candidate; use a distinct reference for a regression gate')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--base-dir', type=Path)
    parser.add_argument('--key-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.base_dir is None:
            parser.error('--base-dir is required; use an explicit baseline source directory')
        problems = check(args.root.resolve(), args.base_dir.resolve())
        if problems:
            raise ValueError('\n'.join(problems))
        key = source_key(args.base_dir.resolve())
        print(key if args.key_only else f'baseline source: {args.base_dir.resolve()} (inputs {key})')
    except (OSError, ValueError) as error:
        print('check_perf_base: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
