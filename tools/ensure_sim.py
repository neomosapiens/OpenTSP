#!/usr/bin/env python3
"""Build a simulator only when its complete recorded build inputs change."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from source_files import source_paths


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def context(root):
    spec = root / 'opentsp-arch-spec'
    paths = []
    for directory in ('model', 'c_emulator'):
        source = spec / directory
        if not source.is_dir():
            raise ValueError('missing simulator source directory: ' + str(source))
        paths.extend(source_paths(source))
    paths.extend(spec / name for name in ('config/schema.json', 'tools/build_sim.sh', 'tools/schema_header.py'))
    tools = {}
    for name in ('sail', 'cc'):
        binary = shutil.which(name)
        if binary is None:
            raise ValueError(name + ' is missing; source the specification toolchain')
        resolved = Path(binary).resolve()
        version = subprocess.check_output([binary, '--version'], text=True, timeout=10).splitlines()[0]
        tools[name] = {'path': str(resolved), 'version': version, 'sha256': digest(resolved)}
        if name == 'sail':
            runtime = (Path(binary).parent / '../share/sail/lib').resolve()
            if not runtime.is_dir():
                raise ValueError('missing Sail C runtime: ' + str(runtime))
            tools[name]['c_runtime'] = {p.relative_to(runtime).as_posix(): digest(p) for p in sorted(source_paths(runtime))}
    python = Path(sys.executable).resolve()
    tools['python'] = {'path': str(python), 'version': platform.python_version(), 'sha256': digest(python)}
    return {'inputs': {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}, 'tools': tools,
            'machine': platform.machine(), 'system': platform.system(),
            'environment': {k: os.environ.get(k, '') for k in ('GMP_PREFIX', 'GMP_LIBS', 'Z_LIBS')}}


def ensure(root, binary):
    binary = binary.resolve()
    binary.parent.mkdir(parents=True, exist_ok=True)
    receipt = binary.with_name(binary.name + '.build.json')
    with binary.with_name(binary.name + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = context(root)
        try:
            previous = json.loads(receipt.read_text())
        except (OSError, ValueError):
            previous = {}
        if binary.is_file() and os.access(binary, os.X_OK) and previous.get('context') == current and previous.get('binary_sha256') == digest(binary):
            return
        print('ensure_sim: building ' + str(binary) + ' (unrecorded or changed build inputs)', file=sys.stderr)
        with tempfile.TemporaryDirectory(prefix=binary.name + '.build-', dir=binary.parent) as temporary:
            candidate = Path(temporary) / binary.name
            subprocess.run(['bash', str(root / 'opentsp-arch-spec/tools/build_sim.sh'), str(candidate)], cwd=root, check=True)
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                raise ValueError('simulator build did not produce an executable')
            if context(root) != current:
                raise ValueError('simulator inputs changed during the build; no candidate was installed')
            data = {'context': current, 'binary_sha256': digest(candidate)}
            new_receipt = Path(temporary) / 'receipt.json'
            new_receipt.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
            os.replace(candidate, binary)
            os.replace(new_receipt, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    args = parser.parse_args()
    try:
        ensure(Path(__file__).resolve().parents[1], args.binary)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print('ensure_sim: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
