#!/usr/bin/env python3
"""Compare the checkout with an explicit source baseline and its own Python environment."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile

from check_perf_base import check, source_key

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ('P1_1_64', 'P8_4_128')


class GateError(Exception):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def run(argv, directory, code, **kwargs):
    try:
        return subprocess.run(list(map(str, argv)), cwd=directory, check=True, **kwargs)
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateError(f'command failed in {directory}: {argv}: {error}', code) from error


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_context(directory):
    def output(*args):
        return run(args, directory, 2, capture_output=True, text=True).stdout.strip()
    sysroot = Path(output('rustc', '--print', 'sysroot'))
    names = ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTUP_TOOLCHAIN', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_TARGET')
    return dict(rustc=output('rustc', '-vV'), cargo=output('cargo', '-V'),
                rustc_sha256=sha(sysroot / 'bin/rustc'), cargo_sha256=sha(sysroot / 'bin/cargo'),
                flags={name: os.environ.get(name, '') for name in names})


def baseline_binaries(baseline, cache):
    receipt = cache / 'binaries.json'
    paths = {name: cache / name for name in ('tsp-iss', 'ots-asm')}
    try:
        saved = json.loads(receipt.read_text())
        if all(path.is_file() and os.access(path, os.X_OK) and saved.get(name) == sha(path) for name, path in paths.items()):
            return paths
    except (OSError, ValueError):
        pass
    run(['cargo', 'build', '--locked', '--release', '-q', '-p', 'tsp-iss', '-p', 'ots-asm', '--target-dir', cache / 'target'], baseline, 3)
    with tempfile.TemporaryDirectory(dir=cache, prefix='publish-') as temporary:
        staging = Path(temporary)
        digests = {}
        for name in paths:
            shutil.copy2(cache / 'target/release' / name, staging / name)
            digests[name] = sha(staging / name)
        (staging / 'binaries.json').write_text(json.dumps(digests, indent=2) + '\n')
        for name, path in paths.items():
            os.replace(staging / name, path)
        os.replace(staging / 'binaries.json', receipt)
    return paths


def fixture(directory, python, binaries, output, code):
    # Verify the interpreter actually selects this side's checkout before invoking its generators.
    probe = '''import pathlib,sys,opentsp_tools,opentsp_reference,opentsp_vllm.models.tinystories.build_programs
root=pathlib.Path(sys.argv[1]).resolve()
for module in (opentsp_tools,opentsp_reference,opentsp_vllm.models.tinystories.build_programs):
 p=pathlib.Path(module.__file__).resolve()
 if not p.is_relative_to(root): raise SystemExit(f'{module.__name__} belongs to {p}, not {root}')
'''
    env = dict(os.environ, OPENTSP_ROOT=str(directory), OTS_ASM=str(binaries['ots-asm']), TSP_ISS=str(binaries['tsp-iss']))
    run([python, '-c', probe, directory], directory, code, env=env)
    prepare = [python, directory / 'tools/iss_bench_prep.py', '--out', output, *BUCKETS]
    # The canonical builder owns caching and input invalidation. This driver keeps no second fixture cache.
    run([python, '-m', 'opentsp_vllm.models.tinystories.build_programs', '--layers', '8', '--bucket', BUCKETS[0], '--bucket', BUCKETS[1]], directory, code, env=env)
    run(prepare, directory, code, env=env)
    for bucket in BUCKETS:
        run([binaries['tsp-iss'], 'run', '--profile', 'ref', '--weights', output / (bucket + '.safetensors'), output / (bucket + '.elf')], directory, code, stdout=subprocess.DEVNULL)


def main():
    argparse.ArgumentParser(description=__doc__, epilog='Set ISS_PERF_BASE_DIR and optionally ISS_PERF_BASE_PYTHON, ISS_PERF_TREE_PYTHON, ISS_PERF_PAIRS and the ISS_PERF_* limits.').parse_args()
    value = os.environ.get('ISS_PERF_BASE_DIR')
    if not value:
        raise GateError('set ISS_PERF_BASE_DIR to a distinct source checkout; no repository history is used', 2)
    baseline = Path(value).expanduser().resolve()
    problems = check(ROOT, baseline)
    if problems:
        raise GateError('\n'.join(problems), 2)
    context = dict(source=source_key(baseline), compiler=compiler_context(baseline))
    key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    cache = Path(os.environ.get('ISS_PERF_CACHE', str(Path.home() / '.cache/opentsp/iss-perf'))) / key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binaries = baseline_binaries(baseline, cache)
        if source_key(baseline) != context['source'] or compiler_context(baseline) != context['compiler']:
            raise GateError('baseline build inputs changed during the build', 3)
        (cache / 'inputs.json').write_text(json.dumps(context, indent=2, sort_keys=True) + '\n')
        candidate_source = source_key(ROOT)
        run(['cargo', 'build', '--locked', '--release', '-q', '-p', 'tsp-iss', '-p', 'ots-asm', '--target-dir', ROOT / 'target'], ROOT, 2)
        if source_key(ROOT) != candidate_source:
            raise GateError('candidate source changed during the build', 2)
        candidate = {name: ROOT / 'target/release' / name for name in binaries}
        candidate['tsp-iss'] = Path(os.environ.get('ISS_PERF_TREE', str(candidate['tsp-iss']))).resolve()
        base_python = os.environ.get('ISS_PERF_BASE_PYTHON', sys.executable)
        tree_python = os.environ.get('ISS_PERF_TREE_PYTHON', sys.executable)
        with tempfile.TemporaryDirectory(prefix='iss-perf-fixtures-') as temporary:
            base_fixture, tree_fixture = Path(temporary) / 'base', Path(temporary) / 'candidate'
            fixture(baseline, base_python, binaries, base_fixture, 3)
            fixture(ROOT, tree_python, candidate, tree_fixture, 4)
            command = [sys.executable, ROOT / 'tools/iss_ab_overhead.py', '-n', os.environ.get('ISS_PERF_PAIRS', '7'), '--gate', os.environ.get('ISS_PERF_GATE', '1.15'), '--gate-small', os.environ.get('ISS_PERF_GATE_SMALL', '1.30')]
            instruction_limit = os.environ.get('ISS_PERF_INSN_GATE', '1.01')
            if instruction_limit != '0':
                command.extend(['--insn-gate', instruction_limit])
            if os.environ.get('ISS_PERF_INSN_ONLY', '0') == '1':
                if instruction_limit == '0':
                    raise GateError('ISS_PERF_INSN_ONLY requires the instruction leg', 2)
                command.append('--insn-only')
            command.extend(['--dir', 'base=' + str(base_fixture), '--dir', 'tree=' + str(tree_fixture)])
            for bucket in BUCKETS:
                command.extend(['-p', bucket])
            programs = os.environ.get('ISS_PERF_PROGRAMS', 'b1 b3 b4 dense_f16 dense_i8 vxm_valid transc syncnotify').split()
            for program in programs:
                command.extend(['--bench', ROOT / 'opentsp-iss/crates/tsp-iss/benches/programs' / (program + '.txt')])
            command.extend(['base=' + shlex.quote(str(binaries['tsp-iss'])), 'tree=' + shlex.quote(str(candidate['tsp-iss']))])
            result = subprocess.run(list(map(str, command)), cwd=ROOT)
            if result.returncode:
                return result.returncode
        if source_key(ROOT) != candidate_source or source_key(baseline) != context['source']:
            raise GateError('source inputs changed during the comparison; result is invalid', 2)
        if os.environ.get('ISS_PERF_INSN_ONLY', '0') != '1':
            run(['cargo', 'test', '--locked', '--release', '-q', '-p', 'tsp-iss', '--test', 'iss7', '--', '--ignored'], ROOT, 2)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except GateError as error:
        print('iss-perf-gate: ' + str(error), file=sys.stderr)
        sys.exit(error.code)
