#!/usr/bin/env python3
"""Run unchanged prove-extra suites in bounded, isolated process groups."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import threading

SUITES = ('domains', 'ties', 'frontend', 'wide', 'hold', 'packing', 'numerics', 'fptheory')


RUNNING = set()
LOCK = threading.Lock()
INTERRUPTED = threading.Event()

def stop_all():
    INTERRUPTED.set()
    with LOCK:
        groups = list(RUNNING)
    if not groups:
        return
    # A container PID 1 need not reap orphaned grandchildren. Linux adoption
    # is process-wide; all waiting and signals remain scoped to OUR groups.
    libc, previous, reaping_error = None, None, None
    if sys.platform == "linux":
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        prior = ctypes.c_int()
        if libc.prctl(37, ctypes.byref(prior), 0, 0, 0) != 0 or libc.prctl(36, 1, 0, 0, 0) != 0:
            reaping_error = OSError(ctypes.get_errno(), "cannot enable owned-child reaping")
        else:
            previous = prior.value
    try:
        for pid in groups:
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        # Finish before the enclosing mutation runner's five-second KILL.
        deadline = time.monotonic() + 2
        while libc is not None:
            live = False
            for group in groups:
                while True:
                    try:
                        child, _ = os.waitpid(-group, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if child == 0:
                        break
                try:
                    os.killpg(group, 0)
                    live = True
                except ProcessLookupError:
                    pass
            if not live:
                break
            if time.monotonic() >= deadline:
                for group in groups:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Give adopted children a bounded second chance to be reaped.
                if time.monotonic() >= deadline + 1:
                    raise RuntimeError("owned gate descendants remain after cancellation")
            time.sleep(0.02)
    finally:
        if libc is not None and previous is not None and libc.prctl(36, previous, 0, 0, 0) != 0:
            raise OSError("cannot restore child-subreaper setting")
    if reaping_error is not None:
        raise reaping_error


def run(script, spec, jobs, suites):
    if sys.platform != 'linux':
        raise ValueError('PROVE_EXTRA_JOBS>1 is supported only on Linux; use serial PROVE_EXTRA_JOBS=1')
    if not 1 <= jobs <= 8:
        raise ValueError('PROVE_EXTRA_JOBS must be an integer from 1 through 8')
    if not suites or len(suites) != len(set(suites)):
        raise ValueError('suite inventory is empty or duplicated')
    for suite in suites:
        if suite not in SUITES:
            raise ValueError('unknown suite: ' + suite)
        for suffix in ('.sail', '.txt'):
            if not (spec / 'test/formal/props' / (suite + suffix)).is_file():
                raise ValueError('missing suite source/inventory: ' + suite + suffix)
    fail_fast = os.environ.get('PROVE_FAIL_FAST') == '1'
    INTERRUPTED.clear()
    keep = os.environ.get('PROVE_KEEP')
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='parallel-extra-', dir=keep))
    active = {}
    completed = {}
    pending = list(suites)
    interrupted = [None]

    def cancel(signum, _frame):
        interrupted[0] = signum

    old_handlers = {s: signal.signal(s, cancel) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while pending or active:
            while pending and len(active) < jobs and interrupted[0] is None:
                suite = pending.pop(0)
                child_work = work / suite
                child_work.mkdir()
                log = (work / (suite + '.log')).open('w')
                env = dict(os.environ, PROVE_EXTRA_JOBS='1', PROVE_KEEP=str(child_work))
                try:
                    process = subprocess.Popen(['bash', str(script), suite], env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                with LOCK:
                    RUNNING.add(process.pid)
                active[suite] = (process, log)
                print(f'prove_extra: started {suite} pid={process.pid} log={work / (suite + ".log")}', flush=True)
            for suite, (process, log) in list(active.items()):
                result = process.poll()
                if result is None:
                    continue
                log.close()
                completed[suite] = result
                del active[suite]
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    with LOCK:
                        RUNNING.discard(process.pid)
                print((work / (suite + '.log')).read_text(), end='', flush=True)
                print(f'prove_extra: completed {suite} exit={result}', flush=True)
            if interrupted[0] is not None:
                return 128 + interrupted[0]
            if fail_fast and any(result != 0 for result in completed.values()):
                return 1
            if active:
                time.sleep(0.2)
        if set(completed) != set(suites):
            raise ValueError('completed suite inventory differs from request')
        receipt = {'requested': suites, 'results': completed, 'jobs': jobs}
        (work / 'results.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return 0 if all(result == 0 for result in completed.values()) else 1
    finally:
        try:
            try:
                stop_all()
            finally:
                for process, log in active.values():
                    process.wait(timeout=1)
                    log.close()
                RUNNING.clear()
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            if not keep:
                import shutil
                shutil.rmtree(work)


def main(argv):
    if len(argv) != 4:
        raise ValueError('expected script spec jobs "suite ..."')
    return run(Path(argv[0]).resolve(), Path(argv[1]).resolve(), int(argv[2]), argv[3].split())


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except (ValueError, OSError) as error:
        print('prove_extra: ' + str(error), file=sys.stderr)
        sys.exit(2)
