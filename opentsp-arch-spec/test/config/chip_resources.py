#!/usr/bin/env python3
"""Exercise coordinator resource ownership using real failing Linux syscalls."""
import ctypes
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def run(sim, config, library, fault=None, index=1, data=""):
    env = dict(os.environ)
    if fault:
        env.update(LD_PRELOAD=str(library))
        env["OPENTSP_TEST_" + fault] = str(index)
    child = subprocess.Popen([sim, str(config), "--chips", "2"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env, start_new_session=True)
    try:
        out, err = child.communicate(data, timeout=5)
    except BaseException:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        raise
    adopted = []
    until = time.monotonic() + 1
    while time.monotonic() < until:
        try:
            pid, status = os.waitpid(-child.pid, os.WNOHANG)
        except ChildProcessError:
            break
        if pid:
            adopted.append((pid, status))
        else:
            time.sleep(.01)
    assert not adopted, (fault, "coordinator did not reap children", adopted)
    return child.returncode, out, err


def main():
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc":
        print("chip resource injection: Linux glibc required; not exercised on this host")
        return
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(36, 1, 0, 0, 0) == 0, "subreaper unavailable"
    sim = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="chip-resources-") as directory:
        library = Path(directory) / "faults.so"
        subprocess.run(["cc", "-shared", "-fPIC", str(Path(__file__).with_name("chip_resource_faults.c")),
                        "-o", str(library)], check=True, timeout=30)
        config = ROOT / "config/mini.json"
        for index in range(1, 9):
            code, _, err = run(sim, config, library, "ALLOC", index)
            assert code == 2 and "out of memory" in err and "TEST forked" not in err, (index, code, err)
        route = "C 0\nQ 0 10000001 00000000\nQ 8c 00000019 50010000\nS 0 1 " + " ".join(["01"] * 16) + "\n"
        for fault, index, code, marker, data in [
            ("FORK", 2, 2, "fork-failed", ""),
            ("PIPE", 3, 2, "pipe-failed", ""),
            ("READ", 1, 4, "read-failed", ""),
            ("WRITE", 1, 5, "write-failed", ""),
            ("EPIPE", 1, 5, "epipe-write", ""),
            ("REALLOC", 1, 2, "realloc-failed", route),
        ]:
            actual, _, err = run(sim, config, library, fault, index, data)
            assert actual == code and "TEST " + marker in err, (fault, actual, err)
        # IPC recovery must not silently suppress the caller's stdout SIGPIPE policy.
        reader, writer = os.pipe()
        os.close(reader)
        try:
            result = subprocess.run([sim, str(config), "--chips", "2"], input=b"",
                                    stdout=writer, stderr=subprocess.PIPE, timeout=5)
        finally:
            os.close(writer)
        assert result.returncode == -signal.SIGPIPE, result.returncode
    print(json.dumps({"resource_failures": 14, "stdout_sigpipe": "preserved"}))


if __name__ == "__main__":
    main()
