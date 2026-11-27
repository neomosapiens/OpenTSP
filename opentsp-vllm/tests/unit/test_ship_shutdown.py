"""The launcher's shutdown: Ctrl-C ends the server promptly even with a request in flight (vLLM 0.33 alone waits for the client and ignores a second Ctrl-C).

The child is a stand-in for `vllm serve` (`stub_server.py`) with the behaviour observed on the real server: the engine core is killed at the first signal, the HTTP side waits
for its open connections, later signals do nothing. Each test runs the launcher's `supervise` as its own process group leader and signals that group, as a terminal's Ctrl-C does.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

STUB = os.path.join(os.path.dirname(__file__), "stub_server.py")
SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
CODE = "import sys; from opentsp_vllm import serve; sys.exit(serve.supervise([sys.executable, %r, sys.argv[1]], dict(__import__('os').environ), grace=float(sys.argv[2])))"


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture
def server(tmp_path):
    """(launcher process, info dict); every process of the test is killed at the end."""
    info = tmp_path / "info.json"
    launcher = []

    def start(grace):
        p = subprocess.Popen([sys.executable, "-c", CODE % STUB, str(info), str(grace)], env=dict(os.environ, PYTHONPATH=SRC), start_new_session=True,
                             stderr=subprocess.PIPE, text=True)
        launcher.append(p)
        deadline = time.monotonic() + 30
        while not (info.exists() and info.read_text()):
            assert p.poll() is None and time.monotonic() < deadline, "the stand-in server did not start"
            time.sleep(0.05)
        time.sleep(0.2)
        return p, json.loads(info.read_text())

    yield start
    for p in launcher:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        if p.stderr:
            p.stderr.close()
    if info.exists() and info.read_text():
        for key in ("pid", "engine"):
            try:
                os.kill(json.loads(info.read_text())[key], signal.SIGKILL)
            except ProcessLookupError:
                pass


def hold_request(port):
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 1000\r\n\r\n")  # the body never ends: a request in flight
    time.sleep(0.3)
    return s


def wait_exit(p, limit):
    t0 = time.monotonic()
    try:
        rc = p.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the launcher was still running {limit} s after the signal")
    return rc, time.monotonic() - t0


def test_the_first_ctrl_c_with_a_request_in_flight_ends_the_server_within_the_grace_time_with_status_130(server):
    p, info = server(grace=2)
    client = hold_request(info["port"])
    os.killpg(p.pid, signal.SIGINT)
    rc, took = wait_exit(p, 8)
    assert rc == 130 and took < 5, (rc, took)
    assert "did not end within 2 s" in p.stderr.read()
    time.sleep(0.2)
    assert not alive(info["pid"]) and not alive(info["engine"]), "no process of the server is left"
    client.close()


def test_a_second_ctrl_c_ends_the_server_at_once_with_status_130(server):
    p, info = server(grace=60)
    client = hold_request(info["port"])
    os.killpg(p.pid, signal.SIGINT)
    time.sleep(1)
    assert p.poll() is None, "the first Ctrl-C alone waits for the grace time"
    os.killpg(p.pid, signal.SIGINT)
    rc, took = wait_exit(p, 3)
    assert rc == 130 and took < 2, (rc, took)
    assert "second Ctrl-C" in p.stderr.read()
    time.sleep(0.2)
    assert not alive(info["pid"]) and not alive(info["engine"])
    client.close()


def test_an_idle_server_ends_with_status_0_within_a_second(server):
    p, info = server(grace=60)
    os.killpg(p.pid, signal.SIGINT)
    rc, took = wait_exit(p, 5)
    assert rc == 0 and took < 1.5, (rc, took)
    assert not alive(info["pid"]) and not alive(info["engine"])


def test_sigterm_is_the_first_ctrl_c_too(server):
    p, info = server(grace=1)
    client = hold_request(info["port"])
    os.killpg(p.pid, signal.SIGTERM)
    rc, took = wait_exit(p, 6)
    assert rc == 130 and took < 4
    assert not alive(info["pid"]) and not alive(info["engine"])
    client.close()


def test_a_server_that_fails_gives_its_own_exit_status(tmp_path):
    code = "import sys; from opentsp_vllm import serve; sys.exit(serve.supervise([sys.executable, '-c', 'import sys; sys.exit(3)'], dict(__import__('os').environ), grace=1))"
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=SRC), timeout=20)
    assert r.returncode == 3


def test_a_helper_process_left_behind_by_a_server_that_ended_is_cleared_away(tmp_path):
    pidfile = tmp_path / "helper.pid"
    child = f"import subprocess, sys; p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); open({str(pidfile)!r}, 'w').write(str(p.pid))"
    code = f"import sys; from opentsp_vllm import serve; sys.exit(serve.supervise([sys.executable, '-c', {child!r}], dict(__import__('os').environ), grace=1))"
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=SRC), timeout=20)
    helper = int(pidfile.read_text())
    try:
        assert r.returncode == 0
        time.sleep(0.2)
        assert not alive(helper), "the helper of the ended server is gone"
    finally:
        if alive(helper):
            os.kill(helper, signal.SIGKILL)


def test_sighup_the_terminal_closed_is_the_first_ctrl_c_too(server):
    p, info = server(grace=1)
    os.kill(p.pid, signal.SIGHUP)        # a closed terminal hangs up the launcher only: the server has a session of its own
    rc, took = wait_exit(p, 6)
    assert rc == 0 and took < 2, (rc, took)
    assert not alive(info["pid"]) and not alive(info["engine"])


def test_the_server_ends_when_the_launcher_is_killed_with_sigkill(tmp_path):
    """The child of the launcher calls what the platform plugin calls (`register`), as every process of the real server does."""
    pids = tmp_path / "pids.json"
    child = ("import json, subprocess, sys, time; from opentsp_vllm import register; register(); "
             "h = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)']); "
             f"open({str(pids)!r}, 'w').write(json.dumps([__import__('os').getpid(), h.pid])); time.sleep(600)")
    code = f"import sys; from opentsp_vllm import serve; sys.exit(serve.supervise([sys.executable, '-c', {child!r}], dict(__import__('os').environ, OPENTSP_VLLM='1'), grace=1))"
    launcher = subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=SRC), start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        while not (pids.exists() and pids.read_text()):
            assert launcher.poll() is None and time.monotonic() < deadline, "the stand-in server did not start"
            time.sleep(0.05)
        server_pid, helper = json.loads(pids.read_text())
        assert alive(server_pid) and alive(helper)
        os.kill(launcher.pid, signal.SIGKILL)
        launcher.wait()
        deadline = time.monotonic() + 5
        while (alive(server_pid) or alive(helper)) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not alive(server_pid) and not alive(helper), "the server and its helper outlive a launcher killed with SIGKILL"
    finally:
        for pid in json.loads(pids.read_text()) if pids.exists() and pids.read_text() else []:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_the_watch_does_nothing_without_a_launcher():
    from opentsp_vllm import watchdog
    assert watchdog.start(env={}) is None
    assert watchdog.start(env={"OPENTSP_VLLM_PARENT": "not a pid"}) is None
