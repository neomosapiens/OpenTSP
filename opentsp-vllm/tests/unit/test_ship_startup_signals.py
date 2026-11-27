"""Deterministic boundary tests of the actual installed launcher handlers.

Popen is injected at its entry/return boundary; these are no real vLLM runs.
The existing shutdown family separately exercises real child process groups.
"""
import os
import signal

import pytest

from opentsp_vllm import serve

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


@pytest.mark.parametrize("signum", SIGNALS)
@pytest.mark.parametrize("where", ["spawn", "poll"])
def test_first_signal_is_forwarded_once_across_spawn_boundary(monkeypatch, signum, where):
    old = {sig: signal.getsignal(sig) for sig in SIGNALS}
    sent = []
    group_sent = []

    class Child:
        pid = 987654321
        returncode = 0

        def poll(self):
            if where == "poll":
                signal.getsignal(signum)(signum, None)
            return self.returncode

    def spawn(*args, **kwargs):
        assert kwargs["start_new_session"]
        assert kwargs["env"]["OPENTSP_VLLM_PARENT"] == str(os.getpid())
        if where == "spawn":
            signal.getsignal(signum)(signum, None)
        return Child()

    monkeypatch.setattr(serve.subprocess, "Popen", spawn)
    monkeypatch.setattr(serve.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(serve, "_killpg", lambda pid, sig: group_sent.append((pid, sig)))
    monkeypatch.setattr(serve, "_group_alive", lambda pid: False)
    try:
        assert serve.supervise(["unused"], {}) == 0
        assert sent.count((Child.pid, signal.SIGINT)) == 1
        assert group_sent == [(Child.pid, signal.SIGKILL)]
    finally:
        assert {sig: signal.getsignal(sig) for sig in SIGNALS} == old


@pytest.mark.parametrize("signum", SIGNALS)
def test_spawn_error_after_pending_signal_preserves_error_and_restores_handlers(monkeypatch, signum):
    old = {sig: signal.getsignal(sig) for sig in SIGNALS}

    def spawn(*args, **kwargs):
        signal.getsignal(signum)(signum, None)
        raise OSError("startup failed")

    monkeypatch.setattr(serve.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="startup failed"):
        serve.supervise(["unused"], {})
    assert {sig: signal.getsignal(sig) for sig in SIGNALS} == old


def test_second_signal_during_spawn_kills_the_eventual_child(monkeypatch, capsys):
    sent = []

    class Child:
        pid = 987654321
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self):
            assert self.returncode == -signal.SIGKILL
            return self.returncode

    child = Child()

    def send(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGKILL:
            child.returncode = -sig
        return True

    def spawn(*args, **kwargs):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    monkeypatch.setattr(serve.subprocess, "Popen", spawn)
    monkeypatch.setattr(serve.os, "kill", send)
    monkeypatch.setattr(serve, "_killpg", send)
    monkeypatch.setattr(serve, "_group_alive", lambda pid: False)
    assert serve.supervise(["unused"], {}, grace=60) == 130
    assert sent.count((child.pid, signal.SIGINT)) == 1
    assert (child.pid, signal.SIGKILL) in sent
    assert "second Ctrl-C" in capsys.readouterr().err


def test_postspawn_error_cleans_owned_group_and_reaps_child(monkeypatch):
    old = {sig: signal.getsignal(sig) for sig in SIGNALS}
    sent = []
    reaped = []

    class Child:
        pid = 987654321

        def poll(self):
            raise RuntimeError("supervision failed")

        def wait(self):
            reaped.append(self.pid)

    monkeypatch.setattr(serve.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(serve, "_killpg", lambda pid, sig: sent.append((pid, sig)))
    with pytest.raises(RuntimeError, match="supervision failed"):
        serve.supervise(["unused"], {})
    assert (Child.pid, signal.SIGKILL) in sent
    assert reaped == [Child.pid]
    assert {sig: signal.getsignal(sig) for sig in SIGNALS} == old


@pytest.mark.parametrize("signum", SIGNALS)
def test_live_child_ready_before_popen_returns_receives_pending_signal(monkeypatch, signum):
    """A pipe readiness handshake exposes the exact spawn boundary without sleeps."""
    import select
    import subprocess
    import sys
    actual_popen = subprocess.Popen
    children = []
    code = ("import signal, sys; "
            "signal.signal(signal.SIGINT, lambda *_: sys.exit(0)); "
            "print('ready', flush=True); signal.pause()")

    def spawn(*args, **kwargs):
        child = actual_popen(*args, **kwargs, stdout=subprocess.PIPE, text=True)
        children.append(child)
        assert select.select([child.stdout], [], [], 10)[0], "child did not become ready"
        assert child.stdout.readline() == "ready\n"
        signal.getsignal(signum)(signum, None)
        return child

    monkeypatch.setattr(serve.subprocess, "Popen", spawn)
    try:
        assert serve.supervise([sys.executable, "-c", code], dict(os.environ), grace=1) == 0
        assert children[0].poll() == 0
        assert not serve._group_alive(children[0].pid)
    finally:
        for child in children:
            if child.poll() is None:
                serve._killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
            child.stdout.close()
