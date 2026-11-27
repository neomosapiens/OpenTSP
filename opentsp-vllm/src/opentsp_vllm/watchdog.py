"""The server ends with its launcher.

`python -m opentsp_vllm.serve` starts the server in a process group of its own and passes its pid in `OPENTSP_VLLM_PARENT`. A launcher that is killed with SIGKILL, or
that dies by a fault, cannot clean up; the server would go on, a child of nobody, holding its port. Every process of the server calls `start` (through the platform
plugin's `register`); a thread there looks at the launcher once in `POLL` seconds and, when the launcher is gone, kills the server's whole process group. Like the
rest of the plugin it imports nothing of vLLM.
"""
import os
import signal
import threading
import time

POLL = 0.5


def _gone(pid):
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:       # the pid exists (another user's): not gone
        return False


def start(env=None, poll=POLL, on_gone=None):
    """Start the watch when `OPENTSP_VLLM_PARENT` names a launcher; returns the thread, or None when it does not (a server started without the launcher)."""
    env = os.environ if env is None else env
    try:
        parent = int(env.get("OPENTSP_VLLM_PARENT", ""))
    except ValueError:
        return None
    if parent <= 1:
        return None

    def kill_group():
        os.killpg(os.getpgrp(), signal.SIGKILL)

    def watch():
        while not _gone(parent):
            time.sleep(poll)
        (on_gone or kill_group)()

    t = threading.Thread(target=watch, name="opentsp-launcher-watch", daemon=True)
    t.start()
    return t
