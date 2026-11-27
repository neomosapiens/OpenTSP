"""A stand-in for `vllm serve` that behaves as the observed vLLM 0.33 does on Ctrl-C: an engine core child process, an idempotent signal handler (the first signal
kills the engine, later ones are ignored) and an HTTP side that waits for its open connections before it exits (a request in flight keeps it up).

    python stub_server.py INFO.json     # writes {"pid", "engine", "port"} when it listens
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

info = sys.argv[1]
engine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
stopping = threading.Event()
open_connections = []


def on_signal(signum, frame):
    if stopping.is_set():
        return  # vLLM's handler: `if shutdown_event.is_set: return`
    stopping.set()
    engine.kill()
    engine.wait()  # reaped: a zombie would still answer kill(pid, 0)


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen()
server.settimeout(0.1)
with open(info, "w") as f:
    json.dump({"pid": os.getpid(), "engine": engine.pid, "port": server.getsockname()[1]}, f)
while True:
    try:
        conn, _ = server.accept()
        open_connections.append(conn)
    except socket.timeout:
        pass
    if stopping.is_set() and not open_connections:
        sys.exit(0)  # idle: the server ends at once
    for c in list(open_connections):
        c.setblocking(False)
        try:
            if c.recv(4096) == b"":  # the client closed
                open_connections.remove(c)
        except BlockingIOError:
            pass
        except OSError:
            open_connections.remove(c)
    time.sleep(0.02)
