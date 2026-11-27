"""Private ISS device process. A wedged native run cannot block its host watchdog."""

import functools
import hashlib
import math
import os
import pickle
import socket
import struct
import subprocess
import sys
import time

from .contracts import DomainFault

# One frame holds a control message or one scheduler batch of logits.
# Weights and resident KV are loaded directly inside the device process.
MAX_FRAME = 64 * 1024 * 1024
HEADER = struct.Struct("!Q32s")


# This private socketpair only connects the host to its own interpreter.
def _send(sock, message):
    data = pickle.dumps(message, protocol=5)
    if len(data) > MAX_FRAME:
        raise ValueError("device frame exceeds the transport limit")
    sock.sendall(HEADER.pack(len(data), hashlib.sha256(data).digest()))
    sock.sendall(data)


def _read(sock, n, deadline):
    chunks = bytearray()
    while len(chunks) < n:
        if deadline is not None:
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            if time.monotonic() >= deadline:
                raise TimeoutError("device reply deadline expired")
        part = sock.recv(min(n - len(chunks), 1 << 20))
        if not part:
            raise EOFError("device process exited")
        chunks.extend(part)
    return chunks


def _receive(sock, deadline=None):
    size, digest = HEADER.unpack(_read(sock, HEADER.size, deadline))
    if not 0 < size <= MAX_FRAME:
        raise ValueError("invalid device frame length")
    data = _read(sock, size, deadline)
    if hashlib.sha256(data).digest() != digest:
        raise ValueError("device frame integrity check failed")
    return pickle.loads(data)


def _seconds(variable, default):
    value = float(os.environ.get(variable, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{variable} must be a finite positive number")
    return value


class DeviceProcess:
    def __init__(self, factory):
        self.timeout = _seconds("OPENTSP_RAS_STEP_TIMEOUT_S", 300)
        self.load_timeout = _seconds("OPENTSP_RAS_LOAD_TIMEOUT_S", 3600)
        self.incarnation = os.urandom(16)
        self.sequence = 0
        self.sock, child = socket.socketpair()
        try:
            self.proc = subprocess.Popen([sys.executable, "-m", __name__, str(child.fileno())], pass_fds=(child.fileno(),))
        except BaseException:
            self.sock.close()
            raise
        finally:
            child.close()
        try:
            self.info = self._call("create", factory)
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        timeout = self.load_timeout if name in ("create", "load_model") else self.timeout
        deadline = time.monotonic() + timeout
        self.sock.settimeout(timeout)
        try:
            self.sequence += 1
            tag = (self.incarnation, self.sequence)
            _send(self.sock, (tag, name, args))
            received, ok, result = _receive(self.sock, deadline)
            if received != tag:
                raise DomainFault("stale device reply")
            if type(ok) is not bool or (not ok and not isinstance(result, Exception)):
                raise ValueError("invalid device reply")
        except (OSError, EOFError, ValueError, TypeError, pickle.PickleError) as exc:
            raise DomainFault(f"device unavailable during {name}: {exc}") from exc
        if not ok:
            raise result
        return result

    def __getattr__(self, name):
        return functools.partial(self._call, name)

    def close(self):
        self.sock.close()
        if self.proc.poll() is None:
            self.proc.kill()
        # Confirm process death before its output or storage can be reused.
        self.proc.wait(timeout=5)


def serve(fd):
    with socket.socket(fileno=fd) as sock:
        chip = None
        incarnation = None
        sequence = 0
        while True:
            try:
                tag, name, args = _receive(sock)
            except EOFError:
                return
            try:
                sequence += 1
                if sequence == 1:
                    incarnation = tag[0]
                if tag != (incarnation, sequence) or (name == "create") != (sequence == 1):
                    raise DomainFault("out-of-order device command")
                if name == "create":
                    factory, = args
                    chip = factory()
                    result = chip.info
                else:
                    result = getattr(chip, name)(*args)
                _send(sock, (tag, True, result))
            except Exception as exc:
                if not isinstance(exc, (ValueError, RuntimeError, OSError)):
                    exc = RuntimeError(f"{type(exc).__name__}: {exc}")
                _send(sock, (tag, False, exc))


if __name__ == "__main__":
    serve(int(sys.argv[1]))
