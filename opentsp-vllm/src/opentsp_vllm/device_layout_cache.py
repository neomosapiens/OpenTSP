"""Opt-in local cache for derived SRAM layouts.

The cache trusts the account that owns it. The caller supplies an identity
over verified canonical sources and checks the returned physical layout.
"""
from hashlib import sha256
import os
from pathlib import Path
import stat
import tempfile

MAX_IMAGE = 128 * 1024 * 1024


def _private(path, *, directory):
    return ((stat.S_ISDIR if directory else stat.S_ISREG)(path.st_mode)
            and path.st_uid == os.getuid() and not path.st_mode & 0o077)


def _read(path, key):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not _private(status, directory=False) or not 64 <= status.st_size <= 64 + MAX_IMAGE:
            raise ValueError("unsafe or oversized device layout cache entry")
        image = stream.read(64 + MAX_IMAGE + 1)
    payload = image[64:]
    if len(image) != status.st_size or image[:32] != key or image[32:64] != sha256(payload).digest():
        raise ValueError("invalid device layout cache entry")
    return payload


def _write(path, key, data):
    if len(data) > MAX_IMAGE:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".layout-", delete=False) as stream:
        staged = Path(stream.name)
        try:
            stream.write(key)
            stream.write(sha256(data).digest())
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(staged, path)
        finally:
            staged.unlink(missing_ok=True)


def enabled():
    return bool(os.environ.get("OPENTSP_DEVICE_LAYOUT_CACHE"))


def resolve(identity, build, decode, *, directory=None):
    """Use the validated bytes read from a private cache, or build from canonical input."""
    root = directory or os.environ.get("OPENTSP_DEVICE_LAYOUT_CACHE")
    if not root:
        return decode(build())
    key = sha256(b"opentsp.sram-layout.v1\0" + identity).digest()
    root = Path(root).expanduser()
    path = root / (key.hex() + ".layout")
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        private = _private(root.lstat(), directory=True)
    except OSError:
        private = False
    if private:
        try:
            return decode(_read(path, key))
        except (OSError, ValueError):
            pass
    data = build()
    resolved = decode(data)
    if private:
        try:
            _write(path, key, data)
        except OSError:
            pass
    return resolved
