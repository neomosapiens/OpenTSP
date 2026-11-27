"""Element vectors and their byte planes (topology.md sec. 3): lane = element; a w-byte element occupies w aligned
streams, plane k holding byte k (little-endian) of every lane, and is stored as w SRAM words."""
import numpy as np

DTYPES = {"i8": np.int8, "u8": np.uint8, "i16": np.int16, "u16": np.uint16, "i32": np.int32, "u32": np.uint32,
          "f16": np.float16, "f32": np.float32}


def width(t):
    """Bytes of an element of type t; a boolean type b8, b16, b32 is as wide as its integer."""
    return int(t[1:]) // 8 if t[0] == "b" else np.dtype(DTYPES[t]).itemsize


def to_planes(arr):
    """A little-endian element vector -> [plane0 bytes, plane1 bytes, ...]."""
    a = np.ascontiguousarray(arr, dtype=np.dtype(arr.dtype).newbyteorder("<"))
    b = a.view(np.uint8).reshape(len(a), a.itemsize)
    return [bytes(b[:, k]) for k in range(a.itemsize)]


def from_planes(planes, t):
    b = np.stack([np.frombuffer(p, np.uint8) for p in planes], axis=1)
    return b.copy().view(np.dtype(DTYPES[t]).newbyteorder("<")).reshape(-1)


def rand_int(rng, t, n):
    info = np.iinfo(DTYPES[t])
    return rng.integers(info.min, info.max, size=n, endpoint=True, dtype=DTYPES[t])


def rand_float(rng, t, n, lo=-12, hi=12):
    """Finite values with random signs, exponents in [lo, hi) and full random significands, plus 0 and -0."""
    m = rng.random(n) + 1.0
    v = np.ldexp(m, rng.integers(lo, hi, size=n)) * rng.choice([-1.0, 1.0], size=n)
    v = v.astype(DTYPES[t])
    v[0], v[min(1, n - 1)] = 0.0, -0.0
    return v
