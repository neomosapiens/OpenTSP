"""The reference twin of each kernel: numpy functions that follow the operation order of the
kernel emitters in `kernels/` and compute every step with `numerics`. A vector is an array whose last axis is the
lane axis; vectors of a batch (heads, cache slots) are the leading axes."""
import numpy as np

from opentsp_tools.programs import numerics as nx

LANES = 320          # lanes of the unit that sums an MXM dot product (`ref`)

GELU_C0 = np.float32(0.044715)
GELU_C1 = np.float32(0.7978845608028654)      # sqrt(2/pi), rounded to f32
ONE = np.float32(1.0)
HALF = np.float32(0.5)


def _add(a, b):
    return nx.binary("add", "f32", a, b)


def _sub(a, b):
    return nx.binary("sub", "f32", a, b)


def _mul(a, b):
    return nx.binary("mul", "f32", a, b)


def to_f16(x):
    return nx.convert("f32", "f16", x)


def from_f16(x):
    return nx.convert("f16", "f32", x)


def residual(a, b):
    return _add(a, b)


def matvec_f16(w, x, b=None, lanes=LANES):
    """y = W x (+ b): x rounded to f16 once, the exact dot rounded once to f32, then the f32 bias add."""
    y = nx.dot_f16(w, to_f16(x), lanes)
    return y if b is None else _add(y, b)


def reduce_bcast(op, x, segment=None):
    """Reduce per superlane, the SXM butterfly over each segment of four superlanes (`segment` superlanes: 8 is the
    three-round tree of the C = 128 softmax), then Distribute: every lane holds its segment's total."""
    return nx.distribute_first(nx.reduce_bcast(op, "f32", x, segment))


def layernorm(x, gamma, beta, eps):
    """Two-pass mean and variance in f32 over the lane axis (a power of two of 16 lanes); RSqrt correctly rounded."""
    inv_n = np.float32(1.0 / x.shape[-1])
    d = _sub(x, _mul(reduce_bcast("add", x), inv_n))
    var = _mul(reduce_bcast("add", _mul(d, d)), inv_n)
    r = nx.transcendental("rsqrt", "f32", _add(var, np.float32(eps)))
    return _add(_mul(_mul(d, r), gamma), beta)


def head_lane_masks(heads, hidden):
    """(heads, hidden) f16: 1.0 in the lanes of head h, +0 elsewhere."""
    m = np.zeros((heads, hidden), np.float16)
    dk = hidden // heads
    for h in range(heads):
        m[h, h * dk:(h + 1) * dk] = 1.0
    return m


def head_mask(q16, lane_mask):
    """q (f16) kept in the lanes where the mask is set, +0 elsewhere."""
    return nx.mask("f16", q16, lane_mask)


def scores(k16, q, mask, lane_masks, lanes=LANES):
    """Per head h: the dot of every cache row (k16: slots x hidden, f16) with q rounded to f16 and masked to head h,
    scaled by 0.5, plus the additive mask (0 or -inf per slot). Result (heads, slots) f32."""
    qm = head_mask(to_f16(q), lane_masks)
    return _add(_mul(nx.dot_f16(k16, qm, lanes), HALF), mask)


def softmax(s, segment=None):
    """Over the last axis (the cache slots of a head), f32; a masked slot (-inf) gives +0. 1/s is rsqrt(s*s).
    `segment` as for reduce_bcast."""
    e = nx.transcendental("exp", "f32", _sub(s, reduce_bcast("max", s, segment)))
    tot = reduce_bcast("add", e, segment)
    return _mul(e, nx.transcendental("rsqrt", "f32", _mul(tot, tot)))


def expand(p, dk):
    """p (heads, slots) f32 -> (slots, heads * dk): row s holds p[h][s] in the dk lanes of head h; p is rounded to f16
    once and the expansion is exact."""
    return np.repeat(from_f16(to_f16(p)).T, dk, axis=1)


def context(p_lanes, v):
    """sum over slots of p_lanes[s] * v[s] (both slots x lanes, f32) in four interleaved sequential sums over the
    slots s mod 4, each starting at +0, combined as (c0 + c1) + (c2 + c3)."""
    prod = _mul(p_lanes, v)
    acc = []
    for j in range(4):
        a = np.zeros(prod.shape[1:], np.float32)
        for row in prod[j::4]:
            a = _add(a, row)
        acc.append(a)
    return _add(_add(acc[0], acc[1]), _add(acc[2], acc[3]))


def gelu(x):
    """0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3))) as nine VXM operations."""
    x3 = _mul(_mul(x, x), x)
    v = _mul(_add(x, _mul(x3, GELU_C0)), GELU_C1)
    return _mul(_mul(_add(nx.transcendental("tanh", "f32", v), ONE), HALF), x)
