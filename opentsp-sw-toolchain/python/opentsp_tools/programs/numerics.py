"""The numpy twin of the lane arithmetic (microarchitecture.md sec. 5 and 6).

Every function takes and returns numpy arrays of the element type's dtype (booleans and compare results are the
unsigned integer of the element width, all ones for true). Floats are IEEE binary16 and binary32, round to
nearest even, subnormals kept, one canonical NaN per format. Integers wrap, saturate or convert as the types
say. The code does not call the oracle for anything but the transcendentals the f64 fast path cannot decide:
those few elements (a result within 2**-40 relative of a rounding boundary, or an input with no finite result)
are evaluated by `opentsp-arch-spec/test/oracle/numerics.py` (`fp_transcendental`, mpmath), so a result is
always the correctly rounded one. `STATS` counts them.
"""

import numpy as np

from opentsp_tools.programs.data import DTYPES

UNSIGNED = {1: np.uint8, 2: np.uint16, 4: np.uint32}
CNAN = {"f16": 0x7E00, "f32": 0x7FC00000}
BOOL_WIDTH = {"b8": 1, "b16": 2, "b32": 4}
INT_RANGE = {t: (np.iinfo(DTYPES[t]).min, np.iinfo(DTYPES[t]).max) for t in DTYPES if t[0] in "iu"}
STATS = {"tr_total": 0, "tr_oracle": 0}
TR_NAMES = ("tanh", "exp", "rsqrt", "log")
_oracle_module = None


def bits(a):
    """The unsigned-integer view of an array (same width)."""
    a = np.ascontiguousarray(a)
    return a.view(UNSIGNED[a.dtype.itemsize])


def from_bits(b, t):
    return np.ascontiguousarray(b, dtype=UNSIGNED[np.dtype(DTYPES[t]).itemsize]).view(DTYPES[t])


def truth_mask(t, truth):
    """All ones where `truth`, else zero, as the unsigned integer of the element width of type t."""
    w = np.dtype(DTYPES[t] if t in DTYPES else UNSIGNED[BOOL_WIDTH[t]]).itemsize
    ones = np.iinfo(UNSIGNED[w]).max
    return np.where(truth, UNSIGNED[w](ones), UNSIGNED[w](0))


def canon(a):
    """Every NaN of a float array becomes the canonical NaN of its format."""
    t = "f16" if a.dtype == np.float16 else "f32"
    nan = np.isnan(a)
    if nan.any():
        a = a.copy()
        bits(a)[nan] = CNAN[t]
    return a


def _is_float(t):
    return t in ("f16", "f32")


# -- binary lane kernels -----------------------------------------------------------------------------------------

def _fminmax(a, b, want_max):
    first = (a > b) if want_max else (a < b)
    second = (b > a) if want_max else (b < a)
    # equal operands: max picks +0 over -0, min picks -0 over +0 (the sign decides, not the operand order)
    tie = np.where(np.signbit(a), b, a) if want_max else np.where(np.signbit(a), a, b)
    r = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.where(first, a, np.where(second, b, tie))))
    return canon(r.astype(a.dtype))


def binary(op, t, a, b):
    """add, sub, mul, max, min on any numeric type; add_sat, sub_sat, mul_sat, add_mod, sub_mod, mul_mod, and, or,
    xor on the integer types."""
    a, b = np.asarray(a, dtype=DTYPES[t]), np.asarray(b, dtype=DTYPES[t])
    if _is_float(t):
        with np.errstate(all="ignore"):
            if op == "add":
                return canon(np.add(a, b, dtype=a.dtype))
            if op == "sub":
                return canon(np.subtract(a, b, dtype=a.dtype))
            if op == "mul":
                return canon(np.multiply(a, b, dtype=a.dtype))
        return _fminmax(a, b, op == "max")
    if op in ("and", "or", "xor"):
        return {"and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[op](a, b)
    if op in ("max", "min"):
        return (np.maximum if op == "max" else np.minimum)(a, b)
    base, _, kind = op.partition("_")
    unsigned_mul = t[0] == "u" and base == "mul"
    wide = np.uint64 if unsigned_mul else np.int64
    x, y = a.astype(wide), b.astype(wide)
    with np.errstate(all="ignore"):
        v = {"add": x + y, "sub": x - y, "mul": x * y}[base]
    if kind == "sat":
        lo, hi = INT_RANGE[t]
        v = np.clip(v, lo, hi)
    return v.astype(DTYPES[t])


def cmp(mode, t, a, b):
    """All ones where `a <mode> b` (a NaN operand compares false, except `ne`)."""
    a, b = np.asarray(a, dtype=DTYPES[t]), np.asarray(b, dtype=DTYPES[t])
    r = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[mode]
    return truth_mask(t, r)


def select(t, a, b, m):
    """dst = mask != 0 ? a : b; the mask is any array whose element is true when any of its bits is set."""
    m = np.asarray(m)
    set_bits = bits(m) != 0                       # not `m != 0`: a float mask of -0.0 is true
    return np.where(set_bits, np.asarray(a, dtype=DTYPES[t]), np.asarray(b, dtype=DTYPES[t]))


def mask(t, a, m):
    return select(t, a, np.zeros((), dtype=DTYPES[t]), m)


# -- unary lane kernels ------------------------------------------------------------------------------------------

def unary(op, t, a):
    """neg, abs, relu on signed integers (wrapping) and floats; not on integers."""
    a = np.asarray(a, dtype=DTYPES[t])
    if op == "not":
        return np.invert(a)
    if _is_float(t):
        if op == "neg":
            return canon(-a)
        if op == "abs":
            return canon(np.abs(a))
        # relu: NaN -> canonical NaN, -0 and negatives -> +0
        r = np.where(np.signbit(a) | (a == 0), np.zeros((), dtype=a.dtype), a).astype(a.dtype)
        return canon(np.where(np.isnan(a), a, r))
    w = a.astype(np.int64)
    return {"neg": -w, "abs": np.abs(w), "relu": np.maximum(w, 0)}[op].astype(a.dtype)


# -- conversions -------------------------------------------------------------------------------------------------

def convert(st, dt, a):
    """Integers saturate, floats round to nearest even, NaN gives 0 as an integer, a boolean is `value != 0`."""
    a = np.asarray(a, dtype=DTYPES[st])
    if dt in BOOL_WIDTH:
        return truth_mask(dt, a != 0)
    if _is_float(dt):
        with np.errstate(all="ignore"):
            return canon(a.astype(np.float64).astype(DTYPES[dt]))
    with np.errstate(all="ignore"):
        x = a.astype(np.float64)
        if _is_float(st):
            x = np.where(np.isnan(x), 0.0, np.rint(x))
    lo, hi = INT_RANGE[dt]
    return np.clip(x, lo, hi).astype(DTYPES[dt])


# -- Reduce ------------------------------------------------------------------------------------------------------

def reduce(op, t, x):
    """Per superlane (16 lanes of the last axis): lane 16g = the fold over lanes 16g..16g+15 in ascending lane order,
    one rounding (or wrap) per step; every other lane is zero. op: add, max or min."""
    x = np.asarray(x, dtype=DTYPES[t])
    cols = x.reshape(x.shape[:-1] + (-1, 16))
    acc = cols[..., 0]
    for j in range(1, 16):
        acc = binary(op, t, acc, cols[..., j])
    out = np.zeros(cols.shape, dtype=x.dtype)
    out[..., 0] = acc
    return out.reshape(x.shape)


SEGMENT = 4      # superlanes of one reduce-and-broadcast tree


def rotation_map(vlanes, rot, segment=SEGMENT):
    """The Permute map that moves, inside every segment of `segment` superlanes, superlane g to g - rot (modulo the
    segment) and keeps the lane within the superlane: dst[l] = src[map[l]]. A last segment that the vector cuts short
    rotates inside its own superlanes, so the map stays a bijection (LoadMap requires one)."""
    lane = np.arange(vlanes)
    g, j = lane // 16, lane % 16
    base = (g // segment) * segment
    size = np.minimum(segment, vlanes // 16 - base)
    return 16 * (base + (g - base + rot) % size) + j


def reduce_bcast(op, t, x, segment=None):
    """Reduce per superlane, then rounds of `Permute by a rotation and combine`, the rotations halving from segment / 2
    to 1 superlane : lane 16g of every superlane of a segment holds the segment's total,
    bit-identical in all of them (the combine is commutative); every other lane is zero. The last axis is the lane
    axis. By default the segment is SEGMENT superlanes (2 and 1: two rounds), or the whole vector when it has fewer
    (a power of two); `segment` = 8 is the three-round tree of the C = 128 softmax (rotations 4, 2, 1), whose last
    segment may be partial (rotation_map)."""
    s = reduce(op, t, x)
    superlanes = s.shape[-1] // 16
    if segment is None:
        segment = min(SEGMENT, superlanes)
        assert superlanes % segment == 0, superlanes
    assert segment & (segment - 1) == 0, segment
    rot = segment // 2
    while rot:
        s = binary(op, t, s, s[..., rotation_map(s.shape[-1], rot, segment)])
        rot //= 2
    return s


def distribute_first(x):
    """Distribute with a map of zeros: every lane of a superlane takes the superlane's lane 0."""
    x = np.asarray(x)
    return np.repeat(x[..., ::16], 16, axis=-1)


# -- transcendentals ---------------------------------------------------------------------------------------------

def oracle():
    """The maintained architecture numerical reference used by exact fallbacks."""
    from opentsp_reference import numerics
    return numerics


def _libm(op, x):
    return {"tanh": np.tanh, "exp": np.exp, "log": np.log, "rsqrt": lambda v: 1.0 / np.sqrt(v)}[op](x)


def _i8_transcendental(op, a):
    """The exact value rounded to the nearest integer (ties to even) and saturated; +-infinity at 0 saturates,
    a negative input gives 0 for rsqrt and log (microarchitecture.md sec. 5)."""
    v = a.astype(np.float64)
    with np.errstate(all="ignore"):
        y = _libm(op, v)
    r = np.rint(y)
    r = np.where(v == 0, {"tanh": 0, "exp": 1, "rsqrt": 127, "log": -128}[op], r)
    if op in ("rsqrt", "log"):
        r = np.where(v < 0, 0, r)
    return np.clip(r, -128, 127).astype(np.int8)


def _float_special(op, x):
    """(mask, value) of the float inputs whose result needs no search: NaN, the infinities and the zeros."""
    nan, inf, zero = np.isnan(x), np.isinf(x), x == 0
    pos_inf, neg_inf = {"tanh": (1.0, -1.0), "exp": (np.inf, 0.0), "rsqrt": (0.0, np.nan), "log": (np.inf, np.nan)}[op]
    at_zero = {"tanh": np.copysign(0.0, x), "exp": 1.0, "rsqrt": np.copysign(np.inf, x), "log": -np.inf}[op]
    val = np.where(nan, np.nan, np.where(inf, np.where(x > 0, pos_inf, neg_inf), np.where(zero, at_zero, 0.0)))
    return nan | inf | zero, val


def transcendental(op, t, a):
    """tanh, exp, rsqrt, log of a lane vector of type i8, f16 or f32, correctly rounded."""
    a = np.asarray(a, dtype=DTYPES[t])
    if t == "i8":
        return _i8_transcendental(op, a)
    shape = a.shape
    with np.errstate(all="ignore"):
        x = a.astype(np.float64).reshape(-1)
    special, sval = _float_special(op, x)
    with np.errstate(all="ignore"):
        y = _libm(op, x)
        out = y.astype(a.dtype)
        r = out.astype(np.float64)
        up = np.nextafter(out, a.dtype.type(np.inf)).astype(np.float64)
        dn = np.nextafter(out, a.dtype.type(-np.inf)).astype(np.float64)
        tol = np.abs(y) * 2.0 ** -40
        decided = np.isfinite(y) & np.isfinite(r) & (r != 0) & (y + tol < (r + up) / 2) & (y - tol > (r + dn) / 2)
        info = np.finfo(a.dtype)
        top = float(info.max)
        over = top + (top - float(np.nextafter(info.max, a.dtype.type(0)))) / 2    # rounds to infinity from here
        under = float(info.smallest_subnormal) / 2                                 # rounds to +0 below here
        decided |= np.isposinf(y) | (y - tol > over) | ((y >= 0) & (y + tol < under))
        if op in ("rsqrt", "log"):
            special |= x < 0                            # a negative finite input gives the canonical NaN
            sval = np.where(x < 0, np.nan, sval)
        if op == "log":
            special |= x == 1                           # log(1) = +0 exactly
            sval = np.where(x == 1, 0.0, sval)
        out[special] = sval[special].astype(a.dtype)
    out = canon(out)
    todo = np.flatnonzero(~(special | decided))
    STATS["tr_total"] += x.size
    STATS["tr_oracle"] += todo.size
    if todo.size:
        on = oracle()
        fmt = on.F16 if t == "f16" else on.F32
        code = {"tanh": on.TANH, "exp": on.EXP, "rsqrt": on.RSQRT, "log": on.LOG}[op]
        xb, ob = bits(a.reshape(-1)), bits(out)
        for i in todo:
            ob[i] = on.fp_transcendental(code, fmt, int(xb[i]))
    return out.reshape(shape)


# -- the MXM f16 dot product -------------------------------------------------------------------------------------

def _decompose_f16(h):
    """(signed significand, exponent) of f16 values with value = significand * 2**exponent; finite inputs only."""
    b = bits(h).astype(np.int64)
    e = (b >> 10) & 31
    m = b & 1023
    mant = np.where(e == 0, m, m + 1024)
    exp = np.where(e == 0, -24, e - 25)
    return np.where(b >> 15, -mant, mant), exp


def _round_f32(n, scale):
    """The float32 nearest (ties to even) to the Python integer n times 2**scale (overflow gives infinity)."""
    if n == 0:
        return np.float32(0.0)
    neg, n = n < 0, abs(n)
    lead = n.bit_length() - 1 + scale                 # exponent of the leading bit
    quantum = max(lead - 23, -149)
    drop = quantum - scale
    if drop <= 0:
        m = n << -drop
    else:
        m, rem = n >> drop, n & ((1 << drop) - 1)
        half = 1 << (drop - 1)
        m += rem > half or (rem == half and m & 1)
    v = np.float32(np.inf) if m.bit_length() + quantum > 128 else np.float32(np.ldexp(float(m), quantum))
    return -v if neg else v


def _dot_row(w, x):
    """One row of w dotted with x, exactly: Python integers, rounded once to f32 (the oracle's rule for specials)."""
    with np.errstate(all="ignore"):
        p = w.astype(np.float32) * x.astype(np.float32)      # exact: two 11-bit significands
    if np.isnan(p).any():
        return np.float32(np.nan)
    if np.isinf(p).any():
        signs = np.signbit(p[np.isinf(p)])
        return np.float32(np.nan) if signs.any() and not signs.all() else np.float32(-np.inf if signs[0] else np.inf)
    mw, ew = _decompose_f16(w)
    mx, ex = _decompose_f16(x)
    prod = (mw * mx).astype(object) << (ew + ex + 48).astype(object)
    return _round_f32(int(prod.sum()), -48)


def _quantum_exp(v):
    """Exponent of the lowest bit of each f16 value (value = integer * 2**q); 4096 for zero."""
    return np.where(v == 0, 4096, np.maximum(np.frexp(v)[1] - 11, -24))


def dot_f16(w, x, lanes=None):
    """Rows of w (R x K, f16) dotted with x (..., K, f16): the exact sum of the exact products, rounded once to f32;
    the result is (..., R). A NaN operand, an infinity times zero, or products of both infinities give the canonical
    NaN, otherwise an infinite product gives that infinity. An exact zero sum is -0 only when every one of the unit's
    `lanes` products is -0 (default K; the lanes beyond K carry +0 products). An f64 sum decides every element whose
    result is clear of a rounding boundary; the others are summed as integers."""
    w = np.atleast_2d(np.asarray(w, dtype=np.float16))
    x = np.asarray(x, dtype=np.float16)
    rows, k = w.shape
    lanes = k if lanes is None else lanes
    assert x.shape[-1] == k and k <= lanes
    lead = x.shape[:-1]
    x2 = x.reshape(-1, k)
    wf, xf = w.astype(np.float64), x2.astype(np.float64)
    with np.errstate(all="ignore"):
        prod = wf[None] * xf[:, None]                          # (B, R, K), exact
        s, a = prod.sum(axis=-1), np.abs(prod).sum(axis=-1)
        bound = a * ((k + 2) * 2.0 ** -52)
        r32 = s.astype(np.float32)
        r64 = r32.astype(np.float64)
        up = np.nextafter(r32, np.float32(np.inf)).astype(np.float64)
        dn = np.nextafter(r32, np.float32(-np.inf)).astype(np.float64)
        finite = np.isfinite(w).all(axis=1)[None, :] & np.isfinite(x2).all(axis=1)[:, None]
        # the f64 sum is exact when every partial sum is a multiple of the smallest product quantum q and below
        # 2**53 q; an exact value or tie then needs no margin
        qexp = (_quantum_exp(wf)[None] + _quantum_exp(xf)[:, None]).min(axis=-1)
        exact = a < np.ldexp(1.0, np.minimum(qexp + 53, 1000))
        clear = (s + bound < (r64 + up) / 2) & (s - bound > (r64 + dn) / 2)
        decided = finite & (np.abs(s) < 2.0 ** 126) & (exact | clear)
        zero = finite & (a == 0)
        negative_zero = (np.signbit(w)[None] ^ np.signbit(x2)[:, None]).all(axis=-1) & (k == lanes)
    out = np.where(zero, np.where(negative_zero, np.float32(-0.0), np.float32(0.0)), np.where(decided, r32, 0))
    out = out.astype(np.float32)
    for b, r in np.argwhere(~(decided | zero)):
        out[b, r] = _dot_row(w[r], x2[b])
    return canon(out).reshape(lead + (rows,))
