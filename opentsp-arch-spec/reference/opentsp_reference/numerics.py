"""Element kernels of the oracle (microarchitecture.md sec. 5 and 6, isa.md VXM/MXM rows).

Every value is a raw bit pattern (Python int) of the element width. Floats are
IEEE-754 binary16 / binary32 with round-to-nearest-even, computed exactly with
fractions; transcendentals are correctly rounded with mpmath at increasing precision.
"""
from fractions import Fraction
import math

try:
    import mpmath
except ImportError as e:  # fail at import, not on the first transcendental
    raise ImportError("the numerical reference requires mpmath. Install the specification tools described in docs/installation.md") from e

TYPES = ['i8', 'f16', 'i32', 'f32', 'u8', 'i16', 'u16', 'u32', 'b8', 'b16', 'b32', 'e4m3', 'e5m2']
# The profile extension types (VXM type code bit 4): bf16 is code 16 and exists only in a profile that declares it.
EXT_TYPES = ('bf16',)
CODE = {**{t: i for i, t in enumerate(TYPES)}, 'bf16': 16}
TYPE_NAME = {c: t for t, c in CODE.items()}
WIDTH = dict(i8=1, f16=2, i32=4, f32=4, u8=1, i16=2, u16=2, u32=4, b8=1, b16=2, b32=4, e4m3=1, e5m2=1, bf16=2)
FP8 = ('e4m3', 'e5m2')
FLOATS = ('f16', 'f32', 'bf16') + FP8
BOOLS = ('b8', 'b16', 'b32')
UNSIGNED = ('u8', 'u16', 'u32')


def nbits(t):
    return 8 * WIDTH[t]


def ones(t):
    return (1 << nbits(t)) - 1


def is_float(t):
    return t in FLOATS


def is_bool(t):
    return t in BOOLS


def is_int(t):
    return not is_float(t) and not is_bool(t)


def is_signed_int(t):
    return is_int(t) and t not in UNSIGNED


# ---------------------------------------------------------------- integers

def sval(x, n):
    x &= (1 << n) - 1
    return x - (1 << n) if x >> (n - 1) else x


def ival(t, x):
    """Integer value of raw x as type t."""
    return x & ones(t) if t in UNSIGNED else sval(x, nbits(t))


def irange(t):
    n = nbits(t)
    return (0, (1 << n) - 1) if t in UNSIGNED else (-(1 << (n - 1)), (1 << (n - 1)) - 1)


def sat(t, v):
    lo, hi = irange(t)
    return lo if v < lo else hi if v > hi else v


def raw(t, v):
    return v & ones(t)


# ---------------------------------------------------------------- IEEE floats

class Fmt:
    """A binary float format. ieee=False is e4m3: no infinity, the all-ones pattern is the only NaN,
    the top exponent code is an ordinary one and an overflow saturates to the largest finite value."""
    def __init__(self, width, ebits, mbits, ieee=True):
        self.width, self.ebits, self.mbits, self.ieee = width, ebits, mbits, ieee
        self.bias = (1 << (ebits - 1)) - 1
        self.emin = 1 - self.bias
        self.sign = 1 << (width - 1)
        self.expmask = ((1 << ebits) - 1) << mbits
        self.cnan = (((1 << ebits) - 1) << mbits) | (1 << (mbits - 1)) if ieee else (1 << (width - 1)) - 1
        self.maxfin = (self.expmask | ((1 << mbits) - 2)) if not ieee else None

    def inf(self, neg):
        """The infinity; for e4m3 the saturation value (+-448)."""
        if not self.ieee:
            return (self.sign if neg else 0) | self.maxfin
        return (self.sign if neg else 0) | self.expmask

    def zero(self, neg):
        return self.sign if neg else 0


F16 = Fmt(16, 5, 10)
F32 = Fmt(32, 8, 23)
E5M2 = Fmt(8, 5, 2)
E4M3 = Fmt(8, 4, 3, ieee=False)
BF16 = Fmt(16, 8, 7)


def fmt(t):
    return {'f16': F16, 'bf16': BF16, 'e4m3': E4M3, 'e5m2': E5M2}.get(t, F32)


def fclass(f, x):
    """('nan'|'inf'|'zero'|'fin', neg, Fraction magnitude or None)."""
    x &= (1 << f.width) - 1
    neg = bool(x & f.sign)
    e = (x >> f.mbits) & ((1 << f.ebits) - 1)
    m = x & ((1 << f.mbits) - 1)
    if f.ieee and e == (1 << f.ebits) - 1:
        return ('inf' if m == 0 else 'nan', neg, None)
    if not f.ieee and e == (1 << f.ebits) - 1 and m == (1 << f.mbits) - 1:
        return ('nan', neg, None)
    if e == 0 and m == 0:
        return ('zero', neg, Fraction(0))
    if e == 0:
        return ('fin', neg, Fraction(m) * Fraction(2) ** (f.emin - f.mbits))
    return ('fin', neg, Fraction(m + (1 << f.mbits)) * Fraction(2) ** (e - f.bias - f.mbits))


def is_nan(f, x):
    return fclass(f, x)[0] == 'nan'


def rnd_int(q):
    """Round a non-negative Fraction to an integer, ties to even."""
    fl = q.numerator // q.denominator
    r = q - fl
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and fl & 1):
        fl += 1
    return fl


def fround(f, neg, mag):
    """Round neg * mag (mag a non-negative Fraction) to format f, RNE; overflow -> inf."""
    if mag == 0:
        return f.zero(neg)
    # exponent of the leading bit
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    if Fraction(2) ** e > mag:
        e -= 1
    if e < f.emin:
        e = f.emin
    q = e - f.mbits                 # quantum exponent
    m = rnd_int(mag / Fraction(2) ** q)
    if m == 1 << (f.mbits + 1):
        m >>= 1
        e += 1
    s = f.sign if neg else 0
    if m == 0:
        return f.zero(neg)
    if m < (1 << f.mbits):          # subnormal
        return s | m
    if f.ieee and e + f.bias >= (1 << f.ebits) - 1:
        return f.inf(neg)
    if not f.ieee and (e + f.bias, m - (1 << f.mbits)) >= ((1 << f.ebits) - 1, (1 << f.mbits) - 1):
        return f.inf(neg)           # e4m3: saturate; the all-ones fraction of the top exponent is NaN
    return s | ((e + f.bias) << f.mbits) | (m - (1 << f.mbits))


def fp_add(f, x, y, negate_y=False):
    ca, na, ma = fclass(f, x)
    cb, nb, mb = fclass(f, y)
    if negate_y:
        nb = not nb
    if ca == 'nan' or cb == 'nan':
        return f.cnan
    if ca == 'inf' and cb == 'inf':
        return f.inf(na) if na == nb else f.cnan
    if ca == 'inf':
        return f.inf(na)
    if cb == 'inf':
        return f.inf(nb)
    if ca == 'zero' and cb == 'zero':
        return f.zero(na and nb)
    s = (-ma if na else ma) + (-mb if nb else mb)
    if s == 0:
        return f.zero(False)
    return fround(f, s < 0, abs(s))


def fp_mul(f, x, y):
    ca, na, ma = fclass(f, x)
    cb, nb, mb = fclass(f, y)
    neg = na != nb
    if ca == 'nan' or cb == 'nan':
        return f.cnan
    if (ca == 'inf' and cb == 'zero') or (ca == 'zero' and cb == 'inf'):
        return f.cnan
    if ca == 'inf' or cb == 'inf':
        return f.inf(neg)
    if ca == 'zero' or cb == 'zero':
        return f.zero(neg)
    return fround(f, neg, ma * mb)


def fvalue(f, x):
    """Signed comparable value: Fraction, or +-inf as float; zero signs collapse."""
    c, n, m = fclass(f, x)
    if c == 'inf':
        return -math.inf if n else math.inf
    return -m if n else m


def fcmp(f, x, y):
    a, b = fvalue(f, x), fvalue(f, y)
    return (a > b) - (a < b)


def fp_max(f, x, y):
    nx, ny = is_nan(f, x), is_nan(f, y)
    if nx and ny:
        return f.cnan
    if nx:
        return y
    if ny:
        return x
    c = fcmp(f, x, y)
    if c > 0:
        return x
    if c < 0:
        return y
    return y if x & f.sign else x       # max(-0, +0) = +0


def fp_min(f, x, y):
    nx, ny = is_nan(f, x), is_nan(f, y)
    if nx and ny:
        return f.cnan
    if nx:
        return y
    if ny:
        return x
    c = fcmp(f, x, y)
    if c < 0:
        return x
    if c > 0:
        return y
    return x if x & f.sign else y       # min(-0, +0) = -0


def fp_convert(src, dst, x):
    c, n, m = fclass(src, x)
    if c == 'nan':
        return dst.cnan
    if c == 'inf':
        return dst.inf(n)
    if c == 'zero':
        return dst.zero(n)
    return fround(dst, n, m)


def fp_of_int(f, v):
    return fround(f, v < 0, Fraction(abs(v)))


def int_of_fp(f, x, bits):
    """Round to nearest integer (ties to even), saturate to a signed bits-wide range; NaN -> 0."""
    hi, lo = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    c, n, m = fclass(f, x)
    if c in ('nan', 'zero'):
        return 0
    if c == 'inf':
        return lo if n else hi
    r = rnd_int(m)
    s = -r if n else r
    return hi if s > hi else lo if s < lo else s


# ---------------------------------------------------------------- transcendentals

TANH, EXP, RSQRT, LOG = 21, 22, 23, 24


def _exact_or_interval(op, x, prec):
    """Exact Fraction result if one exists, else (lo, hi) Fractions bracketing it."""
    if op == RSQRT:
        # 1/sqrt(x) is rational iff x is the square of a rational
        n, d = x.numerator, x.denominator
        rn, rd = math.isqrt(n), math.isqrt(d)
        if rn * rn == n and rd * rd == d:
            return Fraction(rd, rn)
    mpmath.mp.prec = prec
    xm = mpmath.mpf(x.numerator) / x.denominator
    v = {TANH: mpmath.tanh, EXP: mpmath.exp, RSQRT: lambda z: 1 / mpmath.sqrt(z), LOG: mpmath.log}[op](xm)
    sign, man, exp, _ = v._mpf_          # man_exp drops the sign
    fv = Fraction(int(man)) * Fraction(2) ** int(exp) if man else Fraction(0)
    if sign:
        fv = -fv
    err = abs(fv) * Fraction(1, 2 ** (prec - 12)) + Fraction(1, 2 ** (prec * 4))
    return (fv - err, fv + err)


def _correct(op, x, rounder):
    prec = 96
    while prec <= 8192:
        r = _exact_or_interval(op, x, prec)
        if isinstance(r, Fraction):
            return rounder(r)
        a, b = rounder(r[0]), rounder(r[1])
        if a == b:
            return a
        prec *= 2
    raise RuntimeError('transcendental did not converge')


def fp_transcendental(op, f, x):
    c, neg, m = fclass(f, x)
    if c == 'nan':
        return f.cnan
    if c == 'inf':
        if op == TANH:
            return fp_of_int(f, -1 if neg else 1)
        if op == EXP:
            return f.zero(False) if neg else f.inf(False)
        return f.cnan if neg else (f.zero(False) if op == RSQRT else f.inf(False))
    if c == 'zero':
        if op == TANH:
            return f.zero(neg)      # the canonical signed zero, no stray bits
        if op == EXP:
            return fp_of_int(f, 1)
        if op == RSQRT:
            return f.inf(neg)
        return f.inf(True)
    if op in (RSQRT, LOG) and neg:
        return f.cnan
    if op == LOG and m == 1:
        return f.zero(False)
    xv = -m if neg else m
    if op == EXP and abs(xv) >= 512:
        return f.zero(False) if neg else f.inf(False)
    if op == TANH and abs(xv) >= 64:
        return fp_of_int(f, -1 if neg else 1)
    return _correct(op, xv, lambda v: fround(f, v < 0, abs(v)))


def i8_transcendental(op, x):
    v = sval(x, 8)
    if v == 0:
        r = {EXP: 1, RSQRT: 127, LOG: -128}.get(op, 0)
    elif op in (RSQRT, LOG) and v < 0:
        r = 0
    elif op == EXP and v > 5:
        r = 127
    elif op == EXP and v < -1:
        r = 0
    else:
        def rint(q):
            return -rnd_int(-q) if q < 0 else rnd_int(q)
        r = _correct(op, Fraction(v), rint)
    return sat('i8', r) & 0xff


# ---------------------------------------------------------------- VXM lane kernels

BINOPS = ['add', 'add_sat', 'add_mod', 'sub', 'sub_sat', 'sub_mod', 'mul', 'mul_sat', 'mul_mod',
          'max', 'min', 'and', 'or', 'xor']


def k_binary(op, t, x, y):
    """op: binary opcode 0..13 (encoding.md VXM table)."""
    if is_bool(t):
        a, b = (x & ones(t)) != 0, (y & ones(t)) != 0
        r = {11: a and b, 12: a or b, 13: a != b}.get(op)
        return ones(t) if r else 0
    if is_float(t):
        f = fmt(t)
        if op == 0:
            return fp_add(f, x, y)
        if op == 3:
            return fp_add(f, x, y, True)
        if op == 6:
            return fp_mul(f, x, y)
        if op == 9:
            return fp_max(f, x, y)
        if op == 10:
            return fp_min(f, x, y)
        return 0
    a, b = ival(t, x), ival(t, y)
    if op in (0, 2):
        return raw(t, a + b)
    if op == 1:
        return raw(t, sat(t, a + b))
    if op in (3, 5):
        return raw(t, a - b)
    if op == 4:
        return raw(t, sat(t, a - b))
    if op in (6, 8):
        return raw(t, a * b)
    if op == 7:
        return raw(t, sat(t, a * b))
    if op == 9:
        return raw(t, max(a, b))
    if op == 10:
        return raw(t, min(a, b))
    if op == 11:
        return x & y & ones(t)
    if op == 12:
        return (x | y) & ones(t)
    if op == 13:
        return (x ^ y) & ones(t)
    return 0


def k_unary(op, t, x):
    """op: 14 not, 15 neg, 16 abs, 20 relu, 21..24 transcendentals."""
    if 21 <= op <= 24:
        return fp_transcendental(op, fmt(t), x) if is_float(t) else i8_transcendental(op, x)
    if is_float(t):
        f = fmt(t)
        c, n, _ = fclass(f, x)
        if op in (15, 16, 20) and c == 'nan':
            return f.cnan
        if op == 15:
            return (x ^ f.sign) & ((1 << f.width) - 1)
        if op == 16:
            return x & ~f.sign & ((1 << f.width) - 1)
        if op == 20:
            return f.zero(False) if (n or c == 'zero') else x & ((1 << f.width) - 1)
        return 0
    if op == 14:
        if is_bool(t):
            return ones(t) if (x & ones(t)) == 0 else 0
        return ~x & ones(t)
    if is_bool(t):          # only Not is defined on a boolean
        return 0
    a = ival(t, x)          # signed or unsigned per the type
    if op == 15:
        return raw(t, -a)
    if op == 16:
        return raw(t, abs(a))
    if op == 20:
        return raw(t, max(a, 0))
    return 0


CMP_MODES = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']


def k_cmp(mode, t, x, y):
    if is_bool(t):
        a, b = (x & ones(t)) != 0, (y & ones(t)) != 0
        r = (a == b) if mode == 0 else (a != b)
    elif is_float(t):
        f = fmt(t)
        if is_nan(f, x) or is_nan(f, y):
            r = mode == 1
        else:
            c = fcmp(f, x, y)
            r = [c == 0, c != 0, c < 0, c <= 0, c > 0, c >= 0][mode]
    else:
        a, b = ival(t, x), ival(t, y)
        r = [a == b, a != b, a < b, a <= b, a > b, a >= b][mode]
    return ones(t) if r else 0


def nonzero(t, x):
    return (x & ones(t)) != 0


# fixed conversions 25..32: (source type, destination type)
CONV = {25: ('i8', 'f32'), 26: ('f32', 'i8'), 27: ('i8', 'f16'), 28: ('f16', 'i8'),
        29: ('i8', 'i32'), 30: ('i32', 'i8'), 31: ('f16', 'f32'), 32: ('f32', 'f16')}


def k_convert(op, x):
    st, dt = CONV[op]
    return k_convert_any(st, dt, x)


def k_convert_any(st, dt, x):
    """Integers saturate, floats round to nearest even, bool = value != 0 (NaN is true)."""
    if is_float(st):
        truth = fclass(fmt(st), x)[0] != 'zero'
    else:
        truth = (x & ones(st)) != 0
    if is_bool(dt):
        return ones(dt) if truth else 0
    if is_float(dt):
        if is_float(st):
            return fp_convert(fmt(st), fmt(dt), x)
        return fp_of_int(fmt(dt), (1 if truth else 0) if is_bool(st) else ival(st, x))
    if is_bool(st):
        v = 1 if truth else 0
    elif is_float(st):
        if dt in UNSIGNED:
            v = max(int_of_fp(fmt(st), x, nbits(dt) + 1), 0)
        else:
            v = int_of_fp(fmt(st), x, nbits(dt))
    else:
        v = ival(st, x)
    return raw(dt, sat(dt, v))


# ---------------------------------------------------------------- MXM dot products

def f16_to_f32(x):
    return fp_convert(F16, F32, x)


def dot_fp8(f, ws, xs):
    """ABC with fp8 weights and activations of format f: as dot_f16, the exact sum of the exact products,
    rounded once to f32. e4m3 has no infinity."""
    total = Fraction(0)
    inf_signs = set()
    all_negzero = True
    for w, x in zip(ws, xs):
        cw, nw, mw = fclass(f, w)
        cx, nx_, mx = fclass(f, x)
        neg = nw != nx_
        if cw == 'nan' or cx == 'nan' or {cw, cx} == {'inf', 'zero'}:
            return F32.cnan
        if 'inf' in (cw, cx):
            inf_signs.add(neg)
            all_negzero = False
        elif 'zero' in (cw, cx):
            if not neg:
                all_negzero = False
        else:
            all_negzero = False
            total += -(mw * mx) if neg else mw * mx
    if inf_signs:
        return F32.cnan if len(inf_signs) == 2 else F32.inf(inf_signs.pop())
    if total == 0:
        return F32.zero(all_negzero)
    return fround(F32, total < 0, abs(total))


def dot_f16(ws, xs):
    """microarchitecture.md 6 (ABC f16): the exact sum of the exact products, rounded once to
    f32 (RNE). Any NaN operand, Inf x 0, or +Inf and -Inf products give the canonical NaN;
    otherwise an infinite product gives that infinity. An exact zero sum is -0 only if
    every product is -0. Overflow rounds to infinity."""
    total = Fraction(0)
    inf_signs = set()
    all_negzero = True
    for w, x in zip(ws, xs):
        p = fp_mul(F32, f16_to_f32(w), f16_to_f32(x))  # exact: f16 x f16 fits f32
        c, n, m = fclass(F32, p)
        if c == 'nan':
            return F32.cnan
        if c == 'inf':
            inf_signs.add(n)
            all_negzero = False
        elif c == 'fin':
            all_negzero = False
            total += -m if n else m
        elif not n:
            all_negzero = False
    if inf_signs:
        return F32.cnan if len(inf_signs) == 2 else F32.inf(inf_signs.pop())
    if total == 0:
        return F32.zero(all_negzero)
    return fround(F32, total < 0, abs(total))


# ---- the wide accumulator and block-scaled fp8 (extensions wide_acc and block_fp8; microarchitecture.md sec. 6) ----
# An exact dot product is (special, units, negzero): special None, 'nan', 'pinf' or 'ninf'; units the exact sum of the finite
# products in units of 2^-48; negzero: every product is -0 (a zero sum is then -0).
UNIT = Fraction(1, 1 << 48)


def _exact(pairs):
    """pairs: (fclass of the weight, fclass of the activation, extra power of two) for every lane."""
    total = Fraction(0)
    inf_signs = set()
    nan = False
    negzero = True
    for (cw, nw, mw), (cx, nx_, mx), sh in pairs:
        neg = nw != nx_
        if cw == 'nan' or cx == 'nan' or {cw, cx} == {'inf', 'zero'}:
            nan = True
        elif 'inf' in (cw, cx):
            inf_signs.add(neg)
            negzero = False
        elif 'zero' in (cw, cx):
            if not neg:
                negzero = False
        else:
            negzero = False
            p = mw * mx * Fraction(2) ** sh
            total += -p if neg else p
    units = total / UNIT
    assert units.denominator == 1
    special = 'nan' if nan or len(inf_signs) == 2 else ('pinf' if inf_signs == {False} else ('ninf' if inf_signs == {True} else None))
    return (special, int(units), negzero)


def exact_f16(ws, xs):
    return _exact((fclass(F16, w), fclass(F16, x), 0) for w, x in zip(ws, xs))


def exact_fp8(f, ws, xs):
    return _exact((fclass(f, w), fclass(f, x), 0) for w, x in zip(ws, xs))


def block_exp(scales, k):
    """The exponent of the 16-lane block of lane k: the low four bits of the byte at the block's first lane, signed."""
    n = scales[16 * (k // 16)] & 15
    return n - 16 if n >= 8 else n


def exact_bfp8(wd, wsc, xd, xsc):
    return _exact((fclass(E4M3, wd[k]), fclass(E4M3, xd[k]), block_exp(wsc, k) + block_exp(xsc, k)) for k in range(len(wd)))


def round_exact(d):
    """The f32 product of an exact dot product: its sum rounded once."""
    special, units, negzero = d
    if special == 'nan':
        return F32.cnan
    if special:
        return F32.inf(special == 'ninf')
    if units == 0:
        return F32.zero(negzero)
    return fround(F32, units < 0, abs(Fraction(units)) * UNIT)


def wide_of(d):
    """A lane of the wide accumulator: (special, units, negzero) with the zero flag only for a zero lane."""
    special, units, negzero = d
    return (special, 0, False) if special else (None, units, units == 0 and negzero)


def _wrap128(v):
    v %= 1 << 128
    return v - (1 << 128) if v >= 1 << 127 else v


def wide_add(a, b):
    sa, ua, za = a
    sb, ub, zb = b
    if sa == 'nan' or sb == 'nan' or {sa, sb} == {'pinf', 'ninf'}:
        return ('nan', 0, False)
    if sa:
        return a
    if sb:
        return b
    return (None, _wrap128(ua + ub), za and zb)


def wide_read(w):
    """The read-out of a wide lane: rounded once to f32."""
    return round_exact((w[0], w[1], w[2]))


WIDE_ZERO = (None, 0, False)


def dot_bf16(ws, xs):
    """ABC with bf16 weights and activations (a profile extension): the exact sum of the exact products rounded once to f32,
    as dot_fp8 does for its format."""
    return dot_fp8(BF16, ws, xs)
