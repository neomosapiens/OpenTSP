#!/usr/bin/env python3
"""The numpy twin opentsp_tools.programs.numerics against the Python oracle (opentsp-arch-spec/test/oracle/numerics.py)
( sec. 3.3).

Coverage: every f16 and i8 unary input, every i8 pair, 10**5 pairs per binary operation (floats: random bit
patterns, near-equal and opposite operands, special values), every i8 and f16 transcendental input and 20000 f32
bit patterns per transcendental, conversions, Reduce folds, and f16 dot products with cancellation, infinities
and signed zeros. The oracle is evaluated in a process pool; a run takes about a minute.

usage: python3 tests/programs/gen/test_numerics.py
"""
import multiprocessing
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs.data import DTYPES

from opentsp_reference import numerics as on  # noqa: E402

PAIRS = 100000
FLOAT_OPS = {"add": 0, "sub": 3, "mul": 6, "max": 9, "min": 10}
SPECIAL = {
    "f16": [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00,
            0x7C01, 0xFE00, 0x3555, 0x4248],
    "f32": [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000,
            0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFC00000, 0x4B800000, 0x33800000],
}
_pool = None


def pool():
    global _pool
    if _pool is None:
        _pool = multiprocessing.get_context("fork").Pool(min(os.cpu_count() or 1, 12))
    return _pool


def tearDownModule():
    if _pool is not None:
        _pool.close()
        _pool.join()


def _run(fn, consts, cols):
    return [fn(*consts, *row) for row in zip(*cols)]


def oracle_map(fn, consts, *cols):
    """[fn(*consts, *row) for row in zip(*cols)], evaluated in the pool in contiguous chunks; fn is a module-level
    function and the columns are lists."""
    size = -(-len(cols[0]) // 48)
    jobs = [(fn, consts, [c[i:i + size] for c in cols]) for i in range(0, len(cols[0]), size)]
    return [v for part in pool().starmap(_run, jobs) for v in part]


def ints(a):
    return [int(v) for v in nx.bits(np.asarray(a))]


def raw_array(t, vals):
    return nx.from_bits(np.array(vals, dtype=np.uint64), t)


def float_inputs(rng, t, n):
    """Random bit patterns, near-equal and opposite pairs (cancellation, ties), and the special values crossed."""
    width = 16 if t == "f16" else 32
    top = 1 << width
    third = n // 3
    a = rng.integers(0, top, size=n, dtype=np.uint64)
    b = rng.integers(0, top, size=n, dtype=np.uint64)
    near = a[:third] + rng.integers(-3, 4, size=third).astype(np.uint64)
    b[:third] = near % np.uint64(top)
    b[third:2 * third] = a[third:2 * third] ^ np.uint64(top >> 1)
    sp = np.array(SPECIAL[t], dtype=np.uint64)
    cross = np.stack(np.meshgrid(sp, sp), -1).reshape(-1, 2)
    a[:len(cross)], b[:len(cross)] = cross[:, 0], cross[:, 1]
    # exponents close together make the cancellation cases frequent: pin the exponent field of half of the rest
    m = (1 << (10 if t == "f16" else 23)) - 1
    lo = 2 * third
    a[lo:] = (a[lo:] & np.uint64(m)) | (b[lo:] & np.uint64(top - 1 - m))
    return raw_array(t, a), raw_array(t, b)


def same(got, want_bits, label):
    g = ints(got)
    bad = [i for i, (x, y) in enumerate(zip(g, want_bits)) if x != y]
    if bad:
        raise AssertionError(f"{label}: {len(bad)} mismatches, first at {bad[0]}: twin {g[bad[0]]:#x}, "
                             f"oracle {want_bits[bad[0]]:#x}")


class Binary(unittest.TestCase):
    def test_float_ops(self):
        rng = np.random.default_rng(1)
        for t in ("f16", "f32"):
            a, b = float_inputs(rng, t, PAIRS)
            for op, code in FLOAT_OPS.items():
                want = oracle_map(on.k_binary, (code, t), ints(a), ints(b))
                same(nx.binary(op, t, a, b), want, f"{op}.{t}")

    def test_integer_ops(self):
        rng = np.random.default_rng(2)
        for t in ("i8", "u8", "i16", "u16", "i32", "u32"):
            info = np.iinfo(DTYPES[t])
            if t in ("i8", "u8"):
                a = np.repeat(np.arange(info.min, info.max + 1), 256).astype(DTYPES[t])
                b = np.tile(np.arange(info.min, info.max + 1), 256).astype(DTYPES[t])
            else:
                edge = np.array([info.min, info.min + 1, -1 if info.min < 0 else 0, 0, 1, info.max - 1, info.max],
                                dtype=np.int64)
                a = rng.integers(info.min, info.max, size=PAIRS, endpoint=True).astype(DTYPES[t])
                b = rng.integers(info.min, info.max, size=PAIRS, endpoint=True).astype(DTYPES[t])
                cross = np.stack(np.meshgrid(edge, edge), -1).reshape(-1, 2)
                a[:len(cross)], b[:len(cross)] = cross[:, 0], cross[:, 1]
            for code, op in enumerate(on.BINOPS):
                want = oracle_map(on.k_binary, (code, t), ints(a), ints(b))
                same(nx.binary(op, t, a, b), want, f"{op}.{t}")


class Unary(unittest.TestCase):
    def test_every_f16_input(self):
        a = raw_array("f16", np.arange(1 << 16))
        for op, code in (("neg", 15), ("abs", 16), ("relu", 20)):
            same(nx.unary(op, "f16", a), [on.k_unary(code, "f16", x) for x in range(1 << 16)], f"{op}.f16")

    def test_signed_integers(self):
        rng = np.random.default_rng(3)
        for t in ("i8", "i16", "i32"):
            info = np.iinfo(DTYPES[t])
            a = (np.arange(info.min, info.max + 1) if t != "i32" else
                 np.concatenate([rng.integers(info.min, info.max, size=50000, endpoint=True),
                                 [info.min, info.max, 0, -1, 1]])).astype(DTYPES[t])
            for op, code in (("neg", 15), ("abs", 16), ("relu", 20)):
                same(nx.unary(op, t, a), [on.k_unary(code, t, x) for x in ints(a)], f"{op}.{t}")
        a = np.arange(256).astype(np.uint8)
        same(nx.unary("not", "u8", a), [on.k_unary(14, "u8", x) for x in range(256)], "not.u8")

    def test_compare_and_select(self):
        rng = np.random.default_rng(4)
        for t in ("f16", "f32", "i8", "i32", "u16"):
            if t[0] == "f":
                a, b = float_inputs(rng, t, 20000)
            else:
                info = np.iinfo(DTYPES[t])
                a = rng.integers(info.min, info.max, size=20000, endpoint=True).astype(DTYPES[t])
                b = np.where(rng.random(20000) < 0.3, a, rng.integers(info.min, info.max, size=20000,
                                                                      endpoint=True)).astype(DTYPES[t])
            for k, mode in enumerate(on.CMP_MODES):
                want = oracle_map(on.k_cmp, (k, t), ints(a), ints(b))
                same(nx.cmp(mode, t, a, b), want, f"cmp.{mode}.{t}")
        # a mask element is true when any bit is set: -0.0 and NaN are true
        m = raw_array("f32", [0, 0x80000000, 0x7FC00000, 0x3F800000])
        got = nx.mask("f32", raw_array("f32", [0x40000000] * 4), m)
        self.assertEqual(ints(got), [0, 0x40000000, 0x40000000, 0x40000000])


class Convert(unittest.TestCase):
    PAIRS = [("f16", "f32"), ("f32", "f16"), ("f16", "i8"), ("f32", "i8"), ("i8", "f16"), ("i8", "f32"),
             ("i8", "i32"), ("i32", "i8"), ("i16", "f16"), ("u16", "f16"), ("i32", "f32"), ("u32", "f32"),
             ("f16", "u8"), ("f32", "i32"), ("f32", "u16"), ("i8", "u8"), ("u8", "i8"), ("f16", "b8"),
             ("f32", "b32"), ("i16", "b16"), ("i8", "i16")]

    def inputs(self, rng, st):
        if st == "f16":
            return raw_array("f16", np.arange(1 << 16))
        if st[0] == "f":
            a, _ = float_inputs(rng, st, 60000)
            # values around the integer rounding ties and the saturation limits
            tie = (rng.integers(-300, 300, size=20000) + 0.5).astype(np.float32)
            far = (rng.random(20000) * 2e9 - 1e9).astype(np.float32)
            return np.concatenate([a, tie, far])
        info = np.iinfo(DTYPES[st])
        if st in ("i8", "u8"):
            return np.arange(info.min, info.max + 1).astype(DTYPES[st])
        return rng.integers(info.min, info.max, size=60000, endpoint=True).astype(DTYPES[st])

    def test_pairs(self):
        rng = np.random.default_rng(5)
        for st, dt in self.PAIRS:
            a = self.inputs(rng, st)
            want = oracle_map(on.k_convert_any, (st, dt), ints(a))
            got = nx.convert(st, dt, a)
            if dt in nx.BOOL_WIDTH:
                got = got.astype(nx.UNSIGNED[nx.BOOL_WIDTH[dt]])
            same(got, want, f"convert {st}->{dt}")


def _fold(op, t, vals):
    code = {"add": 0, "max": 9, "min": 10}[op]
    acc = vals[0]
    for v in vals[1:]:
        acc = on.k_binary(code, t, acc, v)
    return acc


class Reduce(unittest.TestCase):
    def test_fold_order_and_layout(self):
        rng = np.random.default_rng(6)
        for t in ("f16", "f32", "i8", "i32"):
            for op in ("add", "max", "min"):
                if t[0] == "f":
                    a, _ = float_inputs(rng, t, 16 * 400)
                else:
                    info = np.iinfo(DTYPES[t])
                    a = rng.integers(info.min, info.max, size=16 * 400, endpoint=True).astype(DTYPES[t])
                got = nx.reduce(op, t, a)
                words = ints(a)
                want = []
                for g in range(400):
                    want += [_fold(op, t, words[16 * g:16 * g + 16])] + [0] * 15
                same(got, want, f"reduce.{op}.{t}")
        # the order is observable: 2**24 + 1 + 1 rounds twice when 2**24 is last, not when it is first
        v = np.zeros(16, np.float32)
        v[0], v[1], v[15] = 1.0, 1.0, 2.0 ** 24
        self.assertEqual(int(nx.bits(nx.reduce("add", "f32", v))[0]), 0x4B800001)


class ReduceBcast(unittest.TestCase):
    def test_tree_against_oracle_folds_and_totals_are_identical(self):
        rng = np.random.default_rng(10)
        for t in ("f16", "f32", "i8", "i32"):
            for op in ("add", "max", "min"):
                for trial in range(8):
                    if t[0] == "f":
                        a = nx.canon(raw_array(t, rng.integers(0, 1 << (16 if t == "f16" else 32), size=320))
                                     .reshape(-1)) if trial == 0 else \
                            (rng.standard_normal(320) * 2.0 ** rng.integers(-6, 6, size=320)).astype(DTYPES[t])
                    else:
                        info = np.iinfo(DTYPES[t])
                        a = rng.integers(info.min, info.max, size=320, endpoint=True).astype(DTYPES[t])
                    code = {"add": 0, "max": 9, "min": 10}[op]
                    words = ints(a)
                    tot = [_fold(op, t, words[16 * g:16 * g + 16]) for g in range(20)]
                    for rot in (2, 1):
                        tot = [on.k_binary(code, t, tot[g], tot[g // 4 * 4 + (g % 4 + rot) % 4]) for g in range(20)]
                    want = []
                    for g in range(20):
                        want += [tot[g]] + [0] * 15
                    got = nx.reduce_bcast(op, t, a)
                    same(got, want, f"reduce_bcast.{op}.{t}")
                    self.assertEqual({tot[g] for g in range(4)}, {tot[0]}, "totals differ inside a segment")


class Batched(unittest.TestCase):
    """The lane axis is the last axis: a leading axis is a batch of independent vectors."""

    def test_reduce_bcast_batch_equals_rows(self):
        rng = np.random.default_rng(11)
        a = (rng.standard_normal((5, 64)) * 8).astype(np.float32)
        for op in ("add", "max"):
            got = nx.reduce_bcast(op, "f32", a)
            self.assertEqual(got.shape, (5, 64))
            for i in range(5):
                self.assertEqual(ints(got[i]), ints(nx.reduce_bcast(op, "f32", a[i])))
            self.assertEqual(len({int(v) for v in ints(got[0][::16])}), 1, "totals differ inside the segment")

    def test_distribute_first_broadcasts_lane_zero_of_each_superlane(self):
        x = np.arange(64, dtype=np.float32).reshape(2, 32)
        got = nx.distribute_first(x)
        self.assertEqual(got.shape, x.shape)
        self.assertTrue((got[:, :16] == x[:, :1]).all() and (got[:, 16:] == x[:, 16:17]).all())

    def test_dot_batch_against_oracle(self):
        rng = np.random.default_rng(12)
        for k in (16, 64, 128):
            w = (rng.standard_normal((40, k)) * 0.5).astype(np.float16)
            x = (rng.standard_normal((6, k)) * 2).astype(np.float16)
            x[2, k // 2:] = 0
            got = nx.dot_f16(w, x)
            self.assertEqual(got.shape, (6, 40))
            want = [on.dot_f16([int(v) for v in nx.bits(w[r])], [int(v) for v in nx.bits(x[b])])
                    for b in range(6) for r in range(40)]
            same(got.reshape(-1), want, f"dot_f16 batch k={k}")

    def test_dot_cancellation_and_special_rows_take_the_exact_path(self):
        k = 64
        w = np.zeros((7, k), np.float16)
        x = np.ones(k, np.float16)
        w[0, :2] = (1024, -1024)
        w[0, 2] = 2.0 ** -14
        w[1, :3] = (65504, -65504, 1)
        w[2, :2] = 65504                                  # 131008: no f32 overflow
        w[3, 0] = 6e-8
        w[4, 0], w[5, 0] = np.inf, np.nan
        w[6, :2] = (np.inf, -np.inf)
        want = [on.dot_f16([int(v) for v in nx.bits(w[r])], [int(v) for v in nx.bits(x)]) for r in range(7)]
        same(nx.dot_f16(w, x), want, "dot_f16 cancellation")
        w = np.full((1, 128), 1.0, np.float16)            # the sequential f32 sum differs from the exact one
        w[0, 0], x = 2.0 ** 14, np.full(128, 2.0 ** -10, np.float16)
        x[0] = 2.0 ** 10
        same(nx.dot_f16(w, x), [on.dot_f16([int(v) for v in nx.bits(w[0])], [int(v) for v in nx.bits(x)])],
             "dot_f16 long sum")

    def test_negative_zero_needs_every_lane_of_the_unit(self):
        # microarchitecture.md sec. 6: an exact zero sum is -0 only when every product is -0; lanes beyond K are +0
        w = np.full((1, 320), -0.0, np.float16)
        x = np.zeros(320, np.float16)
        self.assertEqual(int(nx.bits(nx.dot_f16(w, x))[0]), 0x80000000)
        self.assertEqual(int(nx.bits(nx.dot_f16(w[:, :64], x[:64], lanes=320))[0]), 0)
        self.assertEqual(int(nx.bits(nx.dot_f16(w[:, :64], x[:64]))[0]), 0x80000000)


class Transcendental(unittest.TestCase):
    CODES = {"tanh": on.TANH, "exp": on.EXP, "rsqrt": on.RSQRT, "log": on.LOG}

    def check(self, t, bit_patterns, label):
        a = raw_array(t, bit_patterns)
        for op, code in self.CODES.items():
            want = oracle_map(on.k_unary, (code, t), [int(x) for x in bit_patterns])
            same(nx.transcendental(op, t, a), want, f"{op}.{t} {label}")

    def test_every_i8_input(self):
        a = np.arange(-128, 128).astype(np.int8)
        for op, code in self.CODES.items():
            same(nx.transcendental(op, "i8", a), [on.i8_transcendental(code, x) for x in range(-128, 128)], f"{op}.i8")

    def test_every_f16_input(self):
        before = dict(nx.STATS)
        self.check("f16", np.arange(1 << 16), "all inputs")
        # the f64 fast path decides every input but a handful: the twin does not lean on the oracle
        self.assertLess(nx.STATS["tr_oracle"] - before["tr_oracle"], 64)

    def test_f32_sample(self):
        rng = np.random.default_rng(7)
        pats = np.concatenate([rng.integers(0, 1 << 32, size=14000, dtype=np.uint64),
                               np.array(SPECIAL["f32"], dtype=np.uint64),
                               # the useful range of each function, as ordinary values
                               nx.bits((rng.standard_normal(5000) * 8).astype(np.float32)).astype(np.uint64),
                               nx.bits(np.ldexp(rng.random(1000) + 1, rng.integers(-130, 8, size=1000))
                                       .astype(np.float32)).astype(np.uint64)])
        self.check("f32", pats, "sample")


class DotF16(unittest.TestCase):
    def rows(self, rng, n, k):
        """Random rows; cancelling rows (a product and its negation, with a tiny leftover); sums that need many
        bits; infinities, NaNs and signed zeros."""
        w = raw_array("f16", rng.integers(0, 1 << 16, size=(n, k)).reshape(-1)).reshape(n, k)
        x = raw_array("f16", rng.integers(0, 1 << 16, size=(n, k)).reshape(-1)).reshape(n, k)
        small = (rng.standard_normal((n, k)) * 2.0 ** rng.integers(-8, 8, size=(n, 1))).astype(np.float16)
        w[: n // 2] = small[: n // 2]
        x[: n // 2] = (rng.standard_normal((n // 2, k)) * 4).astype(np.float16)
        h = k // 2
        # cancellation: lane j and lane j + h hold products of opposite sign and equal magnitude
        q = n // 4
        w[q:2 * q, h:] = -w[q:2 * q, :h]
        x[q:2 * q, h:] = x[q:2 * q, :h]
        w[q:2 * q, 0] = np.float16(2.0 ** -20)
        # +-0 rows and one infinity
        w[2 * q] = 0.0
        x[2 * q] = -1.0
        w[2 * q + 1] = np.float16(-0.0)
        x[2 * q + 1] = np.float16(1.0)
        w[2 * q + 2, 0], x[2 * q + 2, 0] = np.inf, 0.0
        w[2 * q + 3, :2], x[2 * q + 3, :2] = np.inf, (1.0, -1.0)
        w[2 * q + 4, 3], x[2 * q + 4, 3] = np.inf, 2.0
        return w, x

    def test_against_oracle(self):
        rng = np.random.default_rng(8)
        k = 64
        w, x = self.rows(rng, 600, k)
        want = oracle_map(on.dot_f16, (), [[int(v) for v in nx.bits(w[r])] for r in range(len(w))],
                          [[int(v) for v in nx.bits(x[r])] for r in range(len(w))])
        got = np.stack([nx.dot_f16(w[r:r + 1], x[r])[0] for r in range(len(w))])
        same(got, want, "dot_f16")

    def test_full_width_matrix(self):
        rng = np.random.default_rng(9)
        w = (rng.standard_normal((20, 320)) * 3).astype(np.float16)
        x = (rng.standard_normal(320) * 3).astype(np.float16)
        want = [on.dot_f16([int(v) for v in nx.bits(w[r])], [int(v) for v in nx.bits(x)]) for r in range(20)]
        same(nx.dot_f16(w, x), want, "dot_f16 320 lanes")


if __name__ == "__main__":
    unittest.main()
