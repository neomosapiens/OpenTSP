#!/usr/bin/env python3
"""The width-768 twins of opentsp_tools.programs.ref_wide: the tiled matvec, LayerNorm over three vectors, the mask from the
page counts, the full-pool attention (12 heads of 64) against float64 mathematics and against their structural rules (the order
of the f32 sums, which a bit-exact kernel must keep; every operation inside is checked against the Python oracle by test_numerics.py).

usage: python3 tests/programs/gen/test_ref_wide.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools.programs import ref_wide as rw


def f32(rng, *shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


class TiledMatvec(unittest.TestCase):
    def test_matches_float64_at_widths_768_and_3072(self):
        rng = np.random.default_rng(1)
        for n_in in (768, 3072):
            w = (f32(rng, 40, n_in) * 0.05).astype(np.float16)
            x, b = f32(rng, n_in), f32(rng, 40)
            want = w.astype(np.float64) @ x.astype(np.float16).astype(np.float64) + b
            np.testing.assert_allclose(rw.matvec_tiled(w, x, b), want, rtol=2e-5, atol=2e-5)

    def test_tiles_are_added_in_ascending_order_in_f32(self):
        # tile partials 2^24 (4096 * 4096), 1, 1: ((2^24 + 1) + 1) = 2^24 in f32, whereas 2^24 + (1 + 1) = 2^24 + 2
        w = np.zeros((1, 768), np.float16)
        x = np.zeros(768, np.float32)
        w[0, [0, 256, 512]] = (4096.0, 1.0, 1.0)
        x[[0, 256, 512]] = (4096.0, 1.0, 1.0)
        self.assertEqual(float(rw.matvec_tiled(w, x)[0]), 2.0 ** 24)
        w[0, [0, 256, 512]] = (1.0, 1.0, 4096.0)                           # (1 + 1) + 2^24
        x[[0, 256, 512]] = (1.0, 1.0, 4096.0)
        self.assertEqual(float(rw.matvec_tiled(w, x)[0]), 2.0 ** 24 + 2)

    def test_a_tile_is_one_exact_dot_rounded_once(self):
        # 255 terms of 2^-24 and one of 1 inside a tile: the exact sum 1 + 255 * 2^-24 rounds once; f32 adds in sequence would not
        w = np.full((1, 256), 2.0 ** -12, np.float16)
        x = np.full(256, 2.0 ** -12, np.float32)
        w[0, 0], x[0] = 1.0, 1.0
        got = float(rw.matvec_tiled(w, x)[0])
        self.assertEqual(got, float(np.float32(1.0 + 255 * 2.0 ** -24)))

    def test_pad_lanes_are_zero_and_change_nothing(self):
        rng = np.random.default_rng(2)
        w = (f32(rng, 8, 300) * 0.1).astype(np.float16)
        x = f32(rng, 300)
        want = rw.matvec_tiled(np.pad(w, ((0, 0), (0, 212))), np.pad(x, (0, 212)))
        self.assertTrue(np.array_equal(rw.matvec_tiled(w, x), want))


class LayerNormWide(unittest.TestCase):
    def test_matches_float64(self):
        rng = np.random.default_rng(3)
        x, g, b = f32(rng, 768, scale=3), f32(rng, 768) + 1, f32(rng, 768)
        d = x.astype(np.float64) - x.astype(np.float64).mean()
        want = d / np.sqrt((d * d).mean() + 1e-5) * g + b
        np.testing.assert_allclose(rw.layernorm_wide(x, g, b, 1e-5), want, rtol=2e-4, atol=2e-5)

    def test_every_lane_sees_the_same_total(self):
        rng = np.random.default_rng(4)
        x = f32(rng, 3, 256, scale=2)
        tot = rw._vector_total("add", x, 16)
        self.assertTrue((tot == tot[0]).all())
        np.testing.assert_allclose(tot[0], x.astype(np.float64).sum(), rtol=1e-5)

    def test_the_vectors_are_added_in_ascending_order_after_the_superlane_folds(self):
        # vector totals (one value in lane 0 of each vector): 2^24, 1, 1 -> (2^24 + 1) + 1 = 2^24, not 2^24 + 2
        x = np.zeros((3, 256), np.float32)
        x[0, 0], x[1, 0], x[2, 0] = 2.0 ** 24, 1.0, 1.0
        self.assertEqual(float(rw._vector_total("add", x, 16)[0]), 2.0 ** 24)
        x[:] = 0
        x[0, 0], x[1, 0], x[2, 0] = 1.0, 1.0, 2.0 ** 24                     # (1 + 1) + 2^24 = 2^24 + 2
        self.assertEqual(float(rw._vector_total("add", x, 16)[0]), 2.0 ** 24 + 2)

    def test_the_fold_inside_a_superlane_is_ascending(self):
        x = np.zeros((3, 256), np.float32)
        x[0, 0], x[0, 1], x[0, 2], x[0, 3] = 2.0 ** 24, 1.0, 1.0, 1.0       # ((((2^24 + 1) + 1) + 1) = 2^24
        self.assertEqual(float(rw._vector_total("add", x, 16)[0]), 2.0 ** 24)

    def test_a_width_that_is_not_whole_vectors_is_refused(self):
        with self.assertRaises(AssertionError):
            rw.layernorm_wide(np.zeros(300, np.float32), np.ones(300, np.float32), np.zeros(300, np.float32), 1e-5)


class Mask(unittest.TestCase):
    def test_visible_slots_are_the_page_prefixes(self):
        counts = np.array([16, 3, 0, 16])
        m = rw.mask_from_encoding(counts, 64)
        vis = m == 0
        self.assertTrue(vis[:16].all() and vis[16:19].all() and not vis[19:32].any() and vis[48:].all())
        self.assertTrue((m[~vis] == -np.inf).all() and m.dtype == np.float32)

    def test_a_pool_that_does_not_match_the_pages_is_refused(self):
        with self.assertRaises(AssertionError):
            rw.mask_from_encoding(np.array([16, 3]), 64)


class PoolAttention(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.pool, self.n = 256, 768
        self.k16 = rw.split(f32(rng, self.pool, self.n, scale=0.5)).astype(np.float16)
        self.v16 = rw.split(f32(rng, self.pool, self.n, scale=0.5)).astype(np.float16)
        self.q = f32(rng, self.n, scale=0.5)
        counts = np.zeros(self.pool // 16, np.int64)
        counts[:9] = 16
        counts[9] = 5                                                      # 149 visible slots
        self.mask = rw.mask_from_encoding(counts, self.pool)
        self.vis = self.mask == 0

    def f64(self, pv16):
        out = []
        for h in range(12):
            lanes = slice(64 * h, 64 * h + 64)
            k = rw.join(self.k16, self.n)[:, lanes].astype(np.float64)
            q = rw.to_f16(self.q).astype(np.float64)[lanes]
            s = 0.125 * (k @ q)
            s = np.where(self.vis, s, -np.inf)
            p = np.exp(s - s.max())
            p /= p.sum()
            v = rw.join(self.v16, self.n)[:, lanes].astype(np.float64)
            out.append(p @ v)
        return np.concatenate(out)

    def test_both_context_forms_match_float64(self):
        want = self.f64(None)
        for pv in ("mxm", "vxm"):
            got = rw.attention_pool(self.q, self.k16, self.v16, self.mask, 12, pv=pv)
            np.testing.assert_allclose(got, want, rtol=5e-3, atol=2e-4, err_msg=pv)     # p is rounded to f16 once

    def test_masked_slots_have_probability_zero_whatever_they_hold(self):
        s = rw.scores_wide(self.k16, self.q, self.mask, 12, 0.125)
        p = rw.softmax_pool(s)
        self.assertTrue((p[:, ~self.vis] == 0).all())
        np.testing.assert_allclose(p.astype(np.float64).sum(-1), 1.0, rtol=1e-5)
        k2, v2 = self.k16.copy(), self.v16.copy()
        k2[~self.vis] = np.float16(7.0)
        v2[~self.vis] = np.float16(-3.0)
        a = rw.attention_pool(self.q, self.k16, self.v16, self.mask, 12)
        b = rw.attention_pool(self.q, k2, v2, self.mask, 12)
        self.assertTrue(np.array_equal(a, b))

    def test_the_score_of_a_head_uses_only_its_64_lanes(self):
        q2 = self.q.copy()
        q2[64:128] += 5.0                                                   # head 1 only
        a = rw.scores_wide(self.k16, self.q, self.mask, 12, 0.125)
        b = rw.scores_wide(self.k16, q2, self.mask, 12, 0.125)
        self.assertTrue(np.array_equal(np.delete(a, 1, axis=0), np.delete(b, 1, axis=0)))
        self.assertFalse(np.array_equal(a[1], b[1]))

    def test_the_context_of_a_head_lies_in_its_lanes(self):
        ctx = rw.context_mxm(rw.softmax_pool(rw.scores_wide(self.k16, self.q, self.mask, 12, 0.125)), self.v16)
        self.assertEqual(ctx.shape, (768,))
        v2 = self.v16.copy()
        v2[:, 0, 64:128] = np.float16(1.0)                                  # head 1's columns of V (vector 0, lanes 64..127)
        c2 = rw.context_mxm(rw.softmax_pool(rw.scores_wide(self.k16, self.q, self.mask, 12, 0.125)), v2)
        self.assertTrue(np.array_equal(np.delete(ctx.reshape(12, 64), 1, 0), np.delete(c2.reshape(12, 64), 1, 0)))

    def test_the_chunks_of_the_softmax_sum_are_added_in_ascending_order(self):
        # lane 0 of three chunks holds the scores 0, -17, -17: e = 1, t, t with t below half an ulp of 1 but 2 t above it.
        # Ascending, (1 + t) + t = 1 and the probability of the first slot is exactly 1; in any other order it is below 1.
        s = np.full((1, 384), -np.inf, np.float32)
        s[0, [0, 128, 256]] = (0.0, -17.0, -17.0)
        p = rw.softmax_pool(s)
        self.assertEqual(float(p[0, 0]), 1.0)
        t = np.exp(np.float32(-17.0))
        self.assertTrue(np.float32(1.0) + t == np.float32(1.0) and np.float32(1.0) + (t + t) > np.float32(1.0))

    def test_the_pool_must_be_whole_chunks(self):
        with self.assertRaises(AssertionError):
            rw.softmax_pool(np.zeros((12, 200), np.float32))


class NumericsOfTheContext(unittest.TestCase):
    def test_the_mxm_form_adds_the_chunks_in_ascending_order(self):
        # head 0, three chunks of 128 slots with partial sums 2^24 (from 4096 * 4096), 1 and 1: the result of the chain
        # ((2^24 + 1) + 1) is 2^24; the reversed chain would give 2^24 + 2
        slots = 384
        p = np.zeros((12, slots), np.float32)
        v = np.zeros((slots, 3, rw.TILE), np.float16)
        p[0, [0, 128, 256]] = (1.0, 1.0, 1.0)
        v[0, 0, 0], v[128, 0, 0], v[256, 0, 0] = 1.0, 1.0, 1.0
        self.assertEqual(float(rw.context_mxm(p, v)[0]), 3.0)
        # unequal scale: p of the first chunk is 2048 (an f16), v is 8192: product 2^24
        p[0, 0], v[0, 0, 0] = 2048.0, 8192.0
        got = float(rw.context_mxm(p, v)[0])
        self.assertEqual(got, 2.0 ** 24)                                   # (2^24 + 1) + 1 in f32 ascending order
        p[0, [0, 128, 256]] = (1.0, 2048.0, 2048.0)
        v[0, 0, 0], v[128, 0, 0], v[256, 0, 0] = 1.0, 8192.0, 8192.0
        self.assertEqual(float(rw.context_mxm(p, v)[0]), 2.0 ** 25)       # 1 + 2^24 rounds to 2^24 (even), + 2^24 = 2^25


    def test_the_vxm_form_sums_the_slots_in_two_interleaved_chains(self):
        # slot 0 = 2^24, slots 1 and 3 = 1: two chains give c0 = 2^24, c1 = 1 + 1 = 2, sum 2^24 + 2; one chain loses both ones (2^24);
        # four chains give (2^24 + 1) + (0 + 1) = 2^24
        p = np.zeros((12, 4), np.float32)
        v = np.zeros((4, 3, rw.TILE), np.float16)
        p[0, :] = (2048.0, 1.0, 0.0, 1.0)
        v[:, 0, 0] = (8192.0, 1.0, 0.0, 1.0)
        self.assertEqual(float(rw.context_vxm(p, v)[0]), 2.0 ** 24 + 2.0)
        self.assertEqual(float(rw.context_vxm(p, v, chains=1)[0]), 2.0 ** 24)
        self.assertEqual(float(rw.context_vxm(p, v, chains=4)[0]), 2.0 ** 24)


class AttentionInHeadPairTiles(unittest.TestCase):
    """Two heads of 64 lanes per vector (tile 128): every head's dot, softmax and sums lie inside the head, so the result equals the 256-lane layout."""

    def test_masked_nonfinite_keys_and_values_do_not_change_context(self):
        rng = np.random.default_rng(91)
        q = rng.standard_normal(128).astype(np.float32)
        k = rng.standard_normal((256, 1, 128)).astype(np.float16)
        v = rng.standard_normal(k.shape).astype(np.float16)
        mask = np.zeros(256, np.float32)
        mask[[17, 18, 19, 145, 146, 147]] = -np.inf
        want = rw.attention_pool(q, k, v, mask, 2, 0.125, pv="vxm", tile=128)
        for arr in (k, v):
            arr[[17, 145], 0, :] = np.inf
            arr[[18, 146], 0, :] = -np.inf
            arr[[19, 147], 0, :] = np.nan
        got = rw.attention_pool(q, k, v, mask, 2, 0.125, pv="vxm", tile=128)
        np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))

    def test_attention_of_head_pairs_equals_the_four_head_layout(self):
        rng = np.random.default_rng(5)
        heads, slots = 12, 256
        q = rng.standard_normal(heads * rw.DK).astype(np.float32)
        k = rng.standard_normal((slots, heads * rw.DK)).astype(np.float16)
        v = rng.standard_normal((slots, heads * rw.DK)).astype(np.float16)
        mask = np.where(rng.random(slots) < 0.2, -np.inf, 0).astype(np.float32)
        mask[0] = 0
        a128 = rw.attention_pool(q, rw.split(k, 128), rw.split(v, 128), mask, heads, 0.125, pv="vxm", tile=128)
        a256 = rw.attention_pool(q, rw.split(k, 256), rw.split(v, 256), mask, heads, 0.125, pv="vxm", tile=256)
        self.assertTrue(np.array_equal(a128, a256))
        s128 = rw.scores_wide(rw.split(k, 128), q, mask, heads, 0.125, 128)
        self.assertEqual(s128.shape, (heads, slots))
        self.assertFalse(np.array_equal(s128[3], s128[2]))


if __name__ == "__main__":
    unittest.main()
