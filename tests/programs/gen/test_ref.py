#!/usr/bin/env python3
"""The kernel twins of opentsp_tools.programs.ref against float64 mathematics and against their structural rules (every operation
inside them is checked bit for bit against the Python oracle by test_numerics.py).

usage: python3 tests/programs/gen/test_ref.py
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs import ref

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "showcase", "tinystories"))
from opentsp_vllm.models.tinystories import paged_reference as reference_paged


def f32(rng, *shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


class Twins(unittest.TestCase):
    def test_reduce_bcast_fills_every_lane_with_the_segment_total(self):
        x = f32(np.random.default_rng(1), 3, 64, scale=4)
        for op, want in (("add", x.astype(np.float64).sum(-1)), ("max", x.max(-1))):
            got = ref.reduce_bcast(op, x)
            self.assertTrue((got == got[:, :1]).all())
            np.testing.assert_allclose(got[:, 0], want, rtol=1e-5)

    def test_layernorm_matches_float64(self):
        rng = np.random.default_rng(2)
        x, g, b = f32(rng, 64, scale=3), f32(rng, 64) + 1, f32(rng, 64)
        d = x.astype(np.float64) - x.astype(np.float64).mean()
        want = d / np.sqrt((d * d).mean() + 1e-5) * g + b
        np.testing.assert_allclose(ref.layernorm(x, g, b, 1e-5), want, rtol=1e-4, atol=1e-5)

    def test_softmax_rows_sum_to_one_and_masked_slots_are_zero(self):
        s = f32(np.random.default_rng(3), 5, 64, scale=2)
        s[:, 10:50] = -np.inf
        p = ref.softmax(s)
        self.assertTrue((p[:, 10:50] == 0).all())
        np.testing.assert_allclose(p.astype(np.float64).sum(-1), 1.0, rtol=1e-5)

    def test_head_masks_partition_the_lanes(self):
        m = ref.head_lane_masks(16, 64)
        self.assertEqual(m.shape, (16, 64))
        self.assertTrue((m.astype(np.float32).sum(0) == 1).all())
        q = ref.to_f16(f32(np.random.default_rng(4), 64))
        parts = np.stack([ref.head_mask(q, m[h]) for h in range(16)])
        self.assertTrue(np.array_equal(parts.astype(np.float32).sum(0), q.astype(np.float32)))

    def test_scores_pick_the_dot_of_the_head_lanes(self):
        rng = np.random.default_rng(5)
        k16 = f32(rng, 64, 64).astype(np.float16)
        q = f32(rng, 64)
        mask = np.zeros(64, np.float32)
        mask[7] = -np.inf
        sc = ref.scores(k16, q, mask, ref.head_lane_masks(16, 64))
        q16 = ref.to_f16(q).astype(np.float64)
        for h in (0, 5, 15):
            lanes = slice(4 * h, 4 * h + 4)
            want = 0.5 * (k16[:, lanes].astype(np.float64) @ q16[lanes])
            ok = np.arange(64) != 7
            np.testing.assert_allclose(sc[h][ok], want[ok], rtol=1e-6, atol=1e-7)
        self.assertTrue((sc[:, 7] == -np.inf).all())

    def test_expand_repeats_each_head_over_its_lanes(self):
        p = f32(np.random.default_rng(6), 16, 64)
        e = ref.expand(p, 4)
        self.assertEqual(e.shape, (64, 64))
        self.assertTrue(np.array_equal(e[9, 8:12], np.repeat(ref.from_f16(ref.to_f16(p))[2, 9], 4)))

    def test_attention_context_matches_float64(self):
        rng = np.random.default_rng(7)
        k16, v = f32(rng, 64, 64).astype(np.float16), f32(rng, 64, 64)
        q = f32(rng, 64)
        mask = np.zeros(64, np.float32)
        p = ref.softmax(ref.scores(k16, q, mask, ref.head_lane_masks(16, 64)))
        got = ref.context(ref.expand(p, 4), v)
        want = np.concatenate([p[h].astype(np.float64) @ v[:, 4 * h:4 * h + 4] for h in range(16)])
        np.testing.assert_allclose(got, want, rtol=5e-3, atol=1e-4)      # p is rounded to f16 once

    def test_context_sums_in_four_interleaved_chains(self):
        # 2^24 + 1 + 1 + 1 gives 2^24 in one sequential chain, 2^24 + 2 as (2^24 + 1) + (1 + 1) of interleaved chains
        p = np.ones((64, 1), np.float32)
        v = np.zeros((64, 1), np.float32)
        v[0, 0], v[1, 0], v[2, 0], v[3, 0] = 2.0 ** 24, 1.0, 1.0, 1.0
        self.assertEqual(float(ref.context(p, v)[0]), 2.0 ** 24 + 2)
        v[:4] = 0
        v[0, 0], v[4, 0], v[8, 0], v[12, 0] = 2.0 ** 24, 1.0, 1.0, 1.0
        self.assertEqual(float(ref.context(p, v)[0]), 2.0 ** 24)

    def test_gelu_matches_float64(self):
        x = f32(np.random.default_rng(8), 128, scale=2)
        xd = x.astype(np.float64)
        want = 0.5 * xd * (1 + np.tanh(np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)))
        np.testing.assert_allclose(ref.gelu(x), want, rtol=1e-5, atol=1e-6)

    def test_matvec_rounds_the_input_to_f16_once(self):
        rng = np.random.default_rng(9)
        w = (f32(rng, 8, 64) * 0.1).astype(np.float16)
        x, b = f32(rng, 64), f32(rng, 8)
        want = nx.binary("add", "f32", nx.dot_f16(w, x.astype(np.float16), 320), b)
        self.assertTrue(np.array_equal(ref.matvec_f16(w, x, b), want))


class Tree8(unittest.TestCase):
    """The reduce-and-broadcast tree over a segment of eight superlanes (C = 128,  step 5 and 2.7 item 5)."""

    def test_rotation_maps_are_bijections_with_a_partial_last_segment(self):
        for vlanes in (128, 256, 320):
            for rot in (4, 2, 1):
                m = nx.rotation_map(vlanes, rot, 8)
                self.assertEqual(sorted(m), list(range(vlanes)), (vlanes, rot))
                self.assertTrue((m % 16 == np.arange(vlanes) % 16).all())

    def test_the_partial_segment_rotates_inside_itself(self):
        m = nx.rotation_map(320, 4, 8)
        self.assertEqual(int(m[256]), 256)                  # rotation 4 of a segment of four is the identity
        self.assertEqual(int(m[16 * 16]), 16 * 16)
        self.assertEqual(int(nx.rotation_map(320, 1, 8)[16 * 19]), 16 * 16)     # superlane 19 -> 16
        self.assertEqual(int(nx.rotation_map(320, 1, 8)[16 * 7]), 0)

    def test_segment_four_maps_are_unchanged(self):
        for rot in (2, 1):
            g = np.arange(320) // 16
            want = 16 * ((g // 4) * 4 + (g % 4 + rot) % 4) + np.arange(320) % 16
            self.assertTrue(np.array_equal(nx.rotation_map(320, rot), want))

    def test_numerics_reduce_bcast_takes_the_segment(self):
        x = f32(np.random.default_rng(11), 3, 320, scale=4)
        self.assertTrue(np.array_equal(nx.reduce_bcast("add", "f32", x), nx.reduce_bcast("add", "f32", x, 4)))
        s = nx.reduce_bcast("max", "f32", x, 8)
        for seg in range(2):
            want = x[:, 128 * seg:128 * seg + 128].max(-1)
            self.assertTrue(np.array_equal(s[:, 128 * seg:128 * seg + 128:16], np.repeat(want[:, None], 8, 1)))

    def test_ref_softmax_with_a_segment_equals_the_reference_paged_tree_on_each_segment(self):
        rng = np.random.default_rng(12)
        s = f32(rng, 4, 320, scale=3)
        s[0, 10:50] = -np.inf
        s[1, 70:128] = -np.inf                      # an empty upper quarter
        s[2, 64:128] = -np.inf                      # an empty upper half
        s[3, 128 + 5:256] = -np.inf
        got = ref.softmax(s, 8)
        for r in range(4):
            for seg in range(2):
                part = s[r, 128 * seg:128 * seg + 128]
                self.assertTrue(np.array_equal(got[r, 128 * seg:128 * seg + 128], reference_paged.softmax_full(part)), (r, seg))

    def test_an_empty_upper_half_makes_the_tree_the_four_superlane_tree(self):
        #  item 5: the first round (rotation 4) adds +0, so rounds 2 and 1 are the tree of four superlanes
        rng = np.random.default_rng(13)
        s = f32(rng, 128, scale=3)
        s[64:] = -np.inf
        full = reference_paged.softmax_full(s)
        self.assertTrue(np.array_equal(full[:64], ref.softmax(s[:64])))
        self.assertTrue((full[64:] == 0).all())

    def test_the_order_of_the_rotations_matters(self):
        #  item 5: with the order 1, 2, 4 the totals differ in the last bit, so the order 4, 2, 1 is required
        rng = np.random.default_rng(1)
        differ = 0
        for _ in range(50):
            e = np.exp(f32(rng, 128, scale=3))
            tot = {}
            for order in ((4, 2, 1), (1, 2, 4)):
                s = nx.reduce("add", "f32", e)
                for r in order:
                    s = nx.binary("add", "f32", s, s[nx.rotation_map(128, r, 8)])
                tot[order] = s
            differ += not np.array_equal(tot[(4, 2, 1)], tot[(1, 2, 4)])
            self.assertTrue(np.array_equal(tot[(4, 2, 1)], nx.reduce_bcast("add", "f32", e, 8)))
        self.assertGreater(differ, 10)

    def test_default_segment_keeps_the_two_round_tree(self):
        x = f32(np.random.default_rng(14), 64, scale=2)
        self.assertTrue(np.array_equal(ref.softmax(x), ref.softmax(x, None)))
        self.assertTrue(np.array_equal(ref.reduce_bcast("add", x), ref.reduce_bcast("add", x, None)))


if __name__ == "__main__":
    unittest.main()
