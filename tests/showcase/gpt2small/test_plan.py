"""Tests of the per-chip placement planner (`gpt2s_plan.py`): it reproduces the design's chip-count script at 320 lanes (matrices 52,224 words per
layer, linear [2, 7, 3] at pool 1024), gives the row layouts (256 and 128 lanes) and the host cost of the map vectors of a step.

usage: python3 tests/showcase/gpt2small/test_plan.py"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opentsp_vllm.models.gpt2 import plan as P

REF = json.load(open(P.DEFAULT_CONFIG))
WPS = REF["sram"]["words_per_slice"]
WORDS_CHIP = REF["sram"]["slices"] * WPS
M = P.Gpt2Dims()


class Counts(unittest.TestCase):
    def test_design_script_at_320_lanes(self):
        lay = P.Layout(320, v_planes=4)
        self.assertEqual(lay.matrix_words(M), 52224)                     # the design's figure
        n, layers, _ = P.partition(lay, M, 1024, "linear", WORDS_CHIP)
        self.assertEqual((n, layers), (3, [2, 7, 3]))

    def test_matrix_words_by_tile_width(self):
        self.assertEqual(P.Layout(256).matrix_words(M), 55296)
        self.assertEqual(P.Layout(128).matrix_words(M), 110592)          # twice 256: every d_out row has twice the vectors

    def test_128_lanes_needs_more_chips_than_256(self):
        n256 = P.partition(P.Layout(256), M, 1024, "linear", WORDS_CHIP)[0]
        n128 = P.partition(P.Layout(128), M, 1024, "linear", WORDS_CHIP)[0]
        self.assertEqual((n256, n128), (3, 5))
        self.assertEqual(P.partition(P.Layout(128), M, 1024, "ring", WORDS_CHIP)[0], 4)

    def test_table_alone_nearly_fills_a_chip_at_128_lanes(self):
        self.assertEqual(P.Layout(128).table_words(M), 603084)

    def test_pool_size_changes_the_kv_words(self):
        lay = P.Layout(128)
        self.assertEqual(lay.kv_words(M, 1024), 24576)
        self.assertEqual(lay.kv_words(M, 2048), 2 * 24576)


class PlaneConstraint(unittest.TestCase):
    def test_one_group_per_hemisphere_holds_few_layers(self):
        self.assertEqual(P.layers_per_chip(P.Layout(256), M, 1024, WPS, 1), 3)
        self.assertEqual(P.layers_per_chip(P.Layout(128), M, 1024, WPS, 1), 1)
        self.assertEqual(P.layers_per_chip(P.Layout(128), M, 1024, WPS, 2), 3)

    def test_the_east_group_carries_the_k_pool(self):
        r = P.plane_fit(P.Layout(128), M, 1024, 1, WPS)
        self.assertGreater(r["east"], r["west"])
        self.assertEqual(r["east"] - r["west"], P.Layout(128).k_words(M, 1024) / 16)

    def test_fits_is_the_boundary(self):
        r1 = P.plane_fit(P.Layout(128), M, 1024, 1, WPS)
        r2 = P.plane_fit(P.Layout(128), M, 1024, 2, WPS)
        self.assertTrue(r1["fits"])
        self.assertFalse(r2["fits"])


class StageRegions(unittest.TestCase):
    def test_one_layer_of_gpt2_small_at_128_lanes_fits(self):
        regions, bad = P.stage_regions(P.Layout(128), M, 1024, 1, WPS)
        self.assertEqual(bad, [])
        self.assertEqual([r for r in regions if r[:2] == ("west plane", "weights")], [("west plane", "weights", 0x200, 0x200 + 3456)])

    def test_two_layers_overlap_the_k_pool(self):
        _, bad = P.stage_regions(P.Layout(128), M, 1024, 2, WPS)
        self.assertIn((("east plane", "weights"), ("east plane", "K pool")), bad)

    def test_constants_are_those_of_the_generator(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "programs", "gen"))
        from opentsp_vllm.models.gpt2 import layer as L
        from opentsp_vllm.models.gpt2 import attention as A
        regions, _ = P.stage_regions(P.Layout(128), M, 1024, 1, WPS)
        by = {(r[0], r[1]): r for r in regions}
        self.assertEqual(by["east plane", "weights"][2], L.WBASE)
        self.assertEqual(by["east plane", "K pool"][2], L.K_BASE)
        self.assertEqual(by["V slices", "V pool"][2], A.V_BASE)
        self.assertLess(L.SINK + 6, by["east plane", "text"][2])         # the sink words of a K Scatter are below the text


class HostCost(unittest.TestCase):
    def test_map_vectors_scale_with_tiles(self):
        self.assertEqual(P.Layout(256).map_vectors(M), 25)               # the accepted 24 + 1
        self.assertEqual(P.Layout(128).map_vectors(M), 49)

    def test_cycles_from_the_profile(self):
        cpv = REF["host"]["cycles_per_vector"]
        self.assertEqual(P.host_step(P.Layout(128), M, 12, cpv), (588, 588 * cpv))


if __name__ == "__main__":
    unittest.main()
