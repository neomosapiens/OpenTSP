#!/usr/bin/env python3
"""The geometry the generators derive from a profile (`opentsp_tools.programs.geometry`): the sizes of every profile, the install groups, and the
refusal text of a program family that a profile cannot hold.

usage: python3 tests/programs/gen/test_geometry.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools.programs import geometry as G
from opentsp_tools import profile as tprofile

# profile: (lanes, tile, head_tile, chunk, rounds, chunk_rounds, pack, install_rows)
EXPECTED = {
    "ref": (320, 256, 128, 128, 4, 3, 5, 160),
    "test/ref_slow": (320, 256, 128, 128, 4, 3, 5, 160),
    "test/ref_small": (320, 256, 128, 64, 4, 2, 5, 80),
    "test/sl8": (128, 128, 128, 64, 3, 2, 2, 64),
    "test/sl32": (512, 256, 128, 128, 4, 3, 8, 160),
    "test/mid_min": (64, 64, 64, 16, 2, 0, 1, 16),
    "mini": (16, 16, 16, 8, 0, 0, 0, 8),
}


def sizes(g):
    return (g.lanes, g.tile, g.head_tile, g.chunk, g.rounds, g.chunk_rounds, g.pack, g.install_rows)


class Derived(unittest.TestCase):
    def test_sizes_of_every_profile(self):
        for name, want in EXPECTED.items():
            self.assertEqual(sizes(G.of(tprofile.load(name))), want, name)

    def test_ref_is_the_geometry_the_kernels_were_written_for(self):
        g = G.of(tprofile.load("ref"))
        self.assertEqual((g.lanes, g.tile, g.chunk, g.pack), (320, 256, 128, 5))

    def test_the_tile_is_a_power_of_two_of_superlanes_whatever_the_count(self):
        for sl in range(1, 33):
            g = G.Geometry("x", sl, 320, 44, 8192, 12)
            self.assertEqual(g.tile_superlanes & (g.tile_superlanes - 1), 0)
            self.assertLessEqual(g.tile_superlanes, min(sl, 16))
            self.assertGreater(2 * g.tile_superlanes, min(sl, 16))

    def test_a_chunk_fits_the_vector_and_the_matrix(self):
        for rows in range(16, 321, 16):
            for sl in (1, 4, 8, 20, 32):
                g = G.Geometry("x", sl, rows, 44, 8192, 12)
                self.assertLessEqual(g.chunk, g.lanes)
                self.assertLessEqual(g.chunk, g.install_rows)
                self.assertLessEqual(g.chunk, 128)

    def test_the_install_group_is_sixteen_slices_of_which_the_plane_keeps_its_own_hemisphere(self):
        ref = G.of(tprofile.load("ref"))
        self.assertEqual(ref.install_group(0), list(range(0, 16)))
        self.assertEqual(ref.install_group(2), list(range(44, 60)))
        small = G.of(tprofile.load("test/ref_small"))      # 8 slices per hemisphere: the other hemisphere supplies the rest
        self.assertEqual(small.install_group(0), list(range(0, 8)) + list(range(44, 52)))
        self.assertEqual(small.install_group(3), list(range(44, 52)) + list(range(0, 8)))
        for plane in range(4):
            group = small.install_group(plane)
            self.assertEqual(len(set(group)), 16)
            self.assertTrue(all((s % 44) < small.slices_per_hemisphere for s in group))


class Refusal(unittest.TestCase):
    NEEDS = G.Needs("a test program", [
        (G.Need("superlanes", 8, "superlanes", "the tile"), lambda g: g.superlanes),
        (G.Need("words per slice", 4096, "words per slice", "the pools"), lambda g: g.words)])

    def test_the_first_unmet_requirement_is_named_with_both_numbers(self):
        with self.assertRaises(G.ProfileUnfit) as e:
            self.NEEDS.check(tprofile.load("test/mid_min"))
        msg = str(e.exception)
        self.assertIn("profile test/mid_min does not fit a test program", msg)
        self.assertIn("needs 8 superlanes, profile has 4", msg)
        self.assertIn("first unmet requirement: superlanes", msg)

    def test_a_smaller_memory_is_a_resource_message(self):
        with self.assertRaises(G.ProfileUnfit) as e:
            self.NEEDS.check(tprofile.load("test/ref_small"))
        self.assertIn("needs 4096 words per slice, profile has 1024", str(e.exception))

    def test_a_larger_profile_than_the_program_needs_fits(self):
        for name in ("ref", "test/sl32", "test/sl8"):
            self.assertIsNotNone(self.NEEDS.check(tprofile.load(name)))


class Capacity(unittest.TestCase):
    """What an MXM plane holds is checked where a generator installs: a matrix that does not fit is refused with the rows it needs, not
    left to fail at run time."""

    def test_the_weight_buffer_holds_mxm_rows_stream_values(self):
        ref, sl8 = tprofile.load("ref"), tprofile.load("test/sl8")
        G.check_install(ref, 20, "a matrix")            # 20 LW cycles of 16 stream values = 320 rows
        G.check_install(sl8, 8, "a matrix")             # 128 rows
        with self.assertRaises(G.ProfileUnfit) as e:
            G.check_install(sl8, 9, "the matrix of 72 f16 rows")
        self.assertIn("profile test/sl8 does not fit the matrix of 72 f16 rows: needs 144 MXM rows, profile has 128", str(e.exception))
        self.assertIn("first unmet requirement: mxm.rows", str(e.exception))
        with self.assertRaises(G.ProfileUnfit):
            G.check_install(ref, 21, "x")

    def test_the_expansion_and_the_k_chunk_of_the_wide_programs_are_refused_where_the_plane_is_too_small(self):
        from opentsp_vllm.models.gpt2 import attention as ship_attn
        from opentsp_vllm.models.gpt2 import wide as ship_wide
        sl8 = tprofile.load("test/sl8")
        with self.assertRaisesRegex(G.ProfileUnfit, "does not fit the expansion matrix of 2 heads of 64 lanes: needs 256 MXM rows, profile has 128"):
            ship_wide.wide_expand(sl8, slots=32, heads=2)
        with self.assertRaisesRegex(G.ProfileUnfit, "does not fit the matrix of 128 f16 rows: needs 256 MXM rows, profile has 128"):
            ship_attn.attention(sl8, tiles=1, chunks=2)

    def test_the_wide_programs_take_the_tile_of_the_profile(self):
        from opentsp_vllm.models.gpt2 import wide as ship_wide
        for name, tile in (("ref", 256), ("test/sl32", 256), ("test/sl8", 128)):
            text, _ = ship_wide.matvec_768(tprofile.load(name))
            self.assertIn(f"in {768 // tile} tiles of {tile}", text)


if __name__ == "__main__":
    unittest.main()
