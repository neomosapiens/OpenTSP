#!/usr/bin/env python3
"""The kernel cookbook  cannot drift from the code: every generated block of the document
equals what cookbook.py produces now.

usage: python3 tests/programs/gen/test_cookbook.py"""
import os
import sys
import unittest
import warnings

warnings.simplefilter("ignore", ResourceWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cookbook  # noqa: E402
from opentsp_tools import profile as tprofile


class Document(unittest.TestCase):
    def test_every_block_equals_its_generator(self):
        doc = open(cookbook.DOC).read()
        self.assertEqual(cookbook.render(doc), doc, "run: python3 tests/programs/gen/cookbook.py --update")

    def test_every_generator_has_a_block_and_each_block_a_generator(self):
        names = set(cookbook.blocks_in(open(cookbook.DOC).read()))
        self.assertEqual(names, set(cookbook.BLOCKS))

    def test_a_stale_listing_is_found(self):
        doc = open(cookbook.DOC).read()
        stale = doc.replace("rsqrt.f32 E:12, E:8.chain", "rsqrt.f32 E:12, E:9.chain", 1)
        self.assertNotEqual(stale, doc)
        self.assertNotEqual(cookbook.render(stale), stale)

    def test_a_missing_block_is_refused(self):
        doc = open(cookbook.DOC).read()
        with self.assertRaises(ValueError):
            cookbook.render(doc.replace("<!-- cookbook:listing-newton -->", "<!-- cookbook:listing-newtonx -->", 1)
                            .replace("<!-- /cookbook:listing-newton -->", "<!-- /cookbook:listing-newtonx -->", 1))


class Listing(unittest.TestCase):
    TEXT = "bundle @cycle=5 {\n  VXM[0]: add.f32 E:8, E:0, E:4\n}\nbundle @cycle=9 {\n  VXM[1]: mul.f32 E:12, E:8.chain, E:8.chain\n}\n"

    def test_an_instruction_without_a_note_fails(self):
        p = tprofile.load("ref")
        with self.assertRaises(ValueError):
            cookbook.listing(p, self.TEXT, [(r"VXM\[0\]: add", "sum")], [])

    def test_the_annotation_takes_its_numbers_from_the_profile(self):
        p = tprofile.load("test/ref_slow")
        text = cookbook.listing(p, self.TEXT, [(r"VXM\[0\]: add", "add d_func={d_func} d_skew={d_skew}"), (r"VXM\[1\]", "mul")], [])
        d_func, d_skew = p.timing("add")
        self.assertIn(f"d_func={d_func} d_skew={d_skew}", text)
        self.assertIn("+4", text)                                  # cycles since the previous listed line

    def test_the_timing_of_an_instruction_text_is_the_row_of_its_type(self):
        hw = tprofile.load("test/ref_hw")                          # Max.i8 (3, 2), Max.f32 (7, 2), class row 5
        self.assertEqual(cookbook.timing_of(hw, "max.i8 E:8, E:0, E:4"), ("max", 3, 2))
        self.assertEqual(cookbook.timing_of(hw, "max.f32 E:8, E:0, E:4"), ("max", 7, 2))
        self.assertEqual(cookbook.timing_of(hw, "max.f16 E:8, E:0, E:4"), ("max", 5, 2))
        self.assertEqual(cookbook.timing_of(hw, "nosuchop E:8"), None)               # not a mnemonic: no row

    def test_hidden_instructions_are_counted(self):
        p = tprofile.load("ref")
        text = cookbook.listing(p, self.TEXT, [(r"VXM\[1\]", "second")], [(r"VXM\[0\]", "adds")])
        self.assertIn("... 1 adds (cycle 5)", text)

    def test_listings_differ_between_profiles(self):
        """The ref_slow division listing has other latencies than the ref one: the numbers come from the profile."""
        import numpy as np
        import division
        outs = []
        for name in ("ref", "test/ref_slow"):
            p = tprofile.load(name)
            k, _ = division.program(p, "cookbook/division", [("rsqrt_sq", np.arange(1, p.vlanes + 1, dtype=np.float32))], "x")
            outs.append(cookbook.listing(p, k.done("x")[0], cookbook.DIV_NOTES, cookbook.DIV_HIDE))
        self.assertNotEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main(warnings="ignore")
