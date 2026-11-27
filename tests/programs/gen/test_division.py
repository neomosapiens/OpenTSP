#!/usr/bin/env python3
"""Tests of the reciprocal idioms (division.py) and of their measured domains (division_domains.py).

usage: [TSP_ISS=target/release/tsp-iss] [OTS_ASM=target/release/ots-asm] python3 tests/programs/gen/test_division.py

The domains are measured by running the idioms on the ISS (every binade of f32 values, then the edges to the last bit). The
tests repeat that measurement and compare it with the checked-in division_domains.json (the numbers the cookbook quotes,
and the edges the golden programs pin), check that the numpy twin and the ISS agree on every sampled value, and check
the edges and the special values in the checked-in golden case."""
import json
import os
import sys
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore", ResourceWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import division  # noqa: E402
import division_domains as dd  # noqa: E402
from opentsp_tools import profile as tprofile

TSP_ISS = os.environ.get("TSP_ISS") or os.path.join(tprofile.ROOT, "target", "release", "tsp-iss")
OTS_ASM = os.environ.get("OTS_ASM") or os.path.join(tprofile.ROOT, "target", "release", "ots-asm")
CASE = os.path.join(tprofile.ROOT, "tests", "programs", "vxm", "division_edges.ref.case.json")


def domains():
    with open(division.DOMAINS) as f:
        return json.load(f)


def f32_bits_list(bits):
    return np.array(bits, dtype=np.uint32).view(np.float32)


class UlpError(unittest.TestCase):
    def test_exact_and_neighbours(self):
        x = f32_bits_list([0x40000000, 0x40400000])                 # 2.0 and 3.0
        exact = np.array([0.5, 1 / 3], dtype=np.float32)
        err = dd.ulp_error("rsqrt_pair", exact, x)
        self.assertLess(err[0], 1e-6)
        self.assertLess(err[1], 0.5)                               # 1/3 rounded is within half an ulp
        nxt = np.nextafter(exact, np.float32(1.0))
        self.assertAlmostEqual(float(dd.ulp_error("rsqrt_pair", nxt, x)[0]), 1.0, places=6)

    def test_not_finite_and_sign(self):
        x = f32_bits_list([0x40000000, 0xC0000000])
        self.assertTrue(np.isinf(dd.ulp_error("rsqrt_pair", np.array([np.inf, 1.0], dtype=np.float32), x)[0]))
        self.assertLess(dd.ulp_error("rsqrt_sq", np.array([0.5, 0.5], dtype=np.float32), x)[1], 1e-6)   # 1/|x|
        self.assertTrue(np.isinf(dd.ulp_error("rsqrt_sq", np.array([np.nan, 0.5], dtype=np.float32), x)[0]))


class Twin(unittest.TestCase):
    def test_known_values(self):
        x = f32_bits_list([0x3F800000, 0x40400000, 0x7149F2CA, 0x0DA24260])      # 1, 3, 1e30, 1e-30
        self.assertEqual(division.twin("rsqrt_sq", x).tolist()[:2], [1.0, np.float32(1 / 3)])
        self.assertEqual(division.twin("rsqrt_sq", x)[2], 0.0)        # x*x overflows
        self.assertTrue(np.isinf(division.twin("rsqrt_sq", x)[3]))     # x*x underflows
        self.assertEqual(float(division.twin("rsqrt_pair", x)[1]), float(np.float32(0.33333331)))
        self.assertEqual(division.twin("newton_sq", np.array([-1.0], dtype=np.float32))[0], 3.0)


@unittest.skipUnless(os.access(TSP_ISS, os.X_OK) and os.access(OTS_ASM, os.X_OK), "tsp-iss and ots-asm are needed")
class Measured(unittest.TestCase):
    def test_iss_measurement_equals_the_checked_in_file(self):
        """The whole sweep and the edge searches on the ISS (about 10 s); the ISS and the twin agree on every value."""
        got = dd.measure(dd.Iss(TSP_ISS, OTS_ASM))
        self.assertEqual(json.dumps(got, indent=1) + "\n", open(division.DOMAINS).read())

    def test_documented_bounds_hold(self):
        for idiom, d in domains()["idioms"].items():
            self.assertLessEqual(d["max_ulp_accurate"], d["bound_ulp"], idiom)
            self.assertEqual(d["bound_ulp"], dd.BOUND[idiom])

    def test_known_edges(self):
        """x*x is a normal f32 exactly from 2^-63 and overflows from 2^64: the edges the arithmetic dictates."""
        d = domains()["idioms"]["rsqrt_sq"]
        self.assertEqual(d["accurate_binades"], [-63, 63])
        self.assertEqual(int(d["last_usable_bits"], 16), 0x5F800000 - 1)
        self.assertEqual(int(d["first_usable_bits"], 16), 0x1A000000 + 1)       # x*x = 2^-150 rounds to zero


class GoldenCase(unittest.TestCase):
    """The expectations of the checked-in golden program tests/programs/vxm/division_edges.ref.case.json."""

    def decode(self, case, j):
        planes = [bytes.fromhex(case["expect"]["words"][f"{44 + i}:{division.ADDR + j}"]) for i in range(4)]
        return np.stack([np.frombuffer(pl, dtype=np.uint8) for pl in planes], axis=1).copy().view("<u4").reshape(-1).view(np.float32)

    def test_edges_are_pinned(self):
        case, dom = json.load(open(CASE)), domains()
        for n, idiom in enumerate(division.IDIOMS):
            res = self.decode(case, 2 * n)
            bits = division.edge_bits(idiom, dom)
            d = dom["idioms"][idiom]
            lo, hi = int(d["first_usable_bits"], 16), int(d["last_usable_bits"], 16)
            at = {b: res[i] for i, b in enumerate(bits)}           # a bit pattern -> its result (the first lane of each)
            usable = lambda r: bool(np.isfinite(r) and r != 0)    # noqa: E731
            self.assertFalse(usable(at[lo - 1]), idiom)
            self.assertTrue(usable(at[lo]), idiom)
            self.assertTrue(usable(at[hi]), idiom)
            if hi + 1 in at:
                self.assertFalse(usable(at[hi + 1]), idiom)

    def test_special_values_are_pinned(self):
        case = json.load(open(CASE))
        for n, idiom in enumerate(division.IDIOMS):
            res = self.decode(case, 2 * n + 1)
            want = division.twin(idiom, division.fill(division.SPECIAL_BITS, len(res)))
            self.assertEqual(res.view(np.uint32).tolist(), want.view(np.uint32).tolist(), idiom)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
