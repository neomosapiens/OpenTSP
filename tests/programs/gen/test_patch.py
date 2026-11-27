#!/usr/bin/env python3
"""Tests of the patch sites of opentsp_tools.patch.

usage: TSP_SIM=target/spec/tsp_sim [TSP_ISS=target/release/tsp-iss] [OTS_ASM=target/release/ots-asm]
       python3 tests/programs/gen/test_patch.py

A kernel emitter marks an operand as a patch site (name, range); the text carries a placeholder; after lowering each
site is located by assembling the program twice with two placeholder values, and the image differences must be exactly
the recorded sites. The host patcher then applies {site: value} to the image bytes with range checks. The tests check
the self-check in both directions, the range checks, the patcher's effect on the image (only the site bits change),
and, on tsp_sim and the ISS, that a program patched with two address sets reads the intended words.
"""
import dataclasses
import os
import random
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy as np

warnings.simplefilter("ignore", ResourceWarning)     # tsp_as leaves its encoding files to the garbage collector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools import emit
from opentsp_tools import patch
from opentsp_tools import profile as tprofile
from opentsp_tools.programs.kernels import Builder
from opentsp_tools.pager import CycleView
from opentsp_tools.stations import ScheduleError

sys.path.insert(0, os.path.join(tprofile.SPEC, "tools"))
from opentsp_tools import refasm
tsp_as = refasm.load()  # noqa: E402

PROFILES = ["ref", "mini"]
TSP_SIM = os.environ.get("TSP_SIM") or os.path.join(tprofile.ROOT, "target", "spec", "tsp_sim")
TSP_ISS = os.environ.get("TSP_ISS") or os.path.join(tprofile.ROOT, "target", "release", "tsp-iss")
OTS_ASM = os.environ.get("OTS_ASM") or os.path.join(tprofile.ROOT, "target", "release", "ots-asm")

# The fields of encoding.json, written out: read/write `addr` is bits 47..32, permute `map` bits 33..30.
ADDR = (32, 16)
MAP = (30, 4)
SRCS, DSTS, DST_ADDR = (0, 4, 8), (44, 48, 52), (0x20, 0x21, 0x22)
LO, HI = 0x10, 0x1F          # the range of the three patched reads
BASE = (0x10, 0x10, 0x10)    # their placeholders


def reads(p, patched=True):
    """Three MEM reads of slices 0, 4, 8, each written to an east slice; with `patched` the read addresses are the
    sites rd0, rd1, rd2 (range LO..HI, placeholder BASE)."""
    b = Builder(p, "patch test: three patched reads")
    for i in range(3):
        v = b.read(i, [SRCS[i]], [BASE[i]], "E", 4 * i,
                   patch=[(f"rd{i}", LO, HI)] if patched else None)
        b.write(v, [DSTS[i]], [DST_ADDR[i]])
    return b


def with_permute(p):
    """The three reads and a Permute whose map index is a site."""
    b = reads(p)
    d = patch.Decl("pm", "map", 0, 15, 3)
    b.prog.add(40, "SXM.W.PRM", f"permute W:8, {d.token}, W:12", site=d)
    return b


def view_of(b):
    b.finish()
    return CycleView.of(b.prog)


def assemble_object(p, view, values=None):
    """The bytes of the object `ots-asm` assembles from the program of `view` with the sites set to `values`."""
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "p.ots"), os.path.join(d, "p.elf")
        with open(src, "w") as f:
            f.write(view.program(p).text(values))
        subprocess.run([OTS_ASM, "asm", "--config", p.path, "-o", out, src], check=True, capture_output=True)
        with open(out, "rb") as f:
            return f.read()


class Recording(unittest.TestCase):
    def test_text_of_a_placeholder_is_the_text_of_a_literal(self):
        for name in PROFILES:
            p = tprofile.load(name)
            self.assertEqual(reads(p).finish(), reads(p, patched=False).finish())

    def test_sites_are_listed_with_their_instruction(self):
        p = tprofile.load("ref")
        sites = reads(p).prog.sites
        self.assertEqual([s.name for s in sites], ["rd0", "rd1", "rd2"])
        self.assertEqual([(s.queue, s.field, s.lo, s.hi, s.placeholder, s.index) for s in sites],
                         [(SRCS[i], "addr", LO, HI, BASE[i], 0) for i in range(3)])

    def test_a_site_outside_its_own_range_is_refused(self):
        p = tprofile.load("ref")
        b = Builder(p)
        with self.assertRaisesRegex(patch.PatchError, "placeholder"):
            b.read(0, [0], [0x30], "E", 0, patch=[("x", LO, HI)])

    def test_duplicate_site_name_is_refused(self):
        p = tprofile.load("ref")
        b = Builder(p)
        b.read(0, [0], [0x10], "E", 0, patch=[("x", LO, HI)])
        with self.assertRaisesRegex(patch.PatchError, "x"):
            b.read(1, [4], [0x10], "E", 4, patch=[("x", LO, HI)])


class Locating(unittest.TestCase):
    def test_diff_of_two_assemblies_is_exactly_the_sites(self):
        for name in PROFILES:
            p = tprofile.load(name)
            view = view_of(with_permute(p))
            t = patch.locate(view, p)
            self.assertEqual(list(t.names), ["rd0", "rd1", "rd2", "pm"])
            self.assertEqual(list(t.lo), [ADDR[0]] * 3 + [MAP[0]])
            self.assertEqual(list(t.width), [ADDR[1]] * 3 + [MAP[1]])
            # Independent of the table: the bits that differ between two placeholder assemblies.
            a, b = patch.lower(view, p, {"rd0": 0, "rd1": 0, "rd2": 0, "pm": 0}), \
                patch.lower(view, p, {"rd0": 0x1FFF, "rd1": 0x1FFF, "rd2": 0x1FFF, "pm": 15})
            diff = {(q, u, bit) for q in a for u, (x, y) in enumerate(zip(a[q], b[q])) for bit in range(32)
                    if (x ^ y) >> bit & 1}
            want = set()
            for i in range(len(t.names)):
                # a read address carries sram.addr_bits of the 16 bits of its field
                w = p.addr_bits if i < 3 else t.width[i]
                for bit in range(t.lo[i], t.lo[i] + w):
                    unit, ub = (t.unit[i] + (0 if bit >= 32 else 1), bit % 32) if t.nunits[i] == 2 \
                        else (t.unit[i], bit)
                    want.add((int(t.queue[i]), int(unit), ub))
            self.assertEqual(diff, want)
            self.assertEqual({q: len(v) for q, v in a.items()}, {q: len(v) for q, v in b.items()})

    def test_the_map_field_crosses_the_unit_boundary(self):
        p = tprofile.load("ref")
        t = patch.locate(view_of(with_permute(p)), p)
        i = list(t.names).index("pm")
        self.assertEqual((t.nunits[i], t.lo[i], t.width[i]), (2, 30, 4))
        self.assertTrue(t.lo[i] < 32 < t.lo[i] + t.width[i])

    def test_located_fields_avoid_the_opcode_and_the_length_tag(self):
        p = tprofile.load("ref")
        t = patch.locate(view_of(with_permute(p)), p)
        for i in range(len(t.names)):
            field = ((1 << int(t.width[i])) - 1) << int(t.lo[i])
            self.assertEqual(field & int(t.fixed[i]), 0, t.names[i])
            self.assertEqual(field >> 60, 0)

    def test_text_that_is_the_same_for_every_placeholder_is_not_a_difference(self):
        p = tprofile.load("ref")
        view = view_of(reads(p))
        t = patch.locate(view, p)
        self.assertEqual(len(t.names), 3)
        self.assertEqual(sorted(set(t.queue)), sorted(SRCS))


def first_bank1(p):
    return next(a for a in range(p.words) if p.bank(a) == 1)


class Banks(unittest.TestCase):
    """A read and a write of one thread in one bundle must be in different banks (OTS-E-MEM-001): the assemblies of
    `locate` use legal values, and a site's range stays in one bank inside the SRAM."""

    def bundle(self, p):
        b1 = first_bank1(p)
        prog = emit.Program(p, "x")
        prog.add(5, "MEM[4]", "read @r, E:0, 0x0", site=patch.Decl("r", "addr", 0x10, 0x1F, 0x10))
        prog.add(5, "MEM[4]", "write @w, E:1, 0x0", site=patch.Decl("w", "addr", b1, b1 + 15, b1))
        return CycleView.of(prog), b1

    def test_a_read_and_a_write_on_different_banks_in_one_bundle(self):
        for name in PROFILES:
            p = tprofile.load(name)
            view, b1 = self.bundle(p)
            with self.assertRaises(tsp_as.AsmError):       # both fields at 0 are in one bank: the all-zero assembly is illegal
                patch.lower(view, p, {"r": 0, "w": 0})
            t = patch.locate(view, p)
            self.assertEqual(list(t.names), ["r", "w"])
            self.assertEqual(list(t.lo), [ADDR[0]] * 2)
            self.assertEqual(list(t.width), [ADDR[1]] * 2)
            image = patch.to_image(patch.lower(view, p))
            vals = {"r": 0x1F, "w": b1 + 9}
            out = patch.apply(image, t, vals)
            self.assertEqual(patch.to_units(out), patch.lower(view, p, vals))

    def lossy(self, keep_mask, site, view, p):
        """`patch._assemble` for an assembler that drops the bits of `site` outside `keep_mask` (they keep the placeholder's
        value): the self-check of `locate` must notice."""
        real = patch._assemble
        decl = next(x for x in view.sites if x.name == site)

        def assemble(v, pp, values=None):
            vals = dict(values or {})
            if site in vals:
                vals[site] = (vals[site] & keep_mask) | (decl.placeholder & ~keep_mask)
            return real(v, pp, vals)
        return unittest.mock.patch.object(patch, "_assemble", assemble)

    def test_the_fallback_verifies_every_bit_of_a_pool_site_but_the_bank_bit_by_extremes(self):
        # the bundle's all-zero assembly is illegal, so `locate` takes the fallback; a site in bank 1 declared 0x1000..0x100f
        # has its whole field checked, not only its low four bits
        for name in PROFILES:
            p = tprofile.load(name)
            view, b1 = self.bundle(p)
            patch.locate(view, p)
            with self.lossy(0xF, "w", view, p), self.assertRaisesRegex(patch.PatchError, "site w"):
                patch.locate(view, p)
            with self.lossy(0xFF, "w", view, p), self.assertRaisesRegex(patch.PatchError, "site w"):
                patch.locate(view, p)

    def test_the_fallback_verifies_the_bank_bit_by_flipping_every_mem_site(self):
        for name in PROFILES:
            p = tprofile.load(name)
            view, b1 = self.bundle(p)
            bank, real = 1 << p.bank_bit, patch._assemble
            decls = {x.name: x for x in view.sites}

            def drops_the_flip(v, pp, values=None):          # an assembler that loses the bank bit when a flip asks for it
                vals = dict(values or {})
                if vals and all(vals.get(n) == d.placeholder ^ bank for n, d in decls.items()):
                    vals = {n: d.placeholder for n, d in decls.items()}
                return real(v, pp, vals)
            with unittest.mock.patch.object(patch, "_assemble", drops_the_flip), \
                    self.assertRaisesRegex(patch.PatchError, "site [rw]: the bank bit"):
                patch.locate(view, p)

    def test_a_permute_map_beside_a_cross_bank_read_and_write_is_verified_on_all_four_bits(self):
        for name in PROFILES:
            p = tprofile.load(name)
            view, b1 = self.bundle(p)
            prog = view.program(p)
            prog.add(40, "SXM.W.PRM", "permute W:8, @pm, W:12", site=patch.Decl("pm", "map", 0, 1, 0))
            v = CycleView.of(prog)
            t = patch.locate(v, p)
            self.assertIn("pm", list(t.names))
            real = patch._assemble

            def two_bits(vw, pp, values=None):                 # the assembler ignores bits 3..1 of the map
                vals = dict(values or {})
                if "pm" in vals:
                    vals["pm"] &= 1
                return real(vw, pp, vals)
            with unittest.mock.patch.object(patch, "_assemble", two_bits), self.assertRaisesRegex(patch.PatchError, "site pm"):
                patch.locate(v, p)

    def test_a_range_beyond_the_sram_is_refused(self):
        p = tprofile.load("mini")
        with self.assertRaisesRegex(patch.PatchError, r"x.*words"):
            Builder(p).read(0, [0], [0x10], "E", 0, patch=[("x", 0x10, p.words)])

    def test_a_range_across_two_banks_is_refused(self):
        for name in PROFILES:
            p = tprofile.load(name)
            edge = first_bank1(p)
            with self.assertRaisesRegex(patch.PatchError, r"x.*bank"):
                Builder(p).read(0, [0], [edge - 1], "E", 0, patch=[("x", edge - 1, edge)])

    def test_an_any_bank_site_may_cross_the_bank_and_conflicts_with_a_write_of_its_thread_in_either_bank(self):
        for name in PROFILES:
            p = tprofile.load(name)
            edge = first_bank1(p)
            b = Builder(p)
            b.read(0, [0], [edge - 1], "E", 0, patch=[("x", edge - 1, edge, 1, 1, True)])
            for addr in (0x10, edge + 2):
                with self.assertRaisesRegex(ScheduleError, "R and W"):
                    b.t.bank(0, 0, "W", addr)
            b.t.bank(0, 0, "W", 0, thread=1)            # another thread never clashes
            c = Builder(p)
            c.t.bank(0, 0, "W", 0x10)
            with self.assertRaisesRegex(ScheduleError, "R and W"):
                c.read(0, [0], [edge - 1], "E", 0, patch=[("x", edge - 1, edge, 1, 1, True)])
            view = CycleView.of(b.prog)
            t = patch.locate(view, p)
            image = patch.to_image(patch.lower(view, p))
            for vals in ({"x": edge - 1}, {"x": edge}):
                self.assertEqual(patch.to_units(patch.apply(image, t, vals)), patch.lower(view, p, vals))

    def test_a_range_longer_than_a_bank_is_refused(self):
        p = tprofile.load("mini")
        with self.assertRaisesRegex(patch.PatchError, r"x.*bank"):
            Builder(p).read(0, [0], [0], "E", 0, patch=[("x", 0, p.words - 1)])

    def test_a_range_of_one_value_is_refused(self):
        with self.assertRaisesRegex(patch.PatchError, "two values"):
            patch.Decl("x", "addr", 5, 5, 5)


class AllowList(unittest.TestCase):
    """Only the fields the design patches can be sites: a MEM read or write `addr` and a Permute, Distribute or
    Loadmap `map`. Any other field can change the run (loop counts, configuration, threads), not only its operands."""

    REFUSED = [
        ("an AGT iteration count", "iterations @x", "iterations", "n"),
        ("an AGT countdown", "countdown @x", "countdown", "d"),
        ("an AGT step", "step @x", "step", "a"),
        ("an AGT stream step", "streamstep @x", "streamstep", "s"),
        ("the lane configuration", "config @x", "config", "v"),
        ("the vector length", "setvl @x", "setvl", "n"),
        ("the thread of a read", "read 0x10, E:0, @x", "read", "thread"),
        ("the thread of a write", "write 0x10, E:0, @x", "write", "thread"),
        ("a Repeat count", "repeat @x, 0x1", "repeat", "n"),
        ("a Nop count", "nop @x", "nop", "n"),
        ("a Gather's k", "gather E:0, W:1, @x, 0x0", "gather", "k"),
        ("a stream of a read", "read 0x10, @x, 0x0", "read", "dst"),
    ]

    def test_every_other_field_is_refused_by_name(self):
        p = tprofile.load("ref")
        for what, text, op, field in self.REFUSED:
            prog = emit.Program(p, "x")
            with self.assertRaisesRegex(patch.PatchError, rf"{op}.*{field}.*(not a patch site|allowed)", msg=what):
                prog.add(3, "MEM[0]", text, site=patch.Decl("x", field, 0, 1, 0))
            self.assertEqual(prog.sites, [], what)

    def test_locate_refuses_a_record_of_such_a_field(self):
        p = tprofile.load("ref")
        prog = emit.Program(p, "x")
        prog.add(3, "MEM[0]", "iterations @x")
        prog.sites.append(patch.Site("x", 0, 3, 0, "iterations", "n", 0, 9, 5))
        with self.assertRaisesRegex(patch.PatchError, "x.*iterations"):
            patch.locate(CycleView.of(prog), p)

    def test_the_fields_of_the_design_are_accepted(self):
        p = tprofile.load("ref")
        prog = emit.Program(p, "x")
        for i, (q, text, field) in enumerate([("MEM[0]", "read @s, E:0, 0x0", "addr"),
                                              ("MEM[4]", "write @s, E:0, 0x0", "addr"),
                                              ("SXM.W.PRM", "permute W:8, @s, W:12", "map"),
                                              ("SXM.W.DST[0]", "distribute W:8, @s, W:12", "map"),
                                              ("SXM.W.TSP[0]", "loadmap @s, W:12", "map")]):
            prog.add(3 + i, q, text.replace("@s", f"@s{i}"), site=patch.Decl(f"s{i}", field, 0, 3, 1))
        self.assertEqual(len(prog.sites), 5)


class SelfCheck(unittest.TestCase):
    def view(self):
        p = tprofile.load("ref")
        return p, view_of(with_permute(p))

    def wrong(self, **kw):
        p, view = self.view()
        view.sites[1] = dataclasses.replace(view.sites[1], **kw)
        with self.assertRaises(patch.PatchError) as cm:
            patch.locate(view, p)
        return str(cm.exception)

    def test_wrong_cycle(self):
        self.assertIn("rd1", self.wrong(cycle=7))

    def test_wrong_queue(self):
        self.assertIn("rd1", self.wrong(queue=5))

    def test_wrong_index(self):
        self.assertIn("rd1", self.wrong(index=1))

    def test_wrong_field(self):
        self.assertIn("rd1", self.wrong(op="permute", field="map"))

    def test_unknown_field(self):
        self.assertIn("rd1", self.wrong(field="nosuch"))

    def test_a_record_on_another_sites_instruction_leaves_bits_unrecorded(self):
        p, view = self.view()
        s2 = view.sites[2]
        view.sites[1] = dataclasses.replace(view.sites[1], queue=s2.queue, cycle=s2.cycle, index=s2.index)
        with self.assertRaisesRegex(patch.PatchError, "no site records"):
            patch.locate(view, p)

    def test_a_record_on_an_instruction_that_has_no_token_does_not_differ(self):
        p, view = self.view()
        c, q, t = next(e for e in view.entries if e[1] == DSTS[0])
        self.assertTrue(t.startswith("write 0x20"))
        view.sites[1] = dataclasses.replace(view.sites[1], op="write", queue=q, cycle=c, hi=0x3F, placeholder=0x20)
        with self.assertRaisesRegex(patch.PatchError, "rd1 does not differ"):
            patch.locate(view, p)

    def test_a_record_whose_instruction_carries_another_value(self):
        p, view = self.view()
        c, q, t = next(e for e in view.entries if e[1] == DSTS[0])
        view.sites[1] = dataclasses.replace(view.sites[1], op="write", queue=q, cycle=c)
        with self.assertRaisesRegex(patch.PatchError, "rd1.*carries"):
            patch.locate(view, p)

    def test_a_record_without_a_token(self):
        p, view = self.view()
        view.sites.append(patch.Site("ghost", 0, 0, 0, "read", "addr", 0, 15, 0))
        with self.assertRaisesRegex(patch.PatchError, "ghost"):
            patch.locate(view, p)

    def test_a_token_without_a_record(self):
        p, view = self.view()
        del view.sites[2]
        with self.assertRaisesRegex(patch.PatchError, "rd2"):
            patch.locate(view, p)

    def test_a_range_beyond_the_field(self):
        self.assertIn("rd1", self.wrong(hi=1 << 13))

    def test_a_literal_where_a_token_is_recorded(self):
        p, view = self.view()
        view.entries = [(c, q, t.replace("@rd0", "0x11")) for c, q, t in view.entries]
        with self.assertRaisesRegex(patch.PatchError, "rd0"):
            patch.locate(view, p)


class Apply(unittest.TestCase):
    def setUp(self):
        self.p = tprofile.load("ref")
        self.view = view_of(with_permute(self.p))
        self.t = patch.locate(self.view, self.p)
        self.image = patch.to_image(patch.lower(self.view, self.p))

    @staticmethod
    def full(**over):
        return {"rd0": LO, "rd1": LO, "rd2": LO, "pm": 3, **over}

    def test_out_of_range_is_refused_naming_the_site(self):
        for v in (LO - 1, HI + 1, -1, 1 << 13):
            with self.assertRaisesRegex(patch.PatchError, r"rd1.*%d" % v):
                patch.apply(self.image, self.t, self.full(rd1=v))
        with self.assertRaisesRegex(patch.PatchError, "pm"):
            patch.apply(self.image, self.t, self.full(pm=16))

    def test_values_that_are_not_integers_are_refused_naming_the_site(self):
        for v in (None, float("nan"), float("inf"), 1.5, 16.0, "16", True, [16]):
            with self.assertRaisesRegex(patch.PatchError, "rd1"):
                patch.apply(self.image, self.t, self.full(rd1=v))
        out = patch.apply(self.image, self.t, self.full(rd1=np.int32(LO + 1)))      # numpy integers are integers
        self.assertNotEqual(out, self.image)

    def test_a_refused_patch_leaves_the_image_alone(self):
        before = {q: bytes(b) for q, b in self.image.items()}
        with self.assertRaises(patch.PatchError):
            patch.apply(self.image, self.t, self.full(rd0=LO + 1, rd1=HI + 1))
        self.assertEqual({q: bytes(b) for q, b in self.image.items()}, before)

    def test_unknown_and_missing_names(self):
        with self.assertRaisesRegex(patch.PatchError, "nosuch"):
            patch.apply(self.image, self.t, {"nosuch": 1})
        with self.assertRaisesRegex(patch.PatchError, "rd2"):
            patch.apply(self.image, self.t, {"rd0": LO, "rd1": LO, "pm": 1})
        out = patch.apply(self.image, self.t, {"rd0": LO + 1}, strict=False)
        self.assertNotEqual(out, self.image)

    def test_only_the_site_bits_change(self):
        vals = {"rd0": 0x1F, "rd1": 0x13, "rd2": 0x10, "pm": 9}
        out = patch.apply(self.image, self.t, vals)
        self.assertEqual({q: len(b) for q, b in out.items()}, {q: len(b) for q, b in self.image.items()})
        allowed = {q: [0] * (len(b) // 4) for q, b in self.image.items()}
        for i in range(len(self.t.names)):
            for bit in range(self.t.lo[i], self.t.lo[i] + self.t.width[i]):
                u, ub = (self.t.unit[i] + (0 if bit >= 32 else 1), bit % 32) if self.t.nunits[i] == 2 \
                    else (self.t.unit[i], bit)
                allowed[int(self.t.queue[i])][u] |= 1 << ub
        for q, b in out.items():
            for u in range(len(b) // 4):
                x = int.from_bytes(b[4 * u:4 * u + 4], "little") ^ int.from_bytes(self.image[q][4 * u:4 * u + 4], "little")
                self.assertEqual(x & ~allowed[q][u], 0, (q, u))

    def test_patched_image_is_the_assembly_of_the_patched_text(self):
        vals = {"rd0": 0x1F, "rd1": 0x13, "rd2": 0x10, "pm": 9}
        out = patch.apply(self.image, self.t, vals)
        self.assertEqual(patch.to_units(out), patch.lower(self.view, self.p, vals))

    def test_the_source_image_is_not_modified(self):
        before = {q: bytes(b) for q, b in self.image.items()}
        patch.apply(self.image, self.t, {"rd0": HI}, strict=False)
        self.assertEqual({q: bytes(b) for q, b in self.image.items()}, before)

    def test_a_table_whose_field_reaches_the_opcode_is_refused(self):
        lo = np.array(self.t.lo)
        lo[0] = 54            # bits 54..66 would cross the opcode and the tag
        with self.assertRaises(patch.PatchError):
            patch.Table(self.t.names, self.t.queue, self.t.unit, self.t.nunits, lo, self.t.width, self.t.vmin,
                        self.t.vmax, self.t.placeholder, self.t.fixed)
        lo[0] = 40            # bits 40..52: bits 52..45 of a read are fixed (reserved zero) in its match pattern
        with self.assertRaises(patch.PatchError):
            patch.Table(self.t.names, self.t.queue, self.t.unit, self.t.nunits, lo, self.t.width, self.t.vmin,
                        self.t.vmax, self.t.placeholder, self.t.fixed)

    def test_npz_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "patches.npz")
            patch.save(self.t, path)
            t2 = patch.load(path)
        self.assertEqual(list(t2.names), list(self.t.names))
        for col in ("queue", "unit", "nunits", "lo", "width", "vmin", "vmax", "placeholder", "fixed"):
            np.testing.assert_array_equal(getattr(t2, col), getattr(self.t, col))
        out = patch.apply(self.image, t2, self.full(rd0=LO + 2))
        self.assertEqual(out, patch.apply(self.image, self.t, self.full(rd0=LO + 2)))

    def test_split_of_a_logical_unit_into_image_and_blocks(self):
        self.assertEqual(patch.split_position(5, 10, 8), (None, 5))
        self.assertEqual(patch.split_position(10, 10, 8), (0, 0))
        self.assertEqual(patch.split_position(27, 10, 8), (2, 1))


class PagedApply(unittest.TestCase):
    """The positions of a site in a paged queue: in the initial image or a block, and a two-unit field that
    straddles a block boundary is patched in both blocks."""

    B = 4            # units of a block
    IMAGE = 6        # units of the initial image

    def paged(self, unit):
        t = patch.Table(["pm"], [5], [unit], [2], [30], [4], [0], [15], [3], [0xF << 60])
        return patch.Paged(t, {5: self.IMAGE}, self.B)

    def stored(self, word_map, q=5):
        image = {q: bytearray(b"".join(0x10000000.to_bytes(4, "little") for _ in range(self.IMAGE)))}
        blocks = {(q, j): bytearray(b"".join((0x20000000 + 16 * j + i).to_bytes(4, "little") for i in range(self.B)))
                  for j in range(3)}
        return image, blocks

    def test_where_maps_each_unit_of_the_site(self):
        self.assertEqual(self.paged(2).where(0), [(None, 2), (None, 3)])
        self.assertEqual(self.paged(5).where(0), [(None, 5), (0, 0)])
        self.assertEqual(self.paged(6 + 3).where(0), [(0, 3), (1, 0)])
        self.assertEqual(self.paged(6 + 6).where(0), [(1, 2), (1, 3)])

    def test_a_field_across_two_blocks_is_patched_in_both(self):
        paged = self.paged(6 + 3)
        image, blocks = self.stored(None)
        im2, bl2 = patch.apply_paged(image, blocks, paged, {"pm": 9})
        unit = lambda b, k: int.from_bytes(b[4 * k:4 * k + 4], "little")
        # bits 33..30 of the word (first unit << 32 | second unit): bits 1..0 of the first, 31..30 of the second
        first, second = unit(bl2[(5, 0)], 3), unit(bl2[(5, 1)], 0)
        self.assertEqual(((first & 3) << 2) | (second >> 30), 9)
        self.assertEqual((first & ~3, second & 0x3FFFFFFF), (0x20000003 & ~3, 0x20000010))
        self.assertEqual(im2, image)
        self.assertEqual([k for k in bl2 if bl2[k] != blocks[k]], [(5, 0), (5, 1)])
        self.assertEqual(blocks[(5, 0)], self.stored(None)[1][(5, 0)])          # the source is not modified

    def test_a_block_of_the_wrong_size_or_an_image_of_another_length_is_refused(self):
        paged = self.paged(2)
        image, blocks = self.stored(None)
        blocks[(5, 1)] = blocks[(5, 1)][:8]
        with self.assertRaisesRegex(patch.PatchError, "block 1"):
            patch.apply_paged(image, blocks, paged, {"pm": 1})
        image, blocks = self.stored(None)
        image[5] += b"\0\0\0\0"
        with self.assertRaisesRegex(patch.PatchError, "initial image"):
            patch.apply_paged(image, blocks, paged, {"pm": 1})


def sim_run(exe, p, images, preload, dump, max_cycles=300):
    text = tsp_as.input_text(images)
    for (s, a), w in preload.items():
        text += f"S {s:x} {a:x} " + " ".join(f"{x:x}" for x in w) + "\n"
    for s, a in dump:
        text += f"D {s:x} {a:x}\n"
    r = subprocess.run([exe, p.path, "--trace", "--lint", "--max-cycles", str(max_cycles)], input=text.encode(),
                       capture_output=True, timeout=300)
    assert r.returncode == 0, (r.returncode, r.stderr.decode()[:300])
    out = {"diags": [], "lints": [], "words": {}}
    for ln in r.stdout.decode().splitlines():
        f = ln.split()
        if ln.startswith("T diag"):
            out["diags"].append((int(f[2]), int(f[4]), f[5]))
        elif ln.startswith("T lint"):
            out["lints"].append((int(f[2]), int(f[4]), f[5]))
        elif ln.startswith("W "):
            out["words"][(int(f[1], 16), int(f[2], 16))] = bytes(int(x, 16) for x in f[3:])
    return out


class RunPatchedReads(unittest.TestCase):
    """The same text, patched with address set A and set B, reads words A and words B."""

    SETS = [(0x10, 0x11, 0x12), (0x1F, 0x13, 0x10), (0x10, 0x10, 0x10), (0x15, 0x1E, 0x17)]

    def run_all(self, exe, name):
        p = tprofile.load(name)
        rng = random.Random(7)
        view = view_of(reads(p))
        t = patch.locate(view, p)
        base = patch.to_image(patch.lower(view, p))
        data = {(SRCS[i], a): bytes(rng.randrange(256) for _ in range(p.vlanes)) for i in range(3)
                for a in range(LO, HI + 1)}
        before = patch.to_units(base)
        for want in self.SETS:
            image = patch.apply(base, t, {f"rd{i}": a for i, a in enumerate(want)})
            units = patch.to_units(image)
            self.assertEqual({q: len(u) for q, u in units.items()}, {q: len(u) for q, u in before.items()})
            out = sim_run(exe, p, units, data, [(DSTS[i], DST_ADDR[i]) for i in range(3)])
            self.assertEqual(out["diags"], [])
            self.assertEqual(out["lints"], [])
            for i in range(3):
                self.assertEqual(out["words"][(DSTS[i], DST_ADDR[i])][:p.vlanes], data[(SRCS[i], want[i])],
                                 f"{name} {want} read {i}")

    def test_on_tsp_sim(self):
        for name in ("ref", "mini"):
            self.run_all(TSP_SIM, name)

    def test_on_the_iss(self):
        if not os.access(TSP_ISS, os.X_OK):
            self.skipTest(f"ISS not built ({TSP_ISS}); cargo build --release -p tsp-iss")
        for name in ("ref", "mini"):
            self.run_all(TSP_ISS, name)


class RunPatchedMap(unittest.TestCase):
    """The Permute's map index is a site: two maps are loaded, and the same text permutes by map 0 or by map 1 as
    patched (the program of sxm/permute_maps, reduced to its two working maps)."""

    SRC, DST = 0x10, 0x20

    def program(self, p, rng):
        n, s = p.vlanes, p.superlanes
        wide = n > 256
        x_slice, map_slice = (8, 12) if p.slices_per_hemisphere > 4 else (0, 1)
        x = np.array([rng.randrange(256) for _ in range(n)], np.uint8)
        maps = [np.array(rng.sample(range(n), n)) for _ in range(2)]
        maps[1][np.array([rng.random() < 0.25 for _ in range(n)])] = 1000       # masked lanes: zero fill
        b = Builder(p, "patch test: a patched permute map")
        preload = {(x_slice, self.SRC): bytes(x)}
        t_load = 45
        for i, m in enumerate(maps):
            planes = [(m & 255).astype(np.uint8)] + ([(m >> 8).astype(np.uint8)] if wide else [])
            for j, pl in enumerate(planes):
                preload[(map_slice + j, self.SRC + i)] = bytes(pl)
            vs = [b.read_for(t_load, "loadmap", 124, [map_slice + j], [self.SRC + i], "W", 8 + j)
                  for j in range(len(planes))]
            mnem = "loadmapwide" if wide else "loadmap"
            b.unit(124, mnem, f"{mnem} {i:#x}, " + ", ".join(f"W:{8 + j}" for j in range(len(planes))), vs)
            t_load += s + 4
        vx = b.read_for(t_load - 4 + s, "permute", 124, [x_slice], [self.SRC], "W", 0)
        d = patch.Decl("pm", "map", 0, 1, 0)
        _, v = b.unit(124, "permute", f"permute W:0, {d.token}, E:4", [vx], dst=("E", 4, 1), site=d)
        b.write(v, [44], [self.DST])
        want = [bytes(np.where(m < n, x[np.minimum(m, n - 1)], 0).astype(np.uint8)) for m in maps]
        return view_of(b), preload, want

    def run_all(self, exe, name):
        p = tprofile.load(name)
        view, preload, want = self.program(p, random.Random(11))
        t = patch.locate(view, p)
        self.assertEqual(list(t.names), ["pm"])
        base = patch.to_image(patch.lower(view, p))
        for idx in (0, 1, 0):
            units = patch.to_units(patch.apply(base, t, {"pm": idx}))
            out = sim_run(exe, p, units, preload, [(44, self.DST)], max_cycles=600)
            self.assertEqual(out["diags"], [], f"{name} map {idx}")
            self.assertEqual(out["lints"], [])
            self.assertEqual(out["words"][(44, self.DST)][:p.vlanes], want[idx], f"{name} map {idx}")
        self.assertNotEqual(want[0], want[1])

    def test_on_tsp_sim(self):
        for name in ("ref", "mini"):
            self.run_all(TSP_SIM, name)

    def test_on_the_iss(self):
        if not os.access(TSP_ISS, os.X_OK):
            self.skipTest(f"ISS not built ({TSP_ISS}); cargo build --release -p tsp-iss")
        for name in ("ref", "mini"):
            self.run_all(TSP_ISS, name)


class OtsAsm(unittest.TestCase):
    def test_both_assemblers_lower_every_placeholder_to_the_same_object(self):
        if not os.access(OTS_ASM, os.X_OK):
            self.skipTest(f"ots-asm not built ({OTS_ASM})")
        p = tprofile.load("ref")
        view = view_of(with_permute(p))
        with tempfile.TemporaryDirectory() as d:
            for vals in ({}, {"rd0": 0, "rd1": 0, "rd2": 0, "pm": 0},
                         {"rd0": 0x1FFF, "rd1": 0x1FFF, "rd2": 0x1FFF, "pm": 15}):
                src = os.path.join(d, "p.ots")
                with open(src, "w") as f:
                    f.write(view.program(p).text(vals))
                a, b = os.path.join(d, "a.elf"), os.path.join(d, "b.elf")
                subprocess.run([OTS_ASM, "asm", "--config", p.path, "-o", a, src], check=True, capture_output=True)
                subprocess.run([sys.executable, os.path.join(tprofile.SPEC, "tools", "tsp_as.py"), "asm", src, "-o", b,
                                "--config", p.path], check=True, capture_output=True)
                with open(a, "rb") as fa, open(b, "rb") as fb:
                    self.assertEqual(fa.read(), fb.read(), vals)


class CheckedPath(unittest.TestCase):
    """`patch.apply_object`: the table of `locate` in the format of `ots-asm patch`, and the command's per-parcel checks
    on what `patch.apply` writes unchecked."""

    def setUp(self):
        self.assertTrue(os.access(OTS_ASM, os.X_OK), f"ots-asm is needed ({OTS_ASM}); cargo build --release -p ots-asm")
        self.p = tprofile.load("ref")

    def assemble(self, view, values=None):
        return assemble_object(self.p, view, values)

    def test_the_table_has_the_commands_columns(self):
        view = view_of(with_permute(self.p))
        rows = patch.asm_table(patch.locate(view, self.p), view.sites)
        self.assertEqual(rows["format"], 1)
        self.assertEqual([(r["name"], r["op"], r["field"], r["kind"]) for r in rows["sites"]],
                         [("rd0", "read", "addr", "sram_addr"), ("rd1", "read", "addr", "sram_addr"),
                          ("rd2", "read", "addr", "sram_addr"), ("pm", "permute", "map", "map_id")])
        self.assertTrue(all(set(r) >= {"name", "queue", "unit", "op", "field", "kind", "min", "max"} for r in rows["sites"]))

    def test_the_patched_object_is_the_object_of_the_patched_text(self):
        """One and two-unit instructions (a read's address, a Permute's map that spans both units), the identity patch, and a
        grid site (stride and span) are written exactly as the assembler would assemble the literal program."""
        view = view_of(with_permute(self.p))
        table, base = patch.locate(view, self.p), self.assemble(view)
        self.assertEqual(patch.apply_object(base, table, view.sites, {}, self.p), base)
        for vals in ({"rd0": 0x1F, "rd1": 0x13, "rd2": 0x10, "pm": 9}, {"rd1": 0x11}, {"pm": 15}):
            self.assertEqual(patch.apply_object(base, table, view.sites, vals, self.p), self.assemble(view, vals), vals)

    def test_apply_and_the_command_write_the_same_bits(self):
        view = view_of(with_permute(self.p))
        table, base = patch.locate(view, self.p), self.assemble(view)
        vals = {"rd0": 0x1F, "rd1": 0x13, "rd2": 0x10, "pm": 9}
        self.assertEqual(patch.to_units(patch.apply(patch.to_image(patch.lower(view, self.p)), table, vals)),
                         patch.lower(view, self.p, vals))
        self.assertEqual(patch.apply_object(base, table, view.sites, vals, self.p), self.assemble(view, vals))

    def test_a_value_outside_its_range_and_an_unknown_site_are_refused_with_the_commands_text(self):
        view = view_of(reads(self.p))
        table, base = patch.locate(view, self.p), self.assemble(view)
        with self.assertRaisesRegex(patch.PatchError, "rd0"):
            patch.apply_object(base, table, view.sites, {"rd0": HI + 1}, self.p)
        with self.assertRaisesRegex(patch.PatchError, "nosuch"):
            patch.apply_object(base, table, view.sites, {"nosuch": 1}, self.p)

    def test_a_read_moved_into_the_bank_of_its_write_is_refused_by_the_command_and_written_by_apply(self):
        """The bypass of assembly.md ("Patching an assembled object"): with a table that lets the read's address into the
        write's bank, `apply` writes an image that fails when the model dispatches it; the command refuses it."""
        view, b1 = Banks().bundle(self.p)
        t = patch.locate(view, self.p)
        wide = patch.Table(t.names, t.queue, t.unit, t.nunits, t.lo, t.width, [0x10, b1], [b1 + 15, b1 + 15], t.placeholder, t.fixed)
        vals = {"r": b1 + 2, "w": b1 + 9}
        image = patch.apply(patch.to_image(patch.lower(view, self.p)), wide, vals)         # no check: the parcel is invalid
        with self.assertRaises(tsp_as.AsmError):                                          # the assembler does not accept the text
            patch.lower(view, self.p, vals)
        self.assertEqual(len(image), 1)
        with self.assertRaisesRegex(patch.PatchError, "OTS-E-MEM-001"):
            patch.apply_object(self.assemble(view), wide, view.sites, vals, self.p)
        ok = patch.apply_object(self.assemble(view), wide, view.sites, {"r": 0x1F, "w": b1 + 9}, self.p)
        self.assertEqual(ok, self.assemble(view, {"r": 0x1F, "w": b1 + 9}))


class CaseRecord(unittest.TestCase):
    def test_a_case_without_sites_is_unchanged_and_with_sites_lists_them(self):
        p = tprofile.load("ref")
        t = patch.locate(view_of(reads(p)), p)
        c0 = emit.case("x/y", p)
        self.assertNotIn("sites", c0)
        c1 = emit.case("x/y", p, sites=t)
        self.assertEqual([s["name"] for s in c1["sites"]], ["rd0", "rd1", "rd2"])
        self.assertEqual(c1["sites"][0], {"name": "rd0", "queue": SRCS[0], "unit": int(t.unit[0]), "nunits": 2,
                                          "lo": ADDR[0], "width": ADDR[1], "min": LO, "max": HI,
                                          "placeholder": BASE[0]})


if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
