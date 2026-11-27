#!/usr/bin/env python3
"""Unit tests of the toolchain program generators.

usage: python3 tests/programs/gen/test_program_generators.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools.programs.case import Kit
from opentsp_tools import profile as tprofile
from opentsp_tools.programs import data
from opentsp_tools import stations
from opentsp_tools import sched
from opentsp_tools import emit
from opentsp_tools import build
from opentsp_tools.programs import kernels
from opentsp_tools.programs.kernels import install
from opentsp_tools.programs.kernels.flow import Flow, Vec

PROFILES = ["ref", "mini", "test/mid_min", "test/ref_small", "test/mini_small", "test/ref_slow"]


def expected_gap(p, queue, row, kind):
    """Station assignments from the architecture, expressed independently of encoding.json."""
    L = p.layout
    other = {"E": "W", "W": "E"}
    if queue <= 87:
        s = queue % 44
        if s >= p.slices_per_hemisphere:
            return None
        q = s // 4
        home = "W" if queue < 44 else "E"
        # A west slice writes towards the VXM on row E (upstream gap 18 - q) and is fed by row W (24 + q).
        return L["mem_up_gap_base"] - q if row == other[home] else L["mem_down_gap_base"] + q
    if 88 <= queue <= 103:
        return L["vxm_gap"]
    if 104 <= queue <= 115:
        west = queue < 110
        if kind == "read":
            return L["mxm_in_gap"] if row == ("W" if west else "E") else None
        return L["mxm_acc_gap"] if row == ("E" if west else "W") else None
    if queue in (116, 117, 118, 119, 130, 131, 132, 133):
        return L["sxm_dst_gap"] if row == ("W" if queue < 130 else "E") else None
    if 120 <= queue <= 125 or 134 <= queue <= 139:
        west = queue < 130
        if kind == "read":
            return L["sxm_turn_in_gap"] if row == ("W" if west else "E") else None
        return L["sxm_turn_out_gap"] if row == ("E" if west else "W") else None
    return "skip"


class StationTable(unittest.TestCase):
    def test_every_queue_and_row(self):
        for name in PROFILES:
            p = tprofile.load(name)
            for queue in range(144):
                for row in "EW":
                    for kind in ("read", "write"):
                        want = expected_gap(p, queue, row, kind)
                        if want == "skip":
                            continue
                        got = stations.gap(p, queue, row, kind)
                        self.assertEqual(got, want, (name, queue, row, kind))

    def test_unit_of_queue(self):
        self.assertEqual(stations.unit_of(0), "MEM_W")
        self.assertEqual(stations.unit_of(44), "MEM_E")
        self.assertEqual(stations.unit_of(88), "VXM")
        self.assertEqual(stations.unit_of(0xFFFF), "CU")


class Dispatch(unittest.TestCase):
    def test_assembly_md_example(self):
        # assembly.md sec. 3: reads on slices 4 and 5 at 33 meet the VXM dispatched at 40 (row E, d_skew 2) ...
        p = tprofile.load("ref")
        self.assertEqual(stations.consumer_dispatch(p, 33, "read", 4, "add", 88, "E"), 40)
        self.assertEqual(stations.consumer_dispatch(p, 33, "read", 5, "add", 88, "E"), 40)
        # ... and its result meets the Write on MEM[44] dispatched at 48.
        self.assertEqual(stations.consumer_dispatch(p, 40, "add", 88, "write", 44, "E"), 48)

    def test_upstream_read_is_refused(self):
        p = tprofile.load("ref")
        # The VXM's result on row W flows towards the west slices; slice 44 on row W is upstream of gap 21.
        with self.assertRaises(stations.ScheduleError):
            stations.consumer_dispatch(p, 40, "add", 88, "write", 44, "W")
        # A slice never sees its own write (equal gaps).
        with self.assertRaises(stations.ScheduleError):
            stations.consumer_dispatch(p, 33, "read", 4, "write", 5, "E")

    def test_timing_comes_from_the_profile(self):
        slow = tprofile.load("test/ref_slow")
        self.assertEqual(slow.timing("read"), (7, 0))
        self.assertEqual(slow.timing("add"), (8, 2))
        self.assertEqual(stations.consumer_dispatch(slow, 33, "read", 4, "add", 88, "E"), 33 + 7 + 4 - 2)

    def test_typed_timing_rows_win_over_the_entry_and_the_class_row(self):
        # a `timing.overrides` row keyed `<entry>.<type>` is the row of that element type; test/ref_hw has Max.i8 (3, 2),
        # Max.f32 (7, 2) and the class row 5 for the other types.
        hw = tprofile.load("test/ref_hw")
        self.assertEqual(hw.timing("max", "i8"), (3, 2))
        self.assertEqual(hw.timing("max", "f32"), (7, 2))
        self.assertEqual(hw.timing("max", "f16"), (5, 2))
        self.assertEqual(hw.timing("max", ""), (5, 2))       # "": the entry's own row, no type asked for
        # an entry that has typed rows in the profile is not timed without a type (the class row would be a silent guess)
        with self.assertRaises(tprofile.UntypedTiming) as ctx:
            hw.timing("max")
        self.assertIn("max", str(ctx.exception))
        self.assertIn("element type", str(ctx.exception))
        # an entry without a typed row is, as before
        self.assertEqual(hw.timing("add"), hw.timing("add", ""))
        # the entry's own row covers a type that has no typed row
        self.assertEqual(hw.timing("exp", "f32"), hw.timing("exp"))
        # the commit of a producer follows its typed row
        self.assertEqual(stations.commit(hw, 40, "max", 88, "E", "i8")[1], 43)
        self.assertEqual(stations.commit(hw, 40, "max", 88, "E", "f32")[1], 47)
        self.assertEqual(stations.commit(hw, 40, "max", 88, "E", "")[1], 45)
        self.assertEqual(stations.consumer_dispatch(hw, 40, "max", 88, "write", 44, "E", producer_type="f32"),
                         stations.consumer_dispatch(hw, 40, "max", 88, "write", 44, "E", producer_type="") + 2)

    def test_convert_timing_uses_the_source_type(self):
        # Sail timing_type_of(VXM_CVT(src, dst, ...)) is the source. test/convert_src: Convert.u8 (3, 2), Convert.f32 (7, 2).
        # Taking the last assembly field (the destination) would swap the two rows.
        p = tprofile.load("test/convert_src")
        self.assertEqual(p.timing("convert.u8.f32"), (3, 2))
        self.assertEqual(p.timing("convert.f32.u8"), (7, 2))
        self.assertEqual(p.timing("convert", "u8"), (3, 2))
        self.assertEqual(stations.commit(p, 40, "convert.u8.f32", 88, "E")[1], 43)
        self.assertEqual(stations.commit(p, 40, "convert.f32.u8", 88, "E")[1], 47)


class TypedRowsEverywhere(unittest.TestCase):
    """Every checked-in case generator times each VXM operation with its element type: generated on a profile that gives typed rows to
    every (entry, type) pair the generators use, none asks an entry that has typed rows for its row without a type (`UntypedTiming`).
    The cases that need the reference VXM latency for other reasons are listed: they fail on the plain profile `test/ref_slow` too."""

    NEEDS_REFERENCE_LATENCY = {
        "ship/ts_wide_context16", "ship/ts_wide_context96", "ship/ts_attention1x2",        # chained add geometry of kernels/context.py
        "pager/long_vxm", "pager/long_mem", "pager/long_sxm", "pager/long_mxm",            # table of loop lengths keyed by profile name
    }

    def test_no_generator_asks_for_a_row_without_the_type(self):
        import copy
        import build
        ref = tprofile.load("ref")
        pairs = set()
        real = tprofile.Profile.timing

        def record(self, mnemonic, dtype=None):
            m, d = mnemonic, dtype
            if d is None and "." in m:
                parts = m.split(".")
                m = parts[0]
                at = tprofile._type_field(tprofile.entry(m))
                d = parts[at] if at is not None and at < len(parts) else None
            if d:
                pairs.add((m, d))
            return real(self, mnemonic, dtype)

        with mock.patch.object(tprofile.Profile, "timing", record):
            for name, fn, _, kw in build.registry():
                if name not in self.NEEDS_REFERENCE_LATENCY:
                    fn(ref, **dict(kw))
        self.assertGreater(len(pairs), 40)
        typed = copy.copy(ref)
        typed.cfg = copy.deepcopy(ref.cfg)
        typed.name = "ref"          # the layout and size tables of the cases are keyed on the profile name; the typed rows are in `cfg`
        for m, d in sorted(pairs):
            d_func, d_skew = ref.timing(m, "")
            typed.cfg["timing"]["overrides"].append({"entry": f"{tprofile.entry(m)['name']}.{d}", "d_func": d_func + 2, "d_skew": d_skew + 1})
        untyped = {}
        for name, fn, _, kw in build.registry():
            if name in self.NEEDS_REFERENCE_LATENCY:
                continue
            try:
                fn(typed, **dict(kw))
            except tprofile.UntypedTiming as e:
                untyped[name] = str(e)
        self.assertEqual(untyped, {})


class Tracker(unittest.TestCase):
    def test_collision_is_refused(self):
        p = tprofile.load("ref")
        t = sched.Tracker(p)
        t.write("E", 0, gap=17, cycle=38, tag="a")
        with self.assertRaises(stations.ScheduleError):
            t.write("E", 0, gap=17, cycle=38, tag="b")

    def test_read_resolves_to_its_producer(self):
        p = tprofile.load("ref")
        t = sched.Tracker(p)
        t.write("E", 0, gap=17, cycle=38, tag="x")
        t.read("E", 0, gap=21, cycle=42, want="x")
        t.check()

    def test_upstream_read_is_refused(self):
        p = tprofile.load("ref")
        t = sched.Tracker(p)
        t.write("E", 0, gap=24, cycle=38, tag="x")
        t.read("E", 0, gap=21, cycle=35, want="x")   # same diagonal, but gap 21 is upstream of 24
        with self.assertRaises(stations.ScheduleError):
            t.check()

    def test_intervening_writer_is_refused(self):
        p = tprofile.load("ref")
        t = sched.Tracker(p)
        t.write("E", 0, gap=17, cycle=38, tag="x")
        t.write("E", 0, gap=19, cycle=40, tag="y")   # same diagonal, between the producer and the reader
        t.read("E", 0, gap=21, cycle=42, want="x")
        with self.assertRaises(stations.ScheduleError):
            t.check()

    def test_slot_and_bank_rules(self):
        p = tprofile.load("ref")
        t = sched.Tracker(p)
        t.slot(4, 33, "R", thread=0)
        with self.assertRaises(stations.ScheduleError):
            t.slot(4, 33, "R", thread=0)
        t.slot(4, 33, "R", thread=1)
        t.slot(88, 40, "insn")
        with self.assertRaises(stations.ScheduleError):
            t.slot(88, 40, "insn")
        t.bank(4, 50, "R", 0x10, thread=0)
        with self.assertRaises(stations.ScheduleError):
            t.bank(4, 50, "W", 0x20, thread=0)       # same bank (bit 12 clear on both)
        t.bank(4, 51, "R", 0x10, thread=0)
        t.bank(4, 51, "W", 0x1010, thread=0)         # different banks


class Agt(unittest.TestCase):
    """The worked examples of microarchitecture.md sec. 4 (A = 100, reference 0 at 10)."""

    def refs(self, records):
        return [(c - 10, a - 100) for c, a, _ in kernels.agt_refs(10, 100, 0, records)]

    def test_step(self):
        self.assertEqual(self.refs([("step", 2)]), [(0, 0), (1, 2)])

    def test_countdown(self):
        self.assertEqual(self.refs([("countdown", 1)]), [(0, 0), (2, 0)])

    def test_step_then_countdown_is_two_levels(self):
        self.assertEqual(self.refs([("step", 0), ("countdown", 1)]), [(0, 0), (1, 0), (3, 0), (4, 0)])

    def test_iterations_countdown_step(self):
        self.assertEqual(self.refs([("iterations", 3), ("countdown", 2), ("step", 1)]),
                         [(0, 0), (1, 0), (2, 0), (5, 1), (6, 1), (7, 1)])

    def test_cube(self):
        self.assertEqual([a for _, a in self.refs([("step", 1)] * 3)], list(range(8)))
        with self.assertRaises(stations.ScheduleError):
            self.refs([("step", 1)] * 5)


class Units(unittest.TestCase):
    """MXM and SXM queue names (assembly.md sec. 3 table) and Builder.unit (J1)."""

    def test_queue_names(self):
        for q, name in ((104, "MXM[0].LW"), (105, "MXM[0].ABC"), (106, "MXM[0].ACC"), (110, "MXM[2].LW"),
                        (115, "MXM[3].ACC"), (116, "SXM.W.DST[0]"), (119, "SXM.W.DST[3]"), (120, "SXM.W.TSP[0]"),
                        (122, "SXM.W.SHF[0]"), (123, "SXM.W.SHF[1]"), (124, "SXM.W.PRM"), (125, "SXM.W.SEL"),
                        (130, "SXM.E.DST[0]"), (136, "SXM.E.SHF[0]"), (137, "SXM.E.SHF[1]"), (139, "SXM.E.SEL"),
                        (126, "C2C.W.TX[0]"), (127, "C2C.W.TX[1]"), (128, "C2C.W.RX[0]"), (129, "C2C.W.RX[1]"),
                        (140, "C2C.E.TX[0]"), (141, "C2C.E.TX[1]"), (142, "C2C.E.RX[0]"), (143, "C2C.E.RX[1]")):
            self.assertEqual(kernels.qname(q), name)
        for q in (130 + 14, 144, 200):
            with self.assertRaises(stations.ScheduleError):
                kernels.qname(q)

    def test_unit_meets_operands_and_commits(self):
        p = tprofile.load("ref")
        b = kernels.Builder(p)
        v = b.read(32, [4], [0x10], "E", 0)
        t = b.when(v, "shift", 136)
        cycle, out = b.unit(136, "shift", "shift E:0, up, 0x3, W:4", [v], ("W", 4, 1))
        self.assertEqual(cycle, t)
        self.assertEqual((out.row, out.stream, out.gap, out.cycle), ("W", 4, 4, t + 20))   # sxm_cross d_func 20
        self.assertIn("SXM.E.SHF[0]: shift E:0, up, 0x3, W:4", b.finish())

    def test_unit_refuses_operands_at_two_cycles_and_a_wrong_cycle(self):
        p = tprofile.load("ref")
        b = kernels.Builder(p)
        v0 = b.read(32, [4], [0x10], "E", 0)
        v1 = b.read(33, [5], [0x10], "E", 1)
        with self.assertRaises(stations.ScheduleError):
            b.unit(136, "sxselect", "x", [v0, v1])
        with self.assertRaises(stations.ScheduleError):
            b.unit(136, "shift", "x", [v0], cycle=b.when(v0, "shift", 136) + 1)

    def test_unit_without_sources_takes_its_cycle(self):
        b = kernels.Builder(tprofile.load("ref"))
        self.assertEqual(b.unit(104, "iw", "iw.i8", cycle=70), (70, None))


class Planes(unittest.TestCase):
    def test_sub_value_reads_resolve_to_the_producer_planes(self):
        p = tprofile.load("ref")
        b = kernels.Builder(p)
        v = b._write("E", 0, 16, 4, 100, "t")
        quad = v.planes(8, 4)
        self.assertEqual((quad.stream, quad.width, quad.first), (8, 4, 8))
        b._read(quad, 44, "write", b.when(quad, "write", 44))
        b.finish()
        with self.assertRaises(stations.ScheduleError):
            v.planes(14, 4)

    def test_a_sub_value_read_at_the_wrong_cycle_is_refused(self):
        p = tprofile.load("ref")
        b = kernels.Builder(p)
        v = b._write("E", 0, 16, 4, 100, "t")
        b._read(v.planes(4, 4), 44, "write", b.when(v, "write", 44) + 1)
        with self.assertRaises(stations.ScheduleError):
            b.finish()


def flow(name="ref"):
    k = Kit(tprofile.load(name), "t/flow", "test", 1)
    return k, Flow(k)


def f32_vec(k, scale=1.0):
    v = np.zeros(k.p.vlanes, np.float32)
    v[:8] = (np.arange(8) + 1) * scale
    return v


class FlowScheduling(unittest.TestCase):
    def test_vector_operands_are_read_to_meet_the_operation(self):
        k, f = flow()
        a, b = f.const("a", f32_vec(k), "W", 1), f.const("b", f32_vec(k), "W", 2)
        v = f.vxm(0, "add.f32", [a, b])
        d_func, d_skew = k.p.timing("add")
        t = v.cycle - d_func
        self.assertEqual(t, max(f.bound(a, "add", 88), f.bound(b, "add", 88)))
        text = k.b.finish()
        self.assertIn(f"@cycle={t + d_skew - 5 - (21 - 17)}", text)      # a: Read d_func 5, quad 1 writes at gap 17
        self.assertIn(f"@cycle={t + d_skew - 5 - (21 - 16)}", text)      # b: quad 2 writes at gap 16

    def test_a_vector_is_not_read_before_it_is_ready(self):
        k, f = flow()
        late = Vec([4, 5, 6, 7], 0x80, 100, "W")
        v = f.vxm(0, "add.f32", [late, late])
        read_at = v.cycle - k.p.timing("add")[0] + k.p.timing("add")[1] - 3 - 5
        self.assertGreaterEqual(read_at, 100)

    def test_a_typed_timing_row_times_the_operation_it_names(self):
        # test/ref_hw: Max.i8 (3, 2), Max.f32 (7, 2), class row 5. Same operands, same dispatch; the result commits at the type's d_func.
        hw = tprofile.load("test/ref_hw")
        self.assertEqual(hw.timing("max.i8"), (3, 2))
        self.assertEqual(hw.timing("cmp.lt.f32"), hw.timing("cmp"))
        done = {}
        for t in ("i8", "f32", "f16"):
            k, f = flow("test/ref_hw")
            a, b = f.const("a", f32_vec(k), "W", 1), f.const("b", f32_vec(k), "W", 2)
            done[t] = f.vxm(0, f"max.{t}", [a, b]).cycle
        self.assertEqual(done["f32"] - done["i8"], 7 - 3)
        self.assertEqual(done["f16"] - done["i8"], 5 - 3)
        # a chained consumer is dispatched from that commit
        k, f = flow("test/ref_hw")
        a = f.const("a", f32_vec(k), "W", 1)
        first = f.vxm(0, "max.i8", [a, a])
        second = f.vxm(1, "add.i8", [first, a], chain=(0,))
        d_func, d_skew = hw.timing("add.i8")
        self.assertEqual(second.cycle, first.cycle + 1 - d_skew + d_func)
        # Flow places the result's stream by the commit of the typed row too
        k, f = flow("test/ref_hw")
        a = f.const("a", f32_vec(k), "W", 1)
        with mock.patch.object(stations, "commit", wraps=stations.commit) as commit:
            f.vxm(0, "max.i8", [a, a])
        self.assertIn("max.i8", [c.args[2] for c in commit.call_args_list])
        self.assertNotIn("max", [c.args[2] for c in commit.call_args_list])

    def test_convert_schedule_follows_the_source_type_row(self):
        # Builder.vxm / Flow.vxm pass the full op to Profile.timing; convert.u8.f32 must commit as Convert.u8, not Convert.f32.
        src = tprofile.load("test/convert_src")
        done = {}
        for op in ("convert.u8.f32", "convert.f32.u8"):
            k, f = flow("test/convert_src")
            a = f.const("a", f32_vec(k), "W", 1)
            done[op] = f.vxm(0, op, [a]).cycle
        self.assertEqual(done["convert.f32.u8"] - done["convert.u8.f32"], 7 - 3)
        self.assertEqual(src.timing("convert.u8.f32")[0], 3)
        self.assertEqual(src.timing("convert.f32.u8")[0], 7)

    def test_an_operand_used_twice_is_read_once(self):
        k, f = flow()
        a = f.const("a", f32_vec(k), "W", 1)
        f.vxm(0, "mul.f32", [a, a])
        self.assertEqual(k.b.finish().count("MEM[4]: read"), 1)

    def test_reads_that_share_a_slice_are_staggered_by_a_cycle(self):
        k, f = flow()
        a, b = f.const("a", f32_vec(k), "W", 1), f.const("b", f32_vec(k), "W", 1)
        t = [f.vxm(i, "add.f32", [a, a]).cycle for i in range(2)]
        self.assertEqual(t[1], t[0] + 1)
        self.assertNotEqual(a.addr, b.addr)

    def test_a_chained_operand_fixes_the_cycle_and_a_vector_that_is_not_ready_is_refused(self):
        k, f = flow()
        a = f.const("a", f32_vec(k), "W", 1)
        first = f.vxm(0, "add.f32", [a, a])
        second = f.vxm(1, "mul.f32", [first, a], chain=(0,))
        self.assertEqual(second.cycle - k.p.timing("mul")[0], first.cycle + 1 - k.p.timing("mul")[1])
        with self.assertRaises(stations.ScheduleError):
            f.vxm(2, "mul.f32", [first, Vec([8, 9, 10, 11], 0x90, 10 ** 6, "W")], chain=(0,))

    def test_store_returns_the_vector_ready_after_the_write(self):
        k, f = flow()
        a = f.const("a", f32_vec(k), "W", 1)
        v = f.vxm(0, "add.f32", [a, a])
        want = f32_vec(k) * 2
        out = f.store(v, 2, want)
        self.assertEqual((out.hemi, out.slices, out.ready), ("E", [52, 53, 54, 55], k.last + 1))
        self.assertGreaterEqual(out.addr, 1 << k.p.bank_bit)
        self.assertEqual(k.words[(52, out.addr)], bytes(want.view(np.uint8).reshape(-1, 4)[:, 0]))

    def test_an_accumulator_value_may_be_stored_in_either_hemisphere(self):
        k, f = flow()
        _, acc = k.b.unit(106, "acc", "acc.f32 E:8, replace, 0", cycle=100, dst=("E", 8, 4))
        self.assertEqual(f.store(acc, 1, hemi="W").slices, [4, 5, 6, 7])

    def test_table_binds_one_extern_per_plane_with_a_word_per_vector(self):
        k, f = flow()
        vecs = f.table("t", [f32_vec(k, s) for s in (1, 2, 3)], "E", 4)
        self.assertEqual([v.addr for v in vecs], [vecs[0].addr + i for i in range(3)])
        self.assertEqual(vecs[2].slices, [60, 61, 62, 63])
        self.assertEqual(vecs[2].names, [f"tp{pl}+0x2" for pl in range(4)])
        self.assertEqual(len(k.weights["tp3"][1]), 3 * k.p.vlanes)

    def test_streams_are_taken_in_aligned_rotation(self):
        k, f = flow()
        got = [f.stream("E", w) for w in (1, 4, 2, 4)]
        self.assertEqual(got, [0, 4, 8, 12])
        self.assertEqual({f.stream("W", 4) for _ in range(9)}, set(range(0, 32, 4)))

    def test_stream_for_skips_a_group_another_value_is_committed_on_in_that_cycle(self):
        k, f = flow()
        f.avoid = True
        g_w, c_w = 21, 200
        k.b.t.write("E", 0, g_w, c_w, "other")           # a value placed ahead of the cursor holds E:0 in that cycle
        self.assertEqual(f.stream_for("E", 1, g_w, c_w), 1)
        self.assertEqual(f.stream_for("E", 1, g_w, c_w + 5), 2)       # another cycle: nothing in the way, the rotation goes on

    def test_stream_for_is_the_rotation_unless_asked_to_avoid(self):
        k, f = flow()
        k.b.t.write("E", 0, 21, 200, "other")
        self.assertFalse(f.avoid)
        self.assertEqual(f.stream_for("E", 1, 21, 200), 0)         # the default is the rotation alone: programs that never
        self.assertEqual(f.stream_for("E", 1, 21, 200), 1)         # place work ahead of the cursor keep their text

    def test_stream_for_names_the_failure_when_every_group_is_taken(self):
        k, f = flow()
        f.avoid = True
        for s in range(k.p.streams):
            k.b.t.write("E", s, 21, 200, "other")
        with self.assertRaises(stations.ScheduleError):
            f.stream_for("E", 1, 21, 200)


class Install(unittest.TestCase):
    def test_weight_layout_holds_plane_j_mod_2_of_rows_8i_plus_j_div_2(self):
        w = (np.arange(16 * 5).reshape(16, 5) / 8).astype(np.float16)
        words = install.weight_planes(w, 320)
        planes = [data.to_planes(np.concatenate([row, np.zeros(315, np.float16)])) for row in w]
        for j in range(16):
            for i in range(2):
                self.assertEqual(words[j][i], planes[8 * i + j // 2][j % 2])

    def test_planes_map_to_hemispheres_and_consecutive_queues(self):
        self.assertEqual([install.plane_queues(i) for i in range(4)],
                         [(0, "W", 104), (0, "W", 107), (44, "E", 110), (44, "E", 113)])
        with self.assertRaises(stations.ScheduleError):
            install.plane_queues(4)


class TrackerFree(unittest.TestCase):
    def test_free_follows_the_slot_rule(self):
        t = sched.Tracker(tprofile.load("ref"))
        self.assertTrue(t.free(4, 10, "R"))
        t.slot(4, 10, "R")
        self.assertFalse(t.free(4, 10, "R"))
        self.assertTrue(t.free(4, 10, "W"))
        t.slot(90, 10)
        self.assertFalse(t.free(90, 10))
        self.assertTrue(t.free(90, 11))


class Emit(unittest.TestCase):
    def test_text(self):
        prog = emit.Program(tprofile.load("ref"))
        prog.extern("x", 4, 0x10, 1)
        prog.add(40, "VXM[0]", "add.i8 E:8, E:0, E:4")
        prog.add(33, "MEM[4]", "read x, E:0, 0x0")
        text = prog.text()
        self.assertTrue(text.startswith(".arch opentsp-tsa\n.profile ref\n.extern x, slice=4, addr=0x10, size=0x1\n"))
        self.assertLess(text.index("@cycle=33"), text.index("@cycle=40"))
        self.assertEqual(emit.Program(tprofile.load("test/mid_min")).text().split("\n")[1], ".profile mid_min")


class HarnessInput(unittest.TestCase):
    def test_decimal_case_locations_are_written_in_hexadecimal(self):
        case = {"profile": "ref", "preload": {"44:16": "070a"}, "dump": ["44:16"]}
        builder = build.CaseBuilder(SimpleNamespace(asm="ots-asm"))
        artifacts = (("native.elf", None), ("reference.elf", None))
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(builder, "_assemble", return_value=artifacts), \
                mock.patch.object(build, "sh", return_value=SimpleNamespace(stdout=b"")):
            text, _ = builder.build_input(case, tmp, tmp)
        self.assertEqual(text, "S 2c 10 7 a\nD 2c 10\n")


if __name__ == "__main__":
    unittest.main()
