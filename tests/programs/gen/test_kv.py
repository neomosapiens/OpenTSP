#!/usr/bin/env python3
"""Tests of the paged attention kernels ( sec. 2.4, 4.3):
kv_install, the patched V reads of context and kv_write (opentsp_tools.programs.kernels.kv_install.py, context.py, kv_write.py), their
patch sites, the tracker's pool write-window rule and the host's no-two-lanes-write-one-word rule.

usage: TSP_SIM=target/spec/tsp_sim [TSP_ISS=target/release/tsp-iss] python3 tests/programs/gen/test_kv.py

The kernel cases of tests/programs/model/ts_kv_* (built by kv.py, run by tests/programs/run.py) hold one block table
each. Here every case is built once with its placeholders, its sites are located (patch.locate), and 20 random block
tables (a random number of live blocks, the null block beyond, permuted ids) are patched into the image with the host
patcher and run: zero diagnostics, zero IQ lints, and the dumped words equal the expectations bit-exact reference (`paged_reference`) gives for
that table. The patched image must also equal the lowering of the program generated with that table, so the sites are
complete. `tsp_sim` runs the first table of every case, the ISS all of them; a missing ISS binary fails the test.
"""
import os
import random
import sys
import unittest
import unittest.mock
import warnings

warnings.simplefilter("ignore", ResourceWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kv  # noqa: E402
import model  # noqa: E402
from test_patch import TSP_ISS, TSP_SIM, sim_run, view_of  # noqa: E402
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools import patch
from opentsp_tools import profile as tprofile
from opentsp_tools import sched
from opentsp_tools import stations
from opentsp_tools.programs.kernels import Builder

TABLES = 20


def preload_of(k):
    """The S lines of a case: every extern's words (the case's weights) at the slice and address of the extern."""
    out, vl = {}, k.p.vlanes
    for name, s, a, n in k.b.prog.externs:
        raw = k.weights[name][1]
        for i in range(n):
            out[(s, a + i)] = raw[i * vl:(i + 1) * vl]
    return out


def random_table(rng, C):
    live = rng.randint(1, C // 16)
    return rng.sample(range(1, kv.NB), live), live


def run_table(test, exe, p, base, view_table, built, k):
    """Patch the placeholder image `base` with the sites of the program generated for one table (`built`) and run it."""
    values = {s.name: s.placeholder for s in built.b.prog.sites}
    image = patch.apply(base, view_table, values)
    test.assertEqual(patch.to_units(image), patch.lower(view_of(built.b), p), "the sites do not cover the program")
    out = sim_run(exe, p, patch.to_units(image), preload_of(built), built.dump, max_cycles=built.last + 64)
    test.assertEqual(out["diags"], [])
    test.assertEqual(out["lints"], [])
    for key, want in built.words.items():
        test.assertEqual(out["words"][key][:p.vlanes], want, f"slice {key[0]} word {key[1]:#x}")


class Cases:
    """Shared: build the placeholder program of a case once, then one built program per table."""

    def setUp(self):
        self.p = tprofile.load("ref")

    def sweep(self, make, tables, exes):
        k0 = make(tables[0])
        view = view_of(k0.b)
        t = patch.locate(view, self.p)
        base = patch.to_image(patch.lower(view, self.p))
        for i, tab in enumerate(tables):
            built = make(tab)
            built.b.finish()
            for exe in exes if i == 0 else exes[-1:]:
                run_table(self, exe, self.p, base, t, built, k0)
        return t


def exes():
    """tsp_sim and the ISS: an ISS binary that is missing is a failure, not a quieter run (set TSP_ISS)."""
    if not os.access(TSP_ISS, os.X_OK):
        raise AssertionError(f"no ISS binary at {TSP_ISS} (TSP_ISS): the sweeps run on tsp_sim and on the ISS")
    return [TSP_SIM, TSP_ISS]


class PatchedKernels(Cases, unittest.TestCase):
    def scores_tables(self, C):
        rng = random.Random(C)
        out = []
        for _ in range(TABLES):
            table, live = random_table(rng, C)
            out.append((table, 16 * (live - 1) + rng.randrange(16)))
        return out

    def test_kv_scores_over_random_block_tables(self):
        for C in (64, 128):
            tabs = self.scores_tables(C)
            t = self.sweep(lambda tp, C=C: kv.kv_scores(self.p, C, table=tp[0], pos=tp[1], kit_only=True), tabs, exes())
            self.assertEqual(len(t), 16 * C // 16)       # one site per K slice and block

    def test_kv_context_over_random_block_tables(self):
        for C in (64, 128):
            tabs = self.scores_tables(C)
            t = self.sweep(lambda tp, C=C: kv.kv_context(self.p, C, table=tp[0], pos=tp[1], kit_only=True), tabs, exes())
            self.assertEqual(len(t), 4 * C // 16)        # one site per V slice and block

    def test_kv_write_over_random_lanes(self):
        rng = random.Random(5)
        lanes = [(rng.randrange(1, kv.NB), rng.randrange(16)) for _ in range(TABLES - 2)] + [(None, 0), (kv.NB - 1, 15)]
        t = self.sweep(lambda lane: kv.kv_write(self.p, block=lane[0], offset=lane[1], kit_only=True), lanes, exes())
        self.assertEqual(len(t), 16 + 4)                 # 8 slice pairs x 2 planes, 4 V slices


class Ranges(Cases, unittest.TestCase):
    def test_a_read_site_takes_blocks_up_to_the_last_and_a_write_site_stops_at_the_sink(self):
        geo = kvpool.Geometry(self.p)
        k = kv.kv_scores(self.p, 64, kit_only=True)
        t = patch.locate(view_of(k.b), self.p)
        image = patch.to_image(patch.lower(view_of(k.b), self.p))
        name = str(t.names[0])
        patch.apply(image, t, {name: geo.kaddr(kv.LAYER, 0)}, strict=False)
        patch.apply(image, t, {name: geo.kaddr(kv.LAYER, geo.blocks - 1)}, strict=False)
        with self.assertRaises(patch.PatchError):
            patch.apply(image, t, {name: geo.kaddr(kv.LAYER, geo.sink)}, strict=False)
        w = kv.kv_write(self.p, kit_only=True)
        tw = patch.locate(view_of(w.b), self.p)
        imw = patch.to_image(patch.lower(view_of(w.b), self.p))
        name = str(tw.names[0])
        patch.apply(imw, tw, {name: geo.kaddr(kv.LAYER, geo.sink)}, strict=False)
        with self.assertRaises(patch.PatchError):
            patch.apply(imw, tw, {name: geo.kaddr(kv.LAYER, 0)}, strict=False)     # the null block is never written

    def test_a_k_site_refuses_the_words_of_another_layer_and_the_middle_of_a_block(self):
        geo = kvpool.Geometry(self.p)
        k = kv.kv_scores(self.p, 64, kit_only=True)
        t = patch.locate(view_of(k.b), self.p)
        image = patch.to_image(patch.lower(view_of(k.b), self.p))
        name = str(t.names[0])
        patch.apply(image, t, {name: geo.kaddr(kv.LAYER, 3)}, strict=False)
        for bad in (geo.kaddr(0, 3), geo.kaddr(kv.LAYER, 3) + 1, geo.kaddr(kv.LAYER, 3) - 1):
            with self.assertRaisesRegex(patch.PatchError, "grid"):
                patch.apply(image, t, {name: bad}, strict=False)

    def test_a_k_write_site_takes_the_block_word_or_the_next_and_refuses_another_layer(self):
        geo = kvpool.Geometry(self.p)
        w = kv.kv_write(self.p, kit_only=True)
        tw = patch.locate(view_of(w.b), self.p)
        imw = patch.to_image(patch.lower(view_of(w.b), self.p))
        name = next(str(n) for n in tw.names if str(n).startswith("kw."))
        for ok in (geo.kaddr(kv.LAYER, 3), geo.kaddr(kv.LAYER, 3) + 1, geo.kaddr(kv.LAYER, geo.sink)):
            patch.apply(imw, tw, {name: ok}, strict=False)
        for bad in (geo.kaddr(0, 3), geo.kaddr(0, 3) + 1, geo.kaddr(kv.LAYER, 3) + 2):
            with self.assertRaisesRegex(patch.PatchError, "grid"):
                patch.apply(imw, tw, {name: bad}, strict=False)

    def test_a_v_read_site_refuses_the_middle_of_a_block(self):
        geo = kvpool.Geometry(self.p)
        c = kv.kv_context(self.p, 64, kit_only=True)
        t = patch.locate(view_of(c.b), self.p)
        image = patch.to_image(patch.lower(view_of(c.b), self.p))
        name = str(t.names[0])
        patch.apply(image, t, {name: geo.vaddr(kv.LAYER, 5)}, strict=False)
        with self.assertRaisesRegex(patch.PatchError, "grid"):
            patch.apply(image, t, {name: geo.vaddr(kv.LAYER, 5) + 3}, strict=False)

    def test_the_grid_survives_the_saved_table(self):
        import tempfile
        k = kv.kv_scores(self.p, 64, kit_only=True)
        t = patch.locate(view_of(k.b), self.p)
        self.assertEqual(set(t.stride.tolist()), {2 * kvpool.Geometry(self.p).cfg.layers})
        with tempfile.TemporaryDirectory() as d:
            patch.save(t, os.path.join(d, "t.npz"))
            u = patch.load(os.path.join(d, "t.npz"))
        self.assertEqual(u.stride.tolist(), t.stride.tolist())
        self.assertEqual(u.span.tolist(), t.span.tolist())

    def test_a_declared_grid_must_hold_the_placeholder(self):
        with self.assertRaisesRegex(patch.PatchError, "grid"):
            patch.Decl("x", "addr", 0x1000, 0x1100, 0x1003, 4)

    def test_every_pool_site_is_in_one_bank(self):
        geo = kvpool.Geometry(self.p)
        for lo, hi, *_ in (geo.krange(kv.LAYER), geo.krange(kv.LAYER, True), geo.vrange(kv.LAYER), geo.vrange(kv.LAYER, True)):
            self.assertEqual(self.p.bank(lo), self.p.bank(hi))
        self.assertEqual({self.p.bank(geo.krange(kv.LAYER)[0]), self.p.bank(geo.vrange(kv.LAYER)[0])}, {1})


class WindowRule(unittest.TestCase):
    """Tracker.window: a pool write window takes no other reference of the slice to its bank in that cycle."""

    def setUp(self):
        self.p = tprofile.load("ref")
        self.bank1 = 1 << self.p.bank_bit

    def test_a_read_of_the_pool_bank_in_the_window_is_refused(self):
        t = sched.Tracker(self.p)
        t.window(44, 100, self.bank1 + 5)
        with self.assertRaises(stations.ScheduleError):
            t.bank(44, 100, "R", self.bank1 + 9)

    def test_a_window_over_a_read_of_its_bank_is_refused(self):
        t = sched.Tracker(self.p)
        t.bank(44, 100, "R", self.bank1 + 9)
        with self.assertRaises(stations.ScheduleError):
            t.window(44, 100, self.bank1 + 5)

    def test_a_read_of_the_other_bank_or_another_cycle_or_slice_is_fine(self):
        t = sched.Tracker(self.p)
        t.window(44, 100, self.bank1 + 5)
        t.bank(44, 100, "R", 0x20)
        t.bank(44, 101, "R", self.bank1 + 9)
        t.bank(45, 100, "R", self.bank1 + 9)
        t.window(44, 102, self.bank1 + 5)
        t.bank(44, 102, "W", self.bank1 + 5)

    def test_the_window_also_holds_for_the_other_thread(self):
        t = sched.Tracker(self.p)
        t.window(44, 100, self.bank1 + 5)
        with self.assertRaises(stations.ScheduleError):
            t.bank(44, 100, "R", self.bank1 + 9, thread=1)

    def test_the_builder_refuses_a_pool_write_beside_a_read_of_its_bank(self):
        b = Builder(self.p, "window")
        v = b.read(0, [44], [0x20], "E", 0)
        t = b.when(v, "write", 48)
        b.read(t, [48], [self.bank1 + 3], "E", 4)
        with self.assertRaises(stations.ScheduleError):
            b.write(v, [48], [self.bank1 + 7], window=True)

    def test_the_builder_takes_a_pool_write_beside_a_read_of_the_other_bank(self):
        b = Builder(self.p, "window")
        v = b.read(0, [44], [0x20], "E", 0)
        t = b.when(v, "write", 48)
        b.read(t, [48], [0x40], "E", 4)
        b.write(v, [48], [self.bank1 + 7], window=True)


    def test_the_builder_refuses_a_window_beside_a_thread_1_read_of_its_bank_and_takes_it_without_the_window(self):
        # for thread 0 the window rule equals OTS-E-MEM-001; the other thread's Read is what only the window refuses
        def build(window):
            b = Builder(self.p, "window")
            v = b.read(0, [44], [0x20], "E", 0)
            t = b.when(v, "write", 48)
            b.read(t, [48], [self.bank1 + 3], "E", 4, thread=1)
            b.write(v, [48], [self.bank1 + 7], window=window)
        build(False)
        with self.assertRaises(stations.ScheduleError):
            build(True)

    def test_the_builder_refuses_a_thread_1_read_after_a_window(self):
        b = Builder(self.p, "window")
        v = b.read(0, [44], [0x20], "E", 0)
        t = b.when(v, "write", 48)
        b.write(v, [48], [self.bank1 + 7], window=True)
        with self.assertRaises(stations.ScheduleError):
            b.read(t, [48], [self.bank1 + 3], "E", 4, thread=1)

    def test_every_write_of_the_kv_write_kit_is_a_window_of_the_pool_bank(self):
        k = kv.kv_write(self.p, kit_only=True)
        sites = [s for s in k.b.prog.sites]
        self.assertEqual(len(sites), 20)
        self.assertEqual(len(k.b.t.windows), 20)           # 16 K slices and 4 V slices, one Write each
        self.assertEqual(set(k.b.t.windows.values()), {1})
        self.assertEqual({q for q, _ in k.b.t.windows}, set(range(44, 60)) | set(kvpool.Geometry(self.p).vslices(kv.LAYER)))


class Composition(Cases, unittest.TestCase):
    def test_read_v_blocks_refuses_a_row_that_has_streams_allocated(self):
        from opentsp_tools.programs.kernels import context as k_context
        from opentsp_tools.programs.kernels.flow import Flow
        geo = kvpool.Geometry(self.p)
        k = kv.kit(self.p, "x", "x", 1)
        f = Flow(k)
        f.const("a", __import__("numpy").zeros(self.p.vlanes, "float32"), "E", 1)
        f.vxm(0, "add.f32", [f.const("b", __import__("numpy").zeros(self.p.vlanes, "float32"), "E", 1)] * 2)
        self.assertGreater(f.streams["E"], f.low["E"])
        with self.assertRaisesRegex(stations.ScheduleError, "before the Flow allocates"):
            k_context.read_v_blocks(f, 30, [1], lambda b: geo.vaddr(kv.LAYER, b), geo.vslices(kv.LAYER), geo.vrange(kv.LAYER))

    def test_a_pool_write_beside_a_static_bank_0_read_is_located_by_the_fallback_and_patches_exactly(self):
        # : "weight reads (bank 0) may share a cycle with K-pool writes (bank 1)": the all-zero assembly of the
        # write site is then a bank 0 Write beside a bank 0 Read (OTS-E-MEM-001), so `locate` takes its bank-bit fallback
        k = kv.kv_write(self.p, kit_only=True)
        (q, c), = [kc for kc in sorted(k.b.t.windows) if kc[0] == 44]
        k.b.read(c, [q], [0x20], "E", 31)
        k.b.finish()
        view = view_of(k.b)
        with unittest.mock.patch.object(patch, "_flipped_banks", wraps=patch._flipped_banks) as flipped:
            t = patch.locate(view, self.p)
        self.assertEqual(flipped.call_count, 1)
        self.assertEqual(len(t), 20)
        geo = kvpool.Geometry(self.p)
        base = patch.to_image(patch.lower(view, self.p))
        values = {s.name: s.placeholder for s in k.b.prog.sites}
        name = next(s.name for s in k.b.prog.sites if s.queue == 44 and s.cycle == c)
        values[name] = geo.kaddr(kv.LAYER, geo.sink)
        image = patch.apply(base, t, values)
        self.assertEqual(patch.to_units(image), patch.lower(view, self.p, values))
        for exe in exes():
            out = sim_run(exe, self.p, patch.to_units(image), preload_of(k), k.dump, max_cycles=k.last + 64)
            self.assertEqual(out["diags"], [])
            self.assertEqual(out["lints"], [])


class PoolText(unittest.TestCase):
    """I-3: the serve planner keeps the text of the K slices out of the K pool's bank; for the V slices it cannot."""

    def plans(self):
        L, _ = kvpool.serve()
        prof = L.load_profile("ref")
        return [(n, L.plan_serve(L.serve_config(n), prof)) for n in (2, 8)]

    def test_the_k_pool_never_shares_a_bank_with_the_text_of_its_slices(self):
        L, _ = kvpool.serve()
        for n, plan in self.plans():
            self.assertEqual([o for o in L.pool_text_overlaps(plan) if o[0] == "kp"], [], n)

    def test_the_v_pools_of_odd_layers_share_the_bank_of_the_text_and_those_of_even_layers_do_not(self):
        L, _ = kvpool.serve()
        for n, plan in self.plans():
            got = {o[0] for o in L.pool_text_overlaps(plan)}
            self.assertEqual(got, {f"vp{l}" for l in range(1, n, 2)}, n)
            for o in L.pool_text_overlaps(plan):
                self.assertEqual(plan.by_name[o[0]].bank, plan.by_name[o[1]].bank)

    def test_every_v_slice_has_a_pool_in_each_bank_so_the_text_always_meets_one(self):
        for n, plan in self.plans():
            for l in range(0, n, 2):
                even, odd = plan.by_name[f"vp{l}"], plan.by_name[f"vp{l + 1}"]
                self.assertEqual(even.slices, odd.slices)
                self.assertEqual({even.bank, odd.bank}, {0, 1})


class Trees(unittest.TestCase):
    """I-2: the two-round (segment 4) and three-round (segment 8) trees in one program, by `base` of load_maps."""

    def setUp(self):
        self.p = tprofile.load("ref")

    def test_a_layernorm_and_a_128_slot_softmax_share_one_program_and_are_reference_exact(self):
        built = model.trees(self.p, kit_only=True)
        built.b.finish()
        ids = sorted({int(t.split()[1].rstrip(","), 16) for bs in built.b.prog.bundles.values() for _, t in bs
                      if t.startswith("loadmap")})
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6])        # six of the 16 map ids; the Distribute map 3 is shared
        for exe in exes():
            out = sim_run(exe, self.p, patch.lower(view_of(built.b), self.p), preload_of(built), built.dump,
                          max_cycles=built.last + 64)
            self.assertEqual(out["diags"], [])
            self.assertEqual(out["lints"], [])
            for key, want in built.words.items():
                self.assertEqual(out["words"][key][:self.p.vlanes], want, f"slice {key[0]} word {key[1]:#x}")

    def test_the_default_base_keeps_the_ids_words_and_names_of_the_two_round_tree(self):
        from opentsp_tools.programs.kernels import reduce_bcast as rb
        k = model.kit(self.p, "x", "x", 1)
        rb.load_maps(k, 60)
        self.assertEqual([e[0] for e in k.b.prog.externs], ["m00", "m01", "m10", "m11"][:len(k.b.prog.externs)])
        self.assertEqual(sorted(e[2] for e in k.b.prog.externs), [rb.MAP_WORD, rb.MAP_WORD, rb.MAP_WORD + 1, rb.MAP_WORD + 1])


class WriteSet(unittest.TestCase):
    def setUp(self):
        self.p = tprofile.load("ref")
        self.geo = kvpool.Geometry(self.p)

    def test_lanes_writing_other_rows_are_accepted_and_padding_lanes_write_the_sink(self):
        words = kvpool.check_writes([(3, 0), (3, 1), (4, 0), None, None], self.geo, kv.LAYER)
        self.assertEqual(len(words), 3 * 6)

    def test_two_lanes_writing_one_row_are_refused(self):
        with self.assertRaisesRegex(stations.ScheduleError, "lanes 0 and 2"):
            kvpool.check_writes([(3, 5), (3, 6), (3, 5)], self.geo, kv.LAYER)

    def test_two_rows_of_one_slice_pair_are_different_words(self):
        # rows 3 and 11 of a block are in slice pair 3, words 0 and 1: no collision
        kvpool.check_writes([(3, 3), (3, 11)], self.geo, kv.LAYER)

    def test_a_real_lane_may_not_write_the_null_block_or_the_sink(self):
        for b in (0, self.geo.sink):
            with self.assertRaises(stations.ScheduleError):
                kvpool.check_writes([(b, 0)], self.geo, kv.LAYER)


if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
