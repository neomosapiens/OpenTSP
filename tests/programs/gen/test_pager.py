#!/usr/bin/env python3
"""Tests of the instruction pager opentsp_tools.pager.

usage: TSP_SIM=target/spec/tsp_sim python3 tests/programs/gen/test_pager.py

The pager turns a cycle view whose lowered queue images exceed `icu.iq_depth` into a fetch-driven one. The tests
check, on the lowering and on `tsp_sim`: the unpaged object is refused; every paged image fits; following the
fetches returns the original cycle view; the planner's occupancy rule is the model's (a one-cycle-late fetch is an
underflow, reported by `tsp_sim` as well); the planner fails loudly where no legal fetch cycle exists.
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opentsp_tools import patch
from opentsp_tools import pager
from opentsp_tools import profile as tprofile
from opentsp_tools import sched
from opentsp_tools import stations
from opentsp_tools.programs.kernels import Builder, qname
from opentsp_tools.stations import ScheduleError

sys.path.insert(0, os.path.join(tprofile.SPEC, "tools"))
from opentsp_tools import refasm
tsp_as = refasm.load()  # noqa: E402

PROFILES = ["ref", "mini", "test/mid_min", "test/fetch_alt"]
TSP_SIM = os.environ.get("TSP_SIM") or os.path.join(tprofile.ROOT, "target", "spec", "tsp_sim")
TSP_ISS = os.environ.get("TSP_ISS") or os.path.join(tprofile.ROOT, "target", "release", "tsp-iss")


def cap_units(p):
    return p.icu["iq_depth"] // 4


def vxm_builder(p, n, spacing=2, alu=0, t0=60):
    """n `add.i8` on VXM[alu], one every `spacing` cycles; no operand is ever on a stream, so each one reports
    OTS-E-TIM-001 at its execute cycle (dispatch + d_skew): the diagnostics are the schedule."""
    b = Builder(p, "pager test")
    for i in range(n):
        b.raw(t0 + spacing * i, 88 + alu, "add.i8 E:8, E:0, E:4")
    return b


def assemble(view, p):
    return tsp_as.assemble(view.program(p).text(), config=p.path)[1]


def sim_run(view, blocks, p, config=None, max_cycles=20000):
    """Run (view, blocks) on tsp_sim with --trace --lint: {'diags': [(cycle, queue, code)], 'lints': [...]}."""
    config = config or p.path
    isa, prog = tsp_as.assemble(view.program(p).text(), config=config)
    tensors = {n: (tsp_as.RAW_DTYPE, [bytes(w) for w in ws]) for n, ws in pager.tensor_words(blocks).items()}
    text = tsp_as.preload_text(tsp_as.bind_weights(list(prog.externs.values()), tensors))
    text += tsp_as.input_text(prog.images)
    r = subprocess.run([TSP_SIM, config, "--trace", "--lint", "--max-cycles", str(max_cycles)], input=text.encode(),
                       capture_output=True, timeout=900)
    assert r.returncode in (0, 3), r.stderr.decode()[:300]
    out = {"diags": [], "lints": [], "E": None}
    for ln in r.stdout.decode().splitlines():
        f = ln.split()
        if ln.startswith("T diag"):
            out["diags"].append((int(f[2]), int(f[4]), f[5]))
        elif ln.startswith("T lint"):
            out["lints"].append((int(f[2]), int(f[4]), f[5]))
        elif ln.startswith("E "):
            out["E"] = ln
    return out


def flat(parcels):
    return sorted((c, q, w) for q, ps in parcels.items() for c, ws in ps for w in ws)


def raised(streams):
    return sorted((c, q, w) for c, q, w, _ in tsp_as.raise_images(streams))


class Unpaged(unittest.TestCase):
    def test_2000_instruction_queue_is_refused(self):
        """A program that needs paging: 2 000 VXM operations on one queue lower to an image above iq_depth; the
        assembler warns (OTS-E-ASM-003) and the loader refuses the object."""
        p = tprofile.load("ref")
        b = vxm_builder(p, 2000)
        warnings = []
        isa, prog = tsp_as.assemble(b.finish(), config=p.path, warn=warnings.append)
        self.assertEqual(len(warnings), 1)
        self.assertIn("OTS-E-ASM-003 (warning)", warnings[0])
        self.assertGreater(len(prog.images[88]), cap_units(p))
        with self.assertRaises(tsp_as.AsmError) as e:
            tsp_as.read_object(tsp_as.write_object(prog, p.path), config=p.path)
        self.assertEqual(e.exception.code, "OTS-E-ASM-003")


class Paged(unittest.TestCase):
    def check(self, p, view, tracker=None):
        tracker = tracker or sched.Tracker(p)
        view2, blocks, externs = pager.page(view, p, tracker)
        tracker.check()
        prog = assemble(view2, p)
        for q, us in prog.images.items():
            self.assertLessEqual(len(us), cap_units(p), f"queue {q}")
        tsp_as.read_object(tsp_as.write_object(prog, p.path), config=p.path)       # a loader accepts the object
        streams = pager.follow(view2, blocks, p)
        full = pager.CycleView(view.entries + view2.added, view2.externs)
        self.assertEqual(raised(streams), flat(pager.encode(full, p)), "following the fetches returns the cycle view")
        self.assertEqual(pager.iq_findings(view2, blocks, p), [])
        return view2, blocks, externs

    def test_2000_vxm_instructions_on_every_profile(self):
        for name in PROFILES:
            p = tprofile.load(name)
            with self.subTest(profile=name):
                b = vxm_builder(p, 2000)
                view = pager.CycleView.of(b.prog)
                view2, blocks, externs = self.check(p, view, b.t)
                units = p.vlanes // 2
                self.assertTrue(blocks)
                self.assertEqual({blk.queue for blk in blocks}, {88})
                for blk in blocks:
                    self.assertEqual(blk.kind, "ifetch")
                    self.assertEqual(sum(len(t[3]) for t in blk.tensors), 2 * p.vlanes)
                    self.assertTrue(all(len(t[3]) == p.vlanes for t in blk.tensors))
                self.assertEqual(len(blocks), len({blk.cycle for blk in blocks}))
                self.assertEqual([blk.index for blk in blocks], list(range(len(blocks))))
                self.assertEqual(sorted(blk.cycle for blk in blocks), [blk.cycle for blk in blocks])
                self.assertEqual(len(blocks) * units >= 2000 * 2, True)
                names = [n for n, *_ in externs]
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(len(n) <= 15 for n in names))

    def test_a_queue_within_the_image_is_left_alone(self):
        p = tprofile.load("ref")
        b = vxm_builder(p, 100)
        view = pager.CycleView.of(b.prog)
        view2, blocks, externs = pager.page(view, p, b.t)
        self.assertEqual((view2.entries, blocks, externs), (view.entries, [], []))

    def test_padding_only_in_the_last_block(self):
        p = tprofile.load("ref")
        b = vxm_builder(p, 700)
        view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t)
        units = [[int.from_bytes(t[3][k:k + 4], "little") for t in blk.tensors for k in range(0, len(t[3]), 4)]
                 for blk in blocks]
        for u in units[:-1]:
            self.assertNotIn(tsp_as.PADDING, u)
        self.assertEqual(units[-1][-1], tsp_as.PADDING)

    def test_random_programs(self):
        """Random queues of every kind (VXM, MXM, SXM, MEM, bundled MEM parcels), random spacing and gaps: the
        three properties hold on every profile."""
        for name in PROFILES:
            p = tprofile.load(name)
            for seed in range(4):
                rng = random.Random(1000 * seed + len(name))
                with self.subTest(profile=name, seed=seed):
                    b = Builder(p, "random")
                    queues = [88 + rng.randrange(16), 104 + rng.randrange(12), 120, 124] + \
                        [rng.randrange(p.slices_per_hemisphere) for _ in range(2)]
                    for q in dict.fromkeys(queues):
                        t = 50 + rng.randrange(20)
                        for _ in range(rng.randrange(350, 700) if q > 87 else rng.randrange(100, 150)):
                            t += rng.choice([2, 2, 3, 5, 9, 40]) if q > 87 else rng.choice([2, 3, 4, 7])
                            if q <= 87:
                                b.raw(t, q, f"read {rng.randrange(0, 8):#x}, E:0, 0x0", slot="R")
                                if rng.random() < 0.4:
                                    b.raw(t, q, f"write {(1 << p.bank_bit) + 8 + rng.randrange(8):#x}, E:1, 0x0", slot="W")
                            elif 88 <= q <= 103:
                                b.raw(t, q, "add.i8 E:8, E:0, E:4")
                            else:
                                b.raw(t, q, f"setvl {rng.randrange(1, p.superlanes + 1):#x}")
                    self.check(p, pager.CycleView.of(b.prog), b.t)


class Placement(unittest.TestCase):
    def test_second_local_fetch_read_can_share_an_opposite_bank_write(self):
        p = tprofile.load("mini")
        for write_addr, legal in ((0, True), (p.words // 2, False)):
            with self.subTest(write_addr=write_addr):
                b = Builder(p, "two-cycle fetch bank arbitration")
                b.raw(101, 4, f"write {write_addr:#x}, E:0, 0x0", slot="W")
                b.t.bank(4, 101, "W", write_addr)
                planner = pager._Pager(pager.CycleView.of(b.prog), p, b.t, None, False)
                self.assertEqual(planner.mem_plan(4, 100, 0) is not None, legal)
                self.assertIsNone(planner.mem_plan(4, 101, 0), "first cycle has an existing parcel")

    def test_fetch_reads_meet_the_unit_on_its_gap(self):
        """Each ifetch of a VXM queue names one stream operand and has two Reads of text on two slices of one quad, the
        first timed by the scheduling equation and the second one cycle later (the stream carries one vector per cycle;
        cu_insts.sail IFETCH), both onto that stream, free along its diagonal (`Tracker.check` sees no foreign producer)."""
        p = tprofile.load("ref")
        b = vxm_builder(p, 900)
        view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t)
        by_cycle = {}
        for c, q, text in view2.added:
            by_cycle.setdefault(c, []).append((q, text))
        d_func, d_skew = p.timing("read")[0], p.timing("ifetch")[1]
        fetches = [(c, text) for c, q, text in view2.added if q == 88]
        self.assertEqual(len(fetches), len(blocks))
        for (t, text), blk in zip(fetches, blocks):
            stream = int(text.split(":")[1])
            row = text.split()[1].split(":")[0]
            names = {n for n, *_ in blk.tensors}
            reads = [(c, q, tx) for c, q, tx in view2.added if q != 88 and tx.split()[1].rstrip(",") in names]
            self.assertEqual(len(reads), 2)
            for i, ((c, q, tx), (name, s, addr, _)) in enumerate(zip(sorted(reads, key=lambda r: r[1]), blk.tensors)):
                self.assertEqual(q, s)
                self.assertEqual(tx.split()[2], f"{row}:{stream},", "both Reads put their word on the operand's stream")
                g_w = stations.need_gap(p, q, row, "write")
                g_r = stations.need_gap(p, 88, row, "read")
                self.assertEqual(c, t + d_skew - (g_r - g_w) - d_func + i)
        b.t.check()

    def test_ifetchm_stays_out_of_loops(self):
        """A MEM queue running R-path AGT loops: IfetchM goes to idle cycles outside every loop window, alone in
        its parcel, and reads two words of the slice's own SRAM at the block's address."""
        p = tprofile.load("ref")
        b = Builder(p, "mem loops")
        t = 40
        for i in range(60):
            b.raw(t, 8, f"read {i:#x}, E:0, 0x0", slot="R")
            for j, (field, val) in enumerate((("countdown", 2), ("step", 1), ("iterations", 6))):
                b.raw(t + 1 + j, 8, f"{field} {val:#x}", slot="config")
            t += 30
        view = pager.CycleView.of(b.prog)
        view2, blocks, _ = pager.page(view, p, b.t)
        loops = [(40 + 30 * i, 40 + 30 * i + 3 + 5 * 3 + 3) for i in range(60)]
        self.assertTrue(blocks)
        for c, q, text in view2.added:
            self.assertEqual((q, text.split()[0]), (8, "ifetchm"))
            self.assertFalse(any(lo <= c <= hi for lo, hi in loops), f"ifetchm at {c} inside a loop")
            self.assertFalse(any(c == e[0] for e in view.entries if e[1] == 8))
        for blk in blocks:
            name, s, addr, data = blk.tensors[0]
            self.assertEqual((blk.kind, s, len(data), addr % 2), ("ifetchm", 8, 2 * p.vlanes, 0))
        Paged().check(p, view)

    def test_a_busy_stream_pair_is_avoided(self):
        """A foreign producer on the diagonal of the pair the first fetch would use moves the fetch to another pair."""
        p = tprofile.load("ref")
        view = pager.CycleView.of(vxm_builder(p, 600).prog)
        first = [e for e in pager.page(view, p, sched.Tracker(p))[0].added if e[1] == 88][0]
        row, s0 = first[2].split()[1].split(":")
        busy = sched.Tracker(p)
        g_r = stations.need_gap(p, 88, row, "read")
        for s in (int(s0), int(s0) + 1):
            busy.write(row, s, g_r - 1, first[0] - 1, "foreign")
        again = [e for e in pager.page(view, p, busy)[0].added if e[1] == 88][0]
        self.assertNotEqual(first[2], again[2])

    def test_no_idle_cycle_fails_loudly(self):
        """A VXM queue that dispatches every cycle has no cycle for a fetch (rule (c))."""
        p = tprofile.load("ref")
        b = vxm_builder(p, 500, spacing=1)
        with self.assertRaises(ScheduleError) as e:
            pager.page(pager.CycleView.of(b.prog), p, b.t)
        self.assertIn(qname(88), str(e.exception))

    def test_control_channel_cannot_be_paged(self):
        p = tprofile.load("ref")
        b = Builder(p, "cu")
        for i in range(400):
            b.raw(10 + 2 * i, 0xFFFF, "config 0x0")
        with self.assertRaises(ScheduleError) as e:
            pager.page(pager.CycleView.of(b.prog), p, b.t)
        self.assertIn("CU", str(e.exception))

    def test_a_barrier_is_refused(self):
        """After a Sync a parcel waits, so the cycles of the view are not dispatch cycles."""
        p = tprofile.load("ref")
        b = vxm_builder(p, 600)
        b.raw(30, 0xFFFF, "sync")
        with self.assertRaises(ScheduleError) as e:
            pager.page(pager.CycleView.of(b.prog), p, b.t)
        self.assertIn("sync", str(e.exception))


class Constraints(unittest.TestCase):
    def test_every_barrier_is_refused_and_a_caller_with_actual_cycles_may_lift_it(self):
        """Sync, Notify, Deskew and RuntimeDeskew each move the dispatch cycles of their queue (a release cycle, a HAC epoch
        boundary), so the pager refuses all four unless `allow_waits` says the view holds the actual cycles."""
        p = tprofile.load("ref")
        for queue, text in ((0xFFFF, "sync"), (0xFFFF, "notify"), (126, "deskew"), (126, "runtime_deskew 0x1")):
            b = vxm_builder(p, 600)
            b.raw(30, queue, text)
            with self.assertRaises(ScheduleError, msg=text) as e:
                pager.page(pager.CycleView.of(b.prog), p, b.t)
            self.assertIn("after a barrier", str(e.exception), text)
            self.assertIn(text.split()[0], str(e.exception), text)
        b = vxm_builder(p, 600)
        b.raw(30, 0xFFFF, "sync")
        view, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t, allow_waits=True)
        self.assertTrue(blocks)

    def test_the_module_docstring_names_every_constraint_the_pager_enforces(self):
        """The documentation of the constraints is the module's own text: each refusal's subject is in it."""
        doc = pager.__doc__
        for word in ("Barriers", "Sync", "Notify", "Deskew", "RuntimeDeskew", "Repeat", "AGT", "Control channel", "SetVL", "Config",
                     "floor", "allow_waits", "barrier.release_latency", "c2c.hac_period"):
            self.assertIn(word, doc)
        self.assertEqual(sorted(pager.WAITS), ["deskew", "notify", "runtime_deskew", "sync"])

    def test_the_pager_is_the_toolchains(self):
        """The pager and the scheduling model it stands on are `opentsp_tools`, in the toolchain, and are independent of the serving layer."""
        tools = os.path.join(tprofile.ROOT, "opentsp-sw-toolchain", "python", "opentsp_tools")
        self.assertEqual(os.path.dirname(pager.__file__), tools)
        self.assertIs(pager, sys.modules["opentsp_tools.pager"])
        for name in os.listdir(tools):
            if name.endswith(".py"):
                with open(os.path.join(tools, name)) as fh:
                    text = fh.read()
                self.assertNotRegex(text, r"(?m)^\s*(from|import) (tests|opentsp_vllm)\b", name)

    def test_repeat_on_a_paged_queue_is_refused(self):
        """The occupancy plan does not model the head parcel's hold during a Repeat, so a queue that must be paged
        may not repeat (a queue within its image may)."""
        p = tprofile.load("ref")
        b = vxm_builder(p, 600)
        b.raw(50, 88, "repeat 0x3, 0x2")
        with self.assertRaises(ScheduleError) as e:
            pager.page(pager.CycleView.of(b.prog), p, b.t)
        self.assertIn("repeat", str(e.exception))
        self.assertIn(qname(88), str(e.exception))
        short = vxm_builder(p, 100)
        short.raw(50, 88, "repeat 0x3, 0x2")
        view = pager.CycleView.of(short.prog)
        self.assertEqual(pager.page(view, p, short.t)[0].entries, view.entries)

    def test_a_repeat_window_ends_after_its_last_iteration_at_spacing_d_plus_one(self):
        """cu_repeat.yaml: iteration k at p + k (d + 1). `repeat 0x3, 0x2` after an instruction at cycle 10 has its iterations
        at 13, 16 and 19; the queue is busy up to and including 19, so the window is (11, 20)."""
        texts = {10: ["add.i8 s0, s1"], 11: ["repeat 0x3, 0x2"]}
        self.assertEqual(pager._windows(texts, False), [(11, 20, "Q")])

    def test_follow_checks_that_an_ifetch_reads_its_block(self):
        """The Reads that put a block on a stream pair must bind the block's own tensors to the stream the Ifetch
        names (the stream's row and number)."""
        p = tprofile.load("ref")
        b = vxm_builder(p, 600)
        view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t)
        pager.follow(view2, blocks, p)
        flip = lambda es: [(c, q, t.replace(", W:", ", E:") if ", W:" in t else t.replace(", E:", ", W:"))
                           if t.startswith("read tx88b0") else (c, q, t) for c, q, t in es]
        wrong = pager.CycleView(flip(view2.entries), view2.externs)
        wrong.added = flip(view2.added)
        with self.assertRaises(ScheduleError) as e:
            pager.follow(wrong, blocks, p)
        self.assertIn("stream", str(e.exception))


class FetchParameters(unittest.TestCase):
    def test_blocks_follow_the_profile(self):
        """R12: block size, delivery rate and latency are the profile's. The same program pages differently on a
        profile with other fetch values, and each plan is clean under that profile's model."""
        plans = {}
        for name in ("test/mid_min", "test/fetch_alt"):
            p = tprofile.load(name)
            b = vxm_builder(p, 800, spacing=3)
            view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t)
            self.assertEqual(pager.iq_findings(view2, blocks, p), [])
            plans[name] = (len(blocks), [blk.cycle for blk in blocks])
        self.assertNotEqual(plans["test/mid_min"], plans["test/fetch_alt"])


class ModelAgreesWithTspSim(unittest.TestCase):
    def test_plan_is_lint_free_and_dispatches_on_schedule(self):
        for name in ("test/mid_min", "test/fetch_alt", "ref"):
            p = tprofile.load(name)
            b = vxm_builder(p, 450, spacing=2 if name != "ref" else 3)
            view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t)
            out = sim_run(view2, blocks, p)
            self.assertEqual(out["lints"], [], name)
            gap = 2 if name != "ref" else 3
            want = [(60 + gap * i + p.timing("add")[1], 88, "OTS-E-TIM-001") for i in range(450)]
            self.assertEqual([d for d in out["diags"] if d[1] == 88], want, name)

    def test_the_model_reports_what_tsp_sim_reports_when_the_fetch_is_late(self):
        """The plan made for the shipped latency (4) and rate (16 bytes per cycle) runs on a profile with a slower
        fetch: the model and tsp_sim report the same first underflow (OTS-W-IQ-001). Later ones are not comparable:
        the late queue dispatches its ifetches after the planned cycle, off the Read that feeds them, so the text it
        brings in is no longer the plan's."""
        p = tprofile.load("test/mid_min")
        b = vxm_builder(p, 450, spacing=2)
        view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), p, b.t)
        slow = json.load(open(p.path))
        slow["icu"]["fetch_latency"] = 4 + 6
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "slow.json")
            json.dump(slow, open(path, "w"))
            ps = tprofile.Profile.__new__(tprofile.Profile)
            ps.__dict__.update(p.__dict__)
            ps.icu = slow["icu"]
            want = pager.iq_findings(view2, blocks, ps)
            got = sim_run(view2, blocks, p, config=path)["lints"]
        self.assertTrue(want)
        self.assertEqual(want[:1], got[:1])
        self.assertEqual(want[0][2], "OTS-W-IQ-001")

    def test_the_model_reports_the_first_overflow_that_tsp_sim_reports_when_the_fetch_is_early(self):
        """The plan made for a slow fetch (latency 4 + 200) runs on the shipped latency: every block arrives 200 cycles
        before it was planned, the IQ is fuller than planned, and the model and tsp_sim report the same first
        overflow (the diagnostic OTS-E-IQ-002). Later ones are not comparable (a lost unit changes the queue's text)."""
        p = tprofile.load("test/mid_min")
        slow_cfg = json.load(open(p.path))
        slow_cfg["icu"]["fetch_latency"] = 4 + 200
        slow_cfg["icu"]["fetch_staging_depth"] = 63      # about 10 payloads are in flight at 200 cycles of latency
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "slow.json")
            json.dump(slow_cfg, open(path, "w"))
            ps = tprofile.Profile.__new__(tprofile.Profile)
            ps.__dict__.update(p.__dict__)
            ps.icu = slow_cfg["icu"]
            b = vxm_builder(ps, 450, spacing=2)
            view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), ps, b.t)
            want = pager.iq_findings(view2, blocks, p)
            out = sim_run(view2, blocks, p)
            got = [d for d in out["diags"] if d[2] == "OTS-E-IQ-002"]
        self.assertTrue(want)
        self.assertEqual(want[0][2], "OTS-E-IQ-002")
        self.assertEqual(want[:1], got[:1])

    def test_the_model_reports_the_first_staging_overflow_that_tsp_sim_reports(self):
        """The plan made for a profile whose fetch takes 34 cycles (8 payloads of staging) runs on a profile with the same
        latency whose staging FIFO holds one payload: the second fetch finds the first payload still on its way and
        fetches nothing, and the model and tsp_sim report the same first OTS-E-IQ-003 (cu_iq.sail iq_fetch). Later ones are
        not comparable (the queue's text is no longer the plan's)."""
        p = tprofile.load("test/mid_min")
        cfg = json.load(open(p.path))
        cfg["icu"]["fetch_latency"] = 34
        plan = tprofile.Profile.__new__(tprofile.Profile)
        plan.__dict__.update(p.__dict__)
        plan.icu = dict(cfg["icu"])
        b = vxm_builder(plan, 450, spacing=2)
        view2, blocks, _ = pager.page(pager.CycleView.of(b.prog), plan, b.t)
        self.assertEqual(pager.iq_findings(view2, blocks, plan), [])
        cfg["icu"]["fetch_staging_depth"] = 1
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "staging.json")
            json.dump(cfg, open(path, "w"))
            run = tprofile.Profile.__new__(tprofile.Profile)
            run.__dict__.update(p.__dict__)
            run.icu = cfg["icu"]
            want = [f for f in pager.iq_findings(view2, blocks, run) if f[2] == "OTS-E-IQ-003"]
            got = [d for d in sim_run(view2, blocks, p, config=path)["diags"] if d[2] == "OTS-E-IQ-003"]
        self.assertTrue(want)
        self.assertEqual(want[:1], got[:1])


# -- programs with patch sites ----------------------------------------------------------------------------------
SRC_SLICE, DST_SLICE = 8, 52
RD_LO, RD_N = 0x10, 32                    # the read sites' range: bank 0 words RD_LO .. RD_LO + RD_N - 1
WR_LO = 1 << 12                           # the write sites' range: bank 1 words WR_LO .. WR_LO + n - 1


def copy_builder(p, n):
    """n copies of a source word of slice 8 to a word of slice 52, one every 6 cycles. The read address of copy i is the
    patch site rd<i> and its write address the site wr<i> (a pool write window), so both queues carry n sites and, at
    2 units a parcel plus the gap, more text than the IQ holds."""
    b = Builder(p, "paged sites")
    for i in range(n):
        v = b.read(60 + 6 * i, [SRC_SLICE], [RD_LO], "E", 0, patch=[(f"rd{i}", RD_LO, RD_LO + RD_N - 1)])
        b.write(v, [DST_SLICE], [WR_LO + i], patch=[(f"wr{i}", WR_LO, WR_LO + n - 1)], window=True)
    return b


def run_images(exe, p, view2, blocks, images, dump, preload, max_cycles=2000):
    """Run (images, blocks) of a paged program on the simulator at `exe`: {'diags', 'lints', 'words': {(slice, addr): bytes}}.
    `images` {queue: bytearray}; `blocks` the Blocks whose tensors carry the (patched) text."""
    isa, prog = tsp_as.assemble(view2.program(p).text(), config=p.path)
    tensors = {n: (tsp_as.RAW_DTYPE, [bytes(w) for w in ws]) for n, ws in pager.tensor_words(blocks).items()}
    text = tsp_as.preload_text(tsp_as.bind_weights(list(prog.externs.values()), tensors))
    text += tsp_as.input_text(patch.to_units(images))
    for (s, a), w in preload.items():
        text += f"S {s:x} {a:x} " + " ".join(f"{x:x}" for x in w) + "\n"
    for s, a in dump:
        text += f"D {s:x} {a:x}\n"
    r = subprocess.run([exe, p.path, "--trace", "--lint", "--max-cycles", str(max_cycles)], input=text.encode(),
                       capture_output=True, timeout=900)
    assert r.returncode in (0, 3), r.stderr.decode()[:300]
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


class PagedSites(unittest.TestCase):
    N = 150

    def paged(self, name="ref", regions=None):
        p = tprofile.load(name)
        b = copy_builder(p, self.N)
        view = pager.CycleView.of(b.prog)
        view2, blocks, externs = pager.page(view, p, b.t, regions=regions)
        b.t.check()
        return p, view, view2, blocks

    def test_a_paged_program_with_sites_builds_and_its_sites_are_located(self):
        p, view, view2, blocks = self.paged()
        self.assertEqual({blk.queue for blk in blocks}, {SRC_SLICE, DST_SLICE})
        paged = pager.locate(view2, blocks, p)
        t = paged.table
        self.assertEqual(sorted(str(n) for n in t.names), sorted(s.name for s in view.sites))
        prog = tsp_as.assemble(view2.program(p).text(), config=p.path)[1]
        self.assertEqual(paged.image_len, {q: len(prog.images[q]) for q in (SRC_SLICE, DST_SLICE)})
        by_name = {s.name: s for s in view.sites}
        data = {(blk.queue, blk.index): b"".join(x[3] for x in blk.tensors) for blk in blocks}
        for i, name in enumerate(t.names):
            site = by_name[str(name)]
            self.assertEqual(int(t.queue[i]), site.queue)
            units = []
            for blk, u in paged.where(i):
                if blk is None:
                    raw = bytes(patch.to_image({site.queue: prog.images[site.queue]})[site.queue])
                else:
                    raw = data[(site.queue, blk)]
                units.append(int.from_bytes(raw[4 * u:4 * u + 4], "little"))
            word = units[0] if len(units) == 1 else (units[0] << 32) | units[1]
            self.assertEqual((word >> int(t.lo[i])) & ((1 << int(t.width[i])) - 1), site.placeholder, name)
        self.assertTrue(any(blk is not None for i in range(len(t)) for blk, _ in paged.where(i)),
                        "some sites are in text blocks")
        self.assertTrue(any(blk is None for i in range(len(t)) for blk, _ in paged.where(i)),
                        "some sites are in the initial images")

    def test_the_double_assembly_holds_for_paged_programs(self):
        """A site recorded on another instruction than the one that carries its token fails the self-check also after
        paging."""
        p, view, view2, blocks = self.paged()
        wrong = list(view2.full.sites)
        s = wrong[3]
        wrong[3] = type(s)(s.name, s.queue, s.cycle + 6, s.index, s.op, s.field, s.lo, s.hi, s.placeholder, s.stride, s.span)
        view2.full.sites = wrong
        with self.assertRaises(patch.PatchError):
            pager.locate(view2, blocks, p)

    def patched_run(self, exe, name="ref"):
        p, view, view2, blocks = self.paged(name)
        paged = pager.locate(view2, blocks, p)
        prog = tsp_as.assemble(view2.program(p).text(), config=p.path)[1]
        images = patch.to_image(prog.images)
        data = {(blk.queue, blk.index): bytearray(b"".join(x[3] for x in blk.tensors)) for blk in blocks}
        rng = random.Random(5)
        words = {(SRC_SLICE, RD_LO + a): bytes(rng.randrange(256) for _ in range(p.vlanes)) for a in range(RD_N)}
        for trial in range(3):
            reads = [rng.randrange(RD_N) for _ in range(self.N)]
            writes = rng.sample(range(self.N), self.N)
            values = {f"rd{i}": RD_LO + reads[i] for i in range(self.N)}
            values.update({f"wr{i}": WR_LO + writes[i] for i in range(self.N)})
            im2, bl2 = patch.apply_paged(images, data, paged, values)
            blocks2 = pager.with_block_bytes(blocks, {k: bytes(v) for k, v in bl2.items()})
            out = run_images(exe, p, view2, blocks2, im2, [(DST_SLICE, WR_LO + i) for i in range(self.N)], words,
                             max_cycles=60 + 6 * self.N + 400)
            self.assertEqual(out["diags"], [])
            self.assertEqual(out["lints"], [])
            for i in range(self.N):
                self.assertEqual(out["words"][(DST_SLICE, WR_LO + writes[i])][:p.vlanes],
                                 words[(SRC_SLICE, RD_LO + reads[i])], f"trial {trial} copy {i}")

    def test_patched_paged_program_on_tsp_sim(self):
        self.patched_run(TSP_SIM)

    def test_patched_paged_program_on_the_iss(self):
        self.assertTrue(os.access(TSP_ISS, os.X_OK), f"ISS not built ({TSP_ISS})")
        self.patched_run(TSP_ISS)

    def test_text_blocks_stay_in_the_given_regions(self):
        """The blocks of a slice come from its region, from the end of it downwards; a slice without a region has none."""
        regions = {SRC_SLICE: (0x1800, 0x1900), DST_SLICE: (0x1c00, 0x1d00)}
        p, view, view2, blocks = self.paged(regions=regions)
        names = {}
        for n, s, a, size in view2.externs:
            if n.startswith("tx"):
                names.setdefault(s, []).append((a, a + size))
        self.assertEqual(set(names), {SRC_SLICE, DST_SLICE})
        for s, spans in names.items():
            lo, hi = regions[s]
            self.assertTrue(all(lo <= a and e <= hi for a, e in spans), (s, spans))
            self.assertEqual(max(e for _, e in spans), hi)
        pager.locate(view2, blocks, p)
        with self.assertRaises(ScheduleError) as e:
            self.paged(regions={SRC_SLICE: (0x1800, 0x1900)})
        self.assertIn(qname(DST_SLICE), str(e.exception))
        self.assertIn("no text region", str(e.exception))
        with self.assertRaises(ScheduleError) as e:
            self.paged(regions={SRC_SLICE: (0x1800, 0x1900), DST_SLICE: (0x1c00, 0x1c01)})
        self.assertIn("no free word", str(e.exception))

    def test_a_fetch_never_shares_a_cycle_with_a_patched_write(self):
        """A pool write window takes the slice's bank for the cycle. The pager puts an `ifetchm` only in a cycle
        in which its queue dispatches nothing else (the R path is idle and the queue has no parcel), so no `ifetchm` is
        in the bundle of a patched Write, and none is inside the AGT loop of one."""
        p, view, view2, blocks = self.paged()
        writes = {(s.queue, s.cycle) for s in view.sites if s.op == "write"}
        self.assertEqual(len(writes), self.N)
        fetches = [(q, c) for c, q, text in view2.added if text.startswith("ifetchm")]
        self.assertTrue(fetches)
        self.assertEqual(sorted(set(fetches) & writes), [])
        dispatched = {(q, c) for c, q, text in view2.entries + view2.added if not text.startswith("ifetchm")}
        self.assertEqual(sorted(set(fetches) & dispatched), [])


class FetchAgainstWriteWindow(unittest.TestCase):
    def test_a_free_cycle_beside_a_write_is_the_only_one_the_planner_offers(self):
        """A queue that dispatches a patched Write in every cycle but one gets its fetch in that one."""
        p = tprofile.load("ref")
        b = Builder(p, "fetch beside writes")
        for i in range(400):
            if i % 40 != 39:
                v = b.read(60 + i, [SRC_SLICE], [RD_LO], "E", 4 * (i % 8))
                b.write(v, [DST_SLICE], [WR_LO + i % 300], patch=[(f"wr{i}", WR_LO, WR_LO + 299)], window=True)
        view = pager.CycleView.of(b.prog)
        view2, blocks, _ = pager.page(view, p, b.t)
        writes = {(s.queue, s.cycle) for s in view.sites}
        fetches = [(q, c) for c, q, text in view2.added if text.startswith("ifetch")]
        self.assertTrue(fetches)
        self.assertTrue(any(q <= 87 and text.startswith("ifetch ") for _, q, text in view2.added),
                        "a dense MEM queue needs stream fetch when no two-cycle local-fetch gap exists")
        self.assertEqual(sorted(set(fetches) & writes), [])
        pager.locate(view2, blocks, p)


if __name__ == "__main__":
    unittest.main(verbosity=1)
