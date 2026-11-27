#!/usr/bin/env python3
"""opentsp-arch-spec/tools/perf/derive_peaks.py: peaks, bandwidths, roofline knees and the weight-install bound derived from a profile.

Run: python3 tests/perf/test_derive_peaks.py   (needs target/spec/tsp_sim, or TSP_SIM, for the install-bound agreement tests, and
target/release/tsp-iss with ots-asm for the activity-counter field names: `just perf-model` builds what is missing)."""
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
TOOL = os.path.join(SPEC, "tools", "perf", "derive_peaks.py")
sys.path.insert(0, os.path.join(SPEC, "tools", "perf"))

import derive_peaks as dp  # noqa: E402


def profile(name):
    path = os.path.join(SPEC, "config", name + ".json")
    with open(path) as fh:
        return json.load(fh)


def run_cli(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


class RefPeaks(unittest.TestCase):
    """The derived peaks of `ref` at 900 MHz."""

    def setUp(self):
        self.q = dp.derive(profile("ref"), 900.0)

    def test_int8_tops(self):
        self.assertAlmostEqual(self.q["int8_tops"].value, 737.28, places=2)
        self.assertEqual(f"{self.q['int8_tops'].value:.1f}", "737.3")

    def test_fp16_tflops(self):
        self.assertEqual(f"{self.q['fp16_tflops'].value:.1f}", "184.3")

    def test_fp8_has_the_int8_rate(self):
        self.assertEqual(self.q["fp8_tops"].value, self.q["int8_tops"].value)

    def test_sram_capacity_mib(self):
        self.assertEqual(self.q["sram_capacity_mib"].value, 220.0)
        self.assertEqual(self.q["sram_capacity_bytes"].value, 88 * 8192 * 320)

    def test_sram_bandwidth(self):
        self.assertEqual(f"{self.q['sram_bw_tb_s'].value:.1f}", "50.7")
        self.assertEqual(f"{self.q['sram_bw_tib_s'].value:.1f}", "46.1")

    def test_stream_bandwidth(self):
        self.assertEqual(f"{self.q['stream_bw_tb_s'].value:.2f}", "18.43")

    def test_c2c(self):
        self.assertEqual(f"{self.q['c2c_gb_s_per_link'].value:.1f}", "12.0")
        self.assertEqual(f"{self.q['c2c_gb_s_total'].value:.1f}", "192.0")

    def test_host_link(self):
        self.assertEqual(f"{self.q['host_gb_s'].value:.1f}", "28.8")

    def test_roofline_knees(self):
        self.assertEqual(f"{self.q['knee_int8_sram'].value:.2f}", "14.55")
        self.assertEqual(f"{self.q['knee_fp16_sram'].value:.2f}", "3.64")
        # against the chip-to-chip links the knee is far higher: a model that streams activations over C2C is link-bound
        self.assertGreater(self.q["knee_int8_c2c"].value, 3000)

    def test_install_bound(self):
        # 409,600 weights, 20 issue cycles at 2 x 32 streams x 320 B, MEM d_func 5, 16 gaps from the farthest of the 32
        # nearest slices to the MXM input: the first ABC at cycle 41 (a schedule with the first ABC at 41 runs clean on tsp_sim).
        self.assertEqual(self.q["install_bytes"].value, 409600)
        self.assertEqual(self.q["install_issue_cycles"].value, 20)
        self.assertEqual(self.q["install_transit_cycles"].value, 16)
        self.assertEqual(self.q["install_first_abc_cycle"].value, 41)
        self.assertEqual(self.q["install_last_arrival_cycle"].value, 40)

    def test_install_bound_names_the_weight_install_latency(self):
        # the formula in profile keys: (issues - 1) + mxm_iw.d_func + mem.d_func + transit; 19 + 1 + 5 + 16 on ref
        self.assertIn("timing.mxm_iw.d_func", self.q["install_first_abc_cycle"].formula)

    def test_every_value_is_labelled(self):
        for q in self.q.values():
            self.assertIn("derived", q.label)
            self.assertIn("clock is an assumption", q.label)
            self.assertTrue(q.formula)


class Scaling(unittest.TestCase):
    def test_clock_scales_rates_not_capacity_or_cycles(self):
        a, b = dp.derive(profile("ref"), 900.0), dp.derive(profile("ref"), 1000.0)
        self.assertEqual(f"{b['int8_tops'].value:.1f}", "819.2")
        self.assertAlmostEqual(b["sram_bw_tb_s"].value / a["sram_bw_tb_s"].value, 1000 / 900)
        self.assertEqual(a["sram_capacity_bytes"].value, b["sram_capacity_bytes"].value)
        self.assertEqual(a["install_first_abc_cycle"].value, b["install_first_abc_cycle"].value)
        # the paper's SRAM figure "55 TiB/s" is loose: 51.2 TiB/s at 1 GHz
        self.assertEqual(f"{b['sram_bw_tib_s'].value:.1f}", "51.2")

    def test_mini(self):
        q = dp.derive(profile("mini"), 900.0)
        self.assertAlmostEqual(q["int8_tops"].value, 4 * 16 * 16 * 2 * 0.9e9 / 1e12)
        self.assertEqual(q["sram_capacity_bytes"].value, 88 * 256 * 16)
        # one superlane: 16 B per cycle per link, 24 cycles per vector
        self.assertAlmostEqual(q["c2c_gb_s_per_link"].value, 16 / 24 * 0.9)

    def test_occupancy_halves_the_fp16_rate(self):
        cfg = profile("ref")
        cfg["timing"]["mxm_abc"]["occupancy_f16"] = 2
        q = dp.derive(cfg, 900.0)
        self.assertEqual(f"{q['fp16_tflops'].value:.1f}", "92.2")

    def test_fewer_slices_than_streams_lengthens_the_install(self):
        # test/ref_small: 8 MEM slices per hemisphere, so 8 < 32 readers
        with open(os.path.join(SPEC, "config", "test", "ref_small.json")) as fh:
            q = dp.derive(json.load(fh), 900.0)
        self.assertEqual(q["install_bytes"].value, 4 * 160 * 320)
        self.assertEqual(q["install_issue_cycles"].value, 40)  # ceil(204800 / (2 x 8 x 320))


def with_mxm(**keys):
    """`ref` with some `mxm` keys changed."""
    cfg = profile("ref")
    cfg["mxm"].update(keys)
    return cfg


class MxmStructureKeys(unittest.TestCase):
    """The four MXM structure keys each move their own figure and no other: planes per hemisphere, f16 rows, FP8 rate, accumulator depth."""

    def peaks(self, cfg):
        return dp.derive(cfg, 900.0)

    def test_ref_is_unchanged(self):
        q = self.peaks(profile("ref"))
        self.assertEqual(f"{q['int8_tops'].value:.1f}", "737.3")
        self.assertEqual(f"{q['fp8_tops'].value:.1f}", "737.3")
        self.assertEqual(f"{q['fp16_tflops'].value:.1f}", "184.3")
        self.assertEqual(q["install_bytes"].value, 409600)

    def test_one_plane_halves_the_int8_rate_and_the_install_and_has_no_f16(self):
        q = self.peaks(profile("test/one_plane"))
        self.assertAlmostEqual(q["int8_tops"].value, 368.64, places=2)
        self.assertAlmostEqual(q["fp8_tops"].value, 368.64, places=2)
        self.assertEqual(q["fp16_tflops"].value, 0)
        self.assertEqual(q["install_bytes"].value, 204800)
        # one plane per hemisphere takes 16 streams (an IW carries 16 rows), so 16 readers: 204,800 / (2 x 16 x 320) = 20 issue cycles, as on ref,
        # and the farthest of the 4 nearest quads is 4 quads nearer the MXM input than ref's farthest of 8: transit 12, first ABC 19 + 1 + 5 + 12
        self.assertEqual(q["install_readers"].value, 16)
        self.assertEqual(q["install_issue_cycles"].value, 20)
        self.assertEqual(q["install_transit_cycles"].value, 12)
        self.assertEqual(q["install_first_abc_cycle"].value, 37)
        notes = "\n".join(dp.structure_notes(q))
        self.assertIn("no plane can hold f16", notes)

    def test_the_planes_key_alone_moves_the_rate_of_every_type(self):
        one, two = self.peaks(with_mxm(planes_per_hemisphere=1)), self.peaks(profile("ref"))
        self.assertAlmostEqual(one["int8_tops"].value * 2, two["int8_tops"].value)
        self.assertEqual(one["fp16_tflops"].value, 0)
        self.assertEqual(one["sram_capacity_bytes"].value, two["sram_capacity_bytes"].value)

    def test_fp8_rate_two_doubles_only_the_fp8_peak(self):
        q, r = self.peaks(profile("test/fp8_rate2")), self.peaks(profile("ref"))
        self.assertAlmostEqual(q["fp8_tops"].value, 1474.56, places=2)
        self.assertEqual(q["int8_tops"].value, r["int8_tops"].value)
        self.assertEqual(q["fp16_tflops"].value, r["fp16_tflops"].value)
        self.assertEqual(q["install_bytes"].value, r["install_bytes"].value)
        self.assertIn("mxm.fp8_rate", q["fp8_tops"].formula)

    def test_f16_tandem_doubles_only_the_fp16_peak(self):
        q, r = self.peaks(profile("test/f16_tandem")), self.peaks(profile("ref"))
        self.assertAlmostEqual(q["fp16_tflops"].value, 368.64, places=2)
        self.assertEqual(q["int8_tops"].value, r["int8_tops"].value)
        self.assertEqual(q["fp8_tops"].value, r["fp8_tops"].value)
        self.assertEqual(q["install_bytes"].value, r["install_bytes"].value)
        self.assertIn("mxm.f16_rows", q["fp16_tflops"].formula)

    def test_f16_rows_scale_the_fp16_peak_linearly(self):
        base = self.peaks(profile("ref"))["fp16_tflops"].value
        self.assertAlmostEqual(self.peaks(with_mxm(f16_rows=240))["fp16_tflops"].value, base * 1.5)

    def test_acc_depth_sets_the_accumulator_state_and_no_peak(self):
        q, r = self.peaks(profile("test/acc8")), self.peaks(profile("ref"))
        self.assertEqual(r["mxm_accumulator_bytes"].value, 4 * 2 * 320 * 4)
        self.assertEqual(q["mxm_accumulator_bytes"].value, 4 * 8 * 320 * 4)
        for name in ("int8_tops", "fp8_tops", "fp16_tflops", "install_bytes"):
            self.assertEqual(q[name].value, r[name].value, name)

    def test_the_extremes_of_all_four_keys_together(self):
        cfg = profile("test/mxm_wide")
        v = 16 * cfg["geometry"]["superlanes"]
        q = self.peaks(cfg)
        self.assertAlmostEqual(q["int8_tops"].value, 4 * 16 * v * 2 * 0.9e9 / 1e12)
        self.assertAlmostEqual(q["fp8_tops"].value, 2 * q["int8_tops"].value)
        self.assertAlmostEqual(q["fp16_tflops"].value, 2 * 16 * v * 2 * 0.9e9 / 1e12 / cfg["timing"]["mxm_abc"]["occupancy_f16"])
        self.assertEqual(q["mxm_accumulator_bytes"].value, 4 * 32 * v * 4)

    def test_a_profile_without_a_structure_key_is_refused_naming_it(self):
        for key in ("planes_per_hemisphere", "f16_rows", "fp8_rate", "acc_depth"):
            cfg = profile("ref")
            del cfg["mxm"][key]
            with self.assertRaises(dp.ProfileError) as cm:
                dp.derive(cfg, 900.0)
            self.assertIn(f"mxm.{key}", str(cm.exception))

    def test_the_cli_prints_the_new_peaks(self):
        r = run_cli(os.path.join(SPEC, "config", "test", "fp8_rate2.json"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("1,474.6", r.stdout)
        r = run_cli(os.path.join(SPEC, "config", "test", "one_plane.json"))
        self.assertIn("no plane can hold f16", r.stdout)
        self.assertIn("368.6", r.stdout)
        notes = json.loads(run_cli(os.path.join(SPEC, "config", "test", "one_plane.json"), "--json").stdout)["notes"]
        self.assertTrue(any("no plane can hold f16" in n for n in notes), notes)


class InjectionLimit(unittest.TestCase):
    """A chip injects at most 4 vectors per cycle (4 transmit queues, a Send holds its queue for one cycle): the chip-to-chip total is the
    smaller of the sum of its links and that limit."""

    def peaks(self, cfg):
        return dp.derive(cfg, 900.0)

    def test_ref_is_bound_by_its_links(self):
        q = self.peaks(profile("ref"))
        self.assertEqual(f"{q['c2c_gb_s_total'].value:.1f}", "192.0")
        self.assertEqual(f"{q['c2c_gb_s_injection'].value:.1f}", "1152.0")
        self.assertIn("the links bind", q["c2c_gb_s_total"].formula)

    def test_128_links_are_bound_by_the_injection_limit(self):
        cfg = profile("ref")
        cfg["c2c"]["links"] = 128
        q = self.peaks(cfg)
        self.assertAlmostEqual(q["c2c_gb_s_links"].value, 1536.0)
        self.assertAlmostEqual(q["c2c_gb_s_total"].value, 1152.0)
        self.assertIn("the injection limit binds", q["c2c_gb_s_total"].formula)
        self.assertAlmostEqual(q["knee_int8_c2c"].value, 640.0)

    def test_a_narrow_chip_with_16_links_is_bound_by_the_injection_limit(self):
        cfg = profile("ref")
        cfg["geometry"]["superlanes"] = 1
        cfg["c2c"]["cycles_per_vector"] = 2
        q = self.peaks(cfg)
        self.assertAlmostEqual(q["c2c_gb_s_links"].value, 115.2)
        self.assertAlmostEqual(q["c2c_gb_s_total"].value, 57.6)

    def test_the_crossover_is_at_4_vectors_per_cycle(self):
        cfg = profile("ref")
        cfg["c2c"]["links"] = 96  # 96 links x 1 vector per 24 cycles = 4 vectors per cycle
        q = self.peaks(cfg)
        self.assertAlmostEqual(q["c2c_gb_s_links"].value, q["c2c_gb_s_injection"].value)
        cfg["c2c"]["links"] = 97
        self.assertAlmostEqual(self.peaks(cfg)["c2c_gb_s_total"].value, q["c2c_gb_s_injection"].value)


PROFILES = os.path.join(ROOT, "tests", "perf", "profiles")
SIM = os.environ.get("TSP_SIM", os.path.join(ROOT, "target", "spec", "tsp_sim"))
PUBLISHED = os.path.join(SPEC, "test", "directed", "published_bounds.py")


def measured_first_abc(path):
    """Black box: the cycle of the first ABC that sees every installed weight, as test/directed/published_bounds.py measures it on tsp_sim.
    Returns (the directed test's own bound, the measurement)."""
    if not os.path.exists(SIM):
        raise AssertionError(f"{SIM} is missing: build it with opentsp-arch-spec/tools/build_sim.sh")
    r = subprocess.run([sys.executable, PUBLISHED, SIM, path], capture_output=True, text=True, timeout=900,
                       env={**os.environ, "PUBLISHED_BOUNDS_VARIANT": "1"})
    m = re.search(r"is at cycle (\d+) .*measured (\d+)", r.stdout)
    if not m:
        raise AssertionError(f"published_bounds.py gave no measurement on {path}:\n{r.stdout[-400:]}{r.stderr[-400:]}")
    return int(m.group(1)), int(m.group(2))


class InstallBoundAgreesWithTheMeasurement(unittest.TestCase):
    """derive_peaks.py's install bound is the cycle the directed test measures, on profiles that vary the keys of the formula."""

    def tool_bound(self, path):
        with open(path) as fh:
            return int(dp.derive(json.load(fh), 900.0)["install_first_abc_cycle"].value)

    def agree(self, path):
        directed_bound, measured = measured_first_abc(path)
        tool = self.tool_bound(path)
        self.assertEqual(measured, directed_bound, f"{path}: the directed test's own bound differs from its measurement")
        self.assertEqual(tool, measured, f"{path}: derive_peaks says {tool}, the simulator measures {measured}")
        return tool

    def test_ref(self):
        self.assertEqual(self.agree(os.path.join(SPEC, "config", "ref.json")), 41)

    def test_ref_slow(self):
        self.agree(os.path.join(SPEC, "config", "test", "ref_slow.json"))

    def test_iw3(self):
        # ref with timing.mxm_iw.d_func 3: 43, two cycles above ref
        self.assertEqual(self.agree(os.path.join(PROFILES, "iw3.json")), 43)

    def test_the_fixture_is_ref_with_one_key_changed(self):
        with open(os.path.join(PROFILES, "iw3.json")) as fh:
            iw3 = json.load(fh)
        ref = profile("ref")
        self.assertEqual(iw3["timing"]["mxm_iw"]["d_func"], 3)
        iw3["timing"]["mxm_iw"]["d_func"] = ref["timing"]["mxm_iw"]["d_func"]
        self.assertEqual(iw3, ref)

    def test_the_fixtures_are_what_the_generator_writes(self):
        # tests/perf/make_profiles.py writes them from the ref of this tree, so a key the schema starts to require cannot be missing
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "perf", "make_profiles.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_iw3_mem3(self):
        self.assertEqual(self.agree(os.path.join(PROFILES, "iw3mem3.json")), 41)

    def test_the_formula_alone_on_iw3_mem3(self):
        # no simulator: (issues - 1) + IW + MEM + transit = 19 + 3 + 3 + 16 = 41 (measured once by hand: 41)
        self.assertEqual(self.tool_bound(os.path.join(PROFILES, "iw3mem3.json")), 41)


ASM = os.environ.get("OTS_ASM", os.path.join(ROOT, "target", "release", "ots-asm"))
ISS = os.environ.get("TSP_ISS", os.path.join(ROOT, "target", "release", "tsp-iss"))


class CounterFieldsOfARealRun(unittest.TestCase):
    """Section 8 of performance.md names the fields of `tsp-iss run --stats`: every path it names exists in the report of a real run, and
    the run length it gives is the one the priced energy uses."""

    @classmethod
    def setUpClass(cls):
        for what, path in (("ots-asm", ASM), ("tsp-iss", ISS)):
            if not os.path.exists(path):
                raise AssertionError(f"{path} is missing: build it with `cargo build --release -p ots-asm -p tsp-iss`")
        cls.tmp = tempfile.TemporaryDirectory()
        ref = os.path.join(SPEC, "config", "ref.json")
        obj, stats = os.path.join(cls.tmp.name, "p.elf"), os.path.join(cls.tmp.name, "stats.json")
        src = os.path.join(ROOT, "tests", "programs", "mem", "gather_scatter.ref.ots")
        subprocess.run([ASM, "asm", "--config", ref, "-o", obj, src], check=True, capture_output=True)
        subprocess.run([ISS, "run", "--config", ref, f"--stats={stats}", "--energy", os.path.join(ROOT, "opentsp-iss", "perf", "energy.yaml"),
                        "--clock-mhz", "900", obj], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        with open(stats) as fh:
            cls.doc = json.load(fh)
        with open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")) as fh:
            text = fh.read()
        cls.section = text[text.index("## 8. Reading the activity counters"):]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def resolve(self, path):
        node = self.doc
        for part in path.split("."):
            if part.startswith("<"):  # a placeholder: the operand types the matrix unit counts
                self.assertIsInstance(node, dict, path)
                self.assertEqual(sorted(node), ["f16", "fp8", "i8", "u8"], path)
                node = node["i8"]
            else:
                self.assertIn(part, node, f"{path}: {part} is not a field of the report")
                node = node[part]
        return node

    def test_every_field_path_of_section_8_exists_in_a_real_report(self):
        paths = sorted(set(re.findall(r"`((?:activity|energy)(?:\.[A-Za-z0-9_<>]+)+)`", self.section)))
        self.assertGreaterEqual(len(paths), 8, paths)
        for path in paths:
            self.resolve(path)

    def test_the_per_type_mac_counter_is_the_documented_path_and_sums_to_the_total(self):
        self.assertIn("`activity.mxm.abc_by_type.<type>.macs`", self.section)
        mxm = self.doc["activity"]["mxm"]
        self.assertEqual(mxm["macs"], sum(t["macs"] for t in mxm["abc_by_type"].values()))
        self.assertNotIsInstance(mxm["macs"], dict)  # `activity.mxm.macs` is a total, not a per-type map

    def test_the_run_length_is_the_last_cycle_plus_one(self):
        self.assertIn("`last_cycle` + 1", self.section)
        self.assertEqual(self.doc["energy"]["cycles"], self.doc["last_cycle"] + 1)

    def test_the_other_top_level_names_exist(self):
        for name in ("last_cycle", "activity", "energy"):
            self.assertIn(name, self.doc)


class Document(unittest.TestCase):
    """opentsp-arch-spec/docs/performance.md quotes the derived figures: they must be the tool's output."""

    def setUp(self):
        with open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")) as fh:
            self.text = fh.read()

    def test_the_ref_table_is_the_tool_output(self):
        q = dp.derive(profile("ref"), 900.0)
        for text in (f"{q['int8_tops'].value:.1f} TOPS", f"{q['fp16_tflops'].value:.1f} TFLOPS", "220 MiB",
                     f"{q['sram_bw_tb_s'].value:.2f} TB/s = {q['sram_bw_tib_s'].value:.2f} TiB/s",
                     f"{q['stream_bw_tb_s'].value:.2f} TB/s = {q['stream_bw_tib_s'].value:.2f} TiB/s",
                     f"{q['c2c_gb_s_per_link'].value:.2f} GB/s", f"{q['c2c_gb_s_total'].value:.1f} GB/s",
                     f"{q['host_gb_s'].value:.2f} GB/s", f"{q['knee_int8_sram'].value:.2f} operations per byte",
                     f"{q['knee_fp16_sram'].value:.3f} FLOP per byte", f"{q['knee_int8_stream'].value:.2f} operations per byte",
                     f"{q['knee_int8_c2c'].value:,.0f} operations per byte", f"{q['knee_int8_host'].value:,.0f} operations per byte",
                     f"**first ABC = (issue cycles − 1) + `timing.mxm_iw.d_func` + `timing.mem.d_func` + transit = 19 + 1 + 5 + 16 = {q['install_first_abc_cycle'].value}**"):
            self.assertIn(text, self.text)

    def test_the_peaks_table_rows(self):
        r, m, g = dp.derive(profile("ref"), 900.0), dp.derive(profile("mini"), 900.0), dp.derive(profile("ref"), 1000.0)

        both = lambda q: f"{q['sram_bw_tb_s'].value:.2f} TB/s = {q['sram_bw_tib_s'].value:.2f} TiB/s"
        self.assertIn(f"| INT8 peak | 2 P × R × V × 2 × f | {r['int8_tops'].value:.1f} TOPS | {m['int8_tops'].value:.3f} TOPS | {g['int8_tops'].value:.1f} TOPS |", self.text)
        self.assertIn(f"| FP16 peak | 2 × F × V × 2 × f / `occupancy_f16` (0 when P = 1) | {r['fp16_tflops'].value:.1f} TFLOPS | {m['fp16_tflops'].value:.4f} TFLOPS | {g['fp16_tflops'].value:.1f} TFLOPS |", self.text)
        self.assertIn(f"| {both(r)} | {m['sram_bw_tb_s'].value:.3f} TB/s | {both(g)} |", self.text)
        stream = lambda q: f"{q['stream_bw_tb_s'].value:.2f} TB/s = {q['stream_bw_tib_s'].value:.2f} TiB/s"
        self.assertIn(f"| {stream(r)} | {m['stream_bw_tb_s'].value:.4f} TB/s | {stream(g)} |", self.text)
        self.assertIn(f"| {r['c2c_gb_s_per_link'].value:.2f} GB/s | {m['c2c_gb_s_per_link'].value:.4f} GB/s | {g['c2c_gb_s_per_link'].value:.2f} GB/s |", self.text)
        self.assertIn(f"| {r['c2c_gb_s_total'].value:.1f} GB/s | {m['c2c_gb_s_total'].value:.3f} GB/s | {g['c2c_gb_s_total'].value:.1f} GB/s |", self.text)
        self.assertIn(f"| {r['host_gb_s'].value:.2f} GB/s | {m['host_gb_s'].value:.3f} GB/s | {g['host_gb_s'].value:.2f} GB/s |", self.text)
        self.assertIn(f"| 220 MiB | {m['sram_capacity_mib'].value:.4f} MiB | 220 MiB |", self.text)

    def test_the_mxm_key_table_is_the_tool_output(self):
        for name in ("ref", "test/one_plane", "test/fp8_rate2", "test/f16_tandem", "test/acc8"):
            q = dp.derive(profile(name), 900.0)
            row = (f"| {q['int8_tops'].value:,.1f} TOPS | {q['fp8_tops'].value:,.1f} TOPS | "
                   + (f"{q['fp16_tflops'].value:,.1f} TFLOPS" if q["fp16_tflops"].value else "0 (no plane can hold f16)")
                   + f" | {int(q['install_bytes'].value):,} | {int(q['mxm_accumulator_bytes'].value):,} B |")
            self.assertIn(f"| `{name}` |" if name == "ref" else f"| `{name}` |", self.text)
            self.assertIn(row, self.text, name)

    def test_the_injection_limit_text_is_the_tool_output(self):
        cfg = profile("ref")
        cfg["c2c"]["links"] = 128
        q = dp.derive(cfg, 900.0)
        self.assertIn(f"{q['c2c_gb_s_injection'].value:,.0f} GB/s on `ref`", self.text)
        self.assertIn(f"prints {q['c2c_gb_s_total'].value:,.0f} GB/s, not {q['c2c_gb_s_links'].value:,.0f}", self.text)
        self.assertIn(f"{q['knee_int8_c2c'].value:.0f} operations per byte, not 480", self.text)
        cfg = profile("ref")
        cfg["geometry"]["superlanes"], cfg["c2c"]["cycles_per_vector"] = 1, 2
        q = dp.derive(cfg, 900.0)
        self.assertIn(f"the links sum to {q['c2c_gb_s_links'].value:.1f} GB/s and the chip injects {q['c2c_gb_s_total'].value:.1f} GB/s", self.text)

    def test_the_stated_deviations(self):
        q = dp.derive(profile("ref"), 900.0)
        self.assertEqual(f"{(q['int8_tops'].value / 750 - 1) * 100:.1f}", "-1.7")
        self.assertEqual(f"{(q['fp16_tflops'].value / 188 - 1) * 100:.1f}", "-2.0")
        self.assertEqual(f"{(dp.derive(profile('ref'), 1000.0)['int8_tops'].value / 820 - 1) * 100:.1f}", "-0.1")
        self.assertEqual(f"{(55 / dp.derive(profile('ref'), 1000.0)['sram_bw_tib_s'].value - 1) * 100:.1f}", "7.4")
        self.assertEqual(f"{(q['c2c_gb_s_per_link'].value / 12.5 - 1) * 100:.1f}", "-4.0")
        self.assertEqual(f"{(q['c2c_gb_s_per_link'].value / 15 - 1) * 100:.0f}", "-20")
        self.assertEqual(f"{(q['host_gb_s'].value / (16e9 * 16 / 8 * 128 / 130 / 1e9) - 1) * 100:.1f}", "-8.6")
        for text in ("−1.7 %", "−0.1 %", "−2.0 %", "7.4 %", "−4.0 %", "−20 %", "−8.6 %"):
            self.assertIn(text, self.text)


class LinkRateNotes(unittest.TestCase):
    """The byte rates of the links and the host, and the warning when a profile's link is faster than any published figure."""

    def test_byte_rates_per_cycle(self):
        q = dp.derive(profile("ref"), 900.0)
        self.assertAlmostEqual(q["c2c_bytes_per_cycle_per_link"].value, 320 / 24)
        self.assertAlmostEqual(q["host_bytes_per_cycle"].value, 32.0)
        self.assertFalse(q["c2c_bytes_per_cycle_per_link"].clock_dependent)

    def test_ref_has_no_note(self):
        self.assertEqual(dp.rate_notes(dp.derive(profile("ref"), 900.0)), [])

    def test_a_wide_chip_with_the_ref_cycle_counts_is_noted(self):
        notes = dp.rate_notes(dp.derive(profile("test/sl32"), 900.0))
        text = "\n".join(notes)
        self.assertIn("19.20 GB/s", text)
        self.assertIn("15 GB/s", text)
        self.assertIn("46.08 GB/s", text)
        self.assertIn("31.5 GB/s", text)
        self.assertEqual(len(notes), 2)

    def test_the_cli_prints_the_notes_and_the_json_has_them(self):
        path = os.path.join(SPEC, "config", "test", "sl32.json")
        r = run_cli(path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("note: the link rate of 19.20 GB/s", r.stdout)
        notes = json.loads(run_cli(path, "--json").stdout)["notes"]
        self.assertEqual(len(notes), 2)
        self.assertEqual(json.loads(run_cli(os.path.join(SPEC, "config", "ref.json"), "--json").stdout)["notes"], [])

    def test_the_rate_is_per_cycle_so_a_slower_clock_does_not_trigger_it(self):
        self.assertEqual(dp.rate_notes(dp.derive(profile("test/sl32"), 450.0)), [])


class TextClaims(unittest.TestCase):
    """The hand-written statements of performance.md and the timing_bounds block of provenance.yaml that derive from the tools' formulas."""

    def setUp(self):
        with open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")) as fh:
            self.text = fh.read()
        with open(os.path.join(SPEC, "isa-db", "provenance.yaml")) as fh:
            self.prov = fh.read()
        with open(os.path.join(ROOT, "opentsp-iss", "crates", "tsp-iss", "src", "slices", "activity.rs")) as fh:
            self.activity_src = fh.read()

    def test_counters_are_described_as_present_with_their_field_names(self):
        self.assertNotIn("not** in this tree yet", self.text)
        self.assertNotIn("added by a separate change", self.text)
        for text in ("tsp-iss run --stats", "off unless `--stats` or `--energy`", "`activity`", "`activity.mxm.macs`",
                     "`activity.mem.read_bytes`", "`activity.c2c.sent_vectors`", "`activity.active_cycles`", "`last_cycle`"):
            self.assertIn(text, self.text)
        # the field names the section uses exist in the counters' own source
        for field in ("macs", "read_bytes", "write_bytes", "sent_vectors", "received_vectors", "active_cycles", "element_ops", "per_link"):
            self.assertIn(field, self.activity_src)

    def test_the_vxm_row_has_the_group_forms(self):
        self.assertIn(f"| {16 * 320 * 4:,} operations (16 × 320 × 4) at 8 bits; {16 * 320 * 2:,} (16 × 320 × 2) at 16 bits |", self.text)
        row = next(l for l in self.text.splitlines() if "`.sg4` group forms" in l)
        for text in ("derived from `isa.md:40`", "not measured", "restricted to the types and operations that have the group bit"):
            self.assertIn(text, row)

    def test_serving_runs_on_ref_geometry_only(self):
        self.assertIn("run on `ref`-geometry profiles only", self.text)

    def test_the_bounds_wording(self):
        for text in ('exceeds "less than 40" (at most 39) by 2', "measures 39",
                     "(section I)", "(section V)", "the source quotes both figures"):
            self.assertIn(text, self.text)
        self.assertNotIn("(derived, not run)", self.text)
        self.assertNotIn("exceeded by one cycle", self.text)
        self.assertNotIn("not yet run on a variant profile", self.prov)
        self.assertNotIn("exceeded by 1", self.prov)
        for text in ("41 exceeds 'less than 40' (at most 39) by 2 cycles", "sec. I", "sec. V", "< 43 us", "less than 49 us"):
            self.assertIn(text, self.prov)


class PresentSlices(unittest.TestCase):
    def test_absent_slices_have_no_capacity_or_bandwidth(self):
        # test/ref_small: 8 MEM slices per hemisphere exist (topology.md section 1), sram.slices still says 88 queue ids
        with open(os.path.join(SPEC, "config", "test", "ref_small.json")) as fh:
            q = dp.derive(json.load(fh), 900.0)
        self.assertEqual(q["sram_capacity_bytes"].value, 16 * 1024 * 320)
        self.assertEqual(q["sram_present_slices"].value, 16)
        self.assertAlmostEqual(q["sram_bw_tb_s"].value, 16 * 320 * 2 * 0.9e9 / 1e12)


class Cli(unittest.TestCase):
    def test_table_is_labelled_and_names_the_clock(self):
        r = run_cli(os.path.join(SPEC, "config", "ref.json"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("derived; clock is an assumption", r.stdout)
        self.assertIn("900", r.stdout)
        self.assertIn("737.3", r.stdout)
        self.assertIn("184.3", r.stdout)

    def test_clock_argument(self):
        r = run_cli(os.path.join(SPEC, "config", "ref.json"), "--clock-mhz", "1000")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("819.2", r.stdout)

    def test_json(self):
        r = run_cli(os.path.join(SPEC, "config", "ref.json"), "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(d["clock_mhz"], 900.0)
        self.assertEqual(d["quantities"]["install_first_abc_cycle"]["value"], 41)
        self.assertIn("clock is an assumption", d["quantities"]["int8_tops"]["label"])

    def test_clock_is_not_a_schema_key(self):
        with open(os.path.join(SPEC, "config", "schema.json")) as fh:
            self.assertNotIn("clock", fh.read().lower())

    def test_bad_clock_is_refused_with_a_message(self):
        for bad in ("0", "-5", "fast", "nan", "inf"):
            r = run_cli(os.path.join(SPEC, "config", "ref.json"), "--clock-mhz", bad)
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn("--clock-mhz", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_missing_profile_is_refused_with_a_message(self):
        r = run_cli("/nonexistent/profile.json")
        self.assertEqual(r.returncode, 2)
        self.assertIn("/nonexistent/profile.json", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_not_json_and_missing_key(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.json")
            with open(bad, "w") as fh:
                fh.write("{not json")
            r = run_cli(bad)
            self.assertEqual(r.returncode, 2)
            self.assertIn("not valid JSON", r.stderr)
            cfg = profile("ref")
            del cfg["mxm"]["rows"]
            part = os.path.join(d, "part.json")
            with open(part, "w") as fh:
                json.dump(cfg, fh)
            r = run_cli(part)
            self.assertEqual(r.returncode, 2)
            self.assertIn("mxm.rows", r.stderr)
            self.assertNotIn("Traceback", r.stderr)



class ExtensionPeaks(unittest.TestCase):
    """Check the extension parameters for accumulator size, matrix rows, bf16 and bfp8 rates."""

    def test_wide_accumulator_state(self):
        # model: acc (4 B) and acc128 (16 B) per row for each of acc_depth accumulators of each of the 4 planes
        self.assertEqual(dp.derive(profile("test/ref_ext"), 900.0)["mxm_accumulator_bytes"].value, 4 * 2 * 320 * 20)
        self.assertEqual(dp.derive(profile("ref"), 900.0)["mxm_accumulator_bytes"].value, 10240)

    def test_accumulator_state_counts_rows_not_lanes(self):
        # the model's accumulators are vector(mxm_rows, ...): sl32 has 512 lanes and 320 rows
        self.assertEqual(dp.derive(profile("test/sl32"), 900.0)["mxm_accumulator_bytes"].value, 10240)

    def test_bf16_and_bfp8_take_the_f16_rate(self):
        q = dp.derive(profile("test/ref_ext"), 900.0)
        self.assertEqual(q["bfp8_tflops"].value, q["fp16_tflops"].value)
        self.assertEqual(q["bf16_tflops"].value, 0)  # ref_ext declares wide_acc and block_fp8, not bf16
        q = dp.derive(profile("test/ext_bf16"), 900.0)
        self.assertEqual(q["bf16_tflops"].value, q["fp16_tflops"].value)
        self.assertEqual(q["bfp8_tflops"].value, 0)
        q = dp.derive(profile("ref"), 900.0)
        self.assertEqual((q["bf16_tflops"].value, q["bfp8_tflops"].value), (0, 0))

    def test_the_documents_state_the_extension_figures(self):
        perf = (Path(SPEC) / "docs/performance.md").read_text()
        design = (Path(SPEC) / "docs/design-space.md").read_text()
        for text in ("bfp8_tflops", "both types take the capacity (a stream pair per row, `mxm.f16_rows` rows)", "(4 + 16 · [`mxm.acc_bits` = 128])"):
            self.assertIn(text, perf)
        self.assertNotIn("D × V × 4", perf)
        self.assertIn("`mxm.acc_bits` and `extensions`", design)

    def test_the_bfp8_cost_is_stated(self):
        r = run_cli(os.path.join(SPEC, "config", "test", "ref_ext.json"))
        self.assertIn("bfp8 (extension block_fp8) takes the storage and the rate of f16", r.stdout)
        self.assertNotIn("bfp8 (extension", run_cli(os.path.join(SPEC, "config", "ref.json")).stdout)


if __name__ == "__main__":
    unittest.main()
