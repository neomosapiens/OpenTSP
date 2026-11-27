#!/usr/bin/env python3
"""opentsp-arch-spec/tools/perf/derive_profile.py: a schema-valid profile from the `ref` template by stated, overridable rules
.

Run: python3 tests/perf/test_derive_profile.py   (needs target/spec/tsp_sim, or TSP_SIM, to check that the harness accepts what
the tool derives: build it with opentsp-arch-spec/tools/build_sim.sh)."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
TOOLS = os.path.join(SPEC, "tools")
TOOL = os.path.join(TOOLS, "perf", "derive_profile.py")
SIM = os.environ.get("TSP_SIM", os.path.join(ROOT, "target", "spec", "tsp_sim"))
sys.path.insert(0, os.path.join(TOOLS, "perf"))
sys.path.insert(0, TOOLS)

import derive_peaks  # noqa: E402
import derive_profile as dpr  # noqa: E402
from opentsp_reference import schema as schema_invariants  # noqa: E402


with open(os.path.join(SPEC, "config", "schema.json")) as _fh:
    _schema = json.load(_fh)["properties"]
SCHEMA_MAX_SUPERLANES = _schema["geometry"]["properties"]["superlanes"]["maximum"]
SCHEMA_MAX_ROWS = _schema["mxm"]["properties"]["rows"]["maximum"]


def ref():
    with open(os.path.join(SPEC, "config", "ref.json")) as fh:
        return json.load(fh)


def derive(*args, check=True):
    r = subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stderr
    return r


def sim_accepts(profile):
    """tsp_sim loads the profile (schema, the model's own checks) and runs an empty program: exit 0 and the end-of-run line."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.json")
        with open(path, "w") as fh:
            json.dump(profile, fh)
        return subprocess.run([SIM, path], input="", capture_output=True, text=True)


class Identity(unittest.TestCase):
    def test_the_ref_geometry_reproduces_ref_exactly(self):
        r = derive("--superlanes", "20", "--rows", "320", "--slices-per-hemisphere", "44")
        self.assertEqual(json.loads(r.stdout), ref())

    def test_defaults_come_from_the_template(self):
        self.assertEqual(json.loads(derive().stdout), ref())


class Derived(unittest.TestCase):
    def check_valid(self, profile):
        self.assertEqual(schema_invariants.check(profile), [])
        self.assertEqual(dpr.schema_errors(profile), [])
        r = sim_accepts(profile)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("E - -1 0", r.stdout)
        derive_peaks.derive(profile, 900.0)  # a derived profile is readable by the peaks tool

    def test_four_superlanes_eight_slices(self):
        r = derive("--superlanes", "4", "--rows", "64", "--slices-per-hemisphere", "8")
        p = json.loads(r.stdout)
        self.assertEqual(p["geometry"]["superlanes"], 4)
        self.assertEqual(p["mxm"]["rows"], 64)
        self.assertEqual(p["geometry"]["mem_slices_per_hemisphere"], 8)
        # the stated rules: sxm_map_ready = mxm_abc.d_func - d_skew = sxm_cross.d_func - d_skew = superlanes
        t = p["timing"]
        self.assertEqual(t["sxm_map_ready"], 4)
        self.assertEqual(t["mxm_abc"]["d_func"] - t["mxm_abc"]["d_skew"], 4)
        self.assertEqual(t["sxm_cross"]["d_func"] - t["sxm_cross"]["d_skew"], 4)
        permute = next(o for o in t["overrides"] if o["entry"] == "Permute")
        self.assertEqual(permute["d_func"] - permute["d_skew"], 4)
        # gaps from the quad count: 2 quads instead of 11, so the units outside the MEM quads keep their distance to them
        self.assertEqual(p["layout"]["mxm_in_gap"], 43 - 9)
        self.assertEqual(p["layout"]["c2c_tx_gap"], 38 - 9)
        self.assertEqual(p["layout"]["c2c_rx_gap"], 5 + 9)
        self.assertEqual(p["layout"]["mxm_acc_gap"], 2 + 9)
        self.assertEqual(p["layout"]["vxm_gap"], 21)
        self.check_valid(p)

    def test_one_superlane(self):
        p = json.loads(derive("--superlanes", "1", "--rows", "16", "--slices-per-hemisphere", "44").stdout)
        self.assertEqual(p["timing"]["sxm_map_ready"], 1)
        self.assertEqual(p["timing"]["mxm_abc"]["d_func"], 1)
        self.check_valid(p)

    def test_two_superlanes_and_default_rows(self):
        p = json.loads(derive("--superlanes", "2", "--slices-per-hemisphere", "12").stdout)
        self.assertEqual(p["mxm"]["rows"], 32)  # rule: one row per lane
        self.check_valid(p)

    def test_the_input_template_is_not_changed(self):
        before = ref()
        derive("--superlanes", "4")
        self.assertEqual(ref(), before)


class WideChips(unittest.TestCase):
    """Superlane counts above the 20 of ref: the rows rule is capped by the schema, and the links keep their published rate."""

    def test_every_superlane_count_up_to_the_schema_maximum_derives_a_valid_profile_without_rows(self):
        for n in range(1, SCHEMA_MAX_SUPERLANES + 1):
            p = json.loads(derive("--superlanes", str(n)).stdout)
            self.assertEqual(p["mxm"]["rows"], min(16 * n, SCHEMA_MAX_ROWS), n)
            self.assertEqual(dpr.schema_errors(p), [], n)

    def test_the_capped_rows_rule_is_printed_with_the_reason(self):
        r = derive("--superlanes", "32")
        self.assertIn("320 is a cap", r.stderr)
        self.assertIn("not square above 20 superlanes", r.stderr)

    def test_the_derived_32_superlane_profile_is_accepted_by_the_harness(self):
        p = json.loads(derive("--superlanes", "32").stdout)
        r = sim_accepts(p)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_the_docstring_and_the_document_state_the_schema_range(self):
        doc = open(os.path.join(TOOLS, "perf", "derive_profile.py")).read()
        text = open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")).read()
        for stale in ("1, 2, 4 or 20", "1, 2, 4, 20"):
            self.assertNotIn(stale, doc)
            self.assertNotIn(stale, text)
        self.assertIn(f"1 to {SCHEMA_MAX_SUPERLANES} superlanes", text)


class F16Rows(unittest.TestCase):
    """One rule for the f16 capacity at every width: mxm.f16_rows = mxm.rows / 2 (the ratio of ref, HC34 p.28), stated and overridable."""

    def test_the_f16_rows_are_half_the_rows_at_every_width(self):
        for n in range(1, SCHEMA_MAX_SUPERLANES + 1):
            p = json.loads(derive("--superlanes", str(n)).stdout)
            self.assertEqual(p["mxm"]["f16_rows"], p["mxm"]["rows"] // 2, n)
            self.assertEqual(dpr.schema_errors(p), [], n)

    def test_the_f16_rows_never_decrease_when_the_chip_gets_wider(self):
        rows = [json.loads(derive("--superlanes", str(n)).stdout)["mxm"]["f16_rows"] for n in range(1, SCHEMA_MAX_SUPERLANES + 1)]
        self.assertEqual(rows, sorted(rows))

    def test_a_row_count_gives_half_of_it(self):
        self.assertEqual(json.loads(derive("--rows", "96").stdout)["mxm"]["f16_rows"], 48)

    def test_a_row_count_set_by_hand_brings_its_f16_rows(self):
        p = json.loads(derive("--set", "mxm.rows=160").stdout)
        self.assertEqual(p["mxm"]["f16_rows"], 80)

    def test_the_rule_is_listed_and_the_change_is_reported_with_it(self):
        r = derive("--superlanes", "8")
        self.assertIn("mxm.f16_rows = mxm.rows / 2", r.stderr)
        self.assertIn("mxm.f16_rows: 160 -> 64", r.stderr)
        self.assertIn("mxm.f16_rows = mxm.rows / 2", open(os.path.join(TOOLS, "perf", "derive_profile.py")).read())

    def test_set_wins_and_a_tandem_template_is_brought_back_to_half_unless_set(self):
        p = json.loads(derive("--set", "mxm.f16_rows=320").stdout)
        self.assertEqual(p["mxm"]["f16_rows"], 320)
        with tempfile.TemporaryDirectory() as d:
            tandem = os.path.join(d, "tandem.json")
            with open(tandem, "w") as fh:
                json.dump(dict(ref(), mxm=dict(ref()["mxm"], f16_rows=320)), fh)
            r = derive("--template", tandem)
        self.assertEqual(json.loads(r.stdout)["mxm"]["f16_rows"], 160)
        self.assertIn("mxm.f16_rows: 320 -> 160", r.stderr)

    def test_the_document_states_the_rule(self):
        text = open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")).read()
        self.assertIn("`mxm.f16_rows` = `mxm.rows` / 2", text)


class AddressWidth(unittest.TestCase):
    """Rule R4 derives the address width with the bank bit: a slice of 2^k words needs k address bits, and at least the 13 of ref."""

    def derived(self, words, *extra):
        r = derive("--set", f"sram.words_per_slice={words}", *extra)
        return json.loads(r.stdout), r.stderr

    def test_more_than_8192_words_bring_the_address_width_and_a_valid_profile(self):
        for words, bank, addr in ((16384, 13, 14), (32768, 14, 15), (65536, 15, 16)):
            p, err = self.derived(words)
            self.assertEqual((p["sram"]["bank_bit"], p["sram"]["addr_bits"]), (bank, addr), words)
            self.assertEqual(dpr.schema_errors(p), [], words)
            self.assertIn(f"sram.addr_bits: 13 -> {addr}", err)

    def test_the_harness_accepts_the_derived_wide_slices(self):
        for words in (16384, 65536):
            p, _ = self.derived(words)
            r = sim_accepts(p)
            self.assertEqual(r.returncode, 0, (words, r.stderr))
            self.assertIn("E - -1 0", r.stdout)

    def test_fewer_words_keep_the_13_bits_of_ref(self):
        for words, bank in ((256, 7), (1024, 9), (8192, 12)):
            p, _ = self.derived(words)
            self.assertEqual((p["sram"]["bank_bit"], p["sram"]["addr_bits"]), (bank, 13), words)
            self.assertEqual(dpr.schema_errors(p), [], words)

    def test_a_wider_address_set_by_hand_wins(self):
        p, err = self.derived(16384, "--set", "sram.addr_bits=16")
        self.assertEqual(p["sram"]["addr_bits"], 16)
        self.assertIn("override", err)
        self.assertEqual(dpr.schema_errors(p), [])

    def test_the_rule_is_listed(self):
        self.assertIn("sram.addr_bits", derive("--set", "sram.words_per_slice=16384").stderr.split("numbers changed")[0])

    def test_no_words_override_leaves_the_template_address_width(self):
        self.assertEqual(json.loads(derive("--superlanes", "4").stdout)["sram"], ref()["sram"])


class LinkRates(unittest.TestCase):
    """The link and host rates of a derived profile come from the published byte rates, not from the vector width."""

    def rates(self, n):
        p = json.loads(derive("--superlanes", str(n)).stdout)
        vlanes = 16 * n
        return (vlanes / p["c2c"]["cycles_per_vector"], vlanes / p["host"]["cycles_per_vector"],
                p["c2c"]["cycles_per_vector"], p["host"]["cycles_per_vector"])

    def test_ref_width_reproduces_ref(self):
        _, _, c2c, host = self.rates(20)
        self.assertEqual((c2c, host), (ref()["c2c"]["cycles_per_vector"], ref()["host"]["cycles_per_vector"]))

    def test_a_wider_vector_takes_more_cycles_on_the_same_link(self):
        _, _, c2c, host = self.rates(32)
        self.assertEqual(c2c, 37)  # ceil(512 B / (12.5 GB/s / 900 MHz))
        self.assertEqual(host, 16)  # ceil(512 B / 32 B per cycle)

    def test_no_derived_chip_has_a_faster_link_than_the_published_one(self):
        for n in range(1, SCHEMA_MAX_SUPERLANES + 1):
            link, host, _, _ = self.rates(n)
            self.assertLessEqual(link * 900e6 / 1e9, 12.5 + 1e-9, n)  # 4 lanes x 25 Gb/s
            self.assertLessEqual(host, 32 + 1e-9, n)
            if n >= 8:  # the integer cycle count rounds a narrow vector coarsely
                self.assertGreaterEqual(link * 900e6 / 1e9, 12.5 * 0.9, n)

    def test_the_rule_is_printed_and_set_overrides_it(self):
        r = derive("--superlanes", "32", "--set", "c2c.cycles_per_vector=24")
        self.assertIn("R5", r.stderr)
        self.assertIn("c2c.cycles_per_vector: 24 -> 37", derive("--superlanes", "32").stderr)
        self.assertEqual(json.loads(r.stdout)["c2c"]["cycles_per_vector"], 24)

    def test_no_superlane_argument_leaves_the_template_rates(self):
        p = json.loads(derive("--rows", "160").stdout)
        self.assertEqual(p["c2c"], ref()["c2c"])
        self.assertEqual(p["host"], ref()["host"])


class Report(unittest.TestCase):
    def test_rules_and_changed_numbers_are_printed(self):
        r = derive("--superlanes", "4", "--rows", "64", "--slices-per-hemisphere", "8")
        text = r.stderr
        self.assertIn("derived default", text)
        self.assertIn("overridable", text)
        self.assertIn("timing.sxm_map_ready: 20 -> 4", text)
        self.assertIn("layout.mxm_in_gap: 43 -> 34", text)
        self.assertIn("mxm.rows: 320 -> 64", text)
        self.assertIn("not a model change", text)
        self.assertNotIn("derived default", r.stdout)  # stdout is the profile alone, so `> profile.json` works

    def test_unchanged_keys_are_not_listed(self):
        r = derive("--superlanes", "4", "--rows", "64", "--slices-per-hemisphere", "8")
        self.assertNotIn("barrier.release_latency", r.stderr)


class Overrides(unittest.TestCase):
    def test_set_wins_over_a_rule_and_is_labelled(self):
        r = derive("--superlanes", "4", "--set", "timing.sxm_map_ready=9", "--set", "layout.vxm_gap=22")
        p = json.loads(r.stdout)
        self.assertEqual(p["timing"]["sxm_map_ready"], 9)
        self.assertEqual(p["layout"]["vxm_gap"], 22)
        self.assertIn("override", r.stderr)
        self.assertIn("timing.sxm_map_ready: 20 -> 9", r.stderr)

    def test_words_per_slice_brings_its_bank_bit(self):
        p = json.loads(derive("--superlanes", "4", "--set", "sram.words_per_slice=1024").stdout)
        self.assertEqual(p["sram"]["words_per_slice"], 1024)
        self.assertEqual(p["sram"]["bank_bit"], 9)

    def test_unknown_key_and_bad_value_are_refused(self):
        r = derive("--set", "bogus.key=1", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("bogus.key", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        r = derive("--set", "mxm.rows=lots", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("mxm.rows", r.stderr)
        r = derive("--set", "mxm.rows", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("KEY=VALUE", r.stderr)


class Refusals(unittest.TestCase):
    def test_a_geometry_outside_the_schema_is_refused_with_the_reason(self):
        r = derive("--superlanes", "40", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("geometry.superlanes", r.stderr)
        self.assertIn(f"is above the maximum {SCHEMA_MAX_SUPERLANES}", r.stderr)
        self.assertEqual(r.stdout, "")
        self.assertNotIn("Traceback", r.stderr)

    def test_allow_invalid_prints_it_labelled_and_the_harness_agrees(self):
        r = derive("--superlanes", "40", "--allow-invalid", check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOT SCHEMA-VALID", r.stderr)
        p = json.loads(r.stdout)
        self.assertEqual(p["geometry"]["superlanes"], 40)
        sim = sim_accepts(p)
        self.assertNotEqual(sim.returncode, 0)
        self.assertIn("geometry.superlanes", sim.stderr)

    def test_too_many_quads_and_bad_numbers(self):
        r = derive("--slices-per-hemisphere", "48", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("geometry.mem_slices_per_hemisphere", r.stderr)
        for flag, bad in (("--superlanes", "0"), ("--superlanes", "x"), ("--rows", "-16"), ("--slices-per-hemisphere", "2.5")):
            r = derive(flag, bad, check=False)
            self.assertEqual(r.returncode, 2, (flag, bad))
            self.assertIn(flag, r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_rows_must_be_a_multiple_of_sixteen(self):
        r = derive("--rows", "100", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("mxm.rows", r.stderr)

    def test_missing_template(self):
        r = derive("--template", "/nonexistent/t.json", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("/nonexistent/t.json", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_a_template_without_the_keys_the_rules_need(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w") as fh:
                json.dump({"geometry": {"superlanes": 20}}, fh)
            r = derive("--template", path, check=False)
            self.assertEqual(r.returncode, 2)
            self.assertIn("lacks a key the rules need", r.stderr)
            self.assertIn(path, r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_interrupt_is_reported_not_a_traceback(self):
        import derive_peaks as dp_
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            dp_.run(lambda: (_ for _ in ()).throw(KeyboardInterrupt()), "derive_profile")
        self.assertEqual(cm.exception.code, 130)
        self.assertIn("interrupted", err.getvalue())

    def test_out_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            derive("--superlanes", "4", "--out", path)
            with open(path) as fh:
                self.assertEqual(json.load(fh)["geometry"]["superlanes"], 4)



class Extensions(unittest.TestCase):
    """: a template with extensions derives, and `extensions` and `mxm.acc_bits` keep the schema relation under --set."""

    def template(self, name):
        return os.path.join(SPEC, "config", "test", name + ".json")

    def test_a_template_with_extensions_derives(self):
        for name in ("ref_ext", "ext_bf16", "mini_ext"):
            r = derive("--template", self.template(name), "--superlanes", "16")
            p = json.loads(r.stdout)
            with open(self.template(name)) as fh:
                t = json.load(fh)
            self.assertEqual(p["extensions"], t["extensions"], name)
            self.assertEqual(p["mxm"]["acc_bits"], t["mxm"]["acc_bits"], name)

    def test_set_extensions_list_sets_acc_bits(self):
        p = json.loads(derive("--set", 'extensions=["wide_acc"]').stdout)
        self.assertEqual((p["extensions"], p["mxm"]["acc_bits"]), (["wide_acc"], 128))

    def test_set_acc_bits_lists_wide_acc(self):
        p = json.loads(derive("--set", "mxm.acc_bits=128").stdout)
        self.assertEqual((p["extensions"], p["mxm"]["acc_bits"]), (["wide_acc"], 128))
        p = json.loads(derive("--template", self.template("ref_ext"), "--set", "mxm.acc_bits=32").stdout)
        self.assertEqual((p["extensions"], p["mxm"]["acc_bits"]), (["block_fp8"], 32))

    def test_set_extensions_must_be_a_list_of_names(self):
        r = derive("--set", "extensions=3", check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("JSON list", r.stderr)


if __name__ == "__main__":
    unittest.main()
