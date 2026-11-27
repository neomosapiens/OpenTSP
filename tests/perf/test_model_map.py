#!/usr/bin/env python3
"""opentsp-arch-spec/tools/perf/model_map.py: an analytic mapping of a decoder-only transformer onto N chips of a profile
SRAM footprint, pipeline versus tensor-parallel split, MXM cycles from the derived rates, C2C cost, tokens per
second with the binding bound named. Every figure is an estimate; the tool is checked against the one measured layer.

Run: python3 tests/perf/test_model_map.py   (reads only the shipped profiles and tests/perf/models/*.json)."""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join(ROOT, "opentsp-arch-spec")
TOOL = os.path.join(SPEC, "tools", "perf", "model_map.py")
MODELS = os.path.join(ROOT, "tests", "perf", "models")
sys.path.insert(0, os.path.join(SPEC, "tools", "perf"))

import model_map as mm  # noqa: E402

# measured on the TinyStories simulation fixtures
MEASURED_SERIAL_LAYER = 11362
MEASURED_DECODE_STEP = 26398
# The all-reduce of 13 vectors (4,096 bytes) on the modelled 64 chips, from the keys of ref (L = 216, 24 cycles per vector, relay 1 + 38 - 5 = 34):
# the node-local exchange of 8 chips (216 + 13 x 24 = 528), the global-link step with one relay (34 + 528), the two chips that own the node's
# seven global links exchange their partial sums (34 + 528), and each serves three chips of the node (34 + 528).
RELAY = 34
STEP = 216 + 13 * 24
ALLREDUCE_4096_64 = STEP + 3 * (RELAY + STEP)
# The 7B model's partial sums are f16 by default (act_bytes 2): 4,096 x 2 = 8,192 bytes, 26 vectors per all-reduce.
STEP_F16 = 216 + 26 * 24
MODELLED_ALLREDUCE_7B_64 = STEP_F16 + 3 * (RELAY + STEP_F16)


def ref():
    with open(os.path.join(SPEC, "config", "ref.json")) as fh:
        return json.load(fh)


def model(name):
    with open(os.path.join(MODELS, name + ".json")) as fh:
        return json.load(fh)


def run_cli(*args):
    return subprocess.run([sys.executable, TOOL, *args], capture_output=True, text=True)


class Collectives(unittest.TestCase):
    """The flat pricing: a direct link between every pair of chips, the hypothetical wiring of `--topology flat`.
    Link rates and hop latencies of ref: 24 cycles per 320-byte vector per link, 216 cycles of latency, 16 links."""

    def setUp(self):
        self.cfg = ref()

    def test_allreduce_four_chips_four_vectors(self):
        c = mm.allreduce_cycles(self.cfg, chips=4, nbytes=1280)
        self.assertEqual(c["ring"], 2 * 3 * (216 + 1 * 24))  # 6 steps of one 320-byte vector
        self.assertEqual(c["recursive_doubling"], 2 * (216 + 4 * 24))  # 2 steps of the whole vector
        self.assertEqual(c["direct"], 216 + 4 * 24)  # 3 peers on 3 links, one step
        self.assertEqual(min(c, key=c.get), "direct")

    def test_direct_needs_a_link_per_peer(self):
        c = mm.allreduce_cycles(self.cfg, chips=64, nbytes=4096)
        self.assertNotIn("direct", c)  # 63 peers, 16 links
        self.assertEqual(c["recursive_doubling"], 6 * (216 + 13 * 24))
        self.assertEqual(c["ring"], 2 * 63 * (216 + 1 * 24))

    def test_one_chip_has_no_collective(self):
        self.assertEqual(mm.allreduce_cycles(self.cfg, chips=1, nbytes=4096), {"none": 0})
        self.assertEqual(mm.allgather_cycles(self.cfg, chips=1, shard_bytes=64), {"none": 0})

    def test_allgather_four_chips(self):
        c = mm.allgather_cycles(self.cfg, chips=4, shard_bytes=320)
        self.assertEqual(c["ring"], 3 * (216 + 24))
        self.assertEqual(c["recursive_doubling"], (216 + 24) + (216 + 48))
        self.assertEqual(c["direct"], 216 + 24)

    def test_more_links_per_hop_shortens_the_transfer_not_the_latency(self):
        one = mm.allreduce_cycles(self.cfg, chips=64, nbytes=4096, links_per_hop=1)["recursive_doubling"]
        four = mm.allreduce_cycles(self.cfg, chips=64, nbytes=4096, links_per_hop=4)["recursive_doubling"]
        self.assertEqual(one, 6 * (216 + 13 * 24))
        self.assertEqual(four, 6 * (216 + 4 * 24))  # ceil(13 / 4) = 4 vectors per link


class Footprint(unittest.TestCase):
    def setUp(self):
        self.cfg = ref()

    def test_dense_7b_footprint_and_minimum_chips(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)
        layer = 4 * 4096 * 4096 + 3 * 4096 * 11008  # attention (K and V as wide as Q: 32 KV heads) and gated FFN
        head = 32000 * 4096
        self.assertEqual(r["footprint"]["weights_bytes"], 32 * layer + head)  # 1 byte per weight
        kv = 32 * 4096 * 2 * 4096 * 2  # layers x context x (K and V) x kv_dim x 2 bytes
        self.assertEqual(r["footprint"]["kv_bytes_per_sequence"], kv)
        cap = 88 * 8192 * 320
        self.assertEqual(r["footprint"]["sram_bytes_per_chip"], cap)
        self.assertEqual(r["footprint"]["min_chips_total"], math.ceil((32 * layer + head + kv) / cap))
        self.assertEqual(r["footprint"]["min_chips_total"], 38)

    def test_pipeline_cannot_split_below_a_layer_but_tensor_parallel_can(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)
        pipe, tens = r["strategies"]["pipeline"], r["strategies"]["tensor"]
        self.assertEqual(pipe["chips_used"], 32)  # one layer per chip, the other 32 chips idle
        self.assertFalse(pipe["fits"])  # a layer's weights plus its KV cache exceed 220 MiB
        self.assertTrue(tens["fits"])
        self.assertIn(r["best"], ("tensor", "hybrid"))  # never the pipeline: it does not fit

    def test_dense_70b_minimum_chips(self):
        r = mm.map_model(model("dense-70b"), self.cfg, chips=64)
        layer = 2 * 8192 * 8192 + 2 * 8192 * 1024 + 3 * 8192 * 28672  # GQA: 8 KV heads of 128
        head = 32000 * 8192
        kv = 80 * 4096 * 2 * 1024 * 2
        cap = 88 * 8192 * 320
        self.assertEqual(r["footprint"]["min_chips_total"], math.ceil((80 * layer + head + kv) / cap))
        self.assertFalse(r["strategies"]["tensor"]["fits"])  # 64 chips are far too few for 70 B at 1 byte per weight
        self.assertIsNone(r["best"])

    def test_sram_fraction_shrinks_the_capacity(self):
        r = mm.map_model(model("tinystories"), self.cfg, chips=1, sram_fraction=0.5)
        self.assertEqual(r["footprint"]["sram_bytes_per_chip"], 88 * 8192 * 320 // 2)


class Cycles(unittest.TestCase):
    def setUp(self):
        self.cfg = ref()

    def test_tinystories_layer_is_install_bound_by_arithmetic(self):
        r = mm.map_model(model("tinystories"), self.cfg, chips=1)
        s = r["strategies"]["pipeline"]
        rate = 2 * 32 * 320  # install bytes per cycle
        weights = (4 * 64 * 64 + 2 * 64 * 256) * 2
        kv = 64 * 2 * 64 * 2
        self.assertEqual(s["install_cycles_layer"], math.ceil((weights + kv) / rate))
        self.assertEqual(s["install_cycles_layer"], 6)
        self.assertEqual(s["layer_cycles"], 6)
        self.assertEqual(s["compute_bound"], "install")
        head = math.ceil(50257 * 64 * 2 / rate)
        self.assertEqual(s["step_cycles"], 8 * 6 + head)

    def test_tensor_parallel_dense_7b_on_64_chips(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)
        s = r["strategies"]["tensor"]
        per_chip_weights = (4 * 4096 * 4096 + 3 * 4096 * 11008) // 64
        per_chip_kv = 4096 * 2 * 4096 * 2 // 32  # sharded over the 32 KV heads, not over 64 chips
        self.assertEqual(s["install_cycles_layer"], math.ceil((per_chip_weights + per_chip_kv) / 20480))
        self.assertEqual(s["install_cycles_layer"], 257)
        self.assertEqual(s["mac_cycles_layer"], math.ceil((per_chip_weights + 2 * 4096 * 4096 // 32) / 409600))  # attention over 32 heads
        self.assertEqual(s["mac_cycles_layer"], 11)
        self.assertEqual(s["collective_algorithm"], "hierarchical")
        self.assertEqual(s["collective_cycles_per_allreduce"], MODELLED_ALLREDUCE_7B_64)
        self.assertEqual(s["layer_cycles"], 257 + 2 * MODELLED_ALLREDUCE_7B_64)
        self.assertEqual(s["binding"], "C2C collective latency")

    def test_prefill_amortises_the_install_over_the_prompt(self):
        d = mm.map_model(model("dense-7b"), self.cfg, chips=64, phase="decode")["strategies"]["tensor"]
        p = mm.map_model(model("dense-7b"), self.cfg, chips=64, phase="prefill", prompt_len=512)["strategies"]["tensor"]
        self.assertGreater(p["mac_cycles_layer"], 100 * d["mac_cycles_layer"])
        self.assertEqual(p["compute_bound"], "MAC")  # 512 tokens per weight install: MAC bound
        # tokens per second counts the prompt's tokens
        self.assertGreater(p["tokens_per_s"], d["tokens_per_s"])

    def test_batch_raises_mac_work_not_install_work(self):
        a = mm.map_model(model("dense-7b"), self.cfg, chips=64, batch=1)["strategies"]["tensor"]
        b = mm.map_model(model("dense-7b"), self.cfg, chips=64, batch=8)["strategies"]["tensor"]
        self.assertEqual(b["mac_cycles_layer"], math.ceil(8 * (3162112 + 1048576) / 409600))
        self.assertGreater(b["install_cycles_layer"], a["install_cycles_layer"])  # each sequence installs its own KV cache

    def test_fp16_weights_use_the_fp16_mac_rate(self):
        m = dict(model("dense-7b"), weight_bytes=2)
        s = mm.map_model(m, self.cfg, chips=64, batch=8)["strategies"]["tensor"]
        macs = 8 * ((4 * 4096 * 4096 + 3 * 4096 * 11008) // 64 + 2 * 4096 * 4096 // 32)
        self.assertEqual(s["mac_cycles_layer"], math.ceil(macs / 102400))

    def test_pipeline_hop_and_latency(self):
        r = mm.map_model(model("tinystories"), self.cfg, chips=2)
        s = r["strategies"]["pipeline"]
        hop = 216 + 1 * 24  # 64 x 2 bytes = 128 bytes: one vector
        self.assertEqual(s["hop_cycles"], hop)
        self.assertEqual(s["c2c_bytes_per_token"], 128)
        self.assertEqual(s["chips_used"], 2)
        stage = 4 * 6
        self.assertEqual(s["step_cycles"], stage + hop + (stage + math.ceil(50257 * 64 * 2 / 20480)))
        self.assertGreaterEqual(s["tokens_per_s_pipelined"], s["tokens_per_s"])

    def test_layer_overhead_is_added_and_named(self):
        r = mm.map_model(model("tinystories"), self.cfg, chips=1, layer_overhead_cycles=1000)
        s = r["strategies"]["pipeline"]
        self.assertEqual(s["layer_cycles"], 6 + 1000)
        self.assertEqual(s["binding"], "latency floor (layer overhead)")

    def test_everything_is_an_estimate(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)
        self.assertEqual(r["label"], "estimate")
        for s in r["strategies"].values():
            self.assertEqual(s["label"], "estimate")


def tmp_json(d, name, obj):
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        json.dump(obj, fh)
    return path


class ModelledWiring(unittest.TestCase):
    """The collectives are priced on the wiring `tsp_sim --chips 64` builds: 8 fully connected nodes of 8 chips, one global link per node pair,
    owned by two chips of each node (global ports 4 per chip), a relay costing d_func(c2c) + c2c_tx_gap - c2c_rx_gap = 34 cycles."""

    def setUp(self):
        self.cfg = ref()

    def test_the_relay_cost_is_the_measured_one(self):
        # relay_directed.py measures 2 L + 34 from the first Send to the last Receive of a relay: 466 on ref
        self.assertEqual(mm.relay_cycles(self.cfg), RELAY)
        self.assertEqual(2 * 216 + mm.relay_cycles(self.cfg), 466)

    def test_the_64_chip_allreduce_is_not_a_direct_link_to_the_distance_32_partner(self):
        name, cycles, phases = mm.allreduce_on(self.cfg, mm.modelled_wiring(), group=64, nbytes=4096)
        self.assertEqual(name, "hierarchical")
        self.assertEqual(cycles, ALLREDUCE_4096_64)
        self.assertNotEqual(cycles, 6 * (216 + 13 * 24))  # the flat recursive doubling over direct links
        self.assertEqual([p["phase"] for p in phases],
                         ["node-local all-reduce", "global links", "owner chips exchange", "owner chips serve the node"])
        self.assertEqual(sum(p["cycles"] for p in phases), cycles)
        # the cross-node step is a global-link transfer after a relay, not a plain link step
        self.assertEqual(phases[1]["cycles"], RELAY + STEP)

    def test_a_node_is_fully_connected_up_to_eight_chips(self):
        for chips in (2, 4, 8):
            name, cycles, _ = mm.allreduce_on(self.cfg, mm.modelled_wiring(), group=chips, nbytes=4096)
            self.assertEqual(cycles, STEP, chips)  # one direct exchange: chips - 1 <= 7 local links
        name, cycles, _ = mm.allreduce_on(self.cfg, mm.modelled_wiring(), group=1, nbytes=4096)
        self.assertEqual(cycles, 0)

    def test_two_nodes_need_only_the_global_steps(self):
        name, cycles, phases = mm.allreduce_on(self.cfg, mm.modelled_wiring(), group=16, nbytes=4096)
        # one node pair, one global link owned by one chip: local, global (relay), then that chip serves seven peers
        self.assertEqual([p["phase"] for p in phases], ["node-local all-reduce", "global links", "owner chips serve the node"])
        self.assertEqual(cycles, STEP + 2 * (RELAY + STEP))

    def test_the_allgather_crosses_the_global_links_too(self):
        name, cycles, phases = mm.allgather_on(self.cfg, mm.modelled_wiring(), group=64, shard_bytes=500)
        self.assertEqual(name, "hierarchical")
        self.assertGreater(cycles, 0)
        self.assertIn("global links", [p["phase"] for p in phases])

    def test_the_pipeline_hop_across_a_node_boundary_takes_three_link_hops_and_two_relays(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)
        s = r["strategies"]["pipeline"]
        v = 26  # 4,096 f16 activations (act_bytes 2) = 8,192 bytes: 26 vectors of 320 bytes
        inside = 216 + v * 24
        across = 3 * 216 + 2 * RELAY + v * 24
        self.assertEqual(s["chips_used"], 32)
        # 32 stages on 32 consecutive chips: 31 hops, three of them cross a node boundary (chips 7 to 8, 15 to 16, 23 to 24)
        self.assertEqual(s["c2c_cycles_per_step"], 28 * inside + 3 * across)

    def test_a_hybrid_split_fits_70b_on_400_chips(self):
        r = mm.map_model(model("dense-70b"), self.cfg, chips=400)
        h = r["strategies"]["hybrid"]
        self.assertEqual((h["tensor_group"], h["stages"]), (8, 50))
        self.assertTrue(h["fits"], h["bytes_per_chip"] / 2**20)
        self.assertFalse(r["strategies"]["tensor"]["fits"])
        self.assertEqual(r["best"], "hybrid")
        self.assertGreater(h["tokens_per_s"], 0)

    def test_hybrid_groups_stay_inside_a_node(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)
        h = r["strategies"]["hybrid"]
        self.assertIn(h["tensor_group"], (2, 4, 8))
        self.assertLessEqual(h["tensor_group"] * h["stages"], 64)

    def warnings(self, chips, cfg=None):
        return mm.map_model(model("dense-7b"), cfg or self.cfg, chips=chips)["warnings"]

    def test_every_chip_count_a_run_accepts_is_free_of_a_limit_warning(self):
        # tsp_sim --chips takes 1..264 and the last node may be partial: 12, 72, 128 and 264 chips are runs, not warnings about a limit
        for chips in (1, 8, 12, 64, 72, 128, 264):
            self.assertEqual([x for x in self.warnings(chips) if "limit" in x or "tsp_sim" in x or "cannot" in x], [], chips)

    def test_beyond_a_run_but_addressable_needs_a_topology_file(self):
        w = self.warnings(400)
        self.assertTrue(any("400" in x and "264" in x and "--topology" in x for x in w), w)
        self.assertFalse(any("cannot address" in x for x in w), w)
        r = run_cli(os.path.join(MODELS, "dense-70b.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "400")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("warning", r.stderr)
        self.assertIn("WARNING", r.stdout)  # the report carries it too

    def test_beyond_the_512_a_chip_id_names_is_not_addressable(self):
        w = self.warnings(600)
        self.assertTrue(any("600" in x and "512" in x and "cannot address" in x for x in w), w)
        self.assertEqual([x for x in self.warnings(512) if "cannot address" in x], [])

    def test_the_global_ports_of_the_harness_wiring_join_33_nodes_at_most(self):
        w = self.warnings(272)  # 34 nodes of 8: a node has 32 global ports (8 chips x 4)
        self.assertTrue(any("34 nodes" in x and "global" in x for x in w), w)
        self.assertEqual([x for x in self.warnings(264) if "global" in x], [])

    def test_a_partial_last_node_is_priced_and_said_so(self):
        w = self.warnings(12)
        self.assertTrue(any("12" in x and "4 chips" in x for x in w), w)
        self.assertFalse(any("not a count" in x for x in w), w)

    def test_a_profile_with_too_few_links_for_the_wiring_is_warned_about(self):
        cfg = ref()
        cfg["c2c"]["links"] = 8
        self.assertTrue(any("links" in x and "11" in x for x in self.warnings(64, cfg)), self.warnings(64, cfg))
        cfg["c2c"]["links"] = 128
        self.assertEqual([x for x in self.warnings(64, cfg) if "links" in x], [])

    def test_the_modelled_wiring_is_named_in_the_report(self):
        r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "64")
        self.assertIn("wiring: modelled", r.stdout)
        self.assertNotIn("HYPOTHETICAL", r.stdout)


class PartialSumPayload(unittest.TestCase):
    """The collectives carry `act_bytes` per element: the width of a partial sum (f16, 2 bytes, unless the model says otherwise)."""

    def setUp(self):
        self.cfg = ref()

    def run7b(self, **change):
        return mm.map_model(dict(model("dense-7b"), **change), self.cfg, chips=64)["strategies"]

    def test_the_default_is_an_f16_partial_sum_whatever_the_weight_type(self):
        for wb in (1, 2):
            s = self.run7b(weight_bytes=wb)["tensor"]
            self.assertEqual(s["c2c_bytes_per_token"], 32 * 2 * 4096 * 2 + 32000 * 2, wb)
        self.assertEqual(self.run7b()["tensor"]["collective_cycles_per_allreduce"], MODELLED_ALLREDUCE_7B_64)

    def test_the_payload_follows_act_bytes(self):
        by = {b: self.run7b(act_bytes=b) for b in (1, 2, 4)}
        step = {b: 216 + -(-4096 * b // 320) * 24 for b in (1, 2, 4)}
        for b in (1, 2, 4):
            self.assertEqual(by[b]["tensor"]["collective_cycles_per_allreduce"], step[b] + 3 * (RELAY + step[b]), b)
        self.assertEqual(by[2], self.run7b())
        # a wider partial sum is never faster, and the hybrid stays the best split that fits in all three cases
        rates = [by[b]["tensor"]["tokens_per_s"] for b in (1, 2, 4)]
        self.assertEqual(rates, sorted(rates, reverse=True))
        self.assertLess(rates[2], rates[1] * 0.7)
        for b in (1, 2, 4):
            self.assertGreater(by[b]["hybrid"]["tokens_per_s"], by[b]["tensor"]["tokens_per_s"], b)

    def test_a_bad_act_bytes_is_refused_with_the_key_name(self):
        for bad in (0, -1, 3, 8, 1.5, "x", True, None, [2]):
            with self.assertRaises(mm.ModelError, msg=repr(bad)) as cm:
                mm.check_model(dict(model("dense-7b"), act_bytes=bad))
            self.assertIn("act_bytes", str(cm.exception))
        for good in (1, 2, 4):
            mm.check_model(dict(model("dense-7b"), act_bytes=good))

    def test_the_cli_refuses_it_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (0, "x", 3):
                path = tmp_json(d, "m.json", dict(model("dense-7b"), act_bytes=bad))
                r = run_cli(path, os.path.join(SPEC, "config", "ref.json"), "--chips", "4")
                self.assertEqual(r.returncode, 2, bad)
                self.assertIn("act_bytes", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_the_key_is_documented_where_the_model_keys_are(self):
        with open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")) as source:
            text = source.read()
        self.assertIn("`act_bytes` 1, 2 or 4", text)
        self.assertIn("act_bytes", mm.__doc__)


class PipelineFull(unittest.TestCase):
    """The 'pipeline full' rate needs one sequence per stage in flight, and the cache of that many sequences must fit."""

    def setUp(self):
        self.cfg = ref()

    def test_it_is_given_only_when_the_batch_it_needs_fits(self):
        r = mm.map_model(model("dense-7b"), self.cfg, chips=64)["strategies"]
        h = r["hybrid"]
        self.assertEqual(h["pipelined_sequences"], h["stages"])
        self.assertFalse(h["pipelined_fits"])
        self.assertIsNone(h["tokens_per_s_pipelined"])
        self.assertGreater(h["pipelined_bytes_per_chip"], 220 * 2**20)
        self.assertTrue(h["fits"])
        big = mm.map_model(model("dense-7b"), self.cfg, chips=64, batch=8)["strategies"]["hybrid"]
        self.assertEqual(big["pipelined_sequences"], 8 * big["stages"])

    def test_a_model_whose_cache_is_small_keeps_its_rate(self):
        s = mm.map_model(model("tinystories"), self.cfg, chips=2)["strategies"]["pipeline"]
        self.assertTrue(s["pipelined_fits"])
        self.assertEqual(s["pipelined_sequences"], 2)
        self.assertGreaterEqual(s["tokens_per_s_pipelined"], s["tokens_per_s"])

    def test_the_report_says_what_the_figure_needs(self):
        m = os.path.join(MODELS, "dense-7b.json")
        out = run_cli(m, os.path.join(SPEC, "config", "ref.json"), "--chips", "64").stdout
        self.assertNotIn("tokens/s with the pipeline full", out)
        self.assertIn("pipeline full would need 8 sequences in flight", out)
        self.assertIn("DOES NOT FIT", out)
        out = run_cli(os.path.join(MODELS, "tinystories.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "2").stdout
        self.assertIn("tokens/s with the pipeline full: 2 sequences in flight", out)


class LinkRateWarning(unittest.TestCase):
    def test_a_profile_with_a_faster_link_than_any_published_one_is_warned_about(self):
        with open(os.path.join(SPEC, "config", "test", "sl32.json")) as fh:
            wide = json.load(fh)
        w = mm.map_model(model("dense-7b"), wide, chips=64)["warnings"]
        self.assertTrue(any("link rate" in x and "19.20 GB/s" in x for x in w), w)
        self.assertEqual(mm.map_model(model("dense-7b"), ref(), chips=64)["warnings"], [])


class Topology(unittest.TestCase):
    """`--topology flat` and `--topology FILE` price wirings the harness does not build; every report line says so."""

    def test_flat_is_the_old_direct_link_pricing(self):
        r = mm.map_model(model("dense-7b"), ref(), chips=64, topology=mm.flat_wiring())
        s = r["strategies"]["tensor"]
        self.assertEqual(s["collective_algorithm"], "recursive_doubling")
        self.assertEqual(s["collective_cycles_per_allreduce"], 6 * (216 + 26 * 24))  # 4,096 f16 partial sums: 26 vectors
        self.assertTrue(r["wiring"]["hypothetical"])

    def test_a_wiring_parameter_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = tmp_json(d, "w.json", {"name": "dual-link", "chips_per_node": 8, "global_links_per_node_pair": 2})
            one = mm.map_model(model("dense-7b"), ref(), chips=64)["strategies"]["tensor"]
            two = mm.map_model(model("dense-7b"), ref(), chips=64, topology=mm.load_wiring_params(path))["strategies"]["tensor"]
        self.assertLess(two["collective_cycles_per_allreduce"], one["collective_cycles_per_allreduce"])

    def test_cli_labels_every_hypothetical_line(self):
        with tempfile.TemporaryDirectory() as d:
            path = tmp_json(d, "w.json", {"name": "dual-link", "chips_per_node": 8, "global_links_per_node_pair": 2})
            r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "64", "--wiring-params", path)
            f = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "64", "--topology", "flat")
        for out in (r, f):
            self.assertEqual(out.returncode, 0, out.stderr)
            lines = [l for l in out.stdout.splitlines() if l.lstrip().startswith(("C2C:", "step:", "wiring:", "model_map:"))]
            self.assertTrue(lines)
            for line in lines:
                self.assertIn("HYPOTHETICAL", line)

    def test_bad_topology_files_are_refused_with_a_message(self):
        m, p = os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json")
        with tempfile.TemporaryDirectory() as d:
            cases = [(tmp_json(d, "a.json", {"name": "x", "chips_per_node": 0}), "chips_per_node"),
                     (tmp_json(d, "b.json", {"name": "x", "chips_per_node": 8, "global_links_per_node_pair": "two"}), "global_links_per_node_pair"),
                     (tmp_json(d, "c.json", {"chips_per_node": 8}), "name"),
                     ("/nonexistent/wiring.json", "/nonexistent/wiring.json")]
            bad = os.path.join(d, "d.json")
            with open(bad, "w") as fh:
                fh.write("{not json")
            cases.append((bad, "not valid JSON"))
            for path, needle in cases:
                r = run_cli(m, p, "--chips", "64", "--wiring-params", path)
                self.assertEqual(r.returncode, 2, path)
                self.assertIn(needle, r.stderr)
                self.assertNotIn("Traceback", r.stderr)


GEN_TOPOLOGY = os.path.join(SPEC, "tools", "gen_topology.py")


def generated(*args):
    """The topology file `tools/gen_topology.py` writes, parsed."""
    r = subprocess.run([sys.executable, GEN_TOPOLOGY, *args], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


class SimulatorTopologyFile(unittest.TestCase):
    """`--topology FILE` reads the file `tsp_sim --topology` reads (the files tools/gen_topology.py writes), not a parametric description."""

    def price(self, doc, chips=None, cfg=None):
        with tempfile.TemporaryDirectory() as d:
            path = tmp_json(d, "t.json", doc)
            w = mm.load_topology(path)
            n = chips or doc["chips"]
            return w, mm.map_model(model("dense-7b"), cfg or ref(), chips=n, topology=w)

    def test_the_dragonfly_of_64_prices_like_the_modelled_wiring(self):
        w, r = self.price(generated("dragonfly", "64"))
        base = mm.map_model(model("dense-7b"), ref(), chips=64)
        self.assertEqual((w["chips_per_node"], w["global_links_per_node_pair"], w["global_ports_per_chip"]), (8, 1, 4))
        self.assertFalse(w["hypothetical"])
        for split in ("tensor", "hybrid", "pipeline"):
            self.assertEqual(r["strategies"][split]["step_cycles"], base["strategies"][split]["step_cycles"], split)
        self.assertEqual(r["strategies"]["tensor"]["collective_cycles_per_allreduce"], 3462)

    def test_the_dragonfly_of_72_is_read_with_nine_nodes(self):
        w, r = self.price(generated("dragonfly", "72"))
        self.assertEqual(w["chips"], 72)
        self.assertEqual((w["chips_per_node"], w["nodes"]), (8, 9))
        self.assertEqual(r["chips"], 72)
        self.assertEqual(r["warnings"], [])

    def test_a_partial_last_node_is_read(self):
        w, _ = self.price(generated("dragonfly", "20"))
        self.assertEqual((w["chips_per_node"], w["nodes"]), (8, 3))

    def test_a_single_node(self):
        w, r = self.price(generated("node8"))
        self.assertEqual((w["chips_per_node"], w["nodes"]), (8, 1))
        self.assertEqual(r["strategies"]["tensor"]["collective_cycles_per_allreduce"],
                         mm.map_model(model("dense-7b"), ref(), chips=8)["strategies"]["tensor"]["collective_cycles_per_allreduce"])

    def test_the_report_names_the_simulator_wiring_and_does_not_call_it_hypothetical(self):
        with tempfile.TemporaryDirectory() as d:
            path = tmp_json(d, "df72.json", generated("dragonfly", "72"))
            r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--topology", path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("wiring: simulator wiring df72.json", r.stdout)
        self.assertNotIn("HYPOTHETICAL", r.stdout)
        self.assertIn("72 chip(s)", r.stdout)  # --chips defaults to the file's chips

    def test_chips_must_agree_with_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = tmp_json(d, "df72.json", generated("dragonfly", "72"))
            r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "64", "--topology", path)
        self.assertEqual(r.returncode, 2)
        self.assertIn("72", r.stderr)
        self.assertIn("--chips", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_a_wiring_that_is_not_nodes_of_fully_connected_chips_is_refused_with_the_reason(self):
        for args in (("torus", "4"), ("tray8",)):
            with tempfile.TemporaryDirectory() as d:
                path = tmp_json(d, "t.json", generated(*args))
                r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--topology", path)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn("fully connected", r.stderr)
            self.assertIn("--topology flat", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_malformed_files_are_refused_with_a_message(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [({"links": []}, "chips"), ({"chips": 600, "links": []}, "512"), ({"chips": 8}, "links"),
                     ({"chips": 8, "links": [{"a": [0, 0]}]}, "links[0]"), ({"chips": 8, "links": [{"a": [9, 0], "b": [1, 0]}]}, "chip 9"),
                     ({"chips": 8, "links": [{"a": [0, "x"], "b": [1, 0]}]}, "links[0]"), ([1, 2], "JSON object")]
            for i, (doc, needle) in enumerate(cases):
                path = tmp_json(d, f"b{i}.json", doc)
                r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "ref.json"), "--chips", "8", "--topology", path)
                self.assertEqual(r.returncode, 2, doc)
                self.assertIn(needle, r.stderr)
                self.assertNotIn("Traceback", r.stderr)


def with_mxm(**keys):
    cfg = ref()
    cfg["mxm"].update(keys)
    return cfg


class InjectionLimitOfTheMapper(unittest.TestCase):
    """A chip injects at most 4 vectors per cycle: a collective that drives more links at once than 4 x c2c.cycles_per_vector is capped."""

    def wide(self):
        cfg = ref()
        cfg["c2c"]["links"] = 128
        return cfg

    def test_a_direct_exchange_with_many_peers_is_bound_by_the_injection(self):
        # 100 chips: 99 peers, each sent 26 vectors (8,320 bytes): 2,574 vectors at 4 per cycle = 644 cycles, above 26 x 24 = 624
        c = mm.allreduce_cycles(self.wide(), chips=100, nbytes=8320)
        self.assertEqual(c["direct"], 216 + math.ceil(26 * 99 / 4))
        g = mm.allgather_cycles(self.wide(), chips=100, shard_bytes=8320)
        self.assertEqual(g["direct"], 216 + math.ceil(26 * 99 / 4))

    def test_a_direct_exchange_with_few_peers_keeps_the_link_rate(self):
        c = mm.allreduce_cycles(ref(), chips=16, nbytes=8320)
        self.assertEqual(c["direct"], 216 + 26 * 24)

    def test_links_per_hop_beyond_the_injection_limit_is_capped_and_said_so(self):
        cfg = self.wide()
        m = model("dense-7b")
        capped = mm.map_model(m, cfg, chips=64, links_per_hop=128)
        edge = mm.map_model(m, cfg, chips=64, links_per_hop=96)
        self.assertEqual(capped["strategies"]["pipeline"]["hop_cycles"], edge["strategies"]["pipeline"]["hop_cycles"])
        self.assertTrue(any("links-per-hop 128" in x and "96" in x for x in capped["warnings"]), capped["warnings"])
        self.assertEqual([x for x in edge["warnings"] if "links-per-hop" in x], [])
        self.assertEqual(capped["links_per_hop"], 96)


class MxmStructureRates(unittest.TestCase):
    """The mapper's MAC rates follow planes_per_hemisphere, f16_rows and fp8_rate, as derive_peaks.py's peaks do."""

    def layer_mac(self, cfg, m, **kw):
        s = mm.map_model(m, cfg, chips=64, batch=64, **kw)["strategies"]["tensor"]
        return s["mac_cycles_layer"]

    def test_the_rates_equal_the_peaks(self):
        import derive_peaks as dp
        for name in ("ref", "test/one_plane", "test/fp8_rate2", "test/f16_tandem", "test/mxm_wide"):
            with open(os.path.join(SPEC, "config", name + ".json")) as fh:
                cfg = json.load(fh)
            rt, q = mm.rates(cfg), dp.derive(cfg, 1000.0)
            self.assertAlmostEqual(rt["int8_macs"] * 2 * 1e9 / 1e12, q["int8_tops"].value, msg=name)
            self.assertAlmostEqual(rt["fp8_macs"] * 2 * 1e9 / 1e12, q["fp8_tops"].value, msg=name)
            self.assertAlmostEqual(rt["f16_macs"] * 2 * 1e9 / 1e12, q["fp16_tflops"].value, msg=name)

    def test_one_plane_doubles_the_int8_mac_cycles(self):
        a, b = self.layer_mac(ref(), model("dense-7b")), self.layer_mac(with_mxm(planes_per_hemisphere=1), model("dense-7b"))
        self.assertLessEqual(abs(b - 2 * a), 1)
        self.assertGreater(a, 100)

    def test_one_plane_has_no_f16_rate_and_an_f16_model_is_refused(self):
        self.assertEqual(mm.rates(with_mxm(planes_per_hemisphere=1))["f16_macs"], 0)
        with self.assertRaises(mm.ModelError) as cm:
            mm.map_model(model("tinystories"), with_mxm(planes_per_hemisphere=1), chips=1)
        self.assertIn("f16", str(cm.exception))
        self.assertIn("planes_per_hemisphere", str(cm.exception))
        r = run_cli(os.path.join(MODELS, "tinystories.json"), os.path.join(SPEC, "config", "test", "one_plane.json"), "--chips", "1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("planes_per_hemisphere", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_an_int8_model_runs_on_one_plane_and_the_report_says_the_f16_check_is_skipped(self):
        r = run_cli(os.path.join(MODELS, "dense-7b.json"), os.path.join(SPEC, "config", "test", "one_plane.json"), "--chips", "64")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("not run", r.stdout)
        self.assertNotIn("Traceback", r.stderr)

    def test_fp8_weights_follow_the_fp8_rate(self):
        fp8 = dict(model("dense-7b"), weight_type="fp8")
        base = self.layer_mac(ref(), model("dense-7b"))
        self.assertEqual(self.layer_mac(ref(), fp8), base)  # rate 1: the INT8 rate
        two = self.layer_mac(with_mxm(fp8_rate=2), fp8)
        self.assertLessEqual(abs(two - base / 2), 1)
        self.assertEqual(self.layer_mac(with_mxm(fp8_rate=2), model("dense-7b")), base)  # INT8 weights do not use the FP8 rate

    def test_the_weight_type_must_agree_with_the_weight_bytes(self):
        for bad in ({"weight_type": "fp8", "weight_bytes": 2}, {"weight_type": "f16", "weight_bytes": 1}, {"weight_type": "bf16"}):
            with self.assertRaises(mm.ModelError) as cm:
                mm.map_model(dict(model("dense-7b"), **bad), ref(), chips=64)
            self.assertIn("weight_type", str(cm.exception))

    def test_bf16_and_bfp8_weights_need_their_extension_and_take_the_f16_rate(self):
        # : the types are weight types of two bytes; the rate is that of f16 and exists only with the extension
        m = dict(model("dense-7b"), weight_bytes=2, kv_bytes=2)
        def ext(name):
            with open(os.path.join(SPEC, "config", "test", name + ".json")) as source:
                return json.load(source)
        base = self.layer_mac(ref(), m)
        self.assertEqual(self.layer_mac(ext("ext_bf16"), dict(m, weight_type="bf16")), self.layer_mac(ext("ext_bf16"), m))
        self.assertEqual(self.layer_mac(ext("ref_ext"), dict(m, weight_type="bfp8")), base)
        for cfg, t in ((ref(), "bf16"), (ref(), "bfp8"), (ext("ext_bf16"), "bfp8"), (ext("ref_ext"), "bf16")):
            with self.assertRaises(mm.ModelError) as cm:
                mm.map_model(dict(m, weight_type=t), cfg, chips=64)
            self.assertIn(f"weight_type {t}", str(cm.exception))

    def test_f16_rows_scale_the_f16_mac_cycles(self):
        m = dict(model("dense-7b"), weight_bytes=2, kv_bytes=2)
        base = self.layer_mac(ref(), m)
        self.assertLessEqual(abs(self.layer_mac(with_mxm(f16_rows=320), m) - base / 2), 1)

    def test_one_plane_per_hemisphere_takes_16_streams_so_half_the_install_rate(self):
        # an IW carries 16 rows: a plane installs from 16 streams, so the install rate of a chip follows its planes up to 32 readers
        self.assertEqual(mm.rates(ref())["install_bytes"], 2 * 32 * 320)
        self.assertEqual(mm.rates(with_mxm(planes_per_hemisphere=1))["install_bytes"], 2 * 16 * 320)


class Validation(unittest.TestCase):
    """The tool against the one measured layer: a throughput bound, so far below a latency-bound program."""

    def test_error_against_the_measured_layer_and_step(self):
        rows = mm.validate_measured(ref())
        by = {r["name"]: r for r in rows}
        layer, step = by["serial layer"], by["8-layer decode step"]
        self.assertEqual(layer["measured_cycles"], MEASURED_SERIAL_LAYER)
        self.assertEqual(step["measured_cycles"], MEASURED_DECODE_STEP)
        self.assertEqual(layer["model_cycles"], 6)
        self.assertEqual(step["model_cycles"], 8 * 6 + math.ceil(50257 * 64 * 2 / 20480))
        for r in rows:
            self.assertAlmostEqual(r["error_percent"], (r["model_cycles"] - r["measured_cycles"]) / r["measured_cycles"] * 100, places=6)
            self.assertLess(r["model_cycles"], r["measured_cycles"])  # a lower bound, never above the measurement
        self.assertLess(layer["error_percent"], -99)

    def test_calibrated_overhead_reproduces_the_layer_it_was_taken_from(self):
        rows = mm.validate_measured(ref())
        overhead = next(r for r in rows if r["name"] == "serial layer")["overhead_cycles"]
        self.assertEqual(overhead, MEASURED_SERIAL_LAYER - 6)
        r = mm.map_model(model("tinystories"), ref(), chips=1, layer_overhead_cycles=overhead)
        self.assertEqual(r["strategies"]["pipeline"]["layer_cycles"], MEASURED_SERIAL_LAYER)


class Document(unittest.TestCase):
    """performance.md quotes the check against the measured program and the example rows: they must be the tool's output."""

    def setUp(self):
        with open(os.path.join(ROOT, "opentsp-arch-spec", "docs", "performance.md")) as fh:
            self.text = fh.read()

    def test_validation_rows(self):
        for r in mm.validate_measured(ref()):
            line = f"| {r['measured_cycles']:,} cycles | {r['model_cycles']:,} cycles | {r['error_percent']:.2f} % |".replace("-", "−")
            self.assertIn(line, self.text)

    def test_example_rows(self):
        mib = lambda s, d=1: f"{s['bytes_per_chip'] / 2**20:,.{d}f}"
        t64 = mm.map_model(model("dense-7b"), ref(), chips=64)["strategies"]
        ten, hyb, pipe = t64["tensor"], t64["hybrid"], t64["pipeline"]
        ar = ten["collective_cycles_per_allreduce"]
        self.assertIn(f"| yes ({mib(ten)}) | {ten['layer_cycles']:,} ({ten['install_cycles_layer']} install + 2 all-reduces of {ar:,}) "
                      f"| {ten['tokens_per_s']:,.0f} | {ten['binding']} |", self.text)
        self.assertIn(f"| yes ({mib(hyb)}) | {hyb['layer_cycles']:,} ({hyb['install_cycles_layer']:,} install + 2 all-reduces of "
                      f"{hyb['collective_cycles_per_allreduce']:,}) | {hyb['tokens_per_s']:,.0f} | {hyb['binding']} |", self.text)
        self.assertEqual((hyb["stages"], hyb["tensor_group"]), (8, 8))
        self.assertIn(f"| no ({mib(pipe, 0)}) | {pipe['layer_cycles']:,} |", self.text)
        b16 = mm.map_model(model("dense-7b"), ref(), chips=64, batch=16)["strategies"]["tensor"]
        self.assertIn(f"| no ({mib(b16, 0)}) | {b16['layer_cycles']:,} |", self.text)
        t40 = mm.map_model(model("dense-7b"), ref(), chips=40)["strategies"]["tensor"]
        self.assertIn(f"| no ({mib(t40)}) | {t40['layer_cycles']:,} |", self.text)
        r70 = mm.map_model(model("dense-70b"), ref(), chips=400)
        ten70, hyb70 = r70["strategies"]["tensor"], r70["strategies"]["hybrid"]
        self.assertIn(f"| no ({mib(ten70)}) | {ten70['layer_cycles']:,} |", self.text)
        self.assertIn(f"| yes ({mib(hyb70)}) | {hyb70['layer_cycles']:,} | {hyb70['tokens_per_s']:,.0f} | {hyb70['binding']} |", self.text)
        self.assertEqual((hyb70["stages"], hyb70["tensor_group"]), (50, 8))
        self.assertIn(f"at least {r70['footprint']['min_chips_total']} chips hold the 70B model", self.text)

    def test_the_text_states_the_modelled_wiring_and_the_flat_comparison(self):
        for text in ("34 cycles", "2 L + 34 = 466", "528 + 3 × (34 + 528) = 2,214", "No 64-chip collective was run on `tsp_sim`",
                     "`--topology flat`", "**hypothetical**", "3,168 cycles per all-reduce, 4,190 tokens per second"):
            self.assertIn(text, self.text)
        self.assertEqual(mm.map_model(model("dense-7b"), ref(), chips=64)["strategies"]["tensor"]["collective_cycles_per_allreduce"], 3462)
        self.assertEqual(mm.map_model(dict(model("dense-7b"), act_bytes=1), ref(), chips=64)["strategies"]["tensor"]["collective_cycles_per_allreduce"], 2214)
        flat = mm.map_model(model("dense-7b"), ref(), chips=64, topology=mm.flat_wiring())["strategies"]["tensor"]
        self.assertEqual(flat["collective_cycles_per_allreduce"], 6 * STEP_F16)
        # The one-byte payload assumes requantisation and hypothetical direct links.
        one = mm.map_model(dict(model("dense-7b"), act_bytes=1), ref(), chips=64, topology=mm.flat_wiring())["strategies"]["tensor"]
        self.assertEqual((one["collective_cycles_per_allreduce"], round(one["tokens_per_s"])), (3168, 4190))

    def test_the_payload_sensitivity_table(self):
        rows = {1: "requantised partial sums (requantisation outside the estimate)", 2: "f16 partial sums",
                4: "the 32-bit accumulators sent unconverted"}
        for a, meaning in rows.items():
            t = mm.map_model(dict(model("dense-7b"), act_bytes=a), ref(), chips=64)["strategies"]
            label = f"{a} (default)" if a == 2 else str(a)
            self.assertIn(f"| {label} | {meaning} | {t['tensor']['collective_cycles_per_allreduce']:,} cycles | "
                          f"{t['tensor']['tokens_per_s']:,.0f} | {t['hybrid']['tokens_per_s']:,.0f} |", self.text)

    def test_the_width_sensitivity_table(self):
        import derive_profile as dpr
        with open(os.path.join(SPEC, "config", "test", "sl32.json")) as fh:
            shipped = json.load(fh)
        derived, _ = dpr.derive(ref(), 32)
        for name, cfg in (("`ref` (20 superlanes)", ref()), ("derived, 32 superlanes", derived), ("`test/sl32`, as shipped", shipped)):
            t = mm.map_model(model("dense-7b"), cfg, chips=64)["strategies"]
            link = 16 * cfg["geometry"]["superlanes"] / cfg["c2c"]["cycles_per_vector"] * 0.9
            row = (f"| {name} | {cfg['c2c']['cycles_per_vector']} and {cfg['host']['cycles_per_vector']} | {link:.2f} GB/s"
                   f"{' (above any published figure; the tool warns)' if link > 15 else ''} | "
                   f"{t['tensor']['tokens_per_s']:,.0f} | {t['hybrid']['tokens_per_s']:,.0f} |")
            self.assertIn(row, self.text)

    def test_calibration_overhead(self):
        overhead = mm.validate_measured(ref())[0]["overhead_cycles"]
        self.assertIn(f"{overhead:,} cycles reproduces", self.text)


class Cli(unittest.TestCase):
    def profile(self):
        return os.path.join(SPEC, "config", "ref.json")

    def test_text_report(self):
        r = run_cli(os.path.join(MODELS, "dense-7b.json"), self.profile(), "--chips", "64")
        self.assertEqual(r.returncode, 0, r.stderr)
        for word in ("estimate", "pipeline", "tensor", "binding", "tokens/s", "C2C"):
            self.assertIn(word, r.stdout)
        self.assertIn("percent", r.stdout)  # the validation against the measured layer is printed with every report

    def test_json(self):
        r = run_cli(os.path.join(MODELS, "dense-7b.json"), self.profile(), "--chips", "64", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(d["label"], "estimate")
        self.assertEqual(d["strategies"]["tensor"]["install_cycles_layer"], 257)
        self.assertEqual(d["validation"][0]["measured_cycles"], MEASURED_SERIAL_LAYER)

    def test_bad_inputs_are_refused_with_a_message(self):
        m = os.path.join(MODELS, "dense-7b.json")
        for args, needle in (
            ([m, self.profile(), "--chips", "0"], "--chips"),
            ([m, self.profile(), "--chips", "x"], "--chips"),
            ([m, self.profile(), "--chips", "4", "--batch", "0"], "--batch"),
            ([m, self.profile(), "--chips", "4", "--sram-fraction", "1.5"], "--sram-fraction"),
            ([m, self.profile(), "--chips", "4", "--phase", "prefill"], "--prompt-len"),
            (["/nonexistent/model.json", self.profile(), "--chips", "4"], "/nonexistent/model.json"),
            ([m, "/nonexistent/profile.json", "--chips", "4"], "/nonexistent/profile.json"),
        ):
            r = run_cli(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(needle, r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_malformed_model_files(self):
        with tempfile.TemporaryDirectory() as d:
            for change, needle in (({"heads": 7}, "heads"), ({"kv_heads": 5}, "kv_heads"), ({"weight_bytes": 4}, "weight_bytes"),
                                   ({"ffn_matrices": 4}, "ffn_matrices"), ({"layers": 0}, "layers"), ({"hidden": "big"}, "hidden")):
                bad = dict(model("dense-7b"), **change)
                path = os.path.join(d, "m.json")
                with open(path, "w") as fh:
                    json.dump(bad, fh)
                r = run_cli(path, self.profile(), "--chips", "4")
                self.assertEqual(r.returncode, 2, change)
                self.assertIn(needle, r.stderr)
                self.assertNotIn("Traceback", r.stderr)
            missing = dict(model("dense-7b"))
            del missing["vocab"]
            path = os.path.join(d, "m.json")
            with open(path, "w") as fh:
                json.dump(missing, fh)
            r = run_cli(path, self.profile(), "--chips", "4")
            self.assertEqual(r.returncode, 2)
            self.assertIn("vocab", r.stderr)


if __name__ == "__main__":
    unittest.main()
