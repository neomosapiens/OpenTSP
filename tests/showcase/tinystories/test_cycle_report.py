"""Validate cycle and timing reports for TinyStories buckets."""
import json
import os
import sys

import pytest

from opentsp_vllm.models.tinystories import paths as _paths

sys.path.insert(0, os.path.join(_paths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.tinystories import build_programs as BP
import cycle_report as CR
from opentsp_vllm.models.tinystories import program_cache as PC

GRID = [(T, Ls, C) for C in (64, 128) for T in (1, 2, 4, 8) for Ls in (1, 2, 4) if Ls <= T]


def synthetic(layers, lane64, lane128, rate=190000.0):
    """A table of `layers` layers whose cycles are exactly linear in the lanes (and 554 a logit row beyond the first)."""
    buckets = {}
    for T, Ls, C in GRID:
        c = T * (lane64 if C == 64 else lane128) + (Ls - 1) * 554
        buckets[PC.bucket_name((T, Ls, C))] = {"cycles": c, "build_seconds": 10.0 * T + 5, "iss_seconds": c / rate, "text_words": 1,
                                               "patch_sites": 1, "image_bytes": 1}
    return {f"d{layers}": {"layers": layers, "weights": "w", "buckets": buckets}}


@pytest.fixture
def table():
    t = synthetic(8, 90000, 112000)
    t.update(synthetic(2, 23000, 28000))
    return t


def test_the_lane_cost_is_the_slope_of_the_cycles_over_the_lanes(table):
    one, slope, dev = CR.lane_cost(CR.entry_of(table, 8), 64)
    assert (one, slope, dev) == (90000, 90000, 0)
    assert CR.lane_cost(CR.entry_of(table, 8), 128)[1] == 112000


def test_the_coarse_grid_wastes_lanes_only_between_its_values(table):
    e = CR.entry_of(table, 8)
    assert CR.padding_cost(e, 64, [1, 2, 4, 8])[2] == 0
    coarse, fine, waste = CR.padding_cost(e, 64, [3, 5])         # 4 and 8 lanes for 3 and 5
    assert (coarse, fine) == (6 * 90000, 4 * 90000) and waste == pytest.approx(1 / 3)


def test_tokens_per_second_do_not_depend_on_the_lane_count_when_lanes_are_serial(table):
    e = CR.entry_of(table, 8)
    rates = {T: CR.tokens_per_second(e, T, 64, CR.PLANNING_RATE) for T in (1, 2, 4, 8)}
    assert len({round(v, 9) for v in rates.values()}) == 1 and rates[1] == pytest.approx(CR.PLANNING_RATE / 90000)


def test_the_fine_grid_has_twenty_pairs_and_costs_more_to_build(table):
    assert len(CR.grid_pairs(range(1, 9))) == 20 and len(CR.grid_pairs((1, 2, 4, 8))) == 9
    e = CR.entry_of(table, 8)
    assert CR.grid_build_seconds(e, fine=True) > 2 * CR.grid_build_seconds(e)


def test_the_report_has_every_table_and_renders(table):
    r = CR.report(table)
    assert len(r["cycles8"]) == len(r["cycles2"]) == 18 and len(r["tps8"]) == 8
    assert r["preemption"]["full"][0] == 16 * CR.cyc(CR.entry_of(table, 8), (8, 1, 128))
    assert r["t_grid"]["suite"]["break_even_runs"] > 10
    text = CR.markdown(r)
    assert "P(8, 4, 128)" in text and "break-even" in text and "Replay, steps at T = 2 and 4 alternately" in text


def test_a_missing_entry_is_an_error(table):
    with pytest.raises(KeyError):
        CR.entry_of({k: v for k, v in table.items() if v["layers"] == 8}, 2)


def test_a_built_bucket_is_timed_and_the_timing_is_merged_into_the_cycle_table(tmp_path):
    model = PC.Model(2)
    cache = str(tmp_path)
    BP.build_bucket(model, (1, 1, 64), cache)
    rec = CR.time_bucket(model, (1, 1, 64), cache, repeats=2)
    assert rec["cycles"] == 8194 and rec["wall"] > 0 and rec["rate"] == pytest.approx(rec["cycles"] / rec["wall"])
    CR.time_buckets(2, cache, 1, [(1, 1, 64)], log=lambda *_: None)
    table = CR.write_cycles(cache)
    got = table[model.digest]["buckets"]["P1_1_64"]
    assert got["cycles"] == 8194 and got["iss_cycles_per_second"] > 0
    assert json.load(open(os.path.join(cache, "cycles.json")))[model.digest]["buckets"]["P1_1_64"]["iss_seconds"] == got["iss_seconds"]
    with pytest.raises(BP.BuildError, match="is not built"):
        CR.time_buckets(2, cache, 1, [(2, 1, 64)], log=lambda *_: None)


def test_the_bucket_with_the_most_text_is_found_per_queue(tmp_path):
    model = PC.Model(2)
    sizes = {(1, 1, 64): {0: 5, 1: 9}, (8, 4, 128): {0: 7, 1: 9, 2: 3}, (2, 1, 64): {0: 6, 2: 4}}
    for bucket, qs in sizes.items():
        d = PC.bucket_dir(str(tmp_path), model, bucket)
        os.makedirs(d)
        PC.write_json(os.path.join(d, "program.json"), {"blocks": [{"queue": q, "index": 0, "kind": "x", "tensors": [["t", 0, 0, w]]} for q, w in qs.items()]})
    top, short, best, n = CR.largest_text(str(tmp_path), model)
    assert (top, n) == ("P8_4_128", 3) and short == [2] and best[0] == (7, "P8_4_128") and best[2] == (4, "P2_1_64")


def test_an_incomplete_entry_is_never_picked_and_the_message_says_what_to_build(table):
    """PERF-1: the report took the first entry of the layer count, a 15-bucket one, and died with a KeyError on P2_1_128."""
    partial = synthetic(2, 99999, 99999)["d2"]
    del partial["buckets"]["P2_1_128"]
    t = dict(table)
    t = {"a_incomplete": partial, **t}
    assert CR.entry_of(t, 2) is table["d2"]
    assert len(CR.report(t)["cycles2"]) == 18
    with pytest.raises(KeyError, match=r"no complete entry of 2 layers.*a_incomplete.*lacks 1 of 18.*P2_1_128.*LAYERS=2 just vllm-programs"):
        CR.entry_of({"a_incomplete": partial}, 2)
    with pytest.raises(KeyError, match="a_incomplete.*lacks 1 of 18"):
        CR.entry_of(t, 2, "a_inc")


def test_two_complete_entries_need_an_explicit_digest_and_the_report_names_the_one_it_read(table):
    other = synthetic(2, 1000, 2000)["d2"]
    t = {**table, "e_other": other}
    with pytest.raises(KeyError, match="2 complete entries of 2 layers.*--digest"):
        CR.entry_of(t, 2)
    r = CR.report(t, digests={2: "e_other"})
    assert r["digests"] == {8: "d8", 2: "e_other"}
    assert r["cycles2"][0][1] == 1000 and CR.report(t, digests={2: "d2"})["cycles2"][0][1] == 23000
