"""The step plan of the SHIP-style pipeline (`ship_plan`): lanes, slots (the Scatter maps' source) and mask encodings from vLLM's block tables."""

import os
import sys

import numpy as np
import pytest

from opentsp_vllm import ship_plan as sp
from opentsp_vllm.step_plan import PlanRequest, StepPlanError

VOCAB = 50257


def req(rid, known, blocks, computed, n):
    return PlanRequest(rid, list(known), list(blocks), computed, n)


def plan(*reqs):
    return sp.plan_step(list(reqs), vocab=VOCAB)


def test_the_pool_is_64_pages_of_16_slots_and_block_0_is_vllms_null_block():
    assert (sp.PAGE, sp.PAGES, sp.POOL_SLOTS, sp.NUM_BLOCKS) == (16, 64, 1024, 65)
    assert sp.page_of(1) == 0 and sp.page_of(64) == 63


def test_a_decode_lane_has_the_slot_of_its_block_and_the_encoding_of_its_pages():
    p = plan(req("a", range(100, 121), [5, 2], 20, 1))
    lane = p.lanes[0]
    assert (lane.token, lane.position, lane.block_table, lane.logits) == (120, 20, (5, 2), True)
    # position 20 is offset 4 of block 2 = page 1: slot 16 + 4
    assert p.patches["slots"].tolist() == [20]
    assert p.patches["ids"].tolist() == [120] and p.patches["positions"].tolist() == [20]
    counts = p.patches["counts"][0]
    # block 5 is page 4 (full), block 2 is page 1 (5 slots live: offsets 0 .. 4); every other page is not the request's
    assert counts[4] == 16 and counts[1] == 5 and counts.sum() == 21
    assert p.bucket == (1, 1, 1024) and p.logit_req_ids == ("a",)


def test_the_slot_follows_the_block_not_the_position():
    p = plan(req("a", range(40), [64, 1, 33], 35, 1))
    assert p.patches["slots"].tolist() == [32 * 16 + 3]  # block 33 is page 32, offset 35 mod 16 = 3
    assert p.patches["counts"][0, [63, 0, 32]].tolist() == [16, 16, 4]


def test_a_chunk_is_one_lane_per_token_in_ascending_positions_and_only_the_last_needs_logits():
    p = plan(req("a", range(10, 30), [1, 2], 3, 17))  # positions 3 .. 19 of 20 known tokens
    assert [ln.position for ln in p.lanes] == list(range(3, 20))
    assert [ln.logits for ln in p.lanes] == [False] * 16 + [True]
    assert p.n_real == 17 and p.n_logits == 1 and p.padding_lanes == 0
    # lane i sees positions 0 .. 3 + i: the encodings grow by one slot per lane
    assert [int(c.sum()) for c in p.patches["counts"]] == list(range(4, 21))


def test_two_requests_keep_their_pages_apart():
    p = plan(req("a", range(10, 20), [7], 9, 1), req("b", range(50, 70), [3, 9], 19, 1))
    ca, cb = p.patches["counts"]
    assert np.flatnonzero(ca).tolist() == [6] and np.flatnonzero(cb).tolist() == [2, 8]
    assert p.logit_req_ids == ("a", "b") and p.patches["slots"].tolist() == [6 * 16 + 9, 8 * 16 + 3]


def test_a_prefix_cached_page_is_in_the_encodings_of_both_requests():
    shared = [4]
    pa = plan(req("a", range(100, 118), shared + [5], 17, 1))
    pb = plan(req("b", range(100, 117), shared + [6], 16, 1))
    assert pa.patches["counts"][0, 3] == 16 and pb.patches["counts"][0, 3] == 16
    assert pa.patches["counts"][0, 4] == 2 and pb.patches["counts"][0, 5] == 1


@pytest.mark.parametrize(
    "r, msg",
    [
        (req("a", [1, 2], [1], 0, 0), "0 tokens scheduled"),
        (req("a", [1, 2], [1], 0, 3), "exceed the 2 known"),
        (req("a", [1, VOCAB], [1], 0, 2), "outside the vocabulary"),
        (req("a", [1, 2], [65], 0, 2), "outside the pool of 65 blocks"),
        (req("a", list(range(20)), [1], 0, 20), "does not cover position 19"),
        (req("a", list(range(20)), [1, 0], 0, 20), "null block"),
        (req("a", list(range(20)), [3, 3], 0, 20), "twice"),
    ],
)
def test_a_request_the_pool_cannot_hold_is_refused_before_the_chips_are_touched(r, msg):
    with pytest.raises(StepPlanError, match=msg):
        plan(r)


def test_the_longest_request_fills_the_pool():
    p = plan(req("a", range(1024), list(range(1, 65)), 1023, 1))
    assert p.patches["counts"][0].tolist() == [16] * 64 and p.patches["slots"].tolist() == [63 * 16 + 15]
    with pytest.raises(StepPlanError, match="beyond the pool"):
        plan(req("a", range(1025), list(range(1, 65)), 1024, 1))


def test_no_request_is_refused():
    with pytest.raises(StepPlanError, match="no request"):
        sp.plan_step([], vocab=VOCAB)


def test_the_encoding_is_the_stand_alone_drivers_counts_for_a_contiguous_sequence():
    """For blocks 1, 2, ... (pages 0, 1, ...) the encoding is `ship_system.counts_for(position + 1)` flattened: the stand-alone driver's own."""
    root = os.environ.get("OPENTSP_ROOT") or os.path.abspath(os.path.join(os.path.dirname(sp.__file__), "..", "..", ".."))
    showcase = os.path.join(root, "tests", "showcase", "gpt2small")
    sys.path.insert(0, showcase)
    try:

        
        from opentsp_vllm.models.gpt2 import system as ship_system
    finally:
        sys.path.remove(showcase)
    for pos in (0, 15, 16, 17, 127, 128, 500, 1023):
        want = ship_system.counts_for(pos + 1).reshape(-1)
        got = sp.encoding(list(range(1, pos // 16 + 2)), pos)
        assert np.array_equal(got, want), pos


def test_eight_rows_keep_separate_pages_in_the_short_pool():
    ids = tuple(f"r{i}" for i in range(8))
    p = sp.plan_step([req(rid, [100 + i], [i + 9], 0, 1) for i, rid in enumerate(ids)],
                     vocab=VOCAB, kv_chunks=2)
    assert p.bucket == (8, 8, 256) and p.logit_req_ids == ids
    assert p.patches["slots"].tolist() == list(range(128, 256, 16))
    assert p.patches["counts"].shape == (8, 16)
    assert np.array_equal(p.patches["counts"][:, 8:], np.eye(8, dtype=np.int32))
    with pytest.raises(StepPlanError, match="outside the pool of 17 blocks"):
        sp.plan_step([req("a", [1], [17], 0, 1)], vocab=VOCAB, kv_chunks=2)


def test_short_pool_crosses_physical_chunk_boundary_and_rejects_position_256():
    p = sp.plan_step([req("a", range(17), [8, 9], 16, 1)], vocab=VOCAB, kv_chunks=2)
    assert p.patches["slots"].tolist() == [128]
    assert p.patches["counts"][0, 7:9].tolist() == [16, 1]
    with pytest.raises(StepPlanError, match="beyond the pool of 256 slots"):
        sp.plan_step([req("a", range(257), range(1, 17), 256, 1)], vocab=VOCAB, kv_chunks=2)


def test_ras_host_plans_the_device_pool_shape_without_contacting_the_device():
    from opentsp_vllm.ras import ManagedChip
    from opentsp_vllm.ship_chip import ShipChip

    managed = object.__new__(ManagedChip)
    managed.name = "ship"
    managed.info = ShipChip(kv_chunks=2).info
    p = managed.plan_step([req("a", [1], [16], 0, 1)])
    assert p.bucket == (1, 1, 256)
    assert p.patches["counts"].shape == (1, 16)
    with pytest.raises(StepPlanError, match="outside the pool of 17 blocks"):
        managed.plan_step([req("a", [1], [17], 0, 1)])
