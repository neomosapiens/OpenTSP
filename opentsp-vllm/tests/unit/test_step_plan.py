"""StepPlan ( step 3, 3.3, 3.4): scheduled requests and request state to lanes, bucket and patch values."""

import itertools

import numpy as np
import pytest

import reference_loader as oracle
from opentsp_vllm import step_plan as sp
from opentsp_vllm.step_plan import PlanRequest, StepPlanError, plan_step

VOCAB = 2000


def req(rid, known, blocks, computed, n):
    return PlanRequest(rid, list(known), list(blocks), computed, n)


def plan(*reqs, **kw):
    return plan_step(list(reqs), vocab=kw.pop("vocab", VOCAB), **kw)


def test_grid_and_constants_are_the_designs():
    reference_paged, _, _ = oracle.load()
    assert (sp.BLOCK, sp.POOL_BLOCKS, sp.NULL) == (reference_paged.BLOCK, reference_paged.POOL_BLOCKS, reference_paged.NULL)
    assert (sp.GRID_T, sp.GRID_L, sp.GRID_C) == (reference_paged.GRID_T, reference_paged.GRID_L, reference_paged.GRID_C)
    assert sp.SINK == sp.POOL_BLOCKS


def test_choose_bucket_equals_the_oracles_on_the_whole_domain():
    reference_paged, _, _ = oracle.load()
    for n, l, pos in itertools.product(range(1, 9), range(0, 5), range(0, 128)):
        if l > n:
            continue
        assert sp.choose_bucket(n, l, pos) == reference_paged.choose_bucket(n, l, pos), (n, l, pos)


def test_choose_bucket_refuses_what_no_bucket_holds():
    for args in ((9, 1, 0), (4, 5, 0), (1, 1, 128)):
        with pytest.raises(ValueError):
            sp.choose_bucket(*args)
    # the test model has programs for C = 64 only
    with pytest.raises(ValueError):
        sp.choose_bucket(1, 1, 64, grid_c=(64,))


def test_decode_batch_of_three_runs_in_t4_l4_with_one_padding_lane():
    p = plan(
        req("a", range(10, 30), [1, 2], 19, 1),
        req("b", range(40, 45), [3], 4, 1),
        req("c", range(50, 70), [4, 5], 19, 1),
    )
    assert p.bucket == (4, 4, 64)
    assert (p.n_real, p.n_logits, p.padding_lanes) == (3, 3, 1)
    assert p.logit_req_ids == ("a", "b", "c")
    assert [(ln.req_id, ln.token, ln.position) for ln in p.lanes[:3]] == [("a", 29, 19), ("b", 44, 4), ("c", 69, 19)]
    assert [ln.block_table for ln in p.lanes[:3]] == [(1, 2), (3,), (4, 5)]
    pad = p.lanes[3]
    assert (pad.req_id, pad.token, pad.logits) == (None, None, False)
    assert p.context_padding == 64 - 20


def test_logit_lanes_come_first_in_batch_order_and_prefill_lanes_follow():
    # a is mid-prefill (3 of 9 known tokens scheduled), b decodes, c finishes its prompt with a chunk of 2
    p = plan(
        req("a", range(100, 109), [1], 0, 3),
        req("b", range(200, 206), [2], 5, 1),
        req("c", range(300, 308), [3], 6, 2),
    )
    assert p.n_real == 6 and p.bucket == (8, 2, 64)
    assert p.logit_req_ids == ("b", "c")
    assert [(ln.req_id, ln.position, ln.logits) for ln in p.lanes[:6]] == [
        ("b", 5, True),
        ("c", 7, True),
        ("a", 0, False),
        ("a", 1, False),
        ("a", 2, False),
        ("c", 6, False),
    ]
    assert [ln.token for ln in p.lanes[:6]] == [205, 307, 100, 101, 102, 306]


def test_a_chunk_that_does_not_reach_the_last_known_token_needs_no_logits():
    p = plan(req("a", range(20), [1, 2], 0, 8))
    assert (p.n_real, p.n_logits, p.logit_req_ids) == (8, 0, ())
    assert p.bucket == (8, 1, 64)  # L is at least 1: the grid has no L = 0
    assert [ln.position for ln in p.lanes] == list(range(8))


def test_the_last_chunk_of_a_prompt_asks_logits_for_its_last_lane_only():
    p = plan(req("a", range(10), [1], 4, 6))
    assert p.logit_req_ids == ("a",)
    assert [ln.logits for ln in p.real_lanes] == [True, False, False, False, False, False]
    assert p.real_lanes[0].position == 9  # the logit lane is the last token


def test_a_resumed_request_recomputes_from_position_zero_and_samples_at_its_end():
    known = list(range(5)) + [900, 901, 902, 903]  # 5 prompt tokens and 4 sampled ones
    mid = plan(req("r", known, [7, 8], 0, 4))
    assert mid.n_logits == 0
    last = plan(req("r", known, [7, 8], 6, 3))
    assert last.logit_req_ids == ("r",) and last.real_lanes[0].token == 903
    assert [ln.position for ln in last.real_lanes] == [8, 6, 7]


def test_context_bucket_follows_the_largest_position():
    assert plan(req("a", range(64), [1, 2, 3, 4], 63, 1)).bucket[2] == 64
    p = plan(req("a", range(65), [1, 2, 3, 4, 5], 64, 1))
    assert p.bucket[2] == 128 and p.context_padding == 128 - 65
    assert plan(req("a", range(70), [1] * 5, 0, 8)).bucket == (8, 1, 64)


def test_block_tables_hold_only_the_blocks_up_to_the_lanes_position():
    # the scheduler may have allocated more blocks than the position needs
    p = plan(req("a", range(20), [1, 2, 3, 4], 19, 1))
    assert p.real_lanes[0].block_table == (1, 2)


def test_grid_c_limits_the_buckets():
    plan(req("a", range(64), [1, 2, 3, 4], 63, 1), grid_c=(64,))
    with pytest.raises(StepPlanError, match="context"):
        plan(req("a", range(65), [1] * 5, 64, 1), grid_c=(64,))


@pytest.mark.parametrize(
    "bad, match",
    [
        (req("a", range(20), [1], 19, 1), "block table"),  # position 19 needs two blocks
        (req("a", [VOCAB] * 3, [1], 2, 1), "token"),
        (req("a", [3, -1], [1], 1, 1), "token"),
        (req("a", range(5), [1, 128], 4, 1), "block"),  # block 128 is the sink, not a block of a request
        (req("a", range(5), [-1], 4, 1), "block"),
        (req("a", range(5), [0], 4, 1), "null block"),
        # the null block inside the prefix of the table, not at the position read: the emitter refuses it too
        (req("a", range(40), [0, 2, 3], 39, 1), "null block"),
        (req("a", range(40), [1, 0, 3], 39, 1), "null block"),
        (req("a", range(5), [1], 4, 0), "scheduled"),
        (req("a", range(5), [1], 4, 2), "known"),  # more lanes than tokens that exist
        (req("a", range(5), [1], 6, 1), "computed"),
    ],
)
def test_patch_values_are_range_checked_before_the_chip_is_touched(bad, match):
    with pytest.raises(StepPlanError, match=match):
        plan(bad)


def test_more_than_eight_lanes_or_four_logit_rows_are_refused():
    with pytest.raises(StepPlanError, match="lanes"):
        plan(req("a", range(20), [1, 2], 0, 9))
    with pytest.raises(StepPlanError, match="logit"):
        plan(*[req(f"r{i}", range(5), [i + 1], 4, 1) for i in range(5)])
    with pytest.raises(StepPlanError, match="no request"):
        plan()


def test_the_step_does_not_mutate_its_inputs():
    r = req("a", range(10), [1], 4, 6)
    before = (list(r.token_ids), list(r.block_ids), r.num_computed, r.num_scheduled)
    plan(r)
    assert before == (list(r.token_ids), list(r.block_ids), r.num_computed, r.num_scheduled)


# ---- the patch values (pure data) -----------------------------------------------------------------------------


def test_patch_kinds_are_the_designs_and_a_subset_can_be_asked_for():
    assert sp.PATCH_KINDS == ("kblock", "vblock", "kwrite", "vwrite", "emb_part", "emb_map", "wpe", "mask")
    p = plan(req("a", range(5), [1], 4, 1), patch_kinds=("wpe", "mask"))
    assert set(p.patches) == {"wpe", "mask"}
    with pytest.raises(StepPlanError, match="patch kind"):
        plan(req("a", range(5), [1], 4, 1), patch_kinds=("nope",))


def test_patch_shapes_and_dtypes():
    p = plan(req("a", range(5), [1], 4, 1), req("b", range(70), [2, 3, 4, 5, 6], 69, 1), vocab=50257)
    T, _, C = p.bucket
    assert (T, C) == (2, 128)
    shapes = {k: v.shape for k, v in p.patches.items()}
    assert shapes == {
        "kblock": (2, 8),
        "vblock": (2, 8),
        "kwrite": (2, 8, 2),
        "vwrite": (2, 2),
        "emb_part": (2, 3),  # 50257 vocabulary rows packed 5 per word: 10052 words, 4096 per part
        "emb_map": (2,),
        "wpe": (2,),
        "mask": (2,),
    }
    assert all(v.dtype == np.int32 for v in p.patches.values())


def test_real_lane_patch_values():
    # token 12347 = 5 * 2469 + 2 (embedding word 2469, segment 2), at position 35 = block 2 of the table, row 3
    p = plan(req("a", [1, 2] + [0] * 33 + [12347], [9, 8, 7, 6, 5], 35, 1), vocab=50257)
    assert p.bucket == (1, 1, 64)
    q = p.patches
    assert list(q["kblock"][0]) == [9, 8, 7, 0]
    assert list(q["vblock"][0]) == [9, 8, 7, 0]
    assert q["wpe"][0] == 35 and q["mask"][0] == 35
    assert q["emb_map"][0] == 12347 % 5 == 2
    assert list(q["emb_part"][0]) == [2469, sp.PART_ROWS, sp.PART_ROWS]


def test_embedding_patch_selects_the_owning_part_and_the_zero_word_elsewhere():
    toks = [0, 4999 * 5 + 3, 4096 * 5 + 1, 8192 * 5 + 4, 50256]
    parts = []
    for i, t in enumerate(toks):
        p = plan(req("a", [t] * (i + 1), [1], i, 1), vocab=50257)
        parts.append((list(p.patches["emb_part"][0]), int(p.patches["emb_map"][0])))
    z = sp.PART_ROWS
    assert parts == [
        ([0, z, z], 0),
        ([z, 4999 - 4096, z], 3),
        ([z, 0, z], 1),
        ([z, z, 0], 4),
        ([z, z, 50256 // 5 - 8192], 1),
    ]


def test_write_patches_send_the_real_address_to_one_slice_pair_and_the_sink_to_the_other_seven():
    # position 21: block index 1 of the table, row 5; K row 5 lives in slice pair 5 mod 8, word half 5 div 8
    p = plan(req("a", range(22), [3, 4], 21, 1))
    assert list(p.patches["vwrite"][0]) == [4, 5]
    kw = p.patches["kwrite"][0]
    assert [int(kw[pair, 0]) for pair in range(8)] == [sp.SINK] * 5 + [4] + [sp.SINK] * 2
    assert [int(kw[pair, 1]) for pair in range(8)] == [0] * 8
    # position 28 -> row 12: pair 4, half 1
    p = plan(req("a", range(29), [3, 4], 28, 1))
    kw = p.patches["kwrite"][0]
    assert [int(kw[pair, 0]) for pair in range(8)] == [sp.SINK] * 4 + [4] + [sp.SINK] * 3
    assert int(kw[4, 1]) == 1


def test_the_k_word_half_changes_exactly_at_row_eight_of_a_block():
    # K row o lives in slice pair o mod 8, word half o div 8: the half is 0 up to row 7, 1 from row 8, and 0 again at row 0 of the next block
    for position, pair, half, block in ((7, 7, 0, 3), (8, 0, 1, 3), (15, 7, 1, 3), (16, 0, 0, 4), (23, 7, 0, 4), (24, 0, 1, 4)):
        p = plan(req("a", range(position + 1), [3, 4], position, 1))
        kw = p.patches["kwrite"][0]
        assert [int(kw[q, 0]) for q in range(8)] == [sp.SINK] * pair + [block] + [sp.SINK] * (7 - pair), position
        assert [int(kw[q, 1]) for q in range(8)] == [half] * 8, position


def test_padding_lane_patch_values_are_zero_and_sink():
    p = plan(req("a", range(22), [3, 4], 21, 1), vocab=50257)  # T = 1: no padding lane
    assert p.padding_lanes == 0
    p = plan(req("a", range(22), [3, 4], 21, 1), req("b", range(3), [5], 2, 1), req("c", range(3), [6], 2, 1), vocab=50257)
    assert p.bucket[0] == 4
    q = p.patches
    assert list(q["kblock"][3]) == [0, 0, 0, 0]
    assert list(q["vblock"][3]) == [0, 0, 0, 0]
    assert list(q["kwrite"][3][:, 0]) == [sp.SINK] * 8
    assert list(q["vwrite"][3]) == [sp.SINK, 0]
    assert list(q["kwrite"][3][:, 1]) == [0] * 8
    assert list(q["emb_part"][3]) == [sp.PART_ROWS] * 3
    assert (q["emb_map"][3], q["wpe"][3], q["mask"][3]) == (0, 0, 0)


def test_plan_is_deterministic():
    r = [req("a", range(30), [1, 2], 20, 7), req("b", range(9), [3], 8, 1)]
    a, b = plan_step(r, vocab=VOCAB), plan_step(r, vocab=VOCAB)
    assert a.bucket == b.bucket and a.lanes == b.lanes
    assert all(np.array_equal(a.patches[k], b.patches[k]) for k in a.patches)


def test_patch_count_is_the_number_of_values():
    p = plan(req("a", range(5), [1], 4, 1))
    assert p.patch_count == sum(v.size for v in p.patches.values())


def test_a_plan_error_is_a_chip_error_and_a_value_error():
    # : an out-of-range patch value or step is a TspChipError; ValueError stays for callers of plan_step
    from opentsp_vllm.chip import TspChipError
    from opentsp_vllm.step_plan import StepPlanError

    assert issubclass(StepPlanError, TspChipError) and issubclass(StepPlanError, ValueError)
