""" : prefix caching, chunked prefill and preemption through a real vLLM engine on the TSP platform, on the test
model (sec. 6.6), on `ReferenceChip` first and then on `IssChip`.

Every logit row the chip returns is compared bit for bit with the row `paged_reference.generate_paged` gives the same token of the same
request alone (sec. 5.11: batch, chunk, prefix-hit and preemption invariance), and the token ids with its greedy ids.
"""

import numpy as np
import pytest
from vllm import SamplingParams

import scheduler_helpers as H

CHIPS = ["exact", "iss"]
MODEL = "test"
BLOCK = 16
BUDGET = 8


def prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(1, 50257, n)]


def greedy(n):
    return SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)


def chunks(probe, name):
    """The number of tokens the scheduler gave the request in each step it was scheduled."""
    return [n for tokens, _, _ in probe.schedules for rid, n in tokens.items() if H.name_of(rid) == name]


def step_sizes(probe):
    return [sum(tokens.values()) for tokens, _, _ in probe.schedules]


def cached(outs, name):
    return outs[name].num_cached_tokens or 0


def tokens_of(outs, name):
    return list(outs[name].outputs[0].token_ids)


def logit_rows(e):
    """The logit rows of the recorded steps of a run of one request, in order."""
    return np.stack([np.asarray(result.logits)[0] for _, result in e.steps if len(result.logits)])


# -- prefix caching  ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chip", CHIPS)
def test_a_request_after_a_shared_prefix_request_hits_the_cache_and_equals_a_cold_run(make_llm, chip):
    e = make_llm(chip, MODEL)
    donor = prompt(1, 21)
    shared = donor[:BLOCK] + prompt(2, 5)
    n = 3
    probe = H.probe(e)

    cold = e.generate([shared], greedy(n))[0]
    assert (cold.num_cached_tokens or 0) == 0
    cold_logits, cold_lanes = logit_rows(e), sum(plan.n_real for plan, _ in e.steps)
    assert list(cold.outputs[0].token_ids) == H.reference_reference(shared, n)[0]
    assert np.array_equal(cold_logits, H.reference_reference(shared, n)[1])

    e.llm.reset_prefix_cache()
    e.generate([donor], greedy(n))
    e.clear()
    probe.clear()
    warm = e.generate([shared], greedy(n))[0]
    assert warm.num_cached_tokens == BLOCK, "the first block of the donor is the request's prefix"
    assert list(warm.outputs[0].token_ids) == list(cold.outputs[0].token_ids)
    assert np.array_equal(logit_rows(e), cold_logits), "a prefix hit changes no logit"
    assert sum(plan.n_real for plan, _ in e.steps) == cold_lanes - BLOCK, "the cached block is not recomputed"
    assert step_sizes(probe)[0] == len(shared) - BLOCK, "the uncached rest of the prompt fits one step"

    e.llm.reset_prefix_cache()
    e.clear()
    again = e.generate([shared], greedy(n))[0]
    assert (again.num_cached_tokens or 0) == 0, "reset_prefix_cache drops the blocks"
    assert np.array_equal(logit_rows(e), cold_logits)


@pytest.mark.parametrize("chip", CHIPS)
def test_a_repeated_prompt_hits_every_usable_block_but_recomputes_its_last_token(make_llm, chip):
    e = make_llm(chip, MODEL)
    p = prompt(3, 2 * BLOCK)
    n = 2
    first = e.generate([p], greedy(n))[0]
    assert (first.num_cached_tokens or 0) == 0
    e.clear()
    second = e.generate([p], greedy(n))[0]
    assert second.num_cached_tokens == BLOCK, "the last token of the prompt is always computed: the second block is not used"
    assert list(second.outputs[0].token_ids) == list(first.outputs[0].token_ids) == H.reference_reference(p, n)[0]
    assert np.array_equal(logit_rows(e), H.reference_reference(p, n)[1])


@pytest.mark.parametrize("chip", CHIPS)
def test_a_request_hits_the_block_another_request_fills_in_the_same_step(make_llm, chip):
    """`r` decodes while `a` fills the 16 tokens of block 0 in the last chunk of its prompt; the scheduler admits `b` in that very
    step with the block already cached: `b` reads words that this step writes (ordering of the K and V writes)."""
    e = make_llm(chip, MODEL)
    pr, pa = prompt(4, 3), prompt(5, BLOCK)
    pb = pa + prompt(6, 4)
    probe = H.probe(e)
    outs = H.run(e, {0: [("r", pr), ("a", pa), ("b", pb)]}, {"r": greedy(8), "a": greedy(3), "b": greedy(3)})
    assert cached(outs, "b") == BLOCK and cached(outs, "a") == 0
    step = next(i for i, (_, _, new) in enumerate(probe.schedules) if any(H.name_of(r) == "b" for r in new))
    tokens = {H.name_of(r): n for r, n in probe.schedules[step][0].items()}
    assert tokens["a"] == 4, "this step completes block 0 of `a`"
    assert sum(tokens.values()) <= BUDGET
    requests = {"r": (pr, 8), "a": (pa, 3), "b": (pb, 3)}
    for name, (p, n) in requests.items():
        assert tokens_of(outs, name) == H.reference_reference(p, n)[0], name
    assert H.check_rows_against_reference(e, requests) == {"r": 8, "a": 3, "b": 3}


@pytest.mark.parametrize("chip", CHIPS)
def test_requests_added_together_share_two_blocks_and_every_logit_equals_reference(make_llm, chip):
    e = make_llm(chip, MODEL)
    pa = prompt(7, 2 * BLOCK + 4)
    pb = pa[: 2 * BLOCK] + prompt(8, 5)
    outs = H.run(e, {0: [("a", pa), ("b", pb)]}, greedy(3))
    assert cached(outs, "a") == 0 and cached(outs, "b") == 2 * BLOCK
    for name, p in (("a", pa), ("b", pb)):
        assert tokens_of(outs, name) == H.reference_reference(p, 3)[0], name
    assert H.check_rows_against_reference(e, {"a": (pa, 3), "b": (pb, 3)}) == {"a": 3, "b": 3}


@pytest.mark.parametrize("chip", CHIPS)
def test_an_evicted_prefix_misses_and_the_request_still_equals_reference(make_llm, chip):
    """With 5 usable blocks, four other requests that fill whole blocks push the donor's block out of the LRU free queue."""
    e = make_llm(chip, MODEL, num_gpu_blocks_override=6, enable_prefix_caching=True)
    donor = prompt(40, BLOCK + 1)
    shared = donor[:BLOCK] + prompt(41, 3)
    e.generate([donor], greedy(1))
    hit = e.generate([shared], greedy(2))[0]
    assert hit.num_cached_tokens == BLOCK, "the block is still cached while the pool is quiet"
    fillers = [prompt(50 + i, 2 * BLOCK + 1) for i in range(3)]
    for p in fillers:
        e.generate([p], greedy(1))
    e.clear()
    miss = e.generate([shared], greedy(2))[0]
    assert (miss.num_cached_tokens or 0) == 0, "the donor's block was evicted"
    assert list(miss.outputs[0].token_ids) == list(hit.outputs[0].token_ids) == H.reference_reference(shared, 2)[0]
    assert np.array_equal(logit_rows(e), H.reference_reference(shared, 2)[1])


# -- chunked prefill  ------------------------------------------------------------------------------------


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("length,expected", [(1, [1]), (BUDGET, [8]), (BUDGET + 1, [8, 1]), (20, [8, 8, 4])])
def test_a_prompt_is_prefilled_in_chunks_of_the_budget_and_equals_reference(make_llm, chip, length, expected):
    e = make_llm(chip, MODEL)
    p = prompt(10 + length, length)
    n = 3
    probe = H.probe(e)
    out = e.generate([p], greedy(n))[0]
    assert step_sizes(probe) == expected + [1] * (n - 1), "the prompt in chunks of at most 8, then one token per decode step"
    assert list(out.outputs[0].token_ids) == H.reference_reference(p, n)[0]
    assert np.array_equal(logit_rows(e), H.reference_reference(p, n)[1])
    assert e.steps[0][0].bucket[0] == {1: 1, 8: 8, 9: 8, 20: 8}[length], "a short prompt runs in one step of the smallest T that holds it"


@pytest.mark.parametrize("chip", CHIPS)
def test_two_prompts_share_the_budget_of_each_step(make_llm, chip):
    e = make_llm(chip, MODEL)
    pa, pb = prompt(21, 20), prompt(22, 5)
    probe = H.probe(e)
    outs = H.run(e, {0: [("a", pa), ("b", pb)]}, greedy(3))
    assert all(size <= BUDGET for size in step_sizes(probe))
    assert chunks(probe, "a")[0] == BUDGET
    assert any(len(tokens) == 2 and sum(tokens.values()) == BUDGET for tokens, _, _ in probe.schedules), "a step is shared by both"
    for name, p in (("a", pa), ("b", pb)):
        assert tokens_of(outs, name) == H.reference_reference(p, 3)[0], name
    assert H.check_rows_against_reference(e, {"a": (pa, 3), "b": (pb, 3)}) == {"a": 3, "b": 3}


@pytest.mark.parametrize("chip", CHIPS)
def test_a_smaller_budget_changes_neither_tokens_nor_logits(make_llm, chip):
    e = make_llm(chip, MODEL, max_num_batched_tokens=4)
    p = prompt(23, 20)
    probe = H.probe(e)
    out = e.generate([p], greedy(3))[0]
    assert step_sizes(probe) == [4, 4, 4, 4, 4, 1, 1], "a prompt of 20 tokens in chunks of 4"
    assert list(out.outputs[0].token_ids) == H.reference_reference(p, 3)[0]
    assert np.array_equal(logit_rows(e), H.reference_reference(p, 3)[1])


# -- preemption  -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("prefix_caching", [True, False])
def test_preemption_and_every_request_equals_its_solo_run(make_llm, chip, prefix_caching):
    """6 blocks, 5 usable: three requests of 12 prompt tokens and 12 new tokens each need a second block at token 17. With the prefix
    cache on, the victim's full block is still cached when it resumes and nothing is recomputed ; with it off the
    victim recomputes the tokens it had."""
    e = make_llm(chip, MODEL, num_gpu_blocks_override=6, enable_prefix_caching=prefix_caching)
    requests = {name: (prompt(30 + i, 12), 12) for i, name in enumerate("abc")}
    solos = {name: e.solo(p, greedy(n)) for name, (p, n) in requests.items()}
    probe = H.probe(e)
    e.clear()
    outs = H.run(e, {0: [(name, p) for name, (p, _) in requests.items()]}, greedy(12))
    assert len(probe.preemptions) > 0, "the pool is too small for three requests: the scheduler preempts"
    assert any(resumed for _, resumed, _ in probe.schedules), "a preempted request resumes with new blocks"
    for name, (p, n) in requests.items():
        assert tokens_of(outs, name) == solos[name][0] == H.reference_reference(p, n)[0], name
    H.check_rows_against_reference(e, requests)
    lanes = {name: H.lanes_of(e, name) for name in requests}
    assert all(n >= 23 for n in lanes.values()), "a request computes its prompt and all but its last token"
    if prefix_caching:
        assert sum(lanes.values()) == 3 * 23, "the victim's cached block is hit when it resumes: nothing is recomputed"
    else:
        assert sum(lanes.values()) > 3 * 23, "the victim recomputed the tokens it had"
    assert e.llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.usage == 0, "every block is free again"
