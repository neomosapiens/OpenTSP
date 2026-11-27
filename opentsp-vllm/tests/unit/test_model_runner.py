"""TspModelRunner  on ReferenceChip with synthetic SchedulerOutputs: request state (new, cached, resumed, finished,
preempted, unscheduled), block copies and zeroing, the step plan, the chip run, sampling with vLLM's own Sampler and
InputBatch, and ModelRunnerOutput. The tokens are checked against the oracle's `generate_paged`."""

import numpy as np
import pytest
import torch

import reference_loader as oracle
import vllm_helpers as H
from reference_chip import ReferenceChip
from opentsp_vllm.model_runner import TspModelRunner
from opentsp_vllm.chip import TspChipError
from scripted import Alloc, Driver, run
from vllm.sampling_params import SamplingParams

reference_paged, W, config = oracle.load()
CFG = config.ModelConfig(layers=2, vocab=2000, max_pos=128)


@pytest.fixture(scope="module")
def weights():
    return W.synthetic(CFG, seed=1)


@pytest.fixture(scope="module")
def vllm_config(tmp_path_factory):
    return H.engine_config(H.write_model_dir(tmp_path_factory.mktemp("rm")))


@pytest.fixture
def env(vllm_config):
    chip = ReferenceChip(CFG, grid_c=(64,))
    chip.load_model("synthetic:1")
    runner = TspModelRunner(vllm_config, chip)
    return runner, chip, Driver(runner), Alloc()


def prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(1, CFG.vocab, n)]


def solo(w, p, n_new):
    """The oracle's greedy continuation: the n_new generated tokens and the logits rows that produced them."""
    g = reference_paged.generate_paged(w, p, n_new, C=64)
    rows = [g.steps[pos].logits for pos in range(len(p) - 1, len(g.tokens) - 1)]
    return g.tokens[len(p):], rows


# ---- one request, from a new request to the oracle's tokens ---------------------------------------------------------------


def test_a_chunked_prefill_and_greedy_decode_equals_the_oracle_bit_for_bit(env, weights):
    runner, chip, d, alloc = env
    p = prompt(1, 20)
    d.add("a", p, [])
    run(d, {"a": 8}, alloc)
    want, rows = solo(weights, p, 8)
    assert d.reqs["a"]["out"] == want
    got = [r.logits[0] for r in chip.results if r.logits.shape[0]]
    assert len(got) == len(rows) and all(np.array_equal(a, b) for a, b in zip(got, rows))


def test_the_output_of_a_step_lists_the_batch_with_empty_lists_for_requests_still_in_prefill(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(1, 12), alloc.take(1))
    out = d.step({"a": 8})
    assert out.req_ids == ["a"] and out.req_id_to_index == {"a": 0}
    assert out.sampled_token_ids == [[]] and out.logprobs is None
    assert chip.plans[-1].n_logits == 0
    out = d.step({"a": 4})  # the chunk reaches the last prompt token
    assert len(out.sampled_token_ids[0]) == 1 and 0 <= out.sampled_token_ids[0][0] < CFG.vocab


def test_the_runner_appends_the_sampled_token_to_its_own_request_state(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(2, 5), alloc.take(1))
    out = d.step({"a": 5})
    tok = out.sampled_token_ids[0][0]
    assert runner.requests["a"].output_token_ids == [tok]
    assert runner.requests["a"].num_computed_tokens == 0  # the scheduler's value of the next step replaces it
    d.step({"a": 1})
    assert chip.plans[-1].real_lanes[0].token == tok and chip.plans[-1].real_lanes[0].position == 5
    assert runner.requests["a"].num_computed_tokens == 5
    assert runner.input_batch.num_tokens_no_spec[0] == 7


# ---- batches and chunking ------------------------------------------------------------------------------------------------------


def test_continuous_batching_gives_every_request_its_solo_tokens(env, weights):
    runner, chip, d, alloc = env
    ps = {"a": prompt(3, 18), "b": prompt(4, 5), "c": prompt(5, 33)}
    for rid, p in ps.items():
        d.add(rid, p, [])
    run(d, {"a": 6, "b": 6, "c": 6}, alloc)
    for rid, p in ps.items():
        assert d.reqs[rid]["out"] == solo(weights, p, 6)[0], rid
    # decoding requests shared steps: some plan has two or more logit lanes
    assert max(pl.n_logits for pl in chip.plans) >= 2


def test_a_request_that_joins_later_and_one_that_finishes_early(env, weights):
    runner, chip, d, alloc = env
    pa, pb = prompt(6, 9), prompt(7, 6)
    d.add("a", pa, alloc.take(1))
    d.step({"a": 8})
    d.step({"a": 1})  # a samples its first token
    d.add("b", pb, alloc.take(1))
    d.step({"a": 1, "b": 6})
    run(d, {"a": 5, "b": 4}, alloc)
    assert d.reqs["a"]["out"][:5] == solo(weights, pa, 5)[0]
    assert d.reqs["b"]["out"][:4] == solo(weights, pb, 4)[0]


def test_a_prefix_cache_hit_starts_after_the_shared_blocks_and_equals_a_cold_run(env, weights):
    runner, chip, d, alloc = env
    shared = prompt(8, 32)
    d.add("cold", shared + [5, 6, 7], [])
    run(d, {"cold": 4}, alloc)
    first_two = list(d.reqs["cold"]["blocks"][:2])
    d.add("hit", shared + [5, 6, 7], first_two, computed=32)
    runs_before = len(chip.plans)
    d.grow("hit", alloc.take(1))
    run(d, {"hit": 4}, alloc)
    assert d.reqs["hit"]["out"] == d.reqs["cold"]["out"] == solo(weights, shared + [5, 6, 7], 4)[0]
    # the hit computed the three tokens after the shared blocks only
    first = chip.plans[runs_before]
    assert first.n_real == 3 and min(ln.position for ln in first.real_lanes) == 32


# ---- request states: finished, preempted, unscheduled, resumed -----------------------------------------------------------------


def test_finished_requests_are_dropped_from_the_state_and_the_batch(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(9, 4), alloc.take(1))
    d.add("b", prompt(10, 4), alloc.take(1))
    d.step({"a": 4, "b": 4})
    assert set(runner.requests) == {"a", "b"} and set(runner.input_batch.req_ids) == {"a", "b"}
    d.step({"b": 1}, finished=["a"])
    assert set(runner.requests) == {"b"} and runner.input_batch.req_ids == ["b"]
    assert "a" not in runner.input_batch.req_id_to_index


def test_a_finished_id_that_is_submitted_again_is_a_new_request(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(11, 5), alloc.take(1))
    d.step({"a": 5})
    d.reqs.pop("a")
    d.add("a", prompt(12, 6), alloc.take(1))
    out = d.step({"a": 6}, finished=["a"])  # the abort and the resubmission arrive in one step
    assert runner.requests["a"].prompt_token_ids == d.reqs["a"]["prompt"]
    assert runner.requests["a"].output_token_ids == out.sampled_token_ids[0]


def test_a_request_that_is_unscheduled_but_not_preempted_keeps_its_state_and_continues(env, weights):
    runner, chip, d, alloc = env
    pa, pb = prompt(13, 3), prompt(14, 5)
    d.add("a", pa, alloc.take(1))
    d.add("b", pb, alloc.take(1))
    d.step({"a": 3, "b": 5})  # both sample their first token
    d.step({"a": 1})  # b is left out of this step: running, but not scheduled
    assert "b" in runner.requests and runner.requests["b"].output_token_ids == d.reqs["b"]["out"]
    assert runner.input_batch.req_ids == ["a"], "an unscheduled request leaves the persistent batch"
    d.step({"a": 1, "b": 1})  # back, as a cached request with no new blocks and not resumed
    assert set(runner.input_batch.req_ids) == {"a", "b"}
    run(d, {"a": 4, "b": 4}, alloc)
    assert d.reqs["a"]["out"][:4] == solo(weights, pa, 4)[0]
    assert d.reqs["b"]["out"][:4] == solo(weights, pb, 4)[0]


def test_an_unscheduled_request_that_gets_a_new_block_when_it_returns(env, weights):
    runner, chip, d, alloc = env
    p = prompt(15, 15)
    d.add("a", p, alloc.take(1))
    d.add("b", prompt(16, 4), alloc.take(1))
    d.step({"a": 4, "b": 4})
    d.step({"a": 4, "b": 1})
    d.step({"a": 7, "b": 1})  # the last prompt tokens of a: it samples
    d.step({"b": 1})  # a is left out
    d.grow("a", alloc.take(1))  # and comes back with a second block, not resumed
    run(d, {"a": 4, "b": 3}, alloc)
    assert d.reqs["a"]["out"][:4] == solo(weights, p, 4)[0]


def test_a_preempted_request_is_resumed_with_new_blocks_and_recomputes_its_known_tokens(env, weights):
    runner, chip, d, alloc = env
    p = prompt(17, 10)
    d.add("a", p, alloc.take(1))
    d.step({"a": 8})
    d.step({"a": 2})
    d.step({"a": 1})
    out_before = list(d.reqs["a"]["out"])
    assert len(out_before) == 2
    old_blocks = list(d.reqs["a"]["blocks"])
    d.preempt("a")  # the scheduler frees the blocks; the request is not in the next step
    d.add("b", prompt(18, 3), alloc.take(1))
    d.step({"b": 3})
    assert "a" in runner.requests and "a" not in runner.input_batch.req_ids
    new_blocks = alloc.take(1)
    d.resume_with("a", new_blocks, computed=0)
    plans_before = len(chip.plans)
    run(d, {"a": 5, "b": 2}, alloc)
    # the recomputation started at position 0 on the new blocks only; its last known token produced the next token
    redo = [[ln for ln in pl.real_lanes if ln.req_id == "a"] for pl in chip.plans[plans_before:]]
    redo = [lanes for lanes in redo if lanes]
    assert min(ln.position for ln in redo[0]) == 0
    used = {b for lanes in redo for ln in lanes for b in ln.block_table}
    assert used.isdisjoint(old_blocks) and set(new_blocks) <= used
    assert d.reqs["a"]["out"][: len(out_before)] == out_before
    assert d.reqs["a"]["out"][:5] == solo(weights, p, 5)[0]


def test_a_resumed_request_with_prefix_hit_blocks_starts_after_them(env, weights):
    runner, chip, d, alloc = env
    p = prompt(19, 20)
    d.add("a", p, [])
    run(d, {"a": 2}, alloc)
    blocks = list(d.reqs["a"]["blocks"])
    d.preempt("a")
    # resumed with the first block still cached: 16 positions computed
    d.resume_with("a", blocks, computed=16)
    run(d, {"a": 4}, alloc)
    assert d.reqs["a"]["out"][:4] == solo(weights, p, 4)[0]


def test_resumed_ids_still_in_the_batch_are_cleared_first(env):
    # reset_prefix_cache can preempt and resume a request in one step: its id is in the batch and in resumed_req_ids
    runner, chip, d, alloc = env
    d.add("a", prompt(20, 6), alloc.take(1))
    d.step({"a": 6})
    d.preempt("a")
    d.resume_with("a", alloc.take(1), computed=0)
    out = d.step({"a": 7})
    assert runner.requests["a"].block_ids == (d.reqs["a"]["blocks"],)
    assert len(out.sampled_token_ids[0]) == 1
    assert runner.input_batch.req_ids == ["a"]


def test_block_ids_of_a_cached_request_grow_by_the_new_blocks(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(21, 16), alloc.take(1))
    d.step({"a": 8})
    d.grow("a", alloc.take(1))
    d.step({"a": 8})
    assert runner.requests["a"].block_ids == (d.reqs["a"]["blocks"],)
    assert len(d.reqs["a"]["blocks"]) == 2


def test_num_output_tokens_from_the_scheduler_truncates_discarded_outputs(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(22, 5), alloc.take(1))
    d.step({"a": 5})
    d.step({"a": 1})
    first, second = d.reqs["a"]["out"]
    sentinel = (second + 1) % CFG.vocab
    state, batch = runner.requests["a"], runner.input_batch
    state.output_token_ids[1] = sentinel  # the output the scheduler is about to discard, made distinguishable
    batch.token_ids_cpu[0, 5 + 1] = sentinel
    so = d.scheduler_output({"a": 1})
    so.scheduled_cached_reqs.num_output_tokens = [1]  # the scheduler discarded the last output
    so.scheduled_cached_reqs.num_computed_tokens = [5]
    out = runner.execute_model(so)
    # truncated to the prompt and one output, the lane at position 5 is the last known token: it has a logit row and samples
    assert chip.plans[-1].n_logits == 1 and chip.plans[-1].real_lanes[0].token == first
    assert out.sampled_token_ids == [[second]], "the discarded token was sampled again, not kept"
    assert state.output_token_ids == [first, second] and sentinel not in state.output_token_ids
    assert int(batch.num_tokens_no_spec[0]) == 5 + 2
    assert int(batch.token_ids_cpu[0, 6]) == second


def test_the_sampler_never_changes_the_logits_the_chip_returned(env):
    runner, chip, d, alloc = env
    params = SamplingParams(temperature=0.5, max_tokens=4, seed=7)
    d.add("a", prompt(27, 3), alloc.take(1), params=params)
    d.step({"a": 3})
    assert np.array_equal(chip.results[-1].logits, solo_rows_for(chip))


def solo_rows_for(chip):
    """The logits of the last chip step as ReferenceChip computed them, recomputed on a fresh chip from the same plan."""
    twin = ReferenceChip(CFG, grid_c=(64,))
    twin.load_model("synthetic:1")
    return twin.step(chip.plans[-1]).logits


# ---- block copies and zeroing ------------------------------------------------------------------------------------------------


def test_block_copies_and_zeroing_happen_before_the_run_and_are_counted(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(23, 16), alloc.take(1))
    d.step({"a": 8})
    d.step({"a": 8})
    b = d.reqs["a"]["blocks"][0]
    k, v = chip.pool.k[b].copy(), chip.pool.v[b].copy()
    assert k.any()
    d.add("c", prompt(24, 3), alloc.take(1))
    dst = alloc.take(1)[0]
    d.step({"c": 3}, zero=[dst], copies=[(b, dst)])
    assert np.array_equal(chip.pool.k[dst], k) and np.array_equal(chip.pool.v[dst], v)
    st = runner.stats.snapshot()
    assert (st["blocks_zeroed"], st["blocks_copied"]) == (1, 1)


def test_zeroing_comes_before_copying(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(25, 8), alloc.take(1))
    d.step({"a": 8})
    b = d.reqs["a"]["blocks"][0]
    dst = alloc.take(1)[0]
    d.add("c", prompt(26, 2), alloc.take(1))
    d.step({"c": 2}, zero=[dst], copies=[(b, dst)])
    assert chip.pool.k[dst].any(), "zero then copy leaves the copy, not zeros"


def test_a_step_with_nothing_scheduled_still_applies_finishes_and_block_work(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(27, 8), alloc.take(1))
    d.step({"a": 8})
    b = d.reqs["a"]["blocks"][0]
    runs = len(chip.plans)
    out = d.step({}, finished=["a"], copies=[(b, 9)])
    assert out.req_ids == [] and out.sampled_token_ids == [] and len(chip.plans) == runs
    assert chip.pool.k[9].any() and not runner.requests


# ---- sampling through vLLM's Sampler on ReferenceChip logits --------------------------------------------------------------------------


def test_greedy_picks_the_argmax_of_the_chip_logits(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(28, 6), alloc.take(1))
    out = d.step({"a": 6})
    assert out.sampled_token_ids[0] == [int(np.argmax(chip.results[-1].logits[0]))]


def test_logprobs_equal_log_softmax_of_the_chip_logits_through_the_same_torch(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(29, 6), alloc.take(1), params=SamplingParams(temperature=0.0, logprobs=3, max_tokens=8))
    out = d.step({"a": 6})
    lp, tok = out.logprobs, out.sampled_token_ids[0][0]
    ref = torch.log_softmax(torch.from_numpy(chip.results[-1].logits), dim=-1)
    top = torch.topk(ref[0], 3)
    # row: the sampled token, then the top 3
    assert lp.logprob_token_ids[0].tolist() == [tok] + top.indices.tolist()
    assert np.array_equal(lp.logprobs[0], np.concatenate([[ref[0, tok].item()], top.values.numpy()]))
    assert lp.sampled_token_ranks[0] == 1


def fresh(vllm_config):
    chip = ReferenceChip(CFG, grid_c=(64,))
    chip.load_model("synthetic:1")
    return Driver(TspModelRunner(vllm_config, chip)), Alloc(), chip


def test_a_seeded_request_samples_the_same_tokens_alone_and_batched(vllm_config):
    pa = prompt(30, 9)
    params = SamplingParams(temperature=0.9, top_k=40, seed=1234, max_tokens=8)
    d, alloc, _ = fresh(vllm_config)
    d.add("x", pa, [], params=params)
    run(d, {"x": 6}, alloc)
    # x with a greedy neighbour, a seeded neighbour and a neighbour that is long in prefill while x samples
    d2, alloc2, chip2 = fresh(vllm_config)
    d2.add("x", pa, [], params=params)
    d2.add("y", prompt(31, 5), [])
    d2.add("z", prompt(32, 40), [], params=SamplingParams(temperature=0.7, seed=5, max_tokens=8))
    run(d2, {"x": 6, "y": 6, "z": 3}, alloc2)
    assert d2.reqs["x"]["out"] == d.reqs["x"]["out"]
    assert max(pl.n_logits for pl in chip2.plans) >= 2


def test_the_sampler_uses_exactly_the_requests_seed(vllm_config):
    # the tokens are those of torch's CPU generator seeded with 1234 over the scripted chip's logits (torch 2.x on CPU); another seed gives other tokens
    def sampled(seed):
        d, alloc, _ = fresh(vllm_config)
        d.add("x", prompt(30, 9), [], params=SamplingParams(temperature=0.9, top_k=40, seed=seed, max_tokens=8))
        run(d, {"x": 6}, alloc)
        return d.reqs["x"]["out"]

    assert sampled(1234) == [485, 91, 1313, 987, 987, 987]
    assert sampled(1235) == [485, 246, 246, 246, 246, 246]


def test_a_seeded_request_in_mid_prefill_does_not_advance_its_generator(vllm_config):
    pz = prompt(33, 12)
    params = SamplingParams(temperature=1.0, seed=99, max_tokens=8)
    alone, alloc1, _ = fresh(vllm_config)
    alone.add("z", pz, [], params=params)
    run(alone, {"z": 5}, alloc1)
    d, alloc, chip = fresh(vllm_config)
    d.add("x", prompt(34, 3), alloc.take(1))
    d.add("z", pz, alloc.take(1), params=params)
    d.step({"x": 3, "z": 4})  # x samples while z's row is sampled and discarded
    d.step({"x": 1, "z": 4})
    d.step({"x": 1, "z": 4})  # z reaches its last prompt token
    assert [pl.n_logits for pl in chip.plans] == [1, 1, 2]
    run(d, {"x": 4, "z": 5}, alloc)
    assert d.reqs["z"]["out"][:5] == alone.reqs["z"]["out"]


def test_two_identical_runs_give_identical_ids_logits_and_plans(vllm_config):
    def once():
        d, alloc, chip = fresh(vllm_config)
        params = SamplingParams(temperature=0.8, top_p=0.9, seed=7, max_tokens=8)
        d.add("a", prompt(34, 11), [], params=params)
        d.add("b", prompt(35, 4), [])
        run(d, {"a": 5, "b": 5}, alloc)
        return d, chip

    (d1, c1), (d2, c2) = once(), once()
    assert d1.reqs["a"]["out"] == d2.reqs["a"]["out"] and d1.reqs["b"]["out"] == d2.reqs["b"]["out"]
    assert len(c1.results) == len(c2.results)
    assert all(np.array_equal(a.logits, b.logits) for a, b in zip(c1.results, c2.results))
    assert [pl.bucket for pl in c1.plans] == [pl.bucket for pl in c2.plans]


# ---- the chip and the statistics ------------------------------------------------------------------------------------------------


def test_per_step_statistics_record_the_bucket_lanes_and_padding(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(36, 5), alloc.take(1))
    d.add("b", prompt(37, 3), alloc.take(1))
    d.add("c", prompt(38, 2), alloc.take(1))
    d.step({"a": 5, "b": 3})
    st = runner.stats.last
    assert st["bucket"] == (8, 2, 64) and (st["real_lanes"], st["padding_lanes"]) == (8, 0)
    d.step({"a": 1, "b": 1, "c": 2})
    st = runner.stats.last
    assert st["bucket"] == (4, 4, 64) and (st["real_lanes"], st["padding_lanes"]) == (4, 0)
    agg = runner.stats.snapshot()
    assert agg["steps"] == 2 and agg["real_lanes"] == 12 and agg["padding_lanes"] == 0
    assert agg["buckets"] == {"8x2x64": 1, "4x4x64": 1}
    for key in ("host_prep_s", "chip_s", "sampler_s", "patch_count", "context_padding", "cycles"):
        assert key in agg


def test_a_chip_error_is_recorded_as_the_failure_and_propagates(env):
    runner, chip, d, alloc = env
    d.add("a", prompt(39, 5), alloc.take(1))
    d.step({"a": 5})
    d.add("b", prompt(40, 5), [alloc.take(1)[0]])

    def boom(plan):
        raise TspChipError("diagnostic OTS-E-TEST at cycle 5")

    chip.step = boom
    with pytest.raises(TspChipError, match="OTS-E-TEST"):
        d.step({"b": 5})
    assert "OTS-E-TEST" in runner.failure


def test_a_step_the_programs_cannot_hold_fails_before_the_chip_is_touched(tmp_path):
    cfg = H.engine_config(H.write_model_dir(tmp_path), max_model_len=128)
    d, alloc, chip = fresh(cfg)  # the chip has programs for C = 64 only
    d.add("a", prompt(41, 70), [])
    d.grow("a", alloc.take(5))
    d.reqs["a"]["computed"] = 64  # a prefix hit up to position 64
    with pytest.raises(ValueError, match="context"):
        d.step({"a": 6})
    assert chip.plans == []


def test_the_runner_refuses_a_chip_whose_vocabulary_differs_from_the_model_config(vllm_config):
    chip = ReferenceChip(config.ModelConfig(layers=2, vocab=97, max_pos=128), grid_c=(64,))
    chip.load_model("synthetic:1")
    with pytest.raises(ValueError, match="vocab"):
        TspModelRunner(vllm_config, chip)
