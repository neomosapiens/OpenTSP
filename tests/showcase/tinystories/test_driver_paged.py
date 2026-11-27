"""Test the paged TinyStories host driver without vLLM.

A toy scheduler (block pool with refcounts, prefix caching, chunked prefill, continuous batching, preemption by recompute,
greedy sampling) builds the lanes of every step; three KV modes run them: `exact` (the oracle, no ISS), `iss` (the bucket
programs of the cache on `tsp-iss`, the K/V pool carried in SRAM across steps) and `staged` (the gate driver's one-lane step with the
K/V history written by the host). The reference of every token is `paged_reference.generate_paged` of the request alone.

The tests that run the ISS carry the marker `paged_iss` (and `slow`: `just showcase-reference` leaves them out, `just showcase-paged`
runs them); the full-model scenario is `paged_full` (`just showcase-paged-full`)."""
import os
import random
import sys

import numpy as np
import pytest

from opentsp_vllm.models.tinystories import paths as _paths

sys.path.insert(0, os.path.join(_paths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.tinystories import driver_paged as DP
from opentsp_vllm.models.tinystories import reference as ReferenceM
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import weights as W

LAYERS = 2
VOCAB = 50257
SHARED = [31337, 4242, 777, 50000, 13, 9, 1234, 5, 6, 7, 8, 1010, 2020, 3030, 4040, 5050]      # one full block
SUFFIX = [[11, 12, 13], [21, 22, 23, 24], [31, 32], [41, 42, 43, 44, 45]]


@pytest.fixture(scope="module")
def w():
    return W.synthetic(L.serve_config(LAYERS), 1)


@pytest.fixture(scope="module")
def alone(w):
    """`generate_paged` of one request alone, memoised: the reference of every scenario."""
    memo = {}

    def run(prompt, n_new):
        key = (tuple(prompt), n_new)
        if key not in memo:
            memo[key] = G.generate_paged(w, prompt, n_new, C=64 if len(prompt) + n_new <= 64 else 128)
        return memo[key]
    return run


def shared_prompts(n=4):
    return [SHARED + SUFFIX[i] for i in range(n)]


def drive(w, subs, mode="exact", config=None, **kw):
    """Run `subs` (step, prompt, max_new) to completion: ({req_id: new ids}, the driver)."""
    drv = DP.Driver(w, config or DP.SchedulerConfig(), mode, **kw)
    out = drv.run([DP.Submission(s, f"r{i}", p, n) for i, (s, p, n) in enumerate(subs)])
    return out, drv


# -- the scheduler alone ------------------------------------------------------------------------------------------------------
def test_one_request_runs_alone_like_generate_paged(w, alone):
    prompt = shared_prompts()[0]
    out, drv = drive(w, [(0, prompt, 5)])
    assert out["r0"] == alone(prompt, 5).tokens[len(prompt):]
    assert len(out["r0"]) == 5


def test_no_block_leaks_after_all_requests_finish(w):
    out, drv = drive(w, [(0, p, 4) for p in shared_prompts()])
    m = drv.scheduler.blocks
    assert m.refs == {} and m.num_free == drv.scheduler.config.num_blocks - 1
    m.check()


def test_a_prefix_hit_shares_the_block_without_a_copy(w):
    """B joins after A has computed the full block of the shared prefix: B's first block is A's block (refcount 2), B's first lane
    is position 16, and no K or V row is written into the shared block after B joined."""
    cfg = DP.SchedulerConfig(token_budget=8)
    sched = DP.Scheduler(cfg)
    a, b = shared_prompts(2)
    sched.submit("A", a, 6)
    writes = []
    chip = DP.ReferenceChip(w, cfg.num_blocks)
    real = chip.pool.write_row
    chip.pool.write_row = lambda blk, *x: (writes.append(blk), real(blk, *x))
    joined = None
    for i in range(20):
        if i == 3:
            sched.submit("B", b, 6)
            joined = len(writes)
        step = sched.step()
        if step is None:
            break
        sched.finish_step(DP.sample(step, chip.run(step)))
        if i == 3:
            lanes_b = [ln for ln in step.lanes if ln.req_id == "B"]
            assert min(ln.position for ln in lanes_b) == G.BLOCK, "B starts after the cached block"
            ta = next(ln for ln in step.lanes if ln.req_id == "A").block_table
            tb = lanes_b[0].block_table
            assert ta[0] == tb[0] and sched.blocks.pool.refs[ta[0]] == 2
            shared = ta[0]
    assert shared not in writes[joined:], "the shared block is never written again"
    assert sched.requests["B"].cached_tokens == G.BLOCK


def test_chunked_prefill_respects_the_token_budget(w, alone):
    prompt = shared_prompts()[1]                        # 20 tokens
    cfg = DP.SchedulerConfig(token_budget=3)
    out, drv = drive(w, [(0, prompt, 4)], config=cfg)
    sizes = [len(s.lanes) for s in drv.steps]
    assert max(sizes) == 3 and sizes[:6] == [3] * 6 and sum(sizes) == len(prompt) + 4 - 1
    assert [s.n_logits for s in drv.steps if s.n_logits] and all(s.n_logits == 0 for s in drv.steps[:6])
    for s in drv.steps:                                # consecutive positions, one logit lane only at the last known token
        pos = sorted(ln.position for ln in s.lanes)
        assert pos == list(range(pos[0], pos[0] + len(pos)))
    assert out["r0"] == alone(prompt, 4).tokens[len(prompt):]


def test_requests_join_and_finish_between_steps(w, alone):
    subs = [(0, shared_prompts()[0], 6), (2, shared_prompts()[1], 3), (5, shared_prompts()[2], 4)]
    out, drv = drive(w, subs)
    for i, (_, p, n) in enumerate(subs):
        assert out[f"r{i}"] == alone(p, n).tokens[len(p):], i
    sizes = [len({ln.req_id for ln in s.lanes}) for s in drv.steps]
    assert max(sizes) >= 2 and all(len(s.lanes) <= 8 for s in drv.steps)
    assert all(s.bucket == G.choose_bucket(len(s.lanes), s.n_logits, max(ln.position for ln in s.lanes)) for s in drv.steps)


def test_logit_lanes_come_first_and_only_the_last_known_token_has_logits(w):
    out, drv = drive(w, [(0, p, 3) for p in shared_prompts()])
    for s in drv.steps:
        flags = [ln.logits for ln in s.lanes]
        assert flags == sorted(flags, reverse=True) and sum(flags) == s.n_logits <= 4
        assert len({ln.req_id for ln in s.lanes if ln.logits}) == s.n_logits


def test_preemption_by_recompute_gives_the_tokens_of_the_run_without_it(w, alone):
    """The design's 6-block scenario: 3 requests of 12 prompt tokens and 12 new tokens, 5 usable blocks, each needs a second block
    at position 16: the youngest is preempted, re-enters the queue and re-prefills (prompt and its sampled tokens)."""
    prompts = [[100 + 10 * i + j for j in range(12)] for i in range(3)]
    subs = [(0, p, 12) for p in prompts]
    small, drv = drive(w, subs, config=DP.SchedulerConfig(num_blocks=6, max_model_len=64))
    roomy, _ = drive(w, subs)
    assert drv.scheduler.preemptions >= 1 and sum(r.preemptions for r in drv.scheduler.requests.values()) >= 1
    assert small == roomy
    for i, p in enumerate(prompts):
        assert small[f"r{i}"] == alone(p, 12).tokens[len(p):], i
    assert drv.scheduler.blocks.refs == {}


def test_a_preempted_request_hits_the_prefix_cache_where_its_blocks_survive(w):
    prompts = [[100 + 10 * i + j for j in range(12)] for i in range(3)]
    _, drv = drive(w, [(0, p, 12) for p in prompts], config=DP.SchedulerConfig(num_blocks=6, max_model_len=64))
    victims = [r for r in drv.scheduler.requests.values() if r.preemptions]
    assert victims and any(r.cached_tokens > 0 for r in victims)


def test_eos_stops_a_request(w, alone):
    prompt = shared_prompts()[0]
    ref = alone(prompt, 6).tokens[len(prompt):]
    out, _ = drive(w, [(0, prompt, 6)], eos=ref[2])
    assert out["r0"] == ref[:ref.index(ref[2]) + 1]


def test_a_request_longer_than_the_model_length_is_refused():
    sched = DP.Scheduler(DP.SchedulerConfig(max_model_len=64))
    with pytest.raises(DP.SchedulerError, match="max_model_len"):
        sched.submit("x", list(range(60)), 8)
    with pytest.raises(DP.SchedulerError, match="at least"):
        DP.Scheduler(DP.SchedulerConfig(num_blocks=3, max_model_len=64))


def test_the_pool_accounting_catches_a_double_free():
    m = DP.BlockManager(DP.SchedulerConfig(num_blocks=8))
    b = m.allocate()
    m.free([b])
    with pytest.raises(ValueError):
        m.free([b])


# -- four requests with a shared prefix against generate_paged, on bit-exact reference -------------------------------------------------------------
SCENARIOS = [  # (seed, joins, token budget, max_seqs)
    (1, [0, 0, 0, 0], 8, 4), (2, [0, 1, 3, 4], 8, 4), (3, [0, 2, 2, 7], 5, 4), (4, [0, 3, 6, 9], 3, 2), (5, [4, 0, 1, 2], 6, 3),
]


@pytest.mark.parametrize("seed, joins, budget, seqs", SCENARIOS, ids=lambda x: str(x) if not isinstance(x, list) else "".join(map(str, x)))
def test_four_requests_with_a_shared_prefix_equal_generate_paged_on_reference(w, alone, seed, joins, budget, seqs):
    rng = random.Random(seed)
    prompts = shared_prompts()
    news = [rng.randint(3, 6) for _ in prompts]
    cfg = DP.SchedulerConfig(token_budget=budget, max_seqs=seqs)
    out, drv = drive(w, [(j, p, n) for j, p, n in zip(joins, prompts, news)], config=cfg)
    for i, (p, n) in enumerate(zip(prompts, news)):
        assert out[f"r{i}"] == alone(p, n).tokens[len(p):], f"seed {seed} request {i}"
    drv.scheduler.blocks.check()
    assert drv.scheduler.blocks.refs == {}


def test_requests_submitted_together_hit_the_block_the_first_has_computed(w):
    """The token budget serialises the prefill: A computes the shared block over two steps, the others are admitted in the pass that
    follows and take it from the cache (no copy), a staggered start the same."""
    for joins in ([0, 0, 0, 0], [0, 3, 6, 9]):
        _, drv = drive(w, [(j, p, 3) for j, p in zip(joins, shared_prompts())])
        assert [drv.scheduler.requests[f"r{i}"].cached_tokens for i in range(4)] == [0, G.BLOCK, G.BLOCK, G.BLOCK], joins


# -- the scheduler's rules, pinned one by one (no chip: the sampled token is always 7) -------------------------------------------------
def plan_only(subs, **cfg):
    """The steps of `subs` (step, prompt, max_new) on the null chip: (the driver, {req_id: all its lanes in order})."""
    drv = DP.Driver(None, DP.SchedulerConfig(**cfg), "none", rng=random.Random(0))
    drv.run([DP.Submission(s, f"r{i}", tuple(p), n) for i, (s, p, n) in enumerate(subs)])
    lanes = {}
    for st in drv.steps:
        for ln in st.lanes:
            lanes.setdefault(ln.req_id, []).append(ln)
    return drv, lanes


def test_the_block_hash_covers_every_token_of_the_block():
    """A prompt whose first 15 tokens equal a cached block's and whose 16th differs gets no hit (a hash that ignores the last
    token of the block would serve it the wrong K/V words); the same block does hit."""
    other = SHARED[:15] + [SHARED[15] + 1]
    drv, _ = plan_only([(0, SHARED + [11], 6), (6, other + [11, 12], 2), (7, SHARED + [11, 12], 2)])
    assert [drv.scheduler.requests[f"r{i}"].cached_tokens for i in range(3)] == [0, 0, G.BLOCK]
    assert DP.block_hash(None, SHARED) != DP.block_hash(None, other) and DP.block_hash(None, SHARED) == DP.block_hash(None, list(SHARED))


def test_a_prompt_that_is_all_cached_blocks_recomputes_its_last_token():
    """vLLM never serves a whole prompt from the cache: the hit is capped at len(tokens) - 1, so the last prompt token is
    recomputed (it is the lane that has the logits). A 16-token prompt resubmitted after its twin finished hits nothing and runs
    from position 0; a 32-token one hits one block and runs from position 16."""
    one, two = list(SHARED), SHARED + [(t + 1) % VOCAB for t in SHARED]
    drv, lanes = plan_only([(0, one, 2), (0, two, 2), (20, one, 2), (20, two, 2)])
    reqs = drv.scheduler.requests
    assert reqs["r0"].status == reqs["r1"].status == "finished"
    assert reqs["r2"].cached_tokens == 0 and reqs["r3"].cached_tokens == G.BLOCK
    assert lanes["r2"][0].position == 0 and [ln.position for ln in lanes["r2"] if ln.logits][0] == G.BLOCK - 1
    assert lanes["r3"][0].position == G.BLOCK and [ln.position for ln in lanes["r3"] if ln.logits][0] == 2 * G.BLOCK - 1


def test_a_finished_request_frees_its_blocks_last_block_first():
    """The free queue is FIFO and the blocks go back in reverse order, so the tail of a request is evicted before its head (the
    head is the part another request can still hit)."""
    sched = DP.Scheduler(DP.SchedulerConfig(num_blocks=8, max_model_len=64))
    req = sched.submit("a", list(range(100, 140)), 2)
    held = []
    while sched.has_work():
        step = sched.step()
        held = list(req.blocks) or held
        sched.finish_step({r: 7 for r in step.logit_req_ids})
    assert len(held) == 3 and list(sched.blocks.pool.free_queue)[-3:] == held[::-1]


def test_a_preempted_request_restarts_from_position_zero_except_for_the_blocks_it_hits():
    """The youngest request is preempted by recompute: right after the preempting step its `computed` is 0 and it holds no block;
    its next lanes start at the end of the prefix hit it is readmitted with (here its first block survived: 16) and cover every
    position of its tokens, prompt and sampled, from there."""
    sched = DP.Scheduler(DP.SchedulerConfig(num_blocks=6, max_model_len=64))
    for i in range(3):
        sched.submit(f"r{i}", [100 + 10 * i + j for j in range(12)], 12)
    victim, seen = None, []
    while sched.has_work():
        step = sched.step()
        if victim is None and step.preempted:
            victim = sched.requests[step.preempted[0]]
            assert victim.computed == 0 and victim.blocks == [] and victim.hashes == [] and victim.status == "waiting"
            assert victim.preemptions == 1 and sched.waiting[0] is victim
        elif victim is not None:
            seen += [ln.position for ln in step.lanes if ln.req_id == victim.req_id]
        sched.finish_step({r: 7 for r in step.logit_req_ids})
    assert victim is not None and victim.cached_tokens == G.BLOCK
    assert seen == list(range(victim.cached_tokens, len(victim.tokens) - 1))


# -- the random schedule of  -------------------------------------------------------------------------------------------------
STEPS = (6, 8, 12, 20, 30, 60, 100)


def check_schedule(sch, steps):
    assert len(sch.steps) == steps
    assert len(sch.submissions) >= 3 and any(s.step > 0 for s in sch.submissions)
    assert sch.prefix_hit_tokens >= G.BLOCK
    assert sch.config.num_blocks < G.POOL_BLOCKS
    if steps >= 8:
        assert sch.finished_before_end >= 1
    if steps >= 12:
        assert sch.preemptions >= 1
    for s in sch.steps:
        assert 1 <= len(s.lanes) <= 8 and s.n_logits <= 4
        assert all(0 < b < sch.config.num_blocks for ln in s.lanes for b in ln.block_table)
        assert s.bucket == G.choose_bucket(len(s.lanes), s.n_logits, max(ln.position for ln in s.lanes))
    hooks = [s for s in sch.steps if s.block_copies or s.zero_blocks]
    if steps >= 6:
        assert hooks and all(set(c[1] for c in s.block_copies).isdisjoint(b for ln in s.lanes for b in ln.block_table) for s in hooks)


@pytest.mark.parametrize("steps", STEPS)
def test_random_schedule_has_the_asked_number_of_steps_and_its_coverage_for_20_seeds(steps):
    for seed in range(20):
        sch = DP.random_schedule(seed, steps)
        check_schedule(sch, steps)
        if seed < 4:
            assert sch == DP.random_schedule(seed, steps), f"seed {seed}: not deterministic"


def test_random_schedule_steps_default_to_12_and_are_bounded():
    assert DP.random_schedule(2) == DP.random_schedule(2, 12)
    for steps in (3, 201):
        with pytest.raises(ValueError, match="4 to 200"):
            DP.random_schedule(0, steps)
    for kw in ({"max_model_len": 96}, {"max_model_len": 128, "steps": 20}):
        with pytest.raises(ValueError, match="max_model_len"):
            DP.random_schedule(0, **kw)


def test_the_shortest_and_the_longest_schedule_exist_for_every_step_count_in_between_for_a_seed():
    for steps in (4, 5, 7, 9, 11, 13, 17, 25, 40, 75, 150, 200):
        check_schedule(DP.random_schedule(1, steps), steps)


@pytest.mark.parametrize("seed", range(5))
def test_a_random_schedule_leaves_the_global_random_state_alone(seed):
    random.seed(99)
    before = random.getstate()
    DP.random_schedule(seed, 12)
    assert random.getstate() == before


@pytest.mark.parametrize("seed", range(5))
def test_random_schedule_has_joins_finishes_prefix_hits_and_a_preemption(seed):
    sch = DP.random_schedule(seed, 12)
    assert len(sch.steps) == 12 and sch == DP.random_schedule(seed, 12)
    assert sch.preemptions >= 1 and sch.prefix_hit_tokens >= G.BLOCK
    assert len(sch.submissions) >= 3 and any(s.step > 0 for s in sch.submissions) and sch.finished_before_end >= 1
    check_schedule(sch, 12)


@pytest.mark.parametrize("steps", (40, 60))
def test_a_schedule_with_max_model_len_128_reaches_the_c128_buckets(steps):
    for seed in range(5):
        sch = DP.random_schedule(seed, steps, max_model_len=128)
        assert sch.config.max_model_len == 128 and sch.config.num_blocks in (9, 10)
        assert any(s.bucket[2] == 128 for s in sch.steps) and any(s.bucket[2] == 64 for s in sch.steps)
        assert sch.preemptions >= 1 and len(sch.steps) == steps
        assert sch == DP.random_schedule(seed, steps, max_model_len=128)
        assert max(ln.position for s in sch.steps for ln in s.lanes) >= 64


def test_a_copied_block_is_a_cache_replica_that_a_later_request_hits():
    """`register_copy`: the cache maps the hash to the copy; a request admitted afterwards takes the copy (not the source)."""
    sched = DP.Scheduler(DP.SchedulerConfig(num_blocks=8, max_model_len=64))
    sched.submit("a", SHARED + [11], 3)
    while sched.blocks.cache == {}:
        sched.finish_step({r: 7 for r in sched.step().logit_req_ids})
    src = next(iter(sched.blocks.hash_of))
    dst = sched.blocks.pool.free_queue[-1]                      # the last block to be handed out
    sched.register_copy(src, dst)
    assert sched.blocks.hash_of == {dst: DP.block_hash(None, SHARED)} and sched.blocks.replicas == {dst}
    sched.submit("b", SHARED + [21, 22], 2)
    step = sched.step()
    assert sched.requests["b"].blocks[0] == dst and sched.replica_hits == 1 and ("b", G.BLOCK) in step.hits
    sched.check()
    with pytest.raises(DP.SchedulerError, match="live or cached"):
        sched.register_copy(src, dst)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_copy_hook_of_a_random_schedule_changes_a_later_logit(w, seed, monkeypatch):
    """The copied block is read by a later lane (a cache replica): without the copy some logit differs, so the hook is checked by
    its effect, not only by its being applied."""
    sch = DP.random_schedule(seed, 20)
    assert sch.hook and any(s.block_copies for s in sch.steps)

    def run():
        drv = DP.Driver(w, sch.config, "exact", hook=sch.hook())
        drv.run(sch.submissions, max_steps=20)
        return [r.logits.tobytes() for r in drv.records]
    with_copy = run()
    monkeypatch.setattr(DP.ReferenceChip, "copy_block", lambda self, src, dst: None)
    assert run() != with_copy


@pytest.mark.parametrize("seed", range(3))
def test_the_steps_are_the_step_plans_of_opentsp_vllm(seed):
    """`Step.requests` are `PlanRequest` fields: 's `plan_step` (read-only import) gives the lanes, the order and the bucket
    of the toy scheduler for every step of a random schedule."""
    from opentsp_vllm import step_plan as SP
    for step in DP.random_schedule(seed, 12).steps:
        plan = SP.plan_step([SP.PlanRequest(*r) for r in step.requests], vocab=VOCAB)
        assert plan.bucket == step.bucket and plan.n_logits == step.n_logits
        assert [(ln.req_id, ln.token, ln.position, ln.block_table, ln.logits) for ln in plan.real_lanes] == \
               [(ln.req_id, ln.token, ln.position, ln.block_table, ln.logits) for ln in step.lanes]


def test_random_schedules_differ_between_seeds():
    shapes = {tuple(len(s.lanes) for s in DP.random_schedule(seed, 12).steps) for seed in range(5)}
    assert len(shapes) >= 3


def shape(step):
    return (step.bucket, step.n_logits, tuple((ln.req_id, ln.position, ln.block_table, ln.logits) for ln in step.lanes),
            step.block_copies, step.zero_blocks)


@pytest.mark.parametrize("seed", [0, 3])
def test_a_random_schedule_replays_on_reference_with_the_same_lanes(w, seed):
    """`Driver(w, s.config, mode).run(s.submissions)` runs the lanes of `s.steps` whatever the chip samples."""
    sch = DP.random_schedule(seed, 12)
    drv = DP.Driver(w, sch.config, "exact", hook=sch.hook())
    drv.run(sch.submissions, max_steps=12)
    assert [shape(s) for s in drv.steps] == [shape(s) for s in sch.steps]



# -- the ISS: the cached bucket programs, the pool carried in SRAM across steps -----------------------------------------------------
def paged_iss(f):
    return pytest.mark.paged_iss(pytest.mark.slow(f))


@pytest.fixture(scope="module")
def model():
    from opentsp_vllm.models.tinystories import program_cache as PC
    return PC.Model(LAYERS)


def logits_equal(a, b, what):
    """Two records of the same schedule: every logit row of every step bit-identical."""
    assert len(a.records) == len(b.records), what
    for i, (x, y) in enumerate(zip(a.records, b.records)):
        assert shape(x.step) == shape(y.step), f"{what}: step {i} lanes differ"
        assert x.logits.shape == y.logits.shape and x.logits.tobytes() == y.logits.tobytes(), f"{what}: step {i} logits differ"


def run_pair(w, model, subs, config=None):
    g, gd = drive(w, subs, "exact", config)
    i, idrv = drive(w, subs, "iss", config, model=model)
    return g, gd, i, idrv


@paged_iss
def test_four_requests_with_a_shared_prefix_equal_generate_paged_on_the_iss_and_its_logits_equal_the_oracle(w, alone, model):
    """The 4-request scenario of the , joins staggered, on the 2-layer cached programs: tokens equal `generate_paged` of
    each request alone, and every logit row of the run equals the bit-exact reference-mode run of the same schedule bit for bit."""
    prompts = shared_prompts()
    news = [4, 3, 4, 3]
    subs = [(j, p, n) for j, p, n in zip([0, 1, 4, 5], prompts, news)]
    g, gd, i, idrv = run_pair(w, model, subs)
    for k, (p, n) in enumerate(zip(prompts, news)):
        assert i[f"r{k}"] == alone(p, n).tokens[len(p):] == g[f"r{k}"], k
    logits_equal(gd, idrv, "iss against exact")
    assert sum(r.cached_tokens for r in idrv.scheduler.requests.values()) >= G.BLOCK, "a prefix hit on the chip's pool"
    idrv.scheduler.check()


@paged_iss
def test_a_preemption_on_the_iss_gives_the_tokens_of_the_run_without_it(w, alone, model):
    prompts = [[100 + 10 * i + j for j in range(12)] for i in range(3)]
    subs = [(0, p, 9) for p in prompts]
    small = DP.SchedulerConfig(num_blocks=6, max_model_len=64)
    got, drv = drive(w, subs, "iss", small, model=model)
    roomy, rdrv = drive(w, subs, "exact")
    assert drv.scheduler.preemptions >= 1 and rdrv.scheduler.preemptions == 0
    assert got == roomy
    for k, p in enumerate(prompts):
        assert got[f"r{k}"] == alone(p, 9).tokens[len(p):], k
    assert drv.scheduler.blocks.refs == {}


@paged_iss
@pytest.mark.parametrize("seed", [0, 1])
def test_a_random_schedule_with_a_preemption_and_the_hooks_is_bit_exact_on_the_iss(w, model, seed):
    """Joins, finishes, prefix hits, a preemption, a block copy and a zeroing between runs: the chip's logits equal bit-exact reference's on
    every step (the driver also verifies every written K and V word against the oracle)."""
    sch = DP.random_schedule(seed, 12)
    g = DP.Driver(w, sch.config, "exact", hook=sch.hook())
    g.run(sch.submissions, max_steps=12)
    c = DP.Driver(w, sch.config, "iss", model=model, hook=sch.hook())
    c.run(sch.submissions, max_steps=12)
    logits_equal(g, c, f"seed {seed}")
    assert sch.preemptions >= 1 and any(r.cycles for r in c.records)


@paged_iss
def test_the_driver_notices_a_chip_that_differs_from_reference(w, model, monkeypatch):
    """The comparison does compare: one flipped bit in a logit the chip returns is reported with the step."""
    drv = DP.Driver(w, DP.SchedulerConfig(), "iss", model=model)
    real = drv.chip.PC.decode_logits

    def flipped(*a, **k):
        out = real(*a, **k)
        out.view(np.uint32)[0, 7] ^= 1
        return out
    monkeypatch.setattr(drv.chip.PC, "decode_logits", flipped)
    with pytest.raises(DP.DriverError, match="step 0.*logit row 0 differs from bit-exact reference"):
        drv.run([DP.Submission(0, "r0", (5, 6), 2)])


@paged_iss
def test_requests_admitted_in_the_pass_that_fills_the_shared_block_hit_it_on_the_iss(w, alone, model, monkeypatch):
    """A and B at step 0, budget 7: A's first step fills 7 of the 16 shared positions, its second fills the block, and B, admitted in
    that pass, takes the block from the cache: tokens equal `generate_paged` and the bit-exact reference run, every logit and K/V word equal bit-exact reference's.
    The steps' cycles are the programs' schedule lengths, and no `tsp-iss` call may block for more than 30 minutes (AGENTS.md)."""
    timeouts = []
    real = DP.subprocess.run
    monkeypatch.setattr(DP.subprocess, "run", lambda *a, **k: (timeouts.append(k.get("timeout")), real(*a, **k))[1])
    prompts, news = [SHARED + [11], SHARED + [21, 22, 23, 24]], [3, 3]
    subs = [(0, p, n) for p, n in zip(prompts, news)]
    g, gd, i, idrv = run_pair(w, model, subs, DP.SchedulerConfig(token_budget=7))
    reqs = idrv.scheduler.requests
    assert [reqs["r0"].cached_tokens, reqs["r1"].cached_tokens] == [0, G.BLOCK]
    assert any(("r1", G.BLOCK) in st.hits and st.index > 0 for st in idrv.steps)
    for k, (p, n) in enumerate(zip(prompts, news)):
        assert i[f"r{k}"] == alone(p, n).tokens[len(p):] == g[f"r{k}"], k
    logits_equal(gd, idrv, "iss against exact")
    assert timeouts and all(t is not None and t <= 1800 for t in timeouts)
    for r in idrv.records:
        assert r.cycles == idrv.chip.arts[r.step.bucket].record["schedule_cycles"] > 0


@paged_iss
def test_a_word_missing_from_the_carry_is_a_driver_error_naming_the_step(w, model):
    """`verify` finds the K/V words of the step in the carried pool: with an empty one it says which step and word, not a KeyError."""
    sched = DP.Scheduler(DP.SchedulerConfig())
    sched.submit("a", [5, 6], 2)
    step = sched.step()
    logits = DP.ReferenceChip(model.w).run(step)
    chip = DP.IssChip(model)
    with pytest.raises(DP.DriverError, match=r"step 0 P\(.*\): the K/V word \(.*\) of layer 0, block 1, position 1 was not carried"):
        chip._verify(step, logits)


# -- staged against paged (the 8-layer step, one lane per run, K/V written by the host) ---------------------------------------------
def test_the_ts_layout_and_the_paged_layout_give_the_same_tokens_on_the_8_layer_model_for_several_prompts():
    """The caveat behind the staged comparison below (an f16 rounding flip at a few percent of the steps): on bit-exact reference alone,
    `reference_model.generate` (the TS layout) equals `generate_paged` over 12 seeded prompts of the 8-layer synthetic model, so the one
    prompt the staged chip is compared on is not a freak. (The chip itself is compared on tokens only, one prompt.)"""
    w8 = W.synthetic(L.serve_config(8), 1)
    rng = random.Random(5)
    for _ in range(12):
        prompt = [rng.randrange(VOCAB) for _ in range(rng.randint(2, 5))]
        assert ReferenceM.generate(w8, prompt, 3).tokens == G.generate_paged(w8, prompt, 3).tokens, prompt


@paged_iss
def test_staged_and_paged_give_the_same_tokens_on_one_short_request(tmp_path):
    """A 3-token prompt and 3 new tokens on the 8-layer synthetic model: `kv_mode=staged` (the gate driver's step, the K/V history written
    by the host every step), `kv_mode=iss` (the bucket program, the pool in SRAM) and `generate_paged` agree. (The TS layout and
    the paged layout differ by an f16 rounding flip at a few percent of the steps: the prompt is one for which they do
    not move the argmax; `test_paged_reference` bounds the difference.)"""
    from opentsp_vllm.models.tinystories import program_cache as PC
    model = PC.Model(8)
    prompt, n_new = [7454, 2402, 257], 3
    ref = G.generate_paged(model.w, prompt, n_new).tokens[len(prompt):]
    subs = [(0, prompt, n_new)]
    staged, sdrv = drive(model.w, subs, "staged", work=str(tmp_path))
    paged, pdrv = drive(model.w, subs, "iss", model=model)
    assert staged["r0"] == paged["r0"] == ref
    lanes = [sum(len(st.lanes) for st in d.steps) for d in (sdrv, pdrv)]
    assert lanes == [len(prompt) + n_new - 1] * 2 and len(sdrv.steps) == len(pdrv.steps) == 1 + n_new - 1
    assert all(r.cycles for r in sdrv.records) and all(r.cycles for r in pdrv.records)


# -- the full 8-layer model (`just showcase-paged-full`) -----------------------------------------------------------------------------------
@pytest.mark.paged_full
def test_the_full_model_runs_two_requests_sharing_a_prefix_on_the_iss(capsys):
    """Two requests sharing one full block, 4 new tokens each, on the 8-layer model (the real checkpoint when the Hugging Face
    cache has it, else synthetic weights): tokens equal `generate_paged`, every step's logits equal bit-exact reference's (the driver verifies)."""
    ckpt = "calum/tinystories-gpt2-3M"
    try:
        s = DP.run_scenario("full", 8, ckpt, os.environ.get("OPENTSP_VLLM_CACHE"), say=print)
    except Exception as e:                                    # no cached checkpoint, or no way to read it
        if "checkpoint" not in repr(e).lower() and "huggingface" not in repr(e).lower() and "offline" not in repr(e).lower():
            raise
        print(f"checkpoint unavailable ({type(e).__name__}): synthetic weights")
        s = DP.run_scenario("full", 8, None, os.environ.get("OPENTSP_VLLM_CACHE"), say=print)
    assert s["ok"] and any(r["hits"] for r in s["rows"])
