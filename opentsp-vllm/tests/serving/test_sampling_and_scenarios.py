"""the sampling features of  (temperature, top-k, top-p, min-p, seed, logprobs, penalties, logit bias,
bad words, allowed token ids, `n` = 2), through a real vLLM `LLM` on the TSP plugin, on the 2-layer test model. The reference is bit-exact reference
(`sampling_helpers.Oracle`): the sampler runs on the host over the chip's logits, which equal bit-exact reference's bit for bit, so the expected tokens
(greedy under a transform) and the expected sets (top-k, nucleus, min-p) are computed from bit-exact reference's logits in numpy.
"""

import numpy as np
import pytest
import torch

import sampling_helpers as H
from vllm import SamplingParams

from opentsp_vllm.models.tinystories import layout
from engine import reference_tokens as Reference_TOKENS

CFG = layout.serve_config(2)  # the test model: 2 layers, the GPT-2 vocabulary of 50 257 words
CHIPS = ("exact", "iss")


@pytest.fixture(scope="module")
def oracle():
    return H.Oracle(CFG, seed=1)


@pytest.fixture(scope="module", params=CHIPS)
def eng(request, make_llm):
    return make_llm(request.param, "test")


@pytest.fixture(autouse=True)
def fresh_prefix_cache(eng):
    eng.llm.reset_prefix_cache()


def run(eng, prompts, params):
    """The completions, in order, of each (prompt, params) pair generated in one batch."""
    if not isinstance(params, list):
        params = [params] * len(prompts)
    return [o.outputs for o in eng.generate(prompts, params)]


def greedy(**kw):
    return SamplingParams(temperature=0.0, ignore_eos=True, **kw)


PROMPT = H.prompt_ids(3, 6, CFG.vocab)
N = 6


def test_greedy_equals_the_oracle(eng, oracle):
    (out,) = run(eng, [PROMPT], greedy(max_tokens=N))
    assert list(out[0].token_ids) == oracle.generate(PROMPT, N)


def test_a_tiny_temperature_is_greedy(eng, oracle):
    (out,) = run(eng, [PROMPT], SamplingParams(temperature=1e-4, seed=5, ignore_eos=True, max_tokens=N))
    assert list(out[0].token_ids) == oracle.generate(PROMPT, N)


def test_temperature_one_samples_other_than_greedy_tokens(eng, oracle):
    prompts = [PROMPT] * 4
    outs = run(eng, prompts, [SamplingParams(temperature=1.0, seed=100 + i, ignore_eos=True, max_tokens=N) for i in range(4)])
    sampled = [list(o[0].token_ids) for o in outs]
    assert any(s != oracle.generate(PROMPT, N) for s in sampled)
    assert len({tuple(s) for s in sampled}) > 1, "four seeds gave one stream"


def _members(oracle, prompt, tokens, keep):
    """Check that every sampled token lies in the set `keep(logits_row)` of the bit-exact reference logits that picked it."""
    rows = oracle.teacher_logits(prompt, tokens[:-1])
    for k, t in enumerate(tokens):
        allowed = keep(rows[k])
        assert t in allowed, f"position {k}: token {t} outside the expected set {sorted(allowed)[:10]}..."


SEEDS = list(range(8))


def test_top_k_samples_stay_in_the_top_k_of_the_reference_logits(eng, oracle):
    k = 5
    outs = run(eng, [PROMPT] * len(SEEDS), [SamplingParams(temperature=0.8, top_k=k, seed=s, ignore_eos=True, max_tokens=N) for s in SEEDS])
    for o in outs:
        _members(oracle, PROMPT, list(o[0].token_ids), lambda r: set(np.argsort(-r, kind="stable")[:k].tolist()))


def test_top_k_one_is_greedy(eng, oracle):
    (out,) = run(eng, [PROMPT], SamplingParams(temperature=2.0, top_k=1, seed=9, ignore_eos=True, max_tokens=N))
    assert list(out[0].token_ids) == oracle.generate(PROMPT, N)


def _softmax(row, temperature=1.0):
    z = row.astype(np.float64) / temperature
    z -= z.max()
    p = np.exp(z)
    return p / p.sum()


def _nucleus(row, top_p):
    """The smallest prefix of the tokens by decreasing probability whose mass reaches top_p (at least one token)."""
    p = _softmax(row)
    order = np.argsort(-p, kind="stable")
    before = np.cumsum(p[order]) - p[order]
    return set(order[before < top_p].tolist())


def test_top_p_samples_stay_in_the_nucleus_of_the_reference_logits(eng, oracle):
    top_p = 0.3
    outs = run(eng, [PROMPT] * len(SEEDS), [SamplingParams(temperature=1.0, top_p=top_p, seed=s, ignore_eos=True, max_tokens=N) for s in SEEDS])
    sizes = []
    for o in outs:
        tokens = list(o[0].token_ids)
        _members(oracle, PROMPT, tokens, lambda r: _nucleus(r, top_p + 1e-4))
        sizes += [len(_nucleus(r, top_p)) for r in oracle.teacher_logits(PROMPT, tokens[:-1])]
    assert max(sizes) > 1, "the nucleus is a single token everywhere: the test says nothing about top_p"


def test_min_p_samples_have_probability_at_least_min_p_times_the_maximum(eng, oracle):
    min_p = 0.5
    outs = run(eng, [PROMPT] * len(SEEDS), [SamplingParams(temperature=1.0, min_p=min_p, seed=s, ignore_eos=True, max_tokens=N) for s in SEEDS])
    widths = []

    def keep(r):
        p = _softmax(r)
        return set(np.nonzero(p >= min_p * p.max() * (1 - 1e-5))[0].tolist())

    for o in outs:
        tokens = list(o[0].token_ids)
        _members(oracle, PROMPT, tokens, keep)
        widths += [len(keep(r)) for r in oracle.teacher_logits(PROMPT, tokens[:-1])]
    assert max(widths) > 1, "min_p keeps one token everywhere: the test says nothing about min_p"


def test_seeded_sampling_alone_equals_batched(eng):
    params = lambda: SamplingParams(temperature=0.9, top_p=0.95, seed=1234, ignore_eos=True, max_tokens=N)  # noqa: E731
    (alone,) = run(eng, [PROMPT], params())
    others = [H.prompt_ids(40 + i, 3 + 2 * i, CFG.vocab) for i in range(3)]
    mixed = [SamplingParams(temperature=1.1, seed=77 + i, ignore_eos=True, max_tokens=N + i) for i in range(3)]
    for position in (0, 2, 3):  # the seeded request first, in the middle, last
        prompts = others[:position] + [PROMPT] + others[position:]
        plist = mixed[:position] + [params()] + mixed[position:]
        outs = run(eng, prompts, plist)
        assert list(outs[position][0].token_ids) == list(alone[0].token_ids), f"the seeded request at row {position}"
    # an unseeded neighbour does not disturb it either
    outs = run(eng, [others[0], PROMPT], [SamplingParams(temperature=1.0, ignore_eos=True, max_tokens=N), params()])
    assert list(outs[1][0].token_ids) == list(alone[0].token_ids)


def test_different_seeds_give_different_streams(eng):
    outs = run(eng, [PROMPT] * 2, [SamplingParams(temperature=1.5, seed=s, ignore_eos=True, max_tokens=8) for s in (1, 2)])
    assert list(outs[0][0].token_ids) != list(outs[1][0].token_ids)


def test_logprobs_equal_log_softmax_of_the_reference_logits(eng, oracle):
    k = 3
    (out,) = run(eng, [PROMPT], greedy(max_tokens=N, logprobs=k))
    tokens = list(out[0].token_ids)
    rows = oracle.teacher_logits(PROMPT, tokens[:-1])
    assert len(out[0].logprobs) == N
    for step, (row, lp) in enumerate(zip(rows, out[0].logprobs)):
        expect = torch.log_softmax(torch.from_numpy(row).unsqueeze(0), dim=-1, dtype=torch.float32)[0]
        top = torch.topk(expect, k)
        assert tokens[step] in lp and set(top.indices.tolist()) <= set(lp), f"step {step}: the top-{k} tokens"
        for tok, entry in lp.items():
            assert entry.logprob == pytest.approx(float(expect[tok]), abs=0, rel=0), f"step {step} token {tok}"
        ranks = {tok: entry.rank for tok, entry in lp.items()}
        order = torch.argsort(expect, descending=True).tolist()
        assert all(ranks[tok] == order.index(tok) + 1 for tok in ranks), f"step {step} ranks"


def test_logprobs_zero_returns_only_the_sampled_token(eng, oracle):
    (out,) = run(eng, [PROMPT], greedy(max_tokens=3, logprobs=0))
    tokens = list(out[0].token_ids)
    rows = oracle.teacher_logits(PROMPT, tokens[:-1])
    for step, lp in enumerate(out[0].logprobs):
        assert list(lp) == [tokens[step]]
        expect = torch.log_softmax(torch.from_numpy(rows[step]).unsqueeze(0), dim=-1, dtype=torch.float32)[0, tokens[step]]
        assert lp[tokens[step]].logprob == float(expect)


def test_logprobs_in_a_batch_equal_the_solo_values(eng):
    ps = [PROMPT, H.prompt_ids(8, 9, CFG.vocab), H.prompt_ids(9, 4, CFG.vocab)]
    solo = [run(eng, [p], greedy(max_tokens=4, logprobs=2))[0][0] for p in ps]
    batch = run(eng, ps, greedy(max_tokens=4, logprobs=2))
    for s, b in zip(solo, batch):
        assert list(b[0].token_ids) == list(s.token_ids)
        assert b[0].logprobs == s.logprobs


# -- penalties, logit bias, bad words, allowed ids: greedy under a transform of the bit-exact reference logits ---------------------------------


def _repeat(prompt, out, vocab):
    seen = np.zeros(vocab, bool)
    seen[list(prompt) + list(out)] = True
    return seen


def test_presence_penalty(eng, oracle):
    pen = 2.0

    def tf(row, prompt, out):
        row[sorted(set(out))] -= pen
        return row

    # the penalty is at most 2, so it changes the greedy stream only where the top two logits are close: pick such a prompt
    prompt = next(p for p in (H.prompt_ids(s, 6, CFG.vocab) for s in range(200, 260)) if oracle.generate(p, 8, tf) != oracle.generate(p, 8))
    (out,) = run(eng, [prompt], greedy(max_tokens=8, presence_penalty=pen))
    assert list(out[0].token_ids) == oracle.generate(prompt, 8, tf)
    assert list(out[0].token_ids) != oracle.generate(prompt, 8)


def test_frequency_penalty(eng, oracle):
    pen = 1.5

    def tf(row, prompt, out):
        for t in set(out):
            row[t] -= pen * out.count(t)
        return row

    (out,) = run(eng, [PROMPT], greedy(max_tokens=8, frequency_penalty=pen))
    expect = oracle.generate(PROMPT, 8, tf)
    assert list(out[0].token_ids) == expect
    assert expect != oracle.generate(PROMPT, 8)


def test_repetition_penalty_covers_the_prompt_and_the_output(eng, oracle):
    pen = 4.0

    def tf(row, prompt, out):
        seen = _repeat(prompt, out, row.shape[0])
        row[seen] = np.where(row[seen] > 0, row[seen] / pen, row[seen] * pen)
        return row

    (out,) = run(eng, [PROMPT], greedy(max_tokens=8, repetition_penalty=pen))
    expect = oracle.generate(PROMPT, 8, tf)
    assert list(out[0].token_ids) == expect
    assert expect != oracle.generate(PROMPT, 8)


def test_penalties_apply_per_request_in_a_batch(eng, oracle):
    p2 = H.prompt_ids(21, 5, CFG.vocab)
    outs = run(eng, [PROMPT, p2], [greedy(max_tokens=6, presence_penalty=2.0), greedy(max_tokens=6)])

    def tf(row, prompt, out):
        row[sorted(set(out))] -= 2.0
        return row

    assert list(outs[0][0].token_ids) == oracle.generate(PROMPT, 6, tf)
    assert list(outs[1][0].token_ids) == oracle.generate(p2, 6)


def test_logit_bias_forces_and_bans_tokens(eng, oracle):
    base = oracle.generate(PROMPT, 5)
    forced = 1999
    banned = {base[0]: -100.0, base[1]: -100.0}

    (a,) = run(eng, [PROMPT], greedy(max_tokens=5, logit_bias={forced: 100.0}))
    assert list(a[0].token_ids) == [forced] * 5

    def tf(row, prompt, out):
        for t, b in banned.items():
            row[t] += b
        return row

    (b,) = run(eng, [PROMPT], greedy(max_tokens=5, logit_bias=banned))
    expect = oracle.generate(PROMPT, 5, tf)
    assert list(b[0].token_ids) == expect
    assert base[0] not in expect and base[1] not in expect


def test_bad_words_ban_single_tokens_and_sequences(eng, oracle):
    base = oracle.generate(PROMPT, 6)
    sp = greedy(max_tokens=6)
    # a one-token bad word is banned everywhere; a two-token one bans its second token only after its first
    sp._bad_words_token_ids = [[base[0]], [base[2], base[3]]]
    (out,) = run(eng, [PROMPT], sp)

    def tf(row, prompt, out):
        row[base[0]] = -np.inf
        if out and out[-1] == base[2]:
            row[base[3]] = -np.inf
        return row

    assert list(out[0].token_ids) == oracle.generate(PROMPT, 6, tf)
    assert base[0] not in out[0].token_ids


def test_allowed_token_ids_restrict_the_choice(eng, oracle):
    allowed = [3, 17, 900, 1500]
    (out,) = run(eng, [PROMPT], greedy(max_tokens=5, allowed_token_ids=allowed))

    def tf(row, prompt, out):
        mask = np.full_like(row, -np.inf)
        mask[allowed] = row[allowed]
        return mask

    assert list(out[0].token_ids) == oracle.generate(PROMPT, 5, tf)
    assert set(out[0].token_ids) <= set(allowed)


# -- n = 2 -------------------------------------------------------------------------------------------------------------------


def test_n_two_deterministic_children_give_two_identical_completions_and_the_second_hits_the_prefix(eng, oracle):
    prompt = H.prompt_ids(50, 20, CFG.vocab)  # one full block and 4 tokens
    sp = SamplingParams(n=2, temperature=1.0, top_k=1, seed=3, ignore_eos=True, max_tokens=4)  # n > 1 needs sampling; top_k=1 is greedy
    (outs,) = eng.generate([prompt], sp)
    assert [o.index for o in outs.outputs] == [0, 1]
    expect = oracle.generate(prompt, 4)
    assert [list(o.token_ids) for o in outs.outputs] == [expect, expect]
    assert outs.num_cached_tokens == 16, "the second child shares the first child's full prompt block"


def test_n_two_sampled_completions_follow_their_child_seeds(eng, oracle):
    prompt = H.prompt_ids(51, 20, CFG.vocab)
    (both,) = run(eng, [prompt], SamplingParams(n=2, temperature=1.2, seed=31, ignore_eos=True, max_tokens=5))
    assert len(both) == 2 and list(both[0].token_ids) != list(both[1].token_ids)
    # the front end gives child i the seed seed + i: each completion equals the solo request with that seed
    for i, o in enumerate(both):
        (solo,) = run(eng, [prompt], SamplingParams(temperature=1.2, seed=31 + i, ignore_eos=True, max_tokens=5))
        assert list(o.token_ids) == list(solo[0].token_ids), f"child {i}"


# -- the two full-model scenarios of  (8 layers, max_model_len 128) -----------------------------------------------


def _record(chip, name, eng, wall, outs, before):
    """What the results note reports: steps, lanes, cycles, wall time, host time per step (written when OPENTSP_SERVING_RECORDS
    names a directory)."""
    import json
    import os

    steps = eng.steps
    cycles = [r.cycles for _, r in steps if r.cycles is not None]
    now = eng.worker.get_tsp_stats()
    stats = {k: now[k] - before[k] for k in ("host_prep_s", "chip_s", "sampler_s")}
    rec = {
        "chip": chip,
        "scenario": name,
        "steps": len(steps),
        "lanes": sum(p.n_real for p, _ in steps),
        "padding_lanes": sum(p.padding_lanes for p, _ in steps),
        "buckets": sorted({list(p.bucket).__repr__() for p, _ in steps}),
        "cycles": sum(cycles) if cycles else None,
        "wall_s": round(wall, 3),
        "generated_tokens": sum(len(o.token_ids) for out in outs for o in out.outputs),
        "host_prep_s": stats.get("host_prep_s"),
        "chip_s": stats.get("chip_s"),
        "sampler_s": stats.get("sampler_s"),
    }
    d = os.environ.get("OPENTSP_SERVING_RECORDS")
    if d:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{chip}-{name}.json"), "w") as f:
            json.dump(rec, f, indent=1)
    return rec


@pytest.fixture(scope="module", params=CHIPS)
def full(request, make_llm):
    return request.param, make_llm(request.param, "full")


def test_full_model_one_request_four_prompt_tokens_eight_new_tokens(full):
    import time

    chip, e = full
    e.llm.reset_prefix_cache()
    e.clear()
    prompt = [7454, 2402, 257, 640]
    before = e.worker.get_tsp_stats()
    t0 = time.perf_counter()
    outs = e.generate([prompt], greedy(max_tokens=8))
    wall = time.perf_counter() - t0
    assert list(outs[0].outputs[0].token_ids) == Reference_TOKENS(prompt, 8, "full")
    assert [p.n_real for p, _ in e.steps] == [4] + [1] * 7, "the 4-token prompt is one chunk, then one step per token"
    _record(chip, "full-one-request", e, wall, outs, before)


def test_full_model_two_requests_sharing_a_block_the_second_hits_the_prefix(full):
    import time

    chip, e = full
    e.llm.reset_prefix_cache()
    e.clear()
    shared = H.prompt_ids(60, 16, CFG.vocab)
    prompts = [shared + [11, 12], shared + [21, 22]]
    before = e.worker.get_tsp_stats()
    t0 = time.perf_counter()
    outs = e.generate(prompts, [greedy(max_tokens=4)] * 2)
    wall = time.perf_counter() - t0
    for p, o in zip(prompts, outs):
        assert list(o.outputs[0].token_ids) == Reference_TOKENS(p, 4, "full")
    assert sorted(o.num_cached_tokens for o in outs) == [0, 16], "one full block is shared"
    _record(chip, "full-shared-prefix", e, wall, outs, before)


def test_the_sampler_gives_the_same_tokens_and_logprobs_on_chip_logits_and_on_reference_logits(make_llm):
    sp = lambda: SamplingParams(temperature=0.9, top_p=0.9, seed=4242, ignore_eos=True, max_tokens=N, logprobs=3)  # noqa: E731
    prompts = [PROMPT, H.prompt_ids(81, 9, CFG.vocab)]
    outs = {chip: [o.outputs[0] for o in make_llm(chip, "test").generate(prompts, [sp(), sp()])] for chip in ("exact", "iss")}
    for exact_ref, iss in zip(outs["exact"], outs["iss"]):
        assert list(exact_ref.token_ids) == list(iss.token_ids)
        assert exact_ref.logprobs == iss.logprobs
