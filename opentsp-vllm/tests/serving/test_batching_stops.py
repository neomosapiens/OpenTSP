""" : continuous batching, streaming, stop strings, stop token ids, EOS, `ignore_eos`, `max_tokens`,
`min_tokens` and abort, through a real `LLM` engine on the 2-layer test model, on `ReferenceChip` and on `IssChip`.

The engine is driven step by step (`Rig.drive`): requests join and leave between steps, the output processor streams deltas, and every
chip step's logit rows are recorded per request. The references are the request's own solo run on the same chip (tokens and logits
bit for bit: batch invariance) and the bit-exact reference oracle (the same greedy loop on `paged_reference`)."""

import numpy as np
import pytest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

import batching_helpers as H
from batching_helpers import EOS, reference_run, same_bits

CHIPS = ["exact", "iss"]
CUMULATIVE, DELTA, FINAL_ONLY = (
    RequestOutputKind.CUMULATIVE,
    RequestOutputKind.DELTA,
    RequestOutputKind.FINAL_ONLY,
)


@pytest.fixture(scope="module", params=CHIPS)
def engine_rig(request, make_llm):
    # the stop strings and the streamed text need the model's tokenizer: the serving engine is built with it
    return H.Rig(make_llm(chip=request.param, model="test", **H.tokenizer_kwargs()))


@pytest.fixture
def rig(engine_rig):
    assert engine_rig.idle()
    engine_rig.llm.reset_prefix_cache()
    yield engine_rig
    assert engine_rig.idle()


def prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(1, H.VOCAB, n)]


def greedy(max_tokens, **kw):
    kw.setdefault("ignore_eos", True)
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, **kw)


# the greedy stream of P2 under the synthetic weights, 2 layers, is [a, a, b, b, b, ...]: b first occurs at the third token
P2 = [42094, 13148, 5494, 15001, 20797]


def assert_rows_equal(rows, reference, what):
    assert len(rows) == len(reference), f"{what}: {len(rows)} logit rows against {len(reference)}"
    for i, (a, b) in enumerate(zip(rows, reference)):
        assert same_bits(a, b), f"{what}: the logits of token {i} differ"


# -- continuous batching -------------------------------------------------------------------------------------------------------


def test_requests_that_join_and_leave_between_steps_equal_their_solo_runs(rig):
    specs = [  # id, prompt, max_tokens, step of arrival
        ("r0", prompt(100, 7), 6, 0),
        ("r1", prompt(101, 3), 8, 1),
        ("r2", prompt(102, 8), 3, 3),
        ("r3", prompt(103, 5), 7, 5),
    ]
    batch = rig.drive([(rid, p, greedy(n), at) for rid, p, n, at in specs])
    # the batch really overlapped: a later request started while an earlier one ran, and one left before the others ended
    assert batch["r0"].first_step < batch["r1"].first_step <= batch["r0"].finish_step
    assert batch["r2"].first_step <= batch["r1"].finish_step and batch["r3"].first_step <= batch["r1"].finish_step
    assert batch["r2"].finish_step < batch["r1"].finish_step < batch["r3"].finish_step
    assert max(len(set(lanes)) for _, _, lanes in rig.last_log) >= 3, "three requests shared a step"
    assert max(len(ids) for ids, _, _ in rig.last_log) >= 2, "two requests got a logit row in one step"
    for rid, p, n, _ in specs:
        got = batch[rid]
        solo = rig.solo(p, greedy(n))
        assert got.tokens == solo.tokens == H.reference_tokens(p, n), rid
        assert got.finish_reason == "length"
        assert_rows_equal(got.rows, solo.rows, f"{rid} against its solo run")
        assert_rows_equal(got.rows, reference_run(p, n)[1], f"{rid} against bit-exact reference")


def test_more_requests_than_sequences_queue_and_join_as_others_leave(rig):
    # max_num_seqs is 4 and the budget 8 lanes: the fifth request waits for a place
    specs = [(f"q{i}", prompt(110 + i, 6), 3 + i % 3) for i in range(5)]
    batch = rig.drive([(rid, p, greedy(n), 0) for rid, p, n in specs])
    assert batch["q4"].first_step > min(batch[f"q{i}"].first_step for i in range(4))
    for rid, p, n in specs:
        solo = rig.solo(p, greedy(n))
        assert batch[rid].tokens == solo.tokens == H.reference_tokens(p, n), rid
        assert_rows_equal(batch[rid].rows, solo.rows, rid)


def test_llm_generate_with_prompts_of_different_lengths_equals_the_reference_streams(rig):
    # the public batch API: six prompts, two more than the sequence limit, of lengths 1 to 8 and different budgets
    prompts = [prompt(170 + i, n) for i, n in enumerate((1, 8, 3, 6, 2, 5))]
    budgets = [4, 2, 6, 3, 5, 1]
    params = [greedy(n, output_kind=FINAL_ONLY) for n in budgets]
    outs = rig.llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
    assert [o.request_id for o in outs] == sorted(o.request_id for o in outs)
    for o, p, n in zip(outs, prompts, budgets):
        assert list(o.outputs[0].token_ids) == H.reference_tokens(p, n) and o.outputs[0].finish_reason == "length"
    assert rig.kv_usage() == 0


def test_a_request_joining_a_running_one_reproduces_its_solo_logits_bit_for_bit(rig):
    long, late = prompt(120, 8), prompt(121, 8)
    batch = rig.drive([("long", long, greedy(8), 0), ("late", late, greedy(8), 4)])
    assert batch["late"].first_step > batch["long"].first_step
    for rid, p in (("long", long), ("late", late)):
        solo = rig.solo(p, greedy(8))
        assert batch[rid].tokens == solo.tokens
        assert_rows_equal(batch[rid].rows, solo.rows, rid)


# -- streaming -----------------------------------------------------------------------------------------------------------------


def test_streamed_deltas_concatenate_to_the_final_text(rig):
    specs = [("s0", P2, 8), ("s1", prompt(130, 4), 6), ("s2", prompt(131, 7), 5)]
    final = {rid: rig.solo(p, greedy(n, output_kind=FINAL_ONLY), rid) for rid, p, n in specs}
    # the three streams run together, one request joining a step late: `stream=True` is `RequestOutputKind.DELTA`
    stream = rig.drive([(rid, p, greedy(n, output_kind=DELTA), 1 if rid == "s2" else 0) for rid, p, n in specs])
    for rid, p, n in specs:
        s, f = stream[rid], final[rid]
        assert f.tokens and len(f.deltas) == 1, "a final-only request returns its output once"
        assert s.tokens == f.tokens and s.text == f.text and s.text, rid
        assert "".join(text for _, text in s.deltas) == f.text, "the concatenated deltas are the final text"
        assert [ids for ids, _ in s.deltas] == [[t] for t in f.tokens], "stream_interval 1: one new token per output"
        assert s.finish_reason == f.finish_reason == "length"


def test_cumulative_outputs_extend_each_other_and_end_at_the_final_text(rig):
    cum = rig.solo(P2, greedy(8, output_kind=CUMULATIVE))
    final = rig.solo(P2, greedy(8, output_kind=FINAL_ONLY))
    assert cum.text == final.text and len(cum.texts) == 8
    assert all(b.startswith(a) for a, b in zip(cum.texts, cum.texts[1:]))
    assert [len(ids) for ids, _ in cum.deltas] == list(range(1, 9))


# -- stop strings --------------------------------------------------------------------------------------------------------------


def first_hit(texts, s):
    """The index of the first output whose cumulative text contains `s`."""
    return next(k for k, t in enumerate(texts) if s in t)


@pytest.fixture
def p2_stream(rig):
    base = rig.solo(P2, greedy(8, output_kind=CUMULATIVE))
    assert base.tokens[2] not in base.tokens[:2], "the test needs a stream whose third token is a new one"
    return base


def test_a_stop_string_ends_the_request_at_the_token_that_completes_it(rig, p2_stream):
    base = p2_stream
    s = rig.tok.decode([base.tokens[2]])
    k = first_hit(base.texts, s)
    assert k == 2
    r = rig.solo(P2, greedy(8, stop=[s], ignore_eos=False))
    assert r.finish_reason == "stop" and r.stop_reason == s
    assert r.tokens == base.tokens[: k + 1] and r.tokens == H.reference_tokens(P2, 8)[: k + 1]
    assert r.text == base.texts[k][: base.texts[k].index(s)], "the stop string is cut off the text"
    assert_rows_equal(r.rows, reference_run(P2, k + 1)[1], "the rows of the stopped request")
    inc = rig.solo(P2, greedy(8, stop=[s], include_stop_str_in_output=True))
    assert inc.tokens == r.tokens and inc.text == base.texts[k][: base.texts[k].index(s) + len(s)]


def test_a_stop_string_that_spans_two_tokens_is_found(rig, p2_stream):
    base = p2_stream
    s = base.texts[1][-2:] + base.texts[2][len(base.texts[1]) :][:2]  # the end of token 1 and the start of token 2
    assert len(s) == 4 and s not in base.texts[1]
    k = first_hit(base.texts, s)
    assert k == 2
    r = rig.solo(P2, greedy(8, stop=["never in this text", s]))
    assert r.finish_reason == "stop" and r.stop_reason == s
    assert r.tokens == base.tokens[: k + 1]
    assert r.text == base.texts[k][: base.texts[k].index(s)]


def test_a_stop_string_that_never_appears_changes_nothing(rig, p2_stream):
    r = rig.solo(P2, greedy(8, stop=["the stream has no such string"]))
    assert r.tokens == p2_stream.tokens and r.text == p2_stream.text and r.finish_reason == "length"


def test_a_stopping_request_leaves_the_others_of_the_batch_unaffected(rig, p2_stream):
    base = p2_stream
    s = rig.tok.decode([base.tokens[2]])
    others = [("o1", prompt(140, 6), 6), ("o2", prompt(141, 4), 8)]
    batch = rig.drive(
        [("stopper", P2, greedy(8, stop=[s]), 0)] + [(rid, p, greedy(n), 0) for rid, p, n in others]
    )
    assert batch["stopper"].finish_reason == "stop" and batch["stopper"].tokens == base.tokens[:3]
    assert batch["stopper"].finish_step < min(batch[rid].finish_step for rid, _, _ in others)
    solo = rig.solo(P2, greedy(8, stop=[s]))
    assert_rows_equal(batch["stopper"].rows, solo.rows, "stopper")
    for rid, p, n in others:
        alone = rig.solo(p, greedy(n))
        assert batch[rid].tokens == alone.tokens and batch[rid].finish_reason == "length", rid
        assert_rows_equal(batch[rid].rows, alone.rows, rid)


# -- stop token ids, EOS, ignore_eos, max_tokens, min_tokens --------------------------------------------------------------------


def test_a_stop_token_id_ends_the_request_at_its_first_occurrence(rig, p2_stream):
    t = p2_stream.tokens
    r = rig.solo(P2, greedy(8, stop_token_ids=[t[2]]))
    assert r.finish_reason == "stop" and r.stop_reason == t[2]
    assert r.tokens == t[:3] == H.reference_tokens(P2, 8)[:3], "the stop token is the last token of the output"
    first = rig.solo(P2, greedy(8, stop_token_ids=[999, t[0]]))
    assert first.tokens == t[:1] and first.finish_reason == "stop" and first.stop_reason == t[0]
    never = rig.solo(P2, greedy(8, stop_token_ids=[999]))
    assert never.tokens == t and never.finish_reason == "length" and never.stop_reason is None


def eos_pick(min_tokens, bias=100.0):
    def pick(logits, generated):
        biased = logits.copy()
        biased[EOS] += bias
        if len(generated) < min_tokens:
            biased[EOS] = -np.inf
        return int(np.argmax(biased))

    return pick


def test_the_eos_token_ends_the_request_unless_ignore_eos(rig):
    # the synthetic model never emits the end-of-text token, so a logit bias makes it the greedy choice
    bias = {EOS: 100.0}
    now = rig.solo(P2, SamplingParams(temperature=0.0, max_tokens=8, logit_bias=bias))
    assert now.tokens == [EOS] and now.finish_reason == "stop" and now.stop_reason is None
    assert now.text == "", "the end-of-text token is special: not in the text"
    assert now.tokens == reference_run(P2, 8, eos_pick(0), stop=lambda out: out[-1] == EOS)[0]
    ignored = rig.solo(P2, SamplingParams(temperature=0.0, max_tokens=8, logit_bias=bias, ignore_eos=True))
    expected, rows = reference_run(P2, 8, eos_pick(0))
    assert ignored.tokens == expected == [EOS] * 8 and ignored.finish_reason == "length"
    assert_rows_equal(ignored.rows, rows, "ignore_eos")


def test_max_tokens_ends_the_request_after_exactly_that_many_tokens(rig):
    for n in (1, 2, 5):
        r = rig.solo(P2, greedy(n))
        expected, rows = reference_run(P2, n)
        assert r.tokens == expected and len(r.tokens) == n and r.finish_reason == "length" and r.stop_reason is None, n
        assert_rows_equal(r.rows, rows, f"max_tokens {n}")


def test_min_tokens_suppresses_the_eos_token_and_stop_token_ids_until_reached(rig):
    bias = {EOS: 100.0}
    r = rig.solo(P2, SamplingParams(temperature=0.0, max_tokens=8, min_tokens=3, logit_bias=bias))
    expected, rows = reference_run(P2, 8, eos_pick(3), stop=lambda out: out[-1] == EOS)
    assert r.tokens == expected and len(r.tokens) == 4 and r.tokens[3] == EOS and EOS not in r.tokens[:3]
    assert r.finish_reason == "stop"
    assert_rows_equal(r.rows, rows, "eos after min_tokens")

    first = H.reference_tokens(P2, 1)[0]
    plain = rig.solo(P2, greedy(8, stop_token_ids=[first]))
    assert plain.tokens == [first], "without min_tokens the first token stops the request"

    def masked(logits, generated):
        logits = logits.copy()
        if len(generated) < 3:
            logits[first] = -np.inf
        return int(np.argmax(logits))

    expected, rows = reference_run(P2, 8, masked, stop=lambda out: out[-1] == first and len(out) >= 3)
    held = rig.solo(P2, greedy(8, stop_token_ids=[first], min_tokens=3))
    assert held.tokens == expected and len(held.tokens) >= 3 and first not in held.tokens[:2]
    assert_rows_equal(held.rows, rows, "stop id after min_tokens")


def test_the_stop_conditions_of_a_batch_end_each_request_at_its_own_token(rig, p2_stream):
    t = p2_stream.tokens
    s = rig.tok.decode([t[2]])
    batch = rig.drive(
        [
            ("by_string", P2, greedy(8, stop=[s]), 0),
            ("by_id", P2, greedy(8, stop_token_ids=[t[2]]), 0),
            ("by_eos", P2, SamplingParams(temperature=0.0, max_tokens=8, min_tokens=2, logit_bias={EOS: 100.0}), 1),
            ("by_length", P2, greedy(5), 1),
        ]
    )
    assert batch["by_string"].tokens == batch["by_id"].tokens == t[:3]
    assert batch["by_string"].finish_reason == batch["by_id"].finish_reason == "stop"
    assert batch["by_string"].stop_reason == s and batch["by_id"].stop_reason == t[2]
    assert batch["by_eos"].tokens == reference_run(P2, 8, eos_pick(2), stop=lambda out: out[-1] == EOS)[0]
    assert batch["by_eos"].finish_reason == "stop" and batch["by_eos"].tokens[-1] == EOS
    assert batch["by_length"].tokens == t[:5] and batch["by_length"].finish_reason == "length"


# -- abort ---------------------------------------------------------------------------------------------------------------------


def test_an_abort_mid_stream_leaves_the_others_unaffected_and_frees_the_blocks(rig):
    pa, pb, pc = prompt(150, 8), prompt(151, 8), prompt(152, 5)
    batch = rig.drive(
        [("a", pa, greedy(8), 0), ("b", pb, greedy(8), 0), ("c", pc, greedy(8), 1)], aborts={"b": 4}
    )
    assert batch["b"].aborted and batch["b"].finish_reason is None and batch["b"].finish_step is None
    assert 0 < len(batch["b"].tokens) < 8, "b was aborted while it was generating"
    solo_b = rig.solo(pb, greedy(8))
    assert batch["b"].tokens == solo_b.tokens[: len(batch["b"].tokens)], "what b returned before the abort is its solo stream"
    for rid, p in (("a", pa), ("c", pc)):
        alone = rig.solo(p, greedy(8))
        assert batch[rid].tokens == alone.tokens and batch[rid].finish_reason == "length", rid
        assert_rows_equal(batch[rid].rows, alone.rows, rid)
    # drive checked that the KV usage is back at 0; the worker forgets the requests with the next step
    again = rig.solo(pb, greedy(8), "again")
    assert again.tokens == solo_b.tokens
    assert all(rid.startswith("solo") or rid.startswith("again") for rid in (r.rsplit("-", 1)[0] for r in rig.runner_requests()))
    assert len(rig.runner_requests()) == 1, "only the last request's state is left in the worker"


def test_an_abort_of_a_waiting_request_and_of_an_unknown_id_is_harmless(rig):
    specs = [(f"w{i}", prompt(160 + i, 8), 4) for i in range(5)]  # the fifth cannot start while four run
    rig.engine.abort_request(["no such request"])
    batch = rig.drive([(rid, p, greedy(n), 0) for rid, p, n in specs], aborts={"w4": 1})
    assert batch["w4"].aborted and batch["w4"].tokens == [] and batch["w4"].first_step is None
    for rid, p, n in specs[:4]:
        assert batch[rid].tokens == H.reference_tokens(p, n) and batch[rid].finish_reason == "length", rid


# -- streaming over HTTP: `stream=true` of the OpenAI-compatible server (AsyncLLM, the engine core in its own process) -----------


@pytest.fixture(scope="module", params=CHIPS)
def server(request, tmp_path_factory):
    s = H.Server(request.param, tmp_path_factory.mktemp("serve"))
    yield s
    s.close()


def test_a_streamed_http_completion_concatenates_to_the_whole_one(server):
    import concurrent.futures

    prompts = {"a": P2, "b": prompt(180, 6), "c": prompt(181, 3)}
    budgets = {"a": 8, "b": 6, "c": 5}
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        futures = {r: pool.submit(server.stream, p, max_tokens=budgets[r]) for r, p in prompts.items()}
        streams = {r: f.result() for r, f in futures.items()}
    for r, p in prompts.items():
        whole = server.complete(p, max_tokens=budgets[r])["choices"][0]
        events = streams[r]
        assert len(events) >= budgets[r], "one event per token (stream_interval 1)"
        assert "".join(e["choices"][0]["text"] for e in events) == whole["text"] and whole["text"], r
        assert events[-1]["choices"][0]["finish_reason"] == whole["finish_reason"] == "length"
        assert all(e["choices"][0]["finish_reason"] is None for e in events[:-1])


def test_a_client_that_disconnects_mid_stream_aborts_its_request_and_the_others_continue(server):
    import concurrent.futures
    import time

    other = prompt(190, 8)
    reference = server.complete(other, max_tokens=8)["choices"][0]["text"]
    p2_text = server.complete(P2, max_tokens=4)["choices"][0]["text"]
    generated = lambda: server.metric("vllm:generation_tokens_total")  # noqa: E731
    finished_by_length = lambda: server.metric("vllm:request_success_total", 'finished_reason="length"')  # noqa: E731
    tokens0, length0 = generated(), finished_by_length()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        walker = pool.submit(server.stream, P2, stop_after=2, max_tokens=40)  # would run for 40 tokens
        steady = pool.submit(server.stream, other, max_tokens=8)
        assert len(walker.result()) == 2
        events = steady.result()
    assert "".join(e["choices"][0]["text"] for e in events) == reference
    deadline = time.monotonic() + 30
    while server.metric("vllm:num_requests_running") or server.metric("vllm:num_requests_waiting"):
        assert time.monotonic() < deadline, "the aborted request is still in the engine"
        time.sleep(0.2)
    time.sleep(1)
    assert generated() - tokens0 < 8 + 20, "the aborted request stopped generating soon after the disconnect"
    assert finished_by_length() == length0 + 1, "only the request that was not aborted ran to its length"
    assert server.metric("vllm:kv_cache_usage_perc") == 0
    assert server.complete(P2, max_tokens=4)["choices"][0]["text"] == p2_text, "the engine serves on"
