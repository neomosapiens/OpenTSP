"""The showcase session on a real vLLM engine with the GPT-2 tokenizer, the test model (2 layers, context 64) on `ReferenceChip` and, once, on the
ISS: the streamed text is the final text, bad input is a message, Ctrl-C leaves the engine clean, and requests batch and share a prefix."""

import io

from opentsp_vllm.showcase.session import Session, Settings, run_batch
from vllm import SamplingParams

GREEDY = Settings(max_tokens=8, temperature=0.0, top_k=0, seed=1)
SAMPLED = Settings(max_tokens=10, temperature=0.8, top_k=40, seed=3)


def session(backend, settings):
    out, err = io.StringIO(), io.StringIO()
    return Session(backend, out, err, settings), out, err


def final(backend, prompt, settings):
    """`LLM.generate` of the same request: the text and ids of the whole completion, not streamed."""
    ids = backend.encode(prompt)
    params = SamplingParams(max_tokens=settings.max_tokens, **settings.sampling())
    c = backend.llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0]
    return c.text, list(c.token_ids)


def kv_usage(backend):
    return backend.engine.engine_core.engine_core.scheduler.kv_cache_manager.usage


def test_the_streamed_text_is_the_text_generate_returns(make_backend):
    backend = make_backend("exact")
    s, out, err = session(backend, SAMPLED)
    summary = s.generate("Once upon a time")
    text, ids = final(backend, "Once upon a time", SAMPLED)
    assert summary.text == text and summary.token_ids == ids and len(ids) == 10
    assert out.getvalue().splitlines()[0] == "Once upon a time" + text
    assert err.getvalue() == ""
    assert summary.finish_reason == "length" and summary.steps == 10 and kv_usage(backend) == 0


def test_greedy_tokens_equal_the_oracle(make_backend):
    from engine import reference_tokens

    backend = make_backend("exact")
    s, *_ = session(backend, GREEDY)
    summary = s.generate("The little dog")
    assert summary.token_ids == reference_tokens(backend.encode("The little dog"), 8)


def test_non_ascii_text_is_tokenized_and_streams_without_a_traceback(make_backend):
    backend = make_backend("exact")
    prompt = "Le café de la forêt, 日本 ☕"
    assert len(backend.encode(prompt)) > len(prompt) // 2
    s, out, err = session(backend, SAMPLED)
    summary = s.generate(prompt)
    assert summary.tokens == 10 and err.getvalue() == ""
    assert summary.text == final(backend, prompt, SAMPLED)[0]
    assert out.getvalue().startswith(prompt)


def test_a_prompt_longer_than_the_context_is_a_message_and_the_engine_still_serves(make_backend):
    backend = make_backend("exact")
    s, out, err = session(backend, GREEDY)
    assert s.handle("word " * 80)  # 81 tokens
    assert "error: the prompt is 81 tokens, but the chip's context is 64 positions" in err.getvalue()
    assert out.getvalue() == "" and not backend.busy()
    assert s.handle("A cat") and out.getvalue().startswith("A cat")


def test_a_prompt_at_the_limit_gets_what_room_is_left(make_backend):
    backend = make_backend("exact")
    prompt = "a " * 59  # 60 tokens: room for 4 of the 8 asked
    assert len(backend.encode(prompt)) == 60
    s, out, err = session(backend, GREEDY)
    summary = s.generate(prompt)
    assert summary.tokens == 4 and "generating at most 4" in err.getvalue()


def test_an_empty_prompt_starts_a_story_from_the_start_token(make_backend):
    backend = make_backend("exact")
    s, out, err = session(backend, GREEDY)
    summary = s.generate("")
    assert summary.prompt_tokens == 1 and summary.token_ids and "empty prompt" in err.getvalue()


def test_ctrl_c_in_the_middle_leaves_the_engine_clean_and_the_next_prompt_unchanged(make_backend):
    backend = make_backend("exact")
    reference = final(backend, "Once upon", GREEDY)
    s, out, err = session(backend, GREEDY)
    step = backend.step
    count = []

    def stepping():
        count.append(1)
        if len(count) == 6:
            s.interrupt()
        return step()

    backend.step = stepping
    try:
        summary = s.generate("Once upon")
    finally:
        backend.step = step
    assert summary.interrupted and 0 < summary.tokens < 8
    assert not backend.busy() and kv_usage(backend) == 0
    again = s.generate("Once upon")
    assert (again.text, again.token_ids) == reference and not again.interrupted


def test_requests_batch_and_share_a_common_prefix(make_backend):
    backend = make_backend("exact")
    common = "Once upon a time there was a little girl named Lily who lived near a big green forest in a small house with her mother, and every sunny day she"
    prompts = [common + " found a ball.", common + " met a dog.", common + " saw a bird."]
    assert min(len(backend.encode(p)) for p in prompts) >= 34 and max(len(backend.encode(p)) for p in prompts) < 48
    out, err = io.StringIO(), io.StringIO()
    settings = Settings(max_tokens=6, temperature=0.8, top_k=40, seed=5)
    done = run_batch(backend, prompts, settings, out, err)
    assert err.getvalue() == "" and len(done) == 3 and kv_usage(backend) == 0
    assert [d.text for d in done] == [final(backend, p, settings)[0] for p in prompts]  # batching does not change a stream
    # a follower reuses the KV blocks of the leader's prompt that were complete when it was scheduled (whole 16-token blocks)
    cached = [d.num_cached_tokens for d in done]
    assert cached[0] == 0 and max(cached) >= 16 and all(c % 16 == 0 for c in cached)
    assert "from the prefix cache" in out.getvalue()
    assert any(ln.startswith("req-") and " finished at step " in ln for ln in out.getvalue().splitlines())


def test_the_session_on_the_iss_equals_the_session_on_reference_and_counts_cycles(make_backend):
    exact_ref = make_backend("exact")
    expected = session(exact_ref, GREEDY)[0].generate("The little dog").token_ids
    iss = make_backend("tiny")
    s, out, err = session(iss, GREEDY)
    summary = s.generate("The little dog")
    assert summary.token_ids == expected
    assert summary.cycles > 0 and summary.steps >= 8
    assert "ISS cycles" in out.getvalue() and err.getvalue() == ""
    verbose, vout, _ = session(iss, Settings(max_tokens=3, temperature=0.0, top_k=0, seed=1, verbose=True))
    verbose.generate("The little dog")
    assert "scheduled tokens" in vout.getvalue() and "cycles" in vout.getvalue()
