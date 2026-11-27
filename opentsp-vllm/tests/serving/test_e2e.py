"""Compare greedy vLLM token IDs against the TinyStories oracle with synthetic weights."""

import numpy as np
from engine import reference_tokens
from vllm import SamplingParams

PROMPT = [7454, 2402, 257, 640, 612, 373]
GREEDY = SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True)


def test_single_request_reference(make_llm):
    e = make_llm("exact", "test")
    tokens, logits = e.solo(PROMPT, GREEDY)
    assert tokens == reference_tokens(PROMPT, 6)
    assert logits.shape == (6, 50257)
    assert [int(np.argmax(r)) for r in logits] == tokens


def test_single_request_iss(make_llm):
    e = make_llm("iss", "test")
    tokens, logits = e.solo(PROMPT, GREEDY)
    assert tokens == reference_tokens(PROMPT, 6)
    assert [int(np.argmax(r)) for r in logits] == tokens
    reference_logits = make_llm("exact", "test").solo(PROMPT, GREEDY)[1]
    assert np.array_equal(logits, reference_logits)


def test_vllm_serve_answers_a_completions_request(tmp_path):
    """`vllm serve` (the API server and an engine core process) on the ISS chip: one /v1/completions request over HTTP."""
    import json
    import socket
    import subprocess
    import time
    import urllib.request

    import engine

    path = engine.model_dir(tmp_path, "test")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = engine.serve_env()
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING")
    log = open(tmp_path / "serve.log", "w")
    server = subprocess.Popen(engine.serve_command(path, "iss", "test", port, root=tmp_path), env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 150
        while True:
            assert server.poll() is None, f"the server exited ({server.returncode}):\n{(tmp_path / 'serve.log').read_text()[-3000:]}"
            try:
                urllib.request.urlopen(base + "/health", timeout=2)
                break
            except OSError:
                assert time.monotonic() < deadline, "the server did not become healthy in 150 s"
                time.sleep(1)
        body = json.dumps(
            {"model": str(path), "prompt": PROMPT, "max_tokens": 6, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
        )
        request = urllib.request.Request(base + "/v1/completions", body.encode(), {"content-type": "application/json"})
        answer = json.load(urllib.request.urlopen(request, timeout=60))
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    choice = answer["choices"][0]
    assert choice["token_ids"] == reference_tokens(PROMPT, 6)
    assert choice["finish_reason"] == "length" and answer["usage"]["completion_tokens"] == 6


def test_shadow_mode_runs_every_step_on_the_iss_and_on_reference_and_the_tokens_still_equal_reference(make_llm):
    from opentsp_vllm.shadow import ShadowChip

    e = make_llm("iss", "test", shadow="exact")
    assert isinstance(e.chip, ShadowChip)
    tokens, logits = e.solo(PROMPT, GREEDY)
    assert tokens == reference_tokens(PROMPT, 6)
    assert e.chip.shadow.steps == len(e.steps) > 0
    assert e.chip.failure is None


def test_single_request_full_model_iss(make_llm):
    """The full model (8 layers, contexts 64 and 128) once: a prompt that crosses the 64-position bucket, tokens equal `generate_paged`."""
    prompt = [(97 * i + 11) % 50257 for i in range(60)]
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    e = make_llm("iss", "full")
    tokens, logits = e.solo(prompt, params)
    assert tokens == reference_tokens(prompt, 8, "full")
    assert {plan.bucket[2] for plan, _ in e.steps} == {64, 128}
    assert [int(np.argmax(r)) for r in logits] == tokens


def test_single_request_full_model_reference(make_llm):
    prompt = [(97 * i + 11) % 50257 for i in range(60)]
    params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    e = make_llm("exact", "full")
    tokens, logits = e.solo(prompt, params)
    assert tokens == reference_tokens(prompt, 8, "full")
    assert {plan.bucket[2] for plan, _ in e.steps} == {64, 128}
