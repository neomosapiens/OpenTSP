"""Exercise opt-in ISS schedule replay through vLLM on the two-layer test model.

Two engines run the same requests, one on the normal driver (the default) and one with `opentsp.replay` on: the tokens, the logits (bit
for bit) and the simulated cycles of every step are equal, while the replay engine serves steps from the schedule recorded for their
bucket program (`stats["replay"]`). A binding built without the `replay` feature makes this test fail (`just vllm-binding` builds it).
"""

import os

import numpy as np
from vllm import SamplingParams

import batching_helpers as H

GREEDY = SamplingParams(temperature=0.0, max_tokens=14, ignore_eos=True)


def prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(1, H.VOCAB, n)]


# three requests of different lengths: chunked prefill of 8 tokens a step, then decode with up to three lanes
PROMPTS = [prompt(1, 6), prompt(2, 21), prompt(3, 37)]


def run(e):
    """The tokens, the cycle count and the logits of every chip step of the scenario, from an empty prefix cache."""
    e.llm.reset_prefix_cache()
    e.clear()
    outs = e.generate(PROMPTS, GREEDY)
    tokens = [list(o.outputs[0].token_ids) for o in outs]
    cycles = [r.cycles for _, r in e.steps]
    logits = [np.asarray(r.logits).copy() for _, r in e.steps]
    return tokens, cycles, logits


def stats(e):
    return e.chip._chip.stats()["replay"]


def test_replay_on_and_off_serve_the_same_text_and_the_same_cycles_per_step(make_llm):
    off = make_llm("iss", "test", opentsp={"replay": False})
    on = make_llm("iss", "test", opentsp={"replay": True})
    assert stats(off)["enabled"] is False and stats(on)["enabled"] is True, "the replay engine is not the one with the setting on"
    assert stats(on)["built"], "the binding was built without the `replay` feature: run `just vllm-binding`"
    for k in range(4):
        t0, c0, l0 = run(off)
        t1, c1, l1 = run(on)
        assert t1 == t0, f"round {k}: the served text differs"
        assert c1 == c0 and len(c0) > 10, f"round {k}: the cycles per step differ: {c0} against {c1}"
        assert len(l1) == len(l0) and all(np.array_equal(a, b) for a, b in zip(l1, l0)), f"round {k}: the logits differ"
    s, s0 = stats(on), stats(off)
    print("replay on:", {k: v for k, v in s.items() if k != "shapes"})
    for name, row in s["shapes"].items():
        print("  shape", name, row)
    assert (s0["served"], s0["fallback"]) == (0, 0), "the default engine used the replay"
    assert s["served"] >= 1, f"no step was served from a schedule: {s}"
    assert s["fallback"] == 0, f"a served step had to be repeated on the normal driver: {s}"


def test_the_default_is_the_normal_driver(make_llm):
    """Unset, the engine takes the normal driver unless the environment (`OPENTSP_VLLM_REPLAY`) switches the replay on."""
    e = make_llm("iss", "test")
    from_env = os.environ.get("OPENTSP_VLLM_REPLAY", "").strip().lower() in ("1", "true", "yes", "on")
    assert stats(e)["enabled"] is from_env
    run(e)
    if not from_env:
        assert stats(e)["served"] == 0
