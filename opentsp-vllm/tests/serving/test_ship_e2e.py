"""Numerical parity, prefix reuse and preemption on a two-layer GPT-2 test fixture."""

import os

import numpy as np
import pytest

from opentsp_vllm.ship_chip import DEFAULT_CHECKPOINT, _showcase

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(DEFAULT_CHECKPOINT, "model.safetensors")), reason="no GPT-2 small checkpoint")

LAYERS = 2
PACKAGE = os.environ.get("OPENTSP_SHIP_PACKAGE")
PIPELINE_KWARGS = {"cache": PACKAGE} if PACKAGE else {}
PROMPTS = [
    [464, 3139, 286, 4881, 318],  # " The capital of France is"-like: five token ids
    [7454, 2402, 257, 640, 612, 373, 257, 922, 1110, 329, 257, 3626, 287, 262],  # 14 tokens: crosses the first page of 16 once decoded
]


def make_llm(root, name, **extra):
    from vllm import LLM
    import vllm_helpers

    path = vllm_helpers.write_model_dir(root / name, layers=LAYERS, vocab=50257, positions=1024)
    return LLM(
        model=str(path),
        skip_tokenizer_init=True,
        dtype="float32",
        **{"max_model_len": 1024, **extra},
        max_num_seqs=4,
        max_num_batched_tokens=8,
        enable_chunked_prefill=True,
        additional_config={"opentsp": dict(chip="ship", check=True, ras_domain=str(path), weights=DEFAULT_CHECKPOINT, programs=PACKAGE, layers=LAYERS)},
    )


@pytest.fixture(scope="module")
def llm(tmp_path_factory):
    e = make_llm(tmp_path_factory.mktemp("ship"), "model")
    yield e
    e.llm_engine.engine_core.shutdown()


@pytest.fixture(scope="module")
def stand_alone():
    """The stand-alone form: `Pipeline` of the same 2 layers, each prompt alone in slots 0 .. through the lower-level pipeline."""
    gpt2s_weights, ship_pipeline, cfg = _showcase()
    w = gpt2s_weights.load_checkpoint(cfg, os.path.join(DEFAULT_CHECKPOINT, "model.safetensors"))

    def generate(prompt, n):
        pipe = ship_pipeline.Pipeline(w, LAYERS, **PIPELINE_KWARGS)
        toks, new = list(prompt), []
        for pos in range(len(prompt) + n - 1):
            out, _ = pipe.step(toks[pos], pos, check=False)
            if pos >= len(prompt) - 1:
                nxt = int(np.argmax(pipe.logits(out)))
                toks.append(nxt)
                new.append(nxt)
        return new

    return generate


def gen(llm, prompts, n):
    from vllm import SamplingParams

    out = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True), use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in out]


def test_one_request_gives_the_stand_alone_tokens(llm, stand_alone):
    assert gen(llm, [PROMPTS[0]], 4) == [stand_alone(PROMPTS[0], 4)]


def test_two_requests_at_once_each_give_their_stand_alone_tokens(llm, stand_alone):
    got = gen(llm, PROMPTS, 5)  # 14 + 4 runs and 5 + 4: the second request's 15th to 17th positions are in its second page
    assert got == [stand_alone(p, 5) for p in PROMPTS]


def test_a_prefix_hit_reads_the_cached_blocks_from_the_pools_and_gives_the_stand_alone_tokens(llm, stand_alone):
    """A request whose first 16 tokens (one full block) are another request's gets that block from vLLM's prefix cache: the 16 positions are not run again, their K and V
    rows are read from the pools where the first request left them (resident, no copy), and the tokens are the stand-alone tokens of the whole prompt."""
    from vllm import SamplingParams

    first = PROMPTS[1] + [262, 3139, 286, 262, 4141, 2066]  # 20 tokens
    second = first[:16] + [2576, 508, 373, 257]  # the same block, a different tail
    params = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    a = llm.generate([{"prompt_token_ids": first}], params, use_tqdm=False)[0]
    b = llm.generate([{"prompt_token_ids": second}], params, use_tqdm=False)[0]
    assert a.num_cached_tokens in (0, None) and b.num_cached_tokens == 16
    assert list(a.outputs[0].token_ids) == stand_alone(first, 4) and list(b.outputs[0].token_ids) == stand_alone(second, 4)


def test_a_preempted_request_is_recomputed_into_other_blocks_and_gives_the_stand_alone_tokens(tmp_path_factory, stand_alone):
    """3 usable blocks (48 slots) for two requests of 34 tokens each (3 blocks): vLLM preempts one, frees its blocks (not zeroed) and recomputes it into blocks the other
    request left; both give their stand-alone tokens, and the engine reports the preemption."""
    from vllm import SamplingParams

    e = make_llm(tmp_path_factory.mktemp("ship-small"), "model", max_model_len=48, num_gpu_blocks_override=4, enable_prefix_caching=False, disable_log_stats=False)
    try:
        got = e.generate([{"prompt_token_ids": p} for p in PROMPTS], SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True), use_tqdm=False)
        preempted = sum(m.value for m in e.get_metrics() if m.name == "vllm:num_preemptions")
        assert preempted >= 1, "the scenario must preempt"
        assert [list(o.outputs[0].token_ids) for o in got] == [stand_alone(p, 20) for p in PROMPTS]
    finally:
        e.llm_engine.engine_core.shutdown()


def test_a_resident_system_gives_the_logits_and_cycles_of_a_new_system_per_step(monkeypatch):
    """`Pipeline(resident=True)` (one System, `restart` between steps, pools on the chips) against `resident=False` (a new System per step, pools through the host): the same
    steps, two interleaved sequences in scattered slots, bit-identical logits and equal cycle counts, with far fewer words written by the host."""
    from opentsp_vllm import ship_plan

    gpt2s_weights, ship_pipeline, cfg = _showcase()
    w = gpt2s_weights.load_checkpoint(cfg, os.path.join(DEFAULT_CHECKPOINT, "model.safetensors"))
    cold, warm = ship_pipeline.Pipeline(w, LAYERS, **PIPELINE_KWARGS), ship_pipeline.Pipeline(w, LAYERS, resident=True, **PIPELINE_KWARGS)
    from opentsp_vllm.system import System
    native_write = System.write_words
    aperture_bytes = []
    def counted_write(system, chip, slice_id, address, blob):
        aperture_bytes.append(len(blob))
        return native_write(system, chip, slice_id, address, blob)
    monkeypatch.setattr(System, "write_words", counted_write)
    seqs = {"a": ([464, 3139, 286], [9, 3]), "b": ([7454, 2402], [40, 1])}  # token ids, block table
    steps = [("a", 0), ("b", 0), ("a", 1), ("b", 1), ("a", 2), ("a", 0)]  # the last step rewrites position 0 of a: same slot, same row
    for who, pos in steps:
        toks, blocks = seqs[who]
        slot, counts = ship_plan.slot_of(blocks, pos), ship_plan.encoding(blocks, pos).reshape(-1, ship_plan.PAGES_PER_CHUNK)
        aperture_bytes.clear()
        out_c, st_c = cold.step(toks[pos], pos, check=True, slot=slot, counts=counts)
        assert st_c["write_words"] * 320 == sum(aperture_bytes)
        aperture_bytes.clear()
        out_w, st_w = warm.step(toks[pos], pos, check=True, slot=slot, counts=counts)
        assert st_w["write_words"] * 320 == sum(aperture_bytes)
        assert np.array_equal(out_c.view(np.uint32), out_w.view(np.uint32)), (who, pos)
        assert st_c["cycles"] == st_w["cycles"]
        assert st_w["write_words"] <= st_c["write_words"]
    for stat in (*warm.stats, *cold.stats):
        assert stat["write_words"] == stat["initial_write_words"] + stat["control_write_words"] + stat["kv_write_words"]
        assert stat["control_write_words"] == LAYERS * 81
    assert warm.stats[1]["initial_write_words"] == warm.stats[1]["kv_write_words"] == warm.stats[1]["kv_read_words"] == warm.stats[1]["object_load_bytes"] == 0
    assert cold.stats[1]["kv_write_words"] == cold.stats[1]["kv_read_words"] > 0
    assert cold.stats[1]["object_load_bytes"] > 0
    receipt = os.environ.get("OPENTSP_TRAFFIC_RECEIPT")
    if receipt:
        import json
        from pathlib import Path
        Path(receipt).write_text(json.dumps({"layers": LAYERS, "cold": list(cold.stats), "warm": list(warm.stats)}, indent=2))
    assert 0 < warm.stats[1]["write_words"] < cold.stats[1]["write_words"]
    # Both load the same stage images. The cold path additionally restores
    # the host-carried KV pool, including its all-zero first-step state.
    assert warm.stats[0]["initial_write_words"] == cold.stats[0]["initial_write_words"] > 0
    assert cold.stats[0]["write_words"] == warm.stats[0]["write_words"] + cold.stats[0]["kv_write_words"]
    assert warm.stats[0]["kv_write_words"] == 0 < cold.stats[0]["kv_write_words"]
