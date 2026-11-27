"""Compare the simulated OpenTSP backend with vLLM's CPU backend.

The reference checkpoint is a Hugging Face GPT-2 whose fp32 weights are the f16-rounded TSP weights (`just vllm-ref-checkpoint`,
`tools/vllm_ref_checkpoint.py`). The CPU backend runs in a subprocess (`cpu_backend_run.py`: a process has one platform), with its
attention kernel replaced by plain torch because that kernel does not support the model's head size of 4 (see there).

Two comparisons per request, over the synthetic weights of the test model (2 layers) and of the full model (8 layers):
- greedy tokens equal, except from the first step where the CPU backend's top-1/top-2 margin is below 1e-3 (the FP32 reference rule of the ports
  ): the streams may part there, and the margin is the explanation;
- per step, on the chip's own tokens (the CPU backend scores prompt + the chip's tokens so far), the log-softmax of the chip's logits
  against the CPU backend's: cosine of the mean-centred vectors and the largest absolute difference. The CPU backend does not round
  activations to f16 and sums in another order, so they are close, not equal; the thresholds are calibrated on the measurements the
  results note records (``).
"""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
from vllm import SamplingParams

import sampling_helpers as H

from opentsp_vllm.models.tinystories import layout

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import vllm_ref_checkpoint as ref

CHIPS = ("exact", "iss")
N = 8
PROMPTS = [H.prompt_ids(70 + i, n, layout.serve_config(2).vocab) for i, n in enumerate((6, 12, 20))]
MARGIN = 1e-3
# calibrated on the measurements recorded in the results note (3 requests of 8 steps, 2 and 8 layers): the worst cosine of the centred
# log-softmax vectors was 1 - 1.1e-7 (2 layers) and 1 - 2.0e-7 (8 layers), the worst absolute log-probability difference 3.3e-3 and
# 4.6e-3 (the chip rounds the hidden state to f16 before the head). The cosine threshold is the design's; the difference bound is
# three times the worst measured.
COSINE = 0.9999
MAX_ABS_LOGPROB = 1.5e-2


def centred(x):
    x = np.asarray(x, np.float64)
    return x - x.mean()


def compare_rows(chip_logits, cpu_rows):
    """(worst cosine of the centred log-softmax vectors, worst absolute log-probability difference) over the rows; non-finite or empty input fails.

    `min` and `max` against a running value skip a NaN, so the arrays are checked before they are accumulated."""
    chip_logits, cpu_rows = np.asarray(chip_logits, np.float64), np.asarray(cpu_rows, np.float64)
    assert chip_logits.ndim == 2 and chip_logits.shape == cpu_rows.shape and len(cpu_rows) > 0, (chip_logits.shape, cpu_rows.shape)
    assert np.isfinite(chip_logits).all(), "the chip's logits contain non-finite values"
    assert np.isfinite(cpu_rows).all(), "the CPU backend's log-probabilities contain non-finite values"
    cos, ab = 1.0, 0.0
    for lg, cpu_lp in zip(chip_logits, cpu_rows):
        chip_lp = lg - np.logaddexp.reduce(lg)
        norms = np.linalg.norm(centred(chip_lp)) * np.linalg.norm(centred(cpu_lp))
        assert norms > 0, "a score row has a zero centred norm"
        cos = min(cos, float(centred(chip_lp) @ centred(cpu_lp) / norms))
        ab = max(ab, float(np.abs(chip_lp - cpu_lp).max()))
    assert np.isfinite(cos) and np.isfinite(ab), (cos, ab)
    return cos, ab


def test_compare_rows_rejects_non_finite_or_empty_scores():
    rng = np.random.default_rng(0)
    lg = rng.normal(size=(3, 16))
    good = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
    cos, ab = compare_rows(lg, good)
    assert cos > 0.999999 and ab < 1e-9
    bad = lg.copy()
    bad[1, 3] = np.nan
    for chip, cpu in ((bad, good), (np.full_like(lg, np.nan), good), (lg, np.where(good > -2, np.nan, good)), (lg[:0], good[:0]), (np.zeros((3, 16)), good)):
        with pytest.raises(AssertionError):
            compare_rows(chip, cpu)


def cpu_backend(tmp_path, layers, requests):
    ckpt = ref.write(tmp_path / f"ref-l{layers}", ref.load_weights("synthetic:1", layers))
    req, out = tmp_path / "requests.json", tmp_path / "cpu.npz"
    req.write_text(json.dumps(requests))
    env = os.environ.copy()
    env.setdefault("TRITON_CACHE_DIR", str(tmp_path / "triton-cache"))
    run = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).with_name("cpu_backend_run.py")), str(ckpt), str(req), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    z = np.load(out)
    return z["tokens"], z["logprobs"]


@pytest.mark.parametrize("chip", CHIPS)
@pytest.mark.parametrize("model", ["test", "full"])
def test_chip_logits_and_greedy_tokens_agree_with_the_cpu_backend(make_llm, tmp_path, chip, model):
    e = make_llm(chip, model)
    greedy = SamplingParams(temperature=0.0, max_tokens=N, ignore_eos=True)
    solo = [e.solo(p, greedy) for p in PROMPTS]  # (tokens, logits (N, vocab)) of each request alone
    forced = [p + t[:k] for p, (t, _) in zip(PROMPTS, solo) for k in range(N)]
    cpu_tokens, cpu_logprobs = cpu_backend(
        tmp_path, MODELS_LAYERS[model], {"greedy": PROMPTS, "n": N, "forced": forced}
    )
    worst_cos, worst_abs, parted, evaluated = 1.0, 0.0, [], 0
    for i, (p, (tokens, logits)) in enumerate(zip(PROMPTS, solo)):
        rows = cpu_logprobs[i * N : (i + 1) * N]
        cos, ab = compare_rows(logits, rows)
        worst_cos, worst_abs, evaluated = min(worst_cos, cos), max(worst_abs, ab), evaluated + len(rows)
        mine, theirs = list(tokens), list(cpu_tokens[i])
        first = next((k for k in range(N) if mine[k] != theirs[k]), None)
        if first is not None:
            top2 = np.sort(rows[first])[-2:]  # the CPU backend's scores at that step, on the chip's own prefix
            margin = float(top2[1] - top2[0])
            assert margin < MARGIN, f"request {i}: the streams part at step {first} with the CPU margin {margin:.3e} >= {MARGIN}"
            parted.append((i, first, margin))
    assert evaluated == len(PROMPTS) * N, f"{evaluated} rows evaluated, expected {len(PROMPTS) * N}"
    record = {"model": model, "chip": chip, "worst_cosine": worst_cos, "worst_abs_logprob": worst_abs, "parted": parted, "rows": evaluated}
    d = os.environ.get("OPENTSP_SERVING_RECORDS")
    if d:
        os.makedirs(d, exist_ok=True)
        pathlib.Path(d, f"{chip}-cpu-backend-{model}.json").write_text(json.dumps(record, indent=1))
    print("CPU-BACKEND", record)
    assert worst_cos >= COSINE, record
    assert worst_abs <= MAX_ABS_LOGPROB, record


MODELS_LAYERS = {"test": 2, "full": 8}


def test_the_reference_checkpoint_maps_back_to_the_weights_it_was_made_from():
    for layers in (2, 8):
        w = ref.load_weights("synthetic:1", layers)
        sd = ref.state_dict(w)
        assert H.W.from_gpt2_state_dict(w.cfg, sd).digest() == w.digest()
        assert sd["transformer.h.0.attn.c_attn.weight"].shape == (64, 192) and sd["transformer.h.0.mlp.c_fc.weight"].shape == (64, 256)
