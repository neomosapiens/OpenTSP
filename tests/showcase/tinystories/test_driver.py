"""Tests of the generation driver.

The driver runs the decode step on the ISS once per token (prefill and generation) with the K/V cache and the
embedding kept by the host. Its token sequence and every step's words must equal bit-exact reference's (`reference_model.generate`). The
tests below are the RED test of the (`driver.py --weights synthetic --tokens 4` against bit-exact reference's sequence) and
the pins of what the driver relies on: its per-step tensors equal what the emitter binds, the refusal beyond the cache.
"""
import argparse
import os
import subprocess
import sys

import pytest

from opentsp_vllm.models.tinystories import paths as _paths

from opentsp_tools import profile as tprofile
from opentsp_vllm.models.tinystories import driver
from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import emit_step as es
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories.config import PROMPT, PROMPT_IDS, TINYSTORIES
from opentsp_vllm.models.tinystories import weights as W

HERE = _paths.HERE
DRIVER = os.path.join(HERE, "driver.py")


def run_driver(*args, timeout=900):
    return subprocess.run([sys.executable, DRIVER, *args], capture_output=True, text=True, timeout=timeout)


def ids_of(out, tag="IDS"):
    line = next(ln for ln in out.splitlines() if ln.startswith(tag + " "))
    return [int(x) for x in line.split()[1:]]


def test_four_tokens_equal_reference():
    """The RED test: the driver's ids for the fixed prompt and 4 generated tokens on the synthetic weights equal
    bit-exact reference's."""
    want = exact_ref.generate(W.synthetic(TINYSTORIES, 1), PROMPT_IDS, 4).tokens
    r = run_driver("--weights", "synthetic", "--tokens", "4")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert ids_of(r.stdout) == want
    assert len(want) == len(PROMPT_IDS) + 4


@pytest.fixture(scope="module")
def prog(tmp_path_factory):
    return driver.StepProgram(eb.weights(1), str(tmp_path_factory.mktemp("ts4")))


def test_per_step_tensors_are_what_the_emitter_binds(prog):
    """The host rewrites x0, the mask and every layer's K and V slots itself: at a position with a real cache (5, bit-exact reference's
    own steps) the bytes equal those of the case the emitter builds for that position, and nothing else of the package
    differs from position 0 but the head window `hd*` (the rows around that position's token, which the host head does
    not use)."""
    inp = es.step_inputs(5)
    _, case5 = es.build(tprofile.load("ref"), 5)
    mine = prog.dynamic(inp.x0, inp.k_slots, inp.v_slots, inp.mask)
    assert set(mine) == set(prog.dynamic_names)
    for n, b in mine.items():
        assert b == bytes.fromhex(case5["weights"][n]["hex"]), n
    static = [n for n in case5["weights"] if n not in mine and not n.startswith("hd")]
    assert static and all(case5["weights"][n] == prog.case["weights"][n] for n in static)


def test_a_wrong_dumped_word_fails_the_reference_check(prog):
    """A control: the check compares every dumped word, so one inverted bit in the hidden state, or in a layer's K or V
    row, is reported with its name, and an unmodified step passes."""
    w = eb.weights(1)
    real_step = prog.step

    def flipping(name, plane=0):
        def step(*a):
            words, cycles, tm = real_step(*a)
            b = bytearray(words[name][plane])
            b[3] ^= 1
            words[name][plane] = bytes(b)
            return words, cycles, tm
        return step

    out = driver.generate(w, PROMPT_IDS[:1], 1, prog, {"exact"}, say=lambda *_: None)
    assert out["rows"][0]["cycles"] == 84736
    try:
        for name in ("hidden", "k3", "v7", "x5", "logits2"):
            prog.step = flipping(name)
            with pytest.raises(driver.DriverError, match=f"bit-exact reference: {name}"):
                driver.generate(w, PROMPT_IDS[:1], 1, prog, {"exact"}, say=lambda *_: None)
    finally:
        prog.step = real_step


def test_stream_prints_the_generated_text_as_it_is_produced():
    """`--stream` prints the prompt and every generated token's text between the markers, and what it printed is the
    final TEXT line (or, without a tokenizer, the bracketed ids of the NEW line)."""
    import ast
    r = run_driver("--weights", "synthetic", "--ids", "7454,2402", "--tokens", "3", "--stream")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    streamed = r.stdout.split("--- generation ---\n", 1)[1].split("\n--- end ---", 1)[0]
    assert "pos  token" not in r.stdout                        # the step table is replaced by the stream
    new = next(line for line in r.stdout.splitlines() if line.startswith("NEW ")).split()[1:]
    text_line = next((line for line in r.stdout.splitlines() if line.startswith("TEXT ")), None)
    if text_line is not None:
        assert streamed == ast.literal_eval(text_line[5:])
    else:
        assert streamed.split()[-3:] == [f"[{t}]" for t in new]
    assert len(new) == 3


def test_program_text_is_the_same_at_every_step_and_is_ts3s():
    r = run_driver("--weights", "synthetic", "--ids", "7454,2402", "--tokens", "2", "--check")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert f"text sha256 {es.PINNED_TEXT} (TS-3's pin: True)" in r.stdout
    assert "identical at all 3 steps" in r.stdout and "GATE bit-exact reference PASS: 3 steps" in r.stdout


def test_compat_and_native_interfaces_agree():
    outs = []
    for iface in ("native", "compat"):
        r = run_driver("--weights", "synthetic", "--ids", "7454", "--tokens", "1", "--check", "--interface", iface)
        assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
        outs.append(ids_of(r.stdout))
    assert outs[0] == outs[1] and len(outs[0]) == 2


def test_prompt_beyond_the_cache_is_refused_before_anything_runs():
    ids = ",".join(["7454"] * 60)
    r = run_driver("--weights", "synthetic", "--ids", ids, "--tokens", "10")
    assert r.returncode == 1 and "positions" in r.stdout and "at most 5 tokens" in r.stdout, r.stdout + r.stderr
    assert "program:" not in r.stdout
    w = eb.weights(1)
    with pytest.raises(driver.DriverError, match="at most 5 tokens"):
        driver.generate(w, [7454] * 60, 6, None)
    with pytest.raises(AttributeError):         # 60 + 5 - 1 = 64 positions fit: it gets as far as running a step
        driver.generate(w, [7454] * 60, 5, None)


def test_prompt_needs_a_tokenizer_or_ids():
    base = dict(ids=None, tokenizer="/nonexistent/tokenizer.json")
    ids, tok, note = driver.prompt_ids(argparse.Namespace(prompt=PROMPT, **base))
    assert ids == list(PROMPT_IDS) and tok is None and "known ids" in note
    with pytest.raises(driver.DriverError, match="--ids"):
        driver.prompt_ids(argparse.Namespace(prompt="Another prompt", **base))
    assert driver.prompt_ids(argparse.Namespace(prompt="x", ids="1,2,3", tokenizer=None))[0] == [1, 2, 3]


def test_the_gpt2_tokenizer_gives_the_design_prompt_ids():
    pytest.importorskip("tokenizers", reason="the optional `tokenizers` package is not installed")
    tok, why = driver.load_tokenizer()
    if tok is None:
        pytest.skip(why)
    assert list(tok.encode(PROMPT).ids) == list(PROMPT_IDS)
    assert tok.decode(list(PROMPT_IDS)) == PROMPT


def test_real_checkpoint_compares_exact_rounded_and_fp32():
    """Three tokens with the real checkpoint through bit-exact reference, rounded PyTorch reference and FP32 reference (a visible skip without the checkpoint, PyTorch or
    transformers, which are optional)."""
    from opentsp_vllm.models.tinystories import comparison as comparison_ref
    from opentsp_vllm.models.tinystories import driver_gates
    w, why = comparison_ref.load_weights()
    if w is None:
        pytest.skip(f"the checkpoint is not available: {why}")
    if comparison_ref.torch_missing():
        pytest.skip(comparison_ref.torch_missing())
    if driver_gates.fp32_model(comparison_ref.DEFAULT_CHECKPOINT)[0] is None:
        pytest.skip("the original fp32 model needs transformers and the cached checkpoint")
    r = run_driver("--weights", "real", "--tokens", "3", "--golden", "exact,rounded,fp32")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    for gate in ("bit-exact reference", "rounded PyTorch reference", "FP32 reference"):
        assert f"GATE {gate} PASS" in r.stdout
    assert ids_of(r.stdout)[6:] == [257, 1310, 2576]


def test_a_program_text_that_is_not_ts3s_pin_is_refused(monkeypatch, tmp_path):
    """The program must be the decode step's (its three-engine gate covers that text): a different digest is an error, not a
    printed `False`."""
    monkeypatch.setattr(es, "PINNED_TEXT", "0" * 64)
    with pytest.raises(driver.DriverError, match="TS-3's pinned text"):
        driver.StepProgram(eb.weights(1), str(tmp_path))
