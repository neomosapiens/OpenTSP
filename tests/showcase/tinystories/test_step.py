"""Tests of one decode step (token and position embedding, eight layers, final LayerNorm, the output head).

The expectation of every dumped word is bit-exact reference (`reference_model.decode_step_slots` and the f16 output head), computed before and
independently of the emitter; the program itself runs through `tests/programs/run.py` (`just showcase-block STEP=1`).
These tests pin the expectation, check what the case dumps and binds, that the program text does not depend on the
position, and the parametrisation the serving emitters use.
"""
import functools
import hashlib

import numpy as np
import pytest


from opentsp_tools.programs import data
from opentsp_tools import profile as tprofile
from opentsp_tools.programs import ref
from opentsp_vllm.models.tinystories import emit_step as es
from opentsp_vllm.models.tinystories import reference as exact_ref


def digest(words):
    h = hashlib.sha256()
    for key in sorted(words):
        h.update(key.encode() + words[key])
    return h.hexdigest()


@pytest.fixture(scope="module")
def p():
    return tprofile.load("ref")


@functools.lru_cache(maxsize=None)
def built(pos):
    return es.build(tprofile.load("ref"), pos)


def body(text):
    """The program without its leading comment (which names the case)."""
    return "\n".join(ln for ln in text.split("\n") if not ln.startswith(";"))


@pytest.mark.parametrize("pos", es.POSITIONS)
def test_expectation_is_reference_step(pos):
    """What the chip must write: every layer's output x, the current K and V rows of every layer, the final hidden
    state and the logits of the head window, each equal to bit-exact reference's decode step."""
    inp = es.step_inputs(pos)
    res = exact_ref.decode_step_slots(inp.w, inp.token, pos, inp.k_slots, inp.v_slots, inp.mask, logits=False, trace={})
    want = es.expected_words(inp)
    pad = es.pad
    assert [bytes(b) for b in want["hidden"]] == [bytes(b) for b in data.to_planes(pad(res.hidden))]
    for l in range(inp.cfg.layers):
        assert [bytes(b) for b in want[f"k{l}"]] == [bytes(b) for b in data.to_planes(pad(res.k_cur[l]))], l
        assert [bytes(b) for b in want[f"v{l}"]] == [bytes(b) for b in data.to_planes(pad(res.v_cur[l]))], l
        got = data.to_planes(pad(res.trace[f"x{l + 1}"]))
        assert [bytes(b) for b in want[f"x{l + 1}"]] == [bytes(b) for b in got], l
    h16 = ref.to_f16(res.hidden)
    rows = inp.w.wte[inp.head_lo:inp.head_lo + es.CHUNK * inp.head_chunks]
    logits = np.concatenate([es.nx.dot_f16(rows[i:i + es.CHUNK], h16, ref.LANES)
                             for i in range(0, len(rows), es.CHUNK)])
    for c in range(inp.head_chunks):
        chunk = logits[c * es.CHUNK:(c + 1) * es.CHUNK]
        assert [bytes(b) for b in want[f"logits{c}"]] == [bytes(b) for b in data.to_planes(es.pad(chunk))], c
    assert np.isfinite(res.hidden).all() and np.isfinite(logits).all()


@pytest.mark.parametrize("pos", es.POSITIONS)
def test_expected_words_are_pinned(pos):
    inp = es.step_inputs(pos)
    w = es.expected_words(inp)
    flat = {f"{n}.{i}": bytes(b) for n, planes in w.items() for i, b in enumerate(planes)}
    assert digest(flat) == es.PINNED[pos]


def test_head_window_holds_the_argmax_token():
    """bit-exact reference's token for the step: the greedy token of the host head (f64 sum) is the argmax of the full vocabulary of
    the f16 head the chip computes, and lies in the window of rows the program installs."""
    for pos in es.POSITIONS:
        inp = es.step_inputs(pos)
        res = exact_ref.decode_step_slots(inp.w, inp.token, pos, inp.k_slots, inp.v_slots, inp.mask)
        token = int(np.argmax(res.logits))
        assert token == es.head_argmax(inp.w, res.hidden), pos
        assert inp.head_lo <= token < inp.head_lo + es.CHUNK * inp.head_chunks, pos
        assert inp.argmax == token


def test_prefilled_cache_is_reference_history():
    """The position-5 case's cache is what bit-exact reference's own steps wrote for the prompt tokens 0..4."""
    inp = es.step_inputs(5)
    kv = exact_ref.HostKV(inp.cfg)
    for pos in range(5):
        exact_ref.decode_step(inp.w, kv, es.PROMPT_IDS[pos], pos, logits=False)
    assert np.array_equal(inp.k_slots[:, :5], kv.k) and np.array_equal(inp.v_slots[:, :5], kv.v)
    assert np.all(inp.k_slots[:, 5:] == 0) and np.all(inp.v_slots[:, 5:] == 0)
    assert inp.token == es.PROMPT_IDS[5]


def test_random_cache_covers_exactly_the_cached_positions():
    inp = es.step_inputs(62)
    assert inp.k_slots.shape == (8, 64, 64) and inp.v_slots.dtype == np.float32
    live = np.flatnonzero(inp.mask == 0)
    assert list(live) == list(range(62)) + [63]
    assert np.any(inp.k_slots[:, :62] != 0) and np.all(inp.k_slots[:, 62:] == 0)


def test_case_dumps_what_reference_expects_and_binds_all_layers(p):
    text, case = built(5)
    inp = es.step_inputs(5)
    want = es.expected_words(inp)
    assert len(case["dump"]) == sum(len(v) for v in want.values()) and len(set(case["dump"])) == len(case["dump"])
    assert len(case["expect"]["words"]) == len(case["dump"])
    assert case["expect"]["diagnostics"] == [] and case["expect"]["lints"] == []
    assert case["tags"] == ["ts-step", "sim"]
    names = set(case["weights"])
    for l in range(8):
        assert {f"w{l}s{s:02d}" for s in list(range(16)) + list(range(44, 60))} <= names, l
        assert {f"vc{l}s{s}" for s in range(20, 24)} <= names and {f"kc{l}s{s}" for s in range(44, 60)} <= names
    assert "OTS-E" not in text
    assert case["golden"]["head_rows"] == es.CHUNK * inp.head_chunks and case["golden"]["layers"] == 8


def test_program_text_is_the_same_at_every_position(p):
    """Position and cache contents enter as data only (x0, the mask, the K/V slots, the head rows): one program serves
    every position below the cache length."""
    texts = {pos: body(built(pos)[0]) for pos in es.POSITIONS}
    assert len(set(texts.values())) == 1
    assert hashlib.sha256(texts[0].encode()).hexdigest() == es.PINNED_TEXT


def test_only_data_differs_between_positions(p):
    _, a = built(0)
    _, b = built(62)
    differ = {n for n in a["weights"] if a["weights"][n] != b["weights"][n]}
    data_names = ("x0", "mask", "kc", "vc", "hd")
    assert differ and all(n.startswith(data_names) for n in differ), sorted(differ)[:10]


def test_emitter_weights_are_the_export_packages_tensors():
    """The host driver binds the layer weights from export.py's package: the bytes the emitter's case carries for every
    `w<layer>s<slice>` extern are the package's tensors, and the extern's slice, address and size are the plan's."""
    from opentsp_vllm.models.tinystories import export
    text, case = built(5)
    inp = es.step_inputs(5)
    prof, plan, tensors, objects = export.build(inp.cfg, inp.w)
    names = [n for n in tensors if n.startswith("w") and n[1].isdigit()]
    assert len(names) == 8 * 32
    for n in names:
        assert bytes.fromhex(case["weights"][n]["hex"]) == b"".join(tensors[n][1]), n
        r = plan.by_name[n]
        assert f".extern {n}, slice={r.slices[0]}, addr={r.addr:#x}, size={r.words:#x}" in text, n


def test_planted_defect_in_the_expectation_fails_on_the_iss():
    """A control on the step: the case passes on the ISS legs; with one bit of an expected word inverted, run.py
    reports it."""
    import case_runner as runner_util
    text, case = built(5)
    key = next(iter(case["expect"]["words"]))
    rc, out = runner_util.run_on_iss(text, case)
    assert rc == 0, out
    rc, out = runner_util.run_on_iss(text, case, flip=key)
    assert rc == 1 and f"word {key} lane 0" in out, out


def test_the_largest_head_runs_reference_exact_on_the_iss():
    """MAX_HEAD_CHUNKS (48 chunks, 7 680 vocabulary rows) is a limit that was run: the step builds, and the ISS and the
    native ISS equal bit-exact reference on every dumped word. Beyond it the emitter refuses (a head of 100 chunks runs: test_layout.py)."""
    import case_runner as runner_util
    from opentsp_tools import profile
    text, case = es.build(profile.load("ref"), 5, head_chunks=es.MAX_HEAD_CHUNKS)
    assert case["golden"]["head_rows"] == es.CHUNK * es.MAX_HEAD_CHUNKS
    rc, out = runner_util.run_on_iss(text, case)
    assert rc == 0, out
    with pytest.raises(ValueError):
        es.Step(profile.load("ref"), es.step_inputs(5, head_chunks=es.MAX_HEAD_CHUNKS + 1))


def test_every_queue_fits_or_is_paged(p):
    from opentsp_tools import pager
    text, case = built(0)
    assert case["expect"]["lints"] == [] and 124 in case["golden"]["paged_queues"]
    assert case["golden"]["max_initial_units"] <= pager.Geometry(p).cap


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
