"""Tests of one transformer block (layer 0, one token, a preloaded cache at pos 0, 15, 40, 62).

The expectation of every dumped word is bit-exact reference's layer step (`reference_model.layer_step`), computed before and independently of
the emitter; the program itself runs through `tests/programs/run.py --tag ts-block` (`just showcase-block`). These
tests pin the expectation, check that the case dumps exactly what bit-exact reference says the chip writes, and that every queue of the
program fits the paged front end.
"""
import hashlib

import numpy as np
import pytest


from opentsp_tools.programs import data
from opentsp_tools import profile as tprofile
from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import reference as exact_ref

POSITIONS = (0, 15, 40, 62)
PINNED = {}      # position -> sha256 of the expected words, filled by test_expected_words_are_pinned


def digest(words):
    h = hashlib.sha256()
    for key in sorted(words):
        h.update(key.encode() + words[key])
    return h.hexdigest()


@pytest.fixture(scope="module")
def p():
    return tprofile.load("ref")


@pytest.mark.parametrize("pos", POSITIONS)
def test_expectation_is_reference_layer_step(pos):
    """The words the block must write: x out (4 planes), the current K row (2 planes) and V row (4 planes)."""
    inp = eb.block_inputs(pos)
    out, k_cur, v_cur = exact_ref.layer_step(inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask)
    want = eb.expected_words(inp)
    for name, arr in (("x1", out), ("k", k_cur), ("v", v_cur)):
        got = want[name]
        planes = data.to_planes(np.pad(arr, (0, 320 - len(arr))))
        assert [bytes(g) for g in got] == [bytes(pl) for pl in planes], name
    assert np.isfinite(out).all()


@pytest.mark.parametrize("pos", POSITIONS)
def test_expected_words_are_pinned(pos):
    inp = eb.block_inputs(pos)
    w = eb.expected_words(inp)
    flat = {f"{n}{i}": bytes(b) for n, planes in w.items() for i, b in enumerate(planes)}
    assert digest(flat) == eb.PINNED[pos]


def test_cache_inputs_cover_exactly_the_cached_positions():
    for pos in POSITIONS:
        inp = eb.block_inputs(pos)
        cfg = inp.cfg
        assert inp.k_slots.shape == (cfg.slots, cfg.hidden) and inp.k_slots.dtype == np.float16
        assert inp.v_slots.shape == (cfg.slots, cfg.hidden) and inp.v_slots.dtype == np.float32
        assert np.all(inp.k_slots[pos:] == 0) and np.all(inp.v_slots[pos:] == 0)
        if pos:
            assert np.any(inp.k_slots[:pos] != 0)
        live = np.flatnonzero(inp.mask == 0)
        assert list(live) == list(range(pos)) + [cfg.slots - 1]
        assert np.all(np.isneginf(inp.mask[np.setdiff1d(np.arange(cfg.slots), live)]))


@pytest.mark.parametrize("pos", (0, 62))
def test_case_dumps_what_reference_expects(p, pos):
    """The generated case's expected words are bit-exact reference's, and the weights it binds are the layer's synthetic weights."""
    text, case = eb.build(p, pos)
    inp = eb.block_inputs(pos)
    want = eb.expected_words(inp)
    n = sum(len(v) for v in want.values())
    assert len(case["dump"]) >= n
    assert len(case["expect"]["words"]) == len(case["dump"])
    assert case["expect"]["diagnostics"] == [] and case["expect"]["lints"] == []
    assert case["tags"] == ["ts-block", "sim"]
    names = set(case["weights"])
    assert {f"w0s{s:02d}" for s in list(range(16)) + list(range(44, 60))} <= names      # layer 0's matrices, every plane
    assert {f"vc0s{s}" for s in range(20, 24)} <= names and {f"kc0s{s}" for s in range(44, 60)} <= names
    assert not any(n.startswith(("w1s", "kc1s", "vc1s")) for n in names)               # only layer 0 is bound
    assert "OTS-E" not in text


def test_every_queue_fits_or_is_paged(p):
    """After the pager no initial image exceeds icu.iq_depth, and the permuters (SXM.W.PRM) are among the paged queues."""
    from opentsp_tools import pager
    text, case = eb.build(p, 0)
    assert 124 in case["golden"]["paged_queues"]
    assert case["expect"]["lints"] == []
    cap = pager.Geometry(p).cap
    units = case["golden"]["max_initial_units"]
    assert units <= cap


@pytest.mark.parametrize("pos", POSITIONS)
def test_reference_layer_is_within_1e_2_of_the_float64_layer(pos):
    """The sanity reference of the layer step (no rounding at all, weights as stored): bit-exact reference stays within 1e-2 of the
    largest output lane, so a wrong bit-exact reference expectation cannot hide behind the bit-exact comparison."""
    from opentsp_vllm.models.tinystories import torch_reference
    inp = eb.block_inputs(pos)
    args = (inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask)
    out0 = exact_ref.layer_step(*args)[0].astype(np.float64)
    ref64 = torch_reference.layer_step_f64(*args)
    assert np.max(np.abs(out0 - ref64)) <= 1e-2 * np.max(np.abs(ref64))


def test_rounded_reference_script_skips_with_a_message_when_the_checkpoint_is_absent(capsys):
    """Without PyTorch or the checkpoint the rounded PyTorch reference script prints which one is missing and exits 0 (never fails `just test`)."""
    import check_block
    assert check_block.main(["--checkpoint", "/nonexistent/no-such-checkpoint"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("SKIP check_block:") and ("PyTorch" in out or "no-such-checkpoint" in out)


def test_case_expects_references_ten_written_planes_at_the_right_addresses(p):
    """The planes bit-exact reference says the layer writes (x out 4, the K row 2, the V row 4) are in the case's expected set: the K row
    on the slices of slot 63 of the K cache, the V row at slot 63 of the V cache, x on four consecutive slices at one
    word."""
    from opentsp_vllm.models.tinystories import layout
    text, case = eb.build(p, 0)
    inp = eb.block_inputs(0)
    want = eb.expected_words(inp)
    plan = layout.plan_tinystories(inp.cfg, layout.load_profile("ref"))
    words = case["expect"]["words"]
    kc, vc = plan.by_name["kc0"].addr, plan.by_name["vc0"].addr
    S = inp.cfg.slots
    for j, plane in enumerate(want["k"]):
        assert words[f"{layout.EAST_W[2 * ((S - 1) % 8) + j]}:{kc + (S - 1) // 8}"] == bytes(plane).hex()
    for j, plane in enumerate(want["v"]):
        assert words[f"{layout.VCACHE_QUAD[j]}:{vc + S - 1}"] == bytes(plane).hex()
    where = [[k for k, h in words.items() if h == bytes(plane).hex()] for plane in want["x1"]]
    assert all(len(w) >= 1 for w in where)
    addrs = [{k.split(":")[1] for k in w} for w in where]
    common = set.intersection(*addrs)
    assert common, "the four planes of x are not at one word address"
    a = sorted(common)[0]
    ss = [next(int(k.split(":")[0]) for k in w if k.split(":")[1] == a) for w in where]
    assert ss == list(range(ss[0], ss[0] + 4)) and ss[0] % 4 == 0, ss


def test_planted_defect_in_the_expectation_fails_on_the_iss(p):
    """A control: the case passes on the ISS legs; with one bit of one expected word inverted run.py reports it."""
    import case_runner as runner_util
    text, case = eb.build(p, 0)
    key = next(k for k in case["expect"]["words"])
    rc, out = runner_util.run_on_iss(text, case)
    assert rc == 0, out
    rc, out = runner_util.run_on_iss(text, case, flip=key)
    assert rc == 1 and "FAIL" in out and f"word {key} lane 0" in out, out


def test_rounded_reference_result_line_never_says_pass_without_the_pytorch_comparison():
    from opentsp_vllm.models.tinystories import comparison as comparison
    assert comparison.verdict(True, None) == "PASS"
    assert comparison.verdict(True, "PyTorch is not installed").startswith("SKIP-PARTIAL (bit-exact reference only passed)")
    assert "PASS" not in comparison.verdict(True, "PyTorch is not installed")
    assert comparison.verdict(False, "PyTorch is not installed") == "FAIL"
