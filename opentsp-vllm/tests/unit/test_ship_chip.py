"""`ShipChip` behind the real `TspModelRunner`, with a numpy stand-in for the pipeline: the runner, the step plan and the chip's lane loop are the real ones;
the stand-in plays the chips: it keeps the (token, position) of every pool slot a lane's Scatter slot names, and a lane's logits depend on the visible slots its mask
encoding lists, so any wrong slot, page count or lane order changes the tokens. The tokens are compared with the same stand-in run alone on a contiguous pool
(the stand-alone driver's form), for chunked prefill, interleaved requests on scattered blocks, block reuse and preemption by recompute."""

import numpy as np
import pytest

import vllm_helpers as H
from opentsp_vllm import ship_plan
from opentsp_vllm.chip import ChipResult, TspChipError
from opentsp_vllm.model_runner import TspModelRunner
from opentsp_vllm.step_plan import PlanRequest
from opentsp_vllm.ship_chip import ShipChip
from scripted import Alloc, Driver, run

VOCAB = 50257


class FakePipeline:
    """The pool as (token, position) per slot; `step` is the `Pipeline.step` signature."""

    head = True

    def __init__(self):
        self.pool = {}
        self.calls = []

    def reset(self):
        self.pool = {}

    def step(self, token, pos, check=True, slot=None, counts=None):
        self.calls.append((token, pos, slot, counts.copy()))
        self.pool[slot] = (token, pos)
        visible = [s for s in range(ship_plan.POOL_SLOTS) if (s % 16) < counts.reshape(-1)[s // 16]]
        assert slot in visible, "the lane's own slot is in its encoding"
        h = sum(self.pool[s][0] * 131 + self.pool[s][1] * 7919 for s in visible)
        rng = np.random.default_rng([h % (2**32), token, pos])
        return rng.standard_normal(VOCAB).astype(np.float32), dict(cycles=1000 + pos, pos=pos, slot=slot)

    def batch(self, rows, check=True):
        results = [self.step(t, pos, check=check, slot=slot, counts=counts) for t, pos, slot, counts in rows]
        return [out for out, _ in results], {"cycles": sum(st["cycles"] for _, st in results)}

    def logits(self, out):
        return out


def make_chip():
    chip = ShipChip()
    chip._pipe = FakePipeline()
    return chip


def prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(1, VOCAB, n)]


@pytest.fixture(scope="module")
def vllm_config(tmp_path_factory):
    return H.engine_config(H.write_model_dir(tmp_path_factory.mktemp("ship"), layers=12, vocab=VOCAB, positions=1024), max_model_len=1024,
                           additional_config={"opentsp": {"chip": "ship", "layers": 2}})


@pytest.fixture
def env(vllm_config):
    chip = make_chip()
    runner = TspModelRunner(vllm_config, chip)
    return runner, chip, Driver(runner), Alloc(limit=64)


def solo(p, n_new):
    """The stand-in's greedy continuation of `p`, a single request on a contiguous pool, one token per run."""
    pipe, toks = FakePipeline(), list(p)
    for pos in range(len(p) + n_new - 1):
        out, _ = pipe.step(toks[pos], pos, slot=pos, counts=ship_plan.encoding(list(range(1, pos // 16 + 2)), pos))
        if pos >= len(toks) - 1:
            toks.append(int(np.argmax(out)))
    return toks[len(p):]


def test_the_info_is_gpt2_small_on_a_pool_of_64_pages(env):
    _, chip, _, _ = env
    i = chip.info
    assert (i.layers, i.vocab, i.hidden, i.heads, i.num_blocks, i.grid_c, i.kv_heads, i.head_size) == (12, VOCAB, 768, 12, 65, (1024,), 12, 64)


def test_a_chunked_prefill_and_greedy_decode_equals_the_stand_alone_form(env):
    runner, chip, d, alloc = env
    p = prompt(1, 37)
    d.add("a", p, [])
    run(d, {"a": 5}, alloc)
    assert d.reqs["a"]["out"] == solo(p, 5)
    # every prompt token and every generated token but the last was one lane: 37 + 4 runs
    assert len(chip._pipe.calls) == 37 + 4


def test_two_requests_on_scattered_blocks_each_equal_their_stand_alone_form(env):
    runner, chip, d, alloc = env
    alloc.free = [9, 3, 40, 21, 5, 33, 2, 60, 17, 1, 50, 8]  # not contiguous, not ascending
    pa, pb = prompt(2, 21), prompt(3, 18)
    d.add("a", pa, [])
    d.add("b", pb, [])
    run(d, {"a": 6, "b": 6}, alloc)
    assert d.reqs["a"]["out"] == solo(pa, 6) and d.reqs["b"]["out"] == solo(pb, 6)


def test_blocks_handed_out_again_are_not_zeroed_and_the_result_does_not_depend_on_their_old_content(env):
    runner, chip, d, alloc = env
    pa, pb = prompt(4, 30), prompt(5, 25)
    d.add("a", pa, [])
    done = run(d, {"a": 4}, alloc)
    chip._pipe.pool.update({s: (7, 7) for s in range(ship_plan.POOL_SLOTS) if s not in chip._pipe.pool})
    alloc.give(d.reqs["a"]["blocks"])
    d.add("b", pb, alloc.take(2))
    d.reqs["b"]["blocks"] = list(d.reqs["a"]["blocks"][:2])  # the blocks a freed
    d.step({"b": 8}, finished=done)
    run(d, {"b": 4}, alloc)
    assert d.reqs["b"]["out"] == solo(pb, 4)


def test_preemption_by_recompute_gives_the_same_tokens(env):
    runner, chip, d, alloc = env
    p = prompt(6, 20)
    d.add("a", p, [])
    run(d, {"a": 3}, alloc)
    d.preempt("a")  # the scheduler frees the blocks; the request runs again from position 0 on new blocks
    d.resume_with("a", [30, 31, 32])
    run(d, {"a": 6}, alloc)
    assert d.reqs["a"]["out"] == solo(p, 6)


def test_a_step_returns_the_logit_rows_of_the_lanes_that_need_them_and_the_cycles_of_all_runs(env):
    _, chip, _, _ = env
    toks = prompt(7, 12)
    res = chip.step(chip.plan_step([PlanRequest("a", toks, [1], 0, 4)]))
    assert isinstance(res, ChipResult) and res.logits.shape == (0, VOCAB) and res.cycles == 1000 + 1001 + 1002 + 1003
    res = chip.step(chip.plan_step([PlanRequest("a", toks[:5], [1], 4, 1), PlanRequest("b", toks[:1], [2], 0, 1)]))
    assert res.logits.shape == (2, VOCAB) and res.cycles == 1004 + 1000
    assert chip._pipe.calls[-2][1] == 4 and chip._pipe.calls[-1][1] == 0  # request order


def test_a_failed_run_is_the_chips_failure(env):
    runner, chip, d, alloc = env

    def boom(*a, **k):
        raise RuntimeError("chip 3: diagnostics [OTS-E-TIM-001]")

    chip._pipe.step = boom
    d.add("a", prompt(8, 4), alloc.take(1))
    with pytest.raises(TspChipError, match="OTS-E-TIM-001"):
        d.step({"a": 4})
    assert "OTS-E-TIM-001" in chip.failure and runner.failure


def test_copying_a_block_is_refused_and_zeroing_validates_the_block(env):
    _, chip, _, _ = env
    with pytest.raises(NotImplementedError, match="copy on write"):
        chip.copy_blocks([(1, 2)])
    chip.copy_blocks([])
    chip.zero_blocks([1, 64])
    for bad in (0, 65):
        with pytest.raises(ValueError, match="null block or outside"):
            chip.zero_blocks([bad])


def test_a_chip_without_a_model_says_so():
    chip = ShipChip()
    with pytest.raises(RuntimeError, match="no model is loaded"):
        chip.warm_up()
    with pytest.raises(FileNotFoundError, match="no checkpoint"):
        chip.load_model("/nonexistent")
    with pytest.raises(ValueError, match="synthetic"):
        chip.load_model("synthetic:1")
    chip.close()
    with pytest.raises(RuntimeError, match="closed"):
        chip.zero_pool()
