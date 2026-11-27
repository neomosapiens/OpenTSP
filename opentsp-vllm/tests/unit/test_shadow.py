"""ShadowChip (test only): the chip and the bit-exact reference twin run every step of the same plan, their logits must be bit-identical."""

from types import SimpleNamespace

import numpy as np
import pytest

import reference_loader as oracle
from reference_chip import ReferenceChip
from opentsp_vllm.chip import Chip, ChipResult, TspChipError
from opentsp_vllm.shadow import ShadowChip
from opentsp_vllm.step_plan import PlanRequest, plan_step

_, _, config = oracle.load()
CFG = config.ModelConfig(layers=2, vocab=2000, max_pos=128)


def shadowed(chip_seed=1, shadow_seed=1):
    chip, twin = ReferenceChip(CFG), ReferenceChip(CFG)
    s = ShadowChip(chip, twin, f"synthetic:{shadow_seed}")
    s.load_model(f"synthetic:{chip_seed}")
    return s, chip, twin


def plan(tokens=(5, 6, 7), blocks=(1,), computed=0):
    return plan_step([PlanRequest("a", list(tokens), list(blocks), computed, len(tokens) - computed)], vocab=2000)


def test_the_shadow_chip_is_a_chip_with_the_primary_s_info():
    s, chip, _ = shadowed()
    assert isinstance(s, Chip) and s.info == chip.info and s.failure is None


def test_a_step_returns_the_primary_s_result_when_the_twin_agrees_and_both_see_every_block_operation():
    s, chip, twin = shadowed()
    s.warm_up()
    s.zero_blocks([1, 2])
    s.copy_blocks([(1, 2)])
    r = s.step(plan())
    assert np.array_equal(r.logits, chip.results[-1].logits) and len(twin.results) == 1


def test_a_twin_that_differs_fails_the_step_naming_the_logit_and_leaves_the_chip_failed():
    s, *_ = shadowed(chip_seed=1, shadow_seed=2)
    with pytest.raises(TspChipError, match="shadow") as e:
        s.step(plan())
    assert "step 0" in str(e.value) and s.failure and "shadow" in s.failure


def test_close_closes_both_and_the_pool_zeroing_reaches_both():
    s, chip, twin = shadowed()
    s.zero_pool()
    s.close()
    for c in (chip, twin):
        with pytest.raises(RuntimeError, match="closed"):
            c.warm_up()


def test_shadow_checks_the_representations_of_zero_and_nan():
    def chip(bits):
        logits = np.array([bits], dtype=np.uint32).view(np.float32)
        return SimpleNamespace(step=lambda _: ChipResult(logits))

    message = SimpleNamespace(bucket=(1, 1, 64))
    nan = [0x7fc00001]
    assert ShadowChip(chip(nan), chip(nan), None).step(message).logits.view(np.uint32)[0, 0] == nan[0]
    for a, b in [(0, 0x80000000), (0x7fc00001, 0x7fc00002)]:
        with pytest.raises(TspChipError, match="logit row"):
            ShadowChip(chip([a]), chip([b]), None).step(message)


def test_failed_primary_close_still_closes_the_shadow():
    def fail():
        raise RuntimeError("primary close failed")

    closed = []
    primary = SimpleNamespace(close=fail)
    reference = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="primary close failed"):
        ShadowChip(primary, reference, None).close()
    assert closed == [True]


def test_shadow_preserves_the_primary_step_planning_hook():
    seen = object()
    primary = SimpleNamespace(plan_step=lambda request: request)
    reference = SimpleNamespace(plan_step=lambda _: pytest.fail("the reference does not plan the step"))
    assert ShadowChip(primary, reference, None).plan_step(seen) is seen


def test_shadow_preserves_a_primary_without_a_planning_hook():
    reference = SimpleNamespace(plan_step=lambda _: pytest.fail("the reference does not plan the step"))
    assert getattr(ShadowChip(SimpleNamespace(), reference, None), "plan_step", None) is None
