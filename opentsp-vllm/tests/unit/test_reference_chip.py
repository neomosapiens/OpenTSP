"""The Chip protocol , IssChip without its binding and ReferenceChip, the numpy twin behind the protocol."""

import sys

import numpy as np
import pytest

import reference_loader as oracle
from reference_chip import ReferenceChip
from opentsp_vllm import chip as chip_mod
from opentsp_vllm.chip import Chip, ChipInfo, ChipResult, IssChip
from opentsp_vllm.step_plan import PlanRequest, plan_step

reference_paged, W, config = oracle.load()
CFG = config.ModelConfig(layers=2, vocab=2000, max_pos=128)


def made(**kw):
    c = ReferenceChip(CFG, **kw)
    c.load_model("synthetic:1")
    return c


def req(rid, known, blocks, computed, n):
    return PlanRequest(rid, list(known), list(blocks), computed, n)


def test_reference_chip_is_a_chip_and_reports_its_shape():
    c = made()
    assert isinstance(c, Chip)
    assert c.info == ChipInfo(layers=2, vocab=2000, hidden=64, heads=16, num_blocks=128, grid_c=(64, 128))
    assert made(grid_c=(64,)).info.grid_c == (64,)


def test_the_protocol_names_every_operation_the_runner_uses():
    for name in ("load_model", "warm_up", "zero_pool", "zero_blocks", "copy_blocks", "step", "close", "failure"):
        assert hasattr(Chip, name) or name in getattr(Chip, "__annotations__", {}), name


def test_iss_chip_without_the_compiled_binding_names_the_build_command(monkeypatch):
    # the unit tests run on ReferenceChip and do not need the binding; constructing the ISS chip without it says how to build it
    monkeypatch.setitem(sys.modules, "opentsp_vllm._tsp_iss", None)
    with pytest.raises(ImportError, match="just vllm-binding"):
        IssChip(profile="ref")
    assert issubclass(IssChip, object) and chip_mod.IssChip is IssChip


def test_load_model_needs_a_known_source():
    c = ReferenceChip(CFG)
    with pytest.raises(ValueError, match="synthetic"):
        c.load_model("/nonexistent")
    with pytest.raises(RuntimeError, match="load_model"):
        c.step(plan_step([req("a", [1], [1], 0, 1)], vocab=2000))


def test_step_equals_the_oracles_step_batch_bit_for_bit():
    c = made()
    toks = [5, 6, 7, 8, 9]
    p = plan_step([req("a", toks, [1], 0, 5), req("b", [11, 12], [2], 1, 1)], vocab=2000)
    assert p.bucket == (8, 2, 64)
    r = c.step(p)
    assert isinstance(r, ChipResult) and r.diagnostics == () and r.logits.dtype == np.float32
    # the oracle on the same plan: logit lanes first
    w = W.synthetic(CFG, seed=1)
    pool = reference_paged.BlockPool(CFG)
    lanes = p.real_lanes
    ref = reference_paged.step_batch(
        w, pool, [ln.token for ln in lanes], [ln.position for ln in lanes], [list(ln.block_table) for ln in lanes], p.bucket, p.n_logits
    )
    assert r.logits.shape == (p.n_logits, 2000)
    assert np.array_equal(r.logits, ref.logits)
    assert np.array_equal(c.pool.k, pool.k) and np.array_equal(c.pool.v, pool.v)


def test_a_step_without_logit_rows_returns_an_empty_matrix_and_writes_kv():
    c = made()
    r = c.step(plan_step([req("a", range(8), [1], 0, 4)], vocab=2000))
    assert r.logits.shape == (0, 2000)
    assert np.any(c.pool.k[1, :, :4] != 0) and not np.any(c.pool.k[1, :, 4:])


def test_block_zeroing_and_copies_are_pool_operations():
    c = made()
    c.step(plan_step([req("a", range(8), [1], 0, 8)], vocab=2000))
    k1, v1 = c.pool.k[1].copy(), c.pool.v[1].copy()
    assert np.any(k1)
    c.copy_blocks([(1, 5), (1, 6)])
    assert np.array_equal(c.pool.k[5], k1) and np.array_equal(c.pool.v[6], v1)
    c.zero_blocks([5])
    assert not np.any(c.pool.k[5]) and np.array_equal(c.pool.k[6], k1)
    c.zero_pool()
    assert not np.any(c.pool.k) and not np.any(c.pool.v)


def test_copies_and_zeroing_refuse_the_null_block_and_ids_outside_the_pool():
    c = made()
    with pytest.raises(ValueError):
        c.zero_blocks([0])
    with pytest.raises(ValueError):
        c.copy_blocks([(1, 0)])
    with pytest.raises(ValueError):
        c.zero_blocks([128])  # the sink is not vLLM's to zero
    with pytest.raises(ValueError):
        c.copy_blocks([(1, 200)])


def test_close_makes_the_chip_unusable_and_is_idempotent():
    c = made()
    c.close()
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.step(plan_step([req("a", [1], [1], 0, 1)], vocab=2000))


def test_a_chip_error_is_remembered_as_the_failure():
    c = made()
    assert c.failure is None
    bad = plan_step([req("a", [1], [1], 0, 1)], vocab=2000)
    object.__setattr__(bad.lanes[0], "position", 99)  # corrupt the plan behind the checks: the oracle refuses it
    with pytest.raises(Exception):
        c.step(bad)
    assert c.failure is not None
