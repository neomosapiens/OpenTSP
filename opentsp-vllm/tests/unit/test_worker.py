"""TspWorker ( tables (a) and (b), 5.4 life cycle, 5.8) on ReferenceChip."""

import sys
import unittest.mock

import pytest
import torch
import torch.nn as nn

import reference_loader as oracle
import vllm_helpers as H
from opentsp_vllm import model_runner as model_runner_mod
from opentsp_vllm.chip import IssChip, TspChipError
from opentsp_vllm.worker import TspWorker
from scripted import Alloc, Driver, run
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.worker_base import CompilationTimes, WorkerWrapperBase

_, _, config = oracle.load()

OPTIONS = {"chip": "reference_chip:make_chip", "weights": "synthetic:1", "max_context": 64}
LAYERS = ["transformer.h.0.attn", "transformer.h.1.attn"]


def kv_config(num_blocks=128, layers=LAYERS):
    spec = FullAttentionSpec(block_size=16, num_kv_heads=16, head_size=4, dtype=torch.float32)
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=1, layers=[n], layer_stride=1, block_stride=1) for n in layers],
        kv_cache_groups=[KVCacheGroupSpec(layers, spec)],
    )


@pytest.fixture(scope="module")
def vllm_config(tmp_path_factory):
    return H.engine_config(H.write_model_dir(tmp_path_factory.mktemp("w")), additional_config={"opentsp": OPTIONS})


def started(vllm_config, load=True, **load_kw):
    w = TspWorker(vllm_config, 0, 0, "tcp://127.0.0.1:1", True)
    w.init_device()
    if load:
        w.load_model(**load_kw)
    return w


# ---- life cycle  -----------------------------------------------------------------------------------------------------


def test_init_device_creates_the_chip_and_the_device_is_the_host(vllm_config):
    w = TspWorker(vllm_config, 0, 0, "tcp://127.0.0.1:1", True)
    assert w.chip is None
    w.init_device()
    assert w.chip is not None and w.device == torch.device("cpu")
    assert w.model_runner is None, "the runner needs the loaded model"


def test_the_default_chip_is_the_iss_and_names_the_build_command_without_the_binding(vllm_config, monkeypatch):
    monkeypatch.setitem(sys.modules, "opentsp_vllm._tsp_iss", None)
    cfg = H.engine_config(
        H.write_model_dir(vllm_config.model_config.model), additional_config={"opentsp": {"weights": "synthetic:1"}}
    )
    w = TspWorker(cfg, 0, 0, "tcp://127.0.0.1:1", True)
    with pytest.raises(ImportError, match="just vllm-binding"):
        w.init_device()
    assert IssChip is not None


def test_a_chip_factory_that_does_not_resolve_is_a_clear_error(vllm_config):
    cfg = H.engine_config(
        H.write_model_dir(vllm_config.model_config.model), additional_config={"opentsp": {"chip": "no_such_module:make"}}
    )
    with pytest.raises(ValueError, match="opentsp.chip"):
        TspWorker(cfg, 0, 0, "tcp://127.0.0.1:1", True).init_device()


def test_load_model_loads_the_weights_and_creates_the_runner(vllm_config):
    w = started(vllm_config)
    assert w.model_runner is not None and w.chip.info.layers == 2
    assert isinstance(w.get_model(), nn.Module) and not list(w.get_model().parameters())


def test_load_model_without_weights_names_the_option(vllm_config):
    cfg = H.engine_config(
        H.write_model_dir(vllm_config.model_config.model), additional_config={"opentsp": {"chip": "reference_chip:make_chip"}}
    )
    w = TspWorker(cfg, 0, 0, "tcp://127.0.0.1:1", True)
    w.init_device()
    with pytest.raises(ValueError, match="opentsp.weights"):
        w.load_model()
    w.load_model(load_dummy_weights=True)  # synthetic weights need no option
    assert w.model_runner is not None


def test_methods_that_need_the_chip_say_what_to_call_first(vllm_config):
    w = TspWorker(vllm_config, 0, 0, "tcp://127.0.0.1:1", True)
    for call in (
        lambda: w.load_model(),
        w.get_kv_cache_spec,
        w.compile_or_warm_up_model,
        lambda: w.execute_model(object()),
        w.get_model,
        w.determine_available_memory,
        lambda: w.initialize_from_config(kv_config()),
    ):
        with pytest.raises(RuntimeError, match="init_device|load_model"):
            call()
    assert w.get_tsp_stats()["steps"] == 0 and w.check_health() is None


def test_the_kv_cache_spec_is_one_full_attention_spec_per_layer(vllm_config):
    w = started(vllm_config)
    spec = w.get_kv_cache_spec()
    assert list(spec) == LAYERS
    one = FullAttentionSpec(block_size=16, num_kv_heads=16, head_size=4, dtype=torch.float32)
    assert all(s == one for s in spec.values())


def test_the_kv_spec_has_a_layer_per_layer_of_the_model_8_for_the_full_model(vllm_config):
    cfg8 = config.ModelConfig(layers=8, vocab=2000, max_pos=128)
    with unittest.mock.patch("reference_chip.make_config", return_value=cfg8):
        w = started(vllm_config)
    assert [n for n in w.get_kv_cache_spec()] == [f"transformer.h.{l}.attn" for l in range(8)]


def test_available_memory_is_the_pool_in_bytes(vllm_config):
    w = started(vllm_config)
    page = sum(s.page_size_bytes for s in w.get_kv_cache_spec().values())
    assert w.determine_available_memory() == 128 * page


def test_update_max_model_len_accepts_only_the_configured_value(vllm_config):
    w = started(vllm_config)
    assert w.update_max_model_len(vllm_config.model_config.max_model_len) is None
    with pytest.raises(ValueError, match="max_model_len"):
        w.update_max_model_len(32)


def test_initialize_from_config_checks_the_pool_and_zeroes_it(vllm_config):
    w = started(vllm_config)
    d, alloc = Driver(w.model_runner), Alloc()
    d.add("a", [3, 4, 5], alloc.take(1))
    d.step({"a": 3})
    assert w.chip.pool.k.any()
    w.initialize_from_config(kv_config())
    assert not w.chip.pool.k.any() and not w.chip.pool.v.any()
    with pytest.raises(ValueError, match="num_blocks"):
        w.initialize_from_config(kv_config(num_blocks=129))
    with pytest.raises(ValueError, match="layers"):
        w.initialize_from_config(kv_config(layers=LAYERS[:1]))
    two = kv_config()
    two.kv_cache_groups = two.kv_cache_groups * 2
    with pytest.raises(ValueError, match="group"):
        w.initialize_from_config(two)


def test_the_executor_wrapper_reaches_initialize_from_config_with_the_per_rank_config(vllm_config):
    w = started(vllm_config)
    wrapper = WorkerWrapperBase()
    wrapper.vllm_config = vllm_config
    wrapper.worker = w
    wrapper.initialize_from_config([kv_config()])


def test_warm_up_returns_zero_compilation_times_and_runs_the_chip_once(vllm_config):
    w = started(vllm_config)
    with unittest.mock.patch.object(w.chip, "warm_up", wraps=w.chip.warm_up) as warm:
        assert w.compile_or_warm_up_model() == CompilationTimes(0.0, 0.0)
    warm.assert_called_once_with()


def test_the_supported_tasks_layouts_and_noop_rpcs(vllm_config):
    w = started(vllm_config)
    assert w.get_supported_tasks() == ("generate",)
    assert w.get_supported_kv_cache_layouts() == ["LBNHC"]
    assert w.reset_encoder_cache() is None and w.synchronize_device() is None


# ---- a step through the worker ------------------------------------------------------------------------------------------------------


def test_execute_model_returns_the_model_runner_output_and_never_none(vllm_config):
    w = started(vllm_config)
    d, alloc = Driver(w), Alloc()
    d.add("a", [3, 4, 5], alloc.take(1))
    out = d.step({"a": 3})
    assert out is not None and out.req_ids == ["a"] and len(out.sampled_token_ids[0]) == 1
    assert w.get_tsp_stats()["steps"] == 1


def test_sample_tokens_is_not_reached(vllm_config):
    w = started(vllm_config)
    with pytest.raises(RuntimeError, match="execute_model never returns None"):
        w.sample_tokens(None)


def test_the_stats_aggregate_through_the_rpc_and_are_logged_on_an_interval(vllm_config, monkeypatch):
    w = started(vllm_config)
    logged = []
    monkeypatch.setattr(model_runner_mod, "STATS_LOG_INTERVAL_S", 0.0)
    monkeypatch.setattr(model_runner_mod.logger, "info", lambda msg, *a: logged.append(msg % a))
    d, alloc = Driver(w), Alloc()
    d.add("a", [3, 4, 5], alloc.take(1))
    d.step({"a": 3})
    d.step({"a": 1})
    st = w.get_tsp_stats()
    assert st["steps"] == 2 and st["real_lanes"] == 4 and set(st["buckets"]) == {"4x1x64", "1x1x64"}
    assert logged and "steps=2" in logged[-1]


def test_a_failed_run_makes_check_health_raise(vllm_config):
    w = started(vllm_config)
    assert w.check_health() is None
    d, alloc = Driver(w), Alloc()
    d.add("a", [3, 4, 5], alloc.take(1))

    def boom(plan):
        raise TspChipError("OTS-E-TEST")

    w.chip.step = boom
    with pytest.raises(TspChipError):
        d.step({"a": 3})
    with pytest.raises(RuntimeError, match="OTS-E-TEST"):
        w.check_health()


def test_check_health_raises_for_a_chip_failure_the_runner_did_not_see(vllm_config):
    w = started(vllm_config)
    assert w.model_runner.failure is None
    w.chip._failure = "OTS-E-CHIP at cycle 7"  # a failure outside execute_model, as in warm-up
    with pytest.raises(RuntimeError, match="OTS-E-CHIP"):
        w.check_health()


def test_load_model_refuses_a_max_model_len_the_chips_programs_cannot_hold(tmp_path):
    cfg = H.engine_config(
        H.write_model_dir(tmp_path / "m"),
        max_model_len=128,
        additional_config={"opentsp": {**OPTIONS, "max_context": 128, "chip": "reference_chip:make_chip_64"}},
    )  # the platform accepts 128 (max_context 128); the chip's programs stop at 64
    w = TspWorker(cfg, 0, 0, "tcp://127.0.0.1:1", True)
    w.init_device()
    with pytest.raises(ValueError, match="max_model_len 128 exceeds 64"):
        w.load_model()
    assert w.model_runner is None


def test_shutdown_drops_the_chip_and_is_idempotent(vllm_config):
    w = started(vllm_config)
    chip = w.chip
    w.shutdown()
    w.shutdown()
    assert w.chip is None and w.model_runner is None
    with pytest.raises(RuntimeError, match="closed"):
        chip.warm_up()
    with pytest.raises(RuntimeError, match="init_device"):
        w.execute_model(object())


def test_a_whole_generation_through_the_worker_equals_the_oracle(vllm_config):
    reference_paged, W, _ = oracle.load()
    cfg = config.ModelConfig(layers=2, vocab=2000, max_pos=128)
    prompt = [int(t) for t in range(100, 121)]
    want = reference_paged.generate_paged(W.synthetic(cfg, seed=1), prompt, 6, C=64).tokens[len(prompt):]
    w = started(vllm_config)
    w.initialize_from_config(kv_config())
    w.compile_or_warm_up_model()
    d, alloc = Driver(w), Alloc()
    d.add("a", prompt, [], params=SamplingParams(temperature=0.0, max_tokens=6))
    run(d, {"a": 6}, alloc)
    assert d.reqs["a"]["out"] == want


def test_the_engine_core_cleanup_after_a_worker_started_does_not_crash_the_process(vllm_config):
    """`cleanup_dist_env_and_memory` (the engine core's shutdown, and an exit handler) asks the host-memory cache of the torch
    accelerator to empty because the platform is not CPU; on macOS that call kills the process, so the worker makes it a no-op."""
    import subprocess
    import textwrap

    code = textwrap.dedent(
        """
        import sys
        sys.path[:0] = sys.argv[2:]
        import vllm_helpers as H
        from opentsp_vllm.worker import TspWorker
        cfg = H.engine_config(sys.argv[1], additional_config={"opentsp": {"chip": "reference_chip:make_chip", "weights": "synthetic:1", "max_context": 64}})
        TspWorker(cfg, 0, 0, "tcp://127.0.0.1:1", True).init_device()
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
        cleanup_dist_env_and_memory()
        print("cleaned")
        """
    )
    import os

    env = dict(os.environ, OPENTSP_VLLM="1")
    r = subprocess.run(
        [sys.executable, "-c", code, str(vllm_config.model_config.model)] + H.TESTS_DIRS.split(os.pathsep),
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0 and "cleaned" in r.stdout, (r.returncode, r.stderr[-800:])


class _BindingStub:
    """The part of the compiled binding that `ProgramSet` and `IssChip` touch before they look at the program cache: the test below
    is about the missing cache, so it must not depend on whether `just vllm-binding` has been run."""

    SPEC_VERSION = "0.1"

    @staticmethod
    def profile_sha256(profile):
        return "0" * 64

    class Chip:
        profile_sha256 = "0" * 64

        def __init__(self, profile):
            self.profile = profile

        def restart_mode(self, mode):
            self.mode = mode


def test_the_iss_chip_without_a_program_cache_names_the_build_command(vllm_config, tmp_path, monkeypatch):
    from opentsp_vllm.programs import ProgramsError

    monkeypatch.setitem(sys.modules, "opentsp_vllm._tsp_iss", _BindingStub)
    cfg = H.engine_config(
        H.write_model_dir(vllm_config.model_config.model),
        additional_config={"opentsp": {"weights": "synthetic:1", "max_context": 64, "programs": str(tmp_path)}},
    )
    with pytest.raises(ProgramsError, match="just vllm-programs"):
        TspWorker(cfg, 0, 0, "tcp://127.0.0.1:1", True).init_device()


def test_the_shadow_variable_wraps_the_chip_in_a_shadow_chip_with_a_twin(vllm_config, monkeypatch):
    from opentsp_vllm.shadow import ShadowChip

    monkeypatch.setenv("OPENTSP_VLLM_SHADOW", "reference_chip:make_chip")
    w = started(vllm_config)
    assert isinstance(w.chip, ShadowChip) and w.chip.shadow is not w.chip.chip and w.chip.shadow_weights == "synthetic:1"
    w.initialize_from_config(kv_config())
    w.compile_or_warm_up_model()
    d, alloc = Driver(w), Alloc()
    d.add("a", list(range(100, 108)), [], params=SamplingParams(temperature=0.0, max_tokens=3))
    run(d, {"a": 3}, alloc)
    assert len(d.reqs["a"]["out"]) == 3 and w.check_health() is None
    monkeypatch.setenv("OPENTSP_VLLM_SHADOW", "no_such_module:make")
    with pytest.raises(ValueError, match="OPENTSP_VLLM_SHADOW"):
        TspWorker(vllm_config, 0, 0, "tcp://127.0.0.1:1", True).init_device()


def test_a_shadow_with_other_weights_fails_the_engine_step_with_the_mismatch(vllm_config, monkeypatch):
    monkeypatch.setenv("OPENTSP_VLLM_SHADOW", "reference_chip:make_chip")
    monkeypatch.setenv("OPENTSP_VLLM_SHADOW_WEIGHTS", "synthetic:2")
    w = started(vllm_config)
    w.initialize_from_config(kv_config())
    d, alloc = Driver(w), Alloc()
    d.add("a", list(range(100, 108)), [], params=SamplingParams(temperature=0.0, max_tokens=3))
    with pytest.raises(TspChipError, match="shadow mismatch"):
        run(d, {"a": 3}, alloc)
    assert "shadow mismatch" in str(w.chip.failure)
    with pytest.raises(RuntimeError, match="failed state"):
        w.check_health()
