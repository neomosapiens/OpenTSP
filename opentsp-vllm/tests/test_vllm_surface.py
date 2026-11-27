"""The vLLM surface gate : the pinned vLLM release still has every name, signature and RPC the design
( sec. 5.1) depends on, and the package skeleton matches it.

Run in the vLLM environment (`just vllm-surface`). It needs vLLM and does not skip without it. A re-pin changes PIN here, in
tools/setup_vllm_env.sh and in pyproject.toml together; a failure below then names what the new release moved.
Signatures are compared by parameter names and kinds, never by annotations; the two defaults the worker relies on
(`load_dummy_weights`, `predict_bonus_token`) are compared separately.
"""

import ast
import dataclasses
import importlib
import importlib.metadata
import inspect
import os
import pathlib
import re
import subprocess
import sys
import textwrap
import unittest.mock

import pytest

PIN = "0.33.0"
REPO = pathlib.Path(__file__).resolve().parents[2]
TSP_PLATFORM = "opentsp_vllm.platform.TspPlatform"


def sig(text):
    """The parameter names and kinds of a signature written as `(self, a, *, b=1)`."""
    ns = {}
    exec(f"def f{text}: pass", ns)  # noqa: S102 - the text is a literal of this file
    return params(ns["f"])


def params(fn):
    return [(p.name, p.kind.name) for p in inspect.signature(fn).parameters.values()]


def default_of(fn, name):
    return inspect.signature(fn).parameters[name].default


def resolve(path):
    """`pkg.mod:Cls.attr` -> the object, imported from the installed vLLM."""
    mod, _, attr = path.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split(".") if attr else []:
        obj = getattr(obj, part)
    return obj


def run_python(code, **env):
    full = {k: v for k, v in os.environ.items() if k != "OPENTSP_VLLM"}
    full.update(env)
    out = subprocess.run([sys.executable, "-c", code], env=full, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


# ---- the pin ---------------------------------------------------------------------------------------------------------


def test_installed_vllm_is_the_pin():
    # the source build carries the local version tag "+cpu"
    assert importlib.metadata.version("vllm").split("+")[0] == PIN
    import vllm

    assert vllm.__version__.split("+")[0] == PIN


def test_the_pin_is_the_same_in_the_setup_script_and_the_package():
    script = (REPO / "tools" / "setup_vllm_env.sh").read_text()
    assert re.search(rf'^VLLM_VERSION="{re.escape(PIN)}"$', script, re.M)
    assert re.search(r'^VLLM_SDIST_SHA256="[0-9a-f]{64}"$', script, re.M)
    assert f'"vllm=={PIN}"' in (REPO / "opentsp-vllm" / "pyproject.toml").read_text()


# ---- the extension points (first table) --------------------------------------------------------------------


def test_platform_plugin_group_and_resolver():
    from vllm.plugins import PLATFORM_PLUGINS_GROUP

    assert PLATFORM_PLUGINS_GROUP == "vllm.platform_plugins"
    assert params(resolve("vllm.platforms:resolve_current_platform_cls_qualname")) == sig("()")


def test_platform_interface():
    from vllm.platforms.interface import Platform, PlatformEnum

    assert hasattr(PlatformEnum, "OOT")
    annotations = {}
    for klass in reversed(Platform.__mro__):
        annotations.update(getattr(klass, "__annotations__", {}))
    for attr in ("_enum", "device_name", "device_type", "dispatch_key"):
        assert attr in annotations, attr
    assert Platform.dispatch_key == "CPU"
    assert params(Platform.check_and_update_config) == sig("(vllm_config)")
    assert params(Platform.validate_request) == sig("(processed_inputs, params)")
    assert isinstance(inspect.getattr_static(Platform, "check_and_update_config"), classmethod)
    assert isinstance(inspect.getattr_static(Platform, "validate_request"), classmethod)


def test_config_fields_the_platform_hook_sets_or_checks():
    wanted = {
        "vllm.config.parallel:ParallelConfig": ("worker_cls", "distributed_executor_backend", "world_size"),
        "vllm.config.scheduler:SchedulerConfig": ("async_scheduling", "max_num_seqs", "max_num_batched_tokens"),
        "vllm.config.cache:CacheConfig": ("block_size", "num_gpu_blocks_override", "cache_dtype"),
        "vllm.config.model:ModelConfig": ("max_model_len", "enable_sleep_mode"),
        "vllm.config.vllm:VllmConfig": (
            "additional_config",
            "speculative_config",
            "lora_config",
            "kv_transfer_config",
            "weight_transfer_config",
            "parallel_config",
            "scheduler_config",
            "cache_config",
            "model_config",
        ),
    }
    for path, names in wanted.items():
        have = {f.name for f in dataclasses.fields(resolve(path))}
        assert set(names) <= have, (path, set(names) - have)
    from vllm.config.parallel import DistributedExecutorBackend

    assert "uni" in DistributedExecutorBackend.__args__


def test_scheduler_output_and_runner_output_fields():
    wanted = {
        "vllm.v1.core.sched.output:SchedulerOutput": (
            "scheduled_new_reqs",
            "scheduled_cached_reqs",
            "num_scheduled_tokens",
            "finished_req_ids",
            "preempted_req_ids",
            "kv_cache_block_copies",
            "new_block_ids_to_zero",
        ),
        "vllm.v1.core.sched.output:NewRequestData": (
            "req_id",
            "prompt_token_ids",
            "sampling_params",
            "pooling_params",
            "block_ids",
            "num_computed_tokens",
        ),
        "vllm.v1.core.sched.output:CachedRequestData": (
            "req_ids",
            "resumed_req_ids",
            "new_token_ids",
            "new_block_ids",
            "num_computed_tokens",
            "num_output_tokens",
        ),
        "vllm.v1.outputs:ModelRunnerOutput": ("req_ids", "req_id_to_index", "sampled_token_ids", "logprobs"),
        "vllm.v1.outputs:SamplerOutput": ("sampled_token_ids", "logprobs_tensors"),
        "vllm.v1.kv_cache_interface:KVCacheConfig": ("num_blocks", "kv_cache_tensors", "kv_cache_groups"),
        "vllm.v1.worker.gpu_input_batch:CachedRequestState": (
            "req_id",
            "prompt_token_ids",
            "sampling_params",
            "generator",
            "block_ids",
            "num_computed_tokens",
            "output_token_ids",
        ),
    }
    for path, names in wanted.items():
        have = {f.name for f in dataclasses.fields(resolve(path))}
        assert set(names) <= have, (path, set(names) - have)
    from vllm.v1.outputs import LogprobsLists

    assert "logprobs" in LogprobsLists._fields
    assert params(resolve("vllm.v1.outputs:LogprobsTensors.tolists")) == sig("(self, cu_num_generated_tokens=None)")


def test_kv_cache_spec():
    from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec

    assert issubclass(FullAttentionSpec, AttentionSpec)
    assert dataclasses.is_dataclass(FullAttentionSpec)
    assert {"block_size", "num_kv_heads", "head_size", "dtype"} <= {f.name for f in dataclasses.fields(FullAttentionSpec)}


def test_input_batch_and_sampler():
    ib = "vllm.v1.worker.gpu_input_batch:InputBatch"
    assert params(resolve(ib + ".add_request")) == sig("(self, request)")
    assert params(resolve(ib + ".remove_request")) == sig("(self, req_id)")
    assert params(resolve(ib + ".condense")) == sig("(self)")
    assert params(resolve(ib + ".refresh_metadata")) == sig("(self)")
    assert params(resolve(ib + ".__init__")) == sig(
        "(self, max_num_reqs, max_model_len, max_num_batched_tokens, device, vocab_size, block_sizes,"
        " max_num_blocks_per_req, logitsprocs=None, logitsprocs_need_output_token_ids=False, num_spec_tokens=0,"
        " is_pooling_model=False, cp_kv_cache_interleave_size=1, reasoning_config=None, use_replayssm=False,"
        " slot_mapping_modes=None)"
    )
    assert "generators" in {f.name for f in dataclasses.fields(resolve("vllm.v1.sample.metadata:SamplingMetadata"))}
    sampler = "vllm.v1.sample.sampler:Sampler.forward"
    assert params(resolve(sampler)) == sig("(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None)")
    assert default_of(resolve(sampler), "predict_bonus_token") is False  # the runner calls Sampler without the argument


def test_front_end_classes_exist():
    for path in (
        "vllm.v1.engine.input_processor:InputProcessor",
        "vllm.v1.engine.output_processor:OutputProcessor",
        "vllm.v1.engine.detokenizer:IncrementalDetokenizer",
    ):
        assert inspect.isclass(resolve(path)), path


def test_collective_rpc_entry_points():
    assert params(resolve("vllm.entrypoints.llm:LLM.collective_rpc")) == sig("(self, method, timeout=None, args=(), kwargs=None)")
    async_rpc = resolve("vllm.v1.engine.async_llm:AsyncLLM.collective_rpc")
    assert inspect.iscoroutinefunction(async_rpc)
    assert params(async_rpc) == sig("(self, method, timeout=None, args=(), kwargs=None)")
    assert params(resolve("vllm.v1.executor.uniproc_executor:UniProcExecutor.collective_rpc")) == sig(
        "(self, method, timeout=None, args=(), kwargs=None, non_block=False, single_value=False)"
    )


# ---- WorkerBase and WorkerWrapperBase (table (a)) ------------------------------------------------------------------------

WORKER_BASE = "vllm.v1.worker.worker_base:WorkerBase"
# name -> (signature, "abstract" (the base body raises NotImplementedError) | "default")
TABLE_A = {
    "init_device": ("(self)", "abstract"),
    "load_model": ("(self, *, load_dummy_weights=False)", "abstract"),
    "get_kv_cache_spec": ("(self)", "abstract"),
    "get_supported_kv_cache_layouts": ("(self)", "default"),
    "set_kv_cache_layout": ("(self, kv_cache_layout, cp_kv_cache_interleave_size)", "default"),
    "compile_or_warm_up_model": ("(self)", "abstract"),
    "execute_model": ("(self, scheduler_output)", "abstract"),
    "sample_tokens": ("(self, grammar_output)", "abstract"),
    "get_model": ("(self)", "abstract"),
    "check_health": ("(self)", "default"),
    "synchronize_device": ("(self)", "default"),
    "reset_mm_cache": ("(self)", "default"),
    "release_late_interaction_query_cache": ("(self, query_keys)", "default"),
    "supports_draft_weight_updates": ("(self)", "default"),
    "get_cache_block_size_bytes": ("(self)", "abstract"),
    "add_lora": ("(self, lora_request)", "abstract"),
    "remove_lora": ("(self, lora_id)", "abstract"),
    "pin_lora": ("(self, lora_id)", "abstract"),
    "list_loras": ("(self)", "abstract"),
    "shutdown": ("(self)", "default"),
}


def raises_not_implemented(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    body = [n for n in tree.body[0].body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    exc = exc.func if isinstance(exc, ast.Call) else exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def test_worker_base_load_model_defaults_to_real_weights():
    assert default_of(resolve(WORKER_BASE + ".load_model"), "load_dummy_weights") is False


def test_worker_base_init_and_wrapper():
    assert params(resolve(WORKER_BASE + ".__init__")) == sig(
        "(self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker=False)"
    )
    wrapper = "vllm.v1.worker.worker_base:WorkerWrapperBase"
    assert params(resolve(wrapper + ".init_worker")) == sig("(self, all_kwargs)")
    assert params(resolve(wrapper + ".init_device")) == sig("(self)")
    assert params(resolve(wrapper + ".initialize_from_config")) == sig("(self, kv_cache_configs)")
    assert params(resolve(wrapper + ".__getattr__")) == sig("(self, attr)")
    assert params(resolve(wrapper + ".execute_model")) == sig("(self, scheduler_output)")
    assert params(resolve(wrapper + ".reset_mm_cache")) == sig("(self)")


def test_table_a_against_worker_base():
    from vllm.v1.worker.worker_base import WorkerBase

    for name, (expected, kind) in TABLE_A.items():
        fn = getattr(WorkerBase, name)
        assert params(fn) == sig(expected), name
        assert raises_not_implemented(fn) == (kind == "abstract"), name
    assert isinstance(inspect.getattr_static(WorkerBase, "vocab_size"), property)
    assert not hasattr(WorkerBase, "initialize_from_config"), "initialize_from_config is the wrapper's, not WorkerBase's"
    assert not hasattr(WorkerBase, "determine_available_memory")


def test_compilation_times():
    from vllm.v1.worker.worker_base import CompilationTimes

    assert CompilationTimes._fields == ("language_model", "encoder")


def test_default_kv_cache_layout_preference_starts_with_lbnhc():
    from vllm.v1.attention.backends.utils import _DEFAULT_LAYOUT_PREFERENCE

    assert _DEFAULT_LAYOUT_PREFERENCE[0].name == "LBNHC"


# ---- the RPC names (table (b)) -----------------------------------------------------------------------------------------

RPC_FILES = ("vllm.v1.executor.abstract", "vllm.v1.executor.uniproc_executor", "vllm.v1.engine.core")
# every literal collective_rpc("...") name of the three files in 0.33.0
TABLE_B = {
    "get_kv_cache_spec",
    "get_supported_kv_cache_layouts",
    "set_kv_cache_layout",
    "determine_available_memory",
    "update_max_model_len",
    "initialize_from_config",
    "get_simple_cpu_offload_num_cpu_blocks",
    "compile_or_warm_up_model",
    "get_supported_tasks",
    "execute_model",
    "sample_tokens",
    "shutdown",
    "reset_mm_cache",
    "reset_encoder_cache",
    "synchronize_device",
    "execute_dummy_batch",
    "take_draft_token_ids",
    "supports_draft_weight_updates",
    "get_kv_connector_handshake_metadata",
    "profile",
    "sleep",
    "wake_up",
    "discard",
    "save_sharded_state",
    "compute_weight_checksums",
    "add_lora",
    "remove_lora",
    "pin_lora",
    "list_loras",
}
# the names TspWorker defines or inherits as designed (table (b), rows that name a TspWorker body, plus ours)
USED = TABLE_B - {
    "get_simple_cpu_offload_num_cpu_blocks",
    "execute_dummy_batch",
    "take_draft_token_ids",
    "supports_draft_weight_updates",
    "get_kv_connector_handshake_metadata",
    "profile",
    "sleep",
    "wake_up",
    "discard",
    "save_sharded_state",
    "compute_weight_checksums",
    "add_lora",
    "remove_lora",
    "pin_lora",
    "list_loras",
} | {"get_tsp_stats"}
# the last two rows of table (b): TspWorker must not define the ones WorkerBase lacks, so a call fails with AttributeError
UNDEFINED = {
    "get_simple_cpu_offload_num_cpu_blocks",
    "execute_dummy_batch",
    "take_draft_token_ids",
    "get_kv_connector_handshake_metadata",
    "profile",
    "sleep",
    "wake_up",
    "discard",
    "save_sharded_state",
    "compute_weight_checksums",
}


def collective_rpc_names():
    literal, dynamic = set(), []
    for modname in RPC_FILES:
        path = pathlib.Path(importlib.import_module(modname).__file__)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "collective_rpc":
                arg = node.args[0] if node.args else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literal.add(arg.value)
                else:
                    dynamic.append((modname, ast.unparse(arg) if arg is not None else None))
    return literal, dynamic


def test_rpc_names_are_exactly_table_b():
    literal, dynamic = collective_rpc_names()
    assert literal == TABLE_B, (sorted(literal - TABLE_B), sorted(TABLE_B - literal))
    # the one pass-through (`EngineCore.collective_rpc(method, ...)`); a new variable call site would be unlisted names
    assert dynamic == [("vllm.v1.engine.core", "method")]


def test_table_b_partition_is_consistent():
    assert UNDEFINED <= TABLE_B and not (USED & UNDEFINED)
    assert len(TABLE_B) == 29


# ---- the package skeleton ------------------------------------------------------------------------------------------------


def test_entry_point_is_registered_in_the_platform_group():
    eps = importlib.metadata.entry_points(group="vllm.platform_plugins")
    ours = [e for e in eps if e.name == "opentsp"]
    assert [e.value for e in ours] == ["opentsp_vllm:register"]


def test_plugin_is_inert_without_the_variable_and_selects_tsp_platform_with_it():
    import opentsp_vllm

    with unittest.mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OPENTSP_VLLM", None)
        assert opentsp_vllm.register() is None
        for value in ("0", "", "true"):
            os.environ["OPENTSP_VLLM"] = value
            assert opentsp_vllm.register() is None, value
        os.environ["OPENTSP_VLLM"] = "1"
        assert opentsp_vllm.register() == TSP_PLATFORM


def test_vllm_platform_resolution_follows_the_variable():
    code = "from vllm.platforms import resolve_current_platform_cls_qualname as r; print(r())"
    assert run_python(code, OPENTSP_VLLM="1") == TSP_PLATFORM
    without = run_python(code)
    assert without != TSP_PLATFORM
    assert run_python(code, OPENTSP_VLLM="0") == without


def test_current_platform_is_the_tsp_platform_with_the_variable():
    code = (
        "from vllm.platforms import current_platform as p;"
        "print(type(p).__module__ + '.' + type(p).__qualname__, p.device_name, p.device_type, p.dispatch_key, p.is_out_of_tree())"
    )
    assert run_python(code, OPENTSP_VLLM="1") == f"{TSP_PLATFORM} tsp cpu CPU True"


def test_tsp_platform_attributes_and_hooks():
    from vllm.platforms.interface import Platform, PlatformEnum

    from opentsp_vllm.platform import TspPlatform

    assert issubclass(TspPlatform, Platform)
    assert TspPlatform._enum is PlatformEnum.OOT
    assert (TspPlatform.device_name, TspPlatform.device_type, TspPlatform.dispatch_key) == ("tsp", "cpu", "CPU")
    assert params(TspPlatform.check_and_update_config) == sig("(vllm_config)")
    assert params(TspPlatform.validate_request) == sig("(processed_inputs, params)")
    assert isinstance(inspect.getattr_static(TspPlatform, "check_and_update_config"), classmethod)
    assert isinstance(inspect.getattr_static(TspPlatform, "validate_request"), classmethod)


def tsp_worker():
    from opentsp_vllm.worker import TspWorker

    return TspWorker(unittest.mock.MagicMock(), 0, 0, "tcp://127.0.0.1:1")


def test_tsp_worker_defines_every_method_of_table_a():
    from opentsp_vllm.worker import TspWorker
    from vllm.v1.worker.worker_base import WorkerBase

    assert issubclass(TspWorker, WorkerBase)
    # table (a), column TspWorker: "kept" rows are inherited, every other row has a body of its own
    kept = {
        "set_kv_cache_layout",
        "reset_mm_cache",
        "release_late_interaction_query_cache",
        "supports_draft_weight_updates",
    }
    not_implemented = {"get_cache_block_size_bytes", "add_lora", "remove_lora", "pin_lora", "list_loras"}
    for name, (expected, _) in TABLE_A.items():
        if name in kept or name in not_implemented:
            assert name not in vars(TspWorker), f"{name} is kept from WorkerBase"
        else:
            assert name in vars(TspWorker), f"TspWorker must define {name}"
            assert params(vars(TspWorker)[name]) == sig(expected), name
    assert "vocab_size" not in vars(TspWorker)


def test_tsp_worker_methods_that_need_the_chip_fail_before_init_device():
    worker = tsp_worker()
    needs_chip = {
        "load_model": (),
        "get_kv_cache_spec": (),
        "compile_or_warm_up_model": (),
        "execute_model": (object(),),
        "get_model": (),
        "determine_available_memory": (),
        "initialize_from_config": (object(),),
    }
    for name, args in needs_chip.items():
        with pytest.raises(RuntimeError, match="init_device|load_model"):
            getattr(worker, name)(*args)
    with pytest.raises(RuntimeError, match="not reached"):
        worker.sample_tokens(object())
    assert worker.check_health() is None
    assert worker.get_tsp_stats()["steps"] == 0


def test_tsp_worker_documented_bodies():
    worker = tsp_worker()
    assert worker.get_supported_kv_cache_layouts() == ["LBNHC"]
    assert worker.get_supported_tasks() == ("generate",)
    assert worker.synchronize_device() is None
    assert worker.reset_encoder_cache() is None
    assert worker.shutdown() is None


def test_used_rpc_names_resolve_through_the_executor_wrapper():
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    wrapper = WorkerWrapperBase()
    wrapper.vllm_config = unittest.mock.MagicMock()
    wrapper.worker = tsp_worker()
    for name in sorted(USED):
        assert callable(getattr(wrapper, name)), name
    for name in sorted(UNDEFINED):
        with pytest.raises(AttributeError, match=name):
            getattr(wrapper, name)
    # the wrapper unpacks the per-rank KV config and reaches the worker's own method
    with pytest.raises(RuntimeError, match="init_device"):
        wrapper.initialize_from_config([object()])


def test_source_tree_is_what_the_design_names():
    pkg = REPO / "opentsp-vllm" / "src" / "opentsp_vllm"
    assert {p.name for p in pkg.glob("*.py")} >= {
        "__init__.py",
        "platform.py",
        "worker.py",
        "model_runner.py",
        "step_plan.py",
        "chip.py",
        "programs.py",
        "stats.py",
    }
