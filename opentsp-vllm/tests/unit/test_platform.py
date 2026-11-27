"""TspPlatform.check_and_update_config and validate_request ( feature list, 5.1)."""

import subprocess
import sys

import pytest

from opentsp_vllm.platform import TspPlatform
from vllm.exceptions import VLLMValidationError
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

import vllm_helpers as H


def test_the_current_platform_is_the_tsp_platform_in_the_unit_tests():
    from vllm.platforms import current_platform

    assert isinstance(current_platform, TspPlatform)


# ---- check_and_update_config -------------------------------------------------------------------------------------------------


def test_the_hook_runs_with_the_vllm_config_and_sets_what_the_design_names(model_dir):
    cfg = H.engine_config(model_dir)  # the hook ran inside VllmConfig.__post_init__
    assert cfg.parallel_config.worker_cls == "opentsp_vllm.worker.TspWorker"
    assert cfg.parallel_config.distributed_executor_backend == "uni"
    assert cfg.scheduler_config.async_scheduling is False
    assert cfg.cache_config.block_size == 16
    assert cfg.cache_config.num_gpu_blocks_override == 128
    assert cfg.parallel_config.world_size == 1


def test_async_scheduling_is_turned_off_not_rejected(model_dir):
    cfg = H.engine_config(model_dir, async_scheduling=True)
    assert cfg.scheduler_config.async_scheduling is False


def test_the_pool_size_defaults_to_the_chip_pool_and_a_smaller_override_stays(model_dir):
    # : preemption is forced with an override of 6 blocks (the scheduler's pool, not the chip's)
    assert H.engine_config(model_dir).cache_config.num_gpu_blocks_override == 128
    assert H.engine_config(model_dir, num_gpu_blocks_override=6).cache_config.num_gpu_blocks_override == 6
    assert H.engine_config(model_dir, num_gpu_blocks_override=128).cache_config.num_gpu_blocks_override == 128


@pytest.mark.parametrize("blocks", [129, 1000, 4, 1, 0])
def test_a_pool_override_above_the_chip_pool_or_below_one_request_is_rejected(model_dir, blocks):
    # 64 positions are 4 blocks of 16, plus the null block: 5 is the least that can hold one request
    with pytest.raises(ValueError, match="num_gpu_blocks_override"):
        H.engine_config(model_dir, num_gpu_blocks_override=blocks)


def test_the_smallest_pool_that_holds_one_request_is_accepted(model_dir):
    assert H.engine_config(model_dir, num_gpu_blocks_override=5).cache_config.num_gpu_blocks_override == 5


def test_a_block_size_other_than_16_is_rejected_not_replaced(model_dir):
    with pytest.raises(ValueError, match="block_size"):
        H.engine_config(model_dir, block_size=32)


def test_kv_offloading_is_rejected(model_dir):
    # the offload connector is derived from kv_offloading_size after the config's own kv_transfer_config is read
    with pytest.raises(ValueError, match="kv_offloading"):
        H.engine_config(model_dir, kv_offloading_size=1.0, kv_offloading_backend="native")


def test_the_connector_checks_run_again_after_the_hook_has_set_its_fields(model_dir, monkeypatch):
    import opentsp_vllm.platform as P

    cfg = H.engine_config(model_dir)
    seen = []
    real = P._check_connectors
    monkeypatch.setattr(P, "_check_connectors", lambda c: (seen.append(c.parallel_config.worker_cls), real(c))[1])
    cfg.parallel_config.worker_cls = "auto"
    TspPlatform.check_and_update_config(cfg)
    assert len(seen) == 2 and seen[0] == "auto" and seen[1] == P.WORKER_CLS


def test_the_hook_is_idempotent(model_dir):
    cfg = H.engine_config(model_dir)
    TspPlatform.check_and_update_config(cfg)
    TspPlatform.check_and_update_config(cfg)
    assert cfg.cache_config.block_size == 16


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: setattr(c.model_config, "max_model_len", 129), "max_model_len"),
        (lambda c: setattr(c.scheduler_config, "max_num_seqs", 5), "max_num_seqs"),
        (lambda c: setattr(c.scheduler_config, "max_num_batched_tokens", 9), "max_num_batched_tokens"),
        (lambda c: setattr(c.scheduler_config, "enable_chunked_prefill", False), "chunked prefill"),
        (lambda c: setattr(c.cache_config, "cache_dtype", "fp8"), "cache_dtype"),
        (lambda c: setattr(c.cache_config, "cache_dtype", "int8"), "cache_dtype"),
        (lambda c: setattr(c.parallel_config, "world_size", 2), "world_size"),
        (lambda c: setattr(c.parallel_config, "data_parallel_size", 2), "data_parallel_size"),
        (lambda c: setattr(c, "speculative_config", object()), "speculative"),
        (lambda c: setattr(c, "lora_config", object()), "LoRA"),
        (lambda c: setattr(c, "kv_transfer_config", object()), "kv_transfer_config"),
        (lambda c: setattr(c, "weight_transfer_config", object()), "weight_transfer_config"),
        (lambda c: setattr(c.model_config, "enable_sleep_mode", True), "sleep"),
    ],
)
def test_unsupported_configurations_are_rejected_with_the_option_named(model_dir, mutate, match):
    cfg = H.engine_config(model_dir)
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        TspPlatform.check_and_update_config(cfg)


def test_the_model_must_be_a_text_generation_model(model_dir):
    cfg = H.engine_config(model_dir)
    cfg.model_config.runner_type = "pooling"
    with pytest.raises(ValueError, match="generat"):
        TspPlatform.check_and_update_config(cfg)


def test_max_model_len_is_bounded_by_the_largest_context_of_the_programs(model_dir):
    cfg = H.engine_config(model_dir, max_model_len=64, additional_config={"opentsp": {"max_context": 64}})
    assert cfg.model_config.max_model_len == 64
    cfg.model_config.max_model_len = 65
    with pytest.raises(ValueError, match="max_model_len.*64"):
        TspPlatform.check_and_update_config(cfg)


def test_the_options_are_validated(model_dir):
    ok = {"weights": "synthetic:1", "kv_persist": "warm", "kv_mode": "paged", "max_context": 128}
    cfg = H.engine_config(model_dir, additional_config={"opentsp": ok})
    for bad in (
        {"kv_mode": "mapped"},
        {"kv_persist": "never"},
        {"max_context": 256},
        {"max_context": 96},
        {"weights": 3},
        {"programs": 3},
        {"colour": "red"},
    ):
        cfg.additional_config = {"opentsp": bad}
        with pytest.raises(ValueError, match="opentsp"):
            TspPlatform.check_and_update_config(cfg)
    cfg.additional_config = {"opentsp": "synthetic:1"}
    with pytest.raises(ValueError, match="opentsp"):
        TspPlatform.check_and_update_config(cfg)
    cfg.additional_config = {}
    TspPlatform.check_and_update_config(cfg)


def test_the_v2_model_runner_is_refused(model_dir, monkeypatch):
    # its scheduler output carries prefill_token_ids and preempted_req_ids that the V1 runner contract does not
    cfg = H.engine_config(model_dir)
    monkeypatch.setattr(type(cfg), "use_v2_model_runner", property(lambda self: True))
    with pytest.raises(ValueError, match="V2 model runner"):
        TspPlatform.check_and_update_config(cfg)


def test_the_whole_engine_config_goes_through_the_hook_in_a_fresh_process(model_dir):
    code = (
        "import vllm_helpers as H, sys;"
        f"c = H.engine_config({str(model_dir)!r});"
        "from vllm.platforms import current_platform as p;"
        "print(type(p).__name__, c.parallel_config.worker_cls, c.cache_config.num_gpu_blocks_override)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**H.env_without_platform(), "OPENTSP_VLLM": "1", "PYTHONPATH": H.TESTS_DIRS},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "TspPlatform opentsp_vllm.worker.TspWorker 128"


def test_a_fresh_process_without_the_variable_keeps_the_cpu_platform_and_its_worker(model_dir):
    code = (
        "import vllm_helpers as H;"
        f"c = H.engine_config({str(model_dir)!r});"
        "print(c.parallel_config.worker_cls)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**H.env_without_platform(), "PYTHONPATH": H.TESTS_DIRS},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] != "opentsp_vllm.worker.TspWorker"


# ---- validate_request ------------------------------------------------------------------------------------------------------------

TOKENS = {"type": "token", "prompt_token_ids": [1, 2, 3]}


def reject(inputs, params, parameter):
    with pytest.raises(VLLMValidationError) as e:
        TspPlatform.validate_request(inputs, params)
    assert e.value.parameter == parameter, e.value
    return str(e.value)


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(),
        SamplingParams(temperature=0.8, top_k=5, top_p=0.9, min_p=0.05, seed=7),
        SamplingParams(logprobs=5),
        SamplingParams(logprobs=-1),
        SamplingParams(n=2),
        SamplingParams(presence_penalty=0.5, frequency_penalty=0.5, repetition_penalty=1.1, logit_bias={5: 1.0}),
        SamplingParams(stop=["x"], stop_token_ids=[3], min_tokens=1, max_tokens=4, ignore_eos=True),
        SamplingParams(bad_words=["a"], allowed_token_ids=[1, 2]),
    ],
)
def test_the_supported_sampling_features_are_accepted(params):
    assert TspPlatform.validate_request(TOKENS, params) is None


def test_prompt_logprobs_are_rejected_with_a_clear_message():
    msg = reject(TOKENS, SamplingParams(prompt_logprobs=1), "prompt_logprobs")
    assert "prompt logprobs" in msg and "TSP" in msg
    reject(TOKENS, SamplingParams(prompt_logprobs=-1), "prompt_logprobs")


def test_structured_outputs_are_rejected():
    reject(TOKENS, SamplingParams(structured_outputs=StructuredOutputsParams(regex="a+")), "structured_outputs")


def test_pooling_requests_are_rejected():
    reject(TOKENS, PoolingParams(), "pooling_params")


@pytest.mark.parametrize(
    "inputs, parameter",
    [
        ({"type": "embeds", "prompt_embeds": object()}, "prompt_embeds"),
        ({"type": "multimodal", "prompt_token_ids": [1], "mm_kwargs": {}}, "multi_modal_data"),
        ({"type": "enc_dec"}, "prompt"),
    ],
)
def test_embedding_multimodal_and_encoder_decoder_inputs_are_rejected(inputs, parameter):
    reject(inputs, SamplingParams(), parameter)


def test_host_tensors_are_not_pinned():
    # every tensor lives on the host, and there is no accelerator to pin for
    from vllm.utils.torch_utils import PIN_MEMORY

    assert TspPlatform.is_pin_memory_available() is False
    assert PIN_MEMORY is False


def test_platform_selects_its_supported_runner_and_preserves_explicit_selection(monkeypatch):
    import os
    from opentsp_vllm.platform import TspPlatform

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    TspPlatform.pre_register_and_update()
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0"
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    TspPlatform.pre_register_and_update()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
