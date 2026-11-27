import os

from vllm.platforms.interface import Platform, PlatformEnum

from opentsp_vllm import options, ship_plan

# the limits of the design (sec. 1, 3.3, 5.1): the bucket grid and the pool
BLOCK_SIZE = 16
POOL_BLOCKS = 128
MAX_MODEL_LEN = 128
MAX_NUM_SEQS = 4
MAX_NUM_BATCHED_TOKENS = 8
# the SHIP-style GPT-2 small pipeline (`opentsp.chip` = "ship", `ship_chip.py`): the limits are in `options` (the launcher checks them too)
SHIP_MAX_MODEL_LEN = options.SHIP_MAX_MODEL_LEN
SHIP_POOL_BLOCKS = options.SHIP_POOL_BLOCKS
SHIP_MAX_NUM_SEQS = options.SHIP_MAX_NUM_SEQS
SHIP_MAX_NUM_BATCHED_TOKENS = options.SHIP_MAX_NUM_BATCHED_TOKENS
WORKER_CLS = "opentsp_vllm.worker.TspWorker"


def _reject(option, why):
    raise ValueError(f"TspPlatform: unsupported {option}: {why}")


def _check_connectors(vllm_config):
    """No KV connector, whether given as `kv_transfer_config` or derived from `cache_config.kv_offloading_size` (vLLM turns the
    latter into an OffloadingConnector in the config's own post-init, not before this hook reads it)."""
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        _reject("kv_transfer_config (KV connectors)", "the TSP backend does not serve it")
    if getattr(vllm_config.cache_config, "kv_offloading_size", None) is not None:
        _reject("cache_config.kv_offloading_size", "KV offloading needs an offload connector the worker does not serve")


class TspPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "tsp"
    # every torch tensor vLLM creates lives on the host
    device_type = "cpu"
    dispatch_key = "CPU"

    @classmethod
    def pre_register_and_update(cls, parser=None):
        # On Linux the CPU Triton wheel enables vLLM's automatic V2 selection.
        # Both OpenTSP workers consume the V1 scheduling contract.
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
        super().pre_register_and_update(parser)

    @classmethod
    def is_pin_memory_available(cls):
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config):
        """Rejects what the TSP backend does not serve  and fixes what it dictates: the worker, one process,
        synchronous scheduling and the block pool."""
        opts = options.parse(vllm_config.additional_config)
        parallel, scheduler, cache, model = (
            vllm_config.parallel_config,
            vllm_config.scheduler_config,
            vllm_config.cache_config,
            vllm_config.model_config,
        )
        if parallel.world_size != 1:
            _reject("parallel_config.world_size", f"{parallel.world_size}; one chip per engine, so it must be 1")
        if parallel.data_parallel_size != 1:
            _reject("parallel_config.data_parallel_size", f"{parallel.data_parallel_size}; one chip per engine")
        for name, label in (
            ("speculative_config", "speculative decoding"),
            ("lora_config", "LoRA"),
            ("weight_transfer_config", "weight_transfer_config"),
        ):
            if getattr(vllm_config, name, None) is not None:
                _reject(label, "the TSP backend does not serve it")
        _check_connectors(vllm_config)
        if cache.cache_dtype != "auto":
            _reject("cache_config.cache_dtype", f"{cache.cache_dtype!r}; the KV formats are fixed by the kernels, so it must be 'auto'")
        ship = opts["chip"] == "ship"
        if ship:
            max_len, pool_blocks = ship_plan.kv_limits(opts["kv_chunks"])
            max_seqs, max_tokens = SHIP_MAX_NUM_SEQS, SHIP_MAX_NUM_BATCHED_TOKENS
        else:
            max_len, pool_blocks = MAX_MODEL_LEN, POOL_BLOCKS
            max_seqs, max_tokens = MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS
        limit = max_len if ship else min(max_len, opts["max_context"])
        if model is not None:
            if model.runner_type != "generate":
                _reject("model", f"runner type {model.runner_type!r}; only text generation")
            if model.enable_sleep_mode:
                _reject("model_config.enable_sleep_mode", "sleep mode needs device memory to release")
            if model.max_model_len > limit:
                _reject(
                    "model_config.max_model_len",
                    f"max_model_len {model.max_model_len} exceeds {limit}, the longest context of the "
                    + (f"pool ({pool_blocks - 1} pages of 16 slots)" if ship else "programs (additional_config opentsp.max_context)"),
                )
        if scheduler.max_num_seqs > max_seqs:
            _reject("scheduler_config.max_num_seqs", f"max_num_seqs {scheduler.max_num_seqs} exceeds {max_seqs} logit rows")
        if scheduler.max_num_batched_tokens > max_tokens:
            _reject(
                "scheduler_config.max_num_batched_tokens",
                f"max_num_batched_tokens {scheduler.max_num_batched_tokens} exceeds {max_tokens} token lanes",
            )
        if not scheduler.enable_chunked_prefill:
            _reject("scheduler_config.enable_chunked_prefill", "chunked prefill is always on: a prompt is longer than the lanes")
        if vllm_config.use_v2_model_runner:
            _reject("model runner", "the V2 model runner's scheduler output is not the one this worker reads; unset VLLM_USE_V2_MODEL_RUNNER")
        if cache.block_size != BLOCK_SIZE:
            _reject("cache_config.block_size", f"block_size {cache.block_size}; the chip's KV pages are {BLOCK_SIZE} positions")
        # the scheduler's pool: the chip's 128 blocks unless the user asks for fewer ( forces preemption with 6);
        # more cannot be backed by the chip, and fewer than one full request plus the null block cannot admit anything
        pool = cache.num_gpu_blocks_override
        if pool is not None:
            least = -(-(model.max_model_len if model is not None else limit) // BLOCK_SIZE) + 1
            if pool > pool_blocks:
                _reject("cache_config.num_gpu_blocks_override", f"{pool} exceeds the chip's pool of {pool_blocks} blocks")
            if pool < least:
                _reject(
                    "cache_config.num_gpu_blocks_override",
                    f"{pool} cannot hold one request of max_model_len tokens ({least} blocks including the null block)",
                )
        parallel.worker_cls = WORKER_CLS
        parallel.distributed_executor_backend = "uni"
        if opts["chip"] in ("tiny", "ship", "iss"):
            cls = "opentsp_vllm.ras_scheduler.TspScheduler"
            if getattr(scheduler, "scheduler_cls", None) not in (None, cls):
                _reject("scheduler_cls", "the serving domain requires request-level fault containment")
            scheduler.scheduler_cls = cls
        scheduler.async_scheduling = False
        cache.block_size = BLOCK_SIZE
        if pool is None:
            cache.num_gpu_blocks_override = pool_blocks
        _check_connectors(vllm_config)

    @classmethod
    def validate_request(cls, processed_inputs, params):
        """Raises `VLLMValidationError` for the requests the TSP backend does not serve: prompt logprobs, structured outputs,
        pooling, and inputs that are not plain token ids."""
        from vllm.exceptions import VLLMValidationError
        from vllm.pooling_params import PoolingParams

        def reject(parameter, message):
            raise VLLMValidationError(f"{message} (not supported by the TSP backend)", parameter=parameter)

        if isinstance(params, PoolingParams):
            reject("pooling_params", "pooling requests")
        kind = processed_inputs.get("type")
        if kind == "embeds":
            reject("prompt_embeds", "prompt embeddings")
        if kind == "multimodal":
            reject("multi_modal_data", "multimodal inputs")
        if kind != "token":
            reject("prompt", f"inputs of type {kind!r}")
        if params.prompt_logprobs is not None:
            reject("prompt_logprobs", "prompt logprobs need logits for every prompt token")
        if params.structured_outputs is not None:
            reject("structured_outputs", "structured outputs")
