"""TspModelRunner (request state, step plan, chip run, sampler). .

The request-state logic follows `GPUModelRunner._update_states` of vLLM 0.33.0 (v1/worker/gpu_model_runner.py) for what a
single-rank, synchronous, text-only worker meets; the sampling follows its `_sample` and `_bookkeeping_sync`.
"""

import time

import torch
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from opentsp_vllm import options
from opentsp_vllm.contracts import DomainFault, PlanRequest
from opentsp_vllm.serving_observer import configured_trace, emit, request_label
from opentsp_vllm.step_plan import BLOCK, plan_step
from opentsp_vllm.stats import Stats

logger = init_logger(__name__)

CPU = torch.device("cpu")
STATS_LOG_INTERVAL_S = 10.0


class HostTopKTopP(torch.nn.Module):
    """Filter host logits with vLLM's PyTorch kernel before its native seeded sampler."""

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def forward(self, logits, generators, k, p):
        # The upstream automatic filter selects a device Triton kernel on Linux.
        # The two None arguments tell the sampler the logits are already filtered.
        logits = apply_top_k_top_p_pytorch(logits, k, p)
        return self.sampler.forward_native(logits, generators, None, None)


class TspModelRunner:
    def __init__(self, vllm_config, chip):
        self.vllm_config = vllm_config
        self.chip = chip
        self.model_config = vllm_config.model_config
        scheduler = vllm_config.scheduler_config
        info = chip.info
        if self.model_config.get_vocab_size() != info.vocab:
            raise ValueError(
                f"the model's vocab size {self.model_config.get_vocab_size()} differs from the chip's {info.vocab}: "
                "the model directory and the weights package are of different models"
            )
        self.vocab = info.vocab
        self.grid_c = tuple(info.grid_c)
        self.max_model_len = self.model_config.max_model_len
        self.requests = {}
        self.stats = Stats()
        self.failure = None
        self.input_batch = InputBatch(
            max_num_reqs=scheduler.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=scheduler.max_num_batched_tokens,
            device=CPU,
            vocab_size=self.vocab,
            block_sizes=[BLOCK],
            max_num_blocks_per_req=[cdiv(self.max_model_len, BLOCK)],
            logitsprocs=build_logitsprocs(vllm_config, CPU, PIN_MEMORY, False),
        )
        self.sampler = Sampler(
            logprobs_mode=self.model_config.logprobs_mode, use_fp64_gumbel=self.model_config.use_fp64_gumbel
        )
        self.sampler.topk_topp_sampler = HostTopKTopP(self.sampler.topk_topp_sampler)
        self.options = options.parse(vllm_config.additional_config)
        self._last_log = time.monotonic()
        self._demo_trace = configured_trace()

    # -- one step -----------------------------------------------------------------------------------------------------------

    def execute_model(self, scheduler_output):
        """: request state, block copies and zeroing, step plan, chip run, sampling, output. Never returns None."""
        t_start = time.perf_counter()
        try:
            self._update_states(scheduler_output)
            if not scheduler_output.total_num_scheduled_tokens:
                return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])
            plan = self._plan(scheduler_output)
            if hasattr(scheduler_output, "opentsp_demo_step"):
                plan.demo_step = scheduler_output.opentsp_demo_step
            t_chip = time.perf_counter()
            result = self.chip.step(plan)
            t_sample = time.perf_counter()
            output = self._sample(plan, result)
            if self._demo_trace is not None and plan.demo_step is not None:
                spans = [dict(id=request_label(request_id), ids=tokens,
                              ordinal=len(self.requests[request_id].output_token_ids) - len(tokens) + 1)
                         for request_id, tokens in zip(output.req_ids, output.sampled_token_ids, strict=True)
                         if tokens]
                if spans:
                    emit(self._demo_trace, dict(phase="sampled", step=plan.demo_step, tokens=spans))
            t_end = time.perf_counter()
        except DomainFault as e:
            from .ras_scheduler import InterruptedStep
            # The scheduler owns residency, terminal notifications and KV invalidation.
            self.failure = self.chip.failure
            return InterruptedStep(req_ids=[], req_id_to_index={}, sampled_token_ids=[], reason=str(e))
        except BaseException as e:
            self.failure = f"{type(e).__name__}: {e}"
            raise
        self.stats.record(
            bucket=plan.bucket,
            real_lanes=plan.n_real,
            padding_lanes=plan.padding_lanes,
            context_padding=plan.context_padding,
            cycles=result.cycles,
            patch_count=plan.patch_count,
            host_prep_s=t_chip - t_start,
            chip_s=t_sample - t_chip,
            sampler_s=t_end - t_sample,
        )
        self._log_stats(t_end)
        return output

    def _log_stats(self, now):
        if now - self._last_log < STATS_LOG_INTERVAL_S:
            return
        self._last_log = now
        s = self.stats.snapshot()
        logger.info(
            "TSP steps=%d lanes=%d padding_lanes=%d host_prep_s=%.3f chip_s=%.3f sampler_s=%.3f zeroed=%d copied=%d",
            s["steps"], s["real_lanes"], s["padding_lanes"], s["host_prep_s"], s["chip_s"], s["sampler_s"],
            s["blocks_zeroed"], s["blocks_copied"],
        )

    # -- request state ( step 1) ---------------------------------------------------------------------------------------

    def _update_states(self, so):
        batch = self.input_batch
        for req_id in so.finished_req_ids:
            self.requests.pop(req_id, None)
        # a finished id that is submitted again in the same step is a new request (abort, then resubmission)
        for req_id in so.finished_req_ids:
            batch.remove_request(req_id)

        # block work before the run: new blocks are zeroed, then copies are applied
        if so.new_block_ids_to_zero:
            self.chip.zero_blocks(so.new_block_ids_to_zero)
            self.stats.blocks_zeroed += len(so.new_block_ids_to_zero)
        if so.kv_cache_block_copies:
            self.chip.copy_blocks([(c.src_block_id, c.dst_block_id) for c in so.kv_cache_block_copies])
            self.stats.blocks_copied += len(so.kv_cache_block_copies)

        # preempted and merely unscheduled requests leave the persistent batch and keep their state; a resumed id that is
        # still in the batch (reset_prefix_cache) is cleared first, so it re-enters through the resumed path
        scheduled = so.num_scheduled_tokens.keys()
        resumed = so.scheduled_cached_reqs.resumed_req_ids
        for req_id in batch.req_id_to_index.keys() - (scheduled - resumed):
            batch.remove_request(req_id)

        to_add = []
        for new in so.scheduled_new_reqs:
            if new.req_id in self.requests:
                raise RuntimeError(f"request {new.req_id}: resumable (streaming input) requests are not supported")
            params = new.sampling_params
            generator = None
            if params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=CPU)
                generator.manual_seed(params.seed)
            state = CachedRequestState(
                req_id=new.req_id,
                prompt_token_ids=new.prompt_token_ids,
                mm_features=[],
                sampling_params=params,
                generator=generator,
                block_ids=new.block_ids,
                num_computed_tokens=new.num_computed_tokens,
                output_token_ids=[],
            )
            self.requests[new.req_id] = state
            to_add.append(state)

        cached = so.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            state = self.requests[req_id]
            state.num_computed_tokens = cached.num_computed_tokens[i]
            new_block_ids = cached.new_block_ids[i]
            num_output = cached.num_output_tokens[i]
            index = batch.req_id_to_index.get(req_id)
            if num_output < len(state.output_token_ids):
                # the scheduler discarded outputs (a failed load): align
                del state.output_token_ids[num_output:]
                if index is not None:
                    batch.num_tokens_no_spec[index] = batch.num_prompt_tokens[index] + num_output
            if req_id not in cached.resumed_req_ids:
                if new_block_ids is not None:
                    for ids, new_ids in zip(state.block_ids, new_block_ids):
                        ids.extend(new_ids)
            else:
                assert index is None and new_block_ids is not None
                state.block_ids = new_block_ids
            if index is None:
                to_add.append(state)
                continue
            batch.num_computed_tokens_cpu[index] = state.num_computed_tokens
            if new_block_ids is not None:
                batch.block_table.append_row(new_block_ids, index)

        for state in to_add:
            batch.add_request(state)
        batch.condense()
        batch.refresh_metadata()

    # -- the step plan and the chip run (steps 3 and 4) ---------------------------------------------------------------------------

    def _plan(self, so):
        requests = []
        for req_id in self.input_batch.req_ids:
            state = self.requests[req_id]
            requests.append(
                PlanRequest(
                    req_id,
                    list(state.prompt_token_ids) + state.output_token_ids,
                    state.block_ids[0],
                    state.num_computed_tokens,
                    so.num_scheduled_tokens[req_id],
                )
            )
        planner = getattr(self.chip, "plan_step", None)  # a chip with its own lanes and inputs (`ShipChip`) plans its own step
        if planner is not None:
            return planner(requests)
        return plan_step(requests, vocab=self.vocab, grid_c=self.grid_c)

    # -- sampling and output (steps 5 and 6) ----------------------------------------------------------------------------------------

    def _sample(self, plan, result):
        batch = self.input_batch
        num_reqs = batch.num_reqs
        if not plan.n_logits:
            return ModelRunnerOutput(
                req_ids=batch.req_ids.copy(),
                req_id_to_index=batch.req_id_to_index.copy(),
                sampled_token_ids=[[] for _ in range(num_reqs)],
            )
        rows = [batch.req_id_to_index[r] for r in plan.logit_req_ids]
        # the Sampler scales logits in place (temperature, penalties): it gets a copy, the chip's array stays as returned
        chip_logits = torch.from_numpy(result.logits).clone()
        if rows == list(range(num_reqs)):
            logits = chip_logits
        else:
            # the sampler's metadata is indexed by batch row: the requests that need no token get a row of zeros, discarded below
            logits = torch.zeros((num_reqs, self.vocab), dtype=torch.float32)
            logits[rows] = chip_logits
        discarded = [i for i in range(num_reqs) if i not in set(rows)]
        # a discarded row must not consume its request's seeded random stream (the stock runner rewinds the offset, which a
        # CPU generator does not have)
        saved = {i: batch.generators[i].get_state() for i in discarded if i in batch.generators}
        out = self.sampler(logits=logits, sampling_metadata=batch.sampling_metadata)
        for i, state in saved.items():
            batch.generators[i].set_state(state)
        sampled = out.sampled_token_ids.tolist()
        for i in discarded:
            sampled[i] = []
        logprobs = out.logprobs_tensors.tolists() if out.logprobs_tensors is not None else None
        for i, toks in enumerate(sampled):
            if not toks:
                continue
            start = int(batch.num_tokens_no_spec[i])
            end = start + len(toks)
            assert end <= self.max_model_len, f"sampled token ids exceed max_model_len: {end} > {self.max_model_len}"
            batch.token_ids_cpu[i, start:end] = toks
            batch.is_token_ids[i, start:end] = True
            batch.num_tokens_no_spec[i] = end
            self.requests[batch.req_ids[i]].output_token_ids.extend(toks)
        return ModelRunnerOutput(
            req_ids=batch.req_ids.copy(),
            req_id_to_index=batch.req_id_to_index.copy(),
            sampled_token_ids=sampled,
            logprobs=logprobs,
        )
