"""The worker behind vLLM's own Scheduler (no tokenizer, no engine process): the SchedulerOutputs are the real ones, so a claim of
 about them (prefix hits, chunked prefill, preemption by recompute, abort, no copies and no zeroing for
GPT-2) is checked against 0.33.0 itself. The worker runs on ReferenceChip."""

import pytest

import reference_loader as oracle
import vllm_helpers as H
from opentsp_vllm.worker import TspWorker
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

reference_paged, W, config = oracle.load()
CFG = config.ModelConfig(layers=2, vocab=2000, max_pos=128)
OPTIONS = {"chip": "reference_chip:make_chip", "weights": "synthetic:1", "max_context": 64}


@pytest.fixture(scope="module")
def vllm_config(tmp_path_factory):
    return H.engine_config(H.write_model_dir(tmp_path_factory.mktemp("rs")), additional_config={"opentsp": OPTIONS})


@pytest.fixture(scope="module")
def weights():
    return W.synthetic(CFG, seed=1)


def prompt(seed, n):
    import numpy as np

    return [int(t) for t in np.random.default_rng(seed).integers(1, CFG.vocab, n)]


class Engine:
    """The engine core's loop without its process: schedule, execute_model, update_from_output."""

    def __init__(self, vllm_config, num_blocks=128):
        self.worker = TspWorker(vllm_config, 0, 0, "tcp://127.0.0.1:1", True)
        self.worker.init_device()
        self.worker.load_model()
        specs = self.worker.get_kv_cache_spec()
        spec = next(iter(specs.values()))
        kv = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[KVCacheTensor(size=1, layers=[n], layer_stride=1, block_stride=1) for n in specs],
            kv_cache_groups=[KVCacheGroupSpec(list(specs), spec)],
        )
        self.worker.initialize_from_config(kv)
        vllm_config.cache_config.num_gpu_blocks = num_blocks  # the engine core records the pool size it sized
        self.scheduler = Scheduler(vllm_config, kv, StructuredOutputManager(vllm_config), block_size=16)
        fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
        init_none_hash(fn)
        self.hasher = get_request_block_hasher(16, fn)
        self.requests, self.outputs, self.so_log = {}, {}, []

    def add(self, rid, prompt_ids, max_tokens=6, **params):
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, **params)
        req = Request(rid, prompt_ids, sp, None, block_hasher=self.hasher)
        self.requests[rid], self.outputs[rid] = req, []
        self.scheduler.add_request(req)

    def step(self):
        so = self.scheduler.schedule()
        out = self.worker.execute_model(so)
        eco = self.scheduler.update_from_output(so, out)
        for outs in eco.values():
            for o in outs.outputs:
                self.outputs[o.request_id].extend(o.new_token_ids)
        self.so_log.append(so)
        return so

    def run(self, limit=400):
        for _ in range(limit):
            if not self.scheduler.has_requests():
                return
            self.step()
        raise AssertionError("the engine did not finish")


def solo(weights, p, n):
    return reference_paged.generate_paged(weights, p, n, C=64).tokens[len(p):]


def test_requests_through_the_real_scheduler_equal_their_solo_oracle_runs(vllm_config, weights):
    e = Engine(vllm_config)
    ps = {"a": prompt(1, 30), "b": prompt(2, 6), "c": prompt(3, 17)}
    for rid, p in ps.items():
        e.add(rid, p, max_tokens=7)
    e.run()
    for rid, p in ps.items():
        assert e.outputs[rid] == solo(weights, p, 7), rid
    # the scheduler chunked the 30-token prompt under the 8-token budget and decoded the requests together
    assert max(pl.n_logits for pl in e.worker.chip.plans) >= 2
    assert max(pl.n_real for pl in e.worker.chip.plans) <= 8


def test_gpt2_scheduler_outputs_carry_no_block_copies_and_no_zeroing(vllm_config):
    e = Engine(vllm_config)
    e.add("a", prompt(4, 20))
    e.add("b", prompt(5, 20))
    e.run()
    assert all(so.kv_cache_block_copies is None and not so.new_block_ids_to_zero for so in e.so_log)
    assert e.worker.get_tsp_stats()["blocks_copied"] == 0 and e.worker.get_tsp_stats()["blocks_zeroed"] == 0


def test_a_prefix_cache_hit_skips_the_shared_blocks_and_equals_the_cold_run(vllm_config, weights):
    e = Engine(vllm_config)
    shared = prompt(6, 40)
    e.add("cold", shared + [7, 8], max_tokens=5)
    e.run()
    plans_before = len(e.worker.chip.plans)
    e.add("hit", shared + [7, 8], max_tokens=5)
    e.run()
    assert e.outputs["hit"] == e.outputs["cold"] == solo(weights, shared + [7, 8], 5)
    first = e.worker.chip.plans[plans_before]  # two full blocks of the 42-token prompt are cached
    assert min(ln.position for ln in first.real_lanes) == 32, "the hit recomputes only what follows the cached blocks"


def test_preemption_by_recompute_gives_every_request_its_solo_tokens(vllm_config, weights):
    e = Engine(vllm_config, num_blocks=6)  # 5 usable blocks for three requests that grow to 34 tokens (3 blocks each)
    ps = {rid: prompt(10 + i, 14) for i, rid in enumerate("abc")}
    for rid, p in ps.items():
        e.add(rid, p, max_tokens=20)
    e.run()
    resumed = [rid for so in e.so_log for rid in so.scheduled_cached_reqs.resumed_req_ids]
    assert resumed, "no request was preempted: the pool is too large for the test"
    for rid, p in ps.items():
        assert e.outputs[rid] == solo(weights, p, 20), rid


def test_five_requests_over_the_four_sequence_limit_each_equal_their_solo_run(vllm_config, weights):
    # max_num_seqs is 4 and the budget 8 lanes: the fifth request waits, requests join and leave the batch
    e = Engine(vllm_config)
    ps = {f"r{i}": prompt(20 + i, 12) for i in range(5)}
    for rid, p in ps.items():
        e.add(rid, p, max_tokens=5)
    e.run()
    for rid, p in ps.items():
        assert e.outputs[rid] == solo(weights, p, 5), rid


def test_abort_drops_the_request_state_and_leaves_the_others_unaffected(vllm_config, weights):
    e = Engine(vllm_config)
    pa, pb = prompt(30, 10), prompt(31, 10)
    e.add("a", pa, max_tokens=8)
    e.add("b", pb, max_tokens=8)
    for _ in range(3):
        e.step()
    e.scheduler.finish_requests("a", RequestStatus.FINISHED_ABORTED)
    e.run()
    runner = e.worker.model_runner
    assert "a" not in runner.requests and "b" not in runner.requests, "finished requests are dropped"
    assert e.outputs["b"] == solo(weights, pb, 8)


def test_the_config_hook_keeps_a_preempting_pool_and_a_scheduler_built_from_the_final_config_preempts(tmp_path):
    """The path of the design's preemption scenario (sec. 6.6): EngineArgs(num_gpu_blocks_override=6) through the
    platform hook, then the real Scheduler on the pool the final config names, over the same chip."""
    cfg = H.engine_config(
        H.write_model_dir(tmp_path / "m"), num_gpu_blocks_override=6, additional_config={"opentsp": OPTIONS}
    )
    pool = cfg.cache_config.num_gpu_blocks_override
    assert pool == 6
    e = Engine(cfg, num_blocks=pool)
    ps = {rid: prompt(40 + i, 14) for i, rid in enumerate("abc")}
    for rid, p in ps.items():
        e.add(rid, p, max_tokens=20)
    e.run()
    assert [rid for so in e.so_log for rid in so.scheduled_cached_reqs.resumed_req_ids], "no request was preempted"
    for rid, p in ps.items():
        assert e.outputs[rid] == solo(W.synthetic(CFG, seed=1), p, 20), rid
