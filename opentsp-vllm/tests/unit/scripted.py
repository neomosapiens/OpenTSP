"""A stand-in for vLLM's scheduler in the unit tests: it builds the `SchedulerOutput`s a real scheduler would send for a
scripted schedule (requests join, are chunked, decode, are left out of a step, are preempted and resumed with new blocks,
finish), and feeds the runner's sampled tokens back, as the engine core does."""

from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

GREEDY = SamplingParams(temperature=0.0, max_tokens=64)


class Driver:
    def __init__(self, runner):
        self.runner = runner
        self.reqs = {}
        self.outputs = []

    def add(self, rid, prompt, blocks, params=None, computed=0):
        self.reqs[rid] = dict(
            prompt=list(prompt),
            out=[],
            blocks=list(blocks),
            computed=computed,
            params=params or GREEDY,
            sent=False,
            resumed=False,
            extra=None,
        )

    def preempt(self, rid):
        """The scheduler frees the blocks and recomputes from position 0 once the request runs again."""
        r = self.reqs[rid]
        r["computed"], r["blocks"], r["resumed"] = 0, None, True

    def grow(self, rid, blocks):
        self.reqs[rid]["extra"] = list(blocks)
        self.reqs[rid]["blocks"] += list(blocks)

    def resume_with(self, rid, blocks, computed=0):
        r = self.reqs[rid]
        r["blocks"], r["computed"] = list(blocks), computed

    def scheduler_output(self, n, finished=(), zero=None, copies=None):
        new, ids, resumed, new_blocks, computed, outs = [], [], set(), [], [], []
        for rid, k in n.items():
            r = self.reqs[rid]
            if not r["sent"]:
                new.append(
                    NewRequestData(
                        req_id=rid,
                        prompt_token_ids=r["prompt"],
                        mm_features=[],
                        sampling_params=r["params"],
                        pooling_params=None,
                        block_ids=(list(r["blocks"]),),
                        num_computed_tokens=r["computed"],
                        lora_request=None,
                    )
                )
                r["sent"], r["extra"] = True, None
            else:
                ids.append(rid)
                if r["resumed"]:
                    resumed.add(rid)
                    new_blocks.append((list(r["blocks"]),))
                elif r["extra"]:
                    new_blocks.append((list(r["extra"]),))
                else:
                    new_blocks.append(None)
                computed.append(r["computed"])
                outs.append(len(r["out"]))
                r["resumed"], r["extra"] = False, None
        cached = CachedRequestData(
            req_ids=ids,
            resumed_req_ids=resumed,
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=new_blocks,
            num_computed_tokens=computed,
            num_output_tokens=outs,
        )
        return SchedulerOutput(
            scheduled_new_reqs=new,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=dict(n),
            total_num_scheduled_tokens=sum(n.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            finished_req_ids=set(finished),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=None if zero is None else list(zero),
            kv_cache_block_copies=None if copies is None else [KVCacheBlockCopy(s, d) for s, d in copies],
        )

    def step(self, n, finished=(), zero=None, copies=None):
        so = self.scheduler_output(n, finished, zero, copies)
        out = self.runner.execute_model(so)
        for rid, k in n.items():
            self.reqs[rid]["computed"] += k
        for rid, toks in zip(out.req_ids, out.sampled_token_ids):
            self.reqs[rid]["out"].extend(toks)
        self.outputs.append(out)
        return out

    def known(self, rid):
        return self.reqs[rid]["prompt"] + self.reqs[rid]["out"]


class Alloc:
    """Block ids 1, 2, ... as a free list hands them out (0 is the null block)."""

    def __init__(self, first=1, limit=127):
        self.free = list(range(first, limit + 1))

    def take(self, k):
        out, self.free = self.free[:k], self.free[k:]
        assert len(out) == k, "block pool exhausted"
        return out

    def give(self, ids):
        self.free += list(ids)


def run(d, targets, alloc, budget=8, per_step=None):
    """Schedule `targets` ({req_id: tokens to generate}) with a lane budget per step, chunked prefill and one decode lane per
    running request, until each has generated its tokens; a finished request is reported on the next step."""
    active, done = [rid for rid, n in targets.items() if len(d.reqs[rid]["out"]) < n], []
    while active:
        n, left = {}, budget
        for rid in active:
            r = d.reqs[rid]
            k = min(len(d.known(rid)) - r["computed"], left, (per_step or {}).get(rid, budget))
            if k == 0:
                continue
            need = -(-(r["computed"] + k) // 16) - len(r["blocks"])
            if need > 0:
                d.grow(rid, alloc.take(need))
            n[rid], left = k, left - k
        d.step(n, finished=done)
        done = []
        for rid in list(active):
            if len(d.reqs[rid]["out"]) >= targets[rid]:
                active.remove(rid)
                done.append(rid)
    return done
