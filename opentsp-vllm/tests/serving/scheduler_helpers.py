"""Helpers of test_prefix_chunked_preempt.py: a probe on the engine core's scheduler (the schedule of every step, the preemptions), a
stepper that adds requests between steps, and the bit-exact reference reference of a request with its logits.

They sit beside 's `engine.Engine` (`steps` records the plan and the chip's result of every step) and change none of it.
"""

import functools

import numpy as np

import engine


class Probe:
    """Records, per scheduler call, `{request_id: tokens scheduled}`, the ids that resumed after a preemption, and counts the
    preemptions. Wraps the scheduler of the engine core once."""

    def __init__(self, e):
        self.scheduler = e.llm.llm_engine.engine_core.engine_core.scheduler
        self.schedules = []  # per scheduler call: (num_scheduled_tokens, resumed_req_ids, new_req_ids)
        self.preemptions = []  # request ids, in order
        schedule, preempt = self.scheduler.schedule, self.scheduler._preempt_request

        def recording_schedule(*args, **kwargs):
            so = schedule(*args, **kwargs)
            self.schedules.append(
                (dict(so.num_scheduled_tokens), set(so.scheduled_cached_reqs.resumed_req_ids), {r.req_id for r in so.scheduled_new_reqs})
            )
            return so

        def recording_preempt(request, *args, **kwargs):
            self.preemptions.append(request.request_id)
            return preempt(request, *args, **kwargs)

        self.scheduler.schedule, self.scheduler._preempt_request = recording_schedule, recording_preempt

    def clear(self):
        self.schedules.clear()
        self.preemptions.clear()


def probe(e):
    if not hasattr(e, "probe"):
        e.probe = Probe(e)
    e.probe.clear()
    return e.probe


def run(e, arrivals, params, limit=600):
    """Drives the engine one step at a time. `arrivals` maps a step number to the `(name, prompt)` requests added before that step;
    returns `{name: RequestOutput}` of the finished requests. `params` is a SamplingParams, or a dict name -> SamplingParams."""
    engine_ = e.llm.llm_engine
    finished, last = {}, max(arrivals)
    for step in range(limit):
        for name, prompt in arrivals.get(step, ()):
            sp = params[name] if isinstance(params, dict) else params
            engine_.add_request(name, {"prompt_token_ids": list(prompt)}, sp)
        if not engine_.has_unfinished_requests() and step > last:
            return finished
        for out in engine_.step():
            if out.finished:
                finished[out.request_id] = out
    raise AssertionError("the engine did not finish")


def logits_by_request(e):
    """The chip's logit rows of each request over the recorded steps: `{internal request id: (n, vocab) array}`."""
    rows = {}
    for plan, result in e.steps:
        for i, rid in enumerate(plan.logit_req_ids):
            rows.setdefault(rid, []).append(np.asarray(result.logits)[i])
    return {rid: np.stack(r) for rid, r in rows.items()}


def reference_reference(prompt, n, model="test"):
    tokens, logits = _reference_reference(tuple(prompt), n, model)
    return list(tokens), logits


@functools.lru_cache(maxsize=None)
def _reference_reference(prompt, n, model):
    """`paged_reference.generate_paged` of the request alone, greedy: (generated ids, (n, vocab) logits of the steps that produced
    them), on the synthetic weights of seed 1."""
    reference_paged, weights, _ = engine.reference_loader.load()
    from opentsp_vllm.models.tinystories import layout

    w = engine._weights(model, layout, weights)
    C = 64 if len(prompt) + n <= 64 else 128
    gen = reference_paged.generate_paged(w, prompt, n, C=C)
    # the last token is sampled, never fed: the engine runs one step fewer than `generate_paged`
    rows = [np.asarray(s.logits, dtype=np.float32) for s in gen.steps if s.logits is not None][:n]
    return tuple(gen.tokens[len(prompt):]), np.stack(rows)


def name_of(internal_id):
    """The name a request was added under: the engine appends `-<8 hex digits>`."""
    return internal_id.rsplit("-", 1)[0]


def check_rows_against_reference(e, requests, model="test"):
    """Every logit row the chip returned in the recorded steps equals, bit for bit, bit-exact reference's row of the same token of the same request
    alone, whatever the batch, the chunk, the bucket or the preemptions of the run. `requests` is `{name: (prompt, n_new)}`.
    Returns the number of rows compared per request."""
    refs = {name: reference_reference(prompt, n, model)[1] for name, (prompt, n) in requests.items()}
    compared = dict.fromkeys(requests, 0)
    for plan, result in e.steps:
        for i, rid in enumerate(plan.logit_req_ids):
            lane = plan.lanes[i]
            name = name_of(rid)
            assert lane.req_id == rid and lane.logits, "the first lanes of a plan are its logit lanes, in the order of the rows"
            token_index = lane.position + 1 - len(requests[name][0])
            assert 0 <= token_index < requests[name][1], (name, token_index)
            assert np.array_equal(np.asarray(result.logits)[i], refs[name][token_index]), f"{name}: logits of token {token_index}"
            compared[name] += 1
    return compared


def lanes_of(e, name):
    """The real lanes of a request over the recorded steps."""
    return sum(1 for plan, _ in e.steps for ln in plan.real_lanes if ln.req_id and name_of(ln.req_id) == name)
