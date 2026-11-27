"""Propagate domain failures through vLLM's per-request error protocol."""

from dataclasses import dataclass

from opentsp_vllm.serving_observer import Trace, configured_trace

from vllm.outputs import RequestError
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus


@dataclass
class InterruptedStep(ModelRunnerOutput):
    reason: str = ""


class TspScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        path = configured_trace()
        self._demo_trace = Trace(path) if path else None

    def schedule(self, *args, **kwargs):
        output = super().schedule(*args, **kwargs)
        if self._demo_trace:
            self._demo_trace.schedule(self, output)
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = self._update_from_output(scheduler_output, model_runner_output)
        if self._demo_trace:
            self._demo_trace.output(scheduler_output, outputs)
            if scheduler_output.num_scheduled_tokens:
                self._demo_trace.settled(self)
        return outputs

    def _update_from_output(self, scheduler_output, model_runner_output):
        if not isinstance(model_runner_output, InterruptedStep):
            return super().update_from_output(scheduler_output, model_runner_output)

        # A preempted request may still have a worker-side token history after all
        # its physical blocks have been freed. The scheduler owns the residency map.
        exposed = set(scheduler_output.num_scheduled_tokens)
        exposed.update(r.request_id for r in self.running)
        exposed.update(r.request_id for r in self.kv_holding_waiting)
        failed = self.finish_requests(exposed, RequestStatus.FINISHED_ERROR)
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        if not self.reset_prefix_cache():
            raise RuntimeError("unable to invalidate the fenced domain's KV prefix cache")
        for request in failed:
            response = outputs.setdefault(request.client_index, EngineCoreOutputs())
            response.outputs.append(EngineCoreOutput(
                request_id=request.request_id, new_token_ids=[], finish_reason=request.get_finished_reason(),
                events=request.take_events(), trace_headers=request.trace_headers,
                error=RequestError(code="opentsp_domain_fault", message=model_runner_output.reason),
            ))
        return outputs
