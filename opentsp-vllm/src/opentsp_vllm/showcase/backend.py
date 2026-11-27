"""The vLLM side of the showcase: an `LLM` on the TSP platform (the chip behind it is `IssChip`, or a test chip) driven step by step through
its engine, so that every token is available the moment its step returns. vLLM's CPU does the tokenizing, scheduling, sampling and
detokenizing; the model runs on the chip.

The platform plugin and the in-process engine core are decided by environment variables that must be set before vLLM is imported:
`configure_environment` does it, and `build_llm` calls it."""

import os
import pathlib

from .session import Event, StepInfo


def configure_environment():
    os.environ["OPENTSP_VLLM"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


def paths(root):
    root = pathlib.Path(root)
    return dict(model=root / "model", programs=root / "serving")


def check_prepared(root):
    """The paths of a prepared showcase directory; SystemExit with the remedy when something is missing."""
    p = paths(root)
    for key, probe in (("model", "config.json"), ("programs", "manifest.json")):
        target = p[key] / probe if probe else p[key]
        if not target.exists():
            raise SystemExit(f"the showcase is not prepared: {target} is missing; run `just showcase-prepare` (builds the weight package and the programs once per checkpoint and generator version)")
    return p


def opentsp_options(p, **extra):
    return {"chip": "tiny", "programs": str(p["programs"]), "max_context": 128, **extra}


def build_llm(model, options, **overrides):
    """The `LLM` of the showcase: 4 sequences, 8 token lanes per step, chunked prefill, prefix caching, the model's own tokenizer."""
    configure_environment()
    from vllm import LLM

    kwargs = dict(
        model=str(model),
        dtype="float32",
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=8,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
        additional_config={"opentsp": options},
    )
    kwargs.update(overrides)
    return LLM(**kwargs)


class VllmBackend:
    """`session.Backend` over an `LLM` whose engine core runs in this process."""

    def __init__(self, llm):
        self.llm = llm
        self.engine = llm.llm_engine
        self.tokenizer = llm.get_tokenizer()
        self.max_len = self.engine.model_config.max_model_len
        self.bos = int(self.engine.model_config.hf_config.bos_token_id)
        core = self.engine.engine_core.engine_core
        self.worker = core.model_executor.driver_worker.worker

    def encode(self, text):
        return self.tokenizer.encode(text)

    def submit(self, rid, token_ids, sampling):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        params = SamplingParams(output_kind=RequestOutputKind.DELTA, **sampling)
        self.engine.add_request(rid, {"prompt_token_ids": list(token_ids)}, params)

    def busy(self):
        return self.engine.has_unfinished_requests()

    def abort(self, rid):
        self.engine.abort_request([rid])

    def step(self):
        stats = self.worker.model_runner.stats
        steps = stats.steps
        outputs = self.engine.step()
        info = None
        if stats.steps > steps:
            last = stats.last
            info = StepInfo(last["bucket"], last["real_lanes"], last["padding_lanes"], last["cycles"], last["chip_s"])
        events = []
        for out in outputs:
            c = out.outputs[0]
            events.append(Event(out.request_id, c.text, tuple(c.token_ids), out.finished, c.finish_reason if out.finished else None, out.num_cached_tokens))
        return events, info

    def reset_prefix_cache(self):
        self.llm.reset_prefix_cache()

    def close(self):
        self.engine.engine_core.shutdown()
