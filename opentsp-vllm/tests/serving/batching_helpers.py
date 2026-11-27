"""Helpers of tests/serving/test_batching_stops.py : drive the engine step by step with requests that join and leave, record
the logits every step returns per request, and compute the bit-exact reference of a request.

The recording wraps `chip.step` of the engine's in-process worker, so it works for `ReferenceChip` and `IssChip` alike."""

import copy
import os
from dataclasses import dataclass, field
from functools import cache

import numpy as np

import reference_loader as oracle

VOCAB = 50257
EOS = 50256

def tokenizer_kwargs():
    """`LLM` arguments for an engine that detokenizes: the GPT-2 tokenizer of the Hugging Face cache (the test model's tokenizer)."""
    import pathlib

    from huggingface_hub import snapshot_download

    path = snapshot_download("gpt2", allow_patterns=["tokenizer*", "vocab.json", "merges.txt"], local_files_only=True)
    return dict(skip_tokenizer_init=False, tokenizer=str(pathlib.Path(path)))


@cache
def _oracle():
    paged_reference, weights, config = oracle.load()
    return paged_reference, weights.synthetic(config.TINYSTORIES_L2, seed=1)


def reference_run(prompt, n_new, pick=None, stop=None):
    """Greedy generation on bit-exact reference's paged step (the reference of `generate_paged`), with a hook: `pick(logits, generated) -> token` chooses
    each token (default: the first maximum), `stop(generated) -> bool` ends the run after a token. Returns (tokens, logit rows); one
    row per token, the row that produced it."""
    reference_paged, w = _oracle()
    C = 64
    pool = reference_paged.BlockPool(w.cfg, C // 16 + 1)
    table = [pool.allocate() for _ in range(C // 16)]
    pick = pick or (lambda logits, generated: reference_paged.pick_token(logits))
    seq, out, rows = [int(t) for t in prompt], [], []
    pos = 0
    while len(out) < n_new:
        last = pos >= len(seq) - 1
        r = reference_paged.decode_step_paged(w, pool, seq[pos], pos, table, C=C, logits=last)
        if last:
            rows.append(np.array(r.logits, dtype=np.float32))
            tok = int(pick(rows[-1], out))
            out.append(tok)
            seq.append(tok)
            if stop is not None and stop(out):
                break
        pos += 1
    return out, rows


def reference_tokens(prompt, n):
    reference_paged, w = _oracle()
    return reference_paged.generate_paged(w, prompt, n, C=64).tokens[len(prompt) :]


def same_bits(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.int32), b.view(np.int32))


@dataclass
class Result:
    tokens: list = field(default_factory=list)
    text: str = ""
    deltas: list = field(default_factory=list)  # (token ids, text) of each output the engine returned
    texts: list = field(default_factory=list)  # the text after each output (cumulative kinds)
    finish_reason: object = None
    stop_reason: object = None
    first_step: int = None  # the chip step (of the drive) of the request's first lane
    finish_step: int = None  # the engine step that returned its last output
    rows: list = field(default_factory=list)  # the chip's logit row of every token
    aborted: bool = False
    num_cached_tokens: object = None


class Rig:
    """The engine of a serving `LLM` with the in-process core, driven step by step."""

    def __init__(self, serving):
        """`serving` is what the `make_llm` fixture returns (an `engine.Engine`)."""
        self.serving = serving
        self.llm = llm = serving.llm
        self.engine = llm.llm_engine
        self.core = self.engine.engine_core.engine_core
        self.worker = self.core.model_executor.driver_worker.worker
        self.tok = llm.get_tokenizer()
        self.log = []  # (internal request ids of the logit rows, rows, internal request ids of the real lanes) of every chip step
        chip = self.worker.chip
        if not getattr(chip, "_batching_test_wrapped", False):
            inner = chip.step

            def step(plan):
                result = inner(plan)
                self.log.append(
                    (
                        tuple(plan.logit_req_ids),
                        np.array(result.logits, dtype=np.float32, copy=True),
                        tuple(lane.req_id for lane in plan.real_lanes),
                    )
                )
                return result

            chip.step = step
            chip._batching_test_wrapped = True
        self.last_log = []

    # -- the engine's state

    def kv_usage(self):
        return self.core.scheduler.kv_cache_manager.usage

    def runner_requests(self):
        return dict(self.worker.model_runner.requests)

    def idle(self):
        return not self.engine.has_unfinished_requests() and not self.core.scheduler.has_unfinished_requests()

    # -- one drive

    def drive(self, arrivals, aborts=None, limit=400):
        """Run requests to completion. `arrivals`: (id, prompt ids, SamplingParams, step) tuples, the request is added before the
        engine's `step`-th step (0 is the first); `aborts`: {id: step} aborts before that step. Returns {id: Result}.
        The engine must be idle before and is idle after, with the KV usage back at 0."""
        assert self.idle(), "the engine is not idle"
        assert len({a[0] for a in arrivals}) == len(arrivals)
        aborts = dict(aborts or {})
        results = {a[0]: Result() for a in arrivals}
        kinds = {}
        log0 = len(self.log)
        pending = sorted(arrivals, key=lambda a: a[3])
        step = 0
        while pending or self.engine.has_unfinished_requests():
            if not self.engine.has_unfinished_requests() and pending and pending[0][3] > step:
                step = pending[0][3]  # nothing runs: jump to the next arrival
            while pending and pending[0][3] <= step:
                rid, prompt, params, _ = pending.pop(0)
                params = copy.deepcopy(params)
                kinds[rid] = params.output_kind
                self.engine.add_request(rid, {"prompt_token_ids": list(prompt)}, params)
            for rid in [r for r, s in aborts.items() if s <= step]:
                del aborts[rid]
                self.engine.abort_request([rid])
                if rid in results:
                    results[rid].aborted = True
            for out in self.engine.step():
                res = results[out.request_id]
                comp = out.outputs[0]
                res.deltas.append((list(comp.token_ids), comp.text))
                if str(kinds[out.request_id]).endswith("DELTA"):
                    res.tokens.extend(comp.token_ids)
                    res.text += comp.text
                else:
                    res.tokens = list(comp.token_ids)
                    res.text = comp.text
                res.texts.append(res.text)
                res.num_cached_tokens = out.num_cached_tokens
                if out.finished:
                    res.finish_reason, res.stop_reason, res.finish_step = comp.finish_reason, comp.stop_reason, step
            step += 1
            assert step < limit, "the engine did not finish"
        for index, (ids, rows, lanes) in enumerate(self.log[log0:]):
            for rid in dict.fromkeys(lanes):
                res = results[rid.rsplit("-", 1)[0]]
                if res.first_step is None:
                    res.first_step = index
            for rid, row in zip(ids, rows):
                results[rid.rsplit("-", 1)[0]].rows.append(row)
        self.last_log = self.log[log0:]
        assert self.idle()
        assert self.kv_usage() == 0, f"the KV usage is {self.kv_usage()} after every request ended"
        return results

    def step_records(self, since=0):
        return self.log[since:]

    def solo(self, prompt, params, rid="solo"):
        self.llm.reset_prefix_cache()
        return self.drive([(rid, prompt, params, 0)])[rid]


# -- `vllm serve` over HTTP (the streaming API): the server is its own process, as the engine core of an AsyncLLM must be


class Server:
    """A `vllm serve` process on `chip` with the test model and its tokenizer; `stream` yields the parsed events of a streamed
    completion, `complete` returns a whole one, `metric` reads a gauge of /metrics."""

    def __init__(self, chip, root):
        import json
        import socket
        import subprocess
        import sys
        import time
        import urllib.request

        import engine

        self.path = engine.model_dir(root, "test")
        kw = engine.engine_kwargs(self.path, chip, "test", root=root)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = engine.serve_env()
        env.pop("VLLM_ENABLE_V1_MULTIPROCESSING")
        tok = tokenizer_kwargs()["tokenizer"]
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", kw["model"], "--tokenizer", tok, "--dtype", "float32",
            "--max-model-len", str(kw["max_model_len"]), "--max-num-seqs", "4", "--max-num-batched-tokens", "8", "--port", str(port),
            "--additional-config", json.dumps(kw["additional_config"]),
        ]  # fmt: skip
        self.log_path = os.path.join(root, f"serve-{chip}.log")
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 170
        while True:
            assert self.proc.poll() is None, f"the server exited ({self.proc.returncode}):\n{open(self.log_path).read()[-3000:]}"
            try:
                urllib.request.urlopen(self.base + "/health", timeout=2)
                break
            except OSError:
                assert time.monotonic() < deadline, "the server did not become healthy in 170 s"
                time.sleep(0.5)

    def close(self):
        import subprocess

        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def _request(self, body):
        import json
        import urllib.request

        data = json.dumps(dict(body, model=str(self.path), temperature=0, ignore_eos=True)).encode()
        return urllib.request.Request(self.base + "/v1/completions", data, {"content-type": "application/json"})

    def complete(self, prompt, **body):
        import json
        import urllib.request

        return json.load(urllib.request.urlopen(self._request(dict(body, prompt=prompt)), timeout=120))

    def stream(self, prompt, stop_after=None, **body):
        """The events (dicts) of a streamed completion; with `stop_after` the client closes the connection after that many events."""
        import json
        import urllib.request

        response = urllib.request.urlopen(self._request(dict(body, prompt=prompt, stream=True)), timeout=120)
        events = []
        try:
            for line in response:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                events.append(json.loads(payload))
                if stop_after and len(events) >= stop_after:
                    break
        finally:
            response.close()
        return events

    def metric(self, name, label=None):
        """The sum of the samples of `name` (whose labels contain `label`, e.g. 'finished_reason="abort"') in /metrics."""
        import urllib.request

        total, found = 0.0, False
        for line in urllib.request.urlopen(self.base + "/metrics", timeout=10).read().decode().splitlines():
            if (line.startswith(name + " ") or line.startswith(name + "{")) and (label is None or label in line):
                total, found = total + float(line.rsplit(" ", 1)[1]), True
        if not found:
            raise KeyError(f"{name} {label or ''}")
        return total
