"""The per-step host cost of the vLLM serving path against the step's hardware time (a proof of concept; no package code changed).

One engine of the full model (8 layers) on the ISS chip serves N_REQ requests one after the other (prompt of PROMPT_LEN token ids, NEW new
tokens, greedy): PROMPT_LEN / 8 prefill steps of the bucket P(8,1,64) and NEW - 1 decode steps of P(1,1,64) each. Every step is timed in its
parts by wrappers installed from here (the package is not touched), then replayed once, at once, as the baseline: the same plan straight
through `IssChip.step` (no vLLM, no runner, no sampler), so the cost of the wrapper and the binding alone is measured in the same process
and at the same load. Serve and baseline of one step are neighbours, which is the interleaving of the protocol.

    python3 host_cost.py [--requests 8] [--prompt-len 32] [--new 8] [--out DIR]

Prints the table and writes `host_cost-<commit>.json` (every step's record, the summary, the loads) into `--out` (default `tests/bench/out`).
Needs the vLLM environment and the program cache of the full model (see `just vllm-serve-test`).
"""

import argparse
import json
import os
import pathlib
import random
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "serving"))

import engine
from host_cost_summary import summarise  # noqa: E402  (sets the environment of the in-process engine before vLLM is imported)
import numpy as np
from vllm import LLM, SamplingParams  # noqa: E402

CLOCK_HZ = 900e6  # the clock of the published TSP sources (ISCA20/ISCA22)
PADDING = 0xFFFFFFFF  # encoding.json text_format.padding: an image unit that holds no instruction
STAT_KEYS = ("patch_seconds", "restart_seconds", "run_seconds", "load_seconds", "patch_words", "text_words", "text_switches", "cycles_last")


class Proxy:
    """The binding's `Chip` with `step` timed; every other attribute is the binding's."""

    def __init__(self, chip, rec):
        self._c = chip
        self._rec = rec  # the holder whose `cur` is the dict of the part being measured

    def __getattr__(self, name):
        return getattr(self._c, name)

    def step(self, *a, **kw):
        cur = self._rec.cur
        before = self._c.stats()
        t0 = time.perf_counter()
        try:
            words, cycles, diagnostics = self._c.step(*a, **kw)
        finally:
            call = time.perf_counter() - t0
        after = self._c.stats()
        d = {k: after[k] - before[k] for k in STAT_KEYS}
        cur["binding_call_s"] = call
        cur["patch_s"] = d["patch_seconds"]
        cur["restart_s"] = d["restart_seconds"]
        cur["run_s"] = d["run_seconds"]
        cur["image_and_read_s"] = call - d["patch_seconds"] - d["restart_seconds"] - d["run_seconds"]  # the IQ window writes, the output read, the pyo3 copies
        cur["patch_word_bytes"] = d["patch_words"] * self._c.vlanes
        cur["text_word_bytes"] = d["text_words"] * self._c.vlanes
        cur["text_switches"] = d["text_switches"]
        cur["bytes_read"] = len(words)
        cur["cycles"] = cycles
        return words, cycles, diagnostics


class Holder:
    cur = None


def timed(fn, cur, key):
    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            cur_ = holder.cur
            if cur_ is not None:
                cur_[key] = cur_.get(key, 0.0) + time.perf_counter() - t

    return wrapper


holder = Holder()


def image_bytes(iss, bucket):
    """Bytes of the instruction-queue images of `bucket` as the binding loads them (units that are not padding, 4 bytes each), and the
    queue count: from the binding's test hook, once per bucket, outside the timed steps."""
    program = iss._program(bucket)
    plan = iss._padding_plan(bucket)
    images, _ = iss._chip._prepare_for_test(program.name, program.site_values(plan))
    units = 0
    for b in images.values():
        u = np.frombuffer(b, "<u4")
        units += int((u != PADDING).sum())
    return units * 4, len(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--context", type=int, default=64, help="the context of the chip's programs (64: the 9 buckets of C = 64 only; the steps measured need no more)")
    a = ap.parse_args()
    if a.requests < 2:
        ap.error("--requests must be at least 2 (the first request is the warm-up and is left out of the steady class)")
    if a.prompt_len < 8 or a.prompt_len % 8:
        ap.error("--prompt-len must be a positive multiple of 8 (whole prefill steps of bucket P(8,1,C))")
    if a.new < 2 or a.prompt_len + a.new > 64:
        ap.error("--new must be at least 2 and --prompt-len + --new at most 64 (the buckets measured are the C = 64 ones)")
    engine.MODELS["full"]["context"] = a.context  # the 8-layer model; fewer buckets to build and load, the same programs for these steps
    load_before = os.getloadavg()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        path = engine.model_dir(root, "full")
        llm = LLM(**engine.engine_kwargs(path, "iss", "full", root=root))
        e = engine.Engine(llm, "full")
        worker = e.worker
        iss = worker.chip
        runner = worker.model_runner
        # timing wrappers (instance attributes of this process's objects; the package is unchanged)
        iss._chip = Proxy(iss._chip, holder)
        for p in iss._loaded.values():
            p.site_values = timed(p.site_values, None, "site_values_s")
            p.decode = timed(p.decode, None, "decode_s")
        iss.zero_blocks = timed(iss.zero_blocks, None, "zero_blocks_s")
        iss.copy_blocks = timed(iss.copy_blocks, None, "copy_blocks_s")
        serve_step = iss.step  # Engine's recording wrapper around the bound method
        iss.step = timed(serve_step, None, "chip_step_s")
        class_step = type(iss).step  # the unbound method: the baseline replays a plan through it

        images = {}
        for bucket in ((1, 1, 64), (8, 1, 64)):
            images[bucket] = image_bytes(iss, bucket)

        records = []
        request_index = [0]
        orig_exec = worker.execute_model

        def execute(so):
            t = time.perf_counter()
            try:
                return orig_exec(so)
            finally:
                holder.cur["execute_model_s"] = time.perf_counter() - t

        worker.execute_model = execute
        # the plan of the step: the Engine's recording appends (plan, result) to e.steps
        orig_step = llm.llm_engine.step

        def one_step(*args, **kw):
            e.steps.clear()
            serve = {}
            holder.cur = serve
            t = time.perf_counter()
            try:
                return orig_step(*args, **kw)
            finally:
                serve["outer_s"] = time.perf_counter() - t
                if e.steps:
                    plan, result = e.steps[-1]
                    base = {}
                    holder.cur = base
                    t = time.perf_counter()
                    class_step(iss, plan)
                    base["outer_s"] = time.perf_counter() - t
                    holder.cur = None
                    records.append((plan, serve, base, dict(runner.stats.last), request_index[0]))

        llm.llm_engine.step = one_step
        rng = random.Random(79)
        t_run = time.perf_counter()
        try:
            for i in range(a.requests):
                request_index[0] = i
                prompt = [rng.randrange(1000, 50000) for _ in range(a.prompt_len)]
                llm.reset_prefix_cache()
                e.generate([prompt], SamplingParams(temperature=0.0, max_tokens=a.new, ignore_eos=True))
                print(f"request {i + 1}/{a.requests} done, {len(records)} steps, {time.perf_counter() - t_run:.0f} s", file=sys.stderr, flush=True)
        finally:
            e.close()  # also after an interrupt

    steps = []
    for plan, serve, base, last, req in records:
        img_bytes, _ = images[tuple(plan.bucket)]
        vals = plan.patches
        value_bytes = sum(v.nbytes for v in vals.values()) if isinstance(vals, dict) else len(vals)
        rec = dict(request=req, bucket=list(plan.bucket), n_real=plan.n_real, n_logits=plan.n_logits, runner=dict(last))
        for name, cur in (("serve", serve), ("base", base)):
            cur["value_bytes"] = value_bytes
            cur["image_bytes"] = img_bytes
            cur.setdefault("site_values_s", 0.0)
            cur.setdefault("decode_s", 0.0)
            # everything that is not the ISS's run of the program: Python and vLLM (outer minus the binding call) and the binding's own parts
            cur["python_s"] = cur["outer_s"] - cur["binding_call_s"]
            cur["host_py_s"] = cur["python_s"]  # the Python and vLLM share that stays on any chip
            cur["host_s"] = cur["outer_s"] - cur["run_s"]  # all host work of the step, as measured here
            rec[name] = cur
        rec["hw_s"] = serve["cycles"] / CLOCK_HZ
        steps.append(rec)
    load_after = os.getloadavg()
    summary = dict(serve=summarise(steps, "serve"), base=summarise(steps, "base"))
    result = dict(
        commit=commit, args=vars(a), clock_hz=CLOCK_HZ, load_before=load_before, load_after=load_after, uptime=subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip(),
        images={f"P{b[0]}_{b[1]}_{b[2]}": dict(image_bytes=v[0], queues=v[1]) for b, v in images.items()}, summary=summary, steps=steps,
    )
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"host_cost-{commit}.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(dict(commit=commit, load_before=load_before, load_after=load_after, images=result["images"], summary=summary), indent=1, default=float))


if __name__ == "__main__":
    main()
