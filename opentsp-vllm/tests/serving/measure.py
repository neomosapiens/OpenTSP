"""Measurements of  : engine start-up and its parts (program cache reader, weight package load, warm-up) and the
host overhead per step of one request on the ISS chip. `python3 measure.py [test|full] [--chip iss|exact]` prints one JSON object.

The host overhead of a step is its wall time in `TspWorker.execute_model` minus the binding's run time (`run_seconds` of the chip's stats):
the request state, the step plan, the patch values, the loads, the decoding of the logits, the sampler and the output.
"""

import json
import sys
import tempfile
import time

import engine
from vllm import LLM, SamplingParams

PROMPT = [7454, 2402, 257, 640, 612, 373] + list(range(1000, 1015))  # 21 tokens: three prefill steps of 8, 8, 5, then decode
NEW = 8


def timed(obj, name, record, key):
    fn = getattr(obj, name)

    def wrapper(*a, **kw):
        t = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            record[key] = record.get(key, 0.0) + time.perf_counter() - t

    setattr(obj, name, wrapper)


def main(model="test", chip="iss"):
    parts = {}
    from opentsp_vllm import chip as chip_mod
    from opentsp_vllm import programs as programs_mod

    timed(programs_mod.ProgramSet, "__init__", parts, "program_set_s")
    timed(chip_mod.IssChip, "load_model", parts, "load_package_s")
    timed(chip_mod.IssChip, "warm_up", parts, "warm_up_s")
    with tempfile.TemporaryDirectory() as root:
        path = engine.model_dir(root, model)
        kwargs = engine.engine_kwargs(path, chip, model, root=root)  # builds the programs when the cache lacks them: not timed
        t = time.perf_counter()
        llm = LLM(**kwargs)
        start_s = time.perf_counter() - t
        e = engine.Engine(llm, model)
        walls = []
        execute = e.worker.execute_model

        def recording(so):
            t0 = time.perf_counter()
            try:
                return execute(so)
            finally:
                walls.append(time.perf_counter() - t0)

        e.worker.execute_model = recording
        iss = e.chip._chip if chip == "iss" else None
        before = iss.stats() if iss else {}
        t = time.perf_counter()
        tokens, _ = e.solo(PROMPT, SamplingParams(temperature=0.0, max_tokens=NEW, ignore_eos=True))
        generate_s = time.perf_counter() - t
        after = iss.stats() if iss else {}
        stats = e.worker.get_tsp_stats()
        run_s = after.get("run_seconds", 0.0) - before.get("run_seconds", 0.0)
        steps = len(walls)
        out = dict(
            model=model, chip=chip, prompt_tokens=len(PROMPT), new_tokens=NEW, token_ids=tokens,
            engine_start_s=start_s, **{k: round(v, 3) for k, v in parts.items()},
            generate_s=generate_s, steps=steps, execute_model_s=sum(walls), iss_run_s=run_s,
            host_overhead_per_step_s=(sum(walls) - run_s) / steps, step_wall_s=walls,
            runner=dict(host_prep_s=stats["host_prep_s"], chip_s=stats["chip_s"], sampler_s=stats["sampler_s"], buckets=stats["buckets"]),
            binding={k: after.get(k) for k in ("run_seconds", "load_seconds", "patch_seconds", "restart_seconds", "cycles_total")},
        )
        e.close()
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    chip = sys.argv[sys.argv.index("--chip") + 1] if "--chip" in sys.argv else "iss"
    main(args[0] if args and args[0] in engine.MODELS else "test", chip)
