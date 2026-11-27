"""Compare serving speed with parametrized schedule replay disabled and enabled.
`python3 replay_ab.py [test|full] [--rounds N] [--warmup W] [--long] [--out FILE]` writes one JSON object to FILE (default: standard output, which vLLM's logging shares). Run it with the vLLM environment and this tree's binding
(`just vllm-binding`), like `measure.py`.

Two engines of the same model run in one process, one on the normal driver and one with `opentsp.replay` on; a round runs the same three
requests on both, in the order A B on even rounds and B A on odd rounds (so a drift of the host load does not favour one), and records the
wall time of `generate`, the binding's run time (`run_seconds` of the chip) and the load average before and after. The first `warmup`
rounds of B record the schedules of the buckets (first sight, then the recording step): they are reported, not in the primary metric.
A round starts only when the one-minute load average is at most `REPLAY_AB_MAX_LOAD` (default 12; it waits up to 30 minutes). The primary metric is the median over the other rounds of the A time over the B time of `generate`. Exactness is a hard gate: the tokens,
the logits bit for bit and the cycles of every step of every round are equal in A and B.
"""

import json
import os
import statistics
import sys
import tempfile
import time

import engine
import numpy as np
from vllm import LLM, SamplingParams

H_VOCAB = engine.VOCAB
MAX_LOAD = float(os.environ.get("REPLAY_AB_MAX_LOAD", "12"))
GREEDY = SamplingParams(temperature=0.0, max_tokens=14, ignore_eos=True)


def prompt(seed, n):
    return [int(t) for t in np.random.default_rng(seed).integers(1, H_VOCAB, n)]


PROMPTS = [prompt(1, 6), prompt(2, 21), prompt(3, 37)]
# `--long` (the full model only: contexts up to 128): four longer prompts (four sequences at once), so that the buckets of context 128 run, P(8,4,128) among them
PROMPTS_LONG = [prompt(4, 70), prompt(5, 30), prompt(6, 66), prompt(7, 50)]


def make(root, model, replay):
    path = engine.model_dir(root, model)
    kwargs = engine.engine_kwargs(path, "iss", model, root=root, opentsp={"replay": replay})
    return engine.Engine(LLM(**kwargs), model)


def one(e, prompts):
    e.llm.reset_prefix_cache()
    e.clear()
    chip = e.chip._chip
    before = chip.stats()["run_seconds"]
    t = time.perf_counter()
    outs = e.generate(prompts, GREEDY)
    wall = time.perf_counter() - t
    run = chip.stats()["run_seconds"] - before
    tokens = [list(o.outputs[0].token_ids) for o in outs]
    return dict(
        wall=wall, run=run, tokens=tokens, cycles=[r.cycles for _, r in e.steps],
        logits=[np.asarray(r.logits).copy() for _, r in e.steps],
    )  # fmt: skip


def main(model="test", rounds=9, warmup=2, long=False):
    prompts = PROMPTS_LONG if long else PROMPTS
    with tempfile.TemporaryDirectory() as root:
        a, b = make(root, model, False), make(root, model, True)
        assert a.chip._chip.stats()["replay"]["enabled"] is False and b.chip._chip.stats()["replay"]["enabled"] is True
        rows = []
        for k in range(rounds + warmup):
            order = (a, b) if k % 2 == 0 else (b, a)
            waited = 0
            while os.getloadavg()[0] > MAX_LOAD and waited < 1800:  # a round runs on a quiet host or waits for one (at most 30 minutes)
                time.sleep(15)
                waited += 15
            load0 = os.getloadavg()[0]
            res = {}
            for e in order:
                res["A" if e is a else "B"] = one(e, prompts)
            load1 = os.getloadavg()[0]
            ra, rb = res["A"], res["B"]
            same = (
                ra["tokens"] == rb["tokens"]
                and ra["cycles"] == rb["cycles"]
                and len(ra["logits"]) == len(rb["logits"])
                and all(np.array_equal(x, y) for x, y in zip(ra["logits"], rb["logits"]))
            )
            assert same, f"round {k}: the replay engine's tokens, cycles or logits differ"
            rows.append(dict(round=k, order="AB" if k % 2 == 0 else "BA", wall_a=ra["wall"], wall_b=rb["wall"], run_a=ra["run"], run_b=rb["run"], steps=len(ra["cycles"]), load_before=load0, load_after=load1, waited_s=waited))
        steady = rows[warmup:]
        out = dict(
            model=model, long=long, rounds=rounds, warmup=warmup, rows=rows,
            ratio_wall_median=statistics.median(r["wall_a"] / r["wall_b"] for r in steady),
            ratio_run_median=statistics.median(r["run_a"] / r["run_b"] for r in steady),
            ratio_wall_all_rounds=sum(r["wall_a"] for r in rows) / sum(r["wall_b"] for r in rows),
            ratio_wall_warmup_rounds=sum(r["wall_a"] for r in rows[:warmup]) / max(1e-9, sum(r["wall_b"] for r in rows[:warmup])),
            load_max=max(max(r["load_before"], r["load_after"]) for r in rows),
            exact=True, replay=b.chip._chip.stats()["replay"],
        )  # fmt: skip
        a.close()
        b.close()
    text = json.dumps(out, indent=1, default=float)
    if "--out" in sys.argv:
        with open(sys.argv[sys.argv.index("--out") + 1], "w") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x in engine.MODELS]
    opt = lambda name, d: int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else d  # noqa: E731
    main(args[0] if args else "test", opt("--rounds", 9), opt("--warmup", 2), "--long" in sys.argv)
