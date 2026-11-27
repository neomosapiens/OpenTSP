#!/usr/bin/env python3
"""Report simulated cycle counts and measured ISS timings for cached TinyStories buckets.

    python3 cycle_report.py --time [--layers 2|8] [--repeats 3]   run every built bucket on a padding-only plan, one process at
                                                                  a time, and keep the best wall time in <entry>/timing.json
    python3 cycle_report.py --cycles                              <cache>/cycles.json, with the timing of timing.json merged in
    python3 cycle_report.py --report [--layers 2|8]               the recomputed tables of section 6, as markdown

The timing is the ISS's own run time (`tsp-iss run --stats`, the simulated cycles after the images are loaded): what the platform
binding spends per step, which loads the images from memory . `iss_seconds` of `cycles.json` is that time.
This module is not a generator source: changing it does not change a digest.
"""
import argparse
import json
import os
import sys


from opentsp_vllm.models.tinystories import build_programs as BP
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_vllm import step_plan as SP
from opentsp_vllm.programs import ProgramsError, ProgramSet

PLANNING_RATE = 61400            # simulated cycles per second: 's B4 benchmark, the rate  plans with
# the baseline model of  (T (8 L_C + E) + L F + H(L)), kept to show how far it was from what was measured
L_C = {64: 11362, 128: 11862}
E_LANE, F_ROW = 150, 526


def head(rows):
    return 1400 + 92 * rows


def step_base(layers, bucket):
    T, Ls, C = bucket
    return T * (layers * L_C[C] + E_LANE) + Ls * F_ROW + head(Ls)


def timing_path(cache, model):
    return os.path.join(PC.entry_dir(cache, model), "timing.json")


def time_bucket(model, bucket, cache=None, repeats=3):
    """Best of `repeats` padding-only runs of the cached program; raises on a diagnostic."""
    path = PC.bucket_dir(cache, model, bucket)
    art = BP.Artifacts(path, model.profile)
    names = [str(n) for n in art.paged.table.names]
    values = PC.plan_site_values(art.record["geometry"], names, SP.padding_plan(bucket, vocab=50257))
    iss = BP.tsp_iss()
    best = None
    for _ in range(repeats):
        r = BP.native_run(iss, art, values, model.package, 2 * art.record["schedule_cycles"])
        if r["diags"] or r["returncode"] or r["last_cycle"] is None:
            raise BP.BuildError(f"{PC.bucket_name(bucket)}: diagnostics {r['diags'][:3]}, exit {r['returncode']}")
        if best is None or r["wall"] < best["wall"]:
            best = r
    return {"cycles": best["last_cycle"], "ticks": best["ticks"], "wall": best["wall"], "process": best["seconds"],
            "rate": best["last_cycle"] / best["wall"], "load_average": os.getloadavg()[0], "repeats": repeats}


def time_buckets(layers, cache=None, repeats=3, buckets=None, log=print):
    model = PC.Model(layers)
    path = timing_path(cache, model)
    done = PC.read_json(path) if os.path.exists(path) else {}
    for bucket in buckets or BP.grid(layers):
        if not PC.built(cache, model, bucket):
            raise BP.BuildError(f"{PC.bucket_name(bucket)} is not built; run `LAYERS={layers} just vllm-programs`")
        rec = time_bucket(model, bucket, cache, repeats)
        done[PC.bucket_name(bucket)] = rec
        PC.write_json(path, done)
        log(f"{PC.bucket_name(bucket)}: {rec['cycles']} cycles in {rec['wall']:.3f} s ({rec['rate'] / 1000:.0f}k cycles/s)")
    return done


def text_words_by_queue(blocks):
    """{queue: words of paged text} of the records of the block list in a bucket's `program.json`."""
    out = {}
    for b in blocks:
        out[b["queue"]] = out.get(b["queue"], 0) + sum(t[3] for t in b["tensors"])
    return out


def largest_text(cache, model):
    """(largest bucket name, queues the largest bucket does not hold the maximum of, {queue: (max words, bucket)}) over the built
    buckets of `model`: 's claim that P(8, 4, 128) has the most text in every slice."""
    best, by_bucket = {}, {}
    for bucket in BP.grid(model.layers):
        path = PC.bucket_dir(cache, model, bucket)
        if not os.path.exists(os.path.join(path, "program.json")):
            continue
        by_bucket[bucket] = text_words_by_queue(PC.read_json(os.path.join(path, "program.json"))["blocks"])
        for q, w in by_bucket[bucket].items():
            if w > best.get(q, (0, None))[0]:
                best[q] = (w, PC.bucket_name(bucket))
    top = max(by_bucket, key=lambda b: sum(by_bucket[b].values()))
    short = sorted(q for q, (w, _) in best.items() if by_bucket[top].get(q, 0) < w)
    return PC.bucket_name(top), short, best, len(by_bucket)


def timings(cache=None):
    """{digest: {bucket name: timing}} of the timing.json files of the cache."""
    out = {}
    root = PC.cache_root(cache)
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, name, "timing.json")
        m = os.path.join(root, name, "manifest.json")
        if os.path.exists(p) and os.path.exists(m):
            out[PC.read_json(m)["digest"]] = PC.read_json(p)
    return out


def write_cycles(cache=None):
    """`<cache>/cycles.json`: `PC.write_cycles` plus, where timing.json has the bucket, `iss_seconds` (the ISS run time) and
    `iss_cycles_per_second` of the best padding-only run."""
    table = PC.write_cycles(cache)
    tm = timings(cache)
    for digest, entry in table.items():
        for name, rec in entry["buckets"].items():
            t = tm.get(digest, {}).get(name)
            if t and t["cycles"] == rec["cycles"]:
                rec["iss_seconds"], rec["iss_cycles_per_second"] = round(t["wall"], 4), round(t["rate"])
    PC.write_atomic(os.path.join(PC.cache_root(cache), "cycles.json"), json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


# -- the recomputed numbers -------------------------------------------------------------------------------------------------------
def missing_buckets(entry, layers):
    """The grid buckets of `layers` layers that a table entry does not hold (the report reads every one of them)."""
    return [PC.bucket_name(b) for b in BP.grid(layers) if PC.bucket_name(b) not in entry["buckets"]]


def entry_of(table, layers, digest=None):
    """The entry of `layers` layers the report reads: the one whose digest starts with `digest`, else the only complete one. An
    incomplete entry (a bucket of the grid missing) is never picked; the message names what is missing and the recipe that builds it."""
    cands = {d: e for d, e in table.items() if e["layers"] == layers and (digest is None or d.startswith(digest))}
    if not cands:
        raise KeyError(f"no entry of {layers} layers" + (f" with digest {digest}" if digest else "") + " in the cycle table")
    complete = {d: e for d, e in cands.items() if not missing_buckets(e, layers)}
    if len(complete) == 1:
        return next(iter(complete.values()))
    recipe = f"`LAYERS={layers} just vllm-programs`"
    if not complete:
        why = "; ".join(f"{d[:16]} lacks {len(missing_buckets(e, layers))} of {len(BP.grid(layers))} buckets "
                        f"({', '.join(missing_buckets(e, layers)[:3])}{', ...' if len(missing_buckets(e, layers)) > 3 else ''})"
                        for d, e in cands.items())
        raise KeyError(f"no complete entry of {layers} layers in the cycle table: {why}; build the missing buckets with {recipe}")
    raise KeyError(f"{len(complete)} complete entries of {layers} layers ({', '.join(d[:16] for d in complete)}): choose one with --digest")


def runtime_digests(cache, layers_list):
    """{layers: digest} the runtime reads (`ProgramSet`'s selection of the complete program set), so the report describes the programs
    a served step runs. A cache without a complete set raises `ProgramsError` naming the recipe that builds it."""
    return {layers: ProgramSet(PC.cache_root(cache), layers=layers).digest for layers in layers_list}


def cyc(entry, bucket):
    return entry["buckets"][PC.bucket_name(bucket)]["cycles"]


def sec(entry, bucket, rate=None):
    """ISS seconds of a bucket: at the planning rate, or at `rate`, or measured (`iss_seconds` of the table)."""
    c = cyc(entry, bucket)
    return c / rate if rate else entry["buckets"][PC.bucket_name(bucket)]["iss_seconds"]


def measured_rate(entry):
    """Cycles per second of the whole entry: all cycles over all measured ISS seconds."""
    b = entry["buckets"].values()
    return sum(r["cycles"] for r in b) / sum(r["iss_seconds"] for r in b)


def tokens_per_second(entry, T, C, rate=None):
    """Tokens per second of a step of T lanes with one logit row in context bucket C: T over the step's seconds."""
    b = (T, 1, C)
    return T / sec(entry, b, rate)


def lane_cost(entry, C=64):
    """(cycles of one lane in a P(1, 1, C) step, cycles each further lane adds, worst deviation of P(T, 1, C) from the line)."""
    one = cyc(entry, (1, 1, C))
    slope = (cyc(entry, (8, 1, C)) - one) / 7
    dev = max(abs(cyc(entry, (T, 1, C)) - (one + slope * (T - 1))) for T in (1, 2, 4, 8))
    return one, slope, dev


def smallest_t(n):
    return next(t for t in (1, 2, 4, 8) if t >= n)


def padding_cost(entry, C, ns):
    """For step sizes `ns` (tokens per step, equally likely): (coarse cycles, fine-grid cycles, wasted fraction of the coarse
    cycles). The fine grid (every T 1..8) is the line through P(1, 1, C) and P(8, 1, C)."""
    one, slope, _ = lane_cost(entry, C)
    fine = [one + slope * (n - 1) for n in ns]
    coarse = [cyc(entry, (smallest_t(n), 1, C)) for n in ns]
    return sum(coarse) / len(ns), sum(fine) / len(ns), 1 - sum(fine) / sum(coarse)


def grid_pairs(ts, ls=(1, 2, 4)):
    return [(T, Ls) for T in ts for Ls in ls if Ls <= T]


def grid_build_seconds(entry, fine=False):
    """Sequential build seconds of the coarse grid (the sum of the measured `build_seconds`) or an estimate of the fine grid (every
    T 1..8, L in 1, 2, 4 with L <= T: 20 pairs, 40 programs): per (L, C) the line through the measured T = 1 and T = 8 builds,
    taken at every T. (The measured seconds were taken with several builds at once and are above the uncontended ones.)"""
    b = entry["buckets"]
    if not fine:
        return sum(r["build_seconds"] for r in b.values())
    total = 0.0
    for C in (64, 128):
        for T, Ls in grid_pairs(range(1, 9)):
            lo = b[PC.bucket_name((1, 1, C))]["build_seconds"]
            hi = b[PC.bucket_name((8, Ls, C))]["build_seconds"]
            total += lo + (hi - lo) * (T - 1) / 7
    return total


def scenarios(table, layers_test=2, layers_full=8):
    """Wall-clock seconds of the serving scenarios of  and of , at the planning rate and at the measured rate of each
    model: the same lane counts as the design (an upper bound: every lane is charged the cost of a P(1, 1, 64) step)."""
    t, f = entry_of(table, layers_test), entry_of(table, layers_full)
    lane_t, lane_f = cyc(t, (1, 1, 64)), cyc(f, (1, 1, 64))
    rt, rf = measured_rate(t), measured_rate(f)
    rows = {}

    def put(name, c_test=0, c_full=0):
        rows[name] = {"cycles": c_test + c_full, "planning": (c_test + c_full) / PLANNING_RATE,
                      "measured": c_test / rt + c_full / rf}
    put("test-model scenario: 64 tokens, 30 % padding", 64 * 1.3 * lane_t)
    put("preemption scenario: 72 tokens + 32 recomputed, 30 % padding", (72 + 32) * 1.3 * lane_t)
    put("full model: 1 request, 4-token prompt, 8 new tokens (12 lanes)", 0, 12 * lane_f)
    put("full model: 2 requests sharing a prefix (30 lanes with padding)", 0, 30 * lane_f)
    suite = {"planning": 11 * rows[next(iter(rows))]["planning"], "measured": 11 * rows[next(iter(rows))]["measured"]}
    names = list(rows)
    for k in ("planning", "measured"):
        suite[k] += sum(rows[n][k] for n in names[1:]) + 3 * 20          # three engine start-ups of 20 s (estimate)
    rows["suite, serial (11 test scenarios + the three others + 3 x 20 s start-up)"] = {"cycles": None, **suite}
    # Replay comparison: 60 steps on the test model + 6 on the full model, each run twice (warm restart and replay)
    mix = (cyc(t, (2, 2, 64)) + cyc(t, (4, 4, 64))) / 2
    for name, ct, cf in (("Replay, steps at T = 2 and 4 alternately", 60 * mix, 6 * cyc(f, (1, 1, 64))),
                         ("Replay, every step at P(4, 4, 64) / P(8, 4, 64) (bound)", 60 * cyc(t, (4, 4, 64)), 6 * cyc(f, (8, 4, 64)))):
        put(name, 2 * ct, 2 * cf)
    return rows


def preemption(table):
    """A request preempted at 128 tokens is re-prefilled in 16 steps of P(8, 1, 128) on the full model; the test model's
    preemption scenario recomputes up to 24 tokens, 3 steps of P(8, 1, 64)."""
    f, t = entry_of(table, 8), entry_of(table, 2)
    full, small = 16 * cyc(f, (8, 1, 128)), 3 * cyc(t, (8, 1, 64))
    return {"full": (full, full / PLANNING_RATE, full / measured_rate(f)), "test": (small, small / PLANNING_RATE, small / measured_rate(t))}


def t_grid(table):
    """The numbers of the T-grid decision on the 8-layer model: padding cost for two step-size mixes, the build cost."""
    f = entry_of(table, 8)
    mixes = {"decode of up to 4 requests (n = 1..4)": [1, 2, 3, 4], "every n = 1..8": list(range(1, 9)),
             "n = 1, 2, 4, 8 only (no waste)": [1, 2, 4, 8]}
    out = {"padding": {k: padding_cost(f, 64, ns) for k, ns in mixes.items()}, "lane": lane_cost(f, 64),
           "lane128": lane_cost(f, 128), "build": (grid_build_seconds(f), grid_build_seconds(f, fine=True)),
           "programs": (len(grid_pairs((1, 2, 4, 8))) * 2, len(grid_pairs(range(1, 9))) * 2)}
    # what the finer grid buys on the test model's suite: the padding share of the decode mix off its ISS seconds, against the
    # extra build of the 2-layer grid, once per cache
    t = entry_of(table, 2)
    rows = scenarios(table)
    names = list(rows)
    iss = 11 * rows[names[0]]["measured"] + rows[names[1]]["measured"]            # the test-model scenarios of the suite
    saving = iss * out["padding"]["decode of up to 4 requests (n = 1..4)"][2]
    extra = grid_build_seconds(t, fine=True) - grid_build_seconds(t)
    out["suite"] = {"iss_seconds": iss, "saving_per_run": saving, "extra_build_2_layers": extra, "break_even_runs": extra / saving}
    return out


def report(table, layers_list=(8, 2), digests=None):
    """The tables of  from `cycles.json`'s content: {name: rows}. `digests` ({layers: digest prefix}) binds each layer count
    to one entry (`main` passes the runtime's selection); every table of the report then reads that entry."""
    digests = digests or {}
    chosen = {}
    for ly in layers_list:
        e = entry_of(table, ly, digests.get(ly))
        chosen[next(d for d, x in table.items() if x is e)] = e
    table = chosen
    out = {"digests": {ly: next(d for d, x in table.items() if x["layers"] == ly) for ly in layers_list}}
    for layers in layers_list:
        e = entry_of(table, layers)
        grid = [(T, Ls, C) for C in (64, 128) for T in (1, 2, 4, 8) for Ls in (1, 2, 4) if Ls <= T]
        out[f"cycles{layers}"] = [(b, cyc(e, b), step_base(layers, b), sec(e, b, PLANNING_RATE), sec(e, b)) for b in grid]
        out[f"tps{layers}"] = [(C, T, cyc(e, (T, 1, C)), cyc(e, (T, 1, C)) / T, tokens_per_second(e, T, C, PLANNING_RATE),
                                tokens_per_second(e, T, C)) for C in (64, 128) for T in (1, 2, 4, 8)]
        out[f"rate{layers}"] = measured_rate(e)
        out[f"warmup{layers}"] = (cyc(e, (1, 1, 64)), sec(e, (1, 1, 64), PLANNING_RATE), sec(e, (1, 1, 64)))
    out["scenarios"] = scenarios(table)
    out["preemption"] = preemption(table)
    out["t_grid"] = t_grid(table)
    return out


def markdown(r):
    """The report as markdown tables."""
    L = []
    for layers in (8, 2):
        L += [f"### {layers} layers: cycles and ISS seconds per bucket", "",
              "| Bucket | cycles (measured) | baseline model | ISS s at 61 400 /s | ISS s, run time only |", "|---|---|---|---|---|"]
        L += [f"| P({b[0]}, {b[1]}, {b[2]}) | {c:,} | {m:,} ({100 * (c - m) / m:+.1f} %) | {p:.2f} | {s:.2f} |" for b, c, m, p, s in r[f"cycles{layers}"]]
        L += ["", f"measured rate {r[f'rate{layers}']:,.0f} cycles/s over the grid", ""]
        L += [f"### {layers} layers: tokens per second (one logit row per step)", "",
              "| C | T | cycles | cycles per token | tokens/s at 61 400 /s | tokens/s, ISS run time only |", "|---|---|---|---|---|---|"]
        L += [f"| {C} | {T} | {c:,} | {pt:,.0f} | {a:.3f} | {m:.3f} |" for C, T, c, pt, a, m in r[f"tps{layers}"]]
        c, p, m = r[f"warmup{layers}"]
        L += ["", f"warm-up, P(1, 1, 64) once: {c:,} cycles, {p:.2f} s at 61 400 /s, {m:.2f} s ISS run time only", ""]
    L += ["### Preemption (re-prefill)", ""]
    for k, (c, p, m) in r["preemption"].items():
        L.append(f"- {k}: {c:,} cycles, {p:.1f} s at 61 400 /s, {m:.1f} s ISS run time only")
    L += ["", "### Scenarios and replay comparison", "", "| Scenario | cycles | s at 61 400 /s | s, ISS run time only (excludes binding load and patch writes) |", "|---|---|---|---|"]
    L += [f"| {n} | {('{:,.0f}'.format(v['cycles']) if v['cycles'] else '')} | {v['planning']:.0f} | {v['measured']:.0f} |" for n, v in r["scenarios"].items()]
    g = r["t_grid"]
    L += ["", "### T grid", "", f"lane cost at C = 64: {g['lane'][0]:,.0f} for the first lane, {g['lane'][1]:,.0f} for each further lane "
          f"(worst deviation of P(T, 1, 64) from the line {g['lane'][2]:.0f} cycles); C = 128: {g['lane128'][0]:,.0f}, {g['lane128'][1]:,.0f}",
          "", "| Step sizes | coarse cycles | fine-grid cycles | wasted share of the coarse cycles |", "|---|---|---|---|"]
    L += [f"| {k} | {c:,.0f} | {f:,.0f} | {100 * w:.1f} % |" for k, (c, f, w) in g["padding"].items()]
    L += ["", f"build, 8 layers, sequential seconds: coarse grid ({g['programs'][0]} programs, measured) {g['build'][0]:,.0f}, fine grid ({g['programs'][1]} programs, estimate) {g['build'][1]:,.0f} (x{g['build'][1] / g['build'][0]:.2f})",
          f"test-model suite: {g['suite']['iss_seconds']:.0f} s of ISS time, the finer grid would save at most {g['suite']['saving_per_run']:.0f} s per run "
          f"(the decode mix's waste share); it costs {g['suite']['extra_build_2_layers']:.0f} s more to build the 2-layer grid once, "
          f"break-even after {g['suite']['break_even_runs']:.0f} suite runs"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=None, choices=(2, 8))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--bucket", action="append")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--cycles", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--digest", help="with --layers: the entry (digest prefix) of that layer count the report reads; default: the runtime's selection")
    ap.add_argument("--text", action="store_true", help="which bucket has the most paged text in every queue")
    a = ap.parse_args(argv)
    if a.time:
        for layers in [a.layers] if a.layers else (8, 2):
            time_buckets(layers, a.cache, a.repeats, [PC.parse_bucket(b) for b in a.bucket] if a.bucket else None)
    if a.cycles or a.time:
        table = write_cycles(a.cache)
        print(f"{os.path.join(PC.cache_root(a.cache), 'cycles.json')}: " + ", ".join(f"{d[:16]} {len(e['buckets'])} buckets" for d, e in table.items()))
    if a.text:
        for layers in [a.layers] if a.layers else (8, 2):
            top, short, best, n = largest_text(a.cache, PC.Model(layers))
            print(f"{layers} layers, {n} buckets: most text {top}; queues where another bucket has more: {short or 'none'}; "
                  f"largest queue {max(best.items(), key=lambda kv: kv[1][0])}")
    if a.report:
        if a.digest and not a.layers:
            sys.exit("cycle_report: --digest needs --layers (the report reads one entry of 8 layers and one of 2)")
        try:
            digests = runtime_digests(a.cache, [ly for ly in (8, 2) if ly != a.layers or not a.digest])
            if a.digest:
                digests[a.layers] = a.digest
            r = report(write_cycles(a.cache), (8, 2), digests)
        except (KeyError, ProgramsError) as e:
            sys.exit(f"cycle_report: {e.args[0]}")
        print("report of the cache entries " + ", ".join(f"{ly} layers: {d[:16]}" for ly, d in r["digests"].items()) + "\n")
        print(markdown(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
