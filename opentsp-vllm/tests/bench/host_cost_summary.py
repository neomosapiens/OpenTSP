"""Summary of the step records of `host_cost.py` (no vLLM import): per class and path the mean and spread of the parts, the shares and the
estimates. `python3 host_cost_summary.py out/run1 out/run2 ...` pools the steps of several runs and prints the tables of the report."""

import json
import pathlib
import statistics
import sys


def stat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), mean=statistics.fmean(xs), median=statistics.median(xs), min=min(xs), max=max(xs), stdev=statistics.pstdev(xs))


def summarise(steps, path):
    """Per class (decode, prefill) and steady (no text switch of the served step, not the first request: first-use costs of Python and torch; the baseline of a step is selected with its served step) / all: the mean and spread of the parts, in milliseconds, and the shares."""
    out = {}
    for cls, T in (("decode", 1), ("prefill", 8)):
        for steady in (True, False):
            sel = [s for s in steps if s["bucket"][0] == T and (not steady or (s["serve"]["text_switches"] == 0 and s["request"] >= 1))]
            if not sel:
                continue
            key = f"{cls}-{'steady' if steady else 'all'}"
            hw = [s["hw_s"] for s in sel]
            row = dict(steps=len(sel), cycles=sel[0][path]["cycles"], hw_us=statistics.fmean(hw) * 1e6)
            for k in ("outer_s", "binding_call_s", "run_s", "patch_s", "restart_s", "image_and_read_s", "site_values_s", "decode_s", "python_s", "host_s", "host_py_s"):
                row[k] = stat([s[path].get(k) for s in sel])
            row["share_host"] = stat([s[path]["host_s"] / s["hw_s"] for s in sel])
            row["share_python"] = stat([s[path]["host_py_s"] / s["hw_s"] for s in sel])
            row["share_vs_iss_wall"] = stat([s[path]["host_s"] / (s[path]["host_s"] + s[path]["run_s"]) for s in sel])
            for k in ("patch_word_bytes", "text_word_bytes", "bytes_read", "value_bytes", "image_bytes"):
                row[k] = statistics.fmean([s[path].get(k, 0) for s in sel])
            # estimates: the bytes moved at the published PCIe figure (ISCA22 section 5.2: 3.7 GB/s) and at 25 GB/s; "moved" = patch-site words and
            # text words written, patch values, the logits words read; the images are listed apart (this serving mode loads them at every step)
            moved = row["patch_word_bytes"] + row["text_word_bytes"] + row["value_bytes"] + row["bytes_read"]
            for name, bw in (("3_7", 3.7e9), ("25", 25e9)):
                row[f"estimate_moved_bytes_us_at_{name}GBps"] = moved / bw * 1e6
                row[f"estimate_moved_share_at_{name}GBps"] = moved / bw / (row["hw_us"] * 1e-6)
                row[f"estimate_moved_plus_images_share_at_{name}GBps"] = (moved + row["image_bytes"]) / bw / (row["hw_us"] * 1e-6)
            row["estimate_moved_bytes"] = moved
            out[key] = row
    return out


def load(dirs):
    steps, runs = [], []
    for d in dirs:
        f = next(pathlib.Path(d).glob("host_cost-*.json"))
        j = json.loads(f.read_text())
        runs.append(j)
        steps += j["steps"]
    return steps, runs


def main(dirs):
    steps, runs = load(dirs)
    print("runs:", len(runs), "commit:", {r["commit"] for r in runs}, "load before/after:", [(r["load_before"][0], r["load_after"][0]) for r in runs])
    pooled = {p: summarise(steps, p) for p in ("serve", "base")}
    ms = lambda x: f"{x['mean'] * 1e3:.3f}" if x else "-"
    for p, rows in pooled.items():
        print(f"\n## {p} (pooled over {len(runs)} runs)")
        print("| class | steps | cycles | hw us | outer ms | host ms | python+vLLM ms | binding patch ms | restart ms | image+read ms | ISS run ms | share host (mean / median / min / max) | share python |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for k, r in rows.items():
            sh, sp = r["share_host"], r["share_python"]
            print(f"| {k} | {r['steps']} | {r['cycles']} | {r['hw_us']:.2f} | {ms(r['outer_s'])} | {ms(r['host_s'])} | {ms(r['python_s'])} | {ms(r['patch_s'])} | {ms(r['restart_s'])} | {ms(r['image_and_read_s'])} | {ms(r['run_s'])} | {sh['mean']:.1f} / {sh['median']:.1f} / {sh['min']:.1f} / {sh['max']:.1f} | {sp['mean']:.2f} |")
    print("\n## bytes per step and estimates (served path)")
    print("| class | patch values B | patch-site words B | text words B | logits words read B | images B (this mode) | moved B | est. us at 3.7 GB/s | share at 3.7 | share at 25 | share incl. images at 3.7 | at 25 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in pooled["serve"].items():
        print(f"| {k} | {r['value_bytes']:.0f} | {r['patch_word_bytes']:.0f} | {r['text_word_bytes']:.0f} | {r['bytes_read']:.0f} | {r['image_bytes']:.0f} | {r['estimate_moved_bytes']:.0f} | {r['estimate_moved_bytes_us_at_3_7GBps']:.1f} | {r['estimate_moved_share_at_3_7GBps']:.2f} | {r['estimate_moved_share_at_25GBps']:.2f} | {r['estimate_moved_plus_images_share_at_3_7GBps']:.2f} | {r['estimate_moved_plus_images_share_at_25GBps']:.2f} |")
    print("\n## per run (served, steady): mean share of host / of python, the noise floor")
    for i, run in enumerate(runs):
        s = summarise(run["steps"], "serve")
        print(f"run {i + 1} load {run['load_before'][0]:.1f}->{run['load_after'][0]:.1f}: " + "; ".join(f"{k} host {s[k]['share_host']['mean']:.1f} python {s[k]['share_python']['mean']:.2f} (n={s[k]['steps']})" for k in s if k.endswith("steady")))


if __name__ == "__main__":
    main(sys.argv[1:])
