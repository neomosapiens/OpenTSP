"""Text sizing of the batched steps: the paged instruction text per MEM slice that the pager takes for a
bucket P(T, L, C), measured from the real emitter.

The bucket is built against a serve plan whose `tx<slice>` regions are roomy (`plan_serve(text=...)`); the footprint of a
slice is the distance from the end of its region to the lowest block address the pager took. `layout.TEXT_WORDS` pins the
largest footprint over the 18 buckets of the grid for each model; `test_step_buckets.py` re-derives the pins of the buckets
it builds and checks that every bucket pages into the pinned regions.

    python text_sizing.py --layers 2 [--bucket T L C ...] [--table]      # footprints, or the TEXT_WORDS entry (max over buckets)
"""
import argparse
import json
import time

from opentsp_vllm.models.tinystories import emit_step_batch as esb
from opentsp_vllm.models.tinystories import layout
from opentsp_tools import profile as tprofile

ROOM = 1500            # words of a roomy `tx` region: more than any slice takes


def buckets():
    return [(T, L, C) for C in esb.GRID_C for T in esb.GRID_T for L in esb.GRID_L if L <= T]


def roomy_plan(layers, profile="ref", cfg=None):
    return layout.plan_serve(cfg or layout.serve_config(layers), layout.load_profile(profile),
                             text={s: ROOM for s in layout.TEXT_SLICES})


def footprint(batch, plan):
    """{slice: words} the pager took for the blocks of the built batch."""
    regions = esb.text_regions(plan)
    low = {}
    for blk in batch.kit.blocks:
        for _, s, a, _ in blk.tensors:
            low[s] = min(low.get(s, 1 << 30), a)
    return {s: regions[s][1] - low[s] for s in sorted(low)}


def measure(T, L, C, layers, plan=None, profile="ref", cfg=None):
    """({slice: words}, seconds) of the bucket on the roomy plan (or `plan`) of `profile` (`cfg`: a model other than the two pinned)."""
    plan = plan or roomy_plan(layers, profile, cfg)
    t0 = time.time()
    batch = esb.emit_step_batch(T, L, C, tprofile.load(profile), layers, plan=plan, cfg=cfg)
    return footprint(batch, plan), time.time() - t0


def measured_plan(profile, layers, bucket, margin=None, cfg=None):
    """The serve plan of `profile` whose text regions are the footprint of `bucket` (measured on a roomy plan) plus the margin:
    the pinned sizes (`layout.TEXT_WORDS_T8`) were measured on `ref`, and the text of a profile with other timing differs. The
    plan fits that bucket only; a grid of buckets that share one layout takes the largest footprint of every slice (`largest`)."""
    fp, _ = measure(*bucket, layers, profile=profile, cfg=cfg)
    margin = margin or layout.TEXT_MARGIN_PCT
    text = {s: -(-margin * max(fp.get(s, 0), 1) // 100) for s in layout.TEXT_SLICES}
    return layout.plan_serve(cfg or layout.serve_config(layers), layout.load_profile(profile), text=text)


def largest(per_bucket):
    """The per-slice maximum over {bucket: {slice: words}}."""
    out = {}
    for fp in per_bucket.values():
        for s, w in fp.items():
            out[s] = max(out.get(s, 0), w)
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bucket", nargs=3, type=int, action="append", metavar=("T", "L", "C"))
    ap.add_argument("--table", action="store_true", help="print the largest footprint per slice as a Python dict")
    a = ap.parse_args()
    got = {}
    for T, L, C in (a.bucket and [tuple(b) for b in a.bucket]) or buckets():
        fp, secs = measure(T, L, C, a.layers)
        got[(T, L, C)] = fp
        print(f"P({T},{L},{C}) {a.layers} layers: {sum(fp.values())} words, largest slice {max(fp.values())}, {secs:.0f} s", flush=True)
    if a.table:
        print(json.dumps(largest(got)))


if __name__ == "__main__":
    main()
