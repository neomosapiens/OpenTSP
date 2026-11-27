"""Exercise the batched, paged steps of the whole grid (`emit_step_batch.emit_step_batch`), 18 buckets P(T, L, C) on the 8-layer
model and 9 on the 2-layer test model.

* One placement: every address the program uses, the vectors of the Flow included, lies in a region of the planner
  (`layout.plan_serve`), the same for every bucket. The overlap test computes the extents of the whole program (the
  Reads and Writes with their loops, the patched sites' ranges, the externs and the paged text blocks) for P(1, 1, 64) and
  P(8, 4, 128) on both models and checks them against the planned regions and against each other.
* Text sizing: `layout.TEXT_WORDS_*` are re-derived from the emitter; the grid pages into the pinned regions.
* bit-exact reference-exact: the buckets of the sample, with lanes in different states (a fresh prompt, a context that crosses a block
  boundary, a block table that shares a prefix block with another lane, padding lanes, logit rows 1.. of a lane that is not
  lane 0), compare the logits, the final hidden state and every pool block with `paged_reference.step_batch`.
"""
import os
import re

import numpy as np
import pytest


from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import text_sizing as TS
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories import kvpool
import test_step_batch as SB

GRID = [(T, Ls, C) for C in E.GRID_C for T in E.GRID_T for Ls in E.GRID_L if Ls <= T]
_built = {}


def build(T, Ls, C, layers):
    """The batch of a bucket, built once per test run."""
    key = (T, Ls, C, layers)
    if key not in _built:
        _built[key] = E.emit_step_batch(T, Ls, C, layers=layers)
    return _built[key]


# -- the extents of a program ---------------------------------------------------------------------------------------------
def int_(x):
    return int(x, 16) if x.startswith("0x") else int(x)


def program_extents(batch):
    """[(slice, first word, end word, what)] of every word the whole program (the paged text included: `view2.full`) addresses:
    a Read or Write with its AGT loop (`step`, `iterations`: `iterations` words `step` apart), the range of a patched site
    (every value the host may put there, and its loop), an extern by name and offset."""
    full = batch.kit.view2.full
    externs = {n: (s, a, size) for n, s, a, size in full.externs}
    sites = {x["name"]: x for x in batch.case["sites"]}
    out, last, step = [], {}, {}
    for c, q, text in sorted(full.entries, key=lambda e: (e[0], e[1])):
        w = text.replace(",", " ").split()
        if w[0] in ("read", "write"):
            ref = w[1]
            if ref.startswith("@"):
                site = sites[ref[1:]]
                lo, hi = site["min"], site["max"] + 1
                m = re.fullmatch(r"t\d+\.emb\.p(\d)[lh]", ref[1:])
                if m:        # a lookup reads a row of its part (0..rows - 1) or the part's zero word (the first word of bank 1)
                    rows = batch.sizes.part_rows[int(m[1])]
                    out.append([q, 0, rows, ref])
                    lo, hi = site["max"], site["max"] + 1
            elif ref in externs:
                lo = externs[ref][1]
                hi = lo + 1
            elif "+" in ref:
                name, off = ref.split("+")
                lo = externs[name][1] + int_(off)
                hi = lo + 1
            else:
                lo = int_(ref)
                hi = lo + 1
            out.append([q, lo, hi, ref])
            last[q], step[q] = len(out) - 1, 1
        elif w[0] == "step":
            step[q] = int_(w[1])
        elif w[0] == "iterations":
            e = out[last[q]]
            e[2] += step[q] * (int_(w[1]) - 1)
    for n, (s, a, size) in externs.items():
        out.append([s, a, a + size, n])
    return sorted({tuple(e) for e in out})


def region_of(plan, s, lo, hi):
    """The planned regions of slice `s` that contain every word of [lo, hi): a list (empty: not planned; more than one:
    the planner has overlapping regions)."""
    return [r for r in plan.regions if s in r.slices and r.addr <= lo and hi <= r.addr + r.words]


def covered(plan, s, lo, hi):
    """Whether every word of [lo, hi) of slice `s` is in a planned region (a patched site's range may span adjacent regions:
    the embedding part and its zero word)."""
    regions = sorted((r.addr, r.addr + r.words) for r in plan.regions if s in r.slices)
    a = lo
    for x, y in regions:
        if x <= a < y:
            a = y
    return a >= hi


@pytest.mark.parametrize("layers", [2, 8])
@pytest.mark.parametrize("bucket", [(1, 1, 64), (8, 4, 128)], ids=lambda b: "P%d_%d_%d" % b)
def test_every_word_of_the_program_is_in_one_planned_region(bucket, layers):
    """The extents of the whole program lie in exactly one region each, the regions are disjoint (`Planner.validate`), the
    paged text blocks are in the slice's `tx` region and disjoint from each other and from every other extent, and the
    Flow's own words are the vectors it stored: a word of a lane's arena that the program addresses is a vector of that lane."""
    b = build(*bucket, layers)
    plan = b.plan
    plan.validate()
    ext = program_extents(b)
    bad = [e for e in ext if not covered(plan, *e[:3])]
    assert not bad, f"{len(bad)} extents leave the planned regions, first {bad[:5]}"
    straddle = [e for e in ext if len(region_of(plan, *e[:3])) != 1 and not e[3].startswith("@")]
    assert not straddle, f"{len(straddle)} extents are in no region or in several, first {straddle[:5]}"
    text = [(s, lo, hi, n) for s, lo, hi, n in ext if re.fullmatch(r"tx\d+b\d+[ab]?", n)]
    assert text and all(region_of(plan, s, lo, hi)[0].name == f"tx{s}" for s, lo, hi, _ in text)
    words = {}
    for s, lo, hi, n in text:
        for a in range(lo, hi):
            assert (s, a) not in words, f"text blocks {n} and {words[(s, a)]} share word {a} of slice {s}"
            words[(s, a)] = n
    others = [e for e in ext if e not in text and not (region_of(plan, *e[:3]) and region_of(plan, *e[:3])[0].kind == "text")]
    assert not [e for e in others if any((e[0], a) in words for a in range(e[1], min(e[2], e[1] + 4096)))]
    vectors = {(lane, hemi, quad): set() for (lane, _, hemi, quad, _) in b.flow.vectors}
    for (lane, _, hemi, quad, _), a in b.flow.vectors.items():
        vectors[(lane, hemi, quad)].add(a)
    for (lane, hemi, quad), addrs in vectors.items():
        r = plan.by_name[L.flow_arena_name(lane, hemi, quad)]
        assert all(r.addr <= a < r.addr + r.words for a in addrs)
        touched = {a for s, lo, hi, n in ext if s in r.slices and r.addr <= lo and hi <= r.addr + r.words
                   for a in range(lo, hi) if n != r.name}
        assert touched <= addrs, f"lane {lane} {hemi}{quad}: words {sorted(touched - addrs)[:5]} are addressed but no vector"


def test_the_flow_arena_pins_are_what_the_emitter_takes():
    """`layout.FLOW_ARENA` is the largest number of vectors a lane takes per (hemisphere, quad, scope), at the biggest bucket."""
    b = build(8, 4, 128, 2)
    assert b.flow.used == {(h, q, sc): n for (h, q), parts in L.FLOW_ARENA.items() for sc, n in parts.items() if n}


@pytest.mark.parametrize("pair", [((1, 1, 64), (2, 1, 64)), ((2, 2, 128), (8, 4, 128)), ((1, 1, 128), (4, 4, 128)),
                                  ((1, 1, 64), (1, 1, 128))],
                         ids=lambda p: "P%d_%d_%d_P%d_%d_%d" % (p[0] + p[1]))
def test_the_same_constant_has_the_same_address_in_every_bucket(pair):
    """Constants, tables and the vectors of a lane are at the same words whatever the bucket: not shifted by the other lanes,
    the context or the logit rows (a pair has one layer count: the constants of the layers are in layer order, so they shift
    with the number of layers; only a lane's vectors keep their offsets then)."""
    a, b = (build(*bucket, 2) for bucket in pair)
    assert a.flow.consts and a.flow.consts == b.flow.consts
    common = set(a.flow.vectors) & set(b.flow.vectors)
    assert common and all(a.flow.vectors[k] == b.flow.vectors[k] for k in common)
    assert {k[0] for k in b.flow.vectors} == set(range(pair[1][0])) and {k[0] for k in a.flow.vectors} == set(range(pair[0][0]))
    for name in ("hm", "oh"):
        slices, prefix, _ = L.FLOW_TABLES[name]
        assert a.plan.by_name[f"{prefix}{slices[0]}"].addr == b.plan.by_name[f"{prefix}{slices[0]}"].addr


def test_the_lanes_arenas_do_not_overlap_and_the_scopes_do_not_share_words():
    plan = build(2, 1, 64, 2).plan
    for (hemi, quad), parts in L.FLOW_ARENA.items():
        spans = []
        for sc in L.FLOW_SCOPES:
            lo = L.flow_arena_offset(hemi, quad, sc)
            spans.append((lo, lo + parts.get(sc, 0)))
        assert all(x[1] <= y[0] for x, y in zip(spans, spans[1:])), (hemi, quad, spans)
        regions = [plan.by_name[L.flow_arena_name(t, hemi, quad)] for t in range(L.LANES_MAX)]
        assert all(r.words == sum(parts.values()) for r in regions)
        ends = sorted((r.addr, r.addr + r.words) for r in regions)
        assert all(x[1] <= y[0] for x, y in zip(ends, ends[1:]))


# -- text sizing -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("layers", [2, pytest.param(8, marks=pytest.mark.slow)])
def test_the_text_tables_are_the_emitters_text(layers):
    """Re-derive the single-lane footprint and every slice's maximum over the grid.

    Fetch placement can move text between slices: the largest token bucket is
    not necessarily the largest footprint on every slice. Measuring only the
    two P(8, 4, C) buckets misses these smaller-bucket maxima.
    """
    one, _ = TS.measure(1, 1, 128, layers)
    assert one == L.TEXT_WORDS_ONE_LANE[layers]
    grid = TS.largest({b: TS.measure(*b, layers)[0] for b in GRID})
    assert grid == L.TEXT_WORDS_T8[layers]


@pytest.mark.parametrize("layers", [2, 8])
def test_the_text_regions_have_room_for_eight_lanes(layers):
    """The pinned `tx` regions hold the text of P(8, 4, 128) (the build pages into them), with the margin the design gives; on
    the 2-layer model the K slices 52..59 too."""
    b = build(8, 4, 128, layers)
    fp = TS.footprint(b, b.plan)
    assert fp and all(w <= b.plan.by_name[f"tx{s}"].words for s, w in fp.items())
    assert all(L.TEXT_WORDS_T8[layers][s] >= w * 100 // L.TEXT_MARGIN_PCT for s, w in fp.items())
    if layers == 2:
        assert all(b.plan.by_name[f"tx{s}"].words >= L.TEXT_WORDS_T8[2][s] for s in range(52, 60))


def test_the_budget_of_the_serve_plans():
    """Both serve plans place (bank rule, no overlap) with the sized text regions, and the layout report is complete."""
    for layers in (2, 8):
        cfg = L.serve_config(layers)
        plan = L.plan_serve(cfg, L.load_profile("ref"))
        plan.validate()
        rep = L.report(plan, cfg)
        assert rep["max_bank_words"] <= plan.p.bank_words
        assert L.check_design_rows(rep)


# -- the grid ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bucket", GRID, ids=lambda b: "P%d_%d_%d" % b)
def test_every_bucket_builds_and_pages_at_two_layers(bucket):
    b = build(*bucket, 2)
    T, Ls, C = bucket
    assert b.bucket == bucket and b.case["golden"]["bucket"] == list(bucket)
    assert len(b.hidden) == Ls and {k[0] for k in b.flow.vectors} == set(range(T))
    values = E.site_values(b, [31337] * T, list(range(T)), [[5]] * T)
    assert set(values) == {x["name"] for x in b.case["sites"]}


@pytest.mark.parametrize("bucket", [(2, 2, 32), (3, 1, 64), (1, 4, 64), (4, 3, 64), (16, 4, 64), (8, 8, 128), (4, 4, 256)])
def test_a_bucket_outside_the_grid_names_the_grid(bucket):
    with pytest.raises(ValueError, match=r"not in the grid: T in \(1, 2, 4, 8\) token lanes, L in \(1, 2, 4\) logit rows"):
        E.emit_step_batch(*bucket, layers=2)


# -- host input -----------------------------------------------------------------------------------------------------------------
BAD_LANES = [("pos_needs_two_blocks_table_has_one", (2, 1, 64), 20, [7]),
             ("negative_pos", (2, 1, 64), -1, [7]),
             ("pos_beyond_context", (2, 1, 64), 70, [7, 2, 3, 4, 5]),
             ("pos_equals_context", (2, 1, 64), 64, [7, 2, 3, 4]),
             ("table_longer_than_context", (2, 1, 64), 5, [7, 2, 3, 4, 5]),
             ("table_longer_than_context_128", (1, 1, 128), 5, list(range(1, 10))),
             ("empty_table", (2, 1, 64), 0, []),
             ("block_zero_is_the_null_block", (2, 1, 64), 20, [7, 0]),
             ("block_128_is_out_of_the_pool", (2, 1, 64), 3, [128]),
             ("negative_block", (2, 1, 64), 3, [-1])]


@pytest.mark.parametrize("case", BAD_LANES, ids=lambda c: c[0])
def test_host_input_that_would_corrupt_the_null_block_is_refused(case):
    """The null block (block 0) is what every padding context reads and the sinks absorb the padding writes: a position
    beyond the block table, or a block outside 1..127, must not reach the sites."""
    _, bucket, pos, table = case
    b = build(*bucket, 2)
    with pytest.raises(ValueError):
        E.site_values(b, [5] * bucket[0], [pos] + [0] * (bucket[0] - 1), [table] + [[1]] * (bucket[0] - 1))


def test_the_edges_of_the_host_input_are_accepted():
    b = build(2, 1, 64, 2)
    E.site_values(b, [5, 6], [63, 16], [[1, 2, 3, 127], [127, 1]])
    E.site_values(b, [5], [0], [[1]])
    E.site_values(b, [5], [3], [[1, 2, 3, 4]])         # a table may be longer than the position needs


# -- bit-exact reference-exact lanes -------------------------------------------------------------------------------------------------------------
FRESH = (31337, 0, [5])                 # a fresh prompt: the first token
BOUNDARY = (1234, 16, [7, 2])           # the 17th token: the first row of the second block
SHARED_A = (4242, 21, [7, 2])           # a block table that shares its first block with SHARED_B's
SHARED_B = (777, 18, [7, 3])
LONG = (50000, 100, [3, 6, 1, 9, 11, 4, 8])     # 101 tokens in 7 blocks (C = 128)

# the sample: (bucket, the real lanes, the number of logit rows)
SAMPLE = [((1, 1, 64), [FRESH], 1),
          ((2, 1, 64), [BOUNDARY, FRESH], 1),
          ((4, 4, 64), [SHARED_A, SHARED_B, FRESH], 3),
          ((4, 4, 64), [BOUNDARY, SHARED_B], 2),                       # two padding lanes of four, two logit rows
          ((1, 1, 128), [LONG], 1),
          ((2, 2, 128), [SHARED_A, SHARED_B], 2),
          ((8, 4, 128), [LONG, SHARED_A, SHARED_B, FRESH, BOUNDARY], 4)]


def check_lanes(batch, lanes, n_logits, exe=SB.ISS, seed=100):
    """One step of the batch with `lanes` (token, position, block table) as the real lanes, on the chip against bit-exact reference: the
    logits of the first `n_logits` lanes, their final hidden state and every word of the K and V pools (blocks 0..11 random
    before the step, the others zero, and the sinks) after it. The sinks are the padding lanes' and the non-owning K pairs'
    to write, so the K sink is not compared and the V sink only when there is no padding lane. Returns the run's record."""
    T = batch.bucket[0]
    layers, geo = batch.layers, batch.geo
    cfg = batch.cfg or L.serve_config(layers)
    tokens, positions, tables = [list(x) for x in zip(*lanes)]
    tokens = [t % cfg.vocab for t in tokens]               # a model with a smaller vocabulary takes the sample's tokens modulo it
    w = W.synthetic(cfg, E.SEED)
    pool_pre, pool_post = SB.random_pool(cfg, seed), SB.random_pool(cfg, seed)
    want = G.step_batch(w, pool_post, tokens, positions, tables, batch.bucket, n_logits, unit=batch.kit.p.vlanes, pack=SB.shape_of(batch).pack)
    pre = SB.pool_words(geo, pool_pre, layers)
    every = list(range(geo.sink + 1))
    after = SB.pool_words(geo, pool_post, layers, every)
    dumps = [(s, a) for r in range(n_logits) for _, _, s, a in SB.logit_words(batch, r)]
    dumps += [(s, batch.hidden[r].addr) for r in range(n_logits) for s in batch.hidden[r].slices]
    out = SB.run(exe, batch, E.site_values(batch, tokens, positions, tables), dict(pre), sorted(after) + dumps, batch.kit.last + 200)
    assert out["diags"] == [] and out["lints"] == []
    sink = {sa for l in range(layers) for sa in kvpool.k_words(geo, pool_post, l, [geo.sink])}
    if len(lanes) < T:
        sink |= {sa for l in range(layers) for sa in kvpool.v_words(geo, pool_post, l, [geo.sink])}
    vl = batch.kit.p.vlanes
    bad = [(s, hex(a)) for (s, a), word in sorted(after.items()) if (s, a) not in sink and out["words"][(s, a)][:vl] != word]
    assert not bad, f"{len(bad)} pool words differ, first {bad[:6]}"
    for r in range(n_logits):
        hid = batch.hidden[r]
        got = np.ascontiguousarray(np.stack([np.frombuffer(out["words"][(s, hid.addr)][:64], np.uint8) for s in hid.slices],
                                            axis=1)).reshape(-1).view(np.float32)
        assert got.tobytes() == want.hidden[r].tobytes(), f"hidden state of lane {r}"
        assert SB.chip_logits(out["words"], batch, r, cfg.vocab).tobytes() == want.logits[r].tobytes(), f"logits of lane {r}"
    return out


def sample_id(case):
    (T, Ls, C), lanes, n = case
    return f"P{T}_{Ls}_{C}_{len(lanes)}lanes_{n}rows"


@pytest.mark.parametrize("case", SAMPLE, ids=sample_id)
def test_the_two_layer_sample_is_reference_exact_on_the_iss(case):
    bucket, lanes, n = case
    check_lanes(build(*bucket, 2), lanes, n)


def test_a_two_lane_bucket_is_reference_exact_on_tsp_sim():
    assert os.path.exists(SB.SIM), (f"tsp_sim is not built at {SB.SIM}: `just showcase-reference` builds it "
                                    f"(bash opentsp-arch-spec/tools/build_sim.sh target/spec/tsp_sim; TSP_SIM names another)")
    check_lanes(build(2, 2, 128, 2), [SHARED_A, SHARED_B], 2, exe=SB.SIM)


def test_a_wrong_lane_site_is_caught(monkeypatch):
    """Lane 1's block table sent to lane 0's blocks (a missing per-lane patch) changes the result of lane 1."""
    b = build(2, 2, 64, 2)
    orig = E.site_values

    def wrong(batch, tokens, positions, tables):
        return orig(batch, tokens, positions, [tables[0], tables[0]])
    monkeypatch.setattr(E, "site_values", wrong)
    with pytest.raises(AssertionError):
        check_lanes(b, [SHARED_A, SHARED_B], 2)


FULL = [((1, 1, 64), [FRESH], 1), ((1, 1, 128), [LONG], 1),
        ((4, 4, 128), [LONG, SHARED_A, SHARED_B, BOUNDARY], 4), ((8, 4, 128), [LONG, SHARED_A, SHARED_B, FRESH, BOUNDARY], 4)]


@pytest.mark.parametrize("case", FULL, ids=sample_id)
def test_the_eight_layer_buckets_are_reference_exact_on_the_iss(case):
    """The decode buckets P(1, 1, 64) and P(1, 1, 128), and P(4, 4, 128) and P(8, 4, 128), on the 8-layer model; the build and run
    times are reported."""
    bucket, lanes, n = case
    out = check_lanes(build(*bucket, 8), lanes, n)
    print(f"P{bucket}: build {build(*bucket, 8).seconds}, ISS run {out['seconds']:.0f} s")
