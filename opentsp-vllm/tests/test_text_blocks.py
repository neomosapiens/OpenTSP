"""The binding patches the sites of the paged text blocks .

The programs are the real cached buckets of the 2-layer test model and, for the case where every site is in a text block, P(1, 1, 64)
of the full model. The Python reference is `opentsp_tools.patch.apply_paged` over the cache's `program.json` and embedded ELF text
(`build_programs.Artifacts`). Run in the vLLM environment after `just vllm-binding` (`just vllm-binding-test`); a missing binding
or cache fails, never skips.
"""

import os
import pathlib
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))

import real
from opentsp_vllm import _tsp_iss  # noqa: E402
from opentsp_vllm import step_plan as SP  # noqa: E402
from opentsp_vllm.chip import TspChipError, block_records, patch_columns  # noqa: E402

patch, PC, BP, G = real.patch, real.PC, real.BP, real.G

CASES = [((1, 1, 64), 2), ((4, 4, 64), 2), ((1, 1, 64), 8)]
IDS = ["P(1,1,64) 2 layers", "P(4,4,64) 2 layers", "P(1,1,64) 8 layers"]
VOCAB = 50257


def load(c, art, name="p", model=True):
    d = pathlib.Path(art.path)
    if model:
        c.load_package(real.model(art.record["layers"]).package)
    c.load_program(
        name,
        d / "program.elf",
        patch_table=patch_columns(d / "bindings.npz"),
        blocks=block_records(art.blocks),
        outputs=art.regions,
    )


def reference(art, values):
    return patch.apply_paged(patch.to_image(art.images), art.block_bytes(), art.paged, values)


def arena(c, art):
    """{(queue, index): bytes} of the text blocks as the SRAM holds them."""
    out = {}
    for b in art.blocks:
        out[(b["queue"], b["index"])] = b"".join(bytes(c.read_words(s, a, n)) for _, s, a, n in b["tensors"])
    return out


@pytest.mark.parametrize("bucket,layers", CASES, ids=IDS)
def test_the_binding_patches_images_and_blocks_as_the_python_reference(bucket, layers):
    art = real.art(bucket, layers)
    c = _tsp_iss.Chip("ref")
    load(c, art)
    rng = random.Random(2026111400 + layers + bucket[0])
    in_blocks = int((np.asarray(art.arrays["block0"]) >= 0).sum())
    assert in_blocks > 0, "the test needs sites in text blocks"
    for _ in range(4):
        values = real.random_values(art.paged.table, rng)
        want_images, want_blocks = reference(art, values)
        images, blocks = c._prepare_for_test("p", values)
        assert set(images) == set(want_images)
        assert all(bytes(want_images[q]) == images[q] for q in want_images), "an initial image differs"
        assert set(blocks) == set(want_blocks)
        assert all(bytes(want_blocks[k]) == blocks[k] for k in want_blocks), "a text block differs"
        # the arena holds the patched blocks: every block, not only the ones with sites
        assert arena(c, art) == {k: bytes(v) for k, v in want_blocks.items()}


def test_the_sites_of_the_programs_are_in_blocks_and_in_initial_images():
    """Of the 844 sites of the 8-layer P(1, 1, 64), 420 are in text blocks and 424 in initial images. The binding
    must patch both kinds, and the differential test above covers both."""
    z = real.art((1, 1, 64), 8).arrays
    b0 = np.asarray(z["block0"])
    assert (int((b0 >= 0).sum()), int((b0 < 0).sum())) == (420, 424)


@pytest.mark.parametrize("bucket,layers", CASES[:2], ids=IDS[:2])
def test_a_value_outside_its_range_is_a_patch_error_and_changes_nothing(bucket, layers):
    art = real.art(bucket, layers)
    c = _tsp_iss.Chip("ref")
    load(c, art)
    t = art.paged.table
    rng = random.Random(77)
    ok = real.random_values(t, rng)
    c._prepare_for_test("p", ok)
    before = c.sram_digest()
    paged = np.nonzero(np.asarray(art.arrays["block0"]) >= 0)[0]
    free = np.nonzero(np.asarray(art.arrays["block0"]) < 0)[0]
    for i in [int(paged[0]), int(paged[-1]), int(free[0]), int(free[-1])]:
        n = str(t.names[i])
        for bad in (int(t.vmax[i]) + 1, int(t.vmin[i]) - 1, 1 << 40, -1 - (1 << 40)):
            with pytest.raises(_tsp_iss.PatchError, match=n):
                c._prepare_for_test("p", {**ok, n: bad})
            assert c.sram_digest() == before
    with pytest.raises(_tsp_iss.PatchError, match="unknown patch sites"):
        c._prepare_for_test("p", {**ok, "nosuch": 1})
    with pytest.raises(_tsp_iss.PatchError, match="no value for patch sites"):
        c._prepare_for_test("p", {k: v for k, v in ok.items() if k != str(t.names[0])})
    assert c.sram_digest() == before
    # and the next legal step is as the reference says
    nxt = real.random_values(t, rng)
    _, blocks = c._prepare_for_test("p", nxt)
    assert {k: bytes(v) for k, v in reference(art, nxt)[1].items()} == blocks


def test_a_table_that_does_not_match_the_blocks_is_refused_at_load():
    art = real.art((1, 1, 64), 2)
    d = pathlib.Path(art.path)
    cols = patch_columns(d / "bindings.npz")
    recs = block_records(art.blocks)
    c = _tsp_iss.Chip("ref")
    common = dict(outputs=art.regions)
    for what, kw, text in [
        ("no blocks", dict(patch_table=cols), "blocks"),
        ("blocks without a paged table", dict(patch_table={k: v for k, v in cols.items() if k != "block_units"}, blocks=recs), "paged"),
        ("a block of the wrong size", dict(patch_table={**cols, "block_units": 80}, blocks=recs), "block"),
        ("a missing block", dict(patch_table=cols, blocks=recs[1:]), "block"),
        ("a tensor the package lacks", dict(patch_table=cols, blocks=[(recs[0][0], recs[0][1], [("nosuch", 0, 0, 2)])] + recs[1:]), "nosuch"),
    ]:
        with pytest.raises(ValueError, match=text):
            c.load_program("x", d / "program.elf", **common, **kw)


# ---- an ISS step with patched blocks is bit-exact reference-exact ----


def plan_of(bucket, lanes, n_log):
    real_lanes = [SP.Lane(f"r{i}", int(t), int(p), tuple(tab[: p // 16 + 1]), i < n_log) for i, (t, p, tab) in enumerate(lanes)]
    plan = SP.StepPlan(
        bucket, tuple(real_lanes) + (SP.PAD,) * (bucket[0] - len(real_lanes)), len(real_lanes), n_log, tuple(f"r{i}" for i in range(n_log))
    )
    plan.patches = SP.patch_values(plan, VOCAB)
    return plan


STEPS = [
    ((1, 1, 64), [(777, 20, [3, 5])], 1),
    ((4, 4, 64), [(31337, 0, [5]), (1234, 16, [7, 2]), (4242, 21, [7, 3]), (50000, 40, [8, 6, 1])], 4),
    ((1, 1, 64), [(4242, 21, [3, 5])], 1),  # reads the row the first step wrote: the chip's pool and bit-exact reference's agree
    ((4, 4, 64), [(9, 33, [5, 6, 7]), (13, 63, [9, 4, 2, 11]), (777, 5, [12]), (31337, 1, [13])], 2),
    ((4, 4, 64), [(5, 2, [14]), (6, 3, [15]), (7, 4, [16]), (8, 5, [17])], 4),  # the same bucket again: no text switch
]
STEPS8 = [((1, 1, 64), [(777, 20, [3, 5])], 1), ((1, 1, 64), [(4242, 21, [3, 5])], 1)]


def run_steps(layers, steps, nblk=40, mode="warm", seen=None):
    arts = {b: real.art(b, layers) for b in {s[0] for s in steps}}
    m = real.model(layers)
    geo = real.geometry(layers)
    c = _tsp_iss.Chip("ref")
    c.restart_mode(mode)
    c.load_package(m.package)
    for b, a in arts.items():
        load(c, a, name=PC.bucket_name(b), model=False)
    assert c.stats()["text_switches"] == 0, "the text of a paged program is written by its first step"
    for (s, a_), w in BP.pool_words(geo, real.pool(m.cfg, 5, nblk), layers, list(range(nblk))).items():
        c.write_words(s, a_, w)
    want_pool = real.pool(m.cfg, 5, nblk)
    switches, last = 0, None
    seen = [] if seen is None else seen
    for k, (bucket, lanes, n_log) in enumerate(steps):
        a = arts[bucket]
        plan = plan_of(bucket, lanes, n_log)
        values = PC.plan_site_values(a.record["geometry"], [str(n) for n in a.paged.table.names], plan)
        words, cycles, diags = c.step(PC.bucket_name(bucket), values, 2 * a.record["schedule_cycles"])
        switches += bucket != last
        last = bucket
        assert c.stats()["text_switches"] == switches
        assert list(diags) == []
        want = G.step_batch(
            m.w, want_pool, [t for t, _, _ in lanes], [p for _, p, _ in lanes], [tab[: p // 16 + 1] for _, p, tab in lanes], bucket, n_log
        )
        got = PC.decode_logits(words, bucket[1], VOCAB, a.index)
        for r in range(n_log):
            assert got[r].tobytes() == want.logits[r].tobytes(), (k, bucket, r)
        if a.record["cycles"] is not None:  # a bucket built without a recorded run has none
            assert abs(cycles - a.record["cycles"]) <= 40, (cycles, a.record["cycles"])
        seen.append((bucket, cycles, a.record["schedule_cycles"]))
    # the pool the chip holds is bit-exact reference's, word for word (the sink block is outside the compared blocks)
    want_words = BP.pool_words(geo, want_pool, layers, list(range(nblk)))
    bad = [(s, hex(a_)) for (s, a_), w in sorted(want_words.items()) if bytes(c.read_words(s, a_, 1)) != w]
    assert not bad, bad[:4]
    return c


def test_a_step_with_patched_blocks_equals_the_reference_oracle_across_buckets():
    c = run_steps(2, STEPS)
    st = c.stats()
    assert st["text_switches"] == 4 and st["steps"] == 5 and st["patch_words"] > 0


def test_a_step_of_the_full_model_with_patched_blocks_equals_the_reference_oracle():
    run_steps(8, STEPS8)


def test_warm_and_replay_restarts_give_the_same_steps_and_sram_with_paged_programs():
    warm, replay = run_steps(2, STEPS), run_steps(2, STEPS, mode="replay")
    assert replay.stats()["restarts_replay"] == len(STEPS) and warm.stats()["restarts_warm"] == len(STEPS)
    assert warm.sram_digest() == replay.sram_digest()


# ---- the edges of the value grid, the types, the arena over a sequence of buckets ----


def _on_grid_below(t, i, v):
    lo, st, sp = int(t.vmin[i]), int(t.stride[i]), int(t.span[i])
    while (v - lo) % st >= sp:
        v -= 1
    return v


def _extreme_sets(t):
    """Value sets at the edges: every site at vmin, every site at the largest value of its range on its grid, the last value of the first
    and of the last grid cell."""
    n = len(t.names)
    lo = {str(t.names[i]): int(t.vmin[i]) for i in range(n)}
    hi = {str(t.names[i]): _on_grid_below(t, i, int(t.vmax[i])) for i in range(n)}
    first = {str(t.names[i]): int(t.vmin[i]) + int(t.span[i]) - 1 for i in range(n)}
    last = {}
    for i in range(n):
        top = _on_grid_below(t, i, int(t.vmax[i]))
        last[str(t.names[i])] = top - (top - int(t.vmin[i])) % int(t.stride[i]) if int(t.stride[i]) > 1 else top
    return [lo, hi, first, last]


@pytest.mark.parametrize("bucket,layers", CASES[:2], ids=IDS[:2])
def test_the_extremes_and_the_edges_of_the_value_grid_equal_the_reference(bucket, layers):
    art = real.art(bucket, layers)
    t = art.paged.table
    c = _tsp_iss.Chip("ref")
    load(c, art)
    for values in _extreme_sets(t):
        want_images, want_blocks = reference(art, values)
        images, blocks = c._prepare_for_test("p", values)
        assert all(bytes(want_images[q]) == images[q] for q in want_images)
        assert all(bytes(want_blocks[k]) == blocks[k] for k in want_blocks)
        assert arena(c, art) == {k: bytes(v) for k, v in want_blocks.items()}
    # a value off the grid is refused by both
    off = [i for i in range(len(t.names)) if int(t.span[i]) < int(t.stride[i])]
    assert off, "the bucket has gridded sites (stride > span)"
    ok = real.random_values(t, random.Random(5))
    i = off[0]
    bad = {**ok, str(t.names[i]): int(t.vmin[i]) + int(t.span[i])}
    with pytest.raises(patch.PatchError):
        reference(art, bad)
    with pytest.raises(_tsp_iss.PatchError, match=str(t.names[i])):
        c._prepare_for_test("p", bad)


def test_a_value_of_the_wrong_type_is_refused_by_both_and_numpy_integers_are_accepted_by_both():
    art = real.art((1, 1, 64), 2)
    t = art.paged.table
    c = _tsp_iss.Chip("ref")
    load(c, art)
    ok = real.random_values(t, random.Random(11))
    name = str(t.names[0])
    for bad in (3.0, "3", None, True, b"3", [3], np.float64(3), 2.5):
        with pytest.raises(patch.PatchError):
            reference(art, {**ok, name: bad})
        with pytest.raises(_tsp_iss.PatchError):
            c._prepare_for_test("p", {**ok, name: bad})
    for kind in (np.int64, np.int32, np.uint16):
        v = {**ok, name: kind(ok[name])} if np.iinfo(kind).min <= ok[name] <= np.iinfo(kind).max else ok
        want_images, want_blocks = reference(art, v)
        images, blocks = c._prepare_for_test("p", v)
        assert all(bytes(want_images[q]) == images[q] for q in want_images)
        assert all(bytes(want_blocks[k]) == blocks[k] for k in want_blocks)


def test_the_arena_holds_the_current_bucket_over_a_sequence_of_buckets():
    """The buckets share the text arena (their text tensors overlap): after every step the arena over the blocks of the current
    bucket is `apply_paged`'s, whichever buckets ran before. Fails on a text switch that is skipped after the first one."""
    buckets = [(1, 1, 64), (4, 4, 64), (2, 2, 64)]
    arts = {b: real.art(b, 2) for b in buckets}
    c = _tsp_iss.Chip("ref")
    c.load_package(real.model(2).package)
    for b in buckets:
        load(c, arts[b], name=PC.bucket_name(b), model=False)
    seen = {}
    for b in buckets:
        for blk in arts[b].blocks:
            for _, sl, a, n in blk["tensors"]:
                for k in range(n):
                    seen.setdefault((sl, a + k), set()).add(b)
    shared = sum(1 for v in seen.values() if len(v) > 1)
    assert shared > 100, "the buckets must share words of the arena for this test to mean anything"
    rng = random.Random(3)
    sequence = [buckets[i] for i in (0, 1, 2, 0, 2, 1, 1, 0, 2, 1, 0)]
    switches, last = 0, None
    for b in sequence:
        art = arts[b]
        values = real.random_values(art.paged.table, rng)
        _, blocks = c._prepare_for_test(PC.bucket_name(b), values)
        switches += b != last
        last = b
        want = {k: bytes(v) for k, v in reference(art, values)[1].items()}
        assert blocks == want
        assert arena(c, art) == want, b
        assert c.stats()["text_switches"] == switches
    assert switches == 10  # every change of bucket, none for the repeat


# ---- the resident text is forgotten when the arena may hold something else ----


def _first_text_words(art):
    blk = art.blocks[0]
    _, sl, a, n = blk["tensors"][0]
    return sl, a, n


def test_a_write_into_the_text_region_or_the_load_of_a_plain_program_makes_the_next_step_write_the_text_again():
    a1, a4 = real.art((1, 1, 64), 2), real.art((4, 4, 64), 2)
    c = _tsp_iss.Chip("ref")
    load(c, a1, name="P1")
    values = real.random_values(a1.paged.table, random.Random(8))
    want = {k: bytes(v) for k, v in reference(a1, values)[1].items()}

    def step():
        c._prepare_for_test("P1", values)
        assert arena(c, a1) == want

    step()
    assert c.stats()["text_switches"] == 1
    step()
    assert c.stats()["text_switches"] == 1, "nothing touched the arena"
    # a write that does not meet a text tensor (slice 0, word 0 is not in any block) keeps the text
    c.write_words(0, 0, bytes(c.vlanes))
    step()
    assert c.stats()["text_switches"] == 1
    # a write into a text tensor of the resident program: the text is written again
    sl, addr, n = _first_text_words(a1)
    c.write_words(sl, addr, bytes(n * c.vlanes))
    assert arena(c, a1) != want
    step()
    assert c.stats()["text_switches"] == 2
    # a program that is not paged writes its text at load (here the text of P(4,4,64)): the arena is no longer P1's
    d4 = pathlib.Path(a4.path)
    c.load_program("plain", d4 / "program.elf", outputs=[])
    assert arena(c, a1) != want, "the plain program's text replaced words of P1's"
    step()
    assert c.stats()["text_switches"] == 3


def test_a_panic_inside_a_step_keeps_the_program_loaded_and_the_text_is_written_again():
    art = real.art((1, 1, 64), 2)
    c = _tsp_iss.Chip("ref")
    load(c, art, name="P1")
    plan = plan_of((1, 1, 64), [(777, 20, [3, 5])], 1)
    values = PC.plan_site_values(art.record["geometry"], [str(n) for n in art.paged.table.names], plan)
    limit = 2 * art.record["schedule_cycles"]
    words, cycles, diags = c.step("P1", values, limit)
    steps = c.stats()["steps"]
    c._panic_in_next_step_for_test()
    with pytest.raises(TspChipError, match="panic in the simulator: deliberate panic of _panic_in_next_step_for_test"):
        c.step("P1", values, limit)
    assert c.stats()["steps"] == steps
    switches = c.stats()["text_switches"]
    again = c.step("P1", values, limit)  # still loaded: not "no program loaded"
    assert again[1] == cycles and again[0] == words and list(again[2]) == []
    assert c.stats()["text_switches"] == switches + 1, "after a panic the arena is not known to hold the text"
    with pytest.raises(ValueError, match="no program"):
        c.step("nosuch", values, limit)
