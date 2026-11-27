"""Test `copy_blocks` and `zero_blocks` (`kv_cache_block_copies` and `new_block_ids_to_zero`) over the real pool layout
of the program cache: the K pool (slices 44..59) and the V pool (slices 20..35) of the 2-layer model, by aperture writes, no program.
Run after `just vllm-binding` (`just vllm-binding-test`); a missing binding or cache fails, never skips."""

import os
import pathlib
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))

import real
from opentsp_vllm.chip import IssChip  # noqa: E402
from opentsp_vllm.programs import ProgramSet  # noqa: E402

BP = real.BP
LAYERS = 2


@pytest.fixture(scope="module")
def programs():
    real.bucket_path((1, 1, 64))
    return ProgramSet(real.cache_dir(), layers=LAYERS, grid_c=(64,), buckets=[(1, 1, 64)])


def chip(programs):
    return IssChip("ref", programs=programs)


def fill(c, programs, blocks, seed):
    """Random words in every region of `blocks`; returns {(slice, word): bytes}."""
    rng = random.Random(seed)
    words = {}
    for b in blocks:
        for s, a, n in programs.pool_regions(b):
            for k in range(n):
                w = bytes(rng.randrange(256) for _ in range(c._chip.vlanes))
                c._chip.write_words(s, a + k, w)
                words[(s, a + k)] = w
    return words


def block_words(c, programs, b):
    return {(s, a + k): bytes(c._chip.read_words(s, a + k, 1)) for s, a, n in programs.pool_regions(b) for k in range(n)}


def test_the_regions_of_a_block_are_the_words_of_the_oracle_pool_layout(programs):
    """`pool_regions` covers exactly the words `kvpool.k_words` and `v_words` place a block's rows in, over all layers; the K regions are in
    slices 44..59 and the V regions in 20..35, no region of a block overlaps another block's."""
    m, geo = real.model(LAYERS), real.geometry(LAYERS)
    pool = real.pool(m.cfg, 3, 40)
    seen = {}
    for b in (0, 1, 2, 17, 39, 127, 128):
        regions = programs.pool_regions(b)
        words = {(s, a + k) for s, a, n in regions for k in range(n)}
        assert len(words) == sum(n for *_, n in regions), "regions of a block overlap"
        want = set(BP.pool_words(geo, pool, LAYERS, [b]))
        assert words == want, b
        assert {s for s, *_ in regions if s < 44} <= set(range(20, 36)) and {s for s, *_ in regions if s >= 44} == set(range(44, 60))
        for w in words:
            assert seen.setdefault(w, b) == b, f"word {w} is in blocks {seen[w]} and {b}"
    with pytest.raises(ValueError, match="outside the pool"):
        programs.pool_regions(129)


def test_a_copied_block_equals_its_source_word_for_word_over_all_layers(programs):
    c = chip(programs)
    data = fill(c, programs, [1, 2, 3, 4, 5], seed=1)
    before = {b: block_words(c, programs, b) for b in (1, 2, 3, 4, 5)}
    c.copy_blocks([(1, 2), (3, 4)])
    # word for word, region by region: the destination's word k of a region is the source's
    for src, dst in ((1, 2), (3, 4)):
        for (s, a, n), (s2, a2, n2) in zip(programs.pool_regions(src), programs.pool_regions(dst)):
            assert (s, n) == (s2, n2)
            for k in range(n):
                assert bytes(c._chip.read_words(s2, a2 + k, 1)) == data[(s, a + k)]
    assert block_words(c, programs, 1) == before[1] and block_words(c, programs, 3) == before[3]
    assert block_words(c, programs, 5) == before[5], "a block that is not a target is untouched"
    # copies apply in order: 1 -> 5, then 5 -> 3 gives block 1's words in both
    c.copy_blocks([(1, 5), (5, 3)])
    for dst in (5, 3):
        for (s, a, n), (s2, a2, _) in zip(programs.pool_regions(1), programs.pool_regions(dst)):
            assert bytes(c._chip.read_words(s2, a2, n)) == b"".join(data[(s, a + k)] for k in range(n))


def test_a_zeroed_block_is_zero_and_its_neighbours_are_not(programs):
    c = chip(programs)
    fill(c, programs, [1, 2, 3, 4], seed=2)
    before = {b: block_words(c, programs, b) for b in (1, 2, 3, 4)}
    assert all(any(w) for ws in before.values() for w in ws.values()), "the random fill is not zero"
    c.zero_blocks([2, 3])
    for b in (2, 3):
        assert all(w == bytes(len(w)) for w in block_words(c, programs, b).values())
    assert block_words(c, programs, 1) == before[1] and block_words(c, programs, 4) == before[4]
    c.zero_pool()
    for b in (1, 4, 127, 128):
        assert all(w == bytes(len(w)) for w in block_words(c, programs, b).values())


def test_the_sink_and_null_blocks_are_refused_as_targets_and_nothing_is_written(programs):
    c = chip(programs)
    fill(c, programs, [1, 2], seed=3)
    fill(c, programs, [0, 128], seed=4)
    digest = c._chip.sram_digest()
    for call in (
        lambda: c.zero_blocks([0]),
        lambda: c.zero_blocks([128]),
        lambda: c.zero_blocks([1, 128]),
        lambda: c.zero_blocks([129]),
        lambda: c.copy_blocks([(1, 0)]),
        lambda: c.copy_blocks([(1, 128)]),
        lambda: c.copy_blocks([(1, 2), (2, 128)]),
        lambda: c.copy_blocks([(0, 1)]),
        lambda: c.copy_blocks([(128, 1)]),
    ):
        with pytest.raises(ValueError, match="null block|outside the pool"):
            call()
        assert c._chip.sram_digest() == digest, "a refused call writes nothing, not even its valid part"
