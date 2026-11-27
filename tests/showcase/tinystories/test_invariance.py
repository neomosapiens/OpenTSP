"""Test batch invariance. A token's logits do not depend on its lane position, on the other lanes of the
step, on the T bucket or on the C bucket. The claim is tested twice:

* on the chip: the cached programs of the 2-layer model (`build_programs`) run on the ISS, each step patched from a `StepPlan`
  of `opentsp_vllm` (read-only import), and the logit row of the token under test is compared, bit for bit, with what bit-exact reference
  (`paged_reference.step_batch`) says for that token alone in the smallest bucket (`P(1, 1, C)`);
* on the oracle: the same schedule over all 18 buckets without the chip (bit-exact reference must hold the property that the chip is meant to).

The schedule is seeded; every failure message carries the seed and the step. The only difference accepted is the sign of a zero
(`-0.0` against `+0.0`), which the tests count and print (the design's "signed-zero corner").

What differs between configurations, and what does not: the token, its position, its block table and the contents of the pool are
the token's own inputs and are the same in every configuration; the bucket, the lane index, the other lanes (their tokens,
positions, block tables, which may share the full prefix blocks of the token's table, never a block the token reads that they
write) and the padding lanes are what varies.
"""
import os
import random
import sys

import numpy as np
import pytest

from opentsp_vllm.models.tinystories import paths as _paths

sys.path.insert(0, os.path.join(_paths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.tinystories import build_programs as BP
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_vllm import step_plan as SP
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools import profile as tprofile

VOCAB = 50257
LAYERS = 2
NBLK = 40                       # blocks 1..39 hold random K and V rows, the others are zero
SEED = int(os.environ.get("INVARIANCE_SEED", "20261114"))
CHIP_BUCKETS = [(1, 1, 64), (2, 1, 64), (2, 2, 64), (4, 4, 64), (8, 4, 64), (1, 1, 128), (4, 2, 128), (8, 4, 128)]
GRID = [(T, Ls, C) for C in (64, 128) for T in (1, 2, 4, 8) for Ls in (1, 2, 4) if Ls <= T]


# -- the inputs -----------------------------------------------------------------------------------------------------------------
def make_pool(cfg, seed):
    pool = G.BlockPool(cfg, G.POOL_BLOCKS)
    rng = np.random.default_rng(seed)
    pool.k[1:NBLK] = (rng.standard_normal(pool.k[1:NBLK].shape) * 0.5).astype(np.float16)
    pool.v[1:NBLK] = rng.standard_normal(pool.v[1:NBLK].shape).astype(np.float32)
    return pool


def random_step(rng, buckets):
    """(bucket, lanes, n_logits, index of the token under test, pool seed): lanes are (token, position, block table), the logit
    lanes first; the token under test is a logit lane."""
    bucket = rng.choice(buckets)
    T, Ls, C = bucket
    pos = rng.randrange(64 if C == 64 or rng.random() < 0.7 else C)
    ids = list(range(1, NBLK))
    rng.shuffle(ids)
    table = [ids.pop() for _ in range(pos // 16 + 1)]
    test = (rng.randrange(VOCAB), pos, table)
    n_real = rng.randint(1, T)
    n_log = rng.randint(1, min(Ls, n_real))
    others = []
    for _ in range(n_real - 1):
        k = rng.randrange(pos // 16 + 1) if rng.random() < 0.5 else 0          # full blocks shared with the token under test
        m = rng.randint(1, max(1, min(2, C // 16 - k)))                         # blocks of its own: it writes into the last
        cpos = 16 * (k + m - 1) + rng.randrange(16)
        others.append((rng.randrange(VOCAB), cpos, table[:k] + [ids.pop() for _ in range(m)]))
    j = rng.randrange(n_log)
    lanes = others[:j] + [test] + others[j:n_log - 1] + others[n_log - 1:]
    return bucket, lanes, n_log, j, rng.randrange(1 << 30)


def canonical(bucket_c, token, pos, table):
    return (1, 1, 64 if pos < 64 else 128)


def plan_of(bucket, lanes, n_log):
    real = [SP.Lane(f"r{i}", int(t), int(p), tuple(tab[:p // 16 + 1]), i < n_log) for i, (t, p, tab) in enumerate(lanes)]
    plan = SP.StepPlan(bucket, tuple(real) + (SP.PAD,) * (bucket[0] - len(real)), len(real), n_log,
                       tuple(f"r{i}" for i in range(n_log)))
    plan.patches = SP.patch_values(plan, VOCAB)
    return plan


def zero_sign(a):
    """The array with -0.0 turned into +0.0."""
    return a + np.float32(0.0)


class Worlds:
    """The 2-layer model with its bit-exact reference oracle and the cached programs of its buckets (built on first use)."""

    def __init__(self):
        self.model = PC.Model(LAYERS)
        self.iss = BP.tsp_iss()
        self.geo = kvpool.Geometry(tprofile.load("ref"), LAYERS)
        self.arts = {}
        self.config = os.path.join(_paths.SPEC, "config", f"{self.model.profile}.json")

    def art(self, bucket):
        if bucket not in self.arts:
            BP.build_bucket(self.model, bucket)
            self.arts[bucket] = BP.Artifacts(PC.bucket_dir(None, self.model, bucket), self.model.profile)
        return self.arts[bucket]

    def exact_ref(self, bucket, lanes, n_log, pool_seed):
        tokens, positions, tables = [list(x) for x in zip(*[(t, p, tab[:p // 16 + 1]) for t, p, tab in lanes])]
        pool = make_pool(self.model.cfg, pool_seed)
        return G.step_batch(self.model.w, pool, tokens, positions, tables, bucket, n_log).logits[:n_log]

    def chip(self, bucket, lanes, n_log, pool_seed):
        art = self.art(bucket)
        plan = plan_of(bucket, lanes, n_log)
        names = [str(n) for n in art.paged.table.names]
        values = PC.plan_site_values(art.record["geometry"], names, plan)
        pre = BP.pool_words(self.geo, make_pool(self.model.cfg, pool_seed), LAYERS, list(range(NBLK)))
        dumps = [(s, a + k) for s, a, n in art.regions for k in range(n)]
        out = BP.compat_run(self.iss, art, values, self.model.package, pre, dumps, 2 * art.record["schedule_cycles"], self.config)
        assert not out["diags"] and not out["lints"] and not out["returncode"], (bucket, out["diags"][:3], out["lints"][:3])
        words = b"".join(out["words"][(s, a + k)] for s, a, n in art.regions for k in range(n))
        return PC.decode_logits(words, bucket[1], VOCAB, art.index)[:n_log]


@pytest.fixture(scope="module")
def worlds():
    return Worlds()


class Tally:
    def __init__(self):
        self.steps = self.signed_zero = 0

    def same(self, got, want, what):
        """Bit-identical but for the sign of a zero; counts the steps that differ in a zero's sign only."""
        got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
        self.steps += 1
        if got.tobytes() == want.tobytes():
            return
        if zero_sign(got).tobytes() == zero_sign(want).tobytes():
            self.signed_zero += 1
            return
        bad = np.nonzero(zero_sign(got) != zero_sign(want))[0]
        raise AssertionError(f"{what}: {len(bad)} of {got.size} logits differ, first at {bad[:4].tolist()}: {got[bad[:4]].tolist()} "
                             f"against {want[bad[:4]].tolist()}")


def check_step(worlds, tally, step, chip, tag):
    bucket, lanes, n_log, j, pool_seed = step
    tok, pos, table = lanes[j]
    ref_bucket = canonical(bucket[2], tok, pos, table)
    ref = worlds.exact_ref(ref_bucket, [lanes[j]], 1, pool_seed)[0]                       # the token alone in the smallest bucket
    got = (worlds.chip if chip else worlds.exact_ref)(bucket, lanes, n_log, pool_seed)
    where = f"{tag}: bucket P{bucket}, {len(lanes)} lanes, {n_log} logit rows, token {tok} at {pos} in lane {j}"
    tally.same(got[j], ref, f"{where}: against the token alone in P{ref_bucket}")
    if chip:                                    # and the chip is bit-exact reference itself, row by row, in the bucket it ran
        same = worlds.exact_ref(bucket, lanes, n_log, pool_seed)
        for r in range(n_log):
            tally.same(got[r], same[r], f"{where}: row {r} against bit-exact reference of the same step")


# -- the schedule ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("part", [0, 1], ids=lambda p: f"steps{25 * p}-{25 * p + 24}")
def test_the_chip_is_invariant_over_a_seeded_random_schedule(worlds, part, capsys):
    """50 random steps (two parts of 25) on the 2-layer chip: lane position, batch composition, T and C bucket all vary."""
    rng = random.Random(SEED * 2 + part)
    tally = Tally()
    for i in range(25):
        step = random_step(rng, CHIP_BUCKETS)
        try:
            check_step(worlds, tally, step, True, f"seed {SEED * 2 + part} (INVARIANCE_SEED={SEED}) step {25 * part + i}")
        except AssertionError as e:
            raise AssertionError(f"INVARIANCE COUNTEREXAMPLE seed {SEED * 2 + part} step {25 * part + i}: {e}") from None
    with capsys.disabled():
        print(f"\ninvariance (chip) seed {SEED * 2 + part}: 25 steps, {tally.steps} comparisons, {tally.signed_zero} that differ in "
              f"the sign of a zero only")


def test_the_oracle_is_invariant_over_a_seeded_random_schedule_of_the_whole_grid(worlds, capsys):
    """200 random steps over all 18 buckets on bit-exact reference alone: the property the chip is meant to have."""
    rng = random.Random(SEED)
    tally = Tally()
    for i in range(200):
        step = random_step(rng, GRID)
        try:
            check_step(worlds, tally, step, False, f"seed {SEED} step {i}")
        except AssertionError as e:
            raise AssertionError(f"INVARIANCE COUNTEREXAMPLE seed {SEED} step {i}: {e}") from None
    with capsys.disabled():
        print(f"\ninvariance (bit-exact reference) seed {SEED}: 200 steps, {tally.steps} comparisons, {tally.signed_zero} that differ in the sign of "
              f"a zero only")


# -- the four axes one at a time, on the chip ---------------------------------------------------------------------------------------
TOKEN = (31337, 37, [5, 9, 2])                                   # position 37: three blocks, the last one half full
COMPANIONS = [(4242, 21, [5, 17]), (777, 18, [5, 11]), (50000, 40, [5, 9, 12]), (13, 63, [20, 21, 22, 23]),
              (9, 33, [24, 25, 26]), (1234, 2, [27]), (4, 3, [28])]
SEED_POOL = 7


def run_alone(worlds, bucket, lanes, j, n_log):
    return worlds.chip(bucket, lanes, n_log, SEED_POOL)[j]


def test_the_lane_position_of_a_token_does_not_change_its_logits(worlds):
    ref = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    for j in range(4):
        lanes = COMPANIONS[:j] + [TOKEN] + COMPANIONS[j:3]
        got = run_alone(worlds, (4, 4, 64), lanes, j, 4)
        assert got.tobytes() == ref.tobytes(), f"lane {j}"


def test_the_other_lanes_and_the_padding_do_not_change_its_logits(worlds):
    ref = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    for others in ([], COMPANIONS[:1], COMPANIONS[:3], COMPANIONS[:7]):
        for n_log in {1, min(4, 1 + len(others))}:
            got = run_alone(worlds, (8, 4, 64), [TOKEN] + others, 0, n_log)
            assert got.tobytes() == ref.tobytes(), f"{len(others)} other lanes, {n_log} logit rows, {7 - len(others)} padding lanes"


def test_the_t_bucket_does_not_change_its_logits(worlds):
    ref = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    for T in (1, 2, 4, 8):
        got = run_alone(worlds, (T, 1, 64), [TOKEN] + COMPANIONS[:T - 1], 0, 1)
        assert got.tobytes() == ref.tobytes(), f"T = {T}"


def test_the_c_bucket_does_not_change_its_logits(worlds, capsys):
    ref = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    for bucket in ((4, 2, 64), (4, 2, 128)):
        got = run_alone(worlds, bucket, [TOKEN] + COMPANIONS[:3], 0, 2)
        assert zero_sign(got).tobytes() == zero_sign(ref).tobytes(), f"C = {bucket[2]}"
        with capsys.disabled():
            print(f"\nC = {bucket[2]}: bit-identical {got.tobytes() == ref.tobytes()}")


# -- directed cases: the boundaries of the design's risk list, on the chip ---------------------------------
def alone(worlds, lane, pool_seed, c=None):
    """bit-exact reference of `lane` alone in the smallest bucket that holds it (or of context bucket `c`)."""
    return worlds.exact_ref((1, 1, c or canonical(None, *lane)[2]), [lane], 1, pool_seed)[0]


def check_rows(worlds, tally, bucket, lanes, n_log, pool_seed, what):
    """Every logit row of the chip equals bit-exact reference of that lane alone (bit-identical, a zero's sign aside)."""
    rows = worlds.chip(bucket, lanes, n_log, pool_seed)
    for r in range(n_log):
        tally.same(rows[r], alone(worlds, lanes[r], pool_seed), f"{what}: P{bucket}, lane {r} of {len(lanes)}")


BOUNDARY_POSITIONS = [15, 16, 63, 64, 127]


@pytest.mark.parametrize("pos", BOUNDARY_POSITIONS)
def test_a_token_at_a_block_or_context_boundary_is_invariant(worlds, pos, capsys):
    """The same token at 15 and 16 (a block boundary), 63 and 64 (the C = 64 to C = 128 switch) and 127 (the last slot of C = 128),
    in a fixed pool: alone, in lane 0 and in a later lane of larger T buckets, with companions, and, where both fit (positions
    below 64), at C = 64 and at C = 128: all equal to bit-exact reference of the token alone."""
    rng = random.Random(1000 + pos)
    ids = list(range(1, NBLK))
    rng.shuffle(ids)
    table = [ids.pop() for _ in range(pos // 16 + 1)]
    token = (rng.randrange(VOCAB), pos, table)
    companions = [(rng.randrange(VOCAB), rng.randrange(20), [ids.pop()]) for _ in range(7)]
    tally = Tally()
    c_min = 64 if pos < 64 else 128
    configs = [((1, 1, c_min), 0, 0), ((2, 2, c_min), 1, 1), ((4, 4, c_min), 3, 2), ((8, 4, 128), 7, 3), ((8, 1, 128), 7, 0)]
    if pos < 64:
        configs += [((1, 1, 128), 0, 0), ((4, 2, 64), 2, 1)]
    for bucket, n_others, j in configs:
        T, Ls, _ = bucket
        others = companions[:min(n_others, T - 1)]
        lanes = others[:j] + [token] + others[j:]
        n_log = min(Ls, len(lanes), max(j + 1, 1) if Ls > 1 else 1)
        rows = worlds.chip(bucket, lanes, n_log, SEED_POOL)
        tally.same(rows[j], alone(worlds, token, SEED_POOL), f"position {pos}: P{bucket}, lane {j}")
    if pos < 64:      # both context buckets of the same token, alone: the C switch itself
        tally.same(alone(worlds, token, SEED_POOL, 64), alone(worlds, token, SEED_POOL, 128), f"position {pos}: bit-exact reference at C = 64 and C = 128")
    with capsys.disabled():
        print(f"\nposition {pos}: {tally.steps} comparisons, {tally.signed_zero} differing in the sign of a zero only")


def test_positions_63_and_64_of_one_request_in_one_step_do_not_depend_on_the_lane_order(worlds):
    """A chunk of one request across the C switch (T = 2, C = 128): position 64 reads the row that position 63 writes in the same
    step, so the reference is bit-exact reference of the same step (not the token alone); the rows must not depend on the lane order."""
    rng = random.Random(7)
    ids = list(range(1, NBLK))
    rng.shuffle(ids)
    table = [ids.pop() for _ in range(5)]
    tok = rng.randrange(VOCAB)
    a, b = (tok, 63, table[:4]), (tok, 64, table)
    tally = Tally()
    ab = worlds.chip((2, 2, 128), [a, b], 2, SEED_POOL)
    ba = worlds.chip((2, 2, 128), [b, a], 2, SEED_POOL)
    tally.same(ab[0], ba[1], "position 63 in lane 0 against lane 1")
    tally.same(ab[1], ba[0], "position 64 in lane 1 against lane 0")
    for rows, lanes in ((ab, [a, b]), (ba, [b, a])):
        want = worlds.exact_ref((2, 2, 128), lanes, 2, SEED_POOL)
        for r in range(2):
            tally.same(rows[r], want[r], f"bit-exact reference of the same step, lane {r}")


PREFIX = [5, 9]                                       # two full blocks (positions 0..31) that the lanes share


def shared_prefix_lanes():
    """Lanes that share the two full blocks of PREFIX and diverge in the very next position (32: the first row of a block of their
    own), a lane at the last row of its own block, and one that shares only the first block."""
    return {"A": (101, 32, PREFIX + [20]), "B": (202, 32, PREFIX + [21]), "C": (303, 47, PREFIX + [22]),
            "D": (404, 16, PREFIX[:1] + [23]), "E": (505, 33, PREFIX + [20]), "F": (606, 20, PREFIX[:1] + [24])}


@pytest.mark.parametrize("bucket, order", [((2, 2, 64), "AB"), ((2, 2, 64), "BA"), ((4, 4, 64), "ABCD"), ((4, 4, 64), "DCBA"),
                                           ((4, 2, 128), "BDAC"), ((8, 4, 128), "CABDF"), ((8, 4, 64), "FDCBA")],
                         ids=lambda x: x if isinstance(x, str) else "P%d_%d_%d" % x)
def test_lanes_sharing_a_prefix_and_diverging_at_once_are_invariant(worlds, bucket, order):
    """The lanes read the same two blocks, then different blocks of their own; each lane's logits equal its own bit-exact reference row alone
    wherever the lane is (lane position, T bucket and C bucket vary) and whoever its companions are."""
    lanes_by_name = shared_prefix_lanes()
    lanes = [lanes_by_name[n] for n in order]
    check_rows(worlds, Tally(), bucket, lanes, min(bucket[1], len(lanes)), SEED_POOL, f"order {order}")


# -- the checks do check ----------------------------------------------------------------------------------------------------------
def test_a_token_whose_context_differs_has_different_logits(worlds):
    """The comparison is not vacuous: another block of context, or another position, changes the logits."""
    a = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    other_table = worlds.exact_ref((1, 1, 64), [(TOKEN[0], TOKEN[1], [5, 9, 3])], 1, SEED_POOL)[0]
    other_pos = worlds.exact_ref((1, 1, 64), [(TOKEN[0], TOKEN[1] - 1, TOKEN[2])], 1, SEED_POOL)[0]
    assert a.tobytes() != other_table.tobytes() and a.tobytes() != other_pos.tobytes()


def test_a_companion_writing_into_the_context_of_the_token_is_not_invariance(worlds):
    """The precondition of the claim: a lane that writes a row the token reads changes the token's logits (the property is about
    the token's own inputs; a shared written row is one of them)."""
    ref = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    writer = (99, 36, [5, 9, 2])                  # writes row 4 of block 2, read by the token at 37
    got = worlds.exact_ref((2, 1, 64), [TOKEN, writer], 1, SEED_POOL)[0]
    assert got.tobytes() != ref.tobytes()


def test_the_site_geometry_matches_the_cached_programs(worlds):
    art = worlds.art((1, 1, 64))
    g = art.record["geometry"]
    assert g["kaddr"] == [worlds.geo.kaddr(l, 0) for l in range(LAYERS)] and g["vaddr"] == [worlds.geo.vaddr(l, 0) for l in range(LAYERS)]


def test_the_chip_compare_catches_a_token_that_is_not_the_same(worlds):
    """RED for the chip schedule: the same step with the token under test replaced by its neighbour is a counterexample."""
    ref = worlds.exact_ref((1, 1, 64), [TOKEN], 1, SEED_POOL)[0]
    got = run_alone(worlds, (2, 1, 64), [(TOKEN[0] + 1, TOKEN[1], TOKEN[2]), COMPANIONS[0]], 0, 1)
    with pytest.raises(AssertionError, match="logits differ"):
        Tally().same(got, ref, "mutated")
    Tally().same(zero_sign(ref), np.where(ref == 0, -ref, ref), "a zero's sign alone is accepted")
