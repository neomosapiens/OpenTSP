"""Test the bit-exact oracle for paged and batched TinyStories inference.

The oracle is `paged_reference`; the reference for its mathematics is the whole-step bit-exact reference (`reference_model.decode_step`).
"""
import copy
import hashlib

import numpy as np
import pytest


from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import paged_reference as gp
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import ModelConfig

# the TinyStories shape with a short vocabulary (the vocabulary only sizes the head), for the many-step tests
CFG = ModelConfig(layers=2, vocab=2000, max_pos=128)
FULL_L2 = ModelConfig(layers=2)                    # `tinystories-l2` of  (full vocabulary)


@pytest.fixture(scope="module")
def w():
    return W.synthetic(CFG, seed=1)


def tokens_for(n, seed, vocab=CFG.vocab):
    return [int(t) for t in np.random.default_rng(seed).integers(0, vocab, n)]


def rel(a, b):
    """Largest absolute difference relative to the largest magnitude of b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def prefill_paged(w, pool, table, toks, **kw):
    out = []
    for pos, t in enumerate(toks):
        out.append(gp.decode_step_paged(w, pool, t, pos, table, logits=False, **kw))
    return out


def alloc_table(pool, n_tokens):
    return [pool.allocate() for _ in range(-(-n_tokens // gp.BLOCK))]


def ts_reference(w, paged_steps, toks):
    """The whole-step bit-exact reference on the TS layout (current token in the last slot), fed at every position with the K and V rows the
    paged run produced, so that each step is compared on identical inputs."""
    kv = exact_ref.HostKV(w.cfg)
    out = []
    for pos, (t, p) in enumerate(zip(toks, paged_steps)):
        out.append(exact_ref.decode_step(w, kv, t, pos, logits=False))
        kv.k[:, pos], kv.v[:, pos] = p.k_cur, p.v_cur
    return out


def test_paged_equals_ts_layout_maths_and_any_block_permutation(w):
    toks = tokens_for(20, 1)                       # two blocks: the second opens at position 16
    pool = gp.BlockPool(w.cfg, 16)
    table = alloc_table(pool, len(toks))
    paged = prefill_paged(w, pool, table, toks)
    ts = ts_reference(w, paged, toks)
    for pos, (a, b) in enumerate(zip(paged, ts)):
        assert rel(a.hidden, b.hidden) <= 1e-6, pos
        assert rel(a.k_cur, b.k_cur) <= 1e-6 and rel(a.v_cur, b.v_cur) <= 1e-6, pos
    # bit-equal everywhere but position 16 (measured 1.6e-7: the first position of the second block sums its
    # exponentials in another superlane order); the 1e-6 bound above is slack, not the measurement
    assert [pos for pos, (a, b) in enumerate(zip(paged, ts)) if a.hidden.tobytes() != b.hidden.tobytes()] == [16]

    # physical blocks permuted: the same logical sequence in other blocks gives identical bits
    pool2 = gp.BlockPool(w.cfg, 16)
    junk = [pool2.allocate() for _ in range(5)]
    table2 = list(reversed(alloc_table(pool2, len(toks))))
    for b in junk:
        pool2.free(b)
    paged2 = prefill_paged(w, pool2, table2, toks)
    for pos, (a, b) in enumerate(zip(paged, paged2)):
        assert a.hidden.tobytes() == b.hidden.tobytes(), pos
        assert a.k_cur.tobytes() == b.k_cur.tobytes() and a.v_cur.tobytes() == b.v_cur.tobytes(), pos
    assert table != table2


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
def test_embed_chip_equals_host_embedding_bit_for_bit(w):
    """OR of the parts, the permuter map, the exact widening and the one f32 add are `reference_model.embed`: every vocabulary id
    of a table seeded with every f16 bit pattern class (zeros of both signs, subnormals, infinities, NaNs), and the
    boundary ids of the three table parts."""
    cfg = ModelConfig(layers=1, vocab=50257, max_pos=128)
    rng = np.random.default_rng(5)
    wt = W.synthetic(ModelConfig(layers=1, vocab=50257, max_pos=128), seed=3)
    special = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01],
                       dtype=np.uint16)
    bits = rng.integers(0, 1 << 16, wt.wte.shape, dtype=np.uint16)
    bits[:, :12] = special
    wt.wte = bits.view(np.float16)
    ids = list(rng.integers(0, cfg.vocab, 300)) + [0, 4, 5, 20479, 20480, 20481, 40959, 40960, 50255, 50256]
    for t in ids:
        pos = int(rng.integers(0, 128))
        got, want = gp.embed_chip(wt, int(t), pos), exact_ref.embed(wt, int(t), pos)
        assert got.tobytes() == want.tobytes(), (t, pos)


def test_chip_head_within_f16_rounding_of_the_f64_host_head():
    """The 320-lane `dot_f16` head of the f16 `h`  against the f64 host head `reference_model.head_logits` of the
    f32 `h`: the only difference is the f16 rounding of h, 2^-11 per element, so at most 2^-10 * sum |wte[v] * h|."""
    cfg = ModelConfig(layers=2)
    wf = W.synthetic(cfg, seed=1)
    for seed in (1, 2):
        pool = gp.BlockPool(cfg, 16)
        table = alloc_table(pool, 3)
        toks = tokens_for(3, seed, cfg.vocab)
        r = None
        for pos, t in enumerate(toks):
            r = gp.decode_step_paged(wf, pool, t, pos, table, logits=(pos == 2))
        host = exact_ref.head_logits(wf, r.hidden).astype(np.float64)
        chip = r.logits.astype(np.float64)
        bound = 2.0 ** -10 * (np.abs(wf.wte.astype(np.float64)) @ np.abs(r.hidden.astype(np.float64)))
        assert chip.shape == (cfg.vocab,) and r.logits.dtype == np.float32
        assert np.all(np.abs(chip - host) <= bound)
        assert np.any(chip != host)                    # the head really rounds h to f16
        h16 = r.hidden.astype(np.float16).astype(np.float64)
        assert np.all(np.abs(chip - wf.wte.astype(np.float64) @ h16) <= np.abs(chip) * 2.0 ** -23 + 1e-12)


def test_chip_head_models_the_structural_dot_and_the_sign_of_an_exact_zero_sum():
    """The head is one dot over the 320 lanes of a packed row (5 vocabulary rows of 64 lanes), the entry's own segment against
    h and the four others against +0: an exact zero sum is -0 only when all 320 products are -0, so the neighbouring weights'
    signs decide ( item 3). Tokens 0..4 are all -0.0, 5..9 the same with token 8 = +0.0, 10..14 negative with token 14
    -0.0; the table is 17 tokens (the last packed row is padded with +0)."""
    wte = np.zeros((17, 64), np.float16)
    wte[0:5] = np.float16(-0.0)
    wte[5:10] = np.float16(-0.0)
    wte[8] = np.float16(0.0)
    wte[10:14] = np.float16(-0.5)
    wte[14] = np.float16(-0.0)
    wte[15:17] = np.float16(-0.0)                     # row 3 holds 2 live entries and 3 padding entries (+0)
    ws = type("W", (), {"wte": wte})()
    zero = np.zeros(64, np.float32)
    got = gp.head_logits_chip(ws, zero)
    assert got.shape == (17,) and got.dtype == np.float32
    sign = np.signbit(got)
    assert sign[0:5].all() and not sign[5:10].any() and sign[10:15].all()   # -0 only where every product is -0
    assert not sign[15:17].any()                                              # the padding's +0 products
    assert (got == 0).all()
    # a nonzero h: an ordinary sum, rounded once to f32, whatever the signs
    h = np.linspace(-1, 1, 64).astype(np.float32)
    wte2 = (np.random.default_rng(3).standard_normal((17, 64)) * 0.5).astype(np.float16)
    ws2 = type("W", (), {"wte": wte2})()
    want = wte2.astype(np.float64) @ h.astype(np.float16).astype(np.float64)
    assert np.all(np.abs(gp.head_logits_chip(ws2, h).astype(np.float64) - want) <= np.abs(want) * 2.0 ** -23 + 1e-12)
    # an infinity in a weight row turns the other four entries of its packed row into NaN (infinity times +0)
    wte3 = wte2.copy()
    wte3[1, 5] = np.float16(np.inf)
    got3 = gp.head_logits_chip(type("W", (), {"wte": wte3})(), h)
    assert np.isnan(got3[0:5]).sum() == 4 and np.isinf(got3[1]) and np.isfinite(got3[5:15]).all()


def single_token_results(w, seqs, C):
    """Every token of every sequence run alone, in order, on a private pool: {(seq, pos): StepResult}."""
    res = {}
    for i, toks in enumerate(seqs):
        pool = gp.BlockPool(w.cfg, 16)
        table = alloc_table(pool, len(toks))
        for pos, t in enumerate(toks):
            res[i, pos] = gp.decode_step_paged(w, pool, t, pos, table, C=C, logits=False)
    return res


def run_schedule(w, seqs, schedule, C, bucket_of, order=None):
    """Run `schedule`, a list of steps, each a list of (seq, pos): in one shared pool, one `step_batch` per step on a bucket
    `bucket_of(n_lanes)`. Returns {(seq, pos): lane result (hidden, k rows, v rows)}."""
    pool = gp.BlockPool(w.cfg, 32)
    tables = [alloc_table(pool, len(t)) for t in seqs]
    out = {}
    for step in schedule:
        step = list(step) if order is None else [step[j] for j in order(len(step))]
        b = bucket_of(len(step))
        r = gp.step_batch(w, pool, [seqs[i][p] for i, p in step], [p for _, p in step], [tables[i] for i, _ in step], b)
        for lane, (i, p) in enumerate(step):
            out[i, p] = (r.hidden[lane], r.k_rows[lane], r.v_rows[lane])
    return out


def test_step_batch_lanes_equal_single_token_results():
    """Lanes never interact: any batch composition (decodes of several requests, a prefill chunk of consecutive positions,
    a mix), lane order and bucket gives the bits of the token run alone."""
    w = W.synthetic(CFG, seed=1)
    seqs = [tokens_for(20, 11), tokens_for(9, 12), tokens_for(5, 13)]
    C = 64
    alone = single_token_results(w, seqs, C)
    # chunked prefill of seq 0 (4 consecutive positions per step) beside seq 1 decoding token by token and seq 2 prefilling
    schedules = {
        "chunks": [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 4), (0, 5), (1, 0), (2, 0)], [(0, 6), (1, 1), (2, 1)]],
        "decodes": [[(0, 0)], [(0, 1), (1, 0)], [(0, 2), (1, 1), (2, 0)], [(0, 3), (1, 2), (2, 1)]],
        "one-chunk": [[(1, p) for p in range(8)]],
    }
    orders = {"asis": None, "reversed": lambda n: list(range(n))[::-1], "rotated": lambda n: [(j + 1) % n for j in range(n)]}
    buckets = {"grid": lambda n: (next(t for t in (1, 2, 4, 8) if t >= n), 1, C),
               "t8": lambda n: (8, 2, C)}
    for sname, sched in schedules.items():
        for oname, order in orders.items():
            for bname, bucket_of in buckets.items():
                got = run_schedule(w, seqs, sched, C, bucket_of, order)
                for (i, p), (h, k, v) in got.items():
                    want = alone[i, p]
                    tag = (sname, oname, bname, i, p)
                    assert h.tobytes() == want.hidden.tobytes(), tag
                    assert k.tobytes() == want.k_cur.tobytes() and v.tobytes() == want.v_cur.tobytes(), tag


def test_step_batch_buckets_pad_neutrally_and_validate():
    w = W.synthetic(CFG, seed=1)
    toks = tokens_for(30, 21)
    C_small, C_big = 64, 128
    pool = gp.BlockPool(w.cfg, 16)
    table = alloc_table(pool, len(toks))
    for pos, t in enumerate(toks[:-1]):
        gp.decode_step_paged(w, pool, t, pos, table, logits=False)
    pos = len(toks) - 1
    snap_k, snap_v = pool.k.copy(), pool.v.copy()
    results = []
    for bucket in ((1, 1, C_small), (2, 1, C_small), (8, 4, C_small), (1, 1, C_big), (4, 2, C_big)):
        pool.k[:], pool.v[:] = snap_k, snap_v
        r = gp.step_batch(w, pool, [toks[pos]], [pos], [table], bucket)
        results.append(r)
        # the padding lanes wrote nothing but the sink; the null block is still zero
        assert not pool.k[gp.NULL].any() and not pool.v[gp.NULL].any()
        assert r.logits.shape == (1, w.cfg.vocab)
    for r in results[1:]:                              # bucket invariance, bit for bit 
        assert r.hidden[0].tobytes() == results[0].hidden[0].tobytes()
        assert r.logits.tobytes() == results[0].logits.tobytes()
        assert r.k_rows[0].tobytes() == results[0].k_rows[0].tobytes()
    with pytest.raises(ValueError):
        gp.step_batch(w, pool, [1, 2], [0, 1], [table, table], (1, 1, 64))        # more lanes than T
    with pytest.raises(ValueError):
        gp.step_batch(w, pool, [1], [64], [table], (1, 1, 64))                    # position beyond C
    with pytest.raises(ValueError):
        gp.step_batch(w, pool, [1], [0], [table], (2, 4, 64))                     # L > T


def test_choose_bucket_follows_the_grid_rules():
    assert gp.choose_bucket(1, 1, 0) == (1, 1, 64)
    assert gp.choose_bucket(3, 2, 63) == (4, 2, 64)
    assert gp.choose_bucket(5, 3, 64) == (8, 4, 128)
    assert gp.choose_bucket(8, 1, 127) == (8, 1, 128)
    with pytest.raises(ValueError):
        gp.choose_bucket(9, 1, 0)
    with pytest.raises(ValueError):
        gp.choose_bucket(1, 1, 128)


def layer_composition(w, pool, table, token, pos):
    """The independent composition: `reference_model.layer_step` layer by layer on the TS layout (current token in the last
    slot), its K and V caches taken from the pool's cached positions 0..pos-1, the embedding of the host."""
    cfg = w.cfg
    x = exact_ref.embed(w, token, pos)
    ks, vs = [], []
    for l, lw in enumerate(w.layers):
        k = np.zeros((cfg.slots, cfg.hidden), np.float16)
        v = np.zeros((cfg.slots, cfg.hidden), np.float32)
        for p in range(pos):
            b, o = table[p // gp.BLOCK], p % gp.BLOCK
            k[p], v[p] = pool.k[b, l, o], pool.v[b, l, o]
        mask = np.full(cfg.slots, -np.inf, np.float32)
        mask[:pos] = 0
        mask[cfg.slots - 1] = 0
        x, kc, vc = exact_ref.layer_step(cfg, lw, x, k, v, mask)
        ks.append(kc)
        vs.append(vc)
    from opentsp_tools.programs import ref
    return ref.layernorm(x, w.lnf_g, w.lnf_b, cfg.ln_eps), np.stack(ks), np.stack(vs)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_any_layer_count_equals_the_layer_step_composition(layers):
    """`tinystories-l2`  and other depths: the paged step is the composition of the layer step, the
    K and V rows of layer l come from layer l, and at the last slot (position C - 1) the layouts coincide, so the bits do."""
    cfg = ModelConfig(layers=layers, vocab=1500 if layers != 2 else 50257, max_pos=128)
    wl = W.synthetic(cfg, seed=1)
    assert len(wl.layers) == layers
    pool = gp.BlockPool(cfg, 16)
    table = alloc_table(pool, 64)
    toks = tokens_for(64, 31, cfg.vocab)
    for pos, t in enumerate(toks[:63]):
        gp.decode_step_paged(wl, pool, t, pos, table, logits=False)
    snap = pool.k.copy(), pool.v.copy()
    want_h, want_k, want_v = layer_composition(wl, pool, table, toks[63], 63)
    got = gp.decode_step_paged(wl, pool, toks[63], 63, table)
    assert got.hidden.tobytes() == want_h.tobytes()
    assert got.k_cur.tobytes() == want_k.tobytes() and got.v_cur.tobytes() == want_v.tobytes()
    assert got.k_cur.shape == (layers, cfg.hidden) and got.logits.shape == (cfg.vocab,)
    assert np.isfinite(got.logits).all()
    # an early position: same maths, other slot order
    pool.k[:], pool.v[:] = snap
    want_h, want_k, want_v = layer_composition(wl, pool, table, toks[30], 30)
    got = gp.decode_step_paged(wl, pool, toks[30], 30, table, logits=False)
    assert rel(got.hidden, want_h) <= 1e-6 and rel(got.k_cur, want_k) <= 1e-6


def test_prefix_sharing_reads_the_same_block_and_gives_the_solo_logits():
    w = W.synthetic(CFG, seed=1)
    prefix = tokens_for(16, 41)                              # exactly one full block
    suffix_a, suffix_b = tokens_for(4, 42), tokens_for(5, 43)

    def solo(suffix):
        pool = gp.BlockPool(w.cfg, 16)
        table = alloc_table(pool, 16 + len(suffix))
        res = [gp.decode_step_paged(w, pool, t, p, table, logits=(p == 15 + len(suffix)))
               for p, t in enumerate(prefix + suffix)]
        return res[-1]

    pool = gp.BlockPool(w.cfg, 16)
    shared = pool.allocate()
    own_a = pool.allocate()
    table_a = [shared, own_a]
    res_a = [gp.decode_step_paged(w, pool, t, p, table_a, logits=(p == 19)) for p, t in enumerate(prefix + suffix_a)]
    # B arrives later: the cached block is shared, nothing is copied or recomputed, only its own suffix runs
    pool.share(shared)
    own_b = pool.allocate()
    table_b = [shared, own_b]
    assert table_a[0] == table_b[0] and pool.refs[shared] == 2 and own_a != own_b
    before = pool.k[shared].copy(), pool.v[shared].copy()
    res_b = [gp.decode_step_paged(w, pool, t, 16 + i, table_b, logits=(i == len(suffix_b) - 1))
             for i, t in enumerate(suffix_b)]
    assert pool.k[shared].tobytes() == before[0].tobytes() and pool.v[shared].tobytes() == before[1].tobytes()
    assert res_a[-1].logits.tobytes() == solo(suffix_a).logits.tobytes()
    assert res_b[-1].logits.tobytes() == solo(suffix_b).logits.tobytes()
    assert res_b[-1].logits.tobytes() != res_a[-1].logits.tobytes()
    pool.free(shared)
    assert shared in pool.refs and pool.refs[shared] == 1
    pool.free(shared)
    assert shared not in pool.refs and shared in pool.free_queue


def test_stale_blocks_copies_and_zeroing_do_not_change_results():
    """A freed block keeps its words (the chip does not clear it): a sequence placed on stale blocks gives the bits of a
    fresh pool, because every slot above the position is masked. A copy of a block into another gives the same reads;
    zeroing a block is what a fresh pool holds."""
    w = W.synthetic(CFG, seed=1)
    toks = tokens_for(20, 51)
    fresh = gp.BlockPool(w.cfg, 16)
    t0 = alloc_table(fresh, len(toks))
    want = prefill_paged(w, fresh, t0, toks)

    pool = gp.BlockPool(w.cfg, 16)
    old = [pool.allocate(), pool.allocate()]
    prefill_paged(w, pool, old, tokens_for(20, 52))        # junk in the first blocks
    for b in old:
        pool.free(b)
    for _ in range(len(pool.free_queue) - 2):               # take the others first so the stale blocks come next
        pool.free(pool.allocate())
    table = alloc_table(pool, len(toks))
    assert set(table) == set(old)
    got = prefill_paged(w, pool, table, toks)
    assert all(a.hidden.tobytes() == b.hidden.tobytes() for a, b in zip(got, want))

    a, b = pool.allocate(), pool.allocate()
    pool.copy(table[0], a)
    assert pool.k[a].tobytes() == pool.k[table[0]].tobytes() and pool.v[a].tobytes() == pool.v[table[0]].tobytes()
    pool.zero(b)
    assert not pool.k[b].any() and not pool.v[b].any()
    with pytest.raises(ValueError):
        pool.write_row(gp.NULL, 0, 0, np.zeros(64, np.float16), np.zeros(64, np.float32))
    with pytest.raises(ValueError):
        pool.zero(gp.NULL)


def test_context_of_128_slots_is_one_softmax():
    """`opentsp_tools.programs.ref.softmax` reduces over segments of four superlanes, so a 128-slot vector would be normalised as two
    independent halves; the paged step's full-vector softmax is one distribution, and equals `ref.softmax` at 64 slots."""
    from opentsp_tools.programs import ref
    rng = np.random.default_rng(7)
    s64 = (rng.standard_normal((16, 64)) * 2).astype(np.float32)
    assert gp.softmax_full(s64).tobytes() == ref.softmax(s64).tobytes()
    s128 = (rng.standard_normal((16, 128)) * 2).astype(np.float32)
    full = gp.softmax_full(s128)
    assert np.allclose(full.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(ref.softmax(s128).sum(axis=1), 2.0, atol=1e-5)
    m = np.full(128, -np.inf, np.float32)
    m[:70] = 0                                             # a context beyond the first 64 slots
    sc = s128 + m
    assert np.allclose(gp.softmax_full(sc).sum(axis=1), 1.0, atol=1e-5)
    assert (gp.softmax_full(sc)[:, 70:] == 0).all()


# Empirical bounds of the f16-flip differences between the paged and the TS-layout steps, per model: the share
# of steps that are not bit-equal and the worst relative difference of the hidden state, measured on identical inputs.
# The 2^-10 figure is a measured margin (worst 2.8e-4 on 2 layers over 512 steps, 5.2e-4 on 8 layers over 512), not a theorem.
FLIP_BOUND = {2: dict(seeds=range(1, 9), max_nonequal=0.03),         # measured 7 of 512
              8: dict(seeds=(1, 2, 4, 8), max_nonequal=0.08)}        # measured 14 of 256 (22 of 512 over 8 seeds)


@pytest.mark.parametrize("layers", [2, 8])
def test_ts_layout_flip_bound_over_whole_block_tables(layers):
    """Over 64 positions (four blocks) and several sequences the paged and the TS-layout steps on identical inputs are
    equal except where the other sum order of the exponentials moves a probability across an f16 rounding boundary
    (2^-11 relative on that probability). The model of the design's target is 8 layers with the full vocabulary."""
    cfg = ModelConfig(layers=layers, vocab=2000 if layers == 2 else 50257, max_pos=128)
    wl = W.synthetic(cfg, seed=1)
    spec = FLIP_BOUND[layers]
    worst, nonequal, total = 0.0, 0, 0
    for seed in spec["seeds"]:
        toks = tokens_for(64, seed, cfg.vocab)
        pool = gp.BlockPool(cfg, 16)
        paged = prefill_paged(wl, pool, alloc_table(pool, 64), toks)
        for a, b in zip(paged, ts_reference(wl, paged, toks)):
            assert np.isfinite(a.hidden).all()
            worst = max(worst, rel(a.hidden, b.hidden))
            nonequal += a.hidden.tobytes() != b.hidden.tobytes()
            total += 1
    assert total >= 40 and worst <= 2.0 ** -10
    assert nonequal <= spec["max_nonequal"] * total, (nonequal, total)


def test_step_batch_lanes_equal_single_lane_results_on_8_layers():
    cfg = ModelConfig(max_pos=128)
    wl = W.synthetic(cfg, seed=1)
    seqs = [tokens_for(19, 111, cfg.vocab), tokens_for(6, 112, cfg.vocab), tokens_for(3, 113, cfg.vocab)]
    alone = single_token_results(wl, seqs, 64)
    sched = [[(0, 0), (0, 1), (1, 0), (2, 0)], [(0, 2), (0, 3), (1, 1)] + [(2, 1)], [(0, p) for p in range(4, 12)],
             [(0, 12), (1, 2), (2, 2)], [(0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (1, 3), (1, 4)]]
    for order in (None, lambda n: list(range(n))[::-1]):
        got = run_schedule(wl, seqs, sched, 64, lambda n: (next(t for t in (1, 2, 4, 8) if t >= n), 2 if n > 1 else 1, 64), order)
        assert len(got) == sum(len(st) for st in sched)
        for (i, p), (h, k, v) in got.items():
            assert h.tobytes() == alone[i, p].hidden.tobytes(), (i, p)
            assert k.tobytes() == alone[i, p].k_cur.tobytes() and v.tobytes() == alone[i, p].v_cur.tobytes(), (i, p)
    # head rows of an 8-layer batch: lane 1 is the head of its single result
    pool = gp.BlockPool(cfg, 32)
    tables = [alloc_table(pool, len(t)) for t in seqs]
    r = gp.step_batch(wl, pool, [seqs[0][0], seqs[1][0]], [0, 0], [tables[0], tables[1]], (2, 2, 64))
    assert r.logits[1].tobytes() == gp.head_logits_chip(wl, alone[1, 0].hidden).tobytes()


def test_batch_step_digest_is_pinned():
    """A 3-lane chunk on the 2-layer test model with the full vocabulary (bucket P(4, 2, 64)): hidden states, K and V rows
    and the two logit rows, hashed, so that the serving emitters compare with a fixed value."""
    cfg = FULL_L2
    wl = W.synthetic(cfg, seed=1)
    pool = gp.BlockPool(cfg, 16)
    table = alloc_table(pool, 20)
    toks = tokens_for(20, 77, cfg.vocab)
    for pos in range(17):
        gp.decode_step_paged(wl, pool, toks[pos], pos, table, logits=False)
    r = gp.step_batch(wl, pool, toks[17:20], [17, 18, 19], [table] * 3, (4, 2, 64))
    h = hashlib.sha256()
    for a in (r.hidden, r.k_rows, r.v_rows, r.logits):
        h.update(a.tobytes())
    assert r.logits.shape == (2, cfg.vocab) and r.hidden.shape == (4, cfg.hidden)
    assert h.hexdigest() == "dea413fa761b8a2229cdf8dd6a5b294d357d8a66aa96a02e6f9671cbd92fc2e3"


def step_digest(r):
    h = hashlib.sha256()
    for a in (r.hidden, r.k_cur, r.v_cur):
        h.update(a.tobytes())
    return h.hexdigest()


# regression pins of decode_step_paged at C = 128 (2-layer short-vocabulary model, synthetic seed 1, tokens_for(128, 91)):
# positions above 63 reach the upper half of the 128-slot softmax. Produced by paged_reference; the independent checks are the
# hand-expanded tree and the lane invariance below, and the equality with C = 64 at positions below 64.
PINNED_C128 = {64: "10a4447ff9e9dd88c429808f2ae8dbc9b0c6ece95ce40ca7d887be664c9707a0",
               100: "e41dcbac2bcb9f25afdd6856fd4b23ea2a056e67c7aa127bba8682b1306c9f6a",
               127: "67fbc2a94547627797bd6571f3784f46b33915556e53cff2b0a93ab8cec5790d"}


def test_decode_step_paged_over_128_positions_at_c128():
    w = W.synthetic(CFG, seed=1)
    toks = tokens_for(128, 91)
    pool = gp.BlockPool(w.cfg, 16)
    table = alloc_table(pool, 128)
    pool64 = gp.BlockPool(w.cfg, 16)
    table64 = alloc_table(pool64, 64)
    for pos, t in enumerate(toks):
        r = gp.decode_step_paged(w, pool, t, pos, table, C=128, logits=False)
        assert np.isfinite(r.hidden).all(), pos
        if pos < 64:                                   # bucket invariance over the whole history
            r64 = gp.decode_step_paged(w, pool64, t, pos, table64, C=64, logits=False)
            assert step_digest(r) == step_digest(r64), pos
        if pos in PINNED_C128:
            assert step_digest(r) == PINNED_C128[pos], pos


def hand_total(partials, op):
    """The 8-superlane total as the design's rotations 4, 2, 1 give it, written out: lane 0 of every superlane holds
    ((s0 op s4) op (s2 op s6)) op ((s1 op s5) op (s3 op s7)) (commutative operations, so every superlane agrees)."""
    s = partials
    return op(op(op(s[0], s[4]), op(s[2], s[6])), op(op(s[1], s[5]), op(s[3], s[7])))


def hand_fold(row, op):
    """Reduce of one superlane: the ascending fold over its 16 lanes, in f32 scalar arithmetic."""
    acc = np.float32(row[0])
    for x in row[1:]:
        acc = op(acc, np.float32(x))
    return acc


def test_softmax_full_equals_the_hand_expanded_8_superlane_tree():
    from opentsp_tools.programs import ref
    from opentsp_tools.programs import numerics as nx
    rng = np.random.default_rng(17)
    fadd = lambda a, b: np.float32(a) + np.float32(b)
    fmax = lambda a, b: np.maximum(np.float32(a), np.float32(b))
    for trial in range(6):
        s = (rng.standard_normal((16, 128)) * 3).astype(np.float32)
        if trial >= 3:                                   # a causal mask with a non-empty upper half
            s[:, 70 + 10 * (trial - 3):] = -np.inf
        want = np.zeros_like(s)
        for h in range(16):
            row = s[h]
            mx = hand_total([hand_fold(row[16 * g:16 * g + 16], fmax) for g in range(8)], fmax)
            e = nx.transcendental("exp", "f32", ref._sub(row, np.full(128, mx, np.float32)))
            tot = hand_total([hand_fold(e[16 * g:16 * g + 16], fadd) for g in range(8)], fadd)
            want[h] = ref._mul(e, nx.transcendental("rsqrt", "f32", ref._mul(np.full(128, tot, np.float32),
                                                                            np.full(128, tot, np.float32))))
        assert gp.softmax_full(s).tobytes() == want.tobytes(), trial
    # the upper half carries probability: the two-halves tree is a different function
    assert ref.softmax(s).tobytes() != gp.softmax_full(s).tobytes()


def test_lane_invariance_at_c128_with_lanes_above_position_63():
    w = W.synthetic(CFG, seed=1)
    seqs = [tokens_for(72, 61), tokens_for(12, 62)]
    C = 128
    alone = single_token_results(w, seqs, C)
    sched = [[(0, p) for p in range(8 * i, 8 * i + 8)] for i in range(8)]
    sched += [[(0, 64 + j) for j in range(4)] + [(1, 0), (1, 1)], [(0, 68), (0, 69), (1, 2), (1, 3)],
              [(1, 4 + j) for j in range(4)] + [(0, 70), (0, 71)]]
    for bucket_of in (lambda n: (8, 2, C), lambda n: (8, 4, C)):
        for order in (None, lambda n: list(range(n))[::-1]):
            got = run_schedule(w, seqs, sched, C, bucket_of, order)
            assert any(p > 63 for (_, p) in got)
            for (i, p), (h, k, v) in got.items():
                assert h.tobytes() == alone[i, p].hidden.tobytes(), (i, p)
                assert k.tobytes() == alone[i, p].k_cur.tobytes() and v.tobytes() == alone[i, p].v_cur.tobytes()


def test_attention_over_128_slots_is_one_distribution_per_head():
    """`paged_reference.attention` itself (not only `softmax_full`) at C = 128 with the mask of position 100: the probabilities of every
    head sum to 1 and vanish above the position."""
    rng = np.random.default_rng(23)
    cfg = gp.bucket_cfg(CFG, 128)
    q = rng.standard_normal(cfg.hidden).astype(np.float32)
    k16 = (rng.standard_normal((128, cfg.hidden)) * 0.5).astype(np.float16)
    v32 = rng.standard_normal((128, cfg.hidden)).astype(np.float32)
    parts = {}
    ctx = gp.attention(cfg, q, k16, v32, gp.position_mask(100, 128), parts)
    assert ctx.shape == (cfg.hidden,) and np.isfinite(ctx).all()
    assert np.allclose(parts["probs"].sum(axis=1), 1.0, atol=1e-5)
    assert (parts["probs"][:, 101:] == 0).all() and (parts["probs"][:, 64:101] > 0).any()


def pad_reference(w, C):
    """What a padding lane computes , from `reference_model.layer_step` alone: a zero embedding, a context of zero K and V
    rows in which only slot 0 is unmasked (position 0 of the null block), the current slot masked out."""
    from opentsp_tools.programs import ref
    cfg = bucket_cfg(w, C)
    x = np.zeros(cfg.hidden, np.float32)
    k = np.zeros((cfg.slots, cfg.hidden), np.float16)
    v = np.zeros((cfg.slots, cfg.hidden), np.float32)
    mask = np.full(cfg.slots, -np.inf, np.float32)
    mask[0] = 0
    ks, vs = [], []
    for lw in w.layers:
        x, kc, vc = exact_ref.layer_step(cfg, lw, x, k, v, mask)
        ks.append(kc)
        vs.append(vc)
    return ref.layernorm(x, w.lnf_g, w.lnf_b, cfg.ln_eps), np.stack(ks), np.stack(vs)


def bucket_cfg(w, C):
    return gp.bucket_cfg(w.cfg, C)


def test_padding_lanes_are_finite_null_context_lanes_that_write_only_the_sink():
    w = W.synthetic(CFG, seed=1)
    seqs = [tokens_for(20, 71), tokens_for(9, 72), tokens_for(5, 73)]
    want_h, want_k, want_v = pad_reference(w, 64)         # the TS layout's softmax has no 128-slot form: C = 128 must agree
    for C in (64, 128):
        pools = {}
        for T in (4, 8):
            pool = gp.BlockPool(w.cfg, 32)
            tables = [alloc_table(pool, len(t)) for t in seqs]
            for i, toks in enumerate(seqs):               # history: all but the last token of each sequence
                for pos, t in enumerate(toks[:-1]):
                    gp.decode_step_paged(w, pool, t, pos, tables[i], C=C, logits=False)
            sink_before = pool.k[pool.sink].copy()
            r = gp.step_batch(w, pool, [t[-1] for t in seqs], [len(t) - 1 for t in seqs], tables, (T, 2, C))
            pools[T] = pool
            for rows in (r.hidden, r.k_rows, r.v_rows):
                assert np.isfinite(rows).all()
            assert r.logits.shape == (2, w.cfg.vocab) and np.isfinite(r.logits).all()
            for lane in range(3, T):                       # every padding lane is the null-context lane
                assert r.hidden[lane].tobytes() == want_h.tobytes(), (C, T, lane)
                assert r.k_rows[lane].tobytes() == want_k.tobytes() and r.v_rows[lane].tobytes() == want_v.tobytes()
            assert not pool.k[gp.NULL].any() and not pool.v[gp.NULL].any()
            assert pool.k[pool.sink].tobytes() != sink_before.tobytes()          # the sink took the padding writes
        # padding changes nothing but the sink: every other block is the same with 1 or 5 padding lanes
        assert pools[4].k[:-1].tobytes() == pools[8].k[:-1].tobytes()
        assert pools[4].v[:-1].tobytes() == pools[8].v[:-1].tobytes()


@pytest.mark.parametrize("L", [2, 4])
def test_step_batch_logit_rows_equal_the_head_of_each_single_lane(L):
    w = W.synthetic(CFG, seed=1)
    seqs = [tokens_for(7 + i, 80 + i) for i in range(4)]
    pool = gp.BlockPool(w.cfg, 32)
    tables = [alloc_table(pool, len(t)) for t in seqs]
    for i, toks in enumerate(seqs):
        for pos, t in enumerate(toks[:-1]):
            gp.decode_step_paged(w, pool, t, pos, tables[i], logits=False)
    snap = pool.k.copy(), pool.v.copy()
    alone = []
    for i, toks in enumerate(seqs):
        alone.append(gp.decode_step_paged(w, pool, toks[-1], len(toks) - 1, tables[i], logits=False))
        pool.k[:], pool.v[:] = snap
    r = gp.step_batch(w, pool, [t[-1] for t in seqs], [len(t) - 1 for t in seqs], tables, (4, L, 64), n_logits=L)
    assert r.logits.shape == (L, w.cfg.vocab)
    for i in range(L):
        assert r.logits[i].tobytes() == gp.head_logits_chip(w, alone[i].hidden).tobytes(), i
    assert len({r.logits[i].tobytes() for i in range(L)}) == L


def test_null_block_is_the_only_source_of_context_beyond_the_table():
    """Poison every block outside the real tables with finite extremes (f16 max for K, f32 max for V): real and padding lanes
    are unchanged, because the slots beyond a table are read from the null block and masked."""
    w = W.synthetic(CFG, seed=1)
    seqs = [tokens_for(20, 91), tokens_for(5, 92), tokens_for(33, 93)]
    pool = gp.BlockPool(w.cfg, 32)
    tables = [alloc_table(pool, len(t)) for t in seqs]
    for i, toks in enumerate(seqs):
        for pos, t in enumerate(toks[:-1]):
            gp.decode_step_paged(w, pool, t, pos, tables[i], logits=False)
    args = ([t[-1] for t in seqs], [len(t) - 1 for t in seqs], tables, (8, 1, 64))
    clean = copy.deepcopy(pool)
    want = gp.step_batch(w, clean, *args)
    used = {b for tb in tables for b in tb}
    for b in range(1, pool.num_blocks + 1):
        if b not in used:
            pool.k[b], pool.v[b] = np.finfo(np.float16).max, np.finfo(np.float32).max
    got = gp.step_batch(w, pool, *args)
    for a, b in ((got.hidden, want.hidden), (got.k_rows, want.k_rows), (got.v_rows, want.v_rows),
                 (got.logits, want.logits)):
        assert a.tobytes() == b.tobytes()
    assert np.isfinite(got.hidden).all()


def test_continuation_through_a_copied_block_equals_the_original():
    """Copy-on-write : a copy of a block, in the table instead of the original, is read like the original."""
    w = W.synthetic(CFG, seed=1)
    toks = tokens_for(22, 95)
    pool = gp.BlockPool(w.cfg, 16)
    table = alloc_table(pool, 23)
    prefill_paged(w, pool, table, toks)
    spare = pool.allocate()
    pool.copy(table[1], spare)
    pool.zero(table[1])                                    # the original is gone: only the copy can carry the context
    a, b = copy.deepcopy(pool), copy.deepcopy(pool)
    cow = gp.decode_step_paged(w, a, 5, 22, [table[0], spare])
    b.copy(spare, table[1])
    orig = gp.decode_step_paged(w, b, 5, 22, table[:2])
    assert cow.hidden.tobytes() == orig.hidden.tobytes() and cow.logits.tobytes() == orig.logits.tobytes()
    assert cow.k_cur.tobytes() == orig.k_cur.tobytes()


def test_step_batch_rejects_two_lanes_writing_one_word():
    """: no two lanes write the same word. A collision would make the result depend on lane order."""
    w = W.synthetic(CFG, seed=1)
    pool = gp.BlockPool(w.cfg, 16)
    table = alloc_table(pool, 20)
    other = alloc_table(pool, 20)
    snap = pool.k.copy()
    with pytest.raises(ValueError, match="write"):
        gp.step_batch(w, pool, [1, 2], [3, 3], [table, table], (2, 1, 64))
    with pytest.raises(ValueError, match="write"):
        gp.step_batch(w, pool, [1, 2], [3, 3], [table, [table[0], other[1]]], (2, 1, 64))
    assert pool.k.tobytes() == snap.tobytes()               # refused before any write
    gp.step_batch(w, pool, [1, 2], [3, 3], [table, other], (2, 1, 64))                # distinct words: fine
    gp.step_batch(w, pool, [1, 2], [3, 4], [table, table], (2, 1, 64))


def test_generate_paged_is_greedy_decoding_on_the_chip_head():
    """`generate_paged` (the reference of  and , not `reference_model.generate`): prefill token by token through the paged
    step, logits from `head_logits_chip` only for the last prompt token and after, the first maximum on ties."""
    w = W.synthetic(CFG, seed=1)
    prompt = tokens_for(18, 101)                           # the prefill crosses the block boundary at 16
    gen = gp.generate_paged(w, prompt, 6)
    assert gen.tokens[:18] == prompt and len(gen.tokens) == 24 and len(gen.steps) == 18 + 6   # the last token is fed too, as in `reference_model.generate`
    pool = gp.BlockPool(w.cfg, 16)
    table = alloc_table(pool, 24)
    toks = list(prompt)
    for pos in range(24):
        last = pos >= len(toks) - 1
        r = gp.decode_step_paged(w, pool, toks[pos], pos, table, logits=last)
        assert r.hidden.tobytes() == gen.steps[pos].hidden.tobytes(), pos
        assert (r.logits is None) == (gen.steps[pos].logits is None) == (not last)
        if last:
            assert r.logits.tobytes() == gp.head_logits_chip(w, r.hidden).tobytes() == gen.steps[pos].logits.tobytes()
            if len(toks) - 18 < 6:
                toks.append(int(np.argmax(r.logits)))
    assert toks == gen.tokens
    # first maximum on ties: a duplicated top logit picks the lower id
    tie = gp.pick_token(np.array([0.0, 2.0, 2.0, 1.0], np.float32))
    assert tie == 1
    with pytest.raises(ValueError):
        gp.generate_paged(w, tokens_for(60, 102), 10)      # past the 64 slots of the default bucket


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
