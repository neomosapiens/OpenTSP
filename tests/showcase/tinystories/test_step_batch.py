"""Compare the batched, paged step P(1, 1, 64) (`emit_step_batch.py`) against bit-exact reference (`paged_reference.step_batch`), on the 2-layer test model
and on the full 8-layer model, for a context of 1 token, of 20 tokens (two blocks, the block table permuted) and of 64
tokens (the full context), one lane. The program is patched with the lane's sites (`site_values`), run on the ISS (and, once,
on `tsp_sim`), and what the chip leaves in SRAM is compared with bit-exact reference bit for bit: the logits, the final hidden state, and the
whole K and V pool regions (blocks 0..11 and the sinks of every layer: the new rows in the right words, every other word
unchanged; blocks 12..127 and the sinks included, the K sink excepted: the seven non-owning pairs write to that sink, so a
write of any of them that misses the sink changes another block and fails the compare). Zero diagnostics. One token and one
position per context: at context 1 the softmax is over one unmasked slot (exactly 1), so a wrong K block is invisible there;
the K reads are covered by contexts 20 and 64 (the embedding parts are 's).

The build of the full program takes about 20 s, a run of it on the ISS a few seconds."""
import os
import subprocess

import numpy as np
import pytest


from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools import pager
from opentsp_tools import patch
from opentsp_tools import profile as tprofile
tsp_as = pager.tsp_as

ISS = os.environ.get("TSP_ISS") or os.path.join(tprofile.ROOT, "target", "release", "tsp-iss")
SIM = os.environ.get("TSP_SIM") or os.path.join(tprofile.ROOT, "target", "spec", "tsp_sim")
NB = 12                       # pool blocks 0..NB - 1 hold random rows (block 0 stays zero) and are compared after the step
CONTEXTS = {1: (0, [5]), 20: (19, [7, 2]), 64: (63, [7, 2, 9, 4])}      # tokens in the context: (position, block table)
TOKEN = 31337


@pytest.fixture(scope="module")
def batches():
    cache = {}

    def get(layers):
        if layers not in cache:
            cache[layers] = E.emit_step_batch(1, 1, 64, layers=layers)
        return cache[layers]
    return get


def random_pool(cfg, seed):
    pool = G.BlockPool(cfg, G.POOL_BLOCKS)
    rng = np.random.default_rng(seed)
    pool.k[1:NB] = (rng.standard_normal(pool.k[1:NB].shape) * 0.5).astype(np.float16)
    pool.v[1:NB] = rng.standard_normal(pool.v[1:NB].shape).astype(np.float32)
    return pool


def pool_words(geo, pool, layers, blocks=None):
    """The words of the K and V pool regions of every layer for `blocks` (default: the filled blocks and the sink)."""
    blocks = list(range(NB)) + [geo.sink] if blocks is None else blocks
    out = {}
    for l in range(layers):
        out.update(kvpool.k_words(geo, pool, l, blocks))
        out.update(kvpool.v_words(geo, pool, l, blocks))
    return out


REF_SHAPE = L.Shape(320, L.PACK, L.HEAD_CHUNK, L.HEAD_INSTALLS, L.MASK_CS)


def shape_of(batch):
    return batch.shape or REF_SHAPE


def unpack_logits(words, vocab, shape=REF_SHAPE):
    """The (vocab) f32 logits from the dumped words of the logits region: row 0, install i, segment k at word `pack i + k`, lane r
    of its 4 byte planes = vocabulary entry pack (head_rows i + r) + k."""
    out = np.zeros(layer_vocab(vocab, shape), np.float32)
    for i in range(shape.head_installs):
        for k in range(shape.pack):
            planes = [np.frombuffer(words[i, k, j], np.uint8)[:shape.head_rows] for j in range(4)]
            vec = np.ascontiguousarray(np.stack(planes, axis=1)).reshape(-1).view(np.float32)
            out[shape.pack * (shape.head_rows * i + np.arange(shape.head_rows)) + k] = vec
    return out[:vocab]


def logit_words(batch, row):
    """[(install, segment, slice, address)] of the logit row `row`: the west plane's installs (even i) on the quad `lg`, the east
    plane's (odd i) on the quad `lg2`, word `stride` row + pack i + k of either (315 row + 5 i + k on `ref`)."""
    lg, lg2 = batch.plan.by_name["lg"].addr, batch.plan.by_name["lg2"].addr
    sh = shape_of(batch)
    return [(i, k, s, (lg2 if i % 2 else lg) + sh.logit_stride * row + sh.pack * i + k)
            for i in range(sh.head_installs) for k in range(sh.pack) for s in (L.LOGIT_SLICES_EAST if i % 2 else L.LOGIT_SLICES)]


def chip_logits(words, batch, row, vocab):
    """The (vocab) logits of `row` from the dumped words {(slice, addr): bytes} of `logit_words`."""
    by_ik = {}
    for i, k, s, a in logit_words(batch, row):
        by_ik.setdefault((i, k), []).append(words[(s, a)][:batch.kit.p.vlanes])
    return unpack_logits({(i, k, j): w for (i, k), ws in by_ik.items() for j, w in enumerate(ws)}, vocab, shape_of(batch))


def layer_vocab(vocab, shape=REF_SHAPE):
    return shape.pack * shape.head_rows * shape.head_installs


def sa_not(excluded, s, a):
    return (s, a) not in excluded


def run(exe, batch, values, pre, dumps, max_cycles):
    """Run the batch's program with the sites of `values` set, the pool words `pre` preloaded and the words `dumps` read
    back: {'diags', 'lints', 'words': {(slice, addr): bytes}, 'seconds'}."""
    import time
    k, p = batch.kit, batch.kit.p
    prog = tsp_as.assemble(k.b.prog.text(), config=p.path)[1]
    images = patch.to_image(prog.images)
    block_bytes = {(blk.queue, blk.index): bytearray(b"".join(x[3] for x in blk.tensors)) for blk in k.blocks}
    im2, bl2 = patch.apply_paged(images, block_bytes, k.located, values)
    blocks2 = pager.with_block_bytes(k.blocks, {key: bytes(v) for key, v in bl2.items()})
    tensors = {}
    for n, (_, blob) in k.weights.items():
        blob = bytes(blob)
        tensors[n] = (tsp_as.RAW_DTYPE, [blob[i:i + p.vlanes] for i in range(0, len(blob), p.vlanes)])
    tensors.update({n: (tsp_as.RAW_DTYPE, [bytes(w) for w in ws]) for n, ws in pager.tensor_words(blocks2).items()})
    text = tsp_as.preload_text(tsp_as.bind_weights(list(prog.externs.values()), tensors))
    text += tsp_as.input_text(patch.to_units(im2))
    text += "".join(f"S {s:x} {a:x} " + " ".join(f"{x:x}" for x in w) + "\n" for (s, a), w in pre.items())
    text += "".join(f"D {s:x} {a:x}\n" for s, a in dumps)
    t0 = time.time()
    r = subprocess.run([exe, p.path, "--trace", "--lint", "--max-cycles", str(max_cycles)], input=text.encode(),
                       capture_output=True, timeout=3000)
    seconds = time.time() - t0
    assert r.returncode in (0, 3), r.stderr.decode()[:300]
    out = {"diags": [], "lints": [], "words": {}, "seconds": seconds}
    for ln in r.stdout.decode().splitlines():
        f = ln.split()
        if ln.startswith("T diag"):
            out["diags"].append((int(f[2]), int(f[4]), f[5]))
        elif ln.startswith("T lint"):
            out["lints"].append((int(f[2]), int(f[4]), f[5]))
        elif ln.startswith("W "):
            out["words"][(int(f[1], 16), int(f[2], 16))] = bytes(int(x, 16) for x in f[3:])
    return out


def check_step(batch, ctx, exe=ISS):
    """One step of context size `ctx` on the chip against bit-exact reference; returns the run's record."""
    layers, geo = batch.layers, batch.geo
    cfg = batch.cfg or L.serve_config(layers)
    pos, table = CONTEXTS[ctx]
    w = W.synthetic(cfg, E.SEED)
    pool_pre = random_pool(cfg, 100 + ctx)
    pool_post = random_pool(cfg, 100 + ctx)
    vl = batch.kit.p.vlanes
    token = TOKEN if TOKEN < cfg.vocab else TOKEN % cfg.vocab
    want = G.step_batch(w, pool_post, [token], [pos], [table], batch.bucket, unit=vl, pack=shape_of(batch).pack)
    pre = pool_words(geo, pool_pre, layers)
    every = list(range(geo.sink + 1))
    after = pool_words(geo, pool_post, layers, every)
    assert not (set(pre) - set(after)) and len(after) == layers * (16 * 2 + 4 * 16) * len(every)
    dump_logits = [(s, a) for _, _, s, a in logit_words(batch, 0)]
    hid = batch.hidden[0]
    dump_hidden = [(s, hid.addr) for s in hid.slices]
    out = run(exe, batch, E.site_values(batch, [token], [pos], [table]), {sa: w_ for sa, w_ in pre.items()},
              sorted(after) + dump_logits + dump_hidden, batch.kit.last + 200)
    assert out["diags"] == [] and out["lints"] == []
    k_sink = {sa for l in range(layers) for sa in kvpool.k_words(geo, pool_post, l, [geo.sink])}     # written by the 7 other pairs
    bad = [(s, hex(a)) for (s, a), word in sorted(after.items()) if sa_not(k_sink, s, a) and out["words"][(s, a)][:vl] != word]
    assert not bad, f"{len(bad)} pool words differ, first {bad[:6]}"
    got_h = np.ascontiguousarray(np.stack([np.frombuffer(out["words"][(s, hid.addr)][:64], np.uint8) for s in hid.slices],
                                          axis=1)).reshape(-1).view(np.float32)
    assert got_h.tobytes() == want.hidden[0].tobytes()
    got = chip_logits(out["words"], batch, 0, cfg.vocab)
    assert got.tobytes() == want.logits[0].tobytes()
    return out


@pytest.mark.parametrize("ctx", sorted(CONTEXTS))
def test_the_two_layer_step_is_reference_exact_on_the_iss(batches, ctx):
    check_step(batches(2), ctx)


@pytest.mark.parametrize("ctx", sorted(CONTEXTS))
def test_the_eight_layer_step_is_reference_exact_on_the_iss(batches, ctx):
    check_step(batches(8), ctx)


def test_the_two_layer_step_is_reference_exact_on_tsp_sim(batches):
    check_step(batches(2), 20, exe=SIM)


@pytest.mark.parametrize("bucket", [(1, 4, 64), (2, 4, 64), (1, 1, 32), (3, 1, 64), (16, 1, 64), (8, 8, 64)])
def test_buckets_outside_the_grid_are_refused_and_name_it(bucket):
    with pytest.raises(ValueError, match="not in the grid"):
        E.emit_step_batch(*bucket, layers=2)


def test_every_site_of_the_program_gets_a_value(batches):
    b = batches(2)
    values = E.site_values(b, [TOKEN], [19], [[7, 2]])
    assert len(values) == len(b.case["sites"]) == 220
    assert set(values) == {x["name"] for x in b.case["sites"]}


def test_a_k_write_that_misses_the_sink_is_caught(batches, monkeypatch):
    """The 7 non-owning K pairs written to block 100 instead of the sink change a block the step must not touch."""
    b = batches(2)
    orig = E.site_values

    def wrong(batch, tokens, positions, tables):
        v, pos, lane = orig(batch, tokens, positions, tables), positions[0], 0
        for l in range(batch.layers):
            for r in range(8):
                for pl in range(2):
                    if r != pos % 8:
                        v[f"t{lane}.L{l}.kw.p{r}.{pl}"] = batch.geo.kaddr(l, 100)
        return v
    monkeypatch.setattr(E, "site_values", wrong)
    with pytest.raises(AssertionError, match="pool words differ"):
        check_step(b, 20)


def test_the_text_of_one_lane_is_pinned(batches):
    """Measured 2-layer P(1, 1, 64) text after the two-read local fetch correction:
    712 words in 356 blocks of 59 queues. Numerical expectations are unchanged.
    The full per-slice tables are re-derived in `test_step_buckets.py`.
    """
    b = batches(2)
    words = sum(len(blk_data) // b.kit.p.vlanes for blk in b.kit.blocks for (_, _, _, blk_data) in blk.tensors)
    assert (words, len(b.kit.blocks), len({blk.queue for blk in b.kit.blocks})) == (712, 356, 59)


def test_a_fetch_never_shares_a_bundle_with_a_site_or_a_dispatch(batches):
    """Neither fetch form shares a bundle with another dispatch or a patch site.
    Check the full program, including instructions outside the initial IQ image.
    """
    v = batches(2).kit.view2
    writes = {(s.queue, s.cycle) for s in v.full.sites if s.op == "write"}
    reads = {(s.queue, s.cycle) for s in v.full.sites if s.op == "read"}
    fetch_forms = ("ifetch", "ifetchm")
    fetches = [(q, c) for c, q, t in v.added if t.split()[0] in fetch_forms]
    dispatched = {(q, c) for c, q, t in v.full.entries if t.split()[0] not in fetch_forms}
    assert (len(fetches), len(writes), len(reads)) == (356, 40, 178)
    assert not set(fetches) & (writes | reads)
    assert not set(fetches) & dispatched
