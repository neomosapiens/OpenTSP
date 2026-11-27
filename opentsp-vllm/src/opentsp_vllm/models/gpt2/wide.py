"""The kernels of the SHIP-style GPT-2 small pipeline at width 768 as program cases ( sections 3 to 5): each kernel at the sizes of GPT-2 small on `ref`, with synthetic data, every expectation the
width-768 twin of `opentsp_tools.programs.ref_wide` (whose operations `test_numerics.py` checks against the Python oracle).

Layout of an activation: 256-lane vectors (`ref_wide.TILE`); an f16 matrix of d_in features is `ceil(d_in / 256)` tiles of installed
rows (one install of at most 160 rows per tile and block of output rows)."""
import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import geometry as geo
from opentsp_tools.programs import ref_wide as rw
from opentsp_tools.programs.kernels import install as k_install, matvec_f16 as k_matvec, reduce_bcast as k_rb, wide_ln as k_ln, wide_mask as k_wm, wide_scores as k_wsc, wide_softmax as k_wsm
from opentsp_tools.programs.kernels import scatter as k_sc, gather as k_ga, wide_context as k_wc, expand as k_ex
from opentsp_tools.programs import data
import random
from opentsp_tools.programs.kernels.flow import Flow

PROFILES = ["ref"]
GENERIC = ["test/sl8", "test/sl32"]          # the chunk of slots and the butterfly are the same on every width; the tile is the profile's
WIDTHS = ["test/sl32"]                       # cases whose tile is the profile's (`geometry.Geometry.tile`: 256 lanes on 16 or more superlanes)
TAGS = ("sim", "ts-kernel")
TILE = rw.TILE                              # the tile of `ref`: features of a ship activation are counted in these (768 = 3 tiles)
PLANES = (0, 2)          # the f16 planes a tiled matvec alternates between (west, east)


def kit(p, name, doc, seed):
    return Kit(p, f"ship/ts_{name}", doc, seed, module="ship_wide", tags=TAGS, source="numpy-twin")


def pad_lanes(v, vlanes):
    out = np.zeros(vlanes, np.asarray(v).dtype)
    out[:len(v)] = v
    return out


def bind_tiles(k, w, rows, tiles, prefix="w"):
    """The `rows` x (tiles * TILE) f16 matrix `w` as one extern per slice of each plane that a tile uses: tile t is on plane
    PLANES[t % 2], its words at word `ADDR_W + (t // 2) * rows // 8` of the plane's slices. Returns {tile: (plane, addr, text)}."""
    TILE = geo.of(k.p).tile
    per_plane = {pl: [t for t in range(tiles) if PLANES[t % 2] == pl] for pl in PLANES}
    out = {}
    for pl, ts in per_plane.items():
        if not ts:
            continue
        base = k_install.plane_queues(pl)[0]
        planes = [k_install.weight_planes(w[:, t * TILE:(t + 1) * TILE], k.p.vlanes) for t in ts]
        for j in range(16):
            words = [x for tile_words in planes for x in tile_words[j]]
            k.b.prog.extern(f"{prefix}{pl}s{j}", base + j, k_install.ADDR_W, len(words))
            k.weights[f"{prefix}{pl}s{j}"] = ("raw", b"".join(words))
        for n, t in enumerate(ts):
            off = n * rows // 8
            out[t] = (pl, k_install.ADDR_W + off, (lambda pl_, off_: lambda j, i: f"{prefix}{pl_}s{j}+{off_ + i:#x}")(pl, off))
    return out


def matvec_tiled(f, rows, tiles_in, bias, planes, t0=30):
    """y = W x + b for a block of `rows` (<= 160, a multiple of 8) output rows over `len(tiles_in)` tiles of 256 input features: per
    tile an install, the activation narrowed to f16, ABC and ACC (the result of the tile as an f32 vector whose lane l is row l),
    the partials stored; then the f32 sums in ascending tile order ((p0 + p1) + p2 ...) on chained VXM ALUs and the bias last (the
    order of `ref_wide.matvec_tiled`). Returns the result Value."""
    run, t_next = None, t0
    for t, x in enumerate(tiles_in):
        plane, addr, text = planes[t]
        acc = k_matvec.matvec(f, rows, x, None, plane, addr, text, t_next, alu=0 if plane == 0 else 2,
                              to="E" if plane == 0 else "W")
        part = f.store(acc, 4 + t % 2)
        if run is None:
            run = part
        else:                                           # run = run + part, in f32, kept in MEM for the next tile
            run = f.store(f.vxm(1, "add.f32", [run, part]), 6 + t % 2)
        t_next = max(part.ready, run.ready) + 4
    return run if bias is None else f.vxm(1, "add.f32", [run, bias])


def wide_matvec(p, seed=1, rows=64, tiles=3, name="wide_matvec", doc=None):
    """The tiled matvec of GPT-2 small: `rows` output rows (at most the f16 rows one MXM plane holds: 160 on `ref`), over `tiles` * 256
    input features (768 -> 3 tiles of the 256 lanes of `ref`, 3072 -> 12), cut into the tiles of the profile (`Geometry.tile`)."""
    g = geo.of(p)
    TILE = g.tile
    rows = min(rows, g.install_rows)
    n_in = tiles * rw.TILE
    tiles = n_in // TILE
    k = kit(p, name, doc or f"tiled matvec: {rows} rows x {n_in} input features in {tiles} tiles of {TILE}, f32 bias", seed)
    w = (k.rng.standard_normal((rows, n_in)) * 0.05).astype(np.float16)
    x = (k.rng.standard_normal(n_in) * 1.5).astype(np.float32)
    bias = pad_lanes((k.rng.standard_normal(rows) * 0.5).astype(np.float32), p.vlanes)
    planes = bind_tiles(k, w, rows, tiles)
    f = Flow(k)
    f.base = max(f.base, k_install.ADDR_W + -(-tiles // 2) * rows // 8)       # above the weights
    f.low = {"E": 16, "W": 16}                    # the installs of later tiles run on streams 0..15 beside the earlier tiles' values
    xt = rw.split(x, TILE)
    xs = [f.const(f"x{t}", pad_lanes(xt[t], p.vlanes), "W", 1 + t % 2) for t in range(tiles)]
    y = matvec_tiled(f, rows, xs, f.const("b", bias, "W", 3), planes)
    want = np.zeros(p.vlanes, np.float32)
    want[:rows] = rw.matvec_tiled(w, x, bias[:rows], TILE, p.vlanes)
    f.store(y, 0, want)
    return k.done("wide_matvec")


def matvec_768(p, seed=1):
    return wide_matvec(p, seed, rows=64, tiles=3, name="wide_matvec")


def matvec_160x3(p, seed=2):
    """The largest block (160 rows, the install limit) over a 768-feature input: the blocks of 768 -> 2304 and 768 -> 3072."""
    return wide_matvec(p, seed, rows=160, tiles=3, name="wide_matvec_160x3")


def matvec_160x12(p, seed=3):
    """A 160-row block over a 3072-feature input (12 tiles): the blocks of 3072 -> 768."""
    return wide_matvec(p, seed, rows=160, tiles=12, name="wide_matvec_160x12")


def matvec_128x3(p, seed=4):
    """The last block of 768 -> 768 (768 = 4 * 160 + 128 rows)."""
    return wide_matvec(p, seed, rows=128, tiles=3, name="wide_matvec_128x3")


def wide_layernorm(p, seed=5, n=768):
    """LayerNorm over `n` features held as n / 256 vectors of 256 lanes (768: three): Reduce per vector, vector sums, four butterfly
    rounds over 16 superlanes, Distribute; mean, centring, variance, RSqrt, scale and shift."""
    g = geo.of(p)
    TILE, rounds = g.tile, g.rounds
    k = kit(p, "wide_layernorm", f"LayerNorm of {n} features in {n // TILE} vectors of {TILE} lanes, f32", seed)
    maps = (k_rb.load_maps(k, 60, rounds), k_rb.load_distribute_map(k, 80, k_rb.dist_map(rounds)))
    x = (k.rng.standard_normal(n) * 2 + 0.5).astype(np.float32)
    gamma = (1 + k.rng.standard_normal(n) * 0.1).astype(np.float32)
    beta = (k.rng.standard_normal(n) * 0.1).astype(np.float32)
    f = Flow(k)
    pad = lambda v: pad_lanes(v, p.vlanes)       # noqa: E731
    xs = [f.const(f"x{t}", pad(v), "W", 1) for t, v in enumerate(rw.split(x, TILE))]
    gs = [f.const(f"g{t}", pad(v), "W", 2) for t, v in enumerate(rw.split(gamma, TILE))]
    bs = [f.const(f"b{t}", pad(v), "W", 2) for t, v in enumerate(rw.split(beta, TILE))]
    inv_n, eps = f.splat("n", np.float32(1.0 / n), "W", 3), f.splat("e", 1e-5, "W", 3)
    wide = {} if n // TILE <= 3 else dict(quads=(0, 1, 7), rounds=rounds)      # more than three vectors: the sums of the vectors on other quads
    ys = k_ln.layernorm_wide(f, xs, gs, bs, inv_n, eps, maps, 0x10, (1 << p.bank_bit) + 0x40, **wide)
    want = rw.layernorm_wide(x, gamma, beta, 1e-5, tile=TILE)
    for t, (y, w) in enumerate(zip(ys, rw.split(want, TILE))):
        f.store(y, 5 + t, pad(w))
    return k.done("wide_layernorm")


def layernorm_128(p, seed=14, n=768):
    """LayerNorm over `n` features held as n / 128 vectors of 128 lanes (768: six; two heads of 64 per vector): Reduce per vector, vector sums, three
    butterfly rounds over 8 superlanes, Distribute."""
    k = kit(p, "layernorm128", f"LayerNorm of {n} features in {n // 128} vectors of 128 lanes, f32", seed)
    maps = (k_rb.load_maps(k, 60, 3), k_rb.load_distribute_map(k, 80, k_rb.dist_map(3)))
    x = (k.rng.standard_normal(n) * 2 + 0.5).astype(np.float32)
    gamma = (1 + k.rng.standard_normal(n) * 0.1).astype(np.float32)
    beta = (k.rng.standard_normal(n) * 0.1).astype(np.float32)
    f = Flow(k)
    pad = lambda v: pad_lanes(v, p.vlanes)       # noqa: E731
    xs = [f.const(f"x{t}", pad(v), "W", 1) for t, v in enumerate(rw.split(x, 128))]
    gs = [f.const(f"g{t}", pad(v), "W", 2) for t, v in enumerate(rw.split(gamma, 128))]
    bs = [f.const(f"b{t}", pad(v), "W", 5) for t, v in enumerate(rw.split(beta, 128))]
    inv_n, eps = f.splat("n", np.float32(1.0 / n), "W", 3), f.splat("e", 1e-5, "W", 3)
    ys = k_ln.layernorm_wide(f, xs, gs, bs, inv_n, eps, maps, 0x10, (1 << p.bank_bit) + 0x40, quads=(0, 1, 7), rounds=3)
    want = rw.layernorm_wide(x, gamma, beta, 1e-5, tile=128)
    for t, (y, w) in enumerate(zip(ys, rw.split(want, 128))):
        f.store(y, 8 + t % 3, pad(w))
    return k.done("layernorm128")


def wide_scores(p, seed=6, slots=128, heads=12):
    """The attention scores of 12 heads of 64 lanes (the heads of a tile: 4 in 256 lanes) against `slots` cache rows of f16 K (at most
    the f16 rows of a plane: 128 on `ref`): per tile one install of the tile's K rows, per head the query masked to the head's 64 lanes,
    ABC and ACC, then `scale` (0.125) and the additive mask."""
    g = geo.of(p)
    TILE = g.tile
    slots = min(slots, g.chunk)
    k = kit(p, "wide_scores", f"scores of {heads} heads of 64 lanes over {slots} slots in {heads * 64 // TILE} K tiles", seed)
    n = heads * rw.DK
    tiles = n // TILE
    k16 = (k.rng.standard_normal((slots, tiles, TILE)) * 0.7).astype(np.float16)
    q = (k.rng.standard_normal(n) * 1.5).astype(np.float32)
    mask = np.where(k.rng.random(slots) < 0.25, -np.inf, 0).astype(np.float32)
    mask[0] = 0
    group, words = k_install.group(p, k_wsc.PLANE), slots // 8
    for t in range(tiles):
        for j, ws in enumerate(k_install.weight_planes(k16[:, t], p.vlanes)):
            k.b.prog.extern(f"k{t}s{j}", group[j], k_install.ADDR_W + t * words, words)
            k.weights[f"k{t}s{j}"] = ("raw", b"".join(ws))
    f = Flow(k)
    f.low = {"E": k_wsc.RESULT_STREAMS, "W": 0}
    lane_masks = rw.head_lane_mask_wide(heads, n, TILE)              # (heads, tiles, TILE)
    want = rw.scores_wide(k16, q, mask, heads, 0.125, TILE, p.vlanes)
    half, add_mask = f.splat("half", 0.125, "W", 3), f.const("mask", pad_lanes(mask, p.vlanes), "W", 2)
    per, t0 = TILE // rw.DK, 30
    for t, qt in enumerate(rw.split(q, TILE)):
        masks = [f.const(f"m{h}", pad_lanes(lane_masks[h, t], p.vlanes), "E", h % 3) for h in range(t * per, (t + 1) * per)]
        out = k_wsc.tile_scores(f, slots, f.const(f"q{t}", pad_lanes(qt, p.vlanes), "W", 1), masks, half, add_mask, t0,
                                k_install.ADDR_W + t * words, lambda j, i, t=t: f"k{t}s{j}+{i:#x}")
        for h, v in zip(range(t * per, (t + 1) * per), out):
            f.store(v, h % 10, pad_lanes(want[h], p.vlanes))
        t0 = max(v.cycle for v in out) + 40      # the next install's reads run ahead of its LW, on the streams of the results
    return k.done("wide_scores")


def wide_mask(p, seed=7, chunks=8, page=16):
    """The mask of the full pool (1024 slots = 8 chunks of 8 pages) from the run-time encoding: per chunk an encoding vector (lane 16 g = the count
    of page g, as the host sends it each step), Distribute, compare with the position in the page, select +0 / -inf."""
    k = kit(p, "wide_mask", f"attention mask of {chunks * 128} slots from the page-count encoding, f32", seed)
    pages = 128 // page
    counts = k.rng.integers(0, page + 1, size=(chunks, pages))
    counts[0, 0], counts[0, 1], counts[0, 2] = page, 0, 1                 # a full page, an empty one, one slot
    t_dist = k_rb.load_distribute_map(k, 30, k_wm.DIST)
    f = Flow(k)
    f.low = dict(k_wm.STREAM_LOW)
    encs = []
    for c in range(chunks):
        e = np.zeros(p.vlanes, np.float32)
        e[16 * np.arange(pages)] = counts[c]
        encs.append(e)
    enc = f.table("enc", encs, "E", 0)
    pos = f.const("pos", (np.arange(p.vlanes) % 16).astype(np.float32), "W", 4)
    zero, ninf = f.splat("zero", 0.0, "W", 2), f.splat("ninf", -np.inf, "W", 3)
    for c in range(chunks):
        out = k_wm.chunk_mask(f, enc[c], pos, zero, ninf, (1 << p.bank_bit) + 0x40 + c, t_dist + 10 * c, 2 * (c % 8))
        want = np.zeros(p.vlanes, np.float32) - np.inf
        want[:128] = rw.mask_from_encoding(counts[c], 128, page)
        f.store(out, c, want)
    return k.done("wide_mask")


def scatter_kv(p, seed=8, slot=37, tiles=3):
    """The new token's K and V rows (f32, 768 features = 3 tiles of 256) narrowed to f16 and written into the pool by Scatter: a K tile goes to slice pair
    `slot mod 8` of the K plane at word `slot div 8` (the other seven pairs write the sink word of the tile, as the patched write does), a V tile to its
    two slices at word `slot`; the addresses come from map vectors (one per tile and pair for K, one for V). Preloaded random pool words, so that
    only the scattered words may change."""
    k = kit(p, "scatter_kv", f"K and V rows of {tiles} tiles written by Scatter at slot {slot}", seed)
    rng, b = random.Random(seed), k.b
    f = Flow(k)
    f.low = {"E": 12, "W": 0}
    base_k, sink_k, base_v = 0x100, 0x180, 0x200
    k_slices, v_slices = 44, 60                       # K plane 2 (16 slices, 8 pairs); the V quads
    rows = {}
    staged = {}
    for kind in "kv":
        for t in range(tiles):
            row = (k.rng.standard_normal(TILE) * 1.2).astype(np.float32)
            rows[kind, t] = row
            v16 = f.vxm(0, "conv_f32_f16.f32", [f.const(f"{kind}{t}", pad_lanes(row, p.vlanes), "W", 1 + (t % 3))], to="W", width=2)
            staged[kind, t] = f.store(v16, 4 + t + (3 if kind == "v" else 0))
    pool = {}
    for s_ in list(range(k_slices, k_slices + 16)) + list(range(v_slices, v_slices + 2 * tiles)):
        ws = [bytes(rng.randrange(256) for _ in range(p.vlanes)) for _ in range(0x90)]
        pool[s_] = ws
    img = {s_: {a: bytearray(w) for a, w in enumerate(ws, 0x100)} for s_, ws in pool.items()}
    for s_ in img:
        b.prog.extern(f"pool{s_}", s_, 0x100, len(img[s_]))
        k.weights[f"pool{s_}"] = ("i8", b"".join(bytes(img[s_][a]) for a in sorted(img[s_])))

    def map_word(addr):
        w = bytearray(rng.randrange(256) for _ in range(p.vlanes))
        for g in range(p.superlanes):
            w[16 * g], w[16 * g + 1] = addr & 255, addr >> 8
        return bytes(w)

    t_ref = 70
    for t in range(tiles):
        groups = []
        for r in range(8):
            a = base_k + t * 0x10 + slot // 8 if r == slot % 8 else sink_k + t
            nm = f"mk{t}_{r}"
            b.prog.extern(nm, 6 + r, 0x10 + t, 1)
            k.weights[nm] = ("i8", map_word(a))
            groups.append((6 + r, 0x10 + t, nm, (k_slices + 2 * r, k_slices + 2 * r + 1)))
            for pl in range(2):
                pad = np.zeros(p.vlanes, np.float16)
                pad[:TILE] = rw.to_f16(rows["k", t])
                img[k_slices + 2 * r + pl][a] = bytearray(data.to_planes(pad)[pl])
        sv = staged["k", t]
        t_last = k_sc.scatter_row(b, t_ref, [(sv.slices[0], sv.addr), (sv.slices[1], sv.addr)], groups)
        t_ref = t_last + 40
    for t in range(tiles):
        a = base_v + slot
        nm = f"mv{t}"
        b.prog.extern(nm, 6, 0x20 + t, 1)
        k.weights[nm] = ("i8", map_word(a))
        pad = np.zeros(p.vlanes, np.float16)
        pad[:TILE] = rows["v", t].astype(np.float16)
        sv = staged["v", t]
        t_last = k_sc.scatter_row(b, t_ref, [(sv.slices[0], sv.addr), (sv.slices[1], sv.addr)],
                                  [(6, 0x20 + t, nm, (v_slices + 2 * t, v_slices + 2 * t + 1))])
        for pl in range(2):
            img[v_slices + 2 * t + pl][a] = bytearray(data.to_planes(pad)[pl])
        t_ref = t_last + 40
    for s_ in img:
        for a, w in img[s_].items():
            k.expect(s_, a, bytes(w))
    k.last = t_ref
    return k.done("scatter_kv")


def gather_embed(p, seed=9, token=41, pos=9, vocab=64, positions=32):
    """The embedding row wte[token] + wpe[position] of 768 features: both f16 table rows (three 256-lane tiles, six byte planes on six slices each) read by
    sixteen Gathers per plane by the host's maps, OR-ed, widened to f32 and added; three f32 tiles."""
    k = kit(p, "gather_embed", f"embedding row wte[{token}] + wpe[{pos}] by Gather, 768 features", seed)
    rng, b = random.Random(seed), k.b
    f = Flow(k)
    f.low = {"E": 16, "W": 0}
    tabs = {}
    for name, rows_, first in (("wte", vocab, 8), ("wpe", positions, 16)):
        tab = (k.rng.standard_normal((rows_, 3, TILE)) * 0.4).astype(np.float16)
        tabs[name] = tab
        for j in range(6):
            n, pl = divmod(j, 2)
            ws = []
            for r in range(rows_):
                pad = np.zeros(p.vlanes, np.float16)
                pad[:TILE] = tab[r, n]
                ws.append(data.to_planes(pad)[pl])
            b.prog.extern(f"{name}{j}", first + j, 0x100, rows_)
            k.weights[f"{name}{j}"] = ("i8", b"".join(ws))

    def map_word(addr):
        w = bytearray(rng.randrange(256) for _ in range(p.vlanes))
        for g in range(p.superlanes):
            w[16 * g], w[16 * g + 1] = addr & 255, addr >> 8
        return bytes(w)

    b.prog.extern("mtok", 30, 0x10, 1)
    k.weights["mtok"] = ("i8", map_word(0x100 + token))
    b.prog.extern("mpos", 31, 0x10, 1)
    k.weights["mpos"] = ("i8", map_word(0x100 + pos))
    wte = k_ga.embed_row(f, [8 + j for j in range(6)], (30, 0x10, "mtok", 0), 60, 0, 0x300, 44, 5, 0x500)
    wpe = k_ga.embed_row(f, [16 + j for j in range(6)], (31, 0x10, "mpos", 7), 60, 0, 0x340, 56, 8, 0x510)
    for n in range(3):
        e1, e2 = f.store(wte[n], n), f.store(wpe[n], 3 + n)
        y = f.vxm(1, "add.f32", [e1, e2], to="E")
        want = pad_lanes(tabs["wte"][token, n].astype(np.float32) + tabs["wpe"][pos, n].astype(np.float32), p.vlanes)
        f.store(y, 6 + n, want)
    return k.done("gather_embed")


def wide_softmax(p, seed=10, chunks=8, heads=2):
    """The softmax of `heads` heads over a pool of `chunks` chunks of 128 slots (1024 slots): the maximum and the sum combined over the chunks on ALU chains,
    the tree of 8 superlanes, exp, the reciprocal as rsqrt(sum * sum), the product. Masked slots (-inf), a fully masked chunk and a fully
    masked head tail are in the data; lanes 128..319 of a chunk vector hold finite junk (the machine vector is wider than a chunk)."""
    k = kit(p, "wide_softmax", f"softmax of {heads} heads over {chunks * 128} slots in {chunks} chunks of 128, f32", seed)
    maps = (k_rb.load_maps(k, 60, 3), k_rb.load_distribute_map(k, 80, k_rb.dist_map(3)), 3)
    f = Flow(k)
    L = p.vlanes
    s = (k.rng.standard_normal((heads, chunks, L)) * 3).astype(np.float32)
    s[0, 1, 20:90] = -np.inf
    s[0, 3] = -np.inf                                   # a fully masked chunk
    s[0, 5, 100:128] = -np.inf
    s[1, :, 64:128] = -np.inf                           # the upper half of every chunk masked
    s[1, 7, :] = -np.inf
    s[1, 2, 5] = 11.0
    want = rw.softmax_pool(s.reshape(heads, chunks * L), chunk=L, seg=8).reshape(heads, chunks, L)
    bank1 = 1 << p.bank_bit
    for h in range(heads):
        vecs = [f.const(f"s{h}_{c}", s[h, c], "E", 1 + c) for c in range(chunks)]
        out = k_wsm.softmax_pool(f, vecs, maps, 0x10 + 16 * h, bank1 + 0x40 + 2 * h, [1 + c for c in range(chunks)], at=k.last)
        k.last = max(k.last, max(v.ready for v in out))
        for c, v in enumerate(out):
            for sl, plane in zip(v.slices, data.to_planes(want[h, c])):
                k.expect(sl, v.addr, plane)
    return k.done("wide_softmax")


def wide_context(p, seed=11, slots=16, heads=4):
    """The PV product of one tile of `heads` heads (64 lanes each) over `slots` slots of f16 V on the VXM: the expanded probabilities P_s (f32, lane l =
    probability of head l div 64) and the f16 V rows are read from MEM, widened, multiplied and added in four interleaved chains."""
    TILE = geo.of(p).tile
    heads = min(heads, TILE // rw.DK)               # the heads of one tile
    k = kit(p, f"wide_context{slots}", f"PV of {heads} heads over {slots} slots, f16 V widened, f32 sum in four interleaved chains", seed)
    f = Flow(k)
    probs = np.abs(k.rng.standard_normal((heads, slots))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    v16 = (k.rng.standard_normal((slots, 1, TILE)) * 1.5).astype(np.float16)
    pe = np.repeat(rw.from_f16(rw.to_f16(probs)).T, rw.DK, axis=1)                  # (slots, TILE)
    probability_quads = (1, 2, 4)
    tables = [f.table(f"p{i}", [pad_lanes(r, p.vlanes) for r in pe[i::3]], "E", quad)
              for i, quad in enumerate(probability_quads)]
    ps = [tables[s % 3][s // 3] for s in range(slots)]
    vs = f.table("v", [pad_lanes(r, p.vlanes).astype(np.float16) for r in v16[:, 0]], "E", 3)
    z, zc = k_wc.context(f, ps, vs, f.splat("z", 0.0, "W", 3), 30, (5, 6))
    want = pad_lanes(rw.context_vxm(probs, v16, tile=TILE), p.vlanes)
    f.store(z, 0, want)
    f.store(zc, 1, rw.to_f16(want))
    return k.done(f"wide_context{slots}")


def wide_expand(p, seed=12, slots=32, heads=4):
    """The probabilities of the `heads` heads of one tile (f32, lane s = slot s) to the per-lane vectors P_s (lane l = probability of head l div 64, rounded
    to f16): W' of 256 rows installed from the VXM, then one ABC with the f16 one-hot vector of the slot and one ACC per slot."""
    k = kit(p, f"wide_expand{slots}", f"probabilities of {heads} heads of 64 lanes expanded to the per-lane vectors of {slots} slots", seed)
    f = Flow(k)
    probs = np.abs(k.rng.standard_normal((heads, slots))).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    probs[1, 5:] = 0.0
    hv = [f.const(f"p{h}", pad_lanes(probs[h], p.vlanes), "W", h % 4) for h in range(heads)]
    one_hots = f.table("e", k_ex.one_hot(slots, p.vlanes), "W", 3)
    out = k_ex.expand(f, hv, one_hots, rw.DK, 40, 2)
    pe = np.repeat(rw.from_f16(rw.to_f16(probs)).T, rw.DK, axis=1)
    for s_, v in enumerate(out):
        for sl, plane in zip(v.slices, data.to_planes(pad_lanes(pe[s_], p.vlanes))):
            k.expect(sl, v.addr, plane)
    return k.done(f"wide_expand{slots}")


CASES = [
    ("ship/ts_layernorm128", layernorm_128, PROFILES + GENERIC, {}),
    ("ship/ts_wide_expand32", wide_expand, PROFILES + WIDTHS, {"slots": 32, "heads": 2}),
    ("ship/ts_wide_context16", wide_context, PROFILES + GENERIC, {"slots": 16}),
    ("ship/ts_wide_context96", wide_context, PROFILES, {"slots": 96}),
    ("ship/ts_wide_softmax", wide_softmax, PROFILES + GENERIC, {}),
    ("ship/ts_gather_embed", gather_embed, PROFILES, {}),
    ("ship/ts_scatter_kv", scatter_kv, PROFILES, {}),
    ("ship/ts_wide_mask", wide_mask, PROFILES + GENERIC, {}),
    ("ship/ts_wide_scores", wide_scores, PROFILES, {}),
    ("ship/ts_wide_layernorm", wide_layernorm, PROFILES + WIDTHS, {}),
    ("ship/ts_wide_matvec", matvec_768, PROFILES + GENERIC, {}),
    ("ship/ts_wide_matvec_160x3", matvec_160x3, PROFILES, {}),
    ("ship/ts_wide_matvec_160x12", matvec_160x12, PROFILES, {}),
    ("ship/ts_wide_matvec_128x3", matvec_128x3, PROFILES, {}),
]
