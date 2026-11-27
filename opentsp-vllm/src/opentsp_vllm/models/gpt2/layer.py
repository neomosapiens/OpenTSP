"""One GPT-2 layer of the SHIP-style pipeline as one program ( sections 3 to 5):
LayerNorm, the Q, K, V projections, the new token's K and V rows written into the pool by Scatter (the addresses come from host map vectors),
the full-pool attention (`ship_attn.attend`), the output projection, the residual, LayerNorm, the FFN with GELU and the residual.

Layout: activations are vectors of 128 lanes (two heads of 64; `ship_attn.TILE`), `T` of them for the hidden width 128 T (T = 6 is GPT-2 small:
768, 12 heads, FFN 3072 = 24 vectors). A matrix of d_out rows and d_in features is installed block by block: the block of output tile o and input tile i
is 128 rows of 128 f16 features (16 words on each of the 16 slices of a hemisphere), its partial dot product rounded once to f32; the partials of the
input tiles are added in ascending order and the bias last (`ref_wide.matvec_tiled`, tile 128). The twin of every step is `ref_wide` / `ref`.

Memory (words of a slice): weights from `WBASE` in bank 0 of the 16 slices of each plane (west 0..15, east 44..59), the K pool in bank 1 from `K_BASE`
on the east plane (the pager's text starts at 6144), the V pool on east slices 76..87 from `ship_attn.V_BASE`. Every expectation is the twin; the
whole layer's output is the checked result (`LAYER_DEBUG=1` also expects every intermediate vector). Text beyond the instruction queue is paged."""
import os

import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import ref
from opentsp_tools.programs import ref_wide as rw
from opentsp_tools.programs.kernels import gelu as k_gelu, install as k_install, matvec_f16 as k_matvec, reduce_bcast as k_rb, scatter as k_sc, wide_ln as k_ln
from opentsp_tools.programs import data
from opentsp_tools.programs.kernels.flow import Flow, Vec
from opentsp_vllm.models.gpt2 import attention as sa
from opentsp_vllm.models.gpt2.attention import PagedKit, TILE, CHUNK, SCALE
from opentsp_vllm.models.gpt2.wide import pad_lanes, PLANES

PROFILES = ["ref"]
TAGS = ("sim", "ts-kernel")
WBASE = 0x200            # first word of the weights on the plane slices (bank 0)
K_BASE = 0x1400          # the K pool: bank 1 of the east plane, 16 words per tile and chunk
SINK = 0x1700            # the words the seven other slice pairs of a K Scatter write (one per tile)
GAP = 12
EPS = 1e-5
MAP_K, MAP_V = 0x10, 0x20      # words of the host's map vectors (not on the weight slices nor the busy quads 4 and 5: a Repeat needs an unpaged queue)
MAP_SL = 36              # the K maps on west slices 36..43, the V map on slice MAP_V_SL
MAP_V_SL = 31


def bind_blocks(k, p, mats):
    """The matrices `mats` = [(name, w (d_out, d_in) f16)] as 128 x 128 blocks: block (o, i) on plane PLANES[i % 2], 16 words after the plane's previous block
    (one extern per slice of the plane from `WBASE`). Returns {(name, o, i): (plane, word address, text(j, i))} and the words per plane slice."""
    words = {pl: [[] for _ in range(16)] for pl in PLANES}
    ptr = {pl: 0 for pl in PLANES}
    out = {}
    for name, w in mats:
        for o in range(w.shape[0] // TILE):
            for i in range(w.shape[1] // TILE):
                pl = PLANES[i % 2]
                for j, ws in enumerate(k_install.weight_planes(w[o * TILE:(o + 1) * TILE, i * TILE:(i + 1) * TILE], p.vlanes)):
                    words[pl][j] += ws
                out[name, o, i] = (pl, WBASE + ptr[pl], (lambda pl_, off: lambda j, n: f"lw{pl_}s{j}+{off + n:#x}")(pl, ptr[pl]))
                ptr[pl] += TILE // 8
    for pl in PLANES:
        base = k_install.plane_queues(pl)[0]
        for j in range(16):
            k.b.prog.extern(f"lw{pl}s{j}", base + j, WBASE, len(words[pl][j]))
            k.weights[f"lw{pl}s{j}"] = ("raw", b"".join(words[pl][j]))
    return out, max(ptr.values())


def matvec_block(f, tiles_in, bias, blocks, t0, to="E"):
    """One output tile: y = W x + b over the input tiles `tiles_in` (Vecs), the blocks `blocks[i]` = (plane, addr, text); the order of
    `ref_wide.matvec_tiled`: per input tile an install, ABC and ACC, the partials added in ascending order, the bias last."""
    run, t_next = None, t0
    for t, x in enumerate(tiles_in):
        plane, addr, text = blocks[t]
        acc = k_matvec.matvec(f, TILE, x, None, plane, addr, text, t_next, alu=0 if plane == 0 else 2, to="E" if plane == 0 else "W")
        part = f.store(acc, 4 + t % 2)
        if run is None:
            run = part
        else:
            run = f.store(f.vxm(1, "add.f32", [run, part]), 6 + t % 2)
        t_next = max(part.ready, run.ready) + 4
    return f.vxm(1, "add.f32", [run, bias], to=to)


X_QUAD = 4                # the east quad that holds the host's input tiles
PCI_RX, PCI_TX = 142, 126        # the C2C queues of the host path: PciReceive on the east RX queue (it writes row W), PciSend on the west TX queue (it reads row W)


def _tiles_in(k, f, p, tiles, cycles, op, text):
    """The input tiles from outside the chip: for every byte plane n of every tile (4 per f32 tile) an instruction `text(stream)` of mnemonic `op` on the
    east RX queue at `cycles[n]` puts a vector on stream 24 + n mod 8 of row W and a Write commits it to its slice of east quad `X_QUAD`. Returns one Vec per
    tile and the planes' bytes in arrival order."""
    b, vecs, order = k.b, [], []
    for t, tile in enumerate(tiles):
        a = f.addr("E", X_QUAD, 1)
        sl = [44 + 4 * X_QUAD + j for j in range(4)]
        planes = data.to_planes(pad_lanes(tile, p.vlanes))
        done = 0
        for j in range(4):
            n = 4 * t + j
            st = 24 + n % 8
            _, val = b.unit(PCI_RX, op, text(st), dst=("W", st, 1), cycle=cycles[n])
            done = max(done, b.write(val, [sl[j]], [a]))
            order.append(planes[j].hex())
        vecs.append(Vec(sl, a, done + 1, "E"))
    k.last = max(k.last, done)
    return vecs, order


def pci_in(k, f, p, T, tiles, t0=100):
    """The host's DMA engine pushes the four byte planes of every tile (one vector per host.cycles_per_vector cycles), PciReceive takes them from `t0` on."""
    cpv = p.cfg["host"]["cycles_per_vector"]
    vecs, k.host_in = _tiles_in(k, f, p, tiles, [t0 + cpv * n for n in range(4 * len(tiles))], "pci_receive", lambda st: f"pci_receive W:{st}")
    return vecs


def link_in(k, f, p, tiles, cycles, src, link):
    """The tiles of the previous stage: a Receive on `link` (from chip `src`) at each of `cycles`, one per byte plane."""
    return _tiles_in(k, f, p, tiles, cycles, "receive", lambda st: f"receive {src:#x}, {link:#x}, W:{st}")[0]


def _tiles_out(k, f, p, outs, tiles, op, text, spacing, t_c):
    """Per plane of every output tile a Read of the stored word onto row W and instruction `text(stream)` of mnemonic `op` on the west TX queue, `spacing`
    cycles apart from `t_c` on. Returns the dispatch cycle of each and the planes' bytes in order."""
    b, cycles, order = k.b, [], []
    f.low = {"E": 0, "W": 0}
    for v, tile in zip(outs, tiles):
        planes = data.to_planes(tile)
        for j in range(4):
            val = b.read_for(t_c, op, PCI_TX, [v.slices[j]], [v.addr], "W", f.stream("W", 1))
            b.unit(PCI_TX, op, text(val.stream), [val])
            cycles.append(t_c)
            order.append(planes[j].hex())
            t_c += spacing
    k.last = max(k.last, t_c)
    return cycles, order


def host_out(k, f, p, outs, tiles):
    """The output tiles to the host: PciSend, `host.cycles_per_vector` cycles apart."""
    cpv = p.cfg["host"]["cycles_per_vector"]
    t_c = max([k.last] + [v.ready for v in outs]) + 60
    k.host_out = _tiles_out(k, f, p, outs, tiles, "pci_send", lambda st: f"pci_send W:{st}", cpv, t_c)[1]


def link_out(k, f, p, outs, tiles, dst, link):
    """The output tiles to chip `dst` on `link`: Send, `c2c.cycles_per_vector` cycles apart. Returns the dispatch cycle of each."""
    t_c = max([k.last] + [v.ready for v in outs]) + 60
    return _tiles_out(k, f, p, outs, tiles, "send", lambda st: f"send {dst:#x}, {link:#x}, W:{st}", p.cfg["c2c"]["cycles_per_vector"], t_c)[0]


def params(rng, T):
    h = TILE * T
    g = lambda *shape: rng.standard_normal(shape)               # noqa: E731
    return dict(
        ln1g=(1 + g(h) * 0.1).astype(np.float32), ln1b=(g(h) * 0.1).astype(np.float32),
        wqkv=(g(3 * h, h) * 0.05).astype(np.float16), bqkv=(g(3 * h) * 0.1).astype(np.float32),
        wo=(g(h, h) * 0.05).astype(np.float16), bo=(g(h) * 0.1).astype(np.float32),
        ln2g=(1 + g(h) * 0.1).astype(np.float32), ln2b=(g(h) * 0.1).astype(np.float32),
        wu=(g(4 * h, h) * 0.05).astype(np.float16), bu=(g(4 * h) * 0.1).astype(np.float32),
        wd=(g(h, 4 * h) * 0.025).astype(np.float16), bd=(g(h) * 0.1).astype(np.float32))


def twin(x, P, k16, v16, slot, mask, heads):
    """The layer on the twins of `ref_wide` (tile 128): every intermediate vector, by name."""
    r = {}
    r["xn"] = rw.layernorm_wide(x, P["ln1g"], P["ln1b"], EPS, tile=TILE)
    qkv = rw.matvec_tiled(P["wqkv"], r["xn"], P["bqkv"], tile=TILE)
    h = len(x)
    r["q"], r["k"], r["v"] = qkv[:h], qkv[h:2 * h], qkv[2 * h:]
    k16, v16 = k16.copy(), v16.copy()
    k16[slot] = rw.split(rw.to_f16(r["k"]), TILE)
    v16[slot] = rw.split(rw.to_f16(r["v"]), TILE)
    r["k16"], r["v16"] = k16, v16
    r["ctx"] = rw.attention_pool(r["q"], k16, v16, mask, heads, SCALE, pv="vxm", tile=TILE)
    r["ao"] = rw.matvec_tiled(P["wo"], r["ctx"], P["bo"], tile=TILE)
    r["h1"] = rw._add(x, r["ao"])
    r["xn2"] = rw.layernorm_wide(r["h1"], P["ln2g"], P["ln2b"], EPS, tile=TILE)
    r["u"] = rw.matvec_tiled(P["wu"], r["xn2"], P["bu"], tile=TILE)
    r["g"] = ref.gelu(r["u"])
    r["d"] = rw.matvec_tiled(P["wd"], r["g"], P["bd"], tile=TILE)
    r["out"] = rw._add(r["h1"], r["d"])
    return r


def map_weights(p, T, chunks, slot, seed, *, k_base=None, v_base=None, sink=None):
    """The host's map vectors of a step that writes pool slot `slot` ({extern name: one word}): per tile t and slice pair r (0..7) the K map `mk{t}_{r}`
    (the pair of `slot % 8` gets the word's address, the seven others the sink word of the tile), and the V map `mv`. Only lanes 16 g, 16 g + 1 of a
    superlane g matter; the others are noise from `seed`."""
    k_base = K_BASE if k_base is None else k_base
    v_base = sa.V_BASE if v_base is None else v_base
    sink = SINK if sink is None else sink
    rngm = np.random.default_rng(seed)
    c_s, i_word = slot // CHUNK, (slot % CHUNK) // 8
    names, addresses = [], []
    for t in range(T):
        a = k_base + 16 * (t * chunks + c_s) + i_word
        for r in range(8):
            names.append(f"mk{t}_{r}")
            addresses.append(a if r == slot % 8 else sink + t)
    names.append("mv")
    addresses.append(v_base + slot)
    # A ref word contains a whole number of RNG output words. Drawing the
    # matrix at once keeps the existing seeded byte stream for every map.
    words = rngm.integers(0, 256, (len(names), p.vlanes), dtype=np.uint8)
    addr = np.array(addresses, dtype=np.uint16)[:, None]
    words[:, 0::16] = addr & 255
    words[:, 1::16] = addr >> 8
    return dict(zip(names, map(bytes, words), strict=True))


def layer(p, seed=21, T=2, chunks=2, slot=37, page=16, name=None, pci=False, io=None, t_base=0, x=None, info=None, P=None, pools=None):
    """A layer of hidden width 128 T (2 T heads, FFN 4 x) over a pool of `chunks` chunks of 128 slots; the new token goes to pool slot `slot`.
    `pci`: the stage form of the layer, whose input x arrives from the host (PciReceive of the byte planes of its tiles, written to the east quad
    `X_QUAD`) and whose output goes back to the host (PciSend of the planes of every output tile). `io` = {"inp": "const" | "pci" | "link", "out": "none" |
    "pci" | "link", ...} chooses the ends of a stage in a chain (`chain`): a "link" input names `src` (chip), `link` and the `recv` cycles, a "link" output
    `dst` and `dlink`; `t_base` is the cycle the stage's own work starts at, `x` the input (default: random), `info` receives the output and the Send cycles.
    `P` (default: random, see `params`) are the layer's parameters, `pools` = (k16, v16, counts) the K and V pools (slots, T, 128) f16 and the page counts (chunks, pages)
    (default: random): a real layer and the pools of a real context. The program text does not depend on them."""
    H, heads, slots, pages = TILE * T, 2 * T, CHUNK * chunks, CHUNK // page
    debug = bool(os.environ.get("LAYER_DEBUG"))
    k = PagedKit(p, f"ship/ts_{name or f'layer{T}x{chunks}'}", f"one GPT-2 layer: hidden {H}, {heads} heads, FFN {4 * H}, {slots} pool slots; 128-lane tiles",
                 seed, module="ship_layer", tags=TAGS, source="numpy-twin")
    rng, b = k.rng, k.b
    P = params(rng, T) if P is None else P
    x_rand = rng.standard_normal(H).astype(np.float32)
    x = x_rand if x is None else x
    io = io or ({"inp": "pci", "out": "pci"} if pci else {"inp": "const", "out": "none"})
    k16 = (rng.standard_normal((slots, T, TILE)) * 0.7).astype(np.float16)
    v16 = (rng.standard_normal((slots, T, TILE)) * 1.2).astype(np.float16)
    counts = rng.integers(0, page + 1, size=(chunks, pages))
    counts[0, 0] = page
    if pools is not None:
        k16, v16, counts = np.array(pools[0], np.float16), np.array(pools[1], np.float16), np.array(pools[2], np.int64)
    c_s, pg = slot // CHUNK, (slot % CHUNK) // page
    counts[c_s, pg] = max(counts[c_s, pg], slot % page + 1)               # the new token's slot is live
    mask = np.concatenate([rw.mask_from_encoding(counts[c], CHUNK, page) for c in range(chunks)])
    R = twin(x, P, k16, v16, slot, mask, heads)
    # ---- binding: weights, pools, host maps
    blocks, wmax = bind_blocks(k, p, [("qkv", P["wqkv"]), ("o", P["wo"]), ("u", P["wu"]), ("d", P["wd"])])
    assert WBASE + wmax <= 1 << p.bank_bit, wmax
    sa.bind_k_pool(k, p, k16, T, chunks, K_BASE)
    vs = sa.v_pool(k, p, v16, T)
    f = Flow(k)
    mw = map_weights(p, T, chunks, slot, seed)
    kmaps = {}
    for t in range(T):
        for r in range(8):
            nm = f"mk{t}_{r}"
            b.prog.extern(nm, MAP_SL + r, MAP_K + t, 1)
            k.weights[nm] = ("i8", mw[nm])
            kmaps[t, r] = nm
    b.prog.extern("mv", MAP_V_SL, MAP_V, 1)
    k.weights["mv"] = ("i8", mw["mv"])
    # ---- constants (west quads 4.. keep clear of the weights on quads 0..3)
    dist_id = k_rb.dist_map(3)
    maps = (k_rb.load_maps(k, 60 + t_base, 3), k_rb.load_distribute_map(k, 80 + t_base, dist_id), 3)
    inv_n, eps = f.splat("invn", np.float32(1.0 / H), "W", 4), f.splat("eps", EPS, "W", 4)
    gc = [f.splat(n, v, "W", 4) for n, v in (("gc0", ref.GELU_C0), ("gc1", ref.GELU_C1), ("one", ref.ONE), ("hf", ref.HALF))]
    wq = [5, 6, 7, 8, 9, 10]
    cn = {"n": 0}

    def vecs(name_, arr):
        out = []
        for t, v in enumerate(rw.split(arr, TILE)):
            cn["n"] += 1
            out.append(f.const(f"{name_}{t}", pad_lanes(v, p.vlanes), "W", wq[cn["n"] % len(wq)]))
        return out

    if io["inp"] == "pci":
        xv = pci_in(k, f, p, T, rw.split(x, TILE), 100 + t_base)
    elif io["inp"] == "link":
        xv = link_in(k, f, p, rw.split(x, TILE), io["recv"], io["src"], io["link"])
    else:
        xv = vecs("x", x)
    gamma1, b1, gamma2, b2 = vecs("g1_", P["ln1g"]), vecs("b1_", P["ln1b"]), vecs("g2_", P["ln2g"]), vecs("b2_", P["ln2b"])
    bqkv, bo, bu, bd = vecs("bqkv", P["bqkv"]), vecs("bo", P["bo"]), vecs("bu", P["bu"]), vecs("bd", P["bd"])
    state = {"t": max([maps[1]] + [v.ready for v in xv]) + GAP, "kq": 0}

    def after(*vs_):
        state["t"] = max([state["t"], k.last] + [v.ready for v in vs_ if v is not None]) + GAP
        return state["t"]

    keep_q = [("E", 8), ("E", 9), ("E", 10)]

    def keep(v, want=None, to=None):
        """Store a value for later phases on a rotating east quad (or `to` = (hemi, quad))."""
        hemi, quad = to or keep_q[state["kq"] % 3]
        state["kq"] += 1
        pad = None if want is None else pad_lanes(want, p.vlanes)
        out = f.store(v, quad, expect=pad, hemi=hemi)
        after(out)
        return out

    def ln(src, gam, bet, a0, out_addr, want):
        f.low = {"E": 0, "W": 0}
        ys = k_ln.layernorm_wide(f, src, gam, bet, inv_n, eps, maps[:2], a0, (1 << p.bank_bit) + out_addr, quads=(0, 1, 1 + T), rounds=3, at=after(*src))
        w = rw.split(want, TILE)
        res = [keep(y, w[t] if debug else None) for t, y in enumerate(ys)]
        return res

    def mm(name_, o, tiles_in, bias, want, to="E", dest=None):
        f.low = {"E": 16, "W": 16}
        y = matvec_block(f, tiles_in, bias, [blocks[name_, o, i] for i in range(len(tiles_in))], after(*tiles_in), to=to)
        return keep(y, want, dest)

    # ---- LayerNorm 1, Q K V
    xn = ln(xv, gamma1, b1, 0x10, 0x40, R["xn"])
    qs, ks, vs_new = [], [], []
    for o in range(3 * T):
        part = o // T
        want = rw.split(R[("q", "k", "v")[part]], TILE)[o % T] if debug else None
        y = mm("qkv", o, xn, bqkv[o], want, to="W" if part == 0 else "E", dest=("W", 6 + o % 4) if part == 0 else None)
        (qs, ks, vs_new)[part].append(y)
    # ---- Scatter K and V rows of the new token into the pools
    f.low = {"E": 12, "W": 0}
    t_ref = after() + 30
    for kind, rows in (("k", ks), ("v", vs_new)):
        for t, row in enumerate(rows):
            v16n = f.vxm(0, "conv_f32_f16.f32", [row], to="W", width=2, at=t_ref)
            sv = f.store(v16n, 6 + t % 3, hemi="W")
            t_ref = max(t_ref, sv.ready) + 40
            if kind == "k":
                groups = [(MAP_SL + r, MAP_K + t, kmaps[t, r], (44 + 2 * r, 45 + 2 * r)) for r in range(8)]
            else:
                groups = [(MAP_V_SL, MAP_V, "mv", (sa.V_FIRST + 2 * t, sa.V_FIRST + 2 * t + 1))]
            t_ref = k_sc.scatter_row(b, t_ref, [(sv.slices[0], sv.addr), (sv.slices[1], sv.addr)], groups) + 40
    k.last = max(k.last, t_ref)
    state["t"] = t_ref + 40
    # ---- attention over the pool
    f.low = {"E": 0, "W": 0}
    twin_ = None
    zs, cur = sa.attend(p, k, f, qs, counts, vs, state["t"], want=rw.split(R["ctx"], TILE), page=page, k_base=K_BASE, twin=twin_, maps=maps)
    state["t"] = cur
    after(*zs)
    # ---- output projection, residual, LayerNorm 2
    ao = [mm("o", o, zs, bo[o], rw.split(R["ao"], TILE)[o] if debug else None) for o in range(T)]
    h1 = []
    for o in range(T):
        f.low = {"E": 0, "W": 0}
        y = f.vxm(0, "add.f32", [xv[o], ao[o]], to="E", at=after(ao[o]))
        h1.append(keep(y, rw.split(R["h1"], TILE)[o] if debug else None))
    xn2 = ln(h1, gamma2, b2, 0x20, 0x48, R["xn2"])
    # ---- FFN
    us = []
    gs = []
    for o in range(4 * T):
        u = mm("u", o, xn2, bu[o], rw.split(R["u"], TILE)[o] if debug else None)
        f.low = {"E": 0, "W": 0}
        g_ = k_gelu.gelu_vec(f, u, *gc, to="E")
        gs.append(keep(g_, rw.split(R["g"], TILE)[o] if debug else None))
        us.append(u)
    outs = []
    for o in range(T):
        d = mm("d", o, gs, bd[o], rw.split(R["d"], TILE)[o] if debug else None, to="W", dest=("W", 9))
        f.low = {"E": 0, "W": 0}
        y = f.vxm(0, "add.f32", [h1[o], d], to="E", at=after(d))
        outs.append(keep(y, rw.split(R["out"], TILE)[o], to=("E", 3)))
    out_tiles = [pad_lanes(v, p.vlanes) for v in rw.split(R["out"], TILE)]
    if io["out"] == "pci":
        host_out(k, f, p, outs, out_tiles)
    elif io["out"] == "link":
        dsts = io.get("dsts") or [(io["dst"], io["dlink"])]       # `dsts`: the same tiles to several chips (the head chips), one after the other
        sends_all = [link_out(k, f, p, outs, out_tiles, dst, dl) for dst, dl in dsts]
        if info is not None:
            info["sends"] = sends_all[0]
            info["sends_all"] = sends_all
    if info is not None:
        info["out"], info["x"], info["end"] = R["out"], x, k.last
    text, case = k.done(f"layer{T}x{chunks}", floor=6144)
    if io["inp"] == "pci":
        case["host"] = {"dma": [[0, h] for h in k.host_in]}
        case["tags"] = case["tags"] + ["host"]
    if io["out"] == "pci":
        case["expect"]["egress"] = k.host_out
    return text, case


LINK_RX = 228            # a Receive is placed this many cycles after its Send: the largest link latency, so every latency of the profile is covered


def link_peer(chips, links, c, l):
    """The far end of chip `c`'s link `l`, `(chip, its link)` or None (a port of `tsp_sim.c link_peer`, `harness/links.rs`): K = 2..8 chips are one node
    in which local link l reaches chip (i + l + 1) mod K on its link K - 2 - l; K = 8 N are nodes of 8, and global port p = 4 (i mod 8) + g of node n (link 7 + g)
    reaches node (n + p + 1) mod N on port N - 2 - p."""
    if l >= links:
        return None
    if chips == 1:
        return (c, l)
    nn, nk = (1, chips) if chips <= 8 else (chips // 8, 8)
    node, k = divmod(c, nk)
    if l <= 6:
        return None if l + 1 >= nk else (node * nk + (k + l + 1) % nk, nk - 2 - l)
    if l <= 10 and nn > 1:
        q_ = 4 * k + (l - 7)
        if q_ + 1 >= nn:
            return None
        m, q = (node + q_ + 1) % nn, nn - 2 - q_
        return (m * nk + q // 4, 7 + q % 4)
    return None


def hop(chips, links, a, b):
    """The (send link on chip a, receive link on chip b) of the first link of `a` that reaches `b`."""
    for l in range(links):
        peer = link_peer(chips, links, a, l)
        if peer is not None and peer[0] == b and a != b:
            return l, peer[1]
    raise ValueError(f"chip {a} has no link to chip {b} in a system of {chips} chips")


def chain(p, n=3, T=2, chunks=2, seed=31, slot=37, params=None, pools=None, x0=None, route=None, chips=None, fan=None):
    """`n` stages in a line, one layer each, one program per chip: chip 0 takes x from the host (PciReceive), every chip runs its layer, chip i Sends its
    output tiles on link 0 to chip i + 1, which Receives them on its link 1 (the wiring of `tsp_sim --chips n` for a node of 3 to 8 chips), the last
    chip gives its output to the host (PciSend). Returns [(text, case)] per chip and the info of each stage; the twin of the chain is the stages' twins in
    sequence. Chip i > 0 starts its own work when its input has arrived (`t_base`). `params[i]` and `pools[i]` (see `layer`) are the parameters and the pools of
    stage i, `x0` the host's input. `route` (default: chips 0 .. n - 1) lists the chip of each stage in a system of `chips` chips (default: the stages' count): the
    links of every hop are the first that connect the two chips (`hop`). `fan` (chips): the last stage Sends its output tiles to each of these chips (in turn) instead of
    to the host; `infos[-1]["sends_all"]` then holds the Send cycles per destination."""
    out, infos, x, recv = [], [], x0, None
    chips = chips or n
    route = list(range(n)) if route is None else route
    links = p.cfg["c2c"]["links"]
    for i in range(n):
        io = {"inp": "pci" if i == 0 else "link", "out": "pci" if i == n - 1 and not fan else "link"}
        t_base = 0
        if i > 0:
            io.update(src=route[i - 1], link=hop(chips, links, route[i - 1], route[i])[1], recv=recv)
            t_base = max(0, recv[0] - 100)
        if i < n - 1:
            io.update(dst=route[i + 1], dlink=hop(chips, links, route[i], route[i + 1])[0])
        elif fan:
            io.update(dsts=[(c, hop(chips, links, route[i], c)[0]) for c in fan])
        info = {}
        text, case = layer(p, seed + i, T, chunks, slot, name=f"chain{i}of{n}", io=io, t_base=t_base, x=x, info=info,
                          P=None if params is None else params[i], pools=None if pools is None else pools[i])
        out.append((text, case))
        infos.append(info)
        x = info["out"]
        recv = [c + LINK_RX for c in info.get("sends", [])]
    return out, infos


ON_DEMAND = [
    ("ship/ts_layer2x2", layer, PROFILES, {"T": 2, "chunks": 2}),
    ("ship/ts_layer6x8", layer, PROFILES, {"T": 6, "chunks": 8}),
    ("ship/ts_stage2x2", layer, PROFILES, {"T": 2, "chunks": 2, "name": "stage2x2", "pci": True}),
]

def pci_echo(p, seed=22, tiles=2):
    """The host path of a stage on its own: the host pushes the four byte planes of `tiles` f32 tiles, PciReceive and Write put them into east quad
    `X_QUAD`, and PciSend gives them back (Read and PciSend, one vector per host.cycles_per_vector cycles); the vectors the host gets are the ones it
    sent. The case checks the harness (`H dma`, `Y` lines) and the two instructions on every runner that takes host vectors."""
    k = Kit(p, "ship/ts_pci_echo", f"{4 * tiles} vectors pushed by the host, PciReceive, Write, Read, PciSend", seed, module="ship_layer", tags=TAGS + ("host",),
            source="numpy-twin")
    f = Flow(k)
    xs = [(k.rng.standard_normal(TILE) * 3).astype(np.float32) for _ in range(tiles)]
    vecs = pci_in(k, f, p, tiles, xs)
    host_out(k, f, p, vecs, [pad_lanes(x, p.vlanes) for x in xs])
    text, case = k.done("pci_echo")
    case["host"] = {"dma": [[0, h] for h in k.host_in]}
    case["expect"]["egress"] = k.host_out
    return text, case


CASES = [("ship/ts_pci_echo", pci_echo, PROFILES, {})]
