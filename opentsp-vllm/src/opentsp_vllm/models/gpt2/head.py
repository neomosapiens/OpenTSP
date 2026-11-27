"""A slice of the tied LM head of GPT-2 small as one program (a proof of concept:  section 5, the head
stage): the final LayerNorm of the hidden state the host (or the last layer chip) gives, then the logits of `vtiles` tiles of 128 vocabulary rows, each the
matvec of the 128-row blocks against the 6 input tiles of the LayerNorm output (the layout of `ship_layer`: block (o, i) = 128 rows of 128 f16 features,
partial dot products rounded once to f32, added in ascending order, the zero bias last), every logits tile given to the host by PciSend (4 byte planes).
The twin is `ref_wide` at tile 128. A chip holds `capacity(p, T)` tiles in bank 0 of its weight slices (the same words per plane as a layer's weights), and with `two_banks` more in bank 1."""
import numpy as np

from opentsp_tools.programs import ref_wide as rw
from opentsp_tools.programs.kernels import reduce_bcast as k_rb, wide_ln as k_ln
from opentsp_tools.programs.kernels.flow import Flow
from opentsp_vllm.models.gpt2 import layer as sl
from opentsp_tools.programs.kernels import install as k_install
from opentsp_vllm.models.gpt2.attention import PagedKit, TILE
from opentsp_vllm.models.gpt2.wide import pad_lanes

EPS = sl.EPS


B1_BASE = 0x1100         # first word of the weights in bank 1 (the LayerNorm's scratch words are 0x1000..0x1050), when two banks are used
B1_END = 0x1c00          # the text pages lie above (`done(floor=...)`)


def capacity(p, T=6, two_banks=False):
    """Output tiles of 128 rows one chip's weight slices hold: T blocks per tile, half on each of the two planes, 16 words per block and slice; in bank 0 from
    `WBASE`, and with `two_banks` also from `B1_BASE` to `B1_END` in bank 1."""
    per = 16 * (T // 2)
    n = (1 << p.bank_bit) - sl.WBASE
    if two_banks:
        n += B1_END - B1_BASE
    return n // per


def bind_blocks2(k, p, W, T, two_banks):
    """`sl.bind_blocks` for one matrix; with `two_banks` the blocks that do not fit under the bank boundary go to a second extern per slice at `B1_BASE`
    (a block never straddles). Returns ({(o, i): (plane, address, text)}, [(name, plane, slice, base, words)])."""
    words = {pl: {x: [[] for _ in range(16)] for x in "ab"} for pl in sl.PLANES}
    ptr = {pl: [sl.WBASE, B1_BASE] for pl in sl.PLANES}
    out = {}
    top = (1 << p.bank_bit)
    for o in range(W.shape[0] // TILE):
        for i in range(W.shape[1] // TILE):
            pl = sl.PLANES[i % 2]
            part = 0 if ptr[pl][0] + 16 <= top else 1
            if part == 1 and not two_banks:
                raise ValueError("the weights do not fit in bank 0")
            addr = ptr[pl][part]
            if part == 1 and addr + 16 > B1_END:
                raise ValueError("the weights do not fit")
            for j, ws in enumerate(k_install.weight_planes(W[o * TILE:(o + 1) * TILE, i * TILE:(i + 1) * TILE], p.vlanes)):
                words[pl]["ab"[part]][j] += ws
            base = sl.WBASE if part == 0 else B1_BASE
            out[o, i] = (pl, addr, (lambda sym, off: lambda j, n: f"{sym}{j}+{off + n:#x}")(f"l{'wx'[part]}{pl}s", addr - base))
            ptr[pl][part] += 16
    for pl in sl.PLANES:
        base_q = k_install.plane_queues(pl)[0]
        for part, (x, at) in enumerate((("a", sl.WBASE), ("b", B1_BASE))):
            for j in range(16):
                if words[pl][x][j]:
                    name = f"l{'wx'[part]}{pl}s{j}"
                    k.b.prog.extern(name, base_q + j, at, len(words[pl][x][j]))
                    k.weights[name] = ("raw", b"".join(words[pl][x][j]))
    return out


def twin(x, g, b, W, tile=TILE):
    xn = rw.layernorm_wide(x, g, b, EPS, tile=tile)
    return xn, rw.matvec_tiled(W, xn, np.zeros(len(W), np.float32), tile=tile)


def head(p, seed=41, T=6, vtiles=2, name=None, x=None, g=None, b=None, W=None, two_banks=False, io=None):
    """The LayerNorm and `vtiles` logits tiles of hidden width 128 T; the hidden state x arrives from the host (PciReceive) and the logits go back (PciSend).
    `W` (128 vtiles, 128 T) f16 is the slice of the head table (default: random), `g`, `b` the LayerNorm parameters. `io` = {"src": chip, "link": its Receive link,
    "recv": the Receive cycle of each of the 4 T planes} takes the hidden state from the last layer chip's Sends (the link wiring of `ship_layer.chain` with
    `fan`) instead of the host's DMA; the head's own work then starts at `t_base` = recv[0] - 100."""
    H = TILE * T
    if vtiles > capacity(p, T, two_banks):
        raise ValueError(f"{vtiles} tiles exceed the {capacity(p, T, two_banks)} one chip's weight slices hold")
    k = PagedKit(p, f"ship/ts_{name or f'head{T}x{vtiles}'}", f"final LayerNorm and {vtiles} logits tiles of the tied head, hidden {H}; 128-lane tiles", seed,
                 module="ship_head", tags=sl.TAGS + ("host",), source="numpy-twin")
    rng = k.rng
    x = rng.standard_normal(H).astype(np.float32) if x is None else x
    g = (1 + rng.standard_normal(H) * 0.1).astype(np.float32) if g is None else g
    b = (rng.standard_normal(H) * 0.1).astype(np.float32) if b is None else b
    W = (rng.standard_normal((TILE * vtiles, H)) * 0.05).astype(np.float16) if W is None else W
    xn_want, y_want = twin(x, g, b, W)
    hb = bind_blocks2(k, p, W, T, two_banks)
    blocks = {("h", o, i): v for (o, i), v in hb.items()}
    f = Flow(k)
    dist_id = k_rb.dist_map(3)
    t_base = max(0, io["recv"][0] - 100) if io else 0
    maps = (k_rb.load_maps(k, 60 + t_base, 3), k_rb.load_distribute_map(k, 80 + t_base, dist_id), 3)
    inv_n, eps = f.splat("invn", np.float32(1.0 / H), "W", 4), f.splat("eps", EPS, "W", 4)
    wq = [5, 6, 7, 8, 9, 10]
    cn = {"n": 0}

    def vecs(name_, arr):
        out = []
        for t, v in enumerate(rw.split(arr, TILE)):
            cn["n"] += 1
            out.append(f.const(f"{name_}{t}", pad_lanes(v, p.vlanes), "W", wq[cn["n"] % len(wq)]))
        return out

    xv = sl.link_in(k, f, p, rw.split(x, TILE), io["recv"], io["src"], io["link"]) if io else sl.pci_in(k, f, p, T, rw.split(x, TILE), 100)
    gv, bv = vecs("g", g), vecs("b", b)
    zb = vecs("zb", np.zeros(TILE, np.float32))[0]
    state = {"t": max([maps[1]] + [v.ready for v in xv]) + sl.GAP, "kq": 0}

    def after(*vs_):
        state["t"] = max([state["t"], k.last] + [v.ready for v in vs_ if v is not None]) + sl.GAP
        return state["t"]

    keep_q = [("E", 8), ("E", 9), ("E", 10)]

    def keep(v, to=None):
        hemi, quad = to or keep_q[state["kq"] % 3]
        state["kq"] += 1
        out = f.store(v, quad, hemi=hemi)
        after(out)
        return out

    f.low = {"E": 0, "W": 0}
    ys = k_ln.layernorm_wide(f, xv, gv, bv, inv_n, eps, maps[:2], 0x10, (1 << p.bank_bit) + 0x40, quads=(0, 1, 1 + T), rounds=3, at=after(*xv))
    xn = [keep(y) for y in ys]
    outs = []
    for o in range(vtiles):
        f.low = {"E": 16, "W": 16}
        y = sl.matvec_block(f, xn, zb, [blocks["h", o, i] for i in range(T)], after(*xn), to="E")
        outs.append(keep(y))
    out_tiles = [pad_lanes(v, p.vlanes) for v in rw.split(y_want, TILE)]
    sl.host_out(k, f, p, outs, out_tiles)
    text, case = k.done(f"head{T}x{vtiles}", floor=B1_END if two_banks else 6144)
    if not io:
        case["host"] = {"dma": [[0, h] for h in k.host_in]}
    case["expect"]["egress"] = k.host_out
    return text, case, dict(x=x, xn=xn_want, logits=y_want)


def head_case(p, **kw):
    return head(p, **kw)[:2]


CASES = [("ship/ts_head6x2", head_case, sl.PROFILES, {"T": 6, "vtiles": 2})]
ON_DEMAND = [("ship/ts_head6x80b", head_case, sl.PROFILES, {"T": 6, "vtiles": 80, "two_banks": True}),
             ("ship/ts_head6x131b", head_case, sl.PROFILES, {"T": 6, "vtiles": 131, "two_banks": True})]
