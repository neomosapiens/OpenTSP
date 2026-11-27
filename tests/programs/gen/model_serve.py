"""Reference programs for the serving embedding and output head.

The checked-in cases reduce the vocabulary and table to fit routine simulator tests. The on-demand full-size
cases run on the ISS. Expected values come from the TinyStories numerical reference.
"""

from opentsp_vllm.models.tinystories.embedding import EmbedSizes, tables
from types import SimpleNamespace

import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_tools.programs import numerics as nx
from opentsp_tools import patch as tpatch
from opentsp_tools.programs import ref
from opentsp_tools.programs.kernels import embed as k_embed, lm_head as k_head
from opentsp_tools.programs.kernels.flow import Flow
from opentsp_tools.pager import CycleView

PROFILES = ["ref"]
GENERIC = ["test/sl8", "test/sl32"]          # the packing and the head installs follow the lanes and the MXM rows of the profile
HIDDEN = 64


def showcase():
    """Load the TinyStories layout, numerical references and code emitter on demand."""
    from opentsp_vllm.models.tinystories import emit_block
    from opentsp_vllm.models.tinystories import reference as reference_model
    from opentsp_vllm.models.tinystories import paged_reference as reference_paged
    from opentsp_vllm.models.tinystories import layout
    return layout, reference_model, reference_paged, emit_block


# f16 bit patterns of the table rows that the special cases use: +-0, the smallest and largest subnormals, the smallest
# normal, the largest finite, +-inf, quiet and signalling NaNs with payloads, +-1 and an ordinary value.
SPECIAL_BITS = (0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01, 0x7D55,
                0xFD55, 0x3C00, 0xBC00, 0x3555)


def special_row(j):
    """A 64-lane f16 row made of the special bit patterns, rotated by j."""
    bits = np.resize(np.roll(np.array(SPECIAL_BITS, np.uint16), j), HIDDEN).astype(np.uint16)
    return bits.view(np.float16)


def make_weights(vocab, seed, special_tokens=(), special_positions=(), positions=128):
    """The two tables of a model: `wte` (vocab x 64 f16, N(0, 0.5^2)) and `wpe` (positions x 64, f16 values; the chip
    keeps them as f32), with special rows at the given tokens and positions; in the namespace bit-exact reference's embedding twins read."""
    rng = np.random.default_rng(seed)
    wte = (rng.standard_normal((vocab, HIDDEN)) * 0.5).astype(np.float16)
    wpe = (rng.standard_normal((positions, HIDDEN)) * 0.1).astype(np.float16)
    for j, t in enumerate(special_tokens):
        wte[t] = special_row(j)
    for j, t in enumerate(special_positions):
        wpe[t] = special_row(j + 3)
    return SimpleNamespace(cfg=SimpleNamespace(hidden=HIDDEN, vocab=vocab), wte=wte, wpe=wpe)


def pad(vec, vlanes=ref.LANES):
    out = np.zeros(vlanes, np.asarray(vec).dtype)
    out[:len(vec)] = vec
    return out


# -- embed -----------------------------------------------------------------------------------------------------------------


FULL_SIZES = EmbedSizes([4096, 4096, 1860], 50257)
REDUCED_SIZES = EmbedSizes([8, 8, 3], 93)

# (token, position): the part boundaries (the last row of a part and the first of the next), the first and last token,
# the rows of the special patterns and the first and last position.
FULL_LOOKUPS = ((0, 0), (4, 127), (20479, 5), (20480, 6), (40959, 64), (40963, 63), (50256, 127), (31337, 1))
REDUCED_LOOKUPS = ((0, 0), (39, 127), (40, 3), (79, 64), (83, 63), (92, 1), (17, 2), (61, 126))
FULL_SPECIAL_TOKENS = (0, 20479, 40963, 31337)
REDUCED_SPECIAL_TOKENS = (0, 61, 83, 17)
SPECIAL_POSITIONS = (0, 127, 63, 1, 126)


def packed_tables(layout, sizes, wte, plan, vlanes=ref.LANES):
    """{extern name: (slice, addr, words)} of the embedding parts: byte plane b of packed row w of part q on slice
    `EMB_PAIRS[q][b]`, the zero word `ez` of each plane's slice; `words` is the list of vlanes-byte words."""
    rows = layout._packed_lanes(wte, sizes.rows, sizes.pack, vlanes)
    out, off = {}, 0
    for q, n in enumerate(sizes.part_rows):
        u8 = rows[off:off + n].view(np.uint8).reshape(n, vlanes, 2)
        for b, s in enumerate(layout.EMB_PAIRS[q]):
            out[f"em{s}"] = (s, plan.by_name[f"em{s}"].addr, [bytes(np.ascontiguousarray(u8[w, :, b])) for w in range(n)])
            out[f"ez{s}"] = (s, plan.by_name[f"ez{s}"].addr, [bytes(vlanes)])
        off += n
    return out


def build_embed(p, name, doc, sizes, lookups, special_tokens, seed=1, tags=("sim", "ts-kernel")):
    """The embed kernel on the serve layout's table: one lookup per (token, position) of `lookups`, 5 cycles apart, each
    result (f32 hidden vector) stored to the activation quad. Returns the Kit (the case is `finish_with_sites(kit)`) and
    the namespace of the weights."""
    layout, exact_ref, g0p, _ = showcase()
    plan = layout.plan(layout.load_profile(p.name), mode="serve")
    w = make_weights(sizes.vocab, seed, special_tokens, SPECIAL_POSITIONS, sizes.positions)
    k = Kit(p, name, doc, seed, module="model_serve", tags=tags, source="numpy-twin")
    for ename, (s, addr, words) in packed_tables(layout, sizes, w.wte, plan, p.vlanes).items():
        k.b.prog.extern(ename, s, addr, len(words))
        k.weights[ename] = ("raw", b"".join(words))
    wpe = [layout.f32_planes(w.wpe[pos], p.vlanes) for pos in range(sizes.positions)]
    for j, s in enumerate(layout.WPE_SLICES):
        k.b.prog.extern(f"wp{s}", s, plan.by_name[f"wp{s}"].addr, sizes.positions)
        k.weights[f"wp{s}"] = ("raw", b"".join(v[j] for v in wpe))
    tb = tables(layout, plan, sizes)
    f = Flow(k)
    t_ready = k_embed.load_maps(k, 40, tb)
    for i, (token, pos) in enumerate(lookups):
        parts_addr, wpe_addrs, map_id = lookup_values(tb, sizes, token, pos)
        v = k_embed.embed(f, tb, f"e{i}", parts_addr, wpe_addrs, map_id, t_ready + 5 * i, k_embed.STAGE_ADDR + i)
        with np.errstate(invalid="ignore"):          # a signalling NaN row is cast to a quiet one
            want = exact_ref.embed(w, token, pos)
            if sizes is FULL_SIZES:
                assert want.tobytes() == g0p.embed_chip(w, token, pos).tobytes()
        f.store(v, 4, expect=pad(want, p.vlanes), hemi="E")
    return k, w




def lookup_values(tb, sizes, token, pos):
    """(the three parts' read addresses, the four `wpe` plane addresses, the map id) of one lookup : the part
    that holds the token reads its row, the others their zero word."""
    part, local, seg = sizes.locate(token)
    return ([pt.addr + local if q == part else pt.zero for q, pt in enumerate(tb.parts)],
            [a + pos for a in tb.wpe_addrs], k_embed.FIRST_MAP + seg)


def site_values(tb, sizes, lookups):
    """{site name: value} that set lookup i of a program built by `build_embed` to `lookups[i]`: the host patcher's input."""
    out = {}
    for i, (token, pos) in enumerate(lookups):
        parts_addr, wpe_addrs, map_id = lookup_values(tb, sizes, token, pos)
        for q, a in enumerate(parts_addr):
            for pl in range(2):
                out[f"e{i}.p{q}{'lh'[pl]}"] = a
        for j, a in enumerate(wpe_addrs):
            out[f"e{i}.wpe{j}"] = a
        for pl in range(2):
            out[f"e{i}.map{pl}"] = map_id
    return out


def finish_with_sites(k, fn):
    """`k.done(fn)` with the located patch sites in the case record."""
    view = CycleView.of(k.b.prog)
    text, case = k.done(fn)
    case["sites"] = tpatch.locate(view, k.p).records()
    return text, case


def pack_of(p):
    """The vocabulary rows per packed word of the serving layout on profile `p` (5 on `ref`)."""
    layout, *_ = showcase()
    lp = layout.load_profile(p.name)
    return layout.Shape.of(lp, layout.serve_model(lp)).pack


def reduced_embed(pack):
    """(sizes, lookups, special tokens) of the reduced embedding case at `pack` vocabulary rows per word: three parts of 8, 8 and 3 packed
    rows, the lookups on the part boundaries, the first and the last token (the constants above for 5)."""
    if pack == 5:
        return REDUCED_SIZES, REDUCED_LOOKUPS, REDUCED_SPECIAL_TOKENS
    s = lambda n: min(n, pack - 1)          # noqa: E731
    sizes = EmbedSizes([8, 8, 3], 19 * pack - 1, pack=pack)
    lookups = ((0, 0), (8 * pack - 1, 127), (8 * pack, 3), (16 * pack - 1, 64), (16 * pack + s(3), 63), (19 * pack - 2, 1),
               (3 * pack + s(2), 2), (12 * pack + s(1), 126))
    return sizes, lookups, (0, 12 * pack + s(1), 16 * pack + s(3), 3 * pack + s(2))


def embed(p, seed=1):
    sizes, lookups, special = reduced_embed(pack_of(p))
    k, _ = build_embed(p, "model/ts_embed", "token embedding on chip, reduced table (3 parts of 8, 8, 3 packed rows)",
                       sizes, lookups, special, seed)
    return finish_with_sites(k, "embed")


def embed_full(p, seed=1):
    k, _ = build_embed(p, "model/ts_embed_full", "token embedding on chip, full table (50 257 tokens, 3 parts)",
                       FULL_SIZES, FULL_LOOKUPS, FULL_SPECIAL_TOKENS, seed, tags=("iss", "ts-serve-full"))
    return finish_with_sites(k, "embed_full")


# -- lm_head ---------------------------------------------------------------------------------------------------------------
FINITE_BITS = tuple(b for b in SPECIAL_BITS if (b & 0x7C00) != 0x7C00)
REDUCED_HEAD = dict(vocab=2337, installs=3, rows=4)
FULL_HEAD = dict(vocab=50257, installs=63, rows=3)


def head_weights(vocab, seed, nonfinite=False, pack=5):
    """`wte` for the head cases: N(0, 0.5^2) with rows of the finite special patterns (an infinite or NaN weight row turns the
    other four vocabulary entries that share its lanes into NaN, as on the chip) and three packed rows for the
    sign of an exact zero sum: tokens 15..19 all -0.0, tokens 20..24 the same but token 23 is +0.0, tokens 25..29
    -0.5 with token 29 -0.0. Against a +0 hidden row the first and the third give -0 for all five entries (every one of the
    320 products is -0), the second +0 (one product is +0). With `nonfinite` (the case `ts_lm_head_nonfinite`) four
    more rows carry non-finite weights: +inf in token 31, -inf in token 41, a NaN in token 52 and, in the packed row of
    tokens 60..64 (all -0.0), +inf in token 62."""
    rng = np.random.default_rng(seed)
    wte = (rng.standard_normal((vocab, HIDDEN)) * 0.5).astype(np.float16)
    for j, t in enumerate(t for t in (0, 1, 2, 160, 799, 800, vocab - 1) if t < vocab):
        wte[t] = np.resize(np.roll(np.array(FINITE_BITS, np.uint16), j), HIDDEN).astype(np.uint16).view(np.float16)
    b = 3 * pack                     # three rows of `pack` tokens (15..29 on `ref`): the signs of an exact zero sum
    wte[b:b + 2 * pack] = np.float16(-0.0)
    wte[b + pack + pack - 2] = np.float16(0.0)
    wte[b + 2 * pack:b + 3 * pack - 1] = np.float16(-0.5)
    wte[b + 3 * pack - 1] = np.float16(-0.0)
    if nonfinite:
        wte[31, 5], wte[41, 7], wte[52, 9] = np.float16(np.inf), np.float16(-np.inf), np.float16(np.nan)
        wte[12 * pack:13 * pack] = np.float16(-0.0)
        wte[12 * pack + min(2, pack - 1), 3] = np.float16(np.inf)
    return SimpleNamespace(cfg=SimpleNamespace(hidden=HIDDEN, vocab=vocab), wte=wte)


def hidden_rows(rows, seed):
    """The final-LayerNorm outputs: f32, 64 lanes, `rows` of them (1 or more; the rows below exist when `rows` reaches them).
    Row 0 overflows f16 (70000, 65520 give +-inf: the logits of the whole row are infinite or NaN); row 1 is finite and
    carries the rounding edges (lanes 6 to 8 are observed by the logits: the round-up value and the two ties;
    65519 rounds to the largest finite and 65520 does not: the overflow row 0 and this row sit on either side); row 2 is +0
    in every lane (the -0 corner); row 3 is +0 except three lanes at the underflow and subnormal rounding edges, so
    that their tiny products are the whole sum and the edge is visible; the others are ordinary."""
    rng = np.random.default_rng(seed + 100)
    h = (rng.standard_normal((rows, HIDDEN)) * 2).astype(np.float32)
    if rows > 0:
        h[0, :6] = (-0.0, 1e-8, 70000.0, -65519.0, 65520.0, 0.333333343)[:min(6, HIDDEN)]
    if rows > 1:
        # 1 + 0.75 2^-10 rounds up, the ties 1 + 2^-11 and 1 + 1.5 2^-10 go to the even neighbour
        h[1, :9] = (-0.0, 1e-8, -65519.0, 65519.0, 0.333333343, 6e-8, 1.0 + 0.75 * 2.0 ** -10, 1.0 + 2.0 ** -11,
                    1.0 + 1.5 * 2.0 ** -10)
    if rows > 2:
        h[2] = 0.0
    if rows > 3:
        h[3] = 0.0
        # 1e-8 underflows to +0; 3e-8 is above half the smallest subnormal and rounds up to it; 503.75 2^-24 rounds up to 504 2^-24
        h[3, :3] = (1e-8, 3e-8, 503.75 * 2.0 ** -24)
    return h


def head_tensors(layout, wte, installs, vlanes, shape=None):
    """{slice: [word, ...]} of the packed head table as `layout.build_serve_images` lays it out: install i on the west plane
    when i is even (words 20 (i div 2) of the slices 0..15), else on the east plane (`shape`: the profile's `layout.Shape`, default
    that of `ref`: 160 packed rows of 5 vocabulary rows per install)."""
    pack, rows = (layout.PACK, layout.HEAD_CHUNK) if shape is None else (shape.pack, shape.head_rows)
    packed = layout._packed(wte, installs * rows, pack)
    per = {s: [] for s in layout.WEST_W + layout.EAST_W}
    for i in range(installs):
        group = layout.WEST_W if i % 2 == 0 else layout.EAST_W
        for j, words in enumerate(layout.matrix_words(packed[rows * i:rows * (i + 1)], vlanes)):
            per[group[j]] += words
    return per


def head_logits_of_f16(layout, w, h16, installs, vlanes, shape=None):
    """Logits in vocabulary order of the f16 hidden vector `h16` through the chip's structure (the packed rows against the
    segment vectors, one dot over the unit's `vlanes` lanes each)."""
    pack, rows = (layout.PACK, layout.HEAD_CHUNK) if shape is None else (shape.pack, shape.head_rows)
    packed = layout._packed(w.wte, installs * rows, pack)
    out = np.zeros(len(packed) * pack, np.float32)
    for kseg in range(pack):
        hk = np.zeros(pack * HIDDEN, np.float16)
        hk[HIDDEN * kseg:HIDDEN * (kseg + 1)] = h16
        out[kseg::pack] = nx.dot_f16(packed, hk, vlanes)
    return out[:w.cfg.vocab]


def head_expected(layout, w, h, installs, vlanes, shape=None):
    """{(row, install, segment): f32 vector of vlanes lanes}: the chip's structure (the install's packed rows of `pack` segments of
    64 lanes against the segment vector h_k, one dot over the unit's lanes: 160 rows, 5 segments and 320 lanes on `ref`), asserted
    equal to bit-exact reference's `head_logits_chip` on every vocabulary entry."""
    _, _, g0p, _ = showcase()
    pack, rows = (layout.PACK, layout.HEAD_CHUNK) if shape is None else (shape.pack, shape.head_rows)
    packed = layout._packed(w.wte, installs * rows, pack)
    out = {}
    for r in range(len(h)):
        h16 = ref.to_f16(h[r])
        reference_logits = g0p.head_logits_chip(w, h[r], vlanes, pack)
        for i in range(installs):
            for kseg in range(pack):
                hk = np.zeros(pack * HIDDEN, np.float16)
                hk[HIDDEN * kseg:HIDDEN * (kseg + 1)] = h16
                vec = nx.dot_f16(packed[rows * i:rows * (i + 1)], hk, vlanes)
                tok = pack * (rows * i + np.arange(rows)) + kseg
                live = tok < w.cfg.vocab
                assert vec[live].tobytes() == reference_logits[tok[live]].tobytes(), (r, i, kseg)
                out[(r, i, kseg)] = pad(vec, vlanes)
    return out


def build_lm_head(p, name, doc, vocab, installs, rows, seed=1, tags=("sim", "ts-kernel"), paged=False, nonfinite=False,
                  split=False):
    """`split`: the east plane's installs (odd i) write their results to the quad `lg2` on slices 76..79, which is
    what lets four logit rows page; the words are those of the single quad."""
    layout, _, _, emit_block = showcase()
    lp = layout.load_profile(p.name)
    plan = layout.plan(lp, mode="serve")
    shape = layout.Shape.of(lp, layout.serve_model(lp))
    w = head_weights(vocab, seed, nonfinite, shape.pack)
    h = hidden_rows(rows, seed)
    kit = (type("PagedKit", (emit_block.BlockKit,), {"case_tags": list(tags)}) if paged else Kit)
    k = kit(p, name, doc, seed, module="model_serve", source="numpy-twin", **({} if paged else {"tags": tags}))
    for s, words in head_tensors(layout, w.wte, installs, p.vlanes, shape).items():
        if words:
            k.b.prog.extern(f"hd{s}", s, plan.by_name[f"hd{s}"].addr, len(words))
            k.weights[f"hd{s}"] = ("raw", b"".join(words))
    f = Flow(k)
    hs = [f.const(f"h{r}", pad(h[r], p.vlanes), "W", 5) for r in range(rows)]
    head = k_head.Head(plan.by_name["hd0"].addr, plan.by_name["hd44"].addr, installs, layout.LOGIT_SLICES,
                       plan.by_name["lg"].addr, *((layout.LOGIT_SLICES_EAST, plan.by_name["lg2"].addr) if split else ()),
                       segments=shape.pack, nlw=shape.nlw_head, stride=shape.logit_stride)
    k_head.lm_head(f, hs, head)
    for (r, i, kseg), vec in head_expected(layout, w, h, installs, p.vlanes, shape).items():
        slices, base = head.results(2 * (i % 2))
        addr = base + head.stride * r + shape.pack * i + kseg
        for s, plane in zip(slices, data.to_planes(vec)):
            k.expect(s, addr, plane)
    return k


def reduced_head(p):
    """The reduced head case of profile `p`: three installs and the vocabulary that fills them but 63 packed entries (2 337 tokens on
    `ref`: 3 installs of 160 rows of 5)."""
    layout, *_ = showcase()
    lp = layout.load_profile(p.name)
    shape = layout.Shape.of(lp, layout.serve_model(lp))
    if (shape.pack, shape.head_rows) == (layout.PACK, layout.HEAD_CHUNK):
        return REDUCED_HEAD
    return dict(vocab=3 * shape.head_rows * shape.pack - 63, installs=3, rows=4)


def tokens_of(p):
    """The vocabulary of the reduced head case, as the documents write it (2 337 on `ref`)."""
    return f"{reduced_head(p)['vocab']:,}".replace(",", " ")


def lm_head(p, seed=1):
    k = build_lm_head(p, "model/ts_lm_head", f"packed LM head, reduced (3 installs, 4 logit rows, {tokens_of(p)} tokens)", seed=seed,
                      **reduced_head(p))
    return k.done("lm_head")


def lm_head_split(p, seed=1):
    k = build_lm_head(p, "model/ts_lm_head_split",
                      f"packed LM head, reduced, the east plane's results on a second logit quad (3 installs, 4 logit rows, {tokens_of(p)} tokens)",
                      seed=seed, split=True, **reduced_head(p))
    return k.done("lm_head_split")


def lm_head_nonfinite(p, seed=1):
    k = build_lm_head(p, "model/ts_lm_head_nonfinite",
                      f"packed LM head, reduced, with infinite and NaN weights (3 installs, 3 logit rows, {tokens_of(p)} tokens)",
                      seed=seed, nonfinite=True, **dict(reduced_head(p), rows=3))
    return k.done("lm_head_nonfinite")


def lm_head_full(p, seed=1):
    k = build_lm_head(p, "model/ts_lm_head_full", "packed LM head, full (63 installs, 3 logit rows, 50 257 tokens)",
                      seed=seed, tags=("iss", "ts-serve-full"), paged=True, **FULL_HEAD)
    return k.done("lm_head_full")


def lm_head_full4(p, seed=1):
    k = build_lm_head(p, "model/ts_lm_head_full4",
                      "packed LM head, full (63 installs, 4 logit rows over two logit quads, 50 257 tokens)",
                      seed=seed, tags=("iss", "ts-serve-full"), paged=True, split=True, **dict(FULL_HEAD, rows=4))
    return k.done("lm_head_full4")


ON_DEMAND = [
    ("model/ts_embed_full", embed_full, PROFILES, {}),
    ("model/ts_lm_head_full", lm_head_full, PROFILES, {}),
    ("model/ts_lm_head_full4", lm_head_full4, PROFILES, {}),
]

CASES = [
    ("model/ts_embed", embed, PROFILES + GENERIC, {}),
    ("model/ts_lm_head", lm_head, PROFILES + GENERIC, {}),
    ("model/ts_lm_head_split", lm_head_split, PROFILES, {}),
    ("model/ts_lm_head_nonfinite", lm_head_nonfinite, PROFILES, {}),
]
