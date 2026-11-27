"""Memory layout planner and SRAM images for the TinyStories showcase on the `ref` profile .

`Planner` places named regions (a region = the same word range in each of a set of slices) into the two banks of
each slice, honouring the bank rule: a read-type and a write-type reference of one thread in one cycle must address
different banks (microarchitecture.md 4, OTS-E-MEM-001). The rule is expressed as *conflicts* between regions that
are read and written concurrently in a slice; the planner two-colours the conflict graph (a bank is a colour) and
fails when it is not bipartite. Regions never straddle the bank boundary (an AGT loop must not cross banks).

`plan_tinystories` builds the plan for the model: per layer attention and FFN weights as f16 MXM install images,
the K cache (install layout), the V cache (f32 planes), f32 constants, host inputs, the activation arena, and the
auxiliary vectors; `build_images` fills the weight tensors (`.safetensors` payload).

Word format (assembly.md section 4): a word is vlanes bytes, lane 0 first. An element of w bytes is w byte planes in
w consecutive words (stream byte planes), plane b holding byte b of every lane.
"""
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths
from opentsp_tools.programs import geometry as geo
from opentsp_tools.programs import needs
from opentsp_tools.programs import ref
from opentsp_tools.programs.kernels import embed as k_embed, lm_head as k_head
from opentsp_vllm.models.tinystories.config import ModelConfig, SERVE_SMALL, TINYSTORIES, TINYSTORIES_L2

PROFILE_FILE = os.path.join(_paths.SPEC, "config", "ref.json")


@dataclass
class Profile:
    name: str
    slices: int
    words_per_slice: int
    bank_bit: int
    vlanes: int
    mem_per_hemisphere: int
    mxm_rows: int

    @property
    def bank_words(self):
        return 1 << self.bank_bit

    def bank_of(self, addr):
        return (addr >> self.bank_bit) & 1


def load_profile(name="ref") -> Profile:
    path = os.path.join(_paths.SPEC, "config", name + ".json")
    c = json.load(open(path))
    return Profile(name, c["sram"]["slices"], c["sram"]["words_per_slice"], c["sram"]["bank_bit"],
                   16 * c["geometry"]["superlanes"], c["geometry"]["mem_slices_per_hemisphere"], c["mxm"]["rows"])


@dataclass(frozen=True)
class Shape:
    """The sizes of the serving programs that follow from the profile and the model (see `opentsp_tools.programs.geometry`): the lanes of a
    vector, the vocabulary rows of a packed table word, the f16 rows of one head install and the installs of the vocabulary,
    the context sizes the K install can hold. On `ref` they are the constants of this module (320, 5, 160, 63, (64, 128)); the packing
    stays at the five rows of 64 lanes the kernels were written for on wider vectors, and narrows to what a vector holds."""
    lanes: int
    pack: int
    head_rows: int
    head_installs: int
    ctxs: tuple

    @classmethod
    def of(cls, profile, cfg):
        g = geo.Geometry.of(profile)
        pack = min(PACK, g.pack)
        head_rows = min(HEAD_CHUNK, g.install_rows)
        installs = -(-(-(-cfg.vocab // pack)) // head_rows) if pack else 0
        ctxs = tuple(c for c in MASK_CS if c <= min(g.install_rows, g.lanes))
        return cls(g.lanes, pack, head_rows, installs, ctxs)

    @property
    def nlw_head(self):
        return self.head_rows // 8

    @property
    def logit_stride(self):
        """Words of one logit row in the logits region: install i, segment k at word `pack i + k`."""
        return self.head_installs * self.pack


@dataclass
class Region:
    name: str
    kind: str                 # weight | const | input | kcache | vcache | act | aux | reserved
    slices: Tuple[int, ...]
    words: int                # words per slice
    rw: bool                  # written at run time (by the chip)
    exported: bool = False    # carried by the weight package (.safetensors)
    note: str = ""
    bank: Optional[int] = None
    addr: Optional[int] = None
    fixed: bool = False       # address given by the caller (words the kernels address directly)
    pin: bool = False         # bank given by the caller (a fixed region, or a forced bank)


class Planner:
    def __init__(self, profile: Profile):
        self.p = profile
        self.regions: List[Region] = []
        self.by_name = {}
        self.conflicts = []       # (a, b): read of one with a write of the other in one cycle

    def add(self, name, kind, slices, words, rw, exported=False, note="", addr=None, bank=None):
        """`addr`: a fixed region (words the kernels address directly): it keeps its address and bank, the others are
        placed around it. `bank`: the region goes to that bank whatever the conflicts say (the bank rule is still
        checked by `validate`)."""
        assert name not in self.by_name, name
        r = Region(name, kind, tuple(slices), words, rw, exported, note)
        if addr is not None:
            r.addr, r.bank, r.fixed, r.pin = addr, self.p.bank_of(addr), True, True
        elif bank is not None:
            r.bank, r.pin = bank, True
        self.regions.append(r)
        self.by_name[name] = r
        return r

    def conflict(self, a, b):
        self.conflicts.append((a, b))

    # -- bank colouring --------------------------------------------------------------------------------------
    def colour(self):
        """Two-colour the conflicts that matter (regions sharing a slice). Isolated read-only regions go to bank 0,
        isolated written regions to bank 1; a component is oriented to keep as many as possible there."""
        adj = {r.name: set() for r in self.regions}
        for a, b in self.conflicts:
            ra, rb = self.by_name[a], self.by_name[b]
            if set(ra.slices) & set(rb.slices) and not (ra.pin or rb.pin):
                adj[a].add(b)
                adj[b].add(a)
        col = {}
        for r in self.regions:
            if r.name in col:
                continue
            comp, stack = {r.name: 0}, [r.name]
            while stack:
                u = stack.pop()
                for v in sorted(adj[u]):
                    if v not in comp:
                        comp[v] = 1 - comp[u]
                        stack.append(v)
                    elif comp[v] == comp[u]:
                        raise ValueError(f"bank conflict graph is not bipartite: {u} and {v}")
            good = sum(1 for n, c in comp.items() if c == (1 if self.by_name[n].rw else 0))
            flip = good < len(comp) - good
            for n, c in comp.items():
                col[n] = c ^ int(flip)
        for r in self.regions:
            r.bank = r.bank if r.pin else col[r.name]
        return col

    # -- placement -------------------------------------------------------------------------------------------
    def place(self):
        self.colour()
        ptr = {}
        bw = self.p.bank_words
        fixed = [r for r in self.regions if r.fixed]
        for r in self.regions:
            for s in r.slices:
                assert 0 <= s < self.p.slices, (r.name, s)
            if r.fixed:
                continue
            base = r.bank * bw
            addr = max(ptr.get((s, r.bank), base) for s in r.slices)
            moved = True
            while moved:                      # past every fixed region it would cover
                moved = False
                for f in fixed:
                    if set(f.slices) & set(r.slices) and addr < f.addr + f.words and f.addr < addr + r.words:
                        addr, moved = f.addr + f.words, True
            if addr + r.words > base + bw:
                raise ValueError(f"region {r.name} ({r.kind}) does not fit bank {r.bank}: it needs {r.words} words in slice(s) "
                                 f"{list(r.slices)[:4]}, the bank of the profile has {bw} words ({self.p.words_per_slice} words per slice) "
                                 f"and {addr - base} are taken before it, so the bank would need {addr + r.words - base} words")
            r.addr = addr
            for s in r.slices:
                ptr[(s, r.bank)] = addr + r.words
        self.validate()

    def validate(self):
        """No overlap, inside one bank, inside the slice; every conflict pair in different banks where slices meet."""
        occ = {}
        for r in self.regions:
            assert r.addr is not None, r.name
            assert self.p.bank_of(r.addr) == r.bank == self.p.bank_of(r.addr + r.words - 1), r.name
            assert r.addr + r.words <= self.p.words_per_slice, r.name
            for s in r.slices:
                for other in occ.get(s, []):
                    assert r.addr + r.words <= other.addr or other.addr + other.words <= r.addr, (r.name, other.name, s)
                occ.setdefault(s, []).append(r)
        for a, b in self.conflicts:
            ra, rb = self.by_name[a], self.by_name[b]
            if set(ra.slices) & set(rb.slices):
                assert ra.bank != rb.bank, (a, b)

    # -- report ----------------------------------------------------------------------------------------------
    def usage(self):
        """Words used per slice, per bank, by kind; and totals."""
        per = {}
        for r in self.regions:
            for s in r.slices:
                d = per.setdefault(s, {"bank0": 0, "bank1": 0, "kinds": {}})
                d["bank%d" % r.bank] += r.words
                d["kinds"][r.kind] = d["kinds"].get(r.kind, 0) + r.words
        total = sum(d["bank0"] + d["bank1"] for d in per.values())
        return {"slices_used": len(per), "words_used": total,
                "words_capacity": self.p.slices * self.p.words_per_slice,
                "max_words_in_a_slice": max(d["bank0"] + d["bank1"] for d in per.values()),
                "max_bank0": max(d["bank0"] for d in per.values()), "max_bank1": max(d["bank1"] for d in per.values()),
                "per_slice": {str(s): per[s] for s in sorted(per)}}


# -- the TinyStories plan ----------------------------------------------------------------------------------------
# Slice groups on `ref` (west MEM slices 0..43, east 44..87; quad q = slices 4q..4q+3 in each hemisphere).
WEST_W = tuple(range(0, 16))             # plane 0 (west MXM): quads 0..3
EAST_W = tuple(range(44, 60))            # plane 2 (east MXM): quads 0..3
CONST_QUAD = tuple(range(16, 20))        # west quad 4: f32 constants (planes 0..3), the host mask
VCACHE_QUAD = tuple(range(20, 24))       # west quad 5: V cache (f32 planes)
SCRATCH_QUAD = tuple(range(60, 64))      # east quad 4: the activation arena (f32 planes)
AUX = dict(onehot_hi=64, onehot_zero=65, headmask_hi=66, headmask_zero=67)   # east quad 5
MAPS_QUAD = tuple(range(68, 72))         # east quad 6: reserved for the SXM butterfly / broadcast maps
MAPS_WORDS = 16


def matrices(cfg: ModelConfig, hemisphere):
    """(name, rows, cols) of the f16 matrices a hemisphere's MXM plane installs per layer, in storage order."""
    H, half = cfg.hidden, cfg.half
    if hemisphere == "west":
        return [("wq", H, H), ("wv", H, H), ("wo", H, H), ("up0", half, H), ("dn0", H, half)]
    return [("wk", H, H), ("up1", half, H), ("dn1", H, half)]


def matrix_offset(cfg: ModelConfig, name):
    """Word, in the layer's `w<layer>s<slice>` tensors of its hemisphere, of the first LW of matrix `name` (`matrices` order:
    8 rows per word)."""
    for hemi in ("west", "east"):
        off = 0
        for n, rows, _ in matrices(cfg, hemi):
            if n == name:
                return off
            off += rows // 8
    raise KeyError(name)


def const_vectors(cfg: ModelConfig, serve=False, lanes=ref.LANES):
    """(name, lanes) of every f32 constant vector in the constants quad, in storage order; one word per plane. The serving
    layout (`serve`) holds the kernels' constants as splats of every lane, as the Flow's own `splat` makes them, and the zero
    vector the context chains start from."""
    out = []
    for l in range(cfg.layers):
        out += [(f"L{l}.ln1g", cfg.hidden), (f"L{l}.ln1b", cfg.hidden),
                (f"L{l}.bq", cfg.hidden), (f"L{l}.bk", cfg.hidden), (f"L{l}.bv", cfg.hidden), (f"L{l}.bo", cfg.hidden),
                (f"L{l}.ln2g", cfg.hidden), (f"L{l}.ln2b", cfg.hidden),
                (f"L{l}.up_b0", cfg.half), (f"L{l}.up_b1", cfg.half), (f"L{l}.dn_b", cfg.hidden)]
    out += [("lnf.g", cfg.hidden), ("lnf.b", cfg.hidden)]
    lanes = lanes if serve else cfg.half
    out += [(n, lanes) for n in ("k.gelu_c0", "k.gelu_c1", "k.one", "k.half", "k.eps", "k.inv_h")]
    if serve:
        out.append(("k.zero", lanes))
    return out


def scratch_vectors(cfg: ModelConfig):
    """(name, words per plane-slice) of the activation arena. Each is an f32 vector (one word in each of the
    four slices of the scratch quad); `u*`, `g*` and `d*` are the FFN halves."""
    names = ["x0", "x1", "xn", "q", "ctx", "ao", "h1", "xn2", "u0", "u1", "exact", "rounded", "d0", "d1", "s", "t", "hf"]
    names += [f"sc{h:02d}" for h in range(cfg.heads)] + [f"pr{h:02d}" for h in range(cfg.heads)]
    return names


def scratch_conflicts(cfg: ModelConfig):
    """Pairs read and written in one slice in consecutive pipeline stages of a layer (a producer's output is read
    by the next stage while that stage writes its own). The two residual buffers x0, x1 alternate by layer."""
    e = []
    for x in ("x0", "x1"):
        e += [(x, "xn"), (x, "h1")]
        e += [("h1", x), ("t", x)]
    e += [("xn", "q"), ("ctx", "ao"), ("ao", "h1"), ("h1", "xn2"), ("d0", "s"), ("d1", "s"), ("s", "t"),
          ("pr00", "ctx"), ("xn2", "u0"), ("xn2", "u1"), ("u0", "exact"), ("u1", "rounded"), ("exact", "d0"), ("rounded", "d1"),
          ("x0", "hf"), ("x1", "hf")]
    for h in range(cfg.heads):
        e += [("q", f"sc{h:02d}"), (f"sc{h:02d}", f"pr{h:02d}"), (f"pr{h:02d}", "ctx")]
    return e


# Words the reduce-and-broadcast kernel (`emit_block.Block.scratch`, `reduce_bcast.tree_round`) addresses directly. A use n
# (36 per layer, a0 = 0x400 + 8 n, largest 0x518) writes a0, a0 + 1 on the east slices 44..47 (bank 0), the same words plus
# 0x1000 there (bank 1) and, on west slices 0..3 in bank 1, a0 + 2, a0 + 3 and the output word 0x1400 + n (all of them in
# 0x1400..0x151b). The regions stay so a table never covers them (`test_layout.py` derives them from the emitter).
RB_EAST = tuple(range(44, 48))
RB_WEST = tuple(range(0, 4))
RB_BASE = 0x400
RB_WORDS = 0x120
RB_BANK1_WORDS = 0x120
KFIX_WORDS = 0x50                        # layer 0's weights end at 0x30; the kernels' fixed words are below 0x80
HEAD_CHUNK = 160                         # vocabulary rows one matvec of the output head covers (mxm.rows / 2 f16 rows)


def _add_reduce_scratch(pl):
    pl.add("rbs_e0", "reserved", RB_EAST, RB_WORDS, rw=True, addr=RB_BASE,
           note="reduce_bcast scratch (emit_block SCRATCH), bank 0")
    pl.add("rbs_e1", "reserved", RB_EAST, RB_WORDS, rw=True, addr=pl.p.bank_words + RB_BASE,
           note="reduce_bcast scratch, bank 1")
    pl.add("rbs_w1", "reserved", RB_WEST, RB_BANK1_WORDS, rw=True, addr=pl.p.bank_words + RB_BASE,
           note="reduce_bcast intermediates and outputs, bank 1 of west slices 0..3")


def _add_kernel_scratch(pl, pack=None):
    """The words `embed` and `lm_head` address directly, as fixed regions: the embedding's staging word of lookup i
    (slices 16, 17), the head's hidden-vector staging word of row r (the same slices) and its five segment vectors per row
    (slices 18, 19 and 70, 71). The addresses are the kernels' own constants, so the plan cannot drift from them."""
    pack = PACK if pack is None else pack
    pl.add("kfix_embed", "reserved", k_embed.STAGE, LANES_MAX, rw=True, addr=k_embed.STAGE_ADDR,
           note="embed staging: the permuted byte planes of lookup i wait at word i")
    pl.add("kfix_hid", "reserved", k_head.H_STAGE, LOGIT_ROWS, rw=True, addr=k_head.H_STAGE_ADDR,
           note="lm_head staging: the f16 hidden vector of row r")
    pl.add("kfix_head", "reserved", k_head.HK_WEST + k_head.HK_EAST, pack * LOGIT_ROWS, rw=True,
           addr=k_head.HK_ADDR, note="lm_head segment vectors: row r, segment k at word pack r + k")


def _add_layer_weights(pl, cfg):
    # weights: one tensor per (layer, slice) holding the slice's rows of every matrix of the layer, contiguous
    for hemi, group in (("west", WEST_W), ("east", EAST_W)):
        for l in range(cfg.layers):
            words = sum(rows // 8 for _, rows, _ in matrices(cfg, hemi))
            for s in group:
                pl.add(f"w{l}s{s:02d}", "weight", (s,), words, rw=False, exported=True,
                       note=f"layer {l} {hemi} plane: " + ",".join(n for n, _, _ in matrices(cfg, hemi)))
            if hemi == "west" and l == 0:
                pl.add("kfix", "reserved", group, KFIX_WORDS, rw=False,
                       note="words the kernels address directly (the SXM maps 0x60..0x6a on slices 2, 3: LayerNorm 0x60, 0x61, "
                            "Distribute 0x62, the embedding 0x63..0x67, the C = 128 softmax tree 0x68..0x6a; "
                            "zero word 0x70 on slices 6, 7): the weights of later layers start above them")


def plan_tinystories(cfg: ModelConfig, profile: Profile, head_chunks=0) -> Planner:
    """`head_chunks` > 0 adds the weights of that many 160-row chunks of the tied output head ( `matvec_f16`
    on the embedding rows): chunk c is on plane 2 * (c % 2) as the words 40 * (c // 2) .. of the `hd<slice>` tensors."""
    needs.SHOWCASE.check(profile)
    pl = Planner(profile)
    if head_chunks:
        _add_reduce_scratch(pl)
    _add_layer_weights(pl, cfg)
    for l in range(cfg.layers):
        pl.add(f"kc{l}", "kcache", EAST_W, cfg.slots // 8, rw=True, note=f"layer {l} K cache, install layout, f16")
        pl.add(f"vc{l}", "vcache", VCACHE_QUAD, cfg.slots, rw=True, note=f"layer {l} V cache, f32 planes, one word per slot")
    # constants: one tensor per slice of the quad, word i = plane (slice - base) of constant vector i
    nconst = len(const_vectors(cfg))
    for s in CONST_QUAD:
        pl.add(f"cs{s}", "const", (s,), nconst, rw=False, exported=True, note="f32 constants, one plane per slice")
    pl.add("mask", "input", CONST_QUAD, 1, rw=False, note="host-written f32 mask vector (0 or -inf per slot)")
    # auxiliary vectors: one-hot f16 vectors for the context expansion, head masks
    pl.add("ax64", "aux", (AUX["onehot_hi"],), cfg.slots, rw=False, exported=True, note="one-hot e_s, f16 high plane")
    pl.add("ax65", "aux", (AUX["onehot_zero"],), 1, rw=False, exported=True, note="f16 low plane of the one-hot vectors (zero)")
    pl.add("ax66", "aux", (AUX["headmask_hi"],), cfg.heads, rw=False, exported=True, note="head masks, f16 high plane")
    pl.add("ax67", "aux", (AUX["headmask_zero"],), 1, rw=False, exported=True, note="f16 low plane of the head masks (zero)")
    pl.add("maps", "reserved", MAPS_QUAD, MAPS_WORDS, rw=False, note="SXM butterfly and broadcast maps (TS-1)")
    for n in scratch_vectors(cfg):
        pl.add("a_" + n, "act", SCRATCH_QUAD, 1, rw=True, note="f32 activation vector")
    for a, b in scratch_conflicts(cfg):
        pl.conflict("a_" + a, "a_" + b)
    # the K cache is written (current slot) in slices that also stream weights: weights and caches use other banks
    for l in range(cfg.layers):
        for l2 in range(cfg.layers):
            for s in EAST_W:
                pl.conflict(f"kc{l}", f"w{l2}s{s:02d}")
    for group, plane_chunks in ((WEST_W, (head_chunks + 1) // 2), (EAST_W, head_chunks // 2)):
        for s in group:
            if plane_chunks:
                pl.add(f"hd{s}", "head", (s,), plane_chunks * HEAD_CHUNK // 8, rw=False,
                       note="output head rows (the tied embedding), host-bound per step window")
    pl.place()
    return pl


# -- the serving layout  ----------------------------------------------------------
class LayoutError(ValueError):
    pass


SERVE_LAYERS = (8, 2)                    # tinystories and the 2-layer test model tinystories-l2
SERVE_BLOCKS = 128                       # blocks 0..127 of the pools; block 0 is the zero block
SINK_BLOCK = 128                         # the block redirected writes go to (not in vLLM's range)
BLOCK_TOKENS = 16
CTX_MAX = 128                            # contexts C = 64 and 128
LANES_MAX = 8                            # token lanes of the largest bucket
LOGIT_ROWS = 4                           # logit rows of the largest bucket
HEAD_INSTALLS = 63                       # 63 installs of 160 packed rows 
NLW_HEAD = HEAD_CHUNK // 8               # words per slice of one install
PACK = 5                                 # vocabulary rows per packed row: lanes 64 k .. 64 k + 63 hold row 5 rho + k
EMB_PAIRS = ((36, 37), (38, 39), (40, 41))   # slice pair (low, high byte plane) of each embedding part
WPE_SLICES = tuple(range(40, 44))
WPE_POSITIONS = 128
MASK_SLICES = tuple(range(76, 80))
MASK_CS = (64, 128)
LOGIT_SLICES = tuple(range(72, 76))
LOGIT_SLICES_EAST = tuple(range(76, 80))   # shared with the mask tables (other words): the east plane's results
TEXT_MARGIN_PCT = 110                    # patched block references (about 10 %)

# Words of paged instruction text per MEM slice of the batched steps, from the emitter itself
# (`text_sizing.py` measures them: the footprint of the blocks the pager takes from the end of a roomy `tx` region):
# `TEXT_WORDS_ONE_LANE[layers]` for P(1, 1, 128), `TEXT_WORDS_T8[layers]` the largest of each slice over the 18 buckets of the
# grid; smaller buckets can be largest on individual slices. `test_step_buckets.py` re-derives the full grid.
# The `tx` regions have the grid maximum plus the margin.
TEXT_SLICES = tuple(range(88))               # the slices the roomy plan gives a `tx` region
TEXT_WORDS_ONE_LANE = {
    2: {
        0: 34, 1: 34, 2: 28, 3: 26, 4: 16, 5: 16, 6: 16, 7: 22, 8: 18, 9: 18, 10: 18, 11: 18, 14: 16, 15: 16, 16: 1,
        17: 1, 18: 8, 19: 8, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4, 29: 4, 30: 3, 31: 3,
        32: 1, 33: 1, 36: 4, 37: 4, 38: 4, 39: 4, 40: 5, 41: 5, 42: 4, 43: 4, 44: 30, 45: 32, 46: 32, 47: 32,
        48: 16, 49: 16, 50: 16, 51: 16, 52: 16, 53: 16, 54: 16, 55: 16, 56: 16, 57: 16, 58: 16, 59: 16, 60: 6,
        61: 6, 62: 6, 63: 6, 64: 6, 65: 6, 66: 6, 67: 6, 70: 6, 71: 6, 72: 6, 73: 6, 74: 6, 75: 6, 76: 6, 77: 6,
        78: 6, 79: 6, 80: 1, 81: 1, 82: 1, 83: 1, 84: 3, 85: 3, 86: 1, 87: 1},
    8: {
        0: 92, 1: 92, 2: 82, 3: 76, 4: 32, 5: 32, 6: 32, 7: 52, 8: 30, 9: 30, 10: 30, 11: 30, 12: 1, 13: 1, 14: 26,
        15: 26, 16: 14, 17: 14, 18: 22, 19: 22, 20: 8, 21: 8, 22: 7, 23: 7, 24: 7, 25: 7, 26: 7, 27: 7, 28: 9,
        29: 9, 30: 8, 31: 8, 32: 8, 33: 8, 34: 7, 35: 7, 36: 9, 37: 9, 38: 9, 39: 9, 40: 12, 41: 12, 42: 13, 43: 13,
        44: 90, 45: 98, 46: 94, 47: 96, 48: 34, 49: 34, 50: 28, 51: 28, 52: 26, 53: 26, 54: 26, 55: 26, 56: 26,
        57: 26, 58: 26, 59: 26, 60: 2, 61: 2, 62: 2, 63: 2, 64: 40, 65: 40, 66: 40, 67: 40, 68: 14, 69: 14, 70: 22,
        71: 22, 72: 28, 73: 28, 74: 28, 75: 28, 76: 26, 77: 26, 78: 24, 79: 24, 80: 14, 81: 14, 82: 14, 83: 14,
        84: 12, 85: 12, 86: 12, 87: 12},
}
TEXT_WORDS_T8 = {
    2: {
        0: 156, 1: 156, 2: 148, 3: 134, 4: 56, 5: 56, 6: 52, 7: 88, 8: 58, 9: 58, 10: 56, 11: 56, 12: 30, 13: 30,
        14: 44, 15: 44, 16: 32, 17: 32, 18: 44, 19: 44, 20: 24, 21: 24, 22: 24, 23: 24, 24: 22, 25: 22, 26: 30,
        27: 30, 28: 90, 29: 90, 30: 61, 31: 61, 32: 40, 33: 40, 34: 40, 35: 40, 36: 36, 37: 36, 38: 22, 39: 22,
        40: 18, 41: 18, 42: 17, 43: 17, 44: 162, 45: 178, 46: 178, 47: 178, 48: 58, 49: 58, 50: 54, 51: 54, 52: 44,
        53: 44, 54: 42, 55: 42, 56: 42, 57: 42, 58: 40, 59: 40, 60: 42, 61: 42, 62: 42, 63: 42, 64: 44, 65: 44,
        66: 42, 67: 42, 68: 44, 69: 44, 70: 46, 71: 46, 72: 76, 73: 76, 74: 76, 75: 76, 76: 68, 77: 68, 78: 68,
        79: 68, 80: 36, 81: 36, 82: 38, 83: 38, 84: 34, 85: 34, 86: 38, 87: 38},
    8: {
        0: 556, 1: 556, 2: 510, 3: 448, 4: 166, 5: 166, 6: 158, 7: 304, 8: 182, 9: 182, 10: 178, 11: 178, 12: 38,
        13: 38, 14: 146, 15: 146, 16: 154, 17: 154, 18: 166, 19: 166, 20: 104, 21: 104, 22: 76, 23: 76, 24: 96,
        25: 96, 26: 88, 27: 88, 28: 80, 29: 80, 30: 72, 31: 72, 32: 74, 33: 74, 34: 70, 35: 70, 36: 123, 37: 123,
        38: 123, 39: 123, 40: 123, 41: 123, 42: 134, 43: 134, 44: 572, 45: 634, 46: 624, 47: 626, 48: 178, 49: 178,
        50: 160, 51: 160, 52: 146, 53: 146, 54: 144, 55: 144, 56: 146, 57: 146, 58: 148, 59: 148, 60: 196, 61: 196,
        62: 188, 63: 188, 64: 186, 65: 186, 66: 188, 67: 188, 68: 454, 69: 454, 70: 226, 71: 226, 72: 304, 73: 304,
        74: 240, 75: 240, 76: 234, 77: 234, 78: 198, 79: 198, 80: 166, 81: 166, 82: 144, 83: 144, 84: 154, 85: 154,
        86: 126, 87: 126},
}


# The Flow's own vectors (`emit_step_batch.PlannedFlow`): the f16 tables `hm` (the head masks) and `oh` (the one-hot vectors)
# on the west slices of the weights (bank 0, above the weights: the planes of one table are the two slices), and per lane
# the activation vectors, one word each in bank 1 of the quad the emitter stores them on. A lane's vectors are placed by
# (scope, hemisphere, quad, ordinal in the scope): a layer's vectors reuse the same words in every layer (a stage reads its
# inputs before the next stage writes), the expansion's P_s have a scope of their own so that their number (C) does not
# move the other vectors, and the final LayerNorm's output a third; the words are therefore the same in every bucket.
FLOW_TABLES = {"hm": ((0, 1), "fhm", 16), "oh": ((12, 13), "foh", CTX_MAX)}
FLOW_SCOPES = ("emb", "layer", "ps", "final")
FLOW_ARENA = {       # {(hemisphere, quad): {scope: words per lane}}: the largest the emitter takes (`test_step_buckets.py` re-derives it)
    ("E", 1): {"layer": 2, "final": 1}, ("E", 4): {"emb": 1, "layer": 1, "ps": CTX_MAX // 2}, ("E", 5): {"layer": 2, "ps": CTX_MAX // 2},
    ("E", 6): {"layer": 10}, ("E", 7): {"layer": 19}, ("E", 8): {"layer": 10}, ("E", 9): {"layer": 9},
    ("E", 10): {"layer": 9}, ("W", 1): {"layer": 3}, ("W", 2): {"layer": 1}, ("W", 3): {"layer": 1},
    ("W", 6): {"layer": 2, "final": 1}, ("W", 7): {"layer": 1}, ("W", 8): {"layer": 1}}


def flow_quad(hemi, quad):
    first = (0 if hemi == "W" else 44) + 4 * quad
    return tuple(range(first, first + 4))


def flow_arena_name(lane, hemi, quad):
    return f"fa{lane}_{hemi}{quad}"


def flow_arena_offset(hemi, quad, scope):
    """Word of `scope`'s first vector in a lane's region on (hemi, quad)."""
    parts = FLOW_ARENA[(hemi, quad)]
    return sum(parts.get(sc, 0) for sc in FLOW_SCOPES[:FLOW_SCOPES.index(scope)])


def serve_config(layers) -> ModelConfig:
    if layers not in SERVE_LAYERS:
        raise ValueError(f"serve layouts exist for {SERVE_LAYERS} layers, not {layers}")
    return TINYSTORIES if layers == 8 else TINYSTORIES_L2


def rows_total(vocab, pack):
    return -(-vocab // pack)


def emb_part_rows(vocab, bank_words=4096, pack=PACK):
    """Packed rows of each embedding part: the packed table is cut at the bank size (a region may not cross a bank). The embed
    kernel ORs three parts, so a table that fits fewer is cut into three of about the same size."""
    rows, parts = -(-vocab // pack), []
    total = rows
    while rows > 0:
        parts.append(min(rows, bank_words))
        rows -= parts[-1]
    if len(parts) < len(EMB_PAIRS) <= total:
        base, extra = divmod(total, len(EMB_PAIRS))
        parts = [base + (i < extra) for i in range(len(EMB_PAIRS))]
    if len(parts) > len(EMB_PAIRS):
        part = -(-rows_total(vocab, pack) // len(EMB_PAIRS))
        raise ValueError(f"the vocabulary of {vocab} entries needs {len(parts)} embedding parts of at most {bank_words} packed rows "
                         f"({pack} entries per row) and the layout has {len(EMB_PAIRS)} slice pairs for them: it needs "
                         f"{2 * (1 << (part - 1).bit_length())} words per slice (parts of {part} packed rows in a bank), the profile has "
                         f"{2 * bank_words}")
    return parts


def emb_locate(token, bank_words=4096, pack=PACK):
    """(part, word in the part, segment) of vocabulary row `token`."""
    rho = token // pack
    return rho // bank_words, rho % bank_words, token % pack


def text_words(layers, slice_):
    """The text arena of a slice of the model of `layers` layers: the largest text of the 18 buckets, plus the margin."""
    return -(-TEXT_MARGIN_PCT * TEXT_WORDS_T8[layers][slice_] // 100)


def plan_serve(cfg: ModelConfig, profile: Profile, text=None) -> Planner:
    """The serve plan. `text`: {slice: words} of the `tx` regions instead of the pinned `text_words` (the text sizing builds
    with roomy regions and measures what the pager takes)."""
    needs.SERVE.check(profile)
    n = cfg.layers
    shape = Shape.of(profile, cfg)
    pl = Planner(profile)
    _add_reduce_scratch(pl)
    _add_kernel_scratch(pl, shape.pack)
    _add_layer_weights(pl, cfg)
    for group, installs in ((WEST_W, (shape.head_installs + 1) // 2), (EAST_W, shape.head_installs // 2)):
        for s in group:
            pl.add(f"hd{s}", "head", (s,), installs * shape.nlw_head, rw=False, exported=True,
                   note="packed head table: install i on the west plane when i is even, word nlw (i div 2)")
    for (lo, hi), rows in zip(EMB_PAIRS, emb_part_rows(cfg.vocab, profile.bank_words, shape.pack)):
        for s in (lo, hi):
            pl.add(f"em{s}", "embed", (s,), rows, rw=False, exported=True,
                   note="packed embedding rows, byte plane " + ("0" if s == lo else "1"))
            pl.add(f"ez{s}", "embed", (s,), 1, rw=False, exported=True, bank=1,
                   note="zero word of the part: the read the other parts' lookups use")
    for s in WPE_SLICES:
        pl.add(f"wp{s}", "embed", (s,), WPE_POSITIONS, rw=False, exported=True, bank=1, note="wpe, f32 byte plane")
    for s in MASK_SLICES:
        for c in MASK_CS:
            pl.add(f"mk{c}s{s}", "mask", (s,), c, rw=False, exported=True, note=f"position masks of C = {c}, f32 byte plane")
    nconst = len(const_vectors(cfg, serve=True, lanes=shape.lanes))
    for s in CONST_QUAD:
        pl.add(f"cs{s}", "const", (s,), nconst, rw=False, exported=True,
               note="f32 constants and splats of the program's kernels, one plane per slice")
    # ax64..ax67 are UNUSED by the batched program (the Flow's one-hot and head-mask tables are the planned
    # `foh` / `fhm`); they stay planned and exported, so that the package and the decode step's plan are unchanged
    pl.add("ax64", "aux", (AUX["onehot_hi"],), CTX_MAX, rw=False, exported=True, note="one-hot e_s, f16 high plane")
    pl.add("ax65", "aux", (AUX["onehot_zero"],), 1, rw=False, exported=True, note="f16 low plane of the one-hot vectors (zero)")
    pl.add("ax66", "aux", (AUX["headmask_hi"],), cfg.heads, rw=False, exported=True, note="head masks, f16 high plane")
    pl.add("ax67", "aux", (AUX["headmask_zero"],), 1, rw=False, exported=True, note="f16 low plane of the head masks (zero)")
    pl.add("maps", "reserved", MAPS_QUAD, MAPS_WORDS, rw=False, note="SXM butterfly and broadcast maps (TS-1)")
    blocks = SERVE_BLOCKS + 1
    pl.add("kp", "kpool", EAST_W, blocks * 2 * n, rw=True, bank=1,
           note="K pool: block b, layer l at word 2 n b + 2 l (+1 for rows 8..15); block 0 zero, block 128 the sink")
    for l in range(n):
        q = 20 + 4 * (l // 2)
        pl.add(f"vp{l}", "vpool", tuple(range(q, q + 4)), blocks * BLOCK_TOKENS, rw=True, bank=l % 2,
               note=f"layer {l} V pool: block b at word 16 b, row o at word + o; block 0 zero, block 128 the sink")
    pl.add("lg", "logits", LOGIT_SLICES, LOGIT_ROWS * shape.logit_stride, rw=True, bank=0,
           note="f32 logit vectors of the west plane's installs (even i): row r, install i, segment k at word 315 r + 5 i + k")
    pl.add("lg2", "logits", LOGIT_SLICES_EAST, LOGIT_ROWS * shape.logit_stride, rw=True, bank=0,
           note="the same for the east plane's installs (odd i): another quad, so that L = 4 rows page")
    for name, (slices, prefix, words) in FLOW_TABLES.items():
        for s in slices:
            pl.add(f"{prefix}{s}", "const", (s,), words, rw=False, exported=True, note=f"f16 table {name} of the Flow, byte plane")
    for t in range(LANES_MAX):
        for (hemi, quad), parts in sorted(FLOW_ARENA.items()):
            pl.add(flow_arena_name(t, hemi, quad), "act", flow_quad(hemi, quad), sum(parts.values()), rw=True, bank=1,
                   note=f"lane {t} f32 activation vectors of the Flow on {hemi} quad {quad}, one word each: "
                        + ", ".join(f"{k} {v}" for k, v in parts.items() if v))
    for s in (TEXT_SLICES if text is not None else sorted(TEXT_WORDS_T8[n])):
        pl.add(f"tx{s}", "text", (s,), text_words(n, s) if text is None else text[s], rw=True, bank=0 if s in EAST_W else 1,
               note="paged instruction text of the active bucket, written by the host before the run")
    # weights and head tables stream in the slices whose pool is written; the pool and the text of those slices are in
    # the other bank (a patched write must never meet a read of its bank)
    for s in EAST_W:
        pl.conflict("kp", f"hd{s}")
        pl.conflict("kp", f"tx{s}")
        for l in range(n):
            pl.conflict("kp", f"w{l}s{s:02d}")
    pl.place()
    return pl


def pool_text_overlaps(plan_: Planner):
    """(pool, text, slice) for every KV pool region that shares a slice and a bank with the paged text of that slice: a
    write window of the pool and an instruction fetch of the text may then meet (OTS-E-MEM-001), which only the pager can
    keep apart. The K pool has none. Every V slice carries the pools of two layers, one in each bank, so whichever bank
    holds its text, one of them overlaps: the colouring cannot remove these, the pager rule of  must ."""
    out = []
    for r in plan_.regions:
        if r.kind in ("kpool", "vpool"):
            for s in r.slices:
                t = plan_.by_name.get(f"tx{s}")
                if t is not None and t.bank == r.bank:
                    out.append((r.name, t.name, s))
    return out


def serve_model(profile, layers=2) -> ModelConfig:
    """The serving model of a profile: the 8-layer or the 2-layer test model when the profile holds it (its matrices fit a plane, its
    vocabulary fits the embedding parts), else the small test model `config.SERVE_SMALL` (2 layers)."""
    cfg = serve_config(layers)
    try:
        needs.model_needs(cfg).check(profile)
        emb_part_rows(cfg.vocab, profile.bank_words, Shape.of(profile, cfg).pack)
        return cfg
    except (needs.ProfileUnfit, ValueError):
        return SERVE_SMALL


def plan(profile: Profile, mode="step", layers=8, head_chunks=0) -> Planner:
    """The planner for a mode: "step" (default; the one-token step with `head_chunks` of the head window) or "serve"
    (the batched, paged, whole-model step of the vLLM integration, for 8 layers or the 2-layer test model; on a profile that cannot
    hold the test model, the small one)."""
    if mode == "step":
        return plan_tinystories(serve_config(layers) if layers != 8 else TINYSTORIES, profile, head_chunks)
    if mode == "serve":
        return plan_serve(serve_model(profile, layers), profile)
    raise ValueError(f"unknown layout mode {mode!r}")


# -- the checker artifact ------------------------------------------------------------------------------------------
def _group(r):
    return {"weight": "weights", "act": "arena", "text": "text"}.get(r.kind, r.name)


def report(plan_: Planner, cfg: ModelConfig):
    """`layout_report.json`: every region with its slices, bank and address, the occupancy of every slice and bank by
    region group, and the rows of the design's per-bank table (2.8) with the measured components apart."""
    p = plan_.p
    regions = [{"name": r.name, "kind": r.kind, "slices": list(r.slices), "addr": r.addr, "words": r.words, "bank": r.bank,
                "rw": r.rw, "exported": r.exported} for r in plan_.regions]
    occ = {}
    for r in plan_.regions:
        for s in r.slices:
            d = occ.setdefault(s, [{}, {}])[r.bank]
            d[_group(r)] = d.get(_group(r), 0) + r.words
    slices = {str(s): {f"bank{b}": {"words": sum(occ[s][b].values()), "regions": occ[s][b]} for b in (0, 1)} for s in sorted(occ)}
    rep = {"schema": "opentsp.showcase-layout-report", "version": 1, "profile": p.name, "layers": cfg.layers,
           "bank_words": p.bank_words, "slice_words": p.words_per_slice, "regions": regions, "slices": slices,
           "max_bank_words": max(v[f"bank{b}"]["words"] for v in slices.values() for b in (0, 1)),
           "total_words": sum(r["words"] * len(r["slices"]) for r in regions),
           "text_source": f"measured from emit_step_batch: the largest bucket P(8, 4, 128), {cfg.layers} layers",
           "design_rows": design_rows(plan_, cfg, occ)}
    return rep


def check_report(rep):
    """The rules of  on a report alone: a region fits a bank (and crosses none), a bank of a slice holds at most
    its words, no two regions of a slice overlap."""
    bw = rep["bank_words"]
    spans = {}
    for r in rep["regions"]:
        if r["words"] > bw:
            raise LayoutError(f"region {r['name']} of {r['words']} words exceeds a bank ({bw})")
        if r["addr"] // bw != (r["addr"] + r["words"] - 1) // bw or r["addr"] // bw != r["bank"]:
            raise LayoutError(f"region {r['name']} at {r['addr']:#x} + {r['words']} crosses a bank or is not in bank {r['bank']}")
        for s in r["slices"]:
            spans.setdefault((s, r["bank"]), []).append((r["addr"], r["addr"] + r["words"], r["name"]))
    for (s, b), sp in spans.items():
        sp.sort()
        for x, y in zip(sp, sp[1:]):
            if y[0] < x[1]:
                raise LayoutError(f"regions {x[2]} and {y[2]} overlap in slice {s}")
        if sum(e - a for a, e, _ in sp) > bw:
            raise LayoutError(f"bank {b} of slice {s} exceeds {bw} words")
    return True


def check_design_rows(rep, tol=0.10):
    """Check each slice group's planned bank occupancy against the derived table within `tol`."""
    for row in rep["design_rows"]:
        for bank, want in row["design"].items():
            got = row["planned"][bank]
            if abs(got - want) > tol * want:
                raise LayoutError(f"row {row['row']} {bank}: {got} words against the design's {want} (more than {tol:.0%} off)")
    return True


def flow_arena_words(hemi, quad):
    """Words of the lane arenas of one (hemisphere, quad): LANES_MAX lanes of `FLOW_ARENA`'s vectors."""
    return LANES_MAX * sum(FLOW_ARENA[(hemi, quad)].values())


def design_table(n):
    """Derive per-bank occupancy for n layers (weights 48 and 32 words per layer
    and slice, `kfix`, the head's installs, the reduce scratch span, the K and V pool blocks, per-slice text maxima over the grid
    (`TEXT_WORDS_T8`), the Flow's tables and the arenas of eight lanes): (row, slices, {bank: words})."""
    kfix, rb, pool = KFIX_WORDS, RB_WORDS, (SERVE_BLOCKS + 1) * BLOCK_TOKENS
    t = lambda *s: max(text_words(n, x) for x in s)
    logit_words = LOGIT_ROWS * HEAD_INSTALLS * PACK
    return [
        ("west 0..15", WEST_W, {0: 48 * n + kfix + 32 * NLW_HEAD + CTX_MAX, 1: rb + t(*WEST_W)}),
        ("east 44..59", EAST_W, {0: 32 * n + 31 * NLW_HEAD + rb + t(*EAST_W), 1: rb + (SERVE_BLOCKS + 1) * 2 * n}),
        ("V pool 20..35", tuple(range(20, 20 + 4 * -(-n // 2))), {0: pool, 1: pool + t(*range(20, 20 + 4 * -(-n // 2)))}),
        ("embedding parts 0, 1: 36..39", tuple(range(36, 40)), {0: 4096, 1: 1 + t(*range(36, 40))}),
        ("part 2, wpe: 40..41", (40, 41), {0: 1860, 1: 1 + WPE_POSITIONS + t(40, 41)}),
        ("wpe: 42..43", (42, 43), {1: WPE_POSITIONS + t(42, 43)}),
        ("Flow arena, east quad 4: 60..63", SCRATCH_QUAD, {1: flow_arena_words("E", 4) + t(*SCRATCH_QUAD)}),
        ("one-hot table, Flow arena, east quad 5: 64..67", (64, 65, 66, 67),
         {0: CTX_MAX, 1: flow_arena_words("E", 5) + t(64, 65, 66, 67)}),
        ("logits, Flow arena, east quad 7: 72..75", LOGIT_SLICES, {0: logit_words, 1: flow_arena_words("E", 7) + t(*LOGIT_SLICES)}),
        ("masks, east logits, Flow arena, east quad 8: 76..79", MASK_SLICES,
         {0: sum(MASK_CS) + logit_words, 1: flow_arena_words("E", 8) + t(*MASK_SLICES)}),
        ("constants 16..19", CONST_QUAD, {0: len(const_vectors(serve_config(n), serve=True)) + max(LANES_MAX + LOGIT_ROWS, k_head.SEGMENTS * LOGIT_ROWS), 1: t(*CONST_QUAD)})]


def design_rows(plan_, cfg, occ):
    rows = []
    for name, slices, banks in design_table(cfg.layers):
        planned = {f"bank{b}": max(sum(occ[s][b].values()) if s in occ else 0 for s in slices) for b in banks}
        rows.append({"row": name, "slices": [slices[0], slices[-1]], "design": {f"bank{b}": v for b, v in banks.items()},
                     "planned": planned,
                     "deviation_pct": {f"bank{b}": round(100.0 * (planned[f"bank{b}"] - v) / v, 1) for b, v in banks.items()}})
    return rows


# -- word images ------------------------------------------------------------------------------------------------
def _word(vlanes, lanes_bytes):
    w = np.zeros(vlanes, dtype=np.uint8)
    w[:len(lanes_bytes)] = lanes_bytes
    return w.tobytes()


def f16_planes(vec, vlanes):
    """An f16 vector (lanes 0..n-1, the rest zero) as its two byte-plane words (low byte plane, high byte plane)."""
    b = np.asarray(vec, dtype=np.float16).view(np.uint8).reshape(-1, 2)
    return [_word(vlanes, b[:, 0]), _word(vlanes, b[:, 1])]


def f32_planes(vec, vlanes):
    vec = np.asarray(vec)
    if vec.dtype == np.float16:
        # NumPy's half-to-single cast quiets signaling NaNs on some CPUs and preserves
        # their signaling bit on others. The package stores quiet NaNs with the source payload.
        bits = vec.view(np.uint16)
        nan = (bits & 0x7fff) > 0x7c00
        vec = (bits | (nan.astype(np.uint16) << 9)).view(np.float16)
    b = np.asarray(vec, dtype=np.float32).view(np.uint8).reshape(-1, 4)
    return [_word(vlanes, b[:, k]) for k in range(4)]


def matrix_words(W, vlanes, rows_per_install=8):
    """An f16 matrix (R x K, rows = outputs, columns = lanes) in the install layout of : row r is a stream
    pair; slice j (0..15) of the 16-slice group holds plane (j mod 2) of the rows with r mod 8 = j div 2, row r at
    word r div 8. Returns, per slice j, the list of R/8 words."""
    R, K = W.shape
    assert R % 8 == 0 and K <= vlanes
    out = [[None] * (R // 8) for _ in range(16)]
    for r in range(R):
        lo, hi = f16_planes(W[r], vlanes)
        out[2 * (r % 8)][r // 8] = lo
        out[2 * (r % 8) + 1][r // 8] = hi
    return out


def decode_matrix_words(per_slice, R, K):
    """Inverse of matrix_words (float16 R x K)."""
    W = np.zeros((R, K), dtype=np.float16)
    for r in range(R):
        lo = np.frombuffer(per_slice[2 * (r % 8)][r // 8], dtype=np.uint8)[:K]
        hi = np.frombuffer(per_slice[2 * (r % 8) + 1][r // 8], dtype=np.uint8)[:K]
        W[r] = np.stack([lo, hi], axis=1).reshape(-1).view(np.float16)
    return W


def const_arrays(cfg, w, serve=False, lanes=ref.LANES):
    """name -> f32 vector of every constant of const_vectors."""
    d = {}
    for l, lw in enumerate(w.layers):
        for k in ("ln1g", "ln1b", "bq", "bk", "bv", "bo", "ln2g", "ln2b", "dn_b"):
            d[f"L{l}.{k}"] = lw[k]
        d[f"L{l}.up_b0"], d[f"L{l}.up_b1"] = lw["up_b"][:cfg.half], lw["up_b"][cfg.half:]
    d["lnf.g"], d["lnf.b"] = w.lnf_g, w.lnf_b
    for n, v in (("k.gelu_c0", ref.GELU_C0), ("k.gelu_c1", ref.GELU_C1), ("k.one", ref.ONE), ("k.half", ref.HALF),
                 ("k.eps", np.float32(cfg.ln_eps)), ("k.inv_h", np.float32(1.0 / cfg.hidden))):
        d[n] = np.full(lanes if serve else cfg.half, v, dtype=np.float32)
    if serve:
        d["k.zero"] = np.zeros(lanes, dtype=np.float32)
    return d


def _static_images(cfg: ModelConfig, plan: Planner, w, onehot, serve=False):
    """The tensors of the layer weights, constants and auxiliary vectors (`onehot` one-hot words) and their objects."""
    vl = plan.p.vlanes
    tensors, objects = {}, {}
    for hemi, group in (("west", WEST_W), ("east", EAST_W)):
        for l in range(cfg.layers):
            lw = w.layers[l]
            mats = {"wq": lw["wq"], "wv": lw["wv"], "wo": lw["wo"], "wk": lw["wk"],
                    "up0": lw["up_w"][:cfg.half], "up1": lw["up_w"][cfg.half:],
                    "dn0": lw["dn_w"][:, :cfg.half], "dn1": lw["dn_w"][:, cfg.half:]}
            slice_words = {s: [] for s in group}
            off = 0
            for name, rows, cols in matrices(cfg, hemi):
                per = matrix_words(mats[name], vl)
                for j, s in enumerate(group):
                    slice_words[s] += per[j]
                objects[f"L{l}.{name}"] = {"kind": "matrix_f16", "layer": l, "rows": rows, "cols": cols, "slices": list(group),
                                           "addr": [plan.by_name[f"w{l}s{s:02d}"].addr + off for s in group],
                                           "words_per_slice": rows // 8,
                                           "layout": "slice j: plane (j mod 2) of rows r with r mod 8 = j div 2 at word r div 8"}
                off += rows // 8
            for s in group:
                tensors[f"w{l}s{s:02d}"] = ("u8", slice_words[s])
    consts = const_arrays(cfg, w, serve, vl)
    names = [n for n, _ in const_vectors(cfg, serve, vl)]
    planes = {n: f32_planes(consts[n], vl) for n in names}
    for k, s in enumerate(CONST_QUAD):
        tensors[f"cs{s}"] = ("u8", [planes[n][k] for n in names])
    for i, (n, lanes) in enumerate(const_vectors(cfg, serve, vl)):
        objects[n] = {"kind": "vector_f32", "lanes": lanes, "slices": list(CONST_QUAD),
                      "addr": [plan.by_name[f"cs{s}"].addr + i for s in CONST_QUAD], "words_per_slice": 1,
                      "layout": "slice k: byte plane k"}
    hi_words = []
    for s in range(onehot):
        e = np.zeros(vl, dtype=np.float16)
        e[s] = 1.0
        hi_words.append(f16_planes(e, vl)[1])
    tensors["ax64"] = ("u8", hi_words)
    tensors["ax65"] = ("u8", [bytes(vl)])
    hm = []
    for h in range(cfg.heads):
        m = np.zeros(vl, dtype=np.float16)
        m[h * cfg.dk:(h + 1) * cfg.dk] = 1.0
        hm.append(f16_planes(m, vl)[1])
    tensors["ax66"] = ("u8", hm)
    tensors["ax67"] = ("u8", [bytes(vl)])
    objects["onehot"] = {"kind": "f16_vectors", "count": onehot, "slices": [AUX["onehot_zero"], AUX["onehot_hi"]],
                         "addr": [plan.by_name["ax65"].addr, plan.by_name["ax64"].addr],
                         "layout": "e_s = 1.0 at lane s; slice 65 holds the zero low plane (one word, read with step 0), slice 64 the high planes"}
    objects["headmask"] = {"kind": "f16_vectors", "count": cfg.heads, "slices": [AUX["headmask_zero"], AUX["headmask_hi"]],
                           "addr": [plan.by_name["ax67"].addr, plan.by_name["ax66"].addr],
                           "layout": "1.0 on the head's dk lanes; slice 67 the zero low plane, slice 66 the high planes"}
    return tensors, objects


def _check_images(plan, tensors, vl):
    for name, (dt, words) in tensors.items():
        r = plan.by_name[name]
        assert len(words) == r.words, (name, len(words), r.words)
        assert all(len(x) == vl for x in words)


def build_images(cfg: ModelConfig, plan: Planner, w):
    """{tensor name: (dtype, [word bytes])} for every exported region, and the manifest 'objects' table.
    dtype is 'u8' (the words are byte planes); the package's dtype is informational."""
    vl = plan.p.vlanes
    tensors, objects = _static_images(cfg, plan, w, cfg.slots)
    _check_images(plan, tensors, vl)
    # runtime objects (not in the package): caches, mask, activations
    for l in range(cfg.layers):
        k = plan.by_name[f"kc{l}"]
        objects[f"L{l}.kcache"] = {"kind": "kcache_f16", "layer": l, "slots": cfg.slots, "cols": cfg.hidden,
                                   "slices": list(EAST_W), "addr": [k.addr] * 16, "words_per_slice": cfg.slots // 8,
                                   "current_slot": cfg.slots - 1,
                                   "layout": "install layout of a (slots x H) f16 matrix; the current K row (slot "
                                             f"{cfg.slots - 1}) is the planes in slices {EAST_W[0] + 2 * ((cfg.slots - 1) % 8)}"
                                             f"+0/1 at word {k.addr + (cfg.slots - 1) // 8}"}
        v = plan.by_name[f"vc{l}"]
        objects[f"L{l}.vcache"] = {"kind": "vcache_f32", "layer": l, "slots": cfg.slots, "cols": cfg.hidden,
                                   "slices": list(VCACHE_QUAD), "addr": [v.addr] * 4, "words_per_slice": cfg.slots,
                                   "current_slot": cfg.slots - 1,
                                   "layout": "slot s = word addr + s of each slice, slice k = byte plane k of the f32 row"}
    mk = plan.by_name["mask"]
    objects["mask"] = {"kind": "vector_f32", "lanes": cfg.slots, "slices": list(CONST_QUAD), "addr": [mk.addr] * 4,
                       "words_per_slice": 1, "layout": "slice k: byte plane k (host-written each step)"}
    for n in scratch_vectors(cfg):
        a = plan.by_name["a_" + n]
        objects["act." + n] = {"kind": "vector_f32", "lanes": cfg.half if n[0] in "ug" else cfg.hidden,
                               "slices": list(SCRATCH_QUAD), "addr": [a.addr] * 4, "words_per_slice": 1, "layout": "slice k: byte plane k"}
    return tensors, objects


def decode_const(plan, tensors, name, cfg):
    """f32 vector of a constant from the images (for round-trip tests)."""
    i = [n for n, _ in const_vectors(cfg)].index(name)
    planes = [np.frombuffer(tensors[f"cs{s}"][1][i], dtype=np.uint8) for s in CONST_QUAD]
    lanes = dict(const_vectors(cfg))[name]
    b = np.stack([p[:lanes] for p in planes], axis=1)
    return np.ascontiguousarray(b).reshape(-1).view(np.float32)


# -- images of the serving layout --------------------------------------------------------------------------------------
def mask_vector(c, p, lanes=ref.LANES):
    """The position mask of position p for a context of c slots: slot s of the scores (lane s) is visible when s <= p; the
    lanes beyond c are zero."""
    v = np.zeros(lanes, dtype=np.float32)
    v[:c] = np.where(np.arange(c) <= p, 0.0, -np.inf)
    return v


def _packed(wte, rows, pack=PACK):
    """wte (vocab x 64 f16) as `rows` packed rows of `pack` * 64 lanes (`pack` vocabulary rows each), zero-padded."""
    pad_ = np.zeros((rows * pack, wte.shape[1]), dtype=np.float16)
    pad_[:len(wte)] = wte
    return pad_.reshape(rows, pack * wte.shape[1])


def _packed_lanes(wte, rows, pack, vlanes):
    """`_packed` as rows of `vlanes` lanes: the lanes above the `pack` segments are zero (a vector wider than the packed row)."""
    packed = _packed(wte, rows, pack)
    if packed.shape[1] < vlanes:
        packed = np.concatenate([packed, np.zeros((len(packed), vlanes - packed.shape[1]), np.float16)], axis=1)
    return packed


def _plane_words(rows_u8):
    return [bytes(r) for r in rows_u8]


def build_serve_images(cfg: ModelConfig, plan_: Planner, w):
    """Tensors and objects of the serve layout: the static ones of the step layout, plus the packed head table, the
    embedding parts with their zero words, `wpe` and the mask tables."""
    vl = plan_.p.vlanes
    shape = Shape.of(plan_.p, cfg)
    tensors, objects = _static_images(cfg, plan_, w, CTX_MAX, serve=True)
    rows = -(-cfg.vocab // shape.pack)
    emb = _packed_lanes(w.wte, rows, shape.pack, vl)
    off = 0
    for part, ((lo, hi), n) in enumerate(zip(EMB_PAIRS, emb_part_rows(cfg.vocab, plan_.p.bank_words, shape.pack))):
        u8 = emb[off:off + n].view(np.uint8).reshape(n, vl, 2)
        for plane, s in enumerate((lo, hi)):
            tensors[f"em{s}"] = ("u8", _plane_words(np.ascontiguousarray(u8[:, :, plane])))
            tensors[f"ez{s}"] = ("u8", [bytes(vl)])
        objects[f"embed.part{part}"] = {"kind": "embedding_f16", "part": part, "rows": n, "first_packed_row": off,
                                        "slices": [lo, hi], "addr": [plan_.by_name[f"em{lo}"].addr] * 2,
                                        "zero_addr": [plan_.by_name[f"ez{lo}"].addr] * 2,
                                        "layout": f"word w of the part = packed row first_packed_row + w, lanes 64 k .. 64 k + 63 "
                                                  f"= vocabulary row {shape.pack} (packed row) + k; low byte plane on the first slice, high on the second"}
        off += n
    head = _packed(w.wte, shape.head_installs * shape.head_rows, shape.pack)
    per = {s: [] for s in WEST_W + EAST_W}
    for i in range(shape.head_installs):
        group = WEST_W if i % 2 == 0 else EAST_W
        for j, words in enumerate(matrix_words(head[shape.head_rows * i:shape.head_rows * (i + 1)], vl)):
            per[group[j]] += words
    for s, words in per.items():
        tensors[f"hd{s}"] = ("u8", words)
    objects["head"] = {"kind": "head_f16", "installs": shape.head_installs, "rows_per_install": shape.head_rows, "segments": shape.pack,
                       "west_slices": list(WEST_W), "east_slices": list(EAST_W),
                       "west_addr": plan_.by_name["hd0"].addr, "east_addr": plan_.by_name["hd44"].addr,
                       "layout": f"install i on the west plane when i is even, else east, words {shape.nlw_head} (i div 2) .. +{shape.nlw_head - 1} of the plane's slices "
                                 f"in the install layout of the matrices; row r of install i holds packed row {shape.head_rows} i + r"}
    wp = [f32_planes(w.wpe[p], vl) for p in range(WPE_POSITIONS)]
    for k, s in enumerate(WPE_SLICES):
        tensors[f"wp{s}"] = ("u8", [v[k] for v in wp])
    objects["wpe"] = {"kind": "vector_f32_table", "positions": WPE_POSITIONS, "lanes": cfg.hidden, "slices": list(WPE_SLICES),
                      "addr": [plan_.by_name[f"wp{s}"].addr for s in WPE_SLICES], "layout": "position p = word addr + p, slice k = byte plane k"}
    for c in MASK_CS:
        vecs = [f32_planes(mask_vector(c, p, vl), vl) for p in range(c)]
        for k, s in enumerate(MASK_SLICES):
            tensors[f"mk{c}s{s}"] = ("u8", [v[k] for v in vecs])
        objects[f"mask{c}"] = {"kind": "vector_f32_table", "positions": c, "slices": list(MASK_SLICES),
                               "addr": [plan_.by_name[f"mk{c}s{s}"].addr for s in MASK_SLICES],
                               "layout": f"position p = word addr + p, slice k = byte plane k; lane s of vector p = 0 if s <= p else -inf (s < {c})"}
    for name, vecs in flow_table_vectors(cfg, vl).items():
        slices, prefix, _ = FLOW_TABLES[name]
        planes = [f16_planes(v, vl) for v in vecs]
        for k, s in enumerate(slices):
            tensors[f"{prefix}{s}"] = ("u8", [pl[k] for pl in planes])
        objects[f"flow.{name}"] = {"kind": "f16_vector_table", "count": len(vecs), "slices": list(slices),
                                   "addr": [plan_.by_name[f"{prefix}{s}"].addr for s in slices],
                                   "layout": "vector i = word addr + i, slice k = byte plane k (low, high) of the f16 vector"}
    _check_images(plan_, tensors, vl)
    return tensors, objects


def flow_table_vectors(cfg, vl):
    """{name: the f16 vectors (vl lanes) of the Flow table}: `hm`, the head masks (1.0 on the head's lanes), and `oh`, the
    one-hot vectors e_s (1.0 in lane s) for the contexts of the grid."""
    hm = np.zeros((cfg.heads, vl), np.float16)
    for h in range(cfg.heads):
        hm[h, h * cfg.dk:(h + 1) * cfg.dk] = 1.0
    oh = np.zeros((CTX_MAX, vl), np.float16)
    oh[np.arange(CTX_MAX), np.arange(CTX_MAX)] = 1.0
    return {"hm": list(hm), "oh": list(oh)}


def _rows(tensors, name):
    return np.frombuffer(b"".join(tensors[name][1]), dtype=np.uint8).reshape(len(tensors[name][1]), -1)


def decode_embedding(tensors, vocab, padded=False, pack=PACK):
    """wte (f16) from the embedding parts of the package images."""
    parts = []
    for lo, hi in EMB_PAIRS:
        if f"em{lo}" in tensors:
            parts.append(np.stack([_rows(tensors, f"em{lo}"), _rows(tensors, f"em{hi}")], axis=-1))
    packed = np.concatenate(parts)[:, :pack * 64].reshape(-1, 2 * pack * 64)
    rows = np.ascontiguousarray(packed).view(np.float16).reshape(-1, 64)
    return rows if padded else rows[:vocab]


def decode_head(tensors, vocab, padded=False, shape=None):
    """wte (f16) from the packed head table of the package images (`shape`: the profile's `Shape`, default that of `ref`)."""
    installs, nlw, head_rows, lanes = ((HEAD_INSTALLS, NLW_HEAD, HEAD_CHUNK, PACK * 64) if shape is None else
                                       (shape.head_installs, shape.nlw_head, shape.head_rows, shape.pack * 64))
    out = []
    for i in range(installs):
        group = WEST_W if i % 2 == 0 else EAST_W
        per = [list(tensors[f"hd{s}"][1][nlw * (i // 2):nlw * (i // 2 + 1)]) for s in group]
        out.append(decode_matrix_words(per, head_rows, lanes))
    rows = np.concatenate(out).reshape(-1, 64)
    return rows if padded else rows[:vocab]


def decode_wpe(tensors):
    planes = [_rows(tensors, f"wp{s}")[:, :64] for s in WPE_SLICES]
    return np.ascontiguousarray(np.stack(planes, axis=-1)).reshape(WPE_POSITIONS, -1).view(np.float32)


def decode_mask(tensors, c, p):
    planes = [np.frombuffer(tensors[f"mk{c}s{s}"][1][p], dtype=np.uint8) for s in MASK_SLICES]
    return np.ascontiguousarray(np.stack(planes, axis=1)).reshape(-1).view(np.float32)


def kblock_addr(plan_, layer, block):
    """Word (in each of the 16 K slices) of rows 0..7 of K block `block` of `layer`; rows 8..15 are the next word."""
    if not 0 <= block <= SINK_BLOCK:
        raise ValueError(f"K block {block} outside 0..{SINK_BLOCK}")
    n = plan_.by_name["kp"].words // (SERVE_BLOCKS + 1) // 2
    return plan_.by_name["kp"].addr + 2 * n * block + 2 * layer


def vblock_addr(plan_, layer, block):
    """Word of row 0 of V block `block` of `layer` in each of the 4 slices of the layer's quad."""
    if not 0 <= block <= SINK_BLOCK:
        raise ValueError(f"V block {block} outside 0..{SINK_BLOCK}")
    return plan_.by_name[f"vp{layer}"].addr + BLOCK_TOKENS * block
