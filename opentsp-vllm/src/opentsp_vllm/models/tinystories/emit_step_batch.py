"""Emit one paged TinyStories step as a batched program. It runs in layer order and visits each lane serially.

`emit_step_batch(T, L, C)` builds the program of the bucket P(T, L, C) of the grid (T in 1, 2, 4, 8 token lanes, L in 1, 2,
4 logit rows with L <= T, C in 64, 128 context slots). Per lane the program runs the token embedding on the chip
(`kernels/embed.py`); then, per layer, every lane's first half (LayerNorm 1, the Q, K, V projections, the paged writes of the
new K and V rows: `kv_write`) and then every lane's second half (the K install of the lane's context block by block
(`kv_install`), the scores with the position mask selected by a patched address, the softmax (2 rounds at C = 64, 3 at
C = 128), the expansion, the context over the V rows read block by block, the output projection, the residual, LayerNorm 2,
the MLP and the residual: the decode step's `emit_block.Block`, in its paged mode); the writes of all lanes precede every read of the
layer, so a prefill chunk sees its own earlier tokens. Then the final LayerNorm of the L logit lanes (the first L lanes: the
host puts the lanes that need a next token first) and the packed output head (`kernels/lm_head.py`) for their L rows.

The program carries no step data: every address that depends on the step is a patch site, with a placeholder that makes the
lane a padding lane (the zero block for every context block, the zero words of the embedding table, position 0, the sinks
for the K and V writes). `site_values` is the host patcher's input for a step. Sites are named
`t<lane>.emb.*` (embedding), `t<lane>.L<layer>.mk.s<slice>` (mask), `.kb.b<j>.s<slice>` (K block), `.vb.b<j>.s<slice>`
(V block), `.kw.p<pair>.<plane>` (K write) and `.vw.s<slice>` (V write).

Every word the program addresses is at an address the planner (`layout.plan_serve`) gives, the same for every bucket: the
pools, the weights, the embedding and head tables, the mask tables, the logits, the reduce scratch, the kernel scratch and the
text blocks; the constants and tables of the Flow by name in `cs`, `fhm` and `foh`, and the lane-local vectors in the
per-lane regions `fa<lane>_<quad>` (`PlannedFlow`)."""
import contextlib
import time
from dataclasses import dataclass

from opentsp_tools.programs import data
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools.programs import needs
from opentsp_tools import profile as tprofile
from opentsp_tools.programs.kernels import context as k_context, embed as k_embed, kv_install as k_kvi, kv_write as k_kvw
from opentsp_tools.programs.kernels import lm_head as k_head
from opentsp_tools.programs.kernels.flow import Flow, Vec
from opentsp_tools.stations import ScheduleError
from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import layout
from . import embedding
from opentsp_vllm.models.tinystories import weights as W

SEED = 1
GRID_T, GRID_L, GRID_C = (1, 2, 4, 8), (1, 2, 4), (64, 128)
LANE = 0
EMBED_SPACING = 5             # cycles between the lookups of two lanes (the kernel's own cadence)


def check_bucket(T, L, C):
    if T not in GRID_T or L not in GRID_L or C not in GRID_C or L > T:
        raise ValueError(f"P({T}, {L}, {C}) is not in the grid: T in {GRID_T} token lanes, L in {GRID_L} logit rows with "
                         f"L <= T, C in {GRID_C} context slots (design 3.3)")


# the emitter's names of the constants (`Block.lconst`, `Flow.splat`) and the entries of `layout.const_vectors` they are
LAYER_CONSTS = {"ln1g": "ln1g", "ln1b": "ln1b", "bq": "bq", "bk": "bk", "bv": "bv", "bo": "bo", "ln2g": "ln2g", "ln2b": "ln2b",
                "upb0": "up_b0", "upb1": "up_b1", "dnb": "dn_b"}
FINAL_CONSTS = {"lnfg": "lnf.g", "lnfb": "lnf.b"}
SPLATS = {"invn": "k.inv_h", "eps": "k.eps", "half": "k.half", "hf": "k.half", "zero": "k.zero", "gc0": "k.gelu_c0",
          "gc1": "k.gelu_c1", "one": "k.one"}


def const_name(name):
    """The entry of `layout.const_vectors` that the emitter's constant `name` is: `ln1g_3` is layer 3's, `lnfg_f` the final
    LayerNorm's, `invn` a splat."""
    if name in SPLATS:
        return SPLATS[name]
    base, _, tag = name.partition("_")
    if tag == "f" and base in FINAL_CONSTS:
        return FINAL_CONSTS[base]
    if base in LAYER_CONSTS and tag.isdigit():
        return f"L{int(tag)}.{LAYER_CONSTS[base]}"
    raise ScheduleError(f"constant {name!r} has no place in the serve layout (layout.const_vectors)")


class PlannedFlow(Flow):
    """A Flow that places nothing itself. A constant or table is an entry of the planner's `cs`, `fhm` or `foh` regions, found
    by name; a vector the program stores is a word of the lane's own region `fa<lane>_<hemisphere><quad>`, found by (scope,
    quad, ordinal in the scope). The address of a vector therefore does not depend on the bucket: not on the lanes before it,
    the layers, the context or the logit rows. `used` records, per (hemisphere, quad, scope), how many words were taken (the
    largest over the layers: a scope restarts at its first word in every layer); with `measure` the vectors get no planned
    words and the program is not meant to be assembled (`layout.FLOW_ARENA` is derived from it)."""

    def __init__(self, k, plan, tensors, measure=False):
        super().__init__(k)
        self.plan, self.tensors, self.measure = plan, tensors, measure
        self.lane, self.scope = LANE, "layer"
        self.taken = {}
        self.used = {}
        self.consts = {}               # {entry of layout.const_vectors: word of cs}: the constants the program read
        self.vectors = {}              # {(lane, scope, hemisphere, quad, ordinal): word}: the vectors it stored (the last layer's)

    def enter(self, lane, scope="layer", reset=False):
        """Make `lane`'s `scope` the one the next vectors belong to; `reset` starts it again at its first word."""
        self.lane, self.scope = lane, scope
        if reset:
            for key in [k for k in self.taken if k[:2] == (lane, scope)]:
                del self.taken[key]

    @contextlib.contextmanager
    def scoped(self, scope, reset=True):
        """The vectors stored inside the block are `scope`'s; the scope before it resumes afterwards."""
        lane, before = self.lane, self.scope
        self.enter(lane, scope, reset)
        try:
            yield
        finally:
            self.enter(lane, before)

    def addr(self, hemi, quad, bank):
        if bank != 1:
            raise ScheduleError("the planned Flow stores its vectors in bank 1; constants and tables are placed by name")
        key = (self.lane, self.scope, hemi, quad)
        n = self.taken.get(key, 0)
        self.taken[key] = n + 1
        self.used[(hemi, quad, self.scope)] = max(self.used.get((hemi, quad, self.scope), 0), n + 1)
        if self.measure:
            return (1 << self.p.bank_bit) + n
        have = layout.FLOW_ARENA.get((hemi, quad), {}).get(self.scope, 0)
        if n >= have:
            raise ScheduleError(f"lane {self.lane}: vector {n} of scope {self.scope!r} on {hemi} quad {quad} but "
                                f"layout.FLOW_ARENA has {have} words there")
        base = self.plan.by_name[layout.flow_arena_name(self.lane, hemi, quad)].addr
        self.vectors[(self.lane, self.scope, hemi, quad, n)] = base + layout.flow_arena_offset(hemi, quad, self.scope) + n
        return self.vectors[(self.lane, self.scope, hemi, quad, n)]

    def const(self, name, arr, hemi, quad):
        """The constant `name`: its word in `cs` (the package holds it; the program's own array must equal the image)."""
        idx = [n for n, _ in layout.const_vectors(self.cfg, serve=True, lanes=self.p.vlanes)].index(const_name(name))
        planes = data.to_planes(arr)
        slices = list(layout.CONST_QUAD)
        for pl, s in zip(planes, slices):
            if bytes(pl) != bytes(self.tensors[f"cs{s}"][1][idx]):
                raise ScheduleError(f"constant {name!r} differs from word {idx} of cs{s} in the package")
        base = self.plan.by_name[f"cs{slices[0]}"].addr + idx
        self.consts[const_name(name)] = base
        return Vec(slices, base, 0, "W", [f"cs{s}+{idx:#x}" for s in slices])

    def table(self, name, arrs, hemi, quad):
        """The first len(arrs) vectors of the f16 table `name` (`hm`, `oh`): one Vec per vector, the two planes on two slices."""
        slices, prefix, words = layout.FLOW_TABLES[name]
        if len(arrs) > words:
            raise ScheduleError(f"table {name!r} has {words} vectors, the program asks for {len(arrs)}")
        planes = [data.to_planes(a) for a in arrs]
        for k, s in enumerate(slices):
            have = self.tensors[f"{prefix}{s}"][1]
            if any(bytes(pl[k]) != bytes(have[i]) for i, pl in enumerate(planes)):
                raise ScheduleError(f"table {name!r} differs from {prefix}{s} in the package")
        base = self.plan.by_name[f"{prefix}{slices[0]}"].addr
        return [Vec(list(slices), base + i, 0, "W", [f"{prefix}{s}+{i:#x}" for s in slices]) for i in range(len(arrs))]


class PagedAttention:
    """The paged mode of `emit_block.Block` for the lanes of a step: where the K and V rows go, how the context is read and
    which mask is selected, all as patch sites with the placeholders of a padding lane. `lane` is the lane the emitter is
    working on (the sites' names and the Flow's vectors follow it)."""

    def __init__(self, p, layers, C, tensors=None, plan=None, measure=False, cfg=None):
        check_bucket(1, 1, C)
        self.p, self.C, self.lane = p, C, LANE
        self.tensors, self.measure = tensors, measure
        self.geo = kvpool.Geometry(p, layers, plan, cfg)
        self.plan = self.geo.plan
        self.v_stream = None
        self.flow_ = None

    def flow(self, kit):
        f = PlannedFlow(kit, self.plan, self.tensors, self.measure)
        f.cfg = self.geo.cfg
        self.v_stream = k_context.reserve_v_streams(f, "E")
        self.flow_ = f
        return f

    def set_lane(self, lane, scope="layer", reset=False):
        self.lane = lane
        self.flow_.enter(lane, scope, reset)

    def scope(self, name):
        return self.flow_.scoped(name)

    def prefix(self, blk, kind):
        return f"t{self.lane}.L{blk.L}.{kind}"

    def write_k(self, blk, kk):
        geo, L = self.geo, blk.L
        sink = geo.kaddr(L, geo.sink)
        return k_kvw.write_k_paged(blk.f, kk, [sink] * 8, geo.krange(L, True), prefix=self.prefix(blk, "kw"))

    def write_v(self, blk, vv):
        geo, L = self.geo, blk.L
        slices = geo.vslices(L)
        return blk.b.write(vv, slices, [geo.vaddr(L, geo.sink)] * len(slices), window=True,
                           patch=[(f"{self.prefix(blk, 'vw')}.s{s}", *geo.vrange(L, True)) for s in slices])

    def install_k(self, blk):
        geo, L = self.geo, blk.L
        table = [0] * (self.C // 16)
        return lambda kit, t0: k_kvi.install_k(kit, table, lambda b: geo.kaddr(L, b), geo.krange(L), t0,
                                               prefix=self.prefix(blk, "kb"))

    def mask(self, blk):
        """The position mask of the lane: one patched Read of the mask table's word (the 4 byte planes on the east slices
        76..79) copied to a vector on west quad 2, which the 16 heads' adds then read as an ordinary vector (a patched
        Read per head would need 16 sites per slice)."""
        f, b = blk.f, blk.b
        regions = [self.plan.by_name[f"mk{self.C}s{s}"] for s in layout.MASK_SLICES]
        lo, = {r.addr for r in regions}
        t = blk.after()
        while not all(b.t.free(s, t, "R") for s in layout.MASK_SLICES):
            t += 1
        v = b.read(t, list(layout.MASK_SLICES), [lo] * 4, "W", f.stream("W", 4),
                   patch=[(f"{self.prefix(blk, 'mk')}.s{s}", lo, lo + self.C - 1) for s in layout.MASK_SLICES])
        return f.store(v, 2, hemi="W")

    def v_values(self, blk, t0):
        geo, L = self.geo, blk.L
        return k_context.read_v_blocks(blk.f, t0, [0] * (self.C // 16), lambda b: geo.vaddr(L, b), geo.vslices(L),
                                       geo.vrange(L), prefix=self.prefix(blk, "vb"), stream=self.v_stream)


class BatchKit(eb.BlockKit):
    case_tags = ["ts-step-batch", "iss"]


def bind(k, plan, tensors, name):
    """Region `name` of the plan as an extern of the program, bound from the package tensor of that name."""
    r = plan.by_name[name]
    k.b.prog.extern(name, r.slices[0], r.addr, r.words)
    k.weights[name] = ("raw", b"".join(bytes(w) for w in tensors[name][1]))


def text_regions(plan):
    """{slice: (first word, end word)} of the planner's text regions `tx<slice>`: where the pager may put the blocks."""
    return {r.slices[0]: (r.addr, r.addr + r.words) for r in plan.regions if r.kind == "text"}


@dataclass
class Batch:
    """The program of a bucket and what a step needs to patch it and read its results."""
    bucket: tuple
    layers: int
    text: str
    case: dict
    kit: object
    geo: object
    sizes: object
    tables: object
    tensors: dict
    hidden: list                   # the final hidden Vecs (f32, west quad 6) of the L logit lanes
    head: object
    flow: object                   # the PlannedFlow (`used`: the words taken per scope)
    marks: list
    seconds: dict                  # wall time of the build steps
    cfg: object = None             # the model
    shape: object = None           # `layout.Shape` of the profile and the model

    @property
    def plan(self):
        return self.geo.plan

    @property
    def paged(self):
        return self.kit.located


def emit_step_batch(T, L, C, p=None, layers=8, w=None, seed=SEED, plan=None, measure=False, cfg=None):
    """The program of the bucket P(T, L, C) on profile `p` (default `ref`) for the model of `layers` layers (8, or the
    2-layer test model; `cfg`: another model of the same structure, whose `layers` win) on the weights `w` (default the synthetic
    ones of `seed`). Returns a `Batch`. `plan`: another serve plan (the text sizing builds with roomy text regions); `measure`:
    stop before the program is assembled and report the Flow's words in `Batch.flow.used`. A profile that cannot hold the programs
    or the model is refused with the first unmet requirement (`opentsp_tools.programs.needs`)."""
    check_bucket(T, L, C)
    p = p or tprofile.load("ref")
    cfg = cfg or layout.serve_config(layers)
    layers = cfg.layers
    needs.SERVE.check(p)
    needs.model_needs(cfg).check(p)
    shape = layout.Shape.of(layout.load_profile(p.name), cfg)
    if C not in shape.ctxs:
        raise needs.ProfileUnfit(f"profile {p.name} does not fit the bucket P({T}, {L}, {C}): needs {2 * C} MXM rows for the "
                                 f"{C}-slot K install, profile has {p.mxm_rows}; its contexts are {shape.ctxs} "
                                 f"(first unmet requirement: mxm.rows)")
    plan_given = plan is not None
    clock = {"start": time.time()}
    w = w or W.synthetic(cfg, seed)
    pa = PagedAttention(p, layers, C, plan=plan, measure=measure, cfg=cfg)
    plan = pa.plan
    tensors, _ = layout.build_serve_images(cfg, plan, w)
    pa.tensors = tensors
    k = BatchKit(p, f"model/ts_step_batch_t{T}l{L}c{C}_{layers}", f"batched paged step P({T}, {L}, {C}), {layers} layers",
                 seed, module="emit_step_batch", source="numpy-twin")
    k.regions = text_regions(plan)
    clock["images"] = time.time()
    dummy = [eb.BlockInputs(cfg, 0, 0, lw, None, None, None, None) for lw in w.layers]
    blk = eb.Block(p, dummy[0], kit=k, plan=plan, layer=0, full=False, paged=pa)
    f = blk.f
    sizes = embedding.EmbedSizes(layout.emb_part_rows(cfg.vocab, plan.p.bank_words, shape.pack), cfg.vocab, pack=shape.pack)
    tb = embedding.tables(layout, plan, sizes)
    names = [n for lo, hi in layout.EMB_PAIRS[:len(sizes.part_rows)] for s in (lo, hi) for n in (f"em{s}", f"ez{s}")]
    names += [f"wp{s}" for s in layout.WPE_SLICES] + [f"mk{pa.C}s{s}" for s in layout.MASK_SLICES]
    names += [f"hd{s}" for s in layout.WEST_W + layout.EAST_W]
    names += [f"cs{s}" for s in layout.CONST_QUAD]
    names += [f"{prefix}{s}" for slices, prefix, _ in layout.FLOW_TABLES.values() for s in slices]
    for name in names:
        bind(k, plan, tensors, name)
    # the embedding of every lane: a padding lane until the host patches its sites
    t_ready = k_embed.load_maps(k, 40, tb)
    parts, wpe, map_id = [pt.zero for pt in tb.parts], list(tb.wpe_addrs), k_embed.FIRST_MAP
    xs = []
    for t in range(T):
        pa.set_lane(t, "emb", reset=True)
        emb = k_embed.embed(f, tb, f"t{t}.emb", parts, wpe, map_id, t_ready + EMBED_SPACING * t, k_embed.STAGE_ADDR + t)
        xs.append(f.store(emb, 4, hemi="E"))
    blk.mark("embed")
    for l, li in enumerate(dummy):
        blk.begin(li, l)
        states = []
        for t in range(T):
            pa.set_lane(t, "layer", reset=True)
            blk.rb_words = 0
            states.append(blk.run_a(x=xs[t], to="W", out_quad=1))
        for t in range(T):
            pa.set_lane(t)
            blk.rb_words = 0
            xs[t] = blk.run_b(states[t])
    lw = w
    blk.begin(dummy[-1], cfg.layers)
    blk.sfx = "_f"
    hidden = []
    for t in range(L):
        pa.set_lane(t, "final", reset=True)
        blk.rb_words = 0
        hidden.append(blk.layernorm(xs[t], blk.lconst("lnfg", lw.lnf_g, "W", 2), blk.lconst("lnfb", lw.lnf_b, "W", 2), None,
                                    to="W", quad=6, keep=True))
    blk.mark("final_ln")
    head = k_head.Head(plan.by_name["hd0"].addr, plan.by_name["hd44"].addr, shape.head_installs, layout.LOGIT_SLICES,
                       plan.by_name["lg"].addr, layout.LOGIT_SLICES_EAST, plan.by_name["lg2"].addr,
                       segments=shape.pack, nlw=shape.nlw_head, stride=shape.logit_stride)
    k_head.lm_head(f, hidden, head)
    blk.mark("head")
    clock["emitted"] = time.time()
    if measure:
        return Batch((T, L, C), layers, "", {}, k, pa.geo, sizes, tb, tensors, hidden, head, f, blk.marks, {}, cfg, shape)
    try:
        text, case = k.done("step_batch")
    except ScheduleError as e:
        if "no free word for text" in str(e) and not plan_given:
            raise ScheduleError(f"{e}; the text regions of the serve plan are the sizes measured on `ref` for the 2- and 8-layer "
                                f"models (layout.TEXT_WORDS_T8): size them for profile {p.name} and this model with "
                                "text_sizing.measured_plan and pass the plan") from e
        raise
    clock["paged"] = time.time()
    case["sites"] = k.located.table.records()
    case["golden"].update(layers=layers, bucket=[T, L, C], stage_marks=[[a, b, c] for a, b, c in blk.marks])
    seconds = {"images": clock["images"] - clock["start"], "emit": clock["emitted"] - clock["images"],
               "page": clock["paged"] - clock["emitted"]}
    return Batch((T, L, C), layers, text, case, k, pa.geo, sizes, tb, tensors, hidden, head, f, blk.marks, seconds, cfg, shape)


def lane_values(batch, lane, token, pos, table):
    """{site name: value} that make lane `lane` the token `token` at position `pos` of the request whose block table is
    `table` (block ids in order; the blocks beyond it are the null block). `token` None makes it a padding lane: the zero
    words of the embedding table, position 0, the null block for every context block, the mask of position 0, and the
    sinks for the K and V writes."""
    geo, sizes, tb = batch.geo, batch.sizes, batch.tables
    C = batch.bucket[2]
    real = token is not None
    if not real:
        token, pos, table = 0, 0, []
    else:
        table = list(table)
        if not 0 <= pos < C:
            raise ValueError(f"lane {lane}: position {pos} is outside the context 0..{C - 1} of the bucket")
        if not pos // 16 < len(table) <= C // 16:
            raise ValueError(f"lane {lane}: the block table has {len(table)} blocks; position {pos} needs {pos // 16 + 1} "
                             f"and the bucket takes at most {C // 16}")
        bad = [b for b in table if not 1 <= b <= layout.SERVE_BLOCKS - 1]
        if bad:
            raise ValueError(f"lane {lane}: block ids {bad} are outside 1..{layout.SERVE_BLOCKS - 1} "
                             "(block 0 is the null block that padding reads)")
    padded = list(table) + [0] * (C // 16 - len(table))
    out = {}
    part, local, seg = sizes.locate(token)
    for q, pt in enumerate(tb.parts):
        for pl in range(2):
            out[f"t{lane}.emb.p{q}{'lh'[pl]}"] = pt.addr + local if real and q == part else pt.zero
    for j, a in enumerate(tb.wpe_addrs):
        out[f"t{lane}.emb.wpe{j}"] = a + pos
    for pl in range(2):
        out[f"t{lane}.emb.map{pl}"] = k_embed.FIRST_MAP + (seg if real else 0)
    block, row = padded[pos // 16], pos % 16
    mk = batch.plan.by_name[f"mk{C}s{layout.MASK_SLICES[0]}"].addr
    for l in range(batch.layers):
        pre = f"t{lane}.L{l}"
        for s in layout.MASK_SLICES:
            out[f"{pre}.mk.s{s}"] = mk + pos
        for j, b in enumerate(padded):
            for s in range(kvpool.K_BASE, kvpool.K_BASE + 16):
                out[f"{pre}.kb.b{j}.s{s}"] = geo.kaddr(l, b)
            for s in geo.vslices(l):
                out[f"{pre}.vb.b{j}.s{s}"] = geo.vaddr(l, b)
        for r in range(8):
            for pl in range(2):
                out[f"{pre}.kw.p{r}.{pl}"] = geo.kaddr(l, block) + row // 8 if real and r == row % 8 else geo.kaddr(l, geo.sink)
        for s in geo.vslices(l):
            out[f"{pre}.vw.s{s}"] = geo.vaddr(l, block) + row if real else geo.vaddr(l, geo.sink)
    return out


def site_values(batch, tokens, positions, tables):
    """The values of every site of a step: lane i is the real lane `tokens[i]` at `positions[i]` with block table
    `tables[i]`; the lanes beyond the given ones are padding lanes. The first L lanes have the logit rows."""
    T = batch.bucket[0]
    n = len(tokens)
    if not len(positions) == len(tables) == n or not 1 <= n <= T:
        raise ValueError(f"{n} lanes (positions {len(positions)}, tables {len(tables)}) in a bucket of T = {T}")
    out = {}
    for t in range(T):
        out.update(lane_values(batch, t, *((tokens[t], positions[t], tables[t]) if t < n else (None, 0, []))))
    return out
