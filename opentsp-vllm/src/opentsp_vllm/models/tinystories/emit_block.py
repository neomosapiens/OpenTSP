"""One transformer block of the TinyStories model (layer 0, one token, a preloaded K/V cache), emitted as a
hand-scheduled program on the toolchain kernels.

The program is the layer of `reference_model.layer_step` in the order of the design's kernel table: LayerNorm, the Q, K, V
projections, the K and V rows into the fixed slot 63 of the caches, the scores of 16 heads against the installed K
cache, one softmax per head, the expansion of the probabilities, the context, the output projection, the residual,
LayerNorm, the two FFN halves (up projection, GELU, down projection on the two hemispheres' planes) and the residual.
Stages run one after another (a stage starts after the one before has written its results; the kernels' own
overlap is inside a stage), so the schedule is the sum of the kernels' latencies. The instruction pager brings
the text of every queue over the IQ.

Memory (`layout.py`): west slices 0..15 hold the layer's matrices of the west plane (`w0s<slice>`), east
slices 44..59 those of the east plane, the K cache (`kc0s<slice>`, install layout, slot 63 the current token) and the
V cache lives on west quad 5 (`vc0s<slice>`, word `slot`). Constants, the token's vector and the host mask are
vectors on the quads of the vector `Flow` allocator (words from 0x80); the reduce-and-broadcast scratch words
are at 0x400 (bank 0 of east slices 44..47) and in bank 1 of west slices 0..3.
"""
import contextlib
import functools
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from opentsp_tools.programs.case import Kit
from opentsp_tools.programs import data
from opentsp_tools.programs import needs
from opentsp_tools import pager
from opentsp_tools.programs import ref
from opentsp_tools.programs.kernels import context as k_context, expand as k_expand, gelu as k_gelu, install as k_install
from opentsp_tools.programs.kernels import layernorm as k_layernorm
from opentsp_tools.programs.kernels import matvec_f16 as k_matvec, reduce_bcast as k_rb, scores as k_scores, softmax as k_softmax
from opentsp_tools.programs.kernels import softmax_packed as k_spk, write_row as k_write_row
from opentsp_tools.programs.kernels.flow import Flow, Vec
from opentsp_tools.stations import ScheduleError
from opentsp_tools.programs.trial import Trial
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import layout
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import PROMPT_IDS, TINYSTORIES

CFG = TINYSTORIES
SEED = 1
POSITIONS = (0, 15, 40, 62)
LAYER = 0
GAP = 12             # cycles between a stage's last write and the next stage's first read
SCRATCH = 0x400      # first word of the reduce-and-broadcast scratch (east slices 44..47, bank 0)
EARLY_INSTALL = True   # the matvec weights are installed before x is ready (the plane's own state decides), not behind the cursor
EARLY_EXPAND = True    # the expansion starts when its heads are readable, not a fixed lead behind the cursor
EARLY_CONTEXT = True   # the context's multiplies start when the first expanded vector is readable (slot s at t0 + s)
EARLY_PLACE = True     # the heads' packing starts as each score is stored, not behind the cursor
PACKED = True        # the paged steps pack several heads per softmax call (opentsp_tools.programs.kernels.softmax_packed.py)
TRIALS, TRIAL_STEP = 600, 2      # start cycles a packed softmax call tries

# sha256 of the expected words of each position (test_block.py: `digest`), pinned from bit-exact reference
PINNED = {
    0: "7577e4945bb9994f8ceb4a708bfc3f778a60a3e7e574b23b91a735e7468c0367",
    15: "6703cf41fcea7b53056cf0068d49223f54833c4d9167c1bac0edc8cf542148b1",
    40: "c02223ca297bc1be905f422da6fcf05f464d41f97f403e07986b5b8a0f060ced",
    62: "7917b6ea55a1ecd5d7f177ad8f6cbcd0d31dd8816d35422b42dbad557dbd237c",
}


@dataclass
class BlockInputs:
    cfg: object
    pos: int
    token: int
    layer: dict
    x0: np.ndarray           # (H) f32: wte[token] + wpe[pos]
    k_slots: np.ndarray      # (slots, H) f16: cached keys in slots 0..pos-1, zero elsewhere
    v_slots: np.ndarray      # (slots, H) f32
    mask: np.ndarray         # (slots) f32: 0 for slots 0..pos-1 and the current slot, -inf elsewhere


@functools.lru_cache(maxsize=None)
def weights(seed=SEED):
    return W.synthetic(CFG, seed)


def block_inputs(pos, seed=SEED, w=None):
    """Layer 0 of the synthetic weights, the embedded token at position `pos`, and a random cache of `pos` positions
    (K rows N(0, 0.5^2) rounded to f16, V rows N(0, 1)) in the layout the host preloads ."""
    cfg, w = CFG, (w or weights(seed))
    S, H = cfg.slots, cfg.hidden
    token = PROMPT_IDS[pos % len(PROMPT_IDS)]
    k = np.zeros((S, H), np.float16)
    v = np.zeros((S, H), np.float32)
    k[:pos] = (W.gauss(seed, f"block.k.{pos}", (pos, H)) * 0.5).astype(np.float16)
    v[:pos] = W.gauss(seed, f"block.v.{pos}", (pos, H)).astype(np.float32)
    mask = np.full(S, -np.inf, np.float32)
    mask[:pos] = 0
    mask[S - 1] = 0
    return BlockInputs(cfg, pos, token, w.layers[LAYER], exact_ref.embed(w, token, pos), k, v, mask)


def pad(a, n=320, fill=0):
    out = np.full(n, fill, a.dtype)
    out[:len(a)] = a
    return out


def expected_words(inp):
    """{name: [byte plane words]} of what the chip writes: the layer's output x (4 planes), the current K row (2 planes
    of the f16 row) and the current V row (4 planes)."""
    parts = {}
    out, k_cur, v_cur = exact_ref.layer_step(inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask, parts)
    return {"x1": data.to_planes(pad(out)), "k": data.to_planes(pad(k_cur)), "v": data.to_planes(pad(v_cur))}


def expected_parts(inp):
    """The intermediate vectors of bit-exact reference's layer step (the block dumps them too), by name."""
    parts = {}
    exact_ref.layer_step(inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask, parts)
    return parts


# -- the program ----------------------------------------------------------------------------------------------------
# map ids of the 3-round softmax tree (C = 128): base + 1.. = 9..11, above the embedding's 4..8 (ids 1..3 are the LayerNorm's rotations and
# the Distribute map); the maps' words (0x60 + base + i of west slices 2 and 3) are in `kfix`.
SM_BASE = 8


class BlockKit(Kit):
    """A Kit whose program is paged on `done`: every queue over the IQ becomes fetch-driven, the blocks are weights."""

    case_tags = ["ts-block", "sim"]
    regions = None          # {slice: (first word, end word)}: where the pager may put the text blocks (the plan's `tx<slice>`)
    located = None          # the pager.Paged of the program's patch sites, when it has any

    def done(self, fn):
        view = pager.CycleView.of(self.b.prog)
        insns = {}
        for _, q, _t in view.entries:
            insns[q] = insns.get(q, 0) + 1
        view2, blocks, externs = pager.page(view, self.p, self.b.t, regions=self.regions)
        self.b.t.check()
        self.paged = sorted({blk.queue for blk in blocks})
        self.view2 = view2
        self.blocks = blocks
        self.b.prog = view2.program(self.p)
        for name, words in pager.tensor_words(blocks).items():
            self.weights[name] = ("raw", b"".join(bytes(w) for w in words))
        # the figures of the report: instructions of each queue, and the units of its initial image and full text
        prog = pager.tsp_as.assemble(self.b.prog.text(), config=self.p.path)[1]
        full = pager.follow(view2, blocks, self.p, prog)
        self.stats = {"instructions": {str(q): n for q, n in sorted(insns.items())},
                      "initial_units": {str(q): len(u) for q, u in sorted(prog.images.items())},
                      "text_units": {str(q): len(u) for q, u in sorted(full.items())}}
        if view2.sites:
            self.located = pager.locate(view2, blocks, self.p)
        text, case = super().done(fn)
        case["expect"]["lints"] = []
        cap = pager.Geometry(self.p).cap
        case["golden"].update(paged_queues=self.paged, blocks=len(blocks), icu_cap_units=cap,
                              max_initial_units=max(self.stats["initial_units"].values()), **self.stats)
        case["budget"]["cycles"] = self.last + 48
        case["tags"] = list(self.case_tags)
        return text, case


class Block:
    """The emitter: one object holds the kit, the vector allocator and the cursor."""

    def __init__(self, p, inp, kit=None, plan=None, layer=LAYER, full=True, paged=None):
        """`kit` and `plan` are given when several layers share one program (emit_step.py: `layer` is then the layer
        of `inp`, `full` False dumps only what the host reads: the layer's output and its K and V rows). `paged` (a
        `emit_step_batch.PagedAttention`) switches the attention to the paged pool: the program carries no step data,
        only patch sites ."""
        self.p, self.inp, self.L, self.full = p, inp, layer, full and paged is None
        self.cfg = inp.cfg
        self.paged = paged
        self.C = paged.C if paged else self.cfg.slots         # context slots
        self.bound = set()
        self.plan = plan or layout.plan_tinystories(CFG, layout.load_profile(p.name))
        self.k = kit or BlockKit(p, f"model/ts_block_p{inp.pos:02d}", f"layer 0, one token at position {inp.pos}", SEED,
                                 module="model", source="numpy-twin")
        self.f = paged.flow(self.k) if paged else Flow(self.k)
        self.f.avoid = EARLY_INSTALL     # work placed ahead of the cursor: the result streams are checked against what runs beside them
        self.b = self.k.b
        self.bank1 = 1 << p.bank_bit
        self.t = 0
        self.rb_words = 0            # reduce_bcast scratch words handed out
        self.west_quads = {}
        self.step = kit is not None
        self.maps = None
        self.marks = []              # (layer, stage, cycle at the end of the stage)
        self.sfx = f"_{layer}" if self.step else ""      # the layer's own constants are told apart in a step
        self.shared_vecs = {}
        self.scratch_base = SCRATCH

    def pad(self, a, fill=0):
        """`a` as a vector of the profile's lanes."""
        return pad(a, self.p.vlanes, fill)

    # -- placement
    def const(self, name, arr, hemi="W", quad=1):
        return self.f.const(name, self.pad(np.asarray(arr)), hemi, quad)

    def scratch(self):
        """(a0, out_addr) of the next reduce_bcast: eight words from SCRATCH on east slices 44..47 and a pair of bank 1
        words on west slices 0..3 (reduce_bcast uses a0 .. a0 + 3, two of them in bank 1, and `out_addr`)."""
        n = self.rb_words
        self.rb_words += 1
        return self.scratch_base + 8 * n, self.bank1 + self.scratch_base + n

    def after(self, *vecs):
        """The cursor: a stage starts GAP cycles after everything so far has been written."""
        self.t = max([self.t, self.k.last] + [v.ready for v in vecs if v is not None]) + GAP
        return self.t

    def dump(self, v, quad, arr, hemi=None, keep=False, gap=True):
        """Store the Value `v` on `quad` and dump and expect the f32 vector arr (H lanes, zero beyond); a step dumps
        only the vectors the host reads (`keep`). `gap` False leaves the cursor alone: the caller of a group of
        independent stores moves it once with `after` (each store would otherwise add its own GAP)."""
        out = self.f.store(v, quad, expect=self.pad(np.asarray(arr)) if (self.full or keep) and not self.paged else None,
                           hemi=hemi)
        if gap:
            self.after(out)
        return out

    def shared(self, name, make):
        """A constant every layer uses (bound once, the first time a layer asks for it)."""
        if name not in self.shared_vecs:
            self.shared_vecs[name] = make()
        return self.shared_vecs[name]

    def lconst(self, name, arr, hemi="W", quad=1):
        """A constant of this layer's weights."""
        return self.const(name + self.sfx, arr, hemi, quad)

    def matrix(self, plane, off, rows):
        """(plane, word address, text(j, i)) of the matrix at word `off` of the plane's `w0s<slice>` tensors."""
        base = 0 if plane == 0 else 44
        addr = self.plan.by_name[f"w{self.L}s{base:02d}"].addr + off
        return addr, lambda j, i: f"w{self.L}s{base + j:02d}+{off + i:#x}"

    def plane_state(self, plane):
        """The last dispatch cycle of each MXM instruction kind on `plane`'s three queues so far (-1: none yet): the matrices
        and results of the plane's earlier users are in the program text, so the next user's LW, IW and ABC are placed
        against them."""
        lw = k_install.plane_queues(plane)[2]
        ids = {lw, lw + 1, lw + 2}
        last = {"lw": -1, "iw": -1, "abc": -1, "acc": -1}
        for c, items in self.b.prog.bundles.items():
            for q, t in items:
                if pager.tsp_as.queue_id(q) in ids:
                    op = t.split()[0].split(".")[0]
                    if op in last and c > last[op]:
                        last[op] = c
        return last

    def mv(self, rows, x, bias, plane, off, to="E", alu=0):
        """The matrix-vector product of `matvec` for the Vec `x`. With EARLY_INSTALL the weights are loaded so that the first
        ABC meets the narrowed x as soon as x can be read (the matrix does not wait for x), against the plane's earlier
        users (an IW after their last ABC, an LW after their IW); else the cursor decides, as before."""
        addr, text = self.matrix(plane, off, rows)
        if not EARLY_INSTALL:
            return k_matvec.matvec(self.f, rows, x, bias, plane, addr, text, self.after(x), alu, to)
        f, p = self.f, self.p
        st = self.plane_state(plane)
        lw = k_install.plane_queues(plane)[2]
        row_in = "W" if plane < 2 else "E"
        nlw, d_lw, d_iw = rows // 8, p.timing("lw")[0], p.timing("iw")[0]
        t_abc = f.bound(x, "conv_f32_f16.f32", 88) + p.timing("conv_f32_f16.f32")[0] + f.transit(lw + 1, row_in, alu) - p.timing("abc")[1]
        install = nlw - 1 + d_lw
        t_best = max(t_abc - install - d_iw, st["iw"] + 1, st["abc"] + 2 - install)
        for dt in [-d for d in range(0, 40)] + list(range(1, 400)):
            t0 = t_best + dt
            if t0 < max(st["iw"] + 1, st["abc"] + 2 - install, 0):
                continue
            with Trial(f, [self]) as tri:
                out = k_matvec.matvec(f, rows, x, bias, plane, addr, text, t0, alu, to)
            if tri.ok:
                return out
        raise ScheduleError(f"matvec on plane {plane}: no start cycle from {t_best}")

    def expand(self, probs, one_hots):
        """`expand` of the heads' probabilities. With EARLY_EXPAND the first LW is at the earliest cycle at which the heads it
        converts can be read (and the plane's own state allows), tried from there until the program accepts it; else the
        cursor and a fixed lead decide, as before."""
        f, quads = self.f, ((4, 5) if self.paged else 4)
        if not EARLY_EXPAND:
            return k_expand.expand(f, probs, one_hots, self.cfg.dk, self.after() + 70, quads, hemi="E")
        st = self.plane_state(k_expand.PLANE)
        lws = len(probs) * self.cfg.dk // 8
        install = lws - 1 + self.p.timing("lw")[0]
        t_best = max(k_expand.earliest(f, probs, self.cfg.dk), st["iw"] + 1, st["abc"] + 2 - install, 0)
        for dt in range(400):
            with Trial(f, [self]) as tri:
                out = k_expand.expand(f, probs, one_hots, self.cfg.dk, t_best + dt, quads, hemi="E")
            if tri.ok:
                return out
        raise ScheduleError(f"expand: no start cycle from {t_best}")

    def context_start(self, ps):
        """The first t0 at which `context` can multiply slot s at t0 + `slot_cycle(s)` (t0 + s on `ref`): each expanded vector is readable by then, and its
        Read meets no Write of the expansion still running (a slot, or a bank, of its slices in that cycle)."""
        f, tr = self.f, self.b.t
        at = [k_context.slot_cycle(self.p, s_) for s_ in range(len(ps))]
        t0 = max(f.bound(v, "mul.f32", 88 + s_ % k_context.ADDERS) - at[s_] for s_, v in enumerate(ps))
        while True:
            ok = True
            for s_, v in enumerate(ps):
                queue = 88 + s_ % k_context.ADDERS
                t_rd = self.b.read_cycle_for(t0 + at[s_], "mul.f32", queue, v.slices[0], "W")
                if not (tr.free(queue, t0 + at[s_]) and all(tr.free(sl, t_rd, "R") and tr.bank_ok(sl, t_rd, "R", v.addr)
                                                         for sl in v.slices)):
                    ok = False
                    break
            if ok:
                return t0
            t0 += 1

    # -- weights and caches bound from the case's package
    def bind(self):
        if self.L in self.bound:
            return
        self.bound.add(self.L)
        inp, p, w, plan = self.inp, self.p, self.k.weights, self.plan
        lw, half = inp.layer, self.cfg.half
        mats = {"west": [("wq", lw["wq"]), ("wv", lw["wv"]), ("wo", lw["wo"]), ("up0", lw["up_w"][:half]),
                         ("dn0", lw["dn_w"][:, :half])],
                "east": [("wk", lw["wk"]), ("up1", lw["up_w"][half:]), ("dn1", lw["dn_w"][:, half:])]}
        for hemi, group in (("west", layout.WEST_W), ("east", layout.EAST_W)):
            per = {s: [] for s in group}
            for name, m in mats[hemi]:
                for j, words in enumerate(k_install.weight_planes(m, p.vlanes)):
                    per[group[j]] += words
            for s in group:
                r = plan.by_name[f"w{self.L}s{s:02d}"]
                assert len(per[s]) == r.words
                self.b.prog.extern(r.name, s, r.addr, r.words)
                w[r.name] = ("raw", b"".join(per[s]))
        if self.paged:
            return
        kc, vc = plan.by_name[f"kc{self.L}"], plan.by_name[f"vc{self.L}"]
        self.kc_addr, self.vc_addr = kc.addr, vc.addr
        for j, words in enumerate(k_install.weight_planes(inp.k_slots, p.vlanes)):
            s = layout.EAST_W[j]
            self.b.prog.extern(f"kc{self.L}s{s}", s, kc.addr, kc.words)
            w[f"kc{self.L}s{s}"] = ("raw", b"".join(words))
        planes = [data.to_planes(self.pad(row)) for row in inp.v_slots]
        for j, s in enumerate(layout.VCACHE_QUAD):
            self.b.prog.extern(f"vc{self.L}s{s}", s, vc.addr, vc.words)
            w[f"vc{self.L}s{s}"] = ("raw", b"".join(pl[j] for pl in planes))

    def k_text(self, j, i):
        return f"kc{self.L}s{layout.EAST_W[j]}+{i:#x}"

    # -- the layer
    def layernorm(self, x, g, b_, parts, to="W", quad=1, keep=False):
        a0, out = self.scratch()
        a1, out1 = self.scratch()
        assert a1 == a0 + 8 and out1 == out + 1
        at = self.after(x)
        y = k_layernorm.layernorm(self.f, x, g, b_, self.inv_n, self.eps, self.maps, a0, out, to=to, at=at)
        return self.dump(y, quad, parts, hemi=to, keep=keep)

    def begin(self, inp, layer):
        """Switch to the layer of `inp` (a step runs the layers one after another in one program)."""
        self.inp, self.L = inp, layer
        self.sfx = f"_{layer}" if self.step else ""
        self.rb_words = 0            # the scratch words are free again: the stages of a layer are done with them

    def mark(self, stage):
        """Record the cycle at which `stage` of the current layer has written its results (the report's stage table)."""
        self.marks.append((self.L, stage, max(self.t, self.k.last)))

    def common(self):
        """The reduce-and-broadcast maps and the constants of every layer, bound once."""
        if self.maps is None:
            self.maps = (k_rb.load_maps(self.k, 60), k_rb.load_distribute_map(self.k, 80))
            self.inv_n = self.f.splat("invn", 1 / self.cfg.hidden, "W", 3)
            self.eps = self.f.splat("eps", self.cfg.ln_eps, "W", 3)
            self.t = self.maps[1] + GAP
            self.sm_maps = self.maps
            if self.C == 128:           # the softmax of 8 superlanes: a three-round tree beside the LayerNorm's two
                self.sm_maps = (k_rb.load_maps(self.k, 62, 3, base=SM_BASE), self.maps[1], 3, SM_BASE, k_rb.DIST_MAP)
                self.t = max(self.t, self.sm_maps[0] + GAP)

    def run(self, stop=None, x=None, to="E", out_quad=8):
        """The layer of `inp`: x is the input Vec (layer 0 of a block: the host's x0); `to` the row the output goes to
        (the hemisphere it is stored on) and `out_quad` its quad. Returns the output Vec."""
        st = self.run_a(stop, x, to, out_quad)
        return None if st is None else self.run_b(st, stop)

    def run_a(self, stop=None, x=None, to="E", out_quad=8):
        """The first half of the layer: LayerNorm 1, the Q, K, V projections and the writes of the new K and V rows. A step
        of several lanes runs every lane's first half before any lane's second (the lanes read the rows the others wrote).
        Returns the state `run_b` takes, or None at `stop`."""
        f, k, inp = self.f, self.k, self.inp
        lw = inp.layer
        parts = NullParts() if self.paged else expected_parts(inp)
        self.bind()
        self.common()
        hemi = "W" if to == "W" else "E"
        if x is None:
            x = self.const("x0", inp.x0, "W", 1)
        # LayerNorm 1 and the Q, K, V projections
        xn = self.layernorm(x, self.lconst("ln1g", lw["ln1g"], "W", 2), self.lconst("ln1b", lw["ln1b"], "W", 2),
                            parts["xn"], quad=1)
        self.mark("xn")
        if stop == "xn":
            return
        q = self.mv(self.cfg.hidden, xn, self.lconst("bq", lw["bq"], "W", 2), 0, layout.matrix_offset(self.cfg, "wq"))
        q = self.dump(q, 5, parts["q"])
        kk = self.dump(self.mv(self.cfg.hidden, xn, self.lconst("bk", lw["bk"], "W", 2), 2, layout.matrix_offset(self.cfg, "wk")), 6, parts["k"])
        if self.paged:
            k.last = max(k.last, self.paged.write_k(self, kk))
        else:
            k.last = max(k.last, k_write_row.write_k(f, kk, self.cfg.slots - 1, addr=self.kc_addr))
        vv = self.mv(self.cfg.hidden, xn, self.lconst("bv", lw["bv"], "W", 2), 0, layout.matrix_offset(self.cfg, "wv"), to="W")
        if self.paged:
            k.last = max(k.last, self.paged.write_v(self, vv))
        else:
            k.last = max(k.last, self.b.write(vv, list(layout.VCACHE_QUAD), [self.vc_addr + self.cfg.slots - 1] * 4))
            for j, pl in enumerate(data.to_planes(self.pad(parts["v"]))):
                k.expect(layout.VCACHE_QUAD[j], self.vc_addr + self.cfg.slots - 1, pl)
            for j, pl in enumerate(data.to_planes(self.pad(ref.to_f16(parts["k"])))):
                k.expect(layout.EAST_W[2 * ((self.cfg.slots - 1) % 8) + j], self.kc_addr + (self.cfg.slots - 1) // 8, pl)
        self.after()
        self.mark("kv")
        if stop == "kv":
            return
        return SimpleNamespace(x=x, q=q, lw=lw, parts=parts, to=to, out_quad=out_quad, hemi=hemi)

    def run_b(self, st, stop=None):
        """The second half: the scores over the context, the softmax, the context, the output projection and the FFN.
        Returns the layer's output Vec, or None at `stop`."""
        f, k, p, inp = self.f, self.k, self.p, self.inp
        x, q, lw, parts, to, out_quad = st.x, st.q, st.lw, st.parts, st.to, st.out_quad
        # scores of the 16 heads against the K cache, one softmax each
        head_masks = list([self.pad(r) for r in ref.head_lane_masks(self.cfg.heads, self.cfg.hidden)])
        masks = self.shared("hm", lambda: f.table("hm", head_masks, "W", 0))
        half = self.shared("half", lambda: f.splat("half", 0.5, "W", 1))
        if self.paged:
            scores = k_scores.scores(f, self.C, q, masks, half, self.paged.mask(self), self.t,
                                     install=self.paged.install_k(self))
        else:
            add_mask = self.shared("mask", lambda: self.const("mask", inp.mask, "W", 2))
            # the products of the first half may still be reading and writing: the K install starts at the first cycle that fits
            for dt in range(400 if EARLY_INSTALL else 1):
                with Trial(f, [self]) as tri:
                    scores = k_scores.scores(f, self.cfg.slots, q, masks, half, add_mask, self.t + dt, self.kc_addr, self.k_text)
                if tri.ok:
                    break
            else:
                raise ScheduleError(f"scores: no start cycle from {self.t}")
        sc = [self.dump(v, 7 + h % 2, self.pad(parts["scores"][h]), hemi="E", gap=False) for h, v in enumerate(scores)]
        self.after(*sc)
        self.mark("scores")
        if stop == "scores":
            return
        probs = []
        if self.paged and PACKED:
            placed = k_spk.place(f, sc, self.C, 0 if EARLY_PLACE else self.after())
            pk = k_spk.packs(f, placed, self.C, (9, 10))
            pp = []
            for j, s_ in enumerate(pk):
                a0, out = self.scratch()
                self.scratch()
                # the calls overlap: each starts at the first cycle from its scores' readiness at which it fits
                t_sm = f.bound(s_, "reduce.max.f32", 88)
                for _ in range(TRIALS):
                    with Trial(f, [self]) as tri:
                        pv = k_softmax.softmax(f, s_, self.sm_maps, a0, out, e_quad=9 + j % 2, to="E", at=t_sm)
                        pd = self.dump(pv, 6 + (j * k_spk.per_vector(self.C, p.vlanes)) % 2, None, hemi="E")  # head h on quad 6 + h mod 2
                    if tri.ok:
                        break
                    t_sm += TRIAL_STEP
                else:
                    raise ScheduleError(f"softmax pack {j} found no start in {TRIALS} trials")
                pp.append(pd)
            probs = k_spk.unpack(f, pp, self.C, self.cfg.heads, (6, 7))
            self.after(*probs)
        for h in range(self.cfg.heads if not (self.paged and PACKED) else 0):
            a0, out = self.scratch()
            self.scratch()
            pv = k_softmax.softmax(f, sc[h], self.sm_maps, a0, out, e_quad=9 + h % 2, to="E", at=self.after(sc[h]))
            # the dummy segments (lanes 64..319, scores 0) softmax to 1/64; only lanes 0..63 are live
            want = None if self.paged else ref.softmax(self.pad(parts["scores"][h]))
            assert self.paged or np.array_equal(want[:self.cfg.slots], parts["probs"][h])
            probs.append(self.dump(pv, 6 + h % 2, want, hemi="E"))
        self.mark("softmax")
        if stop == "softmax":
            return
        one_hots = self.shared("oh", lambda: f.table("oh", k_expand.one_hot(self.C, p.vlanes), "W", 3))
        with self.paged.scope("ps") if self.paged else contextlib.nullcontext():
            ps = self.expand(probs, one_hots)
        k.last = max(k.last, ps[-1].ready)
        if self.full:
            pexp = ref.expand(parts["probs"], self.cfg.dk)
            for s_, v in enumerate(ps):
                for sl, plane in zip(v.slices, data.to_planes(self.pad(pexp[s_]))):
                    k.expect(sl, v.addr, plane)
        self.mark("expand")
        if stop == "expand":
            return
        zero = self.shared("zero", lambda: f.splat("zero", 0.0, "W", 3))
        t_ctx = self.after(ps[-1]) if not EARLY_CONTEXT else self.context_start(ps)
        if self.paged:
            vs = self.paged.v_values(self, t_ctx)
        else:
            vs = [Vec(list(layout.VCACHE_QUAD), self.vc_addr + s_, 0, "W") for s_ in range(self.cfg.slots)]
        z, _ = k_context.context(f, ps, vs, zero, t_ctx, (5, 6, 7, 8))
        ctx = self.dump(z, 10, parts["ctx"], hemi="E")
        self.mark("context")
        if stop == "context":
            return
        return self.finish_layer(x, ctx, parts, lw, to, out_quad, stop)

    def finish_layer(self, x, ctx, parts, lw, to, out_quad, stop=None):
        """Output projection, residual, LayerNorm 2 and MLP after attention."""
        f, hemi = self.f, to
        ao = self.dump(self.mv(self.cfg.hidden, ctx, self.lconst("bo", lw["bo"], "W", 2), 0, layout.matrix_offset(self.cfg, "wo")), 9, parts["ao"], hemi="E")
        h1 = self.dump(f.vxm(0, "add.f32", [x, ao], to="W"), 6, parts["h1"], hemi="W")
        self.mark("h1")
        xn2 = self.layernorm(h1, self.lconst("ln2g", lw["ln2g"], "W", 2), self.lconst("ln2b", lw["ln2b"], "W", 2),
                             parts["xn2"], quad=1)
        self.mark("ln2")
        if stop == "ln2":
            return
        # the FFN: halves on the two planes
        consts = self.shared("gelu", lambda: [f.splat(n, v, "W", 3) for n, v in (
            ("gc0", ref.GELU_C0), ("gc1", ref.GELU_C1), ("one", ref.ONE), ("hf", ref.HALF))])
        d = []
        for i, (plane, off_up, off_dn, hemi_i) in enumerate(((0, layout.matrix_offset(self.cfg, "up0"), layout.matrix_offset(self.cfg, "dn0"), "E"),
                                                             (2, layout.matrix_offset(self.cfg, "up1"), layout.matrix_offset(self.cfg, "dn1"), "W"))):
            u_b = self.lconst(f"upb{i}", lw["up_b"][i * self.cfg.half:(i + 1) * self.cfg.half], "W", 2)
            u = self.dump(self.mv(self.cfg.half, xn2, u_b, plane, off_up, to=hemi_i), 8, parts[f"u{i}"], hemi=hemi_i)
            g = self.dump(k_gelu.gelu_vec(f, u, *consts, to=hemi_i), 7, ref.gelu(parts[f"u{i}"]), hemi=hemi_i)
            acc = self.mv(self.cfg.hidden, g, None, plane, off_dn)
            d.append(self.dump(acc, 4 + 2 * i, parts[f"d{i}"], hemi="W" if plane else "E"))
        self.mark("ffn")
        if stop == "ffn":
            return
        y = f.vxm(0, "add.f32", [d[0], d[1]])
        y = f.vxm(1, "add.f32", [y, self.lconst("dnb", lw["dn_b"], "W", 2)], chain=(0,))
        self.dump(y, 7, parts["y"], hemi="E")
        out = f.vxm(2, "add.f32", [h1, y], chain=(1,), to=to)
        res = f.store(out, out_quad, expect=None if self.paged else self.pad(expected_out(self.inp)), hemi=hemi)
        self.after(res)
        self.mark("out")
        return res


class NullParts(dict):
    """The intermediate vectors of a program that carries no step data (a paged step): zeros of the right shape, which
    the emitter's dumps and expectations ignore (`full` is off)."""

    def __missing__(self, key):
        return np.zeros((CFG.heads, 64), np.float32)


def pad_rows(m):
    return [pad(r) for r in m]


def expected_out(inp):
    return exact_ref.layer_step(inp.cfg, inp.layer, inp.x0, inp.k_slots, inp.v_slots, inp.mask)[0]


def build(p, pos, seed=SEED, stop=None, w=None):
    """(program text, case) of the block at position `pos` on profile `p`; `w`: weights other than the synthetic ones
    (opentsp-vllm/src/opentsp_vllm/models/tinystories/torch_reference.py binds the real checkpoint's layer 0)."""
    needs.SHOWCASE.check(p)
    blk = Block(p, block_inputs(pos, seed, w))
    blk.run(stop)
    return blk.k.done("block")
