"""Single-chip TinyStories stages. Only SRAM input words vary between tokens.

KV addresses are physical on the chip. A chunk covers eight 16-token pages.
Scores and the two PV accumulator chains stay on the chip across stage launches.
"""


from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import emit_step_batch as batch
from opentsp_vllm.models.tinystories import layout
from opentsp_tools.programs import needs
from opentsp_tools.programs import ref
from opentsp_tools import stations
from opentsp_tools.programs.kernels import expand, lm_head, scatter, softmax, wide_scores, wide_softmax
from opentsp_tools.programs.kernels.flow import Vec

COPY_QUADS = (0, 1, 2, 3, 4, 5, 9, 10)

CHUNK = 128
CHUNKS = layout.SERVE_BLOCKS * 16 // CHUNK


def fixed_layout(cfg):
    original = layout.plan_serve(cfg, layout.load_profile())
    pl = layout.Planner(original.p)
    # Keep the model's shared weight placement and the fixed kernel workspaces.
    for r in original.regions:
        if r.kind in ("weight", "head", "const", "logits", "reserved"):
            pl.add(r.name, r.kind, r.slices, r.words, r.rw, r.exported, addr=r.addr)
    pl.add("keys", "pool", layout.EAST_W, cfg.layers * (CHUNK * CHUNKS // 8) + 1, True, bank=1)
    for layer in range(cfg.layers):
        q = (4, 5, 9, 10)[layer // 2]
        pl.add(f"v{layer}", "pool", layout.flow_quad("E", q), CHUNK * CHUNKS, True, bank=layer % 2)
    for name, hemi, quad, words in (
        ("hidden", "E", 3, 1),
        ("query", "E", 5, 1),
        ("mask", "W", 9, CHUNKS),
        ("slotmaps", "W", 4, cfg.layers * 9),
        ("krow", "W", 0, 1),
        ("vrow", "W", 1, 1),
        ("even", "W", 3, 1),
        ("odd", "W", 4, 1),
    ):
        pl.add(name, "data", layout.flow_quad(hemi, quad), words, True, bank=1)
    for i in range(2):
        pl.add(f"scores{i}", "data", layout.flow_quad("E", 1 + i), CHUNKS // 2 * cfg.heads, True, bank=0)
        pl.add(f"probs{i}", "data", layout.flow_quad("E", 6 + i), CHUNKS * cfg.heads // 2, True, bank=1)
    for hemi in ("W", "E"):
        for quad in range(11):
            pl.add(f"tmp{hemi}{quad}", "temp", layout.flow_quad(hemi, quad), 180, True, bank=1)
    for s in range(88):
        pl.add(f"tx{s}", "text", [s], 160, True, bank=0 if s in layout.EAST_W else 1)
    pl.place()
    return pl


class FixedFlow(batch.PlannedFlow):
    def addr(self, hemi, quad, bank):
        r = self.plan.by_name[f"tmp{hemi}{quad}"]
        key = (hemi, quad)
        n = self.taken.get(key, 0)
        if n >= r.words:
            raise ValueError(f"{r.name} exceeds {r.words} scratch words")
        self.taken[key] = n + 1
        return r.addr + n


class Model:
    C = CHUNK

    def __init__(self, profile, weights):
        needs.SHOWCASE.check(profile)
        self.p, self.w, self.cfg = profile, weights, weights.cfg
        self.plan = fixed_layout(self.cfg)
        old = layout.plan_serve(self.cfg, layout.load_profile(profile.name))
        self.tensors, _ = layout.build_serve_images(self.cfg, old, weights)

    def vec(self, name, offset=0):
        r = self.plan.by_name[name]
        if not 0 <= offset < r.words:
            raise ValueError(f"{name} offset {offset} exceeds {r.words} words")
        return Vec(list(r.slices), r.addr + offset, 0, "E" if r.slices[0] >= 44 else "W")

    def flow(self, kit):
        f = FixedFlow(kit, self.plan, self.tensors)
        f.cfg = self.cfg
        return f

    def write(self, b, v, dst):
        return b.b.write(v, dst.slices[: v.width], [dst.addr] * v.width)

    def transfer(self, b, src, dst):
        row = "W" if src.hemi == "E" else "E"
        t = b.after(src)
        v = b.b.read(t, src.slices, [src.addr] * src.width, row, b.f.stream(row, src.width))
        # The VXM's bitwise OR with itself copies the bits to either hemisphere.
        v = b.f.vxm(1, "or.u32", [v, v], to=dst.hemi)
        return self.write(b, v, dst)

    def write_k(self, b, kk):
        v = b.f.vxm(0, "conv_f32_f16.f32", [kk], to="W", width=2)
        row = self.vec("krow")
        last = self.write(b, v, row)
        # Maps in each group must be read on different cycles when they share a slice.
        for pair in range(8):
            m = self.vec("slotmaps", 9 * b.L + pair)
            last = scatter.scatter_row(
                b.b,
                last + 50,
                [(s, row.addr) for s in row.slices[:2]],
                [(m.slices[0], m.addr, None, (44 + 2 * pair, 45 + 2 * pair))],
            )
        return last

    def write_v(self, b, vv):
        row = self.vec("vrow")
        last = self.write(b, vv, row)
        m = self.vec("slotmaps", 9 * b.L + 8)
        return scatter.scatter_row(
            b.b,
            max(last, b.k.last) + 50,
            [(s, row.addr) for s in row.slices],
            [(m.slices[0], m.addr, None, self.vec(f"v{b.L}").slices)],
            spare=4,
        )

    def copy_word(self, b, quad):
        """Copy four byte planes entirely on chip. Each Gather byte returns via
        the VXM to a Scatter on the same MEM slice. Both addresses are data."""
        mem = b.b
        first = 44 + 4 * quad
        m = self.vec("slotmaps")
        for byte in range(16):
            t = 50 + 60 * byte
            src = mem.read_for(t, "gather", first, [m.slices[0]], [m.addr], "E", 16)
            for plane in range(4):
                sl = first + plane
                mem._read(src, sl, "gather", t)
                mem.raw(t, sl, f"gather W:{plane}, E:16, {byte:#x}, 0x0", slot="R")
            gap, cycle = stations.commit(self.p, t, "gather", first, "W")
            v = mem._write("W", 0, 4, gap, cycle)
            result = b.f.vxm(0, "or.u32", [v, v], to="E", stream=4)
            tw = mem.when(result, "scatter", first)
            dst = mem.read_for(tw, "scatter", first, [m.slices[1]], [m.addr], "E", 17)
            for plane in range(4):
                sl = first + plane
                mem._read(result.planes(plane, 1), sl, "scatter", tw)
                mem._read(dst, sl, "scatter", tw)
                mem.raw(tw, sl, f"scatter E:{4 + plane}, E:17, {byte:#x}, 0x0", slot="W")
            b.k.last = tw

    def score_vec(self, h, c):
        return self.vec(f"scores{c % 2}", h * (CHUNKS // 2) + c // 2)

    def prob_vec(self, h, c):
        return self.vec(f"probs{h % 2}", c * (self.cfg.heads // 2) + h // 2)

    def scores(self, b, c):
        f = b.f
        f.low["E"] = wide_scores.RESULT_STREAMS
        half = f.splat("half", 0.5, "W", 1)
        masks = f.table("hm", list(eb.pad_rows(ref.head_lane_masks(self.cfg.heads, self.cfg.hidden))), "W", 0)
        addr = self.vec("keys", (b.L * CHUNKS + c) * 16).addr
        at = 100
        for first in range(0, self.cfg.heads, 4):
            out = wide_scores.tile_scores(
                f,
                CHUNK,
                self.vec("query"),
                masks[first : first + 4],
                half,
                self.vec("mask", c),
                at,
                addr,
                lambda j, i: hex(addr + i),
            )
            at = max(self.write(b, v, self.score_vec(first + h, c)) for h, v in enumerate(out)) + 90
        b.k.last = at

    def normalize(self, b):
        f = b.f
        b.common()
        at = b.t + 100
        for h in range(self.cfg.heads):
            f.taken.clear()
            out = wide_softmax.softmax_pool(
                f,
                [self.score_vec(h, c) for c in range(CHUNKS)],
                b.sm_maps,
                eb.SCRATCH,
                b.bank1 + eb.SCRATCH,
                [3 + c % 5 for c in range(CHUNKS)],
                at,
            )
            for c, v in enumerate(out):
                b.k.last = self.transfer(b, v, self.prob_vec(h, c))
            at = b.after() + 90

    def normalize_one(self, b, c):
        """One live physical chunk. Every omitted chunk contributes -inf to
        max and +0 to sum, so its 128-lane reduction is the same tree."""
        b.common()
        at = b.t + 50
        for h in range(self.cfg.heads):
            prob = softmax.softmax(b.f, self.score_vec(h, c), b.sm_maps, eb.SCRATCH, b.bank1 + eb.SCRATCH, at=at)
            at = self.write(b, prob, self.prob_vec(h, c)) + 50
        b.k.last = at

    def context(self, b, c):
        f = b.f
        oh = f.table("oh", expand.one_hot(CHUNK, self.p.vlanes), "W", 3)
        ps = expand.expand(
            f, [self.prob_vec(h, c) for h in range(self.cfg.heads)], oh, self.cfg.dk, 150, (1, 2, 3), hemi="E"
        )
        at = ps[-1].ready + 60
        destinations = [self.vec("even"), self.vec("odd")]
        acc = list(destinations)
        for i, prob in enumerate(ps):
            value = self.vec(f"v{b.L}", c * CHUNK + i)
            # Two four-cycle add chains, staggered by two cycles, carry the
            # physical-slot sums between chunks without regrouping any terms.
            safe = f.vxm(4 + i % 4, "mask.f32", [value, prob], at=at + 2 * i)
            prod = f.vxm(8 + i % 4, "mul.f32", [prob, safe], chain=(1,))
            acc[i % 2] = f.vxm(12 + i % 4, "add.f32", [prod, acc[i % 2]], chain=(0,) if i < 2 else (0, 1), to="W")
        b.k.last = max(self.write(b, v, dst) for v, dst in zip(acc, destinations))

    def emit(self, name):
        import re

        match = re.fullmatch(r"(pre|score|context|post)(\d+)(?:_(\d+))?", name)
        phase, layer, c = (match[1], int(match[2]), int(match[3] or 0)) if match else (name, 0, 0)
        k = eb.BlockKit(self.p, name, "TinyStories fixed physical KV", 1)
        k.regions = batch.text_regions(self.plan)
        inp = eb.BlockInputs(self.cfg, 0, 0, self.w.layers[layer], None, None, None, None)
        b = eb.Block(self.p, inp, kit=k, plan=self.plan, layer=layer, full=False, paged=self)
        for n in self.tensors:
            if n in self.plan.by_name and (
                n.startswith(("cs", "fhm", "foh")) or (phase == "head" and n.startswith("hd"))
            ):
                batch.bind(k, self.plan, self.tensors, n)
        x = self.vec("hidden")
        if name.startswith("copy"):
            self.copy_word(b, int(name[4:]))
        elif phase == "pre":
            st = b.run_a(x=x)
            k.last = self.transfer(b, st.q, self.vec("query"))
            zero = b.f.splat("zero", 0.0, "W", 3)
            for dst in ("even", "odd"):
                k.last = self.transfer(b, zero, self.vec(dst))
        elif phase == "score":
            self.scores(b, c)
        elif phase == "reset":
            # A completely masked chunk has these scores for every head. Active
            # chunks overwrite them after Q/K projection.
            for c in range(CHUNKS):
                for h in range(self.cfg.heads):
                    k.last = self.transfer(b, self.vec("mask", c), self.score_vec(h, c))
        elif name.startswith("single"):
            self.normalize_one(b, int(name[6:]))
        elif phase == "normalize":
            self.normalize(b)
        elif phase == "context":
            self.context(b, c)
        elif phase == "post":
            b.bind()
            b.common()
            ctx = b.f.store(b.f.vxm(0, "add.f32", [self.vec("even"), self.vec("odd")], at=b.t), 10)
            out = b.finish_layer(x, ctx, eb.NullParts(), inp.layer, "W", 1)
            k.last = self.transfer(b, out, x)
        elif phase == "head":
            b.common()
            b.sfx = "_f"
            out = b.layernorm(
                x, b.lconst("lnfg", self.w.lnf_g, "W", 2), b.lconst("lnfb", self.w.lnf_b, "W", 2), None, to="W", quad=6
            )
            r = self.plan.by_name
            head = lm_head.Head(
                r["hd0"].addr,
                r["hd44"].addr,
                layout.HEAD_INSTALLS,
                layout.LOGIT_SLICES,
                r["lg"].addr,
                layout.LOGIT_SLICES_EAST,
                r["lg2"].addr,
            )
            lm_head.lm_head(b.f, [out], head)
        else:
            raise ValueError(f"unknown stage {name}")
        k.done(name)
        return k
