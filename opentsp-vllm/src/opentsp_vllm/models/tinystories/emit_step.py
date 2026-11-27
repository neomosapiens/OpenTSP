"""One decode step of the TinyStories model as one program: the embedded token (the
host installs x0 = wte[token] + wpe[pos]), eight layers of `emit_block.Block` one after another, the final LayerNorm
and the output head as `matvec_f16` stages whose logits are stored for the host to read.

Position and cache length enter the program as data only: the host binds x0, the mask, every layer's K and V
slots and the head rows; the text is the same for every position below the cache length (`slots` - 1 cached positions
and the fixed current slot). What would change the text: the cache length (`slots`: the scores install, one softmax per
head, the expansion and the context loops), the number of head chunks, and the layer count.

The output head runs on the chip over a window of `head_chunks` * 160 vocabulary rows (the rows are data: the host binds
the window it wants, the case binds the chunk holding bit-exact reference's token). The full vocabulary is 315 chunks.
"""
from dataclasses import dataclass

import numpy as np

from opentsp_tools.programs import data
from opentsp_tools.programs import needs
from opentsp_tools.programs import ref
from opentsp_tools.programs import numerics as nx
from opentsp_tools.programs.kernels import install as k_install, matvec_f16 as k_matvec
from opentsp_vllm.models.tinystories import emit_block as eb
from opentsp_vllm.models.tinystories import reference as exact_ref
from opentsp_vllm.models.tinystories import layout
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import PROMPT_IDS, TINYSTORIES

CFG = TINYSTORIES
SEED = 1
POSITIONS = (0, 5, 62, 63)     # 63: the cache edge, slots 0..62 live and the current slot 63
CHUNK = layout.HEAD_CHUNK
HEAD_CHUNKS = 4                  # two chunks per MXM plane: a window of 640 rows
MAX_HEAD_CHUNKS = 48             # the largest head the cases run (test_step.py); 100 also runs (test_layout.py)
NLW = CHUNK // 8                 # LW cycles of one chunk's install
pad = eb.pad

# sha256 of the expected words of each position (test_step.py: `digest`), pinned from bit-exact reference
PINNED_TEXT = "a13d2b014b8219214c62fefc349cc1856dec191c17d428600ec1e35931202956"   # text without its comment
PINNED = {
    0: "75b8749939cf43d576ca8d50ff307e1f83ab27357ccd973fcfd474395ada4165",
    5: "f74b0313ef1271d0eba67699cc132747e1a2579a2a1fab93c9d5babb3591a0c2",
    62: "3aa780171380f04b8a8c5df1181b64a740856c5dc1fc08bee94fa5b32f19f0ba",
    63: "e6fe038558c7f05b5faead457b1d54ea286fff3d9a0d4431f581cf76a9356823",
}


@dataclass
class StepInputs:
    cfg: object
    w: object
    pos: int
    token: int
    x0: np.ndarray            # (H) f32: wte[token] + wpe[pos]
    k_slots: np.ndarray       # (L, slots, H) f16: cached keys in slots 0..pos-1, zero elsewhere
    v_slots: np.ndarray       # (L, slots, H) f32
    mask: np.ndarray          # (slots) f32
    head_lo: int              # first vocabulary row of the head window
    head_chunks: int
    argmax: int               # bit-exact reference's token: the argmax of the full vocabulary


def head_logits(w, hidden):
    """The f16 output head over the whole vocabulary, as the chip computes it (`matvec_f16` per 160 rows: the hidden
    state rounded to f16, the exact dot rounded once to f32)."""
    h16 = ref.to_f16(hidden)
    return np.concatenate([nx.dot_f16(w.wte[i:i + CHUNK], h16, ref.LANES) for i in range(0, len(w.wte), CHUNK)])


def head_argmax(w, hidden):
    return int(np.argmax(head_logits(w, hidden)))


def step_inputs(pos, seed=SEED, w=None, head_chunks=HEAD_CHUNKS):
    """The decode step at `pos` on the synthetic weights (or `w`). A position inside the prompt has the cache bit-exact reference's own
    steps wrote for the prompt tokens before it; a later one a random cache of `pos` positions (K rows N(0, 0.5^2)
    rounded to f16, V rows N(0, 1)) in the layout the host preloads , like the block's cases."""
    cfg, w = CFG, (w or eb.weights(seed))
    S, H, L = cfg.slots, cfg.hidden, cfg.layers
    token = PROMPT_IDS[pos % len(PROMPT_IDS)]
    k = np.zeros((L, S, H), np.float16)
    v = np.zeros((L, S, H), np.float32)
    if pos < len(PROMPT_IDS):
        kv = exact_ref.HostKV(cfg)
        for i in range(pos):
            exact_ref.decode_step(w, kv, PROMPT_IDS[i], i, logits=False)
        k[:, :pos], v[:, :pos] = kv.k, kv.v
    else:
        for l in range(L):
            k[l, :pos] = (W.gauss(seed, f"step.k.{l}.{pos}", (pos, H)) * 0.5).astype(np.float16)
            v[l, :pos] = W.gauss(seed, f"step.v.{l}.{pos}", (pos, H)).astype(np.float32)
    mask = np.full(S, -np.inf, np.float32)
    mask[:pos] = 0
    mask[S - 1] = 0
    res = exact_ref.decode_step_slots(w, token, pos, k, v, mask, logits=False)
    am = head_argmax(w, res.hidden)
    lo = min(am // CHUNK * CHUNK, cfg.vocab - head_chunks * CHUNK)
    return StepInputs(cfg, w, pos, token, exact_ref.embed(w, token, pos), k, v, mask, lo, head_chunks, am)


def layer_inputs(inp):
    """The per-layer BlockInputs of the step: layer l's input vector is bit-exact reference's output of layer l - 1."""
    out, x = [], inp.x0
    for l, lw in enumerate(inp.w.layers):
        out.append(eb.BlockInputs(inp.cfg, inp.pos, inp.token, lw, x, inp.k_slots[l], inp.v_slots[l], inp.mask))
        x = exact_ref.layer_step(inp.cfg, lw, x, inp.k_slots[l], inp.v_slots[l], inp.mask)[0]
    return out


def head_window(inp):
    return inp.w.wte[inp.head_lo:inp.head_lo + inp.head_chunks * CHUNK]


def expected_words(inp):
    """{name: [byte plane words]} of what the chip writes: per layer l the output `x<l+1>`, the current K row `k<l>`
    (f16) and V row `v<l>`; the final hidden state `hidden`; the logits `logits<c>` of each chunk of the head window."""
    out = {}
    for l, li in enumerate(layer_inputs(inp)):
        x, k_cur, v_cur = exact_ref.layer_step(li.cfg, li.layer, li.x0, li.k_slots, li.v_slots, li.mask)
        out[f"x{l + 1}"], out[f"k{l}"], out[f"v{l}"] = (data.to_planes(pad(a)) for a in (x, k_cur, v_cur))
    hidden = ref.layernorm(x, inp.w.lnf_g, inp.w.lnf_b, inp.cfg.ln_eps)
    out["hidden"] = data.to_planes(pad(hidden))
    h16 = ref.to_f16(hidden)
    rows = head_window(inp)
    for c in range(inp.head_chunks):
        out[f"logits{c}"] = data.to_planes(pad(nx.dot_f16(rows[c * CHUNK:(c + 1) * CHUNK], h16, ref.LANES)))
    return out


class StepKit(eb.BlockKit):
    case_tags = ["ts-step", "sim"]


class Step:
    def __init__(self, p, inp):
        if not 0 < inp.head_chunks <= MAX_HEAD_CHUNKS:
            raise ValueError(f"head_chunks must be 1..{MAX_HEAD_CHUNKS}")
        self.p, self.inp = p, inp
        self.plan = layout.plan_tinystories(CFG, layout.load_profile(p.name), inp.head_chunks)
        self.k = StepKit(p, f"model/ts_step_p{inp.pos:02d}", f"decode step at position {inp.pos}", SEED,
                         module="model", source="numpy-twin")
        self.layers = layer_inputs(inp)
        self.blk = eb.Block(p, self.layers[0], kit=self.k, plan=self.plan, layer=0, full=False)
        head_end = max((r.addr + r.words for r in self.plan.regions if r.kind == "head"), default=0)
        self.blk.f.base = max(0x200, -(-head_end // 16) * 16)

    def head(self, hidden, hidden_arr):
        """The output head: chunk c on MXM plane 2 * (c % 2) as the matrix of rows 160 c .. of the window, installed
        from the `hd<slice>` tensors; each chunk's logits are stored (dumped) for the host."""
        blk, f, k, p, inp = self.blk, self.blk.f, self.k, self.p, self.inp
        rows = head_window(inp)
        per = {s: [] for s in layout.WEST_W + layout.EAST_W}
        for c in range(inp.head_chunks):
            plane = 2 * (c % 2)
            group = layout.WEST_W if plane == 0 else layout.EAST_W
            for j, words in enumerate(k_install.weight_planes(rows[c * CHUNK:(c + 1) * CHUNK], p.vlanes)):
                per[group[j]] += words
        for s, words in per.items():
            if words:
                r = self.plan.by_name[f"hd{s}"]
                assert len(words) == r.words
                blk.b.prog.extern(r.name, s, r.addr, r.words)
                k.weights[r.name] = ("raw", b"".join(words))
        h16 = ref.to_f16(hidden_arr)
        f.low = {"E": 16, "W": 16}             # the installs of the two planes run on streams 0..15 of their rows
        t0 = blk.after(hidden)
        free = {0: t0, 2: t0}                  # first cycle a plane's next install may start
        out = []
        for c in range(inp.head_chunks):
            plane = 2 * (c % 2)
            base = 0 if plane == 0 else 44
            off = NLW * (c // 2)
            addr = self.plan.by_name[f"hd{base}"].addr + off
            acc = k_matvec.matvec(f, CHUNK, hidden, None, plane, addr,
                                  lambda j, i, o=off, b=base: f"hd{b + j}+{o + i:#x}", free[plane], alu=plane)
            free[plane] += NLW + p.timing("lw")[0]       # the next install starts after this one's IW (one queue slot)
            want = nx.dot_f16(rows[c * CHUNK:(c + 1) * CHUNK], h16, ref.LANES)
            out.append(f.store(acc, 10, expect=pad(want), hemi="E" if plane == 0 else "W"))
        return out

    def run(self):
        blk, inp = self.blk, self.inp
        x = None
        for l, li in enumerate(self.layers):
            blk.begin(li, l)
            x = blk.run(x=x, to="W", out_quad=1)
        lw = inp.w
        hidden_arr = ref.layernorm(layer_out(self.layers[-1]), lw.lnf_g, lw.lnf_b, inp.cfg.ln_eps)
        blk.begin(self.layers[-1], CFG.layers)
        blk.sfx = "_f"
        hidden = blk.layernorm(x, blk.lconst("lnfg", lw.lnf_g, "W", 2), blk.lconst("lnfb", lw.lnf_b, "W", 2),
                               hidden_arr, to="W", quad=6, keep=True)   # not on a weight slice: installs read those
        self.blk.mark("final_ln")
        self.head(hidden, hidden_arr)


def layer_out(li):
    return exact_ref.layer_step(li.cfg, li.layer, li.x0, li.k_slots, li.v_slots, li.mask)[0]


def build(p, pos, seed=SEED, w=None, head_chunks=HEAD_CHUNKS):
    """(program text, case) of the decode step at `pos` on profile `p`; `w`: weights other than the synthetic ones (the
    real checkpoint, tests/showcase/tinystories/check_step.py)."""
    needs.SHOWCASE.check(p)
    inp = step_inputs(pos, seed, w, head_chunks)
    st = Step(p, inp)
    st.run()
    text, case = st.k.done("step")
    marks = st.blk.marks
    case["golden"]["stage_marks"] = [[l, name, c] for l, name, c in marks]
    case["golden"].update(layers=CFG.layers, head_rows=head_chunks * CHUNK, head_lo=inp.head_lo, token=inp.token,
                          pos=pos, argmax_token=inp.argmax, slots=CFG.slots)
    return text, case
