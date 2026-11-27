"""Model weights: a deterministic synthetic generator, and the mapping from a GPT-2 state dict.

All matrices and the embedding tables hold f16 values (the MXM's only floating-point weight type); biases and
normalisation parameters are f32. Matrices are (out_features, in_features), the layout `y = W x`.
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np

from opentsp_vllm import logical_weights
from opentsp_vllm.models.tinystories.config import ModelConfig

MATRICES = ("wq", "wk", "wv", "wo", "up_w", "dn_w")
VECTORS = ("bq", "bk", "bv", "bo", "up_b", "dn_b", "ln1g", "ln1b", "ln2g", "ln2b")


@dataclass
class Weights:
    cfg: ModelConfig
    wte: np.ndarray                      # (vocab, H) float16
    wpe: np.ndarray                      # (max_pos, H) float16
    layers: list = field(default_factory=list)   # dicts of MATRICES (float16) and VECTORS (float32)
    lnf_g: np.ndarray = None             # (H) float32
    lnf_b: np.ndarray = None

    def digest(self):
        h = hashlib.sha256()
        for a in self.arrays():
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def arrays(self):
        yield self.wte
        yield self.wpe
        for lw in self.layers:
            for k in MATRICES + VECTORS:
                yield lw[k]
        yield self.lnf_g
        yield self.lnf_b

    def validate(self):
        c = self.cfg
        H, F = c.hidden, c.ffn
        assert self.wte.shape == (c.vocab, H) and self.wte.dtype == np.float16
        assert self.wpe.shape == (c.max_pos, H) and self.wpe.dtype == np.float16
        assert len(self.layers) == c.layers
        shapes = dict(wq=(H, H), wk=(H, H), wv=(H, H), wo=(H, H), up_w=(F, H), dn_w=(H, F),
                      bq=(H,), bk=(H,), bv=(H,), bo=(H,), up_b=(F,), dn_b=(H,),
                      ln1g=(H,), ln1b=(H,), ln2g=(H,), ln2b=(H,))
        for lw in self.layers:
            for k, s in shapes.items():
                assert lw[k].shape == s, (k, lw[k].shape, s)
                assert lw[k].dtype == (np.float16 if k in MATRICES else np.float32), k
        assert self.lnf_g.shape == (H,) and self.lnf_g.dtype == np.float32 and self.lnf_b.shape == (H,) and self.lnf_b.dtype == np.float32
        return self


def encode_deployment(model):
    return logical_weights.encode(model, MATRICES + VECTORS, "opentsp.tinystories.logical.v1")


def decode_deployment(cfg, data):
    return logical_weights.decode(data, cfg, Weights, MATRICES + VECTORS)


# -- synthetic weights -------------------------------------------------------------------------------------------
# A counter-based generator built from integer arithmetic and exactly rounded f64 add/mul only, so that a seed
# gives the same weights on every platform and numpy version (no libm, no Generator stream): splitmix64 outputs
# are uniform in [0, 1); four of them summed and scaled give a bell-shaped variate of unit variance.
_M64 = (1 << 64) - 1
_SQRT3 = 1.7320508075688772


def _key(seed, name):
    h = hashlib.sha256(f"opentsp-ts0:{seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def _splitmix(counter):
    z = counter + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform01(seed, name, n):
    base = np.uint64(_key(seed, name))
    ctr = base + np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = _splitmix(ctr)
    return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def gauss(seed, name, shape):
    n = int(np.prod(shape))
    u = uniform01(seed, name, 4 * n).reshape(4, n)
    g = ((u[0] + u[1]) + (u[2] + u[3]) - 2.0) * _SQRT3
    return g.reshape(shape)


def synthetic(cfg: ModelConfig, seed: int = 1) -> Weights:
    """Seeded synthetic weights of the checkpoint's shapes. Scales are chosen so that attention is not uniform
    and every path (biases, LayerNorm affine, tied head) matters: matrices N(0, 0.15^2) for Q and K, N(0, 0.05^2)
    for the rest, biases N(0, 0.02^2), gamma = 1 + N(0, 0.05^2), beta N(0, 0.02^2), wte N(0, 0.2^2),
    wpe N(0, 0.05^2)."""
    H, F = cfg.hidden, cfg.ffn

    def f16(name, shape, std):
        return (gauss(seed, name, shape) * std).astype(np.float16)

    def f32(name, shape, std, mean=0.0):
        return (gauss(seed, name, shape) * std + mean).astype(np.float32)

    w = Weights(cfg, f16("wte", (cfg.vocab, H), 0.2), f16("wpe", (cfg.max_pos, H), 0.05))
    for l in range(cfg.layers):
        p = f"l{l}."
        w.layers.append(dict(
            wq=f16(p + "wq", (H, H), 0.15), wk=f16(p + "wk", (H, H), 0.15),
            wv=f16(p + "wv", (H, H), 0.05), wo=f16(p + "wo", (H, H), 0.05),
            up_w=f16(p + "up_w", (F, H), 0.05), dn_w=f16(p + "dn_w", (H, F), 0.05),
            bq=f32(p + "bq", (H,), 0.02), bk=f32(p + "bk", (H,), 0.02), bv=f32(p + "bv", (H,), 0.02),
            bo=f32(p + "bo", (H,), 0.02), up_b=f32(p + "up_b", (F,), 0.02), dn_b=f32(p + "dn_b", (H,), 0.02),
            ln1g=f32(p + "ln1g", (H,), 0.05, 1.0), ln1b=f32(p + "ln1b", (H,), 0.02),
            ln2g=f32(p + "ln2g", (H,), 0.05, 1.0), ln2b=f32(p + "ln2b", (H,), 0.02)))
    w.lnf_g = f32("lnfg", (H,), 0.05, 1.0)
    w.lnf_b = f32("lnfb", (H,), 0.02)
    return w.validate()


# -- GPT-2 checkpoint mapping ------------------------------------------------------------------------------------
def from_gpt2_state_dict(cfg: ModelConfig, sd) -> Weights:
    """Weights from a GPT-2 (Hugging Face `GPT2LMHeadModel`) state dict of numpy arrays, keys as in
    `model.state_dict` (with or without the `transformer.` prefix). GPT-2's Conv1D stores (in, out): every
    matrix is transposed to (out, in); the fused QKV is split in three; matrices and embeddings are rounded
    to f16, biases and LayerNorm parameters to f32."""
    def get(k):
        for key in (k, "transformer." + k):
            if key in sd:
                return np.asarray(sd[key])
        raise KeyError(k)

    H = cfg.hidden
    w = Weights(cfg, get("wte.weight").astype(np.float16)[:cfg.vocab], get("wpe.weight").astype(np.float16)[:cfg.max_pos])
    for l in range(cfg.layers):
        p = f"h.{l}."
        qkv = get(p + "attn.c_attn.weight").T            # (3H, H)
        bqkv = get(p + "attn.c_attn.bias")
        f32 = lambda a: np.asarray(a, dtype=np.float32)
        w.layers.append(dict(
            wq=qkv[:H].astype(np.float16), wk=qkv[H:2 * H].astype(np.float16), wv=qkv[2 * H:].astype(np.float16),
            wo=get(p + "attn.c_proj.weight").T.astype(np.float16),
            up_w=get(p + "mlp.c_fc.weight").T.astype(np.float16), dn_w=get(p + "mlp.c_proj.weight").T.astype(np.float16),
            bq=f32(bqkv[:H]), bk=f32(bqkv[H:2 * H]), bv=f32(bqkv[2 * H:]), bo=f32(get(p + "attn.c_proj.bias")),
            up_b=f32(get(p + "mlp.c_fc.bias")), dn_b=f32(get(p + "mlp.c_proj.bias")),
            ln1g=f32(get(p + "ln_1.weight")), ln1b=f32(get(p + "ln_1.bias")),
            ln2g=f32(get(p + "ln_2.weight")), ln2b=f32(get(p + "ln_2.bias"))))
    w.lnf_g, w.lnf_b = np.asarray(get("ln_f.weight"), dtype=np.float32), np.asarray(get("ln_f.bias"), dtype=np.float32)
    for lw in w.layers:       # contiguous copies (transposes are views)
        for k in lw:
            lw[k] = np.ascontiguousarray(lw[k])
    return w.validate()


def load_safetensors(cfg: ModelConfig, name="calum/tinystories-gpt2-3M", cache_dir=None) -> Weights:
    """Real weights without PyTorch: the checkpoint's `model.safetensors` in the local Hugging Face cache (never
    downloaded here), read with the `safetensors` package. Raises FileNotFoundError when no cached snapshot has it."""
    import os                                           # noqa: PLC0415
    from safetensors.numpy import load_file             # noqa: PLC0415 (optional dependency)
    root = os.path.join(cache_dir or os.path.expanduser("~/.cache/huggingface/hub"),
                        "models--" + name.replace("/", "--"), "snapshots")
    for snap in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, snap, "model.safetensors")
        if os.path.exists(path):
            return from_gpt2_state_dict(cfg, load_file(path))
    raise FileNotFoundError(f"no model.safetensors of {name} under {root}")


def load_checkpoint(cfg: ModelConfig, name="calum/tinystories-gpt2-3M", cache_dir=None) -> Weights:
    """Optional real-weight path: needs PyTorch and transformers and a network or cached download (never used by
    the tests; the checkpoint and what is derived from it are not committed). Returns f16/f32 weights."""
    from transformers import GPT2LMHeadModel      # noqa: PLC0415 (optional dependency)
    model = GPT2LMHeadModel.from_pretrained(name, cache_dir=cache_dir).eval()
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return from_gpt2_state_dict(cfg, sd)
