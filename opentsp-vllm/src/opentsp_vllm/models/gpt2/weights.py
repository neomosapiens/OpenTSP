"""GPT-2 small weights: the Hugging Face checkpoint rounded the way the chip holds it. Matrices and the two embedding tables are
f16 (the MXM's floating-point weight type), biases and LayerNorm parameters stay f32. Matrices are (out, in): `y = W x`."""
import hashlib
import os
from dataclasses import dataclass, field

import numpy as np

from opentsp_vllm import logical_weights
from opentsp_vllm.models.gpt2.config import Gpt2Config

MATRICES = ("wq", "wk", "wv", "wo", "up_w", "dn_w")
VECTORS = ("bq", "bk", "bv", "bo", "up_b", "dn_b", "ln1g", "ln1b", "ln2g", "ln2b")


@dataclass
class Weights:
    cfg: Gpt2Config
    wte: np.ndarray                       # (vocab, H) f16
    wpe: np.ndarray                       # (max_pos, H) f16
    layers: list = field(default_factory=list)
    lnf_g: np.ndarray = None              # (H) f32
    lnf_b: np.ndarray = None

    def arrays(self):
        yield self.wte
        yield self.wpe
        for lw in self.layers:
            for k in MATRICES + VECTORS:
                yield lw[k]
        yield self.lnf_g
        yield self.lnf_b

    def digest(self):
        h = hashlib.sha256()
        for a in self.arrays():
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def validate(self):
        c = self.cfg
        H, F = c.hidden, c.ffn
        assert self.wte.shape == (c.vocab, H) and self.wte.dtype == np.float16
        assert self.wpe.shape == (c.max_pos, H) and self.wpe.dtype == np.float16
        assert len(self.layers) == c.layers
        shapes = dict(wq=(H, H), wk=(H, H), wv=(H, H), wo=(H, H), up_w=(F, H), dn_w=(H, F), bq=(H,), bk=(H,), bv=(H,), bo=(H,),
                      up_b=(F,), dn_b=(H,), ln1g=(H,), ln1b=(H,), ln2g=(H,), ln2b=(H,))
        for lw in self.layers:
            for k, s in shapes.items():
                assert lw[k].shape == s, (k, lw[k].shape, s)
                assert lw[k].dtype == (np.float16 if k in MATRICES else np.float32), k
        assert self.lnf_g.shape == (H,) and self.lnf_g.dtype == np.float32 and self.lnf_b.shape == (H,) and self.lnf_b.dtype == np.float32
        return self


def from_state_dict(cfg: Gpt2Config, sd) -> Weights:
    """Weights from a GPT-2 state dict of numpy arrays (keys of `GPT2LMHeadModel.state_dict`, with or without the
    `transformer.` prefix). Conv1D stores (in, out): every matrix is transposed to (out, in), the fused QKV split in three."""
    def get(k):
        for key in (k, "transformer." + k):
            if key in sd:
                return np.asarray(sd[key])
        raise KeyError(k)

    H = cfg.hidden
    shape = get("wte.weight").shape
    if shape[1] != H:
        raise ValueError(f"the checkpoint has width {shape[1]}; this export is for GPT-2 small, width {H} (12 heads of {cfg.dk})")
    layers = sum(1 for k in sd if k.replace("transformer.", "").endswith(".attn.c_attn.weight"))
    if layers != cfg.layers:
        raise ValueError(f"the checkpoint has {layers} layers; this export is for GPT-2 small, {cfg.layers} layers")
    if shape[0] < cfg.vocab:
        raise ValueError(f"the checkpoint has {shape[0]} tokens, GPT-2 small has {cfg.vocab}")
    f32 = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    f16m = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.float32).astype(np.float16))
    w = Weights(cfg, f16m(get("wte.weight")[:cfg.vocab]), f16m(get("wpe.weight")[:cfg.max_pos]))
    for l in range(cfg.layers):
        p = f"h.{l}."
        qkv, bqkv = get(p + "attn.c_attn.weight").T, get(p + "attn.c_attn.bias")
        w.layers.append(dict(
            wq=f16m(qkv[:H]), wk=f16m(qkv[H:2 * H]), wv=f16m(qkv[2 * H:]), wo=f16m(get(p + "attn.c_proj.weight").T),
            up_w=f16m(get(p + "mlp.c_fc.weight").T), dn_w=f16m(get(p + "mlp.c_proj.weight").T),
            bq=f32(bqkv[:H]), bk=f32(bqkv[H:2 * H]), bv=f32(bqkv[2 * H:]), bo=f32(get(p + "attn.c_proj.bias")),
            up_b=f32(get(p + "mlp.c_fc.bias")), dn_b=f32(get(p + "mlp.c_proj.bias")),
            ln1g=f32(get(p + "ln_1.weight")), ln1b=f32(get(p + "ln_1.bias")),
            ln2g=f32(get(p + "ln_2.weight")), ln2b=f32(get(p + "ln_2.bias"))))
    w.lnf_g, w.lnf_b = f32(get("ln_f.weight")), f32(get("ln_f.bias"))
    return w.validate()



def save_deployment(path, model):
    """Save one shared logical tensor store for the host and all four chips."""
    logical_weights.write(path, model, MATRICES + VECTORS, "opentsp.gpt2.logical.v1")


def load_deployment(cfg, path):
    """Read the logical tensor shapes and dtypes declared by the release."""
    from pathlib import Path
    return logical_weights.decode(Path(path).read_bytes(), cfg, Weights, MATRICES + VECTORS)


def find_checkpoint(name="gpt2", cache_dir=None):
    """Path of `model.safetensors` of `name` in the local Hugging Face cache (never downloaded here)."""
    if name.endswith(".safetensors") or os.path.isabs(name) or name.startswith("."):
        raise FileNotFoundError(f"checkpoint file {name!r} does not exist (a model name such as gpt2, or a path to an existing .safetensors file)")
    root = os.path.join(cache_dir or os.path.expanduser("~/.cache/huggingface/hub"), "models--" + name.replace("/", "--"), "snapshots")
    for snap in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, snap, "model.safetensors")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no model.safetensors for {name!r} under {root}; "
                            f"fetch it with `hf download {name} model.safetensors`")


def load_checkpoint(cfg: Gpt2Config, path_or_name="gpt2", cache_dir=None) -> Weights:
    """The weights of a checkpoint file (a path to a `.safetensors`) or of a cached model name."""
    from safetensors.numpy import load_file             # noqa: PLC0415
    path = path_or_name if os.path.isfile(str(path_or_name)) else find_checkpoint(path_or_name, cache_dir)
    return from_state_dict(cfg, load_file(path))


def synthetic(cfg: Gpt2Config, seed=1) -> Weights:
    """Seeded random weights of the checkpoint's shapes (small scales so that attention and the FFN stay finite), for tests."""
    rng = np.random.default_rng(seed)
    H, F = cfg.hidden, cfg.ffn
    f16 = lambda shape, s: (rng.standard_normal(shape) * s).astype(np.float16)
    f32 = lambda shape, s, m=0.0: (rng.standard_normal(shape) * s + m).astype(np.float32)
    w = Weights(cfg, f16((cfg.vocab, H), 0.2), f16((cfg.max_pos, H), 0.05))
    for _ in range(cfg.layers):
        w.layers.append(dict(
            wq=f16((H, H), 0.08), wk=f16((H, H), 0.08), wv=f16((H, H), 0.05), wo=f16((H, H), 0.05),
            up_w=f16((F, H), 0.05), dn_w=f16((H, F), 0.03),
            bq=f32((H,), 0.02), bk=f32((H,), 0.02), bv=f32((H,), 0.02), bo=f32((H,), 0.02), up_b=f32((F,), 0.02),
            dn_b=f32((H,), 0.02), ln1g=f32((H,), 0.05, 1.0), ln1b=f32((H,), 0.02), ln2g=f32((H,), 0.05, 1.0),
            ln2b=f32((H,), 0.02)))
    w.lnf_g, w.lnf_b = f32((H,), 0.05, 1.0), f32((H,), 0.02)
    return w.validate()
