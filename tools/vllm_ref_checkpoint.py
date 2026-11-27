#!/usr/bin/env python3
"""`just vllm-ref-checkpoint`: a Hugging Face GPT-2 checkpoint (`config.json` and `model.safetensors`, fp32) whose weights are the
f16-rounded TSP weights, for vLLM's stock CPU backend.

    vllm_ref_checkpoint.py --layers 2 --weights synthetic:1 --out DIR
    vllm_ref_checkpoint.py --layers 8 --weights real --out DIR       (the calum/tinystories-gpt2-3M checkpoint in the local HF cache)

The matrices and embeddings are the f16 values of `opentsp-vllm/src/opentsp_vllm/models/tinystories/weights.py` widened to f32, the biases and LayerNorm
parameters its f32 values, so the CPU backend computes with exactly the weights the chip holds. `weights.from_gpt2_state_dict` is the
inverse of this mapping and the test of this tool checks the round trip.
"""

import argparse
import json
import pathlib

import numpy as np



def load_weights(source, layers):
    from opentsp_vllm.models.tinystories import layout, weights

    cfg = layout.serve_config(layers)
    if source == "real":
        return weights.load_safetensors(cfg)
    kind, _, seed = source.partition(":")
    if kind != "synthetic" or not seed.isdigit():
        raise SystemExit(f"--weights must be synthetic:SEED or real, not {source!r}")
    return weights.synthetic(cfg, seed=int(seed))


def state_dict(w):
    """The `GPT2LMHeadModel` state dict (numpy float32) of the weights; GPT-2's Conv1D stores (in, out)."""
    f32 = lambda a: np.ascontiguousarray(np.asarray(a, dtype=np.float32))  # noqa: E731
    sd = {"transformer.wte.weight": f32(w.wte), "transformer.wpe.weight": f32(w.wpe)}
    for i, lw in enumerate(w.layers):
        p = f"transformer.h.{i}."
        sd[p + "ln_1.weight"], sd[p + "ln_1.bias"] = f32(lw["ln1g"]), f32(lw["ln1b"])
        sd[p + "attn.c_attn.weight"] = f32(np.concatenate([lw["wq"], lw["wk"], lw["wv"]], axis=0).T)
        sd[p + "attn.c_attn.bias"] = f32(np.concatenate([lw["bq"], lw["bk"], lw["bv"]]))
        sd[p + "attn.c_proj.weight"], sd[p + "attn.c_proj.bias"] = f32(lw["wo"].T), f32(lw["bo"])
        sd[p + "ln_2.weight"], sd[p + "ln_2.bias"] = f32(lw["ln2g"]), f32(lw["ln2b"])
        sd[p + "mlp.c_fc.weight"], sd[p + "mlp.c_fc.bias"] = f32(lw["up_w"].T), f32(lw["up_b"])
        sd[p + "mlp.c_proj.weight"], sd[p + "mlp.c_proj.bias"] = f32(lw["dn_w"].T), f32(lw["dn_b"])
    sd["transformer.ln_f.weight"], sd["transformer.ln_f.bias"] = f32(w.lnf_g), f32(w.lnf_b)
    return sd


def config_json(cfg, positions=128):
    return {
        "architectures": ["GPT2LMHeadModel"],
        "model_type": "gpt2",
        "n_embd": cfg.hidden,
        "n_head": cfg.heads,
        "n_layer": cfg.layers,
        "n_inner": cfg.ffn,
        "n_positions": positions,
        "n_ctx": positions,
        "vocab_size": cfg.vocab,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": cfg.ln_eps,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }


def write(out, w, positions=128):
    from safetensors.numpy import save_file

    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sd = state_dict(w)
    sd["transformer.wpe.weight"] = np.ascontiguousarray(sd["transformer.wpe.weight"][:positions])
    save_file(sd, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(config_json(w.cfg, positions), indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layers", type=int, default=2, choices=(2, 8))
    ap.add_argument("--weights", default="synthetic:1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    w = load_weights(a.weights, a.layers)
    out = write(a.out, w)
    print(f"vllm-ref-checkpoint: wrote {out} ({a.layers} layers, weights {a.weights}, digest {w.digest()[:16]})")


if __name__ == "__main__":
    main()
