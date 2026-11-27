"""Runs vLLM's stock CPU backend on a reference checkpoint (`tools/vllm_ref_checkpoint.py`) in its own process, for
`test_cpu_backend.py`: the TSP platform is a different platform of the same vLLM build and a process has one.

    cpu_backend_run.py CHECKPOINT REQUESTS.json OUT.npz

REQUESTS.json: {"greedy": [[prompt ids], ...], "n": N, "forced": [[prompt + forced ids], ...]}. Output: `tokens` (greedy ids of each
greedy prompt, N each) and `logprobs` (one row per forced prompt: the log-softmax of the logits after it, the whole vocabulary).
"""

import json
import os
import sys

os.environ.pop("OPENTSP_VLLM", None)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")  # GiB of the CPU backend's KV cache

import numpy as np


def use_torch_attention():
    """The CPU backend's attention kernel refuses head_dim 4 ("Unsupported CPU attention configuration"; its sizes start at 32), and
    the TinyStories GPT-2 has 16 heads of 4. Everything else of the backend stays (scheduler, KV cache manager, model code, weights
    loader, sampler); the paged attention op and the KV write are replaced by plain torch over the same cache tensors, in fp32."""
    import torch
    import vllm._custom_ops as ops
    from vllm.v1.attention.backends import cpu_attn

    ops.cpu_attn_get_scheduler_metadata = lambda *a, **k: torch.zeros(1, dtype=torch.int32)

    def views(kv_cache):
        nb, kvh, bs, _ = kv_cache.size()
        return kv_cache.view((nb, kvh, bs * 2, -1)).chunk(2, dim=2), bs

    def update(self, layer, key, value, kv_cache, slot_mapping):
        (kc, vc), bs = views(kv_cache)
        slots = slot_mapping[: key.shape[0]].long()
        keep = slots >= 0
        blocks, offs = slots[keep] // bs, slots[keep] % bs
        kc[blocks, :, offs, :] = key[keep]
        vc[blocks, :, offs, :] = value[keep]

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output
        (kc, vc), bs = views(kv_cache)
        starts, lens = attn_metadata.query_start_loc.tolist(), attn_metadata.seq_lens.tolist()
        for r, total in enumerate(lens):
            q = query[starts[r] : starts[r + 1]]  # (nq, heads, hd); the queries are the last nq tokens of the sequence
            blocks = attn_metadata.block_table[r, : -(-total // bs)].long()
            k = kc[blocks].permute(1, 0, 2, 3).reshape(kc.shape[1], -1, kc.shape[3])[:, :total]
            v = vc[blocks].permute(1, 0, 2, 3).reshape(vc.shape[1], -1, vc.shape[3])[:, :total]
            scores = torch.einsum("qhd,hld->hql", q.float(), k.float()) * self.scale
            pos = torch.arange(total - q.shape[0], total).view(1, -1, 1)
            scores = scores.masked_fill(torch.arange(total).view(1, 1, -1) > pos, float("-inf"))
            out = torch.einsum("hql,hld->qhd", torch.softmax(scores, dim=-1), v.float())
            output[starts[r] : starts[r + 1]] = out.reshape(output[starts[r] : starts[r + 1]].shape).to(output.dtype)
        return output

    cpu_attn.CPUAttentionBackendImpl.do_kv_cache_update = update
    cpu_attn.CPUAttentionBackendImpl.forward = forward


def main(checkpoint, requests, out):
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    use_torch_attention()
    assert current_platform.is_cpu(), f"the platform is {current_platform}, not the CPU backend"
    req = json.load(open(requests))
    llm = LLM(
        model=checkpoint,
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=64,
        max_num_seqs=8,
        enable_prefix_caching=True,
        max_logprobs=-1,
        enforce_eager=True,
    )
    tokens = [
        list(o.outputs[0].token_ids)
        for o in llm.generate(
            [{"prompt_token_ids": p} for p in req["greedy"]],
            SamplingParams(temperature=0.0, max_tokens=req["n"], ignore_eos=True),
            use_tqdm=False,
        )
    ]
    rows = []
    for o in llm.generate(
        [{"prompt_token_ids": p} for p in req["forced"]],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, logprobs=-1),
        use_tqdm=False,
    ):
        lp = o.outputs[0].logprobs[0]
        row = np.full(max(lp) + 1, -np.inf, np.float64)
        for t, entry in lp.items():
            row[t] = entry.logprob
        rows.append(row)
    np.savez(out, tokens=np.array(tokens, np.int64), logprobs=np.stack(rows))


if __name__ == "__main__":
    main(*sys.argv[1:4])
