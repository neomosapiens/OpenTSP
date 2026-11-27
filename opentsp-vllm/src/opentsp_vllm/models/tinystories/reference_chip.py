"""NumPy reference device for a TinyStories serving plan."""

import numpy as np

from opentsp_vllm.contracts import ChipInfo, ChipResult, TspChipError
from opentsp_vllm.step_plan import GRID_C, POOL_BLOCKS
from . import layout, paged_reference, weights


class ReferenceChip:
    def __init__(self, cfg, grid_c=GRID_C, num_blocks=POOL_BLOCKS):
        self.cfg = cfg
        self._grid_c = tuple(grid_c)
        self._num_blocks = num_blocks
        self._w = None
        self._pool = None
        self._closed = False
        self._failure = None
        self.steps = 0

    @property
    def info(self):
        c = self.cfg
        return ChipInfo(c.layers, c.vocab, c.hidden, c.heads, self._num_blocks, self._grid_c)

    @property
    def failure(self):
        return self._failure

    @property
    def pool(self):
        return self._pool

    def load_model(self, source):
        kind, _, seed = source.partition(":")
        if kind != "synthetic" or not seed.isdigit():
            raise ValueError(f"ReferenceChip loads synthetic:SEED weights only, not {source!r}")
        self._w = weights.synthetic(self.cfg, seed=int(seed))
        self.zero_pool()

    def warm_up(self):
        self._live()

    def zero_pool(self):
        self._pool = paged_reference.BlockPool(self.cfg, self._num_blocks)

    def zero_blocks(self, block_ids):
        self._live()
        for b in block_ids:
            self._vllm_block(b)
            self._pool.zero(b)

    def copy_blocks(self, copies):
        self._live()
        for src, dst in copies:
            self._vllm_block(src), self._vllm_block(dst)
            self._pool.copy(src, dst)

    def step(self, plan):
        self._live()
        lanes = plan.real_lanes
        try:
            r = paged_reference.step_batch(
                self._w,
                self._pool,
                [ln.token for ln in lanes],
                [ln.position for ln in lanes],
                [list(ln.block_table) for ln in lanes],
                plan.bucket,
                plan.n_logits,
            )
        except Exception as e:
            self._failure = f"{type(e).__name__}: {e}"
            raise TspChipError(self._failure) from e
        self.steps += 1
        return ChipResult(np.asarray(r.logits, dtype=np.float32))

    def close(self):
        self._closed = True

    def _live(self):
        if self._closed:
            raise RuntimeError("ReferenceChip is closed")
        if self._w is None:
            raise RuntimeError("ReferenceChip: load_model has not been called")

    def _vllm_block(self, b):
        if not 0 < b < self._num_blocks:
            raise ValueError(f"block {b} is the null block or outside the pool of {self._num_blocks} blocks")


def make_serving_chip(vllm_config, chip_class=ReferenceChip):
    """The exact reference for the serving model (50257 words; 2 or 8 layers as the engine's model
    directory says), with the programs' context grid of `opentsp.max_context`."""
    layers = vllm_config.model_config.hf_config.n_layer
    context = vllm_config.additional_config.get("opentsp", {}).get("max_context", 128)
    return chip_class(layout.serve_config(layers), grid_c=(64,) if context == 64 else (64, 128))
