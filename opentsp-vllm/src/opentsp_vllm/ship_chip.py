"""Adapt the four-chip GPT-2 small pipeline to the vLLM chip interface.

The vLLM scheduler supplies token rows, masked page addresses and positions.
Simulated devices keep weights and KV resident. The host samples the logits.
"""

from collections import deque
import os
import time

import numpy as np

from opentsp_vllm import ship_plan
from opentsp_vllm.serving_observer import configured_trace, emit, request_label
from opentsp_vllm.contracts import DomainFault
from opentsp_vllm.contracts import ChipInfo, ChipResult, TspChipError

DEFAULT_CHECKPOINT = os.path.expanduser("~/.cache/opentsp/checkpoints/gpt2")

def _showcase(heavy=True):
    """Load the maintained GPT-2 model package lazily."""
    from opentsp_vllm.models.gpt2.config import GPT2_SMALL
    if not heavy:
        return GPT2_SMALL
    from opentsp_vllm.models.gpt2 import weights, pipeline
    return weights, pipeline, GPT2_SMALL


class ShipChip:
    """The four-chip GPT-2 adapter and its explicitly truncated test fixture."""

    def __init__(self, layers=12, programs=None, check=False, log=None, resident=True, history_limit=256, kv_chunks=ship_plan.FULL_KV_CHUNKS, chips=None):
        self._pool_slots, self._num_blocks = ship_plan.kv_limits(kv_chunks)
        self._kv_chunks = kv_chunks
        self._chips = chips
        self._layers, self._programs, self._check, self._resident = layers, programs, check, resident
        self._log = log or (lambda m: None)
        self._demo_trace = configured_trace()
        self._demo_cycle = 0
        self._pipe = None
        self._failure = None
        self._closed = False
        self._weights = None
        try:
            self._cfg = _showcase(heavy=False)
        except ImportError as e:
            raise ImportError(f"ShipChip: {e}") from e
        if not 1 <= layers <= self._cfg.layers:
            raise ValueError(f"layers {layers}: GPT-2 small has {self._cfg.layers} layers")
        self.steps = deque(maxlen=history_limit)  # None requests full capture; zero disables retention.

    @property
    def info(self):
        c = self._cfg
        return ChipInfo(self._layers, c.vocab, c.hidden, c.heads, self._num_blocks, (self._pool_slots,), c.heads, c.hidden // c.heads)

    @property
    def failure(self):
        return self._failure

    @property
    def chips(self):
        return self._pipe.chips if self._pipe else None

    def plan_step(self, requests):
        """The runner's hook: the scheduled requests as lanes and input vectors (`ship_plan.plan_step`)."""
        return ship_plan.plan_step(requests, vocab=self._cfg.vocab, kv_chunks=self._kv_chunks)

    def load_model(self, source):
        """Load shared deployment tensors, or a checkpoint for a reduced fixture."""
        self._live()
        gpt2s_weights, ship_pipeline, cfg = _showcase()
        kw = dict(cache=self._programs) if self._programs else {}
        if self._layers == self._cfg.layers:
            if self._chips != 4:
                raise ValueError("full GPT-2 requires the sealed four-chip route")
            from opentsp_vllm.models.gpt2.four_chip import FourChipPipeline

            self._weights = gpt2s_weights.load_deployment(cfg, source)
            self._pipe = FourChipPipeline(self._weights, self._programs, log=self._log)
            return
        if source.startswith("synthetic:"):
            raise ValueError("synthetic:SEED is the ReferenceChip's source")
        path = os.path.join(source, "model.safetensors")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path}: the reduced fixture requires a GPT-2 checkpoint")
        self._weights = gpt2s_weights.load_checkpoint(cfg, path)
        self._pipe = ship_pipeline.Pipeline(self._weights, self._layers, log=self._log, resident=self._resident,
                                            history_limit=self.steps.maxlen, kv_chunks=self._kv_chunks, **kw)
        if not self._programs or not os.path.isfile(os.path.join(self._programs, "manifest.json")):
            from opentsp_vllm.models.gpt2 import cache as ship_cache  # noqa: PLC0415

            ship_cache.evict(os.path.dirname(self._pipe.dir), retain=self._pipe.dir)

    def warm_up(self):
        self._live()
        self._need_pipe()

    def package_identity(self):
        return self._need_pipe().package_sha256

    def clear_canary(self):
        self._need_pipe().clear_canary()

    def inject_sram(self, chip, slice_id, addr, lane, bits):
        self._need_pipe().completion_flips.extend((chip, slice_id, addr, lane, bit) for bit in bits)

    def probe_sram(self, chip, slice_id, addr):
        # The next private canary allocates the new System and loads all stages.
        self._need_pipe().health_target = (chip, slice_id, addr)

    def zero_pool(self):
        self._live()
        self._need_pipe().reset()

    def zero_blocks(self, block_ids):
        self._live()
        for b in block_ids:
            if not 0 < b < self._num_blocks:
                raise ValueError(f"block {b} is the null block or outside the pool of {self._num_blocks} blocks")
        # nothing to do: see the module docstring (a slot outside the lane's mask encoding gets the probability +0)

    def copy_blocks(self, copies):
        self._live()
        if list(copies):
            raise NotImplementedError("ShipChip: copying a KV block (copy on write) is not implemented; the pool words of a page are not copied between pages yet")

    def step(self, plan):
        """One shared-clock run per scheduled batch, with logit rows returned in lane order."""
        self._live()
        if self._failure:
            raise TspChipError(self._failure)
        pipe = self._need_pipe()
        inputs = plan.patches
        rows, cycles = [], 0
        try:
            lanes = plan.real_lanes
            requests = [(lane.token, lane.position, int(inputs["slots"][i]),
                         inputs["counts"][i].reshape(-1, ship_plan.PAGES_PER_CHUNK))
                        for i, lane in enumerate(lanes)]
            t0 = time.perf_counter()
            options = {}
            if self._demo_trace and self._chips == 4 and requests:
                labels = [request_label(lane.req_id) for lane in lanes]
                positions = [lane.position for lane in lanes]

                def observe(phase, **record):
                    emit(self._demo_trace, dict(phase=phase, step=plan.demo_step, row_ids=labels,
                                                      row_positions=positions, row_logit=[lane.logits for lane in lanes],
                                                      wave_start=self._demo_cycle, **record))

                options["observe"] = observe
            if len(requests) == 1 and not options:
                token, position, slot, counts = requests[0]
                out, st = pipe.step(token, position, check=self._check, slot=slot, counts=counts)
                outputs = [out]
            elif requests:
                outputs, st = pipe.batch(requests, check=self._check, **options)
            if requests:
                cycles = st["cycles"]
                st["wall_s"] = time.perf_counter() - t0
                self.steps.append(st)
                if self._demo_trace:
                    launches = st.get("launch_cycles")
                    if launches is None:
                        launches = [pipe.stages[0].dma_cycles[0] + i * st["initiation_interval"] for i in range(len(lanes))]
                    emit(self._demo_trace, dict(phase="pipeline", row_ids=[request_label(lane.req_id) for lane in lanes],
                                                step=plan.demo_step, ingress_cycles=launches, wave_start=self._demo_cycle,
                                                logit_rows=[lane.logits for lane in lanes], **st))
                    self._demo_cycle += cycles
                rows = [pipe.logits(out) for lane, out in zip(lanes, outputs, strict=True) if lane.logits]
        except DomainFault as e:
            self._failure = str(e)
            raise
        except (RuntimeError, ValueError) as e:
            self._failure = f"{type(e).__name__}: {e}"
            raise TspChipError(self._failure) from e
        logits = np.stack(rows) if rows else np.zeros((0, self._cfg.vocab), np.float32)
        return ChipResult(logits, cycles, ())

    def close(self):
        self._closed = True
        self._pipe = None

    def _live(self):
        if self._closed:
            raise RuntimeError("ShipChip is closed")

    def _need_pipe(self):
        if self._pipe is None:
            raise RuntimeError("ShipChip: no model is loaded (load_model first)")
        return self._pipe

