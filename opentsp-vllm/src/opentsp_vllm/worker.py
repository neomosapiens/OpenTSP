import importlib
import json
import logging
import os
from functools import partial

import torch
import torch.nn as nn
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from opentsp_vllm import options
from opentsp_vllm.model_runner import TspModelRunner
from opentsp_vllm.ras import ManagedChip
from opentsp_vllm.shadow import ShadowChip
from opentsp_vllm.stats import Stats
from opentsp_vllm.step_plan import BLOCK

SYNTHETIC_DEFAULT = "synthetic:1"
logger = logging.getLogger(__name__)
SHADOW_REFERENCE = "opentsp_vllm.models.tinystories.reference_chip:make_serving_chip"


def _configure_host_only_cleanup():
    """vLLM's `cleanup_dist_env_and_memory` empties the accelerator's host-memory cache whenever the platform is not CPU, and the
    torch of macOS crashes the process in that call. This platform has no accelerator memory, so the call has no effect to lose."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return
    if hasattr(accelerator, "empty_host_cache"):
        accelerator.empty_host_cache = lambda: None
    # Linux CPU-only torch raises when vLLM clears a nonexistent device cache.
    if accelerator.current_accelerator() is None:
        accelerator.empty_cache = lambda: None


class TspWorker(WorkerBase):
    """Adapt vLLM worker calls to the selected OpenTSP device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chip = None
        self.options = None
        self._model = None

    @staticmethod
    def _factory(spec, what):
        module, _, name = spec.partition(":")
        try:
            factory = getattr(importlib.import_module(module), name)
            if not callable(factory):
                raise ValueError("selected attribute is not callable")
            return factory
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"{what} {spec!r} does not resolve to a callable: {e}") from e

    def _need_chip(self, what):
        if self.chip is None:
            raise RuntimeError(f"TspWorker.{what}: init_device has not been called")
        return self.chip

    def _need_runner(self, what):
        self._need_chip(what)
        if self.model_runner is None:
            raise RuntimeError(f"TspWorker.{what}: load_model has not been called")
        return self.model_runner

    # table (a)

    def init_device(self):
        self.options = options.parse(self.vllm_config.additional_config)
        spec = self.options["chip"]
        path = self.options["programs"]
        if spec == "iss" and path and os.path.isfile(os.path.join(path, "manifest.json")):
            with open(os.path.join(path, "manifest.json")) as f:
                if json.load(f).get("execution") == "physical-kv":
                    spec = "tiny"
        if spec == "ship":
            from opentsp_vllm.ship_chip import DEFAULT_CHECKPOINT, ShipChip

            o = self.options
            self.chip = ManagedChip(partial(ShipChip, layers=o["layers"], programs=o["programs"],
                                            check=o["check"], resident=o["resident"], kv_chunks=o["kv_chunks"], chips=o["chips"], log=logger.info), "ship", o["ras_domain"])
            if not o["weights"]:
                o["weights"] = (os.path.join(o["programs"], "weights.safetensors")
                                if o["layers"] == 12 and o["programs"] else DEFAULT_CHECKPOINT)
        elif spec == "tiny":
            from opentsp_vllm.step_plan import GRID_C
            from opentsp_vllm.tiny_chip import TinyChip

            if not path:
                raise ValueError("TinyStories requires a serving package from `just showcase-prepare`")
            grid_c = tuple(c for c in GRID_C if c <= self.options["max_context"])
            self.chip = ManagedChip(partial(TinyChip, path, self.model_config.hf_config.n_layer, grid_c, self.options["kv_persist"]), "tiny", self.options["ras_domain"])
            self.options["weights"] = self.options["weights"] or os.path.join(path, "weights.safetensors")
        elif spec == "iss":
            from opentsp_vllm.chip import IssChip
            from opentsp_vllm.programs import GRID_C, ProgramSet

            layers = self.model_config.hf_config.n_layer
            grid_c = tuple(c for c in GRID_C if c <= self.options["max_context"])
            programs = ProgramSet(cache=self.options["programs"], layers=layers, grid_c=grid_c, profile=self.options["profile"])
            if not self.options["weights"]:
                self.options["weights"] = os.path.join(programs.entry, "weights.safetensors")
            self.chip = IssChip(profile=self.options["profile"], programs=programs, mode=self.options["kv_persist"], replay=self.options["replay"])
        else:
            self.chip = self._factory(spec, "additional_config opentsp.chip")(self.vllm_config)
        shadow = os.environ.get("OPENTSP_VLLM_SHADOW")
        if shadow:
            try:
                factory = self._factory(SHADOW_REFERENCE if shadow == "exact" else shadow, "OPENTSP_VLLM_SHADOW")
                twin = factory(self.vllm_config)
            except BaseException:
                self.chip.close()
                self.chip = None
                raise
            self.chip = ShadowChip(self.chip, twin, os.environ.get("OPENTSP_VLLM_SHADOW_WEIGHTS", SYNTHETIC_DEFAULT))
        self.device = torch.device("cpu")
        _configure_host_only_cleanup()

    def load_model(self, *, load_dummy_weights=False):
        chip = self._need_chip("load_model")
        weights = self.options["weights"]
        if load_dummy_weights and not (weights or "").startswith("synthetic:"):
            weights = SYNTHETIC_DEFAULT
        if not weights:
            raise ValueError("additional_config opentsp.weights is not set: give synthetic:SEED or the weights package directory")
        chip.load_model(weights)
        longest = max(chip.info.grid_c)
        if self.model_config.max_model_len > longest:
            raise ValueError(
                f"max_model_len {self.model_config.max_model_len} exceeds {longest}, the longest context of the chip's programs"
            )
        self.model_runner = TspModelRunner(self.vllm_config, chip)
        self._model = nn.Module()

    def get_kv_cache_spec(self):
        info = self._need_chip("get_kv_cache_spec").info
        spec = FullAttentionSpec(block_size=BLOCK, num_kv_heads=info.kv_heads, head_size=info.head_size, dtype=torch.float32)
        return {f"transformer.h.{layer}.attn": spec for layer in range(info.layers)}

    def get_supported_kv_cache_layouts(self):
        # the first entry of vLLM's default preference; this worker has no attention backend to ask
        return ["LBNHC"]

    def compile_or_warm_up_model(self) -> CompilationTimes:
        self._need_runner("compile_or_warm_up_model")
        self.chip.warm_up()
        return CompilationTimes(0.0, 0.0)

    def execute_model(self, scheduler_output):
        return self._need_runner("execute_model").execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        raise RuntimeError("TspWorker.sample_tokens is not reached: execute_model never returns None (structured outputs are rejected)")

    def get_model(self):
        self._need_runner("get_model")
        return self._model

    def check_health(self):
        failure = (self.model_runner and self.model_runner.failure) or (self.chip and self.chip.failure)
        if failure:
            raise RuntimeError(f"the TSP chip is in a failed state: {failure}")

    def synchronize_device(self):
        # a run is synchronous
        return None

    def shutdown(self):
        if self.chip is not None:
            self.chip.close()
        self.chip = None
        self.model_runner = None

    # table (b): RPC names that are not WorkerBase methods

    def determine_available_memory(self):
        chip = self._need_chip("determine_available_memory")
        return chip.info.num_blocks * sum(s.page_size_bytes for s in self.get_kv_cache_spec().values())

    def update_max_model_len(self, max_model_len):
        # the pool is fixed by num_gpu_blocks_override, so auto-fit cannot shorten the context
        if max_model_len != self.model_config.max_model_len:
            raise ValueError(
                f"max_model_len {max_model_len} differs from the configured {self.model_config.max_model_len}: "
                "the TSP pool is fixed, so the engine cannot fit a different length"
            )

    def initialize_from_config(self, kv_cache_config):
        chip = self._need_chip("initialize_from_config")
        if kv_cache_config.num_blocks > chip.info.num_blocks:
            raise ValueError(f"kv cache num_blocks {kv_cache_config.num_blocks} exceeds the chip's pool of {chip.info.num_blocks} blocks")
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError(f"the TSP backend serves one KV cache group, not {len(groups)}")
        if list(groups[0].layer_names) != list(self.get_kv_cache_spec()):
            raise ValueError(f"the KV cache group's layers {groups[0].layer_names} are not the chip's {chip.info.layers} layers")
        if not isinstance(chip, ManagedChip):
            chip.zero_pool()

    def get_supported_tasks(self):
        return ("generate",)

    def get_tsp_stats(self):
        return (self.model_runner.stats if self.model_runner else Stats()).snapshot()

    def reset_encoder_cache(self):
        return None
