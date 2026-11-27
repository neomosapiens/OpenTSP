"""Serve a sealed four-chip GPT-2 model with resident weights and KV."""

from collections import deque
import hashlib
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from opentsp_tools import profile
from opentsp_tools.programs import ref_wide as rw
from opentsp_vllm import ship_plan
from opentsp_vllm.system import System

from . import pipeline, system
from .config import GPT2_SMALL
from .weight_layout import Layout, RESIDENT
from .four_chip_execution import drive_panel, drive_sequential, first_images
from .four_chip_packages import SEALS, manifests, select_variant
from .four_chip_pages import PageTransferBudget, open_wave

POOL = re.compile(r"l[0-3](?:k\d+|vslice\d+)")
PAIR_PERIOD = 30534
PAIR_TRIM = 1614
WIDTH = {"full": 1, "sparse": 1, "paired": 2, "single-page": 1, "single-page-pair": 2}
PAGED = {"single-page", "single-page-pair"}


def resident_words(stage):
    return {name: item for name, item in zip(stage.externs, stage.runs, strict=True) if RESIDENT.fullmatch(name)}


def install_static(sim, stages, *, warm):
    written = 0
    for chip, stage in enumerate(stages):
        for name, (sl, address, payload) in zip(stage.externs, stage.runs, strict=True):
            if not warm or name in stage.warm_names:
                sim.write_words(chip, sl, address, payload)
                written += len(payload)
    return written


class FourChipPipeline(pipeline.Pipeline):
    """Switch the program while leaving each layer's physical data resident."""

    def __init__(self, model, directory, log=print):
        self.w, self.layers = model, 12
        self.chips, self.route, self.kv_chunks = 4, list(range(4)), 2
        self.pool_slots, _ = ship_plan.kv_limits(self.kv_chunks)
        self.p = profile.load("ref")
        self.dir = os.fspath(Path(directory).resolve())
        self.package_sha256 = hashlib.sha256(":".join(SEALS.values()).encode()).hexdigest()
        expected = pipeline.stage_manifest(model, self.layers, self.route, self.chips, self.kv_chunks)
        layout = Layout(model, self.package_sha256)
        self.variants, self.page_plans = {}, {}
        for name, (root, manifest) in manifests(directory).items():
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError(f"{root}: incompatible checkpoint or profile")
            stages = [pipeline.Stage(root, i, i, manifest["units_sha256"][f"s{i}"], layout=layout)
                      for i in self.route]
            if name == "full":
                self.stages = stages
                resident = [resident_words(stage) for stage in stages]
            elif [resident_words(stage) for stage in stages] != resident:
                raise ValueError(f"{root}: program changes resident weights or KV layout")
            if len(stages[0].dma_cycles) != 24 * WIDTH[name]:
                raise ValueError(f"{root}: incompatible ingress layout")
            trim = PAIR_TRIM if name == "single-page-pair" else 0
            context = SimpleNamespace(p=self.p, stages=stages, wave=None)
            period, waves = pipeline.Pipeline._wave_schedule(context, repeat_trim=trim)
            if trim and period != PAIR_PERIOD:
                raise ValueError(f"{root}: cadence differs from qualified period")
            self.variants[name] = (stages, period, waves, first_images(self.p, stages))
            if name in PAGED:
                self.page_plans[name] = open_wave(self.p, root, stages, period, waves)
        self.Ps = [system.layer_params(model.layers[i]) for i in range(self.layers)]
        self.twin_k = [np.zeros((self.pool_slots, 6, 128), np.float16) for _ in range(self.layers)]
        self.twin_v = [np.zeros_like(self.twin_k[0]) for _ in range(self.layers)]
        self.pool, self._warm, self.active_variant = {}, None, None
        self.completion_flips, self.health_target = [], None
        self.stats = deque(maxlen=256)
        log(f"four-chip GPT-2 programs verified at {self.dir}")

    def logits(self, output):
        """The vocabulary chip already produced the logits."""
        return output

    def clear_canary(self):
        if self._warm is not None:
            for chip, stage in enumerate(self.stages[:3]):
                for name, (sl, address, words) in stage.externs.items():
                    if POOL.fullmatch(name):
                        self._warm.write_words(chip, sl, address, bytes(words * self.p.vlanes))
        self.pool.clear()
        for array in self.twin_k + self.twin_v:
            array.fill(0)

    def batch(self, rows, check=True, *, observe=None):
        if not 1 <= len(rows) <= 8:
            raise ValueError("a four-chip wave takes one to eight token rows")
        if any(not 0 <= slot < self.pool_slots or np.shape(counts) != (2, ship_plan.PAGES_PER_CHUNK)
               for _, _, slot, counts in rows):
            raise ValueError("invalid slot or mask for the 256-position KV pool")
        started = time.perf_counter()
        sim, self._warm = self._warm, None
        name = select_variant(rows)
        packets = None
        if name in PAGED:
            prepare, reset = self.page_plans[name]
            if sim is None or self.active_variant != name:
                reset()
            width = WIDTH[name]
            groups = [rows[i:i + width] for i in range(0, len(rows), width)]
            try:
                packets = prepare(groups, self.variants[name][1])
            except PageTransferBudget:
                name = "paired" if width == 2 else "sparse"
        stages, period, waves, boot = self.variants[name]
        initial = switched = 0
        if sim is None:
            sim = System("ref", self.chips)
            initial = install_static(sim, stages, warm=False) // self.p.vlanes
        elif self.active_variant != name:
            switched = install_static(sim, stages, warm=True) // self.p.vlanes
        self.active_variant = name
        if len(rows) == 1 and name != "single-page":
            period, waves = 0, ()
        launch_overhead = time.perf_counter() - started
        target, self.health_target = self.health_target, None
        flips, self.completion_flips = self.completion_flips, []
        if name in PAGED or name == "paired":
            outputs, metric = drive_panel(sim, self.w, stages, rows, self.p, period, waves,
                                          boot=boot, health_target=target, flips=flips,
                                          prepare_pages=(lambda _groups, _period: packets) if packets else None,
                                          observe=observe)
        else:
            outputs, metric = drive_sequential(sim, self.w, self.kv_chunks, stages, period, waves, rows,
                                               health_target=target, flips=flips, boot=boot,
                                               mask_chunk=None if name == "full" else 0, observe=observe)
        if check:
            for (token, pos, slot, counts), out in zip(rows, outputs, strict=True):
                expected = self.twin_step(system.embed(self.w, token, pos), slot, counts)
                normalized = rw.layernorm_wide(expected, self.w.lnf_g, self.w.lnf_b, GPT2_SMALL.ln_eps, tile=128)
                # The reference tiles pad beyond the public vocabulary.
                expected_logits = pipeline.host_logits(self.w, normalized)[:GPT2_SMALL.vocab]
                if not np.array_equal(out.view(np.uint32), expected_logits.view(np.uint32)):
                    raise RuntimeError(f"step {pos}: four-chip logits differ from the twin")
        self._warm = sim
        _, pos, slot, _ = rows[0]
        control = len(rows) * 12 * (50 + (8 if name == "full" else 4))
        stat = dict(pos=pos, slot=slot, cycles=metric["cycles"], load_s=metric["load_s"] + launch_overhead,
                    run_s=metric["native_seconds"], collect_s=metric["collect_s"],
                    write_words=initial+control+switched, initial_write_words=initial,
                    control_write_words=control, kv_write_words=0, kv_read_words=0,
                    object_load_bytes=0, host_vectors_in=24*len(rows),
                    host_vectors_out=metric.get("host_vectors_out", 4*393*len(rows)), stage_ticks=metric["stage_ticks"],
                    tokens=len(rows), initiation_interval=0 if len(rows) == 1 else period//WIDTH[name],
                    launch_cycles=metric["launch_cycles"], completion_cycles=metric["arrival_cycles"],
                    host_text_bytes=metric["text_bytes"], schedule_variant=name, schedule_reload_words=switched,
                    page_text_words=metric.get("page_text_words", 0),
                    minimum_page_text_slack=metric.get("minimum_page_text_slack"),
                    linked_panel_windows=metric.get("linked_panel_windows", 0))
        if check:
            stat["twin_equal"] = True
        self.stats.append(stat)
        return outputs, stat
