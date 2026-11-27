"""TinyStories on one ISS chip with immutable programs and resident physical KV."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from . import serving_artifacts as artifacts
from .ras import check_completion, probe_word
from .contracts import ChipInfo, ChipResult, TspChipError
from .programs import Decoder
from .step_plan import BLOCK, GRID_C, POOL_BLOCKS, StepPlanError, plan_step

CHUNK = 128
CHUNKS = POOL_BLOCKS * BLOCK // CHUNK


def physical_mask(blocks, position):
    blocks = tuple(blocks[: position // BLOCK + 1])
    if (
        not 0 <= position < max(GRID_C)
        or len(blocks) != position // BLOCK + 1
        or len(set(blocks)) != len(blocks)
        or any(not 0 < b < POOL_BLOCKS for b in blocks)
    ):
        raise StepPlanError(f"invalid TinyStories block table {blocks} for position {position}")
    result = np.full(CHUNKS * CHUNK, -np.inf, np.float32)
    for j, block in enumerate(blocks):
        size = min(BLOCK, position + 1 - BLOCK * j)
        result[BLOCK * block : BLOCK * block + size] = 0
    return result


class TinyChip:
    def __init__(self, directory, layers, grid_c, mode="warm"):
        from . import _tsp_iss

        self.root = Path(directory)
        m = json.loads((self.root / "manifest.json").read_text())
        if m.get("execution") != "physical-kv":
            raise ValueError("rebuild the TinyStories package with `just showcase-prepare` for immutable programs")
        self._chip = _tsp_iss.Chip("ref")
        self._chip.restart_mode(mode)
        expected = dict(
            schema=artifacts.SCHEMA,
            version=artifacts.VERSION,
            model="tinystories",
            spec_version="0.1",
            chips=1,
            route=[0],
            layers=layers,
            profile="ref",
            hidden=64,
            heads=16,
            shared_preload="weights.safetensors",
            profile_sha256=self._chip.profile_sha256,
        )
        if m.get("metadata_sha256") != artifacts.metadata_digest(m):
            raise ValueError("TinyStories manifest metadata checksum differs")
        for field, value in expected.items():
            if m.get(field) != value:
                raise ValueError(f"TinyStories package {field} must be {value!r}, found {m.get(field)!r}")
        self.info = ChipInfo(layers, m["vocab"], m["hidden"], m["heads"], POOL_BLOCKS, grid_c)
        self._regions = m["regions"]
        self._identity = m["metadata_sha256"]
        self._stage_hashes = m["stage_descriptors"]
        self._sha = m["package_sha256"]
        self._device_sha = m["device_sha256"]
        self._loaded = {}
        self.failure = None

    def load_model(self, source):
        source = Path(source)
        if source.is_dir():
            source /= "weights.safetensors"
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != self._sha:
            raise ValueError("TinyStories shared weight package digest differs from the manifest")
        from .models.tinystories.program_cache import materialize

        model, device = materialize(data, self.info.layers, expected=self._device_sha, package_identity=self._identity)
        self._chip.load_package(device)
        self._wte, self._wpe = model.wte, model.wpe.astype(np.float32)

    def warm_up(self):
        self._stage("head")

    def _stage(self, name):
        if name not in self._loaded:
            path = self.root / name
            rec = json.loads((path / "program.json").read_text())
            payload = artifacts.verified_payload(path, rec, payload=artifacts.ELF_PAYLOAD)
            if rec["metadata_sha256"] != self._stage_hashes[name]:
                raise ValueError(f"{name} differs from the stage selected by the model manifest")
            if rec["stage"] != name:
                raise ValueError(f"stage {name} contains the record for {rec['stage']}")
            with np.load(io.BytesIO(payload["bindings.npz"]), allow_pickle=False) as z:
                outputs = [tuple(r) for r in z["outputs"].tolist()]
                if name == "head":
                    self._decode = Decoder(z, 1, self.info.vocab)
            self._chip.load_program(
                name,
                payload["program.elf"],
                outputs=outputs,
                resident_text=True,
            )
            self._loaded[name] = rec["max_cycles"]
        return self._loaded[name]

    def plan_step(self, requests):
        return plan_step(requests, vocab=self.info.vocab, grid_c=self.info.grid_c, patch_kinds=())

    def _inputs(self, lane, mask):
        vl = self._chip.vlanes
        x = np.zeros(vl, dtype="<f4")
        x[: self.info.hidden] = self._wte[lane.token].astype(np.float32) + self._wpe[lane.position]
        self._write_vectors("hidden", x[None])
        padded = np.zeros((CHUNKS, vl), dtype="<f4")
        padded[:, :CHUNK] = mask.reshape(CHUNKS, CHUNK)
        self._write_vectors("mask", padded)
        block, row = lane.block_table[-1], lane.position % BLOCK
        _, base, n = self._regions["keys"]
        maps = np.zeros((self.info.layers, 9, vl // 16, 8), dtype="<u2")
        for layer in range(self.info.layers):
            maps[layer, :8, :, 0] = base + n - 1
            maps[layer, row % 8, :, 0] = base + (layer * CHUNKS * CHUNK + block * BLOCK + row) // 8
            maps[layer, 8, :, 0] = self._regions[f"v{layer}"][1] + BLOCK * block + row
        slices, addr, _ = self._regions["slotmaps"]
        self._chip.write_words(slices[0], addr, maps.tobytes())

    def _write_vectors(self, name, words):
        planes = words.view(np.uint8).reshape(len(words), self._chip.vlanes, 4)
        slices, addr, _ = self._regions[name]
        for i, s in enumerate(slices):
            self._chip.write_words(s, addr, planes[:, :, i].tobytes())

    def package_identity(self):
        return self._identity

    def clear_canary(self):
        self.zero_blocks([1])

    def inject_sram(self, chip, slice_id, addr, lane, bits):
        if chip != 0:
            raise ValueError("TinyStories has only chip 0")
        for bit in bits:
            self._chip.inject_sram_next_step(slice_id, addr, lane, bit)

    def probe_sram(self, chip, slice_id, addr):
        if chip != 0:
            raise ValueError("TinyStories has only chip 0")
        probe_word(lambda: self._chip.read_words(slice_id, addr, 1),
                   lambda data: self._chip.write_words(slice_id, addr, data), self._chip.vlanes)

    def _checked_step(self, name, limit):
        try:
            result = self._chip.step(name, None, limit)
        except TspChipError as exc:
            check_completion([(0, self._chip.ras_snapshot())], exc)
        check_completion([(0, self._chip.ras_snapshot())])
        return result

    def step(self, plan):
        cycles = 0
        rows = {}
        # The vLLM bucket places logit lanes first. Prefill must write earlier
        # positions before later positions of the same request read their KV.
        ordered = sorted(enumerate(plan.real_lanes), key=lambda item: item[1].position)
        if self._chip is None:
            raise RuntimeError("TinyStories chip is closed")
        if self.failure:
            raise TspChipError(f"TinyStories chip failed during an earlier step: {self.failure}")
        masks = [(i, lane, physical_mask(lane.block_table, lane.position)) for i, lane in ordered]

        slots = {(lane.block_table[-1], lane.position % BLOCK) for _, lane, _ in masks}
        if len(slots) != len(masks):
            raise StepPlanError("two scheduled tokens write the same TinyStories KV slot")

        def run(name):
            nonlocal cycles
            limit = self._stage(name)
            words, ticks, _ = self._checked_step(name, limit)
            cycles += ticks
            return words

        try:
            for i, lane, mask in masks:
                self._inputs(lane, mask)
                chunks = sorted({b * BLOCK // CHUNK for b in lane.block_table})
                normalize = f"single{chunks[0]}" if len(chunks) == 1 else "normalize"
                if len(chunks) > 1:
                    run("reset")
                for layer in range(self.info.layers):
                    run(f"pre{layer}")
                    for c in chunks:
                        run(f"score{layer}_{c}")
                    run(normalize)
                    for c in chunks:
                        run(f"context{layer}_{c}")
                    run(f"post{layer}")
                if lane.logits:
                    words = run("head")
                    rows[i] = self._decode(words, 1)[0]
        except (TspChipError, ValueError, RuntimeError, OSError) as exc:
            self.failure = str(exc)
            raise
        logits = np.stack([rows[i] for i in sorted(rows)]) if rows else np.empty((0, self.info.vocab), np.float32)
        return ChipResult(logits, cycles)

    def _pool_regions(self, block):
        if not 0 <= block < POOL_BLOCKS:
            raise ValueError(f"invalid TinyStories pool block {block}")
        slices, addr, _ = self._regions["keys"]
        for layer in range(self.info.layers):
            for s in slices:
                yield s, addr + layer * CHUNK * CHUNKS // 8 + block * 2, 2
            vs, va, _ = self._regions[f"v{layer}"]
            for s in vs:
                yield s, va + block * BLOCK, BLOCK

    def zero_blocks(self, block_ids):
        block_ids = list(block_ids)
        for b in block_ids:
            if not 0 < b < POOL_BLOCKS:
                raise ValueError(f"invalid TinyStories pool block {b}")
        for b in block_ids:
            for s, a, n in self._pool_regions(b):
                self._chip.write_words(s, a, bytes(self._chip.vlanes * n))

    def zero_pool(self):
        self.zero_blocks(range(1, POOL_BLOCKS))

    def copy_blocks(self, copies):
        copies = list(copies)
        for src, dst in copies:
            if not 0 < src < POOL_BLOCKS or not 0 < dst < POOL_BLOCKS:
                raise ValueError(f"invalid TinyStories block copy {src} to {dst}")
        slices, addr, _ = self._regions["slotmaps"]

        def set_map(plane, value):
            word = (value.to_bytes(2, "little") + bytes(14)) * (self._chip.vlanes // 16)
            self._chip.write_words(slices[plane], addr, word)

        try:
            for src, dst in copies:
                if src == dst:
                    continue
                for (sl, a, n), (_, dest, _) in zip(self._pool_regions(src), self._pool_regions(dst)):
                    if sl % 4:
                        continue  # one program moves all four byte planes of the quad
                    name = f"copy{(sl - 44) // 4}"
                    limit = self._stage(name)
                    for offset in range(n):
                        set_map(0, a + offset)
                        set_map(1, dest + offset)
                        self._checked_step(name, limit)
        except (TspChipError, ValueError, RuntimeError, OSError) as exc:
            self.failure = str(exc)
            raise

    def stats(self):
        return self._chip.stats()

    def close(self):
        self._chip = None
        self._loaded.clear()
