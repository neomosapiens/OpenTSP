"""Translate GPT-2 logical tensors into the four-chip SRAM layout."""
import hashlib
import json
import re

import numpy as np

from opentsp_tools.programs import data
from opentsp_vllm import device_layout_cache
from . import system

LANES = 320
TILE = 128
RESIDENT = re.compile(r"(?:l[0-3](?:k\d+|vslice\d+|w(?:qkv|o|u|d)_[0-9a-f]+_[0-9a-f]+)|wh_[0-9a-f]+_[0-9a-f]+)")


def stage_parameters(model, chip):
    """Name the checkpoint matrices and FP32 vectors visible to one stage."""
    if chip == 3:
        padded_vocab = ((len(model.wte) + TILE - 1) // TILE) * TILE
        rows = np.zeros((padded_vocab, model.cfg.hidden), np.float16)
        rows[:len(model.wte)] = model.wte
        return {"wh": rows}, {"g": model.lnf_g, "b": model.lnf_b}
    matrices, vectors = {}, {}
    for index, layer in enumerate(model.layers[4 * chip:4 * (chip + 1)]):
        for name, value in system.layer_params(layer).items():
            dest = matrices if value.ndim == 2 else vectors
            dest[f"l{index}{name}"] = value
    return matrices, vectors


def pack_matrix(matrix):
    """Pack each 128×256 coefficient tile in the MXM's sixteen byte-plane bands."""
    rows, cols = matrix.shape
    tiles = matrix.astype("<f2", copy=False).view(np.uint8).reshape(rows // TILE, 16, 8, cols // 256, 256, 2)
    order = tiles.transpose(0, 3, 2, 5, 1, 4).reshape(rows // TILE, cols // 256, 16, 16, 256)
    packed = np.zeros((*order.shape[:-1], LANES), dtype=np.uint8)
    packed[..., :256] = order
    return packed


def vector_word(vector, tile, plane, placement):
    lanes = np.zeros(LANES, dtype="<f4")
    values = vector[tile * TILE:(tile + 1) * TILE]
    if placement in ("low", "both"):
        lanes[:len(values)] = values
    if placement in ("high", "both"):
        lanes[TILE:TILE + len(values)] = values
    return data.to_planes(lanes)[plane]


def tensor_payload(tensor, constants, matrices, vectors):
    name, words, kind, parameter = tensor
    if kind == "zero":
        payload = bytes(words * LANES)
    elif kind == "literal":
        payload = constants[parameter:parameter + words * LANES]
    elif kind == "matrix":
        # The symbol names the physical range and the layout names its coefficient matrix.
        matrix, groups = parameter
        payload = b"".join(matrices[matrix][tile, block, band].tobytes() for tile, block, band in groups)
    elif kind == "vector":
        payload = vector_word(vectors[parameter[0]], *parameter[1:])
    else:
        raise ValueError(f"{name}: unknown preload source {kind!r}")
    if len(payload) != words * LANES:
        raise ValueError(f"{name}: wrong preload length")
    return payload


class Layout:
    """Share materialized coefficient runs across schedule variants."""

    def __init__(self, model, release_identity, *, directory=None):
        self.model = model
        self.identity = [model.digest(), release_identity]
        self.directory = directory
        self.stages = {}
        self.payloads = {}

    def resolve(self, chip, recipe, elf_identity):
        if not recipe:
            raise ValueError("missing layout recipe")
        rows = json.loads(recipe)
        key = json.dumps(["gpt2", self.identity, chip, elf_identity, hashlib.sha256(recipe).hexdigest()]).encode()
        sizes = {}
        for name, count, kind, source in rows:
            if name in sizes or kind not in ("matrix", "vector") or type(count) is not int or count <= 0:
                raise ValueError(f"invalid GPT-2 layout source {kind!r} at {name!r}")
            sizes[name] = count * LANES

        keys = {name: (chip, json.dumps([count, kind, source], separators=(",", ":")))
                for name, count, kind, source in rows}

        def images():
            if chip not in self.stages:
                matrices, vectors = stage_parameters(self.model, chip)
                self.stages[chip] = {key: pack_matrix(value) for key, value in matrices.items()}, vectors
            packed, vectors = self.stages[chip]
            for row in rows:
                ref = keys[row[0]]
                if ref not in self.payloads:
                    self.payloads[ref] = tensor_payload(row, b"", packed, vectors)
                yield self.payloads[ref]

        def decode(data):
            if sum(sizes.values()) != len(data):
                raise ValueError("prebuilt GPT-2 layout length differs from its verified recipe")
            bindings, offset = {}, 0
            checked = {}
            view = memoryview(data)
            for name, size in sizes.items():
                ref = keys[name]
                image = view[offset:offset+size]
                offset += size
                existing = self.payloads.get(ref, checked.get(ref))
                if existing is None:
                    existing = checked[ref] = bytes(image)
                elif image != existing:
                    raise ValueError("prebuilt GPT-2 variants disagree on shared coefficients")
                bindings[name] = existing
            self.payloads.update(checked)
            return bindings

        if self.directory or device_layout_cache.enabled():
            return device_layout_cache.resolve(key, lambda: b"".join(images()), decode, directory=self.directory)
        return dict(zip(sizes, images(), strict=True))
