"""Release program identities for the resident four-chip, 256-position layout."""

import hashlib
import json
from pathlib import Path

import numpy as np

from opentsp_vllm.ship_plan import PAGE, PAGES_PER_CHUNK

FIRST_CHUNK_SLOTS = PAGE * PAGES_PER_CHUNK

# Each sealed manifest in turn binds all four unit descriptors and payloads.
SEALS = {
    'full': '97df781026b79bb6a9792ecb61728e65b3c0e243e3cda6bf94c3efa018a68f88',
    'sparse': '854a11f9535770422a4b23985db6517aef421f1c1c8e2c3bb3e7270d8be92e41',
    'paired': '7764df511d7b73f9c3e76b5fcd9e05233005f239f6439b3cae2d902f3ce0f66c',
    'single-page': '260e50615737cd094a1041a51c95ab4a4369c9e372a39d89e7ec6f66178807f3',
    'single-page-pair': 'dbc386f6083380826af72e4bfd81477db26879d2e49257b603f645eb08bdbffa',
}

def admits(rows):
    """Every row appends to its sole active page, and the pages are distinct."""
    return all(
        0 <= slot < FIRST_CHUNK_SLOTS and counts.shape == (2, PAGES_PER_CHUNK)
        and np.count_nonzero(counts) == 1 and 1 <= counts[0, slot // PAGE] <= PAGE
        and slot % PAGE < counts[0, slot // PAGE]
        for _, _, slot, counts in rows
    ) and len({slot // PAGE for _, _, slot, _ in rows}) == len(rows)


def select_variant(rows):
    """Select the sealed variant for a validated batch of resident KV rows."""
    if any(slot >= FIRST_CHUNK_SLOTS or not np.any(counts[0]) or np.any(counts[1]) for _, _, slot, counts in rows):
        return "full"
    panel = len(rows) == 8 and len({slot // PAGE for _, _, slot, _ in rows}) == 8
    if admits(rows):
        return "single-page-pair" if panel else "single-page"
    return "paired" if panel else "sparse"


def manifests(directory):
    root = Path(directory).expanduser().resolve()
    entries = {}
    for name, digest in SEALS.items():
        stage_root = root / name
        path = stage_root / "manifest.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"{path}: expected the sealed four-chip {name} package")
        entries[name] = stage_root, json.loads(raw)
    return entries
