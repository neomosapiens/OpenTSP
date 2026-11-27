"""Run two in-flight tokens on the same native pipeline and compare with serial issue."""
import os
from pathlib import Path

import numpy as np
import pytest

from opentsp_vllm.ship_chip import DEFAULT_CHECKPOINT, _showcase


@pytest.mark.skipif(not Path(DEFAULT_CHECKPOINT, "model.safetensors").is_file(), reason="GPT-2 checkpoint required")
def test_two_lane_wavefront():
    weights, pipeline, config = _showcase()
    w = weights.load_checkpoint(config, str(Path(DEFAULT_CHECKPOINT, "model.safetensors")))
    cache = os.environ.get("OPENTSP_SHIP_PACKAGE", pipeline.CACHE)
    p = pipeline.Pipeline(w, layers=2, resident=True, cache=cache)
    rows = [(7454, 0, 0, pipeline.S.counts_for(1)), (2402, 1, 1, pipeline.S.counts_for(2))]
    reference = [p.step(t, pos, slot=slot, counts=counts, check=False) for t, pos, slot, counts in rows]
    p.reset()
    outputs, stats = p.batch(rows, check=False)
    assert stats["cycles"] < sum(st["cycles"] for _, st in reference)
    assert stats["initiation_interval"] < reference[0][1]["cycles"]
    for result, (expected, _) in zip(outputs, reference, strict=True):
        np.testing.assert_array_equal(result.view("u4"), expected.view("u4"))
