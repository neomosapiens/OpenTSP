"""Cache publication, aperture accounting and bounded history without expensive model builds."""
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/showcase/gpt2small"))
from opentsp_vllm.models.gpt2 import pipeline as P


def _publish(path, builds, barrier):
    barrier.wait(timeout=10)
    def build(tmp):
        with builds.get_lock():
            builds.value += 1
        time.sleep(0.1)
        Path(tmp, "payload").write_text("complete")
        Path(tmp, "manifest.json").write_text("{}")
    P.ensure_cache(path, build)
    assert Path(path, "payload").read_text() == "complete"


def test_cache_same_key_has_one_publisher(tmp_path):
    ctx = mp.get_context("spawn")
    builds, barrier = ctx.Value("i", 0), ctx.Barrier(2)
    path = str(tmp_path / "key")
    workers = [ctx.Process(target=_publish, args=(path, builds, barrier)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=15)
            assert worker.exitcode == 0
        assert builds.value == 1
        inode = Path(path, "payload").stat().st_ino
        P.ensure_cache(path, lambda _: pytest.fail("published cache rebuilt"))
        assert Path(path, "payload").stat().st_ino == inode
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)


def test_cache_interrupt_removes_only_own_temporary_directory(tmp_path):
    path = str(tmp_path / "key")
    def interrupted(tmp):
        Path(tmp, "partial").write_text("unfinished")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        P.ensure_cache(path, interrupted)
    assert not Path(path).exists()
    assert not list(tmp_path.glob("key.tmp*"))
    P.ensure_cache(path, lambda tmp: Path(tmp, "manifest.json").write_text("{}"))
    assert Path(path, "manifest.json").is_file()


def test_incomplete_cache_is_preserved_and_reported(tmp_path):
    path = tmp_path / "key"
    path.mkdir()
    (path / "partial").write_text("evidence")
    with pytest.raises(RuntimeError, match="incomplete cache"):
        P.ensure_cache(str(path), lambda _: pytest.fail("overwrote incomplete cache"))
    assert (path / "partial").read_text() == "evidence"


@pytest.mark.parametrize("limit, retained", [(256, 256), (None, 5000), (0, 0)])
def test_chip_history_retention_over_many_steps(limit, retained):
    from opentsp_vllm.ship_chip import ShipChip
    chip = ShipChip(history_limit=limit)
    chip._pipe = SimpleNamespace(step=lambda *args, **kwargs: (None, {"cycles": 1}))
    plan = SimpleNamespace(real_lanes=[SimpleNamespace(token=0, position=0, logits=False)],
                           patches={"slots": [0], "counts": [np.zeros(64, np.int32)]})
    for _ in range(5000):
        chip.step(plan)
    assert len(chip.steps) == retained


@pytest.mark.parametrize("resident", [True, False])
def test_pipeline_counts_every_aperture_word_and_bounds_history(monkeypatch, tmp_path, resident):
    # Execute the real Pipeline.step; replace only the simulator and costly build.
    writes = []
    class System:
        def __init__(self, *args): pass
        def load_object(self, *args): pass
        def restart(self): pass
        def write_words(self, *args): writes.append(len(args[-1]))
        def read_words(self, chip, sli, addr, words): return bytes(words * 320)
        def dma_write(self, *args): pass
        def run_to_done(self, **kwargs): return None, {"cycle": 123, "reason": "done"}
        def service_scrub(self): pass
        def ras_snapshot(self, chip): return {"events": 0}
        def finish(self): pass
        def diagnostics(self, chip): return []
        def active_ticks(self, chip): return 23
        def egress(self, chip): return [(0, bytes(320))] * (4 * P.T)
    import opentsp_vllm.system as system
    monkeypatch.setattr(system, "System", System)
    monkeypatch.setattr(P, "source_digest", lambda *args, **kwargs: "test")
    monkeypatch.setattr(P, "build_cache", lambda w, layers, route, chips, directory, *args:
                        Path(directory, "manifest.json").write_text(json.dumps(dict(P.stage_manifest(w, layers, route, chips), units_sha256={"s0": "mock"}))))
    stage = SimpleNamespace(elf=b"\x7fELF", budget=100, dma_cycles=list(range(4 * P.T)),
                            runs=[(0, 0, bytes(320))], word_count=1,
                            externs={"map": (1, 0, 2), "mask": (2, 0, 1), "k0": (3, 0, 1)})
    monkeypatch.setattr(P, "Stage", lambda *args: stage)
    monkeypatch.setattr(P.S, "layer_params", lambda x: None)
    monkeypatch.setattr(P.S, "embed", lambda *args: np.zeros(P.T * P.sl.TILE, np.float32))
    monkeypatch.setattr(P.sl, "map_weights", lambda *args: {"map": bytes(640)})
    monkeypatch.setattr(P.sa, "enc_weights", lambda *args: {"mask": bytes(320)})
    w = SimpleNamespace(layers=[None], digest=lambda: "test")
    pipe = P.Pipeline(w, layers=1, resident=resident, cache=str(tmp_path), log=lambda _: None)
    for pos in range(600):
        writes.clear()
        _, stat = pipe.step(0, pos, check=False)
        assert stat["write_words"] * 320 == sum(writes)
        assert stat["write_words"] == ((4 if pos == 0 else 3) if resident else 5)
        assert stat["initial_write_words"] == int(pos == 0 or not resident)
        assert stat["control_write_words"] == 3
        assert stat["kv_write_words"] == stat["kv_read_words"] == int(not resident)
        assert stat["object_load_bytes"] == (4 if pos == 0 or not resident else 0)
        assert stat["collect_s"] >= 0
    assert len(pipe.stats) == 256
    assert pipe.stats[0]["pos"] == 344


def test_a_complete_stage_from_another_package_is_refused(tmp_path):
    unit = tmp_path / "s0"
    unit.mkdir()
    for name in P.artifacts.PAYLOAD:
        (unit / name).write_bytes(name.encode())
    P.artifacts.seal(unit, {"chip": 0})
    expected_digest = P.artifacts.file_digest(unit / "program.json")
    (unit / "sram.safetensors").write_bytes(b"different weights")
    P.artifacts.seal(unit, {"chip": 0})
    with pytest.raises(ValueError, match="does not match the package manifest"):
        P.Stage(tmp_path, 0, 0, expected_digest)


def test_model_loading_keeps_history_bound_and_three_manifest_caches(monkeypatch, tmp_path):
    import os
    from opentsp_vllm import ship_chip
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").touch()
    caches = tmp_path / "programs"
    caches.mkdir()
    for i in range(4):
        entry = caches / str(i)
        entry.mkdir()
        (entry / "manifest.json").write_text("{}")
        os.utime(entry, (1000 + i, 1000 + i))
    loaded = []
    def pipeline(*args, **kwargs):
        loaded.append(kwargs)
        return SimpleNamespace(dir=str(caches / "0"))
    from opentsp_vllm.models.gpt2.config import GPT2_SMALL
    config = GPT2_SMALL
    weights = SimpleNamespace(load_checkpoint=lambda cfg, path: object())
    monkeypatch.setattr(ship_chip, "_showcase", lambda heavy=True: (weights, SimpleNamespace(Pipeline=pipeline), config) if heavy else config)
    chip = ship_chip.ShipChip(layers=2, programs=str(caches), history_limit=7)
    chip.load_model(str(checkpoint))
    assert loaded[0]["history_limit"] == 7
    assert loaded[0]["cache"] == str(caches)
    assert sorted(p.name for p in caches.iterdir() if p.is_dir()) == ["0", "2", "3"]
    assert sorted(p.name for p in caches.iterdir() if p.is_file()) == ["1.lock"]
    assert (caches / "0" / "manifest.json").is_file(), "the active completed cache is retained"
