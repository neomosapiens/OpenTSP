"""Live fault campaign on both published serving packages (just ras-qualify)."""

import os
from pathlib import Path

import pytest


def _published_model(kind, tmp_path, monkeypatch):
    from vllm import LLM

    root = Path(os.environ.get("OPENTSP_SHOWCASE", Path.home() / ".cache/opentsp/showcase"))
    if kind == "tiny":
        model = root / "model"
        options = dict(chip=kind, programs=str(root / "serving"))
    else:
        bundle = Path(os.environ.get("OPENTSP_SHIP_PACKAGE", Path.home() / ".cache/opentsp/gpt2-four-chip"))
        model = bundle / "model"
        options = dict(chip=kind, weights=str(bundle / "weights.safetensors"), programs=str(bundle))
    monkeypatch.setenv("OPENTSP_RAS_STATE_DIR", str(tmp_path))
    return LLM(model=str(model), dtype="float32", max_model_len=64, max_num_seqs=2, max_num_batched_tokens=8,
              enable_prefix_caching=True, enable_chunked_prefill=True, additional_config={"opentsp": options})


@pytest.mark.parametrize("kind", ["tiny", "ship"])
def test_recovery_and_recurrence_on_published_model(kind, tmp_path, monkeypatch):
    from vllm import SamplingParams
    llm = _published_model(kind, tmp_path, monkeypatch)
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    chip = core.model_executor.driver_worker.worker.chip
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=4)
    def inject(bits):
        chip.backend.inject_sram(0 if kind == "tiny" else 3, 87, 8191, 0, bits)

    def submit(rid):
        engine.add_request(rid, {"prompt_token_ids": [1]}, params)

    def step():
        return {r.request_id: (tuple(r.outputs[0].token_ids), r.outputs[0].finish_reason) for r in engine.step()}

    try:
        submit("resident-a")
        submit("resident-b")
        submit("host-only")
        first = step()
        assert set(first) == {"resident-a", "resident-b"} and all(len(v[0]) == 1 for v in first.values())
        inject([0])
        second = step()
        assert all(len(v[0]) == 2 and v[1] is None for v in second.values())
        inject([128, 129])
        previous = chip.backend
        with previous.sock.dup() as old_channel:
            failed = step()
            old_channel.settimeout(2)
            assert old_channel.recv(1) == b""
        assert previous.proc.poll() is not None and chip.backend.incarnation != previous.incarnation
        assert failed == {rid: (prefix, "error") for rid, (prefix, _) in second.items()}
        assert chip.failure is None
        assert chip.journal.db.execute("SELECT state FROM history ORDER BY rowid").fetchall() == [
            ("starting",), ("ready",), ("fenced",), ("recovering",), ("ready",)]
        fresh = step()
        assert fresh["host-only"] == (first["resident-a"][0], None)
        inject([128, 129])
        assert step()["host-only"] == (fresh["host-only"][0], "error")
        assert "quarantined" in chip.failure
        submit("after-quarantine")
        assert step()["after-quarantine"] == ((), "error")
        assert chip.journal.db.execute("SELECT count(*) FROM history WHERE state='recovering'").fetchone() == (1,)
    finally:
        llm.llm_engine.engine_core.shutdown()


def test_wedged_device_is_reaped_and_remains_quarantined(tmp_path, monkeypatch):
    import signal
    from functools import partial
    from opentsp_vllm.ras import DomainFault, Journal, ManagedChip
    from opentsp_vllm.tiny_chip import TinyChip

    root = Path(os.environ.get("OPENTSP_SHOWCASE", Path.home() / ".cache/opentsp/showcase")) / "serving"
    monkeypatch.setenv("OPENTSP_RAS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENTSP_RAS_DOMAIN", "watchdog")
    device = ManagedChip(partial(TinyChip, root, 8, (32, 64)), "tiny")
    try:
        device.load_model(str(root / "weights.safetensors"))
        old = device.backend
        old.timeout = 0.15
        os.kill(old.proc.pid, signal.SIGSTOP)
        with pytest.raises(DomainFault, match="quarantined"):
            device.zero_blocks([1])
        assert old.proc.poll() == -signal.SIGKILL
    finally:
        device.close()
    with pytest.raises(RuntimeError, match="operator clearance"):
        Journal(tmp_path, "watchdog")
    journal = Journal(tmp_path, "watchdog", maintenance=True)
    try:
        assert [s for s, in journal.db.execute("SELECT state FROM history ORDER BY rowid")] == [
            "starting", "ready", "fenced", "quarantined"]
    finally:
        journal.close()

@pytest.mark.parametrize("kind", ["tiny", "ship"])
def test_fault_invalidates_a_previously_reused_prefix(kind, tmp_path, monkeypatch):
    from vllm import SamplingParams
    llm = _published_model(kind, tmp_path, monkeypatch)
    engine = llm.llm_engine
    chip = engine.engine_core.engine_core.model_executor.driver_worker.worker.chip
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=1)

    def generate(ids):
        return llm.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]

    try:
        prompt = [1] * 16 + [2]
        expected = generate(prompt)
        cached = generate(prompt)
        assert cached.num_cached_tokens == 16 and cached.outputs[0].token_ids == expected.outputs[0].token_ids
        chip.backend.inject_sram(0 if kind == "tiny" else 3, 87, 8191, 0, [128, 129])
        assert generate([2]).outputs[0].finish_reason == "error"
        fresh = generate(prompt)
        assert fresh.num_cached_tokens == 0
        assert fresh.outputs[0].token_ids == expected.outputs[0].token_ids
    finally:
        engine.engine_core.shutdown()
