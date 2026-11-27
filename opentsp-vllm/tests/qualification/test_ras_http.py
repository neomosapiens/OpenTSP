"""Observe a real service interrupt through the streaming completions endpoint."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["tiny", "ship"])
def test_stream_interrupt_and_fresh_request(tmp_path, kind):
    # Test-only hook in the private device interpreter. The serving API has no
    # fault injection endpoint. Trigger after two published positions, once.
    fixture = tmp_path / "sitecustomize.py"
    fixture.write_text("""
import os
import sys
if 'opentsp_vllm.ras_device' in sys.orig_argv:
    if os.environ["TEST_BACKEND"] == "tiny":
        from opentsp_vllm.tiny_chip import TinyChip as Backend
    else:
        from opentsp_vllm.ship_chip import ShipChip as Backend
    step = Backend.step
    def interrupted(self, plan):
        if any(lane.position == 2 for lane in plan.real_lanes):
            try:
                with open(os.environ['FAULT_RECEIPT'], 'x') as receipt:
                    receipt.write('injected')
            except FileExistsError:
                pass
            else:
                self.inject_sram(0 if os.environ["TEST_BACKEND"] == "tiny" else 3, 87, 8191, 0, [128, 129])
        return step(self, plan)
    Backend.step = interrupted
""")
    root = Path(os.environ.get("OPENTSP_SHOWCASE", Path.home() / ".cache/opentsp/showcase"))
    python_paths = [str(tmp_path)]
    inherited_path = os.environ.get("PYTHONPATH")
    if inherited_path:
        python_paths.append(inherited_path)
    env = dict(os.environ, TEST_BACKEND=kind, OPENTSP_RAS_STATE_DIR=str(tmp_path), FAULT_RECEIPT=str(tmp_path / "fault"),
               PYTHONPATH=os.pathsep.join(python_paths))
    env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    limits = ["--host", "127.0.0.1", "--port", str(port), "--max-model-len", "64",
              "--max-num-seqs", "2", "--max-num-batched-tokens", "8"]
    if kind == "ship":
        from opentsp_vllm.serve import build

        bundle = os.environ.get("OPENTSP_SHIP_PACKAGE", str(Path.home() / ".cache/opentsp/gpt2-four-chip"))
        serve_args = build(["--tsp-programs", bundle, *limits])
        model = "gpt2"
    else:
        model = str(root / "model")
        options = {"chip": kind, "programs": str(root / "serving")}
        serve_args = ["serve", model, *limits, "--dtype", "float32",
                      "--additional-config", json.dumps({"opentsp": options})]
    args = [sys.executable, "-m", "vllm.entrypoints.cli.main", *serve_args]

    def complete(stream, count):
        body = dict(model=model, prompt=[1], temperature=0, ignore_eos=True, stream=stream, max_tokens=count)
        request = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                         {"content-type": "application/json"})
        with urllib.request.urlopen(request, timeout=90) as response:
            if stream:
                return [json.loads(line[5:]) for line in response if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"]
            return json.load(response)

    with (tmp_path / "service.log").open("w") as log:
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 150
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except OSError:
                    time.sleep(0.25)
            else:
                raise AssertionError((tmp_path / "service.log").read_text()[-4000:])
            events = complete(True, 5)
            assert (tmp_path / "fault").exists()
            assert "error" in events[-1] or events[-1]["choices"][0]["finish_reason"] == "error", events
            prefix = "".join(e["choices"][0]["text"] for e in events[:-1])
            assert prefix and all(e["choices"][0]["finish_reason"] is None for e in events[:-1])
            fresh = complete(False, 2)["choices"][0]
            assert fresh["finish_reason"] == "length" and fresh["text"] == prefix
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
