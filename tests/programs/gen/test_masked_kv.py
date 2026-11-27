"""Real two-chunk attention: masked stale NaN/Inf must equal the finite baseline.

Run with simulator executable paths as arguments (Sail and ISS). Expectations
are captured before poisoning and do not use the poisoned computation's result.
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
runner_spec = importlib.util.spec_from_file_location("program_test_runner", ROOT / "tests/programs/run.py")
program_runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(program_runner)

from opentsp_vllm.models.gpt2 import attention as A
from opentsp_vllm.models.gpt2 import system as S  # noqa: E402
from opentsp_tools import profile


def run(executables):
    original = A.rw.attention_pool
    def expected_before_poison(q, k, v, mask, *args, **kwargs):
        want = original(q, k, v, mask, *args, **kwargs)
        assert np.isfinite(want).all()
        for chunk in range(2):
            masked = np.flatnonzero(np.isneginf(mask[128 * chunk:128 * (chunk + 1)]))[:3] + 128 * chunk
            assert len(masked) == 3
            for slot, value in zip(masked, (np.inf, -np.inf, np.nan)):
                k[slot, :, :] = value
                v[slot, :, :] = value
        return want
    with patch.object(A.rw, "attention_pool", expected_before_poison), patch.dict(os.environ, {"ATTN_DEBUG": ""}):
        text, case = A.attention(profile.load("ref"), tiles=1, chunks=2, name="masked_nonfinite")
    with tempfile.TemporaryDirectory(prefix="opentsp-masked-kv-") as directory:
        Path(directory, case["program"]).write_text(text)
        runner = program_runner.Runner(S.argparse_ns())
        input_text, _ = runner.build_input(case, directory, directory)
        for index, executable in enumerate(executables):
            out = runner.simulate(executable, case, input_text, directory, f"sim{index}")
            runner.compare(case, out)
            assert not out["diags"], out["diags"]
            print(f"{executable}: masked NaN/+Inf/-Inf in both chunks equals finite baseline", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: test_masked_kv.py SIMULATOR [SIMULATOR ...]")
    run(sys.argv[1:])
