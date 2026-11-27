"""pytest configuration of the TinyStories showcase tests: the `slow` marker selects the tests that
`just showcase-exact` leaves out and `just showcase-exact-full` runs (the full eight-layer text-sizing grid), and a session
program cache: unless the caller set `OPENTSP_VLLM_CACHE`, the tests build the bucket programs they need into a
temporary directory removed at the end, so a test run never writes to (or depends on) the user's `~/.cache/opentsp/vllm`. Set
`OPENTSP_VLLM_CACHE` to keep the buckets between runs (the invariance tests build seven 2-layer buckets, about 4 minutes)."""
import os
from pathlib import Path
import shutil
import tempfile

import pytest

_session_cache = None


def pytest_configure(config):
    global _session_cache
    config.addinivalue_line("markers", "slow: about 3 minutes or more; run by `just showcase-exact-full`, deselected by `just showcase-exact`")
    config.addinivalue_line("markers", "paged_iss: Paged scenarios on the ISS (2-layer cached programs); `just showcase-paged`, also `slow`")
    config.addinivalue_line("markers", "paged_full: Paged full-model scenario (builds 8-layer buckets: about 15 minutes); only with "
                                       "`-m paged_full` (`just showcase-paged-full`)")
    if not os.environ.get("OPENTSP_VLLM_CACHE"):
        _session_cache = tempfile.mkdtemp(prefix="opentsp-vllm-cache-")
        os.environ["OPENTSP_VLLM_CACHE"] = _session_cache


def pytest_collection_modifyitems(config, items):
    """`paged_full` tests run only when asked for by name: a plain run, `-m "not slow"` and `showcase-exact-full` leave them out."""
    if "paged_full" in (config.getoption("markexpr") or ""):
        return
    skipped = [i for i in items if "paged_full" in i.keywords]
    if skipped:
        config.hook.pytest_deselected(items=skipped)
        items[:] = [i for i in items if "paged_full" not in i.keywords]


def pytest_unconfigure(config):
    global _session_cache
    if _session_cache:
        shutil.rmtree(_session_cache, ignore_errors=True)
        os.environ.pop("OPENTSP_VLLM_CACHE", None)
        _session_cache = None



@pytest.fixture
def serving_program_cases(monkeypatch):
    """Import the reduced program cases shared with the assembler and binding tests."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "tests" / "programs" / "gen"))
    import model_serve

    return model_serve
