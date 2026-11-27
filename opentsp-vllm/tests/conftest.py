"""Setup of the binding tests (`just vllm-binding-test`) and the other tests of this directory.

- `$OPENTSP_VLLM_CACHE` is a scratch directory unless the caller set it, so that nothing the tests build or write (program caches,
  failure reproducers) lands in `~/.cache/opentsp/vllm`.
- The listing of `~/.cache/opentsp/vllm/failures` is the same after the session as before it: a test that writes a reproducer
  there fails the session at teardown.
"""

import os
import shutil
import tempfile

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: the 8-layer schedule of test_restart_schedules.py (`just vllm-schedules-full`)")


def _listing(path):
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return None


@pytest.fixture(scope="session", autouse=True)
def scratch_cache_and_untouched_user_failures():
    users = os.path.expanduser("~/.cache/opentsp/vllm/failures")
    before = _listing(users)
    made = None
    if not os.environ.get("OPENTSP_VLLM_CACHE"):
        made = tempfile.mkdtemp(prefix="opentsp-vllm-tests-cache-")
        os.environ["OPENTSP_VLLM_CACHE"] = made
    try:
        yield
    finally:
        if made:
            os.environ.pop("OPENTSP_VLLM_CACHE", None)
            shutil.rmtree(made, ignore_errors=True)
    after = _listing(users)
    assert after == before, f"the tests changed {users}: {before} -> {after}"
