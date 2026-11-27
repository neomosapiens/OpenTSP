"""Unit-test setup: the TSP platform is the current platform of this process (the plugin is inert otherwise), and the test
helpers of the parent directory (`oracle`, `reference_chip`) are importable."""

import os
import pathlib
import sys

os.environ["OPENTSP_VLLM"] = "1"
for p in (pathlib.Path(__file__).resolve().parents[1], pathlib.Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import pytest


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    import vllm_helpers

    return vllm_helpers.write_model_dir(tmp_path_factory.mktemp("model"))
