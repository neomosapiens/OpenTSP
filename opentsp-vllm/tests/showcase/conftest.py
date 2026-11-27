"""Fixtures of the showcase tests that need vLLM: a real engine with the GPT-2 tokenizer on the test model (2 layers, context 64) over
`ReferenceChip` or `TinyChip`, wrapped as the showcase's `VllmBackend`. One engine per chip and module."""

import pathlib
import sys

TESTS = pathlib.Path(__file__).resolve().parents[1]
for p in (TESTS / "serving", TESTS / "unit", TESTS, TESTS.parent / "showcase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import engine
import pytest


@pytest.fixture(scope="module")
def make_backend(tmp_path_factory):
    from opentsp_vllm.showcase.backend import VllmBackend
    from batching_helpers import tokenizer_kwargs
    from vllm import LLM

    backends = {}
    root = tmp_path_factory.mktemp("showcase")

    def make(chip="exact"):
        if chip not in backends:
            path = engine.model_dir(root, "test")
            kwargs = engine.engine_kwargs(path, chip, "test", root=root, enable_prefix_caching=True, disable_log_stats=True, **tokenizer_kwargs())
            if chip == "tiny":
                import reference_loader
                reference_loader.load()
                from opentsp_vllm.models.tinystories import fixed_package
                package = fixed_package.build(root / "programs", layers=2, jobs=2)
                kwargs["additional_config"]["opentsp"] = dict(chip="tiny", programs=str(package), max_context=64)
            backends[chip] = VllmBackend(LLM(**kwargs))
        backends[chip].reset_prefix_cache()
        return backends[chip]

    yield make
    for b in backends.values():
        b.close()
