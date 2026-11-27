"""Fixtures of the serving tests : `make_llm(chip, model, **engine_kwargs)` builds a real vLLM engine on the TSP
platform once per module and key and returns it with its prefix cache reset; the engines are shut down at the end of the module."""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import engine
import pytest


@pytest.fixture(scope="module")
def make_llm(tmp_path_factory):
    from vllm import LLM

    engines = {}
    root = tmp_path_factory.mktemp("serving")

    def make(chip="exact", model="test", shadow=None, **kwargs):
        """`shadow="exact"` sets OPENTSP_VLLM_SHADOW while the engine starts: every step is also run on the bit-exact reference twin and compared."""
        key = (chip, model, shadow, repr(sorted(kwargs.items(), key=lambda kv: kv[0])))
        if key not in engines:
            path = engine.model_dir(root, model)
            with engine.shadow_env(shadow):
                engines[key] = engine.Engine(LLM(**engine.engine_kwargs(path, chip, model, root=root, **dict(kwargs))), model)
        e = engines[key]
        e.llm.reset_prefix_cache()
        e.clear()
        return e

    yield make
    for e in engines.values():
        e.close()
