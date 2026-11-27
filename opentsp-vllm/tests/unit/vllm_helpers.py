"""Helpers of the unit tests that need vLLM: a model directory without weights and the engine configuration of the test model."""

import json
import os
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
TESTS_DIRS = os.pathsep.join([str(HERE.parent), str(HERE)])


def write_model_dir(path, layers=2, vocab=2000, positions=128):
    """The `config.json` of the GPT-2 shape of the oracle's test model; no weights, no tokenizer ."""
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "n_embd": 64,
                "n_head": 16,
                "n_layer": layers,
                "n_positions": positions,
                "n_ctx": positions,
                "vocab_size": vocab,
                "activation_function": "gelu_new",
                "layer_norm_epsilon": 1e-5,
                "bos_token_id": 0,
                "eos_token_id": 0,
            }
        )
    )
    return path


def engine_config(model_dir, **overrides):
    """The VllmConfig of the test model: 4 sequences, 8 batched tokens, context 64."""
    from vllm.engine.arg_utils import EngineArgs

    args = dict(
        model=str(model_dir),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=64,
        max_num_seqs=4,
        max_num_batched_tokens=8,
        enable_chunked_prefill=True,
    )
    args.update(overrides)
    return EngineArgs(**args).create_engine_config()


def env_without_platform():
    return {k: v for k, v in os.environ.items() if k != "OPENTSP_VLLM"}
