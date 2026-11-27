"""Engine helpers of the serving tests : a real vLLM engine (`LLM`, engine core in this process) on the TSP
platform over `ReferenceChip` or `IssChip`, the bit-exact reference reference of a request, and `solo`, which runs a request alone and returns its token ids
and the logits the chip produced for them."""

import contextlib
import os
import pathlib
import sys

# the platform plugin and the in-process engine core must be decided before vLLM is imported
os.environ["OPENTSP_VLLM"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
for p in (pathlib.Path(__file__).resolve().parents[1], pathlib.Path(__file__).resolve().parents[1] / "unit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

import reference_loader
import vllm_helpers
from opentsp_vllm import programs  # noqa: E402

VOCAB = 50257
MODELS = {"test": dict(layers=2, context=64), "full": dict(layers=8, context=128)}
Reference_FACTORY = "reference_chip:make_serving_chip"


def model_dir(root, model):
    return vllm_helpers.write_model_dir(pathlib.Path(root) / f"model-{model}", layers=MODELS[model]["layers"], vocab=VOCAB)


def iss_artifacts(model):
    """Prepare the unified package for this model and select its exact entry."""
    import real

    layers = MODELS[model]["layers"]
    for bucket in programs.grid((64,) if MODELS[model]["context"] == 64 else (64, 128)):
        real.bucket_path(bucket, layers)
    return real.PC.entry_dir(real.cache_dir(), real.model(layers))


def engine_kwargs(path, chip, model, root=None, **overrides):
    """The `LLM` arguments of the serving engine of `model` on `chip`: 4 sequences, 8 batched tokens, token ids in and out."""
    m = MODELS[model]
    options = dict(weights="synthetic:1", max_context=m["context"], chip=Reference_FACTORY if chip == "exact" else "iss")
    if chip == "iss":
        options.pop("weights")
        options["programs"] = iss_artifacts(model)
    options.update(overrides.pop("opentsp", {}))
    kwargs = dict(
        model=str(path),
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=m["context"],
        max_num_seqs=4,
        max_num_batched_tokens=8,
        enable_chunked_prefill=True,
        additional_config={"opentsp": options},
    )
    kwargs.update(overrides)
    return kwargs


@contextlib.contextmanager
def shadow_env(shadow):
    """`OPENTSP_VLLM_SHADOW` set to `shadow` (None: unset) for the duration of an engine start, which reads it."""
    old = os.environ.pop("OPENTSP_VLLM_SHADOW", None)
    if shadow:
        os.environ["OPENTSP_VLLM_SHADOW"] = shadow
    try:
        yield
    finally:
        os.environ.pop("OPENTSP_VLLM_SHADOW", None)
        if old is not None:
            os.environ["OPENTSP_VLLM_SHADOW"] = old


class Engine:
    """An `LLM` with the chip's steps recorded: `steps` is a list of (plan, ChipResult) since the last `clear`."""

    def __init__(self, llm, model):
        self.llm = llm
        self.model = model
        self.steps = []
        core = llm.llm_engine.engine_core.engine_core
        self.worker = core.model_executor.driver_worker.worker
        chip = self.worker.chip
        step = chip.step

        def recording(plan):
            result = step(plan)
            self.steps.append((plan, result))
            return result

        chip.step = recording

    @property
    def chip(self):
        return self.worker.chip

    def clear(self):
        self.steps.clear()

    def generate(self, prompts, params):
        batch = [{"prompt_token_ids": list(p)} for p in prompts]
        return self.llm.generate(batch, params, use_tqdm=False)

    def solo(self, prompt, sampling_params):
        """The request alone on an empty prefix cache: (generated token ids, logits (n, vocab) of the steps that produced them)."""
        self.llm.reset_prefix_cache()
        self.clear()
        out = self.generate([prompt], sampling_params)[0].outputs[0]
        rows = [r for _, result in self.steps for r in np.asarray(result.logits)]
        return list(out.token_ids), np.stack(rows) if rows else np.zeros((0, VOCAB), np.float32)

    def close(self):
        self.llm.llm_engine.engine_core.shutdown()


def reference_tokens(prompt, n, model="test"):
    """`paged_reference.generate_paged` of the request alone on the synthetic weights of seed 1: the ids after the prompt."""
    reference_paged, weights, _ = reference_loader.load()
    from opentsp_vllm.models.tinystories import layout

    w = _weights(model, layout, weights)
    C = 64 if len(prompt) + n <= 64 else 128
    return list(reference_paged.generate_paged(w, prompt, n, C=C).tokens[len(prompt):])


_w = {}


def _weights(model, layout, weights):
    if model not in _w:
        _w[model] = weights.synthetic(layout.serve_config(MODELS[model]["layers"]), seed=1)
    return _w[model]


def serve_env():
    """The environment of a `vllm serve` process: this package, the test chips and the platform plugin switch."""
    here = str(pathlib.Path(__file__).resolve().parents[2])
    paths = [p for p in sys.path if p.startswith(here)]
    return dict(os.environ, OPENTSP_VLLM="1", PYTHONPATH=os.pathsep.join(paths))


def serve_command(path, chip, model, port, root=None):
    """The `vllm serve` command line of the engine `engine_kwargs` describes (token ids in, so no tokenizer)."""
    import json

    kw = engine_kwargs(path, chip, model, root=root)
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", kw["model"], "--skip-tokenizer-init", "--dtype", "float32", "--max-model-len", str(kw["max_model_len"]),
        "--max-num-seqs", "4", "--max-num-batched-tokens", "8", "--port", str(port),
        "--additional-config", json.dumps(kw["additional_config"]),
    ]  # fmt: skip
