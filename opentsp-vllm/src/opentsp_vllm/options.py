"""The `additional_config["opentsp"]` options (last row of the extension-point table).

- `weights`: the ISS defaults to the selected package's shared preload. An explicit weight file or exported weights directory overrides it. The bit-exact reference test chip accepts `synthetic:SEED`.
- `ras_domain`: stable management identity of an independent TinyStories or SHIP execution domain.
- `programs`: a published serving package or a cache root.
- `kv_persist`: `warm` (default) or `replay`; `kv_mode`: `paged` (default) or `staged` (both are the binding's);
- `replay`: unset (default: `$OPENTSP_VLLM_REPLAY`, else off), false or true, the opt-in parametrised schedule replay of the `iss` chip's steps (needs a binding built with the `replay` feature, which `just vllm-binding` does): a step is served from the schedule recorded for its bucket program, with its site values; the logits and cycle counts are the normal driver's, and a bucket that measures under 1.2x is served by the normal driver;
- `profile`: the profile the `iss` chip and its programs are for (default `ref`). The binding's host side (five vocabulary rows of 64 lanes per embedding word, 63 head installs of 160 rows, 320-byte output words, contexts 64 and 128) is written for the geometry of `ref`: a profile of that geometry (other timing: `test/ref_slow`, `test/ref_hw`) is accepted, any other is refused naming the first of its numbers that differs. The programs of other geometries (`test/sl8`, `test/sl32`) are built and run by the generators' own tests (`tests/showcase/tinystories/test_serve_geometry.py`), not through the vLLM binding;
- `max_context`: the largest context bucket of the programs, 64 (the test model) or 128 (default);
- `chip`: `tiny` for TinyStories, `iss` (default), `ship` for four-chip GPT-2 small, or `module:callable`.
  The GPT-2 route needs the sealed `programs` directory and a GPT-2 small `weights` checkpoint. It has two
  KV chunks and keeps weights and cache resident. `check` enables exact comparison against the numerical twin.
  Tests can pass `layers` to instantiate a small one-chip-per-layer fixture with a host vocabulary head.
"""

import json
import os

# the geometry the binding's host side is written for (`ref`): superlanes, MXM rows, MEM slices per hemisphere, words per slice
BINDING_GEOMETRY = (("superlanes", 20), ("MXM rows", 320), ("MEM slices per hemisphere", 44), ("words per slice", 8192))
from opentsp_tools.paths import SPEC
from opentsp_vllm import ship_plan
SPEC_CONFIG = os.environ.get("OPENTSP_SPEC_CONFIG") or os.path.join(SPEC, "config")


def profile_geometry(name):
    """(superlanes, MXM rows, MEM slices per hemisphere, words per slice) of the profile `name` from the specification's configuration
    files, or None when they are not at hand (an installed package without the repository: the program set's own profile check decides)."""
    path = os.path.join(SPEC_CONFIG, name + ".json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        c = json.load(f)
    return c["geometry"]["superlanes"], c["mxm"]["rows"], c["geometry"]["mem_slices_per_hemisphere"], c["sram"]["words_per_slice"]


def check_profile(name):
    """ValueError naming the first number of the profile's geometry that is not the binding's."""
    have = profile_geometry(name)
    if have is None:
        return
    for (what, want), got in zip(BINDING_GEOMETRY, have):
        if got != want:
            shown = ", ".join(f"{w} {n}" for n, w in BINDING_GEOMETRY)
            raise ValueError(f"additional_config opentsp.profile {name!r}: the serving binding is written for the geometry of ref ({shown}); "
                             f"the profile has {got} {what}")


# the limits of the SHIP-style pipeline (the platform enforces them inside vLLM, the launcher before it starts): the pool of 64 pages of 16 slots (block 0 is vLLM's null
# block, so 65 blocks) and the model's 1,024 positions; each scheduled token is one run of the pipeline (about a second of simulation), so a step holds at most 8
SHIP_MAX_MODEL_LEN, SHIP_POOL_BLOCKS = ship_plan.kv_limits(ship_plan.FULL_KV_CHUNKS)
SHIP_MAX_NUM_SEQS = 8
SHIP_MAX_NUM_BATCHED_TOKENS = 8

DEFAULTS = {
    "weights": None, "programs": None, "kv_persist": "warm", "kv_mode": "paged", "max_context": 128, "replay": None, "chip": "iss", "profile": "ref",
    "model": None, "chips": None, "check": False, "layers": None, "resident": True, "ras_domain": None, "kv_chunks": ship_plan.FULL_KV_CHUNKS,
}  # fmt: skip
CHOICES = {"kv_persist": ("warm", "replay"), "kv_mode": ("paged", "staged"), "max_context": (64, 128)}
STRINGS = ("weights", "programs", "chip", "model", "ras_domain", "profile")
SHIP_MODELS = {"gpt2": dict(layers=12, chips=4)}


def parse(additional_config):
    """The validated options with their defaults; ValueError naming the option otherwise."""
    raw = (additional_config or {}).get("opentsp", {})
    if not isinstance(raw, dict):
        raise ValueError(f"additional_config opentsp must be a mapping of options, not {raw!r}")
    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"additional_config opentsp: unknown option {unknown[0]!r}; known: {sorted(DEFAULTS)}")
    out = dict(DEFAULTS)
    out.update(raw)
    for key, choices in CHOICES.items():
        if out[key] not in choices:
            raise ValueError(f"additional_config opentsp.{key} must be one of {choices}, not {out[key]!r}")
    for key in STRINGS:
        if out[key] is not None and not isinstance(out[key], str):
            raise ValueError(f"additional_config opentsp.{key} must be a string, not {out[key]!r}")
    for key in ("chips", "layers"):
        if out[key] is not None and (not isinstance(out[key], int) or isinstance(out[key], bool)):
            raise ValueError(f"additional_config opentsp.{key} must be an integer, not {out[key]!r}")
    if out["replay"] is not None and not isinstance(out["replay"], bool):
        raise ValueError(f"additional_config opentsp.replay must be true or false, not {out['replay']!r}")
    if out["replay"] and out["chip"] not in ("iss", None):
        raise ValueError(f"additional_config opentsp.replay is an option of chip 'iss' only (chip is {out['chip']!r})")
    for key in ("check", "resident"):
        if not isinstance(out[key], bool):
            raise ValueError(f"additional_config opentsp.{key} must be true or false, not {out[key]!r}")
    if out["profile"] != DEFAULTS["profile"] and out["chip"] not in ("iss", None):
        raise ValueError(f"additional_config opentsp.profile is an option of chip 'iss' only (chip is {out['chip']!r})")
    check_profile(out["profile"])
    if out["chip"] == "ship" and out["layers"] is None and "kv_chunks" not in raw:
        out["kv_chunks"] = 2
    ship_plan.kv_limits(out["kv_chunks"])
    if out["chip"] == "ship":
        out.update(ship_shape(out))
    else:
        used = sorted(k for k in ("model", "chips", "layers", "check", "resident", "kv_chunks") if out[k] != DEFAULTS[k])
        if used:
            raise ValueError(f"additional_config opentsp.{used[0]} is an option of chip 'ship' only (chip is {out['chip']!r})")
    return out


def ship_shape(out):
    """The validated shape of the ship chip: `{"model", "chips", "layers"}`."""
    model = out["model"] or "gpt2"
    if model not in SHIP_MODELS:
        raise ValueError(f"additional_config opentsp.model must be one of {tuple(SHIP_MODELS)}, not {model!r}")
    full = SHIP_MODELS[model]
    layers = out["layers"]
    if layers is not None:
        if not 1 <= layers <= 8:
            raise ValueError(f"additional_config opentsp.layers must be 1 to 8 (a truncated model for tests, the head on the host), not {layers}")
        if out["chips"] not in (None, layers):
            raise ValueError(f"additional_config opentsp.chips {out['chips']} does not match opentsp.layers {layers}: a truncated model runs on one chip per layer")
        return dict(model=model, chips=layers, layers=layers)
    chips = full["chips"] if out["chips"] is None else out["chips"]
    if chips != full["chips"]:
        raise ValueError(f"additional_config opentsp.chips {chips}: {model} requires four chips")
    if out["kv_chunks"] != 2 or not out["resident"] or not out["programs"]:
        raise ValueError("four-chip GPT-2 requires resident mode, two KV chunks and a sealed program bundle")
    return dict(model=model, chips=chips, layers=full["layers"])
