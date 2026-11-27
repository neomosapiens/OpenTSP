"""Shared by the rounded PyTorch reference scripts (`check_block.py`, `check_step.py`): loading the real checkpoint, the comparison metrics."""
import importlib.util
import os

import numpy as np

from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories.config import TINYSTORIES

DEFAULT_CHECKPOINT = "calum/tinystories-gpt2-3M"
COSINE_MIN = 0.99999            # , tier rounded PyTorch reference


def verdict(ok, no_torch):
    """The prefix of a result line: FAIL, PASS, or, when the PyTorch comparison did not run, SKIP-PARTIAL (a PASS would
    claim a rounded PyTorch reference agreement that was never measured)."""
    if not ok:
        return "FAIL"
    return "SKIP-PARTIAL (bit-exact reference only passed):" if no_torch else "PASS"


def torch_missing():
    """The SKIP text of the PyTorch comparison, or None when PyTorch is importable."""
    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed (pip install torch into the toolchain venv; transformers is not needed)"
    return None


def load_weights(name=DEFAULT_CHECKPOINT, online=False):
    """(weights, source text) of the real checkpoint, or (None, the exact reason). The cached `model.safetensors` is
    read with the `safetensors` package, without PyTorch; with PyTorch and `transformers` the checkpoint is also loaded
    the usual way (`online`: may download)."""
    errors = []
    try:
        return W.load_safetensors(TINYSTORIES, name), f"cached model.safetensors of {name}"
    except Exception as e:                                           # no safetensors package, no cached snapshot
        errors.append(f"safetensors: {type(e).__name__}: {str(e)[:120]}")
    if not online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return W.load_checkpoint(TINYSTORIES, name), f"transformers checkpoint {name}"
    except Exception as e:
        errors.append(f"transformers: {type(e).__name__}: {str(e)[:120]}")
    return None, "; ".join(errors)


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def agreement(exact_ref, other):
    """(max |exact - other| / max |other|, cosine) of two vectors."""
    err = float(np.max(np.abs(np.asarray(exact_ref, np.float64) - other)) / (np.max(np.abs(other)) + 1e-12))
    return err, cosine(exact_ref, other)
