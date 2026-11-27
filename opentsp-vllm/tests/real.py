"""Build serving test buckets on first use. Use `$OPENTSP_VLLM_CACHE` when set and an isolated directory otherwise.
The helper uses the same program builder, cache and oracle as the TinyStories workload."""

import atexit
import os
import shutil
import tempfile

import reference_loader

reference_loader.load()  # the showcase directory, the reference tools and the generator library on sys.path

from opentsp_vllm.models.tinystories import build_programs as BP
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools import patch
from opentsp_tools import profile as tprofile

__all__ = ["BP", "G", "PC", "kvpool", "patch", "tprofile"]  # what the tests take from here

_cache = None
_models = {}
_arts = {}


def cache_dir():
    global _cache
    if os.environ.get("OPENTSP_VLLM_CACHE"):
        return os.environ["OPENTSP_VLLM_CACHE"]
    if _cache is None:
        _cache = tempfile.mkdtemp(prefix="opentsp-serving-test-")
        atexit.register(shutil.rmtree, _cache, ignore_errors=True)
    return _cache


def model(layers=2):
    if layers not in _models:
        _models[layers] = PC.Model(layers)
    return _models[layers]


def bucket_path(bucket, layers=2):
    BP.build_bucket(model(layers), bucket, cache_dir())
    return PC.bucket_dir(cache_dir(), model(layers), bucket)


def art(bucket, layers=2):
    """`build_programs.Artifacts` of the cached bucket (the Python reference of the patcher)."""
    if (layers, bucket) not in _arts:
        _arts[(layers, bucket)] = BP.Artifacts(bucket_path(bucket, layers), "ref")
    return _arts[(layers, bucket)]


def geometry(layers=2):
    return kvpool.Geometry(tprofile.load("ref"), layers)


def random_values(table, rng):
    """{site: a legal value} drawn uniformly from each site's range, on its grid."""
    out = {}
    for i, n in enumerate(table.names):
        lo, hi, st, sp = int(table.vmin[i]), int(table.vmax[i]), int(table.stride[i]), int(table.span[i])
        while True:
            v = rng.randrange(lo, hi + 1)
            if (v - lo) % st < sp:
                break
        out[str(n)] = v
    return out


def pool(cfg, seed, nblk):
    """A bit-exact reference pool with random K and V rows in blocks 1..nblk-1."""
    import numpy as np

    p = G.BlockPool(cfg, G.POOL_BLOCKS)
    rng = np.random.default_rng(seed)
    p.k[1:nblk] = (rng.standard_normal(p.k[1:nblk].shape) * 0.5).astype(np.float16)
    p.v[1:nblk] = rng.standard_normal(p.v[1:nblk].shape).astype(np.float32)
    return p


VOCAB = 50257


def plan_of(bucket, lanes, n_log, vocab=VOCAB):
    """A `StepPlan` of lanes `(token, position, block table)`, the logit lanes first, padded to the bucket's T."""
    from opentsp_vllm import step_plan as SP

    real_lanes = [SP.Lane(f"r{i}", int(t), int(p), tuple(tab[: p // 16 + 1]), i < n_log) for i, (t, p, tab) in enumerate(lanes)]
    plan = SP.StepPlan(
        bucket, tuple(real_lanes) + (SP.PAD,) * (bucket[0] - len(real_lanes)), len(real_lanes), n_log, tuple(f"r{i}" for i in range(n_log))
    )
    plan.patches = SP.patch_values(plan, vocab)
    return plan


def random_lanes(bucket, rng, nblk=40):
    """Random lanes `(token, position, block table)` for a bucket and the number of logit rows: tables of distinct blocks 1..nblk-1,
    a lane's write block (the last of its table) its own."""
    T, Ls, C = bucket
    n = rng.randint(1, T)
    ids = list(range(1, nblk))
    rng.shuffle(ids)
    lanes = []
    for _ in range(n):
        pos = rng.randrange(C)
        lanes.append((rng.randrange(VOCAB), pos, [ids.pop() for _ in range(pos // 16 + 1)]))
    return lanes, rng.randint(1, min(Ls, n))
