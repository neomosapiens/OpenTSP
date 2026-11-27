"""Digest-keyed cache for TinyStories bucket programs.

    <cache>/<digest>/manifest.json          what the entry is: model, weights, profile, generator, layout, package digests
    <cache>/<digest>/<bucket>/              one bucket, `P<T>_<L>_<C>`:
        program.elf      the object: the initial image of every queue (`ots-asm asm -o`), the placeholders in the patch sites
        bindings.npz      instruction operand sites and output decoding arrays
        program.json      queue block map, SRAM output regions, geometry, build identity and payload hashes
    <cache>/cycles.json                     per digest: bucket -> cycles, build seconds, ISS wall seconds, text words, sites, bytes

The digest covers the generator sources, the profile (`config/<profile>.json`), the layout manifest (which holds the digest of the
weights) and the layer count; the bucket's own digest adds its name. A bucket directory is written under a temporary name and
renamed when complete, so a bucket that exists is complete and parallel builds of different buckets do not meet."""
import contextlib
import ctypes
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

from pathlib import Path

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths
from opentsp_vllm import programs as RT        # the reader's check of a bucket's content: `built` applies the same one
from opentsp_vllm.models.tinystories import export
from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import weights as W
from opentsp_tools import patch

OUTPUT_KEYS = RT.OUTPUT_KEYS

from opentsp_vllm import serving_artifacts as artifacts

SCHEMA = artifacts.SCHEMA
VERSION = artifacts.VERSION
HERE = _paths.HERE


# the sources whose output the cache holds: the emitter of the step and everything it imports, the assembler it targets


def cache_root(path=None):
    return os.path.abspath(path or os.environ.get("OPENTSP_VLLM_CACHE") or os.path.expanduser("~/.cache/opentsp/vllm"))


def generator_files(root=None):
    """All maintained builder/reference sources and architectural interface inputs."""
    root = Path(root or _paths.ROOT)
    directories = ("opentsp-sw-toolchain/python", "opentsp-vllm/src/opentsp_vllm/models/tinystories", "opentsp-arch-spec/reference")
    files = [p for d in directories for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts]
    files += [root / "opentsp-vllm/src/opentsp_vllm/serving_artifacts.py",
              root / "opentsp-arch-spec/config/ref.json", root / "opentsp-arch-spec/tools/tsp_as.py",
              root / "opentsp-arch-spec/isa-db/generated/encoding.json"]
    return sorted(str(p) for p in files)


def generator_digest(files=None):
    """SHA-256 over the relative path and the bytes of every generator source."""
    h = hashlib.sha256()
    for f in files or generator_files():
        h.update(os.path.relpath(f, _paths.ROOT).encode() + b"\0")
        with open(f, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()


def profile_digest(profile="ref"):
    with open(os.path.join(_paths.SPEC, "config", f"{profile}.json"), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Model:
    """What a cache entry is for: the layer count, the weights and the planner's layout (`export.manifest`, serve mode)."""

    def __init__(self, layers=8, seed=1, checkpoint=None, profile="ref", generator=None):
        if layers != 8 and checkpoint:
            raise ValueError("the 2-layer test model has synthetic weights only")
        self.layers, self.profile = layers, profile
        self.cfg = L.serve_config(layers)
        self.source = f"checkpoint:{checkpoint}" if checkpoint else f"synthetic:{seed}"
        self.w = W.load_checkpoint(self.cfg, checkpoint) if checkpoint else W.synthetic(self.cfg, seed)
        prof = L.load_profile(profile)
        plan = L.plan_serve(self.cfg, prof)
        tensors, objects = L.build_serve_images(self.cfg, plan, self.w)
        self.package = export.package_bytes(prof, plan, tensors)
        self.logical = W.encode_deployment(self.w)
        manifest = export.manifest(self.cfg, prof, plan, tensors, objects, self.w.digest(), sha(self.package), "serve")
        self.layout_sha256 = sha(json.dumps(manifest, sort_keys=True).encode())
        self.generator_sha256 = generator or generator_digest()
        self.profile_sha256 = profile_digest(profile)
        self.digest = sha(json.dumps({"schema": SCHEMA, "version": VERSION, "generator": self.generator_sha256,
                                      "profile": self.profile_sha256, "layout": self.layout_sha256, "layers": layers},
                                     sort_keys=True).encode())

    @property
    def key(self):
        return self.digest[:16]

    def manifest(self):
        return artifacts.manifest(
            "tinystories", 1, shared_preload="weights.safetensors", route=[0], spec_version="0.1", digest=self.digest, layers=self.layers,
            weights=self.source, weights_sha256=self.w.digest(), package_sha256=sha(self.logical),
            device_sha256=sha(self.package),
            profile=self.profile, profile_sha256=self.profile_sha256, generator_sha256=self.generator_sha256,
            layout_sha256=self.layout_sha256,
        )



def materialize(data, layers, profile="ref", *, expected=None, package_identity=""):
    """Resolve the verified TinyStories logical store into its serve layout."""
    from opentsp_vllm import device_layout_cache

    cfg = L.serve_config(layers)
    model = W.decode_deployment(cfg, data)

    def build():
        prof = L.load_profile(profile)
        plan = L.plan_serve(cfg, prof)
        tensors, _ = L.build_serve_images(cfg, plan, model)
        return export.package_bytes(prof, plan, tensors)

    def verified(device):
        if expected is not None and sha(device) != expected:
            raise ValueError("TinyStories device layout differs from the manifest")
        return device

    if expected is None:
        return model, build()
    identity = json.dumps(["tinystories", package_identity, profile_digest(profile), layers, sha(data), expected]).encode()
    return model, device_layout_cache.resolve(identity, build, verified)


def bucket_name(bucket):
    return "P%d_%d_%d" % tuple(bucket)


def parse_bucket(name):
    t, l, c = name[1:].split("_")
    return int(t), int(l), int(c)


def entry_dir(cache, model):
    return os.path.join(cache_root(cache), model.key)


def bucket_dir(cache, model, bucket):
    return os.path.join(entry_dir(cache, model), bucket_name(bucket))


def bucket_digest(model, bucket):
    return sha((model.digest + bucket_name(bucket)).encode())


FILES = (*artifacts.ELF_PAYLOAD, "program.json")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def write_atomic(path, text):
    d = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=d, delete=False) as f:
        f.write(text)
    os.replace(f.name, path)


def built(cache, model, bucket):
    """The bucket's record when it is in the cache complete, readable and for this digest, else None. The content check is the
    reader's (`opentsp_vllm.programs.content_problem`): a damaged bucket is not built, so the builder rebuilds it."""
    d = bucket_dir(cache, model, bucket)
    if not all(os.path.isfile(os.path.join(d, f)) for f in FILES):
        return None
    try:
        rec = read_json(os.path.join(d, "program.json"))
    except ValueError:
        return None
    man = model.manifest()
    if man.get("shared_preload"):
        shared = Path(d).parent / "weights.safetensors"
        if not shared.is_file() or artifacts.file_digest(shared) != man["package_sha256"]:
            return None
    if rec.get("digest") != bucket_digest(model, bucket) or RT.content_problem(d, rec):
        return None
    return rec


def new_bucket_dir(cache, model, bucket):
    """A fresh temporary directory beside the final one (`commit` renames it)."""
    os.makedirs(entry_dir(cache, model), exist_ok=True)
    return tempfile.mkdtemp(prefix=bucket_name(bucket) + ".tmp-", dir=entry_dir(cache, model))


@contextlib.contextmanager
def bucket_lock(final):
    """An exclusive lock of one bucket (`flock` on `<bucket dir>.lock`, released by the system when the process dies)."""
    with open(final + ".lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def swap_directories(a, b):
    """Exchange two directories in one atomic step (`renamex_np` RENAME_SWAP on macOS, `renameat2` RENAME_EXCHANGE on Linux): a
    reader of `b` sees the old directory or the new one, never a missing path. NotImplementedError where the system has neither."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        status = libc.renamex_np(os.fsencode(a), os.fsencode(b), 2)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        status = libc.renameat2(-100, os.fsencode(a), -100, os.fsencode(b), 2)
    else:
        raise NotImplementedError(sys.platform)
    if status != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), b)


def write_shared_preload(path, package, digest):
    """Concurrent bucket builders share one immutable weight image."""
    with bucket_lock(str(path)):
        if path.is_file() and artifacts.file_digest(path) == digest:
            return
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            f.write(package)
        os.replace(f.name, path)


def commit(cache, model, bucket, tmp, toolchain=None, replace=False):
    """Publish `tmp` as the bucket directory; returns "published", or "kept" when a complete directory of this digest was already
    there (another builder was first: `tmp` is dropped). `tmp` is sealed first (`files_sha256`). A published directory is never
    deleted in place and the live path is never missing while one is replaced: the new directory is swapped in by one atomic
    exchange (`swap_directories`); where the system has none, the old one is moved aside and, if the rename of the new one fails,
    moved back, so the error leaves the old directory in place. With `replace` (a forced rebuild) a complete directory of this
    digest is replaced too; a directory that is there but not complete or damaged is replaced the same way."""
    final = bucket_dir(cache, model, bucket)
    man = model.manifest()
    if toolchain:
        man["toolchain"] = toolchain
    if man.get("shared_preload"):
        write_shared_preload(Path(final).parent / "weights.safetensors", model.logical, man["package_sha256"])
    write_atomic(os.path.join(entry_dir(cache, model), "manifest.json"), json.dumps(man, indent=1, sort_keys=True) + "\n")
    artifacts.seal(tmp, read_json(os.path.join(tmp, "program.json")), payload=artifacts.ELF_PAYLOAD)
    with bucket_lock(final):
        if not replace and built(cache, model, bucket):
            shutil.rmtree(tmp, ignore_errors=True)
            return "kept"
        if not os.path.exists(final):
            os.rename(tmp, final)
            return "published"
        try:
            swap_directories(tmp, final)
        except NotImplementedError:
            aside = tempfile.mkdtemp(prefix=bucket_name(bucket) + ".old-", dir=entry_dir(cache, model))
            moved = os.path.join(aside, "dir")
            os.rename(final, moved)
            try:
                os.rename(tmp, final)
            except BaseException:
                os.rename(moved, final)
                raise
            finally:
                shutil.rmtree(aside, ignore_errors=True)
        else:
            shutil.rmtree(tmp, ignore_errors=True)      # the exchange left the old directory under the temporary name
    return "published"


def cycles_table(cache):
    """{digest: {"manifest": ..., "buckets": {name: record}}} of every complete bucket under the cache."""
    root = cache_root(cache)
    out = {}
    for m in sorted(glob.glob(os.path.join(root, "*", "manifest.json"))):
        man = read_json(m)
        if man.get("model") != "tinystories":
            continue
        buckets = {}
        for b in sorted(glob.glob(os.path.join(os.path.dirname(m), "P*", "program.json"))):
            if ".tmp-" in b or ".old-" in b:                      # a build in progress or a replaced directory
                continue
            rec = read_json(b)
            if rec.get("digest") == bucket_digest_of(man, rec["bucket"]):
                buckets[bucket_name(rec["bucket"])] = {k: rec[k] for k in ("cycles", "build_seconds", "iss_seconds", "text_words",
                                                                           "patch_sites", "image_bytes")}
        out[man["digest"]] = {"layers": man["layers"], "weights": man["weights"], "buckets": buckets}
    return out


def bucket_digest_of(manifest, bucket):
    return sha((manifest["digest"] + bucket_name(bucket)).encode())


def write_cycles(cache):
    """`<cache>/cycles.json` from the buckets on disk (a whole scan, so the last writer of parallel builds is complete)."""
    table = cycles_table(cache)
    write_atomic(os.path.join(cache_root(cache), "cycles.json"), json.dumps(table, indent=1, sort_keys=True) + "\n")
    return table


# -- the artifacts of a bucket --------------------------------------------------------------------------------------------------
def patch_arrays(paged):
    """The columns of `bindings.npz` for the located sites of a paged program (`opentsp_tools.patch.Paged`)."""
    t = paged.table
    where = [paged.where(i) for i in range(len(t))]
    pick = lambda k, j: np.array([w[min(k, len(w) - 1)][j] for w in where], dtype=np.int64)
    block = lambda k: np.array([-1 if w[min(k, len(w) - 1)][0] is None else w[min(k, len(w) - 1)][0] for w in where], dtype=np.int64)
    qs = sorted(paged.image_len)
    cols = {c: getattr(t, c) for c in patch.COLUMNS}
    return {"format": np.int64(patch.FORMAT), "names": t.names, "stride": t.stride, "span": t.span, **cols,
            "block0": block(0), "unit0": pick(0, 1), "block1": block(1), "unit1": pick(1, 1),
            "image_queues": np.array(qs, dtype=np.int64), "image_units": np.array([paged.image_len[q] for q in qs], dtype=np.int64),
            "block_units": np.int64(paged.block_units)}


def load_patches(path):
    """(patch.Paged, the npz's arrays) of a `bindings.npz`."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k not in OUTPUT_KEYS}
    table = patch.Table(list(arrays["names"]), *[arrays[c] for c in patch.COLUMNS], stride=arrays["stride"], span=arrays["span"])
    image_len = {int(q): int(n) for q, n in zip(arrays["image_queues"], arrays["image_units"])}
    return patch.Paged(table, image_len, int(arrays["block_units"])), arrays


def block_records(blocks):
    return [{"queue": b.queue, "index": b.index, "kind": b.kind, "tensors": [[n, s, a, len(d) // b.vlanes] for n, s, a, d in b.tensors]}
            for b in sorted(blocks, key=lambda b: (b.queue, b.index))]


WORDS_PER_ROW = L.HEAD_INSTALLS * L.PACK      # words of one logit row in one slice


def output_regions(plan, rows):
    """The regions a step with `rows` logit rows returns: for each of the 4 byte-plane slices of the two logit quads, the first
    `rows` rows of 315 words (the west plane's even installs on `lg`, the odd installs of the east plane on `lg2`)."""
    lg, lg2 = plan.by_name["lg"].addr, plan.by_name["lg2"].addr
    return [(s, lg, rows * WORDS_PER_ROW) for s in L.LOGIT_SLICES] + [(s, lg2, rows * WORDS_PER_ROW) for s in L.LOGIT_SLICES_EAST]


def output_index():
    """Arrays that decode the returned words (regions in `output_regions` order, `rows` rows each, 320-byte words) into the
    logits: the f32 of vocabulary entry `vocab_index[i, k, x]` of a row is the little-endian bytes `plane_j` of lane `x` of the
    word `plane_word[i, k, j]` of region `plane_slice[i, k, j]`, j = 0..3 (the low byte plane first)."""
    installs, pack, lanes = L.HEAD_INSTALLS, L.PACK, 160
    ps = np.zeros((installs, pack, 4), np.int32)
    pw = np.zeros((installs, pack, 4), np.int32)
    vi = np.zeros((installs, pack, lanes), np.int32)
    for i in range(installs):
        group = 4 if i % 2 else 0
        for k in range(pack):
            ps[i, k] = [group + j for j in range(4)]
            pw[i, k] = 5 * i + k
            vi[i, k] = pack * (lanes * i + np.arange(lanes)) + k
    return {"plane_slice": ps, "plane_word": pw, "vocab_index": vi}


def decode_logits(words, rows, vocab, index):
    """The (rows, vocab) float32 logits of a step's output bytes (`n regions x rows x 315 x 320` bytes, region order of
    `output_regions`), by the arrays of `bindings.npz`."""
    raw = np.frombuffer(words, np.uint8).reshape(8, rows, WORDS_PER_ROW, -1)
    ps, pw, vi = index["plane_slice"], index["plane_word"], index["vocab_index"]
    out = np.zeros((rows, L.HEAD_INSTALLS * L.PACK * 160), np.float32)
    for r in range(rows):
        planes = raw[ps, r, pw, :160]                                     # (installs, pack, 4, 160)
        vec = np.ascontiguousarray(planes.transpose(0, 1, 3, 2)).view(np.float32)[..., 0]
        out[r, vi.reshape(-1)] = vec.reshape(-1)
    return out[:, :vocab]


# -- the values of a step plan --------------------------------------------------------------------------------------------------
SITE = re.compile(r"t(\d+)\.(?:emb\.(p(\d)[lh]|wpe(\d)|map\d)|L(\d+)\.(mk\.s\d+|kb\.b(\d+)\.s\d+|vb\.b(\d+)\.s\d+|kw\.p(\d)\.(\d)|vw\.s\d+))")


def plan_site_values(geometry, names, plan):
    """{site: value} of a `opentsp_vllm.step_plan.StepPlan` for the sites `names` of a bucket (`program.json` gives `geometry`),
    from the arrays of `plan.patches` alone: the K block of context block j of lane t is `kaddr[l] + kstride * block`, the V block
    `vaddr[l] + vstride * block`, a K write adds the word half and a V write the row, an embedding part is its zero word or its
    word of the token's row, `wpe` and the mask add the position, the Permute map is the segment."""
    g, pt = geometry, plan.patches
    out = {}
    for name in names:
        m = SITE.fullmatch(name)
        if not m:
            raise ValueError(f"site {name!r} is not one of the sites of a step program")
        lane = int(m[1])
        if m[2]:
            if m[3] is not None:
                part = g["parts"][int(m[3])]
                a = int(pt["emb_part"][lane, int(m[3])])
                out[name] = part["zero"] if a == g["part_zero_word"] else part["addr"] + a
            elif m[4] is not None:
                out[name] = g["wpe"][int(m[4])] + int(pt["wpe"][lane])
            else:
                out[name] = g["map_base"] + int(pt["emb_map"][lane])
            continue
        l, rest = int(m[5]), m[6]
        if rest.startswith("mk"):
            out[name] = g["mask_base"] + int(pt["mask"][lane])
        elif rest.startswith("kb"):
            out[name] = g["kaddr"][l] + g["kstride"] * int(pt["kblock"][lane, int(m[7])])
        elif rest.startswith("vb"):
            out[name] = g["vaddr"][l] + g["vstride"] * int(pt["vblock"][lane, int(m[8])])
        elif rest.startswith("kw"):
            block, half = (int(x) for x in pt["kwrite"][lane, int(m[9])])
            out[name] = g["kaddr"][l] + g["kstride"] * block + half
        else:
            block, row = (int(x) for x in pt["vwrite"][lane])
            out[name] = g["vaddr"][l] + g["vstride"] * block + row
    return out
