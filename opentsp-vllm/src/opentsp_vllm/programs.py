"""the bucket program cache reader. .

`ProgramSet` reads what `just vllm-programs` wrote (`<cache>/<digest>/manifest.json` and `<cache>/<digest>/P<T>_<L>_<C>/`, the layout of
`opentsp-vllm/src/opentsp_vllm/models/tinystories/program_cache.py`) and is the `programs` of `IssChip`: `info`, `buckets`, `program(bucket)` and
`pool_regions(block)`. It imports nothing from `tests/`; the small mappings it needs (the site values of a `StepPlan`, the decoding of
the output words, the pool layout) are ported here and compared with the generator's by the tests.
"""

import glob
import hashlib
from io import BytesIO
import json
import os
import re
import shlex
import struct
import zipfile
import zlib

import numpy as np

from opentsp_vllm.contracts import BucketProgram, ChipInfo, block_records, patch_columns

from opentsp_vllm import serving_artifacts as artifacts

SCHEMA = artifacts.SCHEMA
VERSION = artifacts.VERSION
FILES = (*artifacts.ELF_PAYLOAD, "program.json")
GRID_T, GRID_L, GRID_C = (1, 2, 4, 8), (1, 2, 4), (64, 128)

# the serve layout (opentsp-vllm/src/opentsp_vllm/models/tinystories/layout.py): the K pool is slices 44..59 (slice 44 + 2 (o mod 8) + byte plane for row o), the V
# pool of layer l the four slices from 20 + 4 (l div 2); a block is two words of each K slice and sixteen of each V slice
K_BASE, K_SLICES = 44, 16
V_BASE, V_SLICES = 20, 4
BLOCK_ROWS = 16
HIDDEN, HEADS = 64, 16  # the TinyStories GPT-2 of both serve models (2 and 8 layers)
LOGIT_PLANES = 4
PACK = 5  # vocabulary rows per embedding word (step_plan.PACK)
HEAD_INSTALLS, LOGIT_LANES = 63, 160  # the logit head: 63 installs of 160 packed rows (layout.HEAD_INSTALLS)
I32_MIN, I32_MAX = -(2**31), 2**31 - 1
OUTPUT_KEYS = ("plane_slice", "plane_word", "vocab_index")
CONTENT_FILES = artifacts.ELF_PAYLOAD  # the files whose SHA-256 `program.json` records under `files_sha256`


class ProgramsError(RuntimeError):
    """The program cache is missing, stale or not for this chip: the message names the `just` recipe that builds it."""


def default_root():
    return os.path.abspath(os.environ.get("OPENTSP_VLLM_CACHE") or os.path.expanduser("~/.cache/opentsp/vllm"))


def bucket_name(bucket):
    return "P%d_%d_%d" % tuple(bucket)


def parse_bucket(name):
    t, l, c = name[1:].split("_")
    return int(t), int(l), int(c)


def grid(grid_c=GRID_C):
    return [(t, l, c) for c in grid_c for t in GRID_T for l in GRID_L if l <= t]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# what reading the cache can raise while a builder publishes (an atomic rename, a file replaced) or after a crash (a truncated file, a
# directory where a file should be): every one of them is "not readable", never an escape of the reader
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, zipfile.BadZipFile, zlib.error, struct.error)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def output_index():
    """The only decoding of the output words a step program has (`program_cache.output_index`): the f32 of vocabulary entry
    `PACK (160 i + x) + k` is lane `x` of word `5 i + k` of the byte-plane slices of `lg` (even installs `i`) or `lg2` (odd ones)."""
    i = np.arange(HEAD_INSTALLS)[:, None, None]
    k = np.arange(PACK)[None, :, None]
    group = 4 * (i % 2)
    ps = np.broadcast_to(group + np.arange(LOGIT_PLANES)[None, None, :], (HEAD_INSTALLS, PACK, LOGIT_PLANES))
    pw = np.broadcast_to(PACK * i + k, (HEAD_INSTALLS, PACK, LOGIT_PLANES))
    vi = PACK * (LOGIT_LANES * i + np.arange(LOGIT_LANES)[None, None, :]) + k
    return {"plane_slice": ps, "plane_word": pw, "vocab_index": vi}


def _validated_payload(directory, record):
    """Return the verified executable bytes and their validated binding arrays."""
    payload = artifacts.verified_payload(directory, record, payload=artifacts.ELF_PAYLOAD)
    with np.load(BytesIO(payload["bindings.npz"])) as archive:
        if "names" not in archive.files:
            raise ValueError("bindings.npz has no patch site names")
        bindings = {key: archive[key] for key in archive.files}
    for key, want in output_index().items():
        got = bindings.get(key)
        if got is None or got.shape != want.shape:
            raise ValueError(f"bindings.npz {key} has shape {None if got is None else got.shape}, not {want.shape}")
        if key == "vocab_index" and not np.array_equal(np.sort(got, axis=None), np.arange(want.size)):
            raise ValueError(f"bindings.npz {key} is not a permutation of 0..{want.size - 1}")
        bad = np.flatnonzero(got != want)
        if bad.size:
            raise ValueError(f"bindings.npz {key} differs from the program's output layout at {bad.size} places (first at flat index {int(bad[0])})")
    return payload, bindings


def content_problem(directory, record):
    """Check a bucket using the same payload checks as the runtime loader."""
    try:
        _validated_payload(directory, record)
    except READ_ERRORS as error:
        return f"unreadable ({type(error).__name__}: {error})"
    return None


def _binding():
    try:
        from opentsp_vllm import _tsp_iss
    except ImportError as e:
        raise ImportError(
            f"ProgramSet checks the cache against the compiled binding opentsp_vllm._tsp_iss, which is not built or not importable ({e}); "
            "build it with `just vllm-binding`"
        ) from e
    return _tsp_iss


def _list(values):
    return ",".join(str(v) for v in values)


def build_command(layers, buckets, root=None):
    """The `just` command that builds (at least) `buckets`: the recipe reads `LAYERS`, `C`, `T` from the environment and skips the buckets
    that are in the cache already."""
    ts, cs = sorted({b[0] for b in buckets}), sorted({b[2] for b in buckets})
    parts = [f"OPENTSP_VLLM_CACHE={shlex.quote(str(root))}"] if root else []
    parts.append(f"LAYERS={layers}")
    if set(cs) != set(GRID_C):
        parts.append(f"C={_list(cs)}")
    if ts != list(GRID_T):
        parts.append(f"T={_list(ts)}")
    return " ".join(parts + ["just vllm-programs"])


# -- the values of a step plan ---------------------------------------------------------------------------------------------------
PATCH_OF = {"emb": "emb_part", "wpe": "wpe", "map": "emb_map", "mask": "mask", "kb": "kblock", "vb": "vblock", "kw": "kwrite", "vw": "vwrite"}
SITE = re.compile(
    r"t(\d+)\.(?:emb\.(p(\d)[lh]|wpe(\d)|map\d)|L(\d+)\.(mk\.s\d+|kb\.b(\d+)\.s\d+|vb\.b(\d+)\.s\d+|kw\.p(\d)\.(\d)|vw\.s\d+))"
)


class SiteMap:
    """The patch values of a `StepPlan` for the sites of a table, in table order, by one gather per kind of site: lane `t` of the plan's
    `patches` arrays at the site's place, plus the site's base address. The K block of context block j of lane t is
    `kaddr[l] + kstride * block`, the V block `vaddr[l] + vstride * block`, a K write adds the word half and a V write the row, an embedding
    part is its zero word or its word of the token's row, `wpe` and the mask add the position, the Permute map is the segment."""

    def __init__(self, names, geometry, bucket=None):
        g = geometry
        self.bucket = tuple(bucket) if bucket is not None else None
        self.geometry = g
        self.n = len(names)
        self.part_zero_word = g["part_zero_word"]
        cols = {k: ([], [], [], []) for k in ("emb", "wpe", "map", "mask", "kb", "vb", "kw", "vw")}  # index, lane, second index, base
        for i, name in enumerate(names):
            m = SITE.fullmatch(name)
            if not m:
                raise ProgramsError(f"site {name!r} is not one of the sites of a step program")
            lane = int(m[1])
            if m[2]:
                if m[3] is not None:
                    kind, j, base = "emb", int(m[3]), g["parts"][int(m[3])]["addr"]
                elif m[4] is not None:
                    kind, j, base = "wpe", 0, g["wpe"][int(m[4])]
                else:
                    kind, j, base = "map", 0, g["map_base"]
            else:
                layer, rest = int(m[5]), m[6]
                if rest.startswith("mk"):
                    kind, j, base = "mask", 0, g["mask_base"]
                elif rest.startswith("kb"):
                    kind, j, base = "kb", int(m[7]), g["kaddr"][layer]
                elif rest.startswith("vb"):
                    kind, j, base = "vb", int(m[8]), g["vaddr"][layer]
                elif rest.startswith("kw"):
                    kind, j, base = "kw", int(m[9]), g["kaddr"][layer]
                else:
                    kind, j, base = "vw", 0, g["vaddr"][layer]
            for col, v in zip(cols[kind], (i, lane, j, base)):
                col.append(v)
        self.cols = {k: tuple(np.array(c, dtype=np.int64) for c in v) for k, v in cols.items() if v[0]}
        self.zero = np.array([p["zero"] for p in g["parts"]], dtype=np.int64)
        self.kstride, self.vstride = g["kstride"], g["vstride"]

    def values(self, patches):
        """The int32 little-endian bytes of the values, one per site in table order.

        The arrays are checked first, as `emit_step_batch.lane_values` checks the host input of the emitter: integer arrays
        whose values fit int32 (never wrapped), the shapes of the bucket, and per lane either a padding lane (the null block, the
        zero word, the sinks; a plan of padding lanes only is the warm-up) or a real one: a position in `0..C-1` (the mask position is the same), a block table of ids in
        `1..sink-1` with no null block inside it that covers the position and fits the bucket, the K and V writes at the row of the
        position in the block of the table, and an embedding word of a token below the vocabulary. Real lanes come first. A
        violation is a `ValueError`; nothing is patched from a plan that fails."""
        arr = {kind: self._array(patches, kind) for kind in PATCH_OF.values()} if self.bucket else None
        if arr is not None:
            self._check(arr)

        def need(kind):
            return arr[kind] if arr is not None else self._array(patches, kind)

        out = np.empty(self.n, dtype=np.int64)
        for kind, (idx, lane, j, base) in self.cols.items():
            if kind == "emb":
                a = need("emb_part")[lane, j]
                out[idx] = np.where(a == self.part_zero_word, self.zero[j], base + a)
            elif kind == "wpe":
                out[idx] = base + need("wpe")[lane]
            elif kind == "map":
                out[idx] = base + need("emb_map")[lane]
            elif kind == "mask":
                out[idx] = base + need("mask")[lane]
            elif kind == "kb":
                out[idx] = base + self.kstride * need("kblock")[lane, j]
            elif kind == "vb":
                out[idx] = base + self.vstride * need("vblock")[lane, j]
            elif kind == "kw":
                w = need("kwrite")[lane, j]
                out[idx] = base + self.kstride * w[:, 0] + w[:, 1]
            else:
                w = need("vwrite")[lane]
                out[idx] = base + self.vstride * w[:, 0] + w[:, 1]
        if out.size and (int(out.min()) < I32_MIN or int(out.max()) > I32_MAX):
            raise ValueError(f"a patch value ({int(out.min())}..{int(out.max())}) does not fit int32")
        return out.astype("<i4").tobytes()

    @staticmethod
    def _array(patches, kind):
        try:
            a = np.asarray(patches[kind])
        except KeyError:
            raise ValueError(f"the plan has no {kind!r} patch values, which this program's sites need") from None
        if a.dtype.kind not in "iu":
            raise ValueError(f"the {kind!r} patch values are {a.dtype}, not integers")
        if a.size and (int(a.min()) < I32_MIN or int(a.max()) > I32_MAX):
            raise ValueError(f"the {kind!r} patch values {int(a.min())}..{int(a.max())} do not fit int32 (they would wrap)")
        return a.astype(np.int64)

    def _check(self, arr):
        T, _, C = self.bucket
        g, nb = self.geometry, C // 16
        sink, zero, vocab = g["sink"], self.part_zero_word, g["vocab"]
        shapes = {
            "kblock": (T, nb),
            "vblock": (T, nb),
            "kwrite": (T, 8, 2),
            "vwrite": (T, 2),
            "emb_part": (T, len(g["parts"])),
            "emb_map": (T,),
            "wpe": (T,),
            "mask": (T,),
        }
        for kind, shape in shapes.items():
            if arr[kind].shape != shape:
                raise ValueError(f"the {kind!r} patch values have shape {arr[kind].shape}, the bucket {self.bucket} needs {shape}")
        real = arr["vwrite"][:, 0] != sink
        n_real = int(real.sum())
        if not real[:n_real].all():  # no real lane at all is the warm-up plan
            raise ValueError(f"{n_real} real lanes of {T}, and they must be the first lanes (the padding lanes write the sink block)")
        for t in range(T):
            lane = {kind: a[t] for kind, a in arr.items()}
            if real[t]:
                self._check_real(t, lane, nb, sink, zero, vocab)
            else:
                self._check_padding(t, lane, sink, zero)

    @staticmethod
    def _check_padding(t, lane, sink, zero):
        ok = (
            not lane["kblock"].any()
            and not lane["vblock"].any()
            and (lane["kwrite"][:, 0] == sink).all()
            and not lane["kwrite"][:, 1].any()
            and lane["vwrite"].tolist() == [sink, 0]
            and (lane["emb_part"] == zero).all()
            and lane["emb_map"] == 0
            and lane["wpe"] == 0
            and lane["mask"] == 0
        )
        if not ok:
            raise ValueError(f"lane {t}: a padding lane (it writes the sink block) has values of a real one")

    @staticmethod
    def _check_real(t, lane, nb, sink, zero, vocab):
        C = nb * 16
        pos = int(lane["wpe"])
        if not 0 <= pos < C:
            raise ValueError(f"lane {t}: position {pos} is outside the context 0..{C - 1} of the bucket")
        if int(lane["mask"]) != pos:
            raise ValueError(f"lane {t}: the mask position {int(lane['mask'])} is not the position {pos}")
        table = lane["kblock"]
        if (table != lane["vblock"]).any():
            raise ValueError(f"lane {t}: the K block table {table.tolist()} is not the V block table {lane['vblock'].tolist()}")
        n = int(np.count_nonzero(table))
        if n == 0 or (table[:n] == 0).any():
            raise ValueError(f"lane {t}: the block table {table.tolist()} is empty or has the null block (0) inside it")
        if n <= pos // 16:
            raise ValueError(f"lane {t}: the block table has {n} blocks; position {pos} needs {pos // 16 + 1} and the bucket takes at most {nb}")
        bad = [b for b in table[:n].tolist() if not 1 <= b <= sink - 1]
        if bad:
            raise ValueError(f"lane {t}: block ids {bad} are outside 1..{sink - 1} (block 0 is the null block, {sink} the sink)")
        block, row = int(table[pos // 16]), pos % 16
        want = [sink] * 8
        want[row % 8] = block
        if lane["kwrite"][:, 0].tolist() != want or (lane["kwrite"][:, 1] != row // 8).any():
            raise ValueError(f"lane {t}: the K write {lane['kwrite'].tolist()} is not row {row} of block {block}")
        if lane["vwrite"].tolist() != [block, row]:
            raise ValueError(f"lane {t}: the V write {lane['vwrite'].tolist()} is not row {row} of block {block}")
        hit = np.flatnonzero(lane["emb_part"] != zero)
        local = int(lane["emb_part"][hit[0]]) if len(hit) == 1 else -1
        if len(hit) != 1 or not 0 <= local < zero:
            raise ValueError(f"lane {t}: the embedding words {lane['emb_part'].tolist()} are not one word of one part and the zero word elsewhere")
        sub = int(lane["emb_map"])
        token = (int(hit[0]) * zero + local) * PACK + sub
        if not 0 <= sub < PACK or token >= vocab:
            raise ValueError(f"lane {t}: the embedding row {local} of part {int(hit[0])} (rotation {sub}) is token {token}, outside the vocabulary of {vocab}")


class Decoder:
    """The output words of a step as logits: the f32 of vocabulary entry `vocab_index[i, k, x]` of a row is the little-endian bytes
    `plane_j` of lane `x` of word `plane_word[i, k, j]` of region `plane_slice[i, k, j]` (`bindings.npz`); the regions are the first `rows`
    rows of 315 words of the four byte-plane slices of `lg` and of `lg2`, one after the other."""

    def __init__(self, index, rows, vocab):
        self.ps, self.pw, self.vi = index["plane_slice"], index["plane_word"], index["vocab_index"]
        self.rows, self.vocab = rows, vocab
        self.installs, self.pack, self.lanes = self.vi.shape
        self.words_per_row = int(self.pw.max()) + 1

    def __call__(self, words, n_logits):
        raw = np.frombuffer(words, np.uint8)
        per_region = self.rows * self.words_per_row
        if raw.size % (8 * per_region) or raw.size == 0:
            raise ValueError(f"{raw.size} output bytes are not 8 regions of {self.rows} logit rows")
        raw = raw.reshape(8, self.rows, self.words_per_row, -1)
        out = np.zeros((n_logits, self.installs * self.pack * self.lanes), np.float32)
        order = self.vi.reshape(-1)
        for r in range(n_logits):
            planes = raw[self.ps, r, self.pw, : self.lanes]  # (installs, pack, 4, lanes)
            vec = np.ascontiguousarray(planes.transpose(0, 1, 3, 2)).view(np.float32)[..., 0]
            out[r, order] = vec.reshape(-1)
        return np.ascontiguousarray(out[:, : self.vocab])


class ProgramSet:
    """The cached bucket programs of one model: `ProgramSet(cache=None, layers=8, grid_c=(64, 128), buckets=None, digest=None,
    profile="ref")`.

    `cache` is the root (default `$OPENTSP_VLLM_CACHE`, else `~/.cache/opentsp/vllm`). The entry read is the newest one whose manifest is of
    this specification version and profile digest (those of the binding), of `layers` layers, and with every bucket complete and of the
    manifest's digest; `digest` names one entry. `buckets` defaults to the whole grid of `grid_c`. A bucket that is missing or stale fails
    with the `just vllm-programs` command that builds it (`ProgramsError`), and so does a file that cannot be read (a builder is
    publishing, a file was truncated): no read of the cache raises anything else.

    The digests are a build key (what the generator made the bucket from: sources, profile, layer count), not a content check. The
    content is checked when a program is first loaded (`content_problem`): the SHA-256 of each file where `program.json` records it
    (`files_sha256`, written at publication), and `bindings.npz` against the one output layout of a step program. A damaged bucket is a
    `ProgramsError` naming the command that rebuilds it.
    """

    def __init__(self, cache=None, layers=8, grid_c=GRID_C, buckets=None, digest=None, profile="ref"):
        binding = _binding()
        self.root = os.path.abspath(cache) if cache else default_root()
        self._explicit_root = bool(cache or os.environ.get("OPENTSP_VLLM_CACHE"))
        self.layers, self.profile, self.grid_c = layers, profile, tuple(grid_c)
        self._want = sorted(set(buckets)) if buckets else grid(self.grid_c)
        self.spec_version, self.profile_sha256 = binding.SPEC_VERSION, binding.profile_sha256(profile)
        self.manifest, self.entry = self._select(digest)
        self.digest = self.manifest["digest"]
        try:
            self._records = {b: _read_json(os.path.join(self.entry, bucket_name(b), "program.json")) for b in self._want}
            self._programs = {}
            self._pool_layout()
        except READ_ERRORS as e:  # a builder replaced the entry between the check and this read
            raise ProgramsError(self._unreadable(self.entry, e, self._want)) from e

    # -- the entry ------------------------------------------------------------------------------------------------------------
    def _command(self, buckets):
        cache = self.root
        if os.path.isfile(os.path.join(cache, "manifest.json")):
            cache = os.path.dirname(os.path.realpath(cache))
        return build_command(self.layers, buckets, cache if self._explicit_root else None)

    def _unreadable(self, entry, error, buckets):
        return (
            f"the program cache entry {os.path.basename(entry)} in {self.root} is unreadable or changing ({type(error).__name__}: {error}); "
            f"wait for the builder, or build the programs with `{self._command(buckets)}`"
        )

    def _why_not(self, manifest):
        """Why the manifest is not for this chip, or None."""
        if not isinstance(manifest.get("digest"), str):
            return "the manifest has no digest"
        if manifest.get("model") != "tinystories" or manifest.get("chips") != 1 or manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
            return f"schema {manifest.get('schema')!r} version {manifest.get('version')!r}, not {SCHEMA} {VERSION}"
        if manifest.get("route") != [0] or manifest.get("shared_preload") != "weights.safetensors":
            return "expected a single-chip route and the shared weights.safetensors"
        if manifest.get("spec_version") != self.spec_version:
            return f"built for spec_version {manifest.get('spec_version')}, this chip is {self.spec_version}"
        if manifest.get("profile") != self.profile or manifest.get("profile_sha256") != self.profile_sha256:
            return f"built for the profile {manifest.get('profile')} {str(manifest.get('profile_sha256'))[:12]}, this chip's is {self.profile} {self.profile_sha256[:12]}"
        if manifest.get("layers") != self.layers:
            return f"built for {manifest.get('layers')} layers, not {self.layers}"
        return None

    def _problems(self, entry, manifest):
        """{bucket: what is wrong} of the wanted buckets of an entry. A file that cannot be read (it is being replaced, or was
        truncated) is a problem of its bucket, never an exception. The digest checks are of the build key (what the generator made
        the bucket from), not of the content of the files: a flipped bit in `program.elf` is not seen."""
        out = {}
        for b in self._want:
            d = os.path.join(entry, bucket_name(b))
            try:
                if not os.path.isdir(d):
                    out[b] = "missing"
                    continue
                absent = [f for f in FILES if not os.path.isfile(os.path.join(d, f))]
                if absent:
                    out[b] = f"incomplete ({', '.join(absent)} missing)"
                    continue
                rec = _read_json(os.path.join(d, "program.json"))
                want = _sha((manifest["digest"] + bucket_name(b)).encode())
                if rec.get("digest") != want:
                    out[b] = f"program.json digest {str(rec.get('digest'))[:12]} is not {want[:12]} (sha256 of the entry digest and the bucket name)"
                elif rec.get("model_digest") != manifest["digest"]:
                    out[b] = f"program.json model_digest {str(rec.get('model_digest'))[:12]} is not the entry's digest {manifest['digest'][:12]}"
            except READ_ERRORS as e:
                out[b] = f"unreadable ({type(e).__name__}: {e})"
        return out

    def _select(self, digest):
        def mtime(path):
            try:
                return os.path.getmtime(path)
            except OSError:
                return 0.0

        direct = os.path.join(self.root, "manifest.json")
        manifests = [direct] if os.path.isfile(direct) else sorted(glob.glob(os.path.join(self.root, "*", "manifest.json")), key=mtime, reverse=True)
        found, refusals, stale, skipped = [], [], [], []
        for path in manifests:
            entry = os.path.dirname(path)
            try:
                man = _read_json(path)
                if not isinstance(man, dict):
                    raise ValueError("not a JSON object")
            except READ_ERRORS as e:
                skipped.append(f"{os.path.basename(entry)}: manifest.json unreadable ({type(e).__name__}: {e})")
                continue
            if digest is not None and man.get("digest") != digest:
                continue
            why = self._why_not(man)
            if why:
                refusals.append(f"{os.path.basename(entry)}: {why}")
                continue
            problems = self._problems(entry, man)
            if problems:
                stale.append((entry, man, problems))
            else:
                found.append((man, entry))
        if found:
            return found[0]
        unread = f" (skipped, unreadable: {'; '.join(skipped)})" if skipped else ""
        if digest is not None and not (refusals or stale):
            raise ProgramsError(f"{self.root} has no entry of digest {digest}{unread}; build the programs with `{self._command(self._want)}`")
        if refusals and not stale and digest is not None:
            raise ProgramsError(f"the entry {digest[:16]} is not for this chip: {refusals[0]}")
        if stale:
            entry, _, problems = stale[0]
            lines = "; ".join(f"{bucket_name(b)} {why}" for b, why in problems.items())
            gone = [b for b, why in problems.items()]
            raise ProgramsError(
                f"the program cache entry {os.path.basename(entry)} in {self.root} lacks {len(gone)} of {len(self._want)} buckets or has stale ones "
                f"(missing: {', '.join(bucket_name(b) for b in gone)}; details: {lines}); build them with `{self._command(gone)}`"
            )
        seen = f" (entries found but not for this chip: {'; '.join(refusals + skipped)})" if refusals or skipped else ""
        raise ProgramsError(
            f"no program cache entry for {self.layers} layers, spec_version {self.spec_version} and the {self.profile} profile "
            f"{self.profile_sha256[:12]} in {self.root}{seen}; build the programs with `{self._command(self._want)}`"
        )

    def _pool_layout(self):
        keys = ("kaddr", "vaddr", "kstride", "vstride", "blocks", "sink", "vocab")
        layouts = {json.dumps({k: r["geometry"][k] for k in keys}, sort_keys=True) for r in self._records.values()}
        if len(layouts) != 1:
            raise ProgramsError(f"the buckets of {self.entry} disagree on the pool layout")
        self.geometry = next(iter(self._records.values()))["geometry"]

    # -- the programs ---------------------------------------------------------------------------------------------------------
    @property
    def info(self):
        g = self.geometry
        return ChipInfo(self.layers, g["vocab"], HIDDEN, HEADS, g["blocks"], tuple(sorted({b[2] for b in self._want})))

    def buckets(self):
        return list(self._want)

    def check_package(self, data):
        """The weight package (bytes) is the one the programs were built for: its SHA-256 is the manifest's ."""
        if _sha(data) != self.manifest["package_sha256"]:
            raise ProgramsError(
                f"the weight package ({_sha(data)[:12]}) is not the one the programs of {self.entry} were built for "
                f"({self.manifest['package_sha256'][:12]}, weights {self.manifest['weights']}); prepare both again with `just showcase-prepare`"
            )

    def program(self, bucket):
        bucket = tuple(bucket)
        if bucket not in self._records:
            raise KeyError(f"bucket {bucket} is not in this program set ({len(self._want)} buckets)")
        if bucket not in self._programs:
            try:
                self._programs[bucket] = self._load(bucket)
            except READ_ERRORS as e:  # a file of the bucket vanished, was truncated or replaced since the check
                raise ProgramsError(self._unreadable(os.path.join(self.entry, bucket_name(bucket)), e, [bucket])) from e
        return self._programs[bucket]

    def _load(self, bucket):
        d = os.path.join(self.entry, bucket_name(bucket))
        rec = self._records[bucket]
        try:
            payload, bindings = _validated_payload(d, rec)
        except READ_ERRORS as error:
            raise ProgramsError(
                f"the bucket {bucket_name(bucket)} of the program cache entry {os.path.basename(self.entry)} in {self.root} is damaged: "
                f"{error}; rebuild it with `{self._command([bucket])}`"
            ) from error
        names = [str(n) for n in bindings["names"]]
        index = {key: bindings[key] for key in OUTPUT_KEYS}
        sites = SiteMap(names, rec["geometry"], bucket)
        decoder = Decoder(index, bucket[1], rec["geometry"]["vocab"])
        recorded = rec.get("cycles")
        # `tsp-iss run` records the last cycle that did work; the chip's CYCLE counter reads the cycles executed, one more
        counted = None if recorded is None else recorded + 1
        limit = 2 * (recorded if recorded is not None else rec["schedule_cycles"])
        return BucketProgram(
            name=bucket_name(bucket),
            elf=payload["program.elf"],
            site_values=lambda plan: self._site_values(sites, bucket, plan),
            decode=lambda words, plan: decoder(words, plan.n_logits),
            patch_table=patch_columns(bindings),
            outputs=[tuple(r) for r in rec["outputs"]],
            max_cycles=limit,
            cycles=counted,
            blocks=block_records(rec["blocks"]),
            site_names=names,
        )

    @staticmethod
    def _site_values(sites, bucket, plan):
        if tuple(plan.bucket) != bucket:
            raise ValueError(f"the plan is for the bucket {tuple(plan.bucket)}, this program is {bucket}")
        return sites.values(plan.patches)

    def pool_regions(self, block):
        """The `(slice, address, words)` regions of block `block` (0 the null block, 128 the sink) in every layer: two words of each of the 16
        K slices and sixteen words of each of the four V slices."""
        g = self.geometry
        if not 0 <= block <= g["sink"]:
            raise ValueError(f"block {block} is outside the pool 0..{g['sink']}")
        out = []
        for layer in range(self.layers):
            k = g["kaddr"][layer] + g["kstride"] * block
            out += [(K_BASE + s, k, 2) for s in range(K_SLICES)]
            v = g["vaddr"][layer] + g["vstride"] * block
            out += [(V_BASE + 4 * (layer // 2) + s, v, BLOCK_ROWS) for s in range(V_SLICES)]
        return out
