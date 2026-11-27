"""Compile and execute reduced GPT-2 pipelines used for reference tests."""
from collections import deque
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
import time

import numpy as np

from opentsp_vllm import ship_plan
from opentsp_vllm.models.gpt2 import paths as _gpaths
ROOT = _gpaths.ROOT
from opentsp_tools import refasm
tsp_as = refasm.load()
from opentsp_vllm import serving_artifacts as artifacts
from opentsp_tools.programs import data
from opentsp_tools import profile as tprofile
from opentsp_tools.programs import ref_wide as rw
from opentsp_vllm.models.gpt2 import attention as sa
from opentsp_vllm.models.gpt2 import layer as sl
from opentsp_vllm.models.gpt2 import system as S
from opentsp_vllm.models.gpt2.wide import pad_lanes
from opentsp_vllm.models.gpt2.config import GPT2_SMALL

T, CHUNKS = S.T, S.CHUNKS
CACHE = os.path.expanduser("~/.cache/opentsp/ship")
POOL_NAME = re.compile(r"k\d+|v\d+p\d+")
MAX_FIXTURE_LAYERS = 8


def default_route(layers):
    if not 1 <= layers <= MAX_FIXTURE_LAYERS:
        raise ValueError("a reduced GPT-2 fixture requires one to eight layers")
    return list(range(layers)), layers


def digest_files(root):
    """All maintained builder/reference sources and architectural interface inputs."""
    root = Path(root)
    directories = ("opentsp-sw-toolchain/python", "opentsp-vllm/src/opentsp_vllm/models/gpt2", "opentsp-arch-spec/reference")
    files = [p for d in directories for p in (root / d).rglob("*.py") if "__pycache__" not in p.parts]
    files += [root / "opentsp-vllm/src/opentsp_vllm/serving_artifacts.py",
              root / "opentsp-arch-spec/config/ref.json", root / "opentsp-arch-spec/tools/tsp_as.py",
              root / "opentsp-arch-spec/isa-db/generated/encoding.json"]
    return sorted(str(p.relative_to(root)) for p in files)


def source_digest(w, layers, route, chips, root=None, kv_chunks=CHUNKS):
    """The cache key: the content of every file of `digest_files(root)`, the weights' digest and the shape of the build (`root` defaults to this tree)."""
    root = root or ROOT
    h = hashlib.sha256()
    for n in digest_files(root):
        h.update(n.encode() + b"\0" + (Path(root) / n).read_bytes())
    h.update(w.digest().encode() + json.dumps([layers, route, chips, T, kv_chunks]).encode())
    return h.hexdigest()[:20]


def stage_manifest(w, layers, route, chips, kv_chunks=CHUNKS):
    profile = Path(ROOT) / "opentsp-arch-spec/config/ref.json"
    record = artifacts.manifest("gpt2", chips, route=route, layers=layers, weights_sha256=w.digest(),
                                spec_version="0.1", profile="ref", profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest())
    if kv_chunks != CHUNKS:
        record["kv_chunks"] = kv_chunks
    return record


class Stage:
    """A serving unit loaded from its object, physical preloads and host bindings."""

    def __init__(self, directory, index, chip, expected_digest, *, layout=None):
        unit = Path(directory) / f"s{index}"
        descriptor = (unit / "program.json").read_bytes()
        self.meta = json.loads(descriptor)
        observed = (artifacts.unit_digest(self.meta) if layout is not None
                    else hashlib.sha256(descriptor).hexdigest())
        if observed != expected_digest:
            raise ValueError(f"stage {index}: program.json does not match the package manifest")
        payload = (artifacts.verified_payload(unit, self.meta, payload=artifacts.ELF_PAYLOAD)
                   if layout is not None else artifacts.verified_payload(unit, self.meta))
        self.elf = payload["program.elf"]
        if layout is not None and tsp_as.elf_identity(self.elf) != self.meta["elf_sha256"]:
            raise ValueError(f"stage {index}: decoded ELF differs from release identity")
        if self.meta["chip"] != chip:
            raise ValueError(f"stage {index} targets chip {self.meta['chip']}, expected {chip}")
        self.budget = self.meta["budget"]
        with np.load(io.BytesIO(payload["bindings.npz"]), allow_pickle=False) as bindings:
            self.dma_cycles = bindings["dma_cycles"].tolist()
        _, self.images, externs = tsp_as.read_object(self.elf, "ref")
        self.externs = {x["name"]: (x["slice"], x["addr"], x["size"]) for x in externs}
        self.warm_names = None
        if layout is None:
            tensors = tsp_as.read_weights(payload["sram.safetensors"], "ref")
            resolved = {}
        else:
            from .weight_layout import RESIDENT

            sections = tsp_as._read_elf(self.elf)
            tensors = tsp_as.object_preloads(self.elf, "ref", externs)
            resolved = layout.resolve(chip, sections.get(b".tsp.recipe"), self.meta["elf_sha256"])
            if tensors.keys() & resolved.keys() or tensors.keys() | resolved.keys() != self.externs.keys():
                raise ValueError(f"stage {index}: layout sources must cover each extern exactly once")
            zero_records = sections.get(b".tsp.zero", b"")
            self.warm_names = {name for name in self.externs if not RESIDENT.fullmatch(name)}
            for key, phase in struct.iter_unpack("<16sI", zero_records):
                name = key.rstrip(b"\0").decode("ascii")
                if bool(phase & 2) != (name in self.warm_names):
                    raise ValueError(f"stage {index}: zero application phase differs for {name}")
        self.text_pages = {}
        self.runs = []
        self.word_count = 0
        lanes = tsp_as.ISA("ref").pvars["vlanes"]
        for name, (slice_id, addr, size) in self.externs.items():
            if name in resolved:
                payload = resolved.pop(name)
            else:
                if name not in tensors:
                    raise ValueError(f"stage {index}: extern {name} has no preload")
                payload = b"".join(tensors.pop(name)[1])
            if len(payload) != size * lanes:
                raise ValueError(f"stage {index}: extern {name} needs {size} preload words")
            page = re.fullmatch(r"tx(\d+)b(\d+)([ab]?)", name)
            if page:
                q, block, half = page.groups()
                self.text_pages.setdefault(int(q), {})[int(block), half] = payload
            self.runs.append((slice_id, addr, payload))
            self.word_count += size


    def schedule(self):
        """First and last issue of each queue, including its fetched parcels in SRAM."""
        result = {}
        for q, image in self.images.items():
            full = list(image)
            for _, page in sorted(self.text_pages.get(q, {}).items()):
                full.extend(int(u) for u in np.frombuffer(page, dtype="<u4") if int(u) != tsp_as.PADDING)
            entries = tsp_as.raise_images({q: full})
            result[q] = entries[0][0], entries[-1][0]
        return result


def save_stage(st, directory, name, extra, log=print):
    """Publish the same four files as a single-chip bucket, inside the unpublished build directory."""
    unit = Path(directory) / name
    unit.mkdir()
    shutil.copyfile(st["elf"], unit / "program.elf")
    shutil.copyfile(st["sram_weights"], unit / "sram.safetensors")
    dma_cycles = [cycle for cycle, _ in st["case"].get("host", {}).get("dma", [])]
    np.savez(unit / "bindings.npz", dma_cycles=np.asarray(dma_cycles, dtype="<i8"))
    index = int(name[1:])
    record = dict(schema=artifacts.SCHEMA, version=artifacts.VERSION, budget=st["case"]["budget"]["cycles"], chip=extra["route"][index],
                  **{key: value for key, value in extra.items() if key not in ("chips", "route")})
    artifacts.seal(unit, record)
    log(f"  {name}: published program for chip {record['chip']}")
    return artifacts.file_digest(unit / "program.json")


def write_stage_words(s, chip, st):
    """Initialize SRAM from the validated physical SRAM tensors bound to this object's externs."""
    for slice_id, addr, payload in st.runs:
        s.write_words(chip, slice_id, addr, payload)
    return st.word_count


def host_logits(w, xn):
    """The head by the tiled matvec (tile 128, zero bias last, as the chips) of the 393 tiles (the table padded with zero rows), in f32."""
    padded_vocab = ((len(w.wte) + sl.TILE - 1) // sl.TILE) * sl.TILE
    rows = np.zeros((padded_vocab, w.wte.shape[1]), np.float16)
    rows[:len(w.wte)] = w.wte
    return rw.matvec_tiled(rows, xn, np.zeros(len(rows), np.float32), tile=sl.TILE)


def build_cache(w, layers, route, chips, d, log=print, kv_chunks=CHUNKS):
    """Assemble the reduced layer fixture and publish its weights separately."""
    p = tprofile.load("ref")
    Ps = [S.layer_params(w.layers[l]) for l in range(layers)]
    zero = np.zeros((kv_chunks * sl.CHUNK, T, sl.TILE), np.float16)
    pools = [(zero, zero, S.counts_for(1, kv_chunks))] * layers
    work = tempfile.mkdtemp(prefix="ship-build-")
    try:
        t0 = time.time()
        stages, infos = S.build_stages(p, layers, Ps, pools, np.zeros(GPT2_SMALL.hidden, np.float32), 0, work, chunks=kv_chunks, route=route, chips=chips, harness_input=False)
        log(f"programs of {layers} stages generated and assembled: {time.time() - t0:.0f} s; stages end at cycles {[inf['end'] for inf in infos]}")
        os.makedirs(d, exist_ok=True)
        unit_digests = {}
        for i, st in enumerate(stages):
            unit_digests[f"s{i}"] = save_stage(st, d, f"s{i}", dict(end=infos[i]["end"], route=route, chips=chips, layer=i, kind="layer"), log)
        record = stage_manifest(w, layers, route, chips, kv_chunks)
        record["units_sha256"] = unit_digests
        (Path(d) / "manifest.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        return infos
    finally:
        shutil.rmtree(work, ignore_errors=True)


def ensure_cache(directory, build, load=None):
    """Serialize builders for one digest and publish a complete, immutable directory.

    The sibling lock survives publication. Failed builds remove only their own
    temporary directory; an incomplete destination is reported for inspection.
    `load`, when given, opens all artifacts before this digest lock is released,
    so a concurrent eviction cannot remove a stage halfway through a load.
    """
    os.makedirs(os.path.dirname(directory), exist_ok=True)
    with open(directory + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.isfile(os.path.join(directory, "manifest.json")):
            if os.path.exists(directory):
                raise RuntimeError(f"incomplete cache directory {directory}; move it aside before rebuilding")
            tmp = tempfile.mkdtemp(prefix=os.path.basename(directory) + ".tmp", dir=os.path.dirname(directory))
            try:
                build(tmp)
                if not (Path(tmp) / "manifest.json").is_file():
                    raise ValueError("builder did not write the serving manifest")
                os.rename(tmp, directory)
            finally:
                if os.path.isdir(tmp):
                    shutil.rmtree(tmp)
        if load is not None:
            return load(directory)


class Pipeline:
    def __init__(self, w, layers=2, log=print, cache=CACHE, resident=False, history_limit=256, kv_chunks=CHUNKS):
        """Execute the reduced layer fixture with optional device-resident KV."""
        # None explicitly requests full history; zero disables retention.
        self.stats = deque(maxlen=history_limit)
        self.w, self.layers, self.log, self.resident = w, layers, log, resident
        self.pool_slots, _ = ship_plan.kv_limits(kv_chunks)
        self.kv_chunks = kv_chunks
        self._warm = None                                       # the resident System between steps
        self.route, self.chips = default_route(layers)
        self.p = tprofile.load("ref")
        self.Ps = [S.layer_params(w.layers[l]) for l in range(layers)]
        def load(directory):
            manifest_bytes = (Path(directory) / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            self.package_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            expected = stage_manifest(w, layers, self.route, self.chips, kv_chunks)
            if manifest.get("kv_chunks", CHUNKS) != kv_chunks or any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{directory}: incompatible serving manifest or stage placement")
            units = manifest.get("units_sha256")
            if not isinstance(units, dict) or set(units) != {f"s{i}" for i in range(len(self.route))}:
                raise ValueError(f"{directory}: serving manifest has an incomplete stage inventory")
            return [Stage(directory, i, chip, units[f"s{i}"]) for i, chip in enumerate(self.route)]

        # An explicit published package is relocatable and may be read-only.
        # A managed digest entry supplied directly already has a sibling lock;
        # use a read handle so loading does not require modifying that package.
        d = os.fspath(cache)
        if (Path(d) / "manifest.json").is_file():
            try:
                lock = open(d + ".lock", "rb")
            except FileNotFoundError:
                self.stages = load(d)  # immutable published package, outside cache management
            else:
                with lock:
                    fcntl.flock(lock, fcntl.LOCK_SH)
                    self.stages = load(d)
        else:
            d = os.path.join(d, source_digest(w, layers, self.route, self.chips, kv_chunks=kv_chunks))
            self.stages = ensure_cache(d, lambda tmp: build_cache(w, layers, self.route, self.chips, tmp, log, kv_chunks), load)
        self.dir = d
        log(f"programs and weight packages available in {d}")
        self.wave = None
        self.health_target = None
        self.completion_flips = []
        self.pool = {}                                   # (stage, extern name) -> the words last read back from the System
        self.twin_k = [np.zeros((self.pool_slots, T, sl.TILE), np.float16) for _ in range(layers)]
        self.twin_v = [np.zeros_like(self.twin_k[0]) for _ in range(layers)]

    def step(self, token, pos, check=True, slot=None, counts=None):
        """Run one token with its physical KV slot and visible page counts."""
        row = (token, pos, pos if slot is None else slot, S.counts_for(pos + 1, self.kv_chunks) if counts is None else counts)
        outputs, stat = self.batch([row], check)
        return outputs[0], stat

    def _wave_schedule(self, *, repeat_trim=0):
        """Prepare repeated text. A nonzero trim uses per-queue refill timing."""
        if repeat_trim < 0:
            raise ValueError("repeat trim must be nonnegative")
        if self.wave is None:
            schedules = [stage.schedule() for stage in self.stages]
            starts = [min(first for first, _ in sched.values()) for sched in schedules]
            cap = self.p.cfg["icu"]["iq_depth"] // 4
            rate = self.p.cfg["icu"]["fetch_bytes_per_cycle"] // 4
            # Drain the previous lane and admit one full initial image at the configured
            # Enqueue rate before its first parcel. The stage budgets include pipeline drain.
            guard = (cap + rate - 1) // rate + 1
            period = max(stage.budget - start for stage, start in zip(self.stages, starts)) + guard + 128 - repeat_trim
            if period <= 0:
                raise ValueError("repeat trim exhausts the stage period")
            waves = []
            for stage, sched, start in zip(self.stages, schedules, starts):
                injections = {}
                for q, units in stage.images.items():
                    first, last = sched[q]
                    # Gaps in queue text are relative to its last issued parcel. This
                    # repeat image is identical for every subsequent lane and request.
                    gap = period + first - last - 1
                    if gap < 0:
                        raise ValueError(f"queue {q} would still be occupied at the next launch")
                    prefix = [65535] * (gap // 65535)
                    if gap % 65535:
                        prefix.append(gap % 65535)
                    skip = next(i for i, u in enumerate(units) if not tsp_as.is_nop_word(u))
                    repeat = prefix + units[skip:]
                    refill = first if repeat_trim else start
                    injections.setdefault(refill - guard, []).append((q, repeat[:cap]))
                    # A longer initial delay can add timing units. Admit the suffix
                    # only after the first parcel frees space, before its fetch can return.
                    if len(repeat) > cap:
                        injections.setdefault(first + 1, []).append((q, repeat[cap:]))
                waves.append((start - guard, sorted(injections.items())))
            self.wave = (period, waves)
        return self.wave

    def batch(self, rows, check=True):
        """Execute one bounded group of ordered (token, position, KV slot, page counts) tuples on a shared clock."""
        from opentsp_vllm.system import System
        if not 1 <= len(rows) <= 8:
            raise ValueError("a SHIP wave takes one to eight tokens")
        t0 = time.perf_counter()
        w, p = self.w, self.p
        period, waves = self._wave_schedule() if len(rows) > 1 else (0, ())
        if any(not 0 <= slot < self.pool_slots or np.shape(counts) != (self.kv_chunks, ship_plan.PAGES_PER_CHUNK)
               for _, _, slot, counts in rows):
            raise ValueError(f"the slot or page counts do not match the {self.kv_chunks}-chunk KV pool")
        xs = [S.embed(w, token, pos) for token, pos, _, _ in rows]
        # The same per-lane mask goes to every layer in the chain.
        encoded_counts = [sa.enc_weights(p, counts) for _, _, _, counts in rows]
        _, pos, slot, _ = rows[0]
        warm = self._warm is not None
        s, self._warm = self._warm, None                          # a failed step leaves no System to resume from
        object_bytes = 0
        if warm:
            s.restart()
        else:
            s = System("ref", self.chips)
            for st, chip in zip(self.stages, self.route):
                s.load_object(chip, st.elf)
                object_bytes += len(st.elf)
        n_words = 0
        initial_words = control_words = kv_words = kv_reads = 0
        for i, (st, chip) in enumerate(zip(self.stages, self.route)):
            if not warm:
                written = write_stage_words(s, chip, st)
                initial_words += written
                n_words += written
            dyn = dict(sl.map_weights(p, T, self.kv_chunks, slot, 7 + i))
            dyn.update(encoded_counts[0])
            for name, (sli, addr, words) in st.externs.items():
                if POOL_NAME.fullmatch(name) and not self.resident:
                    dyn[name] = self.pool.get((i, name), bytes(words * p.vlanes))
            for name, blob in dyn.items():
                sli, addr, words = st.externs[name]
                assert len(blob) == words * p.vlanes, (name, len(blob), words)
                s.write_words(chip, sli, addr, blob)
                n_words += words
                if POOL_NAME.fullmatch(name):
                    kv_words += words
                else:
                    control_words += words
        for j, x in enumerate(xs):
            vectors = (v for tile in rw.split(x, sl.TILE) for v in data.to_planes(pad_lanes(tile, p.vlanes)))
            for at, vector in zip(self.stages[0].dma_cycles, vectors, strict=True):
                s.dma_write(self.route[0], vector, j * period + at)
        updates = []
        text_bytes = 0
        for j in range(1, len(rows)):
            for i, (boundary, injections) in enumerate(waves):
                for at, images in injections:
                    s.inject_text(self.route[i], j * period + at, images)
                    text_bytes += sum(4 * len(units) for _, units in images)
                updates.append((j * period + boundary, j, i))
        from opentsp_vllm.ras import check_completion, probe_word
        if self.health_target is not None:
            chip, slice_id, addr = self.health_target
            if not 0 <= chip < self.chips:
                raise RuntimeError("health check targets a chip outside the fence")
            # Exercise the implicated address on every member, including idle routing nodes.
            for member in range(self.chips):
                probe_word(lambda: s.read_words(member, slice_id, addr, 1),
                           lambda data: s.write_words(member, slice_id, addr, data), p.vlanes)
            self.health_target = None
        t1 = time.perf_counter()
        budget = max(st.budget for st in self.stages) + 1000 + period * (len(rows) - 1)
        def reports():
            return [(chip, s.ras_snapshot(chip)) for chip in range(self.chips)]

        try:
            for at, j, i in sorted(updates):
                _, res = s.run_to_done(max_cycles=at - 1)
                if res["reason"] != "limit":
                    raise RuntimeError(f"expected SHIP control boundary at {at}, got {res}")
                _, _, next_slot, _ = rows[j]
                dyn = dict(sl.map_weights(p, T, self.kv_chunks, next_slot, 7 + i))
                dyn.update(encoded_counts[j])
                for name, blob in dyn.items():
                    sli, addr, words = self.stages[i].externs[name]
                    s.write_words(self.route[i], sli, addr, blob)
                    n_words += words
                    control_words += words
            _, res = s.run_to_done(max_cycles=budget)
            if res["reason"] != "done":
                raise RuntimeError(f"pipeline stopped at {res['reason']} before completion")
            s.finish()
        except RuntimeError as exc:
            check_completion(reports(), exc)
        for flip in self.completion_flips:
            s.inject_sram(*flip)
        self.completion_flips.clear()
        s.service_scrub()
        check_completion(reports())
        t2 = time.perf_counter()
        for chip in range(self.chips):
            d = s.diagnostics(chip)
            if d:
                raise RuntimeError(f"chip {chip}: diagnostics {d[:3]}")
        tiles = T
        outputs = [[] for _ in rows]
        arrivals = [0] * len(rows)
        collectors = self.route[-1:]
        for chip in collectors:
            vecs = s.egress(chip)
            expected = len(rows) * tiles * 4
            if len(vecs) != expected:
                raise RuntimeError(f"chip {chip} gave {len(vecs)} vectors, expected {expected}")
            for j, parts in enumerate(outputs):
                start = j * tiles * 4
                arrivals[j] = max(arrivals[j], vecs[start + tiles * 4 - 1][0])
                for t in range(tiles):
                    planes = [v[:sl.TILE] for _, v in vecs[start + t * 4:start + (t + 1) * 4]]
                    parts.append(data.from_planes(planes, "f32"))
        outputs = [np.concatenate(parts) for parts in outputs]
        host_out = tiles * 4 * len(collectors) * len(rows)
        if self.resident:
            self._warm = s                                        # the pools stay on the chips
        else:
            for i, (st, chip) in enumerate(zip(self.stages[:self.layers], self.route)):         # the pools as the System leaves them
                for name, (sli, addr, words) in st.externs.items():
                    if POOL_NAME.fullmatch(name):
                        self.pool[i, name] = s.read_words(chip, sli, addr, words)
                        kv_reads += words
        stat = dict(pos=pos, slot=slot, cycles=res["cycle"], load_s=t1 - t0, run_s=t2 - t1,
                    collect_s=time.perf_counter() - t2, write_words=n_words, initial_write_words=initial_words,
                    control_write_words=control_words, kv_write_words=kv_words, kv_read_words=kv_reads,
                    object_load_bytes=object_bytes, host_vectors_in=4 * T * len(rows), host_vectors_out=host_out,
                    stage_ticks=[s.active_ticks(chip) for chip in self.route], tokens=len(rows), initiation_interval=period, completion_cycles=arrivals, host_text_bytes=text_bytes)
        if check:
            for x, out, (_, pos, slot, counts) in zip(xs, outputs, rows):
                want = self.twin_step(x, slot, counts)
                if not np.array_equal(out.view(np.uint32), want.view(np.uint32)):
                    bad = np.flatnonzero(out.view(np.uint32) != want.view(np.uint32))
                    raise RuntimeError(f"step {pos}: the System's output differs from the twin in {len(bad)} lanes (first {bad[:4].tolist()})")
            stat["twin_equal"] = True
        self.stats.append(stat)
        return outputs, stat

    def logits(self, out):
        """Apply the host vocabulary head to the reduced fixture's hidden state."""
        w = self.w
        hidden = rw.layernorm_wide(out, w.lnf_g, w.lnf_b, GPT2_SMALL.ln_eps, tile=sl.TILE)
        res = np.empty(w.cfg.vocab, np.float32)
        for a in range(0, w.cfg.vocab, 8192):
            res[a:a + 8192] = rw.matvec_tiled(w.wte[a:a + 8192], hidden, tile=sl.TILE)
        return res

    def clear_canary(self):
        # Preserve the system that passed qualification. All KV pages become
        # empty, including the canary's physical slot, before normal admission.
        if self._warm is not None:
            for st, chip in zip(self.stages[:self.layers], self.route):
                for name, (sli, addr, words) in st.externs.items():
                    if POOL_NAME.fullmatch(name):
                        self._warm.write_words(chip, sli, addr, bytes(words * self.p.vlanes))
        self.pool.clear()
        for a in self.twin_k + self.twin_v:
            a.fill(0)

    def reset(self):
        """Empty pools: the words last read back from the System (a resident System is dropped: the next step builds a new one) and the twin's K and V rows."""
        self.pool = {}
        self._warm = None
        for a in self.twin_k + self.twin_v:
            a[...] = 0

    def twin_step(self, x, slot, counts):
        mask = S.mask_of(counts)
        for l in range(self.layers):
            r = sl.twin(x, self.Ps[l], self.twin_k[l], self.twin_v[l], slot, mask, GPT2_SMALL.heads)
            self.twin_k[l], self.twin_v[l] = r["k16"], r["v16"]
            x = r["out"]
        return x
