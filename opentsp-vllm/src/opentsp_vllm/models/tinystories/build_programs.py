#!/usr/bin/env python3
"""Build and verify TinyStories bucket programs in the digest-keyed serving cache.

    python -m opentsp_vllm.models.tinystories.build_programs --layers 2|8 [--C 64|128] [--T 1,2,4,8] [--L 1,2,4] [--bucket P8_4_128 ...]
                              [--seed N | --checkpoint NAME] [--cache DIR] [--jobs N] [--force] [--no-run]

`emit_step_batch`, the pager, `ots-asm asm`, the text package, the patch table and the output regions are written under
`<cache>/<digest>/P<T>_<L>_<C>/` (`program_cache.py` describes the files). Then, for every bucket built:
* the patch-table self-check: the program is assembled a second time by `ots-asm` with every patch site set to another legal
  value, and the images must differ from the first assembly exactly in the bits of the recorded sites; the artifacts written to
  the cache (object, text package, patch table) must be that first assembly, and applying the table to them must give the
  second;
* one run on the ISS (`tsp-iss run`, from the artifacts) on a padding-only plan: zero diagnostics, the cycle count;
* one run on a plan of real lanes (compat harness, pool preloaded), compared with `paged_reference.step_batch`: the logits of the logit
  rows, the K and V words of a sample of pool blocks, zero diagnostics, the same cycle count as the padding plan.
A second call with an unchanged digest does nothing. `<cache>/cycles.json` collects the figures of every bucket in the cache."""
import argparse
import os
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from opentsp_vllm.models.tinystories import paths as _paths
from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import paged_reference as G
from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_vllm.models.tinystories import kvpool
from opentsp_tools import pager
from opentsp_tools import patch
tsp_as = pager.tsp_as

NB = 12                                    # pool blocks 0..NB - 1 hold random rows (block 0 stays zero)
POOL_SAMPLE = tuple(range(NB)) + (13, 64, 100, 127)       # the blocks whose words the real-lane run compares, and the sink


class BuildError(RuntimeError):
    pass


def tool(env, name, crate):
    path = os.environ.get(env) or os.path.join(_paths.ROOT, "target", "release", name)
    if not os.access(path, os.X_OK):
        subprocess.run(["cargo", "build", "--release", "-q", "-p", crate], cwd=_paths.ROOT, check=True)
    if not os.access(path, os.X_OK):
        raise BuildError(f"{name} is not at {path}: set ${env} or build it with `cargo build --release -p {crate}`")
    return path


def toolchain_record(*paths):
    """Record the producer executable digests and shipped crate versions."""
    root = Path(_paths.ROOT)
    crates = {
        "ots-asm": root / "opentsp-sw-toolchain/crates/ots-asm/Cargo.toml",
        "tsp-iss": root / "opentsp-iss/crates/tsp-iss/Cargo.toml",
    }
    record = {}
    for path in map(Path, paths):
        crate = crates.get(path.name)
        version = None
        if crate is not None:
            with crate.open("rb") as manifest:
                version = tomllib.load(manifest)["package"]["version"]
        record[path.name] = {"version": version, "sha256": PC.sha(path.read_bytes())}
    return record


def ots_asm():
    return tool("OTS_ASM", "ots-asm", "ots-asm")


def tsp_iss():
    return tool("TSP_ISS", "tsp-iss", "tsp-iss")


# -- the self-check of the patch table ------------------------------------------------------------------------------------------
def other_values(sites, p):
    """{site: a legal value that is not the placeholder}: on the grid of the site, preferring the placeholder's bank. `sites` are
    the records of a case file (`Table.records` or the emitter's `case["sites"]`): name, min, max, placeholder, stride, span."""
    if hasattr(sites, "table"):
        sites = sites.table.records()
    out = {}
    for r in sites:
        lo, hi, st, sp, ph = r["min"], r["max"], r.get("stride", 1), r.get("span", 1), r["placeholder"]
        top = lo + ((hi - lo) // st) * st + min(sp - 1, (hi - lo) % st)
        cands = [v for v in (top, lo, lo + 1, hi, lo + st) if lo <= v <= hi and (v - lo) % st < sp and v != ph]
        same = [v for v in cands if p.bank(v) == p.bank(ph)]
        out[r["name"]] = (same or cands or [ph])[0]
    return out


def lower_full(batch, asm, values=None):
    """{queue: np.uint32 units} of the whole text of the program (initial images and text blocks), assembled by `ots-asm`
    with the sites set to `values` (default: the placeholders)."""
    text = batch.kit.view2.full.program(batch.kit.p).text(values)
    with tempfile.NamedTemporaryFile("w", suffix=".ots") as f:
        f.write(text)
        f.flush()
        r = subprocess.run([asm, "asm", "--input", f.name], capture_output=True, text=True)
    if r.returncode:
        raise BuildError(f"ots-asm asm --input: {r.stderr[:300]}")
    return {q: np.array(us, dtype=np.uint32) for q, us in tsp_as.parse_images(r.stdout).items()}


def field_mask(t, i, delta):
    """{unit: bits of that unit} of the field of site `i` for the changed bits `delta` of its value (the units of the whole
    text of the queue; a two-unit instruction keeps bits 63..32 in its first unit)."""
    lo, n, u = int(t.lo[i]), int(t.nunits[i]), int(t.unit[i])
    word = delta << lo
    return {u: word >> 32, u + 1: word & 0xFFFFFFFF} if n == 2 else {u: word & 0xFFFFFFFF}


def predicted_diff(paged, other):
    t = paged.table
    out = {}
    for i, name in enumerate(t.names):
        for u, m in field_mask(t, i, int(t.placeholder[i]) ^ other[str(name)]).items():
            if m:
                key = (int(t.queue[i]), u)
                out[key] = out.get(key, 0) | m
    return out


def observed_diff(a, b):
    out = {}
    for q in sorted(set(a) | set(b)):
        x, y = a.get(q), b.get(q)
        if x is None or y is None or len(x) != len(y):
            raise BuildError(f"queue {q}: the text length depends on the values of the patch sites")
        for u in np.nonzero(x ^ y)[0]:
            out[(q, int(u))] = int(x[u] ^ y[u])
    return out


class Artifacts:
    """The files of a bucket directory read back: images and externs of the object, the text tensors, the patch table, the
    blocks and the output regions."""

    def __init__(self, path, profile="ref", config=None):
        self.path = path
        self.elf = Path(path, "program.elf").read_bytes()
        self.profile, self.images, self.externs = tsp_as.read_object(self.elf, profile, config)
        self.text = tsp_as.object_preloads(self.elf, profile, self.externs, config)
        self.paged, self.arrays = PC.load_patches(os.path.join(path, "bindings.npz"))
        self.record = PC.read_json(os.path.join(path, "program.json"))
        self.blocks = self.record["blocks"]
        self.regions = [tuple(r) for r in self.record["outputs"]]
        with np.load(os.path.join(path, "bindings.npz")) as z:
            self.index = {k: z[k] for k in PC.OUTPUT_KEYS}
        self.profile_name = profile

    def block_bytes(self, tensors=None):
        """{(queue, index): bytearray} of the text blocks (the tensors of `text` unless given)."""
        tensors = tensors or self.text
        return {(b["queue"], b["index"]): bytearray(b"".join(w for n, *_ in b["tensors"] for w in tensors[n][1])) for b in self.blocks}

    def patched(self, values):
        """(images {queue: [units]}, text tensors {name: (dtype, [words])}) with the sites of `values` set."""
        images = patch.to_image(self.images)
        im2, bl2 = patch.apply_paged(images, self.block_bytes(), self.paged, values)
        tensors = dict(self.text)
        vl = len(next(iter(tensors.values()))[1][0])
        for b in self.blocks:
            data = bytes(bl2[(b["queue"], b["index"])])
            pos = 0
            for n, _, _, words in b["tensors"]:
                tensors[n] = (tensors[n][0], [data[pos + k * vl:pos + (k + 1) * vl] for k in range(words)])
                pos += words * vl
        return patch.to_units(im2), tensors

    def whole_text(self, images, blocks):
        """{queue: np.uint32} of the whole text of each queue from initial images and blocks, the padding of blocks dropped."""
        out = {q: list(us) for q, us in images.items()}
        by_q = {}
        for (q, j), data in blocks.items():
            by_q.setdefault(q, []).append((j, data))
        for q, bl in by_q.items():
            for _, data in sorted(bl):
                out[q] += [u for u in np.frombuffer(bytes(data), "<u4").tolist() if u != tsp_as.PADDING]
        return {q: np.array(us, dtype=np.uint32) for q, us in out.items()}


def self_check(batch, art, asm):
    """The double assembly of , and the artifacts against it; raises `BuildError`. Returns the sites checked."""
    p = batch.kit.p
    paged = art.paged
    recorded = batch.case["sites"]                       # what the emitter marked: the table must be exactly this
    have, want_names = [str(n) for n in paged.table.names], [r["name"] for r in recorded]
    if have != want_names:
        lacks, adds = sorted(set(want_names) - set(have)), sorted(set(have) - set(want_names))
        raise BuildError(f"patch table self-check: the table lacks {len(lacks)} recorded sites {lacks[:3]} and adds {len(adds)} "
                         f"{adds[:3]} ({len(have)} rows, {len(want_names)} recorded sites"
                         f"{'' if set(have) != set(want_names) else ', in another order'})")
    other = other_values(recorded, p)
    a, b = lower_full(batch, asm), lower_full(batch, asm, other)
    want, got = predicted_diff(paged, other), observed_diff(a, b)
    if want != got:
        miss = sorted(set(want) - set(got))[:3]
        extra = sorted(set(got) - set(want))[:3]
        raise BuildError(f"patch table self-check: the images differ in other units than the sites record "
                         f"(missing {miss}, unrecorded {extra}, {len(got)} units differ, {len(want)} predicted)")
    first = art.whole_text(art.images, art.block_bytes())
    if any(not np.array_equal(first.get(q), a.get(q)) for q in set(a) | set(first)):
        raise BuildError("the object and the text package in the cache are not the program assembled from its text")
    im2, tx2 = art.patched(other)
    second = art.whole_text(im2, art.block_bytes(tx2))
    if any(not np.array_equal(second.get(q), b.get(q)) for q in set(b) | set(second)):
        raise BuildError("applying the patch table to the artifacts does not give the program assembled with those values")
    return len(paged.table)


# -- running a bucket from its artifacts ----------------------------------------------------------------------------------------
def padding_values(batch):
    T = batch.bucket[0]
    out = {}
    for t in range(T):
        out.update(E.lane_values(batch, t, None, 0, []))
    return out


def write_run_files(art, values, tmp, model_package):
    """The object with the sites of `values` set and one package of the model's tensors and the patched text blocks (an object
    takes one weight package), as files in `tmp`."""
    images, tensors = art.patched(values)
    prog = SimpleNamespace(profile=art.profile, images=images, externs={x["name"]: x for x in art.externs})
    cfg = os.path.join(_paths.SPEC, "config", f"{art.profile_name}.json")
    allt = dict(tsp_as.read_weights(model_package, art.profile_name, cfg))
    allt.update(tensors)
    obj, pkg = os.path.join(tmp, "run.elf"), os.path.join(tmp, "run.safetensors")
    with open(obj, "wb") as f:
        f.write(tsp_as.write_object(prog, cfg))
    with open(pkg, "wb") as f:
        f.write(tsp_as.write_weights(art.profile, [(n, None, w) for n, (_, w) in allt.items()], cfg))
    return obj, pkg


def native_run(iss, art, values, model_package, max_cycles):
    """`tsp-iss run` on the artifacts patched with `values`: {'diags', 'last_cycle', 'ticks', 'seconds', 'wall'}."""
    with tempfile.TemporaryDirectory(prefix="tinystories-bucket-run-") as tmp:
        obj, pkg = write_run_files(art, values, tmp, model_package)
        stats = os.path.join(tmp, "stats.json")
        t0 = time.time()
        r = subprocess.run([iss, "run", "--profile", art.profile_name, "--weights", pkg, "--lint",
                            f"--stats={stats}", "--max-cycles", str(max_cycles), obj], capture_output=True, text=True)
        seconds = time.time() - t0
        if r.returncode not in (0, 3):
            raise BuildError(f"tsp-iss run exited {r.returncode}: {r.stderr[:300]}")
        st = PC.read_json(stats) if os.path.exists(stats) else {}
    diags = [tuple(ln.split()[1:]) for ln in r.stdout.splitlines() if ln.startswith("E ") and not ln.startswith("E - ")]
    return {"diags": diags, "returncode": r.returncode, "last_cycle": st.get("last_cycle"), "ticks": st.get("ticks"),
            "seconds": seconds, "wall": st.get("wall_seconds"), "stderr": r.stderr}


def compat_run(iss, art, values, model_package, pre, dumps, max_cycles, config):
    """The harness run (`tsp-iss <config>`): the patched images, the externs bound from the packages, the words `pre` preloaded and
    the words `dumps` read back: {'diags', 'lints', 'words', 'seconds'}."""
    model = tsp_as.read_weights(model_package, art.profile_name, config)
    images, tensors = art.patched(values)
    allt = dict(model)
    allt.update(tensors)
    text = tsp_as.preload_text(tsp_as.bind_weights(art.externs, allt))
    text += tsp_as.input_text(images)
    text += "".join(f"S {s:x} {a:x} " + " ".join(f"{x:x}" for x in w) + "\n" for (s, a), w in pre.items())
    text += "".join(f"D {s:x} {a:x}\n" for s, a in dumps)
    t0 = time.time()
    r = subprocess.run([iss, config, "--lint", "--max-cycles", str(max_cycles)], input=text.encode(), capture_output=True, timeout=3000)
    seconds = time.time() - t0
    if r.returncode not in (0, 3):
        raise BuildError(f"tsp-iss exited {r.returncode}: {r.stderr.decode()[:300]}")
    out = {"diags": [], "lints": [], "words": {}, "seconds": seconds, "returncode": r.returncode}
    for ln in r.stdout.decode().splitlines():
        f = ln.split()
        if ln.startswith("E ") and f[1] != "-":
            out["diags"].append(tuple(f[1:]))
        elif ln.startswith("T lint"):
            out["lints"].append(tuple(f[2:]))
        elif ln.startswith("W "):
            out["words"][(int(f[1], 16), int(f[2], 16))] = bytes(int(x, 16) for x in f[3:])
    return out


# a plan of real lanes: (token, position, block table); the first L are the logit lanes
LANES64 = [(31337, 0, [5]), (1234, 16, [7, 2]), (4242, 21, [7, 2]), (777, 18, [7, 3]), (50000, 40, [3, 6, 1]),
           (13, 63, [9, 4, 2, 8]), (4242, 5, [11]), (9, 33, [5, 6, 7])]
LANES128 = [(50000, 100, [3, 6, 1, 9, 11, 4, 8])] + LANES64[1:]


def sample_lanes(bucket):
    T, _, C = bucket
    return (LANES128 if C == 128 else LANES64)[:T]


def random_pool(cfg, seed):
    pool = G.BlockPool(cfg, G.POOL_BLOCKS)
    rng = np.random.default_rng(seed)
    pool.k[1:NB] = (rng.standard_normal(pool.k[1:NB].shape) * 0.5).astype(np.float16)
    pool.v[1:NB] = rng.standard_normal(pool.v[1:NB].shape).astype(np.float32)
    return pool


def pool_words(geo, pool, layers, blocks):
    out = {}
    for l in range(layers):
        out.update(kvpool.k_words(geo, pool, l, blocks))
        out.update(kvpool.v_words(geo, pool, l, blocks))
    return out


def real_lane_run(iss, model, batch, art, lanes=None, values=None, seed=100):
    """One step of `lanes` (default the sample lanes of the bucket; (token, position, block table), logit lanes first) against bit-exact reference
    from the artifacts: logits of the logit rows and the pool words of `POOL_SAMPLE` and the sink. `values`: the site values
    to patch (default `site_values` of the lanes). Returns the run record (raises `BuildError` on a difference or diagnostic)."""
    T, Ls, _ = batch.bucket
    geo, layers, cfg = batch.geo, batch.layers, model.cfg
    lanes = lanes or sample_lanes(batch.bucket)
    tokens, positions, tables = [list(x) for x in zip(*lanes)]
    pool_pre, pool_post = random_pool(cfg, seed), random_pool(cfg, seed)
    nl = min(Ls, len(lanes))                 # the logit rows that are real lanes
    want = G.step_batch(model.w, pool_post, tokens, positions, tables, batch.bucket, nl)
    blocks = list(POOL_SAMPLE) + [geo.sink]
    pre = pool_words(geo, pool_pre, layers, blocks)
    after = pool_words(geo, pool_post, layers, blocks)
    out = compat_run(iss, art, values or E.site_values(batch, tokens, positions, tables), model.package, dict(pre),
                     sorted(after) + [(s, a + k) for s, a, n in art.regions for k in range(n)],
                     2 * batch.kit.last, os.path.join(_paths.SPEC, "config", f"{art.profile_name}.json"))
    if out["diags"] or out["lints"] or out["returncode"]:
        raise BuildError(f"real-lane run: diagnostics {out['diags'][:3]} lints {out['lints'][:3]} exit {out['returncode']}")
    sink = {sa for l in range(layers) for sa in kvpool.k_words(geo, pool_post, l, [geo.sink])}
    if len(lanes) < T:
        sink |= {sa for l in range(layers) for sa in kvpool.v_words(geo, pool_post, l, [geo.sink])}
    bad = [(s, hex(a)) for (s, a), word in sorted(after.items()) if (s, a) not in sink and out["words"][(s, a)][:len(word)] != word]
    if bad:
        raise BuildError(f"real-lane run: {len(bad)} pool words differ from bit-exact reference, first {bad[:4]}")
    words = b"".join(out["words"][(s, a + k)] for s, a, n in art.regions for k in range(n))
    got = PC.decode_logits(words, Ls, cfg.vocab, art.index)
    for r in range(nl):
        if got[r].tobytes() != want.logits[r].tobytes():
            raise BuildError(f"real-lane run: the logits of row {r} differ from bit-exact reference")
    return out


# -- one bucket ---------------------------------------------------------------------------------------------------------------
def write_artifacts(model, batch, tmp, asm):
    """The files of `program_cache` for a built `batch` in the directory `tmp`; returns the sizes."""
    p, kit = batch.kit.p, batch.kit
    cfg = os.path.join(_paths.SPEC, "config", f"{model.profile}.json")
    src = os.path.join(tmp, "program.ots")
    with open(src, "w") as f:
        f.write(kit.b.prog.text())
    r = subprocess.run([asm, "asm", "-o", os.path.join(tmp, "program.elf"), src], capture_output=True, text=True)
    if r.returncode:
        raise BuildError(f"ots-asm asm: {r.stderr[:400]}")
    os.remove(src)
    in_model = set(tsp_as.read_weights(model.package, model.profile, cfg))
    vl = p.vlanes
    own = {n: [blob[k:k + vl] for k in range(0, len(blob), vl)] for n, (_, blob) in kit.weights.items() if n not in in_model}
    blocks = pager.tensor_words(kit.blocks)
    if not set(blocks) <= set(own):
        raise BuildError(f"text blocks {sorted(set(blocks) - set(own))[:3]} are not tensors of the program")
    elf = Path(tmp, "program.elf")
    _, images, externs = tsp_as.read_object(elf.read_bytes(), model.profile, cfg)
    program = SimpleNamespace(profile=model.profile, images=images, externs={x["name"]: x for x in externs})
    text = [(n, None, w) for n, w in own.items() if n in blocks]
    literals = [(n, None, w) for n, w in own.items() if n not in blocks]
    elf.write_bytes(tsp_as.write_object(
        program, cfg,
        text=tsp_as.write_weights(model.profile, text, cfg) if text else b"",
        constants=tsp_as.write_weights(model.profile, literals, cfg) if literals else b"",
    ))
    np.savez(os.path.join(tmp, "bindings.npz"), **PC.patch_arrays(kit.located), **PC.output_index())
    return {"blocks": PC.block_records(kit.blocks), "text_words": sum(len(w) for w in blocks.values()), "patch_sites": len(kit.located.table),
            "image_bytes": 4 * sum(len(u) for u in images.values())}


def geometry(batch):
    """What a host needs to turn a step plan into site values without the generator: the words of the embedding parts, `wpe`, the
    masks, the pool geometry."""
    geo, tb, plan, C = batch.geo, batch.tables, batch.plan, batch.bucket[2]
    return {"parts": [{"addr": pt.addr, "zero": pt.zero, "rows": r} for pt, r in zip(tb.parts, batch.sizes.part_rows)],
            "wpe": list(tb.wpe_addrs), "map_base": int(E.k_embed.FIRST_MAP), "mask_base": plan.by_name[f"mk{C}s{L.MASK_SLICES[0]}"].addr,
            "kaddr": [geo.kaddr(l, 0) for l in range(batch.layers)], "kstride": 2 * batch.layers,
            "vaddr": [geo.vaddr(l, 0) for l in range(batch.layers)], "vstride": 16, "sink": geo.sink, "blocks": geo.blocks,
            "vocab": batch.sizes.vocab, "pack": L.PACK, "part_zero_word": 4096}


def build_bucket(model, bucket, cache=None, run=True, force=False, keep=None):
    """Build `bucket` into the cache and run it once; returns `(status, record)`, status "already built" (nothing was done) or
    "built". `keep`: a list that gets the `Batch` appended when the bucket is built."""
    t_all = time.time()
    if not force:
        rec = PC.built(cache, model, bucket)
        if rec:
            return "already built", rec
    asm = ots_asm()
    iss = tsp_iss()
    tmp = PC.new_bucket_dir(cache, model, bucket)
    try:
        T, Ls, C = bucket
        t0 = time.time()
        batch = E.emit_step_batch(T, Ls, C, layers=model.layers, w=model.w)
        if keep is not None:
            keep.append(batch)
        t_emit = time.time() - t0
        t0 = time.time()
        sizes = write_artifacts(model, batch, tmp, asm)
        t_write = time.time() - t0
        rec = {"schema": PC.SCHEMA, "version": PC.VERSION, "chip": 0, "bucket": list(bucket), "digest": PC.bucket_digest(model, bucket),
               "model_digest": model.digest, "layers": model.layers, "generator_sha256": model.generator_sha256,
               "schedule_cycles": batch.kit.last, "emit_seconds": batch.seconds, "geometry": geometry(batch),
               "outputs": [list(r) for r in PC.output_regions(batch.plan, Ls)], **sizes}
        rec["toolchain"] = toolchain_record(asm, iss)
        PC.write_json(os.path.join(tmp, "program.json"), {**rec, "cycles": None, "build_seconds": None, "iss_seconds": None})
        art = Artifacts(tmp, model.profile)
        t0 = time.time()
        checked = self_check(batch, art, asm)
        rec["self_check"] = {"sites": checked, "seconds": time.time() - t0}
        if run:
            first = native_run(iss, art, padding_values(batch), model.package, 2 * batch.kit.last)
            if first["diags"] or first["returncode"] or first["last_cycle"] is None:
                raise BuildError(f"padding run: diagnostics {first['diags'][:3]}, exit {first['returncode']}")
            real = real_lane_run(iss, model, batch, art)
            rec.update(cycles=first["last_cycle"], iss_ticks=first["ticks"], iss_seconds=first["seconds"], iss_run_seconds=first["wall"],
                       iss_real_lane_seconds=real["seconds"])
        else:
            rec.update(cycles=None, iss_seconds=None)
        rec["build_seconds"] = time.time() - t_all
        rec["seconds"] = {**batch.seconds, "emit_total": t_emit, "artifacts": t_write}
        PC.write_json(os.path.join(tmp, "program.json"), rec)
        if PC.commit(cache, model, bucket, tmp, rec["toolchain"], replace=force) == "kept":
            return "already built", PC.built(cache, model, bucket)
    except BaseException:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return "built", rec


def grid(layers, Cs=None, Ts=None, Ls=None):
    return [(T, L_, C) for C in (Cs or E.GRID_C) for T in (Ts or E.GRID_T) for L_ in (Ls or E.GRID_L) if L_ <= T]


def build_programs(layers=8, buckets=None, cache=None, seed=1, checkpoint=None, run=True, force=False, log=print):
    model = PC.Model(layers, seed, checkpoint)
    out = []
    for bucket in buckets or grid(layers):
        t0 = time.time()
        status, rec = build_bucket(model, bucket, cache, run, force)
        log(f"{PC.bucket_name(bucket)}: {status} in {time.time() - t0:.1f} s, cycles {rec.get('cycles')}, digest {model.key}")
        out.append((bucket, status, rec))
    PC.write_cycles(cache)
    return model, out


def ints(s):
    return [int(x) for x in s.split(",")] if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=int(os.environ.get("LAYERS") or 8), choices=(2, 8))
    ap.add_argument("--C", type=ints, default=ints(os.environ.get("C", "")))
    ap.add_argument("--T", type=ints, default=ints(os.environ.get("T", "")))
    ap.add_argument("--L", type=ints, default=None)
    ap.add_argument("--bucket", action="append", help="one bucket, P<T>_<L>_<C> (repeatable); overrides --C/--T/--L")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", nargs="?", const="calum/tinystories-gpt2-3M")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--jobs", type=int, default=1, help="buckets built at once, one process each")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="build and check the patch table, do not run on the ISS")
    ap.add_argument("--summary", action="store_true", help="only rewrite <cache>/cycles.json from the buckets on disk")
    a = ap.parse_args(argv)
    if a.summary:
        table = PC.write_cycles(a.cache)
        print(f"{os.path.join(PC.cache_root(a.cache), 'cycles.json')}: " +
              ", ".join(f"{d[:16]} {len(e['buckets'])} buckets" for d, e in table.items()))
        return 0
    buckets = [PC.parse_bucket(b) for b in a.bucket] if a.bucket else grid(a.layers, a.C, a.T, a.L)
    if len(set(buckets)) != len(buckets):
        if a.force:
            ap.error("a bucket is named twice and --force would build it twice at once")
        buckets = list(dict.fromkeys(buckets))
    for b in buckets:
        E.check_bucket(*b)
    if a.jobs > 1 and len(buckets) > 1:
        return run_parallel(a, buckets)
    model, out = build_programs(a.layers, buckets, a.cache, a.seed, a.checkpoint, not a.no_run, a.force)
    print(f"{sum(s == 'built' for _, s, _ in out)} built, {sum(s != 'built' for _, s, _ in out)} already built; "
          f"cache {PC.entry_dir(a.cache, model)}")
    return 0


def run_parallel(a, buckets):
    """The buckets in `a.jobs` child processes at a time (each builds one bucket)."""
    base = [sys.executable, os.path.abspath(__file__), "--layers", str(a.layers), "--seed", str(a.seed)]
    if a.checkpoint:
        base += ["--checkpoint", a.checkpoint]
    if a.cache:
        base += ["--cache", a.cache]
    if a.force:
        base.append("--force")
    if a.no_run:
        base.append("--no-run")
    running, failed = [], []
    todo = sorted(dict.fromkeys(buckets), key=lambda b: -b[0] * b[2])      # each bucket once (a duplicate would race)         # the largest first: they set the wall time
    while todo or running:
        while todo and len(running) < a.jobs:
            b = todo.pop(0)
            running.append((b, subprocess.Popen(base + ["--bucket", PC.bucket_name(b)])))
        time.sleep(0.5)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                if item[1].returncode:
                    failed.append(PC.bucket_name(item[0]))
    PC.write_cycles(a.cache)
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
