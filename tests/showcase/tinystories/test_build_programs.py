"""Test the digest-keyed program cache (`program_cache.py`), the bucket build (`build_programs.py`) and the patch table of the
built program.

* A second build with an unchanged digest does nothing; a changed generator source or profile changes the digest.
* The artifacts of a bucket (object, text package, patch table, output regions) are written, read back and run on the ISS
  from the cache files alone.
* The patch-table self-check: the program assembled a second time by `ots-asm` with every site set to another legal
  value differs from the first exactly in the recorded sites; a table with a misplaced site is refused.
* A `StepPlan` of `opentsp_vllm` (read-only import, `step_plan.plan_step`) gives site values equal to `site_values` where they
  must agree and a sink word where both only have to land in the sink, and the program patched with them is bit-exact reference-exact on the
  ISS ('s deferred item).
* The output index decodes the returned words into the logits like the dump of `test_step_batch`.
"""
import multiprocessing
import os
import shutil
import sys
import threading
import time

import numpy as np
import pytest

from opentsp_vllm.models.tinystories import paths as _paths

sys.path.insert(0, os.path.join(_paths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.tinystories import build_programs as BP
from opentsp_vllm.models.tinystories import emit_step_batch as E
from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_vllm import chip as vchip, step_plan as SP
from opentsp_tools import patch
import test_step_batch as SB


@pytest.fixture(scope="module")
def model():
    return PC.Model(2)


@pytest.fixture(scope="module")
def built(model, tmp_path_factory):
    """{bucket: (batch, bucket directory)} of P(1, 1, 64) and P(4, 4, 64) built into one cache of this module."""
    cache = str(tmp_path_factory.mktemp("cache"))
    out = {"cache": cache}
    for bucket in ((1, 1, 64), (4, 4, 64)):
        keep = []
        status, rec = BP.build_bucket(model, bucket, cache, keep=keep)
        assert status == "built"
        out[bucket] = (keep[0], PC.bucket_dir(cache, model, bucket), rec)
    return out


def test_a_second_build_with_an_unchanged_digest_does_nothing(model, built):
    cache = built["cache"]
    path = built[(1, 1, 64)][1]
    before = {f: os.stat(os.path.join(path, f)).st_mtime_ns for f in PC.FILES}
    status, rec = BP.build_bucket(model, (1, 1, 64), cache)
    assert status == "already built" and rec["digest"] == built[(1, 1, 64)][2]["digest"]
    assert rec["build_seconds"] > 1 and rec["cycles"] > 0
    assert {f: os.stat(os.path.join(path, f)).st_mtime_ns for f in PC.FILES} == before


def test_the_digest_covers_the_encoding_the_loader_checks_objects_against():
    """after the specification's encoding.json changed, cached objects were rejected by the loader (OTS-E-ASM-003) because the key did not cover it."""
    rel = [os.path.relpath(f, PC._paths.ROOT) for f in PC.generator_files()]
    assert "opentsp-arch-spec/isa-db/generated/encoding.json" in rel
    assert all(not os.path.isabs(r) and not r.startswith("..") for r in rel), rel


def test_the_digest_covers_the_scheduling_model_and_the_pager_of_the_toolchain():
    """The pager and the scheduling model live in the toolchain (opentsp-sw-toolchain/python/opentsp_tools) and the generators build on them: a change
    there changes the bytes of a bucket program, so every file of the package is part of the key."""
    rel = {os.path.relpath(f, PC._paths.ROOT) for f in PC.generator_files()}
    package = os.path.join(PC._paths.ROOT, "opentsp-sw-toolchain", "python", "opentsp_tools")
    on_disk = {os.path.relpath(os.path.join(package, n), PC._paths.ROOT) for n in os.listdir(package) if n.endswith(".py")}
    assert {"opentsp-sw-toolchain/python/opentsp_tools/pager.py", "opentsp-sw-toolchain/python/opentsp_tools/patch.py"} <= on_disk
    assert on_disk <= rel, sorted(on_disk - rel)


def test_the_digest_follows_the_generator_the_profile_and_the_layers(model, tmp_path, monkeypatch):
    files = PC.generator_files()
    assert model.generator_sha256 == PC.generator_digest(files)
    copy = tmp_path / "emit_step_batch.py"
    src = [f for f in files if f.endswith("emit_step_batch.py")][0]
    shutil.copy(src, copy)
    copy.write_text(copy.read_text() + "\n# a changed generator source\n")
    edited = [str(copy) if f == src else f for f in files]
    changed = PC.Model(2, generator=PC.generator_digest(edited))
    assert changed.digest != model.digest and changed.key != model.key
    monkeypatch.setattr(PC, "profile_digest", lambda profile="ref": "0" * 64)
    other_profile = PC.Model(2)
    assert other_profile.digest != model.digest
    monkeypatch.undo()
    assert PC.Model(8).digest != model.digest and PC.Model(2, seed=2).digest != model.digest
    assert PC.Model(2).digest == model.digest
    assert PC.bucket_digest(model, (1, 1, 64)) != PC.bucket_digest(model, (2, 1, 64))


def test_a_bucket_of_another_digest_is_not_built(model, built):
    other = PC.Model(2, seed=2)
    assert PC.built(built["cache"], model, (1, 1, 64)) and not PC.built(built["cache"], other, (1, 1, 64))


@pytest.mark.parametrize("bucket", [(1, 1, 64), (4, 4, 64)], ids=lambda b: "P%d_%d_%d" % b)
def test_the_patch_table_is_the_chips_and_the_artifacts_are_complete(built, bucket):
    batch, path, rec = built[bucket]
    assert all(os.path.getsize(os.path.join(path, f)) > 0 for f in PC.FILES)
    cols = vchip.patch_columns(os.path.join(path, "bindings.npz"))
    assert set(cols) == {"names", "queue", "unit", "nunits", "lo", "width", "vmin", "vmax", "placeholder", "fixed", "stride", "span",
                         "image_queues", "image_units", "block_units"}
    assert cols["names"] == [x["name"] for x in batch.case["sites"]] and len(cols["names"]) == rec["patch_sites"]
    art = BP.Artifacts(path)
    assert [list(r) for r in art.regions] == rec["outputs"] and len(art.regions) == 8
    # effect-free ISS activity after the last SRAM write: up to 38 cycles since the packed softmax
    assert rec["cycles"] and rec["schedule_cycles"] <= rec["cycles"] <= rec["schedule_cycles"] + 40
    assert rec["self_check"]["sites"] == rec["patch_sites"]
    assert rec["text_words"] > 0 and rec["image_bytes"] > 0 and rec["iss_seconds"] > 0


@pytest.mark.parametrize("bucket", [(1, 1, 64), (4, 4, 64)], ids=lambda b: "P%d_%d_%d" % b)
def test_the_second_assembly_differs_in_exactly_the_recorded_sites(built, bucket):
    batch, path, _ = built[bucket]
    art = BP.Artifacts(path)
    assert BP.self_check(batch, art, BP.ots_asm()) == len(art.paged.table)
    other = BP.other_values(art.paged, batch.kit.p)
    t = art.paged.table
    assert all(other[str(n)] != int(ph) for n, ph in zip(t.names, t.placeholder))      # every site is changed
    assert len(BP.predicted_diff(art.paged, other)) > len(t) // 2


def test_a_misplaced_site_fails_the_self_check(built):
    """RED check of the self-check: a site moved by one unit is a difference the records do not predict."""
    batch, path, _ = built[(1, 1, 64)]
    art = BP.Artifacts(path)
    t = art.paged.table
    i = len(t) // 2
    t.unit[i] += 1                       # a wrong unit: the recorded bits are not the bits that change
    with pytest.raises(BP.BuildError, match="self-check|not the program|does not give"):
        BP.self_check(batch, art, BP.ots_asm())


def without_row(paged, i):
    """The Paged table with row `i` removed (the Table rebuilt, so its index and columns are consistent)."""
    t = paged.table
    keep = [j for j in range(len(t)) if j != i]
    cols = [getattr(t, c)[keep] for c in patch.COLUMNS]
    return patch.Paged(patch.Table(t.names[keep], *cols, stride=t.stride[keep], span=t.span[keep]), paged.image_len, paged.block_units)


def test_a_dropped_site_fails_the_self_check_and_names_it(built):
    """A site the emitter records but the table lost keeps its placeholder in both assemblies; the check must
    compare the table with the emitter's recorded sites."""
    batch, path, _ = built[(1, 1, 64)]
    art = BP.Artifacts(path)
    dropped = str(art.paged.table.names[len(art.paged.table) // 2])
    art.paged = without_row(art.paged, len(art.paged.table) // 2)
    assert len(art.paged.table) == len(batch.case["sites"]) - 1
    with pytest.raises(BP.BuildError, match=r"lacks.*" + dropped.replace(".", r"\.")):
        BP.self_check(batch, art, BP.ots_asm())


def test_an_added_site_fails_the_self_check_and_names_it(built):
    batch, path, _ = built[(1, 1, 64)]
    art = BP.Artifacts(path)
    saved = list(batch.case["sites"])
    batch.case["sites"] = saved[:-1]
    try:
        with pytest.raises(BP.BuildError, match=r"adds.*" + str(art.paged.table.names[-1]).replace(".", r"\.")):
            BP.self_check(batch, art, BP.ots_asm())
    finally:
        batch.case["sites"] = saved


def test_a_table_the_binding_would_refuse_is_refused_here(built):
    batch, path, _ = built[(1, 1, 64)]
    paged, arrays = PC.load_patches(os.path.join(path, "bindings.npz"))
    name = str(paged.table.names[0])
    images = patch.to_image(BP.Artifacts(path).images)
    blocks = BP.Artifacts(path).block_bytes()
    values = {str(n): int(ph) for n, ph in zip(paged.table.names, paged.table.placeholder)}
    patch.apply_paged(images, blocks, paged, values)
    values[name] = int(paged.table.vmax[0]) + 1
    with pytest.raises(patch.PatchError):
        patch.apply_paged(images, blocks, paged, values)


def test_the_cached_program_runs_reference_exact_from_the_files(model, built):
    """The object, the text package and the patch table of the cache, on the ISS, with lanes of every kind."""
    batch, path, _ = built[(4, 4, 64)]
    BP.real_lane_run(BP.tsp_iss(), model, batch, BP.Artifacts(path))


# -- a StepPlan of opentsp_vllm ---------------------------------------------------------------------------------------------------
def plan_of(lanes, bucket_c=64):
    """The `StepPlan` that `plan_step` makes of lanes (token, position, table): each lane is one request, scheduled for one token
    at its position (the chunk reaches the last known token, so every lane has a logit row)."""
    reqs = []
    for i, (tok, pos, table) in enumerate(lanes):
        ids = [1] * pos + [tok]
        reqs.append(SP.PlanRequest(f"r{i}", ids, table, pos, 1))
    return SP.plan_step(reqs, vocab=50257)


PLANS = [[(31337, 0, [5])],
         [(1234, 16, [7, 2]), (31337, 0, [5]), (4242, 21, [7, 2]), (777, 18, [7, 3])],
         [(50000, 40, [3, 6, 1]), (13, 63, [9, 4, 2, 8]), (4242, 5, [11])]]


@pytest.mark.parametrize("lanes", PLANS, ids=lambda p: f"{len(p)}lanes")
def test_a_step_plan_gives_the_site_values_of_the_program(built, lanes):
    """`plan_site_values` (the arrays of `StepPlan.patches` and the geometry of `program.json`) equals `site_values` of the same lanes at
    every site, except the half of the K word that the seven non-owning K pairs write to the sink, which the plan sets to the row's
    half and the generator to 0: both are words of the K sink block."""
    plan = plan_of(lanes)
    T, Ls, C = plan.bucket
    batch, path, rec = built[(4, 4, 64)] if T > 1 else built[(1, 1, 64)]
    assert plan.bucket[0] == batch.bucket[0] and plan.bucket[2] == 64
    art = BP.Artifacts(path)
    names = [str(n) for n in art.paged.table.names]
    got = PC.plan_site_values(rec["geometry"], names, plan)
    tokens, positions, tables = [list(x) for x in zip(*[(ln.token, ln.position, list(ln.block_table)) for ln in plan.real_lanes])]
    want = E.site_values(batch, tokens, positions, tables)
    assert set(got) == set(want)
    sink = {l: batch.geo.kaddr(l, batch.geo.sink) for l in range(batch.layers)}
    for n in names:
        if ".kw." in n and want[n] == sink[int(n.split(".L")[1].split(".")[0])] and got[n] != want[n]:
            assert got[n] - want[n] in (0, 1), n
        else:
            assert got[n] == want[n], n
    # every value is accepted by the table
    patch.apply_paged(patch.to_image(art.images), art.block_bytes(), art.paged, got)


@pytest.mark.parametrize("lanes", PLANS[1:], ids=lambda p: f"{len(p)}lanes")
def test_the_program_patched_from_a_step_plan_is_reference_exact_on_the_iss(model, built, lanes):
    """Compare ISS logits and KV words with the oracle after patching cached images and text from a step plan."""
    plan = plan_of(lanes)
    batch, path, rec = built[(4, 4, 64)]
    art = BP.Artifacts(path)
    names = [str(n) for n in art.paged.table.names]
    values = PC.plan_site_values(rec["geometry"], names, plan)
    real = [(ln.token, ln.position, list(ln.block_table)) for ln in plan.real_lanes]
    BP.real_lane_run(BP.tsp_iss(), model, batch, art, lanes=real, values=values)


def test_a_wrong_plan_value_is_caught_by_the_chip_compare(model, built):
    plan = plan_of(PLANS[1])
    batch, path, rec = built[(4, 4, 64)]
    art = BP.Artifacts(path)
    names = [str(n) for n in art.paged.table.names]
    values = PC.plan_site_values(rec["geometry"], names, plan)
    real = [(ln.token, ln.position, list(ln.block_table)) for ln in plan.real_lanes]
    bad = dict(values)
    lane0 = [n for n in names if n.startswith("t0.L0.kb.b0.")]
    for n in lane0:
        bad[n] = values[n.replace("t0.", "t1.")]       # lane 0 reads lane 1's first block
    with pytest.raises(BP.BuildError):
        BP.real_lane_run(BP.tsp_iss(), model, batch, art, lanes=real, values=bad)


def test_the_output_index_decodes_the_dumped_logit_words(built):
    batch, path, _ = built[(4, 4, 64)]
    art = BP.Artifacts(path)
    rng = np.random.default_rng(3)
    vl = batch.kit.p.vlanes
    words = {(s, a + k): rng.integers(0, 256, vl, dtype=np.uint8).tobytes() for s, a, n in art.regions for k in range(n)}
    flat = b"".join(words[(s, a + k)] for s, a, n in art.regions for k in range(n))
    got = PC.decode_logits(flat, 4, 50257, art.index)
    for r in range(4):
        want = SB.chip_logits(words, batch, r, 50257)
        assert got[r].tobytes() == want.tobytes()


def test_the_cycle_table_collects_the_buckets_of_the_cache(model, built):
    table = PC.write_cycles(built["cache"])
    entry = table[model.digest]
    assert set(entry["buckets"]) == {"P1_1_64", "P4_4_64"} and entry["layers"] == 2
    assert all(b["cycles"] and b["text_words"] for b in entry["buckets"].values())
    assert os.path.isfile(os.path.join(built["cache"], "cycles.json"))
    assert entry["buckets"]["P4_4_64"]["cycles"] > 3 * entry["buckets"]["P1_1_64"]["cycles"]


def test_the_grid_of_a_model_and_its_restrictions():
    assert len(BP.grid(8)) == 18 and len(BP.grid(2)) == 18
    assert len(BP.grid(2, Cs=[64])) == 9 and BP.grid(2, Cs=[128], Ts=[1, 2]) == [(1, 1, 128), (2, 1, 128), (2, 2, 128)]
    assert {PC.parse_bucket(PC.bucket_name(b)) for b in BP.grid(8)} == set(BP.grid(8))
    assert L.serve_config(2).layers == 2


# -- publishing a bucket --------------------------------------------------------------------------------
def fake_bucket(cache, model, bucket, tag):
    d = PC.new_bucket_dir(cache, model, bucket)
    for f in PC.FILES:
        with open(os.path.join(d, f), "w") as fh:
            fh.write("{}")
    np.savez(os.path.join(d, "bindings.npz"), names=np.array(["t0.emb.p0l"]), **PC.output_index())
    PC.write_json(os.path.join(d, "program.json"), {"digest": PC.bucket_digest(model, bucket), "tag": tag})
    return d


class Reader(threading.Thread):
    """Polls the cache for a bucket and counts the polls that find it missing after it was first seen."""

    def __init__(self, cache, model, bucket):
        super().__init__(daemon=True)
        self.args, self.stop, self.seen, self.missing, self.polls = (cache, model, bucket), threading.Event(), False, 0, 0

    def run(self):
        while not self.stop.is_set():
            ok = PC.built(*self.args) is not None
            self.polls += 1
            self.seen = self.seen or ok
            self.missing += self.seen and not ok
            time.sleep(0.002)


def test_a_second_builder_never_makes_the_published_bucket_disappear(model, tmp_path, monkeypatch):
    """Read the published bucket immediately after the second builder deletes a directory.

    Deleting the live directory before its replacement would expose a missing bucket at this exact point.
    """
    cache, bucket = str(tmp_path), (1, 1, 64)
    PC.commit(cache, model, bucket, fake_bucket(cache, model, bucket, "first"))
    second = fake_bucket(cache, model, bucket, "second")
    real_rmtree, observed = shutil.rmtree, []

    def inspect_after_unlink(path, *args, **kwargs):
        real_rmtree(path, *args, **kwargs)
        observed.append(PC.built(cache, model, bucket))

    monkeypatch.setattr(PC.shutil, "rmtree", inspect_after_unlink)
    assert PC.commit(cache, model, bucket, second) == "kept"
    assert [rec and rec["tag"] for rec in observed] == ["first"]
    assert not os.path.exists(second)
    assert PC.built(cache, model, bucket)["tag"] == "first"


def test_an_incomplete_directory_is_replaced_by_one_rename(model, tmp_path):
    cache, bucket = str(tmp_path), (1, 1, 64)
    damaged = PC.bucket_dir(cache, model, bucket)
    os.makedirs(damaged)
    open(os.path.join(damaged, "program.elf"), "w").close()
    assert PC.commit(cache, model, bucket, fake_bucket(cache, model, bucket, "good")) == "published"
    assert PC.built(cache, model, bucket)["tag"] == "good"
    assert sorted(os.listdir(PC.entry_dir(cache, model))) == ["P1_1_64", "P1_1_64.lock", "manifest.json", "weights.safetensors", "weights.safetensors.lock"]


def build_in_child(cache, q):
    m = PC.Model(2)
    q.put(BP.build_bucket(m, (1, 1, 64), cache)[0])


def test_two_concurrent_builders_of_one_bucket_leave_one_complete_bucket(model, tmp_path):
    cache, bucket = str(tmp_path), (1, 1, 64)
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=build_in_child, args=(cache, q)) for _ in range(2)]
    reader = Reader(cache, model, bucket)
    reader.start()
    for pr in procs:
        pr.start()
    out = [q.get(timeout=200) for _ in procs]
    for pr in procs:
        pr.join(30)
    time.sleep(0.05)
    reader.stop.set()
    reader.join()
    assert sorted(out) in (["already built", "built"], ["built", "built"]), out
    assert reader.seen and reader.missing == 0
    rec = PC.built(cache, model, bucket)
    assert rec and rec["cycles"] == 8194 and rec["toolchain"]["ots-asm"]["sha256"]
    assert sorted(os.listdir(PC.entry_dir(cache, model))) == ["P1_1_64", "P1_1_64.lock", "manifest.json", "weights.safetensors", "weights.safetensors.lock"]
    assert "toolchain" in PC.read_json(os.path.join(PC.entry_dir(cache, model), "manifest.json"))


def test_a_bucket_named_twice_is_built_once(tmp_path, monkeypatch):
    started = []

    class Fake:
        def __init__(self, cmd):
            started.append(cmd[-1])
            self.returncode = 0

        def poll(self):
            return 0
    monkeypatch.setattr(BP.subprocess, "Popen", Fake)
    monkeypatch.setattr(PC, "write_cycles", lambda cache: {})
    a = type("A", (), {"layers": 2, "seed": 1, "checkpoint": None, "cache": str(tmp_path), "force": False, "no_run": True, "jobs": 2})
    assert BP.run_parallel(a, [(1, 1, 64), (2, 1, 64), (1, 1, 64)]) == 0
    assert sorted(started) == ["P1_1_64", "P2_1_64"]
    with pytest.raises(SystemExit):
        BP.main(["--layers", "2", "--bucket", "P1_1_64", "--bucket", "P1_1_64", "--force", "--cache", str(tmp_path)])


def test_force_replaces_a_complete_bucket(model, tmp_path):
    """`--force` (needed after a toolchain change) must replace a complete bucket of this digest, not keep it
    (`commit` kept the old directory and `build_bucket` reported "already built")."""
    cache = str(tmp_path)
    status, _ = BP.build_bucket(model, (1, 1, 64), cache, run=False)
    assert status == "built"
    f = os.path.join(PC.bucket_dir(cache, model, (1, 1, 64)), "program.json")
    before = os.stat(f).st_mtime_ns
    status, rec = BP.build_bucket(model, (1, 1, 64), cache, run=False, force=True)
    assert status == "built"
    assert os.stat(f).st_mtime_ns > before
    assert PC.built(cache, model, (1, 1, 64))
    leftovers = [n for n in os.listdir(PC.entry_dir(cache, model)) if ".tmp-" in n or ".old-" in n]
    assert leftovers == []
