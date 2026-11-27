"""Test the `ProgramSet` reader, the site values of a `StepPlan`, the logits decoding, and
`IssChip` over the real cached buckets. The cache is the one of `real.py` (the 2-layer test model; `$OPENTSP_VLLM_CACHE` or a temporary
directory). Run after `just vllm-binding` (`just vllm-binding-test`); a missing binding or cache fails, never skips."""

import gc
import hashlib
import json
import os
import pathlib
import random
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))

import real
from opentsp_vllm import _tsp_iss  # noqa: E402
from opentsp_vllm.chip import BucketProgram, Chip, ChipInfo, IssChip, TspChipError  # noqa: E402
from opentsp_vllm.programs import ProgramSet, ProgramsError  # noqa: E402

PC, BP, G = real.PC, real.BP, real.G
BUCKETS = [(1, 1, 64), (2, 2, 64), (4, 4, 64)]


@pytest.fixture(scope="module")
def cache():
    for b in BUCKETS:
        real.bucket_path(b)
    return real.cache_dir()


@pytest.fixture(scope="module")
def programs(cache):
    return ProgramSet(cache, layers=2, grid_c=(64,), buckets=BUCKETS)


def copy_entry(cache, tmp_path, buckets=((1, 1, 64),)):
    """A cache of its own with the entry of the 2-layer model and only `buckets` (copies: the originals stay as they are)."""
    m = real.model(2)
    src = pathlib.Path(PC.entry_dir(cache, m))
    dst = tmp_path / "cache" / src.name
    dst.mkdir(parents=True)
    shutil.copy(src / "manifest.json", dst / "manifest.json")
    for b in buckets:
        shutil.copytree(src / PC.bucket_name(b), dst / PC.bucket_name(b))
    return tmp_path / "cache", dst


# ---- the binding says what the programs are checked against ----


def test_the_binding_reports_the_specification_version_and_the_profile_digest():
    assert _tsp_iss.SPEC_VERSION == "0.1"
    ref = REPO / "opentsp-arch-spec" / "config" / "ref.json"
    want = hashlib.sha256(ref.read_bytes()).hexdigest()
    assert _tsp_iss.profile_sha256("ref") == want == _tsp_iss.Chip("ref").profile_sha256
    with pytest.raises(ValueError):
        _tsp_iss.profile_sha256("nosuch")


# ---- reading the cache ----


def test_the_program_set_reads_the_entry_of_the_model(programs, cache):
    m, art = real.model(2), real.art((4, 4, 64))
    assert programs.digest == m.digest and programs.layers == 2
    assert programs.buckets() == sorted(BUCKETS)
    assert programs.info == ChipInfo(layers=2, vocab=50257, hidden=64, heads=16, num_blocks=128, grid_c=(64,))
    p = programs.program((4, 4, 64))
    assert isinstance(p, BucketProgram) and p.name == "P4_4_64"
    assert p.outputs == [tuple(r) for r in art.regions]
    assert p.cycles == art.record["cycles"] + 1, "the CYCLE counter reads one more than the ISS's last_cycle"
    assert p.max_cycles == 2 * art.record["cycles"]
    assert programs.program((4, 4, 64)) is p
    with pytest.raises(KeyError):
        programs.program((8, 4, 128))


def test_a_missing_bucket_fails_with_the_command_that_builds_it(cache, tmp_path):
    root, _ = copy_entry(cache, tmp_path)
    with pytest.raises(ProgramsError) as e:
        ProgramSet(root, layers=2, grid_c=(64,))
    msg = str(e.value)
    missing = msg.split("missing: ")[1].split(";")[0].split(", ")
    assert "P2_1_64" in missing and "P8_4_64" in missing and "P1_1_64" not in missing and len(missing) == 8
    assert f"OPENTSP_VLLM_CACHE={root} LAYERS=2 C=64 T=2,4,8 just vllm-programs" in msg
    with pytest.raises(ProgramsError, match=r"OPENTSP_VLLM_CACHE=\S+ LAYERS=2 just vllm-programs"):
        ProgramSet(tmp_path / "empty", layers=2)
    root, _ = copy_entry(cache, tmp_path / "again", [(1, 1, 64), (2, 2, 64)])
    # a cache without the entry of this layer count
    with pytest.raises(ProgramsError, match=r"LAYERS=8 C=64 just vllm-programs"):
        ProgramSet(root, layers=8, grid_c=(64,))


def test_a_bucket_of_another_digest_is_refused(cache, tmp_path):
    root, entry = copy_entry(cache, tmp_path)
    rec = json.loads((entry / "P1_1_64" / "program.json").read_text())
    rec["digest"] = "0" * 64
    (entry / "P1_1_64" / "program.json").write_text(json.dumps(rec))
    with pytest.raises(ProgramsError, match=r"P1_1_64.*digest.*just vllm-programs"):
        ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    (entry / "P1_1_64" / "program.json").write_text(json.dumps({**rec, "digest": PC.bucket_digest(real.model(2), (1, 1, 64))}))
    ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    # a file missing from the bucket directory
    (entry / "P1_1_64" / "bindings.npz").unlink()
    with pytest.raises(ProgramsError, match=r"P1_1_64.*bindings.npz"):
        ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])


def test_an_entry_for_another_specification_or_profile_is_refused(cache, tmp_path):
    root, entry = copy_entry(cache, tmp_path)
    man = json.loads((entry / "manifest.json").read_text())
    for key, value, text in [("spec_version", "0.2", "spec_version 0.2"), ("profile_sha256", "ab" * 32, "profile")]:
        (entry / "manifest.json").write_text(json.dumps({**man, key: value}))
        with pytest.raises(ProgramsError, match=text):
            ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    (entry / "manifest.json").write_text(json.dumps(man))
    assert ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)]).digest == man["digest"]
    ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)], digest=man["digest"])
    with pytest.raises(ProgramsError, match="digest"):
        ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)], digest="1" * 64)


def test_the_cache_root_is_the_option_else_the_environment(cache, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENTSP_VLLM_CACHE", str(cache))
    assert ProgramSet(layers=2, grid_c=(64,), buckets=BUCKETS).root == str(cache)
    monkeypatch.setenv("OPENTSP_VLLM_CACHE", str(tmp_path / "none"))
    with pytest.raises(ProgramsError, match="just vllm-programs"):
        ProgramSet(layers=2, grid_c=(64,), buckets=BUCKETS)
    assert ProgramSet(cache, layers=2, grid_c=(64,), buckets=BUCKETS).root == str(cache)


def test_the_package_digest_is_checked(programs):
    programs.check_package(real.model(2).logical)
    with pytest.raises(ProgramsError, match="weight package"):
        programs.check_package(real.model(2).logical + b"x")


# ---- site values of a StepPlan equal the emitter's ----


@pytest.mark.parametrize("bucket", [(1, 1, 64), (4, 4, 64)])
def test_site_values_of_a_plan_equal_the_emitters(programs, bucket):
    from opentsp_vllm.models.tinystories import emit_step_batch as E

    m = real.model(2)
    batch = E.emit_step_batch(*bucket, layers=2, w=m.w)
    art = real.art(bucket)
    p = programs.program(bucket)
    names = [str(n) for n in art.paged.table.names]
    geo = art.record["geometry"]
    rng = random.Random(4242 + bucket[0])
    kinds = {k: 0 for k in (".emb.p", ".emb.wpe", ".emb.map", ".mk.", ".kb.", ".vb.", ".kw.", ".vw.")}
    for _ in range(12):
        lanes, n_log = real.random_lanes(bucket, rng)
        plan = real.plan_of(bucket, lanes, n_log)
        got = dict(zip(names, np.frombuffer(p.site_values(plan), "<i4").tolist()))
        assert list(got) == names
        want = E.site_values(batch, [t for t, _, _ in lanes], [q for _, q, _ in lanes], [tab[: q // 16 + 1] for _, q, tab in lanes])
        assert set(want) == set(names)
        for n in names:
            if got[n] != want[n]:
                # the one difference of V1G2: the word half of a K write that goes to the sink block (both are words of the sink)
                lay = int(n.split(".L")[1].split(".")[0])
                sink = geo["kaddr"][lay] + geo["kstride"] * geo["sink"]
                assert ".kw." in n and got[n] in (sink, sink + 1) and want[n] in (sink, sink + 1), (n, got[n], want[n])
            for k in kinds:
                kinds[k] += k in n
    assert all(kinds.values()), kinds
    # and the plan's values equal 's `plan_site_values` (the same mapping, ported)
    lanes, n_log = real.random_lanes(bucket, rng)
    plan = real.plan_of(bucket, lanes, n_log)
    ref = PC.plan_site_values(geo, names, plan)
    assert np.frombuffer(p.site_values(plan), "<i4").tolist() == [ref[n] for n in names]


def _block(value, at=1, dtype=None):
    """`value` as context block `at` of lane 0 (K and V tables) and as the block lane 0 writes (position 20 is row 4 of block 1)."""

    def mutate(patches):
        if dtype:
            patches["kblock"], patches["vblock"] = patches["kblock"].astype(dtype), patches["vblock"].astype(dtype)
        patches["kblock"][0, at] = patches["vblock"][0, at] = value
        patches["kwrite"][0, 4, 0] = patches["vwrite"][0, 0] = value

    return mutate


def _position(pos):
    def mutate(patches):
        patches["wpe"][0] = patches["mask"][0] = pos

    return mutate


def _token_row(parts, sub):
    def mutate(patches):
        patches["emb_part"][0] = parts
        patches["emb_map"][0] = sub

    return mutate


def _longer_table(patches):  # one more block than C / 16 = 4
    for kind in ("kblock", "vblock"):
        patches[kind] = np.pad(patches[kind], ((0, 0), (0, 1)), constant_values=7)


def _short_table(patches):
    patches["kblock"][0, 1] = patches["vblock"][0, 1] = 0


def _wrong_type(patches):
    patches["kblock"] = patches["kblock"].astype(np.float64)


def _uint64_beyond_int32(patches):
    patches["emb_map"] = np.array([2**40] + [0] * 3, dtype=np.uint64)


# (id, token, position and block table given to the emitter, the same defect made in the arrays of a plan of P(4, 4, 64) whose lane 0
# is (31337, 20, [5, 6]) and lane 1 (1234, 3, [8]))
INVALID = [
    ("position-C", 31337, 64, [5, 6, 7, 9, 10], _position(64)),
    ("position-minus-1", 31337, -1, [5], _position(-1)),
    ("table-too-short", 31337, 20, [5], _short_table),
    ("table-longer-than-C-over-16", 31337, 20, [5, 6, 7, 9, 10], _longer_table),
    ("null-block-in-the-prefix", 31337, 20, [0, 6], lambda p: p["kblock"].__setitem__((0, 0), 0) or p["vblock"].__setitem__((0, 0), 0)),
    ("block-128-the-sink", 31337, 20, [5, 128], _block(128)),
    ("block-129", 31337, 20, [5, 129], _block(129)),
    ("block-minus-1", 31337, 20, [5, -1], _block(-1)),
    ("block-beyond-int32", 31337, 20, [5, 3 + 2**30], _block(3 + 2**30, dtype=np.int64)),
]


@pytest.fixture(scope="module")
def batch44():
    from opentsp_vllm.models.tinystories import emit_step_batch as E

    return E.emit_step_batch(4, 4, 64, layers=2, w=real.model(2).w)


@pytest.mark.parametrize("name,token,pos,table,mutate", INVALID, ids=[c[0] for c in INVALID])
def test_the_port_refuses_what_the_emitter_refuses(programs, batch44, name, token, pos, table, mutate):
    """Host input that `emit_step_batch.site_values` refuses (a position outside the bucket, a table that does not cover the
    position or is too long, block ids outside 1..127, the null block inside a table, a token outside the vocabulary) is a ValueError of
    the port too (the messages differ), and a value that does not fit int32 is refused, never wrapped."""
    from opentsp_vllm.models.tinystories import emit_step_batch as E

    bucket = (4, 4, 64)
    program = programs.program(bucket)
    valid = [(31337, 20, [5, 6]), (1234, 3, [8])]
    program.site_values(real.plan_of(bucket, valid, 1))  # the valid plan is accepted
    E.site_values(batch44, [31337, 1234], [20, 3], [[5, 6], [8]])  # the twin is accepted by both
    with pytest.raises(ValueError):
        E.site_values(batch44, [token, 1234], [pos, 3], [table, [8]])
    plan = real.plan_of(bucket, valid, 1)
    mutate(plan.patches)
    with pytest.raises(ValueError):
        program.site_values(plan)


def test_other_defects_of_a_plan_are_refused_before_the_run(programs):
    bucket = (4, 4, 64)
    program = programs.program(bucket)
    valid = [(31337, 20, [5, 6]), (1234, 3, [8])]

    def refused(mutate, match):
        plan = real.plan_of(bucket, valid, 1)
        mutate(plan.patches)
        with pytest.raises(ValueError, match=match):
            program.site_values(plan)

    # a token outside the vocabulary: the emitter does not check it (the embedding rows it reads would be someone else's), `plan_step`
    # and the port do
    refused(_token_row([4096, 4096, 1859], 2), "token 50257")
    refused(_token_row([4096, 2171, 4096], -1), "rotation -1")
    refused(_wrong_type, "not integers")
    refused(_uint64_beyond_int32, "do not fit int32")
    refused(lambda p: p["kwrite"].__setitem__((0, 0, 0), 9), "K write")
    refused(lambda p: p["vwrite"].__setitem__((0, 1), 9), "V write")
    refused(lambda p: p["vblock"].__setitem__((0, 0), 9), "V block table")
    refused(lambda p: p["mask"].__setitem__(0, 21), "mask position")
    refused(lambda p: p["emb_part"].__setitem__((1, 1), 3), "one word of one part")
    refused(lambda p: p["wpe"].__setitem__(3, 5), "padding lane")
    refused(lambda p: p.__setitem__("kblock", p["kblock"][:3]), "shape")
    plan = real.plan_of((1, 1, 64), [(5, 0, [1])], 1)
    with pytest.raises(ValueError, match="bucket"):
        program.site_values(plan)  # a plan of another bucket
    # real lanes first: lane 0 turned into padding while lane 1 stays real
    plan = real.plan_of(bucket, valid, 1)
    pad = real.plan_of(bucket, [(1234, 3, [8])], 1).patches
    for kind in plan.patches:
        plan.patches[kind][0] = pad[kind][1]
    with pytest.raises(ValueError, match="first lanes"):
        program.site_values(plan)


def test_a_plan_without_a_patch_kind_the_program_needs_is_refused(programs):
    lanes, n_log = real.random_lanes((1, 1, 64), random.Random(1))
    plan = real.plan_of((1, 1, 64), lanes, n_log)
    del plan.patches["kwrite"]
    with pytest.raises(ValueError, match="kwrite"):
        programs.program((1, 1, 64)).site_values(plan)


# ---- the logits decoding ----


def test_decoded_logits_equal_the_oracle_bit_for_bit_on_a_real_step(programs):
    layers, nblk = 2, 40
    m, geo = real.model(2), real.geometry(2)
    c = _tsp_iss.Chip("ref")
    c.load_package(m.package)
    pool = real.pool(m.cfg, 9, nblk)
    for (s, a), w in BP.pool_words(geo, pool, layers, list(range(nblk))).items():
        c.write_words(s, a, w)
    for bucket, lanes, n_log in [
        ((4, 4, 64), [(31337, 0, [5]), (1234, 16, [7, 2]), (4242, 21, [7, 3]), (50000, 40, [8, 6, 1])], 3),
        ((2, 2, 64), [(777, 5, [12]), (13, 63, [9, 4, 2, 11])], 2),
    ]:
        p = programs.program(bucket)
        d = pathlib.Path(real.art(bucket).path)
        c.load_program(
            p.name, p.elf, text_package=p.text_package, patch_table=p.patch_table, outputs=p.outputs, blocks=p.blocks
        )
        plan = real.plan_of(bucket, lanes, n_log)
        words, _, _ = c.step(p.name, p.site_values(plan), p.max_cycles)
        want = G.step_batch(
            m.w, pool, [t for t, _, _ in lanes], [q for _, q, _ in lanes], [tab[: q // 16 + 1] for _, q, tab in lanes], bucket, n_log
        )
        got = p.decode(words, plan)
        assert got.dtype == np.float32 and got.shape == (n_log, 50257)
        assert got.tobytes() == want.logits.tobytes()
        art = real.art(bucket)
        assert got.tobytes() == PC.decode_logits(words, bucket[1], 50257, art.index)[:n_log].tobytes()
        assert d.is_dir()


# ---- IssChip over the program set ----

STEPS = [
    ((1, 1, 64), [(777, 20, [3, 5])], 1),
    ((4, 4, 64), [(31337, 0, [5]), (1234, 16, [7, 2]), (4242, 21, [7, 3]), (50000, 40, [8, 6, 1])], 4),
    ((2, 2, 64), [(13, 63, [9, 4, 2, 11]), (9, 33, [5, 6, 7])], 2),
    ((1, 1, 64), [(4242, 21, [3, 5])], 1),  # reads the row of the first step
]


def model_dir(tmp_path, package=None):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "weights.safetensors").write_bytes(package or real.model(2).logical)
    return str(d)


def test_iss_chip_over_the_program_set_runs_reference_exact_with_real_cycle_counts(programs, tmp_path):
    c = IssChip("ref", programs=programs, failures=tmp_path / "failures")
    assert isinstance(c, Chip) and c.info == programs.info
    with pytest.raises(ProgramsError, match="weight package"):
        c.load_model(model_dir(tmp_path, real.model(2).logical + b"\0"))
    c.load_model(model_dir(tmp_path))
    c.zero_pool()
    c.warm_up()  # loads the three buckets and runs the smallest on padding lanes: no diagnostics, the recorded cycle count
    assert c.stats()["steps"] == 1 and c.stats()["cycles_last"] == programs.program((1, 1, 64)).cycles
    m = real.model(2)
    pool = G.BlockPool(m.cfg, G.POOL_BLOCKS)
    for k, (bucket, lanes, n_log) in enumerate(STEPS):
        plan = real.plan_of(bucket, lanes, n_log)
        res = c.step(plan)
        want = G.step_batch(
            m.w, pool, [t for t, _, _ in lanes], [q for _, q, _ in lanes], [tab[: q // 16 + 1] for _, q, tab in lanes], bucket, n_log
        )
        assert res.logits.tobytes() == want.logits.tobytes(), (k, bucket)
        assert res.cycles == programs.program(bucket).cycles == c.stats()["cycles_last"], "the binding's cycle count is the recorded one"
        assert res.cycles > 0 and res.diagnostics == ()
    assert c.failure is None
    assert not (tmp_path / "failures").exists(), "no failure, no reproducer"


def test_a_value_the_patch_table_refuses_is_a_failure_with_a_reproducer(programs, tmp_path):
    """Host validation (`SiteMap`) refuses a bad plan before the run; a value that passes it and that the table's range check still
    refuses (here doctored after the port) is the chip's failure, with a reproducer."""
    c = IssChip("ref", programs=programs, failures=tmp_path / "failures")
    c.load_model(model_dir(tmp_path))
    c.zero_pool()
    plan = real.plan_of((1, 1, 64), [(777, 20, [3, 5])], 1)
    p = programs.program((1, 1, 64))
    names, honest = p.site_names, p.site_values
    bad = 5000 * programs.geometry["kstride"] + programs.geometry["kaddr"][0]  # not a block the pool has: the table refuses it

    def doctored(plan):
        v = np.frombuffer(honest(plan), "<i4").copy()
        v[names.index("t0.L0.kb.b0.s44")] = bad
        return v.astype("<i4").tobytes()

    p.site_values = doctored
    try:
        with pytest.raises(TspChipError, match="patch refused") as e:
            c.step(plan)
    finally:
        p.site_values = honest
    man = json.loads((pathlib.Path(e.value.reproducer) / "manifest.json").read_text())
    vals = json.loads((pathlib.Path(e.value.reproducer) / "patch_values.json").read_text())
    assert man["program"] == "P1_1_64" and man["programs_digest"] == programs.digest
    assert bad in vals.values()
    assert len(vals) == len(names)
    assert c.failure and "reproducer" in c.failure


# ---- readers while a builder publishes (no raw OSError) ----


def test_readers_see_a_program_set_or_a_programs_error_while_a_builder_replaces_and_damages_the_files(programs, tmp_path):
    """A writer thread replaces the bucket directory by atomic renames, swaps it for a file, flaps files, truncates the npz files and
    writes broken JSON (always restoring a good copy) while reader threads build `ProgramSet`s and load the program for five seconds:
    every outcome is a success or a `ProgramsError` with the build command, never a raw OSError, JSONDecodeError or zip error."""
    import shutil
    import threading
    import time

    src = pathlib.Path(real.bucket_path((1, 1, 64)))
    root = tmp_path / "cache"
    entry = root / src.parent.name
    good = tmp_path / "good"
    shutil.copytree(src, good)
    entry.mkdir(parents=True)
    shutil.copy(src.parent / "manifest.json", entry / "manifest.json")
    shutil.copytree(good, entry / "P1_1_64")
    bucket, manifest = entry / "P1_1_64", entry / "manifest.json"
    stop = threading.Event()
    rng = random.Random(7)
    counter = iter(range(10**9))

    def swap_in(src_path, dst):
        """Atomic replace of `dst` by a copy of `src_path` (a file or a directory)."""
        tmp = dst.with_name(f".tmp-{next(counter)}-{dst.name}")
        old = dst.with_name(f".old-{next(counter)}-{dst.name}")
        if src_path.is_dir():
            shutil.copytree(src_path, tmp)
        else:
            shutil.copy(src_path, tmp)
        if dst.exists():
            os.rename(dst, old)
        os.rename(tmp, dst)
        shutil.rmtree(old, ignore_errors=True) if old.is_dir() else old.unlink(missing_ok=True)

    def writer():
        files = [f for f in os.listdir(good) if f != "program.json"]
        good_manifest = tmp_path / "good_manifest.json"
        shutil.copy(manifest, good_manifest)
        while not stop.is_set():
            op = rng.randrange(7)
            if op == 0:
                swap_in(good, bucket)
            elif op == 1:  # a file flaps
                f = bucket / rng.choice(files)
                try:
                    os.rename(f, f.with_name(f.name + ".away"))
                    time.sleep(0.0005)
                    os.rename(f.with_name(f.name + ".away"), f)
                except OSError:
                    pass
            elif op == 2:  # a truncated npz, atomically
                f = bucket / "bindings.npz"
                tmp = f.with_name(f.name + ".t")
                tmp.write_bytes((good / f.name).read_bytes()[: rng.randrange(1, 300)])
                os.replace(tmp, f)
                time.sleep(0.0005)
                shutil.copy(good / f.name, tmp)
                os.replace(tmp, f)
            elif op == 3:  # broken JSON in program.json, then the good one
                f = bucket / "program.json"
                tmp = f.with_name(f.name + ".t")
                tmp.write_text('{"digest": "x", "geom')
                os.replace(tmp, f)
                time.sleep(0.0005)
                shutil.copy(good / f.name, tmp)
                os.replace(tmp, f)
            elif op == 4:  # a file where the directory should be, then the directory
                os.rename(bucket, entry / ".away")
                bucket.write_text("not a directory")
                time.sleep(0.0005)
                bucket.unlink()
                os.rename(entry / ".away", bucket)
            elif op == 5:  # the manifest is truncated, then good again
                tmp = manifest.with_name("manifest.json.t")
                tmp.write_bytes(good_manifest.read_bytes()[: rng.randrange(0, 40)])
                os.replace(tmp, manifest)
                time.sleep(0.0005)
                shutil.copy(good_manifest, tmp)
                os.replace(tmp, manifest)
            else:  # the manifest is a list, a directory is missing
                tmp = manifest.with_name("manifest.json.t")
                tmp.write_text("[1, 2]")
                os.replace(tmp, manifest)
                shutil.copy(good_manifest, tmp)
                os.replace(tmp, manifest)

    outcomes, bad, lock = {"ok": 0, "programs_error": 0}, [], threading.Lock()

    def reader():
        while not stop.is_set():
            try:
                ps = ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
                ps.program((1, 1, 64))
                kind = "ok"
            except ProgramsError as e:
                assert "just vllm-programs" in str(e) or "being" in str(e) or "changing" in str(e), str(e)
                kind = "programs_error"
            except BaseException as e:  # noqa: BLE001 - the point of the test
                with lock:
                    bad.append(f"{type(e).__name__}: {e}")
                continue
            with lock:
                outcomes[kind] += 1

    gc.collect()  # a chip of an earlier test in a reference cycle must not be freed (unsendable) by a collection inside a reader thread
    gc.disable()
    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(5)
    stop.set()
    for t in threads:
        t.join(30)
    gc.enable()
    assert not any(t.is_alive() for t in threads)
    assert not bad, (len(bad), sorted(set(bad))[:5])
    assert outcomes["ok"] > 0 and outcomes["programs_error"] > 0, outcomes


# ---- the digest is a build key; messages and skipped manifests ----


def test_a_model_digest_mismatch_names_its_field_and_an_unreadable_manifest_is_reported(cache, tmp_path):
    root, entry = copy_entry(cache, tmp_path)
    bucket = entry / "P1_1_64" / "program.json"
    rec = json.loads(bucket.read_text())
    bucket.write_text(json.dumps({**rec, "model_digest": "f" * 64}))
    with pytest.raises(ProgramsError) as e:
        ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    assert "model_digest ffffffffffff is not the entry's digest" in str(e.value) and "program.json digest" not in str(e.value)
    bucket.write_text(json.dumps({**rec, "digest": "0" * 64}))
    with pytest.raises(ProgramsError, match=r"program.json digest 000000000000 is not .*sha256 of the entry digest and the bucket name"):
        ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    # a second entry whose manifest is unreadable is listed, not silently skipped
    other = root / "deadbeefdeadbeef"
    other.mkdir()
    (other / "manifest.json").write_text('{"digest": "dead')
    shutil.rmtree(entry)
    with pytest.raises(ProgramsError, match=r"deadbeefdeadbeef: manifest.json unreadable \(JSONDecodeError") as e:
        ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    assert "just vllm-programs" in str(e.value)


def test_the_digest_is_a_build_key_and_the_content_is_checked_by_the_recorded_sha256(cache, tmp_path):
    """The digests are a build key; the content is checked at load against `files_sha256` of `program.json`: a
    damaged `program.elf` with the right key is refused with the command that rebuilds it."""
    root, entry = copy_entry(cache, tmp_path)
    assert "build key" in ProgramSet.__doc__ and "not a content check" in ProgramSet.__doc__
    ps = ProgramSet(root, layers=2, grid_c=(64,), buckets=[(1, 1, 64)])
    (entry / "P1_1_64" / "program.elf").write_bytes(b"damaged")
    with pytest.raises(ProgramsError, match="program.elf sha256.*publication.*just vllm-programs"):
        ps.program((1, 1, 64))


def test_iss_chip_refuses_a_program_set_of_another_profile(programs):
    with pytest.raises(ProgramsError, match="profile ref .*this chip is mini"):
        IssChip("mini", programs=programs)
    IssChip("ref", programs=programs)
