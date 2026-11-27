"""Tests of the memory-layout planner, the SRAM images, and the `.safetensors` package ."""
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

from opentsp_vllm.models.tinystories import paths as _paths

from opentsp_vllm.models.tinystories import layout as L
from opentsp_vllm.models.tinystories.layout import Region
from opentsp_vllm.models.tinystories import weights as W
from opentsp_vllm.models.tinystories import export
from opentsp_tools import refasm
tsp_as = refasm.load()
from opentsp_vllm.models.tinystories.config import TINYSTORIES

CFG = TINYSTORIES


@pytest.fixture(scope="module")
def built():
    w = W.synthetic(CFG, seed=1)
    prof, plan, tensors, objects = export.build(CFG, w)
    return w, prof, plan, tensors, objects


def test_planner_rejects_an_odd_conflict_cycle():
    pl = L.Planner(L.load_profile("ref"))
    for n in "abc":
        pl.add(n, "act", (0,), 1, rw=True)
    pl.conflict("a", "b")
    pl.conflict("b", "c")
    pl.conflict("c", "a")
    with pytest.raises(ValueError, match="bipartite"):
        pl.place()


def test_planner_bank_rule_and_isolated_defaults():
    pl = L.Planner(L.load_profile("ref"))
    pl.add("ro", "weight", (3,), 10, rw=False)
    pl.add("rw", "act", (3,), 10, rw=True)
    pl.add("other", "act", (4,), 5, rw=True)       # a different slice: no constraint, written -> bank 1
    pl.conflict("ro", "rw")
    pl.place()
    assert (pl.by_name["ro"].bank, pl.by_name["rw"].bank, pl.by_name["other"].bank) == (0, 1, 1)
    assert pl.by_name["ro"].addr == 0 and pl.by_name["rw"].addr == 4096


def test_planner_region_must_fit_its_bank():
    pl = L.Planner(L.load_profile("ref"))
    pl.add("big", "weight", (0,), 4097, rw=False)
    with pytest.raises(ValueError, match="does not fit"):
        pl.place()


def test_plan_covers_every_tensor_and_respects_the_profile(built):
    w, prof, plan, tensors, objects = built
    names = [r.name for r in plan.regions]
    assert len(names) == len(set(names))
    u = plan.usage()
    assert u["max_words_in_a_slice"] < prof.words_per_slice // 4          # design: well under 8192
    # per layer: 16 west slices x 48 words and 16 east slices x (32 weight + 8 K cache) words
    for l in range(CFG.layers):
        for s in L.WEST_W:
            assert plan.by_name[f"w{l}s{s:02d}"].words == 48
        for s in L.EAST_W:
            assert plan.by_name[f"w{l}s{s:02d}"].words == 32
        assert plan.by_name[f"kc{l}"].words == 8 and plan.by_name[f"vc{l}"].words == 64
    assert u["per_slice"]["0"]["bank0"] == 8 * 48 + L.KFIX_WORDS      # fixed words 0x60..0x70 sit in the gap
    assert plan.by_name["w0s00"].addr == 0 and plan.by_name["w1s00"].addr == 0x80
    assert all(plan.by_name[f"w{l}s{s:02d}"].addr >= 0x80 for l in range(1, CFG.layers) for s in L.WEST_W)
    assert u["per_slice"]["44"]["bank0"] == 8 * 32 and u["per_slice"]["44"]["bank1"] == 8 * 8
    assert u["per_slice"]["20"]["bank1"] == 8 * 64                          # 512 V-cache words per slice
    n_exported = sum(1 for r in plan.regions if r.exported)
    assert n_exported == 8 * 32 + 4 + 4


def test_bank_rule_holds_for_every_concurrent_pair(built):
    w, prof, plan, tensors, objects = built
    for a, b in plan.conflicts:
        ra, rb = plan.by_name[a], plan.by_name[b]
        if set(ra.slices) & set(rb.slices):
            assert prof.bank_of(ra.addr) != prof.bank_of(rb.addr), (a, b)
    # weights are read-only and share slices with the K cache: weights in bank 0, the cache in bank 1
    assert all(plan.by_name[f"kc{l}"].bank == 1 for l in range(CFG.layers))
    assert all(r.bank == 0 for r in plan.regions if r.kind in ("weight", "const"))
    # the residual buffers alternate by layer, so both get the same bank (neither is read while the other is written
    # in one slice), and the pipeline stages around them use the other bank
    b = lambda n: plan.by_name["a_" + n].bank
    assert b("x0") == b("x1") != b("xn") == b("h1") == b("t")


def test_regions_do_not_overlap_and_externs_are_legal(built):
    w, prof, plan, tensors, objects = built
    plan.validate()
    seen = {}
    for r in plan.regions:
        if not r.exported:
            continue
        assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,14}", r.name)
        assert r.addr + r.words <= prof.words_per_slice and r.slices[0] < prof.slices
        seen.setdefault(r.slices[0], []).append((r.addr, r.addr + r.words))
    for s, spans in seen.items():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), s


def test_matrix_install_layout_roundtrip_and_planes():
    rng = np.random.default_rng(1)
    M = (rng.standard_normal((64, 64)) * 0.1).astype(np.float16)
    per = L.matrix_words(M, 320)
    assert len(per) == 16 and all(len(p) == 8 for p in per)
    # row 9: r mod 8 = 1 -> slices 2 (low plane) and 3 (high plane), word 9 div 8 = 1
    lo = np.frombuffer(per[2][1], dtype=np.uint8)
    hi = np.frombuffer(per[3][1], dtype=np.uint8)
    row = np.stack([lo[:64], hi[:64]], axis=1).reshape(-1).view(np.float16)
    assert np.array_equal(row, M[9]) and not lo[64:].any() and not hi[64:].any()
    assert np.array_equal(L.decode_matrix_words(per, 64, 64), M)


def test_f32_vector_planes():
    v = np.array([1.0, -2.5, 3.1e-5, 1e30], dtype=np.float32)
    p = L.f32_planes(v, 320)
    assert len(p) == 4 and all(len(x) == 320 for x in p)
    back = np.stack([np.frombuffer(x, dtype=np.uint8)[:4] for x in p], axis=1).reshape(-1).view(np.float32)
    assert np.array_equal(back.view(np.uint32), v.view(np.uint32))


def test_widened_half_planes_keep_sign_and_payload_and_quiet_signaling_nans():
    half = np.array([0x7d55, 0xfd55, 0xfe01, 0x8000, 0x0001, 0x7c00], np.uint16)
    planes = L.f32_planes(half.view(np.float16), len(half))
    words = np.array([list(p) for p in planes], np.uint8).T.copy().view(np.uint32).ravel()
    assert words.tolist() == [0x7feaa000, 0xffeaa000, 0xffc02000, 0x80000000, 0x33800000, 0x7f800000]


def test_package_roundtrip_through_the_reference_tools(built, tmp_path):
    w, prof, plan, tensors, objects = built
    pkg = export.package_bytes(prof, plan, tensors)
    read = tsp_as.read_weights(pkg, "ref")
    assert set(read) == {r.name for r in plan.regions if r.exported}
    externs = [{"name": r.name, "slice": r.slices[0], "addr": r.addr, "size": r.words} for r in plan.regions if r.exported]
    pre = tsp_as.bind_weights(externs, read)          # the SRAM words the loader writes before cycle 0
    sram = {(s, a): word for s, a, word in pre}
    assert len(sram) == len(pre)                       # no two externs write one word
    # decode every matrix of every layer from the SRAM words and compare with the weights
    for l in (0, 3, 7):
        lw = w.layers[l]
        want = {"wq": lw["wq"], "wv": lw["wv"], "wo": lw["wo"], "wk": lw["wk"], "up0": lw["up_w"][:128], "up1": lw["up_w"][128:],
                "dn0": lw["dn_w"][:, :128], "dn1": lw["dn_w"][:, 128:]}
        for name, M in want.items():
            o = objects[f"L{l}.{name}"]
            per = [[sram[(s, a + k)] for k in range(o["words_per_slice"])] for s, a in zip(o["slices"], o["addr"])]
            assert np.array_equal(L.decode_matrix_words(per, o["rows"], o["cols"]), M), (l, name)
    for name, lanes in L.const_vectors(CFG):
        o = objects[name]
        planes = [np.frombuffer(sram[(s, a)], dtype=np.uint8)[:lanes] for s, a in zip(o["slices"], o["addr"])]
        v = np.ascontiguousarray(np.stack(planes, axis=1)).reshape(-1).view(np.float32)
        want = L.const_arrays(CFG, w)[name]
        assert np.array_equal(v.view(np.uint32), want.view(np.uint32)), name
    # one-hot vectors and head masks
    o = objects["onehot"]
    for s in (0, 17, 63):
        hi = np.frombuffer(sram[(o["slices"][1], o["addr"][1] + s)], dtype=np.uint8)
        assert hi[s] == 0x3C and hi.sum() == 0x3C
    assert not np.frombuffer(sram[(o["slices"][0], o["addr"][0])], dtype=np.uint8).any()
    o = objects["headmask"]
    hi = np.frombuffer(sram[(o["slices"][1], o["addr"][1] + 5)], dtype=np.uint8)
    assert (np.flatnonzero(hi) == np.arange(20, 24)).all() and (hi[20:24] == 0x3C).all()


def test_objects_table_is_consistent_with_regions(built):
    w, prof, plan, tensors, objects = built
    o = objects["L3.kcache"]
    assert o["slices"] == list(range(44, 60)) and o["addr"][0] == plan.by_name["kc3"].addr and o["current_slot"] == 63
    o = objects["L5.vcache"]
    assert o["slices"] == [20, 21, 22, 23] and o["words_per_slice"] == 64
    # the matrix objects of one tensor are contiguous inside it
    q, v, wo = (objects[f"L2.{n}"] for n in ("wq", "wv", "wo"))
    assert v["addr"][0] == q["addr"][0] + 8 and wo["addr"][0] == v["addr"][0] + 8
    assert objects["L2.up0"]["addr"][0] == q["addr"][0] + 24 and objects["L2.dn0"]["addr"][0] == q["addr"][0] + 40


def test_export_cli_wpack_parity_and_determinism(tmp_path):
    out1, out2 = tmp_path / "a", tmp_path / "b"
    env = dict(os.environ)
    for out in (out1, out2):
        r = subprocess.run([sys.executable, os.path.join(_paths.HERE, "export.py"), "--synthetic", "1", "--out", str(out), "--raw"],
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
    p1, p2 = (open(o / "model.safetensors", "rb").read() for o in (out1, out2))
    assert p1 == p2
    m = json.load(open(out1 / "layout.json"))
    assert m["schema"] == "opentsp.showcase-layout" and m["package_sha256"] == hashlib.sha256(p1).hexdigest()
    assert len(m["externs"]) == 264 and m["usage"]["max_words_in_a_slice"] > 0
    # the same package from the toolchain's own `wpack` (raw files, tensor order of the manifest)
    args = [f"{e['name']}={e['dtype']}:{out1 / 'raw' / (e['name'] + '.bin')}" for e in m["externs"]]
    r = subprocess.run([sys.executable, os.path.join(_paths.SPEC, "tools", "tsp_as.py"), "wpack", "--profile", "ref",
                        "-o", str(tmp_path / "wpack.safetensors")] + args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert open(tmp_path / "wpack.safetensors", "rb").read() == p1


def test_synthetic_package_is_pinned():
    w = W.synthetic(CFG, seed=1)
    assert w.digest() == PINNED_WEIGHTS_SHA
    prof, plan, tensors, objects = export.build(CFG, w)
    assert hashlib.sha256(export.package_bytes(prof, plan, tensors)).hexdigest() == PINNED_PACKAGE_SHA


PINNED_WEIGHTS_SHA = "c0ff39c8c9b300dc56d60a885a52b291a9f81a544f14f68fe8e9d26111f542e8"
PINNED_PACKAGE_SHA = "dd2d52234f111d3c0b6782ab9eea107755c1fdd4e666e388438efe0b183da7e0"


def test_forty_eight_head_chunks_are_placed_without_a_bank_or_overlap_violation():
    """The largest head the step emitter accepts (emit_step.MAX_HEAD_CHUNKS): 24 chunks per plane, 480 words in each of
    the 32 head slices, all in bank 0 behind the layers' weights, and validate holds (no overlap, inside one bank)."""
    from opentsp_vllm.models.tinystories import emit_step as es
    prof = L.load_profile("ref")
    plan = L.plan_tinystories(CFG, prof, es.MAX_HEAD_CHUNKS)
    plan.validate()
    head = [r for r in plan.regions if r.kind == "head"]
    assert len(head) == 32 and {r.words for r in head} == {es.MAX_HEAD_CHUNKS // 2 * es.NLW}
    assert all(r.bank == 0 for r in head)
    assert max(r.addr + r.words for r in head) == 0x3b0


def test_the_head_never_overlaps_the_kernels_reduce_scratch():
    """A head of 100 chunks failed at word 0x15e6 of slice 40. The cause was in the layout: the
    reduce-and-broadcast kernel (emit_block `scratch`) uses fixed words (bank 0 of the east slices 44..47 from 0x400,
    bank 1 of the same slices and of west slices 0..3 from 0x1400) that the planner did not know, and the east head
    table (0x100 + 1 000 words) covered the first of them. The plan now holds them as fixed regions and places
    around them."""
    prof = L.load_profile("ref")
    plan = L.plan_tinystories(CFG, prof, 100)
    plan.validate()
    scr = [r for r in plan.regions if r.name.startswith("rbs")]
    assert {(r.slices, r.addr, r.words) for r in scr} == {
        (L.RB_EAST, L.RB_BASE, L.RB_WORDS), (L.RB_EAST, prof.bank_words + L.RB_BASE, L.RB_WORDS),
        (L.RB_WEST, prof.bank_words + L.RB_BASE, L.RB_BANK1_WORDS)}
    hd44 = plan.by_name["hd44"]
    assert hd44.addr >= L.RB_BASE + L.RB_WORDS                      # moved past the scratch, still in bank 0
    assert hd44.addr + hd44.words <= prof.bank_words
    assert plan.by_name["hd0"].bank == hd44.bank == 0


def test_planted_overlap_with_a_fixed_region_fails_validation():
    pl = L.Planner(L.load_profile("ref"))
    pl.add("fixed", "reserved", (3,), 16, rw=False, addr=0x40)
    pl.add("w", "weight", (3,), 8, rw=False)
    pl.place()
    assert pl.by_name["w"].addr == 0 and pl.by_name["fixed"].addr == 0x40
    pl.by_name["w"].addr = 0x48                                    # planted: inside the fixed region
    with pytest.raises(AssertionError):
        pl.validate()


def test_a_head_of_100_chunks_runs_reference_exact_on_the_iss(monkeypatch):
    """The failing case of the head of 100 chunks on both ISS modes, now with the scratch known to the planner."""
    from opentsp_vllm.models.tinystories import emit_step as es
    import case_runner as runner_util
    from opentsp_tools import profile
    monkeypatch.setattr(es, "MAX_HEAD_CHUNKS", 100)
    text, case = es.build(profile.load("ref"), 5, head_chunks=100)
    assert case["golden"]["head_rows"] == 16000
    rc, out = runner_util.run_on_iss(text, case)
    assert rc == 0, out


# -- : the layout of the batched, paged, whole-model step ("serve" mode) ----------
SERVE_LAYERS = (8, 2)


@pytest.fixture(scope="module", params=SERVE_LAYERS, ids=lambda n: f"l{n}")
def serve(request):
    cfg = L.serve_config(request.param)
    prof = L.load_profile("ref")
    plan = L.plan(prof, mode="serve", layers=request.param)
    return cfg, prof, plan, L.report(plan, cfg)


def test_the_default_mode_is_the_step_layout():
    prof = L.load_profile("ref")
    a, b = L.plan(prof), L.plan_tinystories(CFG, prof)
    assert [(r.name, r.addr, r.words) for r in a.regions] == [(r.name, r.addr, r.words) for r in b.regions]
    with pytest.raises(ValueError):
        L.plan(prof, mode="nope")
    with pytest.raises(ValueError):
        L.plan(prof, mode="serve", layers=3)


def test_serve_regions_follow_the_slice_assignments_of_the_design(serve):
    cfg, prof, plan, rep = serve
    n, by = cfg.layers, plan.by_name
    # layer weights and the packed head: west 0..15 and east 44..59, bank 0
    assert all(by[f"w{l}s{s:02d}"].bank == 0 for l in range(n) for s in L.WEST_W + L.EAST_W)
    assert {by[f"hd{s}"].words for s in L.WEST_W} == {32 * L.NLW_HEAD} and {by[f"hd{s}"].words for s in L.EAST_W} == {31 * L.NLW_HEAD}
    assert all(by[f"hd{s}"].bank == 0 and by[f"hd{s}"].exported for s in L.WEST_W + L.EAST_W)
    # embedding: three parts of 4 096, 4 096 and 1 860 packed rows on the slice pairs 36/37, 38/39, 40/41, bank 0;
    # the zero word of each part in bank 1 of the same slice
    for part, (rows, pair) in enumerate(((4096, (36, 37)), (4096, (38, 39)), (1860, (40, 41)))):
        for s in pair:
            r = by[f"em{s}"]
            assert (r.slices, r.words, r.bank) == ((s,), rows, 0) and r.exported
            z = by[f"ez{s}"]
            assert (z.slices, z.words, z.bank, z.exported) == ((s,), 1, 1, True)
    # wpe: one f32 plane per slice 40..43, 128 positions, bank 1
    assert all((by[f"wp{s}"].words, by[f"wp{s}"].bank) == (128, 1) for s in range(40, 44))
    # mask tables for C = 64 and 128: one word per position and plane on 76..79
    assert all((by[f"mk64s{s}"].words, by[f"mk128s{s}"].words) == (64, 128) for s in range(76, 80))
    # pools: K in bank 1 of the 16 east slices, 129 blocks (block 0 the zero block, block 128 the sink) of 2 words per layer;
    # V of layer l in the quad of l div 2, bank l mod 2, 16 words per block
    kp = by["kp"]
    assert (kp.slices, kp.bank, kp.words) == (L.EAST_W, 1, 129 * 2 * n)
    for l in range(n):
        v = by[f"vp{l}"]
        assert (v.slices, v.bank, v.words) == (tuple(range(20 + 4 * (l // 2), 24 + 4 * (l // 2))), l % 2, 129 * 16)
    # logits for L <= 4 rows: 4 x 63 x 5 vectors on 72..75
    assert (by["lg"].slices, by["lg"].words, by["lg"].rw) == (tuple(range(72, 76)), 4 * 63 * 5, True)
    # the east plane's logit quad (L = 4 rows page), same layout as `lg`, on the slices of the masks
    assert (by["lg2"].slices, by["lg2"].words, by["lg2"].rw, by["lg2"].bank) == (L.LOGIT_SLICES_EAST, 4 * 63 * 5, True, 0)
    # constants (with the splats the Flow reads), one-hot and head masks, text and the Flow's lane arenas
    assert by["cs16"].words == len(L.const_vectors(cfg, serve=True)) and by["ax64"].words == 128 and by["ax66"].words == cfg.heads
    assert (by["fhm0"].words, by["fhm1"].words, by["foh12"].words, by["foh13"].words) == (16, 16, 128, 128)
    assert len([r for r in plan.regions if r.kind == "act"]) == L.LANES_MAX * len(L.FLOW_ARENA)
    assert all(r.words == sum(L.FLOW_ARENA[(h, q)].values()) and r.bank == 1 for (h, q) in L.FLOW_ARENA
               for r in [plan.by_name[L.flow_arena_name(t, h, q)] for t in range(L.LANES_MAX)])
    assert all(r.kind == "text" and r.rw for r in plan.regions if r.name.startswith("tx"))
    assert {r.bank for r in plan.regions if r.name.startswith("tx") and 44 <= r.slices[0] <= 59} == {0}


def test_serve_layout_has_no_oversized_crossing_or_overlapping_region(serve):
    cfg, prof, plan, rep = serve
    plan.validate()
    L.check_report(rep)                           # the checker artifact's own rules, independent of the planner
    assert rep["max_bank_words"] <= prof.bank_words
    # the pools lie in one bank each, so a patched access has a known bank 
    assert len({r.bank for r in plan.regions if r.kind == "kpool"}) == 1
    # every conflict pair between a region and the K pool (weights stream in the slices that the pool is written in)
    for a, b in plan.conflicts:
        ra, rb = plan.by_name[a], plan.by_name[b]
        if set(ra.slices) & set(rb.slices):
            assert ra.bank != rb.bank, (a, b)


def _plant(rep, **kw):
    import copy
    bad = copy.deepcopy(rep)
    bad["regions"].append(dict(name="planted", slices=[5], bank=0, rw=False, kind="weight", **kw))
    return bad


def test_checker_controls_oversized_crossing_and_overlapping_regions_fail(serve):
    cfg, prof, plan, rep = serve
    with pytest.raises(L.LayoutError, match="exceeds"):
        L.check_report(_plant(rep, addr=0, words=prof.bank_words + 1))
    with pytest.raises(L.LayoutError, match="crosses"):
        L.check_report(_plant(rep, addr=prof.bank_words - 4, words=8))
    first = next(r for r in rep["regions"] if 5 in r["slices"] and r["bank"] == 0)
    with pytest.raises(L.LayoutError, match="overlap"):
        L.check_report(_plant(rep, addr=first["addr"], words=first["words"]))
    L.check_report(rep)                           # and the unplanted report is accepted


def test_the_report_matches_the_per_bank_table_of_the_design(serve):
    """ as revised in : every row and bank of the table, the planned occupancy without exclusions,
    within 10 %; and the total."""
    cfg, prof, plan, rep = serve
    rows = {r["row"]: r for r in rep["design_rows"]}
    assert len(rows) == 11
    assert L.check_design_rows(rep)
    assert rep["total_words"] == plan_total(plan)
    if cfg.layers == 8:      # the table of the , bank by bank (the numbers pinned from the planner)
        # The Flow's tables and lane arenas are planned (the one-hot table in the west, eight lanes of vectors on the
        # east quads 4, 5, 7 and 8), the east logit quad is on 76..79, and text uses each slice's full-grid maximum.
        want = {"west 0..15": (1232, 900), "east 44..59": (1862, 2352), "V pool 20..35": (2064, 2194),
                "embedding parts 0, 1: 36..39": (4096, 137), "part 2, wpe: 40..41": (1860, 265),
                "constants 16..19": (117, 183)}
        for name, (b0, b1) in want.items():
            assert (rows[name]["planned"]["bank0"], rows[name]["planned"]["bank1"]) == (b0, b1), name
        assert {n: (r["planned"].get("bank0"), r["planned"]["bank1"]) for n, r in rows.items() if "arena" in n or "logits" in n} == {
            "Flow arena, east quad 4: 60..63": (None, 744), "one-hot table, Flow arena, east quad 5: 64..67": (128, 735),
            "logits, Flow arena, east quad 7: 72..75": (1260, 487),
            "masks, east logits, Flow arena, east quad 8: 76..79": (1452, 338)}
        assert rows["wpe: 42..43"]["planned"] == {"bank1": 276} and rep["total_words"] == 192117
        assert sum(L.text_words(8, s) for s in L.TEXT_WORDS_T8[8]) == 18985 and sum(L.TEXT_WORDS_T8[8].values()) == 17226
    else:
        assert rows["west 0..15"]["planned"] == {"bank0": 816 + L.CTX_MAX, "bank1": 288 + L.text_words(2, 0)}
        assert rows["east 44..59"]["planned"]["bank1"] == 288 + 129 * 4


def plan_total(plan):
    return sum(r.words * len(r.slices) for r in plan.regions)


def test_a_planted_deviation_in_a_row_is_caught(serve):
    """The control of the table test: a row 12 % over the table fails the rule, 5 % under passes."""
    import copy
    cfg, prof, plan, rep = serve
    bad = copy.deepcopy(rep)
    row = bad["design_rows"][0]
    row["planned"]["bank0"] = int(row["design"]["bank0"] * 1.12)
    with pytest.raises(L.LayoutError, match="west 0..15"):
        L.check_design_rows(bad)
    row["planned"]["bank0"] = int(row["design"]["bank0"] * 0.95)
    assert L.check_design_rows(bad)


def test_the_text_tables_cover_the_slices_of_the_plan():
    """`tx<slice>` regions cover each slice's measured maximum over the bucket grid
    (re-derived by `test_step_buckets.py`), with the margin; the one-lane table is part of it."""
    for n in SERVE_LAYERS:
        plan = L.plan_serve(L.serve_config(n), L.load_profile("ref"))
        assert {r.slices[0] for r in plan.regions if r.kind == "text"} == set(L.TEXT_WORDS_T8[n])
        assert all(r.words == L.text_words(n, r.slices[0]) for r in plan.regions if r.kind == "text")
        assert all(L.TEXT_WORDS_ONE_LANE[n].get(s, 0) <= w for s, w in L.TEXT_WORDS_T8[n].items())


@pytest.fixture(scope="module", params=SERVE_LAYERS, ids=lambda n: f"l{n}")
def serve_images(request):
    cfg = L.serve_config(request.param)
    w = W.synthetic(cfg, seed=1)
    plan = L.plan(L.load_profile("ref"), mode="serve", layers=request.param)
    tensors, objects = L.build_serve_images(cfg, plan, w)
    return cfg, w, plan, tensors, objects


def same_bits(a, b):
    """Equal shape and equal bit patterns (`-0.0` differs from `0.0`, a NaN equals itself)."""
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}"))


def test_the_round_trips_keep_negative_zero_nan_and_infinity():
    """A checkpoint may hold -0.0; `np.array_equal` would call it equal to +0.0, the bits are compared."""
    w = W.synthetic(L.serve_config(2), seed=1)
    w.wte[7, 0], w.wte[8, 3], w.wte[50256, 63], w.wte[9, 1] = -0.0, np.nan, -np.inf, np.float16(np.nan).view(np.uint16) | 0x1
    plan = L.plan(L.load_profile("ref"), mode="serve", layers=2)
    tensors, _ = L.build_serve_images(L.serve_config(2), plan, w)
    assert same_bits(L.decode_embedding(tensors, w.cfg.vocab), w.wte) and same_bits(L.decode_head(tensors, w.cfg.vocab), w.wte)
    pos, neg = np.zeros(2, np.float16), np.zeros(2, np.float16)
    neg[0] = -0.0
    assert np.array_equal(pos, neg) and not same_bits(pos, neg)                    # the control: only the bits differ


def test_head_and_embedding_tables_decode_back_to_wte(serve_images):
    cfg, w, plan, tensors, objects = serve_images
    assert same_bits(L.decode_embedding(tensors, cfg.vocab), w.wte)        # every row, the padding rows included
    assert same_bits(L.decode_head(tensors, cfg.vocab), w.wte)
    # the padding entries are zero: 50 260 packed entries in the embedding, 50 400 in the head
    emb = L.decode_embedding(tensors, cfg.vocab, padded=True)
    head = L.decode_head(tensors, cfg.vocab, padded=True)
    assert emb.shape == (50260, 64) and not emb[cfg.vocab:].any()
    assert head.shape == (50400, 64) and not head[cfg.vocab:].any()
    # the last chunk of the head: install 62 holds entries 5 * 160 * 62 .. 50 399, 145 of them vocabulary rows
    assert same_bits(head[49600:cfg.vocab], w.wte[49600:])
    # the zero words and the wpe planes
    for s in (36, 37, 38, 39, 40, 41):
        assert not any(tensors[f"ez{s}"][1][0])
    wpe = L.decode_wpe(tensors)
    assert same_bits(wpe, w.wpe[:128].astype(np.float32))


def test_a_planted_defect_in_the_head_table_is_caught_by_the_round_trip(serve_images):
    cfg, w, plan, tensors, objects = serve_images
    bad = dict(tensors)
    words = list(bad["hd44"][1])
    words[7] = bytes([words[7][0] ^ 1]) + words[7][1:]
    bad["hd44"] = ("u8", words)
    assert not same_bits(L.decode_head(bad, cfg.vocab), w.wte)


def test_masks_constants_and_onehot_images(serve_images):
    cfg, w, plan, tensors, objects = serve_images
    for C in (64, 128):
        for p in (0, 5, C - 1):
            v = L.decode_mask(tensors, C, p)
            assert v.shape == (320,) and (v[:p + 1] == 0).all() and np.isneginf(v[p + 1:C]).all() and not v[C:].any()
    hi = [np.frombuffer(x, dtype=np.uint8) for x in tensors["ax64"][1]]
    assert len(hi) == 128 and all(h[s] == 0x3C and h.sum() == 0x3C for s, h in enumerate(hi))
    consts = L.const_arrays(cfg, w, serve=True)
    names = [n for n, _ in L.const_vectors(cfg, serve=True)]
    assert len(names) == (cfg.layers * 11 + 2 + 6 + 1)       # + k.zero, a splat of the batched Flow and len(tensors["cs16"][1]) == len(names)
    assert np.array_equal(L.decode_const(plan, tensors, "lnf.g", cfg).view(np.uint32), consts["lnf.g"].view(np.uint32))


def test_zero_words_and_zero_and_sink_blocks_lie_where_the_design_says(serve):
    """The zero word of each embedding part is word 4 096 (the first of bank 1); block 0 (zero) and block 128
    (sink) of every layer lie inside the K pool and the layer's V pool."""
    cfg, prof, plan, rep = serve
    for s in (36, 37, 38, 39, 40, 41):
        assert plan.by_name[f"ez{s}"].addr == prof.bank_words
    kp = plan.by_name["kp"]
    for l in range(cfg.layers):
        v = plan.by_name[f"vp{l}"]
        for block in (0, L.SINK_BLOCK):
            k0 = L.kblock_addr(plan, l, block)
            assert kp.addr <= k0 and k0 + 2 <= kp.addr + kp.words
            v0 = L.vblock_addr(plan, l, block)
            assert v.addr <= v0 and v0 + L.BLOCK_TOKENS <= v.addr + v.words
    assert L.kblock_addr(plan, 0, 0) == kp.addr and L.kblock_addr(plan, cfg.layers - 1, L.SINK_BLOCK) + 2 == kp.addr + kp.words
    assert L.vblock_addr(plan, 0, L.SINK_BLOCK) + L.BLOCK_TOKENS == plan.by_name["vp0"].addr + plan.by_name["vp0"].words


def test_pool_block_addressing(serve):
    cfg, prof, plan, rep = serve
    n = cfg.layers
    # K block b of layer l: word 16 b' + 2 l of the K slices, with b' the block index and stride 2 n words per block
    assert L.kblock_addr(plan, 0, 0) == plan.by_name["kp"].addr
    assert L.kblock_addr(plan, n - 1, 127) == plan.by_name["kp"].addr + 2 * n * 127 + 2 * (n - 1)
    assert L.kblock_addr(plan, 0, L.SINK_BLOCK) == plan.by_name["kp"].addr + 2 * n * 128
    assert L.vblock_addr(plan, n - 1, 3) == plan.by_name[f"vp{n - 1}"].addr + 16 * 3
    assert L.vblock_addr(plan, 0, L.SINK_BLOCK) == plan.by_name["vp0"].addr + 16 * 128
    with pytest.raises(ValueError):
        L.kblock_addr(plan, 0, 129)
    assert L.emb_locate(0) == (0, 0, 0) and L.emb_locate(5) == (0, 1, 0) and L.emb_locate(4096 * 5 + 7) == (1, 1, 2)
    assert L.emb_locate(50256) == (2, 1859, 1)


def test_the_report_names_the_source_of_the_text_arena(serve):
    cfg, prof, plan, rep = serve
    assert rep["text_source"] == f"measured from emit_step_batch: the largest bucket P(8, 4, 128), {cfg.layers} layers"


def test_serve_export_writes_package_manifest_and_report(tmp_path):
    out = tmp_path / "s"
    r = subprocess.run([sys.executable, os.path.join(_paths.HERE, "export.py"), "--synthetic", "1", "--mode", "serve",
                        "--layers", "2", "--out", str(out), "--raw"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.load(open(out / "layout.json"))
    rep = json.load(open(out / "layout_report.json"))
    assert m["mode"] == "serve" and m["model"]["layers"] == 2 and rep["layers"] == 2
    fa = m["flow_arena"]                 # The Flow's lane arenas (one placement for all buckets) are in the manifest
    assert fa["lanes"] == 8 and fa["region"] == "fa<t>_<hemisphere><quad>" and fa["bank"] == 1
    assert fa["words"]["E4"] == L.FLOW_ARENA[("E", 4)] and fa["tables"]["oh"] == ["foh", [12, 13], 128]
    assert m["pool_conflicts"]["region"] == "kp" and "tx<s>" in m["pool_conflicts"]["against"]
    assert rep["text_source"].startswith("measured from emit_step_batch")
    L.check_report(rep)
    assert m["package_sha256"] == hashlib.sha256(open(out / "model.safetensors", "rb").read()).hexdigest()
    args = [f"{e['name']}={e['dtype']}:{out / 'raw' / (e['name'] + '.bin')}" for e in m["externs"]]
    r = subprocess.run([sys.executable, os.path.join(_paths.SPEC, "tools", "tsp_as.py"), "wpack", "--profile", "ref",
                        "-o", str(tmp_path / "wpack.safetensors")] + args, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert open(tmp_path / "wpack.safetensors", "rb").read() == open(out / "model.safetensors", "rb").read()
    # the default mode is untouched by the new options
    r = subprocess.run([sys.executable, os.path.join(_paths.HERE, "export.py"), "--synthetic", "1", "--layers", "2",
                        "--out", str(tmp_path / "x")], capture_output=True, text=True)
    assert r.returncode != 0


def load_and_dump(plan, tensors, names, work):
    """Load the named tensors into SRAM through the weight package, run an empty program on the ISS (the compatible mode:
    `ots-asm objinput` text, a `D` line per word) and return {name: (dtype, [words read back])}."""
    import case_runner as runner_util
    runner_util.ensure_binaries()
    prof = plan.p
    cfg_path = os.path.join(_paths.SPEC, "config", prof.name + ".json")
    names = list(names)
    pkg = tsp_as.write_weights(prof.name, [(n, tensors[n][0], tensors[n][1]) for n in names])
    open(work / "p.safetensors", "wb").write(pkg)
    lines = [".arch opentsp-tsa", f".profile {prof.name}"]
    lines += [f".extern {n}, slice={plan.by_name[n].slices[0]}, addr={plan.by_name[n].addr:#x}, size={plan.by_name[n].words:#x}"
              for n in names]
    lines += [".section text", "bundle @cycle=1 {", "  VXM[0]: add.f16 E:8, E:0, E:2", "}"]
    open(work / "p.ots", "w").write("\n".join(lines) + "\n")
    r = subprocess.run([runner_util.ASM, "asm", "--config", cfg_path, "-o", str(work / "p.elf"), str(work / "p.ots")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([runner_util.ASM, "objinput", "--config", cfg_path, "--weights", str(work / "p.safetensors"), str(work / "p.elf")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = r.stdout + "".join(f"D {plan.by_name[n].slices[0]:x} {plan.by_name[n].addr + i:x}\n"
                              for n in names for i in range(plan.by_name[n].words))
    r = subprocess.run([runner_util.ISS, cfg_path, "--max-cycles", "100"], input=text, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[:500]
    words = {}
    for ln in r.stdout.split("\n"):
        if ln.startswith("W "):
            f = ln.split()
            words[(int(f[1], 16), int(f[2], 16))] = bytes(int(x, 16) for x in f[3:])
    return {n: ("u8", [words[(plan.by_name[n].slices[0], plan.by_name[n].addr + i)] for i in range(plan.by_name[n].words)])
            for n in names}


def test_the_full_head_loads_and_reads_back_on_the_iss(serve_images, tmp_path):
    """The packed head table of  (63 installs, 20 160 words), the embedding table with its zero words, `wpe`, the
    mask tables, the constants and the one-hot and head-mask vectors, loaded into SRAM by the weight package and read back
    through the dump of the ISS: every word equals the package's, and the tables decode to wte."""
    cfg, w, plan, tensors, objects = serve_images
    names = [r.name for r in plan.regions if r.exported and r.kind != "weight"]       # everything but the layer weights
    got = load_and_dump(plan, tensors, names, tmp_path)
    assert {n for n in names if n.startswith(("hd", "em", "ez", "wp", "mk", "cs", "ax", "fhm", "foh"))} == set(names)
    assert all(got[n][1] == list(tensors[n][1]) for n in names)                     # every word, byte for byte
    assert same_bits(L.decode_head(got, cfg.vocab), w.wte)
    assert same_bits(L.decode_embedding(got, cfg.vocab), w.wte)


def test_the_unpacked_head_cannot_cover_the_vocabulary():
    """With the reduce scratch known, an east plane holds 139 chunks of 20 words in bank 0, so 278 chunks (44 480 rows)
    fit and the 315 chunks of an unpacked head of 50 257 rows do not."""
    prof = L.load_profile("ref")
    L.plan_tinystories(CFG, prof, 278).validate()
    for chunks in (280, 315):
        with pytest.raises(ValueError, match="does not fit"):
            L.plan_tinystories(CFG, prof, chunks)


def _kernel_accesses(monkeypatch, chunks=4):
    """(writes, reads) of numeric addresses {(slice, addr)} that the step emitter's kernels issue on the MEM slices of the
    reduce-and-broadcast scratch (west 0..3, east 44..47); reads of named externs (weights, head rows) are not numeric."""
    from opentsp_vllm.models.tinystories import emit_step as es
    from opentsp_tools import profile
    from opentsp_tools.programs.kernels import Builder
    writes, reads = set(), set()
    orig_w, orig_r = Builder.write, Builder.read

    def write(self, v, slices, addrs, *a, **kw):
        writes.update(zip(slices, addrs))
        return orig_w(self, v, slices, addrs, *a, **kw)

    def read(self, cycle, slices, addrs, row, stream, *a, **kw):
        if not kw.get("addr_text"):
            reads.update(zip(slices, addrs))
        return orig_r(self, cycle, slices, addrs, row, stream, *a, **kw)

    monkeypatch.setattr(Builder, "write", write)
    monkeypatch.setattr(Builder, "read", read)
    es.build(profile.load("ref"), 5, head_chunks=chunks)
    return writes, reads


def test_every_fixed_kernel_address_lies_in_a_reduce_scratch_region(monkeypatch):
    """The regions `rbs_*` are derived from nothing but the emitter's own accesses: every numeric word that the kernels
    write or read on west 0..3 and east 44..47, outside the regions of the plan that the emitter binds by name (the K cache),
    lies in an `rbs_*` region (so the planner cannot drift from `Block.scratch` and `tree_round` again)."""
    writes, reads = _kernel_accesses(monkeypatch)
    plan = L.plan_tinystories(CFG, L.load_profile("ref"), 4)
    rbs = [r for r in plan.regions if r.name.startswith("rbs_")]
    scratch_slices = set(L.RB_WEST) | set(L.RB_EAST)
    hits = {(s, a) for s, a in writes | reads if s in scratch_slices and a >= L.RB_BASE and not
            any(r.name.startswith("kc") and s in r.slices and r.addr <= a < r.addr + r.words for r in plan.regions)}
    assert len(hits) >= 98 + 98          # 98 words on west 0..3 bank 1 (each of 4 slices) and the east words
    outside = sorted((s, hex(a)) for s, a in hits if not any(s in r.slices and r.addr <= a < r.addr + r.words for r in rbs))
    assert not outside, outside
    west = {a for s, a in hits if s in L.RB_WEST}
    assert min(west) == 0x1400 and max(west) == 0x151b                 # the span the region must cover
    # the planted control: a west region of the old size 0x24 leaves addresses outside
    short = [r for r in rbs if r.name != "rbs_w1"] + [Region("rbs_w1", "reserved", L.RB_WEST, 0x24, True, addr=0x1400)]
    assert any(not any(s in r.slices and r.addr <= a < r.addr + r.words for r in short) for s, a in hits if s in L.RB_WEST)


def _serve_kernel_accesses(monkeypatch, serving_program_cases):
    """{(slice, addr)} of the numeric words the `embed` and `lm_head` kernels write or read, built as the reduced program
    cases on the serving layout (reads of named externs are not numeric)."""
    from opentsp_tools import profile
    from opentsp_tools.programs.kernels import Builder
    hit = set()
    orig_w, orig_r = Builder.write, Builder.read

    def write(self, v, slices, addrs, *a, **kw):
        hit.update(zip(slices, addrs))
        return orig_w(self, v, slices, addrs, *a, **kw)

    def read(self, cycle, slices, addrs, row, stream, *a, **kw):
        if not kw.get("addr_text"):
            hit.update(zip(slices, addrs))
        return orig_r(self, cycle, slices, addrs, row, stream, *a, **kw)

    monkeypatch.setattr(Builder, "write", write)
    monkeypatch.setattr(Builder, "read", read)
    p = profile.load("ref")
    serving_program_cases.embed(p)
    serving_program_cases.lm_head(p)
    return hit


@pytest.mark.parametrize("layers", L.SERVE_LAYERS)
def test_the_serve_plan_declares_the_scratch_of_embed_and_lm_head(layers):
    """The words the two kernels address directly are regions `kfix_*` of the serve plan, so the planner places
    every table around them and `validate` rejects an overlap; the regions follow the kernels' own constants."""
    from opentsp_tools.programs.kernels import embed as ke, lm_head as kh
    plan = L.plan(L.load_profile("ref"), mode="serve", layers=layers)
    got = {r.name: (r.slices, r.addr, r.words, r.fixed) for r in plan.regions if r.name.startswith("kfix_")}
    assert got == {"kfix_embed": (ke.STAGE, ke.STAGE_ADDR, L.LANES_MAX, True),
                   "kfix_hid": (kh.H_STAGE, kh.H_STAGE_ADDR, L.LOGIT_ROWS, True),
                   "kfix_head": (kh.HK_WEST + kh.HK_EAST, kh.HK_ADDR, kh.SEGMENTS * L.LOGIT_ROWS, True)}
    plan.validate()


def test_a_region_over_the_kernel_scratch_of_the_serve_plan_is_rejected():
    pl = L.plan_serve(L.TINYSTORIES_L2, L.load_profile("ref"))
    pl.add("clash", "act", (16,), 4, rw=True, addr=0x802)
    with pytest.raises(AssertionError):
        pl.validate()


def test_every_fixed_address_of_embed_and_lm_head_lies_in_a_kfix_region(monkeypatch, serving_program_cases):
    """The `kfix_*` regions are checked against the accesses of the kernels' own program cases: every numeric word on their
    slices above the constants (0x400 and up) is inside one, and the cases touch every word of them but the unused
    lookups' (the reduced embed case issues the 8 lookups, the head case 4 logit rows)."""
    hit = _serve_kernel_accesses(monkeypatch, serving_program_cases)
    plan = L.plan(L.load_profile("ref"), mode="serve")
    kf = [r for r in plan.regions if r.name.startswith("kfix_")]
    assert len(kf) == 3
    slices = {s for r in kf for s in r.slices}
    mine = {(s, a) for s, a in hit if s in slices and a >= 0x400}
    outside = sorted((s, hex(a)) for s, a in mine if not any(s in r.slices and r.addr <= a < r.addr + r.words for r in kf))
    assert not outside, outside
    covered = {(s, r.addr + i) for r in kf for s in r.slices for i in range(r.words)}
    assert mine == covered
