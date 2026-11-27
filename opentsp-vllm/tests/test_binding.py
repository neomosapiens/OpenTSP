"""The binding of the ISS (part 1): `opentsp_vllm._tsp_iss.Chip`, and `IssChip` over it.

Run in the vLLM environment after `just vllm-binding` (`just vllm-binding-test`). A missing binding fails, never skips.
The programs are the ones of `tests/programs/gen/test_patch.py` (three patched reads, a patched permute map), a ported
program case (`model/ts_kv_write`) and golden vectors of the specification; they run on a small amount of SRAM state,
so every test is a few cycles long. The package under test is the one of this tree (`opentsp-vllm/src`), whichever
tree the venv's editable install points to.
"""

import hashlib
import json
import os
import pathlib
import random
import subprocess
import sys
import types

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))
sys.path.insert(0, str(REPO / "tests" / "programs" / "gen"))
sys.path.insert(0, str(REPO / "opentsp-arch-spec" / "tools"))

import test_patch as tp
from opentsp_tools import refasm
tsp_as = refasm.load()
from opentsp_tools import patch
from opentsp_tools import profile as tprofile

from opentsp_vllm import _tsp_iss  # noqa: E402  (a missing binding is a failure: build it with `just vllm-binding`)
from opentsp_vllm.chip import Chip, ChipInfo, IssChip, TspChipError, patch_columns  # noqa: E402

PROGRAM_CASES = REPO / "tests" / "programs"


def assemble(text, config, tmp, name):
    src, out = tmp / f"{name}.ots", tmp / f"{name}.elf"
    src.write_text(text)
    subprocess.run(
        [sys.executable, str(REPO / "opentsp-arch-spec" / "tools" / "tsp_as.py"), "asm", str(src), "-o", str(out), "--config", config],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


def object_of(profile, images, config):
    return tsp_as.write_object(types.SimpleNamespace(profile=profile, images=images, externs={}), config)


class Reads:
    """Three MEM reads of slices 0, 4, 8 (addresses 0x10..0x1f, patched) written to slices 44, 48, 52: output = the three
    words read. Data is written through the aperture before a step."""

    def __init__(self, tmp):
        self.p = tprofile.load("ref")
        self.view = tp.view_of(tp.reads(self.p))
        self.table = patch.locate(self.view, self.p)
        self.base = patch.to_image(patch.lower(self.view, self.p))
        self.elf = assemble(self.view.program(self.p).text({}), self.p.path, tmp, "reads")
        self.outputs = [(tp.DSTS[i], tp.DST_ADDR[i], 1) for i in range(3)]
        rng = random.Random(7)
        self.data = {
            (tp.SRCS[i], a): bytes(rng.randrange(256) for _ in range(self.p.vlanes))
            for i in range(3)
            for a in range(tp.LO, tp.HI + 1)
        }

    def load(self, chip, name="reads", table=True):
        chip.load_program(
            name,
            self.elf,
            patch_table=patch_columns(self.table) if table else None,
            outputs=self.outputs,
        )

    def preload(self, chip):
        for (s, a), w in self.data.items():
            chip.write_words(s, a, w)

    @staticmethod
    def values(addrs):
        return {f"rd{i}": a for i, a in enumerate(addrs)}

    def want(self, addrs):
        return b"".join(self.data[(tp.SRCS[i], a)] for i, a in enumerate(addrs))


class PermuteMap:
    """The permute of `test_patch.RunPatchedMap`: one site, the map index; one word of output."""

    def __init__(self, tmp):
        self.p = tprofile.load("ref")
        self.view, self.preload_words, self.want_words = tp.RunPatchedMap().program(self.p, random.Random(11))
        self.table = patch.locate(self.view, self.p)
        self.elf = assemble(self.view.program(self.p).text({}), self.p.path, tmp, "permute")
        self.outputs = [(44, tp.RunPatchedMap.DST, 1)]

    def load(self, chip, name="permute"):
        chip.load_program(name, self.elf, patch_table=patch_columns(self.table), outputs=self.outputs)

    def preload(self, chip):
        for (s, a), w in self.preload_words.items():
            chip.write_words(s, a, w)


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("binding")


@pytest.fixture(scope="module")
def reads(tmp):
    return Reads(tmp)


@pytest.fixture(scope="module")
def permute(tmp):
    return PermuteMap(tmp)


def chip(mode="warm"):
    c = _tsp_iss.Chip("ref")
    c.restart_mode(mode)
    return c


# ---- a program runs and its patches are applied ----


def test_a_patched_program_reads_the_patched_words(reads):
    c = chip()
    reads.load(c)
    reads.preload(c)
    for addrs in [(0x10, 0x11, 0x12), (0x1F, 0x13, 0x10), (0x15, 0x1E, 0x17)]:
        words, cycles, diags = c.step("reads", reads.values(addrs), 1000)
        assert words == reads.want(addrs)
        assert list(diags) == []
        assert cycles > 0


def test_patches_applied_in_rust_equal_the_python_patcher(reads, tmp):
    c = chip()
    reads.load(c)
    reads.preload(c)
    rng = random.Random(3)
    for k in range(4):
        vals = reads.values([rng.randrange(tp.LO, tp.HI + 1) for _ in range(3)])
        units = patch.to_units(patch.apply(reads.base, reads.table, vals))
        c2 = chip()
        c2.load_program("by_python", object_of("ref", units, reads.p.path), outputs=reads.outputs)
        reads.preload(c2)
        a = c.step("reads", vals, 1000)
        b = c2.step("by_python", None, 1000)
        assert a == b, vals
    # the dense form (int32 per site in table order) is the same patch
    vals = reads.values([0x12, 0x1E, 0x14])
    dense = np.array([vals[str(n)] for n in reads.table.names], dtype="<i4").tobytes()
    assert c.step("reads", dense, 1000) == c.step("reads", vals, 1000)


def test_the_permute_map_is_patched_and_the_two_maps_differ(permute):
    c = chip()
    permute.load(c)
    permute.preload(c)
    for idx in (0, 1, 0):
        words, _, _ = c.step("permute", {"pm": idx}, 2000)
        assert words[: len(permute.want_words[idx])] == permute.want_words[idx]
    assert permute.want_words[0] != permute.want_words[1]


def test_a_refused_patch_leaves_the_chip_untouched(reads):
    c = chip()
    reads.load(c)
    reads.preload(c)
    ok = reads.values([0x10, 0x11, 0x12])
    c.step("reads", ok, 1000)
    before, steps = c.sram_digest(), c.stats()["steps"]
    for bad, text in [
        ({**ok, "rd1": tp.HI + 1}, "rd1"),
        ({**ok, "rd0": tp.LO - 1}, "rd0"),
        ({**ok, "nosuch": 1}, "unknown patch sites"),
        ({"rd0": 0x10}, "no value for patch sites"),
    ]:
        with pytest.raises(ValueError, match=text):
            c.step("reads", bad, 1000)
    with pytest.raises(ValueError, match="patch sites"):
        c.step("reads", None, 1000)
    with pytest.raises(ValueError, match="no program"):
        c.step("nothing", None, 1000)
    assert (c.sram_digest(), c.stats()["steps"]) == (before, steps)


def test_a_malformed_table_is_refused(reads):
    cols = patch_columns(reads.table)
    c = chip()
    for key, value, text in [
        ("lo", cols["lo"][:-1], "wrong length"),
        ("names", ["rd0", "rd0", "rd2"], "duplicate"),
        ("vmax", [1 << 20] * 3, "range"),
    ]:
        with pytest.raises(ValueError, match=text):
            c.load_program("x", reads.elf, patch_table={**cols, key: value}, outputs=reads.outputs)
    with pytest.raises(ValueError, match="no column"):
        c.load_program("x", reads.elf, patch_table={k: v for k, v in cols.items() if k != "span"})


# ---- differential tests of the Rust patcher against `patch.apply` ----


def gridded(table, stride, span):
    """The `Reads` table with every site on a grid: a value is accepted when (v - vmin) % stride < span. The placeholder is
    the grid's first value."""
    cols = {c: getattr(table, c) for c in patch.COLUMNS}
    cols["placeholder"] = table.vmin
    return patch.Table(table.names, *[cols[c] for c in patch.COLUMNS], stride=[stride] * len(table), span=[span] * len(table))


def python_refuses(base, table, values):
    try:
        return patch.to_units(patch.apply(base, table, values)), None
    except patch.PatchError as e:
        return None, e


def test_the_exception_class_is_patch_error_and_a_value_error(reads):
    assert _tsp_iss.PatchError.__name__ == "PatchError" and issubclass(_tsp_iss.PatchError, ValueError)
    c = chip()
    reads.load(c)
    ok = reads.values([0x10, 0x11, 0x12])
    for bad in [
        {**ok, "rd1": tp.HI + 1},
        {**ok, "nosuch": 1},
        {"rd0": 0x10},
        {**ok, "rd0": 1.5},
        {**ok, "rd0": True},
        {**ok, "rd0": "16"},
        {**ok, "rd0": None},
        {**ok, "rd0": 1 << 70},
        {**ok, "rd0": -(1 << 70)},
        {**ok, 3: 0x10},
        b"\x00\x00\x00",
    ]:
        with pytest.raises(_tsp_iss.PatchError):
            c.step("reads", bad, 1000)
        if isinstance(bad, dict):
            assert python_refuses(reads.base, reads.table, bad)[1] is not None, bad
    with pytest.raises(_tsp_iss.PatchError, match="patch sites"):
        c.step("reads", None, 1000)
    with pytest.raises(TypeError):
        c.step("reads", [1, 2, 3], 1000)


def test_numpy_integers_are_patch_values_as_for_the_python_patcher(reads):
    """`patch.apply` takes any `numbers.Integral`, and the plans' patch values are numpy integers: the binding takes them
    (anything with `__index__`, bool excepted) and refuses what `patch.apply` refuses (floats, strings, None, bool)."""

    class Indexed:  # not an `int`, not registered as `Integral`, but an index: accepted by the binding
        def __init__(self, v):
            self.v = v

        def __index__(self):
            return self.v

    c, py_chip = chip(), chip()
    reads.load(c)
    reads.preload(c)
    reads.preload(py_chip)
    base = [0x10, 0x11, 0x12]
    for make in (np.int64, np.int32, np.int16, np.uint8, np.uint16, np.uint32, np.uint64, int):
        vals = {f"rd{i}": make(a) for i, a in enumerate(base)}
        units, err = python_refuses(reads.base, reads.table, vals)
        assert err is None, make
        py_chip.load_program("by_python", object_of("ref", units, reads.p.path), outputs=reads.outputs)
        assert c.step("reads", vals, 1000) == py_chip.step("by_python", None, 1000), make
    # an array element, a scalar of a numpy array and a mixed dict
    arr = np.array([0x13, 0x14, 0x15], dtype=np.int64)
    vals = {f"rd{i}": arr[i] for i in range(3)}
    assert c.step("reads", vals, 1000)[0] == reads.want([0x13, 0x14, 0x15])
    assert c.step("reads", {"rd0": np.int32(0x11), "rd1": 0x12, "rd2": np.uint8(0x13)}, 1000)[0] == reads.want([0x11, 0x12, 0x13])
    assert c.step("reads", {f"rd{i}": Indexed(a) for i, a in enumerate(base)}, 1000)[0] == reads.want(base)
    # the same refusals as `patch.apply`, with one class
    ok = reads.values(base)
    for bad in [1.5, np.float64(16.0), "16", None, True, False, np.bool_(True), b"16", [16], np.int64(tp.HI + 1), np.int32(-1), np.uint64(2**63)]:
        vals = {**ok, "rd0": bad}
        assert python_refuses(reads.base, reads.table, vals)[1] is not None, repr(bad)
        with pytest.raises(_tsp_iss.PatchError):
            c.step("reads", vals, 1000)


def test_a_grid_of_stride_and_span_is_accepted_and_refused_as_the_python_patcher_does(reads):
    rng = random.Random(20261114)
    names = [str(n) for n in reads.table.names]
    counts = {"accepted": 0, "refused": 0, "values": 0}
    for stride, span in [(4, 2), (8, 1), (16, 16), (3, 3), (5, 2)]:
        table = gridded(reads.table, stride, span)
        c, py_chip = chip(), chip()
        c.load_program("grid", reads.elf, patch_table=patch_columns(table), outputs=reads.outputs)
        reads.preload(c)
        reads.preload(py_chip)
        for _ in range(120):
            vals = {}
            for n in names:
                kind = rng.randrange(4)
                if kind == 0:  # on the grid
                    k = rng.randrange((tp.HI - tp.LO) // stride + 1)
                    vals[n] = tp.LO + k * stride + rng.randrange(span)
                elif kind == 1:  # anywhere in the range
                    vals[n] = rng.randrange(tp.LO, tp.HI + 1)
                elif kind == 2:  # off the range on either side
                    vals[n] = rng.choice([tp.LO - rng.randrange(1, 40), tp.HI + rng.randrange(1, 40)])
                else:  # the edges
                    vals[n] = rng.choice([tp.LO, tp.HI, tp.LO + span - 1, tp.LO + stride - 1, tp.LO + stride])
            counts["values"] += len(vals)
            units, err = python_refuses(reads.base, table, vals)
            if err is not None:
                with pytest.raises(_tsp_iss.PatchError):
                    c.step("grid", vals, 1000)
                counts["refused"] += 1
                continue
            py_chip.load_program("by_python", object_of("ref", units, reads.p.path), outputs=reads.outputs)
            assert c.step("grid", vals, 1000) == py_chip.step("by_python", None, 1000), (stride, span, vals)
            counts["accepted"] += 1
    assert counts["values"] >= 1000 and counts["accepted"] >= 40 and counts["refused"] >= 40, counts


def kv_table(p, kind, C, rng):
    """A  program at its reduced size (C = 64) built through the generator, with its patch table and image."""
    import kv
    import test_kv
    from test_patch import view_of

    make = {"scores": kv.kv_scores, "context": kv.kv_context}[kind]
    tabs = []
    for _ in range(6):
        table, live = test_kv.random_table(rng, C)
        tabs.append((table, 16 * (live - 1) + rng.randrange(16)))
    first = make(p, C, table=tabs[0][0], pos=tabs[0][1], kit_only=True)
    view = view_of(first.b)
    return view, patch.locate(view, p), patch.to_image(patch.lower(view, p)), tabs, make


def test_the_patcher_equals_python_on_real_kv_tables(tmp):
    p = tprofile.load("ref")
    rng = random.Random(1114)
    compared = 0
    for kind in ("scores", "context"):
        C = 64
        view, table, base, tabs, make = kv_table(p, kind, C, rng)
        assert len(table) >= 16 and any(int(x) > 1 for x in table.stride), "a table of the block grid"
        elf = assemble(view.program(p).text({}), p.path, tmp, f"kv_{kind}")
        c, py_chip = chip(), chip()
        c.load_program("kv", elf, patch_table=patch_columns(table), outputs=[])
        # the block tables of the generator: the sites' placeholders of each program are the values to patch in
        for tab, pos in tabs:
            built = make(p, C, table=tab, pos=pos, kit_only=True)
            built.b.finish()
            vals = {s.name: s.placeholder for s in built.b.prog.sites}
            units, err = python_refuses(base, table, vals)
            assert err is None
            py_chip.load_program("by_python", object_of("ref", units, p.path), outputs=[])
            assert c.step("kv", vals, 500_000) == py_chip.step("by_python", None, 500_000)
            assert c.sram_digest() == py_chip.sram_digest()
            compared += 1
        # off-grid, out of range and in-range values over a few sites at a time: refused or accepted as Python does
        names = [str(n) for n in table.names]
        for _ in range(300):
            vals = {s.name: s.placeholder for s in built.b.prog.sites}
            for n in rng.sample(names, min(3, len(names))):
                i = list(table.names).index(n)
                vals[n] = int(table.vmin[i]) + rng.randrange(-3, int(table.vmax[i] - table.vmin[i]) + 4)
            units, err = python_refuses(base, table, vals)
            if err is not None:
                with pytest.raises(_tsp_iss.PatchError):
                    c.step("kv", vals, 500_000)
            else:
                py_chip.load_program("by_python", object_of("ref", units, p.path), outputs=[])
                assert c.step("kv", vals, 500_000) == py_chip.step("by_python", None, 500_000)
            compared += 1
    assert compared >= 500, compared


def test_the_patcher_equals_python_on_the_kv_write_table(tmp):
    import kv
    from test_patch import view_of

    p = tprofile.load("ref")
    first = kv.kv_write(p, block=3, offset=2, kit_only=True)
    view = view_of(first.b)
    table, base = patch.locate(view, p), patch.to_image(patch.lower(view, p))
    elf = assemble(view.program(p).text({}), p.path, tmp, "kv_write")
    c, py_chip = chip(), chip()
    c.load_program("kvw", elf, patch_table=patch_columns(table), outputs=[])
    for block, offset in [(3, 2), (5, 7), (1, 0), (None, 0), (8, 15), (2, 9)]:
        built = kv.kv_write(p, block=block, offset=offset, kit_only=True)
        built.b.finish()
        vals = {s.name: s.placeholder for s in built.b.prog.sites}
        units, err = python_refuses(base, table, vals)
        assert err is None, (block, offset, err)
        py_chip.load_program("by_python", object_of("ref", units, p.path), outputs=[])
        assert c.step("kvw", vals, 500_000) == py_chip.step("by_python", None, 500_000)
        assert c.sram_digest() == py_chip.sram_digest()


def test_the_patcher_equals_python_on_an_embedding_table_with_any_bank_sites(tmp, monkeypatch):
    """The  `embed` program at its reduced size (`model/ts_embed`): its lookups are MEM sites on either bank
    (`any_bank`), so a value may lie in bank 0 or in bank 1. Rust and Python must accept and refuse the same values and
    the patched runs must be the same."""
    if "TSP_ISS" not in os.environ:
        monkeypatch.setenv("TSP_ISS", "/nonexistent")  # the module reads it; nothing here runs a binary (and the variable is restored)
    import model_serve as ms
    import test_serve_kernels as tsk

    p = tsk.P
    rp = tsk.RunPatched()
    k, table, base, preload = rp.program(ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS, ms.REDUCED_SPECIAL_TOKENS)
    view = tsk.CycleView.of(k.b.prog)
    elf = assemble(view.program(p).text({}), p.path, tmp, "embed")
    outputs = [(s, a, 1) for (s, a) in k.dump]

    def with_preload():
        c = chip()
        for (s, a), w in preload.items():
            c.write_words(s, a, w)
        return c

    c, py_chip = with_preload(), with_preload()
    c.load_program("embed", elf, patch_table=patch_columns(table), outputs=outputs)
    plan = tsk.LAYOUT.plan(tsk.LAYOUT.load_profile("ref"), mode="serve")
    tables = ms.tables(tsk.LAYOUT, plan, ms.REDUCED_SIZES)
    rng = random.Random(1114)
    bank_of = lambda v: (v >> 12) & 1  # `ref`: 0x1000 words per bank
    seen = {"values": 0, "accepted": 0, "refused": 0, "bank0": 0, "bank1": 0}
    names = [str(n) for n in table.names]
    for it in range(60):
        vals = ms.site_values(tables, ms.REDUCED_SIZES, rp.lookups(rng, ms.REDUCED_SIZES, len(ms.REDUCED_LOOKUPS)))
        if it % 2:  # corrupt some values: off the grid, out of range, in the other bank
            for n in rng.sample(names, 6):
                i = names.index(n)
                vals[n] = int(table.vmin[i]) + rng.randrange(-3, int(table.vmax[i] - table.vmin[i]) + 4)
        seen["values"] += len(vals)
        units, err = python_refuses(base, table, vals)
        if err is not None:
            with pytest.raises(_tsp_iss.PatchError):
                c.step("embed", vals, 5000)
            seen["refused"] += 1
            continue
        py_chip.load_program("by_python", object_of("ref", units, p.path), outputs=outputs)
        assert c.step("embed", vals, 5000) == py_chip.step("by_python", None, 5000), it
        seen["accepted"] += 1
        for v in vals.values():
            seen[f"bank{bank_of(v)}"] += 1
    assert len(table) >= 32 and seen["values"] >= 300, (len(table), seen)
    assert seen["accepted"] >= 10 and seen["refused"] >= 10, seen
    assert seen["bank0"] > 0 and seen["bank1"] > 0, f"accepted values in both banks: {seen}"


def table_columns(**over):
    base = dict(
        names=["a", "b"], queue=[1, 1], unit=[0, 1], nunits=[1, 1], lo=[0, 0], width=[4, 4], vmin=[0, 0],
        vmax=[15, 15], placeholder=[0, 0], fixed=[0, 0], stride=[1, 1], span=[1, 1],
    )
    base.update(over)
    return base


def test_the_table_checks_of_the_python_validator_are_made_in_rust(reads):
    # a placeholder off the grid, a field on the opcode bits, two sites on the same bits: refused by `Table.validate` and
    # by the binding alike
    cases = [
        ("placeholder", dict(stride=[4, 4], span=[2, 2], placeholder=[3, 0]), "placeholder"),
        ("fixed bits", dict(fixed=[0x1, 0x0]), "opcode"),
        ("overlap", dict(unit=[0, 0], lo=[0, 2]), "overlaps"),
    ]
    c = chip()
    for what, over, text in cases:
        cols = table_columns(**over)
        with pytest.raises(patch.PatchError):
            patch.Table(cols["names"], *[cols[k] for k in patch.COLUMNS], stride=cols["stride"], span=cols["span"])
        with pytest.raises(_tsp_iss.PatchError, match=text):
            c.load_program("x", reads.elf, patch_table=cols, outputs=reads.outputs)
    # the same table without the fault is accepted by both
    cols = table_columns()
    patch.Table(cols["names"], *[cols[k] for k in patch.COLUMNS])
    c.load_program("x", reads.elf, patch_table=cols, outputs=reads.outputs)


# ---- warm restart and replay ----


def test_warm_and_replay_give_identical_runs(reads, permute):
    warm, replay = chip("warm"), chip("replay")
    assert (warm.restart_mode(), replay.restart_mode()) == ("warm", "replay")
    for c in (warm, replay):
        reads.load(c)
        permute.load(c)
        reads.preload(c)
        permute.preload(c)
    rng = random.Random(5)
    for k in range(8):
        if k % 2:
            args = ("permute", {"pm": rng.randrange(2)}, 2000)
        else:
            args = ("reads", reads.values([rng.randrange(tp.LO, tp.HI + 1) for _ in range(3)]), 1000)
        a, b = warm.step(*args), replay.step(*args)
        assert a == b, (k, args)
        assert warm.sram_digest() == replay.sram_digest(), (k, args)
    sw, sr = warm.stats(), replay.stats()
    assert (sw["restarts_warm"], sw["restarts_replay"]) == (8, 0)
    assert (sr["restarts_warm"], sr["restarts_replay"]) == (0, 8) and sr["replay_words"] > 0
    assert sw["cycles_total"] == sr["cycles_total"] > 0


def test_the_sram_of_a_step_is_there_for_the_next_one(reads, permute):
    # the permute program writes slice 44 only: the words the reads wrote to slices 48 and 52 are still there after the
    # next run's restart, in both modes
    for mode in ("warm", "replay"):
        c = chip(mode)
        reads.load(c)
        permute.load(c)
        reads.preload(c)
        permute.preload(c)
        addrs = (0x10, 0x11, 0x12)
        c.step("reads", reads.values(addrs), 1000)
        c.step("permute", {"pm": 1}, 2000)
        for i in (1, 2):
            assert c.read_words(tp.DSTS[i], tp.DST_ADDR[i], 1) == reads.data[(tp.SRCS[i], addrs[i])], (mode, i)
        assert c.read_words(44, tp.RunPatchedMap.DST, 1)[: len(permute.want_words[1])] == permute.want_words[1]


def test_restart_mode_is_validated_and_reported():
    c = chip()
    assert c.restart_mode() == "warm"
    assert c.restart_mode("replay") == "replay" and c.restart_mode() == "replay"
    with pytest.raises(ValueError, match="warm"):
        c.restart_mode("cold")
    assert c.restart_mode() == "replay"


# ---- packages and externs ----


def case_files(name, tmp):
    case = json.loads((PROGRAM_CASES / "model" / f"{name}.ref.case.json").read_text())
    p = tprofile.load(case["profile"])
    elf = assemble((PROGRAM_CASES / "model" / case["program"]).read_text(), p.path, tmp, name)
    tensors = [
        (n, None, [bytes.fromhex(w["hex"])[i : i + p.vlanes] for i in range(0, len(bytes.fromhex(w["hex"])), p.vlanes)])
        for n, w in case["weights"].items()
    ]
    return case, p, elf, tsp_as.write_weights(case["profile"], tensors, p.path)


def test_a_ported_case_runs_from_its_package_to_its_expected_words(tmp):
    case, p, elf, package = case_files("ts_kv_write", tmp)
    c = chip()
    assert c.load_package(package) == len(case["weights"])
    c.load_program("kv_write", elf, outputs=[])
    words, cycles, diags = c.step("kv_write", None, 100_000)
    assert words == b"" and list(diags) == []
    assert 0 < cycles <= case["budget"]["cycles"]
    for key, hexword in case["expect"]["words"].items():
        s, a = (int(x) for x in key.split(":"))
        assert c.read_words(s, a, 1)[: p.vlanes] == bytes.fromhex(hexword), key
    # the same package given as a file
    path = tmp / "kv.safetensors"
    path.write_bytes(package)
    c2 = chip()
    assert c2.load_package(str(path)) == len(case["weights"])
    c2.load_program("kv_write", elf, outputs=[])
    c2.step("kv_write", None, 100_000)
    assert c2.sram_digest() == c.sram_digest()


def test_externs_of_a_model_package_are_written_once(tmp):
    case, p, elf, package = case_files("ts_kv_write", tmp)
    c = chip()
    c.load_package(package)
    c.load_program("a", elf, outputs=[])
    n = c.stats()["extern_words"]
    assert n > 0
    c.load_program("b", elf, outputs=[])
    assert c.stats()["extern_words"] == n, "the same extern of the same package is not written again"
    c.load_program("c", elf, outputs=[], bind_model=False)
    assert c.stats()["extern_words"] == n


COPY_TEXT = """.section text
bundle @cycle=0 {
  MEM[0]: read 0x0, E:0, 0x0
}
bundle @cycle=11 {
  MEM[44]: write 0x1, E:0, 0x0
}
"""


def copy_objects(tmp):
    """A mini program that copies MEM[0] word 0 to MEM[44] word 1, and the same text with two externs over its input:
    `aaa` (one word at 0) and `bbb` (two words at 2), with a package whose `bbb` has one word: `aaa` binds, `bbb` does not."""
    mini = tprofile.load("mini")
    header = ".arch opentsp-tsa\n.profile mini\n"
    good = assemble(header + COPY_TEXT, mini.path, tmp, "copy-good")
    externs = ".extern aaa, slice=0, addr=0x0, size=0x1\n.extern bbb, slice=0, addr=0x2, size=0x2\n"
    bad = assemble(header + externs + COPY_TEXT, mini.path, tmp, "copy-bad")
    one = bytes([0xBB]) * mini.vlanes
    package = tsp_as.write_weights("mini", [("aaa", None, [one]), ("bbb", None, [one])], mini.path)
    return mini, good, bad, package


@pytest.mark.parametrize("mode", ["warm", "replay"])
@pytest.mark.parametrize("via", ["text_package", "model_package"])
def test_a_rejected_load_changes_no_sram_no_program_and_no_written_extern(tmp, mode, via):
    mini, good, bad, package = copy_objects(tmp)
    c = _tsp_iss.Chip("mini")
    c.restart_mode(mode)
    c.write_words(0, 0, bytes([0xAA]) * mini.vlanes)
    c.load_program("good", good, outputs=[(44, 1, 1)])
    before = c.step("good", max_cycles=1000)
    assert before[0] == bytes([0xAA]) * mini.vlanes
    digest, words = c.sram_digest(), c.stats()["extern_words"]
    load = {"text_package": lambda name: c.load_program(name, bad, text_package=package, outputs=[])}
    if via == "model_package":
        c.load_package(package)
        load["model_package"] = lambda name: c.load_program(name, bad, outputs=[])
    for name in ("bad", "good"):  # a new name, and the name of the loaded program
        with pytest.raises(ValueError, match="OTS-E-ASM-004.*extern bbb"):
            load[via](name)
        assert c.sram_digest() == digest, f"a rejected load of {name!r} wrote SRAM"
        assert c.stats()["extern_words"] == words
        assert c.step("good", max_cycles=1000) == before, f"a rejected load of {name!r} changed the program 'good'"
    with pytest.raises(ValueError, match="no program"):
        c.step("bad", max_cycles=1000)
    if via == "model_package":
        # `aaa` was not recorded as written: a program that binds only `aaa` writes it now
        only_aaa = assemble(
            ".arch opentsp-tsa\n.profile mini\n.extern aaa, slice=0, addr=0x0, size=0x1\n" + COPY_TEXT, mini.path, tmp, "copy-aaa"
        )
        c.load_program("aaa", only_aaa, outputs=[(44, 1, 1)])
        assert c.stats()["extern_words"] == words + 1
        assert c.step("aaa", max_cycles=1000)[0] == bytes([0xBB]) * mini.vlanes


def test_a_package_for_another_profile_or_a_bad_package_is_refused(tmp):
    c = chip()
    with pytest.raises(ValueError):
        c.load_package(b"not a package")
    mini = tprofile.load("mini")
    other = tsp_as.write_weights("mini", [("w", None, [bytes(mini.vlanes)])], mini.path)
    with pytest.raises(ValueError, match="profile"):
        c.load_package(other)
    with pytest.raises(ValueError):
        c.load_package(str(tmp / "no-such-file.safetensors"))


# ---- the aperture ----


def test_words_round_trip_through_the_aperture():
    c = chip()
    n = c.vlanes
    rng = random.Random(1)
    data = bytes(rng.randrange(256) for _ in range(5 * n))
    c.write_words(20, 0x1000, data)
    assert c.read_words(20, 0x1000, 5) == data
    assert c.read_words(20, 0x1002, 2) == data[2 * n : 4 * n]
    assert c.read_words(20, 0x1005, 1) == bytes(n)
    with pytest.raises(ValueError, match="whole number"):
        c.write_words(20, 0, bytes(n + 1))
    with pytest.raises(TspChipError):
        c.read_words(20, 1 << 20, 1)
    with pytest.raises(TspChipError):
        c.write_words(200, 0, bytes(n))
    assert c.read_words(20, 0x1000, 5) == data


def test_a_huge_read_is_refused_before_a_buffer_is_sized_and_a_write_is_all_or_nothing():
    c = chip()
    n = c.vlanes
    end = 0x2000  # words per slice of `ref`
    with pytest.raises(TspChipError, match="beyond the slice end"):
        c.read_words(20, 0, 2**32 - 1)
    with pytest.raises(TspChipError, match="beyond the slice end"):
        c.read_words(20, end - 1, 2)
    assert c.read_words(20, end - 2, 2) == bytes(2 * n)
    before = c.sram_digest()
    data = bytes([7]) * (3 * n)
    with pytest.raises(TspChipError, match="beyond the slice end"):
        c.write_words(20, end - 2, data)  # the third word is outside: not even the first two are written
    assert c.sram_digest() == before and c.read_words(20, end - 2, 2) == bytes(2 * n)
    c.write_words(20, end - 3, data)
    assert c.read_words(20, end - 3, 3) == data


# ---- failures ----


def mini_vector(name):
    doc = json.loads((REPO / "opentsp-arch-spec" / "conformance" / "model" / "mini.json").read_text())
    return next(v for v in doc["vectors"] if v["name"] == name)


def vector_object(v, config):
    images = {int(q): [int(u, 16) for u in units.split()] for q, units in sorted(v["text"].items(), key=lambda t: (int(t[0]) == 0xFFFF, int(t[0])))}
    return object_of("mini", images, config)


def test_a_diagnostic_raises_tsp_chip_error_with_cycle_queue_and_code():
    mini = tprofile.load("mini")
    c = _tsp_iss.Chip("mini")
    for name, (cycle, queue, code) in {
        "directed-020": (20, 1, "OTS-E-DEC-003"),
        "directed-002": (10, 88, "OTS-E-RES-001"),
    }.items():
        v = mini_vector(name)
        assert v["expect"]["diagnostics"][0] == {"cycle": cycle, "queue": queue, "code": code}
        c.load_program(name, vector_object(v, mini.path), outputs=[])
        with pytest.raises(TspChipError, match=code) as e:
            c.step(name, None, 10_000)
        assert (e.value.cycle, e.value.queue, e.value.code) == (cycle, queue, code)
        assert e.value.diagnostics[0] == (cycle, queue, code)
        assert isinstance(e.value, RuntimeError)
    # the chip is usable after it, and a run without diagnostics returns normally
    ok = mini_vector("directed-000")
    c.load_program("ok", vector_object(ok, mini.path), outputs=[])
    assert c.step("ok", None, 10_000)[2] in ((), [])


def test_two_diagnostics_are_all_reported():
    mini = tprofile.load("mini")
    c = _tsp_iss.Chip("mini")
    v = mini_vector("directed-026")
    c.load_program("two", vector_object(v, mini.path), outputs=[])
    with pytest.raises(TspChipError) as e:
        c.step("two", None, 10_000)
    assert [d[2] for d in e.value.diagnostics] == ["OTS-E-DEC-003", "OTS-E-DEC-003"] and "1 more" in str(e.value)


def test_a_chip_not_done_by_the_cycle_limit_raises_and_recovers(reads):
    c = chip()
    reads.load(c)
    reads.preload(c)
    ok = reads.values([0x10, 0x11, 0x12])
    with pytest.raises(TspChipError, match="not done") as e:
        c.step("reads", ok, 3)
    assert e.value.cycle is None and e.value.diagnostics == []
    assert c.step("reads", ok, 1000)[0] == reads.want((0x10, 0x11, 0x12))


def test_a_rust_panic_is_a_tsp_chip_error_not_a_base_exception():
    c = chip()
    with pytest.raises(TspChipError, match="panic in the simulator: deliberate") as e:
        c._panic_for_test()
    assert isinstance(e.value, Exception)


def test_the_error_class_is_the_packages():
    import opentsp_vllm.chip as chip_module

    mini = tprofile.load("mini")
    c = _tsp_iss.Chip("mini")
    c.load_program("d", vector_object(mini_vector("directed-020"), mini.path), outputs=[])
    with pytest.raises(chip_module.TspChipError):
        c.step("d", None, 10_000)
    assert issubclass(_tsp_iss.TspIssError, RuntimeError)


# ---- statistics ----


def test_stats_count_steps_cycles_patches_and_restarts(reads):
    c = chip()
    reads.load(c)
    reads.preload(c)
    s0 = c.stats()
    assert (s0["steps"], s0["cycles_total"], s0["patches_applied"], s0["mode"]) == (0, 0, 0, "warm")
    total = 0
    for k in range(3):
        total += c.step("reads", reads.values([0x10 + k, 0x11, 0x12]), 1000)[1]
    s = c.stats()
    assert (s["steps"], s["cycles_total"], s["cycles_last"]) == (3, total, total // 3)
    assert (s["patches_applied"], s["restarts_warm"], s["restarts_replay"]) == (9, 3, 0)
    assert s["run_seconds"] > 0 and s["restart_seconds"] >= 0 and s["load_seconds"] > 0
    c.restart_mode("replay")
    c.step("reads", reads.values([0x10, 0x11, 0x12]), 1000)
    assert c.stats()["restarts_replay"] == 1 and c.stats()["mode"] == "replay"


def test_the_sram_digest_is_the_hash_of_the_nonzero_words():
    c = chip()
    assert c.sram_digest() == hashlib.sha256(b"").digest()
    w = bytes(i % 251 for i in range(c.vlanes))
    c.write_words(5, 0x40, w)
    c.write_words(3, 0x10, bytes(c.vlanes))
    assert c.sram_digest() == hashlib.sha256(bytes([5]) + (0x40).to_bytes(4, "little") + w).digest()


# ---- IssChip over the binding ----


class Programs:
    """A `ProgramSet` of one bucket: the three patched reads. The output word is read as 80 float32 logits."""

    def __init__(self, reads):
        self.reads = reads
        self.info = ChipInfo(layers=2, vocab=80, hidden=64, heads=2, num_blocks=4, grid_c=(64,))

    def buckets(self):
        return [(1, 1, 64)]

    def program(self, bucket):
        from opentsp_vllm.chip import BucketProgram

        r = self.reads
        return BucketProgram(
            name="reads",
            elf=r.elf,
            patch_table=patch_columns(r.table),
            outputs=[(tp.DSTS[0], tp.DST_ADDR[0], 1)],
            max_cycles=2000,
            site_values=lambda plan: r.values([tp.LO + (plan.lanes[0].position + i) % 16 for i in range(3)]),
            decode=lambda words, plan: np.frombuffer(words, dtype="<f4").reshape(1, -1).copy(),
        )

    def pool_regions(self, block):
        return [(20, 0x1000 + 2 * block, 2), (21, 0x1000 + 2 * block, 2)]


def issc(reads, **kw):
    c = IssChip("ref", programs=Programs(reads), **kw)
    reads.preload(c._chip)
    return c


def plan_of(position, token=7):
    from opentsp_vllm.step_plan import PlanRequest, plan_step

    return plan_step(
        [PlanRequest("r", list(range(token, token + position + 1)), [1], position, 1)],
        vocab=80,
        grid_c=(64,),
        patch_kinds=("wpe",),
    )


def test_iss_chip_satisfies_the_chip_protocol(reads):
    c = issc(reads)
    assert isinstance(c, Chip)
    assert c.info.vocab == 80 and c.failure is None
    c.warm_up()
    res = c.step(plan_of(3))
    assert res.logits.dtype == np.float32 and res.logits.shape == (1, 80)
    assert res.logits.tobytes() == reads.data[(tp.SRCS[0], tp.LO + 3)]
    assert res.cycles > 0 and res.diagnostics == ()
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.step(plan_of(3))


def test_iss_chip_in_both_modes_gives_the_same_logits_and_pool(reads):
    results = {}
    for mode in ("warm", "replay"):
        c = issc(reads, mode=mode)
        c.zero_pool()
        out = [c.step(plan_of(k)).logits.tobytes() for k in (0, 5, 9, 15, 2)]
        c.copy_blocks([(1, 2)])
        c.zero_blocks([3])
        results[mode] = (out, c._chip.sram_digest(), c._chip.stats()["cycles_total"])
    assert results["warm"] == results["replay"]


def test_iss_chip_pool_operations_go_through_the_aperture(reads):
    c = issc(reads)
    n = c._chip.vlanes
    for b in (1, 2, 3):
        for s, a, k in Programs(reads).pool_regions(b):
            c._chip.write_words(s, a, bytes([b]) * (k * n))
    c.copy_blocks([(1, 2), (2, 3)])
    for b in (2, 3):
        for s, a, k in Programs(reads).pool_regions(b):
            assert c._chip.read_words(s, a, k) == bytes([1]) * (k * n), "copies are applied in order: 1 -> 2, then 2 -> 3"
    c.zero_blocks([1])
    for s, a, k in Programs(reads).pool_regions(1):
        assert c._chip.read_words(s, a, k) == bytes(k * n)
    c.zero_pool()
    for b in range(5):
        for s, a, k in Programs(reads).pool_regions(b):
            assert c._chip.read_words(s, a, k) == bytes(k * n)
    with pytest.raises(ValueError, match="null block"):
        c.zero_blocks([0])


def test_iss_chip_failure_is_recorded_and_reraised(reads, tmp_path):
    class Bad(Programs):
        def program(self, bucket):
            p = super().program(bucket)
            p.max_cycles = 3
            return p

    c = IssChip("ref", programs=Bad(reads), failures=tmp_path / "failures")
    assert c.failure is None
    with pytest.raises(TspChipError, match="not done"):
        c.step(plan_of(1))
    assert c.failure and "not done" in c.failure


def test_iss_chip_refused_patch_value_is_a_recorded_failure(reads, tmp_path):
    class Bad(Programs):
        def program(self, bucket):
            p = super().program(bucket)
            p.site_values = lambda plan: reads.values([tp.HI + 5, tp.LO, tp.LO])
            return p

    c = issc(reads)
    c.step(plan_of(1))
    c = IssChip("ref", programs=Bad(reads), failures=tmp_path / "failures")
    reads.preload(c._chip)
    with pytest.raises(TspChipError, match="patch refused.*rd0") as e:
        c.step(plan_of(1))
    assert isinstance(e.value.__cause__, _tsp_iss.PatchError)
    assert c.failure and "patch refused" in c.failure


def test_a_chip_without_failures_option_or_programs_root_never_defaults_to_the_users_home(monkeypatch, tmp_path):
    """An explicit scratch cache owns the default failure directory, even when
    the caller runs the rest of the suite against a shared program cache."""
    scratch = tmp_path / "cache"
    monkeypatch.setenv("OPENTSP_VLLM_CACHE", str(scratch))
    home = os.path.expanduser("~/.cache/opentsp/vllm")
    where = os.path.abspath(IssChip("ref")._failures_dir())
    assert where == str(scratch / "failures")
    assert not where.startswith(home + os.sep), where


def test_iss_chip_refuses_unknown_options_and_a_stats_call_after_close(reads):
    with pytest.raises(TypeError, match="unknown options.*modee"):
        IssChip("ref", programs=Programs(reads), modee="replay")
    c = issc(reads)
    assert c.stats()["steps"] == 0
    c.close()
    with pytest.raises(RuntimeError, match="closed"):
        c.stats()


def test_iss_chip_without_the_binding_names_the_build_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "opentsp_vllm._tsp_iss", None)
    with pytest.raises(ImportError, match="just vllm-binding"):
        IssChip("ref")


def test_importing_the_package_does_not_need_the_binding():
    code = "import sys; sys.modules['opentsp_vllm._tsp_iss'] = None; import opentsp_vllm, opentsp_vllm.chip as c; print(c.IssChip.__name__)"
    env = {**os.environ, "PYTHONPATH": str(REPO / "opentsp-vllm" / "src")}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "IssChip", r.stderr


# ---- a failed step writes a reproducer ----


class DiagnosticPrograms(Programs):
    """One bucket whose program raises two diagnostics (a conformance vector of the `mini` profile); no patch table."""

    def __init__(self):
        self.mini = tprofile.load("mini")
        self.v = mini_vector("directed-020")
        self.info = ChipInfo(layers=2, vocab=80, hidden=64, heads=2, num_blocks=4, grid_c=(64,))

    def program(self, bucket):
        from opentsp_vllm.chip import BucketProgram

        return BucketProgram(
            name="vector",
            elf=vector_object(self.v, self.mini.path),
            outputs=[],
            max_cycles=10_000,
            site_values=lambda plan: None,
            decode=lambda words, plan: np.zeros((1, 80), np.float32),
        )


def reproducers(root):
    return sorted(p for p in pathlib.Path(root).iterdir() if p.is_dir())


def test_a_diagnostic_step_writes_a_reproducer_and_raises_tsp_chip_error(tmp_path):
    c = IssChip("mini", programs=DiagnosticPrograms(), failures=tmp_path / "failures")
    plan = plan_of(1)
    with pytest.raises(TspChipError, match="OTS-E-DEC-003") as e:
        c.step(plan)
    (path,) = reproducers(tmp_path / "failures")
    assert e.value.reproducer == str(path) and str(path) in c.failure and "OTS-E-DEC-003" in c.failure
    man = json.loads((path / "manifest.json").read_text())
    assert man["program"] == "vector" and man["bucket"] == [1, 1, 64] and man["profile"] == "mini"
    assert (man["cycle"], man["queue"], man["code"]) == (20, 1, "OTS-E-DEC-003")
    assert man["diagnostics"][0] == [20, 1, "OTS-E-DEC-003"] and man["error"] == str(e.value).split(" (reproducer")[0]
    assert man["sram_digest"] == c._chip.sram_digest().hex()
    assert man["plan"]["lanes"][0] == {"token": 8, "position": 1, "block_table": [1], "logits": True}
    assert man["spec_version"] == _tsp_iss.SPEC_VERSION and json.loads((path / "patch_values.json").read_text()) == {}


def test_a_step_not_done_and_a_refused_patch_write_reproducers_with_the_patch_values(reads, tmp_path):
    class Bad(Programs):
        def program(self, bucket):
            p = super().program(bucket)
            p.max_cycles = 3
            return p

    c = IssChip("ref", programs=Bad(reads), failures=tmp_path)
    with pytest.raises(TspChipError, match="not done") as e:
        c.step(plan_of(2))
    (path,) = reproducers(tmp_path)
    assert json.loads((path / "patch_values.json").read_text()) == reads.values([tp.LO + 2, tp.LO + 3, tp.LO + 4])
    man = json.loads((path / "manifest.json").read_text())
    assert man["cycle"] is None and man["diagnostics"] == [] and e.value.reproducer == str(path)

    class Refused(Programs):
        def program(self, bucket):
            p = super().program(bucket)
            p.site_values = lambda plan: reads.values([tp.HI + 5, tp.LO, tp.LO])
            return p

    c = IssChip("ref", programs=Refused(reads), failures=tmp_path)
    with pytest.raises(TspChipError, match="patch refused") as e:
        c.step(plan_of(1))
    assert len(reproducers(tmp_path)) == 2 and e.value.reproducer in {str(p) for p in reproducers(tmp_path)}


def test_a_reproducer_that_cannot_be_written_never_hides_the_failure(reads, tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x")

    class Bad(Programs):
        def program(self, bucket):
            p = super().program(bucket)
            p.max_cycles = 3
            return p

    c = IssChip("ref", programs=Bad(reads), failures=blocker / "failures")
    with pytest.raises(TspChipError, match="not done") as e:
        c.step(plan_of(1))
    assert e.value.reproducer is None and "no reproducer" in c.failure


def test_the_failures_directory_defaults_to_the_one_of_the_program_set(reads, tmp_path):
    class Bad(Programs):
        root = str(tmp_path)

        def program(self, bucket):
            p = super().program(bucket)
            p.max_cycles = 3
            return p

    c = IssChip("ref", programs=Bad(reads))
    with pytest.raises(TspChipError):
        c.step(plan_of(1))
    assert len(reproducers(tmp_path / "failures")) == 1


def test_warm_up_checks_the_recorded_cycles_and_keeps_the_reproducer_for_a_mismatch(reads, tmp_path):
    class Recorded(Programs):
        def __init__(self, reads, cycles):
            super().__init__(reads)
            self.cycles = cycles

        def program(self, bucket):
            p = super().program(bucket)
            p.cycles = self.cycles
            return p

    ok = issc(reads, failures=tmp_path)
    ok.warm_up()
    cycles = ok.step(plan_of(3)).cycles
    # a program whose recorded count is the real one passes; another count is a failure
    good = IssChip("ref", programs=Recorded(reads, cycles), failures=tmp_path)
    reads.preload(good._chip)
    good.warm_up()
    bad = IssChip("ref", programs=Recorded(reads, cycles + 1), failures=tmp_path)
    reads.preload(bad._chip)
    with pytest.raises(TspChipError, match="recorded") as e:
        bad.warm_up()
    assert bad.failure and e.value.reproducer and pathlib.Path(e.value.reproducer).is_dir()


def test_every_step_checks_the_recorded_cycle_count_not_only_the_warm_up(reads, tmp_path):
    """The program is static, so its cycle count is the same every step; a doctored count fails the first real step, with a
    reproducer."""

    class Recorded(Programs):
        def __init__(self, reads, cycles):
            super().__init__(reads)
            self.cycles = cycles

        def program(self, bucket):
            p = super().program(bucket)
            p.cycles = self.cycles
            return p

    cycles = issc(reads).step(plan_of(3)).cycles
    good = IssChip("ref", programs=Recorded(reads, cycles), failures=tmp_path / "failures")
    reads.preload(good._chip)
    for position in (3, 4, 5):
        assert good.step(plan_of(position)).cycles == cycles
    bad = IssChip("ref", programs=Recorded(reads, cycles - 1), failures=tmp_path / "failures")
    reads.preload(bad._chip)
    with pytest.raises(TspChipError, match=f"took {cycles} cycles, recorded {cycles - 1}") as e:
        bad.step(plan_of(3))
    assert bad.failure and "reproducer" in bad.failure
    assert pathlib.Path(e.value.reproducer, "patch_values.json").is_file()


def test_a_warm_up_of_a_program_without_a_recorded_cycle_count_says_so(reads, tmp_path, caplog):
    import logging

    c = issc(reads, failures=tmp_path)  # `Programs` has no recorded cycles
    with caplog.at_level(logging.WARNING, logger="opentsp_vllm.chip"):
        c.warm_up()
    assert any("not checked" in r.getMessage() for r in caplog.records), caplog.text


def test_iss_chip_refuses_programs_built_for_another_profile(reads):
    from opentsp_vllm.programs import ProgramsError

    class Built(Programs):
        def __init__(self, reads, profile, sha):
            super().__init__(reads)
            self.profile, self.profile_sha256 = profile, sha

    ref_sha = _tsp_iss.profile_sha256("ref")
    IssChip("ref", programs=Built(reads, "ref", ref_sha))
    with pytest.raises(ProgramsError, match="profile mini"):
        IssChip("ref", programs=Built(reads, "mini", ref_sha))
    with pytest.raises(ProgramsError, match="profile ref"):
        IssChip("ref", programs=Built(reads, "ref", "ab" * 32))
    with pytest.raises(ProgramsError, match="this chip is mini"):
        IssChip("mini", programs=Built(reads, "ref", ref_sha))
