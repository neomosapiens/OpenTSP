#!/usr/bin/env python3
"""Tests of the serving kernels `embed` and `lm_head`: the expectations of the cases are bit-exact reference's own functions (`paged_reference.embed_chip`, `reference.embed`,
`head_logits_chip`), the tables are the serve layout's images, and the programs run bit-exact reference-exact on `tsp_sim` and the ISS,
with the lookups' patch sites set to tokens other than the placeholders'.

usage: TSP_SIM=target/spec/tsp_sim [TSP_ISS=target/release/tsp-iss] [OTS_ASM=target/release/ots-asm]
       python3 tests/programs/gen/test_serve_kernels.py
"""
import json
import os
import random
import subprocess
import sys
import unittest
import warnings

import numpy as np

warnings.simplefilter("ignore", ResourceWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build  # noqa: E402
import model_serve as ms  # noqa: E402
import test_patch  # noqa: E402
from opentsp_tools.programs import data
from opentsp_tools import patch
from opentsp_tools import profile as tprofile
from opentsp_tools.programs import ref
from opentsp_tools.pager import CycleView

TSP_SIM = os.environ.get("TSP_SIM") or os.path.join(tprofile.ROOT, "target", "spec", "tsp_sim")
TSP_ISS = os.environ.get("TSP_ISS") or os.path.join(tprofile.ROOT, "target", "release", "tsp-iss")
OTS_ASM = os.environ.get("OTS_ASM") or os.path.join(tprofile.ROOT, "target", "release", "ots-asm")
RUN = os.path.join(tprofile.ROOT, "tests", "programs", "run.py")
P = tprofile.load("ref")
LAYOUT, Reference, ReferenceP, _ = ms.showcase()


def generated(name):
    files = build.generate([name], ["ref"], None)
    return files, json.loads(files[os.path.join(*name.split("/")) + ".ref.case.json"])


class Tables(unittest.TestCase):
    """The tables the cases bind are the serve layout's images: they decode back to `wte`."""

    def test_embedding_parts_decode_to_wte(self):
        for sizes in (ms.FULL_SIZES, ms.REDUCED_SIZES):
            plan = LAYOUT.plan(LAYOUT.load_profile("ref"), mode="serve")
            w = ms.make_weights(sizes.vocab, 3, (0, 7), (1,))
            tensors = {n: ("u8", words) for n, (_, _, words) in ms.packed_tables(LAYOUT, sizes, w.wte, plan).items()}
            got = LAYOUT.decode_embedding(tensors, sizes.vocab)
            self.assertEqual(got.tobytes(), w.wte.tobytes())

    def test_head_table_decodes_to_wte(self):
        for head in (ms.FULL_HEAD, ms.REDUCED_HEAD):
            w = ms.head_weights(head["vocab"], 3)
            tensors = {f"hd{s}": ("u8", words) for s, words in ms.head_tensors(LAYOUT, w.wte, head["installs"], P.vlanes).items()}
            if head["installs"] == LAYOUT.HEAD_INSTALLS:
                self.assertEqual(LAYOUT.decode_head(tensors, head["vocab"]).tobytes(), w.wte.tobytes())
            else:
                rows = [LAYOUT.decode_matrix_words([list(tensors[f"hd{s}"][1][LAYOUT.NLW_HEAD * (i // 2):LAYOUT.NLW_HEAD * (i // 2 + 1)])
                                                    for s in (LAYOUT.WEST_W if i % 2 == 0 else LAYOUT.EAST_W)], 160, 320)
                        for i in range(head["installs"])]
                got = np.concatenate(rows).reshape(-1, 64)[:head["vocab"]]
                self.assertEqual(got.tobytes(), w.wte.tobytes())


class ServeImages(unittest.TestCase):
    """`model_serve` re-implements the packing loops of the cases; they must not drift from `layout.build_serve_images`."""

    def test_the_tables_of_the_cases_are_the_images_of_the_serve_layout_byte_for_byte(self):
        from opentsp_vllm.models.tinystories import weights as W
        cfg = LAYOUT.serve_config(2)
        plan = LAYOUT.plan(LAYOUT.load_profile("ref"), mode="serve", layers=2)
        w = W.synthetic(cfg, seed=1)
        mine = ms.make_weights(cfg.vocab, 3, ms.FULL_SPECIAL_TOKENS, ms.SPECIAL_POSITIONS)
        w.wte, w.wpe = mine.wte, mine.wpe
        tensors, _ = LAYOUT.build_serve_images(cfg, plan, w)
        self.assertEqual(cfg.vocab, ms.FULL_SIZES.vocab)
        k, _ = ms.build_embed(P, "x/embed", "x", ms.FULL_SIZES, ms.FULL_LOOKUPS[:1], ms.FULL_SPECIAL_TOKENS, seed=3)
        bound = {name: (s, a, n) for name, s, a, n in k.b.prog.externs if not name.startswith("emm")}   # emm: the maps
        self.assertEqual(len(bound), 6 * 2 + 4)             # em and ez of 6 slices, wp of 4
        for name, (s, a, n) in bound.items():
            self.assertEqual(a, plan.by_name[name].addr, name)
            self.assertEqual(bytes(k.weights[name][1]), b"".join(tensors[name][1]), name)
        head = ms.head_tensors(LAYOUT, mine.wte, LAYOUT.HEAD_INSTALLS, P.vlanes)
        for s, words in head.items():
            self.assertEqual(b"".join(words), b"".join(tensors[f"hd{s}"][1]), f"hd{s}")
        self.assertEqual(sorted(f"hd{s}" for s in head), sorted(n for n in tensors if n.startswith("hd")))


class HeadCases(unittest.TestCase):
    """What the head cases observe."""

    def expected(self):
        w = ms.head_weights(ms.REDUCED_HEAD["vocab"], 1)
        h = ms.hidden_rows(ms.REDUCED_HEAD["rows"], 1)
        return w, h, ms.head_expected(LAYOUT, w, h, ms.REDUCED_HEAD["installs"], P.vlanes)

    def entry(self, exp, r, token):
        rho, k = divmod(token, 5)
        return exp[(r, rho // LAYOUT.HEAD_CHUNK, k)][rho % LAYOUT.HEAD_CHUNK]

    def test_negative_zero_sums_are_in_the_expectation_and_the_neighbour_decides(self):
        _, _, exp = self.expected()
        for t in range(15, 20):
            self.assertEqual(self.entry(exp, 2, t).tobytes(), np.float32(-0.0).tobytes(), t)    # five -0 weight rows
        for t in range(20, 25):
            self.assertEqual(self.entry(exp, 2, t).tobytes(), np.float32(0.0).tobytes(), t)     # one +0 product
        for t in range(25, 30):
            self.assertEqual(self.entry(exp, 2, t).tobytes(), np.float32(-0.0).tobytes(), t)    # negative weights and -0
        self.assertEqual(self.entry(exp, 2, 100).tobytes(), np.float32(0.0).tobytes())          # an ordinary row: +0

    def test_the_overflow_row_is_all_non_finite_and_the_rounding_row_is_finite(self):
        w, h, exp = self.expected()
        rows = {r: np.array([self.entry(exp, r, t) for t in range(w.cfg.vocab)]) for r in (0, 1, 2)}
        self.assertFalse(np.isfinite(rows[0]).any())
        self.assertTrue(np.isfinite(rows[1]).all())
        self.assertTrue(np.isfinite(rows[2]).all())

    def logits_of(self, w, h16):
        return ms.head_logits_of_f16(LAYOUT, w, h16, ms.REDUCED_HEAD["installs"], P.vlanes)

    def test_the_rounding_row_observes_the_rounding_of_h_to_f16(self):
        """Row 1 carries lanes whose f16 rounding to nearest differs from truncation toward zero (lane 6: 1 + 0.75 2^-10,
        lane 8: the tie 1 + 1.5 2^-10 that goes to the even neighbour above) and from rounding ties away
        from zero (lane 7: the tie 1 + 2^-11 that goes to the even 1.0). Each such lane, alone, changes the row's logits
        under the wrong rounding. (Lanes 2 to 5 round the same way under all three rules; they are in the row for the
        finite range and the overflow edge, not for rounding.)"""
        w, h, exp = self.expected()
        h16 = ref.to_f16(h[1])
        x = h[1].astype(np.float64)
        mine = np.array([self.entry(exp, 1, t) for t in range(w.cfg.vocab)])
        self.assertEqual(self.logits_of(w, h16).tobytes(), mine.tobytes())
        trunc = h16.copy()
        down = np.abs(h16.astype(np.float64)) > np.abs(x)                 # rounded away from zero
        trunc[down] = np.nextafter(h16[down], np.float16(0.0))
        ulp = np.spacing(np.abs(h16)).astype(np.float64)
        tie = np.abs(np.abs(x) - np.abs(h16.astype(np.float64))) == ulp / 2
        away = h16.copy()
        up_tie = tie & (np.abs(h16.astype(np.float64)) < np.abs(x))        # ties rounded down: "away" goes up
        away[up_tie] = np.nextafter(h16[up_tie], np.float16(np.inf) * np.sign(h16[up_tie]))
        edge = np.arange(9)                                                   # the other lanes are ordinary random values
        self.assertEqual(set(np.flatnonzero(trunc[edge] != h16[edge])), {6, 8})
        self.assertEqual(set(np.flatnonzero(away[edge] != h16[edge])), {7})
        for mutant in (trunc, away):
            for lane in edge[mutant[edge] != h16[edge]]:
                one = h16.copy()
                one[lane] = mutant[lane]
                self.assertNotEqual(self.logits_of(w, one).tobytes(), mine.tobytes(), f"lane {lane}")

    def test_the_underflow_row_makes_the_single_products_visible(self):
        """Row 3 is +0 except lane 0 (1e-8, which underflows to +0), lane 1 (3e-8, which rounds up to the smallest subnormal)
        and lane 2 (503.75 2^-24, which rounds up to 504 2^-24): the logits are the sum of the two products. A kernel that
        rounds 1e-8 up, flushes 3e-8 or truncates lane 2 gives other logits. (In row 1 such products vanish in a sum of
        O(1) terms, which is why the edge has its own row.)"""
        w, h, exp = self.expected()
        h16 = ref.to_f16(h[3])
        self.assertEqual(h16[0], 0)
        self.assertEqual(float(h16[1]), 2.0 ** -24)
        self.assertEqual(float(h16[2]), 504 * 2.0 ** -24)
        mine = np.array([self.entry(exp, 3, t) for t in range(w.cfg.vocab)])
        self.assertEqual(self.logits_of(w, h16).tobytes(), mine.tobytes())
        self.assertTrue((mine != 0).sum() > w.cfg.vocab // 2)
        for lane, bad in ((0, np.float16(2.0 ** -24)), (1, np.float16(0.0)), (2, np.float16(503 * 2.0 ** -24))):
            one = h16.copy()
            one[lane] = bad
            self.assertNotEqual(self.logits_of(w, one).tobytes(), mine.tobytes(), lane)

    def test_non_finite_weights_poison_their_packed_row_in_the_nonfinite_case(self):
        """Tokens 31 (+inf), 41 (-inf), 52 (NaN) and 62 (+inf among -0.0 weights) with the +0 hidden row: an infinity times
        the neighbours' +0 is NaN, so the five entries of each packed row are NaN."""
        w = ms.head_weights(ms.REDUCED_HEAD["vocab"], 1, nonfinite=True)
        out = ms.head_logits_of_f16(LAYOUT, w, np.zeros(64, np.float16), ms.REDUCED_HEAD["installs"], P.vlanes)
        for rho in (6, 8, 10, 12):
            self.assertTrue(np.isnan(out[5 * rho:5 * rho + 5]).all(), rho)
        self.assertEqual(int(np.isnan(out).sum()), 20)
        out = ms.head_logits_of_f16(LAYOUT, w, ref.to_f16(np.linspace(-1, 1, 64).astype(np.float32)),
                                    ms.REDUCED_HEAD["installs"], P.vlanes)
        self.assertTrue(np.isinf(out[31]) and np.isinf(out[41]) and np.isnan(out[52]) and np.isinf(out[62]))
        self.assertTrue(np.isnan(out[30]) and np.isnan(out[61]))               # the neighbours of an infinity


class HeadCost(unittest.TestCase):
    """The head cost grows with the number of logit rows L (H(L) = 1 400 + 92 L); 1, 2 and 3 rows build, and the
    reduced 1, 2 and 3-row cases are bit-exact reference-exact on the ISS. L = 4 writes the east plane's results to the second
    logit quad (split=True): the same 92 cycles per row, H(4) = 1 767."""

    FULL_CYCLES = {1: 1493, 2: 1585, 3: 1675}
    FULL_CYCLES_SPLIT = {1: 1493, 2: 1585, 3: 1675, 4: 1767}

    def test_split_head_costs_the_same_per_row_up_to_four_rows(self):
        got = {}
        for rows in (1, 2, 3, 4):
            k = ms.build_lm_head(P, "x/head", "x", paged=True, seed=1, split=True, **dict(ms.FULL_HEAD, rows=rows))
            got[rows] = k.done("lm_head_full")[1]["budget"]["cycles"]
        self.assertEqual(got, self.FULL_CYCLES_SPLIT)

    def test_full_head_cycles_grow_with_the_rows(self):
        got = {}
        for rows in (1, 2, 3):
            k = ms.build_lm_head(P, "x/head", "x", paged=True, seed=1, **dict(ms.FULL_HEAD, rows=rows))
            got[rows] = k.done("lm_head_full")[1]["budget"]["cycles"]
        self.assertTrue(got[1] < got[2] < got[3], got)
        self.assertEqual(got, self.FULL_CYCLES)

    def test_reduced_one_two_three_row_cases_on_the_iss(self):
        import tempfile
        self.assertTrue(os.access(TSP_ISS, os.X_OK), f"ISS not built ({TSP_ISS})")
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "model"))
            names = []
            for rows in (1, 2, 3):
                k = ms.build_lm_head(P, f"model/ts_head_rows{rows}", f"{rows} rows", seed=1, tags=("iss",),
                                     **dict(ms.REDUCED_HEAD, rows=rows))
                text, case = k.done("lm_head")
                with open(os.path.join(d, "model", f"ts_head_rows{rows}.ref.ots"), "w") as f:
                    f.write(text)
                with open(os.path.join(d, "model", f"ts_head_rows{rows}.ref.case.json"), "w") as f:
                    f.write(build.render(case))
                names.append(f"model/ts_head_rows{rows}")
            r = subprocess.run([sys.executable, RUN, "--profile", "ref", "--sim", TSP_SIM, "--iss", TSP_ISS, "--asm", OTS_ASM,
                                "--dir", d, *names], capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout[-1500:] + r.stderr[-800:])
        for n in names:
            self.assertIn(f"PASS {n} [ref] iss", r.stdout)


class Expectations(unittest.TestCase):
    """The dumped words of a case are the twin's result and nothing else."""

    def test_embed_words_are_reference_embed_for_every_lookup(self):
        _, case = generated("model/ts_embed")
        w = ms.make_weights(ms.REDUCED_SIZES.vocab, 1, ms.REDUCED_SPECIAL_TOKENS, ms.SPECIAL_POSITIONS)
        by_addr = {}
        for key, hexw in case["expect"]["words"].items():
            s, a = map(int, key.split(":"))
            by_addr.setdefault(a, {})[s] = bytes.fromhex(hexw)
        self.assertEqual(len(by_addr), len(ms.REDUCED_LOOKUPS))
        for (token, pos), (a, planes) in zip(ms.REDUCED_LOOKUPS, sorted(by_addr.items())):
            with np.errstate(invalid="ignore"):
                want = data.to_planes(ms.pad(Reference.embed(w, token, pos)))
            self.assertEqual([planes[s] for s in sorted(planes)], want, f"token {token} position {pos}")

    def test_the_lookups_cover_the_part_boundaries_and_the_special_patterns(self):
        for sizes, lookups, special in ((ms.FULL_SIZES, ms.FULL_LOOKUPS, ms.FULL_SPECIAL_TOKENS),
                                        (ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS, ms.REDUCED_SPECIAL_TOKENS)):
            located = {t: sizes.locate(t) for t, _ in lookups}
            first_rows = {sizes.part_rows[0] - 1, sizes.part_rows[0], sum(sizes.part_rows[:2]) - 1, sum(sizes.part_rows[:2]),
                          sizes.rows - 1}
            rows = {t // 5 for t, _ in lookups}
            self.assertTrue(first_rows <= rows, (sorted(first_rows - rows), sorted(rows)))
            self.assertEqual({q for q, _, _ in located.values()}, {0, 1, 2})
            self.assertEqual({s for _, _, s in located.values()}, {0, 1, 2, 3, 4})
            self.assertTrue({t for t, _ in lookups} >= set(special))
            self.assertTrue({p for _, p in lookups} >= {0, 127})

    def test_the_head_cases_carry_every_logit_vector(self):
        _, case = generated("model/ts_lm_head")
        rows, installs = ms.REDUCED_HEAD["rows"], ms.REDUCED_HEAD["installs"]
        self.assertEqual(len(case["expect"]["words"]), rows * installs * 5 * 4)


class Sites(unittest.TestCase):
    def test_every_read_address_the_map_and_the_wpe_words_are_sites(self):
        _, case = generated("model/ts_embed")
        names = {s["name"] for s in case["sites"]}
        for i in range(len(ms.REDUCED_LOOKUPS)):
            want = ({f"e{i}.p{q}{pl}" for q in range(3) for pl in "lh"} | {f"e{i}.wpe{j}" for j in range(4)}
                    | {f"e{i}.map{pl}" for pl in range(2)})
            self.assertTrue(want <= names, sorted(want - names))
        self.assertEqual(len(names), 12 * len(ms.REDUCED_LOOKUPS))

    def test_a_value_outside_the_range_of_a_site_is_refused(self):
        k, _ = ms.build_embed(P, "x/embed", "x", ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS[:1], ms.REDUCED_SPECIAL_TOKENS)
        ms.finish_with_sites(k, "embed")
        view = CycleView.of(k.b.prog)
        table = patch.locate(view, P)
        image = patch.to_image(patch.lower(view, P))
        good = ms.site_values(ms.tables(LAYOUT, LAYOUT.plan(LAYOUT.load_profile("ref"), mode="serve"), ms.REDUCED_SIZES),
                              ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS[:1])
        patch.apply(image, table, good)
        for name, bad in (("e0.map0", 9), ("e0.map1", 3), ("e0.p0l", 4097), ("e0.wpe0", 4097 + 128)):
            with self.assertRaisesRegex(patch.PatchError, name):
                patch.apply(image, table, dict(good, **{name: bad}))


class CheckedPatch(unittest.TestCase):
    """The embedding lookup, a program of the serving kernels with twelve patch sites per lookup (read addresses, the
    positional words, the Permute map), patched through `ots-asm patch`: the object is the object of the literal program."""

    def test_ots_asm_patches_the_embed_program_as_the_assembler_would_assemble_it(self):
        sizes, lookups = ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS[:3]
        k, _ = ms.build_embed(P, "x/embed", "x", sizes, lookups, ms.REDUCED_SPECIAL_TOKENS)
        ms.finish_with_sites(k, "embed")
        view = CycleView.of(k.b.prog)
        table = patch.locate(view, P)
        self.assertEqual(len(table), 12 * len(lookups))
        base = test_patch.assemble_object(P, view)
        tb = ms.tables(LAYOUT, LAYOUT.plan(LAYOUT.load_profile("ref"), mode="serve"), sizes)
        rng = random.Random(3)
        for _ in range(4):
            values = ms.site_values(tb, sizes, [(rng.randrange(sizes.vocab), rng.randrange(128)) for _ in lookups])
            patched = patch.apply_object(base, table, view.sites, values, P)
            self.assertEqual(patched, test_patch.assemble_object(P, view, values))
            self.assertNotEqual(patched, base)


class RunPatched(unittest.TestCase):
    """The embed program patched with lookups other than its placeholders': the reduced geometry on both simulators, the
    full-size geometry on the ISS."""

    SETS = 5

    def program(self, sizes, lookups, special):
        k, _ = ms.build_embed(P, "x/embed", "x", sizes, lookups, special)
        ms.finish_with_sites(k, "embed")
        view = CycleView.of(k.b.prog)
        table = patch.locate(view, P)
        base = patch.to_image(patch.lower(view, P))
        preload = {}
        for name, s, a, n in k.b.prog.externs:
            blob = bytes(k.weights[name][1])
            for j in range(n):
                preload[(s, a + j)] = blob[j * P.vlanes:(j + 1) * P.vlanes]
        return k, table, base, preload

    def lookups(self, rng, sizes, n):
        """Tokens at the rows of the part boundaries, the first rows after them (rows 4097..4099 of the full table: local
        words 1..3 of part 1) and the ends, in every segment, and random ones; positions at the ends and random."""
        p0, p1 = sizes.part_rows[:2]
        rows = [0, p0 - 1, p0, p0 + 1, p0 + 2, p0 + 3, p0 + p1 - 1, p0 + p1, sizes.rows - 1]
        edge = [min(5 * r + rng.randrange(5), sizes.vocab - 1) for r in rows]
        return [(rng.choice(edge + [rng.randrange(sizes.vocab)]), rng.choice([0, 127, rng.randrange(128)])) for _ in range(n)]

    def run_all(self, exe, sizes, lookups, special, sets):
        k, table, base, preload = self.program(sizes, lookups, special)
        plan = LAYOUT.plan(LAYOUT.load_profile("ref"), mode="serve")
        tb = ms.tables(LAYOUT, plan, sizes)
        w = ms.make_weights(sizes.vocab, 1, special, ms.SPECIAL_POSITIONS)
        rng = random.Random(5)
        seen = set()
        for _ in range(sets):
            patched = self.lookups(rng, sizes, len(lookups))
            seen |= {sizes.locate(t)[:2] for t, _ in patched}
            units = patch.to_units(patch.apply(base, table, ms.site_values(tb, sizes, patched)))
            out = test_patch.sim_run(exe, P, units, preload, k.dump, max_cycles=400)
            self.assertEqual(out["diags"], [])
            self.assertEqual(out["lints"], [])
            for i, (token, pos) in enumerate(patched):
                with np.errstate(invalid="ignore"):
                    want = data.to_planes(ms.pad(Reference.embed(w, token, pos)))
                got = [out["words"][sa][:P.vlanes] for sa in k.dump[4 * i:4 * i + 4]]
                self.assertEqual(got, want, f"{os.path.basename(exe)} token {token} position {pos}")
        return seen

    def need_iss(self):
        self.assertTrue(os.access(TSP_ISS, os.X_OK), f"ISS not built ({TSP_ISS}); cargo build --release -p tsp-iss")

    def test_on_tsp_sim(self):
        self.run_all(TSP_SIM, ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS, ms.REDUCED_SPECIAL_TOKENS, self.SETS)

    def test_on_the_iss(self):
        self.need_iss()
        self.run_all(TSP_ISS, ms.REDUCED_SIZES, ms.REDUCED_LOOKUPS, ms.REDUCED_SPECIAL_TOKENS, self.SETS)

    def test_full_size_on_the_iss(self):
        """V = 50 257: tokens at rows 4095..4099, 8191..8192 and the last row (parts 0, 1, 2 with their zero words at bank 1,
        word 0x1000), other segments, other positions."""
        self.need_iss()
        seen = self.run_all(TSP_ISS, ms.FULL_SIZES, ms.FULL_LOOKUPS, ms.FULL_SPECIAL_TOKENS, 2)
        self.assertEqual({q for q, _ in seen}, {0, 1, 2})
        self.assertTrue(any(local in (1, 2, 3) for q, local in seen if q == 1))


class Runs(unittest.TestCase):
    def run_cases(self, names, *extra):
        r = subprocess.run([sys.executable, RUN, "--profile", "ref", "--sim", TSP_SIM, "--iss", TSP_ISS, "--asm", OTS_ASM,
                            *extra, *names], capture_output=True, text=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-1000:])
        return r.stdout

    def test_reduced_cases_run_on_both_simulators(self):
        out = self.run_cases(["model/ts_embed", "model/ts_lm_head", "model/ts_lm_head_nonfinite", "model/ts_lm_head_split"])
        self.assertIn("all passed", out)
        for case in ("ts_embed", "ts_lm_head", "ts_lm_head_nonfinite", "ts_lm_head_split"):
            for sim in ("tsp_sim", "iss", "iss-native"):
                self.assertIn(f"PASS model/{case} [ref] {sim}", out)

    def test_full_size_cases_run_on_the_iss(self):
        """V = 50 257 (the parts 4 096 + 4 096 + 1 860 packed rows, 63 installs of the head), built into a scratch directory
        (the cases are on demand: 30 MB of case files)."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            names = ["model/ts_embed_full", "model/ts_lm_head_full", "model/ts_lm_head_full4"]
            files = build.generate(names, ["ref"], None)
            self.assertEqual(len(files), 6)
            os.makedirs(os.path.join(d, "model"))
            for rel, text in files.items():
                with open(os.path.join(d, rel), "w") as f:
                    f.write(text)
            out = self.run_cases(names, "--dir", d)
        self.assertIn("all passed", out)
        for case in ("ts_embed_full", "ts_lm_head_full", "ts_lm_head_full4"):
            for sim in ("iss", "iss-native"):
                self.assertIn(f"PASS model/{case} [ref] {sim}", out)


if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
