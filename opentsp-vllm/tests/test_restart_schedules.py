"""Compare warm restart, replay and an independent shadow over randomized multi-request ISS schedules.

Each schedule is `driver_paged.random_schedule(seed, steps)`: staggered joins, finishes, prefix hits, a preemption by recompute, a block
copy whose destination a later lane reads and a zeroing. It is run step by step by the driver's scheduler on three chips at once (`schedule_rig.Rig`):
a warm-restart `IssChip`, a replay-restart `IssChip` over the same `ProgramSet`, and a shadow chip rebuilt from the host's record of the SRAM
only. After every step the logits, the diagnostics, the cycle counts and the SHA-256 digests of the SRAM are required to be equal, the state
digests of warm and replay too, and the logits, the sampled tokens and the lanes are required to be those of the bit-exact reference oracle
(`driver_paged.Driver(..., "exact")` on the same schedule). A failing seed is printed so that it replays.

Run in the vLLM environment after `just vllm-binding`: `just vllm-schedules` (2-layer model, 5 seeds), `just vllm-schedules-full` (the
8-layer model, one 6-step schedule; marked `slow`). The buckets are built into `$OPENTSP_VLLM_CACHE` (a scratch directory by default) on first
use. A missing binding or cache fails, never skips.
"""

import json
import os
import pathlib
import sys
import time
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))

import real
import schedule_rig as SR
from opentsp_vllm.chip import BucketProgram, IssChip, TspChipError  # noqa: E402
from opentsp_vllm.programs import ProgramSet, grid  # noqa: E402

from opentsp_vllm.models.tinystories import driver_paged as D

SEEDS = (1, 2, 3, 4, 13)  # printed by a failure; `pytest -k seed4` replays one
STEPS = 12
FULL_SEED, FULL_STEPS = 6, 6
CHECKPOINT = "calum/tinystories-gpt2-3M"
TIMINGS = []  # one row per schedule, printed at the end of the session (`-s`) and written to $OPENTSP_V2A2B_TIMINGS when set


@pytest.fixture(scope="module")
def programs2():
    SR.build_buckets(2, grid((64,)))
    return ProgramSet(real.cache_dir(), layers=2, grid_c=(64,))


def reference_run(model, schedule, steps):
    drv = D.Driver(model.w, schedule.config, "exact", hook=schedule.hook())
    out = drv.run(schedule.submissions, max_steps=steps)
    assert len(drv.steps) == steps
    return drv, out


def run_schedule(seed, steps, model, programs, tmp_path, grid_c=(64,), label=""):
    """The schedule on the rig and on the bit-exact reference driver; asserts everything that holds for any schedule and returns the rig and the driver."""
    schedule = D.random_schedule(seed, steps)
    want, want_tokens = reference_run(model, schedule, steps)
    rig = SR.Rig(programs, model.logical, tmp_path, grid_c, label=label or f"seed {seed}")
    drv = D.Driver(model.w, schedule.config, "exact", hook=schedule.hook())  # the scheduler; its chip is the rig
    drv.chip = rig
    t0 = time.perf_counter()
    tokens = drv.run(schedule.submissions, max_steps=steps)
    wall = time.perf_counter() - t0
    assert len(drv.steps) == steps, f"seed {seed}: {len(drv.steps)} steps ran, not {steps}"
    for i, (got, ref) in enumerate(zip(drv.records, want.records)):
        shape = [(ln.req_id, ln.position, ln.block_table, ln.logits) for ln in got.step.lanes]
        assert shape == [(ln.req_id, ln.position, ln.block_table, ln.logits) for ln in schedule.steps[i].lanes], f"seed {seed} step {i}: the lanes are not the schedule's"
        assert got.step.lanes == ref.step.lanes and got.step.bucket == ref.step.bucket, f"seed {seed} step {i}: the lanes differ from the bit-exact reference driver's"
        assert got.logits.shape == ref.logits.shape and got.logits.tobytes() == ref.logits.tobytes(), f"seed {seed} step {i}: the logits differ from bit-exact reference"
    assert tokens == want_tokens, f"seed {seed}: the tokens differ from the bit-exact reference driver's"
    rig.final_checks()
    sched = drv.scheduler
    TIMINGS.append({
        "label": label or f"seed {seed}", "steps": steps, "wall_seconds": wall, "setup_seconds": rig.setup_seconds,
        "rebuild_seconds": rig.rebuild_seconds, "reference_seconds": sum(r.seconds for r in want.records), "timings": rig.timings,
        "preemptions": sched.preemptions, "replica_hits": sched.replica_hits, "hit_tokens": schedule.prefix_hit_tokens,
        "copies": sum(len(s.block_copies) for s in drv.steps), "zeroings": sum(len(s.zero_blocks) for s in drv.steps),
    })
    return schedule, rig, drv


def test_a_schedule_is_a_pure_function_of_its_seed():
    a, b = D.random_schedule(SEEDS[0], STEPS), D.random_schedule(SEEDS[0], STEPS)
    assert [s.lanes for s in a.steps] == [s.lanes for s in b.steps] and a.config == b.config
    assert len(SEEDS) == 5 and len(a.steps) == STEPS


@pytest.mark.parametrize("seed", SEEDS, ids=[f"seed{s}" for s in SEEDS])
def test_warm_replay_and_the_shadow_agree_after_every_step_and_equal_reference(seed, programs2, tmp_path):
    schedule, rig, drv = run_schedule(seed, STEPS, real.model(2), programs2, tmp_path)
    # the coverage of the schedule on the chips: a join after step 0, a prefix hit, a finish, a preemption, a copy and a zeroing
    assert any(s.hits for s in drv.steps) and schedule.finished_before_end >= 1 and drv.scheduler.preemptions >= 1
    copies = [(n, src, dst) for n, s in enumerate(drv.steps) for src, dst in s.block_copies]
    assert copies and any(s.zero_blocks for s in drv.steps)
    # a copy that a later lane reads: the logits of that lane equal bit-exact reference's (checked for every step above), so the copy was exact
    assert any(dst in ln.block_table for n, _, dst in copies for s in drv.steps[n + 1 :] for ln in s.lanes), f"seed {seed}: no lane reads a copy"
    assert drv.scheduler.replica_hits >= 1
    assert all(t["cycles"] == programs2.program(t["bucket"]).cycles for t in rig.timings)
    assert len({t["bucket"] for t in rig.timings}) >= 2, "the schedule uses more than one bucket"


# ---- the checks of the rig bite ----


def test_a_word_missing_from_the_record_makes_the_shadow_differ(programs2, tmp_path):
    schedule = D.random_schedule(SEEDS[0], STEPS)
    rig = SR.Rig(programs2, real.model(2).logical, tmp_path, label="doctored record")
    real_changes = rig.warm._chip.sram_changes

    def dropping_one_word():
        changes = real_changes()
        return changes[:-1] if changes else changes

    rig.warm._chip.sram_changes = dropping_one_word
    drv = D.Driver(real.model(2).w, schedule.config, "exact", hook=schedule.hook())
    drv.chip = rig
    with pytest.raises(AssertionError, match="SRAM digests differ after the run"):
        drv.run(schedule.submissions, max_steps=2)


def test_a_copy_that_writes_nothing_is_seen(programs2, tmp_path):
    rig = SR.Rig(programs2, real.model(2).logical, tmp_path, label="doctored copy")
    plan = one_lane_plan()  # writes row 0 of block 1 in each layer: the copy has something to copy
    rig.warm.step(plan)
    rig.replay.step(plan)
    rig._sync_program("the test's step")
    rig.warm.copy_blocks = lambda copies: None  # the warm chip's copy writes nothing, the replay chip's does
    with pytest.raises(AssertionError, match="SRAM digests differ after the copy"):
        rig.copy_block(1, 2)


def one_lane_plan():
    from opentsp_vllm.step_plan import PlanRequest, plan_step

    return plan_step([PlanRequest("r", (777,), (1,), 0, 1)], vocab=SR.VOCAB)


# ---- a failure in the middle of a schedule: a reproducer, the error, and the chip goes on ----


class Swappable:
    """A `ProgramSet` that hands out one doctored program for one bucket while `armed`."""

    def __init__(self, inner):
        self._inner, self.armed, self.bucket, self.program_ = inner, False, None, None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def program(self, bucket):
        if self.armed and tuple(bucket) == self.bucket:
            return self.program_
        return self._inner.program(bucket)


def diagnostic_program():
    """A program of the ref profile that raises OTS-E-DEC-003 at cycle 20 on queue 1 (conformance vector `directed-020`)."""
    doc = json.loads((REPO / "opentsp-arch-spec" / "conformance" / "model" / "ref.json").read_text())
    v = next(v for v in doc["vectors"] if v["name"] == "directed-020")
    images = {int(q): [int(u, 16) for u in units.split()] for q, units in sorted(v["text"].items(), key=lambda t: (int(t[0]) == 0xFFFF, int(t[0])))}
    config = real.tprofile.load("ref").path
    elf = real.BP.tsp_as.write_object(types.SimpleNamespace(profile="ref", images=images, externs={}), config)
    return BucketProgram(name="vector", elf=elf, outputs=[], max_cycles=10_000, site_values=lambda plan: None, decode=lambda words, plan: None)


class FaultyChip:
    """A driver chip over one warm `IssChip` that fails once, at step `fail_at`, in the way `kind` says, then runs the same step again."""

    def __init__(self, programs, package, tmp_path, fail_at, kind):
        self.swap = Swappable(programs)
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "weights.safetensors").write_bytes(package)
        self.chip = IssChip("ref", programs=self.swap, failures=tmp_path / "failures")
        self.chip.load_model(str(model_dir))
        self.chip.zero_pool()
        self.chip.warm_up()
        self.fail_at, self.kind, self.error, self.failed_step, self.stats_before = fail_at, kind, None, None, None
        self.last = {}

    def copy_block(self, src, dst):
        self.chip.copy_blocks([(src, dst)])

    def zero_block(self, block):
        self.chip.zero_blocks([block])

    def run(self, step):
        from opentsp_vllm.step_plan import PlanRequest, plan_step

        plan = plan_step([PlanRequest(*r) for r in step.requests], vocab=SR.VOCAB)
        if step.index == self.fail_at:
            self.failed_step = step
            self.stats_before = self.chip.stats()["steps"]
            self.chip._program(plan.bucket)  # loaded, so that the honest program is back after the failure
            program = self.swap._inner.program(plan.bucket)
            if self.kind == "diagnostic":
                self.swap.armed, self.swap.bucket, self.swap.program_ = True, tuple(plan.bucket), diagnostic_program()
                del self.chip._loaded[tuple(plan.bucket)]
            old = (program.cycles, program.max_cycles)
            if self.kind == "cycles":
                program.cycles += 1
            elif self.kind == "limit":
                program.max_cycles = 3
            try:
                with pytest.raises(TspChipError) as e:
                    self.chip.step(plan)
            finally:
                program.cycles, program.max_cycles = old
                self.swap.armed = False
                self.chip._loaded.pop(tuple(plan.bucket), None)
            self.error = e.value
            assert self.chip.failure and "reproducer" in self.chip.failure
        res = self.chip.step(plan)  # the chip is usable: the honest step, as if it had not failed
        self.last = {"cycles": res.cycles}
        return res.logits


@pytest.mark.parametrize("kind", ["diagnostic", "cycles", "limit"])
def test_a_failing_step_in_a_schedule_writes_a_reproducer_raises_and_the_chip_goes_on(kind, programs2, tmp_path):
    seed, steps, fail_at = 2, 6, 3
    schedule = D.random_schedule(seed, steps)
    want, want_tokens = reference_run(real.model(2), schedule, steps)
    chip = FaultyChip(programs2, real.model(2).logical, tmp_path, fail_at, kind)
    drv = D.Driver(real.model(2).w, schedule.config, "exact", hook=schedule.hook())
    drv.chip = chip
    tokens = drv.run(schedule.submissions, max_steps=steps)
    # every step of the schedule, the failed one run again, equals bit-exact reference
    assert tokens == want_tokens
    for i, (got, ref) in enumerate(zip(drv.records, want.records)):
        assert got.logits.tobytes() == ref.logits.tobytes(), (kind, i)
    e, step = chip.error, chip.failed_step
    assert isinstance(e, TspChipError) and e.reproducer and pathlib.Path(e.reproducer).parent == tmp_path / "failures"
    if kind == "diagnostic":
        assert (e.cycle, e.queue, e.code) == (20, 1, "OTS-E-DEC-003") and e.diagnostics[0] == (20, 1, "OTS-E-DEC-003")
    else:
        assert e.code is None and len(e.diagnostics) == 0
        assert ("recorded" if kind == "cycles" else "not done") in str(e)
    man = json.loads((pathlib.Path(e.reproducer) / "manifest.json").read_text())
    assert man["schema"] == "opentsp.vllm-failure" and man["bucket"] == list(step.bucket)
    assert man["program"] == ("vector" if kind == "diagnostic" else real.PC.bucket_name(step.bucket))
    assert man["programs_digest"] == programs2.digest and man["error"] == str(e)
    # the plan of the step: its lanes in order, with the tokens, positions and block tables the scheduler gave
    assert man["plan"]["n_real"] == len(step.lanes) and man["plan"]["n_logits"] == step.n_logits
    assert man["plan"]["lanes"] == [{"token": ln.token, "position": ln.position, "block_table": list(ln.block_table), "logits": ln.logits} for ln in step.lanes]
    # the step: how many runs the chip had completed (the warm-up and the steps before it), which a run that did not finish does not count
    done = chip.stats_before + (0 if kind == "limit" else 1)
    assert man["stats"]["steps"] == done, (man["stats"]["steps"], done)
    assert len(json.loads((pathlib.Path(e.reproducer) / "patch_values.json").read_text())) == (0 if kind == "diagnostic" else len(programs2.program(step.bucket).site_names))
    assert len(list((tmp_path / "failures").iterdir())) == 1


def test_a_reproducer_names_the_digest_of_the_sram_after_the_failure_and_of_nothing_else(programs2, tmp_path):
    schedule = D.random_schedule(2, 6)
    chip = FaultyChip(programs2, real.model(2).logical, tmp_path, 3, "cycles")
    drv = D.Driver(real.model(2).w, schedule.config, "exact", hook=schedule.hook())
    drv.chip = chip
    drv.run(schedule.submissions, max_steps=4)
    man = json.loads((pathlib.Path(chip.error.reproducer) / "manifest.json").read_text())
    # the step failed after its run, and the honest step ran again on the same words: the same SRAM, so the digest is the chip's now
    assert man["sram_digest"] == chip.chip._chip.sram_digest().hex()


# ---- the full model ----


def checkpoint_cached():
    snaps = pathlib.Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub" / "models--calum--tinystories-gpt2-3M" / "snapshots"
    return snaps.is_dir() and any(snaps.iterdir())


@pytest.mark.slow
def test_the_full_model_schedule_warm_replay_shadow_and_reference_agree(tmp_path):
    """One 6-step schedule on the 8-layer model: the real checkpoint when it is cached under ~/.cache/huggingface, else synthetic seed 1."""
    schedule = D.random_schedule(FULL_SEED, FULL_STEPS)
    buckets = sorted({tuple(s.bucket) for s in schedule.steps})
    checkpoint = CHECKPOINT if checkpoint_cached() else None
    if checkpoint:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = SR.build_buckets(8, buckets, checkpoint)
    programs = ProgramSet(real.cache_dir(), layers=8, grid_c=(64,), buckets=buckets, digest=model.digest)
    assert programs.manifest["weights"] == model.source
    _, rig, drv = run_schedule(FULL_SEED, FULL_STEPS, model, programs, tmp_path, label=f"full model ({model.source}) seed {FULL_SEED}")
    TIMINGS[-1]["weights"] = model.source
    assert len(rig.timings) == FULL_STEPS
    assert any(st.block_copies for st in drv.steps) and any(st.zero_blocks for st in drv.steps), "the hooks are exercised on the full model"


@pytest.fixture(scope="session", autouse=True)
def report_timings():
    yield
    if TIMINGS:
        for t in TIMINGS:
            per = [sum(x["warm"] for x in t["timings"]) / t["steps"], sum(x["replay"] for x in t["timings"]) / t["steps"]]
            print(f"\n[schedule] {t['label']}: {t['steps']} steps in {t['wall_seconds']:.1f} s (warm {per[0]:.2f} s, replay {per[1]:.2f} s per step; "
                  f"setup {t['setup_seconds']:.1f} s, rebuild {t['rebuild_seconds']:.1f} s)")
        path = os.environ.get("OPENTSP_V2A2B_TIMINGS")
        if path:
            with open(path, "w") as f:
                json.dump(TIMINGS, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
