"""Run a randomized multi-request restart schedule step by step
on a warm-restart `IssChip`, a replay-restart `IssChip` (the same `ProgramSet`) and an independent shadow chip that is rebuilt from the
host's record only, with every step checked on all of them.

The steps are planned by `opentsp_vllm.step_plan.plan_step` from the `PlanRequest` fields of the scheduler's step (the path the serving runner takes), and the plan's lanes are required to be the scheduler's. `Rig` has the chip interface of `driver_paged.Driver` (`run`,
`copy_block`, `zero_block`, `last`), so the driver's scheduler drives the chips with the tokens they sample themselves.

The record is what a host that owns the SRAM aperture can keep: every aperture write it made (`RecordingChip`) and, after every program,
the words that program changed (`Chip.sram_changes`). The shadow is a new `_tsp_iss.Chip` that only receives that record: no restart, no
step, no program.
"""

import time

import real
from opentsp_vllm import _tsp_iss
from opentsp_vllm.chip import IssChip
from opentsp_vllm.step_plan import PlanRequest, plan_step

VOCAB = 50257


class RecordingChip:
    """The binding's chip with every aperture write logged (`write_words`); everything else is the chip's own."""

    def __init__(self, chip, log):
        self._inner, self._log = chip, log

    def write_words(self, slice, addr, data):
        self._log.append((int(slice), int(addr), bytes(data)))
        return self._inner.write_words(slice, addr, data)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def apply_writes(chip, writes):
    for s, a, data in writes:
        chip.write_words(s, a, data)


def apply_changes(chip, changes):
    for s, a, data in changes:
        chip.write_words(s, a, bytes(data))


def rebuild(record):
    """A new chip whose SRAM is built from the record alone: the entries in order, `("writes", [(slice, addr, bytes)])` through the
    aperture and `("effect", [(slice, addr, bytes)])` the words a program changed."""
    chip = _tsp_iss.Chip("ref")
    for kind, items in record:
        (apply_writes if kind == "writes" else apply_changes)(chip, items)
    return chip


def output_words(chip, program):
    return b"".join(bytes(chip.read_words(s, a, n)) for s, a, n in program.outputs)


class Rig:
    """`programs` a `ProgramSet`, `package` the weight package bytes, `tmp` a directory for the model directory and the reproducers."""

    def __init__(self, programs, package, tmp, grid_c=(64,), label=""):
        self.programs, self.grid_c, self.label = programs, tuple(grid_c), label
        model_dir = tmp / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "weights.safetensors").write_bytes(package)
        self.log = []  # the aperture writes of the warm chip since the last sync
        self.warm = IssChip("ref", programs=programs, mode="warm", failures=tmp / "failures")
        self.replay = IssChip("ref", programs=programs, mode="replay", failures=tmp / "failures")
        self.warm._chip = RecordingChip(self.warm._chip, self.log)
        for c in (self.warm, self.replay):
            c.load_model(str(model_dir))
            c.zero_pool()
            c.warm_up()
        self.shadow = _tsp_iss.Chip("ref")
        self.new_state = _tsp_iss.Chip("ref").state_digest()
        self.probes = 0
        self.record = []
        self.steps = 0
        self.last = {}
        self.timings = []
        self.plans = []
        t0 = time.perf_counter()
        writes = list(self.log)  # the zeroing of the pool: no word changes; the first record of the program's effect is the whole SRAM
        del self.log[:]
        apply_writes(self.shadow, writes)
        self.record.append(("writes", writes))
        self._sync_program("the load and the warm-up")
        self.setup_seconds = time.perf_counter() - t0

    # -- the record ------------------------------------------------------------------------------------------------------------
    def _where(self, what):
        return f"{self.label}: step {self.steps}: {what}"

    def _same_sram(self, what, chips):
        digests = {name: c.sram_digest().hex()[:16] for name, c in chips.items()}
        assert len(set(digests.values())) == 1, self._where(f"the SRAM digests differ after {what}: {digests}")

    def _sync_hooks(self, what):
        """The aperture writes made since the last sync: logged, applied to the shadow, and the shadow's SRAM must be the warm chip's."""
        writes = list(self.log)
        del self.log[:]
        self.warm._chip.sram_changes()  # the host's snapshot now includes the writes; their effect is `writes`
        apply_writes(self.shadow, writes)
        self.record.append(("writes", writes))
        self._same_sram(what, {"warm": self.warm._chip, "replay": self.replay._chip, "shadow": self.shadow})

    def _sync_program(self, what):
        """The words the program changed: recorded, applied to the shadow."""
        changes = self.warm._chip.sram_changes()
        apply_changes(self.shadow, changes)
        self.record.append(("effect", changes))
        self._same_sram(what, {"warm": self.warm._chip, "replay": self.replay._chip, "shadow": self.shadow})
        return len(changes)

    # -- the driver's chip interface ---------------------------------------------------------------------------------------------
    def copy_block(self, src, dst):
        self.warm.copy_blocks([(src, dst)])
        self.replay.copy_blocks([(src, dst)])
        self._sync_hooks(f"the copy of block {src} to {dst}")

    def zero_block(self, block):
        self.warm.zero_blocks([block])
        self.replay.zero_blocks([block])
        self._sync_hooks(f"the zeroing of block {block}")

    def plan(self, step):
        plan = plan_step([PlanRequest(*r) for r in step.requests], vocab=VOCAB, grid_c=self.grid_c)
        want = tuple((ln.req_id, ln.token, ln.position, tuple(ln.block_table), ln.logits) for ln in step.lanes)
        got = tuple((ln.req_id, ln.token, ln.position, tuple(ln.block_table), ln.logits) for ln in plan.lanes[: plan.n_real])
        assert got == want, self._where(f"plan_step's lanes {got} are not the scheduler's {want}")
        assert plan.bucket == step.bucket and plan.n_logits == step.n_logits, self._where("plan_step chose another bucket")
        return plan

    def run(self, step):
        plan = self.plan(step)
        self.plans.append(plan)
        program = self.programs.program(plan.bucket)
        times = {}
        results = {}
        for name, chip in (("warm", self.warm), ("replay", self.replay)):
            t0 = time.perf_counter()
            results[name] = chip.step(plan)
            times[name] = time.perf_counter() - t0
        warm, replay = results["warm"], results["replay"]
        assert warm.logits.tobytes() == replay.logits.tobytes(), self._where("the logits of warm and replay differ")
        assert warm.diagnostics == () and replay.diagnostics == (), self._where(f"diagnostics {warm.diagnostics} / {replay.diagnostics}")
        assert warm.cycles == replay.cycles == program.cycles, self._where(
            f"cycles {warm.cycles} / {replay.cycles}, the program's recorded count is {program.cycles}"
        )
        assert self.warm._chip.state_digest() == self.replay._chip.state_digest(), self._where("the state digests of warm and replay differ")
        # what a warm restart leaves of the registers is not visible in the next step (a program installs the state it uses): the state
        # right after the restart is a new chip's
        assert self.warm._chip._restart_for_test() == self.new_state, self._where("the state after the warm restart is not a new chip's")
        self.probes += 1
        t0 = time.perf_counter()
        n_changed = self._sync_program(f"the run of {plan.bucket}")
        words = output_words(self.shadow, program)
        shadow_logits = program.decode(words, plan)
        assert shadow_logits.tobytes() == warm.logits.tobytes(), self._where("the logits read from the shadow's SRAM differ")
        times["shadow"] = time.perf_counter() - t0
        self.timings.append(
            {"step": self.steps, "bucket": tuple(plan.bucket), "cycles": warm.cycles, "changed_words": n_changed, **times}
        )
        self.steps += 1
        self.last = {"cycles": warm.cycles}
        return warm.logits

    def final_checks(self):
        """The whole record rebuilt into a new chip: the warm chip's SRAM, from the record alone."""
        t0 = time.perf_counter()
        fresh = rebuild(self.record)
        assert fresh.sram_digest() == self.warm._chip.sram_digest(), f"{self.label}: the SRAM rebuilt from the whole record differs"
        self.rebuild_seconds = time.perf_counter() - t0
        stats = self.warm.stats(), self.replay.stats()
        assert stats[0]["restarts_warm"] == self.steps + 1 + self.probes and stats[0]["restarts_replay"] == 0, stats[0]
        assert stats[1]["restarts_replay"] == self.steps + 1 and stats[1]["restarts_warm"] == 0, stats[1]
        return fresh


def build_buckets(layers, buckets, checkpoint=None):
    """The buckets that are not yet in the cache, built in parallel (`build_programs.py`, one process each)."""
    import os
    import subprocess
    import sys

    cache = real.cache_dir()
    model = real.PC.Model(layers, 1, checkpoint) if checkpoint else real.model(layers)
    todo = [b for b in buckets if not os.path.isfile(os.path.join(real.PC.bucket_dir(cache, model, b), "program.json"))]
    if todo:
        script = os.path.join(real.BP.__file__)
        cmd = [sys.executable, script, "--layers", str(layers), "--cache", cache, "--jobs", str(min(6, len(todo)))]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        for b in todo:
            cmd += ["--bucket", real.PC.bucket_name(b)]
        subprocess.run(cmd, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    return model
