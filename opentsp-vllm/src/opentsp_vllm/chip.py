"""ISS backend for the shared serving chip protocol.

The model runner uses the same protocol for the native ISS and the independent
NumPy implementation exercised by the serving tests.
"""

from . import contracts
BucketProgram = contracts.BucketProgram
Chip = contracts.Chip
ChipInfo = contracts.ChipInfo
ChipResult = contracts.ChipResult
TspChipError = contracts.TspChipError
block_records = contracts.block_records
patch_columns = contracts.patch_columns

import importlib
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


class IssChip:
    """`Chip` over the binding (`opentsp_vllm._tsp_iss`). The bucket programs and the pool layout come from `programs`:
    `info` (a `ChipInfo`), `buckets`, `program(bucket)` -> `BucketProgram` and `pool_regions(block)` -> the `(slice, addr, words)`
    regions of one block of every layer. `mode` is the restart mode, `"warm"` or `"replay"` ; default
    `$OPENTSP_VLLM_KV_PERSIST`, else `"warm"`. `failures` is the directory a failed step writes its reproducer under (default
    `<programs.root>/failures`, else `$OPENTSP_VLLM_CACHE/failures`, else `~/.cache/opentsp/vllm/failures`).
    `replay` (default `$OPENTSP_VLLM_REPLAY`, else False) is the opt-in parametrised schedule replay of the paged steps : a step is
    served from the schedule recorded for its bucket program shape, with its site values; the logits, the diagnostics and the cycle counts
    are the normal driver's, a step the gate refuses or a fault during a run takes the normal driver, and a bucket that measures under
    1.2x is served by the normal driver. `stats["replay"]` has the counters. It needs a binding built with the `replay` feature
    (`just vllm-binding` does that).
    """

    def __init__(self, profile="ref", programs=None, mode=None, failures=None, replay=None, **options):
        try:
            _tsp_iss = importlib.import_module("opentsp_vllm._tsp_iss")
        except ImportError as e:
            raise ImportError(
                "IssChip needs the compiled binding opentsp_vllm._tsp_iss, which is not built or not importable "
                f"({e}); build it with `just vllm-binding`"
            ) from e
        if options:
            raise TypeError(f"IssChip: unknown options {sorted(options)} (known: profile, programs, mode, failures, replay)")
        mode = mode or os.environ.get("OPENTSP_VLLM_KV_PERSIST") or "warm"
        self._tsp_iss = _tsp_iss
        self._chip = _tsp_iss.Chip(profile)
        self._chip.restart_mode(mode)
        if replay is None:
            replay = os.environ.get("OPENTSP_VLLM_REPLAY", "").strip().lower() in ("1", "true", "yes", "on")
        if replay:
            self._chip.param_replay(True)
        self._programs = programs
        self._profile = profile
        self._check_programs_are_for_this_chip()
        self._failures = failures
        self._loaded = {}
        self._failure = None
        self._closed = False

    def _check_programs_are_for_this_chip(self):
        """A program set built for another profile is refused at construction : `programs.profile` and
        `programs.profile_sha256`, when it has them, are this chip's."""
        have = getattr(self._programs, "profile", None)
        sha = getattr(self._programs, "profile_sha256", None)
        if have is None and sha is None:
            return
        mine = self._chip.profile_sha256
        if (have is not None and have != self._profile) or (sha is not None and sha != mine):
            from opentsp_vllm.programs import ProgramsError

            raise ProgramsError(
                f"the programs were built for the profile {have} ({str(sha)[:12]}), this chip is {self._profile} ({mine[:12]}); "
                f"build the programs for this profile with `just vllm-programs`"
            )

    @property
    def info(self):
        return self._need_programs().info

    @property
    def failure(self):
        return self._failure

    def load_model(self, source):
        """The shared serving preload or an exported weights directory containing `weights.safetensors`."""
        self._live()
        if source.startswith("synthetic:"):
            raise ValueError("IssChip loads an exported package directory (`just showcase-prepare`); synthetic:SEED is the ReferenceChip's source")
        path = source if os.path.isfile(source) else os.path.join(source, "weights.safetensors")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path}: no weight package; prepare it with `just showcase-prepare`")
        with open(path, "rb") as f:
            data = f.read()
        check = getattr(self._programs, "check_package", None)
        if check:
            check(data)  # the programs were built for this package 
        from .models.tinystories.program_cache import materialize

        manifest = getattr(self._programs, "manifest", {})
        _, device = materialize(data, self.info.layers, self._profile, expected=manifest.get("device_sha256"),
                                package_identity=manifest.get("digest", ""))
        self._guard(self._chip.load_package, device)

    def warm_up(self):
        """Load every bucket program, then run the smallest on padding lanes: no diagnostics, and the recorded cycle count."""
        self._live()
        programs = self._need_programs()
        for bucket in programs.buckets():
            self._program(bucket)
        smallest = min(programs.buckets())
        program = self._program(smallest)
        if program.cycles is None:
            logger.warning(
                "warm-up of bucket %s: the program was built without a recorded run, so its cycle count is not checked "
                "(here or in any step; build with a run to have it)",
                smallest,
            )
        self.step(self._padding_plan(smallest))  # the recorded cycle count is checked by every step

    def zero_pool(self):
        self._live()
        n = self.info.num_blocks
        self._zero(range(n + 1))  # the pool, then the sink block after it

    def zero_blocks(self, block_ids):
        self._live()
        ids = list(block_ids)
        for b in ids:
            self._vllm_block(b)
        self._zero(ids)

    def copy_blocks(self, copies):
        self._live()
        programs = self._need_programs()
        copies = list(copies)
        for src, dst in copies:
            self._vllm_block(src), self._vllm_block(dst)
        for src, dst in copies:
            for (s, a, n), (s2, a2, _) in zip(programs.pool_regions(src), programs.pool_regions(dst)):
                self._guard(self._chip.write_words, s2, a2, self._guard(self._chip.read_words, s, a, n))

    def step(self, plan):
        self._live()
        program = self._program(plan.bucket)
        values = program.site_values(plan)
        try:
            words, cycles, diagnostics = self._chip.step(program.name, values, program.max_cycles)
        except TspChipError as e:
            self._fail(plan, program, e, values)
        except self._tsp_iss.PatchError as e:
            # a patch value the table refuses is a defect of the plan: the same failure as a diagnostic 
            self._fail(plan, program, TspChipError(f"bucket {plan.bucket}: patch refused: {e}"), values, cause=e)
        if program.cycles is not None and cycles != program.cycles:
            # the program is static, so its run time is the same every step: another count is a defect (state-dependent timing)
            self._fail(plan, program, TspChipError(f"bucket {plan.bucket}: the step took {cycles} cycles, recorded {program.cycles}"), values)
        return ChipResult(program.decode(words, plan), cycles, tuple(diagnostics))

    def _fail(self, plan, program, error, values=None, cause=None):
        """Record `error` as the chip's failure, write the reproducer of the step and raise `error` (a `TspChipError`)."""
        path, why = self._write_reproducer(plan, program, error, values)
        error.reproducer = path
        self._failure = f"{error} (reproducer: {path})" if path else f"{error} (no reproducer: {why})"
        raise error from cause

    def _failures_dir(self):
        if self._failures:
            return str(self._failures)
        root = getattr(self._programs, "root", None)
        if root is None:
            from opentsp_vllm.programs import default_root

            root = default_root()
        return os.path.join(root, "failures")

    def _write_reproducer(self, plan, program, error, values):
        """`<failures>/<time>-<program>-<pid>/` with `manifest.json` (the program, the plan, the error and its diagnostics, the digest of the
        SRAM after the failure, the counters) and `patch_values.json` ({site: value}); `(path, None)`, or `(None, why)`: a failure to write
        it never hides the failure being reported."""
        try:
            import numpy as np

            if isinstance(values, (bytes, bytearray)) and program.site_names:
                values = dict(zip(program.site_names, np.frombuffer(values, "<i4").tolist()))
            patch = {str(k): int(v) for k, v in (values or {}).items()}
            base = os.path.join(self._failures_dir(), f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{program.name}-{os.getpid()}")
            path, n = base, 0
            while os.path.exists(path):
                n += 1
                path = f"{base}-{n}"
            os.makedirs(path)
            lanes = [
                {"token": ln.token, "position": ln.position, "block_table": list(ln.block_table), "logits": ln.logits}
                for ln in plan.lanes[: plan.n_real]
            ]
            manifest = {
                "schema": "opentsp.vllm-failure",
                "version": 1,
                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "spec_version": self._tsp_iss.SPEC_VERSION,
                "profile": self._profile,
                "program": program.name,
                "bucket": list(plan.bucket),
                "programs_digest": getattr(self._programs, "digest", None),
                "error": str(error),
                "cycle": error.cycle,
                "queue": error.queue,
                "code": error.code,
                "diagnostics": [list(d) for d in error.diagnostics],
                "sram_digest": self._chip.sram_digest().hex(),
                "stats": self._chip.stats(),
                "plan": {"n_real": plan.n_real, "n_logits": plan.n_logits, "lanes": lanes},
                "patch_values": "patch_values.json",
            }
            with open(os.path.join(path, "patch_values.json"), "w") as f:
                json.dump(patch, f, sort_keys=True)
            with open(os.path.join(path, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=1, sort_keys=True, default=str)
            return path, None
        except Exception as e:  # noqa: BLE001 - the reproducer is a courtesy
            return None, f"{type(e).__name__}: {e}"

    def stats(self):
        self._live()
        return self._chip.stats()

    def close(self):
        self._closed = True
        self._chip = None
        self._loaded.clear()

    def _live(self):
        if self._closed:
            raise RuntimeError("IssChip is closed")

    def _need_programs(self):
        if self._programs is None:
            raise RuntimeError("IssChip has no bucket programs: pass `programs` (the program cache reader)")
        return self._programs

    def _program(self, bucket):
        if bucket not in self._loaded:
            p = self._need_programs().program(bucket)
            self._guard(
                self._chip.load_program,
                p.name,
                p.elf,
                text_package=p.text_package,
                patch_table=p.patch_table,
                outputs=p.outputs,
                blocks=p.blocks,
            )
            self._loaded[bucket] = p
        return self._loaded[bucket]

    def _padding_plan(self, bucket):
        from opentsp_vllm.step_plan import PAD, StepPlan, patch_values

        plan = StepPlan(bucket, (PAD,) * bucket[0], 0, 0, ())
        plan.patches = patch_values(plan, self.info.vocab)
        return plan

    def _vllm_block(self, b):
        n = self.info.num_blocks
        if not 0 < b < n:
            raise ValueError(f"block {b} is the null block or outside the pool of {n} blocks")

    def _zero(self, blocks):
        programs = self._need_programs()
        words = self._chip.vlanes
        for b in blocks:
            for s, a, n in programs.pool_regions(b):
                self._guard(self._chip.write_words, s, a, bytes(n * words))

    def _guard(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TspChipError as e:
            self._failure = str(e)
            raise
        except self._tsp_iss.PatchError as e:
            self._failure = f"patch table refused: {e}"
            raise TspChipError(self._failure) from e
