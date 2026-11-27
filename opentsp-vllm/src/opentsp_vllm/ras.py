"""Host-side completion, fencing and recovery for one synchronous ISS serving domain."""

from .contracts import DomainFault, PlanRequest

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

log = logging.getLogger(__name__)




def check_completion(reports, error=None):
    """Use the independent management latch even if the program cleared every CSR."""
    reports = [(chip, r) for chip, r in reports if r["events"]]
    if reports:
        log.info("OpenTSP RAS %s", reports)
    if any(r["fatal_count"] for _, r in reports):
        # An unrelated diagnostic or lost completion precludes localized recovery.
        records = getattr(error, "diagnostics", ())
        localized = error is None or getattr(error, "ras_completion", False) or (records and all(d[-1] == "OTS-E-ECC-001" for d in records))
        raise DomainFault("unsafe chip completion", reports, localized=localized) from error
    if error is not None:
        raise error


def probe_word(read, write, width):
    """Destructive address test on a fenced cold instance. Restore the trusted preload."""
    saved = bytes(read())
    if len(saved) != width:
        raise RuntimeError("incomplete SRAM probe read")
    for pattern in (0x00, 0xff, 0x55, 0xaa):
        data = bytes([pattern]) * width
        write(data)
        if read() != data:
            raise RuntimeError("SRAM probe failed")
    write(saved)
    if read() != saved:
        raise RuntimeError("SRAM probe could not restore the preload")


class Journal:
    """One physical/logical domain owner. Recovery history is outside the device reset."""
    @staticmethod
    def path(directory, name):
        return Path(directory) / (hashlib.sha256(name.encode()).hexdigest() + ".sqlite")

    def __init__(self, directory, name, *, maintenance=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = self.path(directory, name)
        if maintenance and not path.is_file():
            raise ValueError(f"unknown RAS domain {name!r}")
        self.lock = open(str(path) + ".lock", "ab")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError(f"RAS domain {name!r} already has an owner") from None
        try:
            self.db = sqlite3.connect(path)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS history (time_ns INTEGER, state TEXT, fault_key TEXT, detail TEXT)")
            row = self.db.execute("SELECT state FROM history ORDER BY rowid DESC LIMIT 1").fetchone()
            if not maintenance and row and row[0] not in ("closed", "cleared"):
                raise RuntimeError(f"RAS domain {name!r} requires operator clearance ({path})")
        except BaseException:
            self.close()
            raise

    def record(self, state, key="", detail=None):
        with self.db:
            self.db.execute("INSERT INTO history VALUES (?,?,?,?)",
                            (time.time_ns(), state, key, json.dumps(detail, sort_keys=True)))

    def attempted(self, key):
        return self.db.execute("""SELECT 1 FROM history WHERE state='recovering' AND fault_key=?
            AND rowid > COALESCE((SELECT MAX(rowid) FROM history WHERE state='cleared'),0) LIMIT 1""",
                               (key,)).fetchone() is not None

    def close(self):
        if hasattr(self, "db"):
            self.db.close()
        self.lock.close()


class ManagedChip:
    """Own the only serving reference to a synchronous backend and its replacement."""
    def __init__(self, create, name, domain=None):
        from functools import partial
        from .ras_device import DeviceProcess

        self.create = partial(DeviceProcess, create)
        self.backend = self.create()
        self.info = self.backend.info
        self.name = name
        self.domain = domain
        self.journal = None
        self.failure = None

    def __getattr__(self, attr):
        return getattr(self.backend, attr)

    def plan_step(self, requests):
        # Plan on the host, including when the quarantined process has exited.
        if self.name == "ship":
            from .ship_plan import PAGE, PAGES_PER_CHUNK, plan_step

            usable_pages = self.info.num_blocks - 1
            chunks, remainder = divmod(usable_pages, PAGES_PER_CHUNK)
            if remainder or self.info.grid_c != (usable_pages * PAGE,):
                raise ValueError("inconsistent SHIP KV geometry reported by the device")
            return plan_step(requests, vocab=self.info.vocab, kv_chunks=chunks)
        from .step_plan import plan_step
        return plan_step(requests, vocab=self.info.vocab, grid_c=self.info.grid_c, patch_kinds=())

    def load_model(self, source):
        self.source = source
        directory = os.environ.get("OPENTSP_RAS_STATE_DIR", "~/.local/state/opentsp/ras")
        domain = self.domain or os.environ.get("OPENTSP_RAS_DOMAIN") or f"{self.name}:{Path(source).resolve()}"
        try:
            self.journal = Journal(os.path.expanduser(directory), domain)
            self.journal.record("starting")
            self.backend.load_model(source)
            self.identity = self.backend.package_identity()
            self.reference = self._canary(self.backend)
            self.journal.record("ready")
        except BaseException:
            self.failure = "startup health or load failed"
            self.backend.close()
            if self.journal:
                self.journal.close()
                self.journal = None
            raise

    @staticmethod
    def _canary(chip):
        # Fresh, private KV page. The exact logits pin the fault-free startup
        # baseline for this model, program and profile, including every stage.
        plan = chip.plan_step([PlanRequest("ras-health", (1,), (1,), 0, 1)])
        logits = chip.step(plan).logits
        import numpy as np
        if logits.shape != (1, chip.info.vocab) or not np.isfinite(logits).all():
            raise RuntimeError("incomplete RAS canary")
        digest = hashlib.sha256(logits.tobytes()).digest()
        chip.clear_canary()
        return digest

    def _run(self, method, *args):
        if self.failure:
            raise DomainFault(self.failure)
        try:
            return getattr(self.backend, method)(*args)
        except (RuntimeError, ValueError, OSError) as exc:
            fault = exc if isinstance(exc, DomainFault) else DomainFault(str(exc))
            try:
                self._fence(fault)
            except Exception as error:
                self.failure = f"quarantined: containment or journal failure: {error}"
                log.exception("OpenTSP RAS fence failed")
            raise DomainFault(f"interrupted request: {fault}; {self.failure or 'domain recovered'}",
                              fault.reports, localized=False) from exc

    def _fence(self, fault):
        self.failure = f"quarantined: {fault}"
        detail = {"reason": str(fault), "reports": fault.reports}
        target = fault.target
        # Conservatively treat another failure anywhere in the same chip and
        # SRAM slice as recurrent, even with a different syndrome or address.
        key = f"sram:{target[0]}:{target[1]}" if target is not None else ""
        try:
            self.journal.record("fenced", key, detail)
        finally:
            # Process exit is the fence acknowledgement in this emulator.
            # Each process owns its complete SRAM, link queues and DMA buffers.
            self.backend.close()
        if target is None or self.journal.attempted(key):
            self.journal.record("quarantined", key, detail)
            return
        self.journal.record("recovering", key, detail)
        try:
            fresh = self.create()
            self.backend = fresh
            fresh.load_model(self.source)
            if fresh.package_identity() != self.identity:
                raise RuntimeError("reload changed the trusted package identity")
            fresh.probe_sram(*target)
            if self._canary(fresh) != self.reference:
                raise RuntimeError("cold canary differs from startup logits")
            self.journal.record("ready", key)
        except Exception as error:
            self.failure = f"quarantined after recovery failure: {error}"
            self.backend.close()
            self.journal.record("quarantined", key, self.failure)
        else:
            self.failure = None

    def step(self, plan):
        return self._run("step", plan)

    def zero_blocks(self, blocks):
        return self._run("zero_blocks", blocks)

    def copy_blocks(self, copies):
        return self._run("copy_blocks", copies)

    def close(self):
        self.backend.close()
        journal, self.journal = self.journal, None
        if journal is not None:
            try:
                if not self.failure:
                    journal.record("closed")
            finally:
                journal.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect a RAS domain or record an offline operator clearance")
    parser.add_argument("domain", help="OPENTSP_RAS_DOMAIN, or the backend name and resolved weights path separated by a colon")
    parser.add_argument("--state-dir", default=os.getenv("OPENTSP_RAS_STATE_DIR", "~/.local/state/opentsp/ras"))
    parser.add_argument("--clear", metavar="SERVICE_RECORD", help="operator's reason after physical inspection or repair")
    args = parser.parse_args()
    if args.clear is not None and not args.clear.strip():
        parser.error("clearance requires a service record")
    directory = os.path.expanduser(args.state_dir)
    if args.clear:
        journal = Journal(directory, args.domain, maintenance=True)
        try:
            journal.record("cleared", detail=args.clear)
        finally:
            journal.close()
    path = Journal.path(directory, args.domain).resolve()
    if not path.is_file():
        parser.error(f"unknown RAS domain {args.domain!r}")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        for row in db.execute("SELECT time_ns,state,fault_key,detail FROM history ORDER BY rowid"):
            print(json.dumps(row))
    finally:
        db.close()


if __name__ == "__main__":
    main()
