"""The builder's `built` applies the reader's content check, and a forced publication never leaves the live
path missing or the old bucket stranded (`probe_cache.py`). The buckets are small synthetic ones in a temporary
cache: a valid `bindings.npz` and placeholder bytes elsewhere; the shared cache is not touched."""
import errno
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from opentsp_vllm.models.tinystories import paths as _paths

sys.path.insert(0, os.path.join(_paths.ROOT, "opentsp-vllm", "src"))
from opentsp_vllm.models.tinystories import program_cache as PC
from opentsp_vllm import programs as RT

B = (1, 1, 64)
MANIFEST = {"digest": "d" * 64, "schema": PC.SCHEMA}


@pytest.fixture
def model():
    return SimpleNamespace(key="d" * 16, digest=MANIFEST["digest"], manifest=lambda: dict(MANIFEST))


def bucket(cache, model, tag):
    """An unpublished bucket directory with valid content; `tag` tells the generations apart."""
    d = PC.new_bucket_dir(cache, model, B)
    for f in ("program.elf",):
        with open(os.path.join(d, f), "w") as fh:
            fh.write(json.dumps({"tag": tag}))
    np.savez(os.path.join(d, "bindings.npz"), names=np.array(["t0.emb.p0l"]), **PC.output_index())
    PC.write_json(os.path.join(d, "program.json"), {"digest": PC.bucket_digest(model, B), "tag": tag})
    return d


def tag(cache, model):
    return PC.read_json(os.path.join(PC.bucket_dir(cache, model, B), "program.json"))["tag"]


class Reader(threading.Thread):
    """Polls the live path of the bucket and counts the polls that find it missing."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path, self.stop, self.missing, self.polls = path, threading.Event(), 0, 0

    def run(self):
        while not self.stop.is_set():
            self.polls += 1
            self.missing += not os.path.isfile(os.path.join(self.path, "program.json"))
            time.sleep(0.0005)


def test_the_reader_and_the_builder_decode_the_outputs_alike():
    for key, want in PC.output_index().items():
        assert np.array_equal(RT.output_index()[key], want), key


def test_a_published_bucket_is_built_and_records_the_digests_of_its_files(tmp_path, model):
    cache = str(tmp_path)
    assert PC.commit(cache, model, B, bucket(cache, model, "a")) == "published"
    rec = PC.built(cache, model, B)
    assert rec and sorted(rec["files_sha256"]) == sorted(RT.CONTENT_FILES)


@pytest.mark.parametrize("damage", ["patches", "elf", "vocab"])
def test_a_damaged_bucket_is_not_built_and_the_builder_replaces_it(tmp_path, model, damage):
    """The reader refuses a damaged bucket, so `built` says it is not complete and an unforced build replaces it."""
    cache = str(tmp_path)
    PC.commit(cache, model, B, bucket(cache, model, "old"))
    d = PC.bucket_dir(cache, model, B)
    if damage == "patches":
        open(os.path.join(d, "bindings.npz"), "wb").write(b"not a valid npz")
    elif damage == "elf":
        open(os.path.join(d, "program.elf"), "w").write("{}")
    else:
        index = PC.output_index()
        v = index["vocab_index"]
        index["vocab_index"] = np.where(v == 2402, 257, np.where(v == 257, 2402, v))
        np.savez(os.path.join(d, "bindings.npz"), **index)
    assert PC.built(cache, model, B) is None
    assert PC.commit(cache, model, B, bucket(cache, model, "new")) == "published"
    assert tag(cache, model) == "new" and PC.built(cache, model, B)


def test_a_forced_replacement_never_leaves_the_live_path_missing(tmp_path, model, monkeypatch):
    """`commit` exchanges the live directory with the new one: the path is never missing. A pause in every `os.rename`
    would stretch a window between two renames; the exchange has none."""
    cache = str(tmp_path)
    PC.commit(cache, model, B, bucket(cache, model, "old"))
    real = os.rename
    monkeypatch.setattr(PC.os, "rename", lambda a, b: (time.sleep(0.3), real(a, b))[1])
    new = bucket(cache, model, "new")
    reader = Reader(PC.bucket_dir(cache, model, B))
    reader.start()
    time.sleep(0.05)
    assert PC.commit(cache, model, B, new, replace=True) == "published"
    time.sleep(0.05)
    reader.stop.set()
    reader.join()
    assert reader.polls > 20 and reader.missing == 0, f"the bucket was missing in {reader.missing} of {reader.polls} polls"
    assert tag(cache, model) == "new"
    assert sorted(os.listdir(PC.entry_dir(cache, model))) == ["P1_1_64", "P1_1_64.lock", "manifest.json"]


@pytest.mark.parametrize("swap", ["unavailable", "fails"])
def test_a_failed_replacement_leaves_the_old_bucket_in_place(tmp_path, model, monkeypatch, swap):
    """ENOSPC on the rename of the new directory must not strand the old one under `.old-*` with the live path absent."""
    cache = str(tmp_path)
    PC.commit(cache, model, B, bucket(cache, model, "old"))
    new = bucket(cache, model, "new")
    real = os.rename

    def unavailable(a, b):
        raise NotImplementedError("no exchange")

    def fails(a, b):
        raise OSError(errno.ENOSPC, "simulated local disk exhaustion")

    def rename(a, b):
        if a == new:
            raise OSError(errno.ENOSPC, "simulated local disk exhaustion")
        return real(a, b)
    monkeypatch.setattr(PC, "swap_directories", unavailable if swap == "unavailable" else fails)
    monkeypatch.setattr(PC.os, "rename", rename)
    with pytest.raises(OSError, match="simulated"):
        PC.commit(cache, model, B, new, replace=True)
    assert tag(cache, model) == "old" and PC.built(cache, model, B)
    assert [n for n in os.listdir(PC.entry_dir(cache, model)) if ".old-" in n] == []


def test_the_exchange_is_available_here(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "x").write_text("a"), (b / "x").write_text("b")
    PC.swap_directories(a, b)
    assert (a / "x").read_text() == "b" and (b / "x").read_text() == "a"
