"""`showcase-prepare` keeps what is current and replaces what is stale: a weight package written for an older program layout must not be kept
(the programs built next would be paired with it and read the wrong words)."""

import hashlib
from pathlib import Path

import pytest
from opentsp_vllm.models.tinystories.program_cache import write_shared_preload


def test_a_stale_weight_package_is_replaced(tmp_path):
    out = tmp_path / "weights.safetensors"
    out.write_bytes(b"old layout")
    write_shared_preload(out, b"new layout", hashlib.sha256(b"new layout").hexdigest())
    assert out.read_bytes() == b"new layout"
    assert not list(out.parent.glob("*.tmp"))


def test_a_current_weight_package_is_left_untouched(tmp_path):
    out = tmp_path / "weights.safetensors"
    out.write_bytes(b"same")
    before = out.stat().st_mtime_ns
    write_shared_preload(out, b"same", hashlib.sha256(b"same").hexdigest())
    assert out.read_bytes() == b"same"
    assert out.stat().st_mtime_ns == before


def test_prepare_pins_its_exact_package(tmp_path):
    from opentsp_vllm.showcase import prepare

    for key in ("first", "second"):
        pkg = tmp_path / "programs" / key
        pkg.mkdir(parents=True)
        (pkg / "manifest.json").write_text("{}")
        prepare.select_package(tmp_path, key)
        assert (tmp_path / "serving").resolve() == pkg
    assert (tmp_path / "programs" / "first" / "manifest.json").is_file()


def test_incomplete_cached_model_file_is_repaired(tmp_path, monkeypatch):
    from opentsp_vllm.showcase import prepare

    source = tmp_path / "complete"
    source.write_bytes(b"complete source")
    monkeypatch.setattr(prepare, "_cached", lambda repo, name: source)
    model = tmp_path / "model"
    model.mkdir()
    target = model / "config.json"
    target.write_bytes(b"partial")

    prepare.write_model_dir(model)
    assert target.read_bytes() == source.read_bytes()
    stamp = target.stat().st_mtime_ns
    prepare.write_model_dir(model)
    assert target.stat().st_mtime_ns == stamp


def test_interrupted_model_copy_never_publishes_the_partial_file(tmp_path, monkeypatch):
    from opentsp_vllm.showcase import prepare

    monkeypatch.setattr(prepare, "_cached", lambda repo, name: Path(name))

    def fail_during_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("simulated local write failure")

    monkeypatch.setattr(prepare.shutil, "copyfile", fail_during_copy)
    model = tmp_path / "model"
    with pytest.raises(OSError, match="simulated local write failure"):
        prepare.write_model_dir(model)
    assert list(model.iterdir()) == []
