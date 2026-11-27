"""Both serving models publish the same file contract."""
import hashlib
import json

import pytest
from opentsp_vllm import serving_artifacts as artifacts


def test_manifest_and_unit_integrity(tmp_path):
    for name in artifacts.PAYLOAD:
        (tmp_path / name).write_bytes(name.encode())
    artifacts.seal(tmp_path, {"chip": 3})
    assert {p.name for p in tmp_path.iterdir()} == set(artifacts.FILES)
    record = json.loads((tmp_path / "program.json").read_bytes())
    artifacts.verify(tmp_path, record)
    assert record["files_sha256"]["program.elf"] == hashlib.sha256(b"program.elf").hexdigest()
    (tmp_path / "sram.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="sram.safetensors"):
        artifacts.verify(tmp_path, record)
    del record["files_sha256"]["sram.safetensors"]
    with pytest.raises(ValueError, match="inventory"):
        artifacts.verify(tmp_path, record)


def test_bucket_uses_the_verified_bytes_after_a_directory_replacement(tmp_path):
    import numpy as np
    from opentsp_vllm import programs

    bucket = (1, 1, 64)
    stage = tmp_path / programs.bucket_name(bucket)
    stage.mkdir()
    (stage / "program.elf").write_bytes(b"published object")
    columns = ("names", "queue", "unit", "nunits", "lo", "width", "vmin", "vmax", "placeholder", "fixed", "stride", "span")
    np.savez(stage / "bindings.npz", **{name: [] for name in columns}, **programs.output_index())
    geometry = {"part_zero_word": 0, "parts": [], "kstride": 1, "vstride": 1, "vocab": 2}
    artifacts.seal(stage, {"geometry": geometry, "cycles": 1, "outputs": [], "blocks": []}, payload=artifacts.ELF_PAYLOAD)
    reader = programs.ProgramSet.__new__(programs.ProgramSet)
    reader.entry = reader.root = str(tmp_path)
    reader._records = {bucket: json.loads((stage / "program.json").read_bytes())}
    reader._programs = {}

    compiled = reader.program(bucket)
    stage.rename(tmp_path / "prior")
    stage.mkdir()
    (stage / "program.elf").write_bytes(b"replacement object")

    assert compiled.elf == b"published object"
    assert compiled.text_package is None
