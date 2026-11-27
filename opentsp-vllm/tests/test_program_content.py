"""The reader checks the content of a bucket before a step runs. A structurally valid `bindings.npz` whose vocabulary
map was altered (two entries swapped: the values stay a permutation) moved the highest logit to another token with the same cycles and
no diagnostic (`probe_wrong_logits.py`); a file whose bytes differ from the SHA-256 recorded at publication is refused
too. The buckets are copies of the shared cache's 2-layer P(1, 1, 64) bucket in a temporary directory; the originals stay unchanged."""

import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))

from opentsp_vllm import serving_artifacts
from opentsp_vllm.programs import ProgramSet, ProgramsError, bucket_name  # noqa: E402

B = (1, 1, 64)


@pytest.fixture(scope="module")
def source():
    """The shared entry the runtime reads for the 2-layer model's P(1, 1, 64), and the SHA-256 of its files."""
    import real

    entry = pathlib.Path(real.bucket_path(B)).parent
    shas = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (entry / bucket_name(B)).iterdir()}
    yield entry
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (entry / bucket_name(B)).iterdir()} == shas


def copy(source, tmp_path):
    dst = tmp_path / "cache" / source.name
    dst.mkdir(parents=True)
    shutil.copy(source / "manifest.json", dst / "manifest.json")
    shutil.copytree(source / bucket_name(B), dst / bucket_name(B))
    return tmp_path / "cache", dst / bucket_name(B)


def load(root):
    return ProgramSet(str(root), layers=2, grid_c=(64,), buckets=[B]).program(B)


def rewrite_index(path, change):
    with np.load(path) as z:
        a = {k: z[k] for k in z.files}
    change(a)
    np.savez(path, **a)
    # Verify semantic validation independently of the transport checksum.
    serving_artifacts.seal(path.parent, json.loads((path.parent / "program.json").read_bytes()),
                           payload=serving_artifacts.ELF_PAYLOAD)


def test_an_unchanged_copy_loads(source, tmp_path):
    root, _ = copy(source, tmp_path)
    assert load(root).name == "P1_1_64"


@pytest.mark.parametrize("a,b", [(2402, 257), (2402, 1110)])
def test_a_vocabulary_map_with_two_entries_swapped_is_refused(source, tmp_path, a, b):
    root, d = copy(source, tmp_path)

    def swap(z):
        v = z["vocab_index"]
        z["vocab_index"] = np.where(v == a, b, np.where(v == b, a, v))

    rewrite_index(d / "bindings.npz", swap)
    with pytest.raises(ProgramsError, match="bindings.npz.*vocab_index"):
        load(root)


def test_a_vocabulary_map_that_is_not_a_permutation_is_refused(source, tmp_path):
    root, d = copy(source, tmp_path)

    def dup(z):
        z["vocab_index"] = z["vocab_index"].copy()
        z["vocab_index"].flat[1] = z["vocab_index"].flat[0]

    rewrite_index(d / "bindings.npz", dup)
    with pytest.raises(ProgramsError, match="bindings.npz.*vocab_index"):
        load(root)


def test_a_moved_byte_plane_is_refused(source, tmp_path):
    root, d = copy(source, tmp_path)

    def move(z):
        z["plane_word"] = z["plane_word"].copy()
        z["plane_word"][0, 0, 0] += 1

    rewrite_index(d / "bindings.npz", move)
    with pytest.raises(ProgramsError, match="bindings.npz.*plane_word"):
        load(root)


def test_a_file_that_differs_from_its_recorded_sha256_is_refused(source, tmp_path):
    root, d = copy(source, tmp_path)
    assert load(root).name == "P1_1_64"
    elf = bytearray((d / "program.elf").read_bytes())
    elf[len(elf) // 2] ^= 1
    (d / "program.elf").write_bytes(bytes(elf))
    with pytest.raises(ProgramsError, match="program.elf.*sha256"):
        load(root)
