"""Recovery guidance in the package's error messages: a missing weight package names the recipe that prepares it, and every `just <recipe>` the
package recommends exists in the justfile."""

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]

from opentsp_vllm.chip import IssChip  # noqa: E402


def recipes():
    return set(re.findall(r"^([a-z][a-z0-9-]*)(?:\s[^:]*)?:", (REPO / "justfile").read_text(), re.M))


def test_a_missing_weight_package_names_the_prepare_recipe(tmp_path):
    chip = IssChip("ref")
    with pytest.raises(FileNotFoundError, match=r"no weight package; prepare it with `just showcase-prepare`"):
        chip.load_model(str(tmp_path / "absent"))


def test_every_recommended_recipe_exists():
    known = recipes()
    assert "showcase-prepare" in known
    named = {}
    for path in sorted((REPO / "opentsp-vllm" / "src" / "opentsp_vllm").rglob("*.py")):
        for name in re.findall(r"`just ([a-z][a-z0-9-]*)", path.read_text()):
            named.setdefault(name, path.name)
    assert named, "no recipe recommendation found"
    missing = {name: where for name, where in named.items() if name not in known}
    assert not missing, f"recipes recommended but absent from the justfile: {missing}"


def test_the_showcase_document_states_the_block_copy_path():
    text = (REPO / "docs" / "demos.md").read_text()
    # The fixed physical-KV backend copies each page on chip.
    for needle in ("Gather and Scatter", "two address maps", "does not read or write the KV payload",
                   "Warm restarts retain SRAM between stages", "explicit replay mode copies SRAM through host memory"):
        assert needle in text, needle
