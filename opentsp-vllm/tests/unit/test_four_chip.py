"""Four-chip admission, package identities and selection are fail-closed."""
import pytest
import numpy as np

from opentsp_vllm import options
from opentsp_vllm.models.gpt2.four_chip_packages import manifests, select_variant


def test_four_chip_requires_compact_resident_package():
    good = dict(chip="ship", chips=4, programs="/bundle", kv_chunks=2)
    shape = options.parse({"opentsp": good})
    assert (shape["layers"], shape["chips"]) == (12, 4)
    for malformed in (dict(kv_chunks=8), dict(resident=False), dict(programs=None)):
        with pytest.raises(ValueError, match="requires resident mode, two KV chunks and a sealed program bundle"):
            options.parse({"opentsp": {**good, **malformed}})


def test_bundle_rejects_unpinned_programs(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed four-chip full"):
        manifests(tmp_path)


def test_selection_uses_page_path_and_general_fallback():
    count = np.zeros((2, 8), dtype=int)
    count[0, 1] = 1
    assert select_variant([(0, 0, 16, count)]) == "single-page"
    count[0, 0] = 1
    assert select_variant([(0, 0, 16, count)]) == "sparse"
    count[1, 0] = 1
    assert select_variant([(0, 0, 128, count)]) == "full"
    rows = []
    for page in range(8):
        count = np.zeros((2, 8), dtype=int)
        count[0, page] = 1
        rows.append((0, 0, page*16, count))
    assert select_variant(rows) == "single-page-pair"
