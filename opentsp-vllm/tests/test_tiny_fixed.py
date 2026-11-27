"""Fixed instruction execution, physical-order numerics and KV isolation on the ISS."""

import numpy as np

import reference_loader
from opentsp_vllm.step_plan import PlanRequest
from opentsp_vllm.tiny_chip import TinyChip, physical_mask

reference_loader.load()
from opentsp_vllm.models.tinystories import fixed_package
from opentsp_vllm.models.tinystories import paged_reference as oracle
from opentsp_vllm.models.tinystories import layout
from opentsp_vllm.models.tinystories import weights
from opentsp_tools.programs import ref
from opentsp_tools.programs import ref_wide


def reference(w, keys, values, token, position, table, chains=None):
    """Softmax over all physical slots and two ascending PV chains. Keys and
    values here are test-owned reference arrays, independent of device SRAM."""
    x = w.wte[token].astype(np.float32) + w.wpe[position].astype(np.float32)
    slots = sorted(table[p // 16] * 16 + p % 16 for p in range(position + 1))
    current = table[position // 16] * 16 + position % 16
    for l, lw in enumerate(w.layers):
        _, q, k, v = oracle.project(w.cfg, lw, x)
        keys[l, current], values[l, current] = ref.to_f16(k), v
        scores = np.full((w.cfg.heads, keys.shape[1]), -np.inf, np.float32)
        scores[:, slots] = ref.scores(
            keys[l, slots], q, np.zeros(len(slots), np.float32), ref.head_lane_masks(w.cfg.heads, w.cfg.hidden)
        )
        probs = ref_wide.softmax_pool(scores)
        p = np.repeat(ref.from_f16(ref.to_f16(probs[:, slots])).T, w.cfg.dk, axis=1)
        acc = np.zeros((2, w.cfg.hidden), np.float32)
        for i, slot in enumerate(slots):
            acc[slot % 2] += p[i] * values[l, slot]
        if chains is not None:
            chains[:] = acc
        x = oracle.finish_context(w.cfg, lw, x, acc[0] + acc[1])
    return oracle.head_logits_chip(w, ref.layernorm(x, w.lnf_g, w.lnf_b, w.cfg.ln_eps))


def check_device_copy_boundary(chip, src, dst):
    """Exercise the real stages while rejecting host KV payload access."""
    inner = chip._chip
    slices, addr, _ = chip._regions["slotmaps"]
    writes, steps = [], []

    class CopyBoundary:
        def __getattr__(self, name):
            return getattr(inner, name)

        def read_words(self, *args):
            raise AssertionError("block copy read KV payload through the host")

        def write_words(self, sl, word, data):
            assert sl in slices[:2] and word == addr and len(data) == inner.vlanes
            writes.append((sl, word, len(data)))
            return inner.write_words(sl, word, data)

        def step(self, name, *args):
            assert name.startswith("copy")
            steps.append(name)
            return inner.step(name, *args)

    chip._chip = CopyBoundary()
    try:
        chip.copy_blocks([(src, src)])
        assert not writes and not steps, "identity copy launches no stages"
        chip.copy_blocks([(src, dst)])
    finally:
        chip._chip = inner
    expected = sum(n for sl, _, n in chip._pool_regions(src) if sl % 4 == 0)
    assert len(steps) == expected and len(writes) == 2 * expected
    return dict(stages=len(steps), map_writes=len(writes), map_bytes=sum(w[2] for w in writes), host_payload_reads=0)


def test_physical_pages_and_copy_on_one_chip(tmp_path):
    root = fixed_package.build(tmp_path, layers=2, jobs=2)
    chip = TinyChip(root, 2, (64, 128))
    chip.load_model(root / "weights.safetensors")
    w = weights.synthetic(layout.serve_config(2), 1)
    keys = np.zeros((2, 2048, 64), np.float16)
    values = np.zeros((2, 2048, 64), np.float32)
    tokens = list(range(11, 30))
    # Nonfinite stale words in an adjacent page of an active physical chunk.
    for s, a, n in chip._pool_regions(126):
        chip._chip.write_words(s, a, bytes([255]) * (n * chip._chip.vlanes))
    table = [127, 1]
    for start, count in ((0, 8), (8, 8), (16, 3)):
        result = chip.step(chip.plan_step([PlanRequest("a", tokens[: start + count], table, start, count)]))
        for pos in range(start, start + count):
            expected = reference(w, keys, values, tokens[pos], pos, table)
        np.testing.assert_array_equal(result.logits[0].view("u4"), expected.view("u4"))
    before = [chip._chip.read_words(s, a, n) for s, a, n in chip._pool_regions(127)]
    check_device_copy_boundary(chip, 127, 3)
    after = [chip._chip.read_words(s, a, n) for s, a, n in chip._pool_regions(3)]
    assert before == after and any(any(w) for w in after)
    # Exercise the maximum context and all eight request pages across distinct
    # physical chunks. Compare the actual accumulator chains before narrowing.
    order = [127, 1, 97, 17, 81, 33, 65, 49]
    rng = np.random.default_rng(314)
    for block in order:
        for layer in range(2):
            base = block * 16
            key = rng.normal(size=(16, 64)).astype("<f2")
            value = rng.normal(size=(16, 64)).astype("<f4")
            keys[layer, base : base + 16], values[layer, base : base + 16] = key, value
            ks, ka, _ = chip._regions["keys"]
            vs, va, _ = chip._regions[f"v{layer}"]
            padded_k = np.zeros((16, 320), "<f2")
            padded_k[:, :64] = key
            packed = padded_k.view("u1").reshape(2, 8, 320, 2)
            for j, s in enumerate(ks):
                chip._chip.write_words(s, ka + layer * 256 + block * 2, packed[:, j // 2, :, j % 2].tobytes())
            padded_v = np.zeros((16, 320), "<f4")
            padded_v[:, :64] = value
            planes = padded_v.view("u1").reshape(16, 320, 4)
            for j, s in enumerate(vs):
                chip._chip.write_words(s, va + base, planes[:, :, j].tobytes())
    request = PlanRequest("max", [17] * 128, order, 127, 1)
    result = chip.step(chip.plan_step([request]))
    expected_chains = np.empty((2, 64), "<f4")
    expected = reference(w, keys, values, 17, 127, order, expected_chains)
    np.testing.assert_array_equal(result.logits[0].view("u4"), expected.view("u4"))
    for i, name in enumerate(("even", "odd")):
        slices, addr, _ = chip._regions[name]
        planes = [np.frombuffer(chip._chip.read_words(s, addr, 1), "u1") for s in slices]
        actual = np.stack(planes, axis=1).view("<u4")[:64, 0]
        np.testing.assert_array_equal(actual, expected_chains[i].view("<u4"))
    counts = chip.stats()
    assert counts["patches_applied"] == counts["patch_words"] == counts["replay_words"] == 0


def test_physical_mask_selects_only_visible_rows():
    mask = physical_mask([127, 1], 17)
    assert np.flatnonzero(mask == 0).tolist() == [16, 17, *range(2032, 2048)]
    assert np.all(np.isneginf(mask[18:2032]))
    assert np.all(np.isneginf(mask[:16]))
