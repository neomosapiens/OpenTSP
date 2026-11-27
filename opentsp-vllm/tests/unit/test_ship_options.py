"""The options and the platform limits of chip `ship` (GPT-2 small on the SHIP-style pipeline): what a person who mistypes a value sees."""

import pytest

import vllm_helpers as H
from opentsp_vllm import options

SHIP = {"opentsp": {"chip": "ship", "programs": "/example/four-chip-bundle"}}


@pytest.fixture(scope="module")
def gpt2_dir(tmp_path_factory):
    return H.write_model_dir(tmp_path_factory.mktemp("gpt2"), layers=12, vocab=50257, positions=1024)


def cfg(path, opentsp=None, **kw):
    kw.setdefault("max_model_len", 256)
    return H.engine_config(path, additional_config={"opentsp": dict(SHIP["opentsp"], **(opentsp or {}))}, **kw)


def test_the_defaults_of_chip_ship_are_resident_four_chip_gpt2():
    o = options.parse(SHIP)
    assert (o["model"], o["chips"], o["layers"], o["check"], o["resident"]) == ("gpt2", 4, 12, False, True)
    assert o["kv_chunks"] == 2


def test_a_truncated_model_runs_on_one_chip_per_layer_with_the_head_on_the_host():
    o = options.parse({"opentsp": {"chip": "ship", "layers": 2}})
    assert (o["chips"], o["layers"]) == (2, 2)


@pytest.mark.parametrize(
    "opts, msg",
    [
        ({"chips": 7}, "chips 7: gpt2 requires four chips"),
        ({"chips": "4"}, "must be an integer"),
        ({"programs": None}, "sealed program bundle"),
        ({"resident": False}, "requires resident mode"),
        ({"kv_chunks": 8}, "two KV chunks"),
        ({"model": "gpt2-xl"}, "model must be one of"),
        ({"model": 3}, "must be a string"),
        ({"check": "yes"}, "check must be true or false"),
        ({"resident": "no"}, "resident must be true or false"),
        ({"layers": 9}, "layers must be 1 to 8"),
        ({"layers": 2, "chips": 15}, "does not match opentsp.layers"),
    ],
)
def test_a_bad_option_value_names_the_option_and_what_is_allowed(opts, msg):
    with pytest.raises(ValueError, match=msg):
        options.parse({"opentsp": dict(SHIP["opentsp"], **opts)})


def test_the_ship_options_belong_to_chip_ship_only():
    for key, value in (("model", "gpt2"), ("chips", 15), ("check", True), ("resident", False)):
        with pytest.raises(ValueError, match=f"opentsp.{key} is an option of chip 'ship' only"):
            options.parse({"opentsp": {key: value}})


def test_chip_ship_sets_a_pool_of_sixteen_pages_and_256_positions(gpt2_dir):
    c = cfg(gpt2_dir)
    assert c.cache_config.num_gpu_blocks_override == 17 and c.cache_config.block_size == 16
    assert c.model_config.max_model_len == 256 and c.parallel_config.worker_cls == "opentsp_vllm.worker.TspWorker"


def test_the_tinystories_limits_are_unchanged(gpt2_dir):
    with pytest.raises(ValueError, match="max_model_len 1024 exceeds 128"):
        H.engine_config(gpt2_dir, max_model_len=1024)


@pytest.mark.parametrize(
    "kw, msg",
    [
        (dict(max_num_seqs=9, max_num_batched_tokens=9), "max_num_seqs 9 exceeds 8"),
        (dict(max_num_batched_tokens=64), "max_num_batched_tokens 64 exceeds 8"),
        (dict(num_gpu_blocks_override=18), "18 exceeds the chip's pool of 17 blocks"),
        (dict(num_gpu_blocks_override=16), "cannot hold one request of max_model_len tokens"),
    ],
)
def test_the_platform_rejects_what_the_pipeline_cannot_serve(gpt2_dir, kw, msg):
    with pytest.raises(ValueError, match=msg):
        cfg(gpt2_dir, **kw)


def test_the_whole_pool_is_one_request_of_the_full_context(gpt2_dir):
    # 256 positions are 16 blocks of 16 plus the null block.
    assert cfg(gpt2_dir, num_gpu_blocks_override=17).cache_config.num_gpu_blocks_override == 17


def test_short_pool_accepts_eight_requests_but_only_16_usable_pages(gpt2_dir):
    c = cfg(gpt2_dir, opentsp={"kv_chunks": 2}, max_model_len=256, max_num_seqs=8)
    assert c.cache_config.num_gpu_blocks_override == 17
    assert c.scheduler_config.max_num_seqs == 8
    with pytest.raises(ValueError, match="exceeds 256"):
        cfg(gpt2_dir, opentsp={"kv_chunks": 2}, max_model_len=257)
