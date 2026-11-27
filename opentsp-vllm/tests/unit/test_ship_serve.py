"""The launcher `python -m opentsp_vllm.serve`: its command line becomes `vllm serve` arguments and the `opentsp` options, a bad option is a message and exit status 2."""

import json

import pytest

from opentsp_vllm import serve


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    from opentsp_vllm.models.gpt2 import four_chip_packages

    monkeypatch.setenv("OPENTSP_GPT2_4_PROGRAMS", str(tmp_path))
    monkeypatch.setenv("OPENTSP_RAS_DOMAIN", "unit-test-domain")
    monkeypatch.setattr(four_chip_packages, "manifests", lambda path: {})
    (tmp_path / "weights.safetensors").touch()
    (tmp_path / "model").mkdir()
    (tmp_path / "model/config.json").touch()
    return str(tmp_path)


def opts(cmd):
    return json.loads(cmd[cmd.index("--additional-config") + 1])["opentsp"]


def test_the_default_is_the_four_chip_pipeline_with_a_small_step(deployment):
    cmd = serve.build(["--tsp-programs", deployment, "--port", "9000"])
    assert cmd[:2] == ["serve", deployment + "/model"] and ["--dtype", "float32"] == cmd[2:4]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "4" and cmd[cmd.index("--max-num-batched-tokens") + 1] == "8"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "9000", "a vLLM option passes through"
    assert cmd[cmd.index("--max-model-len") + 1] == "256"
    assert opts(cmd) == {"chip": "ship", "model": "gpt2", "chips": 4, "kv_chunks": 2, "weights": deployment + "/weights.safetensors",
                         "programs": deployment, "ras_domain": "unit-test-domain"}


@pytest.mark.parametrize("args", [["--host", "0.0.0.0"], ["--host=::1"], ["--uds", "/tmp/opentsp.sock"]])
def test_an_explicit_listen_address_is_preserved(deployment, args):
    cmd = serve.build(["--tsp-programs", deployment, *args])
    assert all(arg in cmd for arg in args)
    assert "127.0.0.1" not in cmd


def test_the_options_of_the_launcher_reach_the_platform(deployment):
    cmd = serve.build(["--tsp-programs", deployment, "--tsp-check", "--max-num-seqs", "2"])
    assert opts(cmd)["check"] is True
    assert opts(cmd)["chips"] == 4
    assert cmd[cmd.index("--max-num-seqs") + 1] == "2" and cmd.count("--max-num-seqs") == 1, "an explicit value is kept"


def test_the_model_answers_to_its_name_and_to_its_directory(deployment):
    """Requests may name GPT-2 or the bundle model directory. An explicit served name is kept."""
    cmd = serve.build(["--tsp-programs", deployment])
    i = cmd.index("--served-model-name")
    assert cmd[i + 1:i + 3] == ["gpt2", deployment + "/model"]
    cmd = serve.build(["--tsp-programs", deployment, "--served-model-name", "mine"])
    assert cmd.count("--served-model-name") == 1 and cmd[cmd.index("--served-model-name") + 1] == "mine"
    cmd = serve.build(["--tsp-programs", deployment, "--served-model-name=mine"])
    assert not any(c == "--served-model-name" for c in cmd) and "--served-model-name=mine" in cmd


@pytest.mark.parametrize("args,msg", [(["--tsp-chips", "7"], "chips"), (["--tsp-model", "gpt3"], "model"), (["--tsp-programs", "/nonexistent"], "gpt2-programs"),
                                      (["--tsp-shutdown-timeout", "nan"], "--tsp-shutdown-timeout"), (["--tsp-shutdown-timeout", "inf"], "--tsp-shutdown-timeout"),
                                      (["--tsp-shutdown-timeout", "-1"], "--tsp-shutdown-timeout"), (["--tsp-programs", "/dev/null"], "not a directory")])
def test_a_bad_option_is_a_message_and_exit_status_2(args, msg, deployment, capsys):
    argv = args if "--tsp-programs" in args else ["--tsp-programs", deployment, *args]
    assert serve.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("opentsp_vllm.serve: error:") and msg in err


@pytest.mark.parametrize(
    "args,flag,value,limit,what",
    [
        (["--max-num-batched-tokens", "64"], "--max-num-batched-tokens", "64", 8, "the 8 token lanes of one step"),
        (["--max-num-batched-tokens=64"], "--max-num-batched-tokens", "64", 8, "the 8 token lanes of one step"),
        (["--max-model-len", "2048"], "--max-model-len", "2048", 256, "the longest context of the pool (16 pages of 16 slots)"),
        (["--max-model-len=2k"], "--max-model-len", "2k", 256, "the longest context of the pool (16 pages of 16 slots)"),
        (["--max-num-seqs", "9"], "--max-num-seqs", "9", 8, "the 8 logit rows of one step"),
    ],
)
def test_a_vllm_option_beyond_the_chip_limits_is_two_lines_and_exit_status_2(args, flag, value, limit, what, deployment, capsys):
    assert serve.main(["--tsp-programs", deployment, *args]) == 2
    lines = capsys.readouterr().err.splitlines()
    assert lines == [
        f"opentsp_vllm.serve: error: {flag} {value} exceeds {what}",
        f"opentsp_vllm.serve: the largest value is {limit}: give {flag} {limit} or less, or leave it out",
    ]


@pytest.mark.parametrize("args", [["--max-num-batched-tokens", "8"], ["--max-model-len", "256"], ["--max-model-len=128"], ["--max-num-seqs", "1"], ["--max-model-len", "-1"], ["--max-num-seqs", "x"]])
def test_a_vllm_option_within_the_limits_or_not_understood_goes_to_vllm_unchanged(args, deployment):
    cmd = serve.build(["--tsp-programs", deployment, *args])
    assert args[0] in cmd


def test_an_explicit_program_bundle_reaches_the_worker(deployment, tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "model").mkdir(parents=True)
    (bundle / "model/config.json").touch()
    (bundle / "weights.safetensors").touch()
    cmd = serve.build(["--tsp-programs", str(bundle)])
    assert opts(cmd)["programs"] == str(bundle)


def test_a_group_that_answers_eperm_is_gone_not_a_crash(monkeypatch):
    """macOS answers EPERM to `killpg(pgid, 0)` for a group whose leader has ended (a launcher that stopped the server while its engine was still starting ended in a traceback)."""
    import os
    import sys

    def eperm(pgid, sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", eperm)
    assert serve._group_alive(12345) is False
    assert serve.supervise([sys.executable, "-c", "pass"], dict(os.environ), grace=1) == 0


def test_short_pool_and_eight_sequences_are_selectable(deployment):
    cmd = serve.build(["--tsp-programs", deployment, "--tsp-kv-chunks", "2", "--max-num-seqs", "8"])
    assert opts(cmd)["kv_chunks"] == 2
    assert cmd[cmd.index("--max-model-len") + 1] == "256"
    assert cmd[cmd.index("--max-num-seqs") + 1] == "8"


def test_short_pool_refuses_a_larger_context_before_startup(deployment):
    with pytest.raises(serve.LauncherError, match="16 pages of 16 slots"):
        serve.build(["--tsp-programs", deployment, "--tsp-kv-chunks", "2", "--max-model-len=257"])


def test_an_unsealed_program_bundle_is_rejected_before_startup(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "model").mkdir(parents=True)
    (bundle / "model/config.json").touch()
    (bundle / "weights.safetensors").touch()
    (bundle / "full").mkdir(parents=True)
    (bundle / "full" / "manifest.json").write_text("{}")
    with pytest.raises(serve.LauncherError, match="expected the sealed four-chip full package"):
        serve.build(["--tsp-programs", str(bundle)])
