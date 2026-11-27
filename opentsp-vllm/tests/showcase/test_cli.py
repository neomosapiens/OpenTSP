"""The showcase command line without vLLM: argument errors are one-line messages, the serve command carries the real weights and the limits."""

import argparse
import json
import socket

from opentsp_vllm.showcase import cli
import pytest
from opentsp_vllm.showcase.session import ShowcaseError


def parse(*argv):
    return cli.parser().parse_args(list(argv))


@pytest.mark.parametrize(
    "argv, message",
    [
        (["chat", "--max-tokens", "0"], "--max-tokens must be at least 1, not 0"),
        (["chat", "--temperature", "5"], "--temperature must be between 0 and 2"),
        (["chat", "--top-k", "-1"], "--top-k must be between 0 and"),
    ],
)
def test_a_bad_sampling_flag_names_the_flag(argv, message):
    with pytest.raises(ShowcaseError, match=message):
        cli.check_settings(parse(*argv))


def test_main_turns_a_bad_flag_into_exit_code_2_and_a_message(capsys):
    assert cli.main(["chat", "--max-tokens", "0"]) == 2
    assert capsys.readouterr().err == "error: --max-tokens must be at least 1, not 0\n"


def test_the_seed_flag_takes_a_number_or_none():
    assert parse("chat", "--seed", "7").seed == 7
    assert parse("chat", "--seed", "none").seed is None
    assert parse("chat").seed == 1
    for bad in ("abc", "-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli.seed_arg(bad)


def test_an_unprepared_showcase_names_the_remedy(tmp_path):
    with pytest.raises(SystemExit, match="just showcase-prepare"):
        cli.B.check_prepared(tmp_path)


def prepared(tmp_path):
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "config.json").write_text("{}")
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "manifest.json").write_text("{}")
    return tmp_path


def test_the_serve_command_has_the_real_weights_the_limits_and_the_model_name(tmp_path):
    root = prepared(tmp_path)
    command = cli.serve_command(root, port=8123)
    assert command[1:5] == ["-m", "vllm.entrypoints.cli.main", "serve", str(root / "model")]
    text = " ".join(command)
    for expected in ("--served-model-name tinystories", "--max-model-len 128", "--max-num-seqs 4", "--max-num-batched-tokens 8", "--port 8123", "--enable-chunked-prefill"):
        assert expected in text
    options = json.loads(command[command.index("--additional-config") + 1])["opentsp"]
    assert options == dict(chip="tiny", programs=str(root / "serving"), max_context=128)


def test_serve_refuses_a_port_that_is_taken():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        with pytest.raises(ShowcaseError, match=f"127.0.0.1:{port} is already in use"):
            cli.check_port_free("127.0.0.1", port)


@pytest.mark.parametrize(
    "argv, message",
    [
        (["serve", "--port", "65536"], "--port must be between 1 and 65535, not 65536"),
        (["serve", "--port", "0"], "--port must be between 1 and 65535, not 0"),
        (["chat", "--seed", "18446744073709551616"], "--seed must be at most 9223372036854775807"),
        (["batch", "--seed", "9223372036854775808"], "--seed must be at most 9223372036854775807"),
        (["batch", "-n", "-1"], "-n must be at least 0, not -1"),
        (["chat", "--prompt", "Left\rRight"], "--prompt contains a control character"),
        (["batch", "--prompt", "ok", "--prompt", "a\x1b[2Jb"], "--prompt contains a control character"),
    ],
)
def test_a_bad_option_exits_2_before_the_model_loads(argv, message, capsys, monkeypatch):
    def no_load(root):
        raise AssertionError("the model was loaded before the options were checked")

    monkeypatch.setattr(cli, "load", no_load)
    monkeypatch.setattr(cli.B, "check_prepared", no_load)
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and message in err and "Traceback" not in err


def test_the_largest_seed_and_port_are_accepted():
    cli.check_options(parse("chat", "--seed", str(2**63 - 1)))
    cli.check_options(parse("serve", "--port", "65535"))
    cli.check_options(parse("batch", "-n", "0"))
