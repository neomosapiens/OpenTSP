"""The showcase session without vLLM and without a chip (a scripted backend): command parsing, settings, streaming order, the messages of
bad input, Ctrl-C, the batch printout."""

import io

import pytest
from scripted_backend import DEFAULT, ScriptedBackend
from opentsp_vllm.showcase.session import Command, Session, Settings, ShowcaseError, parse_line, run_batch


def make(backend=None, **kw):
    out, err = io.StringIO(), io.StringIO()
    backend = backend or ScriptedBackend()
    return Session(backend, out, err, **kw), backend, out, err


# -- parsing


def test_blank_line_is_empty_and_a_plain_line_is_a_prompt():
    assert parse_line("   \n") == ("empty",)
    assert parse_line("Once upon a time\n") == ("prompt", "Once upon a time")
    assert parse_line("  leading spaces stay") == ("prompt", "  leading spaces stay")


def test_double_slash_escapes_a_prompt_that_starts_with_a_slash():
    assert parse_line("//usr/bin is a path") == ("prompt", "/usr/bin is a path")


@pytest.mark.parametrize(
    "line, command",
    [
        ("/max 30", Command("max", 30)),
        ("/tokens 5", Command("max", 5)),
        ("/temp 0.5", Command("temp", 0.5)),
        ("/temperature 0", Command("temp", 0.0)),
        ("/topk 0", Command("topk", 0)),
        ("/top-k 12", Command("topk", 12)),
        ("/seed 7", Command("seed", 7)),
        ("/seed none", Command("seed", None)),
        ("/verbose", Command("verbose", None)),
        ("/verbose on", Command("verbose", True)),
        ("/verbose off", Command("verbose", False)),
        ("/quit", Command("quit")),
        ("/exit", Command("quit")),
        ("/q", Command("quit")),
        ("/help", Command("help")),
        ("/set", Command("set")),
        ("/start", Command("start")),
        ("/MAX 9", Command("max", 9)),
    ],
)
def test_commands(line, command):
    assert parse_line(line) == ("command", command)


@pytest.mark.parametrize(
    "line, message",
    [
        ("/bogus", "unknown command /bogus"),
        ("/max", "needs a value"),
        ("/max ten", "whole number"),
        ("/max 0", "at least 1"),
        ("/temp hot", "needs a number"),
        ("/temp 3", "between 0 and 2"),
        ("/temp -1", "between 0 and 2"),
        ("/topk -2", "between 0 and"),
        ("/seed -1", "between 0 and"),
        ("/seed x", "whole number"),
        ("/verbose maybe", "on or off"),
        ("/quit now", "takes no argument"),
    ],
)
def test_bad_commands_say_what_is_wrong(line, message):
    with pytest.raises(ShowcaseError, match=message):
        parse_line(line)


def test_commands_change_the_settings_and_show_them():
    s, _, out, err = make()
    for line in ("/max 12", "/temp 0", "/topk 0", "/seed none", "/verbose on"):
        assert s.handle(line)
    assert s.settings == Settings(max_tokens=12, temperature=0.0, top_k=0, seed=None, verbose=True)
    assert "max 12 tokens, temperature 0, top-k off, seed random, verbose on" in out.getvalue()
    assert err.getvalue() == ""


def test_a_bad_command_is_a_message_not_an_exception_and_changes_nothing():
    s, _, out, err = make()
    before = s.settings
    assert s.handle("/temp hot")
    assert s.settings == before
    assert err.getvalue().startswith("error: /temp needs a number")
    assert out.getvalue() == ""


def test_quit_ends_the_session():
    s, *_ = make()
    assert s.handle("/quit") is False
    assert s.handle("hello") is True


def test_help_names_every_command():
    s, _, out, _ = make()
    s.handle("/help")
    text = out.getvalue()
    for word in ("/max", "/temp", "/topk", "/seed", "/verbose", "/start", "/quit", "Ctrl-C", "128 positions"):
        assert word in text


# -- one generation


def test_the_text_is_written_piece_by_piece_before_the_next_step():
    seen = []
    holder = {}

    def on_step(n):
        seen.append(holder["out"].getvalue())

    s, backend, out, _ = make(ScriptedBackend(on_step=on_step))
    holder["out"] = out
    summary = s.generate("Once")
    # before step n the stream holds the prompt and the first n pieces
    assert seen == ["Once" + "".join(DEFAULT[:n]) for n in range(len(DEFAULT))]
    assert summary.text == "".join(DEFAULT)
    assert out.getvalue().startswith("Once" + "".join(DEFAULT) + "\n")


def test_streamed_text_equals_the_final_text_and_the_summary_line_reports_the_run():
    s, _, out, _ = make(ScriptedBackend(cycles=2500))
    summary = s.generate("The")
    lines = out.getvalue().splitlines()
    assert lines[0] == "The" + summary.text
    assert summary.tokens == len(DEFAULT) and summary.steps == len(DEFAULT) and summary.cycles == 2500 * len(DEFAULT)
    assert lines[1].startswith("[5 tokens | prompt 3 | 12,500 ISS cycles in 5 steps | ")
    assert lines[1].endswith(" tokens/s | stopped: length]")


def test_the_sampling_settings_reach_the_backend():
    s, backend, *_ = make(settings=Settings(max_tokens=3, temperature=0.9, top_k=5, seed=42))
    s.generate("Hi")
    (rid, ids, sampling), = backend.submitted
    assert ids == [ord("H"), ord("i")]
    assert sampling == dict(temperature=0.9, top_k=5, seed=42, max_tokens=3)


def test_a_prompt_that_is_too_long_is_refused_with_a_message_and_nothing_is_submitted():
    s, backend, out, err = make()
    assert s.handle("x" * 200)
    assert backend.submitted == []
    message = err.getvalue()
    assert message.startswith("error: the prompt is 200 tokens, but the chip's context is 128 positions")
    assert "at most 127" in message and "Traceback" not in message
    assert out.getvalue() == ""
    assert s.handle("ok") and len(backend.submitted) == 1  # the session carries on


def test_a_prompt_of_exactly_the_context_is_refused_and_one_token_less_is_served_with_one_new_token():
    s, backend, _, err = make()
    s.handle("x" * 128)
    assert backend.submitted == [] and "128 tokens" in err.getvalue()
    err.truncate(0), err.seek(0)
    s.handle("x" * 127)
    assert backend.submitted[0][2]["max_tokens"] == 1
    assert "leaving room for 1 new ones: generating at most 1" in err.getvalue()


def test_max_tokens_is_lowered_to_what_the_context_holds_with_a_note():
    s, backend, _, err = make(settings=Settings(max_tokens=100))
    s.handle("y" * 100)
    assert backend.submitted[0][2]["max_tokens"] == 28
    assert "generating at most 28" in err.getvalue()


def test_an_empty_prompt_starts_from_the_end_of_text_token():
    s, backend, out, err = make()
    s.handle("/start")
    assert backend.submitted[0][1] == [backend.bos]
    assert "empty prompt" in err.getvalue()


def test_non_ascii_text_goes_through_unchanged():
    s, backend, out, _ = make(ScriptedBackend(script={"req-1": [" é", " ☕", "!"]}))
    summary = s.generate("café")
    assert summary.text == " é ☕!"
    assert out.getvalue().startswith("café é ☕!")


def test_ctrl_c_aborts_the_request_after_the_current_step_and_the_session_goes_on():
    holder = {}

    def on_step(n):
        if n == 2:
            holder["session"].interrupt()

    s, backend, out, _ = make(ScriptedBackend(on_step=on_step))
    holder["session"] = s
    summary = s.generate("Once")
    assert summary.interrupted and backend.aborted == ["req-1"] and not backend.busy()
    assert summary.text == "".join(DEFAULT[:3])  # the step that was running still delivers its token
    assert "interrupted]" in out.getvalue() and "stopped:" not in out.getvalue()
    again = s.generate("Again")
    assert not again.interrupted and again.text == "".join(DEFAULT)


def test_verbose_shows_a_line_per_step_instead_of_streaming():
    s, _, out, _ = make(settings=Settings(verbose=True))
    s.generate("Hi")
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("  step   1  scheduled tokens 1  logit capacity 1  context capacity 64  1,000 cycles  0.01 s  -> ' there'")
    assert len([ln for ln in lines if ln.startswith("  step")]) == len(DEFAULT)
    assert "".join(DEFAULT) in out.getvalue()  # the whole text once, after the steps


# -- several requests at once


def test_the_batch_prints_each_stream_when_it_completes_and_the_cached_prefix():
    script = {"req-1": ["a", "b", "c"], "req-2": ["d"], "req-3": ["e", "f"]}
    backend = ScriptedBackend(script=script, cached={"req-1": 0, "req-2": 32, "req-3": 32})
    out, err = io.StringIO(), io.StringIO()
    done = run_batch(backend, ["one", "two", "three"], Settings(), out, err)
    lines = out.getvalue().splitlines()
    finished = [ln.split(" finished")[0] for ln in lines if " finished at step " in ln]
    assert finished == ["req-2", "req-3", "req-1"]  # in the order they complete
    assert [d.text for d in done] == ["abc", "d", "ef"]
    assert "[1 tokens, 32 of 3 prompt tokens from the prefix cache" in out.getvalue()
    assert lines[-1].startswith("3 requests, 6 tokens in 3 scheduler steps, 3,000 ISS cycles, ")


def test_a_batch_prompt_that_is_too_long_is_reported_and_the_others_run():
    backend = ScriptedBackend()
    out, err = io.StringIO(), io.StringIO()
    done = run_batch(backend, ["ok", "x" * 300, "fine"], Settings(), out, err)
    assert len(done) == 2
    assert err.getvalue().startswith("error: prompt 2: the prompt is 300 tokens")


def test_the_summary_line_reports_the_prompt_tokens_served_from_the_prefix_cache():
    s, _, out, _ = make(ScriptedBackend(cached={"req-1": 32}))
    summary = s.generate("p" * 40)
    assert summary.num_cached_tokens == 32
    assert "prompt 40 (32 from the prefix cache)" in out.getvalue()


def test_the_summary_gives_the_time_to_the_first_token():
    ticks = iter(range(0, 1000, 3))  # a clock that advances 3 s per reading
    s, _, out, _ = make(ScriptedBackend(prefill_steps=2), clock=lambda: float(next(ticks)))
    summary = s.generate("p" * 20)
    assert summary.first_token_seconds is not None and 0 < summary.first_token_seconds < summary.seconds
    assert f"(first token after {summary.first_token_seconds:.1f} s)" in out.getvalue()


def test_on_a_terminal_the_prefill_shows_its_progress_and_erases_it_before_the_first_token():
    s, _, out, _ = make(ScriptedBackend(prefill_steps=2), color=True)
    s.generate("p" * 20)
    text = out.getvalue()
    assert "prefilling the prompt on the chip: 8/20 tokens" in text and "prefilling the prompt on the chip: 16/20 tokens" in text
    erase = "\0338\033[K"
    plain = text.replace("\033[2m", "").replace("\033[0m", "")
    for piece in ("\0337", erase):
        plain = plain.replace(piece, "")
    assert "".join(DEFAULT) in plain
    after_clear = text.rsplit(erase, 1)[1]
    assert after_clear.startswith("".join(DEFAULT[:1]))  # the first token follows the last erase directly


def test_without_a_terminal_there_is_no_progress_note():
    s, _, out, _ = make(ScriptedBackend(prefill_steps=2))
    s.generate("p" * 20)
    assert "prefilling" not in out.getvalue() and "\033" not in out.getvalue()
