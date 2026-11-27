"""The summary of `host_cost.py`: shares, the steady selection (served step without a text switch, not the first request) and the estimates."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from host_cost_summary import summarise  # noqa: E402


def cur(host, run, switches=0, py=0.001):
    return dict(outer_s=host + run, host_s=host, run_s=run, python_s=py, host_py_s=py, binding_call_s=host + run - py, patch_s=0.0, restart_s=0.0,
                image_and_read_s=0.0, site_values_s=0.0, decode_s=0.0, text_switches=switches, cycles=900, patch_word_bytes=320.0,
                text_word_bytes=0.0, bytes_read=3200.0, value_bytes=4.0, image_bytes=1000.0)


def step(request, T, host, switches=0):
    return dict(request=request, bucket=[T, 1, 64], hw_s=1e-6, serve=cur(host, 0.5, switches), base=cur(host / 2, 0.5))


def test_the_shares_and_the_steady_selection():
    steps = [
        step(0, 1, 9e-3),  # first request: not steady
        step(1, 1, 2e-6),  # steady decode: share 2
        step(1, 1, 4e-6),  # steady decode: share 4
        step(1, 1, 90e-6, switches=1),  # text switch: not steady
        step(1, 8, 3e-6),  # steady prefill
    ]
    s = summarise(steps, "serve")
    d = s["decode-steady"]
    assert d["steps"] == 2 and abs(d["share_host"]["mean"] - 3.0) < 1e-9
    assert s["decode-all"]["steps"] == 4
    assert s["prefill-steady"]["steps"] == 1 and abs(s["prefill-steady"]["share_host"]["mean"] - 3.0) < 1e-9
    # the baseline of a step is selected with its served step: a switch in the served step removes the baseline too
    assert summarise(steps, "base")["decode-steady"]["steps"] == 2
    # bytes moved: 320 + 4 + 3200 = 3524 B; at 3.7 GB/s that is 0.9524 us, against 0.9 us of hardware time (900 cycles at 1 us here: hw_us = 1.0)
    assert abs(d["estimate_moved_bytes"] - 3524.0) < 1e-9
    assert abs(d["estimate_moved_share_at_3_7GBps"] - 3524 / 3.7e9 / 1e-6) < 1e-9
