"""The opt-in demo trace carries allocation facts and no private model inputs."""
import json
from types import SimpleNamespace as NS

from opentsp_vllm.serving_observer import Trace


def test_demo_trace(tmp_path):
    file = tmp_path / "demo.jsonl"
    tag = "a" * 32
    a = f"cmpl-opentsp-demo-{tag}-0-private-engine-suffix"
    b = "a-private-external-request-name"
    blocks = [NS(block_id=i, ref_cnt=refs, block_hash=key) for i, (refs, key) in
              enumerate(((0, None), (2, b"private-prefix"), (1, None), (0, b"retained"), (0, None)))]
    manager = NS(block_pool=NS(blocks=blocks), get_block_ids=lambda r: ([1, 2] if r == a else [1],))
    requests = {a: NS(num_prompt_tokens=18, num_computed_tokens=20),
                b: NS(num_prompt_tokens=32, num_computed_tokens=20)}
    scheduler = NS(kv_cache_manager=manager, requests=requests)
    output = NS(num_scheduled_tokens={a: 1, b: 4}, scheduled_new_reqs=[NS(req_id=b, num_computed_tokens=16)])
    trace = Trace(file)
    trace.schedule(scheduler, output)
    trace.settled(scheduler)
    before, after = [json.loads(line) for line in file.read_text().splitlines()]
    assert before["phase"] == "scheduled"
    assert before["requests"][0] == dict(id=tag, prompt=18, filled=19, prefill=0, decode=1, blocks=[1, 2])
    assert before["requests"][1]["prefill"] == 4
    assert before["requests"][1]["prefix"] == 16
    assert before["pages"] == [[1, 2, True], [2, 1, False], [3, 0, True]]
    assert after["phase"] == "settled" and after["requests"][0]["filled"] == 20
    assert "private" not in file.read_text()


def test_multiplexed_request_labels_include_the_prompt_index():
    from opentsp_vllm.serving_observer import request_label

    tag = "b" * 32
    assert [request_label(f"cmpl-opentsp-wave-{tag}-{i}-private") for i in range(8)] == [f"{tag}/{i}" for i in range(8)]


def test_device_projection_joins_waves_and_preserves_output_barriers():
    from pathlib import Path
    from runpy import run_path

    project = run_path(str(Path(__file__).parents[1] / "demo" / "serve.py"))["project_request_cycles"]
    tag = "a" * 32
    a, b = f"{tag}/0", f"{tag}/1"

    def wave(rows, completions, logits, cycles=100):
        return dict(phase="pipeline", row_ids=rows, completion_cycles=completions, logit_rows=logits, cycles=cycles)

    rows = [
        wave(["warmup"], [75], [True]),
        wave([a], [62], [False]),
        wave([a, b], [71, 87], [True, True]),
        wave(["unrelated"], [66], [True]),
        wave([a, b], [68, 87], [True, True]),
    ]
    for index, record in enumerate(rows):
        record["wave_start"] = 100 * index
    assert project(rows, tag) == {
        "0": {"start": 100, "outputs": [300, 500]},
        "1": {"start": 200, "outputs": [300, 500]},
    }
    assert project(rows, "b" * 32) == {}
    assert project([rows[0], *rows[2:]], tag) is None
    rows[2].pop("logit_rows")
    assert project(rows, tag) is None


def test_same_shape_batches_keep_distinct_identity_and_output_ordinals(tmp_path):
    path = tmp_path / "events.jsonl"
    request = "cmpl-opentsp-wave-" + "a" * 32 + "-0-session"
    scheduler = NS(
        kv_cache_manager=NS(get_block_ids=lambda req: ([1],), block_pool=NS(blocks=[])),
        requests={request: NS(num_prompt_tokens=1, num_computed_tokens=1)},
    )
    trace = Trace(path)
    for token in [262, 983]:
        output = NS(num_scheduled_tokens={request: 1}, scheduled_new_reqs=[])
        trace.schedule(scheduler, output)
        trace.output(output, {0: NS(outputs=[NS(request_id=request, new_token_ids=[token], finish_reason=None)])})
    scheduled, first, later, second = [json.loads(line) for line in path.read_text().splitlines()]
    assert scheduled["step"] == first["step"] == 1
    assert later["step"] == second["step"] == 2
    assert output.opentsp_demo_step == 2
    assert [first["tokens"][0]["ordinal"], second["tokens"][0]["ordinal"]] == [1, 2]
    assert [first["tokens"][0]["ids"], second["tokens"][0]["ids"]] == [[262], [983]]


def test_relay_rejects_a_swapped_batch_with_identical_members_and_token(tmp_path):
    from pathlib import Path
    from runpy import run_path

    cls = run_path(str(Path(__file__).parents[1] / "demo" / "live_trace.py"))["LiveTrace"]
    for swapped in (False, True):
        path = tmp_path / str(swapped)
        path.touch()
        tag = "a" * 32
        cursor = cls(path, tag, False)
        records = []
        for step in (1, 2):
            span = dict(id=tag, ids=[55], ordinal=step)
            records.extend([
                dict(phase="scheduled", step=step, requests=[dict(id=tag, decode=1, prefill=0)]),
                dict(phase="pipeline", step=step, row_ids=[tag], logit_rows=[True]),
                dict(phase="sampled", step=step, tokens=[span]),
            ])
        for ordinal in (1, 2):
            records.append(dict(phase="tokens", step=3 - ordinal if swapped else ordinal,
                                tokens=[dict(id=tag, ids=[55], ordinal=ordinal)]))
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        list(cursor.events())
        chunk = {"choices": [{"index": 0, "text": " x x", "token_ids": [55, 55]}]}
        output, = cursor.annotate(chunk)
        assert output["verified"] is not swapped
        assert output["batches"] == ([] if swapped else [1, 2])
        assert chunk["choices"][0]["text"] == " x x"
