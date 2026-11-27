"""The binding's `System`: N chips in lock step with the host interface of every chip, held to `tsp_sim --chips N`.

Run in the vLLM environment after `just vllm-binding` (`just vllm-binding-test`). A missing binding fails, never skips. The oracle
is the model simulator (an explicit `TSP_SIM` is used unchanged; only the default `target/spec/tsp_sim` is built by
`tools/ensure_sim.sh` when stale). The caller identifies an explicit oracle's source. Every comparison is of
the chip's `E` line, the dumped `W` words, the `Y` vectors PciSend gave the host, and the `T srf`, `T sram`, `T dma`, `T irq` and
`T diag` events with their cycles. The programs are in `system_rig.py`.
"""

import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "opentsp-vllm" / "src"))
sys.path.insert(0, str(HERE))

import system_rig as rig

from opentsp_vllm import _tsp_iss  # noqa: E402  (a missing binding is a failure: build it with `just vllm-binding`)
from opentsp_vllm.chip import TspChipError  # noqa: E402

System = _tsp_iss.System
PROFILES = ["mini", "ref"]


def test_maximum_cycle_limit_is_not_a_timestamp_sentinel():
    maximum = (1 << 64) - 1
    for limit in (0, 1, maximum - 2, maximum - 1, maximum):
        assert System("mini", 1).run(max_cycles=limit)["reason"] == "done"
    s = System("mini", 1)
    s.csr_write(0, 0x300, 1, maximum)
    assert s.run(max_cycles=maximum - 2)["reason"] == "limit"
    assert s.csr_read(0, 0x300) == 0
    assert s.run(max_cycles=maximum)["reason"] == "done"
    assert s.csr_read(0, 0x300) == 1
    assert s.cycle == 1 << 64
    assert s.cycles(0) == 1 << 64
    assert s.active_ticks(0) == 2
    with pytest.raises(ValueError, match="beyond the u64 timestamp domain"):
        s.csr_write(0, 0x300, 0)


@pytest.fixture(scope="module")
def sim():
    return rig.sim_path()


def agree(s, chips, oracle):
    """The system's view of every chip equals the oracle's (a failure names the chip and the item)."""
    mine = rig.seen_by_system(s, chips)
    for c, (m, o) in enumerate(zip(mine, oracle)):
        for item in ("E", "W", "Y", "dma", "irq", "diag", "srf", "sram"):
            assert m[item] == o[item], f"chip {c}, {item}: system {str(m[item])[:300]} != tsp_sim {str(o[item])[:300]}"


def run_done(s):
    r = s.run(stop_on_notify=False, raise_on_diag=False)
    assert r["reason"] == "done", r
    s.finish(raise_on_diag=False)
    return r


# ---- (a) a 2-chip Send/Receive through System gives the results and cycles of `tsp_sim --chips 2` -----------------------

@pytest.mark.parametrize("profile", PROFILES)
def test_two_chip_send_receive_equals_tsp_sim(sim, profile):
    p = rig.Profile(profile)
    chips, val = rig.send_receive(p)
    oracle = rig.run_oracle(sim, p, chips)
    assert oracle[1]["W"][(44, 50)] == val and oracle[0]["diag"] == [] and oracle[1]["diag"] == [], "the scenario is sound"
    s = rig.new_system(System, p, chips)
    run_done(s)
    assert bytes(s.read_words(1, 44, 50, 1)) == val
    agree(s, chips, oracle)
    # the vector arrived L cycles after it was sent: the receiver's write commits L cycles after the sender's read
    assert s.window == p.L and s.chips == 2 and s.vlanes == p.vl


@pytest.mark.parametrize("profile", PROFILES)
def test_the_window_ends_before_the_earliest_arrival(sim, profile):
    """Chip 0 sends at the first cycle of the run and chip 1 receives at the first cycle the vector can be there: the Receive must see it."""
    p = rig.Profile(profile)
    chips = rig.tight_transmit(p)
    oracle = rig.run_oracle(sim, p, chips)
    assert oracle[1]["diag"] == [] and oracle[1]["W"][(0, 70)][0] == 1, "the scenario is sound: the notification vector arrived"
    s = rig.new_system(System, p, chips)
    run_done(s)
    agree(s, chips, oracle)


def test_the_window_and_the_link_options_are_those_of_the_harness(sim):
    p = rig.Profile("mini")
    chips, val = rig.send_receive(p)
    for kw, args in [
        ({"link_latency": 211}, ("--link-latency", "211")),
        ({"link_latency_seed": 4}, ("--link-latency-seed", "4")),
        ({"link_jitter_seed": 3}, ("--link-jitter-seed", "3")),
    ]:
        # a different latency moves the Receive: the program of the nominal latency finds nothing and reports TIM-001, as in the oracle
        oracle = rig.run_oracle(sim, p, chips, *args)
        s = rig.new_system(System, p, chips, **kw)
        run_done(s)
        agree(s, chips, oracle)
    assert System("mini", 2, link_latency=211).window == 211
    with pytest.raises(ValueError, match="exclude one another"):
        System("mini", 2, link_latency=211, link_latency_seed=1)


@pytest.mark.parametrize("chips,src,link,dst,far", [(8, 0, 2, 3, None), (8, 6, 3, 2, None), (5, 1, 0, 2, None), (16, 0, 7, 8, 7)])
def test_a_node_of_several_chips_is_wired_as_tsp_sim_wires_it(sim, chips, src, link, dst, far):
    p = rig.Profile("mini")
    cs, val = rig.far_send(p, chips, src, link, dst, far)
    oracle = rig.run_oracle(sim, p, cs)
    assert oracle[dst]["W"][(44, 50)] == val and all(o["diag"] == [] for o in oracle), "the scenario is sound"
    s = rig.new_system(System, p, cs)
    run_done(s)
    agree(s, cs, oracle)
    assert bytes(s.read_words(dst, 44, 50, 1)) == val


def test_objects_come_from_bytes_a_path_or_a_manifest(tmp_path):
    from opentsp_tools import refasm
    tsp_as = refasm.load()

    p = rig.Profile("mini")
    cs, val = rig.send_receive(p)
    objects = [c.object() for c in cs]
    (tmp_path / "chip1.elf").write_bytes(objects[1])
    manifest = tsp_as.write_manifest("mini", objects, p.path)
    by_manifest = System("mini", 2)
    by_manifest.load_manifest(manifest)
    (tmp_path / "m.otsm").write_bytes(manifest)
    by_path = System("mini", 2)
    by_path.load_manifest(tmp_path / "m.otsm")
    mixed = System("mini", 2)
    mixed.load_object(0, objects[0])
    mixed.load_object(1, tmp_path / "chip1.elf")
    for s in (by_manifest, by_path, mixed):
        s.write_words(0, 44, 1, val)
        s.run(raise_on_diag=False)
        assert bytes(s.read_words(1, 44, 50, 1)) == val
    with pytest.raises(ValueError, match="the manifest holds 2 objects, the system has 3 chips"):
        System("mini", 3).load_manifest(manifest)
    with pytest.raises(ValueError, match="manifest is for profile"):
        System("ref", 2).load_manifest(manifest)
    with pytest.raises(ValueError, match="chip.elf|No such file"):
        System("mini", 2).load_object(0, tmp_path / "chip.elf")


# ---- (b) PciReceive to PciSend echo on one chip, and on two chips ---------------------------------------------------------

@pytest.mark.parametrize("profile", PROFILES)
def test_echo_on_one_chip_equals_tsp_sim(sim, profile):
    p = rig.Profile(profile)
    chips = rig.echo_one_chip(p, vectors=3)
    oracle = rig.run_oracle(sim, p, chips)
    assert [v for _, v in oracle[0]["Y"]] == [p.val(k) for k in range(3)], "the scenario is sound: the host gets its vectors back"
    s = rig.new_system(System, p, chips)
    run_done(s)
    assert [bytes(v) for _, v in s.egress(0)] == [p.val(k) for k in range(3)]
    agree(s, chips, oracle)


@pytest.mark.parametrize("profile", PROFILES)
def test_echo_on_two_chips_equals_tsp_sim(sim, profile):
    p = rig.Profile(profile)
    chips = rig.echo_two_chips(p)
    oracle = rig.run_oracle(sim, p, chips)
    assert [v for _, v in oracle[1]["Y"]] == [p.val(3)] and oracle[0]["Y"] == [], "the scenario is sound"
    s = rig.new_system(System, p, chips)
    run_done(s)
    assert s.egress(0) == [] and [bytes(v) for _, v in s.egress(1)] == [p.val(3)]
    agree(s, chips, oracle)


# ---- (c) back pressure and diagnostics --------------------------------------------------------------------------------------

def test_dma_back_pressure_waits_and_drops_nothing(sim):
    """The model's DMA engine waits while the ingress FIFO is full (`T dma wait`) and drops nothing; it raises no diagnostic. The host
    sees it as `ingress` and `dma_events`, and every vector reaches the program once the program takes them."""
    p = rig.Profile("mini")
    n = p.depth + 4
    c = rig.Chip(p)
    for k in range(n):
        c.dma.append((0, p.val(k)))
        recv = 400 + k * p.cpv
        c.at(recv, rig.ERX, rig.hio(rig.PCI_RECEIVE, rig.W(24 + k % 8)))
        c.drain(44, 10 + k, 24 + k % 8, "W", p.rx_gap, recv + p.d_c2c)
    c.dumps = [(44, 10 + k) for k in range(n)]
    oracle = rig.run_oracle(sim, p, [c])
    assert ("wait" in [w for _, w in oracle[0]["dma"]]) and oracle[0]["diag"] == [], "the scenario is sound"
    s = rig.new_system(System, p, [c])
    # before the program takes any, the FIFO is full and the head request waits
    r = s.run(max_cycles=399)
    assert r["reason"] == "limit" and r["cycle"] == 400
    assert s.ingress(0) == (p.depth, 4, True)
    assert s.dma_events(0)[-1][1] == "wait"
    run_done(s)
    assert s.ingress(0) == (0, 0, False)
    agree(s, [c], oracle)
    assert [bytes(s.read_words(0, 44, 10 + k, 1)) for k in range(n)] == [p.val(k) for k in range(n)]


def test_a_diagnostic_is_an_exception_with_the_models_code_and_cycle(sim):
    """PciReceive with an empty ingress FIFO is OTS-E-TIM-001 (as Receive); run raises `TspChipError` with the code, cycle, queue and chip,
    and the same diagnostic is the oracle's."""
    p = rig.Profile("mini")
    c0, c1 = rig.Chip(p), rig.Chip(p)
    c1.at(rig.T0, rig.ERX, rig.hio(rig.PCI_RECEIVE, rig.W(24)))
    oracle = rig.run_oracle(sim, p, [c0, c1])
    assert oracle[1]["diag"] == [(rig.T0, rig.ERX, "OTS-E-TIM-001")], "the scenario is sound"
    s = rig.new_system(System, p, [c0, c1])
    with pytest.raises(TspChipError) as e:
        s.run()
    err = e.value
    assert (err.code, err.cycle, err.queue, err.chip) == ("OTS-E-TIM-001", rig.T0, rig.ERX, 1)
    assert err.diagnostics == [(rig.T0, rig.ERX, "OTS-E-TIM-001")] and err.chip_diagnostics == [(1, rig.T0, rig.ERX, "OTS-E-TIM-001")]
    assert "chip 1" in str(err) and "OTS-E-TIM-001" in str(err) and str(rig.T0) in str(err)
    # the diagnostics stay readable and the system can run on: to the end, with the oracle's results
    assert s.diagnostics(1) == [(rig.T0, rig.ERX, "OTS-E-TIM-001")] and s.diagnostics(0) == []
    run_done(s)
    agree(s, [c0, c1], oracle)


def test_finish_reports_a_stranded_queue(sim):
    """A program whose Receive never gets its vector leaves text no event can dispatch: model_finish raises OTS-E-IQ-004 at `finish`
    when the run is done, as the harness's `E` line has it."""
    p = rig.Profile("mini")
    c0 = rig.Chip(p)
    c0.at(10, rig.VXMQ, rig.hio(rig.HOST_WAIT, 0, 3))  # parks for good: nobody rings
    c0.at(11, rig.VXMQ, rig.hio(rig.HOST_NOTIFY, 0, 5))
    oracle = rig.run_oracle(sim, p, [c0])
    s = rig.new_system(System, p, [c0])
    assert s.run()["reason"] == "done"
    assert s.irqs(0) == [] and s.csr_read(0, 48) == 0
    s.finish(raise_on_diag=False)
    agree(s, [c0], oracle)


# ---- the host acts while the chips run ----------------------------------------------------------------------------------------

def test_a_dma_written_after_a_stopped_run_equals_the_same_vector_in_the_script(sim):
    """The host stops the system with a cycle limit, writes DMA vectors (two of them out of order) and the run resumes: the results are the
    ones of a script that had the same vectors at the same cycles."""
    p = rig.Profile("mini")
    c = rig.Chip(p)
    for k in range(2):
        recv = 300 + k * p.cpv
        c.at(recv, rig.ERX, rig.hio(rig.PCI_RECEIVE, rig.W(24 + k)))
        c.drain(44, 10 + k, 24 + k, "W", p.rx_gap, recv + p.d_c2c)
    c.dumps = [(44, 10), (44, 11)]
    scripted = rig.Chip(p)
    scripted.recs, scripted.dumps = list(c.recs), list(c.dumps)
    scripted.dma = [(200, p.val(1)), (190, p.val(0))]
    oracle = rig.run_oracle(sim, p, [scripted])
    assert oracle[0]["diag"] == [] and oracle[0]["W"][(44, 10)] == p.val(0), "the scenario is sound: the earlier request goes first"
    s = rig.new_system(System, p, [c])
    r = s.run(max_cycles=150)
    assert (r["reason"], r["cycle"]) == ("limit", 151)
    assert s.dma_write(0, p.val(1), cycle=200) == 200
    assert s.dma_write(0, p.val(0), cycle=190) == 190
    run_done(s)
    agree(s, [c], oracle)


def test_notify_stops_the_run_and_it_resumes(sim):
    """HostNotify is an interrupt to the host: run returns at the end of the window it was delivered in, with the vector, chip and cycle
    of `T irq`; the next run carries on to the end, with the oracle's results."""
    p = rig.Profile("mini")
    chips, val = rig.send_receive(p)
    chips[1].at(rig.T0 + p.L + 60, rig.VXMQ, rig.hio(rig.HOST_NOTIFY, 0, 3))
    oracle = rig.run_oracle(sim, p, chips)
    assert len(oracle[1]["irq"]) == 1, "the scenario is sound"
    s = rig.new_system(System, p, chips)
    r = s.run(raise_on_diag=False)
    assert r["reason"] == "notify" and r["notifies"] == [(1, oracle[1]["irq"][0][0], 3)]
    assert r["cycle"] > oracle[1]["irq"][0][0] and s.csr_read(1, 48) == 1 << 3 and s.csr_read(1, 50) == 1
    again = s.run(raise_on_diag=False)
    assert again["reason"] == "done" and again["notifies"] == []
    s.finish(raise_on_diag=False)
    agree(s, chips, oracle)
    # with stop_on_notify=False the interrupt does not stop the run
    t = rig.new_system(System, p, chips)
    assert t.run(stop_on_notify=False, raise_on_diag=False)["reason"] == "done"


def test_doorbell_and_csr_writes_of_the_host(sim):
    """HostWait parks a queue until the host rings the doorbell: `ring` at a cycle equals `H db`; the CSR space is the chip's own.
    The host acts at or after the cycle the system stands at (a vector on a link may arrive there); a short `window` brings it close to
    the program's own cycles."""
    import subprocess

    p = rig.Profile("mini")
    c = rig.Chip(p)
    c.at(10, rig.VXMQ, rig.hio(rig.HOST_WAIT, 0, 3))
    c.at(11, rig.VXMQ, rig.hio(rig.HOST_NOTIFY, 0, 5))
    # window 1: the system stands at cycle 12 once the program has parked, and the ring at 50 is the oracle's `H db 32 3`
    s = System("mini", 2, trace=True, window=1)
    assert s.window == 1
    s.load_object(0, c.object())
    s.load_object(1, rig.Chip(p).object())
    r = s.run(max_cycles=20)
    assert r["reason"] == "done" and r["cycle"] == 11 and s.cycles(0) == 11
    assert s.host_waiting(0) == [(rig.VXMQ, 3, 10)] and s.host_waiting(1) == [], "the queue is parked on doorbell 3 since cycle 10"
    out = subprocess.run([sim, p.path, "--trace"], input=rig.oracle_text([c]).replace("C 0\n", "", 1) + "H db 32 3\nR 30\nR 32\nR 0\n",
                         capture_output=True, text=True).stdout
    assert "T irq 51 msix 5" in out
    assert s.ring(0, 3, cycle=50) == 50
    r = s.run(raise_on_diag=False)
    assert r["reason"] == "notify" and r["notifies"] == [(0, 51, 5)] and r["cycle"] == 52, r
    assert (s.csr_read(0, 48), s.csr_read(0, 50), s.csr_read(0, 0), s.csr_read(1, 0)) == (1 << 5, 1, 0, 1)
    assert s.host_waiting(0) == []
    s.csr_write(0, 48, 1 << 5)  # the cause bit clears by writing a one
    assert s.run(raise_on_diag=False)["reason"] == "done" and s.csr_read(0, 48) == 0
    # the default window is the link latency: the same program stands at 216 when it is done, and the host acts from there
    t = System("mini", 2, trace=True)
    t.load_object(0, c.object())
    assert t.run()["cycle"] == p.L
    with pytest.raises(ValueError, match="cycle 50 is in the past: the system stands at cycle 216"):
        t.ring(0, 3, cycle=50)
    assert t.ring(0, 3) == p.L
    with pytest.raises(ValueError, match="window: at least 1 cycle"):
        System("mini", 2, window=0)


# ---- (d) a 3-chip pipeline echo ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("profile", PROFILES)
def test_three_chip_pipeline_echo_equals_tsp_sim(sim, profile, capsys):
    p = rig.Profile(profile)
    chips = rig.echo_three_chips(p, vectors=2)
    oracle = rig.run_oracle(sim, p, chips)
    assert [v for _, v in oracle[2]["Y"]] == [p.val(5), p.val(6)] and oracle[0]["Y"] == oracle[1]["Y"] == [], "the scenario is sound"
    s = rig.new_system(System, p, chips)
    run_done(s)
    assert [bytes(v) for _, v in s.egress(2)] == [p.val(5), p.val(6)]
    agree(s, chips, oracle)
    with capsys.disabled():
        print(
            f"\n3-chip echo ({profile}, {p.vl} lanes, link latency {p.L}): host in at chip 0 cycle 0, out at chip 2 cycles "
            f"{[c for c, _ in s.egress(2)]}; cycles per chip {[s.cycles(c) for c in range(3)]}; system at cycle {s.cycle}"
        )


@pytest.mark.parametrize("window", [1, 7, 215])
@pytest.mark.parametrize("workers", ["1", "3"])
def test_the_results_do_not_depend_on_the_window(sim, window, workers, monkeypatch):
    """The native worker and window settings preserve the reference link behavior."""
    monkeypatch.setenv("OPENTSP_ISS_CHIP_WORKERS", workers)
    p = rig.Profile("mini")
    chips = rig.echo_three_chips(p, vectors=2)
    oracle = rig.run_oracle(sim, p, chips, "--window", str(window))
    s = rig.new_system(System, p, chips, window=window)
    assert s.window == window
    run_done(s)
    agree(s, chips, oracle)
    assert [bytes(v) for _, v in s.egress(2)] == [p.val(5), p.val(6)]


# ---- bad input -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chips", [1, 9, 15, 264])
def test_system_builtin_wiring_accepts_its_supported_range(chips):
    s = System("mini", chips)
    assert s.chips == chips
    assert s.run()["reason"] == "done"


@pytest.mark.parametrize("chips", [0, 265, 512, 513])
def test_system_builtin_wiring_rejects_outside_its_supported_range(chips):
    with pytest.raises(ValueError, match=rf"^{chips} chips: the built-in wiring supports 1\.\.264 chips$"):
        System("mini", chips)


def test_bad_input_has_clear_errors():
    p = rig.Profile("mini")
    s = System("mini", 2)
    with pytest.raises(ValueError, match=r"chip 2: the system has 2 chips \(0\.\.1\)"):
        s.read_words(2, 0, 0, 1)
    with pytest.raises(ValueError, match=r"chip 5: the system has 2 chips"):
        s.dma_write(5, bytes(p.vl))
    with pytest.raises(ValueError, match=r"a DMA vector is 16 bytes \(one per lane\), not 3"):
        s.dma_write(0, b"abc")
    with pytest.raises(ValueError, match="not a whole number of 16-byte words"):
        s.write_words(0, 0, 0, b"abc")
    with pytest.raises(ValueError):
        System("nonesuch", 2)
    other = rig.Profile("ref")
    ref_object = rig.Chip(other).object()
    with pytest.raises(ValueError, match=r'^chip 0: OTS-E-ASM-003: object is for profile "ref", not "mini"'):
        s.load_object(0, ref_object)
    s.load_object(0, rig.Chip(p).object())
    with pytest.raises(ValueError, match="an object is loaded already"):
        s.load_object(0, rig.Chip(p).object())
    s.run()
    with pytest.raises(ValueError, match="load every object before the first run"):
        s.load_object(1, rig.Chip(p).object())
    with pytest.raises(ValueError, match=r"cycle 3 is in the past: the system stands at cycle \d+"):
        s.dma_write(0, bytes(p.vl), cycle=3)
    with pytest.raises(TspChipError, match="beyond the slice end"):
        s.read_words(0, 0, 10**6, 1)


def test_an_interrupted_run_can_be_resumed_or_discarded(sim):
    """A run with a cycle limit stops cleanly (`limit`, the cycle every chip stands at), resumes to the oracle's results, and a system
    that is dropped instead leaves nothing behind."""
    p = rig.Profile("mini")
    chips, val = rig.send_receive(p)
    oracle = rig.run_oracle(sim, p, chips)
    s = rig.new_system(System, p, chips)
    mid = rig.T0 + p.L // 2  # the vector is on the link
    r = s.run(max_cycles=mid)
    assert r["reason"] == "limit" and r["cycle"] == mid + 1 == s.cycle
    assert bytes(s.read_words(1, 44, 50, 1)) == bytes(p.vl), "the receiver has not written yet"
    r = s.run(max_cycles=mid)  # the same limit again: nothing more happens
    assert r["reason"] == "limit" and r["cycle"] == mid + 1
    run_done(s)
    agree(s, chips, oracle)
    discarded = rig.new_system(System, p, chips)
    discarded.run(max_cycles=mid)
    del discarded


def test_the_python_layer_builds_a_system_from_objects_and_runs_through_the_interrupts(sim):
    from opentsp_vllm.system import System as PySystem

    p = rig.Profile("mini")
    chips = rig.echo_three_chips(p, vectors=1)
    chips[2].at(rig.T0 + 600, rig.VXMQ, rig.hio(rig.HOST_NOTIFY, 0, 7))
    s = PySystem.from_objects("mini", [c.object() for c in chips])
    for c, ch in enumerate(chips):
        for cycle, v in ch.dma:
            s.dma_write(c, v, cycle)
        for (sl, a), v in ch.sram.items():
            s.write_words(c, sl, a, v)
    assert isinstance(s, System) and s.chips == 3
    notifies, last = s.run_to_done()
    assert last["reason"] == "done" and [(c, v) for c, _, v in notifies] == [(2, 7)]
    assert s.vectors(2) == [p.val(5)] and s.vectors(2, start=1) == [] and s.vectors(0) == []


# ---- (d) restart: a system that runs again with its SRAM kept (the pipeline's warm restart) ------------------------------------

def feed(s, chips):
    """The host's vectors of every chip's DMA list (what `new_system` pushes after the SRAM words)."""
    for c, ch in enumerate(chips):
        for cycle, v in ch.dma:
            s.dma_write(c, v, cycle)


@pytest.mark.parametrize("profile", PROFILES)
def test_a_system_counts_no_activity_unless_asked(profile):
    """Serving reads no activity counter: a system built, run and restarted leaves every counter at zero. `set_activity_counting` turns
    the counters on, and the setting survives a restart."""
    p = rig.Profile(profile)
    chips, _ = rig.send_receive(p)
    n = range(len(chips))
    s = rig.new_system(System, p, chips)
    run_done(s)
    assert [s.activity_issued(c) for c in n] == [0] * len(chips)
    s.restart()
    feed(s, chips)
    run_done(s)
    assert [s.activity_issued(c) for c in n] == [0] * len(chips), "a restarted system does not count either"
    counted = rig.new_system(System, p, chips)
    counted.set_activity_counting(True)
    run_done(counted)
    first = [counted.activity_issued(c) for c in n]
    assert all(x > 0 for x in first), "the counters count when asked"
    counted.restart()
    feed(counted, chips)
    run_done(counted)
    assert [counted.activity_issued(c) for c in n] == [2 * x for x in first], "counts accumulate across restart"
    counted.set_activity_counting(False)
    assert [counted.activity_issued(c) for c in n] == [2 * x for x in first], "disabling retains collected counts"
    counted.restart()
    feed(counted, chips)
    run_done(counted)
    assert [counted.activity_issued(c) for c in n] == [2 * x for x in first], "disabled runs add no activity"
    counted.set_activity_counting(True)
    counted.restart()
    feed(counted, chips)
    run_done(counted)
    assert [counted.activity_issued(c) for c in n] == [3 * x for x in first], "reenabling resumes cumulative counting"


@pytest.mark.parametrize("profile", PROFILES)
def test_a_restart_keeps_the_sram_and_the_next_run_is_the_first_run_again(sim, profile):
    """After `restart` the system stands at cycle 0 with empty host interfaces, no diagnostics and the SRAM of the first run (the words the program
    wrote included); the host's vectors pushed again give the results, the events and the cycles of the first run, which are `tsp_sim --chips 2`'s."""
    p = rig.Profile(profile)
    chips = rig.echo_two_chips(p)
    oracle = rig.run_oracle(sim, p, chips)
    s = rig.new_system(System, p, chips)
    run_done(s)
    agree(s, chips, oracle)
    first = (s.cycle, [bytes(v) for _, v in s.egress(1)])
    assert first[1] == [p.val(3)]
    s.restart()
    assert s.cycle == 0 and s.egress(1) == [] and s.diagnostics(0) == [] and s.ingress(0) == (0, 0, False)
    feed(s, chips)
    run_done(s)
    agree(s, chips, oracle)
    assert (s.cycle, [bytes(v) for _, v in s.egress(1)]) == first


def test_a_restart_keeps_every_word_of_the_sram_and_only_what_the_host_writes_changes(sim):
    p = rig.Profile("mini")
    chips, val = rig.send_receive(p)
    oracle = rig.run_oracle(sim, p, chips)
    s = rig.new_system(System, p, chips)
    run_done(s)
    assert bytes(s.read_words(1, 44, 50, 1)) == val
    other = bytes((b + 1) & 255 for b in val)
    s.write_words(0, 44, 1, other)  # the host changes the word the sender reads
    s.restart()
    assert bytes(s.read_words(1, 44, 50, 1)) == val, "the word the receiver wrote is still there after the restart"
    assert bytes(s.read_words(0, 44, 1, 1)) == other, "the host's write is kept"
    run_done(s)
    assert bytes(s.read_words(1, 44, 50, 1)) == other, "the run read the changed word"
    s.write_words(0, 44, 1, val)
    s.restart()
    run_done(s)
    agree(s, chips, oracle)


def test_a_restart_clears_the_diagnostics_and_the_run_reports_them_again(sim):
    p = rig.Profile("mini")
    c0, c1 = rig.Chip(p), rig.Chip(p)
    c1.at(rig.T0, rig.ERX, rig.hio(rig.PCI_RECEIVE, rig.W(24)))  # an empty ingress FIFO: OTS-E-TIM-001
    s = rig.new_system(System, p, [c0, c1])
    with pytest.raises(TspChipError):
        s.run()
    first = s.diagnostics(1)
    assert first == [(rig.T0, rig.ERX, "OTS-E-TIM-001")]
    s.restart()
    assert s.diagnostics(1) == [] and s.cycle == 0
    with pytest.raises(TspChipError) as e:
        s.run()
    assert (e.value.code, e.value.cycle, e.value.chip) == ("OTS-E-TIM-001", rig.T0, 1) and s.diagnostics(1) == first


# Explicit oracle ownership and default-build behavior.

def forbid_build(*args, **kwargs):
    raise AssertionError("explicit TSP_SIM must not invoke a build")


def test_explicit_oracle_is_used_unchanged_even_when_older_than_sources(monkeypatch, tmp_path):
    sim = tmp_path / "pinned oracle"
    sim.write_text("#!/bin/sh\nprintf pinned\\n\n")
    sim.chmod(0o755)
    os.utime(sim, (1, 1))
    before = sim.read_bytes(), sim.stat().st_mtime_ns
    monkeypatch.setenv("TSP_SIM", str(sim))
    monkeypatch.setattr(rig.subprocess, "run", forbid_build)
    assert rig.sim_path() == str(sim.resolve())
    assert (sim.read_bytes(), sim.stat().st_mtime_ns) == before


@pytest.mark.parametrize("kind", ["missing", "directory", "nonexecutable"])
def test_invalid_explicit_oracle_fails_clearly_without_build(monkeypatch, tmp_path, kind):
    sim = tmp_path / kind
    if kind == "directory":
        sim.mkdir()
    elif kind == "nonexecutable":
        sim.write_text("not executable")
        sim.chmod(0o644)
    monkeypatch.setenv("TSP_SIM", str(sim))
    monkeypatch.setattr(rig.subprocess, "run", forbid_build)
    with pytest.raises((FileNotFoundError, PermissionError), match="TSP_SIM=.*" + kind):
        rig.sim_path()


def test_relative_explicit_oracle_retains_the_callers_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sim = pathlib.Path("relative") / "oracle"
    sim.parent.mkdir()
    sim.write_text("#!/bin/sh\nexit 0\n")
    sim.chmod(0o755)
    monkeypatch.setenv("TSP_SIM", str(sim))
    monkeypatch.setattr(rig.subprocess, "run", forbid_build)
    assert rig.sim_path() == str(sim.resolve())


@pytest.mark.parametrize("setting", [None, ""])
def test_unset_or_empty_override_ensures_the_default_oracle(monkeypatch, tmp_path, setting):
    if setting is None:
        monkeypatch.delenv("TSP_SIM", raising=False)
    else:
        monkeypatch.setenv("TSP_SIM", setting)
    monkeypatch.setattr(rig, "REPO", tmp_path)
    expected = tmp_path / "target/spec/tsp_sim"
    calls = []

    def ensure(command, *, check):
        assert check is True
        calls.append(command)
        expected.parent.mkdir(parents=True)
        expected.write_text("#!/bin/sh\nexit 0\n")
        expected.chmod(0o755)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(rig.subprocess, "run", ensure)
    assert rig.sim_path() == str(expected)
    assert calls == [["bash", str(tmp_path / "tools/ensure_sim.sh"), str(expected)]]


@pytest.mark.parametrize("shadow", [False, True])
def test_bare_explicit_oracle_launches_the_validated_file(monkeypatch, tmp_path, shadow):
    monkeypatch.chdir(tmp_path)
    oracle = tmp_path / "oracle"
    oracle.write_text("#!/bin/sh\necho pinned\n")
    oracle.chmod(0o755)
    shadow_dir = tmp_path / "path"
    shadow_dir.mkdir()
    other = shadow_dir / "oracle"
    other.write_text("#!/bin/sh\necho shadow\n")
    other.chmod(0o755)
    monkeypatch.setenv("PATH", str(shadow_dir) if shadow else "/usr/bin:/bin")
    monkeypatch.setenv("TSP_SIM", "oracle")
    selected = rig.sim_path()
    completed = subprocess.run([selected], check=True, capture_output=True, text=True)
    assert completed.stdout == "pinned\n", "execute the validated local file, never a PATH shadow"
    assert selected == str(oracle.resolve())


def test_vector_receive_observer_is_opt_in_and_drainable():
    profile = rig.Profile("mini")
    chips, value = rig.send_receive(profile)
    system = rig.new_system(System, profile, chips)
    system.observe_link_receives()
    assert system.drain_link_receives(1) == []
    run_done(system)
    assert bytes(system.read_words(1, 44, 50, 1)) == value
    assert system.drain_link_receives(0) == []
    assert system.drain_link_receives(1) == [(rig.T0 + profile.L, 0)]
    assert system.drain_link_receives(1) == []
    system.restart()
    run_done(system)
    assert system.drain_link_receives(1) == []
