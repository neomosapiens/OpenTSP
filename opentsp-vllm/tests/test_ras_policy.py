"""Management rules at the boundary of the chip, service process and journal."""

import socket
import threading

import pytest

from opentsp_vllm.ras import DomainFault, Journal, ManagedChip
from opentsp_vllm.ras_device import DeviceProcess, _receive, _send


def record(word, overflow=False):
    return dict(events=1, fatal_count=1, sram_word=word, overflow=overflow)


@pytest.mark.parametrize("reports", [
    [(0, record(None))],
    [(0, record((1, 2), True))],
    [(0, record((1, 2))), (1, record((1, 2)))],
])
def test_incomplete_or_ambiguous_evidence(reports):
    assert DomainFault("unsafe", reports).target is None


def test_device_launch_failure_closes_both_socket_endpoints(monkeypatch):
    from opentsp_vllm import ras_device

    host, child = socket.socketpair()
    monkeypatch.setattr(ras_device.socket, "socketpair", lambda: (host, child))

    def fail_to_launch(*args, **kwargs):
        raise OSError("injected device launch failure")

    monkeypatch.setattr(ras_device.subprocess, "Popen", fail_to_launch)
    try:
        with pytest.raises(OSError, match="injected device launch failure"):
            DeviceProcess(None)
        assert host.fileno() == child.fileno() == -1
    finally:
        host.close()
        child.close()


def test_old_device_reply_cannot_complete_new_request():
    device = object.__new__(DeviceProcess)
    device.timeout = 1
    device.sequence = 8
    device.incarnation = b"new-device"
    device.sock, peer = socket.socketpair()
    def stale():
        with peer:
            tag, _, _ = _receive(peer)
            _send(peer, ((b"old-device", tag[1]), True, "delayed output"))
    worker = threading.Thread(target=stale)
    worker.start()
    try:
        with pytest.raises(DomainFault, match="stale device reply"):
            device._call("step")
    finally:
        device.sock.close()
        worker.join()


def test_crash_during_recovery_and_service_clearance(tmp_path):
    j = Journal(tmp_path, "domain")
    try:
        j.record("recovering", "sram:3:87")
        with pytest.raises(RuntimeError, match="owner"):
            Journal(tmp_path, "domain", maintenance=True)
    finally:
        j.close()
    with pytest.raises(RuntimeError, match="clearance"):
        Journal(tmp_path, "domain")
    j = Journal(tmp_path, "domain", maintenance=True)
    try:
        assert j.attempted("sram:3:87")
        j.record("cleared", detail="replaced SRAM")
        assert not j.attempted("sram:3:87")
    finally:
        j.close()


def test_failed_close_record_releases_the_journal_after_the_device_stops(tmp_path, monkeypatch):
    class Backend:
        closed = False

        def close(self):
            self.closed = True

    device = object.__new__(ManagedChip)
    device.failure = None
    device.backend = Backend()
    device.journal = Journal(tmp_path, "domain")
    journal = device.journal

    def fail_to_record(*args):
        raise OSError("could not write journal")

    monkeypatch.setattr(journal, "record", fail_to_record)
    try:
        with pytest.raises(OSError, match="could not write journal"):
            device.close()
        assert device.backend.closed
        maintenance = Journal(tmp_path, "domain", maintenance=True)
        maintenance.close()
        assert device.journal is None
    finally:
        journal.close()


def test_failed_component_check_closes_replacement(tmp_path):
    class Broken:
        closed = False
        def close(self):
            self.closed = True

        def load_model(self, source):
            pass

        def package_identity(self):
            return "trusted"

        def probe_sram(self, *target):
            raise RuntimeError("component fault persists")

    device = object.__new__(ManagedChip)
    device.source = "model"
    device.identity = "trusted"
    device.backend = Broken()
    old = device.backend
    device.create = Broken
    device.journal = Journal(tmp_path, "health")
    try:
        device._fence(DomainFault("unsafe", [(3, record((87, 8191)))]))
        assert old.closed and device.backend.closed and "component fault persists" in device.failure
        assert device.journal.attempted("sram:3:87")
    finally:
        device.close()


def test_operator_can_read_live_incident_and_only_clear_offline(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from opentsp_vllm.ras import main
    journal = Journal(tmp_path, "edge")
    argv = ["ras", "edge", "--state-dir", str(tmp_path)]
    try:
        journal.record("fenced", "sram:5:6", {"reason": "uncorrectable"})
        monkeypatch.setattr(sys, "argv", argv)
        main()
        assert json.loads(capsys.readouterr().out)[1:3] == ["fenced", "sram:5:6"]
        monkeypatch.setattr(sys, "argv", argv + ["--clear", "repair-123"])
        with pytest.raises(RuntimeError, match="already has an owner"):
            main()
    finally:
        journal.close()
    main()
    assert json.loads(capsys.readouterr().out.splitlines()[-1])[1] == "cleared"


def test_corrupted_private_frame_rejected_before_decoding():
    from opentsp_vllm.ras_device import HEADER
    host, device = socket.socketpair()
    with host, device:
        device.sendall(HEADER.pack(3, bytes(32)) + b"bad")
        with pytest.raises(ValueError, match="integrity"):
            _receive(host)
