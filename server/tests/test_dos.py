"""Integration tests against a real 86Box booting DOS 6.22."""

from __future__ import annotations

import hashlib
import os
import time

from box86mcp import keys as K
from box86mcp.protocol import Automation


def type_line(auto: Automation, text: str) -> None:
    shift = K.scancode("shift")
    for ch in text + "\n":
        code, shifted = K.char_keys(ch)
        if shifted:
            auto.keys(True, [shift])
        auto.keys(True, [code])
        auto.step(20)
        auto.keys(False, [code])
        if shifted:
            auto.keys(False, [shift])
        auto.step(20)


def test_held_vm_does_not_advance(dos_vm):
    a = dos_vm.auto
    t0 = a.status()["ms"]
    time.sleep(0.5)
    assert a.status()["ms"] == t0


def test_step_is_exact(dos_vm):
    a = dos_vm.auto
    t0 = a.status()["ms"]
    r = a.step(250)
    assert r["ms"] == t0 + 250
    assert r["held"] is True


def test_boot_to_prompt_and_run_ver(dos_vm):
    a = dos_vm.auto
    assert a.wait_text("C:\\>", 60000)["matched"], a.text().render()
    type_line(a, "VER")
    assert a.wait_text("MS-DOS Version 6.22", 5000)["matched"], a.text().render()
    scr = a.text()
    assert scr.mode == 3 and scr.cols == 80


def test_screenshot(dos_vm, tmp_path):
    a = dos_vm.auto
    a.wait_text("C:\\>", 60000)
    shot = tmp_path / "s.png"
    r = a.screenshot(str(shot))
    assert r["w"] >= 640 and r["h"] >= 350
    assert shot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_memory_roundtrip(dos_vm):
    a = dos_vm.auto
    # Interrupt vector table is plain RAM: 0x4f0 is the BIOS "intra-application" area.
    a.memw(0x4F0, b"\xde\xad\xbe\xef")
    assert a.memr(0x4F0, 4) == b"\xde\xad\xbe\xef"
    assert a.wait_mem(0x4F0, b"\xde\xad", 10)["matched"]


def test_floppy_transfer(dos_vm, tmp_path):
    a = dos_vm.auto
    assert a.wait_text("C:\\>", 60000)["matched"]
    src = tmp_path / "HELLO.TXT"
    src.write_text("hello from the host\r\n")
    dos_vm.load_floppy(dos_vm.make_floppy([src]))
    type_line(a, "TYPE A:\\HELLO.TXT")
    assert a.wait_text("hello from the host", 10000)["matched"], a.text().render()


def _frame_hash(vm, path) -> str:
    vm.auto.screenshot(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_deterministic_boot(tmp_path):
    """Two cold boots driven identically should show the same frame at the same emulated time."""
    from box86mcp.vm import VM
    from conftest import make_dos_vm

    hashes = []
    for i in range(2):
        vm = VM(make_dos_vm(tmp_path / f"vm{i}"))
        vm.start(held=True)
        try:
            vm.auto.step(8000)
            type_line(vm.auto, "DIR")
            vm.auto.step(1000)
            hashes.append(_frame_hash(vm, tmp_path / f"f{i}.png"))
        finally:
            vm.stop()
    assert hashes[0] == hashes[1]


def test_serial_capture(dos_vm):
    a = dos_vm.auto
    assert a.wait_text("C:\\>", 60000)["matched"]
    type_line(a, "ECHO hello serial> COM1")
    a.step(500)
    assert "hello serial" in dos_vm.serial_read()
    assert dos_vm.serial_read() == ""  # consumed


def test_info_and_crash_detection(dos_vm, tmp_path):
    a = dos_vm.auto
    assert a.wait_text("C:\\>", 60000)["matched"]
    info = a.info()
    assert info["machine"] == "486sp3c" and info["mem_kb"] == 16384
    assert info["hard_resets"] == 1 and info["soft_resets"] == 0
    # lidt [0x106] with a zero limit, then int 3: #GP, double fault, triple fault.
    crash = tmp_path / "CRASH.COM"
    crash.write_bytes(bytes.fromhex("0f011e0601cc") + bytes(6))
    dos_vm.load_floppy(dos_vm.make_floppy([crash]))
    type_line(a, "A:\\CRASH.COM")
    for _ in range(20):  # loading from floppy takes a second or two of emulated time
        a.step(250)
        if a.info()["soft_resets"]:
            break
    assert a.info()["soft_resets"] == 1


def test_cdrom_insert_and_eject(dos_vm, tmp_path):
    import pycdlib

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "Long File Name.txt").write_text("x")
    (tmp_path / "README.TXT").write_text("y")
    iso = dos_vm.make_iso([tmp_path / "README.TXT", tmp_path / "data"])
    check = pycdlib.PyCdlib()
    check.open(str(iso))
    names = sorted(c.file_identifier().decode() for c in check.list_children(iso_path="/DATA"))
    joliet = sorted(c.file_identifier().decode("utf-16-be") for c in check.list_children(joliet_path="/data"))
    check.close()
    assert "LONG_FIL.TXT;1" in names and "Long File Name.txt" in joliet

    dos_vm.load_cdrom(iso)
    assert [c["file"] for c in dos_vm.auto.info()["cdrom"]] == [str(iso)]
    dos_vm.auto.call("cd eject 0")
    assert dos_vm.auto.info()["cdrom"] == []


def test_record_frames_by_emulated_time(dos_vm, tmp_path):
    a = dos_vm.auto
    assert a.wait_text("C:\\>", 60000)["matched"]
    frames = tmp_path / "frames"
    frames.mkdir()
    start = a.status()["ms"]
    a.call(f"record start 100 {frames}")
    type_line(a, "DIR")
    a.step(1000)
    end = a.status()["ms"]
    assert a.call("record stop")["frames"] == len(list(frames.glob("*.png")))
    stamps = sorted(int(p.stem) for p in frames.glob("*.png"))
    # At most one frame per 100 ms of emulated time, named by when it was shown.
    assert 5 <= len(stamps) <= (end - start) // 100 + 1
    assert all(b - a >= 100 for a, b in zip(stamps, stamps[1:]))
    assert start <= stamps[0] and stamps[-1] <= end


def test_tcp_transport_requires_token(tmp_path, monkeypatch):
    """TCP: loopback only, and every client must present the per-run token first."""
    import socket as pysocket

    from conftest import make_dos_vm
    from box86mcp.vm import VM

    monkeypatch.setenv("BOX86_TRANSPORT", "tcp")
    vm = VM(make_dos_vm(tmp_path / "vm"))
    vm.start(held=True)
    try:
        assert vm.endpoint.startswith("tcp:127.0.0.1:")
        assert vm.auto.wait_text("C:\\>", 60000)["matched"]  # the authenticated client works
        if os.name == "nt":
            # No mode bits here: the token is owner-only through the ACL of the run directory, which
            # tempfile.mkdtemp restricts to its owner (Python 3.12.4 and later).
            import re
            import subprocess

            acl = subprocess.run(["icacls", vm.token_file], capture_output=True, text=True).stdout
            assert not re.search(r"(Users|Authenticated Users|Everyone):", acl), acl
        else:
            assert oct(os.stat(vm.token_file).st_mode & 0o777) == "0o600"
        host, port = vm.endpoint[4:].rsplit(":", 1)
        vm.auto.close()  # one client at a time
        for first_line in (b"status\n", b"auth 0000\n"):
            with pysocket.create_connection((host, int(port)), timeout=10) as s:
                s.sendall(first_line)
                assert b"authentication required" in s.recv(4096)
        vm.auto.connect()
        assert vm.auto.status()["held"] is True
    finally:
        vm.stop()


def test_attach_over_tcp_with_inline_screenshots(tmp_path, monkeypatch):
    """vm_attach-style control of an already running 86Box: endpoint + token, frames over the socket."""
    from conftest import make_dos_vm
    from box86mcp.vm import VM, AttachedVM

    monkeypatch.setenv("BOX86_TRANSPORT", "tcp")
    vm = VM(make_dos_vm(tmp_path / "vm"))
    vm.start(held=True)
    try:
        token = open(vm.token_file).read().strip()
        vm.auto.close()  # one client at a time
        other = AttachedVM(vm.endpoint, token)
        assert other.auto.wait_text("C:\\>", 60000)["matched"]
        shot = other.auto.screenshot()
        assert shot["png"][:8] == b"\x89PNG\r\n\x1a\n" and shot["w"] >= 640
        other.stop()
        vm.auto.connect()
    finally:
        vm.stop()


def test_memory_tools_bounds_find_and_diff(dos_vm):
    import pytest

    from box86mcp import server

    server._vm = dos_vm
    try:
        dos_vm.auto.step(3000)
        size = dos_vm.mem_bytes()
        assert size == 16384 * 1024
        with pytest.raises(ValueError):
            server.read_memory(size - 8, 16)
        with pytest.raises(ValueError):
            server.write_memory(size, "00")
        # The BIOS keeps the conventional memory size, in KB, at 0x413: 639 here, as it reserves the
        # top 1 KB for its extended data area.
        assert "0x413" in server.find_memory(value=639, size=2, start=0x400, end=0x500)
        # Snapshot, change a word, and narrow down to it.
        addr = 0x7F000
        server.write_memory(addr, "10 00")
        server.snapshot_memory("t", start=0x70000, end=0x80000)
        server.write_memory(addr, "20 00")
        out = server.diff_memory("increased", name="t", size=2)
        assert f"{addr:#x}=32" in out
        server.write_memory(addr, "05 00")
        out = server.diff_memory("decreased", name="t", size=2)
        # Every alignment is searched, so the word straddling the low byte survives too.
        assert f"{addr:#x}=5" in out and int(out.split()[0]) <= 2
    finally:
        server._vm = None


def test_vm_create_from_template(tmp_path):
    from conftest import make_dos_vm

    from box86mcp import server
    from box86mcp.vm import machine_of

    tpl = make_dos_vm(tmp_path / "tpl")
    cfg = (tpl / "86box.cfg").read_text().replace("mem_size = 16384", "mem_size = 8192")
    cfg = cfg.replace("[General]", "[General]\nuuid = 11111111-2222-3333-4444-555555555555")
    (tmp_path / "custom.cfg").write_text(cfg)
    nvr = tmp_path / "saved.nvr"
    nvr.write_bytes(bytes(256))
    out = server.vm_create(str(tmp_path / "new"), cylinders=100, heads=16, sectors=63,
                           config=str(tmp_path / "custom.cfg"), nvr=str(nvr))
    new = tmp_path / "new"
    text = (new / "86box.cfg").read_text()
    assert "mem_size = 8192" in text and "hdd_01_parameters = 63, 16, 100, 0, ide" in text
    assert "hdd_01_fn = hdd.img" in text
    assert "uuid" not in text  # the template's identity stays behind
    assert (new / "nvr" / f"{machine_of(new)}.nvr").read_bytes() == bytes(256)
    assert "as 486sp3c.nvr" in out


def test_copied_vm_with_a_network_card_starts(tmp_path):
    """A uuid from another folder plus a network card made 86Box ask whether the VM was moved or
    copied, before starting the machine: headless, a hang. start() drops the stale uuid."""
    from conftest import make_dos_vm
    from box86mcp.vm import VM

    d = make_dos_vm(tmp_path / "copy")
    cfg = d / "86box.cfg"
    text = cfg.read_text().replace("[General]", "[General]\nuuid = 11111111-2222-3333-4444-555555555555")
    cfg.write_text(text + "\n[Network]\nnet_01_card = ne2kpnp\nnet_01_net_type = slirp\n")
    vm = VM(d)
    vm.auto.timeout = 30
    vm.start(held=True)
    try:
        assert vm.auto.wait_text("C:\\>", 60000)["matched"]
    finally:
        vm.stop()


def test_fast_forward_toggles_and_status_reports_it(dos_vm):
    a = dos_vm.auto
    assert a.status()["ff"] is False
    assert a.fast_forward(True)["ff"] is True
    assert a.fast_forward()["ff"] is True and a.status()["ff"] is True
    r = a.step(100)  # still exact while held
    assert r["held"] is True
    assert a.fast_forward(False)["ff"] is False


def test_soft_power_off_without_a_power_button(dos_vm):
    # The 486 has neither ACPI nor a soft power button, so quit soft is a plain power-off.
    assert dos_vm.power_off_soft(timeout=10)
    assert not dos_vm.running()


def test_cfg_rewrite_keeps_line_endings(tmp_path):
    """86Box on Windows writes 86box.cfg with CRLF; rewriting it (dropping the uuid) keeps them."""
    from box86mcp.vm import _rewrite_cfg

    for nl in ("\r\n", "\n"):
        cfg = tmp_path / "86box.cfg"
        cfg.write_bytes(nl.join(["[General]", "uuid = x", "vid_renderer = qt_software", ""]).encode())
        lines = [l for l in cfg.read_text().splitlines() if not l.startswith("uuid")]
        _rewrite_cfg(cfg, lines)
        assert cfg.read_bytes() == nl.join(["[General]", "vid_renderer = qt_software", ""]).encode()
