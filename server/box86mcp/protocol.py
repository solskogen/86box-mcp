"""Client for the 86Box automation socket (one text line out, one JSON line back)."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass


class AutomationError(RuntimeError):
    pass


# Automation socket protocol versions this client understands.
SUPPORTED_PROTOCOLS = {1}


CP437 = (
    "\x00☺☻♥♦♣♠•◘○◙♂♀♪♫☼►◄↕‼¶§▬↨↑↓→←∟↔▲▼"
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~⌂"
    "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀"
    "αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\xa0"
)


def decode_cp437(data: bytes) -> str:
    return "".join(" " if b == 0 else CP437[b] for b in data)


@dataclass
class TextScreen:
    mode: int
    cols: int
    rows: int
    cursor: tuple[int, int]
    lines: list[str]
    attrs: list[bytes]

    def render(self, cursor: bool = True) -> str:
        out = [line.rstrip() for line in self.lines]
        while out and not out[-1]:
            out.pop()
        text = "\n".join(out)
        if cursor:
            text += f"\n[mode {self.mode:02x}h {self.cols}x{self.rows}, cursor row {self.cursor[0]} col {self.cursor[1]}]"
        return text


class Automation:
    """Synchronous client. Each call blocks until 86Box has executed the command.

    endpoint is "unix:<path>" (or a bare path) or "tcp:<host>:<port>". TCP needs the token 86Box
    wrote to token_file; it is sent once, right after connecting.
    """

    def __init__(self, endpoint: str, token_file: str | None = None, timeout: float = 600.0):
        self.endpoint = endpoint
        self.token_file = token_file
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self._buf = b""
        # Called with (request, reply) when a timed command ends early because 86Box's GDB stub has
        # stopped the CPU; returns the reply to use.
        self.on_debugger_stop = None

    @property
    def path(self) -> str:
        return self.endpoint[5:] if self.endpoint.startswith("unix:") else self.endpoint

    def _open(self) -> socket.socket:
        if self.endpoint.startswith("tcp:"):
            host, _, port = self.endpoint[4:].rpartition(":")
            return socket.create_connection((host, int(port)), timeout=self.timeout)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(self.path)
        except OSError:
            s.close()
            raise
        return s

    def connect(self, wait: float = 30.0) -> None:
        deadline = time.monotonic() + wait
        while True:
            try:
                s = self._open()
                s.settimeout(self.timeout)
                self.sock = s
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() > deadline:
                    raise AutomationError(f"86Box automation endpoint {self.endpoint} did not come up")
                time.sleep(0.1)
        if self.endpoint.startswith("tcp:") and self.token_file:
            # Without a token (86Box run with --automation-insecure) there's nothing to send; an 86Box
            # that does want one answers the first request with "authentication required".
            with open(self.token_file) as f:
                token = f.read().strip()
            self.call(f"auth {token}")

    def close(self) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None
        self._buf = b""

    def call(self, line: str) -> dict:
        if "\n" in line:
            raise ValueError("request must be a single line")
        if not self.sock:
            self.connect()
        assert self.sock
        self.sock.sendall(line.encode("latin-1") + b"\n")
        while b"\n" not in self._buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                self.close()
                raise AutomationError("86Box closed the automation socket")
            self._buf += chunk
        raw, self._buf = self._buf.split(b"\n", 1)
        reply = json.loads(raw)
        if not reply.get("ok"):
            raise AutomationError(reply.get("error", "unknown error"))
        if reply.get("stopped") == "debugger" and self.on_debugger_stop:
            return self.on_debugger_stop(line, reply)
        return reply

    # Convenience wrappers ------------------------------------------------

    def status(self) -> dict:
        return self.call("status")

    def info(self) -> dict:
        r = self.call("info")
        r.pop("ok", None)
        return r

    def hold(self) -> dict:
        return self.call("hold")

    def release(self) -> dict:
        return self.call("release")

    def fast_forward(self, on: bool | None = None) -> dict:
        """Turn fast forward on or off (None: just report it). The reply's ff is the state now."""
        return self.call("ff" if on is None else f"ff {'on' if on else 'off'}")

    def check_quit_soft(self) -> None:
        """Raise unless this 86Box understands quit soft."""
        # Before 86Box#8382, any "quit ..." was a hard quit. #8384 (ff, which status reports) was
        # merged after it, so a status with ff means quit soft is understood.
        if "ff" not in self.status():
            raise AutomationError("this 86Box predates 'quit soft' (86Box#8382); it would power off hard")

    def quit_soft(self) -> dict:
        """Press the machine's power button: ACPI, or the board's own, else a plain power-off."""
        self.check_quit_soft()
        return self.call("quit soft")

    def step(self, ms: int) -> dict:
        return self.call(f"step {int(ms)}")

    def keys(self, down: bool, scancodes: list[int]) -> dict:
        if not scancodes:
            return self.status()
        codes = " ".join(f"0x{c:x}" for c in scancodes)
        return self.call(f"key {'down' if down else 'up'} {codes}")

    def mouse(self, dx: int, dy: int, buttons: int) -> dict:
        return self.call(f"mouse {int(dx)} {int(dy)} {int(buttons)}")

    def screenshot(self, path: str | None = None) -> dict:
        """With a path, 86Box writes the PNG there (on its own machine). Without, the PNG comes back
        in the reply as bytes under "png", which also works across a network."""
        if path:
            return self.call(f"screenshot {path}")
        import base64

        r = self.call("screenshot")
        r["png"] = base64.b64decode(r["png"])
        return r

    def text(self) -> TextScreen:
        r = self.call("text")
        return TextScreen(
            mode=r["mode"],
            cols=r["cols"],
            rows=r["rows"],
            cursor=tuple(r["cursor"]),
            lines=[decode_cp437(bytes.fromhex(h)) for h in r["chars"]],
            attrs=[bytes.fromhex(h) for h in r["attrs"]],
        )

    def memr(self, addr: int, length: int) -> bytes:
        return bytes.fromhex(self.call(f"memr 0x{addr:x} {int(length)}")["data"])

    def memw(self, addr: int, data: bytes) -> dict:
        return self.call(f"memw 0x{addr:x} {data.hex()}")

    def regs(self) -> dict:
        r = self.call("regs")
        r.pop("ok", None)
        return r

    def wait_text(self, text: str, timeout_ms: int) -> dict:
        return self.call(f"wait text {int(timeout_ms)} {text}")

    def wait_mem(self, addr: int, data: bytes, timeout_ms: int) -> dict:
        return self.call(f"wait mem {int(timeout_ms)} 0x{addr:x} {data.hex()}")
