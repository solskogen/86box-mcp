"""mcpctl: drive the 86box MCP server from a shell, one tool call at a time, and record it.

The VM lives inside one MCP session, so a daemon keeps that session open and
takes calls over a UNIX socket:

    python -m box86mcp.ctl serve --record rec/ &      # start the session (and a recording)
    python -m box86mcp.ctl call vm_start '{"vm_dir": "~/vms/test"}'
    python -m box86mcp.ctl call screenshot            # images are saved; their paths printed
    python -m box86mcp.ctl stop
    python -m box86mcp.ctl render rec/ install.mp4    # captioned video of every step

Every call goes through MCP exactly as an agent's would. With --record, the
frame after each call is saved with the call as its caption.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import socket
import sys
import tempfile
from pathlib import Path

DEFAULT_SOCK = os.path.join(os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir(), "box86mcp-ctl.sock")
# Windows' asyncio has no UNIX sockets: there the daemon listens on loopback TCP, and the "socket"
# path is an owner-only file holding the port and a random token that every request must carry.
USE_TCP = os.name == "nt" or os.environ.get("BOX86_CTL_TCP") == "1"   # the override is for testing
SERVER = Path(sys.executable).with_name("box86mcp")
# Calls that already return a frame, or after which there is no machine to look at.
NO_EXTRA_FRAME = {"screenshot", "act", "vm_stop", "vm_create", "dos_create_vm", "dos_install_files"}
# Read-only calls: nothing changes on screen, so they are not recorded at all.
READ_ONLY = {"screen_text", "registers", "vm_info", "vm_status", "read_memory", "serial_read"}


async def serve(sock_path: str, record: Path | None, out: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    out.mkdir(parents=True, exist_ok=True)
    if record:
        record.mkdir(parents=True, exist_ok=True)
    seq = len(list(record.glob("*.png"))) if record else 0

    async with stdio_client(StdioServerParameters(command=str(SERVER), env=dict(os.environ))) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            done = asyncio.Event()

            async def save_frame(png: bytes, caption: str) -> None:
                nonlocal seq
                if not record:
                    return
                seq += 1
                (record / f"{seq:05d}.png").write_bytes(png)
                with open(record / "captions.jsonl", "a") as f:
                    f.write(json.dumps({"frame": f"{seq:05d}.png", "caption": caption}) + "\n")

            token = secrets.token_hex(16)

            async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                req = json.loads(await reader.readline())
                if USE_TCP and not secrets.compare_digest(str(req.get("token")), token):
                    writer.close()      # loopback TCP is open to every local process
                    return
                if req.get("tool") == "__stop__":
                    writer.write(b'{"text": "stopping"}\n')
                    await writer.drain()
                    writer.close()
                    done.set()
                    return
                tool, args = req["tool"], req.get("args", {})
                caption = f"{tool}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
                if tool == "note":
                    # A narration caption at the current emulated time (no MCP call).
                    st = await session.call_tool("vm_status", {})
                    ms = json.loads(st.content[0].text)["ms"]
                    with open(out / "calls.jsonl", "a") as f:
                        f.write(json.dumps({"start_ms": ms, "end_ms": ms, "tool": "note",
                                            "caption": "NOTE: " + args.get("text", "")}) + "\n")
                    writer.write(b'{"text": "noted"}\n')
                    await writer.drain()
                    writer.close()
                    return
                reply: dict = {"images": []}

                async def emulated_ms():
                    try:
                        st = await session.call_tool("vm_status", {})
                        return json.loads(st.content[0].text)["ms"] if not st.is_error else None
                    except Exception:
                        return None

                start_ms = None if tool in READ_ONLY else await emulated_ms()
                try:
                    res = await session.call_tool(tool, args)
                    reply["text"] = "\n".join(c.text for c in res.content if c.type == "text")
                    reply["is_error"] = bool(res.is_error)
                    for i, c in enumerate(c for c in res.content if c.type == "image"):
                        png = base64.b64decode(c.data)
                        path = out / f"{tool}-{seq + 1:05d}-{i}.png"
                        path.write_bytes(png)
                        reply["images"].append(str(path))
                        await save_frame(png, caption)
                    if record and tool not in NO_EXTRA_FRAME | READ_ONLY and not res.is_error:
                        shot = await session.call_tool("screenshot", {"max_width": 640})
                        imgs = [c for c in shot.content if c.type == "image"]
                        if imgs:
                            await save_frame(base64.b64decode(imgs[0].data), caption)
                except Exception as e:  # report, keep serving
                    reply["text"], reply["is_error"] = repr(e), True
                if tool not in READ_ONLY:
                    end_ms = await emulated_ms()
                    with open(out / "calls.jsonl", "a") as f:
                        f.write(json.dumps({"start_ms": start_ms, "end_ms": end_ms, "tool": tool,
                                            "caption": caption}) + "\n")
                writer.write((json.dumps(reply) + "\n").encode())
                await writer.drain()
                writer.close()

            if os.path.exists(sock_path):
                os.unlink(sock_path)
            if USE_TCP:
                srv = await asyncio.start_server(handle, "127.0.0.1", 0, limit=1 << 24)
                port = srv.sockets[0].getsockname()[1]
                fd = os.open(sock_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w") as f:
                    json.dump({"port": port, "token": token}, f)
            else:
                old = os.umask(0o177)       # owner-only, from the moment it exists
                try:
                    srv = await asyncio.start_unix_server(handle, sock_path, limit=1 << 24)
                finally:
                    os.umask(old)
            print(f"mcpctl serving on {sock_path}", flush=True)
            async with srv:
                await done.wait()
            os.unlink(sock_path)


def _connect(sock_path: str) -> tuple[socket.socket, str | None]:
    if USE_TCP:
        with open(sock_path) as f:
            where = json.load(f)
        return socket.create_connection(("127.0.0.1", where["port"])), where["token"]
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    return s, None


def call(sock_path: str, tool: str, args: dict) -> int:
    s, token = _connect(sock_path)
    s.sendall((json.dumps({"tool": tool, "args": args, "token": token}) + "\n").encode())
    data = b""
    while not data.endswith(b"\n"):
        chunk = s.recv(1 << 20)
        if not chunk:
            break
        data += chunk
    reply = json.loads(data)
    print(reply.get("text", ""))
    for img in reply.get("images", []):
        print(f"[image] {img}")
    return 1 if reply.get("is_error") else 0


def render(record: Path, output: Path, seconds_per_frame: float = 0.6) -> None:
    """Captioned video (mp4 via ffmpeg) or GIF of a recording."""
    import subprocess
    import tempfile

    from PIL import Image, ImageDraw, ImageFont

    entries = [json.loads(line) for line in open(record / "captions.jsonl")]
    # Recordings made before READ_ONLY existed captured read-only calls too.
    entries = [e for e in entries if e["caption"].split("(", 1)[0] not in READ_ONLY]
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    frames = []
    last_pixels = None
    for n, e in enumerate(entries, 1):
        img = Image.open(record / e["frame"]).convert("RGB")
        # Skip frames where nothing on screen changed (idle prompts, waiting).
        pixels = img.tobytes()
        if pixels == last_pixels:
            continue
        last_pixels = pixels
        # Fit into a fixed 640x480 box: odd frame shapes (mode switches) must not stretch the video.
        scale = min(640 / img.width, 480 / img.height)
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
        canvas = Image.new("RGB", (640, 480 + 28), (24, 24, 28))
        canvas.paste(img, ((640 - img.width) // 2, (480 - img.height) // 2))
        caption = f"{n:3d}  {e['caption']}"
        if len(caption) > 84:
            caption = caption[:81] + "..."
        ImageDraw.Draw(canvas).text((6, 480 + 6), caption, fill=(230, 230, 230), font=font)
        frames.append(canvas)
    if output.suffix == ".gif":
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=int(seconds_per_frame * 1000),
                       loop=0, optimize=True)
        return
    height = max(f.height for f in frames)
    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(frames):
            padded = Image.new("RGB", (640, height + (height % 2)), (24, 24, 28))
            padded.paste(f, (0, 0))
            padded.save(f"{tmp}/{i:05d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(1 / seconds_per_frame),
                        "-i", f"{tmp}/%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                        str(output)], check=True)


def render_timeline(frames_dir: Path, output: Path, calls_log: Path | None = None, fast_factor: int = 8,
                    linger_ms: int = 1500, max_still: int = 5) -> None:
    """Video from an 86Box recording (frames named by emulated ms).

    Frames play at real emulated speed while a non-step call is running and for linger_ms after it,
    so pointer moves and typing are smooth; other stretches (steps, waits: copying, booting) keep only
    every fast_factor-th frame. Runs of identical frames are cut to max_still frames.
    """
    import subprocess
    import tempfile

    from PIL import Image, ImageChops, ImageDraw, ImageFont

    files = sorted(frames_dir.glob("*.png"))
    if len(files) < 2:
        raise ValueError(f"not enough frames in {frames_dir}")
    stamps = [int(f.stem) for f in files]
    interval = max(1, min(b - a for a, b in zip(stamps, stamps[1:]) if b > a))
    calls = []
    if calls_log and calls_log.exists():
        calls = [json.loads(l) for l in open(calls_log)]
        calls = [c for c in calls if c.get("start_ms") is not None]
    # Real speed only around input (pointer moves, typing, media swaps); polling and waiting fast-forward.
    interactive = {"mouse", "tablet", "together", "mouse_to", "mouse_state", "mouse_drag", "press_keys", "type_text", "act", "floppy_insert",
                   "floppy_eject", "cdrom_insert", "cdrom_eject", "reset", "dos_put_floppy", "note"}
    active = [(c["start_ms"], (c["end_ms"] or c["start_ms"]) + linger_ms) for c in calls if c["tool"] in interactive]

    shown = [c for c in calls if c["tool"] not in {"screenshot", "step", "note"} | READ_ONLY]
    notes = [c for c in calls if c["tool"] == "note"]

    def note_at(ms: int, hold_ms: int = 8000) -> str:
        current = ""
        for c in notes:
            if c["start_ms"] <= ms < c["start_ms"] + hold_ms:
                current = c["caption"]
        return current

    def caption_at(ms: int) -> str:
        current = ""
        for c in shown:
            if c["start_ms"] <= ms:
                current = c["caption"]
        return current

    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    with tempfile.TemporaryDirectory() as tmp:
        n, skipped, still, last = 0, 0, 0, None
        for f, ms in zip(files, stamps):
            is_active = any(a <= ms <= b for a, b in active)
            if not is_active:
                skipped += 1
                if skipped % fast_factor:
                    continue
            img = Image.open(f).convert("RGB")
            # "Still" tolerates a blinking caret (16-32 px); a moving pointer changes 100+ px.
            if last is not None and last.size == img.size:
                diff = ImageChops.difference(img, last).convert("L").point(lambda v: 255 if v > 24 else 0)
                still = still + 1 if diff.histogram()[255] < 60 else 0
            else:
                still = 0
            last = img
            if still >= max_still:
                continue
            scale = min(640 / img.width, 480 / img.height)
            img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
            top = 44 if notes else 0
            canvas = Image.new("RGB", (640, top + 480 + 28), (24, 24, 28))
            canvas.paste(img, ((640 - img.width) // 2, top + (480 - img.height) // 2))
            draw = ImageDraw.Draw(canvas)
            speed = "1x" if is_active else f"{fast_factor}x"
            text = f"{ms / 1000:7.1f}s {speed:>3}  {caption_at(ms)}"
            draw.text((6, top + 486), text[:84], fill=(230, 230, 230), font=font)
            note = note_at(ms)
            if note:
                # Narration: wrap onto two lines in the top band.
                words, lines, line = note.split(), [], ""
                for w in words:
                    if len(line) + len(w) + 1 > 74:
                        lines.append(line)
                        line = w
                    else:
                        line = f"{line} {w}".strip()
                lines.append(line)
                for i, l in enumerate(lines[:2]):
                    draw.text((6, 4 + i * 18), l, fill=(255, 214, 64), font=font)
            canvas.save(f"{tmp}/{n:06d}.png")
            n += 1
        fps = 1000 / interval
        if output.suffix == ".gif":
            imgs = [Image.open(f"{tmp}/{i:06d}.png") for i in range(n)]
            imgs[0].save(output, save_all=True, append_images=imgs[1:], duration=int(interval), loop=0)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:g}", "-i", f"{tmp}/%06d.png",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", str(output)], check=True)


def main() -> None:
    p = argparse.ArgumentParser(prog="mcpctl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sock", default=DEFAULT_SOCK)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--record", type=Path)
    s.add_argument("--out", type=Path, default=Path("mcpctl-out"))
    c = sub.add_parser("call")
    c.add_argument("tool")
    c.add_argument("args", nargs="?", default="{}")
    sub.add_parser("stop")
    r = sub.add_parser("render")
    r.add_argument("record", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--seconds-per-frame", type=float, default=0.6)
    t = sub.add_parser("render-timeline", help="video from a record_start recording")
    t.add_argument("frames", type=Path)
    t.add_argument("output", type=Path)
    t.add_argument("--calls", type=Path, help="calls.jsonl written by serve")
    t.add_argument("--fast-factor", type=int, default=8)
    a = p.parse_args()
    if a.cmd == "serve":
        asyncio.run(serve(a.sock, a.record, a.out))
    elif a.cmd == "call":
        # Screen text has box drawing and the like, which Windows' ANSI code page (what Python
        # writes to a pipe there) can't encode.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.exit(call(a.sock, a.tool, json.loads(a.args)))
    elif a.cmd == "stop":
        call(a.sock, "__stop__", {})
    elif a.cmd == "render":
        render(a.record, a.output, a.seconds_per_frame)
    elif a.cmd == "render-timeline":
        render_timeline(a.frames, a.output, a.calls, a.fast_factor)


if __name__ == "__main__":
    main()
