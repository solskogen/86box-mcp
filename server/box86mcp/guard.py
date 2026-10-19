"""Run a command, and stop it if the process that started us goes away.

    python -m box86mcp.guard [--no-activate] <command> [args...]

stdin must be a pipe whose write end only the controller holds. When the controller exits for any
reason (even SIGKILL) the kernel closes that end; the guard reads EOF and stops the command: SIGTERM
(docker/86box.sh turns that into `docker kill`), then SIGKILL if it hasn't gone in 10 s. Without
this, a controller that died without calling VM.stop() left 86Box, and its container, running.

The guard exits with the command's exit status. SIGTERM and SIGINT sent to the guard are passed on
to the command, and the guard keeps waiting, so whoever signalled it still sees the command finish.

--no-activate (Windows): the command's first window shows without taking the focus from whatever
the user is doing (SW_SHOWNOACTIVATE). It has to be the guard that asks: a STARTUPINFO isn't inherited.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading

GRACE_S = 10
SW_SHOWNOACTIVATE = 4


def main(argv: list[str]) -> int:
    no_activate = argv[:1] == ["--no-activate"]
    if no_activate:
        argv = argv[1:]
    if not argv:
        print("usage: python -m box86mcp.guard [--no-activate] <command> [args...]", file=sys.stderr)
        return 2
    startupinfo = None
    if no_activate and os.name == "nt":
        startupinfo = subprocess.STARTUPINFO(dwFlags=subprocess.STARTF_USESHOWWINDOW, wShowWindow=SW_SHOWNOACTIVATE)
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, startupinfo=startupinfo)

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    def watch_controller():
        try:
            while os.read(0, 4096):
                pass
        except OSError:
            pass
        # EOF: the controller has gone.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(GRACE_S)
            except subprocess.TimeoutExpired:
                child.kill()

    threading.Thread(target=watch_controller, daemon=True).start()
    while True:
        try:
            return child.wait()
        except InterruptedError:
            continue


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
