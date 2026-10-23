"""Process helpers shared by the stages that launch CS2 (generate #13, render #15).

HLAE and CS2 only ever run offline: every launch goes through ``require_insecure``.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

log = logging.getLogger(__name__)


def require_insecure(cmd: Sequence[str]) -> None:
    """Raise ``ValueError`` unless ``-insecure`` is in ``cmd``.

    The flag counts when it is an argv element of its own or a word of the value after ``-cmdLine`` (the HLAE custom
    loader passes the CS2 command line as a single argument).

    Args:
        cmd: The argv list about to be started.

    Raises:
        ValueError: If ``-insecure`` is missing.
    """
    tokens: list[str] = []
    for i, part in enumerate(cmd):
        tokens.append(part)
        if i > 0 and cmd[i - 1].lower() == "-cmdline":
            tokens.extend(part.split())
    if "-insecure" not in tokens:
        raise ValueError("refusing to start CS2 without -insecure (HLAE and CS2 only ever run offline)")


def spawn_kwargs(windows: bool) -> dict[str, Any]:
    """``subprocess.Popen`` keyword arguments for a detached, silent child.

    Args:
        windows: Whether this is Windows (adds ``CREATE_NEW_PROCESS_GROUP``).

    Returns:
        The keyword arguments.
    """
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if windows:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # Ctrl-C in our console must not reach CS2
    return kwargs


def kill_tree(proc: Any, run: Callable[..., Any], windows: bool, wait_s: float = 30) -> None:
    """Kill ``proc`` and its children (``taskkill /T /F`` on Windows). Safe to call twice.

    Args:
        proc: A ``subprocess.Popen``-like object.
        run: ``subprocess.run``-like, used for ``taskkill``.
        windows: Whether this is Windows.
        wait_s: How long to wait for the process to exit after the kill.
    """
    if proc is None or proc.poll() is not None:
        return
    if windows:
        run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, check=False, timeout=30)
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        log.warning("process (pid %s) did not exit after kill", proc.pid)
