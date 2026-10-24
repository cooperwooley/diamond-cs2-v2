"""Process helpers shared by the stages that launch CS2 (generate #13, render #15).

HLAE and CS2 only ever run offline: every launch goes through ``require_offline``.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

from cs2wm import config as cs2wm_config

log = logging.getLogger(__name__)


def require_offline(cmd: Sequence[str]) -> None:
    """Raise ``ValueError`` unless ``cmd`` launches CS2 offline: ``-insecure`` present, nothing that connects.

    This is the single launch-safety check (generate and render call it right before ``Popen``). The value after
    ``-cmdLine`` (the HLAE custom loader passes the CS2 command line as one argument) is split into tokens with
    ``config.split_command_line``, so ``+connect`` inside it is seen, and a quoted path that merely contains
    ``-insecure`` does not count as the flag. The rules are ``config.launch_option_problem``.

    Args:
        cmd: The argv list about to be started.

    Raises:
        ValueError: If ``-insecure`` is missing, or ``-secure``/``connect`` options are present.
    """
    tokens: list[str] = []
    for i, part in enumerate(cmd):
        tokens.append(part)
        if i > 0 and cmd[i - 1].lower() == "-cmdline":
            try:
                tokens.extend(cs2wm_config.split_command_line(part))
            except ValueError as exc:
                raise ValueError(f"refusing to start CS2: unparsable -cmdLine ({exc})") from exc
    problem = cs2wm_config.launch_option_problem(tokens)
    if problem:
        raise ValueError(f"refusing to start CS2: {problem} (HLAE and CS2 only ever run offline)")


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


def kill_by_name(run: Callable[..., Any], windows: bool, names: Sequence[str]) -> None:
    """Kill every process with one of the image ``names`` and their children (``taskkill /IM /T /F``; no-op off Windows).

    The stages kill the process trees they started (``kill_tree``). The supervisor calls this after a failed stage
    as well, so that no stray CS2 or HLAE window is left behind to block the next launch (only one CS2 can run).

    Args:
        run: ``subprocess.run``-like, used for ``taskkill``.
        windows: Whether this is Windows.
        names: Image names, e.g. ``["cs2.exe", "HLAE.exe"]``.
    """
    if not windows:
        return
    for name in names:
        run(["taskkill", "/T", "/F", "/IM", name], capture_output=True, check=False, timeout=30)
