"""Client for CS2's ``-netconport`` TCP console (#13), used for chunked recording (option B)."""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

# CS2 silently truncates a console line at about 512 characters, which drops everything after the cut (#14); stay well below.
MAX_LINE_CHARS = 400


def _split_commands(text: str) -> list[str]:
    """Split a ``;``-joined command line into its commands; a ``;`` inside double quotes does not split."""
    commands: list[str] = []
    current: list[str] = []
    quoted = False
    for ch in text:
        if ch == '"':
            quoted = not quoted
        if ch == ";" and not quoted:
            commands.append("".join(current))
            current = []
        else:
            current.append(ch)
    commands.append("".join(current))
    return [c.strip() for c in commands if c.strip()]


def split_lines(cmd: str, limit: int = MAX_LINE_CHARS) -> list[str]:
    """Split a long ``;``-joined command line into lines of at most ``limit`` characters.

    Commands are kept whole and in order; consecutive commands share a line while they fit.

    Args:
        cmd: Command line, e.g. ``"sv_cheats 1; tv_record x"``.
        limit: Maximum line length.

    Returns:
        The lines to send, each run by CS2 in turn.

    Raises:
        ValueError: If a single command is longer than ``limit`` (it cannot be split without changing its meaning), or
            if ``cmd`` holds a newline.
    """
    if "\n" in cmd or "\r" in cmd:
        raise ValueError("a netcon command must not contain a newline; send each command on its own")
    lines: list[str] = []
    current = ""
    for command in _split_commands(cmd):
        if len(command) > limit:
            raise ValueError(f"netcon command is {len(command)} characters, over the {limit} limit: {command[:60]}...")
        joined = f"{current}; {command}" if current else command
        if len(joined) <= limit:
            current = joined
        else:
            lines.append(current)
            current = command
    if current:
        lines.append(current)
    return lines


class NetconClient:
    """Sends console commands to a running CS2 over TCP.

    One short connection is opened per command. CS2 only opens the port some time after launch, so connecting is
    retried with exponential backoff.

    Args:
        host: Console host.
        port: The ``-netconport`` value (2121 in the #14 notes).
        timeout_s: Socket timeout in seconds, also the cap on one ``send`` call's read phase.
        idle_s: A reply is complete once no data arrived for this long (CS2 sends no end-of-output marker).
        connect_retries: Extra connection attempts per ``send`` after the first one fails.
        backoff_s: Delay before the first retry; doubled after each failure, capped at ``max_backoff_s``.
        max_backoff_s: Cap for the retry delay.
        clock: Monotonic clock (injectable for tests).
        sleep: Sleep function (injectable for tests).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2121,
        timeout_s: float = 5.0,
        idle_s: float = 0.5,
        connect_retries: int = 3,
        backoff_s: float = 0.5,
        max_backoff_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.idle_s = idle_s
        self.connect_retries = connect_retries
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self._clock = clock
        self._sleep = sleep

    def _connect(self, retries: int) -> socket.socket:
        """Open a connection, retrying with exponential backoff."""
        delay = self.backoff_s
        for attempt in range(retries + 1):
            try:
                return socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            except OSError as exc:
                if attempt == retries:
                    raise ConnectionError(f"netcon {self.host}:{self.port} unreachable: {exc}") from exc
                log.debug("netcon connect failed (%s), retry in %.1f s", exc, delay)
                self._sleep(delay)
                delay = min(delay * 2, self.max_backoff_s)
        raise AssertionError("unreachable")  # pragma: no cover

    def send(self, cmd: str, retries: int | None = None) -> str:
        """Send one console command.

        Args:
            cmd: Command line, e.g. ``"stop; record dcs2_dm_0001"``. A line over ``MAX_LINE_CHARS`` is sent as
                several lines on the same connection (see ``split_lines``).
            retries: Override for ``connect_retries``.

        Returns:
            Console output received until the connection was idle for ``idle_s`` (or closed, or ``timeout_s`` passed).

        Raises:
            ConnectionError: If the console port is unreachable or the connection is reset after the command was sent.
            ValueError: If one command alone is over ``MAX_LINE_CHARS`` (nothing is sent).
        """
        payload = "".join(f"{line}\n" for line in split_lines(cmd)).encode("utf-8")
        sock = self._connect(self.connect_retries if retries is None else retries)
        chunks: list[bytes] = []
        with sock:
            try:
                sock.sendall(payload)
            except OSError as exc:
                raise ConnectionError(f"netcon send failed: {exc}") from exc
            deadline = self._clock() + self.timeout_s
            sock.settimeout(self.idle_s)
            while self._clock() < deadline:
                try:
                    data = sock.recv(65536)
                except TimeoutError:  # idle: the reply is complete
                    break
                except OSError as exc:  # reset by CS2: the command may not have run, so do not report success
                    raise ConnectionError(f"netcon connection lost after sending {cmd!r}: {exc}") from exc
                if not data:
                    break
                chunks.append(data)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def wait_ready(self, timeout_s: float, interval_s: float = 2.0) -> None:
        """Block until the console port accepts connections.

        Args:
            timeout_s: Give up after this long.
            interval_s: Pause between attempts.

        Raises:
            ConnectionError: If the port never opened.
        """
        deadline = self._clock() + timeout_s
        while True:
            try:
                self.send("echo dcs2_ready", retries=0)
                return
            except ConnectionError:
                if self._clock() >= deadline:
                    raise
                self._sleep(interval_s)
