"""Client for CS2's ``-netconport`` TCP console (#13), used for chunked recording (option B)."""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable

log = logging.getLogger(__name__)


class NetconClient:
    """Sends console commands to a running CS2 over TCP.

    One short connection is opened per command. CS2 only opens the port some time after launch, so connecting is
    retried with exponential backoff.

    Args:
        host: Console host.
        port: The ``-netconport`` value (2121 in the #14 notes).
        timeout_s: Socket timeout in seconds, also the cap on one ``send`` call's read phase.
        idle_s: A reply is complete once no data arrived for this long. CONFIRM(#14): CS2 sends no end-of-output marker.
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
            cmd: Command line, e.g. ``"stop; record dcs2_dm_0001"``.
            retries: Override for ``connect_retries``.

        Returns:
            Console output received until the connection was idle for ``idle_s`` (or closed, or ``timeout_s`` passed).

        Raises:
            ConnectionError: If the console port is unreachable.
        """
        sock = self._connect(self.connect_retries if retries is None else retries)
        chunks: list[bytes] = []
        with sock:
            try:
                sock.sendall(cmd.encode("utf-8") + b"\n")
            except OSError as exc:
                raise ConnectionError(f"netcon send failed: {exc}") from exc
            deadline = self._clock() + self.timeout_s
            sock.settimeout(self.idle_s)
            while self._clock() < deadline:
                try:
                    data = sock.recv(65536)
                except TimeoutError:  # idle: the reply is complete
                    break
                except OSError:  # reset by CS2 while it quits
                    break
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
