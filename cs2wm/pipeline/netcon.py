"""Client for CS2's ``-netconport`` TCP console (#13), used for chunked recording (option B)."""

from __future__ import annotations


class NetconClient:
    """Sends console commands to a running CS2 over TCP.

    Args:
        host: Console host.
        port: The ``-netconport`` value (2121 in the #14 notes).
        timeout_s: Socket timeout in seconds.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 2121, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def send(self, cmd: str) -> str:
        """Send one console command.

        Args:
            cmd: Command line, e.g. ``"stop; record dcs2_dm_0001"``.

        Returns:
            Console output received before the timeout.

        Raises:
            ConnectionError: If the console port is unreachable.
        """
        # TODO(#13): connect, write cmd + "\n", read until idle.
        raise NotImplementedError("TODO(#13): NetconClient.send")
