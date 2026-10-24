"""Tests for cs2wm.pipeline.netcon and cs2wm.pipeline.ntfy, against a local fake TCP server."""

import socket
import socketserver
import struct
import threading
import time

import pytest
import requests

from cs2wm.pipeline import ntfy
from cs2wm.pipeline.netcon import NetconClient


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int = 0, hold_open: bool = True) -> None:
        self.received: list[str] = []
        self.hold_open = hold_open
        super().__init__(("127.0.0.1", port), _Handler)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline().decode().strip()
        self.server.received.append(line)
        self.wfile.write(f"console: {line}\n".encode())
        self.wfile.flush()
        if self.server.hold_open:  # like CS2: the reply ends by silence, not by closing
            self.request.settimeout(5)
            try:
                self.request.recv(1)
            except OSError:
                pass


class _ResetHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.rfile.readline()
        self.request.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self.request.close()  # linger 0: close sends RST, not FIN


def test_reset_after_send_raises() -> None:
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _ResetHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = NetconClient(port=srv.server_address[1], idle_s=1, timeout_s=3)
        with pytest.raises(ConnectionError, match="lost"):
            client.send("stop")
    finally:
        srv.shutdown()
        srv.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    srv = _Server()
    yield srv
    srv.stop()


def test_send_returns_reply_after_idle(server: _Server) -> None:
    client = NetconClient(port=server.port, idle_s=0.2, timeout_s=3)
    t0 = time.monotonic()
    assert client.send("stop; record x") == "console: stop; record x\n"
    assert time.monotonic() - t0 < 2  # ended by idleness, not by the overall timeout
    assert server.received == ["stop; record x"]


def test_send_returns_when_server_closes() -> None:
    srv = _Server(hold_open=False)
    try:
        assert NetconClient(port=srv.port, idle_s=2, timeout_s=5).send("status") == "console: status\n"
    finally:
        srv.stop()


def test_connect_retries_with_exponential_backoff() -> None:
    delays: list[float] = []
    client = NetconClient(port=_free_port(), connect_retries=3, backoff_s=0.5, max_backoff_s=1.5, sleep=delays.append)
    with pytest.raises(ConnectionError, match="unreachable"):
        client.send("echo hi")
    assert delays == [0.5, 1.0, 1.5]


def test_connect_succeeds_once_the_port_opens() -> None:
    port = _free_port()
    servers: list[_Server] = []

    def sleep(_: float) -> None:  # CS2 opens its port while we wait
        if not servers:
            servers.append(_Server(port))

    client = NetconClient(port=port, idle_s=0.2, timeout_s=3, connect_retries=5, sleep=sleep)
    try:
        assert client.send("echo hi") == "console: echo hi\n"
    finally:
        for s in servers:
            s.stop()


def test_wait_ready_times_out() -> None:
    now = [0.0]

    def sleep(s: float) -> None:
        now[0] += s

    client = NetconClient(port=_free_port(), clock=lambda: now[0], sleep=sleep)
    with pytest.raises(ConnectionError):
        client.wait_ready(10, interval_s=2)
    assert 10 <= now[0] <= 14


def test_wait_ready_returns_when_up(server: _Server) -> None:
    NetconClient(port=server.port, idle_s=0.1).wait_ready(5)
    assert server.received == ["echo dcs2_ready"]


class _Post:
    def __init__(self, exc: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.exc = exc

    def __call__(self, url: str, **kwargs) -> None:
        self.calls.append((url, kwargs))
        if self.exc:
            raise self.exc


def test_notify_posts_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    post = _Post()
    monkeypatch.setattr(requests, "post", post)
    ntfy.notify("CS2 patched", title="dcs2", priority="high", tags=["warning", "x"], server="https://n.sh/", topic="t")
    ((url, kwargs),) = post.calls
    assert url == "https://n.sh/t"
    assert kwargs["data"] == b"CS2 patched"
    assert kwargs["headers"] == {"Priority": "high", "Title": "dcs2", "Tags": "warning,x"}
    assert kwargs["timeout"] > 0


def test_notify_without_topic_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DCS2_NTFY_TOPIC", raising=False)
    post = _Post()
    monkeypatch.setattr(requests, "post", post)
    ntfy.notify("hi")  # topic from the config: DCS2_NTFY_TOPIC is unset
    assert post.calls == []


def test_notify_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", _Post(requests.ConnectionError("down")))
    ntfy.notify("hi", server="https://n.sh", topic="t")
