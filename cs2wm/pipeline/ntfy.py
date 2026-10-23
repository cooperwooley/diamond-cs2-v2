"""ntfy.sh notifications (#20). The topic comes from ``DCS2_NTFY_TOPIC`` and is never committed."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import requests

log = logging.getLogger(__name__)


def notify(
    message: str,
    title: str | None = None,
    priority: str = "default",
    tags: Sequence[str] = (),
    *,
    server: str | None = None,
    topic: str | None = None,
) -> None:
    """Post a notification to ``supervisor.ntfy_server``/``supervisor.ntfy_topic``.

    Does nothing (and does not raise) when no topic is configured, and swallows network errors: a failed
    notification must never stop a night of generation.

    Args:
        message: Body text.
        title: Optional title.
        priority: ntfy priority (``min`` ... ``urgent``).
        tags: ntfy tags / emoji shortcodes.
        server: ntfy server; defaults to ``supervisor.ntfy_server``.
        topic: ntfy topic; defaults to ``supervisor.ntfy_topic`` (from ``DCS2_NTFY_TOPIC``).
    """
    if server is None or topic is None:
        from cs2wm.config import load_config

        sup = load_config().supervisor
        server = server or sup.ntfy_server
        topic = topic or sup.ntfy_topic
    if not topic:
        return
    headers = {"Priority": priority}
    if title:
        headers["Title"] = title
    if tags:
        headers["Tags"] = ",".join(tags)
    try:
        requests.post(f"{server.rstrip('/')}/{topic}", data=message.encode("utf-8"), headers=headers, timeout=10)
    except requests.RequestException as exc:
        log.warning("ntfy notification failed: %s", exc)
