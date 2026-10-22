"""ntfy.sh notifications (#20). The topic comes from ``DCS2_NTFY_TOPIC`` and is never committed."""

from __future__ import annotations

from collections.abc import Sequence


def notify(message: str, title: str | None = None, priority: str = "default", tags: Sequence[str] = ()) -> None:
    """Post a notification to ``supervisor.ntfy_server``/``supervisor.ntfy_topic``.

    Does nothing (and does not raise) when no topic is configured.

    Args:
        message: Body text.
        title: Optional title.
        priority: ntfy priority (``min`` ... ``urgent``).
        tags: ntfy tags / emoji shortcodes.
    """
    # TODO(#20): requests.post with Title/Priority/Tags headers; swallow network errors.
    raise NotImplementedError("TODO(#20): notify")
