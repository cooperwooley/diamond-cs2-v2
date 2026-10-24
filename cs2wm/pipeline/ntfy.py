"""ntfy.sh notifications (#20). The topic comes from ``DCS2_NTFY_TOPIC`` and is never committed.

``notify`` does the posting. The ``format_*`` functions are pure: they turn numbers into the text of the four
supervisor messages (start, hourly heartbeat, alert, morning summary), so the wording is testable without a network.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

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
        # The exception text holds the request URL, which ends in the secret topic: log only its type.
        log.warning("ntfy notification failed (%s)", type(exc).__name__)


@dataclass(frozen=True)
class Message:
    """A notification ready for ``notify(body, title=..., priority=..., tags=...)``."""

    title: str
    body: str
    priority: str = "default"
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    """The numbers of a heartbeat or the end of a night (all hours are hours of demo time at ``timing.fps``).

    Attributes:
        generated_h: Demo time of the demos that were segmented.
        rendered_h: Rendered frames.
        synced_h: QA-passing frames that reached the Mill.
        qa_pass_rate: QA-passing episodes / extracted episodes, or None before the first extract.
        working_gb: Size of the working set (frames, raw, labels, hdf5, hdf5_fullres).
        free_gb: Free space on the data drive.
        stage: What the supervisor is doing, e.g. ``render dm-0007``.
        counts: Demos per state.
    """

    generated_h: float
    rendered_h: float
    synced_h: float
    qa_pass_rate: float | None
    working_gb: float
    free_gb: float
    stage: str = "idle"
    counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class NightSummary:
    """What the morning summary reports.

    Attributes:
        snapshot: Totals at the end of the night.
        target_hours: The night's target.
        started: When the supervisor started.
        ended: When it stopped.
        reason: Why it stopped (deadline, target reached, a guard, ...).
        failures: Failed demos as ``(stage, reason, count)``.
        stage_wall_s: Wall seconds spent in each stage tonight.
        stage_runs: Successful runs of each stage tonight.
        alerts: Alert titles sent tonight.
    """

    snapshot: Snapshot
    target_hours: float
    started: datetime
    ended: datetime
    reason: str
    failures: Sequence[tuple[str, str, int]] = ()
    stage_wall_s: Mapping[str, float] = field(default_factory=dict)
    stage_runs: Mapping[str, int] = field(default_factory=dict)
    alerts: Sequence[str] = ()


def _hhmm(t: datetime) -> str:
    return t.strftime("%a %H:%M")


def _duration(seconds: float) -> str:
    minutes = round(seconds / 60)
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate:.0%}"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{n} {state}" for state, n in counts.items() if n) or "no demos"


def format_start(
    target_hours: float,
    deadline: datetime,
    now: datetime,
    build: int | None,
    working_gb: float,
    free_gb: float,
    profile: str,
) -> Message:
    """The message sent when a night starts: target, deadline, CS2 build and disk."""
    body = (
        f"Target {target_hours:g} h synced, profile {profile}.\n"
        f"Deadline {_hhmm(deadline)} ({_duration((deadline - now).total_seconds())} from now).\n"
        f"CS2 build {build if build is not None else 'unknown'}.\n"
        f"Disk: working set {working_gb:.1f} GB, {free_gb:.0f} GB free."
    )
    return Message("dcs2: night started", body, "default", ("moon",))


def format_heartbeat(snap: Snapshot, target_hours: float, now: datetime, deadline: datetime) -> Message:
    """The hourly heartbeat: hours generated, rendered and synced, QA pass rate, disk and the current stage."""
    body = (
        f"Hours: {snap.generated_h:.2f} generated, {snap.rendered_h:.2f} rendered, "
        f"{snap.synced_h:.2f}/{target_hours:g} synced.\n"
        f"QA pass rate {_rate(snap.qa_pass_rate)}.\n"
        f"Disk: working set {snap.working_gb:.1f} GB, {snap.free_gb:.0f} GB free.\n"
        f"Now: {snap.stage}. Demos: {_counts(snap.counts)}.\n"
        f"{_duration((deadline - now).total_seconds())} to the deadline."
    )
    return Message("dcs2: heartbeat", body, "low", ("hourglass_flowing_sand",))


def format_alert(kind: str, detail: str) -> Message:
    """An immediate high-priority alert (repeated failures or a guard trip).

    Args:
        kind: Short headline, e.g. ``CS2 patched``.
        detail: What happened and what the supervisor does about it.
    """
    return Message(f"dcs2 alert: {kind}", detail, "high", ("rotating_light",))


def format_summary(s: NightSummary) -> Message:
    """The morning summary: totals, failures by stage and reason, and time per stage."""
    snap = s.snapshot
    lines = [
        f"{_hhmm(s.started)} to {_hhmm(s.ended)} ({_duration((s.ended - s.started).total_seconds())}): {s.reason}.",
        (
            f"Synced {snap.synced_h:.2f} h of {s.target_hours:g} h target "
            f"({snap.generated_h:.2f} generated, {snap.rendered_h:.2f} rendered). QA pass rate {_rate(snap.qa_pass_rate)}."
        ),
        f"Demos: {_counts(snap.counts)}.",
    ]
    if s.failures:
        lines.append("Failures by stage and reason:")
        lines += [f"  {stage}: {reason} (x{n})" for stage, reason, n in s.failures]
    else:
        lines.append("No failed demos.")
    if s.stage_wall_s:
        per_stage = [
            f"{stage} {_duration(wall)} ({s.stage_runs.get(stage, 0)} runs)" for stage, wall in s.stage_wall_s.items()
        ]
        lines.append("Time per stage: " + ", ".join(per_stage) + ".")
    lines.append(f"Disk: working set {snap.working_gb:.1f} GB, {snap.free_gb:.0f} GB free.")
    if s.alerts:
        lines.append("Alerts: " + "; ".join(s.alerts) + ".")
    ok = not s.alerts
    title = "dcs2: morning summary" + ("" if ok else " (needs attention)")
    return Message(title, "\n".join(lines), "default" if ok else "high", ("sunrise",) if ok else ("warning",))
