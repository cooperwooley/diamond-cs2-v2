"""qa-summary: aggregate ``qa/*.jsonl`` (written by extract) into the numbers the #19 pilot checks (#19).

The pilot passes when the QA drop rate is below ``DROP_RATE_LIMIT`` of the episodes, or the drops are explained by the
failure counts printed here.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig

DROP_RATE_LIMIT = 0.20
_SINCE_RE = re.compile(r"(\d+(?:\.\d+)?)([smhd])")
_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


@dataclass
class KindSummary:
    """QA totals of one episode kind.

    Attributes:
        episodes: Episodes seen.
        passed: Episodes that passed QA.
        frames: Labeled frames of all episodes.
        passed_frames: Labeled frames of the passing episodes.
        failures: Failing episodes by reason (the check name, e.g. ``fire_recall``); an episode with several
            reasons counts once per reason.
    """

    episodes: int = 0
    passed: int = 0
    frames: int = 0
    passed_frames: int = 0
    failures: Counter = field(default_factory=Counter)

    @property
    def pass_rate(self) -> float:
        """Share of episodes that passed (1.0 when there are none)."""
        return self.passed / self.episodes if self.episodes else 1.0

    def passed_hours(self, fps: float) -> float:
        """Hours of passing frames at ``fps``."""
        return self.passed_frames / fps / 3600


@dataclass
class QASummary:
    """Result of ``summarize``.

    Attributes:
        kinds: Totals per episode kind.
        demos: Demos whose QA file was read.
        bad_lines: QA lines that were not valid records (skipped).
    """

    kinds: dict[str, KindSummary] = field(default_factory=dict)
    demos: int = 0
    bad_lines: int = 0

    @property
    def episodes(self) -> int:
        """Episodes over all kinds."""
        return sum(k.episodes for k in self.kinds.values())

    @property
    def drop_rate(self) -> float:
        """Share of episodes that failed QA."""
        total = self.episodes
        return 1.0 - sum(k.passed for k in self.kinds.values()) / total if total else 0.0


def parse_since(text: str) -> float:
    """Parse ``30m``, ``24h``, ``2d`` or ``90s`` into seconds.

    Raises:
        ValueError: If ``text`` is not a number followed by s, m, h or d.
    """
    match = _SINCE_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"invalid --since {text!r}: expected a number and one of s, m, h, d, e.g. 24h")
    return float(match.group(1)) * _UNIT_S[match.group(2)]


def summarize(
    cfg: DictConfig, kind: str | None = None, since_s: float | None = None, now: float | None = None
) -> QASummary:
    """Aggregate every ``qa/*.jsonl``.

    Args:
        cfg: A validated cs2wm config.
        kind: Only this episode kind.
        since_s: Only QA files modified in the last ``since_s`` seconds (extract rewrites a demo's file, so its
            mtime is when the demo was last labeled).
        now: Current time (default: ``time.time()``).

    Returns:
        The totals.
    """
    now = time.time() if now is None else now
    summary = QASummary()
    for path in sorted(Path(cfg.paths.qa).glob("*.jsonl")):
        if since_s is not None and now - path.stat().st_mtime > since_s:
            continue
        summary.demos += 1
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    rec_kind, passed, frames = str(rec["kind"]), bool(rec["passed"]), int(rec["n_frames"])
                    reasons = [str(r) for r in rec.get("reasons", [])]
                except (ValueError, KeyError, TypeError):
                    summary.bad_lines += 1
                    continue
                if kind is not None and rec_kind != kind:
                    continue
                k = summary.kinds.setdefault(rec_kind, KindSummary())
                k.episodes += 1
                k.frames += frames
                if passed:
                    k.passed += 1
                    k.passed_frames += frames
                for reason in reasons:
                    k.failures[reason.split()[0]] += 1
    return summary


def format_summary(summary: QASummary, fps: float) -> str:
    """Render the summary as the text ``cs2wm qa-summary`` prints."""
    lines = [f"QA summary: {summary.demos} demo(s), {summary.episodes} episode(s)"]
    for name in sorted(summary.kinds):
        k = summary.kinds[name]
        lines.append(
            f"  {name}: {k.passed}/{k.episodes} episodes passed ({k.pass_rate:.1%}); "
            f"{k.passed_frames}/{k.frames} frames; {k.passed_hours(fps):.3f} h passing"
        )
        for reason, count in k.failures.most_common():
            lines.append(f"    failed {reason}: {count}")
    total_h = sum(k.passed_hours(fps) for k in summary.kinds.values())
    verdict = "OK" if summary.drop_rate < DROP_RATE_LIMIT else "ABOVE LIMIT, explain with the reasons above"
    lines.append(f"  passing hours (all kinds): {total_h:.3f}")
    lines.append(f"  drop rate: {summary.drop_rate:.1%} (limit < {DROP_RATE_LIMIT:.0%}): {verdict}")
    if summary.bad_lines:
        lines.append(f"  skipped {summary.bad_lines} unreadable QA line(s)")
    return "\n".join(lines)
