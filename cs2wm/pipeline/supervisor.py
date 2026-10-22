"""supervisor: runs stages 1-6 unattended for each demo, as one Windows scheduled task (#20).

Per demo: generate -> segment -> render -> extract -> pack -> sync, with up to ``supervisor.max_attempts`` attempts
and exponential backoff per stage, a disk guard (``supervisor.max_working_gb``), a CS2 patch guard (render a 10 s
smoke test when the build id changes), and ntfy start/hourly/alert/morning messages. State is persisted atomically in
``state/pipeline.json`` so a reboot resumes where it stopped.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig


class StageState(enum.Enum):
    """Per-demo pipeline state: pending -> segmented -> rendered -> extracted -> packed -> synced | failed."""

    PENDING = "pending"
    SEGMENTED = "segmented"
    RENDERED = "rendered"
    EXTRACTED = "extracted"
    PACKED = "packed"
    SYNCED = "synced"
    FAILED = "failed"


@dataclass
class PipelineState:
    """Persistent supervisor state.

    Attributes:
        path: ``state/pipeline.json``.
        demos: State and failure reason per demo id.
    """

    path: Path
    demos: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        """Load the state file (empty state if it does not exist)."""
        raise NotImplementedError("TODO(#20): PipelineState.load")

    def save(self) -> None:
        """Write the state atomically (temp file + rename)."""
        raise NotImplementedError("TODO(#20): PipelineState.save")


class Supervisor:
    """Drives every pending demo through the stages.

    Args:
        cfg: A validated cs2wm config.
        state: Persistent state.
    """

    def __init__(self, cfg: DictConfig, state: PipelineState) -> None:
        self.cfg = cfg
        self.state = state

    def run_until(self, until: str, target_hours: float, profile: str = "dm") -> None:
        """Loop until the deadline or target is reached.

        Args:
            until: Deadline as ``HH:MM`` local time.
            target_hours: Stop once this many QA-passing hours are synced.
            profile: Generation profile (``dm`` or ``smoke``).
        """
        # TODO(#20): default schedule "generate 60 min -> advance everything pending", with guards and ntfy.
        raise NotImplementedError("TODO(#20): Supervisor.run_until")

    def status_table(self) -> str:
        """Return the state table and totals printed by ``cs2wm status``."""
        raise NotImplementedError("TODO(#20): Supervisor.status_table")
