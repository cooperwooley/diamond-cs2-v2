"""generate: unattended offline CS2 bot-deathmatch demo generation, Windows only (#13).

Launches CS2 offline as a listen server, records demos in N-minute chunks to ``demos/<kind>/<demo_id>.dem`` with a
``<demo_id>.json`` sidecar (start time, profile, chunk length, CS2 build id, cfg sha256, cs2wm git sha), and restarts
on a crash or hang. The launcher REFUSES to launch CS2 without ``-insecure``: HLAE and CS2 are only ever run offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from omegaconf import DictConfig


def build_launch_command(cfg: DictConfig, profile: str, netconport: int | None = None) -> list[str]:
    """Build the CS2 launch command line from ``generate.launch_options``.

    Args:
        cfg: A validated cs2wm config.
        profile: ``dm``, ``smoke`` or ``human``; selects ``config/cs2/generate_<profile>.cfg``.
        netconport: Optional ``-netconport`` for option (B) chunking.

    Returns:
        The argv list.

    Raises:
        ValueError: If ``-insecure`` is not among the launch options.
    """
    # TODO(#13): assert -insecure first, then append +exec generate_<profile>.
    raise NotImplementedError("TODO(#13): build_launch_command")


class CS2Process:
    """A supervised CS2 process with a hang watchdog.

    Args:
        cmd: Launch command from ``build_launch_command``.
        demo_dir: Directory watched for demo growth.
        hang_timeout_min: Kill and relaunch after this long without demo-file growth
            (``generate.hang_timeout_min``).
    """

    def __init__(self, cmd: Sequence[str], demo_dir: Path, hang_timeout_min: float) -> None:
        self.cmd = list(cmd)
        self.demo_dir = demo_dir
        self.hang_timeout_min = hang_timeout_min

    def __enter__(self) -> CS2Process:  # noqa: PYI034
        """Launch CS2."""
        # TODO(#13): subprocess.Popen; refuse if "-insecure" is missing from self.cmd.
        raise NotImplementedError("TODO(#13): CS2Process.__enter__")

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Kill CS2 if still running."""
        raise NotImplementedError("TODO(#13): CS2Process.__exit__")

    def poll(self) -> str:
        """Advance the watchdog state machine.

        Returns:
            ``"running"``, ``"exited"`` or ``"hung"``.
        """
        raise NotImplementedError("TODO(#13): CS2Process.poll")

    def kill(self) -> None:
        """Kill the process tree."""
        raise NotImplementedError("TODO(#13): CS2Process.kill")


def run(cfg: DictConfig, profile: str, hours: float, chunk_min: float | None = None) -> list[Path]:
    """CLI stage: generate demos for ``hours`` hours.

    Args:
        cfg: A validated cs2wm config.
        profile: ``dm``, ``smoke`` or ``human``.
        hours: Wall-clock budget.
        chunk_min: Chunk length override (``generate.chunk_min``).

    Returns:
        Paths of the kept demos. A truncated last chunk is kept only if ``cs2wm inspect-demo`` parses it.
    """
    # TODO(#13): patch guard (cs2_build_id vs generate.expected_build), chunk schedule, watchdog loop, sidecars.
    raise NotImplementedError("TODO(#13): generate.run")
