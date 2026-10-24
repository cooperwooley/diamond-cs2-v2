"""sync: push packed HDF5, demos and QA to the Mill, verify checksums, free local disk (#16).

Pushes ``hdf5/``, ``hdf5_fullres/``, ``demos/`` and ``qa/`` with ``rsync -a --partial`` (via ``wsl.exe -e rsync`` on
Windows, with ``wslpath`` translation; ``rsync`` directly on Linux) using ``ssh -o BatchMode=yes``. Idempotent and
resumable. Only HDF5 whose remote sha256 matches the local ``manifest.jsonl`` is ever deleted; demos stay local.

Layout (see ``cs2wm.pipeline.pack``): a demo's low-res episodes and ``manifest.jsonl`` live in
``hdf5/<kind>/<demo_id>/``, its full-res subset in ``hdf5_fullres/<kind>/<demo_id>/``, the demo itself in
``demos/<kind>/<demo_id>.dem`` and its QA in ``qa/<demo_id>.jsonl``. Each manifest line is a JSON object with a
``file`` (``<name>.hdf5``, or ``fullres/<name>.hdf5`` for the full-res twin) and its hex ``sha256``; if a file has
several lines the last one wins (a re-pack).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig

from cs2wm.pipeline.pack import sha256_file

logger = logging.getLogger(__name__)

# Subdirectories pushed to ``<remote_root>/<name>/``, as config path keys (same names as on the Mill, issue #3).
_TREES = ("hdf5", "hdf5_fullres", "demos", "qa")
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_MANIFEST_FILE_RE = re.compile(r"(fullres/)?[A-Za-z0-9][A-Za-z0-9._-]*\.hdf5")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
# ``sync.remote_root`` may only hold these characters plus ``$USER``/``${USER}``, so it is safe inside double quotes.
_ROOT_RE = re.compile(r"[A-Za-z0-9_./-]+")
_ROOT_VAR_RE = re.compile(r"\$\{USER\}|\$USER")
# BatchMode: never prompt. The rest make a dead connection fail in about a minute instead of hanging unattended.
_SSH_OPTS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
)
_RSYNC_IO_TIMEOUT_S = 300


class SyncError(RuntimeError):
    """Raised when the Mill is unreachable without a password prompt or a transport command cannot run."""


class Transport:
    """Builds and runs the ssh/rsync commands; the only place that knows about the remote host and WSL.

    Tests subclass it (override ``ssh_argv`` and ``remote_spec``) so a local directory acts as the Mill.

    Args:
        host: ``~/.ssh/config`` alias of the Mill (``sync.host``).
        windows: Route commands through ``wsl.exe -e`` (default: ``os.name == "nt"``). The ssh key lives in WSL.
        runner: ``subprocess.run`` stand-in (receives bytes on stdin, may return bytes or str).
    """

    def __init__(
        self,
        host: str,
        windows: bool | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        self.host = host
        self.windows = os.name == "nt" if windows is None else windows
        self._runner = runner

    def _wrap(self, argv: list[str]) -> list[str]:
        return ["wsl.exe", "-e", *argv] if self.windows else argv

    def ssh_argv(self, remote_cmd: str) -> list[str]:
        """Argv that runs ``remote_cmd`` (a shell command line) on the Mill without ever prompting."""
        return self._wrap(["ssh", *_SSH_OPTS, self.host, remote_cmd])

    def rsync_argv(self, src: str, dest: str, files_from_stdin: bool = False) -> list[str]:
        """Argv for ``rsync -a --partial`` of ``src`` (already a path the rsync binary can read) to ``dest``.

        Args:
            src: Local source, ending in ``/`` to copy the directory's contents.
            dest: Remote destination as returned by ``remote_spec``.
            files_from_stdin: Copy only the files listed on stdin (relative to ``src``), NUL-separated.

        Returns:
            The argv list.
        """
        argv = [
            "rsync",
            "-a",
            "--partial",
            f"--timeout={_RSYNC_IO_TIMEOUT_S}",
            "--exclude=*.tmp",
            # pack stages, swaps and deletes whole demo dirs under these suffixes; never push a half-done one
            "--exclude=*.staging/",
            "--exclude=*.old/",
            "--exclude=*.deleting/",
            "-e",
            shlex.join(["ssh", *_SSH_OPTS]),
        ]
        if files_from_stdin:
            argv += ["--files-from=-", "--from0"]
        return self._wrap([*argv, src, dest])

    def remote_spec(self, path: str) -> str:
        """rsync destination for an absolute remote path."""
        return f"{self.host}:{path}"

    def local_path(self, path: Path) -> str:
        """Path as the rsync binary sees it: ``wslpath`` translation on Windows, ``as_posix`` elsewhere."""
        if not self.windows:
            return path.as_posix()
        proc = self.run(["wsl.exe", "-e", "wslpath", "-a", "-u", str(path)], timeout=60)
        if proc.returncode != 0 or not proc.stdout.strip():
            raise SyncError(f"wslpath failed for {path}: {proc.stderr.strip()}")
        return proc.stdout.strip()

    def run(
        self, argv: Sequence[str], stdin: str | None = None, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run an argv list (never through a shell) and capture its output.

        Stdin is sent as UTF-8 bytes, so text mode cannot turn ``\\n`` into ``\\r\\n`` on Windows.

        Args:
            argv: Command and arguments.
            stdin: Text for the command's stdin, sent byte for byte.
            timeout: Seconds before the command is killed.

        Returns:
            The finished process with decoded output; the caller checks ``returncode``.

        Raises:
            SyncError: If the executable cannot be started or the timeout expires.
        """
        try:
            proc = self._runner(
                list(argv),
                input=None if stdin is None else stdin.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except OSError as e:
            raise SyncError(f"cannot run {argv[0]!r}: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise SyncError(f"{argv[0]!r} timed out after {timeout}s: {shlex.join(map(str, argv))[:200]}") from e
        out, err = (
            b.decode("utf-8", "replace") if isinstance(b, bytes) else (b or "") for b in (proc.stdout, proc.stderr)
        )
        return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


@dataclass
class DemoResult:
    """Outcome of one demo.

    Attributes:
        demo_id: Demo id.
        kind: Episode kind (``dm``, ``human``, ``smoke``).
        pushed: True if the rsyncs that carry this demo's ``demos/`` and ``qa/`` files succeeded (the HDF5 is covered
            by checksum verification instead).
        expected: Manifest entries (distinct files).
        verified: Entries whose remote sha256 matches the manifest.
        deleted: Local HDF5 files deleted by this run.
        errors: Everything that went wrong; any entry blocks ``synced``.
    """

    demo_id: str
    kind: str
    pushed: bool = False
    expected: int = 0
    verified: int = 0
    deleted: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def synced(self) -> bool:
        """True once the whole manifest is on the Mill with matching checksums (what #20 records)."""
        return self.pushed and self.expected > 0 and self.verified == self.expected and not self.errors


@dataclass
class SyncSummary:
    """Result of ``run``: one ``DemoResult`` per demo with a manifest, rsync errors, and skipped bad directories."""

    demos: list[DemoResult] = field(default_factory=list)
    push_errors: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def by_demo(self) -> dict[str, DemoResult]:
        """Results keyed by demo id."""
        return {d.demo_id: d for d in self.demos}

    @property
    def ok(self) -> bool:
        """True if nothing failed anywhere."""
        return not self.push_errors and not self.skipped and all(not d.errors for d in self.demos)


@dataclass(frozen=True)
class _Entry:
    rel: str  # path under the sync root, e.g. "hdf5/dm/demo0/ep0.hdf5"
    local: Path
    sha256: str


def _ssh_timeout(cfg: DictConfig) -> float:
    return float(cfg.sync.ssh_timeout_s)


def _remote_root(transport: Transport, cfg: DictConfig) -> str:
    """Expand ``$USER`` on the Mill: the config value goes through the remote shell, unquoted but restricted."""
    root = str(cfg.sync.remote_root)
    if not _ROOT_RE.fullmatch(_ROOT_VAR_RE.sub("", root)):
        raise SyncError(f"sync.remote_root {root!r} may only contain [A-Za-z0-9_./-] and $USER")
    # No quotes: the root is restricted above, and embedded quotes do not survive wsl.exe's argument re-parsing.
    proc = transport.run(transport.ssh_argv(f"printf %s {root}"), timeout=_ssh_timeout(cfg))
    lines = proc.stdout.strip().splitlines()
    # A login banner may precede the path, so take the last line.
    resolved = lines[-1].strip() if lines else ""
    if proc.returncode != 0 or not resolved.startswith("/"):
        raise SyncError(f"could not resolve sync.remote_root {root!r} on {transport.host}: {proc.stderr.strip()}")
    return resolved


def _connect(transport: Transport, cfg: DictConfig) -> str:
    """Fail fast if key auth is missing, then create the remote directories.

    Returns:
        The expanded absolute remote root.

    Raises:
        SyncError: If ssh cannot log in non-interactively or the remote directories cannot be made.
    """
    proc = transport.run(transport.ssh_argv("true"), timeout=_ssh_timeout(cfg))
    if proc.returncode != 0:
        raise SyncError(
            f"cannot log in to {transport.host!r} without a password (ssh -o BatchMode=yes {transport.host} true "
            f"exited {proc.returncode}: {proc.stderr.strip() or 'no output'}). Key-based SSH is not set up: "
            "finish issue #3 (ssh-keygen, the 'Host mill' alias in ~/.ssh/config, ssh-copy-id) first."
        )
    root = _remote_root(transport, cfg)
    dirs = " ".join(shlex.quote(f"{root}/{t}") for t in _TREES)
    proc = transport.run(transport.ssh_argv(f"mkdir -p {dirs}"), timeout=_ssh_timeout(cfg))
    if proc.returncode != 0:
        raise SyncError(f"mkdir -p under {root} failed on {transport.host}: {proc.stderr.strip()}")
    return root


def _demo_files(cfg: DictConfig, tree: str, demo_id: str) -> list[str]:
    """Paths (relative to ``paths.<tree>``) that belong to one demo."""
    base = Path(cfg.paths[tree])
    if tree in ("hdf5", "hdf5_fullres"):
        pattern = f"*/{demo_id}/*"
    elif tree == "demos":
        pattern = f"*/{demo_id}.*"
    else:
        pattern = f"{demo_id}.*"
    return sorted(p.relative_to(base).as_posix() for p in base.glob(pattern) if p.is_file())


def _push_trees(transport: Transport, cfg: DictConfig, root: str, demo_id: str | None) -> dict[str, str]:
    """Rsync each tree; keep going after a failure so one bad directory does not block the rest.

    Returns:
        Error message by tree name, for the trees that failed.
    """
    errors: dict[str, str] = {}
    for tree in _TREES:
        base = Path(cfg.paths[tree])
        if not base.is_dir():
            continue
        files = None if demo_id is None else _demo_files(cfg, tree, demo_id)
        if files is not None and not files:
            continue
        argv = transport.rsync_argv(
            transport.local_path(base).rstrip("/") + "/",
            transport.remote_spec(f"{root}/{tree}/"),
            files_from_stdin=files is not None,
        )
        # NUL-separated (--from0) so no newline or carriage return can end up in a name.
        stdin = None if files is None else "".join(f + "\0" for f in files)
        try:
            proc = transport.run(argv, stdin=stdin, timeout=float(cfg.sync.rsync_timeout_s))
        except SyncError as e:
            errors[tree] = f"rsync of {tree}/ failed: {e}"
            continue
        if proc.returncode != 0:
            errors[tree] = f"rsync of {tree}/ failed (exit {proc.returncode}): {proc.stderr.strip()}"
    return errors


def push(cfg: DictConfig, demo_id: str | None = None, transport: Transport | None = None) -> None:
    """Rsync the local tree to ``<sync.host>:<sync.remote_root>``.

    Args:
        cfg: A validated cs2wm config.
        demo_id: Restrict to one demo (all pending demos if None).
        transport: Command builder/runner (default: ``Transport(sync.host)``).

    Raises:
        SyncError: If key auth is not set up or rsync fails (after attempting every directory).
        ValueError: If ``demo_id`` is not a plain id.
    """
    if demo_id is not None:
        _check_demo_id(demo_id)
    transport = transport or Transport(str(cfg.sync.host))
    errors = _push_trees(transport, cfg, _connect(transport, cfg), demo_id)
    if errors:
        raise SyncError("; ".join(errors.values()))


def _check_demo_id(demo_id: str) -> None:
    if not _NAME_RE.fullmatch(demo_id) or ".." in demo_id:
        raise ValueError(f"invalid demo_id {demo_id!r}: expected [A-Za-z0-9][A-Za-z0-9._-]* without '..'")


def _read_manifest(cfg: DictConfig, kind: str, demo_id: str) -> tuple[list[_Entry], list[str]]:
    """Parse ``hdf5/<kind>/<demo_id>/manifest.jsonl``; bad lines become errors, never entries."""
    hdf5_dir = Path(cfg.paths.hdf5) / kind / demo_id
    fullres_dir = Path(cfg.paths.hdf5_fullres) / kind / demo_id
    entries: dict[str, _Entry] = {}
    errors: list[str] = []
    with open(hdf5_dir / "manifest.jsonl", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                name, sha = rec["file"], rec["sha256"]
            except (ValueError, KeyError, TypeError):
                errors.append(f"manifest line {n} is not a record with 'file' and 'sha256'")
                continue
            if not (isinstance(name, str) and _MANIFEST_FILE_RE.fullmatch(name) and isinstance(sha, str)):
                errors.append(f"manifest line {n}: bad file or sha256")
                continue
            if not _SHA256_RE.fullmatch(sha):
                errors.append(f"manifest line {n}: sha256 for {name} is not 64 lowercase hex digits")
                continue
            if name.startswith("fullres/"):
                rel, local = f"hdf5_fullres/{kind}/{demo_id}/{name[8:]}", fullres_dir / name[8:]
            else:
                rel, local = f"hdf5/{kind}/{demo_id}/{name}", hdf5_dir / name
            entries[rel] = _Entry(rel, local, sha)  # last line wins
    return list(entries.values()), errors


def _remote_check(transport: Transport, cfg: DictConfig, root: str, entries: Sequence[_Entry]) -> set[str]:
    """One ``sha256sum -c`` on the Mill for ``entries``.

    Returns:
        The ``rel`` paths reported ``OK``. A missing or differing file is simply absent.

    Raises:
        SyncError: If the command itself failed (ssh down, sha256sum missing, unexpected exit code).
    """
    if not entries:
        return set()
    listing = "".join(f"{e.sha256}  {e.rel}\n" for e in entries)
    proc = transport.run(
        transport.ssh_argv(f"cd {shlex.quote(root)} && LC_ALL=C sha256sum -c -"),
        stdin=listing,
        timeout=float(cfg.sync.verify_timeout_s),
    )
    # sha256sum exits 1 when some file failed, which is data, not an error; anything else is.
    if proc.returncode not in (0, 1):
        raise SyncError(f"remote sha256sum -c exited {proc.returncode}: {proc.stderr.strip()}")
    ok = {line[: -len(": OK")] for line in proc.stdout.splitlines() if line.endswith(": OK")}
    return ok & {e.rel for e in entries}


def _find_demos(cfg: DictConfig, demo_id: str | None) -> tuple[list[tuple[str, str]], list[str]]:
    """``(kind, demo_id)`` of every demo with a manifest, plus messages for directories with unusable names."""
    base = Path(cfg.paths.hdf5)
    pattern = f"*/{demo_id or '*'}/manifest.jsonl"
    found: list[tuple[str, str]] = []
    skipped: list[str] = []
    for p in sorted(base.glob(pattern)):
        kind, name = p.parent.parent.name, p.parent.name
        if all(_NAME_RE.fullmatch(n) and ".." not in n for n in (kind, name)):
            found.append((kind, name))
        else:
            skipped.append(f"skipped {p.parent}: kind and demo id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    return found, skipped


def _verify_demo(
    transport: Transport, root: str, cfg: DictConfig, kind: str, demo_id: str
) -> tuple[list[Path], int, int, list[str]]:
    """Verify one demo against its manifest.

    Returns:
        ``(deletable, verified, expected, errors)``: local files that exist, match the manifest and are on the
        Mill with the same sha256; the count of remotely verified entries; the manifest size; and the problems.
    """
    entries, errors = _read_manifest(cfg, kind, demo_id)
    ok = _remote_check(transport, cfg, root, entries)
    deletable: list[Path] = []
    for e in entries:
        if e.rel not in ok:
            errors.append(f"{e.rel}: remote file is missing or its sha256 differs from the manifest")
        elif e.local.exists():
            # The Mill copy matches the manifest; only delete if the local file is that same content.
            if sha256_file(e.local) == e.sha256:
                deletable.append(e.local)
            else:
                errors.append(f"{e.rel}: local file differs from the manifest, kept")
    return deletable, len(ok), len(entries), errors


def verify_remote(cfg: DictConfig, demo_id: str, transport: Transport | None = None) -> list[Path]:
    """Check remote sha256s against the local manifest, one ``ssh ... sha256sum -c`` per demo.

    Args:
        cfg: A validated cs2wm config.
        demo_id: Demo to verify.
        transport: Command builder/runner (default: ``Transport(sync.host)``).

    Returns:
        Local files that still exist, match the manifest, and whose remote copy matches too.

    Raises:
        SyncError: If the Mill cannot be reached, the verification command fails, or the demo has no manifest.
        ValueError: If ``demo_id`` is not a plain id.
    """
    _check_demo_id(demo_id)
    demos, _ = _find_demos(cfg, demo_id)
    if not demos:
        raise SyncError(f"no manifest.jsonl for demo {demo_id!r} under {cfg.paths.hdf5}")
    transport = transport or Transport(str(cfg.sync.host))
    root = _connect(transport, cfg)
    return [p for kind, did in demos for p in _verify_demo(transport, root, cfg, kind, did)[0]]


def delete_verified(files: list[Path]) -> int:
    """Delete local HDF5 files that ``verify_remote`` confirmed. Never deletes anything else.

    Args:
        files: Verified local files.

    Returns:
        Number of files deleted.

    Raises:
        ValueError: If any path is not a ``.hdf5`` file (nothing is deleted then).
    """
    bad = [str(p) for p in files if Path(p).suffix != ".hdf5"]
    if bad:
        raise ValueError(f"refusing to delete non-HDF5 files: {bad}")
    deleted = 0
    for p in files:
        try:
            Path(p).unlink()
            deleted += 1
        except FileNotFoundError:
            pass
    return deleted


def run(cfg: DictConfig, transport: Transport | None = None) -> SyncSummary:
    """CLI stage: push, verify, then delete verified HDF5 (``sync.delete_local_hdf5``).

    A failure in one demo (or one rsync) never stops the others, and a re-run continues where this one stopped.

    Args:
        cfg: A validated cs2wm config.
        transport: Command builder/runner (default: ``Transport(sync.host)``).

    Returns:
        Per-demo results (pushed, verified, deleted, errors) and any rsync errors; also logged.

    Raises:
        SyncError: If key auth is not set up (checked before anything is transferred).
    """
    transport = transport or Transport(str(cfg.sync.host))
    root = _connect(transport, cfg)
    failed = _push_trees(transport, cfg, root, None)
    demos, skipped = _find_demos(cfg, None)
    summary = SyncSummary(push_errors=list(failed.values()), skipped=skipped)
    for err in [*summary.push_errors, *summary.skipped]:
        logger.error("sync: %s", err)

    # HDF5 safety rests on checksum verification, so a failed hdf5 push only shows up as unverified files.
    pushed = not ({"demos", "qa"} & failed.keys())
    for kind, demo_id in demos:
        result = DemoResult(demo_id, kind, pushed=pushed)
        summary.demos.append(result)
        try:
            deletable, result.verified, result.expected, result.errors = _verify_demo(
                transport, root, cfg, kind, demo_id
            )
            # Delete only after verification returned cleanly, and only what it confirmed.
            if cfg.sync.delete_local_hdf5:
                result.deleted = delete_verified(deletable)
        except (SyncError, OSError) as e:
            result.errors.append(f"verification failed, nothing deleted: {e}")
        logger.log(
            logging.INFO if not result.errors else logging.WARNING,
            "sync %s/%s: pushed=%s verified=%d/%d deleted=%d errors=%d",
            kind,
            demo_id,
            result.pushed,
            result.verified,
            result.expected,
            result.deleted,
            len(result.errors),
        )
        for err in result.errors:
            logger.warning("sync %s/%s: %s", kind, demo_id, err)
    logger.info(
        "sync done: %d demos, %d synced, %d files deleted, %d errors",
        len(summary.demos),
        sum(d.synced for d in summary.demos),
        sum(d.deleted for d in summary.demos),
        len(summary.push_errors) + len(summary.skipped) + sum(len(d.errors) for d in summary.demos),
    )
    return summary
