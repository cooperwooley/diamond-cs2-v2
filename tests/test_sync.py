"""Tests for cs2wm.pipeline.sync, with a local temp directory standing in for the Mill."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cs2wm.cli import main
from cs2wm.config import load_config
from cs2wm.pipeline import sync
from cs2wm.pipeline.pack import append_manifest, write_episode
from cs2wm.pipeline.sync import DemoResult, SyncError, Transport

H, W = 150, 280
SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=30",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
]


class LocalTransport(Transport):
    """The "Mill" is a local directory: remote commands run in bash, rsync copies to a plain path."""

    def __init__(self) -> None:
        super().__init__("mill", windows=False)
        self.ssh_cmds: list[str] = []

    def ssh_argv(self, remote_cmd: str) -> list[str]:
        self.ssh_cmds.append(remote_cmd)
        return ["bash", "-c", remote_cmd]

    def remote_spec(self, path: str) -> str:
        return path


class NoPushTransport(LocalTransport):
    """Rsync does nothing, so the remote keeps whatever state the test arranged."""

    def rsync_argv(self, src: str, dest: str, files_from_stdin: bool = False) -> list[str]:
        return ["true"]


class BrokenCheckTransport(LocalTransport):
    """``sha256sum`` cannot run (like a dropped ssh connection, exit 255)."""

    def ssh_argv(self, remote_cmd: str) -> list[str]:
        if "sha256sum" in remote_cmd:
            return ["bash", "-c", "echo connection lost >&2; exit 255"]
        return super().ssh_argv(remote_cmd)


def real_run(argv, **kw):
    """The real ``subprocess.run`` (callers pass ``check`` already), for spies that wrap the default runner."""
    return subprocess.run(argv, **kw)  # noqa: PLW1510


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    return tmp_path / "mill" / "dcs2"


@pytest.fixture
def cfg(tmp_path: Path, remote: Path):
    return load_config(
        overrides=[f"paths.root={tmp_path / 'data'}", f"sync.remote_root={remote}", "pack.fullres_mod=1"]
    )


def make_demo(cfg, demo_id: str, kind: str = "dm", episodes: int = 2, seed: int = 0) -> list[Path]:
    """Pack real episodes with pack.write_episode and record them in the manifest like pack.run will."""
    rng = np.random.default_rng(seed)
    out = Path(cfg.paths.hdf5) / kind / demo_id
    full = Path(cfg.paths.hdf5_fullres) / kind / demo_id
    paths = []
    for i in range(episodes):
        frames = rng.integers(0, 256, (3, H, W, 3), dtype=np.uint8)
        actions = rng.uniform(-1, 1, (3, 51)).astype(np.float32)
        meta = {
            "episode_id": f"{demo_id}_e{i}",
            "demo_id": demo_id,
            "kind": kind,
            "player": None,
            "start_tick": 128,
            "cs2_build": "1",
            "demoparser2_version": "0",
            "cs2wm_git_sha": "abc",
        }
        res = write_episode(frames, actions, meta, out, cfg, fullres_dir=full)
        for name, sha in res.sha256.items():
            append_manifest(out / "manifest.jsonl", {"file": name, "frames": res.n_frames, "sha256": sha})
        paths += [res.low_res_path, res.fullres_path]
    demo = Path(cfg.paths.demos) / kind / f"{demo_id}.dem"
    demo.parent.mkdir(parents=True, exist_ok=True)
    demo.write_bytes(demo_id.encode() * 100)
    qa = Path(cfg.paths.qa) / f"{demo_id}.jsonl"
    qa.parent.mkdir(parents=True, exist_ok=True)
    qa.write_text('{"ok": true}\n')
    return paths


def corrupt_keeping_stat(remote_file: Path, local_file: Path) -> None:
    """Flip bytes but keep size and mtime, so rsync's quick check cannot repair it."""
    data = bytearray(remote_file.read_bytes())
    data[len(data) // 2] ^= 0xFF
    remote_file.write_bytes(bytes(data))
    st = local_file.stat()
    os.utime(remote_file, ns=(st.st_atime_ns, st.st_mtime_ns))


# --- happy path -------------------------------------------------------------------------------------------------


def test_run_pushes_verifies_and_deletes_hdf5(cfg, remote) -> None:
    files = make_demo(cfg, "d1")
    summary = sync.run(cfg, LocalTransport())

    assert summary.ok
    r = summary.by_demo["d1"]
    assert (r.kind, r.pushed, r.expected, r.verified, r.deleted, r.errors) == ("dm", True, 4, 4, 4, [])
    assert r.synced
    assert not any(f.exists() for f in files)
    # Everything is on the "Mill"; the demo also stays local.
    for tree, name in [
        ("hdf5/dm/d1", "d1_e0.hdf5"),
        ("hdf5_fullres/dm/d1", "d1_e1.hdf5"),
        ("hdf5/dm/d1", "manifest.jsonl"),
        ("demos/dm", "d1.dem"),
        ("qa", "d1.jsonl"),
    ]:
        assert (remote / tree / name).is_file(), (tree, name)
    assert (Path(cfg.paths.demos) / "dm" / "d1.dem").is_file()
    assert (Path(cfg.paths.hdf5) / "dm" / "d1" / "manifest.jsonl").is_file()


def test_run_uses_one_sha256sum_per_demo(cfg) -> None:
    make_demo(cfg, "d1")
    make_demo(cfg, "d2", kind="smoke", seed=1)
    t = LocalTransport()
    sync.run(cfg, t)
    assert sum("sha256sum -c" in c for c in t.ssh_cmds) == 2


def test_delete_local_hdf5_false_keeps_files(tmp_path, remote) -> None:
    cfg = load_config(
        overrides=[f"paths.root={tmp_path / 'data'}", f"sync.remote_root={remote}", "sync.delete_local_hdf5=false"]
    )
    files = [f for f in make_demo(cfg, "d1") if f]
    r = sync.run(cfg, LocalTransport()).by_demo["d1"]
    assert r.synced and r.deleted == 0
    assert all(f.exists() for f in files)


def test_remote_root_user_is_expanded_by_the_remote_shell(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("USER", "alice")
    cfg = load_config(overrides=[f"paths.root={tmp_path / 'data'}", f"sync.remote_root={tmp_path}/scratch/$USER/dcs2"])
    make_demo(cfg, "d1")
    assert sync.run(cfg, LocalTransport()).by_demo["d1"].synced
    assert (tmp_path / "scratch" / "alice" / "dcs2" / "demos" / "dm" / "d1.dem").is_file()


@pytest.mark.parametrize("bad", ["/x; rm -rf /", "/x/$(whoami)", "/x/`id`", "/x y", "/x/$HOME"])
def test_remote_root_rejects_shell_metacharacters(bad, tmp_path) -> None:
    cfg = load_config(overrides=[f"paths.root={tmp_path}", f"sync.remote_root={bad}"])
    t = LocalTransport()
    with pytest.raises(SyncError, match="remote_root"):
        sync._remote_root(t, cfg)
    assert t.ssh_cmds == []  # rejected before anything reaches the remote shell


def test_remote_root_command_has_no_embedded_quotes(cfg) -> None:
    t = LocalTransport()
    sync._connect(t, cfg)
    assert [c for c in t.ssh_cmds if c.startswith("printf")] == [f"printf %s {cfg.sync.remote_root}"]


# --- partial failure and delete safety --------------------------------------------------------------------------


def test_one_corrupted_remote_file_blocks_only_its_demo(cfg, remote) -> None:
    make_demo(cfg, "bad")
    good_files = make_demo(cfg, "good", seed=1)
    sync.push(cfg, transport=LocalTransport())
    victim = Path(cfg.paths.hdf5) / "dm" / "bad" / "bad_e0.hdf5"
    corrupt_keeping_stat(remote / "hdf5/dm/bad/bad_e0.hdf5", victim)

    summary = sync.run(cfg, LocalTransport())

    assert not summary.ok
    bad, good = summary.by_demo["bad"], summary.by_demo["good"]
    assert good.synced and good.deleted == 4
    assert not any(f.exists() for f in good_files)
    assert not bad.synced and (bad.verified, bad.expected) == (3, 4)
    # Only the three verified files of the bad demo are gone; the corrupted one's local copy is kept.
    assert bad.deleted == 3 and victim.exists()
    assert any("bad_e0.hdf5" in e and "differs" in e for e in bad.errors)


def test_resume_after_partial_failure_finishes_the_job(cfg, remote) -> None:
    make_demo(cfg, "d1")
    sync.push(cfg, transport=LocalTransport())
    victim = Path(cfg.paths.hdf5) / "dm" / "d1" / "d1_e0.hdf5"
    corrupt_keeping_stat(remote / "hdf5/dm/d1/d1_e0.hdf5", victim)
    assert not sync.run(cfg, LocalTransport()).ok and victim.exists()

    (remote / "hdf5/dm/d1/d1_e0.hdf5").unlink()  # operator removes the bad copy
    r = sync.run(cfg, LocalTransport()).by_demo["d1"]

    assert r.synced and r.errors == [] and r.deleted == 1
    assert not victim.exists()


def test_missing_remote_file_is_never_deleted_locally(cfg, remote) -> None:
    files = make_demo(cfg, "d1")
    sync.push(cfg, transport=LocalTransport())
    (remote / "hdf5/dm/d1/d1_e0.hdf5").unlink()

    r = sync.run(cfg, NoPushTransport()).by_demo["d1"]

    assert not r.synced and r.verified == 3 and r.deleted == 3
    assert files[0].exists()
    assert sum(f.exists() for f in files) == 1
    assert any("d1_e0.hdf5" in e and "missing" in e for e in r.errors)


def test_checksum_mismatch_is_never_deleted_locally(cfg, remote) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    sync.push(cfg, transport=LocalTransport())
    (remote / "hdf5_fullres/dm/d1/d1_e0.hdf5").write_bytes(b"truncated")

    r = sync.run(cfg, NoPushTransport()).by_demo["d1"]

    assert (r.verified, r.deleted) == (1, 1)
    assert not files[0].exists() and files[1].exists()


def test_verification_command_failure_deletes_nothing(cfg) -> None:
    files = make_demo(cfg, "d1")
    summary = sync.run(cfg, BrokenCheckTransport())

    r = summary.by_demo["d1"]
    assert not summary.ok and not r.synced and r.deleted == 0
    assert all(f.exists() for f in files)
    assert any("nothing deleted" in e and "connection lost" in e for e in r.errors)


def test_local_file_changed_since_manifest_is_kept(cfg) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    sync.push(cfg, transport=LocalTransport())
    files[0].write_bytes(files[0].read_bytes() + b"x")  # repacked after the manifest line

    r = sync.run(cfg, NoPushTransport()).by_demo["d1"]

    assert files[0].exists() and not files[1].exists()
    assert any("local file differs" in e for e in r.errors)


def test_failed_rsync_deletes_nothing_unverified(cfg) -> None:
    class FailingRsync(LocalTransport):
        def rsync_argv(self, src, dest, files_from_stdin=False):
            return ["bash", "-c", "echo disk full >&2; exit 11"]

    files = make_demo(cfg, "d1")
    summary = sync.run(cfg, FailingRsync())

    assert not summary.ok and len(summary.push_errors) == 4
    assert "disk full" in summary.push_errors[0]
    r = summary.by_demo["d1"]
    assert not r.pushed and not r.synced and r.deleted == 0 and r.verified == 0
    assert all(f.exists() for f in files)


def test_bad_manifest_lines_are_reported_and_valid_ones_still_sync(cfg) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    manifest = Path(cfg.paths.hdf5) / "dm" / "d1" / "manifest.jsonl"
    with open(manifest, "a") as f:
        f.write("{not json\n")
        f.write(json.dumps({"file": "../../etc/passwd.hdf5", "sha256": "0" * 64}) + "\n")
        f.write(json.dumps({"file": "x.hdf5", "sha256": "XYZ"}) + "\n")

    r = sync.run(cfg, LocalTransport()).by_demo["d1"]

    assert r.expected == 2 and r.verified == 2 and r.deleted == 2
    assert len(r.errors) == 3 and not r.synced
    assert not any(f.exists() for f in files)


def test_last_manifest_line_wins_after_a_repack(cfg) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    manifest = Path(cfg.paths.hdf5) / "dm" / "d1" / "manifest.jsonl"
    append_manifest(manifest, {"file": "d1_e0.hdf5", "frames": 3, "sha256": "0" * 64})  # stale, then the real one:
    append_manifest(manifest, json.loads(manifest.read_text().splitlines()[0]))

    r = sync.run(cfg, LocalTransport()).by_demo["d1"]

    assert r.synced and r.expected == 2
    assert not files[0].exists()


def test_delete_verified_refuses_non_hdf5(tmp_path) -> None:
    keep, dem = tmp_path / "a.hdf5", tmp_path / "a.dem"
    keep.write_bytes(b"x")
    dem.write_bytes(b"x")
    with pytest.raises(ValueError, match="non-HDF5"):
        sync.delete_verified([keep, dem])
    assert keep.exists() and dem.exists()
    assert sync.delete_verified([keep, tmp_path / "gone.hdf5"]) == 1
    assert dem.exists()


# --- idempotence, scoping -----------------------------------------------------------------------------------------


def test_rerun_is_idempotent(cfg, remote) -> None:
    make_demo(cfg, "d1")
    first = sync.run(cfg, LocalTransport())
    snapshot = {p.relative_to(remote): p.stat().st_mtime_ns for p in remote.rglob("*") if p.is_file()}

    second = sync.run(cfg, LocalTransport())

    assert first.by_demo["d1"].deleted == 4 and second.by_demo["d1"].deleted == 0
    assert second.ok and second.by_demo["d1"].synced and second.by_demo["d1"].verified == 4
    assert snapshot == {p.relative_to(remote): p.stat().st_mtime_ns for p in remote.rglob("*") if p.is_file()}


def test_push_one_demo_leaves_others_alone(cfg, remote) -> None:
    make_demo(cfg, "d1")
    make_demo(cfg, "d2", seed=1)
    sync.push(cfg, "d1", LocalTransport())
    names = {p.name for p in remote.rglob("*") if p.is_file()}
    assert {"d1_e0.hdf5", "manifest.jsonl", "d1.dem", "d1.jsonl"} <= names
    assert not {n for n in names if n.startswith("d2")}


def test_verify_remote_returns_only_verified_existing_files(cfg, remote) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    sync.push(cfg, transport=LocalTransport())
    (remote / "hdf5/dm/d1/d1_e0.hdf5").unlink()
    assert sync.verify_remote(cfg, "d1", LocalTransport()) == [files[1]]
    with pytest.raises(SyncError, match="no manifest"):
        sync.verify_remote(cfg, "nope", LocalTransport())
    with pytest.raises(ValueError):
        sync.verify_remote(cfg, "../x", LocalTransport())


def test_demoresult_synced_needs_everything() -> None:
    assert DemoResult("d", "dm", pushed=True, expected=2, verified=2).synced
    assert not DemoResult("d", "dm", pushed=False, expected=2, verified=2).synced
    assert not DemoResult("d", "dm", pushed=True, expected=2, verified=1).synced
    assert not DemoResult("d", "dm", pushed=True, expected=0, verified=0).synced
    assert not DemoResult("d", "dm", pushed=True, expected=1, verified=1, errors=["x"]).synced


# --- preflight ----------------------------------------------------------------------------------------------------


class NoKeyTransport(LocalTransport):
    def ssh_argv(self, remote_cmd: str) -> list[str]:
        self.ssh_cmds.append(remote_cmd)
        return ["bash", "-c", "echo 'mill: Permission denied (publickey,password).' >&2; exit 255"]


def test_preflight_failure_is_clear_and_transfers_nothing(cfg, remote) -> None:
    files = make_demo(cfg, "d1")
    with pytest.raises(SyncError) as e:
        sync.run(cfg, NoKeyTransport())
    msg = str(e.value)
    assert "#3" in msg and "BatchMode=yes mill true" in msg and "Permission denied" in msg
    assert not remote.exists() and all(f.exists() for f in files)
    with pytest.raises(SyncError, match="#3"):
        sync.push(cfg, transport=NoKeyTransport())


def test_cli_returns_1_with_message_on_preflight_failure(cfg, monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(sync, "Transport", lambda host: NoKeyTransport())
    config = tmp_path / "c.yaml"
    config.write_text(Path(__file__).resolve().parent.parent.joinpath("config/cs2wm.yaml").read_text())
    with pytest.raises(SystemExit) as e:
        main(["--config", str(config), "sync"])
    assert e.value.code == 1
    assert "#3" in capsys.readouterr().err


def test_missing_ssh_binary_is_a_syncerror() -> None:
    def boom(*a, **k):
        raise FileNotFoundError("ssh")

    with pytest.raises(SyncError, match="cannot run"):
        Transport("mill", windows=False, runner=boom).run(["ssh"])


# --- argv construction --------------------------------------------------------------------------------------------


def test_linux_argv() -> None:
    t = Transport("mill", windows=False)
    assert t.ssh_argv("cd '/s' && sha256sum -c -") == [
        "ssh",
        *SSH_OPTS,
        "mill",
        "cd '/s' && sha256sum -c -",
    ]
    assert t.rsync_argv("/data/hdf5/", t.remote_spec("/s/hdf5/")) == [
        "rsync",
        "-a",
        "--partial",
        "--timeout=300",
        "--exclude=*.tmp",
        "-e",
        "ssh " + " ".join(SSH_OPTS),
        "/data/hdf5/",
        "mill:/s/hdf5/",
    ]
    assert t.rsync_argv("/a/", "mill:/b/", files_from_stdin=True)[-4:-2] == ["--files-from=-", "--from0"]
    assert t.local_path(Path("/data/x")) == "/data/x"


def test_windows_argv_goes_through_wsl_with_wslpath() -> None:
    calls: list[list[str]] = []

    def fake(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="/mnt/c/dcs2/data/hdf5\n", stderr="")

    t = Transport("mill", windows=True, runner=fake)
    assert t.ssh_argv("true") == ["wsl.exe", "-e", "ssh", *SSH_OPTS, "mill", "true"]
    local = t.local_path(Path("C:/dcs2/data/hdf5"))
    assert local == "/mnt/c/dcs2/data/hdf5"
    assert calls[0][:5] == ["wsl.exe", "-e", "wslpath", "-a", "-u"]
    argv = t.rsync_argv(local + "/", t.remote_spec("/s/hdf5/"))
    assert argv[:5] == ["wsl.exe", "-e", "rsync", "-a", "--partial"]
    assert argv[-2:] == ["/mnt/c/dcs2/data/hdf5/", "mill:/s/hdf5/"]
    assert argv[argv.index("ssh " + " ".join(SSH_OPTS)) - 1] == "-e"


def test_wslpath_failure_is_reported() -> None:
    def fake(argv, **kw):
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="wslpath: bad path")

    with pytest.raises(SyncError, match="wslpath"):
        Transport("mill", windows=True, runner=fake).local_path(Path("Z:/nope"))


def test_default_platform_detection(monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "nt")
    assert Transport("mill").windows
    monkeypatch.setattr(os, "name", "posix")
    assert not Transport("mill").windows


# --- stdin bytes, timeouts, per-demo state, names -----------------------------------------------------------------


def test_stdin_is_sent_as_exact_bytes_without_carriage_returns(cfg) -> None:
    """Text mode would turn \\n into \\r\\n on Windows, so the listings must reach the runner as bytes."""
    make_demo(cfg, "d1", episodes=1)
    sent: dict[str, bytes] = {}

    def spy(argv, input=None, **kw):
        if "--files-from=-" in argv:
            sent["files"] = input
        elif "sha256sum" in argv[-1]:
            sent["check"] = input
        return real_run(argv, input=input, **kw)

    t = LocalTransport()
    t._runner = spy
    sync.push(cfg, "d1", t)
    sync.verify_remote(cfg, "d1", t)

    assert all(isinstance(v, bytes) and b"\r" not in v for v in sent.values())
    # rsync --files-from=- --from0 (last call is qa/): NUL-terminated names relative to the tree, nothing else.
    assert sent["files"] == b"d1.jsonl\0"
    # sha256sum -c -: "<hex>  <rel>\n" per manifest entry, LF only.
    lines = sent["check"].split(b"\n")
    assert lines[-1] == b"" and len(lines) == 3
    for line, rel in zip(lines, [b"hdf5/dm/d1/d1_e0.hdf5", b"hdf5_fullres/dm/d1/d1_e0.hdf5"]):
        digest, name = line.split(b"  ")
        assert len(digest) == 64 and name == rel


def test_timeout_becomes_syncerror_and_nothing_is_deleted(cfg) -> None:
    files = make_demo(cfg, "d1")

    def hang_on_verify(argv, **kw):
        if "sha256sum" in argv[-1]:
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        return real_run(argv, **kw)

    t = LocalTransport()
    t._runner = hang_on_verify
    summary = sync.run(cfg, t)

    r = summary.by_demo["d1"]
    assert not summary.ok and not r.synced and r.deleted == 0
    assert all(f.exists() for f in files)
    assert any("timed out" in e and "nothing deleted" in e for e in r.errors)


def test_rsync_timeout_is_a_push_error_not_a_crash(cfg) -> None:
    files = make_demo(cfg, "d1")

    def hang_on_rsync(argv, **kw):
        if argv[0] == "rsync":
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        return real_run(argv, **kw)

    t = LocalTransport()
    t._runner = hang_on_rsync
    summary = sync.run(cfg, t)
    assert len(summary.push_errors) == 4 and "timed out" in summary.push_errors[0]
    assert all(f.exists() for f in files)


def test_timeouts_come_from_config(cfg) -> None:
    seen: dict[str, float] = {}

    def spy(argv, **kw):
        seen[argv[0] if argv[0] == "rsync" else argv[-1].split()[0]] = kw["timeout"]
        return real_run(argv, **kw)

    make_demo(cfg, "d1", episodes=1)
    t = LocalTransport()
    t._runner = spy
    sync.run(cfg, t)
    assert seen["rsync"] == cfg.sync.rsync_timeout_s
    assert seen["true"] == seen["mkdir"] == seen["printf"] == cfg.sync.ssh_timeout_s
    assert seen["cd"] == cfg.sync.verify_timeout_s


def test_qa_push_failure_does_not_unsync_verified_hdf5(cfg) -> None:
    class QaFails(LocalTransport):
        def rsync_argv(self, src, dest, files_from_stdin=False):
            if dest.rstrip("/").endswith("/qa"):
                return ["bash", "-c", "echo qa down >&2; exit 23"]
            return super().rsync_argv(src, dest, files_from_stdin)

    files = make_demo(cfg, "d1")
    summary = sync.run(cfg, QaFails())

    r = summary.by_demo["d1"]
    assert not summary.ok and len(summary.push_errors) == 1 and "qa/" in summary.push_errors[0]
    # The HDF5 is verified and freed; the demo is not "synced" until its QA reaches the Mill.
    assert (r.verified, r.deleted) == (4, 4) and not any(f.exists() for f in files)
    assert not r.pushed and not r.synced and r.errors == []

    again = sync.run(cfg, LocalTransport())  # QA recovers on the next run
    assert again.ok and again.by_demo["d1"].synced


def test_hdf5_push_failure_is_covered_by_verification_not_by_pushed(cfg) -> None:
    class Hdf5Fails(LocalTransport):
        def rsync_argv(self, src, dest, files_from_stdin=False):
            if dest.rstrip("/").endswith("/hdf5"):
                return ["bash", "-c", "exit 23"]
            return super().rsync_argv(src, dest, files_from_stdin)

    files = make_demo(cfg, "d1", episodes=1)
    r = sync.run(cfg, Hdf5Fails()).by_demo["d1"]
    # The low-res file never arrived, so it is unverified and kept; the full-res one is freed.
    assert r.pushed and not r.synced and r.verified == 1 and r.deleted == 1
    assert files[0].exists() and not files[1].exists()


def test_directories_with_unusable_names_are_skipped_and_reported(cfg) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    bad = Path(cfg.paths.hdf5) / "dm" / "bad name"
    bad.mkdir()
    (bad / "manifest.jsonl").write_text('{"file": "x.hdf5", "sha256": "' + "0" * 64 + '"}\n')
    victim = bad / "x.hdf5"
    victim.write_bytes(b"x")

    summary = sync.run(cfg, LocalTransport())

    assert [d.demo_id for d in summary.demos] == ["d1"] and summary.by_demo["d1"].synced
    assert not summary.ok and len(summary.skipped) == 1 and "bad name" in summary.skipped[0]
    assert victim.exists() and not files[0].exists()


@pytest.mark.parametrize("name", ["../x.hdf5", "a/b.hdf5", "x\n.hdf5", "/abs.hdf5", "fullres/../x.hdf5", "x.dem"])
def test_unsafe_manifest_file_names_are_errors_and_never_touched(cfg, tmp_path, name) -> None:
    files = make_demo(cfg, "d1", episodes=1)
    manifest = Path(cfg.paths.hdf5) / "dm" / "d1" / "manifest.jsonl"
    outside = tmp_path / "data" / "hdf5" / "dm" / "x.hdf5"
    outside.write_bytes(b"precious")
    append_manifest(manifest, {"file": name, "sha256": "0" * 64})

    r = sync.run(cfg, LocalTransport()).by_demo["d1"]

    assert r.expected == 2 and len(r.errors) == 1 and not r.synced
    assert "bad file" in r.errors[0]
    assert outside.read_bytes() == b"precious"
    assert not any(f.exists() for f in files)  # the two honest entries still sync
