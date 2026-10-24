"""The ``cs2wm`` command line. Each subcommand dispatches to a stub module; see the linked issue for its spec."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig


def _add_demo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("demo", nargs="?", help="demo id or path to a .dem file")
    p.add_argument("--all-pending", action="store_true", help="process every demo that has not passed this stage")
    p.add_argument(
        "--retry-failed", action="store_true", help="with --all-pending: also retry demos that failed in this stage"
    )


def _parse_models(items: Sequence[str] | None) -> dict[str, Path]:
    """Parse ``name=path`` pairs."""
    out: dict[str, Path] = {}
    for item in items or ():
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected name=path, got {item!r}")
        out[name] = Path(path)
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand."""
    parser = argparse.ArgumentParser(prog="cs2wm", description="CS2 world-model data pipeline, eval and play.")
    parser.add_argument("--config", default=None, help="path to cs2wm.yaml (default: config/cs2wm.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")

    p = sub.add_parser("generate", help="generate demos with offline CS2 (#13)")
    p.add_argument("--profile", choices=["dm", "smoke", "human"], default="dm")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--chunk-min", type=float, default=None, help="demo chunk length (default: generate.chunk_min)")

    p = sub.add_parser("segment", help="plan lives, spectate schedule and episodes (#10)")
    _add_demo_args(p)
    p.add_argument("--policy", choices=["longest_remaining", "fixed_player"], default=None)
    p.add_argument("--player", default=None, help="player name (or numeric user_id) for --policy fixed_player")

    p = sub.add_parser("render", help="render a demo with HLAE (#15)")
    _add_demo_args(p)
    p.add_argument(
        "--cfg-only", action="store_true", help="write the render cfg, print the HLAE command, do not launch"
    )
    p.add_argument("--plan", type=Path, default=None, help="render_plan.json to use with --cfg-only (no demo needed)")

    p = sub.add_parser("extract", help="derive 51-dim action labels and QA (#11)")
    _add_demo_args(p)

    p = sub.add_parser("pack", help="write HDF5 episodes (#12)")
    _add_demo_args(p)

    sub.add_parser("sync", help="push to the Mill, verify, free local disk (#16)")

    p = sub.add_parser("supervise", help="run the pipeline unattended (#20)")
    p.add_argument(
        "--until",
        default="07:30",
        help="deadline, HH:MM local time (the next such time, so 07:30 at 23:00 is tomorrow)",
    )
    p.add_argument("--target-hours", type=float, default=20.0, help="stop once this many QA-passing hours are synced")
    p.add_argument("--profile", choices=["dm", "smoke"], default="dm")

    sub.add_parser("status", help="print the pipeline state table (#20)")

    p = sub.add_parser("cleanup", help="free working-set disk (#20); do not run while `supervise` is running")
    p.add_argument("--failed", action="store_true", help="delete frames/raw/labels (and pack .staging) of failed demos")
    p.add_argument("--dry-run", action="store_true", help="only list what would be deleted")
    p.add_argument(
        "--force",
        action="store_true",
        help="also clean failed demos that would still be retried (e.g. failures from manual runs)",
    )

    p = sub.add_parser("qa-summary", help="summarise extract QA (#19)")
    p.add_argument("--kind", choices=["dm", "human", "smoke"], default=None)
    p.add_argument("--since", default=None, help="only demos newer than this, e.g. 24h")

    p = sub.add_parser("inspect-demo", help="print a demo's header, players and event counts (#14, #11)")
    p.add_argument("demo", help="path to a .dem file")

    p = sub.add_parser("label-check", help="hand-check label accuracy (#18)")
    lc = p.add_subparsers(dest="label_cmd", required=True, metavar="<action>")
    q = lc.add_parser("render", help="render the overlay clip and review sheet")
    q.add_argument("--episode", required=True, type=Path, help="packed episode HDF5")
    q.add_argument("--full-res-frames", required=True, type=Path, help="directory of full-res rendered frames")
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--frames", type=int, default=1920)
    q.add_argument("--out", required=True, type=Path, help="output clip.mp4")
    q = lc.add_parser("score", help="score a filled-in review sheet")
    q.add_argument("sheet", type=Path, help="clip_labels.csv")
    q.add_argument("--out", required=True, type=Path, help="e.g. results/label_accuracy.json")

    p = sub.add_parser("play", help="play the world model interactively (#28)")
    p.add_argument("--ckpt", required=True, type=Path, help="v2 agent checkpoint")
    p.add_argument("--upsampler-ckpt", type=Path, default=None, help="v1 checkpoint providing the upsampler")
    p.add_argument("--no-upsampler", action="store_true")
    p.add_argument("--denoising-steps", type=int, choices=[1, 3], default=1)
    p.add_argument("--context", type=Path, default=None, help="held-out test HDF5 for the initial context")
    p.add_argument("--record", type=Path, default=None, help="write the session to this MP4 (#35)")
    p.add_argument("--benchmark", type=int, default=None, metavar="N", help="benchmark FPS over N frames and exit")

    p = sub.add_parser("eval", help="evaluations (#27, #32, #33, #34)")
    ev = p.add_subparsers(dest="eval_cmd", required=True, metavar="<experiment>")
    q = ev.add_parser("upsampler", help="v1 upsampler compatibility check (#27)")
    q.add_argument("--ckpt", required=True, type=Path)
    q.add_argument("--fullres", required=True, type=Path, help="v2 hdf5_fullres directory")
    q.add_argument("--n", type=int, default=256)
    q = ev.add_parser("h1", help="controllability (#32)")
    q.add_argument("--models", nargs="+", required=True, metavar="NAME=CKPT", help="e.g. v2=... v1=... shuffled=...")
    q.add_argument("--contexts", type=int, default=50)
    q.add_argument("--seed", type=int, default=0)
    q = ev.add_parser("h2", help="volumetric smoke (#33)")
    q.add_argument("--models", nargs="+", required=True, metavar="NAME=CKPT", help="e.g. v2=... v1=...")
    q.add_argument("--seed", type=int, default=0)
    q = ev.add_parser("carried", help="carried-over v1 evals: loss curves, rollouts, FID (#34)")
    q.add_argument("--models", nargs="+", required=True, metavar="NAME=CKPT", help="e.g. v2=... v1=...")
    q.add_argument("--v2-metrics", type=Path, default=None, help="v2 metrics.jsonl")
    q.add_argument("--data", type=Path, default=None, help="test split directory")
    q.add_argument("--out", type=Path, default=Path("results/carried"))

    p = sub.add_parser("report-assets", help="regenerate report tables and figures (#36)")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("report"))
    return parser


def _resolve_demo(arg: str, cfg: DictConfig) -> Path:
    """Return the ``.dem`` file for a path or a demo id (searched as ``paths.demos/<kind>/<id>.dem``)."""
    path = Path(arg)
    if path.is_file():
        return path
    matches = sorted(Path(cfg.paths.demos).glob(f"*/{arg}.dem"))
    if not matches:
        raise SystemExit(f"demo not found: {arg} (not a file, and no {cfg.paths.demos}/*/{arg}.dem)")
    return matches[0]


def _cfg(args: argparse.Namespace) -> DictConfig:
    from cs2wm.config import load_config

    return load_config(args.config)


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _setup_logging(cfg: DictConfig) -> None:
    """Log the ``cs2wm`` loggers to stderr and to ``paths.logs/cs2wm.log``; safe to call again."""
    log = logging.getLogger("cs2wm")
    for handler in [h for h in log.handlers if getattr(h, "_cs2wm_cli", False)]:
        log.removeHandler(handler)
        handler.close()
    Path(cfg.paths.logs).mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(Path(cfg.paths.logs) / "cs2wm.log", encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._cs2wm_cli = True  # type: ignore[attr-defined]
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def _supervisor(cfg: DictConfig):
    """A supervisor over ``state/pipeline.json``."""
    from cs2wm.pipeline.supervisor import PipelineState, StateError, Supervisor

    try:
        return Supervisor(cfg, PipelineState.load(Path(cfg.paths.state) / "pipeline.json"))
    except StateError as exc:
        raise SystemExit(f"error: {exc}") from exc


def _refuse_if_past(sup, stage: str, demo_id: str) -> None:
    """Refuse an explicit re-run of a stage the demo is already past when the stage's inputs are gone.

    Running it would fail and overwrite a good state (``synced``, ``packed``, ...) with ``failed``.
    """
    from cs2wm.pipeline.supervisor import _OUTPUT_STATE, STAGES

    order = [s.value for s in dict.fromkeys([*_OUTPUT_STATE.values()])]
    rec = sup.state.demos.get(demo_id)
    if rec is None or rec["state"] not in order:
        return
    if order.index(rec["state"]) >= order.index(_OUTPUT_STATE[stage].value) and not sup.inputs_present(stage, demo_id):
        raise SystemExit(
            f"{stage}: {demo_id} is already {rec['state']} and the inputs of {stage} are gone (of {STAGES}); "
            "refusing to re-run, the state is untouched"
        )


def _run_stage_cmd(stage: str, args: argparse.Namespace, **options: str | None) -> int:
    """Run a per-demo stage on one demo or on ``--all-pending``, recording every demo in the pipeline state.

    Returns:
        1 if any demo failed (the others still ran), else 0.
    """
    cfg = _cfg(args)
    _setup_logging(cfg)
    sup = _supervisor(cfg)
    if args.all_pending:
        if args.demo:
            raise SystemExit(f"{stage}: give a demo or --all-pending, not both")
        if sup.state.discover(cfg):
            sup.state.save()
        ids = sup.state.pending_for(stage, args.retry_failed)
    elif args.demo:
        path = _resolve_demo(args.demo, cfg)
        ids = [path.stem]
        sup.explicit_paths[path.stem] = path
        _refuse_if_past(sup, stage, path.stem)
        if path.stem not in sup.state.demos:
            sup.state.mark(
                path.stem, "pending", **({"kind": path.parent.name} if path.parent.name in cfg.episodes.kinds else {})
            )
    else:
        raise SystemExit(f"{stage}: give a demo id or a path to a .dem file, or --all-pending")
    report = sup.run_stage(stage, ids, **options)
    for demo_id, output in report.outputs.items():
        print(f"{stage} {demo_id}: {output}")
    print(f"{stage}: {len(report.done)} done, {len(report.failed)} failed, {len(ids)} considered")
    for demo_id, reason in report.failed.items():
        print(f"  FAILED {demo_id}: {reason}", file=sys.stderr)
    return 1 if report.failed else 0


def _dispatch(args: argparse.Namespace) -> int:
    """Run the parsed subcommand and return its exit code."""
    cmd = args.cmd
    if cmd == "generate":
        from cs2wm.pipeline import generate

        generate.run(_cfg(args), args.profile, args.hours, args.chunk_min)
    elif cmd == "segment":
        return _run_stage_cmd("segment", args, policy=args.policy, player=args.player)
    elif cmd == "render":
        from cs2wm.pipeline import render

        cfg = _cfg(args)
        if args.cfg_only:
            if args.all_pending:
                raise SystemExit("render: --cfg-only works on one demo or a plan, not --all-pending")
            if args.plan is None and not args.demo:
                raise SystemExit("render --cfg-only: give a demo id or --plan <render_plan.json>")
            demo = (
                _resolve_demo(args.demo, cfg)
                if args.demo and (args.plan is None or Path(args.demo).is_file())
                else None
            )
            plan_path = args.plan or Path(cfg.paths.plans) / (demo or Path(args.demo)).stem / "render_plan.json"
            out = render.write_cfg_only(plan_path, cfg, demo)
            print(f"cfg installed: {out.installed}\ncfg copy:      {out.copy}\nHLAE argv:\n  " + "\n  ".join(out.argv))
            print(f"\nPowerShell:\n{out.powershell}")
            return 0
        return _run_stage_cmd("render", args)
    elif cmd in ("extract", "pack"):
        return _run_stage_cmd(cmd, args)
    elif cmd == "sync":
        from cs2wm.pipeline import sync

        cfg = _cfg(args)
        _setup_logging(cfg)
        try:
            ok = _supervisor(cfg).run_sync().ok
        except sync.SyncError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if not ok:
            print("error: sync finished with errors, see the log above", file=sys.stderr)
            raise SystemExit(1)
    elif cmd == "supervise":
        cfg = _cfg(args)
        _setup_logging(cfg)
        try:
            return _supervisor(cfg).run_until(args.until, args.target_hours, args.profile).exit_code
        except ValueError as exc:
            raise SystemExit(f"supervise: {exc}") from exc
    elif cmd == "cleanup":
        from cs2wm.pipeline.supervisor import cleanup_failed

        if not args.failed:
            raise SystemExit("cleanup: say what to clean; the only choice is --failed")
        cfg = _cfg(args)
        found = cleanup_failed(cfg, _supervisor(cfg).state, args.dry_run, args.force)
        for demo_id, path, size in found:
            print(f"{'would delete' if args.dry_run else 'deleted'} {path} ({size / 1e6:.1f} MB) [{demo_id}]")
        total = sum(size for _, _, size in found)
        print(
            f"{len(found)} dir(s), {total / 1e9:.2f} GB{' (dry run)' if args.dry_run else ''}; demos and QA untouched"
        )
    elif cmd == "status":
        cfg = _cfg(args)
        sup = _supervisor(cfg)
        sup.state.discover(cfg)  # shown as pending, not saved: status never writes
        print(sup.status_table())
    elif cmd == "qa-summary":
        from cs2wm.pipeline import qa_summary

        cfg = _cfg(args)
        try:
            since = qa_summary.parse_since(args.since) if args.since else None
        except ValueError as exc:
            raise SystemExit(f"qa-summary: {exc}") from exc
        print(qa_summary.format_summary(qa_summary.summarize(cfg, args.kind, since), float(cfg.timing.fps)))
    elif cmd == "inspect-demo":
        import json

        from cs2wm import demo

        print(json.dumps(demo.inspect(args.demo), indent=2))
    elif cmd == "label-check":
        from cs2wm.eval import label_accuracy

        if args.label_cmd == "render":
            label_accuracy.render_overlay_clip(
                args.episode, args.full_res_frames, args.start, args.frames, args.out, _cfg(args)
            )
            label_accuracy.make_review_sheet(
                args.episode, args.start, args.frames, args.out.with_name(args.out.stem + "_labels.csv")
            )
        else:
            label_accuracy.score_review_sheet(args.sheet, args.out)
    elif cmd == "play":
        from cs2wm.play import loop

        loop.run(
            _cfg(args),
            args.ckpt,
            args.upsampler_ckpt,
            args.no_upsampler,
            args.denoising_steps,
            args.context,
            args.record,
            args.benchmark,
        )
    elif cmd == "eval":
        _dispatch_eval(args)
    elif cmd == "report-assets":
        from cs2wm.report import latex

        latex.run(args.results, args.out)
    return 0


def _dispatch_eval(args: argparse.Namespace) -> None:
    """Call the stub behind ``cs2wm eval <experiment>``."""
    sub = args.eval_cmd
    if sub == "upsampler":
        from cs2wm.eval import upsampler_check

        upsampler_check.run_upsampler_check(args.ckpt, args.fullres, args.n)
    elif sub == "h1":
        from cs2wm.eval import h1_controllability

        h1_controllability.run_h1(_parse_models(args.models), _cfg(args), Path("results/h1"), args.contexts, args.seed)
    elif sub == "h2":
        from cs2wm.eval import h2_smoke

        h2_smoke.run_h2(_parse_models(args.models), _cfg(args), Path("results/h2"), args.seed)
    elif sub == "carried":
        from cs2wm.eval import fid, loss_curve, rollouts

        models = _parse_models(args.models)
        # TODO(#34): wire v1/v2 checkpoints and data into the three ported evals.
        loss_curve.run_loss_curve(args.v2_metrics or Path(), Path("results/v1"), args.out)
        rollouts.run_rollouts(models["v2"], args.data or Path(), args.out)
        fid.compute_fid(models["v2"], args.data or Path())


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``cs2wm`` command.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 1 if a stage failed for any demo, 2 when the command is not implemented yet,
        3 when `supervise` ended on a guard trip or an alert.
    """
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
