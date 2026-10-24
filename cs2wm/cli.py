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
    p.add_argument("--player", default=None, help="player name for --policy fixed_player")

    p = sub.add_parser("render", help="render a demo with HLAE (#15)")
    _add_demo_args(p)

    p = sub.add_parser("extract", help="derive 51-dim action labels and QA (#11)")
    _add_demo_args(p)

    p = sub.add_parser("pack", help="write HDF5 episodes (#12)")
    _add_demo_args(p)

    sub.add_parser("sync", help="push to the Mill, verify, free local disk (#16)")

    p = sub.add_parser("supervise", help="run the pipeline unattended (#20)")
    p.add_argument("--until", default="07:30", help="deadline, HH:MM local time")
    p.add_argument("--target-hours", type=float, default=20.0)
    p.add_argument("--profile", choices=["dm", "smoke"], default="dm")

    sub.add_parser("status", help="print the pipeline state table (#20)")

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


def _cfg(args: argparse.Namespace) -> DictConfig:
    from cs2wm.config import load_config

    return load_config(args.config)


def _dispatch(args: argparse.Namespace) -> None:
    """Call the stub behind the parsed subcommand."""
    cmd = args.cmd
    if cmd == "generate":
        from cs2wm.pipeline import generate

        generate.run(_cfg(args), args.profile, args.hours, args.chunk_min)
    elif cmd == "segment":
        from cs2wm.pipeline import segment

        # TODO(#10): resolve --all-pending and demo ids through the supervisor state.
        segment.run(Path(args.demo or ""), _cfg(args), args.policy, args.player)
    elif cmd == "render":
        from cs2wm.pipeline import render

        render.run(Path(args.demo or ""), _cfg(args))
    elif cmd == "extract":
        from cs2wm.pipeline import extract_actions

        extract_actions.run(Path(args.demo or ""), _cfg(args))
    elif cmd == "pack":
        from cs2wm.pipeline import pack

        pack.run(args.demo or "", _cfg(args))
    elif cmd == "sync":
        from cs2wm.pipeline import sync

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        try:
            ok = sync.run(_cfg(args)).ok
        except sync.SyncError as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        if not ok:
            print("error: sync finished with errors, see the log above", file=sys.stderr)
            raise SystemExit(1)
    elif cmd == "supervise":
        from cs2wm.pipeline.supervisor import PipelineState, Supervisor

        cfg = _cfg(args)
        Supervisor(cfg, PipelineState(Path(cfg.paths.state) / "pipeline.json")).run_until(
            args.until, args.target_hours, args.profile
        )
    elif cmd == "status":
        from cs2wm.pipeline.supervisor import PipelineState, Supervisor

        cfg = _cfg(args)
        print(Supervisor(cfg, PipelineState.load(Path(cfg.paths.state) / "pipeline.json")).status_table())
    elif cmd == "qa-summary":
        # TODO(#19): aggregate qa/*.jsonl by kind and age (--kind, --since).
        raise NotImplementedError("TODO(#19): qa-summary")
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
        Process exit code: 0 on success, 2 when the command is not implemented yet.
    """
    args = build_parser().parse_args(argv)
    try:
        _dispatch(args)
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
