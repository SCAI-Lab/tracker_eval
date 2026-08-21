"""Single public entry point for the standard and pseudo-detection protocols."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import trackeval


def _trackeval_root() -> Path:
    return Path(trackeval.__file__).resolve().parent.parent


def _run(module: str, arguments: Sequence[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *map(str, arguments)],
        check=True,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker-eval-protocol",
        description="Compute tracker capability tables or pseudo-detection robustness results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    capability = commands.add_parser(
        "capabilities",
        help="Build shared caches, tracker profiles and manuscript result tables.",
    )
    capability.add_argument("--trackers-base-dir", required=True, type=Path)
    capability.add_argument("--gt-folder", required=True, type=Path)
    capability.add_argument("--local-gt-folder", required=True, type=Path)
    capability.add_argument("--detections-dir", required=True, type=Path)
    capability.add_argument("--output-dir", required=True, type=Path)
    capability.add_argument("--trackers", required=True)
    capability.add_argument("--reference-tracker", required=True)
    capability.add_argument("--split", default="test")
    capability.add_argument(
        "--tracker-subfolder",
        default=None,
        help="Default: <split>/data, matching tracker-eval output layout.",
    )
    capability.add_argument("--fps", type=float, default=15.0)
    capability.add_argument("--workers", type=int, default=1)
    capability.add_argument("--recompute-trackers", default=None)
    capability.add_argument("--force", action="store_true")

    pseudo = commands.add_parser(
        "pseudo",
        help="Evaluate all clean/dropout/instability/combined conditions.",
    )
    pseudo.add_argument("--trackers-dir", required=True, type=Path)
    pseudo.add_argument("--gt-folder", required=True, type=Path)
    pseudo.add_argument("--output-dir", required=True, type=Path)
    pseudo.add_argument(
        "--pseudo-spec",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cli" / "pseudo_det_spec.yaml",
    )
    pseudo.add_argument("--seqmap-gt-folder", type=Path, default=None)
    pseudo.add_argument("--split", default="test")
    pseudo.add_argument(
        "--tracker-subfolder",
        default=None,
        help="Default: <split>/data, matching tracker-eval output layout.",
    )
    pseudo.add_argument("--workers", type=int, default=12)
    pseudo.add_argument("--trackers", nargs="*", default=[])
    pseudo.add_argument("--variants", nargs="*", default=[])
    pseudo.add_argument("--recompute-trackers", nargs="*", default=[])
    pseudo.add_argument("--recompute-variants", nargs="*", default=[])
    pseudo.add_argument("--dry-run", action="store_true")
    return parser


def _capability_command(args: argparse.Namespace) -> None:
    cache_dir = args.output_dir / "_capability_cache"
    common = [
        "--trackeval-root", str(_trackeval_root()),
        "--trackers-base-dir", str(args.trackers_base_dir),
        "--gt-folder", str(args.gt_folder),
        "--trackers", args.trackers,
        "--reference-tracker", args.reference_tracker,
        "--split-to-eval", args.split,
        "--tracker-sub-folder", args.tracker_subfolder or f"{args.split}/data",
        "--fps", str(args.fps),
        "--num-workers", str(args.workers),
    ]
    profile_args = [
        *common,
        "--local-gt-folder", str(args.local_gt_folder),
        "--detections-dir", str(args.detections_dir),
        "--output-dir", str(cache_dir),
    ]
    result_args = [
        *common,
        "--capability-cache-dir", str(cache_dir),
        "--output-dir", str(args.output_dir),
    ]
    if args.recompute_trackers:
        profile_args += ["--recompute-trackers", args.recompute_trackers]
        result_args += ["--recompute-trackers", args.recompute_trackers]
    if args.force:
        profile_args.append("--force-common")
        result_args.append("--recompute-all")
    _run("tracker_eval.protocol.profiles", profile_args)
    _run("tracker_eval.protocol.results", result_args)


def _pseudo_command(args: argparse.Namespace) -> None:
    command = [
        "--trackeval-root", str(_trackeval_root()),
        "--trackers-dir", str(args.trackers_dir),
        "--gt-folder", str(args.gt_folder),
        "--output-dir", str(args.output_dir),
        "--pseudo-spec", str(args.pseudo_spec),
        "--split", args.split,
        "--tracker-subfolder", args.tracker_subfolder or f"{args.split}/data",
        "--num-workers", str(args.workers),
    ]
    for name in ("trackers", "variants", "recompute_trackers", "recompute_variants"):
        values = getattr(args, name)
        if values:
            command.extend(["--" + name.replace("_", "-"), *values])
    if args.seqmap_gt_folder:
        command += ["--seqmap-gt-folder", str(args.seqmap_gt_folder)]
    if args.dry_run:
        command.append("--dry-run")
    _run("tracker_eval.protocol.pseudo_results", command)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.command == "capabilities":
        _capability_command(args)
    else:
        _pseudo_command(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
