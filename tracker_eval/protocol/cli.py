"""Single public entry point for the standard and pseudo-detection protocols."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import trackeval

from tracker_eval.protocol import defaults


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
    capability.add_argument("--fps", type=float, default=defaults.FPS)
    capability.add_argument(
        "--success-iou-thr",
        type=float,
        default=defaults.SUCCESS_IOU_THRESHOLD,
    )
    capability.add_argument(
        "--matchable-sim-thr",
        type=float,
        default=defaults.MATCHABLE_SIMILARITY_THRESHOLD,
    )
    capability.add_argument(
        "--pre-gap-visible-frames",
        type=int,
        default=defaults.PRE_GAP_VISIBLE_FRAMES,
    )
    capability.add_argument(
        "--max-gap-age-s",
        type=float,
        default=defaults.MAX_GAP_AGE_SECONDS,
        help="Default: 2.0 s, matching the corrected RAL evaluation.",
    )
    capability.add_argument(
        "--recovery-gap-bin-edges-frames",
        default=defaults.RECOVERY_GAP_BIN_EDGES_FRAMES,
        help=(
            "Half-open recovery bins. Default: "
            f"{defaults.RECOVERY_GAP_BIN_EDGES_FRAMES}."
        ),
    )
    capability.add_argument(
        "--continuity-max-detector-gap-frames",
        type=int,
        default=defaults.CONTINUITY_MAX_DETECTOR_GAP_FRAMES,
    )
    capability.add_argument(
        "--continuity-warmup-frames",
        type=int,
        default=defaults.CONTINUITY_WARMUP_FRAMES,
    )
    capability.add_argument(
        "--continuity-chunk-s",
        type=float,
        default=defaults.CONTINUITY_CHUNK_SECONDS,
    )
    capability.add_argument(
        "--continuity-min-nn-frames",
        type=int,
        default=defaults.CONTINUITY_MIN_NN_FRAMES,
    )
    capability.add_argument(
        "--initialization-max-s",
        type=float,
        default=defaults.INITIALIZATION_MAX_SECONDS,
    )
    capability.add_argument(
        "--initialization-validation-s",
        type=float,
        default=defaults.INITIALIZATION_VALIDATION_SECONDS,
    )
    capability.add_argument(
        "--nn-min-m",
        type=float,
        default=defaults.NN_MIN_METERS,
    )
    capability.add_argument(
        "--nn-max-m",
        type=float,
        default=defaults.NN_MAX_METERS,
    )
    capability.add_argument(
        "--nn-bin-width-m",
        type=float,
        default=defaults.NN_BIN_WIDTH_METERS,
    )
    capability.add_argument(
        "--hota-spread-quantiles",
        default=defaults.HOTA_SPREAD_QUANTILES,
    )
    capability.add_argument(
        "--profile-ci-quantiles",
        default=defaults.PROFILE_CI_QUANTILES,
    )
    capability.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=defaults.BOOTSTRAP_REPLICATES,
    )
    capability.add_argument(
        "--runtime-min-frames-per-count",
        type=int,
        default=defaults.RUNTIME_MIN_FRAMES_PER_COUNT,
    )
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
        "--success-iou-thr", str(args.success_iou_thr),
        "--matchable-sim-thr", str(args.matchable_sim_thr),
        "--pre-gap-visible-frames", str(args.pre_gap_visible_frames),
        "--max-gap-age-s", str(args.max_gap_age_s),
        "--recovery-gap-bin-edges-frames",
        args.recovery_gap_bin_edges_frames,
        "--hota-spread-quantiles", args.hota_spread_quantiles,
        "--profile-ci-quantiles", args.profile_ci_quantiles,
        "--bootstrap-replicates", str(args.bootstrap_replicates),
        "--runtime-min-frames-per-count",
        str(args.runtime_min_frames_per_count),
        "--num-workers", str(args.workers),
    ]
    profile_args = [
        *common,
        "--local-gt-folder", str(args.local_gt_folder),
        "--detections-dir", str(args.detections_dir),
        "--output-dir", str(cache_dir),
        "--reuse-assignment-observations-from",
        str(args.output_dir / "cache" / "assignment_observations"),
    ]
    result_args = [
        *common,
        "--capability-cache-dir", str(cache_dir),
        "--output-dir", str(args.output_dir),
        "--reuse-capability-tracker-caches",
        "--continuity-max-detector-gap-frames",
        str(args.continuity_max_detector_gap_frames),
        "--continuity-warmup-frames",
        str(args.continuity_warmup_frames),
        "--continuity-chunk-s", str(args.continuity_chunk_s),
        "--continuity-min-nn-frames",
        str(args.continuity_min_nn_frames),
        "--initialization-max-s", str(args.initialization_max_s),
        "--initialization-validation-s",
        str(args.initialization_validation_s),
        "--nn-min-m", str(args.nn_min_m),
        "--nn-max-m", str(args.nn_max_m),
        "--nn-bin-width-m", str(args.nn_bin_width_m),
    ]
    recompute_trackers = (
        args.trackers if args.force else args.recompute_trackers
    )
    if recompute_trackers:
        profile_args += ["--recompute-trackers", recompute_trackers]
        result_args += ["--recompute-trackers", recompute_trackers]
    if args.force:
        profile_args.append("--force-common")
        result_args.append("--recompute-all")
    print(
        "[tracker-eval-protocol] Stage 1/2: common events, broad profiles, "
        "HOTA events and reusable assignments",
        flush=True,
    )
    _run("tracker_eval.protocol.profiles", profile_args)
    print(
        "[tracker-eval-protocol] Stage 2/2: final tables from cached "
        "assignments",
        flush=True,
    )
    _run("tracker_eval.protocol.results", result_args)
    print(
        "[tracker-eval-protocol] Capability protocol complete: "
        f"{args.output_dir}",
        flush=True,
    )


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
