# tracker_eval/cli/run_all_trackers.py

from __future__ import annotations

import argparse
from typing import List, Optional

from tracker_eval.cli.run_tracker import main as run_one_tracker_main


TRACKER_ORDER = [
    "headroom",
    "elptnet",
    "cbmot",
    # "fastpoly",
     "ab3dmot",
    #  "gnnpmb",
    #  "simpletrack",
]

# Optional baked-in default per-tracker workers (only used if you want it).
# Leave as None to not use baked-in defaults.
# Example:
DEFAULT_WORKERS_PER_TRACKER = {
    "headroom": 8,
    "elptnet": 4,
    "cbmot": 8,
    "fastpoly": 4,
    "ab3dmot": 4,
    "gnnpmb": 6,
    "simpletrack": 8,
}
# DEFAULT_WORKERS_PER_TRACKER = None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker_eval.run_all_trackers",
        description="Run all tracker adapters sequentially on one or more JRDB splits.",
    )

    # Splits
    p.add_argument(
        "--split_root",
        type=str,
        nargs="+",
        required=True,
        help="One or more split roots (e.g. /mnt/nvme/JRDB_track/train /mnt/nvme/JRDB_track/test)",
    )
    p.add_argument(
        "--split_name",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit split names (must match count of split_root if provided).",
    )

    # Shared IO
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--detections_subdir", type=str, default="detections_3D")
    p.add_argument(
        "--labels_subdir",
        type=str,
        default="labels_3d",
        help="GT labels subfolder name under split_root (default: labels_3d). Used by headroom.",
    )

    # Shared runtime/output behavior
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--limit_sequences", type=int, default=None)
    p.add_argument("--include_sequences", type=str, nargs="*", default=None)
    p.add_argument("--exclude_sequences", type=str, nargs="*", default=None)
    p.add_argument("--no_skip_existing", action="store_true")
    p.add_argument("--write_tracks_json", action="store_true")
    p.add_argument("--tracker_subfolder", type=str, default="data")
    p.add_argument("--no_kitti_score", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # Parallel options (forwarded to run_tracker)
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Run sequences in parallel using multiple processes (one tracker instance per sequence). "
             "In parallel mode, timing/per-frame profiling is disabled.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for --parallel, applied to ALL trackers unless overridden. "
             "0 means auto (cpu_count).",
    )
    p.add_argument(
        "--num_workers_per_tracker",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional per-tracker worker counts for --parallel. "
            "If provided with length 1, it is broadcast to all trackers. "
            "If provided with length equal to the number of trackers being run (after --only filtering), "
            "it is used as (headroom, elptnet, cbmot, fastpoly, ab3dmot, gnnpmb) in the run order. "
            "Example: --num_workers_per_tracker 2 8 8 4 8 2"
        ),
    )
    p.add_argument(
        "--parallel_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for --parallel (default: spawn).",
    )

    # Control
    p.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of trackers to run (names: elptnet, cbmot, fastpoly, ab3dmot, gnnpmb, simpletrack, headroom).",
    )

    # Pseudo-detector variants (optional)
    p.add_argument(
        "--variants",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Detection variants to run. Example: --variants clean dropout_L1 combo_A. "
            "If omitted, runs only the base detections_subdir."
        ),
    )
    p.add_argument(
        "--variants_subdir",
        type=str,
        default=None,
        help=(
            "Parent folder that contains per-variant detection folders. "
            "Example: detections_3D_pseudo. "
            "If set and --variants is used, detections_subdir becomes <variants_subdir>/<variant>."
        ),
    )
    p.add_argument(
        "--variants_from_manifest",
        type=str,
        default=None,
        help=(
            "Path to a manifest.json produced by generate_pseudo_detections_from_gt.py. "
            "If set and --variants is omitted, variants are taken from manifest (includes 'clean')."
        ),
    )
    p.add_argument(
        "--exclude_variants",
        type=str,
        nargs="*",
        default=None,
        help="Optional variants to skip (works with --variants or --variants_from_manifest).",
    )

    # Odometry / global coordinate option
    p.add_argument(
        "--global_coords",
        action="store_true",
        help="If set: transform detections and GT labels into global coordinates using odometry before tracking.",
    )
    p.add_argument(
        "--odometry_root",
        type=str,
        default="",
        help="Root containing odometry CSVs at <odometry_root>/<split_name>/odometry/<seq>.csv. "
             "Required if --global_coords is set.",
    )

    return p


def _append_shared(args: argparse.Namespace, argv: List[str]) -> List[str]:
    argv += ["--split_root", *args.split_root]
    if args.split_name is not None and len(args.split_name) > 0:
        argv += ["--split_name", *args.split_name]

    argv += ["--out_root", args.out_root]
    argv += ["--detections_subdir", args.detections_subdir]
    argv += ["--labels_subdir", args.labels_subdir]

    argv += ["--warmup_steps", str(args.warmup_steps)]
    if args.limit_sequences is not None:
        argv += ["--limit_sequences", str(args.limit_sequences)]
    if args.include_sequences is not None and len(args.include_sequences) > 0:
        argv += ["--include_sequences", *args.include_sequences]
    if args.exclude_sequences is not None and len(args.exclude_sequences) > 0:
        argv += ["--exclude_sequences", *args.exclude_sequences]

    if args.no_skip_existing:
        argv += ["--no_skip_existing"]
    if args.write_tracks_json:
        argv += ["--write_tracks_json"]
    if args.tracker_subfolder is not None:
        argv += ["--tracker_subfolder", args.tracker_subfolder]
    if args.no_kitti_score:
        argv += ["--no_kitti_score"]
    if args.quiet:
        argv += ["--quiet"]

    if args.variants is not None and len(args.variants) > 0:
        argv += ["--variants", *args.variants]
    if args.exclude_variants is not None and len(args.exclude_variants) > 0:
        argv += ["--exclude_variants", *args.exclude_variants]
    if args.variants_subdir is not None:
        argv += ["--variants_subdir", args.variants_subdir]
    if args.variants_from_manifest is not None:
        argv += ["--variants_from_manifest", args.variants_from_manifest]

    # Odometry / global coordinate option
    if bool(args.global_coords):
        argv += ["--global_coords"]

        # Only forward odometry_root when global mode is requested
        if args.odometry_root is None or str(args.odometry_root).strip() == "":
            raise ValueError("--odometry_root must be set when --global_coords is used.")
        argv += ["--odometry_root", str(args.odometry_root)]


    return argv


def _resolve_workers_for_trackers(
    trackers: List[str],
    *,
    parallel: bool,
    num_workers: int,
    num_workers_per_tracker: Optional[List[int]],
) -> List[int]:
    """
    Returns a list of workers aligned with `trackers`.
    Priority:
      1) --num_workers_per_tracker (len 1 broadcast OR exact length match)
      2) DEFAULT_WORKERS_PER_TRACKER mapping (if provided)
      3) --num_workers (single value for all)
    If not parallel, returns zeros.
    """
    if not parallel:
        return [0] * len(trackers)

    if num_workers_per_tracker is not None and len(num_workers_per_tracker) > 0:
        xs = [int(x) for x in num_workers_per_tracker]
        if len(xs) == 1:
            return [xs[0]] * len(trackers)
        if len(xs) == len(trackers):
            return xs
        raise ValueError(
            f"--num_workers_per_tracker length must be 1 or equal to number of trackers being run "
            f"({len(trackers)}). Got {len(xs)}."
        )

    if DEFAULT_WORKERS_PER_TRACKER is not None:
        out = []
        for t in trackers:
            out.append(int(DEFAULT_WORKERS_PER_TRACKER.get(t, num_workers)))
        return out

    return [int(num_workers)] * len(trackers)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    only = set([t.strip() for t in (args.only or []) if t.strip()]) or None
    trackers = [t for t in TRACKER_ORDER if (only is None or t in only)]

    workers_list = _resolve_workers_for_trackers(
        trackers,
        parallel=bool(args.parallel),
        num_workers=int(args.num_workers),
        num_workers_per_tracker=args.num_workers_per_tracker,
    )

    if not args.quiet:
        print("[tracker_eval] Will run trackers in order:", ", ".join(trackers))
        print("[tracker_eval] Splits:", ", ".join(args.split_root))

        if args.parallel:
            parts = [f"{t}:{w if w != 0 else 'auto'}" for t, w in zip(trackers, workers_list)]
            print(f"[tracker_eval] Parallel mode ON | start_method={args.parallel_start_method}")
            print("[tracker_eval] Workers per tracker:", ", ".join(parts))

    for t, w in zip(trackers, workers_list):
        if not args.quiet:
            print("")
            print("=" * 80)
            print(f"[tracker_eval] Running tracker: {t}")
            print("=" * 80)

        cmd: List[str] = []
        cmd = _append_shared(args, cmd)
        cmd += ["--tracker", t]

        # Forward parallel flags with per-tracker worker count
        if args.parallel:
            cmd += ["--parallel"]
            cmd += ["--num_workers", str(int(w))]
            cmd += ["--parallel_start_method", str(args.parallel_start_method)]

        rc = run_one_tracker_main(cmd)
        if rc != 0:
            raise SystemExit(rc)

    if not args.quiet:
        print("")
        print("[tracker_eval] All trackers completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
