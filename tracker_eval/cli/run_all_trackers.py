# tracker_eval/cli/run_all_trackers.py

from __future__ import annotations

import argparse
from typing import List, Optional

from tracker_eval.cli.run_tracker import main as run_one_tracker_main


TRACKER_ORDER = [
    "elptnet",
    "cbmot",
    "fastpoly",
    "ab3dmot",
    "gnnpmb",
    "simpletrack",
]


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

    # Tracker-specific paths / params (same as your current usage)
    p.add_argument("--elptnet_cfg_file", type=str, default="/home/scai/trackers/ELPTNet/jrdb.yaml")
    p.add_argument("--elptnet_track_class", type=str, default="pedestrian")
    p.add_argument("--elptnet_fps", type=float, default=15.0)

    p.add_argument("--fastpoly_config", type=str, default="/home/scai/trackers/FastPoly/config/nusc_config.yaml")

    p.add_argument(
        "--ab3dmot_log_dir",
        type=str,
        default=None,
        help="If set, AB3DMOT logs go here; otherwise AB3DMOT adapter default.",
    )

    p.add_argument(
        "--gnnpmb_parameters_path",
        type=str,
        default="/home/scai/trackers/GnnPmbTracker/configs/gnnpmb_parameters.json",
    )
    p.add_argument("--gnnpmb_classification", type=str, default="pedestrian")
    p.add_argument("--gnnpmb_fps", type=float, default=15.0)

    p.add_argument(
        "--simpletrack_config",
        type=str,
        default="/home/scai/trackers/SimpleTrack/configs/nu_configs/giou.yaml",
    )

    # CBMOT core ones you’ve been using
    p.add_argument("--cbmot_track_class", type=str, default="pedestrian")
    p.add_argument("--cbmot_fps", type=float, default=15.0)

    # Control
    p.add_argument(
        "--only",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of trackers to run (names: elptnet, cbmot, fastpoly, ab3dmot, gnnpmb, simpletrack).",
    )

    return p


def _append_shared(args: argparse.Namespace, argv: List[str]) -> List[str]:
    argv += ["--split_root", *args.split_root]
    if args.split_name is not None and len(args.split_name) > 0:
        argv += ["--split_name", *args.split_name]

    argv += ["--out_root", args.out_root]
    argv += ["--detections_subdir", args.detections_subdir]

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
    return argv


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    only = set([t.strip() for t in (args.only or []) if t.strip()]) or None
    trackers = [t for t in TRACKER_ORDER if (only is None or t in only)]

    if not args.quiet:
        print("[tracker_eval] Will run trackers in order:", ", ".join(trackers))
        print("[tracker_eval] Splits:", ", ".join(args.split_root))

    for t in trackers:
        if not args.quiet:
            print("")
            print("=" * 80)
            print(f"[tracker_eval] Running tracker: {t}")
            print("=" * 80)

        cmd: List[str] = []
        cmd = _append_shared(args, cmd)
        cmd += ["--tracker", t]

        # tracker-specific forwarding
        if t == "elptnet":
            cmd += ["--elptnet_cfg_file", args.elptnet_cfg_file]
            cmd += ["--elptnet_track_class", args.elptnet_track_class]
            cmd += ["--elptnet_fps", str(args.elptnet_fps)]
        elif t == "cbmot":
            cmd += ["--cbmot_track_class", args.cbmot_track_class]
            cmd += ["--cbmot_fps", str(args.cbmot_fps)]
            # You can add more CBMOT knobs here later if you want.
        elif t == "fastpoly":
            cmd += ["--fastpoly_config", args.fastpoly_config]
        elif t == "ab3dmot":
            if args.ab3dmot_log_dir is not None:
                cmd += ["--ab3dmot_log_dir", args.ab3dmot_log_dir]
        elif t == "gnnpmb":
            cmd += ["--gnnpmb_parameters_path", args.gnnpmb_parameters_path]
            cmd += ["--gnnpmb_classification", args.gnnpmb_classification]
            cmd += ["--gnnpmb_fps", str(args.gnnpmb_fps)]
        elif t == "simpletrack":
            cmd += ["--simpletrack_config", args.simpletrack_config]
        else:
            raise ValueError(f"Unhandled tracker in run_all_trackers: {t}")

        # Run by calling the same python entry logic (no subprocess needed)
        rc = run_one_tracker_main(cmd)
        if rc != 0:
            raise SystemExit(rc)

    if not args.quiet:
        print("")
        print("[tracker_eval] All trackers completed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
