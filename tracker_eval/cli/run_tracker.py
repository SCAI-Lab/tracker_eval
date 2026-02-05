# tracker_eval/cli/run_tracker.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from tracker_eval.runner.run_split import run_tracker_on_split
from tracker_eval.trackers.ab3dmot_adapter import AB3DMOTAdapter, AB3DMOTConfig
from tracker_eval.trackers.simpletrack_adapter import SimpleTrackAdapter, SimpleTrackConfig
from tracker_eval.trackers.fastpoly_adapter import FastPolyAdapter, FastPolyConfig
from tracker_eval.trackers.gnnpmbtracker_adapter import GNNPMBAdapter, GNNPMBConfig
from tracker_eval.trackers.cbmot_adapter import CBMOTAdapter, CBMOTConfig




def _parse_list_arg(xs: Optional[List[str]]) -> Optional[List[str]]:
    if xs is None:
        return None
    out: List[str] = []
    for x in xs:
        if x is None:
            continue
        # allow comma-separated and repeated flags
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        out.extend(parts)
    return out or None


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker_eval.run_tracker",
        description="Run a 3D MOT tracker on a JRDB split (Jetson runtime eval + export KITTI txt for JRDB toolkit).",
    )

    # IO
    p.add_argument("--split_root", type=str, required=True, help="Path to split root, e.g. /mnt/nvme/JRDB_track/test")
    p.add_argument("--split_name", type=str, default=None, help="Name for this split (default: folder name of split_root)")
    p.add_argument("--out_root", type=str, required=True, help="Output root directory for tracker outputs")

    p.add_argument(
        "--detections_subdir",
        type=str,
        default="detections_3D",
        help="Detections subfolder name under split_root (default: detections_3D)",
    )

    # Tracker selection (only AB3DMOT for now)
    p.add_argument(
        "--tracker",
        type=str,
        default="ab3dmot",
        choices=["ab3dmot", "simpletrack", "fastpoly", "gnnpmb", "cbmot", "elptnet"],
        help="Tracker to run (currently only: ab3dmot)",
    )

    # Runtime options
    p.add_argument("--warmup_steps", type=int, default=0, help="Exclude first N frames per sequence from timing stats")
    p.add_argument("--limit_sequences", type=int, default=None, help="Run only first N sequences (after filtering)")
    p.add_argument(
        "--include_sequences",
        type=str,
        nargs="*",
        default=None,
        help="Only run these sequences (repeat flag or comma-separate names)",
    )
    p.add_argument(
        "--exclude_sequences",
        type=str,
        nargs="*",
        default=None,
        help="Skip these sequences (repeat flag or comma-separate names)",
    )
    p.add_argument("--no_skip_existing", action="store_true", help="Do NOT skip sequences with existing KITTI outputs")
    p.add_argument(
        "--write_tracks_json",
        action="store_true",
        help="Also write internal tracks JSON per sequence (debug/transfer)",
    )

    # Output options
    p.add_argument(
        "--tracker_subfolder",
        type=str,
        default="data",
        help="Subfolder name under <out_root>/<tracker_name>/ for KITTI txt (default: data)",
    )
    p.add_argument(
        "--no_kitti_score",
        action="store_true",
        help="Do not write score column in KITTI txt (default writes score column).",
    )

    # AB3DMOT params
    g = p.add_argument_group("AB3DMOT parameters")
    g.add_argument("--ab3dmot_max_age", type=int, default=7)
    g.add_argument("--ab3dmot_min_hits", type=int, default=5)
    g.add_argument("--ab3dmot_thresh_iou", type=float, default=0.33)
    g.add_argument("--ab3dmot_thresh_dist", type=float, default=0.5)
    g.add_argument(
        "--ab3dmot_metrics",
        type=str,
        default="iou_3d,dist_3d",
        help="Comma-separated metrics list, default: iou_3d,dist_3d",
    )
    g.add_argument(
        "--ab3dmot_log_dir",
        type=str,
        default=None,
        help="Directory to store AB3DMOT per-sequence log files (<seq>.txt). If not set, logs go to /tmp.",
    )

    # SimpleTrack params
    g2 = p.add_argument_group("SimpleTrack parameters")
    g2.add_argument(
        "--simpletrack_config",
        type=str,
        default=None,
        help="Path to SimpleTrack YAML config (e.g. /home/scai/trackers/SimpleTrack/configs/nu_configs/giou.yaml)",
    )

    g3 = p.add_argument_group("FastPoly parameters")
    g3.add_argument(
        "--fastpoly_config",
        type=str,
        default=None,
        help="Path to FastPoly YAML config (e.g. /home/scai/trackers/FastPoly/config/nusc_config.yaml)",
    )   

    g4 = p.add_argument_group("GNNPMB parameters")
    g4.add_argument(
        "--gnnpmb_parameters_path",
        type=str,
        default=None,
        help="Path to GNNPMB parameters JSON (the one used by readout_parameters()).",
    )
    g4.add_argument(
        "--gnnpmb_classification",
        type=str,
        default="pedestrian",
        help="Class key in the parameters JSON (default: pedestrian).",
    )
    g4.add_argument(
        "--gnnpmb_fps",
        type=float,
        default=15.0,
        help="FPS used to compute dt if timestamps are not provided (default: 15.0 for JRDB).",
    )
    g4.add_argument(
        "--gnnpmb_no_nms",
        action="store_true",
        help="Disable upstream NMS inside adapter (default: NMS enabled).",
    )
    g4.add_argument(
        "--gnnpmb_giou_gating",
        type=float,
        default=-0.5,
        help="giou_gating passed into PMBMGNN update() (default: -0.5).",
    )
    g4.add_argument(
        "--gnnpmb_ped_empty_meas_extract_thr",
        type=float,
        default=0.7,
        help="If pedestrian and no measurements, use extractStates_with_custom_thr(thr=...) (default: 0.7).",
    )

    g5 = p.add_argument_group("CBMOT parameters")
    g5.add_argument("--cbmot_hungarian", action="store_true")
    g5.add_argument("--cbmot_max_age", type=int, default=40)
    g5.add_argument("--cbmot_min_hits", type=int, default=1)
    g5.add_argument("--cbmot_score_decay", type=float, default=0.0)
    g5.add_argument("--cbmot_active_th", type=float, default=1.0)
    g5.add_argument("--cbmot_deletion_th", type=float, default=0.0)
    g5.add_argument("--cbmot_detection_th", type=float, default=0.0)
    g5.add_argument("--cbmot_score_update", type=str, default=None)
    g5.add_argument("--cbmot_model_path", type=str, default=None)
    g5.add_argument("--cbmot_fps", type=float, default=15.0)
    g5.add_argument("--cbmot_track_class", type=str, default="pedestrian")
    g5.add_argument("--cbmot_export_score", action="store_true")

    g6 = p.add_argument_group("ELPTnet parameters")
    g6.add_argument("--elptnet_cfg_file", type=str, required=False, default=None,
                    help="Path to ELPTnet jrdb.yaml")
    g6.add_argument("--elptnet_fps", type=float, default=15.0)
    g6.add_argument("--elptnet_track_class", type=str, default="pedestrian")
    g6.add_argument("--elptnet_input_score", type=float, default=0.0)
    g6.add_argument("--elptnet_export_score", action="store_true")
    g6.add_argument("--elptnet_timestamp_mode", type=str, default="frame_index",
                    choices=["frame_index", "seconds"])


    # Misc
    p.add_argument("--quiet", action="store_true", help="Reduce printing")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    split_root = Path(args.split_root)
    if args.split_name is None:
        split_name = split_root.name
    else:
        split_name = str(args.split_name)

    include = _parse_list_arg(args.include_sequences)
    exclude = _parse_list_arg(args.exclude_sequences)

    if args.tracker == "ab3dmot":
        metrics = tuple([m.strip() for m in str(args.ab3dmot_metrics).split(",") if m.strip()])
        if len(metrics) == 0:
            metrics = ("iou_3d", "dist_3d")

        cfg = AB3DMOTConfig(
            max_age=int(args.ab3dmot_max_age),
            min_hits=int(args.ab3dmot_min_hits),
            thresh_3d_iou=float(args.ab3dmot_thresh_iou),
            thresh_3d_dist=float(args.ab3dmot_thresh_dist),
            metrics=metrics,  # type: ignore[arg-type]
            log_dir=args.ab3dmot_log_dir,
        )
        tracker = AB3DMOTAdapter(cfg=cfg)
        tracker_name = tracker.name
    elif args.tracker == "simpletrack":
        if args.simpletrack_config is None:
            raise ValueError("--simpletrack_config is required when --tracker simpletrack")

        cfg = SimpleTrackConfig(config_path=str(args.simpletrack_config))
        tracker = SimpleTrackAdapter(cfg=cfg)
        tracker_name = tracker.name
    elif args.tracker == "fastpoly":
        import yaml
        from tracker_eval.trackers.fastpoly_adapter import FastPolyAdapter, FastPolyConfig

        with open(args.fastpoly_config, "r") as f:
            cfg_dict = yaml.safe_load(f)

        tracker = FastPolyAdapter(
            cfg=FastPolyConfig(
                config=cfg_dict,
                seq_id=0,
                has_velo=False,
                is_key_frame=True,
                use_numeric_frame_id=True,
            )
        )
        tracker_name = tracker.name
    elif args.tracker == "gnnpmb":
        if args.gnnpmb_parameters_path is None:
            raise ValueError("--gnnpmb_parameters_path is required when --tracker gnnpmb")

        tracker = GNNPMBAdapter(
            cfg=GNNPMBConfig(
                parameters_path=str(args.gnnpmb_parameters_path),
                classification=str(args.gnnpmb_classification),
                use_nms=not bool(args.gnnpmb_no_nms),
                fps=float(args.gnnpmb_fps),
                giou_gating=float(args.gnnpmb_giou_gating),
                ped_empty_meas_extract_thr=float(args.gnnpmb_ped_empty_meas_extract_thr),
            )
        )
        tracker_name = tracker.name
    elif args.tracker == "cbmot":
        cfg = CBMOTConfig(
            hungarian=bool(args.cbmot_hungarian),
            max_age=int(args.cbmot_max_age),
            min_hits=int(args.cbmot_min_hits),
            score_decay=float(args.cbmot_score_decay),
            active_th=float(args.cbmot_active_th),
            deletion_th=float(args.cbmot_deletion_th),
            detection_th=float(args.cbmot_detection_th),
            score_update=args.cbmot_score_update if args.cbmot_score_update not in ("", "none", "None") else None,
            model_path=args.cbmot_model_path,
            fps=float(args.cbmot_fps),
            track_class=str(args.cbmot_track_class) if args.cbmot_track_class not in ("", "none", "None") else None,
            export_score=bool(args.cbmot_export_score),
        )
        tracker = CBMOTAdapter(cfg=cfg)
        tracker_name = tracker.name
    elif args.tracker == "elptnet":
        if args.elptnet_cfg_file is None:
            raise ValueError("--elptnet_cfg_file is required when --tracker elptnet")

        from tracker_eval.trackers.elptnet_adapter import ELPTnetAdapter, ELPTnetConfig

        cfg = ELPTnetConfig(
            cfg_file=str(args.elptnet_cfg_file),
            fps=float(args.elptnet_fps),
            track_class=str(args.elptnet_track_class),
            input_score=float(args.elptnet_input_score),
            export_score=bool(args.elptnet_export_score),
            timestamp_mode=str(args.elptnet_timestamp_mode),
        )
        tracker = ELPTnetAdapter(cfg=cfg)
        tracker_name = tracker.name
    else:
        raise ValueError(f"Unsupported tracker: {args.tracker}")

    summary = run_tracker_on_split(
        split_root=str(split_root),
        split_name=split_name,
        tracker=tracker,
        tracker_name=tracker_name,
        out_root=str(args.out_root),
        detections_subdir=str(args.detections_subdir),
        warmup_steps=int(args.warmup_steps),
        limit_sequences=int(args.limit_sequences) if args.limit_sequences is not None else None,
        include_sequences=include,
        exclude_sequences=exclude,
        write_kitti_txt=True,
        write_tracks_json=bool(args.write_tracks_json),
        kitti_use_score=not bool(args.no_kitti_score),
        tracker_subfolder=str(args.tracker_subfolder),
        skip_existing_kitti=not bool(args.no_skip_existing),
        verbose=not bool(args.quiet),
    )

    # Print a concise end-of-run summary
    agg = summary.aggregate
    tracker_dir = summary.io.get("tracker_dir", "")
    kitti_dir = summary.io.get("kitti_dir", "")

    if not args.quiet:
        fps_w = agg.get("fps", {}).get("frame_weighted", 0.0)
        mean_ms = agg.get("step_ms", {}).get("mean_of_mean", 0.0)
        nseq = agg.get("num_sequences", 0)
        nframes = agg.get("num_frames_total", 0)
        print("")
        print("[tracker_eval] Done.")
        print(f"[tracker_eval] Sequences run: {nseq} | Frames total: {nframes}")
        print(f"[tracker_eval] FPS (frame-weighted): {fps_w:.2f} | Mean step (mean-of-mean): {mean_ms:.2f} ms")
        print(f"[tracker_eval] Outputs: {tracker_dir}")
        print(f"[tracker_eval] KITTI txt: {kitti_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
