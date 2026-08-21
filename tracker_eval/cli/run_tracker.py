"""Run one or more tracker configurations on JRDB detection sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from tracker_eval.runner.run_batch import BatchRunRequest, run_batch
from tracker_eval.runner.run_split import (
    _build_tracker_from_spec,
    run_tracker_on_split,
)
from tracker_eval.trackers.pedreftrack_adapter import PEDREFTRACK_MODES
from tracker_eval.trackers.paths import (
    ELPTNET_CONFIG,
    FASTPOLY_CONFIG,
    GNNPMB_CONFIG,
    SIMPLETRACK_CONFIG,
)


TRACKER_ORDER = [
    "pedreftrack",
    "elptnet",
    "cbmot",
    "fastpoly",
    "ab3dmot",
    "gnnpmb",
    "simpletrack",
]


def split_values(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        out.extend(
            part.strip()
            for part in str(value).split(",")
            if part.strip()
        )
    return out


def normalize_split_names(
    split_roots: Sequence[str],
    split_names: Optional[Sequence[str]],
) -> List[str]:
    if not split_names:
        return [Path(root).name for root in split_roots]
    names = list(split_names)
    if len(names) != len(split_roots):
        raise ValueError(
            "Provide one --split_name per --split_root, or omit it."
        )
    return [str(name) for name in names]


def resolve_trackers(values: Optional[Sequence[str]]) -> List[str]:
    requested = split_values(values) or ["ab3dmot"]
    if "all" in requested:
        requested = list(TRACKER_ORDER)
    unknown = [name for name in requested if name not in TRACKER_ORDER]
    if unknown:
        raise ValueError(f"Unknown tracker(s): {', '.join(unknown)}")
    return list(dict.fromkeys(requested))


def resolve_pedreftrack_modes(values: Optional[Sequence[str]]) -> List[str]:
    requested = split_values(values) or ["no_gt"]
    if "all" in requested:
        requested = list(PEDREFTRACK_MODES)
    unknown = [mode for mode in requested if mode not in PEDREFTRACK_MODES]
    if unknown:
        raise ValueError(
            "Unknown PedRefTrack mode(s): " + ", ".join(unknown)
        )
    return [mode for mode in PEDREFTRACK_MODES if mode in requested]


def _load_manifest_variants(path: str) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    variants = [
        str(entry.get("name", "")).strip()
        for entry in manifest.get("variants", [])
        if str(entry.get("name", "")).strip()
    ]
    if not variants:
        raise ValueError(f"No variants found in {path}")
    return variants


def canonical_output_name(
    tracker: str,
    *,
    global_coords: bool,
    pedreftrack_mode: Optional[str] = None,
    variant: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Place coordinate, PedRefTrack mode, variant, and suffix tags canonically."""
    name = str(tracker)
    if global_coords:
        name += "__global"
    if tracker == "pedreftrack":
        mode = str(pedreftrack_mode or "no_gt")
        if mode not in PEDREFTRACK_MODES:
            raise ValueError(f"Invalid PedRefTrack mode: {mode}")
        name += f"_{mode}"
    for value in (variant, suffix):
        if value is None:
            continue
        clean = str(value).strip().strip("_")
        if clean:
            if "/" in clean or "\\" in clean:
                raise ValueError("Output name tags cannot contain slashes.")
            name += f"_{clean}"
    return name


def build_argparser(
    *,
    split_root_required: bool = True,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker_eval.run_tracker",
        description=(
            "Run one or more trackers, PedRefTrack modes, variants, and JRDB "
            "splits. --parallel uses one shared sequence scheduler."
        ),
    )
    parser.add_argument(
        "--split_root",
        nargs="+",
        required=split_root_required,
    )
    parser.add_argument("--split_name", nargs="*", default=None)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--detections_subdir", default="detections_3D")
    parser.add_argument("--labels_subdir", default="labels_3d")
    parser.add_argument(
        "--trackers",
        "--tracker",
        dest="trackers",
        nargs="+",
        default=None,
        help="Space/comma-separated trackers, or 'all'.",
    )
    parser.add_argument(
        "--pedreftrack_modes",
        "--pedreftrack-modes",
        nargs="*",
        default=None,
        help="no_gt (default), gt_assisted, or all.",
    )

    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--limit_sequences", type=int, default=None)
    parser.add_argument("--include_sequences", nargs="*", default=None)
    parser.add_argument("--exclude_sequences", nargs="*", default=None)
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument("--tracker_subfolder", default="data")
    parser.add_argument("--tracker_name_suffix", default=None)
    parser.add_argument("--no_kitti_score", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--parallel", action="store_true")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Global process limit for the shared scheduler. Zero uses the "
            "sum of selected per-tracker defaults."
        ),
    )
    parser.add_argument(
        "--parallel_start_method",
        choices=["spawn", "fork", "forkserver"],
        default="spawn",
    )

    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--variants_subdir", default=None)
    parser.add_argument("--variants_from_manifest", default=None)
    parser.add_argument("--exclude_variants", nargs="*", default=None)

    parser.add_argument("--global_coords", action="store_true")
    parser.add_argument("--odometry_root", default="")

    group = parser.add_argument_group("AB3DMOT parameters")
    group.add_argument("--ab3dmot_max_age", type=int, default=15)
    group.add_argument("--ab3dmot_min_hits", type=int, default=3)
    group.add_argument("--ab3dmot_thresh_dist", type=float, default=0.5)
    group.add_argument("--ab3dmot_thresh_iou", type=float, default=-0.4)
    group.add_argument("--ab3dmot_metrics", default="giou_3d")
    group.add_argument("--ab3dmot_log_dir", default=None)

    group = parser.add_argument_group("SimpleTrack parameters")
    group.add_argument(
        "--simpletrack_config",
        default=str(SIMPLETRACK_CONFIG),
    )

    group = parser.add_argument_group("FastPoly parameters")
    group.add_argument(
        "--fastpoly_config",
        default=str(FASTPOLY_CONFIG),
    )

    group = parser.add_argument_group("GNN-PMB parameters")
    group.add_argument(
        "--gnnpmb_parameters_path",
        default=str(GNNPMB_CONFIG),
    )
    group.add_argument("--gnnpmb_classification", default="pedestrian")
    group.add_argument("--gnnpmb_fps", type=float, default=15.0)
    group.add_argument("--gnnpmb_nms", action="store_true")
    group.add_argument("--gnnpmb_giou_gating", type=float, default=-0.5)
    group.add_argument(
        "--gnnpmb_ped_empty_meas_extract_thr",
        type=float,
        default=0.5,
    )

    group = parser.add_argument_group("CBMOT parameters")
    group.add_argument(
        "--cbmot_hungarian",
        dest="cbmot_hungarian",
        action="store_true",
        default=True,
    )
    group.add_argument(
        "--no_cbmot_hungarian",
        dest="cbmot_hungarian",
        action="store_false",
    )
    group.add_argument("--cbmot_max_age", type=int, default=31)
    group.add_argument("--cbmot_min_hits", type=int, default=1)
    group.add_argument("--cbmot_score_decay", type=float, default=0.05)
    group.add_argument("--cbmot_active_th", type=float, default=0.80)
    group.add_argument("--cbmot_deletion_th", type=float, default=0.00)
    group.add_argument("--cbmot_detection_th", type=float, default=0.15)
    group.add_argument("--cbmot_score_update", default="multiplication")
    group.add_argument("--cbmot_model_path", default=None)
    #
    # XY filtering model; KF is appropriate when detector velocity is unavailable.
    group.add_argument(
        "--cbmot_motion_model",
        choices=["KF", "PointTracker"],
        default="KF",
    )
    # Maximum XY detection-to-track association distance in metres.
    group.add_argument("--cbmot_distance_gate_m", type=float, default=0.7)
    # Optional confidence decay per second; overrides per-frame score_decay.
    group.add_argument("--cbmot_score_decay_per_s", type=float, default=0.4)
    #
    group.add_argument("--cbmot_fps", type=float, default=15.0)
    group.add_argument("--cbmot_track_class", default="pedestrian")
    group.add_argument("--cbmot_export_score", action="store_true")

    group = parser.add_argument_group("ELPTNet parameters")
    group.add_argument(
        "--elptnet_cfg_file",
        default=str(ELPTNET_CONFIG),
    )
    group.add_argument("--elptnet_fps", type=float, default=15.0)
    group.add_argument("--elptnet_track_class", default="pedestrian")
    group.add_argument("--elptnet_input_score", type=float, default=0.5)
    #
    # Maximum duration in seconds for emitting CA predictions during a gap.
    group.add_argument("--elptnet_output_coast_s", type=float, default=0.5)
    # Maximum ELPTNet association cost/distance in metres.
    group.add_argument("--elptnet_association_gate_m", type=float, default=0.2)
    #
    group.add_argument("--elptnet_export_score", action="store_true")
    # group.add_argument(
    #     "--elptnet_timestamp_mode",
    #     choices=["frame_index", "seconds"],
    #     default="frame_index",
    # )

    group = parser.add_argument_group("PedRefTrack parameters")
    # Input detection frequency in hertz.
    group.add_argument("--pedreftrack_fps", type=float, default=15.0)
    # Moving-track identity-retention duration in seconds.
    group.add_argument("--pedreftrack_T_reid_base_s", type=float, default=2.5)
    # Static-track identity-retention duration in seconds.
    group.add_argument("--pedreftrack_T_reid_static_s", type=float, default=5.0)
    # Supported time in seconds at which the lowest confirmation score applies.
    group.add_argument(
        "--pedreftrack_confirmation_target_s",
        type=float,
        default=0.25,
    )
    # Detector score required to confirm a track from one matched observation.
    group.add_argument(
        "--pedreftrack_confirmation_one_hit_score",
        type=float,
        default=0.95,
    )
    # Lowest mean score accepted at or after the confirmation target.
    group.add_argument(
        "--pedreftrack_confirmation_min_score",
        type=float,
        default=0.50,
    )
    # Maximum detector gap in seconds retained for an unconfirmed track.
    group.add_argument(
        "--pedreftrack_tentative_max_gap_s",
        type=float,
        default=0.50,
    )
    # Required track history in seconds before residual-based coasting applies.
    group.add_argument(
        "--pedreftrack_motion_robustness_history_s",
        type=float,
        default=1.00,
    )
    # Recent residual window in seconds that can shorten adaptive coasting.
    group.add_argument(
        "--pedreftrack_motion_robustness_immediate_history_s",
        type=float,
        default=0.25,
    )
    # Residual error in metres below which maximum coasting is retained.
    group.add_argument(
        "--pedreftrack_motion_error_free_m",
        type=float,
        default=0.05,
    )
    # Additional residual error in metres that halves the coasting extension.
    group.add_argument(
        "--pedreftrack_motion_error_half_decay_m",
        type=float,
        default=0.038,
    )
    # Minimum motion-selected detector-gap output duration in seconds.
    group.add_argument("--pedreftrack_T_out_min_s", type=float, default=0.50)
    # Maximum motion-selected detector-gap output duration in seconds.
    group.add_argument("--pedreftrack_T_out_max_s", type=float, default=2.0)
    # Minimum BEV IoU association threshold for first pass association.
    group.add_argument("--pedreftrack_assoc_iou_first_pass_thr", type=float, default=0.33)
    # Maximum short-term XY association distance in metres.
    group.add_argument("--pedreftrack_dist_gate_m", type=float, default=0.4)
    # Maximum vertical association distance in metres; <=0 disables it.
    group.add_argument("--pedreftrack_z_gate_m", type=float, default=0.5)
    # Maximum covariance-derived XY association radius in metres.
    group.add_argument(
        "--pedreftrack_kf_max_gate_m",
        type=float,
        default=1.0,
    )
    return parser


def build_tracker_spec(
    args: argparse.Namespace,
    tracker: str,
    *,
    pedreftrack_mode: str = "no_gt",
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {"tracker": tracker}
    if tracker == "ab3dmot":
        metrics = split_values([args.ab3dmot_metrics]) or [
            "iou_3d",
            "dist_3d",
        ]
        spec["cfg"] = {
            "max_age": args.ab3dmot_max_age,
            "min_hits": args.ab3dmot_min_hits,
            "thresh_3d_iou": args.ab3dmot_thresh_iou,
            "thresh_3d_dist": args.ab3dmot_thresh_dist,
            "metrics": metrics,
            "log_dir": args.ab3dmot_log_dir,
        }
    elif tracker == "simpletrack":
        spec["cfg"] = {"config_path": str(args.simpletrack_config)}
    elif tracker == "fastpoly":
        with Path(args.fastpoly_config).open(
            "r", encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        spec["cfg"] = {
            "config": config,
            "seq_id": 0,
            "has_velo": False,
            "is_key_frame": True,
            "use_numeric_frame_id": True,
            "force_class_label": None,
        }
    elif tracker == "gnnpmb":
        spec["cfg"] = {
            "parameters_path": str(args.gnnpmb_parameters_path),
            "classification": str(args.gnnpmb_classification),
            "use_nms": bool(args.gnnpmb_nms),
            "fps": args.gnnpmb_fps,
            "giou_gating": args.gnnpmb_giou_gating,
            "ped_empty_meas_extract_thr":
                args.gnnpmb_ped_empty_meas_extract_thr,
        }
    elif tracker == "cbmot":
        spec["cfg"] = {
            "hungarian": bool(args.cbmot_hungarian),
            "max_age": args.cbmot_max_age,
            "min_hits": args.cbmot_min_hits,
            "score_decay": args.cbmot_score_decay,
            "active_th": args.cbmot_active_th,
            "deletion_th": args.cbmot_deletion_th,
            "detection_th": args.cbmot_detection_th,
            "score_update": args.cbmot_score_update,
            "model_path": args.cbmot_model_path,
            "motion_model": args.cbmot_motion_model,
            "distance_gate_m": args.cbmot_distance_gate_m,
            "score_decay_per_s": args.cbmot_score_decay_per_s,
            "fps": args.cbmot_fps,
            "track_class": args.cbmot_track_class,
            "export_score": bool(args.cbmot_export_score),
        }
    elif tracker == "elptnet":
        spec["cfg"] = {
            "cfg_file": str(args.elptnet_cfg_file),
            "output_coast_s": args.elptnet_output_coast_s,
            "association_gate_m": args.elptnet_association_gate_m,
            "fps": args.elptnet_fps,
            "track_class": args.elptnet_track_class,
            "input_score": args.elptnet_input_score,
            "export_score": bool(args.elptnet_export_score),
            # "timestamp_mode": args.elptnet_timestamp_mode,
        }
    elif tracker == "pedreftrack":
        spec["cfg"] = {
            "mode": pedreftrack_mode,
            "fps": args.pedreftrack_fps,
            "T_reid_base_s": args.pedreftrack_T_reid_base_s,
            "T_reid_static_s": args.pedreftrack_T_reid_static_s,
            "confirmation_target_s":
                args.pedreftrack_confirmation_target_s,
            "confirmation_one_hit_score":
                args.pedreftrack_confirmation_one_hit_score,
            "confirmation_min_score":
                args.pedreftrack_confirmation_min_score,
            "tentative_max_gap_s":
                args.pedreftrack_tentative_max_gap_s,
            "motion_robustness_history_s":
                args.pedreftrack_motion_robustness_history_s,
            "motion_robustness_immediate_history_s":
                args.pedreftrack_motion_robustness_immediate_history_s,
            "motion_error_free_m":
                args.pedreftrack_motion_error_free_m,
            "motion_error_half_decay_m":
                args.pedreftrack_motion_error_half_decay_m,
            "T_out_min_s": args.pedreftrack_T_out_min_s,
            "T_out_max_s": args.pedreftrack_T_out_max_s,
            "assoc_iou_first_pass_thr": args.pedreftrack_assoc_iou_first_pass_thr,
            "dist_gate_m": args.pedreftrack_dist_gate_m,
            "z_gate_m": args.pedreftrack_z_gate_m,
            "kf_max_gate_m": args.pedreftrack_kf_max_gate_m,
        }
    else:
        raise ValueError(f"Unsupported tracker: {tracker}")
    return spec


def selected_variants(args: argparse.Namespace) -> List[Optional[str]]:
    variants = split_values(args.variants)
    if not variants and args.variants_from_manifest:
        variants = _load_manifest_variants(args.variants_from_manifest)
    excluded = set(split_values(args.exclude_variants))
    variants = [value for value in variants if value not in excluded]
    if (args.variants or args.variants_from_manifest) and not variants:
        raise ValueError("No pseudo-detection variants remain selected.")
    return variants or [None]


def make_requests(
    args: argparse.Namespace,
    *,
    variant_overrides: Optional[Sequence[Dict[str, str]]] = None,
) -> List[BatchRunRequest]:
    trackers = resolve_trackers(args.trackers)
    pedreftrack_modes = resolve_pedreftrack_modes(args.pedreftrack_modes)
    split_roots = [str(value) for value in args.split_root]
    split_names = normalize_split_names(split_roots, args.split_name)
    include = split_values(args.include_sequences) or None
    exclude = split_values(args.exclude_sequences) or None

    if variant_overrides is None:
        variants = []
        for variant in selected_variants(args):
            if variant is None:
                variants.append(
                    {
                        "name": "",
                        "detections_subdir": str(
                            args.detections_subdir
                        ),
                        "labels_subdir": str(args.labels_subdir),
                    }
                )
            else:
                if not args.variants_subdir:
                    raise ValueError(
                        "--variants_subdir is required with variants."
                    )
                variants.append(
                    {
                        "name": str(variant),
                        "detections_subdir": str(
                            Path(args.variants_subdir) / variant
                        ),
                        "labels_subdir": str(args.labels_subdir),
                    }
                )
    else:
        variants = [dict(entry) for entry in variant_overrides]

    requests: List[BatchRunRequest] = []
    for tracker in trackers:
        modes: Sequence[Optional[str]] = (
            pedreftrack_modes if tracker == "pedreftrack" else [None]
        )
        for mode in modes:
            spec = build_tracker_spec(
                args,
                tracker,
                pedreftrack_mode=str(mode or "no_gt"),
            )
            for split_root, split_name in zip(
                split_roots, split_names
            ):
                for variant in variants:
                    name = canonical_output_name(
                        tracker,
                        global_coords=bool(args.global_coords),
                        pedreftrack_mode=mode,
                        variant=variant.get("name") or None,
                        suffix=args.tracker_name_suffix,
                    )
                    requests.append(
                        BatchRunRequest(
                            tracker_key=tracker,
                            tracker_spec=spec,
                            tracker_name=name,
                            split_root=split_root,
                            split_name=split_name,
                            out_root=str(args.out_root),
                            detections_subdir=str(
                                variant["detections_subdir"]
                            ),
                            labels_subdir=str(
                                variant["labels_subdir"]
                            ),
                            use_gt_if_available=(
                                tracker != "pedreftrack"
                                or mode == "gt_assisted"
                            ),
                            warmup_steps=args.warmup_steps,
                            limit_sequences=args.limit_sequences,
                            include_sequences=include,
                            exclude_sequences=exclude,
                            kitti_use_score=not args.no_kitti_score,
                            tracker_subfolder=args.tracker_subfolder,
                            skip_existing_kitti=not args.no_skip_existing,
                            global_coords=bool(args.global_coords),
                            odometry_root=str(args.odometry_root),
                        )
                    )
    return requests


def execute_requests(
    args: argparse.Namespace,
    requests: Sequence[BatchRunRequest],
) -> int:
    verbose = not bool(args.quiet)
    if args.parallel:
        summaries = run_batch(
            requests,
            num_workers=args.num_workers,
            start_method=args.parallel_start_method,
            verbose=verbose,
        )
    else:
        summaries = []
        for request in requests:
            tracker = _build_tracker_from_spec(request.tracker_spec)
            summaries.append(
                run_tracker_on_split(
                    split_root=request.split_root,
                    split_name=request.split_name,
                    tracker=tracker,
                    tracker_name=request.tracker_name,
                    out_root=request.out_root,
                    detections_subdir=request.detections_subdir,
                    labels_subdir=request.labels_subdir,
                    use_gt_if_available=request.use_gt_if_available,
                    warmup_steps=request.warmup_steps,
                    limit_sequences=request.limit_sequences,
                    include_sequences=request.include_sequences,
                    exclude_sequences=request.exclude_sequences,
                    kitti_use_score=request.kitti_use_score,
                    tracker_subfolder=request.tracker_subfolder,
                    skip_existing_kitti=request.skip_existing_kitti,
                    verbose=verbose,
                    parallel=False,
                    global_coords=request.global_coords,
                    odometry_root=request.odometry_root,
                )
            )
    failures = [
        f"{summary.tracker_name}/{row.get('seq_name', '')}"
        for summary in summaries
        for row in summary.sequences
        if row.get("status") == "error"
    ]
    if failures:
        print("[tracker_eval] Failed sequence jobs:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    requests = make_requests(args)
    return execute_requests(args, requests)


if __name__ == "__main__":
    raise SystemExit(main())
