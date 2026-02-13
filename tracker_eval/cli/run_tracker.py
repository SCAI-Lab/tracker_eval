# tracker_eval/cli/run_tracker.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import yaml

from tracker_eval.runner.run_split import run_tracker_on_split

from tracker_eval.trackers.ab3dmot_adapter import AB3DMOTAdapter, AB3DMOTConfig
from tracker_eval.trackers.simpletrack_adapter import SimpleTrackAdapter, SimpleTrackConfig
from tracker_eval.trackers.fastpoly_adapter import FastPolyAdapter, FastPolyConfig
from tracker_eval.trackers.gnnpmbtracker_adapter import GNNPMBAdapter, GNNPMBConfig
from tracker_eval.trackers.cbmot_adapter import CBMOTAdapter, CBMOTConfig
from tracker_eval.trackers.headroom_adapter import HeadroomAdapter, HeadroomConfig


def _parse_list_arg(xs: Optional[List[str]]) -> Optional[List[str]]:
    """
    Accept repeated flags and/or comma-separated values.

    Example:
      --include_sequences a b,c --include_sequences d
    => ["a", "b", "c", "d"]
    """
    if xs is None:
        return None
    out: List[str] = []
    for x in xs:
        if x is None:
            continue
        parts = [p.strip() for p in str(x).split(",") if p.strip()]
        out.extend(parts)
    return out or None


def _load_variants_from_manifest(path: str) -> List[str]:
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        m = json.load(f)
    vs = []
    for v in m.get("variants", []):
        name = str(v.get("name", "")).strip()
        if name:
            vs.append(name)
    if not vs:
        raise ValueError(f"No variants found in manifest: {p}")
    return vs


def _normalize_split_names(split_roots: Sequence[str], split_names: Optional[Sequence[str]]) -> List[str]:
    roots = [Path(r) for r in split_roots]
    if split_names is None or len(split_names) == 0:
        return [p.name for p in roots]
    names = list(split_names)
    if len(names) == 1 and len(roots) > 1:
        raise ValueError(
            f"--split_name provided once ('{names[0]}') but multiple --split_root were given "
            f"({len(roots)}). Provide one --split_name per split_root or omit --split_name."
        )
    if len(names) != len(roots):
        raise ValueError(
            f"Number of --split_name ({len(names)}) must match number of --split_root ({len(roots)})."
        )
    return [str(n) for n in names]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker_eval.run_tracker",
        description=(
            "Run a 3D MOT tracker on one or more JRDB splits (Jetson runtime eval + export KITTI txt for JRDB toolkit)."
        ),
    )

    # IO
    p.add_argument(
        "--split_root",
        type=str,
        nargs="+",
        required=True,
        help="One or more split roots, e.g. /mnt/nvme/JRDB_track/train /mnt/nvme/JRDB_track/test",
    )
    p.add_argument(
        "--split_name",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit split name(s). Must match number of split_root if provided.",
    )
    p.add_argument("--out_root", type=str, required=True, help="Output root directory for tracker outputs")

    p.add_argument(
        "--detections_subdir",
        type=str,
        default="detections_3D",
        help="Detections subfolder name under split_root (default: detections_3D)",
    )

    p.add_argument(
        "--labels_subdir",
        type=str,
        default="labels_3d",
        help="GT labels subfolder name under split_root (default: labels_3d). Used by headroom.",
    )

    # Tracker selection
    p.add_argument(
        "--tracker",
        type=str,
        default="ab3dmot",
        choices=["ab3dmot", "simpletrack", "fastpoly", "gnnpmb", "cbmot", "elptnet", "headroom"],
        help="Tracker to run.",
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

    # Output options
    p.add_argument(
        "--tracker_subfolder",
        type=str,
        default="data",
        help="Subfolder name under <out_root>/<tracker_name>/<split_name>/ for KITTI txt (default: data)",
    )
    p.add_argument(
        "--no_kitti_score",
        action="store_true",
        help="Do not write score column in KITTI txt (default writes score column).",
    )

    # Parallel options
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
        help="Number of worker processes for --parallel. 0 means auto (cpu_count).",
    )
    p.add_argument(
        "--parallel_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for --parallel (default: spawn).",
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

    # AB3DMOT params
    g = p.add_argument_group("AB3DMOT parameters")
    g.add_argument("--ab3dmot_max_age", type=int, default=15)
    g.add_argument("--ab3dmot_min_hits", type=int, default=3)
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
        default="/home/scai/trackers/SimpleTrack/configs/nu_configs/giou.yaml",
        help="Path to SimpleTrack YAML config "
        "(e.g. /home/scai/trackers/SimpleTrack/configs/nu_configs/giou.yaml)",
    )

    # FastPoly params
    g3 = p.add_argument_group("FastPoly parameters")
    g3.add_argument(
        "--fastpoly_config",
        type=str,
        default="/home/scai/trackers/FastPoly/config/nusc_config.yaml",
        help="Path to FastPoly YAML config (e.g. /home/scai/trackers/FastPoly/config/nusc_config.yaml)",
    )

    # GNNPMB params
    g4 = p.add_argument_group("GNNPMB parameters")
    g4.add_argument(
        "--gnnpmb_parameters_path",
        type=str,
        default="/home/scai/trackers/GnnPmbTracker/configs/gnnpmb_parameters.json",
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

    # CBMOT params
    g5 = p.add_argument_group("CBMOT parameters")
    g5.add_argument("--cbmot_hungarian", action="store_true")
    g5.add_argument("--cbmot_max_age", type=int, default=15)
    g5.add_argument("--cbmot_min_hits", type=int, default=2)
    g5.add_argument("--cbmot_score_decay", type=float, default=0.2)
    g5.add_argument("--cbmot_active_th", type=float, default=1.0)
    g5.add_argument("--cbmot_deletion_th", type=float, default=0.0)
    g5.add_argument("--cbmot_detection_th", type=float, default=0.5)
    g5.add_argument("--cbmot_score_update", type=str, default=None)
    g5.add_argument("--cbmot_model_path", type=str, default=None)
    g5.add_argument("--cbmot_fps", type=float, default=15.0)
    g5.add_argument("--cbmot_track_class", type=str, default="pedestrian")
    g5.add_argument("--cbmot_export_score", action="store_true")

    # ELPTnet params
    g6 = p.add_argument_group("ELPTnet parameters")
    g6.add_argument("--elptnet_cfg_file", type=str, required=False, default="/home/scai/trackers/ELPTNet/jrdb.yaml", help="Path to ELPTnet jrdb.yaml")
    g6.add_argument("--elptnet_fps", type=float, default=15.0)
    g6.add_argument("--elptnet_track_class", type=str, default="pedestrian")
    g6.add_argument("--elptnet_input_score", type=float, default=0.5)
    g6.add_argument("--elptnet_export_score", action="store_true")
    g6.add_argument(
        "--elptnet_timestamp_mode",
        type=str,
        default="frame_index",
        choices=["frame_index", "seconds"],
    )

    # Headroom params
    g7 = p.add_argument_group("Headroom parameters")
    g7.add_argument("--headroom_fps", type=float, default=15.0)

    g7.add_argument("--headroom_T_reid_base_s", type=float, default=1.0)
    g7.add_argument("--headroom_T_reid_static_s", type=float, default=2.0)

    g7.add_argument("--headroom_score_floor", type=float, default=0.5)
    g7.add_argument("--headroom_score_power", type=float, default=1.5)
    g7.add_argument("--headroom_tau_hit_s", type=float, default=0.10)
    g7.add_argument("--headroom_tau_miss_s", type=float, default=2.0)
    g7.add_argument("--headroom_theta_on", type=float, default=0.50)
    g7.add_argument("--headroom_min_hits", type=int, default=2)

    g7.add_argument("--headroom_T_out_min_s", type=float, default=0.30)
    g7.add_argument("--headroom_T_out_max_s", type=float, default=1.0)
    g7.add_argument("--headroom_T_out_gamma", type=float, default=1.0)

    g7.add_argument("--headroom_dist_gate_m", type=float, default=0.45)
    g7.add_argument("--headroom_z_gate_m", type=float, default=1.0)
    g7.add_argument("--headroom_assoc_topk", type=int, default=10)
    g7.add_argument("--headroom_assoc_iou_weight", type=float, default=5.0)

    g7.add_argument("--headroom_v_static_thr_mps", type=float, default=0.20)
    g7.add_argument("--headroom_jitter_thr_m", type=float, default=0.15)
    g7.add_argument("--headroom_static_window", type=int, default=15)

    g7.add_argument("--headroom_gt_stride", type=int, default=100000)
    g7.add_argument("--headroom_fp_offset", type=int, default=10000000)

    # Misc
    p.add_argument("--quiet", action="store_true", help="Reduce printing")

    return p


def _build_tracker_and_name(args: argparse.Namespace) -> Tuple[object, str]:
    """
    Returns (tracker_instance, tracker_name).
    Sequential mode uses this (single tracker instance reused across sequences, reset per sequence).
    """
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
        return tracker, tracker.name

    if args.tracker == "simpletrack":
        if args.simpletrack_config is None:
            raise ValueError("--simpletrack_config is required when --tracker simpletrack")
        cfg = SimpleTrackConfig(config_path=str(args.simpletrack_config))
        tracker = SimpleTrackAdapter(cfg=cfg)
        return tracker, tracker.name

    if args.tracker == "fastpoly":
        if args.fastpoly_config is None:
            raise ValueError("--fastpoly_config is required when --tracker fastpoly")
        with open(args.fastpoly_config, "r", encoding="utf-8") as f:
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
        return tracker, tracker.name

    if args.tracker == "gnnpmb":
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
        return tracker, tracker.name

    if args.tracker == "cbmot":
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
        return tracker, tracker.name

    if args.tracker == "elptnet":
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
        return tracker, tracker.name

    if args.tracker == "headroom":
        cfg = HeadroomConfig(
            fps=float(args.headroom_fps),

            T_reid_base_s=float(args.headroom_T_reid_base_s),
            T_reid_static_s=float(args.headroom_T_reid_static_s),

            score_floor=float(args.headroom_score_floor),
            score_power=float(args.headroom_score_power),
            tau_hit_s=float(args.headroom_tau_hit_s),
            tau_miss_s=float(args.headroom_tau_miss_s),
            theta_on=float(args.headroom_theta_on),
            min_hits=int(args.headroom_min_hits),

            T_out_min_s=float(args.headroom_T_out_min_s),
            T_out_max_s=float(args.headroom_T_out_max_s),
            T_out_gamma=float(args.headroom_T_out_gamma),

            dist_gate_m=float(args.headroom_dist_gate_m),
            z_gate_m=float(args.headroom_z_gate_m),
            assoc_topk=int(args.headroom_assoc_topk),
            assoc_iou_weight=float(args.headroom_assoc_iou_weight),

            v_static_thr_mps=float(args.headroom_v_static_thr_mps),
            jitter_thr_m=float(args.headroom_jitter_thr_m),
            static_window=int(args.headroom_static_window),

            gt_stride=int(args.headroom_gt_stride),
            fp_offset=int(args.headroom_fp_offset),
        )
        tracker = HeadroomAdapter(cfg=cfg)
        return tracker, tracker.name

    raise ValueError(f"Unsupported tracker: {args.tracker}")


def _build_tracker_spec(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build a picklable tracker spec used by parallel workers to create a fresh tracker instance
    inside each process.
    """
    spec: Dict[str, Any] = {"tracker": str(args.tracker)}

    if args.tracker == "ab3dmot":
        metrics = [m.strip() for m in str(args.ab3dmot_metrics).split(",") if m.strip()] or ["iou_3d", "dist_3d"]
        spec["cfg"] = {
            "max_age": int(args.ab3dmot_max_age),
            "min_hits": int(args.ab3dmot_min_hits),
            "thresh_3d_iou": float(args.ab3dmot_thresh_iou),
            "thresh_3d_dist": float(args.ab3dmot_thresh_dist),
            "metrics": list(metrics),
            "log_dir": args.ab3dmot_log_dir,
        }
        return spec

    if args.tracker == "simpletrack":
        spec["cfg"] = {"config_path": str(args.simpletrack_config)}
        return spec

    if args.tracker == "fastpoly":
        if args.fastpoly_config is None:
            raise ValueError("--fastpoly_config is required when --tracker fastpoly")
        with open(args.fastpoly_config, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        spec["cfg"] = {
            "config": cfg_dict,
            "seq_id": 0,
            "has_velo": False,
            "is_key_frame": True,
            "use_numeric_frame_id": True,
            "force_class_label": None,
        }
        return spec

    if args.tracker == "gnnpmb":
        spec["cfg"] = {
            "parameters_path": str(args.gnnpmb_parameters_path),
            "classification": str(args.gnnpmb_classification),
            "use_nms": not bool(args.gnnpmb_no_nms),
            "fps": float(args.gnnpmb_fps),
            "giou_gating": float(args.gnnpmb_giou_gating),
            "ped_empty_meas_extract_thr": float(args.gnnpmb_ped_empty_meas_extract_thr),
        }
        return spec

    if args.tracker == "cbmot":
        spec["cfg"] = {
            "hungarian": bool(args.cbmot_hungarian),
            "max_age": int(args.cbmot_max_age),
            "min_hits": int(args.cbmot_min_hits),
            "score_decay": float(args.cbmot_score_decay),
            "active_th": float(args.cbmot_active_th),
            "deletion_th": float(args.cbmot_deletion_th),
            "detection_th": float(args.cbmot_detection_th),
            "score_update": args.cbmot_score_update if args.cbmot_score_update not in ("", "none", "None") else None,
            "model_path": args.cbmot_model_path,
            "fps": float(args.cbmot_fps),
            "track_class": str(args.cbmot_track_class) if args.cbmot_track_class not in ("", "none", "None") else None,
            "export_score": bool(args.cbmot_export_score),
        }
        return spec

    if args.tracker == "elptnet":
        spec["cfg"] = {
            "cfg_file": str(args.elptnet_cfg_file),
            "fps": float(args.elptnet_fps),
            "track_class": str(args.elptnet_track_class),
            "input_score": float(args.elptnet_input_score),
            "export_score": bool(args.elptnet_export_score),
            "timestamp_mode": str(args.elptnet_timestamp_mode),
        }
        return spec

    if args.tracker == "headroom":
        spec["cfg"] = {
            "fps": float(args.headroom_fps),

            "T_reid_base_s": float(args.headroom_T_reid_base_s),
            "T_reid_static_s": float(args.headroom_T_reid_static_s),

            "score_floor": float(args.headroom_score_floor),
            "score_power": float(args.headroom_score_power),
            "tau_hit_s": float(args.headroom_tau_hit_s),
            "tau_miss_s": float(args.headroom_tau_miss_s),
            "theta_on": float(args.headroom_theta_on),
            "min_hits": int(args.headroom_min_hits),

            "T_out_min_s": float(args.headroom_T_out_min_s),
            "T_out_max_s": float(args.headroom_T_out_max_s),
            "T_out_gamma": float(args.headroom_T_out_gamma),

            "dist_gate_m": float(args.headroom_dist_gate_m),
            "z_gate_m": float(args.headroom_z_gate_m),
            "assoc_topk": int(args.headroom_assoc_topk),
            "assoc_iou_weight": float(args.headroom_assoc_iou_weight),

            "v_static_thr_mps": float(args.headroom_v_static_thr_mps),
            "jitter_thr_m": float(args.headroom_jitter_thr_m),
            "static_window": int(args.headroom_static_window),

            "gt_stride": int(args.headroom_gt_stride),
            "fp_offset": int(args.headroom_fp_offset),
        }
        return spec

    raise ValueError(f"Unsupported tracker for spec: {args.tracker}")


def _tracker_base_name_from_args(args: argparse.Namespace) -> str:
    """
    Base output name for the tracker folder, matching adapter defaults.
    """
    # In your adapters you consistently use these names, and they match choices.
    # Keeping this simple avoids instantiating heavy trackers in parallel mode.
    return str(args.tracker)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    split_roots: List[str] = [str(s) for s in args.split_root]
    split_names: List[str] = _normalize_split_names(split_roots, args.split_name)

    include = _parse_list_arg(args.include_sequences)
    exclude = _parse_list_arg(args.exclude_sequences)

    variants = _parse_list_arg(args.variants)
    exclude_variants = set(_parse_list_arg(args.exclude_variants) or [])

    if variants is None and args.variants_from_manifest is not None:
        variants = _load_variants_from_manifest(args.variants_from_manifest)

    if variants is not None:
        variants = [v for v in variants if v not in exclude_variants]
        if not variants:
            raise ValueError("No variants selected after exclude_variants filtering.")
    variant_list = variants or [None]

    parallel = bool(args.parallel)
    num_workers = int(args.num_workers)
    start_method = str(args.parallel_start_method)

    # Build either:
    #  - sequential tracker instance + base name
    #  - parallel tracker spec + base name
    if parallel:
        tracker_spec = _build_tracker_spec(args)
        tracker_base_name = _tracker_base_name_from_args(args)
        tracker_obj = None
    else:
        tracker_obj, tracker_base_name = _build_tracker_and_name(args)
        tracker_spec = None

    all_summaries = []
    for split_root, split_name in zip(split_roots, split_names):
        for v in variant_list:
            if v is None:
                det_subdir = str(args.detections_subdir)
                tracker_name = tracker_base_name
            else:
                if args.variants_subdir is None:
                    raise ValueError("--variants_subdir is required when using --variants / --variants_from_manifest")
                det_subdir = str(Path(args.variants_subdir) / str(v))
                tracker_name = f"{tracker_base_name}__{v}"

            if not args.quiet:
                print("")
                print(f"[tracker_eval] Split={split_name} | tracker={tracker_name} | dets={det_subdir}")
                if parallel:
                    print(f"[tracker_eval] Parallel: workers={num_workers or 'auto'} | start_method={start_method}")

            summary = run_tracker_on_split(
                split_root=str(split_root),
                split_name=str(split_name),

                tracker=tracker_obj,                 # sequential only
                tracker_spec=tracker_spec,           # parallel only

                tracker_name=tracker_name,
                out_root=str(args.out_root),
                detections_subdir=det_subdir,
                labels_subdir=str(args.labels_subdir),

                warmup_steps=int(args.warmup_steps),
                limit_sequences=int(args.limit_sequences) if args.limit_sequences is not None else None,
                include_sequences=include,
                exclude_sequences=exclude,

                write_kitti_txt=True,
                kitti_use_score=not bool(args.no_kitti_score),
                tracker_subfolder=str(args.tracker_subfolder),
                skip_existing_kitti=not bool(args.no_skip_existing),
                verbose=not bool(args.quiet),

                parallel=parallel,
                num_workers=num_workers,
                parallel_start_method=start_method,
            )
            all_summaries.append(summary)

            if not args.quiet:
                agg = summary.aggregate
                fps_w = agg.get("fps", {}).get("frame_weighted", 0.0)
                mean_ms = agg.get("step_ms", {}).get("mean_of_mean", 0.0)
                nseq = agg.get("num_sequences", 0)
                nframes = agg.get("num_frames_total", 0)
                tracker_dir = summary.io.get("tracker_dir", "")
                kitti_dir = summary.io.get("kitti_dir", "")
                print(f"[tracker_eval] Done: {tracker_name} on {split_name}")
                print(f"[tracker_eval] Sequences: {nseq} | Frames: {nframes}")
                if not parallel:
                    print(f"[tracker_eval] FPS(w): {fps_w:.2f} | Mean step: {mean_ms:.2f} ms")
                else:
                    print("[tracker_eval] (parallel mode) Timing/profiling disabled.")
                print(f"[tracker_eval] Outputs: {tracker_dir}")
                print(f"[tracker_eval] KITTI txt: {kitti_dir}")

    if (not args.quiet) and len(all_summaries) > 1:
        print("")
        print(f"[tracker_eval] Completed {len(all_summaries)} runs for tracker base '{tracker_base_name}'.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
