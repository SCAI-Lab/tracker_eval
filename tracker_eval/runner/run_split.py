# tracker_eval/runner/run_split.py

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import traceback

import numpy as np

from tracker_eval.data.jrdb_io import (
    list_sequence_jsons,
    sequence_name_from_json_filename,
    load_jrdb_labels_3d,
    load_jrdb_detections_3d,
)

from tracker_eval.runner.run_sequence import (
    SequenceRunStats,
    run_tracker_on_sequence,
    write_sequence_outputs,
    Tracker3D,
)

from tracker_eval.common.odometry_transform import (
    load_odometry_csv,
    transform_sequence_to_global,
    build_timestamps_by_frame_from_odometry,
)


# ----------------------------
# Summary datatypes
# ----------------------------

@dataclass
class SplitRunSummary:
    """
    Summary of a full split run (multiple sequences).
    """
    split_name: str
    tracker_name: str
    num_sequences: int
    num_frames_total: int
    warmup_steps: int
    sequences: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    io: Dict[str, Any]


# ----------------------------
# Helpers
# ----------------------------

def _safe_mkdir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _aggregate_sequence_stats(stats: Sequence[SequenceRunStats]) -> Dict[str, Any]:
    """
    Aggregates per-sequence runtime stats into overall metrics.

    In parallel mode we append "dummy" stats with timings set to 0 but num_frames populated,
    so counts are correct but timing fields stay 0.
    """
    if not stats:
        return {
            "num_sequences": 0,
            "num_frames_total": 0,
            "fps": {
                "mean_over_sequences": 0.0,
                "median_over_sequences": 0.0,
                "min_over_sequences": 0.0,
                "max_over_sequences": 0.0,
                "frame_weighted": 0.0,
            },
            "step_ms": {
                "mean_of_mean": 0.0,
                "median_of_mean": 0.0,
                "p90_of_mean": 0.0,
                "p99_of_mean": 0.0,
                "mean_of_p50": 0.0,
                "mean_of_p90": 0.0,
                "mean_of_p99": 0.0,
            },
        }

    fps_list = [float(s.fps) for s in stats]
    mean_ms_list = [float(s.mean_step_ms) for s in stats]
    p50_list = [float(s.p50_step_ms) for s in stats]
    p90_list = [float(s.p90_step_ms) for s in stats]
    p99_list = [float(s.p99_step_ms) for s in stats]
    frames_list = [int(s.num_frames) for s in stats]

    total_frames = int(sum(frames_list))
    total_time_s_approx = float(sum((f * ms) for f, ms in zip(frames_list, mean_ms_list)) / 1000.0)
    fps_weighted = float(total_frames / total_time_s_approx) if total_time_s_approx > 0 else 0.0

    agg = {
        "num_sequences": int(len(stats)),
        "num_frames_total": total_frames,
        "fps": {
            "mean_over_sequences": float(np.mean(fps_list)) if fps_list else 0.0,
            "median_over_sequences": float(np.median(fps_list)) if fps_list else 0.0,
            "min_over_sequences": float(np.min(fps_list)) if fps_list else 0.0,
            "max_over_sequences": float(np.max(fps_list)) if fps_list else 0.0,
            "frame_weighted": fps_weighted,
        },
        "step_ms": {
            "mean_of_mean": float(np.mean(mean_ms_list)) if mean_ms_list else 0.0,
            "median_of_mean": float(np.median(mean_ms_list)) if mean_ms_list else 0.0,
            "p90_of_mean": _percentile(mean_ms_list, 90),
            "p99_of_mean": _percentile(mean_ms_list, 99),
            "mean_of_p50": float(np.mean(p50_list)) if p50_list else 0.0,
            "mean_of_p90": float(np.mean(p90_list)) if p90_list else 0.0,
            "mean_of_p99": float(np.mean(p99_list)) if p99_list else 0.0,
        },
    }
    return agg


def _write_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    _safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Union[str, Path], rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    _safe_mkdir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------
# Parallel: tracker construction
# ----------------------------

def _build_tracker_from_spec(spec: Dict[str, Any]) -> Tracker3D:
    """
    Construct a fresh tracker instance from a picklable spec (created by CLI).
    This runs inside worker processes in parallel mode.
    """
    tracker_key = str(spec.get("tracker", "")).strip()
    cfg = spec.get("cfg", {}) or {}

    if tracker_key == "ab3dmot":
        from tracker_eval.trackers.ab3dmot_adapter import AB3DMOTAdapter, AB3DMOTConfig
        metrics = tuple(cfg.get("metrics", ["iou_3d", "dist_3d"]))
        tcfg = AB3DMOTConfig(
            max_age=int(cfg.get("max_age", 15)),
            min_hits=int(cfg.get("min_hits", 3)),
            thresh_3d_iou=float(cfg.get("thresh_3d_iou", 0.33)),
            thresh_3d_dist=float(cfg.get("thresh_3d_dist", 0.5)),
            metrics=metrics,  # type: ignore[arg-type]
            log_dir=cfg.get("log_dir", None),
        )
        return AB3DMOTAdapter(cfg=tcfg)

    if tracker_key == "simpletrack":
        from tracker_eval.trackers.simpletrack_adapter import SimpleTrackAdapter, SimpleTrackConfig
        tcfg = SimpleTrackConfig(config_path=str(cfg.get("config_path")))
        return SimpleTrackAdapter(cfg=tcfg)

    if tracker_key == "fastpoly":
        from tracker_eval.trackers.fastpoly_adapter import FastPolyAdapter, FastPolyConfig
        tcfg = FastPolyConfig(
            config=cfg.get("config", {}),
            seq_id=int(cfg.get("seq_id", 0)),
            has_velo=bool(cfg.get("has_velo", False)),
            is_key_frame=bool(cfg.get("is_key_frame", True)),
            force_class_label=cfg.get("force_class_label", None),
            use_numeric_frame_id=bool(cfg.get("use_numeric_frame_id", True)),
        )
        return FastPolyAdapter(cfg=tcfg)

    if tracker_key == "gnnpmb":
        from tracker_eval.trackers.gnnpmbtracker_adapter import GNNPMBAdapter, GNNPMBConfig
        tcfg = GNNPMBConfig(
            parameters_path=str(cfg.get("parameters_path")),
            classification=str(cfg.get("classification", "pedestrian")),
            use_nms=bool(cfg.get("use_nms", True)),
            fps=float(cfg.get("fps", 15.0)),
            giou_gating=float(cfg.get("giou_gating", -0.5)),
            ped_empty_meas_extract_thr=float(cfg.get("ped_empty_meas_extract_thr", 0.7)),
        )
        return GNNPMBAdapter(cfg=tcfg)

    if tracker_key == "cbmot":
        from tracker_eval.trackers.cbmot_adapter import CBMOTAdapter, CBMOTConfig
        tcfg = CBMOTConfig(
            hungarian=bool(cfg.get("hungarian", False)),
            max_age=int(cfg.get("max_age", 15)),
            min_hits=int(cfg.get("min_hits", 2)),
            score_decay=float(cfg.get("score_decay", 0.2)),
            active_th=float(cfg.get("active_th", 1.0)),
            deletion_th=float(cfg.get("deletion_th", 0.0)),
            detection_th=float(cfg.get("detection_th", 0.5)),
            score_update=cfg.get("score_update", None),
            model_path=cfg.get("model_path", None),
            fps=float(cfg.get("fps", 15.0)),
            track_class=cfg.get("track_class", "pedestrian"),
            export_score=bool(cfg.get("export_score", False)),
        )
        return CBMOTAdapter(cfg=tcfg)

    if tracker_key == "elptnet":
        from tracker_eval.trackers.elptnet_adapter import ELPTnetAdapter, ELPTnetConfig
        tcfg = ELPTnetConfig(
            cfg_file=str(cfg.get("cfg_file")),
            fps=float(cfg.get("fps", 15.0)),
            track_class=str(cfg.get("track_class", "pedestrian")),
            input_score=float(cfg.get("input_score", 0.5)),
            export_score=bool(cfg.get("export_score", False)),
            timestamp_mode=str(cfg.get("timestamp_mode", "frame_index")),
        )
        return ELPTnetAdapter(cfg=tcfg)

    if tracker_key == "headroom":
        from tracker_eval.trackers.headroom_adapter import HeadroomAdapter, HeadroomConfig
        from tracker_eval.trackers.headroom_kf_adapter import HeadroomTrackerKF, HeadroomKFConfig
        tcfg = HeadroomConfig(
        # tcfg = HeadroomKFConfig(
            fps=float(cfg.get("fps", 15.0)),

            T_reid_base_s=float(cfg.get("T_reid_base_s", 1.0)),
            T_reid_static_s=float(cfg.get("T_reid_static_s", 2.0)),

            score_floor=float(cfg.get("score_floor", 0.5)),
            score_power=float(cfg.get("score_power", 1.5)),
            tau_hit_s=float(cfg.get("tau_hit_s", 0.10)),
            tau_miss_s=float(cfg.get("tau_miss_s", 2.0)),
            theta_on=float(cfg.get("theta_on", 0.50)),
            min_hits=int(cfg.get("min_hits", 2)),

            T_out_min_s=float(cfg.get("T_out_min_s", 0.30)),
            T_out_max_s=float(cfg.get("T_out_max_s", 1.0)),
            T_out_gamma=float(cfg.get("T_out_gamma", 1.0)),

            dist_gate_m=float(cfg.get("dist_gate_m", 0.45)),
            z_gate_m=float(cfg.get("z_gate_m", 1.0)),
            assoc_topk=int(cfg.get("assoc_topk", 10)),
            assoc_iou_weight=float(cfg.get("assoc_iou_weight", 5.0)),

            v_static_thr_mps=float(cfg.get("v_static_thr_mps", 0.20)),
            jitter_thr_m=float(cfg.get("jitter_thr_m", 0.15)),
            static_window=int(cfg.get("static_window", 15)),

            gt_stride=int(cfg.get("gt_stride", 100000)),
            fp_offset=int(cfg.get("fp_offset", 10000000)),
        )
        return HeadroomAdapter(cfg=tcfg)
        # return HeadroomTrackerKF(cfg=tcfg)

    raise ValueError(f"Unknown tracker in spec: {tracker_key}")


def _run_one_sequence_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker entry (must be top-level for multiprocessing pickling).

    In parallel mode:
      - NO profiling/timing
      - NO per-frame stats files
      - we still return correct num_frames for aggregate counts.
    """
    seq_name = str(job["seq_name"])
    det_json_path = str(job["det_json_path"])
    out_kitti_txt = str(job["out_kitti_txt"]) if job.get("out_kitti_txt") else ""
    kitti_use_score = bool(job.get("kitti_use_score", True))
    warmup_steps = int(job.get("warmup_steps", 0))
    tracker_spec = job["tracker_spec"]
    split_name = str(job.get("split_name", ""))
    tracker_name = str(job.get("tracker_name", ""))

    labels_subdir = str(job.get("labels_subdir", "labels_3d"))
    split_root = str(job.get("split_root", ""))

    global_coords = bool(job.get("global_coords", False))
    odometry_root = Path(str(job.get("odometry_root", "")))

    try:
        # Build tracker instance inside this process
        tracker = _build_tracker_from_spec(tracker_spec)

        # Load GT if requested/available and tracker supports it
        gt_by_frame = None
        gt_json_path = ""
        if bool(job.get("use_gt_if_available", True)) and hasattr(tracker, "step_with_gt"):
            gt_json_path = str(Path(split_root) / labels_subdir / f"{seq_name}.json")
            if Path(gt_json_path).exists():
                gt_by_frame = load_jrdb_labels_3d(gt_json_path)
        # Load detections
        dets_by_frame = load_jrdb_detections_3d(str(det_json_path))

        timestamps_by_frame = None

        if global_coords:
            # Load odometry for this sequence
            odo_csv = odometry_root / split_name / "odometry" / f"{seq_name}.csv"
            pose_by_idx = load_odometry_csv(str(odo_csv))

            # Transform detections and GT into global coordinates
            dets_by_frame = transform_sequence_to_global(dets_by_frame, pose_by_idx)
            if gt_by_frame is not None:
                gt_by_frame = transform_sequence_to_global(dict(gt_by_frame), pose_by_idx)

            # Provide timestamps to tracker if it uses them
            timestamps_by_frame = build_timestamps_by_frame_from_odometry(str(odo_csv), dets_by_frame)

        tracks_by_frame, stats, frame_stats = run_tracker_on_sequence(
            seq_name=seq_name,
            detections_by_frame=dets_by_frame,
            tracker=tracker,
            frames=None,
            warmup_steps=warmup_steps,
            gt_by_frame=gt_by_frame,
            timestamps_by_frame=timestamps_by_frame,
            profile=False,
        )


        # IMPORTANT: provide correct num_frames for aggregate counts in parent
        num_frames = int(getattr(stats, "num_frames", 0))

        meta = {
            "split": split_name,
            "seq_name": seq_name,
            "tracker_name": tracker_name,
            "warmup_steps": int(warmup_steps),
            "detections_json": str(det_json_path),
            "gt_json": str(gt_json_path) if gt_by_frame is not None else "",
        }

        write_sequence_outputs(
            seq_name=seq_name,
            tracks_by_frame=tracks_by_frame,
            out_kitti_txt=out_kitti_txt if out_kitti_txt else None,
            kitti_use_score=kitti_use_score,
        )

        row = stats.as_dict()
        row.update(
            {
                "status": "ok",
                "num_frames": num_frames,  # ensure present (even if stats.as_dict already includes it)
                "detections_json": str(det_json_path),
                "out_kitti_txt": out_kitti_txt,
                "frame_stats_csv": "",
            }
        )
        return row

    except Exception as e:
        return {
            "seq_name": seq_name,
            "status": "error",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "detections_json": str(det_json_path),
            "out_kitti_txt": out_kitti_txt,
            "frame_stats_csv": "",
            "num_frames": 0,
            "fps": 0.0,
            "mean_step_ms": 0.0,
            "p50_step_ms": 0.0,
            "p90_step_ms": 0.0,
            "p99_step_ms": 0.0,
            "total_time_s": 0.0,
        }



# ----------------------------
# Public API
# ----------------------------

def run_tracker_on_split(
    *,
    split_root: Union[str, Path],
    split_name: str,

    # Sequential mode: pass tracker instance
    tracker: Optional[Tracker3D],
    # Parallel mode: pass tracker spec
    tracker_spec: Optional[Dict[str, Any]] = None,

    tracker_name: str,
    out_root: Union[str, Path],
    detections_subdir: str = "detections_3D",
    labels_subdir: str = "labels_3d",
    use_gt_if_available: bool = True,

    # Runtime options
    warmup_steps: int = 0,
    limit_sequences: Optional[int] = None,
    include_sequences: Optional[Sequence[str]] = None,
    exclude_sequences: Optional[Sequence[str]] = None,
    sort_sequences: bool = True,

    # Output options
    write_kitti_txt: bool = True,
    kitti_use_score: bool = True,
    tracker_subfolder: str = "data",
    skip_existing_kitti: bool = True,

    # Logging
    verbose: bool = True,

    # Parallel options
    parallel: bool = False,
    num_workers: int = 0,
    parallel_start_method: str = "spawn",

    # Global coordinate option
    global_coords: bool = False,
    odometry_root: Union[str, Path] = "",
) -> SplitRunSummary:
    """
    Sequential mode: original behavior (timing + per-frame stats written).
    Parallel mode: sequences processed concurrently (one tracker instance per sequence),
                   timing/per-frame profiling disabled.
    """
    split_root = Path(split_root)
    det_dir = split_root / detections_subdir
    if not det_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {det_dir}")
    
    global_coords = bool(global_coords)
    odometry_root = Path(odometry_root) if str(odometry_root) else Path()

    if global_coords and (not str(odometry_root)):
        raise ValueError("--global_coords requires --odometry_root to be set.")


    out_root = Path(out_root)
    tracker_dir = out_root / tracker_name / split_name
    out_kitti_dir = tracker_dir / tracker_subfolder

    _safe_mkdir(out_kitti_dir)

    write_frame_stats = (not parallel)
    out_frame_stats_dir = tracker_dir / "frame_stats"
    if write_frame_stats:
        _safe_mkdir(out_frame_stats_dir)

    # Discover sequences from detection JSON files
    det_files = list_sequence_jsons(str(det_dir))
    seqs = [sequence_name_from_json_filename(f) for f in det_files]

    # Filter include/exclude
    if include_sequences is not None:
        include_set = set(include_sequences)
        seqs = [s for s in seqs if s in include_set]
    if exclude_sequences is not None:
        exclude_set = set(exclude_sequences)
        seqs = [s for s in seqs if s not in exclude_set]

    if sort_sequences:
        seqs.sort()

    if limit_sequences is not None:
        seqs = seqs[: int(limit_sequences)]

    if verbose:
        print(f"[tracker_eval] Split '{split_name}': {len(seqs)} sequence(s) found in {det_dir}")
        if parallel:
            nw = num_workers if num_workers > 0 else (os.cpu_count() or 1)
            print(f"[tracker_eval] Parallel mode: workers={nw} start_method={parallel_start_method}")

    per_seq_rows: List[Dict[str, Any]] = []
    seq_stats: List[SequenceRunStats] = []

    # ------------------------------------------------------------
    # Sequential path
    # ------------------------------------------------------------
    if not parallel:
        if tracker is None:
            raise ValueError("Sequential mode requires a tracker instance (tracker=...).")

        for idx, seq_name in enumerate(seqs):
            det_json_path = det_dir / f"{seq_name}.json"
            if not det_json_path.exists():
                if verbose:
                    print(f"[tracker_eval] WARNING: missing detections file: {det_json_path} (skipping)")
                continue

            out_kitti_txt = out_kitti_dir / f"{seq_name}.txt"

            if skip_existing_kitti and write_kitti_txt and out_kitti_txt.exists():
                if verbose:
                    print(f"[tracker_eval] ({idx+1}/{len(seqs)}) {seq_name}: output exists, skipping")
                per_seq_rows.append(
                    {
                        "seq_name": seq_name,
                        "status": "skipped_existing",
                        "out_kitti_txt": str(out_kitti_txt),
                        "frame_stats_csv": "",
                    }
                )
                continue

            if verbose:
                print(f"[tracker_eval] ({idx+1}/{len(seqs)}) Running {seq_name} ...")

            gt_by_frame = None
            gt_json_path = split_root / labels_subdir / f"{seq_name}.json"
            if use_gt_if_available and hasattr(tracker, "step_with_gt"):
                if gt_json_path.exists():
                    gt_by_frame = load_jrdb_labels_3d(str(gt_json_path))
                else:
                    if verbose:
                        print(f"[tracker_eval]   NOTE: GT not found: {gt_json_path} (running without GT)")

            # Load detections explicitly (so we can optionally transform)
            dets_by_frame = load_jrdb_detections_3d(str(det_json_path))

            timestamps_by_frame = None
            if global_coords:
                odo_csv = Path(odometry_root) / str(split_name) / "odometry" / f"{seq_name}.csv"
                if not odo_csv.exists():
                    raise FileNotFoundError(f"Missing odometry CSV for {seq_name}: {odo_csv}")

                pose_by_idx = load_odometry_csv(str(odo_csv))

                # Transform detections and GT into global coordinates
                dets_by_frame = transform_sequence_to_global(dets_by_frame, pose_by_idx)
                if gt_by_frame is not None:
                    gt_by_frame = transform_sequence_to_global(dict(gt_by_frame), pose_by_idx)

                # Provide timestamps (seconds) derived from odometry CSV
                timestamps_by_frame = build_timestamps_by_frame_from_odometry(str(odo_csv), dets_by_frame)

            # Run sequence (NOTE: frames=None is OK, because run_tracker_on_sequence unions det+GT keys)
            tracks_by_frame, stats, frame_stats = run_tracker_on_sequence(
                seq_name=seq_name,
                detections_by_frame=dets_by_frame,
                tracker=tracker,
                frames=None,
                warmup_steps=warmup_steps,
                gt_by_frame=gt_by_frame,
                timestamps_by_frame=timestamps_by_frame,
                profile=True,
            )

            meta = {
                "split": split_name,
                "seq_name": seq_name,
                "tracker_name": tracker_name,
                "warmup_steps": int(warmup_steps),
                "detections_json": str(det_json_path),
                "gt_json": str(gt_json_path) if gt_by_frame is not None else "",
            }

            write_sequence_outputs(
                seq_name=seq_name,
                tracks_by_frame=tracks_by_frame,
                out_kitti_txt=str(out_kitti_txt) if write_kitti_txt else None,
                kitti_use_score=kitti_use_score,
            )

            frame_stats_path = out_frame_stats_dir / f"{seq_name}.csv"
            fieldnames = [
                "frame_id", "frame_idx", "timestamp_s",
                "step_ms", "fps_inst",
                "num_det_in", "num_tracks_out", "num_gt",
                "is_warmup",
            ]
            _write_csv(frame_stats_path, frame_stats, fieldnames=fieldnames)

            seq_stats.append(stats)

            row = stats.as_dict()
            row.update(
                {
                    "status": "ok",
                    "detections_json": str(det_json_path),
                    "out_kitti_txt": str(out_kitti_txt) if write_kitti_txt else "",
                    "frame_stats_csv": str(frame_stats_path),
                }
            )
            per_seq_rows.append(row)

            if verbose:
                print(
                    f"[tracker_eval]   {seq_name}: fps={stats.fps:.2f}, "
                    f"mean={stats.mean_step_ms:.2f}ms, p90={stats.p90_step_ms:.2f}ms, p99={stats.p99_step_ms:.2f}ms"
                )

    # ------------------------------------------------------------
    # Parallel path
    # ------------------------------------------------------------
    else:
        if tracker_spec is None:
            raise ValueError("Parallel mode requires tracker_spec (picklable tracker configuration).")

        jobs: List[Dict[str, Any]] = []
        skipped = 0

        for seq_name in seqs:
            det_json_path = det_dir / f"{seq_name}.json"
            if not det_json_path.exists():
                if verbose:
                    print(f"[tracker_eval] WARNING: missing detections file: {det_json_path} (skipping)")
                continue

            out_kitti_txt = out_kitti_dir / f"{seq_name}.txt"

            if skip_existing_kitti and write_kitti_txt and out_kitti_txt.exists():
                skipped += 1
                per_seq_rows.append(
                    {
                        "seq_name": seq_name,
                        "status": "skipped_existing",
                        "out_kitti_txt": str(out_kitti_txt),
                        "frame_stats_csv": "",
                    }
                )
                continue

            jobs.append(
                {
                    "seq_name": seq_name,
                    "det_json_path": str(det_json_path),
                    "out_kitti_txt": str(out_kitti_txt) if write_kitti_txt else "",
                    "kitti_use_score": bool(kitti_use_score),
                    "warmup_steps": int(warmup_steps),

                    "tracker_spec": tracker_spec,
                    "split_name": str(split_name),
                    "tracker_name": str(tracker_name),

                    "split_root": str(split_root),
                    "labels_subdir": str(labels_subdir),
                    "use_gt_if_available": bool(use_gt_if_available),

                    "global_coords": bool(global_coords),
                    "odometry_root": str(odometry_root),

                }
            )

        if verbose:
            print(f"[tracker_eval] Parallel: scheduling {len(jobs)} job(s) (skipped={skipped}).")

        nw = num_workers if num_workers > 0 else (os.cpu_count() or 1)
        import concurrent.futures
        import multiprocessing as mp
        ctx = mp.get_context(parallel_start_method)

        done = 0
        total = len(jobs)

        with concurrent.futures.ProcessPoolExecutor(max_workers=nw, mp_context=ctx) as ex:
            futs = [ex.submit(_run_one_sequence_worker, job) for job in jobs]

            try:
                for fut in concurrent.futures.as_completed(futs):
                    row = fut.result()
                    if verbose and row.get("status") != "ok":
                        print(f"[tracker_eval]   ERROR: {row.get('seq_name','')} -> {row.get('error','')}")
                        tb = row.get("traceback", "")
                        if tb:
                            print(tb)

                    per_seq_rows.append(row)

                    done += 1

                    # In parallel mode: keep counts correct, timings at 0
                    if row.get("status") == "ok":
                        seq_stats.append(
                            SequenceRunStats(
                                seq_name=str(row.get("seq_name", "")),
                                num_frames=int(row.get("num_frames", 0)),
                                total_time_s=0.0,
                                fps=0.0,
                                mean_step_ms=0.0,
                                p50_step_ms=0.0,
                                p90_step_ms=0.0,
                                p99_step_ms=0.0,
                            )
                        )

                    if verbose:
                        print(f"[tracker_eval]   Done: {row.get('seq_name', '')} ({done}/{total})")

            except KeyboardInterrupt:
                if verbose:
                    print("\n[tracker_eval] Ctrl+C received. Cancelling and terminating workers...")

                # Cancel futures that haven't started
                for f in futs:
                    f.cancel()

                # Stop accepting work and don't wait
                ex.shutdown(wait=False, cancel_futures=True)

                # Terminate worker processes (private API but effective)
                procs = getattr(ex, "_processes", {})
                for p in list(procs.values()):
                    try:
                        if p.is_alive():
                            p.terminate()
                    except Exception:
                        pass

                # Escalate if needed
                time.sleep(0.2)
                for p in list(procs.values()):
                    try:
                        if p.is_alive():
                            p.kill()
                    except Exception:
                        pass

                raise

    # Build summary
    agg = _aggregate_sequence_stats(seq_stats)
    num_frames_total = int(agg.get("num_frames_total", 0))

    summary = SplitRunSummary(
        split_name=str(split_name),
        tracker_name=str(tracker_name),
        # Keep this as the total number of sequences discovered (historical behavior).
        num_sequences=int(len(seqs)),
        num_frames_total=num_frames_total,
        warmup_steps=int(warmup_steps),
        sequences=per_seq_rows,
        aggregate=agg,
        io={
            "split_root": str(split_root),
            "detections_dir": str(det_dir),
            "out_root": str(out_root),
            "tracker_dir": str(tracker_dir),
            "kitti_dir": str(out_kitti_dir),
        },
    )

    # Write summary artifacts
    summary_json_path = tracker_dir / "runtime_summary.json"
    _write_json(summary_json_path, asdict(summary))

    csv_path = tracker_dir / "runtime_summary.csv"
    fieldnames = [
        "seq_name",
        "status",
        "num_frames",
        "fps",
        "mean_step_ms",
        "p50_step_ms",
        "p90_step_ms",
        "p99_step_ms",
        "detections_json",
        "out_kitti_txt",
        "frame_stats_csv",
    ]
    _write_csv(csv_path, per_seq_rows, fieldnames=fieldnames)

    if verbose:
        print(f"[tracker_eval] Wrote summary: {summary_json_path}")
        print(f"[tracker_eval] Wrote CSV:     {csv_path}")
        if not parallel and seq_stats:
            print(
                f"[tracker_eval] Aggregate fps (weighted): {agg['fps']['frame_weighted']:.2f} | "
                f"mean step ms (mean-of-mean): {agg['step_ms']['mean_of_mean']:.2f}"
            )
        elif parallel:
            print("[tracker_eval] Parallel mode: timing/per-frame profiling disabled (metrics are 0).")
        else:
            print("[tracker_eval] No sequences were run (all skipped or none found).")

    return summary
