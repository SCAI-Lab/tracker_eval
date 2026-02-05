# tracker_eval/runner/run_split.py

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tracker_eval.common.types import frame_sort_key
from tracker_eval.data.jrdb_io import list_sequence_jsons, sequence_name_from_json_filename
from tracker_eval.runner.run_sequence import (
    SequenceRunStats,
    run_tracker_from_detections_json,
    write_sequence_outputs,
    Tracker3D,
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
    sequences: List[Dict[str, Any]]          # list of SequenceRunStats dicts (+ paths)
    aggregate: Dict[str, Any]               # aggregate runtime stats
    io: Dict[str, Any]                      # important paths used


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

    Provides both:
      - simple mean/median over sequences
      - frame-weighted FPS (based on frames and mean step times)
      - pooled latency percentiles (approx) by concatenating per-frame latencies not available here,
        so we approximate using per-seq pXX and means (we report sequence-level percentiles instead).

    Note: We only have per-sequence percentiles (p50/p90/p99), not per-frame samples.
    So we report:
      - distribution of per-seq p50/p90/p99
      - distribution of per-seq mean_step_ms
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

    # Frame-weighted FPS:
    # We approximate total effective time by summing (num_frames * mean_step_ms).
    # This slightly differs from per-sequence warmup exclusions and internal effective_time_s,
    # but is stable and comparable as a split-level summary.
    total_frames = int(sum(frames_list))
    total_time_s_approx = float(sum((f * ms) for f, ms in zip(frames_list, mean_ms_list)) / 1000.0)
    fps_weighted = float(total_frames / total_time_s_approx) if total_time_s_approx > 0 else 0.0

    agg = {
        "num_sequences": int(len(stats)),
        "num_frames_total": total_frames,
        "fps": {
            "mean_over_sequences": float(np.mean(fps_list)),
            "median_over_sequences": float(np.median(fps_list)),
            "min_over_sequences": float(np.min(fps_list)),
            "max_over_sequences": float(np.max(fps_list)),
            "frame_weighted": fps_weighted,
        },
        "step_ms": {
            "mean_of_mean": float(np.mean(mean_ms_list)),
            "median_of_mean": float(np.median(mean_ms_list)),
            "p90_of_mean": _percentile(mean_ms_list, 90),
            "p99_of_mean": _percentile(mean_ms_list, 99),
            "mean_of_p50": float(np.mean(p50_list)),
            "mean_of_p90": float(np.mean(p90_list)),
            "mean_of_p99": float(np.mean(p99_list)),
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
# Public API
# ----------------------------

def run_tracker_on_split(
    *,
    split_root: Union[str, Path],
    split_name: str,
    tracker: Tracker3D,
    tracker_name: str,
    out_root: Union[str, Path],
    detections_subdir: str = "detections_3D",
    # Runtime options
    warmup_steps: int = 0,
    limit_sequences: Optional[int] = None,
    include_sequences: Optional[Sequence[str]] = None,
    exclude_sequences: Optional[Sequence[str]] = None,
    sort_sequences: bool = True,
    # Output options
    write_kitti_txt: bool = True,
    write_tracks_json: bool = False,
    kitti_use_score: bool = True,
    tracker_subfolder: str = "data",
    skip_existing_kitti: bool = True,
    # Logging
    verbose: bool = True,
) -> SplitRunSummary:
    """
    Run a tracker over all sequences in a split (e.g., JRDB_track/test or JRDB_track/train_val),
    using precomputed detections JSON files. Measures runtime stats and writes outputs.

    Output layout (JRDB toolkit compatible):
      <out_root>/<tracker_name>/<tracker_subfolder>/<seq>.txt

    Additionally writes:
      <out_root>/<tracker_name>/runtime_summary.json
      <out_root>/<tracker_name>/runtime_summary.csv

    Parameters
    ----------
    split_root:
        Path to split folder (e.g., /mnt/nvme/JRDB_track/test).
    split_name:
        Name used in summary (e.g., "test", "train_val").
    tracker:
        Tracker instance implementing Tracker3D (AB3DMOT adapter will plug in here).
    tracker_name:
        Used for output folder naming.
    out_root:
        Root output folder.
    detections_subdir:
        Usually "detections_3D" inside split_root.
    warmup_steps:
        Steps excluded from timing (per sequence).
    limit_sequences:
        If set, only runs the first N sequences after filtering/sorting.
    include_sequences / exclude_sequences:
        Optional allow/deny list by sequence name.
    write_kitti_txt:
        If True, writes KITTI/JRDB .txt per sequence (recommended).
    write_tracks_json:
        If True, writes internal JSON per sequence (debug/transfer).
    skip_existing_kitti:
        If True, skip running a sequence if the KITTI output already exists.
    """
    split_root = Path(split_root)
    det_dir = split_root / detections_subdir
    if not det_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {det_dir}")

    out_root = Path(out_root)
    tracker_dir = out_root / tracker_name
    out_kitti_dir = tracker_dir / tracker_subfolder
    out_json_dir = tracker_dir / "tracks_json"

    _safe_mkdir(out_kitti_dir)
    if write_tracks_json:
        _safe_mkdir(out_json_dir)

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

    per_seq_rows: List[Dict[str, Any]] = []
    seq_stats: List[SequenceRunStats] = []

    for idx, seq_name in enumerate(seqs):
        det_json_path = det_dir / f"{seq_name}.json"
        if not det_json_path.exists():
            # should not happen because we discovered from dir listing, but keep robust
            if verbose:
                print(f"[tracker_eval] WARNING: missing detections file: {det_json_path} (skipping)")
            continue

        out_kitti_txt = out_kitti_dir / f"{seq_name}.txt"
        out_tracks_json = out_json_dir / f"{seq_name}.json" if write_tracks_json else None

        if skip_existing_kitti and write_kitti_txt and out_kitti_txt.exists():
            if verbose:
                print(f"[tracker_eval] ({idx+1}/{len(seqs)}) {seq_name}: output exists, skipping")
            # still record a placeholder row so you know it was skipped
            per_seq_rows.append(
                {
                    "seq_name": seq_name,
                    "status": "skipped_existing",
                    "out_kitti_txt": str(out_kitti_txt),
                    "out_tracks_json": str(out_tracks_json) if out_tracks_json else "",
                }
            )
            continue

        if verbose:
            print(f"[tracker_eval] ({idx+1}/{len(seqs)}) Running {seq_name} ...")

        # Run sequence
        tracks_by_frame, stats = run_tracker_from_detections_json(
            seq_name=seq_name,
            detections_json_path=str(det_json_path),
            tracker=tracker,
            warmup_steps=warmup_steps,
        )

        # Write outputs
        meta = {
            "split": split_name,
            "seq_name": seq_name,
            "tracker_name": tracker_name,
            "warmup_steps": int(warmup_steps),
            "detections_json": str(det_json_path),
        }

        write_sequence_outputs(
            seq_name=seq_name,
            tracks_by_frame=tracks_by_frame,
            out_tracks_json=str(out_tracks_json) if out_tracks_json is not None else None,
            out_kitti_txt=str(out_kitti_txt) if write_kitti_txt else None,
            kitti_use_score=kitti_use_score,
            meta=meta,
        )

        seq_stats.append(stats)

        row = stats.as_dict()
        row.update(
            {
                "status": "ok",
                "detections_json": str(det_json_path),
                "out_kitti_txt": str(out_kitti_txt) if write_kitti_txt else "",
                "out_tracks_json": str(out_tracks_json) if out_tracks_json is not None else "",
            }
        )
        per_seq_rows.append(row)

        if verbose:
            print(
                f"[tracker_eval]   {seq_name}: fps={stats.fps:.2f}, "
                f"mean={stats.mean_step_ms:.2f}ms, p90={stats.p90_step_ms:.2f}ms, p99={stats.p99_step_ms:.2f}ms"
            )

    # Build summary
    agg = _aggregate_sequence_stats(seq_stats)
    num_frames_total = int(agg.get("num_frames_total", 0))

    summary = SplitRunSummary(
        split_name=str(split_name),
        tracker_name=str(tracker_name),
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
            "tracks_json_dir": str(out_json_dir) if write_tracks_json else "",
        },
    )

    # Write summary artifacts
    summary_json_path = tracker_dir / f"runtime_summary_{split_name}.json"
    _write_json(summary_json_path, asdict(summary))

    # CSV only for successful rows (keeps it clean), but we also include skipped with empty metrics
    csv_path = tracker_dir / f"runtime_summary_{split_name}.csv"
    # Choose stable columns
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
        "out_tracks_json",
    ]
    _write_csv(csv_path, per_seq_rows, fieldnames=fieldnames)

    if verbose:
        print(f"[tracker_eval] Wrote summary: {summary_json_path}")
        print(f"[tracker_eval] Wrote CSV:     {csv_path}")
        if seq_stats:
            print(
                f"[tracker_eval] Aggregate fps (weighted): {agg['fps']['frame_weighted']:.2f} | "
                f"mean step ms (mean-of-mean): {agg['step_ms']['mean_of_mean']:.2f}"
            )
        else:
            print("[tracker_eval] No sequences were run (all skipped or none found).")

    return summary
