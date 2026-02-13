# tracker_eval/runner/run_sequence.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from tracker_eval.common.types import Detection, FrameData, frame_sort_key
from tracker_eval.data.jrdb_io import load_jrdb_detections_3d
from tracker_eval.export.jrdb_kitti_writer import TrackRow3D, write_sequence_kitti_txt


# ----------------------------
# Tracker interface
# ----------------------------

class Tracker3D(Protocol):
    def reset_sequence(self, seq_name: str) -> None: ...
    def step(self, frame_id: str, detections: Any, *, timestamp: Optional[float] = None) -> Any: ...


# ----------------------------
# Timing / stats
# ----------------------------

@dataclass
class SequenceRunStats:
    seq_name: str
    num_frames: int
    total_time_s: float
    fps: float
    mean_step_ms: float
    p50_step_ms: float
    p90_step_ms: float
    p99_step_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seq_name": self.seq_name,
            "num_frames": int(self.num_frames),
            "total_time_s": float(self.total_time_s),
            "fps": float(self.fps),
            "mean_step_ms": float(self.mean_step_ms),
            "p50_step_ms": float(self.p50_step_ms),
            "p90_step_ms": float(self.p90_step_ms),
            "p99_step_ms": float(self.p99_step_ms),
        }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _as_int_frame_id(frame_id: str) -> int:
    s = str(frame_id).strip()
    if "." in s:
        s = s.split(".")[0]
    return int(s)


# ----------------------------
# Core runner helpers
# ----------------------------

def _tracker_reset_sequence(tracker: Any, seq_name: str) -> None:
    if hasattr(tracker, "reset_sequence"):
        tracker.reset_sequence(seq_name)
        return
    if hasattr(tracker, "reset"):
        tracker.reset()
        return
    raise AttributeError("Tracker must implement reset_sequence(seq_name) or reset().")


def _maybe_set_gt_for_sequence(tracker: Any, gt_by_frame: Optional[Mapping[str, FrameData]]) -> None:
    if gt_by_frame is None:
        return
    if hasattr(tracker, "set_gt_for_sequence"):
        tracker.set_gt_for_sequence(dict(gt_by_frame))  # type: ignore[attr-defined]


def _tracker_step(
    tracker: Any,
    frame_id: str,
    dets: List[Detection],
    *,
    timestamp: Optional[float] = None,
    gt_dets: Optional[List[Detection]] = None,
) -> FrameData:
    if gt_dets is not None and hasattr(tracker, "step_with_gt"):
        try:
            out = tracker.step_with_gt(frame_id, dets, gt_dets, timestamp=timestamp)  # type: ignore[attr-defined]
        except TypeError:
            out = tracker.step_with_gt(frame_id, dets, gt_dets)  # type: ignore[attr-defined]
    else:
        try:
            out = tracker.step(frame_id, dets, timestamp=timestamp)
        except TypeError:
            out = tracker.step(frame_id, dets)

    if isinstance(out, FrameData):
        return out
    if isinstance(out, list):
        return FrameData(frame_id=frame_id, dets=out)

    raise TypeError(
        "Tracker.step must return FrameData (preferred) or List[Detection] (legacy). "
        f"Got: {type(out)}"
    )


# ----------------------------
# Core runner
# ----------------------------

def run_tracker_on_sequence(
    *,
    seq_name: str,
    detections_by_frame: Mapping[str, FrameData],
    tracker: Tracker3D,
    frames: Optional[List[str]] = None,
    warmup_steps: int = 0,
    gt_by_frame: Optional[Mapping[str, FrameData]] = None,
    timestamps_by_frame: Optional[Mapping[str, float]] = None,
    profile: bool = True,
) -> Tuple[Dict[str, FrameData], SequenceRunStats, List[Dict[str, Any]]]:
    """
    If profile=False:
      - no perf_counter timing measurements
      - returns stats with 0 timing fields
      - returns empty frame_stats list
    """
    frame_stats: List[Dict[str, Any]] = []

    _tracker_reset_sequence(tracker, seq_name)
    _maybe_set_gt_for_sequence(tracker, gt_by_frame)

    if frames is None:
        keys = set(detections_by_frame.keys())
        if gt_by_frame is not None:
            keys |= set(gt_by_frame.keys())
        frames = sorted(keys, key=frame_sort_key)

    tracks_by_frame: Dict[str, FrameData] = {}

    if not profile:
        # No profiling: just run the tracker
        for frame_id in frames:
            dets = detections_by_frame.get(frame_id, FrameData(frame_id=frame_id, dets=[])).dets

            gt_dets = None
            if gt_by_frame is not None:
                gt_dets = gt_by_frame.get(frame_id, FrameData(frame_id=frame_id, dets=[])).dets

            ts_frame = None
            if timestamps_by_frame is not None:
                ts_frame = timestamps_by_frame.get(frame_id, None)

            tracked_fd = _tracker_step(tracker, frame_id, dets, timestamp=ts_frame, gt_dets=gt_dets)

            if tracked_fd.frame_id != frame_id:
                tracked_fd.frame_id = frame_id  # type: ignore[attr-defined]
            tracks_by_frame[frame_id] = tracked_fd

        stats = SequenceRunStats(
            seq_name=seq_name,
            num_frames=len(frames),
            total_time_s=0.0,
            fps=0.0,
            mean_step_ms=0.0,
            p50_step_ms=0.0,
            p90_step_ms=0.0,
            p99_step_ms=0.0,
        )
        return tracks_by_frame, stats, frame_stats

    # Profiling path (original behavior)
    step_times_ms: List[float] = []

    t0 = time.perf_counter()
    for i, frame_id in enumerate(frames):
        dets = detections_by_frame.get(frame_id, FrameData(frame_id=frame_id, dets=[])).dets
        gt_dets = None
        if gt_by_frame is not None:
            gt_dets = gt_by_frame.get(frame_id, FrameData(frame_id=frame_id, dets=[])).dets

        ts_frame = None
        if timestamps_by_frame is not None:
            ts_frame = timestamps_by_frame.get(frame_id, None)

        ts = time.perf_counter()
        tracked_fd = _tracker_step(tracker, frame_id, dets, timestamp=ts_frame, gt_dets=gt_dets)
        te = time.perf_counter()

        if tracked_fd.frame_id != frame_id:
            tracked_fd.frame_id = frame_id  # type: ignore[attr-defined]

        tracks_by_frame[frame_id] = tracked_fd

        dt_ms = (te - ts) * 1000.0
        if i >= warmup_steps:
            step_times_ms.append(dt_ms)

        num_det_in = int(len(dets))
        num_tracks_out = int(len(tracked_fd.dets))
        num_gt = int(len(gt_dets)) if gt_dets is not None else 0

        fps_inst = float(1000.0 / dt_ms) if dt_ms > 1e-9 else 0.0

        row = {
            "frame_id": frame_id,
            "frame_idx": _as_int_frame_id(frame_id),
            "step_ms": float(dt_ms),
            "fps_inst": float(fps_inst),
            "num_det_in": num_det_in,
            "num_tracks_out": num_tracks_out,
            "num_gt": num_gt,
            "is_warmup": bool(i < warmup_steps),
        }
        if ts_frame is not None:
            row["timestamp_s"] = float(ts_frame)

        frame_stats.append(row)

    t1 = time.perf_counter()
    total_time_s = max(1e-12, (t1 - t0))

    num_frames = len(frames)
    effective_frames = max(1, (num_frames - warmup_steps))

    if step_times_ms:
        effective_time_s = max(1e-12, sum(step_times_ms) / 1000.0)
    else:
        effective_time_s = total_time_s

    fps = float(effective_frames / effective_time_s)

    stats = SequenceRunStats(
        seq_name=seq_name,
        num_frames=num_frames,
        total_time_s=total_time_s,
        fps=fps,
        mean_step_ms=float(np.mean(step_times_ms)) if step_times_ms else 0.0,
        p50_step_ms=_percentile(step_times_ms, 50),
        p90_step_ms=_percentile(step_times_ms, 90),
        p99_step_ms=_percentile(step_times_ms, 99),
    )
    return tracks_by_frame, stats, frame_stats


# ----------------------------
# Export helpers for JRDB toolkit
# ----------------------------

def tracks_by_frame_to_kitti_rows(
    tracks_by_frame: Mapping[str, FrameData],
    *,
    default_score: float = 1.0,
) -> Dict[str, List[TrackRow3D]]:
    out: Dict[str, List[TrackRow3D]] = {}
    for frame_id, fd in tracks_by_frame.items():
        rows: List[TrackRow3D] = []
        for det in fd.dets:
            b = det.box
            box7 = np.asarray([b.cx, b.cy, b.cz, b.l, b.w, b.h, b.rot_z], dtype=np.float32)
            score = float(det.score) if det.score is not None else float(default_score)
            rows.append(TrackRow3D(track_id=int(det.track_id), box7=box7, score=score))
        out[frame_id] = rows
    return out


def write_sequence_outputs(
    *,
    seq_name: str,
    tracks_by_frame: Dict[str, FrameData],
    out_kitti_txt: Optional[str] = None,
    kitti_use_score: bool = True,
) -> None:
    
    rows_by_frame = tracks_by_frame_to_kitti_rows(tracks_by_frame)
    write_sequence_kitti_txt(
        out_txt_path=out_kitti_txt,
        tracks_by_frame=rows_by_frame,
        class_name="Pedestrian",
        truncated=0,
        occluded=0,
        alpha=0.0,
        bbox2d=(0.0, 0.0, 0.0, 0.0),
        use_score=kitti_use_score,
        sort_rows=True,
    )


# ----------------------------
# Convenience: run from JRDB detections file
# ----------------------------

def run_tracker_from_detections_json(
    *,
    seq_name: str,
    detections_json_path: str,
    tracker: Tracker3D,
    warmup_steps: int = 0,
    gt_by_frame: Optional[Mapping[str, FrameData]] = None,
    timestamps_by_frame: Optional[Mapping[str, float]] = None,
    profile: bool = True,
) -> Tuple[Dict[str, FrameData], SequenceRunStats, List[Dict[str, Any]]]:
    dets_by_frame = load_jrdb_detections_3d(detections_json_path)

    keys = set(dets_by_frame.keys())
    if gt_by_frame is not None:
        keys |= set(gt_by_frame.keys())
    frames = sorted(keys, key=frame_sort_key)

    return run_tracker_on_sequence(
        seq_name=seq_name,
        detections_by_frame=dets_by_frame,
        tracker=tracker,
        frames=frames,
        warmup_steps=warmup_steps,
        gt_by_frame=gt_by_frame,
        timestamps_by_frame=timestamps_by_frame,
        profile=profile,
    )
