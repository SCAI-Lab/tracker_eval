# tracker_eval/runner/run_sequence.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from tracker_eval.common.types import Detection, FrameData, frame_sort_key
from tracker_eval.data.jrdb_io import load_jrdb_detections_3d
from tracker_eval.data.tracks_io import save_tracks_json
from tracker_eval.export.jrdb_kitti_writer import TrackRow3D, write_sequence_kitti_txt


# ----------------------------
# Tracker interface
# ----------------------------

class Tracker3D(Protocol):
    """
    Unified tracker interface for tracker_eval runners.

    Recommended (TrackerBase-compatible):
      - reset_sequence(seq_name) -> None
      - step(frame_id, detections, timestamp=None) -> FrameData

    Optional (for GT-assisted / headroom trackers):
      - set_gt_for_sequence(gt_by_frame: Dict[str, FrameData]) -> None
      OR
      - step_with_gt(frame_id, detections: List[Detection], gt: List[Detection], timestamp=None) -> FrameData

    Backward-compatible support:
      - reset() -> None
      - step(frame_id, detections: List[Detection]) -> List[Detection]
    """

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


# ----------------------------
# Core runner helpers
# ----------------------------

def _tracker_reset_sequence(tracker: Any, seq_name: str) -> None:
    """
    Prefer TrackerBase-style reset_sequence(seq_name).
    Fall back to legacy reset().
    """
    if hasattr(tracker, "reset_sequence"):
        tracker.reset_sequence(seq_name)
        return
    if hasattr(tracker, "reset"):
        tracker.reset()
        return
    raise AttributeError("Tracker must implement reset_sequence(seq_name) or reset().")


def _maybe_set_gt_for_sequence(tracker: Any, gt_by_frame: Optional[Mapping[str, FrameData]]) -> None:
    """
    If tracker supports set_gt_for_sequence(...), preload GT once per sequence.
    This is the cleanest integration for GT-assisted trackers (e.g. headroom).
    """
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
    """
    Prefer TrackerBase-style:
      - step(frame_id, dets, timestamp=...)
    Optional GT-assisted:
      - step_with_gt(frame_id, dets, gt_dets, timestamp=...)
    Fall back to legacy:
      - step(frame_id, dets) -> List[Detection]
    """
    # GT-assisted path if available and gt is provided
    if gt_dets is not None and hasattr(tracker, "step_with_gt"):
        try:
            out = tracker.step_with_gt(frame_id, dets, gt_dets, timestamp=timestamp)  # type: ignore[attr-defined]
        except TypeError:
            # older signature without timestamp
            out = tracker.step_with_gt(frame_id, dets, gt_dets)  # type: ignore[attr-defined]
    else:
        # Normal step
        try:
            out = tracker.step(frame_id, dets, timestamp=timestamp)
        except TypeError:
            # legacy trackers not accepting timestamp kwarg
            out = tracker.step(frame_id, dets)

    # TrackerBase returns FrameData
    if isinstance(out, FrameData):
        return out

    # Legacy trackers return List[Detection]
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
) -> Tuple[Dict[str, FrameData], SequenceRunStats]:
    """
    Run a tracker on a sequence's detections.

    Supported tracker styles:

    Preferred (TrackerBase-style):
      - tracker.reset_sequence(seq_name)
      - tracker.step(frame_id, detections, timestamp=None) -> FrameData

    Optional GT-assisted:
      - tracker.set_gt_for_sequence(gt_by_frame)   (called once per sequence), and/or
      - tracker.step_with_gt(frame_id, dets, gt_dets, timestamp=None) -> FrameData

    Legacy:
      - tracker.reset()
      - tracker.step(frame_id, dets: List[Detection]) -> List[Detection]

    Parameters
    ----------
    seq_name:
        Sequence name (for stats and reset_sequence()).
    detections_by_frame:
        dict: frame_id -> FrameData
    tracker:
        Tracker instance.
    frames:
        Optional explicit frame ordering. If None, inferred and sorted.
    warmup_steps:
        Number of initial frames to run but exclude from timing stats.
    gt_by_frame:
        Optional GT mapping frame_id -> FrameData. Used only by GT-assisted trackers.
    timestamps_by_frame:
        Optional mapping frame_id -> timestamp_s (float). If omitted, timestamp=None is passed.

    Returns
    -------
    tracks_by_frame:
        dict: frame_id -> FrameData of tracked detections (track_id assigned)
    stats:
        timing stats (fps etc.)
    """
    # Reset tracker for this sequence
    _tracker_reset_sequence(tracker, seq_name)

    # Preload GT if tracker supports it
    _maybe_set_gt_for_sequence(tracker, gt_by_frame)

    # Determine frame ordering
    if frames is None:
        keys = set(detections_by_frame.keys())
        if gt_by_frame is not None:
            keys |= set(gt_by_frame.keys())
        frames = sorted(keys, key=frame_sort_key)

    # Run
    step_times_ms: List[float] = []
    tracks_by_frame: Dict[str, FrameData] = {}

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

        # Ensure frame_id is consistent
        if tracked_fd.frame_id != frame_id:
            tracked_fd.frame_id = frame_id  # type: ignore[attr-defined]

        tracks_by_frame[frame_id] = tracked_fd

        dt_ms = (te - ts) * 1000.0
        if i >= warmup_steps:
            step_times_ms.append(dt_ms)

    t1 = time.perf_counter()
    total_time_s = max(1e-12, (t1 - t0))

    num_frames = len(frames)
    effective_frames = max(1, (num_frames - warmup_steps))

    # Effective time based on measured (post-warmup) step times
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
    return tracks_by_frame, stats


# ----------------------------
# Export helpers for JRDB toolkit
# ----------------------------

def tracks_by_frame_to_kitti_rows(
    tracks_by_frame: Mapping[str, FrameData],
    *,
    default_score: float = 1.0,
) -> Dict[str, List[TrackRow3D]]:
    """
    Convert our internal FrameData/Detection format into TrackRow3D for KITTI writer.

    Important:
    - Our internal Box3D is center-based (cx,cy,cz,l,w,h,rot_z).
    """
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
    out_tracks_json: Optional[str] = None,
    out_kitti_txt: Optional[str] = None,
    kitti_use_score: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write outputs for a single sequence.
      - JSON (our internal format) is optional but convenient.
      - KITTI txt is what JRDB toolkit consumes.
    """
    if out_tracks_json is not None:
        save_tracks_json(
            seq_name=seq_name,
            tracks_by_frame=tracks_by_frame,
            out_path=out_tracks_json,
            meta=meta,
            box_as_dict=True,
        )

    if out_kitti_txt is not None:
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
) -> Tuple[Dict[str, FrameData], SequenceRunStats]:
    """
    Load a MinkUNet detections_3D JSON and run tracker.

    Optional:
      - gt_by_frame: pass GT frames if you run a GT-assisted tracker (e.g., headroom)
      - timestamps_by_frame: if you have real timestamps
    """
    dets_by_frame = load_jrdb_detections_3d(detections_json_path)

    # Default frame order: union of det and gt frames (if GT provided)
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
    )
