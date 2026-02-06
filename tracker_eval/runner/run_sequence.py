# tracker_eval/runner/run_sequence.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData, frame_sort_key
from tracker_eval.data.jrdb_io import load_jrdb_detections_3d
from tracker_eval.data.tracks_io import save_tracks_json
from tracker_eval.export.jrdb_kitti_writer import (
    TrackRow3D,
    write_sequence_kitti_txt,
)


# ----------------------------
# Tracker interface
# ----------------------------

class Tracker3D(Protocol):
    """
    Unified tracker interface for tracker_eval runners.

    Recommended (TrackerBase-compatible):
      - reset_sequence(seq_name) -> None
      - step(frame_id, detections, timestamp=None) -> FrameData

    Backward-compatible support:
      - reset() -> None
      - step(frame_id, detections: List[Detection]) -> List[Detection]
    """

    # Preferred API (matches TrackerBase)
    def reset_sequence(self, seq_name: str) -> None:
        ...

    def step(
        self,
        frame_id: str,
        detections: Any,
        *,
        timestamp: Optional[float] = None,
    ) -> Any:
        ...


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
# Core runner
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


def _tracker_step(tracker: Any, frame_id: str, dets: List[Detection]) -> FrameData:
    """
    Prefer TrackerBase-style step(...) -> FrameData.
    Fall back to legacy step(...) -> List[Detection].
    """
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

def run_tracker_on_sequence(
    *,
    seq_name: str,
    detections_by_frame: Mapping[str, FrameData],
    tracker: Tracker3D,
    frames: Optional[List[str]] = None,
    warmup_steps: int = 0,
) -> Tuple[Dict[str, FrameData], SequenceRunStats]:
    """
    Run a tracker on a sequence's detections.

    This runner supports two tracker styles:

    Preferred (TrackerBase-style):
      - tracker.reset_sequence(seq_name)
      - tracker.step(frame_id, detections, timestamp=None) -> FrameData

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

    Returns
    -------
    tracks_by_frame:
        dict: frame_id -> FrameData of tracked detections (track_id assigned)
    stats:
        timing stats (fps etc.)
    """
    # Reset tracker for this sequence (prefers reset_sequence, falls back to reset)
    _tracker_reset_sequence(tracker, seq_name)

    # Determine frame ordering
    if frames is None:
        frames = list(detections_by_frame.keys())
        frames.sort(key=frame_sort_key)

    # Run
    step_times_ms: List[float] = []
    tracks_by_frame: Dict[str, FrameData] = {}

    t0 = time.perf_counter()
    for i, frame_id in enumerate(frames):
        dets = detections_by_frame.get(frame_id, FrameData(frame_id=frame_id, dets=[])).dets

        ts = time.perf_counter()
        tracked_fd = _tracker_step(tracker, frame_id, dets)  # -> FrameData
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
    - JRDB toolkit 3D IoU code expects y_top = cy + h/2 (we convert in writer helper).
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
        # convert to expected writer input
        rows_by_frame = tracks_by_frame_to_kitti_rows(tracks_by_frame)

        # writer expects mapping frame->list[TrackRow3D]
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
) -> Tuple[Dict[str, FrameData], SequenceRunStats]:
    """
    Load a MinkUNet detections_3D JSON and run tracker.
    """
    dets_by_frame = load_jrdb_detections_3d(detections_json_path)
    frames = sorted(dets_by_frame.keys(), key=frame_sort_key)
    return run_tracker_on_sequence(
        seq_name=seq_name,
        detections_by_frame=dets_by_frame,
        tracker=tracker,
        frames=frames,
        warmup_steps=warmup_steps,
    )
