# tracker_eval/trackers/ab3dmot_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import os

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig


@dataclass(frozen=True)
class AB3DMOTConfig:
    """
    Configuration for AB3DMOT.

    These defaults match common AB3DMOT usage, but you should tune for JRDB.
    """
    max_age: int = 7
    min_hits: int = 5
    thresh_3d_iou: float = 0.33
    thresh_3d_dist: float = 0.5
    metrics: Tuple[str, str] = ("iou_3d", "dist_3d")
    log_dir: Optional[str] = None  # If set, passed to AB3DMOT as a path string



def _reorder_to_ab3dmot_cam_frame(boxes: np.ndarray) -> np.ndarray:
    """
    Convert from our internal LiDAR-base convention:
      (x, y, z, l, w, h, theta)  where x-forward, y-left, z-up
    to AB3DMOT's "camera frame" convention used in your snippet:
      (h, w, l, -y, -z, x, theta)

    Input:
      boxes: (N, 7)

    Output:
      boxes_cam: (N, 7)
    """
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"Expected boxes shape (N,7), got {boxes.shape}")
    inds = [5, 4, 3, 1, 2, 0, 6]
    out = boxes[:, inds].copy()
    out[:, 3] *= -1.0  # -y
    out[:, 4] *= -1.0  # -z
    return out


def _reorder_from_ab3dmot_cam_frame(trks: np.ndarray) -> np.ndarray:
    """
    Convert AB3DMOT output from camera convention back to our internal:

    From (h, w, l, -y, -z, x, theta, ID)
      to (x, y, z, l, w, h, theta, ID)

    Input:
      trks: (M, 8)

    Output:
      out: (M, 8) with columns (x, y, z, l, w, h, theta, ID)
    """
    if trks.ndim != 2 or trks.shape[1] != 8:
        raise ValueError(f"Expected trks shape (M,8), got {trks.shape}")
    inds = [5, 3, 4, 2, 1, 0, 6, 7]
    out = trks[:, inds].copy()
    out[:, 1] *= -1.0  # y
    out[:, 2] *= -1.0  # z
    return out


class AB3DMOTAdapter(TrackerBase):
    """
    AB3DMOT adapter for tracker_eval.

    - Input detections: FrameData with Detection.box in JRDB base (cx,cy,cz,l,w,h,rot_z)
    - Output tracks: FrameData with persistent track IDs in Detection.track_id

    Notes
    -----
    AB3DMOT's `track()` signature differs slightly across forks.
    This adapter supports the calling pattern you provided:

        trks = tracker.track(trk_input, fr_idx, seq_name)[0][0]

    If your AB3DMOT fork returns a different nesting, adjust `_extract_tracks(...)`.
    """

    def __init__(
        self,
        *,
        cfg: Optional[AB3DMOTConfig] = None,
        run_cfg: Optional[TrackerRunConfig] = None,
        name: str = "ab3dmot",
        version: str = "unknown",
    ) -> None:
        self.cfg = cfg or AB3DMOTConfig()
        info = TrackerInfo(
            name=name,
            version=version,
            description="AB3DMOT adapter (3D IoU + dist) for JRDB detections.",
            extra={
                "max_age": self.cfg.max_age,
                "min_hits": self.cfg.min_hits,
                "thresh_3d_iou": self.cfg.thresh_3d_iou,
                "thresh_3d_dist": self.cfg.thresh_3d_dist,
                "metrics": list(self.cfg.metrics),
            },
        )
        super().__init__(info, run_cfg=run_cfg)

        self._tracker = None
        self._seq_name: Optional[str] = None
        self._frame_index: int = 0

    def _reset_sequence_impl(self, seq_name: str) -> None:
        self._seq_name = seq_name
        self._frame_index = 0

        try:
            from AB3DMOT_libs.model import AB3DMOT  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import AB3DMOT from 'AB3DMOT_libs.model'. "
                "Make sure AB3DMOT is installed and available in PYTHONPATH on Jetson."
            ) from e

        max_age = int(self.cfg.max_age)
        min_hits = int(self.cfg.min_hits)
        thres = [float(self.cfg.thresh_3d_iou), float(self.cfg.thresh_3d_dist)]
        metric = list(self.cfg.metrics)

        # IMPORTANT: your AB3DMOT fork expects `log` to be a FILE PATH, because it does open(self.log, 'w')
        log_path = None
        if self.cfg.log_dir is not None:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            log_path = os.path.join(self.cfg.log_dir, f"{seq_name}.txt")

        # Some forks might allow log=None; yours likely wants a string.
        # If log_path is None, use a safe default file in /tmp to avoid breaking.
        if log_path is None:
            log_path = os.path.join("/tmp", f"ab3dmot_{seq_name}.txt")

        self._tracker = AB3DMOT(
            max_age=max_age,
            min_hits=min_hits,
            thres=thres,
            metric=metric,
            log=str(log_path),
        )


    def _extract_tracks(self, out: object) -> np.ndarray:
        """
        Normalize AB3DMOT output to an (M, 8) numpy array in cam convention:
          (h, w, l, -y, -z, x, theta, ID)

        Your snippet uses: tracker.track(...)[0][0]
        but forks vary. We try to handle the common patterns robustly.
        """
        if out is None:
            return np.zeros((0, 8), dtype=np.float32)

        # Common pattern: nested lists/tuples
        # Try progressively to reach a numpy array of shape (M,8).
        cand = out
        for _ in range(3):
            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                cand = cand[0]
            else:
                break

        arr = np.asarray(cand)
        if arr.size == 0:
            return np.zeros((0, 8), dtype=np.float32)

        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError(f"AB3DMOT output has unexpected shape {arr.shape}; expected (M,>=8).")

        # Keep only first 8 columns if extra are present
        arr = arr[:, :8].astype(np.float32, copy=False)
        return arr

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        if self._tracker is None:
            raise RuntimeError("AB3DMOTAdapter: tracker is not initialized. Did you call reset_sequence()?")

        # Build detections array (N,7) in our internal order
        # (x, y, z, l, w, h, theta)
        dets_list: List[List[float]] = []
        scores_list: List[float] = []
        for det in detections.dets:
            b = det.box
            dets_list.append([b.cx, b.cy, b.cz, b.l, b.w, b.h, b.rot_z])
            scores_list.append(float(det.score) if det.score is not None else 1.0)

        if len(dets_list) == 0:
            dets_np = np.zeros((0, 7), dtype=np.float32)
            info_np = np.zeros((0, 7), dtype=np.float32)
        else:
            dets_np = np.asarray(dets_list, dtype=np.float32)
            dets_np = _reorder_to_ab3dmot_cam_frame(dets_np)
            # AB3DMOT expects an "info" matrix; in your snippet it is zeros_like(dets)
            info_np = np.zeros_like(dets_np, dtype=np.float32)

        trk_input = {"dets": dets_np, "info": info_np}

        # Call AB3DMOT (your snippet: track(trk_input, fr_idx, seq_name))
        out = self._tracker.track(trk_input, self._frame_index, self._seq_name)  # type: ignore[attr-defined]
        self._frame_index += 1

        trks_cam = self._extract_tracks(out)  # (M,8) cam convention
        trks_base = _reorder_from_ab3dmot_cam_frame(trks_cam)  # (M,8) base convention

        out_dets: List[Detection] = []
        for row in trks_base:
            x, y, z, l, w, h, theta, tid = row.tolist()
            track_id = int(tid)

            box = Box3D(
                cx=float(x),
                cy=float(y),
                cz=float(z),
                l=float(l),
                w=float(w),
                h=float(h),
                rot_z=float(theta),
            )

            # AB3DMOT itself doesn't output a confidence; keep None or use 1.0.
            # If you want to propagate detection score, you'd need association mapping.
            out_dets.append(
                Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    box=box,
                    score=None,
                    label="pedestrian",
                    raw_label_id=None,
                )
            )

        return FrameData(frame_id=frame_id, dets=out_dets)
