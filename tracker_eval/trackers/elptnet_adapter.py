# tracker_eval/trackers/elptnet_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig


@dataclass(frozen=True)
class ELPTnetConfig:
    """
    Adapter config for ELPTnet / Tracker3D.

    Key points:
      - We instantiate Tracker3D with box_type="OpenPCDet" so that ELPTnet does NOT
        apply KITTI-specific conversion (which would reorder/rotate/shift z).
      - We pass timestamp as consecutive int (frame index). ELPTnet expects consecutive
        timestamps and often uses them as dict keys.

    Args:
      cfg_file:
        Path to their jrdb.yaml (or equivalent).
      fps:
        Used only if timestamp_mode="seconds".
      track_class:
        Filter input detections by label. For JRDB, "pedestrian".
      input_score:
        Drop detections with score < input_score before tracking.
      export_score:
        If True, fill Detection.score with 1.0 (ELPTnet online API does not return scores).
      timestamp_mode:
        "frame_index" -> timestamp=int(frame_idx)
        "seconds"     -> timestamp=int(round(frame_idx / fps))  (still int and consecutive-ish)
    """
    cfg_file: str
    fps: float = 15.0

    track_class: str = "pedestrian"
    input_score: float = 0.0
    export_score: bool = False

    timestamp_mode: str = "frame_index"


class ELPTnetAdapter(TrackerBase):
    """
    Adapter around ELPTnet's `tracker.tracker.Tracker3D`.

    We use ELPTnet's online tracking API:
        tracked_bbs, tracked_ids = Tracker3D.tracking(...)

    IMPORTANT (from ELPTnet code):
      - If box_type == "Kitti", ELPTnet assumes input boxes are (h,w,l,x,y,z,yaw)
        and converts them into (x,y,z,l,w,h,yaw), also changing yaw and z.
      - Our tracker_eval uses (x,y,z,l,w,h,yaw). Therefore we MUST use box_type="OpenPCDet"
        so convert_bbs_type returns boxes unchanged.
    """

    def __init__(
        self,
        *,
        cfg: ELPTnetConfig,
        name: str = "elptnet",
        version: str = "0.1",
        run_cfg: Optional[TrackerRunConfig] = None,
    ) -> None:
        super().__init__(
            TrackerInfo(
                name=name,
                version=version,
                description="ELPTnet adapter using tracker.tracker.Tracker3D (online step).",
                extra={
                    "cfg_file": cfg.cfg_file,
                    "fps": cfg.fps,
                    "track_class": cfg.track_class,
                    "input_score": cfg.input_score,
                    "timestamp_mode": cfg.timestamp_mode,
                    "box_type": "OpenPCDet",
                },
            ),
            run_cfg=run_cfg,
        )
        self.cfg = cfg

        self._tracker: Optional[Any] = None
        self._elp_cfg: Optional[Any] = None
        self._frame_idx: int = 0

    # ---------------------------
    # Imports
    # ---------------------------

    @staticmethod
    def _lazy_imports() -> Dict[str, Any]:
        # These imports must work after you install ELPTnet repo as a package.
        from tracker.config import cfg as base_cfg  # type: ignore
        from tracker.config import cfg_from_yaml_file  # type: ignore
        from tracker.tracker import Tracker3D  # type: ignore

        return {
            "base_cfg": base_cfg,
            "cfg_from_yaml_file": cfg_from_yaml_file,
            "Tracker3D": Tracker3D,
        }

    # ---------------------------
    # Helpers
    # ---------------------------

    def _timestamp_value(self) -> int:
        """
        ELPTnet expects consecutive timestamps and frequently uses them as keys.
        Use int.

        - frame_index: 0,1,2,3,...
        - seconds:     round(i/fps) but still int; if you want strict consecutive integers,
                       prefer frame_index. (seconds can repeat if fps is high and rounding).
        """
        if self.cfg.timestamp_mode == "seconds":
            # keep int; but beware: rounding can cause duplicates at high fps.
            return int(round(float(self._frame_idx) / float(self.cfg.fps)))
        return int(self._frame_idx)

    def _detections_to_elpt_boxes(self, dets: List[Detection]) -> np.ndarray:
        """
        Convert Detection.box (JRDB) -> np.ndarray [N,7] for ELPTnet when box_type="OpenPCDet".

        With box_type="OpenPCDet", ELPTnet does NOT reorder/transform boxes, so we pass:
            [x, y, z, l, w, h, yaw] = [cx, cy, cz, l, w, h, rot_z]

        NOTE:
          If you later discover ELPTnet expects z-bottom instead of z-center, change:
            out[i, 2] = cz - h/2
          But only do that if you see a consistent vertical shift.
        """
        n = len(dets)
        out = np.zeros((n, 7), dtype=np.float32)
        for i, d in enumerate(dets):
            b = d.box
            out[i, 0] = float(b.cx)
            out[i, 1] = float(b.cy)
            out[i, 2] = float(b.cz)
            out[i, 3] = float(b.l)
            out[i, 4] = float(b.w)
            out[i, 5] = float(b.h)
            out[i, 6] = float(b.rot_z)
        return out

    # ---------------------------
    # TrackerBase overrides
    # ---------------------------

    def _reset_sequence_impl(self, seq_name: str) -> None:
        mods = self._lazy_imports()
        base_cfg = mods["base_cfg"]
        cfg_from_yaml_file = mods["cfg_from_yaml_file"]
        Tracker3D = mods["Tracker3D"]

        # Load their config object from YAML into their cfg container
        elp_cfg = cfg_from_yaml_file(self.cfg.cfg_file, base_cfg)

        # IMPORTANT: box_type="OpenPCDet" to avoid their KITTI reordering/conversion.
        tracker = Tracker3D(box_type="OpenPCDet", tracking_features=False, config=elp_cfg)

        self._elp_cfg = elp_cfg
        self._tracker = tracker
        self._frame_idx = 0

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        if self._tracker is None or self._elp_cfg is None:
            raise RuntimeError("ELPTnetAdapter used before reset_sequence().")

        # Filter by class
        dets_in: List[Detection] = [
            d for d in detections.dets
            if (self.cfg.track_class is None or d.label == self.cfg.track_class)
        ]

        # Score gating (adapter-level)
        if float(self.cfg.input_score) > 0.0:
            dets_in = [
                d for d in dets_in
                if (d.score is None or float(d.score) >= float(self.cfg.input_score))
            ]

        # Prepare arrays
        if len(dets_in) == 0:
            bbs = np.zeros((0, 7), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            bbs = self._detections_to_elpt_boxes(dets_in)
            scores = np.asarray(
                [float(d.score) if d.score is not None else 1.0 for d in dets_in],
                dtype=np.float32,
            )

        # Pose:
        # Passing None is safest unless you truly have per-frame poses in the format they expect.
        pose = None

        # Timestamp: must be consecutive int
        ts = self._timestamp_value()

        tracked_bbs, tracked_ids = self._tracker.tracking(
            bbs_3D=bbs,
            features=None,
            scores=scores,
            pose=pose,
            timestamp=ts,
        )

        self._frame_idx += 1

        tracked_bbs = np.asarray(tracked_bbs)
        tracked_ids = np.asarray(tracked_ids)

        if tracked_bbs.size == 0:
            return FrameData(frame_id=frame_id, dets=[])

        if tracked_bbs.ndim != 2 or tracked_bbs.shape[1] != 7:
            raise RuntimeError(f"ELPTnet tracker returned unexpected bbs shape: {tracked_bbs.shape}")
        if tracked_ids.ndim != 1 or tracked_ids.shape[0] != tracked_bbs.shape[0]:
            raise RuntimeError(
                f"ELPTnet tracker returned unexpected ids shape: {tracked_ids.shape} vs bbs {tracked_bbs.shape}"
            )

        out_dets: List[Detection] = []
        for bb, tid in zip(tracked_bbs.tolist(), tracked_ids.tolist()):
            x, y, z, l, w, h, yaw = [float(v) for v in bb]

            score_out: Optional[float] = 1.0 if self.cfg.export_score else None

            out_dets.append(
                Detection(
                    frame_id=frame_id,
                    track_id=int(tid),
                    box=Box3D(cx=x, cy=y, cz=z, l=l, w=w, h=h, rot_z=yaw),
                    score=score_out,
                    label=self.cfg.track_class,
                    raw_label_id=None,
                )
            )

        return FrameData(frame_id=frame_id, dets=out_dets)
