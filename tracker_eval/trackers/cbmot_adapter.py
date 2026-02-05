# tracker_eval/trackers/cbmot_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig


@dataclass(frozen=True)
class CBMOTConfig:
    """
    Adapter config for CBMOT (PubTracker / CenterTrack-style).

    Notes:
      - CBMOT's PubTracker.step_centertrack expects per-frame detections in a nuScenes-like dict format:
        keys used by tracker.py:
          - detection_name (str)
          - translation (list len 3)
          - size (list len 3)          (order not super important for 2D matching, but we keep consistent)
          - rotation (quat list len 4) (w,x,y,z)
          - velocity (list len 2 or 3) (tracker uses [:2])
          - detection_score (float)
          - attribute_name (str)       (not used by tracker logic; we provide placeholder)

      - Matching is done in XY (2D center). It uses 'velocity' as a *back-step* offset:
            det['tracking'] = np.array(det['velocity'][:2]) * -1 * time_lag
        so if you don't provide velocity, set it to zeros.

      - Output tracklets have 'tracking_id' (int), and geometry fields like translation/size/rotation.
    """

    # Expected dataset type in PubTracker (only affects allowed class list + velocity_error thresholds)
    dataset: str = "Nuscenes"  # "Nuscenes" or "Waymo"; keep Nuscenes to allow 'pedestrian'

    # Core tracker params (PubTracker __init__)
    hungarian: bool = False
    max_age: int = 40
    min_hits: int = 1
    score_decay: float = 0.0          # "noise" in code (score decay for matched tracks)
    active_th: float = 1.0
    deletion_th: float = 0.0
    detection_th: float = 0.0
    score_update: Optional[str] = None  # None, 'nn', 'addition', 'max', etc.
    model_path: Optional[str] = None

    # Adapter-level control
    fps: float = 15.0                 # JRDB default; easy to swap for another dataset
    track_class: str = "pedestrian"   # filter + label output; set None-like to track all

    # If True, we put CBMOT tracking_score into Detection.score; otherwise keep None
    export_score: bool = False


class CBMOTAdapter(TrackerBase):
    """
    Adapter for CBMOT (PubTracker).

    This adapter runs CBMOT in "PointTracker" mode (default in their code),
    which matches detections to previous tracks using XY distance + per-class thresholds.

    We do NOT use:
      - nuScenes sample tokens
      - annotations / train_data
      - fusion

    We just feed detections each frame and read back 'tracking_id' + updated geometry.
    """

    def __init__(
        self,
        *,
        cfg: CBMOTConfig,
        name: str = "cbmot",
        version: str = "0.1",
        run_cfg: Optional[TrackerRunConfig] = None,
    ) -> None:
        super().__init__(
            TrackerInfo(
                name=name,
                version=version,
                description="CBMOT adapter (PubTracker / CenterTrack-style), XY center matching + optional score update.",
                extra={
                    "hungarian": cfg.hungarian,
                    "max_age": cfg.max_age,
                    "min_hits": cfg.min_hits,
                    "fps": cfg.fps,
                    "track_class": cfg.track_class,
                },
            ),
            run_cfg=run_cfg,
        )
        self.cfg = cfg

        self._tracker: Optional[Any] = None
        self._frame_counter: int = 0
        self._last_timestamp_s: Optional[float] = None

    # ---------------------------
    # Lazy imports
    # ---------------------------

    @staticmethod
    def _lazy_import_pubtracker() -> Any:
        # If you install CBMOT as a top-level package `cbmot`,
        # this import becomes: from cbmot.tracker import PubTracker
        #
        # If you prefer vendoring CBMOT source inside your repo,
        # you can adjust this import path accordingly.
        from cbmot.tracker import PubTracker  # type: ignore
        return PubTracker

    # ---------------------------
    # Helpers: Box/quat conversion
    # ---------------------------

    @staticmethod
    def _yaw_to_quat_wxyz(yaw: float) -> List[float]:
        """
        Convert yaw (about +z) to quaternion [w, x, y, z].
        CBMOT expects rotation stored in this order (typical nuScenes JSON).
        """
        half = 0.5 * float(yaw)
        w = float(np.cos(half))
        x = 0.0
        y = 0.0
        z = float(np.sin(half))
        return [w, x, y, z]

    @staticmethod
    def _quat_wxyz_to_yaw(q: Sequence[float]) -> float:
        """
        Convert quaternion [w, x, y, z] to yaw about z.
        We assume roll/pitch are near 0.
        """
        if len(q) != 4:
            return 0.0
        w, x, y, z = [float(v) for v in q]
        # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2)) but with x=y=0, reduces to 2*atan2(z,w)
        return float(2.0 * np.arctan2(z, w))

    def _timestamp_for_frame(self) -> float:
        """
        Stable monotonic timestamp derived from frame index and cfg.fps.
        """
        if self.cfg.fps <= 0:
            raise ValueError(f"CBMOTConfig.fps must be > 0, got {self.cfg.fps}")
        return float(self._frame_counter / self.cfg.fps)

    def _detections_to_cbmot_results(self, dets: Sequence[Detection]) -> List[Dict[str, Any]]:
        """
        Convert tracker_eval Detection -> CBMOT detection dict list.
        """
        results: List[Dict[str, Any]] = []
        for det in dets:
            if self.cfg.track_class and det.label != self.cfg.track_class:
                continue

            b = det.box
            score = float(det.score) if det.score is not None else 1.0

            # CBMOT expects:
            # translation: [x,y,z]
            # size: [w,l,h] in nuScenes; their tracker doesn't care for matching,
            # but we keep a consistent order for round-tripping.
            translation = [float(b.cx), float(b.cy), float(b.cz)]
            size = [float(b.w), float(b.l), float(b.h)]
            rotation = self._yaw_to_quat_wxyz(b.rot_z)

            # Velocity: JRDB detections usually don't include. Provide zeros.
            # CBMOT uses velocity only for back-step prediction of det position.
            velocity = [0.0, 0.0, 0.0]

            results.append(
                {
                    "detection_name": str(det.label),
                    "translation": translation,
                    "size": size,
                    "rotation": rotation,
                    "velocity": velocity,
                    "detection_score": float(score),
                    "attribute_name": "",  # placeholder; not used in tracker.py logic
                }
            )
        return results

    def _tracklet_to_detection(self, frame_id: str, trk: Dict[str, Any]) -> Detection:
        """
        Convert CBMOT tracklet dict -> tracker_eval Detection.
        """
        # track id
        tid_raw = trk.get("tracking_id", None)
        if tid_raw is None:
            raise RuntimeError("CBMOT tracklet missing 'tracking_id'")

        try:
            tid = int(tid_raw)
        except Exception as e:
            raise RuntimeError(f"CBMOT tracklet tracking_id not int-like: {tid_raw}") from e

        # geometry
        translation = trk.get("translation", None)
        size = trk.get("size", None)
        rotation = trk.get("rotation", None)

        if translation is None or size is None or rotation is None:
            raise RuntimeError("CBMOT tracklet missing translation/size/rotation")

        if len(translation) < 3 or len(size) < 3 or len(rotation) < 4:
            raise RuntimeError(
                f"CBMOT tracklet invalid geometry lengths: "
                f"translation={len(translation)}, size={len(size)}, rotation={len(rotation)}"
            )

        cx, cy, cz = float(translation[0]), float(translation[1]), float(translation[2])

        # size stored as [w,l,h] in our adapter
        w, l, h = float(size[0]), float(size[1]), float(size[2])

        yaw = self._quat_wxyz_to_yaw(rotation)

        # label
        label = trk.get("detection_name", None) or trk.get("tracking_name", None) or self.cfg.track_class or "pedestrian"
        label = str(label)

        # score
        score_out: Optional[float] = None
        if self.cfg.export_score:
            # CBMOT sets tracking_score = detection_score (unless fusion)
            # In step_centertrack they keep detection_score updated; main.py copies to tracking_score.
            s = trk.get("tracking_score", None)
            if s is None:
                s = trk.get("detection_score", None)
            if s is not None:
                score_out = float(s)

        return Detection(
            frame_id=frame_id,
            track_id=tid,
            box=Box3D(cx=cx, cy=cy, cz=cz, l=l, w=w, h=h, rot_z=yaw),
            score=score_out,
            label=label if (self.cfg.track_class is None or label == self.cfg.track_class) else (self.cfg.track_class or label),
            raw_label_id=None,
        )

    # ---------------------------
    # TrackerBase overrides
    # ---------------------------

    def _reset_sequence_impl(self, seq_name: str) -> None:
        PubTracker = self._lazy_import_pubtracker()

        self._tracker = PubTracker(
            hungarian=bool(self.cfg.hungarian),
            max_age=int(self.cfg.max_age),
            noise=float(self.cfg.score_decay),
            active_th=float(self.cfg.active_th),
            min_hits=int(self.cfg.min_hits),
            score_update=self.cfg.score_update,
            deletion_th=float(self.cfg.deletion_th),
            detection_th=float(self.cfg.detection_th),
            dataset=str(self.cfg.dataset),
            model_path=str(self.cfg.model_path) if self.cfg.model_path is not None else "LeakyReLU.th",
        )

        # Ensure clean state (PubTracker.__init__ calls reset(), but keep explicit)
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()

        self._frame_counter = 0
        self._last_timestamp_s = None

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        if self._tracker is None:
            raise RuntimeError("CBMOTAdapter used before reset_sequence().")

        # Timestamping:
        # - prefer provided timestamp if runner supplies it
        # - else generate from frame index and cfg.fps
        if timestamp is None:
            ts = self._timestamp_for_frame()
        else:
            ts = float(timestamp)

        # Compute time_lag in seconds
        if self._last_timestamp_s is None:
            time_lag = 1.0 / float(self.cfg.fps)  # reasonable first-frame lag
        else:
            time_lag = ts - self._last_timestamp_s
            # Guard against weird non-monotonic clocks
            if time_lag <= 0:
                time_lag = 1.0 / float(self.cfg.fps)

        self._last_timestamp_s = ts
        self._frame_counter += 1

        # Convert dets to CBMOT format
        results = self._detections_to_cbmot_results(detections.dets)

        # CBMOT behavior: if no detections, it clears tracks and returns []
        # This is fine; we just output empty for that frame.
        outs = self._tracker.step_centertrack(
            results=results,
            annotated_data=None,
            time_lag=float(time_lag),
            version="v1.0-test",   # only relevant for train_data; we always set train_data=False
            train_data=False,
        )

        # outs is list of tracklets dicts
        if outs is None:
            return FrameData(frame_id=frame_id, dets=[])

        out_dets: List[Detection] = []
        for trk in outs:
            # Optional: CBMOT main.py filters inactive tracks:
            #   if 'active' in item and item['active'] < min_hits: continue
            # We'll mirror that to match their default output.
            if "active" in trk:
                try:
                    if int(trk["active"]) < int(self.cfg.min_hits):
                        continue
                except Exception:
                    pass

            # Also apply class filter at output if desired
            if self.cfg.track_class is not None:
                name = trk.get("detection_name", None) or trk.get("tracking_name", None)
                if name is not None and str(name) != self.cfg.track_class:
                    continue

            out_dets.append(self._tracklet_to_detection(frame_id, trk))

        return FrameData(frame_id=frame_id, dets=out_dets)
