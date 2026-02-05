# tracker_eval/trackers/fastpoly_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig


@dataclass(frozen=True)
class FastPolyConfig:
    """
    Minimal adapter config for FastPoly.

    We keep it intentionally small: you can pass the FastPoly YAML-loaded dict
    directly, and we only override the bare minimum fields needed for single-seq usage.
    """
    # The FastPoly YAML config dict (already loaded with yaml.safe_load)
    config: Dict[str, Any]

    # Sequence id used by FastPoly internal outputs
    # (FastPoly wants data_info['seq_id'] to exist)
    seq_id: int = 0

    # Whether detections include velocity (we do NOT in JRDB)
    has_velo: bool = False

    # If True, treat every frame as "key frame" if FastPoly uses it
    # (some Poly-MOT forks need is_key_frame)
    is_key_frame: bool = True

    # If you want to force a specific class label
    # (otherwise we use 'pedestrian' from NUSC_CONSTANT)
    force_class_label: Optional[int] = None

    # If True, we will add a small increasing timestamp (sec) derived from frame index
    # (FastPoly uses frame_id as timestamp in some places; providing numeric helps sometimes)
    use_numeric_frame_id: bool = True


class FastPolyAdapter(TrackerBase):
    """
    Adapter for FastPoly (Poly-MOT derived).

    FastPoly expects per-frame `data_info` dict with fields including:
      - is_first_frame (bool)
      - frame_id (timestamp-like)
      - seq_id (int)
      - no_dets (bool)
      - det_num (int)
      - np_dets: np.ndarray [det_num, 14]
          format: [x,y,z,w,l,h,vx,vy,qw,qx,qy,qz,score,class_label]
      - box_dets: np.ndarray of NuscBox objects [det_num]
      - np_dets_bottom_corners: np.ndarray [det_num, 4, 2]
      - np_dets_norm_corners: np.ndarray [det_num, 4] or similar (FastPoly util returns it)

    We build those using FastPoly's own conversion:
      pre_processing.nusc_data_conversion.arraydet2box(np_dets)
    """

    def __init__(
        self,
        *,
        cfg: FastPolyConfig,
        name: str = "fastpoly",
        version: str = "0.1",
        run_cfg: Optional[TrackerRunConfig] = None,
    ) -> None:
        super().__init__(
            TrackerInfo(
                name=name,
                version=version,
                description="FastPoly adapter (Poly-MOT based), using FastPoly native data_info + arraydet2box.",
            ),
            run_cfg=run_cfg,
        )
        self.cfg = cfg

        self._tracker: Optional[Any] = None
        self._seq_name: Optional[str] = None
        self._seq_id: int = int(cfg.seq_id)

        # For generating numeric frame_id if desired
        self._frame_counter: int = 0

        # Cache class label lookup
        self._ped_label: Optional[int] = None

    # ---------------------------
    # FastPoly imports / helpers
    # ---------------------------

    @staticmethod
    def _lazy_imports() -> Tuple[Any, Any]:
        """
        Returns:
          Tracker class, arraydet2box function
        """
        from tracking.nusc_tracker import Tracker  # type: ignore
        from pre_processing.nusc_data_conversion import arraydet2box  # type: ignore
        return Tracker, arraydet2box

    def _get_ped_label(self) -> int:
        """
        FastPoly class labels:
          CLASS_SEG_TO_STR_CLASS: {'pedestrian': 4, ...}
        """
        if self.cfg.force_class_label is not None:
            return int(self.cfg.force_class_label)

        if self._ped_label is None:
            from data.script.NUSC_CONSTANT import CLASS_SEG_TO_STR_CLASS  # type: ignore
            self._ped_label = int(CLASS_SEG_TO_STR_CLASS["pedestrian"])
        return int(self._ped_label)

    @staticmethod
    def _yaw_to_quat_z(yaw: float) -> Tuple[float, float, float, float]:
        """
        Convert yaw (about z axis) to quaternion (w, x, y, z).
        """
        half = 0.5 * float(yaw)
        qw = float(np.cos(half))
        qx = 0.0
        qy = 0.0
        qz = float(np.sin(half))
        return qw, qx, qy, qz

    def _frame_id_numeric(self, frame_id: str) -> int:
        """
        Convert "000123.pcd" -> 123
        """
        s = str(frame_id)
        if "." in s:
            s = s.split(".", 1)[0]
        s = s.strip()
        if s == "":
            return self._frame_counter
        try:
            return int(s)
        except Exception:
            # fallback not desired; but this is not algorithmic fallback,
            # it's just a stable numeric key if parsing fails
            return self._frame_counter

    def _detections_to_np_dets(self, dets: Sequence[Detection]) -> np.ndarray:
        """
        Build np_dets [N, 14]:
          [x,y,z,w,l,h,vx,vy,qw,qx,qy,qz,score,class_label]
        """
        n = len(dets)
        out = np.zeros((n, 14), dtype=np.float32)
        cls_label = float(self._get_ped_label())

        for i, det in enumerate(dets):
            b = det.box
            score = float(det.score) if det.score is not None else 1.0

            qw, qx, qy, qz = self._yaw_to_quat_z(b.rot_z)

            # NOTE: FastPoly expects size as (w, l, h) in np_dets columns [3,4,5]
            out[i, 0] = float(b.cx)
            out[i, 1] = float(b.cy)
            out[i, 2] = float(b.cz)
            out[i, 3] = float(b.w)
            out[i, 4] = float(b.l)
            out[i, 5] = float(b.h)

            # velocities: not available in JRDB detections
            out[i, 6] = 0.0
            out[i, 7] = 0.0

            # quaternion (w,x,y,z)
            out[i, 8] = float(qw)
            out[i, 9] = float(qx)
            out[i, 10] = float(qy)
            out[i, 11] = float(qz)

            out[i, 12] = float(score)
            out[i, 13] = float(cls_label)

        return out

    def _build_data_info(
        self,
        *,
        frame_id: str,
        dets: Sequence[Detection],
        is_first_frame: bool,
    ) -> Dict[str, Any]:
        """
        Construct FastPoly data_info dict, using FastPoly's arraydet2box to build geometry.
        """
        _, arraydet2box = self._lazy_imports()

        det_num = len(dets)

        # FastPoly uses frame_id as timestamp-like in state_predict/update
        # We'll provide numeric if configured, otherwise original string.
        if self.cfg.use_numeric_frame_id:
            frame_id_value: Any = self._frame_id_numeric(frame_id)
        else:
            frame_id_value = str(frame_id)

        data_info: Dict[str, Any] = {
            "is_first_frame": bool(is_first_frame),
            "frame_id": frame_id_value,
            "seq_id": int(self._seq_id),
            "has_velo": bool(self.cfg.has_velo),
            "no_dets": bool(det_num == 0),
        }

        if det_num == 0:
            # Must still provide expected arrays with correct shapes
            data_info.update(
                {
                    "det_num": 0,
                    "np_dets": np.zeros((0, 14), dtype=np.float32),
                    "box_dets": np.zeros((0,), dtype=object),
                    "np_dets_bottom_corners": np.zeros((0, 4, 2), dtype=np.float32),
                    "np_dets_norm_corners": np.zeros((0, 4), dtype=np.float32),
                }
            )
            return data_info

        np_dets = self._detections_to_np_dets(dets)

        # Use FastPoly native converter to construct NuscBox and corners.
        # IMPORTANT: arraydet2box is designed for (N,14) and returns arrays of length N.
        box_dets, bottom_corners, norm_corners = arraydet2box(np_dets)

        # Strong sanity checks (no silent fallback)
        if box_dets is None or len(box_dets) != det_num:
            raise RuntimeError(f"FastPoly arraydet2box returned {0 if box_dets is None else len(box_dets)} boxes for {det_num} detections.")
        if bottom_corners is None or len(bottom_corners) != det_num:
            raise RuntimeError(f"FastPoly arraydet2box returned invalid bottom_corners for {det_num} detections.")
        if norm_corners is None or len(norm_corners) != det_num:
            raise RuntimeError(f"FastPoly arraydet2box returned invalid norm_corners for {det_num} detections.")

        data_info.update(
            {
                "det_num": int(det_num),
                "np_dets": np.asarray(np_dets, dtype=np.float32),
                "box_dets": np.asarray(box_dets, dtype=object),
                "np_dets_bottom_corners": np.asarray(bottom_corners),
                "np_dets_norm_corners": np.asarray(norm_corners),
            }
        )
        return data_info

    # ---------------------------
    # TrackerBase overrides
    # ---------------------------

    def _reset_sequence_impl(self, seq_name: str) -> None:
        """
        Create a new FastPoly Tracker and reset per sequence.
        """
        Tracker, _ = self._lazy_imports()

        self._seq_name = str(seq_name)
        self._frame_counter = 0

        # Instantiate tracker with provided config dict
        self._tracker = Tracker(self.cfg.config)

        # Make sure internal state reset (FastPoly also does this on first frame,
        # but we keep adapter explicit)
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        """
        Feed one frame into FastPoly and convert its outputs into our FrameData.
        """
        if self._tracker is None:
            raise RuntimeError("FastPolyAdapter used before reset_sequence().")

        is_first = (self._frame_counter == 0)
        self._frame_counter += 1

        data_info = self._build_data_info(
            frame_id=frame_id,
            dets=detections.dets,
            is_first_frame=is_first,
        )

        # Run tracking (in-place updates data_info)
        self._tracker.tracking(data_info)

        # Corner case: no output
        if data_info.get("no_val_track_result", False):
            return FrameData(frame_id=frame_id, dets=[])

        # Parse results.
        # Preferred: np_track_res exists; it's a list of per-track "infos".
        # But shape/content differs across forks; we'll robustly extract:
        #   - tracking id from the last 3 appended fields OR from explicit column
        #   - box from box_track_res if available; else from np_track_res first columns.
        np_track_res = data_info.get("np_track_res", None)
        box_track_res = data_info.get("box_track_res", None)

        out_dets: List[Detection] = []

        # If box_track_res exists, it is often the easiest: list of NuscBox.
        # But we still need track_id for each.
        if np_track_res is None:
            # No numeric results => can't assign IDs
            raise RuntimeError("FastPoly returned no np_track_res; cannot extract track IDs.")

        # Ensure array
        np_res = np.asarray(np_track_res, dtype=np.float32)

        if np_res.ndim != 2:
            raise RuntimeError(f"FastPoly np_track_res unexpected shape: {np_res.shape}")

        # FastPoly code comments:
        #   np_track_res: [num, 17] add 'tracking_id', 'seq_id', 'frame_id'
        # So last 3 columns are [tracking_id, seq_id, frame_id] (in that order).
        if np_res.shape[1] < 3:
            raise RuntimeError(f"FastPoly np_track_res has too few columns: {np_res.shape}")

        tracking_ids = np_res[:, -3].astype(int)

        # Get boxes:
        # - If box_track_res provided: list of nuScenes Box / FastPoly NuscBox per track result
        # - Else: we refuse (no safe fallback).
        if box_track_res is not None:
            boxes_obj = list(box_track_res)
            if len(boxes_obj) != len(tracking_ids):
                raise RuntimeError(
                    f"FastPoly output mismatch: {len(boxes_obj)} boxes vs {len(tracking_ids)} ids"
                )

            for box_obj, tid in zip(boxes_obj, tracking_ids.tolist()):
                # FastPoly's NuscBox is a subclass of nuScenes Box.
                # Geometry lives in:
                #   box_obj.center -> np.array([x, y, z])
                #   box_obj.wlh    -> np.array([w, l, h])
                center = getattr(box_obj, "center", None)
                wlh = getattr(box_obj, "wlh", None)
                if center is None or wlh is None:
                    raise RuntimeError(
                        "FastPoly box_track_res objects must have .center and .wlh (nuScenes Box API). "
                        f"Got type={type(box_obj)}"
                    )

                center = np.asarray(center, dtype=float).reshape(-1)
                wlh = np.asarray(wlh, dtype=float).reshape(-1)
                if center.shape[0] != 3 or wlh.shape[0] != 3:
                    raise RuntimeError(
                        f"Unexpected box.center / box.wlh shapes: center={center.shape}, wlh={wlh.shape}"
                    )

                x, y, z = float(center[0]), float(center[1]), float(center[2])
                w, l, h = float(wlh[0]), float(wlh[1]), float(wlh[2])

                # Yaw: FastPoly's NuscBox sets self.yaw = self.orientation.radians
                if hasattr(box_obj, "yaw"):
                    yaw = float(getattr(box_obj, "yaw"))
                else:
                    # If yaw not present, compute yaw from quaternion about z.
                    orient = getattr(box_obj, "orientation", None)
                    if orient is None or not hasattr(orient, "elements"):
                        raise RuntimeError(
                            "Cannot extract yaw: box has no .yaw and no usable .orientation quaternion."
                        )
                    # Quaternion elements are [w, x, y, z]
                    qw, qx, qy, qz = [float(v) for v in orient.elements]
                    yaw = float(2.0 * np.arctan2(qz, qw))

                # Score: keep None (JRDB toolkit doesn't require it; don't guess)
                score: Optional[float] = None

                out_dets.append(
                    Detection(
                        frame_id=frame_id,
                        track_id=int(tid),
                        box=Box3D(cx=x, cy=y, cz=z, l=l, w=w, h=h, rot_z=yaw),
                        score=score,
                        label="pedestrian",
                        raw_label_id=None,
                    )
                )

        else:
            # If box_track_res absent, we must extract geometry from np_res.
            # We cannot safely guess its columns across forks, so we refuse.
            raise RuntimeError(
                "FastPoly returned no box_track_res; cannot reliably extract 3D boxes."
            )

        return FrameData(frame_id=frame_id, dets=out_dets)
