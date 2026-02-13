# tracker_eval/trackers/gnnpmbtracker_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig


@dataclass(frozen=True)
class GNNPMBConfig:
    """
    Adapter config for GNN-PMB (PMBM-GNN) tracker.

    parameters_path:
        Path to the JSON parameters file used by the upstream repo
        (the one consumed by readout_parameters()).

    classification:
        Object class to track. Your repo's parameters dict must contain this key.

    use_nms:
        Whether to run the repo's NMS before tracking.

    fps:
        Default frame rate used to compute dt when runner does not provide timestamps.
        JRDB is 15 Hz, so default dt is 1/15.

    giou_gating:
        Passed to gnnpmb_filter.update(..., giou_gating=...).
        In upstream run script, pedestrian often uses -0.5.

    ped_empty_meas_extract_thr:
        Upstream special-case: if classification == 'pedestrian' and Z_k is empty,
        they call extractStates_with_custom_thr(filter_updated, thr=0.7).
    """
    parameters_path: str = "/home/scai/trackers/GnnPmbTracker/configs/gnnpmb_parameters.json"
    classification: str = "pedestrian"
    use_nms: bool = True

    fps: float = 15.0  # JRDB default

    giou_gating: float = -0.5
    ped_empty_meas_extract_thr: float = 0.7


class GNNPMBAdapter(TrackerBase):
    """
    Adapter for the GNN-PMB (PMBM-GNN) tracker in your repo.

    Upstream modules used:
      - trackers/PMBMGNN/PMBMGNN_Filter_Point_Target.py   (PMBMGNN_Filter)
      - trackers/PMBMGNN/util.py                         (gen_filter_model)
      - utils/utils.py                                   (nms, readout_parameters, gen_measurement_of_this_class)

    This adapter:
      - converts tracker_eval Detection(Box3D) -> upstream Z_k dicts
      - runs predict/update/prune/extract each frame
      - converts extracted states -> FrameData with persistent integer track_id

    Assumptions:
      - Input Detection.box is Box3D(cx,cy,cz,l,w,h,rot_z) with rot_z=yaw about +z.
      - Upstream expects nuScenes-like dicts with:
          translation: [x,y,z]
          size: [w,l,h]   (nuScenes Box.wlh ordering used in your utils)
          rotation: [qw,qx,qy,qz] yaw-only quaternion
      - Ego motion is not provided in tracker_eval inputs; we pass ego=[0,0].
    """

    def __init__(
        self,
        *,
        cfg: GNNPMBConfig,
        name: str = "gnnpmb",
        version: str = "0.1",
        run_cfg: Optional[TrackerRunConfig] = None,
    ) -> None:
        super().__init__(
            TrackerInfo(
                name=name,
                version=version,
                description="GNN-PMB adapter (PMBM-GNN) wrapping trackers/PMBMGNN with per-frame Detection I/O.",
                extra={
                    "classification": cfg.classification,
                    "use_nms": cfg.use_nms,
                    "parameters_path": cfg.parameters_path,
                    "fps": cfg.fps,
                },
            ),
            run_cfg=run_cfg,
        )
        self.cfg = cfg

        # lazy-loaded repo modules
        self._gnn_filter_mod: Optional[Any] = None
        self._util_mod: Optional[Any] = None
        self._utils_mod: Optional[Any] = None

        # runtime state per sequence
        self._tracker: Optional[Any] = None
        self._filter_model: Optional[Dict[str, Any]] = None
        self._filter_pruned: Optional[Any] = None

        self._prev_timestamp_s: Optional[float] = None
        self._frame_counter: int = 0

        # parameters parsed from JSON
        self._params: Optional[Dict[str, Any]] = None
        self._algo: Dict[str, Any] = {}

    # ---------------------------
    # Imports / helpers
    # ---------------------------

    @staticmethod
    def _lazy_imports() -> Tuple[Any, Any, Any]:
        """
        Returns:
          PMBMGNN_Filter_Point_Target module,
          trackers.PMBMGNN.util module,
          utils.utils module
        """
        from trackers.PMBMGNN import PMBMGNN_Filter_Point_Target as pmbmgnn_tracker  # type: ignore
        from trackers.PMBMGNN import util as pmbmgnn_util  # type: ignore
        from utils import utils as repo_utils  # type: ignore
        return pmbmgnn_tracker, pmbmgnn_util, repo_utils

    @staticmethod
    def _yaw_to_quat_z(yaw: float) -> List[float]:
        """
        Convert yaw (about z axis) to quaternion [qw, qx, qy, qz].
        """
        half = 0.5 * float(yaw)
        return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]

    @staticmethod
    def _quat_to_yaw(q: Sequence[float]) -> float:
        """
        Convert quaternion [qw, qx, qy, qz] to yaw about z.
        """
        qw, qx, qy, qz = [float(x) for x in q]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _dt_from_inputs(self, timestamp: Optional[float]) -> float:
        """
        Determine dt (seconds) for the tracker.

        - If runner provides timestamp: dt = timestamp - prev_timestamp
        - Else: dt = 1/fps for frames after the first, 0 for the first frame
        """
        if timestamp is None:
            if self._frame_counter <= 0:
                return 0.0
            if self.cfg.fps <= 0:
                raise ValueError(f"cfg.fps must be > 0, got {self.cfg.fps}")
            return float(1.0 / float(self.cfg.fps))

        if self._prev_timestamp_s is None:
            self._prev_timestamp_s = float(timestamp)
            return 0.0

        dt = float(timestamp - self._prev_timestamp_s)
        if dt < 0:
            raise ValueError(f"Non-monotonic timestamp: prev={self._prev_timestamp_s}, cur={timestamp}, dt={dt}")
        self._prev_timestamp_s = float(timestamp)
        return dt

    def _dets_to_measurements(self, dets: Sequence[Detection]) -> List[Dict[str, Any]]:
        """
        Convert tracker_eval detections into upstream measurement dicts (nuScenes-like).
        """
        out: List[Dict[str, Any]] = []
        for d in dets:
            b = d.box
            score = float(d.score) if d.score is not None else 1.0

            out.append(
                {
                    "translation": [float(b.cx), float(b.cy), float(b.cz)],
                    # IMPORTANT: nuScenes ordering used in your repo utilities is wlh => [w, l, h]
                    "size": [float(b.w), float(b.l), float(b.h)],
                    "rotation": self._yaw_to_quat_z(b.rot_z),
                    "velocity": [0.0, 0.0],
                    "detection_name": self.cfg.classification,
                    "detection_score": score,
                }
            )
        return out

    def _states_to_detections(self, frame_id: str, est: Dict[str, Any]) -> List[Detection]:
        """
        Convert upstream extracted states dict -> list of tracker_eval Detection.
        """
        out: List[Detection] = []

        means = est.get("mean", [])
        elevations = est.get("elevation", [])
        sizes = est.get("size", [])
        rotations = est.get("rotation", [])
        ids = est.get("id", [])
        scores = est.get("detection_score", [])

        n = len(means)
        if not (len(elevations) == len(sizes) == len(rotations) == len(ids) == n):
            raise RuntimeError(
                "GNNPMB extractStates returned inconsistent field lengths: "
                f"mean={len(means)}, elevation={len(elevations)}, size={len(sizes)}, "
                f"rotation={len(rotations)}, id={len(ids)}"
            )

        for i in range(n):
            tid = int(ids[i])

            mean = np.asarray(means[i]).reshape(-1)
            if mean.shape[0] != 4:
                raise RuntimeError(f"GNNPMB returned mean with unexpected shape: {mean.shape}")

            x = float(mean[0])
            y = float(mean[1])
            z = float(elevations[i])

            size = sizes[i]
            if len(size) != 3:
                raise RuntimeError(f"GNNPMB returned size with unexpected length: {len(size)}")
            w = float(size[0])
            l = float(size[1])
            h = float(size[2])

            rot = rotations[i]
            yaw = self._quat_to_yaw(rot)

            score: Optional[float] = float(scores[i]) if i < len(scores) else None

            out.append(
                Detection(
                    frame_id=frame_id,
                    track_id=tid,
                    box=Box3D(cx=x, cy=y, cz=z, l=l, w=w, h=h, rot_z=yaw),
                    score=score,
                    label=self.cfg.classification,
                    raw_label_id=None,
                )
            )

        return out

    # ---------------------------
    # TrackerBase overrides
    # ---------------------------

    def _reset_sequence_impl(self, seq_name: str) -> None:
        """
        Load parameters, construct filter model, and create a new PMBMGNN_Filter.
        """
        pmbmgnn_tracker, pmbmgnn_util, repo_utils = self._lazy_imports()
        self._gnn_filter_mod = pmbmgnn_tracker
        self._util_mod = pmbmgnn_util
        self._utils_mod = repo_utils

        # Load params JSON
        import json

        with open(self.cfg.parameters_path, "r") as f:
            self._params = json.load(f)

        if self.cfg.classification not in self._params:
            raise KeyError(
                f"Classification '{self.cfg.classification}' not found in parameters file "
                f"'{self.cfg.parameters_path}'. Keys: {list(self._params.keys())}"
            )

        # Read class-specific parameters
        (
            birth_rate,
            P_s,
            P_d,
            use_ds_as_pd,
            clutter_rate,
            bernoulli_gating,
            extraction_thr,
            ber_thr,
            poi_thr,
            eB_thr,
            detection_score_thr,
            nms_score,
            confidence_score,
            P_init,
        ) = repo_utils.readout_parameters(self.cfg.classification, self._params)

        self._algo = {
            "birth_rate": float(birth_rate),
            "P_s": float(P_s),
            "P_d": float(P_d),
            "use_ds_as_pd": bool(use_ds_as_pd),
            "clutter_rate": float(clutter_rate),
            "bernoulli_gating": float(bernoulli_gating),
            "extraction_thr": float(extraction_thr),
            "ber_thr": float(ber_thr),
            "poi_thr": float(poi_thr),
            "eB_thr": float(eB_thr),
            "detection_score_thr": float(detection_score_thr),
            "nms_score": float(nms_score),
            "confidence_score": float(confidence_score),
            "P_init": float(P_init),
        }

        # Build filter model (matches trackers/PMBMGNN/util.py)
        self._filter_model = pmbmgnn_util.gen_filter_model(
            average_number_of_clutter_per_frame=self._algo["clutter_rate"],
            p_S=self._algo["P_s"],
            p_D=self._algo["P_d"],
            classification=self.cfg.classification,
            extraction_thr=self._algo["extraction_thr"],
            ber_thr=self._algo["ber_thr"],
            poi_thr=self._algo["poi_thr"],
            eB_thr=self._algo["eB_thr"],
            ber_gating=self._algo["bernoulli_gating"],
            use_ds_as_pd=self._algo["use_ds_as_pd"],
            P_init=self._algo["P_init"],
            use_giou=False,
            gating_mode="mahalanobis",
        )

        # Create tracker instance
        self._tracker = pmbmgnn_tracker.PMBMGNN_Filter(self._filter_model)

        # Reset sequence runtime state
        self._filter_pruned = None
        self._prev_timestamp_s = None
        self._frame_counter = 0

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        """
        One frame: build Z_k, predict/update/prune, extract states, return FrameData.
        """
        if self._tracker is None or self._filter_model is None or self._utils_mod is None:
            raise RuntimeError("GNNPMBAdapter used before reset_sequence().")

        repo_utils = self._utils_mod

        dt_s = self._dt_from_inputs(timestamp)

        # Build measurement list (nuScenes-like dicts)
        meas_all = self._dets_to_measurements(detections.dets)

        # Keep only this classification + score threshold (repo helper)
        Z_k_all = repo_utils.gen_measurement_of_this_class(
            self._algo["detection_score_thr"], meas_all, self.cfg.classification
        )

        # NMS (optional)
        if self.cfg.use_nms and len(Z_k_all) > 0:
            keep_idx = repo_utils.nms(Z_k_all, threshold=self._algo["nms_score"])
            Z_k = [Z_k_all[i] for i in keep_idx]
        else:
            Z_k = Z_k_all

        # Predict
        if self._filter_pruned is None:
            filter_predicted = self._tracker.predict_initial_step(Z_k, self._algo["birth_rate"])
        else:
            ego_info = [0.0, 0.0]  # no ego in tracker_eval I/O; keep stable
            filter_predicted = self._tracker.predict(
                ego_info,
                dt_s,
                self._filter_pruned,
                Z_k,
                self._algo["birth_rate"],
            )

        # Update
        filter_updated = self._tracker.update(
            Z_k,
            filter_predicted,
            confidence_score=self._algo["confidence_score"],
            giou_gating=float(self.cfg.giou_gating),
        )

        # Extract
        if self.cfg.classification == "pedestrian":
            if len(Z_k) == 0:
                est = self._tracker.extractStates_with_custom_thr(
                    filter_updated, float(self.cfg.ped_empty_meas_extract_thr)
                )
            else:
                est = self._tracker.extractStates(filter_updated)
        else:
            est = self._tracker.extractStates(filter_updated)

        # Prune for next step
        self._filter_pruned = self._tracker.prune(filter_updated)

        self._frame_counter += 1

        out_dets = self._states_to_detections(frame_id=frame_id, est=est)
        return FrameData(frame_id=frame_id, dets=out_dets)
