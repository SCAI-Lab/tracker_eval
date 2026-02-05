# tracker_eval/trackers/simpletrack_adapter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import yaml

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig
from mot_3d.data_protos import BBox  


@dataclass(frozen=True)
class SimpleTrackConfig:
    """
    Config for SimpleTrack adapter.

    config_path:
      Path to a SimpleTrack YAML (e.g. configs/nu_configs/giou.yaml).

    Notes:
      - We intentionally do not "tune" beyond using their shipped configs.
      - If you later decide to do minimal tweaks (e.g. score_threshold), do it by editing YAML,
        not via code, to preserve benchmark fairness/reproducibility.
    """
    config_path: str


class SimpleTrackAdapter(TrackerBase):
    """
    SimpleTrack adapter for tracker_eval using the mot_3d library.

    Inputs:
      - FrameData containing detections in JRDB base convention:
        Box3D(cx, cy, cz, l, w, h, rot_z)
      - Detection.score optional

    Outputs:
      - FrameData containing tracked detections with persistent integer IDs.

    Implementation notes:
      - SimpleTrack operates on its own FrameData and BBox types.
      - We provide dets only (pc/ego/aux_info set to None by default).
      - Box conversion: our (x,y,z,l,w,h,yaw) -> SimpleTrack array format.
        SimpleTrack commonly uses [x, y, z, h, w, l, yaw, score] in many pipelines.
      - We attempt to convert in a robust way using mot_3d.data_protos.BBox helpers.
        If your installed mot_3d version differs, you may only need to adjust
        the `_bbox_from_array(...)` / `_array_from_bbox(...)` helpers below.
    """

    def __init__(
        self,
        *,
        cfg: SimpleTrackConfig,
        run_cfg: Optional[TrackerRunConfig] = None,
        name: str = "simpletrack",
        version: str = "mot_3d",
    ) -> None:
        self.cfg = cfg
        info = TrackerInfo(
            name=name,
            version=version,
            description="SimpleTrack adapter (mot_3d.MOTModel) using upstream YAML configs.",
            extra={"config_path": self.cfg.config_path},
        )
        super().__init__(info, run_cfg=run_cfg)

        self._mot = None
        self._configs: Optional[dict] = None
        self._frame_index: int = 0

    # ----------------------------
    # Internal conversion helpers
    # ----------------------------

    @staticmethod
    def _det_to_simpletrack_array(det: Detection) -> np.ndarray:
        """
        SimpleTrack's BBox.array2bbox expects:
        [x, y, z, o, l, w, h] (+ optional score as 8th)
        where:
        o = heading/yaw
        """
        b = det.box
        score = float(det.score) if det.score is not None else 1.0
        return np.array([b.cx, b.cy, b.cz, b.rot_z, b.l, b.w, b.h, score], dtype=np.float32)

    # ----------------------------
    # TrackerBase required methods
    # ----------------------------

    def _reset_sequence_impl(self, seq_name: str) -> None:
        self._frame_index = 0

        # Lazy import so tracker_eval can be imported even without mot_3d installed.
        try:
            from mot_3d.mot import MOTModel  # type: ignore
            from mot_3d.frame_data import FrameData as STFrameData  # noqa: F401  # type: ignore
        except Exception as e:
            raise ImportError(
                "Could not import SimpleTrack (mot_3d). "
                "Install it with: pip install -e /path/to/SimpleTrack"
            ) from e

        # Load YAML config exactly as provided by upstream
        with open(self.cfg.config_path, "r") as f:
            configs = yaml.load(f, Loader=yaml.Loader)

        if not isinstance(configs, dict):
            raise ValueError(f"SimpleTrack YAML did not parse into a dict: {self.cfg.config_path}")

        self._configs = configs
        self._mot = MOTModel(configs)

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        if self._mot is None or self._configs is None:
            raise RuntimeError("SimpleTrackAdapter: tracker is not initialized. Did you call reset_sequence()?")

        from mot_3d.frame_data import FrameData as STFrameData  # type: ignore

        # IMPORTANT: mot_3d.frame_data.FrameData expects dets as arrays, not BBox objects.
        st_dets: List[np.ndarray] = []
        for det in detections.dets:
            st_dets.append(self._det_to_simpletrack_array(det))


        # Build SimpleTrack FrameData
        # We provide dets; other fields are optional depending on config.
        # If config expects pc/nms, mot_3d may use dets already pre-NMSed; we keep it as-is.
        st_frame = STFrameData(
            dets=st_dets,
            ego=None,
            pc=None,
            det_types=[2] * len(st_dets),          # constant class token; fine for your benchmark
            aux_info={"is_key_frame": True},       # REQUIRED by MOTModel.frame_mot()
            time_stamp=float(self._frame_index) if timestamp is None else float(timestamp),
        )

        # Run tracking
        results = self._mot.frame_mot(st_frame)
        self._frame_index += 1

        # results is typically list of tuples: (bbox, id, state, type)
        out_dets: List[Detection] = []
        for trk in results:
            if not isinstance(trk, (list, tuple)) or len(trk) < 2:
                continue

            bbox_obj = trk[0]          # mot_3d.data_protos.BBox instance
            track_id = int(trk[1])

            # SimpleTrack's BBox.bbox2array returns:
            #   [x, y, z, o, l, w, h] (+ optional score as 8th)
            arr = np.asarray(BBox.bbox2array(bbox_obj), dtype=np.float32).reshape(-1)

            if arr.shape[0] < 7:
                raise ValueError(f"Unexpected bbox array shape from SimpleTrack: {arr.shape}")

            x = float(arr[0])
            y = float(arr[1])
            z = float(arr[2])
            yaw = float(arr[3])   # 'o' in their code
            l = float(arr[4])
            w = float(arr[5])
            h = float(arr[6])
            score = float(arr[7]) if arr.shape[0] >= 8 else None

            box = Box3D(cx=x, cy=y, cz=z, l=l, w=w, h=h, rot_z=yaw)

            out_dets.append(
                Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    box=box,
                    score=score,
                    label="pedestrian",
                    raw_label_id=None,
                )
            )

        return FrameData(frame_id=frame_id, dets=out_dets)
