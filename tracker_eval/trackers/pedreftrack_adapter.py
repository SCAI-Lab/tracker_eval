"""Thin tracker-evaluation adapter for the bundled PedRefTrack core."""

from __future__ import annotations

from typing import Optional, Sequence

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig
from tracker_eval.trackers.implementations.pedreftrack import (
    PEDREFTRACK_MODES,
    PedRefTrack,
    PedRefTrackConfig,
    normalize_pedreftrack_mode,
    pedreftrack_local_name,
)


def _to_tracker_eval(frame: object) -> FrameData:
    frame_id = str(getattr(frame, "frame_id"))
    converted = []
    for det in getattr(frame, "dets"):
        box = det.box
        converted.append(
            Detection(
                frame_id=frame_id,
                track_id=int(det.track_id),
                box=Box3D(
                    cx=float(box.cx),
                    cy=float(box.cy),
                    cz=float(box.cz),
                    l=float(box.l),
                    w=float(box.w),
                    h=float(box.h),
                    rot_z=float(box.rot_z),
                ),
                score=None if det.score is None else float(det.score),
                label=str(det.label),
                raw_label_id=det.raw_label_id,
            )
        )
    return FrameData(frame_id=frame_id, dets=converted)


class PedRefTrackAdapter(TrackerBase):
    """Expose PedRefTrack through the common tracker-evaluation interface."""

    def __init__(
        self,
        *,
        cfg: Optional[PedRefTrackConfig] = None,
        run_cfg: Optional[TrackerRunConfig] = None,
        name: Optional[str] = None,
    ) -> None:
        self.cfg = cfg or PedRefTrackConfig()
        mode = normalize_pedreftrack_mode(self.cfg.mode)
        super().__init__(
            TrackerInfo(
                name=name or pedreftrack_local_name(mode),
                version="1.0",
                description=(
                    "PedRefTrack adapter; two-pass IoU/XY association, "
                    "seconds-based confirmation and adaptive coasting."
                ),
                extra=dict(self.cfg.__dict__),
            ),
            run_cfg=run_cfg,
        )
        self._core = PedRefTrack(cfg=self.cfg)

    def _reset_sequence_impl(self, seq_name: str) -> None:
        self._core.reset_sequence(seq_name)

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        return _to_tracker_eval(
            self._core.step(
                frame_id,
                detections.dets,
                timestamp=timestamp,
            )
        )

    def step_with_gt(
        self,
        frame_id: str,
        detections: Sequence[Detection],
        gt_dets: Optional[Sequence[Detection]] = None,
        *,
        timestamp: Optional[float] = None,
    ) -> FrameData:
        out = _to_tracker_eval(
            self._core.step_with_gt(
                frame_id,
                detections,
                gt_dets,
                timestamp=timestamp,
            )
        )
        if self.run_cfg.enforce_unique_ids_per_frame:
            self._assert_unique_track_ids(out)
        return out


__all__ = [
    "PEDREFTRACK_MODES",
    "PedRefTrackAdapter",
    "PedRefTrackConfig",
    "normalize_pedreftrack_mode",
    "pedreftrack_local_name",
]
