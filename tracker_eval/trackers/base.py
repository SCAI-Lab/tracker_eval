# tracker_eval/trackers/base.py

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from tracker_eval.common.types import Detection, FrameData


@dataclass(frozen=True)
class TrackerInfo:
    """
    Metadata describing a tracker instance.
    Useful for logging and for naming output folders.
    """
    name: str
    version: str = "0.0"
    description: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackerTiming:
    """
    Simple runtime accounting for FPS and per-frame latencies.
    Stored per sequence run (reset each sequence).
    """
    frame_times_s: List[float] = field(default_factory=list)
    num_frames: int = 0

    def add(self, dt_s: float) -> None:
        self.frame_times_s.append(float(dt_s))
        self.num_frames += 1

    @property
    def total_time_s(self) -> float:
        return float(sum(self.frame_times_s))

    @property
    def mean_time_s(self) -> float:
        if not self.frame_times_s:
            return 0.0
        return float(sum(self.frame_times_s) / len(self.frame_times_s))

    @property
    def fps(self) -> float:
        total = self.total_time_s
        if total <= 0.0:
            return 0.0
        return float(self.num_frames / total)

    @property
    def p95_time_s(self) -> float:
        if not self.frame_times_s:
            return 0.0
        xs = sorted(self.frame_times_s)
        idx = int(round(0.95 * (len(xs) - 1)))
        return float(xs[idx])


@dataclass(frozen=True)
class TrackerRunConfig:
    """
    Runner-level config (not algorithm config).
    """
    enforce_unique_ids_per_frame: bool = True
    # If True, we verify that output IDs are unique within each frame (TrackEval requirement)
    # and throw on duplicates early.


class TrackerBase(ABC):
    """
    Base class for all trackers in tracker_eval.

    Philosophy:
      - Input to a tracker per frame is a list of detections (Detection objects) or FrameData.
      - Output per frame is FrameData containing *tracked* Detection objects
        where Detection.track_id is a persistent integer ID.
      - This class also provides timing hooks to compute FPS etc.

    Minimal methods subclasses must implement:
      - reset_sequence(seq_name)
      - step(frame_id, detections, timestamp=None) -> FrameData

    Optional:
      - close() for releasing GPU memory etc.

    Expected calling pattern (runner will do this):
      tracker.reset_sequence(seq)
      for each frame:
         tracks_frame = tracker.step(frame_id, dets_frame)
      tracker.close()  (once at program end, optional)
    """

    def __init__(
        self,
        info: TrackerInfo,
        *,
        run_cfg: Optional[TrackerRunConfig] = None,
    ) -> None:
        self.info = info
        self.run_cfg = run_cfg or TrackerRunConfig()

        self._seq_name: Optional[str] = None
        self._timing = TrackerTiming()

    # ---------------------------
    # Lifecycle
    # ---------------------------

    @property
    def name(self) -> str:
        return self.info.name
    

    def reset_sequence(self, seq_name: str) -> None:
        """
        Public reset wrapper used by runners.
        Resets timing and allows tracker-specific reset.
        """
        self._seq_name = str(seq_name)
        self._timing = TrackerTiming()
        self._reset_sequence_impl(seq_name=str(seq_name))


    @abstractmethod
    def _reset_sequence_impl(self, seq_name: str) -> None:
        """
        Tracker-specific state reset.
        Called once per sequence.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Optional cleanup hook.
        Override if you allocate GPU memory / torch models / open files, etc.
        """
        # default: no-op
        return

    # ---------------------------
    # Main step API
    # ---------------------------

    def step(
        self,
        frame_id: str,
        detections: Union[FrameData, Sequence[Detection]],
        *,
        timestamp: Optional[float] = None,
    ) -> FrameData:
        """
        One tracker step.

        Parameters
        ----------
        frame_id:
            Typically "000123.pcd" (JRDB frame key used in your JSONs).
        detections:
            Either FrameData or list of Detection. These are per-frame detections
            (no persistent IDs required; can be -1).
        timestamp:
            Optional timestamp in seconds. Not required for JRDB evaluation,
            but can be useful for motion models.

        Returns
        -------
        FrameData:
            FrameData(frame_id, dets=[Detection(track_id=...), ...]) where track_id
            must be unique within this frame.
        """
        if isinstance(detections, FrameData):
            det_frame = detections
        else:
            det_frame = FrameData(frame_id=str(frame_id), dets=list(detections))

        t0 = time.perf_counter()
        out = self._step_impl(frame_id=str(frame_id), detections=det_frame, timestamp=timestamp)
        dt = time.perf_counter() - t0

        self._timing.add(dt)

        if self.run_cfg.enforce_unique_ids_per_frame:
            self._assert_unique_track_ids(out)

        return out

    @abstractmethod
    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        """
        Tracker-specific step implementation.
        Must return FrameData with persistent IDs in Detection.track_id.
        """
        raise NotImplementedError

    # ---------------------------
    # Stats / reporting
    # ---------------------------

    def get_timing(self) -> TrackerTiming:
        """
        Per-sequence timing stats (resets every reset_sequence()).
        """
        return self._timing

    def get_run_summary(self) -> Dict[str, Any]:
        """
        Convenience dict for logging / saving.
        """
        return {
            "tracker": {
                "name": self.info.name,
                "version": self.info.version,
                "description": self.info.description,
                "extra": dict(self.info.extra),
            },
            "sequence": self._seq_name,
            "timing": {
                "num_frames": self._timing.num_frames,
                "total_time_s": self._timing.total_time_s,
                "mean_time_s": self._timing.mean_time_s,
                "p95_time_s": self._timing.p95_time_s,
                "fps": self._timing.fps,
            },
        }

    # ---------------------------
    # Validation helpers
    # ---------------------------

    @staticmethod
    def _assert_unique_track_ids(frame: FrameData) -> None:
        ids = [int(d.track_id) for d in frame.dets]
        if len(ids) != len(set(ids)):
            seen = set()
            dup = []
            for i in ids:
                if i in seen:
                    dup.append(i)
                seen.add(i)
            raise ValueError(
                f"Tracker produced duplicate track_id(s) within frame '{frame.frame_id}': "
                f"{sorted(set(dup))}. TrackEval requires IDs to be unique per frame."
            )


# ---------------------------
# A minimal baseline tracker
# ---------------------------

class PassthroughDetectionsAsTracks(TrackerBase):
    """
    Useful for debugging I/O:
      - assigns a new unique ID to each detection each frame (no temporal association)
      - This will perform terribly on ID metrics (as expected), but it's great for sanity checks.

    Output IDs are unique per frame, but not persistent across frames.
    """

    def __init__(
        self,
        *,
        name: str = "passthrough_per_frame",
        version: str = "1.0",
        run_cfg: Optional[TrackerRunConfig] = None,
    ) -> None:
        super().__init__(TrackerInfo(name=name, version=version, description="No tracking; new IDs every frame."),
                         run_cfg=run_cfg)
        self._next_id: int = 1

    def _reset_sequence_impl(self, seq_name: str) -> None:
        self._next_id = 1

    def _step_impl(
        self,
        frame_id: str,
        detections: FrameData,
        timestamp: Optional[float],
    ) -> FrameData:
        out_dets: List[Detection] = []
        for det in detections.dets:
            # create a new Detection with a fresh track_id
            out_dets.append(
                Detection(
                    frame_id=frame_id,
                    track_id=self._next_id,
                    box=det.box,
                    score=det.score,
                    label=det.label,
                    raw_label_id=det.raw_label_id,
                )
            )
            self._next_id += 1
        return FrameData(frame_id=frame_id, dets=out_dets)
