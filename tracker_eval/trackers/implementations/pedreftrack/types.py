# tracker_eval/common/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union


Number = Union[int, float]


@dataclass(frozen=True)
class Box3D:
    """
    JRDB box convention:
        cx, cy, cz, l, w, h, rot_z

    Units: meters and radians.
    Coordinate frame: JRDB "base" (x forward, y left, z up) per your pipeline.
    """
    cx: float
    cy: float
    cz: float
    l: float
    w: float
    h: float
    rot_z: float

    def as_list(self) -> List[float]:
        return [self.cx, self.cy, self.cz, self.l, self.w, self.h, self.rot_z]

    def as_dict(self) -> Dict[str, float]:
        return {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "cz": float(self.cz),
            "l": float(self.l),
            "w": float(self.w),
            "h": float(self.h),
            "rot_z": float(self.rot_z),
        }

    @staticmethod
    def from_list(vals: Iterable[Number]) -> "Box3D":
        v = list(vals)
        if len(v) != 7:
            raise ValueError(f"Box3D.from_list expects 7 values, got {len(v)}")
        return Box3D(
            cx=float(v[0]),
            cy=float(v[1]),
            cz=float(v[2]),
            l=float(v[3]),
            w=float(v[4]),
            h=float(v[5]),
            rot_z=float(v[6]),
        )

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Box3D":
        required = ["cx", "cy", "cz", "l", "w", "h", "rot_z"]
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"Box3D.from_dict missing keys: {missing}")
        return Box3D(
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            cz=float(d["cz"]),
            l=float(d["l"]),
            w=float(d["w"]),
            h=float(d["h"]),
            rot_z=float(d["rot_z"]),
        )

    @staticmethod
    def from_any(obj: Any) -> "Box3D":
        """
        Accepts:
          - list/tuple length 7
          - dict with cx/cy/cz/l/w/h/rot_z
        """
        if isinstance(obj, (list, tuple)):
            return Box3D.from_list(obj)
        if isinstance(obj, dict):
            return Box3D.from_dict(obj)
        raise TypeError(f"Unsupported box type: {type(obj)}")


@dataclass(frozen=True)
class Detection:
    """
    A single object instance in a frame.

    track_id:
      - For GT: parsed from label_id (e.g., 'pedestrian:14' -> 14)
      - For tracker output: tracker-assigned integer ID
      - For detections (no tracking): can be -1
    """
    frame_id: str
    track_id: int
    box: Box3D
    score: Optional[float] = None
    label: str = "pedestrian"     # class name, fixed for now
    raw_label_id: Optional[str] = None  # original string label_id if available

    def as_json_dict(self, box_as_dict: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "track_id": int(self.track_id),
            "label": self.label,
        }
        out["box"] = self.box.as_dict() if box_as_dict else self.box.as_list()
        if self.score is not None:
            out["score"] = float(self.score)
        if self.raw_label_id is not None:
            out["raw_label_id"] = self.raw_label_id
        return out


@dataclass
class FrameData:
    frame_id: str
    dets: List[Detection]


@dataclass
class SequenceData:
    """
    Optional container for future stages (matching/metrics).
    """
    seq_name: str
    frames: List[str]                      # ordered frame ids
    gt_by_frame: Dict[str, FrameData]      # frame -> GT detections
    pred_by_frame: Dict[str, FrameData]    # frame -> predicted detections (detections or tracks)


def parse_label_id(label_id: str) -> Tuple[str, int]:
    """
    JRDB label_id format is typically '<class>:<int>', e.g. 'pedestrian:14'.
    Returns: (class_name, numeric_id)

    If parsing fails, class_name is label_id prefix or 'unknown', numeric_id = -1.
    """
    if not isinstance(label_id, str) or ":" not in label_id:
        return ("unknown", -1)
    cls, num = label_id.split(":", 1)
    try:
        return (cls, int(num))
    except Exception:
        return (cls, -1)


def frame_sort_key(frame_id: str) -> Tuple[int, str]:
    """
    Sort frame ids like '000123.pcd' numerically if possible; else lexicographically.
    """
    # Extract last run of digits
    digits = ""
    for ch in frame_id:
        if ch.isdigit():
            digits += ch
    if digits:
        try:
            return (int(digits), frame_id)
        except Exception:
            pass
    return (10**18, frame_id)
