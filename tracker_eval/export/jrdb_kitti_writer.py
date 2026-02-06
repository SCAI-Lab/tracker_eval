# tracker_eval/export/jrdb_kitti_writer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------
# Public datatypes (lightweight)
# -----------------------------

@dataclass(frozen=True)
class TrackRow3D:
    """One tracked object at one frame in *internal* convention.

    Internal convention expected by this writer:
      - box7_center: (cx, cy, cz, l, w, h, rot_z)
        where (cx,cy,cz) is center, z-up, x-forward, y-left (JRDB base)
      - score: optional confidence (float). If missing, writer uses 1.0.
    """
    track_id: int
    box7: np.ndarray  # shape (7,) in internal center convention
    score: Optional[float] = 1.0


FrameKey = Union[int, str]
TracksByFrame = Mapping[FrameKey, Sequence[TrackRow3D]]


# -----------------------------
# Core helpers
# -----------------------------

def _as_int_frame(frame: FrameKey) -> int:
    """Convert a frame key to an int frame index.

    Accepts:
      - int
      - strings like "000123.pcd", "000123", "123"
    """
    if isinstance(frame, int):
        return frame
    s = str(frame).strip()
    if "." in s:
        s = s.split(".")[0]
    if s == "":
        raise ValueError("Empty frame key.")
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"Cannot parse frame key '{frame}' into int.") from e


def _wrap_to_2pi(angle: float) -> float:
    """Wrap angle to [0, 2*pi)."""
    two_pi = float(2.0 * np.pi)
    return float(angle) % two_pi


def trackeval_xyzwhd_from_internal_center(box7_center: np.ndarray) -> np.ndarray:
    """Convert internal JRDB-base box to TrackEval JRDB3DBox 3D det convention.

    Internal (tracker_eval pipeline):
      box7_center = (cx, cy, cz, l, w, h, rot_z)
      - x forward, y left, z up
      - center-based
      - l along +x, w along +y, h along +z
      - rot_z about +z

    TrackEval JRDB3DBox expects columns 10..16 to be:
      (x, y, z, w, h, d, yaw)   where box_format='xyzwhd'
        - (x,y,z) are center coordinates in their internal coordinate convention
        - w,h,d are size dims (width, height, depth/length)
        - yaw is rotation_y in JRDB toolkit / KITTI-style convention

    JRDB toolkit / TrackEval coordinate mapping (as in jrdb_toolkit convert scripts):
      x = -cy
      y = -cz + h/2
      z =  cx
      yaw = wrap_to_2pi(-rot_z)

    Dimension mapping:
      w_out = w
      h_out = h
      d_out = l   (depth/length)

    Returns:
      np.ndarray shape (7,) = [x, y, z, w, h, d, yaw]
    """
    box7_center = np.asarray(box7_center, dtype=np.float32).reshape(-1)
    if box7_center.shape[0] != 7:
        raise ValueError(f"Expected box7 of shape (7,), got {box7_center.shape}.")

    cx, cy, cz, l, w, h, rot_z = [float(v) for v in box7_center.tolist()]

    x = -cy
    y = -cz + 0.5 * h
    z = cx
    yaw = _wrap_to_2pi(-rot_z)

    w_out = w
    h_out = h
    d_out = l

    return np.array([x, y, z, w_out, h_out, d_out, yaw], dtype=np.float32)


def _validate_tracks_unique_ids_per_frame(tracks_by_frame: TracksByFrame) -> None:
    """TrackEval requires IDs to be unique within each timestep."""
    for fk, rows in tracks_by_frame.items():
        ids = [int(r.track_id) for r in rows]
        if len(ids) != len(set(ids)):
            seen = set()
            dup = []
            for _id in ids:
                if _id in seen:
                    dup.append(_id)
                seen.add(_id)
            raise ValueError(
                f"Duplicate track_id(s) within a single frame ({fk}): {sorted(set(dup))}. "
                "TrackEval requires IDs to be unique per frame."
            )


# -----------------------------
# KITTI/JRDB writer
# -----------------------------

def write_sequence_kitti_txt(
    out_txt_path: Union[str, Path],
    tracks_by_frame: TracksByFrame,
    *,
    class_name: str = "pedestrian",
    truncated: int = 0,
    occluded: int = 0,
    alpha: float = -1.0,
    bbox2d: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
    use_score: bool = True,
    sort_rows: bool = True,
) -> None:
    """Write a single sequence file in KITTI-tracking style format compatible with JRDB3DBox.

    Each line (17 columns if use_score=False, else 18 columns):
      0   frame
      1   track_id
      2   type
      3   truncated
      4   occluded
      5   alpha
      6-9 bbox2d (x1, y1, x2, y2)  -> can be dummy for 3D-only eval
      10-16 3D fields in TrackEval JRDB3DBox convention (box_format='xyzwhd'):
            (x, y, z, w, h, d, yaw)
      17  score (optional)

    Notes:
      - This matches TrackEval's JRDB3DBox loader expectations:
            raw_data['dets_3d'][t] = time_data[:, 10:17]
        and then uses box_format='xyzwhd' internally.
      - For 3D-only evaluation, bbox2d fields can be -1 or 0.
    """
    out_txt_path = Path(out_txt_path)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_tracks_unique_ids_per_frame(tracks_by_frame)

    x1_2d, y1_2d, x2_2d, y2_2d = bbox2d

    all_rows: List[Tuple[int, int, str]] = []
    for fk, rows in tracks_by_frame.items():
        frame_idx = _as_int_frame(fk)
        for r in rows:
            tid = int(r.track_id)

            box7 = np.asarray(r.box7, dtype=np.float32).reshape(-1)
            if box7.shape[0] != 7:
                raise ValueError(f"Frame {fk} track_id {tid}: expected box7 shape (7,), got {box7.shape}.")
            if np.isnan(box7).any():
                raise ValueError(f"Frame {fk} track_id {tid}: box7 contains NaNs.")

            det7 = trackeval_xyzwhd_from_internal_center(box7)  # (x,y,z,w,h,d,yaw)
            score = float(r.score) if (r.score is not None) else 1.0

            parts: List[str] = [
                str(frame_idx),
                str(tid),
                str(class_name),
                str(int(truncated)),
                str(int(occluded)),
                f"{float(alpha):.6f}",
                f"{float(x1_2d):.6f}",
                f"{float(y1_2d):.6f}",
                f"{float(x2_2d):.6f}",
                f"{float(y2_2d):.6f}",
            ]

            parts += [f"{float(v):.6f}" for v in det7.tolist()]

            if use_score:
                parts.append(f"{score:.6f}")

            line = " ".join(parts)
            all_rows.append((frame_idx, tid, line))

    if sort_rows:
        all_rows.sort(key=lambda x: (x[0], x[1]))

    with out_txt_path.open("w", encoding="utf-8") as f:
        for _, _, line in all_rows:
            f.write(line + "\n")
