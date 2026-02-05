# tracker_eval/export/jrdb_kitti_writer.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


# -----------------------------
# Public datatypes (lightweight)
# -----------------------------

@dataclass(frozen=True)
class TrackRow3D:
    """One tracked object at one frame in *internal* convention.

    Internal convention expected by this writer:
      - box7: (cx, cy, cz, l, w, h, yaw)  with cy = center height
      - score: optional confidence
    """
    track_id: int
    box7: np.ndarray  # shape (7,)
    score: float = 1.0


FrameKey = Union[int, str]
TracksByFrame = Mapping[FrameKey, Sequence[TrackRow3D]]


# -----------------------------
# Core conversion helpers
# -----------------------------

def _as_int_frame(frame: FrameKey) -> int:
    """Convert a frame key to an int frame index.

    Accepts:
      - int
      - strings like "000123.pcd", "000123", "123"
    """
    if isinstance(frame, int):
        return frame
    s = str(frame)
    # strip extension if present
    if "." in s:
        s = s.split(".")[0]
    # strip leading zeros safely
    s = s.strip()
    if s == "":
        raise ValueError("Empty frame key.")
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"Cannot parse frame key '{frame}' into int.") from e


def jrdb_box_from_internal_center(box7_center: np.ndarray) -> np.ndarray:
    """Convert internal (cx,cy,cz,l,w,h,yaw) to JRDB-toolkit 3D IoU convention:
       (x, y_top, z, l, h, w, theta)

    JRDB-toolkit's IoU code treats y as 'top' because it uses (y - h) as bottom.
    """
    box7_center = np.asarray(box7_center, dtype=np.float32).reshape(-1)
    if box7_center.shape[0] != 7:
        raise ValueError(f"Expected box7 of shape (7,), got {box7_center.shape}.")

    cx, cy, cz, l, w, h, yaw = box7_center.tolist()
    y_top = cy + 0.5 * h

    # Order: x, y, z, l, h, w, theta
    out = np.array([cx, y_top, cz, l, h, w, yaw], dtype=np.float32)
    return out


def _validate_tracks_unique_ids_per_frame(tracks_by_frame: TracksByFrame) -> None:
    for fk, rows in tracks_by_frame.items():
        ids = [int(r.track_id) for r in rows]
        if len(ids) != len(set(ids)):
            # find duplicates for better error message
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
    class_name: str = "Pedestrian",
    truncated: int = 0,
    occluded: int = 0,
    alpha: float = 0.0,
    bbox2d: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    use_score: bool = True,
    sort_rows: bool = True,
) -> None:
    """Write a single sequence file in KITTI-tracking-style format expected by JRDB toolkit.

    Each line (18 columns):
      0 frame
      1 track_id
      2 type
      3 truncated
      4 occluded
      5 alpha
      6-9 bbox2d (l,t,r,b)  (we put zeros by default)
      10-16 3D box7 in JRDB-toolkit convention: (x, y_top, z, l, h, w, theta)
      17 score (if use_score=True) else omitted (but JRDB loader supports both)

    Parameters
    ----------
    out_txt_path:
        Output path for <seq>.txt
    tracks_by_frame:
        dict: frame_key -> list[TrackRow3D]
    class_name:
        Usually "Pedestrian" (case-insensitive in loader).
    bbox2d:
        Placeholder if no 2D boxes (defaults to zeros).
    use_score:
        If True, writes 18 columns with score at col 17.
        If False, writes 17 columns (no score).
    sort_rows:
        If True: sorts by frame then id for reproducibility.
    """
    out_txt_path = Path(out_txt_path)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_tracks_unique_ids_per_frame(tracks_by_frame)

    # Gather rows as tuples for deterministic sorting
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

            box7_jrdb = jrdb_box_from_internal_center(box7)
            score = float(r.score) if r.score is not None else 1.0

            # Compose the full KITTI/JRDB row
            # 0..5
            parts: List[str] = [
                str(frame_idx),
                str(tid),
                str(class_name),
                str(int(truncated)),
                str(int(occluded)),
                f"{float(alpha):.6f}",
            ]
            # 6..9
            l2d, t2d, r2d, b2d = bbox2d
            parts += [f"{float(l2d):.6f}", f"{float(t2d):.6f}", f"{float(r2d):.6f}", f"{float(b2d):.6f}"]

            # 10..16  (x, y_top, z, l, h, w, theta)
            parts += [f"{float(v):.6f}" for v in box7_jrdb.tolist()]

            # 17
            if use_score:
                parts.append(f"{score:.6f}")

            line = " ".join(parts)
            all_rows.append((frame_idx, tid, line))

    if sort_rows:
        all_rows.sort(key=lambda x: (x[0], x[1]))

    with out_txt_path.open("w", encoding="utf-8") as f:
        for _, _, line in all_rows:
            f.write(line + "\n")


def write_tracker_folder_for_split(
    out_root: Union[str, Path],
    tracker_name: str,
    tracks_by_sequence: Mapping[str, TracksByFrame],
    *,
    tracker_subfolder: str = "data",
    **write_kwargs,
) -> Path:
    """Write multiple sequences in the JRDB toolkit expected structure:

      <out_root>/<tracker_name>/<tracker_subfolder>/<seq>.txt

    Returns the tracker folder path.
    """
    out_root = Path(out_root)
    tracker_dir = out_root / tracker_name / tracker_subfolder
    tracker_dir.mkdir(parents=True, exist_ok=True)

    for seq, tracks_by_frame in tracks_by_sequence.items():
        out_txt = tracker_dir / f"{seq}.txt"
        write_sequence_kitti_txt(out_txt, tracks_by_frame, **write_kwargs)

    return out_root / tracker_name
