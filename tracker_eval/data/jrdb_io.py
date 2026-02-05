# tracker_eval/data/jrdb_io.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from tracker_eval.common.types import Box3D, Detection, FrameData, parse_label_id, frame_sort_key


# --- Expected common top-level keys for different JSON types ---
LABELS_TOPLEVEL_CANDIDATES = ["labels", "annotations", "frames", "data"]
DETS_TOPLEVEL_CANDIDATES = ["detections", "dets", "predictions"]
TRACKS_TOPLEVEL_CANDIDATES = ["tracks"]


def _load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _pick_frames_dict(root: Dict[str, Any], candidates: List[str]) -> Dict[str, Any]:
    """
    Try to find a dict mapping frame_id -> list[entries] under one of candidates.
    If not found and root itself looks like that, use root.
    """
    for k in candidates:
        if k in root and isinstance(root[k], dict):
            return root[k]
    # root might already be frame->list
    if all(isinstance(v, list) for v in root.values()):
        return root  # type: ignore
    raise ValueError(f"Could not find frame dict under keys {candidates} and root is not frame->list.")


def _parse_entry_to_detection(entry: Dict[str, Any], frame_id: str, default_label: str, default_track_id: int) -> Detection:
    """
    Parse a single per-object entry from either JRDB GT or our detections/tracks.

    Accepts both:
      - box as dict {'cx':..,'cy':..,...}
      - box as list [cx,cy,cz,l,w,h,rot_z]
    """
    if "box" not in entry:
        raise ValueError("Entry missing 'box' field.")

    box = Box3D.from_any(entry["box"])

    # label_id in JRDB is 'pedestrian:14'. For detections it might be 'pedestrian:-1' or absent.
    raw_label_id = entry.get("label_id", None)
    label = default_label
    track_id = default_track_id

    if raw_label_id is not None:
        cls, tid = parse_label_id(str(raw_label_id))
        label = cls if cls != "unknown" else default_label
        track_id = tid if tid is not None else default_track_id

    # For tracker outputs, we expect 'track_id' explicitly (preferred)
    if "track_id" in entry:
        try:
            track_id = int(entry["track_id"])
        except Exception:
            raise ValueError(f"Invalid 'track_id': {entry['track_id']}")

    score = entry.get("score", None)
    score_f = float(score) if score is not None else None

    return Detection(
        frame_id=frame_id,
        track_id=track_id,
        box=box,
        score=score_f,
        label=label,
        raw_label_id=str(raw_label_id) if raw_label_id is not None else None,
    )


def load_jrdb_labels_3d(seq_json_path: str) -> Dict[str, FrameData]:
    """
    Load a single JRDB labels_3d JSON file for one sequence.

    Output:
      dict frame_id -> FrameData(frame_id, dets)
    with det.track_id = numeric GT ID (from label_id).
    """
    root = _load_json(seq_json_path)
    if not isinstance(root, dict):
        raise ValueError("labels_3d JSON root must be an object/dict.")

    frames_dict = _pick_frames_dict(root, LABELS_TOPLEVEL_CANDIDATES)

    out: Dict[str, FrameData] = {}
    for frame_id, entries in frames_dict.items():
        if not isinstance(entries, list):
            continue
        dets: List[Detection] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            # For GT, label_id is expected; if absent, track_id will be -1
            dets.append(_parse_entry_to_detection(e, frame_id=frame_id, default_label="pedestrian", default_track_id=-1))
        out[frame_id] = FrameData(frame_id=frame_id, dets=dets)

    return out


def load_jrdb_detections_3d(seq_json_path: str) -> Dict[str, FrameData]:
    """
    Load a single detections_3D JSON file for one sequence (your MinkUNet outputs).

    Output:
      dict frame_id -> FrameData(frame_id, dets)
    with det.track_id typically -1 (detections don't have persistent IDs).
    """
    root = _load_json(seq_json_path)
    if not isinstance(root, dict):
        raise ValueError("detections JSON root must be an object/dict.")

    # your files are expected to have top-level 'detections'
    frames_dict = _pick_frames_dict(root, DETS_TOPLEVEL_CANDIDATES)

    out: Dict[str, FrameData] = {}
    for frame_id, entries in frames_dict.items():
        if not isinstance(entries, list):
            continue
        dets: List[Detection] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            dets.append(_parse_entry_to_detection(e, frame_id=frame_id, default_label="pedestrian", default_track_id=-1))
        out[frame_id] = FrameData(frame_id=frame_id, dets=dets)

    return out


def list_sequence_jsons(directory: str) -> List[str]:
    """
    Return sorted list of *.json files in a directory.
    """
    files = [f for f in os.listdir(directory) if f.endswith(".json") and not f.startswith(".")]
    files.sort()
    return files


def sequence_name_from_json_filename(filename: str) -> str:
    """
    Assuming per-sequence file name is <seq_name>.json
    """
    if filename.endswith(".json"):
        return filename[:-5]
    return filename


def get_ordered_frames(frame_data: Dict[str, FrameData]) -> List[str]:
    """
    Return frames sorted numerically by frame_id if possible.
    """
    frames = list(frame_data.keys())
    frames.sort(key=frame_sort_key)
    return frames


def load_sequence_pair(
    labels_3d_dir: str,
    detections_3d_dir: str,
    seq_name: str,
) -> Tuple[Dict[str, FrameData], Dict[str, FrameData], List[str]]:
    """
    Convenience: load GT and detections for a sequence, and return common ordered frame list.

    Returns:
      gt_by_frame, det_by_frame, frames_order
    """
    gt_path = os.path.join(labels_3d_dir, f"{seq_name}.json")
    det_path = os.path.join(detections_3d_dir, f"{seq_name}.json")

    gt_by_frame = load_jrdb_labels_3d(gt_path)
    det_by_frame = load_jrdb_detections_3d(det_path)

    # Use union of frames; later evaluation can intersect if needed
    frames = sorted(set(gt_by_frame.keys()) | set(det_by_frame.keys()), key=frame_sort_key)
    return gt_by_frame, det_by_frame, frames
