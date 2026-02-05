# tracker_eval/data/tracks_io.py

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from tracker_eval.common.types import Box3D, Detection, FrameData, frame_sort_key


TRACKS_FORMAT_VERSION = "1.0"


def save_tracks_json(
    seq_name: str,
    tracks_by_frame: Dict[str, FrameData],
    out_path: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    box_as_dict: bool = True,
) -> None:
    """
    Save tracker outputs for one sequence into a single JSON file.

    Schema:
    {
      "_format": "tracker_eval_tracks",
      "_version": "1.0",
      "_meta": {...},
      "seq": "<seq_name>",
      "tracks": {
         "<frame_id>": [
            {
              "track_id": 12,
              "label": "pedestrian",
              "score": 0.93,           # optional
              "box": {cx,cy,cz,l,w,h,rot_z}   # or list if box_as_dict=False
            },
            ...
         ],
         ...
      }
    }
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tracks_dict: Dict[str, List[Dict[str, Any]]] = {}
    for frame_id, frame_data in tracks_by_frame.items():
        det_list: List[Dict[str, Any]] = []
        for det in frame_data.dets:
            det_list.append(det.as_json_dict(box_as_dict=box_as_dict))
        tracks_dict[frame_id] = det_list

    payload: Dict[str, Any] = {
        "_format": "tracker_eval_tracks",
        "_version": TRACKS_FORMAT_VERSION,
        "_meta": meta or {},
        "seq": seq_name,
        "tracks": tracks_dict,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_tracks_json(path: str) -> Dict[str, FrameData]:
    """
    Load tracker outputs saved by save_tracks_json().

    Returns:
      dict frame_id -> FrameData(frame_id, dets)
    """
    with open(path, "r") as f:
        root = json.load(f)

    if not isinstance(root, dict):
        raise ValueError("Tracks JSON root must be an object/dict.")
    if "tracks" not in root or not isinstance(root["tracks"], dict):
        raise ValueError("Tracks JSON must have a top-level 'tracks' dict.")

    tracks_root = root["tracks"]
    out: Dict[str, FrameData] = {}

    for frame_id, entries in tracks_root.items():
        if not isinstance(entries, list):
            continue

        dets: List[Detection] = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            if "box" not in e:
                raise ValueError(f"Track entry missing 'box' in frame {frame_id}")

            box = Box3D.from_any(e["box"])

            if "track_id" not in e:
                raise ValueError(f"Track entry missing 'track_id' in frame {frame_id}")
            track_id = int(e["track_id"])

            label = str(e.get("label", "pedestrian"))
            score = e.get("score", None)
            score_f = float(score) if score is not None else None

            dets.append(
                Detection(
                    frame_id=frame_id,
                    track_id=track_id,
                    box=box,
                    score=score_f,
                    label=label,
                    raw_label_id=None,
                )
            )

        out[frame_id] = FrameData(frame_id=frame_id, dets=dets)

    return out


def get_ordered_frames(tracks_by_frame: Dict[str, FrameData]) -> List[str]:
    frames = list(tracks_by_frame.keys())
    frames.sort(key=frame_sort_key)
    return frames
