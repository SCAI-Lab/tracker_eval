#!/usr/bin/env python3
"""
Precompute frame-conditioned JRDB proxy values and bin-agnostic HOTA events.

This module assigns tracker-independent proxy values directly to every GT pedestrian
occurrence. No reference-tracker eligibility filter is calculated or stored;
the analysis population is always all GT pedestrian occurrences.

Proxies:

* ``nn_distance_m``: instantaneous horizontal nearest-neighbour distance;
* ``detector_gap_length_s``: total duration of the bounded detector gap
  containing a missing GT occurrence;
* ``det_center_error_m``: instantaneous horizontal centre error of the accepted
  raw detector/GT match.

A bounded detector gap is a maximal run of missing detector observations with
a matched detector observation on both sides inside one contiguous GT
trajectory segment. Its default length is the number of continuously missing
GT occurrences divided by the configured frame rate. At 15 Hz, 15 missing
frames equal 1.0 second.

HOTA association context is unchanged: GT/tracker global alignment and
identity-pair association weights are computed over the full sequence. Stage 2
only selects individual GT occurrences according to their proxy bins.

The frame-proxy cache is tracker independent. Use ``--reuse-proxies-from``
to hard-link or copy a compatible proxy cache into a new output directory.

The HOTA event cache is tracker specific. Use ``--reuse-hota-events-from`` to
reuse compatible tracker/sequence events, or
``--recompute-hota-events-for`` to update only selected trackers in an
existing cache without reading or rebuilding the proxy tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PROXIES = [
    "nn_distance_m",
    "detector_gap_length_s",
    "det_center_error_m",
]
CACHE_VERSION = "tracker_eval_cache"

def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [x.strip() for x in value.split(",") if x.strip()]
    return values or None


def safe_tracker_name(name: str) -> str:
    return str(name).replace("/", "__").replace("\\", "__").replace(":", "_")


def stable_signature(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def add_trackeval_to_path(trackeval_root: Path) -> None:
    root = str(Path(trackeval_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def infer_jrdb_gt_folder_from_trackers(trackers_base_dir: Path) -> Path:
    trackers_base_dir = Path(trackers_base_dir).resolve()
    data_root = trackers_base_dir.parents[2]
    candidates = [
        data_root / "gt__global" / "jrdb" / trackers_base_dir.name,
        data_root / "gt" / "jrdb" / trackers_base_dir.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not infer the JRDB global GT folder. Tried:\n"
        + "\n".join(str(x) for x in candidates)
        + "\nPass --gt-folder explicitly."
    )


def trackeval_xyzwhd_to_center_xy(boxes_xyzwhd: np.ndarray) -> np.ndarray:
    """Decode TrackEval JRDB3DBox rows to JRDB-base ground-plane centres.

    The writer maps an internal centre ``(cx, cy, cz)`` to TrackEval as
    ``x_te=-cy, y_te=-cz+h/2, z_te=cx``. Therefore ``(cx,cy)=(z_te,-x_te)``.
    The same decoding works for local and global TrackEval files.
    """
    boxes = np.asarray(boxes_xyzwhd, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"Expected boxes shaped (N, 7), got {boxes.shape}")
    if len(boxes) == 0:
        return np.empty((0, 2), dtype=float)
    return np.column_stack((boxes[:, 2], -boxes[:, 0]))


def make_dataset(
    trackeval,
    trackers_base_dir: Path,
    gt_folder: Path,
    tracker: str,
    split_to_eval: str,
    tracker_sub_folder: str,
    matchable_sim_thr: float,
):
    config = dict(trackeval.datasets.JRDB3DBox.get_default_dataset_config())
    config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_base_dir),
            "TRACKERS_TO_EVAL": [tracker],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "SPLIT_TO_EVAL": split_to_eval,
            "TRACKER_SUB_FOLDER": tracker_sub_folder,
            "OUTPUT_FOLDER": None,
            "PRINT_CONFIG": False,
            "COMPUTE_GT_STATS": True,
            "GT_DENSITY_RADIUS": 2.0,
            "GT_DENSITY_RADII_M": [2.0],
            "GT_STORE_FRAME_STATS_FOR_EVENTS": True,
            "GT_MATCHABLE_SIM_THR": float(matchable_sim_thr),
            "GT_DISTANCE_3D": False,
        }
    )
    return trackeval.datasets.JRDB3DBox(config)


def load_preprocessed_sequence(dataset, tracker: str, seq: str) -> dict:
    raw_gt = dataset._load_raw_file(tracker, seq, is_gt=True)
    raw_tr = dataset._load_raw_file(tracker, seq, is_gt=False)
    raw: dict[str, Any] = {}
    raw.update(raw_gt)
    raw.update(raw_tr)
    raw["num_timesteps"] = raw_gt["num_timesteps"]
    raw["seq"] = seq
    raw["similarity_scores"] = [
        dataset._calculate_similarities(raw_gt["gt_dets_3d"][t], raw_tr["tracker_dets_3d"][t])
        for t in range(int(raw_gt["num_timesteps"]))
    ]
    data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
    data["seq"] = seq
    return data



def _frame_key_to_int(frame_key: Any) -> int:
    text = str(frame_key).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    try:
        return int(text)
    except ValueError:
        matches = re.findall(r"\d+", text)
        if not matches:
            raise ValueError(f"Cannot convert frame key {frame_key!r} to an integer index")
        return int(matches[-1])


def trackeval_xyzwhd_to_internal_boxes(boxes_xyzwhd: np.ndarray) -> np.ndarray:
    """Convert JRDB TrackEval ``xyzwhd`` rows to ``cx,cy,cz,l,w,h,rot_z``.

    The tracker writer uses ``x_te=-cy``, ``y_te=-cz+h/2``, ``z_te=cx``,
    ``w_te=w``, ``h_te=h``, ``d_te=l`` and ``yaw_te=-rot_z``.
    """
    boxes = np.asarray(boxes_xyzwhd, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"Expected boxes shaped (N, 7), got {boxes.shape}")
    if len(boxes) == 0:
        return np.empty((0, 7), dtype=float)
    out = np.empty_like(boxes, dtype=float)
    out[:, 0] = boxes[:, 2]                 # cx
    out[:, 1] = -boxes[:, 0]                # cy
    out[:, 2] = 0.5 * boxes[:, 4] - boxes[:, 1]  # cz
    out[:, 3] = boxes[:, 5]                 # l
    out[:, 4] = boxes[:, 3]                 # w
    out[:, 5] = boxes[:, 4]                 # h
    out[:, 6] = -boxes[:, 6]                # rot_z
    return out


def load_local_gt_trackeval_txt(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """Load local GT pedestrian boxes as frame -> original ID -> internal box."""
    if not path.exists():
        raise FileNotFoundError(f"Missing local GT sequence: {path}")
    frames: dict[int, dict[int, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 17:
                raise ValueError(f"{path}:{line_no}: expected at least 17 columns, got {len(parts)}")
            frame = int(float(parts[0]))
            gt_id = int(float(parts[1]))
            if str(parts[2]).strip().lower() != "pedestrian":
                continue
            te_box = np.asarray([float(v) for v in parts[10:17]], dtype=float).reshape(1, 7)
            box = trackeval_xyzwhd_to_internal_boxes(te_box)[0]
            frame_map = frames.setdefault(frame, {})
            if gt_id in frame_map:
                raise ValueError(f"{path}:{line_no}: duplicate GT ID {gt_id} in frame {frame}")
            frame_map[gt_id] = box
    return frames


def _pick_detection_frames(root: dict[str, Any]) -> dict[str, Any]:
    for key in ("detections", "dets", "predictions"):
        value = root.get(key)
        if isinstance(value, dict):
            return value
    if root and all(isinstance(value, list) for value in root.values()):
        return root
    raise ValueError("Detection JSON has no frame dictionary under detections/dets/predictions")


def _parse_detection_box(entry: dict[str, Any]) -> np.ndarray:
    box = entry.get("box")
    if isinstance(box, dict):
        required = ("cx", "cy", "cz", "l", "w", "h", "rot_z")
        missing = [key for key in required if key not in box]
        if missing:
            raise ValueError(f"Detection box is missing fields {missing}: {box!r}")
        return np.asarray([float(box[key]) for key in required], dtype=float)
    if isinstance(box, (list, tuple)) and len(box) == 7:
        return np.asarray([float(value) for value in box], dtype=float)
    raise ValueError(f"Unsupported or missing detection box: {box!r}")


def load_local_detections_json(
    path: Path,
    score_min: float | None,
) -> dict[int, np.ndarray]:
    """Load complete detector boxes in the original JRDB local/base frame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing detection sequence: {path}")
    with path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    if not isinstance(root, dict):
        raise ValueError(f"Detection JSON root must be a dict: {path}")
    frame_dict = _pick_detection_frames(root)
    out: dict[int, np.ndarray] = {}
    for frame_key, entries in frame_dict.items():
        if not isinstance(entries, list):
            continue
        frame = _frame_key_to_int(frame_key)
        boxes: list[np.ndarray] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label_id = entry.get("label_id")
            if label_id is not None:
                class_name = str(label_id).split(":", 1)[0].strip().lower()
                if class_name and class_name != "pedestrian":
                    continue
            score_raw = entry.get("score")
            score = float(score_raw) if score_raw is not None else None
            if score_min is not None and (score is None or score < float(score_min)):
                continue
            boxes.append(_parse_detection_box(entry))
        out[frame] = (
            np.asarray(boxes, dtype=float).reshape(-1, 7)
            if boxes else np.empty((0, 7), dtype=float)
        )
    return out


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _bev_corners_ccw(box: np.ndarray) -> np.ndarray:
    cx, cy, _, length, width, _, yaw = [float(x) for x in box]
    if length <= 0 or width <= 0:
        return np.empty((0, 2), dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    u = np.asarray([c, s], dtype=float)
    v = np.asarray([-s, c], dtype=float)
    hu, hv = 0.5 * length * u, 0.5 * width * v
    centre = np.asarray([cx, cy], dtype=float)
    return np.asarray([
        centre + hu + hv,
        centre - hu + hv,
        centre - hu - hv,
        centre + hu - hv,
    ], dtype=float)


def _polygon_area(vertices: np.ndarray) -> float:
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 3:
        return 0.0
    x, y = vertices[:, 0], vertices[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _segment_line_intersection(p: np.ndarray, q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = q - p
    edge = b - a
    denom = _cross2(edge, direction)
    if abs(denom) < 1e-12:
        return q.copy()
    t = _cross2(edge, a - p) / denom
    return p + t * direction


def _clip_convex_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [np.asarray(point, dtype=float) for point in np.asarray(subject, dtype=float)]
    clip = np.asarray(clip, dtype=float)
    for i in range(len(clip)):
        if not output:
            break
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        input_vertices = output
        output = []
        prev = input_vertices[-1]
        prev_inside = _cross2(b - a, prev - a) >= -1e-10
        for curr in input_vertices:
            curr_inside = _cross2(b - a, curr - a) >= -1e-10
            if curr_inside:
                if not prev_inside:
                    output.append(_segment_line_intersection(prev, curr, a, b))
                output.append(curr)
            elif prev_inside:
                output.append(_segment_line_intersection(prev, curr, a, b))
            prev, prev_inside = curr, curr_inside
    return np.asarray(output, dtype=float).reshape(-1, 2) if output else np.empty((0, 2), dtype=float)


def oriented_box_iou_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Oriented 3D IoU for internal ``cx,cy,cz,l,w,h,rot_z`` boxes."""
    a = np.asarray(box_a, dtype=float).reshape(7)
    b = np.asarray(box_b, dtype=float).reshape(7)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    if np.any(a[3:6] <= 0) or np.any(b[3:6] <= 0):
        return 0.0
    corners_a = _bev_corners_ccw(a)
    corners_b = _bev_corners_ccw(b)
    intersection_area = _polygon_area(_clip_convex_polygon(corners_a, corners_b))
    z_overlap = max(
        0.0,
        min(a[2] + 0.5 * a[5], b[2] + 0.5 * b[5])
        - max(a[2] - 0.5 * a[5], b[2] - 0.5 * b[5]),
    )
    intersection = intersection_area * z_overlap
    volume_a = float(a[3] * a[4] * a[5])
    volume_b = float(b[3] * b[4] * b[5])
    union = volume_a + volume_b - intersection
    if union <= 1e-12:
        return 0.0
    return float(np.clip(intersection / union, 0.0, 1.0))


def gated_hungarian_center_matches(
    gt_xy: np.ndarray,
    det_xy: np.ndarray,
    max_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one centre-distance matching with explicit unmatched dummies.

    Returns matched GT row indices, detection column indices and distances. A
    real pair is never accepted beyond ``max_distance_m``.
    """
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    det_xy = np.asarray(det_xy, dtype=float).reshape(-1, 2)
    n_gt, n_det = len(gt_xy), len(det_xy)
    if n_gt == 0 or n_det == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=float),
        )
    gate = float(max_distance_m)
    if not np.isfinite(gate) or gate <= 0:
        raise ValueError("--det-match-max-dist-m must be a positive finite value")
    delta = gt_xy[:, None, :] - det_xy[None, :, :]
    distances = np.linalg.norm(delta, axis=2)

    size = n_gt + n_det
    # Lexicographic objective: maximize the number of valid pairs first, then
    # minimize their total centre distance. Since every normalized valid edge
    # costs at most 1 and there can be at most ``size`` such edges, an unmatched
    # GT cost of ``size + 1`` makes losing one valid match more expensive than
    # any possible distance improvement among all remaining matches.
    unmatched_gt_cost = float(size + 1)
    large = float(1000 * (size + 1))
    cost = np.full((size, size), large, dtype=float)
    valid = distances <= gate + 1e-12
    cost[:n_gt, :n_det][valid] = distances[valid] / gate

    # Each GT and each detection has its own dummy option.
    cost[np.arange(n_gt), n_det + np.arange(n_gt)] = unmatched_gt_cost
    cost[n_gt + np.arange(n_det), np.arange(n_det)] = 0.0
    cost[n_gt:, n_det:] = 0.0

    rows, cols = linear_sum_assignment(cost)
    keep = (rows < n_gt) & (cols < n_det)
    rows = rows[keep]
    cols = cols[keep]
    if len(rows):
        keep_valid = distances[rows, cols] <= gate + 1e-12
        rows = rows[keep_valid]
        cols = cols[keep_valid]
    return (
        rows.astype(np.int32, copy=False),
        cols.astype(np.int32, copy=False),
        distances[rows, cols].astype(float, copy=False),
    )


def compute_detector_observation_features(
    data: dict,
    local_gt_by_frame: dict[int, dict[int, np.ndarray]],
    detections_by_frame: dict[int, np.ndarray],
    det_match_max_dist_m: float,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    """Match all local GTs to local detections and cache raw-box quality."""
    n_t = int(data["num_timesteps"])
    orig_map = np.asarray(
        data.get("gt_orig_ids", np.arange(int(data["num_gt_ids"]))), dtype=np.int64
    )
    output = {
        "det_present": [],
        "det_match_distance_m": [],
        "det_center_error_m": [],
        "det_box_iou_3d": [],
    }
    total_local_gt = 0
    total_local_det = 0
    total_local_matches = 0
    iou_sum = 0.0
    missing_global_occurrences: list[tuple[int, int]] = []

    for t in range(n_t):
        local_gt_map = local_gt_by_frame.get(t, {})
        local_ids = np.asarray(sorted(local_gt_map), dtype=np.int64)
        local_boxes = (
            np.asarray([local_gt_map[int(gid)] for gid in local_ids], dtype=float).reshape(-1, 7)
            if len(local_ids) else np.empty((0, 7), dtype=float)
        )
        det_boxes = np.asarray(
            detections_by_frame.get(t, np.empty((0, 7))), dtype=float
        ).reshape(-1, 7)
        rows, cols, distances = gated_hungarian_center_matches(
            local_boxes[:, :2], det_boxes[:, :2], max_distance_m=det_match_max_dist_m
        )
        matched_by_orig: dict[int, tuple[float, float]] = {}
        for row, col, distance in zip(rows.tolist(), cols.tolist(), distances.tolist()):
            iou = oriented_box_iou_3d(local_boxes[row], det_boxes[col])
            matched_by_orig[int(local_ids[row])] = (float(distance), float(iou))
            if np.isfinite(iou):
                iou_sum += float(iou)

        total_local_gt += len(local_ids)
        total_local_det += len(det_boxes)
        total_local_matches += len(rows)

        internal_ids = np.asarray(data["gt_ids"][t], dtype=int)
        present = np.zeros(len(internal_ids), dtype=bool)
        match_distance = np.full(len(internal_ids), np.nan, dtype=float)
        centre_error = np.full(len(internal_ids), np.nan, dtype=float)
        box_iou = np.full(len(internal_ids), np.nan, dtype=float)
        for local_index, internal_id in enumerate(internal_ids):
            original_id = int(orig_map[int(internal_id)])
            if original_id not in local_gt_map:
                missing_global_occurrences.append((t, original_id))
                continue
            match = matched_by_orig.get(original_id)
            if match is not None:
                distance, iou = match
                present[local_index] = True
                match_distance[local_index] = distance
                centre_error[local_index] = distance
                box_iou[local_index] = iou
        output["det_present"].append(present)
        output["det_match_distance_m"].append(match_distance)
        output["det_center_error_m"].append(centre_error)
        output["det_box_iou_3d"].append(box_iou)

    if missing_global_occurrences:
        examples = ", ".join(f"(frame={t}, id={gid})" for t, gid in missing_global_occurrences[:10])
        raise RuntimeError(
            f"Could not align {len(missing_global_occurrences)} global TrackEval GT occurrences "
            f"to local GT by (frame, original ID). Examples: {examples}"
        )
    return output, {
        "num_local_gt_occurrences": int(total_local_gt),
        "num_local_detection_occurrences": int(total_local_det),
        "num_local_gt_detection_matches": int(total_local_matches),
        "local_gt_detection_match_fraction": (
            float(total_local_matches / total_local_gt) if total_local_gt else np.nan
        ),
        "mean_matched_detector_iou_3d": (
            float(iou_sum / total_local_matches) if total_local_matches else np.nan
        ),
    }


def compute_spatial_context_features(data: dict) -> dict[str, list[np.ndarray]]:
    """Calculate density and nearest-neighbour context from all GT pedestrians."""
    n_t = int(data["num_timesteps"])
    output = {"density_1m": [], "density_2m": [], "nn": []}
    area_1m = float(np.pi)
    area_2m = float(4.0 * np.pi)

    for t in range(n_t):
        boxes = np.asarray(data["gt_dets_3d"][t], dtype=float)
        n = len(boxes)
        if n == 0:
            for key in output:
                output[key].append(np.empty(0, dtype=float))
            continue
        centres = trackeval_xyzwhd_to_center_xy(boxes)
        if n == 1:
            density_1m = np.zeros(1, dtype=float)
            density_2m = np.zeros(1, dtype=float)
            nn = np.full(1, np.nan, dtype=float)
        else:
            delta = centres[:, None, :] - centres[None, :, :]
            distances = np.linalg.norm(delta, axis=2)
            np.fill_diagonal(distances, np.inf)
            density_1m = np.sum(distances <= 1.0, axis=1).astype(float) / area_1m
            density_2m = np.sum(distances <= 2.0, axis=1).astype(float) / area_2m
            nn = np.min(distances, axis=1)
            nn[~np.isfinite(nn)] = np.nan
        output["density_1m"].append(density_1m)
        output["density_2m"].append(density_2m)
        output["nn"].append(nn)
    return output


def _longest_true_run(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool).reshape(-1)
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _count_true_runs(values: np.ndarray) -> int:
    values = np.asarray(values, dtype=bool).reshape(-1)
    if len(values) == 0:
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))



def build_frame_proxy_cache(
    data: dict,
    detector_features: dict[str, list[np.ndarray]],
    spatial_features: dict[str, list[np.ndarray]],
    fps: float,
    gap_length_definition: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Create one proxy value per GT pedestrian occurrence.

    ``detector_gap_length_s`` is finite only on missing occurrences belonging
    to bounded internal gaps. Leading and trailing detector absences are
    intentionally excluded because they represent initialization/termination
    rather than through-gap coasting and reacquisition.
    """
    n_t = int(data["num_timesteps"])
    n_gt = int(data["num_gt_ids"])
    orig_map = np.asarray(data.get("gt_orig_ids", np.arange(n_gt)), dtype=np.int64)

    frame_offsets = np.zeros(n_t + 1, dtype=np.int64)
    for t in range(n_t):
        frame_offsets[t + 1] = frame_offsets[t] + len(data["gt_ids"][t])
    total = int(frame_offsets[-1])

    gt_internal_flat = np.empty(total, dtype=np.int32)
    gt_orig_flat = np.empty(total, dtype=np.int64)
    frame_index_flat = np.empty(total, dtype=np.int32)
    detector_present_flat = np.zeros(total, dtype=bool)
    detector_match_distance_flat = np.full(total, np.nan, dtype=np.float32)
    detector_center_error_flat = np.full(total, np.nan, dtype=np.float32)
    detector_box_iou_flat = np.full(total, np.nan, dtype=np.float32)
    nn_distance_flat = np.full(total, np.nan, dtype=np.float64)

    gap_id_flat = np.full(total, -1, dtype=np.int64)
    gap_missing_frames_flat = np.zeros(total, dtype=np.int32)
    gap_interval_frames_flat = np.zeros(total, dtype=np.int32)
    gap_length_s_flat = np.full(total, np.nan, dtype=np.float64)

    local_flat_index: list[dict[int, int]] = []
    frames_by_gid: list[list[int]] = [[] for _ in range(n_gt)]

    for t in range(n_t):
        start = int(frame_offsets[t])
        ids = np.asarray(data["gt_ids"][t], dtype=int)
        mapping: dict[int, int] = {}
        frame_nn = np.asarray(spatial_features["nn"][t], dtype=float)
        if len(frame_nn) != len(ids):
            raise RuntimeError(
                f"{data['seq']} frame {t}: NN support {len(frame_nn)} "
                f"does not match GT support {len(ids)}"
            )
        for local, gid in enumerate(ids):
            flat = start + local
            gid_int = int(gid)
            mapping[gid_int] = flat
            frames_by_gid[gid_int].append(t)
            gt_internal_flat[flat] = gid_int
            gt_orig_flat[flat] = int(orig_map[gid_int])
            frame_index_flat[flat] = int(t)
            detector_present_flat[flat] = bool(
                detector_features["det_present"][t][local]
            )
            detector_match_distance_flat[flat] = float(
                detector_features["det_match_distance_m"][t][local]
            )
            detector_center_error_flat[flat] = float(
                detector_features["det_center_error_m"][t][local]
            )
            detector_box_iou_flat[flat] = float(
                detector_features["det_box_iou_3d"][t][local]
            )
            nn_distance_flat[flat] = float(frame_nn[local])
        local_flat_index.append(mapping)

    gap_id = 0
    bounded_gap_count = 0
    bounded_gap_missing_occurrences = 0
    censored_missing_occurrences = 0

    for gid in range(n_gt):
        all_frames = frames_by_gid[gid]
        if not all_frames:
            continue
        seg_start = 0
        while seg_start < len(all_frames):
            seg_end = seg_start + 1
            while (
                seg_end < len(all_frames)
                and all_frames[seg_end] == all_frames[seg_end - 1] + 1
            ):
                seg_end += 1
            segment = all_frames[seg_start:seg_end]
            flats = np.asarray(
                [local_flat_index[t][gid] for t in segment], dtype=np.int64
            )
            present = detector_present_flat[flats]

            pos = 0
            while pos < len(segment):
                if bool(present[pos]):
                    pos += 1
                    continue
                run_end = pos + 1
                while run_end < len(segment) and not bool(present[run_end]):
                    run_end += 1

                bounded = (
                    pos > 0
                    and run_end < len(segment)
                    and bool(present[pos - 1])
                    and bool(present[run_end])
                )
                missing_frames = int(run_end - pos)
                run_flats = flats[pos:run_end]

                if bounded:
                    previous_hit_frame = int(segment[pos - 1])
                    next_hit_frame = int(segment[run_end])
                    interval_frames = int(next_hit_frame - previous_hit_frame)
                    if gap_length_definition == "observation_interval":
                        length_frames = interval_frames
                    elif gap_length_definition == "missing_frames":
                        length_frames = missing_frames
                    else:  # guarded by argparse; retained for programmatic use
                        raise ValueError(
                            f"Unsupported gap length definition: "
                            f"{gap_length_definition!r}"
                        )

                    gap_id_flat[run_flats] = int(gap_id)
                    gap_missing_frames_flat[run_flats] = missing_frames
                    gap_interval_frames_flat[run_flats] = interval_frames
                    gap_length_s_flat[run_flats] = float(length_frames / float(fps))
                    gap_id += 1
                    bounded_gap_count += 1
                    bounded_gap_missing_occurrences += missing_frames
                else:
                    censored_missing_occurrences += missing_frames

                pos = run_end
            seg_start = seg_end

    rows: list[dict[str, Any]] = []
    for flat in range(total):
        rows.append(
            {
                "seq": str(data["seq"]),
                "frame": int(frame_index_flat[flat]),
                "gt_internal_id": int(gt_internal_flat[flat]),
                "gt_id": int(gt_orig_flat[flat]),
                "detector_present": bool(detector_present_flat[flat]),
                "detector_match_distance_m": float(
                    detector_match_distance_flat[flat]
                ),
                "detector_box_iou_3d": float(detector_box_iou_flat[flat]),
                "detector_gap_id": int(gap_id_flat[flat]),
                "detector_gap_missing_frames": int(
                    gap_missing_frames_flat[flat]
                ),
                "detector_gap_interval_frames": int(
                    gap_interval_frames_flat[flat]
                ),
                "nn_distance_m": float(nn_distance_flat[flat]),
                "detector_gap_length_s": float(gap_length_s_flat[flat]),
                "det_center_error_m": float(
                    detector_center_error_flat[flat]
                ),
            }
        )

    payload: dict[str, np.ndarray] = {
        "num_timesteps": np.asarray([n_t], dtype=np.int64),
        "frame_gt_offsets": frame_offsets,
        "gt_internal_ids_flat": gt_internal_flat,
        "gt_orig_ids_flat": gt_orig_flat,
        "frame_indices_flat": frame_index_flat,
        "detector_present_flat": detector_present_flat,
        "detector_match_distance_m_flat": detector_match_distance_flat,
        "detector_center_error_m_flat": detector_center_error_flat,
        "detector_box_iou_3d_flat": detector_box_iou_flat,
        "detector_gap_id_flat": gap_id_flat,
        "detector_gap_missing_frames_flat": gap_missing_frames_flat,
        "detector_gap_interval_frames_flat": gap_interval_frames_flat,
        "proxy__nn_distance_m": nn_distance_flat,
        "proxy__detector_gap_length_s": gap_length_s_flat,
        "proxy__det_center_error_m": detector_center_error_flat.astype(
            np.float64, copy=False
        ),
    }
    support = {
        "num_gt_occurrences": int(total),
        "num_detector_matched_occurrences": int(
            np.sum(detector_present_flat)
        ),
        "num_detector_missing_occurrences": int(
            np.sum(~detector_present_flat)
        ),
        "num_bounded_gaps": int(bounded_gap_count),
        "num_bounded_gap_missing_occurrences": int(
            bounded_gap_missing_occurrences
        ),
        "num_censored_missing_occurrences": int(
            censored_missing_occurrences
        ),
    }
    return payload, rows, support


def save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def precompute_proxy_sequence(
    *,
    dataset,
    seq: str,
    pedreftrack_tracker: str,
    local_gt_folder: Path,
    detections_dir: Path,
    proxy_dir: Path,
    fps: float,
    gap_length_definition: str,
    det_match_max_dist_m: float,
    det_score_min: float | None,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_path = Path(proxy_dir) / f"{seq}.npz"
    frame_csv = Path(proxy_dir) / f"{seq}.frames.csv.gz"
    if out_path.exists() and not force:
        if not frame_csv.exists():
            raise FileNotFoundError(
                f"Existing proxy cache lacks {frame_csv}; "
                "rerun with --force-recompute"
            )
        rows = pd.read_csv(frame_csv).to_dict("records")
        return rows, {
            "seq": seq,
            "status": "cached",
            "num_frame_rows": int(len(rows)),
            "proxy_path": str(out_path),
        }

    data = load_preprocessed_sequence(dataset, pedreftrack_tracker, seq)
    local_gt_path = Path(local_gt_folder) / f"{seq}.txt"
    detection_path = Path(detections_dir) / f"{seq}.json"
    local_gt = load_local_gt_trackeval_txt(local_gt_path)
    detections = load_local_detections_json(
        detection_path, score_min=det_score_min
    )
    detector_features, detector_support = (
        compute_detector_observation_features(
            data,
            local_gt_by_frame=local_gt,
            detections_by_frame=detections,
            det_match_max_dist_m=det_match_max_dist_m,
        )
    )
    spatial_features = compute_spatial_context_features(data)
    payload, rows, frame_support = build_frame_proxy_cache(
        data,
        detector_features,
        spatial_features,
        fps=fps,
        gap_length_definition=gap_length_definition,
    )
    save_npz_atomic(out_path, payload)
    pd.DataFrame(rows).to_csv(frame_csv, index=False)

    support = {
        "seq": seq,
        "status": "computed",
        "num_gt_ids": int(data["num_gt_ids"]),
        "num_frame_rows": int(len(rows)),
        "analysis_population": "all_gt_occurrences",
        "proxy_path": str(out_path),
        **frame_support,
        **detector_support,
    }
    return rows, support



PROXY_COMPATIBILITY_KEYS = [
    "cache_version",
    "gt_folder",
    "local_gt_folder",
    "detections_dir",
    "sequences",
    "fps",
    "gap_length_definition",
    "gap_duration_formula",
    "gap_population",
    "det_match_max_dist_m",
    "det_score_min",
    "detector_matching",
    "detector_coordinate_frame",
    "spatial_context_source",
    "analysis_population",
    "eligibility_filter",
]


def resolve_cache_root(path: Path) -> Path:
    """Resolve either a cache directory or one of its direct subdirectories."""
    path = Path(path).resolve()
    if path.name in {"frame_proxies", "hota_events"}:
        return path.parent
    return path


def validate_proxy_reuse_source(
    source: Path,
    requested_metadata: dict[str, Any],
) -> Path:
    """Validate that a source cache has compatible tracker-independent proxies."""
    cache_root = resolve_cache_root(source)
    metadata_path = cache_root / "metadata.json"
    proxy_root = cache_root / "frame_proxies"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Proxy reuse source lacks metadata.json: {metadata_path}"
        )
    if not proxy_root.is_dir():
        raise FileNotFoundError(
            f"Proxy reuse source lacks frame_proxies/: {proxy_root}"
        )

    with metadata_path.open("r", encoding="utf-8") as handle:
        source_metadata = json.load(handle)

    changed = [
        key
        for key in PROXY_COMPATIBILITY_KEYS
        if source_metadata.get(key) != requested_metadata.get(key)
    ]
    if changed:
        details = "\n".join(
            f"  {key}: source={source_metadata.get(key)!r}, "
            f"requested={requested_metadata.get(key)!r}"
            for key in changed
        )
        raise RuntimeError(
            "The proxy reuse source is incompatible with the requested HOTA cache "
            f"proxy definition:\n{details}"
        )
    return proxy_root


def reuse_proxy_sequence_files(
    source_proxy_root: Path,
    destination_proxy_root: Path,
    seq: str,
) -> dict[str, Any]:
    """Hard-link one sequence's proxy NPZ/CSV files, falling back to copying."""
    source_proxy_root = Path(source_proxy_root)
    destination_proxy_root = Path(destination_proxy_root)
    source_npz = source_proxy_root / f"{seq}.npz"
    source_csv = source_proxy_root / f"{seq}.frames.csv.gz"
    destination_npz = destination_proxy_root / source_npz.name
    destination_csv = destination_proxy_root / source_csv.name

    for source in (source_npz, source_csv):
        if not source.is_file():
            raise FileNotFoundError(
                f"Proxy reuse source is incomplete; missing {source}"
            )

    destination_proxy_root.mkdir(parents=True, exist_ok=True)
    statuses: list[str] = []
    for source, destination in (
        (source_npz, destination_npz),
        (source_csv, destination_csv),
    ):
        if destination.exists():
            statuses.append("cached")
            continue
        try:
            os.link(source, destination)
            statuses.append("hardlink")
        except OSError:
            shutil.copy2(source, destination)
            statuses.append("copy")

    if all(status == "cached" for status in statuses):
        status = "cached"
    elif "copy" in statuses:
        status = "reused_copy"
    else:
        status = "reused_hardlink"
    return {
        "seq": seq,
        "status": status,
        "source_proxy_root": str(source_proxy_root),
        "destination_proxy_root": str(destination_proxy_root),
    }


def require_existing_proxy_cache(
    output_dir: Path,
    proxy_dir: Path,
    sequences: list[str],
) -> None:
    """Require a complete existing proxy cache for event-only tracker updates."""
    required = [
        Path(output_dir) / "frame_proxy_values.csv.gz",
        Path(output_dir) / "proxy_support_by_sequence.csv",
    ]
    for seq in sequences:
        required.extend(
            [
                Path(proxy_dir) / f"{seq}.npz",
                Path(proxy_dir) / f"{seq}.frames.csv.gz",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:10])
        suffix = (
            f"\n  ... and {len(missing) - 10} more"
            if len(missing) > 10
            else ""
        )
        raise FileNotFoundError(
            "--recompute-hota-events-for performs an event-only update and "
            "requires an existing complete HOTA cache proxy cache. Missing:\n"
            f"{preview}{suffix}\nBuild the proxy cache first or use "
            "--reuse-proxies-from in a normal run."
        )


def resolve_event_source_root(path: Path) -> Path:
    path = Path(path).resolve()
    direct = path / "hota_events"
    return direct if direct.is_dir() else path


def reuse_event_file(
    source_root: Path,
    tracker: str,
    seq: str,
    destination: Path,
    force: bool,
) -> dict[str, Any] | None:
    """Hard-link one compatible prior event file, falling back to copying."""
    destination = Path(destination)
    if destination.exists() and not force:
        return {
            "tracker": tracker,
            "seq": seq,
            "status": "cached",
            "path": str(destination),
        }

    source = (
        resolve_event_source_root(source_root)
        / safe_tracker_name(tracker)
        / f"{seq}.npz"
    )
    if not source.exists():
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        status = "reused_hardlink"
    except OSError:
        shutil.copy2(source, destination)
        status = "reused_copy"
    return {
        "tracker": tracker,
        "seq": seq,
        "status": status,
        "source": str(source),
        "path": str(destination),
    }


def precompute_hota_event_sequence(
    *,
    trackeval_root: Path,
    trackers_base_dir: Path,
    gt_folder: Path,
    tracker: str,
    seq: str,
    split_to_eval: str,
    tracker_sub_folder: str,
    matchable_sim_thr: float,
    output_path: Path,
    force: bool,
    preprocessed_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one tracker/sequence's bin-independent HOTA event representation.

    ``preprocessed_data`` is an optional in-process reuse hook.  Capability
    profiling already needs the same TrackEval-preprocessed sequence for its
    fixed-threshold assignments, so accepting it avoids loading and computing
    corrected 3D similarities twice.  Omitting it preserves the standalone
    helper's original behavior.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if output_path.exists() and not force:
        return {"tracker": tracker, "seq": seq, "status": "cached", "path": str(output_path)}

    add_trackeval_to_path(trackeval_root)
    import trackeval  # noqa: E402

    if preprocessed_data is None:
        dataset = make_dataset(
            trackeval,
            trackers_base_dir,
            gt_folder,
            tracker,
            split_to_eval,
            tracker_sub_folder,
            matchable_sim_thr,
        )
        data = load_preprocessed_sequence(dataset, tracker, seq)
    else:
        data = preprocessed_data
    alpha_labels = np.asarray(trackeval.metrics.HOTA().array_labels, dtype=np.float64)
    n_alpha = len(alpha_labels)
    n_t = int(data["num_timesteps"])
    n_gt_ids = int(data["num_gt_ids"])
    n_tr_ids = int(data["num_tracker_ids"])

    gt_offsets = np.zeros(n_t + 1, dtype=np.int64)
    tr_offsets = np.zeros(n_t + 1, dtype=np.int64)
    for t in range(n_t):
        gt_offsets[t + 1] = gt_offsets[t] + len(data["gt_ids"][t])
        tr_offsets[t + 1] = tr_offsets[t] + len(data["tracker_ids"][t])
    total_gt = int(gt_offsets[-1])
    total_tr = int(tr_offsets[-1])

    gt_internal_flat = np.empty(total_gt, dtype=np.int32)
    gt_orig_flat = np.empty(total_gt, dtype=np.int64)
    tr_internal_flat = np.empty(total_tr, dtype=np.int32)
    tr_orig_flat = np.empty(total_tr, dtype=np.int64)
    tr_frame_flat = np.empty(total_tr, dtype=np.int32)
    nearest_gt_flat = np.full(total_tr, -1, dtype=np.int64)
    nearest_gt_distance = np.full(total_tr, np.inf, dtype=np.float32)
    assigned_gt_flat = np.full(total_tr, -1, dtype=np.int64)
    assigned_similarity = np.full(total_tr, -np.inf, dtype=np.float32)
    assigned_gt_internal = np.full(total_tr, -1, dtype=np.int32)

    gt_orig_map = np.asarray(data.get("gt_orig_ids", np.arange(n_gt_ids)), dtype=np.int64)
    tr_orig_map = np.asarray(data.get("tracker_orig_ids", np.arange(n_tr_ids)), dtype=np.int64)
    gt_count = np.zeros(n_gt_ids, dtype=np.int64)
    tr_count = np.zeros(n_tr_ids, dtype=np.int64)
    potential = np.zeros((n_gt_ids, n_tr_ids), dtype=np.float64)

    for t in range(n_t):
        ga, gb = int(gt_offsets[t]), int(gt_offsets[t + 1])
        ta, tb = int(tr_offsets[t]), int(tr_offsets[t + 1])
        gt_ids = np.asarray(data["gt_ids"][t], dtype=int)
        tr_ids = np.asarray(data["tracker_ids"][t], dtype=int)
        if len(gt_ids):
            gt_internal_flat[ga:gb] = gt_ids
            gt_orig_flat[ga:gb] = gt_orig_map[gt_ids]
            gt_count[gt_ids] += 1
        if len(tr_ids):
            tr_internal_flat[ta:tb] = tr_ids
            tr_orig_flat[ta:tb] = tr_orig_map[tr_ids]
            tr_frame_flat[ta:tb] = t
            tr_count[tr_ids] += 1

        sim = np.asarray(data["similarity_scores"][t], dtype=float)
        if len(gt_ids) and len(tr_ids):
            denominator = sim.sum(axis=0)[None, :] + sim.sum(axis=1)[:, None] - sim
            sim_iou = np.zeros_like(sim)
            valid = denominator > np.finfo(float).eps
            sim_iou[valid] = sim[valid] / denominator[valid]
            potential[gt_ids[:, None], tr_ids[None, :]] += sim_iou

            gt_xy = trackeval_xyzwhd_to_center_xy(data["gt_dets_3d"][t])
            tr_xy = trackeval_xyzwhd_to_center_xy(data["tracker_dets_3d"][t])
            delta = tr_xy[:, None, :] - gt_xy[None, :, :]
            distances = np.sqrt(np.sum(delta * delta, axis=2))
            nearest_local = np.argmin(distances, axis=1)
            nearest_gt_flat[ta:tb] = ga + nearest_local
            nearest_gt_distance[ta:tb] = distances[np.arange(len(tr_ids)), nearest_local]

    denominator = gt_count[:, None] + tr_count[None, :] - potential
    global_alignment = np.divide(
        potential,
        np.maximum(denominator, 1e-10),
        out=np.zeros_like(potential),
        where=np.maximum(denominator, 1e-10) > 0,
    )
    match_counts = np.zeros((n_alpha, n_gt_ids, n_tr_ids), dtype=np.int32)

    for t in range(n_t):
        ga, gb = int(gt_offsets[t]), int(gt_offsets[t + 1])
        ta, tb = int(tr_offsets[t]), int(tr_offsets[t + 1])
        gt_ids = np.asarray(data["gt_ids"][t], dtype=int)
        tr_ids = np.asarray(data["tracker_ids"][t], dtype=int)
        if not len(gt_ids) or not len(tr_ids):
            continue
        sim = np.asarray(data["similarity_scores"][t], dtype=float)
        score = global_alignment[gt_ids[:, None], tr_ids[None, :]] * sim
        rows, cols = linear_sum_assignment(-score)
        for row, col in zip(rows, cols):
            tr_flat = ta + int(col)
            gt_flat = ga + int(row)
            similarity = float(sim[row, col])
            assigned_gt_flat[tr_flat] = gt_flat
            assigned_gt_internal[tr_flat] = int(gt_ids[row])
            assigned_similarity[tr_flat] = similarity
            accepted = similarity >= alpha_labels - np.finfo(float).eps
            if np.any(accepted):
                match_counts[accepted, int(gt_ids[row]), int(tr_ids[col])] += 1

    pair_present = np.any(match_counts > 0, axis=0)
    pair_gt, pair_tr = np.nonzero(pair_present)
    pair_counts = match_counts[:, pair_gt, pair_tr].T.astype(np.int32, copy=False)
    pair_lookup = {
        int(g) * max(1, n_tr_ids) + int(r): idx
        for idx, (g, r) in enumerate(zip(pair_gt.tolist(), pair_tr.tolist()))
    }
    assigned_pair_index = np.full(total_tr, -1, dtype=np.int32)
    for tr_flat in np.flatnonzero(assigned_gt_internal >= 0):
        tr_id = int(tr_internal_flat[tr_flat])
        gt_id = int(assigned_gt_internal[tr_flat])
        assigned_pair_index[tr_flat] = pair_lookup.get(gt_id * max(1, n_tr_ids) + tr_id, -1)

    payload = {
        "alpha_labels": alpha_labels,
        "num_timesteps": np.asarray([n_t], dtype=np.int64),
        "frame_gt_offsets": gt_offsets,
        "frame_tracker_offsets": tr_offsets,
        "gt_internal_ids_flat": gt_internal_flat,
        "gt_orig_ids_flat": gt_orig_flat,
        "tracker_internal_ids_flat": tr_internal_flat,
        "tracker_orig_ids_flat": tr_orig_flat,
        "tracker_frame_indices_flat": tr_frame_flat,
        "gt_id_counts": gt_count,
        "tracker_id_counts": tr_count,
        "nearest_gt_flat_index": nearest_gt_flat,
        "nearest_gt_distance_m": nearest_gt_distance,
        "assigned_gt_flat_index": assigned_gt_flat,
        "assigned_similarity": assigned_similarity,
        "assigned_pair_index": assigned_pair_index,
        "pair_gt_internal_ids": pair_gt.astype(np.int32),
        "pair_tracker_internal_ids": pair_tr.astype(np.int32),
        "pair_match_counts": pair_counts,
    }
    save_npz_atomic(output_path, payload)
    return {
        "tracker": tracker,
        "seq": seq,
        "status": "computed",
        "num_gt_occurrences": total_gt,
        "num_tracker_occurrences": total_tr,
        "num_association_pairs": int(len(pair_gt)),
        "path": str(output_path),
    }



def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute HOTA cache frame-conditioned proxies and full-sequence "
            "HOTA event context."
        )
    )
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--trackers-base-dir", type=Path, required=True)
    parser.add_argument(
        "--gt-folder",
        type=Path,
        default=None,
        help="Global JRDB TrackEval GT folder; inferred when omitted.",
    )
    parser.add_argument(
        "--local-gt-folder",
        type=Path,
        required=True,
        help="Original local TrackEval GT label_02 directory.",
    )
    parser.add_argument(
        "--detections-dir",
        type=Path,
        required=True,
        help="Original local raw detector JSON directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-proxies-from",
        type=Path,
        default=None,
        help=(
            "Optional compatible cache directory, or its frame_proxies "
            "directory. Per-sequence tracker-independent proxy files are "
            "hard-linked where possible and copied otherwise. Compatibility "
            "of GT, detections, FPS and proxy settings is verified."
        ),
    )
    parser.add_argument(
        "--reuse-hota-events-from",
        type=Path,
        default=None,
        help=(
            "Optional prior cache directory, or its hota_events directory. "
            "Compatible event files are hard-linked where possible and copied "
            "otherwise; missing files are computed normally."
        ),
    )
    parser.add_argument(
        "--exclude-hota-event-reuse",
        default=None,
        help=(
            "Optional comma-separated tracker names whose event files must be "
            "computed from current tracker outputs rather than reused from "
            "--reuse-hota-events-from."
        ),
    )
    parser.add_argument(
        "--recompute-hota-events-for",
        default=None,
        help=(
            "Efficient in-place update mode. Recompute only the listed "
            "comma-separated trackers in an existing cache, skip the proxy "
            "stage entirely, and leave all other HOTA event files untouched. "
            "Do not combine this with --force-recompute."
        ),
    )
    parser.add_argument(
        "--pedreftrack-tracker",
        default="pedreftrack__global_gt_assisted",
    )
    parser.add_argument(
        "--trackers",
        required=True,
        help="Comma-separated __global tracker folders.",
    )
    parser.add_argument("--split-to-eval", default="test")
    parser.add_argument("--tracker-sub-folder", default="data")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument(
        "--gap-length-definition",
        choices=["missing_frames", "observation_interval"],
        default="missing_frames",
        help=(
            "missing_frames (default): number of continuously missing GT "
            "occurrences divided by --fps, so 15 missing frames equal 1.0 s "
            "at 15 Hz; observation_interval: next matched frame minus previous "
            "matched frame."
        ),
    )
    parser.add_argument(
        "--det-match-max-dist-m",
        type=float,
        default=0.30,
        help="Hard local XY gate for raw detector-to-GT matching.",
    )
    parser.add_argument(
        "--det-score-min",
        type=float,
        default=None,
        help="Optional common detector score filter.",
    )
    parser.add_argument("--matchable-sim-thr", type=float, default=0.30)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be a positive finite value")

    trackers = parse_csv_list(args.trackers) or []
    if not trackers:
        raise ValueError("--trackers must contain at least one tracker")
    if args.pedreftrack_tracker not in trackers:
        trackers.append(args.pedreftrack_tracker)
    trackers = list(dict.fromkeys(trackers))

    exclude_hota_event_reuse = set(
        parse_csv_list(args.exclude_hota_event_reuse) or []
    )
    recompute_hota_events_for = set(
        parse_csv_list(args.recompute_hota_events_for) or []
    )
    if recompute_hota_events_for and args.force_recompute:
        raise ValueError(
            "--recompute-hota-events-for is an event-only selective update; "
            "do not combine it with --force-recompute."
        )

    for name in [args.pedreftrack_tracker, *trackers]:
        if not (
            str(name).endswith("__global")
            or "__global_" in str(name)
        ):
            raise ValueError(
                "HOTA cache HOTA event data requires __global tracker outputs, "
                f"got {name!r}"
            )

    trackeval_root = args.trackeval_root.resolve()
    trackers_base_dir = args.trackers_base_dir.resolve()
    gt_folder = (
        args.gt_folder.resolve()
        if args.gt_folder is not None
        else infer_jrdb_gt_folder_from_trackers(trackers_base_dir)
    )
    local_gt_folder = args.local_gt_folder.resolve()
    detections_dir = args.detections_dir.resolve()
    output_dir = args.output_dir.resolve()
    proxy_dir = output_dir / "frame_proxies"
    event_dir = output_dir / "hota_events"
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy_dir.mkdir(parents=True, exist_ok=True)
    event_dir.mkdir(parents=True, exist_ok=True)

    add_trackeval_to_path(trackeval_root)
    import trackeval  # noqa: E402

    pedreftrack_dataset = make_dataset(
        trackeval,
        trackers_base_dir,
        gt_folder,
        args.pedreftrack_tracker,
        args.split_to_eval,
        args.tracker_sub_folder,
        args.matchable_sim_thr,
    )
    sequences = list(pedreftrack_dataset.seq_list)

    metadata = {
        "cache_version": CACHE_VERSION,
        "trackeval_root": str(trackeval_root),
        "trackeval_import": str(trackeval.__file__),
        "trackers_base_dir": str(trackers_base_dir),
        "gt_folder": str(gt_folder),
        "local_gt_folder": str(local_gt_folder),
        "detections_dir": str(detections_dir),
        "pedreftrack_tracker": args.pedreftrack_tracker,
        "trackers": trackers,
        "sequences": sequences,
        "split_to_eval": args.split_to_eval,
        "tracker_sub_folder": args.tracker_sub_folder,
        "proxies": PROXIES,
        "aggregation_unit": "GT_pedestrian_frame_occurrence",
        "association_context": "full_sequence_global_alignment",
        "fps": float(args.fps),
        "gap_length_definition": args.gap_length_definition,
        "gap_duration_formula": (
            "continuous_missing_GT_occurrences / fps"
            if args.gap_length_definition == "missing_frames"
            else "(next_matched_frame - previous_matched_frame) / fps"
        ),
        "gap_population": (
            "bounded_internal_detector_gaps_missing_occurrences_only"
        ),
        "det_match_max_dist_m": float(args.det_match_max_dist_m),
        "spatial_context_source": (
            "all_global_GT_rigid_transform_invariant_XY_distances"
        ),
        "det_score_min": args.det_score_min,
        "detector_matching": (
            "per_frame_augmented_hungarian_local_xy_centres_all_local_gt"
        ),
        "detector_coordinate_frame": "JRDB_local_base_no_odometry_transform",
        "matchable_sim_thr": float(args.matchable_sim_thr),
        "analysis_population": "all_gt_occurrences",
        "eligibility_filter": "none",
        "reused_proxies_from": (
            str(args.reuse_proxies_from.resolve())
            if args.reuse_proxies_from is not None
            else None
        ),
        "reused_hota_events_from": (
            str(args.reuse_hota_events_from.resolve())
            if args.reuse_hota_events_from is not None
            else None
        ),
        "excluded_hota_event_reuse_trackers": sorted(
            exclude_hota_event_reuse
        ),
        "last_selective_event_update_trackers": sorted(
            recompute_hota_events_for
        ),
    }

    metadata_path = output_dir / "metadata.json"
    immutable_keys = [
        "cache_version",
        "trackeval_root",
        "trackers_base_dir",
        "gt_folder",
        "local_gt_folder",
        "detections_dir",
        "pedreftrack_tracker",
        "sequences",
        "split_to_eval",
        "tracker_sub_folder",
        "proxies",
        "aggregation_unit",
        "association_context",
        "fps",
        "gap_length_definition",
        "gap_duration_formula",
        "gap_population",
        "det_match_max_dist_m",
        "spatial_context_source",
        "det_score_min",
        "detector_matching",
        "detector_coordinate_frame",
        "matchable_sim_thr",
        "analysis_population",
        "eligibility_filter",
    ]

    if metadata_path.exists() and not args.force_recompute:
        with metadata_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        changed = [
            key
            for key in immutable_keys
            if previous.get(key) != metadata.get(key)
        ]
        if changed:
            details = "\n".join(
                f"  {key}: existing={previous.get(key)!r}, "
                f"requested={metadata.get(key)!r}"
                for key in changed
            )
            raise RuntimeError(
                "The output directory contains a cache made with different "
                f"settings:\n{details}\nUse a new --output-dir, or rerun "
                "with --force-recompute."
            )
        trackers = list(
            dict.fromkeys([*previous.get("trackers", []), *trackers])
        )
        metadata["trackers"] = trackers

    requested_event_overrides = (
        exclude_hota_event_reuse | recompute_hota_events_for
    )
    unknown_event_overrides = sorted(
        requested_event_overrides.difference(trackers)
    )
    if unknown_event_overrides:
        raise ValueError(
            "Selective HOTA event tracker names are not present in the cache "
            f"tracker list: {unknown_event_overrides}"
        )
    if recompute_hota_events_for and not metadata_path.exists():
        raise FileNotFoundError(
            "--recompute-hota-events-for requires an existing HOTA cache "
            f"--output-dir with metadata.json: {metadata_path}"
        )

    metadata["signature"] = stable_signature(
        {key: metadata.get(key) for key in immutable_keys}
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[HOTA cache stage 1] cache: {output_dir}")
    print(
        f"[HOTA cache stage 1] sequences={len(sequences)}, "
        f"trackers={len(trackers)}"
    )
    print(
        "[HOTA cache stage 1] aggregation unit: individual GT pedestrian "
        "occurrences; association context: full sequence"
    )
    print(
        f"[HOTA cache stage 1] bounded-gap length definition: "
        f"{args.gap_length_definition}; fps={float(args.fps):g}"
    )
    if args.gap_length_definition == "missing_frames":
        print(
            "[HOTA cache stage 1] gap duration = continuous missing frames / fps; "
            f"15 frames = {15.0 / float(args.fps):g} s"
        )

    if recompute_hota_events_for:
        print(
            "[HOTA cache stage 1] selective HOTA event update: "
            + ", ".join(sorted(recompute_hota_events_for))
        )
    elif exclude_hota_event_reuse:
        print(
            "[HOTA cache stage 1] HOTA event reuse excluded for: "
            + ", ".join(sorted(exclude_hota_event_reuse))
        )

    if recompute_hota_events_for:
        require_existing_proxy_cache(output_dir, proxy_dir, sequences)
        print(
            "[HOTA cache stage 1] event-only update: existing frame-proxy cache "
            "validated and proxy loading/rewrite skipped"
        )
    else:
        proxy_reuse_status: list[dict[str, Any]] = []
        if args.reuse_proxies_from is not None and not args.force_recompute:
            source_proxy_root = validate_proxy_reuse_source(
                args.reuse_proxies_from,
                metadata,
            )
            for seq in sequences:
                proxy_reuse_status.append(
                    reuse_proxy_sequence_files(
                        source_proxy_root,
                        proxy_dir,
                        seq,
                    )
                )
            counts = (
                pd.DataFrame(proxy_reuse_status)["status"]
                .value_counts()
                .to_dict()
            )
            print(
                "[HOTA cache stage 1] proxy files prepared from reuse source: "
                f"{counts}"
            )

        all_frame_rows: list[dict[str, Any]] = []
        proxy_support: list[dict[str, Any]] = []
        sequence_iter = (
            tqdm(sequences, desc="[HOTA cache proxies] sequences")
            if tqdm is not None
            else sequences
        )
        for seq in sequence_iter:
            rows, support = precompute_proxy_sequence(
                dataset=pedreftrack_dataset,
                seq=seq,
                pedreftrack_tracker=args.pedreftrack_tracker,
                local_gt_folder=local_gt_folder,
                detections_dir=detections_dir,
                proxy_dir=proxy_dir,
                fps=args.fps,
                gap_length_definition=args.gap_length_definition,
                det_match_max_dist_m=args.det_match_max_dist_m,
                det_score_min=args.det_score_min,
                force=args.force_recompute,
            )
            all_frame_rows.extend(rows)
            proxy_support.append(support)

        frames_df = pd.DataFrame(all_frame_rows)
        frames_csv = output_dir / "frame_proxy_values.csv.gz"
        frames_df.to_csv(frames_csv, index=False)
        parquet_path = output_dir / "frame_proxy_values.parquet"
        try:
            frames_df.to_parquet(parquet_path, index=False)
            parquet_status = str(parquet_path)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            parquet_status = (
                f"not written ({type(exc).__name__}: {exc})"
            )
            print(
                "[HOTA cache stage 1] Parquet unavailable; CSV.GZ remains complete: "
                f"{exc}"
            )

        pd.DataFrame(proxy_support).to_csv(
            output_dir / "proxy_support_by_sequence.csv", index=False
        )
        print(
            f"[HOTA cache stage 1] frame occurrences: {frames_csv} "
            f"({len(frames_df)} rows)"
        )
        print(f"[HOTA cache stage 1] parquet: {parquet_status}")

    tasks: list[tuple[str, str, Path]] = []
    for tracker in trackers:
        tracker_folder = event_dir / safe_tracker_name(tracker)
        tracker_folder.mkdir(parents=True, exist_ok=True)
        for seq in sequences:
            tasks.append(
                (tracker, seq, tracker_folder / f"{seq}.npz")
            )

    event_manifest: list[dict[str, Any]] = []
    compute_tasks: list[tuple[str, str, Path]] = []
    forced_event_trackers = (
        exclude_hota_event_reuse | recompute_hota_events_for
    )
    source_root = (
        args.reuse_hota_events_from.resolve()
        if args.reuse_hota_events_from is not None
        else None
    )
    for tracker, seq, path in tasks:
        if recompute_hota_events_for and tracker not in recompute_hota_events_for:
            event_manifest.append(
                {
                    "tracker": tracker,
                    "seq": seq,
                    "status": (
                        "untouched_existing"
                        if path.exists()
                        else "untouched_missing"
                    ),
                    "path": str(path),
                }
            )
            continue

        if tracker in forced_event_trackers:
            compute_tasks.append((tracker, seq, path))
            continue

        if path.exists() and not args.force_recompute:
            event_manifest.append(
                {
                    "tracker": tracker,
                    "seq": seq,
                    "status": "cached",
                    "path": str(path),
                }
            )
            continue

        reused = None
        if source_root is not None and not args.force_recompute:
            reused = reuse_event_file(
                source_root,
                tracker,
                seq,
                path,
                force=False,
            )
        if reused is None:
            compute_tasks.append((tracker, seq, path))
        else:
            event_manifest.append(reused)

    t0 = perf_counter()
    worker_count = max(1, int(args.num_workers))
    if worker_count == 1:
        task_iter = (
            tqdm(compute_tasks, desc="[events] tracker/sequence")
            if tqdm is not None
            else compute_tasks
        )
        for tracker, seq, path in task_iter:
            event_manifest.append(
                precompute_hota_event_sequence(
                    trackeval_root=trackeval_root,
                    trackers_base_dir=trackers_base_dir,
                    gt_folder=gt_folder,
                    tracker=tracker,
                    seq=seq,
                    split_to_eval=args.split_to_eval,
                    tracker_sub_folder=args.tracker_sub_folder,
                    matchable_sim_thr=args.matchable_sim_thr,
                    output_path=path,
                    force=(
                        args.force_recompute
                        or tracker in forced_event_trackers
                    ),
                )
            )
    elif compute_tasks:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    precompute_hota_event_sequence,
                    trackeval_root=trackeval_root,
                    trackers_base_dir=trackers_base_dir,
                    gt_folder=gt_folder,
                    tracker=tracker,
                    seq=seq,
                    split_to_eval=args.split_to_eval,
                    tracker_sub_folder=args.tracker_sub_folder,
                    matchable_sim_thr=args.matchable_sim_thr,
                    output_path=path,
                    force=(
                        args.force_recompute
                        or tracker in forced_event_trackers
                    ),
                ): (tracker, seq)
                for tracker, seq, path in compute_tasks
            }
            iterator = (
                tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="[events] completed",
                )
                if tqdm is not None
                else as_completed(futures)
            )
            for future in iterator:
                tracker, seq = futures[future]
                try:
                    event_manifest.append(future.result())
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed event cache for {tracker}/{seq}"
                    ) from exc

    manifest_df = pd.DataFrame(event_manifest)
    manifest_df.to_csv(output_dir / "hota_event_manifest.csv", index=False)
    elapsed = perf_counter() - t0
    status_counts = (
        manifest_df["status"].value_counts().to_dict()
        if len(manifest_df) and "status" in manifest_df
        else {}
    )
    print(
        f"[HOTA cache stage 1] HOTA events complete in {elapsed:.1f} s | "
        f"{status_counts}"
    )
    print(
        f"[HOTA cache stage 1] cache signature: {metadata['signature']}"
    )


if __name__ == "__main__":
    main()
