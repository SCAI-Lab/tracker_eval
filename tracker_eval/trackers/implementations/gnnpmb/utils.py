"""Runtime-only utilities retained from the GNN-PMB implementation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


class BBox:
    """Small box representation used by the upstream association functions."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        h: float = 0.0,
        w: float = 0.0,
        l: float = 0.0,
        o: float = 0.0,
    ) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.h, self.w, self.l, self.o = float(h), float(w), float(l), float(o)
        self.s: Optional[float] = None

    @staticmethod
    def bbox2array(box: "BBox") -> np.ndarray:
        values = [box.x, box.y, box.z, box.o, box.l, box.w, box.h]
        if box.s is not None:
            values.append(float(box.s))
        return np.asarray(values, dtype=float)

    @staticmethod
    def array2bbox(values: Sequence[float]) -> "BBox":
        box = BBox(
            x=values[0], y=values[1], z=values[2], o=values[3],
            l=values[4], w=values[5], h=values[6],
        )
        if len(values) >= 8:
            box.s = float(values[7])
        return box

    @staticmethod
    def box2corners2d(box: "BBox") -> np.ndarray:
        center = np.array([box.x, box.y], dtype=float)
        local = np.array(
            [
                [box.l / 2, box.w / 2],
                [box.l / 2, -box.w / 2],
                [-box.l / 2, -box.w / 2],
                [-box.l / 2, box.w / 2],
            ],
            dtype=float,
        )
        cosine, sine = np.cos(box.o), np.sin(box.o)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        return local @ rotation.T + center


def _quat_wxyz_to_yaw(rotation: Sequence[float]) -> float:
    qw, qx, qy, qz = [float(value) for value in rotation]
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def nu_array2mot_bbox(entry: Dict[str, Any]) -> BBox:
    translation = entry["translation"]
    size = entry["size"]
    box = BBox(
        x=translation[0],
        y=translation[1],
        z=translation[2],
        w=size[0],
        l=size[1],
        h=size[2],
        o=_quat_wxyz_to_yaw(entry["rotation"]),
    )
    score = entry.get("detection_score", entry.get("tracking_score"))
    if score is not None:
        box.s = float(score)
    return box


def _polygon(box: BBox) -> Polygon:
    return Polygon(BBox.box2corners2d(box))


def iou3d(box_a: BBox, box_b: BBox) -> Tuple[float, float]:
    poly_a, poly_b = _polygon(box_a), _polygon(box_b)
    overlap_area = float(poly_a.intersection(poly_b).area)
    union_area = float(poly_a.area + poly_b.area - overlap_area)
    iou_2d = overlap_area / max(union_area, 1e-10)
    top_a, bottom_a = box_a.z + box_a.h / 2, box_a.z - box_a.h / 2
    top_b, bottom_b = box_b.z + box_b.h / 2, box_b.z - box_b.h / 2
    overlap_height = max(0.0, min(top_a, top_b) - max(bottom_a, bottom_b))
    overlap_volume = overlap_area * overlap_height
    union_volume = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - overlap_volume
    return float(iou_2d), float(overlap_volume / max(union_volume, 1e-10))


def giou2d(box_a: BBox, box_b: BBox) -> float:
    poly_a, poly_b = _polygon(box_a), _polygon(box_b)
    intersection = float(poly_a.intersection(poly_b).area)
    union = float(poly_a.area + poly_b.area - intersection)
    enclosing = float(poly_a.union(poly_b).convex_hull.area)
    if enclosing <= 0.0:
        return -1.0
    return float(intersection / max(union, 1e-10) - (enclosing - union) / enclosing)


def giou3d(measurement: Dict[str, Any], track: Dict[str, Any]) -> float:
    box_a, box_b = nu_array2mot_bbox(measurement), nu_array2mot_bbox(track)
    poly_a, poly_b = _polygon(box_a), _polygon(box_b)
    overlap_area = float(poly_a.intersection(poly_b).area)
    top_a, bottom_a = box_a.z + box_a.h / 2, box_a.z - box_a.h / 2
    top_b, bottom_b = box_b.z + box_b.h / 2, box_b.z - box_b.h / 2
    overlap_height = max(0.0, min(top_a, top_b) - max(bottom_a, bottom_b))
    intersection = overlap_area * overlap_height
    union = box_a.w * box_a.l * box_a.h + box_b.w * box_b.l * box_b.h - intersection
    enclosing_area = float(poly_a.union(poly_b).convex_hull.area)
    enclosing_height = max(top_a, top_b) - min(bottom_a, bottom_b)
    enclosing = enclosing_area * enclosing_height
    if enclosing <= 0.0:
        return -1.0
    return float(intersection / max(union, 1e-10) - (enclosing - union) / enclosing)


def readout_parameters(classification: str, parameters: Dict[str, Any]) -> Tuple[Any, ...]:
    values = parameters[classification]
    return (
        values["birth_rate"], values["p_s"], values["p_d"],
        values["use_ds_as_pd"], values["clutter_rate"],
        values["bernoulli_gating"], values["extraction_thr"],
        values["ber_thr"], values["poi_thr"], values["eB_thr"],
        values["detection_score_thr"], values["nms_score"],
        values["confidence_score"], values["P_init"],
    )


def gen_measurement_of_this_class(
    detection_score_thr: float,
    estimated_bboxes_at_current_frame: Iterable[Dict[str, Any]],
    classification: str,
) -> List[Dict[str, Any]]:
    return [
        box for box in estimated_bboxes_at_current_frame
        if box["detection_name"] == classification
        and float(box["detection_score"]) > float(detection_score_thr)
    ]


def nms(
    detections: Sequence[Dict[str, Any]],
    threshold: float = 0.1,
    threshold_high: float = 1.0,
    threshold_yaw: float = 0.3,
) -> List[int]:
    """Return score-ordered indices after the upstream 3D-IoU suppression rule."""
    del threshold_high, threshold_yaw
    boxes = [nu_array2mot_bbox(det) for det in detections]
    scores = np.asarray([box.s if box.s is not None else 0.0 for box in boxes])
    order = list(np.argsort(scores)[::-1].astype(int))
    keep: List[int] = []
    while order:
        index = order.pop(0)
        box = boxes[index]
        if box.l <= 0 or box.w <= 0 or box.h <= 0:
            continue
        keep.append(index)
        order = [
            candidate for candidate in order
            if iou3d(box, boxes[candidate])[1] <= float(threshold)
        ]
    return keep


def _orientation_difference(value: float) -> float:
    if value > np.pi / 2:
        value -= np.pi
    if value < -np.pi / 2:
        value += np.pi
    return float(value)


def _mahalanobis_distance(
    detection: BBox,
    track: BBox,
    inverse_innovation: Optional[np.ndarray],
) -> float:
    difference = (BBox.bbox2array(detection)[:7] - BBox.bbox2array(track)[:7]).reshape(-1, 1)
    difference[3] = _orientation_difference(float(difference[3]))
    if inverse_innovation is None:
        return float(np.linalg.norm(difference))
    return float(np.sqrt((difference.T @ inverse_innovation @ difference)[0, 0]))


def _distance_matrix(
    detections: Sequence[BBox],
    tracks: Sequence[BBox],
    association: str,
    innovation_matrices: Optional[Sequence[np.ndarray]],
) -> np.ndarray:
    matrix = np.empty((len(detections), len(tracks)), dtype=float)
    for det_index, detection in enumerate(detections):
        for track_index, track in enumerate(tracks):
            if association == "iou":
                matrix[det_index, track_index] = 1.0 - iou3d(detection, track)[1]
            elif association == "giou":
                matrix[det_index, track_index] = 1.0 - giou2d(detection, track)
            else:
                inverse = None
                if association == "m_dis" and innovation_matrices is not None:
                    inverse = np.linalg.inv(innovation_matrices[track_index])
                matrix[det_index, track_index] = _mahalanobis_distance(detection, track, inverse)
    return matrix


def associate_dets_to_tracks(
    detections: Sequence[BBox],
    tracks: Sequence[BBox],
    mode: str,
    asso: str,
    dist_threshold: float = 0.9,
    trk_innovation_matrix: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    if not detections or not tracks:
        return [], np.arange(len(detections)), np.arange(len(tracks))
    distances = _distance_matrix(detections, tracks, asso, trk_innovation_matrix)
    if mode == "bipartite":
        rows, columns = linear_sum_assignment(distances)
        candidates = np.stack([rows, columns], axis=1)
    elif mode == "greedy":
        candidates_list = []
        work = distances.copy()
        while work.size and np.isfinite(work).any():
            row, column = np.unravel_index(np.argmin(work), work.shape)
            candidates_list.append([row, column])
            work[row, :] = np.inf
            work[:, column] = np.inf
            if len(candidates_list) >= min(work.shape):
                break
        candidates = np.asarray(candidates_list, dtype=int).reshape(-1, 2)
    else:
        raise ValueError(f"Unknown association mode: {mode}")

    matches = [
        pair.astype(int) for pair in candidates
        if distances[pair[0], pair[1]] <= float(dist_threshold)
    ]
    matched_detections = {int(pair[0]) for pair in matches}
    matched_tracks = {int(pair[1]) for pair in matches}
    unmatched_detections = np.asarray(
        [index for index in range(len(detections)) if index not in matched_detections],
        dtype=int,
    )
    unmatched_tracks = np.asarray(
        [index for index in range(len(tracks)) if index not in matched_tracks],
        dtype=int,
    )
    return matches, unmatched_detections, unmatched_tracks


__all__ = [
    "associate_dets_to_tracks",
    "gen_measurement_of_this_class",
    "giou2d",
    "giou3d",
    "nms",
    "readout_parameters",
]
