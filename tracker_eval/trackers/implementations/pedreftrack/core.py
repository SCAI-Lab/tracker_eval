"""Lightweight PedRefTrack tracker with optional GT-assisted identity handling.

Both modes use the same seconds-based confirmation, 2D constant-velocity
Kalman filter, matched-residual coasting controller, and hidden-track
re-identification lifetime. Matched output geometry uses filtered XY, a fast
bottom/height filter, a slower size filter, and a short pi-periodic detector-
yaw mean. ``no_gt`` performs ordinary detector association, while
``gt_assisted`` uses GT identity bookkeeping and GT displacement prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Set

import math
import numpy as np

from .types import Box3D, Detection, FrameData

from .geometry import (
    linear_sum_assignment,
    cKDTree,
    _precompute_bev_rects,
    bev_iou_oriented_cached,
    _UnionFind,
    _gate_pair_distance_only,
)


# -----------------------------
# Config
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def _assign_cost_edges_with_dummies(
    cfg,
    n_tracks: int,
    n_detections: int,
    edges: List[Tuple[int, int, float]],
) -> List[Tuple[int, int]]:
    """Assign sparse cost edges with an explicit unmatched option per track."""
    if n_tracks == 0 or n_detections == 0 or not edges:
        return []

    forbidden = float(cfg.forbidden_cost)
    unmatched = float(cfg.unmatched_cost)

    union_find = _UnionFind(n_tracks + n_detections)
    for track_index, detection_index, _ in edges:
        union_find.union(
            int(track_index), n_tracks + int(detection_index)
        )

    component_tracks: Dict[int, Set[int]] = {}
    component_detections: Dict[int, Set[int]] = {}
    component_edges: Dict[int, List[Tuple[int, int, float]]] = {}
    for track_index, detection_index, edge_cost in edges:
        root = union_find.find(int(track_index))
        component_tracks.setdefault(root, set()).add(int(track_index))
        component_detections.setdefault(root, set()).add(
            int(detection_index)
        )
        component_edges.setdefault(root, []).append(
            (
                int(track_index),
                int(detection_index),
                float(edge_cost),
            )
        )

    matches: List[Tuple[int, int]] = []
    for root, edge_list in component_edges.items():
        tracks = sorted(component_tracks.get(root, set()))
        detections = sorted(component_detections.get(root, set()))
        if not tracks or not detections:
            continue
        track_lookup = {
            track_index: local
            for local, track_index in enumerate(tracks)
        }
        detection_lookup = {
            detection_index: local
            for local, detection_index in enumerate(detections)
        }
        n_local_tracks = len(tracks)
        n_local_detections = len(detections)
        cost = np.full(
            (
                n_local_tracks,
                n_local_detections + n_local_tracks,
            ),
            forbidden,
            dtype=np.float64,
        )
        for track_index, detection_index, edge_cost in edge_list:
            row = track_lookup[track_index]
            column = detection_lookup[detection_index]
            cost[row, column] = min(
                cost[row, column], float(edge_cost)
            )

        # A separate dummy prevents two tracks from competing for one shared
        # unmatched column while keeping every track free to remain unmatched.
        cost[
            np.arange(n_local_tracks),
            n_local_detections + np.arange(n_local_tracks),
        ] = unmatched

        if linear_sum_assignment is None:
            candidates: List[Tuple[float, int, int]] = []
            for row in range(n_local_tracks):
                for column in range(n_local_detections):
                    if cost[row, column] < unmatched:
                        candidates.append(
                            (float(cost[row, column]), row, column)
                        )
            candidates.sort(key=lambda item: item[0])
            used_rows: Set[int] = set()
            used_columns: Set[int] = set()
            for _, row, column in candidates:
                if row in used_rows or column in used_columns:
                    continue
                used_rows.add(row)
                used_columns.add(column)
                matches.append(
                    (tracks[row], detections[column])
                )
            continue

        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows.tolist(), columns.tolist()):
            if (
                column < n_local_detections
                and cost[row, column] < unmatched
                and cost[row, column] < forbidden * 0.5
            ):
                matches.append(
                    (tracks[row], detections[column])
                )
    return matches


def _assign_two_pass_hungarian_with_dummies(
    cfg,
    n_tracks: int,
    n_detections: int,
    edges: List[Tuple[int, int, float, float]],
    *,
    track_miss_dt: Optional[np.ndarray] = None,
    use_stale_penalty: bool = True,
) -> List[Tuple[int, int]]:
    """Associate by BEV IoU first, then XY distance on unmatched items."""
    if n_tracks == 0 or n_detections == 0 or not edges:
        return []

    stale_lambda = float(cfg.stale_lambda_m_per_s)
    stale_cap = float(cfg.stale_cap_s)
    use_stale = (
        bool(use_stale_penalty)
        and track_miss_dt is not None
        and stale_lambda > 0.0
        and stale_cap > 0.0
    )

    def _stale_cost(track_index: int) -> float:
        if not use_stale:
            return 0.0
        miss_dt = max(0.0, float(track_miss_dt[int(track_index)]))
        return stale_lambda * min(miss_dt, stale_cap)

    iou_threshold = float(cfg.assoc_iou_first_pass_thr)
    iou_edges = [
        (
            int(track_index),
            int(detection_index),
            1.0 - float(iou) + _stale_cost(int(track_index)),
        )
        for track_index, detection_index, iou, _ in edges
        if float(iou) >= iou_threshold
    ]
    first_matches = _assign_cost_edges_with_dummies(
        cfg,
        n_tracks,
        n_detections,
        iou_edges,
    )

    used_tracks = {int(track_index) for track_index, _ in first_matches}
    used_detections = {
        int(detection_index) for _, detection_index in first_matches
    }
    distance_edges = [
        (
            int(track_index),
            int(detection_index),
            float(distance) + _stale_cost(int(track_index)),
        )
        for track_index, detection_index, _, distance in edges
        if (
            int(track_index) not in used_tracks
            and int(detection_index) not in used_detections
        )
    ]
    second_matches = _assign_cost_edges_with_dummies(
        cfg,
        n_tracks,
        n_detections,
        distance_edges,
    )
    return first_matches + second_matches

PEDREFTRACK_MODES: Tuple[str, ...] = ("no_gt", "gt_assisted")


def normalize_pedreftrack_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in PEDREFTRACK_MODES:
        raise ValueError(
            f"Unknown PedRefTrack mode {value!r}; expected "
            f"{', '.join(PEDREFTRACK_MODES)}."
        )
    return mode


def pedreftrack_local_name(mode: str) -> str:
    return f"pedreftrack_{normalize_pedreftrack_mode(mode)}"


@dataclass(frozen=True)
class PedRefTrackConfig:
    """Configuration for the final two-mode PedRefTrack architecture."""

    # Select detector-only ``no_gt`` or oracle ``gt_assisted`` operation.
    mode: str = "no_gt"

    # Input detection frequency in hertz.
    fps: float = 15.0

    # Maximum short-term XY association distance in metres.
    dist_gate_m: float = 0.4

    # Maximum vertical association distance in metres; non-positive disables it.
    z_gate_m: float = 0.5

    # Minimum BEV IoU accepted by the first association pass.
    assoc_iou_first_pass_thr: float = 0.33

    # Bottom-elevation EMA half-life in seconds.
    geometry_bottom_half_life_s: float = 0.125

    # Height EMA half-life in seconds.
    geometry_height_half_life_s: float = 0.35

    # Length/width EMA half-life in seconds.
    geometry_size_half_life_s: float = 0.50

    # Finite matched-detector yaw averaging window in seconds.
    geometry_yaw_smoothing_window_s: float = 0.25

    # Supported time in seconds at which confirmation reaches its lowest threshold.
    confirmation_target_s: float = 0.25

    # Detector score required to confirm from one matched observation.
    confirmation_one_hit_score: float = 0.95

    # Lowest mean detector score accepted once confirmation_target_s is reached.
    confirmation_min_score: float = 0.50

    # Maximum detector gap in seconds retained for an unconfirmed track.
    tentative_max_gap_s: float = 0.50

    # Required track history in seconds before residual-based coasting is enabled.
    motion_robustness_history_s: float = 1.00

    # Recent residual window in seconds that can conservatively shorten coasting.
    motion_robustness_immediate_history_s: float = 0.25

    # Residual error in metres below which the maximum coast duration is used.
    motion_error_free_m: float = 0.05

    # Additional residual error in metres that halves the coasting extension.
    motion_error_half_decay_m: float = 0.038

    # Minimum motion-selected detector-gap output duration in seconds.
    T_out_min_s: float = 0.50

    # Maximum motion-selected detector-gap output duration in seconds.
    T_out_max_s: float = 2.0

    # Hidden-track retention duration in seconds for moving pedestrians.
    T_reid_base_s: float = 2.5

    # Hidden-track retention duration in seconds for static pedestrians.
    T_reid_static_s: float = 5.0

    # Number of matched centres used by the internal static-motion test.
    static_window: int = 15

    # Maximum filtered speed in metres per second classified as static.
    v_static_thr_mps: float = 0.30

    # Maximum centre spread in metres classified as static.
    jitter_thr_m: float = 0.20

    # Maximum candidates retained per track before assignment.
    assoc_topk: int = 10

    # Assignment cost used for each track's explicit unmatched option.
    unmatched_cost: float = 1.0

    # Added assignment cost per second of tentative-track staleness.
    stale_lambda_m_per_s: float = 0.20

    # Maximum staleness duration in seconds contributing to assignment cost.
    stale_cap_s: float = 1.5

    # Centre distance in metres at which predicted closing paths cap coasting.
    near_collision_distance_m: float = 0.40

    # Initial Kalman position standard deviation in metres.
    kf_initial_pos_std_m: float = 0.15

    # Initial Kalman velocity standard deviation in metres per second.
    kf_initial_vel_std_mps: float = 1.00

    # Kalman position-measurement standard deviation in metres.
    kf_measurement_std_m: float = 0.15

    # Kalman acceleration-noise standard deviation in metres per second squared.
    kf_acceleration_std_mps2: float = 1.50

    # Squared Mahalanobis threshold for uncertainty-aware association.
    kf_maha_gate_d2: float = 9.21

    # Maximum covariance-derived XY association radius in metres.
    kf_max_gate_m: float = 1.00

    # Internal finite cost assigned to forbidden Hungarian edges.
    forbidden_cost: float = 1e6

    # Confidence written to every exported tracker box.
    output_score: float = 1.0

    # Identity stride separating successive GT-assisted re-identification epochs.
    gt_stride: int = 100_000

    # Starting identity offset for ordinary detector-associated tracks.
    fp_offset: int = 10_000_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_pedreftrack_mode(self.mode))
        if not np.isfinite(float(self.fps)) or float(self.fps) <= 0.0:
            raise ValueError("fps must be positive and finite")

        positive_time_fields = {
            "confirmation_target_s": self.confirmation_target_s,
            "tentative_max_gap_s": self.tentative_max_gap_s,
            "motion_robustness_history_s": self.motion_robustness_history_s,
            "motion_robustness_immediate_history_s":
                self.motion_robustness_immediate_history_s,
            "T_reid_base_s": self.T_reid_base_s,
            "T_reid_static_s": self.T_reid_static_s,
        }
        invalid_times = [
            name
            for name, value in positive_time_fields.items()
            if not np.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid_times:
            raise ValueError(
                "PedRefTrack time parameters must be positive and finite: "
                + ", ".join(invalid_times)
            )
        if float(self.motion_robustness_immediate_history_s) > float(
            self.motion_robustness_history_s
        ):
            raise ValueError(
                "motion_robustness_immediate_history_s must not exceed "
                "motion_robustness_history_s"
            )
        if (
            not np.isfinite(float(self.motion_error_free_m))
            or float(self.motion_error_free_m) < 0.0
        ):
            raise ValueError(
                "motion_error_free_m must be non-negative and finite"
            )
        if (
            not np.isfinite(float(self.motion_error_half_decay_m))
            or float(self.motion_error_half_decay_m) <= 0.0
        ):
            raise ValueError(
                "motion_error_half_decay_m must be positive and finite"
            )
        one_hit_score = float(self.confirmation_one_hit_score)
        minimum_score = float(self.confirmation_min_score)
        if (
            not np.isfinite(one_hit_score)
            or not np.isfinite(minimum_score)
            or not 0.0 <= minimum_score <= one_hit_score <= 1.0
        ):
            raise ValueError(
                "Require 0 <= confirmation_min_score <= "
                "confirmation_one_hit_score <= 1"
            )
        if (
            not np.isfinite(float(self.T_out_min_s))
            or not np.isfinite(float(self.T_out_max_s))
            or float(self.T_out_min_s) < 0.0
            or float(self.T_out_max_s) < float(self.T_out_min_s)
        ):
            raise ValueError(
                "Require 0 <= T_out_min_s <= T_out_max_s, both finite"
            )
        if (
            not np.isfinite(float(self.unmatched_cost))
            or float(self.unmatched_cost) <= 0.0
            or float(self.unmatched_cost)
            >= 0.5 * float(self.forbidden_cost)
        ):
            raise ValueError(
                "unmatched_cost must be positive, finite, and well below "
                "forbidden_cost"
            )
        if (
            not np.isfinite(float(self.assoc_iou_first_pass_thr))
            or not 0.0 <= float(self.assoc_iou_first_pass_thr) <= 1.0
        ):
            raise ValueError(
                "assoc_iou_first_pass_thr must be finite and in [0, 1]"
            )
        positive_geometry_fields = {
            "geometry_bottom_half_life_s":
                self.geometry_bottom_half_life_s,
            "geometry_height_half_life_s":
                self.geometry_height_half_life_s,
            "geometry_size_half_life_s": self.geometry_size_half_life_s,
            "geometry_yaw_smoothing_window_s":
                self.geometry_yaw_smoothing_window_s,
        }
        invalid_geometry = [
            name
            for name, value in positive_geometry_fields.items()
            if not np.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid_geometry:
            raise ValueError(
                "Geometry-filter parameters must be positive and finite: "
                + ", ".join(invalid_geometry)
            )
        if (
            not np.isfinite(float(self.near_collision_distance_m))
            or float(self.near_collision_distance_m) <= 0.0
        ):
            raise ValueError(
                "near_collision_distance_m must be positive and finite"
            )
        positive_kf_fields = {
            "kf_initial_pos_std_m": self.kf_initial_pos_std_m,
            "kf_initial_vel_std_mps": self.kf_initial_vel_std_mps,
            "kf_measurement_std_m": self.kf_measurement_std_m,
            "kf_acceleration_std_mps2": self.kf_acceleration_std_mps2,
            "kf_maha_gate_d2": self.kf_maha_gate_d2,
            "kf_max_gate_m": self.kf_max_gate_m,
        }
        invalid = [
            name
            for name, value in positive_kf_fields.items()
            if not np.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid:
            raise ValueError(
                "Kalman-filter parameters must be positive and finite: "
                + ", ".join(invalid)
            )
        if float(self.kf_max_gate_m) < float(self.dist_gate_m):
            raise ValueError(
                "kf_max_gate_m must be at least dist_gate_m"
            )


# -----------------------------
# Track state
# -----------------------------

@dataclass
class _TrackState:
    # Output id used in exported detections
    tid: int

    # Family bookkeeping
    is_gt: bool
    gt_id: Optional[int] = None
    epoch: int = 0

    # Boxes
    out_box: Box3D = None  # ordinary tracker state used for association/output

    # Frequency-independent pre-confirmation state. ``confirm_support_s`` is
    # the gap-discounted detector-supported age used to query the confirmation
    # boundary. ``confirm_score_weight_s`` and ``confirm_score_integral_s``
    # form a time-weighted score estimate. Evidence is capped to the target
    # duration so weak ancient scores cannot dominate indefinitely.
    confirm_support_s: float = 0.0
    confirm_score_weight_s: float = 0.0
    confirm_score_integral_s: float = 0.0
    confirm_gap_s: float = 0.0
    confirm_has_match: bool = False

    confirmed: bool = False

    # Frozen output-coasting horizon for the current detection gap.
    # None means the track is currently observed or no gap limit has been set.
    coast_limit_s: Optional[float] = None

    # Matched pre-correction KF residuals: (wall-clock timestamp, error metres).
    # Missing frames create no samples and do not directly change the error.
    motion_samples: List[Tuple[float, float]] = field(default_factory=list)

    # Start of the logical detector-supported track history. It is used only
    # to prevent a very young track with one small residual from immediately
    # receiving a long coast. Gaps do not reset this wall-clock age.
    motion_history_start_t: Optional[float] = None

    # Times (seconds)
    last_seen_t: float = 0.0   # last time matched to a detection (not GT)
    last_pred_t: float = 0.0   # last time out_box was advanced (prediction or observation)

    # Observed history (detections only)
    obs_centers: List[Tuple[float, float]] = field(default_factory=list)

    # Filtered kinematics (2D)
    filt_xy: Tuple[float, float] = (0.0, 0.0)
    filt_vxy: Tuple[float, float] = (0.0, 0.0)

    # Minimal CV Kalman state [px, py, vx, vy] and its 4x4 covariance.
    # Arrays are owned per track; no FilterPy object or general matrix inverse
    # is created in the frame loop.
    kf_x: np.ndarray = field(
        default_factory=lambda: np.zeros((4,), dtype=np.float64)
    )
    kf_P: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    kf_initialized: bool = False

    # Causal output-geometry state. Bottom elevation is filtered separately
    # from height so height noise cannot directly move the pedestrian's feet.
    geometry_initialized: bool = False
    geometry_bottom_z: float = 0.0
    geometry_l: float = 0.0
    geometry_w: float = 0.0
    geometry_h: float = 0.0
    geometry_yaw_cos2: float = 1.0
    geometry_yaw_sin2: float = 0.0
    geometry_yaw_history: List[Tuple[float, float]] = field(
        default_factory=list
    )
    geometry_last_match_t: Optional[float] = None

    def reset_epoch_state(self) -> None:
        self.confirm_support_s = 0.0
        self.confirm_score_weight_s = 0.0
        self.confirm_score_integral_s = 0.0
        self.confirm_gap_s = 0.0
        self.confirm_has_match = False
        self.confirmed = False
        self.coast_limit_s = None
        self.motion_samples = []
        self.motion_history_start_t = None
        self.obs_centers = []
        self.geometry_initialized = False
        self.geometry_bottom_z = 0.0
        self.geometry_l = 0.0
        self.geometry_w = 0.0
        self.geometry_h = 0.0
        self.geometry_yaw_cos2 = 1.0
        self.geometry_yaw_sin2 = 0.0
        self.geometry_yaw_history = []
        self.geometry_last_match_t = None
        

    def push_observation(
        self,
        cfg: PedRefTrackConfig,
        box: Box3D,
    ) -> None:
        # Store the corrected centre, so static inference and prediction refer
        # to the same filtered trajectory rather than a second raw-centre
        # velocity estimate.
        centre = (
            self.filt_xy
            if np.all(np.isfinite(np.asarray(self.filt_xy, dtype=float)))
            else (float(box.cx), float(box.cy))
        )
        self.obs_centers.append(
            (float(centre[0]), float(centre[1]))
        )
        if len(self.obs_centers) > cfg.static_window:
            self.obs_centers = self.obs_centers[-cfg.static_window:]


def _innovation_terms(
    tr: _TrackState,
    cfg: PedRefTrackConfig,
) -> Tuple[float, float, float, float, float, float]:
    """Return S and S^-1 terms for the track's 2D position innovation."""
    measurement_var = float(cfg.kf_measurement_std_m) ** 2
    covariance = tr.kf_P
    s00 = float(covariance[0, 0] + measurement_var)
    s01 = float(0.5 * (covariance[0, 1] + covariance[1, 0]))
    s11 = float(covariance[1, 1] + measurement_var)

    # Numerical floors are deliberately tiny relative to detector noise.
    s00 = max(s00, 1e-9)
    s11 = max(s11, 1e-9)
    determinant = max(s00 * s11 - s01 * s01, 1e-12)
    inv00 = s11 / determinant
    inv01 = -s01 / determinant
    inv11 = s00 / determinant
    return s00, s01, s11, inv00, inv01, inv11


def _build_kf_candidates(
    cfg: PedRefTrackConfig,
    tracks: Sequence[_TrackState],
    track_boxes: Sequence[Box3D],
    detection_boxes: Sequence[Box3D],
    *,
    detection_xy: np.ndarray,
    track_corners: Sequence[np.ndarray],
    track_areas: Sequence[float],
    detection_corners: Sequence[np.ndarray],
    detection_areas: Sequence[float],
) -> List[Tuple[int, int, float, float]]:
    """
    Build sparse KF-aware association edges.

    ``dist_gate_m`` preserves the original short-horizon acceptance region.
    Beyond it, a detection must pass the covariance-aware Mahalanobis gate.
    The candidate radius is always capped by ``kf_max_gate_m``.

    The fourth edge value is the raw XY distance in metres used by the second
    association pass.
    """
    n_tracks = len(tracks)
    n_detections = len(detection_boxes)
    if n_tracks == 0 or n_detections == 0:
        return []

    base_gate = float(cfg.dist_gate_m)
    max_gate = float(cfg.kf_max_gate_m)
    gate_d2 = float(cfg.kf_maha_gate_d2)
    topk = int(max(1, cfg.assoc_topk))
    z_gate = float(cfg.z_gate_m)

    track_xy = np.asarray(
        [[box.cx, box.cy] for box in track_boxes],
        dtype=np.float64,
    )
    radii = np.empty((n_tracks,), dtype=np.float64)
    innovation_cache: List[
        Tuple[float, float, float, float, float, float]
    ] = []
    for index, tr in enumerate(tracks):
        terms = _innovation_terms(tr, cfg)
        innovation_cache.append(terms)
        s00, s01, s11, _, _, _ = terms
        discriminant = max(
            0.0,
            (s00 - s11) * (s00 - s11) + 4.0 * s01 * s01,
        )
        largest_eigenvalue = 0.5 * (
            s00 + s11 + math.sqrt(discriminant)
        )
        maha_radius = math.sqrt(
            gate_d2 * max(largest_eigenvalue, 1e-9)
        )
        radii[index] = min(
            max_gate,
            max(base_gate, float(maha_radius)),
        )

    if cKDTree is not None:
        tree = cKDTree(detection_xy)
        neighbours = tree.query_ball_point(track_xy, r=radii)
    else:
        neighbours = []
        for index in range(n_tracks):
            delta = detection_xy - track_xy[index : index + 1, :]
            distance_squared = np.sum(delta * delta, axis=1)
            neighbours.append(
                np.where(
                    distance_squared <= radii[index] * radii[index]
                )[0].tolist()
            )

    edges: List[Tuple[int, int, float, float]] = []
    for track_index, candidate_indices in enumerate(neighbours):
        if not candidate_indices:
            continue
        tr = tracks[track_index]
        box_track = track_boxes[track_index]
        _, _, _, inv00, inv01, inv11 = innovation_cache[
            track_index
        ]

        candidates: List[Tuple[float, int]] = []
        for detection_index in candidate_indices:
            box_detection = detection_boxes[int(detection_index)]
            if (
                z_gate > 0.0
                and abs(float(box_track.cz) - float(box_detection.cz))
                > z_gate
            ):
                continue

            dx = float(box_detection.cx - tr.kf_x[0])
            dy = float(box_detection.cy - tr.kf_x[1])
            euclidean = math.sqrt(dx * dx + dy * dy)
            if euclidean > max_gate:
                continue

            d2 = (
                inv00 * dx * dx
                + 2.0 * inv01 * dx * dy
                + inv11 * dy * dy
            )
            within_base_gate = euclidean <= base_gate
            within_maha_gate = d2 <= gate_d2
            if not (within_base_gate or within_maha_gate):
                continue

            candidates.append(
                (float(euclidean), int(detection_index))
            )

        if len(candidates) > topk:
            candidates.sort(key=lambda item: item[0])
            candidates = candidates[:topk]

        for euclidean, detection_index in candidates:
            iou = bev_iou_oriented_cached(
                track_corners[track_index],
                float(track_areas[track_index]),
                detection_corners[detection_index],
                float(detection_areas[detection_index]),
            )
            edges.append(
                (
                    int(track_index),
                    int(detection_index),
                    float(iou),
                    float(euclidean),
                )
            )
    return edges


# -----------------------------
# PedRefTrack tracker
# -----------------------------

class PedRefTrack:
    """Pure-Python PedRefTrack implementation.

    The core has no dependency on the tracker-evaluation runner or ROS2.  It
    accepts the lightweight datatypes from :mod:`.types`; adapters are
    responsible for converting their external message or benchmark formats.
    """

    def __init__(
        self,
        *,
        cfg: Optional[PedRefTrackConfig] = None,
    ) -> None:
        self.cfg = cfg or PedRefTrackConfig()
        self.mode = normalize_pedreftrack_mode(self.cfg.mode)
        self._gt_assisted = self.mode == "gt_assisted"

        # GT-family tracks: gt_id -> TrackState
        self._gt_tracks: Dict[int, _TrackState] = {}

        # FP-family tracks: tid -> TrackState
        self._fp_tracks: Dict[int, _TrackState] = {}
        self._fp_next_tid: int = int(self.cfg.fp_offset)

        # Time
        self._t: float = 0.0
        self._frame_idx: int = 0

        # GT displacement cache (gt_id -> previous GT box)
        self._prev_gt_by_id: Dict[int, Box3D] = {}

    def reset_sequence(self, seq_name: str = "") -> None:
        self._gt_tracks = {}
        self._fp_tracks = {}
        self._fp_next_tid = int(self.cfg.fp_offset)

        self._t = 0.0
        self._frame_idx = 0

        self._prev_gt_by_id = {}

    # -----------------------------
    # Confirmation and motion-consistency evidence
    # -----------------------------

    def _confirmation_required_score(self, support_s: float) -> float:
        """Return the normalized late-drop score boundary.

        The shape is fixed in normalized time, so
        ``confirmation_target_s`` retains its physical-time meaning at any
        detector frequency. With the default 0.95/0.50 endpoints and 15 Hz,
        the boundary is approximately 0.95, 0.88, 0.78, 0.56 and 0.50 on
        matched observations 1--5. Only the existing latency and endpoint
        parameters remain user-facing.
        """
        cfg = self.cfg
        fraction = _clamp(
            float(max(0.0, support_s))
            / max(1e-12, float(cfg.confirmation_target_s)),
            0.0,
            1.0,
        )

        # Required-score height as a fraction of the endpoint range.
        # The sharpest descent is intentionally delayed until the latter part
        # of the requested confirmation interval.
        knots = (
            (0.0, 1.0),
            (4.0 / 15.0, 38.0 / 45.0),
            (8.0 / 15.0, 28.0 / 45.0),
            (12.0 / 15.0, 6.0 / 45.0),
            (1.0, 0.0),
        )
        shape = 0.0
        for (left_x, left_y), (right_x, right_y) in zip(
            knots[:-1], knots[1:]
        ):
            if fraction <= right_x + 1e-12:
                local = (
                    (fraction - left_x)
                    / max(1e-12, right_x - left_x)
                )
                shape = (1.0 - local) * left_y + local * right_y
                break

        return float(
            float(cfg.confirmation_min_score)
            + shape
            * (
                float(cfg.confirmation_one_hit_score)
                - float(cfg.confirmation_min_score)
            )
        )

    def _reset_unconfirmed_evidence(self, tr: _TrackState) -> None:
        """Clear only tentative evidence; the logical GT placeholder may live."""
        tr.confirm_support_s = 0.0
        tr.confirm_score_weight_s = 0.0
        tr.confirm_score_integral_s = 0.0
        tr.confirm_gap_s = 0.0
        tr.confirm_has_match = False

    def _confirmation_on_match(
        self,
        tr: _TrackState,
        score: Optional[float],
        dt: float,
    ) -> None:
        """Update seconds-based confirmation evidence on a matched frame."""
        if tr.confirmed:
            tr.confirm_gap_s = 0.0
            return

        dt = float(max(1e-9, dt))
        observed_score = (
            float(score)
            if score is not None and np.isfinite(float(score))
            # Deliberately conservative and not user-configurable.
            else 0.0
        )
        observed_score = _clamp(observed_score, 0.0, 1.0)

        if tr.confirm_gap_s > 0.0:
            retention = _clamp(
                1.0
                - float(tr.confirm_gap_s)
                / float(self.cfg.tentative_max_gap_s),
                0.0,
                1.0,
            )
            tr.confirm_support_s *= retention
            tr.confirm_score_weight_s *= retention
            tr.confirm_score_integral_s *= retention

        if tr.confirm_has_match:
            # The first observation initializes the clock at 0 seconds. Each
            # later matched update advances it by the actual elapsed time.
            tr.confirm_support_s = min(
                float(self.cfg.confirmation_target_s),
                tr.confirm_support_s + dt,
            )
        else:
            tr.confirm_support_s = 0.0
            tr.confirm_has_match = True

        tr.confirm_score_weight_s += dt
        tr.confirm_score_integral_s += dt * observed_score
        tr.confirm_gap_s = 0.0

        # Retain no more score-history weight than the target confirmation
        # duration. Scaling both terms preserves their weighted mean.
        maximum_weight = float(self.cfg.confirmation_target_s)
        if tr.confirm_score_weight_s > maximum_weight:
            scale = maximum_weight / tr.confirm_score_weight_s
            tr.confirm_score_weight_s *= scale
            tr.confirm_score_integral_s *= scale

        recent_score = (
            tr.confirm_score_integral_s
            / max(1e-12, tr.confirm_score_weight_s)
        )
        required_score = self._confirmation_required_score(
            tr.confirm_support_s
        )
        if recent_score + 1e-12 >= required_score:
            tr.confirmed = True

    def _confirmation_on_miss(self, tr: _TrackState, dt: float) -> None:
        """Accumulate a tentative gap; confirmation never occurs on a miss."""
        if tr.confirmed or not tr.confirm_has_match:
            return

        dt = float(max(0.0, dt))
        tr.confirm_gap_s += dt
        if tr.confirm_gap_s + 1e-12 >= float(
            self.cfg.tentative_max_gap_s
        ):
            self._reset_unconfirmed_evidence(tr)

    def _motion_on_match(
        self,
        tr: _TrackState,
        innovation_m: float,
    ) -> None:
        """Record one matched pre-correction KF residual."""
        now_s = float(self._t)
        if tr.motion_history_start_t is None:
            tr.motion_history_start_t = now_s
        tr.motion_samples.append(
            (now_s, float(max(0.0, innovation_m)))
        )
        self._refresh_motion_error(tr)

    def _refresh_motion_error(self, tr: _TrackState) -> float:
        """Refresh effective error using only matched samples in time windows."""
        cfg = self.cfg
        now_s = float(self._t)
        history_s = float(cfg.motion_robustness_history_s)
        immediate_s = float(
            cfg.motion_robustness_immediate_history_s
        )
        oldest_history_s = now_s - history_s

        tr.motion_samples = [
            (sample_t, residual_m)
            for sample_t, residual_m in tr.motion_samples
            if sample_t >= oldest_history_s - 1e-12
        ]
        if not tr.motion_samples:
            return math.inf

        all_residuals = np.asarray(
            [residual_m for _, residual_m in tr.motion_samples],
            dtype=np.float64,
        )
        mean_all_m = float(np.mean(all_residuals))

        oldest_immediate_s = now_s - immediate_s
        immediate_residuals = np.asarray(
            [
                residual_m
                for sample_t, residual_m in tr.motion_samples
                if sample_t >= oldest_immediate_s - 1e-12
            ],
            dtype=np.float64,
        )
        mean_immediate_m = (
            float(np.mean(immediate_residuals))
            if immediate_residuals.size
            else mean_all_m
        )
        effective_error_m = max(mean_all_m, mean_immediate_m)
        return float(effective_error_m)

    def _motion_extension_factor(self, error_m: float) -> float:
        """Return q=2^(-max(0,e-e_free)/delta_e_half) in [0, 1]."""
        cfg = self.cfg
        if not np.isfinite(float(error_m)):
            return 0.0
        excess_error = max(
            0.0,
            float(error_m) - float(cfg.motion_error_free_m),
        )
        return _clamp(
            2.0 ** (
                -excess_error
                / max(1e-12, float(cfg.motion_error_half_decay_m))
            ),
            0.0,
            1.0,
        )

    def _T_out_from_motion_error(self, tr: _TrackState) -> float:
        """Map matched-residual error to duration after a full track history."""
        cfg = self.cfg
        minimum_s = float(cfg.T_out_min_s)
        maximum_s = float(cfg.T_out_max_s)
        if maximum_s <= minimum_s:
            return minimum_s

        # A trajectory must span the requested wall-clock history before one
        # or even a few accurate matches may extend output. Detector gaps do
        # not reset this age and do not otherwise reduce robustness.
        history_start_s = tr.motion_history_start_t
        history_available = (
            history_start_s is not None
            and float(self._t) - float(history_start_s) + 1e-12
            >= float(cfg.motion_robustness_history_s)
        )
        if not history_available:
            return minimum_s

        effective_error_m = self._refresh_motion_error(tr)
        extension_factor = self._motion_extension_factor(
            effective_error_m
        )
        return float(
            minimum_s
            + (maximum_s - minimum_s) * extension_factor
        )

    @staticmethod
    def _clearance_entry_time_s(
        relative_position: np.ndarray,
        relative_velocity: np.ndarray,
        clearance_m: float,
        horizon_s: float,
    ) -> Optional[float]:
        """First future time at which two CV centres enter a clearance disc."""
        relative_position = np.asarray(
            relative_position, dtype=np.float64
        ).reshape(2)
        relative_velocity = np.asarray(
            relative_velocity, dtype=np.float64
        ).reshape(2)
        clearance_sq = float(clearance_m) ** 2
        c_term = float(relative_position @ relative_position - clearance_sq)
        if c_term <= 0.0:
            return 0.0

        a_term = float(relative_velocity @ relative_velocity)
        if a_term <= 1e-12:
            return None
        b_term = float(2.0 * (relative_position @ relative_velocity))
        discriminant = b_term * b_term - 4.0 * a_term * c_term
        if discriminant < 0.0:
            return None

        root = math.sqrt(max(0.0, discriminant))
        enter_s = (-b_term - root) / (2.0 * a_term)
        exit_s = (-b_term + root) / (2.0 * a_term)
        if exit_s < 0.0:
            return None
        enter_s = max(0.0, float(enter_s))
        if enter_s > float(horizon_s) + 1e-12:
            return None
        return enter_s

    def _clearance_limited_coast_s(
        self,
        tr: _TrackState,
        motion_limit_s: float,
    ) -> float:
        """Cheap detector-only near-collision cap evaluated once per gap.

        Only other confirmed tracks supported in the current frame are used.
        A pair is considered only when it starts outside the configured
        clearance, is closing, and its relative CV path enters that clearance.
        Pedestrians already close at gap onset are deliberately ignored here:
        their current proximity alone is not evidence that CV coasting will
        create an impossible interaction. GT-assisted modes keep their
        ordinary motion-selected horizon.
        """
        cfg = self.cfg
        motion_limit_s = float(motion_limit_s)
        if self.mode != "no_gt" or not tr.confirmed:
            return motion_limit_s

        frame_dt = 1.0 / max(1e-6, float(cfg.fps))
        current_gap_s = max(0.0, float(self._t - tr.last_seen_t))
        remaining_s = max(0.0, motion_limit_s - current_gap_s)
        if remaining_s <= 1e-12:
            return motion_limit_s

        clearance_m = float(cfg.near_collision_distance_m)
        target_position = np.asarray(tr.kf_x[:2], dtype=np.float64)
        target_velocity = np.asarray(tr.kf_x[2:4], dtype=np.float64)

        earliest_entry_s: Optional[float] = None
        all_tracks = list(self._gt_tracks.values()) + list(
            self._fp_tracks.values()
        )
        for other in all_tracks:
            if other is tr or not other.confirmed or not other.kf_initialized:
                continue
            # Require current detector support. This prevents one uncertain
            # coasting trajectory from unnecessarily truncating another.
            other_gap_s = max(0.0, float(self._t - other.last_seen_t))
            if other_gap_s > 1.5 * frame_dt + 1e-12:
                continue

            other_position = np.asarray(
                other.kf_x[:2], dtype=np.float64
            )
            other_velocity = np.asarray(
                other.kf_x[2:4], dtype=np.float64
            )
            relative_position = target_position - other_position
            relative_velocity = target_velocity - other_velocity
            start_distance_m = float(np.linalg.norm(relative_position))

            # Existing proximity is not itself a reason to truncate. This
            # avoids penalizing people already walking together in a group.
            if start_distance_m <= clearance_m + 1e-12:
                continue

            # Require clear closing motion before solving for entry. A
            # non-negative radial dot product means constant/separating range
            # under the relative CV model.
            if float(relative_position @ relative_velocity) >= -1e-12:
                continue

            entry_s = self._clearance_entry_time_s(
                relative_position,
                relative_velocity,
                clearance_m,
                remaining_s,
            )
            if entry_s is not None and (
                earliest_entry_s is None
                or entry_s < earliest_entry_s
            ):
                earliest_entry_s = float(entry_s)

        if earliest_entry_s is None:
            return motion_limit_s

        # Stop one nominal frame before entering the near-collision region.
        # Physical plausibility has priority over the ordinary motion minimum,
        # so this cap may deliberately reduce output below T_out_min_s.
        clearance_limit_s = (
            current_gap_s + max(0.0, earliest_entry_s - frame_dt)
        )
        return _clamp(
            clearance_limit_s,
            0.0,
            motion_limit_s,
        )

    def _start_miss_if_needed(self, tr: _TrackState) -> None:
        """Freeze the motion- and clearance-selected duration at gap start."""
        if tr.coast_limit_s is None:
            motion_limit_s = self._T_out_from_motion_error(tr)
            tr.coast_limit_s = self._clearance_limited_coast_s(
                tr, motion_limit_s
            )

    @staticmethod
    def _end_miss(tr: _TrackState) -> None:
        """Clear the frozen coast duration when a detection is matched."""
        tr.coast_limit_s = None

    # -----------------------------
    # Static inference (detections only)
    # -----------------------------

    def _is_static(self, tr: _TrackState) -> bool:
        cfg = self.cfg
        if len(tr.obs_centers) < max(3, cfg.static_window // 2):
            return False
        xs = np.array([p[0] for p in tr.obs_centers], dtype=np.float64)
        ys = np.array([p[1] for p in tr.obs_centers], dtype=np.float64)
        mx = float(xs.mean())
        my = float(ys.mean())
        rad = float(np.max(np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)))
        return (
            math.hypot(*tr.filt_vxy) < float(cfg.v_static_thr_mps)
            and rad < float(cfg.jitter_thr_m)
        )

    # -----------------------------
    # Output geometry
    # -----------------------------

    @staticmethod
    def _ema_alpha(dt_s: float, half_life_s: float) -> float:
        """Time-based EMA gain whose value is frequency independent."""
        dt_s = max(0.0, float(dt_s))
        half_life_s = max(1e-12, float(half_life_s))
        return _clamp(
            -math.expm1(-math.log(2.0) * dt_s / half_life_s),
            0.0,
            1.0,
        )

    @staticmethod
    def _detector_geometry(
        box: Box3D,
    ) -> Tuple[float, float, float, float, float]:
        """Return bottom, length, width, height and length-axis yaw.

        The longer horizontal side is consistently represented as ``length``.
        When detector dimensions arrive in the opposite order, swapping them
        and rotating yaw by pi/2 preserves the original physical footprint.
        """
        length = max(1e-6, float(box.l))
        width = max(1e-6, float(box.w))
        height = max(1e-6, float(box.h))
        yaw = float(box.rot_z)
        if width > length:
            length, width = width, length
            yaw += 0.5 * math.pi
        bottom = float(box.cz) - 0.5 * height
        return bottom, length, width, height, yaw

    @staticmethod
    def _set_geometry_yaw(tr: _TrackState, yaw: float) -> None:
        tr.geometry_yaw_cos2 = float(math.cos(2.0 * float(yaw)))
        tr.geometry_yaw_sin2 = float(math.sin(2.0 * float(yaw)))

    @staticmethod
    def _geometry_yaw(tr: _TrackState) -> float:
        return float(
            0.5
            * math.atan2(
                float(tr.geometry_yaw_sin2),
                float(tr.geometry_yaw_cos2),
            )
        )

    def _initialize_geometry(
        self,
        tr: _TrackState,
        det_box: Box3D,
    ) -> None:
        """Initialize output geometry from a detector box, never from GT."""
        bottom, length, width, height, yaw = self._detector_geometry(
            det_box
        )
        tr.geometry_bottom_z = float(bottom)
        tr.geometry_l = float(length)
        tr.geometry_w = float(width)
        tr.geometry_h = float(height)
        self._set_geometry_yaw(tr, yaw)
        tr.geometry_yaw_history = [(float(self._t), float(yaw))]
        tr.geometry_last_match_t = float(self._t)
        tr.geometry_initialized = True

    def _update_detector_yaw_mean(
        self,
        tr: _TrackState,
        detector_yaw: float,
    ) -> None:
        """Average matched detector yaw over a finite pi-periodic window."""
        now_s = float(self._t)
        oldest_s = (
            now_s - float(self.cfg.geometry_yaw_smoothing_window_s)
        )
        tr.geometry_yaw_history = [
            (sample_t, sample_yaw)
            for sample_t, sample_yaw in tr.geometry_yaw_history
            if sample_t >= oldest_s - 1e-12
        ]
        tr.geometry_yaw_history.append(
            (now_s, float(detector_yaw))
        )

        mean_cos2 = float(
            np.mean(
                [
                    math.cos(2.0 * sample_yaw)
                    for _, sample_yaw in tr.geometry_yaw_history
                ]
            )
        )
        mean_sin2 = float(
            np.mean(
                [
                    math.sin(2.0 * sample_yaw)
                    for _, sample_yaw in tr.geometry_yaw_history
                ]
            )
        )
        norm = math.hypot(mean_cos2, mean_sin2)
        if norm <= 1e-12:
            # Exactly orthogonal samples make the circular mean undefined;
            # prefer the current detector orientation for a prompt turn.
            self._set_geometry_yaw(tr, detector_yaw)
        else:
            tr.geometry_yaw_cos2 = float(mean_cos2 / norm)
            tr.geometry_yaw_sin2 = float(mean_sin2 / norm)

    def _update_geometry(
        self,
        tr: _TrackState,
        det_box: Box3D,
    ) -> None:
        """Update causal geometry filters on one detector-supported frame."""
        if not tr.geometry_initialized:
            self._initialize_geometry(tr, det_box)
            return

        cfg = self.cfg
        nominal_dt_s = 1.0 / max(1e-6, float(cfg.fps))
        previous_t = tr.geometry_last_match_t
        elapsed_s = (
            nominal_dt_s
            if previous_t is None
            else max(nominal_dt_s, float(self._t) - float(previous_t))
        )
        bottom, length, width, height, detector_yaw = (
            self._detector_geometry(det_box)
        )

        alpha_bottom = self._ema_alpha(
            elapsed_s, cfg.geometry_bottom_half_life_s
        )
        alpha_height = self._ema_alpha(
            elapsed_s, cfg.geometry_height_half_life_s
        )
        alpha_size = self._ema_alpha(
            elapsed_s, cfg.geometry_size_half_life_s
        )
        tr.geometry_bottom_z += alpha_bottom * (
            float(bottom) - tr.geometry_bottom_z
        )
        tr.geometry_h += alpha_height * (
            float(height) - tr.geometry_h
        )
        tr.geometry_l += alpha_size * (
            float(length) - tr.geometry_l
        )
        tr.geometry_w += alpha_size * (
            float(width) - tr.geometry_w
        )

        self._update_detector_yaw_mean(tr, detector_yaw)

        tr.geometry_last_match_t = float(self._t)

    def _box_from_geometry(
        self,
        tr: _TrackState,
        cx: float,
        cy: float,
        *,
        fallback: Box3D,
    ) -> Box3D:
        """Construct one output box from the current filtered geometry."""
        if not tr.geometry_initialized:
            return Box3D(
                cx=float(cx),
                cy=float(cy),
                cz=float(fallback.cz),
                l=float(fallback.l),
                w=float(fallback.w),
                h=float(fallback.h),
                rot_z=float(fallback.rot_z),
            )
        return Box3D(
            cx=float(cx),
            cy=float(cy),
            cz=float(tr.geometry_bottom_z + 0.5 * tr.geometry_h),
            l=float(tr.geometry_l),
            w=float(tr.geometry_w),
            h=float(tr.geometry_h),
            rot_z=self._geometry_yaw(tr),
        )
    
    def _kf_reset(
        self,
        tr: _TrackState,
        xy: Tuple[float, float],
    ) -> None:
        """Initialize one compact [px, py, vx, vy] Kalman state."""
        cfg = self.cfg
        tr.kf_x = np.array(
            [float(xy[0]), float(xy[1]), 0.0, 0.0],
            dtype=np.float64,
        )
        position_variance = float(cfg.kf_initial_pos_std_m) ** 2
        velocity_variance = float(cfg.kf_initial_vel_std_mps) ** 2
        tr.kf_P = np.diag(
            [
                position_variance,
                position_variance,
                velocity_variance,
                velocity_variance,
            ]
        ).astype(np.float64)
        tr.kf_initialized = True
        tr.filt_xy = (float(xy[0]), float(xy[1]))
        tr.filt_vxy = (0.0, 0.0)

    def _ensure_kf(self, tr: _TrackState, box: Box3D) -> None:
        if not tr.kf_initialized:
            self._kf_reset(
                tr,
                (float(box.cx), float(box.cy)),
            )

    def _kf_predict(
        self,
        tr: _TrackState,
        dt: float,
        *,
        oracle_delta_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Predict one 4D CV state and covariance.

        For ``full`` GT PedRefTrack, the ordinary covariance propagation is kept
        but the predicted mean uses the exact GT displacement. This preserves
        the existing oracle semantics while maintaining a useful fallback
        velocity if GT displacement later becomes unavailable.
        """
        if tr.out_box is None:
            return
        self._ensure_kf(tr, tr.out_box)
        dt = float(max(1e-6, dt))
        previous_x = tr.kf_x.copy()

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        predicted_x = transition @ previous_x

        acceleration_variance = (
            float(self.cfg.kf_acceleration_std_mps2) ** 2
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        process_noise = acceleration_variance * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        predicted_P = (
            transition @ tr.kf_P @ transition.T + process_noise
        )

        if oracle_delta_xy is not None:
            dx, dy = oracle_delta_xy
            predicted_x[0] = float(previous_x[0] + dx)
            predicted_x[1] = float(previous_x[1] + dy)
            predicted_x[2] = float(dx / dt)
            predicted_x[3] = float(dy / dt)

        tr.kf_x = predicted_x
        tr.kf_P = 0.5 * (predicted_P + predicted_P.T)
        tr.filt_xy = (
            float(predicted_x[0]),
            float(predicted_x[1]),
        )
        tr.filt_vxy = (
            float(predicted_x[2]),
            float(predicted_x[3]),
        )

    def _kf_correct(
        self,
        tr: _TrackState,
        det_box: Box3D,
    ) -> float:
        """Correct XY and update filtered output geometry from a match."""
        self._ensure_kf(tr, det_box)
        measurement = np.array(
            [float(det_box.cx), float(det_box.cy)],
            dtype=np.float64,
        )
        innovation = measurement - tr.kf_x[:2]
        innovation_m = float(np.linalg.norm(innovation))
        _, _, _, inv00, inv01, inv11 = _innovation_terms(
            tr, self.cfg
        )
        inverse_innovation = np.array(
            [[inv00, inv01], [inv01, inv11]],
            dtype=np.float64,
        )

        cross_covariance = tr.kf_P[:, :2].copy()
        kalman_gain = cross_covariance @ inverse_innovation
        tr.kf_x = tr.kf_x + kalman_gain @ innovation

        # H selects the first two state entries, so H@P is P[:2, :].
        corrected_P = tr.kf_P - kalman_gain @ tr.kf_P[:2, :]
        corrected_P = 0.5 * (corrected_P + corrected_P.T)
        diagonal = np.maximum(np.diag(corrected_P), 1e-9)
        np.fill_diagonal(corrected_P, diagonal)
        tr.kf_P = corrected_P

        filtered_xy = (
            float(tr.kf_x[0]),
            float(tr.kf_x[1]),
        )
        tr.filt_xy = filtered_xy
        tr.filt_vxy = (
            float(tr.kf_x[2]),
            float(tr.kf_x[3]),
        )

        self._update_geometry(tr, det_box)
        tr.out_box = self._box_from_geometry(
            tr,
            filtered_xy[0],
            filtered_xy[1],
            fallback=det_box,
        )
        return float(max(0.0, innovation_m))


    # -----------------------------
    # GT helpers
    # -----------------------------

    @staticmethod
    def _build_gt_maps(gt_dets: Optional[Sequence[Detection]]) -> Dict[int, Box3D]:
        out: Dict[int, Box3D] = {}
        if gt_dets is None:
            return out
        for g in gt_dets:
            try:
                gid = int(g.track_id)
            except Exception:
                continue
            if g.box is None:
                continue
            out[gid] = g.box
        return out

    def _gt_delta_for_id(
        self, gt_id: int, gt_now: Dict[int, Box3D]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Return (dx, dy, dz) from prev_gt to current_gt for gt_id.
        (We intentionally do NOT use yaw as a gate or predictor.)
        """
        if gt_id not in gt_now:
            return None
        if gt_id not in self._prev_gt_by_id:
            return None
        prev = self._prev_gt_by_id[gt_id]
        cur = gt_now[gt_id]
        dx = float(cur.cx - prev.cx)
        dy = float(cur.cy - prev.cy)
        dz = float(cur.cz - prev.cz)
        return dx, dy, dz

    def _gt_tid(self, gt_id: int, epoch: int) -> int:
        return int(gt_id + int(epoch) * int(self.cfg.gt_stride))

    # -----------------------------
    # Prediction (coasting)
    # -----------------------------

    def _predict_track(
        self,
        tr: _TrackState,
        dt: float,
        *,
        gt_now: Optional[Dict[int, Box3D]] = None,
        allow_gt_motion: bool = False,
    ) -> None:
        """Advance the KF state, optionally with GT displacement."""
        box = tr.out_box
        if box is None:
            return

        oracle_delta_xy: Optional[Tuple[float, float]] = None
        dz = 0.0
        if (
            allow_gt_motion
            and tr.is_gt
            and tr.gt_id is not None
            and gt_now is not None
        ):
            delta = self._gt_delta_for_id(int(tr.gt_id), gt_now)
            if delta is not None:
                dx, dy, dz = delta
                oracle_delta_xy = (float(dx), float(dy))

        self._kf_predict(
            tr,
            dt,
            oracle_delta_xy=oracle_delta_xy,
        )

        if tr.geometry_initialized and abs(float(dz)) > 0.0:
            # GT-assisted prediction may move the whole filtered box
            # vertically, but detector geometry still initializes and updates
            # its height, extent and orientation.
            tr.geometry_bottom_z += float(dz)

        predicted_fallback = Box3D(
            cx=float(tr.kf_x[0]),
            cy=float(tr.kf_x[1]),
            cz=float(box.cz + dz),
            l=float(box.l),
            w=float(box.w),
            h=float(box.h),
            rot_z=float(box.rot_z),
        )
        tr.out_box = self._box_from_geometry(
            tr,
            float(tr.kf_x[0]),
            float(tr.kf_x[1]),
            fallback=predicted_fallback,
        )

    # -----------------------------
    # Public stepping API
    # -----------------------------

    def step(
        self,
        frame_id: str,
        detections: Sequence[Detection],
        *,
        timestamp: Optional[float] = None,
    ) -> FrameData:
        return self.step_with_gt(
            frame_id,
            detections,
            gt_dets=None,
            timestamp=timestamp,
        )

    def step_with_gt(
        self,
        frame_id: str,
        detections: Sequence[Detection],
        gt_dets: Optional[Sequence[Detection]] = None,
        *,
        timestamp: Optional[float] = None,
    ) -> FrameData:
        cfg = self.cfg

        # ---- time bookkeeping ----
        if timestamp is None:
            dt_frame = 1.0 / float(max(1e-6, cfg.fps))
            self._t = float(self._t + dt_frame)
        else:
            if self._frame_idx == 0:
                self._t = float(timestamp)
                dt_frame = 1.0 / float(max(1e-6, cfg.fps))
            else:
                dt_frame = float(max(1e-6, float(timestamp) - float(self._t)))
                self._t = float(timestamp)

        t_now = float(self._t)
        self._frame_idx += 1

        # no_gt deliberately ignores GT even if a caller supplies it.
        gt_now = (
            self._build_gt_maps(gt_dets)
            if self._gt_assisted
            else {}
        )

        # GT-assisted mode keeps one logical family per GT identity.
        for gid, gbox in gt_now.items():
            if gid not in self._gt_tracks:
                tid0 = self._gt_tid(gid, 0)
                xy0 = (float(gbox.cx), float(gbox.cy))
                tr = _TrackState(
                    tid=int(tid0),
                    is_gt=True,
                    gt_id=int(gid),
                    epoch=0,
                    out_box=gbox,
                    confirmed=False,
                    last_seen_t=t_now,   # prevents immediate "expiry before ever matched"
                    last_pred_t=t_now,
                    filt_xy=xy0,
                    filt_vxy=(0.0, 0.0),
                )
                self._kf_reset(tr, xy0)
                self._gt_tracks[gid] = tr

        # GT-family tracks use exact GT displacement in gt_assisted mode.
        for gid, tr in self._gt_tracks.items():
            if tr.last_pred_t < t_now - 1e-12:
                self._predict_track(
                    tr,
                    dt_frame,
                    gt_now=gt_now,
                    allow_gt_motion=True,
                )
                tr.last_pred_t = t_now
                self._gt_tracks[gid] = tr

        # Ordinary tracks always use the detector-only KF prediction.
        for tid, tr in list(self._fp_tracks.items()):
            if tr.last_pred_t < t_now - 1e-12:
                self._predict_track(
                    tr,
                    dt_frame,
                    gt_now=None,
                    allow_gt_motion=False,
                )
                tr.last_pred_t = t_now
                self._fp_tracks[tid] = tr

        # ---- Prepare detections arrays ----
        det_list = list(detections)
        det_indices: List[int] = []
        det_boxes: List[Box3D] = []

        for i, d in enumerate(det_list):
            if d.box is None:
                continue
            det_indices.append(i)
            det_boxes.append(d.box)

        det_xy = (
            np.array([[b.cx, b.cy] for b in det_boxes], dtype=np.float64)
            if det_boxes else np.zeros((0, 2), dtype=np.float64)
        )
        det_corners, det_areas = _precompute_bev_rects(det_boxes)

        # ============================================================
        # 1) Assign detections to GT tracks (GATE on prediction, COST on GT)
        # ============================================================
        gt_ids = list(self._gt_tracks.keys())

        # "gate" boxes = what the tracker believes (prediction / filtered state)
        gt_gate_boxes = [self._gt_tracks[gid].out_box for gid in gt_ids]
        gt_gate_xy = (
            np.array([[b.cx, b.cy] for b in gt_gate_boxes], dtype=np.float64)
            if gt_gate_boxes else np.zeros((0, 2), dtype=np.float64)
        )

        # "cost" boxes = actual GT at this frame if available, else fall back to gate box
        gt_cost_boxes = [gt_now.get(gid, self._gt_tracks[gid].out_box) for gid in gt_ids]
        gt_cost_corners, gt_cost_areas = _precompute_bev_rects(gt_cost_boxes)

        gt_miss_dt = np.array([t_now - self._gt_tracks[gid].last_seen_t for gid in gt_ids], dtype=np.float64)

        # Build sparse candidate edges:
        #  - neighbor query + dist/z gate are computed on gt_gate_boxes (prediction)
        #  - dist + iou used in cost are computed on gt_cost_boxes (actual GT when present)
        gt_edges: List[Tuple[int, int, float, float]] = []

        nG = len(gt_gate_boxes)
        nD = len(det_boxes)
        if nG > 0 and nD > 0:
            dist_gate = float(cfg.dist_gate_m)
            topk = int(max(1, cfg.assoc_topk))

            if cKDTree is not None:
                tree = cKDTree(det_xy)
                neigh = tree.query_ball_point(gt_gate_xy, r=dist_gate)
            else:
                neigh = []
                r2 = dist_gate ** 2
                for gi in range(nG):
                    dx = det_xy[:, 0] - gt_gate_xy[gi, 0]
                    dy = det_xy[:, 1] - gt_gate_xy[gi, 1]
                    d2 = dx * dx + dy * dy
                    idx = np.where(d2 <= r2)[0]
                    neigh.append(idx.tolist())

            for gi in range(nG):
                cand_js = neigh[gi]
                if not cand_js:
                    continue

                # top-k pruning using gate-space distance (prediction)
                if len(cand_js) > topk:
                    dxy = det_xy[np.array(cand_js)] - gt_gate_xy[gi : gi + 1, :]
                    d2 = np.sum(dxy * dxy, axis=1)
                    keep = np.argpartition(d2, topk)[:topk]
                    cand_js = [cand_js[k] for k in keep.tolist()]

                b_gate = gt_gate_boxes[gi]
                b_cost = gt_cost_boxes[gi]

                for dj in cand_js:
                    b_det = det_boxes[dj]

                    # gate check uses predicted state
                    ok, _dist_gate_val = _gate_pair_distance_only(cfg, b_gate, b_det)
                    if not ok:
                        continue

                    # cost uses GT state
                    dx = float(b_cost.cx - b_det.cx)
                    dy = float(b_cost.cy - b_det.cy)
                    dist_cost = float(math.sqrt(dx * dx + dy * dy))

                    iou = bev_iou_oriented_cached(
                        gt_cost_corners[gi], float(gt_cost_areas[gi]),
                        det_corners[dj], float(det_areas[dj]),
                    )

                    gt_edges.append((gi, dj, float(iou), float(dist_cost)))

        gt_matches_local = _assign_two_pass_hungarian_with_dummies(
            cfg, len(gt_gate_boxes), len(det_boxes), gt_edges,
            track_miss_dt=gt_miss_dt
        )

        gt_matches: Dict[int, int] = {}  # gt_id -> det_idx_in_det_list
        used_det_local: Set[int] = set()
        for ti, dj in gt_matches_local:
            gid = int(gt_ids[ti])
            det_global = int(det_indices[dj])
            gt_matches[gid] = det_global
            used_det_local.add(dj)

        # ---- Update GT tracks evidence + observation ----
        for gid in gt_ids:
            tr = self._gt_tracks[gid]
            if gid in gt_matches:
                di = gt_matches[gid]
                det = det_list[di]
                if det.box is None:
                    continue

                is_static = self._is_static(tr)
                T_reid = float(cfg.T_reid_static_s if is_static else cfg.T_reid_base_s)
                miss_dt = float(t_now - tr.last_seen_t)
                if miss_dt > T_reid:
                    tr.epoch += 1
                    tr.tid = self._gt_tid(int(gid), int(tr.epoch))
                    tr.reset_epoch_state()
                    xy = (float(det.box.cx), float(det.box.cy))
                    self._kf_reset(tr, xy)

                self._end_miss(tr)
                self._confirmation_on_match(tr, det.score, dt_frame)
                innovation_m = self._kf_correct(
                    tr, det.box
                )
                self._motion_on_match(tr, innovation_m)
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box)

                self._gt_tracks[gid] = tr
            else:
                self._start_miss_if_needed(tr)
                self._confirmation_on_miss(tr, dt_frame)
                self._gt_tracks[gid] = tr

        # ============================================================
        # 2) FP association on remaining detections (CONFIRMED-FIRST)
        # ============================================================

        # Remaining detections after GT matching, expressed as:
        #  - remaining_det_boxes: boxes to match against FP tracks
        #  - remaining_det_map: local index -> global det_list index
        remaining_det_boxes: List[Box3D] = []
        remaining_det_map: List[int] = []
        for local_j, global_i in enumerate(det_indices):
            if local_j in used_det_local:
                continue
            d = det_list[global_i]
            if d.box is None:
                continue
            remaining_det_map.append(global_i)
            remaining_det_boxes.append(d.box)

        # Confirmed tracks associate first; tentative tracks can only consume
        # detections left by the established trajectories.
        fp_confirmed_tids: List[int] = []
        fp_tentative_tids: List[int] = []
        for tid, tr in self._fp_tracks.items():
            if tr.confirmed:
                fp_confirmed_tids.append(int(tid))
            else:
                fp_tentative_tids.append(int(tid))

        fp_matched_tids: Set[int] = set()
        fp_used_det_locals: Set[int] = set()

        # Helper: run one FP matching pass for a given subset of tids,
        # using only currently-available remaining detections.
        def _match_fp_subset(fp_tids_subset: List[int]) -> None:
            nonlocal fp_matched_tids, fp_used_det_locals

            if not fp_tids_subset or not remaining_det_boxes:
                return

            # Build "available detections" view after previous FP pass consumption
            avail_det_locals = [j for j in range(len(remaining_det_boxes)) if j not in fp_used_det_locals]
            if not avail_det_locals:
                return

            avail_boxes = [remaining_det_boxes[j] for j in avail_det_locals]

            avail_xy = (
                np.array([[b.cx, b.cy] for b in avail_boxes], dtype=np.float64)
                if avail_boxes else np.zeros((0, 2), dtype=np.float64)
            )
            avail_corners, avail_areas = _precompute_bev_rects(avail_boxes)

            fp_tracks_subset = [
                self._fp_tracks[tid] for tid in fp_tids_subset
            ]
            fp_boxes_subset = [
                track.out_box for track in fp_tracks_subset
            ]
            fp_miss_dt = np.array([t_now - self._fp_tracks[tid].last_seen_t for tid in fp_tids_subset], dtype=np.float64)
            fp_corners, fp_areas = _precompute_bev_rects(fp_boxes_subset)

            edges = _build_kf_candidates(
                cfg,
                fp_tracks_subset,
                fp_boxes_subset,
                avail_boxes,
                detection_xy=avail_xy,
                track_corners=fp_corners,
                track_areas=fp_areas,
                detection_corners=avail_corners,
                detection_areas=avail_areas,
            )
            subset_is_confirmed = all(
                track.confirmed for track in fp_tracks_subset
            )
            matches_local = _assign_two_pass_hungarian_with_dummies(
                cfg, len(fp_boxes_subset), len(avail_boxes), edges,
                track_miss_dt=fp_miss_dt,
                # Confirmed tracks already receive first association priority.
                # Do not additionally make their absolute acceptance region
                # shrink with time since the last detection.
                use_stale_penalty=not subset_is_confirmed,
            )

            for ti, dj in matches_local:
                tid = int(fp_tids_subset[ti])
                tr = self._fp_tracks[tid]

                # dj is local index into avail_boxes -> map back to remaining_det_boxes local index
                det_local = int(avail_det_locals[dj])
                det_global_i = int(remaining_det_map[det_local])
                det = det_list[det_global_i]
                if det.box is None:
                    continue

                self._end_miss(tr)
                self._confirmation_on_match(tr, det.score, dt_frame)
                innovation_m = self._kf_correct(
                    tr, det.box
                )
                self._motion_on_match(tr, innovation_m)
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box)

                self._fp_tracks[tid] = tr
                fp_matched_tids.add(tid)
                fp_used_det_locals.add(det_local)

        # Pass 1: confirmed FP tracks get first shot.
        _match_fp_subset(fp_confirmed_tids)

        # Pass 2: tentative FP tracks use only remaining detections.
        _match_fp_subset(fp_tentative_tids)


        # ---- FP misses: short tentative lifetime, separate confirmed T_reid ----
        fp_to_delete: List[int] = []
        for tid, tr in list(self._fp_tracks.items()):
            tid_i = int(tid)
            if tid_i in fp_matched_tids:
                continue

            self._start_miss_if_needed(tr)
            self._confirmation_on_miss(tr, dt_frame)
            miss_dt = float(t_now - tr.last_seen_t)
            if not tr.confirmed:
                delete_track = (
                    miss_dt + 1e-12
                    >= float(cfg.tentative_max_gap_s)
                )
            else:
                is_static = self._is_static(tr)
                T_reid = float(
                    cfg.T_reid_static_s
                    if is_static
                    else cfg.T_reid_base_s
                )
                delete_track = miss_dt > T_reid
            if delete_track:
                fp_to_delete.append(tid_i)
            else:
                self._fp_tracks[tid_i] = tr

        for tid in fp_to_delete:
            self._fp_tracks.pop(tid, None)

        # ---- Spawn new FP tracks from remaining unmatched detections ----
        for det_local, det_global_i in enumerate(remaining_det_map):
            if det_local in fp_used_det_locals:
                continue
            det = det_list[det_global_i]
            if det.box is None:
                continue

            tid = int(self._fp_next_tid)
            self._fp_next_tid += 1
            xy0 = (float(det.box.cx), float(det.box.cy))
            tr = _TrackState(
                tid=tid,
                is_gt=False,
                gt_id=None,
                epoch=0,
                out_box=det.box,
                confirmed=False,
                last_seen_t=t_now,
                last_pred_t=t_now,
                motion_history_start_t=t_now,
                filt_xy=xy0,
                filt_vxy=(0.0, 0.0),
            )
            self._kf_reset(tr, xy0)
            self._initialize_geometry(tr, det.box)
            self._confirmation_on_match(tr, det.score, dt_frame)
            tr.push_observation(cfg, det.box)
            self._fp_tracks[tid] = tr

        # ---- Build output detections (GT + FP) ----
        out_dets: List[Detection] = []

        for gid, tr in self._gt_tracks.items():
            miss_dt = float(t_now - tr.last_seen_t)
            T_out = (
                tr.coast_limit_s
                if tr.coast_limit_s is not None
                else self._T_out_from_motion_error(tr)
            )
            if tr.confirmed and (miss_dt <= T_out or miss_dt <= 1e-6):
                out_dets.append(
                    Detection(
                        frame_id=str(frame_id),
                        track_id=int(tr.tid),
                        box=tr.out_box,
                        score=float(cfg.output_score),
                        label="pedestrian",
                        raw_label_id=None,
                    )
                )
                self._gt_tracks[gid] = tr

        for tid, tr in self._fp_tracks.items():
            miss_dt = float(t_now - tr.last_seen_t)
            T_out = (
                tr.coast_limit_s
                if tr.coast_limit_s is not None
                else self._T_out_from_motion_error(tr)
            )
            if tr.confirmed and (miss_dt <= T_out or miss_dt <= 1e-6):
                out_dets.append(
                    Detection(
                        frame_id=str(frame_id),
                        track_id=int(tr.tid),
                        box=tr.out_box,
                        score=float(cfg.output_score),
                        label="pedestrian",
                        raw_label_id=None,
                    )
                )
                self._fp_tracks[int(tid)] = tr

        # ---- Update GT cache for displacement ----
        self._prev_gt_by_id = dict(gt_now) if self._gt_assisted else {}

        return FrameData(frame_id=str(frame_id), dets=out_dets)
