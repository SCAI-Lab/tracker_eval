from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Set

import math
import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig

# Optional (fast) LAP solver
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None  # type: ignore

# Optional (fast) neighbor queries
try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore


# -----------------------------
# Geometry: BEV oriented IoU
# -----------------------------

def _wrap_to_pi(a: float) -> float:
    a = float(a)
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _rect_corners_xy(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """Return (4,2) rectangle corners CCW in XY."""
    dx = 0.5 * float(l)
    dy = 0.5 * float(w)
    local = np.array([[ dx,  dy],
                      [ dx, -dy],
                      [-dx, -dy],
                      [-dx,  dy]], dtype=np.float64)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)
    pts = local @ R.T
    pts[:, 0] += float(cx)
    pts[:, 1] += float(cy)
    return pts


def _poly_area(poly: np.ndarray) -> float:
    """Shoelace area for polygon (N,2). Assumes points ordered."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Check if point p is inside the half-plane to the left of edge a->b (CCW)."""
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-12


def _line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of segment p1->p2 with infinite line a->b, assuming they are not parallel."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(a[0]), float(a[1])
    x4, y4 = float(b[0]), float(b[1])

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2.copy()

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py], dtype=np.float64)


def _sutherland_hodgman_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """
    Clip a convex subject polygon by a convex clip polygon (both CCW).
    Returns intersection polygon points (K,2).
    """
    if subject is None or len(subject) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    output = subject.copy()
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        input_list = output
        if len(input_list) == 0:
            break
        out_pts = []
        s = input_list[-1]
        for e in input_list:
            if _is_inside(e, a, b):
                if not _is_inside(s, a, b):
                    out_pts.append(_line_intersection(s, e, a, b))
                out_pts.append(e)
            elif _is_inside(s, a, b):
                out_pts.append(_line_intersection(s, e, a, b))
            s = e
        output = np.array(out_pts, dtype=np.float64) if len(out_pts) > 0 else np.zeros((0, 2), dtype=np.float64)
    return output


def bev_iou_oriented(a: Box3D, b: Box3D) -> float:
    """Oriented rectangle IoU in XY (ignores z)."""
    pa = _rect_corners_xy(a.cx, a.cy, a.l, a.w, a.rot_z)
    pb = _rect_corners_xy(b.cx, b.cy, b.l, b.w, b.rot_z)
    ia = _poly_area(pa)
    ib = _poly_area(pb)
    if ia <= 0.0 or ib <= 0.0:
        return 0.0
    inter_poly = _sutherland_hodgman_clip(pa, pb)
    inter = _poly_area(inter_poly)
    union = ia + ib - inter
    if union <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, inter / union)))


def _precompute_bev_rects(boxes: Sequence[Box3D]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Precompute BEV rectangle corners and areas for a list of Box3D.
    Returns:
      corners: (N,4,2) float64
      areas:   (N,) float64
    """
    n = len(boxes)
    if n == 0:
        return np.zeros((0, 4, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    corners = np.zeros((n, 4, 2), dtype=np.float64)
    areas = np.zeros((n,), dtype=np.float64)
    for i, b in enumerate(boxes):
        c = _rect_corners_xy(b.cx, b.cy, b.l, b.w, b.rot_z)
        corners[i] = c
        areas[i] = _poly_area(c)
    return corners, areas


def bev_iou_oriented_cached(
    corners_a: np.ndarray, area_a: float,
    corners_b: np.ndarray, area_b: float,
) -> float:
    """Oriented IoU in XY using precomputed corners/areas."""
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_poly = _sutherland_hodgman_clip(corners_a, corners_b)
    inter = _poly_area(inter_poly)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    x = inter / union
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# -----------------------------
# Config
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


@dataclass(frozen=True)
class HeadroomConfig:
    """
    Headroom tracker config (v2).

    This tracker keeps two track families:
      - GT-family tracks: one logical track per GT id, output id changes only after re-id expiry (epoching)
      - FP-family tracks: standard tracks spawned from leftover detections

    GT is used only as:
      - oracle motion during misses (GT displacement)
      - identity bookkeeping for GT-family track IDs (epoching)
    """

    # Time base
    fps: float = 15.0

    # Association gating (distance-only, optional z)
    dist_gate_m: float = 0.35
    z_gate_m: float = 1.0  # set <=0 to disable

    # Candidate pruning (speed)
    assoc_topk: int = 10  # max candidates per track (after radius query)

    # Assignment cost: cost = (1 - iou) * assoc_iou_weight + dist
    assoc_iou_weight: float = 10.0
    forbidden_cost: float = 1e6  # used for non-edges in Hungarian

    # Evidence model (smoothed score + miss decay + hysteresis)
    score_floor: float = 0.50     # scores <= floor contribute ~0 evidence
    score_power: float = 1.7      # x = s_norm^p
    tau_hit_s: float = 0.20       # evidence rise time constant (seconds)
    tau_miss_s: float = 0.60      # evidence decay time constant (seconds)
    theta_on: float = 0.55        # confirm threshold
    theta_off: float = 0.35       # deconfirm threshold (hysteresis)
    min_hits: int = 2             # min matched detections before first confirmation

    # Output coasting after miss (seconds), based on CURRENT evidence (not peak)
    T_out_min_s: float = 0.10
    T_out_max_s: float = 0.50
    T_out_gamma: float = 1.5

    # ReID / forgetting (seconds), independent of evidence
    T_reid_base_s: float = 1.0
    T_reid_static_s: float = 2.0

    # Static inference from observed history only (detections only)
    static_window: int = 10
    v_static_thr_mps: float = 0.10
    jitter_thr_m: float = 0.20
    vel_ema_beta: float = 0.8

    # Prediction when GT displacement is not available (and for FP tracks)
    use_const_vel_coast: bool = True

    # Output score for exported tracks
    output_score: float = 1.0

    # ID namespaces
    gt_stride: int = 100_000
    fp_offset: int = 10_000_000


# -----------------------------
# Small utilities
# -----------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


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
    expired: bool = False  # True if miss_dt > T_reid (epoch ended); next matches belong to current epoch id already

    # Boxes
    out_box: Box3D = None  # predicted / last output box
    obs_box: Box3D = None  # last observed (matched detection) box

    # Evidence state
    evidence: float = 0.0
    confirmed: bool = False
    hits: int = 0  # matched detections since last reset (epoch start)

    # Times (seconds)
    last_seen_t: float = 0.0   # last time matched to a detection (not GT)
    last_emit_t: float = 0.0
    last_pred_t: float = 0.0   # last time out_box was advanced (prediction or observation)

    # Observed history (detections only)
    obs_centers: List[Tuple[float, float]] = field(default_factory=list)
    v_ema: float = 0.0
    vel_xy_ema: Tuple[float, float] = (0.0, 0.0)

    def reset_epoch_state(self) -> None:
        self.evidence = 0.0
        self.confirmed = False
        self.hits = 0
        self.obs_centers = []
        self.v_ema = 0.0
        self.vel_xy_ema = (0.0, 0.0)

    def push_observation(self, cfg: HeadroomConfig, box: Box3D, dt: float) -> None:
        self.obs_centers.append((float(box.cx), float(box.cy)))
        if len(self.obs_centers) > cfg.static_window:
            self.obs_centers = self.obs_centers[-cfg.static_window:]

        if len(self.obs_centers) >= 2 and dt > 1e-6:
            (x0, y0) = self.obs_centers[-2]
            (x1, y1) = self.obs_centers[-1]
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            speed = math.sqrt(vx * vx + vy * vy)

            b = float(cfg.vel_ema_beta)
            self.v_ema = b * self.v_ema + (1.0 - b) * speed
            self.vel_xy_ema = (
                b * self.vel_xy_ema[0] + (1.0 - b) * vx,
                b * self.vel_xy_ema[1] + (1.0 - b) * vy,
            )


# -----------------------------
# Headroom tracker (v2)
# -----------------------------

class HeadroomAdapter(TrackerBase):
    """
    Headroom tracker v2.

    Pipeline summary:
      1) Maintain GT-family tracks keyed by GT id (one logical track per GT id).
         Output tid = gt_id + epoch * gt_stride; epoch increments only when miss_dt > T_reid.
      2) Match detections to GT-family tracks first (component-wise Hungarian on sparse candidate graph),
         using distance-only gating and cost = (1-iou)*W + dist.
      3) Remaining detections are matched to FP-family tracks similarly; remaining detections spawn FP tracks.
      4) Evidence model uses only detector score:
         - smoothed evidence with tau_hit/tau_miss
         - confirmation with hysteresis (theta_on/theta_off) and min_hits
      5) On misses:
         - GT-family tracks coast using GT displacement when available (oracle motion)
         - FP tracks coast using constant velocity from detections only (optional)
         - output coasting window depends on current evidence (not peak)
    """

    def __init__(
        self,
        *,
        cfg: Optional[HeadroomConfig] = None,
        run_cfg: Optional[TrackerRunConfig] = None,
        name: str = "headroom",
        version: str = "2.0",
    ) -> None:
        self.cfg = cfg or HeadroomConfig()

        info = TrackerInfo(
            name=name,
            version=version,
            description="GT-assisted headroom tracker v2: GT-centric epochs + FP tracks, component Hungarian, distance-only gating, smoothed evidence w/ hysteresis.",
            extra={k: getattr(self.cfg, k) for k in self.cfg.__dict__.keys()},
        )
        super().__init__(info, run_cfg=run_cfg)

        # GT-family tracks: gt_id -> TrackState
        self._gt_tracks: Dict[int, _TrackState] = {}

        # FP-family tracks: tid -> TrackState
        self._fp_tracks: Dict[int, _TrackState] = {}
        self._fp_next_tid: int = int(self.cfg.fp_offset)

        # Time
        self._t: float = 0.0
        self._last_t: Optional[float] = None
        self._frame_idx: int = 0
        self._last_frame_id: Optional[str] = None

        # GT displacement cache (gt_id -> previous GT box)
        self._prev_gt_by_id: Dict[int, Box3D] = {}

    def _reset_sequence_impl(self, seq_name: str) -> None:
        self._gt_tracks = {}
        self._fp_tracks = {}
        self._fp_next_tid = int(self.cfg.fp_offset)

        self._t = 0.0
        self._last_t = None
        self._frame_idx = 0
        self._last_frame_id = None

        self._prev_gt_by_id = {}

    # -----------------------------
    # Evidence model
    # -----------------------------

    def _score_to_x(self, score: Optional[float]) -> float:
        s = float(score) if score is not None else 1.0
        floor = float(self.cfg.score_floor)
        sN = _clamp((s - floor) / max(1e-9, (1.0 - floor)), 0.0, 1.0)
        x = float(sN ** float(self.cfg.score_power))
        return x

    @staticmethod
    def _alpha(dt: float, tau: float) -> float:
        dt = float(max(0.0, dt))
        tau = float(max(1e-9, tau))
        return float(1.0 - math.exp(-dt / tau))

    def _evidence_on_match(self, tr: _TrackState, score: Optional[float], dt: float) -> None:
        a = self._alpha(dt, float(self.cfg.tau_hit_s))
        x = self._score_to_x(score)
        tr.evidence = float((1.0 - a) * tr.evidence + a * x)
        tr.hits += 1

        if (not tr.confirmed) and (tr.hits >= int(self.cfg.min_hits)) and (tr.evidence >= float(self.cfg.theta_on)):
            tr.confirmed = True

    def _evidence_on_miss(self, tr: _TrackState, dt: float) -> None:
        a = self._alpha(dt, float(self.cfg.tau_miss_s))
        tr.evidence = float((1.0 - a) * tr.evidence)

    def _T_out_from_evidence(self, E: float) -> float:
        cfg = self.cfg
        # map E in [theta_on..1] -> [0..1]
        x = _clamp((float(E) - float(cfg.theta_on)) / max(1e-9, (1.0 - float(cfg.theta_on))), 0.0, 1.0)
        x = float(x ** float(cfg.T_out_gamma))
        return float(cfg.T_out_min_s + (cfg.T_out_max_s - cfg.T_out_min_s) * x)

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
        return (float(tr.v_ema) < float(cfg.v_static_thr_mps)) and (rad < float(cfg.jitter_thr_m))

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

    def _predict_track(self, tr: _TrackState, dt: float, gt_now: Optional[Dict[int, Box3D]] = None) -> None:
        """
        Advance tr.out_box forward by dt.
          - GT tracks: use GT displacement if available, else fallback CV/hold.
          - FP tracks: fallback CV/hold.
        """
        cfg = self.cfg
        b = tr.out_box
        if b is None:
            return

        applied = False
        if tr.is_gt and tr.gt_id is not None and gt_now is not None:
            delta = self._gt_delta_for_id(int(tr.gt_id), gt_now)
            if delta is not None:
                dx, dy, dz = delta
                tr.out_box = Box3D(
                    cx=float(b.cx + dx),
                    cy=float(b.cy + dy),
                    cz=float(b.cz + dz),
                    l=float(b.l),
                    w=float(b.w),
                    h=float(b.h),
                    rot_z=float(b.rot_z),  # keep last observed yaw (det yaw is noisy; don't predict yaw)
                )
                applied = True

        if (not applied) and cfg.use_const_vel_coast:
            is_static = self._is_static(tr)
            if is_static:
                # Hold still
                return
            vx, vy = tr.vel_xy_ema
            tr.out_box = Box3D(
                cx=float(b.cx + vx * dt),
                cy=float(b.cy + vy * dt),
                cz=float(b.cz),
                l=float(b.l),
                w=float(b.w),
                h=float(b.h),
                rot_z=float(b.rot_z),
            )

    # -----------------------------
    # Gating + candidate generation
    # -----------------------------

    def _gate_pair_distance_only(self, box_tr: Box3D, box_det: Box3D) -> Tuple[bool, float]:
        """
        Distance-only gate (+ optional z).
        Returns (ok, dist_xy).
        """
        cfg = self.cfg
        if float(cfg.z_gate_m) > 0.0:
            if abs(float(box_tr.cz - box_det.cz)) > float(cfg.z_gate_m):
                return (False, float("inf"))

        dx = float(box_tr.cx - box_det.cx)
        dy = float(box_tr.cy - box_det.cy)
        dist = float(math.sqrt(dx * dx + dy * dy))
        if dist > float(cfg.dist_gate_m):
            return (False, dist)
        return (True, dist)

    def _build_candidates(
        self,
        track_boxes: List[Box3D],
        det_boxes: List[Box3D],
        *,
        track_xy: np.ndarray,
        det_xy: np.ndarray,
        track_corners: np.ndarray,
        track_areas: np.ndarray,
        det_corners: np.ndarray,
        det_areas: np.ndarray,
    ) -> List[Tuple[int, int, float, float]]:
        """
        Build sparse candidate edges (ti, dj, iou, dist) using radius query and optional top-k pruning.
        """
        cfg = self.cfg
        nT = len(track_boxes)
        nD = len(det_boxes)
        if nT == 0 or nD == 0:
            return []

        # Build neighbor structure for detections
        if cKDTree is not None:
            tree = cKDTree(det_xy)
            # Query within dist_gate
            neigh = tree.query_ball_point(track_xy, r=float(cfg.dist_gate_m))
        else:
            # Fallback: brute force
            neigh = []
            for i in range(nT):
                dx = det_xy[:, 0] - track_xy[i, 0]
                dy = det_xy[:, 1] - track_xy[i, 1]
                d2 = dx * dx + dy * dy
                idx = np.where(d2 <= float(cfg.dist_gate_m) ** 2)[0]
                neigh.append(idx.tolist())

        topk = int(max(1, cfg.assoc_topk))
        edges: List[Tuple[int, int, float, float]] = []

        for ti in range(nT):
            cand_js = neigh[ti]
            if not cand_js:
                continue

            # If too many candidates, keep closest top-k by distance
            if len(cand_js) > topk:
                dxy = det_xy[np.array(cand_js)] - track_xy[ti : ti + 1, :]
                d2 = np.sum(dxy * dxy, axis=1)
                keep = np.argpartition(d2, topk)[:topk]
                cand_js = [cand_js[k] for k in keep.tolist()]

            btr = track_boxes[ti]
            for dj in cand_js:
                bdet = det_boxes[dj]
                ok, dist = self._gate_pair_distance_only(btr, bdet)
                if not ok:
                    continue
                iou = bev_iou_oriented_cached(
                    track_corners[ti], float(track_areas[ti]),
                    det_corners[dj], float(det_areas[dj]),
                )
                edges.append((ti, dj, float(iou), float(dist)))

        return edges

    # -----------------------------
    # Component-wise Hungarian assignment
    # -----------------------------

    def _assign_component_hungarian(
        self,
        nT: int,
        nD: int,
        edges: List[Tuple[int, int, float, float]],
    ) -> List[Tuple[int, int]]:
        """
        Solve assignment on the sparse graph by connected components.
        Returns matches as (ti, dj).

        Cost: (1 - iou) * assoc_iou_weight + dist
        """
        if nT == 0 or nD == 0 or not edges:
            return []

        W = float(self.cfg.assoc_iou_weight)
        big = float(self.cfg.forbidden_cost)

        # Build connected components over bipartite graph
        uf = _UnionFind(nT + nD)
        for ti, dj, _, _ in edges:
            uf.union(ti, nT + dj)

        comp_tracks: Dict[int, Set[int]] = {}
        comp_dets: Dict[int, Set[int]] = {}
        comp_edges: Dict[int, List[Tuple[int, int, float, float]]] = {}

        for ti, dj, iou, dist in edges:
            r = uf.find(ti)
            comp_tracks.setdefault(r, set()).add(ti)
            comp_dets.setdefault(r, set()).add(dj)
            comp_edges.setdefault(r, []).append((ti, dj, iou, dist))

        matches: List[Tuple[int, int]] = []

        for r, e_list in comp_edges.items():
            Tset = sorted(comp_tracks.get(r, set()))
            Dset = sorted(comp_dets.get(r, set()))
            if not Tset or not Dset:
                continue

            t_index = {ti: k for k, ti in enumerate(Tset)}
            d_index = {dj: k for k, dj in enumerate(Dset)}

            nt = len(Tset)
            nd = len(Dset)
            cost = np.full((nt, nd), big, dtype=np.float64)

            for ti, dj, iou, dist in e_list:
                ii = t_index[ti]
                jj = d_index[dj]
                c = (1.0 - float(iou)) * W + float(dist)
                # keep best edge if multiple
                if c < cost[ii, jj]:
                    cost[ii, jj] = c

            if linear_sum_assignment is None:
                # Fallback: greedy on cost (still component-local)
                flat: List[Tuple[float, int, int]] = []
                for ii in range(nt):
                    for jj in range(nd):
                        if cost[ii, jj] < big * 0.5:
                            flat.append((float(cost[ii, jj]), ii, jj))
                flat.sort(key=lambda x: x[0])
                used_i: Set[int] = set()
                used_j: Set[int] = set()
                for c, ii, jj in flat:
                    if ii in used_i or jj in used_j:
                        continue
                    used_i.add(ii)
                    used_j.add(jj)
                    matches.append((Tset[ii], Dset[jj]))
                continue

            row_ind, col_ind = linear_sum_assignment(cost)
            for ii, jj in zip(row_ind.tolist(), col_ind.tolist()):
                if cost[ii, jj] >= big * 0.5:
                    continue
                matches.append((Tset[ii], Dset[jj]))

        return matches

    # -----------------------------
    # Public stepping API
    # -----------------------------

    def _step_impl(self, frame_id: str, detections: FrameData, timestamp: Optional[float]) -> FrameData:
        return self.step_with_gt(frame_id, detections.dets, gt_dets=None, timestamp=timestamp)

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
        self._last_frame_id = str(frame_id)

        # ---- GT map ----
        gt_now = self._build_gt_maps(gt_dets)

        # Ensure GT tracks exist for currently known GT ids
        for gid, gbox in gt_now.items():
            if gid not in self._gt_tracks:
                tid0 = self._gt_tid(gid, 0)
                tr = _TrackState(
                    tid=int(tid0),
                    is_gt=True,
                    gt_id=int(gid),
                    epoch=0,
                    expired=False,
                    out_box=gbox,
                    obs_box=gbox,
                    evidence=0.0,
                    confirmed=False,
                    hits=0,
                    last_seen_t=t_now,   # prevents immediate "expiry before ever matched"
                    last_emit_t=0.0,
                    last_pred_t=t_now,
                )
                self._gt_tracks[gid] = tr

        # ---- Predict (advance out_box) for all tracks ----
        # GT tracks: use GT displacement when available
        for gid, tr in self._gt_tracks.items():
            # Advance only once per frame
            if tr.last_pred_t < t_now - 1e-12:
                self._predict_track(tr, dt_frame, gt_now=gt_now)
                tr.last_pred_t = t_now
                self._gt_tracks[gid] = tr

        # FP tracks: CV/hold
        for tid, tr in list(self._fp_tracks.items()):
            if tr.last_pred_t < t_now - 1e-12:
                self._predict_track(tr, dt_frame, gt_now=None)
                tr.last_pred_t = t_now
                self._fp_tracks[tid] = tr

        # ---- Prepare detections arrays ----
        det_list = list(detections)
        det_indices: List[int] = []
        det_boxes: List[Box3D] = []
        det_scores: List[Optional[float]] = []

        for i, d in enumerate(det_list):
            if d.box is None:
                continue
            det_indices.append(i)
            det_boxes.append(d.box)
            det_scores.append(d.score)

        det_xy = np.array([[b.cx, b.cy] for b in det_boxes], dtype=np.float64) if det_boxes else np.zeros((0, 2), dtype=np.float64)
        det_corners, det_areas = _precompute_bev_rects(det_boxes)

        # ---- 1) Assign detections to GT tracks ----
        gt_ids = list(self._gt_tracks.keys())
        gt_boxes = [self._gt_tracks[gid].out_box for gid in gt_ids]

        gt_xy = np.array([[b.cx, b.cy] for b in gt_boxes], dtype=np.float64) if gt_boxes else np.zeros((0, 2), dtype=np.float64)
        gt_corners, gt_areas = _precompute_bev_rects(gt_boxes)

        gt_edges = self._build_candidates(
            track_boxes=gt_boxes,
            det_boxes=det_boxes,
            track_xy=gt_xy,
            det_xy=det_xy,
            track_corners=gt_corners,
            track_areas=gt_areas,
            det_corners=det_corners,
            det_areas=det_areas,
        )
        gt_matches_local = self._assign_component_hungarian(len(gt_boxes), len(det_boxes), gt_edges)

        # Translate to (gt_id, det_global_index)
        gt_matches: Dict[int, int] = {}  # gt_id -> det_idx_in_det_list
        used_det_local: Set[int] = set()
        for ti, dj in gt_matches_local:
            gid = int(gt_ids[ti])
            det_global = int(det_indices[dj])
            gt_matches[gid] = det_global
            used_det_local.add(dj)

        # ---- Update GT tracks evidence + observation ----
        gt_matched_ids: Set[int] = set(gt_matches.keys())
        for gid in gt_ids:
            tr = self._gt_tracks[gid]
            if gid in gt_matches:
                di = gt_matches[gid]
                det = det_list[di]
                if det.box is None:
                    continue

                # If track was expired (miss_dt > T_reid), start a new epoch now (new output id, fresh evidence)
                is_static = self._is_static(tr)
                T_reid = float(cfg.T_reid_static_s if is_static else cfg.T_reid_base_s)
                miss_dt = float(t_now - tr.last_seen_t)
                if miss_dt > T_reid:
                    tr.epoch += 1
                    tr.tid = self._gt_tid(int(gid), int(tr.epoch))
                    tr.expired = False
                    tr.reset_epoch_state()
                else:
                    tr.expired = False

                # Evidence + motion update
                dt_obs = max(1e-6, t_now - tr.last_seen_t)
                self._evidence_on_match(tr, det.score, dt_frame)

                tr.obs_box = det.box
                tr.out_box = det.box
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box, dt_obs)

                self._gt_tracks[gid] = tr
            else:
                # miss: evidence decay (do not delete GT track)
                self._evidence_on_miss(tr, dt_frame)
                self._gt_tracks[gid] = tr

        # ---- 2) FP association on remaining detections ----
        remaining_det_indices: List[int] = []
        remaining_det_boxes: List[Box3D] = []
        remaining_det_scores: List[Optional[float]] = []
        remaining_det_map: List[int] = []  # local remaining -> global det_list index

        for local_j, global_i in enumerate(det_indices):
            if local_j in used_det_local:
                continue
            d = det_list[global_i]
            if d.box is None:
                continue
            remaining_det_map.append(global_i)
            remaining_det_indices.append(local_j)  # index into det_boxes
            remaining_det_boxes.append(d.box)
            remaining_det_scores.append(d.score)

        # Build remaining arrays
        rem_xy = np.array([[b.cx, b.cy] for b in remaining_det_boxes], dtype=np.float64) if remaining_det_boxes else np.zeros((0, 2), dtype=np.float64)
        rem_corners, rem_areas = _precompute_bev_rects(remaining_det_boxes)

        fp_tids = list(self._fp_tracks.keys())
        fp_boxes = [self._fp_tracks[tid].out_box for tid in fp_tids]
        fp_xy = np.array([[b.cx, b.cy] for b in fp_boxes], dtype=np.float64) if fp_boxes else np.zeros((0, 2), dtype=np.float64)
        fp_corners, fp_areas = _precompute_bev_rects(fp_boxes)

        fp_edges = self._build_candidates(
            track_boxes=fp_boxes,
            det_boxes=remaining_det_boxes,
            track_xy=fp_xy,
            det_xy=rem_xy,
            track_corners=fp_corners,
            track_areas=fp_areas,
            det_corners=rem_corners,
            det_areas=rem_areas,
        )
        fp_matches_local = self._assign_component_hungarian(len(fp_boxes), len(remaining_det_boxes), fp_edges)

        fp_matched_tids: Set[int] = set()
        fp_used_det_locals: Set[int] = set()

        for ti, dj in fp_matches_local:
            tid = int(fp_tids[ti])
            tr = self._fp_tracks[tid]
            det_global_i = int(remaining_det_map[dj])
            det = det_list[det_global_i]
            if det.box is None:
                continue

            dt_obs = max(1e-6, t_now - tr.last_seen_t)
            self._evidence_on_match(tr, det.score, dt_frame)

            tr.obs_box = det.box
            tr.out_box = det.box
            tr.last_seen_t = t_now
            tr.push_observation(cfg, det.box, dt_obs)

            self._fp_tracks[tid] = tr
            fp_matched_tids.add(tid)
            fp_used_det_locals.add(dj)

        # FP misses: decay + delete after T_reid
        fp_to_delete: List[int] = []
        for tid in fp_tids:
            if tid in fp_matched_tids:
                continue
            tr = self._fp_tracks[tid]
            self._evidence_on_miss(tr, dt_frame)

            is_static = self._is_static(tr)
            T_reid = float(cfg.T_reid_static_s if is_static else cfg.T_reid_base_s)
            miss_dt = float(t_now - tr.last_seen_t)
            if miss_dt > T_reid:
                fp_to_delete.append(tid)
            else:
                self._fp_tracks[tid] = tr

        for tid in fp_to_delete:
            self._fp_tracks.pop(tid, None)

        # ---- Spawn new FP tracks from remaining unmatched detections ----
        for dj, det_global_i in enumerate(remaining_det_map):
            if dj in fp_used_det_locals:
                continue
            det = det_list[det_global_i]
            if det.box is None:
                continue

            tid = int(self._fp_next_tid)
            self._fp_next_tid += 1

            tr = _TrackState(
                tid=tid,
                is_gt=False,
                gt_id=None,
                epoch=0,
                expired=False,
                out_box=det.box,
                obs_box=det.box,
                evidence=0.0,
                confirmed=False,
                hits=0,
                last_seen_t=t_now,
                last_emit_t=0.0,
                last_pred_t=t_now,
            )
            # First observation updates evidence/hits, but min_hits prevents same-frame emission.
            self._evidence_on_match(tr, det.score, dt_frame)
            tr.push_observation(cfg, det.box, dt=max(1e-6, dt_frame))
            self._fp_tracks[tid] = tr

        # ---- Build output detections (GT + FP) ----
        out_dets: List[Detection] = []

        # GT tracks: never deleted; epoch increments are handled on match after long miss
        for gid, tr in self._gt_tracks.items():
            miss_dt = float(t_now - tr.last_seen_t)
            T_out = self._T_out_from_evidence(tr.evidence)
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
                tr.last_emit_t = t_now
                self._gt_tracks[gid] = tr

        # FP tracks: emit only if confirmed and within output coasting window
        for tid, tr in self._fp_tracks.items():
            miss_dt = float(t_now - tr.last_seen_t)
            T_out = self._T_out_from_evidence(tr.evidence)
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
                tr.last_emit_t = t_now
                self._fp_tracks[tid] = tr

        # ---- Update GT cache for displacement ----
        self._prev_gt_by_id = dict(gt_now)

        return FrameData(frame_id=str(frame_id), dets=out_dets)
