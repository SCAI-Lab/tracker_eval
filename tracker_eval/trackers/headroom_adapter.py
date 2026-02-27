from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Set

import math
import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig

from tracker_eval.utils import (
    linear_sum_assignment,
    cKDTree,
    _precompute_bev_rects,
    bev_iou_oriented_cached,
    _UnionFind,
    _build_candidates,
    _assign_component_hungarian,
    _gate_pair_distance_only,
)


# -----------------------------
# Config
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def _gain_from_tau(dt: float, tau: float) -> float:
    tau = max(1e-6, float(tau))
    dt = max(1e-6, float(dt))
    return float(1.0 - math.exp(-dt / tau))

def _norm2(xy: Tuple[float, float]) -> float:
    return float(math.sqrt(xy[0]*xy[0] + xy[1]*xy[1]))

def _dot(a, b) -> float:
    return float(a[0]*b[0] + a[1]*b[1])

def _sub(a, b):
    return (float(a[0]-b[0]), float(a[1]-b[1]))

def _add(a, b):
    return (float(a[0]+b[0]), float(a[1]+b[1]))

def _mul(a, s: float):
    return (float(a[0]*s), float(a[1]*s))

def _unit(a):
    n = _norm2(a)
    if n < 1e-6:
        return (1.0, 0.0)
    return (a[0]/n, a[1]/n)


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
    dist_gate_m: float = 0.4
    z_gate_m: float = 1.0  # set <=0 to disable

    # Candidate pruning (speed)
    assoc_topk: int = 10  # max candidates per track (after radius query)

    # Assignment cost: cost = (1 - iou) * assoc_iou_weight + dist
    assoc_iou_weight: float = 1.0
    forbidden_cost: float = 1e6  # used for non-edges in Hungarian

    # Evidence model (smoothed score + miss decay + hysteresis)
    score_floor: float = 0.5     # scores <= floor contribute ~0 evidence
    score_power: float = 1.5      # x = s_norm^p
    tau_hit_s: float = 0.1       # evidence rise time constant (seconds)
    tau_miss_s: float = 2.0      # evidence decay time constant (seconds)
    theta_on: float = 0.50        # confirm threshold
    min_hits: int = 2             # min matched detections before first confirmation

    # Output coasting after miss (seconds), based on CURRENT evidence (not peak)
    T_out_min_s: float = 0.30
    T_out_max_s: float = 1.0
    T_out_gamma: float = 1.0

    # ReID / forgetting (seconds), independent of evidence
    T_reid_base_s: float = 1.0
    T_reid_static_s: float = 2.0

    # Static inference from observed history only (detections only)
    static_window: int = 15
    v_static_thr_mps: float = 0.20
    jitter_thr_m: float = 0.15
    vel_ema_beta: float = 0.8

    # Staleness penalty (apply to ALL tracks in assignment cost)
    stale_lambda_m_per_s: float = 0.20   # your 0.2m/1s
    stale_cap_s: float = 1.5            # clamp miss_dt to this

    # --- anisotropic smoothing params ---
    tau_par_move_s: float = 0.25     # 0.2–0.3s desired
    tau_perp_move_s: float = 0.70    # suppress perpendicular jitter more
    tau_static_s: float = 0.80       # strong smoothing when static
    tau_vel_s: float = 0.30          # velocity smoothing

    v_enter_mps: float = 0.10        # easy to enter moving
    v_exit_mps: float = 0.05         # harder to exit moving
    enter_frames: int = 3            # ~0.2s at 15 Hz
    exit_frames: int = 8             # ~0.53s at 15 Hz

    dir_consistency_cos: float = 0.5 # require some direction consistency to enter moving


    # Prediction when GT displacement is not available (and for FP tracks)
    use_const_vel_coast: bool = True

    # Output score for exported tracks
    output_score: float = 1.0

    # ID namespaces
    gt_stride: int = 100_000
    fp_offset: int = 10_000_000


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

    # Filtered kinematics (2D)
    filt_xy: Tuple[float, float] = (0.0, 0.0)
    filt_vxy: Tuple[float, float] = (0.0, 0.0)
    prev_filt_xy: Tuple[float, float] = (0.0, 0.0)

    # motion mode hysteresis
    moving: bool = False
    move_streak: int = 0
    still_streak: int = 0

    # for direction consistency checks
    last_v_meas: Tuple[float, float] = (0.0, 0.0)


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
         - confirmation with hysteresis (theta_on) and min_hits
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
    
    def _anisotropic_correct(self, tr: _TrackState, det_box: Box3D, dt: float) -> None:
        cfg = self.cfg
        dt = float(max(1e-6, dt))

        # predicted position is current out_box center (you already predicted earlier)
        p_pred = (float(tr.out_box.cx), float(tr.out_box.cy))
        z = (float(det_box.cx), float(det_box.cy))
        r = _sub(z, p_pred)

        # --- quick motion estimate from measurements ---
        v_meas = _mul(_sub(z, tr.prev_filt_xy), 1.0/dt)
        v_meas_norm = _norm2(v_meas)

        # direction consistency to avoid jitter triggering "moving"
        last_vm = tr.last_v_meas
        cos_sim = 0.0
        if _norm2(last_vm) > 1e-3 and v_meas_norm > 1e-3:
            cos_sim = _dot(_unit(last_vm), _unit(v_meas))

        tr.last_v_meas = v_meas

        # --- hysteresis: enter moving quickly, exit slowly ---
        good_dir = (cos_sim >= cfg.dir_consistency_cos) or (_norm2(last_vm) <= 1e-3)
        if (v_meas_norm >= cfg.v_enter_mps) and good_dir:
            tr.move_streak += 1
            tr.still_streak = 0
        elif v_meas_norm <= cfg.v_exit_mps:
            tr.still_streak += 1
            tr.move_streak = 0
        else:
            # ambiguous zone: don't flip aggressively
            tr.move_streak = max(0, tr.move_streak - 1)
            tr.still_streak = max(0, tr.still_streak - 1)

        if (not tr.moving) and (tr.move_streak >= cfg.enter_frames):
            tr.moving = True
        if tr.moving and (tr.still_streak >= cfg.exit_frames):
            tr.moving = False

        # --- choose gains ---
        if not tr.moving:
            # isotropic strong smoothing when static
            k = _gain_from_tau(dt, cfg.tau_static_s)
            p_new = _add(p_pred, _mul(r, k))
        else:
            # anisotropic smoothing when moving
            v_hat = tr.filt_vxy
            if _norm2(v_hat) < 1e-3:
                v_hat = v_meas

            u = _unit(v_hat)
            r_par = _mul(u, _dot(r, u))
            r_perp = _sub(r, r_par)

            k_par = _gain_from_tau(dt, cfg.tau_par_move_s)
            k_perp = _gain_from_tau(dt, cfg.tau_perp_move_s)

            p_new = _add(p_pred, _add(_mul(r_par, k_par), _mul(r_perp, k_perp)))

        # --- update filtered velocity (EMA on velocity) ---
        v_new_meas = _mul(_sub(p_new, tr.prev_filt_xy), 1.0/dt)
        a_v = _gain_from_tau(dt, cfg.tau_vel_s)
        tr.filt_vxy = _add(_mul(tr.filt_vxy, (1.0 - a_v)), _mul(v_new_meas, a_v))

        tr.prev_filt_xy = p_new
        tr.filt_xy = p_new

        # write filtered center into out_box (this affects next frame association too)
        tr.out_box = Box3D(
            cx=float(p_new[0]),
            cy=float(p_new[1]),
            cz=float(det_box.cz),
            l=float(det_box.l),
            w=float(det_box.w),
            h=float(det_box.h),
            rot_z=float(det_box.rot_z),
        )
        tr.obs_box = det_box  # keep raw observation if you still want it


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
                xy0 = (float(gbox.cx), float(gbox.cy))
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
                    filt_xy=xy0, prev_filt_xy=xy0, filt_vxy=(0.0, 0.0),
                )
                self._gt_tracks[gid] = tr

        # ---- Predict (advance out_box) for all tracks ----
        # GT tracks: use GT displacement when available
        for gid, tr in self._gt_tracks.items():
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

        det_xy = (
            np.array([[b.cx, b.cy] for b in det_boxes], dtype=np.float64)
            if det_boxes else np.zeros((0, 2), dtype=np.float64)
        )
        det_corners, det_areas = _precompute_bev_rects(det_boxes)

        # ============================================================
        # 1) Assign detections to GT tracks (unchanged)
        # ============================================================
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

        gt_matches_local = _assign_component_hungarian(
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
                    tr.expired = False
                    tr.reset_epoch_state()
                    xy = (float(det.box.cx), float(det.box.cy))
                    tr.filt_xy = xy
                    tr.prev_filt_xy = xy
                    tr.filt_vxy = (0.0, 0.0)
                    tr.moving = False
                    tr.move_streak = 0
                    tr.still_streak = 0
                    tr.last_v_meas = (0.0, 0.0)
                else:
                    tr.expired = False

                dt_obs = max(1e-6, t_now - tr.last_seen_t)
                self._evidence_on_match(tr, det.score, dt_frame)

                # tr.obs_box = det.box
                # tr.out_box = det.box
                self._anisotropic_correct(tr, det.box, dt_obs)
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box, dt_obs)

                self._gt_tracks[gid] = tr
            else:
                self._evidence_on_miss(tr, dt_frame)
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

        rem_xy = (
            np.array([[b.cx, b.cy] for b in remaining_det_boxes], dtype=np.float64)
            if remaining_det_boxes else np.zeros((0, 2), dtype=np.float64)
        )
        rem_corners, rem_areas = _precompute_bev_rects(remaining_det_boxes)

        # Split FP tracks by confirmed state
        # fp_confirmed_tids: List[int] = []
        # fp_tentative_tids: List[int] = []
        # for tid, tr in self._fp_tracks.items():
        #     if tr.confirmed:
        #         fp_confirmed_tids.append(int(tid))
        #     else:
        #         fp_tentative_tids.append(int(tid))

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

            fp_boxes_subset = [self._fp_tracks[tid].out_box for tid in fp_tids_subset]
            fp_miss_dt = np.array([t_now - self._fp_tracks[tid].last_seen_t for tid in fp_tids_subset], dtype=np.float64)
            fp_xy = (
                np.array([[b.cx, b.cy] for b in fp_boxes_subset], dtype=np.float64)
                if fp_boxes_subset else np.zeros((0, 2), dtype=np.float64)
            )
            fp_corners, fp_areas = _precompute_bev_rects(fp_boxes_subset)

            edges = _build_candidates(
                cfg,
                fp_boxes_subset,
                avail_boxes,
                gt_xy=fp_xy,
                det_xy=avail_xy,
                gt_corners=fp_corners,
                gt_areas=fp_areas,
                det_corners=avail_corners,
                det_areas=avail_areas,
                min_iou=0.0,
            )
            matches_local = _assign_component_hungarian(
                cfg, len(fp_boxes_subset), len(avail_boxes), edges,
                track_miss_dt=fp_miss_dt
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

                dt_obs = max(1e-6, t_now - tr.last_seen_t)
                self._evidence_on_match(tr, det.score, dt_frame)

                # tr.obs_box = det.box
                # tr.out_box = det.box
                self._anisotropic_correct(tr, det.box, dt_obs)
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box, dt_obs)

                self._fp_tracks[tid] = tr
                fp_matched_tids.add(tid)
                fp_used_det_locals.add(det_local)

        # # Pass 1: confirmed FP tracks get first shot
        # _match_fp_subset(fp_confirmed_tids)

        # # Pass 2: tentative FP tracks can use remaining detections
        # _match_fp_subset(fp_tentative_tids)

        fp_all_tids = [int(tid) for tid in self._fp_tracks.keys()]
        _match_fp_subset(fp_all_tids)


        # ---- FP misses: decay + delete after T_reid (unchanged logic, but uses fp_matched_tids) ----
        fp_to_delete: List[int] = []
        for tid, tr in list(self._fp_tracks.items()):
            tid_i = int(tid)
            if tid_i in fp_matched_tids:
                continue

            self._evidence_on_miss(tr, dt_frame)

            is_static = self._is_static(tr)
            T_reid = float(cfg.T_reid_static_s if is_static else cfg.T_reid_base_s)
            miss_dt = float(t_now - tr.last_seen_t)
            if miss_dt > T_reid:
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
                expired=False,
                out_box=det.box,
                obs_box=det.box,
                evidence=0.0,
                confirmed=False,
                hits=0,
                last_seen_t=t_now,
                last_emit_t=0.0,
                last_pred_t=t_now,
                filt_xy=xy0, prev_filt_xy=xy0, filt_vxy=(0.0, 0.0),
            )
            self._evidence_on_match(tr, det.score, dt_frame)
            tr.push_observation(cfg, det.box, dt=max(1e-6, dt_frame))
            self._fp_tracks[tid] = tr

        # ---- Build output detections (GT + FP) ----
        out_dets: List[Detection] = []

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
                self._fp_tracks[int(tid)] = tr

        # ---- Update GT cache for displacement ----
        self._prev_gt_by_id = dict(gt_now)

        return FrameData(frame_id=str(frame_id), dets=out_dets)