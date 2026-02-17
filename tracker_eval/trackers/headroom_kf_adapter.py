# tracker_eval/trackers/headroom_tracker.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set

import math
import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.trackers.base import TrackerBase, TrackerInfo, TrackerRunConfig

from tracker_eval.utils import (
    _precompute_bev_rects,
    cKDTree,
    bev_iou_oriented_cached,
    _UnionFind, 
)

from scipy.optimize import linear_sum_assignment as _lsa

def _maybe_build_kdtree(det_xy: np.ndarray):
    return cKDTree(det_xy) if (cKDTree is not None and det_xy.shape[0] > 0) else None

def _chi2_gate_d2_2d(prob: float) -> float:
    # 2 DoF chi-square quantiles (approx)
    # 0.95 -> 5.991, 0.99 -> 9.210, 0.995 -> 10.597, 0.9973 -> 11.829
    if prob >= 0.997:
        return 11.829
    if prob >= 0.995:
        return 10.597
    if prob >= 0.99:
        return 9.210
    if prob >= 0.95:
        return 5.991
    return 5.991


def _get_maha_gate_d2(cfg) -> float:
    # Prefer explicit maha_gate_d2; else maha_gate_prob; else default 0.99
    if hasattr(cfg, "maha_gate_d2"):
        try:
            return float(getattr(cfg, "maha_gate_d2"))
        except Exception:
            pass
    if hasattr(cfg, "maha_gate_prob"):
        try:
            return float(_chi2_gate_d2_2d(float(getattr(cfg, "maha_gate_prob"))))
        except Exception:
            pass
    return float(_chi2_gate_d2_2d(0.99))


def _z_gate_ok(cfg, bgt: Box3D, bdet: Box3D) -> bool:
    zg = float(getattr(cfg, "z_gate_m", 0.0))
    if zg <= 0.0:
        return True
    return abs(float(bgt.cz) - float(bdet.cz)) <= zg


def _sym2(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def _S_pos_from_track_and_R(track_state, R: np.ndarray) -> np.ndarray:
    # H selects px,py => HPH^T is P[:2,:2]
    P = track_state.imm.P  # fused 6x6
    return _sym2(P[:2, :2] + R)


def _maha_d2(y: np.ndarray, S: np.ndarray) -> float:
    try:
        sol = np.linalg.solve(S, y)
        return float(y.T @ sol)
    except np.linalg.LinAlgError:
        Sinv = np.linalg.pinv(S)
        return float(y.T @ Sinv @ y)


def _maha_preprune_radius(S: np.ndarray, gate_d2: float) -> float:
    """
    Safe Euclidean radius for KD-tree pre-pruning:
      if y^T S^{-1} y <= gate_d2 then ||y||^2 <= gate_d2 * lambda_max(S)
      => r = sqrt(gate_d2 * lambda_max(S))
    """
    try:
        lam_max = float(np.max(np.linalg.eigvalsh(S)))
        lam_max = max(lam_max, 1e-12)
    except Exception:
        lam_max = max(0.5 * float(np.trace(S)), 1e-12)
    return float(math.sqrt(max(0.0, gate_d2 * lam_max)))


def _euclid_dist_xy(bgt: Box3D, bdet: Box3D) -> float:
    dx = float(bdet.cx) - float(bgt.cx)
    dy = float(bdet.cy) - float(bgt.cy)
    return float(math.sqrt(dx * dx + dy * dy))


def _build_candidates_maha_kf(
    cfg,
    *,
    track_states: List,      # aligned with gt_boxes
    gt_boxes: List[Box3D],
    det_boxes: List[Box3D],
    gt_xy: np.ndarray,
    det_xy: np.ndarray,
    gt_corners: np.ndarray,
    gt_areas: np.ndarray,
    det_corners: np.ndarray,
    det_areas: np.ndarray,
    min_iou: Optional[float] = None,
    R_fn=None,               # callable(track_state)->2x2, e.g. self._R_det
    det_tree: Optional[Any] = None,  # optional prebuilt cKDTree(det_xy)
) -> List[Tuple[int, int, float, float]]:
    """
    Build sparse candidate edges (gi, dj, iou, dist) using:
      - KD-tree pre-prune with radius derived from Mahalanobis gate
      - z-gate (optional)
      - Mahalanobis gate using S = Ppos + R(track)
      - top-k prune (by Euclidean distance)
      - IoU computation only when needed (weight>0 or min_iou>0)

    Returns edges in the SAME format as your utils version.
    """
    nG = len(gt_boxes)
    nD = len(det_boxes)
    if nG == 0 or nD == 0:
        return []
    if R_fn is None:
        raise ValueError("R_fn must be provided (e.g. self._R_det)")

    # backwards-compatible min_iou default
    if min_iou is None and hasattr(cfg, "tp_iou_thr"):
        try:
            min_iou = float(getattr(cfg, "tp_iou_thr"))
        except Exception:
            min_iou = None

    # treat <=0 as "disabled"
    if min_iou is not None and float(min_iou) <= 1e-12:
        min_iou_eff: Optional[float] = None
    else:
        min_iou_eff = float(min_iou) if min_iou is not None else None

    topk = int(max(1, int(getattr(cfg, "assoc_topk", 10))))
    gate_d2 = _get_maha_gate_d2(cfg)

    # Only compute IoU if it affects cost or pruning
    W = float(getattr(cfg, "assoc_iou_weight", 0.0))
    need_iou = (W > 1e-12) or (min_iou_eff is not None)

    dist_gate = float(getattr(cfg, "dist_gate_m", 0.0))
    use_dist_gate = dist_gate > 0.0

    # KD-tree over detections (optionally passed in)
    if det_tree is not None:
        tree = det_tree
    else:
        tree = cKDTree(det_xy) if cKDTree is not None else None

    edges: List[Tuple[int, int, float, float]] = []

    for gi in range(nG):
        bgt = gt_boxes[gi]
        tr = track_states[gi]

        # Build innovation covariance S for this track: S = Ppos + R
        R = R_fn(tr)
        S = _S_pos_from_track_and_R(tr, R)

        # Precompute S^{-1} once per track
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)

        # KD-tree pre-prune radius from Mahalanobis gate
        r_maha = _maha_preprune_radius(S, gate_d2)

        # Keep your current behavior: cap query radius to dist_gate (if provided)
        r_query = min(dist_gate, r_maha) if use_dist_gate else r_maha
        if r_query <= 1e-12:
            continue

        px = float(tr.imm.px)
        py = float(tr.imm.py)
        q = np.array([px, py], dtype=np.float64)

        if tree is not None:
            cand_js = tree.query_ball_point(q, r=r_query)
        else:
            dxv = det_xy[:, 0] - px
            dyv = det_xy[:, 1] - py
            d2e = dxv * dxv + dyv * dyv
            cand_js = np.where(d2e <= (r_query * r_query))[0].tolist()

        if not cand_js:
            continue

        if len(cand_js) > topk:
            idx = np.array(cand_js, dtype=np.int64)
            dxy = det_xy[idx] - q.reshape(1, 2)
            d2 = np.sum(dxy * dxy, axis=1)
            kth = max(0, topk - 1)
            keep = np.argpartition(d2, kth)[:topk]
            cand_js = idx[keep].tolist()

        # Full gates + (optional) IoU
        for dj in cand_js:
            bdet = det_boxes[dj]

            if not _z_gate_ok(cfg, bgt, bdet):
                continue

            # innovation y = z - Hx (H selects px,py)
            dx = float(bdet.cx) - px
            dy = float(bdet.cy) - py

            # Mahalanobis d^2 = [dx dy] * Sinv * [dx dy]^T  (scalar, no allocations)
            d2m = dx * (Sinv[0, 0] * dx + Sinv[0, 1] * dy) + dy * (Sinv[1, 0] * dx + Sinv[1, 1] * dy)
            if d2m > gate_d2:
                continue

            if need_iou:
                iou = bev_iou_oriented_cached(
                    gt_corners[gi], float(gt_areas[gi]),
                    det_corners[dj], float(det_areas[dj]),
                )
                if min_iou_eff is not None and float(iou) < float(min_iou_eff):
                    continue
                iou_f = float(iou)
            else:
                # IoU not used in cost and not used for pruning
                iou_f = 0.0

            dist = float(math.sqrt(dx * dx + dy * dy))
            edges.append((gi, int(dj), iou_f, dist))

    return edges



def _assign_component_hungarian_local(
    cfg: Any,
    nG: int,
    nD: int,
    edges: List[Tuple[int, int, float, float]],
    *,
    track_miss_dt: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Component-wise Hungarian (or greedy fallback).
    Uses cfg.assoc_iou_weight and cfg.forbidden_cost.

    Optimizations:
      - fast-path for 1xN or Nx1 components (no Hungarian)
      - optional staleness penalty per track-row:
          cost += stale_lambda_m_per_s * min(miss_dt, stale_cap_s)
    """
    if nG == 0 or nD == 0 or not edges:
        return []

    W = float(getattr(cfg, "assoc_iou_weight", 1.0))
    big = float(getattr(cfg, "forbidden_cost", 1e6))

    lam = float(getattr(cfg, "stale_lambda_m_per_s", 0.0))
    cap = float(getattr(cfg, "stale_cap_s", 0.0))
    use_stale = (track_miss_dt is not None) and (lam > 0.0) and (cap > 0.0)

    uf = _UnionFind(nG + nD)
    for gi, dj, _, _ in edges:
        uf.union(int(gi), nG + int(dj))

    comp_gt: Dict[int, Set[int]] = {}
    comp_det: Dict[int, Set[int]] = {}
    comp_edges: Dict[int, List[Tuple[int, int, float, float]]] = {}

    for gi, dj, iou, dist in edges:
        gi = int(gi)
        dj = int(dj)
        r = uf.find(gi)
        comp_gt.setdefault(r, set()).add(gi)
        comp_det.setdefault(r, set()).add(dj)
        comp_edges.setdefault(r, []).append((gi, dj, float(iou), float(dist)))

    matches: List[Tuple[int, int]] = []

    for r, e_list in comp_edges.items():
        Gset = sorted(comp_gt.get(r, set()))
        Dset = sorted(comp_det.get(r, set()))
        if not Gset or not Dset:
            continue

        g_index = {gi: k for k, gi in enumerate(Gset)}
        d_index = {dj: k for k, dj in enumerate(Dset)}

        ng = len(Gset)
        nd = len(Dset)
        cost = np.full((ng, nd), big, dtype=np.float64)

        for gi, dj, iou, dist in e_list:
            ii = g_index[gi]
            jj = d_index[dj]
            c = (1.0 - float(iou)) * W + float(dist)
            if c < cost[ii, jj]:
                cost[ii, jj] = c

        if use_stale:
            row_pen = np.zeros((ng,), dtype=np.float64)
            for ii, gi in enumerate(Gset):
                md = float(track_miss_dt[int(gi)])
                md = max(0.0, min(md, cap))
                row_pen[ii] = lam * md
            cost += row_pen[:, None]

        # ---- Fast paths for trivial components ----
        thresh = big * 0.5

        if ng == 1:
            jj = int(np.argmin(cost[0, :]))
            if float(cost[0, jj]) < thresh:
                matches.append((Gset[0], Dset[jj]))
            continue

        if nd == 1:
            ii = int(np.argmin(cost[:, 0]))
            if float(cost[ii, 0]) < thresh:
                matches.append((Gset[ii], Dset[0]))
            continue

        # ---- Hungarian (or greedy fallback) ----
        if _lsa is None:
            flat: List[Tuple[float, int, int]] = []
            for ii in range(ng):
                for jj in range(nd):
                    c = float(cost[ii, jj])
                    if c < thresh:
                        flat.append((c, ii, jj))
            flat.sort(key=lambda x: x[0])
            used_i: Set[int] = set()
            used_j: Set[int] = set()
            for c, ii, jj in flat:
                if ii in used_i or jj in used_j:
                    continue
                used_i.add(ii)
                used_j.add(jj)
                matches.append((Gset[ii], Dset[jj]))
            continue

        row_ind, col_ind = _lsa(cost)
        for ii, jj in zip(row_ind.tolist(), col_ind.tolist()):
            if float(cost[ii, jj]) >= thresh:
                continue
            matches.append((Gset[ii], Dset[jj]))

    return matches


# =============================================================================
# Small helpers
# =============================================================================

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def _norm2_xy(vx: float, vy: float) -> float:
    return float(math.sqrt(vx * vx + vy * vy))

def _unit_xy(vx: float, vy: float) -> Tuple[float, float]:
    n = _norm2_xy(vx, vy)
    if n < 1e-9:
        return (1.0, 0.0)
    return (float(vx / n), float(vy / n))

def _outer2(u: Tuple[float, float]) -> np.ndarray:
    return np.array([[u[0] * u[0], u[0] * u[1]],
                     [u[1] * u[0], u[1] * u[1]]], dtype=np.float64)

def _R_anisotropic_from_heading(
    sigma_par: float,
    sigma_perp: float,
    vx: float,
    vy: float,
    *,
    min_speed: float,
) -> np.ndarray:
    """
    Build 2x2 measurement covariance that is anisotropic in the direction of velocity.
      R = sigma_perp^2 * I + (sigma_par^2 - sigma_perp^2) * (u u^T)
    where u is unit velocity direction.
    If speed < min_speed, returns isotropic with sigma_perp (or sigma_par; caller decides).
    """
    s = _norm2_xy(vx, vy)
    if s < float(min_speed):
        sig = float(sigma_perp)
        return np.array([[sig * sig, 0.0], [0.0, sig * sig]], dtype=np.float64)

    ux, uy = _unit_xy(vx, vy)
    U = _outer2((ux, uy))
    I = np.eye(2, dtype=np.float64)
    sp2 = float(sigma_par) * float(sigma_par)
    st2 = float(sigma_perp) * float(sigma_perp)
    return st2 * I + (sp2 - st2) * U

def _update_box_center_keep_shape(b: Box3D, cx: float, cy: float, cz: Optional[float] = None) -> Box3D:
    if b is None:
        raise ValueError("Box3D is None")
    return Box3D(
        cx=float(cx),
        cy=float(cy),
        cz=float(b.cz if cz is None else cz),
        l=float(b.l),
        w=float(b.w),
        h=float(b.h),
        rot_z=float(b.rot_z),
    )

def _box_from_measurement(det_box: Box3D, cx: float, cy: float) -> Box3D:
    """Use det_box shape/yaw but overwrite center xy with filtered xy."""
    return Box3D(
        cx=float(cx),
        cy=float(cy),
        cz=float(det_box.cz),
        l=float(det_box.l),
        w=float(det_box.w),
        h=float(det_box.h),
        rot_z=float(det_box.rot_z),
    )

# =============================================================================
# IMM + KF core (2D, common 6D state: [px, py, vx, vy, ax, ay])
# =============================================================================

def _H_pos6() -> np.ndarray:
    # Measure only position [px, py]
    H = np.zeros((2, 6), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H

H_POS = _H_pos6()

def _F_static6(dt: float, vel_damp: float) -> np.ndarray:
    """
    Static-ish model:
      px' = px
      py' = py
      vx' = vel_damp * vx
      vy' = vel_damp * vy
      ax' = 0
      ay' = 0
    """
    F = np.zeros((6, 6), dtype=np.float64)
    F[0, 0] = 1.0
    F[1, 1] = 1.0
    F[2, 2] = float(vel_damp)
    F[3, 3] = float(vel_damp)
    # accel reset (0)
    return F

def _F_cv6(dt: float) -> np.ndarray:
    """
    CV-ish model with acceleration states forced to 0:
      px' = px + vx*dt
      py' = py + vy*dt
      vx' = vx
      vy' = vy
      ax' = 0
      ay' = 0
    """
    F = np.zeros((6, 6), dtype=np.float64)
    F[0, 0] = 1.0
    F[0, 2] = float(dt)
    F[1, 1] = 1.0
    F[1, 3] = float(dt)
    F[2, 2] = 1.0
    F[3, 3] = 1.0
    return F

def _F_ca6(dt: float) -> np.ndarray:
    """
    Constant acceleration model:
      p' = p + v*dt + 0.5*a*dt^2
      v' = v + a*dt
      a' = a
    """
    dt = float(dt)
    dt2 = dt * dt
    F = np.eye(6, dtype=np.float64)
    F[0, 2] = dt
    F[1, 3] = dt
    F[0, 4] = 0.5 * dt2
    F[1, 5] = 0.5 * dt2
    F[2, 4] = dt
    F[3, 5] = dt
    return F

def _Q_diag6(
    q_p: float,
    q_v: float,
    q_a: float,
    dt: float,
) -> np.ndarray:
    """
    Simple diagonal process noise scaled by dt:
      Q = diag([q_p^2, q_p^2, q_v^2, q_v^2, q_a^2, q_a^2]) * dt
    This is not the most physically "correct" discretization, but is stable,
    easy to tune, and works well when you mainly want smoothing + gating.
    """
    dt = float(max(1e-9, dt))
    qp2 = float(q_p) * float(q_p) * dt
    qv2 = float(q_v) * float(q_v) * dt
    qa2 = float(q_a) * float(q_a) * dt
    return np.diag([qp2, qp2, qv2, qv2, qa2, qa2]).astype(np.float64)

def _symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)

def _gaussian_log_likelihood(innov: np.ndarray, S: np.ndarray) -> float:
    """
    log N(innov; 0, S) up to constant:
      -0.5 * (log(det(S)) + innov^T S^{-1} innov)
    """
    try:
        # Cholesky for stability
        L = np.linalg.cholesky(S)
        # Solve L y = innov
        y = np.linalg.solve(L, innov)
        maha = float(np.dot(y, y))
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        return -0.5 * (logdet + maha)
    except np.linalg.LinAlgError:
        # Fallback
        Sinv = np.linalg.pinv(S)
        maha = float(innov.T @ Sinv @ innov)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            logdet = float(np.log(max(1e-12, np.abs(np.linalg.det(S)))))
        return -0.5 * (float(logdet) + maha)

class _KF6:
    """
    Minimal linear KF for 6D state with 2D position measurements.
    """
    def __init__(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x = x0.astype(np.float64).reshape(6,)
        self.P = P0.astype(np.float64).reshape(6, 6)

    def predict(self, F: np.ndarray, Q: np.ndarray) -> None:
        self.x = (F @ self.x).reshape(6,)
        self.P = _symmetrize(F @ self.P @ F.T + Q)

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        """
        Returns log-likelihood of the measurement under this model
        (used by IMM).
        """
        z = z.astype(np.float64).reshape(2,)
        R = R.astype(np.float64).reshape(2, 2)

        # innovation
        y = z - (H_POS @ self.x)
        S = H_POS @ self.P @ H_POS.T + R
        ll = _gaussian_log_likelihood(y, S)

        # Kalman gain
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)

        K = self.P @ H_POS.T @ Sinv  # (6x2)
        self.x = (self.x + K @ y).reshape(6,)
        I = np.eye(6, dtype=np.float64)
        self.P = _symmetrize((I - K @ H_POS) @ self.P)

        return float(ll)

    def shift_position(self, dx: float, dy: float, *, set_vel: bool = False, dt: float = 0.0) -> None:
        """
        Deterministic position shift (used for GT delta coasting).
        Does not change covariance (perfect control).
        Optionally sets velocity from delta/dt.
        """
        self.x[0] += float(dx)
        self.x[1] += float(dy)
        if set_vel and dt > 1e-9:
            self.x[2] = float(dx) / float(dt)
            self.x[3] = float(dy) / float(dt)
            # accel left as-is (or could be zeroed)

class _IMM6:
    """
    3-model IMM over shared 6D state:
      model 0: static-ish
      model 1: CV-ish
      model 2: CA
    """
    def __init__(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        Pi: np.ndarray,
        mu0: np.ndarray,
        *,
        static_vel_damp: float,
        q_static: Tuple[float, float, float],
        q_cv: Tuple[float, float, float],
        q_ca: Tuple[float, float, float],
    ) -> None:
        self.Pi = Pi.astype(np.float64).reshape(3, 3)
        self.mu = mu0.astype(np.float64).reshape(3,)
        self.mu = self.mu / max(1e-12, float(np.sum(self.mu)))

        self.static_vel_damp = float(static_vel_damp)
        self.q_static = tuple(float(x) for x in q_static)  # (q_p, q_v, q_a)
        self.q_cv = tuple(float(x) for x in q_cv)
        self.q_ca = tuple(float(x) for x in q_ca)

        self.models = [
            _KF6(x0.copy(), P0.copy()),
            _KF6(x0.copy(), P0.copy()),
            _KF6(x0.copy(), P0.copy()),
        ]

        # fused estimate cache
        self.x = x0.astype(np.float64).reshape(6,)
        self.P = P0.astype(np.float64).reshape(6, 6)

    def _mix(self) -> np.ndarray:
        """
        Mixing step. Returns c (normalizers) of shape (3,).
        """
        mu = self.mu
        Pi = self.Pi

        # c_j = sum_i mu_i * Pi_ij
        c = (mu.reshape(1, 3) @ Pi).reshape(3,)
        c = np.maximum(c, 1e-12)

        # mu_{i|j} = mu_i * Pi_ij / c_j
        mu_ij = (mu.reshape(3, 1) * Pi) / c.reshape(1, 3)  # (3,3)

        # mixed initial for each model j
        xs = [m.x.copy() for m in self.models]
        Ps = [m.P.copy() for m in self.models]

        for j in range(3):
            x0j = np.zeros((6,), dtype=np.float64)
            for i in range(3):
                x0j += mu_ij[i, j] * xs[i]

            P0j = np.zeros((6, 6), dtype=np.float64)
            for i in range(3):
                dx = (xs[i] - x0j).reshape(6, 1)
                P0j += mu_ij[i, j] * (Ps[i] + dx @ dx.T)

            self.models[j].x = x0j
            self.models[j].P = _symmetrize(P0j)

        return c

    def predict(self, dt: float) -> None:
        dt = float(max(1e-6, dt))
        self._mix()

        # predict each model
        # static
        F0 = _F_static6(dt, self.static_vel_damp)
        Q0 = _Q_diag6(*self.q_static, dt)
        self.models[0].predict(F0, Q0)

        # cv
        F1 = _F_cv6(dt)
        Q1 = _Q_diag6(*self.q_cv, dt)
        self.models[1].predict(F1, Q1)

        # ca
        F2 = _F_ca6(dt)
        Q2 = _Q_diag6(*self.q_ca, dt)
        self.models[2].predict(F2, Q2)

        self._fuse()

    def update(self, z_xy: Tuple[float, float], R: np.ndarray) -> None:
        z = np.array([float(z_xy[0]), float(z_xy[1])], dtype=np.float64)

        # likelihoods
        lls = np.zeros((3,), dtype=np.float64)
        for j in range(3):
            lls[j] = self.models[j].update(z, R)

        # Update model probabilities:
        # mu_j <- c_j * Lambda_j ; normalize
        # Here we approximate c_j using mu_prev * Pi (already baked by mixing step).
        # A standard IMM would store c from mixing, but using current mu*Pi is fine.
        mu_prev = self.mu
        c = (mu_prev.reshape(1, 3) @ self.Pi).reshape(3,)
        c = np.maximum(c, 1e-12)

        # convert log-likelihood to likelihood safely
        m = float(np.max(lls))
        Lambda = np.exp(lls - m)

        mu_new = c * Lambda
        s = float(np.sum(mu_new))
        if s <= 1e-12:
            mu_new = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            mu_new = mu_new / s

        self.mu = mu_new
        self._fuse()

    def _fuse(self) -> None:
        mu = self.mu
        xs = [m.x for m in self.models]
        Ps = [m.P for m in self.models]

        x = np.zeros((6,), dtype=np.float64)
        for j in range(3):
            x += mu[j] * xs[j]

        P = np.zeros((6, 6), dtype=np.float64)
        for j in range(3):
            dx = (xs[j] - x).reshape(6, 1)
            P += mu[j] * (Ps[j] + dx @ dx.T)

        self.x = x
        self.P = _symmetrize(P)

    def shift_position(self, dx: float, dy: float, *, set_vel: bool = False, dt: float = 0.0) -> None:
        for m in self.models:
            m.shift_position(dx, dy, set_vel=set_vel, dt=dt)
        self._fuse()

    @property
    def px(self) -> float:
        return float(self.x[0])

    @property
    def py(self) -> float:
        return float(self.x[1])

    @property
    def vx(self) -> float:
        return float(self.x[2])

    @property
    def vy(self) -> float:
        return float(self.x[3])

    @property
    def mu_static(self) -> float:
        return float(self.mu[0])

# =============================================================================
# Headroom KF/IMM config
# =============================================================================

@dataclass(frozen=True)
class HeadroomKFConfig:
    """
    KF/IMM version of headroom tracker.
    Defaults chosen to approximate the current behavior:
      - Strong smoothing, especially perpendicular to motion direction
      - Static vs moving handled by IMM (plus you can still use history-based static for reid)
      - GT coasting uses GT displacement (delta) as a perfect control shift (no extra uncertainty)
    """

    # Time base
    fps: float = 15.0

    # Association gating (same API as utils)
    dist_gate_m: float = 0.4
    z_gate_m: float = 1.0  # set <=0 to disable
    assoc_topk: int = 10
    assoc_iou_weight: float = 0.5
    forbidden_cost: float = 1e6

    # Evidence model (your current defaults)
    score_floor: float = 0.0
    score_power: float = 1.5
    tau_hit_s: float = 0.20
    tau_miss_s: float = 3.0
    theta_on: float = 0.50
    min_hits: int = 3

    # Output coasting after miss (seconds)
    T_out_min_s: float = 0.50
    T_out_max_s: float = 1.50
    T_out_gamma: float = 1.0

    # ReID / forgetting (seconds)
    T_reid_base_s: float = 2.5
    T_reid_static_s: float = 5.0

    # History-based static inference (kept to approximate your current reid behavior)
    static_window: int = 15
    v_static_thr_mps: float = 0.30
    jitter_thr_m: float = 0.20
    vel_ema_beta: float = 0.8

    # Staleness penalty (used inside assignment cost)
    stale_lambda_m_per_s: float = 0.20
    stale_cap_s: float = 1.5

    # -----------------------
    # IMM/KF motion parameters
    # -----------------------

    # Initial covariance (roughly: position std, velocity std, accel std)
    kf_init_pos_std_m: float = 0.25
    kf_init_vel_std_mps: float = 1.0
    kf_init_acc_std_mps2: float = 2.0

    # Static model velocity damping (0 => snap v->0 each predict; 0.3-0.7 => gentle)
    imm_static_vel_damp: float = 0.3

    # Process noise "std" scales (fed into diagonal Q, scaled by dt)
    # (q_p, q_v, q_a)
    imm_q_static: Tuple[float, float, float] = (0.02, 0.10, 0.10)
    imm_q_cv: Tuple[float, float, float] = (0.05, 0.40, 0.20)
    imm_q_ca: Tuple[float, float, float] = (0.05, 0.60, 0.80)

    # IMM transition matrix (rows i -> cols j)
    # Tuned for inertia: static tends to stay static; moving tends to stay moving.
    imm_Pi: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        (0.90, 0.10, 0.00),  # static -> (static, cv, ca)
        (0.08, 0.87, 0.05),  # cv     -> ...
        (0.02, 0.10, 0.88),  # ca     -> ...
    )

    # Initial model probabilities (static, cv, ca)
    imm_mu0: Tuple[float, float, float] = (0.60, 0.35, 0.05)

    # Detection measurement noise (anisotropic, rotated with velocity)
    kf_meas_sigma_par_m: float = 0.15
    kf_meas_sigma_perp_m: float = 0.35
    kf_meas_min_speed_mps: float = 0.15  # below this, use isotropic with sigma_perp

    # Static fallback measurement sigma (used when speed is very low and you want extra smoothing)
    kf_meas_sigma_static_m: float = 0.25

    maha_gate_d2: float = 9.21

    # -----------------------
    # GT coasting behavior
    # -----------------------
    # If True, GT-family prediction uses GT delta as a perfect control shift (like your current code)
    gt_use_delta_predict: bool = True

    # If True, also set velocity from GT delta/dt (default False to preserve "offset rides along GT")
    gt_set_vel_from_delta: bool = False

    # If True, when a GT track is missed, additionally do a strong KF update toward GT position.
    # NOTE: this will reduce any persistent offset between detections and GT (more "oracle-like").
    gt_strong_update_on_miss: bool = False

    # Strength of GT strong update (position std in meters)
    gt_strong_meas_sigma_m: float = 0.02

    # Output score for exported tracks
    output_score: float = 1.0

    # ID namespaces
    gt_stride: int = 100_000
    fp_offset: int = 10_000_000

# =============================================================================
# Track state
# =============================================================================

@dataclass
class _TrackState:
    tid: int
    is_gt: bool
    gt_id: Optional[int] = None
    epoch: int = 0

    # boxes
    out_box: Box3D = None  # predicted / last output box
    obs_box: Box3D = None  # last observed (matched detection) box

    # filter
    imm: _IMM6 = None

    # evidence
    evidence: float = 0.0
    confirmed: bool = False
    hits: int = 0

    # times
    last_seen_t: float = 0.0  # last time matched to a detection (not GT)
    last_emit_t: float = 0.0
    last_pred_t: float = 0.0

    # observed history (detections only) for your static inference
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

    def push_observation(self, cfg: HeadroomKFConfig, box: Box3D, dt: float) -> None:
        self.obs_centers.append((float(box.cx), float(box.cy)))
        if len(self.obs_centers) > int(cfg.static_window):
            self.obs_centers = self.obs_centers[-int(cfg.static_window):]

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

# =============================================================================
# Headroom KF tracker
# =============================================================================

class HeadroomTrackerKF(TrackerBase):
    """
    KF/IMM version of Headroom tracker.

    Keeps the same headroom philosophy:
      - GT-family: identity is GT-keyed + epoching; motion is "headroom" via GT delta when available
      - FP-family: standard tracks on leftover detections
      - association: sparse candidate graph + component Hungarian, cost = dist + (1-iou)*W (+ staleness inside utils)
      - evidence/headroom output gating unchanged

    Motion core changes:
      - replaces custom anisotropic smoother + EMA CV with an IMM (static/CV/CA) KF
      - detection updates use anisotropic (velocity-rotated) measurement covariances
      - optional strong GT update on miss (disabled by default to approximate your current behavior)
    """

    def __init__(
        self,
        *,
        cfg: Optional[HeadroomKFConfig] = None,
        run_cfg: Optional[TrackerRunConfig] = None,
        name: str = "headroom_kf",
        version: str = "3.0",
    ) -> None:
        self.cfg = cfg or HeadroomKFConfig()

        info = TrackerInfo(
            name=name,
            version=version,
            description="Headroom tracker (KF/IMM): GT epochs + FP tracks, sparse Hungarian association, score-evidence headroom gating, motion via IMM KF with rotated anisotropic R; GT delta coasting.",
            extra={k: getattr(self.cfg, k) for k in self.cfg.__dict__.keys()},
        )
        super().__init__(info, run_cfg=run_cfg)

        self._gt_tracks: Dict[int, _TrackState] = {}
        self._fp_tracks: Dict[int, _TrackState] = {}
        self._fp_next_tid: int = int(self.cfg.fp_offset)

        self._t: float = 0.0
        self._frame_idx: int = 0

        self._prev_gt_by_id: Dict[int, Box3D] = {}

    def _reset_sequence_impl(self, seq_name: str) -> None:
        self._gt_tracks = {}
        self._fp_tracks = {}
        self._fp_next_tid = int(self.cfg.fp_offset)

        self._t = 0.0
        self._frame_idx = 0
        self._prev_gt_by_id = {}

    # -------------------------------------------------------------------------
    # Evidence model
    # -------------------------------------------------------------------------

    def _score_to_x(self, score: Optional[float]) -> float:
        s = float(score) if score is not None else 1.0
        floor = float(self.cfg.score_floor)
        sN = _clamp((s - floor) / max(1e-9, (1.0 - floor)), 0.0, 1.0)
        return float(sN ** float(self.cfg.score_power))

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
        x = _clamp((float(E) - float(cfg.theta_on)) / max(1e-9, (1.0 - float(cfg.theta_on))), 0.0, 1.0)
        x = float(x ** float(cfg.T_out_gamma))
        return float(cfg.T_out_min_s + (cfg.T_out_max_s - cfg.T_out_min_s) * x)

    # -------------------------------------------------------------------------
    # Static inference (kept for reid behavior)
    # -------------------------------------------------------------------------

    def _is_static_hist(self, tr: _TrackState) -> bool:
        cfg = self.cfg
        if len(tr.obs_centers) < max(3, int(cfg.static_window) // 2):
            return False
        xs = np.array([p[0] for p in tr.obs_centers], dtype=np.float64)
        ys = np.array([p[1] for p in tr.obs_centers], dtype=np.float64)
        mx = float(xs.mean())
        my = float(ys.mean())
        rad = float(np.max(np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)))
        return (float(tr.v_ema) < float(cfg.v_static_thr_mps)) and (rad < float(cfg.jitter_thr_m))

    # -------------------------------------------------------------------------
    # GT helpers
    # -------------------------------------------------------------------------

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

    def _gt_delta_for_id(self, gt_id: int, gt_now: Dict[int, Box3D]) -> Optional[Tuple[float, float, float]]:
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

    # -------------------------------------------------------------------------
    # IMM/KF construction and measurement covariances
    # -------------------------------------------------------------------------

    def _make_imm(self, xy: Tuple[float, float]) -> _IMM6:
        cfg = self.cfg

        x0 = np.zeros((6,), dtype=np.float64)
        x0[0] = float(xy[0])
        x0[1] = float(xy[1])

        ppos = float(cfg.kf_init_pos_std_m) ** 2
        pvel = float(cfg.kf_init_vel_std_mps) ** 2
        pacc = float(cfg.kf_init_acc_std_mps2) ** 2
        P0 = np.diag([ppos, ppos, pvel, pvel, pacc, pacc]).astype(np.float64)

        Pi = np.array(cfg.imm_Pi, dtype=np.float64).reshape(3, 3)
        mu0 = np.array(cfg.imm_mu0, dtype=np.float64).reshape(3,)

        return _IMM6(
            x0=x0,
            P0=P0,
            Pi=Pi,
            mu0=mu0,
            static_vel_damp=float(cfg.imm_static_vel_damp),
            q_static=cfg.imm_q_static,
            q_cv=cfg.imm_q_cv,
            q_ca=cfg.imm_q_ca,
        )

    def _R_det(self, tr: _TrackState) -> np.ndarray:
        """
        Detection measurement covariance. Uses anisotropy rotated with the *predicted* velocity.
        If speed is low, use isotropic smoothing with kf_meas_sigma_static_m.
        """
        cfg = self.cfg
        vx, vy = tr.imm.vx, tr.imm.vy
        speed = _norm2_xy(vx, vy)

        if speed < float(cfg.kf_meas_min_speed_mps):
            sig = float(cfg.kf_meas_sigma_static_m)
            return np.array([[sig * sig, 0.0], [0.0, sig * sig]], dtype=np.float64)

        return _R_anisotropic_from_heading(
            sigma_par=float(cfg.kf_meas_sigma_par_m),
            sigma_perp=float(cfg.kf_meas_sigma_perp_m),
            vx=vx,
            vy=vy,
            min_speed=float(cfg.kf_meas_min_speed_mps),
        )

    def _R_gt_strong(self) -> np.ndarray:
        sig = float(self.cfg.gt_strong_meas_sigma_m)
        return np.array([[sig * sig, 0.0], [0.0, sig * sig]], dtype=np.float64)

    # -------------------------------------------------------------------------
    # Prediction / correction (KF/IMM)
    # -------------------------------------------------------------------------

    def _predict_track(self, tr: _TrackState, dt: float, gt_now: Optional[Dict[int, Box3D]] = None) -> None:
        """
        Advance track state by dt and write predicted center into out_box.
        GT tracks may use GT delta as a perfect control shift (like your original code).
        """
        cfg = self.cfg
        dt = float(max(1e-6, dt))

        if tr.out_box is None or tr.imm is None:
            return

        # Default: predict using IMM
        tr.imm.predict(dt)

        # Optional: for GT tracks, apply GT delta coasting as perfect control shift
        if tr.is_gt and cfg.gt_use_delta_predict and (gt_now is not None) and (tr.gt_id is not None):
            delta = self._gt_delta_for_id(int(tr.gt_id), gt_now)
            if delta is not None:
                dx, dy, dz = delta
                tr.imm.shift_position(dx, dy, set_vel=bool(cfg.gt_set_vel_from_delta), dt=dt)
                # also shift cz with GT delta, preserving your original coasting behavior on z
                tr.out_box = _update_box_center_keep_shape(tr.out_box, tr.imm.px, tr.imm.py, cz=float(tr.out_box.cz + dz))
                return

        # Normal predicted box update
        tr.out_box = _update_box_center_keep_shape(tr.out_box, tr.imm.px, tr.imm.py)

    def _correct_with_detection(self, tr: _TrackState, det_box: Box3D) -> None:
        """
        KF update with detection measurement, using rotated anisotropic R.
        Writes filtered center into out_box but keeps detection shape/yaw (like your original code).
        """
        if tr.imm is None:
            return

        R = self._R_det(tr)
        tr.imm.update((float(det_box.cx), float(det_box.cy)), R=R)
        tr.out_box = _box_from_measurement(det_box, tr.imm.px, tr.imm.py)
        tr.obs_box = det_box

    def _correct_with_gt_strong(self, tr: _TrackState, gt_box: Box3D) -> None:
        """
        Optional strong update toward GT position (only used on miss if enabled).
        NOTE: This reduces any persistent detection-vs-GT offset (more oracle-like).
        """
        if tr.imm is None:
            return
        R = self._R_gt_strong()
        tr.imm.update((float(gt_box.cx), float(gt_box.cy)), R=R)
        # Keep current shape/yaw, only shift center and cz to GT
        tr.out_box = _update_box_center_keep_shape(tr.out_box, tr.imm.px, tr.imm.py, cz=float(gt_box.cz))

    # -------------------------------------------------------------------------
    # Public step API
    # -------------------------------------------------------------------------

    def _step_impl(self, frame_id: str, detections: FrameData, timestamp: Optional[float]) -> FrameData:
        # By default, no GT is provided (same as your adapter).
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

        # ---- GT map ----
        gt_now = self._build_gt_maps(gt_dets)

        # Ensure GT tracks exist for currently known GT ids
        for gid, gbox in gt_now.items():
            if gid not in self._gt_tracks:
                tid0 = self._gt_tid(gid, 0)
                imm = self._make_imm((float(gbox.cx), float(gbox.cy)))
                tr = _TrackState(
                    tid=int(tid0),
                    is_gt=True,
                    gt_id=int(gid),
                    epoch=0,
                    out_box=gbox,
                    obs_box=gbox,
                    imm=imm,
                    evidence=0.0,
                    confirmed=False,
                    hits=0,
                    last_seen_t=t_now,  # prevents immediate expiry before ever matched
                    last_emit_t=0.0,
                    last_pred_t=t_now,
                )
                self._gt_tracks[gid] = tr

        # ---- Predict all tracks ----
        for gid, tr in self._gt_tracks.items():
            if tr.last_pred_t < t_now - 1e-12:
                self._predict_track(tr, dt_frame, gt_now=gt_now)
                tr.last_pred_t = t_now
                self._gt_tracks[gid] = tr

        for tid, tr in list(self._fp_tracks.items()):
            if tr.last_pred_t < t_now - 1e-12:
                self._predict_track(tr, dt_frame, gt_now=None)
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
        # 1) Assign detections to GT tracks
        # ============================================================
        gt_ids = list(self._gt_tracks.keys())
        gt_boxes = [self._gt_tracks[gid].out_box for gid in gt_ids]
        gt_miss_dt = np.array([t_now - self._gt_tracks[gid].last_seen_t for gid in gt_ids], dtype=np.float64)

        gt_xy = (
            np.array([[b.cx, b.cy] for b in gt_boxes], dtype=np.float64)
            if gt_boxes else np.zeros((0, 2), dtype=np.float64)
        )
        gt_corners, gt_areas = _precompute_bev_rects(gt_boxes)

        gt_track_states = [self._gt_tracks[gid] for gid in gt_ids]  # same order as gt_boxes

        gt_det_tree = _maybe_build_kdtree(det_xy)

        gt_edges = _build_candidates_maha_kf(
            cfg,
            track_states=gt_track_states,
            gt_boxes=gt_boxes,
            det_boxes=det_boxes,
            gt_xy=gt_xy,
            det_xy=det_xy,
            gt_corners=gt_corners,
            gt_areas=gt_areas,
            det_corners=det_corners,
            det_areas=det_areas,
            min_iou=0.0,
            R_fn=self._R_det,
            det_tree=gt_det_tree,   # <-- add this
        )

        gt_matches_local = _assign_component_hungarian_local(
            cfg, len(gt_boxes), len(det_boxes), gt_edges,
            track_miss_dt=gt_miss_dt,
        )

        gt_matches: Dict[int, int] = {}  # gt_id -> det_idx_in_det_list
        used_det_local: Set[int] = set()
        for ti, dj in gt_matches_local:
            gid = int(gt_ids[ti])
            det_global = int(det_indices[dj])
            gt_matches[gid] = det_global
            used_det_local.add(int(dj))

        # ---- Update GT tracks evidence + KF ----
        for gid in gt_ids:
            tr = self._gt_tracks[gid]

            if gid in gt_matches:
                di = gt_matches[gid]
                det = det_list[di]
                if det.box is None:
                    continue

                # epoching based on detection misses (as before)
                is_static = self._is_static_hist(tr)
                T_reid = float(cfg.T_reid_static_s if is_static else cfg.T_reid_base_s)
                miss_dt = float(t_now - tr.last_seen_t)
                if miss_dt > T_reid:
                    tr.epoch += 1
                    tr.tid = self._gt_tid(int(gid), int(tr.epoch))
                    tr.reset_epoch_state()

                    # reset IMM around the new observation
                    tr.imm = self._make_imm((float(det.box.cx), float(det.box.cy)))

                # evidence uses detector score only
                self._evidence_on_match(tr, det.score, dt_frame)

                # KF update with detection
                self._correct_with_detection(tr, det.box)

                # bookkeeping
                dt_obs = max(1e-6, t_now - tr.last_seen_t)
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box, dt_obs)

                self._gt_tracks[gid] = tr

            else:
                # miss: decay evidence
                self._evidence_on_miss(tr, dt_frame)

                # optional strong GT update on miss (off by default to preserve old behavior)
                if cfg.gt_strong_update_on_miss and (gid in gt_now):
                    self._correct_with_gt_strong(tr, gt_now[gid])

                self._gt_tracks[gid] = tr

        # ============================================================
        # 2) FP association on remaining detections
        # ============================================================
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

        fp_matched_tids: Set[int] = set()
        fp_used_det_locals: Set[int] = set()

        def _match_fp_subset(fp_tids_subset: List[int]) -> None:
            nonlocal fp_matched_tids, fp_used_det_locals
            if not fp_tids_subset or not remaining_det_boxes:
                return

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

            fp_states_subset = [self._fp_tracks[tid] for tid in fp_tids_subset]

            fp_det_tree = _maybe_build_kdtree(avail_xy)

            edges = _build_candidates_maha_kf(
                cfg,
                track_states=fp_states_subset,
                gt_boxes=fp_boxes_subset,
                det_boxes=avail_boxes,
                gt_xy=fp_xy,
                det_xy=avail_xy,
                gt_corners=fp_corners,
                gt_areas=fp_areas,
                det_corners=avail_corners,
                det_areas=avail_areas,
                min_iou=0.0,
                R_fn=self._R_det,
                det_tree=fp_det_tree,   # <-- add this
            )


            matches_local = _assign_component_hungarian_local(
                cfg, len(fp_boxes_subset), len(avail_boxes), edges,
                track_miss_dt=fp_miss_dt
            )

            for ti, dj in matches_local:
                tid = int(fp_tids_subset[ti])
                tr = self._fp_tracks[tid]

                det_local = int(avail_det_locals[dj])
                det_global_i = int(remaining_det_map[det_local])
                det = det_list[det_global_i]
                if det.box is None:
                    continue

                # evidence update
                self._evidence_on_match(tr, det.score, dt_frame)

                # KF update
                self._correct_with_detection(tr, det.box)

                # bookkeeping
                dt_obs = max(1e-6, t_now - tr.last_seen_t)
                tr.last_seen_t = t_now
                tr.push_observation(cfg, det.box, dt_obs)

                self._fp_tracks[tid] = tr
                fp_matched_tids.add(tid)
                fp_used_det_locals.add(det_local)

        # (kept as your current behavior: one pass over all FP)
        fp_all_tids = [int(tid) for tid in self._fp_tracks.keys()]
        _match_fp_subset(fp_all_tids)

        # ---- FP misses: decay + delete after T_reid ----
        fp_to_delete: List[int] = []
        for tid, tr in list(self._fp_tracks.items()):
            tid_i = int(tid)
            if tid_i in fp_matched_tids:
                continue

            self._evidence_on_miss(tr, dt_frame)

            is_static = self._is_static_hist(tr)
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

            imm = self._make_imm((float(det.box.cx), float(det.box.cy)))
            tr = _TrackState(
                tid=tid,
                is_gt=False,
                gt_id=None,
                epoch=0,
                out_box=det.box,
                obs_box=det.box,
                imm=imm,
                evidence=0.0,
                confirmed=False,
                hits=0,
                last_seen_t=t_now,
                last_emit_t=0.0,
                last_pred_t=t_now,
            )

            # immediate evidence and update
            self._evidence_on_match(tr, det.score, dt_frame)
            self._correct_with_detection(tr, det.box)
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

        # ---- Update GT cache for delta coasting ----
        self._prev_gt_by_id = dict(gt_now)

        return FrameData(frame_id=str(frame_id), dets=out_dets)
