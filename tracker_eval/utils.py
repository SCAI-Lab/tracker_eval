# tracker_eval/utils.py
from __future__ import annotations

import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from tracker_eval.common.types import Box3D

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


# ============================================================
# Deterministic utilities (shared)
# ============================================================

def _stable_u32(s: str) -> int:
    return zlib.adler32(s.encode("utf-8")) & 0xFFFFFFFF


def _seed_u32(base_seed: int, *parts: Any) -> int:
    h = int(base_seed) & 0xFFFFFFFF
    for p in parts:
        h = (h + _stable_u32(str(p))) & 0xFFFFFFFF
    return h


def _wrap_angle_rad_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _ceil_frames(seconds: float, fps: float) -> int:
    return max(1, int(math.ceil(float(seconds) * float(fps))))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _safe_pos(x: float, eps: float = 1e-3) -> float:
    return float(max(eps, float(x)))


def _trunc_normal(
    rng: np.random.Generator,
    mu: float,
    sigma: float,
    size: Tuple[int, ...] | int,
    n_sigma: float = 2.0,
    max_tries: int = 50,
    dtype=np.float32,
) -> np.ndarray:
    """
    Sample from Normal(mu, sigma) but truncate to [mu - n_sigma*sigma, mu + n_sigma*sigma]
    using rejection sampling. If it doesn't converge within max_tries, clip remaining values.
    """
    mu = float(mu)
    sigma = float(sigma)

    if sigma <= 0.0:
        return np.full(size, mu, dtype=dtype)

    cap = n_sigma * sigma
    lo, hi = mu - cap, mu + cap

    out = rng.normal(mu, sigma, size=size).astype(dtype, copy=False)
    mask = (out < lo) | (out > hi)

    tries = 0
    while mask.any() and tries < max_tries:
        out[mask] = rng.normal(mu, sigma, size=int(mask.sum())).astype(dtype, copy=False)
        mask = (out < lo) | (out > hi)
        tries += 1

    if mask.any():
        out = np.clip(out, lo, hi).astype(dtype, copy=False)

    return out


# ============================================================
# JSON / parsing helpers (shared)
# ============================================================

def _parse_frame_key(k: Any) -> str:
    return str(k)


def _parse_label_id(label_id: Any) -> Tuple[str, int]:
    """
    Forgiving version:
      - "pedestrian:18" -> ("pedestrian", 18)
      - if no ":" or parse fails -> (cls, -1)
    """
    s = str(label_id) if label_id is not None else ""
    if ":" not in s:
        return s.strip(), -1
    cls, tid = s.split(":", 1)
    try:
        return cls.strip(), int(tid)
    except Exception:
        return cls.strip(), -1


def _parse_label_id_strict(label_id: Any) -> Tuple[str, int]:
    """
    Strict version:
      - requires ":" and integer tid
    """
    s = str(label_id)
    if ":" not in s:
        raise ValueError(f"Unexpected label_id format (no ':'): {label_id}")
    cls, tid = s.split(":", 1)
    return cls.strip(), int(tid)


def _load_per_frame_dict(
    path: Path,
    *,
    container_keys: Sequence[str] = ("detections", "labels", "annotations", "frames", "data"),
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads a JRDB-ish JSON and returns:
      frame_key -> list[object]
    Supports:
      - {"detections": {...}} or {"labels": {...}} etc
      - direct dict {"000123.pcd": [ ... ], ... }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in container_keys:
            if key in data and isinstance(data[key], dict):
                v = data[key]
                if all(isinstance(x, list) for x in v.values()):
                    return v  # type: ignore[return-value]

        if all(isinstance(v, list) for v in data.values()):
            return data  # type: ignore[return-value]

    raise ValueError(f"Could not find per-frame dict in {path}")


def _load_frame_dict_any(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    # Backwards-compatible alias used by build_score_distributions_from_gt_det.py
    return _load_per_frame_dict(path)


def _load_labels_3d_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    # Backwards-compatible alias used by convert_gt_to_kitti_3d.py and generate_pseudo_detections_from_gt.py
    return _load_per_frame_dict(path, container_keys=("labels", "annotations", "frames", "data", "detections"))


def _box7_from_label_obj(obj: Dict[str, Any]) -> np.ndarray:
    """
    Accepts box in either dict or list form.
    Returns internal center box7: (cx, cy, cz, l, w, h, rot_z)
    """
    if "box" not in obj:
        raise ValueError("Missing 'box' in label object.")
    box = obj["box"]
    if isinstance(box, dict):
        needed = ["cx", "cy", "cz", "l", "w", "h", "rot_z"]
        for k in needed:
            if k not in box:
                raise ValueError(f"Missing '{k}' in box dict: keys={list(box.keys())}")
        cx = float(box["cx"])
        cy = float(box["cy"])
        cz = float(box["cz"])
        l = float(box["l"])
        w = float(box["w"])
        h = float(box["h"])
        rot_z = float(box["rot_z"])
        return np.array([cx, cy, cz, l, w, h, rot_z], dtype=np.float32)

    if isinstance(box, (list, tuple)):
        if len(box) != 7:
            raise ValueError(f"Box list must have length 7, got {len(box)}")
        return np.asarray(box, dtype=np.float32).reshape(7,)

    raise ValueError(f"Unsupported box type: {type(box)}")


def _box_from_obj(obj: Dict[str, Any]) -> Box3D:
    """
    Parse detection/label object into Box3D.
    Accepts obj["box"] as dict with keys cx,cy,cz,l,w,h,rot_z OR list/tuple length 7.
    """
    if "box" not in obj:
        raise ValueError("Missing 'box' field.")
    box = obj["box"]
    if isinstance(box, dict):
        needed = ["cx", "cy", "cz", "l", "w", "h", "rot_z"]
        for k in needed:
            if k not in box:
                raise ValueError(f"Missing '{k}' in box dict. Keys={list(box.keys())}")
        return Box3D(
            cx=float(box["cx"]),
            cy=float(box["cy"]),
            cz=float(box["cz"]),
            l=float(box["l"]),
            w=float(box["w"]),
            h=float(box["h"]),
            rot_z=float(box["rot_z"]),
        )

    if isinstance(box, (list, tuple)) and len(box) == 7:
        cx, cy, cz, l, w, h, rot_z = [float(x) for x in box]
        return Box3D(cx=cx, cy=cy, cz=cz, l=l, w=w, h=h, rot_z=rot_z)

    raise ValueError(f"Unsupported box type: {type(box)}")


# ============================================================
# Geometry helpers (bottom-fixed z) (shared)
# ============================================================

def _bottom_z_from_box7(box7: np.ndarray) -> float:
    return float(box7[2]) - 0.5 * float(box7[5])


def _set_height_keep_bottom(box7: np.ndarray, new_h: float) -> None:
    """
    Mutates box7 in-place:
      - sets h=new_h (>=eps)
      - adjusts cz so that bottom (cz - h/2) remains unchanged
    """
    bottom = _bottom_z_from_box7(box7)
    h = _safe_pos(new_h)
    box7[5] = float(h)
    box7[2] = float(bottom + 0.5 * h)


# ============================================================
# Geometry: BEV oriented IoU (CCW-safe) (shared)
# ============================================================

def _poly_signed_area(poly: np.ndarray) -> float:
    """Signed shoelace area (positive => CCW). poly: (N,2)."""
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _poly_area(poly: np.ndarray) -> float:
    """Unsigned area."""
    return float(abs(_poly_signed_area(poly)))


def _rect_corners_xy(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """
    Return (4,2) rectangle corners CCW in XY.

    IMPORTANT: The clipper assumes CCW clip polygons.
    """
    dx = 0.5 * float(l)
    dy = 0.5 * float(w)

    # CCW order:
    # top-right -> top-left -> bottom-left -> bottom-right
    local = np.array(
        [[ dx,  dy],
         [-dx,  dy],
         [-dx, -dy],
         [ dx, -dy]],
        dtype=np.float64,
    )

    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float64)

    pts = local @ R.T
    pts[:, 0] += float(cx)
    pts[:, 1] += float(cy)

    # Safety: enforce CCW
    if _poly_signed_area(pts) < 0.0:
        pts = pts[::-1].copy()

    return pts


def _is_inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Point p inside half-plane to the left of edge a->b (CCW clip polygon)."""
    return float((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= -1e-12


def _line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of segment p1->p2 with infinite line a->b, assuming not parallel."""
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
    """Clip convex subject polygon by convex clip polygon (both CCW)."""
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


def _precompute_bev_rects(boxes: Sequence[Box3D]) -> Tuple[np.ndarray, np.ndarray]:
    """Return corners (N,4,2) and areas (N,) for BEV rectangles."""
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


# ============================================================
# Matching helpers (shared)
# ============================================================

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


def _gate_pair_distance_only(cfg: Any, box_gt: Box3D, box_det: Box3D) -> Tuple[bool, float]:
    """
    Uses cfg.z_gate_m and cfg.dist_gate_m.
    Returns (ok, dist_xy). If z-gate fails returns (False, inf).
    """
    if float(getattr(cfg, "z_gate_m", 0.0)) > 0.0:
        if abs(float(box_gt.cz - box_det.cz)) > float(getattr(cfg, "z_gate_m")):
            return (False, float("inf"))

    dx = float(box_gt.cx - box_det.cx)
    dy = float(box_gt.cy - box_det.cy)
    dist = float(math.sqrt(dx * dx + dy * dy))
    if dist > float(getattr(cfg, "dist_gate_m")):
        return (False, dist)
    return (True, dist)


def _build_candidates(
    cfg: Any,
    gt_boxes: List[Box3D],
    det_boxes: List[Box3D],
    *,
    gt_xy: np.ndarray,
    det_xy: np.ndarray,
    gt_corners: np.ndarray,
    gt_areas: np.ndarray,
    det_corners: np.ndarray,
    det_areas: np.ndarray,
    min_iou: Optional[float] = None,
) -> List[Tuple[int, int, float, float]]:
    """
    Build sparse candidate edges (gi, dj, iou, dist) using radius query + top-k pruning.

    - Uses cfg.dist_gate_m, cfg.z_gate_m, cfg.assoc_topk
    - If min_iou is None and cfg has tp_iou_thr, uses that (backwards-compatible).
    """
    nG = len(gt_boxes)
    nD = len(det_boxes)
    if nG == 0 or nD == 0:
        return []

    if min_iou is None and hasattr(cfg, "tp_iou_thr"):
        try:
            min_iou = float(getattr(cfg, "tp_iou_thr"))
        except Exception:
            min_iou = None

    dist_gate = float(getattr(cfg, "dist_gate_m"))
    topk = int(max(1, int(getattr(cfg, "assoc_topk", 10))))

    if cKDTree is not None:
        tree = cKDTree(det_xy)
        neigh = tree.query_ball_point(gt_xy, r=dist_gate)
    else:
        neigh = []
        r2 = dist_gate ** 2
        for i in range(nG):
            dx = det_xy[:, 0] - gt_xy[i, 0]
            dy = det_xy[:, 1] - gt_xy[i, 1]
            d2 = dx * dx + dy * dy
            idx = np.where(d2 <= r2)[0]
            neigh.append(idx.tolist())

    edges: List[Tuple[int, int, float, float]] = []

    for gi in range(nG):
        cand_js = neigh[gi]
        if not cand_js:
            continue

        if len(cand_js) > topk:
            dxy = det_xy[np.array(cand_js)] - gt_xy[gi : gi + 1, :]
            d2 = np.sum(dxy * dxy, axis=1)
            keep = np.argpartition(d2, topk)[:topk]
            cand_js = [cand_js[k] for k in keep.tolist()]

        bgt = gt_boxes[gi]
        for dj in cand_js:
            bdet = det_boxes[dj]
            ok, dist = _gate_pair_distance_only(cfg, bgt, bdet)
            if not ok:
                continue

            iou = bev_iou_oriented_cached(
                gt_corners[gi], float(gt_areas[gi]),
                det_corners[dj], float(det_areas[dj]),
            )

            if min_iou is not None and float(iou) < float(min_iou):
                continue

            edges.append((gi, dj, float(iou), float(dist)))

    return edges


def _assign_component_hungarian(
    cfg: Any,
    nG: int,
    nD: int,
    edges: List[Tuple[int, int, float, float]],
) -> List[Tuple[int, int]]:
    """
    Component-wise Hungarian (or greedy fallback).
    Uses cfg.assoc_iou_weight and cfg.forbidden_cost.
    """
    if nG == 0 or nD == 0 or not edges:
        return []

    W = float(getattr(cfg, "assoc_iou_weight"))
    big = float(getattr(cfg, "forbidden_cost"))

    uf = _UnionFind(nG + nD)
    for gi, dj, _, _ in edges:
        uf.union(gi, nG + dj)

    comp_gt: Dict[int, Set[int]] = {}
    comp_det: Dict[int, Set[int]] = {}
    comp_edges: Dict[int, List[Tuple[int, int, float, float]]] = {}

    for gi, dj, iou, dist in edges:
        r = uf.find(gi)
        comp_gt.setdefault(r, set()).add(gi)
        comp_det.setdefault(r, set()).add(dj)
        comp_edges.setdefault(r, []).append((gi, dj, iou, dist))

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

        if linear_sum_assignment is None:
            flat: List[Tuple[float, int, int]] = []
            for ii in range(ng):
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
                matches.append((Gset[ii], Dset[jj]))
            continue

        row_ind, col_ind = linear_sum_assignment(cost)
        for ii, jj in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[ii, jj] >= big * 0.5:
                continue
            matches.append((Gset[ii], Dset[jj]))

    return matches


# ============================================================
# Small numeric helper (shared)
# ============================================================

def _stats(x: np.ndarray) -> Dict[str, Any]:
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
    }
