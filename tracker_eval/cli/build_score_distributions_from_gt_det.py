# tracker_eval/cli/build_score_distributions_from_gt_det.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from tracker_eval.common.types import Box3D

from tracker_eval.utils import (
    linear_sum_assignment,
    cKDTree,
    _load_frame_dict_any,
    _parse_label_id,
    _box_from_obj,
    _precompute_bev_rects,
    _build_candidates,
    _assign_component_hungarian,
    _stats,
)



# ============================================================
# Matching config + assignment helpers
# ============================================================

@dataclass(frozen=True)
class AssocConfig:
    dist_gate_m: float = 0.4
    z_gate_m: float = 1.0
    assoc_topk: int = 10
    assoc_iou_weight: float = 0.5
    forbidden_cost: float = 1e6
    tp_iou_thr: float = 0.15


def match_one_frame(
    cfg: AssocConfig,
    gt_boxes: List[Box3D],
    det_boxes: List[Box3D],
) -> Set[int]:
    """
    Returns set of detection indices that are TPs.
    Because we IoU-prune edges in _build_candidates, every matched pair is a TP.
    """
    nG = len(gt_boxes)
    nD = len(det_boxes)
    if nD == 0 or nG == 0:
        return set()

    gt_xy = np.array([[b.cx, b.cy] for b in gt_boxes], dtype=np.float64)
    det_xy = np.array([[b.cx, b.cy] for b in det_boxes], dtype=np.float64)

    gt_corners, gt_areas = _precompute_bev_rects(gt_boxes)
    det_corners, det_areas = _precompute_bev_rects(det_boxes)

    edges = _build_candidates(
        cfg,
        gt_boxes, det_boxes,
        gt_xy=gt_xy,
        det_xy=det_xy,
        gt_corners=gt_corners,
        gt_areas=gt_areas,
        det_corners=det_corners,
        det_areas=det_areas,
    )

    pairs = _assign_component_hungarian(cfg, nG, nD, edges)
    return set(dj for (_, dj) in pairs)

def _collect_boxes_scores_for_class(
    objs: List[Dict[str, Any]],
    class_name: str,
    *,
    is_gt: bool,
    min_det_score: float,
) -> Tuple[List[Box3D], List[float]]:
    boxes: List[Box3D] = []
    scores: List[float] = []
    cls_target = str(class_name).lower()

    for obj in objs:
        label_id = obj.get("label_id", obj.get("label", None))
        cls, _tid = _parse_label_id(label_id)
        if cls_target and (str(cls).lower() != cls_target):
            continue

        try:
            b = _box_from_obj(obj)
        except Exception:
            continue

        if is_gt:
            boxes.append(b)
        else:
            sc = obj.get("score", None)
            if sc is None:
                continue
            s = float(sc)
            if s < float(min_det_score):
                continue
            boxes.append(b)
            scores.append(s)

    return boxes, scores

def _pava_non_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Pool-Adjacent-Violators Algorithm (PAVA) for isotonic regression with non-decreasing constraint.
    Solves: min sum_i w_i (x_i - y_i)^2  s.t. x_0 <= x_1 <= ... <= x_{n-1}
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = int(y.size)
    if n == 0:
        return y

    # Blocks: each block has (start, end_exclusive, sum_w, sum_wy)
    starts: List[int] = []
    ends: List[int] = []
    sum_w: List[float] = []
    sum_wy: List[float] = []

    for i in range(n):
        wi = float(max(0.0, w[i]))
        yi = float(y[i])
        starts.append(i)
        ends.append(i + 1)
        sum_w.append(wi)
        sum_wy.append(wi * yi)

        # Merge while violating monotonicity
        while len(starts) >= 2:
            k = len(starts) - 1
            k0 = k - 1

            v0 = (sum_wy[k0] / sum_w[k0]) if sum_w[k0] > 0.0 else 0.0
            v1 = (sum_wy[k] / sum_w[k]) if sum_w[k] > 0.0 else 0.0
            if v0 <= v1 + 1e-15:
                break

            # Merge blocks k0 and k
            ends[k0] = ends[k]
            sum_w[k0] += sum_w[k]
            sum_wy[k0] += sum_wy[k]
            starts.pop()
            ends.pop()
            sum_w.pop()
            sum_wy.pop()

    out = np.zeros(n, dtype=np.float64)
    for s, e, sw, swy in zip(starts, ends, sum_w, sum_wy):
        v = (swy / sw) if sw > 0.0 else 0.0
        out[s:e] = v
    return out


def build_score_calibrator_from_tp_fp(
    tp_scores: np.ndarray,
    fp_scores: np.ndarray,
    *,
    score_min: float,
    score_max: float = 1.0,
    bins: int = 50,
    laplace: float = 1.0,
    monotonic: bool = True,
    edge_mode: str = "uniform",  # "uniform" or "quantile"
) -> Dict[str, np.ndarray]:
    """
    Build a simple score->precision calibrator:
      precision(bin) = (TP + a) / (TP + FP + 2a)   with a=laplace

    Returns dict with:
      edges: (bins+1,)
      prec:  (bins,)
      tp_counts: (bins,)
      fp_counts: (bins,)
    """
    tp = np.asarray(tp_scores, dtype=np.float64)
    fp = np.asarray(fp_scores, dtype=np.float64)

    smin = float(score_min)
    smax = float(score_max)
    smin = max(0.0, min(1.0, smin))
    smax = max(smin + 1e-9, min(1.0, smax))

    bins = int(max(2, bins))
    a = float(max(0.0, laplace))

    # Choose bin edges
    if edge_mode == "quantile":
        all_scores = np.concatenate([tp, fp], axis=0) if (tp.size + fp.size) > 0 else np.array([], dtype=np.float64)
        # Restrict to [smin, smax] for stable quantiles
        if all_scores.size > 0:
            all_scores = np.clip(all_scores, smin, smax)
        if all_scores.size >= bins:
            qs = np.linspace(0.0, 1.0, bins + 1)
            edges = np.quantile(all_scores, qs)
            edges[0] = smin
            edges[-1] = smax
            # If quantiles collapse (duplicate edges), fall back to uniform
            if np.unique(edges).size < (bins + 1):
                edges = np.linspace(smin, smax, bins + 1, dtype=np.float64)
        else:
            edges = np.linspace(smin, smax, bins + 1, dtype=np.float64)
    else:
        edges = np.linspace(smin, smax, bins + 1, dtype=np.float64)

    # Histogram TP/FP in bins
    tp_clip = np.clip(tp, edges[0], edges[-1] - 1e-12) if tp.size else tp
    fp_clip = np.clip(fp, edges[0], edges[-1] - 1e-12) if fp.size else fp

    tp_counts, _ = np.histogram(tp_clip, bins=edges)
    fp_counts, _ = np.histogram(fp_clip, bins=edges)

    tp_counts = tp_counts.astype(np.float64)
    fp_counts = fp_counts.astype(np.float64)

    denom = tp_counts + fp_counts
    base_rate = float(tp.size / max(1.0, (tp.size + fp.size)))

    # Laplace/Beta(a,a) smoothing
    prec = (tp_counts + a) / (denom + 2.0 * a)

    # Fill completely empty bins with global base rate (keeps things sane)
    empty = denom <= 0.0
    if np.any(empty):
        prec[empty] = base_rate

    # Optional monotonic non-decreasing enforcement with isotonic regression
    if bool(monotonic):
        w = denom.copy()
        # give empty bins tiny weight so they don't dominate
        w[empty] = 1e-6
        prec = _pava_non_decreasing(prec, w)

    return {
        "edges": edges.astype(np.float32),
        "prec": prec.astype(np.float32),
        "tp_counts": tp_counts.astype(np.float32),
        "fp_counts": fp_counts.astype(np.float32),
    }

# ============================================================
# Main processing
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker_eval.build_score_distributions_from_gt_det",
        description="Build TP/FP score distributions by matching detections to GT per frame using sparse Hungarian.",
    )
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--splits", type=str, nargs="*", default=["train_val", "test"])
    p.add_argument("--detections_subdir", type=str, default="detections_3D")
    p.add_argument("--labels_subdir", type=str, default="labels_3d")
    p.add_argument("--class_name", type=str, default="pedestrian")
    p.add_argument("--min_det_score", type=float, default=0.5)

    p.add_argument("--dist_gate_m", type=float, default=0.45)
    p.add_argument("--z_gate_m", type=float, default=1.0)
    p.add_argument("--assoc_topk", type=int, default=10)
    p.add_argument("--assoc_iou_weight", type=float, default=5.0)
    p.add_argument("--tp_iou_thr", type=float, default=0.25)
    p.add_argument("--forbidden_cost", type=float, default=1e6)

    # ---- score calibration output ----
    p.add_argument("--write_calibrator", action="store_true",
                   help="If set, also write a score calibrator (binned precision curve) into the output npz.")
    p.add_argument("--calib_bins", type=int, default=50)
    p.add_argument("--calib_laplace", type=float, default=1.0)
    p.add_argument("--calib_monotonic", type=int, default=1, help="1=on, 0=off")
    p.add_argument("--calib_edge_mode", type=str, default="uniform", choices=["uniform", "quantile"])


    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--out_prefix", type=str, default=None)

    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix or f"scores_{str(args.class_name).lower()}"
    out_npz = out_dir / f"{prefix}.npz"
    out_summary = out_dir / f"{prefix}.summary.json"

    cfg = AssocConfig(
        dist_gate_m=float(args.dist_gate_m),
        z_gate_m=float(args.z_gate_m),
        assoc_topk=int(args.assoc_topk),
        assoc_iou_weight=float(args.assoc_iou_weight),
        forbidden_cost=float(args.forbidden_cost),
        tp_iou_thr=float(args.tp_iou_thr),
    )

    tp_scores_all: List[float] = []
    fp_scores_all: List[float] = []
    per_split: Dict[str, Any] = {}

    if not args.quiet:
        print(f"[score_dist] dataset_root: {root}")
        print(f"[score_dist] splits: {args.splits}")
        print(f"[score_dist] class_name: {args.class_name}")
        print(f"[score_dist] min_det_score: {args.min_det_score}")
        print(f"[score_dist] cfg: {cfg}")
        if linear_sum_assignment is None:
            print("[score_dist] NOTE: SciPy not available -> greedy assignment fallback.")
        if cKDTree is None:
            print("[score_dist] NOTE: SciPy KDTree not available -> brute-force neighbor queries (slower).")

    for split in args.splits:
        split_dir = root / split
        det_dir = split_dir / str(args.detections_subdir)
        gt_dir = split_dir / str(args.labels_subdir)

        if not det_dir.exists() or not gt_dir.exists():
            if not args.quiet:
                print(f"[score_dist] skip split='{split}' (missing dirs: {det_dir} or {gt_dir})")
            continue

        det_paths = sorted(det_dir.glob("*.json"))
        if not det_paths:
            if not args.quiet:
                print(f"[score_dist] split='{split}' has no detection json files in {det_dir}")
            continue

        n_files = 0
        n_frames = 0
        n_det = 0
        n_tp = 0
        n_fp = 0

        for det_path in det_paths:
            seq = det_path.stem
            gt_path = gt_dir / f"{seq}.json"
            if not gt_path.exists():
                if not args.quiet:
                    print(f"[score_dist] WARN: missing GT for {seq} in split='{split}'")
                continue

            try:
                det_frames = _load_frame_dict_any(det_path)
                gt_frames = _load_frame_dict_any(gt_path)
            except Exception as e:
                if not args.quiet:
                    print(f"[score_dist] WARN: failed to load {seq}: {e}")
                continue

            frame_keys = sorted(set(det_frames.keys()) | set(gt_frames.keys()))
            if not frame_keys:
                continue

            for fk in frame_keys:
                det_objs = det_frames.get(fk, [])
                gt_objs = gt_frames.get(fk, [])

                det_boxes, det_scores = _collect_boxes_scores_for_class(
                    det_objs, args.class_name, is_gt=False, min_det_score=float(args.min_det_score)
                )
                gt_boxes, _ = _collect_boxes_scores_for_class(
                    gt_objs, args.class_name, is_gt=True, min_det_score=0.0
                )

                n_frames += 1
                if not det_boxes:
                    continue

                tp_det_idx = match_one_frame(cfg, gt_boxes, det_boxes)
                n_det += len(det_scores)

                for j in sorted(tp_det_idx):
                    if 0 <= j < len(det_scores):
                        tp_scores_all.append(float(det_scores[j]))
                        n_tp += 1

                for j, s in enumerate(det_scores):
                    if j not in tp_det_idx:
                        fp_scores_all.append(float(s))
                        n_fp += 1

            n_files += 1

        per_split[str(split)] = {
            "n_files": int(n_files),
            "n_frames_processed": int(n_frames),
            "n_det_scores_seen": int(n_det),
            "n_tp": int(n_tp),
            "n_fp": int(n_fp),
        }

        if not args.quiet:
            print(f"[score_dist] split='{split}': files={n_files}, frames={n_frames}, dets={n_det}, tp={n_tp}, fp={n_fp}")

    tp_arr = np.asarray(tp_scores_all, dtype=np.float32)
    fp_arr = np.asarray(fp_scores_all, dtype=np.float32)

    save_dict: Dict[str, Any] = {"tp_scores": tp_arr, "fp_scores": fp_arr}

    if bool(args.write_calibrator):
        calib = build_score_calibrator_from_tp_fp(
            tp_arr, fp_arr,
            score_min=float(args.min_det_score),
            score_max=1.0,
            bins=int(args.calib_bins),
            laplace=float(args.calib_laplace),
            monotonic=bool(int(args.calib_monotonic)),
            edge_mode=str(args.calib_edge_mode),
        )
        # Store under stable names for headroom_adapter.py
        save_dict.update(
            score_calib_edges=calib["edges"],
            score_calib_prec=calib["prec"],
            score_calib_tp_counts=calib["tp_counts"],
            score_calib_fp_counts=calib["fp_counts"],
        )

    np.savez_compressed(out_npz, **save_dict)


    summary: Dict[str, Any] = {
        "dataset_root": str(root),
        "splits": list(args.splits),
        "detections_subdir": str(args.detections_subdir),
        "labels_subdir": str(args.labels_subdir),
        "class_name": str(args.class_name),
        "min_det_score": float(args.min_det_score),
        "assoc_config": cfg.__dict__,
        "tp_scores": _stats(tp_arr),
        "fp_scores": _stats(fp_arr),
        "per_split": per_split,
        "notes": {
            "scipy_linear_sum_assignment": bool(linear_sum_assignment is not None),
            "scipy_ckdtree": bool(cKDTree is not None),
            "tp_definition": (
                "Per-frame 1-1 matching using sparse Hungarian on edges passing distance/z gates, "
                f"AND IoU >= tp_iou_thr ({cfg.tp_iou_thr}). Matched detections are TP, others are FP."
            ),
        },
    }

    if bool(args.write_calibrator):
        summary["score_calibrator"] = {
            "bins": int(args.calib_bins),
            "laplace": float(args.calib_laplace),
            "monotonic": bool(int(args.calib_monotonic)),
            "edge_mode": str(args.calib_edge_mode),
            "keys_in_npz": [
                "score_calib_edges",
                "score_calib_prec",
                "score_calib_tp_counts",
                "score_calib_fp_counts",
            ],
            "score_range": [float(args.min_det_score), 1.0],
        }

    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if not args.quiet:
        print(f"[score_dist] wrote: {out_npz}")
        print(f"[score_dist] wrote: {out_summary}")
        print(f"[score_dist] TP scores: n={tp_arr.size}, FP scores: n={fp_arr.size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
