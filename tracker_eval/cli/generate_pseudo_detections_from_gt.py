# tracker_eval/cli/generate_pseudo_detections_from_gt.py
from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from tracker_eval.utils import (
    _seed_u32,
    _wrap_angle_rad_pi,
    _ceil_frames,
    _clip01,
    _safe_pos,
    _trunc_normal,
    _parse_frame_key,
    _parse_label_id_strict,
    _box7_from_label_obj,
    _load_labels_3d_json,
    _set_height_keep_bottom,
)


def _all_frame_keys_from_gt_json(gt_json: Path) -> List[str]:
    frame_dict = _load_labels_3d_json(gt_json)
    keys: List[str] = []
    for k in frame_dict.keys():
        fr = int(str(k).split(".")[0])
        keys.append(f"{fr:06d}.pcd")
    return sorted(keys)


# ============================================================
# Score distribution loading + sampling
# ============================================================

@dataclass(frozen=True)
class ScoreDistributions:
    tp_scores: np.ndarray  # (N,)
    fp_scores: np.ndarray  # (M,)


def _load_score_distributions_json(path: Path) -> ScoreDistributions:
    """
    Supports a few reasonable schemas:
      A) {"tp_scores":[...], "fp_scores":[...]}
      B) {"scores":{"tp":[...], "fp":[...]}}
      C) {"tp":[...], "fp":[...]}  (fallback keys)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    tp = None
    fp = None

    if isinstance(data, dict):
        if "tp_scores" in data and "fp_scores" in data:
            tp = data.get("tp_scores")
            fp = data.get("fp_scores")
        elif "scores" in data and isinstance(data["scores"], dict):
            tp = data["scores"].get("tp", None)
            fp = data["scores"].get("fp", None)
        else:
            # very common fallback
            tp = data.get("tp", None)
            fp = data.get("fp", None)

    if tp is None or fp is None:
        raise ValueError(
            f"Score distributions JSON missing TP/FP arrays. Got keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    tp_arr = np.asarray(tp, dtype=np.float32).reshape(-1)
    fp_arr = np.asarray(fp, dtype=np.float32).reshape(-1)

    # Keep only finite and clip to [0,1] (your detector is 0.5..1, but be safe)
    tp_arr = tp_arr[np.isfinite(tp_arr)]
    fp_arr = fp_arr[np.isfinite(fp_arr)]
    tp_arr = np.clip(tp_arr, 0.0, 1.0)
    fp_arr = np.clip(fp_arr, 0.0, 1.0)

    if tp_arr.size == 0 or fp_arr.size == 0:
        raise ValueError(f"Score distributions are empty after filtering. tp={tp_arr.size}, fp={fp_arr.size}")

    return ScoreDistributions(tp_scores=tp_arr, fp_scores=fp_arr)


class ScoreSampler:
    """
    Deterministic score sampler: samples with replacement from TP or FP arrays.
    """
    def __init__(self, dists: ScoreDistributions) -> None:
        self.tp = dists.tp_scores
        self.fp = dists.fp_scores

    @staticmethod
    def _sample_from(arr: np.ndarray, rng: np.random.Generator) -> float:
        if arr.size <= 0:
            return 1.0
        j = int(rng.integers(0, arr.size))
        return float(arr[j])

    def sample_tp(self, rng: np.random.Generator) -> float:
        return self._sample_from(self.tp, rng)

    def sample_fp(self, rng: np.random.Generator) -> float:
        return self._sample_from(self.fp, rng)


# ============================================================
# Corruption config
# ============================================================

@dataclass
class VariantCfg:
    name: str
    fps: float
    class_name: str = "pedestrian"
    severity: float = 1.0

    # -------------------------
    # 1) Dropout / FN bursts
    # -------------------------
    dropout_enable: bool = False
    dropout_p_start: float = 0.0
    dropout_min_s: float = 0.0
    dropout_max_s: float = 0.0

    # ----------------------------------------------------
    # 2) Temporal instability / hypothesis switching
    # ----------------------------------------------------
    instability_enable: bool = False
    instability_k_modes: int = 3
    instability_p_switch: float = 0.0

    # mode biases (constant while in a mode)
    instability_mode_xy_sigma_m: float = 0.0
    instability_mode_yaw_sigma_rad: float = 0.0
    instability_mode_lwh_sigma_rel: float = 0.0  # relative: dims *= (1 + rel)

    # per-frame jitter around the chosen mode
    instability_jitter_xy_sigma_m: float = 0.0
    instability_jitter_yaw_sigma_rad: float = 0.0
    instability_jitter_lwh_sigma_rel: float = 0.0

    instability_p_yaw_random: float = 0.0

    # ----------------------------------------------------
    # 3) Confuser FP tracklets (moving + static)
    # ----------------------------------------------------
    confuser_enable: bool = False
    confuser_p_start: float = 0.0
    confuser_min_s: float = 0.0
    confuser_max_s: float = 0.0
    confuser_max_active: int = 1

    confuser_p_static: float = 0.0

    # Moving confuser params
    confuser_offset_xy_mu_m: float = 0.0
    confuser_offset_xy_sigma_m: float = 0.0
    confuser_yaw_sigma_rad: float = 0.0
    confuser_lwh_sigma_rel: float = 0.0

    confuser_jitter_xy_sigma_m: float = 0.0
    confuser_jitter_yaw_sigma_rad: float = 0.0
    confuser_jitter_lwh_sigma_rel: float = 0.0

    confuser_p_yaw_random: float = 0.0

    # Static confuser params (0 => fallback to moving)
    confuser_static_offset_xy_mu_m: float = 0.0
    confuser_static_offset_xy_sigma_m: float = 0.0
    confuser_static_yaw_sigma_rad: float = 0.0
    confuser_static_lwh_sigma_rel: float = 0.0

    confuser_static_jitter_xy_sigma_m: float = 0.0
    confuser_static_jitter_yaw_sigma_rad: float = 0.0
    confuser_static_jitter_lwh_sigma_rel: float = 0.0

    confuser_static_p_yaw_random: float = 0.0

    confuser_only_when_primary_present: bool = True

    # -------------------------
    # Score handling
    # -------------------------
    # Backwards compatible: used when score_mode == "constant" or no sampler provided.
    score_value: float = 1.0

    # New: "sample" or "constant"
    score_mode: str = "constant"

    # Optional: path in YAML; can be overridden by CLI
    score_dists_json: str = ""


def _apply_severity_once(cfg_in: VariantCfg) -> VariantCfg:
    """
    Returns a NEW VariantCfg with severity applied exactly once.
    (Does not mutate input.)
    """
    cfg = VariantCfg(**cfg_in.__dict__)
    s = float(cfg.severity)

    # probabilities scale (clipped)
    cfg.dropout_p_start = _clip01(cfg.dropout_p_start * s)
    cfg.instability_p_switch = _clip01(cfg.instability_p_switch * s)
    cfg.instability_p_yaw_random = _clip01(cfg.instability_p_yaw_random * s)

    cfg.confuser_p_start = _clip01(cfg.confuser_p_start * s)
    cfg.confuser_p_yaw_random = _clip01(cfg.confuser_p_yaw_random * s)
    cfg.confuser_p_static = _clip01(cfg.confuser_p_static)  # ratio, not severity-scaled
    cfg.confuser_static_p_yaw_random = _clip01(cfg.confuser_static_p_yaw_random * s)

    # magnitudes scale
    cfg.instability_mode_xy_sigma_m *= s
    cfg.instability_mode_yaw_sigma_rad *= s
    cfg.instability_mode_lwh_sigma_rel *= s
    cfg.instability_jitter_xy_sigma_m *= s
    cfg.instability_jitter_yaw_sigma_rad *= s
    cfg.instability_jitter_lwh_sigma_rel *= s

    cfg.confuser_offset_xy_mu_m *= s
    cfg.confuser_offset_xy_sigma_m *= s
    cfg.confuser_yaw_sigma_rad *= s
    cfg.confuser_lwh_sigma_rel *= s
    cfg.confuser_jitter_xy_sigma_m *= s
    cfg.confuser_jitter_yaw_sigma_rad *= s
    cfg.confuser_jitter_lwh_sigma_rel *= s

    cfg.confuser_static_offset_xy_mu_m *= s
    cfg.confuser_static_offset_xy_sigma_m *= s
    cfg.confuser_static_yaw_sigma_rad *= s
    cfg.confuser_static_lwh_sigma_rel *= s
    cfg.confuser_static_jitter_xy_sigma_m *= s
    cfg.confuser_static_jitter_yaw_sigma_rad *= s
    cfg.confuser_static_jitter_lwh_sigma_rel *= s

    cfg.instability_k_modes = max(1, int(cfg.instability_k_modes))
    cfg.confuser_max_active = max(0, int(cfg.confuser_max_active))
    cfg.score_mode = str(cfg.score_mode).strip().lower() or "constant"
    return cfg


# ============================================================
# Failure mode models
# ============================================================

def _make_dropout_keep_mask(
    frames: np.ndarray,
    fps: float,
    p_start: float,
    min_s: float,
    max_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Bursty dropout:
      each frame, with prob p_start start a dropout burst of duration U[min_s,max_s] seconds.
    """
    n = int(frames.shape[0])
    keep = np.ones((n,), dtype=bool)
    if n == 0:
        return keep

    p_start = _clip01(p_start)
    if p_start <= 0.0 or max_s <= 0.0:
        return keep

    min_k = _ceil_frames(min_s, fps)
    max_k = _ceil_frames(max_s, fps)
    max_k = max(min_k, max_k)

    dropout_until = -10**9
    for i, fr in enumerate(frames.tolist()):
        if fr <= dropout_until:
            keep[i] = False
            continue

        if rng.random() < p_start:
            dur = int(rng.integers(min_k, max_k + 1))
            dropout_until = fr + dur - 1
            keep[i] = False
        else:
            keep[i] = True
    return keep


def _apply_instability_hypothesis_switching(
    frames: np.ndarray,
    gt_boxes: np.ndarray,
    cfg: VariantCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Primary stream: one detection per GT frame (before dropout),
    using K hypotheses (modes) with Markov switching.

    Internal box7: (cx,cy,cz,l,w,h,rot_z)
    """
    boxes = gt_boxes.astype(np.float32).copy()
    n = int(frames.shape[0])
    if n == 0 or not cfg.instability_enable:
        return boxes

    K = max(1, int(cfg.instability_k_modes))
    p_switch = _clip01(cfg.instability_p_switch)

    # Mode biases: sampled once per track (TRUNCATED to ±2σ for plausibility)
    mode_xy = _trunc_normal(rng, 0.0, float(cfg.instability_mode_xy_sigma_m), size=(K, 2), n_sigma=2.0).astype(np.float32, copy=False)
    mode_yaw = _trunc_normal(rng, 0.0, float(cfg.instability_mode_yaw_sigma_rad), size=(K,), n_sigma=2.0).astype(np.float32, copy=False)
    mode_lwh_rel = _trunc_normal(rng, 0.0, float(cfg.instability_mode_lwh_sigma_rel), size=(K, 3), n_sigma=2.0).astype(np.float32, copy=False)

    mode_idx = int(rng.integers(0, K))

    for i in range(n):
        if i > 0 and (rng.random() < p_switch) and K > 1:
            j = int(rng.integers(0, K - 1))
            if j >= mode_idx:
                j += 1
            mode_idx = j

        bxy = mode_xy[mode_idx]
        byaw = float(mode_yaw[mode_idx])
        blwh = mode_lwh_rel[mode_idx]

        # Apply mode bias
        boxes[i, 0] += float(bxy[0])
        boxes[i, 1] += float(bxy[1])
        boxes[i, 6] = _wrap_angle_rad_pi(float(boxes[i, 6]) + byaw)

        # Apply size mode bias (relative) with bottom-fixed height handling
        new_l = _safe_pos(float(boxes[i, 3]) * (1.0 + float(blwh[0])))
        new_w = _safe_pos(float(boxes[i, 4]) * (1.0 + float(blwh[1])))
        new_h = _safe_pos(float(boxes[i, 5]) * (1.0 + float(blwh[2])))

        boxes[i, 3] = float(new_l)
        boxes[i, 4] = float(new_w)
        _set_height_keep_bottom(boxes[i], float(new_h))

        # Per-frame jitter
        if cfg.instability_jitter_xy_sigma_m > 0.0:
            boxes[i, 0] += float(_trunc_normal(rng, 0.0, float(cfg.instability_jitter_xy_sigma_m), size=(), n_sigma=2.0))
            boxes[i, 1] += float(_trunc_normal(rng, 0.0, float(cfg.instability_jitter_xy_sigma_m), size=(), n_sigma=2.0))

        if cfg.instability_jitter_lwh_sigma_rel > 0.0:
            jlwh = _trunc_normal(rng, 0.0, float(cfg.instability_jitter_lwh_sigma_rel), size=(3,), n_sigma=2.0).astype(np.float32, copy=False)
            boxes[i, 3] = _safe_pos(float(boxes[i, 3]) * (1.0 + float(jlwh[0])))
            boxes[i, 4] = _safe_pos(float(boxes[i, 4]) * (1.0 + float(jlwh[1])))

            new_h2 = _safe_pos(float(boxes[i, 5]) * (1.0 + float(jlwh[2])))
            _set_height_keep_bottom(boxes[i], float(new_h2))

        if cfg.instability_p_yaw_random > 0.0 and (rng.random() < float(cfg.instability_p_yaw_random)):
            boxes[i, 6] = _wrap_angle_rad_pi(float(rng.uniform(-math.pi, math.pi)))
        elif cfg.instability_jitter_yaw_sigma_rad > 0.0:
            boxes[i, 6] = _wrap_angle_rad_pi(
                float(boxes[i, 6]) + float(_trunc_normal(rng, 0.0, float(cfg.instability_jitter_yaw_sigma_rad), size=(), n_sigma=2.0))
            )

    return boxes.astype(np.float32)


def _sample_offset_xy(mu: float, sigma: float, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Sample an offset vector with magnitude ~ N(mu, sigma) folded to >=0,
    direction uniform.

    Magnitude sampling is truncated to ±2σ (then abs).
    """
    mu = float(mu)
    sigma = float(sigma)
    if sigma > 0.0:
        mag = float(abs(_trunc_normal(rng, mu, sigma, size=(), n_sigma=2.0)))
    else:
        mag = float(abs(mu))
    ang = float(rng.uniform(-math.pi, math.pi))
    return mag * math.cos(ang), mag * math.sin(ang)


@dataclass
class _ActiveConfuser:
    end_frame: int
    kind: str  # "moving" or "static"
    bias_xy: Tuple[float, float]
    bias_yaw: float
    bias_lwh_rel: Tuple[float, float, float]
    static_anchor_xy: Optional[Tuple[float, float]] = None


def _emit_confuser_tracklets_for_track(
    frames: np.ndarray,
    gt_boxes: np.ndarray,
    primary_keep: np.ndarray,
    cfg: VariantCfg,
    rng: np.random.Generator,
) -> Dict[int, List[np.ndarray]]:
    """
    Generate confuser FP tracklets near the GT track.
    Returns dict: frame_int -> list of box7.
    """
    out: Dict[int, List[np.ndarray]] = {}
    n = int(frames.shape[0])
    if n == 0 or not cfg.confuser_enable or cfg.confuser_p_start <= 0.0 or cfg.confuser_max_active <= 0:
        return out

    fps = float(cfg.fps)
    min_k = _ceil_frames(cfg.confuser_min_s, fps) if cfg.confuser_max_s > 0.0 else 1
    max_k = _ceil_frames(cfg.confuser_max_s, fps) if cfg.confuser_max_s > 0.0 else 1
    max_k = max(min_k, max_k)

    active: List[_ActiveConfuser] = []

    def _pick_static_param(v_static: float, v_moving: float) -> float:
        return float(v_static) if float(v_static) != 0.0 else float(v_moving)

    for i in range(n):
        fr = int(frames[i])

        # retire expired
        active = [a for a in active if fr <= a.end_frame]

        # optionally only emit/start when primary is present
        if cfg.confuser_only_when_primary_present and not bool(primary_keep[i]):
            continue

        # start new confuser tracklet?
        if (len(active) < int(cfg.confuser_max_active)) and (rng.random() < float(cfg.confuser_p_start)):
            dur = int(rng.integers(min_k, max_k + 1))
            end_fr = fr + dur - 1

            is_static = (rng.random() < float(cfg.confuser_p_static))
            kind = "static" if is_static else "moving"

            if kind == "moving":
                off_mu = float(cfg.confuser_offset_xy_mu_m)
                off_sig = float(cfg.confuser_offset_xy_sigma_m)
                yaw_sig = float(cfg.confuser_yaw_sigma_rad)
                lwh_sig = float(cfg.confuser_lwh_sigma_rel)
            else:
                off_mu = _pick_static_param(cfg.confuser_static_offset_xy_mu_m, cfg.confuser_offset_xy_mu_m)
                off_sig = _pick_static_param(cfg.confuser_static_offset_xy_sigma_m, cfg.confuser_offset_xy_sigma_m)
                yaw_sig = _pick_static_param(cfg.confuser_static_yaw_sigma_rad, cfg.confuser_yaw_sigma_rad)
                lwh_sig = _pick_static_param(cfg.confuser_static_lwh_sigma_rel, cfg.confuser_lwh_sigma_rel)

            bx, by = _sample_offset_xy(off_mu, off_sig, rng)
            byaw = float(_trunc_normal(rng, 0.0, yaw_sig, size=(), n_sigma=2.0)) if yaw_sig > 0.0 else 0.0

            if lwh_sig > 0.0:
                blwh = _trunc_normal(rng, 0.0, lwh_sig, size=(3,), n_sigma=2.0).astype(np.float32, copy=False)
                bias_lwh = (float(blwh[0]), float(blwh[1]), float(blwh[2]))
            else:
                bias_lwh = (0.0, 0.0, 0.0)

            static_anchor_xy: Optional[Tuple[float, float]] = None
            if kind == "static":
                gt = gt_boxes[i].astype(np.float32)
                static_anchor_xy = (float(gt[0]) + float(bx), float(gt[1]) + float(by))

            active.append(
                _ActiveConfuser(
                    end_frame=end_fr,
                    kind=kind,
                    bias_xy=(bx, by),
                    bias_yaw=byaw,
                    bias_lwh_rel=bias_lwh,
                    static_anchor_xy=static_anchor_xy,
                )
            )

        if not active:
            continue

        gt = gt_boxes[i].astype(np.float32)
        cx, cy, cz, l, w, h, rot_z = [float(v) for v in gt.tolist()]

        for a in active:
            b = gt.copy()

            if a.kind == "moving":
                b[0] = cx + float(a.bias_xy[0])
                b[1] = cy + float(a.bias_xy[1])

                jitter_xy = float(cfg.confuser_jitter_xy_sigma_m)
                jitter_yaw = float(cfg.confuser_jitter_yaw_sigma_rad)
                jitter_lwh = float(cfg.confuser_jitter_lwh_sigma_rel)
                p_yaw_rand = float(cfg.confuser_p_yaw_random)
            else:
                ax, ay = a.static_anchor_xy if a.static_anchor_xy is not None else (cx, cy)
                b[0] = float(ax)
                b[1] = float(ay)

                jitter_xy = _pick_static_param(cfg.confuser_static_jitter_xy_sigma_m, cfg.confuser_jitter_xy_sigma_m)
                jitter_yaw = _pick_static_param(cfg.confuser_static_jitter_yaw_sigma_rad, cfg.confuser_jitter_yaw_sigma_rad)
                jitter_lwh = _pick_static_param(cfg.confuser_static_jitter_lwh_sigma_rel, cfg.confuser_jitter_lwh_sigma_rel)
                p_yaw_rand = float(cfg.confuser_static_p_yaw_random) if float(cfg.confuser_static_p_yaw_random) != 0.0 else float(cfg.confuser_p_yaw_random)

            # yaw: random with prob, else biased + jitter (jitter applies on non-random-yaw frames)
            if p_yaw_rand > 0.0 and (rng.random() < p_yaw_rand):
                b[6] = _wrap_angle_rad_pi(float(rng.uniform(-math.pi, math.pi)))
            else:
                b[6] = _wrap_angle_rad_pi(rot_z + float(a.bias_yaw))
                if jitter_yaw > 0.0:
                    b[6] = _wrap_angle_rad_pi(
                        float(b[6]) + float(_trunc_normal(rng, 0.0, float(jitter_yaw), size=(), n_sigma=2.0))
                    )

            # size bias (relative) with bottom-fixed height
            new_l = _safe_pos(l * (1.0 + float(a.bias_lwh_rel[0])))
            new_w = _safe_pos(w * (1.0 + float(a.bias_lwh_rel[1])))
            new_h = _safe_pos(h * (1.0 + float(a.bias_lwh_rel[2])))

            b[3] = float(new_l)
            b[4] = float(new_w)
            _set_height_keep_bottom(b, float(new_h))

            # per-frame jitter
            if jitter_xy > 0.0:
                b[0] += float(_trunc_normal(rng, 0.0, float(jitter_xy), size=(), n_sigma=2.0))
                b[1] += float(_trunc_normal(rng, 0.0, float(jitter_xy), size=(), n_sigma=2.0))

            if jitter_lwh > 0.0:
                jlwh = _trunc_normal(rng, 0.0, float(jitter_lwh), size=(3,), n_sigma=2.0).astype(np.float32, copy=False)
                b[3] = _safe_pos(float(b[3]) * (1.0 + float(jlwh[0])))
                b[4] = _safe_pos(float(b[4]) * (1.0 + float(jlwh[1])))
                new_h2 = _safe_pos(float(b[5]) * (1.0 + float(jlwh[2])))
                _set_height_keep_bottom(b, float(new_h2))

            out.setdefault(fr, []).append(b.astype(np.float32))

    return out


# ============================================================
# Output structure: per-frame detections with scores
# ============================================================

@dataclass
class _DetRow:
    box7: np.ndarray
    score: float


def _sample_primary_score(cfg: VariantCfg, sampler: Optional[ScoreSampler], rng_score: np.random.Generator) -> float:
    if cfg.score_mode == "sample" and sampler is not None:
        return float(sampler.sample_tp(rng_score))
    return float(cfg.score_value)


def _sample_confuser_score(cfg: VariantCfg, sampler: Optional[ScoreSampler], rng_score: np.random.Generator) -> float:
    if cfg.score_mode == "sample" and sampler is not None:
        return float(sampler.sample_fp(rng_score))
    return float(cfg.score_value)


def _generate_pseudo_boxes_by_frame(
    gt_by_tid: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cfg: VariantCfg,  # severity already applied once
    rng_seed_base: int,
    variant_name: str,
    seq_name: str,
    score_sampler: Optional[ScoreSampler],
) -> Dict[str, List[_DetRow]]:
    """
    Returns: frame_key(str like '000123.pcd') -> list of _DetRow(box7, score)

    Score policy:
      - Primary detections (GT-derived): sample from TP distribution
      - Confusers (FP tracklets): sample from FP distribution
    """
    out_by_frame: Dict[str, List[_DetRow]] = {}

    for tid, (frames, gt_boxes) in gt_by_tid.items():
        # Geometry RNG (existing behavior)
        rng = np.random.default_rng(_seed_u32(rng_seed_base, variant_name, seq_name, "tid", int(tid), "geom"))

        # Separate RNG stream for scores (keeps determinism stable even if geom logic changes a bit)
        rng_score = np.random.default_rng(_seed_u32(rng_seed_base, variant_name, seq_name, "tid", int(tid), "score"))

        frames_i = frames.astype(np.int32)
        gt_i = gt_boxes.astype(np.float32)

        # Primary detection path: instability (hypothesis switching)
        primary_boxes = _apply_instability_hypothesis_switching(frames_i, gt_i, cfg, rng)

        # Dropout / FN bursts applied to primary detections
        keep = np.ones((len(frames_i),), dtype=bool)
        if cfg.dropout_enable:
            keep = _make_dropout_keep_mask(
                frames=frames_i,
                fps=float(cfg.fps),
                p_start=float(cfg.dropout_p_start),
                min_s=float(cfg.dropout_min_s),
                max_s=float(cfg.dropout_max_s),
                rng=rng,
            )

        # Confuser FP tracklets
        conf_by_frame_int = _emit_confuser_tracklets_for_track(
            frames=frames_i,
            gt_boxes=gt_i,
            primary_keep=keep,
            cfg=cfg,
            rng=rng,
        )

        # Emit primary detections (TP score distribution)
        for fr, box, ok in zip(frames_i.tolist(), primary_boxes, keep.tolist()):
            if not ok:
                continue
            frame_key = f"{int(fr):06d}.pcd"
            s = _sample_primary_score(cfg, score_sampler, rng_score)
            out_by_frame.setdefault(frame_key, []).append(_DetRow(box7=box.astype(np.float32), score=float(s)))

        # Emit confusers (FP score distribution)
        for fr_int, boxes_list in conf_by_frame_int.items():
            frame_key = f"{int(fr_int):06d}.pcd"
            for b in boxes_list:
                s = _sample_confuser_score(cfg, score_sampler, rng_score)
                out_by_frame.setdefault(frame_key, []).append(_DetRow(box7=b.astype(np.float32), score=float(s)))

    return out_by_frame


# ============================================================
# Spec parsing / variant expansion
# ============================================================

def _variant_cfg_from_dict(name: str, base: Dict[str, Any], override: Dict[str, Any]) -> VariantCfg:
    merged = dict(base)
    merged.update(override)

    cfg = VariantCfg(
        name=name,
        fps=float(merged.get("fps", 15.0)),
        class_name=str(merged.get("class_name", "pedestrian")),
        severity=float(merged.get("severity", 1.0)),

        dropout_enable=bool(merged.get("dropout_enable", False)),
        dropout_p_start=float(merged.get("dropout_p_start", 0.0)),
        dropout_min_s=float(merged.get("dropout_min_s", 0.0)),
        dropout_max_s=float(merged.get("dropout_max_s", 0.0)),

        instability_enable=bool(merged.get("instability_enable", False)),
        instability_k_modes=int(merged.get("instability_k_modes", 3)),
        instability_p_switch=float(merged.get("instability_p_switch", 0.0)),
        instability_mode_xy_sigma_m=float(merged.get("instability_mode_xy_sigma_m", 0.0)),
        instability_mode_yaw_sigma_rad=float(merged.get("instability_mode_yaw_sigma_rad", 0.0)),
        instability_mode_lwh_sigma_rel=float(merged.get("instability_mode_lwh_sigma_rel", 0.0)),
        instability_jitter_xy_sigma_m=float(merged.get("instability_jitter_xy_sigma_m", 0.0)),
        instability_jitter_yaw_sigma_rad=float(merged.get("instability_jitter_yaw_sigma_rad", 0.0)),
        instability_jitter_lwh_sigma_rel=float(merged.get("instability_jitter_lwh_sigma_rel", 0.0)),
        instability_p_yaw_random=float(merged.get("instability_p_yaw_random", 0.0)),

        confuser_enable=bool(merged.get("confuser_enable", False)),
        confuser_p_start=float(merged.get("confuser_p_start", 0.0)),
        confuser_min_s=float(merged.get("confuser_min_s", 0.0)),
        confuser_max_s=float(merged.get("confuser_max_s", 0.0)),
        confuser_max_active=int(merged.get("confuser_max_active", 1)),
        confuser_p_static=float(merged.get("confuser_p_static", 0.0)),

        confuser_offset_xy_mu_m=float(merged.get("confuser_offset_xy_mu_m", 0.0)),
        confuser_offset_xy_sigma_m=float(merged.get("confuser_offset_xy_sigma_m", 0.0)),
        confuser_yaw_sigma_rad=float(merged.get("confuser_yaw_sigma_rad", 0.0)),
        confuser_lwh_sigma_rel=float(merged.get("confuser_lwh_sigma_rel", 0.0)),

        confuser_jitter_xy_sigma_m=float(merged.get("confuser_jitter_xy_sigma_m", 0.0)),
        confuser_jitter_yaw_sigma_rad=float(merged.get("confuser_jitter_yaw_sigma_rad", 0.0)),
        confuser_jitter_lwh_sigma_rel=float(merged.get("confuser_jitter_lwh_sigma_rel", 0.0)),
        confuser_p_yaw_random=float(merged.get("confuser_p_yaw_random", 0.0)),

        confuser_static_offset_xy_mu_m=float(merged.get("confuser_static_offset_xy_mu_m", 0.0)),
        confuser_static_offset_xy_sigma_m=float(merged.get("confuser_static_offset_xy_sigma_m", 0.0)),
        confuser_static_yaw_sigma_rad=float(merged.get("confuser_static_yaw_sigma_rad", 0.0)),
        confuser_static_lwh_sigma_rel=float(merged.get("confuser_static_lwh_sigma_rel", 0.0)),
        confuser_static_jitter_xy_sigma_m=float(merged.get("confuser_static_jitter_xy_sigma_m", 0.0)),
        confuser_static_jitter_yaw_sigma_rad=float(merged.get("confuser_static_jitter_yaw_sigma_rad", 0.0)),
        confuser_static_jitter_lwh_sigma_rel=float(merged.get("confuser_static_jitter_lwh_sigma_rel", 0.0)),
        confuser_static_p_yaw_random=float(merged.get("confuser_static_p_yaw_random", 0.0)),

        confuser_only_when_primary_present=bool(merged.get("confuser_only_when_primary_present", True)),

        # scores
        score_value=float(merged.get("score_value", 1.0)),
        score_mode=str(merged.get("score_mode", "constant")),
        score_dists_json=str(merged.get("score_dists_json", "")),
    )

    # clip probabilities
    cfg.dropout_p_start = _clip01(cfg.dropout_p_start)
    cfg.instability_p_switch = _clip01(cfg.instability_p_switch)
    cfg.instability_p_yaw_random = _clip01(cfg.instability_p_yaw_random)

    cfg.confuser_p_start = _clip01(cfg.confuser_p_start)
    cfg.confuser_p_yaw_random = _clip01(cfg.confuser_p_yaw_random)
    cfg.confuser_p_static = _clip01(cfg.confuser_p_static)
    cfg.confuser_static_p_yaw_random = _clip01(cfg.confuser_static_p_yaw_random)

    cfg.instability_k_modes = max(1, int(cfg.instability_k_modes))
    cfg.confuser_max_active = max(0, int(cfg.confuser_max_active))

    cfg.score_mode = str(cfg.score_mode).strip().lower() or "constant"
    return cfg


def _expand_variants_from_spec(spec: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
    base = dict(spec.get("base", {}))
    sweeps = dict(spec.get("single_mode_sweeps", {}))
    combos = list(spec.get("combos", []))

    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    sweep_variants: List[Tuple[str, Dict[str, Any]]] = []

    for mode, levels in sweeps.items():
        if not isinstance(levels, list):
            raise ValueError(f"single_mode_sweeps.{mode} must be a list")
        lookup.setdefault(str(mode), {})
        for lev in levels:
            if not isinstance(lev, dict):
                continue
            lev_name = str(lev.get("name", "")).strip()
            if not lev_name:
                raise ValueError(f"Missing 'name' in sweep level for mode '{mode}'")
            lookup[str(mode)][lev_name] = dict(lev)
            sweep_variants.append((f"{mode}_{lev_name}", dict(lev)))

    combo_variants: List[Tuple[str, Dict[str, Any]]] = []
    for c in combos:
        if not isinstance(c, dict):
            continue
        cname = str(c.get("name", "")).strip()
        use = c.get("use", {})
        if not cname or not isinstance(use, dict):
            raise ValueError("Each combo must have 'name' and dict 'use'")
        merged: Dict[str, Any] = {}
        for mode, lev_name in use.items():
            mode_s = str(mode)
            lev_s = str(lev_name)
            if mode_s not in lookup or lev_s not in lookup[mode_s]:
                raise ValueError(f"Combo '{cname}' references missing {mode_s}:{lev_s}")
            merged.update(dict(lookup[mode_s][lev_s]))
        if "overrides" in c and isinstance(c["overrides"], dict):
            merged.update(dict(c["overrides"]))
        combo_variants.append((cname, merged))

    # clean baseline
    clean = ("clean", {
        "dropout_enable": False,
        "dropout_p_start": 0.0,
        "dropout_min_s": 0.0,
        "dropout_max_s": 0.0,

        "instability_enable": False,
        "instability_k_modes": 3,
        "instability_p_switch": 0.0,
        "instability_mode_xy_sigma_m": 0.0,
        "instability_mode_yaw_sigma_rad": 0.0,
        "instability_mode_lwh_sigma_rel": 0.0,
        "instability_jitter_xy_sigma_m": 0.0,
        "instability_jitter_yaw_sigma_rad": 0.0,
        "instability_jitter_lwh_sigma_rel": 0.0,
        "instability_p_yaw_random": 0.0,

        "confuser_enable": False,
        "confuser_p_start": 0.0,
        "confuser_min_s": 0.0,
        "confuser_max_s": 0.0,
        "confuser_max_active": 0,
        "confuser_p_static": 0.0,

        "confuser_offset_xy_mu_m": 0.0,
        "confuser_offset_xy_sigma_m": 0.0,
        "confuser_yaw_sigma_rad": 0.0,
        "confuser_lwh_sigma_rel": 0.0,
        "confuser_jitter_xy_sigma_m": 0.0,
        "confuser_jitter_yaw_sigma_rad": 0.0,
        "confuser_jitter_lwh_sigma_rel": 0.0,
        "confuser_p_yaw_random": 0.0,

        "confuser_static_offset_xy_mu_m": 0.0,
        "confuser_static_offset_xy_sigma_m": 0.0,
        "confuser_static_yaw_sigma_rad": 0.0,
        "confuser_static_lwh_sigma_rel": 0.0,
        "confuser_static_jitter_xy_sigma_m": 0.0,
        "confuser_static_jitter_yaw_sigma_rad": 0.0,
        "confuser_static_jitter_lwh_sigma_rel": 0.0,
        "confuser_static_p_yaw_random": 0.0,

        "confuser_only_when_primary_present": True,
    })

    variants = [clean] + sweep_variants + combo_variants
    return base, variants


# ============================================================
# Load GT-by-track from JSON
# ============================================================

def _load_gt_internal_by_tid_from_json(gt_json: Path, class_name: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    frame_dict = _load_labels_3d_json(gt_json)

    by_tid: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    for frame_key_raw, objs in frame_dict.items():
        frame_key = _parse_frame_key(frame_key_raw)
        fr_str = frame_key.split(".")[0]
        fr = int(fr_str)

        for obj in objs:
            label_id = obj.get("label_id", None)
            if label_id is None:
                continue
            cls, tid = _parse_label_id_strict(label_id)
            if cls.lower() != str(class_name).lower():
                continue
            box7 = _box7_from_label_obj(obj)
            by_tid.setdefault(int(tid), []).append((fr, box7))

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for tid, items in by_tid.items():
        items.sort(key=lambda x: x[0])
        frames = np.array([fr for fr, _ in items], dtype=np.int32)
        boxes = np.stack([b for _, b in items], axis=0).astype(np.float32)
        out[int(tid)] = (frames, boxes)
    return out


# ============================================================
# Write detections JSON in detector schema
# ============================================================

def _write_detections_json(
    out_json: Path,
    dets_by_frame: Dict[str, List[_DetRow]],
    class_name: str,
    all_frame_keys: Optional[List[str]] = None,
) -> None:
    dets: Dict[str, List[Dict[str, Any]]] = {}

    frame_keys = sorted(dets_by_frame.keys()) if all_frame_keys is None else list(all_frame_keys)

    for frame_key in frame_keys:
        rows: List[Dict[str, Any]] = []
        for d in dets_by_frame.get(frame_key, []):
            box7 = d.box7
            cx, cy, cz, l, w, h, rot_z = [float(v) for v in box7.tolist()]
            rows.append({
                "box": {"cx": cx, "cy": cy, "cz": cz, "h": h, "l": l, "rot_z": rot_z, "w": w},
                "label_id": f"{class_name}:-1",
                "file_id": str(frame_key),
                "score": float(d.score),
            })
        dets[str(frame_key)] = rows

    payload = {"detections": dets}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker_eval.generate_pseudo_detections_from_gt",
        description="Generate pseudo-detections (JSON) from GT (labels_3d JSON) with controlled failure modes.",
    )
    p.add_argument("--split_root", type=str, required=True)
    p.add_argument("--labels_subdir", type=str, default="labels_3d")
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--out_detections_subdir", type=str, default="detections_3D_pseudo")
    p.add_argument("--seed", type=int, default=0)

    # New: score distribution file (overrides YAML score_dists_json)
    p.add_argument("--score_dists_json", type=str, default=None,
                   help="Path to JSON containing TP/FP score arrays; enables score_mode=sample if provided.")

    p.add_argument("--include_variants", type=str, nargs="*", default=None)
    p.add_argument("--exclude_variants", type=str, nargs="*", default=None)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    split_root = Path(args.split_root)
    labels_dir = split_root / str(args.labels_subdir)

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_subdir not found: {labels_dir}")

    spec_path = Path(args.spec)
    if not spec_path.exists():
        raise FileNotFoundError(f"spec not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    base, variant_defs = _expand_variants_from_spec(spec)

    include = set(args.include_variants) if args.include_variants else None
    exclude = set(args.exclude_variants) if args.exclude_variants else set()
    variant_defs = [(n, o) for (n, o) in variant_defs if (include is None or n in include) and (n not in exclude)]
    if not variant_defs:
        raise ValueError("No variants selected after include/exclude filtering.")

    gt_seq_paths = sorted(labels_dir.glob("*.json"))
    if not gt_seq_paths:
        raise FileNotFoundError(f"No GT .json files found in {labels_dir}")

    out_root = split_root / str(args.out_detections_subdir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load global score sampler once (if provided)
    score_sampler: Optional[ScoreSampler] = None
    score_dists_path_cli = Path(args.score_dists_json) if args.score_dists_json else None
    if score_dists_path_cli is not None:
        if not score_dists_path_cli.exists():
            raise FileNotFoundError(f"--score_dists_json not found: {score_dists_path_cli}")
        score_sampler = ScoreSampler(_load_score_distributions_json(score_dists_path_cli))

    manifest: Dict[str, Any] = {
        "split_root": str(split_root),
        "labels_subdir": str(args.labels_subdir),
        "out_detections_subdir": str(args.out_detections_subdir),
        "seed": int(args.seed),
        "spec_path": str(spec_path),
        "base": base,
        "score_dists_json": str(score_dists_path_cli) if score_dists_path_cli is not None else None,
        "variants": [],
    }

    if not args.quiet:
        print(f"[tracker_eval] GT input: {labels_dir} ({len(gt_seq_paths)} sequences)")
        print(f"[tracker_eval] Output:   {out_root}")
        print(f"[tracker_eval] Variants: {len(variant_defs)}")
        if score_sampler is not None:
            print(f"[tracker_eval] Scores:   sampling enabled (TP/FP distributions loaded)")
        else:
            print(f"[tracker_eval] Scores:   constant (cfg.score_value) unless YAML enables sampling")

    for vi, (vname, voverride) in enumerate(variant_defs):
        cfg_raw = _variant_cfg_from_dict(vname, base, voverride)
        cfg = _apply_severity_once(cfg_raw)

        # Determine per-variant score sampler + mode:
        #  - CLI score_dists_json overrides everything.
        #  - Otherwise, if YAML sets score_mode=sample and score_dists_json, load it.
        v_score_sampler = score_sampler
        if v_score_sampler is None:
            # Try YAML-provided path
            if cfg.score_mode == "sample":
                if not cfg.score_dists_json:
                    raise ValueError(
                        f"Variant '{vname}' has score_mode=sample but no score_dists_json set (and no --score_dists_json)."
                    )
                p = Path(cfg.score_dists_json)
                if not p.exists():
                    raise FileNotFoundError(f"Variant '{vname}' score_dists_json not found: {p}")
                v_score_sampler = ScoreSampler(_load_score_distributions_json(p))

        # If sampler exists, force score_mode to sample (so YAML doesn't accidentally keep constant)
        if v_score_sampler is not None:
            cfg.score_mode = "sample"

        vdir = out_root / vname
        vdir.mkdir(parents=True, exist_ok=True)

        manifest["variants"].append({
            "name": vname,
            "override": voverride,
            "resolved_cfg": cfg.__dict__,
        })

        if not args.quiet:
            print(f"[tracker_eval] ({vi+1}/{len(variant_defs)}) variant='{vname}'")

        for si, gt_json in enumerate(gt_seq_paths):
            seq = gt_json.stem
            all_frame_keys = _all_frame_keys_from_gt_json(gt_json)
            gt_by_tid = _load_gt_internal_by_tid_from_json(gt_json, class_name=cfg.class_name)

            dets_by_frame = _generate_pseudo_boxes_by_frame(
                gt_by_tid=gt_by_tid,
                cfg=cfg,
                rng_seed_base=int(args.seed),
                variant_name=vname,
                seq_name=seq,
                score_sampler=v_score_sampler,
            )

            out_json = vdir / f"{seq}.json"
            _write_detections_json(
                out_json=out_json,
                dets_by_frame=dets_by_frame,
                class_name=cfg.class_name,
                all_frame_keys=all_frame_keys,
            )

            if not args.quiet and (si + 1) % 10 == 0:
                print(f"  ... {si+1}/{len(gt_seq_paths)} sequences")

    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not args.quiet:
        print(f"[tracker_eval] Done. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
