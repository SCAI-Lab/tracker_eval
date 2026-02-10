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


# ============================================================
# Deterministic utilities
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


# ============================================================
# GT JSON parsing (from convert_gt_to_kitti_3d.py)
# ============================================================

def _parse_frame_key(k: str) -> str:
    return str(k)


def _parse_label_id(label_id: str) -> Tuple[str, int]:
    s = str(label_id)
    if ":" not in s:
        raise ValueError(f"Unexpected label_id format (no ':'): {label_id}")
    cls, tid = s.split(":", 1)
    return cls.strip(), int(tid)


def _box7_from_label_obj(obj: Dict[str, Any]) -> np.ndarray:
    """
    Returns internal center box7: (cx, cy, cz, l, w, h, rot_z)
    Accepts obj["box"] as dict or list length 7.
    """
    if "box" not in obj:
        raise ValueError("Missing 'box' in label object.")
    box = obj["box"]
    if isinstance(box, dict):
        needed = ["cx", "cy", "cz", "l", "w", "h", "rot_z"]
        for k in needed:
            if k not in box:
                raise ValueError(f"Missing '{k}' in box dict: keys={list(box.keys())}")
        return np.array(
            [float(box["cx"]), float(box["cy"]), float(box["cz"]),
             float(box["l"]), float(box["w"]), float(box["h"]), float(box["rot_z"])],
            dtype=np.float32,
        )
    if isinstance(box, (list, tuple)):
        if len(box) != 7:
            raise ValueError(f"Box list must have length 7, got {len(box)}")
        return np.asarray(box, dtype=np.float32).reshape(7,)
    raise ValueError(f"Unsupported box type: {type(box)}")


def _load_labels_3d_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("labels", "annotations", "frames", "data"):
        if key in data and isinstance(data[key], dict):
            return data[key]  # type: ignore[return-value]

    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        return data  # type: ignore[return-value]

    raise ValueError(f"Could not find per-frame labels dict in {path}")


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

    # with some probability, yaw becomes essentially random (pedestrian yaw chaos)
    instability_p_yaw_random: float = 0.0

    # ----------------------------------------------------
    # 3) Confuser FP tracklets (persistent near-GT FPs)
    # ----------------------------------------------------
    confuser_enable: bool = False
    confuser_p_start: float = 0.0
    confuser_min_s: float = 0.0
    confuser_max_s: float = 0.0
    confuser_max_active: int = 1

    # tracklet bias relative to GT (constant per confuser tracklet)
    confuser_offset_xy_mu_m: float = 0.0
    confuser_offset_xy_sigma_m: float = 0.0
    confuser_yaw_sigma_rad: float = 0.0
    confuser_lwh_sigma_rel: float = 0.0

    # per-frame jitter on the confuser tracklet
    confuser_jitter_xy_sigma_m: float = 0.0
    confuser_jitter_yaw_sigma_rad: float = 0.0
    confuser_jitter_lwh_sigma_rel: float = 0.0

    confuser_p_yaw_random: float = 0.0

    # Should confusers only be emitted when the primary detection exists this frame?
    confuser_only_when_primary_present: bool = True

    # score handling (your choice for now)
    score_value: float = 1.0


def _apply_severity(cfg: VariantCfg) -> VariantCfg:
    s = float(cfg.severity)

    # probabilities scale (clipped)
    cfg.dropout_p_start = _clip01(cfg.dropout_p_start * s)
    cfg.instability_p_switch = _clip01(cfg.instability_p_switch * s)
    cfg.instability_p_yaw_random = _clip01(cfg.instability_p_yaw_random * s)

    cfg.confuser_p_start = _clip01(cfg.confuser_p_start * s)
    cfg.confuser_p_yaw_random = _clip01(cfg.confuser_p_yaw_random * s)

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

    cfg.instability_k_modes = max(1, int(cfg.instability_k_modes))
    cfg.confuser_max_active = max(0, int(cfg.confuser_max_active))
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
    Create one "primary" detection per GT frame (before dropout),
    using K hypotheses (modes) with Markov switching.

    Internal box7: (cx,cy,cz,l,w,h,rot_z)
    """
    boxes = gt_boxes.astype(np.float32).copy()
    n = int(frames.shape[0])
    if n == 0 or not cfg.instability_enable:
        return boxes

    K = max(1, int(cfg.instability_k_modes))
    p_switch = _clip01(cfg.instability_p_switch)

    # Mode biases: sampled once per track
    mode_xy = rng.normal(0.0, float(cfg.instability_mode_xy_sigma_m), size=(K, 2)).astype(np.float32)
    mode_yaw = rng.normal(0.0, float(cfg.instability_mode_yaw_sigma_rad), size=(K,)).astype(np.float32)
    mode_lwh_rel = rng.normal(0.0, float(cfg.instability_mode_lwh_sigma_rel), size=(K, 3)).astype(np.float32)

    # Start in a random mode
    mode_idx = int(rng.integers(0, K))

    for i in range(n):
        if i > 0 and (rng.random() < p_switch) and K > 1:
            # pick a different mode
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

        # Apply size mode bias (relative)
        l = float(boxes[i, 3]) * (1.0 + float(blwh[0]))
        w = float(boxes[i, 4]) * (1.0 + float(blwh[1]))
        h = float(boxes[i, 5]) * (1.0 + float(blwh[2]))
        boxes[i, 3] = _safe_pos(l)
        boxes[i, 4] = _safe_pos(w)
        boxes[i, 5] = _safe_pos(h)

        # Per-frame jitter
        if cfg.instability_jitter_xy_sigma_m > 0.0:
            boxes[i, 0] += float(rng.normal(0.0, float(cfg.instability_jitter_xy_sigma_m)))
            boxes[i, 1] += float(rng.normal(0.0, float(cfg.instability_jitter_xy_sigma_m)))

        if cfg.instability_jitter_lwh_sigma_rel > 0.0:
            jlwh = rng.normal(0.0, float(cfg.instability_jitter_lwh_sigma_rel), size=(3,)).astype(np.float32)
            boxes[i, 3] = _safe_pos(float(boxes[i, 3]) * (1.0 + float(jlwh[0])))
            boxes[i, 4] = _safe_pos(float(boxes[i, 4]) * (1.0 + float(jlwh[1])))
            boxes[i, 5] = _safe_pos(float(boxes[i, 5]) * (1.0 + float(jlwh[2])))

        if cfg.instability_p_yaw_random > 0.0 and (rng.random() < float(cfg.instability_p_yaw_random)):
            boxes[i, 6] = _wrap_angle_rad_pi(float(rng.uniform(-math.pi, math.pi)))
        elif cfg.instability_jitter_yaw_sigma_rad > 0.0:
            boxes[i, 6] = _wrap_angle_rad_pi(float(boxes[i, 6]) + float(rng.normal(0.0, float(cfg.instability_jitter_yaw_sigma_rad))))

    return boxes.astype(np.float32)


def _sample_offset_xy(mu: float, sigma: float, rng: np.random.Generator) -> Tuple[float, float]:
    """
    Sample an offset vector with magnitude ~ N(mu, sigma) truncated to >=0,
    direction uniform.
    """
    mu = float(mu)
    sigma = float(sigma)
    mag = float(abs(rng.normal(mu, sigma))) if sigma > 0.0 else float(abs(mu))
    ang = float(rng.uniform(-math.pi, math.pi))
    return mag * math.cos(ang), mag * math.sin(ang)


@dataclass
class _ActiveConfuser:
    end_frame: int
    bias_xy: Tuple[float, float]
    bias_yaw: float
    bias_lwh_rel: Tuple[float, float, float]


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

            bx, by = _sample_offset_xy(cfg.confuser_offset_xy_mu_m, cfg.confuser_offset_xy_sigma_m, rng)
            byaw = float(rng.normal(0.0, float(cfg.confuser_yaw_sigma_rad))) if cfg.confuser_yaw_sigma_rad > 0.0 else 0.0

            if cfg.confuser_lwh_sigma_rel > 0.0:
                blwh = rng.normal(0.0, float(cfg.confuser_lwh_sigma_rel), size=(3,)).astype(np.float32)
                bias_lwh = (float(blwh[0]), float(blwh[1]), float(blwh[2]))
            else:
                bias_lwh = (0.0, 0.0, 0.0)

            active.append(_ActiveConfuser(end_frame=end_fr, bias_xy=(bx, by), bias_yaw=byaw, bias_lwh_rel=bias_lwh))

        if not active:
            continue

        # emit confusers this frame
        gt = gt_boxes[i].astype(np.float32)
        cx, cy, cz, l, w, h, rot_z = [float(v) for v in gt.tolist()]

        for a in active:
            b = gt.copy()

            # constant bias
            b[0] = cx + float(a.bias_xy[0])
            b[1] = cy + float(a.bias_xy[1])

            # yaw: either random or biased + jitter
            if cfg.confuser_p_yaw_random > 0.0 and (rng.random() < float(cfg.confuser_p_yaw_random)):
                b[6] = _wrap_angle_rad_pi(float(rng.uniform(-math.pi, math.pi)))
            else:
                b[6] = _wrap_angle_rad_pi(rot_z + float(a.bias_yaw))

            # size bias (relative)
            b[3] = _safe_pos(l * (1.0 + float(a.bias_lwh_rel[0])))
            b[4] = _safe_pos(w * (1.0 + float(a.bias_lwh_rel[1])))
            b[5] = _safe_pos(h * (1.0 + float(a.bias_lwh_rel[2])))

            # per-frame jitter
            if cfg.confuser_jitter_xy_sigma_m > 0.0:
                b[0] += float(rng.normal(0.0, float(cfg.confuser_jitter_xy_sigma_m)))
                b[1] += float(rng.normal(0.0, float(cfg.confuser_jitter_xy_sigma_m)))

            if cfg.confuser_jitter_lwh_sigma_rel > 0.0:
                jlwh = rng.normal(0.0, float(cfg.confuser_jitter_lwh_sigma_rel), size=(3,)).astype(np.float32)
                b[3] = _safe_pos(float(b[3]) * (1.0 + float(jlwh[0])))
                b[4] = _safe_pos(float(b[4]) * (1.0 + float(jlwh[1])))
                b[5] = _safe_pos(float(b[5]) * (1.0 + float(jlwh[2])))

            if cfg.confuser_p_yaw_random <= 0.0 and cfg.confuser_jitter_yaw_sigma_rad > 0.0:
                b[6] = _wrap_angle_rad_pi(float(b[6]) + float(rng.normal(0.0, float(cfg.confuser_jitter_yaw_sigma_rad))))

            out.setdefault(fr, []).append(b.astype(np.float32))

    return out


def _generate_pseudo_boxes_by_frame(
    gt_by_tid: Dict[int, Tuple[np.ndarray, np.ndarray]],
    cfg_in: VariantCfg,
    rng_seed_base: int,
    variant_name: str,
    seq_name: str,
) -> Dict[str, List[np.ndarray]]:
    """
    Returns: frame_key(str like '000123.pcd') -> list of internal box7 arrays
    """
    cfg = _apply_severity(cfg_in)
    out_by_frame: Dict[str, List[np.ndarray]] = {}

    for tid, (frames, gt_boxes) in gt_by_tid.items():
        rng = np.random.default_rng(_seed_u32(rng_seed_base, variant_name, seq_name, "tid", int(tid)))

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

        # Confuser FP tracklets (may be gated to primary presence via cfg.confuser_only_when_primary_present)
        conf_by_frame_int = _emit_confuser_tracklets_for_track(
            frames=frames_i,
            gt_boxes=gt_i,          # confusers anchored to GT
            primary_keep=keep,      # for optional gating
            cfg=cfg,
            rng=rng,
        )

        # Emit primary detections
        for fr, box, ok in zip(frames_i.tolist(), primary_boxes, keep.tolist()):
            if not ok:
                continue
            frame_key = f"{int(fr):06d}.pcd"
            out_by_frame.setdefault(frame_key, []).append(box.astype(np.float32))

        # Emit confusers
        for fr_int, boxes_list in conf_by_frame_int.items():
            frame_key = f"{int(fr_int):06d}.pcd"
            out_by_frame.setdefault(frame_key, []).extend([b.astype(np.float32) for b in boxes_list])

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
        confuser_offset_xy_mu_m=float(merged.get("confuser_offset_xy_mu_m", 0.0)),
        confuser_offset_xy_sigma_m=float(merged.get("confuser_offset_xy_sigma_m", 0.0)),
        confuser_yaw_sigma_rad=float(merged.get("confuser_yaw_sigma_rad", 0.0)),
        confuser_lwh_sigma_rel=float(merged.get("confuser_lwh_sigma_rel", 0.0)),
        confuser_jitter_xy_sigma_m=float(merged.get("confuser_jitter_xy_sigma_m", 0.0)),
        confuser_jitter_yaw_sigma_rad=float(merged.get("confuser_jitter_yaw_sigma_rad", 0.0)),
        confuser_jitter_lwh_sigma_rel=float(merged.get("confuser_jitter_lwh_sigma_rel", 0.0)),
        confuser_p_yaw_random=float(merged.get("confuser_p_yaw_random", 0.0)),
        confuser_only_when_primary_present=bool(merged.get("confuser_only_when_primary_present", True)),

        score_value=float(merged.get("score_value", 1.0)),
    )

    # clip probabilities
    cfg.dropout_p_start = _clip01(cfg.dropout_p_start)
    cfg.instability_p_switch = _clip01(cfg.instability_p_switch)
    cfg.instability_p_yaw_random = _clip01(cfg.instability_p_yaw_random)
    cfg.confuser_p_start = _clip01(cfg.confuser_p_start)
    cfg.confuser_p_yaw_random = _clip01(cfg.confuser_p_yaw_random)

    cfg.instability_k_modes = max(1, int(cfg.instability_k_modes))
    cfg.confuser_max_active = max(0, int(cfg.confuser_max_active))
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

    # clean baseline (no corruptions)
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
        "confuser_offset_xy_mu_m": 0.0,
        "confuser_offset_xy_sigma_m": 0.0,
        "confuser_yaw_sigma_rad": 0.0,
        "confuser_lwh_sigma_rel": 0.0,
        "confuser_jitter_xy_sigma_m": 0.0,
        "confuser_jitter_yaw_sigma_rad": 0.0,
        "confuser_jitter_lwh_sigma_rel": 0.0,
        "confuser_p_yaw_random": 0.0,
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
            cls, tid = _parse_label_id(label_id)
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
# Write detections JSON in your detector schema
# ============================================================

def _write_detections_json(
    out_json: Path,
    boxes_by_frame: Dict[str, List[np.ndarray]],
    class_name: str,
    score_value: float = 1.0,
) -> None:
    """
    Writes:
      { "detections": { "000123.pcd": [ {box, label_id, file_id, score}, ... ], ... } }

    Notes:
      - We do NOT encode identity in label_id. We set it to e.g. "pedestrian:-1" for all detections.
      - If you later want unique per-detection IDs, you can change label_id here.
    """
    dets: Dict[str, List[Dict[str, Any]]] = {}
    for frame_key in sorted(boxes_by_frame.keys()):
        rows: List[Dict[str, Any]] = []
        for box7 in boxes_by_frame[frame_key]:
            cx, cy, cz, l, w, h, rot_z = [float(v) for v in box7.tolist()]
            rows.append({
                "box": {"cx": cx, "cy": cy, "cz": cz, "h": h, "l": l, "rot_z": rot_z, "w": w},
                "label_id": f"{class_name}:-1",
                "file_id": str(frame_key),
                "score": float(score_value),
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
        spec = json.load(f)

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

    manifest: Dict[str, Any] = {
        "split_root": str(split_root),
        "labels_subdir": str(args.labels_subdir),
        "out_detections_subdir": str(args.out_detections_subdir),
        "seed": int(args.seed),
        "spec_path": str(spec_path),
        "base": base,
        "variants": [],
    }

    if not args.quiet:
        print(f"[tracker_eval] GT input: {labels_dir} ({len(gt_seq_paths)} sequences)")
        print(f"[tracker_eval] Output:   {out_root}")
        print(f"[tracker_eval] Variants: {len(variant_defs)}")

    for vi, (vname, voverride) in enumerate(variant_defs):
        cfg = _variant_cfg_from_dict(vname, base, voverride)
        cfg = _apply_severity(cfg)

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

            gt_by_tid = _load_gt_internal_by_tid_from_json(gt_json, class_name=cfg.class_name)
            boxes_by_frame = _generate_pseudo_boxes_by_frame(
                gt_by_tid=gt_by_tid,
                cfg_in=cfg,
                rng_seed_base=int(args.seed),
                variant_name=vname,
                seq_name=seq,
            )

            out_json = vdir / f"{seq}.json"
            _write_detections_json(
                out_json=out_json,
                boxes_by_frame=boxes_by_frame,
                class_name=cfg.class_name,
                score_value=cfg.score_value,
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
