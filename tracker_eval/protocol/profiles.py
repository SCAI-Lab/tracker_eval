#!/usr/bin/env python3
"""
JRDB tracker capability profiles under fixed detections.

This script implements the compact evaluation protocol discussed for the RAL
paper. It produces:

1. Global accuracy/runtime overview
   - HOTA and its decomposition are calculated directly from the tracker
     outputs through cached TrackEval/HOTA-event machinery.
   - Global HOTA uses the official count/TP-weighted sequence aggregation,
     rather than an arithmetic mean of sequence HOTA values.
   - TrackEval global HOTA versus median per-frame tracker-step FPS.
   - Vertical P10--P90 spread of per-sequence HOTA, weighted by the common
     number of GT occurrences in each sequence.
   - Horizontal P10--P90 spread of per-frame FPS.
   - Horizontal dot-and-whisker HOTA gap to a GT-assisted PedRefTrack reference.
     The marker is the difference between global HOTA values; whiskers are
     the P10--P90 range of paired per-sequence HOTA gaps.

2. Direct tracker capability profiles on globally defined event populations
   - Gap bridging: continuously correct same-ID output versus elapsed detector
     gap age, after a configurable number of consecutive visible detections.
   - Post-gap recovery: the first detected GT frame after a bounded gap has the
     same tracker ID as the last detected frame before the gap.
   - Identity continuation versus matched-detector centre-to-centre nearest-
     neighbour distance.
   - Identity-preserving update success versus detector-centre error.
   - Failure-rate views (per 1000 events), direct confuser captures, fixed
     close-encounter windows, isolated robustness, temporal detector jitter,
     and fixed-window jitter robustness.
   - Tracker-step FPS versus number of input detections.

All direct success denominators are tracker independent. Events are defined
only from GT and the shared detector observations. A tracker that fails to
establish or preserve a correct trajectory counts as a failure rather than
being removed from its test population.

Matching for the direct profiles
--------------------------------
Tracker outputs are matched to GT independently at each frame using a one-to-
one Hungarian assignment that maximizes the raw TrackEval 3D-box similarity.
Only pairs with similarity >= --success-iou-thr are accepted. Identity
continuation is then evaluated from the original tracker IDs across frames.
This deliberately avoids importing full-sequence HOTA association context into
these direct success profiles.

Incremental cache design
------------------------
common/sequences/<seq>.npz
    Tracker-independent GT/detector observations and event definitions.
profiles/<tracker>/<seq>.csv.gz
    Per-sequence aggregate counts for all direct success/failure profiles.
hota_events/<tracker>/<seq>.npz
    Cached full-sequence HOTA matching/association sufficient statistics.
    These are generated from current tracker outputs and used for both
    per-sequence and globally combined HOTA/decomposition tables.

Adding trackers computes only their missing profile and HOTA-event caches.
Use --recompute-trackers to refresh selected tracker outputs. Common event
caches are reusable as long as their metadata signature is unchanged.

Runtime is intentionally separate: FPS is loaded only from ``frame_stats`` for
trackers listed in ``--runtime-plot-trackers``. Missing timing data merely
omits that tracker from runtime-dependent plots; it never suppresses HOTA or
the direct capability profiles. No previously exported ``results/test``
summary CSV is read.

The script should be placed next to hota_cache.py, whose
JRDB/TrackEval loading and detector-observation helpers are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SCRIPT_VERSION = "tracker_eval_protocol"
CACHE_VERSION = "tracker_eval_cache"

TRACKER_LABELS = {
    "pedreftrack__global_gt_assisted": "GT-assisted PedRefTrack",
    "pedreftrack__global_no_gt": "PedRefTrack (no GT)",
    "fastpoly__global": "Fast-Poly",
    "simpletrack__global": "SimpleTrack",
    "elptnet__global": "ELPTNet (box)",
    "gnnpmb__global": "GNN-PMB",
    "cbmot__global": "CBMOT",
    "ab3dmot__global": "AB3DMOT",
}

TRACKER_FAMILY_MARKERS = {
    "ab3dmot": "o",
    "cbmot": "s",
    "elptnet": "D",
    "fastpoly": "^",
    "gnnpmb": "v",
    "pedreftrack": "X",
    "simpletrack": "P",
}

VARIANT_LINESTYLES: list[Any] = [
    "--",
    ":",
    "-.",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
]


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_float_edges(value: str) -> list[float]:
    edges: list[float] = []
    for token in parse_csv_list(value):
        token_l = token.lower()
        if token_l in {"inf", "+inf", "infinity", "+infinity"}:
            edges.append(float("inf"))
        elif token_l in {"-inf", "-infinity"}:
            edges.append(float("-inf"))
        else:
            edges.append(float(token))
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"Invalid strictly increasing bin edges: {value}")
    return edges


def parse_mapping(value: str | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"Mapping item must be key=value, got: {item}")
        key, mapped = item.split("=", 1)
        key, mapped = key.strip(), mapped.strip()
        if not key or not mapped:
            raise ValueError(f"Invalid mapping item: {item}")
        mapping[key] = mapped
    return mapping


def stable_signature(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def safe_tracker_name(name: str) -> str:
    return str(name).replace("/", "__").replace("\\", "__").replace(":", "_")


def tracker_label(name: str) -> str:
    return TRACKER_LABELS.get(name, name.replace("__global", "").replace("_", " "))


def tracker_family(name: str) -> str:
    return str(name).split("__", 1)[0].split("_", 1)[0]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def atomic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)


def load_helper(script_path: Path):
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing HOTA cache helper script: {script_path}")
    module_name = f"jrdb_hota_cache_helper_{hashlib.md5(str(script_path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def finite_quantiles(values: Iterable[float], q_lo: float, q_hi: float) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(array, q_lo)), float(np.quantile(array, q_hi))


def weighted_quantile(
    values: Iterable[float],
    weights: Iterable[float],
    quantile: float,
) -> float:
    """Return a deterministic left-continuous weighted empirical quantile."""
    value_array = np.asarray(list(values), dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (weight_array > 0)
    )
    if not np.any(valid):
        return float("nan")
    value_array = value_array[valid]
    weight_array = weight_array[valid]
    order = np.argsort(value_array, kind="mergesort")
    value_array = value_array[order]
    weight_array = weight_array[order]
    cumulative = np.cumsum(weight_array)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(value_array[min(index, len(value_array) - 1)])


def weighted_quantiles(
    values: Iterable[float],
    weights: Iterable[float],
    q_lo: float,
    q_hi: float,
) -> tuple[float, float]:
    values_list = list(values)
    weights_list = list(weights)
    return (
        weighted_quantile(values_list, weights_list, q_lo),
        weighted_quantile(values_list, weights_list, q_hi),
    )


def parse_int_edges(value: str) -> list[int]:
    edges = [int(item) for item in parse_csv_list(value)]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"Invalid strictly increasing integer bin edges: {value}")
    return edges


def cluster_bootstrap_interval(
    sequence_counts: pd.DataFrame,
    *,
    replicates: int,
    q_lo: float,
    q_hi: float,
    seed: int,
) -> tuple[float, float]:
    """Sequence-cluster bootstrap interval for a ratio of summed counts."""
    if sequence_counts.empty:
        return float("nan"), float("nan")
    eligible = pd.to_numeric(
        sequence_counts["n_eligible"], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    success = pd.to_numeric(
        sequence_counts["n_success"], errors="coerce"
    ).fillna(0).to_numpy(dtype=float)
    if eligible.sum() <= 0:
        return float("nan"), float("nan")
    if replicates <= 0 or len(eligible) <= 1:
        rate = float(success.sum() / eligible.sum())
        return rate, rate
    rng = np.random.default_rng(int(seed))
    sample_indices = rng.integers(
        0, len(eligible), size=(int(replicates), len(eligible))
    )
    denominator = eligible[sample_indices].sum(axis=1)
    numerator = success[sample_indices].sum(axis=1)
    rates = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator > 0,
    )
    rates = rates[np.isfinite(rates)]
    if rates.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(rates, q_lo)),
        float(np.quantile(rates, q_hi)),
    )


def deterministic_seed(base_seed: int, *parts: Any) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int((int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32))


def format_bin(left: float, right: float, closed_right: bool = False) -> str:
    left_text = "−∞" if math.isinf(left) and left < 0 else f"{left:g}"
    right_text = "∞" if math.isinf(right) else f"{right:g}"
    return f"[{left_text}, {right_text}{']' if closed_right else ')'}"


def assign_bins(values: np.ndarray, edges: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int16)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result
    idx = np.searchsorted(np.asarray(edges, dtype=float), values[finite], side="right") - 1
    idx[values[finite] == edges[-1]] = len(edges) - 2
    valid = (idx >= 0) & (idx < len(edges) - 1)
    out = np.full(idx.shape, -1, dtype=np.int16)
    out[valid] = idx[valid].astype(np.int16)
    result[finite] = out
    return result


def build_tracker_styles(trackers: list[str]) -> dict[str, dict[str, Any]]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    families: list[str] = []
    for tracker in trackers:
        family = tracker_family(tracker)
        if family not in families:
            families.append(family)
    family_colors = {
        family: (cycle[index % len(cycle)] if cycle else None)
        for index, family in enumerate(families)
    }
    styles: dict[str, dict[str, Any]] = {}
    for family in families:
        members = [tracker for tracker in trackers if tracker_family(tracker) == family]
        canonical = next((tracker for tracker in members if tracker == f"{family}__global"), members[0])
        variants = sorted(tracker for tracker in members if tracker != canonical)
        styles[canonical] = {
            "color": family_colors[family],
            "marker": TRACKER_FAMILY_MARKERS.get(family, "o"),
            "linestyle": "-",
        }
        for index, tracker in enumerate(variants):
            styles[tracker] = {
                "color": family_colors[family],
                "marker": TRACKER_FAMILY_MARKERS.get(family, "o"),
                "linestyle": VARIANT_LINESTYLES[index % len(VARIANT_LINESTYLES)],
            }
    return styles


# -----------------------------------------------------------------------------
# Common tracker-independent event cache
# -----------------------------------------------------------------------------

def _common_sequence_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    destination = Path(payload["common_dir"]) / "sequences" / f"{payload['seq']}.npz"
    if destination.exists() and not payload["force"]:
        return {"seq": payload["seq"], "status": "cached"}

    helper = load_helper(Path(payload["helper_script"]))
    helper.add_trackeval_to_path(Path(payload["trackeval_root"]))
    import trackeval  # noqa: WPS433

    seq = str(payload["seq"])
    dataset = helper.make_dataset(
        trackeval,
        Path(payload["trackers_base_dir"]),
        Path(payload["gt_folder"]),
        str(payload["anchor_tracker"]),
        str(payload["split_to_eval"]),
        str(payload["tracker_sub_folder"]),
        float(payload["matchable_sim_thr"]),
    )
    data = helper.load_preprocessed_sequence(dataset, str(payload["anchor_tracker"]), seq)
    n_t = int(data["num_timesteps"])
    n_gt_ids = int(data["num_gt_ids"])
    orig_map = np.asarray(data.get("gt_orig_ids", np.arange(n_gt_ids)), dtype=np.int64)

    frame_offsets = np.zeros(n_t + 1, dtype=np.int64)
    for frame in range(n_t):
        frame_offsets[frame + 1] = frame_offsets[frame] + len(data["gt_ids"][frame])
    total_gt = int(frame_offsets[-1])
    gt_internal_flat = np.empty(total_gt, dtype=np.int32)
    gt_orig_flat = np.empty(total_gt, dtype=np.int64)
    frame_flat = np.empty(total_gt, dtype=np.int32)
    gt_boxes_flat = np.empty((total_gt, 7), dtype=np.float32)
    frame_to_flat: list[dict[int, int]] = []

    for frame in range(n_t):
        start, end = int(frame_offsets[frame]), int(frame_offsets[frame + 1])
        ids = np.asarray(data["gt_ids"][frame], dtype=int)
        boxes = helper.trackeval_xyzwhd_to_internal_boxes(data["gt_dets_3d"][frame])
        mapping: dict[int, int] = {}
        for local, gid in enumerate(ids.tolist()):
            flat = start + local
            mapping[int(gid)] = flat
            gt_internal_flat[flat] = int(gid)
            gt_orig_flat[flat] = int(orig_map[int(gid)])
            frame_flat[flat] = frame
            gt_boxes_flat[flat] = boxes[local]
        frame_to_flat.append(mapping)

    # Reconstruct the one-to-one local GT/detector pairing so that
    # the raw matched detector centre is available for detector-centre NN,
    # temporal jitter, isolation, and direct-confuser event definitions. The
    # HOTA cache cache can still provide the already verified bounded-gap IDs.
    local_gt_folder = Path(payload["local_gt_folder"])
    if (local_gt_folder / "label_02").is_dir():
        local_gt_folder = local_gt_folder / "label_02"
    local_gt = helper.load_local_gt_trackeval_txt(
        local_gt_folder / f"{seq}.txt"
    )
    detections = helper.load_local_detections_json(
        Path(payload["detections_dir"]) / f"{seq}.json",
        score_min=payload.get("det_score_min"),
    )
    detector_box_flat = np.full((total_gt, 7), np.nan, dtype=np.float32)
    local_gt_box_flat = np.full((total_gt, 7), np.nan, dtype=np.float32)
    for frame in range(n_t):
        local_gt_map = local_gt.get(frame, {})
        local_ids = np.asarray(sorted(local_gt_map), dtype=np.int64)
        local_boxes = (
            np.asarray(
                [local_gt_map[int(gid)] for gid in local_ids], dtype=float
            ).reshape(-1, 7)
            if len(local_ids)
            else np.empty((0, 7), dtype=float)
        )
        det_boxes = np.asarray(
            detections.get(frame, np.empty((0, 7))), dtype=float
        ).reshape(-1, 7)
        rows, cols, _ = helper.gated_hungarian_center_matches(
            local_boxes[:, :2],
            det_boxes[:, :2],
            max_distance_m=float(payload["det_match_max_dist_m"]),
        )
        matched_by_orig = {
            int(local_ids[row]): np.asarray(det_boxes[col], dtype=float)
            for row, col in zip(rows.tolist(), cols.tolist())
        }
        start, end = int(frame_offsets[frame]), int(frame_offsets[frame + 1])
        for local, internal_id in enumerate(gt_internal_flat[start:end]):
            original_id = int(orig_map[int(internal_id)])
            local_gt_box = local_gt_map.get(original_id)
            if local_gt_box is None:
                raise RuntimeError(
                    "Could not align global GT occurrence to local GT for "
                    f"{seq}, frame={frame}, original_id={original_id}"
                )
            local_gt_box_flat[start + local] = np.asarray(
                local_gt_box, dtype=np.float32
            )
            matched_box = matched_by_orig.get(original_id)
            if matched_box is not None:
                detector_box_flat[start + local] = matched_box.astype(
                    np.float32
                )

    detector_present_computed = np.all(
        np.isfinite(detector_box_flat), axis=1
    )
    center_error_flat_computed = np.full(total_gt, np.nan, dtype=float)
    detector_error_xy_global_flat = np.full(
        (total_gt, 2), np.nan, dtype=np.float32
    )
    local_error_xy = (
        detector_box_flat[detector_present_computed, :2].astype(float)
        - local_gt_box_flat[
            detector_present_computed, :2
        ].astype(float)
    )
    yaw_delta = (
        gt_boxes_flat[detector_present_computed, 6].astype(float)
        - local_gt_box_flat[
            detector_present_computed, 6
        ].astype(float)
    )
    cos_yaw = np.cos(yaw_delta)
    sin_yaw = np.sin(yaw_delta)
    global_error_xy = np.column_stack([
        cos_yaw * local_error_xy[:, 0]
        - sin_yaw * local_error_xy[:, 1],
        sin_yaw * local_error_xy[:, 0]
        + cos_yaw * local_error_xy[:, 1],
    ])
    detector_error_xy_global_flat[detector_present_computed] = (
        global_error_xy.astype(np.float32)
    )
    center_error_flat_computed[detector_present_computed] = (
        np.linalg.norm(global_error_xy, axis=1)
    )

    reuse_cached = payload.get("reuse_hota_cache")
    cached_path = (
        Path(reuse_cached) / "frame_proxies" / f"{seq}.npz"
        if reuse_cached
        else None
    )
    if cached_path is not None and cached_path.exists():
        cached = load_npz(cached_path)
        if not (
            np.array_equal(cached["frame_gt_offsets"], frame_offsets)
            and np.array_equal(cached["gt_orig_ids_flat"], gt_orig_flat)
        ):
            raise RuntimeError(f"HOTA cache/common GT mismatch for {seq}")
        cached_present = np.asarray(
            cached["detector_present_flat"], dtype=bool
        )
        if not np.array_equal(cached_present, detector_present_computed):
            raise RuntimeError(
                f"HOTA cache/raw detector matching mismatch for {seq}. "
                "Use a compatible --reuse-hota-cache or omit it."
            )
        detector_present = detector_present_computed
        gap_id_flat = np.asarray(
            cached["detector_gap_id_flat"], dtype=np.int64
        )
        center_error_flat = center_error_flat_computed
        source = "reused_hota_cache_plus_raw_detector_boxes"
    else:
        detector_features, _ = helper.compute_detector_observation_features(
            data,
            local_gt_by_frame=local_gt,
            detections_by_frame=detections,
            det_match_max_dist_m=float(payload["det_match_max_dist_m"]),
        )
        spatial_features = helper.compute_spatial_context_features(data)
        proxy_payload, _, _ = helper.build_frame_proxy_cache(
            data,
            detector_features,
            spatial_features,
            fps=float(payload["fps"]),
            gap_length_definition="missing_frames",
        )
        detector_present = detector_present_computed
        proxy_present = np.asarray(
            proxy_payload["detector_present_flat"], dtype=bool
        )
        if not np.array_equal(proxy_present, detector_present):
            raise RuntimeError(
                f"Detector feature/raw box matching mismatch for {seq}"
            )
        gap_id_flat = np.asarray(
            proxy_payload["detector_gap_id_flat"], dtype=np.int64
        )
        center_error_flat = center_error_flat_computed
        source = "computed_with_raw_detector_boxes"

    # Nearest-neighbour difficulty is detector centre-to-centre distance, not
    # GT centre distance. Only pedestrians with matched shared detections enter.
    detected_nn_flat = np.full(total_gt, np.nan, dtype=np.float32)
    detected_nn_gt_internal_flat = np.full(total_gt, -1, dtype=np.int32)
    for frame in range(n_t):
        start, end = int(frame_offsets[frame]), int(frame_offsets[frame + 1])
        present_local = np.flatnonzero(detector_present[start:end])
        if present_local.size <= 1:
            continue
        xy = np.asarray(
            detector_box_flat[start:end, :2], dtype=float
        )[present_local]
        distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest_local_index = np.argmin(distances, axis=1)
        nearest = distances[
            np.arange(len(present_local)), nearest_local_index
        ]
        detected_nn_flat[start + present_local] = nearest.astype(np.float32)
        neighbor_occurrence_local = present_local[nearest_local_index]
        detected_nn_gt_internal_flat[start + present_local] = (
            gt_internal_flat[start + neighbor_occurrence_local]
        )

    # Consecutive visible detector history ending at every GT occurrence.
    visible_run_flat = np.zeros(total_gt, dtype=np.int16)
    for gid in range(n_gt_ids):
        run = 0
        for frame in range(n_t):
            flat = frame_to_flat[frame].get(gid)
            if flat is None:
                run = 0
                continue
            if detector_present[flat]:
                run += 1
            else:
                run = 0
            visible_run_flat[flat] = min(run, np.iinfo(np.int16).max)

    atomic_npz(destination, {
        "num_timesteps": np.asarray([n_t], dtype=np.int64),
        "frame_gt_offsets": frame_offsets,
        "gt_internal_ids_flat": gt_internal_flat,
        "gt_orig_ids_flat": gt_orig_flat,
        "frame_indices_flat": frame_flat,
        "gt_boxes_internal_flat": gt_boxes_flat,
        "local_gt_boxes_internal_flat": local_gt_box_flat,
        "detector_present_flat": detector_present,
        "detector_gap_id_flat": gap_id_flat,
        "detector_center_error_m_flat": center_error_flat.astype(np.float32),
        "detector_center_error_xy_global_flat": (
            detector_error_xy_global_flat
        ),
        "detector_box_internal_flat": detector_box_flat,
        "detected_nn_distance_m_flat": detected_nn_flat,
        "detected_nn_gt_internal_id_flat": detected_nn_gt_internal_flat,
        "detector_visible_run_flat": visible_run_flat,
    })
    return {"seq": seq, "status": source, "gt_occurrences": total_gt}


# -----------------------------------------------------------------------------
# Local direct-profile matching and tracker cache
# -----------------------------------------------------------------------------

def local_gt_to_tracker_assignments(data: dict, threshold: float) -> list[dict[int, int]]:
    """Per-frame GT-internal-id -> original tracker-id using raw similarity."""
    n_t = int(data["num_timesteps"])
    tracker_orig = np.asarray(
        data.get("tracker_orig_ids", np.arange(int(data["num_tracker_ids"]))),
        dtype=np.int64,
    )
    assignments: list[dict[int, int]] = []
    eps = np.finfo(float).eps
    for frame in range(n_t):
        gt_ids = np.asarray(data["gt_ids"][frame], dtype=int)
        tr_ids = np.asarray(data["tracker_ids"][frame], dtype=int)
        mapping: dict[int, int] = {}
        if len(gt_ids) and len(tr_ids):
            similarity = np.asarray(data["similarity_scores"][frame], dtype=float)
            score = similarity.copy()
            score[score < threshold - eps] = -1e9
            rows, cols = linear_sum_assignment(-score)
            for row, col in zip(rows.tolist(), cols.tolist()):
                if similarity[row, col] >= threshold - eps:
                    mapping[int(gt_ids[row])] = int(tracker_orig[int(tr_ids[col])])
        assignments.append(mapping)
    return assignments


def ensure_hota_event(
    helper,
    payload: dict[str, Any],
    tracker: str,
    seq: str,
    destination: Path,
) -> str:
    if destination.exists() and not payload["force_tracker"]:
        return "cached"
    reuse_hota = payload.get("reuse_hota_events_from")
    if reuse_hota:
        reused = helper.reuse_event_file(
            Path(reuse_hota),
            tracker,
            seq,
            destination,
            force=bool(payload["force_tracker"]),
        )
        if reused is not None:
            return str(reused["status"])
    result = helper.precompute_hota_event_sequence(
        trackeval_root=Path(payload["trackeval_root"]),
        trackers_base_dir=Path(payload["trackers_base_dir"]),
        gt_folder=Path(payload["gt_folder"]),
        tracker=tracker,
        seq=seq,
        split_to_eval=str(payload["split_to_eval"]),
        tracker_sub_folder=str(payload["tracker_sub_folder"]),
        matchable_sim_thr=float(payload["matchable_sim_thr"]),
        output_path=destination,
        force=bool(payload["force_tracker"]),
    )
    return str(result["status"])

def pair_association_weights(event: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.asarray(event["pair_match_counts"], dtype=float)
    n_alpha = len(event["alpha_labels"])
    if counts.size == 0:
        empty = np.empty((0, n_alpha), dtype=float)
        return empty, empty, empty
    pair_gt = np.asarray(event["pair_gt_internal_ids"], dtype=int)
    pair_tr = np.asarray(event["pair_tracker_internal_ids"], dtype=int)
    gt_counts = np.asarray(event["gt_id_counts"], dtype=float)[pair_gt, None]
    tr_counts = np.asarray(event["tracker_id_counts"], dtype=float)[pair_tr, None]
    ass_a = counts / np.maximum(1.0, gt_counts + tr_counts - counts)
    ass_re = counts / np.maximum(1.0, gt_counts)
    ass_pr = counts / np.maximum(1.0, tr_counts)
    return ass_a, ass_re, ass_pr

def scalar_hota(
    tp: np.ndarray,
    fn: np.ndarray,
    fp: np.ndarray,
    loc_sum: np.ndarray,
    ass_a_sum: np.ndarray,
    ass_re_sum: np.ndarray,
    ass_pr_sum: np.ndarray,
) -> dict[str, float]:
    tp = np.asarray(tp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    fp = np.asarray(fp, dtype=float)
    ass_a = ass_a_sum / np.maximum(1.0, tp)
    ass_re = ass_re_sum / np.maximum(1.0, tp)
    ass_pr = ass_pr_sum / np.maximum(1.0, tp)
    loca = np.maximum(1e-10, loc_sum) / np.maximum(1e-10, tp)
    det_re = tp / np.maximum(1.0, tp + fn)
    det_pr = tp / np.maximum(1.0, tp + fp)
    det_a = tp / np.maximum(1.0, tp + fn + fp)
    hota = np.sqrt(det_a * ass_a)
    return {
        "HOTA": 100.0 * float(np.mean(hota)),
        "DetA": 100.0 * float(np.mean(det_a)),
        "DetRe": 100.0 * float(np.mean(det_re)),
        "DetPr": 100.0 * float(np.mean(det_pr)),
        "AssA": 100.0 * float(np.mean(ass_a)),
        "AssRe": 100.0 * float(np.mean(ass_re)),
        "AssPr": 100.0 * float(np.mean(ass_pr)),
        "LocA": 100.0 * float(np.mean(loca)),
        "TP_mean": float(np.mean(tp)),
        "FN_mean": float(np.mean(fn)),
        "FP_mean": float(np.mean(fp)),
    }

def aggregate_global_hota(event_paths: list[Path]) -> dict[str, float]:
    alpha_labels = None
    tp = fn = fp = loc = ass_a_sum = ass_re_sum = ass_pr_sum = None
    for path in event_paths:
        event = load_npz(path)
        alpha = np.asarray(event["alpha_labels"], dtype=float)
        if alpha_labels is None:
            alpha_labels = alpha
            shape = len(alpha)
            tp = np.zeros(shape, dtype=float)
            fn = np.zeros(shape, dtype=float)
            fp = np.zeros(shape, dtype=float)
            loc = np.zeros(shape, dtype=float)
            ass_a_sum = np.zeros(shape, dtype=float)
            ass_re_sum = np.zeros(shape, dtype=float)
            ass_pr_sum = np.zeros(shape, dtype=float)
        pair_a, pair_re, pair_pr = pair_association_weights(event)
        similarity = np.asarray(event["assigned_similarity"], dtype=float)
        pair_index = np.asarray(event["assigned_pair_index"], dtype=int)
        n_gt = len(event["gt_orig_ids_flat"])
        n_tr = len(event["tracker_orig_ids_flat"])
        for a, threshold in enumerate(alpha):
            accepted = (pair_index >= 0) & (similarity >= threshold - np.finfo(float).eps)
            indices = np.flatnonzero(accepted)
            count = float(len(indices))
            tp[a] += count
            fn[a] += n_gt - count
            fp[a] += n_tr - count
            if len(indices):
                loc[a] += float(np.sum(similarity[indices]))
                p = pair_index[indices]
                ass_a_sum[a] += float(np.sum(pair_a[p, a]))
                ass_re_sum[a] += float(np.sum(pair_re[p, a]))
                ass_pr_sum[a] += float(np.sum(pair_pr[p, a]))
    if alpha_labels is None:
        return {key: float("nan") for key in ("HOTA", "DetA", "DetRe", "DetPr", "AssA", "AssRe", "AssPr", "LocA")}
    return scalar_hota(tp, fn, fp, loc, ass_a_sum, ass_re_sum, ass_pr_sum)

def _tracker_sequence_worker(payload: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    tracker = str(payload["tracker"])
    seq = str(payload["seq"])
    tracker_safe = safe_tracker_name(tracker)
    destination = Path(payload["profile_dir"]) / tracker_safe / f"{seq}.csv.gz"
    event_path = Path(payload["hota_event_dir"]) / tracker_safe / f"{seq}.npz"

    helper = load_helper(Path(payload["helper_script"]))
    helper.add_trackeval_to_path(Path(payload["trackeval_root"]))
    import trackeval  # noqa: WPS433

    event_status = ensure_hota_event(helper, payload, tracker, seq, event_path)
    if destination.exists() and not payload["force_tracker"]:
        return {
            "tracker": tracker,
            "seq": seq,
            "status": "cached",
            "hota_event_status": event_status,
        }

    dataset = helper.make_dataset(
        trackeval,
        Path(payload["trackers_base_dir"]),
        Path(payload["gt_folder"]),
        tracker,
        str(payload["split_to_eval"]),
        str(payload["tracker_sub_folder"]),
        float(payload["matchable_sim_thr"]),
    )
    data = helper.load_preprocessed_sequence(dataset, tracker, seq)
    common = load_npz(Path(payload["common_dir"]) / "sequences" / f"{seq}.npz")

    n_t = int(data["num_timesteps"])
    frame_offsets = np.asarray(common["frame_gt_offsets"], dtype=np.int64)
    gt_ids_flat = np.asarray(common["gt_internal_ids_flat"], dtype=np.int32)
    frame_flat = np.asarray(common["frame_indices_flat"], dtype=np.int32)
    detector_present = np.asarray(common["detector_present_flat"], dtype=bool)
    gap_id_flat = np.asarray(common["detector_gap_id_flat"], dtype=np.int64)
    center_error_flat = np.asarray(
        common["detector_center_error_m_flat"], dtype=float
    )
    detector_error_xy = np.asarray(
        common["detector_center_error_xy_global_flat"], dtype=float
    )
    detected_nn_flat = np.asarray(common["detected_nn_distance_m_flat"], dtype=float)
    detected_nn_gid_flat = np.asarray(
        common["detected_nn_gt_internal_id_flat"], dtype=np.int32
    )
    visible_run_flat = np.asarray(common["detector_visible_run_flat"], dtype=np.int16)

    expected_offsets = np.asarray(
        np.concatenate([[0], np.cumsum([len(x) for x in data["gt_ids"]])]),
        dtype=np.int64,
    )
    if not np.array_equal(frame_offsets, expected_offsets):
        raise RuntimeError(f"Tracker/common frame offsets differ for {tracker}:{seq}")

    frame_to_flat: list[dict[int, int]] = []
    for frame in range(n_t):
        start, end = int(frame_offsets[frame]), int(frame_offsets[frame + 1])
        frame_to_flat.append({int(gid): start + local for local, gid in enumerate(gt_ids_flat[start:end])})

    gt_to_track = local_gt_to_tracker_assignments(data, float(payload["success_iou_thr"]))
    max_gap_age_frames = int(payload["max_gap_age_frames"])
    pre_gap_visible_frames = int(payload["pre_gap_visible_frames"])
    nn_edges = list(payload["nn_edges"])
    center_edges = list(payload["center_edges"])
    jitter_edges = list(payload["jitter_edges"])
    recovery_edges = list(payload["recovery_edges_frames"])
    window_frames = int(payload["encounter_window_frames"])
    isolated_nn_thr_m = float(payload["isolated_nn_thr_m"])

    profile_specs: dict[str, list[float]] = {
        "gap": list(range(max_gap_age_frames + 1)),
        "recovery": recovery_edges,
        "nn": nn_edges,
        "nn_failure": nn_edges,
        "nn_confuser_capture": nn_edges,
        "nn_failure_not_established": nn_edges,
        "nn_failure_current_miss": nn_edges,
        "nn_failure_id_change": nn_edges,
        "center_error": center_edges,
        "center_failure": center_edges,
        "center_isolated_failure": center_edges,
        "jitter_failure": jitter_edges,
        "nn_window_failure": nn_edges,
        "jitter_window_failure": jitter_edges,
    }
    counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for profile, edges in profile_specs.items():
        n_bins = (
            max_gap_age_frames
            if profile == "gap"
            else len(edges) - 1
        )
        counts[profile] = (
            np.zeros(n_bins, dtype=np.int64),
            np.zeros(n_bins, dtype=np.int64),
        )

    def add_event(profile: str, bin_index: int, outcome: bool) -> None:
        eligible, success = counts[profile]
        if 0 <= int(bin_index) < len(eligible):
            eligible[int(bin_index)] += 1
            success[int(bin_index)] += int(bool(outcome))

    # Main gap curve: end-to-end continuous same-ID output at every missing
    # age. A target not correctly established before the gap is a failure.
    # Recovery: same pre-gap ID at the first detector-supported frame after the
    # bounded gap; prediction through the missing frames is not required.
    for gap_id in np.unique(gap_id_flat[gap_id_flat >= 0]).tolist():
        flats = np.flatnonzero(gap_id_flat == int(gap_id))
        flats = flats[np.argsort(frame_flat[flats])]
        if flats.size == 0:
            continue
        gid = int(gt_ids_flat[flats[0]])
        first_missing = int(frame_flat[flats[0]])
        pre_frame = first_missing - 1
        if pre_frame < 0:
            continue
        pre_flat = frame_to_flat[pre_frame].get(gid)
        if pre_flat is None or int(visible_run_flat[pre_flat]) < pre_gap_visible_frames:
            continue

        pre_track = gt_to_track[pre_frame].get(gid)
        continuously_correct = pre_track is not None
        max_age = min(int(flats.size), max_gap_age_frames)
        for age_frame in range(1, max_age + 1):
            frame = pre_frame + age_frame
            current_track = gt_to_track[frame].get(gid)
            continuously_correct = bool(
                continuously_correct
                and pre_track is not None
                and current_track == pre_track
            )
            add_event("gap", age_frame - 1, continuously_correct)

        gap_length = int(flats.size)
        post_frame = first_missing + gap_length
        post_flat = (
            frame_to_flat[post_frame].get(gid)
            if post_frame < n_t
            else None
        )
        recovery_bin = (
            int(
                np.searchsorted(
                    np.asarray(recovery_edges, dtype=int),
                    gap_length,
                    side="right",
                )
                - 1
            )
            if recovery_edges[0] <= gap_length < recovery_edges[-1]
            else -1
        )
        if (
            recovery_bin >= 0
            and post_flat is not None
            and detector_present[post_flat]
        ):
            post_track = gt_to_track[post_frame].get(gid)
            add_event(
                "recovery",
                recovery_bin,
                pre_track is not None and post_track == pre_track,
            )

    # Adjacent detector-supported target updates. Every event is common across
    # trackers; failure includes no correct pre-track, a missing current output,
    # or a changed ID. Direct confuser capture is a stricter diagnostic:
    # either the target's previous ID moves to its current nearest neighbour or
    # the neighbour's previous ID moves onto the target.
    for frame in range(1, n_t):
        start, end = int(frame_offsets[frame]), int(frame_offsets[frame + 1])
        for flat in range(start, end):
            gid = int(gt_ids_flat[flat])
            prev_flat = frame_to_flat[frame - 1].get(gid)
            if prev_flat is None:
                continue
            if not (detector_present[prev_flat] and detector_present[flat]):
                continue

            previous_track = gt_to_track[frame - 1].get(gid)
            current_track = gt_to_track[frame].get(gid)
            identity_success = bool(
                previous_track is not None
                and current_track is not None
                and previous_track == current_track
            )

            nn_distance = float(detected_nn_flat[flat])
            nn_bin = int(assign_bins(np.asarray([nn_distance]), nn_edges)[0])
            if nn_bin >= 0:
                neighbor_gid = int(detected_nn_gid_flat[flat])
                neighbor_previous_track = (
                    gt_to_track[frame - 1].get(neighbor_gid)
                    if neighbor_gid >= 0
                    else None
                )
                neighbor_current_track = (
                    gt_to_track[frame].get(neighbor_gid)
                    if neighbor_gid >= 0
                    else None
                )
                confuser_capture = bool(
                    previous_track is not None
                    and (
                        neighbor_current_track == previous_track
                        or (
                            neighbor_previous_track is not None
                            and current_track == neighbor_previous_track
                        )
                    )
                )
                not_established = previous_track is None
                current_miss = bool(
                    previous_track is not None
                    and current_track is None
                    and not confuser_capture
                )
                id_change = bool(
                    previous_track is not None
                    and current_track is not None
                    and current_track != previous_track
                    and not confuser_capture
                )
                add_event("nn", nn_bin, identity_success)
                add_event("nn_failure", nn_bin, not identity_success)
                add_event(
                    "nn_confuser_capture", nn_bin, confuser_capture
                )
                add_event(
                    "nn_failure_not_established",
                    nn_bin,
                    not_established,
                )
                add_event(
                    "nn_failure_current_miss", nn_bin, current_miss
                )
                add_event("nn_failure_id_change", nn_bin, id_change)

            center_error = float(center_error_flat[flat])
            center_bin = int(assign_bins(np.asarray([center_error]), center_edges)[0])
            if center_bin >= 0:
                add_event("center_error", center_bin, identity_success)
                add_event(
                    "center_failure", center_bin, not identity_success
                )
                prev_nn = float(detected_nn_flat[prev_flat])
                isolated = (
                    (not np.isfinite(prev_nn) or prev_nn >= isolated_nn_thr_m)
                    and (
                        not np.isfinite(nn_distance)
                        or nn_distance >= isolated_nn_thr_m
                    )
                )
                if isolated:
                    add_event(
                        "center_isolated_failure",
                        center_bin,
                        not identity_success,
                    )

            jitter = float(
                np.linalg.norm(
                    detector_error_xy[flat]
                    - detector_error_xy[prev_flat]
                )
            )
            jitter_bin = int(
                assign_bins(np.asarray([jitter]), jitter_edges)[0]
            )
            if jitter_bin >= 0:
                add_event(
                    "jitter_failure", jitter_bin, not identity_success
                )

    # Fixed non-overlapping windows avoid counting every overlapping slice of a
    # long easy run. The close-encounter window is binned by its minimum raw
    # detector NN separation; the jitter window by its P90 detector-error step.
    n_gt_ids = int(data["num_gt_ids"])
    for gid in range(n_gt_ids):
        visible_frames: list[int] = []

        def consume_visible_run(run: list[int]) -> None:
            if len(run) < window_frames:
                return
            for offset in range(0, len(run) - window_frames + 1, window_frames):
                window = run[offset : offset + window_frames]
                flats = [frame_to_flat[t][gid] for t in window]
                track_ids = [gt_to_track[t].get(gid) for t in window]
                same_id_success = bool(
                    track_ids[0] is not None
                    and all(track_id == track_ids[0] for track_id in track_ids)
                )

                nn_values = np.asarray(
                    [detected_nn_flat[flat] for flat in flats], dtype=float
                )
                finite_nn = nn_values[np.isfinite(nn_values)]
                if finite_nn.size:
                    min_nn = float(np.min(finite_nn))
                    nn_window_bin = int(
                        assign_bins(
                            np.asarray([min_nn]), nn_edges
                        )[0]
                    )
                    if nn_window_bin >= 0:
                        add_event(
                            "nn_window_failure",
                            nn_window_bin,
                            not same_id_success,
                        )

                jitter_values = np.asarray(
                    [
                        np.linalg.norm(
                            detector_error_xy[flats[index]]
                            - detector_error_xy[flats[index - 1]]
                        )
                        for index in range(1, len(flats))
                    ],
                    dtype=float,
                )
                if jitter_values.size and np.all(np.isfinite(jitter_values)):
                    p90_jitter = float(np.quantile(jitter_values, 0.90))
                    jitter_window_bin = int(
                        assign_bins(
                            np.asarray([p90_jitter]), jitter_edges
                        )[0]
                    )
                    if jitter_window_bin >= 0:
                        add_event(
                            "jitter_window_failure",
                            jitter_window_bin,
                            not same_id_success,
                        )

        for frame_index in range(n_t):
            flat = frame_to_flat[frame_index].get(gid)
            if flat is not None and detector_present[flat]:
                visible_frames.append(frame_index)
            else:
                consume_visible_run(visible_frames)
                visible_frames = []
        consume_visible_run(visible_frames)

    rows: list[dict[str, Any]] = []
    for profile, (eligible, success) in counts.items():
        if profile == "gap":
            edge_values = [
                (index + 1) / float(payload["fps"])
                for index in range(len(eligible))
            ]
            for index in range(len(eligible)):
                rows.append({
                    "profile": profile,
                    "bin_index": index,
                    "x_left": edge_values[index],
                    "x_right": edge_values[index],
                    "n_eligible": int(eligible[index]),
                    "n_success": int(success[index]),
                })
            continue
        edges = profile_specs[profile]
        for index in range(len(eligible)):
            rows.append({
                "profile": profile,
                "bin_index": index,
                "x_left": float(edges[index]),
                "x_right": float(edges[index + 1]),
                "n_eligible": int(eligible[index]),
                "n_success": int(success[index]),
            })

    profile_frame = pd.DataFrame(rows)
    profile_frame.insert(0, "seq", seq)
    profile_frame.insert(0, "tracker", tracker)
    atomic_csv_gz(destination, profile_frame)
    return {
        "tracker": tracker,
        "seq": seq,
        "status": "computed",
        "rows": len(profile_frame),
        "hota_event_status": event_status,
    }


# -----------------------------------------------------------------------------
# Global HOTA and runtime summaries
# -----------------------------------------------------------------------------

HOTA_METRIC_FIELDS = (
    "HOTA",
    "DetA",
    "DetRe",
    "DetPr",
    "AssA",
    "AssRe",
    "AssPr",
    "LocA",
    "TP_mean",
    "FN_mean",
    "FP_mean",
)


def build_hota_tables(
    event_dir: Path,
    trackers: list[str],
    sequences: list[str],
    reference: str,
    q_lo: float,
    q_hi: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate per-sequence and globally combined HOTA from cached events.

    The event caches are generated directly from the selected tracker outputs.
    ``aggregate_global_hota`` uses sufficient-statistic aggregation: TP/FN/FP
    and localization sums
    are accumulated over sequences, while association contributions retain
    their HOTA pair weights. The resulting global score is therefore not an
    arithmetic mean of sequence HOTA values.
    """
    per_sequence_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []

    for tracker in trackers:
        paths = [
            event_dir / safe_tracker_name(tracker) / f"{seq}.npz"
            for seq in sequences
        ]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing HOTA-event cache for {tracker}: {missing[0]}. "
                "Run without --plot-only, or include the tracker in "
                "--recompute-trackers."
            )

        for seq, path in zip(sequences, paths):
            metrics = aggregate_global_hota([path])
            event = load_npz(path)
            per_sequence_rows.append({
                "tracker": tracker,
                "label": tracker_label(tracker),
                "seq": seq,
                "gt_occurrences": int(
                    len(event["gt_orig_ids_flat"])
                ),
                **metrics,
            })

        combined = aggregate_global_hota(paths)
        global_rows.append({
            "tracker": tracker,
            "label": tracker_label(tracker),
            **combined,
        })

    per_sequence = pd.DataFrame(per_sequence_rows)
    global_metrics = pd.DataFrame(global_rows)

    summary_rows: list[dict[str, Any]] = []
    for tracker in trackers:
        seq_frame = per_sequence[per_sequence["tracker"] == tracker]
        qlo, qhi = weighted_quantiles(
            seq_frame["HOTA"],
            seq_frame["gt_occurrences"],
            q_lo,
            q_hi,
        )
        combined = global_metrics[global_metrics["tracker"] == tracker].iloc[0]
        row = {
            "tracker": tracker,
            "label": tracker_label(tracker),
            "hota_combined": float(combined["HOTA"]),
            "hota_seq_qlo": qlo,
            "hota_seq_qhi": qhi,
            "n_sequences": int(len(seq_frame)),
        }
        for field in HOTA_METRIC_FIELDS:
            row[field] = float(combined[field])
        summary_rows.append(row)
    hota_summary = pd.DataFrame(summary_rows)

    if reference not in set(hota_summary["tracker"]):
        raise ValueError(f"Reference tracker has no HOTA result: {reference}")
    reference_combined = float(
        hota_summary.loc[
            hota_summary["tracker"] == reference, "hota_combined"
        ].iloc[0]
    )
    reference_seq = per_sequence[
        per_sequence["tracker"] == reference
    ][["seq", "HOTA", "gt_occurrences"]].rename(
        columns={
            "HOTA": "hota_reference",
            "gt_occurrences": "gt_occurrences_reference",
        }
    )

    gap_rows: list[dict[str, Any]] = []
    for tracker in trackers:
        tracker_combined = float(
            hota_summary.loc[
                hota_summary["tracker"] == tracker, "hota_combined"
            ].iloc[0]
        )
        tracker_seq = per_sequence[
            per_sequence["tracker"] == tracker
        ][["seq", "HOTA", "gt_occurrences"]].rename(
            columns={"HOTA": "hota_tracker"}
        )
        paired = tracker_seq.merge(reference_seq, on="seq", how="inner")
        paired["gap_to_reference"] = (
            paired["hota_reference"] - paired["hota_tracker"]
        )
        if not np.array_equal(
            paired["gt_occurrences"].to_numpy(dtype=int),
            paired["gt_occurrences_reference"].to_numpy(dtype=int),
        ):
            raise RuntimeError(
                f"GT sequence weights differ for {tracker} and {reference}"
            )
        qlo, qhi = weighted_quantiles(
            paired["gap_to_reference"],
            paired["gt_occurrences"],
            q_lo,
            q_hi,
        )
        gap_rows.append({
            "tracker": tracker,
            "label": tracker_label(tracker),
            "hota_gap_combined": reference_combined - tracker_combined,
            "paired_seq_gap_qlo": qlo,
            "paired_seq_gap_qhi": qhi,
            "n_paired_sequences": int(len(paired)),
        })

    return hota_summary, pd.DataFrame(gap_rows), per_sequence


def global_gap_decomposition(global_metrics: pd.DataFrame, reference: str) -> pd.DataFrame:
    reference_rows = global_metrics[global_metrics["tracker"] == reference]
    if reference_rows.empty:
        return pd.DataFrame()
    ref = reference_rows.iloc[0]
    rows: list[dict[str, Any]] = []
    ref_approx = math.sqrt(max(float(ref.DetA), 0.0) * max(float(ref.AssA), 0.0))
    for row in global_metrics.itertuples(index=False):
        approx = math.sqrt(max(float(row.DetA), 0.0) * max(float(row.AssA), 0.0))
        replace_det = math.sqrt(max(float(ref.DetA), 0.0) * max(float(row.AssA), 0.0))
        replace_ass = math.sqrt(max(float(row.DetA), 0.0) * max(float(ref.AssA), 0.0))
        rows.append({
            "tracker": row.tracker,
            "reference": reference,
            "HOTA": float(row.HOTA),
            "HOTA_gap_to_reference": float(row.HOTA - ref.HOTA),
            "DetA_gap": float(row.DetA - ref.DetA),
            "AssA_gap": float(row.AssA - ref.AssA),
            "DetRe_gap": float(row.DetRe - ref.DetRe),
            "DetPr_gap": float(row.DetPr - ref.DetPr),
            "AssRe_gap": float(row.AssRe - ref.AssRe),
            "AssPr_gap": float(row.AssPr - ref.AssPr),
            "LocA_gap": float(row.LocA - ref.LocA),
            "sqrt_DetA_AssA": approx,
            "sqrt_reference_DetA_tracker_AssA": replace_det,
            "sqrt_tracker_DetA_reference_AssA": replace_ass,
            "approx_gain_if_DetA_replaced": replace_det - approx,
            "approx_gain_if_AssA_replaced": replace_ass - approx,
            "reference_sqrt_DetA_AssA": ref_approx,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def frame_stats_directory(
    trackers_base_dir: Path,
    tracker: str,
    source_map: dict[str, str],
    split_name: str,
) -> Path | None:
    # Use the exact tracker's timing unless the user explicitly maps a variant
    # to another timing source. Silent family fallback can misrepresent runtime.
    source = source_map.get(tracker, tracker)
    path = trackers_base_dir / source / split_name / "frame_stats"
    return path if path.is_dir() else None


def load_frame_stats(
    trackers_base_dir: Path,
    tracker: str,
    source_map: dict[str, str],
    split_name: str,
) -> pd.DataFrame:
    directory = frame_stats_directory(
        trackers_base_dir, tracker, source_map, split_name
    )
    if directory is None:
        return pd.DataFrame(columns=["step_ms", "fps", "num_det_in"])
    chunks: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.columns = [str(column).strip() for column in frame.columns]
        step_ms = pd.to_numeric(frame.get("step_ms", np.nan), errors="coerce")
        num_det = pd.to_numeric(frame.get("num_det_in", np.nan), errors="coerce")
        valid = np.isfinite(step_ms) & (step_ms > 0)
        if not np.any(valid):
            continue
        chunks.append(pd.DataFrame({
            "step_ms": step_ms[valid].to_numpy(float),
            "fps": 1000.0 / step_ms[valid].to_numpy(float),
            "num_det_in": num_det[valid].to_numpy(float),
        }))
    if not chunks:
        return pd.DataFrame(columns=["step_ms", "fps", "num_det_in"])
    return pd.concat(chunks, ignore_index=True)


def load_runtime_tables(
    trackers_base_dir: Path,
    runtime_trackers: list[str],
    source_map: dict[str, str],
    split_name: str,
    q_lo: float,
    q_hi: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load only pre-recorded frame_stats for requested runtime trackers."""
    summary_rows: list[dict[str, Any]] = []
    runtime_frames: dict[str, pd.DataFrame] = {}
    for tracker in runtime_trackers:
        source = source_map.get(tracker, tracker)
        frame = load_frame_stats(
            trackers_base_dir, tracker, source_map, split_name
        )
        if frame.empty:
            print(
                f"WARNING: no usable frame_stats for {tracker} "
                f"(source={source}); omitted from runtime plots",
                file=sys.stderr,
            )
            continue
        runtime_frames[tracker] = frame
        qlo, qhi = finite_quantiles(frame["fps"], q_lo, q_hi)
        summary_rows.append({
            "tracker": tracker,
            "runtime_source_tracker": source,
            "fps_median": float(frame["fps"].median()),
            "fps_qlo": qlo,
            "fps_qhi": qhi,
            "n_runtime_frames": int(len(frame)),
        })
    return pd.DataFrame(summary_rows), runtime_frames


def merge_hota_runtime(
    hota_summary: pd.DataFrame,
    runtime_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join runtime onto HOTA; HOTA remains available without timing."""
    if runtime_summary.empty:
        result = hota_summary.copy()
        result["runtime_source_tracker"] = np.nan
        result["fps_median"] = np.nan
        result["fps_qlo"] = np.nan
        result["fps_qhi"] = np.nan
        result["n_runtime_frames"] = 0
        return result
    result = hota_summary.merge(runtime_summary, on="tracker", how="left")
    result["n_runtime_frames"] = (
        pd.to_numeric(result["n_runtime_frames"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return result


def aggregate_runtime_by_detection_count(
    runtime_frames: dict[str, pd.DataFrame],
    trackers: list[str],
    min_frames: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tracker in trackers:
        frame = runtime_frames.get(tracker, pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame = frame[
            np.isfinite(frame["num_det_in"]) & np.isfinite(frame["fps"])
        ].copy()
        frame["num_det_in"] = np.rint(frame["num_det_in"]).astype(int)
        for count, group in frame.groupby("num_det_in"):
            if len(group) < min_frames:
                continue
            rows.append({
                "tracker": tracker,
                "label": tracker_label(tracker),
                "num_det_in": int(count),
                "fps_median": float(group["fps"].median()),
                "fps_q10": float(group["fps"].quantile(0.10)),
                "fps_q90": float(group["fps"].quantile(0.90)),
                "n_frames": int(len(group)),
            })
    return pd.DataFrame(
        rows,
        columns=[
            "tracker",
            "label",
            "num_det_in",
            "fps_median",
            "fps_q10",
            "fps_q90",
            "n_frames",
        ],
    )


# -----------------------------------------------------------------------------
# Profile aggregation
# -----------------------------------------------------------------------------

def aggregate_profile_caches(
    profile_dir: Path,
    trackers: list[str],
    sequences: list[str],
    fps: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    q_lo: float,
    q_hi: float,
) -> pd.DataFrame:
    """Aggregate common-denominator profile counts with sequence bootstrap CIs."""
    aggregate_rows: list[dict[str, Any]] = []
    for tracker in trackers:
        paths = [profile_dir / safe_tracker_name(tracker) / f"{seq}.csv.gz" for seq in sequences]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing profile cache for {tracker}: {missing[0]}")
        frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        for (profile, bin_index), subset in frame.groupby(
            ["profile", "bin_index"], sort=True
        ):
            index = int(bin_index)
            left = float(subset["x_left"].iloc[0])
            right = float(subset["x_right"].iloc[0])
            eligible = int(subset["n_eligible"].sum())
            success = int(subset["n_success"].sum())
            rate = success / eligible if eligible else float("nan")
            ci_lo, ci_hi = cluster_bootstrap_interval(
                subset[["seq", "n_eligible", "n_success"]],
                replicates=bootstrap_replicates,
                q_lo=q_lo,
                q_hi=q_hi,
                seed=deterministic_seed(
                    bootstrap_seed, tracker, profile, index
                ),
            )
            if profile == "gap":
                x = left
                bin_label = f"{x:.3f} s"
            elif profile == "recovery":
                first_frame = int(round(left))
                last_frame = int(round(right)) - 1
                x = 0.5 * (first_frame + last_frame) / fps
                bin_label = (
                    f"{first_frame}–{last_frame} frames\n"
                    f"({first_frame / fps:.2f}–{last_frame / fps:.2f} s)"
                )
            else:
                x = 0.5 * (left + right)
                bin_label = format_bin(left, right)
            aggregate_rows.append({
                "tracker": tracker,
                "label": tracker_label(tracker),
                "profile": str(profile),
                "bin_index": index,
                "x": float(x),
                "x_left": left,
                "x_right": right,
                "bin_label": bin_label,
                "n_eligible": eligible,
                "n_success": success,
                "rate": rate,
                "rate_ci_lo": ci_lo,
                "rate_ci_hi": ci_hi,
                "events_per_1000": 1000.0 * rate,
                "events_per_1000_ci_lo": 1000.0 * ci_lo,
                "events_per_1000_ci_hi": 1000.0 * ci_hi,
                "bootstrap_replicates": int(bootstrap_replicates),
                "n_sequence_clusters": int(subset["seq"].nunique()),
            })
    return pd.DataFrame(aggregate_rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_global_overview(
    global_table: pd.DataFrame,
    gap_table: pd.DataFrame,
    hota_fps_trackers: list[str],
    gap_plot_trackers: list[str],
    reference: str,
    output_base: Path,
    realtime_fps: float,
) -> None:
    style_trackers = list(dict.fromkeys(
        hota_fps_trackers + gap_plot_trackers + [reference]
    ))
    styles = build_tracker_styles(style_trackers)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.3, 3.55),
        gridspec_kw={"width_ratios": [1.12, 0.88]},
    )

    # Panel A requires both freshly calculated HOTA and reusable frame_stats.
    ax = axes[0]
    ax.axvline(
        realtime_fps,
        linestyle="--",
        linewidth=1.5,
        label=f"Real-time ({realtime_fps:g} Hz)",
    )
    reference_rows = global_table[global_table["tracker"] == reference]
    if reference_rows.empty:
        raise ValueError(f"Reference tracker missing from global table: {reference}")
    reference_row = reference_rows.iloc[0]
    ax.axhline(
        float(reference_row.hota_combined),
        linestyle="--",
        linewidth=1.5,
        color="0.35",
        label=f"GT PedRefTrack = {reference_row.hota_combined:.1f}",
    )
    ax.axhline(
        float(reference_row.hota_seq_qlo),
        linestyle=":",
        linewidth=1.0,
        color="0.5",
    )
    ax.axhline(
        float(reference_row.hota_seq_qhi),
        linestyle=":",
        linewidth=1.0,
        color="0.5",
        label="PedRefTrack weighted sequence P10–P90",
    )

    for tracker in hota_fps_trackers:
        row = global_table[global_table["tracker"] == tracker]
        if row.empty:
            print(
                f"WARNING: no calculated HOTA for {tracker}; "
                "omitted from HOTA–FPS plot",
                file=sys.stderr,
            )
            continue
        row = row.iloc[0]
        if not (
            np.isfinite(row.get("fps_median", np.nan))
            and float(row.fps_median) > 0
        ):
            # load_runtime_tables already emitted the detailed warning.
            continue
        style = styles[tracker]
        ax.hlines(
            row.hota_combined,
            row.fps_qlo,
            row.fps_qhi,
            color=style["color"],
            alpha=0.65,
            linewidth=1.2,
        )
        ax.vlines(
            row.fps_median,
            row.hota_seq_qlo,
            row.hota_seq_qhi,
            color=style["color"],
            alpha=0.65,
            linewidth=1.2,
        )
        ax.scatter(
            row.fps_median,
            row.hota_combined,
            marker=style["marker"],
            color=style["color"],
            s=42,
            zorder=3,
        )
        ax.annotate(
            row.label,
            (row.fps_median, row.hota_combined),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Tracker-step FPS")
    ax.set_ylabel("Global HOTA (%)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(a) Accuracy–runtime overview", fontsize=9)

    # Panel B uses HOTA only and therefore can include variants without timing.
    ax = axes[1]
    subset = gap_table[
        gap_table["tracker"].isin(gap_plot_trackers)
    ].copy()
    subset = subset.sort_values(
        "hota_gap_combined", ascending=True
    ).reset_index(drop=True)
    y = np.arange(len(subset))
    for position, row in zip(y, subset.itertuples(index=False)):
        style = styles[row.tracker]
        ax.hlines(
            position,
            row.paired_seq_gap_qlo,
            row.paired_seq_gap_qhi,
            color=style["color"],
            linewidth=1.6,
            alpha=0.75,
        )
        ax.scatter(
            row.hota_gap_combined,
            position,
            marker=style["marker"],
            color=style["color"],
            s=42,
            zorder=3,
        )
    ax.axvline(0.0, linestyle="--", linewidth=1.0, color="0.35")
    ax.set_yticks(y, subset["label"])
    ax.set_xlabel("HOTA gap to GT PedRefTrack (pp)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("(b) Remaining tracker-side gap", fontsize=9)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_capability_profiles(
    profiles: pd.DataFrame,
    runtime: pd.DataFrame,
    profile_trackers: list[str],
    runtime_trackers: list[str],
    output_base: Path,
    realtime_fps: float,
) -> None:
    styles = build_tracker_styles(profile_trackers)
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.25))

    ax = axes[0, 0]
    _plot_profile_lines(
        ax, profiles, "gap", profile_trackers, styles, scale=100.0
    )
    ax.set_xlabel("Elapsed detector-gap age (s)")
    ax.set_ylabel("Continuous same-ID success (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.25)
    ax.set_title("(a) Gap-bridging capability", fontsize=9)

    ax = axes[0, 1]
    nn = profiles[profiles["profile"] == "nn"]
    _plot_profile_lines(
        ax, profiles, "nn", profile_trackers, styles, scale=100.0
    )
    if not nn.empty:
        ticks = nn.drop_duplicates("bin_index").sort_values("bin_index")
        ax.set_xticks(ticks["x"], ticks["bin_label"], rotation=25, ha="right")
    ax.set_xlabel("Matched-detector nearest-neighbour distance (m)")
    ax.set_ylabel("Identity-continuation success (%)")
    ax.set_ylim(0, 102)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.set_title("(b) Association under close detections", fontsize=9)

    ax = axes[1, 0]
    center = profiles[profiles["profile"] == "center_error"]
    _plot_profile_lines(
        ax,
        profiles,
        "center_error",
        profile_trackers,
        styles,
        scale=100.0,
    )
    if not center.empty:
        ticks = center.drop_duplicates("bin_index").sort_values("bin_index")
        ax.set_xticks(ticks["x"], ticks["bin_label"], rotation=25, ha="right")
    ax.set_xlabel("Detection-centre error (m)")
    ax.set_ylabel("Identity-preserving update success (%)")
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.25)
    ax.set_title("(c) Robustness to displaced measurements", fontsize=9)

    ax = axes[1, 1]
    runtime_styles = build_tracker_styles(runtime_trackers)
    for tracker in runtime_trackers:
        subset = runtime[runtime["tracker"] == tracker].sort_values("num_det_in")
        if subset.empty:
            continue
        style = runtime_styles[tracker]
        ax.plot(subset["num_det_in"], subset["fps_median"], label=tracker_label(tracker), linewidth=1.5, marker=style["marker"], markersize=3, markevery=max(1, len(subset) // 10), color=style["color"], linestyle=style["linestyle"])
    ax.axhline(realtime_fps, linestyle="--", linewidth=1.2, color="0.25", label=f"Real-time ({realtime_fps:g} Hz)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of input detections")
    ax.set_ylabel("Tracker-step FPS")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title("(d) Runtime scaling", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    runtime_handle = plt.Line2D([], [], linestyle="--", color="0.25", label=f"Real-time ({realtime_fps:g} Hz)")
    fig.legend(handles + [runtime_handle], labels + [runtime_handle.get_label()], loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=6.4, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_base)


def _plot_profile_lines(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    profile: str,
    trackers: list[str],
    styles: dict[str, dict[str, Any]],
    *,
    scale: float,
    show_ci: bool = False,
) -> None:
    for tracker in trackers:
        subset = profiles[
            (profiles["profile"] == profile)
            & (profiles["tracker"] == tracker)
        ].sort_values("x")
        if subset.empty:
            continue
        style = styles[tracker]
        x = subset["x"].to_numpy(dtype=float)
        y = scale * subset["rate"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            label=tracker_label(tracker),
            linewidth=1.45,
            marker=style["marker"],
            markersize=3.5,
            color=style["color"],
            linestyle=style["linestyle"],
        )
        if show_ci:
            lo = scale * subset["rate_ci_lo"].to_numpy(dtype=float)
            hi = scale * subset["rate_ci_hi"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                lo,
                hi,
                color=style["color"],
                alpha=0.045,
                linewidth=0,
            )


def _set_profile_ticks_with_support(
    ax: plt.Axes,
    profiles: pd.DataFrame,
    profile: str,
) -> None:
    subset = profiles[profiles["profile"] == profile]
    if subset.empty:
        return
    common = (
        subset.sort_values(["bin_index", "tracker"])
        .drop_duplicates("bin_index")
        .sort_values("bin_index")
    )
    labels = [
        f"{row.bin_label}\nN={int(row.n_eligible):,}"
        for row in common.itertuples(index=False)
    ]
    ax.set_xticks(common["x"], labels, rotation=25, ha="right")


def plot_gap_recovery(
    profiles: pd.DataFrame,
    trackers: list[str],
    output_base: Path,
) -> None:
    styles = build_tracker_styles(trackers)
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.65))
    _plot_profile_lines(
        axes[0], profiles, "gap", trackers, styles, scale=100.0, show_ci=True
    )
    axes[0].set_xlabel("Elapsed detector-gap age (s)")
    axes[0].set_ylabel("Continuous same-ID success (%)")
    axes[0].set_ylim(0, 102)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("(a) Prediction through the gap", fontsize=9)

    _plot_profile_lines(
        axes[1],
        profiles,
        "recovery",
        trackers,
        styles,
        scale=100.0,
        show_ci=True,
    )
    _set_profile_ticks_with_support(axes[1], profiles, "recovery")
    axes[1].set_xlabel("Total bounded detector-gap duration")
    axes[1].set_ylabel("Same-ID recovery at first post-gap frame (%)")
    axes[1].set_ylim(0, 102)
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title("(b) Post-gap identity recovery", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=6.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, output_base)


def plot_association_diagnostics(
    profiles: pd.DataFrame,
    trackers: list[str],
    output_base: Path,
) -> None:
    styles = build_tracker_styles(trackers)
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.35))
    specifications = [
        (
            axes[0, 0],
            "nn_failure",
            "Adjacent update failures / 1000",
            "(a) All identity-update failures",
        ),
        (
            axes[0, 1],
            "nn_confuser_capture",
            "Direct captures / 1000",
            "(b) Direct target–confuser captures",
        ),
        (
            axes[1, 0],
            "nn_window_failure",
            "8-frame window failures / 1000",
            "(c) Fixed close-encounter windows",
        ),
    ]
    for ax, profile, ylabel, title in specifications:
        _plot_profile_lines(
            ax,
            profiles,
            profile,
            trackers,
            styles,
            scale=1000.0,
            show_ci=True,
        )
        _set_profile_ticks_with_support(ax, profiles, profile)
        ax.invert_xaxis()
        ax.set_xlabel(
            "Matched-detector NN distance (m); closer → harder"
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9)

    # Mutually exclusive failure composition in the closest retained bin.
    ax = axes[1, 1]
    categories = [
        ("nn_failure_not_established", "Not established", "#9e9e9e"),
        ("nn_failure_current_miss", "Current miss", "#4c78a8"),
        ("nn_failure_id_change", "ID change", "#f58518"),
        ("nn_confuser_capture", "Direct confuser", "#e45756"),
    ]
    y_positions = np.arange(len(trackers))
    cumulative = np.zeros(len(trackers), dtype=float)
    for profile, label, color in categories:
        values = []
        for tracker in trackers:
            subset = profiles[
                (profiles["profile"] == profile)
                & (profiles["tracker"] == tracker)
                & (profiles["bin_index"] == 0)
            ]
            values.append(
                float(subset["events_per_1000"].iloc[0])
                if not subset.empty
                else 0.0
            )
        values_array = np.asarray(values, dtype=float)
        ax.barh(
            y_positions,
            values_array,
            left=cumulative,
            color=color,
            label=label,
            height=0.72,
        )
        cumulative += values_array
    ax.set_yticks(y_positions, [tracker_label(t) for t in trackers])
    ax.invert_yaxis()
    ax.set_xlabel("Failures / 1000 events")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("(d) Closest-bin failure composition", fontsize=9)
    ax.legend(fontsize=6.2, loc="lower right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=6.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_base)


def plot_robustness_diagnostics(
    profiles: pd.DataFrame,
    trackers: list[str],
    output_base: Path,
    isolated_nn_thr_m: float,
    encounter_window_frames: int,
) -> None:
    styles = build_tracker_styles(trackers)
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.35))
    specifications = [
        (
            axes[0, 0],
            "center_failure",
            "Detection-centre error (m)",
            "Update failures / 1000",
            "(a) All displaced measurements",
        ),
        (
            axes[0, 1],
            "center_isolated_failure",
            "Detection-centre error (m)",
            "Update failures / 1000",
            f"(b) Isolated targets (NN ≥ {isolated_nn_thr_m:g} m)",
        ),
        (
            axes[1, 0],
            "jitter_failure",
            "Detector-error step (m)",
            "Update failures / 1000",
            "(c) Frame-to-frame detector jitter",
        ),
        (
            axes[1, 1],
            "jitter_window_failure",
            "P90 detector-error step in window (m)",
            f"{encounter_window_frames}-frame failures / 1000",
            "(d) Sustained jitter windows",
        ),
    ]
    for ax, profile, xlabel, ylabel, title in specifications:
        _plot_profile_lines(
            ax,
            profiles,
            profile,
            trackers,
            styles,
            scale=1000.0,
            show_ci=True,
        )
        _set_profile_ticks_with_support(ax, profiles, profile)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_title(title, fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        fontsize=6.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_base)


# -----------------------------------------------------------------------------
# CLI and orchestration
# -----------------------------------------------------------------------------

def run_jobs(function, jobs: list[dict[str, Any]], workers: int, description: str) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if workers <= 1:
        iterator: Iterable[dict[str, Any]] = jobs
        if tqdm is not None:
            iterator = tqdm(jobs, desc=description)
        return [function(job) for job in iterator]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        future_map = {executor.submit(function, job): job for job in jobs}
        iterator = as_completed(future_map)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_map), desc=description)
        for future in iterator:
            results.append(future.result())
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="JRDB fixed-detection tracker capability profiles."
    )
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--trackers-base-dir", type=Path, required=True)
    parser.add_argument("--gt-folder", type=Path, required=True)
    parser.add_argument("--local-gt-folder", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, required=True)
    parser.add_argument("--results-test-dir", type=Path, default=None, help="Deprecated and ignored; HOTA is calculated directly from tracker outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hota-cache-script",
        type=Path,
        default=Path(__file__).with_name("hota_cache.py"),
    )
    parser.add_argument(
        "--reuse-hota-cache",
        type=Path,
        default=None,
        help=(
            "Optional HOTA cache frame-proxy cache used only for shared detector/GT "
            "features. It is not used as a HOTA-score source."
        ),
    )
    parser.add_argument(
        "--reuse-hota-events-from",
        type=Path,
        default=None,
        help=(
            "Optional compatible HOTA-event cache to reuse explicitly. "
            "Omit this to calculate HOTA events freshly from current tracker outputs."
        ),
    )
    parser.add_argument("--trackers", type=str, required=True)
    parser.add_argument("--reference-tracker", type=str, required=True)
    parser.add_argument("--global-plot-trackers", type=str, default=None)
    parser.add_argument("--runtime-plot-trackers", type=str, default=None)
    parser.add_argument("--anchor-tracker", type=str, default=None)
    parser.add_argument("--frame-stats-source-map", type=str, default=None, help="Optional tracker=source pairs for variants sharing frame_stats.")
    parser.add_argument("--recompute-trackers", type=str, default=None)
    parser.add_argument("--force-common", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--split-to-eval", type=str, default="test")
    parser.add_argument("--tracker-sub-folder", type=str, default="data")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--pre-gap-visible-frames", type=int, default=8)
    parser.add_argument("--max-gap-age-s", type=float, default=1.0)
    parser.add_argument("--success-iou-thr", type=float, default=0.30)
    parser.add_argument("--matchable-sim-thr", type=float, default=0.30)
    parser.add_argument("--det-match-max-dist-m", type=float, default=0.30)
    parser.add_argument("--det-score-min", type=float, default=None)
    parser.add_argument(
        "--nn-bin-edges",
        type=str,
        default="0.25,0.5,0.75,1.0",
        help=(
            "Matched-detector centre-to-centre NN bins. The default excludes "
            "the under-supported distance range below 0.25 m."
        ),
    )
    parser.add_argument("--center-error-bin-edges", type=str, default="0,0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument(
        "--jitter-bin-edges",
        type=str,
        default="0,0.05,0.10,0.15,0.20,0.30,0.45,0.60",
        help=(
            "Bins for ||(det-GT)_t - (det-GT)_{t-1}|| and window P90."
        ),
    )
    parser.add_argument(
        "--recovery-gap-bin-edges-frames",
        type=str,
        default="1,4,7,10,13,16",
        help="Half-open total-gap bins; default gives 1-3,...,13-15 frames.",
    )
    parser.add_argument(
        "--encounter-window-frames",
        type=int,
        default=8,
        help="Length of non-overlapping close-encounter/jitter windows.",
    )
    parser.add_argument(
        "--isolated-nn-thr-m",
        type=float,
        default=1.0,
        help=(
            "A robustness update is isolated when detector NN distance is at "
            "least this value (or no matched neighbour exists) at both frames."
        ),
    )
    parser.add_argument("--hota-spread-quantiles", type=str, default="0.10,0.90")
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2000,
        help="Sequence-cluster bootstrap replicates for direct-profile intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260725,
    )
    parser.add_argument(
        "--profile-ci-quantiles",
        type=str,
        default="0.025,0.975",
        help="Two quantiles for sequence-cluster bootstrap intervals.",
    )
    parser.add_argument("--realtime-fps", type=float, default=10.0)
    parser.add_argument("--runtime-min-frames-per-count", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trackers = parse_csv_list(args.trackers)
    if not trackers:
        raise ValueError("--trackers is empty")
    if args.reference_tracker not in trackers:
        trackers.append(args.reference_tracker)
    global_plot_trackers = parse_csv_list(args.global_plot_trackers) or [tracker for tracker in trackers if tracker != args.reference_tracker]
    runtime_plot_trackers = parse_csv_list(args.runtime_plot_trackers) or global_plot_trackers
    recompute = set(parse_csv_list(args.recompute_trackers))
    source_map = parse_mapping(args.frame_stats_source_map)
    anchor_tracker = args.anchor_tracker or trackers[0]
    nn_edges = parse_float_edges(args.nn_bin_edges)
    center_edges = parse_float_edges(args.center_error_bin_edges)
    jitter_edges = parse_float_edges(args.jitter_bin_edges)
    recovery_edges = parse_int_edges(
        args.recovery_gap_bin_edges_frames
    )
    quantiles = parse_float_edges(args.hota_spread_quantiles)
    profile_ci_quantiles = parse_float_edges(
        args.profile_ci_quantiles
    )
    if len(quantiles) != 2 or not (0 <= quantiles[0] < quantiles[1] <= 1):
        raise ValueError("--hota-spread-quantiles must contain two values in [0,1]")
    q_lo, q_hi = quantiles
    if (
        len(profile_ci_quantiles) != 2
        or not (
            0
            <= profile_ci_quantiles[0]
            < profile_ci_quantiles[1]
            <= 1
        )
    ):
        raise ValueError(
            "--profile-ci-quantiles must contain two values in [0,1]"
        )
    profile_ci_lo, profile_ci_hi = profile_ci_quantiles
    max_gap_age_frames = int(round(float(args.max_gap_age_s) * float(args.fps)))
    if max_gap_age_frames < 1:
        raise ValueError("--max-gap-age-s is too small")
    if args.pre_gap_visible_frames < 1:
        raise ValueError("--pre-gap-visible-frames must be >= 1")
    if nn_edges[0] < 0.25 - 1e-12:
        raise ValueError(
            "--nn-bin-edges must start at 0.25 m or above because the "
            "sub-0.25 m population is intentionally excluded."
        )
    if recovery_edges[0] != 1:
        raise ValueError(
            "--recovery-gap-bin-edges-frames must start at frame 1"
        )
    if recovery_edges[-1] - 1 > max_gap_age_frames:
        raise ValueError(
            "Recovery bins extend beyond --max-gap-age-s; make the final "
            "edge at most max_gap_age_frames + 1."
        )
    if args.encounter_window_frames < 2:
        raise ValueError("--encounter-window-frames must be >= 2")
    if args.isolated_nn_thr_m <= 0:
        raise ValueError("--isolated-nn-thr-m must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be >= 0")

    output_dir = args.output_dir.resolve()
    common_dir = output_dir / "common"
    profile_dir = output_dir / "profiles"
    hota_event_dir = output_dir / "hota_events"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for directory in (
        common_dir / "sequences",
        profile_dir,
        hota_event_dir,
        tables_dir,
        figures_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    helper = load_helper(args.hota_cache_script)
    helper.add_trackeval_to_path(args.trackeval_root)
    import trackeval  # noqa: WPS433
    anchor_dataset = helper.make_dataset(
        trackeval,
        args.trackers_base_dir.resolve(),
        args.gt_folder.resolve(),
        anchor_tracker,
        args.split_to_eval,
        args.tracker_sub_folder,
        args.matchable_sim_thr,
    )
    sequences = list(anchor_dataset.seq_list)

    common_settings = {
        "cache_version": CACHE_VERSION,
        "trackeval_root": str(args.trackeval_root.resolve()),
        "trackers_base_dir": str(args.trackers_base_dir.resolve()),
        "gt_folder": str(args.gt_folder.resolve()),
        "local_gt_folder": str(args.local_gt_folder.resolve()),
        "detections_dir": str(args.detections_dir.resolve()),
        "split_to_eval": args.split_to_eval,
        "tracker_sub_folder": args.tracker_sub_folder,
        "fps": float(args.fps),
        "det_match_max_dist_m": float(args.det_match_max_dist_m),
        "matchable_sim_thr": float(args.matchable_sim_thr),
        "det_score_min": args.det_score_min,
        "reuse_hota_cache": str(args.reuse_hota_cache.resolve()) if args.reuse_hota_cache else None,
        "sequences": sequences,
    }
    signature = stable_signature(common_settings)
    metadata_path = common_dir / "metadata.json"
    if metadata_path.exists() and not args.force_common:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("signature") != signature:
            raise RuntimeError("Common cache settings changed. Use a new --output-dir or --force-common.")
    else:
        atomic_json(metadata_path, {**common_settings, "signature": signature})

    if not args.plot_only:
        common_jobs = [{
            **common_settings,
            "helper_script": str(args.hota_cache_script.resolve()),
            "common_dir": str(common_dir),
            "anchor_tracker": anchor_tracker,
            "seq": seq,
            "force": bool(args.force_common),
        } for seq in sequences]
        common_status = run_jobs(_common_sequence_worker, common_jobs, args.num_workers, "Common capability events")
        pd.DataFrame(common_status).to_csv(common_dir / "build_status.csv", index=False)

        jobs: list[dict[str, Any]] = []
        for tracker in trackers:
            tracker_meta = {
                "cache_version": CACHE_VERSION,
                "common_signature": signature,
                "tracker": tracker,
                "success_iou_thr": float(args.success_iou_thr),
                "pre_gap_visible_frames": int(args.pre_gap_visible_frames),
                "max_gap_age_frames": int(max_gap_age_frames),
                "nn_edges": nn_edges,
                "center_edges": center_edges,
                "jitter_edges": jitter_edges,
                "recovery_edges_frames": recovery_edges,
                "encounter_window_frames": int(
                    args.encounter_window_frames
                ),
                "isolated_nn_thr_m": float(args.isolated_nn_thr_m),
            }
            meta_path = profile_dir / safe_tracker_name(tracker) / "metadata.json"
            tracker_signature = stable_signature(tracker_meta)
            if meta_path.exists() and tracker not in recompute:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
                if existing.get("signature") != tracker_signature:
                    raise RuntimeError(f"Profile settings changed for {tracker}. Use --recompute-trackers or a new output directory.")
            atomic_json(meta_path, {**tracker_meta, "signature": tracker_signature})
            for seq in sequences:
                jobs.append({
                    "helper_script": str(args.hota_cache_script.resolve()),
                    "trackeval_root": str(args.trackeval_root.resolve()),
                    "trackers_base_dir": str(args.trackers_base_dir.resolve()),
                    "gt_folder": str(args.gt_folder.resolve()),
                    "split_to_eval": args.split_to_eval,
                    "tracker_sub_folder": args.tracker_sub_folder,
                    "matchable_sim_thr": float(args.matchable_sim_thr),
                    "success_iou_thr": float(args.success_iou_thr),
                    "fps": float(args.fps),
                    "pre_gap_visible_frames": int(args.pre_gap_visible_frames),
                    "max_gap_age_frames": int(max_gap_age_frames),
                    "nn_edges": nn_edges,
                    "center_edges": center_edges,
                    "jitter_edges": jitter_edges,
                    "recovery_edges_frames": recovery_edges,
                    "encounter_window_frames": int(
                        args.encounter_window_frames
                    ),
                    "isolated_nn_thr_m": float(
                        args.isolated_nn_thr_m
                    ),
                    "common_dir": str(common_dir),
                    "profile_dir": str(profile_dir),
                    "hota_event_dir": str(hota_event_dir),
                    "reuse_hota_events_from": (
                        str(args.reuse_hota_events_from.resolve())
                        if args.reuse_hota_events_from
                        else None
                    ),
                    "tracker": tracker,
                    "seq": seq,
                    "force_tracker": tracker in recompute,
                })
        tracker_status = run_jobs(_tracker_sequence_worker, jobs, args.num_workers, "Tracker capability profiles")
        pd.DataFrame(tracker_status).to_csv(output_dir / "tracker_build_status.csv", index=False)

    profiles = aggregate_profile_caches(
        profile_dir,
        trackers,
        sequences,
        args.fps,
        args.bootstrap_replicates,
        args.bootstrap_seed,
        profile_ci_lo,
        profile_ci_hi,
    )
    denominator_check = profiles.groupby(
        ["profile", "bin_index"]
    )["n_eligible"].nunique()
    if (denominator_check > 1).any():
        bad_profile, bad_bin = denominator_check[
            denominator_check > 1
        ].index[0]
        raise RuntimeError(
            "Tracker-dependent direct-profile denominator detected for "
            f"{bad_profile}, bin {bad_bin}"
        )
    profiles.to_csv(
        tables_dir / "direct_profile_summary.csv", index=False
    )
    profiles[profiles["profile"] == "gap"].to_csv(
        tables_dir / "gap_bridging_success.csv", index=False
    )
    profiles[profiles["profile"] == "recovery"].to_csv(
        tables_dir / "post_gap_same_id_recovery.csv", index=False
    )
    profiles[profiles["profile"] == "nn"].to_csv(
        tables_dir / "identity_success_vs_nn_distance.csv", index=False
    )
    profiles[profiles["profile"] == "center_error"].to_csv(
        tables_dir / "identity_update_success_vs_center_error.csv",
        index=False,
    )
    profiles[profiles["profile"].str.startswith("nn_")].to_csv(
        tables_dir / "association_failure_diagnostics.csv", index=False
    )
    profiles[
        profiles["profile"].str.startswith(("center_", "jitter_"))
        & ~profiles["profile"].isin(["center_error"])
    ].to_csv(
        tables_dir / "robustness_failure_diagnostics.csv", index=False
    )

    hota_summary, gap_table, hota_per_sequence = build_hota_tables(
        hota_event_dir,
        trackers,
        sequences,
        args.reference_tracker,
        q_lo,
        q_hi,
    )
    hota_summary.to_csv(tables_dir / "global_hota_metrics.csv", index=False)
    hota_per_sequence.to_csv(tables_dir / "hota_per_sequence.csv", index=False)
    metric_decomposition = global_gap_decomposition(
        hota_summary, args.reference_tracker
    )
    metric_decomposition.to_csv(
        tables_dir / "global_metric_gap_to_reference.csv", index=False
    )
    gap_table.to_csv(
        tables_dir / "paired_hota_gap_to_reference.csv", index=False
    )

    runtime_summary, runtime_frames = load_runtime_tables(
        args.trackers_base_dir.resolve(),
        runtime_plot_trackers,
        source_map,
        args.split_to_eval,
        q_lo,
        q_hi,
    )
    global_table = merge_hota_runtime(hota_summary, runtime_summary)
    global_table.to_csv(tables_dir / "global_hota_fps.csv", index=False)

    runtime = aggregate_runtime_by_detection_count(
        runtime_frames,
        runtime_plot_trackers,
        args.runtime_min_frames_per_count,
    )
    runtime.to_csv(
        tables_dir / "runtime_vs_detection_count.csv", index=False
    )

    coverage = (
        profiles.sort_values(["profile", "bin_index", "tracker"])
        .drop_duplicates(["profile", "bin_index"])
        [[
            "profile",
            "bin_index",
            "x",
            "x_left",
            "x_right",
            "bin_label",
            "n_eligible",
        ]]
        .reset_index(drop=True)
    )
    coverage.to_csv(tables_dir / "profile_coverage.csv", index=False)

    plot_global_overview(
        global_table,
        gap_table,
        runtime_plot_trackers,
        global_plot_trackers,
        args.reference_tracker,
        figures_dir / "global_hota_runtime_and_pedreftrack_gap",
        args.realtime_fps,
    )
    plot_capability_profiles(
        profiles,
        runtime,
        trackers,
        runtime_plot_trackers,
        figures_dir / "tracker_capability_profiles_2x2",
        args.realtime_fps,
    )
    plot_gap_recovery(
        profiles,
        trackers,
        figures_dir / "gap_bridging_and_post_gap_recovery",
    )
    plot_association_diagnostics(
        profiles,
        trackers,
        figures_dir / "association_confuser_diagnostics",
    )
    plot_robustness_diagnostics(
        profiles,
        trackers,
        figures_dir / "detector_jitter_robustness_diagnostics",
        args.isolated_nn_thr_m,
        args.encounter_window_frames,
    )

    run_metadata = {
        "script_version": SCRIPT_VERSION,
        "cache_version": CACHE_VERSION,
        "trackers": trackers,
        "reference_tracker": args.reference_tracker,
        "global_plot_trackers": global_plot_trackers,
        "runtime_plot_trackers": runtime_plot_trackers,
        "sequences": sequences,
        "pre_gap_visible_frames": args.pre_gap_visible_frames,
        "max_gap_age_s_requested": args.max_gap_age_s,
        "max_gap_age_frames": max_gap_age_frames,
        "max_gap_age_s_actual": max_gap_age_frames / args.fps,
        "success_iou_thr": args.success_iou_thr,
        "nn_edges": nn_edges,
        "center_error_edges": center_edges,
        "jitter_edges": jitter_edges,
        "recovery_gap_bin_edges_frames": recovery_edges,
        "encounter_window_frames": args.encounter_window_frames,
        "isolated_nn_thr_m": args.isolated_nn_thr_m,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "profile_ci_quantiles": [
            profile_ci_lo,
            profile_ci_hi,
        ],
        "hota_spread_quantiles": [q_lo, q_hi],
        "reuse_hota_cache_for_common_features": (
            str(args.reuse_hota_cache.resolve())
            if args.reuse_hota_cache
            else None
        ),
        "reuse_hota_events_from": (
            str(args.reuse_hota_events_from.resolve())
            if args.reuse_hota_events_from
            else None
        ),
        "output_files": {
            "global_hota_metrics": str(tables_dir / "global_hota_metrics.csv"),
            "hota_per_sequence": str(tables_dir / "hota_per_sequence.csv"),
            "global_metric_gap_to_reference": str(
                tables_dir / "global_metric_gap_to_reference.csv"
            ),
            "global_hota_fps": str(tables_dir / "global_hota_fps.csv"),
            "paired_hota_gap": str(
                tables_dir / "paired_hota_gap_to_reference.csv"
            ),
            "capability_profiles": str(
                figures_dir / "tracker_capability_profiles_2x2.pdf"
            ),
            "gap_recovery": str(
                figures_dir
                / "gap_bridging_and_post_gap_recovery.pdf"
            ),
            "association_diagnostics": str(
                figures_dir / "association_confuser_diagnostics.pdf"
            ),
            "robustness_diagnostics": str(
                figures_dir
                / "detector_jitter_robustness_diagnostics.pdf"
            ),
        },
        "notes": {
            "hota_center": "HOTA calculated from current tracker outputs and globally combined from HOTA event sufficient statistics",
            "hota_event_reuse": "disabled unless --reuse-hota-events-from is explicitly provided; --reuse-hota-cache affects common detector features only",
            "hota_vertical_spread": "GT-occurrence-weighted per-sequence P10-P90",
            "fps_center": "median over all usable per-frame tracker-step FPS from frame_stats; no warm-up exclusion",
            "fps_horizontal_spread": "per-frame P10-P90",
            "pedreftrack_gap_center": "reference COMBINED HOTA minus tracker COMBINED HOTA",
            "pedreftrack_gap_spread": "paired GT-occurrence-weighted per-sequence gap P10-P90",
            "direct_profiles": "globally defined tracker-independent event populations",
            "direct_profile_uncertainty": "sequence-cluster bootstrap interval using --profile-ci-quantiles",
            "nearest_neighbour": "matched detector centre-to-centre distance; events below 0.25 m excluded",
            "post_gap_recovery": "same correctly localized tracker ID at the first detector-supported GT frame after the gap; absent pre-track is failure",
            "obsolete_results_test_dir": "not read; --results-test-dir is accepted only as a deprecated ignored argument",
            "runtime_missing_policy": "skip tracker only from runtime-dependent plots; retain HOTA and direct profiles",
        },
    }
    atomic_json(output_dir / "run_metadata.json", run_metadata)
    print(f"Finished. Results written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
