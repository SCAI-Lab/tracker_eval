#!/usr/bin/env python3
"""
Incrementally compute JRDB/RAL tracker-result tables for the plotting notebook.

The protocol defines initialization/re-acquisition as the earliest tracker output that
starts within a configurable latency window and then remains continuously and
correctly assigned to the GT with one unchanged tracker ID for a configurable
validation duration. Both durations are specified in seconds and converted to
frames with ``ceil(duration * fps)``.

The tracker list is a registry, not part of the evaluation protocol. On every
run, the script:

* reuses complete per-tracker/per-sequence profile and HOTA-event caches;
* computes only missing caches for newly listed trackers;
* recomputes only trackers explicitly named by ``--recompute-trackers``;
* rebuilds the aggregate CSV tables from the requested tracker registry; and
* rescans frame_stats for every tracker, even when its evaluation cache exists;
  and
* rebuilds a tracker-common stable-establishment population from reusable
  per-frame assignment caches.

No figures are produced. The notebook can load the resulting tables from
``OUTPUT_DIR/tables`` without running TrackEval or tracker evaluation.

The capability profiles use a fresh per-frame Hungarian assignment on raw
TrackEval 3D similarity. Event populations are defined only from GT and the
shared detector cache, so every tracker receives the same population. Global
HOTA is computed from selected tracker outputs through per-sequence HOTA event
caches; exported TrackEval result CSVs are never used as metric inputs.

Compatible caches are reused when their protocol-defining settings match.
Tracker membership, reference
selection, timing availability, aggregation quantiles, bootstrap settings, and
the initialization settings do not invalidate the expensive per-sequence
caches. Changing tracker membership does rebuild the inexpensive common
post-gap population because its eligibility explicitly requires all requested
trackers to have lost and restarted the same GT trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCRIPT_VERSION = "tracker_eval_protocol"
CACHE_PROTOCOL_VERSION = "tracker_eval_direct_profiles"
ASSIGNMENT_CACHE_VERSION = 1

LABELS = {
    "pedreftrack__global_gt_assisted": "GT-assisted PedRefTrack",
    "pedreftrack__global_no_gt": "PedRefTrack (no GT)",
    "fastpoly__global": "Fast-Poly",
    "simpletrack__global": "SimpleTrack",
    "elptnet__global": "ELPTNet (box)",
    "gnnpmb__global": "GNN-PMB",
    "cbmot__global": "CBMOT",
    "ab3dmot__global": "AB3DMOT",
}

COMMON_CHUNK_COLUMNS = (
    "seq",
    "event_id",
    "gt_internal_id",
    "gt_orig_id",
    "interval_index",
    "chunk_index",
    "interval_start_frame",
    "interval_end_frame",
    "chunk_start_frame",
    "chunk_end_frame",
    "chunk_frames",
    "chunk_s",
    "warmup_frames",
    "max_detector_gap_frames",
    "n_nn_frames",
    "nn_p10_m",
)


def parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_edges(value: str) -> list[int]:
    edges = [int(item) for item in parse_csv_list(value)]
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"Invalid strictly increasing integer edges: {value}")
    return edges


def parse_float_pair(value: str, name: str) -> tuple[float, float]:
    values = [float(item) for item in parse_csv_list(value)]
    if len(values) != 2 or not (0 <= values[0] < values[1] <= 1):
        raise ValueError(f"{name} must contain two increasing values in [0,1]")
    return values[0], values[1]


def parse_mapping(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"Expected tracker=source, got: {item}")
        key, mapped = (part.strip() for part in item.split("=", 1))
        if not key or not mapped:
            raise ValueError(f"Invalid tracker mapping: {item}")
        result[key] = mapped
    return result


def safe_name(value: str) -> str:
    return str(value).replace("/", "__").replace("\\", "__").replace(":", "_")


def label(tracker: str) -> str:
    return LABELS.get(
        tracker,
        tracker.replace("__global", "").replace("_", " "),
    )


def load_module(path: Path, prefix: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    module_name = f"{prefix}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, "TRACKER_LABELS"):
        module.TRACKER_LABELS.update(LABELS)
    return module


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.gz")
    frame.to_csv(temporary, index=False, compression="gzip")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def qvalue(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan")
    try:
        return float(np.quantile(array, q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(array, q, interpolation="linear"))


def stable_signature(payload: dict[str, Any]) -> str:
    content = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def split_detector_intervals(
    supported_frames: list[int],
    gt_frames: set[int],
    max_missing_frames: int,
) -> list[list[int]]:
    """Join detector-supported endpoints only across GT-present short gaps."""
    if not supported_frames:
        return []
    intervals: list[list[int]] = [[int(supported_frames[0])]]
    for value in supported_frames[1:]:
        current = int(value)
        previous = int(intervals[-1][-1])
        missing = current - previous - 1
        gt_continuous = all(
            frame in gt_frames for frame in range(previous + 1, current)
        )
        if 0 <= missing <= max_missing_frames and gt_continuous:
            intervals[-1].append(current)
        else:
            intervals.append([current])
    return intervals


def build_common_chunks(
    *,
    sequence: str,
    common_path: Path,
    output_path: Path,
    fps: float,
    max_detector_gap_frames: int,
    warmup_frames: int,
    chunk_frames: int,
    min_nn_frames: int,
    force: bool,
) -> dict[str, Any]:
    """Build tracker-independent strict-continuity chunks for one sequence."""
    if output_path.exists() and not force:
        cached = pd.read_csv(output_path)
        return {
            "seq": sequence,
            "status": "cached",
            "n_chunks": int(len(cached)),
        }

    common = load_npz(common_path)
    required = {
        "num_timesteps",
        "frame_gt_offsets",
        "gt_internal_ids_flat",
        "gt_orig_ids_flat",
        "gt_boxes_internal_flat",
        "detector_present_flat",
    }
    missing = required - set(common)
    if missing:
        raise KeyError(f"{sequence}: common cache lacks {sorted(missing)}")

    n_frames = int(np.asarray(common["num_timesteps"]).reshape(-1)[0])
    offsets = np.asarray(common["frame_gt_offsets"], dtype=np.int64)
    gids = np.asarray(common["gt_internal_ids_flat"], dtype=np.int32)
    orig_ids = np.asarray(common["gt_orig_ids_flat"], dtype=np.int64)
    gt_boxes = np.asarray(common["gt_boxes_internal_flat"], dtype=float)
    detector_present = np.asarray(common["detector_present_flat"], dtype=bool)
    if len(offsets) != n_frames + 1:
        raise RuntimeError(f"{sequence}: malformed frame offsets")

    occurrences: dict[int, list[tuple[int, int]]] = defaultdict(list)
    original_by_gid: dict[int, int] = {}
    nearest: dict[tuple[int, int], float] = {}

    for frame in range(n_frames):
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        for flat in range(start, end):
            gid = int(gids[flat])
            occurrences[gid].append((frame, flat))
            original_by_gid.setdefault(gid, int(orig_ids[flat]))
        if end - start < 2:
            continue
        flats = np.arange(start, end, dtype=np.int64)
        xy = gt_boxes[flats, :2]
        distances = np.linalg.norm(
            xy[:, None, :] - xy[None, :, :],
            axis=2,
        )
        np.fill_diagonal(distances, np.inf)
        for local_index, flat in enumerate(flats.tolist()):
            distance = float(np.min(distances[local_index]))
            if np.isfinite(distance):
                nearest[(int(gids[flat]), frame)] = distance

    rows: list[dict[str, Any]] = []
    for gid, gid_occurrences in occurrences.items():
        gt_frame_set = {int(frame) for frame, _ in gid_occurrences}
        supported = [
            int(frame)
            for frame, flat in gid_occurrences
            if bool(detector_present[flat])
        ]
        for interval_index, interval in enumerate(
            split_detector_intervals(
                supported,
                gt_frame_set,
                max_detector_gap_frames,
            )
        ):
            if not interval:
                continue
            interval_start, interval_end = int(interval[0]), int(interval[-1])
            start = interval_start + warmup_frames
            chunk_index = 0
            while start + chunk_frames - 1 <= interval_end:
                end = start + chunk_frames - 1
                nn_values = [
                    nearest[(gid, frame)]
                    for frame in range(start, end + 1)
                    if (gid, frame) in nearest
                ]
                if len(nn_values) >= min_nn_frames:
                    rows.append(
                        {
                            "seq": sequence,
                            "event_id": (
                                f"{sequence}|{gid}|{interval_index}|"
                                f"{start}-{end}"
                            ),
                            "gt_internal_id": gid,
                            "gt_orig_id": original_by_gid[gid],
                            "interval_index": interval_index,
                            "chunk_index": chunk_index,
                            "interval_start_frame": interval_start,
                            "interval_end_frame": interval_end,
                            "chunk_start_frame": start,
                            "chunk_end_frame": end,
                            "chunk_frames": chunk_frames,
                            "chunk_s": chunk_frames / fps,
                            "warmup_frames": warmup_frames,
                            "max_detector_gap_frames": (
                                max_detector_gap_frames
                            ),
                            "n_nn_frames": len(nn_values),
                            "nn_p10_m": qvalue(nn_values, 0.10),
                        }
                    )
                start += chunk_frames
                chunk_index += 1

    frame = pd.DataFrame(rows, columns=COMMON_CHUNK_COLUMNS)
    if not frame.empty and frame["event_id"].duplicated().any():
        raise RuntimeError(f"{sequence}: duplicate chunk IDs")
    atomic_csv_gz(output_path, frame)
    return {
        "seq": sequence,
        "status": "computed",
        "n_chunks": int(len(frame)),
    }


def _direct_tracker_sequence_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Calculate direct profiles, HOTA events, and reusable assignments."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    tracker = str(payload["tracker"])
    sequence = str(payload["sequence"])
    destination = (
        Path(payload["profile_dir"])
        / safe_name(tracker)
        / f"{sequence}.csv.gz"
    )
    assignment_path = (
        Path(payload["assignment_dir"])
        / safe_name(tracker)
        / f"{sequence}.npz"
    )
    force_tracker = bool(payload["force_tracker"])
    profile_needed = bool(force_tracker or not destination.exists())
    assignment_needed = bool(force_tracker or not assignment_path.exists())
    profiles_module = load_module(Path(payload["profiles_script"]), "ral_profiles")
    helper = profiles_module.load_helper(Path(payload["hota_cache_script"]))
    helper.add_trackeval_to_path(Path(payload["trackeval_root"]))
    import trackeval  # noqa: WPS433

    hota_path = (
        Path(payload["hota_event_dir"])
        / safe_name(tracker)
        / f"{sequence}.npz"
    )
    hota_payload = {
        "force_tracker": force_tracker,
        "reuse_hota_events_from": payload["reuse_hota_events_from"],
        "trackeval_root": payload["trackeval_root"],
        "trackers_base_dir": payload["trackers_base_dir"],
        "gt_folder": payload["gt_folder"],
        "split_to_eval": payload["split_to_eval"],
        "tracker_sub_folder": payload["tracker_sub_folder"],
        "matchable_sim_thr": payload["matchable_sim_thr"],
    }
    event_status = profiles_module.ensure_hota_event(
        helper,
        hota_payload,
        tracker,
        sequence,
        hota_path,
    )
    if not profile_needed and not assignment_needed:
        return {
            "tracker": tracker,
            "seq": sequence,
            "status": "cached",
            "hota_event_status": event_status,
            "profile_status": "cached",
            "assignment_status": "cached",
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
    data = helper.load_preprocessed_sequence(dataset, tracker, sequence)
    common = load_npz(
        Path(payload["common_dir"]) / "sequences" / f"{sequence}.npz"
    )
    chunks_path = Path(payload["chunk_dir"]) / f"{sequence}.csv.gz"
    chunks = pd.read_csv(chunks_path) if chunks_path.exists() else pd.DataFrame()

    n_frames = int(data["num_timesteps"])
    offsets = np.asarray(common["frame_gt_offsets"], dtype=np.int64)
    gids_flat = np.asarray(common["gt_internal_ids_flat"], dtype=np.int32)
    frame_flat = np.asarray(common["frame_indices_flat"], dtype=np.int32)
    detector_present = np.asarray(common["detector_present_flat"], dtype=bool)
    gap_id_flat = np.asarray(common["detector_gap_id_flat"], dtype=np.int64)
    visible_run = np.asarray(
        common["detector_visible_run_flat"],
        dtype=np.int16,
    )
    expected_offsets = np.concatenate(
        [[0], np.cumsum([len(ids) for ids in data["gt_ids"]])]
    ).astype(np.int64)
    if not np.array_equal(offsets, expected_offsets):
        raise RuntimeError(
            f"{tracker}:{sequence}: tracker/common frame offsets differ"
        )

    frame_to_flat: list[dict[int, int]] = []
    for frame in range(n_frames):
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        frame_to_flat.append(
            {
                int(gid): start + local
                for local, gid in enumerate(gids_flat[start:end])
            }
        )
    assignments = profiles_module.local_gt_to_tracker_assignments(
        data,
        float(payload["success_iou_thr"]),
    )

    if assignment_needed:
        assigned_tracker_id = np.full(len(gids_flat), -1, dtype=np.int64)
        for frame, mapping in enumerate(assignments):
            for gid, tracker_id in mapping.items():
                flat = frame_to_flat[frame].get(int(gid))
                if flat is None:
                    raise RuntimeError(
                        f"{tracker}:{sequence}: assignment references "
                        f"unknown GT internal ID {gid} at frame {frame}"
                    )
                assigned_tracker_id[flat] = int(tracker_id)
        atomic_npz(
            assignment_path,
            {
                "cache_version": np.asarray(
                    [ASSIGNMENT_CACHE_VERSION],
                    dtype=np.int16,
                ),
                "num_timesteps": np.asarray([n_frames], dtype=np.int64),
                "frame_gt_offsets": offsets.astype(np.int64, copy=False),
                "gt_internal_ids_flat": gids_flat.astype(
                    np.int32,
                    copy=False,
                ),
                "frame_indices_flat": frame_flat.astype(
                    np.int32,
                    copy=False,
                ),
                "assigned_tracker_orig_id_flat": assigned_tracker_id,
            },
        )

    if not profile_needed:
        return {
            "tracker": tracker,
            "seq": sequence,
            "status": "computed_assignment_only",
            "hota_event_status": event_status,
            "profile_status": "cached",
            "assignment_status": "computed",
        }

    max_gap_age_frames = int(payload["max_gap_age_frames"])
    recovery_edges = list(payload["recovery_edges_frames"])
    gap_eligible = np.zeros(max_gap_age_frames, dtype=np.int64)
    gap_success = np.zeros(max_gap_age_frames, dtype=np.int64)
    recovery_eligible = np.zeros(len(recovery_edges) - 1, dtype=np.int64)
    recovery_success = np.zeros(len(recovery_edges) - 1, dtype=np.int64)

    for gap_id in np.unique(gap_id_flat[gap_id_flat >= 0]).tolist():
        flats = np.flatnonzero(gap_id_flat == int(gap_id))
        flats = flats[np.argsort(frame_flat[flats])]
        if not len(flats):
            continue
        gid = int(gids_flat[flats[0]])
        first_missing = int(frame_flat[flats[0]])
        pre_frame = first_missing - 1
        if pre_frame < 0:
            continue
        pre_flat = frame_to_flat[pre_frame].get(gid)
        if (
            pre_flat is None
            or int(visible_run[pre_flat])
            < int(payload["pre_gap_visible_frames"])
        ):
            continue

        pre_track = assignments[pre_frame].get(gid)
        continuous = pre_track is not None
        for age in range(1, min(len(flats), max_gap_age_frames) + 1):
            current_track = assignments[pre_frame + age].get(gid)
            continuous = bool(
                continuous
                and pre_track is not None
                and current_track == pre_track
            )
            gap_eligible[age - 1] += 1
            gap_success[age - 1] += int(continuous)

        gap_length = int(len(flats))
        bin_index = int(
            np.searchsorted(recovery_edges, gap_length, side="right") - 1
        )
        post_frame = first_missing + gap_length
        post_flat = (
            frame_to_flat[post_frame].get(gid)
            if post_frame < n_frames
            else None
        )
        if (
            0 <= bin_index < len(recovery_eligible)
            and post_flat is not None
            and bool(detector_present[post_flat])
        ):
            post_track = assignments[post_frame].get(gid)
            recovery_eligible[bin_index] += 1
            recovery_success[bin_index] += int(
                pre_track is not None and post_track == pre_track
            )

    nn_edges = np.asarray(payload["nn_edges"], dtype=float)
    continuity_eligible = np.zeros(len(nn_edges) - 1, dtype=np.int64)
    continuity_success = np.zeros(len(nn_edges) - 1, dtype=np.int64)
    if not chunks.empty:
        for row in chunks.itertuples(index=False):
            value = float(row.nn_p10_m)
            index = int(np.searchsorted(nn_edges, value, side="right") - 1)
            if math.isclose(value, float(nn_edges[-1])):
                index = len(nn_edges) - 2
            if not (0 <= index < len(continuity_eligible)):
                continue
            gid = int(row.gt_internal_id)
            observed = [
                assignments[frame].get(gid)
                for frame in range(
                    int(row.chunk_start_frame),
                    int(row.chunk_end_frame) + 1,
                )
            ]
            strict_success = (
                len(observed) == int(row.chunk_frames)
                and all(track_id is not None for track_id in observed)
                and len(set(observed)) == 1
            )
            continuity_eligible[index] += 1
            continuity_success[index] += int(strict_success)

    rows: list[dict[str, Any]] = []
    for index in range(max_gap_age_frames):
        rows.append(
            {
                "tracker": tracker,
                "seq": sequence,
                "profile": "gap_prediction",
                "bin_index": index,
                "x_left": (index + 1) / float(payload["fps"]),
                "x_right": (index + 1) / float(payload["fps"]),
                "n_eligible": int(gap_eligible[index]),
                "n_success": int(gap_success[index]),
            }
        )
    for index in range(len(recovery_edges) - 1):
        rows.append(
            {
                "tracker": tracker,
                "seq": sequence,
                "profile": "post_gap_recovery",
                "bin_index": index,
                "x_left": int(recovery_edges[index]),
                "x_right": int(recovery_edges[index + 1]),
                "n_eligible": int(recovery_eligible[index]),
                "n_success": int(recovery_success[index]),
            }
        )
    for index in range(len(nn_edges) - 1):
        rows.append(
            {
                "tracker": tracker,
                "seq": sequence,
                "profile": "strict_continuity",
                "bin_index": index,
                "x_left": float(nn_edges[index]),
                "x_right": float(nn_edges[index + 1]),
                "n_eligible": int(continuity_eligible[index]),
                "n_success": int(continuity_success[index]),
            }
        )
    atomic_csv_gz(destination, pd.DataFrame(rows))
    return {
        "tracker": tracker,
        "seq": sequence,
        "status": "computed",
        "hota_event_status": event_status,
        "profile_status": "computed",
        "assignment_status": (
            "computed" if assignment_needed else "cached"
        ),
    }


def run_jobs(
    function,
    jobs: list[dict[str, Any]],
    workers: int,
    description: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    print(f"{description}: {len(jobs)} jobs")
    if workers <= 1:
        return [function(job) for job in jobs]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, job): job for job in jobs}
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed == len(futures) or completed % max(1, len(futures) // 10) == 0:
                print(f"  {description}: {completed}/{len(futures)}")
    return results


def aggregate_profiles(
    *,
    profiles_module,
    profile_dir: Path,
    trackers: list[str],
    sequences: list[str],
    fps: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci_lo: float,
    ci_hi: float,
) -> pd.DataFrame:
    """Aggregate event-weighted rates and sequence-cluster intervals."""
    rows: list[dict[str, Any]] = []
    for tracker in trackers:
        paths = [
            profile_dir / safe_name(tracker) / f"{sequence}.csv.gz"
            for sequence in sequences
        ]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing direct-profile cache for {tracker}: {missing[0]}"
            )
        frame = pd.concat(
            [pd.read_csv(path) for path in paths],
            ignore_index=True,
        )
        for (profile, bin_index), subset in frame.groupby(
            ["profile", "bin_index"],
            sort=True,
        ):
            left = float(subset["x_left"].iloc[0])
            right = float(subset["x_right"].iloc[0])
            eligible = int(subset["n_eligible"].sum())
            success = int(subset["n_success"].sum())
            rate = success / eligible if eligible else float("nan")
            interval_lo, interval_hi = profiles_module.cluster_bootstrap_interval(
                subset[["seq", "n_eligible", "n_success"]],
                replicates=bootstrap_replicates,
                q_lo=ci_lo,
                q_hi=ci_hi,
                seed=profiles_module.deterministic_seed(
                    bootstrap_seed,
                    tracker,
                    profile,
                    int(bin_index),
                ),
            )
            if profile == "gap_prediction":
                x = left
            elif profile == "post_gap_recovery":
                # Half-open frame bins [left,right); place a numeric marker at
                # the midpoint of the represented integer gap lengths.
                x = 0.5 * (left + right - 1.0) / fps
            else:
                x = 0.5 * (left + right)
            rows.append(
                {
                    "tracker": tracker,
                    "label": label(tracker),
                    "profile": profile,
                    "bin_index": int(bin_index),
                    "x": x,
                    "x_left": left,
                    "x_right": right,
                    "n_eligible": eligible,
                    "n_success": success,
                    "success_rate": rate,
                    "success_percent": 100.0 * rate,
                    "ci_lo": interval_lo,
                    "ci_hi": interval_hi,
                    "ci_lo_percent": 100.0 * interval_lo,
                    "ci_hi_percent": 100.0 * interval_hi,
                    "n_sequence_clusters": int(subset["seq"].nunique()),
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )
    return pd.DataFrame(rows)


def contiguous_frame_segments(frames: Iterable[int]) -> list[list[int]]:
    """Split sorted GT-occurrence frames into contiguous trajectory segments."""
    ordered = sorted({int(frame) for frame in frames})
    if not ordered:
        return []
    segments: list[list[int]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame == segments[-1][-1] + 1:
            segments[-1].append(frame)
        else:
            segments.append([frame])
    return segments


def first_assigned_track(
    assigned_flat: np.ndarray,
    frame_to_flat: list[dict[int, int]],
    gid: int,
    start_frame: int,
    end_frame: int,
) -> tuple[int | None, int | None]:
    """Return the first accepted tracker ID/frame in an inclusive interval."""
    for frame in range(int(start_frame), int(end_frame) + 1):
        flat = frame_to_flat[frame].get(int(gid))
        if flat is None:
            break
        tracker_id = int(assigned_flat[flat])
        if tracker_id >= 0:
            return frame, tracker_id
    return None, None


def first_stable_assigned_run(
    assigned_ids: np.ndarray,
    *,
    max_start_frames: int,
    validation_frames: int,
    forbidden_tracker_id: int | None = None,
) -> dict[str, int | None]:
    """Find the earliest fully validated constant-ID run.

    ``assigned_ids`` covers exactly ``max_start_frames + validation_frames -
    1`` consecutive GT-present frames beginning at the detector-support
    anchor. Missing accepted output is encoded as a negative ID. Constant-ID
    runs are scanned once, so a failed run is skipped in full rather than
    testing every frame inside it as another candidate start.
    """
    assigned_ids = np.asarray(assigned_ids, dtype=np.int64).reshape(-1)
    required_frames = max_start_frames + validation_frames - 1
    if len(assigned_ids) != required_frames:
        raise ValueError(
            "Stable-run input must contain exactly "
            f"{required_frames} frames, got {len(assigned_ids)}"
        )

    first_output_offsets = np.flatnonzero(assigned_ids >= 0)
    first_output_offset = (
        int(first_output_offsets[0]) if len(first_output_offsets) else None
    )
    first_output_id = (
        int(assigned_ids[first_output_offset])
        if first_output_offset is not None
        else None
    )

    run_count = 0
    candidate_run_count = 0
    forbidden_run_count = 0
    offset = 0
    while offset < required_frames:
        tracker_id = int(assigned_ids[offset])
        run_end = offset + 1
        while (
            run_end < required_frames
            and int(assigned_ids[run_end]) == tracker_id
        ):
            run_end += 1

        if tracker_id >= 0:
            run_count += 1
            if offset < max_start_frames:
                candidate_run_count += 1
                if (
                    forbidden_tracker_id is not None
                    and tracker_id == int(forbidden_tracker_id)
                ):
                    forbidden_run_count += 1
                elif run_end - offset >= validation_frames:
                    return {
                        "stable_start_offset": offset,
                        "stable_end_offset": offset + validation_frames - 1,
                        "stable_tracker_id": tracker_id,
                        "first_output_offset": first_output_offset,
                        "first_output_tracker_id": first_output_id,
                        "n_output_runs": run_count,
                        "n_candidate_runs": candidate_run_count,
                        "n_forbidden_candidate_runs": forbidden_run_count,
                    }
        offset = run_end

    return {
        "stable_start_offset": None,
        "stable_end_offset": None,
        "stable_tracker_id": None,
        "first_output_offset": first_output_offset,
        "first_output_tracker_id": first_output_id,
        "n_output_runs": run_count,
        "n_candidate_runs": candidate_run_count,
        "n_forbidden_candidate_runs": forbidden_run_count,
    }


def load_assignment_observation(
    *,
    path: Path,
    common: dict[str, np.ndarray],
    tracker: str,
    sequence: str,
) -> np.ndarray:
    """Load and structurally validate one reusable assignment cache."""
    payload = load_npz(path)
    required = {
        "cache_version",
        "num_timesteps",
        "frame_gt_offsets",
        "gt_internal_ids_flat",
        "frame_indices_flat",
        "assigned_tracker_orig_id_flat",
    }
    missing = required - set(payload)
    if missing:
        raise KeyError(
            f"{tracker}:{sequence}: assignment cache lacks {sorted(missing)}"
        )
    version = int(np.asarray(payload["cache_version"]).reshape(-1)[0])
    if version != ASSIGNMENT_CACHE_VERSION:
        raise RuntimeError(
            f"{tracker}:{sequence}: unsupported assignment cache version "
            f"{version}; expected {ASSIGNMENT_CACHE_VERSION}"
        )
    for key in (
        "frame_gt_offsets",
        "gt_internal_ids_flat",
        "frame_indices_flat",
    ):
        if not np.array_equal(payload[key], common[key]):
            raise RuntimeError(
                f"{tracker}:{sequence}: assignment/common mismatch for {key}"
            )
    assigned = np.asarray(
        payload["assigned_tracker_orig_id_flat"],
        dtype=np.int64,
    )
    if len(assigned) != len(common["gt_internal_ids_flat"]):
        raise RuntimeError(
            f"{tracker}:{sequence}: malformed flat assignment length"
        )
    return assigned


def build_initialization_tables(
    *,
    profiles_module,
    common_sequence_dir: Path,
    assignment_dir: Path,
    trackers: list[str],
    sequences: list[str],
    fps: float,
    max_start_frames: int,
    max_start_s: float,
    validation_frames: int,
    validation_s: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci_lo: float,
    ci_hi: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build common events and stable-establishment latency curves.

    A tracker succeeds at the earliest accepted output whose original tracker
    ID remains accepted for the focal GT at every frame of the complete
    validation horizon. The stable run must start in the first
    ``max_start_frames`` after detector support appears. Every common event
    therefore needs ``max_start_frames + validation_frames - 1`` consecutive
    GT-present frames from its anchor.

    Post-gap events are tracker-common. Every requested tracker must have an
    accepted pre-gap ID, be absent at the final detector-missing frame, and
    first return after the gap with a different ID. Its validated stable run
    must also differ from that tracker's pre-gap ID.
    """
    required_gt_frames = max_start_frames + validation_frames - 1

    event_columns = [
        "event_id",
        "seq",
        "event_type",
        "gt_internal_id",
        "gt_orig_id",
        "anchor_frame",
        "horizon_end_frame",
        "max_start_frames",
        "max_start_s",
        "max_start_time_s",
        "validation_frames",
        "validation_s",
        "validation_time_s",
        "required_gt_frames",
        "segment_start_frame",
        "segment_end_frame",
        "n_detector_supported_frames",
        "detector_support_fraction",
        "detector_gap_id",
        "gap_first_missing_frame",
        "gap_last_missing_frame",
        "gap_missing_frames",
        "pre_gap_frame",
    ]
    observation_columns = [
        "event_id",
        "seq",
        "event_type",
        "tracker",
        "anchor_frame",
        "first_output_frame",
        "first_output_latency_frames",
        "first_output_tracker_id",
        "stable_start_frame",
        "stable_end_frame",
        "latency_frames",
        "latency_seconds",
        "stably_established_within_horizon",
        "stable_tracker_id",
        "pre_gap_tracker_id",
        "n_output_runs_scanned",
        "n_candidate_runs_scanned",
        "n_forbidden_candidate_runs",
        "failure_reason",
    ]
    audit_columns = [
        "candidate_id",
        "seq",
        "event_type",
        "gt_internal_id",
        "gt_orig_id",
        "anchor_frame",
        "required_horizon_end_frame",
        "detector_gap_id",
        "eligible",
        "exclusion_reason",
        "affected_trackers",
    ]

    event_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for sequence in sequences:
        common = load_npz(common_sequence_dir / f"{sequence}.npz")
        required = {
            "num_timesteps",
            "frame_gt_offsets",
            "gt_internal_ids_flat",
            "gt_orig_ids_flat",
            "frame_indices_flat",
            "detector_present_flat",
            "detector_gap_id_flat",
        }
        missing = required - set(common)
        if missing:
            raise KeyError(f"{sequence}: common cache lacks {sorted(missing)}")

        n_frames = int(np.asarray(common["num_timesteps"]).reshape(-1)[0])
        offsets = np.asarray(common["frame_gt_offsets"], dtype=np.int64)
        gids_flat = np.asarray(common["gt_internal_ids_flat"], dtype=np.int32)
        orig_ids_flat = np.asarray(common["gt_orig_ids_flat"], dtype=np.int64)
        frame_flat = np.asarray(common["frame_indices_flat"], dtype=np.int32)
        detector_present = np.asarray(
            common["detector_present_flat"], dtype=bool
        )
        gap_id_flat = np.asarray(
            common["detector_gap_id_flat"], dtype=np.int64
        )
        if len(offsets) != n_frames + 1:
            raise RuntimeError(f"{sequence}: malformed frame offsets")

        frame_to_flat: list[dict[int, int]] = []
        frames_by_gid: dict[int, list[int]] = defaultdict(list)
        original_by_gid: dict[int, int] = {}
        for frame in range(n_frames):
            start_flat = int(offsets[frame])
            end_flat = int(offsets[frame + 1])
            mapping: dict[int, int] = {}
            for flat in range(start_flat, end_flat):
                gid = int(gids_flat[flat])
                mapping[gid] = flat
                frames_by_gid[gid].append(frame)
                original_by_gid.setdefault(gid, int(orig_ids_flat[flat]))
            frame_to_flat.append(mapping)

        assignments = {
            tracker: load_assignment_observation(
                path=assignment_dir / safe_name(tracker) / f"{sequence}.npz",
                common=common,
                tracker=tracker,
                sequence=sequence,
            )
            for tracker in trackers
        }
        segments_by_gid = {
            gid: contiguous_frame_segments(frames)
            for gid, frames in frames_by_gid.items()
        }

        def horizon_flats_for(gid: int, anchor: int) -> list[int] | None:
            horizon_end = anchor + required_gt_frames - 1
            if anchor < 0 or horizon_end >= n_frames:
                return None
            flats = [
                frame_to_flat[frame].get(gid)
                for frame in range(anchor, horizon_end + 1)
            ]
            if any(flat is None for flat in flats):
                return None
            return [int(flat) for flat in flats]

        def append_eligible_event(
            *,
            event_id: str,
            event_type: str,
            gid: int,
            anchor: int,
            segment: list[int],
            horizon_flats: list[int],
            detector_gap_id: int,
            gap_first_missing_frame: int,
            gap_last_missing_frame: int,
            gap_missing_frames: int,
            pre_gap_frame: int,
            pre_gap_ids: dict[str, int] | None,
        ) -> None:
            horizon_end = anchor + required_gt_frames - 1
            detector_count = int(
                np.sum(detector_present[np.asarray(horizon_flats, dtype=int)])
            )
            event_rows.append(
                {
                    "event_id": event_id,
                    "seq": sequence,
                    "event_type": event_type,
                    "gt_internal_id": gid,
                    "gt_orig_id": original_by_gid[gid],
                    "anchor_frame": anchor,
                    "horizon_end_frame": horizon_end,
                    "max_start_frames": max_start_frames,
                    "max_start_s": max_start_s,
                    "max_start_time_s": max_start_frames / fps,
                    "validation_frames": validation_frames,
                    "validation_s": validation_s,
                    "validation_time_s": validation_frames / fps,
                    "required_gt_frames": required_gt_frames,
                    "segment_start_frame": int(segment[0]),
                    "segment_end_frame": int(segment[-1]),
                    "n_detector_supported_frames": detector_count,
                    "detector_support_fraction": (
                        detector_count / required_gt_frames
                    ),
                    "detector_gap_id": detector_gap_id,
                    "gap_first_missing_frame": gap_first_missing_frame,
                    "gap_last_missing_frame": gap_last_missing_frame,
                    "gap_missing_frames": gap_missing_frames,
                    "pre_gap_frame": pre_gap_frame,
                }
            )

            flat_indices = np.asarray(horizon_flats, dtype=np.int64)
            for tracker in trackers:
                pre_gap_id = (
                    None
                    if pre_gap_ids is None
                    else int(pre_gap_ids[tracker])
                )
                assigned_ids = assignments[tracker][flat_indices]
                result = first_stable_assigned_run(
                    assigned_ids,
                    max_start_frames=max_start_frames,
                    validation_frames=validation_frames,
                    forbidden_tracker_id=pre_gap_id,
                )

                stable_offset = result["stable_start_offset"]
                stable_end_offset = result["stable_end_offset"]
                stable = stable_offset is not None
                first_output_offset = result["first_output_offset"]

                start_window_ids = assigned_ids[:max_start_frames]
                if stable:
                    failure_reason = "success"
                elif not np.any(start_window_ids >= 0):
                    failure_reason = "no_output_in_start_window"
                elif (
                    pre_gap_id is not None
                    and np.all(
                        start_window_ids[start_window_ids >= 0] == pre_gap_id
                    )
                ):
                    failure_reason = "only_pre_gap_id_in_start_window"
                else:
                    failure_reason = "no_continuous_same_id_validation"

                observation_rows.append(
                    {
                        "event_id": event_id,
                        "seq": sequence,
                        "event_type": event_type,
                        "tracker": tracker,
                        "anchor_frame": anchor,
                        "first_output_frame": (
                            anchor + int(first_output_offset)
                            if first_output_offset is not None
                            else np.nan
                        ),
                        "first_output_latency_frames": (
                            int(first_output_offset) + 1
                            if first_output_offset is not None
                            else np.nan
                        ),
                        "first_output_tracker_id": (
                            result["first_output_tracker_id"]
                            if first_output_offset is not None
                            else np.nan
                        ),
                        "stable_start_frame": (
                            anchor + int(stable_offset)
                            if stable
                            else np.nan
                        ),
                        "stable_end_frame": (
                            anchor + int(stable_end_offset)
                            if stable
                            else np.nan
                        ),
                        "latency_frames": (
                            int(stable_offset) + 1 if stable else np.nan
                        ),
                        "latency_seconds": (
                            (int(stable_offset) + 1) / fps
                            if stable
                            else np.nan
                        ),
                        "stably_established_within_horizon": bool(stable),
                        "stable_tracker_id": (
                            result["stable_tracker_id"]
                            if stable
                            else np.nan
                        ),
                        "pre_gap_tracker_id": (
                            pre_gap_id if pre_gap_id is not None else np.nan
                        ),
                        "n_output_runs_scanned": result["n_output_runs"],
                        "n_candidate_runs_scanned": result[
                            "n_candidate_runs"
                        ],
                        "n_forbidden_candidate_runs": result[
                            "n_forbidden_candidate_runs"
                        ],
                        "failure_reason": failure_reason,
                    }
                )

        # One genuine beginning per GT internal trajectory: its earliest
        # contiguous GT segment, anchored at that segment's first detector hit.
        for gid, segments in segments_by_gid.items():
            first_segment = segments[0]
            candidate_id = f"{sequence}|start|{gid}"
            supported_frames = [
                frame
                for frame in first_segment
                if detector_present[frame_to_flat[frame][gid]]
            ]
            if not supported_frames:
                audit_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "seq": sequence,
                        "event_type": "trajectory_start",
                        "gt_internal_id": gid,
                        "gt_orig_id": original_by_gid[gid],
                        "anchor_frame": np.nan,
                        "required_horizon_end_frame": np.nan,
                        "detector_gap_id": -1,
                        "eligible": False,
                        "exclusion_reason": (
                            "no_detector_support_in_first_gt_segment"
                        ),
                        "affected_trackers": "",
                    }
                )
                continue

            anchor = int(supported_frames[0])
            horizon_flats = horizon_flats_for(gid, anchor)
            if horizon_flats is None:
                audit_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "seq": sequence,
                        "event_type": "trajectory_start",
                        "gt_internal_id": gid,
                        "gt_orig_id": original_by_gid[gid],
                        "anchor_frame": anchor,
                        "required_horizon_end_frame": (
                            anchor + required_gt_frames - 1
                        ),
                        "detector_gap_id": -1,
                        "eligible": False,
                        "exclusion_reason": "insufficient_gt_horizon",
                        "affected_trackers": "",
                    }
                )
                continue

            append_eligible_event(
                event_id=candidate_id,
                event_type="trajectory_start",
                gid=gid,
                anchor=anchor,
                segment=first_segment,
                horizon_flats=horizon_flats,
                detector_gap_id=-1,
                gap_first_missing_frame=-1,
                gap_last_missing_frame=-1,
                gap_missing_frames=0,
                pre_gap_frame=-1,
                pre_gap_ids=None,
            )
            audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seq": sequence,
                    "event_type": "trajectory_start",
                    "gt_internal_id": gid,
                    "gt_orig_id": original_by_gid[gid],
                    "anchor_frame": anchor,
                    "required_horizon_end_frame": (
                        anchor + required_gt_frames - 1
                    ),
                    "detector_gap_id": -1,
                    "eligible": True,
                    "exclusion_reason": "eligible",
                    "affected_trackers": "",
                }
            )

        # Post-gap events are common to the complete requested tracker set.
        gap_ids = np.unique(gap_id_flat[gap_id_flat >= 0]).tolist()
        for gap_id in gap_ids:
            gap_flats = np.flatnonzero(gap_id_flat == int(gap_id))
            gap_flats = gap_flats[np.argsort(frame_flat[gap_flats])]
            if not len(gap_flats):
                continue
            gid_values = np.unique(gids_flat[gap_flats])
            if len(gid_values) != 1:
                raise RuntimeError(
                    f"{sequence}: detector gap {gap_id} spans multiple GT IDs"
                )
            gid = int(gid_values[0])
            first_missing = int(frame_flat[gap_flats[0]])
            last_missing = int(frame_flat[gap_flats[-1]])
            pre_frame = first_missing - 1
            anchor = last_missing + 1
            candidate_id = f"{sequence}|restart|{gid}|gap{int(gap_id)}"
            required_horizon_end = anchor + required_gt_frames - 1

            def reject(reason: str, affected: Iterable[str] = ()) -> None:
                audit_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "seq": sequence,
                        "event_type": "post_gap_reacquisition",
                        "gt_internal_id": gid,
                        "gt_orig_id": original_by_gid.get(gid, -1),
                        "anchor_frame": anchor,
                        "required_horizon_end_frame": required_horizon_end,
                        "detector_gap_id": int(gap_id),
                        "eligible": False,
                        "exclusion_reason": reason,
                        "affected_trackers": "|".join(affected),
                    }
                )

            if not (0 <= pre_frame < anchor < n_frames):
                reject("invalid_gap_bounds")
                continue
            pre_flat = frame_to_flat[pre_frame].get(gid)
            post_flat = frame_to_flat[anchor].get(gid)
            if (
                pre_flat is None
                or post_flat is None
                or not bool(detector_present[pre_flat])
                or not bool(detector_present[post_flat])
            ):
                reject("gap_not_detector_bounded")
                continue

            segment = next(
                (
                    candidate
                    for candidate in segments_by_gid[gid]
                    if anchor in candidate
                ),
                None,
            )
            if segment is None:
                reject("post_gap_gt_segment_missing")
                continue
            horizon_flats = horizon_flats_for(gid, anchor)
            if horizon_flats is None:
                reject("insufficient_gt_horizon")
                continue

            pre_ids = {
                tracker: int(assignments[tracker][pre_flat])
                for tracker in trackers
            }
            missing_pre = [
                tracker
                for tracker, tracker_id in pre_ids.items()
                if tracker_id < 0
            ]
            if missing_pre:
                reject("not_all_trackers_established_pre_gap", missing_pre)
                continue

            last_gap_flat = frame_to_flat[last_missing].get(gid)
            if last_gap_flat is None:
                reject("last_gap_gt_occurrence_missing")
                continue
            not_lost = [
                tracker
                for tracker in trackers
                if int(assignments[tracker][last_gap_flat]) >= 0
            ]
            if not_lost:
                reject("not_all_trackers_lost_before_return", not_lost)
                continue

            first_after = {
                tracker: first_assigned_track(
                    assignments[tracker],
                    frame_to_flat,
                    gid,
                    anchor,
                    int(segment[-1]),
                )
                for tracker in trackers
            }
            never_reacquired = [
                tracker
                for tracker, (first_frame, _) in first_after.items()
                if first_frame is None
            ]
            if never_reacquired:
                reject("not_all_trackers_reacquired", never_reacquired)
                continue
            reused_id = [
                tracker
                for tracker, (_, first_id) in first_after.items()
                if int(first_id) == pre_ids[tracker]
            ]
            if reused_id:
                reject("not_all_trackers_started_new_id", reused_id)
                continue

            append_eligible_event(
                event_id=candidate_id,
                event_type="post_gap_reacquisition",
                gid=gid,
                anchor=anchor,
                segment=segment,
                horizon_flats=horizon_flats,
                detector_gap_id=int(gap_id),
                gap_first_missing_frame=first_missing,
                gap_last_missing_frame=last_missing,
                gap_missing_frames=int(len(gap_flats)),
                pre_gap_frame=pre_frame,
                pre_gap_ids=pre_ids,
            )
            audit_rows.append(
                {
                    "candidate_id": candidate_id,
                    "seq": sequence,
                    "event_type": "post_gap_reacquisition",
                    "gt_internal_id": gid,
                    "gt_orig_id": original_by_gid[gid],
                    "anchor_frame": anchor,
                    "required_horizon_end_frame": required_horizon_end,
                    "detector_gap_id": int(gap_id),
                    "eligible": True,
                    "exclusion_reason": "eligible",
                    "affected_trackers": "",
                }
            )

    events = pd.DataFrame(event_rows, columns=event_columns)
    observations = pd.DataFrame(
        observation_rows, columns=observation_columns
    )
    audit = pd.DataFrame(audit_rows, columns=audit_columns)
    if not events.empty and events["event_id"].duplicated().any():
        raise RuntimeError("Duplicate initialization event IDs")
    if not observations.empty and observations.duplicated(
        subset=["event_id", "tracker"]
    ).any():
        raise RuntimeError("Duplicate tracker rows in initialization events")
    expected_rows = len(events) * len(trackers)
    if len(observations) != expected_rows:
        raise RuntimeError(
            "Initialization observation matrix is incomplete: "
            f"expected {expected_rows}, got {len(observations)}"
        )

    summary_rows: list[dict[str, Any]] = []
    for event_type in [
        "trajectory_start",
        "post_gap_reacquisition",
        "all",
    ]:
        type_observations = (
            observations
            if event_type == "all"
            else observations[observations["event_type"] == event_type]
        )
        for tracker in trackers:
            tracker_rows = type_observations[
                type_observations["tracker"] == tracker
            ].copy()
            latency = pd.to_numeric(
                tracker_rows["latency_frames"], errors="coerce"
            )
            for elapsed in range(1, max_start_frames + 1):
                tracker_rows["_success"] = (
                    np.isfinite(latency) & (latency <= elapsed)
                ).astype(np.int64)
                sequence_counts = (
                    tracker_rows.groupby("seq", as_index=False)
                    .agg(
                        n_eligible=("event_id", "size"),
                        n_success=("_success", "sum"),
                    )
                )
                eligible = int(len(tracker_rows))
                success = int(tracker_rows["_success"].sum())
                rate = success / eligible if eligible else float("nan")
                if eligible:
                    interval_lo, interval_hi = profiles_module.cluster_bootstrap_interval(
                        sequence_counts,
                        replicates=bootstrap_replicates,
                        q_lo=ci_lo,
                        q_hi=ci_hi,
                        seed=profiles_module.deterministic_seed(
                            bootstrap_seed,
                            "stable_initialization",
                            event_type,
                            tracker,
                            elapsed,
                        ),
                    )
                else:
                    interval_lo, interval_hi = float("nan"), float("nan")
                summary_rows.append(
                    {
                        "tracker": tracker,
                        "label": label(tracker),
                        "event_type": event_type,
                        "elapsed_frames": elapsed,
                        "elapsed_seconds": elapsed / fps,
                        "n_eligible": eligible,
                        "n_stably_established_by_time": success,
                        "stable_established_rate": rate,
                        "stable_established_percent": 100.0 * rate,
                        "ci_lo": interval_lo,
                        "ci_hi": interval_hi,
                        "ci_lo_percent": 100.0 * interval_lo,
                        "ci_hi_percent": 100.0 * interval_hi,
                        "n_sequence_clusters": int(
                            tracker_rows["seq"].nunique()
                        ),
                        "bootstrap_replicates": bootstrap_replicates,
                        "validation_frames": validation_frames,
                        "validation_s": validation_s,
                        "required_gt_frames": required_gt_frames,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    denominator_audit = summary.pivot_table(
        index=["event_type", "elapsed_frames", "elapsed_seconds"],
        columns="tracker",
        values="n_eligible",
        aggfunc="first",
    )
    if not denominator_audit.empty:
        inconsistent = denominator_audit[
            denominator_audit.nunique(axis=1, dropna=False) != 1
        ]
        if not inconsistent.empty:
            raise RuntimeError(
                "Initialization common-denominator check failed for "
                f"{len(inconsistent)} rows"
            )
    return events, observations, summary, audit, denominator_audit.reset_index()

def metric_definitions(
    initialization_max_start_frames: int,
    initialization_max_start_s: float,
    initialization_validation_frames: int,
    initialization_validation_s: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "panel": "a",
                "metric": "prediction_through_gap",
                "population": (
                    "Bounded detector-missing runs with GT present, shared "
                    "detector support immediately before/after, and the "
                    "configured consecutive detector-supported prehistory."
                ),
                "success": (
                    "An accepted tracker match exists before the gap and the "
                    "same original tracker ID is accepted at every missing "
                    "frame through the evaluated elapsed age."
                ),
            },
            {
                "panel": "b",
                "metric": "post_gap_identity_recovery",
                "population": "The same bounded gaps and prehistory as panel a.",
                "success": (
                    "The accepted tracker ID at the first detector-supported "
                    "post-gap frame equals the accepted pre-gap ID; tracker "
                    "output inside the gap is ignored."
                ),
            },
            {
                "panel": "c",
                "metric": "strict_1s_continuity",
                "population": (
                    "Non-overlapping 1 s chunks after an 8-frame warm-up, "
                    "inside GT-continuous intervals whose detector gaps are "
                    "at most 3 frames; p10 NN is computed from GT only."
                ),
                "success": (
                    "Every calendar frame in the chunk has an accepted "
                    "tracker match to the focal GT and all accepted matches "
                    "use exactly one original tracker ID."
                ),
            },
            {
                "panel": "d",
                "metric": "runtime_by_input_count",
                "population": "Recorded tracker-only frame_stats rows.",
                "success": (
                    "Median FPS with per-frame P10-P90 at each exact rounded "
                    "number of input detections; detector runtime is excluded."
                ),
            },
            {
                "panel": "initialization",
                "metric": "stable_trajectory_establishment_latency",
                "population": (
                    "A common set containing GT-trajectory beginnings and "
                    "post-detector-gap returns. From the detector-support "
                    "anchor, GT must remain present for "
                    f"{initialization_max_start_frames} + "
                    f"{initialization_validation_frames} - 1 frames. Post-gap "
                    "events additionally require every requested tracker to "
                    "have an accepted pre-gap ID, be absent on the final gap "
                    "frame, and first return with a different ID."
                ),
                "success": (
                    "Cumulative percentage whose earliest accepted output "
                    "starts by elapsed time k and then remains continuously "
                    "assigned to the GT with one unchanged tracker ID for "
                    f"{initialization_validation_s:g} s "
                    f"({initialization_validation_frames} frames). The start "
                    "must occur within "
                    f"{initialization_max_start_s:g} s "
                    f"({initialization_max_start_frames} frames). A post-gap "
                    "validated ID must differ from the pre-gap ID."
                ),
            },
            {
                "panel": "global_a",
                "metric": "official_combined_hota_vs_runtime",
                "population": "Complete test split.",
                "success": (
                    "Official count/TP-weighted combined HOTA from current "
                    "outputs; vertical weighted sequence P10-P90 and "
                    "horizontal per-frame FPS P10-P90."
                ),
            },
            {
                "panel": "global_b",
                "metric": "paired_hota_deficit",
                "population": "Complete test split paired by sequence.",
                "success": (
                    "Reference combined HOTA minus tracker combined HOTA; "
                    "whiskers are GT-occurrence-weighted P10-P90 of paired "
                    "per-sequence deficits."
                ),
            },
        ]
    )


def protocol_mismatches(
    existing: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare only settings that determine expensive cache contents."""
    mismatches: dict[str, dict[str, Any]] = {}
    for key, current_value in current.items():
        existing_value = existing.get(key, None)
        if existing_value != current_value:
            mismatches[key] = {
                "cached": existing_value,
                "requested": current_value,
            }
    return mismatches


def aggregation_settings_compatible(
    metadata: dict[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    ci_lo: float,
    ci_hi: float,
) -> bool:
    """Whether an existing aggregate profile table can be retained."""
    if not metadata:
        return False
    cached_ci = metadata.get("profile_ci_quantiles")
    cached_replicates = metadata.get("bootstrap_replicates")
    # Metadata without an explicit seed uses the fixed protocol default.
    cached_seed = metadata.get("bootstrap_seed", 20260727)
    return bool(
        cached_ci == [ci_lo, ci_hi]
        and cached_replicates == bootstrap_replicates
        and cached_seed == bootstrap_seed
    )


def collect_frame_stats(
    *,
    trackers_base_dir: Path,
    trackers: list[str],
    source_map: dict[str, str],
    split_name: str,
    q_lo: float,
    q_hi: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """Rescan and retain frame-level timing data for every tracker."""
    availability_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    raw_parts: list[pd.DataFrame] = []
    runtime_frames: dict[str, pd.DataFrame] = {}

    for tracker in trackers:
        source = source_map.get(tracker, tracker)
        exact_directory = (
            trackers_base_dir / tracker / split_name / "frame_stats"
        )
        source_directory = (
            trackers_base_dir / source / split_name / "frame_stats"
        )
        exact_files = (
            sorted(exact_directory.glob("*.csv"))
            if exact_directory.is_dir()
            else []
        )
        source_files = (
            sorted(source_directory.glob("*.csv"))
            if source_directory.is_dir()
            else []
        )

        tracker_parts: list[pd.DataFrame] = []
        usable_files = 0
        read_error_files = 0
        files_without_step_ms = 0
        for path in source_files:
            try:
                frame = pd.read_csv(path)
            except Exception as error:  # Keep other trackers usable.
                read_error_files += 1
                print(
                    f"WARNING: could not read frame_stats {path}: {error}",
                    file=sys.stderr,
                )
                continue
            frame.columns = [str(column).strip() for column in frame.columns]
            if "step_ms" not in frame.columns:
                files_without_step_ms += 1
                continue
            step_ms = pd.to_numeric(frame["step_ms"], errors="coerce")
            if "num_det_in" in frame.columns:
                num_det = pd.to_numeric(
                    frame["num_det_in"],
                    errors="coerce",
                )
            else:
                num_det = pd.Series(
                    np.nan,
                    index=frame.index,
                    dtype=float,
                )
            valid = np.isfinite(step_ms) & (step_ms > 0)
            if not np.any(valid):
                continue
            usable_files += 1
            valid_step_ms = step_ms[valid].to_numpy(float)
            tracker_parts.append(
                pd.DataFrame(
                    {
                        "tracker": tracker,
                        "label": label(tracker),
                        "runtime_source_tracker": source,
                        "sequence": path.stem,
                        "frame_stats_file": path.name,
                        "step_ms": valid_step_ms,
                        "fps": 1000.0 / valid_step_ms,
                        "num_det_in": num_det[valid].to_numpy(float),
                    }
                )
            )

        if tracker_parts:
            raw = pd.concat(tracker_parts, ignore_index=True)
            raw_parts.append(raw)
            compact = raw[["step_ms", "fps", "num_det_in"]].copy()
            runtime_frames[tracker] = compact
            fps_values = raw["fps"].to_numpy(float)
            summary_rows.append(
                {
                    "tracker": tracker,
                    "label": label(tracker),
                    "runtime_source_tracker": source,
                    "fps_median": qvalue(fps_values, 0.50),
                    "fps_qlo": qvalue(fps_values, q_lo),
                    "fps_qhi": qvalue(fps_values, q_hi),
                    "n_runtime_frames": int(len(raw)),
                }
            )
            status = "available_mapped" if source != tracker else "available"
            n_runtime_frames = int(len(raw))
            n_detection_count_rows = int(
                np.isfinite(raw["num_det_in"].to_numpy(float)).sum()
            )
        else:
            if not source_directory.is_dir():
                status = "directory_missing"
            elif not source_files:
                status = "no_csv_files"
            else:
                status = "no_usable_rows"
            n_runtime_frames = 0
            n_detection_count_rows = 0

        availability_rows.append(
            {
                "tracker": tracker,
                "label": label(tracker),
                "runtime_source_tracker": source,
                "uses_mapped_source": bool(source != tracker),
                "exact_frame_stats_dir": str(exact_directory),
                "exact_directory_exists": bool(exact_directory.is_dir()),
                "n_exact_csv_files": int(len(exact_files)),
                "frame_stats_dir": str(source_directory),
                "source_directory_exists": bool(source_directory.is_dir()),
                "n_source_csv_files": int(len(source_files)),
                "n_usable_csv_files": int(usable_files),
                "n_read_error_files": int(read_error_files),
                "n_files_without_step_ms": int(files_without_step_ms),
                "has_usable_frame_stats": bool(tracker_parts),
                "n_runtime_frames": n_runtime_frames,
                "n_rows_with_num_det_in": n_detection_count_rows,
                "status": status,
            }
        )

    availability_columns = [
        "tracker",
        "label",
        "runtime_source_tracker",
        "uses_mapped_source",
        "exact_frame_stats_dir",
        "exact_directory_exists",
        "n_exact_csv_files",
        "frame_stats_dir",
        "source_directory_exists",
        "n_source_csv_files",
        "n_usable_csv_files",
        "n_read_error_files",
        "n_files_without_step_ms",
        "has_usable_frame_stats",
        "n_runtime_frames",
        "n_rows_with_num_det_in",
        "status",
    ]
    summary_columns = [
        "tracker",
        "label",
        "runtime_source_tracker",
        "fps_median",
        "fps_qlo",
        "fps_qhi",
        "n_runtime_frames",
    ]
    raw_columns = [
        "tracker",
        "label",
        "runtime_source_tracker",
        "sequence",
        "frame_stats_file",
        "step_ms",
        "fps",
        "num_det_in",
    ]
    availability = pd.DataFrame(
        availability_rows,
        columns=availability_columns,
    )
    runtime_summary = pd.DataFrame(summary_rows, columns=summary_columns)
    raw_frame_stats = (
        pd.concat(raw_parts, ignore_index=True)
        if raw_parts
        else pd.DataFrame(columns=raw_columns)
    )
    return availability, runtime_summary, runtime_frames, raw_frame_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally compute JRDB/RAL tracker tables without plotting, "
            "including common stable-establishment latency curves."
        )
    )
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--trackers-base-dir", type=Path, required=True)
    parser.add_argument("--gt-folder", type=Path, required=True)
    parser.add_argument("--capability-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profiles-script",
        type=Path,
        default=Path(__file__).with_name(
            "profiles.py"
        ),
        help=(
            "profile helper script for canonical TrackEval/HOTA aggregation. "
            "No previously exported metric table is read from it."
        ),
    )
    parser.add_argument(
        "--hota-cache-script",
        type=Path,
        default=Path(__file__).with_name(
            "hota_cache.py"
        ),
    )
    parser.add_argument(
        "--trackers",
        type=str,
        required=True,
        help=(
            "Complete comma-separated tracker registry to retain in the "
            "output tables. Missing tracker caches are computed automatically."
        ),
    )
    parser.add_argument("--reference-tracker", type=str, required=True)
    parser.add_argument(
        "--recompute-trackers",
        type=str,
        default=None,
        help=(
            "Comma-separated trackers whose profile, HOTA, and assignment "
            "caches must be recomputed even when complete caches exist."
        ),
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help=(
            "Explicitly rebuild common chunks and every requested tracker. "
            "Normally unnecessary; required only after a protocol change."
        ),
    )
    parser.add_argument(
        "--frame-stats-source-map",
        type=str,
        default=None,
        help=(
            "Optional tracker=source mappings for deliberately shared timing "
            "data. Exact tracker timing is otherwise "
            "required and no family fallback is performed."
        ),
    )
    parser.add_argument("--split-to-eval", type=str, default="test")
    parser.add_argument("--tracker-sub-folder", type=str, default="data")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--success-iou-thr", type=float, default=0.30)
    parser.add_argument("--matchable-sim-thr", type=float, default=0.30)
    parser.add_argument("--pre-gap-visible-frames", type=int, default=8)
    parser.add_argument("--max-gap-age-s", type=float, default=1.0)
    parser.add_argument(
        "--recovery-gap-bin-edges-frames",
        type=str,
        default="1,4,7,10,13,16",
    )
    parser.add_argument(
        "--continuity-max-detector-gap-frames",
        type=int,
        default=3,
    )
    parser.add_argument("--continuity-warmup-frames", type=int, default=8)
    parser.add_argument("--continuity-chunk-s", type=float, default=1.0)
    parser.add_argument("--continuity-min-nn-frames", type=int, default=3)
    parser.add_argument(
        "--initialization-max-s",
        type=float,
        default=0.50,
        help=(
            "Maximum time after detector support in which a validated "
            "stable tracker run may begin. Converted with ceil(seconds*fps)."
        ),
    )
    parser.add_argument(
        "--initialization-validation-s",
        type=float,
        default=1.00,
        help=(
            "Continuous correct same-ID duration required after a candidate "
            "start. Converted with ceil(seconds*fps)."
        ),
    )
    parser.add_argument("--nn-min-m", type=float, default=0.0)
    parser.add_argument("--nn-max-m", type=float, default=1.0)
    parser.add_argument("--nn-bin-width-m", type=float, default=0.10)
    parser.add_argument(
        "--hota-spread-quantiles",
        type=str,
        default="0.10,0.90",
    )
    parser.add_argument(
        "--profile-ci-quantiles",
        type=str,
        default="0.025,0.975",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    parser.add_argument(
        "--runtime-min-frames-per-count",
        type=int,
        default=1,
    )
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trackers = list(dict.fromkeys(parse_csv_list(args.trackers)))
    if not trackers:
        raise ValueError("--trackers is empty")
    if args.reference_tracker not in trackers:
        trackers.append(args.reference_tracker)

    recompute_trackers = set(parse_csv_list(args.recompute_trackers))
    unknown_recompute = recompute_trackers - set(trackers)
    if unknown_recompute:
        raise ValueError(
            "Unknown --recompute-trackers: "
            + ", ".join(sorted(unknown_recompute))
        )
    source_map = parse_mapping(args.frame_stats_source_map)
    unknown_mapped_trackers = set(source_map) - set(trackers)
    if unknown_mapped_trackers:
        raise ValueError(
            "Frame-stats mappings have trackers outside --trackers: "
            + ", ".join(sorted(unknown_mapped_trackers))
        )

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.pre_gap_visible_frames < 1:
        raise ValueError("--pre-gap-visible-frames must be >=1")
    if args.continuity_max_detector_gap_frames < 0:
        raise ValueError("--continuity-max-detector-gap-frames must be >=0")
    if args.continuity_warmup_frames < 0:
        raise ValueError("--continuity-warmup-frames must be >=0")
    if args.continuity_min_nn_frames < 1:
        raise ValueError("--continuity-min-nn-frames must be >=1")
    if args.initialization_max_s <= 0:
        raise ValueError("--initialization-max-s must be positive")
    if args.initialization_validation_s <= 0:
        raise ValueError("--initialization-validation-s must be positive")
    if not (args.nn_max_m > args.nn_min_m >= 0):
        raise ValueError("Require 0 <= --nn-min-m < --nn-max-m")
    if args.nn_bin_width_m <= 0:
        raise ValueError("--nn-bin-width-m must be positive")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be >=0")
    if args.runtime_min_frames_per_count < 1:
        raise ValueError("--runtime-min-frames-per-count must be >=1")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be >=1")

    recovery_edges = parse_int_edges(args.recovery_gap_bin_edges_frames)
    max_gap_age_frames = int(round(args.max_gap_age_s * args.fps))
    if max_gap_age_frames < 1:
        raise ValueError("--max-gap-age-s is too short")
    if recovery_edges[0] != 1:
        raise ValueError("Recovery bins must start at frame 1")
    if recovery_edges[-1] - 1 > max_gap_age_frames:
        raise ValueError("Recovery bins extend past --max-gap-age-s")
    chunk_frames = int(math.floor(args.continuity_chunk_s * args.fps + 0.5))
    if chunk_frames < 2:
        raise ValueError("--continuity-chunk-s is too short")
    initialization_max_start_frames = int(
        math.ceil(args.initialization_max_s * args.fps - 1e-12)
    )
    initialization_validation_frames = int(
        math.ceil(args.initialization_validation_s * args.fps - 1e-12)
    )
    initialization_required_gt_frames = (
        initialization_max_start_frames
        + initialization_validation_frames
        - 1
    )

    n_nn_bins_float = (
        (args.nn_max_m - args.nn_min_m) / args.nn_bin_width_m
    )
    n_nn_bins = int(round(n_nn_bins_float))
    if not math.isclose(
        n_nn_bins_float,
        n_nn_bins,
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "NN range must be exactly divisible by --nn-bin-width-m"
        )
    nn_edges = np.linspace(
        args.nn_min_m,
        args.nn_max_m,
        n_nn_bins + 1,
    ).tolist()
    hota_q_lo, hota_q_hi = parse_float_pair(
        args.hota_spread_quantiles,
        "--hota-spread-quantiles",
    )
    ci_lo, ci_hi = parse_float_pair(
        args.profile_ci_quantiles,
        "--profile-ci-quantiles",
    )

    capability_dir = args.capability_cache_dir.resolve()
    common_dir = capability_dir / "common"
    common_sequence_dir = common_dir / "sequences"
    if not common_sequence_dir.is_dir():
        raise FileNotFoundError(
            "Missing capability profile common cache: "
            f"{common_sequence_dir}"
        )
    sequences = sorted(
        path.stem for path in common_sequence_dir.glob("*.npz")
    )
    if not sequences:
        raise RuntimeError(f"No sequence caches in {common_sequence_dir}")

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    chunk_dir = cache_dir / "common_chunks"
    profile_dir = cache_dir / "direct_profiles"
    hota_event_dir = cache_dir / "hota_events"
    assignment_dir = cache_dir / "assignment_observations"
    tables_dir = output_dir / "tables"
    for directory in (
        chunk_dir,
        profile_dir,
        hota_event_dir,
        assignment_dir,
        tables_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    profiles_script = args.profiles_script.resolve()
    hota_cache_script = args.hota_cache_script.resolve()
    profiles_module = load_module(profiles_script, "ral_profiles_main")

    capability_metadata_path = common_dir / "metadata.json"
    capability_common_signature = (
        json.loads(
            capability_metadata_path.read_text(encoding="utf-8")
        ).get("signature")
        if capability_metadata_path.exists()
        else None
    )
    protocol_settings = {
        "trackeval_root": str(args.trackeval_root.resolve()),
        "trackers_base_dir": str(args.trackers_base_dir.resolve()),
        "gt_folder": str(args.gt_folder.resolve()),
        "capability_cache_dir": str(capability_dir),
        "capability_common_signature": capability_common_signature,
        "sequences": sequences,
        "split_to_eval": args.split_to_eval,
        "tracker_sub_folder": args.tracker_sub_folder,
        "fps": args.fps,
        "success_iou_thr": args.success_iou_thr,
        "matchable_sim_thr": args.matchable_sim_thr,
        "pre_gap_visible_frames": args.pre_gap_visible_frames,
        "max_gap_age_frames": max_gap_age_frames,
        "recovery_edges_frames": recovery_edges,
        "continuity_max_detector_gap_frames": (
            args.continuity_max_detector_gap_frames
        ),
        "continuity_warmup_frames": args.continuity_warmup_frames,
        "continuity_chunk_frames": chunk_frames,
        "continuity_min_nn_frames": args.continuity_min_nn_frames,
        "nn_edges": nn_edges,
    }
    protocol_signature = stable_signature(protocol_settings)

    metadata_path = cache_dir / "computation_metadata.json"
    if metadata_path.exists():
        previous_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    else:
        previous_metadata = {}

    mismatches = (
        protocol_mismatches(previous_metadata, protocol_settings)
        if previous_metadata
        else {}
    )
    if mismatches and not args.recompute_all:
        details = "\n".join(
            f"  {key}: cached={value['cached']!r}, "
            f"requested={value['requested']!r}"
            for key, value in list(mismatches.items())[:10]
        )
        raise RuntimeError(
            "Existing expensive caches use a different evaluation protocol. "
            "Use a new --output-dir, or pass --recompute-all to explicitly "
            "replace all requested caches. Differences:\n"
            + details
        )

    force_all = bool(args.recompute_all)
    if force_all:
        recompute_trackers = set(trackers)

    chunk_status: list[dict[str, Any]] = []
    for sequence in sequences:
        chunk_status.append(
            build_common_chunks(
                sequence=sequence,
                common_path=common_sequence_dir / f"{sequence}.npz",
                output_path=chunk_dir / f"{sequence}.csv.gz",
                fps=args.fps,
                max_detector_gap_frames=(
                    args.continuity_max_detector_gap_frames
                ),
                warmup_frames=args.continuity_warmup_frames,
                chunk_frames=chunk_frames,
                min_nn_frames=args.continuity_min_nn_frames,
                force=force_all,
            )
        )
    atomic_csv(
        tables_dir / "common_chunk_build_status.csv",
        pd.DataFrame(chunk_status),
    )

    jobs: list[dict[str, Any]] = []
    sequence_plan_rows: list[dict[str, Any]] = []
    cached_status_rows: list[dict[str, Any]] = []
    profile_changed_trackers: set[str] = set()
    tracker_order = {tracker: index for index, tracker in enumerate(trackers)}

    for tracker in trackers:
        force_tracker = tracker in recompute_trackers
        for sequence in sequences:
            profile_path = (
                profile_dir
                / safe_name(tracker)
                / f"{sequence}.csv.gz"
            )
            hota_path = (
                hota_event_dir
                / safe_name(tracker)
                / f"{sequence}.npz"
            )
            assignment_path = (
                assignment_dir
                / safe_name(tracker)
                / f"{sequence}.npz"
            )
            profile_exists = profile_path.is_file()
            hota_exists = hota_path.is_file()
            assignment_exists = assignment_path.is_file()
            needs_job = bool(
                force_tracker
                or not profile_exists
                or not hota_exists
                or not assignment_exists
            )
            if force_tracker:
                action = "recompute_tracker"
            else:
                missing_parts = [
                    name
                    for name, exists in (
                        ("profile", profile_exists),
                        ("hota", hota_exists),
                        ("assignment", assignment_exists),
                    )
                    if not exists
                ]
                action = (
                    "compute_" + "_and_".join(missing_parts)
                    if missing_parts
                    else "reuse"
                )

            sequence_plan_rows.append(
                {
                    "tracker": tracker,
                    "seq": sequence,
                    "profile_cache_existed_before": profile_exists,
                    "hota_cache_existed_before": hota_exists,
                    "assignment_cache_existed_before": assignment_exists,
                    "recompute_requested": force_tracker,
                    "scheduled": needs_job,
                    "action": action,
                }
            )
            if force_tracker or not profile_exists:
                profile_changed_trackers.add(tracker)
            if not needs_job:
                cached_status_rows.append(
                    {
                        "tracker": tracker,
                        "seq": sequence,
                        "status": "cached",
                        "hota_event_status": "cached",
                        "profile_status": "cached",
                        "assignment_status": "cached",
                        "action": action,
                    }
                )
                continue

            jobs.append(
                {
                    "tracker": tracker,
                    "sequence": sequence,
                    "profiles_script": str(profiles_script),
                    "hota_cache_script": str(hota_cache_script),
                    "trackeval_root": str(args.trackeval_root.resolve()),
                    "trackers_base_dir": str(
                        args.trackers_base_dir.resolve()
                    ),
                    "gt_folder": str(args.gt_folder.resolve()),
                    "common_dir": str(common_dir),
                    "chunk_dir": str(chunk_dir),
                    "profile_dir": str(profile_dir),
                    "hota_event_dir": str(hota_event_dir),
                    "assignment_dir": str(assignment_dir),
                    "reuse_hota_events_from": (
                        None if force_tracker else str(capability_dir)
                    ),
                    "split_to_eval": args.split_to_eval,
                    "tracker_sub_folder": args.tracker_sub_folder,
                    "fps": args.fps,
                    "success_iou_thr": args.success_iou_thr,
                    "matchable_sim_thr": args.matchable_sim_thr,
                    "pre_gap_visible_frames": args.pre_gap_visible_frames,
                    "max_gap_age_frames": max_gap_age_frames,
                    "recovery_edges_frames": recovery_edges,
                    "nn_edges": nn_edges,
                    "force_tracker": force_tracker,
                }
            )

    sequence_plan = pd.DataFrame(sequence_plan_rows)
    cache_plan = (
        sequence_plan.groupby("tracker", sort=False)
        .agg(
            n_sequences=("seq", "size"),
            n_profile_caches_before=(
                "profile_cache_existed_before",
                "sum",
            ),
            n_hota_caches_before=("hota_cache_existed_before", "sum"),
            n_assignment_caches_before=(
                "assignment_cache_existed_before",
                "sum",
            ),
            n_jobs_scheduled=("scheduled", "sum"),
            recompute_requested=("recompute_requested", "max"),
        )
        .reset_index()
    )
    cache_plan["_order"] = cache_plan["tracker"].map(tracker_order)
    cache_plan = cache_plan.sort_values("_order").drop(columns="_order")
    atomic_csv(tables_dir / "tracker_cache_plan.csv", cache_plan)
    print("Tracker cache plan:")
    print(cache_plan.to_string(index=False))

    computed_status = run_jobs(
        _direct_tracker_sequence_worker,
        jobs,
        args.num_workers,
        "Missing/recomputed tracker-sequence results",
    )
    action_by_key = {
        (row["tracker"], row["seq"]): row["action"]
        for row in sequence_plan_rows
    }
    for row in computed_status:
        row["action"] = action_by_key[(row["tracker"], row["seq"])]
    all_status = pd.DataFrame(cached_status_rows + computed_status)
    if not all_status.empty:
        all_status["_order"] = all_status["tracker"].map(tracker_order)
        all_status = (
            all_status.sort_values(["_order", "seq"])
            .drop(columns="_order")
            .reset_index(drop=True)
        )
    atomic_csv(
        tables_dir / "tracker_sequence_build_status.csv",
        all_status,
    )

    for tracker in trackers:
        for sequence in sequences:
            profile_path = (
                profile_dir
                / safe_name(tracker)
                / f"{sequence}.csv.gz"
            )
            hota_path = (
                hota_event_dir
                / safe_name(tracker)
                / f"{sequence}.npz"
            )
            assignment_path = (
                assignment_dir
                / safe_name(tracker)
                / f"{sequence}.npz"
            )
            if not profile_path.is_file():
                raise FileNotFoundError(
                    f"Missing profile cache after computation: {profile_path}"
                )
            if not hota_path.is_file():
                raise FileNotFoundError(
                    f"Missing HOTA cache after computation: {hota_path}"
                )
            if not assignment_path.is_file():
                raise FileNotFoundError(
                    "Missing assignment cache after computation: "
                    f"{assignment_path}"
                )

    profiles_path = tables_dir / "direct_profile_summary.csv"
    can_reuse_aggregate = bool(
        profiles_path.is_file()
        and aggregation_settings_compatible(
            previous_metadata,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
    )
    existing_profiles = (
        pd.read_csv(profiles_path)
        if can_reuse_aggregate
        else pd.DataFrame()
    )
    existing_profile_trackers = (
        set(existing_profiles["tracker"].astype(str))
        if not existing_profiles.empty
        and "tracker" in existing_profiles.columns
        else set()
    )
    aggregate_trackers = [
        tracker
        for tracker in trackers
        if (
            not can_reuse_aggregate
            or tracker in profile_changed_trackers
            or tracker not in existing_profile_trackers
        )
    ]
    retained_trackers = [
        tracker for tracker in trackers if tracker not in aggregate_trackers
    ]
    retained_profiles = (
        existing_profiles[
            existing_profiles["tracker"].isin(retained_trackers)
        ].copy()
        if retained_trackers
        else pd.DataFrame()
    )
    computed_profiles = (
        aggregate_profiles(
            profiles_module=profiles_module,
            profile_dir=profile_dir,
            trackers=aggregate_trackers,
            sequences=sequences,
            fps=args.fps,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
        )
        if aggregate_trackers
        else pd.DataFrame()
    )
    profile_parts = [
        frame
        for frame in (retained_profiles, computed_profiles)
        if not frame.empty
    ]
    if not profile_parts:
        raise RuntimeError("No aggregate profile rows were produced")
    profiles = pd.concat(profile_parts, ignore_index=True)
    profiles["label"] = profiles["tracker"].map(label)
    profiles["_order"] = profiles["tracker"].map(tracker_order)
    profiles = (
        profiles.sort_values(["_order", "profile", "bin_index"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    atomic_csv(profiles_path, profiles)

    denominator_check = profiles.pivot_table(
        index=["profile", "bin_index"],
        columns="tracker",
        values="n_eligible",
        aggfunc="first",
    )
    if not denominator_check.empty:
        row_min = denominator_check.min(axis=1, skipna=False)
        row_max = denominator_check.max(axis=1, skipna=False)
        if not np.all(row_min.to_numpy() == row_max.to_numpy()):
            inconsistent = denominator_check[
                row_min.to_numpy() != row_max.to_numpy()
            ]
            raise RuntimeError(
                "Tracker-independent denominator check failed for: "
                + ", ".join(
                    f"{profile}[{int(index)}]"
                    for profile, index in inconsistent.index.tolist()[:10]
                )
            )
    atomic_csv(
        tables_dir / "common_denominator_audit.csv",
        denominator_check.reset_index(),
    )
    for profile, filename in (
        ("gap_prediction", "prediction_through_gap.csv"),
        ("post_gap_recovery", "post_gap_identity_recovery.csv"),
        ("strict_continuity", "strict_continuity_1s_p10_gt_nn.csv"),
    ):
        atomic_csv(
            tables_dir / filename,
            profiles[profiles["profile"] == profile].copy(),
        )

    strict_support = profiles[
        (profiles["tracker"] == args.reference_tracker)
        & (profiles["profile"] == "strict_continuity")
    ][
        [
            "bin_index",
            "x_left",
            "x_right",
            "x",
            "n_eligible",
            "n_success",
        ]
    ].copy()
    atomic_csv(
        tables_dir / "strict_continuity_common_support.csv",
        strict_support,
    )

    (
        initialization_events,
        initialization_observations,
        initialization_summary,
        initialization_audit,
        initialization_denominator_audit,
    ) = build_initialization_tables(
        profiles_module=profiles_module,
        common_sequence_dir=common_sequence_dir,
        assignment_dir=assignment_dir,
        trackers=trackers,
        sequences=sequences,
        fps=args.fps,
        max_start_frames=initialization_max_start_frames,
        max_start_s=args.initialization_max_s,
        validation_frames=initialization_validation_frames,
        validation_s=args.initialization_validation_s,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )
    atomic_csv_gz(
        tables_dir / "initialization_common_events.csv.gz",
        initialization_events,
    )
    atomic_csv_gz(
        tables_dir / "initialization_tracker_events.csv.gz",
        initialization_observations,
    )
    atomic_csv(
        tables_dir / "initialization_latency_summary.csv",
        initialization_summary,
    )
    atomic_csv_gz(
        tables_dir / "initialization_event_audit.csv.gz",
        initialization_audit,
    )
    initialization_audit_summary = (
        initialization_audit.groupby(
            ["event_type", "eligible", "exclusion_reason"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "n_candidates"})
    )
    atomic_csv(
        tables_dir / "initialization_event_audit_summary.csv",
        initialization_audit_summary,
    )
    atomic_csv(
        tables_dir / "initialization_common_denominator_audit.csv",
        initialization_denominator_audit,
    )

    (
        frame_stats_availability,
        runtime_summary,
        runtime_frames,
        raw_frame_stats,
    ) = collect_frame_stats(
        trackers_base_dir=args.trackers_base_dir.resolve(),
        trackers=trackers,
        source_map=source_map,
        split_name=args.split_to_eval,
        q_lo=hota_q_lo,
        q_hi=hota_q_hi,
    )
    atomic_csv(
        tables_dir / "frame_stats_availability.csv",
        frame_stats_availability,
    )
    atomic_csv(tables_dir / "runtime_summary.csv", runtime_summary)
    atomic_csv_gz(
        tables_dir / "runtime_frame_stats.csv.gz",
        raw_frame_stats,
    )

    runtime_by_count = profiles_module.aggregate_runtime_by_detection_count(
        runtime_frames,
        trackers,
        args.runtime_min_frames_per_count,
    )
    runtime_by_count_columns = [
        "tracker",
        "label",
        "num_det_in",
        "fps_median",
        "fps_q10",
        "fps_q90",
        "n_frames",
    ]
    if runtime_by_count.empty:
        runtime_by_count = pd.DataFrame(
            columns=runtime_by_count_columns
        )
    else:
        runtime_by_count["label"] = runtime_by_count["tracker"].map(label)
        runtime_by_count = runtime_by_count[runtime_by_count_columns]
    atomic_csv(
        tables_dir / "runtime_vs_detection_count.csv",
        runtime_by_count,
    )

    hota_summary, hota_gap, per_sequence_hota = profiles_module.build_hota_tables(
        hota_event_dir,
        trackers,
        sequences,
        args.reference_tracker,
        hota_q_lo,
        hota_q_hi,
    )
    for frame in (hota_summary, hota_gap, per_sequence_hota):
        if "label" in frame.columns:
            frame["label"] = frame["tracker"].map(label)
    global_table = profiles_module.merge_hota_runtime(
        hota_summary,
        runtime_summary,
    )
    atomic_csv(tables_dir / "global_hota_metrics.csv", hota_summary)
    atomic_csv(tables_dir / "hota_per_sequence.csv", per_sequence_hota)
    atomic_csv(tables_dir / "hota_gap_to_reference.csv", hota_gap)
    atomic_csv(tables_dir / "global_hota_runtime.csv", global_table)
    atomic_csv(
        tables_dir / "metric_definitions.csv",
        metric_definitions(
            initialization_max_start_frames,
            args.initialization_max_s,
            initialization_validation_frames,
            args.initialization_validation_s,
        ),
    )

    metadata_payload = {
        "script_version": SCRIPT_VERSION,
        "cache_protocol_version": CACHE_PROTOCOL_VERSION,
        **protocol_settings,
        "protocol_signature": protocol_signature,
        "trackers": trackers,
        "reference_tracker": args.reference_tracker,
        "frame_stats_source_map": source_map,
        "hota_spread_quantiles": [hota_q_lo, hota_q_hi],
        "profile_ci_quantiles": [ci_lo, ci_hi],
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "runtime_min_frames_per_count": (
            args.runtime_min_frames_per_count
        ),
        "initialization_max_s": args.initialization_max_s,
        "initialization_max_start_frames": (
            initialization_max_start_frames
        ),
        "initialization_validation_s": args.initialization_validation_s,
        "initialization_validation_frames": (
            initialization_validation_frames
        ),
        "initialization_required_gt_frames": (
            initialization_required_gt_frames
        ),
        "initialization_tracker_registry_signature": stable_signature(
            {"trackers": trackers}
        ),
    }
    atomic_json(metadata_path, metadata_payload)

    available_runtime_trackers = frame_stats_availability.loc[
        frame_stats_availability["has_usable_frame_stats"],
        "tracker",
    ].astype(str).tolist()
    run_summary = {
        "script_version": SCRIPT_VERSION,
        "protocol_signature": protocol_signature,
        "output_dir": str(output_dir),
        "n_sequences": len(sequences),
        "trackers": trackers,
        "reference_tracker": args.reference_tracker,
        "recompute_trackers": sorted(recompute_trackers),
        "n_tracker_sequence_jobs_scheduled": int(len(jobs)),
        "aggregate_profiles_recomputed_for": aggregate_trackers,
        "aggregate_profiles_reused_for": retained_trackers,
        "frame_stats_available_for": available_runtime_trackers,
        "frame_stats_missing_for": [
            tracker
            for tracker in trackers
            if tracker not in available_runtime_trackers
        ],
        "initialization_max_s": args.initialization_max_s,
        "initialization_max_start_frames": (
            initialization_max_start_frames
        ),
        "initialization_validation_s": args.initialization_validation_s,
        "initialization_validation_frames": (
            initialization_validation_frames
        ),
        "initialization_required_gt_frames": (
            initialization_required_gt_frames
        ),
        "initialization_common_events": int(len(initialization_events)),
        "initialization_trajectory_start_events": int(
            np.sum(
                initialization_events["event_type"]
                == "trajectory_start"
            )
        ),
        "initialization_post_gap_reacquisition_events": int(
            np.sum(
                initialization_events["event_type"]
                == "post_gap_reacquisition"
            )
        ),
        "tables_dir": str(tables_dir),
    }
    atomic_json(
        output_dir / "computation_run_summary.json",
        run_summary,
    )

    print()
    print("Frame-stats availability:")
    print(
        frame_stats_availability[
            [
                "tracker",
                "runtime_source_tracker",
                "status",
                "n_source_csv_files",
                "n_runtime_frames",
            ]
        ].to_string(index=False)
    )
    print()
    print(json.dumps(run_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
