"""Shared sequence scheduler for multi-tracker and multi-variant inference."""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

from tracker_eval.data.jrdb_io import (
    list_sequence_jsons,
    sequence_name_from_json_filename,
)
from tracker_eval.runner.run_sequence import SequenceRunStats
from tracker_eval.runner.run_split import (
    SplitRunSummary,
    _aggregate_sequence_stats,
    _run_one_sequence_worker,
    _safe_mkdir,
    _write_csv,
    _write_json,
)


DEFAULT_WORKERS_PER_TRACKER: Dict[str, int] = {
    "pedreftrack": 12,
    "elptnet": 2,
    "cbmot": 6,
    "fastpoly": 6,
    "ab3dmot": 6,
    "gnnpmb": 6,
    "simpletrack": 6,
}


@dataclass(frozen=True)
class BatchRunRequest:
    """One tracker/configuration/split output run expanded into sequence jobs."""

    tracker_key: str
    tracker_spec: Dict[str, Any]
    tracker_name: str
    split_root: str
    split_name: str
    out_root: str
    detections_subdir: str = "detections_3D"
    labels_subdir: str = "labels_3d"
    use_gt_if_available: bool = True
    warmup_steps: int = 0
    limit_sequences: Optional[int] = None
    include_sequences: Optional[Sequence[str]] = None
    exclude_sequences: Optional[Sequence[str]] = None
    kitti_use_score: bool = True
    tracker_subfolder: str = "data"
    skip_existing_kitti: bool = True
    global_coords: bool = False
    odometry_root: str = ""


@dataclass
class _PreparedRun:
    request: BatchRunRequest
    tracker_dir: Path
    kitti_dir: Path
    detections_dir: Path
    num_sequences: int
    rows: List[Dict[str, Any]]
    jobs: List[Dict[str, Any]]


def _prepare_run(index: int, request: BatchRunRequest) -> _PreparedRun:
    split_root = Path(request.split_root)
    detections_dir = split_root / request.detections_subdir
    if not detections_dir.exists():
        raise FileNotFoundError(
            f"Detections directory not found: {detections_dir}"
        )
    if request.global_coords and not str(request.odometry_root).strip():
        raise ValueError("global_coords requires odometry_root.")

    tracker_dir = (
        Path(request.out_root)
        / request.tracker_name
        / request.split_name
    )
    kitti_dir = tracker_dir / request.tracker_subfolder
    _safe_mkdir(kitti_dir)

    sequences = [
        sequence_name_from_json_filename(path)
        for path in list_sequence_jsons(str(detections_dir))
    ]
    sequences.sort()
    if request.include_sequences is not None:
        include = set(request.include_sequences)
        sequences = [name for name in sequences if name in include]
    if request.exclude_sequences is not None:
        exclude = set(request.exclude_sequences)
        sequences = [name for name in sequences if name not in exclude]
    if request.limit_sequences is not None:
        sequences = sequences[: int(request.limit_sequences)]

    rows: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []
    for sequence in sequences:
        det_path = detections_dir / f"{sequence}.json"
        out_path = kitti_dir / f"{sequence}.txt"
        if request.skip_existing_kitti and out_path.exists():
            rows.append(
                {
                    "seq_name": sequence,
                    "status": "skipped_existing",
                    "out_kitti_txt": str(out_path),
                    "frame_stats_csv": "",
                }
            )
            continue
        jobs.append(
            {
                "run_id": int(index),
                "tracker_key": request.tracker_key,
                "seq_name": sequence,
                "det_json_path": str(det_path),
                "det_size_bytes": int(det_path.stat().st_size),
                "out_kitti_txt": str(out_path),
                "kitti_use_score": request.kitti_use_score,
                "warmup_steps": request.warmup_steps,
                "tracker_spec": request.tracker_spec,
                "split_name": request.split_name,
                "tracker_name": request.tracker_name,
                "split_root": request.split_root,
                "labels_subdir": request.labels_subdir,
                "use_gt_if_available": request.use_gt_if_available,
                "global_coords": request.global_coords,
                "odometry_root": request.odometry_root,
            }
        )

    # Start expensive crowded sequences early. Because all runs share this
    # scheduler, finishing short sequences can immediately release a worker to
    # another mode, variant, or tracker.
    jobs.sort(
        key=lambda job: (
            -int(job["det_size_bytes"]),
            str(job["seq_name"]),
        )
    )
    return _PreparedRun(
        request=request,
        tracker_dir=tracker_dir,
        kitti_dir=kitti_dir,
        detections_dir=detections_dir,
        num_sequences=len(sequences),
        rows=rows,
        jobs=jobs,
    )


def _summary_for_run(prepared: _PreparedRun) -> SplitRunSummary:
    prepared.rows.sort(key=lambda row: str(row.get("seq_name", "")))
    stats: List[SequenceRunStats] = []
    for row in prepared.rows:
        if row.get("status") != "ok":
            continue
        stats.append(
            SequenceRunStats(
                seq_name=str(row.get("seq_name", "")),
                num_frames=int(row.get("num_frames", 0)),
                total_time_s=0.0,
                fps=0.0,
                mean_step_ms=0.0,
                p50_step_ms=0.0,
                p90_step_ms=0.0,
                p99_step_ms=0.0,
            )
        )
    aggregate = _aggregate_sequence_stats(stats)
    request = prepared.request
    summary = SplitRunSummary(
        split_name=request.split_name,
        tracker_name=request.tracker_name,
        num_sequences=prepared.num_sequences,
        num_frames_total=int(aggregate.get("num_frames_total", 0)),
        warmup_steps=request.warmup_steps,
        sequences=prepared.rows,
        aggregate=aggregate,
        io={
            "split_root": request.split_root,
            "detections_dir": str(prepared.detections_dir),
            "out_root": request.out_root,
            "tracker_dir": str(prepared.tracker_dir),
            "kitti_dir": str(prepared.kitti_dir),
        },
    )
    _write_json(
        prepared.tracker_dir / "runtime_summary.json",
        asdict(summary),
    )
    _write_csv(
        prepared.tracker_dir / "runtime_summary.csv",
        prepared.rows,
        fieldnames=[
            "seq_name",
            "status",
            "num_frames",
            "fps",
            "mean_step_ms",
            "p50_step_ms",
            "p90_step_ms",
            "p99_step_ms",
            "detections_json",
            "out_kitti_txt",
            "frame_stats_csv",
        ],
    )
    return summary


def run_batch(
    requests: Sequence[BatchRunRequest],
    *,
    num_workers: int = 0,
    workers_per_tracker: Optional[Mapping[str, int]] = None,
    start_method: str = "spawn",
    verbose: bool = True,
) -> List[SplitRunSummary]:
    """Run all sequence jobs through one process pool without run barriers."""
    if not requests:
        return []
    if start_method not in {"spawn", "fork", "forkserver"}:
        raise ValueError(
            "start_method must be spawn, fork, or forkserver."
        )

    limits = dict(DEFAULT_WORKERS_PER_TRACKER)
    if workers_per_tracker is not None:
        limits.update(
            {
                str(key): int(value)
                for key, value in workers_per_tracker.items()
            }
        )
    tracker_order = list(
        dict.fromkeys(request.tracker_key for request in requests)
    )
    for tracker_key in tracker_order:
        if tracker_key not in limits:
            raise ValueError(
                f"No worker limit configured for {tracker_key!r}."
            )
        if limits[tracker_key] < 1:
            raise ValueError(
                f"Worker limit for {tracker_key!r} must be >= 1."
            )

    prepared = [
        _prepare_run(index, request)
        for index, request in enumerate(requests)
    ]
    merged_jobs: Dict[str, List[Dict[str, Any]]] = {
        key: [] for key in tracker_order
    }
    for run in prepared:
        merged_jobs[run.request.tracker_key].extend(run.jobs)
    queues: Dict[str, Deque[Dict[str, Any]]] = {}
    for key in tracker_order:
        merged_jobs[key].sort(
            key=lambda job: (
                -int(job["det_size_bytes"]),
                str(job["seq_name"]),
                int(job["run_id"]),
            )
        )
        queues[key] = deque(merged_jobs[key])
    total_jobs = sum(len(run.jobs) for run in prepared)
    if total_jobs == 0:
        return [_summary_for_run(run) for run in prepared]

    default_total = sum(limits[key] for key in tracker_order)
    worker_total = int(num_workers) if int(num_workers) > 0 else default_total
    worker_total = max(1, min(worker_total, total_jobs))
    active_by_tracker = {key: 0 for key in tracker_order}
    completed = 0
    cursor = 0

    if verbose:
        print(
            f"[tracker_eval] Shared scheduler: {total_jobs} sequence job(s), "
            f"{worker_total} process(es), start_method={start_method}"
        )
        print(
            "[tracker_eval] Per-tracker limits: "
            + ", ".join(
                f"{key}={limits[key]}" for key in tracker_order
            )
        )

    context = mp.get_context(start_method)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_total,
        mp_context=context,
    ) as executor:
        active: Dict[
            concurrent.futures.Future[Dict[str, Any]],
            tuple[int, str],
        ] = {}

        def submit_available() -> None:
            nonlocal cursor
            while len(active) < worker_total:
                chosen: Optional[str] = None
                for offset in range(len(tracker_order)):
                    key = tracker_order[
                        (cursor + offset) % len(tracker_order)
                    ]
                    if (
                        queues[key]
                        and active_by_tracker[key] < limits[key]
                    ):
                        chosen = key
                        cursor = (
                            cursor + offset + 1
                        ) % len(tracker_order)
                        break
                if chosen is None:
                    return
                job = queues[chosen].popleft()
                future = executor.submit(_run_one_sequence_worker, job)
                active[future] = (int(job["run_id"]), chosen)
                active_by_tracker[chosen] += 1

        submit_available()
        try:
            while active:
                done, _ = concurrent.futures.wait(
                    active,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    run_id, tracker_key = active.pop(future)
                    active_by_tracker[tracker_key] -= 1
                    row = future.result()
                    prepared[run_id].rows.append(row)
                    completed += 1
                    if verbose:
                        status = str(row.get("status", "unknown"))
                        print(
                            f"[tracker_eval] {completed}/{total_jobs} "
                            f"{prepared[run_id].request.tracker_name}/"
                            f"{row.get('seq_name', '')}: {status}"
                        )
                        if status == "error":
                            print(row.get("traceback", row.get("error", "")))
                submit_available()
        except KeyboardInterrupt:
            for future in active:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            processes = getattr(executor, "_processes", {})
            for process in list(processes.values()):
                if process.is_alive():
                    process.terminate()
            time.sleep(0.2)
            for process in list(processes.values()):
                if process.is_alive():
                    process.kill()
            raise

    summaries = [_summary_for_run(run) for run in prepared]
    if verbose:
        print(
            f"[tracker_eval] Completed {len(summaries)} run(s). "
            "Parallel timing and per-frame profiling were disabled."
        )
    return summaries
