#!/usr/bin/env python3
"""Fast, incremental evaluation of the ten RA-L pseudo-detection variants.

Tracker input and result folders use the runner's single canonical naming
order, for example ``fastpoly__global_instability_L1`` and
``pedreftrack__global_no_gt_dropout_L2``. Input folders are exposed to
TrackEval through temporary aliases without renaming the saved results.

Only the canonical protocol is considered: Clean, three Dropout levels,
three Instability levels, and three symmetric Combined levels. Confuser
folders outside the allowlist are ignored. Every
condition is evaluated against the same canonical global JRDB ground truth.

It calls the official TrackEval dataset preprocessing, HOTA metric, and
sequence-combination functions directly. HOTA, CLEAR (including IDSW and
fragmentation), and Count results are cached per tracker/sequence.

JRDB's optional GT density, speed, matchability, and event-statistics
calculation is disabled because none of those side products is read here.
This does not change GT/tracker preprocessing, 3D similarity, metric matching,
or official sequence aggregation.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Avoid nested BLAS/OpenMP parallelism inside the process-level worker pool.
# This must happen before NumPy/SciPy is imported by this module or TrackEval.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import yaml


SCRIPT_VERSION = "tracker_eval_protocol"
CACHE_VERSION = 1

VARIANT_ORDER = (
    "clean",
    "dropout_L1",
    "dropout_L2",
    "dropout_L3",
    "instability_L1",
    "instability_L2",
    "instability_L3",
    "combined_L1",
    "combined_L2",
    "combined_L3",
)


@dataclass(frozen=True)
class VariantDefinition:
    name: str
    failure_mode: str


@dataclass(frozen=True)
class EvaluationJob:
    tracker_base: str
    input_name: str
    input_path: Path
    variant: str
    failure_mode: str
    gt_path: Path
    result_name: str
    result_path: Path
    recompute: bool


def _split_cli_values(values: Optional[Sequence[str]]) -> List[str]:
    output: List[str] = []
    for value in values or []:
        output.extend(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    return output


def load_variant_definitions(
    spec_path: Path,
) -> Dict[str, VariantDefinition]:
    with spec_path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict):
        raise ValueError(f"Expected a YAML mapping in {spec_path}")

    definitions: Dict[str, VariantDefinition] = {}
    sweeps = spec.get("single_mode_sweeps", {})
    if not isinstance(sweeps, dict):
        raise ValueError("single_mode_sweeps must be a mapping")

    for raw_mode, raw_entries in sweeps.items():
        mode = str(raw_mode).strip()
        if not isinstance(raw_entries, list):
            raise ValueError(
                f"single_mode_sweeps.{mode} must be a list"
            )
        for raw_item in raw_entries:
            if not isinstance(raw_item, dict) or "name" not in raw_item:
                raise ValueError(
                    f"Invalid entry under single_mode_sweeps.{mode}"
                )
            item_name = str(raw_item["name"]).strip()
            variant = "clean" if mode == "clean" else (
                item_name
                if item_name.startswith(mode + "_")
                else f"{mode}_{item_name}"
            )
            definition = VariantDefinition(
                name=variant,
                failure_mode=mode,
            )
            if variant in definitions:
                raise ValueError(
                    f"Duplicate pseudo variant in spec: {variant}"
                )
            definitions[variant] = definition

    raw_combos = spec.get("combos", [])
    if not isinstance(raw_combos, list):
        raise ValueError("combos must be a list")
    for raw_item in raw_combos:
        if not isinstance(raw_item, dict) or "name" not in raw_item:
            raise ValueError("Invalid entry under combos")
        variant = str(raw_item["name"]).strip()
        match = re.match(r"^(.*?)(?:_L\d+|_[A-Za-z]\b)", variant)
        failure_mode = (
            match.group(1)
            if match is not None
            else variant
        )
        definition = VariantDefinition(
            name=variant,
            failure_mode=failure_mode,
        )
        if variant in definitions:
            raise ValueError(
                f"Duplicate pseudo variant in spec: {variant}"
            )
        definitions[variant] = definition

    missing = [
        name for name in VARIANT_ORDER if name not in definitions
    ]
    if missing:
        raise ValueError(
            "The pseudo specification does not define the complete "
            "protocol. Missing: " + ", ".join(missing)
        )

    # The explicit allowlist keeps unrelated conditions out of the protocol.
    return {name: definitions[name] for name in VARIANT_ORDER}


def _gt_data_folder(path: Path) -> Optional[Path]:
    """Return a GT data directory in TrackEval or tracker-output layout."""
    for child in ("label_02", "data"):
        candidate = path / child
        if candidate.is_dir():
            return candidate
    return None


def _valid_gt_source(path: Path) -> bool:
    return path.is_dir() and _gt_data_folder(path) is not None


def resolve_gt_folder(
    gt_base_dir: Path,
    definition: VariantDefinition,
    split: str,
) -> Path:
    del split
    candidates = [
        gt_base_dir / "GT__global",
        gt_base_dir,
    ]

    for candidate in candidates:
        if _valid_gt_source(candidate):
            return candidate.resolve()

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"{definition.name}: no complete canonical GT folder found. Tried: "
        f"{expected}"
    )


def resolve_seqmap_file(
    trackers_dir: Path,
    gt_base_dir: Path,
    split: str,
    explicit: Optional[Path],
) -> Path:
    filename = f"evaluate_tracking.seqmap.{split}"
    candidates: List[Path] = []
    if explicit is not None:
        explicit = explicit.expanduser().resolve()
        candidates.append(
            explicit / filename if explicit.is_dir() else explicit
        )

    candidates.extend(
        [
            gt_base_dir / "GT__global" / filename,
            gt_base_dir / filename,
        ]
    )

    # Infer the ordinary TrackEval global-GT location from:
    #   .../data/trackers/jrdb/jrdb_3d_box_test
    # ->.../data/gt__global/jrdb/jrdb_3d_box_test
    parents = trackers_dir.parents
    if len(parents) >= 3:
        data_root = parents[2]
        candidates.extend(
            [
                data_root
                / "gt__global"
                / "jrdb"
                / trackers_dir.name
                / filename,
                data_root
                / "gt"
                / "jrdb"
                / trackers_dir.name
                / filename,
            ]
        )

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the TrackEval sequence map. Supply "
        f"--seqmap-gt-folder with a directory containing {filename}. "
        "Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def stage_gt_folder(
    source: Path,
    staging_parent: Path,
    seqmap_file: Path,
    split: str,
) -> Path:
    """Expose tracker-style GT data as a TrackEval GT folder."""
    staged = staging_parent / source.name
    staged.mkdir(parents=True, exist_ok=False)
    data_folder = _gt_data_folder(source)
    if data_folder is None:
        raise FileNotFoundError(
            f"GT source has neither label_02 nor data: {source}"
        )
    (staged / "label_02").symlink_to(
        data_folder, target_is_directory=True
    )
    (staged / f"evaluate_tracking.seqmap.{split}").symlink_to(
        seqmap_file
    )
    return staged


def parse_tracker_variant_name(
    directory_name: str,
    definitions: Mapping[str, VariantDefinition],
) -> Optional[Tuple[str, VariantDefinition]]:
    if directory_name.lower().startswith("gt"):
        return None

    # Longest first prevents a shorter token from matching inside a longer one.
    for variant in sorted(definitions, key=len, reverse=True):
        suffix = "_" + variant
        if not directory_name.endswith(suffix):
            continue
        stem = directory_name[: -len(suffix)]
        marker = "__global"
        if marker not in stem:
            continue
        tracker_name, mode_suffix = stem.split(marker, 1)
        if not tracker_name or (mode_suffix and not mode_suffix.startswith("_")):
            continue
        tracker_base = tracker_name + mode_suffix
        return tracker_base, definitions[variant]
    return None


def selector_matches(tracker_base: str, selectors: Sequence[str]) -> bool:
    for selector in selectors:
        if fnmatch.fnmatchcase(tracker_base, selector):
            return True
        if tracker_base == selector or tracker_base.startswith(selector + "_"):
            return True
    return False


def variant_selector_matches(
    variant: str,
    selectors: Sequence[str],
) -> bool:
    return any(
        fnmatch.fnmatchcase(variant, selector)
        for selector in selectors
    )


def recompute_requested(
    tracker_base: str,
    variant: str,
    tracker_selectors: Sequence[str],
    variant_selectors: Sequence[str],
) -> bool:
    """Apply one selector dimension alone, or their intersection together."""
    if not tracker_selectors and not variant_selectors:
        return False
    tracker_matches = (
        not tracker_selectors
        or selector_matches(tracker_base, tracker_selectors)
    )
    variant_matches = (
        not variant_selectors
        or variant_selector_matches(variant, variant_selectors)
    )
    return tracker_matches and variant_matches


def summary_is_complete(path: Path, clear_threshold: float) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = {
                str(value).strip() for value in (reader.fieldnames or [])
            }
            required = {
                "seq",
                "HOTA",
                "IDSW",
                "Frag",
                "CLR_TP",
                "CLEAR_Threshold",
            }
            if not required.issubset(fieldnames):
                return False
            for row in reader:
                if str(row.get("seq", "")).strip().upper() != "COMBINED":
                    continue
                recorded = float(row["CLEAR_Threshold"])
                return bool(np.isclose(recorded, clear_threshold))
            return False
    except (OSError, csv.Error, UnicodeError, TypeError, ValueError):
        return False


def discover_jobs(
    trackers_dir: Path,
    gt_base_dir: Path,
    output_dir: Path,
    definitions: Mapping[str, VariantDefinition],
    tracker_selectors: Sequence[str],
    variant_selectors: Sequence[str],
    recompute_tracker_selectors: Sequence[str],
    recompute_variant_selectors: Sequence[str],
    tracker_subfolder: str,
    split: str,
    clear_threshold: float,
) -> Tuple[List[EvaluationJob], List[EvaluationJob]]:
    discovered: List[EvaluationJob] = []
    seen_results: Dict[str, str] = {}

    for path in sorted(trackers_dir.iterdir()):
        if not path.is_dir():
            continue
        parsed = parse_tracker_variant_name(path.name, definitions)
        if parsed is None:
            continue
        tracker_base, definition = parsed
        if tracker_selectors and not selector_matches(
            tracker_base, tracker_selectors
        ):
            continue
        if variant_selectors and not variant_selector_matches(
            definition.name, variant_selectors
        ):
            continue
        data_folder = path / tracker_subfolder
        if not data_folder.is_dir():
            raise FileNotFoundError(
                f"Discovered tracker folder lacks {tracker_subfolder!r}: "
                f"{path}"
            )

        result_name = path.name
        if result_name in seen_results:
            raise RuntimeError(
                "Two input folders map to the same canonical result "
                f"{result_name}: {seen_results[result_name]} and {path.name}"
            )
        seen_results[result_name] = path.name

        gt_path = resolve_gt_folder(
            gt_base_dir, definition, split
        )
        recompute = recompute_requested(
            tracker_base,
            definition.name,
            recompute_tracker_selectors,
            recompute_variant_selectors,
        )
        discovered.append(
            EvaluationJob(
                tracker_base=tracker_base,
                input_name=path.name,
                input_path=path.resolve(),
                variant=definition.name,
                failure_mode=definition.failure_mode,
                gt_path=gt_path,
                result_name=result_name,
                result_path=(output_dir / result_name).resolve(),
                recompute=recompute,
            )
        )

    pending = [
        job
        for job in discovered
        if job.recompute
        or not summary_is_complete(
            job.result_path / "pedestrian_summary.csv",
            clear_threshold,
        )
    ]
    return discovered, pending


def add_trackeval_to_path(trackeval_root: Path) -> None:
    root = str(Path(trackeval_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def make_dataset(
    trackeval,
    *,
    trackers_dir: Path,
    gt_folder: Path,
    tracker_name: str,
    split: str,
    tracker_subfolder: str,
):
    """Construct the official JRDB3DBox dataset without unused GT analytics."""
    config = dict(
        trackeval.datasets.JRDB3DBox.get_default_dataset_config()
    )
    config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(trackers_dir),
            "OUTPUT_FOLDER": None,
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "SPLIT_TO_EVAL": str(split),
            "TRACKER_SUB_FOLDER": str(tracker_subfolder),
            "PRINT_CONFIG": False,
            # These fields are analysis side-products added in this TrackEval
            # fork. HOTA reads none of them.
            "COMPUTE_GT_STATS": False,
            "COMPUTE_GT_SPEED_SAVGOL": False,
            "GT_STORE_FRAME_STATS_FOR_EVENTS": False,
        }
    )
    return trackeval.datasets.JRDB3DBox(config)


def dataset_sequences(
    trackeval_root: Path,
    trackers_dir: Path,
    gt_folder: Path,
    tracker_name: str,
    split: str,
    tracker_subfolder: str,
) -> List[str]:
    add_trackeval_to_path(trackeval_root)
    import trackeval  # noqa: WPS433

    dataset = make_dataset(
        trackeval,
        trackers_dir=trackers_dir,
        gt_folder=gt_folder,
        tracker_name=tracker_name,
        split=split,
        tracker_subfolder=tracker_subfolder,
    )
    _, sequences, classes = dataset.get_eval_info()
    if classes != ["pedestrian"]:
        raise RuntimeError(
            f"Unexpected TrackEval class registry: {classes!r}"
        )
    return sorted(str(sequence) for sequence in sequences)


def _cache_path(
    cache_root: Path,
    result_name: str,
    sequence: str,
) -> Path:
    return cache_root / result_name / f"{sequence}.npz"


def _save_npz_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _cache_payload_is_valid(payload: Mapping[str, Any]) -> bool:
    if "cache_version" not in payload:
        return False
    version = int(np.asarray(payload["cache_version"]).reshape(-1)[0])
    return version == CACHE_VERSION


def _cache_has_hota(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    required = {
        "cache_version",
        "hota__HOTA",
        "hota__DetA",
        "hota__AssA",
        "hota__HOTA_TP",
        "hota__HOTA_FN",
        "hota__HOTA_FP",
        "count__Dets",
        "count__GT_Dets",
        "count__IDs",
        "count__GT_IDs",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            return bool(
                required.issubset(payload.files)
                and _cache_payload_is_valid(payload)
            )
    except (OSError, ValueError, KeyError):
        return False


def _cache_has_clear(path: Path, clear_threshold: float) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    required = {
        "cache_version",
        "clear_threshold",
        "clear__IDSW",
        "clear__Frag",
        "clear__CLR_TP",
        "clear__CLR_FN",
        "clear__CLR_FP",
    }
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not (
                required.issubset(payload.files)
                and _cache_payload_is_valid(payload)
            ):
                return False
            recorded = float(
                np.asarray(payload["clear_threshold"]).reshape(-1)[0]
            )
            return bool(np.isclose(recorded, clear_threshold))
    except (OSError, ValueError, KeyError):
        return False


def _cache_is_complete(path: Path, clear_threshold: float) -> bool:
    return bool(
        _cache_has_hota(path)
        and _cache_has_clear(path, clear_threshold)
    )


def _load_cache_payload(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            if not _cache_payload_is_valid(payload):
                return {}
            return {
                key: np.asarray(payload[key]).copy()
                for key in payload.files
            }
    except (OSError, ValueError, KeyError):
        return {}


def _metric_payload(prefix: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        f"{prefix}__{field}": np.asarray(value)
        for field, value in result.items()
    }


def _evaluate_tracker_chunk(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate several sequences for one tracker using one dataset instance."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    trackeval_root = Path(str(payload["trackeval_root"]))
    add_trackeval_to_path(trackeval_root)
    import trackeval  # noqa: WPS433

    tracker_name = str(payload["input_name"])
    dataset = make_dataset(
        trackeval,
        trackers_dir=Path(str(payload["trackers_dir"])),
        gt_folder=Path(str(payload["gt_folder"])),
        tracker_name=tracker_name,
        split=str(payload["split"]),
        tracker_subfolder=str(payload["tracker_subfolder"]),
    )
    hota_metric = trackeval.metrics.HOTA()
    count_metric = trackeval.metrics.Count()
    clear_threshold = float(payload["clear_threshold"])
    clear_metric = trackeval.metrics.CLEAR(
        {
            "THRESHOLD": clear_threshold,
            "PRINT_CONFIG": False,
        }
    )
    cache_root = Path(str(payload["cache_root"]))
    result_name = str(payload["result_name"])
    force_recompute = bool(payload.get("force_recompute", False))

    status_rows: List[Dict[str, Any]] = []
    for sequence in payload["sequences"]:
        sequence = str(sequence)
        destination = _cache_path(
            cache_root, result_name, sequence
        )
        started = time.perf_counter()
        try:
            need_hota = force_recompute or not _cache_has_hota(destination)
            need_clear = force_recompute or not _cache_has_clear(
                destination, clear_threshold
            )
            if not need_hota and not need_clear:
                status_rows.append(
                    {
                        "result_name": result_name,
                        "sequence": sequence,
                        "status": "cached",
                        "seconds": 0.0,
                        "error": "",
                    }
                )
                continue

            raw_data = dataset.get_raw_seq_data(
                tracker_name, sequence, True
            )
            data = dataset.get_preprocessed_seq_data(
                raw_data, "pedestrian"
            )
            cache_payload = (
                {}
                if force_recompute
                else _load_cache_payload(destination)
            )
            cache_payload.update({
                "cache_version": np.asarray(
                    [CACHE_VERSION], dtype=np.int16
                ),
                "script_version": np.asarray([SCRIPT_VERSION]),
                "result_name": np.asarray([result_name]),
                "input_name": np.asarray([tracker_name]),
                "sequence": np.asarray([sequence]),
            })
            computed_metrics: List[str] = []
            if need_hota:
                hota_result = hota_metric.eval_sequence(data)
                count_result = count_metric.eval_sequence(data)
                cache_payload.update(_metric_payload("hota", hota_result))
                cache_payload.update(_metric_payload("count", count_result))
                computed_metrics.append("HOTA")
            if need_clear:
                clear_result = clear_metric.eval_sequence(data)
                cache_payload.update(_metric_payload("clear", clear_result))
                cache_payload["clear_threshold"] = np.asarray(
                    [clear_threshold], dtype=np.float64
                )
                computed_metrics.append("CLEAR")
            _save_npz_atomic(destination, cache_payload)
            status_rows.append(
                {
                    "result_name": result_name,
                    "sequence": sequence,
                    "status": "computed_" + "_".join(computed_metrics),
                    "seconds": time.perf_counter() - started,
                    "error": "",
                }
            )
        except Exception as error:  # returned to the parent with context
            status_rows.append(
                {
                    "result_name": result_name,
                    "sequence": sequence,
                    "status": "failed",
                    "seconds": time.perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return status_rows


def _load_sequence_result(
    path: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        hota = {
            key[len("hota__") :]: np.asarray(payload[key]).copy()
            for key in payload.files
            if key.startswith("hota__")
        }
        count = {
            key[len("count__") :]: np.asarray(payload[key]).item()
            for key in payload.files
            if key.startswith("count__")
        }
        clear = {
            key[len("clear__") :]: np.asarray(payload[key]).item()
            for key in payload.files
            if key.startswith("clear__")
        }
    return hota, clear, count


def _summary_row(metric, result: Mapping[str, Any]) -> Dict[str, str]:
    """Mirror TrackEval's _BaseMetric._summary_row formatting."""
    values: Dict[str, str] = {}
    for field in metric.summary_fields:
        if field in metric.float_array_fields:
            value = 100.0 * float(np.mean(result[field]))
            values[field] = f"{value:1.5g}"
        elif field in metric.float_fields:
            value = 100.0 * float(result[field])
            values[field] = f"{value:1.5g}"
        elif field in metric.loss_fields:
            values[field] = f"{float(result[field]):1.5g}"
        elif field in metric.integer_fields:
            values[field] = str(int(result[field]))
        else:
            raise NotImplementedError(
                f"Unsupported summary field: {field}"
            )
    return values


def _detailed_row(metric, result: Mapping[str, Any]) -> Dict[str, Any]:
    fields = list(metric.float_fields) + list(metric.integer_fields)
    values: List[Any] = [
        result[field]
        for field in metric.float_fields + metric.integer_fields
    ]
    alpha_labels = [int(100 * value) for value in metric.array_labels]
    for field in metric.float_array_fields + metric.integer_array_fields:
        array = np.asarray(result[field])
        fields.extend(f"{field}___{alpha}" for alpha in alpha_labels)
        fields.append(f"{field}___AUC")
        values.extend(array.tolist())
        values.append(float(np.mean(array)))
    return dict(zip(fields, values))


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate_and_write_result(
    *,
    trackeval_root: Path,
    cache_root: Path,
    output_dir: Path,
    result_name: str,
    sequences: Sequence[str],
    write_detailed: bool,
    clear_threshold: float,
) -> None:
    """Use TrackEval's official metric combiners and emit its CSV schema."""
    add_trackeval_to_path(trackeval_root)
    import trackeval  # noqa: WPS433

    hota_metric = trackeval.metrics.HOTA()
    clear_metric = trackeval.metrics.CLEAR(
        {
            "THRESHOLD": clear_threshold,
            "PRINT_CONFIG": False,
        }
    )
    count_metric = trackeval.metrics.Count()
    sequence_hota: Dict[str, Dict[str, Any]] = {}
    sequence_clear: Dict[str, Dict[str, Any]] = {}
    sequence_count: Dict[str, Dict[str, Any]] = {}
    for sequence in sequences:
        hota, clear, count = _load_sequence_result(
            _cache_path(cache_root, result_name, str(sequence))
        )
        sequence_hota[str(sequence)] = hota
        sequence_clear[str(sequence)] = clear
        sequence_count[str(sequence)] = count

    combined_hota = hota_metric.combine_sequences(sequence_hota)
    combined_clear = clear_metric.combine_sequences(sequence_clear)
    combined_count = count_metric.combine_sequences(sequence_count)
    table_hota = {
        **sequence_hota,
        "COMBINED_SEQ": combined_hota,
    }
    table_count = {
        **sequence_count,
        "COMBINED_SEQ": combined_count,
    }
    table_clear = {
        **sequence_clear,
        "COMBINED_SEQ": combined_clear,
    }

    summary_rows: List[Dict[str, Any]] = []
    for sequence in sorted(sequence_hota):
        summary_rows.append(
            {
                "seq": sequence,
                "CLEAR_Threshold": f"{clear_threshold:.8g}",
                **_summary_row(hota_metric, table_hota[sequence]),
                **_summary_row(clear_metric, table_clear[sequence]),
                **_summary_row(count_metric, table_count[sequence]),
            }
        )
    summary_rows.append(
        {
            "seq": "COMBINED",
            "CLEAR_Threshold": f"{clear_threshold:.8g}",
            **_summary_row(hota_metric, combined_hota),
            **_summary_row(clear_metric, combined_clear),
            **_summary_row(count_metric, combined_count),
        }
    )
    summary_fields = (
        ["seq", "CLEAR_Threshold"]
        + list(hota_metric.summary_fields)
        + list(clear_metric.summary_fields)
        + list(count_metric.summary_fields)
    )
    result_dir = output_dir / result_name
    _write_csv_atomic(
        result_dir / "pedestrian_summary.csv",
        summary_rows,
        summary_fields,
    )

    if write_detailed:
        detailed_rows: List[Dict[str, Any]] = []
        for sequence in sorted(sequence_hota):
            detailed_rows.append(
                {
                    "seq": sequence,
                    "CLEAR_Threshold": clear_threshold,
                    **_detailed_row(hota_metric, table_hota[sequence]),
                    **_detailed_row(clear_metric, table_clear[sequence]),
                    **_detailed_row(count_metric, table_count[sequence]),
                }
            )
        detailed_rows.append(
            {
                "seq": "COMBINED",
                "CLEAR_Threshold": clear_threshold,
                **_detailed_row(hota_metric, combined_hota),
                **_detailed_row(clear_metric, combined_clear),
                **_detailed_row(count_metric, combined_count),
            }
        )
        detailed_fields = list(detailed_rows[0].keys())
        _write_csv_atomic(
            result_dir / "pedestrian_detailed.csv",
            detailed_rows,
            detailed_fields,
        )
    else:
        # Do not leave a detailed table from an older parameterization next
        # to a newly recomputed summary.
        (result_dir / "pedestrian_detailed.csv").unlink(
            missing_ok=True
        )


def split_tracker_chunks(
    missing_by_result: Mapping[str, Sequence[str]],
    num_workers: int,
) -> List[Tuple[str, List[str]]]:
    """Split sequence work while reusing one dataset object per task."""
    active = [
        (result_name, list(sequences))
        for result_name, sequences in missing_by_result.items()
        if sequences
    ]
    if not active:
        return []
    n_active = len(active)
    chunks: List[Tuple[str, List[str]]] = []
    for index, (result_name, sequences) in enumerate(active):
        if n_active >= num_workers:
            requested_chunks = 1
        else:
            requested_chunks = (
                num_workers // n_active
                + int(index < num_workers % n_active)
            )
        n_chunks = min(len(sequences), max(1, requested_chunks))
        for part in np.array_split(
            np.asarray(sequences, dtype=object), n_chunks
        ):
            values = [str(value) for value in part.tolist()]
            if values:
                chunks.append((result_name, values))
    return chunks


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    discovered: Sequence[EvaluationJob],
    attempted_names: Iterable[str],
    failed_names: Iterable[str],
) -> None:
    attempted = set(attempted_names)
    failed = set(failed_names)
    rows = []
    for job in discovered:
        summary = job.result_path / "pedestrian_summary.csv"
        if job.result_name in failed:
            status = "failed"
        elif job.result_name in attempted:
            status = (
                "recomputed" if job.recompute else "computed"
            )
        elif summary_is_complete(summary, args.clear_threshold):
            status = "skipped_existing"
        else:
            status = "not_run"
        rows.append(
            {
                **{
                    key: (
                        str(value)
                        if isinstance(value, Path)
                        else value
                    )
                    for key, value in asdict(job).items()
                },
                "status": status,
                "summary_path": str(summary),
                "summary_complete": summary_is_complete(
                    summary, args.clear_threshold
                ),
            }
        )

    payload = {
        "script_version": SCRIPT_VERSION,
        "cache_version": CACHE_VERSION,
        "pseudo_spec": str(args.pseudo_spec),
        "trackeval_root": str(args.trackeval_root),
        "trackers_dir": str(args.trackers_dir),
        "gt_base_dir": str(args.gt_base_dir),
        "output_dir": str(args.output_dir),
        "cache_dir": str(args.cache_dir),
        "split": args.split,
        "metrics": args.metrics,
        "clear_threshold": args.clear_threshold,
        "num_workers": args.num_workers,
        "write_detailed": not args.summary_only,
        "trackers": args.trackers,
        "variants": args.variants,
        "recompute_trackers": args.recompute_trackers,
        "recompute_variants": args.recompute_variants,
        "evaluations": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically discover global JRDB pseudo-detection tracker "
            "outputs for the ten-condition protocol, evaluate every "
            "condition against canonical global GT, and compute "
            "official TrackEval HOTA and CLEAR through a global, resumable "
            "worker pool."
        )
    )
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--trackers-dir", type=Path, required=True)
    parser.add_argument(
        "--gt-folder",
        "--gt-base-dir",
        dest="gt_base_dir",
        type=Path,
        default=None,
        help=(
            "Canonical global JRDB GT folder (label_02 layout), or a parent "
            "containing tracker-style GT__global/data. No condition-specific "
            "GT is used. Default: --trackers-dir."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pseudo-spec", type=Path, required=True)
    parser.add_argument(
        "--seqmap-gt-folder",
        type=Path,
        default=None,
        help=(
            "Optional TrackEval GT folder (or seqmap file) used only for "
            "evaluate_tracking.seqmap.<split>. Normally inferred from the "
            "standard trackeval_layout/data/gt__global tree."
        ),
    )
    parser.add_argument(
        "--trackers",
        nargs="*",
        default=[],
        help=(
            "Optional tracker-family scope. Space/comma-separated names and "
            "shell wildcards are accepted. Empty means every discovered "
            "tracker family."
        ),
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[],
        help=(
            "Optional variant scope, e.g. clean or combined_*. "
            "Empty means all ten canonical variants."
        ),
    )
    parser.add_argument(
        "--recompute-trackers",
        nargs="*",
        default=[],
        help=(
            "Space/comma-separated tracker families to recompute. "
            "Examples: fastpoly or fastpoly,cbmot. Shell wildcards are "
            "also accepted."
        ),
    )
    parser.add_argument(
        "--recompute-variants",
        nargs="*",
        default=[],
        help=(
            "Variants to force-recompute, with shell wildcards accepted. "
            "Used alone, this applies to every tracker in scope. When both "
            "recompute selector types are supplied, their intersection is "
            "recomputed."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument(
        "--clear-threshold",
        type=float,
        default=0.05,
        help=(
            "3D similarity threshold used by CLEAR/IDSW. The default 0.05 "
            "is deliberately localization-relaxed so IDSW primarily measures "
            "identity mixing under detector instability."
        ),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--tracker-subfolder", default="data")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Per-tracker/per-sequence cache root. Default: "
            "OUTPUT_DIR/_pseudo_hota_cache."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["HOTA"],
        help=(
            "Retained for CLI compatibility. HOTA, CLEAR, and Count are "
            "always evaluated by the protocol."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Write pedestrian_summary.csv only. By default the command also writes "
            "the TrackEval-compatible pedestrian_detailed.csv."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Retained for CLI compatibility. The command always attempts all "
            "independent worker chunks and reports every failure together."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.trackeval_root = args.trackeval_root.expanduser().resolve()
    args.trackers_dir = args.trackers_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.pseudo_spec = args.pseudo_spec.expanduser().resolve()
    args.gt_base_dir = (
        args.trackers_dir
        if args.gt_base_dir is None
        else args.gt_base_dir.expanduser().resolve()
    )
    args.trackers = _split_cli_values(args.trackers)
    args.variants = _split_cli_values(args.variants)
    args.recompute_trackers = _split_cli_values(
        args.recompute_trackers
    )
    args.recompute_variants = _split_cli_values(
        args.recompute_variants
    )
    args.metrics = _split_cli_values(args.metrics)
    args.cache_dir = (
        args.output_dir
        / "_pseudo_hota_cache"
        if args.cache_dir is None
        else args.cache_dir.expanduser().resolve()
    )

    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if not (0.0 <= args.clear_threshold <= 1.0):
        parser.error("--clear-threshold must be in [0, 1]")
    if not args.metrics:
        parser.error("--metrics must contain at least one metric")
    if [metric.upper() for metric in args.metrics] != ["HOTA"]:
        parser.error(
            "Keep --metrics HOTA for CLI compatibility. CLEAR and Count are "
            "included automatically."
        )
    if not args.trackers_dir.is_dir():
        raise FileNotFoundError(
            f"Tracker directory not found: {args.trackers_dir}"
        )
    if not args.gt_base_dir.is_dir():
        raise FileNotFoundError(
            f"GT base directory not found: {args.gt_base_dir}"
        )
    if not args.pseudo_spec.is_file():
        raise FileNotFoundError(
            f"Pseudo specification not found: {args.pseudo_spec}"
        )

    trackeval_package = args.trackeval_root / "trackeval" / "__init__.py"
    if not trackeval_package.is_file():
        raise FileNotFoundError(
            "TrackEval package not found under --trackeval-root: "
            f"{trackeval_package}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    definitions = load_variant_definitions(args.pseudo_spec)
    for option_name, selectors in (
        ("--variants", args.variants),
        ("--recompute-variants", args.recompute_variants),
    ):
        unmatched = [
            selector
            for selector in selectors
            if not any(
                fnmatch.fnmatchcase(name, selector)
                for name in definitions
            )
        ]
        if unmatched:
            parser.error(
                f"{option_name} selector(s) match no canonical "
                "variant: " + ", ".join(unmatched)
            )
    seqmap_file = resolve_seqmap_file(
        trackers_dir=args.trackers_dir,
        gt_base_dir=args.gt_base_dir,
        split=args.split,
        explicit=args.seqmap_gt_folder,
    )
    discovered, pending = discover_jobs(
        trackers_dir=args.trackers_dir,
        gt_base_dir=args.gt_base_dir,
        output_dir=args.output_dir,
        definitions=definitions,
        tracker_selectors=args.trackers,
        variant_selectors=args.variants,
        recompute_tracker_selectors=args.recompute_trackers,
        recompute_variant_selectors=args.recompute_variants,
        tracker_subfolder=args.tracker_subfolder,
        split=args.split,
        clear_threshold=args.clear_threshold,
    )
    if not discovered:
        raise RuntimeError(
            "No matching global pseudo-detection tracker folders were "
            "discovered."
        )

    print(
        "[discover] Canonical variants: "
        f"{len(definitions)}"
    )
    print(f"[discover] Canonical GT:             {args.gt_base_dir}")
    print(f"[discover] Tracker/variant folders: {len(discovered)}")
    print(f"[discover] Pending evaluations:      {len(pending)}")
    print(
        f"[discover] Existing results skipped: "
        f"{len(discovered) - len(pending)}"
    )
    if args.trackers:
        print(
            "[discover] Tracker scope:          "
            + ", ".join(args.trackers)
        )
    if args.variants:
        print(
            "[discover] Variant scope:          "
            + ", ".join(args.variants)
        )
    if args.recompute_trackers or args.recompute_variants:
        print(
            "[discover] Recompute trackers:     "
            + (", ".join(args.recompute_trackers) or "<all in scope>")
        )
        print(
            "[discover] Recompute variants:     "
            + (", ".join(args.recompute_variants) or "<all in scope>")
        )
    print(f"[discover] Sequence map:             {seqmap_file}")
    print(
        f"[discover] CLEAR/IDSW threshold:     "
        f"{args.clear_threshold:g}"
    )

    gt_groups: Dict[Path, List[EvaluationJob]] = defaultdict(list)
    for job in pending:
        gt_groups[job.gt_path].append(job)

    print("\nVariant-to-GT groups:")
    for gt_path, jobs in sorted(
        gt_groups.items(), key=lambda item: item[0].name
    ):
        print(f"  {gt_path.name}: {len(jobs)} pending result(s)")
        for job in sorted(jobs, key=lambda value: value.result_name):
            reason = "recompute" if job.recompute else "missing"
            print(
                f"    [{reason}] {job.input_name} -> {job.result_name}"
            )

    manifest_path = (
        args.output_dir
        / "pseudo_detection_evaluation_manifest.json"
    )
    if args.dry_run:
        print("\n[dry-run] No TrackEval sequence jobs were started.")
        return 0
    if not pending:
        write_manifest(
            manifest_path, args, discovered, [], []
        )
        print("\n[done] Every discovered result is already complete.")
        print(f"[done] Manifest: {manifest_path}")
        return 0

    failed: List[str] = []
    attempted: List[str] = [job.result_name for job in pending]
    all_status_rows: List[Dict[str, Any]] = []
    evaluation_started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="tracker_eval_pseudo_"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        staging_gt_root = temporary_root / "gt"
        staging_gt_root.mkdir()

        staged_gt_by_source: Dict[Path, Path] = {}
        sequences_by_result: Dict[str, List[str]] = {}
        for gt_path, jobs in sorted(
            gt_groups.items(), key=lambda item: item[0].name
        ):
            staged_gt_by_source[gt_path] = stage_gt_folder(
                source=gt_path,
                staging_parent=staging_gt_root,
                seqmap_file=seqmap_file,
                split=args.split,
            )
            representative = sorted(
                jobs, key=lambda value: value.result_name
            )[0]
            group_sequences = dataset_sequences(
                trackeval_root=args.trackeval_root,
                trackers_dir=args.trackers_dir,
                gt_folder=staged_gt_by_source[gt_path],
                tracker_name=representative.input_name,
                split=args.split,
                tracker_subfolder=args.tracker_subfolder,
            )
            if not group_sequences:
                raise RuntimeError(
                    f"No TrackEval sequences for GT {gt_path}"
                )
            for job in jobs:
                sequences_by_result[job.result_name] = group_sequences

        job_by_result = {
            job.result_name: job
            for job in pending
        }
        missing_by_result: Dict[str, List[str]] = {}
        total_cached_before = 0
        hota_cached_before = 0
        clear_cached_before = 0
        total_sequence_candidates = 0
        for job in pending:
            sequences = sequences_by_result[job.result_name]
            missing: List[str] = []
            for sequence in sequences:
                total_sequence_candidates += 1
                cache_path = _cache_path(
                    args.cache_dir, job.result_name, sequence
                )
                if not job.recompute and _cache_has_hota(cache_path):
                    hota_cached_before += 1
                if not job.recompute and _cache_has_clear(
                    cache_path, args.clear_threshold
                ):
                    clear_cached_before += 1
                if job.recompute or not _cache_is_complete(
                    cache_path, args.clear_threshold
                ):
                    missing.append(sequence)
                else:
                    total_cached_before += 1
                    all_status_rows.append(
                        {
                            "result_name": job.result_name,
                            "sequence": sequence,
                            "status": "cached",
                            "seconds": 0.0,
                            "error": "",
                        }
                    )
            missing_by_result[job.result_name] = missing

        chunks = split_tracker_chunks(
            missing_by_result, args.num_workers
        )
        total_missing = sum(
            len(sequences)
            for sequences in missing_by_result.values()
        )
        print(
            "\n[cache] Complete tracker-sequence results: "
            f"{total_cached_before} reusable, {total_missing} pending"
        )
        print(
            "[cache] HOTA sequence results:             "
            f"{hota_cached_before} reusable, "
            f"{total_sequence_candidates - hota_cached_before} pending"
        )
        print(
            "[cache] CLEAR/IDSW sequence results:       "
            f"{clear_cached_before} reusable, "
            f"{total_sequence_candidates - clear_cached_before} pending"
        )
        print(
            f"[evaluate] {len(chunks)} worker chunk(s), "
            f"{args.num_workers} process(es)"
        )

        chunk_payloads: List[Dict[str, Any]] = []
        for result_name, sequences in chunks:
            job = job_by_result[result_name]
            chunk_payloads.append(
                {
                    "trackeval_root": str(args.trackeval_root),
                    "trackers_dir": str(args.trackers_dir),
                    "gt_folder": str(
                        staged_gt_by_source[job.gt_path]
                    ),
                    "input_name": job.input_name,
                    "result_name": result_name,
                    "sequences": sequences,
                    "split": args.split,
                    "tracker_subfolder": args.tracker_subfolder,
                    "cache_root": str(args.cache_dir),
                    "clear_threshold": args.clear_threshold,
                    "force_recompute": job.recompute,
                }
            )

        if args.num_workers == 1:
            completed_sequences = 0
            for payload in chunk_payloads:
                rows = _evaluate_tracker_chunk(payload)
                all_status_rows.extend(rows)
                completed_sequences += len(rows)
                print(
                    f"[progress] {completed_sequences}/{total_missing} "
                    "tracker-sequences"
                )
        elif chunk_payloads:
            completed_sequences = 0
            with ProcessPoolExecutor(
                max_workers=args.num_workers
            ) as executor:
                futures = {
                    executor.submit(
                        _evaluate_tracker_chunk, payload
                    ): payload
                    for payload in chunk_payloads
                }
                for future in as_completed(futures):
                    payload = futures[future]
                    try:
                        rows = future.result()
                    except Exception as error:
                        rows = [
                            {
                                "result_name": payload["result_name"],
                                "sequence": sequence,
                                "status": "failed",
                                "seconds": float("nan"),
                                "error": (
                                    f"{type(error).__name__}: {error}"
                                ),
                            }
                            for sequence in payload["sequences"]
                        ]
                    all_status_rows.extend(rows)
                    completed_sequences += len(rows)
                    print(
                        f"[progress] {completed_sequences}/"
                        f"{total_missing} tracker-sequences"
                    )

        sequence_failures = [
            row
            for row in all_status_rows
            if row["status"] == "failed"
        ]
        for row in sequence_failures:
            print(
                "[error] "
                f"{row['result_name']}:{row['sequence']}: "
                f"{row['error']}",
                file=sys.stderr,
            )
        failed.extend(
            sorted(
                {
                    str(row["result_name"])
                    for row in sequence_failures
                }
            )
        )

        for job in pending:
            if job.result_name in failed:
                continue
            sequences = sequences_by_result[job.result_name]
            missing_after = [
                sequence
                for sequence in sequences
                if not _cache_is_complete(
                    _cache_path(
                        args.cache_dir,
                        job.result_name,
                        sequence,
                    ),
                    args.clear_threshold,
                )
            ]
            if missing_after:
                failed.append(job.result_name)
                print(
                    f"[error] {job.result_name}: incomplete cache; "
                    f"first missing sequence={missing_after[0]}",
                    file=sys.stderr,
                )
                continue
            try:
                aggregate_and_write_result(
                    trackeval_root=args.trackeval_root,
                    cache_root=args.cache_dir,
                    output_dir=args.output_dir,
                    result_name=job.result_name,
                    sequences=sequences,
                    write_detailed=not args.summary_only,
                    clear_threshold=args.clear_threshold,
                )
            except Exception as error:
                failed.append(job.result_name)
                print(
                    f"[error] Failed to aggregate {job.result_name}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                )

    status_path = (
        args.output_dir
        / "pseudo_detection_sequence_status.csv"
    )
    if all_status_rows:
        ordered_status = sorted(
            all_status_rows,
            key=lambda row: (
                str(row["result_name"]),
                str(row["sequence"]),
            ),
        )
        _write_csv_atomic(
            status_path,
            ordered_status,
            [
                "result_name",
                "sequence",
                "status",
                "seconds",
                "error",
            ],
        )

    incomplete = [
        job.result_name
        for job in pending
        if job.result_name not in failed
        and not summary_is_complete(
            job.result_path / "pedestrian_summary.csv",
            args.clear_threshold,
        )
    ]
    if incomplete:
        failed.extend(incomplete)
        print(
            "[error] TrackEval finished without complete summaries for: "
            + ", ".join(incomplete),
            file=sys.stderr,
        )

    failed = sorted(set(failed))

    write_manifest(
        manifest_path,
        args,
        discovered,
        attempted,
        failed,
    )
    if failed:
        print(
            "\n[done] Failed/incomplete results: "
            + ", ".join(sorted(set(failed))),
            file=sys.stderr,
        )
        print(f"[done] Manifest: {manifest_path}")
        return 1

    print(
        f"\n[done] Evaluated {len(pending)} result(s); "
        f"skipped {len(discovered) - len(pending)} existing result(s)."
    )
    print(
        f"[done] Wall time: {time.perf_counter() - evaluation_started:.1f} s"
    )
    print(f"[done] Results:  {args.output_dir}")
    print(f"[done] Cache:    {args.cache_dir}")
    if all_status_rows:
        print(f"[done] Status:   {status_path}")
    print(f"[done] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
