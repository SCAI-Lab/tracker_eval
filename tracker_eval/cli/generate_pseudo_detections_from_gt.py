"""Generate the final controlled JRDB pseudo-detection variants.

This contains the standard-GT protocol used in the RA-L analysis: Clean, three
Dropout levels, three Instability levels, and three symmetric Combined levels.
Existing variant JSON files are reused only when the previous manifest confirms
that their generation configuration and GT definition are compatible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import yaml

from tracker_eval.utils import (
    _box7_from_label_obj,
    _ceil_frames,
    _clip01,
    _load_labels_3d_json,
    _parse_frame_key,
    _parse_label_id_strict,
    _safe_pos,
    _seed_u32,
    _set_height_keep_bottom,
    _trunc_normal,
    _wrap_angle_rad_pi,
)


EXPECTED_VARIANTS = {
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
}
ALLOWED_MODES = {"clean", "dropout", "instability"}


def _split_values(values: Optional[Iterable[str]]) -> List[str]:
    output: List[str] = []
    for value in values or []:
        output.extend(
            item.strip()
            for item in str(value).split(",")
            if item.strip()
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Score sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreDistribution:
    tp_scores: np.ndarray


def _load_score_distribution(path: Path) -> ScoreDistribution:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as payload:
            if "tp_scores" not in payload.files:
                raise ValueError(f"{path} lacks the required 'tp_scores' array")
            scores = np.asarray(payload["tp_scores"], dtype=np.float32).reshape(-1)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON mapping in {path}")
        if "tp_scores" in payload:
            raw = payload["tp_scores"]
        elif isinstance(payload.get("scores"), dict):
            raw = payload["scores"].get("tp")
        else:
            raw = payload.get("tp")
        if raw is None:
            raise ValueError(f"{path} lacks TP score samples")
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported score-distribution file: {path}")

    scores = scores[np.isfinite(scores)]
    scores = np.clip(scores, 0.0, 1.0)
    if scores.size == 0:
        raise ValueError(f"No finite TP scores remain in {path}")
    return ScoreDistribution(tp_scores=scores)


class ScoreSampler:
    def __init__(self, distribution: ScoreDistribution) -> None:
        self._scores = distribution.tp_scores

    def sample(self, rng: np.random.Generator) -> float:
        index = int(rng.integers(0, self._scores.size))
        return float(self._scores[index])


# ---------------------------------------------------------------------------
# Variant configuration and corruption models
# ---------------------------------------------------------------------------


@dataclass
class VariantCfg:
    name: str
    fps: float
    class_name: str = "pedestrian"
    severity: float = 1.0

    dropout_enable: bool = False
    dropout_p_start: float = 0.0
    dropout_min_s: float = 0.0
    dropout_max_s: float = 0.0

    instability_enable: bool = False
    instability_k_modes: int = 3
    instability_p_switch: float = 0.0
    instability_mode_xy_sigma_m: float = 0.0
    instability_mode_yaw_sigma_rad: float = 0.0
    instability_mode_lwh_sigma_rel: float = 0.0
    instability_jitter_xy_sigma_m: float = 0.0
    instability_jitter_yaw_sigma_rad: float = 0.0
    instability_jitter_lwh_sigma_rel: float = 0.0
    instability_p_yaw_random: float = 0.0

    score_value: float = 1.0
    score_mode: str = "constant"
    score_dists: str = ""

    dropout_level: str = ""
    instability_level: str = ""


def _variant_cfg_from_dict(
    name: str,
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> VariantCfg:
    merged = dict(base)
    merged.update(dict(override))
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
        instability_mode_xy_sigma_m=float(
            merged.get("instability_mode_xy_sigma_m", 0.0)
        ),
        instability_mode_yaw_sigma_rad=float(
            merged.get("instability_mode_yaw_sigma_rad", 0.0)
        ),
        instability_mode_lwh_sigma_rel=float(
            merged.get("instability_mode_lwh_sigma_rel", 0.0)
        ),
        instability_jitter_xy_sigma_m=float(
            merged.get("instability_jitter_xy_sigma_m", 0.0)
        ),
        instability_jitter_yaw_sigma_rad=float(
            merged.get("instability_jitter_yaw_sigma_rad", 0.0)
        ),
        instability_jitter_lwh_sigma_rel=float(
            merged.get("instability_jitter_lwh_sigma_rel", 0.0)
        ),
        instability_p_yaw_random=float(
            merged.get("instability_p_yaw_random", 0.0)
        ),
        score_value=float(merged.get("score_value", 1.0)),
        score_mode=str(merged.get("score_mode", "constant")),
        score_dists=str(merged.get("score_dists", "")),
        dropout_level=str(merged.get("dropout_level", "")),
        instability_level=str(merged.get("instability_level", "")),
    )

    cfg.dropout_p_start = _clip01(cfg.dropout_p_start)
    cfg.instability_p_switch = _clip01(cfg.instability_p_switch)
    cfg.instability_p_yaw_random = _clip01(cfg.instability_p_yaw_random)
    cfg.instability_k_modes = max(1, cfg.instability_k_modes)
    cfg.score_mode = cfg.score_mode.strip().lower() or "constant"

    if cfg.fps <= 0.0:
        raise ValueError(f"{name}: fps must be positive")
    if cfg.score_mode not in {"constant", "sample"}:
        raise ValueError(f"{name}: score_mode must be 'constant' or 'sample'")
    if cfg.instability_p_yaw_random != 0.0:
        raise ValueError(
            f"{name}: random yaw replacement is not part of the retained protocol"
        )
    if cfg.dropout_enable:
        if cfg.dropout_p_start <= 0.0:
            raise ValueError(f"{name}: dropout_p_start must be positive")
        if cfg.dropout_min_s < 0.0 or cfg.dropout_max_s < cfg.dropout_min_s:
            raise ValueError(f"{name}: invalid dropout duration interval")
    return cfg


def _apply_severity_once(cfg_in: VariantCfg) -> VariantCfg:
    cfg = VariantCfg(**asdict(cfg_in))
    severity = float(cfg.severity)
    cfg.dropout_p_start = _clip01(cfg.dropout_p_start * severity)
    cfg.instability_p_switch = _clip01(
        cfg.instability_p_switch * severity
    )
    cfg.instability_p_yaw_random = _clip01(
        cfg.instability_p_yaw_random * severity
    )
    cfg.instability_mode_xy_sigma_m *= severity
    cfg.instability_mode_yaw_sigma_rad *= severity
    cfg.instability_mode_lwh_sigma_rel *= severity
    cfg.instability_jitter_xy_sigma_m *= severity
    cfg.instability_jitter_yaw_sigma_rad *= severity
    cfg.instability_jitter_lwh_sigma_rel *= severity
    return cfg


def _make_dropout_keep_mask(
    frames: np.ndarray,
    fps: float,
    p_start: float,
    min_s: float,
    max_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(frames.shape[0])
    keep = np.ones(n, dtype=bool)
    if n == 0 or p_start <= 0.0 or max_s <= 0.0:
        return keep

    minimum = _ceil_frames(min_s, fps)
    maximum = max(minimum, _ceil_frames(max_s, fps))
    dropout_until = -10**9
    for index, frame in enumerate(frames.tolist()):
        if frame <= dropout_until:
            keep[index] = False
        elif rng.random() < _clip01(p_start):
            duration = int(rng.integers(minimum, maximum + 1))
            dropout_until = int(frame) + duration - 1
            keep[index] = False
    return keep


def _apply_instability_hypothesis_switching(
    frames: np.ndarray,
    gt_boxes: np.ndarray,
    cfg: VariantCfg,
    rng: np.random.Generator,
) -> np.ndarray:
    boxes = gt_boxes.astype(np.float32).copy()
    n = int(frames.shape[0])
    if n == 0 or not cfg.instability_enable:
        return boxes

    modes = max(1, int(cfg.instability_k_modes))
    switch_probability = _clip01(cfg.instability_p_switch)
    mode_xy = _trunc_normal(
        rng,
        0.0,
        float(cfg.instability_mode_xy_sigma_m),
        size=(modes, 2),
        n_sigma=2.0,
    ).astype(np.float32, copy=False)
    mode_yaw = _trunc_normal(
        rng,
        0.0,
        float(cfg.instability_mode_yaw_sigma_rad),
        size=(modes,),
        n_sigma=2.0,
    ).astype(np.float32, copy=False)
    mode_lwh = _trunc_normal(
        rng,
        0.0,
        float(cfg.instability_mode_lwh_sigma_rel),
        size=(modes, 3),
        n_sigma=2.0,
    ).astype(np.float32, copy=False)
    mode_index = int(rng.integers(0, modes))

    for index in range(n):
        if (
            index > 0
            and modes > 1
            and rng.random() < switch_probability
        ):
            new_index = int(rng.integers(0, modes - 1))
            if new_index >= mode_index:
                new_index += 1
            mode_index = new_index

        boxes[index, 0] += float(mode_xy[mode_index, 0])
        boxes[index, 1] += float(mode_xy[mode_index, 1])
        boxes[index, 6] = _wrap_angle_rad_pi(
            float(boxes[index, 6]) + float(mode_yaw[mode_index])
        )

        relative = mode_lwh[mode_index]
        boxes[index, 3] = _safe_pos(
            float(boxes[index, 3]) * (1.0 + float(relative[0]))
        )
        boxes[index, 4] = _safe_pos(
            float(boxes[index, 4]) * (1.0 + float(relative[1]))
        )
        height = _safe_pos(
            float(boxes[index, 5]) * (1.0 + float(relative[2]))
        )
        _set_height_keep_bottom(boxes[index], height)

        if cfg.instability_jitter_xy_sigma_m > 0.0:
            boxes[index, 0] += float(
                _trunc_normal(
                    rng,
                    0.0,
                    cfg.instability_jitter_xy_sigma_m,
                    size=(),
                    n_sigma=2.0,
                )
            )
            boxes[index, 1] += float(
                _trunc_normal(
                    rng,
                    0.0,
                    cfg.instability_jitter_xy_sigma_m,
                    size=(),
                    n_sigma=2.0,
                )
            )

        if cfg.instability_jitter_lwh_sigma_rel > 0.0:
            jitter = _trunc_normal(
                rng,
                0.0,
                cfg.instability_jitter_lwh_sigma_rel,
                size=(3,),
                n_sigma=2.0,
            ).astype(np.float32, copy=False)
            boxes[index, 3] = _safe_pos(
                float(boxes[index, 3]) * (1.0 + float(jitter[0]))
            )
            boxes[index, 4] = _safe_pos(
                float(boxes[index, 4]) * (1.0 + float(jitter[1]))
            )
            height = _safe_pos(
                float(boxes[index, 5]) * (1.0 + float(jitter[2]))
            )
            _set_height_keep_bottom(boxes[index], height)

        if cfg.instability_jitter_yaw_sigma_rad > 0.0:
            boxes[index, 6] = _wrap_angle_rad_pi(
                float(boxes[index, 6])
                + float(
                    _trunc_normal(
                        rng,
                        0.0,
                        cfg.instability_jitter_yaw_sigma_rad,
                        size=(),
                        n_sigma=2.0,
                    )
                )
            )
    return boxes.astype(np.float32)


# ---------------------------------------------------------------------------
# GT loading and detector-schema output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceData:
    name: str
    frame_keys: Tuple[str, ...]
    by_track: Dict[int, Tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class DetectionRow:
    box7: np.ndarray
    score: float


def _load_sequence(path: Path, class_name: str) -> SequenceData:
    frame_dict = _load_labels_3d_json(path)
    by_track_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    canonical_keys: List[str] = []

    for raw_key, objects in frame_dict.items():
        frame_key = _parse_frame_key(raw_key)
        frame = int(frame_key.split(".")[0])
        canonical_keys.append(f"{frame:06d}.pcd")
        for obj in objects:
            label_id = obj.get("label_id")
            if label_id is None:
                continue
            object_class, track_id = _parse_label_id_strict(label_id)
            if object_class.lower() != class_name.lower():
                continue
            by_track_rows.setdefault(int(track_id), []).append(
                (frame, _box7_from_label_obj(obj))
            )

    by_track: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for track_id, rows in by_track_rows.items():
        rows.sort(key=lambda item: item[0])
        frames = np.asarray([item[0] for item in rows], dtype=np.int32)
        boxes = np.stack([item[1] for item in rows]).astype(np.float32)
        by_track[track_id] = (frames, boxes)

    return SequenceData(
        name=path.stem,
        frame_keys=tuple(sorted(set(canonical_keys))),
        by_track=by_track,
    )


def _sample_score(
    cfg: VariantCfg,
    sampler: Optional[ScoreSampler],
    rng: np.random.Generator,
) -> float:
    if cfg.score_mode == "sample" and sampler is not None:
        return sampler.sample(rng)
    return float(cfg.score_value)


def _generate_sequence(
    sequence: SequenceData,
    cfg: VariantCfg,
    seed: int,
    score_sampler: Optional[ScoreSampler],
) -> Dict[str, List[DetectionRow]]:
    output: Dict[str, List[DetectionRow]] = {}
    for track_id, (frames, gt_boxes) in sequence.by_track.items():
        if cfg.instability_enable:
            instability_rng = np.random.default_rng(
                _seed_u32(
                    seed,
                    sequence.name,
                    "tid",
                    int(track_id),
                    "instability",
                    cfg.instability_level,
                )
            )
            boxes = _apply_instability_hypothesis_switching(
                frames,
                gt_boxes,
                cfg,
                instability_rng,
            )
        else:
            boxes = gt_boxes.copy()

        keep = np.ones(len(frames), dtype=bool)
        if cfg.dropout_enable:
            dropout_rng = np.random.default_rng(
                _seed_u32(
                    seed,
                    sequence.name,
                    "tid",
                    int(track_id),
                    "dropout",
                    cfg.dropout_level,
                )
            )
            keep = _make_dropout_keep_mask(
                frames,
                cfg.fps,
                cfg.dropout_p_start,
                cfg.dropout_min_s,
                cfg.dropout_max_s,
                dropout_rng,
            )

        # Sampling precedes dropout so surviving frames share exactly the same
        # confidence realization in every retained variant.
        score_rng = np.random.default_rng(
            _seed_u32(
                seed,
                sequence.name,
                "tid",
                int(track_id),
                "primary-score",
            )
        )
        scores = [
            _sample_score(cfg, score_sampler, score_rng)
            for _ in frames
        ]

        for frame, box, is_kept, score in zip(
            frames.tolist(), boxes, keep.tolist(), scores
        ):
            if not is_kept:
                continue
            key = f"{int(frame):06d}.pcd"
            output.setdefault(key, []).append(
                DetectionRow(box7=box.astype(np.float32), score=float(score))
            )
    return output


def _write_detections(
    path: Path,
    detections: Mapping[str, Sequence[DetectionRow]],
    frame_keys: Sequence[str],
    class_name: str,
) -> None:
    payload: Dict[str, List[Dict[str, Any]]] = {}
    for frame_key in frame_keys:
        rows: List[Dict[str, Any]] = []
        for detection in detections.get(frame_key, []):
            cx, cy, cz, length, width, height, yaw = [
                float(value) for value in detection.box7.tolist()
            ]
            rows.append(
                {
                    "box": {
                        "cx": cx,
                        "cy": cy,
                        "cz": cz,
                        "h": height,
                        "l": length,
                        "rot_z": yaw,
                        "w": width,
                    },
                    "label_id": f"{class_name}:-1",
                    "file_id": frame_key,
                    "score": float(detection.score),
                }
            )
        payload[frame_key] = rows

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump({"detections": payload}, stream)


# ---------------------------------------------------------------------------
# Specification and manifest
# ---------------------------------------------------------------------------


def _variant_name(mode: str, level_name: str) -> str:
    if mode == "clean":
        return "clean"
    return (
        level_name
        if level_name.startswith(mode + "_")
        else f"{mode}_{level_name}"
    )


def _expand_variants(
    spec: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Tuple[str, str, Dict[str, Any]]]]:
    base = dict(spec.get("base", {}))
    sweeps = spec.get("single_mode_sweeps", {})
    if not isinstance(sweeps, dict):
        raise ValueError("single_mode_sweeps must be a mapping")
    unknown_modes = set(map(str, sweeps)) - ALLOWED_MODES
    if unknown_modes:
        raise ValueError(
            "The protocol accepts only clean, dropout, and instability; found "
            f"{sorted(unknown_modes)}"
        )
    variants: List[Tuple[str, str, Dict[str, Any]]] = []
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_mode, entries in sweeps.items():
        mode = str(raw_mode)
        if not isinstance(entries, list):
            raise ValueError(f"single_mode_sweeps.{mode} must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ValueError(f"Invalid entry in single_mode_sweeps.{mode}")
            level_name = str(entry["name"]).strip()
            if not level_name:
                raise ValueError(f"Empty level name in {mode}")
            override = dict(entry)
            if mode == "dropout":
                override["dropout_level"] = level_name
            elif mode == "instability":
                override["instability_level"] = level_name
            lookup.setdefault(mode, {})[level_name] = dict(override)
            variants.append(
                (_variant_name(mode, level_name), mode, override)
            )

    combos = spec.get("combos", [])
    if not isinstance(combos, list):
        raise ValueError("combos must be a list")
    for entry in combos:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError("Each combined entry must contain a name")
        name = str(entry["name"]).strip()
        use = entry.get("use")
        if not isinstance(use, dict) or set(map(str, use)) != {
            "dropout",
            "instability",
        }:
            raise ValueError(
                f"{name}: Combined must reference exactly dropout and instability"
            )
        merged: Dict[str, Any] = {}
        for mode in ("instability", "dropout"):
            level_name = str(use[mode]).strip()
            try:
                component = lookup[mode][level_name]
            except KeyError as error:
                raise ValueError(
                    f"{name}: missing {mode} level {level_name!r}"
                ) from error
            merged.update(component)
        if isinstance(entry.get("overrides"), dict):
            merged.update(dict(entry["overrides"]))
        variants.append((name, "combined", merged))

    names = {name for name, _mode, _override in variants}
    if names != EXPECTED_VARIANTS:
        raise ValueError(
            "The spec must expand to the ten agreed variants. "
            f"Missing={sorted(EXPECTED_VARIANTS - names)}, "
            f"extra={sorted(names - EXPECTED_VARIANTS)}"
        )
    return base, variants


_GENERATION_CFG_FIELDS = (
    "fps",
    "class_name",
    "severity",
    "dropout_enable",
    "dropout_p_start",
    "dropout_min_s",
    "dropout_max_s",
    "instability_enable",
    "instability_k_modes",
    "instability_p_switch",
    "instability_mode_xy_sigma_m",
    "instability_mode_yaw_sigma_rad",
    "instability_mode_lwh_sigma_rel",
    "instability_jitter_xy_sigma_m",
    "instability_jitter_yaw_sigma_rad",
    "instability_jitter_lwh_sigma_rel",
    "instability_p_yaw_random",
    "score_value",
    "score_mode",
    "score_dists",
    "dropout_level",
    "instability_level",
)


def _previous_variant_is_compatible(
    previous_entry: Optional[Mapping[str, Any]],
    cfg: VariantCfg,
    *,
    previous_seed_matches: bool,
) -> bool:
    if previous_entry is None or not previous_seed_matches:
        return False
    if str(previous_entry.get("gt_key", "original")) != "original":
        return False
    previous_cfg = previous_entry.get("resolved_cfg")
    if not isinstance(previous_cfg, dict):
        return False
    if bool(previous_cfg.get("confuser_enable", False)):
        return False
    current = asdict(cfg)
    return all(
        previous_cfg.get(field) == current.get(field)
        for field in _GENERATION_CFG_FIELDS
    )


def _severity_metadata(name: str, mode: str) -> Tuple[str, int]:
    if mode == "clean":
        return "Clean", -1
    for index, label in enumerate(("L1", "L2", "L3")):
        if name.endswith(f"_{label}"):
            return ("Mild", "Moderate", "Severe")[index], index
    return "", -1


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker-eval-generate-pseudo",
        description=(
            "Incrementally generate Clean, Dropout L1-L3, Instability L1-L3, "
            "and symmetric Combined L1-L3 pseudo detections against unchanged "
            "JRDB GT."
        ),
    )
    parser.add_argument("--split_root", required=True)
    parser.add_argument("--split_name", default=None)
    parser.add_argument("--labels_subdir", default="labels_3d")
    parser.add_argument(
        "--odometry_root",
        required=True,
        help=(
            "Stored in manifest.json for later --global_coords tracker runs; "
            "Generation itself stays in the original local coordinates."
        ),
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument(
        "--out_detections_subdir",
        default="detections_3D_pseudo",
        help="Output directory name under the split root.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score_dists",
        default=None,
        help="Optional JSON/NPZ TP score distribution overriding the YAML path.",
    )
    parser.add_argument("--include_variants", nargs="*", default=None)
    parser.add_argument("--exclude_variants", nargs="*", default=None)
    parser.add_argument(
        "--overwrite_variants",
        nargs="*",
        default=None,
        help="Comma/space-separated variant names whose existing JSONs are replaced.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    split_root = Path(args.split_root)
    split_name = str(args.split_name or split_root.name)
    labels_dir = split_root / args.labels_subdir
    spec_path = Path(args.spec)
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"GT labels directory not found: {labels_dir}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"Specification not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict):
        raise ValueError(f"Expected a YAML mapping in {spec_path}")
    base, all_variants = _expand_variants(spec)

    include = set(_split_values(args.include_variants)) or None
    exclude = set(_split_values(args.exclude_variants))
    variants = [
        item
        for item in all_variants
        if (include is None or item[0] in include) and item[0] not in exclude
    ]
    if not variants:
        raise ValueError("No variants remain selected")

    selected_names = {name for name, _mode, _override in variants}
    overwrite = set(_split_values(args.overwrite_variants))
    unknown_overwrite = overwrite - selected_names
    if unknown_overwrite:
        raise ValueError(
            f"--overwrite_variants contains unselected names: {sorted(unknown_overwrite)}"
        )

    gt_paths = sorted(labels_dir.glob("*.json"))
    if not gt_paths:
        raise FileNotFoundError(f"No GT JSON files found in {labels_dir}")
    class_name = str(base.get("class_name", "pedestrian"))
    sequences = [_load_sequence(path, class_name) for path in gt_paths]

    score_mode = str(base.get("score_mode", "constant")).strip().lower()
    score_path_value = args.score_dists or base.get("score_dists")
    score_sampler: Optional[ScoreSampler] = None
    score_path: Optional[Path] = None
    if score_mode == "sample":
        if not score_path_value:
            raise ValueError("score_mode=sample requires score_dists")
        score_path = Path(str(score_path_value))
        if not score_path.is_file():
            raise FileNotFoundError(f"Score distribution not found: {score_path}")
        score_sampler = ScoreSampler(_load_score_distribution(score_path))

    out_root = split_root / args.out_detections_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    previous_manifest_path = out_root / "manifest.json"
    previous_manifest: Dict[str, Any] = {}
    if previous_manifest_path.is_file():
        try:
            with previous_manifest_path.open("r", encoding="utf-8") as stream:
                loaded = json.load(stream)
            if isinstance(loaded, dict):
                previous_manifest = loaded
        except (OSError, json.JSONDecodeError):
            previous_manifest = {}
    previous_by_name = {
        str(entry.get("name")): entry
        for entry in previous_manifest.get("variants", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    previous_seed_matches = previous_manifest.get("seed") == int(args.seed)
    manifest_variants: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    if not args.quiet:
        print(f"[tracker_eval] GT:       {labels_dir}")
        print(f"[tracker_eval] Output:   {out_root}")
        print(f"[tracker_eval] Variants: {len(variants)}")
        print(
            "[tracker_eval] Existing sequence JSONs are reused only when the "
            "previous manifest confirms an identical standard-GT generation "
            "configuration and the variant is not explicitly overwritten."
        )

    for variant_index, (name, mode, override) in enumerate(variants, start=1):
        cfg = _apply_severity_once(
            _variant_cfg_from_dict(name, base, override)
        )
        if mode == "clean" and (cfg.dropout_enable or cfg.instability_enable):
            raise ValueError(f"{name}: Clean must not enable a corruption")
        if mode == "combined" and not (
            cfg.dropout_enable and cfg.instability_enable
        ):
            raise ValueError(
                f"{name}: Combined must enable both dropout and instability"
            )
        variant_dir = out_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        reused = 0
        previous_compatible = _previous_variant_is_compatible(
            previous_by_name.get(name),
            cfg,
            previous_seed_matches=previous_seed_matches,
        )

        for sequence in sequences:
            output_path = variant_dir / f"{sequence.name}.json"
            if (
                name not in overwrite
                and previous_compatible
                and output_path.is_file()
                and output_path.stat().st_size > 0
            ):
                reused += 1
                continue
            detections = _generate_sequence(
                sequence,
                cfg,
                int(args.seed),
                score_sampler,
            )
            _write_detections(
                output_path,
                detections,
                sequence.frame_keys,
                cfg.class_name,
            )
            generated += 1

        severity, severity_index = _severity_metadata(name, mode)
        manifest_variants.append(
            {
                "name": name,
                "failure_mode": mode,
                "severity": severity,
                "severity_index": severity_index,
                "override": override,
                "resolved_cfg": asdict(cfg),
                "detections_subdir": str(
                    Path(args.out_detections_subdir) / name
                ),
                "gt_key": "original",
                "labels_subdir": str(args.labels_subdir),
                "gt_tracker_name": "GT",
                "gt_tracker_name_global": "GT__global",
            }
        )
        status_rows.append(
            {
                "variant": name,
                "n_sequences": len(sequences),
                "n_generated": generated,
                "n_reused": reused,
                "overwrite_requested": name in overwrite,
                "previous_manifest_compatible": previous_compatible,
            }
        )
        if not args.quiet:
            print(
                f"[tracker_eval] ({variant_index}/{len(variants)}) {name}: "
                f"generated={generated}, reused={reused}"
            )

    manifest = {
        "manifest_format": "pseudo_detection_protocol",
        "protocol": "clean_dropout_instability_combined_standard_gt",
        "split_root": str(split_root),
        "split_name": split_name,
        "labels_subdir": str(args.labels_subdir),
        "out_detections_subdir": str(args.out_detections_subdir),
        "odometry_root": str(args.odometry_root),
        "seed": int(args.seed),
        "spec_path": str(spec_path),
        "spec_sha256": _sha256(spec_path),
        "base": base,
        "score_dists": str(score_path) if score_path is not None else None,
        "standard_gt_only": True,
        "gt_definitions": {
            "original": {
                "labels_subdir": str(args.labels_subdir),
                "gt_tracker_name": "GT",
                "gt_tracker_name_global": "GT__global",
            }
        },
        "variants": manifest_variants,
        "trackeval_groups": {
            "GT__global": [entry["name"] for entry in manifest_variants]
        },
    }
    manifest_path = out_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    variant_map_path = out_root / "variant_gt_map.csv"
    with variant_map_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "variant",
            "failure_mode",
            "severity",
            "detections_subdir",
            "labels_subdir",
            "gt_key",
            "gt_tracker_name",
            "gt_tracker_name_global",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for entry in manifest_variants:
            writer.writerow(
                {
                    "variant": entry["name"],
                    "failure_mode": entry["failure_mode"],
                    "severity": entry["severity"],
                    "detections_subdir": entry["detections_subdir"],
                    "labels_subdir": entry["labels_subdir"],
                    "gt_key": entry["gt_key"],
                    "gt_tracker_name": entry["gt_tracker_name"],
                    "gt_tracker_name_global": entry[
                        "gt_tracker_name_global"
                    ],
                }
            )

    status_path = out_root / "generation_status.csv"
    with status_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "variant",
                "n_sequences",
                "n_generated",
                "n_reused",
                "overwrite_requested",
                "previous_manifest_compatible",
            ],
        )
        writer.writeheader()
        writer.writerows(status_rows)

    if not args.quiet:
        print(f"[tracker_eval] Manifest: {manifest_path}")
        print(f"[tracker_eval] Variant/GT map: {variant_map_path}")
        print(f"[tracker_eval] Generation status: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
