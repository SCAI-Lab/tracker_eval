# tracker_eval/cli/convert_gt_to_kitti_3d.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tracker_eval.export.jrdb_kitti_writer import TrackRow3D, write_sequence_kitti_txt


def _parse_frame_key(k: str) -> str:
    # Keep original frame keys (e.g. "000123.pcd") because writer can parse to int
    return str(k)


def _parse_label_id(label_id: str) -> Tuple[str, int]:
    """
    "pedestrian:18" -> ("pedestrian", 18)
    """
    s = str(label_id)
    if ":" not in s:
        raise ValueError(f"Unexpected label_id format (no ':'): {label_id}")
    cls, tid = s.split(":", 1)
    cls = cls.strip()
    tid_i = int(tid)
    return cls, tid_i


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


def _load_labels_3d_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns dict: frame_key -> list of label objects.

    JRDB label json variants:
      - {"labels": { "000123.pcd": [ {...}, ... ], ... }}
      - {"annotations": {...}} / {"frames": {...}} / {"data": {...}}
      - or direct dict { "000123.pcd": [ ... ], ... }
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("labels", "annotations", "frames", "data"):
        if key in data and isinstance(data[key], dict):
            return data[key]  # type: ignore[return-value]

    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        return data  # type: ignore[return-value]

    raise ValueError(f"Could not find per-frame labels dict in {path}")


def convert_split_gt(
    *,
    split_root: Path,
    out_root: Path,
    split_name: Optional[str] = None,
    labels_subdir: str = "labels_3d",
    out_tracker_name: str = "GT",
    tracker_subfolder: str = "data",
    class_name_out: str = "pedestrian",
    include_sequences: Optional[List[str]] = None,
    exclude_sequences: Optional[List[str]] = None,
    use_score: bool = False,
    bbox2d: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0),
    verbose: bool = True,
) -> None:
    """
    Convert JRDB labels_3d JSON into KITTI-tracking style txt compatible with JRDB3DBox evaluation.

    Input:
      <split_root>/<labels_subdir>/<seq>.json   (default labels_subdir="labels_3d")

    Output:
      <out_root>/<out_tracker_name>/<split_name>/<tracker_subfolder>/<seq>.txt
        e.g. /mnt/nvme/tracker_eval_outputs/GT/train_val/data/<seq>.txt
    """
    split_root = Path(split_root)
    if split_name is None:
        split_name = split_root.name

    labels_dir = split_root / labels_subdir
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    out_dir = out_root / out_tracker_name / split_name / tracker_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_files = sorted(labels_dir.glob("*.json"))

    if include_sequences is not None:
        include_set = set(include_sequences)
        seq_files = [p for p in seq_files if p.stem in include_set]
    if exclude_sequences is not None:
        exclude_set = set(exclude_sequences)
        seq_files = [p for p in seq_files if p.stem not in exclude_set]

    if verbose:
        print(f"[tracker_eval] GT convert split '{split_name}': {len(seq_files)} sequence(s) found in {labels_dir}")
        print(f"[tracker_eval] Writing GT txt to: {out_dir}")

    for i, seq_path in enumerate(seq_files):
        seq = seq_path.stem
        if verbose:
            print(f"[tracker_eval] ({i+1}/{len(seq_files)}) Converting {seq} ...")

        frame_dict = _load_labels_3d_json(seq_path)

        tracks_by_frame: Dict[str, List[TrackRow3D]] = {}

        for frame_key_raw, objs in frame_dict.items():
            frame_key = _parse_frame_key(frame_key_raw)
            rows: List[TrackRow3D] = []
            for obj in objs:
                label_id = obj.get("label_id", None)
                if label_id is None:
                    continue
                cls, tid = _parse_label_id(label_id)
                if cls.lower() != "pedestrian":
                    continue
                box7 = _box7_from_label_obj(obj)
                rows.append(TrackRow3D(track_id=int(tid), box7=box7, score=None))
            tracks_by_frame[frame_key] = rows

        out_txt = out_dir / f"{seq}.txt"
        write_sequence_kitti_txt(
            out_txt,
            tracks_by_frame,
            class_name=class_name_out,
            truncated=0,
            occluded=0,
            alpha=-1.0,
            bbox2d=bbox2d,
            use_score=bool(use_score),
            sort_rows=True,
        )

    if verbose:
        print(f"[tracker_eval] GT conversion done for split '{split_name}'.")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracker_eval.convert_gt_to_kitti_3d",
        description="Convert JRDB labels_3d JSON into JRDB3DBox-compatible KITTI-tracking txt files.",
    )

    p.add_argument(
        "--split_root",
        type=str,
        nargs="+",
        required=True,
        help="One or more split roots, e.g. /mnt/nvme/JRDB_track/train_val /mnt/nvme/JRDB_track/test",
    )
    p.add_argument(
        "--split_name",
        type=str,
        nargs="*",
        default=None,
        help="Optional split name(s). Must match number of split_root if provided.",
    )
    p.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root, e.g. /mnt/nvme/tracker_eval_outputs",
    )

    p.add_argument(
        "--labels_subdir",
        type=str,
        default="labels_3d",
        help="Labels subfolder under split_root (default: labels_3d)",
    )
    p.add_argument(
        "--tracker_subfolder",
        type=str,
        default="data",
        help="Subfolder under .../GT/<split>/ (default: data)",
    )

    p.add_argument("--include_sequences", type=str, nargs="*", default=None)
    p.add_argument("--exclude_sequences", type=str, nargs="*", default=None)

    p.add_argument("--use_score", action="store_true", help="Write a score column (default false for GT)")
    p.add_argument(
        "--bbox2d_zero",
        action="store_true",
        help="Write bbox2d as 0s instead of -1s (default: -1).",
    )
    p.add_argument("--quiet", action="store_true")
    return p


def _normalize_split_names(split_roots: List[str], split_names: Optional[List[str]]) -> List[str]:
    roots = [Path(r) for r in split_roots]
    if split_names is None or len(split_names) == 0:
        return [p.name for p in roots]
    if len(split_names) != len(roots):
        raise ValueError("If --split_name is provided, it must match the number of --split_root values.")
    return [str(x) for x in split_names]


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    split_roots = [str(x) for x in args.split_root]
    split_names = _normalize_split_names(split_roots, args.split_name)

    bbox2d = (0.0, 0.0, 0.0, 0.0) if args.bbox2d_zero else (-1.0, -1.0, -1.0, -1.0)

    for root, name in zip(split_roots, split_names):
        convert_split_gt(
            split_root=Path(root),
            split_name=name,
            out_root=Path(args.out_root),
            labels_subdir=str(args.labels_subdir),
            out_tracker_name="GT",
            tracker_subfolder=str(args.tracker_subfolder),
            include_sequences=args.include_sequences,
            exclude_sequences=args.exclude_sequences,
            use_score=bool(args.use_score),
            bbox2d=bbox2d,
            verbose=not bool(args.quiet),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
