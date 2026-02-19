# tracker_eval/cli/convert_gt_to_kitti_3d.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tracker_eval.export.jrdb_kitti_writer import TrackRow3D, write_sequence_kitti_txt
from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.common.odometry_transform import load_odometry_csv, transform_frame_data_to_global

from tracker_eval.utils import (
    _parse_frame_key,
    _parse_label_id_strict,
    _box7_from_label_obj,
    _load_labels_3d_json,
)


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
    global_coords: bool = False,
    odometry_root: Optional[Path] = None,
) -> None:
    """
    Convert JRDB labels_3d JSON into KITTI-tracking style txt compatible with JRDB3DBox evaluation.

    Input:
      <split_root>/<labels_subdir>/<seq>.json   (default labels_subdir="labels_3d")

    Output:
      <out_root>/<out_tracker_name>/<split_name>/<tracker_subfolder>/<seq>.txt
        e.g. /mnt/nvme/tracker_eval_outputs/GT/train/data/<seq>.txt
    """
    split_root = Path(split_root)
    if split_name is None:
        split_name = split_root.name

    labels_dir = split_root / labels_subdir
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    out_dir_local = out_root / out_tracker_name / split_name / tracker_subfolder
    out_dir_local.mkdir(parents=True, exist_ok=True)

    out_dir_global = None
    odo_base = None
    if global_coords:
        if odometry_root is None or str(odometry_root).strip() == "":
            raise ValueError("--global_coords requires --odometry_root")
        out_dir_global = out_root / f"{out_tracker_name}__global" / split_name / tracker_subfolder
        out_dir_global.mkdir(parents=True, exist_ok=True)

        # Expected layout: <odometry_root>/<split_name>/odometry/<seq>.csv
        odo_base = Path(odometry_root) / split_name / "odometry"


    seq_files = sorted(labels_dir.glob("*.json"))

    if include_sequences is not None:
        include_set = set(include_sequences)
        seq_files = [p for p in seq_files if p.stem in include_set]
    if exclude_sequences is not None:
        exclude_set = set(exclude_sequences)
        seq_files = [p for p in seq_files if p.stem not in exclude_set]

    if verbose:
        print(f"[tracker_eval] GT convert split '{split_name}': {len(seq_files)} sequence(s) found in {labels_dir}")
        print(f"[tracker_eval] Writing GT txt to: {out_dir_local}")
        if global_coords:
            print(f"[tracker_eval] Writing GT__global txt to: {out_dir_global}")
            print(f"[tracker_eval] Using odometry from: {odo_base}")


    for i, seq_path in enumerate(seq_files):
        seq = seq_path.stem
        if verbose:
            print(f"[tracker_eval] ({i+1}/{len(seq_files)}) Converting {seq} ...")

        frame_dict = _load_labels_3d_json(seq_path)

        tracks_local_by_frame: Dict[str, List[TrackRow3D]] = {}
        tracks_global_by_frame: Optional[Dict[str, List[TrackRow3D]]] = {} if global_coords else None

        pose_by_idx = None
        if global_coords:
            odo_csv = (odo_base / f"{seq}.csv")  # type: ignore[operator]
            if not odo_csv.exists():
                raise FileNotFoundError(f"Missing odometry CSV for {seq}: {odo_csv}")
            pose_by_idx = load_odometry_csv(str(odo_csv))

        for frame_key_raw, objs in frame_dict.items():
            frame_key = _parse_frame_key(frame_key_raw)

            dets_local: List[Detection] = []
            rows_local: List[TrackRow3D] = []

            for obj in objs:
                label_id = obj.get("label_id", None)
                if label_id is None:
                    continue
                cls, tid = _parse_label_id_strict(label_id)
                if cls.lower() != "pedestrian":
                    continue

                box7 = _box7_from_label_obj(obj)  # [cx,cy,cz,l,w,h,rot_z]
                b = Box3D(
                    cx=float(box7[0]),
                    cy=float(box7[1]),
                    cz=float(box7[2]),
                    l=float(box7[3]),
                    w=float(box7[4]),
                    h=float(box7[5]),
                    rot_z=float(box7[6]),
                )

                det = Detection(
                    frame_id=str(frame_key),
                    track_id=int(tid),
                    box=b,
                    score=None,
                    label="pedestrian",
                    raw_label_id=str(label_id),
                )
                dets_local.append(det)

                rows_local.append(TrackRow3D(track_id=int(tid), box7=box7, score=None))

            tracks_local_by_frame[frame_key] = rows_local

            if global_coords and tracks_global_by_frame is not None:
                if pose_by_idx is None:
                    raise RuntimeError("global_coords=True but pose_by_idx is None (odometry not loaded).")

                fd_local = FrameData(frame_id=str(frame_key), dets=dets_local)
                fd_global = transform_frame_data_to_global(fd_local, pose_by_idx)

                rows_global: List[TrackRow3D] = []
                for detg in fd_global.dets:
                    bg = detg.box
                    box7g = np.asarray([bg.cx, bg.cy, bg.cz, bg.l, bg.w, bg.h, bg.rot_z], dtype=np.float32)
                    rows_global.append(TrackRow3D(track_id=int(detg.track_id), box7=box7g, score=None))

                tracks_global_by_frame[frame_key] = rows_global

        # Local GT
        out_txt_local = out_dir_local / f"{seq}.txt"
        write_sequence_kitti_txt(
            out_txt_local,
            tracks_local_by_frame,
            class_name=class_name_out,
            truncated=0,
            occluded=0,
            alpha=-1.0,
            bbox2d=bbox2d,
            use_score=bool(use_score),
            sort_rows=True,
        )

        # Global GT
        if global_coords and out_dir_global is not None and tracks_global_by_frame is not None:
            out_txt_global = out_dir_global / f"{seq}.txt"
            write_sequence_kitti_txt(
                out_txt_global,
                tracks_global_by_frame,
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
        help="One or more split roots, e.g. /mnt/nvme/JRDB_track/train /mnt/nvme/JRDB_track/test",
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

    p.add_argument(
        "--global_coords",
        action="store_true",
        help="If set: also export GT transformed to global coordinates using odometry.",
    )
    p.add_argument(
        "--odometry_root",
        type=str,
        default="",
        help="Root containing odometry CSVs at <odometry_root>/<split_name>/odometry/<seq>.csv. "
            "Required if --global_coords is set.",
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
            global_coords=bool(args.global_coords),
            odometry_root=Path(args.odometry_root) if str(args.odometry_root).strip() != "" else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
