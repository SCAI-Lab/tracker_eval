"""Convert source JRDB label JSON into TrackEval's GT directory layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tracker_eval.common.odometry_transform import (
    load_odometry_csv,
    transform_frame_data_to_global,
)
from tracker_eval.common.types import Box3D, Detection, FrameData
from tracker_eval.export.jrdb_kitti_writer import TrackRow3D, write_sequence_kitti_txt
from tracker_eval.utils import (
    _box7_from_label_obj,
    _load_labels_3d_json,
    _parse_frame_key,
    _parse_label_id_strict,
)


def prepare_gt(
    labels_dir: Path,
    gt_folder: Path,
    split: str,
    odometry_root: Optional[Path] = None,
) -> None:
    label_output = gt_folder / "label_02"
    label_output.mkdir(parents=True, exist_ok=True)
    seqmap_rows: List[str] = []

    for source in sorted(labels_dir.glob("*.json")):
        poses = (
            load_odometry_csv(
                str(odometry_root / split / "odometry" / f"{source.stem}.csv")
            )
            if odometry_root is not None
            else None
        )
        frame_objects = _load_labels_3d_json(source)
        rows: Dict[str, List[TrackRow3D]] = {}
        max_frame = -1
        for raw_frame, objects in frame_objects.items():
            frame = _parse_frame_key(raw_frame)
            frame_index = int(frame.split(".")[0])
            max_frame = max(max_frame, frame_index)
            detections: List[Detection] = []
            for obj in objects:
                label_id = obj.get("label_id")
                if label_id is None:
                    continue
                class_name, track_id = _parse_label_id_strict(label_id)
                if class_name.lower() != "pedestrian":
                    continue
                box7 = _box7_from_label_obj(obj)
                detections.append(
                    Detection(
                        frame_id=frame,
                        track_id=int(track_id),
                        box=Box3D.from_list(box7),
                        score=None,
                        label="pedestrian",
                    )
                )
            frame_data = FrameData(frame_id=frame, dets=detections)
            if poses is not None:
                frame_data = transform_frame_data_to_global(frame_data, poses)
            output_rows = [
                TrackRow3D(
                    track_id=int(detection.track_id),
                    box7=detection.box.as_list(),
                    score=None,
                )
                for detection in frame_data.dets
            ]
            rows[frame] = output_rows
        if max_frame < 0:
            raise ValueError(f"No frames found in {source}")
        write_sequence_kitti_txt(
            label_output / f"{source.stem}.txt",
            rows,
            class_name="pedestrian",
            bbox2d=(-1.0, -1.0, -1.0, -1.0),
            use_score=False,
        )
        seqmap_rows.append(f"{source.stem} 0 0 {max_frame + 1}\n")

    if not seqmap_rows:
        raise FileNotFoundError(f"No label JSON files found in {labels_dir}")
    (gt_folder / f"evaluate_tracking.seqmap.{split}").write_text(
        "".join(seqmap_rows), encoding="utf-8"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker-eval-prepare-gt",
        description="Create label_02 and seqmap files for bundled TrackEval.",
    )
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--gt-folder", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--global-coords",
        action="store_true",
        help="Transform source labels with row-aligned odometry before export.",
    )
    parser.add_argument(
        "--odometry-root",
        type=Path,
        default=None,
        help="Root containing <split>/odometry/<sequence>.csv.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if args.global_coords and args.odometry_root is None:
        raise ValueError("--global-coords requires --odometry-root")
    prepare_gt(
        args.labels_dir,
        args.gt_folder,
        str(args.split),
        args.odometry_root if args.global_coords else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
