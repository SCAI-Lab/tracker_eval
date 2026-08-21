"""Run the bundled, corrected JRDB 3D TrackEval evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import trackeval


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker-eval-evaluate",
        description="Evaluate KITTI/JRDB 3D tracker text files with HOTA, CLEAR and Identity.",
    )
    parser.add_argument("--gt-folder", required=True, type=Path)
    parser.add_argument("--trackers-folder", required=True, type=Path)
    parser.add_argument("--output-folder", required=True, type=Path)
    parser.add_argument("--split", required=True, help="Name used by evaluate_tracking.seqmap.<split>.")
    parser.add_argument("--trackers", nargs="*", default=None)
    parser.add_argument(
        "--tracker-subfolder",
        default=None,
        help="Default: <split>/data, matching tracker-eval output layout.",
    )
    parser.add_argument("--output-subfolder", default="")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    for path, label in (
        (args.gt_folder, "GT folder"),
        (args.trackers_folder, "trackers folder"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    args.output_folder.mkdir(parents=True, exist_ok=True)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "USE_PARALLEL": bool(args.parallel),
            "NUM_PARALLEL_CORES": int(args.workers),
            "PRINT_RESULTS": not args.quiet,
            "PRINT_CONFIG": not args.quiet,
            "TIME_PROGRESS": not args.quiet,
            "PLOT_CURVES": False,
            "SKIP_EXISTING": bool(args.skip_existing),
        }
    )
    dataset_config = trackeval.datasets.JRDB3DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(args.gt_folder),
            "TRACKERS_FOLDER": str(args.trackers_folder),
            "OUTPUT_FOLDER": str(args.output_folder),
            "TRACKERS_TO_EVAL": args.trackers,
            "CLASSES_TO_EVAL": ["pedestrian"],
            "SPLIT_TO_EVAL": str(args.split),
            "TRACKER_SUB_FOLDER": str(args.tracker_subfolder or f"{args.split}/data"),
            "OUTPUT_SUB_FOLDER": str(args.output_subfolder),
            "PRINT_CONFIG": not args.quiet,
        }
    )

    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(config={"THRESHOLD": 0.3, "PRINT_CONFIG": not args.quiet}),
        trackeval.metrics.Identity(config={"PRINT_CONFIG": not args.quiet}),
    ]
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.JRDB3DBox(dataset_config)
    evaluator.evaluate([dataset], metrics, is_3d=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
