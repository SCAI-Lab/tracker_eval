"""Run all tracker/variant/sequence jobs from a pseudo-detection manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tracker_eval.cli.run_tracker import (
    build_argparser as build_tracker_argparser,
    execute_requests,
    make_requests,
    split_values,
)


def _has_option(argv: Sequence[str], *names: str) -> bool:
    return any(
        token in names
        or any(token.startswith(name + "=") for name in names)
        for token in argv
    )


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("manifest_format") != "pseudo_detection_protocol":
        raise ValueError("Expected a tracker_eval pseudo-detection manifest")
    if not bool(manifest.get("standard_gt_only", False)):
        raise ValueError("The pseudo protocol expects a standard-GT-only manifest")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Manifest contains no variants")
    for entry in variants:
        if str(entry.get("gt_key", "original")) != "original":
            raise ValueError(
                f"Variant {entry.get('name')!r} does not use standard GT"
            )
    return manifest


def _wrapper_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "-h" in raw or "--help" in raw:
        parser = build_tracker_argparser(split_root_required=False)
        parser.prog = "tracker-eval-pseudo"
        parser.add_argument(
            "--manifest",
            required=True,
            help="Standard-GT pseudo-detection manifest.json",
        )
        parser.add_argument("--dry_run", action="store_true")
        parser.print_help()
        return 0

    wrapper, remaining = _wrapper_parser().parse_known_args(raw)
    if not wrapper.manifest:
        raise ValueError("--manifest is required")
    manifest_path = Path(wrapper.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)

    if not _has_option(remaining, "--split_root"):
        remaining += ["--split_root", str(manifest["split_root"])]
    if not _has_option(remaining, "--split_name"):
        remaining += ["--split_name", str(manifest["split_name"])]
    if not _has_option(remaining, "--detections_subdir"):
        remaining += [
            "--detections_subdir",
            str(manifest["out_detections_subdir"]),
        ]
    if _has_option(remaining, "--global_coords"):
        if not _has_option(remaining, "--odometry_root"):
            remaining += [
                "--odometry_root",
                str(manifest["odometry_root"]),
            ]

    parser = build_tracker_argparser(split_root_required=False)
    parser.prog = "tracker-eval-pseudo"
    args = parser.parse_args(remaining)
    requested = set(split_values(args.variants))
    excluded = set(split_values(args.exclude_variants))

    overrides: List[Dict[str, str]] = []
    for entry in manifest["variants"]:
        name = str(entry.get("name", "")).strip()
        if not name or (requested and name not in requested):
            continue
        if name in excluded:
            continue
        overrides.append(
            {
                "name": name,
                "detections_subdir": str(entry["detections_subdir"]),
                "labels_subdir": str(entry["labels_subdir"]),
            }
        )
    if not overrides:
        raise ValueError("No pseudo variants remain selected")

    requests = make_requests(args, variant_overrides=overrides)
    if wrapper.dry_run:
        print(
            f"[tracker_eval] Dry run: {len(requests)} "
            "tracker/mode/variant/split run(s)"
        )
        for request in requests:
            print(
                f"  {request.tracker_name}: "
                f"detections={request.detections_subdir}, "
                f"labels={request.labels_subdir}"
            )
        return 0
    return execute_requests(args, requests)


if __name__ == "__main__":
    raise SystemExit(main())
