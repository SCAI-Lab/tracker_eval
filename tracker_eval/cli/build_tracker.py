"""Build optional native tracker components."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tracker_eval.trackers.paths import IMPLEMENTATIONS_DIR


def build_gnnpmb(build_dir: Optional[Path] = None) -> Path:
    source_dir = IMPLEMENTATIONS_DIR / "gnnpmb" / "murty"
    output_dir = source_dir
    if build_dir is None:
        repository_root = Path(__file__).resolve().parents[2]
        build_dir = repository_root / ".build" / "gnnpmb-murty"
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    pybind11_dir = subprocess.run(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DMURTY_OUTPUT_DIR={output_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--parallel"],
        check=True,
    )
    artifacts = sorted(output_dir.glob("_murty*.so"))
    if not artifacts:
        raise RuntimeError(f"Murty build completed but no extension appeared in {output_dir}")
    return artifacts[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build optional native components. GNN-PMB requires Eigen3 "
            "(Ubuntu: sudo apt install libeigen3-dev)."
        )
    )
    parser.add_argument("component", choices=["gnnpmb"])
    parser.add_argument("--build-dir", type=Path, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.component == "gnnpmb":
        artifact = build_gnnpmb(args.build_dir)
        print(f"Built GNN-PMB Murty extension: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
