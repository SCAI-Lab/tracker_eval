from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Headless backend for PNG/video rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class Box2D:
    cx: float
    cy: float
    l: float
    w: float
    yaw: float  # rad, around +z


@dataclass(frozen=True)
class TrackItem:
    track_id: int
    box: Box2D


# ----------------------------
# CLI
# ----------------------------

def _parse_tuple2_floats(vals: List[str], name: str) -> Tuple[float, float]:
    if len(vals) != 2:
        raise ValueError(f"{name} expects exactly 2 numbers, got: {vals}")
    return float(vals[0]), float(vals[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "Visualize tracker_eval outputs (predicted tracks vs GT) in BEV and export an MP4."
    )
    ap.add_argument("--split_root", required=True, type=str,
                    help="Split root containing labels_3d/ and detections_3D/ (e.g. /mnt/nvme/JRDB_track/test)")
    ap.add_argument("--out_root", required=True, type=str,
                    help="tracker_eval output root (e.g. /mnt/nvme/tracker_eval_outputs)")
    ap.add_argument("--tracker", required=True, type=str,
                    help="Tracker output folder name inside out_root (e.g. ab3dmot, simpletrack, fastpoly)")
    ap.add_argument("--sequence", required=True, type=str,
                    help="Sequence name (e.g. cubberly-auditorium-2019-04-22_1)")

    ap.add_argument("--out_mp4", required=True, type=str,
                    help="Output mp4 path")

    ap.add_argument("--mode", choices=["centers", "boxes"], default="boxes",
                    help="centers: plot centers only; boxes: plot BEV rectangles + center dot")
    ap.add_argument("--xlim", nargs=2, required=True,
                    help="x limits in meters: xmin xmax")
    ap.add_argument("--ylim", nargs=2, required=True,
                    help="y limits in meters: ymin ymax")

    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=140)

    ap.add_argument("--workers", type=int, default=-1,
                    help="PNG rendering workers. Default = cpu_count()-1.")
    ap.add_argument("--chunksize", type=int, default=8,
                    help="multiprocessing chunksize for imap.")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep temp folder for debugging.")

    ap.add_argument("--title", type=str, default="tracker_eval BEV visualization")

    return ap.parse_args()


# ----------------------------
# IO: GT labels
# ----------------------------

def _parse_box7(obj_box) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns (cx, cy, cz, l, w, h, yaw)
    Strict.
    """
    if isinstance(obj_box, dict):
        needed = ["cx", "cy", "cz", "l", "w", "h", "rot_z"]
        for k in needed:
            if k not in obj_box:
                raise ValueError(f"GT box dict missing key '{k}': {obj_box.keys()}")
        return (
            float(obj_box["cx"]), float(obj_box["cy"]), float(obj_box["cz"]),
            float(obj_box["l"]), float(obj_box["w"]), float(obj_box["h"]),
            float(obj_box["rot_z"]),
        )
    if isinstance(obj_box, list) or isinstance(obj_box, tuple):
        if len(obj_box) != 7:
            raise ValueError(f"GT box list must have len=7, got len={len(obj_box)}")
        return tuple(float(x) for x in obj_box)  # type: ignore
    raise TypeError(f"Unsupported GT box type: {type(obj_box)}")


def load_gt_tracks_by_frame(labels_json: Path) -> Dict[int, List[TrackItem]]:
    """
    Returns dict: frame_index -> list[TrackItem] for GT.
    frame_index is int parsed from '000123.pcd' -> 123.
    track_id from label_id numeric suffix (e.g. 'pedestrian:14' -> 14).
    """
    if not labels_json.exists():
        raise FileNotFoundError(labels_json)

    with labels_json.open("r") as f:
        d = json.load(f)

    if "labels" not in d or not isinstance(d["labels"], dict):
        raise ValueError("labels json must contain top-level dict key 'labels'")

    out: Dict[int, List[TrackItem]] = {}
    labels: dict = d["labels"]

    for frame_name, objs in labels.items():
        if not isinstance(frame_name, str) or not frame_name.endswith(".pcd"):
            raise ValueError(f"Unexpected frame key: {frame_name}")
        frame_idx_str = frame_name.split(".")[0]
        if not frame_idx_str.isdigit():
            raise ValueError(f"Frame key is not numeric: {frame_name}")
        frame_idx = int(frame_idx_str)

        if not isinstance(objs, list):
            raise ValueError(f"Frame '{frame_name}' must map to a list of objects.")

        frame_items: List[TrackItem] = []
        for o in objs:
            if not isinstance(o, dict):
                raise ValueError("Each GT object must be a dict.")

            if "label_id" not in o:
                raise ValueError("GT object missing 'label_id'")
            label_id = str(o["label_id"])
            if ":" not in label_id:
                raise ValueError(f"GT label_id must contain ':': {label_id}")
            tid_str = label_id.split(":")[-1]
            if not tid_str.isdigit():
                raise ValueError(f"GT label_id numeric suffix must be int: {label_id}")
            track_id = int(tid_str)

            if "box" not in o:
                raise ValueError("GT object missing 'box'")
            cx, cy, cz, l, w, h, yaw = _parse_box7(o["box"])
            _ = cz, h  # BEV ignores cz/h but they must exist

            frame_items.append(TrackItem(track_id=track_id, box=Box2D(cx=cx, cy=cy, l=l, w=w, yaw=yaw)))

        out[frame_idx] = frame_items

    return out


# ----------------------------
# IO: Predictions (KITTI-like txt)
# ----------------------------

def load_pred_tracks_by_frame(pred_txt: Path) -> Dict[int, List[TrackItem]]:
    """
    Parse tracker_eval's JRDB-toolkit-style txt written by jrdb_kitti_writer.py.

    Each line (18 cols if score, 17 otherwise):
      0 frame
      1 track_id
      ...
      10..16 box7 in JRDB-toolkit convention:
          (x, y_top, z, l, h, w, theta)

    For BEV we should plot in the ground plane (x,z) if y is vertical (as JRDB IoU code assumes),
    so we map:
      bev_cx = x
      bev_cy = z
      bev_l  = l
      bev_w  = w
      bev_yaw = theta

    We also recover center height if needed:
      y_center = y_top - 0.5*h
    (not used for BEV).
    """
    if not pred_txt.exists():
        raise FileNotFoundError(pred_txt)

    out: Dict[int, List[TrackItem]] = {}
    with pred_txt.open("r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 17:
                raise ValueError(
                    f"Pred file {pred_txt} line {line_no}: expected >=17 columns, got {len(parts)}"
                )

            frame = int(float(parts[0]))
            tid = int(float(parts[1]))

            # JRDB-toolkit box convention written by jrdb_kitti_writer:
            # (x, y_top, z, l, h, w, theta)
            x = float(parts[10])
            y_top = float(parts[11])
            z = float(parts[12])      # not used for xy BEV
            l = float(parts[13])
            h = float(parts[14])      # vertical size (used only to recover y_center)
            w = float(parts[15])
            theta = float(parts[16])

            # Recover the *center* y in ground plane:
            y = y_top - 0.5 * h

            # BEV in x-y:
            box2d = Box2D(cx=x, cy=y, l=l, w=w, yaw=theta)

            # BEV uses x-z plane because y is vertical in JRDB-toolkit IoU.
            out.setdefault(frame, []).append(
                TrackItem(
                    track_id=tid,
                    box=box2d,
                )
            )

    return out



# ----------------------------
# Geometry: rectangle corners in BEV
# ----------------------------

def bev_box_corners(box: Box2D) -> np.ndarray:
    """
    Returns (4,2) corners in CCW order.
    JRDB base frame convention assumed:
      x forward, y left, yaw about +z.
    """
    cx, cy, l, w, yaw = box.cx, box.cy, box.l, box.w, box.yaw
    # local rectangle corners around origin (cx,cy): forward is +x
    dx = l * 0.5
    dy = w * 0.5
    local = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ], dtype=np.float64)

    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = local @ R.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


# ----------------------------
# Color mapping
# ----------------------------

def build_id_color_map(ids: List[int], cmap_name: str) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Deterministic mapping: distribute ids across a colormap.
    """
    if not ids:
        return {}
    ids_sorted = sorted(set(ids))
    cmap = plt.get_cmap(cmap_name)
    # Avoid very light/near-white endpoints
    lo, hi = 0.25, 0.95
    n = len(ids_sorted)
    colors = [cmap(lo + (hi - lo) * (i / max(1, n - 1))) for i in range(n)]
    return {tid: colors[i] for i, tid in enumerate(ids_sorted)}


# ----------------------------
# Precompute to NPZ for parallel rendering
# ----------------------------

def save_frame_npz(
    out_dir: Path,
    frame_idx: int,
    gt_items: List[TrackItem],
    pred_items: List[TrackItem],
) -> None:
    """
    Save arrays:
      gt:  (N, 6) [id, cx, cy, l, w, yaw]
      pred:(M, 6)
    """
    gt_arr = np.array([[it.track_id, it.box.cx, it.box.cy, it.box.l, it.box.w, it.box.yaw] for it in gt_items],
                      dtype=np.float64) if gt_items else np.zeros((0, 6), dtype=np.float64)
    pr_arr = np.array([[it.track_id, it.box.cx, it.box.cy, it.box.l, it.box.w, it.box.yaw] for it in pred_items],
                      dtype=np.float64) if pred_items else np.zeros((0, 6), dtype=np.float64)
    np.savez_compressed(str(out_dir / f"frame_{frame_idx:06d}.npz"), frame_idx=frame_idx, gt=gt_arr, pred=pr_arr)


# ----------------------------
# Worker rendering
# ----------------------------

_W_NPZ_DIR: Optional[Path] = None
_W_PNG_DIR: Optional[Path] = None
_W_MODE: str = "boxes"
_W_XLIM: Tuple[float, float] = (-10.0, 10.0)
_W_YLIM: Tuple[float, float] = (-10.0, 10.0)
_W_DPI: int = 140
_W_TITLE: str = ""
_W_GT_COLORS: Dict[int, Tuple[float, float, float, float]] = {}
_W_PR_COLORS: Dict[int, Tuple[float, float, float, float]] = {}

def _worker_init(npz_dir: str, png_dir: str, mode: str,
                 xlim: Tuple[float, float], ylim: Tuple[float, float],
                 dpi: int, title: str,
                 gt_colors: Dict[int, Tuple[float, float, float, float]],
                 pr_colors: Dict[int, Tuple[float, float, float, float]]) -> None:
    global _W_NPZ_DIR, _W_PNG_DIR, _W_MODE, _W_XLIM, _W_YLIM, _W_DPI, _W_TITLE, _W_GT_COLORS, _W_PR_COLORS
    _W_NPZ_DIR = Path(npz_dir)
    _W_PNG_DIR = Path(png_dir)
    _W_MODE = mode
    _W_XLIM = xlim
    _W_YLIM = ylim
    _W_DPI = int(dpi)
    _W_TITLE = title
    _W_GT_COLORS = dict(gt_colors)
    _W_PR_COLORS = dict(pr_colors)


def _render_one(frame_idx: int) -> int:
    assert _W_NPZ_DIR is not None and _W_PNG_DIR is not None

    d = np.load(str(_W_NPZ_DIR / f"frame_{frame_idx:06d}.npz"), allow_pickle=False)
    gt = d["gt"]  # (N,6)
    pr = d["pred"]  # (M,6)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(f"{_W_TITLE} | frame {frame_idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(_W_XLIM[0], _W_XLIM[1])
    ax.set_ylim(_W_YLIM[0], _W_YLIM[1])
    ax.grid(True, linewidth=0.3, alpha=0.35)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # GT first (reds), then Pred (blues) so pred is visually on top
    if _W_MODE == "centers":
        if gt.shape[0] > 0:
            for row in gt:
                tid = int(row[0])
                cx, cy = float(row[1]), float(row[2])
                ax.scatter([cx], [cy], s=28, marker="o",
                           color=_W_GT_COLORS[tid], edgecolors="none", alpha=0.95)
        if pr.shape[0] > 0:
            for row in pr:
                tid = int(row[0])
                cx, cy = float(row[1]), float(row[2])
                ax.scatter([cx], [cy], s=28, marker="o",
                           color=_W_PR_COLORS[tid], edgecolors="black", linewidths=0.3, alpha=0.95)
    else:
        # boxes mode
        if gt.shape[0] > 0:
            for row in gt:
                tid = int(row[0])
                box = Box2D(cx=float(row[1]), cy=float(row[2]), l=float(row[3]), w=float(row[4]), yaw=float(row[5]))
                corners = bev_box_corners(box)
                poly = Polygon(corners, closed=True, fill=False, linewidth=1.2,
                               edgecolor=_W_GT_COLORS[tid], alpha=0.95)
                ax.add_patch(poly)
                ax.scatter([box.cx], [box.cy], s=14, marker="o", color=_W_GT_COLORS[tid], edgecolors="none")
        if pr.shape[0] > 0:
            for row in pr:
                tid = int(row[0])
                box = Box2D(cx=float(row[1]), cy=float(row[2]), l=float(row[3]), w=float(row[4]), yaw=float(row[5]))
                corners = bev_box_corners(box)
                poly = Polygon(corners, closed=True, fill=False, linewidth=1.2,
                               edgecolor=_W_PR_COLORS[tid], alpha=0.95)
                ax.add_patch(poly)
                ax.scatter([box.cx], [box.cy], s=14, marker="o",
                           color=_W_PR_COLORS[tid], edgecolors="black", linewidths=0.3)

    # legend (minimal)
    ax.text(0.01, 0.99, "GT (reds) vs Pred (blues)", transform=ax.transAxes,
            ha="left", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85))

    out_png = _W_PNG_DIR / f"frame_{frame_idx:06d}.png"
    fig.savefig(str(out_png), dpi=_W_DPI)
    plt.close(fig)
    return frame_idx


# ----------------------------
# ffmpeg
# ----------------------------

def stitch_mp4_from_pngs(png_dir: Path, out_mp4: Path, fps: int) -> None:
    pattern = str(png_dir / "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(int(fps)),
        "-i", pattern,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stdout}")


# ----------------------------
# Main
# ----------------------------

def _auto_workers(user_workers: int) -> int:
    if user_workers is not None and int(user_workers) > 0:
        return int(user_workers)
    ncpu = os.cpu_count() or 1
    return max(1, ncpu - 1)


def main() -> int:
    args = parse_args()
    split_root = Path(args.split_root)
    out_root = Path(args.out_root)
    tracker = args.tracker
    seq = args.sequence
    out_mp4 = Path(args.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    xlim = _parse_tuple2_floats(args.xlim, "--xlim")
    ylim = _parse_tuple2_floats(args.ylim, "--ylim")

    gt_json = split_root / "labels_3d" / f"{seq}.json"
    pred_txt = out_root / tracker / "data" / f"{seq}.txt"

    gt_by_frame = load_gt_tracks_by_frame(gt_json)
    pr_by_frame = load_pred_tracks_by_frame(pred_txt)

    # frames: intersection is allowed, but we want consistent timeline based on GT
    # Strict: we visualize exactly GT frames, missing pred => empty list.
    frames = sorted(gt_by_frame.keys())
    if not frames:
        raise RuntimeError("No frames found in GT labels.")

    # collect ids for stable coloring across entire sequence
    gt_ids_all: List[int] = []
    pr_ids_all: List[int] = []
    for fr in frames:
        gt_ids_all.extend([it.track_id for it in gt_by_frame.get(fr, [])])
        pr_ids_all.extend([it.track_id for it in pr_by_frame.get(fr, [])])

    gt_colors = build_id_color_map(gt_ids_all, cmap_name="Reds")
    pr_colors = build_id_color_map(pr_ids_all, cmap_name="Blues")

    # temp dirs
    tmp_dir = Path(tempfile.mkdtemp(prefix="tracker_eval_viz_"))
    npz_dir = tmp_dir / "npz"
    png_dir = tmp_dir / "png"
    npz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    try:
        # write per-frame npz (sequential, cheap)
        for fr in frames:
            save_frame_npz(npz_dir, fr, gt_by_frame.get(fr, []), pr_by_frame.get(fr, []))

        # parallel render
        workers = _auto_workers(args.workers)
        chunksize = int(args.chunksize)

        import multiprocessing as mp
        ctx = mp.get_context("fork")  # fastest on Linux; matches your earlier pattern

        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                str(npz_dir),
                str(png_dir),
                str(args.mode),
                xlim, ylim,
                int(args.dpi),
                str(args.title),
                gt_colors,
                pr_colors,
            ),
            maxtasksperchild=200,
        ) as pool:
            for _ in pool.imap_unordered(_render_one, frames, chunksize=chunksize):
                pass

        # stitch mp4
        stitch_mp4_from_pngs(png_dir, out_mp4, fps=int(args.fps))
        print(f"[tracker_eval] Wrote video: {out_mp4}")

    finally:
        if args.keep_tmp:
            print(f"[tracker_eval] Keeping tmp: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
