#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class Box3DJRDB:
    """
    JRDB base (internal convention):
      - center: (cx, cy, cz)  [x forward, y left, z up]
      - size: (l, w, h)
      - yaw: rot_z (rad) around +z
    """
    cx: float
    cy: float
    cz: float
    l: float
    w: float
    h: float
    yaw: float


@dataclass(frozen=True)
class TrackItem3D:
    track_id: int
    box: Box3DJRDB


# ----------------------------
# CLI
# ----------------------------

def _parse_tuple2_floats(vals: List[str], name: str) -> Tuple[float, float]:
    if len(vals) != 2:
        raise ValueError(f"{name} expects exactly 2 numbers, got: {vals}")
    return float(vals[0]), float(vals[1])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "Visualize TrackEval/JRDB3DBox-exported .txt (predicted tracks vs GT) in XY/XZ/YZ projections and export MP4s."
    )

    ap.add_argument("--out_root", required=True, type=str,
                    help="tracker_eval output root (e.g. /mnt/nvme/tracker_eval_outputs or rsynced server folder)")
    ap.add_argument("--tracker", required=True, type=str,
                    help="Tracker folder name inside out_root (e.g. ab3dmot, fastpoly, gnnpmb, cbmot, elptnet)")
    ap.add_argument("--sequence", required=True, type=str,
                    help="Sequence name (e.g. bytes-cafe-2019-02-07_0)")
    ap.add_argument("--split_name", required=True, type=str,
                    help="Split name used in your tracker_eval layout (e.g. train_val or test)")

    # Both GT and pred read from txt
    ap.add_argument("--gt_tracker_name", type=str, default="GT",
                    help="Folder name for GT export inside out_root (default: GT).")
    ap.add_argument("--gt_subfolder", type=str, default="data",
                    help="GT subfolder under out_root/<gt_tracker_name>/<split_name>/ (default: data).")
    ap.add_argument("--pred_subfolder", type=str, default="data",
                    help="Pred subfolder under out_root/<tracker>/<split_name>/ (default: data).")

    ap.add_argument("--out_dir", required=True, type=str,
                    help="Output directory where the 3 mp4s (xy/xz/yz) will be written.")

    ap.add_argument("--mode", choices=["centers", "boxes"], default="boxes",
                    help="centers: plot centers only; boxes: plot rectangles + center dot "
                         "(yaw meaningful mainly for XY)")

    # Per-view axis limits (optional; if omitted we auto-fit from GT+Pred with margin)
    ap.add_argument("--xlim_xy", nargs=2, default=None, help="XY view x limits: xmin xmax")
    ap.add_argument("--ylim_xy", nargs=2, default=None, help="XY view y limits: ymin ymax")

    ap.add_argument("--xlim_xz", nargs=2, default=None, help="XZ view x limits: xmin xmax")
    ap.add_argument("--ylim_xz", nargs=2, default=None, help="XZ view z limits: zmin zmax")

    ap.add_argument("--xlim_yz", nargs=2, default=None, help="YZ view y limits: ymin ymax")
    ap.add_argument("--ylim_yz", nargs=2, default=None, help="YZ view z limits: zmin zmax")

    ap.add_argument("--auto_margin", type=float, default=2.0,
                    help="Margin (meters) added around auto-fit limits if per-view limits not provided.")

    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=140)

    ap.add_argument("--workers", type=int, default=-1,
                    help="PNG rendering workers. Default = cpu_count()-1.")
    ap.add_argument("--chunksize", type=int, default=8,
                    help="multiprocessing chunksize for imap.")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep temp folder for debugging.")
    ap.add_argument("--title", type=str, default="TrackEval/JRDB3DBox visualization")

    ap.add_argument("--no_boxes_in_xz_yz", action="store_true",
                    help="If set, XZ and YZ views always show centers (boxes are less meaningful there).")

    return ap.parse_args()


# ----------------------------
# TXT parsing (TrackEval JRDB3DBox format - UPDATED)
# ----------------------------

def _wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    a = float(a)
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _parse_trackeval_txt_to_jrdb_base(txt_path: Path) -> Dict[int, List[TrackItem3D]]:
    """
    Read TrackEval/JRDB3DBox style .txt and convert 3D fields back to JRDB base (cx,cy,cz,l,w,h,rot_z).

    Expected columns:
      0 frame
      1 id
      2 class
      3 trunc
      4 occ
      5 alpha
      6-9 bbox2d
      10..16 = (x, y, z, w, h, d, yaw)   <--- IMPORTANT (new correct format)
      17 score (optional)

    Where (x,y,z,w,h,d,yaw) are in JRDB-toolkit/TrackEval coordinate convention and box_format='xyzwhd'.

    Conversion used by your exporter:
      x = -cy
      y = -cz + h/2
      z = cx
      yaw = wrap_to_2pi(-rot_z)
      d = l

    Inverse:
      cx = z
      cy = -x
      cz = -(y - h/2)
      rot_z = -yaw (wrapped)
      l = d
      w = w
      h = h
    """
    if not txt_path.exists():
        raise FileNotFoundError(txt_path)

    out: Dict[int, List[TrackItem3D]] = {}

    with txt_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 17:
                raise ValueError(f"{txt_path} line {line_no}: expected >=17 cols, got {len(parts)}")

            frame = int(float(parts[0]))
            tid = int(float(parts[1]))

            # (x, y, z, w, h, d, yaw)
            x = float(parts[10])
            y = float(parts[11])
            z = float(parts[12])
            w = float(parts[13])
            h = float(parts[14])
            d = float(parts[15])  # depth/length
            yaw = float(parts[16])

            cx = z
            cy = -x
            cz = -(y - 0.5 * h)
            rot_z = _wrap_to_pi(-yaw)

            l = d
            box = Box3DJRDB(cx=cx, cy=cy, cz=cz, l=l, w=w, h=h, yaw=rot_z)
            out.setdefault(frame, []).append(TrackItem3D(track_id=tid, box=box))

    return out


# ----------------------------
# Geometry helpers
# ----------------------------

def rect_corners_xy(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """(4,2) corners CCW for an oriented rectangle in XY."""
    dx = l * 0.5
    dy = w * 0.5
    local = np.array([[ dx,  dy],
                      [ dx, -dy],
                      [-dx, -dy],
                      [-dx,  dy]], dtype=np.float64)
    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    world = local @ R.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


# ----------------------------
# Views (XY, XZ, YZ)  (diag removed)
# ----------------------------

VIEW_NAMES = ("xy", "xz", "yz")

def project_point(box: Box3DJRDB, view: str) -> Tuple[float, float]:
    if view == "xy":
        return box.cx, box.cy
    if view == "xz":
        return box.cx, box.cz
    if view == "yz":
        return box.cy, box.cz
    raise ValueError(f"Unknown view: {view}")


def project_box_polygon(box: Box3DJRDB, view: str) -> Optional[np.ndarray]:
    """
    Returns a (4,2) polygon for the box in given view if meaningful.
    - xy: oriented rectangle using yaw.
    - xz/yz: axis-aligned rectangle using (l,h) or (w,h) ignoring yaw.
    """
    if view == "xy":
        return rect_corners_xy(box.cx, box.cy, box.l, box.w, box.yaw)

    if view == "xz":
        u0 = box.cx - 0.5 * box.l
        u1 = box.cx + 0.5 * box.l
        v0 = box.cz - 0.5 * box.h
        v1 = box.cz + 0.5 * box.h
        return np.array([[u1, v1], [u1, v0], [u0, v0], [u0, v1]], dtype=np.float64)

    if view == "yz":
        u0 = box.cy - 0.5 * box.w
        u1 = box.cy + 0.5 * box.w
        v0 = box.cz - 0.5 * box.h
        v1 = box.cz + 0.5 * box.h
        return np.array([[u1, v1], [u1, v0], [u0, v0], [u0, v1]], dtype=np.float64)

    raise ValueError(f"Unknown view: {view}")


def view_axes_labels(view: str) -> Tuple[str, str]:
    if view == "xy":
        return "x (m)", "y (m)"
    if view == "xz":
        return "x (m)", "z (m)"
    if view == "yz":
        return "y (m)", "z (m)"
    raise ValueError(view)


# ----------------------------
# Color mapping
# ----------------------------

def build_id_color_map(ids: List[int], cmap_name: str) -> Dict[int, Tuple[float, float, float, float]]:
    if not ids:
        return {}
    ids_sorted = sorted(set(ids))
    cmap = plt.get_cmap(cmap_name)
    lo, hi = 0.25, 0.95
    n = len(ids_sorted)
    colors = [cmap(lo + (hi - lo) * (i / max(1, n - 1))) for i in range(n)]
    return {tid: colors[i] for i, tid in enumerate(ids_sorted)}


# ----------------------------
# Precompute NPZ for rendering
# ----------------------------

def save_frame_npz(out_dir: Path, frame_idx: int,
                   gt_items: List[TrackItem3D], pr_items: List[TrackItem3D]) -> None:
    """
    Save arrays:
      gt:  (N, 8) [id, cx, cy, cz, l, w, h, yaw]
      pr:  (M, 8)
    """
    def _arr(items: List[TrackItem3D]) -> np.ndarray:
        if not items:
            return np.zeros((0, 8), dtype=np.float64)
        return np.array([[it.track_id, it.box.cx, it.box.cy, it.box.cz, it.box.l, it.box.w, it.box.h, it.box.yaw]
                         for it in items], dtype=np.float64)

    gt_arr = _arr(gt_items)
    pr_arr = _arr(pr_items)
    np.savez_compressed(str(out_dir / f"frame_{frame_idx:06d}.npz"),
                        frame_idx=frame_idx, gt=gt_arr, pred=pr_arr)


# ----------------------------
# Worker rendering (per-view)
# ----------------------------

_W_NPZ_DIR: Optional[Path] = None
_W_PNG_DIR: Optional[Path] = None
_W_MODE: str = "boxes"
_W_DPI: int = 140
_W_TITLE: str = ""
_W_GT_COLORS: Dict[int, Tuple[float, float, float, float]] = {}
_W_PR_COLORS: Dict[int, Tuple[float, float, float, float]] = {}
_W_VIEW: str = "xy"
_W_XLIM: Tuple[float, float] = (-10.0, 10.0)
_W_YLIM: Tuple[float, float] = (-10.0, 10.0)
_W_NO_BOXES_XZ_YZ: bool = False


def _worker_init(npz_dir: str, png_dir: str,
                 mode: str, dpi: int, title: str,
                 gt_colors: Dict[int, Tuple[float, float, float, float]],
                 pr_colors: Dict[int, Tuple[float, float, float, float]],
                 view: str,
                 xlim: Tuple[float, float], ylim: Tuple[float, float],
                 no_boxes_xz_yz: bool) -> None:
    global _W_NPZ_DIR, _W_PNG_DIR, _W_MODE, _W_DPI, _W_TITLE, _W_GT_COLORS, _W_PR_COLORS
    global _W_VIEW, _W_XLIM, _W_YLIM, _W_NO_BOXES_XZ_YZ
    _W_NPZ_DIR = Path(npz_dir)
    _W_PNG_DIR = Path(png_dir)
    _W_MODE = str(mode)
    _W_DPI = int(dpi)
    _W_TITLE = str(title)
    _W_GT_COLORS = dict(gt_colors)
    _W_PR_COLORS = dict(pr_colors)
    _W_VIEW = str(view)
    _W_XLIM = xlim
    _W_YLIM = ylim
    _W_NO_BOXES_XZ_YZ = bool(no_boxes_xz_yz)


def _render_one(frame_idx: int) -> int:
    assert _W_NPZ_DIR is not None and _W_PNG_DIR is not None

    d = np.load(str(_W_NPZ_DIR / f"frame_{frame_idx:06d}.npz"), allow_pickle=False)
    gt = d["gt"]    # (N,8)
    pr = d["pred"]  # (M,8)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title(f"{_W_TITLE} | {_W_VIEW.upper()} | frame {frame_idx}")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(_W_XLIM[0], _W_XLIM[1])
    ax.set_ylim(_W_YLIM[0], _W_YLIM[1])
    ax.grid(True, linewidth=0.3, alpha=0.35)

    xlabel, ylabel = view_axes_labels(_W_VIEW)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def _draw(items_arr: np.ndarray,
              colors: Dict[int, Tuple[float, float, float, float]],
              is_pred: bool) -> None:
        if items_arr.shape[0] == 0:
            return
        for row in items_arr:
            tid = int(row[0])
            box = Box3DJRDB(cx=float(row[1]), cy=float(row[2]), cz=float(row[3]),
                            l=float(row[4]), w=float(row[5]), h=float(row[6]), yaw=float(row[7]))
            u, v = project_point(box, _W_VIEW)

            show_boxes = (_W_MODE == "boxes") and not (_W_NO_BOXES_XZ_YZ and _W_VIEW in ("xz", "yz"))
            if not show_boxes:
                ax.scatter([u], [v], s=28, marker="o",
                           color=colors.get(tid, (0, 0, 0, 1)),
                           edgecolors=("black" if is_pred else "none"),
                           linewidths=(0.3 if is_pred else 0.0),
                           alpha=0.95)
            else:
                poly = project_box_polygon(box, _W_VIEW)
                if poly is not None:
                    patch = Polygon(poly, closed=True, fill=False, linewidth=1.2,
                                    edgecolor=colors.get(tid, (0, 0, 0, 1)), alpha=0.95)
                    ax.add_patch(patch)
                ax.scatter([u], [v], s=14, marker="o",
                           color=colors.get(tid, (0, 0, 0, 1)),
                           edgecolors=("black" if is_pred else "none"),
                           linewidths=(0.3 if is_pred else 0.0),
                           alpha=0.95)

    _draw(gt, _W_GT_COLORS, is_pred=False)
    _draw(pr, _W_PR_COLORS, is_pred=True)

    ax.text(0.01, 0.99, "GT (reds) vs Pred (blues) | both from exported .txt",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
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
# Utils: workers + auto limits
# ----------------------------

def _auto_workers(user_workers: int) -> int:
    if user_workers is not None and int(user_workers) > 0:
        return int(user_workers)
    ncpu = os.cpu_count() or 1
    return max(1, ncpu - 1)


def _gather_all_boxes(gt_by_frame: Dict[int, List[TrackItem3D]],
                      pr_by_frame: Dict[int, List[TrackItem3D]],
                      frames: List[int]) -> List[Box3DJRDB]:
    boxes: List[Box3DJRDB] = []
    for fr in frames:
        boxes.extend([it.box for it in gt_by_frame.get(fr, [])])
        boxes.extend([it.box for it in pr_by_frame.get(fr, [])])
    return boxes


def _auto_limits_for_view(boxes: List[Box3DJRDB], view: str, margin: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not boxes:
        return (-10.0, 10.0), (-10.0, 10.0)
    pts = np.array([project_point(b, view) for b in boxes], dtype=np.float64)
    umin = float(np.min(pts[:, 0])) - margin
    umax = float(np.max(pts[:, 0])) + margin
    vmin = float(np.min(pts[:, 1])) - margin
    vmax = float(np.max(pts[:, 1])) + margin
    if umax - umin < 1e-3:
        umin -= 1.0
        umax += 1.0
    if vmax - vmin < 1e-3:
        vmin -= 1.0
        vmax += 1.0
    return (umin, umax), (vmin, vmax)


def _maybe_parse_limits(arg_xy: Optional[List[str]]) -> Optional[Tuple[float, float]]:
    if arg_xy is None:
        return None
    return _parse_tuple2_floats(arg_xy, "limits")


# ----------------------------
# Render one view video
# ----------------------------

def render_view_video(
    *,
    view: str,
    frames: List[int],
    npz_dir: Path,
    out_mp4: Path,
    fps: int,
    mode: str,
    dpi: int,
    title: str,
    gt_colors: Dict[int, Tuple[float, float, float, float]],
    pr_colors: Dict[int, Tuple[float, float, float, float]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    workers: int,
    chunksize: int,
    keep_tmp: bool,
    no_boxes_xz_yz: bool,
) -> None:
    png_dir = out_mp4.parent / f"tmp_png_{view}_{out_mp4.stem}"
    png_dir.mkdir(parents=True, exist_ok=True)

    try:
        import multiprocessing as mp
        ctx = mp.get_context("fork")

        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                str(npz_dir),
                str(png_dir),
                str(mode),
                int(dpi),
                str(title),
                gt_colors,
                pr_colors,
                str(view),
                xlim, ylim,
                bool(no_boxes_xz_yz),
            ),
            maxtasksperchild=200,
        ) as pool:
            for _ in pool.imap_unordered(_render_one, frames, chunksize=int(chunksize)):
                pass

        stitch_mp4_from_pngs(png_dir, out_mp4, fps=int(fps))
        print(f"[tracker_eval_viz] Wrote video ({view}): {out_mp4}")

    finally:
        if keep_tmp:
            print(f"[tracker_eval_viz] Keeping PNG folder: {png_dir}")
        else:
            shutil.rmtree(png_dir, ignore_errors=True)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    args = parse_args()

    out_root = Path(args.out_root)
    split_name = str(args.split_name)
    tracker = str(args.tracker)
    seq = str(args.sequence)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_txt = out_root / str(args.gt_tracker_name) / split_name / str(args.gt_subfolder) / f"{seq}.txt"
    pr_txt = out_root / tracker / split_name / str(args.pred_subfolder) / f"{seq}.txt"

    gt_by_frame = _parse_trackeval_txt_to_jrdb_base(gt_txt)
    pr_by_frame = _parse_trackeval_txt_to_jrdb_base(pr_txt)

    frames = sorted(gt_by_frame.keys())
    if not frames:
        raise RuntimeError(f"No frames found in GT txt: {gt_txt}")

    # Stable coloring
    gt_ids_all: List[int] = []
    pr_ids_all: List[int] = []
    for fr in frames:
        gt_ids_all.extend([it.track_id for it in gt_by_frame.get(fr, [])])
        pr_ids_all.extend([it.track_id for it in pr_by_frame.get(fr, [])])

    gt_colors = build_id_color_map(gt_ids_all, cmap_name="Reds")
    pr_colors = build_id_color_map(pr_ids_all, cmap_name="Blues")

    # Cache NPZ once
    tmp_dir = Path(tempfile.mkdtemp(prefix="tracker_eval_viz3_"))
    npz_dir = tmp_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fr in frames:
            save_frame_npz(npz_dir, fr, gt_by_frame.get(fr, []), pr_by_frame.get(fr, []))

        boxes_all = _gather_all_boxes(gt_by_frame, pr_by_frame, frames)
        auto_margin = float(args.auto_margin)

        lims: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        for view in VIEW_NAMES:
            lims[view] = _auto_limits_for_view(boxes_all, view, margin=auto_margin)

        # Apply per-view overrides
        xlim_xy = _maybe_parse_limits(args.xlim_xy)
        ylim_xy = _maybe_parse_limits(args.ylim_xy)
        xlim_xz = _maybe_parse_limits(args.xlim_xz)
        ylim_xz = _maybe_parse_limits(args.ylim_xz)
        xlim_yz = _maybe_parse_limits(args.xlim_yz)
        ylim_yz = _maybe_parse_limits(args.ylim_yz)

        if xlim_xy is not None: lims["xy"] = (xlim_xy, lims["xy"][1])
        if ylim_xy is not None: lims["xy"] = (lims["xy"][0], ylim_xy)
        if xlim_xz is not None: lims["xz"] = (xlim_xz, lims["xz"][1])
        if ylim_xz is not None: lims["xz"] = (lims["xz"][0], ylim_xz)
        if xlim_yz is not None: lims["yz"] = (xlim_yz, lims["yz"][1])
        if ylim_yz is not None: lims["yz"] = (lims["yz"][0], ylim_yz)

        workers = _auto_workers(args.workers)
        chunksize = int(args.chunksize)

        base = f"{seq}__{tracker}__{split_name}"
        for view in VIEW_NAMES:
            out_mp4 = out_dir / f"{base}__{view}.mp4"
            xlim, ylim = lims[view]
            render_view_video(
                view=view,
                frames=frames,
                npz_dir=npz_dir,
                out_mp4=out_mp4,
                fps=int(args.fps),
                mode=str(args.mode),
                dpi=int(args.dpi),
                title=str(args.title),
                gt_colors=gt_colors,
                pr_colors=pr_colors,
                xlim=xlim,
                ylim=ylim,
                workers=workers,
                chunksize=chunksize,
                keep_tmp=bool(args.keep_tmp),
                no_boxes_xz_yz=bool(args.no_boxes_in_xz_yz),
            )

        print(f"[tracker_eval_viz] Done. Videos in: {out_dir}")

    finally:
        if args.keep_tmp:
            print(f"[tracker_eval_viz] Keeping tmp: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
