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

# Headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


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
        "Visualize TrackEval/JRDB3DBox-exported .txt (predicted tracks vs GT) in XY/XZ/YZ and export MP4s. (Fast version)"
    )

    ap.add_argument("--out_root", required=True, type=str,
                    help="tracker_eval output root (e.g. /mnt/nvme/tracker_eval_outputs or rsynced server folder)")
    ap.add_argument("--tracker", required=True, type=str,
                    help="Tracker folder name inside out_root (e.g. ab3dmot, fastpoly, gnnpmb, cbmot, elptnet)")
    ap.add_argument("--sequence", required=True, type=str,
                    help="Sequence name (e.g. bytes-cafe-2019-02-07_0)")
    ap.add_argument("--split_name", required=True, type=str,
                    help="Split name used in your tracker_eval layout (e.g. train or test)")

    ap.add_argument("--gt_tracker_name", type=str, default="GT",
                    help="Folder name for GT export inside out_root (default: GT).")
    ap.add_argument("--gt_subfolder", type=str, default="data",
                    help="GT subfolder under out_root/<gt_tracker_name>/<split_name>/ (default: data).")
    ap.add_argument("--pred_subfolder", type=str, default="data",
                    help="Pred subfolder under out_root/<tracker>/<split_name>/ (default: data).")

    ap.add_argument("--out_dir", required=True, type=str,
                    help="Output directory where the 3 mp4s (xy/xz/yz) will be written.")

    ap.add_argument("--mode", choices=["centers", "boxes"], default="boxes",
                    help="centers: plot centers only; boxes: plot rectangles + center dot")

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

    # NOTE: We still keep dpi for sizing. With PPM (default), dpi mainly affects canvas size.
    ap.add_argument("--dpi", type=int, default=140)

    ap.add_argument("--workers", type=int, default=-1,
                    help="Frame rendering workers. Default = cpu_count()-1.")
    ap.add_argument("--chunksize", type=int, default=32,
                    help="multiprocessing chunksize for imap (higher = less overhead).")
    ap.add_argument("--keep_tmp", action="store_true", help="Keep temp folder for debugging.")
    ap.add_argument("--title", type=str, default="TrackEval/JRDB3DBox visualization (fast)")

    ap.add_argument("--no_boxes_in_xz_yz", action="store_true",
                    help="If set, XZ and YZ views always show centers (boxes less meaningful there).")

    # Speed knobs
    ap.add_argument("--img_format", choices=["ppm", "png"], default="ppm",
                    help="Frame image format before stitching. PPM is much faster but uses more disk. (default: ppm)")
    ap.add_argument("--ffmpeg_preset", type=str, default="veryfast",
                    help="ffmpeg x264 preset (e.g. ultrafast, veryfast, fast, medium). (default: veryfast)")
    ap.add_argument("--ffmpeg_crf", type=int, default=23,
                    help="ffmpeg x264 CRF (lower=better, larger files; higher=faster/smaller). (default: 23)")
    ap.add_argument("--ffmpeg_threads", type=int, default=0,
                    help="ffmpeg threads (0 = auto).")

    ap.add_argument("--mp_start", choices=["auto", "fork", "spawn", "forkserver"], default="auto",
                    help="Multiprocessing start method. auto prefers fork when available. (default: auto)")

    ap.add_argument("--stream_to_ffmpeg", action="store_true",
                    help="Stream raw RGB frames directly to ffmpeg (no temp images). "
                         "Fastest IO, but runs single-process per view.")

    return ap.parse_args()


# ----------------------------
# TXT parsing (TrackEval JRDB3DBox format)
# ----------------------------

def _wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    a = float(a)
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _parse_trackeval_txt_to_jrdb_base(txt_path: Path) -> Dict[int, List[TrackItem3D]]:
    """
    Expected columns:
      0 frame
      1 id
      ...
      10..16 = (x, y, z, w, h, d, yaw)
      17 score (optional)

    Inverse mapping:
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

            x = float(parts[10])
            y = float(parts[11])
            z = float(parts[12])
            w = float(parts[13])
            h = float(parts[14])
            d = float(parts[15])
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
# Views + geometry
# ----------------------------

VIEW_NAMES = ("xy", "xz", "yz")


def view_axes_labels(view: str) -> Tuple[str, str]:
    if view == "xy":
        return "x (m)", "y (m)"
    if view == "xz":
        return "x (m)", "z (m)"
    if view == "yz":
        return "y (m)", "z (m)"
    raise ValueError(view)


def rect_polyline_xy(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """(5,2) closed polyline for oriented rectangle in XY."""
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
    # close
    world = np.vstack([world, world[0]])
    return world


def rect_polyline_axis_aligned(u: float, v: float, du: float, dv: float) -> np.ndarray:
    """(5,2) closed polyline axis-aligned around center (u,v) with extents du, dv."""
    u0 = u - 0.5 * du
    u1 = u + 0.5 * du
    v0 = v - 0.5 * dv
    v1 = v + 0.5 * dv
    poly = np.array([[u1, v1], [u1, v0], [u0, v0], [u0, v1], [u1, v1]], dtype=np.float64)
    return poly


def project_centers(arr: np.ndarray, view: str) -> np.ndarray:
    """
    arr: (N,8) [id, cx, cy, cz, l, w, h, yaw]
    returns (N,2) centers in selected view
    """
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if view == "xy":
        return np.stack([arr[:, 1], arr[:, 2]], axis=1)
    if view == "xz":
        return np.stack([arr[:, 1], arr[:, 3]], axis=1)
    if view == "yz":
        return np.stack([arr[:, 2], arr[:, 3]], axis=1)
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


def colors_for_tids(tids: np.ndarray,
                    color_map: Dict[int, Tuple[float, float, float, float]]) -> np.ndarray:
    # Fast enough in practice; if you have 1000s of boxes/frame and it’s still slow,
    # we can replace this with a LUT-based approach.
    return np.array([color_map.get(int(t), (0, 0, 0, 1)) for t in tids], dtype=np.float64)


# ----------------------------
# Cache (fast): .npy per frame + mmap reads
# ----------------------------

def cache_frame_npy(cache_dir: Path, out_idx: int,
                    gt_items: List[TrackItem3D], pr_items: List[TrackItem3D]) -> None:
    """
    Save:
      gt_{out_idx:06d}.npy : (N,8) [id, cx, cy, cz, l, w, h, yaw]
      pr_{out_idx:06d}.npy : (M,8)
    Uncompressed + mmap-friendly.
    """
    def _arr(items: List[TrackItem3D]) -> np.ndarray:
        if not items:
            return np.zeros((0, 8), dtype=np.float64)
        return np.array([[it.track_id, it.box.cx, it.box.cy, it.box.cz,
                          it.box.l, it.box.w, it.box.h, it.box.yaw]
                         for it in items], dtype=np.float64)

    np.save(str(cache_dir / f"gt_{out_idx:06d}.npy"), _arr(gt_items))
    np.save(str(cache_dir / f"pr_{out_idx:06d}.npy"), _arr(pr_items))


# ----------------------------
# Auto limits
# ----------------------------

def _gather_all_boxes(gt_by_frame: Dict[int, List[TrackItem3D]],
                      pr_by_frame: Dict[int, List[TrackItem3D]],
                      frames: List[int]) -> List[Box3DJRDB]:
    boxes: List[Box3DJRDB] = []
    for fr in frames:
        boxes.extend([it.box for it in gt_by_frame.get(fr, [])])
        boxes.extend([it.box for it in pr_by_frame.get(fr, [])])
    return boxes


def _auto_limits_for_view(boxes: List[Box3DJRDB], view: str, margin: float
                          ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not boxes:
        return (-10.0, 10.0), (-10.0, 10.0)

    if view == "xy":
        pts = np.array([[b.cx, b.cy] for b in boxes], dtype=np.float64)
    elif view == "xz":
        pts = np.array([[b.cx, b.cz] for b in boxes], dtype=np.float64)
    elif view == "yz":
        pts = np.array([[b.cy, b.cz] for b in boxes], dtype=np.float64)
    else:
        raise ValueError(view)

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


def _auto_workers(user_workers: int) -> int:
    if user_workers is not None and int(user_workers) > 0:
        return int(user_workers)
    ncpu = os.cpu_count() or 1
    return max(1, ncpu - 1)


# ----------------------------
# Fast frame writing (PPM) and ffmpeg
# ----------------------------

def write_ppm_from_canvas(fig: plt.Figure, out_path: Path) -> None:
    """
    Write binary PPM (P6) from the Agg canvas.
    Compatible with newer Matplotlib where tostring_rgb() may not exist.
    """
    fig.canvas.draw()

    # Preferred: print_to_buffer() -> returns RGBA bytes + (w,h)
    try:
        buf, (w, h) = fig.canvas.print_to_buffer()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    except Exception:
        # Fallback: buffer_rgba()
        rgba = np.asarray(fig.canvas.buffer_rgba())  # (h,w,4)
        h, w = rgba.shape[:2]

    rgb = rgba[..., :3]  # drop alpha

    with out_path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())



def stitch_mp4_from_images(
    img_dir: Path, img_ext: str, out_mp4: Path,
    fps: int, preset: str, crf: int, threads: int
) -> None:
    pattern = str(img_dir / f"frame_%06d.{img_ext}")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(int(fps)),
        "-i", pattern,
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if int(threads) > 0:
        cmd += ["-threads", str(int(threads))]
    cmd.append(str(out_mp4))

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stdout}")


# ----------------------------
# Worker rendering (per-view) with figure reuse + batching
# ----------------------------

_W_CACHE_DIR: Optional[Path] = None
_W_IMG_DIR: Optional[Path] = None
_W_MODE: str = "boxes"
_W_DPI: int = 140
_W_TITLE: str = ""
_W_GT_COLORS: Dict[int, Tuple[float, float, float, float]] = {}
_W_PR_COLORS: Dict[int, Tuple[float, float, float, float]] = {}
_W_VIEW: str = "xy"
_W_XLIM: Tuple[float, float] = (-10.0, 10.0)
_W_YLIM: Tuple[float, float] = (-10.0, 10.0)
_W_NO_BOXES_XZ_YZ: bool = False
_W_FRAME_IDS: Optional[np.ndarray] = None
_W_IMG_FORMAT: str = "ppm"

# Persistent artists per worker
_W_FIG: Optional[plt.Figure] = None
_W_AX: Optional[plt.Axes] = None
_W_TITLE_OBJ = None
_W_GT_SC = None
_W_PR_SC = None
_W_GT_LC: Optional[LineCollection] = None
_W_PR_LC: Optional[LineCollection] = None


def _worker_init(
    cache_dir: str, img_dir: str,
    mode: str, dpi: int, title: str,
    gt_colors: Dict[int, Tuple[float, float, float, float]],
    pr_colors: Dict[int, Tuple[float, float, float, float]],
    view: str,
    xlim: Tuple[float, float], ylim: Tuple[float, float],
    no_boxes_xz_yz: bool,
    frame_ids_path: str,
    img_format: str,
) -> None:
    global _W_CACHE_DIR, _W_IMG_DIR, _W_MODE, _W_DPI, _W_TITLE, _W_GT_COLORS, _W_PR_COLORS
    global _W_VIEW, _W_XLIM, _W_YLIM, _W_NO_BOXES_XZ_YZ, _W_FRAME_IDS, _W_IMG_FORMAT
    global _W_FIG, _W_AX, _W_TITLE_OBJ, _W_GT_SC, _W_PR_SC, _W_GT_LC, _W_PR_LC

    _W_CACHE_DIR = Path(cache_dir)
    _W_IMG_DIR = Path(img_dir)
    _W_MODE = str(mode)
    _W_DPI = int(dpi)
    _W_TITLE = str(title)
    _W_GT_COLORS = dict(gt_colors)
    _W_PR_COLORS = dict(pr_colors)
    _W_VIEW = str(view)
    _W_XLIM = xlim
    _W_YLIM = ylim
    _W_NO_BOXES_XZ_YZ = bool(no_boxes_xz_yz)
    _W_FRAME_IDS = np.load(frame_ids_path)  # (K,)
    _W_IMG_FORMAT = str(img_format)

    # Create and keep a single figure per worker (major speedup vs per-frame subplots)
    _W_FIG, _W_AX = plt.subplots(1, 1, figsize=(8, 8), dpi=_W_DPI)
    ax = _W_AX
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(_W_XLIM[0], _W_XLIM[1])
    ax.set_ylim(_W_YLIM[0], _W_YLIM[1])
    ax.grid(True, linewidth=0.3, alpha=0.35)

    xlabel, ylabel = view_axes_labels(_W_VIEW)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    _W_TITLE_OBJ = ax.set_title("")

    # Static legend box
    ax.text(
        0.01, 0.99,
        "GT (reds) vs Pred (blues) | both from exported .txt",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
    )

    # Persistent collections (updated per frame)
    _W_GT_SC = ax.scatter([], [], s=14, marker="o", alpha=0.95, edgecolors="none")
    _W_PR_SC = ax.scatter([], [], s=14, marker="o", alpha=0.95,
                          edgecolors="black", linewidths=0.3)

    _W_GT_LC = LineCollection([], linewidths=1.2, alpha=0.95)
    _W_PR_LC = LineCollection([], linewidths=1.2, alpha=0.95)
    ax.add_collection(_W_GT_LC)
    ax.add_collection(_W_PR_LC)


def _build_segments_and_colors(arr: np.ndarray, view: str, color_map: Dict[int, Tuple[float, float, float, float]],
                               show_boxes: bool) -> Tuple[List[np.ndarray], List[Tuple[float, float, float, float]]]:
    if (not show_boxes) or arr.shape[0] == 0:
        return [], []

    tids = arr[:, 0].astype(np.int64)
    cols = [color_map.get(int(t), (0, 0, 0, 1)) for t in tids]

    segs: List[np.ndarray] = []

    if view == "xy":
        # oriented rectangles (yaw), needs per-row
        for row in arr:
            cx, cy, l, w, yaw = float(row[1]), float(row[2]), float(row[4]), float(row[5]), float(row[7])
            segs.append(rect_polyline_xy(cx, cy, l, w, yaw))
        return segs, cols

    if view == "xz":
        # axis aligned: (x,z) with extents (l,h)
        for row in arr:
            cx, cz, l, h = float(row[1]), float(row[3]), float(row[4]), float(row[6])
            segs.append(rect_polyline_axis_aligned(cx, cz, l, h))
        return segs, cols

    if view == "yz":
        # axis aligned: (y,z) with extents (w,h)
        for row in arr:
            cy, cz, w, h = float(row[2]), float(row[3]), float(row[5]), float(row[6])
            segs.append(rect_polyline_axis_aligned(cy, cz, w, h))
        return segs, cols

    raise ValueError(view)


def _render_one(out_idx: int) -> int:
    """
    out_idx is sequential 0..K-1 (so ffmpeg patterns are continuous).
    Frame ID shown in title is the original frame id from the txt.
    """
    assert _W_CACHE_DIR is not None and _W_IMG_DIR is not None
    assert _W_AX is not None and _W_FIG is not None
    assert _W_FRAME_IDS is not None
    assert _W_GT_SC is not None and _W_PR_SC is not None
    assert _W_GT_LC is not None and _W_PR_LC is not None
    assert _W_TITLE_OBJ is not None

    orig_frame = int(_W_FRAME_IDS[out_idx])

    gt = np.load(str(_W_CACHE_DIR / f"gt_{out_idx:06d}.npy"), mmap_mode="r")  # (N,8)
    pr = np.load(str(_W_CACHE_DIR / f"pr_{out_idx:06d}.npy"), mmap_mode="r")  # (M,8)

    ax = _W_AX

    # Update title (cheap: update text, not recreate)
    _W_TITLE_OBJ.set_text(f"{_W_TITLE} | {_W_VIEW.upper()} | frame {orig_frame}")

    # Decide what we draw
    show_boxes = (_W_MODE == "boxes") and not (_W_NO_BOXES_XZ_YZ and _W_VIEW in ("xz", "yz"))

    # Centers (batched)
    gt_centers = project_centers(gt, _W_VIEW)
    pr_centers = project_centers(pr, _W_VIEW)

    gt_cols = colors_for_tids(gt[:, 0] if gt.shape[0] else np.zeros((0,), dtype=np.int64), _W_GT_COLORS)
    pr_cols = colors_for_tids(pr[:, 0] if pr.shape[0] else np.zeros((0,), dtype=np.int64), _W_PR_COLORS)

    # Adjust marker size depending on mode
    s = 28 if (_W_MODE == "centers" or (not show_boxes)) else 14
    _W_GT_SC.set_sizes(np.full((gt_centers.shape[0],), s, dtype=np.float64))
    _W_PR_SC.set_sizes(np.full((pr_centers.shape[0],), s, dtype=np.float64))

    _W_GT_SC.set_offsets(gt_centers)
    _W_PR_SC.set_offsets(pr_centers)

    # Facecolors update per point
    _W_GT_SC.set_facecolors(gt_cols)
    _W_PR_SC.set_facecolors(pr_cols)

    # Boxes (batched with LineCollection)
    gt_segs, gt_seg_cols = _build_segments_and_colors(gt, _W_VIEW, _W_GT_COLORS, show_boxes)
    pr_segs, pr_seg_cols = _build_segments_and_colors(pr, _W_VIEW, _W_PR_COLORS, show_boxes)

    _W_GT_LC.set_segments(gt_segs)
    _W_PR_LC.set_segments(pr_segs)
    _W_GT_LC.set_color(gt_seg_cols if gt_seg_cols else [])
    _W_PR_LC.set_color(pr_seg_cols if pr_seg_cols else [])

    # Save frame image
    out_img = _W_IMG_DIR / f"frame_{out_idx:06d}.{_W_IMG_FORMAT}"
    if _W_IMG_FORMAT == "ppm":
        write_ppm_from_canvas(_W_FIG, out_img)
    else:
        # PNG fallback (slower)
        _W_FIG.savefig(str(out_img), dpi=_W_DPI)

    return out_idx


# ----------------------------
# View rendering
# ----------------------------

def _pick_start_method(user_choice: str) -> str:
    import multiprocessing as mp
    methods = mp.get_all_start_methods()
    if user_choice != "auto":
        if user_choice not in methods:
            raise RuntimeError(f"Requested mp_start={user_choice} not available. Available: {methods}")
        return user_choice
    # auto: prefer fork if available (often fastest), else spawn
    if "fork" in methods:
        return "fork"
    return "spawn"


def render_view_video_multiproc(
    *,
    view: str,
    num_frames: int,
    cache_dir: Path,
    frame_ids_path: Path,
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
    img_format: str,
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    ffmpeg_threads: int,
    mp_start: str,
) -> None:
    img_dir = out_mp4.parent / f"tmp_frames_{view}_{out_mp4.stem}"
    img_dir.mkdir(parents=True, exist_ok=True)

    try:
        import multiprocessing as mp
        ctx = mp.get_context(mp_start)

        frame_indices = list(range(num_frames))

        with ctx.Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(
                str(cache_dir),
                str(img_dir),
                str(mode),
                int(dpi),
                str(title),
                gt_colors,
                pr_colors,
                str(view),
                xlim, ylim,
                bool(no_boxes_xz_yz),
                str(frame_ids_path),
                str(img_format),
            ),
            # keep workers alive; restarting workers costs time
            maxtasksperchild=None,
        ) as pool:
            for _ in pool.imap_unordered(_render_one, frame_indices, chunksize=int(chunksize)):
                pass

        stitch_mp4_from_images(
            img_dir=img_dir,
            img_ext=str(img_format),
            out_mp4=out_mp4,
            fps=int(fps),
            preset=str(ffmpeg_preset),
            crf=int(ffmpeg_crf),
            threads=int(ffmpeg_threads),
        )
        print(f"[tracker_eval_viz_fast] Wrote video ({view}): {out_mp4}")

    finally:
        if keep_tmp:
            print(f"[tracker_eval_viz_fast] Keeping frames folder: {img_dir}")
        else:
            shutil.rmtree(img_dir, ignore_errors=True)


def render_view_video_streaming(
    *,
    view: str,
    num_frames: int,
    cache_dir: Path,
    frame_ids: np.ndarray,
    out_mp4: Path,
    fps: int,
    mode: str,
    dpi: int,
    title: str,
    gt_colors: Dict[int, Tuple[float, float, float, float]],
    pr_colors: Dict[int, Tuple[float, float, float, float]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    no_boxes_xz_yz: bool,
    ffmpeg_preset: str,
    ffmpeg_crf: int,
    ffmpeg_threads: int,
) -> None:
    """
    Fastest IO path: render -> RGB bytes -> stdin to ffmpeg.
    Single-process per view (no temp images).
    """
    # Setup figure once
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=int(dpi))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.grid(True, linewidth=0.3, alpha=0.35)

    xlabel, ylabel = view_axes_labels(view)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title_obj = ax.set_title("")
    ax.text(
        0.01, 0.99,
        "GT (reds) vs Pred (blues) | both from exported .txt",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
    )

    gt_sc = ax.scatter([], [], s=14, marker="o", alpha=0.95, edgecolors="none")
    pr_sc = ax.scatter([], [], s=14, marker="o", alpha=0.95, edgecolors="black", linewidths=0.3)

    gt_lc = LineCollection([], linewidths=1.2, alpha=0.95)
    pr_lc = LineCollection([], linewidths=1.2, alpha=0.95)
    ax.add_collection(gt_lc)
    ax.add_collection(pr_lc)

    # Prepare ffmpeg stdin pipeline
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(int(fps)),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", str(ffmpeg_preset),
        "-crf", str(int(ffmpeg_crf)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if int(ffmpeg_threads) > 0:
        cmd += ["-threads", str(int(ffmpeg_threads))]
    cmd.append(str(out_mp4))

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None

    try:
        for out_idx in range(num_frames):
            orig_frame = int(frame_ids[out_idx])
            title_obj.set_text(f"{title} | {view.upper()} | frame {orig_frame}")

            gt = np.load(str(cache_dir / f"gt_{out_idx:06d}.npy"), mmap_mode="r")
            pr = np.load(str(cache_dir / f"pr_{out_idx:06d}.npy"), mmap_mode="r")

            show_boxes = (mode == "boxes") and not (no_boxes_xz_yz and view in ("xz", "yz"))

            gt_centers = project_centers(gt, view)
            pr_centers = project_centers(pr, view)

            gt_cols = colors_for_tids(gt[:, 0] if gt.shape[0] else np.zeros((0,), dtype=np.int64), gt_colors)
            pr_cols = colors_for_tids(pr[:, 0] if pr.shape[0] else np.zeros((0,), dtype=np.int64), pr_colors)

            s = 28 if (mode == "centers" or (not show_boxes)) else 14
            gt_sc.set_sizes(np.full((gt_centers.shape[0],), s, dtype=np.float64))
            pr_sc.set_sizes(np.full((pr_centers.shape[0],), s, dtype=np.float64))

            gt_sc.set_offsets(gt_centers)
            pr_sc.set_offsets(pr_centers)
            gt_sc.set_facecolors(gt_cols)
            pr_sc.set_facecolors(pr_cols)

            gt_segs, gt_seg_cols = _build_segments_and_colors(gt, view, gt_colors, show_boxes)
            pr_segs, pr_seg_cols = _build_segments_and_colors(pr, view, pr_colors, show_boxes)
            gt_lc.set_segments(gt_segs)
            pr_lc.set_segments(pr_segs)
            gt_lc.set_color(gt_seg_cols if gt_seg_cols else [])
            pr_lc.set_color(pr_seg_cols if pr_seg_cols else [])

            fig.canvas.draw()
            rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            p.stdin.write(rgb.tobytes())

        p.stdin.close()
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg streaming failed with return code {rc}")

        print(f"[tracker_eval_viz_fast] Wrote video ({view}) [streaming]: {out_mp4}")

    finally:
        try:
            if p.stdin and not p.stdin.closed:
                p.stdin.close()
        except Exception:
            pass
        plt.close(fig)


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

    # Build stable coloring
    gt_ids_all: List[int] = []
    pr_ids_all: List[int] = []
    for fr in frames:
        gt_ids_all.extend([it.track_id for it in gt_by_frame.get(fr, [])])
        pr_ids_all.extend([it.track_id for it in pr_by_frame.get(fr, [])])

    gt_colors = build_id_color_map(gt_ids_all, cmap_name="Reds")
    pr_colors = build_id_color_map(pr_ids_all, cmap_name="Blues")

    # Cache into a tmp dir (fast .npy, mmap-friendly)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tracker_eval_viz_fast_"))
    cache_dir = tmp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: make frames contiguous indices for ffmpeg patterns
    frame_ids = np.array(frames, dtype=np.int32)
    frame_ids_path = cache_dir / "frame_ids.npy"
    np.save(str(frame_ids_path), frame_ids)

    try:
        for out_idx, fr in enumerate(frames):
            cache_frame_npy(cache_dir, out_idx, gt_by_frame.get(fr, []), pr_by_frame.get(fr, []))

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
        mp_start = _pick_start_method(str(args.mp_start))

        for view in VIEW_NAMES:
            out_mp4 = out_dir / f"{base}__{view}.mp4"
            xlim, ylim = lims[view]

            if args.stream_to_ffmpeg:
                render_view_video_streaming(
                    view=view,
                    num_frames=len(frames),
                    cache_dir=cache_dir,
                    frame_ids=frame_ids,
                    out_mp4=out_mp4,
                    fps=int(args.fps),
                    mode=str(args.mode),
                    dpi=int(args.dpi),
                    title=str(args.title),
                    gt_colors=gt_colors,
                    pr_colors=pr_colors,
                    xlim=xlim,
                    ylim=ylim,
                    no_boxes_xz_yz=bool(args.no_boxes_in_xz_yz),
                    ffmpeg_preset=str(args.ffmpeg_preset),
                    ffmpeg_crf=int(args.ffmpeg_crf),
                    ffmpeg_threads=int(args.ffmpeg_threads),
                )
            else:
                render_view_video_multiproc(
                    view=view,
                    num_frames=len(frames),
                    cache_dir=cache_dir,
                    frame_ids_path=frame_ids_path,
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
                    img_format=str(args.img_format),
                    ffmpeg_preset=str(args.ffmpeg_preset),
                    ffmpeg_crf=int(args.ffmpeg_crf),
                    ffmpeg_threads=int(args.ffmpeg_threads),
                    mp_start=mp_start,
                )

        print(f"[tracker_eval_viz_fast] Done. Videos in: {out_dir}")

    finally:
        if args.keep_tmp:
            print(f"[tracker_eval_viz_fast] Keeping tmp: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
