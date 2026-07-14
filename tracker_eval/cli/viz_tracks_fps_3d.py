#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ============================================================
# Data structures
# ============================================================

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


@dataclass(frozen=True)
class FrameStat:
    fps_inst: float
    step_ms: float
    is_warmup: bool


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        "3D visualization video (pointcloud + GT/Pred tracks + current FPS tile) from TrackEval/JRDB exports."
    )

    # TrackEval/JRDB txt inputs
    ap.add_argument("--out_root", required=True, type=str,
                    help="tracker_eval output root (e.g. /mnt/nvme/tracker_eval_outputs)")
    ap.add_argument("--tracker", required=True, type=str,
                    help="Tracker folder name inside out_root (e.g. fastpoly__global)")
    ap.add_argument("--sequence", required=True, type=str,
                    help="Sequence name (e.g. cubberly-auditorium-2019-04-22_1)")
    ap.add_argument("--split_name", required=True, type=str,
                    help="Split name (e.g. train/test)")

    ap.add_argument("--gt_tracker_name", type=str, default="GT")
    ap.add_argument("--gt_subfolder", type=str, default="data")
    ap.add_argument("--pred_subfolder", type=str, default="data")

    # Output
    ap.add_argument("--out_path", required=True, type=str,
                    help="Output MP4 path (normal mode) or PNG path (preview mode)")
    ap.add_argument("--preview", action="store_true",
                    help="Preview mode: render one frame PNG instead of full video")
    ap.add_argument("--preview_frame", type=int, default=None,
                    help="Original frame id to preview (from txt col0). Fallback to out_idx if in range.")
    ap.add_argument("--preview_out_idx", type=int, default=None,
                    help="Sequential frame index (0..N-1) to preview. Overrides --preview_frame.")

    # Frame stats / current FPS tile
    ap.add_argument("--frame_stats_root", type=str, default=None,
                    help="Folder with <sequence>.csv frame stats. Default: <out_root>/<tracker>/<split_name>/frame_stats")
    ap.add_argument("--fps_green_hz", type=float, default=10.0)
    ap.add_argument("--fps_orange_hz", type=float, default=8.0)

    # FPS tile styling
    ap.add_argument("--fps_label_text", type=str, default="FPS")
    ap.add_argument("--fps_label_fontsize", type=float, default=18.0)
    ap.add_argument("--fps_value_fontsize", type=float, default=24.0)
    ap.add_argument("--fps_value_decimals", type=int, default=2)

    # Figure / video
    ap.add_argument("--fps", type=int, default=10, help="Output video FPS (normal mode)")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--fig_w", type=float, default=10.0)
    ap.add_argument("--fig_h", type=float, default=9.0)

    ap.add_argument("--ffmpeg_preset", type=str, default="veryfast")
    ap.add_argument("--ffmpeg_crf", type=int, default=23)
    ap.add_argument("--ffmpeg_threads", type=int, default=0)
    ap.add_argument("--img_format", choices=["ppm", "png"], default="ppm")
    ap.add_argument("--keep_tmp", action="store_true")

    # 3D camera / view
    ap.add_argument("--cam_elev", type=float, default=28.0)
    ap.add_argument("--cam_azim", type=float, default=-120.0)
    ap.add_argument("--cam_roll", type=float, default=None,
                    help="Optional camera roll in degrees (if supported by matplotlib)")
    ap.add_argument("--cam_dist", type=float, default=None,
                    help="Optional camera distance (closer/smaller -> closer view; mpl-version dependent)")
    ap.add_argument("--box_aspect", nargs=3, default=None,
                    help="Optional box aspect x y z, e.g. 1 1 1")
    ap.add_argument("--hide_axes", action="store_true",
                    help="Hide axes, ticks, grid, labels (visual-only mode)")
    ap.add_argument("--show_ground_outline", action="store_true",
                    help="Draw a rectangle at z=0 around the visible XY area")

    # 3D axis limits
    ap.add_argument("--xlim", nargs=2, default=None)
    ap.add_argument("--ylim", nargs=2, default=None)
    ap.add_argument("--zlim", nargs=2, default=None)
    ap.add_argument("--auto_margin_xy", type=float, default=2.0)
    ap.add_argument("--auto_margin_z", type=float, default=0.5)

    # Track style
    ap.add_argument("--mode", choices=["centers", "boxes"], default="boxes")
    ap.add_argument("--center_size_boxes", type=float, default=10.0)
    ap.add_argument("--center_size_centers", type=float, default=22.0)
    ap.add_argument("--center_alpha", type=float, default=0.9)
    ap.add_argument("--box_lw", type=float, default=0.8)
    ap.add_argument("--heading_lw", type=float, default=1.0)

    # Point cloud (NEW: optional but supported)
    ap.add_argument("--pc_dir", type=str, default=None,
                    help=("Directory containing per-frame point clouds. "
                          "Supported filenames per frame: <frame_id>.npy, <out_idx>.npy, zero-padded variants, "
                          "or .npz with key 'pc'/'points'."))
    ap.add_argument("--pc_optional", action="store_true",
                    help="If set, missing pointcloud frames are allowed (track-only frame renders)")
    ap.add_argument("--pc_z_min", type=float, default=0.10)
    ap.add_argument("--pc_z_max", type=float, default=3.00)
    ap.add_argument("--max_points", type=int, default=120000)
    ap.add_argument("--point_size", type=float, default=0.5)
    ap.add_argument("--point_alpha", type=float, default=0.5)
    ap.add_argument("--point_seed", type=int, default=0,
                    help="Seed for deterministic point downsampling")
    ap.add_argument("--point_color", type=str, default=None,
                    help="Optional single matplotlib color for pointcloud (default uses mpl default)")

    return ap.parse_args()


# ============================================================
# Helpers
# ============================================================

def _parse_tuple2_floats(vals: Optional[List[str]], name: str) -> Optional[Tuple[float, float]]:
    if vals is None:
        return None
    if len(vals) != 2:
        raise ValueError(f"{name} expects exactly 2 numbers, got: {vals}")
    return float(vals[0]), float(vals[1])


def _parse_tuple3_floats(vals: Optional[List[str]], name: str) -> Optional[Tuple[float, float, float]]:
    if vals is None:
        return None
    if len(vals) != 3:
        raise ValueError(f"{name} expects exactly 3 numbers, got: {vals}")
    return float(vals[0]), float(vals[1]), float(vals[2])


def _wrap_to_pi(a: float) -> float:
    a = float(a)
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")


# ============================================================
# TXT parsing (TrackEval JRDB3DBox format)
# ============================================================

def _parse_trackeval_txt_to_jrdb_base(txt_path: Path) -> Dict[int, List[TrackItem3D]]:
    """
    Expected columns:
      0 frame
      1 id
      ...
      10..16 = (x, y, z, w, h, d, yaw)

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

            out.setdefault(frame, []).append(
                TrackItem3D(
                    track_id=tid,
                    box=Box3DJRDB(cx=cx, cy=cy, cz=cz, l=d, w=w, h=h, yaw=rot_z)
                )
            )
    return out


# ============================================================
# Frame stats CSV parsing (fps tile)
# ============================================================

def load_frame_stats_csv(csv_path: Path) -> Dict[int, FrameStat]:
    """
    Maps frame_idx -> FrameStat.
    Expected columns include:
      frame_idx, step_ms, fps_inst, is_warmup
    """
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out: Dict[int, FrameStat] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame_idx", "step_ms", "fps_inst", "is_warmup"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

        for row in reader:
            idx = int(row["frame_idx"])
            out[idx] = FrameStat(
                fps_inst=float(row["fps_inst"]),
                step_ms=float(row["step_ms"]),
                is_warmup=_parse_bool(row["is_warmup"]),
            )
    return out


# ============================================================
# Geometry
# ============================================================

def box3d_wire_segments(box: Box3DJRDB) -> List[np.ndarray]:
    cx, cy, cz = box.cx, box.cy, box.cz
    l, w, h = box.l, box.w, box.h
    yaw = box.yaw

    dx = l * 0.5
    dy = w * 0.5
    dz = h * 0.5

    local = np.array([
        [ dx,  dy, -dz],  # 0
        [ dx, -dy, -dz],  # 1
        [-dx, -dy, -dz],  # 2
        [-dx,  dy, -dz],  # 3
        [ dx,  dy,  dz],  # 4
        [ dx, -dy,  dz],  # 5
        [-dx, -dy,  dz],  # 6
        [-dx,  dy,  dz],  # 7
    ], dtype=np.float64)

    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pts = local @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts[:, 2] += cz

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    return [pts[[i, j], :] for i, j in edges]


def heading_segment(box: Box3DJRDB, scale: float = 0.6) -> np.ndarray:
    front_len = max(0.1, 0.5 * box.l * scale)
    x2 = box.cx + front_len * math.cos(box.yaw)
    y2 = box.cy + front_len * math.sin(box.yaw)
    z2 = box.cz
    return np.array([[box.cx, box.cy, box.cz], [x2, y2, z2]], dtype=np.float64)


# ============================================================
# Color mapping
# ============================================================

def build_id_color_map(ids: List[int], cmap_name: str) -> Dict[int, Tuple[float, float, float, float]]:
    if not ids:
        return {}
    ids_sorted = sorted(set(ids))
    cmap = plt.get_cmap(cmap_name)
    lo, hi = 0.25, 0.95
    n = len(ids_sorted)
    colors = [cmap(lo + (hi - lo) * (i / max(1, n - 1))) for i in range(n)]
    return {tid: colors[i] for i, tid in enumerate(ids_sorted)}


# ============================================================
# Limits / scene fit
# ============================================================

def _gather_all_boxes(
    gt_by_frame: Dict[int, List[TrackItem3D]],
    pr_by_frame: Dict[int, List[TrackItem3D]],
    frames: List[int]
) -> List[Box3DJRDB]:
    boxes: List[Box3DJRDB] = []
    for fr in frames:
        boxes.extend([it.box for it in gt_by_frame.get(fr, [])])
        boxes.extend([it.box for it in pr_by_frame.get(fr, [])])
    return boxes


def _auto_3d_limits(
    boxes: List[Box3DJRDB], margin_xy: float, margin_z: float
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    if not boxes:
        return (-10, 10), (-10, 10), (0, 2.5)

    xs, ys, zs = [], [], []
    for b in boxes:
        rxy = 0.5 * math.sqrt(b.l * b.l + b.w * b.w)
        xs.extend([b.cx - rxy, b.cx + rxy])
        ys.extend([b.cy - rxy, b.cy + rxy])
        zs.extend([b.cz - 0.5 * b.h, b.cz + 0.5 * b.h])

    xlim = (float(min(xs)) - margin_xy, float(max(xs)) + margin_xy)
    ylim = (float(min(ys)) - margin_xy, float(max(ys)) + margin_xy)
    zlim = (float(min(zs)) - margin_z, float(max(zs)) + margin_z)

    def _fix(a: Tuple[float, float], pad: float) -> Tuple[float, float]:
        if a[1] - a[0] < 1e-6:
            c = 0.5 * (a[0] + a[1])
            return (c - pad, c + pad)
        return a

    return _fix(xlim, 1.0), _fix(ylim, 1.0), _fix(zlim, 0.5)


# ============================================================
# Point cloud loading
# ============================================================

def _try_load_pc_file(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None

    if path.suffix.lower() == ".npy":
        arr = np.load(str(path))
    elif path.suffix.lower() == ".npz":
        z = np.load(str(path))
        if "pc" in z:
            arr = z["pc"]
        elif "points" in z:
            arr = z["points"]
        else:
            # fallback: first array key
            keys = list(z.keys())
            if not keys:
                return None
            arr = z[keys[0]]
    else:
        return None

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Pointcloud array must be 2D, got {arr.shape} from {path}")
    if arr.shape[1] >= 3:
        return arr[:, :3].astype(np.float64, copy=False)
    if arr.shape[0] >= 3:
        # maybe shape (3,N)
        return arr[:3, :].T.astype(np.float64, copy=False)
    raise ValueError(f"Pointcloud array needs xyz columns, got {arr.shape} from {path}")


def load_pointcloud_for_frame(
    pc_dir: Path,
    orig_frame_id: int,
    out_idx: int,
) -> Optional[np.ndarray]:
    """
    Tries common naming conventions:
      - <orig_frame_id>.npy / .npz
      - <orig_frame_id:06d>.npy / .npz
      - <out_idx>.npy / .npz
      - <out_idx:06d>.npy / .npz
      - <out_idx:06d>.pcd.npy (if preconverted)
    """
    candidates = [
        pc_dir / f"{orig_frame_id}.npy",
        pc_dir / f"{orig_frame_id}.npz",
        pc_dir / f"{orig_frame_id:06d}.npy",
        pc_dir / f"{orig_frame_id:06d}.npz",

        pc_dir / f"{out_idx}.npy",
        pc_dir / f"{out_idx}.npz",
        pc_dir / f"{out_idx:06d}.npy",
        pc_dir / f"{out_idx:06d}.npz",

        pc_dir / f"{out_idx:06d}.pcd.npy",
        pc_dir / f"{out_idx:06d}.pcd.npz",
    ]
    for p in candidates:
        arr = _try_load_pc_file(p)
        if arr is not None:
            return arr
    return None


def preprocess_pointcloud(
    pc: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float64)
    if pc.shape[0] == 0:
        return pc

    mask = (pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)
    pc = pc[mask]
    if pc.shape[0] == 0:
        return pc

    if max_points is not None and max_points > 0 and pc.shape[0] > max_points:
        idx = rng.choice(pc.shape[0], size=max_points, replace=False)
        pc = pc[idx]
    return pc


# ============================================================
# FPS tile
# ============================================================

def fps_bg_color(
    fps_hz: float,
    green_hz: float,
    orange_hz: float,
) -> Tuple[float, float, float, float]:
    if fps_hz >= green_hz:
        return (0.72, 0.90, 0.72, 1.0)  # light green
    if fps_hz >= orange_hz:
        return (1.00, 0.78, 0.45, 1.0)  # orange
    return (0.95, 0.50, 0.50, 1.0)      # red


def draw_current_fps_tile(
    ax,
    *,
    fps_value: Optional[float],
    green_hz: float,
    orange_hz: float,
    label_text: str,
    label_fontsize: float,
    value_fontsize: float,
    decimals: int,
) -> None:
    ax.clear()

    # Background color by current fps
    if fps_value is None or not np.isfinite(fps_value):
        bg = (0.85, 0.85, 0.85, 1.0)
        value_str = "n/a"
    else:
        bg = fps_bg_color(float(fps_value), green_hz, orange_hz)
        value_str = f"{float(fps_value):.{max(0, int(decimals))}f} Hz"

    ax.set_facecolor(bg)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(False)

    # Top label
    ax.text(
        0.5, 0.78,
        label_text,
        ha="center", va="center",
        fontsize=label_fontsize,
        fontweight="bold",
        color="black",
        transform=ax.transAxes,
    )

    # Large current value
    ax.text(
        0.5, 0.33,
        value_str,
        ha="center", va="center",
        fontsize=value_fontsize,
        fontweight="bold",
        color="black",
        transform=ax.transAxes,
    )


# ============================================================
# 3D drawing
# ============================================================

def _boxes_to_arrays(items: List[TrackItem3D]) -> Tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    centers = np.array([[it.box.cx, it.box.cy, it.box.cz] for it in items], dtype=np.float64)
    tids = np.array([it.track_id for it in items], dtype=np.int64)
    return centers, tids


def _colors_for_tids(
    tids: np.ndarray,
    color_map: Dict[int, Tuple[float, float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    return [color_map.get(int(t), (0, 0, 0, 1)) for t in tids]


def draw_3d_scene(
    ax3d,
    *,
    pc_xyz: Optional[np.ndarray],
    gt_items: List[TrackItem3D],
    pr_items: List[TrackItem3D],
    gt_colors: Dict[int, Tuple[float, float, float, float]],
    pr_colors: Dict[int, Tuple[float, float, float, float]],
    mode: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    cam_elev: float,
    cam_azim: float,
    cam_roll: Optional[float],
    cam_dist: Optional[float],
    box_aspect: Optional[Tuple[float, float, float]],
    hide_axes: bool,
    show_ground_outline: bool,
    center_size_boxes: float,
    center_size_centers: float,
    center_alpha: float,
    box_lw: float,
    heading_lw: float,
    point_size: float,
    point_alpha: float,
    point_color: Optional[str],
) -> None:
    ax3d.clear()

    # Camera
    if cam_roll is None:
        ax3d.view_init(elev=cam_elev, azim=cam_azim)
    else:
        try:
            ax3d.view_init(elev=cam_elev, azim=cam_azim, roll=cam_roll)
        except TypeError:
            ax3d.view_init(elev=cam_elev, azim=cam_azim)

    if cam_dist is not None:
        try:
            ax3d.dist = float(cam_dist)  # may work depending on mpl version
        except Exception:
            pass

    # Limits first (needed for aspect)
    ax3d.set_xlim(xlim)
    ax3d.set_ylim(ylim)
    ax3d.set_zlim(zlim)

    # Aspect (match your prior style)
    if box_aspect is not None:
        try:
            ax3d.set_box_aspect(box_aspect)
        except Exception:
            pass
    else:
        try:
            x0, x1 = ax3d.get_xlim3d()
            y0, y1 = ax3d.get_ylim3d()
            z0, z1 = ax3d.get_zlim3d()
            ax3d.set_box_aspect([x1 - x0, y1 - y0, z1 - z0])
        except Exception:
            pass

    # Axes visibility (visual-only mode)
    if hide_axes:
        ax3d.set_axis_off()
        try:
            ax3d.grid(False)
        except Exception:
            pass
    else:
        ax3d.set_xlabel("x (m)")
        ax3d.set_ylabel("y (m)")
        ax3d.set_zlabel("z (m)")
        ax3d.grid(True, linewidth=0.4, alpha=0.35)
        try:
            ax3d.xaxis.pane.set_alpha(0.05)
            ax3d.yaxis.pane.set_alpha(0.05)
            ax3d.zaxis.pane.set_alpha(0.05)
        except Exception:
            pass

    # Optional ground outline (minimal reference)
    if show_ground_outline:
        x0, x1 = xlim
        y0, y1 = ylim
        ground = np.array([
            [[x0, y0, 0.0], [x1, y0, 0.0]],
            [[x1, y0, 0.0], [x1, y1, 0.0]],
            [[x1, y1, 0.0], [x0, y1, 0.0]],
            [[x0, y1, 0.0], [x0, y0, 0.0]],
        ], dtype=np.float64)
        ax3d.add_collection3d(
            Line3DCollection(ground, colors=[(0.2, 0.2, 0.2, 0.35)], linewidths=0.8)
        )

    # Point cloud
    if pc_xyz is not None and pc_xyz.shape[0] > 0:
        scatter_kwargs = dict(
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
            depthshade=False,
        )
        if point_color is not None:
            scatter_kwargs["c"] = point_color
        ax3d.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], **scatter_kwargs)

    # Track centers
    gt_centers, gt_tids = _boxes_to_arrays(gt_items)
    pr_centers, pr_tids = _boxes_to_arrays(pr_items)

    gt_cols = _colors_for_tids(gt_tids, gt_colors)
    pr_cols = _colors_for_tids(pr_tids, pr_colors)

    center_s = center_size_boxes if mode == "boxes" else center_size_centers

    if gt_centers.shape[0] > 0:
        ax3d.scatter(gt_centers[:, 0], gt_centers[:, 1], gt_centers[:, 2],
                     s=center_s, c=gt_cols, marker="o", depthshade=False,
                     edgecolors="none", alpha=center_alpha)
    if pr_centers.shape[0] > 0:
        ax3d.scatter(pr_centers[:, 0], pr_centers[:, 1], pr_centers[:, 2],
                     s=center_s, c=pr_cols, marker="o", depthshade=False,
                     edgecolors="black", linewidths=0.2, alpha=center_alpha)

    # 3D boxes / heading
    if mode == "boxes":
        gt_segments: List[np.ndarray] = []
        gt_seg_cols: List[Tuple[float, float, float, float]] = []
        pr_segments: List[np.ndarray] = []
        pr_seg_cols: List[Tuple[float, float, float, float]] = []

        gt_heading_segments: List[np.ndarray] = []
        gt_heading_cols: List[Tuple[float, float, float, float]] = []
        pr_heading_segments: List[np.ndarray] = []
        pr_heading_cols: List[Tuple[float, float, float, float]] = []

        for it in gt_items:
            c = gt_colors.get(int(it.track_id), (0, 0, 0, 1))
            for seg in box3d_wire_segments(it.box):
                gt_segments.append(seg)
                gt_seg_cols.append(c)
            gt_heading_segments.append(heading_segment(it.box))
            gt_heading_cols.append(c)

        for it in pr_items:
            c = pr_colors.get(int(it.track_id), (0, 0, 0, 1))
            for seg in box3d_wire_segments(it.box):
                pr_segments.append(seg)
                pr_seg_cols.append(c)
            pr_heading_segments.append(heading_segment(it.box))
            pr_heading_cols.append(c)

        if gt_segments:
            ax3d.add_collection3d(Line3DCollection(gt_segments, colors=gt_seg_cols, linewidths=box_lw, alpha=0.95))
        if pr_segments:
            ax3d.add_collection3d(Line3DCollection(pr_segments, colors=pr_seg_cols, linewidths=box_lw, alpha=0.95))
        if gt_heading_segments:
            ax3d.add_collection3d(Line3DCollection(gt_heading_segments, colors=gt_heading_cols, linewidths=heading_lw, alpha=0.95))
        if pr_heading_segments:
            ax3d.add_collection3d(Line3DCollection(pr_heading_segments, colors=pr_heading_cols, linewidths=heading_lw, alpha=0.95))


# ============================================================
# Figure helpers / IO
# ============================================================

def create_figure(fig_w: float, fig_h: float, dpi: int):
    # Taller top tile + main 3D plot. No titles/legend.
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=False)
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1.15, 8.85], hspace=0.03, figure=fig)
    ax_tile = fig.add_subplot(gs[0, 0])
    ax3d = fig.add_subplot(gs[1, 0], projection="3d")
    return fig, ax_tile, ax3d


def write_ppm_from_canvas(fig: plt.Figure, out_path: Path) -> None:
    fig.canvas.draw()
    try:
        buf, (w, h) = fig.canvas.print_to_buffer()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    except Exception:
        rgba = np.asarray(fig.canvas.buffer_rgba())
        h, w = rgba.shape[:2]
    rgb = rgba[..., :3]
    with out_path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def stitch_mp4_from_images(
    img_dir: Path, img_ext: str, out_mp4: Path,
    fps: int, preset: str, crf: int, threads: int
) -> None:
    pattern = str(img_dir / f"frame_%06d.{img_ext}")
    cmd = [
        "ffmpeg", "-y",
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


# ============================================================
# Frame selection helpers
# ============================================================

def resolve_preview_out_idx(
    preview_out_idx: Optional[int],
    preview_frame: Optional[int],
    frame_ids: np.ndarray
) -> int:
    n = len(frame_ids)
    if n == 0:
        raise RuntimeError("No frames available")

    if preview_out_idx is not None:
        idx = int(preview_out_idx)
        if idx < 0 or idx >= n:
            raise ValueError(f"--preview_out_idx out of range [0,{n-1}]: {idx}")
        return idx

    if preview_frame is not None:
        matches = np.where(frame_ids == int(preview_frame))[0]
        if matches.size > 0:
            return int(matches[0])
        pf = int(preview_frame)
        if 0 <= pf < n:
            return pf
        raise ValueError(
            f"--preview_frame={preview_frame} not found as original frame id and not a valid out_idx [0,{n-1}]"
        )

    return 0


# ============================================================
# Rendering core
# ============================================================

def render_one_frame(
    *,
    fig,
    ax_tile,
    ax3d,
    out_idx: int,
    frame_ids: np.ndarray,
    gt_by_frame: Dict[int, List[TrackItem3D]],
    pr_by_frame: Dict[int, List[TrackItem3D]],
    stats_by_frame_idx: Dict[int, FrameStat],
    pc_dir: Optional[Path],
    pc_optional: bool,
    gt_colors: Dict[int, Tuple[float, float, float, float]],
    pr_colors: Dict[int, Tuple[float, float, float, float]],
    # styles
    fps_green_hz: float,
    fps_orange_hz: float,
    fps_label_text: str,
    fps_label_fontsize: float,
    fps_value_fontsize: float,
    fps_value_decimals: int,
    mode: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    cam_elev: float,
    cam_azim: float,
    cam_roll: Optional[float],
    cam_dist: Optional[float],
    box_aspect: Optional[Tuple[float, float, float]],
    hide_axes: bool,
    show_ground_outline: bool,
    center_size_boxes: float,
    center_size_centers: float,
    center_alpha: float,
    box_lw: float,
    heading_lw: float,
    point_size: float,
    point_alpha: float,
    point_color: Optional[str],
    pc_z_min: float,
    pc_z_max: float,
    max_points: Optional[int],
    rng: np.random.Generator,
) -> None:
    orig_frame_id = int(frame_ids[out_idx])

    # Current fps
    st = stats_by_frame_idx.get(out_idx, None)
    fps_now = None if st is None else st.fps_inst

    draw_current_fps_tile(
        ax_tile,
        fps_value=fps_now,
        green_hz=fps_green_hz,
        orange_hz=fps_orange_hz,
        label_text=fps_label_text,
        label_fontsize=fps_label_fontsize,
        value_fontsize=fps_value_fontsize,
        decimals=fps_value_decimals,
    )

    # Pointcloud (optional, but main visualization)
    pc_xyz = None
    if pc_dir is not None:
        pc_raw = load_pointcloud_for_frame(pc_dir=pc_dir, orig_frame_id=orig_frame_id, out_idx=out_idx)
        if pc_raw is None and not pc_optional:
            raise FileNotFoundError(
                f"Pointcloud not found for frame orig={orig_frame_id}, out_idx={out_idx} in {pc_dir}"
            )
        if pc_raw is not None:
            pc_xyz = preprocess_pointcloud(
                pc_raw,
                z_min=pc_z_min,
                z_max=pc_z_max,
                max_points=max_points,
                rng=rng,
            )

    draw_3d_scene(
        ax3d=ax3d,
        pc_xyz=pc_xyz,
        gt_items=gt_by_frame.get(orig_frame_id, []),
        pr_items=pr_by_frame.get(orig_frame_id, []),
        gt_colors=gt_colors,
        pr_colors=pr_colors,
        mode=mode,
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        cam_elev=cam_elev,
        cam_azim=cam_azim,
        cam_roll=cam_roll,
        cam_dist=cam_dist,
        box_aspect=box_aspect,
        hide_axes=hide_axes,
        show_ground_outline=show_ground_outline,
        center_size_boxes=center_size_boxes,
        center_size_centers=center_size_centers,
        center_alpha=center_alpha,
        box_lw=box_lw,
        heading_lw=heading_lw,
        point_size=point_size,
        point_alpha=point_alpha,
        point_color=point_color,
    )


def render_preview_png(
    *,
    out_png: Path,
    out_idx: int,
    frame_ids: np.ndarray,
    gt_by_frame,
    pr_by_frame,
    stats_by_frame_idx,
    pc_dir: Optional[Path],
    pc_optional: bool,
    gt_colors,
    pr_colors,
    args: argparse.Namespace,
) -> None:
    fig, ax_tile, ax3d = create_figure(fig_w=float(args.fig_w), fig_h=float(args.fig_h), dpi=int(args.dpi))
    rng = np.random.default_rng(int(args.point_seed) + int(out_idx))
    try:
        render_one_frame(
            fig=fig, ax_tile=ax_tile, ax3d=ax3d,
            out_idx=out_idx,
            frame_ids=frame_ids,
            gt_by_frame=gt_by_frame,
            pr_by_frame=pr_by_frame,
            stats_by_frame_idx=stats_by_frame_idx,
            pc_dir=pc_dir, pc_optional=pc_optional,
            gt_colors=gt_colors, pr_colors=pr_colors,
            fps_green_hz=float(args.fps_green_hz),
            fps_orange_hz=float(args.fps_orange_hz),
            fps_label_text=str(args.fps_label_text),
            fps_label_fontsize=float(args.fps_label_fontsize),
            fps_value_fontsize=float(args.fps_value_fontsize),
            fps_value_decimals=int(args.fps_value_decimals),
            mode=str(args.mode),
            xlim=args._xlim_final, ylim=args._ylim_final, zlim=args._zlim_final,
            cam_elev=float(args.cam_elev),
            cam_azim=float(args.cam_azim),
            cam_roll=(None if args.cam_roll is None else float(args.cam_roll)),
            cam_dist=(None if args.cam_dist is None else float(args.cam_dist)),
            box_aspect=args._box_aspect_final,
            hide_axes=bool(args.hide_axes),
            show_ground_outline=bool(args.show_ground_outline),
            center_size_boxes=float(args.center_size_boxes),
            center_size_centers=float(args.center_size_centers),
            center_alpha=float(args.center_alpha),
            box_lw=float(args.box_lw),
            heading_lw=float(args.heading_lw),
            point_size=float(args.point_size),
            point_alpha=float(args.point_alpha),
            point_color=(None if args.point_color is None else str(args.point_color)),
            pc_z_min=float(args.pc_z_min),
            pc_z_max=float(args.pc_z_max),
            max_points=(None if int(args.max_points) <= 0 else int(args.max_points)),
            rng=rng,
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_png), dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)
        print(f"[viz_tracks_fps_3d] Wrote preview PNG: {out_png}")
    finally:
        plt.close(fig)


def render_video(
    *,
    out_mp4: Path,
    frame_ids: np.ndarray,
    gt_by_frame,
    pr_by_frame,
    stats_by_frame_idx,
    pc_dir: Optional[Path],
    pc_optional: bool,
    gt_colors,
    pr_colors,
    args: argparse.Namespace,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="viz_tracks_fps_3d_"))
    img_dir = tmp_dir / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_tile, ax3d = create_figure(fig_w=float(args.fig_w), fig_h=float(args.fig_h), dpi=int(args.dpi))

    try:
        n = len(frame_ids)
        base_seed = int(args.point_seed)

        for out_idx in range(n):
            rng = np.random.default_rng(base_seed + out_idx)

            render_one_frame(
                fig=fig, ax_tile=ax_tile, ax3d=ax3d,
                out_idx=out_idx,
                frame_ids=frame_ids,
                gt_by_frame=gt_by_frame,
                pr_by_frame=pr_by_frame,
                stats_by_frame_idx=stats_by_frame_idx,
                pc_dir=pc_dir, pc_optional=pc_optional,
                gt_colors=gt_colors, pr_colors=pr_colors,
                fps_green_hz=float(args.fps_green_hz),
                fps_orange_hz=float(args.fps_orange_hz),
                fps_label_text=str(args.fps_label_text),
                fps_label_fontsize=float(args.fps_label_fontsize),
                fps_value_fontsize=float(args.fps_value_fontsize),
                fps_value_decimals=int(args.fps_value_decimals),
                mode=str(args.mode),
                xlim=args._xlim_final, ylim=args._ylim_final, zlim=args._zlim_final,
                cam_elev=float(args.cam_elev),
                cam_azim=float(args.cam_azim),
                cam_roll=(None if args.cam_roll is None else float(args.cam_roll)),
                cam_dist=(None if args.cam_dist is None else float(args.cam_dist)),
                box_aspect=args._box_aspect_final,
                hide_axes=bool(args.hide_axes),
                show_ground_outline=bool(args.show_ground_outline),
                center_size_boxes=float(args.center_size_boxes),
                center_size_centers=float(args.center_size_centers),
                center_alpha=float(args.center_alpha),
                box_lw=float(args.box_lw),
                heading_lw=float(args.heading_lw),
                point_size=float(args.point_size),
                point_alpha=float(args.point_alpha),
                point_color=(None if args.point_color is None else str(args.point_color)),
                pc_z_min=float(args.pc_z_min),
                pc_z_max=float(args.pc_z_max),
                max_points=(None if int(args.max_points) <= 0 else int(args.max_points)),
                rng=rng,
            )

            out_img = img_dir / f"frame_{out_idx:06d}.{args.img_format}"
            if args.img_format == "ppm":
                write_ppm_from_canvas(fig, out_img)
            else:
                fig.savefig(str(out_img), dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.02)

            if (out_idx + 1) % 100 == 0 or (out_idx + 1) == n:
                print(f"[viz_tracks_fps_3d] Rendered {out_idx + 1}/{n} frames")

        stitch_mp4_from_images(
            img_dir=img_dir,
            img_ext=str(args.img_format),
            out_mp4=out_mp4,
            fps=int(args.fps),
            preset=str(args.ffmpeg_preset),
            crf=int(args.ffmpeg_crf),
            threads=int(args.ffmpeg_threads),
        )
        print(f"[viz_tracks_fps_3d] Wrote MP4: {out_mp4}")

    finally:
        plt.close(fig)
        if args.keep_tmp:
            print(f"[viz_tracks_fps_3d] Keeping tmp frames dir: {img_dir}")
            print(f"[viz_tracks_fps_3d] Tmp root: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()

    out_root = Path(args.out_root)
    tracker = str(args.tracker)
    split_name = str(args.split_name)
    seq = str(args.sequence)

    gt_txt = out_root / str(args.gt_tracker_name) / split_name / str(args.gt_subfolder) / f"{seq}.txt"
    pr_txt = out_root / tracker / split_name / str(args.pred_subfolder) / f"{seq}.txt"

    gt_by_frame = _parse_trackeval_txt_to_jrdb_base(gt_txt)
    pr_by_frame = _parse_trackeval_txt_to_jrdb_base(pr_txt)

    frames = sorted(gt_by_frame.keys())
    if not frames:
        raise RuntimeError(f"No frames found in GT txt: {gt_txt}")
    frame_ids = np.array(frames, dtype=np.int32)

    # Frame stats CSV
    if args.frame_stats_root is None:
        frame_stats_root = out_root / tracker / split_name / "frame_stats"
    else:
        frame_stats_root = Path(args.frame_stats_root)
    frame_stats_csv = frame_stats_root / f"{seq}.csv"
    stats_by_frame_idx = load_frame_stats_csv(frame_stats_csv)

    # Color maps
    gt_ids_all: List[int] = []
    pr_ids_all: List[int] = []
    for fr in frames:
        gt_ids_all.extend([it.track_id for it in gt_by_frame.get(fr, [])])
        pr_ids_all.extend([it.track_id for it in pr_by_frame.get(fr, [])])
    gt_colors = build_id_color_map(gt_ids_all, "Reds")
    pr_colors = build_id_color_map(pr_ids_all, "Blues")

    # Limits
    boxes_all = _gather_all_boxes(gt_by_frame, pr_by_frame, frames)
    xlim_auto, ylim_auto, zlim_auto = _auto_3d_limits(
        boxes_all, margin_xy=float(args.auto_margin_xy), margin_z=float(args.auto_margin_z)
    )
    args._xlim_final = _parse_tuple2_floats(args.xlim, "xlim") or xlim_auto
    args._ylim_final = _parse_tuple2_floats(args.ylim, "ylim") or ylim_auto
    args._zlim_final = _parse_tuple2_floats(args.zlim, "zlim") or zlim_auto
    args._box_aspect_final = _parse_tuple3_floats(args.box_aspect, "box_aspect")

    # Pointcloud dir
    pc_dir = None if args.pc_dir is None else Path(args.pc_dir)
    if pc_dir is not None and not pc_dir.exists():
        raise FileNotFoundError(f"--pc_dir does not exist: {pc_dir}")

    out_path = Path(args.out_path)

    print("[viz_tracks_fps_3d] Inputs:")
    print(f"  GT txt:        {gt_txt}")
    print(f"  Pred txt:      {pr_txt}")
    print(f"  Frame stats:   {frame_stats_csv}")
    print(f"  Pointcloud dir:{pc_dir}")
    print(f"  #frames:       {len(frame_ids)}")
    print(f"  xlim/ylim/zlim: {args._xlim_final} | {args._ylim_final} | {args._zlim_final}")
    print(f"  camera elev/azim/roll/dist: {args.cam_elev} / {args.cam_azim} / {args.cam_roll} / {args.cam_dist}")

    if args.preview:
        if out_path.suffix.lower() != ".png":
            print("[viz_tracks_fps_3d] Warning: preview mode usually uses a .png --out_path")
        out_idx = resolve_preview_out_idx(
            preview_out_idx=args.preview_out_idx,
            preview_frame=args.preview_frame,
            frame_ids=frame_ids
        )
        render_preview_png(
            out_png=out_path,
            out_idx=out_idx,
            frame_ids=frame_ids,
            gt_by_frame=gt_by_frame,
            pr_by_frame=pr_by_frame,
            stats_by_frame_idx=stats_by_frame_idx,
            pc_dir=pc_dir,
            pc_optional=bool(args.pc_optional),
            gt_colors=gt_colors,
            pr_colors=pr_colors,
            args=args,
        )
    else:
        if out_path.suffix.lower() != ".mp4":
            print("[viz_tracks_fps_3d] Warning: normal mode usually uses a .mp4 --out_path")
        render_video(
            out_mp4=out_path,
            frame_ids=frame_ids,
            gt_by_frame=gt_by_frame,
            pr_by_frame=pr_by_frame,
            stats_by_frame_idx=stats_by_frame_idx,
            pc_dir=pc_dir,
            pc_optional=bool(args.pc_optional),
            gt_colors=gt_colors,
            pr_colors=pr_colors,
            args=args,
        )

    print("[viz_tracks_fps_3d] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())