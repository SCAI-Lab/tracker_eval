from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData


@dataclass(frozen=True)
class Pose:
    t: np.ndarray      # (3,)
    q: np.ndarray      # (4,) qx,qy,qz,qw


def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # normalized quaternion -> rotation matrix
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),       1.0 - 2.0*(xx + yy)],
    ], dtype=np.float64)
    return R


def _quat_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    # yaw (about z) from quaternion
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n < 1e-12:
        return 0.0
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    siny_cosp = 2.0 * (qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0 * (qy*qy + qz*qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def load_odometry_csv(csv_path: str) -> Dict[int, Pose]:
    """
    Load odometry CSV into dict: frame_idx -> Pose.
    Assumes the CSV rows are in frame order and correspond 1:1 to frame indices (0..N-1)
    even if timestamps differ or frames are missing.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"Odometry CSV not found: {p}")

    poses: Dict[int, Pose] = {}
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required = {"timestamp_ns", "x", "y", "z", "qx", "qy", "qz", "qw"}
        if not required.issubset(set(r.fieldnames or [])):
            raise ValueError(f"Odometry CSV missing required columns. Have: {r.fieldnames}")

        for i, row in enumerate(r):
            t = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            q = np.array([float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"])], dtype=np.float64)
            poses[i] = Pose(t=t, q=q)

    if not poses:
        raise ValueError(f"Odometry CSV empty: {p}")
    return poses


def _frame_id_to_int(frame_id: str) -> int:
    s = str(frame_id).strip()
    if "." in s:
        s = s.split(".")[0]
    return int(s)


def _world_from_sensor_terms(
    pose: Pose,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return the exact (R, t, yaw) terms used by this package for local->global.

    The package convention is intentionally kept in one function. In
    particular, JRDB yaw uses the package's empirical
    ``yaw_global = yaw_local - ego_yaw`` rule rather than standard yaw
    composition.
    """
    qx, qy, qz, qw = (
        float(pose.q[0]),
        float(pose.q[1]),
        float(pose.q[2]),
        float(pose.q[3]),
    )
    R = _quat_to_R(qx, qy, qz, qw)
    t = pose.t.astype(np.float64, copy=False)
    yaw = _quat_yaw(qx, qy, qz, qw)

    return R, t, float(yaw)


def transform_box7_local_to_global(
    box7: np.ndarray,
    pose: Pose,
) -> np.ndarray:
    """
    Transform ``[cx,cy,cz,l,w,h,rot_z]`` with the package's exact convention.
    """
    b = np.asarray(box7, dtype=np.float64).reshape(7)
    R, t, yaw = _world_from_sensor_terms(pose)
    out = b.copy()
    out[:3] = (R @ b[:3]) + t
    # Deliberately non-standard JRDB/package yaw rule. Do not change.
    out[6] = float(b[6] - yaw)
    return out


def transform_frame_data_to_global(
    fd: FrameData,
    pose_by_frame_idx: Dict[int, Pose],
    *,
    missing_ok: bool = False,
) -> FrameData:
    """
    Transform all detections in FrameData from local->global using pose for this frame.

    The CSV pose is T_world_sensor: p_w = R p_s + t.
    """
    k = _frame_id_to_int(fd.frame_id)
    pose = pose_by_frame_idx.get(k, None)
    if pose is None:
        if missing_ok:
            return fd
        raise KeyError(f"No odometry pose for frame_idx={k} (frame_id={fd.frame_id})")

    out_dets = []
    for det in fd.dets:
        b: Box3D = det.box
        local_box7 = np.asarray(
            [b.cx, b.cy, b.cz, b.l, b.w, b.h, b.rot_z],
            dtype=np.float64,
        )
        global_box7 = transform_box7_local_to_global(local_box7, pose)

        bw = Box3D(
            cx=float(global_box7[0]),
            cy=float(global_box7[1]),
            cz=float(global_box7[2]),
            l=float(global_box7[3]),
            w=float(global_box7[4]),
            h=float(global_box7[5]),
            # NOTE (JRDB yaw sign fix):
            # Empirically, JRDB labels/detections in the local (base) frame already rotate with ego motion
            # as if an ego-compensation was applied with the wrong sign. If we compose yaw "correctly"
            # (rot_z + ego_yaw), boxes spin/double-rotate when the robot turns.
            # Therefore we apply the *reverse* composition here: subtract ego yaw.
            rot_z=float(global_box7[6]),
        )

        out_dets.append(
            Detection(
                frame_id=det.frame_id,
                track_id=int(det.track_id),
                box=bw,
                score=det.score,
                label=det.label,
                raw_label_id=det.raw_label_id,
            )
        )

    return FrameData(frame_id=fd.frame_id, dets=out_dets)


def transform_sequence_to_global(
    data_by_frame: Dict[str, FrameData],
    pose_by_frame_idx: Dict[int, Pose],
) -> Dict[str, FrameData]:
    out: Dict[str, FrameData] = {}
    for fid, fd in data_by_frame.items():
        out[fid] = transform_frame_data_to_global(fd, pose_by_frame_idx)
    return out


def build_timestamps_by_frame_from_odometry(
    pose_csv_path: str,
    detections_by_frame: Dict[str, FrameData],
) -> Dict[str, float]:
    """
    Create timestamps_by_frame (seconds) from the odometry CSV timestamps, keyed by frame_id.
    Assumes CSV row index == int(frame_id).
    """
    ts: Dict[int, float] = {}
    with Path(pose_csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            ts_ns = int(row["timestamp_ns"])
            ts[i] = float(ts_ns) * 1e-9

    out: Dict[str, float] = {}
    for frame_id in detections_by_frame.keys():
        k = _frame_id_to_int(frame_id)
        if k in ts:
            out[frame_id] = ts[k]
    return out
