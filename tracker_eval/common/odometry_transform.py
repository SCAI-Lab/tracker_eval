from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from tracker_eval.common.types import Box3D, Detection, FrameData


# These describe SOURCE local yaw encoding, not the output coordinate frame.
YawConvention = Literal["jrdb_clockwise", "standard_ccw"]
YAW_CONVENTIONS = ("jrdb_clockwise", "standard_ccw")
DEFAULT_DETECTION_YAW_CONVENTION: YawConvention = "jrdb_clockwise"
DEFAULT_GT_YAW_CONVENTION: YawConvention = "jrdb_clockwise"


def validate_yaw_convention(convention: str) -> None:
    if convention not in YAW_CONVENTIONS:
        raise ValueError(f"Unsupported yaw convention: {convention!r}; choose {YAW_CONVENTIONS}")


def decode_local_yaw_to_ccw(rot_z: float, convention: YawConvention) -> float:
    """Decode source yaw into positive-CCW local yaw (radians about +z).

    JRDB labels use clockwise-positive rot_z in this base-frame pipeline,
    as verified on train/test sequences. The default for the original
    JRDB-trained PersonMinkUNet detections assumes they retain that target
    encoding. Select standard_ccw if a detector outputs in CCW convention.
    """
    validate_yaw_convention(convention)
    return -float(rot_z) if convention == "jrdb_clockwise" else float(rot_z)


def wrap_to_pi(angle: float) -> float:
    """Wrap radians to [-pi, pi)."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


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
    Return (R_world_sensor, t_world_sensor, ego_yaw_ccw).

    These pose terms do not depend on the source box yaw encoding.
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
    *,
    yaw_convention: YawConvention = "jrdb_clockwise",
) -> np.ndarray:
    """Transform a source-local box into canonical global positive-CCW form.

    Position uses p_world = R @ p_local + t. Yaw first decodes the source:
      jrdb_clockwise (default): yaw_world_ccw = -rot_z + ego_yaw_ccw
      standard_ccw:             yaw_world_ccw =  rot_z + ego_yaw_ccw
    The yaw-only box representation retains the existing planar orientation
    approximation for poses with roll/pitch; dimensions are unchanged.
    """
    b = np.asarray(box7, dtype=np.float64).reshape(7)
    R, t, yaw = _world_from_sensor_terms(pose)
    out = b.copy()
    out[:3] = (R @ b[:3]) + t
    local_yaw_ccw = decode_local_yaw_to_ccw(b[6], yaw_convention)
    out[6] = wrap_to_pi(local_yaw_ccw + yaw)
    return out


def transform_frame_data_to_global(
    fd: FrameData,
    pose_by_frame_idx: Dict[int, Pose],
    *,
    yaw_convention: YawConvention = "jrdb_clockwise",
    missing_ok: bool = False,
) -> FrameData:
    """
    Transform all detections in FrameData from local->global using pose for this frame.

    The CSV pose is T_world_sensor: p_w = R p_s + t. Output yaw is CCW.
    With missing_ok=True, a missing pose returns the source frame unchanged;
    runners use the strict default to avoid mixing local and global frames.
    """
    validate_yaw_convention(yaw_convention)
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
        global_box7 = transform_box7_local_to_global(
            local_box7, pose, yaw_convention=yaw_convention,
        )

        bw = Box3D(
            cx=float(global_box7[0]),
            cy=float(global_box7[1]),
            cz=float(global_box7[2]),
            l=float(global_box7[3]),
            w=float(global_box7[4]),
            h=float(global_box7[5]),
            # Source yaw has been decoded to CCW and composed with ego yaw.
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
    *,
    yaw_convention: YawConvention = "jrdb_clockwise",
) -> Dict[str, FrameData]:
    """Convert source-local frames to global boxes with canonical CCW yaw."""
    validate_yaw_convention(yaw_convention)
    out: Dict[str, FrameData] = {}
    for fid, fd in data_by_frame.items():
        out[fid] = transform_frame_data_to_global(
            fd, pose_by_frame_idx, yaw_convention=yaw_convention,
        )
    return out


def normalize_frame_data_yaw(
    fd: FrameData,
    *,
    yaw_convention: YawConvention = "jrdb_clockwise",
) -> FrameData:
    """Decode source-local yaw to canonical local CCW without moving boxes.

    Use for local-only tracking/export. Global transforms already decode source
    yaw themselves: do not normalize first and then pass jrdb_clockwise to a
    global transform, as that would flip the sign twice. Inputs are not mutated;
    centers, dimensions, IDs, scores and other detection metadata are preserved.
    """
    validate_yaw_convention(yaw_convention)
    return replace(fd, dets=[
        replace(det, box=replace(
            det.box,
            rot_z=wrap_to_pi(decode_local_yaw_to_ccw(det.box.rot_z, yaw_convention)),
        ))
        for det in fd.dets
    ])


def normalize_sequence_yaw(
    data_by_frame: Dict[str, FrameData],
    *,
    yaw_convention: YawConvention = "jrdb_clockwise",
) -> Dict[str, FrameData]:
    """Normalize source yaw for a local-only sequence; no odometry is needed."""
    validate_yaw_convention(yaw_convention)
    return {
        fid: normalize_frame_data_yaw(fd, yaw_convention=yaw_convention)
        for fid, fd in data_by_frame.items()
    }


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
