"""Minimal nuScenes-compatible box used by the FastPoly runtime subset."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion

from tracker_eval.trackers.implementations.fastpoly.data.script.NUSC_CONSTANT import (
    CLASS_SEG_TO_STR_CLASS,
)


class NuscBox:
    """Geometry-only replacement for the nuScenes development-kit ``Box``."""

    def __init__(
        self,
        center: List[float],
        size: List[float],
        rotation: List[float],
        label: int = -1,
        score: float = np.nan,
        velocity: Tuple[float, float, float] = (np.nan, np.nan, np.nan),
        name: Optional[str] = None,
        token: Optional[str] = None,
        init_geo: bool = True,
    ) -> None:
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.wlh = np.asarray(size, dtype=float).reshape(3)
        self.orientation = self.abs_orientation_axisZ(Quaternion(rotation))
        self.label = int(label)
        self.score = float(score)
        self.velocity = tuple(float(value) for value in velocity)
        self.name = name
        self.token = token

        self.yaw = float(self.orientation.radians)
        self.name_label = CLASS_SEG_TO_STR_CLASS[name]
        self.tracking_id = None
        self.corners_ = None
        self.bottom_corners_ = None
        self.volume = None
        self.area = None
        self.norm_corners_ = None
        if init_geo:
            self.reset_box_infos()

    @staticmethod
    def abs_orientation_axisZ(orientation: Quaternion) -> Quaternion:
        return -orientation if orientation.axis[-1] < 0 else orientation

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        width, length, height = self.wlh * float(wlh_factor)
        corners = np.array(
            [
                length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1]),
                width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1]),
                height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1]),
            ],
            dtype=float,
        )
        corners = self.orientation.rotation_matrix @ corners
        return corners + self.center.reshape(3, 1)

    def box_volum(self) -> float:
        return float(np.prod(self.wlh))

    def box_bottom_area(self) -> float:
        return float(self.wlh[0] * self.wlh[1])

    def norm_corners(self) -> np.ndarray:
        top = np.min(self.bottom_corners_[:, 0])
        left = np.max(self.bottom_corners_[:, 1])
        bottom = np.max(self.bottom_corners_[:, 0])
        right = np.min(self.bottom_corners_[:, 1])
        return np.array([top, right, bottom, left])

    def reset_box_infos(self) -> None:
        self.corners_ = self.corners()
        self.bottom_corners_ = self.corners_[:, [2, 3, 7, 6]][:2].T
        self.volume = self.box_volum()
        self.area = self.box_bottom_area()
        self.norm_corners_ = self.norm_corners()

    def __repr__(self) -> str:
        return (
            f"NuscBox(center={self.center.tolist()}, wlh={self.wlh.tolist()}, "
            f"yaw={self.yaw:.3f}, tracking_id={self.tracking_id})"
        )
