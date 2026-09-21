"""相机安装外参标定所需的公共运动接口与场地运动参数。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Tuple

from ....core.models import Pose2D


class CalibrationMotionController(Protocol):
    """标定只需要底盘提供的三个同步二维运动接口。"""

    def read_pose(self) -> Pose2D:
        ...

    def turn_to_world_yaw(self, target_yaw_rad: float) -> None:
        ...

    def drive_to_world_xy(self, target_world_xy: Tuple[float, float]) -> None:
        ...


@dataclass(frozen=True)
class CameraCalibrationConfig:
    """唯一需要按场地调整的两段标定运动。"""

    turn_angle_rad: float = math.radians(30.0)
    drive_distance_m: float = 0.20


def _validate_calibration_config(config: CameraCalibrationConfig) -> None:
    if not isinstance(config, CameraCalibrationConfig):
        raise ValueError("calibration_config 必须为 CameraCalibrationConfig")
    if not 0.0 < config.turn_angle_rad <= math.radians(45):
        raise ValueError("turn_angle_rad 必须在 0 到 45° 之间")
    if not 0.0 < config.drive_distance_m <= 0.50:
        raise ValueError("drive_distance_m 必须在 0 到 0.50 m 之间")


__all__ = [
    "CalibrationMotionController",
    "CameraCalibrationConfig",
]
