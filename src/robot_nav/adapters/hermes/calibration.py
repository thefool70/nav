"""标定所需的地图位姿运动控制，以及 Hermes + D435i 外参标定入口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from ...core.models import Pose2D, RelativePoseCommand
from ..realsense.d435i_camera import D435iConfig
from ..realsense.calibration import (
    CameraCalibrationConfig,
)
from .adapter import HermesAdapter, HermesConfig


@dataclass(frozen=True)
class HermesCalibrationResult:
    """Hermes 使用的完整外参与结果文件。"""

    output_path: Path
    extrinsic_residual_m: float


class HermesCalibrationMotion:
    """把公共标定所需的世界系运动转换为 Adapter 相对位姿命令。"""

    def __init__(self, adapter: HermesAdapter) -> None:
        self._adapter = adapter

    def read_pose(self) -> Pose2D:
        return self._adapter.read_pose()

    def turn_to_world_yaw(self, target_yaw_rad: float) -> None:
        pose = self.read_pose()
        self._adapter.send_relative_pose(
            RelativePoseCommand(
                yaw_rad=_angle_difference(target_yaw_rad, pose.yaw_rad)
            )
        )

    def drive_to_world_xy(self, target_world_xy: Tuple[float, float]) -> None:
        pose = self.read_pose()
        delta_x = target_world_xy[0] - pose.x_m
        delta_y = target_world_xy[1] - pose.y_m
        cosine = math.cos(pose.yaw_rad)
        sine = math.sin(pose.yaw_rad)
        self._adapter.send_relative_pose(
            RelativePoseCommand(
                forward_m=delta_x * cosine + delta_y * sine,
                left_m=-delta_x * sine + delta_y * cosine,
            )
        )


def calibrate_hermes(
    base_url: str,
    camera_config: D435iConfig,
    calibration_config: CameraCalibrationConfig,
    output_path: Path,
    action_timeout_s: float = 120.0,
    minimum_localization_quality: int = 1,
    progress: Optional[Callable[[str], None]] = None,
    camera_factory=None,
) -> HermesCalibrationResult:
    """用 Hermes 地图位姿执行标定运动并保存完整外参。"""
    from ..realsense.d435i_camera import D435iCalibrationCamera
    from .d435i_calibration import calibrate_hermes_d435i

    factory = camera_factory or D435iCalibrationCamera
    camera = factory(camera_config)
    try:
        result = calibrate_hermes_d435i(
            base_url=base_url, camera=camera, output_path=output_path,
            turn_angle_deg=math.degrees(calibration_config.turn_angle_rad),
            drive_distance_m=calibration_config.drive_distance_m,
            action_timeout_s=action_timeout_s,
            minimum_localization_quality=minimum_localization_quality,
            progress=progress if progress is not None else lambda _message: None,
        )
    finally:
        camera.close()
    return HermesCalibrationResult(result.output_path, result.extrinsic_residual_m)


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "HermesCalibrationMotion",
    "CameraCalibrationConfig",
    "HermesCalibrationResult",
    "calibrate_hermes",
]
