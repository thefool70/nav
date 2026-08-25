"""Hermes 底盘上的 L515 外参标定入口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from ...core.models import Pose2D, RelativePoseCommand
from ..realsense import L515Config
from ..realsense.calibration import (
    CameraCalibrationConfig,
    calibrate_l515_extrinsics,
)
from .adapter import SlamtecL515Adapter, SlamtecL515Config
from .mount_config import save_camera_extrinsics


@dataclass(frozen=True)
class SlamtecCalibrationResult:
    """Hermes 使用的完整外参与结果文件。"""

    output_path: Path
    extrinsic_residual_m: float


class _HermesCalibrationMotion:
    """把公共标定所需的世界系运动转换为 Adapter 相对位姿命令。"""

    def __init__(self, adapter: SlamtecL515Adapter) -> None:
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


def calibrate_slamtec_l515(
    base_url: str,
    camera_config: L515Config,
    calibration_config: CameraCalibrationConfig,
    output_path: Path,
    action_timeout_s: float = 120.0,
    minimum_localization_quality: int = 1,
    progress: Optional[Callable[[str], None]] = None,
) -> SlamtecCalibrationResult:
    """用 Hermes 地图位姿执行标定运动并保存完整外参。"""
    adapter_config = SlamtecL515Config(
        base_url=base_url,
        camera=None,
        action_timeout_s=action_timeout_s,
        minimum_localization_quality=minimum_localization_quality,
    )
    with SlamtecL515Adapter(
        adapter_config,
        on_action_progress=progress,
    ) as adapter:
        result = calibrate_l515_extrinsics(
            _HermesCalibrationMotion(adapter),
            camera_config,
            calibration_config,
            progress,
        )
    saved_path = save_camera_extrinsics(
        output_path,
        result.extrinsics,
        result.diagnostics,
    )
    if progress is not None:
        progress(f"标定完成，外参已保存到 {saved_path}")
    return SlamtecCalibrationResult(
        saved_path,
        result.extrinsic_residual_m,
    )


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "CameraCalibrationConfig",
    "SlamtecCalibrationResult",
    "calibrate_slamtec_l515",
]
