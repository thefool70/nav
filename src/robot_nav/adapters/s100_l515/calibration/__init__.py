"""S100 底盘上的 L515 外参标定入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ...realsense import L515Config
from ...realsense.calibration import (
    CameraCalibrationConfig,
    calibrate_l515_extrinsics,
)
from ..mapping import CameraMount
from ..motion import S100MotionConfig, S100MotionController
from ..mount_config import save_camera_mount
from ..s100_serial import S100SerialConfig, S100SerialConnection


@dataclass(frozen=True)
class CameraCalibrationResult:
    """S100 使用的安装外参与结果文件。"""

    camera_mount: CameraMount
    output_path: Path
    extrinsic_residual_m: float


def calibrate_s100_l515(
    serial_config: S100SerialConfig,
    camera_config: L515Config,
    motion_config: S100MotionConfig,
    calibration_config: CameraCalibrationConfig,
    output_path: Path,
    progress: Optional[Callable[[str], None]] = None,
) -> CameraCalibrationResult:
    """打开 S100 后调用底盘无关标定流程，并保存 S100 外参文件。"""
    with S100SerialConnection(serial_config) as connection:
        controller = S100MotionController(connection, motion_config)
        controller.preflight()
        result = calibrate_l515_extrinsics(
            controller,
            camera_config,
            calibration_config,
            progress,
        )

    extrinsics = result.extrinsics
    mount = CameraMount(
        height_m=extrinsics.height_m,
        forward_m=extrinsics.forward_m,
        left_m=extrinsics.left_m,
        yaw_rad=extrinsics.yaw_rad,
        pitch_down_rad=extrinsics.pitch_down_rad,
        roll_rad=extrinsics.roll_rad,
    )
    saved_path = save_camera_mount(
        output_path,
        mount,
        result.diagnostics,
    )
    if progress is not None:
        progress(f"标定完成，外参已保存到 {saved_path}")
    return CameraCalibrationResult(
        mount,
        saved_path,
        result.extrinsic_residual_m,
    )


__all__ = [
    "CameraCalibrationConfig",
    "CameraCalibrationResult",
    "calibrate_s100_l515",
]
