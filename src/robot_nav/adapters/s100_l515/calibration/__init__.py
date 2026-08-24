"""S100 与 L515 外参标定的四阶段入口。

``calibrate_s100_l515`` 是阅读起点；地面、视觉运动和外参求解分别位于同目录
的三个小模块中。
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ....core.models import Pose2D
from ..l515_camera import L515Camera, L515Capture, L515Config
from ..l515_motion import (
    L515MotionConfig,
    L515MotionSensor,
    L515StationaryMotion,
)
from ..mapping import CameraMount
from ..motion import S100MotionConfig, S100MotionController
from ..mount_config import save_camera_mount
from ..s100_serial import S100SerialConfig, S100SerialConnection
from .floor import FloorEstimate, estimate_floor, mount_angles_from_up
from .planar import solve_planar_extrinsic
from .visual_motion import (
    CapturedFrame,
    estimate_motion_pairs,
    require_visual_features,
)


ProgressCallback = Callable[[str], None]
_WARMUP_FRAMES = 15
_SETTLE_FRAMES = 5
_MAX_ROLL_RAD = math.radians(5.0)


@dataclass(frozen=True)
class CameraCalibrationConfig:
    """唯一需要按场地调整的两段标定运动。"""

    turn_angle_rad: float = math.radians(30.0)
    drive_distance_m: float = 0.20


@dataclass(frozen=True)
class CameraCalibrationResult:
    """最终外参与最直接的质量指标。"""

    camera_mount: CameraMount
    output_path: Path
    extrinsic_residual_m: float


@dataclass(frozen=True)
class _CollectedData:
    frames: Tuple[CapturedFrame, ...]
    floor: FloorEstimate
    up_in_color: Any
    pitch_down_rad: float
    roll_rad: float


def calibrate_s100_l515(
    serial_config: S100SerialConfig,
    camera_config: L515Config,
    motion_config: S100MotionConfig,
    calibration_config: CameraCalibrationConfig,
    output_path: Path,
    progress: Optional[ProgressCallback] = None,
) -> CameraCalibrationResult:
    """依次采集 IMU、拟合地面、执行运动、求解并保存外参。"""
    _validate_calibration_config(calibration_config)
    report = progress if progress is not None else lambda _message: None
    np = importlib.import_module("numpy")
    cv2 = importlib.import_module("cv2")

    stationary, up_in_depth = _read_stationary_imu(
        camera_config,
        np,
        report,
    )
    collected = _collect_calibration_data(
        serial_config,
        camera_config,
        motion_config,
        calibration_config,
        up_in_depth,
        cv2,
        np,
        report,
    )

    report("[4/4] 对齐 RGB-D 运动与底盘里程计，求解外参……")
    motion_pairs = estimate_motion_pairs(
        collected.frames,
        collected.up_in_color,
        cv2,
        np,
        report,
    )
    forward_m, left_m, yaw_rad, residual_m = solve_planar_extrinsic(
        motion_pairs,
        np,
    )
    mount = CameraMount(
        height_m=collected.floor.height_m,
        forward_m=forward_m,
        left_m=left_m,
        yaw_rad=yaw_rad,
        pitch_down_rad=collected.pitch_down_rad,
    )
    diagnostics: Dict[str, Any] = {
        "roll_deg": round(math.degrees(collected.roll_rad), 6),
        "floor_inlier_ratio": round(collected.floor.inlier_ratio, 6),
        "imu_floor_angle_deg": round(
            math.degrees(collected.floor.imu_angle_rad),
            6,
        ),
        "imu_acceleration_std_mps2": round(
            stationary.acceleration_std_mps2,
            6,
        ),
        "imu_angular_velocity_std_radps": round(
            stationary.angular_velocity_std_radps,
            6,
        ),
        "extrinsic_residual_m": round(residual_m, 6),
        "motion_visual_inliers": [
            pair.visual_inliers for pair in motion_pairs
        ],
    }
    saved_path = save_camera_mount(output_path, mount, diagnostics)
    report(f"标定完成，外参已保存到 {saved_path}")
    return CameraCalibrationResult(mount, saved_path, residual_m)


def _read_stationary_imu(
    camera_config: L515Config,
    np: Any,
    report: ProgressCallback,
) -> Tuple[L515StationaryMotion, Any]:
    report("[1/4] 采集 L515 Motion Module 静止数据……")
    with L515MotionSensor(
        L515MotionConfig(serial_number=camera_config.serial_number)
    ) as sensor:
        stationary = sensor.read_stationary()
    up_in_depth = _normalized(
        np.asarray(stationary.acceleration_mps2, dtype=float),
        np,
        "IMU 重力方向",
    )
    return stationary, up_in_depth


def _collect_calibration_data(
    serial_config: S100SerialConfig,
    camera_config: L515Config,
    motion_config: S100MotionConfig,
    calibration_config: CameraCalibrationConfig,
    up_in_depth: Any,
    cv2: Any,
    np: Any,
    report: ProgressCallback,
) -> _CollectedData:
    """检查初始观测，再保存每段受控运动后的停车帧。"""
    report("[2/4] 打开 RGB-D 与底盘，拟合地面并检查安装姿态……")
    with S100SerialConnection(serial_config) as connection:
        controller = S100MotionController(connection, motion_config)
        controller.preflight()
        with L515Camera(camera_config) as camera:
            initial = _capture_after_settle(camera, _WARMUP_FRAMES)
            up_hint = _imu_up_in_color(camera, up_in_depth, np)
            floor = estimate_floor(initial, up_hint, np)
            up_in_color = _normalized(
                up_hint + floor.up_in_color,
                np,
                "IMU 与地面融合方向",
            )
            pitch_down_rad, roll_rad = mount_angles_from_up(up_in_color)
            if abs(roll_rad) > _MAX_ROLL_RAD:
                raise RuntimeError(
                    "L515 侧倾超过二维模型允许范围："
                    f"{math.degrees(roll_rad):.2f}°；请调平后重试"
                )
            require_visual_features(initial, cv2)

            start_pose = controller.read_pose()
            frames = [CapturedFrame("起点", start_pose, initial)]
            report(
                "地面检查通过："
                f"height={floor.height_m:.3f} m，"
                f"pitch-down={math.degrees(pitch_down_rad):.2f}°，"
                f"roll={math.degrees(roll_rad):.2f}°"
            )
            _perform_calibration_motion(
                frames,
                start_pose,
                camera,
                controller,
                calibration_config,
                report,
            )
            return _CollectedData(
                tuple(frames),
                floor,
                up_in_color,
                pitch_down_rad,
                roll_rad,
            )


def _perform_calibration_motion(
    frames: List[CapturedFrame],
    start_pose: Pose2D,
    camera: L515Camera,
    controller: S100MotionController,
    config: CameraCalibrationConfig,
    report: ProgressCallback,
) -> None:
    report("[3/4] 开始受控运动：左转、回正、右转、回正、前进……")
    yaw_targets = (
        ("左转", start_pose.yaw_rad + config.turn_angle_rad),
        ("左转后回正", start_pose.yaw_rad),
        ("右转", start_pose.yaw_rad - config.turn_angle_rad),
        ("右转后回正", start_pose.yaw_rad),
    )
    for label, target_yaw in yaw_targets:
        report(f"  {label}")
        controller.turn_to_world_yaw(target_yaw)
        frames.append(_capture_with_pose(label, camera, controller))

    report(f"  向前移动 {config.drive_distance_m:.2f} m")
    target_xy = (
        start_pose.x_m
        + config.drive_distance_m * math.cos(start_pose.yaw_rad),
        start_pose.y_m
        + config.drive_distance_m * math.sin(start_pose.yaw_rad),
    )
    controller.drive_to_world_xy(target_xy)
    frames.append(_capture_with_pose("前进", camera, controller))


def _capture_with_pose(
    label: str,
    camera: L515Camera,
    controller: S100MotionController,
) -> CapturedFrame:
    capture = _capture_after_settle(camera, _SETTLE_FRAMES)
    return CapturedFrame(label, controller.read_pose(), capture)


def _capture_after_settle(
    camera: L515Camera,
    frame_count: int,
) -> L515Capture:
    capture: Optional[L515Capture] = None
    for _ in range(frame_count):
        capture = camera.capture()
    if capture is None:
        raise RuntimeError("没有采集到 L515 RGB-D 帧")
    return capture


def _imu_up_in_color(camera: L515Camera, up_in_depth: Any, np: Any) -> Any:
    depth_to_color = np.asarray(
        camera.depth_to_color_rotation,
        dtype=float,
    ).reshape(3, 3)
    return _normalized(
        depth_to_color @ up_in_depth,
        np,
        "彩色相机重力方向",
    )


def _normalized(vector: Any, np: Any, name: str) -> Any:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise RuntimeError(f"{name} 无法归一化")
    return vector / norm


def _validate_calibration_config(config: CameraCalibrationConfig) -> None:
    if not isinstance(config, CameraCalibrationConfig):
        raise ValueError("calibration_config 必须为 CameraCalibrationConfig")
    if not 0.0 < config.turn_angle_rad <= math.radians(45):
        raise ValueError("turn_angle_rad 必须在 0 到 45° 之间")
    if not 0.0 < config.drive_distance_m <= 0.50:
        raise ValueError("drive_distance_m 必须在 0 到 0.50 m 之间")


__all__ = [
    "CameraCalibrationConfig",
    "CameraCalibrationResult",
    "calibrate_s100_l515",
]
