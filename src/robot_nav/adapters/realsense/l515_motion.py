"""读取 L515 Motion Module 的静止 IMU 数据。"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple


Vector3 = Tuple[float, float, float]


@dataclass(frozen=True)
class L515MotionConfig:
    """IMU 采样参数；L515 支持 100/200/400 Hz。"""

    sample_rate_hz: int = 200
    sample_count: int = 200
    wait_timeout_s: float = 5.0
    serial_number: Optional[str] = None
    max_gyro_norm_radps: float = 0.08
    max_gyro_std_radps: float = 0.05
    max_acceleration_std_mps2: float = 0.35


@dataclass(frozen=True)
class L515StationaryMotion:
    """静止样本的均值和最大轴向标准差。

    librealsense 会用设备内部外参把 IMU 方向对齐到深度光学坐标系。
    """

    acceleration_mps2: Vector3
    angular_velocity_radps: Vector3
    acceleration_std_mps2: float
    angular_velocity_std_radps: float


class L515MotionSensor:
    """独占打开 L515 IMU，并验证采样期间相机保持静止。"""

    def __init__(self, config: L515MotionConfig) -> None:
        _validate_config(config)
        try:
            self._rs = importlib.import_module("pyrealsense2")
            self._np = importlib.import_module("numpy")
        except ImportError as exc:
            raise ImportError(
                "L515MotionSensor 需要 pyrealsense2 和 numpy"
            ) from exc

        self.config = config
        self._pipeline: Optional[Any] = None
        self._started = False
        try:
            self._start()
        except Exception:
            self.close()
            raise

    def read_stationary(self) -> L515StationaryMotion:
        """收集一批加速度和角速度，运动或振动明显时拒绝标定。"""
        acceleration = []
        angular_velocity = []
        pipeline = self._require_open()
        deadline = time.monotonic() + self.config.wait_timeout_s

        while (
            len(acceleration) < self.config.sample_count
            or len(angular_velocity) < self.config.sample_count
        ):
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise RuntimeError(
                    "L515 IMU 采样超时："
                    f"accel={len(acceleration)}/{self.config.sample_count}, "
                    f"gyro={len(angular_velocity)}/{self.config.sample_count}"
                )
            timeout_ms = max(1, int(round(remaining_s * 1000.0)))
            try:
                frames = pipeline.wait_for_frames(timeout_ms)
            except Exception as exc:
                raise RuntimeError(f"L515 等待 IMU 数据失败：{exc}") from exc

            if len(acceleration) < self.config.sample_count:
                sample = _read_motion_frame(
                    frames,
                    self._rs.stream.accel,
                )
                if sample is not None:
                    acceleration.append(sample)
            if len(angular_velocity) < self.config.sample_count:
                sample = _read_motion_frame(
                    frames,
                    self._rs.stream.gyro,
                )
                if sample is not None:
                    angular_velocity.append(sample)

        acceleration_array = self._np.asarray(acceleration, dtype=float)
        gyro_array = self._np.asarray(angular_velocity, dtype=float)
        acceleration_mean = acceleration_array.mean(axis=0)
        gyro_mean = gyro_array.mean(axis=0)
        acceleration_std = float(acceleration_array.std(axis=0).max())
        gyro_std = float(gyro_array.std(axis=0).max())

        acceleration_norm = float(self._np.linalg.norm(acceleration_mean))
        gyro_norm = float(self._np.linalg.norm(gyro_mean))
        if not 7.0 <= acceleration_norm <= 12.5:
            raise RuntimeError(
                "L515 加速度模长异常："
                f"{acceleration_norm:.3f} m/s²，无法确定重力方向"
            )
        if gyro_norm > self.config.max_gyro_norm_radps:
            raise RuntimeError(
                "采集 IMU 时相机仍在转动："
                f"{gyro_norm:.3f} rad/s"
            )
        if gyro_std > self.config.max_gyro_std_radps:
            raise RuntimeError(
                "采集 IMU 时角速度波动过大："
                f"标准差 {gyro_std:.3f} rad/s"
            )
        if acceleration_std > self.config.max_acceleration_std_mps2:
            raise RuntimeError(
                "采集 IMU 时振动过大："
                f"加速度标准差 {acceleration_std:.3f} m/s²"
            )

        return L515StationaryMotion(
            acceleration_mps2=_vector3(acceleration_mean),
            angular_velocity_radps=_vector3(gyro_mean),
            acceleration_std_mps2=acceleration_std,
            angular_velocity_std_radps=gyro_std,
        )

    def close(self) -> None:
        """幂等停止 IMU 数据流。"""
        pipeline = self._pipeline
        started = self._started
        self._pipeline = None
        self._started = False
        if pipeline is not None and started:
            try:
                pipeline.stop()
            except Exception:
                pass

    def __enter__(self) -> "L515MotionSensor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _start(self) -> None:
        rs = self._rs
        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.serial_number is not None:
            stream_config.enable_device(self.config.serial_number.strip())
        stream_config.enable_stream(
            rs.stream.accel,
            rs.format.motion_xyz32f,
            self.config.sample_rate_hz,
        )
        stream_config.enable_stream(
            rs.stream.gyro,
            rs.format.motion_xyz32f,
            self.config.sample_rate_hz,
        )
        self._pipeline = pipeline
        pipeline.start(stream_config)
        self._started = True

    def _require_open(self) -> Any:
        if self._pipeline is None or not self._started:
            raise RuntimeError("L515MotionSensor 已关闭")
        return self._pipeline


def _read_motion_frame(frames: Any, stream: Any) -> Optional[Vector3]:
    frame = frames.first_or_default(stream)
    if not frame:
        return None
    data = frame.as_motion_frame().get_motion_data()
    return float(data.x), float(data.y), float(data.z)


def _vector3(values: Any) -> Vector3:
    return float(values[0]), float(values[1]), float(values[2])


def _validate_config(config: L515MotionConfig) -> None:
    if not isinstance(config, L515MotionConfig):
        raise ValueError("config 必须为 L515MotionConfig")
    if config.sample_rate_hz not in (100, 200, 400):
        raise ValueError("sample_rate_hz 必须为 100、200 或 400")
    if (
        isinstance(config.sample_count, bool)
        or not isinstance(config.sample_count, int)
        or config.sample_count <= 0
    ):
        raise ValueError("sample_count 必须为正整数")
    for name in (
        "wait_timeout_s",
        "max_gyro_norm_radps",
        "max_gyro_std_radps",
        "max_acceleration_std_mps2",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} 必须为正有限数")
    if config.serial_number is not None and (
        not isinstance(config.serial_number, str)
        or not config.serial_number.strip()
    ):
        raise ValueError("serial_number 必须为非空字符串或 None")


__all__ = [
    "L515MotionConfig",
    "L515MotionSensor",
    "L515StationaryMotion",
]
