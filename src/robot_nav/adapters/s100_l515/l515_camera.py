"""Intel RealSense L515 的对齐 RGB-D 采集边界。"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from ...core.models import CameraIntrinsics


@dataclass(frozen=True)
class L515Config:
    """L515 流配置；默认 QVGA 以适配 USB 2 链路。"""

    width: int = 320
    height: int = 240
    fps: int = 30
    wait_timeout_s: float = 2.0
    serial_number: Optional[str] = None


@dataclass(frozen=True)
class L515Capture:
    """同一 frameset 的 RGB、米制深度与对齐后的针孔内参。"""

    timestamp_s: float
    rgb: Any
    depth_m: Any
    camera_intrinsics: CameraIntrinsics


class L515Camera:
    """启动 L515，并返回对齐到 RGB 像素坐标的深度图。"""

    def __init__(self, config: L515Config) -> None:
        _validate_config(config)
        try:
            self._rs = importlib.import_module("pyrealsense2")
            self._np = importlib.import_module("numpy")
        except ImportError as exc:
            raise ImportError(
                "L515Camera 需要 pyrealsense2 和 numpy"
            ) from exc

        self.config = config
        self._pipeline: Optional[Any] = None
        self._align: Optional[Any] = None
        self._depth_scale_m = 0.0
        self._depth_to_color_rotation: Optional[Tuple[float, ...]] = None
        self._started = False
        try:
            self._start()
        except Exception:
            self.close()
            raise

    def _start(self) -> None:
        """配置并启动一组同尺寸的 RGB8 与 Z16 数据流。"""
        rs = self._rs
        pipeline = rs.pipeline()
        stream_config = rs.config()
        if self.config.serial_number is not None:
            stream_config.enable_device(self.config.serial_number.strip())
        stream_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps,
        )
        stream_config.enable_stream(
            rs.stream.depth,
            self.config.width,
            self.config.height,
            rs.format.z16,
            self.config.fps,
        )

        self._pipeline = pipeline
        profile = pipeline.start(stream_config)
        self._started = True
        depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        if not math.isfinite(depth_scale) or depth_scale <= 0.0:
            raise RuntimeError("L515 返回了无效的 depth_scale")
        self._depth_scale_m = depth_scale
        depth_profile = profile.get_stream(rs.stream.depth)
        color_profile = profile.get_stream(rs.stream.color)
        extrinsics = depth_profile.get_extrinsics_to(color_profile)
        column_major = tuple(float(value) for value in extrinsics.rotation)
        self._depth_to_color_rotation = tuple(
            column_major[column * 3 + row]
            for row in range(3)
            for column in range(3)
        )
        self._align = rs.align(rs.stream.color)

    @property
    def depth_to_color_rotation(self) -> Tuple[float, ...]:
        """返回把深度/IMU方向转到彩色光学坐标系的按行展开矩阵。"""
        rotation = self._depth_to_color_rotation
        if rotation is None:
            raise RuntimeError("L515Camera 已关闭")
        return rotation

    def capture(self) -> L515Capture:
        """等待并返回一帧对齐 RGB-D；原始深度 0 保持为 0.0 无效值。"""
        pipeline = self._require_open()
        timeout_ms = max(1, int(round(self.config.wait_timeout_s * 1000.0)))
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
            aligned = self._align.process(frames)
        except Exception as exc:
            raise RuntimeError(f"L515 等待 RGB-D 帧失败：{exc}") from exc

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("L515 对齐结果缺少 RGB 或深度帧")

        rgb = self._np.asanyarray(color_frame.get_data())
        raw_depth = self._np.asanyarray(depth_frame.get_data())
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError("L515 RGB 帧不是 H×W×3 图像")
        if raw_depth.ndim != 2 or raw_depth.shape != rgb.shape[:2]:
            raise RuntimeError("L515 对齐后的 RGB 与深度尺寸不一致")

        rgb_copy = rgb[:, :, :3].astype(self._np.uint8, copy=True)
        depth_m = raw_depth.astype(self._np.float32) * self._depth_scale_m
        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        return L515Capture(
            timestamp_s=time.monotonic(),
            rgb=rgb_copy,
            depth_m=depth_m,
            camera_intrinsics=CameraIntrinsics(
                fx=float(intrinsics.fx),
                fy=float(intrinsics.fy),
                cx=float(intrinsics.ppx),
                cy=float(intrinsics.ppy),
            ),
        )

    def close(self) -> None:
        """幂等停止相机数据流。"""
        pipeline = self._pipeline
        started = self._started
        self._pipeline = None
        self._align = None
        self._depth_to_color_rotation = None
        self._started = False
        if pipeline is not None and started:
            try:
                pipeline.stop()
            except Exception:
                pass

    def __enter__(self) -> "L515Camera":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _require_open(self) -> Any:
        if self._pipeline is None or not self._started or self._align is None:
            raise RuntimeError("L515Camera 已关闭")
        return self._pipeline


def _validate_config(config: L515Config) -> None:
    """在占用相机前校验流参数。"""
    if not isinstance(config, L515Config):
        raise ValueError("config 必须为 L515Config")
    for name in ("width", "height", "fps"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    timeout = config.wait_timeout_s
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0.0
    ):
        raise ValueError("wait_timeout_s 必须为正有限数")
    if config.serial_number is not None and (
        not isinstance(config.serial_number, str)
        or not config.serial_number.strip()
    ):
        raise ValueError("serial_number 必须为非空字符串或 None")


__all__ = ["L515Camera", "L515Capture", "L515Config"]
