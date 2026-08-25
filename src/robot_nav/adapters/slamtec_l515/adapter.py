"""把 SLAMTEC Hermes 与外接 L515 组合为统一 ChassisInterface。"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Tuple

from ...core.models import (
    CameraExtrinsics,
    NavigationFrame,
    ObstacleMap,
    Pose2D,
    RelativePoseCommand,
)
from ..realsense import L515Camera, L515Capture, L515Config
from .rest_client import (
    SlamtecExploreMap,
    SlamtecRestClient,
    SlamtecRobotHealth,
    SlamtecSlamState,
    resolve_action_name,
)


MotionFrameCallback = Callable[[NavigationFrame], None]


@dataclass(frozen=True)
class SlamtecL515Config:
    """Hermes REST、可选 L515 和同步 Action 执行配置。"""

    base_url: str = "http://192.168.11.1:1448"
    camera: Optional[L515Config] = field(default_factory=L515Config)
    camera_extrinsics_in_robot: CameraExtrinsics = field(
        default_factory=CameraExtrinsics
    )
    request_timeout_s: float = 5.0
    action_timeout_s: float = 120.0
    action_poll_interval_s: float = 0.2
    motion_frame_interval_s: float = 0.5
    minimum_localization_quality: int = 1
    position_tolerance_m: float = 0.03
    yaw_tolerance_rad: float = math.radians(1.0)


class SlamtecL515Adapter:
    """由 Hermes 提供地图/位姿/规划控制，由外接 L515 提供 RGB-D。"""

    def __init__(
        self,
        config: SlamtecL515Config,
        on_motion_frame: Optional[MotionFrameCallback] = None,
    ) -> None:
        _validate_config(config)
        self.config = config
        self._on_motion_frame = on_motion_frame
        self._last_motion_frame_s = float("-inf")
        self._client = SlamtecRestClient(
            config.base_url, config.request_timeout_s
        )
        self._camera: Optional[L515Camera] = None

        action_names = self._client.get_action_names()
        self._move_to_action = resolve_action_name(
            action_names, "MoveToAction"
        )
        self._rotate_to_action = resolve_action_name(
            action_names, "RotateToAction"
        )
        self._action_names = action_names
        try:
            if config.camera is not None:
                self._camera = L515Camera(config.camera)
        except Exception:
            self.close()
            raise

    @property
    def has_camera(self) -> bool:
        """当前 Adapter 是否启用了外接 L515。"""
        return self._camera is not None

    @property
    def action_names(self) -> Tuple[str, ...]:
        """返回固件公布的 Action 名称，供预检排错。"""
        return self._action_names

    def get_robot_info(self) -> Mapping[str, Any]:
        """读取底盘型号与固件信息，供预检输出。"""
        return self._client.get_robot_info()

    def get_localization_quality(self) -> int:
        """读取当前 0-100 定位质量，供预检与运动前检查。"""
        return self._client.get_localization_quality()

    def get_slam_state(self) -> SlamtecSlamState:
        """读取建图/定位模式及其质量。"""
        return self._client.get_slam_state()

    def get_robot_health(self) -> SlamtecRobotHealth:
        """读取底盘健康摘要。"""
        return self._client.get_robot_health()

    def read_pose(self) -> Pose2D:
        """读取 Hermes 当前地图位姿，供标定控制复用。"""
        return self._client.get_pose()

    def read_frame(self) -> NavigationFrame:
        """组合 Hermes 地图/位姿与同机 L515 对齐 RGB-D。"""
        capture = self._camera.capture() if self._camera is not None else None
        pose = self._client.get_pose()
        slamtec_map = self._client.get_explore_map()
        return _build_navigation_frame(
            timestamp_s=time.monotonic(),
            pose=pose,
            slamtec_map=slamtec_map,
            capture=capture,
            camera_extrinsics=self.config.camera_extrinsics_in_robot,
        )

    def send_relative_pose(self, command: RelativePoseCommand) -> None:
        """把机器人局部相对位姿转换为 Hermes 的全局规划与原地转向。"""
        _validate_command(command)
        self._require_motion_ready()

        start_pose = self._client.get_pose()
        target_xy = _relative_target_world(start_pose, command)
        target_yaw = _wrap_angle(start_pose.yaw_rad + command.yaw_rad)
        translation = math.hypot(command.forward_m, command.left_m)

        if translation > self.config.position_tolerance_m:
            self._execute_action(
                self._move_to_action,
                {"target": {"x": target_xy[0], "y": target_xy[1], "z": 0.0}},
            )

        should_restore_yaw = (
            translation > self.config.position_tolerance_m
            or abs(command.yaw_rad) > self.config.yaw_tolerance_rad
        )
        if should_restore_yaw:
            current_pose = self._client.get_pose()
            yaw_error = abs(
                _angle_difference(target_yaw, current_pose.yaw_rad)
            )
            if yaw_error > self.config.yaw_tolerance_rad:
                self._execute_action(
                    self._rotate_to_action,
                    {"angle": target_yaw},
                )
        self._publish_motion_frame(force=True)

    def _require_motion_ready(self) -> None:
        """建图时接受 quality=0；纯定位时才应用质量阈值。"""
        health = self._client.get_robot_health()
        if health.has_fatal or health.has_error:
            level = "fatal" if health.has_fatal else "error"
            raise RuntimeError(f"Hermes 健康状态为 {level}，拒绝运动")

        state = self._client.get_slam_state()
        if state.mapping_enabled:
            return
        if not state.localization_enabled:
            raise RuntimeError("Hermes 未启用建图或定位，拒绝运动")
        if (
            state.localization_quality
            < self.config.minimum_localization_quality
        ):
            raise RuntimeError(
                "Hermes 定位质量不足，拒绝运动："
                f"{state.localization_quality} < "
                f"{self.config.minimum_localization_quality}"
            )

    def _execute_action(
        self, action_name: str, options: Mapping[str, Any]
    ) -> None:
        """创建并同步等待一个 Action，中断时立即请求底盘终止。"""
        action_id = self._client.create_action(action_name, options)
        try:
            self._client.wait_for_action(
                action_id=action_id,
                timeout_s=self.config.action_timeout_s,
                poll_interval_s=self.config.action_poll_interval_s,
                on_poll=self._publish_motion_frame,
            )
        except BaseException:
            try:
                self._client.abort_current_action()
            except RuntimeError:
                pass
            raise

    def _publish_motion_frame(self, force: bool = False) -> None:
        """运动期间按固定间隔向 Rerun 发布真实底盘帧。"""
        if self._on_motion_frame is None:
            return
        now = time.monotonic()
        elapsed_s = now - self._last_motion_frame_s
        if not force and elapsed_s < self.config.motion_frame_interval_s:
            return
        self._on_motion_frame(self.read_frame())
        self._last_motion_frame_s = now

    def close(self) -> None:
        """幂等关闭外接 L515；REST 客户端没有常驻连接。"""
        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.close()

    def __enter__(self) -> "SlamtecL515Adapter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _build_navigation_frame(
    timestamp_s: float,
    pose: Pose2D,
    slamtec_map: SlamtecExploreMap,
    capture: Optional[L515Capture],
    camera_extrinsics: CameraExtrinsics,
) -> NavigationFrame:
    """把两类设备数据冻结为 core 只读的同一地图坐标帧。"""
    obstacle_map = _to_obstacle_map(slamtec_map)
    if capture is None:
        return NavigationFrame(
            timestamp_s=timestamp_s,
            pose=pose,
            obstacle_map=obstacle_map,
            camera_extrinsics_in_robot=camera_extrinsics,
        )
    return NavigationFrame(
        timestamp_s=timestamp_s,
        pose=pose,
        obstacle_map=obstacle_map,
        depth=_convert_depth(capture.depth_m),
        rgb=_convert_rgb(capture.rgb),
        camera_intrinsics=capture.camera_intrinsics,
        camera_extrinsics_in_robot=camera_extrinsics,
    )


def _to_obstacle_map(source: SlamtecExploreMap) -> ObstacleMap:
    """转换 Hermes 6.3 栅格值和边界原点为项目统一占用图。"""
    rows = []
    for row_index in range(source.height):
        offset = row_index * source.width
        raw_row = source.cells[offset : offset + source.width]
        rows.append(
            tuple(_convert_occupancy(value) for value in raw_row)
        )
    half_cell = 0.5 * source.resolution_m
    return ObstacleMap(
        occupancy=tuple(rows),
        resolution_m=source.resolution_m,
        origin=Pose2D(
            x_m=source.origin_x_m + half_cell,
            y_m=source.origin_y_m + half_cell,
            yaw_rad=0.0,
        ),
        frame_id="slamtec_map",
    )


def _convert_occupancy(value: int) -> Optional[float]:
    # Hermes 6.3.2 实机：0 未知，1..127 可通行，128..255 为占用证据。
    if value == 0:
        return None
    if value <= 127:
        return 0.0
    return 1.0


def _convert_rgb(image: Any):
    rows = image.tolist() if hasattr(image, "tolist") else image
    return tuple(
        tuple((int(pixel[0]), int(pixel[1]), int(pixel[2])) for pixel in row)
        for row in rows
    )


def _convert_depth(image: Any):
    rows = image.tolist() if hasattr(image, "tolist") else image
    return tuple(
        tuple(
            float(value) if _is_finite(value) and float(value) > 0.0 else None
            for value in row
        )
        for row in rows
    )


def _relative_target_world(
    pose: Pose2D, command: RelativePoseCommand
) -> Tuple[float, float]:
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        pose.x_m + command.forward_m * cosine - command.left_m * sine,
        pose.y_m + command.forward_m * sine + command.left_m * cosine,
    )


def _validate_command(command: RelativePoseCommand) -> None:
    if not isinstance(command, RelativePoseCommand) or not all(
        _is_finite(value)
        for value in (command.forward_m, command.left_m, command.yaw_rad)
    ):
        raise ValueError("command 必须为有限 RelativePoseCommand")


def _validate_config(config: SlamtecL515Config) -> None:
    if not isinstance(config, SlamtecL515Config):
        raise ValueError("config 必须为 SlamtecL515Config")
    if config.camera is not None and not isinstance(config.camera, L515Config):
        raise ValueError("camera 必须为 L515Config 或 None")
    if not isinstance(
        config.camera_extrinsics_in_robot, CameraExtrinsics
    ) or not all(
        _is_finite(value)
        for value in (
            config.camera_extrinsics_in_robot.forward_m,
            config.camera_extrinsics_in_robot.left_m,
            config.camera_extrinsics_in_robot.height_m,
            config.camera_extrinsics_in_robot.yaw_rad,
            config.camera_extrinsics_in_robot.pitch_down_rad,
            config.camera_extrinsics_in_robot.roll_rad,
        )
    ):
        raise ValueError(
            "camera_extrinsics_in_robot 必须为有限 CameraExtrinsics"
        )
    for name in (
        "request_timeout_s",
        "action_timeout_s",
        "action_poll_interval_s",
        "motion_frame_interval_s",
        "position_tolerance_m",
        "yaw_tolerance_rad",
    ):
        value = getattr(config, name)
        if not _is_finite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须为正有限数")
    quality = config.minimum_localization_quality
    if (
        isinstance(quality, bool)
        or not isinstance(quality, int)
        or not 0 <= quality <= 100
    ):
        raise ValueError("minimum_localization_quality 必须为 0 到 100 的整数")


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return _wrap_angle(target_rad - current_rad)


def _is_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


__all__ = ["SlamtecL515Adapter", "SlamtecL515Config"]
