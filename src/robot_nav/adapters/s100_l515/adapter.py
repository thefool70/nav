"""把 S100 与 L515 组合为项目统一的 ChassisInterface。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from ...core.models import NavigationFrame, Pose2D, RelativePoseCommand
from ..realsense import L515Camera, L515Capture, L515Config
from .mapping import CameraMount, DepthOccupancyMap, DepthOccupancyMapConfig
from .motion import S100MotionConfig, S100MotionController
from .planner import plan_known_free_path
from .ros_slam import RosSlamConfig, RosSlamSource
from .s100_serial import S100SerialConfig, S100SerialConnection


MotionFrameCallback = Callable[[NavigationFrame], None]


@dataclass(frozen=True)
class S100L515Config:
    """S100、L515、建图和低速运动执行的组合配置。"""

    serial: S100SerialConfig
    camera_mount: CameraMount
    camera: L515Config = field(default_factory=L515Config)
    obstacle_map: DepthOccupancyMapConfig = field(
        default_factory=DepthOccupancyMapConfig
    )
    motion: S100MotionConfig = field(default_factory=S100MotionConfig)
    slam: Optional[RosSlamConfig] = None
    robot_radius_m: float = 0.25
    max_translation_step_m: float = 0.20
    max_translation_steps: int = 100


class S100L515Adapter:
    """读取真实 RGB-D/里程计，并同步执行核心输出的相对位姿命令。"""

    def __init__(
        self,
        config: S100L515Config,
        on_motion_frame: Optional[MotionFrameCallback] = None,
    ) -> None:
        _validate_config(config)
        self.config = config
        self._on_motion_frame = on_motion_frame
        self._camera: Optional[L515Camera] = None
        self._connection: Optional[S100SerialConnection] = None
        self._motion: Optional[S100MotionController] = None
        self._map: Optional[DepthOccupancyMap] = None
        self._slam: Optional[RosSlamSource] = None
        self._last_frame: Optional[NavigationFrame] = None

        try:
            self._connection = S100SerialConnection(config.serial)
            self._motion = S100MotionController(
                self._connection, config.motion
            )
            self._motion.preflight()
            if config.slam is None:
                self._camera = L515Camera(config.camera)
                self._map = DepthOccupancyMap(config.obstacle_map)
            else:
                self._slam = RosSlamSource(
                    config.slam,
                    config.camera_mount,
                    lambda: self._require_motion().pose,
                )
        except Exception:
            self.close()
            raise

    def read_frame(self) -> NavigationFrame:
        """读取里程计与对齐 RGB-D，更新占用图并返回统一导航帧。"""
        motion = self._require_motion()
        pose = motion.read_pose()
        if self._slam is not None:
            frame = self._slam.read_frame()
            self._last_frame = frame
            return frame

        camera, obstacle_map = self._require_direct_sensors()
        capture = camera.capture()
        obstacle_map.update(
            capture.depth_m,
            capture.camera_intrinsics,
            pose,
            self.config.camera_mount,
        )
        frame = _build_navigation_frame(
            capture,
            pose,
            obstacle_map,
            self.config.camera_mount,
        )
        self._last_frame = frame
        return frame

    def send_relative_pose(self, command: RelativePoseCommand) -> None:
        """规划已知自由区路径，分段驱动差速底盘，最后恢复命令目标朝向。"""
        _validate_command(command)
        motion = self._require_motion()
        start_pose = (
            self._last_frame.pose
            if self._last_frame is not None
            else self.read_frame().pose
        )
        target_world_xy = _relative_target_world(start_pose, command)
        target_yaw = _wrap_angle(start_pose.yaw_rad + command.yaw_rad)

        if math.hypot(command.forward_m, command.left_m) > 1.0e-9:
            self._move_to_world_xy(target_world_xy)
        current_world_pose = (
            self._last_frame.pose
            if self._last_frame is not None
            else self.read_frame().pose
        )
        odometry_pose = motion.read_pose()
        motion.turn_to_world_yaw(
            odometry_pose.yaw_rad
            + _angle_difference(target_yaw, current_world_pose.yaw_rad)
        )
        self._publish_stopped_frame()

    def _move_to_world_xy(self, target_world_xy: Tuple[float, float]) -> None:
        """每前进一小段重新采集深度和规划，直到到达相对位姿目标。"""
        motion = self._require_motion()
        frame = (
            self._last_frame
            if self._last_frame is not None
            else self.read_frame()
        )
        for _ in range(self.config.max_translation_steps):
            pose = frame.pose
            remaining = math.hypot(
                target_world_xy[0] - pose.x_m,
                target_world_xy[1] - pose.y_m,
            )
            if remaining <= self.config.motion.position_tolerance_m:
                return

            path = plan_known_free_path(
                frame.obstacle_map,
                (pose.x_m, pose.y_m),
                target_world_xy,
                self.config.robot_radius_m,
            )
            next_world_xy = _limit_step(
                (pose.x_m, pose.y_m),
                path[0],
                self.config.max_translation_step_m,
            )
            self._drive_map_step(motion, pose, next_world_xy)
            frame = self._publish_stopped_frame()

        raise RuntimeError("S100 分段规划次数已用尽，仍未到达相对位姿目标")

    def _drive_map_step(
        self,
        motion: S100MotionController,
        map_pose: Pose2D,
        target_map_xy: Tuple[float, float],
    ) -> None:
        """把地图系下一小段路径转换为轮速里程计中的局部执行目标。"""
        delta_x = target_map_xy[0] - map_pose.x_m
        delta_y = target_map_xy[1] - map_pose.y_m
        distance = math.hypot(delta_x, delta_y)
        if distance <= 1.0e-9:
            return
        heading_map = math.atan2(delta_y, delta_x)
        relative_heading = _angle_difference(heading_map, map_pose.yaw_rad)
        odometry_pose = motion.read_pose()
        heading_odometry = odometry_pose.yaw_rad + relative_heading
        motion.drive_to_world_xy(
            (
                odometry_pose.x_m + distance * math.cos(heading_odometry),
                odometry_pose.y_m + distance * math.sin(heading_odometry),
            )
        )

    def _publish_stopped_frame(self) -> NavigationFrame:
        """动作停止后采集新帧，并按需发送给可视化界面。"""
        frame = self.read_frame()
        if self._on_motion_frame is not None:
            self._on_motion_frame(frame)
        return frame

    def close(self) -> None:
        """停止 ROS 数据源，再关闭底盘串口与本地 L515 数据源。"""
        slam = self._slam
        connection = self._connection
        camera = self._camera
        self._slam = None
        if slam is not None:
            slam.close()
        self._motion = None
        self._connection = None
        self._camera = None
        self._map = None
        self._last_frame = None
        if connection is not None:
            connection.close()
        if camera is not None:
            camera.close()

    def __enter__(self) -> "S100L515Adapter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _require_motion(self) -> S100MotionController:
        if self._motion is None:
            raise RuntimeError("S100L515Adapter 已关闭")
        return self._motion

    def _require_direct_sensors(
        self,
    ) -> Tuple[L515Camera, DepthOccupancyMap]:
        if self._camera is None or self._map is None:
            raise RuntimeError("本地 L515 数据源未启用")
        return self._camera, self._map


def _build_navigation_frame(
    capture: L515Capture,
    pose: Pose2D,
    obstacle_map: DepthOccupancyMap,
    camera_mount: CameraMount,
) -> NavigationFrame:
    """把设备原始数组冻结为 core 可直接消费的 NavigationFrame。"""
    return NavigationFrame(
        timestamp_s=capture.timestamp_s,
        pose=pose,
        obstacle_map=obstacle_map.to_obstacle_map(),
        depth=_convert_depth(capture.depth_m),
        rgb=_convert_rgb(capture.rgb),
        camera_intrinsics=capture.camera_intrinsics,
        camera_pose_in_robot=Pose2D(
            x_m=camera_mount.forward_m,
            y_m=camera_mount.left_m,
            yaw_rad=camera_mount.yaw_rad,
        ),
    )


def _convert_rgb(image: Any):
    rows = image.tolist() if hasattr(image, "tolist") else image
    return tuple(
        tuple(
            (int(pixel[0]), int(pixel[1]), int(pixel[2]))
            for pixel in row
        )
        for row in rows
    )


def _convert_depth(image: Any):
    rows = image.tolist() if hasattr(image, "tolist") else image
    return tuple(
        tuple(
            float(value)
            if _is_finite(value) and float(value) > 0.0
            else None
            for value in row
        )
        for row in rows
    )


def _relative_target_world(
    pose: Pose2D,
    command: RelativePoseCommand,
) -> Tuple[float, float]:
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        pose.x_m + command.forward_m * cosine - command.left_m * sine,
        pose.y_m + command.forward_m * sine + command.left_m * cosine,
    )


def _limit_step(
    start: Tuple[float, float],
    target: Tuple[float, float],
    maximum_distance_m: float,
) -> Tuple[float, float]:
    delta_x = target[0] - start[0]
    delta_y = target[1] - start[1]
    distance = math.hypot(delta_x, delta_y)
    if distance <= maximum_distance_m:
        return target
    scale = maximum_distance_m / distance
    return start[0] + delta_x * scale, start[1] + delta_y * scale


def _validate_command(command: RelativePoseCommand) -> None:
    if not isinstance(command, RelativePoseCommand) or not all(
        _is_finite(value)
        for value in (command.forward_m, command.left_m, command.yaw_rad)
    ):
        raise ValueError("command 必须为有限 RelativePoseCommand")


def _validate_config(config: S100L515Config) -> None:
    if not isinstance(config, S100L515Config):
        raise ValueError("config 必须为 S100L515Config")
    if not isinstance(config.serial, S100SerialConfig):
        raise ValueError("serial 必须为 S100SerialConfig")
    if not isinstance(config.camera_mount, CameraMount):
        raise ValueError("camera_mount 必须为 CameraMount")
    mount_values = (
        config.camera_mount.height_m,
        config.camera_mount.forward_m,
        config.camera_mount.left_m,
        config.camera_mount.yaw_rad,
        config.camera_mount.pitch_down_rad,
    )
    if not all(_is_finite(value) for value in mount_values):
        raise ValueError("camera_mount 的位置和角度必须为有限数")
    if float(config.camera_mount.height_m) <= 0.0:
        raise ValueError("camera_mount.height_m 必须为正数")
    if not isinstance(config.camera, L515Config):
        raise ValueError("camera 必须为 L515Config")
    if not isinstance(config.obstacle_map, DepthOccupancyMapConfig):
        raise ValueError("obstacle_map 必须为 DepthOccupancyMapConfig")
    if not isinstance(config.motion, S100MotionConfig):
        raise ValueError("motion 必须为 S100MotionConfig")
    if config.slam is not None and not isinstance(config.slam, RosSlamConfig):
        raise ValueError("slam 必须为 RosSlamConfig 或 None")
    if (
        not _is_finite(config.robot_radius_m)
        or float(config.robot_radius_m) <= 0.0
    ):
        raise ValueError("robot_radius_m 必须为正有限数")
    if (
        not _is_finite(config.max_translation_step_m)
        or float(config.max_translation_step_m) <= 0.0
    ):
        raise ValueError("max_translation_step_m 必须为正有限数")
    if (
        isinstance(config.max_translation_steps, bool)
        or not isinstance(config.max_translation_steps, int)
        or config.max_translation_steps <= 0
    ):
        raise ValueError("max_translation_steps 必须为正整数")


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return _wrap_angle(target_rad - current_rad)


def _is_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["S100L515Adapter", "S100L515Config"]
