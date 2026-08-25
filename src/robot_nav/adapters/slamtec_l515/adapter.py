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
from ..chassis import MotionStalledError, RecoverableMotionError
from ..realsense import L515Camera, L515Capture, L515Config
from .observed_map import L515ObservedMap
from .rest_client import (
    SlamtecActionError,
    SlamtecExploreMap,
    SlamtecRestClient,
    SlamtecRobotHealth,
    SlamtecSlamState,
    resolve_action_name,
)


MotionFrameCallback = Callable[[NavigationFrame], None]
ActionProgressCallback = Callable[[str], None]


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
    action_progress_interval_s: float = 2.0
    action_stall_timeout_s: float = 15.0
    action_stall_translation_m: float = 0.02
    action_stall_rotation_rad: float = math.radians(1.0)
    motion_frame_interval_s: float = 0.5
    minimum_localization_quality: int = 1
    position_tolerance_m: float = 0.03
    # 与 core 的扫描朝向容差一致，避免为已经可接受的微小误差再创建 Action。
    yaw_tolerance_rad: float = math.radians(5.0)


@dataclass
class _ActionMonitorState:
    started_s: float
    last_sample_s: float
    last_motion_s: float
    last_motion_pose: Pose2D


class _ActionStalledError(RuntimeError):
    """活跃 Hermes Action 长时间没有产生有效位姿变化。"""


class SlamtecL515Adapter:
    """由 Hermes 提供地图/位姿/规划控制，由外接 L515 提供 RGB-D。"""

    def __init__(
        self,
        config: SlamtecL515Config,
        on_motion_frame: Optional[MotionFrameCallback] = None,
        on_action_progress: Optional[ActionProgressCallback] = None,
    ) -> None:
        _validate_config(config)
        self.config = config
        self._on_motion_frame = on_motion_frame
        self._on_action_progress = on_action_progress
        self._last_motion_frame_s = float("-inf")
        self._client = SlamtecRestClient(
            config.base_url, config.request_timeout_s
        )
        self._camera: Optional[L515Camera] = None
        self._observed_map = L515ObservedMap()

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
        obstacle_map = _to_obstacle_map(slamtec_map)
        if capture is not None:
            obstacle_map = self._observed_map.update(
                obstacle_map,
                pose,
                capture,
                self.config.camera_extrinsics_in_robot,
            )
        return _build_navigation_frame(
            timestamp_s=time.monotonic(),
            pose=pose,
            obstacle_map=obstacle_map,
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
        self._report_action_progress(
            "Hermes command: "
            f"start_pose=({start_pose.x_m:.3f}, {start_pose.y_m:.3f}, "
            f"{math.degrees(start_pose.yaw_rad):.2f}°), "
            f"relative=({command.forward_m:.3f}, "
            f"{command.left_m:.3f}, "
            f"{math.degrees(command.yaw_rad):.2f}°), "
            f"translation={translation:.3f} m, "
            f"target=({target_xy[0]:.3f}, {target_xy[1]:.3f}, "
            f"{math.degrees(target_yaw):.2f}°)"
        )

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
        """监控活跃 Action 的反馈和位姿；不计入 VLM 等待时间。"""
        action_id = self._client.create_action(action_name, options)
        action_label = action_name.rsplit(".", 1)[-1]
        started_s = time.monotonic()
        started_pose = self._client.get_pose()
        monitor_state = _ActionMonitorState(
            started_s=started_s,
            last_sample_s=float("-inf"),
            last_motion_s=started_s,
            last_motion_pose=started_pose,
        )
        self._report_action_progress(
            f"Hermes Action #{action_id} {action_label} 已创建。"
        )

        def monitor_action(status: int, stage: str) -> None:
            self._monitor_action(
                action_id,
                action_label,
                monitor_state,
                status,
                stage,
                action_name == self._move_to_action,
            )

        try:
            self._client.wait_for_action(
                action_id=action_id,
                timeout_s=self.config.action_timeout_s,
                poll_interval_s=self.config.action_poll_interval_s,
                on_poll=monitor_action,
            )
        except BaseException as exc:
            detail = str(exc) or type(exc).__name__
            abort_error: Optional[RuntimeError] = None
            try:
                self._client.abort_current_action()
            except RuntimeError as caught_abort_error:
                abort_error = caught_abort_error

            if (
                abort_error is None
                and action_name == self._rotate_to_action
                and isinstance(exc, _ActionStalledError)
            ):
                pose, target_yaw, yaw_error = self._read_rotation_error(options)
                if yaw_error <= self.config.yaw_tolerance_rad:
                    self._report_action_progress(
                        f"Hermes Action #{action_id} {action_label} 状态停滞，"
                        "但实际朝向已经到达目标，按完成处理："
                        f"pose={math.degrees(pose.yaw_rad):.2f}°，"
                        f"target={math.degrees(target_yaw):.2f}°，"
                        f"error={math.degrees(yaw_error):.2f}°。"
                    )
                    return

            if abort_error is not None:
                detail = f"{detail}；终止 Action 失败：{abort_error}"
            self._report_action_progress(
                f"Hermes Action #{action_id} {action_label} 失败：{detail}"
            )
            if action_name == self._move_to_action and isinstance(
                exc, _ActionStalledError
            ):
                raise MotionStalledError(str(exc)) from exc
            if action_name == self._move_to_action and isinstance(
                exc, SlamtecActionError
            ):
                raise RecoverableMotionError(str(exc)) from exc
            raise
        completion_detail = ""
        if action_name == self._move_to_action:
            final_pose = self._client.get_pose()
            translated_m = math.hypot(
                final_pose.x_m - started_pose.x_m,
                final_pose.y_m - started_pose.y_m,
            )
            completion_detail = (
                f"，final_pose=({final_pose.x_m:.3f}, "
                f"{final_pose.y_m:.3f}, "
                f"{math.degrees(final_pose.yaw_rad):.2f}°)，"
                f"translated={translated_m:.3f} m"
            )
            if translated_m < self.config.action_stall_translation_m:
                self._report_action_progress(
                    "Hermes MoveToAction 已结束但没有有效平移："
                    f"{translated_m:.3f} m"
                )
                raise RecoverableMotionError(
                    "Hermes MoveToAction 已结束，但底盘未产生有效平移："
                    f"{translated_m:.3f} m"
                )
        self._report_action_progress(
            f"Hermes Action #{action_id} {action_label} 完成，"
            f"耗时 {time.monotonic() - started_s:.1f}s"
            f"{completion_detail}。"
        )

    def _read_rotation_error(
        self,
        options: Mapping[str, Any],
    ) -> Tuple[Pose2D, float, float]:
        """读取 RotateToAction 当前朝向与目标朝向的最短角误差。"""
        raw_target = options.get("angle")
        if not _is_finite(raw_target):
            raise RuntimeError("RotateToAction 缺少有限目标角度")
        target_yaw = _wrap_angle(float(raw_target))
        pose = self._client.get_pose()
        return (
            pose,
            target_yaw,
            abs(_angle_difference(target_yaw, pose.yaw_rad)),
        )

    def _monitor_action(
        self,
        action_id: int,
        action_label: str,
        state: _ActionMonitorState,
        status: int,
        stage: str,
        requires_translation: bool,
    ) -> None:
        """采样活跃 Action 位姿，定期报告并识别真正的底盘停滞。"""
        self._publish_motion_frame()
        now = time.monotonic()
        if (
            now - state.last_sample_s
            < self.config.action_progress_interval_s
        ):
            return

        pose = self._client.get_pose()
        moved_m = math.hypot(
            pose.x_m - state.last_motion_pose.x_m,
            pose.y_m - state.last_motion_pose.y_m,
        )
        turned_rad = abs(
            _angle_difference(pose.yaw_rad, state.last_motion_pose.yaw_rad)
        )
        made_progress = moved_m >= self.config.action_stall_translation_m
        if not requires_translation:
            made_progress = (
                made_progress
                or turned_rad >= self.config.action_stall_rotation_rad
            )
        if made_progress:
            state.last_motion_pose = pose
            state.last_motion_s = now

        elapsed_s = now - state.started_s
        still_s = now - state.last_motion_s
        stage_text = stage or "-"
        self._report_action_progress(
            f"Hermes Action #{action_id} {action_label}: "
            f"status={_action_status_text(status)}, "
            f"elapsed={elapsed_s:.1f}s, still={still_s:.1f}s, "
            f"pose=({pose.x_m:.2f}, {pose.y_m:.2f}, "
            f"{math.degrees(pose.yaw_rad):.1f}°), stage={stage_text}"
        )
        state.last_sample_s = now

        if still_s >= self.config.action_stall_timeout_s:
            raise _ActionStalledError(
                f"Hermes Action {action_id} 已连续 {still_s:.1f} 秒"
                "没有产生足够位姿变化"
            )

    def _report_action_progress(self, message: str) -> None:
        """向入口报告底盘 Action 进度；不参与运动控制。"""
        if self._on_action_progress is not None:
            self._on_action_progress(message)

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
    obstacle_map: ObstacleMap,
    capture: Optional[L515Capture],
    camera_extrinsics: CameraExtrinsics,
) -> NavigationFrame:
    """把两类设备数据冻结为 core 只读的同一地图坐标帧。"""
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
        "action_progress_interval_s",
        "action_stall_timeout_s",
        "action_stall_translation_m",
        "action_stall_rotation_rad",
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


def _action_status_text(status: int) -> str:
    return {0: "new", 1: "working", 3: "paused"}.get(status, str(status))


def _is_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


__all__ = ["SlamtecL515Adapter", "SlamtecL515Config"]
