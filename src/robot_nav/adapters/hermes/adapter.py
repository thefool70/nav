"""把 SLAMTEC Hermes 与外接 D435i 组合为统一 ChassisInterface。"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

from ...core.models import (
    CameraExtrinsics,
    NavigationFrame,
    ObstacleMap,
    Pose2D,
    RelativePoseCommand,
)
from ...core.path_validation import UnknownPathMeasurement, measure_unknown_path_length
from ...core.timing import measure_stage
from ..chassis import (
    MotionPathUnknownError,
    MotionBlockedError,
    MotionStalledError,
    RecoverableMotionError,
)
from ..realsense import D435iCapture
from ..realsense.d435i_camera import D435iCamera, D435iConfig
from .front_obstruction import (
    FrontObstruction,
    FrontObstructionDetector,
    FrontObstructionProgress,
)
from .observed_map import HERMES_OBSTACLE_INFLATION_RADIUS_M, HermesObservedMap
from .remote.wire import RgbdPoseCapture
from .rest_client import (
    HermesActionError,
    HermesExploreMap,
    HermesRestClient,
    HermesRobotHealth,
    HermesSlamState,
    resolve_action_name,
)


MotionFrameCallback = Callable[[NavigationFrame], None]
ActionProgressCallback = Callable[[str], None]
MotionPlanCallback = Callable[
    [
        Optional[Tuple[float, float]],
        Tuple[Tuple[float, float], ...],
    ],
    None,
]

# Hermes 地图是无符号字节：0 未知，1..127 自由，128..255 占用。
_OCCUPANCY_VALUES = (None,) + (0.0,) * 127 + (1.0,) * 128


class RgbdCamera(Protocol):
    """Hermes 组帧只要求采集与关闭，允许本地 USB 或远程 RGB-D 来源。"""

    def capture(self, *, after_s: float = 0.0) -> D435iCapture:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class HermesConfig:
    """Hermes REST、可选 D435i 和同步 Action 执行配置。"""

    base_url: str = "http://192.168.11.1:1448"
    camera: Optional[D435iConfig] = field(default_factory=D435iConfig)
    camera_serial: Optional[str] = None
    camera_extrinsics_in_robot: CameraExtrinsics = field(
        default_factory=CameraExtrinsics
    )
    request_timeout_s: float = 5.0
    action_timeout_s: float = 120.0
    action_poll_interval_s: float = 0.2
    action_progress_interval_s: float = 2.0
    action_stall_timeout_s: float = 1.0
    action_stall_translation_m: float = 0.02
    action_stall_rotation_rad: float = math.radians(1.0)
    # 到位后的提前收尾，与是否需要下发微小平移的 position_tolerance_m 分开。
    action_arrival_position_m: float = 0.3
    action_arrival_hold_s: float = 0.001
    max_unknown_path_m: float = 1.5
    motion_frame_interval_s: float = 0.5
    front_blockage_distance_m: float = 0.5
    blocked_pose_radius_m: float = 0.5
    blocked_pose_duration_s: float = 10.0
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
    path_error_reported: bool = False
    unknown_path_length_m: Optional[float] = None
    arrival_since_s: Optional[float] = None
    arrival_pose: Optional[Pose2D] = None


class _ActionArrivedError(RuntimeError):
    """位姿已到达且稳定，交回控制权，由下一 Action 替换当前任务。"""


class _ActionStalledError(RuntimeError):
    """活跃 Hermes Action 长时间没有产生有效位姿变化。"""


class HermesAdapter:
    """由 Hermes 提供地图/位姿/规划控制，由外接 D435i 提供 RGB-D。"""

    def __init__(
        self,
        config: HermesConfig,
        on_motion_frame: Optional[MotionFrameCallback] = None,
        on_continuous_frame: Optional[MotionFrameCallback] = None,
        on_action_progress: Optional[ActionProgressCallback] = None,
        on_motion_plan: Optional[MotionPlanCallback] = None,
        *,
        camera_factory: Callable[[D435iConfig], RgbdCamera] = D435iCamera,
        on_chassis_status: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
    ) -> None:
        _validate_config(config)
        self.config = config
        self._on_motion_frame = on_motion_frame
        self._on_continuous_frame = on_continuous_frame
        self._on_action_progress = on_action_progress
        self._on_motion_plan = on_motion_plan
        self._on_chassis_status = on_chassis_status
        self._last_motion_frame_s = float("-inf")
        self._frame_build_lock = threading.Lock()
        self._map_condition = threading.Condition()
        # 地图线程发布完整转换结果；相机与地图独立更新，不宣称两者同时采集。
        self._latest_map = None
        self._map_updated_s = 0.0
        self._map_thread = None
        self._camera_after_s = 0.0
        self._continuous_frame_stop = threading.Event()
        self._continuous_frame_thread = None
        self._continuous_frame_error = None
        self._client = HermesRestClient(
            config.base_url, config.request_timeout_s, on_status=on_chassis_status
        )
        self._active_action_id: Optional[int] = None
        self._action_pending = False
        self._camera: Optional[RgbdCamera] = None
        self._observed_map = HermesObservedMap()
        self._front_obstruction_detector = FrontObstructionDetector(
            maximum_depth_m=config.front_blockage_distance_m,
            movement_radius_m=config.blocked_pose_radius_m,
            duration_s=config.blocked_pose_duration_s,
        )
        self._front_obstruction_lock = threading.Lock()
        self._pending_front_obstruction: Optional[FrontObstruction] = None
        self._front_obstruction_progress_started_s: Optional[float] = None
        self._front_obstruction_progress_second = -1

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
                camera_config = config.camera
                if config.camera_serial is not None:
                    camera_config = replace(
                        camera_config, serial_number=config.camera_serial
                    )
                self._camera = camera_factory(camera_config)
            self._map_thread = threading.Thread(target=self._map_loop, name="hermes-map", daemon=True)
            self._map_thread.start()
            # 前向挡路检测依赖运动期间连续 RGB-D；是否开启 Rerun 不改变检测行为。
            if self._camera is not None:
                self._continuous_frame_thread = threading.Thread(
                    target=self._continuous_frame_loop, name="hermes-frames", daemon=True)
                self._continuous_frame_thread.start()
        except Exception:
            self.close()
            raise

    @property
    def has_camera(self) -> bool:
        """当前 Adapter 是否启用了外接 D435i。"""
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

    def get_slam_state(self) -> HermesSlamState:
        """读取建图/定位模式及其质量。"""
        return self._client.get_slam_state()

    def get_robot_health(self) -> HermesRobotHealth:
        """读取底盘健康摘要。"""
        return self._client.get_robot_health()

    def read_pose(self) -> Pose2D:
        """读取 Hermes 当前地图位姿，供标定控制复用。"""
        return self._client.get_pose()

    def read_frame(self) -> NavigationFrame:
        """组合 Hermes 地图/位姿与同机 D435i 对齐 RGB-D。"""
        self._raise_continuous_frame_error()
        return self._capture_frame()

    def stop(self) -> None:
        """没有下一动作或需要原地处理时，取消残留 Action 并确认终态。"""
        if self._action_pending:
            try:
                self._cancel_active_action()
            finally:
                self._camera_after_s = time.monotonic()

    def send_relative_pose(self, command: RelativePoseCommand) -> None:
        """执行普通运动，用于启动、标定、转向和相对位姿运动。"""
        try:
            self._send_relative_pose(command)
        finally:
            self._camera_after_s = time.monotonic()

    def send_relative_pose_in_known_space(
        self, command: RelativePoseCommand, obstacle_map: ObstacleMap,
        *, reference_pose: Pose2D,
    ) -> None:
        """按决策位姿固定探索／回退的世界目标，用决策地图约束实际路径。"""
        try:
            self._send_relative_pose(
                command, known_space_map=obstacle_map, reference_pose=reference_pose,
            )
        finally:
            self._camera_after_s = time.monotonic()

    def close(self) -> None:
        """退出先取消遗留动作，再关闭采集；取消失败仍显式上报。"""
        cancellation_error = None
        try:
            self._cancel_active_action()
        except RuntimeError as exc:
            cancellation_error = exc
        self._continuous_frame_stop.set()
        with self._map_condition:
            self._map_condition.notify_all()
        frame_thread = self._continuous_frame_thread
        self._continuous_frame_thread = None
        if frame_thread is not None and frame_thread is not threading.current_thread():
            camera_timeout_s = (
                self.config.camera.wait_timeout_s
                if self.config.camera is not None
                else 0.0
            )
            frame_thread.join(
                timeout=min(
                    30.0,
                    2.0 * self.config.request_timeout_s
                    + camera_timeout_s
                    + 2.0,
                )
            )
        if self._map_thread is not None:
            self._map_thread.join()
            self._map_thread = None
        self._on_continuous_frame = None
        self._on_motion_frame = None
        camera = self._camera
        self._camera = None
        if camera is not None:
            camera.close()
        if cancellation_error is not None:
            raise RuntimeError(f"退出时取消 Hermes Action 失败：{cancellation_error}") from cancellation_error

    def __enter__(self) -> "HermesAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def _capture_frame(self) -> NavigationFrame:
        """读取最新相机包和地图快照；串行更新观察图，避免并发改写历史。"""
        timings = []
        waiting = time.monotonic()
        with self._frame_build_lock:
            acquired = time.monotonic()
            timings.append({"stage": "frame.build_lock_wait", "started_monotonic_s": waiting,
                            "ended_monotonic_s": acquired, "duration_s": acquired - waiting,
                            "completed": True})
            with measure_stage(timings, "frame.camera_capture"):
                capture = (
                    self._camera.capture(after_s=self._camera_after_s)
                    if self._camera is not None
                    else None
                )
            if isinstance(capture, RgbdPoseCapture):
                pose = capture.pose
            else:
                with measure_stage(timings, "frame.get_pose"):
                    pose = self._client.get_pose()
            with measure_stage(timings, "frame.map_snapshot"):
                with self._map_condition:
                    self._map_condition.wait_for(lambda: (
                        self._latest_map is not None or self._continuous_frame_error is not None
                        or self._continuous_frame_stop.is_set()
                    ))
                    self._raise_continuous_frame_error()
                    if self._continuous_frame_stop.is_set():
                        raise RuntimeError("Hermes 数据采集已关闭")
                    if time.monotonic() - self._map_updated_s > (
                        self.config.request_timeout_s + self.config.motion_frame_interval_s
                    ):
                        raise RuntimeError("Hermes 地图更新超时")
                    navigation_map = self._latest_map
                    map_age_s = time.monotonic() - self._map_updated_s
            timings[-1]["map_age_s"] = map_age_s
            # 完整图用于物体定位与停靠；探索图和视觉图随后按相机 FOV 限制公开范围。
            obstacle_map = navigation_map
            visibility_map = None
            if capture is not None:
                with measure_stage(timings, "frame.update_observed_map"):
                    obstacle_map, visibility_map = self._observed_map.update(
                        obstacle_map,
                        pose,
                        capture,
                        self.config.camera_extrinsics_in_robot,
                    )
            timestamp_s = capture.timestamp_s if isinstance(capture, RgbdPoseCapture) else time.monotonic()
        # 相机包和地图快照已固定；RGB-D 转换不改历史图，无需继续占用采集锁。
        with measure_stage(timings, "frame.build"):
            frame = _build_navigation_frame(
                timestamp_s=timestamp_s,
                pose=pose,
                obstacle_map=obstacle_map,
                capture=capture,
                camera_extrinsics=self.config.camera_extrinsics_in_robot,
                navigation_map=navigation_map,
                visibility_map=visibility_map,
                timings=timings,
            )
        return replace(frame, acquisition_timings=tuple(timings))

    def _map_loop(self) -> None:
        """独立请求并转换完整地图，发布成功结果；失败不沿用旧图掩盖故障。"""
        while not self._continuous_frame_stop.is_set():
            try:
                navigation_map = _to_obstacle_map(self._client.get_explore_map())
                with self._map_condition:
                    self._latest_map = navigation_map
                    self._map_updated_s = time.monotonic()
                    self._map_condition.notify_all()
            except BaseException as exc:
                with self._map_condition:
                    self._continuous_frame_error = exc
                    self._map_condition.notify_all()
                return
            self._continuous_frame_stop.wait(self.config.motion_frame_interval_s)

    def _continuous_frame_loop(self) -> None:
        """持续组帧供挡路检测使用，并在启用时同时交给可视化。"""
        callback = self._on_continuous_frame
        while not self._continuous_frame_stop.is_set():
            try:
                if self._on_chassis_status is not None:
                    try:
                        self._client.get_robot_health()
                    except RuntimeError as exc:
                        self._on_chassis_status("health", {"read_error": str(exc)})
                frame = self._capture_frame()
                obstruction = self._front_obstruction_detector.observe(frame)
                self._report_front_obstruction_progress(
                    self._front_obstruction_detector.progress()
                )
                if obstruction is not None:
                    wall_map = frame.visibility_map or frame.obstacle_map
                    with self._frame_build_lock:
                        wall_cell_count = self._observed_map.add_permanent_wall(
                            wall_map,
                            obstruction.center_world_xy,
                            obstruction.heading_world_rad,
                        )
                    with self._front_obstruction_lock:
                        self._pending_front_obstruction = obstruction
                    depth_detail = (
                        f"depth={obstruction.depth_m:.3f} m，"
                        if obstruction.depth_m is not None
                        else "depth=未用于定位，"
                    )
                    self._report_action_progress(
                        "底盘持续停留在阻塞半径内："
                        f"{depth_detail}"
                        f"duration={obstruction.duration_s:.1f}s，"
                        f"samples={obstruction.sample_count}；"
                        f"已在 ({obstruction.center_world_xy[0]:.3f}, "
                        f"{obstruction.center_world_xy[1]:.3f}) 横向建立永久人工墙，"
                        f"新增 {wall_cell_count} 个原始占用格。"
                    )
                if callback is not None:
                    callback(frame)
            except BaseException as exc:
                with self._map_condition:
                    self._continuous_frame_error = exc
                    self._map_condition.notify_all()
                self._report_action_progress(
                    f"Hermes/D435i 连续视觉帧停止：{str(exc) or type(exc).__name__}"
                )
                return
            self._continuous_frame_stop.wait(self.config.motion_frame_interval_s)

    def _report_front_obstruction_progress(
        self,
        progress: Optional[FrontObstructionProgress],
    ) -> None:
        """每秒记录一次候选计时，让未触发原因能从 Action 日志直接看出。"""
        if progress is None:
            return
        elapsed_second = int(progress.duration_s)
        with self._front_obstruction_lock:
            new_candidate = (
                progress.started_s != self._front_obstruction_progress_started_s
            )
            if not new_candidate and elapsed_second <= self._front_obstruction_progress_second:
                return
            self._front_obstruction_progress_started_s = progress.started_s
            self._front_obstruction_progress_second = elapsed_second
        prefix = "开始底盘阻塞计时" if new_candidate else "底盘阻塞计时"
        depth_detail = (
            f"{progress.depth_m:.3f} m"
            if progress.depth_m is not None
            else "未检测到可靠近深度"
        )
        restart = (
            f"，重新计时原因={progress.restart_reason}"
            if new_candidate and progress.restart_reason
            else ""
        )
        self._report_action_progress(
            f"{prefix}：duration={progress.duration_s:.1f}s，"
            f"depth={depth_detail}，"
            f"translation={progress.displacement_m:.3f} m，"
            f"samples={progress.sample_count}{restart}。"
        )

    def _raise_continuous_frame_error(self) -> None:
        """把后台设备错误带回主循环，而不是静默停止采样。"""
        error = self._continuous_frame_error
        if error is None:
            return
        raise RuntimeError(
            "Hermes/D435i 数据采集失败："
            f"{str(error) or type(error).__name__}"
        ) from error

    def _send_relative_pose(
        self,
        command: RelativePoseCommand,
        known_space_map: Optional[ObstacleMap] = None,
        reference_pose: Optional[Pose2D] = None,
    ) -> None:
        """把机器人局部相对位姿转换为 Hermes 的全局规划与原地转向。"""
        _validate_command(command)
        dispatch_timings = []
        with measure_stage(dispatch_timings, "motion.ready_check"):
            self._require_motion_ready()
        with measure_stage(dispatch_timings, "motion.start_pose"):
            start_pose = self._client.get_pose()
        # 相对命令基于决策帧，不能再用运动后的朝向旋转一次，否则世界目标会漂移。
        reference_pose = reference_pose if reference_pose is not None else start_pose
        target_xy = _relative_target_world(reference_pose, command)
        target_yaw = _wrap_angle(reference_pose.yaw_rad + command.yaw_rad)
        translation = math.hypot(target_xy[0] - start_pose.x_m, target_xy[1] - start_pose.y_m)
        self._report_action_progress(
            "Hermes command: "
            f"start_pose=({start_pose.x_m:.3f}, {start_pose.y_m:.3f}, "
            f"{math.degrees(start_pose.yaw_rad):.2f}°), "
            f"reference_pose=({reference_pose.x_m:.3f}, {reference_pose.y_m:.3f}, "
            f"{math.degrees(reference_pose.yaw_rad):.2f}°), "
            f"relative=({command.forward_m:.3f}, "
            f"{command.left_m:.3f}, "
            f"{math.degrees(command.yaw_rad):.2f}°), "
            f"translation={translation:.3f} m, "
            f"target=({target_xy[0]:.3f}, {target_xy[1]:.3f}, "
            f"{math.degrees(target_yaw):.2f}°)"
        )

        sent_action = False
        if translation > self.config.position_tolerance_m:
            self._execute_action(
                self._move_to_action,
                {"target": {"x": target_xy[0], "y": target_xy[1], "z": 0.0}},
                target_world_xy=target_xy,
                known_space_map=known_space_map,
                dispatch_timings=dispatch_timings,
            )
            sent_action = True
            dispatch_timings = []

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
                    dispatch_timings=dispatch_timings,
                )
                sent_action = True
        if not sent_action:
            self.stop()  # 零位移命令没有新 Action 可替换旧任务。
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
        self,
        action_name: str,
        options: Mapping[str, Any],
        target_world_xy: Optional[Tuple[float, float]] = None,
        known_space_map: Optional[ObstacleMap] = None,
        dispatch_timings=None,
    ) -> None:
        """在平移动作生命周期内启停前向深度挡路检测。"""
        detect_front_obstruction = action_name == self._move_to_action
        if detect_front_obstruction:
            with self._front_obstruction_lock:
                self._pending_front_obstruction = None
                self._front_obstruction_progress_started_s = None
                self._front_obstruction_progress_second = -1
            self._front_obstruction_detector.start_translation(target_world_xy)
        try:
            return self._execute_monitored_action(
                action_name,
                options,
                target_world_xy=target_world_xy,
                known_space_map=known_space_map,
                dispatch_timings=dispatch_timings,
            )
        finally:
            if detect_front_obstruction:
                self._front_obstruction_detector.stop_translation()

    def _execute_monitored_action(
        self,
        action_name: str,
        options: Mapping[str, Any],
        target_world_xy: Optional[Tuple[float, float]] = None,
        known_space_map: Optional[ObstacleMap] = None,
        dispatch_timings=None,
    ) -> None:
        """监控活跃 Action 的反馈和位姿；不计入 VLM 等待时间。"""
        action_label = action_name.rsplit(".", 1)[-1]
        target_yaw = float(options["angle"]) if action_name == self._rotate_to_action else None
        action_id = None
        if dispatch_timings is None:
            dispatch_timings = []
        self._action_pending = True
        try:
            previous_action_id = self._active_action_id
            # POST 失败时无法确定新任务是否被受理，异常路径必须取消 :current。
            self._active_action_id = None
            with measure_stage(dispatch_timings, "motion.create_action"):
                action_id = self._client.create_action(action_name, options)
            self._active_action_id = action_id
            started_s = time.monotonic()
            with measure_stage(dispatch_timings, "motion.monitor_pose"):
                started_pose = self._client.get_pose()
            monitor_state = _ActionMonitorState(
                started_s=started_s, last_sample_s=float("-inf"),
                last_motion_s=started_s, last_motion_pose=started_pose,
            )
            self._report_action_progress(f"Hermes Action #{action_id} {action_label} 已创建。")
            self._report_action_progress(
                f"Hermes Action #{action_id} 下发计时（秒）："
                + ", ".join(f"{span['stage']}={span['duration_s']:.4f}" for span in dispatch_timings)
            )
            if previous_action_id is not None:
                self._report_action_progress(f"Hermes Action #{action_id} 替换到位的 Action #{previous_action_id}。")
            self._report_motion_plan(target_world_xy, ())

            def monitor_action(status: int, stage: str) -> Optional[float]:
                self._monitor_action(
                    action_id, action_label, monitor_state, status, stage,
                    action_name == self._move_to_action, target_world_xy, known_space_map,
                    target_yaw,
                )
                if monitor_state.arrival_since_s is not None:
                    return max(0.0, self.config.action_arrival_hold_s
                               - (time.monotonic() - monitor_state.arrival_since_s))

            try:
                self._client.wait_for_action(
                    action_id=action_id,
                    timeout_s=self.config.action_timeout_s,
                    poll_interval_s=self.config.action_poll_interval_s,
                    on_poll=monitor_action,
                )
            except _ActionArrivedError as arrived:
                # Hermes 创建新 Action 会替换旧任务；保留 ID，供替换或停止时收尾。
                self._report_action_progress(
                    f"Hermes Action #{action_id} {action_label} 按位姿确认到达，"
                    f"保留任务等待下一动作替换：{arrived}"
                )
            else:
                self._active_action_id = None
                self._action_pending = False
        except BaseException as exc:
            detail = str(exc) or type(exc).__name__
            abort_error: Optional[RuntimeError] = None
            try:
                self._cancel_active_action()
            except RuntimeError as caught_abort_error:
                abort_error = caught_abort_error
            self._report_motion_plan(None, ())

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
                detail = f"{detail}；终止 Action 或确认结束失败：{abort_error}"
            self._report_action_progress(
                f"Hermes Action #{action_id} {action_label} 失败：{detail}"
            )
            if abort_error is not None:
                raise RuntimeError(detail) from abort_error
            if isinstance(exc, MotionPathUnknownError):
                # 保留异常类型以触发整片屏蔽，路径仅用于记录取消原因。
                raise
            if isinstance(exc, _ActionStalledError):
                raise MotionStalledError(str(exc)) from exc
            if isinstance(exc, HermesActionError):
                raise RecoverableMotionError(str(exc)) from exc
            raise
        if not self._action_pending:
            self._report_motion_plan(None, ())
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
            if target_world_xy is not None:
                target_error_m = math.hypot(
                    final_pose.x_m - target_world_xy[0], final_pose.y_m - target_world_xy[1],
                )
                completion_detail += f"，target_error={target_error_m:.3f} m"
            if translated_m < self.config.action_stall_translation_m:
                self._report_action_progress(
                    "Hermes MoveToAction 已结束但没有有效平移："
                    f"{translated_m:.3f} m"
                )
                raise RecoverableMotionError(
                    "Hermes MoveToAction 已结束，但底盘未产生有效平移："
                    f"{translated_m:.3f} m"
                )
        outcome = "到位交接" if self._action_pending else "完成"
        self._report_action_progress(
            f"Hermes Action #{action_id} {action_label} {outcome}，"
            f"耗时 {time.monotonic() - started_s:.1f}s"
            f"{completion_detail}。"
        )

    def _cancel_active_action(self) -> None:
        """Action 创建后的所有异常都取消；取得 ID 时还要确认终态。"""
        if not self._action_pending:
            return
        self._client.abort_current_action()
        if self._active_action_id is not None:
            self._client.wait_for_action(
                action_id=self._active_action_id,
                timeout_s=self.config.request_timeout_s,
                poll_interval_s=self.config.action_poll_interval_s,
                require_success=False,
            )
        self._active_action_id = None
        self._action_pending = False

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
        target_world_xy: Optional[Tuple[float, float]],
        known_space_map: Optional[ObstacleMap] = None,
        target_yaw: Optional[float] = None,
    ) -> None:
        """每次轮询判断到位与停滞，仅终端输出按进度间隔节流。"""
        self._raise_continuous_frame_error()
        if requires_translation:
            self._raise_front_obstruction(action_id)
        pose = self._client.get_pose()
        now = time.monotonic()
        if requires_translation and known_space_map is not None:
            measurement = self._check_known_space_path(action_id, target_world_xy, known_space_map, pose)
            state.unknown_path_length_m = measurement.unknown_length_m if measurement is not None else None
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

        self._check_stable_arrival(
            state, status, pose, now, target_world_xy, target_yaw,
            requires_translation,
        )
        still_s = now - state.last_motion_s
        if not requires_translation and still_s >= self.config.action_stall_timeout_s:
            raise _ActionStalledError(
                f"Hermes Action {action_id} 已连续 {still_s:.1f} 秒"
                "没有产生足够位姿变化"
            )
        self._publish_motion_frame()
        if now - state.last_sample_s < self.config.action_progress_interval_s:
            return

        if (
            requires_translation and known_space_map is None
            and self._on_motion_plan is not None
        ):
            remaining_path = self._read_remaining_path(state)
            self._report_motion_plan(
                target_world_xy,
                remaining_path,
            )

        elapsed_s = now - state.started_s
        stage_text = stage or "-"
        unknown_detail = (
            f", unknown_path={state.unknown_path_length_m:.3f}/{self.config.max_unknown_path_m:.3f} m"
            if state.unknown_path_length_m is not None else ""
        )
        self._report_action_progress(
            f"Hermes Action #{action_id} {action_label}: "
            f"status={_action_status_text(status)}, "
            f"elapsed={elapsed_s:.1f}s, still={still_s:.1f}s, "
            f"pose=({pose.x_m:.2f}, {pose.y_m:.2f}, "
            f"{math.degrees(pose.yaw_rad):.1f}°), stage={stage_text}{unknown_detail}"
        )
        state.last_sample_s = now

    def _raise_front_obstruction(self, action_id: int) -> None:
        """人工墙写入完成后取消当前动作，让核心从新地图重新决策。"""
        with self._front_obstruction_lock:
            obstruction = self._pending_front_obstruction
        if obstruction is None:
            return
        raise MotionBlockedError(
            f"Hermes Action {action_id} 的底盘已在"
            f" {self.config.blocked_pose_radius_m:.3f} m 范围内停留"
            f" {obstruction.duration_s:.1f}s，人工墙已建立，取消本次移动"
        )

    def _pose_at_action_target(
        self, pose: Pose2D, target_xy: Optional[Tuple[float, float]], target_yaw: Optional[float],
    ) -> bool:
        """MoveTo 只检查停靠位置，RotateTo 只检查朝向；两者仍按顺序执行。"""
        if target_xy is not None:
            return math.hypot(pose.x_m - target_xy[0], pose.y_m - target_xy[1]) <= self.config.action_arrival_position_m
        if target_yaw is not None:
            return abs(_angle_difference(target_yaw, pose.yaw_rad)) <= self.config.yaw_tolerance_rad
        return False

    def _check_stable_arrival(
        self, state: _ActionMonitorState, status: int, pose: Pose2D, now: float,
        target_xy: Optional[Tuple[float, float]], target_yaw: Optional[float], requires_translation: bool,
    ) -> None:
        """目标容差内稳定达到配置时长后收尾；实际确认精度受轮询间隔限制。"""
        if (
            status != 1
            or (requires_translation and state.last_motion_s <= state.started_s)
            or not self._pose_at_action_target(pose, target_xy, target_yaw)
        ):
            state.arrival_since_s = None
            state.arrival_pose = None
            return
        anchor = state.arrival_pose
        if (
            anchor is None
            or math.hypot(pose.x_m - anchor.x_m, pose.y_m - anchor.y_m) >= self.config.action_stall_translation_m
            or abs(_angle_difference(pose.yaw_rad, anchor.yaw_rad)) >= self.config.action_stall_rotation_rad
        ):
            state.arrival_since_s = now
            state.arrival_pose = pose
            return
        if state.arrival_since_s is not None and now - state.arrival_since_s >= self.config.action_arrival_hold_s:
            if target_xy is not None:
                error = f"位置误差 {math.hypot(pose.x_m - target_xy[0], pose.y_m - target_xy[1]):.3f} m"
            else:
                error = f"角度误差 {math.degrees(abs(_angle_difference(target_yaw, pose.yaw_rad))):.2f}°"
            raise _ActionArrivedError(f"{error}，稳定 {now - state.arrival_since_s:.3f}s，阈值 {self.config.action_arrival_hold_s:.3f}s")

    def _check_known_space_path(
        self,
        action_id: int,
        target_world_xy: Optional[Tuple[float, float]],
        obstacle_map: ObstacleMap,
        pose: Pose2D,
    ) -> Optional[UnknownPathMeasurement]:
        """每次轮询独立计算当前位置及剩余路径的未知长度，超过配置上限才取消。"""
        # 检查所需路径读取失败属于系统错误，不能降级为仅隐藏可视化。
        remaining_path = self._client.get_remaining_path()
        self._report_motion_plan(target_world_xy, remaining_path)
        if not remaining_path:
            # 规划尚未发布路径时继续等待，不能据此判定目标不可达。
            return
        path_world_xy = ((pose.x_m, pose.y_m),) + remaining_path
        with self._frame_build_lock:
            wall_crossing = self._observed_map.first_artificial_wall_crossing(
                path_world_xy
            )
        if wall_crossing is not None:
            raise MotionBlockedError(
                f"Hermes Action {action_id} 的剩余路径将穿过永久人工墙："
                f"首次交点 ({wall_crossing[0]:.3f}, {wall_crossing[1]:.3f})，"
                "取消本次移动"
            )
        measurement = measure_unknown_path_length(path_world_xy, obstacle_map)
        limit_m = self.config.max_unknown_path_m
        # 只吸收纳米量级的浮点误差；恰好达到上限时仍允许继续。
        if measurement.unknown_length_m > limit_m + 1e-9:
            raise MotionPathUnknownError(
                f"Hermes Action {action_id} 的剩余路径未知长度超限："
                f"未知区域及地图外累计 {measurement.unknown_length_m:.3f} m，"
                f"允许上限 {limit_m:.3f} m，路径总长 {measurement.total_length_m:.3f} m，"
                f"首个未知格(row, col)={measurement.first_unknown_cell}。",
                path_world_xy=path_world_xy,
                unknown_length_m=measurement.unknown_length_m,
                limit_m=limit_m,
                total_path_length_m=measurement.total_length_m,
            )
        return measurement

    def _read_remaining_path(
        self,
        state: _ActionMonitorState,
    ) -> Tuple[Tuple[float, float], ...]:
        """尽力读取底盘路径；可视化接口异常不影响运动 Action。"""
        if state.path_error_reported:
            return ()
        try:
            return self._client.get_remaining_path()
        except RuntimeError as exc:
            if not state.path_error_reported:
                state.path_error_reported = True
                self._report_action_progress(
                    f"Hermes 剩余路径暂不可用，仅隐藏路径可视化：{exc}"
                )
            return ()

    def _report_motion_plan(
        self,
        target_world_xy: Optional[Tuple[float, float]],
        remaining_path_world_xy: Tuple[Tuple[float, float], ...],
    ) -> None:
        """发布底盘实际目标和剩余路径；可视化故障由入口统一处理。"""
        if self._on_motion_plan is not None:
            self._on_motion_plan(target_world_xy, remaining_path_world_xy)

    def _report_action_progress(self, message: str) -> None:
        """向入口报告底盘 Action 进度；不参与运动控制。"""
        if self._on_action_progress is not None:
            self._on_action_progress(message)

    def _publish_motion_frame(self, force: bool = False) -> None:
        """运动期间按固定间隔向 Rerun 发布真实底盘帧。"""
        if self._on_continuous_frame is not None:
            return
        if self._on_motion_frame is None:
            return
        now = time.monotonic()
        elapsed_s = now - self._last_motion_frame_s
        if not force and elapsed_s < self.config.motion_frame_interval_s:
            return
        self._on_motion_frame(self.read_frame())
        self._last_motion_frame_s = now


def _build_navigation_frame(
    timestamp_s: float,
    pose: Pose2D,
    obstacle_map: ObstacleMap,
    capture: Optional[D435iCapture],
    camera_extrinsics: CameraExtrinsics,
    navigation_map: Optional[ObstacleMap] = None,
    visibility_map: Optional[ObstacleMap] = None,
    *, timings=None,
) -> NavigationFrame:
    """把两类设备数据冻结为 core 只读的同一地图坐标帧。"""
    if capture is None:
        return NavigationFrame(
            timestamp_s=timestamp_s,
            pose=pose,
            obstacle_map=obstacle_map,
            camera_extrinsics_in_robot=camera_extrinsics,
            navigation_map=navigation_map,
            navigation_clearance_m=HERMES_OBSTACLE_INFLATION_RADIUS_M,
            visibility_map=visibility_map,
        )
    with measure_stage(timings, "frame.convert_depth"):
        depth = _convert_depth(capture.depth_m)
    with measure_stage(timings, "frame.convert_rgb"):
        rgb = _convert_rgb(capture.rgb)
    return NavigationFrame(
        timestamp_s=timestamp_s,
        pose=pose,
        obstacle_map=obstacle_map,
        depth=depth,
        rgb=rgb,
        camera_intrinsics=capture.camera_intrinsics,
        camera_extrinsics_in_robot=camera_extrinsics,
        navigation_map=navigation_map,
        navigation_clearance_m=HERMES_OBSTACLE_INFLATION_RADIUS_M,
        visibility_map=visibility_map,
    )


def _to_obstacle_map(source: HermesExploreMap) -> ObstacleMap:
    """转换 Hermes 6.3 栅格值和边界原点为项目统一占用图。"""
    rows = []
    for row_index in range(source.height):
        offset = row_index * source.width
        raw_row = source.cells[offset : offset + source.width]
        rows.append(tuple(map(_OCCUPANCY_VALUES.__getitem__, raw_row)))
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


def _convert_rgb(image: Any):
    """相机边界已统一为 H×W×3 uint8 数组；转成核心使用的只读行序列。"""
    return tuple(tuple(map(tuple, row)) for row in image.tolist())


def _convert_depth(image: Any):
    """相机输出米制浮点数组；传感器无效深度在此转成 None。"""
    import numpy as np

    values = image.astype(object)
    values[~(np.isfinite(image) & (image > 0.0))] = None
    return tuple(map(tuple, values.tolist()))


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


def _validate_config(config: HermesConfig) -> None:
    if not isinstance(config, HermesConfig):
        raise ValueError("config 必须为 HermesConfig")
    if config.camera is not None and not isinstance(config.camera, D435iConfig):
        raise ValueError("camera 必须为 D435iConfig 或 None")
    if config.camera_serial is not None and not isinstance(config.camera_serial, str):
        raise ValueError("camera_serial 必须为字符串或 None")
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
        "action_arrival_position_m",
        "action_arrival_hold_s",
        "motion_frame_interval_s",
        "front_blockage_distance_m",
        "blocked_pose_radius_m",
        "blocked_pose_duration_s",
        "position_tolerance_m",
        "yaw_tolerance_rad",
    ):
        value = getattr(config, name)
        if not _is_finite(value) or float(value) <= 0.0:
            raise ValueError(f"{name} 必须为正有限数")
    quality = config.minimum_localization_quality
    if not _is_finite(config.max_unknown_path_m) or config.max_unknown_path_m < 0.0:
        raise ValueError("max_unknown_path_m 必须为非负有限米数")
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


__all__ = ["HermesAdapter", "HermesConfig"]
