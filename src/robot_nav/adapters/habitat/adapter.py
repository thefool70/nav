"""把 Habitat-Sim 转换为项目统一底盘接口。

Habitat 使用 ``x/y/z`` 三维坐标，项目内部使用二维 ``x/y``：内部 ``x`` 等于
Habitat ``x``，内部 ``y`` 等于 Habitat ``-z``。本模块是唯一知道这项换算的
位置，核心算法不依赖 Habitat。
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from ..chassis import RecoverableMotionError
from ...core.models import (
    CameraIntrinsics,
    NavigationFrame,
    ObstacleMap,
    Pose2D,
    RelativePoseCommand,
)


MotionFrameCallback = Callable[[NavigationFrame], None]
MotionPlanCallback = Callable[
    [
        Optional[Tuple[float, float]],
        Tuple[Tuple[float, float], ...],
    ],
    None,
]


@dataclass(frozen=True)
class HabitatConfig:
    """Habitat 场景、相机和局部障碍图配置。"""

    scene_path: str
    width: int = 320
    height: int = 240
    hfov_deg: float = 90.0
    sensor_height_m: float = 1.0
    map_resolution_m: float = 0.10
    observed_range_m: float = 5.0
    seed: int = 1
    gpu_device_id: int = -1


class HabitatChassisAdapter:
    """同步 Habitat 底盘实现：读取感知并用原生动作执行高层导航。"""

    def __init__(
        self,
        config: HabitatConfig,
        on_motion_frame: Optional[MotionFrameCallback] = None,
        on_motion_plan: Optional[MotionPlanCallback] = None,
    ) -> None:
        self._validate_config(config)
        self.config = config
        self._on_motion_frame = on_motion_frame
        self._on_motion_plan = on_motion_plan
        self._habitat_sim = self._import_habitat_sim()
        self._sim = None
        self._agent = None
        self._pathfinder = None
        try:
            self._sim = self._create_simulator()
            self._agent = self._sim.get_agent(0)
            self._pathfinder = self._sim.pathfinder
            if not self._pathfinder.is_loaded:
                raise RuntimeError("Habitat 场景没有加载可用的 navmesh")
            self._pathfinder.seed(config.seed)
            self._place_agent_at_random_point()
            self._initialize_obstacle_map()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _import_habitat_sim() -> Any:
        """延迟导入 Habitat，使普通核心环境无需安装仿真依赖。"""
        try:
            return importlib.import_module("habitat_sim")
        except ImportError as exc:
            raise RuntimeError(
                "当前 Python 环境未安装 habitat-sim，请激活 robot-nav-habitat"
            ) from exc

    @staticmethod
    def _validate_config(config: HabitatConfig) -> None:
        """在创建昂贵的 Simulator 前校验基础配置。"""
        if not isinstance(config, HabitatConfig):
            raise ValueError("config 必须为 HabitatConfig")
        if not Path(config.scene_path).is_file():
            raise ValueError(f"Habitat 场景不存在：{config.scene_path}")
        if (
            isinstance(config.width, bool)
            or isinstance(config.height, bool)
            or not isinstance(config.width, int)
            or not isinstance(config.height, int)
            or config.width <= 0
            or config.height <= 0
        ):
            raise ValueError("相机宽高必须为正整数")
        numeric_values = (
            config.hfov_deg,
            config.sensor_height_m,
            config.map_resolution_m,
            config.observed_range_m,
        )
        if not all(_is_finite(value) and float(value) > 0.0 for value in numeric_values):
            raise ValueError("相机和地图参数必须为正有限值")
        if not 0.0 < config.hfov_deg < 180.0:
            raise ValueError("hfov_deg 必须位于 (0, 180)")
        if (
            isinstance(config.seed, bool)
            or not isinstance(config.seed, int)
            or isinstance(config.gpu_device_id, bool)
            or not isinstance(config.gpu_device_id, int)
            or config.gpu_device_id < -1
        ):
            raise ValueError("seed 和 gpu_device_id 必须为有效整数")

    def _create_simulator(self) -> Any:
        """创建带有对齐 RGB 与深度相机的 Habitat Simulator。"""
        habitat_sim = self._habitat_sim
        simulator_config = habitat_sim.SimulatorConfiguration()
        simulator_config.scene_id = str(Path(self.config.scene_path).resolve())
        simulator_config.gpu_device_id = self.config.gpu_device_id

        color_sensor = self._camera_sensor_spec(
            "color_sensor", habitat_sim.SensorType.COLOR
        )
        depth_sensor = self._camera_sensor_spec(
            "depth_sensor", habitat_sim.SensorType.DEPTH
        )
        depth_sensor.channels = 1

        agent_config = habitat_sim.agent.AgentConfiguration()
        agent_config.height = self.config.sensor_height_m
        agent_config.sensor_specifications = [color_sensor, depth_sensor]
        configuration = habitat_sim.Configuration(simulator_config, [agent_config])
        return habitat_sim.Simulator(configuration)

    def _camera_sensor_spec(self, uuid: str, sensor_type: Any) -> Any:
        """构造一个与机器人前向对齐的针孔相机。"""
        spec = self._habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.sensor_subtype = self._habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [self.config.height, self.config.width]
        spec.position = [0.0, self.config.sensor_height_m, 0.0]
        spec.hfov = self.config.hfov_deg
        return spec

    def _place_agent_at_random_point(self) -> None:
        """使用固定随机种子选择起点，并把初始朝向设为 Habitat 默认前向。"""
        state = self._agent.get_state()
        position = self._pathfinder.get_random_navigable_point()
        if not all(_is_finite(position[index]) for index in range(3)):
            raise RuntimeError("Habitat navmesh 无法生成有效起点")
        state.position = position
        state.rotation = state.rotation.__class__(1.0, 0.0, 0.0, 0.0)
        self._agent.set_state(state)

    def _initialize_obstacle_map(self) -> None:
        """保存当前楼层完整 navmesh，并初始化尚未观察的掩码。"""
        state = self._agent.get_state()
        topdown = self._pathfinder.get_topdown_view(
            self.config.map_resolution_m, float(state.position[1])
        )
        raw_rows = topdown.tolist()
        if not raw_rows or not raw_rows[0]:
            raise RuntimeError("Habitat navmesh 无法生成二维障碍图")
        # Habitat 的数组行从 min-z 向 max-z 增长；内部 y=-z 且栅格行应沿
        # +y 增长，因此在 Adapter 边界翻转一次，core 不再感知这项差异。
        self._navigable = tuple(
            tuple(bool(value) for value in row) for row in reversed(raw_rows)
        )
        self._observed = [
            [False for _ in range(len(self._navigable[0]))]
            for _ in range(len(self._navigable))
        ]

        minimum, _ = self._pathfinder.get_bounds()
        last_sample_z = float(minimum[2]) + (
            len(self._navigable) - 1
        ) * self.config.map_resolution_m
        self._map_origin = Pose2D(
            x_m=float(minimum[0]),
            y_m=-last_sample_z,
            yaw_rad=0.0,
        )

    def read_frame(self) -> NavigationFrame:
        """读取当前同步 RGB-D，并生成仅暴露局部已知区域的障碍图。"""
        self._require_open()
        observations = self._sim.get_sensor_observations()
        pose = self._pose_from_agent_state(self._agent.get_state())
        self._update_observed_cells(pose)
        return self._build_navigation_frame(observations, pose)

    def _build_navigation_frame(
        self,
        observations: Any,
        pose: Pose2D,
    ) -> NavigationFrame:
        """把同一步的 Habitat 观测与位姿转换为统一导航帧。"""
        return NavigationFrame(
            timestamp_s=time.monotonic(),
            pose=pose,
            obstacle_map=self._build_obstacle_map(),
            depth=_convert_depth(observations["depth_sensor"]),
            rgb=_convert_rgb(observations["color_sensor"]),
            camera_intrinsics=self._camera_intrinsics(),
        )

    def _camera_intrinsics(self) -> CameraIntrinsics:
        """由 Habitat 水平视场角计算像素制针孔相机内参。"""
        focal_length = (self.config.width * 0.5) / math.tan(
            math.radians(self.config.hfov_deg) * 0.5
        )
        return CameraIntrinsics(
            fx=focal_length,
            fy=focal_length,
            cx=(self.config.width - 1) * 0.5,
            cy=(self.config.height - 1) * 0.5,
        )

    def _pose_from_agent_state(self, state: Any) -> Pose2D:
        """把 Habitat 三维位姿转换为内部二维位姿。"""
        rotation = state.rotation
        imaginary = rotation.imag
        qx, qy, qz = (float(imaginary[0]), float(imaginary[1]), float(imaginary[2]))
        qw = float(rotation.real)

        # 四元数旋转 Habitat 局部前向 (0, 0, -1) 后的水平分量。
        forward_x = -2.0 * (qx * qz + qw * qy)
        forward_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
        yaw = math.atan2(-forward_z, forward_x)
        return Pose2D(
            x_m=float(state.position[0]),
            y_m=-float(state.position[2]),
            yaw_rad=yaw,
        )

    def _update_observed_cells(self, pose: Pose2D) -> None:
        """用相机视锥和 navmesh 遮挡更新已知地图。"""
        resolution = self.config.map_resolution_m
        near_radius = max(0.30, 2.0 * resolution)
        near_squared = near_radius * near_radius
        range_squared = self.config.observed_range_m**2
        half_fov = math.radians(self.config.hfov_deg) * 0.5
        robot_col = int(round((pose.x_m - self._map_origin.x_m) / resolution))
        robot_row = int(round((pose.y_m - self._map_origin.y_m) / resolution))
        range_cells = int(math.ceil(self.config.observed_range_m / resolution))
        first_row = max(0, robot_row - range_cells)
        last_row = min(len(self._observed), robot_row + range_cells + 1)
        first_col = max(0, robot_col - range_cells)
        last_col = min(len(self._observed[0]), robot_col + range_cells + 1)

        for row_index in range(first_row, last_row):
            observed_row = self._observed[row_index]
            world_y = self._map_origin.y_m + row_index * resolution
            for col_index in range(first_col, last_col):
                if observed_row[col_index]:
                    continue
                world_x = self._map_origin.x_m + col_index * resolution
                delta_x = world_x - pose.x_m
                delta_y = world_y - pose.y_m
                distance_squared = delta_x * delta_x + delta_y * delta_y
                in_near_area = distance_squared <= near_squared
                in_camera_view = (
                    distance_squared <= range_squared
                    and abs(_angle_difference(math.atan2(delta_y, delta_x), pose.yaw_rad))
                    <= half_fov
                    and self._has_line_of_sight(
                        robot_row,
                        robot_col,
                        row_index,
                        col_index,
                    )
                )
                if in_near_area or in_camera_view:
                    observed_row[col_index] = True

    def _has_line_of_sight(
        self,
        start_row: int,
        start_col: int,
        end_row: int,
        end_col: int,
    ) -> bool:
        """沿栅格射线检查遮挡；终点障碍本身仍视为可观测。"""
        row, col = start_row, start_col
        delta_col = abs(end_col - start_col)
        delta_row = -abs(end_row - start_row)
        step_col = 1 if start_col < end_col else -1
        step_row = 1 if start_row < end_row else -1
        error = delta_col + delta_row

        while (row, col) != (end_row, end_col):
            if (row, col) != (start_row, start_col):
                if not (
                    0 <= row < len(self._navigable)
                    and 0 <= col < len(self._navigable[row])
                ):
                    return False
                if not self._navigable[row][col]:
                    return False
            doubled_error = 2 * error
            if doubled_error >= delta_row:
                error += delta_row
                col += step_col
            if doubled_error <= delta_col:
                error += delta_col
                row += step_row
        return True

    def _build_obstacle_map(self) -> ObstacleMap:
        """把 navmesh 与已知掩码转换为核心占用栅格契约。"""
        occupancy = tuple(
            tuple(
                (0.0 if self._navigable[row][col] else 1.0)
                if self._observed[row][col]
                else None
                for col in range(len(self._navigable[row]))
            )
            for row in range(len(self._navigable))
        )
        return ObstacleMap(
            occupancy=occupancy,
            resolution_m=self.config.map_resolution_m,
            origin=self._map_origin,
            frame_id="habitat_world",
        )

    def send_relative_pose(self, command: RelativePoseCommand) -> None:
        """沿 navmesh 逐步移动到相对目标，再转到命令指定朝向。"""
        self._require_open()
        if not isinstance(command, RelativePoseCommand) or not all(
            _is_finite(value)
            for value in (command.forward_m, command.left_m, command.yaw_rad)
        ):
            raise ValueError("command 必须为有限 RelativePoseCommand")

        state = self._agent.get_state()
        start_pose = self._pose_from_agent_state(state)
        target_yaw_world = start_pose.yaw_rad + command.yaw_rad
        translation_m = math.hypot(command.forward_m, command.left_m)
        if translation_m > 1.0e-9:
            target, path_world_xy = self._plan_relative_target(
                state,
                start_pose,
                command,
            )
            target_world_xy = (float(target[0]), -float(target[2]))
            self._report_motion_plan(target_world_xy, path_world_xy)
            try:
                self._follow_path(target)
            finally:
                self._report_motion_plan(None, ())
        self._turn_to_world_yaw(target_yaw_world)

    def _plan_relative_target(
        self,
        state: Any,
        pose: Pose2D,
        command: RelativePoseCommand,
    ) -> Tuple[Any, Tuple[Tuple[float, float], ...]]:
        """把相对平移投影到 navmesh，返回实际目标和最短路径。"""
        cosine = math.cos(pose.yaw_rad)
        sine = math.sin(pose.yaw_rad)
        world_dx = command.forward_m * cosine - command.left_m * sine
        world_dy = command.forward_m * sine + command.left_m * cosine

        requested = state.position.copy()
        requested[0] = float(state.position[0]) + world_dx
        requested[2] = float(state.position[2]) - world_dy
        snapped = self._pathfinder.snap_point(requested)
        if not all(_is_finite(snapped[index]) for index in range(3)):
            raise RecoverableMotionError(
                "Habitat 无法把相对位姿目标投影到 navmesh"
            )

        snap_error = math.hypot(
            float(snapped[0]) - float(requested[0]),
            float(snapped[2]) - float(requested[2]),
        )
        tolerance = max(0.25, 2.0 * self.config.map_resolution_m)
        if snap_error > tolerance:
            raise RecoverableMotionError(
                "Habitat 相对位姿目标离可导航区域过远"
            )

        shortest_path = self._habitat_sim.ShortestPath()
        shortest_path.requested_start = state.position
        shortest_path.requested_end = snapped
        if not self._pathfinder.find_path(shortest_path) or not shortest_path.points:
            raise RecoverableMotionError(
                "Habitat 找不到相对位姿目标的可行路径"
            )
        path_world_xy = tuple(
            (float(point[0]), -float(point[2]))
            for point in shortest_path.points
        )
        return snapped, path_world_xy

    def _follow_path(self, target: Any) -> None:
        """用 Habitat GreedyGeodesicFollower 执行到目标的离散动作。"""
        follower = self._sim.make_greedy_follower(agent_id=0)
        try:
            actions = follower.find_path(target)
        except self._habitat_sim.errors.GreedyFollowerError as exc:
            raise RecoverableMotionError(
                "Habitat 无法把 navmesh 路径转换为动作"
            ) from exc

        for action in actions:
            if action is None:
                break
            self._step_action(action)

    def _turn_to_world_yaw(self, target_yaw_world: float) -> None:
        """用 Habitat 默认转向动作逼近世界系目标朝向。"""
        action_space = self._agent.agent_config.action_space
        left_step_rad = math.radians(
            float(action_space["turn_left"].actuation.amount)
        )
        right_step_rad = math.radians(
            float(action_space["turn_right"].actuation.amount)
        )
        turn_step_rad = min(left_step_rad, right_step_rad)
        if not math.isfinite(turn_step_rad) or turn_step_rad <= 0.0:
            raise RuntimeError("Habitat 转向动作步长无效")

        tolerance_rad = turn_step_rad * 0.5
        max_steps = int(math.ceil(2.0 * math.pi / turn_step_rad)) + 1
        for _ in range(max_steps):
            pose = self._pose_from_agent_state(self._agent.get_state())
            difference = _angle_difference(target_yaw_world, pose.yaw_rad)
            if abs(difference) <= tolerance_rad:
                return
            action = "turn_left" if difference > 0.0 else "turn_right"
            self._step_action(action)
        raise RuntimeError("Habitat 无法在有限动作内转到目标朝向")

    def _step_action(self, action: Any) -> None:
        """执行一个 Habitat 动作，更新地图并按需发布该步传感器帧。"""
        observations = self._sim.step(action)
        pose = self._pose_from_agent_state(self._agent.get_state())
        self._update_observed_cells(pose)
        if self._on_motion_frame is not None:
            self._on_motion_frame(self._build_navigation_frame(observations, pose))

    def _report_motion_plan(
        self,
        target_world_xy: Optional[Tuple[float, float]],
        path_world_xy: Tuple[Tuple[float, float], ...],
    ) -> None:
        """发布 navmesh 实际目标和路径；显示失败不影响仿真运动。"""
        callback = self._on_motion_plan
        if callback is None:
            return
        try:
            callback(target_world_xy, path_world_xy)
        except Exception:
            self._on_motion_plan = None

    def _require_open(self) -> None:
        """拒绝在 Adapter 关闭后继续读写仿真。"""
        if self._sim is None:
            raise RuntimeError("HabitatChassisAdapter 已关闭")

    def close(self) -> None:
        """幂等释放 Habitat 资源。"""
        simulator = getattr(self, "_sim", None)
        self._sim = None
        if simulator is not None:
            simulator.close()

    def __enter__(self) -> "HabitatChassisAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _convert_rgb(image: Any):
    """把 Habitat H×W×(3/4) 图像转换为内部不可变 RGB 序列。"""
    return tuple(
        tuple(tuple(int(channel) for channel in pixel[:3]) for pixel in row)
        for row in image.tolist()
    )


def _convert_depth(image: Any):
    """把 Habitat 米制深度转换为内部不可变深度序列。"""
    rows = []
    for row in image.tolist():
        converted_row = []
        for raw_value in row:
            value = raw_value[0] if isinstance(raw_value, (list, tuple)) else raw_value
            converted = float(value)
            converted_row.append(
                converted if math.isfinite(converted) and converted > 0.0 else None
            )
        rows.append(tuple(converted_row))
    return tuple(rows)


def _angle_difference(target_rad: float, current_rad: float) -> float:
    """返回范围 [-π, π) 的最短有符号角差。"""
    return (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi


def _is_finite(value: Any) -> bool:
    """value 是否为有限数，明确拒绝 bool。"""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["HabitatChassisAdapter", "HabitatConfig"]
