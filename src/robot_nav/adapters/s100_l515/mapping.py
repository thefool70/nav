"""把 L515 深度投影为固定在 S100 里程计坐标系中的二维占用图。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from ...core.models import CameraIntrinsics, ObstacleMap, Pose2D


@dataclass(frozen=True)
class CameraMount:
    """L515 在机器人坐标系中的安装位姿。

    forward_m、left_m、height_m 分别沿机器人前、左、上方向；yaw_rad 正值向左；
    pitch_down_rad 正值表示镜头光轴向地面俯视。当前不支持 roll。
    """

    height_m: float
    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_down_rad: float = 0.0


@dataclass(frozen=True)
class DepthOccupancyMapConfig:
    """固定正方形地图及深度投影参数，所有长度单位均为米。"""

    size_m: float = 20.0
    resolution_m: float = 0.10
    min_depth_m: float = 0.20
    max_depth_m: float = 6.0
    sample_stride_px: int = 8
    min_obstacle_height_m: float = 0.05
    max_obstacle_height_m: float = 1.50
    floor_tolerance_m: float = 0.10
    robot_clear_radius_m: float = 0.25
    frame_id: str = "s100_odom"


class DepthOccupancyMap:
    """累计稀疏深度射线，输出 unknown/free/occupied 三态栅格。"""

    _MIN_EVIDENCE = -5
    _MAX_EVIDENCE = 5
    _FREE_UPDATE = -1
    _OCCUPIED_UPDATE = 3

    def __init__(self, config: DepthOccupancyMapConfig) -> None:
        _validate_map_config(config)
        self.config = config
        cell_count = int(math.ceil(config.size_m / config.resolution_m))
        self._height = cell_count
        self._width = cell_count
        self._origin = Pose2D(
            x_m=-(cell_count // 2) * config.resolution_m,
            y_m=-(cell_count // 2) * config.resolution_m,
            yaw_rad=0.0,
        )
        self._seen = [bytearray(cell_count) for _ in range(cell_count)]
        self._evidence = [
            [0 for _ in range(cell_count)] for _ in range(cell_count)
        ]

    def update(
        self,
        depth_m: Any,
        intrinsics: CameraIntrinsics,
        robot_pose: Pose2D,
        camera_mount: CameraMount,
    ) -> None:
        """用一帧米制深度更新自由射线与障碍端点。"""
        height, width = _depth_shape(depth_m)
        _validate_intrinsics(intrinsics)
        _validate_pose(robot_pose, "robot_pose")
        _validate_camera_mount(camera_mount)

        camera_world_xy = _camera_world_xy(robot_pose, camera_mount)
        ray_start = self._world_to_cell(camera_world_xy)
        stride = self.config.sample_stride_px
        for row in range(stride // 2, height, stride):
            for col in range(stride // 2, width, stride):
                raw_depth = depth_m[row][col]
                if not _is_finite_number(raw_depth):
                    continue
                distance_m = float(raw_depth)
                if not self.config.min_depth_m <= distance_m <= self.config.max_depth_m:
                    continue

                point_robot = _depth_pixel_to_robot(
                    col,
                    row,
                    distance_m,
                    intrinsics,
                    camera_mount,
                )
                point_height_m = point_robot[2]
                if (
                    point_height_m < -self.config.floor_tolerance_m
                    or point_height_m > self.config.max_obstacle_height_m
                ):
                    continue

                point_world = _robot_point_to_world(point_robot, robot_pose)
                ray_end = self._world_to_cell((point_world[0], point_world[1]))
                is_obstacle = (
                    self.config.min_obstacle_height_m
                    <= point_height_m
                    <= self.config.max_obstacle_height_m
                )
                self._integrate_ray(ray_start, ray_end, is_obstacle)

        self._mark_robot_area_free(robot_pose)

    def to_obstacle_map(self) -> ObstacleMap:
        """冻结当前证据为 core 使用的 0.0/1.0/None 栅格。"""
        occupancy = tuple(
            tuple(
                None
                if not self._seen[row][col]
                else (1.0 if self._evidence[row][col] > 0 else 0.0)
                for col in range(self._width)
            )
            for row in range(self._height)
        )
        return ObstacleMap(
            occupancy=occupancy,
            resolution_m=self.config.resolution_m,
            origin=self._origin,
            frame_id=self.config.frame_id,
        )

    def _integrate_ray(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        endpoint_is_obstacle: bool,
    ) -> None:
        """射线经过格记自由；有效高度端点额外记障碍。"""
        cells = tuple(_bresenham_cells(start, end))
        free_cells = cells[:-1] if endpoint_is_obstacle else cells
        for row, col in free_cells:
            if self._in_bounds(row, col):
                self._add_evidence(row, col, self._FREE_UPDATE)
        if endpoint_is_obstacle:
            row, col = end
            if self._in_bounds(row, col):
                self._add_evidence(row, col, self._OCCUPIED_UPDATE)

    def _mark_robot_area_free(self, pose: Pose2D) -> None:
        """确保机器人当前占据区域不会因深度噪声成为障碍。"""
        center_row, center_col = self._world_to_cell((pose.x_m, pose.y_m))
        radius_cells = int(
            math.ceil(
                self.config.robot_clear_radius_m / self.config.resolution_m
            )
        )
        radius_squared = self.config.robot_clear_radius_m**2
        for row in range(center_row - radius_cells, center_row + radius_cells + 1):
            for col in range(
                center_col - radius_cells, center_col + radius_cells + 1
            ):
                if not self._in_bounds(row, col):
                    continue
                dx = (col - center_col) * self.config.resolution_m
                dy = (row - center_row) * self.config.resolution_m
                if dx * dx + dy * dy <= radius_squared:
                    self._seen[row][col] = 1
                    self._evidence[row][col] = self._MIN_EVIDENCE

    def _world_to_cell(self, world_xy: Tuple[float, float]) -> Tuple[int, int]:
        # 与 core.geometry.world_to_nearest_grid_cell 保持相同的半值向上规则。
        col = int(
            math.floor(
                (world_xy[0] - self._origin.x_m)
                / self.config.resolution_m
                + 0.5
            )
        )
        row = int(
            math.floor(
                (world_xy[1] - self._origin.y_m)
                / self.config.resolution_m
                + 0.5
            )
        )
        return row, col

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self._height and 0 <= col < self._width

    def _add_evidence(self, row: int, col: int, update: int) -> None:
        self._seen[row][col] = 1
        self._evidence[row][col] = min(
            self._MAX_EVIDENCE,
            max(self._MIN_EVIDENCE, self._evidence[row][col] + update),
        )


def _depth_pixel_to_robot(
    col: int,
    row: int,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    mount: CameraMount,
) -> Tuple[float, float, float]:
    """把 RealSense 右/下/前坐标转换为机器人前/左/上坐标。"""
    camera_right = (col - intrinsics.cx) * depth_m / intrinsics.fx
    camera_down = (row - intrinsics.cy) * depth_m / intrinsics.fy
    camera_forward = depth_m
    camera_left = -camera_right
    camera_up = -camera_down

    pitch_cos = math.cos(mount.pitch_down_rad)
    pitch_sin = math.sin(mount.pitch_down_rad)
    pitched_forward = camera_forward * pitch_cos + camera_up * pitch_sin
    pitched_up = -camera_forward * pitch_sin + camera_up * pitch_cos

    yaw_cos = math.cos(mount.yaw_rad)
    yaw_sin = math.sin(mount.yaw_rad)
    robot_forward = pitched_forward * yaw_cos - camera_left * yaw_sin
    robot_left = pitched_forward * yaw_sin + camera_left * yaw_cos
    return (
        mount.forward_m + robot_forward,
        mount.left_m + robot_left,
        mount.height_m + pitched_up,
    )


def _robot_point_to_world(
    point_robot: Tuple[float, float, float],
    pose: Pose2D,
) -> Tuple[float, float, float]:
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        pose.x_m + point_robot[0] * cosine - point_robot[1] * sine,
        pose.y_m + point_robot[0] * sine + point_robot[1] * cosine,
        point_robot[2],
    )


def _camera_world_xy(
    pose: Pose2D,
    mount: CameraMount,
) -> Tuple[float, float]:
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        pose.x_m + mount.forward_m * cosine - mount.left_m * sine,
        pose.y_m + mount.forward_m * sine + mount.left_m * cosine,
    )


def _bresenham_cells(
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Iterable[Tuple[int, int]]:
    """返回包含首尾格的整数栅格射线。"""
    row, col = start
    end_row, end_col = end
    delta_col = abs(end_col - col)
    delta_row = -abs(end_row - row)
    step_col = 1 if col < end_col else -1
    step_row = 1 if row < end_row else -1
    error = delta_col + delta_row
    while True:
        yield row, col
        if row == end_row and col == end_col:
            return
        doubled = 2 * error
        if doubled >= delta_row:
            error += delta_row
            col += step_col
        if doubled <= delta_col:
            error += delta_col
            row += step_row


def _depth_shape(depth_m: Any) -> Tuple[int, int]:
    try:
        height = len(depth_m)
        width = len(depth_m[0])
    except (TypeError, IndexError):
        raise ValueError("depth_m 必须为非空二维数组") from None
    if height <= 0 or width <= 0:
        raise ValueError("depth_m 必须为非空二维数组")
    return height, width


def _validate_intrinsics(intrinsics: CameraIntrinsics) -> None:
    if not isinstance(intrinsics, CameraIntrinsics):
        raise ValueError("intrinsics 必须为 CameraIntrinsics")
    values = (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    if not all(_is_finite_number(value) for value in values):
        raise ValueError("相机内参必须为有限数")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("相机焦距必须为正数")


def _validate_camera_mount(mount: CameraMount) -> None:
    if not isinstance(mount, CameraMount):
        raise ValueError("camera_mount 必须为 CameraMount")
    values = (
        mount.height_m,
        mount.forward_m,
        mount.left_m,
        mount.yaw_rad,
        mount.pitch_down_rad,
    )
    if not all(_is_finite_number(value) for value in values):
        raise ValueError("相机安装参数必须为有限数")
    if mount.height_m <= 0.0:
        raise ValueError("相机安装高度必须为正数")


def _validate_map_config(config: DepthOccupancyMapConfig) -> None:
    if not isinstance(config, DepthOccupancyMapConfig):
        raise ValueError("config 必须为 DepthOccupancyMapConfig")
    positive = (
        config.size_m,
        config.resolution_m,
        config.min_depth_m,
        config.max_depth_m,
        config.max_obstacle_height_m,
    )
    non_negative = (
        config.min_obstacle_height_m,
        config.floor_tolerance_m,
        config.robot_clear_radius_m,
    )
    if not all(_is_finite_number(value) and float(value) > 0.0 for value in positive):
        raise ValueError("地图尺寸、分辨率、深度和最大障碍高度必须为正有限数")
    if not all(
        _is_finite_number(value) and float(value) >= 0.0
        for value in non_negative
    ):
        raise ValueError("障碍下限、地面容差和机器人半径必须为非负有限数")
    if config.min_depth_m >= config.max_depth_m:
        raise ValueError("min_depth_m 必须小于 max_depth_m")
    if config.min_obstacle_height_m >= config.max_obstacle_height_m:
        raise ValueError("障碍高度下限必须小于上限")
    if (
        isinstance(config.sample_stride_px, bool)
        or not isinstance(config.sample_stride_px, int)
        or config.sample_stride_px <= 0
    ):
        raise ValueError("sample_stride_px 必须为正整数")
    if not isinstance(config.frame_id, str) or not config.frame_id.strip():
        raise ValueError("frame_id 必须为非空字符串")


def _validate_pose(pose: Pose2D, name: str) -> None:
    if not isinstance(pose, Pose2D) or not all(
        _is_finite_number(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    ):
        raise ValueError(f"{name} 必须为有限 Pose2D")


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "CameraMount",
    "DepthOccupancyMap",
    "DepthOccupancyMapConfig",
]
