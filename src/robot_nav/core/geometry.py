"""共用空间计算：位姿与栅格转换、动作相对位姿、路径未知长度。

距离使用米，角度使用弧度；栅格为 (row, col)，origin 表示 (0, 0) 格中心。
转换不限制地图边界，地图外路径也要计入未知长度；算法与 Adapter 共用这些规则。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

from .models import ActionKind, NavigationAction, Pose2D, RelativePoseCommand, ObstacleMap


Cell = Tuple[int, int]
WorldPoint = Tuple[float, float]


@dataclass(frozen=True)
class UnknownPathMeasurement:
    """同一次路径快照的测量结果；长度单位米，首个未知格仅用于定位。"""

    unknown_length_m: float
    total_length_m: float
    first_unknown_cell: Optional[Cell]


def action_command(
    action: Optional[NavigationAction], pose: Pose2D,
) -> Optional[RelativePoseCommand]:
    """将世界系目标位姿转换为以输入帧为基准的相对位姿（米、弧度）。"""
    if action is None:
        return None
    if action.action is not ActionKind.MOVE_TO_POSE:
        return action.command
    destination = action.destination
    if destination is None:
        raise ValueError("世界系移动缺少目标位姿")
    forward, left = world_point_to_robot((destination.x_m, destination.y_m), pose)
    return RelativePoseCommand(forward_m=forward, left_m=left,
        yaw_rad=shortest_turn_to_heading(pose.yaw_rad, destination.yaw_rad))


def wrap_angle(angle_rad: float) -> float:
    """把弧度角归一化到 [-π, π)。"""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def world_point_to_robot(
    world_xy: Tuple[float, float], robot_pose: Pose2D
) -> Tuple[float, float]:
    """同一世界系中的点转到机器人局部系，返回 (前向米数, 左向米数)。"""
    delta_x = world_xy[0] - robot_pose.x_m
    delta_y = world_xy[1] - robot_pose.y_m
    cosine, sine = math.cos(robot_pose.yaw_rad), math.sin(robot_pose.yaw_rad)
    return delta_x * cosine + delta_y * sine, -delta_x * sine + delta_y * cosine


def grid_cell_center_to_world(
    row: int, col: int, obstacle_map: ObstacleMap
) -> Tuple[float, float]:
    """返回栅格中心的世界坐标（米），包含地图原点的平移和旋转。"""
    origin = obstacle_map.origin
    local_x = col * obstacle_map.resolution_m
    local_y = row * obstacle_map.resolution_m
    cosine, sine = math.cos(origin.yaw_rad), math.sin(origin.yaw_rad)
    return (origin.x_m + local_x * cosine - local_y * sine,
            origin.y_m + local_x * sine + local_y * cosine)


def world_to_nearest_grid_cell(
    world_xy: Tuple[float, float], obstacle_map: ObstacleMap
) -> Tuple[int, int]:
    """世界点转最近栅格 (row, col)；半值向上取整，不用银行家舍入。"""
    local_x, local_y = world_point_to_robot(world_xy, obstacle_map.origin)
    col = math.floor(local_x / obstacle_map.resolution_m + 0.5)
    row = math.floor(local_y / obstacle_map.resolution_m + 0.5)
    return row, col


def shortest_turn_to_heading(
    current_heading_rad: float, target_heading_rad: float
) -> float:
    """返回从当前朝向转到目标朝向的有符号最短角差，范围 [-π, π)（弧度）。"""
    current_heading = require_finite_angle(current_heading_rad, "current_heading_rad")
    target_heading = require_finite_angle(target_heading_rad, "target_heading_rad")
    return wrap_angle(target_heading - current_heading)


def require_finite_angle(value: float, name: str) -> float:
    """要求有限弧度角，不做字符串或其他类型的兼容转换。"""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite angle")
    return value


def measure_unknown_path_length(
    path_world_xy: Sequence[WorldPoint],
    obstacle_map: ObstacleMap,
) -> UnknownPathMeasurement:
    """累加每段路径位于未知格及地图外的实际长度；不按经过格数近似。

    路径与地图须在同一坐标系，调用方应把当前位置加在剩余路径前。
    沿格边行走时任一侧未知即计入一次；只接触格角不产生长度。重复路径段按
    实际行程累加，零长度段不贡献长度。占据格与车体净空仍由底盘规划器处理。
    """
    resolution = float(obstacle_map.resolution_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("路径检查需要正有限地图分辨率")
    grid = obstacle_map.occupancy
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("路径检查需要非空矩形地图")

    previous = None
    unknown_lengths = []
    segment_lengths = []
    first_unknown_cell = None
    for point in path_world_xy:
        local_x, local_y = world_point_to_robot(point, obstacle_map.origin)
        # origin 是 (0, 0) 格中心；平移半格后可用 floor 定位格子及交叉边界。
        current = (local_x / resolution + 0.5, local_y / resolution + 0.5)
        if previous is not None:
            dx, dy = current[0] - previous[0], current[1] - previous[1]
            length_m = math.hypot(dx, dy) * resolution
            if not math.isfinite(length_m):
                raise ValueError("路径线段长度必须为有限值")
            segment_lengths.append(length_m)
            if length_m > 0.0:
                cuts = _segment_grid_crossings(previous, current, len(grid[0]), len(grid))
                for start_t, end_t in zip(cuts, cuts[1:]):
                    # 相邻切点之间不跨格边，中点可代表整段；长度按 t 的占比精确累计。
                    middle_t = (start_t + end_t) * 0.5
                    cell = _unknown_cell_at(
                        (previous[0] + dx * middle_t, previous[1] + dy * middle_t), grid,
                    )
                    if cell is not None:
                        unknown_lengths.append(length_m * (end_t - start_t))
                        if first_unknown_cell is None:
                            first_unknown_cell = cell
        previous = current
    return UnknownPathMeasurement(math.fsum(unknown_lengths), math.fsum(segment_lengths), first_unknown_cell)


def _segment_grid_crossings(start: WorldPoint, end: WorldPoint, width: int, height: int) -> Tuple[float, ...]:
    """按穿越格边的参数 t 切分线段；地图外无须枚举无限延伸的格线。"""
    cuts = {0.0, 1.0}
    for first, last, bound in ((start[0], end[0], width), (start[1], end[1], height)):
        delta = last - first
        if delta == 0.0:
            continue
        lower = max(0, math.ceil(min(first, last)))
        upper = min(bound, math.floor(max(first, last)))
        for boundary in range(lower, upper + 1):
            t = (boundary - first) / delta
            if 0.0 < t < 1.0:
                cuts.add(t)
    return tuple(sorted(cuts))


def _unknown_cell_at(point: WorldPoint, grid) -> Optional[Cell]:
    """线段内部取样；恰好沿格边时检查两侧，未知部分只计一次。"""
    for row in _touching_axis_cells(point[1]):
        for col in _touching_axis_cells(point[0]):
            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or grid[row][col] is None:
                return row, col
    return None


def _touching_axis_cells(coordinate: float) -> Tuple[int, ...]:
    """坐标落在格边时返回两侧格号，否则只返回所在格号。"""
    boundary = round(coordinate)
    if math.isclose(coordinate, boundary, rel_tol=0.0, abs_tol=1e-10):
        return boundary - 1, boundary
    return (math.floor(coordinate),)
