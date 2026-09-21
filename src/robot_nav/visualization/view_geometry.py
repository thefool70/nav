"""导航世界坐标到可视化坐标的转换，以及机器人轮廓和路径线段计算。"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..core.geometry import world_to_nearest_grid_cell
from ..core.models import NavigationFrame, Pose2D, RelativePoseCommand

HISTORY_DASH_LENGTH_M = 0.12
HISTORY_DASH_GAP_M = 0.08
ROBOT_FRONT_M = 0.30
ROBOT_REAR_M = 0.20
ROBOT_HALF_WIDTH_M = 0.22


def robot_triangle_world(
    pose: Pose2D,
) -> Tuple[Tuple[float, float], ...]:
    """返回指向机器人前方的闭合三角形世界坐标。"""
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    local_points = (
        (ROBOT_FRONT_M, 0.0),
        (-ROBOT_REAR_M, ROBOT_HALF_WIDTH_M),
        (-ROBOT_REAR_M, -ROBOT_HALF_WIDTH_M),
        (ROBOT_FRONT_M, 0.0),
    )
    return tuple(
        (
            pose.x_m + forward_m * cosine - left_m * sine,
            pose.y_m + forward_m * sine + left_m * cosine,
        )
        for forward_m, left_m in local_points
    )


def world_to_view_point(
    world_xy: Tuple[float, float],
) -> Tuple[float, float]:
    """翻转世界 Y 轴，使 Rerun 的二维画布按数学坐标显示 Y 向上。"""
    return (world_xy[0], -world_xy[1])


def world_to_view_vector(
    world_vector: Tuple[float, float],
) -> Tuple[float, float]:
    """把世界系向量转换到 Y 向上的 Rerun 二维显示坐标。"""
    return (world_vector[0], -world_vector[1])


def world_yaw_to_view_vector(
    yaw_rad: float,
    length_m: float,
) -> Tuple[float, float]:
    return world_to_view_vector(
        (math.cos(yaw_rad) * length_m, math.sin(yaw_rad) * length_m)
    )


def command_world_vector(
    command: RelativePoseCommand,
    pose: Pose2D,
) -> Tuple[float, float]:
    """把机器人系前/左平移转换为世界系向量。"""
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        command.forward_m * cosine - command.left_m * sine,
        command.forward_m * sine + command.left_m * cosine,
    )


def world_to_map_pixel(
    world_xy: Tuple[float, float],
    frame: NavigationFrame,
) -> Optional[Tuple[float, float]]:
    """把世界点转换到 ``np.flipud`` 后的占据图像素坐标。"""
    obstacle_map = frame.obstacle_map
    row, col = world_to_nearest_grid_cell(world_xy, obstacle_map)
    height = len(obstacle_map.occupancy)
    width = len(obstacle_map.occupancy[0]) if height else 0
    if not 0 <= row < height or not 0 <= col < width:
        return None
    return (float(col), float(height - 1 - row))


def world_yaw_to_map_vector(
    yaw_rad: float,
    frame: NavigationFrame,
    length_m: float,
) -> Tuple[float, float]:
    """把世界朝向转换为翻转后地图图像中的像素向量。"""
    obstacle_map = frame.obstacle_map
    relative_yaw = yaw_rad - obstacle_map.origin.yaw_rad
    length_cells = length_m / obstacle_map.resolution_m
    return (
        math.cos(relative_yaw) * length_cells,
        -math.sin(relative_yaw) * length_cells,
    )


def point_along_heading(
    origin: Tuple[float, float],
    heading_rad: float,
    distance_m: float,
) -> Tuple[float, float]:
    return (
        origin[0] + math.cos(heading_rad) * distance_m,
        origin[1] + math.sin(heading_rad) * distance_m,
    )


def dashed_line_segments(
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]:
    """把一条线拆成短线段，模拟 Rerun 0.22 尚不支持的虚线。"""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = math.hypot(delta_x, delta_y)
    if length <= 1e-9:
        return ()
    direction_x = delta_x / length
    direction_y = delta_y / length
    segments = []
    distance = 0.0
    while distance < length:
        segment_end = min(distance + HISTORY_DASH_LENGTH_M, length)
        segments.append(
            (
                (
                    start[0] + direction_x * distance,
                    start[1] + direction_y * distance,
                ),
                (
                    start[0] + direction_x * segment_end,
                    start[1] + direction_y * segment_end,
                ),
            )
        )
        distance += HISTORY_DASH_LENGTH_M + HISTORY_DASH_GAP_M
    return tuple(segments)
