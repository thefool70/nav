"""米制坐标转换；输入遵循 Pose2D / ObstacleMap 契约，物理数据在导航边界检查。

栅格以 (row, col) 索引，列沿地图 origin 局部 +x，行沿局部 +y。
转换不检查地图数组边界，地图外点也可用于计算路径未知长度。
"""

from __future__ import annotations

import math
from typing import Tuple

from .models import ObstacleMap, Pose2D


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


__all__ = ["wrap_angle", "world_point_to_robot", "grid_cell_center_to_world", "world_to_nearest_grid_cell"]
