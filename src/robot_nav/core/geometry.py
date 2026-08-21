"""坐标转换与角度归一化工具。

约定：角度单位为弧度（逆时针为正），距离单位为米。栅格以 (row, col) 索引，
col 沿地图 origin 局部 +x 方向增长，row 沿 origin 局部 +y 方向增长。

所有公开函数对类型不可转换、非有限值和非法分辨率统一抛 ValueError，
不泄漏 TypeError。
"""

from __future__ import annotations

import math
from typing import Tuple

from .models import ObstacleMap, Pose2D


def _require_finite(value, name):
    """返回 float(value)，拒绝 bool、不可转换与非有限输入，统一抛 ValueError。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _require_non_negative_int(value, name):
    """校验 value 为非负整数（不含 bool），否则抛 ValueError。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_finite(value, name):
    """校验 value 为正有限值并返回 float(value)，否则抛 ValueError。"""
    converted = _require_finite(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return converted


def wrap_angle(angle_rad: float) -> float:
    """把角度归一化到 [-π, π)（弧度）。非有限值抛 ValueError。"""
    angle = _require_finite(angle_rad, "angle_rad")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_point_to_robot(
    world_xy: Tuple[float, float], robot_pose: Pose2D
) -> Tuple[float, float]:
    """把世界坐标点转换到机器人局部系，返回 (x 向前, y 向左)，单位为米。

    依赖：world_xy 与 robot_pose 属于同一世界坐标系。
    """
    if not isinstance(robot_pose, Pose2D):
        raise ValueError("robot_pose must be a Pose2D")
    try:
        world_x, world_y = world_xy
    except (TypeError, ValueError):
        raise ValueError("world_xy must be a pair of finite numbers") from None
    world_x = _require_finite(world_x, "world_x")
    world_y = _require_finite(world_y, "world_y")
    x_m = _require_finite(robot_pose.x_m, "robot_pose.x_m")
    y_m = _require_finite(robot_pose.y_m, "robot_pose.y_m")
    yaw_rad = _require_finite(robot_pose.yaw_rad, "robot_pose.yaw_rad")
    delta_x = world_x - x_m
    delta_y = world_y - y_m
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    local_x = delta_x * cos_yaw + delta_y * sin_yaw
    local_y = -delta_x * sin_yaw + delta_y * cos_yaw
    return (local_x, local_y)


def grid_cell_center_to_world(
    row: int, col: int, obstacle_map: ObstacleMap
) -> Tuple[float, float]:
    """返回栅格 (row, col) 中心的世界坐标（米）。

    列沿 origin 局部 +x，行沿 origin 局部 +y；坐标转换同时考虑 origin 位姿
    的 x、y 与 yaw。row 与 col 必须为非负整数；不检查地图数组边界。
    """
    if not isinstance(obstacle_map, ObstacleMap):
        raise ValueError("obstacle_map must be an ObstacleMap")
    _require_non_negative_int(row, "row")
    _require_non_negative_int(col, "col")
    resolution_m = _require_positive_finite(obstacle_map.resolution_m, "resolution_m")
    if not isinstance(obstacle_map.origin, Pose2D):
        raise ValueError("obstacle_map.origin must be a Pose2D")
    x_m = _require_finite(obstacle_map.origin.x_m, "origin.x_m")
    y_m = _require_finite(obstacle_map.origin.y_m, "origin.y_m")
    yaw_rad = _require_finite(obstacle_map.origin.yaw_rad, "origin.yaw_rad")
    local_x = col * resolution_m
    local_y = row * resolution_m
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    world_x = x_m + local_x * cos_yaw - local_y * sin_yaw
    world_y = y_m + local_x * sin_yaw + local_y * cos_yaw
    return (world_x, world_y)


def _nearest_int(value: float) -> int:
    """返回离 value 最近的整数，半值向上取整（规则明确，避免 round 的银行家舍入）。"""
    return int(math.floor(value + 0.5))


def world_to_nearest_grid_cell(
    world_xy: Tuple[float, float], obstacle_map: ObstacleMap
) -> Tuple[int, int]:
    """返回离世界坐标点最近的栅格下标 (row, col)。

    最近格规则：局部坐标除以分辨率后，半值向上取整（floor(x + 0.5)），
    不使用 Python round 的银行家舍入；不检查地图数组边界。
    """
    if not isinstance(obstacle_map, ObstacleMap):
        raise ValueError("obstacle_map must be an ObstacleMap")
    resolution_m = _require_positive_finite(obstacle_map.resolution_m, "resolution_m")
    if not isinstance(obstacle_map.origin, Pose2D):
        raise ValueError("obstacle_map.origin must be a Pose2D")
    try:
        world_x, world_y = world_xy
    except (TypeError, ValueError):
        raise ValueError("world_xy must be a pair of finite numbers") from None
    world_x = _require_finite(world_x, "world_x")
    world_y = _require_finite(world_y, "world_y")
    x_m = _require_finite(obstacle_map.origin.x_m, "origin.x_m")
    y_m = _require_finite(obstacle_map.origin.y_m, "origin.y_m")
    yaw_rad = _require_finite(obstacle_map.origin.yaw_rad, "origin.yaw_rad")
    delta_x = world_x - x_m
    delta_y = world_y - y_m
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    local_x = delta_x * cos_yaw + delta_y * sin_yaw
    local_y = -delta_x * sin_yaw + delta_y * cos_yaw
    col = _nearest_int(local_x / resolution_m)
    row = _nearest_int(local_y / resolution_m)
    return (row, col)
