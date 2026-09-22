"""优先用 RGB-D 定位；失败时沿检测框方向或光轴查找假设目标的障碍表面。"""

from __future__ import annotations

import math

from .grounding import ground_target_bbox
from .geometry import world_point_to_robot
from .models import NavigationFrame, TargetEstimate


def localize_segmented_object(frame: NavigationFrame, bbox_norm, mask=None) -> TargetEstimate:
    """只需存在可计算的正深度；没有掩码时直接使用检测框。"""
    if frame.depth is None or frame.camera_intrinsics is None:
        return TargetEstimate(False, "object_depth_or_intrinsics_missing")
    return ground_target_bbox(
        bbox_norm, frame.depth, frame.camera_intrinsics, frame.pose, frame.camera_extrinsics_in_robot,
        target_mask=mask,
    )


def localize_obstacle_on_image_ray(frame: NavigationFrame, bbox_norm=None) -> TargetEstimate:
    """用拍摄位姿投出射线，以首个已知障碍表面作为物体位置假设；不设假定距离。"""
    grid = frame.navigation_map if frame.navigation_map is not None else frame.obstacle_map
    ray = _image_ray_world(frame, bbox_norm)
    if ray is None:
        return TargetEstimate(False, "object_ray_calibration_invalid")
    origin, heading = ray
    hit = _first_obstacle_on_ray(grid, origin, heading)
    if hit is None:
        return TargetEstimate(False, "object_ray_has_no_obstacle")
    base_xy = world_point_to_robot(hit, frame.pose)
    return TargetEstimate(
        success=True,
        reason="target_assumed_from_bbox_obstacle" if bbox_norm is not None else "target_assumed_from_front_obstacle",
        target_base_xy=base_xy, target_world_xy=hit,
        distance_m=math.hypot(*base_xy), bearing_rad=math.atan2(base_xy[1], base_xy[0]),
        sample_count=0,
    )


def _image_ray_world(frame, bbox_norm):
    """框中心像素生成相机射线；无框使用光轴，roll/pitch/yaw 与 RGB-D 定位一致。"""
    left, up = 0.0, 0.0
    if bbox_norm is not None:
        k = frame.camera_intrinsics
        if frame.rgb is None or not frame.rgb or not frame.rgb[0] or k is None:
            return None
        if not all(math.isfinite(value) for value in (k.fx, k.fy, k.cx, k.cy)) or min(k.fx, k.fy) <= 0:
            return None
        if len(bbox_norm) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in bbox_norm):
            return None
        x0, y0, x1, y1 = bbox_norm
        if x0 >= x1 or y0 >= y1:
            return None
        u = (x0 + x1) * 0.5 * len(frame.rgb[0])
        v = (y0 + y1) * 0.5 * len(frame.rgb)
        left, up = (k.cx - u) / k.fx, (k.cy - v) / k.fy
    e, pose = frame.camera_extrinsics_in_robot, frame.pose
    rolled_left = left * math.cos(e.roll_rad) + up * math.sin(e.roll_rad)
    rolled_up = -left * math.sin(e.roll_rad) + up * math.cos(e.roll_rad)
    forward = math.cos(e.pitch_down_rad) + rolled_up * math.sin(e.pitch_down_rad)
    if math.hypot(forward, rolled_left) < 1e-9:
        return None
    heading = pose.yaw_rad + e.yaw_rad + math.atan2(rolled_left, forward)
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    origin = (pose.x_m + c * e.forward_m - s * e.left_m,
              pose.y_m + s * e.forward_m + c * e.left_m)
    return origin, heading


def _first_obstacle_on_ray(grid, origin_world, heading_world):
    """按穿过格子边界的顺序遍历射线，返回首个占用格的进入点；未知格不提供距离。"""
    height = len(grid.occupancy)
    width = len(grid.occupancy[0]) if height else 0
    if not height or not width:
        return None
    x, y = world_point_to_robot(origin_world, grid.origin)
    yaw = heading_world - grid.origin.yaw_rad
    dx, dy = math.cos(yaw), math.sin(yaw)
    resolution = grid.resolution_m
    start, end = 0.0, math.inf
    # 地图原点是格中心，数组外边界位于半格之外。
    for value, direction, count in ((x, dx, width), (y, dy, height)):
        low, high = -0.5 * resolution, (count - 0.5) * resolution
        if abs(direction) < 1e-12:
            if not low <= value < high:
                return None
            continue
        first, last = sorted(((low - value) / direction, (high - value) / direction))
        start, end = max(start, first), min(end, last)
    if start >= end:
        return None
    # 取边界内侧确定首格；命中位置仍使用精确的进入距离。
    inside = start + min(resolution * 1e-7, (end - start) * 0.5)
    col = math.floor((x + inside * dx) / resolution + 0.5)
    row = math.floor((y + inside * dy) / resolution + 0.5)
    step_col, step_row = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    next_x = ((col + 0.5 * step_col) * resolution - x) / dx if abs(dx) >= 1e-12 else math.inf
    next_y = ((row + 0.5 * step_row) * resolution - y) / dy if abs(dy) >= 1e-12 else math.inf
    delta_x = resolution / abs(dx) if abs(dx) >= 1e-12 else math.inf
    delta_y = resolution / abs(dy) if abs(dy) >= 1e-12 else math.inf
    distance = start
    while 0 <= row < height and 0 <= col < width and distance < end:
        value = grid.occupancy[row][col]
        if value is not None and value > 0.5:
            return (origin_world[0] + distance * math.cos(heading_world),
                    origin_world[1] + distance * math.sin(heading_world))
        distance = min(next_x, next_y)
        cross_x, cross_y = next_x <= next_y, next_y <= next_x
        if cross_x:
            col += step_col
            next_x += delta_x
        if cross_y:
            row += step_row
            next_y += delta_y
    return None
