"""局部视觉覆盖：筛选待检查 Frontier，用检查成功帧的 RGB-D 记录观察范围。

覆盖记录包含固定世界网格采样和本轮 Frontier 点，不额外触发扫描。
这是相机高度附近的二维可见性近似，不把地图已知状态当作语义检查结果。
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from .geometry import (
    grid_cell_center_to_world,
    world_point_to_robot,
    world_to_nearest_grid_cell,
    wrap_angle,
)
from .models import FrontierCandidate, NavigationFrame, ObservationView, ObstacleMap


WorldPoint = Tuple[float, float]
OBSERVATION_RANGE_M = 4.0
OBSERVATION_SPACING_M = 0.25
MIN_OBSERVATION_DISTANCE_M = 0.5
VIEW_EDGE_MARGIN_RAD = math.radians(5.0)
VIEWPOINT_CHANGE_RAD = math.radians(30.0)
DEPTH_MARGIN_M = 0.10
SAME_POSITION_REUSE_M = 0.10


def camera_world_position(frame: NavigationFrame) -> WorldPoint:
    """返回相机光心的世界平面位置，包含安装平移。"""
    extrinsics = frame.camera_extrinsics_in_robot
    cosine, sine = math.cos(frame.pose.yaw_rad), math.sin(frame.pose.yaw_rad)
    return (
        frame.pose.x_m + cosine * extrinsics.forward_m - sine * extrinsics.left_m,
        frame.pose.y_m + sine * extrinsics.forward_m + cosine * extrinsics.left_m,
    )


def frontier_observation_points(
    frame: NavigationFrame,
    candidates: Sequence[FrontierCandidate],
) -> Tuple[WorldPoint, ...]:
    """返回局部可见 Frontier 的全部边界点，新旧候选都可提供待检查方向。

    使用边界格而非单个代表点，避免聚类缩减观察方向；不扫描隔墙或远处边界。
    """
    origin = camera_world_position(frame)
    points = {
        grid_cell_center_to_world(row, col, frame.obstacle_map)
        for candidate in candidates
        for row, col in candidate.frontier_cells
    }
    return tuple(
        point for point in sorted(points)
        if 0.10 < math.dist(origin, point) <= OBSERVATION_RANGE_M
        and _has_map_line_of_sight(frame.obstacle_map, origin, point)
    )


def _local_coverage_points(frame: NavigationFrame) -> Tuple[WorldPoint, ...]:
    """采样当前图像可记录的局部覆盖；这些点本身不用于发起扫描。"""
    origin = camera_world_position(frame)
    spacing = OBSERVATION_SPACING_M
    points = []
    for ix in range(
        math.ceil((origin[0] - OBSERVATION_RANGE_M) / spacing),
        math.floor((origin[0] + OBSERVATION_RANGE_M) / spacing) + 1,
    ):
        for iy in range(
            math.ceil((origin[1] - OBSERVATION_RANGE_M) / spacing),
            math.floor((origin[1] + OBSERVATION_RANGE_M) / spacing) + 1,
        ):
            point = (ix * spacing, iy * spacing)
            distance = math.dist(origin, point)
            if (
                MIN_OBSERVATION_DISTANCE_M <= distance <= OBSERVATION_RANGE_M
                and _has_map_line_of_sight(frame.obstacle_map, origin, point)
            ):
                points.append(point)
    # 记录首层未知边界的覆盖，后续地图公开时可复用；不推测边界后方可见。
    points.extend(_visible_unknown_boundary_points(frame.obstacle_map, origin))
    return tuple(sorted(set(points)))


def _visible_unknown_boundary_points(
    obstacle_map: ObstacleMap, origin: WorldPoint,
) -> Tuple[WorldPoint, ...]:
    """返回视线能到达的首层未知格，仅用于当前图像的覆盖记录。"""
    grid = obstacle_map.occupancy
    row, col = world_to_nearest_grid_cell(origin, obstacle_map)
    radius = math.ceil(OBSERVATION_RANGE_M / obstacle_map.resolution_m) + 1
    points = []
    for r in range(max(0, row - radius), min(len(grid), row + radius + 1)):
        for c in range(max(0, col - radius), min(len(grid[0]), col + radius + 1)):
            if grid[r][c] is not None or not any(
                _free_cell(grid, neighbor)
                for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            ):
                continue
            point = grid_cell_center_to_world(r, c, obstacle_map)
            if (
                0.10 < math.dist(origin, point) <= OBSERVATION_RANGE_M
                and _has_map_line_of_sight(obstacle_map, origin, point)
            ):
                points.append(point)
    return tuple(points)


def unobserved_observation_points(
    points: Sequence[WorldPoint],
    frame: NavigationFrame,
    views: Sequence[ObservationView],
) -> Tuple[WorldPoint, ...]:
    """复用已检查的同一区域；明显换角度或走近后允许重新检查。"""
    origin = camera_world_position(frame)
    coverage = {}
    checked_directions = set()
    for view in views:
        for point in view.visible_world_xy:
            coverage.setdefault(tuple(point), []).append(view)
        # 同一位置已经检查过此视锥，无需为仍被遮挡的点重复拍摄。
        # 新露出的点不在旧视锥的地图可见点列表中，仍然会补查。
        if math.dist(_view_origin(view), origin) <= SAME_POSITION_REUSE_M:
            checked_directions.update(tuple(point) for point in view.map_visible_world_xy)
    return tuple(
        point for point in points
        if point not in checked_directions
        and not any(_similar_viewpoint(point, origin, view) for view in coverage.get(point, ()))
    )


def capture_observation_view(
    frame: NavigationFrame,
    camera_center_offset_rad: float,
    horizontal_fov_rad: float,
    *,
    observation_points: Sequence[WorldPoint] = (),
) -> ObservationView:
    """冻结当前图像的覆盖；调用方仅在语义检查成功后将它加入 observed_views。"""
    origin = camera_world_position(frame)
    heading = frame.pose.yaw_rad + camera_center_offset_rad
    depth_available = _aligned_depth_available(frame)
    map_points = set(_local_coverage_points(frame))
    # Frontier 格心未必落在固定采样网格上，显式记录才能按同一点复用检查结果。
    map_points.update(
        point for point in observation_points
        if 0.10 < math.dist(origin, point) <= OBSERVATION_RANGE_M
        and _has_map_line_of_sight(frame.obstacle_map, origin, point)
    )
    in_view = tuple(
        point for point in sorted(map_points)
        if abs(wrap_angle(_bearing(origin, point) - heading))
        <= max(0.0, horizontal_fov_rad / 2.0 - VIEW_EDGE_MARGIN_RAD)
    )
    visible = ()
    if depth_available:
        visible = tuple(
            point for point in in_view if _depth_supports_point(frame, point)
        )
    return ObservationView(
        pose=frame.pose,
        camera_heading_world_rad=heading,
        horizontal_fov_rad=horizontal_fov_rad,
        timestamp_s=frame.timestamp_s,
        camera_world_xy=origin,
        visible_world_xy=visible,
        map_visible_world_xy=in_view,
        depth_coverage_available=depth_available,
    )


def _has_map_line_of_sight(
    obstacle_map: ObstacleMap, origin: WorldPoint, target: WorldPoint,
) -> bool:
    """只允许视线穿过已知自由格；终点障碍可观察，禁止穿过对角墙角。"""
    grid = obstacle_map.occupancy
    start = world_to_nearest_grid_cell(origin, obstacle_map)
    end = world_to_nearest_grid_cell(target, obstacle_map)
    if not (0 <= end[0] < len(grid) and 0 <= end[1] < len(grid[0])):
        return False
    steps = max(1, math.ceil(math.dist(origin, target) / (obstacle_map.resolution_m / 2.0)))
    previous = start
    for index in range(1, steps + 1):
        fraction = index / steps
        cell = world_to_nearest_grid_cell((
            origin[0] + (target[0] - origin[0]) * fraction,
            origin[1] + (target[1] - origin[1]) * fraction,
        ), obstacle_map)
        if cell[0] != previous[0] and cell[1] != previous[1]:
            if not all(_free_cell(grid, side) for side in (
                (previous[0], cell[1]), (cell[0], previous[1]),
            )):
                return False
        if cell == end:
            return True
        if cell != start and not _free_cell(grid, cell):
            return False
        previous = cell
    return True


def _free_cell(grid, cell: Tuple[int, int]) -> bool:
    row, col = cell
    return (
        0 <= row < len(grid) and 0 <= col < len(grid[0])
        and grid[row][col] is not None and grid[row][col] <= 0.5
    )


def _view_origin(view: ObservationView) -> WorldPoint:
    return view.camera_world_xy or (view.pose.x_m, view.pose.y_m)


def _bearing(origin: WorldPoint, point: WorldPoint) -> float:
    return math.atan2(point[1] - origin[1], point[0] - origin[0])


def _similar_viewpoint(point: WorldPoint, origin: WorldPoint, view: ObservationView) -> bool:
    previous = _view_origin(view)
    angle_change = abs(wrap_angle(_bearing(previous, point) - _bearing(origin, point)))
    previous_distance = math.dist(previous, point)
    distance = math.dist(origin, point)
    # 显著接近后，原本很小的物体可能变得可辨认；不能仅因坐标曾覆盖而跳过。
    return (
        angle_change <= VIEWPOINT_CHANGE_RAD
        and previous_distance <= max(distance * 1.5, distance + 0.5)
    )


def _aligned_depth_available(frame: NavigationFrame) -> bool:
    depth, rgb, intrinsics = frame.depth, frame.rgb, frame.camera_intrinsics
    if depth is None or rgb is None or intrinsics is None:
        return False
    try:
        height = len(depth)
        width = len(depth[0]) if height else 0
        return (
            height > 0 and width > 0 and len(rgb) == height
            and all(len(row) == width for row in depth)
            and all(len(row) == width for row in rgb)
            and all(math.isfinite(float(value)) for value in (
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            ))
            and intrinsics.fx > 0.0 and intrinsics.fy > 0.0
        )
    except (TypeError, ValueError):
        return False


def _depth_supports_point(frame: NavigationFrame, point: WorldPoint) -> bool:
    """在相机高度及其上下 0.25 m 投影采样；任一高度被深度遮挡则不登记覆盖。"""
    intrinsics = frame.camera_intrinsics
    depth = frame.depth
    extrinsics = frame.camera_extrinsics_in_robot
    forward, left = world_point_to_robot(point, frame.pose)
    forward -= extrinsics.forward_m
    left -= extrinsics.left_m
    cyaw, syaw = math.cos(extrinsics.yaw_rad), math.sin(extrinsics.yaw_rad)
    pitched_forward = cyaw * forward + syaw * left
    rolled_left = -syaw * forward + cyaw * left
    cp, sp = math.cos(extrinsics.pitch_down_rad), math.sin(extrinsics.pitch_down_rad)
    cr, sr = math.cos(extrinsics.roll_rad), math.sin(extrinsics.roll_rad)
    for up in (-0.25, 0.0, 0.25):
        camera_forward = cp * pitched_forward - sp * up
        rolled_up = sp * pitched_forward + cp * up
        camera_left = cr * rolled_left - sr * rolled_up
        camera_up = sr * rolled_left + cr * rolled_up
        if camera_forward <= 0.0:
            return False
        col = round(intrinsics.cx - intrinsics.fx * camera_left / camera_forward)
        row = round(intrinsics.cy - intrinsics.fy * camera_up / camera_forward)
        if not (1 <= row < len(depth) - 1 and 1 <= col < len(depth[0]) - 1):
            return False
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            value = depth[row + dy][col + dx]
            if value is None or isinstance(value, bool):
                return False
            try:
                value = float(value)
            except (TypeError, ValueError):
                return False
            if (
                not math.isfinite(value) or value <= 0.0
                or camera_forward > value + DEPTH_MARGIN_M
            ):
                return False
    return True
