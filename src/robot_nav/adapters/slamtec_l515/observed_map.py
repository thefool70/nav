"""用 L515 水平 FOV 与障碍遮挡筛选 Hermes 地图，再膨胀禁行区域。"""

from __future__ import annotations

import math
from typing import Any, Optional, Set, Tuple

from ...core.geometry import (
    grid_cell_center_to_world,
    world_point_to_robot,
    world_to_nearest_grid_cell,
)
from ...core.models import CameraExtrinsics, ObstacleMap, Pose2D
from ..realsense import L515Capture


Cell = Tuple[int, int]
WorldGridKey = Tuple[int, int]

# Hermes 外形为 0.465 m × 0.545 m；0.36 m 是半对角线向上取整后的保守半径。
HERMES_OBSTACLE_INFLATION_RADIUS_M = 0.36
START_AREA_RADIUS_M = 0.50
MAX_FOV_DISTANCE_M = 5.0
# 邻近 3 列消除单列深度孔洞，15 cm 余量吸收深度噪声与地图格误差。
FOV_DEPTH_MARGIN_M = 0.15
FOV_DEPTH_COLUMN_RADIUS_PX = 1


class L515ObservedMap:
    """累计未被障碍遮挡的 FOV 与出生点地图格，并输出膨胀有效图。"""

    def __init__(
        self,
        inflation_radius_m: float = HERMES_OBSTACLE_INFLATION_RADIUS_M,
        max_fov_distance_m: float = MAX_FOV_DISTANCE_M,
        start_area_radius_m: float = START_AREA_RADIUS_M,
    ) -> None:
        if not math.isfinite(inflation_radius_m) or inflation_radius_m <= 0.0:
            raise ValueError("inflation_radius_m 必须为正有限数")
        if (
            not math.isfinite(max_fov_distance_m)
            or max_fov_distance_m <= 0.0
        ):
            raise ValueError("max_fov_distance_m 必须为正有限数")
        if (
            not math.isfinite(start_area_radius_m)
            or start_area_radius_m <= 0.0
        ):
            raise ValueError("start_area_radius_m 必须为正有限数")

        self._inflation_radius_m = float(inflation_radius_m)
        self._max_fov_distance_m = float(max_fov_distance_m)
        self._start_area_radius_m = float(start_area_radius_m)
        self._anchor: Optional[Pose2D] = None
        self._resolution_m: Optional[float] = None
        self._start_world_xy: Optional[Tuple[float, float]] = None
        self._seen_world_keys: Set[WorldGridKey] = set()

    def update(
        self,
        source: ObstacleMap,
        robot_pose: Pose2D,
        capture: L515Capture,
        camera_extrinsics: CameraExtrinsics,
    ) -> ObstacleMap:
        """累计当前可见格，并返回未见为 None 的膨胀占用图。"""
        self._prepare_map_geometry(source)
        if self._start_world_xy is None:
            self._start_world_xy = (float(robot_pose.x_m), float(robot_pose.y_m))

        visible_cells = _cells_in_camera_fov(
            source,
            robot_pose,
            capture,
            camera_extrinsics,
            self._max_fov_distance_m,
        )
        # 每帧重算，确保 Hermes 扩图后仍补入出生点附近的新格。
        visible_cells.update(
            _cells_in_world_radius(
                self._start_world_xy,
                self._start_area_radius_m,
                source,
            )
        )
        for cell in visible_cells:
            self._seen_world_keys.add(self._cell_to_world_key(cell, source))

        occupancy = [[None for _ in row] for row in source.occupancy]
        for row, col in self._seen_cells_in(source):
            occupancy[row][col] = source.occupancy[row][col]
        _inflate_obstacles(
            occupancy,
            source.resolution_m,
            self._inflation_radius_m,
        )
        return ObstacleMap(
            occupancy=tuple(tuple(row) for row in occupancy),
            resolution_m=source.resolution_m,
            origin=source.origin,
            frame_id=source.frame_id,
        )

    def _prepare_map_geometry(self, obstacle_map: ObstacleMap) -> None:
        """地图分辨率或方向变化时清空不再兼容的累计可见格。"""
        resolution = float(obstacle_map.resolution_m)
        yaw = float(obstacle_map.origin.yaw_rad)
        geometry_changed = (
            self._resolution_m is not None
            and (
                not math.isclose(resolution, self._resolution_m, abs_tol=1e-9)
                or self._anchor is None
                or not math.isclose(
                    yaw, float(self._anchor.yaw_rad), abs_tol=1e-9
                )
            )
        )
        if self._anchor is None or geometry_changed:
            self._anchor = obstacle_map.origin
            self._resolution_m = resolution
            self._seen_world_keys.clear()

    def _cell_to_world_key(
        self, cell: Cell, obstacle_map: ObstacleMap
    ) -> WorldGridKey:
        """用首次地图格网作为锚点，使 Hermes 扩图后仍能找回已见格。"""
        anchor = self._require_anchor()
        resolution = self._require_resolution()
        world_x, world_y = grid_cell_center_to_world(
            cell[0], cell[1], obstacle_map
        )
        delta_x = world_x - anchor.x_m
        delta_y = world_y - anchor.y_m
        cosine = math.cos(anchor.yaw_rad)
        sine = math.sin(anchor.yaw_rad)
        local_x = delta_x * cosine + delta_y * sine
        local_y = -delta_x * sine + delta_y * cosine
        return (
            _nearest_int(local_y / resolution),
            _nearest_int(local_x / resolution),
        )

    def _seen_cells_in(self, obstacle_map: ObstacleMap) -> Set[Cell]:
        """把累计世界格网投回 Hermes 当前可能扩展过的地图数组。"""
        anchor = self._require_anchor()
        resolution = self._require_resolution()
        cosine = math.cos(anchor.yaw_rad)
        sine = math.sin(anchor.yaw_rad)
        cells = set()
        for key_row, key_col in self._seen_world_keys:
            local_x = key_col * resolution
            local_y = key_row * resolution
            world_xy = (
                anchor.x_m + local_x * cosine - local_y * sine,
                anchor.y_m + local_x * sine + local_y * cosine,
            )
            cell = world_to_nearest_grid_cell(world_xy, obstacle_map)
            if _cell_in_map(cell, obstacle_map):
                cells.add(cell)
        return cells

    def _require_anchor(self) -> Pose2D:
        if self._anchor is None:
            raise RuntimeError("L515 可见地图尚未初始化")
        return self._anchor

    def _require_resolution(self) -> float:
        if self._resolution_m is None:
            raise RuntimeError("L515 可见地图尚未初始化")
        return self._resolution_m


def _cells_in_camera_fov(
    obstacle_map: ObstacleMap,
    robot_pose: Pose2D,
    capture: L515Capture,
    camera_extrinsics: CameraExtrinsics,
    max_distance_m: float,
) -> Set[Cell]:
    """返回水平 FOV 内且未被 L515 深度视界截断的地图格。"""
    _, width = capture.depth_m.shape
    intrinsics = capture.camera_intrinsics
    depth_horizon = _farthest_valid_depth_by_column(capture.depth_m)
    camera_world = _robot_point_to_world(
        (camera_extrinsics.forward_m, camera_extrinsics.left_m),
        robot_pose,
    )
    camera_pose = Pose2D(
        x_m=camera_world[0],
        y_m=camera_world[1],
        yaw_rad=robot_pose.yaw_rad + camera_extrinsics.yaw_rad,
    )
    rows, columns = _cells_near_world_point(
        camera_world, max_distance_m, obstacle_map
    )
    visible = set()
    for row in rows:
        for col in columns:
            world_xy = grid_cell_center_to_world(row, col, obstacle_map)
            forward_m, left_m = world_point_to_robot(world_xy, camera_pose)
            distance_m = math.hypot(forward_m, left_m)
            if forward_m <= 0.0 or distance_m > max_distance_m:
                continue
            image_col = _horizontal_image_column(
                forward_m,
                left_m,
                float(intrinsics.fx),
                float(intrinsics.cx),
                width,
            )
            if image_col is None:
                continue
            if _depth_horizon_reaches(
                depth_horizon,
                image_col,
                forward_m,
            ):
                visible.add((row, col))
    return visible


def _horizontal_image_column(
    forward_m: float,
    left_m: float,
    focal_length_px: float,
    principal_x_px: float,
    image_width: int,
) -> Optional[int]:
    """把相机水平平面中的方向投影到图像列。"""
    image_col = _nearest_int(
        principal_x_px - focal_length_px * left_m / forward_m
    )
    if not 0 <= image_col < image_width:
        return None
    return image_col


def _farthest_valid_depth_by_column(
    depth_m: Any,
) -> Tuple[Optional[float], ...]:
    """返回每个图像列跨全部高度的最远有效深度。"""
    height, width = depth_m.shape
    farthest: list[Optional[float]] = [None] * int(width)
    for row in range(int(height)):
        for col in range(int(width)):
            raw_depth = depth_m[row][col]
            try:
                value = float(raw_depth)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or value <= 0.0:
                continue
            current_farthest = farthest[col]
            if current_farthest is None or value > current_farthest:
                farthest[col] = value
    return tuple(farthest)


def _depth_horizon_reaches(
    farthest_depth_by_column: Tuple[Optional[float], ...],
    image_col: int,
    required_forward_m: float,
) -> bool:
    """判断目标方向是否有深度射线到达；无有效深度时使用理论 FOV。"""
    first_col = max(0, image_col - FOV_DEPTH_COLUMN_RADIUS_PX)
    last_col = min(
        len(farthest_depth_by_column),
        image_col + FOV_DEPTH_COLUMN_RADIUS_PX + 1,
    )
    valid_depths = tuple(
        depth
        for depth in farthest_depth_by_column[first_col:last_col]
        if depth is not None
    )
    if not valid_depths:
        return True
    return max(valid_depths) + FOV_DEPTH_MARGIN_M >= required_forward_m


def _cells_near_world_point(
    world_xy: Tuple[float, float],
    radius_m: float,
    obstacle_map: ObstacleMap,
) -> Tuple[range, range]:
    """返回覆盖世界系正方形包围框的有效行列范围。"""
    corner_cells = tuple(
        world_to_nearest_grid_cell(
            (world_xy[0] + x_offset, world_xy[1] + y_offset),
            obstacle_map,
        )
        for x_offset in (-radius_m, radius_m)
        for y_offset in (-radius_m, radius_m)
    )
    height = len(obstacle_map.occupancy)
    width = len(obstacle_map.occupancy[0]) if height else 0
    if height == 0 or width == 0:
        return range(0), range(0)
    first_row = min(
        height - 1, max(0, min(cell[0] for cell in corner_cells))
    )
    last_row = min(
        height - 1, max(0, max(cell[0] for cell in corner_cells))
    )
    first_col = min(
        width - 1, max(0, min(cell[1] for cell in corner_cells))
    )
    last_col = min(
        width - 1, max(0, max(cell[1] for cell in corner_cells))
    )
    return (
        range(first_row, last_row + 1),
        range(first_col, last_col + 1),
    )


def _cells_in_world_radius(
    world_xy: Tuple[float, float],
    radius_m: float,
    obstacle_map: ObstacleMap,
) -> Set[Cell]:
    """返回以世界坐标点为圆心、指定半径内的地图格。"""
    rows, columns = _cells_near_world_point(
        world_xy, radius_m, obstacle_map
    )
    cells: Set[Cell] = set()
    for row in rows:
        for col in columns:
            center_x, center_y = grid_cell_center_to_world(
                row, col, obstacle_map
            )
            if (
                math.hypot(center_x - world_xy[0], center_y - world_xy[1])
                <= radius_m
            ):
                cells.add((row, col))
    return cells


def _robot_point_to_world(
    robot_xy: Tuple[float, float], pose: Pose2D
) -> Tuple[float, float]:
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    return (
        pose.x_m + robot_xy[0] * cosine - robot_xy[1] * sine,
        pose.y_m + robot_xy[0] * sine + robot_xy[1] * cosine,
    )


def _inflate_obstacles(
    occupancy: list,
    resolution_m: float,
    radius_m: float,
) -> None:
    """把已见障碍周围一个底盘半径内的格子标为不可通行。"""
    occupied_cells = [
        (row_index, col_index)
        for row_index, row in enumerate(occupancy)
        for col_index, value in enumerate(row)
        if _is_occupied(value)
    ]
    offsets = _inflation_offsets(resolution_m, radius_m)
    height = len(occupancy)
    width = len(occupancy[0]) if height else 0
    for row, col in occupied_cells:
        for row_offset, col_offset in offsets:
            near_row = row + row_offset
            near_col = col + col_offset
            if 0 <= near_row < height and 0 <= near_col < width:
                occupancy[near_row][near_col] = 1.0


def _inflation_offsets(
    resolution_m: float, radius_m: float
) -> Tuple[Cell, ...]:
    maximum_offset = int(math.ceil(radius_m / resolution_m))
    return tuple(
        (row_offset, col_offset)
        for row_offset in range(-maximum_offset, maximum_offset + 1)
        for col_offset in range(-maximum_offset, maximum_offset + 1)
        if math.hypot(row_offset, col_offset) * resolution_m <= radius_m
    )


def _cell_in_map(cell: Cell, obstacle_map: ObstacleMap) -> bool:
    height = len(obstacle_map.occupancy)
    width = len(obstacle_map.occupancy[0]) if height else 0
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _is_occupied(value: Optional[float]) -> bool:
    return value is not None and float(value) > 0.5


def _nearest_int(value: float) -> int:
    return int(math.floor(value + 0.5))


__all__ = [
    "HERMES_OBSTACLE_INFLATION_RADIUS_M",
    "L515ObservedMap",
    "MAX_FOV_DISTANCE_M",
    "START_AREA_RADIUS_M",
]
