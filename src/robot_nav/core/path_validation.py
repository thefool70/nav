"""按算法地图测量底盘路径落在未知区域内的长度，不依赖底盘或可视化接口。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .geometry import world_point_to_robot
from .models import ObstacleMap


Cell = Tuple[int, int]
WorldPoint = Tuple[float, float]


@dataclass(frozen=True)
class UnknownPathMeasurement:
    """同一次路径快照的测量结果；长度单位米，首个未知格仅用于定位。"""

    unknown_length_m: float
    total_length_m: float
    first_unknown_cell: Optional[Cell]


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
    boundary = round(coordinate)
    if math.isclose(coordinate, boundary, rel_tol=0.0, abs_tol=1e-10):
        return boundary - 1, boundary
    return (math.floor(coordinate),)
