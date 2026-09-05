"""按算法地图检查底盘路径是否穿过未知区，不依赖底盘或可视化接口。"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Sequence, Tuple

from .geometry import world_point_to_robot
from .models import ObstacleMap


Cell = Tuple[int, int]
WorldPoint = Tuple[float, float]


def first_unknown_path_cell(
    path_world_xy: Sequence[WorldPoint],
    obstacle_map: ObstacleMap,
) -> Optional[Cell]:
    """返回路径经过的首个未知格，地图外也算未知；无路径或未经过则返回 None。

    路径与地图须在同一坐标系，调用方应把当前位置加在剩余路径前。
    逐段遍历栅格，格角交叉同时检查两侧，避免稀疏路径点漏掉中间未知区。
    占据格的碰撞与车体净空仍由底盘规划器处理。
    """
    resolution = float(obstacle_map.resolution_m)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("路径检查需要正有限地图分辨率")
    grid = obstacle_map.occupancy
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("路径检查需要非空矩形地图")

    previous = None
    for point in path_world_xy:
        local_x, local_y = world_point_to_robot(point, obstacle_map.origin)
        # origin 是 (0, 0) 格中心；平移半格后可用 floor 定位格子及交叉边界。
        current = (local_x / resolution + 0.5, local_y / resolution + 0.5)
        segment_start = current if previous is None else previous
        for row, col in _segment_cells(segment_start, current):
            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
                return row, col
            if grid[row][col] is None:
                return row, col
        previous = current
    return None


def _segment_cells(start: WorldPoint, end: WorldPoint) -> Iterator[Cell]:
    """遍历栅格坐标线段；坐标以格边界为整数，输出 (row, col)。"""
    x, y = start
    end_x, end_y = end
    col, row = math.floor(x), math.floor(y)
    end_col, end_row = math.floor(end_x), math.floor(end_y)
    dx, dy = end_x - x, end_y - y
    step_col = 1 if dx > 0.0 else -1
    step_row = 1 if dy > 0.0 else -1
    yield row, col

    while col != end_col or row != end_row:
        next_x = (
            ((col + 1 if dx > 0.0 else col) - x) / dx
            if col != end_col else math.inf
        )
        next_y = (
            ((row + 1 if dy > 0.0 else row) - y) / dy
            if row != end_row else math.inf
        )
        if math.isclose(next_x, next_y, rel_tol=0.0, abs_tol=1e-12):
            yield row, col + step_col
            yield row + step_row, col
            col += step_col
            row += step_row
        elif next_x < next_y:
            col += step_col
        else:
            row += step_row
        yield row, col
