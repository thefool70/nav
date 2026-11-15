"""由当前激光直接更新历史栅格；底盘概率图只提供格网几何。"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from robot_nav.interfaces.geometry import world_point_to_robot
from robot_nav.interfaces.types import ObstacleMap, Pose2D
from robot_nav.platforms.hermes.rest_client import HermesLaserScan


Cell = Tuple[int, int]


class HermesLaserMap:
    """地图线程独占累积图；新射线清除旧障碍，未扫描区域保留历史值。"""

    def __init__(self) -> None:
        self._anchor = None
        self._resolution = None
        self._frame_id = None
        self._occupancy: Dict[Cell, float] = {}

    def update(self, geometry: ObstacleMap, scan: HermesLaserScan) -> ObstacleMap:
        """返回完整实时激光历史图；不读取 geometry 的概率占用值。"""
        if (self._anchor is None or self._resolution != geometry.resolution_m
                or self._frame_id != geometry.frame_id
                or self._anchor.yaw_rad != geometry.origin.yaw_rad):
            self._anchor = geometry.origin
            self._resolution = geometry.resolution_m
            self._frame_id = geometry.frame_id
            self._occupancy.clear()
        start = self._grid_point((scan.pose.x_m, scan.pose.y_m))
        free = set()
        occupied = set()
        for distance, angle in scan.points:
            heading = scan.pose.yaw_rad + angle
            end = self._grid_point((scan.pose.x_m + distance * math.cos(heading),
                                    scan.pose.y_m + distance * math.sin(heading)))
            cells = _ray_cells(start, end)
            free.update(cells[:-1])
            occupied.add(cells[-1])
        # 同一帧的命中优先于其他射线的自由证据，避免细障碍被交叉射线擦除。
        for cell in free:
            self._occupancy[cell] = 0.0
        for cell in occupied:
            self._occupancy[cell] = 1.0

        # 数组至少覆盖底盘格网，并随激光历史扩展，不等待底盘概率图扩图。
        origin = self._grid_point((geometry.origin.x_m, geometry.origin.y_m))
        row0, col0 = math.floor(origin[1]), math.floor(origin[0])
        height, width = len(geometry.occupancy), len(geometry.occupancy[0])
        first_row = min(row0, min((cell[0] for cell in self._occupancy), default=row0))
        first_col = min(col0, min((cell[1] for cell in self._occupancy), default=col0))
        last_row = max(row0 + height - 1, max((cell[0] for cell in self._occupancy), default=row0))
        last_col = max(col0 + width - 1, max((cell[1] for cell in self._occupancy), default=col0))
        values = [[None] * (last_col - first_col + 1) for _ in range(last_row - first_row + 1)]
        for (row, col), value in self._occupancy.items():
            values[row - first_row][col - first_col] = value
        local_x, local_y = first_col * self._resolution, first_row * self._resolution
        cosine, sine = math.cos(self._anchor.yaw_rad), math.sin(self._anchor.yaw_rad)
        return ObstacleMap(
            occupancy=tuple(tuple(row) for row in values), resolution_m=self._resolution,
            origin=Pose2D(self._anchor.x_m + local_x * cosine - local_y * sine,
                          self._anchor.y_m + local_x * sine + local_y * cosine,
                          self._anchor.yaw_rad), frame_id=self._frame_id,
        )

    def _grid_point(self, world_xy: Tuple[float, float]) -> Tuple[float, float]:
        """返回以格边界为整数的连续坐标，供射线逐格遍历。"""
        x, y = world_point_to_robot(world_xy, self._anchor)
        return x / self._resolution + 0.5, y / self._resolution + 0.5


def _ray_cells(start: Tuple[float, float], end: Tuple[float, float]) -> Tuple[Cell, ...]:
    """逐格遍历激光线段；仅接触格角的邻格不标自由，末格留给命中。"""
    x, y = start
    col, row = math.floor(x), math.floor(y)
    end_col, end_row = math.floor(end[0]), math.floor(end[1])
    dx, dy = end[0] - x, end[1] - y
    step_x, step_y = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
    next_x = ((col + 1 - x) / dx if dx > 0 else (col - x) / dx) if dx else math.inf
    next_y = ((row + 1 - y) / dy if dy > 0 else (row - y) / dy) if dy else math.inf
    delta_x = abs(1 / dx) if dx else math.inf
    delta_y = abs(1 / dy) if dy else math.inf
    cells = [(row, col)]
    while (row, col) != (end_row, end_col):
        # 已到末格的轴不再前进，避免端点恰好在格角时越过终点。
        tx = next_x if col != end_col else math.inf
        ty = next_y if row != end_row else math.inf
        if tx <= ty:
            col += step_x
            next_x += delta_x
        if ty <= tx:
            row += step_y
            next_y += delta_y
        cells.append((row, col))
    return tuple(cells)
