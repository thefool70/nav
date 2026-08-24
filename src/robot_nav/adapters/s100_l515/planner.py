"""S100 真机 Adapter 使用的已知自由栅格 A* 路径规划。"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

from ...core.geometry import (
    grid_cell_center_to_world,
    world_to_nearest_grid_cell,
)
from ...core.models import ObstacleMap


Cell = Tuple[int, int]
WorldPoint = Tuple[float, float]


def plan_known_free_path(
    obstacle_map: ObstacleMap,
    start_world_xy: WorldPoint,
    goal_world_xy: WorldPoint,
    robot_radius_m: float,
) -> Tuple[WorldPoint, ...]:
    """在已知自由格上规划路径，返回不含起点、包含终点的世界系路点。

    未知格与占用值大于 0.5 的格不可通行；已知障碍会按 robot_radius_m 膨胀。
    起终点越界、被阻挡或没有路径时抛出明确 RuntimeError。
    """
    grid = _normalize_grid(obstacle_map)
    radius_m = _non_negative_finite(robot_radius_m, "robot_radius_m")
    height, width = len(grid), len(grid[0])
    start = world_to_nearest_grid_cell(start_world_xy, obstacle_map)
    goal = world_to_nearest_grid_cell(goal_world_xy, obstacle_map)
    _require_in_bounds(start, height, width, "起点")
    _require_in_bounds(goal, height, width, "终点")

    unknown = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value is None
    }
    occupied = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value is not None and value > 0.5
    }
    blocked = unknown | _inflate_obstacles(
        occupied,
        radius_m,
        obstacle_map.resolution_m,
        height,
        width,
    )
    if start in blocked:
        raise RuntimeError("路径起点不在已知安全自由区")
    if goal in blocked:
        raise RuntimeError("路径终点不在已知安全自由区")
    if start == goal:
        return ((float(goal_world_xy[0]), float(goal_world_xy[1])),)

    cells = _astar(start, goal, blocked, height, width)
    if not cells:
        raise RuntimeError("当前已知自由区中找不到到底盘目标的路径")
    turns = _keep_turning_points(cells)
    waypoints = [
        grid_cell_center_to_world(row, col, obstacle_map)
        for row, col in turns[1:]
    ]
    waypoints[-1] = (float(goal_world_xy[0]), float(goal_world_xy[1]))
    return tuple(waypoints)


def _astar(
    start: Cell,
    goal: Cell,
    blocked: Set[Cell],
    height: int,
    width: int,
) -> Tuple[Cell, ...]:
    """八邻接 A*；对角移动要求两侧正交格也可通行。"""
    queue = [(float(_octile_distance(start, goal)), 0.0, start)]
    distance: Dict[Cell, float] = {start: 0.0}
    previous: Dict[Cell, Cell] = {}
    while queue:
        _, queued_distance, current = heapq.heappop(queue)
        if queued_distance > distance.get(current, math.inf):
            continue
        if current == goal:
            return _reconstruct_path(previous, current)
        for neighbor, step_cost in _neighbors(
            current, blocked, height, width
        ):
            candidate_distance = queued_distance + step_cost
            if candidate_distance >= distance.get(neighbor, math.inf):
                continue
            distance[neighbor] = candidate_distance
            previous[neighbor] = current
            priority = candidate_distance + _octile_distance(neighbor, goal)
            heapq.heappush(
                queue,
                (float(priority), candidate_distance, neighbor),
            )
    return ()


def _neighbors(
    cell: Cell,
    blocked: Set[Cell],
    height: int,
    width: int,
) -> Iterable[Tuple[Cell, float]]:
    row, col = cell
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            neighbor = (row + row_offset, col + col_offset)
            if not _is_free(neighbor, blocked, height, width):
                continue
            diagonal = row_offset != 0 and col_offset != 0
            if diagonal and (
                not _is_free(
                    (row + row_offset, col), blocked, height, width
                )
                or not _is_free(
                    (row, col + col_offset), blocked, height, width
                )
            ):
                continue
            yield neighbor, math.sqrt(2.0) if diagonal else 1.0


def _inflate_obstacles(
    occupied: Set[Cell],
    radius_m: float,
    resolution_m: float,
    height: int,
    width: int,
) -> Set[Cell]:
    """按欧氏距离膨胀已知障碍，不额外膨胀未知区域。"""
    radius_cells = int(math.ceil(radius_m / resolution_m))
    offsets = tuple(
        (row_offset, col_offset)
        for row_offset in range(-radius_cells, radius_cells + 1)
        for col_offset in range(-radius_cells, radius_cells + 1)
        if math.hypot(row_offset, col_offset) * resolution_m <= radius_m
    )
    inflated = set(occupied)
    for row, col in occupied:
        for row_offset, col_offset in offsets:
            candidate = (row + row_offset, col + col_offset)
            if 0 <= candidate[0] < height and 0 <= candidate[1] < width:
                inflated.add(candidate)
    return inflated


def _keep_turning_points(path: Sequence[Cell]) -> Tuple[Cell, ...]:
    """删除同一直线上的中间格，保留起点、转折点和终点。"""
    if len(path) <= 2:
        return tuple(path)
    result = [path[0]]
    previous_direction = _cell_direction(path[0], path[1])
    for index in range(1, len(path) - 1):
        next_direction = _cell_direction(path[index], path[index + 1])
        if next_direction != previous_direction:
            result.append(path[index])
        previous_direction = next_direction
    result.append(path[-1])
    return tuple(result)


def _reconstruct_path(
    previous: Dict[Cell, Cell],
    current: Cell,
) -> Tuple[Cell, ...]:
    reversed_path = [current]
    while current in previous:
        current = previous[current]
        reversed_path.append(current)
    reversed_path.reverse()
    return tuple(reversed_path)


def _cell_direction(first: Cell, second: Cell) -> Cell:
    return second[0] - first[0], second[1] - first[1]


def _octile_distance(first: Cell, second: Cell) -> float:
    delta_row = abs(first[0] - second[0])
    delta_col = abs(first[1] - second[1])
    diagonal = min(delta_row, delta_col)
    straight = max(delta_row, delta_col) - diagonal
    return diagonal * math.sqrt(2.0) + straight


def _is_free(
    cell: Cell,
    blocked: Set[Cell],
    height: int,
    width: int,
) -> bool:
    return (
        0 <= cell[0] < height
        and 0 <= cell[1] < width
        and cell not in blocked
    )


def _normalize_grid(
    obstacle_map: ObstacleMap,
) -> Tuple[Tuple[Optional[float], ...], ...]:
    if not isinstance(obstacle_map, ObstacleMap):
        raise ValueError("obstacle_map 必须为 ObstacleMap")
    try:
        rows = tuple(tuple(row) for row in obstacle_map.occupancy)
    except TypeError:
        raise ValueError("occupancy 必须为非空矩形栅格") from None
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("occupancy 必须为非空矩形栅格")
    normalized = []
    for row in rows:
        normalized_row = []
        for value in row:
            if value is None:
                normalized_row.append(None)
                continue
            if isinstance(value, bool):
                raise ValueError("occupancy 只能包含有限数或 None")
            try:
                converted = float(value)
            except (TypeError, ValueError):
                raise ValueError("occupancy 只能包含有限数或 None") from None
            if not math.isfinite(converted):
                raise ValueError("occupancy 只能包含有限数或 None")
            normalized_row.append(converted)
        normalized.append(tuple(normalized_row))
    resolution = obstacle_map.resolution_m
    if (
        isinstance(resolution, bool)
        or not isinstance(resolution, (int, float))
        or not math.isfinite(resolution)
        or resolution <= 0.0
    ):
        raise ValueError("resolution_m 必须为正有限数")
    return tuple(normalized)


def _require_in_bounds(
    cell: Cell,
    height: int,
    width: int,
    name: str,
) -> None:
    if not (0 <= cell[0] < height and 0 <= cell[1] < width):
        raise RuntimeError(f"路径{name}超出当前占用图范围")


def _non_negative_finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须为非负有限数")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须为非负有限数") from None
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} 必须为非负有限数")
    return converted


__all__ = ["plan_known_free_path"]
