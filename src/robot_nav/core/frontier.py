"""从占用栅格提取可达 Frontier，并按几何与首选方向排序。"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Sequence, Set, Tuple

from .geometry import grid_cell_center_to_world, world_to_nearest_grid_cell
from .models import FrontierCandidate, ObstacleMap, Pose2D


Cell = Tuple[int, int]
GridValues = Tuple[Tuple[Optional[float], ...], ...]

PATH_DISTANCE_SCORE_WEIGHT = 0.05
PREFERRED_HEADING_SCORE_WEIGHT = 0.75


def find_frontier_candidates(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    preferred_heading_world_rad: Optional[float] = None,
    excluded_world_xy: Sequence[Tuple[float, float]] = (),
    min_frontier_length_m: float = 0.5,
    min_goal_distance_m: float = 0.35,
    max_search_path_distance_m: float = 3.0,
) -> Tuple[FrontierCandidate, ...]:
    """在局部路径范围内返回探索候选；候选不是路径或安全证明。"""
    grid = _normalize_grid(obstacle_map)
    resolution = _positive_finite(obstacle_map.resolution_m, "resolution_m")
    minimum_length = _non_negative_finite(
        min_frontier_length_m, "min_frontier_length_m"
    )
    minimum_distance = _non_negative_finite(
        min_goal_distance_m, "min_goal_distance_m"
    )
    maximum_distance = _positive_finite(
        max_search_path_distance_m, "max_search_path_distance_m"
    )
    preferred_heading = _optional_heading(preferred_heading_world_rad)
    excluded_points = _normalize_points(excluded_world_xy)

    free_cells = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value is not None and value <= 0.5
    }
    if not free_cells:
        return ()

    requested_seed = world_to_nearest_grid_cell(
        (pose.x_m, pose.y_m), obstacle_map
    )
    seed = _nearest_free_cell(requested_seed, free_cells)
    maximum_steps = int(math.floor(maximum_distance / resolution))
    reachable_distance = _reachable_free_distances(
        seed, free_cells, maximum_steps
    )
    reachable_cells = set(reachable_distance)
    map_frontier_cells = _find_frontier_cells(grid, reachable_cells)
    range_frontier_cells = (
        _find_range_frontier_cells(reachable_cells, free_cells)
        - map_frontier_cells
    )

    minimum_cells = max(1, int(math.ceil(minimum_length / resolution)))
    candidate_groups = []
    for component in _connected_components(map_frontier_cells):
        if len(component) >= minimum_cells:
            candidate_groups.append(
                ("map_frontier", component, _component_midpoint(component))
            )

    direction_reference = (
        preferred_heading if preferred_heading is not None else pose.yaw_rad
    )
    for direction_heading, cells in _group_cells_by_direction(
        range_frontier_cells,
        obstacle_map,
        pose,
        direction_reference,
    ):
        candidate_groups.append(
            (
                "range_frontier",
                cells,
                _cell_nearest_heading(
                    cells, direction_heading, obstacle_map, pose
                ),
            )
        )

    excluded_radius = max(0.4, 2.0 * resolution)
    candidates = []
    for frontier_kind, cells, (row, col) in candidate_groups:
        path_distance = reachable_distance[(row, col)] * resolution
        if path_distance < minimum_distance:
            continue
        world_xy = grid_cell_center_to_world(row, col, obstacle_map)
        if _is_excluded(world_xy, excluded_points, excluded_radius):
            continue
        heading = math.atan2(world_xy[1] - pose.y_m, world_xy[0] - pose.x_m)
        frontier_length = len(cells) * resolution
        score = frontier_length - PATH_DISTANCE_SCORE_WEIGHT * path_distance
        if preferred_heading is not None:
            score += PREFERRED_HEADING_SCORE_WEIGHT * math.cos(
                _angle_difference(heading, preferred_heading)
            )
        candidates.append(
            FrontierCandidate(
                candidate_id=f"{frontier_kind}:{row}:{col}",
                row=row,
                col=col,
                world_xy=world_xy,
                heading_world_rad=heading,
                frontier_cells=tuple(sorted(cells)),
                frontier_cell_count=len(cells),
                path_distance_m=path_distance,
                score=score,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.path_distance_m,
            candidate.candidate_id,
        )
    )
    return tuple(candidates)


def _normalize_grid(obstacle_map: ObstacleMap) -> GridValues:
    """校验并冻结矩形占用栅格。"""
    if not isinstance(obstacle_map, ObstacleMap):
        raise ValueError("obstacle_map must be an ObstacleMap")
    try:
        rows = tuple(tuple(row) for row in obstacle_map.occupancy)
    except TypeError:
        raise ValueError("occupancy must be a rectangular grid") from None
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("occupancy must be a non-empty rectangular grid")

    normalized = []
    for row in rows:
        normalized_row = []
        for value in row:
            if value is None:
                normalized_row.append(None)
                continue
            if isinstance(value, bool):
                raise ValueError("occupancy values must be finite numbers or None")
            try:
                converted = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    "occupancy values must be finite numbers or None"
                ) from None
            if not math.isfinite(converted):
                raise ValueError("occupancy values must be finite numbers or None")
            normalized_row.append(converted)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def _nearest_free_cell(requested: Cell, free_cells: Set[Cell]) -> Cell:
    """返回 requested 本身或欧氏格距最近的自由格。"""
    if requested in free_cells:
        return requested
    return min(
        free_cells,
        key=lambda cell: (
            (cell[0] - requested[0]) ** 2 + (cell[1] - requested[1]) ** 2,
            cell,
        ),
    )


def _reachable_free_distances(
    seed: Cell,
    free_cells: Set[Cell],
    maximum_steps: int,
) -> Dict[Cell, int]:
    """用四邻接 BFS 返回不超过 maximum_steps 的可达自由格步数。"""
    distances = {seed: 0}
    queue = deque([seed])
    while queue:
        row, col = queue.popleft()
        if distances[(row, col)] >= maximum_steps:
            continue
        for neighbor in _four_neighbors(row, col):
            if neighbor in free_cells and neighbor not in distances:
                distances[neighbor] = distances[(row, col)] + 1
                queue.append(neighbor)
    return distances


def _find_frontier_cells(grid: GridValues, reachable: Set[Cell]) -> Set[Cell]:
    """返回八邻域接触未知格的可达自由格。"""
    height, width = len(grid), len(grid[0])
    result = set()
    for row, col in reachable:
        if any(
            grid[near_row][near_col] is None
            for near_row, near_col in _eight_neighbors(row, col)
            if 0 <= near_row < height and 0 <= near_col < width
        ):
            result.add((row, col))
    return result


def _find_range_frontier_cells(
    reachable: Set[Cell], free_cells: Set[Cell]
) -> Set[Cell]:
    """返回局部 BFS 边界上、外侧仍邻接完整地图自由格的格子。"""
    return {
        (row, col)
        for row, col in reachable
        if any(
            neighbor in free_cells and neighbor not in reachable
            for neighbor in _four_neighbors(row, col)
        )
    }


def _group_cells_by_direction(
    cells: Set[Cell],
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    reference_heading: float,
) -> Tuple[Tuple[float, Set[Cell]], ...]:
    """把范围边界按参考方向及其三个正交方向分成四组。"""
    headings = tuple(
        reference_heading + index * math.pi / 2.0 for index in range(4)
    )
    groups = [set() for _ in headings]
    for cell in cells:
        cell_heading = _cell_heading(cell, obstacle_map, pose)
        group_index = min(
            range(len(headings)),
            key=lambda index: (
                abs(_angle_difference(cell_heading, headings[index])),
                index,
            ),
        )
        groups[group_index].add(cell)
    return tuple(
        (headings[index], group)
        for index, group in enumerate(groups)
        if group
    )


def _cell_nearest_heading(
    cells: Set[Cell],
    heading: float,
    obstacle_map: ObstacleMap,
    pose: Pose2D,
) -> Cell:
    """返回一组格子中最接近指定世界航向的真实格。"""
    return min(
        cells,
        key=lambda cell: (
            abs(_angle_difference(_cell_heading(cell, obstacle_map, pose), heading)),
            cell,
        ),
    )


def _cell_heading(
    cell: Cell, obstacle_map: ObstacleMap, pose: Pose2D
) -> float:
    world_xy = grid_cell_center_to_world(cell[0], cell[1], obstacle_map)
    return math.atan2(world_xy[1] - pose.y_m, world_xy[0] - pose.x_m)


def _connected_components(cells: Set[Cell]) -> Tuple[Set[Cell], ...]:
    """按八邻接拆分 Frontier 连通段。"""
    remaining = set(cells)
    components = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        remaining.remove(seed)
        queue = deque([seed])
        while queue:
            row, col = queue.popleft()
            for neighbor in _eight_neighbors(row, col):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return tuple(components)


def _component_midpoint(component: Set[Cell]) -> Cell:
    """返回连通段中最接近行列质心的真实格。"""
    center_row = sum(cell[0] for cell in component) / len(component)
    center_col = sum(cell[1] for cell in component) / len(component)
    return min(
        component,
        key=lambda cell: (
            (cell[0] - center_row) ** 2 + (cell[1] - center_col) ** 2,
            cell,
        ),
    )


def _normalize_points(
    values: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    """校验用于候选抑制的世界坐标。"""
    points = []
    try:
        iterator = iter(values)
    except TypeError:
        raise ValueError("excluded_world_xy must contain finite points") from None
    for value in iterator:
        try:
            x_m, y_m = float(value[0]), float(value[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError("excluded_world_xy must contain finite points") from None
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            raise ValueError("excluded_world_xy must contain finite points")
        points.append((x_m, y_m))
    return tuple(points)


def _is_excluded(
    world_xy: Tuple[float, float],
    excluded_points: Sequence[Tuple[float, float]],
    radius_m: float,
) -> bool:
    return any(
        math.hypot(world_xy[0] - point[0], world_xy[1] - point[1]) <= radius_m
        for point in excluded_points
    )


def _four_neighbors(row: int, col: int) -> Tuple[Cell, ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def _eight_neighbors(row: int, col: int) -> Tuple[Cell, ...]:
    return tuple(
        (row + row_offset, col + col_offset)
        for row_offset in (-1, 0, 1)
        for col_offset in (-1, 0, 1)
        if row_offset != 0 or col_offset != 0
    )


def _optional_heading(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not _is_finite(value):
        raise ValueError("preferred_heading_world_rad must be finite")
    return float(value)


def _positive_finite(value: float, name: str) -> float:
    if not _is_finite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _non_negative_finite(value: float, name: str) -> float:
    if not _is_finite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)


def _is_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "PATH_DISTANCE_SCORE_WEIGHT",
    "PREFERRED_HEADING_SCORE_WEIGHT",
    "find_frontier_candidates",
]
