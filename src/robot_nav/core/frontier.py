"""从占用栅格提取可达 Frontier，并结合几何与语义分数排序。"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .geometry import grid_cell_center_to_world, world_to_nearest_grid_cell
from .models import FrontierCandidate, ObstacleMap, Pose2D


Cell = Tuple[int, int]
GridValues = Tuple[Tuple[Optional[float], ...], ...]

PATH_DISTANCE_SCORE_WEIGHT = 0.05
SEMANTIC_SCORE_WEIGHT = 1.5
FRONTIER_CLEARANCE_SEARCH_M = 0.75


def is_world_point_reachable(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    world_xy: Tuple[float, float],
) -> bool:
    """目标格属于机器人当前四邻接 BFS 可达自由区时返回 True。"""
    grid = _normalize_grid(obstacle_map)
    free_cells = _free_cells(grid)
    if not free_cells:
        return False

    requested_seed = world_to_nearest_grid_cell(
        (pose.x_m, pose.y_m), obstacle_map
    )
    seed = _nearest_free_cell(requested_seed, free_cells)
    target = world_to_nearest_grid_cell(world_xy, obstacle_map)
    return target in _reachable_free_distances(seed, free_cells)


def find_frontier_candidates(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    semantic_scores: Optional[Mapping[str, float]] = None,
    excluded_world_xy: Sequence[Tuple[float, float]] = (),
    min_frontier_span_m: float = 0.5,
    min_goal_distance_m: float = 0.35,
) -> Tuple[FrontierCandidate, ...]:
    """返回可达 Frontier，并优先选择远离占据格的聚类代表点。"""
    grid = _normalize_grid(obstacle_map)
    resolution = _positive_finite(obstacle_map.resolution_m, "resolution_m")
    minimum_span = _non_negative_finite(
        min_frontier_span_m, "min_frontier_span_m"
    )
    minimum_distance = _non_negative_finite(
        min_goal_distance_m, "min_goal_distance_m"
    )
    normalized_scores = _normalize_semantic_scores(semantic_scores)
    excluded_points = _normalize_points(excluded_world_xy)

    free_cells = _free_cells(grid)
    if not free_cells:
        return ()

    requested_seed = world_to_nearest_grid_cell(
        (pose.x_m, pose.y_m), obstacle_map
    )
    seed = _nearest_free_cell(requested_seed, free_cells)
    reachable_distance = _reachable_free_distances(seed, free_cells)
    reachable_cells = set(reachable_distance)
    frontier_cells = _find_frontier_cells(grid, reachable_cells)

    clearance_search_steps = max(
        1,
        int(math.ceil(FRONTIER_CLEARANCE_SEARCH_M / resolution)),
    )
    candidate_groups = tuple(
        component
        for component in _connected_components(frontier_cells)
        if _frontier_span_m(component, resolution) >= minimum_span
    )

    excluded_radius = max(0.4, 2.0 * resolution)
    candidates = []
    for cells in candidate_groups:
        frontier_span = _frontier_span_m(cells, resolution)
        row, col = _safest_frontier_cell(
            cells,
            grid,
            reachable_distance,
            clearance_search_steps,
        )
        path_distance = reachable_distance[(row, col)] * resolution
        if path_distance < minimum_distance:
            continue
        world_xy = grid_cell_center_to_world(row, col, obstacle_map)
        if _is_excluded(world_xy, excluded_points, excluded_radius):
            continue
        candidate_id = f"frontier:{row}:{col}"
        heading = math.atan2(world_xy[1] - pose.y_m, world_xy[0] - pose.x_m)
        score = frontier_span - PATH_DISTANCE_SCORE_WEIGHT * path_distance
        semantic_score = normalized_scores.get(candidate_id)
        if semantic_score is not None:
            score += SEMANTIC_SCORE_WEIGHT * (2.0 * semantic_score - 1.0)
        candidates.append(
            FrontierCandidate(
                candidate_id=candidate_id,
                row=row,
                col=col,
                world_xy=world_xy,
                heading_world_rad=heading,
                frontier_cells=tuple(sorted(cells)),
                frontier_cell_count=len(cells),
                frontier_span_m=frontier_span,
                path_distance_m=path_distance,
                score=score,
                semantic_score=semantic_score,
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


def _frontier_span_m(component: Set[Cell], resolution_m: float) -> float:
    """返回 Frontier 聚类所占完整栅格包围框的对角跨度。"""
    rows = [cell[0] for cell in component]
    cols = [cell[1] for cell in component]
    return math.hypot(
        max(rows) - min(rows) + 1,
        max(cols) - min(cols) + 1,
    ) * resolution_m


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


def _free_cells(grid: GridValues) -> Set[Cell]:
    """返回占据图中的全部已知自由格。"""
    return {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value is not None and value <= 0.5
    }


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
) -> Dict[Cell, int]:
    """用四邻接 BFS 返回当前有效地图中全部可达自由格步数。"""
    distances = {seed: 0}
    queue = deque([seed])
    while queue:
        row, col = queue.popleft()
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


def _safest_frontier_cell(
    component: Set[Cell],
    grid: GridValues,
    reachable_distance: Mapping[Cell, int],
    clearance_search_steps: int,
) -> Cell:
    """优先选择远离占据格、其次靠近聚类质心的 Frontier 格。"""
    center_row = sum(cell[0] for cell in component) / len(component)
    center_col = sum(cell[1] for cell in component) / len(component)
    return min(
        component,
        key=lambda cell: (
            -_occupied_clearance_score(
                cell,
                grid,
                clearance_search_steps,
            ),
            (cell[0] - center_row) ** 2 + (cell[1] - center_col) ** 2,
            reachable_distance[cell],
            cell,
        ),
    )


def _occupied_clearance_score(
    cell: Cell,
    grid: GridValues,
    search_steps: int,
) -> int:
    """返回到最近占据格的平方格距，搜索范围外统一视为更安全。"""
    row, col = cell
    height, width = len(grid), len(grid[0])
    maximum_distance_squared = search_steps * search_steps
    nearest_distance_squared: Optional[int] = None
    for near_row in range(
        max(0, row - search_steps),
        min(height, row + search_steps + 1),
    ):
        for near_col in range(
            max(0, col - search_steps),
            min(width, col + search_steps + 1),
        ):
            value = grid[near_row][near_col]
            if value is None or value <= 0.5:
                continue
            distance_squared = (near_row - row) ** 2 + (near_col - col) ** 2
            if distance_squared > maximum_distance_squared:
                continue
            if (
                nearest_distance_squared is None
                or distance_squared < nearest_distance_squared
            ):
                nearest_distance_squared = distance_squared
    if nearest_distance_squared is None:
        return (search_steps + 1) ** 2
    return nearest_distance_squared


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


def _normalize_semantic_scores(
    values: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """校验 Frontier ID 到 0-1 语义分数的映射。"""
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("semantic_scores must be a mapping or None")
    result = {}
    for candidate_id, raw_score in values.items():
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("semantic_scores keys must be non-empty strings")
        if not _is_finite(raw_score) or not 0.0 <= float(raw_score) <= 1.0:
            raise ValueError("semantic_scores values must be between 0 and 1")
        result[candidate_id] = float(raw_score)
    return result


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


__all__ = [
    "PATH_DISTANCE_SCORE_WEIGHT",
    "SEMANTIC_SCORE_WEIGHT",
    "find_frontier_candidates",
    "is_world_point_reachable",
]
