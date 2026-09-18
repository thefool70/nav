"""从占用栅格提取可达 Frontier，并结合几何与语义分数排序。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .geometry import grid_cell_center_to_world, world_to_nearest_grid_cell
from .models import FrontierCandidate, ObstacleMap, Pose2D


Cell = Tuple[int, int]
GridValues = Tuple[Tuple[Optional[float], ...], ...]

PATH_DISTANCE_SCORE_WEIGHT = 0.05
SEMANTIC_SCORE_WEIGHT = 1.5
FRONTIER_CLEARANCE_SEARCH_M = 0.75
FRONTIER_FRAGMENT_GAP_M = 0.30
MAX_UNKNOWN_HOLE_AREA_M2 = 0.05


@dataclass(frozen=True)
class FrontierExtraction:
    """有效候选与本次提取的小孔洞过滤统计；不修改输入地图。"""

    candidates: Tuple[FrontierCandidate, ...] = ()
    hole_filter_applied: bool = False
    ignored_hole_count: int = 0
    ignored_hole_area_m2: float = 0.0
    ignored_frontier_cell_count: int = 0


def is_world_point_reachable(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    world_xy: Tuple[float, float],
) -> bool:
    """目标格属于机器人当前四邻接 BFS 可达自由区时返回 True。"""
    target = world_to_nearest_grid_cell(world_xy, obstacle_map)
    return target in reachable_free_distances(obstacle_map, pose)


def reachable_free_distances(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    *,
    clearance_m: float = 0.0,
) -> Dict[Cell, int]:
    """返回可达自由格到机器人所在自由区起点的步数；原始地图可指定净空半径。"""
    grid = _normalize_grid(obstacle_map)
    resolution = _positive_finite(obstacle_map.resolution_m, "resolution_m")
    clearance_m = _non_negative_finite(clearance_m, "clearance_m")
    free_cells = _free_cells(grid)
    if clearance_m > 0.0:
        steps = int(math.ceil(clearance_m / resolution))
        offsets = tuple(
            (dr, dc) for dr in range(-steps, steps + 1) for dc in range(-steps, steps + 1)
            if math.hypot(dr, dc) * resolution <= clearance_m
        )
        for row, values in enumerate(grid):
            for col, value in enumerate(values):
                if value is not None and value > 0.5:
                    for dr, dc in offsets:
                        free_cells.discard((row + dr, col + dc))
    if not free_cells:
        return {}

    requested_seed = world_to_nearest_grid_cell(
        (pose.x_m, pose.y_m), obstacle_map
    )
    seed = _nearest_free_cell(requested_seed, free_cells)
    return _reachable_free_distances(seed, free_cells)


def extract_frontiers(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    semantic_scores: Optional[Mapping[str, float]] = None,
    excluded_world_xy: Sequence[Tuple[float, float]] = (),
    min_frontier_span_m: float = 0.5,
    min_goal_distance_m: float = 0.35,
    *,
    visibility_map: Optional[ObstacleMap] = None,
    max_unknown_hole_area_m2: float = MAX_UNKNOWN_HOLE_AREA_M2,
) -> FrontierExtraction:
    """提取可达候选并返回过滤统计；仅用同格网未膨胀图判断小孔洞。"""
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
    hole_area_limit = _non_negative_finite(max_unknown_hole_area_m2, "max_unknown_hole_area_m2")

    free_cells = _free_cells(grid)
    if not free_cells:
        return FrontierExtraction()

    requested_seed = world_to_nearest_grid_cell(
        (pose.x_m, pose.y_m), obstacle_map
    )
    seed = _nearest_free_cell(requested_seed, free_cells)
    reachable_distance = _reachable_free_distances(seed, free_cells)
    reachable_cells = set(reachable_distance)
    frontier_cells = _find_frontier_cells(grid, reachable_cells)
    ignored_unknown = set()
    hole_sizes = ()
    filter_applied = False
    original_frontier_count = len(frontier_cells)
    # 未膨胀图与探索图同格网时才分类，避免膨胀切断未知区域后误判为小孔洞。
    if visibility_map is not None and hole_area_limit > 0.0:
        if not isinstance(visibility_map, ObstacleMap):
            raise ValueError("visibility_map must be an ObstacleMap")
        if (
            visibility_map.frame_id == obstacle_map.frame_id
            and visibility_map.origin == obstacle_map.origin
            and visibility_map.resolution_m == obstacle_map.resolution_m
            and len(visibility_map.occupancy) == len(grid)
            and all(len(row) == len(grid[0]) for row in visibility_map.occupancy)
        ):
            raw_grid = _normalize_grid(visibility_map)
            seeds = {
                neighbor for row, col in frontier_cells
                for neighbor in _eight_neighbors(row, col)
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])
                and grid[neighbor[0]][neighbor[1]] is None
            }
            ignored_unknown, hole_sizes = _small_unknown_holes(raw_grid, seeds, resolution, hole_area_limit)
            frontier_cells = _find_frontier_cells(grid, reachable_cells, ignored_unknown)
            filter_applied = True

    clearance_search_steps = max(
        1,
        int(math.ceil(FRONTIER_CLEARANCE_SEARCH_M / resolution)),
    )
    candidate_groups = tuple(
        component
        for component in _merge_frontier_fragments(frontier_cells, grid, resolution, ignored_unknown)
        if _frontier_span_m(component, resolution) >= minimum_span
    )

    excluded_radius = max(0.4, 2.0 * resolution)
    candidates = []
    for cells in candidate_groups:
        frontier_span = _frontier_span_m(cells, resolution)
        eligible_cells = {
            cell for cell in cells
            if reachable_distance[cell] * resolution >= minimum_distance
            and not _is_excluded(
                grid_cell_center_to_world(*cell, obstacle_map),
                excluded_points,
                excluded_radius,
            )
        }
        if not eligible_cells:
            continue
        row, col = _safest_frontier_cell(
            eligible_cells,
            grid,
            reachable_distance,
            clearance_search_steps,
        )
        path_distance = reachable_distance[(row, col)] * resolution
        world_xy = grid_cell_center_to_world(row, col, obstacle_map)
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
    return FrontierExtraction(
        candidates=tuple(candidates), hole_filter_applied=filter_applied,
        ignored_hole_count=len(hole_sizes),
        ignored_hole_area_m2=sum(hole_sizes) * resolution * resolution,
        ignored_frontier_cell_count=original_frontier_count - len(frontier_cells),
    )


def _small_unknown_holes(
    grid: GridValues, seeds: Set[Cell], resolution_m: float, max_area_m2: float,
) -> Tuple[Set[Cell], Tuple[int, ...]]:
    """只从候选邻接未知格开始八邻接搜索；超面积或连到图边即保留，不遍历整片外部未知区。"""
    height, width = len(grid), len(grid[0])
    cell_area = resolution_m * resolution_m
    ignored, preserved = set(), set()
    sizes = []
    for seed in sorted(seeds):
        if seed in ignored or seed in preserved or grid[seed[0]][seed[1]] is not None:
            continue
        component = {seed}
        queue = deque([seed])
        small_closed = cell_area <= max_area_m2
        while queue and small_closed:
            row, col = queue.popleft()
            if row in (0, height - 1) or col in (0, width - 1):
                small_closed = False
                break
            for neighbor in _eight_neighbors(row, col):
                # 当前格不在图边，八邻格均在图内。
                if grid[neighbor[0]][neighbor[1]] is not None or neighbor in component:
                    continue
                if neighbor in preserved:
                    small_closed = False
                    break
                component.add(neighbor)
                if len(component) * cell_area > max_area_m2:
                    small_closed = False
                    break
                queue.append(neighbor)
        if small_closed:
            ignored.update(component)
            sizes.append(len(component))
        else:
            preserved.update(component)
    return ignored, tuple(sizes)


def _merge_frontier_fragments(
    cells: Set[Cell], grid: GridValues, resolution: float,
    ignored_unknown: Set[Cell],
) -> Tuple[Set[Cell], ...]:
    """合并未知侧朝向相近、自由区短路径不超过 0.30 m 的断段，不穿越障碍。"""
    components = _connected_components(cells)
    owners = {cell: index for index, part in enumerate(components) for cell in part}
    normals = tuple(_unknown_side_normal(part, grid, ignored_unknown) for part in components)
    free = _free_cells(grid)
    steps = int(FRONTIER_FRAGMENT_GAP_M / resolution)
    links = set()
    for index, part in enumerate(components):
        visited = set(part)
        queue = deque((cell, 0) for cell in sorted(part))
        while queue:
            cell, distance = queue.popleft()
            other = owners.get(cell, index)
            if other > index:
                first, second = normals[index], normals[other]
                if first[0] * second[0] + first[1] * second[1] >= math.cos(math.pi / 4):
                    links.add((index, other))
            if distance >= steps:
                continue
            for neighbor in _four_neighbors(*cell):
                if neighbor in free and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

    groups = {index: set(part) for index, part in enumerate(components)}
    parents = list(range(len(components)))
    for first, second in sorted(links):
        while parents[first] != first:
            first = parents[first]
        while parents[second] != second:
            second = parents[second]
        if first == second:
            continue
        groups[first].update(groups[second])
        del groups[second]
        parents[second] = first
    return tuple(groups[index] for index in sorted(groups))


def _unknown_side_normal(
    component: Set[Cell], grid: GridValues, ignored_unknown: Set[Cell],
) -> Tuple[float, float]:
    """用邻接未知格方向的均值区分边界朝向；方向不明确时不跨断口合并。"""
    dr_sum = dc_sum = 0
    for row, col in component:
        for near_row, near_col in _eight_neighbors(row, col):
            if (
                0 <= near_row < len(grid) and 0 <= near_col < len(grid[0])
                and grid[near_row][near_col] is None
                and (near_row, near_col) not in ignored_unknown
            ):
                dr_sum += near_row - row
                dc_sum += near_col - col
    magnitude = math.hypot(dr_sum, dc_sum)
    return (dr_sum / magnitude, dc_sum / magnitude) if magnitude else (0.0, 0.0)


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


def _find_frontier_cells(
    grid: GridValues, reachable: Set[Cell], ignored_unknown: Optional[Set[Cell]] = None,
) -> Set[Cell]:
    """返回八邻域接触未知格的可达自由格。"""
    height, width = len(grid), len(grid[0])
    ignored = ignored_unknown if ignored_unknown is not None else set()
    result = set()
    for row, col in reachable:
        if any(
            grid[near_row][near_col] is None
            and (near_row, near_col) not in ignored
            for near_row, near_col in _eight_neighbors(row, col)
            if 0 <= near_row < height and 0 <= near_col < width
        ):
            result.add((row, col))
    return result


def _connected_components(cells: Set[Cell]) -> Tuple[Set[Cell], ...]:
    """按八邻接提取完整的 Frontier 连通段。"""
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
    "extract_frontiers",
    "FrontierExtraction",
    "is_world_point_reachable",
    "reachable_free_distances",
]
