"""探索历史管理：跨帧区域关联、移动记录和执行状态更新。

约定：观测节点按输入顺序保存（输入顺序即时间顺序），方向在节点内保持输入
顺序；所有返回对象均为不可变新对象。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Sequence, Set, Tuple

from .geometry import grid_cell_center_to_world, world_to_nearest_grid_cell, wrap_angle
from .models import (
    BlockedFrontierRegion,
    FrontierCandidate,
    FrontierRegion,
    ObservationNode,
    ObstacleMap,
    SearchDirection,
    SearchDirectionState,
)


def filter_blocked_frontier_regions(
    obstacle_map: ObstacleMap,
    candidates: Tuple[FrontierCandidate, ...],
    blocked_regions: Tuple[BlockedFrontierRegion, ...],
) -> Tuple[Tuple[FrontierCandidate, ...], Tuple[BlockedFrontierRegion, ...]]:
    """整片过滤被未知路径拒绝的边界，本次运行中持续保留屏蔽。

    按世界边界匹配，不依赖代表点或 region ID。记住匹配过的完整边界，使分裂、
    合并、一格移动及短暂消失不会遗忘屏蔽；已知障碍之间不做邻域匹配。
    """
    candidate_cells = tuple(set(candidate.frontier_cells) for candidate in candidates)
    excluded_indices = set()
    retained = []
    for blocked in blocked_regions:
        _, neighborhood = _boundary_cells_and_neighborhood(
            obstacle_map, blocked.boundary_world_xy,
        )
        boundary = set(blocked.boundary_world_xy)
        for index, cells in enumerate(candidate_cells):
            if cells & neighborhood:
                excluded_indices.add(index)
                boundary.update(
                    grid_cell_center_to_world(row, col, obstacle_map)
                    for row, col in cells
                )
        retained.append(replace(blocked, boundary_world_xy=tuple(sorted(boundary))))
    return (
        tuple(candidate for index, candidate in enumerate(candidates)
              if index not in excluded_indices),
        tuple(retained),
    )


def match_frontier_regions(
    obstacle_map: ObstacleMap,
    candidates: Tuple[FrontierCandidate, ...],
    previous: Tuple[FrontierRegion, ...],
    next_region_id: int,
) -> Tuple[Tuple[FrontierCandidate, ...], Tuple[FrontierRegion, ...], int]:
    """按世界边界重叠关联区域；分裂时最大重叠部分继承 ID，其余分配新 ID。

    允许边界在自由格内移动一个栅格，但不跨越已知障碍。分裂出的旧方向继承暂存
    顺序，合并时保留重叠旧方向中最先应恢复的顺序；消失的区域不再参与移动。
    """
    old_cells = []
    old_neighborhoods = []
    for region in previous:
        cells, neighborhood = _boundary_cells_and_neighborhood(
            obstacle_map, region.boundary_world_xy,
        )
        old_cells.append(cells)
        old_neighborhoods.append(neighborhood)

    matches = []
    for new_index, candidate in enumerate(candidates):
        cells = set(candidate.frontier_cells)
        for old_index, neighborhood in enumerate(old_neighborhoods):
            overlap = len(cells & neighborhood)
            if overlap:
                exact = len(cells & old_cells[old_index])
                matches.append((-exact, -overlap, new_index, old_index))
    assignments = {}
    used_old = set()
    for _, _, new_index, old_index in sorted(matches):
        if new_index not in assignments and old_index not in used_old:
            assignments[new_index] = previous[old_index].region_id
            used_old.add(old_index)

    updated = []
    regions = []
    for index, candidate in enumerate(candidates):
        region_id = assignments.get(index)
        if region_id is None:
            region_id = f"region:{next_region_id}"
            next_region_id += 1
        deferred_order = _inherited_deferred_order(index, matches, previous)
        updated.append(replace(
            candidate, candidate_id=region_id, deferred_order=deferred_order,
        ))
        regions.append(FrontierRegion(
            region_id=region_id,
            boundary_world_xy=tuple(
                grid_cell_center_to_world(row, col, obstacle_map)
                for row, col in candidate.frontier_cells
            ),
            deferred_order=deferred_order,
        ))
    return tuple(updated), tuple(regions), next_region_id


def _boundary_cells_and_neighborhood(
    obstacle_map: ObstacleMap,
    boundary_world_xy: Tuple[Tuple[float, float], ...],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """把世界边界投到当前地图，并沿自由格扩展一格，供关联与屏蔽共用。"""
    grid = obstacle_map.occupancy
    height, width = len(grid), len(grid[0])
    cells = {world_to_nearest_grid_cell(point, obstacle_map) for point in boundary_world_xy}
    neighborhood = set()
    for row, col in cells:
        if not _is_free_grid_cell(grid, row, col, height, width):
            continue
        neighborhood.add((row, col))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if _is_free_grid_cell(grid, row + dr, col + dc, height, width):
                neighborhood.add((row + dr, col + dc))
    return cells, neighborhood


def _inherited_deferred_order(
    candidate_index: int,
    matches: Sequence[Tuple[int, int, int, int]],
    previous: Tuple[FrontierRegion, ...],
) -> Optional[Tuple[int, int]]:
    """优先实际重叠；matches 使用（负实际重叠数，负邻域重叠数，新序号，旧序号）。"""
    parents = [item for item in matches if item[2] == candidate_index]
    exact_parents = [item for item in parents if item[0] < 0]
    if exact_parents:
        parents = exact_parents
    elif parents:
        best_overlap = min(item[1] for item in parents)
        parents = [item for item in parents if item[1] == best_overlap]
    orders = [
        previous[old_index].deferred_order
        for _, _, _, old_index in parents
        if previous[old_index].deferred_order is not None
    ]
    return min(orders, key=lambda order: (-order[0], order[1])) if orders else None


def defer_unselected_frontiers(
    regions: Tuple[FrontierRegion, ...],
    new_candidates: Tuple[FrontierCandidate, ...],
    selected_id: str,
    observation_index: int,
) -> Tuple[FrontierRegion, ...]:
    """选定一次移动后，按本轮评分顺序暂存其他新方向；旧方向保持原顺序。"""
    orders = {
        candidate.candidate_id: (observation_index, rank)
        for rank, candidate in enumerate(new_candidates)
        if candidate.candidate_id != selected_id
    }
    return tuple(
        replace(
            region,
            deferred_order=(
                None if region.region_id == selected_id
                else orders.get(region.region_id, region.deferred_order)
            ),
        )
        for region in regions
    )


def _is_free_grid_cell(grid, row: int, col: int, height: int, width: int) -> bool:
    return (
        0 <= row < height and 0 <= col < width
        and grid[row][col] is not None and grid[row][col] <= 0.5
    )


def _finite_number(value) -> Optional[float]:
    """value 为可转换的有限数（不含 bool）时返回 float(value)，否则 None。"""
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _require_world_xy(value, name) -> Tuple[float, float]:
    """校验 value 为两个有限数并返回规范化 float 二元组，否则抛 ValueError。"""
    try:
        xy = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be two finite numbers") from None
    if len(xy) != 2:
        raise ValueError(f"{name} must be two finite numbers")
    first = _finite_number(xy[0])
    second = _finite_number(xy[1])
    if first is None or second is None:
        raise ValueError(f"{name} must be two finite numbers")
    return (first, second)


def freeze_observation_node(
    node_id: str,
    position_world_xy: Tuple[float, float],
    directions: Sequence[SearchDirection],
    committed_direction_id: str,
) -> ObservationNode:
    """把一次观测固化为不可变 ObservationNode，方向保持输入顺序。

    node_id、direction_id 与 committed_direction_id 去除首尾空白后使用，
    空白字符串非法；heading_world_rad 转为有限 float 并归一化到 [-π, π)；
    candidate_world_xy、command_world_xy 若提供须为有限世界坐标，保存为 float 二元组。
    committed_direction_id 必须存在于 directions 中；对应方向标为 COMMITTED，
    其余方向标为 PENDING。
    """
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be a non-empty string")
    node_id = node_id.strip()
    position = _require_world_xy(position_world_xy, "position_world_xy")
    try:
        directions = tuple(directions)
    except TypeError:
        raise ValueError("directions must be a sequence of SearchDirection") from None
    if not all(isinstance(direction, SearchDirection) for direction in directions):
        raise ValueError("directions must contain only SearchDirection")
    if not isinstance(committed_direction_id, str):
        raise ValueError("committed_direction_id must be an existing direction_id")
    committed_direction_id = committed_direction_id.strip()
    direction_ids = []
    frozen_directions = []
    for direction in directions:
        if (
            not isinstance(direction.direction_id, str)
            or not direction.direction_id.strip()
        ):
            raise ValueError("direction_id must be a non-empty string")
        direction_id = direction.direction_id.strip()
        if direction_id in direction_ids:
            raise ValueError("direction_id must be unique")
        direction_ids.append(direction_id)
        heading = _finite_number(direction.heading_world_rad)
        if heading is None:
            raise ValueError("heading_world_rad must be finite")
        candidate = (
            _require_world_xy(direction.candidate_world_xy, "candidate_world_xy")
            if direction.candidate_world_xy is not None
            else None
        )
        command_point = (
            _require_world_xy(direction.command_world_xy, "command_world_xy")
            if direction.command_world_xy is not None
            else None
        )
        frozen_directions.append(
            SearchDirection(
                direction_id=direction_id,
                heading_world_rad=wrap_angle(heading),
                candidate_world_xy=candidate,
                command_world_xy=command_point,
                state=(
                    SearchDirectionState.COMMITTED
                    if direction_id == committed_direction_id
                    else SearchDirectionState.PENDING
                ),
            )
        )
    if committed_direction_id not in direction_ids:
        raise ValueError("committed_direction_id must be an existing direction_id")
    return ObservationNode(
        node_id=node_id,
        position_world_xy=position,
        directions=tuple(frozen_directions),
    )


def set_observation_direction_state(
    node: ObservationNode,
    direction_id: str,
    new_state: SearchDirectionState,
    execution_reason: Optional[str] = None,
) -> ObservationNode:
    """返回把 node 中 direction_id 方向状态替换为 new_state 后的新节点。

    其余方向保持原样与顺序；direction_id 不存在时抛 ValueError。
    """
    if not isinstance(node, ObservationNode):
        raise ValueError("node must be an ObservationNode")
    if not isinstance(direction_id, str) or not direction_id:
        raise ValueError("direction_id must be a non-empty string")
    if not isinstance(new_state, SearchDirectionState):
        raise ValueError("new_state must be a SearchDirectionState")
    if not any(
        direction.direction_id == direction_id for direction in node.directions
    ):
        raise ValueError("direction_id not found")
    return replace(
        node,
        directions=tuple(
            replace(
                direction,
                state=new_state,
                execution_reason=(
                    direction.execution_reason
                    if execution_reason is None else str(execution_reason)
                ),
            )
            if direction.direction_id == direction_id
            else direction
            for direction in node.directions
        ),
    )
