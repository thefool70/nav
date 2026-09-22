"""Frontier 区域刷新与候选记录读取。

搜索核心的三个行为（扫描补查、探索选择、回退恢复）都依赖同一份 Frontier
提取与区域关联规则，集中放在这里，避免各行为各自实现一遍。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

from .frontier import (
    PATH_DISTANCE_SCORE_WEIGHT,
    FrameFrontierCache,
    FrontierExtraction,
    extract_frame_frontiers,
)
from .history import filter_blocked_frontier_regions, match_frontier_regions
from .models import (
    FrontierCandidate,
    FrontierRegion,
    NavigationFrame,
    SearchDirectionState,
    SearchState,
)
from .timing import TimingSpans, measure_stage


def refresh_frontier_regions(
    frame: NavigationFrame,
    state: SearchState,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> Tuple[SearchState, FrontierExtraction]:
    """保留完整扫描边界，只对移动候选应用区域屏蔽与稳定 ID 关联。"""
    with measure_stage(timings, "frontier.extract"):
        extraction = extract_frame_frontiers(
            frame, tried_candidate_points(state.observation_history),
            cache=frontier_cache, timings=timings,
        )
    with measure_stage(timings, "frontier.match_regions"):
        candidates = extraction.candidates
        candidates, blocked_regions = filter_blocked_frontier_regions(
            frame.obstacle_map, candidates, state.blocked_frontier_regions,
        )
        candidates, regions, next_id = match_frontier_regions(
            frame.obstacle_map, candidates, state.frontier_regions,
            state.next_frontier_region_id,
        )
    active_id = state.active_frontier_id
    if not any(region.region_id == active_id for region in regions):
        active_id = None
    return replace(
        state, frontier_regions=regions, next_frontier_region_id=next_id,
        active_frontier_id=active_id,
        blocked_frontier_regions=blocked_regions,
        frontier_hole_filter_applied=extraction.hole_filter_applied,
        ignored_frontier_hole_count=extraction.ignored_hole_count,
        ignored_frontier_hole_area_m2=extraction.ignored_hole_area_m2,
        ignored_frontier_cell_count=extraction.ignored_frontier_cell_count,
    ), replace(extraction, candidates=candidates)


def scan_coverage_details(state: SearchState) -> Mapping[str, Any]:
    """保留本轮规划时的覆盖数量，首次图像同周期完成时也可在日志中查看。"""
    return {
        "local_observation_point_count": state.scan_local_point_count,
        "pending_observation_view_count": len(state.pending_observation_views),
        "blocked_frontier_region_count": len(state.blocked_frontier_regions),
        "observation_point_count": len(state.scan_observation_points),
        "reused_observation_point_count": max(
            0, state.scan_local_point_count - len(state.scan_observation_points)
        ),
    }


def frontier_candidate_debug(candidate: FrontierCandidate) -> Mapping[str, Any]:
    """拆开候选分数，供命令发送前的可选终端诊断使用。"""
    distance_penalty = PATH_DISTANCE_SCORE_WEIGHT * candidate.path_distance_m
    return {
        "candidate_id": candidate.candidate_id,
        "row": candidate.row,
        "col": candidate.col,
        "world_x_m": candidate.world_xy[0],
        "world_y_m": candidate.world_xy[1],
        "heading_world_rad": candidate.heading_world_rad,
        "frontier_cells": candidate.frontier_cells,
        "frontier_cell_count": candidate.frontier_cell_count,
        "frontier_span_m": candidate.frontier_span_m,
        "path_distance_m": candidate.path_distance_m,
        "distance_penalty": distance_penalty,
        "semantic_score": candidate.semantic_score,
        "semantic_bonus": (
            candidate.score - candidate.frontier_span_m + distance_penalty
        ),
        "deferred_order": candidate.deferred_order,
        "score": candidate.score,
    }


def tried_candidate_points(history) -> Tuple[Tuple[float, float], ...]:
    """屏蔽已完成或执行异常的目标位置，避免原地重复下发同一个探索任务。"""
    points = []
    for node in history:
        for direction in node.directions:
            if direction.state in (SearchDirectionState.INVALIDATED, SearchDirectionState.STALLED):
                point = direction.candidate_world_xy
            elif direction.state is SearchDirectionState.EXPLORED:
                point = direction.command_world_xy
            else:
                point = None
            if point is not None:
                points.append(point)
    return tuple(points)


def block_region(regions, region_id: str):
    """返回把指定区域整片屏蔽后的区域序列（区域本身不再参与候选关联）。"""
    return tuple(
        region for region in regions if region.region_id != region_id
    )


__all__ = [
    "block_region",
    "frontier_candidate_debug",
    "refresh_frontier_regions",
    "scan_coverage_details",
    "tried_candidate_points",
]
