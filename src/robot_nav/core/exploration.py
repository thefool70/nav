"""探索行为：选择下一个 Frontier 停靠点，并处理该移动的可恢复失败。

同一行为的正常推进与可恢复失败放在一起：优先新 Frontier、其余方向暂存、
层进回退入口，以及未知路径超限时屏蔽整片区域、普通失败时淘汰目标点。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

from .frontier import PATH_DISTANCE_SCORE_WEIGHT, SEMANTIC_SCORE_WEIGHT, FrameFrontierCache
from .frontier_regions import frontier_candidate_debug, refresh_frontier_regions, scan_coverage_details
from .geometry import world_point_to_robot
from .history import defer_unselected_frontiers, freeze_observation_node, set_observation_direction_state
from .models import (
    ActionConstraint,
    ActionKind,
    ActionPurpose,
    BlockedFrontierRegion,
    FrontierCandidate,
    FrontierScoreRequest,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObservationNode,
    Pose2D,
    RelativePoseCommand,
    SearchDirection,
    SearchDirectionState,
    SearchPhase,
    SearchState,
)
from .navigation_io import make_action, result
from .scan import shortest_turn_to_heading
from .timing import TimingSpans, measure_stage


def select_exploration_target(
    frame: NavigationFrame,
    state: SearchState,
    frontier_scores: Optional[Mapping[str, float]],
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """优先选择新 Frontier；新候选耗尽时沿当前分支逐个返回父节点。"""
    explored_state = mark_latest_committed_explored(state)
    explored_state, candidates = refresh_frontier_regions(frame, explored_state,
        timings=timings, frontier_cache=frontier_cache,
    )
    if not candidates:
        from .backtracking import begin_backtracking

        return begin_backtracking(frame, explored_state, (), timings=timings, frontier_cache=frontier_cache)

    with measure_stage(timings, "frontier.select_rank"):
        new_candidates = []
        deferred_candidates = []
        for candidate in candidates:
            if candidate.deferred_order is not None:
                deferred_candidates.append(candidate)
                continue
            semantic_score = (frontier_scores or {}).get(candidate.candidate_id)
            semantic_bonus = (
                0.0 if semantic_score is None
                else SEMANTIC_SCORE_WEIGHT * (2.0 * semantic_score - 1.0)
            )
            new_candidates.append(replace(
                candidate,
                semantic_score=semantic_score,
                score=candidate.score + semantic_bonus,
            ))
        ranked_new = tuple(sorted(
            new_candidates,
            key=lambda item: (-item.score, item.path_distance_m, item.candidate_id),
        ))
        # 暂存节点后进先出，同一节点沿用原排序；随机 / VLM 新分数不能插队。
        deferred_candidates.sort(key=lambda item: (
            -item.deferred_order[0], item.deferred_order[1], item.candidate_id,
        ))
        candidates = ranked_new + tuple(deferred_candidates)
    if not ranked_new:
        from .backtracking import begin_backtracking

        return begin_backtracking(frame, explored_state, candidates,
            timings=timings, frontier_cache=frontier_cache,
        )
    details = tuple(frontier_candidate_debug(candidate) for candidate in candidates)
    selection_details = {
        "frontier_selection_source": "new" if ranked_new else "deferred",
        "new_frontier_count": len(ranked_new),
        "deferred_frontier_count": len(deferred_candidates),
    }
    if frontier_scores is None and len(ranked_new) > 1 and (
        explored_state.scan_evidence or explored_state.asynchronous_perception
    ):
        return result(
            NavigationStatus.NEEDS_FRONTIER_SCORES,
            replace(explored_state, phase=SearchPhase.EXPLORING),
            "frontier.score", "只对新 Frontier 评分，旧方向保持暂存顺序。",
            details={
                **selection_details,
                "candidate_count": len(candidates), "frontier_candidates": details,
            },
            frontier_score_request=FrontierScoreRequest(ranked_new),
        )

    with measure_stage(timings, "frontier.commit"):
        return commit_frontier_move(
            frame, explored_state, candidates, ranked_new, frontier_scores,
        )


def commit_frontier_move(
    frame: NavigationFrame,
    state: SearchState,
    candidates: Tuple[FrontierCandidate, ...],
    ranked_new: Tuple[FrontierCandidate, ...],
    frontier_scores: Optional[Mapping[str, float]],
    *,
    parent_node_id: Optional[str] = None,
) -> NavigationResult:
    """记录并提交排在首位的 Frontier；返回父节点本身不消耗探索方向。"""
    selected = candidates[0]
    destination = selected.world_xy
    # 只有实际出发探索时才创建历史；暂存序号指向创建时的父节点位置。
    direction = SearchDirection(
        direction_id=selected.candidate_id,
        heading_world_rad=selected.heading_world_rad,
        candidate_world_xy=selected.world_xy,
        command_world_xy=destination,
    )
    node = freeze_observation_node(
        f"observation:{len(state.observation_history)}",
        (frame.pose.x_m, frame.pose.y_m), (direction,), direction.direction_id,
    )
    from .scan_behavior import reset_scan_after_move

    next_state = reset_scan_after_move(replace(
        state,
        observation_history=state.observation_history + (node,),
        branch_node_ids=state.branch_node_ids + (node.node_id,),
        frontier_regions=defer_unselected_frontiers(
            state.frontier_regions, ranked_new, selected.candidate_id,
            len(state.observation_history),
        ),
        active_frontier_id=selected.candidate_id,
    ))
    return result(
        NavigationStatus.OK, next_state,
        "backtrack.resume" if parent_node_id is not None else "explore.select",
        (
            "优先探索新 Frontier，其余新方向暂存；等待完整移动结束后再决策。"
            if ranked_new
            else "已回到父节点，按保存顺序恢复该节点仍有效的探索方向。"
        ),
        make_action(
            ActionKind.MOVE_TO_POSE,
            destination=Pose2D(destination[0], destination[1], math.atan2(
                destination[1] - frame.pose.y_m, destination[0] - frame.pose.x_m)),
            constraint=ActionConstraint.REQUIRE_KNOWN_PATH,
            purpose=ActionPurpose.RESUME if parent_node_id is not None else ActionPurpose.EXPLORE,
            candidate_id=selected.candidate_id,
            node_id=node.node_id,
        ),
        {
            **scan_coverage_details(state),
            "frontier_selection_source": "new" if ranked_new else "deferred",
            "new_frontier_count": sum(candidate.deferred_order is None for candidate in candidates),
            "deferred_frontier_count": sum(candidate.deferred_order is not None for candidate in candidates),
            "candidate_id": selected.candidate_id,
            "node_id": node.node_id,
            "parent_node_id": parent_node_id,
            "candidate_count": len(candidates),
            "candidate_score": selected.score,
            "frontier_scoring_skipped": frontier_scores is None,
            "frontier_scoring_skip_reason": (
                "restore_deferred" if not ranked_new
                else "single_new_candidate" if len(ranked_new) == 1
                else "no_new_scan_images" if not state.scan_evidence else None
            ),
            "frontier_path_distance_weight": PATH_DISTANCE_SCORE_WEIGHT,
            "frontier_semantic_score_weight": SEMANTIC_SCORE_WEIGHT,
            "frontier_candidates": tuple(frontier_candidate_debug(candidate) for candidate in candidates),
            "destination_world_xy": destination,
        },
    )


def reject_frontier_direction(
    state: SearchState,
    reason: str,
    *,
    candidate_id: str,
    node_id: Optional[str],
    blocked_regions: Tuple[BlockedFrontierRegion, ...] = (),
    rejected_path_world_xy: Tuple[Tuple[float, float], ...] = (),
) -> Optional[NavigationResult]:
    """探索移动失败：屏蔽整片区域或淘汰目标点后回到扫描。

    返回 None 表示该候选已不在当前分支中，调用方不应再处理这次失败。
    """
    for node in reversed(state.observation_history):
        if node_id is not None and node.node_id != node_id:
            continue
        if not any(
            direction.direction_id == candidate_id
            and direction.state is SearchDirectionState.COMMITTED
            for direction in node.directions
        ):
            continue
        updated_node = set_observation_direction_state(
            node,
            candidate_id,
            SearchDirectionState.INVALIDATED,
            execution_reason=reason,
        )
        from .scan_behavior import reset_scan_after_move

        next_state = reset_scan_after_move(replace(
            state,
            active_frontier_id=None,
            blocked_frontier_regions=blocked_regions,
            frontier_regions=(
                tuple(region for region in state.frontier_regions if region.region_id != candidate_id)
                if rejected_path_world_xy else state.frontier_regions
            ),
            observation_history=replace_history_node(state.observation_history, updated_node),
        ))
        return result(
            NavigationStatus.OK,
            next_state,
            "motion.frontier_rejected",
            (
                "路径中的未知长度超限，已屏蔽整个 Frontier 区域，本次运行中不再重试该区域。"
                if rejected_path_world_xy
                else "本次探索动作执行失败，先检查当前位置，再重新选择区域。"
            ),
            details={
                "direction_id": candidate_id,
                "reason": str(reason),
                "rejection_scope": "region" if rejected_path_world_xy else "point",
                "blocked_frontier_region_count": len(blocked_regions),
                "rejected_path_world_xy": rejected_path_world_xy,
            },
        )
    return None


def stall_frontier_direction(
    state: SearchState,
    reason: str,
    *,
    candidate_id: Optional[str],
) -> Optional[NavigationResult]:
    """探索移动停滞：按原地位置结束本次移动，标记方向为停滞并回到扫描。"""
    if state.phase is not SearchPhase.SCANNING:
        return None
    stalled_state = state
    for node in reversed(stalled_state.observation_history):
        if any(
            item.direction_id == candidate_id
            and item.state is SearchDirectionState.COMMITTED
            for item in node.directions
        ):
            updated = set_observation_direction_state(
                node, str(candidate_id), SearchDirectionState.STALLED,
                execution_reason=reason,
            )
            stalled_state = replace(
                stalled_state,
                active_frontier_id=None,
                observation_history=replace_history_node(
                    stalled_state.observation_history, updated
                ),
            )
            break
    return result(
        NavigationStatus.OK,
        stalled_state,
        "motion.stalled_continue",
        "移动连续静止达到门槛，按当前位置结束本次移动并继续扫描。",
        details={
            "direction_id": candidate_id,
            "reason": str(reason),
        },
    )


def release_interrupted_direction(state: SearchState, *, candidate_id: Optional[str], node_id: Optional[str]) -> SearchState:
    """Frontier 移动被感知中断时，把未到达的方向恢复为待探索。"""
    if not isinstance(candidate_id, str) or not candidate_id:
        return state
    for node in reversed(state.observation_history):
        if node_id is not None and node.node_id != node_id:
            continue
        if not any(
            direction.direction_id == candidate_id
            and direction.state is SearchDirectionState.COMMITTED
            for direction in node.directions
        ):
            continue
        updated_node = set_observation_direction_state(
            node, candidate_id, SearchDirectionState.PENDING,
        )
        return replace(
            state,
            frontier_regions=tuple(
                replace(region, deferred_order=(state.observation_history.index(node), 0))
                if region.region_id == candidate_id else region
                for region in state.frontier_regions
            ),
            observation_history=replace_history_node(state.observation_history, updated_node),
        )
    return state


def mark_latest_committed_explored(state: SearchState) -> SearchState:
    """扫描结束后确认最近一次停靠动作完成，不代表整个 Frontier 已探索。"""
    history = state.observation_history
    for node in reversed(history):
        for direction in node.directions:
            if direction.state is SearchDirectionState.COMMITTED:
                updated_node = set_observation_direction_state(
                    node, direction.direction_id, SearchDirectionState.EXPLORED
                )
                return replace(
                    state,
                    observation_history=replace_history_node(history, updated_node),
                )
    return state


def replace_history_node(
    history: Tuple[ObservationNode, ...],
    replacement: ObservationNode,
) -> Tuple[ObservationNode, ...]:
    """按 node_id 替换一个不可变历史节点。"""
    return tuple(
        replacement if node.node_id == replacement.node_id else node
        for node in history
    )


__all__ = [
    "commit_frontier_move",
    "mark_latest_committed_explored",
    "release_interrupted_direction",
    "reject_frontier_direction",
    "replace_history_node",
    "select_exploration_target",
    "stall_frontier_direction",
]
