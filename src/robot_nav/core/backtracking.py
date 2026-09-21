"""回退行为：新 Frontier 耗尽后沿当前分支逐层返回父节点。

同一行为的正常推进与可恢复失败放在一起：返回父节点、恢复该节点的暂存方向、
到达容差不满足或路径未知时跳过失败节点并释放其暂存方向。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple

from .exploration import commit_frontier_move, replace_history_node
from .frontier import FrameFrontierCache
from .frontier_regions import frontier_candidate_debug, refresh_frontier_regions
from .history import set_observation_direction_state
from .models import (
    ActionConstraint,
    ActionKind,
    ActionPurpose,
    FrontierCandidate,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObservationNode,
    Pose2D,
    SearchPhase,
    SearchState,
)
from .navigation_io import invalid_result, make_action, result
from .timing import TimingSpans

BACKTRACK_ARRIVAL_M = 0.25


def begin_backtracking(
    frame: NavigationFrame,
    state: SearchState,
    candidates: Tuple[FrontierCandidate, ...],
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """只返回栈顶节点；到达且没有剩余方向才出栈，继续返回上一层。"""
    from .scan_behavior import reset_scan_after_move

    working_state = replace(reset_scan_after_move(state), active_frontier_id=None)
    nodes = {node.node_id: node for node in working_state.observation_history}
    while working_state.branch_node_ids:
        node = nodes[working_state.branch_node_ids[-1]]
        next_state = replace(
            working_state, phase=SearchPhase.BACKTRACKING, backtrack_node_id=node.node_id,
        )
        distance = distance_to_node(frame.pose, node)
        if distance > BACKTRACK_ARRIVAL_M:
            node_index = working_state.observation_history.index(node)
            return result(
                NavigationStatus.OK,
                next_state,
                "backtrack.return",
                "返回当前分支的上一节点；到达后检查方向，没有剩余方向再退一层。",
                make_action(
                    ActionKind.MOVE_TO_POSE,
                    destination=Pose2D(*node.position_world_xy, math.atan2(
                        node.position_world_xy[1] - frame.pose.y_m,
                        node.position_world_xy[0] - frame.pose.x_m)),
                    constraint=ActionConstraint.REQUIRE_KNOWN_PATH,
                    purpose=ActionPurpose.BACKTRACK,
                ),
                {
                    "node_id": node.node_id,
                    "parent_node_id": node.node_id,
                    "destination_world_xy": node.position_world_xy,
                    "branch_depth": len(next_state.branch_node_ids),
                    "pending_direction_count": sum(
                        candidate.deferred_order is not None
                        and candidate.deferred_order[0] == node_index
                        for candidate in candidates
                    ),
                    "frontier_selection_source": "deferred",
                    "new_frontier_count": sum(candidate.deferred_order is None for candidate in candidates),
                    "deferred_frontier_count": sum(candidate.deferred_order is not None for candidate in candidates),
                    "frontier_candidates": tuple(
                        frontier_candidate_debug(candidate) for candidate in candidates
                    ),
                    "distance_to_parent_m": distance,
                },
            )
        resumed = resume_pending_direction(frame, next_state, node, candidates)
        if resumed is not None:
            return resumed
        # 只有实际到达且方向已耗尽的节点才能退栈；历史记录仍保留用于屏蔽和复盘。
        working_state = replace(
            working_state, branch_node_ids=working_state.branch_node_ids[:-1],
        )
        if any(candidate.deferred_order is None for candidate in candidates):
            from .exploration import select_exploration_target

            return select_exploration_target(frame, working_state, None,
                timings=timings, frontier_cache=frontier_cache,
            )

    return wait_for_semantics_or_finish(working_state)


def continue_backtracking(
    frame: NavigationFrame,
    state: SearchState,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """返回命令已同步结束，检查实际到达位置，再刷新父节点的剩余候选。"""
    node = next(node for node in state.observation_history
                if node.node_id == state.backtrack_node_id)
    distance = distance_to_node(frame.pose, node)
    if distance > BACKTRACK_ARRIVAL_M:
        return recover_backtrack_issue(
            state,
            f"返回动作已结束，距父节点 {distance:.3f} m，超过 {BACKTRACK_ARRIVAL_M:.2f} m 到达容差。",
            issue_kind="not_arrived", actual_pose=frame.pose,
        )
    try:
        state, candidates = refresh_frontier_regions(frame, state,
            timings=timings, frontier_cache=frontier_cache,
        )
    except ValueError as exc:
        return invalid_result(state, f"回到父节点后无法刷新 Frontier：{exc}")
    return begin_backtracking(frame, state, candidates, timings=timings, frontier_cache=frontier_cache)


def recover_backtrack_issue(
    state: SearchState,
    reason: str,
    *,
    issue_kind: str,
    actual_pose: Optional[Pose2D] = None,
    rejected_path_world_xy: Tuple[Tuple[float, float], ...] = (),
) -> NavigationResult:
    """返回动作已结束但未完成：跳过失败节点，释放其暂存方向，从实际位置重新检查。"""
    from .scan_behavior import reset_scan_after_move

    node_index = next(index for index, node in enumerate(state.observation_history)
                      if node.node_id == state.backtrack_node_id)
    node = state.observation_history[node_index]
    released_ids = tuple(
        region.region_id for region in state.frontier_regions
        if region.deferred_order is not None and region.deferred_order[0] == node_index
    )
    next_state = reset_scan_after_move(replace(
        state,
        branch_node_ids=state.branch_node_ids[:-1],
        active_frontier_id=None,
        frontier_regions=tuple(
            replace(region, deferred_order=None) if region.region_id in released_ids else region
            for region in state.frontier_regions
        ),
    ))
    details = {
        "parent_node_id": node.node_id,
        "skipped_backtrack_node_id": node.node_id,
        "issue_kind": issue_kind,
        "reason": str(reason),
        "released_frontier_ids": released_ids,
        "parent_world_xy": node.position_world_xy,
        "rejected_path_world_xy": rejected_path_world_xy,
    }
    if actual_pose is not None:
        details.update(
            actual_world_xy=(actual_pose.x_m, actual_pose.y_m),
            distance_to_parent_m=distance_to_node(actual_pose, node),
            arrival_tolerance_m=BACKTRACK_ARRIVAL_M,
        )
    return result(
        NavigationStatus.OK, next_state, "motion.backtrack_recovered",
        "本次返回未完成，跳过该返回节点；保留其有效 Frontier，从实际位置重新检查并继续探索。",
        details=details,
    )


def resume_pending_direction(
    frame: NavigationFrame,
    state: SearchState,
    node: ObservationNode,
    candidates: Tuple[FrontierCandidate, ...],
) -> Optional[NavigationResult]:
    """恢复已到达节点的首个有效暂存方向；没有剩余方向时返回 None。"""
    node_index = state.observation_history.index(node)
    pending = sorted(
        (candidate for candidate in candidates
         if candidate.deferred_order is not None
         and candidate.deferred_order[0] == node_index),
        key=lambda candidate: (candidate.deferred_order[1], candidate.candidate_id),
    )
    if not pending:
        return None
    selected = pending[0]
    ordered = (selected,) + tuple(
        candidate for candidate in candidates if candidate.candidate_id != selected.candidate_id
    )
    return commit_frontier_move(
        frame, state, ordered, (), None, parent_node_id=node.node_id,
    )


def wait_for_semantics_or_finish(state: SearchState) -> NavigationResult:
    """没有可走方向时先排空已提交的检测；待处理不等于目标不存在。"""
    pending = state.pending_semantic_jobs
    return result(
        NavigationStatus.OK if pending else NavigationStatus.NO_SOLUTION,
        replace(state, phase=SearchPhase.WAITING_FOR_SEMANTICS if pending else SearchPhase.FAILED),
        "perception.wait" if pending else "explore.exhausted",
        (
            f"没有剩余有效探索方向，保持当前位置等待 {pending} 项视觉工作。"
            if pending else (
                "当前分支已退完，视觉队列已处理完，没有剩余有效探索方向。"
                + (f"其中 {state.failed_semantic_jobs} 批检测失败，未完成全部图像检查。"
                   if state.failed_semantic_jobs else "")
            )
        ),
        details={
            "blocked_frontier_region_count": len(state.blocked_frontier_regions),
            "pending_semantic_jobs": pending, "failed_semantic_jobs": state.failed_semantic_jobs,
        },
    )


def distance_to_node(pose: Pose2D, node: ObservationNode) -> float:
    """机器人与父节点位置的二维距离，单位为米。"""
    return math.hypot(pose.x_m - node.position_world_xy[0], pose.y_m - node.position_world_xy[1])


__all__ = [
    "begin_backtracking",
    "continue_backtracking",
    "recover_backtrack_issue",
    "wait_for_semantics_or_finish",
]
