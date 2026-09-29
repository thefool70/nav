"""探索：从地图生成 Frontier，选择方向、记录分支，并在耗尽或失败后回退。

阅读顺序：选点与移动 → 分支回退与恢复 → 区域历史 → 前沿提取。
扫描和目标停靠共用本文件的前沿／可达区计算；本模块不读取设备或调用模型。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import replace, dataclass, field
from typing import Optional, Tuple, Any, Mapping, Sequence, Set, Dict

from .geometry import wrap_angle, grid_cell_center_to_world, world_to_nearest_grid_cell
from .models import (
    ActionConstraint,
    ActionKind,
    ActionPurpose,
    ActionExecutionResult,
    ActionOutcome,
    FrontierCandidate,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObservationNode,
    Pose2D,
    SearchPhase,
    SearchState,
    NavigationAction,
    result,
    BlockedFrontierRegion,
    FrontierScoreRequest,
    SearchDirectionState,
    FrontierRegion,
    ObstacleMap,
)
from .timing import TimingSpans, measure_stage


BACKTRACK_ARRIVAL_M = 0.25
# 格坐标始终按 (row, col)；与世界坐标 (x, y) 的转换集中在 geometry.py。
Cell = Tuple[int, int]
GridValues = Tuple[Tuple[Optional[float], ...], ...]
PATH_DISTANCE_SCORE_WEIGHT = 0.05
SEMANTIC_SCORE_WEIGHT = 1.5
FRONTIER_CLEARANCE_SEARCH_M = 0.75
FRONTIER_FRAGMENT_GAP_M = 0.30
MAX_UNKNOWN_HOLE_AREA_M2 = 0.05
MIN_FRONTIER_SPAN_M = 0.50
MIN_FRONTIER_GOAL_DISTANCE_M = 0.35


@dataclass(frozen=True)
class FrontierExtraction:
    """去除小孔洞后的完整边界、移动候选及过滤统计；不修改输入地图。"""

    # 扫描使用移动筛选前的边界；距离、跨度和历史排除只影响 candidates。
    boundary_cells: Tuple[Cell, ...] = ()
    candidates: Tuple[FrontierCandidate, ...] = ()
    hole_filter_applied: bool = False
    ignored_hole_count: int = 0
    ignored_hole_area_m2: float = 0.0
    ignored_frontier_cell_count: int = 0


@dataclass
class FrameFrontierCache:
    """单周期、单帧的提取结果；排除点不同则重新计算，区域匹配仍由调用者执行。"""

    frame: NavigationFrame
    extractions: Dict[Tuple[Tuple[float, float], ...], FrontierExtraction] = field(default_factory=dict)


# 选点与分支推进


def select_exploration_target(
    frame: NavigationFrame,
    state: SearchState,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """优先选择新 Frontier；新候选耗尽时沿当前分支逐个返回父节点。"""
    explored_state = mark_latest_committed_explored(state)
    explored_state, frontiers = refresh_frontier_regions(frame, explored_state,
        timings=timings, frontier_cache=frontier_cache,
    )
    if not any(item.deferred_order is None for item in frontiers.candidates):
        # 旧方向属于原父节点；先沿分支回去，再按保存顺序恢复。
        _, deferred = _rank_frontier_directions(frontiers.candidates, None)
        return begin_backtracking(frame, explored_state, deferred,
            timings=timings,
        )
    return _select_new_frontiers(frame, explored_state, frontiers.candidates, timings=timings)


def complete_frontier_selection(
    frame: NavigationFrame, state: SearchState, request: FrontierScoreRequest,
    scores: Mapping[str, float], *, timings: Optional[TimingSpans] = None,
) -> NavigationResult:
    """用已有分数完成同帧候选选择；直接记录新移动，不重复区域关联和行为分派。"""
    with measure_stage(timings, "frontier.select_rank"):
        ranked_new, _ = _rank_frontier_directions(request.candidates, scores)
    with measure_stage(timings, "frontier.commit"):
        return commit_frontier_move(frame, state, ranked_new + request.deferred, ranked_new, scores)


def _select_new_frontiers(frame, state, candidates, *, timings):
    """有新候选时只走这一条路径：几何排序 → 按需查询缓存 → 提交。"""
    with measure_stage(timings, "frontier.select_rank"):
        ranked_new, deferred = _rank_frontier_directions(candidates, None)
    candidates = ranked_new + deferred
    if len(ranked_new) > 1 and (state.scan_views or state.asynchronous_perception):
        return result(
            NavigationStatus.NEEDS_FRONTIER_SCORES,
            replace(state, phase=SearchPhase.EXPLORING),
            "frontier.score", "只对新 Frontier 评分，旧方向保持暂存顺序。",
            details={
                "frontier_selection_source": "new",
                "new_frontier_count": len(ranked_new),
                "deferred_frontier_count": len(deferred),
                "candidate_count": len(candidates),
                "frontier_candidates": tuple(frontier_candidate_debug(item) for item in candidates),
            },
            frontier_score_request=FrontierScoreRequest(ranked_new, deferred),
        )

    with measure_stage(timings, "frontier.commit"):
        return commit_frontier_move(
            frame, state, candidates, ranked_new, None,
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
    node = ObservationNode(
        node_id=f"observation:{len(state.observation_history)}",
        position_world_xy=(frame.pose.x_m, frame.pose.y_m),
        candidate_id=selected.candidate_id,
        destination_world_xy=destination,
        heading_world_rad=wrap_angle(selected.heading_world_rad),
    )
    from .scan import reset_scan_after_move, scan_coverage_details

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
        NavigationAction(
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
                else "no_new_scan_images" if not state.scan_views else None
            ),
            "frontier_path_distance_weight": PATH_DISTANCE_SCORE_WEIGHT,
            "frontier_semantic_score_weight": SEMANTIC_SCORE_WEIGHT,
            "frontier_candidates": tuple(frontier_candidate_debug(candidate) for candidate in candidates),
            "destination_world_xy": destination,
        },
    )


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
    state, frontiers = refresh_frontier_regions(frame, state,
        timings=timings, frontier_cache=frontier_cache,
    )
    return begin_backtracking(frame, state, frontiers.candidates, timings=timings)


def begin_backtracking(
    frame: NavigationFrame,
    state: SearchState,
    candidates: Tuple[FrontierCandidate, ...],
    *,
    timings: Optional[TimingSpans] = None,
) -> NavigationResult:
    """只返回栈顶节点；到达且没有剩余方向才出栈，继续返回上一层。"""
    from .scan import reset_scan_after_move

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
                NavigationAction(
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
            return _select_new_frontiers(frame, working_state, candidates, timings=timings)

    return wait_for_semantics_or_finish(working_state)


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


# 执行反馈与恢复


def recover_backtrack_issue(
    state: SearchState,
    reason: str,
    *,
    issue_kind: str,
    actual_pose: Optional[Pose2D] = None,
    rejected_path_world_xy: Tuple[Tuple[float, float], ...] = (),
) -> NavigationResult:
    """返回动作已结束但未完成：跳过失败节点，释放其暂存方向，从实际位置重新检查。"""
    from .scan import reset_scan_after_move

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


def recover_frontier_motion(
    state: SearchState, action: NavigationAction, execution: ActionExecutionResult,
) -> Optional[NavigationResult]:
    """一次解释探索失败：停滞保留实际位置；普通失败排除目标；路径问题屏蔽整片边界。"""
    from .scan import reset_scan_after_move

    stalled = execution.outcome is ActionOutcome.STALLED
    if stalled and state.phase is not SearchPhase.SCANNING:
        return None
    candidate_id, reason = action.candidate_id, execution.reason
    block_region = bool(execution.rejected_path_world_xy) or execution.outcome is ActionOutcome.PATH_BLOCKED
    blocked_regions = state.blocked_frontier_regions
    if block_region and not stalled:
        region = next((item for item in state.frontier_regions if item.region_id == candidate_id), None)
        if region is None:
            # 缺少完整边界时不能降为点屏蔽；交回运行层传播原始执行异常。
            return None
        blocked_regions += (BlockedFrontierRegion(region.region_id, region.boundary_world_xy),)

    node = next((item for item in reversed(state.observation_history)
                 if item.candidate_id == candidate_id and item.state is SearchDirectionState.COMMITTED
                 and (stalled or action.node_id is None or item.node_id == action.node_id)), None)
    if node is None and not stalled:
        return None
    if node is not None:
        updated = replace(node, execution_reason=str(reason), state=(
            SearchDirectionState.STALLED if stalled else SearchDirectionState.INVALIDATED))
        state = replace(state, active_frontier_id=None,
                        observation_history=replace_history_node(state.observation_history, updated))
    if stalled:
        return result(
            NavigationStatus.OK, state, "motion.stalled_continue",
            "移动连续静止达到门槛，按当前位置结束本次移动并继续扫描。",
            details={"direction_id": candidate_id, "reason": str(reason)},
        )

    state = reset_scan_after_move(replace(
        state, blocked_frontier_regions=blocked_regions,
        frontier_regions=(tuple(region for region in state.frontier_regions if region.region_id != candidate_id)
                          if block_region else state.frontier_regions),
    ))
    return result(
        NavigationStatus.OK, state, "motion.frontier_rejected",
        ("移动被取消，已屏蔽整个 Frontier 区域，本次运行中不再重试该区域。" if block_region
         else "本次探索动作执行失败，先检查当前位置，再重新选择区域。"),
        details={
            "direction_id": candidate_id, "reason": str(reason),
            "rejection_scope": "region" if block_region else "point",
            "blocked_frontier_region_count": len(blocked_regions),
            "rejected_path_world_xy": execution.rejected_path_world_xy,
        },
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


# 探索历史与区域关联


def mark_latest_committed_explored(state: SearchState) -> SearchState:
    """扫描结束后确认最近一次停靠动作完成，不代表整个 Frontier 已探索。"""
    history = state.observation_history
    for node in reversed(history):
        if node.state is SearchDirectionState.COMMITTED:
            updated_node = replace(node, state=SearchDirectionState.EXPLORED)
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


def distance_to_node(pose: Pose2D, node: ObservationNode) -> float:
    """机器人与父节点位置的二维距离，单位为米。"""
    return math.hypot(pose.x_m - node.position_world_xy[0], pose.y_m - node.position_world_xy[1])


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


def tried_candidate_points(history: Tuple[ObservationNode, ...]) -> Tuple[Tuple[float, float], ...]:
    """屏蔽已完成或执行异常的目标位置，避免原地重复下发同一个探索任务。"""
    return tuple(
        node.destination_world_xy for node in history
        if node.state is not SearchDirectionState.COMMITTED
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
    old_boundaries = [
        _boundary_cells_and_neighborhood(obstacle_map, region.boundary_world_xy)
        for region in previous
    ]
    # 每个新区域保留自己的重叠父区域；ID 配对与暂存继承共用这份关系。
    overlaps = []
    for candidate in candidates:
        cells = set(candidate.frontier_cells)
        parents = []
        for old_index, (old_cells, neighborhood) in enumerate(old_boundaries):
            overlap = len(cells & neighborhood)
            if overlap:
                parents.append((-len(cells & old_cells), -overlap, old_index))
        overlaps.append(parents)
    # 先按精确重叠、再按邻域重叠配对；旧 ID 只能被一个新区域继承。
    matches = sorted((exact, overlap, new_index, old_index)
                     for new_index, parents in enumerate(overlaps)
                     for exact, overlap, old_index in parents)
    assignments = {}
    used_old = set()
    for _, _, new_index, old_index in matches:
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
        # 暂存顺序独立于 ID 配对继承，分裂后未继承旧 ID 的部分也保留历史次序。
        deferred_order = _inherited_deferred_order(overlaps[index], previous)
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


# 前沿提取与可达区


def extract_frame_frontiers(
    frame: NavigationFrame,
    excluded_world_xy: Tuple[Tuple[float, float], ...],
    cache: Optional[FrameFrontierCache] = None,
    timings: Optional[TimingSpans] = None,
) -> FrontierExtraction:
    """只复用同一个只读帧和相同排除点的几何提取，不缓存状态相关的屏蔽与区域 ID。"""
    use_cache = cache is not None and cache.frame is frame
    excluded_points = _normalize_points(excluded_world_xy)
    if use_cache and excluded_points in cache.extractions:
        with measure_stage(timings, "frontier.cache_hit"):
            return cache.extractions[excluded_points]
    result = extract_frontiers(
        frame.obstacle_map, frame.pose,
        excluded_world_xy=excluded_points,
        visibility_map=frame.visibility_map,
        timings=timings,
    )
    if use_cache:
        cache.extractions[excluded_points] = result
    return result


def extract_frontiers(
    obstacle_map: ObstacleMap,
    pose: Pose2D,
    excluded_world_xy: Sequence[Tuple[float, float]] = (),
    *,
    visibility_map: Optional[ObstacleMap] = None,
    timings: Optional[TimingSpans] = None,
) -> FrontierExtraction:
    """生成边界与几何候选；区域关联后的新旧分层和语义排序由选点入口负责。"""
    with measure_stage(timings, "frontier.prepare"):
        grid = _normalize_grid(obstacle_map)
        resolution = _positive_finite(obstacle_map.resolution_m, "resolution_m")
        excluded_points = _normalize_points(excluded_world_xy)

        free_cells = _free_cells(grid)
        if not free_cells:
            return FrontierExtraction()

    with measure_stage(timings, "frontier.reachable_distances"):
        requested_seed = world_to_nearest_grid_cell(
            (pose.x_m, pose.y_m), obstacle_map
        )
        seed = _nearest_free_cell(requested_seed, free_cells)
        reachable_distance = _reachable_free_distances(seed, free_cells)
        reachable_cells = set(reachable_distance)
    with measure_stage(timings, "frontier.boundary_and_holes"):
        frontier_cells = _find_frontier_cells(grid, reachable_cells)
        original_frontier_count = len(frontier_cells)
        frontier_cells, ignored_unknown, hole_sizes, filter_applied = _filter_frontier_holes(
            obstacle_map, visibility_map, grid, reachable_cells, frontier_cells,
            resolution, MAX_UNKNOWN_HOLE_AREA_M2,
        )

    with measure_stage(timings, "frontier.cluster"):
        clearance_search_steps = max(
            1,
            int(math.ceil(FRONTIER_CLEARANCE_SEARCH_M / resolution)),
        )
        candidate_groups = tuple(
            component
            for component in _merge_frontier_fragments(
                frontier_cells, grid, resolution, ignored_unknown, free_cells,
            )
            if _frontier_span_m(component, resolution) >= MIN_FRONTIER_SPAN_M
        )

    with measure_stage(timings, "frontier.representatives_and_rank"):
        candidates = _build_frontier_candidates(
            obstacle_map, pose, grid, candidate_groups, reachable_distance,
            resolution, MIN_FRONTIER_GOAL_DISTANCE_M, excluded_points, clearance_search_steps,
        )

    return FrontierExtraction(
        boundary_cells=tuple(sorted(frontier_cells)),
        candidates=tuple(candidates), hole_filter_applied=filter_applied,
        ignored_hole_count=len(hole_sizes),
        ignored_hole_area_m2=sum(hole_sizes) * resolution * resolution,
        ignored_frontier_cell_count=original_frontier_count - len(frontier_cells),
    )


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


# 内部排序、边界与栅格计算


def _rank_frontier_directions(
    candidates: Tuple[FrontierCandidate, ...],
    scores: Optional[Mapping[str, float]],
) -> Tuple[Tuple[FrontierCandidate, ...], Tuple[FrontierCandidate, ...]]:
    """给新候选加语义分；暂存方向只恢复原顺序。分别返回新候选与暂存候选。"""
    new, deferred = [], []
    scores = scores or {}
    for candidate in candidates:
        if candidate.deferred_order is not None:
            deferred.append(candidate)
            continue
        semantic_score = scores.get(candidate.candidate_id)
        # 缺分不扣分；0.5 为中性，VLM 只调整已有几何候选的排序。
        bonus = (
            0.0 if semantic_score is None
            else SEMANTIC_SCORE_WEIGHT * (2.0 * semantic_score - 1.0)
        )
        new.append(replace(
            candidate, semantic_score=semantic_score, score=candidate.score + bonus,
        ))
    new.sort(key=lambda item: (-item.score, item.path_distance_m, item.candidate_id))
    # 暂存节点后进先出，同一节点沿用原排名，新的模型分数不能插队。
    deferred.sort(key=lambda item: (
        -item.deferred_order[0], item.deferred_order[1], item.candidate_id,
    ))
    return tuple(new), tuple(deferred)


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
    parents: Sequence[Tuple[int, int, int]],
    previous: Tuple[FrontierRegion, ...],
) -> Optional[Tuple[int, int]]:
    """从本区域的（负实际重叠数，负邻域重叠数，旧序号）继承暂存顺序。"""
    exact_parents = [item for item in parents if item[0] < 0]
    if exact_parents:
        parents = exact_parents
    elif parents:
        best_overlap = min(item[1] for item in parents)
        parents = [item for item in parents if item[1] == best_overlap]
    orders = [
        previous[old_index].deferred_order
        for _, _, old_index in parents
        if previous[old_index].deferred_order is not None
    ]
    return min(orders, key=lambda order: (-order[0], order[1])) if orders else None


def _is_free_grid_cell(grid, row: int, col: int, height: int, width: int) -> bool:
    return (
        0 <= row < height and 0 <= col < width
        and grid[row][col] is not None and grid[row][col] <= 0.5
    )


def _filter_frontier_holes(
    obstacle_map, visibility_map, grid, reachable_cells, frontier_cells, resolution, hole_area_limit,
):
    """在同格网原始图中过滤封闭小孔洞，返回边界、忽略格、孔洞大小及启用标志。"""
    # 未膨胀图与探索图同格网时才分类，避免膨胀切断未知区域后误判为小孔洞。
    if (
        visibility_map is None or hole_area_limit <= 0.0
        or visibility_map.frame_id != obstacle_map.frame_id
        or visibility_map.origin != obstacle_map.origin
        or visibility_map.resolution_m != obstacle_map.resolution_m
        or len(visibility_map.occupancy) != len(grid)
        or any(len(row) != len(grid[0]) for row in visibility_map.occupancy)
    ):
        return frontier_cells, set(), (), False
    raw_grid = _normalize_grid(visibility_map)
    seeds = {
        neighbor for row, col in frontier_cells
        for neighbor in _eight_neighbors(row, col)
        if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0])
        and grid[neighbor[0]][neighbor[1]] is None
    }
    ignored_unknown, hole_sizes = _small_unknown_holes(raw_grid, seeds, resolution, hole_area_limit)
    frontier_cells = _find_frontier_cells(grid, reachable_cells, ignored_unknown)
    return frontier_cells, ignored_unknown, hole_sizes, True


def _build_frontier_candidates(
    obstacle_map, pose, grid, candidate_groups, reachable_distance,
    resolution, minimum_distance, excluded_points, clearance_search_steps,
):
    """每片边界选可用代表点并计算几何分；不修改区域历史、不读取模型分数。"""
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
        # 每片边界只选一个移动代表点，完整边界仍保留给覆盖判断与跨帧关联。
        row, col = _safest_frontier_cell(
            eligible_cells,
            grid,
            reachable_distance,
            clearance_search_steps,
        )
        path_distance = reachable_distance[(row, col)] * resolution
        world_xy = grid_cell_center_to_world(row, col, obstacle_map)
        # 此 ID 只标识本帧格子；match_frontier_regions 再分配跨帧区域 ID。
        candidate_id = f"frontier:{row}:{col}"
        heading = math.atan2(world_xy[1] - pose.y_m, world_xy[0] - pose.x_m)
        score = frontier_span - PATH_DISTANCE_SCORE_WEIGHT * path_distance
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
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.score,
            candidate.path_distance_m,
            candidate.candidate_id,
        )
    )
    return candidates


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
    ignored_unknown: Set[Cell], free: Set[Cell],
) -> Tuple[Set[Cell], ...]:
    """复用本帧自由格，合并未知侧朝向相近且自由区短路径不超过 0.30 m 的断段。"""
    components = _connected_components(cells)
    owners = {cell: index for index, part in enumerate(components) for cell in part}
    normals = tuple(_unknown_side_normal(part, grid, ignored_unknown) for part in components)
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
    rows = tuple(tuple(row) for row in obstacle_map.occupancy)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("occupancy must be a non-empty rectangular grid")
    # 扩图后大量行全为未知；先用元组计数跳过，避免逐格执行 Python 判断。
    if any(value is not None and not math.isfinite(value)
           for row in rows if row.count(None) != len(row) for value in row):
        raise ValueError("occupancy values must be finite numbers or None")
    return rows


def _free_cells(grid: GridValues) -> Set[Cell]:
    """返回占据图中的全部已知自由格。"""
    return {
        (row, col)
        for row, values in enumerate(grid)
        if values.count(None) != len(values)
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
    nearest_distance_squared = (search_steps + 1) ** 2
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
            nearest_distance_squared = min(nearest_distance_squared, distance_squared)
    return nearest_distance_squared


def _normalize_points(
    values: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    """校验用于候选抑制的世界坐标。"""
    points = tuple(values)
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
        raise ValueError("excluded_world_xy must contain finite points")
    return points


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


def _positive_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _non_negative_finite(value: float, name: str) -> float:
    if not math.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return float(value)
