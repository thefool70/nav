"""语义目标搜索的单周期分派与公共输入检查。

阅读入口是 :func:`navigate`。本模块只做三件事：校验公共输入、按
``SearchState.phase`` 与搜索模式选择行为模块、把行为结果原样返回。
具体行为按四类组织，正常推进与可恢复失败放在同一模块：

- :mod:`~robot_nav.core.scan_behavior`：扫描与局部 Frontier 补查。
- :mod:`~robot_nav.core.exploration`：探索候选选择与探索移动失败恢复。
- :mod:`~robot_nav.core.backtracking`：逐层返回父节点与返回失败恢复。
- :mod:`~robot_nav.core.object_approach` / :mod:`~robot_nav.core.scene_target`：
  物体与场景目标处理。

所有跨周期信息都显式保存在 ``SearchState`` 中，且只由本核心更新。
"""

from __future__ import annotations

from typing import Mapping, Optional

from .frontier import FrameFrontierCache
from .models import (
    ActionKind,
    ActionExecutionResult,
    ActionOutcome,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObjectLocalization,
    SearchMode,
    SearchPhase,
    SearchState,
    TargetSearchGoal,
)
from .navigation_io import invalid_result, result, validation_error
from .timing import TimingSpans


def navigate(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    *,
    frontier_scores: Optional[Mapping[str, float]] = None,
    object_localization: Optional[ObjectLocalization] = None,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """推进一个导航周期，并返回本周期请求的动作和下一周期状态。

    ``frame`` 来自底盘 Adapter，感知结果由运行层从异步视觉队列取得后传入。
    函数本身不读设备、不调用模型，也不发送命令。
    """
    reason = validation_error(frame, goal, object_localization)
    if reason is not None:
        return invalid_result(state, reason)

    working_state = state or SearchState()
    if working_state.phase is SearchPhase.STOPPED:
        return result(NavigationStatus.OK, working_state, "stopped", "导航已经停止。")
    if working_state.phase is SearchPhase.COMPLETE:
        return result(NavigationStatus.OK, working_state, "complete", "语义搜索已经完成。")
    if working_state.phase is SearchPhase.FAILED:
        return result(
            NavigationStatus.NO_SOLUTION,
            working_state,
            "failed",
            "语义搜索已经结束，当前状态没有可继续的方向。",
        )
    # 目标线索优先于常规探索；处理期间继续既定线索，不被新 Frontier 插队。
    if working_state.active_target_clue is not None:
        from .target_clue import continue_target_clue
        return continue_target_clue(frame, goal, working_state, object_localization)
    if goal.search_mode is SearchMode.OBJECT:
        from .object_approach import continue_object_history

        history_result = continue_object_history(frame, working_state)
        if history_result is not None:
            return history_result
    # 等待并非终态：新帧若出现可用方向，仍可重新进入扫描与探索。
    if working_state.phase is SearchPhase.WAITING_FOR_SEMANTICS:
        from .frontier_regions import refresh_frontier_regions
        from .observation_coverage import frontier_observation_points, unobserved_observation_points
        from .scan_behavior import continue_scanning, reset_scan_after_move

        refreshed, frontiers = refresh_frontier_regions(
            frame, working_state, timings=timings, frontier_cache=frontier_cache,
        )
        local_points = frontier_observation_points(frame, frontiers.boundary_cells)
        unchecked_points = unobserved_observation_points(
            local_points, frame, refreshed.observed_views + refreshed.pending_observation_views,
        )
        # 没有移动目标时，仍可原地观察新边界；已检查和待分析的覆盖不会重复采集。
        if frontiers.candidates or unchecked_points:
            return continue_scanning(frame, goal, reset_scan_after_move(refreshed),
                timings=timings, frontier_cache=frontier_cache,
            )
        from .backtracking import wait_for_semantics_or_finish

        return wait_for_semantics_or_finish(refreshed)
    if working_state.phase is SearchPhase.BACKTRACKING:
        from .backtracking import continue_backtracking

        return continue_backtracking(frame, working_state,
            timings=timings, frontier_cache=frontier_cache,
        )
    if working_state.phase is SearchPhase.EXPLORING:
        from .exploration import select_exploration_target

        return select_exploration_target(
            frame,
            working_state,
            frontier_scores,
            timings=timings, frontier_cache=frontier_cache,
        )

    from .scan_behavior import continue_scanning

    return continue_scanning(frame, goal, working_state,
        timings=timings, frontier_cache=frontier_cache,
    )


def apply_execution_result(decision: NavigationResult, execution: ActionExecutionResult):
    """解释同步执行反馈。不能恢复的动作返回 None，由运行层传播原始异常。"""
    if execution.outcome is ActionOutcome.SUCCEEDED:
        return decision
    recovered = _recover_motion(decision, execution.reason,
        stalled=execution.outcome is ActionOutcome.STALLED,
        path_blocked=execution.outcome is ActionOutcome.PATH_BLOCKED,
        rejected_path_world_xy=execution.rejected_path_world_xy)
    if recovered is not None and execution.outcome is ActionOutcome.PATH_UNKNOWN:
        from dataclasses import replace
        recovered = replace(recovered, debug=replace(recovered.debug, details={
            **recovered.debug.details,
            "unknown_path_length_m": execution.unknown_length_m,
            "unknown_path_limit_m": execution.limit_m,
            "checked_path_length_m": execution.total_path_length_m,
        }))
    return recovered


def _recover_motion(
    result_in: NavigationResult,
    reason: str,
    *,
    stalled: bool,
    path_blocked: bool = False,
    rejected_path_world_xy: tuple = (),
) -> Optional[NavigationResult]:
    """按失败动作的显式类型与方向语义分派恢复处理。"""
    action = result_in.action
    purpose = action.purpose if action is not None else ActionPurpose.OTHER
    state = result_in.state

    from .object_approach import recover_object_motion

    object_recovery = recover_object_motion(
        action.action if action is not None else ActionKind.MOVE_RELATIVE,
        purpose, state, reason,
    )
    if object_recovery is not None:
        return object_recovery

    if purpose == ActionPurpose.SCAN_TURN:
        from .scan_behavior import recover_scan_turn

        return recover_scan_turn(state, reason)
    if purpose in (ActionPurpose.REVISIT, ActionPurpose.REVISIT_TURN):
        from .scene_target import discard_target_clue

        return discard_target_clue(state, reason)
    if purpose == ActionPurpose.BACKTRACK:
        from .backtracking import recover_backtrack_issue

        return recover_backtrack_issue(
            state, reason,
            issue_kind="stalled" if stalled else ("path_unknown" if rejected_path_world_xy else "failed"),
            rejected_path_world_xy=rejected_path_world_xy,
        )
    if purpose not in (ActionPurpose.EXPLORE, ActionPurpose.RESUME):
        return None

    candidate_id = action.candidate_id
    node_id = action.node_id
    if stalled:
        from .exploration import stall_frontier_direction

        return stall_frontier_direction(
            state, reason,
            candidate_id=candidate_id,
        )

    blocked_regions = state.blocked_frontier_regions
    if rejected_path_world_xy or path_blocked:
        from .models import BlockedFrontierRegion

        region = next((region for region in state.frontier_regions
                       if region.region_id == candidate_id), None)
        if region is None:
            # 缺少完整边界时不能退回点屏蔽，否则会再次进入换点重试循环。
            return None
        blocked_regions += (BlockedFrontierRegion(
            region_id=region.region_id,
            boundary_world_xy=region.boundary_world_xy,
        ),)

    from .exploration import reject_frontier_direction

    return reject_frontier_direction(
        state, reason,
        candidate_id=str(candidate_id), node_id=node_id,
        blocked_regions=blocked_regions,
        block_region=bool(rejected_path_world_xy) or path_blocked,
        rejected_path_world_xy=rejected_path_world_xy,
    )


__all__ = ["navigate", "apply_execution_result"]
