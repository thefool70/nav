"""搜索决策入口：检查输入、接收感知增量、分派扫描／探索／目标处理及运动恢复。

从 navigate 阅读正常分派，从 apply_execution_result 阅读执行反馈。
所有跨周期搜索信息保存在 SearchState；本模块不读设备、不请求模型或写文件。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Tuple

from .exploration import (
    FrameFrontierCache,
    recover_frontier_motion,
    refresh_frontier_regions,
    wait_for_semantics_or_finish,
    continue_backtracking,
    select_exploration_target,
    recover_backtrack_issue,
)
from .models import (
    ActionExecutionResult,
    ActionOutcome,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObjectLocalization,
    SearchPhase,
    SearchState,
    TargetSearchGoal,
    result,
    ObservationView,
    TargetClue,
)
from .scan import (
    continue_scanning,
    frontier_observation_points,
    unobserved_observation_points,
    reset_scan_after_move,
    recover_scan_turn,
)
from .target import (
    recover_object_motion,
    continue_target_search,
    discard_target_clue,
)
from .timing import TimingSpans


def navigate(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    *,
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
    target_result = continue_target_search(frame, goal, working_state, object_localization)
    if target_result is not None:
        return target_result
    # 等待并非终态：新帧若出现可用方向，仍可重新进入扫描与探索。
    if working_state.phase is SearchPhase.WAITING_FOR_SEMANTICS:
        refreshed, frontiers = refresh_frontier_regions(
            frame, working_state, timings=timings, frontier_cache=frontier_cache,
        )
        local_points = frontier_observation_points(frame, frontiers.boundary_cells)
        unchecked_points = unobserved_observation_points(
            local_points, frame, refreshed.observed_views + refreshed.pending_observation_views,
        )
        # 没有移动目标时，仍可原地观察新边界；已检查和待分析的覆盖不会重复采集。
        if frontiers.candidates or unchecked_points:
            return continue_scanning(frame, reset_scan_after_move(refreshed),
                timings=timings, frontier_cache=frontier_cache,
            )

        return wait_for_semantics_or_finish(refreshed)
    if working_state.phase is SearchPhase.BACKTRACKING:
        return continue_backtracking(frame, working_state,
            timings=timings, frontier_cache=frontier_cache,
        )
    if working_state.phase is SearchPhase.EXPLORING:
        return select_exploration_target(
            frame,
            working_state,
            timings=timings, frontier_cache=frontier_cache,
        )

    return continue_scanning(frame, working_state,
        timings=timings, frontier_cache=frontier_cache,
    )


def receive_perception(
    state: SearchState, observed_views: Tuple[ObservationView, ...] = (),
    *, pending: int, failed: int, pending_views: Tuple[ObservationView, ...],
    clue: Optional[TargetClue] = None,
) -> SearchState:
    """只有分析成功的新增覆盖进入 observed_views；采集覆盖仍保持待分析。"""
    # 接收是增量归并；相同拍摄时刻的覆盖只登记一次，pending 不冒充已检查。
    timestamps = {view.timestamp_s for view in state.observed_views}
    return replace(
        state, asynchronous_perception=True,
        observed_views=state.observed_views + tuple(
            view for view in observed_views if view.timestamp_s not in timestamps),
        pending_semantic_jobs=pending, failed_semantic_jobs=failed,
        pending_observation_views=pending_views,
        active_target_clue=state.active_target_clue if clue is None else clue,
    )


def target_handling_active(state: SearchState) -> bool:
    """目标处理或终态期间暂停普通队列，不消费下一条线索。"""
    return state.active_target_clue is not None or state.phase in (
        SearchPhase.REVISITING_TARGET, SearchPhase.LOCALIZING_OBJECT,
        SearchPhase.APPROACHING_OBJECT, SearchPhase.COMPLETE,
        SearchPhase.FAILED, SearchPhase.STOPPED,
    )


def apply_execution_result(decision: NavigationResult, execution: ActionExecutionResult):
    """解释同步执行反馈。不能恢复的动作返回 None，由运行层传播原始异常。"""
    if execution.outcome is ActionOutcome.SUCCEEDED:
        return decision
    recovered = _recover_motion(decision, execution)
    if recovered is not None and execution.outcome is ActionOutcome.PATH_UNKNOWN:
        recovered = replace(recovered, debug=replace(recovered.debug, details={
            **recovered.debug.details,
            "unknown_path_length_m": execution.unknown_length_m,
            "unknown_path_limit_m": execution.limit_m,
            "checked_path_length_m": execution.total_path_length_m,
        }))
    return recovered


def invalid_result(state: Optional[SearchState], reason: str) -> NavigationResult:
    """构造非法输入结果，并把状态置为 FAILED。"""
    failed_state = replace(state or SearchState(), phase=SearchPhase.FAILED)
    return result(NavigationStatus.INVALID_INPUT, failed_state, "input", reason)


def validation_error(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    object_localization: Optional[ObjectLocalization],
) -> Optional[str]:
    """检查进入决策的物理数据；内部状态与字段类型遵循 dataclass 契约。"""
    if not goal.target_text.strip():
        return "目标文本必须为非空字符串"
    if not math.isfinite(frame.timestamp_s):
        return "frame.timestamp_s 必须为有限值"
    if not all(math.isfinite(value) for value in (frame.pose.x_m, frame.pose.y_m, frame.pose.yaw_rad)):
        return "frame.pose 必须为有限位姿"
    for grid in (frame.obstacle_map, frame.visibility_map, frame.navigation_map):
        if grid is None:
            continue
        if not math.isfinite(grid.resolution_m) or grid.resolution_m <= 0:
            return "地图分辨率必须为正有限值"
        if not all(math.isfinite(value) for value in (grid.origin.x_m, grid.origin.y_m, grid.origin.yaw_rad)):
            return "地图原点必须为有限位姿"
        if not grid.frame_id or grid.frame_id != frame.obstacle_map.frame_id:
            return "地图必须使用同一非空坐标系"
    if not math.isfinite(frame.navigation_clearance_m) or frame.navigation_clearance_m < 0:
        return "导航净空必须为非负有限米数"
    if object_localization is not None and object_localization.target_world_xy is not None:
        if not all(math.isfinite(value) for value in object_localization.target_world_xy):
            return "物体定位结果包含非法世界坐标"
    return None


def _recover_motion(
    decision: NavigationResult, execution: ActionExecutionResult,
) -> Optional[NavigationResult]:
    """只按动作目的分派；各职责自行解释完整的执行反馈。"""
    action = decision.action
    if action is None:
        return None
    state, purpose = decision.state, action.purpose
    if purpose in (ActionPurpose.APPROACH, ActionPurpose.FALLBACK, ActionPurpose.FALLBACK_TURN):
        return recover_object_motion(purpose, state, execution.reason)
    if purpose is ActionPurpose.SCAN_TURN:
        return recover_scan_turn(state, execution.reason)
    if purpose in (ActionPurpose.REVISIT, ActionPurpose.REVISIT_TURN):
        return discard_target_clue(state, execution.reason)
    if purpose is ActionPurpose.BACKTRACK:
        return recover_backtrack_issue(
            state, execution.reason,
            issue_kind=("stalled" if execution.outcome is ActionOutcome.STALLED
                        else "path_unknown" if execution.rejected_path_world_xy else "failed"),
            rejected_path_world_xy=execution.rejected_path_world_xy,
        )
    if purpose in (ActionPurpose.EXPLORE, ActionPurpose.RESUME):
        return recover_frontier_motion(state, action, execution)
    return None
