"""搜索核心与运行层之间共享的结果构造与输入校验。

本模块只提供两类内容：
1. 公共输入检查（:func:`validation_error` 与字段级辅助函数）。
2. 单周期结果构造（:func:`result`、:func:`invalid_result`）。

它不决定任何搜索行为：行为分派在 ``navigator``，具体行为在四个行为模块中。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

from .models import (
    ActionKind,
    ActionPurpose,
    ActionConstraint,
    BlockedFrontierRegion,
    FrontierRegion,
    FrontierScoreRequest,
    NavigationAction,
    NavigationDebug,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObjectApproachState,
    ObjectLocalization,
    ObstacleMap,
    ObservationView,
    Pose2D,
    RelativePoseCommand,
    SearchMode,
    SearchPhase,
    SearchState,
    TargetClue,
    TargetConfirmation,
    TargetSearchGoal,
    TargetVisibility,
)

BOUNDARY_TOLERANCE_M = 0.25


def make_action(
    action: ActionKind,
    *,
    destination: Optional[Pose2D] = None,
    command: Optional[RelativePoseCommand] = None,
    constraint: ActionConstraint = ActionConstraint.NONE,
    purpose: ActionPurpose = ActionPurpose.OTHER,
    candidate_id: Optional[str] = None,
    node_id: Optional[str] = None,
) -> NavigationAction:
    """构造一个显式动作，交由运行层执行。"""
    return NavigationAction(
        action=action,
        constraint=constraint,
        destination=destination,
        command=command,
        purpose=purpose,
        candidate_id=candidate_id,
        node_id=node_id,
    )


def result(
    status: NavigationStatus,
    state: SearchState,
    stage: str,
    message: str,
    action: Optional[NavigationAction] = None,
    details: Optional[Mapping[str, Any]] = None,
    frontier_score_request: Optional[FrontierScoreRequest] = None,
) -> NavigationResult:
    """集中构造单周期结果，使各行为只描述状态变化与请求的动作。

    ``stage`` 只用于日志说明；运行层按 ``action.action`` 执行，不解析该字符串。
    """
    return NavigationResult(
        status=status,
        action=action,
        debug=NavigationDebug(stage=stage, message=message, details=details or {}),
        state=state,
        frontier_score_request=frontier_score_request,
    )


def invalid_result(state: Optional[SearchState], reason: str) -> NavigationResult:
    """构造非法输入结果，并把状态置为 FAILED。"""
    failed_state = (
        replace_failed(state) if isinstance(state, SearchState) else SearchState(phase=SearchPhase.FAILED)
    )
    return result(NavigationStatus.INVALID_INPUT, failed_state, "input", reason)


def replace_failed(state: SearchState) -> SearchState:
    """返回 phase 为 FAILED 的状态副本。"""
    from dataclasses import replace

    return replace(state, phase=SearchPhase.FAILED)


def validation_error(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState],
    object_localization: Optional[ObjectLocalization],
) -> Optional[str]:
    """返回非法公共输入的简短原因；合法输入返回 None。

    异步视觉队列是唯一导航流程，因此本检查不再校验同步观测接口
    （``observation`` 与 ``target_confirmation``）。
    """
    if not isinstance(goal, TargetSearchGoal) or not isinstance(
        goal.target_text, str
    ) or not goal.target_text.strip():
        return "目标文本必须为非空字符串"
    if not isinstance(goal.search_mode, SearchMode):
        return "goal.search_mode 必须为 SearchMode"
    if not isinstance(frame, NavigationFrame):
        return "frame 必须为 NavigationFrame"
    if not _is_finite(frame.timestamp_s):
        return "frame.timestamp_s 必须为有限值"
    if not isinstance(frame.pose, Pose2D) or not all(
        _is_finite(value)
        for value in (frame.pose.x_m, frame.pose.y_m, frame.pose.yaw_rad)
    ):
        return "frame.pose 必须为有限 Pose2D"
    if not isinstance(frame.obstacle_map, ObstacleMap):
        return "frame.obstacle_map 必须为 ObstacleMap"
    if not _is_finite(frame.obstacle_map.resolution_m) or float(
        frame.obstacle_map.resolution_m
    ) <= 0.0:
        return "frame.obstacle_map.resolution_m 必须为正有限值"
    if not isinstance(frame.obstacle_map.frame_id, str) or not frame.obstacle_map.frame_id:
        return "frame.obstacle_map.frame_id 必须为非空字符串"
    if frame.visibility_map is not None:
        if (
            not isinstance(frame.visibility_map, ObstacleMap)
            or frame.visibility_map.frame_id != frame.obstacle_map.frame_id
            or not _is_finite(frame.visibility_map.resolution_m)
            or float(frame.visibility_map.resolution_m) <= 0.0
        ):
            return "frame.visibility_map 必须为同坐标系、分辨率有效的未膨胀遮挡图"
    if frame.navigation_map is not None:
        if (
            not isinstance(frame.navigation_map, ObstacleMap)
            or frame.navigation_map.frame_id != frame.obstacle_map.frame_id
            or not _is_finite(frame.navigation_map.resolution_m)
            or float(frame.navigation_map.resolution_m) <= 0.0
        ):
            return "frame.navigation_map 必须为同坐标系、分辨率有效的障碍图"
    if not _is_finite(frame.navigation_clearance_m) or float(frame.navigation_clearance_m) < 0.0:
        return "frame.navigation_clearance_m 必须为非负米数"
    if state is not None and not isinstance(state, SearchState):
        return "state 必须为 SearchState 或 None"
    if object_localization is not None:
        issue = object_localization_error(object_localization)
        if issue is not None:
            return issue
    if state is not None:
        return _state_error(state, goal)
    return None


def _state_error(state: SearchState, goal: TargetSearchGoal) -> Optional[str]:
    """校验 ``SearchState`` 各字段的类型与内部一致性。"""
    if not isinstance(state.phase, SearchPhase):
        return "state.phase 必须为 SearchPhase"
    if not isinstance(state.asynchronous_perception, bool):
        return "state.asynchronous_perception 必须为 bool"
    for count in (state.pending_semantic_jobs, state.failed_semantic_jobs):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return "语义任务计数必须是非负整数"
    clue = state.active_target_clue
    if clue is not None and (
        not isinstance(clue, TargetClue) or not isinstance(clue.pose, Pose2D)
        or not all(_is_finite(value) for value in (clue.pose.x_m, clue.pose.y_m, clue.pose.yaw_rad, clue.timestamp_s))
        or not isinstance(clue.clue_id, str) or not isinstance(clue.map_frame_id, str)
    ):
        return "state.active_target_clue 必须为有效目标线索或 None"
    if clue is not None and any(
        value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1)
        for value in (clue.job_id, clue.view_id)
    ):
        return "物体快照编号必须为正整数或 None"
    object_phases = (SearchPhase.LOCALIZING_OBJECT, SearchPhase.APPROACHING_OBJECT)
    if (state.phase is SearchPhase.REVISITING_TARGET or state.phase in object_phases) and clue is None:
        return "线索返回与物体接近阶段必须保留拍摄位姿"
    if state.phase in object_phases and goal.search_mode is not SearchMode.OBJECT:
        return "物体接近阶段仅用于物体模式"
    context = state.object_approach
    if not isinstance(context, ObjectApproachState):
        return "state.object_approach 必须为 ObjectApproachState"
    if context.target is not None:
        issue = object_localization_error(context.target)
        if issue is not None:
            return issue
    if state.phase is SearchPhase.APPROACHING_OBJECT and (
        context.target is None or context.target.target_world_xy is None
    ):
        return "物体接近阶段必须保留目标位置"
    if context.destination is not None and (
        not isinstance(context.destination, Pose2D)
        or not all(_is_finite(value) for value in (context.destination.x_m, context.destination.y_m, context.destination.yaw_rad))
    ):
        return "物体停靠位姿无效"
    if any(not valid_world_point(point) for point in context.tried_positions):
        return "物体已尝试停靠点无效"
    fallback = context.fallback_clue
    if fallback is not None and (
        not isinstance(fallback, TargetClue) or not isinstance(fallback.pose, Pose2D)
        or not all(_is_finite(value) for value in (fallback.pose.x_m, fallback.pose.y_m, fallback.pose.yaw_rad))
        or not isinstance(fallback.map_frame_id, str)
    ):
        return "物体保底返回线索无效"
    if (
        not isinstance(context.history_localized, bool)
        or isinstance(context.capture_turns, bool)
        or not isinstance(context.capture_turns, int)
        or context.capture_turns < 0
    ):
        return "物体恢复计数无效"
    if (
        isinstance(state.next_frontier_region_id, bool)
        or not isinstance(state.next_frontier_region_id, int)
        or state.next_frontier_region_id < 0
    ):
        return "state.next_frontier_region_id 必须为非负整数"
    if state.active_frontier_id is not None and not isinstance(state.active_frontier_id, str):
        return "state.active_frontier_id 必须为字符串或 None"
    history_node_ids = tuple(node.node_id for node in state.observation_history)
    if (
        not isinstance(state.branch_node_ids, tuple)
        or any(not isinstance(node_id, str) or node_id not in history_node_ids
               for node_id in state.branch_node_ids)
        or len(set(state.branch_node_ids)) != len(state.branch_node_ids)
    ):
        return "state.branch_node_ids 必须为不重复的有效历史节点 ID 元组"
    if state.backtrack_node_id is not None and (
        not isinstance(state.backtrack_node_id, str)
        or not any(node.node_id == state.backtrack_node_id for node in state.observation_history)
    ):
        return "state.backtrack_node_id 必须指向有效历史节点或为 None"
    if state.phase is SearchPhase.BACKTRACKING and state.backtrack_node_id is None:
        return "回退阶段必须指定父节点"
    if state.phase is SearchPhase.BACKTRACKING and (
        not state.branch_node_ids or state.branch_node_ids[-1] != state.backtrack_node_id
    ):
        return "回退目标必须是当前分支的栈顶节点"
    if any(
        not isinstance(region, FrontierRegion)
        or not isinstance(region.region_id, str)
        or not all(valid_world_point(point) for point in region.boundary_world_xy)
        or (
            region.deferred_order is not None
            and (
                not isinstance(region.deferred_order, tuple)
                or len(region.deferred_order) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in region.deferred_order
                )
                or region.deferred_order[0] >= len(state.observation_history)
                or history_node_ids[region.deferred_order[0]] not in state.branch_node_ids
            )
        )
        for region in state.frontier_regions
    ):
        return "state.frontier_regions 必须为有效 FrontierRegion 序列"
    if any(
        not isinstance(region, BlockedFrontierRegion)
        or not isinstance(region.region_id, str)
        or not region.boundary_world_xy
        or not all(valid_world_point(point) for point in region.boundary_world_xy)
        for region in state.blocked_frontier_regions
    ):
        return "state.blocked_frontier_regions 必须包含有效区域边界"
    if any(not valid_world_point(point) for point in state.scan_observation_points):
        return "state.scan_observation_points 必须为有限世界坐标序列"
    if (
        isinstance(state.scan_local_point_count, bool)
        or not isinstance(state.scan_local_point_count, int)
        or state.scan_local_point_count < 0
    ):
        return "state.scan_local_point_count 必须为非负整数"
    if not isinstance(state.observed_views, tuple) or not isinstance(state.pending_observation_views, tuple):
        return "已检查与待处理覆盖必须为 ObservationView 元组"
    if any(
        not isinstance(view, ObservationView)
        or not isinstance(view.pose, Pose2D)
        or not all(_is_finite(value) for value in (
            view.pose.x_m, view.pose.y_m, view.pose.yaw_rad,
            view.camera_heading_world_rad, view.horizontal_fov_rad, view.timestamp_s,
        ))
        or not 0.0 < view.horizontal_fov_rad <= 2.0 * math.pi
        or (view.camera_world_xy is not None and not valid_world_point(view.camera_world_xy))
        or not isinstance(view.depth_coverage_available, bool)
        or not all(valid_world_point(point) for point in view.visible_world_xy)
        or not all(valid_world_point(point) for point in view.map_visible_world_xy)
        for view in state.observed_views + state.pending_observation_views
    ):
        return "已检查与待处理覆盖必须包含有效 ObservationView"
    if not isinstance(state.initial_scan_complete, bool):
        return "state.initial_scan_complete 必须为 bool"
    if state.scan_headings_world_rad:
        if not all(_is_finite(value) for value in state.scan_headings_world_rad):
            return "state.scan_headings_world_rad 必须为有限角度序列"
        if (
            isinstance(state.next_scan_index, bool)
            or not isinstance(state.next_scan_index, int)
            or not 0 <= state.next_scan_index < len(state.scan_headings_world_rad)
        ):
            return "state.next_scan_index 必须位于扫描航向范围内"
    return None


def object_localization_error(value) -> Optional[str]:
    """校验物体定位结果字段。"""
    if not isinstance(value, ObjectLocalization):
        return "物体定位结果必须为 ObjectLocalization"
    if value.target_world_xy is not None and not valid_world_point(value.target_world_xy):
        return "物体定位结果包含非法世界坐标"
    if not isinstance(value.visibility, TargetVisibility) or not isinstance(value.vlm_confirmation, TargetConfirmation):
        return "物体定位的可见性或 VLM 确认类型无效"
    if not isinstance(value.source, str) or not isinstance(value.reason, str):
        return "物体定位来源与原因必须为字符串"
    if isinstance(value.sample_count, bool) or not isinstance(value.sample_count, int) or value.sample_count < 0:
        return "物体定位深度点数无效"
    return None


def valid_world_point(value: object) -> bool:
    """判断 value 是否为包含两个有限数值的世界坐标。"""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    return all(_is_finite(component) for component in value)


def _is_finite(value: object) -> bool:
    """判断 value 是否为有限实数（bool 视为非法）。"""
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "invalid_result",
    "make_action",
    "object_localization_error",
    "result",
    "valid_world_point",
    "validation_error",
]
