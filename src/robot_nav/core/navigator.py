"""语义目标搜索的单周期状态机。

阅读入口是 :func:`navigate`。主流程只有四步：扫描环境、靠近可见目标、
选择 Frontier 探索、在无新路可走时回退。所有跨周期信息都显式保存在
``SearchState`` 中。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

from .frontier import find_frontier_candidates
from .geometry import world_point_to_robot, world_to_nearest_grid_cell
from .grounding import ground_target_bbox
from .history import (
    find_latest_pending_observation_node,
    freeze_observation_node,
    set_observation_direction_state,
)
from .models import (
    NavigationDebug,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObservationNode,
    ObstacleMap,
    Pose2D,
    RelativePoseCommand,
    ScanEvidence,
    SearchDirection,
    SearchDirectionState,
    SearchPhase,
    SearchState,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from .scan import build_uniform_scan_headings, shortest_turn_to_heading


TURN_TOLERANCE_RAD = math.radians(5.0)
TARGET_STANDOFF_M = 0.75
TARGET_REACHED_M = 0.90
MAX_TARGET_APPROACH_ATTEMPTS = 5
BACKTRACK_ARRIVAL_M = 0.25


def navigate(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    observation: Optional[TargetObservation] = None,
) -> NavigationResult:
    """推进一个导航周期，并返回本周期命令和下一周期状态。

    ``frame`` 来自底盘 Adapter；``observation`` 来自独立的目标视觉模块。
    函数本身不读设备、不调用模型，也不发送命令。
    """
    reason = _validation_error(frame, goal, state, observation)
    if reason is not None:
        return _invalid_result(state, reason)

    working_state = state or SearchState()
    if working_state.phase is SearchPhase.COMPLETE:
        return _result(
            NavigationStatus.OK,
            working_state,
            "complete",
            "目标搜索已经完成。",
        )
    if working_state.phase is SearchPhase.FAILED:
        return _result(
            NavigationStatus.NO_SOLUTION,
            working_state,
            "failed",
            "目标搜索已经结束，当前状态没有可继续的方向。",
        )
    if working_state.phase is SearchPhase.BACKTRACKING:
        return _continue_backtracking(frame, working_state)
    if working_state.phase is SearchPhase.LOCALIZING_TARGET:
        return _continue_target_approach(frame, working_state, observation)
    if working_state.phase is SearchPhase.EXPLORING:
        return _select_exploration_target(frame, working_state)

    if not working_state.scan_headings_world_rad:
        working_state = _start_scan(working_state, frame.pose.yaw_rad)
    return _advance_scan(frame, working_state, observation)


def _advance_scan(
    frame: NavigationFrame,
    state: SearchState,
    observation: Optional[TargetObservation],
) -> NavigationResult:
    """对齐并处理当前扫描方向；一轮扫描结束后进入 Frontier 探索。"""
    target_heading = state.scan_headings_world_rad[state.next_scan_index]
    relative_turn = shortest_turn_to_heading(frame.pose.yaw_rad, target_heading)
    if abs(relative_turn) > TURN_TOLERANCE_RAD:
        return _result(
            NavigationStatus.OK,
            state,
            "scan.turn",
            "转向下一个扫描方向。",
            RelativePoseCommand(yaw_rad=relative_turn),
            {
                "scan_index": state.next_scan_index,
                "target_heading_world_rad": target_heading,
            },
        )

    if observation is None:
        return _result(
            NavigationStatus.NOT_IMPLEMENTED,
            state,
            "scan.observe",
            "当前扫描方向需要 TargetObserver 的视觉结果。",
        )
    if observation.visibility is TargetVisibility.UNCERTAIN:
        return _result(
            NavigationStatus.OK,
            state,
            "scan.observe",
            observation.reason or "视觉结果不确定，保持当前位置等待重新观测。",
        )
    if observation.visibility is TargetVisibility.VISIBLE:
        return _approach_visible_target(frame, state, observation)

    evidence = ScanEvidence(
        heading_world_rad=target_heading,
        visibility=TargetVisibility.NOT_VISIBLE,
        direction_score=observation.direction_score,
    )
    next_index = state.next_scan_index + 1
    scanned_state = replace(
        state,
        scan_evidence=state.scan_evidence + (evidence,),
        next_scan_index=min(next_index, len(state.scan_headings_world_rad) - 1),
    )
    if next_index < len(state.scan_headings_world_rad):
        next_heading = state.scan_headings_world_rad[next_index]
        return _result(
            NavigationStatus.OK,
            replace(scanned_state, next_scan_index=next_index),
            "scan.turn",
            "当前方向未发现目标，转向下一扫描方向。",
            RelativePoseCommand(
                yaw_rad=shortest_turn_to_heading(frame.pose.yaw_rad, next_heading)
            ),
            {"scan_index": next_index, "target_heading_world_rad": next_heading},
        )

    return _select_exploration_target(
        frame,
        replace(scanned_state, phase=SearchPhase.EXPLORING),
    )


def _continue_target_approach(
    frame: NavigationFrame,
    state: SearchState,
    observation: Optional[TargetObservation],
) -> NavigationResult:
    """移动后重新观察目标，直到到达目标或目标丢失。"""
    if observation is None:
        return _result(
            NavigationStatus.NOT_IMPLEMENTED,
            state,
            "target.observe",
            "接近目标后需要 TargetObserver 重新观测。",
        )
    if observation.visibility is TargetVisibility.UNCERTAIN:
        return _result(
            NavigationStatus.OK,
            state,
            "target.observe",
            observation.reason or "目标观测不确定，保持当前位置等待重新观测。",
        )
    if observation.visibility is TargetVisibility.VISIBLE:
        return _approach_visible_target(frame, state, observation)

    # 目标丢失后，从当前位置重新开始完整扫描；当前帧可直接作为第一向证据。
    scan_state = _start_scan(state, frame.pose.yaw_rad)
    return _advance_scan(frame, scan_state, observation)


def _approach_visible_target(
    frame: NavigationFrame,
    state: SearchState,
    observation: TargetObservation,
) -> NavigationResult:
    """用目标框和深度定位目标，并生成保持安全距离的相对位姿命令。"""
    if (
        observation.bbox_norm is None
        or frame.depth is None
        or frame.camera_intrinsics is None
    ):
        return _result(
            NavigationStatus.NOT_IMPLEMENTED,
            replace(state, phase=SearchPhase.LOCALIZING_TARGET),
            "target.ground",
            "目标可见，但缺少目标框、深度图或相机内参。",
        )

    estimate = ground_target_bbox(
        observation.bbox_norm,
        frame.depth,
        frame.camera_intrinsics,
        frame.pose,
        frame.camera_pose_in_robot,
    )
    if not estimate.success:
        return _result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.LOCALIZING_TARGET),
            "target.ground",
            f"目标位置估计失败：{estimate.reason}",
            details={"valid_depth_points": estimate.sample_count},
        )

    if estimate.distance_m is not None and estimate.distance_m <= TARGET_REACHED_M:
        return _result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.COMPLETE),
            "target.complete",
            "目标已进入到达距离。",
            details={"target_distance_m": estimate.distance_m},
        )

    if state.target_approach_attempts >= MAX_TARGET_APPROACH_ATTEMPTS:
        return _result(
            NavigationStatus.NO_SOLUTION,
            replace(state, phase=SearchPhase.FAILED),
            "target.approach",
            "连续接近目标次数达到上限。",
        )

    target_forward, target_left = estimate.target_base_xy or (0.0, 0.0)
    distance = estimate.distance_m or math.hypot(target_forward, target_left)
    movement_scale = max(0.0, distance - TARGET_STANDOFF_M) / distance
    command = RelativePoseCommand(
        forward_m=target_forward * movement_scale,
        left_m=target_left * movement_scale,
        yaw_rad=estimate.bearing_rad or 0.0,
    )
    return _result(
        NavigationStatus.OK,
        replace(
            state,
            phase=SearchPhase.LOCALIZING_TARGET,
            target_approach_attempts=state.target_approach_attempts + 1,
        ),
        "target.approach",
        "向目标移动，并保留安全停靠距离。",
        command,
        {
            "target_distance_m": distance,
            "valid_depth_points": estimate.sample_count,
        },
    )


def _select_exploration_target(
    frame: NavigationFrame,
    state: SearchState,
) -> NavigationResult:
    """从可达 Frontier 中选择下一探索点，并保存其余方向供回退。"""
    explored_state = _mark_latest_committed_explored(state)
    try:
        candidates = find_frontier_candidates(
            frame.obstacle_map,
            frame.pose,
            preferred_heading_world_rad=_preferred_heading(state.scan_evidence),
            excluded_world_xy=_excluded_candidate_points(
                explored_state.observation_history
            ),
        )
    except ValueError as exc:
        return _invalid_result(state, f"障碍图无法用于 Frontier：{exc}")

    if not candidates:
        return _begin_backtracking(frame, explored_state)

    directions = tuple(
        SearchDirection(
            direction_id=candidate.candidate_id,
            heading_world_rad=candidate.heading_world_rad,
            candidate_world_xy=candidate.world_xy,
        )
        for candidate in candidates
    )
    node = freeze_observation_node(
        f"observation:{len(explored_state.observation_history)}",
        (frame.pose.x_m, frame.pose.y_m),
        directions,
        directions[0].direction_id,
    )
    next_state = _reset_scan_after_move(
        replace(
            explored_state,
            observation_history=explored_state.observation_history + (node,),
        )
    )
    return _result(
        NavigationStatus.OK,
        next_state,
        "explore.select",
        "选择评分最高的 Frontier，保存其余方向供后续回退。",
        _command_to_world_point(candidates[0].world_xy, frame.pose),
        {
            "candidate_id": candidates[0].candidate_id,
            "candidate_count": len(candidates),
            "candidate_score": candidates[0].score,
        },
    )


def _begin_backtracking(
    frame: NavigationFrame,
    state: SearchState,
) -> NavigationResult:
    """回到最近仍有未探索方向的观测节点。"""
    backtrack_state = _mark_latest_committed_explored(state)
    node = find_latest_pending_observation_node(backtrack_state.observation_history)
    if node is None:
        return _result(
            NavigationStatus.NO_SOLUTION,
            replace(backtrack_state, phase=SearchPhase.FAILED, active_node_id=None),
            "backtrack.empty",
            "没有目标，也没有剩余 Frontier 可探索。",
        )

    next_state = replace(
        backtrack_state,
        phase=SearchPhase.BACKTRACKING,
        active_node_id=node.node_id,
    )
    if _distance_to_node(frame.pose, node) <= BACKTRACK_ARRIVAL_M:
        return _resume_pending_direction(frame, next_state, node)
    return _result(
        NavigationStatus.OK,
        next_state,
        "backtrack.return",
        "返回最近仍有候选方向的观测位置。",
        _command_to_world_point(node.position_world_xy, frame.pose),
        {"node_id": node.node_id},
    )


def _continue_backtracking(
    frame: NavigationFrame,
    state: SearchState,
) -> NavigationResult:
    """检查是否回到观测节点；到达后恢复其中一个候选方向。"""
    node = next(
        (
            item
            for item in state.observation_history
            if item.node_id == state.active_node_id
        ),
        None,
    )
    if node is None:
        return _begin_backtracking(frame, replace(state, active_node_id=None))
    if _distance_to_node(frame.pose, node) > BACKTRACK_ARRIVAL_M:
        return _result(
            NavigationStatus.OK,
            state,
            "backtrack.return",
            "继续返回当前观测位置。",
            _command_to_world_point(node.position_world_xy, frame.pose),
            {"node_id": node.node_id},
        )
    return _resume_pending_direction(frame, state, node)


def _resume_pending_direction(
    frame: NavigationFrame,
    state: SearchState,
    node: ObservationNode,
) -> NavigationResult:
    """在回退节点选择首个仍可用方向，并淘汰已被地图否定的方向。"""
    working_node = node
    working_state = state
    for direction in node.directions:
        if direction.state is not SearchDirectionState.PENDING:
            continue
        if not _candidate_still_available(
            direction.candidate_world_xy, frame.obstacle_map
        ):
            working_node = set_observation_direction_state(
                working_node,
                direction.direction_id,
                SearchDirectionState.INVALIDATED,
            )
            working_state = replace(
                working_state,
                observation_history=_replace_history_node(
                    working_state.observation_history, working_node
                ),
            )
            continue

        working_node = set_observation_direction_state(
            working_node,
            direction.direction_id,
            SearchDirectionState.COMMITTED,
        )
        working_state = replace(
            working_state,
            observation_history=_replace_history_node(
                working_state.observation_history, working_node
            ),
        )
        next_state = _reset_scan_after_move(working_state)
        return _result(
            NavigationStatus.OK,
            next_state,
            "backtrack.resume",
            "从历史观测节点恢复一个尚未探索的方向。",
            _command_to_world_point(direction.candidate_world_xy, frame.pose),
            {"node_id": node.node_id, "direction_id": direction.direction_id},
        )

    return _begin_backtracking(frame, replace(working_state, active_node_id=None))


def _start_scan(state: SearchState, current_heading: float) -> SearchState:
    """从当前位置建立一轮新的四向扫描状态。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=build_uniform_scan_headings(current_heading, 4),
        next_scan_index=0,
        scan_evidence=(),
        active_node_id=None,
        target_approach_attempts=0,
    )


def _reset_scan_after_move(state: SearchState) -> SearchState:
    """移动命令完成后，延迟到下一帧再按新的真实朝向建立扫描。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=(),
        next_scan_index=0,
        scan_evidence=(),
        active_node_id=None,
        target_approach_attempts=0,
    )


def _preferred_heading(evidence: Tuple[ScanEvidence, ...]) -> Optional[float]:
    """返回探索评分最高的扫描方向；没有评分时返回 None。"""
    scored = [item for item in evidence if item.direction_score is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: float(item.direction_score)).heading_world_rad


def _excluded_candidate_points(
    history: Tuple[ObservationNode, ...],
) -> Tuple[Tuple[float, float], ...]:
    """返回全部历史候选点，使新节点只记录真正新增的 Frontier。"""
    return tuple(
        direction.candidate_world_xy
        for node in history
        for direction in node.directions
        if direction.candidate_world_xy is not None
    )


def _mark_latest_committed_explored(state: SearchState) -> SearchState:
    """把最近一次已执行的候选方向标记为已探索。"""
    history = state.observation_history
    for node in reversed(history):
        for direction in node.directions:
            if direction.state is SearchDirectionState.COMMITTED:
                updated_node = set_observation_direction_state(
                    node, direction.direction_id, SearchDirectionState.EXPLORED
                )
                return replace(
                    state,
                    observation_history=_replace_history_node(history, updated_node),
                )
    return state


def _replace_history_node(
    history: Tuple[ObservationNode, ...],
    replacement: ObservationNode,
) -> Tuple[ObservationNode, ...]:
    """按 node_id 替换一个不可变历史节点。"""
    return tuple(
        replacement if node.node_id == replacement.node_id else node
        for node in history
    )


def _candidate_still_available(
    candidate_world_xy: Optional[Tuple[float, float]],
    obstacle_map: ObstacleMap,
) -> bool:
    """候选点越界或已知为障碍时返回 False；未知状态保留历史判断。"""
    if candidate_world_xy is None:
        return False
    try:
        row, col = world_to_nearest_grid_cell(candidate_world_xy, obstacle_map)
        rows = obstacle_map.occupancy
        if row < 0 or col < 0 or row >= len(rows) or col >= len(rows[row]):
            return False
        value = rows[row][col]
        return value is None or (_is_finite(value) and float(value) <= 0.5)
    except (TypeError, ValueError, IndexError):
        return False


def _command_to_world_point(
    target_world_xy: Tuple[float, float],
    pose: Pose2D,
) -> RelativePoseCommand:
    """把世界系目标点转换为统一的机器人相对位姿命令。"""
    forward_m, left_m = world_point_to_robot(target_world_xy, pose)
    target_heading = math.atan2(
        target_world_xy[1] - pose.y_m,
        target_world_xy[0] - pose.x_m,
    )
    return RelativePoseCommand(
        forward_m=forward_m,
        left_m=left_m,
        yaw_rad=shortest_turn_to_heading(pose.yaw_rad, target_heading),
    )


def _distance_to_node(pose: Pose2D, node: ObservationNode) -> float:
    return math.hypot(
        pose.x_m - node.position_world_xy[0],
        pose.y_m - node.position_world_xy[1],
    )


def _validation_error(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState],
    observation: Optional[TargetObservation],
) -> Optional[str]:
    """返回非法公共输入的简短原因；合法输入返回 None。"""
    if not isinstance(goal, TargetSearchGoal) or not isinstance(
        goal.target_text, str
    ) or not goal.target_text.strip():
        return "目标文本必须为非空字符串"
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
    if state is not None and not isinstance(state, SearchState):
        return "state 必须为 SearchState 或 None"
    if observation is not None and not isinstance(observation, TargetObservation):
        return "observation 必须为 TargetObservation 或 None"
    if observation is not None:
        if not isinstance(observation.visibility, TargetVisibility):
            return "observation.visibility 必须为 TargetVisibility"
        if observation.direction_score is not None and (
            not _is_finite(observation.direction_score)
            or not 0.0 <= float(observation.direction_score) <= 1.0
        ):
            return "observation.direction_score 必须位于 0 到 1"
    if state is not None:
        if not isinstance(state.phase, SearchPhase):
            return "state.phase 必须为 SearchPhase"
        if (
            isinstance(state.target_approach_attempts, bool)
            or not isinstance(state.target_approach_attempts, int)
            or state.target_approach_attempts < 0
        ):
            return "state.target_approach_attempts 必须为非负整数"
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


def _invalid_result(state: Optional[SearchState], reason: str) -> NavigationResult:
    failed_state = (
        replace(state, phase=SearchPhase.FAILED)
        if isinstance(state, SearchState)
        else SearchState(phase=SearchPhase.FAILED)
    )
    return _result(
        NavigationStatus.INVALID_INPUT,
        failed_state,
        "input",
        reason,
    )


def _result(
    status: NavigationStatus,
    state: SearchState,
    stage: str,
    message: str,
    command: Optional[RelativePoseCommand] = None,
    details: Optional[Mapping[str, Any]] = None,
) -> NavigationResult:
    """集中构造单周期结果，使各算法步骤只描述状态变化。"""
    return NavigationResult(
        status=status,
        command=command,
        debug=NavigationDebug(stage=stage, message=message, details=details or {}),
        state=state,
    )


def _is_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["navigate"]
