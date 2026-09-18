"""语义目标搜索的单周期状态机。

阅读入口是 :func:`navigate`。主流程只有四步：扫描环境、靠近可见目标、
更新 Frontier 区域、选择下一次完整移动的目标。所有跨周期信息都显式保存在
``SearchState`` 中；新 Frontier 耗尽后沿当前分支逐个返回，直到节点还有探索方向。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple


from .frontier import (
    PATH_DISTANCE_SCORE_WEIGHT,
    SEMANTIC_SCORE_WEIGHT,
    extract_frontiers,
)
from .geometry import (
    world_point_to_robot,
)
from .grounding import ground_target_bbox
from .history import (
    defer_unselected_frontiers,
    filter_blocked_frontier_regions,
    freeze_observation_node,
    match_frontier_regions,
    set_observation_direction_state,
)
from .models import (
    BlockedFrontierRegion,
    FrontierCandidate,
    FrontierRegion,
    FrontierScoreRequest,
    NavigationDebug,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObjectApproachState,
    ObjectLocalization,
    ObservationNode,
    ObservationView,
    ObstacleMap,
    Pose2D,
    RelativePoseCommand,
    ScanEvidence,
    SceneAssessment,
    SceneAssessmentResult,
    SearchDirection,
    SearchDirectionState,
    SearchPhase,
    SearchMode,
    SearchState,
    TargetConfirmation,
    TargetClue,
    TargetConfirmationResult,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from .observation_coverage import (
    camera_world_position,
    capture_observation_view,
    frontier_observation_points,
    unobserved_observation_points,
)
from .scan import (
    build_unobserved_scan_headings,
    build_uniform_scan_headings,
    shortest_turn_to_heading,
)
from .object_approach import continue_object_history, navigate_object_approach, recover_object_motion


TURN_TOLERANCE_RAD = math.radians(5.0)
BACKTRACK_ARRIVAL_M = 0.25
TARGET_STANDOFF_M = 0.75
TARGET_REACHED_M = 0.90
TARGET_INVALID_DEPTH_FALLBACK_M = 5.0
REJECTED_TARGET_RADIUS_M = 0.75
INITIAL_SCAN_TURN_COUNT = 8
INITIAL_SCAN_STEP_RAD = 2.0 * math.pi / INITIAL_SCAN_TURN_COUNT


def navigate(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    observation: Optional[TargetObservation] = None,
    frontier_scores: Optional[Mapping[str, float]] = None,
    target_confirmation: Optional[TargetConfirmationResult] = None,
    scene_assessment: Optional[SceneAssessmentResult] = None,
    object_localization: Optional[ObjectLocalization] = None,
) -> NavigationResult:
    """推进一个导航周期，并返回本周期命令和下一周期状态。

    ``frame`` 来自底盘 Adapter；``observation`` 来自独立的目标视觉模块。
    函数本身不读设备、不调用模型，也不发送命令。
    """
    reason = _validation_error(
        frame,
        goal,
        state,
        observation,
        frontier_scores,
        target_confirmation,
        scene_assessment,
        object_localization,
    )
    if reason is not None:
        return _invalid_result(state, reason)

    working_state = state or SearchState()
    if working_state.phase is SearchPhase.STOPPED:
        return _result(NavigationStatus.OK, working_state, "stopped", "导航已经停止。")
    if working_state.phase is SearchPhase.COMPLETE:
        return _result(
            NavigationStatus.OK,
            working_state,
            "complete",
            "语义搜索已经完成。",
        )
    if working_state.phase is SearchPhase.FAILED:
        return _result(
            NavigationStatus.NO_SOLUTION,
            working_state,
            "failed",
            "语义搜索已经结束，当前状态没有可继续的方向。",
        )
    if working_state.active_target_clue is not None:
        if goal.search_mode is SearchMode.OBJECT:
            return navigate_object_approach(frame, working_state, object_localization)
        return _continue_target_clue(frame, working_state)
    if goal.search_mode is SearchMode.OBJECT:
        history_result = continue_object_history(frame, working_state)
        if history_result is not None:
            return history_result
    if working_state.phase is SearchPhase.VERIFYING_SCENE:
        return _continue_scene_assessment(
            frame,
            working_state,
            scene_assessment,
        )
    if working_state.phase is SearchPhase.VERIFYING_TARGET:
        return _continue_target_confirmation(
            frame,
            goal,
            working_state,
            target_confirmation,
        )
    # 同步观察器保留原目标观测接口；CLI 的扫描只产生后台待处理结果。
    if (
        goal.search_mode is SearchMode.OBJECT
        and observation is not None
        and observation.visibility is TargetVisibility.VISIBLE
    ):
        return _approach_visible_target(frame, goal, working_state, observation)
    if working_state.phase is SearchPhase.LOCALIZING_TARGET:
        return _continue_target_approach(frame, goal, working_state, observation)
    if working_state.phase is SearchPhase.WAITING_FOR_SEMANTICS:
        try:
            working_state, candidates = _refresh_frontier_regions(frame, working_state)
        except ValueError as exc:
            return _invalid_result(working_state, str(exc))
        if candidates:
            return _continue_scanning(frame, goal, _reset_scan_after_move(working_state), observation)
        return _wait_for_semantics_or_finish(working_state)
    if working_state.phase is SearchPhase.BACKTRACKING:
        return _continue_backtracking(frame, working_state)
    if working_state.phase is SearchPhase.EXPLORING:
        return _select_exploration_target(
            frame,
            working_state,
            frontier_scores,
        )

    return _continue_scanning(frame, goal, working_state, observation)


def recover_from_motion_failure(
    result: NavigationResult,
    reason: str,
    *,
    rejected_path_world_xy: Tuple[Tuple[float, float], ...] = (),
) -> Optional[NavigationResult]:
    """Frontier 普通失败淘汰目标点；未知路径长度超限时屏蔽整个连通区域。"""
    object_recovery = recover_object_motion(result, reason)
    if object_recovery is not None:
        return object_recovery
    target_recovery = _reobserve_target_after_motion_issue(
        result,
        reason,
        issue_kind="failed",
        message="目标接近移动未完成，保留底盘实际位置并重新观测目标。",
    )
    if target_recovery is not None:
        return target_recovery

    if result.debug.details.get("next_stage", result.debug.stage) == "scan.turn":
        return _recover_scan_turn(result, reason)
    if result.debug.details.get("next_stage", result.debug.stage) in ("target.revisit", "target.revisit_turn"):
        return _discard_target_clue(result.state, reason)
    if result.debug.details.get("next_stage", result.debug.stage) == "backtrack.return":
        return _recover_backtrack_issue(
            result.state, reason,
            issue_kind="path_unknown" if rejected_path_world_xy else "failed",
            rejected_path_world_xy=rejected_path_world_xy,
        )
    if result.debug.details.get("next_stage", result.debug.stage) not in (
        "explore.select", "backtrack.resume",
    ):
        return None

    direction_id = result.debug.details.get("candidate_id")
    requested_node_id = result.debug.details.get("node_id")
    blocked_regions = result.state.blocked_frontier_regions
    if rejected_path_world_xy:
        region = next((region for region in result.state.frontier_regions
                       if region.region_id == direction_id), None)
        if region is None:
            # 缺少完整边界时不能退回点屏蔽，否则会再次进入换点重试循环。
            return None
        blocked_regions += (BlockedFrontierRegion(
            region_id=region.region_id,
            boundary_world_xy=region.boundary_world_xy,
        ),)
    for node in reversed(result.state.observation_history):
        if requested_node_id is not None and node.node_id != requested_node_id:
            continue
        if not any(
            direction.direction_id == direction_id
            and direction.state is SearchDirectionState.COMMITTED
            for direction in node.directions
        ):
            continue
        updated_node = set_observation_direction_state(
            node,
            str(direction_id),
            SearchDirectionState.INVALIDATED,
            execution_reason=reason,
        )
        next_state = replace(
            result.state,
            phase=SearchPhase.SCANNING,
            scan_headings_world_rad=(),
            next_scan_index=0,
            scan_evidence=(),
            active_frontier_id=None,
            target_approach_attempts=0,
            blocked_frontier_regions=blocked_regions,
            frontier_regions=tuple(
                region for region in result.state.frontier_regions
                if not rejected_path_world_xy or region.region_id != direction_id
            ),
            observation_history=_replace_history_node(
                result.state.observation_history, updated_node
            ),
        )
        return _result(
            NavigationStatus.OK,
            next_state,
            "motion.frontier_rejected",
            (
                "路径中的未知长度超限，已屏蔽整个 Frontier 区域，本次运行中不再重试该区域。"
                if rejected_path_world_xy
                else "本次探索动作执行失败，先检查当前位置，再重新选择区域。"
            ),
            details={
                "direction_id": direction_id,
                "rejected_stage": result.debug.stage,
                "reason": str(reason),
                "rejection_scope": "region" if rejected_path_world_xy else "point",
                "blocked_frontier_region_count": len(blocked_regions),
                "rejected_path_world_xy": rejected_path_world_xy,
            },
        )
    return None


def continue_after_motion_stall(
    result: NavigationResult,
    reason: str,
) -> Optional[NavigationResult]:
    """移动停滞时保留当前位置，并按原算法阶段继续。"""
    object_recovery = recover_object_motion(result, reason)
    if object_recovery is not None:
        return object_recovery
    target_recovery = _reobserve_target_after_motion_issue(
        result,
        reason,
        issue_kind="stalled",
        message="目标接近移动已停滞，按底盘实际位置结束本次动作并重新观测目标。",
    )
    if target_recovery is not None:
        return target_recovery

    if result.debug.details.get("next_stage", result.debug.stage) == "scan.turn":
        return _recover_scan_turn(result, reason)
    if result.debug.details.get("next_stage", result.debug.stage) in ("target.revisit", "target.revisit_turn"):
        return _discard_target_clue(result.state, reason)
    if result.debug.details.get("next_stage", result.debug.stage) == "backtrack.return":
        return _recover_backtrack_issue(result.state, reason, issue_kind="stalled")
    if result.debug.details.get("next_stage", result.debug.stage) not in (
        "explore.select", "backtrack.resume",
    ):
        return None
    if result.state.phase is not SearchPhase.SCANNING:
        return None
    direction_id = result.debug.details.get(
        "candidate_id",
        result.debug.details.get("direction_id"),
    )
    stalled_state = result.state
    for node in reversed(stalled_state.observation_history):
        if any(
            item.direction_id == direction_id
            and item.state is SearchDirectionState.COMMITTED
            for item in node.directions
        ):
            updated = set_observation_direction_state(
                node, str(direction_id), SearchDirectionState.STALLED,
                execution_reason=reason,
            )
            stalled_state = replace(
                stalled_state,
                active_frontier_id=None,
                observation_history=_replace_history_node(
                    stalled_state.observation_history, updated
                ),
            )
            break
    return _result(
        NavigationStatus.OK,
        stalled_state,
        "motion.stalled_continue",
        "移动连续静止达到门槛，按当前位置结束本次移动并继续扫描。",
        details={
            "direction_id": direction_id,
            "stalled_stage": result.debug.stage,
            "reason": str(reason),
        },
    )


def continue_after_target_detection(
    result: NavigationResult,
    reason: str,
) -> Optional[NavigationResult]:
    """运动被后台目标检测打断后，保留实际位置并让下一帧处理检测。"""
    if result.command is None:
        return None
    next_state, released_direction_id = _release_interrupted_direction(result)
    return _result(
        NavigationStatus.OK,
        next_state,
        "motion.target_detected",
        "运动期间检测到目标候选，已停止当前动作并将在下一帧定位目标。",
        details={
            "interrupted_stage": result.debug.stage,
            "released_direction_id": released_direction_id,
            "reason": str(reason),
        },
    )


def _recover_scan_turn(result: NavigationResult, reason: str) -> NavigationResult:
    """转向已停止后按下一帧真实朝向重建剩余观察，不假装该方向已检查。"""
    state = _reset_scan_after_move(result.state)
    return _result(
        NavigationStatus.OK, state, "motion.scan_recovered",
        "扫描转向未完成，按实际朝向和已有观察记录重新规划剩余视角。",
        details={"reason": str(reason)},
    )


def _release_interrupted_direction(
    result: NavigationResult,
) -> Tuple[SearchState, Optional[str]]:
    """Frontier 移动被感知打断时，把未到达的方向恢复为待探索。"""
    stage = result.debug.details.get("next_stage", result.debug.stage)
    if stage == "backtrack.return":
        # 返回途中尚未尝试暂存 Frontier，保留队列，下一帧优先处理检测。
        return _reset_scan_after_move(result.state), None
    if stage not in ("explore.select", "backtrack.resume"):
        return result.state, None
    direction_id = result.debug.details.get(
        "candidate_id",
        result.debug.details.get("direction_id"),
    )
    requested_node_id = result.debug.details.get("node_id")
    if not isinstance(direction_id, str) or not direction_id:
        return result.state, None

    for node in reversed(result.state.observation_history):
        if requested_node_id is not None and node.node_id != requested_node_id:
            continue
        if not any(
            direction.direction_id == direction_id
            and direction.state is SearchDirectionState.COMMITTED
            for direction in node.directions
        ):
            continue
        updated_node = set_observation_direction_state(
            node,
            direction_id,
            SearchDirectionState.PENDING,
        )
        return (
            replace(
                result.state,
                frontier_regions=tuple(
                    replace(
                        region,
                        deferred_order=(result.state.observation_history.index(node), 0),
                    )
                    if region.region_id == direction_id else region
                    for region in result.state.frontier_regions
                ),
                observation_history=_replace_history_node(
                    result.state.observation_history,
                    updated_node,
                ),
            ),
            direction_id,
        )
    return result.state, None


def _reobserve_target_after_motion_issue(
    result: NavigationResult,
    reason: str,
    issue_kind: str,
    message: str,
) -> Optional[NavigationResult]:
    """目标接近动作未完成时，保持目标模式并等待下一帧重新定位。"""
    if result.debug.stage != "target.approach":
        return None
    if result.state.phase is not SearchPhase.LOCALIZING_TARGET:
        return None
    return _result(
        NavigationStatus.OK,
        result.state,
        "motion.target_reobserve",
        message,
        details={
            "motion_issue": issue_kind,
            "reason": str(reason),
            "target_approach_attempts": result.state.target_approach_attempts,
        },
    )


def _continue_scanning(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    observation: Optional[TargetObservation],
) -> NavigationResult:
    """首次环扫；之后补查局部可见 Frontier，两种模式均至少采集当前画面。"""
    working_state = state
    scan_debug_details: Optional[Mapping[str, Any]] = None
    if not working_state.scan_headings_world_rad:
        try:
            working_state, candidates = _refresh_frontier_regions(frame, working_state)
            local_points = frontier_observation_points(frame, candidates)
            points = unobserved_observation_points(
                local_points, frame, working_state.observed_views + working_state.pending_observation_views,
            )
            headings = ()
            if not working_state.initial_scan_complete:
                headings = build_uniform_scan_headings(
                    frame.pose.yaw_rad + INITIAL_SCAN_STEP_RAD,
                    INITIAL_SCAN_TURN_COUNT,
                )
            elif points:
                camera_offset, horizontal_fov = _horizontal_camera_view(frame)
                camera_xy = camera_world_position(frame)
                headings = build_unobserved_scan_headings(
                    points, Pose2D(camera_xy[0], camera_xy[1], frame.pose.yaw_rad),
                    camera_offset, horizontal_fov,
                )
            if not headings:
                # 覆盖可复用也保留当前画面的目标检查，不为此额外转向。
                headings = (frame.pose.yaw_rad,)
        except ValueError as exc:
            return _result(
                NavigationStatus.MISSING_DATA, working_state, "scan.plan",
                f"无法规划待检查视角：{exc}",
            )
        scan_debug_details = {
            **_frontier_scan_debug_details(candidates),
            "local_observation_point_count": len(local_points),
            "observation_point_count": len(points),
            "reused_observation_point_count": len(local_points) - len(points),
            "checked_view_count": len(working_state.observed_views),
        }
        working_state = _start_scan_with_headings(
            replace(
                working_state,
                scan_observation_points=points,
                scan_local_point_count=len(local_points),
            ),
            headings,
        )
    return _advance_scan(
        frame, goal, working_state, observation, scan_debug_details,
    )


def _advance_scan(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    observation: Optional[TargetObservation],
    scan_debug_details: Optional[Mapping[str, Any]] = None,
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
            _scan_debug_details(
                state, target_heading, extra=scan_debug_details
            ),
        )

    if observation is None:
        return _result(
            NavigationStatus.NEEDS_OBSERVATION,
            state,
            "scan.observe",
            "当前扫描方向需要 TargetObserver 的视觉结果。",
            details=_scan_debug_details(
                state, target_heading, extra=scan_debug_details
            ),
        )
    if observation.visibility is TargetVisibility.UNCERTAIN:
        return _result(
            NavigationStatus.OK,
            state,
            "scan.observe",
            observation.reason or "视觉结果不确定，保持当前位置等待重新观测。",
            details=_scan_debug_details(
                state, target_heading, extra=scan_debug_details
            ),
        )
    if (
        goal.search_mode is SearchMode.OBJECT
        and observation.visibility is TargetVisibility.VISIBLE
    ):
        return _approach_visible_target(frame, goal, state, observation)

    try:
        camera_offset, horizontal_fov = _horizontal_camera_view(frame)
    except ValueError as exc:
        return _result(
            NavigationStatus.MISSING_DATA, state, "scan.observe",
            f"无法记录当前已检查视角：{exc}",
        )
    checked_view = capture_observation_view(
        frame, camera_offset, horizontal_fov,
        observation_points=state.scan_observation_points,
    )
    evidence = ScanEvidence(
        heading_world_rad=target_heading,
        visibility=observation.visibility,
        view=checked_view,
    )
    next_index = state.next_scan_index + 1
    scanned_state = replace(
        state,
        scan_evidence=state.scan_evidence + (evidence,),
        next_scan_index=min(next_index, len(state.scan_headings_world_rad) - 1),
        observed_views=(
            state.observed_views + (checked_view,)
            if goal.search_mode is SearchMode.OBJECT and observation.visibility is not TargetVisibility.PENDING
            else state.observed_views
        ),
    )
    if next_index < len(state.scan_headings_world_rad):
        return _continue_scanning(
            frame,
            goal,
            replace(scanned_state, next_scan_index=next_index),
            observation=None,
        )

    return _finish_scan(frame, goal, scanned_state)


def _finish_scan(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
) -> NavigationResult:
    """结束本轮采集并探索；同步场景观察器仍先判断当前位置。"""
    completed_state = replace(
        state,
        phase=(
            SearchPhase.VERIFYING_SCENE
            if goal.search_mode is SearchMode.SCENE and not state.asynchronous_perception
            else SearchPhase.EXPLORING
        ),
        initial_scan_complete=True,
    )
    if goal.search_mode is SearchMode.SCENE and not state.asynchronous_perception:
        return _continue_scene_assessment(frame, completed_state, None)
    return _select_exploration_target(
        frame,
        completed_state,
        frontier_scores=None,
    )


def _continue_target_approach(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    observation: Optional[TargetObservation],
) -> NavigationResult:
    """移动后重新观察目标，直到到达目标或目标丢失。"""
    if observation is None:
        return _result(
            NavigationStatus.NEEDS_OBSERVATION,
            state,
            "target.observe",
            "需要 TargetObserver 用当前画面重新观测目标。",
        )
    if observation.visibility is TargetVisibility.UNCERTAIN:
        return _result(
            NavigationStatus.OK,
            state,
            "target.observe",
            observation.reason or "目标观测不确定，保持当前位置等待重新观测。",
        )
    if observation.visibility is TargetVisibility.VISIBLE:
        return _approach_visible_target(frame, goal, state, observation)

    # 8×45° 环扫只发生在程序开始；目标丢失后按当前 Frontier 重新规划视角。
    scan_state = _reset_scan_after_move(
        replace(state, initial_scan_complete=True)
    )
    # 下一次观测必须带 scan_context，才能进入本轮批量评分图像缓冲。
    return _continue_scanning(frame, goal, scan_state, None)


def _continue_scene_assessment(
    frame: NavigationFrame,
    state: SearchState,
    assessment: Optional[SceneAssessmentResult],
) -> NavigationResult:
    """用当前观察确认场景，未匹配则结束线索或继续探索。"""
    if assessment is None:
        return _result(
            NavigationStatus.NEEDS_SCENE_ASSESSMENT,
            state,
            "scene.assess",
            "需要 VLM 根据当前观察判断是否已经位于目的场景。",
            details={"scan_image_count": len(state.scan_evidence)},
        )
    if assessment.assessment is SceneAssessment.UNCERTAIN:
        return _result(
            NavigationStatus.OK,
            state,
            "scene.assess",
            assessment.reason or "目的场景判断失败，保持当前位置等待重试。",
            details={"scan_image_count": len(state.scan_evidence)},
        )
    state = replace(
        state,
        observed_views=state.observed_views + tuple(
            evidence.view for evidence in state.scan_evidence if evidence.view is not None
        ),
    )
    if assessment.assessment is SceneAssessment.MATCHED:
        return _result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.COMPLETE, active_target_clue=None),
            "scene.complete",
            "VLM 确认机器人已经位于目的场景。",
            details={"scan_image_count": len(state.scan_evidence)},
        )

    return _select_exploration_target(
        frame,
        replace(state, phase=SearchPhase.EXPLORING),
        frontier_scores=None,
    )


def _continue_target_confirmation(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    confirmation: Optional[TargetConfirmationResult],
) -> NavigationResult:
    """候选已接近后等待 VLM 最终确认；否决后屏蔽该世界位置。"""
    target_world_xy = state.pending_target_world_xy
    if target_world_xy is None:
        return _result(
            NavigationStatus.MISSING_DATA,
            state,
            "target.confirm",
            "目标确认阶段缺少候选世界坐标。",
        )
    if confirmation is None:
        return _result(
            NavigationStatus.NEEDS_TARGET_CONFIRMATION,
            state,
            "target.confirm",
            "候选已进入观察距离，需要 VLM 最终确认。",
            details={"target_world_xy": target_world_xy},
        )
    if confirmation.confirmation is TargetConfirmation.UNCERTAIN:
        return _result(
            NavigationStatus.OK,
            state,
            "target.confirm",
            confirmation.reason or "VLM 最终确认失败，保持当前位置等待重试。",
            details={"target_world_xy": target_world_xy},
        )
    if confirmation.confirmation is TargetConfirmation.CONFIRMED:
        return _result(
            NavigationStatus.OK,
            replace(
                state,
                phase=SearchPhase.COMPLETE,
                pending_target_world_xy=None,
                active_target_clue=None,
            ),
            "target.complete",
            "VLM 已最终确认目标。",
            details={"target_world_xy": target_world_xy},
        )

    rejected_points = state.rejected_target_world_xy + (target_world_xy,)
    rejected_state = replace(
        state,
        initial_scan_complete=True,
        rejected_target_world_xy=rejected_points,
        pending_target_world_xy=None,
    )
    result = _continue_scanning(frame, goal, _reset_scan_after_move(rejected_state), None)
    return replace(
        result,
        debug=NavigationDebug(
            stage="target.rejected",
            message="VLM 否决当前候选，已屏蔽该位置，继续处理剩余线索或探索。",
            details={
                **result.debug.details,
                "rejected_target_world_xy": target_world_xy,
                "rejected_target_count": len(rejected_points),
                "next_stage": result.debug.stage,
            },
        ),
    )


def _approach_visible_target(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    observation: TargetObservation,
) -> NavigationResult:
    """用目标掩码内深度定位目标，并生成保持安全距离的相对位姿命令。"""
    state = replace(state, backtrack_node_id=None)
    if (
        observation.bbox_norm is None
        or frame.depth is None
        or frame.camera_intrinsics is None
    ):
        return _result(
            NavigationStatus.MISSING_DATA,
            replace(state, phase=SearchPhase.LOCALIZING_TARGET),
            "target.ground",
            "目标可见，但缺少目标框、深度图或相机内参。",
        )

    estimate = ground_target_bbox(
        observation.bbox_norm,
        frame.depth,
        frame.camera_intrinsics,
        frame.pose,
        frame.camera_extrinsics_in_robot,
        min_valid_points=1,
        no_valid_depth_fallback_m=TARGET_INVALID_DEPTH_FALLBACK_M,
        target_mask=observation.target_mask,
    )
    if not estimate.success:
        return _result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.LOCALIZING_TARGET),
            "target.ground",
            f"目标位置估计失败：{estimate.reason}",
            details={"valid_depth_points": estimate.sample_count},
        )

    if (
        estimate.target_world_xy is not None
        and _is_rejected_target(
            estimate.target_world_xy,
            state.rejected_target_world_xy,
        )
    ):
        if state.phase is SearchPhase.SCANNING:
            # 已被否决的重复检测不重置整轮扫描，否则该视角永远无法完成。
            result = _continue_scanning(
                frame, goal, state,
                TargetObservation(TargetVisibility.NOT_VISIBLE),
            )
        else:
            scan_state = _reset_scan_after_move(
                replace(state, initial_scan_complete=True)
            )
            result = _continue_scanning(frame, goal, scan_state, None)
        return replace(
            result,
            debug=NavigationDebug(
                stage="target.ignore_rejected",
                message="本地检测落在已被 VLM 否决的位置，忽略并继续探索。",
                details={
                    **result.debug.details,
                    "target_world_xy": estimate.target_world_xy,
                    "next_stage": result.debug.stage,
                },
            ),
        )

    if estimate.distance_m is not None and estimate.distance_m <= TARGET_REACHED_M:
        return _result(
            NavigationStatus.NEEDS_TARGET_CONFIRMATION,
            replace(
                state,
                phase=SearchPhase.VERIFYING_TARGET,
                pending_target_world_xy=estimate.target_world_xy,
            ),
            "target.confirm",
            "本地检测目标已进入观察距离，等待 VLM 最终确认。",
            details={
                "target_distance_m": estimate.distance_m,
                "target_world_xy": estimate.target_world_xy,
                "detection_source": observation.source,
                "detection_confidence": observation.confidence,
            },
        )

    target_forward, target_left = estimate.target_base_xy or (0.0, 0.0)
    distance = estimate.distance_m or math.hypot(target_forward, target_left)
    movement_scale = max(0.0, distance - TARGET_STANDOFF_M) / distance
    command = RelativePoseCommand(
        forward_m=target_forward * movement_scale,
        left_m=target_left * movement_scale,
        yaw_rad=estimate.bearing_rad or 0.0,
    )
    used_depth_fallback = (
        estimate.reason == "target_grounded_with_depth_fallback"
    )
    if used_depth_fallback:
        approach_message = (
            "目标区域没有有效深度，按 5 m 上界估计位置并直接接近；"
            "到达后重新观测。"
        )
    elif observation.target_mask is not None:
        approach_message = (
            "按 SAM2 掩码内深度直接接近目标，并保留安全停靠距离；"
            "到达后重新观测。"
        )
    else:
        approach_message = (
            "按当前深度估计直接接近目标，并保留安全停靠距离；"
            "到达后重新观测。"
        )
    return _result(
        NavigationStatus.OK,
        replace(
            state,
            phase=SearchPhase.LOCALIZING_TARGET,
            target_approach_attempts=state.target_approach_attempts + 1,
        ),
        "target.approach",
        approach_message,
        command,
        {
            "command_distance_m": math.hypot(command.forward_m, command.left_m),
            "depth_fallback_m": (
                TARGET_INVALID_DEPTH_FALLBACK_M
                if used_depth_fallback
                else None
            ),
            "depth_fallback_used": used_depth_fallback,
            "target_mask_used": observation.target_mask is not None,
            "target_distance_m": distance,
            "valid_depth_points": estimate.sample_count,
        },
    )


def _select_exploration_target(
    frame: NavigationFrame,
    state: SearchState,
    frontier_scores: Optional[Mapping[str, float]],
) -> NavigationResult:
    """优先选择新 Frontier；新候选耗尽时沿当前分支逐个返回父节点。"""
    explored_state = _mark_latest_committed_explored(state)
    try:
        explored_state, candidates = _refresh_frontier_regions(frame, explored_state)
    except ValueError as exc:
        return _invalid_result(state, f"障碍图无法用于 Frontier：{exc}")
    if not candidates:
        return _begin_backtracking(frame, explored_state, ())

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
        return _begin_backtracking(frame, explored_state, candidates)
    details = tuple(_frontier_candidate_debug(candidate) for candidate in candidates)
    selection_details = {
        "frontier_selection_source": "new" if ranked_new else "deferred",
        "new_frontier_count": len(ranked_new),
        "deferred_frontier_count": len(deferred_candidates),
    }
    if frontier_scores is None and len(ranked_new) > 1 and (
        explored_state.scan_evidence or explored_state.asynchronous_perception
    ):
        return _result(
            NavigationStatus.NEEDS_FRONTIER_SCORES,
            replace(explored_state, phase=SearchPhase.EXPLORING),
            "frontier.score", "只对新 Frontier 评分，旧方向保持暂存顺序。",
            details={
                **selection_details,
                "candidate_count": len(candidates), "frontier_candidates": details,
            },
            frontier_score_request=FrontierScoreRequest(ranked_new),
        )

    return _commit_frontier_move(
        frame, explored_state, candidates, ranked_new, frontier_scores,
    )


def _commit_frontier_move(
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
    next_state = _reset_scan_after_move(replace(
        state,
        observation_history=state.observation_history + (node,),
        branch_node_ids=state.branch_node_ids + (node.node_id,),
        frontier_regions=defer_unselected_frontiers(
            state.frontier_regions, ranked_new, selected.candidate_id,
            len(state.observation_history),
        ),
        active_frontier_id=selected.candidate_id,
    ))
    return _result(
        NavigationStatus.OK, next_state,
        "backtrack.resume" if parent_node_id is not None else "explore.select",
        (
            "优先探索新 Frontier，其余新方向暂存；等待完整移动结束后再决策。"
            if ranked_new
            else "已回到父节点，按保存顺序恢复该节点仍有效的探索方向。"
        ),
        _command_to_world_point(destination, frame.pose),
        {
            **_scan_coverage_details(state),
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
            "frontier_candidates": tuple(_frontier_candidate_debug(candidate) for candidate in candidates),
            "destination_world_xy": destination,
        },
    )


def _begin_backtracking(
    frame: NavigationFrame,
    state: SearchState,
    candidates: Tuple[FrontierCandidate, ...],
) -> NavigationResult:
    """只返回栈顶节点；到达且没有剩余方向才出栈，继续返回上一层。"""
    working_state = replace(_reset_scan_after_move(state), active_frontier_id=None)
    nodes = {node.node_id: node for node in working_state.observation_history}
    while working_state.branch_node_ids:
        node = nodes[working_state.branch_node_ids[-1]]
        next_state = replace(
            working_state, phase=SearchPhase.BACKTRACKING, backtrack_node_id=node.node_id,
        )
        distance = _distance_to_node(frame.pose, node)
        if distance > BACKTRACK_ARRIVAL_M:
            node_index = working_state.observation_history.index(node)
            return _result(
                NavigationStatus.OK,
                next_state,
                "backtrack.return",
                "返回当前分支的上一节点；到达后检查方向，没有剩余方向再退一层。",
                _command_to_world_point(node.position_world_xy, frame.pose),
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
                        _frontier_candidate_debug(candidate) for candidate in candidates
                    ),
                    "distance_to_parent_m": distance,
                },
            )
        resumed = _resume_pending_direction(frame, next_state, node, candidates)
        if resumed is not None:
            return resumed
        # 只有实际到达且方向已耗尽的节点才能退栈；历史记录仍保留用于屏蔽和复盘。
        working_state = replace(
            working_state, branch_node_ids=working_state.branch_node_ids[:-1],
        )
        if any(candidate.deferred_order is None for candidate in candidates):
            return _select_exploration_target(frame, working_state, None)

    return _wait_for_semantics_or_finish(working_state)


def _wait_for_semantics_or_finish(state: SearchState) -> NavigationResult:
    """没有可走方向时先排空已提交的检测；待处理不等于目标不存在。"""
    pending = state.pending_semantic_jobs
    return _result(
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
            **_scan_coverage_details(state),
            "pending_semantic_jobs": pending, "failed_semantic_jobs": state.failed_semantic_jobs,
        },
    )


def _continue_target_clue(
    frame: NavigationFrame, state: SearchState,
) -> NavigationResult:
    """场景线索返回拍摄位置并对齐朝向，到位即完成。"""
    clue = state.active_target_clue
    if clue.map_frame_id != frame.obstacle_map.frame_id:
        return _discard_target_clue(state, "目标线索与当前地图坐标系不同。")
    distance = math.hypot(frame.pose.x_m - clue.pose.x_m, frame.pose.y_m - clue.pose.y_m)
    if state.phase is SearchPhase.REVISITING_TARGET and distance > BACKTRACK_ARRIVAL_M:
        return _discard_target_clue(state, f"返回线索位置的动作结束后仍相距 {distance:.3f} m。")
    if distance > BACKTRACK_ARRIVAL_M:
        command = _command_to_world_point((clue.pose.x_m, clue.pose.y_m), frame.pose)
        command = replace(command, yaw_rad=shortest_turn_to_heading(frame.pose.yaw_rad, clue.pose.yaw_rad))
        return _result(
            NavigationStatus.OK, replace(state, phase=SearchPhase.REVISITING_TARGET, backtrack_node_id=None),
            "target.revisit", "后台检测到目标，返回当时的拍摄位置与朝向。", command,
            {"clue_id": clue.clue_id, "capture_timestamp_s": clue.timestamp_s,
             "destination_world_xy": (clue.pose.x_m, clue.pose.y_m)},
        )
    turn = shortest_turn_to_heading(frame.pose.yaw_rad, clue.pose.yaw_rad)
    if abs(turn) > TURN_TOLERANCE_RAD:
        return _result(
            NavigationStatus.OK, replace(state, phase=SearchPhase.REVISITING_TARGET, backtrack_node_id=None),
            "target.revisit_turn", "已回到拍摄位置，对齐检测画面当时的朝向。",
            RelativePoseCommand(yaw_rad=turn), {"clue_id": clue.clue_id},
        )
    return _result(
        NavigationStatus.OK,
        replace(_reset_scan_after_move(state), phase=SearchPhase.COMPLETE),
        "target.revisit_complete", "已返回目标场景画面的拍摄位置与朝向，搜索完成。",
        details={
            "clue_id": clue.clue_id, "capture_timestamp_s": clue.timestamp_s,
            "destination_world_xy": (clue.pose.x_m, clue.pose.y_m),
            "destination_yaw_rad": clue.pose.yaw_rad,
            "distance_to_capture_m": distance, "heading_error_rad": turn,
        },
    )


def _discard_target_clue(state: SearchState, reason: str) -> NavigationResult:
    """结束本条线索且不发命令，让下一周期优先取下一条；列表耗尽后恢复探索。"""
    clue = state.active_target_clue
    return _result(
        NavigationStatus.OK, _reset_scan_after_move(state), "target.revisit_failed",
        "未能返回本条线索的拍摄位姿，保留实际位置并尝试下一条；线索耗尽后继续探索。",
        details={"clue_id": clue.clue_id if clue is not None else None, "reason": reason},
    )


def _continue_backtracking(
    frame: NavigationFrame,
    state: SearchState,
) -> NavigationResult:
    """返回命令已同步结束，检查实际到达位置，再刷新父节点的剩余候选。"""
    node = next(node for node in state.observation_history
                if node.node_id == state.backtrack_node_id)
    distance = _distance_to_node(frame.pose, node)
    if distance > BACKTRACK_ARRIVAL_M:
        return _recover_backtrack_issue(
            state,
            f"返回动作已结束，距父节点 {distance:.3f} m，超过 {BACKTRACK_ARRIVAL_M:.2f} m 到达容差。",
            issue_kind="not_arrived", actual_pose=frame.pose,
        )
    try:
        state, candidates = _refresh_frontier_regions(frame, state)
    except ValueError as exc:
        return _invalid_result(state, f"回到父节点后无法刷新 Frontier：{exc}")
    return _begin_backtracking(frame, state, candidates)


def _recover_backtrack_issue(
    state: SearchState,
    reason: str,
    *,
    issue_kind: str,
    actual_pose: Optional[Pose2D] = None,
    rejected_path_world_xy: Tuple[Tuple[float, float], ...] = (),
) -> NavigationResult:
    """返回动作已结束但未完成：跳过失败节点，释放其暂存方向，从实际位置重新检查。"""
    node_index = next(index for index, node in enumerate(state.observation_history)
                      if node.node_id == state.backtrack_node_id)
    node = state.observation_history[node_index]
    released_ids = tuple(
        region.region_id for region in state.frontier_regions
        if region.deferred_order is not None and region.deferred_order[0] == node_index
    )
    next_state = _reset_scan_after_move(replace(
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
            distance_to_parent_m=_distance_to_node(actual_pose, node),
            arrival_tolerance_m=BACKTRACK_ARRIVAL_M,
        )
    return _result(
        NavigationStatus.OK, next_state, "motion.backtrack_recovered",
        "本次返回未完成，跳过该返回节点；保留其有效 Frontier，从实际位置重新检查并继续探索。",
        details=details,
    )


def _resume_pending_direction(
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
    return _commit_frontier_move(
        frame, state, ordered, (), None, parent_node_id=node.node_id,
    )


def _distance_to_node(pose: Pose2D, node: ObservationNode) -> float:
    """机器人与父节点位置的二维距离，单位为米。"""
    return math.hypot(pose.x_m - node.position_world_xy[0], pose.y_m - node.position_world_xy[1])


def _frontier_candidate_debug(
    candidate: FrontierCandidate,
) -> Mapping[str, Any]:
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


def _frontier_scan_debug_details(
    candidates: Tuple[FrontierCandidate, ...],
) -> Mapping[str, Any]:
    """返回本轮 Frontier 扫描计划的聚类来源。"""
    return {
        "frontier_scan_candidate_count": len(candidates),
        "frontier_scan_cell_count": len(
            {
                cell
                for candidate in candidates
                for cell in candidate.frontier_cells
            }
        ),
        "frontier_candidates": tuple(
            _frontier_candidate_debug(candidate)
            for candidate in candidates
        ),
    }


def _refresh_frontier_regions(
    frame: NavigationFrame,
    state: SearchState,
) -> Tuple[SearchState, Tuple[FrontierCandidate, ...]]:
    """重提边界、过滤仍被未知路径屏蔽的整片区域，再关联有效候选的稳定 ID。"""
    extraction = extract_frontiers(
        frame.obstacle_map, frame.pose,
        excluded_world_xy=_tried_candidate_points(state.observation_history),
        visibility_map=frame.visibility_map,
    )
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
    ), candidates


def preview_frontier_candidates(frame: NavigationFrame, state: SearchState) -> Tuple[FrontierCandidate, ...]:
    """对固定帧预览有效候选，不提交区域编号或修改导航状态；结果用于提前拍摄。"""
    return _refresh_frontier_regions(frame, state)[1]


def capture_semantic_view(
    frame: NavigationFrame, observation_points: Tuple[Tuple[float, float], ...] = (),
) -> ObservationView:
    """记录固定帧的真实视角与覆盖；调用方须在检测成功后才登记为已检查。"""
    offset, fov = _horizontal_camera_view(frame)
    return capture_observation_view(frame, offset, fov, observation_points=observation_points)


def _horizontal_camera_view(frame: NavigationFrame) -> Tuple[float, float]:
    """返回相机水平视场中心相对底盘的偏角，以及完整水平 FOV。"""
    intrinsics = frame.camera_intrinsics
    if intrinsics is None:
        raise ValueError("缺少相机内参")
    if not _is_finite(intrinsics.fx) or float(intrinsics.fx) <= 0.0:
        raise ValueError("相机 fx 必须为正有限值")
    if not _is_finite(intrinsics.cx):
        raise ValueError("相机 cx 必须为有限值")

    width = _camera_image_width(frame)
    focal_length = float(intrinsics.fx)
    principal_x = float(intrinsics.cx)
    left_pixels = principal_x + 0.5
    right_pixels = width - 0.5 - principal_x
    if left_pixels <= 0.0 or right_pixels <= 0.0:
        raise ValueError("相机 cx 必须位于图像水平范围内")

    left_extent = math.atan2(left_pixels, focal_length)
    right_extent = math.atan2(right_pixels, focal_length)
    intrinsic_center_offset = (left_extent - right_extent) / 2.0
    camera_yaw = frame.camera_extrinsics_in_robot.yaw_rad
    if not _is_finite(camera_yaw):
        raise ValueError("相机 yaw 外参必须为有限值")
    return (
        float(camera_yaw) + intrinsic_center_offset,
        left_extent + right_extent,
    )


def _camera_image_width(frame: NavigationFrame) -> int:
    """返回与相机内参对应的图像宽度；优先使用 RGB，随后使用对齐深度。"""
    image = frame.rgb if frame.rgb is not None else frame.depth
    if image is None:
        raise ValueError("缺少 RGB 或深度图像尺寸")
    try:
        height = len(image)
        width = len(image[0]) if height else 0
    except (TypeError, IndexError):
        raise ValueError("相机图像必须为非空二维数组") from None
    if width < 1:
        raise ValueError("相机图像宽度必须大于零")
    return width


def _start_scan_with_headings(
    state: SearchState,
    headings: Tuple[float, ...],
) -> SearchState:
    """清空上一轮证据并开始执行给定的世界系扫描朝向。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=headings,
        next_scan_index=0,
        scan_evidence=(),
        target_approach_attempts=0,
    )


def _scan_mode(state: SearchState) -> str:
    """区分首次环扫、Frontier 补查和无待查方向时的当前画面采集。"""
    if not state.initial_scan_complete:
        return "initial"
    return "frontier" if state.scan_observation_points else "current_view"


def _scan_debug_details(
    state: SearchState,
    target_heading: float,
    scan_index: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """返回一条扫描决策需要记录的最小上下文。"""
    details = {
        **_scan_coverage_details(state),
        "scan_index": state.next_scan_index if scan_index is None else scan_index,
        "scan_heading_count": len(state.scan_headings_world_rad),
        "scan_mode": _scan_mode(state),
        "target_heading_world_rad": target_heading,
        "checked_view_count": len(state.observed_views),
    }
    if extra is not None:
        details.update(extra)
    return details


def _scan_coverage_details(state: SearchState) -> Mapping[str, Any]:
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


def _reset_scan_after_move(state: SearchState) -> SearchState:
    """移动命令完成后，延迟到下一帧再按新的真实朝向建立扫描。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=(),
        next_scan_index=0,
        scan_evidence=(),
        scan_observation_points=(),
        scan_local_point_count=0,
        target_approach_attempts=0,
        pending_target_world_xy=None,
        backtrack_node_id=None,
        active_target_clue=None,
        object_approach=ObjectApproachState(),
    )


def _is_rejected_target(
    target_world_xy: Tuple[float, float],
    rejected_points: Tuple[Tuple[float, float], ...],
) -> bool:
    """判断本地检测是否落入已被 VLM 否决的位置邻域。"""
    return any(
        math.hypot(
            target_world_xy[0] - rejected_xy[0],
            target_world_xy[1] - rejected_xy[1],
        )
        <= REJECTED_TARGET_RADIUS_M
        for rejected_xy in rejected_points
    )


def _tried_candidate_points(
    history: Tuple[ObservationNode, ...],
) -> Tuple[Tuple[float, float], ...]:
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


def _mark_latest_committed_explored(state: SearchState) -> SearchState:
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



def _validation_error(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState],
    observation: Optional[TargetObservation],
    frontier_scores: Optional[Mapping[str, float]],
    target_confirmation: Optional[TargetConfirmationResult],
    scene_assessment: Optional[SceneAssessmentResult],
    object_localization: Optional[ObjectLocalization],
) -> Optional[str]:
    """返回非法公共输入的简短原因；合法输入返回 None。"""
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
        issue = _object_localization_error(object_localization)
        if issue is not None:
            return issue
    if observation is not None and not isinstance(observation, TargetObservation):
        return "observation 必须为 TargetObservation 或 None"
    if observation is not None:
        if not isinstance(observation.visibility, TargetVisibility):
            return "observation.visibility 必须为 TargetVisibility"
        if not isinstance(observation.source, str):
            return "observation.source 必须为字符串"
        if observation.confidence is not None and (
            not _is_finite(observation.confidence)
            or not 0.0 <= float(observation.confidence) <= 1.0
        ):
            return "observation.confidence 必须位于 0 到 1"
    if target_confirmation is not None:
        if not isinstance(target_confirmation, TargetConfirmationResult):
            return "target_confirmation 必须为 TargetConfirmationResult 或 None"
        if not isinstance(target_confirmation.confirmation, TargetConfirmation):
            return "target_confirmation.confirmation 类型无效"
    if scene_assessment is not None:
        if not isinstance(scene_assessment, SceneAssessmentResult):
            return "scene_assessment 必须为 SceneAssessmentResult 或 None"
        if not isinstance(scene_assessment.assessment, SceneAssessment):
            return "scene_assessment.assessment 类型无效"
    if frontier_scores is not None:
        if not isinstance(frontier_scores, Mapping):
            return "frontier_scores 必须为映射或 None"
        for candidate_id, score in frontier_scores.items():
            if not isinstance(candidate_id, str) or not candidate_id:
                return "frontier_scores 的键必须为非空字符串"
            if not _is_finite(score) or not 0.0 <= float(score) <= 1.0:
                return "frontier_scores 的分数必须位于 0 到 1"
    if state is not None:
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
            issue = _object_localization_error(context.target)
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
        if any(not _valid_world_point(point) for point in context.tried_positions):
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
            or not all(_valid_world_point(point) for point in region.boundary_world_xy)
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
            or not all(_valid_world_point(point) for point in region.boundary_world_xy)
            for region in state.blocked_frontier_regions
        ):
            return "state.blocked_frontier_regions 必须包含有效区域边界"
        if any(not _valid_world_point(point) for point in state.scan_observation_points):
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
            or (view.camera_world_xy is not None and not _valid_world_point(view.camera_world_xy))
            or not isinstance(view.depth_coverage_available, bool)
            or not all(_valid_world_point(point) for point in view.visible_world_xy)
            or not all(_valid_world_point(point) for point in view.map_visible_world_xy)
            for view in state.observed_views + state.pending_observation_views
        ):
            return "已检查与待处理覆盖必须包含有效 ObservationView"
        if not isinstance(state.initial_scan_complete, bool):
            return "state.initial_scan_complete 必须为 bool"
        if (
            state.pending_target_world_xy is not None
            and not _valid_world_point(state.pending_target_world_xy)
        ):
            return "state.pending_target_world_xy 必须为有限世界坐标或 None"
        if any(
            not _valid_world_point(point)
            for point in state.rejected_target_world_xy
        ):
            return "state.rejected_target_world_xy 必须为有限世界坐标序列"
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


def _object_localization_error(value) -> Optional[str]:
    if not isinstance(value, ObjectLocalization):
        return "物体定位结果必须为 ObjectLocalization"
    if value.target_world_xy is not None and not _valid_world_point(value.target_world_xy):
        return "物体定位结果包含非法世界坐标"
    if not isinstance(value.visibility, TargetVisibility) or not isinstance(value.vlm_confirmation, TargetConfirmation):
        return "物体定位的可见性或 VLM 确认类型无效"
    if not isinstance(value.source, str) or not isinstance(value.reason, str):
        return "物体定位来源与原因必须为字符串"
    if isinstance(value.sample_count, bool) or not isinstance(value.sample_count, int) or value.sample_count < 0:
        return "物体定位深度点数无效"
    return None


def _valid_world_point(value: object) -> bool:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    return all(_is_finite(component) for component in value)


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
    frontier_score_request: Optional[FrontierScoreRequest] = None,
) -> NavigationResult:
    """集中构造单周期结果，使各算法步骤只描述状态变化。"""
    return NavigationResult(
        status=status,
        command=command,
        debug=NavigationDebug(stage=stage, message=message, details=details or {}),
        state=state,
        frontier_score_request=frontier_score_request,
    )


def _is_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "capture_semantic_view",
    "continue_after_motion_stall",
    "continue_after_target_detection",
    "navigate",
    "preview_frontier_candidates",
    "recover_from_motion_failure",
]
