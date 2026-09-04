"""语义目标搜索的单周期状态机。

阅读入口是 :func:`navigate`。主流程只有四步：扫描环境、靠近可见目标、
选择 Frontier 探索、在无新路可走时回退。所有跨周期信息都显式保存在
``SearchState`` 中。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

from .frontier import (
    PATH_DISTANCE_SCORE_WEIGHT,
    SEMANTIC_SCORE_WEIGHT,
    find_frontier_candidates,
    is_world_point_reachable,
)
from .geometry import (
    grid_cell_center_to_world,
    world_point_to_robot,
)
from .grounding import ground_target_bbox
from .history import (
    find_latest_pending_observation_node,
    freeze_observation_node,
    set_observation_direction_state,
)
from .models import (
    FrontierCandidate,
    FrontierScoreRequest,
    NavigationDebug,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObservationNode,
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
    TargetConfirmationResult,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from .scan import (
    build_covering_scan_headings,
    build_uniform_scan_headings,
    shortest_turn_to_heading,
)


TURN_TOLERANCE_RAD = math.radians(5.0)
TARGET_STANDOFF_M = 0.75
TARGET_REACHED_M = 0.90
TARGET_INVALID_DEPTH_FALLBACK_M = 5.0
REJECTED_TARGET_RADIUS_M = 0.75
BACKTRACK_ARRIVAL_M = 0.25
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
    )
    if reason is not None:
        return _invalid_result(state, reason)

    working_state = state or SearchState()
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
    # 物体模式下 YOLO-World 持续观察；有效检测可以抢占扫描和探索。
    if (
        goal.search_mode is SearchMode.OBJECT
        and observation is not None
        and observation.visibility is TargetVisibility.VISIBLE
    ):
        return _approach_visible_target(frame, goal, working_state, observation)
    if working_state.phase is SearchPhase.BACKTRACKING:
        return _continue_backtracking(frame, working_state)
    if working_state.phase is SearchPhase.LOCALIZING_TARGET:
        return _continue_target_approach(frame, goal, working_state, observation)
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
) -> Optional[NavigationResult]:
    """目标接近失败时重新观测；Frontier 失败时淘汰当前候选。"""
    target_recovery = _reobserve_target_after_motion_issue(
        result,
        reason,
        issue_kind="failed",
        message="目标接近移动未完成，保留底盘实际位置并重新观测目标。",
    )
    if target_recovery is not None:
        return target_recovery

    if result.debug.stage not in ("explore.select", "backtrack.resume"):
        return None

    direction_key = (
        "candidate_id"
        if result.debug.stage == "explore.select"
        else "direction_id"
    )
    direction_id = result.debug.details.get(direction_key)
    requested_node_id = result.debug.details.get("node_id")
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
        )
        next_state = replace(
            result.state,
            phase=SearchPhase.BACKTRACKING,
            scan_headings_world_rad=(),
            next_scan_index=0,
            scan_evidence=(),
            active_node_id=node.node_id,
            target_approach_attempts=0,
            observation_history=_replace_history_node(
                result.state.observation_history, updated_node
            ),
        )
        return _result(
            NavigationStatus.OK,
            next_state,
            "motion.frontier_rejected",
            "当前 Frontier 无法到达，已淘汰并继续尝试剩余候选。",
            details={
                "direction_id": direction_id,
                "rejected_stage": result.debug.stage,
                "reason": str(reason),
            },
        )
    return None


def continue_after_motion_stall(
    result: NavigationResult,
    reason: str,
) -> Optional[NavigationResult]:
    """移动停滞时保留当前位置，并按原算法阶段继续。"""
    target_recovery = _reobserve_target_after_motion_issue(
        result,
        reason,
        issue_kind="stalled",
        message="目标接近移动已停滞，按底盘实际位置结束本次动作并重新观测目标。",
    )
    if target_recovery is not None:
        return target_recovery

    if result.debug.stage not in ("explore.select", "backtrack.resume"):
        return None
    if result.state.phase is not SearchPhase.SCANNING:
        return None
    direction_id = result.debug.details.get(
        "candidate_id",
        result.debug.details.get("direction_id"),
    )
    return _result(
        NavigationStatus.OK,
        result.state,
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


def _release_interrupted_direction(
    result: NavigationResult,
) -> Tuple[SearchState, Optional[str]]:
    """Frontier 移动被感知打断时，把未到达的方向恢复为待探索。"""
    if result.debug.stage not in ("explore.select", "backtrack.resume"):
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
    """首次连续转动八次 45°；后续只覆盖当前 Frontier 聚类代表点。"""
    working_state = state
    scan_debug_details: Optional[Mapping[str, Any]] = None
    if not working_state.scan_headings_world_rad:
        if not working_state.initial_scan_complete:
            # 从当前朝向的下一个 45° 开始，最后一次回到起始朝向；这样实际
            # 下发八次转向，而不只是把初始朝向算作八个视角中的第一个。
            headings = build_uniform_scan_headings(
                frame.pose.yaw_rad + INITIAL_SCAN_STEP_RAD,
                INITIAL_SCAN_TURN_COUNT,
            )
        else:
            try:
                candidates = _available_frontier_candidates(
                    frame, working_state, semantic_scores=None
                )
            except ValueError as exc:
                return _invalid_result(
                    working_state, f"障碍图无法用于 Frontier 扫描：{exc}"
                )
            if not candidates:
                if goal.search_mode is SearchMode.SCENE:
                    headings = (frame.pose.yaw_rad,)
                else:
                    return _select_exploration_target(
                        frame,
                        working_state,
                        frontier_scores=None,
                    )
            else:
                try:
                    headings = _frontier_scan_headings(frame, candidates)
                except ValueError as exc:
                    return _result(
                        NavigationStatus.MISSING_DATA,
                        working_state,
                        "scan.plan",
                        f"无法按相机视野规划 Frontier 扫描：{exc}",
                    )
                if not headings:
                    if goal.search_mode is SearchMode.SCENE:
                        headings = (frame.pose.yaw_rad,)
                    else:
                        return _select_exploration_target(
                            frame,
                            working_state,
                            frontier_scores=None,
                        )
                scan_debug_details = _frontier_scan_debug_details(candidates)

        working_state = _start_scan_with_headings(working_state, headings)
    return _advance_scan(
        frame,
        goal,
        working_state,
        observation,
        scan_debug_details,
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

    evidence = ScanEvidence(
        heading_world_rad=target_heading,
        visibility=TargetVisibility.NOT_VISIBLE,
    )
    next_index = state.next_scan_index + 1
    scanned_state = replace(
        state,
        scan_evidence=state.scan_evidence + (evidence,),
        next_scan_index=min(next_index, len(state.scan_headings_world_rad) - 1),
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
    """结束本轮采集；场景模式先判断当前位置，物体模式直接探索。"""
    completed_state = replace(
        state,
        phase=(
            SearchPhase.VERIFYING_SCENE
            if goal.search_mode is SearchMode.SCENE
            else SearchPhase.EXPLORING
        ),
        initial_scan_complete=True,
    )
    if goal.search_mode is SearchMode.SCENE:
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
    """整轮扫描结束后先判断当前位置，未到场景时再选择 Frontier。"""
    if assessment is None:
        return _result(
            NavigationStatus.NEEDS_SCENE_ASSESSMENT,
            state,
            "scene.assess",
            "本轮扫描完成，需要 VLM 判断是否已经位于目的场景。",
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
    if assessment.assessment is SceneAssessment.MATCHED:
        return _result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.COMPLETE),
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
            ),
            "target.complete",
            "VLM 已最终确认目标。",
            details={"target_world_xy": target_world_xy},
        )

    rejected_points = state.rejected_target_world_xy + (target_world_xy,)
    scan_state = _reset_scan_after_move(
        replace(
            state,
            initial_scan_complete=True,
            rejected_target_world_xy=rejected_points,
            pending_target_world_xy=None,
        )
    )
    result = _continue_scanning(frame, goal, scan_state, None)
    return replace(
        result,
        debug=NavigationDebug(
            stage="target.rejected",
            message="VLM 否决当前候选，已屏蔽该位置并恢复 Frontier 探索。",
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
    """多个 Frontier 先请求语义分数；单个 Frontier 直接选择。"""
    explored_state = _mark_latest_committed_explored(state)
    try:
        candidates = _available_frontier_candidates(
            frame,
            explored_state,
            semantic_scores=frontier_scores,
        )
    except ValueError as exc:
        return _invalid_result(state, f"障碍图无法用于 Frontier：{exc}")

    if not candidates:
        return _begin_backtracking(frame, explored_state)

    candidate_details = tuple(
        _frontier_candidate_debug(candidate)
        for candidate in candidates
    )
    skip_semantic_scoring = frontier_scores is None and len(candidates) == 1
    if frontier_scores is None and not skip_semantic_scoring:
        scoring_state = replace(explored_state, phase=SearchPhase.EXPLORING)
        return _result(
            NavigationStatus.NEEDS_FRONTIER_SCORES,
            scoring_state,
            "frontier.score",
            "需要对本轮全部 Frontier 一次性语义评分。",
            details={
                "candidate_count": len(candidates),
                "frontier_candidates": candidate_details,
            },
            frontier_score_request=FrontierScoreRequest(candidates),
        )

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
        (
            "仅有一个可达 Frontier，跳过语义评分并直接前往。"
            if skip_semantic_scoring
            else "选择评分最高的 Frontier，保存其余方向供后续回退。"
        ),
        _command_to_world_point(candidates[0].world_xy, frame.pose),
        {
            "candidate_id": candidates[0].candidate_id,
            "candidate_count": len(candidates),
            "candidate_score": candidates[0].score,
            "frontier_scoring_skipped": skip_semantic_scoring,
            "frontier_path_distance_weight": PATH_DISTANCE_SCORE_WEIGHT,
            "frontier_semantic_score_weight": SEMANTIC_SCORE_WEIGHT,
            "frontier_candidates": candidate_details,
        },
    )


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
        if not _candidate_still_reachable(
            direction.candidate_world_xy,
            frame.obstacle_map,
            frame.pose,
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


def _available_frontier_candidates(
    frame: NavigationFrame,
    state: SearchState,
    semantic_scores: Optional[Mapping[str, float]],
) -> Tuple[FrontierCandidate, ...]:
    """按当前地图和历史排除规则返回仍可探索的 Frontier。"""
    return find_frontier_candidates(
        frame.obstacle_map,
        frame.pose,
        semantic_scores=semantic_scores,
        excluded_world_xy=_excluded_candidate_points(state.observation_history),
    )


def _frontier_scan_headings(
    frame: NavigationFrame,
    candidates: Tuple[FrontierCandidate, ...],
) -> Tuple[float, ...]:
    """规划能够覆盖全部 Frontier 聚类代表点的最少相机视角。"""
    point_headings = tuple(
        candidate.heading_world_rad
        for candidate in candidates
    )
    if not point_headings:
        return ()
    camera_center_offset, horizontal_fov = _horizontal_camera_view(frame)
    return build_covering_scan_headings(
        point_headings,
        current_robot_heading_rad=frame.pose.yaw_rad,
        camera_center_offset_rad=camera_center_offset,
        horizontal_fov_rad=horizontal_fov,
    )


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
        active_node_id=None,
        target_approach_attempts=0,
    )


def _scan_mode(state: SearchState) -> str:
    """返回当前扫描类型，写入运行日志用于区分首次与后续扫描。"""
    return "frontier" if state.initial_scan_complete else "initial"


def _scan_debug_details(
    state: SearchState,
    target_heading: float,
    scan_index: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """返回一条扫描决策需要记录的最小上下文。"""
    details = {
        "scan_index": state.next_scan_index if scan_index is None else scan_index,
        "scan_heading_count": len(state.scan_headings_world_rad),
        "scan_mode": _scan_mode(state),
        "target_heading_world_rad": target_heading,
    }
    if extra is not None:
        details.update(extra)
    return details


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
        pending_target_world_xy=None,
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


def _candidate_still_reachable(
    candidate_world_xy: Optional[Tuple[float, float]],
    obstacle_map: ObstacleMap,
    pose: Pose2D,
) -> bool:
    """候选点仍属于机器人当前 BFS 可达自由区时返回 True。"""
    if candidate_world_xy is None:
        return False
    try:
        return is_world_point_reachable(obstacle_map, pose, candidate_world_xy)
    except (TypeError, ValueError):
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
    frontier_scores: Optional[Mapping[str, float]],
    target_confirmation: Optional[TargetConfirmationResult],
    scene_assessment: Optional[SceneAssessmentResult],
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
    if state is not None and not isinstance(state, SearchState):
        return "state 必须为 SearchState 或 None"
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
    "continue_after_motion_stall",
    "continue_after_target_detection",
    "navigate",
    "recover_from_motion_failure",
]
