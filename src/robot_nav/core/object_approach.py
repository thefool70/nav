"""先处理全部历史线索；成功则接近，全都不能定位才返回拍摄位姿并停住。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .geometry import world_point_to_robot
from .object_standoff import plan_object_standoff
from .models import (
    NavigationDebug, NavigationFrame, NavigationResult, NavigationStatus,
    ObjectApproachState, ObjectLocalization, Pose2D, RelativePoseCommand,
    SearchPhase, SearchState,
)


CAPTURE_ARRIVAL_M = 0.25
TURN_TOLERANCE_RAD = math.radians(5.0)
MAX_APPROACH_MOVES = 3


def navigate_object_approach(
    frame: NavigationFrame, state: SearchState,
    localization: Optional[ObjectLocalization] = None,
) -> NavigationResult:
    """每次只消费一个定位结果或生成一条完整运动命令。"""
    clue = state.active_target_clue
    if state.object_approach.fallback_clue is None:
        state = replace(state, object_approach=replace(state.object_approach, fallback_clue=clue))
    if clue.map_frame_id != frame.obstacle_map.frame_id:
        return _discard_clue(state, "物体线索与当前地图坐标系不同。")
    if state.phase is SearchPhase.REVISITING_TARGET:
        return _return_to_capture(frame, state)
    if state.phase is SearchPhase.APPROACHING_OBJECT:
        if state.object_approach.destination is None:
            return _plan_approach(frame, state)
        # 上一周期的停靠命令已同步执行完成；运动失败会清空 destination。
        return _complete_approach(state)
    if localization is None:
        return _request_localization(state)
    if localization.target_world_xy is None:
        return _discard_clue(state, localization.reason or "本张历史画面无法定位物体。")
    return _plan_approach(
        frame,
        replace(state, object_approach=replace(
            state.object_approach, target=localization, destination=None,
            history_localized=True,
        )),
    )


def recover_object_motion(
    result: NavigationResult, reason: str,
) -> Optional[NavigationResult]:
    """接近失败换停靠点；保底返回失败也直接停止。"""
    stage = result.debug.stage
    if stage in ("object.fallback_return", "object.fallback_turn"):
        return _stop_at_fallback(result.state, f"保底返回失败，停在当前位置：{reason}")
    if stage != "object.approach":
        return None
    state = replace(result.state, object_approach=replace(
        result.state.object_approach, destination=None,
    ))
    return _result(state, "object.motion_recovered", "本次停靠未完成，下一周期按最新地图换点。", reason=reason)


def _request_localization(state: SearchState) -> NavigationResult:
    return _result(
        replace(state, phase=SearchPhase.LOCALIZING_OBJECT), "object.localize",
        "用当前历史线索的 RGB-D 定位物体。",
        status=NavigationStatus.NEEDS_OBJECT_LOCALIZATION, localization_source="snapshot",
    )


def continue_object_history(frame: NavigationFrame, state: SearchState) -> Optional[NavigationResult]:
    """当前线索失败后先处理已采集的队列；全部历史都无法定位时才保底返回。"""
    context = state.object_approach
    if context.fallback_clue is None:
        return None
    if state.pending_semantic_jobs:
        return _result(
            replace(state, phase=SearchPhase.WAITING_FOR_SEMANTICS), "object.wait_history",
            "保持当前位置，继续处理已采集的历史画面与线索。",
            pending_semantic_jobs=state.pending_semantic_jobs,
        )
    if context.history_localized:
        return _result(
            replace(state, phase=SearchPhase.SCANNING, object_approach=ObjectApproachState()),
            "object.history_finished", "历史目标未能完成接近，恢复探索。",
        )
    if context.fallback_clue.map_frame_id != frame.obstacle_map.frame_id:
        return _stop_at_fallback(state, "保底拍摄位姿与当前地图不同，停在当前位置。")
    return _return_to_capture(frame, replace(state, phase=SearchPhase.SCANNING,
                                           active_target_clue=context.fallback_clue))


def _return_to_capture(frame: NavigationFrame, state: SearchState) -> NavigationResult:
    clue = state.active_target_clue
    context = state.object_approach
    distance = math.hypot(frame.pose.x_m - clue.pose.x_m, frame.pose.y_m - clue.pose.y_m)
    if state.phase is SearchPhase.REVISITING_TARGET and distance > CAPTURE_ARRIVAL_M:
        return _stop_at_fallback(state, f"保底返回结束后距拍摄点仍有 {distance:.2f} m，停在当前位置。")
    if distance > CAPTURE_ARRIVAL_M:
        return _result(
            replace(state, phase=SearchPhase.REVISITING_TARGET, backtrack_node_id=None),
            "object.fallback_return", "所有历史线索均无法定位，保底返回拍摄位姿后停止。",
            command=_command_to_pose(frame.pose, clue.pose),
            destination_world_xy=(clue.pose.x_m, clue.pose.y_m),
        )
    turn = _turn(frame.pose.yaw_rad, clue.pose.yaw_rad)
    if abs(turn) > TURN_TOLERANCE_RAD:
        if context.capture_turns >= 2:
            return _stop_at_fallback(state, "保底返回两次转向仍未恢复拍摄朝向，停止。")
        return _result(
            replace(state, phase=SearchPhase.REVISITING_TARGET, object_approach=replace(
                context, capture_turns=context.capture_turns + 1,
            )),
            "object.fallback_turn", "已回到拍摄点，对齐朝向后停止。",
            command=RelativePoseCommand(yaw_rad=turn),
        )
    return _stop_at_fallback(state, "所有历史线索均无法定位；已返回拍摄位姿，停止导航。")


def _stop_at_fallback(state: SearchState, reason: str) -> NavigationResult:
    return _result(
        replace(state, phase=SearchPhase.STOPPED, active_target_clue=None),
        "object.fallback_stopped", reason,
    )


def _plan_approach(frame: NavigationFrame, state: SearchState) -> NavigationResult:
    context = state.object_approach
    target = context.target
    if target is None or target.target_world_xy is None:
        return _discard_clue(state, "接近阶段缺少目标位置。")
    if len(context.tried_positions) >= MAX_APPROACH_MOVES:
        return _discard_clue(state, "本条线索已尝试三次停靠，仍未完成接近。")
    destination, planning = plan_object_standoff(frame, target.target_world_xy, context.tried_positions)
    if destination is None:
        return _discard_clue(state, "目标附近没有剩余可达停靠点。", **planning)
    position = (destination.x_m, destination.y_m)
    return _result(
        replace(state, phase=SearchPhase.APPROACHING_OBJECT, object_approach=replace(
            context, destination=destination, tried_positions=context.tried_positions + (position,),
        )),
        "object.approach", "一次前往目标附近的可达停靠点并对准目标，命令执行成功后完成搜索。",
        command=_command_to_pose(frame.pose, destination), destination_world_xy=position,
        target_world_xy=target.target_world_xy, target_source=target.source,
        sample_count=target.sample_count, **planning,
    )


def _complete_approach(state: SearchState) -> NavigationResult:
    """以停靠命令成功结束为完成依据，不再请求新图或重新估计目标距离。"""
    context = state.object_approach
    return _result(
        replace(state, phase=SearchPhase.COMPLETE, active_target_clue=None),
        "object.complete", "停靠命令已执行完成，结束物体搜索。",
        clue_id=state.active_target_clue.clue_id,
        target_world_xy=context.target.target_world_xy,
        destination_world_xy=(context.destination.x_m, context.destination.y_m),
        completion_basis="standoff_command_completed",
    )


def _discard_clue(state: SearchState, reason: str, **details) -> NavigationResult:
    clue = state.active_target_clue
    return _result(
        replace(
            state, phase=SearchPhase.SCANNING, active_target_clue=None,
            object_approach=ObjectApproachState(
                fallback_clue=state.object_approach.fallback_clue,
                history_localized=state.object_approach.history_localized,
            ), scan_headings_world_rad=(), next_scan_index=0,
            scan_evidence=(), scan_observation_points=(), scan_local_point_count=0,
            pending_target_world_xy=None, target_approach_attempts=0, backtrack_node_id=None,
        ),
        "object.clue_failed", "本条线索未完成，保持当前位置，继续下一条历史线索。",
        failed_clue_id=clue.clue_id if clue is not None else None, reason=reason, **details,
    )


def _command_to_pose(start: Pose2D, destination: Pose2D) -> RelativePoseCommand:
    forward, left = world_point_to_robot((destination.x_m, destination.y_m), start)
    return RelativePoseCommand(forward, left, _turn(start.yaw_rad, destination.yaw_rad))


def _turn(start: float, end: float) -> float:
    return (end - start + math.pi) % (2.0 * math.pi) - math.pi


def _result(state, stage, message, command=None, status=NavigationStatus.OK, **details):
    clue = state.active_target_clue
    return NavigationResult(
        status=status, state=state, command=command,
        debug=NavigationDebug(stage, message, {
            "clue_id": clue.clue_id if clue is not None else None,
            "approach_attempts": len(state.object_approach.tried_positions), **details,
        }),
    )
