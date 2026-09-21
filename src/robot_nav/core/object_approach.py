"""物体目标处理：消费历史线索定位结果，规划停靠点并接近。

同一行为的正常推进与可恢复失败放在一起：请求定位、规划停靠、停靠命令完成
即结束，以及停靠未完成时换点、线索失败后继续下一条、全部线索无法定位时
保底返回拍摄位姿。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .models import (
    ActionConstraint,
    ActionKind,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObjectApproachState,
    ObjectLocalization,
    RelativePoseCommand,
    SearchPhase,
    SearchState,
)
from .navigation_io import make_action, result
from .object_standoff import plan_object_standoff
from .scan import shortest_turn_to_heading

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
        return discard_object_clue(state, "物体线索与当前地图坐标系不同。")
    if state.phase is SearchPhase.REVISITING_TARGET:
        return _return_to_capture(frame, state)
    if state.phase is SearchPhase.APPROACHING_OBJECT:
        if state.object_approach.destination is None:
            return plan_approach(frame, state)
        # 上一周期的停靠命令已同步执行完成；运动失败会清空 destination。
        return complete_approach(state)
    if localization is None:
        return request_localization(state)
    if localization.target_world_xy is None:
        return discard_object_clue(state, localization.reason or "本张历史画面无法定位物体。")
    return plan_approach(
        frame,
        replace(state, object_approach=replace(
            state.object_approach, target=localization, destination=None,
            history_localized=True,
        )),
    )


def continue_object_history(frame: NavigationFrame, state: SearchState) -> Optional[NavigationResult]:
    """当前线索失败后先处理已采集的队列；全部历史都无法定位时才保底返回。"""
    context = state.object_approach
    if context.fallback_clue is None:
        return None
    if state.pending_semantic_jobs:
        return object_result(
            replace(state, phase=SearchPhase.WAITING_FOR_SEMANTICS), "object.wait_history",
            "保持当前位置，继续处理已采集的历史画面与线索。",
            pending_semantic_jobs=state.pending_semantic_jobs,
        )
    # 曾定位成功但运动未完成时继续探索；从未定位成功才走返回拍摄点的保底分支。
    if context.history_localized:
        return object_result(
            replace(state, phase=SearchPhase.SCANNING, object_approach=ObjectApproachState()),
            "object.history_finished", "历史目标未能完成接近，恢复探索。",
        )
    if context.fallback_clue.map_frame_id != frame.obstacle_map.frame_id:
        return stop_at_fallback(state, "保底拍摄位姿与当前地图不同，停在当前位置。")
    return _return_to_capture(frame, replace(state, phase=SearchPhase.SCANNING,
                                            active_target_clue=context.fallback_clue))


def request_localization(state: SearchState) -> NavigationResult:
    """请求用当前历史线索的 RGB-D 定位物体。"""
    return object_result(
        replace(state, phase=SearchPhase.LOCALIZING_OBJECT), "object.localize",
        "用当前历史线索的 RGB-D 定位物体。",
        status=NavigationStatus.NEEDS_OBJECT_LOCALIZATION, localization_source="snapshot",
    )


def plan_approach(frame: NavigationFrame, state: SearchState) -> NavigationResult:
    """在目标附近选一个可达停靠点，并请求一次完成的停靠移动。"""
    context = state.object_approach
    target = context.target
    if target is None or target.target_world_xy is None:
        return discard_object_clue(state, "接近阶段缺少目标位置。")
    if len(context.tried_positions) >= MAX_APPROACH_MOVES:
        return discard_object_clue(state, "本条线索已尝试三次停靠，仍未完成接近。")
    destination, planning = plan_object_standoff(frame, target.target_world_xy, context.tried_positions)
    if destination is None:
        return discard_object_clue(state, "目标附近没有剩余可达停靠点。", **planning)
    position = (destination.x_m, destination.y_m)
    return object_result(
        replace(state, phase=SearchPhase.APPROACHING_OBJECT, object_approach=replace(
            context, destination=destination, tried_positions=context.tried_positions + (position,),
        )),
        "object.approach", "一次前往目标附近的可达停靠点并对准目标，命令执行成功后完成搜索。",
        action=make_action(
            ActionKind.MOVE_TO_POSE,
            destination=destination,
            constraint=ActionConstraint.REQUIRE_KNOWN_PATH,
            purpose=ActionPurpose.APPROACH,
        ),
        destination_world_xy=position,
        target_world_xy=target.target_world_xy, target_source=target.source,
        sample_count=target.sample_count, **planning,
    )


def complete_approach(state: SearchState) -> NavigationResult:
    """以停靠命令成功结束为完成依据，不再请求新图或重新估计目标距离。"""
    context = state.object_approach
    return object_result(
        replace(state, phase=SearchPhase.COMPLETE, active_target_clue=None),
        "object.complete", "停靠命令已执行完成，结束物体搜索。",
        clue_id=state.active_target_clue.clue_id,
        target_world_xy=context.target.target_world_xy,
        destination_world_xy=(context.destination.x_m, context.destination.y_m),
        completion_basis="standoff_command_completed",
    )


def discard_object_clue(state: SearchState, reason: str, **details) -> NavigationResult:
    """本条线索无法继续：清空接近状态并回到扫描，继续下一条历史线索。"""
    clue = state.active_target_clue
    from .scan_behavior import reset_scan_after_move

    cleared = reset_scan_after_move(replace(
        state, active_target_clue=None,
        object_approach=ObjectApproachState(
            fallback_clue=state.object_approach.fallback_clue,
            history_localized=state.object_approach.history_localized,
        ),
    ))
    return object_result(
        cleared,
        "object.clue_failed", "本条线索未完成，保持当前位置，继续下一条历史线索。",
        failed_clue_id=clue.clue_id if clue is not None else None, reason=reason, **details,
    )


def recover_object_motion(
    action: ActionKind, purpose: ActionPurpose, state: SearchState, reason: str,
) -> Optional[NavigationResult]:
    """接近失败换停靠点；保底返回失败直接停止。

    ``action`` 与 ``purpose`` 来自失败动作本身，不再解析日志字符串。
    """
    if purpose in (ActionPurpose.FALLBACK, ActionPurpose.FALLBACK_TURN):
        return stop_at_fallback(state, f"保底返回失败，停在当前位置：{reason}")
    if purpose != ActionPurpose.APPROACH:
        return None
    cleared = replace(state, object_approach=replace(
        state.object_approach, destination=None,
    ))
    return object_result(
        cleared, "object.motion_recovered",
        "本次停靠未完成，下一周期按最新地图换点。", reason=reason,
    )


def stop_at_fallback(state: SearchState, reason: str) -> NavigationResult:
    """保底流程终止：停在当前位置。"""
    return object_result(
        replace(state, phase=SearchPhase.STOPPED, active_target_clue=None),
        "object.fallback_stopped", reason,
    )


def object_result(state, stage, message, action=None, status=NavigationStatus.OK, **details):
    """构造带线索上下文的物体目标结果。"""
    clue = state.active_target_clue
    return result(
        status, state, stage, message, action,
        {
            "clue_id": clue.clue_id if clue is not None else None,
            "approach_attempts": len(state.object_approach.tried_positions),
            **details,
        },
    )


def _return_to_capture(frame: NavigationFrame, state: SearchState) -> NavigationResult:
    """保底返回拍摄位姿：先到位，再对齐朝向，成功即停止。"""
    clue = state.active_target_clue
    context = state.object_approach
    distance = math.hypot(frame.pose.x_m - clue.pose.x_m, frame.pose.y_m - clue.pose.y_m)
    if state.phase is SearchPhase.REVISITING_TARGET and distance > CAPTURE_ARRIVAL_M:
        return stop_at_fallback(state, f"保底返回结束后距拍摄点仍有 {distance:.2f} m，停在当前位置。")
    if distance > CAPTURE_ARRIVAL_M:
        return object_result(
            replace(state, phase=SearchPhase.REVISITING_TARGET, backtrack_node_id=None),
            "object.fallback_return", "所有历史线索均无法定位，保底返回拍摄位姿后停止。",
            action=make_action(
                ActionKind.MOVE_TO_POSE,
                destination=clue.pose,
                constraint=ActionConstraint.REQUIRE_KNOWN_PATH,
                purpose=ActionPurpose.FALLBACK,
            ),
            destination_world_xy=(clue.pose.x_m, clue.pose.y_m),
        )
    turn = shortest_turn_to_heading(frame.pose.yaw_rad, clue.pose.yaw_rad)
    if abs(turn) > TURN_TOLERANCE_RAD:
        if context.capture_turns >= 2:
            return stop_at_fallback(state, "保底返回两次转向仍未恢复拍摄朝向，停止。")
        return object_result(
            replace(state, phase=SearchPhase.REVISITING_TARGET, object_approach=replace(
                context, capture_turns=context.capture_turns + 1,
            )),
            "object.fallback_turn", "已回到拍摄点，对齐朝向后停止。",
            action=make_action(
                ActionKind.TURN_IN_PLACE,
                command=RelativePoseCommand(yaw_rad=turn),
                purpose=ActionPurpose.FALLBACK_TURN,
            ),
        )
    return stop_at_fallback(state, "所有历史线索均无法定位；已返回拍摄位姿，停止导航。")


__all__ = [
    "complete_approach",
    "continue_object_history",
    "discard_object_clue",
    "navigate_object_approach",
    "plan_approach",
    "recover_object_motion",
    "request_localization",
    "stop_at_fallback",
]
