"""场景目标的历史线索：返回后台检测到目标时的拍摄位置与朝向。

同一行为的正常推进与可恢复失败放在一起：返回到位、对齐朝向即完成，
以及线索坐标系不符或返回未完成时放弃该线索、继续下一条。
"""

from __future__ import annotations

import math
from dataclasses import replace

from .models import (
    ActionConstraint,
    ActionKind,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    RelativePoseCommand,
    SearchPhase,
    SearchState,
)
from .navigation_io import make_action, result
from .scan import shortest_turn_to_heading

BACKTRACK_ARRIVAL_M = 0.25
TURN_TOLERANCE_RAD = math.radians(5.0)


def continue_scene_target(frame: NavigationFrame, state: SearchState) -> NavigationResult:
    """场景线索返回拍摄位置并对齐朝向，到位即完成。"""
    clue = state.active_target_clue
    if clue.map_frame_id != frame.obstacle_map.frame_id:
        return discard_target_clue(state, "目标线索与当前地图坐标系不同。")
    distance = math.hypot(frame.pose.x_m - clue.pose.x_m, frame.pose.y_m - clue.pose.y_m)
    if state.phase is SearchPhase.REVISITING_TARGET and distance > BACKTRACK_ARRIVAL_M:
        return discard_target_clue(state, f"返回线索位置的动作结束后仍相距 {distance:.3f} m。")
    if distance > BACKTRACK_ARRIVAL_M:
        return result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.REVISITING_TARGET, backtrack_node_id=None),
            "target.revisit",
            "后台检测到目标，返回当时的拍摄位置与朝向。",
            make_action(
                ActionKind.MOVE_TO_POSE,
                destination=clue.pose,
                constraint=ActionConstraint.REQUIRE_KNOWN_PATH,
                purpose=ActionPurpose.REVISIT,
            ),
            {
                "clue_id": clue.clue_id,
                "capture_timestamp_s": clue.timestamp_s,
                "destination_world_xy": (clue.pose.x_m, clue.pose.y_m),
            },
        )
    turn = shortest_turn_to_heading(frame.pose.yaw_rad, clue.pose.yaw_rad)
    if abs(turn) > TURN_TOLERANCE_RAD:
        return result(
            NavigationStatus.OK,
            replace(state, phase=SearchPhase.REVISITING_TARGET, backtrack_node_id=None),
            "target.revisit_turn",
            "已回到拍摄位置，对齐检测画面当时的朝向。",
            make_action(
                ActionKind.TURN_IN_PLACE,
                command=RelativePoseCommand(yaw_rad=turn),
                purpose=ActionPurpose.REVISIT_TURN,
            ),
            {"clue_id": clue.clue_id},
        )
    from .scan_behavior import reset_scan_after_move

    return result(
        NavigationStatus.OK,
        replace(reset_scan_after_move(state), phase=SearchPhase.COMPLETE),
        "target.revisit_complete",
        "已返回目标场景画面的拍摄位置与朝向，搜索完成。",
        details={
            "clue_id": clue.clue_id,
            "capture_timestamp_s": clue.timestamp_s,
            "destination_world_xy": (clue.pose.x_m, clue.pose.y_m),
            "destination_yaw_rad": clue.pose.yaw_rad,
            "distance_to_capture_m": distance,
            "heading_error_rad": turn,
        },
    )


def discard_target_clue(state: SearchState, reason: str) -> NavigationResult:
    """结束本条线索且不发命令，让下一周期优先取下一条；列表耗尽后恢复探索。"""
    clue = state.active_target_clue
    from .scan_behavior import reset_scan_after_move

    return result(
        NavigationStatus.OK,
        reset_scan_after_move(state),
        "target.revisit_failed",
        "未能返回本条线索的拍摄位姿，保留实际位置并尝试下一条；线索耗尽后继续探索。",
        details={
            "clue_id": clue.clue_id if clue is not None else None,
            "reason": reason,
        },
    )


__all__ = ["continue_scene_target", "discard_target_clue"]
