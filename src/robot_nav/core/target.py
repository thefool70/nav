"""目标处理：线索分派、场景返回、物体接近与失败恢复。

continue_target_search 是入口；定位由运行层提供，停靠选点在本文件末尾。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .exploration import reachable_free_distances
from .geometry import (
    shortest_turn_to_heading,
    grid_cell_center_to_world,
    world_to_nearest_grid_cell,
)
from .models import (
    SearchMode,
    ActionConstraint,
    ActionKind,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    RelativePoseCommand,
    SearchPhase,
    SearchState,
    TargetSearchGoal,
    ObjectApproachState,
    ObjectLocalization,
    Pose2D,
    NavigationAction,
    result,
)
from .scan import reset_scan_after_move


BACKTRACK_ARRIVAL_M = 0.25
TURN_TOLERANCE_RAD = math.radians(5.0)
CAPTURE_ARRIVAL_M = 0.25
MAX_APPROACH_MOVES = 3
PREFERRED_STANDOFF_M = 0.75
MIN_STANDOFF_M = 0.60
MAX_STANDOFF_M = 2.0
REUSE_CURRENT_DISTANCE_M = 0.90
FAILED_STANDOFF_EXCLUSION_M = 0.20


def continue_target_search(
    frame: NavigationFrame, goal: TargetSearchGoal, state: SearchState,
    localization: Optional[ObjectLocalization] = None,
) -> Optional[NavigationResult]:
    """目标处理总入口：领取线索后的定位／返回、停靠结果，以及线索耗尽后的去向。

    返回 None 表示没有待处理目标，导航入口可以继续扫描或探索。
    """
    if state.active_target_clue is None:
        return continue_object_history(frame, state) if goal.search_mode is SearchMode.OBJECT else None
    if goal.search_mode is SearchMode.SCENE:
        return continue_scene_target(frame, state)
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
        context = state.object_approach
        return object_result(
            replace(state, phase=SearchPhase.COMPLETE, active_target_clue=None),
            "object.complete", "停靠命令已执行完成，结束物体搜索。",
            clue_id=clue.clue_id, target_world_xy=context.target.target_world_xy,
            destination_world_xy=(context.destination.x_m, context.destination.y_m),
            completion_basis="standoff_command_completed",
        )
    if localization is None:
        return object_result(
            replace(state, phase=SearchPhase.LOCALIZING_OBJECT), "object.localize",
            "用当前历史线索的 RGB-D 定位物体。",
            status=NavigationStatus.NEEDS_OBJECT_LOCALIZATION, localization_source="snapshot",
        )
    if localization.target_world_xy is None:
        return discard_object_clue(state, localization.reason or "本张历史画面无法定位物体。")
    return plan_approach(
        frame,
        replace(state, object_approach=replace(
            state.object_approach, target=localization, destination=None,
            history_localized=True,
        )),
    )


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
            NavigationAction(
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
            NavigationAction(
                ActionKind.TURN_IN_PLACE,
                command=RelativePoseCommand(yaw_rad=turn),
                purpose=ActionPurpose.REVISIT_TURN,
            ),
            {"clue_id": clue.clue_id},
        )

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
        action=NavigationAction(
            ActionKind.MOVE_TO_POSE,
            destination=destination,
            constraint=ActionConstraint.REQUIRE_KNOWN_PATH,
            purpose=ActionPurpose.APPROACH,
        ),
        destination_world_xy=position,
        target_world_xy=target.target_world_xy, target_source=target.source,
        sample_count=target.sample_count, **planning,
    )


def discard_object_clue(state: SearchState, reason: str, **details) -> NavigationResult:
    """本条线索无法继续：清空接近状态并回到扫描，继续下一条历史线索。"""
    clue = state.active_target_clue

    cleared = reset_scan_after_move(replace(
        state, object_approach=ObjectApproachState(
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
    purpose: ActionPurpose, state: SearchState, reason: str,
) -> Optional[NavigationResult]:
    """接近失败清空目的地、下一帧换点；保底返回失败直接停止。"""
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


def plan_object_standoff(frame, target_xy, tried_positions):
    """搜索目标周围完整圆域，返回停靠位姿和可直接写入日志的选点诊断。"""
    grid = frame.navigation_map if frame.navigation_map is not None else frame.obstacle_map
    clearance = frame.navigation_clearance_m if frame.navigation_map is not None else 0.0
    reachable = reachable_free_distances(grid, frame.pose, clearance_m=clearance)
    details = {
        "standoff_map": "full_navigation" if frame.navigation_map is not None else "exploration",
        "standoff_clearance_m": clearance,
        "standoff_search_radius_m": MAX_STANDOFF_M,
        "standoff_reachable_cells": len(reachable),
        "standoff_unknown_cells": 0,
        "standoff_occupied_cells": 0,
        "standoff_unreachable_cells": 0,
        "standoff_excluded_cells": 0,
        "standoff_candidate_count": 0,
    }
    robot_xy = (frame.pose.x_m, frame.pose.y_m)
    current_distance = math.dist(robot_xy, target_xy)
    # 已在合适距离且位置可用时只需对准目标，避免不必要的平移。
    current_cell = world_to_nearest_grid_cell(robot_xy, grid)
    if (
        MIN_STANDOFF_M <= current_distance <= REUSE_CURRENT_DISTANCE_M
        and current_cell in reachable
        and not _near_tried_position(robot_xy, tried_positions)
    ):
        details.update(standoff_candidate_count=1, standoff_distance_m=current_distance,
                       standoff_path_distance_m=0.0)
        return _facing_target(robot_xy, target_xy, frame.camera_extrinsics_in_robot.yaw_rad), details

    # 理想点放在目标朝向机器人这一侧，但候选仍搜索整个圆域。
    heading = math.atan2(robot_xy[1] - target_xy[1], robot_xy[0] - target_xy[0])
    preferred = (target_xy[0] + PREFERRED_STANDOFF_M * math.cos(heading),
                 target_xy[1] + PREFERRED_STANDOFF_M * math.sin(heading))
    center_row, center_col = world_to_nearest_grid_cell(target_xy, grid)
    steps = int(math.ceil(MAX_STANDOFF_M / grid.resolution_m)) + 1
    height = len(grid.occupancy)
    width = len(grid.occupancy[0]) if height else 0
    best = None
    for row in range(max(0, center_row - steps), min(height, center_row + steps + 1)):
        for col in range(max(0, center_col - steps), min(width, center_col + steps + 1)):
            xy = grid_cell_center_to_world(row, col, grid)
            distance = math.dist(xy, target_xy)
            if not MIN_STANDOFF_M <= distance <= MAX_STANDOFF_M:
                continue
            value = grid.occupancy[row][col]
            if value is None:
                details["standoff_unknown_cells"] += 1
            elif value > 0.5:
                details["standoff_occupied_cells"] += 1
            elif (row, col) not in reachable:
                # 包括被净空膨胀排除，以及与机器人所在自由区不连通的格子。
                details["standoff_unreachable_cells"] += 1
            elif _near_tried_position(xy, tried_positions):
                details["standoff_excluded_cells"] += 1
            else:
                details["standoff_candidate_count"] += 1
                path_distance = reachable[(row, col)] * grid.resolution_m
                # 先贴近理想停靠点，再比较路径长度；行列号让同分结果确定。
                rank = (math.dist(xy, preferred), path_distance, row, col)
                if best is None or rank < best[0]:
                    best = (rank, xy, distance, path_distance)
    if best is None:
        return None, details
    _, position, distance, path_distance = best
    details.update(standoff_distance_m=distance, standoff_path_distance_m=path_distance)
    return _facing_target(position, target_xy, frame.camera_extrinsics_in_robot.yaw_rad), details


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
            action=NavigationAction(
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
            action=NavigationAction(
                ActionKind.TURN_IN_PLACE,
                command=RelativePoseCommand(yaw_rad=turn),
                purpose=ActionPurpose.FALLBACK_TURN,
            ),
        )
    return stop_at_fallback(state, "所有历史线索均无法定位；已返回拍摄位姿，停止导航。")


def _near_tried_position(position, tried_positions):
    """排除已失败停靠点附近的位置，避免换到相邻格反复尝试。"""
    return any(math.dist(position, old) < FAILED_STANDOFF_EXCLUSION_M for old in tried_positions)


def _facing_target(position, target_xy, camera_yaw):
    """计算底盘最终朝向，并扣除相机安装 yaw，使相机朝向目标。"""
    yaw = math.atan2(target_xy[1] - position[1], target_xy[0] - position[0]) - camera_yaw
    return Pose2D(position[0], position[1], (yaw + math.pi) % (2.0 * math.pi) - math.pi)
