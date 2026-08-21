"""导航算法主入口。当前仅完成输入校验与扫描状态初始化，视觉观察模块尚未
迁移，故一律不发送命令、不推进扫描下标、不假装完成。"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .models import (
    NavigationDebug,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObstacleMap,
    Pose2D,
    SearchPhase,
    SearchState,
    TargetSearchGoal,
)
from .scan import build_uniform_scan_headings, shortest_turn_to_heading


def _is_finite(value) -> bool:
    """value 是否为可转换的有限数（不含 bool）。"""
    if isinstance(value, bool):
        return False
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(converted)


def _positive_finite(value) -> float | None:
    """value 为正有限可转换数时返回 float(value)，否则 None（不含 bool）。

    结果已规范化，可用于数值比较，避免把可转换字符串直接参与比较而泄漏 TypeError。
    """
    if not _is_finite(value):
        return None
    converted = float(value)
    if converted <= 0.0:
        return None
    return converted


def _validation_error(
    frame: NavigationFrame, goal: TargetSearchGoal, state: Optional[SearchState]
) -> Optional[str]:
    """返回非法输入的简短原因，合法输入返回 None。"""
    if not isinstance(goal, TargetSearchGoal) or (
        not isinstance(goal.target_text, str) or not goal.target_text.strip()
    ):
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
    if _positive_finite(frame.obstacle_map.resolution_m) is None:
        return "frame.obstacle_map.resolution_m 必须为正有限值"
    if (
        not isinstance(frame.obstacle_map.frame_id, str)
        or not frame.obstacle_map.frame_id
    ):
        return "frame.obstacle_map.frame_id 必须为非空字符串"
    if state is not None and not isinstance(state, SearchState):
        return "state 必须为 SearchState 或 None"
    if state is not None and state.scan_headings_world_rad:
        if not isinstance(state.scan_headings_world_rad, (tuple, list)) or not all(
            _is_finite(heading) for heading in state.scan_headings_world_rad
        ):
            return "state.scan_headings_world_rad 必须为有限角度序列"
        next_index = state.next_scan_index
        if (
            isinstance(next_index, bool)
            or not isinstance(next_index, int)
            or not 0 <= next_index < len(state.scan_headings_world_rad)
        ):
            return "state.next_scan_index 必须为扫描航向范围内的整数"
    return None


def _invalid_result(state: Optional[SearchState], reason: str) -> NavigationResult:
    """返回 INVALID_INPUT 结果，state 置为 FAILED 阶段。"""
    failed_state = (
        replace(state, phase=SearchPhase.FAILED)
        if isinstance(state, SearchState)
        else SearchState(phase=SearchPhase.FAILED)
    )
    return NavigationResult(
        status=NavigationStatus.INVALID_INPUT,
        command=None,
        debug=NavigationDebug(stage="input", message=reason),
        state=failed_state,
    )


def navigate(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState | None = None,
) -> NavigationResult:
    """单周期语义目标搜索主入口。

    校验输入后，若缺少扫描航向则以当前 yaw 初始化四向扫描；由于视觉观察
    尚未迁移，合法输入当前返回 NOT_IMPLEMENTED，非法输入返回 INVALID_INPUT，
    两种情况均不发送命令。下一周期状态由调用方从 result.state 读取。
    """
    reason = _validation_error(frame, goal, state)
    if reason is not None:
        return _invalid_result(state, reason)

    if state is not None and state.scan_headings_world_rad:
        working_state = state
    else:
        working_state = SearchState(
            phase=SearchPhase.SCANNING,
            scan_headings_world_rad=build_uniform_scan_headings(
                frame.pose.yaw_rad, view_count=4
            ),
            next_scan_index=0,
            observation_history=(
                state.observation_history if state is not None else ()
            ),
        )

    target_heading_world_rad = working_state.scan_headings_world_rad[
        working_state.next_scan_index
    ]
    relative_turn_rad = shortest_turn_to_heading(
        frame.pose.yaw_rad, target_heading_world_rad
    )
    return NavigationResult(
        status=NavigationStatus.NOT_IMPLEMENTED,
        command=None,
        debug=NavigationDebug(
            stage="scan",
            message="视觉观察尚未迁移，等待视觉模块接入后再执行扫描。",
            details={
                "next_scan_index": working_state.next_scan_index,
                "target_heading_world_rad": target_heading_world_rad,
                "relative_turn_rad": relative_turn_rad,
            },
        ),
        state=working_state,
    )
