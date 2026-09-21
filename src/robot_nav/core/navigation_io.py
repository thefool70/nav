"""搜索核心与运行层之间共享的结果构造与输入校验。

本模块只提供两类内容：
1. 公共输入检查（:func:`validation_error` 与字段级辅助函数）。
2. 单周期结果构造（:func:`result`、:func:`invalid_result`）。

它不决定任何搜索行为：行为分派在 ``navigator``，具体行为在四个行为模块中。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional

from .models import (
    ActionKind,
    ActionPurpose,
    ActionConstraint,
    FrontierScoreRequest,
    NavigationAction,
    NavigationDebug,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObjectLocalization,
    Pose2D,
    RelativePoseCommand,
    SearchPhase,
    SearchState,
    TargetSearchGoal,
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


__all__ = [
    "invalid_result",
    "make_action",
    "result",
    "validation_error",
]
