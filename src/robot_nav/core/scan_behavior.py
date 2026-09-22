"""扫描行为：按局部可见且尚未观察的 Frontier 规划朝向并采集画面。

同一行为的正常推进与可恢复失败放在一起：规划视角、对齐朝向、登记已检查
覆盖，以及转向未完成时按实际朝向重规划。本模块只读写 ``SearchState``。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Optional, Tuple

from .frontier import FrameFrontierCache
from .frontier_regions import refresh_frontier_regions, scan_coverage_details
from .models import (
    ActionKind,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    ObservationView,
    Pose2D,
    RelativePoseCommand,
    ScanEvidence,
    SearchPhase,
    SearchState,
    TargetSearchGoal,
    TargetVisibility,
)
from .navigation_io import make_action, result
from .observation_coverage import (
    camera_world_position,
    capture_observation_view,
    frontier_observation_points,
    unobserved_observation_points,
)
from .scan import (
    build_unobserved_scan_headings,
    shortest_turn_to_heading,
)
from .timing import TimingSpans

TURN_TOLERANCE_RAD = math.radians(5.0)


def continue_scanning(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """每轮都按未观察的局部 Frontier 扫描；没有待查方向时采集当前画面。"""
    working_state = state
    scan_debug_details: Optional[Mapping[str, Any]] = None
    if not working_state.scan_headings_world_rad:
        try:
            working_state, frontiers = refresh_frontier_regions(frame, working_state,
                timings=timings, frontier_cache=frontier_cache,
            )
            local_points = frontier_observation_points(frame, frontiers.boundary_cells)
            # 待分析视角也暂时避免重复采集；只有分析成功才会登记为已检查覆盖。
            points = unobserved_observation_points(
                local_points, frame, working_state.observed_views + working_state.pending_observation_views,
            )
            if points:
                camera_offset, horizontal_fov = horizontal_camera_view(frame)
                camera_xy = camera_world_position(frame)
                headings = build_unobserved_scan_headings(
                    points, Pose2D(camera_xy[0], camera_xy[1], frame.pose.yaw_rad),
                    camera_offset, horizontal_fov,
                )
            else:
                # 覆盖可复用也保留当前画面的目标检查，不为此额外转向。
                headings = (frame.pose.yaw_rad,)
        except ValueError as exc:
            return result(
                NavigationStatus.MISSING_DATA, working_state, "scan.plan",
                f"无法规划待检查视角：{exc}",
            )
        scan_debug_details = {
            "frontier_move_candidate_count": len(frontiers.candidates),
            "frontier_scan_cell_count": len(frontiers.boundary_cells),
            "local_observation_point_count": len(local_points),
            "observation_point_count": len(points),
            "reused_observation_point_count": len(local_points) - len(points),
            "checked_view_count": len(working_state.observed_views),
        }
        working_state = start_scan_with_headings(
            replace(
                working_state,
                scan_observation_points=points,
                scan_local_point_count=len(local_points),
            ),
            headings,
        )
    return advance_scan(
        frame, goal, working_state, scan_debug_details,
        timings=timings, frontier_cache=frontier_cache,
    )


def advance_scan(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    scan_debug_details: Optional[Mapping[str, Any]] = None,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """对齐当前扫描方向并原地转向。视觉结果由运行层从异步队列取得后传入。"""
    target_heading = state.scan_headings_world_rad[state.next_scan_index]
    relative_turn = shortest_turn_to_heading(frame.pose.yaw_rad, target_heading)
    if abs(relative_turn) > TURN_TOLERANCE_RAD:
        return result(
            NavigationStatus.OK,
            state,
            "scan.turn",
            "转向下一个扫描方向。",
            make_action(
                ActionKind.TURN_IN_PLACE,
                command=RelativePoseCommand(yaw_rad=relative_turn),
                purpose=ActionPurpose.SCAN_TURN,
            ),
            scan_debug_details_of(state, target_heading, extra=scan_debug_details),
        )
    return result(
        NavigationStatus.NEEDS_SCAN_CAPTURE, state, "scan.capture",
        "扫描方向已对齐，等待固定画面采集。",
        details=scan_debug_details_of(state, target_heading, extra=scan_debug_details),
    )


def record_scanned_direction(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    target_heading: float,
    scan_debug_details: Optional[Mapping[str, Any]] = None,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """登记已完成的扫描方向与覆盖；一轮扫描结束后进入 Frontier 探索。"""
    try:
        camera_offset, horizontal_fov = horizontal_camera_view(frame)
    except ValueError as exc:
        return result(
            NavigationStatus.MISSING_DATA, state, "scan.observe",
            f"无法记录当前已检查视角：{exc}",
        )
    checked_view = capture_observation_view(
        frame, camera_offset, horizontal_fov,
        observation_points=state.scan_observation_points,
    )
    evidence = ScanEvidence(
        heading_world_rad=target_heading,
        visibility=TargetVisibility.PENDING,
        view=checked_view,
    )
    next_index = state.next_scan_index + 1
    scanned_state = replace(
        state,
        scan_evidence=state.scan_evidence + (evidence,),
        next_scan_index=min(next_index, len(state.scan_headings_world_rad) - 1),

    )
    if next_index < len(state.scan_headings_world_rad):
        return continue_scanning(
            frame,
            goal,
            replace(scanned_state, next_scan_index=next_index),
            timings=timings, frontier_cache=frontier_cache,
        )
    return finish_scan(frame, goal, scanned_state, timings=timings, frontier_cache=frontier_cache)


def finish_scan(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: SearchState,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """结束本轮采集并进入探索；场景模式此前已由后台观察器判定。"""
    completed_state = replace(state, phase=SearchPhase.EXPLORING)
    from .exploration import select_exploration_target

    return select_exploration_target(
        frame,
        completed_state,
        frontier_scores=None,
        timings=timings, frontier_cache=frontier_cache,
    )


def recover_scan_turn(state: SearchState, reason: str) -> NavigationResult:
    """转向已停止后按下一帧真实朝向重建剩余观察，不假装该方向已检查。"""
    return result(
        NavigationStatus.OK, reset_scan_after_move(state), "motion.scan_recovered",
        "扫描转向未完成，按实际朝向和已有观察记录重新规划剩余视角。",
        details={"reason": str(reason)},
    )


def start_scan_with_headings(state: SearchState, headings: Tuple[float, ...]) -> SearchState:
    """清空上一轮证据并开始执行给定的世界系扫描朝向。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=headings,
        next_scan_index=0,
        scan_evidence=(),
    )


def reset_scan_after_move(state: SearchState) -> SearchState:
    """移动命令完成后，延迟到下一帧再按新的真实朝向建立扫描。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=(),
        next_scan_index=0,
        scan_evidence=(),
        scan_observation_points=(),
        scan_local_point_count=0,
        backtrack_node_id=None,
        active_target_clue=None,
    )


def horizontal_camera_view(frame: NavigationFrame) -> Tuple[float, float]:
    """返回相机水平视场中心相对底盘的偏角，以及完整水平 FOV。"""
    intrinsics = frame.camera_intrinsics
    if intrinsics is None:
        raise ValueError("缺少相机内参")
    if not math.isfinite(intrinsics.fx) or float(intrinsics.fx) <= 0.0:
        raise ValueError("相机 fx 必须为正有限值")
    if not math.isfinite(intrinsics.cx):
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
    if not math.isfinite(camera_yaw):
        raise ValueError("相机 yaw 外参必须为有限值")
    return (
        float(camera_yaw) + intrinsic_center_offset,
        left_extent + right_extent,
    )


def capture_semantic_view(
    frame: NavigationFrame, observation_points: Tuple[Tuple[float, float], ...] = (),
) -> ObservationView:
    """记录固定帧的真实视角与覆盖；调用方须在检测成功后才登记为已检查。"""
    offset, fov = horizontal_camera_view(frame)
    return capture_observation_view(frame, offset, fov, observation_points=observation_points)


def scan_debug_details_of(
    state: SearchState,
    target_heading: float,
    scan_index: Optional[int] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """返回一条扫描决策需要记录的最小上下文。"""
    details = {
        **scan_coverage_details(state),
        "scan_index": state.next_scan_index if scan_index is None else scan_index,
        "scan_heading_count": len(state.scan_headings_world_rad),
        "scan_mode": "frontier" if state.scan_observation_points else "current_view",
        "target_heading_world_rad": target_heading,
        "checked_view_count": len(state.observed_views),
    }
    if extra is not None:
        details.update(extra)
    return details


def _camera_image_width(frame: NavigationFrame) -> int:
    """返回与相机内参对应的图像宽度；优先使用 RGB，随后使用对齐深度。"""
    image = frame.rgb if frame.rgb is not None else frame.depth
    if image is None:
        raise ValueError("缺少 RGB 或深度图像尺寸")
    width = len(image[0]) if image else 0
    if width < 1:
        raise ValueError("相机图像宽度必须大于零")
    return width


__all__ = [
    "capture_semantic_view",
    "continue_scanning",
    "recover_scan_turn",
    "reset_scan_after_move",
]
