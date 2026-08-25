"""把导航决策和底盘反馈保存为便于复盘的 JSONL 运行日志。"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, TextIO, Tuple

from .core.geometry import world_to_nearest_grid_cell
from .core.models import (
    NavigationFrame,
    NavigationResult,
    ObservationNode,
    Pose2D,
    RelativePoseCommand,
    SearchState,
    TargetObservation,
)


DEFAULT_RUN_LOG_DIRECTORY = Path("data/run_logs")


def default_run_log_path(adapter_name: str) -> Path:
    """返回带本地时间和微秒的默认日志路径，避免不同运行互相覆盖。"""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    safe_adapter_name = adapter_name.replace("/", "-").replace(" ", "-")
    return DEFAULT_RUN_LOG_DIRECTORY / f"{safe_adapter_name}-{timestamp}.jsonl"


class NavigationRunLogger:
    """逐行写入导航事件；每条写入后立即 flush，异常退出也尽量保留现场。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: Optional[TextIO] = self.path.open(
            "a", encoding="utf-8"
        )
        self._current_cycle: Optional[int] = None

    def log_run_start(
        self,
        adapter_name: str,
        target_text: str,
        max_cycles: int,
        configuration: Mapping[str, Any],
    ) -> None:
        self._write(
            "run_start",
            adapter=adapter_name,
            target=target_text,
            max_cycles=max_cycles,
            configuration=configuration,
        )

    def log_cycle_decision(
        self,
        cycle_index: int,
        frame: NavigationFrame,
        observation: Optional[TargetObservation],
        result: NavigationResult,
    ) -> None:
        """保存发送命令前的输入、算法决定和关键中间结果。"""
        self._current_cycle = cycle_index
        self._write(
            "cycle_decision",
            cycle=cycle_index,
            frame_timestamp_s=frame.timestamp_s,
            pose=_pose_summary(frame.pose),
            obstacle_map=_map_summary(frame),
            camera=_camera_summary(frame),
            observation=_observation_summary(observation),
            result=_result_summary(result, frame.pose),
        )

    def log_cycle_result(
        self,
        cycle_index: int,
        result: NavigationResult,
    ) -> None:
        """保存命令执行或可恢复失败处理后的最终周期状态。"""
        self._write(
            "cycle_result",
            cycle=cycle_index,
            status=result.status.value,
            phase=result.state.phase.value,
            stage=result.debug.stage,
            message=result.debug.message,
            details=_compact_debug_details(result.debug.details),
            state=_state_summary(result.state),
        )
        self._current_cycle = None

    def log_action_progress(self, message: str) -> None:
        self._write(
            "hermes_action",
            cycle=self._current_cycle,
            message=str(message),
        )

    def log_error(self, exc: BaseException) -> None:
        self._write(
            "run_error",
            cycle=self._current_cycle,
            error_type=type(exc).__name__,
            message=str(exc),
        )

    def log_run_end(
        self,
        return_code: int,
        error_message: Optional[str] = None,
    ) -> None:
        self._write(
            "run_end",
            return_code=return_code,
            error=error_message,
        )

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.close()

    def _write(self, event: str, **payload: Any) -> None:
        stream = self._stream
        if stream is None:
            raise RuntimeError("运行日志已经关闭")
        record = {
            "event": event,
            "wall_time": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "monotonic_s": time.monotonic(),
            **payload,
        }
        stream.write(
            json.dumps(
                _jsonable(record),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        stream.flush()


def _result_summary(
    result: NavigationResult,
    pose: Pose2D,
) -> Mapping[str, Any]:
    return {
        "status": result.status.value,
        "phase": result.state.phase.value,
        "stage": result.debug.stage,
        "message": result.debug.message,
        "details": _compact_debug_details(result.debug.details, pose),
        "command": _command_summary(result.command, pose),
        "state": _state_summary(result.state),
    }


def _command_summary(
    command: Optional[RelativePoseCommand],
    pose: Pose2D,
) -> Optional[Mapping[str, Any]]:
    if command is None:
        return None
    cosine = math.cos(pose.yaw_rad)
    sine = math.sin(pose.yaw_rad)
    target_world_xy = (
        pose.x_m + command.forward_m * cosine - command.left_m * sine,
        pose.y_m + command.forward_m * sine + command.left_m * cosine,
    )
    return {
        "forward_m": command.forward_m,
        "left_m": command.left_m,
        "yaw_rad": command.yaw_rad,
        "translation_m": math.hypot(command.forward_m, command.left_m),
        "target_world_xy": target_world_xy,
    }


def _map_summary(frame: NavigationFrame) -> Mapping[str, Any]:
    obstacle_map = frame.obstacle_map
    rows = obstacle_map.occupancy
    height = len(rows)
    width = len(rows[0]) if height else 0
    free_count = 0
    occupied_count = 0
    unknown_count = 0
    for row in rows:
        for value in row:
            if value is None:
                unknown_count += 1
            elif float(value) <= 0.5:
                free_count += 1
            else:
                occupied_count += 1

    robot_cell = world_to_nearest_grid_cell(
        (frame.pose.x_m, frame.pose.y_m), obstacle_map
    )
    robot_cell_value: Optional[float] = None
    robot_cell_state = "out_of_map"
    robot_cell_in_map = (
        0 <= robot_cell[0] < height
        and 0 <= robot_cell[1] < width
    )
    if robot_cell_in_map:
        value = rows[robot_cell[0]][robot_cell[1]]
        robot_cell_value = None if value is None else float(value)
        if value is None:
            robot_cell_state = "unknown"
        elif float(value) <= 0.5:
            robot_cell_state = "free"
        else:
            robot_cell_state = "occupied"

    return {
        "frame_id": obstacle_map.frame_id,
        "width": width,
        "height": height,
        "resolution_m": obstacle_map.resolution_m,
        "origin": _pose_summary(obstacle_map.origin),
        "free_cells": free_count,
        "occupied_cells": occupied_count,
        "unknown_cells": unknown_count,
        "robot_cell": robot_cell,
        "robot_cell_in_map": robot_cell_in_map,
        "robot_cell_value": robot_cell_value,
        "robot_cell_state": robot_cell_state,
    }


def _camera_summary(frame: NavigationFrame) -> Mapping[str, Any]:
    intrinsics = frame.camera_intrinsics
    extrinsics = frame.camera_extrinsics_in_robot
    return {
        "rgb_size": _image_size(frame.rgb),
        "depth_size": _image_size(frame.depth),
        "intrinsics": None
        if intrinsics is None
        else {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        },
        "extrinsics_in_robot": {
            "forward_m": extrinsics.forward_m,
            "left_m": extrinsics.left_m,
            "height_m": extrinsics.height_m,
            "yaw_rad": extrinsics.yaw_rad,
            "pitch_down_rad": extrinsics.pitch_down_rad,
            "roll_rad": extrinsics.roll_rad,
        },
    }


def _image_size(image: Any) -> Optional[Tuple[int, int]]:
    if image is None:
        return None
    height = len(image)
    width = len(image[0]) if height else 0
    return (width, height)


def _observation_summary(
    observation: Optional[TargetObservation],
) -> Optional[Mapping[str, Any]]:
    if observation is None:
        return None
    return {
        "visibility": observation.visibility.value,
        "bbox_norm": observation.bbox_norm,
        "target_mask": _mask_summary(observation.target_mask),
        "reason": observation.reason,
    }


def _mask_summary(mask: Any) -> Optional[Mapping[str, Any]]:
    """只记录掩码尺寸和前景数，避免把整张图写入 JSONL。"""
    if mask is None:
        return None
    try:
        height = len(mask)
        width = len(mask[0]) if height else 0
        foreground_pixels = sum(
            1 for row in mask for selected in row if bool(selected)
        )
    except (TypeError, IndexError, ValueError):
        return {"valid": False}
    return {
        "valid": height > 0 and width > 0,
        "width_px": width,
        "height_px": height,
        "foreground_pixels": foreground_pixels,
    }


def _state_summary(state: SearchState) -> Mapping[str, Any]:
    direction_counts = {
        "pending": 0,
        "committed": 0,
        "explored": 0,
        "invalidated": 0,
    }
    for node in state.observation_history:
        for direction in node.directions:
            direction_counts[direction.state.value] += 1

    active_node = next(
        (
            node
            for node in state.observation_history
            if node.node_id == state.active_node_id
        ),
        None,
    )
    latest_node = (
        state.observation_history[-1]
        if state.observation_history
        else None
    )
    return {
        "phase": state.phase.value,
        "scan_headings_world_rad": state.scan_headings_world_rad,
        "next_scan_index": state.next_scan_index,
        "scan_evidence": tuple(
            {
                "heading_world_rad": evidence.heading_world_rad,
                "visibility": evidence.visibility.value,
            }
            for evidence in state.scan_evidence
        ),
        "initial_scan_complete": state.initial_scan_complete,
        "target_approach_attempts": state.target_approach_attempts,
        "history_node_count": len(state.observation_history),
        "history_direction_counts": direction_counts,
        "active_node_id": state.active_node_id,
        "active_node": _node_summary(active_node),
        "latest_node": _node_summary(latest_node),
    }


def _node_summary(
    node: Optional[ObservationNode],
) -> Optional[Mapping[str, Any]]:
    if node is None:
        return None
    return {
        "node_id": node.node_id,
        "position_world_xy": node.position_world_xy,
        "directions": tuple(
            {
                "direction_id": direction.direction_id,
                "heading_world_rad": direction.heading_world_rad,
                "candidate_world_xy": direction.candidate_world_xy,
                "state": direction.state.value,
            }
            for direction in node.directions
        ),
    }


def _compact_debug_details(
    details: Mapping[str, Any],
    pose: Optional[Pose2D] = None,
) -> Mapping[str, Any]:
    """保留候选评分，但省略可由地图复现的大型 Frontier 格列表。"""
    compact = dict(details)
    candidates = compact.get("frontier_candidates")
    if candidates is not None:
        candidate_summaries = []
        for candidate in candidates:
            summary = {
                key: value
                for key, value in candidate.items()
                if key != "frontier_cells"
            }
            if pose is not None:
                summary["robot_distance_m"] = math.hypot(
                    float(candidate["world_x_m"]) - pose.x_m,
                    float(candidate["world_y_m"]) - pose.y_m,
                )
            candidate_summaries.append(summary)
        compact["frontier_candidates"] = tuple(candidate_summaries)
    return compact


def _pose_summary(pose: Pose2D) -> Mapping[str, float]:
    return {
        "x_m": pose.x_m,
        "y_m": pose.y_m,
        "yaw_rad": pose.yaw_rad,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    return str(value)


__all__ = [
    "NavigationRunLogger",
    "default_run_log_path",
]
