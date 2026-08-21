"""用目标框、对齐深度和相机标定估计目标二维位置。"""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence, Tuple

from .models import CameraIntrinsics, DepthImage, Pose2D, TargetEstimate


def ground_target_bbox(
    bbox_norm: Sequence[float],
    depth_m: DepthImage,
    intrinsics: CameraIntrinsics,
    robot_pose_world: Pose2D,
    camera_pose_in_robot: Pose2D,
    min_depth_m: float = 0.10,
    max_depth_m: float = 5.0,
    min_valid_points: int = 8,
) -> TargetEstimate:
    """返回目标在机器人与世界坐标系中的位置；输入不可靠时显式失败。"""
    bbox = _normalize_bbox(bbox_norm)
    if bbox is None:
        return TargetEstimate(False, "target_bbox_invalid")
    depth = _normalize_depth(depth_m)
    if depth is None:
        return TargetEstimate(False, "target_depth_missing")
    if not _valid_intrinsics(intrinsics) or not _valid_pose(robot_pose_world):
        return TargetEstimate(False, "target_calibration_invalid")
    if not _valid_pose(camera_pose_in_robot):
        return TargetEstimate(False, "camera_pose_invalid")
    if (
        not _is_finite(min_depth_m)
        or not _is_finite(max_depth_m)
        or float(min_depth_m) <= 0.0
        or float(max_depth_m) <= float(min_depth_m)
    ):
        return TargetEstimate(False, "target_depth_range_invalid")
    if (
        isinstance(min_valid_points, bool)
        or not isinstance(min_valid_points, int)
        or min_valid_points < 1
    ):
        return TargetEstimate(False, "target_min_points_invalid")

    height, width = len(depth), len(depth[0])
    if not (
        0.0 <= float(intrinsics.cx) < width
        and 0.0 <= float(intrinsics.cy) < height
    ):
        return TargetEstimate(False, "target_calibration_invalid")
    rows, columns = _central_bbox_pixels(bbox, height, width)
    points_base = []
    for row in rows:
        for col in columns:
            raw_depth = depth[row][col]
            if raw_depth is None or not _is_finite(raw_depth):
                continue
            forward_camera = float(raw_depth)
            if not float(min_depth_m) <= forward_camera <= float(max_depth_m):
                continue
            left_camera = -(
                (float(col) - float(intrinsics.cx))
                * forward_camera
                / float(intrinsics.fx)
            )
            forward_base, left_base = _camera_point_to_robot(
                forward_camera, left_camera, camera_pose_in_robot
            )
            if forward_base > 0.05:
                points_base.append((forward_base, left_base))

    if len(points_base) < min_valid_points:
        return TargetEstimate(
            False,
            "target_depth_points_insufficient",
            sample_count=len(points_base),
        )

    points_base = _keep_near_points(points_base)
    target_forward = float(statistics.median(point[0] for point in points_base))
    target_left = float(statistics.median(point[1] for point in points_base))
    distance = math.hypot(target_forward, target_left)
    if not math.isfinite(distance) or distance <= 0.05:
        return TargetEstimate(
            False, "target_distance_invalid", sample_count=len(points_base)
        )

    bearing = math.atan2(target_left, target_forward)
    cosine = math.cos(float(robot_pose_world.yaw_rad))
    sine = math.sin(float(robot_pose_world.yaw_rad))
    target_world = (
        float(robot_pose_world.x_m)
        + cosine * target_forward
        - sine * target_left,
        float(robot_pose_world.y_m) + sine * target_forward + cosine * target_left,
    )
    return TargetEstimate(
        success=True,
        reason="target_grounded",
        target_base_xy=(target_forward, target_left),
        target_world_xy=target_world,
        distance_m=distance,
        bearing_rad=bearing,
        sample_count=len(points_base),
    )


def _normalize_bbox(
    bbox_norm: Sequence[float],
) -> Optional[Tuple[float, float, float, float]]:
    """校验归一化 xyxy 目标框。"""
    try:
        values = tuple(float(value) for value in bbox_norm)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = values
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        return None
    return x1, y1, x2, y2


def _normalize_depth(depth_m: DepthImage):
    """把任意二维序列冻结为矩形行序列；非法输入返回 None。"""
    try:
        rows = tuple(tuple(row) for row in depth_m)
    except TypeError:
        return None
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        return None
    return rows


def _central_bbox_pixels(
    bbox: Tuple[float, float, float, float], height: int, width: int
) -> Tuple[range, range]:
    """返回目标框中央 70% 区域的像素行列范围。"""
    x1, y1, x2, y2 = bbox
    inset_x = (x2 - x1) * 0.15
    inset_y = (y2 - y1) * 0.15
    first_col = max(0, min(width - 1, int(math.floor((x1 + inset_x) * width))))
    last_col = max(first_col + 1, min(width, int(math.ceil((x2 - inset_x) * width))))
    first_row = max(0, min(height - 1, int(math.floor((y1 + inset_y) * height))))
    last_row = max(first_row + 1, min(height, int(math.ceil((y2 - inset_y) * height))))
    return range(first_row, last_row), range(first_col, last_col)


def _camera_point_to_robot(
    forward_camera: float,
    left_camera: float,
    camera_pose: Pose2D,
) -> Tuple[float, float]:
    """应用相机在机器人二维坐标中的 forward/left/yaw 外参。"""
    cosine = math.cos(float(camera_pose.yaw_rad))
    sine = math.sin(float(camera_pose.yaw_rad))
    return (
        float(camera_pose.x_m) + cosine * forward_camera - sine * left_camera,
        float(camera_pose.y_m) + sine * forward_camera + cosine * left_camera,
    )


def _keep_near_points(
    points: Sequence[Tuple[float, float]],
) -> Sequence[Tuple[float, float]]:
    """保留距离不超过第 60 百分位的点，降低框内背景影响。"""
    distances = sorted(math.hypot(point[0], point[1]) for point in points)
    threshold = distances[int((len(distances) - 1) * 0.60)]
    near_points = [
        point for point in points if math.hypot(point[0], point[1]) <= threshold
    ]
    return near_points or points


def _valid_intrinsics(intrinsics: CameraIntrinsics) -> bool:
    return isinstance(intrinsics, CameraIntrinsics) and all(
        _is_finite(value)
        for value in (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    ) and float(intrinsics.fx) > 0.0 and float(intrinsics.fy) > 0.0


def _valid_pose(pose: Pose2D) -> bool:
    return isinstance(pose, Pose2D) and all(
        _is_finite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    )


def _is_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["ground_target_bbox"]
