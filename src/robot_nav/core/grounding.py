"""用目标框、对齐深度和相机标定估计目标二维位置。"""

from __future__ import annotations

import math
import statistics
from typing import Optional, Sequence, Tuple

from .models import (
    CameraExtrinsics,
    CameraIntrinsics,
    DepthImage,
    MaskImage,
    Pose2D,
    TargetEstimate,
)


def ground_target_bbox(
    bbox_norm: Sequence[float],
    depth_m: DepthImage,
    intrinsics: CameraIntrinsics,
    robot_pose_world: Pose2D,
    camera_extrinsics_in_robot: CameraExtrinsics,
    min_depth_m: float = 0.10,
    max_depth_m: float = 5.0,
    min_valid_points: int = 8,
    no_valid_depth_fallback_m: Optional[float] = None,
    target_mask: Optional[MaskImage] = None,
) -> TargetEstimate:
    """返回目标在机器人与世界坐标系中的位置；输入不可靠时显式失败。

    有 ``target_mask`` 时只使用掩码像素；否则优先使用目标框中央区域，深度
    不足时退回完整目标框。``no_valid_depth_fallback_m`` 只在目标区域完全
    没有有效深度时使用。
    """
    bbox = _normalize_bbox(bbox_norm)
    if bbox is None:
        return TargetEstimate(False, "target_bbox_invalid")
    depth = _normalize_depth(depth_m)
    if depth is None:
        return TargetEstimate(False, "target_depth_missing")
    if not _valid_intrinsics(intrinsics) or not _valid_pose(robot_pose_world):
        return TargetEstimate(False, "target_calibration_invalid")
    if not _valid_extrinsics(camera_extrinsics_in_robot):
        return TargetEstimate(False, "camera_extrinsics_invalid")
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
    if no_valid_depth_fallback_m is not None and (
        not _is_finite(no_valid_depth_fallback_m)
        or not (
            float(min_depth_m)
            <= float(no_valid_depth_fallback_m)
            <= float(max_depth_m)
        )
    ):
        return TargetEstimate(False, "target_depth_fallback_invalid")

    height, width = len(depth), len(depth[0])
    if not (
        0.0 <= float(intrinsics.cx) < width
        and 0.0 <= float(intrinsics.cy) < height
    ):
        return TargetEstimate(False, "target_calibration_invalid")
    fallback_pixel = _bbox_center_pixel(bbox, height, width)
    used_target_mask = target_mask is not None
    if target_mask is not None:
        if not _mask_matches_image(target_mask, height, width):
            return TargetEstimate(False, "target_mask_invalid")
        points_base, fallback_pixel, mask_pixel_count = _depth_points_in_mask(
            depth,
            target_mask,
            intrinsics,
            camera_extrinsics_in_robot,
            float(min_depth_m),
            float(max_depth_m),
        )
        if mask_pixel_count == 0:
            return TargetEstimate(False, "target_mask_empty")
    else:
        rows, columns = _bbox_pixels(bbox, height, width, inset_ratio=0.15)
        points_base = _depth_points_in_robot(
            depth,
            rows,
            columns,
            intrinsics,
            camera_extrinsics_in_robot,
            float(min_depth_m),
            float(max_depth_m),
        )
        if len(points_base) < min_valid_points:
            rows, columns = _bbox_pixels(bbox, height, width, inset_ratio=0.0)
            points_base = _depth_points_in_robot(
                depth,
                rows,
                columns,
                intrinsics,
                camera_extrinsics_in_robot,
                float(min_depth_m),
                float(max_depth_m),
            )

    used_depth_fallback = False
    if not points_base and no_valid_depth_fallback_m is not None:
        center_row, center_col = fallback_pixel
        fallback_point = _camera_pixel_to_robot(
            center_row,
            center_col,
            float(no_valid_depth_fallback_m),
            intrinsics,
            camera_extrinsics_in_robot,
        )
        if fallback_point is not None:
            points_base = [fallback_point]
            used_depth_fallback = True

    if not used_depth_fallback and len(points_base) < min_valid_points:
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
        reason=(
            "target_grounded_with_depth_fallback"
            if used_depth_fallback
            else (
                "target_grounded_with_mask"
                if used_target_mask
                else "target_grounded"
            )
        ),
        target_base_xy=(target_forward, target_left),
        target_world_xy=target_world,
        distance_m=distance,
        bearing_rad=bearing,
        sample_count=0 if used_depth_fallback else len(points_base),
    )


def _depth_points_in_robot(
    depth: Sequence[Sequence[object]],
    rows: range,
    columns: range,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    min_depth_m: float,
    max_depth_m: float,
) -> list[Tuple[float, float]]:
    """把指定像素区域内的有效深度转换为机器人平面点。"""
    points_base = []
    for row in rows:
        for col in columns:
            raw_depth = depth[row][col]
            if raw_depth is None or not _is_finite(raw_depth):
                continue
            forward_camera = float(raw_depth)
            if not min_depth_m <= forward_camera <= max_depth_m:
                continue
            point_base = _camera_pixel_to_robot(
                row,
                col,
                forward_camera,
                intrinsics,
                extrinsics,
            )
            if point_base is not None:
                points_base.append(point_base)
    return points_base


def _depth_points_in_mask(
    depth: Sequence[Sequence[object]],
    mask: MaskImage,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    min_depth_m: float,
    max_depth_m: float,
) -> Tuple[list[Tuple[float, float]], Tuple[int, int], int]:
    """把掩码内有效深度转为机器人平面点，并返回掩码质心像素。"""
    points_base = []
    foreground_count = 0
    row_sum = 0
    col_sum = 0
    for row_index, mask_row in enumerate(mask):
        for col_index, selected in enumerate(mask_row):
            if not bool(selected):
                continue
            foreground_count += 1
            row_sum += row_index
            col_sum += col_index

            raw_depth = depth[row_index][col_index]
            if raw_depth is None or not _is_finite(raw_depth):
                continue
            forward_camera = float(raw_depth)
            if not min_depth_m <= forward_camera <= max_depth_m:
                continue
            point_base = _camera_pixel_to_robot(
                row_index,
                col_index,
                forward_camera,
                intrinsics,
                extrinsics,
            )
            if point_base is not None:
                points_base.append(point_base)

    if foreground_count == 0:
        return points_base, (0, 0), 0
    center_pixel = (
        int(round(row_sum / foreground_count)),
        int(round(col_sum / foreground_count)),
    )
    return points_base, center_pixel, foreground_count


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


def _mask_matches_image(mask: MaskImage, height: int, width: int) -> bool:
    """仅校验掩码为与深度图同尺寸的二维矩形。"""
    try:
        return len(mask) == height and all(len(row) == width for row in mask)
    except TypeError:
        return False


def _bbox_pixels(
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
    inset_ratio: float,
) -> Tuple[range, range]:
    """返回目标框按给定比例向内收缩后的像素行列范围。"""
    x1, y1, x2, y2 = bbox
    inset_x = (x2 - x1) * inset_ratio
    inset_y = (y2 - y1) * inset_ratio
    first_col = max(0, min(width - 1, int(math.floor((x1 + inset_x) * width))))
    last_col = max(first_col + 1, min(width, int(math.ceil((x2 - inset_x) * width))))
    first_row = max(0, min(height - 1, int(math.floor((y1 + inset_y) * height))))
    last_row = max(first_row + 1, min(height, int(math.ceil((y2 - inset_y) * height))))
    return range(first_row, last_row), range(first_col, last_col)


def _bbox_center_pixel(
    bbox: Tuple[float, float, float, float],
    height: int,
    width: int,
) -> Tuple[int, int]:
    """返回目标框中心对应的有效图像像素。"""
    center_col = int((bbox[0] + bbox[2]) * 0.5 * width)
    center_row = int((bbox[1] + bbox[3]) * 0.5 * height)
    return (
        max(0, min(height - 1, center_row)),
        max(0, min(width - 1, center_col)),
    )


def _camera_pixel_to_robot(
    row: int,
    col: int,
    forward_camera: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Optional[Tuple[float, float]]:
    """把一个带深度的相机像素投影到机器人平面。"""
    left_camera = -(
        (float(col) - float(intrinsics.cx))
        * forward_camera
        / float(intrinsics.fx)
    )
    up_camera = -(
        (float(row) - float(intrinsics.cy))
        * forward_camera
        / float(intrinsics.fy)
    )
    forward_base, left_base = _camera_point_to_robot(
        forward_camera,
        left_camera,
        up_camera,
        extrinsics,
    )
    if forward_base <= 0.05:
        return None
    return forward_base, left_base


def _camera_point_to_robot(
    forward_camera: float,
    left_camera: float,
    up_camera: float,
    extrinsics: CameraExtrinsics,
) -> Tuple[float, float]:
    """按 roll、pitch、yaw 顺序把相机光学点转换到机器人平面。"""
    roll_cosine = math.cos(float(extrinsics.roll_rad))
    roll_sine = math.sin(float(extrinsics.roll_rad))
    rolled_left = left_camera * roll_cosine + up_camera * roll_sine
    rolled_up = -left_camera * roll_sine + up_camera * roll_cosine

    pitch_cosine = math.cos(float(extrinsics.pitch_down_rad))
    pitch_sine = math.sin(float(extrinsics.pitch_down_rad))
    pitched_forward = (
        forward_camera * pitch_cosine + rolled_up * pitch_sine
    )

    yaw_cosine = math.cos(float(extrinsics.yaw_rad))
    yaw_sine = math.sin(float(extrinsics.yaw_rad))
    return (
        float(extrinsics.forward_m)
        + yaw_cosine * pitched_forward
        - yaw_sine * rolled_left,
        float(extrinsics.left_m)
        + yaw_sine * pitched_forward
        + yaw_cosine * rolled_left,
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


def _valid_extrinsics(extrinsics: CameraExtrinsics) -> bool:
    return isinstance(extrinsics, CameraExtrinsics) and all(
        _is_finite(value)
        for value in (
            extrinsics.forward_m,
            extrinsics.left_m,
            extrinsics.height_m,
            extrinsics.yaw_rad,
            extrinsics.pitch_down_rad,
            extrinsics.roll_rad,
        )
    )


def _is_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = ["ground_target_bbox"]
