"""历史物体定位：复用已通过融合判定的目标框与 YOLOE 分割，再由 RGB-D 或障碍射线求位置。

localize_object 展示完整流程；下方是同一定位任务的深度与坐标计算。
不选择底盘停靠点、不更新搜索状态；这些决策在 core/target.py。"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import asdict
from itertools import product
from typing import Iterable, Tuple

from ..core.geometry import world_point_to_robot
from ..core.models import (
    NavigationFrame,
    TargetEstimate,
    CameraExtrinsics,
    CameraIntrinsics,
    DepthImage,
    MaskImage,
    Pose2D,
    ObjectLocalization,
    TargetConfirmation,
    TargetObservation,
    TargetVisibility,
)


def localize_object(frame, *, bbox_norm, source, mask, context, on_event) -> ObjectLocalization:
    """复用同帧 YOLOE 掩码测距；无对应掩码或无有效深度时依次尝试框和障碍射线。"""
    started = time.monotonic()
    on_event({"event": "object_localization_started", **context, "pose": asdict(frame.pose),
              "map_frame_id": frame.obstacle_map.frame_id, "timestamp_s": frame.timestamp_s})
    confirmation = (TargetConfirmation.CONFIRMED if source in ("vlm", "vlm+yoloe")
                    else TargetConfirmation.UNCERTAIN)
    reasons = []
    method = "rgbd"
    estimate = localize_segmented_object(frame, bbox_norm, mask)
    target_source = source + ("_yoloe_mask" if mask is not None else "_bbox")
    if not estimate.success and mask is not None:
        reasons.append(estimate.reason)
        estimate = localize_segmented_object(frame, bbox_norm)
        target_source = source + "_bbox"
    if not estimate.success:
        reasons.append(estimate.reason)
        estimate = localize_obstacle_on_image_ray(frame, bbox_norm)
        target_source, method = "bbox_obstacle", "obstacle_assumption"
    reasons.append(estimate.reason)
    result = ObjectLocalization(
        target_world_xy=estimate.target_world_xy if estimate.success else None,
        visibility=TargetVisibility.VISIBLE if estimate.success else TargetVisibility.UNCERTAIN,
        vlm_confirmation=confirmation, source=target_source,
        sample_count=estimate.sample_count, reason="; ".join(reasons),
    )
    record = asdict(result)
    record["visibility"] = result.visibility.value
    record["vlm_confirmation"] = result.vlm_confirmation.value
    record["target_source"] = record.pop("source")
    on_event({"event": "object_localized", **context, **record, "bbox_norm": bbox_norm,
              "detector_source": source, "localization_method": method,
              "mask_used": estimate.success and target_source.endswith("_yoloe_mask"),
              "distance_m": estimate.distance_m, "bearing_rad": estimate.bearing_rad,
              "duration_s": time.monotonic() - started,
              # 仅供 Rerun 画同帧掩码；JSONL 边界排除图像对象。
              "observation_frame": frame,
              "observation": TargetObservation(TargetVisibility.VISIBLE, bbox_norm=bbox_norm,
                                                source=source, target_mask=mask)})
    return result


def localize_obstacle_on_image_ray(frame: NavigationFrame, bbox_norm=None) -> TargetEstimate:
    """用拍摄位姿投出射线，以首个已知障碍表面作为物体位置假设；不设假定距离。"""
    grid = frame.navigation_map if frame.navigation_map is not None else frame.obstacle_map
    ray = _image_ray_world(frame, bbox_norm)
    if ray is None:
        return TargetEstimate(False, "object_ray_calibration_invalid")
    origin, heading = ray
    hit = _first_obstacle_on_ray(grid, origin, heading)
    if hit is None:
        return TargetEstimate(False, "object_ray_has_no_obstacle")
    base_xy = world_point_to_robot(hit, frame.pose)
    return TargetEstimate(
        success=True,
        reason="target_assumed_from_bbox_obstacle" if bbox_norm is not None else "target_assumed_from_front_obstacle",
        target_base_xy=base_xy, target_world_xy=hit,
        distance_m=math.hypot(*base_xy), bearing_rad=math.atan2(base_xy[1], base_xy[0]),
        sample_count=0,
    )


def localize_segmented_object(frame: NavigationFrame, bbox_norm, mask=None) -> TargetEstimate:
    """返回目标在机器人与世界坐标系中的位置；输入不可靠时显式失败。

    有掩码时只使用掩码像素；否则优先使用目标框中央区域，深度
    缺失时退回完整目标框。保留所有有限正深度点，不做背景筛选或假定距离回退。
    """
    if frame.depth is None or frame.camera_intrinsics is None:
        return TargetEstimate(False, "object_depth_or_intrinsics_missing")
    if not _valid_bbox(bbox_norm):
        return TargetEstimate(False, "target_bbox_invalid")
    depth = _normalize_depth(frame.depth)
    if depth is None:
        return TargetEstimate(False, "target_depth_missing")
    intrinsics = frame.camera_intrinsics
    extrinsics = frame.camera_extrinsics_in_robot
    if not _valid_intrinsics(intrinsics) or not _valid_pose(frame.pose):
        return TargetEstimate(False, "target_calibration_invalid")
    if not _valid_extrinsics(extrinsics):
        return TargetEstimate(False, "camera_extrinsics_invalid")
    height, width = len(depth), len(depth[0])
    if not (
        0.0 <= float(intrinsics.cx) < width
        and 0.0 <= float(intrinsics.cy) < height
    ):
        return TargetEstimate(False, "target_calibration_invalid")
    if mask is not None:
        if not _mask_matches_image(mask, height, width):
            return TargetEstimate(False, "target_mask_invalid")
        pixels = (
            (row, col) for row, mask_row in enumerate(mask)
            for col, selected in enumerate(mask_row) if selected
        )
        points_base, mask_pixel_count = _depth_points(depth, pixels, intrinsics, extrinsics)
        if mask_pixel_count == 0:
            return TargetEstimate(False, "target_mask_empty")
    else:
        points_base = _bbox_depth_points(bbox_norm, depth, intrinsics, extrinsics)

    if not points_base:
        return TargetEstimate(False, "target_depth_points_insufficient", sample_count=0)
    return _estimate_world_target(points_base, frame.pose, used_target_mask=mask is not None)


def _image_ray_world(frame, bbox_norm):
    """框中心像素生成相机射线；无框使用光轴，roll/pitch/yaw 与 RGB-D 定位一致。"""
    left, up = 0.0, 0.0
    if bbox_norm is not None:
        k = frame.camera_intrinsics
        if frame.rgb is None or len(frame.rgb) == 0 or len(frame.rgb[0]) == 0 or k is None:
            return None
        if not _valid_intrinsics(k) or not _valid_bbox(bbox_norm):
            return None
        x0, y0, x1, y1 = bbox_norm
        u = (x0 + x1) * 0.5 * len(frame.rgb[0])
        v = (y0 + y1) * 0.5 * len(frame.rgb)
        left, up = (k.cx - u) / k.fx, (k.cy - v) / k.fy
    e, pose = frame.camera_extrinsics_in_robot, frame.pose
    rolled_left = left * math.cos(e.roll_rad) + up * math.sin(e.roll_rad)
    rolled_up = -left * math.sin(e.roll_rad) + up * math.cos(e.roll_rad)
    forward = math.cos(e.pitch_down_rad) + rolled_up * math.sin(e.pitch_down_rad)
    if math.hypot(forward, rolled_left) < 1e-9:
        return None
    heading = pose.yaw_rad + e.yaw_rad + math.atan2(rolled_left, forward)
    c, s = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    origin = (pose.x_m + c * e.forward_m - s * e.left_m,
              pose.y_m + s * e.forward_m + c * e.left_m)
    return origin, heading


def _first_obstacle_on_ray(grid, origin_world, heading_world):
    """按穿过格子边界的顺序遍历射线，返回首个占用格的进入点；未知格不提供距离。"""
    height = len(grid.occupancy)
    width = len(grid.occupancy[0]) if height else 0
    if not height or not width:
        return None
    x, y = world_point_to_robot(origin_world, grid.origin)
    yaw = heading_world - grid.origin.yaw_rad
    dx, dy = math.cos(yaw), math.sin(yaw)
    resolution = grid.resolution_m
    start, end = 0.0, math.inf
    # 地图原点是格中心，数组外边界位于半格之外。
    for value, direction, count in ((x, dx, width), (y, dy, height)):
        low, high = -0.5 * resolution, (count - 0.5) * resolution
        if abs(direction) < 1e-12:
            if not low <= value < high:
                return None
            continue
        first, last = sorted(((low - value) / direction, (high - value) / direction))
        start, end = max(start, first), min(end, last)
    if start >= end:
        return None
    # 取边界内侧确定首格；命中位置仍使用精确的进入距离。
    inside = start + min(resolution * 1e-7, (end - start) * 0.5)
    col = math.floor((x + inside * dx) / resolution + 0.5)
    row = math.floor((y + inside * dy) / resolution + 0.5)
    step_col, step_row = (1 if dx >= 0 else -1), (1 if dy >= 0 else -1)
    next_x = ((col + 0.5 * step_col) * resolution - x) / dx if abs(dx) >= 1e-12 else math.inf
    next_y = ((row + 0.5 * step_row) * resolution - y) / dy if abs(dy) >= 1e-12 else math.inf
    delta_x = resolution / abs(dx) if abs(dx) >= 1e-12 else math.inf
    delta_y = resolution / abs(dy) if abs(dy) >= 1e-12 else math.inf
    distance = start
    while 0 <= row < height and 0 <= col < width and distance < end:
        value = grid.occupancy[row][col]
        if value is not None and value > 0.5:
            return (origin_world[0] + distance * math.cos(heading_world),
                    origin_world[1] + distance * math.sin(heading_world))
        distance = min(next_x, next_y)
        cross_x, cross_y = next_x <= next_y, next_y <= next_x
        if cross_x:
            col += step_col
            next_x += delta_x
        if cross_y:
            row += step_row
            next_y += delta_y
    return None


def _bbox_depth_points(bbox, depth, intrinsics, camera_extrinsics_in_robot):
    """先取目标框内缩区域，无有效点时改用完整框；输出机器人平面上的米制点。"""
    height, width = len(depth), len(depth[0])
    for inset_ratio in (0.15, 0.0):
        rows, columns = _bbox_pixels(bbox, height, width, inset_ratio)
        points_base, _ = _depth_points(
            depth, product(rows, columns), intrinsics, camera_extrinsics_in_robot,
        )
        if points_base:
            return points_base
    return []


def _estimate_world_target(points_base, robot_pose_world, used_target_mask):
    """从机器人系点取中位数，转换成世界位置并保留深度来源诊断。"""
    target_forward = float(statistics.median(point[0] for point in points_base))
    target_left = float(statistics.median(point[1] for point in points_base))
    distance = math.hypot(target_forward, target_left)
    if not math.isfinite(distance):
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
        reason="target_grounded_with_mask" if used_target_mask else "target_grounded",
        target_base_xy=(target_forward, target_left),
        target_world_xy=target_world,
        distance_m=distance,
        bearing_rad=bearing,
        sample_count=len(points_base),
    )


def _depth_points(
    depth: DepthImage,
    pixels: Iterable[Tuple[int, int]],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Tuple[list[Tuple[float, float]], int]:
    """把选中像素的有限正深度投影到机器人平面；像素数用于区分空掩码与深度缺测。"""
    points_base = []
    pixel_count = 0
    for row, col in pixels:
        pixel_count += 1
        raw_depth = depth[row][col]
        if raw_depth is None or not math.isfinite(raw_depth):
            continue
        forward_camera = float(raw_depth)
        if forward_camera <= 0.0:
            continue
        points_base.append(_camera_pixel_to_robot(row, col, forward_camera, intrinsics, extrinsics))
    return points_base, pixel_count


def _normalize_depth(depth_m: DepthImage):
    """把任意二维序列冻结为矩形行序列；非法输入返回 None。"""
    rows = tuple(tuple(row) for row in depth_m)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        return None
    return rows


def _mask_matches_image(mask: MaskImage, height: int, width: int) -> bool:
    """仅校验掩码为与深度图同尺寸的二维矩形。"""
    return len(mask) == height and all(len(row) == width for row in mask)


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


def _camera_pixel_to_robot(
    row: int,
    col: int,
    forward_camera: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Tuple[float, float]:
    """把一个带深度的相机像素投影到机器人平面。"""
    # 像素向右、向下增长；内部使用前、左、上，因此横纵坐标都取负号。
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


def _valid_intrinsics(intrinsics: CameraIntrinsics) -> bool:
    return all(
        math.isfinite(value)
        for value in (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)
    ) and float(intrinsics.fx) > 0.0 and float(intrinsics.fy) > 0.0


def _valid_pose(pose: Pose2D) -> bool:
    return all(
        math.isfinite(value) for value in (pose.x_m, pose.y_m, pose.yaw_rad)
    )


def _valid_extrinsics(extrinsics: CameraExtrinsics) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            extrinsics.forward_m,
            extrinsics.left_m,
            extrinsics.height_m,
            extrinsics.yaw_rad,
            extrinsics.pitch_down_rad,
            extrinsics.roll_rad,
        )
    )


def _valid_bbox(bbox):
    """模型输出、深度定位和障碍射线共用的 0～1 xyxy 框校验。"""
    if bbox is None or len(bbox) != 4:
        return False
    return (all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in bbox)
            and bbox[0] < bbox[2] and bbox[1] < bbox[3])
