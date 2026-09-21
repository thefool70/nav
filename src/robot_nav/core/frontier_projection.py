"""将局部地面 Frontier 投影到对齐 RGB-D，生成可冻结的图像锚点。

世界点的高度采用机器人脚下局部地面 z=0；相机 height_m 是离该平面的高度。
深度表示光轴距离而非欧氏距离。本模块不读取设备、不绘图，也不修改探索状态。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Optional, Sequence, Tuple

from .geometry import world_point_to_robot
from .models import FrontierCandidate, NavigationFrame


GROUND_DEPTH_TOLERANCE_M = 0.15
GROUND_MIN_DEPTH_SAMPLES = 3


@dataclass(frozen=True)
class FrontierImageProjection:
    """原始 RGB 像素锚点及其深度依据；世界坐标、光轴深度单位均为米。"""

    world_xy: Tuple[float, float]
    pixel_xy: Tuple[float, float]
    camera_depth_m: float
    observed_depth_m: float


def project_frontier_ground_points(
    frame: NavigationFrame, candidates: Sequence[FrontierCandidate],
) -> Tuple[FrontierImageProjection, ...]:
    """只返回图内且有深度支持的地面点；缺少标定/深度时不猜测像素位置。"""
    if not candidates or frame.rgb is None or frame.depth is None or frame.camera_intrinsics is None:
        return ()
    height = len(frame.rgb)
    width = len(frame.rgb[0]) if height else 0
    if width < 3 or height < 3 or len(frame.depth) != height or any(len(row) != width for row in frame.depth):
        return ()
    intrinsics = frame.camera_intrinsics
    extrinsics = frame.camera_extrinsics_in_robot
    if not all(_finite(value) for value in (
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
        extrinsics.forward_m, extrinsics.left_m, extrinsics.height_m,
        extrinsics.yaw_rad, extrinsics.pitch_down_rad, extrinsics.roll_rad,
        frame.pose.x_m, frame.pose.y_m, frame.pose.yaw_rad,
    )) or intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0 or extrinsics.height_m <= 0.0:
        return ()
    projections = []
    for candidate in candidates:
        projection = _project_ground_point(frame, candidate.world_xy, width, height)
        if projection is not None:
            projections.append(projection)
    return tuple(projections)


def _project_ground_point(frame, world_xy, width, height) -> Optional[FrontierImageProjection]:
    if len(world_xy) != 2 or not all(_finite(value) for value in world_xy):
        return None
    intrinsics = frame.camera_intrinsics
    extrinsics = frame.camera_extrinsics_in_robot
    forward, left = world_point_to_robot(world_xy, frame.pose)
    forward -= extrinsics.forward_m
    left -= extrinsics.left_m
    up = -extrinsics.height_m
    # 与 grounding._camera_point_to_robot 相反：先撤销 yaw、pitch，再撤销 roll。
    cy, sy = math.cos(extrinsics.yaw_rad), math.sin(extrinsics.yaw_rad)
    pitched_forward = cy * forward + sy * left
    rolled_left = -sy * forward + cy * left
    cp, sp = math.cos(extrinsics.pitch_down_rad), math.sin(extrinsics.pitch_down_rad)
    camera_forward = cp * pitched_forward - sp * up
    rolled_up = sp * pitched_forward + cp * up
    cr, sr = math.cos(extrinsics.roll_rad), math.sin(extrinsics.roll_rad)
    camera_left = cr * rolled_left - sr * rolled_up
    camera_up = sr * rolled_left + cr * rolled_up
    if camera_forward <= 0.0:
        return None
    u = intrinsics.cx - intrinsics.fx * camera_left / camera_forward
    v = intrinsics.cy - intrinsics.fy * camera_up / camera_forward
    # 留一圈像素供邻域核对；超出视野的点不得贴到图片边缘。
    if not (1.0 <= u < width - 1.0 and 1.0 <= v < height - 1.0):
        return None
    col, row = round(u), round(v)
    if not (1 <= col < width - 1 and 1 <= row < height - 1):
        return None
    measured = _supported_ground_depth(frame, col, row, camera_forward)
    if measured is None:
        return None
    return FrontierImageProjection(tuple(world_xy), (u, v), camera_forward, measured)


def _supported_ground_depth(frame, col, row, expected_depth) -> Optional[float]:
    """3×3 邻域至少三点有效、中心有效；排除近处遮挡和明显不符合地面的深度。"""
    depth = frame.depth
    center = depth[row][col]
    if not _finite(center) or center <= 0.0:
        return None
    samples = [float(depth[r][c]) for r in range(row - 1, row + 2) for c in range(col - 1, col + 2)
               if _finite(depth[r][c]) and depth[r][c] > 0.0]
    if len(samples) < GROUND_MIN_DEPTH_SAMPLES:
        return None
    if min(samples) < expected_depth - GROUND_DEPTH_TOLERANCE_M:
        return None
    measured = median(samples)
    if abs(float(center) - expected_depth) > GROUND_DEPTH_TOLERANCE_M or abs(measured - expected_depth) > GROUND_DEPTH_TOLERANCE_M:
        return None
    return measured


def _finite(value) -> bool:
    return value is not None and math.isfinite(value)
