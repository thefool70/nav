"""用深度地面与 IMU 重力方向估计相机高度和倾角。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

from ..l515_camera import L515Capture


_DEPTH_RANGE_M = (0.25, 4.0)
_SAMPLE_STRIDE_PX = 4
_RANSAC_ITERATIONS = 300
_INLIER_THRESHOLD_M = 0.018
_MIN_INLIERS = 300
_MIN_INLIER_RATIO = 0.12
_MAX_IMU_ANGLE_RAD = math.radians(6.0)


@dataclass(frozen=True)
class FloorEstimate:
    up_in_color: Any
    height_m: float
    inlier_ratio: float
    imu_angle_rad: float


def estimate_floor(
    capture: L515Capture,
    up_hint: Any,
    np: Any,
) -> FloorEstimate:
    """在深度图下半部分拟合与 IMU 重力方向一致的地面。"""
    depth = np.asarray(capture.depth_m, dtype=float)
    height, width = depth.shape
    rows = np.arange(height // 2, height, _SAMPLE_STRIDE_PX)
    cols = np.arange(0, width, _SAMPLE_STRIDE_PX)
    col_grid, row_grid = np.meshgrid(cols, rows)
    sampled = depth[np.ix_(rows, cols)]
    valid = (
        np.isfinite(sampled)
        & (sampled >= _DEPTH_RANGE_M[0])
        & (sampled <= _DEPTH_RANGE_M[1])
    )
    z = sampled[valid]
    intrinsics = capture.camera_intrinsics
    points = np.column_stack(
        (
            (col_grid[valid] - intrinsics.cx) * z / intrinsics.fx,
            (row_grid[valid] - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    )
    if len(points) < _MIN_INLIERS:
        raise RuntimeError(
            "用于地面拟合的有效深度点不足："
            f"{len(points)}/{_MIN_INLIERS}"
        )

    best_mask = _ransac_floor_mask(points, up_hint, np)
    inlier_count = 0 if best_mask is None else int(best_mask.sum())
    if best_mask is None or inlier_count < _MIN_INLIERS:
        raise RuntimeError("未找到可靠地面：请保证镜头下方能看到平坦地面")
    inlier_ratio = inlier_count / len(points)
    if inlier_ratio < _MIN_INLIER_RATIO:
        raise RuntimeError(
            "地面深度点占比过低："
            f"{inlier_ratio:.1%} < {_MIN_INLIER_RATIO:.1%}"
        )

    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = right_vectors[-1]
    if float(np.dot(normal, up_hint)) < 0.0:
        normal = -normal
    normal = _normalized(normal, np)
    imu_angle = math.acos(
        max(-1.0, min(1.0, float(np.dot(normal, up_hint))))
    )
    if imu_angle > _MAX_IMU_ANGLE_RAD:
        raise RuntimeError(
            "IMU 与深度地面法向不一致："
            f"{math.degrees(imu_angle):.2f}°"
        )
    height_m = abs(float(np.dot(normal, centroid)))
    if not 0.10 <= height_m <= 2.00:
        raise RuntimeError(f"地面拟合得到异常相机高度：{height_m:.3f} m")
    return FloorEstimate(normal, height_m, inlier_ratio, imu_angle)


def mount_angles_from_up(up_in_color: Any) -> Tuple[float, float]:
    """把彩色光学坐标中的向上方向转换为 pitch-down 与 roll。"""
    up_x, up_y, up_z = (float(value) for value in up_in_color)
    roll_rad = math.atan2(up_x, -up_y)
    pitch_down_rad = math.atan2(-up_z, math.hypot(up_x, up_y))
    return pitch_down_rad, roll_rad


def _ransac_floor_mask(points: Any, up_hint: Any, np: Any) -> Any:
    rng = np.random.default_rng(0)
    best_mask = None
    best_count = 0
    rough_alignment = math.cos(_MAX_IMU_ANGLE_RAD * 2.0)
    for _ in range(_RANSAC_ITERATIONS):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-9:
            continue
        normal = normal / norm
        if float(np.dot(normal, up_hint)) < 0.0:
            normal = -normal
        if float(np.dot(normal, up_hint)) < rough_alignment:
            continue
        offset = -float(np.dot(normal, sample[0]))
        mask = np.abs(points @ normal + offset) <= _INLIER_THRESHOLD_M
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    return best_mask


def _normalized(vector: Any, np: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise RuntimeError("地面法向无法归一化")
    return vector / norm


__all__ = ["FloorEstimate", "estimate_floor", "mount_angles_from_up"]
