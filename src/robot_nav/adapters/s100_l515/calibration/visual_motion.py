"""从停车前后的 RGB-D 帧估计相机平面运动。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

from ....core.models import Pose2D
from ...realsense import L515Capture


_DEPTH_RANGE_M = (0.25, 4.0)
_ORB_FEATURE_COUNT = 2500
_MATCH_RATIO = 0.75
_MIN_PNP_POINTS = 40
_MIN_PNP_INLIERS = 25
_MAX_RELATIVE_TILT_RAD = math.radians(5.0)


@dataclass(frozen=True)
class CapturedFrame:
    label: str
    base_pose: Pose2D
    camera: L515Capture


@dataclass(frozen=True)
class PlanarMotion:
    forward_m: float
    left_m: float
    yaw_rad: float


@dataclass(frozen=True)
class MotionPair:
    label: str
    base: PlanarMotion
    camera: PlanarMotion
    visual_inliers: int


def require_visual_features(capture: L515Capture, cv2: Any) -> None:
    """在底盘移动前拒绝明显缺少纹理的场景。"""
    gray = cv2.cvtColor(capture.rgb, cv2.COLOR_RGB2GRAY)
    keypoints = cv2.ORB_create(nfeatures=_ORB_FEATURE_COUNT).detect(gray, None)
    if len(keypoints) < _MIN_PNP_POINTS * 2:
        raise RuntimeError(
            "当前画面纹理不足，无法进行 RGB-D 运动标定："
            f"仅检测到 {len(keypoints)} 个特征点"
        )


def estimate_motion_pairs(
    frames: Sequence[CapturedFrame],
    up_in_color: Any,
    cv2: Any,
    np: Any,
    report: Callable[[str], None],
) -> List[MotionPair]:
    """估计每两个相邻停车帧之间的底盘运动和相机运动。"""
    pairs: List[MotionPair] = []
    failures = []
    for source, target in zip(frames, frames[1:]):
        label = f"{source.label} → {target.label}"
        try:
            camera_motion, inliers = _estimate_camera_motion(
                source.camera,
                target.camera,
                up_in_color,
                cv2,
                np,
            )
        except RuntimeError as exc:
            failures.append(f"{label}: {exc}")
            report(f"  跳过 {label}：{exc}")
            continue
        base_motion = _relative_base_motion(source.base_pose, target.base_pose)
        pairs.append(MotionPair(label, base_motion, camera_motion, inliers))
        report(
            f"  {label}：视觉内点 {inliers}，"
            f"base Δyaw={math.degrees(base_motion.yaw_rad):.1f}°，"
            f"camera Δyaw={math.degrees(camera_motion.yaw_rad):.1f}°"
        )
    if not pairs:
        raise RuntimeError("所有 RGB-D 运动估计均失败：" + "; ".join(failures))
    return pairs


def _estimate_camera_motion(
    source: L515Capture,
    target: L515Capture,
    up_in_color: Any,
    cv2: Any,
    np: Any,
) -> Tuple[PlanarMotion, int]:
    """以源帧深度和两帧 ORB 对应点估计目标相机的相对运动。"""
    detector = cv2.ORB_create(nfeatures=_ORB_FEATURE_COUNT)
    source_points, source_descriptors = detector.detectAndCompute(
        cv2.cvtColor(source.rgb, cv2.COLOR_RGB2GRAY),
        None,
    )
    target_points, target_descriptors = detector.detectAndCompute(
        cv2.cvtColor(target.rgb, cv2.COLOR_RGB2GRAY),
        None,
    )
    if source_descriptors is None or target_descriptors is None:
        raise RuntimeError("ORB 未生成描述子")
    candidates = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        source_descriptors,
        target_descriptors,
        k=2,
    )
    matches = [
        row[0]
        for row in candidates
        if len(row) == 2 and row[0].distance < _MATCH_RATIO * row[1].distance
    ]
    object_points, image_points = _rgbd_correspondences(
        source,
        target_points,
        source_points,
        matches,
    )
    if len(object_points) < _MIN_PNP_POINTS:
        raise RuntimeError(
            "有效 RGB-D 对应点不足："
            f"{len(object_points)}/{_MIN_PNP_POINTS}"
        )

    object_array = np.asarray(object_points, dtype=np.float32)
    image_array = np.asarray(image_points, dtype=np.float32)
    camera_matrix = _camera_matrix(target.camera_intrinsics, np)
    success, rotation_vector, translation, inlier_indices = cv2.solvePnPRansac(
        object_array,
        image_array,
        camera_matrix,
        None,
        iterationsCount=250,
        reprojectionError=2.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    inlier_count = 0 if inlier_indices is None else len(inlier_indices)
    if not success or inlier_count < _MIN_PNP_INLIERS:
        raise RuntimeError(f"PnP 内点不足：{inlier_count}/{_MIN_PNP_INLIERS}")

    selected = inlier_indices.reshape(-1)
    success, rotation_vector, translation = cv2.solvePnP(
        object_array[selected],
        image_array[selected],
        camera_matrix,
        None,
        rotation_vector,
        translation,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("PnP 内点精化失败")

    target_from_source, _ = cv2.Rodrigues(rotation_vector)
    source_from_target = target_from_source.T
    target_origin_in_source = -source_from_target @ translation.reshape(3)
    color_to_level = _gravity_aligned_basis(up_in_color, np)
    level_rotation = color_to_level @ source_from_target @ color_to_level.T
    level_translation = color_to_level @ target_origin_in_source
    relative_tilt = math.acos(
        max(-1.0, min(1.0, float(level_rotation[2, 2])))
    )
    if relative_tilt > _MAX_RELATIVE_TILT_RAD:
        raise RuntimeError(
            "视觉运动包含异常俯仰/侧倾："
            f"{math.degrees(relative_tilt):.2f}°"
        )
    return (
        PlanarMotion(
            float(level_translation[0]),
            float(level_translation[1]),
            math.atan2(
                float(level_rotation[1, 0]),
                float(level_rotation[0, 0]),
            ),
        ),
        inlier_count,
    )


def _rgbd_correspondences(
    source: L515Capture,
    target_points: Any,
    source_points: Any,
    matches: Sequence[Any],
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    object_points = []
    image_points = []
    depth = source.depth_m
    intrinsics = source.camera_intrinsics
    for match in matches:
        source_pixel = source_points[match.queryIdx].pt
        col = int(round(source_pixel[0]))
        row = int(round(source_pixel[1]))
        if (
            row < 0
            or row >= depth.shape[0]
            or col < 0
            or col >= depth.shape[1]
        ):
            continue
        z = float(depth[row, col])
        if not math.isfinite(z) or not (
            _DEPTH_RANGE_M[0] <= z <= _DEPTH_RANGE_M[1]
        ):
            continue
        object_points.append(
            (
                (source_pixel[0] - intrinsics.cx) * z / intrinsics.fx,
                (source_pixel[1] - intrinsics.cy) * z / intrinsics.fy,
                z,
            )
        )
        image_points.append(target_points[match.trainIdx].pt)
    return object_points, image_points


def _camera_matrix(intrinsics: Any, np: Any) -> Any:
    return np.asarray(
        (
            (intrinsics.fx, 0.0, intrinsics.cx),
            (0.0, intrinsics.fy, intrinsics.cy),
            (0.0, 0.0, 1.0),
        ),
        dtype=float,
    )


def _gravity_aligned_basis(up_in_color: Any, np: Any) -> Any:
    """把彩色光学右/下/前坐标转换为水平前/左/上坐标。"""
    raw_forward = np.asarray((0.0, 0.0, 1.0), dtype=float)
    horizontal_forward = raw_forward - up_in_color * float(
        np.dot(raw_forward, up_in_color)
    )
    horizontal_forward = _normalized(horizontal_forward, np)
    left = _normalized(np.cross(up_in_color, horizontal_forward), np)
    return np.vstack((horizontal_forward, left, up_in_color))


def _relative_base_motion(source: Pose2D, target: Pose2D) -> PlanarMotion:
    delta_x = target.x_m - source.x_m
    delta_y = target.y_m - source.y_m
    cosine = math.cos(source.yaw_rad)
    sine = math.sin(source.yaw_rad)
    return PlanarMotion(
        delta_x * cosine + delta_y * sine,
        -delta_x * sine + delta_y * cosine,
        (target.yaw_rad - source.yaw_rad + math.pi) % (2.0 * math.pi)
        - math.pi,
    )


def _normalized(vector: Any, np: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-9:
        raise RuntimeError("重力对齐方向无法归一化")
    return vector / norm


__all__ = [
    "CapturedFrame",
    "MotionPair",
    "PlanarMotion",
    "estimate_motion_pairs",
    "require_visual_features",
]
