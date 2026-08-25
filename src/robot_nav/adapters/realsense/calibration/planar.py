"""由底盘与相机的平面相对运动求解安装 yaw 和平移。"""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

from .visual_motion import MotionPair, PlanarMotion


_MAX_MOTION_YAW_ERROR_RAD = math.radians(8.0)
_MIN_TURN_RAD = math.radians(10.0)
_MAX_STRAIGHT_YAW_RAD = math.radians(5.0)
_MIN_BASE_TRANSLATION_M = 0.08
_MIN_CAMERA_TRANSLATION_M = 0.05
_MAX_MOUNT_OFFSET_M = 1.0
_MAX_RESIDUAL_M = 0.06


def solve_planar_extrinsic(
    pairs: Sequence[MotionPair],
    np: Any,
) -> Tuple[float, float, float, float]:
    """返回相机在底盘坐标系中的 forward、left、yaw 和平移残差。"""
    if not pairs:
        raise RuntimeError("没有可用于外参求解的运动对")
    _require_consistent_yaw(pairs)

    straight_pairs = [pair for pair in pairs if _is_straight(pair)]
    turn_pairs = [pair for pair in pairs if _is_turn(pair)]
    if not straight_pairs:
        raise RuntimeError("缺少可靠的直行视觉运动，无法求解相机 yaw")
    if len(turn_pairs) < 2:
        raise RuntimeError(
            "可靠旋转运动不足："
            f"{len(turn_pairs)}/2，无法求解相机前向和左向偏移"
        )

    yaw_rad = _estimate_mount_yaw(straight_pairs)
    translation = _estimate_mount_translation(turn_pairs, yaw_rad, np)
    forward_m = float(translation[0])
    left_m = float(translation[1])
    if math.hypot(forward_m, left_m) > _MAX_MOUNT_OFFSET_M:
        raise RuntimeError(
            "求得的相机水平安装偏移异常："
            f"forward={forward_m:.3f} m, left={left_m:.3f} m"
        )

    residual_m = _translation_residual(
        pairs,
        translation,
        yaw_rad,
        np,
    )
    if residual_m > _MAX_RESIDUAL_M:
        raise RuntimeError(
            "相机与底盘运动无法用同一外参解释："
            f"平移残差 {residual_m:.3f} m > {_MAX_RESIDUAL_M:.3f} m"
        )
    return forward_m, left_m, yaw_rad, residual_m


def _require_consistent_yaw(pairs: Sequence[MotionPair]) -> None:
    inconsistent = [
        pair
        for pair in pairs
        if abs(_angle_difference(pair.base.yaw_rad, pair.camera.yaw_rad))
        > _MAX_MOTION_YAW_ERROR_RAD
    ]
    if inconsistent:
        details = ", ".join(_yaw_error_text(pair) for pair in inconsistent)
        raise RuntimeError(f"相机与底盘旋转方向或幅度不一致：{details}")


def _is_straight(pair: MotionPair) -> bool:
    return (
        abs(pair.base.yaw_rad) <= _MAX_STRAIGHT_YAW_RAD
        and _motion_distance(pair.base) >= _MIN_BASE_TRANSLATION_M
        and _motion_distance(pair.camera) >= _MIN_CAMERA_TRANSLATION_M
    )


def _yaw_error_text(pair: MotionPair) -> str:
    error_rad = _angle_difference(pair.base.yaw_rad, pair.camera.yaw_rad)
    return f"{pair.label}={math.degrees(error_rad):.1f}°"


def _is_turn(pair: MotionPair) -> bool:
    return abs(pair.base.yaw_rad) >= _MIN_TURN_RAD


def _estimate_mount_yaw(pairs: Sequence[MotionPair]) -> float:
    """直行时，底盘平移方向与旋转后的相机平移方向相同。"""
    yaw_samples = [
        _angle_difference(
            math.atan2(pair.base.left_m, pair.base.forward_m),
            math.atan2(pair.camera.left_m, pair.camera.forward_m),
        )
        for pair in pairs
    ]
    sine = sum(math.sin(value) for value in yaw_samples)
    cosine = sum(math.cos(value) for value in yaw_samples)
    if math.hypot(sine, cosine) <= 1.0e-9:
        raise RuntimeError("直行运动给出的相机 yaw 相互矛盾")
    return math.atan2(sine, cosine)


def _estimate_mount_translation(
    pairs: Sequence[MotionPair],
    yaw_rad: float,
    np: Any,
) -> Any:
    """由 A·X=X·C 的平移部分堆叠最小二乘方程。"""
    camera_to_base = _rotation_2d(yaw_rad, np)
    identity = np.eye(2, dtype=float)
    coefficient_rows = []
    right_hand_rows = []
    for pair in pairs:
        base_rotation = _rotation_2d(pair.base.yaw_rad, np)
        coefficient_rows.append(base_rotation - identity)
        right_hand_rows.append(
            camera_to_base @ _translation(pair.camera, np)
            - _translation(pair.base, np)
        )
    coefficients = np.vstack(coefficient_rows)
    right_hand = np.concatenate(right_hand_rows)
    translation, _, rank, _ = np.linalg.lstsq(
        coefficients,
        right_hand,
        rcond=None,
    )
    if int(rank) < 2 or not bool(np.all(np.isfinite(translation))):
        raise RuntimeError("旋转运动不足以唯一确定相机水平安装位置")
    return translation


def _translation_residual(
    pairs: Sequence[MotionPair],
    mount_translation: Any,
    yaw_rad: float,
    np: Any,
) -> float:
    camera_to_base = _rotation_2d(yaw_rad, np)
    identity = np.eye(2, dtype=float)
    squared_errors = []
    for pair in pairs:
        base_rotation = _rotation_2d(pair.base.yaw_rad, np)
        predicted_base = (
            camera_to_base @ _translation(pair.camera, np)
            - (base_rotation - identity) @ mount_translation
        )
        error = predicted_base - _translation(pair.base, np)
        squared_errors.append(float(error @ error))
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def _translation(motion: PlanarMotion, np: Any) -> Any:
    return np.asarray((motion.forward_m, motion.left_m), dtype=float)


def _motion_distance(motion: PlanarMotion) -> float:
    return math.hypot(motion.forward_m, motion.left_m)


def _rotation_2d(yaw_rad: float, np: Any) -> Any:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)


def _angle_difference(target_rad: float, current_rad: float) -> float:
    return (target_rad - current_rad + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["solve_planar_extrinsic"]
