"""读取 Hermes 与 D435i 的完整安装外参。

外参描述相机相对底盘的安装位姿，供地图刷新与历史物体定位共用。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Union

from ...core.models import CameraExtrinsics


PathLike = Union[str, Path]
DEFAULT_CAMERA_EXTRINSICS_PATH = Path("data/hermes_d435i/extrinsics.json")


def load_camera_extrinsics(path: PathLike) -> CameraExtrinsics:
    """读取六自由度外参，并拒绝缺失、非有限或非正高度。"""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"相机外参文件不存在：{source}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取相机外参 {source}：{exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"相机外参文件格式无效：{source}")
    values = payload.get("camera_extrinsics")
    if not isinstance(values, dict):
        raise ValueError(f"相机外参缺少 camera_extrinsics：{source}")
    extrinsics = CameraExtrinsics(
        height_m=_finite_field(values, "height_m"),
        forward_m=_finite_field(values, "forward_m"),
        left_m=_finite_field(values, "left_m"),
        yaw_rad=math.radians(_finite_field(values, "yaw_deg")),
        pitch_down_rad=math.radians(
            _finite_field(values, "pitch_down_deg")
        ),
        roll_rad=math.radians(_finite_field(values, "roll_deg")),
    )
    _validate_extrinsics(extrinsics)
    return extrinsics


def _finite_field(payload: Mapping[str, Any], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool):
        raise ValueError(f"相机外参 {name} 必须为有限数")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"相机外参 {name} 必须为有限数") from None
    if not math.isfinite(converted):
        raise ValueError(f"相机外参 {name} 必须为有限数")
    return converted


def _validate_extrinsics(extrinsics: CameraExtrinsics) -> None:
    if not isinstance(extrinsics, CameraExtrinsics):
        raise ValueError("extrinsics 必须为 CameraExtrinsics")
    values = (
        extrinsics.height_m,
        extrinsics.forward_m,
        extrinsics.left_m,
        extrinsics.yaw_rad,
        extrinsics.pitch_down_rad,
        extrinsics.roll_rad,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("相机外参必须全部为有限数")
    if float(extrinsics.height_m) <= 0.0:
        raise ValueError("相机高度必须为正数")


__all__ = [
    "DEFAULT_CAMERA_EXTRINSICS_PATH",
    "load_camera_extrinsics",
]
