"""保存和读取 S100 与 L515 的安装外参。"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union

from .mapping import CameraMount


PathLike = Union[str, Path]
DEFAULT_CAMERA_MOUNT_PATH = Path("data/s100_l515/extrinsics.json")


def load_camera_mount(path: PathLike) -> CameraMount:
    """从标定 JSON 读取 CameraMount，并拒绝缺失或非有限参数。"""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"相机外参文件不存在：{source}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取相机外参 {source}：{exc}") from exc

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"相机外参文件格式无效：{source}")
    mount_payload = payload.get("camera_mount")
    if not isinstance(mount_payload, dict):
        raise ValueError(f"相机外参缺少 camera_mount：{source}")

    mount = CameraMount(
        height_m=_finite_field(mount_payload, "height_m"),
        forward_m=_finite_field(mount_payload, "forward_m"),
        left_m=_finite_field(mount_payload, "left_m"),
        yaw_rad=math.radians(_finite_field(mount_payload, "yaw_deg")),
        pitch_down_rad=math.radians(
            _finite_field(mount_payload, "pitch_down_deg")
        ),
        roll_rad=math.radians(
            _finite_field(mount_payload, "roll_deg", default=0.0)
        ),
    )
    _validate_mount(mount)
    return mount


def save_camera_mount(
    path: PathLike,
    mount: CameraMount,
    diagnostics: Mapping[str, Any],
) -> Path:
    """原子写入人类可读的外参和少量标定质量指标。"""
    _validate_mount(mount)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "device": "Intel RealSense L515 on WHEELTEC S100",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_convention": {
            "translation": "robot forward / left / up, metres",
            "yaw": "positive left, degrees",
            "pitch": "positive down, degrees",
            "roll": "image clockwise, degrees",
        },
        "camera_mount": {
            "height_m": round(float(mount.height_m), 6),
            "forward_m": round(float(mount.forward_m), 6),
            "left_m": round(float(mount.left_m), 6),
            "yaw_deg": round(math.degrees(float(mount.yaw_rad)), 6),
            "pitch_down_deg": round(
                math.degrees(float(mount.pitch_down_rad)), 6
            ),
            "roll_deg": round(
                math.degrees(float(mount.roll_rad)), 6
            ),
        },
        "diagnostics": dict(diagnostics),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except OSError as exc:
        raise RuntimeError(f"无法写入相机外参 {destination}：{exc}") from exc
    return destination


def _finite_field(
    payload: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool):
        raise ValueError(f"相机外参 {name} 必须为有限数")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"相机外参 {name} 必须为有限数") from None
    if not math.isfinite(converted):
        raise ValueError(f"相机外参 {name} 必须为有限数")
    return converted


def _validate_mount(mount: CameraMount) -> None:
    if not isinstance(mount, CameraMount):
        raise ValueError("mount 必须为 CameraMount")
    values = (
        mount.height_m,
        mount.forward_m,
        mount.left_m,
        mount.yaw_rad,
        mount.pitch_down_rad,
        mount.roll_rad,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("相机外参必须全部为有限数")
    if float(mount.height_m) <= 0.0:
        raise ValueError("相机高度必须为正数")


__all__ = [
    "DEFAULT_CAMERA_MOUNT_PATH",
    "load_camera_mount",
    "save_camera_mount",
]
