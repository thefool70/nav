"""真机外参标定入口：独立于导航运行，不启动感知、SLAM 或可视化。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .camera_source import camera_factory
from .adapters.hermes import load_camera_extrinsics
from .adapters.realsense import D435iConfig


def run_calibration_entries(args: argparse.Namespace) -> int:
    """按 ``args.adapter`` 分派到具体的标定入口。"""
    if args.adapter == "calibrate-hermes":
        return _run_hermes_calibration(args)
    raise ValueError(f"未知标定入口：{args.adapter}")


def _run_hermes_calibration(args: argparse.Namespace) -> int:
    """执行 Hermes Action 与 D435i 的独立外参标定。"""
    from .adapters.hermes.calibration import (
        CameraCalibrationConfig,
        calibrate_hermes,
    )

    print(
        "外参标定将原地左右转动并向前移动。请清空周围至少 0.5 m，"
        "准备好急停或独立断电手段，标定期间不要触碰机器人。"
    )
    try:
        result = calibrate_hermes(
            base_url=args.base_url,
            camera_factory=camera_factory(args, calibration=True),
            camera_config=D435iConfig(serial_number=args.camera_serial),
            calibration_config=CameraCalibrationConfig(
                turn_angle_rad=math.radians(args.turn_angle_deg),
                drive_distance_m=args.drive_distance_m,
            ),
            output_path=Path(args.output),
            action_timeout_s=args.action_timeout_s,
            minimum_localization_quality=args.min_localization_quality,
            progress=print,
        )
        extrinsics = load_camera_extrinsics(result.output_path)
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"Hermes/D435i 外参标定失败：{exc}")
        return 1

    print(
        "标定结果："
        f"height={extrinsics.height_m:.3f} m, "
        f"forward={extrinsics.forward_m:.3f} m, "
        f"left={extrinsics.left_m:.3f} m, "
        f"yaw={math.degrees(extrinsics.yaw_rad):.2f}°, "
        f"pitch-down={math.degrees(extrinsics.pitch_down_rad):.2f}°, "
        f"roll={math.degrees(extrinsics.roll_rad):.2f}°, "
        f"residual={result.extrinsic_residual_m:.3f} m"
    )
    return 0


__all__ = ["run_calibration_entries"]
