"""环境创建与准备：只处理设备差异，不装配感知或导航算法。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .camera_source import camera_factory
from .adapters.realsense import D435iConfig
from .adapters.chassis import MotionStalledError, RecoverableMotionError
from .adapters.habitat import HabitatChassisAdapter, HabitatConfig
from .adapters.hermes import HermesAdapter, HermesConfig, HermesRobotHealth, load_camera_extrinsics
from .core.models import CameraExtrinsics, RelativePoseCommand


def validate_environment(args) -> None:
    """在打开设备前检查环境参数；运动授权只适用于真机。"""
    if args.adapter == "habitat":
        return
    if args.base_only and not args.preflight_only:
        raise ValueError("--base-only 只用于 --preflight-only，不可启动导航")
    if args.preflight_only:
        return
    if args.camera_height_m is None and not Path(args.camera_calibration).is_file():
        raise ValueError(
            "缺少本套设备的相机外参：提供 --camera-calibration，"
            "或至少提供 --camera-height-m；D435i 不可沿用其他相机安装外参"
        )
    if not args.enable_motion:
        raise ValueError("真机导航必须显式提供 --enable-motion")


def create_chassis(args, *, on_motion_frame=None,
                   on_motion_plan=None, on_action_progress=print, on_chassis_status=None):
    """将环境参数与公共回调接到具体 Adapter；采样时机由设备实现决定。"""
    if args.adapter == "habitat":
        return HabitatChassisAdapter(
            HabitatConfig(scene_path=args.scene, seed=args.seed,
                          gpu_device_id=args.gpu_device_id,
                          max_unknown_path_m=args.max_unknown_path_m),
            on_motion_frame=on_motion_frame,
            on_motion_plan=on_motion_plan,
        )
    if args.adapter != "hermes":
        raise ValueError(f"未知 Adapter：{args.adapter}")
    config = HermesConfig(
        base_url=args.base_url,
        camera=None if args.base_only else D435iConfig(serial_number=args.camera_serial),
        camera_extrinsics_in_robot=_hermes_extrinsics_from_args(args),
        request_timeout_s=args.request_timeout_s,
        action_poll_interval_s=args.action_poll_interval_s,
        action_progress_interval_s=args.action_progress_interval_s,
        action_stall_translation_m=args.action_stall_translation_m,
        action_stall_rotation_rad=math.radians(args.action_stall_rotation_deg),
        action_arrival_position_m=args.action_arrival_position_m,
        action_arrival_hold_s=args.action_arrival_hold_s,
        motion_frame_interval_s=args.motion_frame_interval_s,
        front_blockage_distance_m=args.front_blockage_distance_m,
        blocked_pose_radius_m=args.blocked_pose_radius_m,
        blocked_pose_duration_s=args.blocked_pose_duration_s,
        position_tolerance_m=args.position_tolerance_m,
        yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
        action_timeout_s=args.action_timeout_s,
        action_stall_timeout_s=args.action_stall_timeout_s,
        max_unknown_path_m=args.max_unknown_path_m,
        minimum_localization_quality=args.min_localization_quality,
    )
    return HermesAdapter(
        config, on_motion_frame=on_motion_frame,
        # 真机连续帧仅供可视化，不生成模型任务。
        on_continuous_frame=on_motion_frame,
        camera_factory=camera_factory(args),
        on_action_progress=on_action_progress, on_motion_plan=on_motion_plan,
        on_chassis_status=on_chassis_status,
    )


def prepare_navigation(args, chassis) -> None:
    """真机专属启动前移；仿真直接进入公共导航。"""
    if args.adapter == "hermes":
        _move_hermes_forward_on_start(chassis, args.startup_forward_m)


def run_preflight(args) -> int:
    """只读真机预检，不创建导航感知、日志或启动动作。"""
    try:
        with create_chassis(args) as chassis:
            _print_hermes_preflight(chassis)
    except RuntimeError as exc:
        print(f"Hermes 导航停止：{exc}")
        return 1
    return 0


def _print_hermes_preflight(chassis) -> None:
    """只读设备状态和一帧数据，输出底盘健康、地图尺寸与相机连接摘要。"""
    info = chassis.get_robot_info()
    slam_state = chassis.get_slam_state()
    health = chassis.get_robot_health()
    frame = chassis.read_frame()
    height = len(frame.obstacle_map.occupancy)
    width = len(frame.obstacle_map.occupancy[0]) if height else 0
    print(
        "Hermes/相机 预检读取完成："
        f"model={info.get('modelName', 'unknown')}，"
        f"firmware={info.get('softwareVersion', 'unknown')}，"
        f"pose=({frame.pose.x_m:.2f}, {frame.pose.y_m:.2f}, "
        f"{frame.pose.yaw_rad:.2f})，"
        f"mode={slam_state.mode}，"
        f"quality={slam_state.localization_quality}，"
        f"health={_hermes_health_text(health)}，"
        f"map={width}×{height}，"
        f"相机={'已启用' if chassis.has_camera else '未启用'}。"
    )


def _move_hermes_forward_on_start(chassis: HermesAdapter, distance_m: float) -> None:
    """Hermes 正式导航启动后先沿当前底盘朝向规划前移一段固定距离。"""
    if distance_m == 0:
        return
    print(
        "Hermes 启动动作：先沿当前朝向前移 "
        f"{distance_m:.1f} m，再开始语义搜索。"
    )
    try:
        chassis.send_relative_pose(RelativePoseCommand(forward_m=distance_m))
    except (MotionStalledError, RecoverableMotionError) as exc:
        print(f"启动前移未完成，动作已结束，从实际位置开始搜索：{exc}", flush=True)


def _hermes_extrinsics_from_args(args: argparse.Namespace) -> CameraExtrinsics:
    """启动时读取固定外参；六项配置齐全时不再读取旧标定文件。"""
    if args.base_only:
        return CameraExtrinsics()
    complete = all(value is not None for value in (
        args.camera_height_m, args.camera_forward_m, args.camera_left_m,
        args.camera_yaw_deg, args.camera_pitch_down_deg, args.camera_roll_deg,
    ))
    if args.camera_source == "remote" and not complete:
        raise RuntimeError(
            "远程相机需要固定外参：先运行 "
            "python hardware/hermes/fetch_camera_extrinsics.py --config config.json"
        )
    calibration_path = Path(args.camera_calibration)
    calibrated = (
        load_camera_extrinsics(calibration_path)
        if not complete and calibration_path.is_file()
        else CameraExtrinsics()
    )
    return CameraExtrinsics(
        height_m=_override(args.camera_height_m, calibrated.height_m),
        forward_m=_override(args.camera_forward_m, calibrated.forward_m),
        left_m=_override(args.camera_left_m, calibrated.left_m),
        yaw_rad=_angle_override(args.camera_yaw_deg, calibrated.yaw_rad),
        pitch_down_rad=_angle_override(args.camera_pitch_down_deg, calibrated.pitch_down_rad),
        roll_rad=_angle_override(args.camera_roll_deg, calibrated.roll_rad),
    )


def _override(value, default):
    return default if value is None else value


def _angle_override(degrees, default_rad):
    """CLI 角度用度，外参与算法使用弧度；未覆盖时保留标定值。"""
    return default_rad if degrees is None else math.radians(degrees)


def _hermes_health_text(health: HermesRobotHealth) -> str:
    if health.has_fatal:
        return "fatal"
    if health.has_error:
        return "error"
    if health.has_warning:
        return "warning"
    return "ok"
