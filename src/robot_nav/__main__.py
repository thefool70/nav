"""robot-nav 通用命令行入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional, Sequence

from .adapters.chassis import ChassisInterface
from .adapters.habitat import HabitatChassisAdapter, HabitatConfig
from .adapters.openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .adapters.perception import TargetObserver
from .adapters.random_observer import RandomScoreTargetObserver
from .adapters.realsense import L515Config
from .adapters.sam2_observer import (
    DEFAULT_SAM2_CHECKPOINT_PATH,
    Sam2ObserverConfig,
    Sam2SegmentingTargetObserver,
)
from .adapters.s100_l515 import (
    CameraMount,
    DEFAULT_CAMERA_MOUNT_PATH,
    RosSlamConfig,
    S100L515Adapter,
    S100L515Config,
    S100MotionConfig,
    S100SerialConfig,
    load_camera_mount,
)
from .adapters.slamtec_l515 import (
    DEFAULT_CAMERA_EXTRINSICS_PATH,
    SlamtecL515Adapter,
    SlamtecL515Config,
    SlamtecRobotHealth,
    load_camera_extrinsics,
)
from .app import run_navigation_cycle
from .core.models import (
    CameraExtrinsics,
    NavigationResult,
    NavigationStatus,
    SearchPhase,
    TargetSearchGoal,
)
from .run_log import NavigationRunLogger, default_run_log_path


OPENCODE_GO_QWEN_ENDPOINT = "https://opencode.ai/zen/go/v1/messages"
QWEN_MODEL = "qwen3.7-plus"
MISSING_VLM_CREDENTIAL = (
    "缺少视觉模型凭据：设置 ROBOT_NAV_VLM_API_KEY，"
    "或先用 opencode auth login 登录 OpenCode Go"
)


def _resolve_vlm_api_key() -> str:
    """优先读取项目环境变量，否则复用 OpenCode Go 本地凭据。"""
    environment_key = os.environ.get("ROBOT_NAV_VLM_API_KEY", "").strip()
    if environment_key:
        return environment_key

    xdg_data_home = os.environ.get("XDG_DATA_HOME", "").strip()
    data_home = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else Path.home() / ".local" / "share"
    )
    auth_path = data_home / "opencode" / "auth.json"
    try:
        auth_payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""

    if not isinstance(auth_payload, dict):
        return ""
    credential = auth_payload.get("opencode-go")
    if not isinstance(credential, dict) or credential.get("type") != "api":
        return ""
    api_key = credential.get("key")
    return api_key.strip() if isinstance(api_key, str) else ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析运行环境并启动所选 Adapter。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    api_key = _resolve_vlm_api_key()

    if args.adapter == "habitat":
        if not args.debug_random_score and not api_key:
            parser.error(MISSING_VLM_CREDENTIAL)
        return _run_habitat(args, api_key)

    if args.adapter == "calibrate-s100-l515":
        if not args.enable_motion:
            parser.error("外参标定会移动真机，必须显式提供 --enable-motion")
        return _run_s100_l515_calibration(args)

    if args.adapter == "calibrate-slamtec-l515":
        if not args.enable_motion:
            parser.error("外参标定会移动真机，必须显式提供 --enable-motion")
        return _run_slamtec_l515_calibration(args)

    if args.adapter == "s100-l515":
        calibration_path = Path(args.camera_calibration)
        if args.camera_height_m is None and not calibration_path.is_file():
            parser.error(
                "缺少相机高度：先运行 calibrate-s100-l515，"
                "或提供 --camera-height-m"
            )
        if not args.preflight_only and not args.target:
            parser.error("s100-l515 导航模式必须提供 --target")
        if not args.preflight_only and not args.enable_motion:
            parser.error("真机导航必须显式提供 --enable-motion")
        if (
            not args.preflight_only
            and not args.debug_random_score
            and not api_key
        ):
            parser.error(MISSING_VLM_CREDENTIAL)
        return _run_s100_l515(args, api_key)

    if args.adapter == "slamtec-l515":
        if args.base_only and not args.preflight_only:
            parser.error("--base-only 只用于 --preflight-only，不可启动导航")
        calibration_path = Path(args.camera_calibration)
        if (
            not args.preflight_only
            and args.camera_height_m is None
            and not calibration_path.is_file()
        ):
            parser.error(
                "缺少相机外参：先运行 calibrate-slamtec-l515，"
                "或至少提供 --camera-height-m"
            )
        if not args.preflight_only and not args.target:
            parser.error("slamtec-l515 导航模式必须提供 --target")
        if not args.preflight_only and not args.enable_motion:
            parser.error("真机导航必须显式提供 --enable-motion")
        if (
            not args.preflight_only
            and not args.debug_random_score
            and not api_key
        ):
            parser.error(MISSING_VLM_CREDENTIAL)
        return _run_slamtec_l515(args, api_key)

    parser.error(f"未知 Adapter：{args.adapter}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """定义通用入口及各 Adapter 的启动参数。"""
    parser = argparse.ArgumentParser(description="运行机器人语义目标搜索")
    adapters = parser.add_subparsers(dest="adapter", required=True)

    habitat = adapters.add_parser("habitat", help="使用 Habitat-Sim Adapter")
    habitat.add_argument("--scene", required=True, help="Habitat .glb 场景路径")
    _add_navigation_arguments(habitat, target_required=True)
    habitat.add_argument(
        "--gpu-device-id",
        type=int,
        default=-1,
        help="Habitat 渲染设备；Mesa/llvmpipe 使用 -1",
    )

    hardware = adapters.add_parser(
        "s100-l515", help="使用 WHEELTEC S100 与 RealSense L515"
    )
    _add_navigation_arguments(hardware, target_required=False)
    _add_s100_device_arguments(hardware)
    hardware.add_argument(
        "--camera-calibration",
        default=str(DEFAULT_CAMERA_MOUNT_PATH),
        help="相机外参 JSON；存在时自动读取，手动参数可覆盖",
    )
    hardware.add_argument(
        "--camera-height-m",
        type=_positive_float,
        help="覆盖标定文件中的 L515 光心高度（米）",
    )
    hardware.add_argument(
        "--camera-forward-m",
        type=_finite_float,
        help="覆盖标定文件中的前向偏移（米）",
    )
    hardware.add_argument(
        "--camera-left-m",
        type=_finite_float,
        help="覆盖标定文件中的左向偏移（米）",
    )
    hardware.add_argument(
        "--camera-yaw-deg",
        type=_finite_float,
        help="覆盖标定文件中的左偏 yaw（度）",
    )
    hardware.add_argument(
        "--camera-pitch-down-deg",
        type=_finite_float,
        help="覆盖标定文件中的向下俯仰角（度）",
    )
    hardware.add_argument(
        "--camera-roll-deg",
        type=_finite_float,
        help="覆盖标定文件中的图像顺时针侧倾角（度）",
    )
    hardware.add_argument(
        "--slam",
        action="store_true",
        help="使用 ROS 2 slam_toolbox 提供位姿与占用图",
    )
    hardware.add_argument(
        "--ros-depth-unit-m",
        type=_positive_float,
        default=0.00025,
        help="ROS Z16 深度每单位的米数；L515 默认 0.00025",
    )
    hardware.add_argument(
        "--preflight-only",
        action="store_true",
        help="只检查 S100 反馈、静止状态和 L515 帧，不执行导航",
    )
    hardware.add_argument(
        "--enable-motion",
        action="store_true",
        help="明确允许真机发送非零运动命令",
    )

    slamtec = adapters.add_parser(
        "slamtec-l515", help="使用 SLAMTEC Hermes 与外接 RealSense L515"
    )
    _add_navigation_arguments(slamtec, target_required=False)
    slamtec.add_argument(
        "--base-url",
        default="http://192.168.11.1:1448",
        help="Hermes Robot Agent 地址",
    )
    slamtec.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 L515 序列号",
    )
    slamtec.add_argument(
        "--sam2-checkpoint",
        default=str(DEFAULT_SAM2_CHECKPOINT_PATH),
        help="SAM2.1 Hiera Small 模型文件",
    )
    slamtec.add_argument(
        "--sam2-device",
        default="cuda",
        help="SAM2 推理设备，默认 cuda",
    )
    slamtec.add_argument(
        "--camera-calibration",
        default=str(DEFAULT_CAMERA_EXTRINSICS_PATH),
        help="完整相机外参 JSON；存在时自动读取，手动参数可覆盖",
    )
    slamtec.add_argument(
        "--camera-height-m",
        type=_positive_float,
        help="覆盖标定文件中的 L515 光心高度（米）",
    )
    slamtec.add_argument(
        "--camera-forward-m",
        type=_finite_float,
        help="覆盖标定文件中的前向偏移（米）",
    )
    slamtec.add_argument(
        "--camera-left-m",
        type=_finite_float,
        help="覆盖标定文件中的左向偏移（米）",
    )
    slamtec.add_argument(
        "--camera-yaw-deg",
        type=_finite_float,
        help="覆盖标定文件中的左偏 yaw（度）",
    )
    slamtec.add_argument(
        "--camera-pitch-down-deg",
        type=_finite_float,
        help="覆盖标定文件中的向下俯仰角（度）",
    )
    slamtec.add_argument(
        "--camera-roll-deg",
        type=_finite_float,
        help="覆盖标定文件中的图像顺时针侧倾角（度）",
    )
    slamtec.add_argument(
        "--action-timeout-s",
        type=_positive_float,
        default=120.0,
        help="单个 Hermes 运动 Action 的超时秒数",
    )
    slamtec.add_argument(
        "--action-stall-timeout-s",
        type=_positive_float,
        default=15.0,
        help="活跃 Action 无足够位姿变化的终止秒数，默认 15",
    )
    slamtec.add_argument(
        "--run-log",
        help=(
            "Hermes 导航 JSONL 日志路径；默认自动保存到 "
            "data/run_logs/"
        ),
    )
    slamtec.add_argument(
        "--min-localization-quality",
        type=_localization_quality,
        default=1,
        help="允许运动的最低定位质量，默认 1（范围 0-100）",
    )
    slamtec.add_argument(
        "--base-only",
        action="store_true",
        help="L515 未连接时只预检 Hermes 位姿、地图与 Action",
    )
    slamtec.add_argument(
        "--preflight-only",
        action="store_true",
        help="只读取设备状态与一帧数据，不执行导航",
    )
    slamtec.add_argument(
        "--enable-motion",
        action="store_true",
        help="明确允许真机创建运动 Action",
    )

    calibration = adapters.add_parser(
        "calibrate-s100-l515",
        help="利用 L515 IMU、RGB-D 和 S100 里程计标定安装外参",
    )
    _add_s100_device_arguments(calibration)
    calibration.add_argument(
        "--output",
        default=str(DEFAULT_CAMERA_MOUNT_PATH),
        help="标定结果 JSON 路径",
    )
    calibration.add_argument(
        "--turn-angle-deg",
        type=_positive_float,
        default=30.0,
        help="左右标定转角，默认 30°",
    )
    calibration.add_argument(
        "--drive-distance-m",
        type=_positive_float,
        default=0.20,
        help="标定直行距离，默认 0.20 m",
    )
    calibration.add_argument(
        "--enable-motion",
        action="store_true",
        help="确认场地清空并允许标定程序移动真机",
    )

    slamtec_calibration = adapters.add_parser(
        "calibrate-slamtec-l515",
        help="利用 L515 IMU/RGB-D 和 Hermes 位姿标定安装外参",
    )
    slamtec_calibration.add_argument(
        "--base-url",
        default="http://192.168.11.1:1448",
        help="Hermes Robot Agent 地址",
    )
    slamtec_calibration.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 L515 序列号",
    )
    slamtec_calibration.add_argument(
        "--output",
        default=str(DEFAULT_CAMERA_EXTRINSICS_PATH),
        help="标定结果 JSON 路径",
    )
    slamtec_calibration.add_argument(
        "--turn-angle-deg",
        type=_positive_float,
        default=30.0,
        help="左右标定转角，默认 30°",
    )
    slamtec_calibration.add_argument(
        "--drive-distance-m",
        type=_positive_float,
        default=0.20,
        help="标定直行距离，默认 0.20 m",
    )
    slamtec_calibration.add_argument(
        "--action-timeout-s",
        type=_positive_float,
        default=120.0,
        help="单个 Hermes 标定 Action 的超时秒数",
    )
    slamtec_calibration.add_argument(
        "--min-localization-quality",
        type=_localization_quality,
        default=1,
        help="定位模式的最低质量；建图模式不应用该阈值",
    )
    slamtec_calibration.add_argument(
        "--enable-motion",
        action="store_true",
        help="确认场地清空并允许标定程序移动真机",
    )
    return parser


def _add_s100_device_arguments(parser: argparse.ArgumentParser) -> None:
    """添加 S100 串口和 L515 设备选择参数。"""
    parser.add_argument(
        "--serial-port",
        default="COM3",
        help="S100 UART4 串口；WSL 启动脚本会自动填入",
    )
    parser.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 L515 序列号",
    )


def _add_navigation_arguments(
    parser: argparse.ArgumentParser,
    target_required: bool,
) -> None:
    parser.add_argument(
        "--target",
        required=target_required,
        help="要搜索的目标描述",
    )
    parser.add_argument(
        "--max-cycles",
        type=_positive_int,
        default=200,
        help="最大导航周期数，默认 200",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="不启动 Rerun 实时可视化",
    )
    parser.add_argument(
        "--debug-random-score",
        action="store_true",
        help="不调用 VLM，为整批 Frontier 返回随机分数",
    )
    parser.add_argument(
        "--debug-frontier",
        action="store_true",
        help="在发送移动命令前打印本轮 Frontier 候选及评分计算",
    )


def _run_habitat(args: argparse.Namespace, api_key: str) -> int:
    """组装 Habitat Adapter 与通用导航循环。"""
    (
        on_cycle,
        on_motion_frame,
        on_vlm_interaction,
    ) = _build_visualization(args)
    observer = _build_observer(
        args.debug_random_score,
        api_key,
        on_vlm_interaction,
    )
    config = HabitatConfig(
        scene_path=args.scene,
        gpu_device_id=args.gpu_device_id,
    )
    with HabitatChassisAdapter(
        config,
        on_motion_frame=on_motion_frame,
    ) as chassis:
        return _run_navigation(
            chassis,
            args.target,
            args.max_cycles,
            observer,
            on_cycle,
            args.debug_frontier,
        )


def _run_s100_l515(args: argparse.Namespace, api_key: str) -> int:
    """组装 S100/L515 Adapter；预检模式不会发送非零速度。"""
    (
        on_cycle,
        on_motion_frame,
        on_vlm_interaction,
    ) = _build_visualization(args)
    config = S100L515Config(
        serial=S100SerialConfig(port=args.serial_port),
        camera=L515Config(serial_number=args.camera_serial),
        camera_mount=_camera_mount_from_args(args),
        slam=(
            RosSlamConfig(depth_unit_m=args.ros_depth_unit_m)
            if args.slam
            else None
        ),
    )
    with S100L515Adapter(
        config,
        on_motion_frame=on_motion_frame,
    ) as chassis:
        if args.preflight_only:
            frame = chassis.read_frame()
            print(
                "S100/L515 预检通过："
                f"pose=({frame.pose.x_m:.2f}, {frame.pose.y_m:.2f}, "
                f"{frame.pose.yaw_rad:.2f})，"
                f"{'SLAM' if args.slam else '本地'} RGB-D 与占用图已生成。"
            )
            return 0

        observer = _build_observer(
            args.debug_random_score,
            api_key,
            on_vlm_interaction,
        )
        return _run_navigation(
            chassis,
            args.target,
            args.max_cycles,
            observer,
            on_cycle,
            args.debug_frontier,
        )


def _run_slamtec_l515(args: argparse.Namespace, api_key: str) -> int:
    """组装 Hermes/L515 Adapter；base-only 仅用于无相机预检。"""
    run_logger = _build_slamtec_run_logger(args)
    return_code = 1
    error_message = None
    try:
        (
            on_cycle,
            on_motion_frame,
            on_vlm_interaction,
        ) = _build_visualization(args)
        config = SlamtecL515Config(
            base_url=args.base_url,
            camera=(
                None
                if args.base_only
                else L515Config(serial_number=args.camera_serial)
            ),
            camera_extrinsics_in_robot=_slamtec_extrinsics_from_args(args),
            action_timeout_s=args.action_timeout_s,
            action_stall_timeout_s=args.action_stall_timeout_s,
            minimum_localization_quality=args.min_localization_quality,
        )
        action_progress = (
            print
            if run_logger is None
            else _action_progress_callback(run_logger)
        )
        with SlamtecL515Adapter(
            config,
            on_motion_frame=on_motion_frame,
            on_action_progress=action_progress,
        ) as chassis:
            if args.preflight_only:
                info = chassis.get_robot_info()
                slam_state = chassis.get_slam_state()
                health = chassis.get_robot_health()
                frame = chassis.read_frame()
                height = len(frame.obstacle_map.occupancy)
                width = len(frame.obstacle_map.occupancy[0]) if height else 0
                print(
                    "Hermes/L515 预检通过："
                    f"model={info.get('modelName', 'unknown')}，"
                    f"firmware={info.get('softwareVersion', 'unknown')}，"
                    f"pose=({frame.pose.x_m:.2f}, {frame.pose.y_m:.2f}, "
                    f"{frame.pose.yaw_rad:.2f})，"
                    f"mode={slam_state.mode}，"
                    f"quality={slam_state.localization_quality}，"
                    f"health={_slamtec_health_text(health)}，"
                    f"map={width}×{height}，"
                    f"L515={'已启用' if chassis.has_camera else '未启用'}。"
                )
                return_code = 0
            else:
                observer = _build_slamtec_observer(
                    args,
                    api_key,
                    on_vlm_interaction,
                )
                return_code = _run_navigation(
                    chassis,
                    args.target,
                    args.max_cycles,
                    observer,
                    on_cycle,
                    args.debug_frontier,
                    run_logger=run_logger,
                )
    except RuntimeError as exc:
        error_message = str(exc)
        if run_logger is not None:
            run_logger.log_error(exc)
        print(f"Hermes 导航停止：{exc}")
        return_code = 1
    except BaseException as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        if run_logger is not None:
            run_logger.log_error(exc)
        raise
    finally:
        if run_logger is not None:
            run_logger.log_run_end(return_code, error_message)
            run_logger.close()
    return return_code


def _build_slamtec_run_logger(
    args: argparse.Namespace,
) -> Optional[NavigationRunLogger]:
    """正式 Hermes 导航自动保存日志；预检不创建运行日志。"""
    if args.preflight_only:
        return None
    path = (
        Path(args.run_log)
        if args.run_log
        else default_run_log_path("slamtec-l515")
    )
    logger = NavigationRunLogger(path)
    logger.log_run_start(
        adapter_name="slamtec-l515",
        target_text=args.target,
        max_cycles=args.max_cycles,
        configuration={
            "base_url": args.base_url,
            "debug_random_score": args.debug_random_score,
            "debug_frontier": args.debug_frontier,
            "rerun_enabled": not args.no_rerun,
            "action_timeout_s": args.action_timeout_s,
            "action_stall_timeout_s": args.action_stall_timeout_s,
            "minimum_localization_quality": args.min_localization_quality,
            "sam2_enabled": not args.debug_random_score,
            "sam2_checkpoint": args.sam2_checkpoint,
            "sam2_device": args.sam2_device,
        },
    )
    print(f"详细运行日志：{logger.path.resolve()}")
    return logger


def _action_progress_callback(run_logger: NavigationRunLogger):
    """同时输出并落盘 Hermes Action 反馈。"""

    def callback(message: str) -> None:
        print(message)
        run_logger.log_action_progress(message)

    return callback


def _run_s100_l515_calibration(args: argparse.Namespace) -> int:
    """执行独立真机外参标定；不启动导航、SLAM 或视觉模型。"""
    from .adapters.s100_l515.calibration import (
        CameraCalibrationConfig,
        calibrate_s100_l515,
    )

    print(
        "外参标定将原地左右转动并向前移动。请清空周围至少 0.5 m，"
        "准备好独立断电手段，标定期间不要触碰机器人。"
    )
    try:
        result = calibrate_s100_l515(
            serial_config=S100SerialConfig(port=args.serial_port),
            camera_config=L515Config(serial_number=args.camera_serial),
            motion_config=S100MotionConfig(),
            calibration_config=CameraCalibrationConfig(
                turn_angle_rad=math.radians(args.turn_angle_deg),
                drive_distance_m=args.drive_distance_m,
            ),
            output_path=Path(args.output),
            progress=print,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"S100/L515 外参标定失败：{exc}")
        return 1

    mount = result.camera_mount
    print(
        "标定结果："
        f"height={mount.height_m:.3f} m, "
        f"forward={mount.forward_m:.3f} m, "
        f"left={mount.left_m:.3f} m, "
        f"yaw={math.degrees(mount.yaw_rad):.2f}°, "
        f"pitch-down={math.degrees(mount.pitch_down_rad):.2f}°, "
        f"roll={math.degrees(mount.roll_rad):.2f}°, "
        f"residual={result.extrinsic_residual_m:.3f} m"
    )
    return 0


def _run_slamtec_l515_calibration(args: argparse.Namespace) -> int:
    """执行 Hermes Action 与 L515 的独立外参标定。"""
    from .adapters.slamtec_l515.calibration import (
        CameraCalibrationConfig,
        calibrate_slamtec_l515,
    )

    print(
        "外参标定将原地左右转动并向前移动。请清空周围至少 0.5 m，"
        "准备好急停或独立断电手段，标定期间不要触碰机器人。"
    )
    try:
        result = calibrate_slamtec_l515(
            base_url=args.base_url,
            camera_config=L515Config(serial_number=args.camera_serial),
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
        print(f"Hermes/L515 外参标定失败：{exc}")
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


def _camera_mount_from_args(args: argparse.Namespace) -> CameraMount:
    """读取默认标定文件，并用显式命令行参数覆盖对应字段。"""
    calibration_path = Path(args.camera_calibration)
    calibrated = (
        load_camera_mount(calibration_path)
        if calibration_path.is_file()
        else None
    )
    if args.camera_height_m is not None:
        height_m = args.camera_height_m
    elif calibrated is not None:
        height_m = calibrated.height_m
    else:
        raise ValueError("缺少相机高度或有效标定文件")
    return CameraMount(
        height_m=height_m,
        forward_m=_mount_value(
            args.camera_forward_m,
            calibrated,
            "forward_m",
        ),
        left_m=_mount_value(args.camera_left_m, calibrated, "left_m"),
        yaw_rad=math.radians(args.camera_yaw_deg)
        if args.camera_yaw_deg is not None
        else _mount_value(None, calibrated, "yaw_rad"),
        pitch_down_rad=math.radians(args.camera_pitch_down_deg)
        if args.camera_pitch_down_deg is not None
        else _mount_value(None, calibrated, "pitch_down_rad"),
        roll_rad=math.radians(args.camera_roll_deg)
        if args.camera_roll_deg is not None
        else _mount_value(None, calibrated, "roll_rad"),
    )


def _slamtec_extrinsics_from_args(
    args: argparse.Namespace,
) -> CameraExtrinsics:
    """读取 Hermes 标定文件，并用显式命令行参数逐项覆盖。"""
    calibration_path = Path(args.camera_calibration)
    calibrated = (
        load_camera_extrinsics(calibration_path)
        if calibration_path.is_file()
        else None
    )
    return CameraExtrinsics(
        height_m=_extrinsic_value(
            args.camera_height_m,
            calibrated,
            "height_m",
        ),
        forward_m=_extrinsic_value(
            args.camera_forward_m,
            calibrated,
            "forward_m",
        ),
        left_m=_extrinsic_value(
            args.camera_left_m,
            calibrated,
            "left_m",
        ),
        yaw_rad=math.radians(args.camera_yaw_deg)
        if args.camera_yaw_deg is not None
        else _extrinsic_value(None, calibrated, "yaw_rad"),
        pitch_down_rad=math.radians(args.camera_pitch_down_deg)
        if args.camera_pitch_down_deg is not None
        else _extrinsic_value(None, calibrated, "pitch_down_rad"),
        roll_rad=math.radians(args.camera_roll_deg)
        if args.camera_roll_deg is not None
        else _extrinsic_value(None, calibrated, "roll_rad"),
    )


def _extrinsic_value(
    override: Optional[float],
    calibrated: Optional[CameraExtrinsics],
    attribute: str,
) -> float:
    if override is not None:
        return override
    if calibrated is not None:
        return float(getattr(calibrated, attribute))
    return 0.0


def _slamtec_health_text(health: SlamtecRobotHealth) -> str:
    if health.has_fatal:
        return "fatal"
    if health.has_error:
        return "error"
    if health.has_warning:
        return "warning"
    return "ok"


def _mount_value(
    override: Optional[float],
    calibrated: Optional[CameraMount],
    attribute: str,
) -> float:
    if override is not None:
        return override
    if calibrated is not None:
        return float(getattr(calibrated, attribute))
    return 0.0


def _run_navigation(
    chassis: ChassisInterface,
    target_text: str,
    max_cycles: int,
    observer: TargetObserver,
    on_cycle,
    debug_frontier: bool,
    run_logger: Optional[NavigationRunLogger] = None,
) -> int:
    """重复执行环境无关的单周期入口，直到完成、失败或达到上限。"""
    goal = TargetSearchGoal(target_text)
    state = None
    for cycle_index in range(1, max_cycles + 1):
        cycle_callback = _with_frontier_debug(on_cycle, debug_frontier)
        if run_logger is not None:
            cycle_callback = _with_run_log(
                cycle_callback,
                run_logger,
                cycle_index,
            )
        result = run_navigation_cycle(
            chassis,
            goal,
            state,
            observer,
            on_cycle=cycle_callback,
        )
        state = result.state
        if run_logger is not None:
            run_logger.log_cycle_result(cycle_index, result)
        _print_cycle(cycle_index, result)

        if state.phase is SearchPhase.COMPLETE:
            return 0
        if result.status in {
            NavigationStatus.NEEDS_OBSERVATION,
            NavigationStatus.NEEDS_FRONTIER_SCORES,
        }:
            # 一次周期只消费一组外部感知输入。状态机若在处理该输入后立即
            # 请求下一组输入，应读取下一帧继续，而不是把“等待输入”当成失败。
            continue
        if (
            state.phase is SearchPhase.FAILED
            or result.status is not NavigationStatus.OK
        ):
            return 1

    print(f"达到最大导航周期数 {max_cycles}，搜索尚未结束。")
    return 1


def _with_run_log(
    on_cycle,
    run_logger: NavigationRunLogger,
    cycle_index: int,
):
    """把发送命令前的完整决策帧写入当前运行日志。"""

    def callback(frame, observation, result) -> None:
        run_logger.log_cycle_decision(
            cycle_index,
            frame,
            observation,
            result,
        )
        if on_cycle is not None:
            on_cycle(frame, observation, result)

    return callback


def _with_frontier_debug(on_cycle, enabled: bool):
    """把可选 Frontier 终端输出接到发送命令前的周期回调。"""
    if not enabled:
        return on_cycle

    def callback(frame, observation, result) -> None:
        if on_cycle is not None:
            on_cycle(frame, observation, result)
        _print_frontier_debug(frame, result)

    return callback


def _print_frontier_debug(frame, result: NavigationResult) -> None:
    """逐项打印本轮 Frontier 候选的评分组成。"""
    if result.debug.stage != "explore.select":
        return
    candidates = result.debug.details.get("frontier_candidates")
    if not candidates:
        return

    path_weight = result.debug.details["frontier_path_distance_weight"]
    semantic_weight = result.debug.details["frontier_semantic_score_weight"]
    print(
        "[Frontier] "
        f"本轮候选={len(candidates)}，"
        f"robot=({frame.pose.x_m:.3f}, {frame.pose.y_m:.3f}) m"
    )
    print(
        "[Frontier] score = 前沿长度 "
        f"- {path_weight:.2f}×路径距离 + 语义奖励；"
        f"语义奖励 = {semantic_weight:.2f}×(2×VLM分数-1)"
    )
    for rank, candidate in enumerate(candidates, start=1):
        selected = " selected" if rank == 1 else ""
        semantic_score = candidate["semantic_score"]
        semantic_text = (
            "none" if semantic_score is None else f"{semantic_score:.3f}"
        )
        print(
            f"[Frontier #{rank:02d}{selected}] "
            f"id={candidate['candidate_id']} "
            f"grid=({candidate['row']}, {candidate['col']}) "
            f"world=({candidate['world_x_m']:.3f}, "
            f"{candidate['world_y_m']:.3f}) m "
            f"cells={candidate['frontier_cell_count']} "
            f"length={candidate['frontier_length_m']:.3f} m "
            f"path={candidate['path_distance_m']:.3f} m "
            f"distance_penalty={candidate['distance_penalty']:.3f} "
            f"vlm={semantic_text} "
            f"semantic_bonus={candidate['semantic_bonus']:+.3f} "
            f"score={candidate['score']:.3f}"
        )


def _build_observer(
    debug_random_score: bool,
    api_key: str,
    on_vlm_interaction=None,
) -> TargetObserver:
    if debug_random_score:
        print(
            "调试随机感知模式：不调用视觉模型，只用于调试扫描、Frontier、"
            "移动和回退，无法识别或到达语义目标。"
        )
        return RandomScoreTargetObserver()
    return OpenAICompatibleTargetObserver(
        OpenAICompatibleConfig(
            endpoint_url=OPENCODE_GO_QWEN_ENDPOINT,
            model=QWEN_MODEL,
            api_key=api_key,
            timeout_s=90.0,
            api_format=OpenAIApiFormat.ANTHROPIC_MESSAGES,
            max_output_tokens=2048,
            # Anthropic Messages 请求会显式发送 thinking=disabled（最低档）。
            reasoning_effort=None,
        ),
        on_vlm_interaction=on_vlm_interaction,
    )


def _build_slamtec_observer(
    args: argparse.Namespace,
    api_key: str,
    on_vlm_interaction=None,
) -> TargetObserver:
    """为 Hermes 语义导航在 VLM 框选之后接入 SAM2 掩码。"""
    box_observer = _build_observer(
        args.debug_random_score,
        api_key,
        on_vlm_interaction,
    )
    if args.debug_random_score:
        return box_observer
    if not isinstance(box_observer, OpenAICompatibleTargetObserver):
        raise RuntimeError("当前目标观察器不支持 SAM2 所需的重新框选接口")

    print(
        f"SAM2 目标掩码已启用：{args.sam2_checkpoint}，"
        f"device={args.sam2_device}。"
    )
    return Sam2SegmentingTargetObserver(
        box_observer,
        Sam2ObserverConfig(
            checkpoint_path=Path(args.sam2_checkpoint),
            device=args.sam2_device,
        ),
    )


def _build_visualization(args: argparse.Namespace):
    if args.no_rerun or getattr(args, "preflight_only", False):
        return None, None, None

    from .visualization import RerunVisualizer

    visualizer = RerunVisualizer(args.target)
    return (
        visualizer.log_cycle,
        visualizer.log_motion_frame,
        visualizer.log_vlm_interaction,
    )


def _print_cycle(cycle_index: int, result: NavigationResult) -> None:
    """输出足以沿算法步骤排错的一行周期信息。"""
    print(
        f"[{cycle_index:03d}] status={result.status.value} "
        f"phase={result.state.phase.value} stage={result.debug.stage} | "
        f"{result.debug.message}"
    )


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def _positive_float(value: str) -> float:
    number = _finite_float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError("必须是正数")
    return number


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("必须是有限数")
    return number


def _localization_quality(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("必须是 0 到 100 的整数")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
