"""robot-nav 通用命令行入口。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

from .adapters.chassis import ChassisInterface, MotionStalledError, RecoverableMotionError
from .adapters.habitat import HabitatChassisAdapter, HabitatConfig
from .adapters.openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .adapters.perception import (
    SemanticAnalyzer,
    TargetObserver,
)
from .adapters.random_observer import RandomScoreTargetObserver
from .adapters.queued_semantics import QueuedSemanticObserver
from .adapters.object_localizer import ObjectLocalizerConfig
from .adapters.orangepi import OrangePiAdapter, OrangePiConfig
from .adapters.realsense import L515Config
from .adapters.realsense.d435i_camera import D435iConfig
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
    RelativePoseCommand,
    SearchMode,
    SearchPhase,
    TargetSearchGoal,
)
from .run_log import NavigationRunLogger, default_run_log_path


OPENCODE_GO_QWEN_ENDPOINT = "https://opencode.ai/zen/go/v1/messages"
QWEN_MODEL = "qwen3.7-plus"
SLAMTEC_STARTUP_FORWARD_M = 1.0
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
    if (
        getattr(args, "search_mode", SearchMode.OBJECT.value)
        == SearchMode.SCENE.value
        and getattr(args, "debug_random_score", False)
    ):
        parser.error("场景搜索需要 VLM，不能与 --debug-random-score 同时使用")

    if args.adapter == "habitat":
        if not args.debug_random_score and not api_key:
            parser.error(MISSING_VLM_CREDENTIAL)
        return _run_habitat(args, api_key)

    if args.adapter == "calibrate-s100-l515":
        if not args.enable_motion:
            parser.error("外参标定会移动真机，必须显式提供 --enable-motion")
        return _run_s100_l515_calibration(args)

    if args.adapter in ("calibrate-slamtec-l515", "calibrate-slamtec-d435i"):
        if not args.enable_motion:
            parser.error("外参标定会移动真机，必须显式提供 --enable-motion")
        return _run_slamtec_l515_calibration(args)

    if args.adapter == "calibrate-orangepi":
        if not args.enable_motion:
            parser.error("D435i 外参标定会移动真机，必须显式提供 --enable-motion")
        return _run_orangepi_calibration(args)

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

    if args.adapter in ("slamtec-l515", "slamtec-d435i", "orangepi"):
        if args.base_only and not args.preflight_only:
            parser.error("--base-only 只用于 --preflight-only，不可启动导航")
        calibration_path = Path(args.camera_calibration)
        if (
            not args.preflight_only
            and args.camera_height_m is None
            and not calibration_path.is_file()
        ):
            parser.error(
                "缺少本套设备的相机外参：提供 --camera-calibration，"
                "或至少提供 --camera-height-m；D435i 不可沿用 L515 安装外参"
            )
        if not args.preflight_only and not args.target:
            parser.error(f"{args.adapter} 导航模式必须提供 --target")
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
        "--seed",
        type=int,
        default=1,
        help="Habitat navmesh 随机起点种子；相同场景和种子可复现实验",
    )
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
        "slamtec-d435i", aliases=["slamtec-l515"], help="使用 SLAMTEC Hermes 与本机 RealSense D435i"
    )
    _add_navigation_arguments(slamtec, target_required=False)
    slamtec.add_argument(
        "--base-url",
        default="http://192.168.11.1:1448",
        help="Hermes Robot Agent 地址",
    )
    slamtec.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 D435i 序列号",
    )
    slamtec.add_argument(
        "--camera-calibration",
        default=str(DEFAULT_CAMERA_EXTRINSICS_PATH),
        help="完整相机外参 JSON；存在时自动读取，手动参数可覆盖",
    )
    slamtec.add_argument(
        "--camera-height-m",
        type=_positive_float,
        help="覆盖标定文件中的 D435i 光心高度（米）",
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
        default=8.0,
        help="活跃 Action 无足够位姿变化的终止秒数，默认 8",
    )
    slamtec.add_argument(
        "--max-unknown-path-m",
        type=_non_negative_float,
        default=1.5,
        help="当前剩余路径允许经过未知区的累计长度（米），超过才取消，默认 1.5",
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
        help="D435i 未连接时只预检 Hermes 位姿、地图与 Action",
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

    orangepi = adapters.add_parser("orangepi", help="经香橙派转发连接 Hermes 与 D435i，算法和控制在开发机")
    _add_navigation_arguments(orangepi, target_required=False)
    orangepi.add_argument("--base-url", default="http://127.0.0.1:11448", help="SSH 转发后的 Hermes REST 地址")
    orangepi.add_argument("--camera-url", default="http://127.0.0.1:18765", help="SSH 转发后的 D435i 数据地址")
    orangepi.add_argument("--camera-request-timeout-s", type=_positive_float, default=5.0)
    orangepi.add_argument("--camera-max-roundtrip-s", type=_positive_float, default=3.0,
                          help="相机帧允许的最大请求往返时间，超过即拒绝该帧")
    orangepi.add_argument("--camera-calibration", default="data/orangepi_d435i/extrinsics.json",
                          help="D435i 在 Hermes 上的安装外参文件，保存在开发机")
    for name in ("height-m", "forward-m", "left-m", "yaw-deg", "pitch-down-deg", "roll-deg"):
        orangepi.add_argument("--camera-" + name, type=_positive_float if name == "height-m" else _finite_float)
    orangepi.add_argument("--action-timeout-s", type=_positive_float, default=120.0)
    orangepi.add_argument("--action-stall-timeout-s", type=_positive_float, default=8.0)
    orangepi.add_argument("--max-unknown-path-m", type=_non_negative_float, default=1.5)
    orangepi.add_argument("--min-localization-quality", type=_localization_quality, default=1)
    orangepi.add_argument("--run-log", help="开发机上的导航 JSONL 日志路径")
    orangepi.add_argument("--preflight-only", action="store_true", help="只读取远程相机及 Hermes 数据，不发送运动命令")
    orangepi.add_argument("--enable-motion", action="store_true", help="允许开发机向 Hermes 发送运动命令")
    orangepi.set_defaults(base_only=False, camera_serial=None)

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
        "calibrate-slamtec-d435i", aliases=["calibrate-slamtec-l515"],
        help="利用本机 D435i IMU/RGB-D 和 Hermes 位姿标定安装外参",
    )
    slamtec_calibration.add_argument(
        "--base-url",
        default="http://192.168.11.1:1448",
        help="Hermes Robot Agent 地址",
    )
    slamtec_calibration.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 D435i 序列号",
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
    remote_calibration = adapters.add_parser("calibrate-orangepi", help="用远程 D435i IMU/RGB-D 与 Hermes 位姿自动标定")
    remote_calibration.add_argument("--base-url", default="http://127.0.0.1:11448")
    remote_calibration.add_argument("--camera-url", default="http://127.0.0.1:18765")
    remote_calibration.add_argument("--output", type=Path, default=Path("data/orangepi_d435i/extrinsics.json"))
    remote_calibration.add_argument("--turn-angle-deg", type=_positive_float, default=30.0)
    remote_calibration.add_argument("--drive-distance-m", type=_positive_float, default=0.20)
    remote_calibration.add_argument("--action-timeout-s", type=_positive_float, default=120.0)
    remote_calibration.add_argument("--min-localization-quality", type=_localization_quality, default=1)
    remote_calibration.add_argument("--camera-request-timeout-s", type=_positive_float, default=5.0)
    remote_calibration.add_argument("--camera-max-roundtrip-s", type=_positive_float, default=3.0)
    remote_calibration.add_argument("--enable-motion", action="store_true", help="明确允许左右转向和短距离平移")
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
        help="要搜索的具体物体或目的场景描述",
    )
    parser.add_argument(
        "--search-mode",
        choices=tuple(mode.value for mode in SearchMode),
        default=SearchMode.OBJECT.value,
        help="搜索具体物体 object，或寻找目的场景 scene；默认 object",
    )
    parser.add_argument("--object-class", help="接近阶段 YOLO 使用的简短类别；默认复用 --target")
    parser.add_argument("--object-python", default=_default_object_python(), help="接近阶段本地模型的 Python；默认复用 robot-nav 环境")
    parser.add_argument("--object-device", default="cuda", help="接近阶段 YOLO/SAM2 设备，默认 cuda")
    parser.add_argument("--object-yolo-model", type=Path, default=Path("data/models/yolo-world/yolov8s-world.pt"))
    parser.add_argument("--object-sam-checkpoint", type=Path, default=Path("data/models/sam2/sam2.1_hiera_small.pt"))
    parser.add_argument("--object-timeout-s", type=_positive_float, default=120.0, help="一次本地模型请求的超时秒数")
    parser.add_argument(
        "--max-cycles",
        type=_positive_int,
        default=200,
        help="最大导航周期数，默认 200",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="关闭 Rerun 实时可视化及自动录制",
    )
    parser.add_argument(
        "--rerun-save",
        type=Path,
        metavar="PATH",
        help="Rerun 录制路径；默认在 data/run_logs/ 自动创建 RRD，不覆盖已有文件",
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
        _,
        on_motion_plan,
        on_semantic_event,
    ) = _build_visualization(args)
    config = HabitatConfig(
        scene_path=args.scene,
        seed=args.seed,
        gpu_device_id=args.gpu_device_id,
    )
    with _build_queued_observer(
        args.debug_random_score, api_key, on_vlm_interaction, on_semantic_event,
        object_config=_object_config(args),
    ) as observer, HabitatChassisAdapter(
        config,
        on_motion_frame=_continuous_perception_frame_callback(
            on_motion_frame, observer, TargetSearchGoal(args.target, SearchMode(args.search_mode)),
        ),
        on_motion_plan=on_motion_plan,
    ) as chassis:
        return _run_navigation(
            chassis,
            args.target,
            SearchMode(args.search_mode),
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
        _,
        _,
        on_semantic_event,
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
    with ExitStack() as stack:
        observer = None
        motion_callback = on_motion_frame
        if not args.preflight_only:
            observer = stack.enter_context(_build_queued_observer(
                args.debug_random_score, api_key, on_vlm_interaction,
                on_semantic_event,
                object_config=_object_config(args),
            ))
            motion_callback = _continuous_perception_frame_callback(
                on_motion_frame, observer, TargetSearchGoal(args.target, SearchMode(args.search_mode)),
            )
        chassis = stack.enter_context(S100L515Adapter(config, on_motion_frame=motion_callback))
        if args.preflight_only:
            frame = chassis.read_frame()
            print(
                "S100/L515 预检通过："
                f"pose=({frame.pose.x_m:.2f}, {frame.pose.y_m:.2f}, "
                f"{frame.pose.yaw_rad:.2f})，"
                f"{'SLAM' if args.slam else '本地'} RGB-D 与占用图已生成。"
            )
            return 0

        return _run_navigation(
            chassis,
            args.target,
            SearchMode(args.search_mode),
            args.max_cycles,
            observer,
            on_cycle,
            args.debug_frontier,
        )


def _run_slamtec_l515(args: argparse.Namespace, api_key: str) -> int:
    """在开发机组装 Hermes 本地或无线 Adapter，共用导航、日志与运动规则。"""
    run_logger = _build_slamtec_run_logger(args)
    return_code = 1
    error_message = None
    observer: Optional[QueuedSemanticObserver] = None
    try:
        (
            on_cycle,
            on_motion_frame,
            on_vlm_interaction,
            _,
            on_motion_plan,
            on_semantic_event,
        ) = _build_visualization(args)
        if not args.preflight_only:
            observer = _build_queued_observer(
                args.debug_random_score,
                api_key,
                on_vlm_interaction,
                on_semantic_event=_semantic_queue_callback(on_semantic_event, run_logger),
                object_config=_object_config(args),
            )
        on_continuous_frame = None
        if observer is not None:
            on_continuous_frame = _continuous_perception_frame_callback(
                on_motion_frame,
                observer,
                TargetSearchGoal(args.target, SearchMode(args.search_mode)),
            )
        remote = args.adapter == "orangepi"
        config_type = OrangePiConfig if remote else SlamtecL515Config
        remote_options = ({
            "camera_url": args.camera_url,
            "camera_request_timeout_s": args.camera_request_timeout_s,
            "camera_max_roundtrip_s": args.camera_max_roundtrip_s,
        } if remote else {})
        config = config_type(
            base_url=args.base_url,
            camera=(
                None
                if args.base_only
                else (D435iConfig(wait_timeout_s=args.camera_request_timeout_s)
                      if remote else D435iConfig(serial_number=args.camera_serial))
            ),
            camera_extrinsics_in_robot=_slamtec_extrinsics_from_args(args),
            action_timeout_s=args.action_timeout_s,
            action_stall_timeout_s=args.action_stall_timeout_s,
            max_unknown_path_m=args.max_unknown_path_m,
            minimum_localization_quality=args.min_localization_quality,
            **remote_options,
        )
        action_progress = (
            print
            if run_logger is None
            else _action_progress_callback(run_logger)
        )
        adapter_type = OrangePiAdapter if remote else SlamtecL515Adapter
        with adapter_type(
            config,
            on_motion_frame=on_motion_frame,
            on_continuous_frame=on_continuous_frame,
            on_action_progress=action_progress,
            on_motion_plan=on_motion_plan,
        ) as chassis:
            if args.preflight_only:
                info = chassis.get_robot_info()
                slam_state = chassis.get_slam_state()
                health = chassis.get_robot_health()
                frame = chassis.read_frame()
                height = len(frame.obstacle_map.occupancy)
                width = len(frame.obstacle_map.occupancy[0]) if height else 0
                print(
                    f"{'Hermes/OrangePi/D435i' if remote else 'Hermes/D435i'} 预检读取完成："
                    f"model={info.get('modelName', 'unknown')}，"
                    f"firmware={info.get('softwareVersion', 'unknown')}，"
                    f"pose=({frame.pose.x_m:.2f}, {frame.pose.y_m:.2f}, "
                    f"{frame.pose.yaw_rad:.2f})，"
                    f"mode={slam_state.mode}，"
                    f"quality={slam_state.localization_quality}，"
                    f"health={_slamtec_health_text(health)}，"
                    f"map={width}×{height}，"
                    f"相机={'已启用' if chassis.has_camera else '未启用'}。"
                )
                return_code = 0
            else:
                if observer is None:
                    raise RuntimeError("Hermes 导航缺少目标观察器")
                _move_slamtec_forward_on_start(chassis)
                return_code = _run_navigation(
                    chassis,
                    args.target,
                    SearchMode(args.search_mode),
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
        if observer is not None:
            observer.close()
        if run_logger is not None:
            run_logger.log_run_end(return_code, error_message)
            run_logger.close()
    return return_code


def _move_slamtec_forward_on_start(chassis: SlamtecL515Adapter) -> None:
    """Hermes 正式导航启动后先沿当前底盘朝向规划前移 1 m。"""
    print(
        "Hermes 启动动作：先沿当前朝向前移 "
        f"{SLAMTEC_STARTUP_FORWARD_M:.1f} m，再开始语义搜索。"
    )
    try:
        chassis.send_relative_pose(RelativePoseCommand(forward_m=SLAMTEC_STARTUP_FORWARD_M))
    except (MotionStalledError, RecoverableMotionError) as exc:
        print(f"启动前移未完成，动作已结束，从实际位置开始搜索：{exc}", flush=True)


def _build_slamtec_run_logger(
    args: argparse.Namespace,
) -> Optional[NavigationRunLogger]:
    """正式 Hermes 导航自动保存日志；预检不创建运行日志。"""
    if args.preflight_only:
        return None
    path = (
        Path(args.run_log)
        if args.run_log
        else default_run_log_path(args.adapter)
    )
    logger = NavigationRunLogger(path)
    logger.log_run_start(
        adapter_name=args.adapter,
        target_text=args.target,
        max_cycles=args.max_cycles,
        configuration={
            "base_url": args.base_url,
            "camera_url": getattr(args, "camera_url", None),
            "camera_max_roundtrip_s": getattr(args, "camera_max_roundtrip_s", None),
            "search_mode": args.search_mode,
            "debug_random_score": args.debug_random_score,
            "debug_frontier": args.debug_frontier,
            "rerun_enabled": not args.no_rerun,
            "action_timeout_s": args.action_timeout_s,
            "action_stall_timeout_s": args.action_stall_timeout_s,
            "max_unknown_path_m": args.max_unknown_path_m,
            "startup_forward_m": SLAMTEC_STARTUP_FORWARD_M,
            "minimum_localization_quality": args.min_localization_quality,
            "perception": "queued_vlm",
            "object_local_models": not args.debug_random_score and args.search_mode == SearchMode.OBJECT.value,
            "object_python": args.object_python,
            "object_class": args.object_class or args.target,
            "object_device": args.object_device,
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
    """执行 Hermes Action 与 D435i 的独立外参标定。"""
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


def _run_orangepi_calibration(args: argparse.Namespace) -> int:
    """仅启动自动标定，不进入导航，不执行导航的启动前移 1 m。"""
    from .adapters.orangepi.calibration import calibrate_orangepi

    print(f"D435i 标定：左右转向 {args.turn_angle_deg:g}°，向前移动 {args.drive_distance_m:.2f} m。"
          "请先停止其他导航进程，清空运动区域并准备独立急停。")
    try:
        result = calibrate_orangepi(
            base_url=args.base_url, camera_url=args.camera_url, output_path=args.output,
            turn_angle_deg=args.turn_angle_deg, drive_distance_m=args.drive_distance_m,
            action_timeout_s=args.action_timeout_s, minimum_localization_quality=args.min_localization_quality,
            camera_request_timeout_s=args.camera_request_timeout_s,
            camera_max_roundtrip_s=args.camera_max_roundtrip_s,
        )
        extrinsics = load_camera_extrinsics(result.output_path)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        print(f"Hermes/D435i 外参标定失败：{exc}")
        return 1
    print(f"标定结果：height={extrinsics.height_m:.3f} m，forward={extrinsics.forward_m:.3f} m，"
          f"left={extrinsics.left_m:.3f} m，yaw={math.degrees(extrinsics.yaw_rad):.2f}°，"
          f"pitch-down={math.degrees(extrinsics.pitch_down_rad):.2f}°，"
          f"roll={math.degrees(extrinsics.roll_rad):.2f}°，residual={result.extrinsic_residual_m:.3f} m")
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
    search_mode: SearchMode,
    max_cycles: int,
    observer: TargetObserver,
    on_cycle,
    debug_frontier: bool,
    run_logger: Optional[NavigationRunLogger] = None,
) -> int:
    """重复执行环境无关的单周期入口，直到完成、失败或达到上限。"""
    goal = TargetSearchGoal(target_text, search_mode)
    state = None
    cycle_index = 0
    decision_cycles = 0
    on_cycle = _optional_callback(on_cycle, "导航可视化")
    while decision_cycles < max_cycles:
        cycle_index += 1
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

        if result.status is NavigationStatus.OK and state.phase is SearchPhase.WAITING_FOR_SEMANTICS:
            if isinstance(observer, QueuedSemanticObserver):
                observer.wait_for_result()
            continue
        decision_cycles += 1

        if state.phase is SearchPhase.COMPLETE:
            return 0
        if state.phase is SearchPhase.STOPPED:
            return 3
        if result.status in {
            NavigationStatus.NEEDS_OBSERVATION,
            NavigationStatus.NEEDS_SCENE_ASSESSMENT,
            NavigationStatus.NEEDS_FRONTIER_SCORES,
            NavigationStatus.NEEDS_TARGET_CONFIRMATION,
            NavigationStatus.NEEDS_OBJECT_LOCALIZATION,
        }:
            # 一次周期只消费一组外部感知输入。状态机若在处理该输入后立即
            # 请求下一组输入，应读取下一帧继续，而不是把“等待输入”当成失败。
            continue
        if (
            state.phase is SearchPhase.FAILED
            or result.status is not NavigationStatus.OK
        ):
            return 1

    print(f"达到最大导航决策周期数 {max_cycles}（不含队列等待），搜索尚未结束。")
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
    """打印父节点返回目标，或本轮 Frontier 候选的评分组成。"""
    details = result.debug.details
    stage = details.get("next_stage", result.debug.stage)
    if stage == "backtrack.return":
        print(
            f"[Frontier] stage={stage}，逐级返回节点={details['parent_node_id']}，"
            f"移动目标={details['destination_world_xy']}，"
            f"分支深度={details['branch_depth']}，"
            f"本节点暂存方向={details['pending_direction_count']}，到达后刷新并决策"
        )
        return
    if stage not in ("explore.select", "backtrack.resume"):
        return
    candidates = result.debug.details.get("frontier_candidates")
    if not candidates:
        return

    path_weight = result.debug.details["frontier_path_distance_weight"]
    semantic_weight = result.debug.details["frontier_semantic_score_weight"]
    print(
        "[Frontier] "
        f"stage={stage}，"
        f"本轮候选={len(candidates)}，"
        f"选择来源={result.debug.details['frontier_selection_source']}，"
        f"新方向={result.debug.details['new_frontier_count']}，"
        f"暂存旧方向={result.debug.details['deferred_frontier_count']}，"
        f"robot=({frame.pose.x_m:.3f}, {frame.pose.y_m:.3f}) m"
    )
    if stage == "backtrack.resume":
        print(f"[Frontier] 已到达父节点={details['parent_node_id']}，恢复该节点暂存方向。")
    print(
        "[Frontier] score = 前沿跨度 "
        f"- {path_weight:.2f}×路径距离 + 语义奖励（仅用于新方向排序）；"
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
            f"span={candidate['frontier_span_m']:.3f} m "
            f"path={candidate['path_distance_m']:.3f} m "
            f"distance_penalty={candidate['distance_penalty']:.3f} "
            f"vlm={semantic_text} "
            f"semantic_bonus={candidate['semantic_bonus']:+.3f} "
            f"deferred_order={candidate['deferred_order']} "
            f"score={candidate['score']:.3f}"
        )
    print(
        f"[Frontier] 本次完整移动目标={result.debug.details['destination_world_xy']}"
    )


def _build_observer(
    debug_random_score: bool,
    api_key: str,
    on_vlm_interaction=None,
) -> SemanticAnalyzer:
    if debug_random_score:
        print(
            "调试随机感知模式：不调用视觉模型，只用于调试扫描、Frontier、"
            "移动和重新选点，无法识别或到达语义目标。"
        )
        return RandomScoreTargetObserver()
    # 同一次导航共用会话标识，普通队列与物体定位请求均沿用它。
    session_id = uuid4().hex
    print(f"VLM 配置：OpenCode Go / {QWEN_MODEL}，会话 {session_id}", flush=True)
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
            opencode_session_id=session_id,
        ),
        on_vlm_interaction=on_vlm_interaction,
    )


def _build_queued_observer(
    debug_random_score: bool,
    api_key: str,
    on_vlm_interaction=None,
    on_semantic_event=None,
    *, object_config: Optional[ObjectLocalizerConfig] = None,
) -> QueuedSemanticObserver:
    """所有 Adapter 和搜索模式共用同一条后台检测与评分链。"""
    analyzer = _build_observer(
        debug_random_score,
        api_key,
        on_vlm_interaction,
    )
    return QueuedSemanticObserver(analyzer, on_event=on_semantic_event, object_config=object_config)


def _default_object_python() -> str:
    """用现有模型环境承接 Habitat 的本地推理，不修改仿真环境依赖。"""
    candidates = (
        Path(sys.prefix).parent / "robot-nav" / "bin" / "python",
        Path.home() / "micromamba" / "envs" / "robot-nav" / "bin" / "python",
    )
    return next((str(path) for path in candidates if path.is_file()), sys.executable)


def _object_config(args) -> Optional[ObjectLocalizerConfig]:
    if args.debug_random_score or args.search_mode != SearchMode.OBJECT.value:
        return None
    return ObjectLocalizerConfig(
        python_executable=args.object_python, class_text=args.object_class or "", device=args.object_device,
        yolo_model=args.object_yolo_model, sam2_checkpoint=args.object_sam_checkpoint,
        timeout_s=args.object_timeout_s,
    )


def _continuous_perception_frame_callback(
    on_motion_frame,
    observer: QueuedSemanticObserver,
    goal: TargetSearchGoal,
):
    """把最新帧送往语义预采样，并旁路记录到 Rerun。"""

    visualization = _optional_callback(on_motion_frame, "运动帧可视化")

    def callback(frame) -> None:
        observer.submit_motion_frame(frame, goal)
        if visualization is not None:
            visualization(frame)

    return callback


def _optional_callback(callback, description):
    """可视化失败后停用该回调，保持感知与导航运行。"""
    if callback is None:
        return None
    enabled = True

    def invoke(*args):
        nonlocal enabled
        if not enabled:
            return
        try:
            callback(*args)
        except Exception as exc:
            enabled = False
            print(f"{description}已停用：{exc}", flush=True)

    return invoke


def _semantic_queue_callback(on_event, run_logger: Optional[NavigationRunLogger]):
    """队列事件分别送往 Rerun 和 JSONL，显示故障不影响日志。"""
    visualization = _optional_callback(on_event, "VLM 队列可视化")

    def callback(event):
        if run_logger is not None:
            run_logger.log_semantic_queue_event(event)
        if visualization is not None:
            visualization(event)

    return callback


def _build_visualization(args: argparse.Namespace):
    if args.no_rerun or getattr(args, "preflight_only", False):
        return None, None, None, None, None, None

    from .visualization import RerunVisualizer

    visualizer = RerunVisualizer(args.target, recording_path=args.rerun_save)
    return (
        visualizer.log_cycle,
        visualizer.log_motion_frame,
        visualizer.log_vlm_interaction,
        visualizer.log_local_perception,
        visualizer.log_motion_plan,
        visualizer.log_semantic_queue_event,
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


def _non_negative_float(value: str) -> float:
    number = _finite_float(value)
    if number < 0.0:
        raise argparse.ArgumentTypeError("必须是不小于 0 的数字")
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
