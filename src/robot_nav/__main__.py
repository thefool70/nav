"""robot-nav 通用命令行入口。"""

from __future__ import annotations

import argparse
import math
import os
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
from .adapters.s100_l515 import (
    CameraMount,
    L515Config,
    RosSlamConfig,
    S100L515Adapter,
    S100L515Config,
    S100SerialConfig,
)
from .app import run_navigation_cycle
from .core.models import (
    NavigationResult,
    NavigationStatus,
    SearchPhase,
    TargetSearchGoal,
)


OPENCODE_ZEN_ENDPOINT = "https://opencode.ai/zen/v1/responses"
MUSE_MODEL = "muse-spark-1.2-contributor-free"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析运行环境并启动所选 Adapter。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    api_key = os.environ.get("ROBOT_NAV_VLM_API_KEY", "").strip()

    if args.adapter == "habitat":
        if not args.debug_random_score and not api_key:
            parser.error("缺少环境变量 ROBOT_NAV_VLM_API_KEY")
        return _run_habitat(args, api_key)

    if args.adapter == "s100-l515":
        if not args.preflight_only and not args.target:
            parser.error("s100-l515 导航模式必须提供 --target")
        if not args.preflight_only and not args.enable_motion:
            parser.error("真机导航必须显式提供 --enable-motion")
        if (
            not args.preflight_only
            and not args.debug_random_score
            and not api_key
        ):
            parser.error("缺少环境变量 ROBOT_NAV_VLM_API_KEY")
        return _run_s100_l515(args, api_key)

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
    hardware.add_argument(
        "--serial-port",
        default="COM3",
        help="S100 UART4 串口，Windows 实测默认 COM3",
    )
    hardware.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 L515 序列号",
    )
    hardware.add_argument(
        "--camera-height-m",
        required=True,
        type=_positive_float,
        help="L515 光心相对底盘原点的实测高度（米）",
    )
    hardware.add_argument(
        "--camera-forward-m",
        type=_finite_float,
        default=0.0,
        help="L515 光心相对底盘原点的前向偏移（米）",
    )
    hardware.add_argument(
        "--camera-left-m",
        type=_finite_float,
        default=0.0,
        help="L515 光心相对底盘原点的左向偏移（米）",
    )
    hardware.add_argument(
        "--camera-yaw-deg",
        type=_finite_float,
        default=0.0,
        help="相机相对底盘向左偏转的角度（度）",
    )
    hardware.add_argument(
        "--camera-pitch-down-deg",
        type=_finite_float,
        default=0.0,
        help="相机光轴向地面俯视的角度（度）",
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
    return parser


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
        help="不调用 VLM，观察器只返回随机方向评分",
    )


def _run_habitat(args: argparse.Namespace, api_key: str) -> int:
    """组装 Habitat Adapter 与通用导航循环。"""
    observer = _build_observer(args.debug_random_score, api_key)
    on_cycle, on_motion_frame = _build_visualization(args)
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
        )


def _run_s100_l515(args: argparse.Namespace, api_key: str) -> int:
    """组装 S100/L515 Adapter；预检模式不会发送非零速度。"""
    on_cycle, on_motion_frame = _build_visualization(args)
    config = S100L515Config(
        serial=S100SerialConfig(port=args.serial_port),
        camera=L515Config(serial_number=args.camera_serial),
        camera_mount=CameraMount(
            height_m=args.camera_height_m,
            forward_m=args.camera_forward_m,
            left_m=args.camera_left_m,
            yaw_rad=math.radians(args.camera_yaw_deg),
            pitch_down_rad=math.radians(args.camera_pitch_down_deg),
        ),
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

        observer = _build_observer(args.debug_random_score, api_key)
        return _run_navigation(
            chassis,
            args.target,
            args.max_cycles,
            observer,
            on_cycle,
        )


def _run_navigation(
    chassis: ChassisInterface,
    target_text: str,
    max_cycles: int,
    observer: TargetObserver,
    on_cycle,
) -> int:
    """重复执行环境无关的单周期入口，直到完成、失败或达到上限。"""
    goal = TargetSearchGoal(target_text)
    state = None
    for cycle_index in range(1, max_cycles + 1):
        result = run_navigation_cycle(
            chassis,
            goal,
            state,
            observer,
            on_cycle=on_cycle,
        )
        state = result.state
        _print_cycle(cycle_index, result)

        if state.phase is SearchPhase.COMPLETE:
            return 0
        if (
            state.phase is SearchPhase.FAILED
            or result.status is not NavigationStatus.OK
        ):
            return 1

    print(f"达到最大导航周期数 {max_cycles}，搜索尚未结束。")
    return 1


def _build_observer(
    debug_random_score: bool,
    api_key: str,
) -> TargetObserver:
    if debug_random_score:
        print(
            "调试随机感知模式：不调用视觉模型，只用于调试扫描、Frontier、"
            "移动和回退，无法识别或到达语义目标。"
        )
        return RandomScoreTargetObserver()
    return OpenAICompatibleTargetObserver(
        OpenAICompatibleConfig(
            endpoint_url=OPENCODE_ZEN_ENDPOINT,
            model=MUSE_MODEL,
            api_key=api_key,
            timeout_s=90.0,
            api_format=OpenAIApiFormat.RESPONSES,
            max_output_tokens=2048,
        )
    )


def _build_visualization(args: argparse.Namespace):
    if args.no_rerun or getattr(args, "preflight_only", False):
        return None, None

    from .visualization import RerunVisualizer

    visualizer = RerunVisualizer(args.target)
    return visualizer.log_cycle, visualizer.log_motion_frame


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


if __name__ == "__main__":
    raise SystemExit(main())
