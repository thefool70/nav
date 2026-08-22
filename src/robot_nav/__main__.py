"""robot-nav 通用命令行入口。"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from .adapters.habitat import HabitatChassisAdapter, HabitatConfig
from .adapters.openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
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
    if not api_key:
        parser.error("缺少环境变量 ROBOT_NAV_VLM_API_KEY")

    if args.adapter == "habitat":
        return _run_habitat(args, api_key)
    parser.error(f"未知 Adapter：{args.adapter}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    """定义通用入口及各 Adapter 的启动参数。"""
    parser = argparse.ArgumentParser(description="运行机器人语义目标搜索")
    adapters = parser.add_subparsers(dest="adapter", required=True)

    habitat = adapters.add_parser("habitat", help="使用 Habitat-Sim Adapter")
    habitat.add_argument("--scene", required=True, help="Habitat .glb 场景路径")
    habitat.add_argument("--target", required=True, help="要搜索的目标描述")
    habitat.add_argument(
        "--max-cycles",
        type=_positive_int,
        default=200,
        help="最大导航周期数，默认 200",
    )
    habitat.add_argument(
        "--gpu-device-id",
        type=int,
        default=-1,
        help="Habitat 渲染设备；Mesa/llvmpipe 使用 -1",
    )
    return parser


def _run_habitat(args: argparse.Namespace, api_key: str) -> int:
    """组装 Habitat Adapter 与通用导航周期。"""
    observer = OpenAICompatibleTargetObserver(
        OpenAICompatibleConfig(
            endpoint_url=OPENCODE_ZEN_ENDPOINT,
            model=MUSE_MODEL,
            api_key=api_key,
            timeout_s=90.0,
            api_format=OpenAIApiFormat.RESPONSES,
            max_output_tokens=2048,
        )
    )
    goal = TargetSearchGoal(args.target)
    state = None
    config = HabitatConfig(
        scene_path=args.scene,
        gpu_device_id=args.gpu_device_id,
    )

    with HabitatChassisAdapter(config) as chassis:
        for cycle_index in range(1, args.max_cycles + 1):
            result = run_navigation_cycle(chassis, goal, state, observer)
            state = result.state
            _print_cycle(cycle_index, result)

            if state.phase is SearchPhase.COMPLETE:
                return 0
            if (
                state.phase is SearchPhase.FAILED
                or result.status is not NavigationStatus.OK
            ):
                return 1

    print(f"达到最大导航周期数 {args.max_cycles}，搜索尚未结束。")
    return 1


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


if __name__ == "__main__":
    raise SystemExit(main())
