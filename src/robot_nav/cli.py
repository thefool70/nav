"""命令行参数定义与参数值校验。"""

import argparse
import math
import sys
from pathlib import Path
from .core.models import SearchMode


def parse_arguments(argv=None):
    """配置文件提供默认值，显式 CLI 参数覆盖；--config 可放在子命令前后。"""
    from .config import load_config, apply_config

    argv = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selector.add_argument("--config", type=Path, default=Path("config.json"))
    selected, remaining = selector.parse_known_args(argv)
    parser, subparsers = _build_parser()
    parser.add_argument("--config", help="统一 JSON 配置文件，默认当前目录 config.json")
    if "--help" in argv or "-h" in argv:
        return parser, parser.parse_args(remaining)
    try:
        apply_config(subparsers, load_config(selected.config), selected.config.resolve().parent)
    except (ValueError, OSError) as exc:
        parser.error(f"配置 {selected.config} 无效：{exc}")
    args = parser.parse_args(remaining)
    args.config = selected.config.resolve()
    _validate_arguments(parser, args)
    if args.object_python is None:
        args.object_python = _default_object_python()
    return parser, args


def _build_parser():
    """定义仿真与真机运行入口的参数。"""
    parser = argparse.ArgumentParser(description="运行机器人语义目标搜索")
    adapters = parser.add_subparsers(dest="adapter", required=True)

    _add_habitat_parser(adapters)
    _add_hermes_parser(adapters)
    return parser, adapters.choices


def _add_habitat_parser(adapters):
    """仿真场景、起点与渲染参数。"""
    habitat = adapters.add_parser("habitat", help="使用 Habitat-Sim Adapter")
    habitat.add_argument("--scene", help="Habitat .glb 场景路径")
    habitat.set_defaults(preflight_only=False)
    _add_navigation_arguments(habitat)
    habitat.add_argument(
        "--seed",
        type=int,
        help="Habitat navmesh 随机起点种子；相同场景和种子可复现实验",
    )
    habitat.add_argument(
        "--gpu-device-id",
        type=int,
        help="Habitat 渲染设备；Mesa/llvmpipe 使用 -1",
    )


def _add_hermes_parser(adapters):
    """真机连接、外参、运动约束与只读预检参数。"""
    hermes = adapters.add_parser(
        "hermes", help="使用 SLAMTEC Hermes 与本机或随车笔记本 D435i"
    )
    _add_navigation_arguments(hermes)
    hermes.add_argument(
        "--base-url",
        help="Hermes Robot Agent 地址；经随车笔记本转发时改成本机转发端口",
    )
    hermes.add_argument(
        "--camera-serial",
        help="有多台 RealSense 时指定 D435i 序列号",
    )
    hermes.add_argument(
        "--camera-calibration",
        help="完整相机外参 JSON；存在时自动读取，手动参数可覆盖",
    )
    hermes.add_argument(
        "--camera-height-m",
        type=_positive_float,
        help="覆盖标定文件中的 D435i 光心高度（米）",
    )
    hermes.add_argument(
        "--camera-forward-m",
        type=_finite_float,
        help="覆盖标定文件中的前向偏移（米）",
    )
    hermes.add_argument(
        "--camera-left-m",
        type=_finite_float,
        help="覆盖标定文件中的左向偏移（米）",
    )
    hermes.add_argument(
        "--camera-yaw-deg",
        type=_finite_float,
        help="覆盖标定文件中的左偏 yaw（度）",
    )
    hermes.add_argument(
        "--camera-pitch-down-deg",
        type=_finite_float,
        help="覆盖标定文件中的向下俯仰角（度）",
    )
    hermes.add_argument(
        "--camera-roll-deg",
        type=_finite_float,
        help="覆盖标定文件中的图像顺时针侧倾角（度）",
    )
    hermes.add_argument(
        "--action-timeout-s",
        type=_positive_float,
        help="单个 Hermes 运动 Action 的超时秒数",
    )
    hermes.add_argument(
        "--action-stall-timeout-s",
        type=_positive_float,
        help="活跃 Action 无足够位姿变化的终止秒数，默认 1",
    )
    hermes.add_argument(
        "--min-localization-quality",
        type=_localization_quality,
        help="允许运动的最低定位质量，默认 1（范围 0-100）",
    )
    hermes.add_argument(
        "--base-only",
        action="store_true",
        help="D435i 未连接时只预检 Hermes 位姿、地图与 Action",
    )
    hermes.add_argument(
        "--preflight-only",
        action="store_true",
        help="只读取设备状态与一帧数据，不执行导航",
    )
    hermes.add_argument(
        "--enable-motion",
        action="store_true",
        help="明确允许真机创建运动 Action",
    )
    hermes.add_argument("--startup-forward-m", type=_non_negative_float, help="启动前移距离，0 表示跳过")
    _add_hermes_tuning(hermes)
    _add_camera_source_arguments(hermes)


def _add_navigation_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """注册仿真与真机共用的导航选项；文件默认值稍后由 apply_config 注入。"""
    _add_vlm_arguments(parser)
    parser.add_argument(
        "--max-unknown-path-m",
        type=_non_negative_float,
        help="当前剩余路径允许经过未知区的累计长度（米），超过才取消，默认 1.5",
    )
    parser.add_argument(
        "--run-log",
        help="导航 JSONL 日志路径；默认自动保存到 data/run_logs/",
    )

    parser.add_argument(
        "--target",
        help="要搜索的具体物体或目的场景描述",
    )
    parser.add_argument(
        "--search-mode",
        choices=tuple(mode.value for mode in SearchMode),
        help="搜索具体物体 object，或寻找目的场景 scene；默认 object",
    )
    parser.add_argument("--object-class", help="接近阶段 YOLO 使用的简短类别；默认复用 --target")
    parser.add_argument(
        "--object-python",
        help="接近阶段本地模型的 Python；默认复用 robot-nav 环境",
    )
    parser.add_argument("--object-device", help="接近阶段 YOLO/SAM2 设备，默认 cuda")
    parser.add_argument(
        "--object-yolo-model",
        type=Path,
    )
    parser.add_argument(
        "--object-sam-checkpoint",
        type=Path,
    )
    parser.add_argument(
        "--object-timeout-s",
        type=_positive_float,
        help="一次本地模型请求的超时秒数",
    )
    parser.add_argument(
        "--max-cycles",
        type=_positive_int,
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


def _default_object_python() -> str:
    """用现有模型环境承接本地推理，不修改仿真环境依赖。"""
    candidates = (
        Path(sys.prefix).parent / "robot-nav" / "bin" / "python",
        Path.home() / "micromamba" / "envs" / "robot-nav" / "bin" / "python",
    )
    return next((str(path) for path in candidates if path.is_file()), sys.executable)


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


def _add_camera_source_arguments(parser):
    parser.add_argument("--camera-source", choices=("local", "remote"),
                        help="D435i 在本机 USB 或随车笔记本上采集")
    parser.add_argument("--camera-endpoint",
                        help="ZMQ IPC 订阅地址；默认经 SSH Unix 套接字转发到随车发布器")
    parser.add_argument("--camera-topic", )
    parser.add_argument("--camera-timeout-s", type=_positive_float,
                        help="订阅新帧的最大等待秒数；不是跨机绝对帧龄")


def _add_vlm_arguments(parser):
    """模型服务可由配置文件设置，也可在本次运行覆盖。"""
    parser.add_argument("--vlm-endpoint")
    parser.add_argument("--vlm-model")
    parser.add_argument("--vlm-api-format", choices=("chat_completions", "responses", "anthropic_messages"))
    parser.add_argument("--vlm-timeout-s", type=_positive_float)
    parser.add_argument("--vlm-max-output-tokens", type=_positive_int)
    parser.add_argument("--rerun", dest="no_rerun", action="store_false", default=argparse.SUPPRESS,
                        help="覆盖配置，开启可视化")
    parser.add_argument("--no-debug-random-score", dest="debug_random_score", action="store_false", default=argparse.SUPPRESS)
    parser.add_argument("--no-debug-frontier", dest="debug_frontier", action="store_false", default=argparse.SUPPRESS)


def _validate_arguments(parser, args):
    """按入口检查参数组合，后续装配直接使用已解析的字段。"""
    if args.adapter == "habitat" and not args.scene:
        parser.error("必须设置 habitat.scene 或 --scene")
    if not args.preflight_only and not args.target:
        parser.error("必须设置 navigation.target 或 --target")
    if args.search_mode == SearchMode.SCENE.value and args.debug_random_score:
        parser.error("场景搜索需要 VLM，不能与 --debug-random-score 同时使用")


def _add_hermes_tuning(parser):
    """运动监控与到位容差；单位由参数名明确表示。"""
    parser.add_argument("--request-timeout-s", type=_positive_float)
    parser.add_argument("--action-poll-interval-s", type=_positive_float)
    parser.add_argument("--action-progress-interval-s", type=_positive_float)
    parser.add_argument("--action-stall-translation-m", type=_positive_float)
    parser.add_argument("--action-stall-rotation-deg", type=_positive_float)
    parser.add_argument("--action-arrival-position-m", type=_positive_float)
    parser.add_argument("--action-arrival-hold-s", type=_positive_float)
    parser.add_argument("--motion-frame-interval-s", type=_positive_float)
    parser.add_argument("--position-tolerance-m", type=_positive_float)
    parser.add_argument("--yaw-tolerance-deg", type=_positive_float)
