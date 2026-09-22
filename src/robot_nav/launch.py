"""运行组件装配：把 CLI 参数组装成底盘、感知与日志。

两种环境共用同一套装配；设备创建与准备在 environment.py。
导航循环由 app.run_navigation 执行。
阅读顺序：入口分派 → 运行日志生命周期 → 组件装配 → 各组件与回调的构造。

``__main__.py`` 只负责解析参数并调用本模块。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from .environment import create_chassis, prepare_navigation, run_preflight, validate_environment
from .adapters.openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .perception.analyzer import SemanticAnalyzer
from .adapters.random_observer import RandomScoreTargetObserver
from .perception.object_localizer import ObjectLocalizerConfig
from .core.models import SearchMode
from .perception import SemanticPerception
from .app import run_navigation
from .runtime_reporting import optional_callback
from .run_log import NavigationRunLogger, default_run_log_path


def run_entries(args: argparse.Namespace, api_key: str) -> int:
    """环境准备独立分派，正式导航只走一套公共装配。"""
    # 此处的 args 已完成文件配置与命令行合并；设备准备条件由 environment 检查。
    validate_environment(args)
    if args.preflight_only:
        return run_preflight(args)
    return _run_logged_navigation(args, api_key)


def _run_logged_navigation(args: argparse.Namespace, api_key: str) -> int:
    """记录一次运行的开始、异常与结束；组件装配和导航交给下一层。"""
    run_logger = _build_run_logger(args)
    return_code = 1
    error_message = None
    try:
        return_code = _assemble_and_run(args, api_key, run_logger)
    except RuntimeError as exc:
        return_code = 1
        error_message = str(exc)
        run_logger.log_error(exc)
        print(f"{args.adapter} 导航停止：{exc}")
    # 中断和其他异常只记录后重抛；finally 仍负责写结束记录、关闭日志。
    except BaseException as exc:
        return_code = 1
        error_message = f"{type(exc).__name__}: {exc}"
        run_logger.log_error(exc)
        raise
    finally:
        try:
            run_logger.log_run_end(return_code, error_message)
        finally:
            run_logger.close()
    return return_code


def _assemble_and_run(args: argparse.Namespace, api_key: str, run_logger: NavigationRunLogger) -> int:
    """创建显示、感知和设备，完成准备后进入两种环境共用的导航循环。"""
    visualization = _build_visualization(args)
    # 感知先创建，设备后创建；退出时先关闭设备，避免继续向已关闭队列送帧。
    with _build_perception(args, api_key, visualization, run_logger) as perception:
        with create_chassis(
            args,
            on_motion_frame=visualization.on_motion_frame,
            on_sample_frame=_motion_prefetch_callback(visualization.on_motion_frame, perception),
            on_motion_plan=visualization.on_motion_plan,
            on_action_progress=_action_progress_callback(run_logger),
        ) as chassis:
            # 仅真机执行启动前移；随后两种环境进入完全相同的导航循环。
            prepare_navigation(args, chassis)
            return run_navigation(
                chassis, args.target, SearchMode(args.search_mode), args.max_cycles,
                perception, on_cycle=visualization.on_cycle, debug_frontier=args.debug_frontier,
                run_logger=run_logger,
            )


def _build_run_logger(args: argparse.Namespace) -> NavigationRunLogger:
    """两种环境均记录导航、语义队列与执行反馈；日志配置不包含模型密钥。"""
    path = Path(args.run_log) if args.run_log else default_run_log_path(args.adapter)
    logger = NavigationRunLogger(path)
    configuration = dict(vars(args))
    configuration.update(
        config_file=str(args.config), perception="queued_vlm",
        rerun_enabled=not args.no_rerun,
        object_local_models=not args.debug_random_score and args.search_mode == SearchMode.OBJECT.value,
        object_class=args.object_class or args.target,
    )
    try:
        logger.log_run_start(
            adapter_name=args.adapter, target_text=args.target,
            max_cycles=args.max_cycles, configuration=configuration,
        )
    except BaseException:
        logger.close()
        raise
    print(f"详细运行日志：{logger.path.resolve()}")
    return logger


def _build_visualization(args: argparse.Namespace) -> _VisualizationCallbacks:
    """将 Rerun 的记录方法按用途命名；关闭显示时各字段为 None。"""
    if args.no_rerun:
        return _VisualizationCallbacks()

    from .visualization import RerunVisualizer

    visualizer = RerunVisualizer(args.target, recording_path=args.rerun_save)
    return _VisualizationCallbacks(
        on_cycle=visualizer.log_cycle,
        on_motion_frame=visualizer.log_motion_frame,
        on_vlm_interaction=optional_callback(visualizer.log_vlm_interaction, "VLM 交互可视化"),
        on_motion_plan=optional_callback(visualizer.log_motion_plan, "路径可视化"),
        on_semantic_event=visualizer.log_semantic_queue_event,
    )


def _build_perception(
    args: argparse.Namespace,
    api_key: str,
    visualization: _VisualizationCallbacks,
    run_logger: NavigationRunLogger,
) -> SemanticPerception:
    """连接队列事件、物体定位配置和模型分析器，构造公共感知模块。"""
    on_event = _perception_event_callback(visualization.on_semantic_event, run_logger)
    object_config = _object_config(args)
    # analyzer 封装模型请求；perception 管理快照、队列和结果交付。
    analyzer = _build_analyzer(args, api_key, visualization.on_vlm_interaction)
    return SemanticPerception(analyzer, on_event=on_event, object_config=object_config)


def _build_analyzer(
    args: argparse.Namespace,
    api_key: str,
    on_vlm_interaction=None,
) -> SemanticAnalyzer:
    """构造语义分析器：调试用随机分数，正式用 OpenCode Go 上的 VLM。"""
    if args.debug_random_score:
        print(
            "调试随机感知模式：不调用视觉模型，只用于调试扫描、Frontier、"
            "移动和重新选点，无法识别或到达语义目标。"
        )
        return RandomScoreTargetObserver()
    # 同一次导航共用会话标识，普通队列与物体定位请求均沿用它。
    session_id = uuid4().hex
    print(f"VLM 配置：{args.vlm_model}，会话 {session_id}", flush=True)
    return OpenAICompatibleTargetObserver(
        OpenAICompatibleConfig(
            endpoint_url=args.vlm_endpoint,
            model=args.vlm_model,
            api_key=api_key,
            timeout_s=args.vlm_timeout_s,
            api_format=OpenAIApiFormat(args.vlm_api_format),
            max_output_tokens=args.vlm_max_output_tokens,
            # Anthropic Messages 请求会显式发送 thinking=disabled（最低档）。
            reasoning_effort=None,
            opencode_session_id=session_id,
        ),
        on_vlm_interaction=on_vlm_interaction,
    )


def _object_config(args) -> Optional[ObjectLocalizerConfig]:
    # 场景搜索和随机评分不需要启动 YOLO/SAM2 的物体定位链。
    if args.debug_random_score or args.search_mode != SearchMode.OBJECT.value:
        return None
    return ObjectLocalizerConfig(
        python_executable=args.object_python, class_text=args.object_class or "", device=args.object_device,
        yolo_model=args.object_yolo_model, sam2_checkpoint=args.object_sam_checkpoint,
        timeout_s=args.object_timeout_s,
    )


def _motion_prefetch_callback(on_motion_frame, perception: SemanticPerception):
    """把最新帧送往语义预采样，并旁路记录到 Rerun。"""
    visualization = optional_callback(on_motion_frame, "运动帧可视化")

    def callback(frame) -> None:
        # 回调由 Adapter 提供帧；感知只接收它，不在这里再次读取设备。
        perception.observe_motion_frame(frame)
        if visualization is not None:
            visualization(frame)

    return callback


def _perception_event_callback(on_event, run_logger: NavigationRunLogger):
    """队列事件分别送往 Rerun 和 JSONL，显示故障不影响日志。"""
    visualization = optional_callback(on_event, "VLM 队列可视化")

    def callback(event):
        run_logger.log_semantic_queue_event(event)
        if visualization is not None:
            visualization(event)

    return callback


def _action_progress_callback(run_logger: NavigationRunLogger):
    """同时输出并落盘设备 Action 反馈。"""

    def callback(message: str) -> None:
        print(message)
        run_logger.log_action_progress(message)

    return callback


@dataclass(frozen=True)
class _VisualizationCallbacks:
    """启动装配用的具名记录函数；只保存回调，不管理资源或导航状态。"""

    on_cycle: Optional[Callable] = None
    on_motion_frame: Optional[Callable] = None
    on_vlm_interaction: Optional[Callable] = None
    on_motion_plan: Optional[Callable] = None
    on_semantic_event: Optional[Callable] = None


__all__ = [
    "run_entries",
    "run_navigation",
]
