"""运行组件装配：把 CLI 参数组装成底盘、感知与日志。

按 Adapter 类型装配组件并启动（Habitat / Hermes）。
导航循环由 app.run_navigation 执行。

``__main__.py`` 只负责解析参数并调用本模块；标定流程在
:mod:`~robot_nav.calibration_launch` 中保持独立入口。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .camera_source import camera_factory
from .adapters.realsense import D435iConfig
from .adapters.chassis import MotionStalledError, RecoverableMotionError
from .adapters.habitat import HabitatChassisAdapter, HabitatConfig
from .adapters.hermes import HermesAdapter, HermesConfig, HermesRobotHealth, load_camera_extrinsics
from .adapters.openai_compatible import (
    OpenAIApiFormat,
    OpenAICompatibleConfig,
    OpenAICompatibleTargetObserver,
)
from .perception.analyzer import SemanticAnalyzer
from .adapters.random_observer import RandomScoreTargetObserver
from .perception.object_localizer import ObjectLocalizerConfig
from .core.models import CameraExtrinsics, RelativePoseCommand, SearchMode
from .perception import SemanticPerception
from .app import run_navigation
from .runtime_reporting import _optional_callback
from .run_log import NavigationRunLogger, default_run_log_path

MISSING_VLM_CREDENTIAL = (
    "缺少视觉模型凭据：设置 ROBOT_NAV_VLM_API_KEY，"
    "或先用 opencode auth login 登录 OpenCode Go"
)


def run_entries(args: argparse.Namespace, api_key: str) -> int:
    """按 ``args.adapter`` 分派到具体的运行入口。"""
    if args.adapter == "habitat":
        return _run_habitat(args, api_key)
    if args.adapter == "hermes":
        return _run_hermes(args, api_key)
    raise ValueError(f"未知 Adapter：{args.adapter}")


def _run_habitat(args: argparse.Namespace, api_key: str) -> int:
    """组装 Habitat Adapter 与通用导航循环。"""
    if not args.debug_random_score and not api_key:
        raise ValueError(MISSING_VLM_CREDENTIAL)
    on_cycle, on_motion_frame, on_vlm_interaction, on_motion_plan, on_semantic_event = (
        _build_visualization(args)
    )
    config = HabitatConfig(
        scene_path=args.scene,
        seed=args.seed,
        gpu_device_id=args.gpu_device_id,
    )
    with _build_perception(
        args, api_key, on_vlm_interaction, on_semantic_event,
        object_config=_object_config(args),
    ) as perception, HabitatChassisAdapter(
        config,
        on_motion_frame=_motion_prefetch_callback(on_motion_frame, perception),
        on_motion_plan=on_motion_plan,
    ) as chassis:
        return run_navigation(
            chassis,
            args.target,
            SearchMode(args.search_mode),
            args.max_cycles,
            perception,
            on_cycle=on_cycle,
            debug_frontier=args.debug_frontier,
        )


def _run_hermes(args: argparse.Namespace, api_key: str) -> int:
    """组装 Hermes 本地直连或远程转发的 Adapter，共用导航、日志与运动规则。"""
    if args.base_only and not args.preflight_only:
        raise ValueError("--base-only 只用于 --preflight-only，不可启动导航")
    calibration_path = Path(args.camera_calibration)
    if (
        not args.preflight_only
        and args.camera_height_m is None
        and not calibration_path.is_file()
    ):
        raise ValueError(
            "缺少本套设备的相机外参：提供 --camera-calibration，"
            "或至少提供 --camera-height-m；D435i 不可沿用其他相机安装外参"
        )
    if not args.preflight_only and not args.target:
        raise ValueError("hermes 导航模式必须提供 --target")
    if not args.preflight_only and not args.enable_motion:
        raise ValueError("真机导航必须显式提供 --enable-motion")
    if not args.preflight_only and not args.debug_random_score and not api_key:
        raise ValueError(MISSING_VLM_CREDENTIAL)

    run_logger = _build_hermes_run_logger(args)
    return_code = 1
    error_message = None
    perception: Optional[SemanticPerception] = None
    try:
        on_cycle, on_motion_frame, on_vlm_interaction, on_motion_plan, on_semantic_event = (
            _build_visualization(args)
        )
        if not args.preflight_only:
            perception = _build_perception(
                args,
                api_key,
                on_vlm_interaction,
                on_semantic_event=_perception_event_callback(on_semantic_event, run_logger),
                object_config=_object_config(args),
            )
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
            position_tolerance_m=args.position_tolerance_m,
            yaw_tolerance_rad=math.radians(args.yaw_tolerance_deg),
            action_timeout_s=args.action_timeout_s,
            action_stall_timeout_s=args.action_stall_timeout_s,
            max_unknown_path_m=args.max_unknown_path_m,
            minimum_localization_quality=args.min_localization_quality,
        )
        action_progress = print if run_logger is None else _action_progress_callback(run_logger)
        with HermesAdapter(
            config,
            on_motion_frame=on_motion_frame,
            on_continuous_frame=(
                _motion_prefetch_callback(on_motion_frame, perception) if perception is not None else None),
            camera_factory=camera_factory(args),
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
                return_code = 0
            else:
                if perception is None:
                    raise RuntimeError("Hermes 导航缺少语义感知模块")
                _move_hermes_forward_on_start(chassis, args.startup_forward_m)
                return_code = run_navigation(
                    chassis,
                    args.target,
                    SearchMode(args.search_mode),
                    args.max_cycles,
                    perception,
                    on_cycle=on_cycle,
                    debug_frontier=args.debug_frontier,
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
        if perception is not None:
            perception.close()
        if run_logger is not None:
            run_logger.log_run_end(return_code, error_message)
            run_logger.close()
    return return_code


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


def _build_hermes_run_logger(args: argparse.Namespace) -> Optional[NavigationRunLogger]:
    """正式 Hermes 导航自动保存日志；预检不创建运行日志。"""
    if args.preflight_only:
        return None
    path = Path(args.run_log) if args.run_log else default_run_log_path("hermes")
    logger = NavigationRunLogger(path)
    logger.log_run_start(
        adapter_name="hermes",
        target_text=args.target,
        max_cycles=args.max_cycles,
        configuration={
            "config_file": str(args.config),
            "vlm_model": args.vlm_model,
            "vlm_api_format": args.vlm_api_format,
            "base_url": args.base_url,
            "camera_source": args.camera_source,
            "camera_endpoint": args.camera_endpoint if args.camera_source == "remote" else None,
            "camera_topic": args.camera_topic if args.camera_source == "remote" else None,
            "camera_timeout_s": args.camera_timeout_s if args.camera_source == "remote" else None,
            "search_mode": args.search_mode,
            "debug_random_score": args.debug_random_score,
            "debug_frontier": args.debug_frontier,
            "rerun_enabled": not args.no_rerun,
            "action_timeout_s": args.action_timeout_s,
            "action_stall_timeout_s": args.action_stall_timeout_s,
            "max_unknown_path_m": args.max_unknown_path_m,
            "startup_forward_m": args.startup_forward_m,
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


def _hermes_extrinsics_from_args(args: argparse.Namespace) -> CameraExtrinsics:
    """读取 Hermes 标定文件，并用显式命令行参数逐项覆盖。"""
    calibration_path = Path(args.camera_calibration)
    calibrated = (
        load_camera_extrinsics(calibration_path)
        if calibration_path.is_file()
        else None
    )
    return CameraExtrinsics(
        height_m=_extrinsic_value(args.camera_height_m, calibrated, "height_m"),
        forward_m=_extrinsic_value(args.camera_forward_m, calibrated, "forward_m"),
        left_m=_extrinsic_value(args.camera_left_m, calibrated, "left_m"),
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


def _hermes_health_text(health: HermesRobotHealth) -> str:
    if health.has_fatal:
        return "fatal"
    if health.has_error:
        return "error"
    if health.has_warning:
        return "warning"
    return "ok"


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


def _build_perception(
    args: argparse.Namespace,
    api_key: str,
    on_vlm_interaction=None,
    on_semantic_event=None,
    *, object_config: Optional[ObjectLocalizerConfig] = None,
) -> SemanticPerception:
    """所有 Adapter 与搜索模式共用同一条后台检测与评分链。"""
    analyzer = _build_analyzer(args, api_key, on_vlm_interaction)
    return SemanticPerception(analyzer, on_event=on_semantic_event, object_config=object_config)


def _object_config(args) -> Optional[ObjectLocalizerConfig]:
    if args.debug_random_score or args.search_mode != SearchMode.OBJECT.value:
        return None
    return ObjectLocalizerConfig(
        python_executable=args.object_python, class_text=args.object_class or "", device=args.object_device,
        yolo_model=args.object_yolo_model, sam2_checkpoint=args.object_sam_checkpoint,
        timeout_s=args.object_timeout_s,
    )


def _motion_prefetch_callback(on_motion_frame, perception: SemanticPerception):
    """把最新帧送往语义预采样，并旁路记录到 Rerun。"""
    visualization = _optional_callback(on_motion_frame, "运动帧可视化")

    def callback(frame) -> None:
        perception.observe_motion_frame(frame)
        if visualization is not None:
            visualization(frame)

    return callback


def _perception_event_callback(on_event, run_logger: Optional[NavigationRunLogger]):
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
        return None, None, None, None, None

    from .visualization import RerunVisualizer

    visualizer = RerunVisualizer(args.target, recording_path=args.rerun_save)
    return (
        visualizer.log_cycle,
        visualizer.log_motion_frame,
        visualizer.log_vlm_interaction,
        visualizer.log_motion_plan,
        visualizer.log_semantic_queue_event,
    )


__all__ = [
    "MISSING_VLM_CREDENTIAL",
    "run_entries",
    "run_navigation",
]
