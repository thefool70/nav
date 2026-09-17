"""顶层单周期入口，串联底盘读取与导航算法。"""

from dataclasses import replace
from typing import Callable, Optional, Tuple

from .adapters.chassis import (
    ChassisInterface,
    KnownSpaceChassisInterface,
    MotionInterruptedError,
    MotionPathUnknownError,
    MotionStalledError,
    RecoverableMotionError,
)
from .adapters.perception import (
    ContinuousTargetObserver,
    ScanObservationContext,
    TargetObserver,
)
from .adapters.queued_semantics import QueuedSemanticObserver
from .core.models import (
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    SearchPhase,
    SearchState,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from .core.navigator import (
    continue_after_motion_stall,
    continue_after_target_detection,
    navigate,
    recover_from_motion_failure,
)


NavigationCycleCallback = Callable[
    [NavigationFrame, Optional[TargetObservation], NavigationResult],
    None,
]


def run_navigation_cycle(
    chassis: ChassisInterface,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    observer: Optional[TargetObserver] = None,
    on_cycle: Optional[NavigationCycleCallback] = None,
) -> NavigationResult:
    """执行一个完整导航周期，并在需要时调用感知与底盘接口。

    本函数只负责编排一次“读取 → 决策 → 感知补充 → 执行动作”；核心跨周期
    状态由 ``SearchState`` 显式传入和返回，后台任务与缓存由观察器管理。
    """
    frame = chassis.read_frame()
    if isinstance(observer, QueuedSemanticObserver):
        state = observer.prepare_cycle(frame, state or SearchState())
        clue = observer.take_target_clue(state)
        if clue is not None:
            state = replace(state, active_target_clue=clue)
        state = _observer_state(observer, frame, state)
    result = navigate(frame, goal, state)
    result = _apply_object_localization(frame, goal, result, observer)
    result, observation = _apply_target_observation(
        frame,
        goal,
        state,
        result,
        observer,
    )
    result = _apply_scene_assessment(result, frame, goal, observer)
    result = _apply_target_confirmation(
        result,
        frame,
        goal,
        observation,
        observer,
    )
    result = _apply_frontier_scores(result, frame, goal, observer)
    if isinstance(observer, QueuedSemanticObserver):
        result = replace(
            result, state=_observer_state(observer, frame, result.state),
            debug=replace(result.debug, details={**result.debug.details, **observer.diagnostics()}),
        )

    if on_cycle is not None:
        on_cycle(frame, observation, result)
    return _execute_command(chassis, result, observer, frame)


def _apply_target_observation(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState],
    result: NavigationResult,
    observer: Optional[TargetObserver],
) -> Tuple[NavigationResult, Optional[TargetObservation]]:
    """按状态机需要读取当前视觉结果；持续观察器可抢占普通决策。"""
    if (
        observer is None
        or result.state.phase in (SearchPhase.COMPLETE, SearchPhase.STOPPED)
        or (state is not None and state.active_target_clue is not None)
    ):
        return result, None

    observation_requested = result.status in {
        NavigationStatus.NEEDS_OBSERVATION,
        NavigationStatus.NEEDS_TARGET_CONFIRMATION,
    }
    if not observation_requested and not isinstance(
        observer,
        ContinuousTargetObserver,
    ):
        return result, None

    _observer_state(observer, frame, result.state)
    observation = observer.observe(
        frame,
        goal,
        _scan_observation_context(result),
    )
    # 显式视觉请求应继续 result.state；持续检测旁路普通命令时则从周期入口
    # state 重新决策，避免在命令执行前误把“命令后的状态”当作当前位置状态。
    decision_state = result.state if observation_requested else state
    if isinstance(observer, QueuedSemanticObserver):
        decision_state = _observer_state(observer, frame, decision_state or SearchState())
    return navigate(frame, goal, decision_state, observation), observation


def _apply_scene_assessment(
    result: NavigationResult,
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    observer: Optional[TargetObserver],
) -> NavigationResult:
    """保留同步观察器的场景判断；异步线索返回不进入此步骤。"""
    if (
        observer is None
        or result.status is not NavigationStatus.NEEDS_SCENE_ASSESSMENT
    ):
        return result
    current_state = _observer_state(observer, frame, result.state)
    assessment = observer.assess_scene(goal)
    return navigate(
        frame,
        goal,
        current_state,
        scene_assessment=assessment,
    )


def _apply_object_localization(frame, goal, result, observer) -> NavigationResult:
    """按请求读取历史 RGB-D；本周期只消费一次定位结果。"""
    if result.status is not NavigationStatus.NEEDS_OBJECT_LOCALIZATION:
        return result
    if not isinstance(observer, QueuedSemanticObserver):
        return result
    state = _observer_state(observer, frame, result.state)
    localization = observer.localize_object(frame, goal, state)
    return navigate(frame, goal, state, object_localization=localization)


def _apply_target_confirmation(
    result: NavigationResult,
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    observation: Optional[TargetObservation],
    observer: Optional[TargetObserver],
) -> NavigationResult:
    """候选接近后调用一次 VLM 最终确认。"""
    if (
        observer is None
        or result.status is not NavigationStatus.NEEDS_TARGET_CONFIRMATION
    ):
        return result
    confirmation_observation = observation or TargetObservation(
        visibility=TargetVisibility.UNCERTAIN,
        reason="最终确认缺少本地检测结果。",
    )
    current_state = _observer_state(observer, frame, result.state)
    confirmation = observer.confirm_target(
        frame,
        goal,
        confirmation_observation,
    )
    return navigate(
        frame,
        goal,
        current_state,
        observation=observation,
        target_confirmation=confirmation,
    )


def _apply_frontier_scores(
    result: NavigationResult,
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    observer: Optional[TargetObserver],
) -> NavigationResult:
    """多个 Frontier 待排序时，一次取得整批语义分数。"""
    request = result.frontier_score_request
    if (
        observer is None
        or result.status is not NavigationStatus.NEEDS_FRONTIER_SCORES
        or request is None
    ):
        return result
    current_state = _observer_state(observer, frame, result.state)
    scores = observer.score_frontiers(request, goal)
    return navigate(
        frame,
        goal,
        current_state,
        frontier_scores=scores,
    )


def _execute_command(
    chassis: ChassisInterface,
    result: NavigationResult,
    observer: Optional[TargetObserver],
    frame: NavigationFrame,
) -> NavigationResult:
    """等待整条目标位置命令结束，再返回下一周期状态；期间不重新选择 Frontier。"""
    if result.status is not NavigationStatus.OK or result.command is None:
        return result

    continuous_observer = (
        observer
        if isinstance(observer, ContinuousTargetObserver)
        else None
    )
    stage = result.debug.details.get("next_stage", result.debug.stage)
    if continuous_observer is not None:
        continuous_observer.set_motion_interrupt_enabled(
            stage not in ("target.approach", "target.revisit", "target.revisit_turn")
        )
    if isinstance(observer, QueuedSemanticObserver):
        observer.set_motion_prefetch_enabled(
            stage in ("explore.select", "backtrack.return", "backtrack.resume")
            and (result.command.forward_m != 0.0 or result.command.left_m != 0.0)
        )
    try:
        if stage in ("explore.select", "backtrack.return", "backtrack.resume", "target.revisit", "object.fallback_return", "object.approach") and isinstance(
            chassis, KnownSpaceChassisInterface,
        ):
            # 物体停靠与选点使用同一份完整地图；探索和返回仍约束在 FOV 缓存图中。
            path_map = (
                frame.navigation_map
                if stage == "object.approach" and frame.navigation_map is not None
                else frame.obstacle_map
            )
            chassis.send_relative_pose_in_known_space(
                result.command, path_map, reference_pose=frame.pose,
            )
        else:
            chassis.send_relative_pose(result.command)
    except MotionInterruptedError as exc:
        continued = continue_after_target_detection(result, str(exc))
        if continued is None:
            raise
        return continued
    except MotionStalledError as exc:
        continued = continue_after_motion_stall(result, str(exc))
        if continued is None:
            raise
        return continued
    except MotionPathUnknownError as exc:
        recovered = recover_from_motion_failure(
            result, str(exc), rejected_path_world_xy=exc.path_world_xy,
        )
        if recovered is None:
            raise
        return replace(recovered, debug=replace(recovered.debug, details={
            **recovered.debug.details,
            "unknown_path_length_m": exc.unknown_length_m,
            "unknown_path_limit_m": exc.limit_m,
            "checked_path_length_m": exc.total_path_length_m,
        }))
    except RecoverableMotionError as exc:
        recovered = recover_from_motion_failure(result, str(exc))
        if recovered is None:
            raise
        return recovered
    finally:
        if isinstance(observer, QueuedSemanticObserver):
            observer.set_motion_prefetch_enabled(False)
        if continuous_observer is not None:
            continuous_observer.set_motion_interrupt_enabled(False)
    return result


def _observer_state(observer, frame: NavigationFrame, state: SearchState) -> SearchState:
    """入队后刷新工作计数，防止同周期误将尚未检查的画面视为已耗尽。"""
    if isinstance(observer, QueuedSemanticObserver):
        state = observer.sync_state(state)
        observer.bind_context(frame, state)
    return state


def _scan_observation_context(
    result: NavigationResult,
) -> Optional[ScanObservationContext]:
    """仅扫描观测帧需要编号，供整轮场景判断和 Frontier 评分。"""
    stage = result.debug.details.get("next_stage", result.debug.stage)
    if stage != "scan.observe":
        return None
    count = result.debug.details.get("scan_heading_count")
    if not isinstance(count, int):
        return None
    index = len(result.state.scan_evidence)
    if count < 1 or not 0 <= index < count:
        return None
    return ScanObservationContext(index=index, count=count)
