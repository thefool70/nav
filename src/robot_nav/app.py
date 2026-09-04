"""顶层单周期入口，串联底盘读取与导航算法。"""

from typing import Callable, Optional, Tuple

from .adapters.chassis import (
    ChassisInterface,
    MotionInterruptedError,
    MotionStalledError,
    RecoverableMotionError,
)
from .adapters.perception import (
    ContinuousTargetObserver,
    ScanObservationContext,
    TargetObserver,
)
from .core.models import (
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
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

    本函数只负责编排一次“读取 → 决策 → 感知补充 → 执行动作”；跨周期状态
    仍全部由 ``SearchState`` 显式传入和返回。
    """
    frame = chassis.read_frame()
    result = navigate(frame, goal, state)
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

    if on_cycle is not None:
        on_cycle(frame, observation, result)
    return _execute_command(chassis, result, observer)


def _apply_target_observation(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    state: Optional[SearchState],
    result: NavigationResult,
    observer: Optional[TargetObserver],
) -> Tuple[NavigationResult, Optional[TargetObservation]]:
    """按状态机需要读取当前视觉结果；持续观察器可抢占普通决策。"""
    if observer is None:
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

    observation = observer.observe(
        frame,
        goal,
        _scan_observation_context(result),
    )
    # 显式视觉请求应继续 result.state；持续检测旁路普通命令时则从周期入口
    # state 重新决策，避免在命令执行前误把“命令后的状态”当作当前位置状态。
    decision_state = result.state if observation_requested else state
    return navigate(frame, goal, decision_state, observation), observation


def _apply_scene_assessment(
    result: NavigationResult,
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    observer: Optional[TargetObserver],
) -> NavigationResult:
    """整轮扫描完成时，用已缓存的多视角图片判断目的场景。"""
    if (
        observer is None
        or result.status is not NavigationStatus.NEEDS_SCENE_ASSESSMENT
    ):
        return result
    assessment = observer.assess_scene(goal)
    return navigate(
        frame,
        goal,
        result.state,
        scene_assessment=assessment,
    )


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
    confirmation = observer.confirm_target(
        frame,
        goal,
        confirmation_observation,
    )
    return navigate(
        frame,
        goal,
        result.state,
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
    scores = observer.score_frontiers(request, goal)
    return navigate(
        frame,
        goal,
        result.state,
        frontier_scores=scores,
    )


def _execute_command(
    chassis: ChassisInterface,
    result: NavigationResult,
    observer: Optional[TargetObserver],
) -> NavigationResult:
    """同步发送有效命令，并把可恢复运动结果映射回导航状态。"""
    if result.status is not NavigationStatus.OK or result.command is None:
        return result

    continuous_observer = (
        observer
        if isinstance(observer, ContinuousTargetObserver)
        else None
    )
    if continuous_observer is not None:
        continuous_observer.set_motion_interrupt_enabled(
            result.debug.stage != "target.approach"
        )
    try:
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
    except RecoverableMotionError as exc:
        recovered = recover_from_motion_failure(result, str(exc))
        if recovered is None:
            raise
        return recovered
    finally:
        if continuous_observer is not None:
            continuous_observer.set_motion_interrupt_enabled(False)
    return result


def _scan_observation_context(
    result: NavigationResult,
) -> Optional[ScanObservationContext]:
    """仅扫描观测帧需要编号，供整轮场景判断和 Frontier 评分。"""
    if result.debug.stage != "scan.observe":
        return None
    count = result.debug.details.get("scan_heading_count")
    if not isinstance(count, int):
        return None
    index = len(result.state.scan_evidence)
    if count < 1 or not 0 <= index < count:
        return None
    return ScanObservationContext(index=index, count=count)
