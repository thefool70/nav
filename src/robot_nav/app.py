"""运行层：把底盘、语义感知与搜索核心连成完整的导航循环。

阅读顺序：run_navigation 总循环 → run_navigation_cycle 单周期 → 下方内部步骤。
单个周期的顺序固定为：

1. 读取一帧底盘数据。
2. 向感知模块取回已完成的分析结果（新增覆盖与目标线索）。
3. 推进搜索核心一次决策，得到显式请求的动作或外部能力请求。
4. 按请求向感知模块补充一次结果（扫描拍摄、Frontier 评分、物体定位）。
5. 记录决策与计时，再同步执行动作，把执行结果交回核心解释。
6. 总循环记录执行后的结果，判断退出条件。

本模块只负责编排：换点、遮蔽区域、跳过父节点、放弃目标线索都由搜索核心决定。
它不解析 ``debug`` 字符串来改变行为，只按 ``NavigationAction`` 的类型执行。
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from time import monotonic
from typing import Any, Callable, Mapping, Optional

from .adapters.chassis import (
    ChassisInterface,
    KnownSpaceChassisInterface,
    MotionPathUnknownError,
    MotionStalledError,
    RecoverableMotionError,
)
from .adapters.perception import ScanObservationContext
from .core.actions import action_command
from .core.perception_flow import capture_context, receive_perception, target_handling_active
from .core.frontier import FrameFrontierCache
from .core.models import (
    ActionKind,
    ActionConstraint,
    ActionPurpose,
    ActionOutcome,
    ActionExecutionResult,
    NavigationAction,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    SearchPhase,
    SearchMode,
    SearchState,
    TargetObservation,
    TargetSearchGoal,
)
from .core.navigator import (
    apply_execution_result,
    navigate,
)
from .core.timing import TimingSpans, measure_stage
from .perception import SemanticPerception
from .run_log import NavigationRunLogger
from .runtime_reporting import (
    record_decision, optional_callback, print_cycle, report_cycle_decision,
)

NavigationCycleCallback = Callable[
    [NavigationFrame, Optional[TargetObservation], NavigationResult],
    None,
]


def run_navigation(
    chassis: ChassisInterface,
    target_text: str,
    search_mode: SearchMode,
    max_cycles: int,
    perception: SemanticPerception,
    *,
    on_cycle=None,
    debug_frontier: bool,
    run_logger: Optional[NavigationRunLogger] = None,
) -> int:
    """重复执行环境无关的单周期入口，直到完成、失败或达到上限。"""

    goal = TargetSearchGoal(target_text, search_mode)
    state = None
    cycle_index = 0
    decision_cycles = 0
    on_cycle = optional_callback(on_cycle, "导航可视化")
    while decision_cycles < max_cycles:
        cycle_index += 1
        if run_logger is not None:
            run_logger.log_cycle_start(cycle_index)
        cycle_callback = partial(
            record_decision, on_cycle=on_cycle, debug_frontier=debug_frontier,
            run_logger=run_logger, cycle_index=cycle_index,
        )
        result = run_navigation_cycle(
            chassis,
            goal,
            state,
            perception,
            on_cycle=cycle_callback,
            on_timing=None if run_logger is None else run_logger.log_cycle_timing,
        )
        state = result.state
        if run_logger is not None:
            run_logger.log_cycle_result(cycle_index, result)
        print_cycle(cycle_index, result)

        if result.status is NavigationStatus.OK and state.phase is SearchPhase.WAITING_FOR_SEMANTICS:
            # 原地等后台结果不消耗决策额度，否则慢模型会提前耗尽 max_cycles。
            perception.wait_for_result()
            continue
        decision_cycles += 1

        if state.phase is SearchPhase.COMPLETE:
            return 0
        if state.phase is SearchPhase.STOPPED:
            return 3
        if result.status in {
            NavigationStatus.NEEDS_SCAN_CAPTURE,
            NavigationStatus.NEEDS_FRONTIER_SCORES,
            NavigationStatus.NEEDS_OBJECT_LOCALIZATION,
        }:
            # 一次周期只消费一组外部感知输入，状态机请求下一组输入时读取下一帧。
            continue
        if state.phase is SearchPhase.FAILED or result.status is not NavigationStatus.OK:
            return 1

    print(f"达到最大导航决策周期数 {max_cycles}（不含队列等待），搜索尚未结束。")
    return 1


def run_navigation_cycle(
    chassis: ChassisInterface,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    perception: Optional[SemanticPerception] = None,
    on_cycle: Optional[NavigationCycleCallback] = None,
    on_timing: Optional[Callable[[dict], None]] = None,
) -> NavigationResult:
    """执行一个完整导航周期，并在需要时调用感知与底盘接口。

    跨周期状态由 ``SearchState`` 显式传入和返回；后台任务与快照由感知模块管理。
    ``on_timing`` 在执行动作前接收本周期阶段计时，不包含运动时长。
    """
    started = monotonic()
    timings = [] if on_timing is not None else None
    with measure_stage(timings, "cycle.read_frame"):
        frame = chassis.read_frame()
    # 本周期补充感知后可能再次决策，复用同一帧的几何提取，避免重复计算。
    frontier_cache = FrameFrontierCache(frame)
    if timings is not None:
        timings.extend(frame.acquisition_timings)

    if perception is not None:
        perception.set_goal(goal)
    working_state = _prepare_state(frame, perception, state or SearchState())
    with measure_stage(timings, "cycle.navigate"):
        decision = navigate(frame, goal, working_state, timings=timings, frontier_cache=frontier_cache)
    # 核心只提出感知请求；运行层取得结果后，仍由核心决定下一步。
    with measure_stage(timings, "cycle.supply_perception"):
        decision = _supply_perception(frame, goal, decision, perception,
            timings=timings, frontier_cache=frontier_cache,
        )
    if perception is not None:
        _sync_perception(frame, decision.state, perception)
        decision = _merge_perception_diagnostics(decision, perception)

    report_cycle_decision(frame, decision, started, timings,
                          on_cycle=on_cycle, on_timing=on_timing)
    # 先记录将要执行的决策，再同步运动；可恢复失败在返回前交给核心解释。
    return _execute_action(chassis, decision, perception, frame)


def _prepare_state(
    frame: NavigationFrame,
    perception: Optional[SemanticPerception],
    state: SearchState,
) -> SearchState:
    """取回后台结果并同步感知计数；返回给核心作为本周期输入的显式状态。"""
    if perception is None:
        return state
    intake = perception.begin_cycle(frame)
    clue = perception.take_target_clue(busy=target_handling_active(state))
    pending, failed, pending_views = perception.pending_counts()
    working_state = receive_perception(state, intake.observed_views,
        pending=pending, failed=failed, pending_views=pending_views, clue=clue)
    _sync_perception(frame, working_state, perception)
    return working_state


def _supply_perception(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    decision: NavigationResult,
    perception: Optional[SemanticPerception],
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """按核心返回的显式请求，向感知模块补充一次结果并重新推进核心。"""
    if perception is None or decision.status in (
        NavigationStatus.INVALID_INPUT,
        NavigationStatus.MISSING_DATA,
        NavigationStatus.NO_SOLUTION,
    ):
        return decision

    if decision.status is NavigationStatus.NEEDS_OBJECT_LOCALIZATION:
        clue = decision.state.active_target_clue
        if clue is None:
            return decision
        with measure_stage(timings, "observer.localize_object"):
            localization = perception.localize_object(frame, goal, clue)
        return navigate(frame, goal, decision.state, object_localization=localization,
            timings=timings, frontier_cache=frontier_cache,
        )

    if decision.status is NavigationStatus.NEEDS_FRONTIER_SCORES:
        request = decision.frontier_score_request
        if request is None:
            return decision
        with measure_stage(timings, "observer.score_frontiers"):
            scores = perception.score_frontiers(request)
        return navigate(frame, goal, decision.state, frontier_scores=scores,
            timings=timings, frontier_cache=frontier_cache,
        )

    if decision.status is NavigationStatus.NEEDS_SCAN_CAPTURE:
        _sync_perception(frame, decision.state, perception)
        return _capture_scan_direction(frame, goal, decision, perception,
            timings=timings, frontier_cache=frontier_cache,
        )
    return decision


def _execute_action(
    chassis: ChassisInterface,
    decision: NavigationResult,
    perception: Optional[SemanticPerception],
    frame: NavigationFrame,
) -> NavigationResult:
    """同步执行动作并交回执行结果；期间按动作类型决定是否允许运动预采样。"""
    action = decision.action
    if decision.status is not NavigationStatus.OK or action is None:
        return decision

    if perception is not None:
        perception.set_motion_prefetch_enabled(_prefetch_allowed(action))
    try:
        _send(chassis, action, frame)
    except (MotionStalledError, MotionPathUnknownError, RecoverableMotionError) as exc:
        if isinstance(exc, MotionPathUnknownError):
            execution = ActionExecutionResult(ActionOutcome.PATH_UNKNOWN, str(exc),
                rejected_path_world_xy=exc.path_world_xy,
                unknown_length_m=exc.unknown_length_m, limit_m=exc.limit_m,
                total_path_length_m=exc.total_path_length_m)
        else:
            execution = ActionExecutionResult(
                ActionOutcome.STALLED if isinstance(exc, MotionStalledError)
                else ActionOutcome.INTERRUPTED, str(exc))
        recovered = apply_execution_result(decision, execution)
        if recovered is None:
            raise
        return recovered
    finally:
        if perception is not None:
            perception.set_motion_prefetch_enabled(False)
    return apply_execution_result(decision, ActionExecutionResult(ActionOutcome.SUCCEEDED))


def _sync_perception(frame, state, perception):
    """向采样线程发布冻结上下文，退出扫描时提交部分批次。"""
    if state.phase is not SearchPhase.SCANNING or target_handling_active(state):
        perception.flush_scan()
    perception.bind_frame(frame, capture_context(frame, state))
    perception.pause_for_target_handling(
        target_handling_active(state) or perception.has_target_clues(),
        reason="target_handling",
    )


def _capture_scan_direction(
    frame: NavigationFrame,
    goal: TargetSearchGoal,
    decision: NavigationResult,
    perception: SemanticPerception,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """采集当前扫描方向并推进核心登记结果；这是扫描的常规推进路径。"""
    from .core.scan_behavior import record_scanned_direction

    state = decision.state
    context = _scan_context(state)
    if context is None:
        return decision
    with measure_stage(timings, "observer.capture_scan"):
        perception.capture_scan_view(frame, context,
            timings=timings, frontier_cache=frontier_cache,
        )
    pending, failed, pending_views = perception.pending_counts()
    state = receive_perception(state, pending=pending, failed=failed, pending_views=pending_views)
    return record_scanned_direction(frame, goal, state, _current_scan_heading(state),
        timings=timings, frontier_cache=frontier_cache,
    )


def _scan_context(state: SearchState) -> Optional[ScanObservationContext]:
    """当前扫描方向在一轮扫描中的编号；不在扫描计划内时返回 None。"""
    count = len(state.scan_headings_world_rad)
    if count < 1 or not 0 <= state.next_scan_index < count:
        return None
    return ScanObservationContext(index=state.next_scan_index, count=count)


def _current_scan_heading(state: SearchState) -> float:
    return state.scan_headings_world_rad[state.next_scan_index]


def _merge_perception_diagnostics(
    decision: NavigationResult,
    perception: SemanticPerception,
) -> NavigationResult:
    """把观测器诊断并入日志详情；仅用于解释与记录，不参与决策。"""
    details: Mapping[str, Any] = perception.diagnostics()
    return replace(decision, debug=replace(decision.debug, details={**decision.debug.details, **details}))


def _prefetch_allowed(action: NavigationAction) -> bool:
    """只在探索与返回的平移动作期间做视觉预采样，目标接近不预采样。"""
    if action.action is not ActionKind.MOVE_TO_POSE:
        return False
    if action.purpose in (ActionPurpose.APPROACH, ActionPurpose.FALLBACK, ActionPurpose.REVISIT):
        return False
    return action.destination is not None


def _send(chassis: ChassisInterface, action: NavigationAction, frame: NavigationFrame) -> None:
    """按动作类型与约束选择已校验的已知空间移动或直接相对移动。"""
    command = action_command(action, frame.pose)
    if command is None:
        raise RecoverableMotionError("动作缺少相对移动量。")
    if action.constraint is ActionConstraint.REQUIRE_KNOWN_PATH:
        if not isinstance(chassis, KnownSpaceChassisInterface):
            raise RuntimeError("Adapter 不支持动作要求的未知路径检查，不能降级执行")
        path_map = (frame.navigation_map
                    if action.purpose is ActionPurpose.APPROACH and frame.navigation_map is not None
                    else frame.obstacle_map)
        # 用决策时的位姿解释相对命令，不能被发送前新读到的位姿改变目标。
        chassis.send_relative_pose_in_known_space(command, path_map, reference_pose=frame.pose)
    else:
        # 没有已知区约束的动作（如扫描转向）走普通同步执行接口。
        chassis.send_relative_pose(command)


__all__ = ["NavigationCycleCallback", "run_navigation", "run_navigation_cycle"]
