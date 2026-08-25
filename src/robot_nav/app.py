"""顶层单周期入口，串联底盘读取与导航算法。"""

from typing import Callable, Optional

from .adapters.chassis import (
    ChassisInterface,
    MotionStalledError,
    RecoverableMotionError,
)
from .adapters.perception import ScanObservationContext, TargetObserver
from .core.models import (
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    SearchState,
    TargetObservation,
    TargetSearchGoal,
)
from .core.navigator import (
    continue_after_motion_stall,
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
    """执行一个导航周期：读取底盘帧、按算法需要观察目标、发送命令。

    仅当 status 为 OK 且 command 非 None 时向底盘发送命令，随后返回结果；
    调用方从 result.state 取下一周期状态。不包含循环或频率假设。未提供
    observer 时仍可执行回退等纯地图步骤，扫描到视觉步骤时会显式暂停。
    on_cycle 若提供，在发送命令前以 (frame, 实际使用的 observation, result)
    调用，供可视化或日志记录使用，不参与算法决策。
    """
    frame = chassis.read_frame()
    result = navigate(frame, goal, state)
    used_observation: Optional[TargetObservation] = None
    if (
        observer is not None
        and result.status is NavigationStatus.NEEDS_OBSERVATION
    ):
        used_observation = observer.observe(
            frame,
            goal,
            _scan_observation_context(result),
        )
        result = navigate(frame, goal, state, used_observation)
    if (
        observer is not None
        and result.status is NavigationStatus.NEEDS_FRONTIER_SCORES
        and result.frontier_score_request is not None
    ):
        frontier_scores = observer.score_frontiers(
            result.frontier_score_request,
            goal,
        )
        result = navigate(
            frame,
            goal,
            result.state,
            frontier_scores=frontier_scores,
        )
    if on_cycle is not None:
        on_cycle(frame, used_observation, result)
    if result.status is NavigationStatus.OK and result.command is not None:
        try:
            chassis.send_relative_pose(result.command)
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
    return result


def _scan_observation_context(
    result: NavigationResult,
) -> Optional[ScanObservationContext]:
    """仅扫描观测帧需要被保留，供本轮 Frontier 批量评分。"""
    if result.debug.stage != "scan.observe":
        return None
    details = result.debug.details
    count = details.get("scan_heading_count")
    if not isinstance(count, int):
        return None
    # 动态计划可能删除尚未执行的朝向；图片按实际完成次数连续编号。
    index = len(result.state.scan_evidence)
    if count < 1 or not 0 <= index < count:
        return None
    return ScanObservationContext(index=index, count=count)
