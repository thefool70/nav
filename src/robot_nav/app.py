"""顶层单周期入口，串联底盘读取与导航算法。"""

from typing import Callable, Optional

from .adapters.chassis import ChassisInterface
from .adapters.perception import TargetObserver
from .core.models import (
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    SearchState,
    TargetObservation,
    TargetSearchGoal,
)
from .core.navigator import navigate


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
        used_observation = observer.observe(frame, goal)
        result = navigate(frame, goal, state, used_observation)
    if on_cycle is not None:
        on_cycle(frame, used_observation, result)
    if result.status is NavigationStatus.OK and result.command is not None:
        chassis.send_relative_pose(result.command)
    return result
