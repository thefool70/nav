"""顶层单周期入口，串联底盘读取与导航算法。"""

from typing import Optional

from .adapters.chassis import ChassisInterface
from .adapters.perception import TargetObserver
from .core.models import (
    NavigationResult,
    NavigationStatus,
    SearchState,
    TargetSearchGoal,
)
from .core.navigator import navigate


def run_navigation_cycle(
    chassis: ChassisInterface,
    goal: TargetSearchGoal,
    state: Optional[SearchState] = None,
    observer: Optional[TargetObserver] = None,
) -> NavigationResult:
    """执行一个导航周期：读取底盘帧、按算法需要观察目标、发送命令。

    仅当 status 为 OK 且 command 非 None 时向底盘发送命令，随后返回结果；
    调用方从 result.state 取下一周期状态。不包含循环或频率假设。未提供
    observer 时仍可执行回退等纯地图步骤，扫描到视觉步骤时会显式暂停。
    """
    frame = chassis.read_frame()
    result = navigate(frame, goal, state)
    if (
        observer is not None
        and result.status is NavigationStatus.NOT_IMPLEMENTED
        and result.debug.stage in {"scan.observe", "target.observe"}
    ):
        result = navigate(frame, goal, state, observer.observe(frame, goal))
    if result.status is NavigationStatus.OK and result.command is not None:
        chassis.send_relative_pose(result.command)
    return result
