"""顶层单周期入口，串联底盘读取与导航算法。"""

from .adapters.chassis import ChassisInterface
from .core.models import NavigationResult, NavigationStatus, SearchState, TargetSearchGoal
from .core.navigator import navigate


def run_navigation_cycle(
    chassis: ChassisInterface,
    goal: TargetSearchGoal,
    state: SearchState | None = None,
) -> NavigationResult:
    """执行一个导航周期：读取一帧，调用 navigate。

    仅当 status 为 OK 且 command 非 None 时向底盘发送命令，随后返回结果；
    调用方从 result.state 取下一周期状态。不包含循环或频率假设。
    """
    frame = chassis.read_frame()
    result = navigate(frame, goal, state)
    if result.status is NavigationStatus.OK and result.command is not None:
        chassis.send_relative_pose(result.command)
    return result
