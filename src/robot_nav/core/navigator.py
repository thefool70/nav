"""导航算法主入口。当前仅搭建框架，不实现具体导航算法。"""

from .models import (
    NavigationDebug,
    NavigationFrame,
    NavigationGoal,
    NavigationResult,
    NavigationStatus,
)


def navigate(frame: NavigationFrame, goal: NavigationGoal) -> NavigationResult:
    """单周期导航主入口：输入一帧感知与目标，输出结果与相对位姿命令。

    当前未实现，固定返回 NOT_IMPLEMENTED、command=None，并在 debug 中说明。
    """
    return NavigationResult(
        status=NavigationStatus.NOT_IMPLEMENTED,
        command=None,
        debug=NavigationDebug(
            stage="navigator",
            message="导航算法尚未实现，当前仅为框架占位。",
            details={"received_timestamp_s": frame.timestamp_s},
        ),
    )
