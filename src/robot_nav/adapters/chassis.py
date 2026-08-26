"""外部系统适配层。当前仅定义最薄的底盘接口协议，不包含厂商实现。"""

from typing import Protocol

from ..core.models import NavigationFrame, RelativePoseCommand


class RecoverableMotionError(RuntimeError):
    """目标点被规划器拒绝或无法到达，算法可淘汰该候选后继续。"""


class MotionStalledError(RuntimeError):
    """移动已停止但当前位置仍可用，算法应从下一帧继续探索。"""


class MotionInterruptedError(RuntimeError):
    """后台感知发现目标，请停止当前动作并从实际位置重新决策。"""


class ChassisInterface(Protocol):
    """底盘最小接口：读取一帧感知，发送相对位姿控制命令。

    厂商原始协议、单位换算与坐标转换由具体实现负责，不在此猜测。当前调用
    模式是同步的：发送函数返回后，下一次 ``read_frame`` 必须能反映本次命令
    已完成或已明确失败。
    """

    def read_frame(self) -> NavigationFrame:
        """读取一帧感知快照（深度、位姿、障碍图与可选图像）。"""
        ...

    def send_relative_pose(self, command: RelativePoseCommand) -> None:
        """同步执行命令；可恢复的不可达或停滞使用对应显式异常。"""
        ...
