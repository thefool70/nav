"""外部系统适配层。当前仅定义最薄的底盘接口协议，不包含厂商实现。"""

from typing import Protocol, Tuple, runtime_checkable

from ..core.models import NavigationFrame, ObstacleMap, Pose2D, RelativePoseCommand


class RecoverableMotionError(RuntimeError):
    """目标点被规划器拒绝或无法到达，算法可淘汰该候选后继续。"""


class MotionPathUnknownError(RecoverableMotionError):
    """路径经过算法未知区；确认停止后向核心传递被拒绝的世界坐标路径。"""

    def __init__(
        self,
        message: str,
        path_world_xy: Tuple[Tuple[float, float], ...],
    ) -> None:
        super().__init__(message)
        self.path_world_xy = tuple(path_world_xy)


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


@runtime_checkable
class KnownSpaceChassisInterface(Protocol):
    """可检查实际规划路径的底盘扩展，当前由 Hermes 实现。"""

    def send_relative_pose_in_known_space(
        self, command: RelativePoseCommand, obstacle_map: ObstacleMap,
        *, reference_pose: Pose2D,
    ) -> None:
        """用决策位姿还原世界目标，按选点地图检查路径；未知路径确认取消后抛异常。"""
        ...
