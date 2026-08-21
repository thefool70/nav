"""外部系统适配层。当前仅定义最薄的底盘接口协议，不包含厂商实现。"""

from typing import Protocol

from ..core.models import NavigationFrame, RelativePoseCommand


class ChassisInterface(Protocol):
    """底盘最小接口：读取一帧感知，发送相对位姿控制命令。

    厂商原始协议、单位换算与坐标转换由具体实现负责，不在此猜测。
    """

    def read_frame(self) -> NavigationFrame:
        """读取一帧感知快照（深度、位姿、障碍图与可选图像）。"""
        ...

    def send_relative_pose(self, command: RelativePoseCommand) -> None:
        """向底盘发送相对当前位姿的控制命令。"""
        ...
