"""视觉/VLM 目标感知边界，不属于底盘 Adapter。

本协议只描述"从一帧感知快照与语义目标描述得到目标观测"这一输入输出，
具体实现依赖视觉/VLM 模型，由感知侧负责，不在此猜测实现细节。
"""

from typing import Protocol

from ..core.models import NavigationFrame, TargetObservation, TargetSearchGoal


class TargetObserver(Protocol):
    """视觉/VLM 目标观测接口。

    ``VISIBLE`` 必须附带 ``bbox_norm``；``NOT_VISIBLE`` 可附带当前方向的
    ``direction_score``。Adapter 只归一化感知结果，不参与导航决策。
    """

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
    ) -> TargetObservation:
        """返回对 frame 中 goal 目标的观测结果。"""
        ...


__all__ = ["TargetObserver"]
