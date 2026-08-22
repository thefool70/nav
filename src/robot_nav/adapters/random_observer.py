"""调试用随机观察器：不感知视觉内容，只返回随机方向评分。

仅用于调试扫描、Frontier、移动和回退流程，不能识别或到达语义目标。
"""

from __future__ import annotations

import random

from ..core.models import (
    NavigationFrame,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)


class RandomScoreTargetObserver:
    """每次观测都返回 NOT_VISIBLE 和 0.0-1.0 均匀随机方向评分的观察器。

    不读取 frame 的视觉内容，也不会返回 VISIBLE 或 bbox_norm；此模式
    只能驱动扫描与探索流程，无法识别或到达语义目标。
    """

    def __init__(self) -> None:
        self._random = random.Random()

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
    ) -> TargetObservation:
        return TargetObservation(
            visibility=TargetVisibility.NOT_VISIBLE,
            direction_score=self._random.random(),
            reason=(
                "调试随机感知：不识别视觉内容，仅生成随机方向评分，"
                "无法找到或到达语义目标。"
            ),
        )


__all__ = ["RandomScoreTargetObserver"]
