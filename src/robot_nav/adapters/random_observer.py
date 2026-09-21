"""调试用随机观察器：不感知视觉内容，批量生成随机 Frontier 分数。

仅用于调试扫描、Frontier、移动和重新选点流程，不能识别或到达语义目标。
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional, Tuple

from .frontier_overlay import BufferedScanImage
from ..core.models import (
    FrontierCandidate,
    NavigationFrame,
    SemanticAnalysis,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)


class RandomScoreTargetObserver:
    """观测始终为 NOT_VISIBLE，一轮 Frontier 一次性生成随机分数。

    不读取 frame 的视觉内容，也不会返回 VISIBLE 或 bbox_norm；此模式
    只能驱动扫描与探索流程，无法识别或到达语义目标。
    """

    def __init__(self) -> None:
        self._random = random.Random()

    def analyze_views(
        self, images: Mapping[int, BufferedScanImage],
        candidates: Tuple[FrontierCandidate, ...], goal: TargetSearchGoal,
        *, trace_context: Optional[Mapping[str, Any]] = None,
    ) -> SemanticAnalysis:
        """走同一队列，但不调用模型；图像中永远不报告目标。"""
        return SemanticAnalysis((), frontier_scores={
            candidate.candidate_id: self._random.random() for candidate in candidates
        })

    def locate_object(self, frame: NavigationFrame, goal: TargetSearchGoal, *, context=None) -> TargetObservation:
        """随机模式不识别物体；正常运行不创建物体定位器。"""
        return TargetObservation(TargetVisibility.NOT_VISIBLE, source="random",
                                 reason="随机评分模式不执行物体定位。")


__all__ = ["RandomScoreTargetObserver"]
