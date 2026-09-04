"""调试用随机观察器：不感知视觉内容，批量生成随机 Frontier 分数。

仅用于调试扫描、Frontier、移动和回退流程，不能识别或到达语义目标。
"""

from __future__ import annotations

import random
from typing import Mapping, Optional

from .perception import ScanObservationContext
from ..core.models import (
    FrontierScoreRequest,
    NavigationFrame,
    SceneAssessment,
    SceneAssessmentResult,
    TargetConfirmation,
    TargetConfirmationResult,
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

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        scan_context: Optional[ScanObservationContext] = None,
    ) -> TargetObservation:
        return TargetObservation(
            visibility=TargetVisibility.NOT_VISIBLE,
            reason=(
                "调试随机感知：不识别视觉内容，"
                "无法找到或到达语义目标。"
            ),
        )

    def score_frontiers(
        self,
        request: FrontierScoreRequest,
        goal: TargetSearchGoal,
    ) -> Mapping[str, float]:
        return {
            candidate.candidate_id: self._random.random()
            for candidate in request.candidates
        }

    def assess_scene(
        self,
        goal: TargetSearchGoal,
    ) -> SceneAssessmentResult:
        """随机评分模式不读取图像，不能判断目的场景。"""
        return SceneAssessmentResult(
            SceneAssessment.UNCERTAIN,
            "随机调试模式不执行目的场景判断。",
        )

    def confirm_target(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        observation: TargetObservation,
    ) -> TargetConfirmationResult:
        """随机调试模式不会产生目标候选，保留显式兜底。"""
        return TargetConfirmationResult(
            TargetConfirmation.UNCERTAIN,
            "随机调试模式不执行目标最终确认。",
        )


__all__ = ["RandomScoreTargetObserver"]
