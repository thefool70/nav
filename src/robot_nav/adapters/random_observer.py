"""调试用单图随机评分，不检测目标，也不能完成语义搜索。"""

import random

from ..core.models import SemanticAnalysis


class RandomScoreTargetObserver:
    def __init__(self) -> None:
        self._random = random.Random()

    def analyze_view(self, image, goal, *, trace_context=None) -> SemanticAnalysis:
        """同张图覆盖的前沿共用一个随机分，与正式单图评分的数据流一致。"""
        return SemanticAnalysis(False, image_score=self._random.random())


__all__ = ["RandomScoreTargetObserver"]
