"""感知模块的单图模型边界；队列、快照与定位策略不依赖 HTTP 实现。"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol

from ..adapters.perception import VlmInputImage
from ..core.models import SemanticAnalysis, TargetSearchGoal


class SemanticAnalyzer(Protocol):
    """每次分析一张原图；found=None 表示失败，False 表示明确未检出。"""

    def analyze_view(
        self, image: VlmInputImage, goal: TargetSearchGoal,
        *, trace_context: Optional[Mapping[str, Any]] = None,
    ) -> SemanticAnalysis: ...


__all__ = ["SemanticAnalyzer"]
