"""感知模块的模型边界：只描述分析器需要满足的接口，不含线程与队列。

真实实现（VLM、YOLO-World、SAM2 进程客户端）在 ``adapters`` 中；本文件只定义
感知策略依赖的最小协议，使 ``semantic_queue`` 不直接耦合具体模型或 HTTP 细节。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Tuple, TYPE_CHECKING

from ..core.models import (
    FrontierCandidate,
    NavigationFrame,
    SemanticAnalysis,
    TargetObservation,
    TargetSearchGoal,
)

if TYPE_CHECKING:
    from ..adapters.frontier_overlay import BufferedScanImage

__all__ = ["SemanticAnalyzer"]


class SemanticAnalyzer(Protocol):
    """对整轮画面做联合目标检测与 Frontier 评分，并支持物体定位。

    实现必须阻塞直到本批全部分析完成，并返回 ``SemanticAnalysis``；
    ``target_view_ids`` 为空元组表示本批未检测到目标，为 None 表示检测失败。
    """

    def analyze_views(
        self,
        images: Mapping[int, BufferedScanImage],
        candidates: Tuple[FrontierCandidate, ...],
        goal: TargetSearchGoal,
        *,
        trace_context: Optional[Mapping[str, Any]] = None,
    ) -> SemanticAnalysis: ...

    def locate_object(
        self, frame: NavigationFrame, goal: TargetSearchGoal,
        *, context: Optional[Mapping[str, Any]] = None,
    ) -> TargetObservation: ...
