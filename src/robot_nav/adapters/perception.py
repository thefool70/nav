"""视觉/VLM 目标感知与 Frontier 批量评分边界。

它不属于底盘 Adapter；具体模型、图像标注和请求方式由感知侧负责。
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple

from ..core.models import (
    FrontierScoreRequest,
    NavigationFrame,
    TargetObservation,
    TargetSearchGoal,
)


@dataclass(frozen=True)
class ScanObservationContext:
    """当前 RGB 在本轮多视角扫描中的位置。"""

    index: int
    count: int


@dataclass(frozen=True)
class VlmInputImage:
    """实际发送给 VLM 的紧凑 RGB 图像。"""

    width_px: int
    height_px: int
    rgb_bytes: bytes


@dataclass(frozen=True)
class VlmInteraction:
    """一条可供 Rerun 完整复盘的 VLM 请求或回应事件。"""

    interaction_id: int
    phase: str
    task: str
    endpoint_url: str
    model: str
    api_format: str
    reasoning_effort: Optional[str]
    max_output_tokens: int
    prompt: str
    image: VlmInputImage
    assistant_text: str = ""
    response_json: str = ""
    parsed_result: str = ""
    bbox_norm: Optional[Tuple[float, float, float, float]] = None
    error: str = ""


class TargetObserver(Protocol):
    """单帧目标观测与一轮 Frontier 批量评分接口。"""

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        scan_context: Optional[ScanObservationContext] = None,
    ) -> TargetObservation:
        """返回对 frame 中 goal 目标的观测结果。"""
        ...

    def score_frontiers(
        self,
        request: FrontierScoreRequest,
        goal: TargetSearchGoal,
    ) -> Mapping[str, float]:
        """一次返回本轮全部 Frontier ID 到 0-1 分数的映射。"""
        ...


class TargetBoxObserver(TargetObserver, Protocol):
    """能够在已确认目标可见时重新框选目标的观察器。"""

    def rebox_visible_target(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
    ) -> TargetObservation:
        """跳过可见性判断，直接在同一帧中重新请求目标框。"""
        ...


__all__ = [
    "ScanObservationContext",
    "TargetBoxObserver",
    "TargetObserver",
    "VlmInputImage",
    "VlmInteraction",
]
