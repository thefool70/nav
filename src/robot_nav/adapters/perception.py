"""视觉/VLM 语义感知与 Frontier 批量评分边界。

它不属于底盘 Adapter；具体模型、图像标注和请求方式由感知侧负责。
导航流程统一使用异步视觉队列，因此本文件只保留队列实现所需的采集上下文与日志事件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


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
    context: Mapping[str, Any] = field(default_factory=dict)
    started_monotonic_s: float = 0.0
    elapsed_s: Optional[float] = None


__all__ = [
    "ScanObservationContext",
    "VlmInputImage",
    "VlmInteraction",
]
