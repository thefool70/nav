"""视觉/VLM 语义感知与 Frontier 批量评分边界。

它不属于底盘 Adapter；具体模型、图像标注和请求方式由感知侧负责。
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Tuple, runtime_checkable

from ..core.models import (
    FrontierScoreRequest,
    NavigationFrame,
    SceneAssessmentResult,
    TargetConfirmationResult,
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


@dataclass(frozen=True)
class LocalPerceptionEvent:
    """一次 YOLO-World + SAM2 推理摘要，不复制 RGB-D 数据。"""

    sequence_index: int
    frame_timestamp_s: float
    observation: TargetObservation
    inference_s: float
    candidate_count: int


class TargetObserver(Protocol):
    """物体/场景观测、Frontier 批量评分与最终确认接口。"""

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

    def assess_scene(
        self,
        goal: TargetSearchGoal,
    ) -> SceneAssessmentResult:
        """根据本轮缓存的扫描图判断是否已经位于目的场景。"""
        ...

    def confirm_target(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        observation: TargetObservation,
    ) -> TargetConfirmationResult:
        """机器人接近候选后，最终确认它是否确为目标。"""
        ...


@runtime_checkable
class ContinuousTargetObserver(TargetObserver, Protocol):
    """运动期间可接收最新帧并请求中断当前动作的本地观察器。"""

    def submit_motion_frame(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
    ) -> None:
        """非阻塞提交运动帧；尚未处理的旧运动帧可以被覆盖。"""
        ...

    def set_motion_interrupt_enabled(self, enabled: bool) -> None:
        """仅在非目标接近动作中允许本地检测中断当前运动。"""
        ...

    def should_interrupt_motion(self) -> bool:
        """有启用后产生的新目标检测时返回 True。"""
        ...

    def close(self) -> None:
        """停止后台推理线程并释放模型。"""
        ...


__all__ = [
    "ContinuousTargetObserver",
    "LocalPerceptionEvent",
    "ScanObservationContext",
    "TargetObserver",
    "VlmInputImage",
    "VlmInteraction",
]
