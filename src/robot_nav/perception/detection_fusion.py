"""同帧 YOLOE 与 VLM 检测融合：置信度门槛、类别匹配与目标框重叠。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from robot_nav.interfaces.types import SemanticAnalysis, TargetObservation


@dataclass(frozen=True)
class DetectionThresholds:
    """两种分数分别设阈值；联合判定要求同一张图且框 IoU 达标，不平均置信度。"""

    vlm_high: float = 0.90
    yolo_high: float = 0.70
    vlm_joint: float = 0.60
    yolo_joint: float = 0.30
    box_iou: float = 0.30

    def __post_init__(self):
        for low, high in ((self.vlm_joint, self.vlm_high), (self.yolo_joint, self.yolo_high)):
            if not 0 < low < high <= 1:
                raise ValueError("检测阈值须满足 0 < 联合门槛 < 单路高置信度门槛 <= 1")
        if not 0 < self.box_iou <= 1:
            raise ValueError("检测框 IoU 门槛须在 (0, 1] 内")


@dataclass(frozen=True)
class TargetMatch:
    """同帧两路检测的判定依据；None 表示未得到完整结论，False 表示未满足门槛。"""

    found: Optional[bool]
    bbox_norm: Optional[Tuple[float, float, float, float]] = None
    source: str = ""
    vlm_confidence: Optional[float] = None
    yolo_confidence: Optional[float] = None
    box_iou: float = 0.0
    reason: str = ""
    class_text: str = ""


def match_target(
    vlm: SemanticAnalysis, yolo: Tuple[TargetObservation, ...],
    thresholds: DetectionThresholds,
) -> TargetMatch:
    """单路高分即可命中；中等分要求同帧、同类别且有效框重合。"""
    confidence = vlm.confidence if vlm.found is not None else None
    best = max(yolo, key=lambda item: item.confidence, default=None)
    yolo_confidence = best.confidence if best is not None else 0.0
    if vlm.found and vlm.bbox_norm is not None and confidence >= thresholds.vlm_high:
        return TargetMatch(True, vlm.bbox_norm, "vlm", confidence, yolo_confidence,
                           reason="vlm_high", class_text=vlm.class_text)
    if best is not None and yolo_confidence >= thresholds.yolo_high:
        return TargetMatch(True, best.bbox_norm, "yoloe", confidence, yolo_confidence,
                           reason="yolo_high", class_text=best.class_text)
    if vlm.found and vlm.bbox_norm is not None and confidence >= thresholds.vlm_joint:
        # 同图可有多个同类物体；不能只比较最高分框，否则会错配不同实例。
        matches = [(box_iou(vlm.bbox_norm, item.bbox_norm), item) for item in yolo
                   if item.confidence >= thresholds.yolo_joint and item.class_text == vlm.class_text]
        overlap, paired = max(matches, key=lambda pair: (pair[0], pair[1].confidence), default=(0.0, None))
        if paired is not None and overlap >= thresholds.box_iou:
            return TargetMatch(True, paired.bbox_norm, "vlm+yoloe", confidence,
                               paired.confidence, overlap, "joint", paired.class_text)
    return TargetMatch(None if vlm.found is None else False,
                       vlm_confidence=confidence, yolo_confidence=yolo_confidence,
                       reason="below_threshold_or_box_mismatch")


def box_iou(first, second) -> float:
    """同一图像中的两个有效归一化 xyxy 框的交并比。"""
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1]))
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)
