"""单图分析契约：图像输入、交互记录、模型协议、提示词与结果解析。

具体模型请求在 adapters；评分与目标检测独立校验，失败不伪装成未检出。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Protocol

from json_repair import repair_json

from ..core.models import SemanticAnalysis, TargetSearchGoal, SearchMode, TargetObservation


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


class SemanticAnalyzer(Protocol):
    """每次分析一张原图；found=None 表示失败，False 表示明确未检出。"""

    def analyze_view(
        self, image: VlmInputImage, goal: TargetSearchGoal,
        *, trace_context: Optional[Mapping[str, Any]] = None,
    ) -> SemanticAnalysis: ...


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


def match_target(
    vlm: SemanticAnalysis, yolo: Tuple[TargetObservation, ...],
    thresholds: DetectionThresholds, search_mode: SearchMode,
) -> TargetMatch:
    """单路高分即可命中；中等分要求同帧的有效框重合。场景仅由 VLM 判断。"""
    confidence = vlm.confidence if vlm.found is not None else None
    best = max(yolo, key=lambda item: item.confidence, default=None)
    yolo_confidence = best.confidence if best is not None else 0.0
    if search_mode is SearchMode.SCENE:
        accepted = None if vlm.found is None else bool(vlm.found and confidence >= thresholds.vlm_high)
        return TargetMatch(accepted, source="vlm", vlm_confidence=confidence,
                           reason="vlm_high" if accepted else "scene_below_threshold")
    if vlm.found and vlm.bbox_norm is not None and confidence >= thresholds.vlm_high:
        return TargetMatch(True, vlm.bbox_norm, "vlm", confidence, yolo_confidence, reason="vlm_high")
    if best is not None and yolo_confidence >= thresholds.yolo_high:
        return TargetMatch(True, best.bbox_norm, "yoloe", confidence, yolo_confidence, reason="yolo_high")
    if vlm.found and vlm.bbox_norm is not None and confidence >= thresholds.vlm_joint:
        # 同图可有多个同类物体；不能只比较最高分框，否则会错配不同实例。
        matches = [(box_iou(vlm.bbox_norm, item.bbox_norm), item) for item in yolo
                   if item.confidence >= thresholds.yolo_joint]
        overlap, paired = max(matches, key=lambda pair: (pair[0], pair[1].confidence), default=(0.0, None))
        if paired is not None and overlap >= thresholds.box_iou:
            return TargetMatch(True, paired.bbox_norm, "vlm+yoloe", confidence,
                               paired.confidence, overlap, "joint")
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


def build_semantic_analysis_prompt(target_text: str, search_mode: SearchMode) -> str:
    """每张原图独立评分；物体框使用 Qwen 的 0–1000 相对坐标，不要求输出编号。"""
    target = json.dumps(target_text, ensure_ascii=False)
    scoring = (
        "score: 0-1 direction promise, not detection confidence. Default 0.5 when target "
        "is absent; raise for supporting context, lower only for contrary context. Ignore distance. "
    )
    if search_mode is SearchMode.SCENE:
        return f"Target: {target}. " + scoring + (
            "Set found=true only if the camera is already inside the target scene; "
            "a sign or a view through its entrance is insufficient. "
            'confidence: 0-1 certainty the camera is inside that scene. '
            'Return JSON only: {"score":0.5,"found":false,"confidence":0.0}.'
        )
    return (
        f'Target: {target}. Return JSON only: {{"score":0.5,"bbox_2d":null,"confidence":0.0}}. '
        "bbox_2d: best matching object's [xmin,ymin,xmax,ymax], normalized 0-1000; null if absent. "
        "confidence: 0-1 certainty this box is the target; lower for ambiguity, 0 if no box. "
        + scoring.rstrip()
    )


def semantic_analysis_schema(search_mode: SearchMode) -> Mapping[str, Any]:
    """Ollama 的输出结构约束；数值范围及框顺序仍由解析器校验。"""
    properties = {"score": {"type": "number", "minimum": 0, "maximum": 1}}
    properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    if search_mode is SearchMode.SCENE:
        properties["found"] = {"type": "boolean"}
    else:
        properties["bbox_2d"] = {"anyOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
             "minItems": 4, "maxItems": 4},
        ]}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def parse_semantic_analysis_response(text: str, search_mode: SearchMode) -> SemanticAnalysis:
    """独立校验方向分与检测；错误不伪装为无线索，有效框在此转换为 0–1。"""
    payload = _extract_json_mapping(text)
    raw_score = payload.get("score")
    score = None
    if type(raw_score) in (int, float) and math.isfinite(raw_score) and 0 <= raw_score <= 1:
        score = float(raw_score)
    scoring_error = "" if score is not None else "score 必须为 0–1 的有限数值"
    found, bbox, detection_error = None, None, ""
    if search_mode is SearchMode.SCENE:
        if type(payload.get("found")) is bool:
            found = payload["found"]
        else:
            detection_error = "场景判断缺少布尔值 found"
    elif "bbox_2d" not in payload:
        detection_error = "物体检测缺少 bbox_2d"
    elif payload["bbox_2d"] is None:
        found = False
    else:
        raw = payload["bbox_2d"]
        if (isinstance(raw, list) and len(raw) == 4
                and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1000 for x in raw)
                and raw[0] < raw[2] and raw[1] < raw[3]):
            bbox = tuple(x / 1000.0 for x in raw)
            found = True
        else:
            detection_error = "bbox_2d 必须为 0–1000 的有效 [xmin,ymin,xmax,ymax] 或 null"
    confidence = payload.get("confidence")
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        found, confidence = None, None
        detection_error = "; ".join(filter(None, (detection_error, "confidence 必须为 0–1 的有限数值")))
    else:
        confidence = float(confidence)
        if search_mode is SearchMode.OBJECT and found is False and confidence != 0:
            found, confidence = None, None
            detection_error = "没有目标框时 confidence 必须为 0"
    return SemanticAnalysis(found, score, bbox, detection_error, scoring_error, confidence=confidence)


def _extract_json_mapping(text: str) -> Mapping[str, Any]:
    """修复 JSON 语法，不补造业务字段；字段类型和含义由调用方校验。"""
    raw = text.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = repair_json(raw, return_objects=True, skip_json_loads=True)
    if not isinstance(payload, Mapping):
        raise ValueError("单图分析回答必须是 JSON 对象")
    return payload
