"""单图评分与视觉 grounding 的短提示词，以及模型输出的边界校验。"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from json_repair import repair_json

from .models import SearchMode, SemanticAnalysis


def build_semantic_analysis_prompt(target_text: str, search_mode: SearchMode) -> str:
    """每张原图独立评分；物体框使用 Qwen 的 0–1000 相对坐标，不要求输出编号。"""
    target = json.dumps(target_text, ensure_ascii=False)
    scoring = (
        f"Target: {target}. Score this view's promise of leading to the target, 0-1; "
        "0.5 if unclear. Not seeing it is not negative evidence. Ignore travel distance. "
    )
    if search_mode is SearchMode.SCENE:
        return scoring + (
            "Set found=true only if the camera is already inside the target scene; "
            "a sign or a view through its entrance is insufficient. "
            'Return JSON only: {"score":0.5,"found":false}.'
        )
    return scoring + (
        "Locate one clearly visible target object. Use bbox_2d=[xmin,ymin,xmax,ymax] "
        "with coordinates normalized to 0-1000; use null if absent. "
        'Return JSON only: {"score":0.5,"bbox_2d":null}.'
    )


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
    return SemanticAnalysis(found, score, bbox, detection_error, scoring_error)


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


__all__ = ["build_semantic_analysis_prompt", "parse_semantic_analysis_response"]
