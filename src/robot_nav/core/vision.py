"""目标视觉提示词与模型回答解析。

本模块只处理文本，不调用模型或读写设备。模型回答统一转换为导航核心已经
定义的可见性、方向评分和归一化目标框。
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional, Tuple

from .models import TargetVisibility


def build_target_visibility_prompt(target_text: str) -> str:
    """构造完整 RGB 图上的目标可见性问题。"""
    target = _target_json(target_text)
    return (
        "你是机器人当前画面的目标存在性检测器。"
        f"目标是 {target}。只依据当前完整 RGB 图像判断目标是否可见；"
        "检查画面边缘、远处和部分遮挡区域，不要依据房间常识猜测。"
        "无法可靠判断时使用 uncertain。只返回 JSON，不要 Markdown："
        '{"visibility":"visible|not_visible|uncertain","reason":"简短依据"}'
    )


def build_search_direction_prompt(target_text: str) -> str:
    """构造目标不可见时的当前方向探索价值问题。"""
    target = _target_json(target_text)
    return (
        f"当前画面没有确认目标 {target}。评估沿相机当前朝向继续探索的价值，"
        "同时考虑前方是否可通行以及之后找到目标的可能性。地图仍负责最终的"
        "可达性和安全判断。只返回 JSON，不要 Markdown："
        '{"search_direction_score":0.0,"reason":"简短依据"}。'
        "search_direction_score 必须在 0 到 1 之间。"
    )


def build_target_grounding_prompt(target_text: str) -> str:
    """构造目标已确认可见时的边界框定位问题。"""
    target = _target_json(target_text)
    return (
        f"目标 {target} 已确认出现在当前完整 RGB 图像中。定位一个最匹配实例，"
        "边界框应紧贴该实例所有可见部分，不要包含大块背景。bbox_2d 使用"
        "[x_min,y_min,x_max,y_max]，每个坐标是 0 到 1000 的图像相对坐标。"
        "只返回 JSON，不要 Markdown："
        '{"bbox_2d":[100,100,900,900],"reason":"简短依据"}'
    )


def parse_target_visibility_response(
    text: str,
) -> Tuple[TargetVisibility, str]:
    """解析目标可见性回答；格式不合法时抛出 ValueError。"""
    payload = _extract_json_mapping(text, "目标可见性")
    raw_visibility = str(payload.get("visibility", "")).strip().lower()
    try:
        visibility = TargetVisibility(raw_visibility)
    except ValueError as exc:
        raise ValueError("目标可见性必须是 visible、not_visible 或 uncertain") from exc
    return visibility, _reason(payload)


def parse_search_direction_response(text: str) -> Tuple[float, str]:
    """解析 0 到 1 的当前方向探索评分。"""
    payload = _extract_json_mapping(text, "方向评分")
    score = _finite_float(payload.get("search_direction_score"))
    if score is None or not 0.0 <= score <= 1.0:
        raise ValueError("search_direction_score 必须是 0 到 1 的有限数")
    return score, _reason(payload)


def parse_target_grounding_response(
    text: str,
) -> Tuple[Tuple[float, float, float, float], str]:
    """解析千分制 bbox_2d，并转换为 0 到 1 的归一化边界框。"""
    payload = _extract_json_mapping(text, "目标框")
    raw_bbox = payload.get("bbox_2d")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise ValueError("bbox_2d 必须包含四个坐标")

    values = []
    for raw_value in raw_bbox:
        value = _finite_float(raw_value)
        if value is None:
            raise ValueError("bbox_2d 坐标必须是有限数")
        values.append(value)
    x_min, y_min, x_max, y_max = values
    if not all(0.0 <= value <= 1000.0 for value in (x_min, y_min, x_max, y_max)):
        raise ValueError("bbox_2d 坐标必须位于 0 到 1000")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bbox_2d 必须满足 x_min < x_max 且 y_min < y_max")

    return (
        x_min / 1000.0,
        y_min / 1000.0,
        x_max / 1000.0,
        y_max / 1000.0,
    ), _reason(payload)


def _target_json(target_text: str) -> str:
    target = str(target_text).strip()
    if not target:
        raise ValueError("目标文本不能为空")
    return json.dumps(target, ensure_ascii=False)


def _extract_json_mapping(text: str, result_name: str) -> Mapping[str, Any]:
    """从可能带少量额外文本的回答中读取第一个 JSON 对象。"""
    raw = str(text).strip()
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"{result_name}回答中没有 JSON 对象")
    try:
        payload, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{result_name}回答不是有效 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{result_name}回答必须是 JSON 对象")
    return payload


def _reason(payload: Mapping[str, Any]) -> str:
    return str(payload.get("reason", "")).strip()[:400]


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "build_search_direction_prompt",
    "build_target_grounding_prompt",
    "build_target_visibility_prompt",
    "parse_search_direction_response",
    "parse_target_grounding_response",
    "parse_target_visibility_response",
]
