"""目标视觉和 Frontier 批量评分的提示词与回答解析。

本模块只处理文本，不调用模型或读写设备。模型回答统一转换为导航核心已经
定义的可见性、Frontier 分数和归一化目标框。
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

from .models import (
    SceneAssessment,
    SemanticAnalysis,
    SearchMode,
    TargetConfirmation,
    TargetVisibility,
)


def build_semantic_analysis_prompt(
    target_text: str,
    marker_labels: Sequence[str],
    view_ids: Sequence[int],
    search_mode: SearchMode,
) -> str:
    """一次检查所有画面并评分可见候选；画面编号和候选编号使用不同前缀。"""
    target = _target_json(target_text)
    detection = (
        "List views whose capture positions are already within the destination scene. "
        "Use the surrounding layout and fixtures as evidence. Seeing the scene through "
        "an entrance, a sign, or a related object alone does not mean arrival. "
        "A destination sign outside an entrance can support a high exploration score "
        "without placing that view in the target list."
        if search_mode is SearchMode.SCENE else
        "List views where the described object is visible. "
        "Inspect all views, including edges and partially occluded objects. "
        "Require direct visual evidence of the object; "
        "room type alone does not prove presence."
    )
    template = {
        "target": {"view_ids": []},
        "scores": {label: 0.5 for label in marker_labels},
    }
    arrival_behavior = (
        "The robot will return to a listed capture position and heading and finish "
        "navigation on arrival, without another visual check. "
        if search_mode is SearchMode.SCENE else
        "The robot will first localize a listed object using the saved RGB-D, then "
        "approach from its current position. It tries the remaining saved views when "
        "localization fails. Only if none can be localized will it return to a capture "
        "pose and stop. List only objects with direct visual evidence. "
    )
    return (
        f"Target: {target}.\n"
        f"Inputs: captured RGB views {list(view_ids)}, identified by V headers. "
        "Treat each view at its capture position, not at the robot's current position; "
        "do not infer a trajectory from tile order. Each F footer label is connected "
        "by a thin line to a small pink ground-plane Frontier anchor in its own view. "
        "Use that anchor's surroundings and the space beyond it; labels and lines "
        "are overlays, not scene evidence.\n"
        f"Task 1 - target detection: {detection} Check every view, including those without F labels. "
        "Return all matching integer view IDs in target.view_ids, ordered from strongest "
        "to weakest evidence, without duplicates. Do not keep only the best view. "
        "Return [] when no view meets the target condition. " + arrival_behavior + "\n"
        "Task 2 - exploration scoring: " + _frontier_scoring_rules() + "\n"
        "Output: return every listed F exactly once; with no F labels return empty scores. "
        "Return JSON only, without explanations or Markdown: "
        + json.dumps(template, ensure_ascii=False)
    )


def _frontier_scoring_rules() -> str:
    """缺少语义线索给中性分；模型不重复计算核心负责的距离和可达性。"""
    return (
        "Independently score each marked direction from 0 to 1 for its semantic "
        "promise of leading to the target. Use 0.5 when evidence is insufficient, "
        "above 0.5 for supporting cues and below 0.5 for contrary cues. "
        "Not seeing the target now is not contrary evidence for future exploration. "
        "Do not force score differences or normalize scores to sum to 1. "
        "Do not score distance, travel cost or reachability; navigation handles those."
    )


def parse_semantic_analysis_response(
    text: str, marker_labels: Sequence[str], view_ids: Sequence[int],
) -> SemanticAnalysis:
    """分别校验检测和分数；某一部分损坏不丢弃另一部分的有效信息。"""
    payload = _extract_json_mapping(text, "联合视觉分析")
    target = payload.get("target")
    target_view_ids = None
    detection_error = ""
    raw_views = target.get("view_ids") if isinstance(target, Mapping) else None
    if not isinstance(raw_views, list):
        detection_error = "联合分析缺少列表 target.view_ids"
    elif any(isinstance(item, bool) or not isinstance(item, int) or item not in view_ids
             for item in raw_views):
        detection_error = "目标线索包含无效的拍摄画面编号"
    elif len(set(raw_views)) != len(raw_views):
        detection_error = "目标线索列表含有重复画面编号"
    else:
        target_view_ids = tuple(raw_views)
    scores = {}
    invalid_labels = []
    raw_scores = payload.get("scores")
    for label in marker_labels:
        raw = raw_scores.get(label) if isinstance(raw_scores, Mapping) else None
        value = None if isinstance(raw, bool) else _finite_float(raw)
        if value is None or not 0.0 <= value <= 1.0:
            invalid_labels.append(label)
        else:
            scores[label] = value
    return SemanticAnalysis(
        target_view_ids, scores, detection_error,
        f"候选分数缺失或无效：{', '.join(invalid_labels)}" if invalid_labels else "",
    )


def build_target_visibility_prompt(target_text: str) -> str:
    """构造完整 RGB 图上的目标可见性问题。"""
    target = _target_json(target_text)
    return (
        "You are the target-presence detector for the robot's current view. "
        f"The target description is {target}. Decide only from the complete "
        "current RGB image whether the target is visible. Inspect image edges, "
        "distant regions, and partially occluded objects. Do not infer presence "
        "from room-level common sense. Choose exactly one of the two labels. "
        "Return exactly one of these JSON objects, without Markdown: "
        '{"visibility":"visible"} or {"visibility":"not_visible"}'
    )


def build_object_localization_prompt(target_text: str) -> str:
    """用同一次请求判断目标身份并取得框，供历史定位和到达后的新图确认共用。"""
    return (
        f"Find the physical object described by {_target_json(target_text)} in this RGB image. "
        "Require direct evidence of the object itself, not room context or a related sign. "
        "If multiple instances match, choose one clearly visible instance. Return JSON only: "
        '{"visibility":"visible","bbox_2d":[xmin,ymin,xmax,ymax]} '
        "with coordinates from 0 to 1000 relative to the full image. "
        'If no matching object is visible, return {"visibility":"not_visible"}.'
    )


def build_frontier_scores_prompt(
    target_text: str,
    marker_labels: Sequence[str],
    search_mode: SearchMode = SearchMode.OBJECT,
) -> str:
    """构造一次性评分多视角 RGB 中全部 Frontier 标记的问题。"""
    target = _target_json(target_text)
    labels = tuple(str(label).strip() for label in marker_labels)
    if (
        not labels
        or any(not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Frontier 标记必须是不重复的非空字符串")
    score_template = {label: 0.5 for label in labels}
    if search_mode is SearchMode.SCENE:
        scoring_rule = (
            f"The destination scene description is {target}. For every marker, "
            "score the likelihood that exploring beyond that marker will lead the "
            "robot into the destination scene"
        )
    else:
        scoring_rule = (
            f"The target description is {target}. For every marker, score the "
            "likelihood of finding the target by exploring beyond that marker"
        )
    prompt_body = (
        "The image is a multi-view RGB contact sheet from one robot scan. Each "
        "colored numeric marker denotes a Frontier exploration direction. "
        f"{scoring_rule}. " + _frontier_scoring_rules() + " "
        "Return every marker exactly once. Return JSON only, with no reasons, "
        "explanation, or Markdown: "
        + json.dumps({"scores": score_template}, ensure_ascii=False)
    )
    return prompt_body


def build_scene_assessment_prompt(scene_text: str) -> str:
    """构造整轮扫描拼图上的目的场景判断问题。"""
    scene = _target_json(scene_text)
    return (
        "The image is a multi-view RGB contact sheet captured from one robot "
        "position after observing the surrounding directions. The destination "
        f"scene description is {scene}. Decide whether the robot's current position "
        "is already inside or at that destination scene. Judge from the overall "
        "surroundings across all views. Do not match merely because the scene is "
        "visible in the distance or through a doorway, or because a related object "
        "or sign is visible. Return exactly one JSON object without Markdown: "
        '{"scene":"matched"} or {"scene":"not_matched"}'
    )


def build_target_grounding_prompt(target_text: str) -> str:
    """构造目标已确认可见时的边界框定位问题。"""
    target = _target_json(target_text)
    return (
        f"The target {target} has already been confirmed visible in the complete "
        "current RGB image. Locate the single best-matching instance. The bounding "
        "box must tightly enclose all visible parts of that instance without large "
        "background regions. Use bbox_2d=[x_min,y_min,x_max,y_max], with every "
        "coordinate expressed on a 0-to-1000 image-relative scale. Return JSON "
        "only, without Markdown: "
        '{"bbox_2d":[100,100,900,900]}'
    )


def build_target_confirmation_prompt(
    target_text: str,
    bbox_norm: Optional[Tuple[float, float, float, float]],
) -> str:
    """构造机器人接近候选后的二元最终确认问题。"""
    target = _target_json(target_text)
    box_text = (
        "not available"
        if bbox_norm is None
        else json.dumps(
            [round(float(value), 4) for value in bbox_norm],
            ensure_ascii=False,
        )
    )
    return (
        "The robot has approached a candidate detected for the target "
        f"{target}. Inspect the current RGB image and decide whether the "
        "candidate clearly matches the target description. The local detector's "
        f"normalized candidate box [x_min,y_min,x_max,y_max] is {box_text}. "
        "When available, the same candidate is outlined in green in the image. "
        "Confirm only when the visible candidate itself matches; reject lookalikes, "
        "background context, and ambiguous instances. Return exactly one JSON "
        "object without Markdown: "
        '{"confirmation":"confirmed"} or {"confirmation":"rejected"}'
    )


def parse_target_visibility_response(
    text: str,
) -> TargetVisibility:
    """解析目标可见性回答；格式不合法时抛出 ValueError。"""
    payload = _extract_json_mapping(text, "目标可见性")
    raw_visibility = str(payload.get("visibility", "")).strip().lower()
    if raw_visibility == TargetVisibility.VISIBLE.value:
        return TargetVisibility.VISIBLE
    if raw_visibility == TargetVisibility.NOT_VISIBLE.value:
        return TargetVisibility.NOT_VISIBLE
    raise ValueError("目标可见性必须是 visible 或 not_visible")


def parse_frontier_scores_response(
    text: str,
    marker_labels: Sequence[str],
) -> Mapping[str, float]:
    """解析一次返回的全部 Frontier 0-1 分数。"""
    labels = tuple(str(label).strip() for label in marker_labels)
    payload = _extract_json_mapping(text, "Frontier 评分")
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, Mapping):
        raise ValueError("Frontier 评分回答缺少 scores 对象")
    scores = {}
    for label in labels:
        score = _finite_float(raw_scores.get(label))
        if score is None or not 0.0 <= score <= 1.0:
            raise ValueError(f"Frontier {label} 分数必须是 0 到 1 的有限数")
        scores[label] = score
    return scores


def parse_target_grounding_response(
    text: str,
) -> Tuple[float, float, float, float]:
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
    )


def parse_target_confirmation_response(text: str) -> TargetConfirmation:
    """解析最终确认；格式不合法时抛出 ValueError。"""
    payload = _extract_json_mapping(text, "目标最终确认")
    raw_confirmation = str(payload.get("confirmation", "")).strip().lower()
    if raw_confirmation == TargetConfirmation.CONFIRMED.value:
        return TargetConfirmation.CONFIRMED
    if raw_confirmation == TargetConfirmation.REJECTED.value:
        return TargetConfirmation.REJECTED
    raise ValueError("目标最终确认必须是 confirmed 或 rejected")


def parse_scene_assessment_response(text: str) -> SceneAssessment:
    """解析目的场景判断；格式不合法时抛出 ValueError。"""
    payload = _extract_json_mapping(text, "目的场景判断")
    raw_scene = str(payload.get("scene", "")).strip().lower()
    if raw_scene == SceneAssessment.MATCHED.value:
        return SceneAssessment.MATCHED
    if raw_scene == SceneAssessment.NOT_MATCHED.value:
        return SceneAssessment.NOT_MATCHED
    raise ValueError("目的场景判断必须是 matched 或 not_matched")


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


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


__all__ = [
    "build_object_localization_prompt",
    "build_frontier_scores_prompt",
    "build_scene_assessment_prompt",
    "build_semantic_analysis_prompt",
    "build_target_confirmation_prompt",
    "build_target_grounding_prompt",
    "build_target_visibility_prompt",
    "parse_frontier_scores_response",
    "parse_scene_assessment_response",
    "parse_semantic_analysis_response",
    "parse_target_confirmation_response",
    "parse_target_grounding_response",
    "parse_target_visibility_response",
]
