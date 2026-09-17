"""通过 OpenAI/Anthropic-compatible API 提供语义搜索判断。"""

from __future__ import annotations

import base64
import json
import math
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..core.models import (
    FrontierScoreRequest,
    FrontierCandidate,
    NavigationFrame,
    SceneAssessment,
    SceneAssessmentResult,
    SearchMode,
    SemanticAnalysis,
    TargetConfirmation,
    TargetConfirmationResult,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from ..core.vision import (
    build_semantic_analysis_prompt,
    parse_semantic_analysis_response,
    build_frontier_scores_prompt,
    build_scene_assessment_prompt,
    build_target_confirmation_prompt,
    build_target_grounding_prompt,
    build_target_visibility_prompt,
    build_object_localization_prompt,
    parse_frontier_scores_response,
    parse_scene_assessment_response,
    parse_target_confirmation_response,
    parse_target_grounding_response,
    parse_target_visibility_response,
)
from .frontier_overlay import (
    BufferedScanImage,
    annotate_bbox_image,
    buffer_scan_image,
    build_frontier_score_sheet,
    build_scan_contact_sheet,
    build_semantic_analysis_sheet,
    pack_rgb_image,
)
from .perception import (
    ScanObservationContext,
    VlmInputImage,
    VlmInteraction,
)


class OpenAIApiFormat(str, Enum):
    """观察器支持的多模态请求格式。"""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """API 连接参数；endpoint_url 必须是完整接口地址。"""

    endpoint_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_s: float = 60.0
    api_format: OpenAIApiFormat = OpenAIApiFormat.CHAT_COMPLETIONS
    max_output_tokens: int = 2048
    reasoning_effort: Optional[str] = None
    opencode_session_id: Optional[str] = None


class OpenAICompatibleTargetObserver:
    """处理物体观测、整轮场景判断和 Frontier 批量评分。"""

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        on_vlm_interaction: Optional[
            Callable[[VlmInteraction], None]
        ] = None,
    ) -> None:
        _validate_config(config)
        self._config = config
        self._on_vlm_interaction = on_vlm_interaction
        self._interaction_index = 0
        self._interaction_lock = threading.Lock()
        self._scan_images: Dict[int, BufferedScanImage] = {}

    def analyze_views(
        self,
        images: Mapping[int, BufferedScanImage],
        candidates: Tuple[FrontierCandidate, ...],
        goal: TargetSearchGoal,
        *, trace_context: Optional[Mapping[str, Any]] = None,
    ) -> SemanticAnalysis:
        """只读取本次快照，一次请求同时检查全部画面并评分可见 Frontier。"""
        try:
            image, markers = build_semantic_analysis_sheet(images, candidates)
            labels = tuple(marker.label for marker in markers)
            prompt = build_semantic_analysis_prompt(goal.target_text, labels, tuple(images), goal.search_mode)
        except Exception as exc:
            return SemanticAnalysis(None, detection_error=_failure_reason("联合分析输入无效", exc))
        context = dict(trace_context or {})
        regions = context.get("region_ids", {})
        positions = {item.candidate_id: item.world_xy for item in candidates}
        context["markers"] = tuple({
            "label": marker.label, "candidate_id": marker.candidate_id,
            "region_id": regions.get(marker.candidate_id, marker.candidate_id),
            "world_xy": positions[marker.candidate_id],
            "view_id": marker.view_id, "source_pixel_xy": marker.source_pixel_xy,
        } for marker in markers)
        interaction = self._begin_interaction("semantic_analysis", prompt, image, context=context)
        assistant_text, response_json = "", ""
        try:
            payload, response_json = self._request_model(prompt, image)
            assistant_text = self._response_text(payload)
            parsed = parse_semantic_analysis_response(assistant_text, labels, tuple(images))
        except Exception as exc:
            self._finish_interaction(interaction, assistant_text, response_json, error=_exception_text(exc))
            return SemanticAnalysis(
                None, detection_error=_failure_reason("联合分析失败", exc),
                interaction_id=interaction.interaction_id,
            )
        result = replace(parsed, interaction_id=interaction.interaction_id, frontier_scores={
            marker.candidate_id: parsed.frontier_scores[marker.label]
            for marker in markers if marker.label in parsed.frontier_scores
        })
        self._finish_interaction(
            interaction, assistant_text, response_json,
            parsed_result=json.dumps({
                "target_view_ids": result.target_view_ids,
                "frontier_scores": result.frontier_scores,
            }, ensure_ascii=False),
            error="; ".join(value for value in (result.detection_error, result.scoring_error) if value),
        )
        return result

    def locate_object(
        self, frame: NavigationFrame, goal: TargetSearchGoal, *, context=None,
    ) -> TargetObservation:
        """一帧一次请求，返回目标身份判断与框；不参与普通扫描的实时抢占。"""
        if frame.rgb is None:
            return _uncertain("物体定位缺少 RGB。")
        image = pack_rgb_image(frame.rgb)
        prompt = build_object_localization_prompt(goal.target_text)
        interaction = self._begin_interaction(
            "object_localization", prompt, image,
            context={**_frame_trace_context(frame, "object_localization"), **(context or {})},
        )
        assistant_text, response_json = "", ""
        try:
            payload, response_json = self._request_model(prompt, image)
            assistant_text = self._response_text(payload)
            visibility = parse_target_visibility_response(assistant_text)
            bbox = parse_target_grounding_response(assistant_text) if visibility is TargetVisibility.VISIBLE else None
        except Exception as exc:
            self._finish_interaction(interaction, assistant_text=assistant_text, response_json=response_json, error=_exception_text(exc))
            return _uncertain(_failure_reason("物体定位请求失败", exc))
        self._finish_interaction(
            interaction, assistant_text=assistant_text, response_json=response_json,
            parsed_result=json.dumps({"visibility": visibility.value, "bbox_norm": bbox}), bbox_norm=bbox,
        )
        return TargetObservation(visibility, bbox_norm=bbox, source="vlm")

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        scan_context: Optional[ScanObservationContext] = None,
    ) -> TargetObservation:
        """物体模式判断单帧；场景模式只缓存当前扫描 RGB。"""
        if frame.rgb is None:
            return _uncertain("当前帧没有 RGB 图像。")

        try:
            if scan_context is not None:
                buffered = self._record_scan_image(frame, scan_context)
                image = VlmInputImage(
                    buffered.width_px,
                    buffered.height_px,
                    buffered.rgb_bytes,
                )
            else:
                image = pack_rgb_image(frame.rgb)
        except Exception as exc:
            return _uncertain(_failure_reason("RGB 输入准备失败", exc))

        if goal.search_mode is SearchMode.SCENE:
            if scan_context is None:
                return _uncertain("场景模式只能在扫描方向中采集 RGB。")
            return TargetObservation(
                visibility=TargetVisibility.NOT_VISIBLE,
                reason="场景模式已缓存当前扫描方向，整轮结束后统一判断。",
                source="scene_scan",
            )

        prompt = build_target_visibility_prompt(goal.target_text)
        interaction = self._begin_interaction(
            "target_visibility",
            prompt,
            image,
            context=_frame_trace_context(frame, "current_view"),
        )
        assistant_text = ""
        response_json = ""
        try:
            response_payload, response_json = self._request_model(
                prompt,
                image,
            )
            assistant_text = self._response_text(response_payload)
            visibility_text = assistant_text
            visibility = parse_target_visibility_response(visibility_text)
        except Exception as exc:
            self._finish_interaction(
                interaction,
                assistant_text=assistant_text,
                response_json=response_json,
                error=_exception_text(exc),
            )
            return _uncertain(_failure_reason("目标可见性判断失败", exc))
        self._finish_interaction(
            interaction,
            assistant_text=assistant_text,
            response_json=response_json,
            parsed_result=json.dumps(
                {"visibility": visibility.value},
                ensure_ascii=False,
            ),
        )

        if visibility is TargetVisibility.NOT_VISIBLE:
            return TargetObservation(visibility=TargetVisibility.NOT_VISIBLE)
        return self._observe_visible_target(
            goal, image, context={**interaction.context, "parent_request_id": interaction.interaction_id},
        )

    def score_frontiers(
        self,
        request: FrontierScoreRequest,
        goal: TargetSearchGoal,
    ) -> Mapping[str, float]:
        """把本轮 Frontier 编号到多视角 RGB，只调用模型一次。"""
        try:
            image, markers = build_frontier_score_sheet(
                self._scan_images,
                request.candidates,
            )
            marker_labels = tuple(marker.label for marker in markers)
            prompt = build_frontier_scores_prompt(
                goal.target_text,
                marker_labels,
                goal.search_mode,
            )
        except Exception:
            self._scan_images.clear()
            return {}

        interaction = self._begin_interaction(
            "frontier_scores",
            prompt,
            image,
        )
        assistant_text = ""
        response_json = ""
        try:
            response_payload, response_json = self._request_model(
                prompt,
                image,
            )
            assistant_text = self._response_text(response_payload)
            marker_scores = parse_frontier_scores_response(
                assistant_text,
                marker_labels,
            )
            candidate_scores = {
                marker.candidate_id: marker_scores[marker.label]
                for marker in markers
            }
        except Exception as exc:
            self._finish_interaction(
                interaction,
                assistant_text=assistant_text,
                response_json=response_json,
                error=_exception_text(exc),
            )
            # 语义评分失败不得阻塞纯地图 Frontier 探索。
            return {}
        finally:
            self._scan_images.clear()
        self._finish_interaction(
            interaction,
            assistant_text=assistant_text,
            response_json=response_json,
            parsed_result=json.dumps(
                candidate_scores,
                ensure_ascii=False,
            ),
        )
        return candidate_scores

    def assess_scene(
        self,
        goal: TargetSearchGoal,
    ) -> SceneAssessmentResult:
        """把本轮全部扫描 RGB 拼成一张图，请求一次目的场景判断。"""
        try:
            image = build_scan_contact_sheet(self._scan_images)
            prompt = build_scene_assessment_prompt(goal.target_text)
        except Exception as exc:
            return SceneAssessmentResult(
                SceneAssessment.UNCERTAIN,
                _failure_reason("目的场景输入准备失败", exc),
            )
        interaction = self._begin_interaction(
            "scene_assessment",
            prompt,
            image,
        )
        assistant_text = ""
        response_json = ""
        try:
            response_payload, response_json = self._request_model(prompt, image)
            assistant_text = self._response_text(response_payload)
            assessment = parse_scene_assessment_response(assistant_text)
        except Exception as exc:
            self._finish_interaction(
                interaction,
                assistant_text=assistant_text,
                response_json=response_json,
                error=_exception_text(exc),
            )
            return SceneAssessmentResult(
                SceneAssessment.UNCERTAIN,
                _failure_reason("目的场景判断失败", exc),
            )

        self._finish_interaction(
            interaction,
            assistant_text=assistant_text,
            response_json=response_json,
            parsed_result=json.dumps(
                {"scene": assessment.value},
                ensure_ascii=False,
            ),
        )
        return SceneAssessmentResult(assessment)

    def record_scan_frame(
        self,
        frame: NavigationFrame,
        context: ScanObservationContext,
    ) -> None:
        """只缓存扫描 RGB，供整轮场景判断和 Frontier 评分。"""
        self._record_scan_image(frame, context)

    def confirm_target(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        observation: TargetObservation,
    ) -> TargetConfirmationResult:
        """机器人接近候选后，用当前 RGB 做一次二元最终确认。"""
        if frame.rgb is None:
            return TargetConfirmationResult(
                TargetConfirmation.UNCERTAIN,
                "当前帧没有 RGB 图像，无法最终确认目标。",
            )
        try:
            image = pack_rgb_image(frame.rgb)
            if observation.bbox_norm is not None:
                image = annotate_bbox_image(image, observation.bbox_norm)
            prompt = build_target_confirmation_prompt(
                goal.target_text,
                observation.bbox_norm,
            )
        except Exception as exc:
            return TargetConfirmationResult(
                TargetConfirmation.UNCERTAIN,
                _failure_reason("最终确认输入准备失败", exc),
            )

        interaction = self._begin_interaction(
            "target_confirmation",
            prompt,
            image,
            bbox_norm=observation.bbox_norm,
            context=_frame_trace_context(frame, "target_confirmation"),
        )
        assistant_text = ""
        response_json = ""
        try:
            response_payload, response_json = self._request_model(prompt, image)
            assistant_text = self._response_text(response_payload)
            confirmation = parse_target_confirmation_response(assistant_text)
        except Exception as exc:
            self._finish_interaction(
                interaction,
                assistant_text=assistant_text,
                response_json=response_json,
                error=_exception_text(exc),
            )
            return TargetConfirmationResult(
                TargetConfirmation.UNCERTAIN,
                _failure_reason("目标最终确认失败", exc),
            )

        self._finish_interaction(
            interaction,
            assistant_text=assistant_text,
            response_json=response_json,
            parsed_result=json.dumps(
                {"confirmation": confirmation.value},
                ensure_ascii=False,
            ),
            bbox_norm=observation.bbox_norm,
        )
        return TargetConfirmationResult(confirmation)

    def _record_scan_image(
        self,
        frame: NavigationFrame,
        context: ScanObservationContext,
    ) -> BufferedScanImage:
        """按扫描下标覆盖重复观测；新一轮从下标 0 重置。"""
        if context.index == 0:
            self._scan_images.clear()
        buffered = buffer_scan_image(frame)
        self._scan_images[context.index] = buffered
        return buffered

    def _observe_visible_target(
        self,
        goal: TargetSearchGoal,
        image: VlmInputImage,
        *, context: Optional[Mapping[str, Any]] = None,
    ) -> TargetObservation:
        """目标可见时取得目标框；没有可靠目标框时不允许导航接近。"""
        prompt = build_target_grounding_prompt(goal.target_text)
        interaction = self._begin_interaction(
            "target_grounding",
            prompt,
            image,
            context=context,
        )
        assistant_text = ""
        response_json = ""
        try:
            response_payload, response_json = self._request_model(
                prompt,
                image,
            )
            assistant_text = self._response_text(response_payload)
            bbox_norm = parse_target_grounding_response(assistant_text)
        except Exception as exc:
            self._finish_interaction(
                interaction,
                assistant_text=assistant_text,
                response_json=response_json,
                error=_exception_text(exc),
            )
            return _uncertain(_failure_reason("目标框定位失败", exc))
        self._finish_interaction(
            interaction,
            assistant_text=assistant_text,
            response_json=response_json,
            parsed_result=json.dumps(
                {"bbox_norm": bbox_norm},
                ensure_ascii=False,
            ),
            bbox_norm=bbox_norm,
        )
        return TargetObservation(
            visibility=TargetVisibility.VISIBLE,
            bbox_norm=bbox_norm,
        )

    def _request_model(
        self,
        prompt: str,
        image: VlmInputImage,
    ) -> Tuple[Any, str]:
        """发送带图请求，并保留未经提取的完整 JSON 回应。"""
        image_url = _packed_rgb_to_png_data_url(image)
        if self._config.api_format is OpenAIApiFormat.CHAT_COMPLETIONS:
            payload = _chat_completions_payload(self._config, prompt, image_url)
        elif self._config.api_format is OpenAIApiFormat.RESPONSES:
            payload = _responses_payload(self._config, prompt, image_url)
        else:
            payload = _anthropic_messages_payload(
                self._config,
                prompt,
                image_url,
            )

        response_payload = self._post_json(payload)
        response_json = json.dumps(
            response_payload,
            ensure_ascii=False,
            indent=2,
        )
        return response_payload, response_json

    def _response_text(self, response_payload: Any) -> str:
        """从当前 API 格式的完整回应中读取 assistant 文本。"""
        if self._config.api_format is OpenAIApiFormat.CHAT_COMPLETIONS:
            return _chat_completions_text(response_payload)
        if self._config.api_format is OpenAIApiFormat.RESPONSES:
            return _responses_text(response_payload)
        return _anthropic_messages_text(response_payload)

    def _begin_interaction(
        self,
        task: str,
        prompt: str,
        image: VlmInputImage,
        bbox_norm: Optional[Tuple[float, float, float, float]] = None,
        *, context: Optional[Mapping[str, Any]] = None,
    ) -> VlmInteraction:
        """在请求发出前记录完整输入，使慢请求期间也能在 Rerun 查看。"""
        with self._interaction_lock:
            self._interaction_index += 1
            interaction_id = self._interaction_index
        interaction = VlmInteraction(
            interaction_id=interaction_id,
            phase="request",
            task=task,
            endpoint_url=self._config.endpoint_url.strip(),
            model=self._config.model.strip(),
            api_format=self._config.api_format.value,
            reasoning_effort=self._config.reasoning_effort,
            max_output_tokens=self._config.max_output_tokens,
            prompt=prompt,
            image=image,
            bbox_norm=bbox_norm,
            context=dict(context or {}),
            started_monotonic_s=time.monotonic(),
        )
        self._emit_interaction(interaction)
        return interaction

    def _finish_interaction(
        self,
        interaction: VlmInteraction,
        assistant_text: str,
        response_json: str,
        parsed_result: str = "",
        bbox_norm: Optional[Tuple[float, float, float, float]] = None,
        error: str = "",
    ) -> None:
        """把原始回应、解析结果或错误补到同一条交互记录。"""
        if error:
            print(f"VLM 请求 R{interaction.interaction_id:06d}（{interaction.task}）失败：{error}", flush=True)
        self._emit_interaction(
            replace(
                interaction,
                phase="response",
                assistant_text=assistant_text,
                response_json=response_json,
                parsed_result=parsed_result,
                bbox_norm=bbox_norm,
                error=error,
                elapsed_s=max(0.0, time.monotonic() - interaction.started_monotonic_s),
            )
        )

    def _emit_interaction(self, interaction: VlmInteraction) -> None:
        """可视化是旁路，记录失败不能影响模型和导航。"""
        if self._on_vlm_interaction is None:
            return
        try:
            self._on_vlm_interaction(interaction)
        except Exception:
            pass

    def _post_json(self, payload: Mapping[str, Any]) -> Any:
        """发送 JSON 请求并解析 JSON 回应。"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "robot-nav/0.1 multimodal client",
        }
        if self._config.api_key.strip():
            if self._config.api_format is OpenAIApiFormat.ANTHROPIC_MESSAGES:
                headers["x-api-key"] = self._config.api_key.strip()
            else:
                headers["Authorization"] = (
                    f"Bearer {self._config.api_key.strip()}"
                )
        if self._config.api_format is OpenAIApiFormat.ANTHROPIC_MESSAGES:
            headers["anthropic-version"] = "2023-06-01"
        if self._config.opencode_session_id is not None:
            headers["x-opencode-session"] = self._config.opencode_session_id
        request = Request(
            self._config.endpoint_url.strip(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._config.timeout_s) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
            detail = f"：{response_body}" if response_body else ""
            raise RuntimeError(f"API 返回 HTTP {exc.code}{detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"API 连接失败：{exc.reason}") from exc
        return response_payload


def _frame_trace_context(frame: NavigationFrame, source: str) -> Mapping[str, Any]:
    """仅用于记录来源，不发送给模型，也不参与目标判断。"""
    return {"source": source, "views": ({
        "view_id": 1, "timestamp_s": frame.timestamp_s,
        "pose": {"x_m": frame.pose.x_m, "y_m": frame.pose.y_m, "yaw_rad": frame.pose.yaw_rad},
        "heading_world_rad": frame.pose.yaw_rad + frame.camera_extrinsics_in_robot.yaw_rad,
        "map_frame_id": frame.obstacle_map.frame_id,
    },)}


def _chat_completions_payload(
    config: OpenAICompatibleConfig,
    prompt: str,
    image_url: str,
) -> Mapping[str, Any]:
    """构造 Chat Completions 多模态请求。"""
    return {
        "model": config.model.strip(),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    }


def _responses_payload(
    config: OpenAICompatibleConfig,
    prompt: str,
    image_url: str,
) -> Mapping[str, Any]:
    """构造 Responses API 多模态请求。"""
    payload = {
        "model": config.model.strip(),
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        "max_output_tokens": config.max_output_tokens,
    }
    if config.reasoning_effort is not None:
        payload["reasoning"] = {
            "effort": config.reasoning_effort.strip(),
        }
    return payload


def _anthropic_messages_payload(
    config: OpenAICompatibleConfig,
    prompt: str,
    image_url: str,
) -> Mapping[str, Any]:
    """构造 Anthropic Messages 多模态请求，并显式关闭 thinking。"""
    media_type, image_base64 = _image_data_url_parts(image_url)
    return {
        "model": config.model.strip(),
        "max_tokens": config.max_output_tokens,
        "thinking": {"type": "disabled"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }


def _validate_config(config: OpenAICompatibleConfig) -> None:
    if not isinstance(config, OpenAICompatibleConfig):
        raise TypeError("config 必须是 OpenAICompatibleConfig")
    if not isinstance(config.endpoint_url, str):
        raise ValueError("endpoint_url 必须是字符串")
    parsed_url = urlsplit(config.endpoint_url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("endpoint_url 必须是完整的 HTTP(S) API 地址")
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("model 不能为空")
    if not isinstance(config.api_key, str):
        raise ValueError("api_key 必须是字符串")
    if config.opencode_session_id is not None and (
        not isinstance(config.opencode_session_id, str)
        or not config.opencode_session_id
        or any(not 33 <= ord(char) <= 126 for char in config.opencode_session_id)
    ):
        raise ValueError("opencode_session_id 必须是不含空白的 ASCII 字符串或 None")
    if not isinstance(config.api_format, OpenAIApiFormat):
        raise ValueError("api_format 必须是 OpenAIApiFormat")
    if config.reasoning_effort is not None:
        if config.api_format is not OpenAIApiFormat.RESPONSES:
            raise ValueError("reasoning_effort 只能用于 Responses API")
        if (
            not isinstance(config.reasoning_effort, str)
            or not config.reasoning_effort.strip()
        ):
            raise ValueError("reasoning_effort 必须是非空字符串或 None")
    if (
        isinstance(config.max_output_tokens, bool)
        or not isinstance(config.max_output_tokens, int)
        or config.max_output_tokens <= 0
    ):
        raise ValueError("max_output_tokens 必须是正整数")
    try:
        timeout_s = float(config.timeout_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_s 必须是正有限数") from exc
    if (
        isinstance(config.timeout_s, bool)
        or not math.isfinite(timeout_s)
        or timeout_s <= 0.0
    ):
        raise ValueError("timeout_s 必须是正有限数")


def _chat_completions_text(payload: Any) -> str:
    """读取 Chat Completions 首个 choice 的文本内容。"""
    if not isinstance(payload, Mapping):
        raise ValueError("API 回应必须是 JSON 对象")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API 回应缺少 choices")
    first = choices[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
        raise ValueError("API 回应缺少 message")
    content = first["message"].get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, Mapping) and str(block.get("text", "")).strip()
        ]
        if parts:
            return "\n".join(parts)
    raise ValueError("API 回应没有 assistant 文本")


def _responses_text(payload: Any) -> str:
    """读取 Responses API 输出中的文本内容。"""
    if not isinstance(payload, Mapping):
        raise ValueError("Responses API 回应必须是 JSON 对象")

    output = payload.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    not isinstance(block, Mapping)
                    or block.get("type") != "output_text"
                ):
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    if payload.get("status") == "incomplete":
        details = payload.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, Mapping) else None
        if isinstance(reason, str) and reason.strip():
            raise ValueError(f"Responses API 回应未完成：{reason.strip()}")
        raise ValueError("Responses API 回应未完成且没有 output_text")
    raise ValueError("Responses API 回应没有 output_text")


def _anthropic_messages_text(payload: Any) -> str:
    """读取 Anthropic Messages 回应中的全部文本块。"""
    if not isinstance(payload, Mapping):
        raise ValueError("Anthropic Messages 回应必须是 JSON 对象")
    content = payload.get("content")
    if not isinstance(content, list):
        raise ValueError("Anthropic Messages 回应缺少 content")
    parts = [
        str(block.get("text", "")).strip()
        for block in content
        if (
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and str(block.get("text", "")).strip()
        )
    ]
    if parts:
        return "\n".join(parts)
    raise ValueError("Anthropic Messages 回应没有 assistant 文本")


def _image_data_url_parts(image_url: str) -> Tuple[str, str]:
    """把 PNG data URL 拆成 Anthropic image source 所需字段。"""
    prefix = "data:image/png;base64,"
    if not image_url.startswith(prefix):
        raise ValueError("图像必须是 PNG base64 data URL")
    image_base64 = image_url[len(prefix) :]
    if not image_base64:
        raise ValueError("PNG base64 数据不能为空")
    return "image/png", image_base64


def _packed_rgb_to_png_data_url(image: VlmInputImage) -> str:
    """把紧凑 RGB bytes 编码为与 Rerun 中显示完全一致的 PNG。"""
    row_bytes = image.width_px * 3
    if len(image.rgb_bytes) != row_bytes * image.height_px:
        raise ValueError("VLM RGB 数据长度与尺寸不匹配")
    scanlines = bytearray()
    for row in range(image.height_px):
        scanlines.append(0)
        start = row * row_bytes
        scanlines.extend(image.rgb_bytes[start : start + row_bytes])
    return _png_scanlines_to_data_url(
        image.width_px,
        image.height_px,
        scanlines,
    )


def _png_scanlines_to_data_url(
    width: int,
    height: int,
    scanlines: bytearray,
) -> str:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines)))
        + _png_chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    body = chunk_type + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _failure_reason(stage: str, exc: Exception) -> str:
    detail = _exception_text(exc)
    return f"{stage}：{detail[:240]}"


def _exception_text(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def _uncertain(reason: str) -> TargetObservation:
    return TargetObservation(
        visibility=TargetVisibility.UNCERTAIN,
        reason=reason,
    )


__all__ = [
    "OpenAIApiFormat",
    "OpenAICompatibleConfig",
    "OpenAICompatibleTargetObserver",
]
