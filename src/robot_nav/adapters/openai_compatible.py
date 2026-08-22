"""通过 OpenAI-compatible API 观察语义目标。"""

from __future__ import annotations

import base64
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from ..core.models import (
    NavigationFrame,
    RgbImage,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from ..core.vision import (
    build_search_direction_prompt,
    build_target_grounding_prompt,
    build_target_visibility_prompt,
    parse_search_direction_response,
    parse_target_grounding_response,
    parse_target_visibility_response,
)


class OpenAIApiFormat(str, Enum):
    """观察器支持的 OpenAI-compatible 请求格式。"""

    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """API 连接参数；endpoint_url 必须是完整接口地址。"""

    endpoint_url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_s: float = 60.0
    api_format: OpenAIApiFormat = OpenAIApiFormat.CHAT_COMPLETIONS
    max_output_tokens: int = 2048


class OpenAICompatibleTargetObserver:
    """把一帧 RGB 经过 VLM 判断后转换为 TargetObservation。"""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        _validate_config(config)
        self._config = config

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
    ) -> TargetObservation:
        """先判断可见性，再按结果定位目标或评估当前探索方向。"""
        if frame.rgb is None:
            return _uncertain("当前帧没有 RGB 图像。")

        try:
            image_url = _rgb_to_png_data_url(frame.rgb)
            visibility_text = self._ask(
                build_target_visibility_prompt(goal.target_text), image_url
            )
            visibility, visibility_reason = parse_target_visibility_response(
                visibility_text
            )
        except Exception as exc:
            return _uncertain(_failure_reason("目标可见性判断失败", exc))

        if visibility is TargetVisibility.UNCERTAIN:
            return _uncertain(visibility_reason or "模型无法确定目标是否可见。")
        if visibility is TargetVisibility.NOT_VISIBLE:
            return self._observe_search_direction(goal, image_url, visibility_reason)
        return self._observe_visible_target(goal, image_url, visibility_reason)

    def _observe_search_direction(
        self,
        goal: TargetSearchGoal,
        image_url: str,
        visibility_reason: str,
    ) -> TargetObservation:
        """目标不可见时取得可选方向评分；评分失败不阻塞 Frontier。"""
        try:
            response = self._ask(
                build_search_direction_prompt(goal.target_text), image_url
            )
            score, direction_reason = parse_search_direction_response(response)
        except Exception as exc:
            return TargetObservation(
                visibility=TargetVisibility.NOT_VISIBLE,
                reason=_join_reasons(
                    visibility_reason,
                    _failure_reason("方向评分失败，继续使用地图探索", exc),
                ),
            )
        return TargetObservation(
            visibility=TargetVisibility.NOT_VISIBLE,
            direction_score=score,
            reason=_join_reasons(visibility_reason, direction_reason),
        )

    def _observe_visible_target(
        self,
        goal: TargetSearchGoal,
        image_url: str,
        visibility_reason: str,
    ) -> TargetObservation:
        """目标可见时取得目标框；没有可靠目标框时不允许导航接近。"""
        try:
            response = self._ask(
                build_target_grounding_prompt(goal.target_text), image_url
            )
            bbox_norm, grounding_reason = parse_target_grounding_response(response)
        except Exception as exc:
            return _uncertain(
                _join_reasons(
                    visibility_reason,
                    _failure_reason("目标框定位失败", exc),
                )
            )
        return TargetObservation(
            visibility=TargetVisibility.VISIBLE,
            bbox_norm=bbox_norm,
            reason=_join_reasons(visibility_reason, grounding_reason),
        )

    def _ask(self, prompt: str, image_url: str) -> str:
        """按配置的 API 格式发送一次带图问题。"""
        if self._config.api_format is OpenAIApiFormat.CHAT_COMPLETIONS:
            payload = _chat_completions_payload(self._config, prompt, image_url)
        else:
            payload = _responses_payload(self._config, prompt, image_url)

        response_payload = self._post_json(payload)
        if self._config.api_format is OpenAIApiFormat.CHAT_COMPLETIONS:
            return _chat_completions_text(response_payload)
        return _responses_text(response_payload)

    def _post_json(self, payload: Mapping[str, Any]) -> Any:
        """发送 JSON 请求并解析 JSON 回应。"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "robot-nav/0.1 OpenAI-compatible client",
        }
        if self._config.api_key.strip():
            headers["Authorization"] = f"Bearer {self._config.api_key.strip()}"
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
            raise RuntimeError(f"API 返回 HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"API 连接失败：{exc.reason}") from exc
        return response_payload


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
    return {
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


def _validate_config(config: OpenAICompatibleConfig) -> None:
    if not isinstance(config, OpenAICompatibleConfig):
        raise TypeError("config 必须是 OpenAICompatibleConfig")
    if not isinstance(config.endpoint_url, str):
        raise ValueError("endpoint_url 必须是字符串")
    parsed_url = urlsplit(config.endpoint_url.strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("endpoint_url 必须是完整的 HTTP(S) Chat Completions 地址")
    if not isinstance(config.model, str) or not config.model.strip():
        raise ValueError("model 不能为空")
    if not isinstance(config.api_key, str):
        raise ValueError("api_key 必须是字符串")
    if not isinstance(config.api_format, OpenAIApiFormat):
        raise ValueError("api_format 必须是 OpenAIApiFormat")
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


def _rgb_to_png_data_url(image: RgbImage) -> str:
    """仅用标准库把内部 RGB 序列编码为无滤波 8-bit PNG。"""
    height = len(image)
    if height <= 0:
        raise ValueError("RGB 图像不能为空")
    width = len(image[0])
    if width <= 0:
        raise ValueError("RGB 图像不能为空")

    scanlines = bytearray()
    for row in image:
        if len(row) != width:
            raise ValueError("RGB 图像每行宽度必须一致")
        scanlines.append(0)
        for pixel in row:
            if not isinstance(pixel, Sequence) or len(pixel) != 3:
                raise ValueError("RGB 像素必须包含三个通道")
            scanlines.extend(_rgb_channel(value) for value in pixel)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines)))
        + _png_chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rgb_channel(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("RGB 通道必须是 0 到 255 的整数")
    try:
        channel = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("RGB 通道必须是 0 到 255 的整数") from exc
    if channel != value or not 0 <= channel <= 255:
        raise ValueError("RGB 通道必须是 0 到 255 的整数")
    return channel


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    body = chunk_type + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _failure_reason(stage: str, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{stage}：{detail[:240]}"


def _join_reasons(first: str, second: str) -> str:
    return "；".join(item for item in (first, second) if item)


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
