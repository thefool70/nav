"""YOLOE 文本模型边界：固定目标提示词，输出原任务类别与同帧分割。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Optional,
    Tuple,
)

import numpy as np

from .yoloe_targets import text_prompts

from ..core.models import (
    TargetObservation,
    TargetVisibility,
)


DEFAULT_YOLOE_MODEL_PATH = Path("data/models/yoloe/yoloe-26s-seg.pt")


@dataclass(frozen=True)
class YoloEConfig:
    """文本在启动时编码一次，每帧复用固定类别和推理参数。"""

    class_texts: Tuple[str, ...]
    model_path: Path = DEFAULT_YOLOE_MODEL_PATH
    device: str = "cuda"
    confidence_threshold: float = 0.25
    image_size: int = 640
    max_detections: int = 10


class YoloEDetector:
    """固定提示词只影响检测器，输出类别仍与 Qwen 使用的任务类别一致。"""

    def __init__(self, config: YoloEConfig, on_stage=None) -> None:
        on_stage = on_stage or (lambda stage: None)
        prompts = text_prompts(config.class_texts)
        model_path = Path(config.model_path).resolve()
        if not model_path.is_file():
            raise RuntimeError(f"缺少 YOLOE 文本模型文件：{model_path}")
        on_stage("importing_yolo")
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError("当前 Python 环境未安装 ultralytics") from exc

        on_stage("loading_yolo")
        # 在权重目录查找 mobileclip2_b.ts，避免切换运行目录时重下载。
        os.chdir(model_path.parent)
        self._model = YOLOE(str(model_path), verbose=False)
        if hasattr(self._model.model.model[-1], "lrpc"):
            raise ValueError(f"当前检测使用文本提示，请配置非 PF 权重：{model_path}")
        self._config = config
        on_stage("encoding_class_text")
        self._model.set_classes(list(prompts))
        self._classes = dict(enumerate(config.class_texts))
        self.selection = {
            "mode": "text",
            "model_path": str(model_path),
            "reason": "本轮目标使用固定文本提示词，未收录目标保留原文",
            "class_mapping": [
                {"target": target, "text_prompt": prompt}
                for target, prompt in zip(config.class_texts, prompts)
            ],
        }
        on_stage("warming_up")
        self.detect_boxes(np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8))

    def detect_boxes(self, rgb) -> Tuple[Tuple[TargetObservation, ...], int]:
        """按置信度返回框与原始 RGB 尺寸掩码；未检出返回空列表。"""
        rgb = _as_rgb_array(rgb)
        # Ultralytics 的 numpy 输入约定为 OpenCV BGR。
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        results = self._model.predict(
            source=bgr,
            conf=self._config.confidence_threshold,
            imgsz=self._config.image_size,
            max_det=self._config.max_detections,
            device=self._config.device,
            half=self._config.device.startswith("cuda") or self._config.device.isdigit(),
            retina_masks=True,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return (), 0

        boxes = results[0].boxes.cpu().numpy()
        normalized_boxes = np.asarray(boxes.xyxyn, dtype=np.float64)
        confidences = np.asarray(boxes.conf, dtype=np.float64)
        classes = np.asarray(boxes.cls, dtype=np.int64)
        candidate_count = len(confidences)
        if candidate_count == 0:
            return (), 0

        masks = results[0].masks
        masks = masks.data.cpu().numpy().astype(bool) if masks is not None else None
        if masks is not None and masks.shape != (candidate_count, *rgb.shape[:2]):
            raise ValueError("YOLOE 掩码与原始 RGB 尺寸或检测框数量不一致")
        observations = []
        for index in np.argsort(-confidences):
            class_id = int(classes[int(index)])
            if class_id not in self._classes:
                continue
            bbox_norm = _normalized_box(normalized_boxes[int(index)])
            confidence = float(confidences[int(index)])
            if bbox_norm is None or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            observations.append(TargetObservation(
                TargetVisibility.VISIBLE, bbox_norm=bbox_norm, source="yoloe",
                confidence=confidence, target_mask=masks[int(index)] if masks is not None else None,
                class_text=self._classes[class_id],
            ))
            if len(observations) >= self._config.max_detections:
                break
        return tuple(observations), candidate_count


def _normalized_box(values: Any) -> Optional[Tuple[float, float, float, float]]:
    """将模型输出的归一化 xyxy 框裁到 [0, 1]，丢弃非有限或无面积的框。"""
    try:
        x_min, y_min, x_max, y_max = (float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        return None
    x_min = min(1.0, max(0.0, x_min))
    y_min = min(1.0, max(0.0, y_min))
    x_max = min(1.0, max(0.0, x_max))
    y_max = min(1.0, max(0.0, y_max))
    if x_min >= x_max or y_min >= y_max:
        return None
    return (x_min, y_min, x_max, y_max)


def _as_rgb_array(rgb: Any) -> np.ndarray:
    if rgb is None:
        raise ValueError("当前帧没有 RGB 图像")
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB 图像必须是 H×W×3 数组")
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)


__all__ = [
    "YoloEDetector",
    "DEFAULT_YOLOE_MODEL_PATH",
    "YoloEConfig",
]
