"""YOLO-World 模型边界：加载模型并返回检测框，不管理导航或分割策略。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Optional,
    Tuple,
)

import numpy as np

from ..core.models import (
    TargetObservation,
    TargetVisibility,
)


DEFAULT_YOLO_WORLD_MODEL_PATH = Path(
    "data/models/yolo-world/yolov8s-world.pt"
)


@dataclass(frozen=True)
class YoloWorldConfig:
    """YOLOv8s-World 的本地推理参数。"""

    class_text: str
    model_path: Path = DEFAULT_YOLO_WORLD_MODEL_PATH
    device: str = "cuda"
    confidence_threshold: float = 0.25
    image_size: int = 640
    max_detections: int = 3


class YoloWorldDetector:
    """只负责 YOLO 框检测，初始化与检测不依赖 SAM2。"""

    def __init__(self, config: YoloWorldConfig, on_stage=None) -> None:
        on_stage = on_stage or (lambda stage: None)
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise RuntimeError(f"缺少 YOLO-World 模型文件：{model_path}")
        on_stage("importing_yolo")
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("当前 Python 环境未安装 ultralytics") from exc

        on_stage("loading_yolo")
        self._model = YOLOWorld(str(model_path), verbose=False)
        self._config = config
        on_stage("encoding_class_text")
        self._model.set_classes([config.class_text.strip()])

    def detect_boxes(self, rgb) -> Tuple[Tuple[TargetObservation, ...], int]:
        """按置信度返回目标框；未检出返回空列表，不运行分割。"""
        rgb = _as_rgb_array(rgb)
        # Ultralytics 的 numpy 输入约定为 OpenCV BGR。
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        results = self._model.predict(
            source=bgr,
            conf=self._config.confidence_threshold,
            imgsz=self._config.image_size,
            max_det=self._config.max_detections,
            device=self._config.device,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return (), 0

        boxes = results[0].boxes.cpu().numpy()
        normalized_boxes = np.asarray(boxes.xyxyn, dtype=np.float64)
        confidences = np.asarray(boxes.conf, dtype=np.float64)
        candidate_count = len(confidences)
        if candidate_count == 0:
            return (), 0

        observations = []
        for index in np.argsort(-confidences):
            bbox_norm = _normalized_box(normalized_boxes[int(index)])
            if bbox_norm is None:
                continue
            observations.append(TargetObservation(
                TargetVisibility.VISIBLE, bbox_norm=bbox_norm, source="yolo_world",
                confidence=float(confidences[int(index)]),
            ))
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
    "YoloWorldDetector",
    "DEFAULT_YOLO_WORLD_MODEL_PATH",
    "YoloWorldConfig",
]
