"""用 SAM2 把检测器边界框细化为像素掩码。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

DEFAULT_SAM2_CHECKPOINT_PATH = Path(
    "data/models/sam2/sam2.1_hiera_small.pt"
)
DEFAULT_SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


@dataclass(frozen=True)
class Sam2ObserverConfig:
    """SAM2.1 Hiera Small 的本地运行参数。"""

    checkpoint_path: Path = DEFAULT_SAM2_CHECKPOINT_PATH
    model_config: str = DEFAULT_SAM2_MODEL_CONFIG
    device: str = "cuda"


class Sam2BoxSegmenter:
    """把一个归一化边界框转换为同尺寸布尔掩码。"""

    def __init__(
        self,
        config: Sam2ObserverConfig = Sam2ObserverConfig(),
    ) -> None:
        _validate_config(config)
        self._predictor = _build_predictor(config)
        self._image_shape: Optional[Tuple[int, int]] = None

    def set_image(self, rgb: Any) -> Tuple[int, int]:
        """编码一帧 RGB，并返回 (height, width) 供多个 YOLO 框复用。"""
        image = _as_rgb_array(rgb)
        self._predictor.set_image(image)
        self._image_shape = image.shape[:2]
        return self._image_shape

    def segment(
        self,
        rgb: Any,
        bbox_norm: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """返回最佳布尔掩码；没有任何前景像素时返回 None。"""
        self.set_image(rgb)
        return self.segment_box(bbox_norm)

    def segment_box(
        self,
        bbox_norm: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """在最近一次 set_image 的编码上分割一个边界框。"""
        if self._image_shape is None:
            raise RuntimeError("SAM2 尚未设置 RGB 图像")
        height, width = self._image_shape
        x_min, y_min, x_max, y_max = bbox_norm
        box_pixels = np.asarray(
            [
                x_min * width,
                y_min * height,
                x_max * width,
                y_max * height,
            ],
            dtype=np.float32,
        )
        masks, scores, _ = self._predictor.predict(
            box=box_pixels,
            multimask_output=False,
        )
        if len(masks) == 0:
            return None
        best_index = int(np.argmax(scores)) if len(scores) else 0
        mask = np.asarray(masks[best_index], dtype=bool)
        if mask.shape != (height, width):
            raise RuntimeError(
                f"SAM2 掩码尺寸 {mask.shape} 与 RGB {(height, width)} 不一致"
            )
        if not bool(np.any(mask)):
            return None
        return mask


def _build_predictor(config: Sam2ObserverConfig) -> Any:
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"缺少 SAM2 模型文件：{checkpoint_path}")
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境未安装官方 SAM2 及其 PyTorch 依赖"
        ) from exc

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"SAM2 配置使用 {config.device}，但当前 PyTorch 看不到 CUDA"
        )
    try:
        model = build_sam2(
            config.model_config,
            str(checkpoint_path),
            device=config.device,
            mode="eval",
            apply_postprocessing=False,
        )
        return SAM2ImagePredictor(model)
    except Exception as exc:
        raise RuntimeError(f"SAM2 模型加载失败：{_exception_text(exc)}") from exc


def _as_rgb_array(rgb: Any) -> np.ndarray:
    if rgb is None:
        raise ValueError("当前帧没有 RGB 图像")
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB 图像必须是 H×W×3 数组")
    return np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)


def _validate_config(config: Sam2ObserverConfig) -> None:
    if not isinstance(config, Sam2ObserverConfig):
        raise ValueError("config 必须为 Sam2ObserverConfig")
    if not isinstance(config.model_config, str) or not config.model_config.strip():
        raise ValueError("model_config 必须为非空字符串")
    if not isinstance(config.device, str) or not config.device.strip():
        raise ValueError("device 必须为非空字符串")


def _exception_text(exc: Exception) -> str:
    return (str(exc).strip() or exc.__class__.__name__)[:240]


__all__ = [
    "DEFAULT_SAM2_CHECKPOINT_PATH",
    "DEFAULT_SAM2_MODEL_CONFIG",
    "Sam2ObserverConfig",
    "Sam2BoxSegmenter",
]
