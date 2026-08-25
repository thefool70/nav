"""用 SAM2 把 VLM 目标框细化为像素掩码。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from ..core.models import (
    FrontierScoreRequest,
    NavigationFrame,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from .perception import ScanObservationContext, TargetBoxObserver


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
    max_box_attempts: int = 2


class Sam2SegmentingTargetObserver:
    """在 VLM 框选之后生成掩码，并在空掩码时请求重新框选。"""

    def __init__(
        self,
        box_observer: TargetBoxObserver,
        config: Sam2ObserverConfig = Sam2ObserverConfig(),
    ) -> None:
        _validate_config(config)
        self._box_observer = box_observer
        self._config = config
        self._predictor = _build_predictor(config)

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        scan_context: Optional[ScanObservationContext] = None,
    ) -> TargetObservation:
        """先取得 VLM 目标框，再用同一帧的 SAM2 掩码确认目标区域。"""
        observation = self._box_observer.observe(frame, goal, scan_context)
        image: Optional[np.ndarray] = None

        for attempt_index in range(self._config.max_box_attempts):
            if observation.visibility is not TargetVisibility.VISIBLE:
                return observation
            if observation.bbox_norm is None:
                return _uncertain("VLM 判定目标可见，但没有返回目标框。")

            try:
                if image is None:
                    image = _as_rgb_array(frame.rgb)
                    self._predictor.set_image(image)
                mask = self._predict_mask(
                    observation.bbox_norm,
                    image.shape[0],
                    image.shape[1],
                )
            except Exception as exc:
                return _uncertain(f"SAM2 分割失败：{_exception_text(exc)}")

            if mask is not None:
                return replace(observation, target_mask=mask)

            if attempt_index + 1 < self._config.max_box_attempts:
                observation = self._box_observer.rebox_visible_target(
                    frame,
                    goal,
                )

        return _uncertain(
            "SAM2 对两次 VLM 目标框都没有生成掩码，等待下一帧重新观察。"
        )

    def score_frontiers(
        self,
        request: FrontierScoreRequest,
        goal: TargetSearchGoal,
    ) -> Mapping[str, float]:
        """Frontier 评分仍完全交给被包装的 VLM 观察器。"""
        return self._box_observer.score_frontiers(request, goal)

    def _predict_mask(
        self,
        bbox_norm: Tuple[float, float, float, float],
        height: int,
        width: int,
    ) -> Optional[np.ndarray]:
        """返回最佳布尔掩码；没有任何前景像素时返回 None。"""
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
    if (
        isinstance(config.max_box_attempts, bool)
        or not isinstance(config.max_box_attempts, int)
        or config.max_box_attempts < 1
    ):
        raise ValueError("max_box_attempts 必须为正整数")


def _uncertain(reason: str) -> TargetObservation:
    return TargetObservation(
        visibility=TargetVisibility.UNCERTAIN,
        reason=reason,
    )


def _exception_text(exc: Exception) -> str:
    return (str(exc).strip() or exc.__class__.__name__)[:240]


__all__ = [
    "DEFAULT_SAM2_CHECKPOINT_PATH",
    "DEFAULT_SAM2_MODEL_CONFIG",
    "Sam2ObserverConfig",
    "Sam2SegmentingTargetObserver",
]
