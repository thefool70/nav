"""持续运行 YOLO-World，并用 SAM2 验证候选框和生成目标掩码。"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np

from ..core.models import (
    FrontierScoreRequest,
    NavigationFrame,
    TargetConfirmation,
    TargetConfirmationResult,
    TargetObservation,
    TargetSearchGoal,
    TargetVisibility,
)
from .perception import (
    LocalPerceptionEvent,
    ScanObservationContext,
)
from .sam2_observer import Sam2BoxSegmenter, Sam2ObserverConfig


DEFAULT_YOLO_WORLD_MODEL_PATH = Path(
    "data/models/yolo-world/yolov8s-world.pt"
)

LocalPerceptionCallback = Callable[
    [NavigationFrame, LocalPerceptionEvent],
    None,
]


class _SemanticAdvisor(Protocol):
    """YOLO 本地检测之外仍由 VLM 承担的两个低频职责。"""

    def record_scan_frame(
        self,
        frame: NavigationFrame,
        context: ScanObservationContext,
    ) -> None: ...

    def score_frontiers(
        self,
        request: FrontierScoreRequest,
        goal: TargetSearchGoal,
    ) -> Mapping[str, float]: ...

    def confirm_target(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        observation: TargetObservation,
    ) -> TargetConfirmationResult: ...


@dataclass(frozen=True)
class YoloWorldSam2Config:
    """YOLOv8s-World 与 SAM2.1 Hiera Small 的本地推理参数。"""

    class_text: str
    model_path: Path = DEFAULT_YOLO_WORLD_MODEL_PATH
    device: str = "cuda"
    confidence_threshold: float = 0.25
    image_size: int = 640
    max_detections: int = 3
    observation_timeout_s: float = 10.0
    sam2: Sam2ObserverConfig = Sam2ObserverConfig()


@dataclass(frozen=True)
class _DetectionRequest:
    sequence_index: int
    frame: NavigationFrame
    required: bool


@dataclass(frozen=True)
class _DetectionOutcome:
    observation: TargetObservation
    inference_s: float
    candidate_count: int


class YoloWorldSam2TargetObserver:
    """最新帧后台观察器；运动帧可丢弃，决策帧必须等待对应结果。"""

    def __init__(
        self,
        semantic_advisor: _SemanticAdvisor,
        config: YoloWorldSam2Config,
        on_local_perception: Optional[LocalPerceptionCallback] = None,
    ) -> None:
        _validate_config(config)
        self._semantic_advisor = semantic_advisor
        self._config = config
        self._on_local_perception = on_local_perception
        self._detector = _YoloWorldSam2Detector(config)

        self._condition = threading.Condition()
        self._required_request: Optional[_DetectionRequest] = None
        self._latest_motion_request: Optional[_DetectionRequest] = None
        self._required_results: Dict[int, _DetectionOutcome] = {}
        self._abandoned_required_sequences: set[int] = set()
        self._next_sequence_index = 0
        self._closed = False
        self._interrupt_enabled = False
        self._interrupt_after_sequence = 0
        self._interrupt_requested = False
        self._suppress_interrupt_until_clear = False
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="yolo-world-sam2",
            daemon=True,
        )
        self._worker.start()

    def observe(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        scan_context: Optional[ScanObservationContext] = None,
    ) -> TargetObservation:
        """缓存必要的扫描 RGB，并等待当前决策帧的本地检测结果。"""
        if scan_context is not None:
            try:
                self._semantic_advisor.record_scan_frame(frame, scan_context)
            except Exception as exc:
                return _uncertain(f"扫描 RGB 缓存失败：{_exception_text(exc)}")

        sequence_index = self._submit(frame, required=True)
        deadline = time.monotonic() + self._config.observation_timeout_s
        with self._condition:
            while sequence_index not in self._required_results:
                if self._closed:
                    return _uncertain("YOLO-World 后台观察器已经关闭。")
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    pending = self._required_request
                    if (
                        pending is not None
                        and pending.sequence_index == sequence_index
                    ):
                        self._required_request = None
                    else:
                        self._abandoned_required_sequences.add(sequence_index)
                    return _uncertain(
                        "YOLO-World + SAM2 当前帧推理超过 "
                        f"{self._config.observation_timeout_s:.1f} 秒。"
                    )
                self._condition.wait(remaining_s)
            outcome = self._required_results.pop(sequence_index)
        return outcome.observation

    def submit_motion_frame(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
    ) -> None:
        """非阻塞提交运动帧；工作线程繁忙时只保留最新一帧。"""
        self._submit(frame, required=False)

    def score_frontiers(
        self,
        request: FrontierScoreRequest,
        goal: TargetSearchGoal,
    ) -> Mapping[str, float]:
        """Frontier 批量评分继续使用现有 VLM。"""
        return self._semantic_advisor.score_frontiers(request, goal)

    def confirm_target(
        self,
        frame: NavigationFrame,
        goal: TargetSearchGoal,
        observation: TargetObservation,
    ) -> TargetConfirmationResult:
        """只有接近本地候选以后才调用 VLM 最终确认。"""
        result = self._semantic_advisor.confirm_target(frame, goal, observation)
        if result.confirmation is TargetConfirmation.REJECTED:
            with self._condition:
                # 允许机器人先离开刚被否决的误检；看到一帧无目标后自动恢复。
                self._suppress_interrupt_until_clear = True
                self._interrupt_requested = False
        return result

    def set_motion_interrupt_enabled(self, enabled: bool) -> None:
        """从启用时刻之后的新运动帧检测才允许中断 Action。"""
        with self._condition:
            self._interrupt_enabled = bool(enabled)
            self._interrupt_requested = False
            self._interrupt_after_sequence = self._next_sequence_index

    def should_interrupt_motion(self) -> bool:
        with self._condition:
            return self._interrupt_enabled and self._interrupt_requested

    def close(self) -> None:
        """停止后台线程；正在执行的单次推理最多等待一个超时窗口。"""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._on_local_perception = None
            self._required_request = None
            self._latest_motion_request = None
            self._required_results.clear()
            self._abandoned_required_sequences.clear()
            self._condition.notify_all()
        self._worker.join(timeout=self._config.observation_timeout_s)

    def _submit(
        self,
        frame: NavigationFrame,
        required: bool,
    ) -> int:
        with self._condition:
            if self._closed:
                raise RuntimeError("YOLO-World 后台观察器已经关闭")
            self._next_sequence_index += 1
            request = _DetectionRequest(
                self._next_sequence_index,
                frame,
                required,
            )
            if required:
                # 决策帧比此前尚未处理的运动帧更新，旧帧不再有价值。
                self._latest_motion_request = None
                self._required_request = request
            else:
                self._latest_motion_request = request
            self._condition.notify_all()
            return request.sequence_index

    def _worker_loop(self) -> None:
        while True:
            request = self._next_request()
            if request is None:
                return
            started_s = time.monotonic()
            try:
                observation, candidate_count = self._detector.detect(
                    request.frame,
                )
            except Exception as exc:
                observation = _uncertain(
                    f"YOLO-World + SAM2 推理失败：{_exception_text(exc)}"
                )
                candidate_count = 0
            outcome = _DetectionOutcome(
                observation=observation,
                inference_s=time.monotonic() - started_s,
                candidate_count=candidate_count,
            )
            self._publish_outcome(request, outcome)

    def _next_request(self) -> Optional[_DetectionRequest]:
        with self._condition:
            while (
                not self._closed
                and self._required_request is None
                and self._latest_motion_request is None
            ):
                self._condition.wait()
            if self._closed:
                return None
            request = self._required_request
            if request is not None:
                self._required_request = None
                return request
            request = self._latest_motion_request
            self._latest_motion_request = None
            return request

    def _publish_outcome(
        self,
        request: _DetectionRequest,
        outcome: _DetectionOutcome,
    ) -> None:
        with self._condition:
            if self._closed:
                return
            if (
                self._suppress_interrupt_until_clear
                and outcome.observation.visibility
                is TargetVisibility.NOT_VISIBLE
            ):
                self._suppress_interrupt_until_clear = False
            if request.required:
                if (
                    request.sequence_index
                    in self._abandoned_required_sequences
                ):
                    self._abandoned_required_sequences.remove(
                        request.sequence_index
                    )
                else:
                    self._required_results[request.sequence_index] = outcome
            else:
                if (
                    not self._suppress_interrupt_until_clear
                    and self._interrupt_enabled
                    and request.sequence_index > self._interrupt_after_sequence
                    and outcome.observation.visibility
                    is TargetVisibility.VISIBLE
                ):
                    self._interrupt_requested = True
            self._condition.notify_all()

        if request.required or self._on_local_perception is None:
            return
        event = LocalPerceptionEvent(
            sequence_index=request.sequence_index,
            frame_timestamp_s=request.frame.timestamp_s,
            observation=outcome.observation,
            inference_s=outcome.inference_s,
            candidate_count=outcome.candidate_count,
        )
        try:
            self._on_local_perception(request.frame, event)
        except Exception:
            # 可视化和日志是旁路，不能终止连续目标检测。
            pass


class _YoloWorldSam2Detector:
    """单线程拥有 YOLO 与 SAM2 模型，避免并发访问 GPU predictor。"""

    def __init__(self, config: YoloWorldSam2Config) -> None:
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise RuntimeError(f"缺少 YOLO-World 模型文件：{model_path}")
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError("当前 Python 环境未安装 ultralytics") from exc

        try:
            self._model = YOLOWorld(str(model_path), verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"YOLO-World 模型加载失败：{_exception_text(exc)}"
            ) from exc
        self._config = config
        target_text = config.class_text.strip()
        try:
            self._model.set_classes([target_text])
        except Exception as exc:
            raise RuntimeError(
                "YOLO-World 类别初始化失败；请确认 CLIP ViT-B/32 权重可用："
                f"{_exception_text(exc)}"
            ) from exc
        self._segmenter = Sam2BoxSegmenter(config.sam2)

    def detect(
        self,
        frame: NavigationFrame,
    ) -> Tuple[TargetObservation, int]:
        """返回置信度最高且能产生 SAM2 掩码的目标候选。"""
        rgb = _as_rgb_array(frame.rgb)
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
            return _not_visible(), 0

        boxes = results[0].boxes.cpu().numpy()
        normalized_boxes = np.asarray(boxes.xyxyn, dtype=np.float64)
        confidences = np.asarray(boxes.conf, dtype=np.float64)
        candidate_count = len(confidences)
        if candidate_count == 0:
            return _not_visible(), 0

        self._segmenter.set_image(rgb)
        for index in np.argsort(-confidences):
            bbox_norm = _normalized_box(normalized_boxes[int(index)])
            if bbox_norm is None:
                continue
            mask = self._segmenter.segment_box(bbox_norm)
            if mask is None:
                continue
            return (
                TargetObservation(
                    visibility=TargetVisibility.VISIBLE,
                    bbox_norm=bbox_norm,
                    target_mask=mask,
                    source="yolo_world_sam2",
                    confidence=float(confidences[int(index)]),
                ),
                candidate_count,
            )

        return (
            TargetObservation(
                visibility=TargetVisibility.NOT_VISIBLE,
                reason="YOLO-World 候选均未生成有效 SAM2 掩码。",
                source="yolo_world_sam2",
            ),
            candidate_count,
        )


def _normalized_box(values: Any) -> Optional[Tuple[float, float, float, float]]:
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


def _not_visible() -> TargetObservation:
    return TargetObservation(
        visibility=TargetVisibility.NOT_VISIBLE,
        source="yolo_world_sam2",
    )


def _uncertain(reason: str) -> TargetObservation:
    return TargetObservation(
        visibility=TargetVisibility.UNCERTAIN,
        reason=reason,
        source="yolo_world_sam2",
    )


def _validate_config(config: YoloWorldSam2Config) -> None:
    if not isinstance(config, YoloWorldSam2Config):
        raise ValueError("config 必须为 YoloWorldSam2Config")
    if not isinstance(config.device, str) or not config.device.strip():
        raise ValueError("YOLO device 必须为非空字符串")
    if not isinstance(config.class_text, str) or not config.class_text.strip():
        raise ValueError("YOLO class_text 必须为非空字符串")
    confidence = config.confidence_threshold
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) <= 1.0
    ):
        raise ValueError("YOLO confidence_threshold 必须位于 (0, 1]")
    for name in ("image_size", "max_detections"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    timeout = config.observation_timeout_s
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
    ):
        raise ValueError("observation_timeout_s 必须为正有限数")


def _exception_text(exc: Exception) -> str:
    return (str(exc).strip() or exc.__class__.__name__)[:240]


__all__ = [
    "DEFAULT_YOLO_WORLD_MODEL_PATH",
    "YoloWorldSam2Config",
    "YoloWorldSam2TargetObserver",
]
