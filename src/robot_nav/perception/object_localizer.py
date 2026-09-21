"""YOLO/VLM 与 RGB-D 定位；测距失败后用图像方向上的障碍位置保底。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import math
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time

from ..core.models import ObjectLocalization, TargetConfirmation, TargetObservation, TargetVisibility
from ..core.object_grounding import localize_obstacle_on_image_ray, localize_segmented_object
from ..adapters.frontier_overlay import pack_rgb_image
from ..adapters.object_model_process import ObjectModelProcess
from .analyzer import SemanticAnalyzer
from .snapshot_store import write_localization_input, write_localization_result


@dataclass(frozen=True)
class ObjectLocalizerConfig:
    """本地模型环境与单次请求上限；两个进程各自按需加载并跨帧复用。"""

    python_executable: str = sys.executable
    class_text: str = ""
    device: str = "cuda"
    yolo_model: Path = Path("data/models/yolo-world/yolov8s-world.pt")
    sam2_checkpoint: Path = Path("data/models/sam2/sam2.1_hiera_small.pt")
    timeout_s: float = 120.0


class ObjectLocalizer:
    def __init__(self, analyzer: SemanticAnalyzer, config: ObjectLocalizerConfig, directory: Path, on_event=None):
        self._analyzer = analyzer
        self._config = config
        self._directory = directory
        self._on_event = on_event
        self._sequence = 0
        self._yolo = ObjectModelProcess("yolo", config.python_executable, directory / "models", config.timeout_s)
        self._sam2 = ObjectModelProcess("sam2", config.python_executable, directory / "models", config.timeout_s)
        print(f"物体接近模型：{config.python_executable}（{config.device}，YOLO/SAM2 按需加载并复用）", flush=True)

    def locate(self, frame, goal, *, context) -> ObjectLocalization:
        """先完成的有效检测直接进入定位，另一检测的失败或迟到不否决它。"""
        if frame.rgb is None:
            return ObjectLocalization(reason="定位所需的历史 RGB 缺失。")
        self._sequence += 1
        folder = self._directory / f"observation-{self._sequence:04d}"
        folder.mkdir(parents=True, exist_ok=False)
        image = pack_rgb_image(frame.rgb)
        write_localization_input(folder, image, frame, context)
        started = time.monotonic()
        active = threading.Event()
        active.set()
        waiting_model = {"name": None}

        def progress(model, stage, elapsed):
            if active.is_set() and waiting_model["name"] in (None, model):
                print(f"物体定位 {folder.name}: {model}/{stage}，{elapsed:.1f}s", flush=True)
                self._emit({"event": "object_progress", **context, "model": model, "stage": stage,
                            "elapsed_s": time.monotonic() - started, "model_elapsed_s": elapsed,
                            "localization_directory": str(folder)})

        self._emit({"event": "object_localization_started", **context, "pose": asdict(frame.pose),
                    "map_frame_id": frame.obstacle_map.frame_id, "timestamp_s": frame.timestamp_s,
                    "localization_directory": str(folder)})
        payload = {
            "rgb_file": str((folder / "input.rgb.gz").resolve()),
            "width_px": image.width_px, "height_px": image.height_px,
            "class_text": self._config.class_text or goal.target_text, "device": self._config.device,
            "yolo_model": str(self._config.yolo_model.resolve()),
            "sam2_checkpoint": str(self._config.sam2_checkpoint.resolve()),
        }
        detections = Queue()

        def detect_yolo():
            raw = self._yolo.request(payload, folder, progress)
            if raw.get("observations"):
                item = raw["observations"][0]
                return TargetObservation(TargetVisibility.VISIBLE, bbox_norm=tuple(item["bbox_norm"]),
                                         confidence=item["confidence"], source="yolo")
            return TargetObservation(
                TargetVisibility.UNCERTAIN if raw.get("error") else TargetVisibility.NOT_VISIBLE,
                source="yolo", reason=raw.get("error", ""),
            )

        def detect_vlm():
            return self._analyzer.locate_object(frame, goal, context={
                **context, "localization_directory": str(folder),
            })

        for name, call in (("yolo", detect_yolo), ("vlm", detect_vlm)):
            threading.Thread(target=_detect, args=(name, call, detections), daemon=True,
                             name=f"object-{name}-detection").start()
        reasons = []
        confirmation = TargetConfirmation.UNCERTAIN
        negative_count = 0
        boxed_observations = []
        try:
            for _ in range(2):
                waiting_model["name"] = None
                observation = self._wait_detection(detections, started, progress)
                if observation is None:
                    reasons.append("等待目标检测结果超时")
                    break
                if observation.source == "vlm":
                    confirmation = {TargetVisibility.VISIBLE: TargetConfirmation.CONFIRMED,
                                    TargetVisibility.NOT_VISIBLE: TargetConfirmation.REJECTED}.get(
                                        observation.visibility, TargetConfirmation.UNCERTAIN)
                if observation.visibility is not TargetVisibility.VISIBLE or not _valid_bbox(observation.bbox_norm):
                    negative_count += observation.visibility is TargetVisibility.NOT_VISIBLE
                    if observation.reason:
                        reasons.append(observation.reason)
                    continue
                boxed_observations.append(observation)
                if frame.depth is None or frame.camera_intrinsics is None:
                    reasons.append("历史深度或内参缺失，无法使用 RGB-D 测距")
                    continue
                candidate_folder = folder / observation.source
                candidate_folder.mkdir()
                progress(observation.source, "detected", time.monotonic() - started)
                waiting_model["name"] = "sam2"
                segment = self._sam2.request({**payload, "bbox_norm": observation.bbox_norm}, candidate_folder, progress)
                mask = None
                if segment.get("mask_file"):
                    try:
                        with gzip.open(candidate_folder / segment["mask_file"], "rt", encoding="utf-8") as stream:
                            mask = json.load(stream)
                    except (OSError, ValueError, EOFError) as exc:
                        reasons.append(f"SAM2 掩码无法读取，使用检测框：{exc}")
                if segment.get("error"):
                    reasons.append(segment["error"])
                estimate = localize_segmented_object(frame, observation.bbox_norm, mask)
                source = observation.source + ("_sam2" if mask is not None else "_bbox")
                if not estimate.success and mask is not None:
                    estimate = localize_segmented_object(frame, observation.bbox_norm)
                    source = observation.source + "_bbox"
                if estimate.success:
                    return self._save_result(folder, ObjectLocalization(
                        estimate.target_world_xy, TargetVisibility.VISIBLE, confirmation,
                        source, estimate.sample_count, "; ".join(reasons),
                    ), context, bbox_norm=observation.bbox_norm, detector_source=observation.source,
                        localization_method="rgbd", distance_m=estimate.distance_m,
                        bearing_rad=estimate.bearing_rad)
                reasons.append(estimate.reason)
            # 两路检测都不能用深度定位时，才尝试障碍假设；已有框就沿框方向，无框才沿光轴。
            for observation in boxed_observations or (None,):
                bbox = observation.bbox_norm if observation is not None else None
                estimate = localize_obstacle_on_image_ray(frame, bbox)
                if not estimate.success:
                    reasons.append(estimate.reason)
                    continue
                source = "bbox_obstacle" if bbox is not None else "front_obstacle"
                return self._save_result(folder, ObjectLocalization(
                    target_world_xy=estimate.target_world_xy,
                    visibility=TargetVisibility.VISIBLE if bbox is not None else TargetVisibility.UNCERTAIN,
                    vlm_confirmation=confirmation, source=source,
                    reason="; ".join(reasons + [estimate.reason]),
                ), context, bbox_norm=bbox,
                    detector_source=observation.source if observation is not None else None,
                    localization_method="obstacle_assumption", distance_m=estimate.distance_m,
                    bearing_rad=estimate.bearing_rad)
            return self._save_result(folder, ObjectLocalization(
                visibility=TargetVisibility.NOT_VISIBLE if negative_count == 2 else TargetVisibility.UNCERTAIN,
                vlm_confirmation=confirmation, reason="; ".join(reasons) or "历史画面未得到目标位置。",
            ), context)
        finally:
            active.clear()

    def _wait_detection(self, detections, started, progress):
        last_report = time.monotonic()
        while time.monotonic() - started < self._config.timeout_s:
            try:
                return _detection_result(detections.get(timeout=0.2))
            except Empty:
                now = time.monotonic()
                if now - last_report >= 5.0:
                    progress("detectors", "waiting", now - started)
                    last_report = now
        try:
            return _detection_result(detections.get_nowait())
        except Empty:
            return None

    def _save_result(self, folder, result, context, **details):
        record = write_localization_result(folder, result, details)
        record["target_source"] = record.pop("source")
        self._emit({"event": "object_localized", **context, **record, "localization_directory": str(folder)})
        return result

    def _emit(self, event):
        if self._on_event is not None:
            self._on_event(event)

    def close(self):
        self._yolo.close()
        self._sam2.close()


def _detect(name, call, results):
    try:
        observation = call()
    except Exception as exc:
        results.put(exc)
    else:
        results.put(observation)


def _detection_result(value):
    """检测线程的程序错误回到调用线程，不能转成“未找到目标”。"""
    if isinstance(value, Exception):
        raise value
    return value


def _valid_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return False
    return (all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in bbox)
            and bbox[0] < bbox[2] and bbox[1] < bbox[3])
