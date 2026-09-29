"""语义感知策略与异步视觉队列：内存快照的 FIFO 分析。

网络线程只产出结果，不操作底盘，也不修改搜索状态。RGB-D 以有界内存快照交接；扫描帧用于方向评分，视频候选用于目标确认，已入队任务不因 Frontier 失效而丢弃。

运行层在每个周期按以下顺序与本模块交互：

1. :meth:`SemanticPerception.begin_cycle` 接收结果并返回新增覆盖；线索通过 take_target_clue 领取。
2. 搜索核心决策后，运行层按返回的显式请求采集画面、查询评分或定位物体。
3. :meth:`SemanticPerception.pending_counts` 给出在途/待处理与失败计数，
   由搜索核心写入 ``SearchState``；本模块不写搜索状态。
"""

from __future__ import annotations

import base64
import math
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Tuple

from ..core.exploration import FrameFrontierCache
from ..adapters.object_model_process import ObjectModelProcess, ObjectModelConfig
from ..adapters.yoloe import YoloEConfig
from ..core.models import (
    FrontierCandidate,
    FrontierScoreRequest,
    NavigationFrame,
    ObjectLocalization,
    ObservationView,
    SearchMode,
    SemanticAnalysis,
    TargetClue,
    TargetSearchGoal,
    TargetObservation,
    TargetVisibility,
)
from ..core.scan import (
    capture_observation_view,
    frontier_observation_points,
    CaptureContext,
    preview_capture_frontiers,
)
from ..core.timing import TimingSpans, measure_stage
from .analyzer import VlmInputImage, SemanticAnalyzer, DetectionThresholds, TargetMatch, match_target, box_iou
from .object_localizer import localize_object
from .snapshot_store import (
    buffer_scan_image, freeze_depth, CapturedView, BufferedScanImage,
    clue_frame, snapshot_bytes, view_trace,
)

# 仅限制大块 RGB-D/掩码；已接纳的扫描和目标线索不会为了给新视频腾位而被逐出。
MAX_SNAPSHOTS = 256
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024 * 1024
MAX_VIDEO_SNAPSHOTS = 64
MAX_VIDEO_CONFIRMATIONS = 32
# 准备中的 NavigationFrame 还持有地图和 Python 元组，不属于紧凑图像预算。
MAX_SCAN_PREPARATIONS = 32


@dataclass(frozen=True)
class _CompletedAnalysis:
    """一张图的分析结果与轻量元数据；不复制 RGB-D 和掩码。"""

    job_id: int
    result: SemanticAnalysis
    map_frame_id: str
    view: ObservationView
    candidates: Tuple[FrontierCandidate, ...]
    match: TargetMatch
    source: str = "scan"


class SemanticPerception:
    """扫描队列、场景判断、方向打分与物体定位的统一入口。

    两种模式共用扫描队列；线索返回、物体接近与终态期间暂停后台。
    """

    def __init__(
        self,
        analyzer: SemanticAnalyzer,
        goal: TargetSearchGoal,
        *,
        on_event: Optional[Callable[[Mapping[str, Any]], None]] = None,
        directory: Optional[Path] = None,
        object_config: Optional[ObjectModelConfig] = None,
        thresholds: DetectionThresholds = DetectionThresholds(),
    ) -> None:
        if directory is None:
            root = Path("data/run_logs")
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="semantic-", dir=root))
        else:
            directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self._analyzer = analyzer
        self._thresholds = thresholds
        self._yolo = None
        self._on_event = on_event
        self._condition = threading.Condition()
        # 扫描图的打包与 YOLO 检测在后台，避免阻塞下一次转向。
        self._snapshot_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic-snapshot")
        self._closed = False
        self._worker_error: Optional[Exception] = None
        self._goal = goal  # 本次导航的固定目标，在线程启动前确定。
        self._jobs: Deque[Tuple[int, Tuple[FrontierCandidate, ...]]] = deque()
        # 快照由队列持有：未命中立即释放，命中保留至主线程定位完成。
        self._snapshots: Dict[int, CapturedView] = {}
        self._completed: Deque[_CompletedAnalysis] = deque()
        self._pending_coverage: Dict[int, ObservationView] = {}
        self._clues: Deque[TargetClue] = deque()
        self._clue_vantages = set()
        # 同一候选的分数与来源一起写入、读取。
        self._scores: Dict[Tuple[str, str, float, float], Tuple[float, int, Optional[int]]] = {}
        self._received_jobs = []
        self._cycle_score_sources = []
        self._submitted_keys = set()
        # 每个位置/朝向保留一次确认与一次高分命中，不因框的像素抖动重复送 VLM。
        self._video_vantages = set()
        self._video_frames = 0
        self._video_received = 0
        self._video_replaced = 0
        self._video_admission_skipped = 0
        # 相机不断覆盖最新帧；只有检测线程取走时才固定用于一次推理的 RGB-D。
        self._latest_video_frame = None
        self._video_interval_s = 1.0 / object_config.frequency_hz if object_config else 0.1
        self._video_worker = None
        # 扫描准备期间保留覆盖，避免核心误判队列耗尽或重复扫描。
        self._scan_submissions: Dict[Future, ObservationView] = {}
        self._submitted = 0
        self._finished = 0
        self._failed = 0
        self._active_job: Optional[int] = None
        self._background_paused = False
        if object_config is not None:
            self._yolo = ObjectModelProcess(object_config.python_executable,
                                            self.directory / "models", object_config.timeout_s)
            self._yolo_payload = {
                "class_text": object_config.class_text or goal.target_text,
                "yolo_model": str(object_config.yolo_model.resolve()), "device": object_config.device,
                "confidence_threshold": thresholds.yolo_joint,
            }
            try:
                loaded = self._yolo.request({**self._yolo_payload, "initialize": True},
                    lambda model, stage, elapsed: print(f"YOLOE 初始化：{stage} ({elapsed:.1f}s)", flush=True))
                if loaded.get("error"):
                    raise RuntimeError(loaded["error"])
            except BaseException:
                self._yolo.close()
                self._snapshot_worker.shutdown(wait=True)
                raise
            self._video_worker = threading.Thread(target=self._run_worker, args=(self._video_loop,),
                                                  name="yoloe-video", daemon=True)
            self._video_worker.start()
        self._worker = threading.Thread(target=self._run_worker, args=(self._worker_loop,), name="semantic-fifo", daemon=True)
        self._worker.start()
        print(f"异步视觉队列：内存快照（最多 {MAX_SNAPSHOTS} 帧 / "
              f"{MAX_SNAPSHOT_BYTES // (1024 * 1024)} MiB）", flush=True)
        if self._yolo is not None:
            print(f"YOLOE 视频上限 {1 / self._video_interval_s:g} Hz，日志：{self.directory / 'models/yoloe.log'}", flush=True)

    # ------------------------------------------------------------------
    # 运行层直接调用的公开边界
    # ------------------------------------------------------------------

    def begin_cycle(
        self, frame: NavigationFrame,
    ) -> Tuple[ObservationView, ...]:
        """接收已完成分析，返回新增覆盖；评分写入缓存，线索留在 FIFO 等待领取。"""
        self._raise_worker_error()
        self._received_jobs = []
        self._cycle_score_sources = []
        with self._condition:
            completed = () if self._background_paused else tuple(self._completed)
            if completed:
                self._completed.clear()
            for item in completed:
                self._pending_coverage.pop(item.job_id, None)
        views = []
        timestamps = set()
        for item in completed:
            view = item.view
            value = item.result.image_score
            if value is not None and view is not None:
                for candidate in item.candidates:
                    key = _target_key(item.map_frame_id, candidate.world_xy, candidate.candidate_id)
                    self._scores.setdefault(key, (value, item.job_id, item.result.interaction_id))
            job_clues = ()
            if view is not None and item.map_frame_id == frame.obstacle_map.frame_id:
                # 检测成功才登记覆盖；评分或检测任一失败，不丢弃另一项有效结果。
                if item.source == "scan" and item.result.found is not None and view.timestamp_s not in timestamps:
                    views.append(view)
                    timestamps.add(view.timestamp_s)
                if item.match.found:
                    # 物体线索按单图任务去重；场景线索仍按拍摄位置和朝向去重。
                    key = (
                        (item.map_frame_id, view.timestamp_s)
                        if self._goal.search_mode is SearchMode.OBJECT
                        else _vantage_key(item.map_frame_id, view.pose)
                    )
                    if key not in self._clue_vantages:
                        self._clue_vantages.add(key)
                        clue = TargetClue(
                            f"semantic:{item.job_id}:1", view.pose, view.timestamp_s, item.map_frame_id,
                            job_id=item.job_id, view_id=1, bbox_norm=item.match.bbox_norm,
                            source=item.match.source, vlm_confidence=item.match.vlm_confidence,
                            yolo_confidence=item.match.yolo_confidence,
                        )
                        self._clues.append(clue)
                        job_clues = (clue.clue_id,)
            if not job_clues or self._goal.search_mode is not SearchMode.OBJECT:
                with self._condition:
                    self._snapshots.pop(item.job_id, None)
            self._received_jobs.append({
                "job_id": item.job_id, "interaction_id": item.result.interaction_id,
                "clue_ids": job_clues, **asdict(item.match),
            })
        return tuple(views)

    def pause_for_target_handling(self, paused: bool, reason: str = "") -> None:
        """搜索核心进入/退出目标处理时显式告知后台暂停状态。"""
        with self._condition:
            if paused != self._background_paused:
                self._background_paused = bool(paused)
                self._condition.notify_all()

    def has_target_clues(self) -> bool:
        return bool(self._clues)

    def take_target_clue(self, *, busy: bool) -> Optional[TargetClue]:
        """按结果可用顺序领取线索；已有目标处理期间不消费下一条。"""
        if busy:
            return None
        return self._clues.popleft() if self._clues else None

    def pending_counts(self) -> Tuple[int, int, Tuple[ObservationView, ...]]:
        """返回（在途与待处理任务数、分析失败批数、待分析覆盖）供核心写入状态。"""
        with self._condition:
            self._raise_worker_error()
            # 快照已接纳但 queued 事件仍在输出时，也属于在途任务。
            pending_ids = set(self._snapshots)
            pending_ids.update(job_id for job_id, _ in self._jobs)
            pending_ids.update(item.job_id for item in self._completed)
            pending_ids.update(clue.job_id for clue in self._clues)
            if self._active_job is not None:
                pending_ids.add(self._active_job)
            pending = len(pending_ids) + len(self._scan_submissions)
            failed = self._failed
            pending_views = tuple(self._pending_coverage.values()) + tuple(
                view for view in self._scan_submissions.values() if view is not None)
        return pending, failed, pending_views

    @property
    def detects_video(self) -> bool:
        return self._yolo is not None

    def observe_rgbd_frame(self, *, rgb, depth_m, intrinsics, pose, timestamp_s,
                           extrinsics, map_frame_id) -> None:
        """相机交付只读 RGB-D；推理线程限频取最新帧，不修改导航状态或取消运动。"""
        if self._yolo is None or self._closed:
            return
        self._raise_worker_error()
        with self._condition:
            self._video_received += 1
            if self._latest_video_frame is not None:
                self._video_replaced += 1
            self._latest_video_frame = (rgb, depth_m, intrinsics, pose, timestamp_s,
                                        extrinsics, map_frame_id, time.monotonic())
            self._condition.notify_all()

    def observe_navigation_frame(self, frame: NavigationFrame) -> None:
        """仿真每个传感器帧也走相同检测入口；地图几何仍只在扫描阶段计算。"""
        if self._yolo is None:
            return
        import numpy as np
        self.observe_rgbd_frame(rgb=np.asarray(frame.rgb, dtype=np.uint8), depth_m=frame.depth,
            intrinsics=frame.camera_intrinsics, pose=frame.pose, timestamp_s=frame.timestamp_s,
            extrinsics=frame.camera_extrinsics_in_robot, map_frame_id=frame.obstacle_map.frame_id)

    def capture_scan_view(
        self,
        frame: NavigationFrame,
        *,
        context: CaptureContext,
        timings: Optional[TimingSpans] = None,
        frontier_cache: Optional[FrameFrontierCache] = None,
    ) -> ObservationView:
        """用同次提交的帧与上下文固定画面和覆盖；打包和 YOLO 检测在后台完成。"""
        candidates, coverage = self._prepare_capture(
            frame, context=context,
            timings=timings, frontier_cache=frontier_cache,
        )
        with self._condition:
            self._raise_worker_error()
            if len(self._scan_submissions) >= MAX_SCAN_PREPARATIONS:
                raise RuntimeError("扫描准备队列已满，拒绝继续采集以免内存无界增长")
            # NavigationFrame 的 RGB-D 和地图是不可变元组；保留这帧即可固定拍摄输入。
            submitted = self._snapshot_worker.submit(self._prepare_scan, frame, candidates, coverage)
            self._scan_submissions[submitted] = coverage
            submitted.add_done_callback(self._scan_work_done)
        return coverage

    def score_frontiers(
        self, map_frame_id: str, request: FrontierScoreRequest,
    ) -> Mapping[str, float]:
        """只查相同世界目标的已完成评分；候选有效性由核心的新地图筛选保证。"""
        scores = {}
        for candidate in request.candidates:
            key = _target_key(map_frame_id, candidate.world_xy, candidate.candidate_id)
            if key not in self._scores:
                continue
            score, job_id, interaction_id = self._scores[key]
            scores[candidate.candidate_id] = score
            self._cycle_score_sources.append({
                "candidate_id": candidate.candidate_id, "score": score,
                "job_id": job_id, "interaction_id": interaction_id,
            })
        return scores

    def localize_object(
        self, frame: NavigationFrame, goal: TargetSearchGoal, clue: TargetClue,
    ) -> ObjectLocalization:
        """历史 RGB-D 优先定位，障碍保底查询当前地图；保留两者各自的时间与位姿。"""
        with self._condition:
            view = self._snapshots.pop(clue.job_id, None)
        if view is None:
            raise RuntimeError(f"目标线索 {clue.clue_id} 缺少内存快照")
        if view.map_frame_id != frame.obstacle_map.frame_id:
            return ObjectLocalization(reason="历史图像与当前地图坐标系不同，继续下一条线索。")
        observation_frame = clue_frame(view, frame)
        matching = max(view.yolo_observations,
                       key=lambda item: box_iou(clue.bbox_norm, item.bbox_norm), default=None)
        overlap = box_iou(clue.bbox_norm, matching.bbox_norm) if matching is not None else 0.0
        mask = (matching.target_mask if matching is not None
                and overlap >= self._thresholds.box_iou else None)
        context = {
            "clue_id": clue.clue_id, "job_id": clue.job_id, "view_id": clue.view_id,
            "source": "object_snapshot",
            "detector_source": clue.source, "vlm_confidence": clue.vlm_confidence,
            "yolo_confidence": clue.yolo_confidence,
            "robot_pose": asdict(frame.pose),
            "localization_map": "full_navigation" if frame.navigation_map is not None else "exploration",
            "map_timestamp_s": frame.timestamp_s,
            "mask_available": mask is not None, "mask_box_iou": overlap,
            "mask_bbox_norm": matching.bbox_norm if mask is not None else None,
        }
        return localize_object(observation_frame, bbox_norm=clue.bbox_norm, source=clue.source,
                               mask=mask, context=context, on_event=self._emit)

    def diagnostics(self) -> Mapping[str, Any]:
        with self._condition:
            return {
                "semantic_submitted": self._submitted, "semantic_finished": self._finished,
                "semantic_failed": self._failed, "semantic_queued": len(self._jobs),
                "semantic_active_job": self._active_job, "semantic_snapshot_storage": "memory",
                "semantic_snapshot_count": len(self._snapshots),
                "semantic_snapshot_bytes": sum(snapshot_bytes(view) for view in self._snapshots.values()),
                "semantic_clues_waiting": len(self._clues),
                "semantic_background_paused": self._background_paused,
                "semantic_received_jobs": tuple(self._received_jobs),
                "semantic_score_sources": tuple(self._cycle_score_sources),
                "yolo_video_frames": self._video_frames,
                "yolo_video_backlog": int(self._latest_video_frame is not None),
                "yolo_video_frequency_hz": 1 / self._video_interval_s,
                "yolo_video_received": self._video_received,
                "yolo_video_replaced": self._video_replaced,
                "yolo_video_admission_skipped": self._video_admission_skipped,
            }

    def wait_for_result(self, timeout_s: float = 1.0) -> None:
        """无完成结果时短暂等待通知；后台异常在等待前后传回导航线程。"""
        with self._condition:
            self._raise_worker_error()
            if not self._completed and not self._closed:
                self._condition.wait(timeout_s)
            self._raise_worker_error()

    def close(self) -> None:
        """停止模型进程和准备线程，释放快照；在途 HTTP 只有限等待。"""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            stopped_jobs = tuple(job_id for job_id, _ in self._jobs)
            if self._active_job is not None:
                stopped_jobs += (self._active_job,)
            self._jobs.clear()
            self._latest_video_frame = None
            self._condition.notify_all()
        if self._yolo is not None:
            self._yolo.close()
        if self._video_worker is not None:
            self._video_worker.join()
        self._snapshot_worker.shutdown(wait=True)
        self._worker.join(timeout=1.0)
        with self._condition:
            self._snapshots.clear()
            self._completed.clear()
            self._pending_coverage.clear()
            self._clues.clear()
        self._emit({"event": "stopped", "job_ids": stopped_jobs})
        self._raise_worker_error()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------------
    # 内部：采集与队列
    # ------------------------------------------------------------------

    def _video_loop(self):
        next_detection_s = 0.0
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._latest_video_frame is not None)
                if self._closed:
                    return
                delay_s = next_detection_s - time.monotonic()
                if delay_s > 0:
                    self._condition.wait(timeout=delay_s)
                    continue
                frame = self._latest_video_frame
                self._latest_video_frame = None
                next_detection_s = time.monotonic() + self._video_interval_s
            self._detect_video_frame(*frame)

    def _detect_video_frame(self, rgb, depth_m, intrinsics, pose, timestamp_s,
                            extrinsics, map_frame_id, queued_s):
        """高分直接发布，中等分有界送 VLM；所有定位依据来自这次推理的同帧 RGB-D。"""
        image = VlmInputImage(rgb.shape[1], rgb.shape[0], rgb.tobytes())
        started = time.monotonic()
        observations = self._detect_yolo(image)
        self._video_frames += 1
        best = max(observations, key=lambda item: item.confidence, default=None)
        high = best is not None and best.confidence >= self._thresholds.yolo_high
        duration_s = time.monotonic() - started
        if best is not None:
            key = (_vantage_key(map_frame_id, pose), high)
            with self._condition:
                admit = not (self._closed or self._background_paused or key in self._video_vantages)
                videos = [view for view in self._snapshots.values() if view.source == "video"]
                confirmations = sum(self._snapshots[job].source == "video" for job, _ in self._jobs)
                if self._active_job is not None:
                    active = self._snapshots.get(self._active_job)
                    confirmations += int(active is not None and active.source == "video")
                if admit and (len(videos) >= MAX_VIDEO_SNAPSHOTS
                              or (not high and confirmations >= MAX_VIDEO_CONFIRMATIONS)):
                    self._video_admission_skipped += 1
                    admit = False
            if admit:
                buffered = BufferedScanImage(image.width_px, image.height_px, image.rgb_bytes, pose,
                                             intrinsics, extrinsics.yaw_rad, extrinsics)
                fov = math.atan2(intrinsics.cx + 0.5, intrinsics.fx) + math.atan2(
                    image.width_px - 0.5 - intrinsics.cx, intrinsics.fx)
                coverage = ObservationView(pose, pose.yaw_rad + extrinsics.yaw_rad, fov, timestamp_s)
                captured = CapturedView(buffered, coverage, map_frame_id,
                    freeze_depth(depth_m, image.width_px, image.height_px), source="video",
                    yolo_observations=observations)
                direct = (match_target(SemanticAnalysis(None), observations, self._thresholds,
                                       self._goal.search_mode) if high else None)
                if self._enqueue_snapshot(captured, (), direct_match=direct):
                    self._video_vantages.add(key)
        # 先发布命中，再记录耗时，避免日志/可视化延迟阻塞高分交付。
        self._emit({"event": "yolo_frame", "timestamp_s": timestamp_s,
                    "frame_count": self._video_frames, "duration_s": duration_s,
                    "queue_wait_s": started - queued_s, "backlog": int(self._latest_video_frame is not None),
                    "received": self._video_received, "replaced": self._video_replaced,
                    "yolo_confidence": best.confidence if best is not None else 0.0,
                    "bbox_norm": best.bbox_norm if best is not None else None, "high": high})

    def _prepare_capture(self, frame, *, context, timings=None, frontier_cache=None):
        """计算主线程推进扫描所需的候选与待检查覆盖，不编码图片。"""
        if frontier_cache is None:
            frontier_cache = FrameFrontierCache(frame)
        with measure_stage(timings, "snapshot.frontier_preview"):
            frontiers = preview_capture_frontiers(
                frame, context, timings=timings, frontier_cache=frontier_cache,
            )
        with measure_stage(timings, "snapshot.observation_points"):
            # 观察完整边界，不能因为某处不适合移动过去就放弃拍摄。
            points = frontier_observation_points(
                frame, frontiers.boundary_cells,
            )
            visible_points = {_xy_key(point) for point in points}
            candidates = tuple(
                item for item in frontiers.candidates
                if item.deferred_order is None and _xy_key(item.world_xy) in visible_points
            )
        with measure_stage(timings, "snapshot.coverage"):
            coverage = capture_observation_view(
                frame, observation_points=context.observation_points + points)
        # 图片分数赋给同一相机视锥内、地图视线可见的候选；不再要求地面像素深度匹配。
        in_view = {_xy_key(point) for point in coverage.map_visible_world_xy}
        return tuple(item for item in candidates if _xy_key(item.world_xy) in in_view), coverage

    def _prepare_scan(self, frame, candidates, coverage):
        """冻结扫描图后入队；YOLO 在 VLM 线程上对这张图检测，准备线程不等待模型。"""
        timings = []
        with measure_stage(timings, "snapshot.image_pack"):
            image = buffer_scan_image(frame)
        with measure_stage(timings, "snapshot.depth_copy"):
            depth = freeze_depth(frame.depth, image.width_px, image.height_px)
        captured = CapturedView(image, coverage, frame.obstacle_map.frame_id, depth)
        timestamp_s = frame.timestamp_s
        del frame  # 不在日志/可视化输出期间继续持有原始元组图像和整幅地图。
        self._enqueue_snapshot(captured, candidates)
        self._emit({"event": "scan_prepared", "timestamp_s": timestamp_s, "spans": timings})

    def _detect_yolo(self, image) -> Tuple[TargetObservation, ...]:
        """扫描和视频共用常驻 YOLOE；RGB 字节和位压缩掩码通过管道交接。"""
        if self._yolo is None:
            return ()
        import numpy as np
        raw = self._yolo.request({**self._yolo_payload, "width_px": image.width_px,
                                 "height_px": image.height_px}, rgb_bytes=image.rgb_bytes)
        if raw.get("error"):
            if self._closed:
                return ()
            raise RuntimeError(f"YOLOE 检测失败：{raw['error']}")
        observations = []
        pixels = image.width_px * image.height_px
        for item in raw["observations"]:
            mask = None
            if item["mask_bits"] is not None:
                bits = base64.b64decode(item["mask_bits"], validate=True)
                if len(bits) != (pixels + 7) // 8:
                    raise ValueError("YOLOE 掩码长度与原始图像尺寸不一致")
                mask = np.unpackbits(np.frombuffer(bits, dtype=np.uint8), count=pixels).reshape(
                    image.height_px, image.width_px).astype(bool)
                mask.flags.writeable = False
            observations.append(TargetObservation(TargetVisibility.VISIBLE, bbox_norm=tuple(item["bbox_norm"]),
                                confidence=item["confidence"], source="yoloe", target_mask=mask))
        return tuple(observations)

    def _scan_work_done(self, future):
        """准备失败传回主线程，不能把未入队的扫描误当作已检查。"""
        error = future.exception()
        with self._condition:
            self._scan_submissions.pop(future, None)
            if error is not None and not self._closed and self._worker_error is None:
                self._worker_error = error
            self._condition.notify_all()

    def _enqueue_snapshot(self, view: CapturedView, candidates, *, direct_match=None) -> bool:
        """内存接纳后保留到结果消费；视频满额跳过新候选，扫描满额明确报告。"""
        key = view.map_frame_id, view.coverage.timestamp_s, view.source
        with self._condition:
            if self._closed or key in self._submitted_keys or (view.source == "video" and self._background_paused):
                return False
            # 扫描入队时尚无掩码，按模型最多输出的实例数预留，后续检测不能突破预算。
            def reserved_bytes(item):
                return item.image.width_px * item.image.height_px * (3 + 4 + YoloEConfig.max_detections)
            total_bytes = sum(reserved_bytes(item) for item in self._snapshots.values())
            if len(self._snapshots) >= MAX_SNAPSHOTS or total_bytes + reserved_bytes(view) > MAX_SNAPSHOT_BYTES:
                if view.source == "video":
                    self._video_admission_skipped += 1
                    return False
                raise RuntimeError("扫描内存快照已达上限；保留现有任务，停止继续采集")
            self._submitted += 1
            job_id = self._submitted
            self._submitted_keys.add(key)
            self._snapshots[job_id] = view
            if view.source == "scan":
                self._pending_coverage[job_id] = view.coverage
            if direct_match is not None:
                ready_s = time.monotonic()
                self._finished += 1
                self._completed.append(_CompletedAnalysis(job_id, SemanticAnalysis(None), view.map_frame_id,
                    view.coverage, (), direct_match, view.source))
                self._condition.notify_all()
        self._emit({"event": "queued", "job_id": job_id, "source": view.source,
                    "storage": "memory", "view_count": 1, "views": (view_trace(view),), "image": view.image,
                    "width_px": view.image.width_px, "height_px": view.image.height_px,
                    "intrinsics": asdict(view.image.intrinsics),
                    "extrinsics": asdict(view.image.camera_extrinsics_in_robot),
                    "depth_available": view.depth is not None,
                    "candidates": tuple({"candidate_id": f"snapshot:{job_id}:{index}",
                                         "score": item.score} for index, item in enumerate(candidates, 1)),
                    "candidate_count": len(candidates)})
        if direct_match is not None:
            self._emit({"event": "completed", "job_id": job_id, "found": True,
                        "bbox_norm": direct_match.bbox_norm, "detection": asdict(direct_match),
                        "ready_monotonic_s": ready_s})
        else:
            with self._condition:
                if not self._closed:
                    self._jobs.append((job_id, candidates))
                    self._condition.notify_all()
        return True

    def _worker_loop(self) -> None:
        """FIFO 读取内存画面并分析；没有命中的大块图像在这里释放。"""
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._worker_error is not None or (not self._background_paused and bool(self._jobs)))
                if self._closed or self._worker_error is not None:
                    return
                job_id, candidates = self._jobs.popleft()
                self._active_job = job_id
                view = self._snapshots[job_id]
            self._emit({"event": "started", "job_id": job_id})
            image = view.image
            if view.source == "scan":
                view = replace(view, yolo_observations=self._detect_yolo(image))
                with self._condition:
                    if self._closed:
                        return
                    self._snapshots[job_id] = view
            job_result = self._analyzer.analyze_view(
                VlmInputImage(image.width_px, image.height_px, image.rgb_bytes), self._goal,
                trace_context={"job_id": job_id, "source": view.source, "storage": "memory",
                    "views": (view_trace(view),),
                    "markers": tuple({"label": f"F{index}", "candidate_id": f"snapshot:{job_id}:{index}",
                                      "region_id": item.candidate_id, "world_xy": item.world_xy, "view_id": 1}
                                     for index, item in enumerate(candidates, 1))})
            matched = match_target(job_result, view.yolo_observations, self._thresholds, self._goal.search_mode)
            ready_s = time.monotonic()
            with self._condition:
                self._active_job = None
                self._finished += 1
                if job_result.found is None:
                    self._failed += 1
                if not self._closed:
                    self._completed.append(_CompletedAnalysis(job_id, job_result, view.map_frame_id,
                        view.coverage, candidates, matched, view.source))
                if not matched.found or self._goal.search_mode is not SearchMode.OBJECT or self._closed:
                    self._snapshots.pop(job_id, None)
                self._condition.notify_all()
            result_record = {**asdict(job_result), "found": matched.found, "bbox_norm": matched.bbox_norm,
                             "vlm": asdict(job_result), "detection": asdict(matched),
                             "ready_monotonic_s": ready_s}
            # 不让 Rerun/JSONL 的输出成为主线程接收结果的前置条件。
            self._emit({"event": "completed", "job_id": job_id, **result_record})
            del view, image

    def _run_worker(self, work) -> None:
        """跨线程传递未预期异常；主循环重新抛出原异常，不伪造模型失败。"""
        try:
            work()
        except Exception as exc:
            traceback.print_exc()
            with self._condition:
                # 保留最初异常，唤醒等待者，由主线程取结果时重新抛出。
                if self._worker_error is None:
                    self._worker_error = exc
                self._condition.notify_all()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise self._worker_error

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(event)


def _xy_key(point):
    return round(point[0], 6), round(point[1], 6)


def _target_key(map_id, point, region_id):
    return (map_id, region_id) + _xy_key(point)


def _vantage_key(map_id, pose):
    return map_id, round(pose.x_m / 0.5), round(pose.y_m / 0.5), round(pose.yaw_rad / math.radians(15.0))


__all__ = ["SemanticPerception"]
