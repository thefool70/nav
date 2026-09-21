"""语义感知策略与异步视觉队列：固定快照的 FIFO 分析。

网络线程只产出结果，不操作底盘，也不修改搜索状态。图片与元数据写入独立运行
目录，预采样只覆盖尚未选取的运动帧；已入队任务不覆盖、不因 Frontier 失效而丢弃。

运行层在每个周期按以下顺序与本模块交互：

1. :meth:`SemanticPerception.begin_cycle` 取回已完成结果，得到新的已检查覆盖与线索。
2. 搜索核心决策后，运行层按返回的显式请求调用对应方法（拍摄、取线索、定位等）。
3. :meth:`SemanticPerception.pending_counts` 给出在途/待处理与失败计数，
   由搜索核心写入 ``SearchState``；本模块不写搜索状态。
"""

from __future__ import annotations

import json
import math
import tempfile
import threading
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Tuple

from ..core.frontier import FrameFrontierCache
from ..core.models import (
    FrontierCandidate,
    FrontierScoreRequest,
    NavigationFrame,
    ObjectLocalization,
    ObservationView,
    Pose2D,
    SearchMode,
    SemanticAnalysis,
    TargetClue,
    TargetSearchGoal,
)
from ..core.scan_behavior import capture_semantic_view
from ..core.perception_flow import CaptureContext, preview_capture_candidates
from ..core.observation_coverage import frontier_observation_points
from ..core.timing import TimingSpans, measure_stage
from ..adapters.frontier_overlay import (
    buffer_scan_image,
    has_frontier_direction_in_view,
    visible_frontier_candidates,
)
from .object_localizer import ObjectLocalizer, ObjectLocalizerConfig
from ..adapters.snapshot_depth import encode_depth
from .analyzer import SemanticAnalyzer
from .snapshot_store import (
    CapturedView,
    read_clue_frame,
    read_snapshot,
    retain_clue_depth,
    view_trace,
    write_snapshot,
)

PREFETCH_TRANSLATION_M = 0.75
PREFETCH_TURN_RAD = math.radians(30.0)
PREFETCH_MAX_YAW_SPEED_RAD_S = math.radians(25.0)


@dataclass(frozen=True)
class SemanticPerceptionConfig:
    """感知模块的可选配置；缺省时按运行目录前缀与本地模型参数推断。"""

    directory: Optional[Path] = None
    object_config: Optional[ObjectLocalizerConfig] = None
    on_event: Optional[Callable[[Mapping[str, Any]], None]] = None


@dataclass(frozen=True)
class PerceptionPause:
    """搜索核心通过显式请求告知感知模块暂停或继续后台任务。"""

    paused: bool
    reason: str = ""


@dataclass(frozen=True)
class CycleIntake:
    """一次周期取回的已完成分析：新的已检查覆盖与目标线索。"""

    observed_views: Tuple[ObservationView, ...]
    clues: Tuple[TargetClue, ...]


@dataclass(frozen=True)
class _CompletedAnalysis:
    job_id: int
    result: SemanticAnalysis
    views: Tuple[Tuple[str, ObservationView], ...]
    candidates: Tuple[FrontierCandidate, ...]
    region_ids: Tuple[str, ...]


class SemanticPerception:
    """扫描队列、场景判断、方向打分与物体定位的统一入口。

    两种模式共用扫描队列；线索返回、物体接近与终态期间暂停后台。
    """

    def __init__(
        self,
        analyzer: SemanticAnalyzer,
        *,
        on_event: Optional[Callable[[Mapping[str, Any]], None]] = None,
        directory: Optional[Path] = None,
        object_config: Optional[ObjectLocalizerConfig] = None,
    ) -> None:
        if directory is None:
            root = Path("data/run_logs")
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="semantic-", dir=root))
        else:
            directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self._analyzer = analyzer
        self._object_localizer = (
            ObjectLocalizer(analyzer, object_config, self.directory / "object-localization", on_event=self._emit)
            if object_config is not None else None
        )
        self._on_event = on_event
        self._condition = threading.Condition()
        self._enqueue_lock = threading.Lock()
        self._closed = False
        self._worker_error: Optional[Exception] = None
        self._goal: Optional[TargetSearchGoal] = None
        self._jobs: Deque[Tuple[int, Path]] = deque()
        self._completed: Deque[_CompletedAnalysis] = deque()
        self._pending_coverage: Dict[int, Tuple[ObservationView, ...]] = {}
        self._clues: Deque[TargetClue] = deque()
        self._clue_vantages = set()
        self._scores: Dict[Tuple[str, str, float, float], float] = {}
        self._score_origins: Dict[Tuple[str, str, float, float], Tuple[int, Optional[int]]] = {}
        self._received_jobs = []
        self._cycle_score_sources = []
        self._submitted_keys = set()
        self._scored_targets = set()
        self._scan_views = []
        self._scan_candidates = {}
        self._submitted = 0
        self._finished = 0
        self._failed = 0
        self._writing = 0
        self._active_job: Optional[int] = None
        self._background_paused = False
        self._prefetch_enabled = False
        self._pending_frame: Optional[Tuple[NavigationFrame, CaptureContext]] = None
        self._capture_busy = False
        self._last_prefetch_pose: Optional[Pose2D] = None
        self._previous_motion_frame: Optional[NavigationFrame] = None
        self._capture_context = None
        self._obstacle_frame_id: str = ""
        self._worker = threading.Thread(target=self._run_worker, args=(self._worker_loop,), name="semantic-fifo", daemon=True)
        self._capture_worker = threading.Thread(target=self._run_worker, args=(self._capture_loop,), name="semantic-capture", daemon=True)
        self._worker.start()
        self._capture_worker.start()
        print(f"异步视觉队列：{self.directory}（固定快照，FIFO）", flush=True)

    # ------------------------------------------------------------------
    # 运行层直接调用的公开边界
    # ------------------------------------------------------------------

    def begin_cycle(
        self, frame: NavigationFrame,
    ) -> CycleIntake:
        """取回已完成分析：给出新的已检查覆盖与目标线索；不修改搜索状态。"""
        self._raise_worker_error()
        self._received_jobs = []
        self._cycle_score_sources = []
        self._obstacle_frame_id = frame.obstacle_map.frame_id
        with self._condition:
            completed = () if self._background_paused else tuple(self._completed)
            if completed:
                self._completed.clear()
            for item in completed:
                self._pending_coverage.pop(item.job_id, None)
        views = []
        timestamps = {view.timestamp_s for view in views}
        new_clues: list[TargetClue] = []
        for item in completed:
            for candidate, region_id in zip(item.candidates, item.region_ids):
                value = item.result.frontier_scores.get(candidate.candidate_id)
                if value is not None and item.views:
                    key = _target_key(item.views[0][0], candidate.world_xy, region_id)
                    self._scores[key] = value
                    self._score_origins[key] = (item.job_id, item.result.interaction_id)
            if item.result.target_view_ids is not None:
                for map_id, view in item.views:
                    if map_id == frame.obstacle_map.frame_id and view.timestamp_s not in timestamps:
                        views.append(view)
                        timestamps.add(view.timestamp_s)
            job_clues = []
            for view_id in item.result.target_view_ids or ():
                index = view_id - 1
                if 0 <= index < len(item.views):
                    map_id, view = item.views[index]
                    key = (
                        (item.job_id, view_id)
                        if self._goal is not None and self._goal.search_mode is SearchMode.OBJECT
                        else _vantage_key(map_id, view.pose)
                    )
                    if map_id == frame.obstacle_map.frame_id and key not in self._clue_vantages:
                        self._clue_vantages.add(key)
                        clue = TargetClue(
                            f"semantic:{item.job_id}:{index + 1}", view.pose, view.timestamp_s, map_id,
                            job_id=item.job_id, view_id=index + 1,
                        )
                        self._clues.append(clue)
                        new_clues.append(clue)
                        job_clues.append(clue.clue_id)
            self._received_jobs.append({
                "job_id": item.job_id, "interaction_id": item.result.interaction_id,
                "clue_ids": tuple(job_clues), "target_view_ids": item.result.target_view_ids,
            })
        return CycleIntake(observed_views=tuple(views), clues=tuple(new_clues))

    def pause_for_target_handling(self, paused: bool, reason: str = "") -> None:
        """搜索核心进入/退出目标处理时显式告知后台暂停状态。"""
        with self._condition:
            if paused != self._background_paused:
                self._background_paused = bool(paused)
                if paused:
                    self._prefetch_enabled = False
                self._condition.notify_all()

    def bind_frame(self, frame: NavigationFrame, context: CaptureContext) -> None:
        """发布最新的机器人观测帧，供评分读取当前地图坐标系。"""
        with self._condition:
            self._obstacle_frame_id = frame.obstacle_map.frame_id
            self._capture_context = context

    def has_target_clues(self) -> bool:
        return bool(self._clues)

    def flush_scan(self) -> None:
        """扫描结束或被目标处理打断时保留已采集的部分批次。"""
        self._flush_scan()

    def set_goal(self, goal: TargetSearchGoal) -> None:
        """登记本次运行的搜索目标；同一队列不能混用不同目标。"""
        with self._condition:
            if self._goal is not None and self._goal != goal:
                raise ValueError("同一视觉队列不能混用不同搜索目标")
            self._goal = goal

    def take_target_clue(self, *, busy: bool) -> Optional[TargetClue]:
        """批次按 FIFO，批内按模型列表顺序；已有目标处理期间不消费下一条。"""
        if busy:
            return None
        return self._clues.popleft() if self._clues else None

    def pending_counts(self) -> Tuple[int, int, Tuple[ObservationView, ...]]:
        """返回（在途与待处理任务数、分析失败批数、待分析覆盖）供核心写入状态。"""
        with self._condition:
            pending = (
                len(self._jobs) + len(self._completed) + self._writing
                + int(self._active_job is not None) + int(self._capture_busy)
                + int(bool(self._scan_views)) + len(self._clues)
                + int(self._pending_frame is not None)
            )
            failed = self._failed
            pending_views = tuple(view for views in self._pending_coverage.values() for view in views)
            pending_views += tuple(item.coverage for item in self._scan_views)
        return pending, failed, pending_views

    def capture_scan_view(
        self,
        frame: NavigationFrame,
        scan_context,
        *,
        timings: Optional[TimingSpans] = None,
        frontier_cache: Optional[FrameFrontierCache] = None,
    ) -> str:
        """采集本轮扫描的一个方向；返回 'submitted' 或 'buffered'；正式采集失败向上传播。"""
        if scan_context.index == 0:
            # 扫描被打断后重建计划时，先提交上一轮已拍到的部分画面。
            with measure_stage(timings, "snapshot.flush_previous"):
                self._flush_scan()
        captured, candidates, _ = self._capture(
            frame, context=self._capture_context,
            timings=timings, frontier_cache=frontier_cache,
        )
        self._scan_views.append(captured)
        self._scan_candidates.update({
            _target_key(captured.map_frame_id, item.world_xy, item.candidate_id): item
            for item in candidates
        })
        if scan_context.index + 1 >= scan_context.count:
            with measure_stage(timings, "snapshot.submit"):
                self._flush_scan()
            return "submitted"
        return "buffered"

    def score_frontiers(self, request: FrontierScoreRequest) -> Mapping[str, float]:
        """只查相同世界目标的已完成评分；候选有效性由核心的新地图筛选保证。"""
        map_id = self._obstacle_frame_id
        scores = {}
        for candidate in request.candidates:
            key = _target_key(map_id, candidate.world_xy, candidate.candidate_id)
            if key not in self._scores:
                continue
            scores[candidate.candidate_id] = self._scores[key]
            job_id, interaction_id = self._score_origins[key]
            self._cycle_score_sources.append({
                "candidate_id": candidate.candidate_id, "score": self._scores[key],
                "job_id": job_id, "interaction_id": interaction_id,
            })
        return scores

    def localize_object(
        self, frame: NavigationFrame, goal: TargetSearchGoal, clue: TargetClue,
    ) -> ObjectLocalization:
        """历史 RGB-D 优先定位，障碍保底查询当前地图；保留两者各自的时间与位姿。"""
        if self._object_localizer is None:
            raise RuntimeError("物体接近缺少本地模型配置。")
        try:
            observation_frame = read_clue_frame(self.directory, clue, frame)
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            return ObjectLocalization(reason=f"历史 RGB-D 不可用，继续下一条线索：{exc}")
        context = {
            "clue_id": clue.clue_id, "job_id": clue.job_id, "view_id": clue.view_id,
            "source": "object_snapshot",
            "robot_pose": asdict(frame.pose),
            "localization_map": "full_navigation" if frame.navigation_map is not None else "exploration",
            "map_timestamp_s": frame.timestamp_s,
        }
        return self._object_localizer.locate(observation_frame, goal, context=context)

    def observe_motion_frame(self, frame: NavigationFrame) -> None:
        """运动期间按预采样开关接收一帧；由运行层在动作入口显式打开。"""
        with self._condition:
            if not self._closed and not self._background_paused and self._prefetch_enabled:
                self._pending_frame = (frame, self._capture_context)
                self._condition.notify_all()

    def set_motion_prefetch_enabled(self, enabled: bool) -> None:
        """由动作入口单独控制 VLM 预采样；只在指定的平移动作期间接收帧。"""
        with self._condition:
            self._prefetch_enabled = bool(enabled) and not self._background_paused

    def diagnostics(self) -> Mapping[str, Any]:
        with self._condition:
            return {
                "semantic_submitted": self._submitted, "semantic_finished": self._finished,
                "semantic_failed": self._failed, "semantic_queued": len(self._jobs),
                "semantic_active_job": self._active_job, "semantic_snapshot_directory": str(self.directory),
                "semantic_clues_waiting": len(self._clues),
                "semantic_background_paused": self._background_paused,
                "semantic_received_jobs": tuple(self._received_jobs),
                "semantic_score_sources": tuple(self._cycle_score_sources),
            }

    def wait_for_result(self, timeout_s: float = 1.0) -> None:
        with self._condition:
            self._raise_worker_error()
            if not self._completed and not self._closed:
                self._condition.wait(timeout_s)
            self._raise_worker_error()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            remaining = len(self._jobs) + int(self._active_job is not None)
            stopped_jobs = tuple(job_id for job_id, _ in self._jobs)
            if self._active_job is not None:
                stopped_jobs += (self._active_job,)
            self._jobs.clear()
            self._pending_frame = None
            self._condition.notify_all()
        self._emit({"event": "stopped", "job_ids": stopped_jobs})
        if self._object_localizer is not None:
            self._object_localizer.close()
        self._capture_worker.join(timeout=1.0)
        self._worker.join(timeout=1.0)
        if remaining:
            print(f"视觉队列因退出停止，{remaining} 批任务未消费；快照保留在 {self.directory}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------------
    # 内部：采集与队列
    # ------------------------------------------------------------------

    def _flush_scan(self) -> None:
        if not self._scan_views:
            return
        views = tuple(self._scan_views)
        with self._condition:
            candidates = tuple(
                item for key, item in self._scan_candidates.items() if key not in self._scored_targets
            )
        self._enqueue(views, candidates, "scan")
        self._scan_views.clear()
        self._scan_candidates.clear()

    def _capture(
        self, frame: NavigationFrame,
        *, context: CaptureContext, timings: Optional[TimingSpans] = None,
        frontier_cache: Optional[FrameFrontierCache] = None,
    ):
        """拍摄当前帧：预览候选、生成拼图与覆盖，并编码深度。"""
        with measure_stage(timings, "snapshot.frontier_preview"):
            candidates = preview_capture_candidates(
                frame, context, timings=timings, frontier_cache=frontier_cache,
            )
        with measure_stage(timings, "snapshot.observation_points"):
            points = frontier_observation_points(frame, candidates)
            visible_points = {_xy_key(point) for point in points}
            candidates = tuple(
                item for item in candidates
                if item.deferred_order is None and _xy_key(item.world_xy) in visible_points
            )
        with measure_stage(timings, "snapshot.image_projection"):
            image = buffer_scan_image(frame, candidates)
            has_direction = has_frontier_direction_in_view(image, candidates)
            candidates = visible_frontier_candidates({1: image}, candidates)
        with measure_stage(timings, "snapshot.coverage"):
            coverage = capture_semantic_view(frame, context.observation_points or points)
        with measure_stage(timings, "snapshot.depth_encode"):
            depth_gzip = encode_depth(frame.depth, image.width_px, image.height_px)
        return CapturedView(image, coverage, frame.obstacle_map.frame_id, depth_gzip), candidates, has_direction

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._worker_error is not None or (not self._background_paused and self._pending_frame is not None)
                )
                if self._closed or self._worker_error is not None:
                    return
                frame, context = self._pending_frame
                self._pending_frame = None
                self._capture_busy = True
            try:
                if self._use_motion_frame(frame):
                    captured, candidates, has_direction = self._capture(frame, context=context)
                    with self._condition:
                        candidates = tuple(
                            item for item in candidates
                            if _target_key(captured.map_frame_id, item.world_xy, item.candidate_id)
                            not in self._scored_targets
                        )
                    if has_direction:
                        # 新拍摄位置仍要检查目标；已有评分的方向不重复请求评分。
                        self._enqueue((captured,), candidates, "motion")
                        self._last_prefetch_pose = frame.pose
            except OSError as exc:
                self._emit({"event": "prefetch_skipped", "reason": str(exc), "timestamp_s": frame.timestamp_s})
            finally:
                with self._condition:
                    self._capture_busy = False
                    self._condition.notify_all()

    def _use_motion_frame(self, frame: NavigationFrame) -> bool:
        previous = self._previous_motion_frame
        self._previous_motion_frame = frame
        if previous is not None:
            dt = frame.timestamp_s - previous.timestamp_s
            if dt <= 0.0 or _turn_distance(frame.pose.yaw_rad, previous.pose.yaw_rad) / dt > PREFETCH_MAX_YAW_SPEED_RAD_S:
                return False
        last = self._last_prefetch_pose
        return last is None or (
            math.hypot(frame.pose.x_m - last.x_m, frame.pose.y_m - last.y_m) >= PREFETCH_TRANSLATION_M
            or _turn_distance(frame.pose.yaw_rad, last.yaw_rad) >= PREFETCH_TURN_RAD
        )

    def _enqueue(self, views, candidates, source: str) -> None:
        # 两个生产者按提交顺序写入并入队，磁盘速度不能改变 FIFO 顺序。
        with self._enqueue_lock:
            self._enqueue_snapshot(views, candidates, source)

    def _enqueue_snapshot(self, views, candidates, source) -> None:
        # 相同拍摄帧只入队一次；无候选的检测批次同样保留。
        key = tuple((view.map_frame_id, view.coverage.timestamp_s) for view in views)
        with self._condition:
            if self._closed or key in self._submitted_keys:
                return
            candidates = tuple(
                item for item in candidates
                if _target_key(views[0].map_frame_id, item.world_xy, item.candidate_id) not in self._scored_targets
            )
            self._submitted += 1
            job_id = self._submitted
            self._writing += 1
            self._submitted_keys.add(key)
            goal = self._goal
        folder = self.directory / f"job-{job_id:06d}"
        try:
            write_snapshot(folder, views, candidates, goal, source, job_id)
        except Exception as exc:
            with self._condition:
                self._writing -= 1
                self._finished += 1
                self._failed += 1
                self._submitted_keys.discard(key)
                self._condition.notify_all()
            self._emit({"event": "snapshot_failed", "job_id": job_id, "reason": str(exc)})
            raise
        self._emit({"event": "queued", "job_id": job_id, "source": source,
                    "view_count": len(views), "candidate_count": len(candidates), "snapshot": str(folder),
                    "views": view_trace(tuple((view.map_frame_id, view.coverage) for view in views))})
        with self._condition:
            self._writing -= 1
            if not self._closed:
                self._jobs.append((job_id, folder))
                self._pending_coverage[job_id] = tuple(view.coverage for view in views)
                self._scored_targets.update(
                    _target_key(views[0].map_frame_id, item.world_xy, item.candidate_id) for item in candidates
                )
                self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._worker_error is not None or (not self._background_paused and bool(self._jobs))
                )
                if self._closed or self._worker_error is not None:
                    return
                job_id, folder = self._jobs.popleft()
                self._active_job = job_id
            self._emit({"event": "started", "job_id": job_id})
            views, candidates, region_ids = (), (), ()
            try:
                images, views, candidates, goal, region_ids, source = read_snapshot(folder)
            except (OSError, ValueError) as exc:
                job_result = SemanticAnalysis(None, detection_error=f"快照读取失败：{exc}")
            else:
                job_result = self._analyzer.analyze_views(images, candidates, goal, trace_context={
                    "job_id": job_id, "source": source, "snapshot": str(folder),
                    "views": view_trace(views),
                    "region_ids": {item.candidate_id: region_id for item, region_id in zip(candidates, region_ids)},
                })
            depth_retention = retain_clue_depth(folder, len(views), job_result.target_view_ids)
            if depth_retention.get("errors"):
                self._emit({"event": "depth_retention_failed", "job_id": job_id,
                            "errors": depth_retention["errors"]})
            result_record = {**asdict(job_result), "depth_retention": depth_retention}
            try:
                (folder / "result.json").write_text(json.dumps(result_record, ensure_ascii=False), encoding="utf-8")
            except OSError as exc:
                self._emit({"event": "result_write_failed", "job_id": job_id, "reason": str(exc)})
            self._emit({"event": "completed", "job_id": job_id, **result_record})
            with self._condition:
                self._active_job = None
                self._finished += 1
                if job_result.target_view_ids is None:
                    self._failed += 1
                for candidate, region_id in zip(candidates, region_ids):
                    if candidate.candidate_id not in job_result.frontier_scores and views:
                        self._scored_targets.discard(_target_key(views[0][0], candidate.world_xy, region_id))
                if not self._closed:
                    self._completed.append(_CompletedAnalysis(job_id, job_result, views, candidates, region_ids))
                self._condition.notify_all()

    def _run_worker(self, work) -> None:
        """跨线程传递未预期异常；主循环重新抛出原异常，不伪造模型失败。"""
        try:
            work()
        except Exception as exc:
            traceback.print_exc()
            with self._condition:
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


def _turn_distance(first, second):
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


__all__ = [
    "CycleIntake",
    "PerceptionPause",
    "SemanticPerception",
    "SemanticPerceptionConfig",
]
