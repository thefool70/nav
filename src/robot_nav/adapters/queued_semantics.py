"""固定视觉快照的 FIFO 队列；网络线程只产出结果，不操作底盘或搜索状态。

图片和元数据写入独立运行目录。预采样只覆盖尚未选取的运动帧；已入队任务不覆盖、
不因 Frontier 失效而丢弃检测。主循环在动作结束后接收覆盖、分数和目标线索。
"""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import threading
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Tuple

from ..core.models import (
    CameraExtrinsics, CameraIntrinsics, FrontierCandidate, FrontierScoreRequest, NavigationFrame,
    ObservationView, Pose2D, SceneAssessment, SceneAssessmentResult, SearchPhase,
    SearchState, SemanticAnalysis, TargetClue, TargetObservation, TargetSearchGoal,
    TargetVisibility,
)
from ..core.navigator import capture_semantic_view, preview_frontier_candidates
from ..core.frontier_projection import FrontierImageProjection
from ..core.observation_coverage import frontier_observation_points
from .frontier_overlay import (
    BufferedScanImage, buffer_scan_image, has_frontier_direction_in_view, visible_frontier_candidates,
)
from .perception import ContinuousTargetObserver, ScanObservationContext, SemanticAnalyzer


PREFETCH_TRANSLATION_M = 0.75
PREFETCH_TURN_RAD = math.radians(30.0)
PREFETCH_MAX_YAW_SPEED_RAD_S = math.radians(25.0)


@dataclass(frozen=True)
class _CapturedView:
    image: BufferedScanImage
    coverage: ObservationView
    map_frame_id: str


@dataclass(frozen=True)
class _CompletedAnalysis:
    job_id: int
    result: SemanticAnalysis
    views: Tuple[Tuple[str, ObservationView], ...]
    candidates: Tuple[FrontierCandidate, ...]
    region_ids: Tuple[str, ...]


class QueuedSemanticObserver:
    """普通扫描按轮入队；检测到目标后暂停后台，核心返回拍摄位姿并结束。"""

    def __init__(
        self,
        analyzer: SemanticAnalyzer,
        *,
        local_observer: Optional[ContinuousTargetObserver] = None,
        on_event: Optional[Callable[[Mapping[str, Any]], None]] = None,
        directory: Optional[Path] = None,
    ) -> None:
        if directory is None:
            root = Path("data/run_logs")
            root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="semantic-", dir=root))
        else:
            directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self._analyzer = analyzer
        self._local = local_observer
        self._on_event = on_event
        self._condition = threading.Condition()
        self._enqueue_lock = threading.Lock()
        self._closed = False
        self._goal: Optional[TargetSearchGoal] = None
        self._state = SearchState(asynchronous_perception=True)
        self._frame: Optional[NavigationFrame] = None
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
        self._pending_frame: Optional[Tuple[NavigationFrame, SearchState]] = None
        self._capture_busy = False
        self._last_prefetch_pose: Optional[Pose2D] = None
        self._previous_motion_frame: Optional[NavigationFrame] = None
        self._worker = threading.Thread(target=self._worker_loop, name="semantic-fifo", daemon=True)
        self._capture_worker = threading.Thread(target=self._capture_loop, name="semantic-capture", daemon=True)
        self._worker.start()
        self._capture_worker.start()
        print(f"异步视觉队列：{self.directory}（固定快照，FIFO）", flush=True)

    def prepare_cycle(self, frame: NavigationFrame, state: SearchState) -> SearchState:
        """仅主循环调用；目标处理期间暂存迟到结果，其余时候接收覆盖、分数与线索。"""
        self._received_jobs = []
        self._cycle_score_sources = []
        with self._condition:
            completed = () if self._should_pause_background(state) else tuple(self._completed)
            if completed:
                self._completed.clear()
            for item in completed:
                self._pending_coverage.pop(item.job_id, None)
        views = list(state.observed_views)
        timestamps = {view.timestamp_s for view in views}
        for item in completed:
            new_clues = []
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
            for view_id in item.result.target_view_ids or ():
                index = view_id - 1
                if 0 <= index < len(item.views):
                    map_id, view = item.views[index]
                    key = _vantage_key(map_id, view.pose)
                    if map_id == frame.obstacle_map.frame_id and key not in self._clue_vantages:
                        self._clue_vantages.add(key)
                        self._clues.append(TargetClue(
                            f"semantic:{item.job_id}:{index + 1}", view.pose, view.timestamp_s, map_id,
                        ))
                        new_clues.append(self._clues[-1].clue_id)
            self._received_jobs.append({
                "job_id": item.job_id, "interaction_id": item.result.interaction_id,
                "clue_ids": tuple(new_clues), "target_view_ids": item.result.target_view_ids,
            })
        state = self.sync_state(replace(state, observed_views=tuple(views)))
        self.bind_context(frame, state)
        return state

    def take_target_clue(self, state: SearchState) -> Optional[TargetClue]:
        """批次按 FIFO，批内按模型列表顺序；已有目标处理期间不消费下一条。"""
        if state.active_target_clue is not None or state.phase in (
            SearchPhase.COMPLETE, SearchPhase.LOCALIZING_TARGET,
            SearchPhase.VERIFYING_TARGET, SearchPhase.REVISITING_TARGET,
        ):
            return None
        return self._clues.popleft() if self._clues else None

    def bind_context(self, frame: NavigationFrame, state: SearchState) -> None:
        """向观察器和预采样线程发布不可变的最新决策上下文。"""
        with self._condition:
            self._frame, self._state = frame, state

    def sync_state(self, state: SearchState) -> SearchState:
        """先同步后台暂停状态，再提交残余扫描并回写待处理数量。"""
        with self._condition:
            paused = self._should_pause_background(state)
            if paused != self._background_paused:
                self._background_paused = paused
                self._condition.notify_all()
            if paused:
                self._prefetch_enabled = False
        if state.phase is not SearchPhase.SCANNING:
            self._flush_scan()
        with self._condition:
            pending = (
                len(self._jobs) + len(self._completed) + self._writing
                + int(self._active_job is not None) + int(self._capture_busy)
                + int(self._pending_frame is not None) + int(bool(self._scan_views))
                + len(self._clues)
            )
            failed = self._failed
            pending_views = tuple(view for views in self._pending_coverage.values() for view in views)
        return replace(
            state, asynchronous_perception=True, pending_semantic_jobs=pending,
            failed_semantic_jobs=failed, pending_observation_views=pending_views,
        )

    def _should_pause_background(self, state: SearchState) -> bool:
        """现有线索优先处理；两条线索之间也不恢复普通任务，终态保持暂停。"""
        return bool(self._clues) or state.active_target_clue is not None or state.phase in (
            SearchPhase.REVISITING_TARGET, SearchPhase.LOCALIZING_TARGET,
            SearchPhase.VERIFYING_TARGET, SearchPhase.VERIFYING_SCENE,
            SearchPhase.COMPLETE, SearchPhase.FAILED,
        )

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

    def observe(
        self, frame: NavigationFrame, goal: TargetSearchGoal,
        scan_context: Optional[ScanObservationContext] = None,
    ) -> TargetObservation:
        self._set_goal(goal)
        if scan_context is not None:
            self._capture_scan(frame, scan_context)
        if self._local is not None:
            # 联合请求已持有独立快照，本地检测不再写共享 VLM 扫描缓存。
            return self._local.observe(frame, goal)
        if scan_context is None and self._state.phase in (
            SearchPhase.LOCALIZING_TARGET, SearchPhase.VERIFYING_TARGET,
        ):
            return self._analyzer.observe(frame, goal)
        reason = (
            "已缓存本方向画面，等待本轮扫描收齐后统一分析。"
            if scan_context is not None and scan_context.index + 1 < scan_context.count
            else "画面已交给后台联合检测与评分。"
        )
        return TargetObservation(TargetVisibility.PENDING, reason=reason)

    def score_frontiers(self, request: FrontierScoreRequest, goal: TargetSearchGoal) -> Mapping[str, float]:
        """只查相同世界目标的已完成评分，候选有效性由核心的新地图筛选保证。"""
        self._set_goal(goal)
        map_id = self._frame.obstacle_map.frame_id
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

    def assess_scene(self, goal: TargetSearchGoal) -> SceneAssessmentResult:
        """满足 TargetObserver 接口；异步场景只使用联合检测，不单独请求确认。"""
        return SceneAssessmentResult(
            SceneAssessment.UNCERTAIN, "异步场景由联合检测判断，返回拍摄位姿后直接结束。",
        )

    def confirm_target(self, frame, goal, observation):
        observer = self._local if self._local is not None else self._analyzer
        return observer.confirm_target(frame, goal, observation)

    def submit_motion_frame(self, frame: NavigationFrame, goal: TargetSearchGoal) -> None:
        self._set_goal(goal)
        if self._local is not None:
            self._local.submit_motion_frame(frame, goal)
        with self._condition:
            if not self._closed and not self._background_paused and self._prefetch_enabled:
                self._pending_frame = (frame, self._state)
                self._condition.notify_all()

    def set_motion_interrupt_enabled(self, enabled: bool) -> None:
        """只控制本地目标检测中断；扫描转向允许检测，但不触发 VLM 预采样。"""
        if self._local is not None:
            self._local.set_motion_interrupt_enabled(enabled)

    def set_motion_prefetch_enabled(self, enabled: bool) -> None:
        """由动作入口单独控制 VLM 预采样；只在指定的平移动作期间接收帧。"""
        with self._condition:
            self._prefetch_enabled = bool(enabled) and not self._background_paused

    def should_interrupt_motion(self) -> bool:
        return self._local.should_interrupt_motion() if self._local is not None else False

    def wait_for_result(self, timeout_s: float = 1.0) -> None:
        with self._condition:
            if not self._completed and not self._closed:
                self._condition.wait(timeout_s)

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
        self._capture_worker.join(timeout=1.0)
        self._worker.join(timeout=1.0)
        if self._local is not None:
            self._local.close()
        if remaining:
            print(f"视觉队列因退出停止，{remaining} 批任务未消费；快照保留在 {self.directory}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _set_goal(self, goal: TargetSearchGoal) -> None:
        with self._condition:
            if self._goal is not None and self._goal != goal:
                raise ValueError("同一视觉队列不能混用不同搜索目标")
            self._goal = goal

    def _capture_scan(self, frame: NavigationFrame, context: ScanObservationContext) -> None:
        """本轮全部方向收齐后提交一次，目标线索和候选分数共享整轮拼图。"""
        if context.index == 0:
            # 扫描被打断后重建计划时，先提交上一轮已拍到的部分画面。
            self._flush_scan()
        captured, candidates, _ = self._capture(frame, self._state)
        self._scan_views.append(captured)
        self._scan_candidates.update({_target_key(captured.map_frame_id, item.world_xy, item.candidate_id): item for item in candidates})
        if context.index + 1 >= context.count:
            self._flush_scan()

    def _flush_scan(self) -> None:
        if not self._scan_views:
            return
        views = tuple(self._scan_views)
        with self._condition:
            candidates = tuple(item for key, item in self._scan_candidates.items() if key not in self._scored_targets)
        self._enqueue(views, candidates, "scan")
        self._scan_views.clear()
        self._scan_candidates.clear()

    def _capture(self, frame: NavigationFrame, state: SearchState):
        candidates = preview_frontier_candidates(frame, state)
        points = frontier_observation_points(frame, candidates)
        visible_points = {_xy_key(point) for point in points}
        candidates = tuple(
            item for item in candidates
            if item.deferred_order is None and _xy_key(item.world_xy) in visible_points
        )
        image = buffer_scan_image(frame, candidates)
        has_visible_direction = has_frontier_direction_in_view(image, candidates)
        candidates = visible_frontier_candidates({1: image}, candidates)
        coverage = capture_semantic_view(frame, state.scan_observation_points or points)
        return _CapturedView(image, coverage, frame.obstacle_map.frame_id), candidates, has_visible_direction

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or (not self._background_paused and self._pending_frame is not None)
                )
                if self._closed:
                    return
                frame, state = self._pending_frame
                self._pending_frame = None
                self._capture_busy = True
            try:
                if self._use_motion_frame(frame):
                    captured, candidates, has_visible_frontier = self._capture(frame, state)
                    with self._condition:
                        candidates = tuple(item for item in candidates
                                           if _target_key(captured.map_frame_id, item.world_xy, item.candidate_id) not in self._scored_targets)
                    if has_visible_frontier:
                        # 新拍摄位置仍要检查目标；已有评分的方向不重复请求评分。
                        self._enqueue((captured,), candidates, "motion")
                        self._last_prefetch_pose = frame.pose
            except Exception as exc:
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

    def _enqueue(self, views: Tuple[_CapturedView, ...], candidates: Tuple[FrontierCandidate, ...], source: str) -> None:
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
            _write_snapshot(folder, views, candidates, goal, source, job_id)
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
                    "views": _view_trace(tuple((view.map_frame_id, view.coverage) for view in views))})
        with self._condition:
            self._writing -= 1
            if not self._closed:
                self._jobs.append((job_id, folder))
                self._pending_coverage[job_id] = tuple(view.coverage for view in views)
                self._scored_targets.update(_target_key(views[0].map_frame_id, item.world_xy, item.candidate_id) for item in candidates)
                self._condition.notify_all()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or (not self._background_paused and bool(self._jobs))
                )
                if self._closed:
                    return
                job_id, folder = self._jobs.popleft()
                self._active_job = job_id
            self._emit({"event": "started", "job_id": job_id})
            views, candidates, region_ids = (), (), ()
            try:
                images, views, candidates, goal, region_ids, source = _read_snapshot(folder)
                result = self._analyzer.analyze_views(images, candidates, goal, trace_context={
                    "job_id": job_id, "source": source, "snapshot": str(folder),
                    "views": _view_trace(views),
                    "region_ids": {item.candidate_id: region_id for item, region_id in zip(candidates, region_ids)},
                })
                if not isinstance(result, SemanticAnalysis):
                    raise TypeError("联合分析器必须返回 SemanticAnalysis")
            except Exception as exc:
                result = SemanticAnalysis(None, detection_error=str(exc) or type(exc).__name__)
            try:
                (folder / "result.json").write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
            except (OSError, TypeError, ValueError) as exc:
                self._emit({"event": "result_write_failed", "job_id": job_id, "reason": str(exc)})
            self._emit({"event": "completed", "job_id": job_id, **asdict(result)})
            with self._condition:
                self._active_job = None
                self._finished += 1
                if result.target_view_ids is None:
                    self._failed += 1
                for candidate, region_id in zip(candidates, region_ids):
                    if candidate.candidate_id not in result.frontier_scores and views:
                        self._scored_targets.discard(_target_key(views[0][0], candidate.world_xy, region_id))
                if not self._closed:
                    self._completed.append(_CompletedAnalysis(job_id, result, views, candidates, region_ids))
                self._condition.notify_all()

    def _emit(self, event: Mapping[str, Any]) -> None:
        callback = self._on_event
        if callback is not None:
            try:
                callback(event)
            except Exception:
                self._on_event = None


def _write_snapshot(folder, views, candidates, goal, source, job_id):
    folder.mkdir()
    metadata = {"source": source, "goal": asdict(goal), "views": [], "candidates": []}
    for index, view in enumerate(views, 1):
        image = view.image
        with gzip.open(folder / f"view-{index}.rgb.gz", "wb") as stream:
            stream.write(image.rgb_bytes)
        metadata["views"].append({
            "map_frame_id": view.map_frame_id, "coverage": asdict(view.coverage),
            "width_px": image.width_px, "height_px": image.height_px,
            "intrinsics": asdict(image.intrinsics), "camera_yaw_rad": image.camera_yaw_rad,
            "camera_extrinsics_in_robot": asdict(image.camera_extrinsics_in_robot),
            "frontier_projections": [asdict(projection) for projection in image.frontier_projections],
        })
    for index, candidate in enumerate(candidates, 1):
        item = asdict(replace(candidate, candidate_id=f"snapshot:{job_id}:{index}", frontier_cells=()))
        item["source_region_id"] = candidate.candidate_id
        metadata["candidates"].append(item)
    (folder / "snapshot.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _read_snapshot(folder):
    from ..core.models import SearchMode

    metadata = json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    goal = TargetSearchGoal(metadata["goal"]["target_text"], SearchMode(metadata["goal"]["search_mode"]))
    images, views, candidates, region_ids = {}, [], [], []
    for index, item in enumerate(metadata["views"], 1):
        raw = item["coverage"]
        raw["pose"] = Pose2D(**raw["pose"])
        for name in ("visible_world_xy", "map_visible_world_xy"):
            raw[name] = tuple(tuple(point) for point in raw[name])
        if raw["camera_world_xy"] is not None:
            raw["camera_world_xy"] = tuple(raw["camera_world_xy"])
        coverage = ObservationView(**raw)
        with gzip.open(folder / f"view-{index}.rgb.gz", "rb") as stream:
            rgb = stream.read()
        images[index] = BufferedScanImage(
            item["width_px"], item["height_px"], rgb, coverage.pose,
            CameraIntrinsics(**item["intrinsics"]), item["camera_yaw_rad"],
            CameraExtrinsics(**item["camera_extrinsics_in_robot"]),
            tuple(FrontierImageProjection(
                tuple(projection["world_xy"]), tuple(projection["pixel_xy"]),
                projection["camera_depth_m"], projection["observed_depth_m"],
            ) for projection in item["frontier_projections"]),
        )
        views.append((item["map_frame_id"], coverage))
    for item in metadata["candidates"]:
        region_ids.append(item.pop("source_region_id"))
        item["world_xy"] = tuple(item["world_xy"])
        item["frontier_cells"] = tuple(tuple(cell) for cell in item["frontier_cells"])
        if item["deferred_order"] is not None:
            item["deferred_order"] = tuple(item["deferred_order"])
        candidates.append(FrontierCandidate(**item))
    return images, tuple(views), tuple(candidates), goal, tuple(region_ids), metadata["source"]


def _view_trace(views):
    """画面编号、拍摄时间与机器人位姿用于关联 VLM 请求，不携带覆盖点大数组。"""
    return tuple({"view_id": index, "map_frame_id": map_id, "timestamp_s": view.timestamp_s,
                  "pose": asdict(view.pose), "heading_world_rad": view.camera_heading_world_rad}
                 for index, (map_id, view) in enumerate(views, 1))


def _xy_key(point):
    return round(point[0], 6), round(point[1], 6)


def _target_key(map_id, point, region_id):
    return (map_id, region_id) + _xy_key(point)


def _vantage_key(map_id, pose):
    return map_id, round(pose.x_m / 0.5), round(pose.y_m / 0.5), round(pose.yaw_rad / math.radians(15.0))


def _turn_distance(first, second):
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)
