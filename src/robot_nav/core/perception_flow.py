"""感知输入的状态归并和采样上下文；队列不读取或修改 SearchState。"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from .frontier import extract_frame_frontiers
from .frontier_regions import tried_candidate_points
from .history import filter_blocked_frontier_regions, match_frontier_regions
from .models import (
    BlockedFrontierRegion, FrontierCandidate, FrontierRegion, NavigationFrame,
    ObservationView, SearchPhase, SearchState, TargetClue,
)


@dataclass(frozen=True)
class CaptureContext:
    """提交帧时冻结的候选关联信息，避免后台读取变化中的搜索状态。"""

    map_frame_id: str
    tried_points: Tuple[Tuple[float, float], ...] = ()
    blocked_regions: Tuple[BlockedFrontierRegion, ...] = ()
    regions: Tuple[FrontierRegion, ...] = ()
    next_region_id: int = 0
    observation_points: Tuple[Tuple[float, float], ...] = ()
    observed_views: Tuple[ObservationView, ...] = ()


def capture_context(frame: NavigationFrame, state: SearchState) -> CaptureContext:
    """冻结筛选与关联候选需要的历史，不把完整导航状态交给队列。"""
    return CaptureContext(frame.obstacle_map.frame_id,
                          tried_candidate_points(state.observation_history),
                          state.blocked_frontier_regions, state.frontier_regions,
                          state.next_frontier_region_id, state.scan_observation_points,
                          state.observed_views)


def preview_capture_candidates(
    frame: NavigationFrame, context: Optional[CaptureContext],
    *, timings=None, frontier_cache=None,
) -> Tuple[FrontierCandidate, ...]:
    """用采样帧和冻结的历史筛选候选；不提交区域 ID 或状态变化。"""
    if context is None or context.map_frame_id != frame.obstacle_map.frame_id:
        raise ValueError("采样帧缺少同一地图的候选上下文")
    extraction = extract_frame_frontiers(frame, context.tried_points,
                                        cache=frontier_cache, timings=timings)
    candidates, _ = filter_blocked_frontier_regions(
        frame.obstacle_map, extraction.candidates, context.blocked_regions)
    return match_frontier_regions(frame.obstacle_map, candidates, context.regions,
                                  context.next_region_id)[0]


def target_handling_active(state: SearchState) -> bool:
    """目标处理或终态期间暂停普通队列，不消费下一条线索。"""
    return state.active_target_clue is not None or state.phase in (
        SearchPhase.REVISITING_TARGET, SearchPhase.LOCALIZING_OBJECT,
        SearchPhase.APPROACHING_OBJECT, SearchPhase.COMPLETE,
        SearchPhase.FAILED, SearchPhase.STOPPED,
    )


def receive_perception(
    state: SearchState, observed_views: Tuple[ObservationView, ...] = (),
    *, pending: int, failed: int, pending_views: Tuple[ObservationView, ...],
    clue: Optional[TargetClue] = None,
) -> SearchState:
    """只有分析成功的新增覆盖进入 observed_views；采集覆盖仍保持待分析。"""
    # 接收是增量归并；相同拍摄时刻的覆盖只登记一次，pending 不冒充已检查。
    timestamps = {view.timestamp_s for view in state.observed_views}
    return replace(
        state, asynchronous_perception=True,
        observed_views=state.observed_views + tuple(
            view for view in observed_views if view.timestamp_s not in timestamps),
        pending_semantic_jobs=pending, failed_semantic_jobs=failed,
        pending_observation_views=pending_views,
        active_target_clue=state.active_target_clue if clue is None else clue,
    )
