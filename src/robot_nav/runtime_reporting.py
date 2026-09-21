"""导航运行的日志、可选可视化回调与终端摘要。"""

from .core.models import NavigationResult
from .core.timing import measure_stage
from .run_log import NavigationRunLogger


def _with_run_log(on_cycle, run_logger: NavigationRunLogger, cycle_index: int):
    """把发送命令前的完整决策帧写入当前运行日志。"""

    def callback(frame, observation, result) -> None:
        timings = []
        with measure_stage(timings, "callback.decision_log"):
            run_logger.log_cycle_decision(cycle_index, frame, observation, result)
        with measure_stage(timings, "callback.visualization_and_debug"):
            if on_cycle is not None:
                on_cycle(frame, observation, result)
        run_logger.log_callback_timing(timings)

    return callback


def _with_frontier_debug(on_cycle, enabled: bool):
    """把可选 Frontier 终端输出接到发送命令前的周期回调。"""
    if not enabled:
        return on_cycle

    def callback(frame, observation, result) -> None:
        if on_cycle is not None:
            on_cycle(frame, observation, result)
        _print_frontier_debug(frame, result)

    return callback


def _print_frontier_debug(frame, result: NavigationResult) -> None:
    """打印父节点返回目标，或本轮 Frontier 候选的评分组成。"""
    details = result.debug.details
    stage = details.get("next_stage", result.debug.stage)
    if stage == "backtrack.return":
        print(
            f"[Frontier] stage={stage}，逐级返回节点={details['parent_node_id']}，"
            f"移动目标={details['destination_world_xy']}，"
            f"分支深度={details['branch_depth']}，"
            f"本节点暂存方向={details['pending_direction_count']}，到达后刷新并决策"
        )
        return
    if stage not in ("explore.select", "backtrack.resume"):
        return
    candidates = result.debug.details.get("frontier_candidates")
    if not candidates:
        return

    path_weight = result.debug.details["frontier_path_distance_weight"]
    semantic_weight = result.debug.details["frontier_semantic_score_weight"]
    print(
        "[Frontier] "
        f"stage={stage}，"
        f"本轮候选={len(candidates)}，"
        f"选择来源={result.debug.details['frontier_selection_source']}，"
        f"新方向={result.debug.details['new_frontier_count']}，"
        f"暂存旧方向={result.debug.details['deferred_frontier_count']}，"
        f"robot=({frame.pose.x_m:.3f}, {frame.pose.y_m:.3f}) m"
    )
    if stage == "backtrack.resume":
        print(f"[Frontier] 已到达父节点={details['parent_node_id']}，恢复该节点暂存方向。")
    print(
        "[Frontier] score = 前沿跨度 "
        f"- {path_weight:.2f}×路径距离 + 语义奖励（仅用于新方向排序）；"
        f"语义奖励 = {semantic_weight:.2f}×(2×VLM分数-1)"
    )
    for rank, candidate in enumerate(candidates, start=1):
        selected = " selected" if rank == 1 else ""
        semantic_score = candidate["semantic_score"]
        semantic_text = "none" if semantic_score is None else f"{semantic_score:.3f}"
        print(
            f"[Frontier #{rank:02d}{selected}] "
            f"id={candidate['candidate_id']} "
            f"grid=({candidate['row']}, {candidate['col']}) "
            f"world=({candidate['world_x_m']:.3f}, "
            f"{candidate['world_y_m']:.3f}) m "
            f"cells={candidate['frontier_cell_count']} "
            f"span={candidate['frontier_span_m']:.3f} m "
            f"path={candidate['path_distance_m']:.3f} m "
            f"distance_penalty={candidate['distance_penalty']:.3f} "
            f"vlm={semantic_text} "
            f"semantic_bonus={candidate['semantic_bonus']:+.3f} "
            f"deferred_order={candidate['deferred_order']} "
            f"score={candidate['score']:.3f}"
        )
    print(f"[Frontier] 本次完整移动目标={result.debug.details['destination_world_xy']}")


def _optional_callback(callback, description):
    """可视化失败后停用该回调，保持感知与导航运行。"""
    if callback is None:
        return None
    enabled = True

    def invoke(*args):
        nonlocal enabled
        if not enabled:
            return
        try:
            callback(*args)
        except Exception as exc:
            enabled = False
            print(f"{description}已停用：{exc}", flush=True)

    return invoke


def _print_cycle(cycle_index: int, result: NavigationResult) -> None:
    """输出足以沿算法步骤排错的一行周期信息。"""
    print(
        f"[{cycle_index:03d}] status={result.status.value} "
        f"phase={result.state.phase.value} stage={result.debug.stage} | "
        f"{result.debug.message}"
    )
