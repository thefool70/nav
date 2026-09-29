"""扫描：规划观察朝向、推进转向与采集请求，计算并复用视觉覆盖。

阅读入口是 continue_scanning；采集上下文、朝向规划与覆盖计算紧随流程。
设备取帧和逐图入队由 app.py 调用感知队列完成，搜索状态只由核心更新。"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import replace, dataclass
from typing import Any, Mapping, Optional, Tuple, Sequence, Iterable

from .exploration import (
    FrameFrontierCache,
    FrontierExtraction,
    refresh_frontier_regions,
    extract_frame_frontiers,
    tried_candidate_points,
    filter_blocked_frontier_regions,
    match_frontier_regions,
    select_exploration_target,
)
from .geometry import (
    shortest_turn_to_heading,
    wrap_angle,
    grid_cell_center_to_world,
    world_point_to_robot,
    world_to_nearest_grid_cell,
)
from .models import (
    ActionKind,
    ActionPurpose,
    NavigationFrame,
    NavigationResult,
    NavigationStatus,
    RelativePoseCommand,
    SearchPhase,
    SearchState,
    NavigationAction,
    result,
    BlockedFrontierRegion,
    FrontierRegion,
    ObservationView,
    ObstacleMap,
)
from .timing import TimingSpans


TURN_TOLERANCE_RAD = math.radians(5.0)
HEADING_BIN_RAD = math.radians(2.0)


@dataclass(frozen=True)
class CaptureContext:
    """提交帧时冻结的候选关联信息，避免后台读取变化中的搜索状态。"""

    map_frame_id: str
    tried_points: Tuple[Tuple[float, float], ...] = ()
    blocked_regions: Tuple[BlockedFrontierRegion, ...] = ()
    regions: Tuple[FrontierRegion, ...] = ()
    next_region_id: int = 0
    observation_points: Tuple[Tuple[float, float], ...] = ()


WorldPoint = Tuple[float, float]
OBSERVATION_RANGE_M = 4.0
OBSERVATION_SPACING_M = 0.25
MIN_OBSERVATION_DISTANCE_M = 0.5
VIEW_EDGE_MARGIN_RAD = math.radians(5.0)
VIEWPOINT_CHANGE_RAD = math.radians(30.0)
DEPTH_MARGIN_M = 0.10
SAME_POSITION_REUSE_M = 0.10


# 扫描推进与采集确认


def continue_scanning(
    frame: NavigationFrame,
    state: SearchState,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """每轮都按未观察的局部 Frontier 扫描；没有待查方向时采集当前画面。"""
    planning_details = {}
    if not state.scan_headings_world_rad:
        try:
            state, frontiers = refresh_frontier_regions(frame, state,
                timings=timings, frontier_cache=frontier_cache,
            )
            local_points = frontier_observation_points(frame, frontiers.boundary_cells)
            # 待分析视角也暂时避免重复采集；只有分析成功才会登记为已检查覆盖。
            points = unobserved_observation_points(
                local_points, frame, state.observed_views + state.pending_observation_views,
            )
            if points:
                headings = build_unobserved_scan_headings(frame, points)
            else:
                # 覆盖可复用也保留当前画面的目标检查，不为此额外转向。
                headings = (frame.pose.yaw_rad,)
        except ValueError as exc:
            return result(
                NavigationStatus.MISSING_DATA, state, "scan.plan",
                f"无法规划待检查视角：{exc}",
            )
        planning_details = {
            "frontier_move_candidate_count": len(frontiers.candidates),
            "frontier_scan_cell_count": len(frontiers.boundary_cells),
        }
        state = replace(
            state, phase=SearchPhase.SCANNING,
            scan_headings_world_rad=headings, scan_views=(),
            scan_observation_points=points, scan_local_point_count=len(local_points),
        )

    # 本周期只请求转向或采集；采集完成后由 record_scanned_direction 推进扫描。
    scan_index = len(state.scan_views)
    target_heading = state.scan_headings_world_rad[scan_index]
    relative_turn = shortest_turn_to_heading(frame.pose.yaw_rad, target_heading)
    details = {
        **scan_coverage_details(state),
        "scan_index": scan_index,
        "scan_heading_count": len(state.scan_headings_world_rad),
        "scan_mode": "frontier" if state.scan_observation_points else "current_view",
        "target_heading_world_rad": target_heading,
        "checked_view_count": len(state.observed_views),
        **planning_details,
    }
    if abs(relative_turn) > TURN_TOLERANCE_RAD:
        return result(
            NavigationStatus.OK,
            state,
            "scan.turn",
            "转向下一个扫描方向。",
            NavigationAction(
                ActionKind.TURN_IN_PLACE,
                command=RelativePoseCommand(yaw_rad=relative_turn),
                purpose=ActionPurpose.SCAN_TURN,
            ),
            details,
        )
    return result(
        NavigationStatus.NEEDS_SCAN_CAPTURE, state, "scan.capture",
        "扫描方向已对齐，等待固定画面采集。",
        details=details,
    )


def record_scanned_direction(
    frame: NavigationFrame,
    state: SearchState,
    captured_view: ObservationView,
    *,
    timings: Optional[TimingSpans] = None,
    frontier_cache: Optional[FrameFrontierCache] = None,
) -> NavigationResult:
    """登记实际提交的覆盖，直接用采集数推进扫描；这里不重复采样或判断目标。"""
    scanned_state = replace(state, scan_views=state.scan_views + (captured_view,))
    if len(scanned_state.scan_views) < len(state.scan_headings_world_rad):
        return continue_scanning(
            frame, scanned_state,
            timings=timings, frontier_cache=frontier_cache,
        )
    # 本轮采集结束便进入探索；已有语义结果由运行层补充，不在这里等待模型。
    return select_exploration_target(
        frame, replace(scanned_state, phase=SearchPhase.EXPLORING),
        timings=timings, frontier_cache=frontier_cache,
    )


def recover_scan_turn(state: SearchState, reason: str) -> NavigationResult:
    """转向已停止后按下一帧真实朝向重建剩余观察，不假装该方向已检查。"""
    return result(
        NavigationStatus.OK, reset_scan_after_move(state), "motion.scan_recovered",
        "扫描转向未完成，按实际朝向和已有观察记录重新规划剩余视角。",
        details={"reason": str(reason)},
    )


def reset_scan_after_move(state: SearchState) -> SearchState:
    """移动命令完成后，延迟到下一帧再按新的真实朝向建立扫描。"""
    return replace(
        state,
        phase=SearchPhase.SCANNING,
        scan_headings_world_rad=(),
        scan_views=(),
        scan_observation_points=(),
        scan_local_point_count=0,
        backtrack_node_id=None,
        active_target_clue=None,
    )


# 冻结采样上下文


def capture_context(frame: NavigationFrame, state: SearchState) -> CaptureContext:
    """冻结筛选与关联候选需要的历史，不把完整导航状态交给队列。"""
    return CaptureContext(frame.obstacle_map.frame_id,
                          tried_candidate_points(state.observation_history),
                          state.blocked_frontier_regions, state.frontier_regions,
                          state.next_frontier_region_id, state.scan_observation_points)


def preview_capture_frontiers(
    frame: NavigationFrame, context: Optional[CaptureContext],
    *, timings=None, frontier_cache=None,
) -> FrontierExtraction:
    """同时返回完整扫描边界与已关联候选；不提交区域 ID 或状态变化。"""
    if context is None or context.map_frame_id != frame.obstacle_map.frame_id:
        raise ValueError("采样帧缺少同一地图的候选上下文")
    extraction = extract_frame_frontiers(frame, context.tried_points,
                                        cache=frontier_cache, timings=timings)
    candidates, _ = filter_blocked_frontier_regions(
        frame.obstacle_map, extraction.candidates, context.blocked_regions)
    candidates, _, _ = match_frontier_regions(
        frame.obstacle_map, candidates, context.regions, context.next_region_id)
    return replace(extraction, candidates=candidates)


# 观察朝向规划


def build_unobserved_scan_headings(
    frame: NavigationFrame, points_world_xy: Sequence[WorldPoint],
) -> Tuple[float, ...]:
    """按相机视场覆盖待查点：先合并 2° 方位桶，再选视角最少、转角较短的方案。"""
    camera_offset, horizontal_fov = horizontal_camera_view(frame)
    origin = camera_world_position(frame)
    current_heading = frame.pose.yaw_rad
    if not all(math.isfinite(value) for value in (*origin, current_heading)):
        raise ValueError("扫描需要有限的相机位置与底盘朝向")
    point_headings = sorted({
        round(wrap_angle(_bearing(origin, point)) / HEADING_BIN_RAD) * HEADING_BIN_RAD
        for point in points_world_xy if math.dist(origin, point) > 1e-6
    })
    if not point_headings:
        return ()
    effective_fov = max(
        horizontal_fov / 2.0,
        horizontal_fov - 2.0 * VIEW_EDGE_MARGIN_RAD - HEADING_BIN_RAD,
    )
    current_camera_heading = wrap_angle(current_heading + camera_offset)
    if all(abs(wrap_angle(heading - current_camera_heading)) <= effective_fov / 2.0
           for heading in point_headings):
        return (wrap_angle(current_heading),)
    return _fewest_covering_robot_headings(
        sorted({heading % (2.0 * math.pi) for heading in point_headings}),
        current_heading, camera_offset, effective_fov,
    )


# 局部可见性与覆盖复用


def frontier_observation_points(
    frame: NavigationFrame,
    frontier_cells: Iterable[Tuple[int, int]],
) -> Tuple[WorldPoint, ...]:
    """将边界格转换为局部可见的观察点，不判断它们能否作为移动目标。

    扫描传入移动筛选前的完整边界；距离和地图遮挡仍限制可观察范围。
    """
    origin = camera_world_position(frame)
    visibility_map = _visibility_map(frame)
    points = {
        grid_cell_center_to_world(row, col, frame.obstacle_map)
        for row, col in frontier_cells
    }
    return tuple(
        point for point in sorted(points)
        if 0.10 < math.dist(origin, point) <= OBSERVATION_RANGE_M
        and _has_map_line_of_sight(visibility_map, origin, point)
    )


def unobserved_observation_points(
    points: Sequence[WorldPoint],
    frame: NavigationFrame,
    views: Sequence[ObservationView],
) -> Tuple[WorldPoint, ...]:
    """复用已检查的同一区域；明显换角度或走近后允许重新检查。"""
    origin = camera_world_position(frame)
    coverage = {}
    checked_directions = set()
    for view in views:
        for point in view.visible_world_xy:
            coverage.setdefault(tuple(point), []).append(view)
        # 同一位置已经检查过此视锥，无需为仍被遮挡的点重复拍摄。
        # 新露出的点不在旧视锥的地图可见点列表中，仍然会补查。
        if math.dist(_view_origin(view), origin) <= SAME_POSITION_REUSE_M:
            checked_directions.update(tuple(point) for point in view.map_visible_world_xy)
    return tuple(
        point for point in points
        if point not in checked_directions
        and not any(_similar_viewpoint(point, origin, view) for view in coverage.get(point, ()))
    )


def capture_observation_view(
    frame: NavigationFrame,
    *,
    observation_points: Sequence[WorldPoint] = (),
) -> ObservationView:
    """冻结当前图像的覆盖；调用方仅在语义检查成功后将它加入 observed_views。"""
    camera_center_offset_rad, horizontal_fov_rad = horizontal_camera_view(frame)
    origin = camera_world_position(frame)
    heading = frame.pose.yaw_rad + camera_center_offset_rad
    visibility_map = _visibility_map(frame)
    depth_available = _aligned_depth_available(frame)
    map_points = set(_local_coverage_points(frame))
    # Frontier 格心未必落在固定采样网格上，显式记录才能按同一点复用检查结果。
    map_points.update(
        point for point in observation_points
        if 0.10 < math.dist(origin, point) <= OBSERVATION_RANGE_M
        and _has_map_line_of_sight(visibility_map, origin, point)
    )
    in_view = tuple(
        point for point in sorted(map_points)
        if abs(wrap_angle(_bearing(origin, point) - heading))
        <= max(0.0, horizontal_fov_rad / 2.0 - VIEW_EDGE_MARGIN_RAD)
    )
    visible = ()
    if depth_available:
        visible = tuple(
            point for point in in_view if _depth_supports_point(frame, point)
        )
    return ObservationView(
        pose=frame.pose,
        camera_heading_world_rad=heading,
        horizontal_fov_rad=horizontal_fov_rad,
        timestamp_s=frame.timestamp_s,
        camera_world_xy=origin,
        visible_world_xy=visible,
        map_visible_world_xy=in_view,
        depth_coverage_available=depth_available,
    )


def camera_world_position(frame: NavigationFrame) -> WorldPoint:
    """返回相机光心的世界平面位置，包含安装平移。"""
    extrinsics = frame.camera_extrinsics_in_robot
    cosine, sine = math.cos(frame.pose.yaw_rad), math.sin(frame.pose.yaw_rad)
    return (
        frame.pose.x_m + cosine * extrinsics.forward_m - sine * extrinsics.left_m,
        frame.pose.y_m + sine * extrinsics.forward_m + cosine * extrinsics.left_m,
    )


def scan_coverage_details(state: SearchState) -> Mapping[str, Any]:
    """保留本轮规划时的覆盖数量，首次图像同周期完成时也可在日志中查看。"""
    return {
        "local_observation_point_count": state.scan_local_point_count,
        "pending_observation_view_count": len(state.pending_observation_views),
        "blocked_frontier_region_count": len(state.blocked_frontier_regions),
        "observation_point_count": len(state.scan_observation_points),
        "reused_observation_point_count": max(
            0, state.scan_local_point_count - len(state.scan_observation_points)
        ),
    }


def horizontal_camera_view(frame: NavigationFrame) -> Tuple[float, float]:
    """返回相机水平视场中心相对底盘的偏角，以及完整水平 FOV。"""
    intrinsics = frame.camera_intrinsics
    if intrinsics is None:
        raise ValueError("缺少相机内参")
    if not math.isfinite(intrinsics.fx) or float(intrinsics.fx) <= 0.0:
        raise ValueError("相机 fx 必须为正有限值")
    if not math.isfinite(intrinsics.cx):
        raise ValueError("相机 cx 必须为有限值")

    width = _camera_image_width(frame)
    focal_length = float(intrinsics.fx)
    principal_x = float(intrinsics.cx)
    left_pixels = principal_x + 0.5
    right_pixels = width - 0.5 - principal_x
    if left_pixels <= 0.0 or right_pixels <= 0.0:
        raise ValueError("相机 cx 必须位于图像水平范围内")

    left_extent = math.atan2(left_pixels, focal_length)
    right_extent = math.atan2(right_pixels, focal_length)
    intrinsic_center_offset = (left_extent - right_extent) / 2.0
    camera_yaw = frame.camera_extrinsics_in_robot.yaw_rad
    if not math.isfinite(camera_yaw):
        raise ValueError("相机 yaw 外参必须为有限值")
    return (
        float(camera_yaw) + intrinsic_center_offset,
        left_extent + right_extent,
    )


# 内部朝向与覆盖计算


def _fewest_covering_robot_headings(
    circular_headings: Sequence[float],
    current_heading: float,
    camera_offset: float,
    field_of_view: float,
) -> Tuple[float, ...]:
    """尝试从每个目标方向切开圆周，保留视角最少的覆盖方案。"""
    full_turn = 2.0 * math.pi
    doubled_headings = tuple(circular_headings) + tuple(
        heading + full_turn for heading in circular_headings
    )
    best_headings = ()
    best_key = None
    point_count = len(circular_headings)
    for first_index in range(point_count):
        stop_index = first_index + point_count
        group_start = first_index
        camera_headings = []
        while group_start < stop_index:
            group_end = bisect_right(
                doubled_headings,
                doubled_headings[group_start] + field_of_view + 1e-12,
                group_start + 1,
                stop_index,
            ) - 1
            camera_headings.append(
                (doubled_headings[group_start] + doubled_headings[group_end])
                / 2.0
            )
            group_start = group_end + 1

        robot_headings = tuple(
            wrap_angle(camera_heading - camera_offset)
            for camera_heading in camera_headings
        )
        ordered_headings = _order_by_nearest_turn(
            robot_headings, current_heading
        )
        plan_key = (
            len(ordered_headings),
            _total_turn_distance(ordered_headings, current_heading),
            ordered_headings,
        )
        if best_key is None or plan_key < best_key:
            best_key = plan_key
            best_headings = ordered_headings
    return best_headings


def _order_by_nearest_turn(
    headings: Tuple[float, ...], current_heading: float
) -> Tuple[float, ...]:
    """每次优先选择转角最小的剩余朝向，减少无关往返旋转。"""
    remaining = list(headings)
    ordered = []
    reference_heading = current_heading
    while remaining:
        nearest_index = min(
            range(len(remaining)),
            key=lambda index: (
                abs(shortest_turn_to_heading(reference_heading, remaining[index])),
                remaining[index],
            ),
        )
        reference_heading = remaining.pop(nearest_index)
        ordered.append(reference_heading)
    return tuple(ordered)


def _total_turn_distance(
    headings: Tuple[float, ...], current_heading: float
) -> float:
    """返回依次执行扫描朝向时的累计绝对转角。"""
    total_turn = 0.0
    reference_heading = current_heading
    for heading in headings:
        total_turn += abs(shortest_turn_to_heading(reference_heading, heading))
        reference_heading = heading
    return total_turn


def _local_coverage_points(frame: NavigationFrame) -> Tuple[WorldPoint, ...]:
    """采样当前图像可记录的局部覆盖；这些点本身不用于发起扫描。"""
    origin = camera_world_position(frame)
    visibility_map = _visibility_map(frame)
    spacing = OBSERVATION_SPACING_M
    points = []
    for ix in range(
        math.ceil((origin[0] - OBSERVATION_RANGE_M) / spacing),
        math.floor((origin[0] + OBSERVATION_RANGE_M) / spacing) + 1,
    ):
        for iy in range(
            math.ceil((origin[1] - OBSERVATION_RANGE_M) / spacing),
            math.floor((origin[1] + OBSERVATION_RANGE_M) / spacing) + 1,
        ):
            point = (ix * spacing, iy * spacing)
            distance = math.dist(origin, point)
            if (
                MIN_OBSERVATION_DISTANCE_M <= distance <= OBSERVATION_RANGE_M
                and _has_map_line_of_sight(visibility_map, origin, point)
            ):
                points.append(point)
    # 记录首层未知边界的覆盖，后续地图公开时可复用；不推测边界后方可见。
    points.extend(_visible_unknown_boundary_points(visibility_map, origin))
    return tuple(sorted(set(points)))


def _visible_unknown_boundary_points(
    obstacle_map: ObstacleMap, origin: WorldPoint,
) -> Tuple[WorldPoint, ...]:
    """返回视线能到达的首层未知格，仅用于当前图像的覆盖记录。"""
    grid = obstacle_map.occupancy
    row, col = world_to_nearest_grid_cell(origin, obstacle_map)
    radius = math.ceil(OBSERVATION_RANGE_M / obstacle_map.resolution_m) + 1
    points = []
    for r in range(max(0, row - radius), min(len(grid), row + radius + 1)):
        for c in range(max(0, col - radius), min(len(grid[0]), col + radius + 1)):
            if grid[r][c] is not None or not any(
                _free_cell(grid, neighbor)
                for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            ):
                continue
            point = grid_cell_center_to_world(r, c, obstacle_map)
            if (
                0.10 < math.dist(origin, point) <= OBSERVATION_RANGE_M
                and _has_map_line_of_sight(obstacle_map, origin, point)
            ):
                points.append(point)
    return tuple(points)


def _visibility_map(frame: NavigationFrame) -> ObstacleMap:
    """视觉射线使用未膨胀遮挡图，避免把机器人净空带当作真实障碍。"""
    return frame.visibility_map if frame.visibility_map is not None else frame.obstacle_map


def _has_map_line_of_sight(
    obstacle_map: ObstacleMap, origin: WorldPoint, target: WorldPoint,
) -> bool:
    """只允许视线穿过已知自由格；终点障碍可观察，禁止穿过对角墙角。"""
    grid = obstacle_map.occupancy
    start = world_to_nearest_grid_cell(origin, obstacle_map)
    end = world_to_nearest_grid_cell(target, obstacle_map)
    if not (0 <= end[0] < len(grid) and 0 <= end[1] < len(grid[0])):
        return False
    steps = max(1, math.ceil(math.dist(origin, target) / (obstacle_map.resolution_m / 2.0)))
    previous = start
    for index in range(1, steps + 1):
        fraction = index / steps
        cell = world_to_nearest_grid_cell((
            origin[0] + (target[0] - origin[0]) * fraction,
            origin[1] + (target[1] - origin[1]) * fraction,
        ), obstacle_map)
        if cell[0] != previous[0] and cell[1] != previous[1]:
            if not all(_free_cell(grid, side) for side in (
                (previous[0], cell[1]), (cell[0], previous[1]),
            )):
                return False
        if cell == end:
            return True
        if cell != start and not _free_cell(grid, cell):
            return False
        previous = cell
    return True


def _free_cell(grid, cell: Tuple[int, int]) -> bool:
    row, col = cell
    return (
        0 <= row < len(grid) and 0 <= col < len(grid[0])
        and grid[row][col] is not None and grid[row][col] <= 0.5
    )


def _view_origin(view: ObservationView) -> WorldPoint:
    return view.camera_world_xy or (view.pose.x_m, view.pose.y_m)


def _bearing(origin: WorldPoint, point: WorldPoint) -> float:
    return math.atan2(point[1] - origin[1], point[0] - origin[0])


def _similar_viewpoint(point: WorldPoint, origin: WorldPoint, view: ObservationView) -> bool:
    previous = _view_origin(view)
    angle_change = abs(wrap_angle(_bearing(previous, point) - _bearing(origin, point)))
    previous_distance = math.dist(previous, point)
    distance = math.dist(origin, point)
    # 显著接近后，原本很小的物体可能变得可辨认；不能仅因坐标曾覆盖而跳过。
    return (
        angle_change <= VIEWPOINT_CHANGE_RAD
        and previous_distance <= max(distance * 1.5, distance + 0.5)
    )


def _aligned_depth_available(frame: NavigationFrame) -> bool:
    """检查 RGB 与深度的尺寸及可用内参；实际深度对齐由 Adapter 保证。"""
    depth, rgb, intrinsics = frame.depth, frame.rgb, frame.camera_intrinsics
    if depth is None or rgb is None or intrinsics is None:
        return False
    height = len(depth)
    width = len(depth[0]) if height else 0
    return (
        height > 0 and width > 0 and len(rgb) == height
        and all(len(row) == width for row in depth)
        and all(len(row) == width for row in rgb)
        and all(math.isfinite(float(value)) for value in (
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
        ))
        and intrinsics.fx > 0.0 and intrinsics.fy > 0.0
    )


def _depth_supports_point(frame: NavigationFrame, point: WorldPoint) -> bool:
    """在相机高度及其上下 0.25 m 投影采样；任一高度被深度遮挡则不登记覆盖。"""
    intrinsics = frame.camera_intrinsics
    depth = frame.depth
    extrinsics = frame.camera_extrinsics_in_robot
    forward, left = world_point_to_robot(point, frame.pose)
    forward -= extrinsics.forward_m
    left -= extrinsics.left_m
    cyaw, syaw = math.cos(extrinsics.yaw_rad), math.sin(extrinsics.yaw_rad)
    pitched_forward = cyaw * forward + syaw * left
    rolled_left = -syaw * forward + cyaw * left
    cp, sp = math.cos(extrinsics.pitch_down_rad), math.sin(extrinsics.pitch_down_rad)
    cr, sr = math.cos(extrinsics.roll_rad), math.sin(extrinsics.roll_rad)
    for up in (-0.25, 0.0, 0.25):
        camera_forward = cp * pitched_forward - sp * up
        rolled_up = sp * pitched_forward + cp * up
        camera_left = cr * rolled_left - sr * rolled_up
        camera_up = sr * rolled_left + cr * rolled_up
        if camera_forward <= 0.0:
            return False
        col = round(intrinsics.cx - intrinsics.fx * camera_left / camera_forward)
        row = round(intrinsics.cy - intrinsics.fy * camera_up / camera_forward)
        if not (1 <= row < len(depth) - 1 and 1 <= col < len(depth[0]) - 1):
            return False
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            value = depth[row + dy][col + dx]
            if value is None:
                return False
            if (
                not math.isfinite(value) or value <= 0.0
                or camera_forward > value + DEPTH_MARGIN_M
            ):
                return False
    return True


def _camera_image_width(frame: NavigationFrame) -> int:
    """返回与相机内参对应的图像宽度；优先使用 RGB，随后使用对齐深度。"""
    image = frame.rgb if frame.rgb is not None else frame.depth
    if image is None:
        raise ValueError("缺少 RGB 或深度图像尺寸")
    width = len(image[0]) if image else 0
    if width < 1:
        raise ValueError("相机图像宽度必须大于零")
    return width
