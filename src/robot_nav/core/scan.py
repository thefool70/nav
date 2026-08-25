"""扫描朝向规划工具。无隐藏状态，非法输入抛 ValueError。

约定：角度单位为弧度（逆时针为正），朝向均为世界坐标系下的方向。
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import Sequence, Tuple

from .geometry import wrap_angle


def _require_finite_angle(value, name):
    """返回 float(value)，拒绝 bool、不可转换与非有限角度，统一抛 ValueError。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite angle")
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite angle") from None
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite angle")
    return converted


def build_uniform_scan_headings(
    start_heading_rad: float, view_count: int = 4
) -> Tuple[float, ...]:
    """返回从 start_heading_rad 起绕一周均匀分布的 view_count 个扫描朝向（弧度），
    默认四向（相对起点 0°、90°、180°、270°）。

    返回朝向全部用 wrap_angle 归一化到 [-π, π)。view_count 必须为正整数
    （不含 bool），start_heading_rad 必须为有限角度。
    """
    if isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 1:
        raise ValueError("view_count must be a positive integer")
    start_heading = _require_finite_angle(start_heading_rad, "start_heading_rad")
    step_rad = 2.0 * math.pi / view_count
    return tuple(
        wrap_angle(start_heading + i * step_rad) for i in range(view_count)
    )


def build_covering_scan_headings(
    point_headings_world_rad: Sequence[float],
    current_robot_heading_rad: float,
    camera_center_offset_rad: float,
    horizontal_fov_rad: float,
) -> Tuple[float, ...]:
    """返回能覆盖全部目标方向的机器人扫描朝向。

    ``point_headings_world_rad`` 是待观察点相对机器人的世界系方位角；
    ``camera_center_offset_rad`` 是相机水平视场中心相对机器人正前方的偏角。
    点已经全部位于当前视野时只返回当前朝向，否则尝试每个圆周切点，并选择
    视角数量最少、连续转角较短的覆盖方案，避免固定扫描无关区域。
    """
    current_heading = _require_finite_angle(
        current_robot_heading_rad, "current_robot_heading_rad"
    )
    camera_offset = _require_finite_angle(
        camera_center_offset_rad, "camera_center_offset_rad"
    )
    field_of_view = _require_finite_angle(
        horizontal_fov_rad, "horizontal_fov_rad"
    )
    if not 0.0 < field_of_view <= 2.0 * math.pi:
        raise ValueError("horizontal_fov_rad must be in (0, 2π]")

    point_headings = tuple(
        _require_finite_angle(value, "point_headings_world_rad item")
        for value in point_headings_world_rad
    )
    if not point_headings:
        return ()

    current_camera_heading = wrap_angle(current_heading + camera_offset)
    half_fov = field_of_view / 2.0
    if all(
        abs(wrap_angle(point_heading - current_camera_heading)) <= half_fov
        for point_heading in point_headings
    ):
        return (wrap_angle(current_heading),)

    circular_headings = sorted(
        {point_heading % (2.0 * math.pi) for point_heading in point_headings}
    )
    return _fewest_covering_robot_headings(
        circular_headings,
        current_heading,
        camera_offset,
        field_of_view,
    )


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


def shortest_turn_to_heading(
    current_heading_rad: float, target_heading_rad: float
) -> float:
    """返回从当前朝向转到目标朝向的有符号最短角差，范围 [-π, π)（弧度）。"""
    current_heading = _require_finite_angle(current_heading_rad, "current_heading_rad")
    target_heading = _require_finite_angle(target_heading_rad, "target_heading_rad")
    return wrap_angle(target_heading - current_heading)


def heading_to_world_direction(heading_rad: float) -> Tuple[float, float]:
    """返回朝向对应的世界系单位方向向量 (x, y)。"""
    heading = _require_finite_angle(heading_rad, "heading_rad")
    return (math.cos(heading), math.sin(heading))
