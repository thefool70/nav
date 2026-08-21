"""扫描朝向规划工具。无隐藏状态，非法输入抛 ValueError。

约定：角度单位为弧度（逆时针为正），朝向均为世界坐标系下的方向。
"""

from __future__ import annotations

import math
from typing import Tuple

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
