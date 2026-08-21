"""观测历史管理：不可变观测节点与方向状态更新。非法输入抛简短 ValueError。

约定：观测节点按输入顺序保存（输入顺序即时间顺序），方向在节点内保持输入
顺序；所有返回对象均为不可变新对象。
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Sequence, Tuple

from .geometry import wrap_angle
from .models import (
    ObservationNode,
    SearchDirection,
    SearchDirectionState,
)


def _finite_number(value) -> Optional[float]:
    """value 为可转换的有限数（不含 bool）时返回 float(value)，否则 None。"""
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _require_world_xy(value, name) -> Tuple[float, float]:
    """校验 value 为两个有限数并返回规范化 float 二元组，否则抛 ValueError。"""
    try:
        xy = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be two finite numbers") from None
    if len(xy) != 2:
        raise ValueError(f"{name} must be two finite numbers")
    first = _finite_number(xy[0])
    second = _finite_number(xy[1])
    if first is None or second is None:
        raise ValueError(f"{name} must be two finite numbers")
    return (first, second)


def freeze_observation_node(
    node_id: str,
    position_world_xy: Tuple[float, float],
    directions: Sequence[SearchDirection],
    committed_direction_id: str,
) -> ObservationNode:
    """把一次观测固化为不可变 ObservationNode，方向保持输入顺序。

    node_id、direction_id 与 committed_direction_id 去除首尾空白后使用，
    空白字符串非法；heading_world_rad 转为有限 float 并归一化到 [-π, π)；
    candidate_world_xy 若提供必须是两个有限数，保存为规范化 float 二元组。
    committed_direction_id 必须存在于 directions 中；对应方向标为 COMMITTED，
    其余方向标为 PENDING。
    """
    if not isinstance(node_id, str) or not node_id.strip():
        raise ValueError("node_id must be a non-empty string")
    node_id = node_id.strip()
    position = _require_world_xy(position_world_xy, "position_world_xy")
    try:
        directions = tuple(directions)
    except TypeError:
        raise ValueError("directions must be a sequence of SearchDirection") from None
    if not all(isinstance(direction, SearchDirection) for direction in directions):
        raise ValueError("directions must contain only SearchDirection")
    if not isinstance(committed_direction_id, str):
        raise ValueError("committed_direction_id must be an existing direction_id")
    committed_direction_id = committed_direction_id.strip()
    direction_ids = []
    frozen_directions = []
    for direction in directions:
        if (
            not isinstance(direction.direction_id, str)
            or not direction.direction_id.strip()
        ):
            raise ValueError("direction_id must be a non-empty string")
        direction_id = direction.direction_id.strip()
        if direction_id in direction_ids:
            raise ValueError("direction_id must be unique")
        direction_ids.append(direction_id)
        heading = _finite_number(direction.heading_world_rad)
        if heading is None:
            raise ValueError("heading_world_rad must be finite")
        candidate = (
            _require_world_xy(direction.candidate_world_xy, "candidate_world_xy")
            if direction.candidate_world_xy is not None
            else None
        )
        frozen_directions.append(
            SearchDirection(
                direction_id=direction_id,
                heading_world_rad=wrap_angle(heading),
                candidate_world_xy=candidate,
                state=(
                    SearchDirectionState.COMMITTED
                    if direction_id == committed_direction_id
                    else SearchDirectionState.PENDING
                ),
            )
        )
    if committed_direction_id not in direction_ids:
        raise ValueError("committed_direction_id must be an existing direction_id")
    return ObservationNode(
        node_id=node_id,
        position_world_xy=position,
        directions=tuple(frozen_directions),
    )


def set_observation_direction_state(
    node: ObservationNode,
    direction_id: str,
    new_state: SearchDirectionState,
) -> ObservationNode:
    """返回把 node 中 direction_id 方向状态替换为 new_state 后的新节点。

    其余方向保持原样与顺序；direction_id 不存在时抛 ValueError。
    """
    if not isinstance(node, ObservationNode):
        raise ValueError("node must be an ObservationNode")
    if not isinstance(direction_id, str) or not direction_id:
        raise ValueError("direction_id must be a non-empty string")
    if not isinstance(new_state, SearchDirectionState):
        raise ValueError("new_state must be a SearchDirectionState")
    if not any(
        direction.direction_id == direction_id for direction in node.directions
    ):
        raise ValueError("direction_id not found")
    return replace(
        node,
        directions=tuple(
            replace(direction, state=new_state)
            if direction.direction_id == direction_id
            else direction
            for direction in node.directions
        ),
    )


def find_latest_pending_observation_node(
    nodes: Sequence[ObservationNode],
) -> Optional[ObservationNode]:
    """返回仍含 pending 方向的最新观测节点（输入顺序中越靠后越新），否则 None。"""
    try:
        ordered = tuple(reversed(nodes))
    except TypeError:
        raise ValueError("nodes must be a sequence of ObservationNode") from None
    for node in ordered:
        if not isinstance(node, ObservationNode):
            raise ValueError("nodes must contain only ObservationNode")
        if any(
            direction.state is SearchDirectionState.PENDING
            for direction in node.directions
        ):
            return node
    return None
