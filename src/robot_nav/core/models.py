"""导航算法内部数据契约，全部只依赖标准库。

约定：长度单位为米，角度单位为弧度（逆时针为正）。除显式标注外，
坐标系由使用处上下文（如 NavigationFrame.pose 的 map.frame_id）决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple, TypeAlias

# 栅格数据：外层为行（y），内层为列（x），None 表示该格未知/无效。
Grid: TypeAlias = Sequence[Sequence[Optional[float]]]
# 深度图：单位为米，None 表示该像素无有效深度。
DepthImage: TypeAlias = Sequence[Sequence[Optional[float]]]
# 单像素 RGB 三元组，取值 0-255。
RGB: TypeAlias = Tuple[int, int, int]
# RGB 图像：外层为行（y），内层为列（x）。
RgbImage: TypeAlias = Sequence[Sequence[RGB]]


@dataclass(frozen=True, slots=True)
class Pose2D:
    """二维位姿。x_m、y_m 单位为米，yaw_rad 逆时针为正。坐标系由使用处声明。"""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True, slots=True)
class ObstacleMap:
    """占用栅格。occupancy[row][col]，值 0.0 自由、1.0 占用、None 未知。

    resolution_m 为每格边长（米/格）；origin 为格 (row=0, col=0) 中心在世界
    坐标系中的位姿，列沿 origin 局部 +x 方向增长，行沿 origin 局部 +y 方向
    增长；frame_id 为地图所在坐标系。
    """

    occupancy: Grid
    resolution_m: float
    origin: Pose2D
    frame_id: str


@dataclass(frozen=True, slots=True)
class NavigationFrame:
    """单周期感知快照。timestamp_s 为采集时刻（秒）；pose 为机器人位姿；
    obstacle_map 为障碍图；depth 与 rgb 可选，depth 单位米。

    契约：pose 在进入 core 前必须已转换到 obstacle_map.frame_id 坐标系，
    core 内部不再做坐标转换。
    """

    timestamp_s: float
    pose: Pose2D
    obstacle_map: ObstacleMap
    depth: Optional[DepthImage] = None
    rgb: Optional[RgbImage] = None


@dataclass(frozen=True, slots=True)
class TargetSearchGoal:
    """语义目标搜索目标，target_text 为对目标的人类可读描述（如 "门口"）。"""

    target_text: str


@dataclass(frozen=True, slots=True)
class RelativePoseCommand:
    """相对机器人当前位姿的移动量。forward_m 向前、left_m 向左、yaw_rad 逆时针为正。"""

    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_rad: float = 0.0


class SearchPhase(Enum):
    """语义目标搜索的阶段。"""

    SCANNING = "scanning"
    LOCALIZING_TARGET = "localizing_target"
    EXPLORING = "exploring"
    BACKTRACKING = "backtracking"
    COMPLETE = "complete"
    FAILED = "failed"


class SearchDirectionState(Enum):
    """单个搜索方向的状态。"""

    PENDING = "pending"
    COMMITTED = "committed"
    EXPLORED = "explored"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class SearchDirection:
    """一次已记录的搜索方向。heading_world_rad 为世界坐标系下的朝向（弧度），
    candidate_world_xy 为可选的目标候选点（米），state 为方向状态。"""

    direction_id: str
    heading_world_rad: float
    candidate_world_xy: Optional[Tuple[float, float]] = None
    state: SearchDirectionState = SearchDirectionState.PENDING


@dataclass(frozen=True, slots=True)
class ObservationNode:
    """一次观测时机器人所在位置及其在该位置记录的方向序列。"""

    node_id: str
    position_world_xy: Tuple[float, float]
    directions: Tuple[SearchDirection, ...]


@dataclass(frozen=True, slots=True)
class SearchState:
    """语义目标搜索的周期状态。scan_headings_world_rad 为世界系扫描朝向
    序列，next_scan_index 为下一个待扫描朝向的下标，observation_history
    按时间顺序保存观测节点。"""

    phase: SearchPhase = SearchPhase.SCANNING
    scan_headings_world_rad: Tuple[float, ...] = ()
    next_scan_index: int = 0
    observation_history: Tuple[ObservationNode, ...] = ()


class NavigationStatus(Enum):
    """导航单周期结果的总体状态。"""

    OK = "ok"
    INVALID_INPUT = "invalid_input"
    NO_SOLUTION = "no_solution"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True, slots=True)
class NavigationDebug:
    """供排错使用的可读信息。stage 为当前算法步骤，message 为人类可读说明，
    details 为附加键值。"""

    stage: str
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NavigationResult:
    """导航单周期输出。command 仅在 status 为 OK 时有意义，否则为 None；
    state 为周期结束后的显式搜索状态，调用方应将其作为下一周期的输入。"""

    status: NavigationStatus
    command: Optional[RelativePoseCommand]
    debug: NavigationDebug
    state: SearchState
