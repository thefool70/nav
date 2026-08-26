"""导航算法内部数据契约，全部只依赖标准库。

约定：长度单位为米，角度单位为弧度（逆时针为正）。除显式标注外，
坐标系由使用处上下文（如 NavigationFrame.pose 的 map.frame_id）决定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple

# 栅格数据：外层为行（y），内层为列（x），None 表示该格未知/无效。
Grid = Sequence[Sequence[Optional[float]]]
# 深度图：单位为米，None 表示该像素无有效深度。
DepthImage = Sequence[Sequence[Optional[float]]]
# 目标掩码：True 表示像素属于目标，尺寸应与 RGB/对齐深度一致。
MaskImage = Sequence[Sequence[bool]]
# 单像素 RGB 三元组，取值 0-255。
RGB = Tuple[int, int, int]
# RGB 图像：外层为行（y），内层为列（x）。
RgbImage = Sequence[Sequence[RGB]]


@dataclass(frozen=True)
class Pose2D:
    """二维位姿。x_m、y_m 单位为米，yaw_rad 逆时针为正。坐标系由使用处声明。"""

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参。fx、fy 单位为像素，cx、cy 为主点像素坐标。"""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraExtrinsics:
    """相机光心在机器人前/左/上坐标系中的安装外参。

    yaw_rad 向左为正，pitch_down_rad 向下为正；roll_rad 表示从相机后方向
    镜头看时图像顺时针倾斜为正。长度单位为米，角度单位为弧度。
    """

    forward_m: float = 0.0
    left_m: float = 0.0
    height_m: float = 0.0
    yaw_rad: float = 0.0
    pitch_down_rad: float = 0.0
    roll_rad: float = 0.0


@dataclass(frozen=True)
class NavigationFrame:
    """单周期感知快照。timestamp_s 为采集时刻（秒）；pose 为机器人位姿；
    obstacle_map 为障碍图；depth 与 rgb 可选，depth 单位米；
    camera_intrinsics 为可选相机内参；camera_extrinsics_in_robot 为相机在
    机器人局部前/左/上坐标系中的六自由度外参（无相机时保持默认零外参）。

    契约：pose 在进入 core 前必须已转换到 obstacle_map.frame_id 坐标系，
    core 内部不再做坐标转换。
    """

    timestamp_s: float
    pose: Pose2D
    obstacle_map: ObstacleMap
    depth: Optional[DepthImage] = None
    rgb: Optional[RgbImage] = None
    camera_intrinsics: Optional[CameraIntrinsics] = None
    camera_extrinsics_in_robot: CameraExtrinsics = field(
        default_factory=CameraExtrinsics
    )


@dataclass(frozen=True)
class TargetSearchGoal:
    """语义目标搜索目标，target_text 为对目标的人类可读描述（如 "门口"）。"""

    target_text: str


class TargetVisibility(Enum):
    """感知输出可见或不可见；UNCERTAIN 仅表示内部感知失败。"""

    VISIBLE = "visible"
    NOT_VISIBLE = "not_visible"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class TargetObservation:
    """感知模块对单帧语义目标的观测结果。

    bbox_norm 为归一化包围盒 (x_min, y_min, x_max, y_max)，取值 0-1；
    target_mask 是与 RGB/对齐深度同尺寸的目标像素掩码；reason 只记录观测
    失败等内部诊断；source 和 confidence 用于区分 YOLO/VLM 等来源及其置信度。
    """

    visibility: TargetVisibility
    bbox_norm: Optional[Tuple[float, float, float, float]] = None
    target_mask: Optional[MaskImage] = None
    reason: str = ""
    source: str = ""
    confidence: Optional[float] = None


class TargetConfirmation(Enum):
    """接近候选目标后的 VLM 最终确认结果。"""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class TargetConfirmationResult:
    """VLM 最终确认及其可选失败原因。"""

    confirmation: TargetConfirmation
    reason: str = ""


@dataclass(frozen=True)
class ScanEvidence:
    """一次扫描中单个方向的目标可见性证据。"""

    heading_world_rad: float
    visibility: TargetVisibility


@dataclass(frozen=True)
class FrontierCandidate:
    """一次扫描中发现的前沿候选点。row、col 为在障碍图中的栅格坐标，
    world_xy 为世界坐标（米）；heading_world_rad 为朝向该候选点的世界
    系方向（弧度）；frontier_cells 保存该前沿包含的全部栅格；
    frontier_cell_count 为其栅格数；path_distance_m 为沿路径到该点的距离
    （米）；score 为探索优先级。"""

    candidate_id: str
    row: int
    col: int
    world_xy: Tuple[float, float]
    heading_world_rad: float
    frontier_cells: Tuple[Tuple[int, int], ...]
    frontier_cell_count: int
    path_distance_m: float
    score: float
    semantic_score: Optional[float] = None


@dataclass(frozen=True)
class FrontierScoreRequest:
    """一次批量语义评分所要覆盖的全部 Frontier 候选。"""

    candidates: Tuple[FrontierCandidate, ...]


@dataclass(frozen=True)
class TargetEstimate:
    """对目标位置的估计结果。target_base_xy 为目标在机器人 base 坐标系下
    的坐标（米，前 x 左 y），target_world_xy 为世界坐标系坐标（米）；
    distance_m 为距离（米），bearing_rad 为机器人局部系方位角（弧度）；
    sample_count 为参与估计的观测样本数。失败时 target_base_xy、
    target_world_xy、distance_m、bearing_rad 为 None。"""

    success: bool
    reason: str
    target_base_xy: Optional[Tuple[float, float]] = None
    target_world_xy: Optional[Tuple[float, float]] = None
    distance_m: Optional[float] = None
    bearing_rad: Optional[float] = None
    sample_count: int = 0


@dataclass(frozen=True)
class RelativePoseCommand:
    """相对机器人当前位姿的移动量。forward_m 向前、left_m 向左、yaw_rad 逆时针为正。"""

    forward_m: float = 0.0
    left_m: float = 0.0
    yaw_rad: float = 0.0


class SearchPhase(Enum):
    """语义目标搜索的阶段。"""

    SCANNING = "scanning"
    LOCALIZING_TARGET = "localizing_target"
    VERIFYING_TARGET = "verifying_target"
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


@dataclass(frozen=True)
class SearchDirection:
    """一次已记录的搜索方向。heading_world_rad 为世界坐标系下的朝向（弧度），
    candidate_world_xy 为可选的目标候选点（米），state 为方向状态。"""

    direction_id: str
    heading_world_rad: float
    candidate_world_xy: Optional[Tuple[float, float]] = None
    state: SearchDirectionState = SearchDirectionState.PENDING


@dataclass(frozen=True)
class ObservationNode:
    """一次观测时机器人所在位置及其在该位置记录的方向序列。"""

    node_id: str
    position_world_xy: Tuple[float, float]
    directions: Tuple[SearchDirection, ...]


@dataclass(frozen=True)
class SearchState:
    """语义目标搜索的周期状态。scan_headings_world_rad 为世界系扫描朝向
    序列，next_scan_index 为下一个待扫描朝向的下标，observation_history
    按时间顺序保存观测节点，scan_evidence 保存最近一次扫描的逐方向观测
    证据；initial_scan_complete 区分首次 8×45° 环扫与后续 Frontier 视场扫描；
    active_node_id 为当前活跃观测节点，target_approach_attempts 为已尝试接近
    目标的次数；pending_target_world_xy 是等待最终确认的目标位置，
    rejected_target_world_xy 保存已被 VLM 否决的位置。"""

    phase: SearchPhase = SearchPhase.SCANNING
    scan_headings_world_rad: Tuple[float, ...] = ()
    next_scan_index: int = 0
    observation_history: Tuple[ObservationNode, ...] = ()
    scan_evidence: Tuple[ScanEvidence, ...] = ()
    active_node_id: Optional[str] = None
    target_approach_attempts: int = 0
    initial_scan_complete: bool = False
    pending_target_world_xy: Optional[Tuple[float, float]] = None
    rejected_target_world_xy: Tuple[Tuple[float, float], ...] = ()


class NavigationStatus(Enum):
    """导航单周期结果的总体状态。"""

    OK = "ok"
    INVALID_INPUT = "invalid_input"
    NO_SOLUTION = "no_solution"
    NEEDS_OBSERVATION = "needs_observation"
    NEEDS_FRONTIER_SCORES = "needs_frontier_scores"
    NEEDS_TARGET_CONFIRMATION = "needs_target_confirmation"
    MISSING_DATA = "missing_data"


@dataclass(frozen=True)
class NavigationDebug:
    """供排错使用的可读信息。stage 为当前算法步骤，message 为人类可读说明，
    details 为附加键值。"""

    stage: str
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NavigationResult:
    """导航单周期输出。command 仅在 status 为 OK 时有意义，否则为 None；
    state 为周期结束后的显式搜索状态，调用方应将其作为下一周期的输入。"""

    status: NavigationStatus
    command: Optional[RelativePoseCommand]
    debug: NavigationDebug
    state: SearchState
    frontier_score_request: Optional[FrontierScoreRequest] = None
