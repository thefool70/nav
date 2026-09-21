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

    navigation_map 可提供同坐标系的完整、未膨胀障碍图，仅用于物体定位与接近；
    navigation_clearance_m 为该图选停靠点时采用的机器人净空半径（米）。
    未提供时沿用 obstacle_map，不额外膨胀。Frontier 与探索路径仍使用 obstacle_map。

    visibility_map 是同坐标系的未膨胀视觉遮挡图，公开范围遵循 Adapter 的观察规则；
    仅用于观测方向与覆盖判定，不用于路径规划。未提供时沿用 obstacle_map。

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
    navigation_map: Optional[ObstacleMap] = None
    navigation_clearance_m: float = 0.0
    visibility_map: Optional[ObstacleMap] = None
    # Adapter 的组帧诊断快照；仅供日志读取，不参与决策。
    acquisition_timings: Tuple[Mapping[str, Any], ...] = field(default=(), compare=False, repr=False)


class SearchMode(str, Enum):
    """搜索语义：寻找具体物体，或判断机器人是否进入目的场景。"""

    OBJECT = "object"
    SCENE = "scene"


@dataclass(frozen=True)
class TargetSearchGoal:
    """语义搜索目标；target_text 描述物体或目的场景。"""

    target_text: str
    search_mode: SearchMode = SearchMode.OBJECT


class TargetVisibility(Enum):
    """可见性结果；UNCERTAIN 表示感知失败，PENDING 表示采集后等待分析。"""

    VISIBLE = "visible"
    NOT_VISIBLE = "not_visible"
    UNCERTAIN = "uncertain"
    PENDING = "pending"


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
    """历史画面中 VLM 检测结论的诊断值，不触发到达后确认。"""

    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class SemanticAnalysis:
    """检测到目标的有序画面与评分；view_ids 空元组表示未检测到，None 表示检测失败。"""

    target_view_ids: Optional[Tuple[int, ...]]
    frontier_scores: Mapping[str, float] = field(default_factory=dict)
    detection_error: str = ""
    scoring_error: str = ""
    interaction_id: Optional[int] = None


@dataclass(frozen=True)
class TargetClue:
    """目标线索的拍摄位姿与快照编号；pose 是机器人位姿，不是物体位置。"""

    clue_id: str
    pose: Pose2D
    timestamp_s: float
    map_frame_id: str
    job_id: Optional[int] = None
    view_id: Optional[int] = None


@dataclass(frozen=True)
class ObjectLocalization:
    """地图系物体位置；source 区分 RGB-D 定位与 bbox_obstacle/front_obstacle 假设。"""

    target_world_xy: Optional[Tuple[float, float]] = None
    visibility: TargetVisibility = TargetVisibility.UNCERTAIN
    vlm_confirmation: TargetConfirmation = TargetConfirmation.UNCERTAIN
    source: str = ""
    sample_count: int = 0
    reason: str = ""


@dataclass(frozen=True)
class ObjectApproachState:
    """当前线索的接近状态，以及处理整批历史线索时保留的返回保底位姿。"""

    target: Optional[ObjectLocalization] = None
    destination: Optional[Pose2D] = None
    tried_positions: Tuple[Tuple[float, float], ...] = ()
    fallback_clue: Optional[TargetClue] = None
    history_localized: bool = False
    capture_turns: int = 0


@dataclass(frozen=True)
class ObservationView:
    """实际采集的水平视角；进入 observed_views 后才代表图像已完成语义检查。

    map_visible_world_xy 是地图可见视锥，仅原地复用；visible_world_xy 进一步
    经过深度遮挡检查，用于跨位置复用。坐标单位米。
    """

    pose: Pose2D
    camera_heading_world_rad: float
    horizontal_fov_rad: float
    timestamp_s: float
    camera_world_xy: Optional[Tuple[float, float]] = None
    visible_world_xy: Tuple[Tuple[float, float], ...] = ()
    map_visible_world_xy: Tuple[Tuple[float, float], ...] = ()
    depth_coverage_available: bool = False


@dataclass(frozen=True)
class ScanEvidence:
    """一次扫描中单个方向的采集证据。

    场景模式不做逐帧目标检测，此时 NOT_VISIBLE 只表示该方向已经完成采集。
    """

    heading_world_rad: float
    visibility: TargetVisibility
    view: Optional[ObservationView] = None


@dataclass(frozen=True)
class FrontierCandidate:
    """一次扫描中发现的前沿候选点。row、col 为在障碍图中的栅格坐标，
    world_xy 为世界坐标（米）；heading_world_rad 为朝向该候选点的世界
    系方向（弧度）；frontier_cells 保存该前沿包含的全部栅格；
    frontier_cell_count 为其栅格数；frontier_span_m 为聚类完整栅格包围框的
    对角跨度（米）；path_distance_m 为沿路径到该点的距离（米）；
    score 为新候选之间的探索优先级；deferred_order 非空时按暂存顺序恢复。"""

    candidate_id: str
    row: int
    col: int
    world_xy: Tuple[float, float]
    heading_world_rad: float
    frontier_cells: Tuple[Tuple[int, int], ...]
    frontier_cell_count: int
    frontier_span_m: float
    path_distance_m: float
    score: float
    semantic_score: Optional[float] = None
    deferred_order: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class FrontierRegion:
    """跨地图更新关联的 Frontier 区域，边界使用世界坐标。

    deferred_order 为暂存时的（观测节点序号，本轮排名）；None 表示可优先探索。
    暂存方向按节点从新到旧、同节点排名从小到大恢复。"""

    region_id: str
    boundary_world_xy: Tuple[Tuple[float, float], ...]
    deferred_order: Optional[Tuple[int, int]] = None


@dataclass(frozen=True)
class BlockedFrontierRegion:
    """因未知路径长度超限被屏蔽的整片边界，独立于当前候选及其编号保存。

    boundary_world_xy 保留已关联的区域边界；本次运行中不自动解除屏蔽。
    """

    region_id: str
    boundary_world_xy: Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class FrontierScoreRequest:
    """本轮允许参与语义评分的新 Frontier 候选，暂存旧方向不参与。"""

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
    """语义目标搜索的阶段。只保留异步视觉队列导航流程使用的阶段。"""

    SCANNING = "scanning"
    EXPLORING = "exploring"
    BACKTRACKING = "backtracking"
    WAITING_FOR_SEMANTICS = "waiting_for_semantics"
    REVISITING_TARGET = "revisiting_target"
    LOCALIZING_OBJECT = "localizing_object"
    APPROACHING_OBJECT = "approaching_object"
    STOPPED = "stopped"
    COMPLETE = "complete"
    FAILED = "failed"


class SearchDirectionState(Enum):
    """单个搜索方向的状态。"""

    PENDING = "pending"
    COMMITTED = "committed"
    EXPLORED = "explored"
    INVALIDATED = "invalidated"
    STALLED = "stalled"


@dataclass(frozen=True)
class SearchDirection:
    """一次已记录的搜索方向。heading_world_rad 为世界坐标系下的朝向（弧度），
    candidate_world_xy 为最终 Frontier，command_world_xy 为提交给底盘的位置（米）；
    当前探索命令直接使用最终位置。
    state 记录本次移动结果，execution_reason 保留执行异常原因。"""

    direction_id: str
    heading_world_rad: float
    candidate_world_xy: Optional[Tuple[float, float]] = None
    state: SearchDirectionState = SearchDirectionState.PENDING
    command_world_xy: Optional[Tuple[float, float]] = None
    execution_reason: str = ""


@dataclass(frozen=True)
class ObservationNode:
    """一次探索移动的出发位置和实际目标；也是本轮暂存方向的父节点。"""

    node_id: str
    position_world_xy: Tuple[float, float]
    directions: Tuple[SearchDirection, ...]


@dataclass(frozen=True)
class SearchState:
    """语义目标搜索的周期状态。scan_headings_world_rad 为世界系扫描朝向
    序列，next_scan_index 为下一个待扫描朝向的下标，observation_history
    按时间顺序保存观测节点，scan_evidence 保存最近一次扫描的逐方向观测
    证据；observed_views 只记录已完成语义检查的视角。
    pending_observation_views 单独保存待分析覆盖，仅用于避免重复采集。
    pending_semantic_jobs 含在途、待接收结果、采样和排队线索，不是 HTTP 请求数。
    scan_observation_points 保存本轮局部待检查 Frontier 边界点，随扫描计划冻结；
    scan_local_point_count 是复用已检查覆盖前的局部可见 Frontier 点数。
    frontier_regions 保存当前有效区域及旧方向的暂存顺序；active_frontier_id
    标识最近选择的区域，扫描转向不改变新旧方向的优先级。
    blocked_frontier_regions 保留因未知路径长度超限被取消的完整区域，防止换代表点重试。
    branch_node_ids 按根到叶保存当前分支的出发节点；逐个返回，已退完节点出栈。
    backtrack_node_id 是正在返回的栈顶父节点，到达后才检查该节点的暂存方向。
    active_target_clue 在物体定位、接近和保底返回期间保留拍摄位姿与快照编号。
    object_approach 保存定位和重试记录，以及所有历史无法定位时的保底返回线索。

    本状态只由搜索核心更新；感知模块与运行层只读取，不代为修改。"""

    phase: SearchPhase = SearchPhase.SCANNING
    scan_headings_world_rad: Tuple[float, ...] = ()
    next_scan_index: int = 0
    observation_history: Tuple[ObservationNode, ...] = ()
    scan_evidence: Tuple[ScanEvidence, ...] = ()
    frontier_regions: Tuple[FrontierRegion, ...] = ()
    # 最近一次 Frontier 提取的孔洞过滤统计，每次刷新覆盖，不是累计屏蔽。
    frontier_hole_filter_applied: bool = False
    ignored_frontier_hole_count: int = 0
    ignored_frontier_hole_area_m2: float = 0.0
    ignored_frontier_cell_count: int = 0
    next_frontier_region_id: int = 0
    active_frontier_id: Optional[str] = None
    observed_views: Tuple[ObservationView, ...] = ()
    scan_observation_points: Tuple[Tuple[float, float], ...] = ()
    scan_local_point_count: int = 0
    initial_scan_complete: bool = False
    blocked_frontier_regions: Tuple[BlockedFrontierRegion, ...] = ()
    backtrack_node_id: Optional[str] = None
    branch_node_ids: Tuple[str, ...] = ()
    asynchronous_perception: bool = False
    pending_semantic_jobs: int = 0
    failed_semantic_jobs: int = 0
    active_target_clue: Optional[TargetClue] = None
    object_approach: ObjectApproachState = field(default_factory=ObjectApproachState)
    pending_observation_views: Tuple[ObservationView, ...] = ()


class NavigationStatus(Enum):
    """导航单周期结果的总体状态。"""

    OK = "ok"
    INVALID_INPUT = "invalid_input"
    NO_SOLUTION = "no_solution"
    NEEDS_SCAN_CAPTURE = "needs_scan_capture"
    NEEDS_FRONTIER_SCORES = "needs_frontier_scores"
    NEEDS_OBJECT_LOCALIZATION = "needs_object_localization"
    MISSING_DATA = "missing_data"


class ActionKind(Enum):
    """搜索核心要求执行的动作类型。运行层按类型执行，不解析字符串步骤名。

    MOVE_TO_POSE 指定世界系目标位姿，路径约束单独声明；TURN_IN_PLACE 为原地转向；
    MOVE_RELATIVE 为不校验未知路径的直接相对移动。"""

    MOVE_TO_POSE = "move_to_pose"
    TURN_IN_PLACE = "turn_in_place"
    MOVE_RELATIVE = "move_relative"


class ActionConstraint(Enum):
    """动作的执行约束。REQUIRE_KNOWN_PATH 表示未知路径超限时按可恢复失败处理。"""

    NONE = "none"
    REQUIRE_KNOWN_PATH = "require_known_path"


class ActionPurpose(Enum):
    """动作的算法用途，供采样策略和失败恢复使用。"""

    OTHER = "other"
    SCAN_TURN = "scan.turn"
    EXPLORE = "explore.move"
    RESUME = "explore.resume"
    BACKTRACK = "backtrack.return"
    REVISIT = "target.revisit"
    REVISIT_TURN = "target.revisit_turn"
    APPROACH = "object.approach"
    FALLBACK = "object.fallback_return"
    FALLBACK_TURN = "object.fallback_turn"


@dataclass(frozen=True)
class NavigationAction:
    """搜索核心请求执行的一个动作。

    destination 为世界系目标位姿；command 为相对移动量；两者按
    action 类型择一。purpose 用于核心解释执行结果后决定后续，例如区分
    "扫描转向"与"接近目标转向"的可恢复失败处理。
    """

    action: ActionKind
    constraint: ActionConstraint = ActionConstraint.NONE
    destination: Optional[Pose2D] = None
    command: Optional[RelativePoseCommand] = None
    purpose: ActionPurpose = ActionPurpose.OTHER
    candidate_id: Optional[str] = None
    node_id: Optional[str] = None


class ActionOutcome(Enum):
    """动作的执行结果类别；SUCCEEDED 以外的可恢复类别由核心决定后续。"""

    SUCCEEDED = "succeeded"
    STALLED = "stalled"
    INTERRUPTED = "interrupted"
    PATH_UNKNOWN = "path_unknown"


@dataclass(frozen=True)
class ActionExecutionResult:
    """运行层交付的同步执行反馈；设备离线等系统错误直接抛出异常。"""

    outcome: ActionOutcome
    reason: str = ""
    rejected_path_world_xy: tuple = ()
    unknown_length_m: Optional[float] = None
    limit_m: Optional[float] = None
    total_path_length_m: Optional[float] = None


@dataclass(frozen=True)
class NavigationDebug:
    """供排错使用的可读信息。stage 为当前算法步骤的说明性标签，
    message 为人类可读说明，details 为附加键值。

    本结构与动作执行无关：运行层按 NavigationAction 执行动作，
    不读取 stage 字符串来判断或改变行为。"""

    stage: str
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NavigationResult:
    """导航单周期输出。action 仅在 status 为 OK 时有意义，否则为 None；
    state 为周期结束后的显式搜索状态，调用方应将其作为下一周期的输入。"""

    status: NavigationStatus
    action: Optional[NavigationAction]
    debug: NavigationDebug
    state: SearchState
    frontier_score_request: Optional[FrontierScoreRequest] = None
