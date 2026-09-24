"""用 MoveToAction 期间的底盘位姿判断阻塞，并用深度细化人工墙位置。"""

from __future__ import annotations

import math
import statistics
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

from ...core.models import NavigationFrame, Pose2D


# 摆头时障碍会离开图像中央，因此观察大部分前向视野，只避开两侧边缘和近处地面。
ROI_LEFT = 0.10
ROI_RIGHT = 0.90
ROI_TOP = 0.15
ROI_BOTTOM = 0.65
MIN_DEPTH_M = 0.10
# 用局部连续深度簇识别人或近物体，避免人在画面边缘时被整幅 ROI 占比稀释。
MIN_NEAR_CLUSTER_SAMPLE_COUNT = 12
MAX_NEIGHBOR_DEPTH_DELTA_M = 0.20
# 同一个人的深度点会落在身体不同部位，世界坐标允许人体尺度内的变化。
MAX_OBSTACLE_SPREAD_M = 0.45
# 人工墙膨胀半径为 0.36 m；墙中心至少留在机器人起点前方 0.60 m。
MIN_WALL_DISTANCE_FROM_ROBOT_M = 0.60
WALL_BEFORE_OBSTACLE_M = 0.15


@dataclass(frozen=True)
class FrontObstruction:
    """持续近距离障碍的位置；墙与机器人到障碍的方向垂直。"""

    center_world_xy: Tuple[float, float]
    heading_world_rad: float
    depth_m: Optional[float]
    duration_s: float
    sample_count: int


@dataclass(frozen=True)
class FrontObstructionProgress:
    """当前连续挡路候选，供动作日志显示计时是否正在累计。"""

    started_s: float
    duration_s: float
    depth_m: Optional[float]
    displacement_m: float
    sample_count: int
    restart_reason: str


@dataclass(frozen=True)
class _NearObservation:
    timestamp_s: float
    depth_m: float
    obstacle_world_xy: Tuple[float, float]
    robot_pose: Pose2D


class FrontObstructionDetector:
    """平移动作中确认底盘长时间没有离开指定半径。"""

    def __init__(
        self,
        *,
        maximum_depth_m: float,
        duration_s: float,
        movement_radius_m: float,
    ) -> None:
        if not math.isfinite(maximum_depth_m) or maximum_depth_m <= MIN_DEPTH_M:
            raise ValueError("maximum_depth_m 必须大于最小有效深度")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s 必须为正有限数")
        if not math.isfinite(movement_radius_m) or movement_radius_m <= 0.0:
            raise ValueError("movement_radius_m 必须为正有限数")
        self._maximum_depth_m = float(maximum_depth_m)
        self._duration_s = float(duration_s)
        self._movement_radius_m = float(movement_radius_m)
        self._lock = threading.Lock()
        self._active = False
        self._triggered = False
        self._samples: list[_NearObservation] = []
        self._near_observation_count = 0
        self._anchor_timestamp_s: Optional[float] = None
        self._anchor_pose: Optional[Pose2D] = None
        self._latest_timestamp_s: Optional[float] = None
        self._latest_pose: Optional[Pose2D] = None
        self._target_world_xy: Optional[Tuple[float, float]] = None
        self._restart_reason = ""

    def start_translation(
        self,
        target_world_xy: Optional[Tuple[float, float]],
    ) -> None:
        """开始一次平移动作，按底盘位移启动新的阻塞观察窗口。"""
        with self._lock:
            self._active = True
            self._triggered = False
            self._samples.clear()
            self._near_observation_count = 0
            self._anchor_timestamp_s = None
            self._anchor_pose = None
            self._latest_timestamp_s = None
            self._latest_pose = None
            self._target_world_xy = target_world_xy
            self._restart_reason = ""

    def stop_translation(self) -> None:
        with self._lock:
            self._active = False
            self._samples.clear()
            self._near_observation_count = 0
            self._anchor_timestamp_s = None
            self._anchor_pose = None
            self._latest_timestamp_s = None
            self._latest_pose = None
            self._target_world_xy = None

    def observe(self, frame: NavigationFrame) -> Optional[FrontObstruction]:
        """接收一帧；底盘在配置半径内停留满时长后返回一次墙中心。"""
        with self._lock:
            if not self._active or self._triggered:
                return None

        timestamp_s = float(frame.timestamp_s)
        near = _front_observation(frame, self._maximum_depth_m)
        with self._lock:
            if not self._active or self._triggered:
                return None

            if self._anchor_timestamp_s is None or self._anchor_pose is None:
                self._anchor_timestamp_s = timestamp_s
                self._anchor_pose = frame.pose
                self._latest_timestamp_s = timestamp_s
                self._latest_pose = frame.pose
                self._record_near_observation(timestamp_s, frame.pose, near)
                return None

            if self._latest_timestamp_s is not None and timestamp_s <= self._latest_timestamp_s:
                return None
            moved_m = math.hypot(
                frame.pose.x_m - self._anchor_pose.x_m,
                frame.pose.y_m - self._anchor_pose.y_m,
            )
            if moved_m >= self._movement_radius_m:
                self._restart_reason = f"机器人平移达到 {moved_m:.3f}m"
                self._samples.clear()
                self._near_observation_count = 0
                self._anchor_timestamp_s = timestamp_s
                self._anchor_pose = frame.pose
                self._latest_timestamp_s = timestamp_s
                self._latest_pose = frame.pose
                self._record_near_observation(timestamp_s, frame.pose, near)
                return None

            self._latest_timestamp_s = timestamp_s
            self._latest_pose = frame.pose
            self._record_near_observation(timestamp_s, frame.pose, near)
            duration_s = timestamp_s - self._anchor_timestamp_s
            if duration_s < self._duration_s:
                return None

            self._triggered = True
            return _obstruction_from_window(
                self._samples,
                duration_s,
                self._near_observation_count,
                self._anchor_pose,
                self._target_world_xy,
            )

    def _record_near_observation(
        self,
        timestamp_s: float,
        pose: Pose2D,
        near: Optional[Tuple[float, Tuple[float, float]]],
    ) -> None:
        """深度只细化墙的位置；看不到近障碍不影响位姿停滞计时。"""
        if near is None:
            return
        depth_m, obstacle_world_xy = near
        self._near_observation_count += 1
        if self._samples:
            first_xy = self._samples[0].obstacle_world_xy
            if math.hypot(
                obstacle_world_xy[0] - first_xy[0],
                obstacle_world_xy[1] - first_xy[1],
            ) > MAX_OBSTACLE_SPREAD_M:
                return
        self._samples.append(
            _NearObservation(timestamp_s, depth_m, obstacle_world_xy, pose)
        )

    def progress(self) -> Optional[FrontObstructionProgress]:
        """返回当前候选的累计时间；不参与检测决策。"""
        with self._lock:
            if (
                not self._active
                or self._triggered
                or self._anchor_timestamp_s is None
                or self._anchor_pose is None
            ):
                return None
            if self._latest_timestamp_s is None or self._latest_pose is None:
                return None
            return FrontObstructionProgress(
                started_s=self._anchor_timestamp_s,
                duration_s=self._latest_timestamp_s - self._anchor_timestamp_s,
                depth_m=self._samples[-1].depth_m if self._samples else None,
                displacement_m=math.hypot(
                    self._latest_pose.x_m - self._anchor_pose.x_m,
                    self._latest_pose.y_m - self._anchor_pose.y_m,
                ),
                sample_count=self._near_observation_count,
                restart_reason=self._restart_reason,
            )


def _front_observation(
    frame: NavigationFrame,
    maximum_depth_m: float,
) -> Optional[Tuple[float, Tuple[float, float]]]:
    """返回宽前向区域内近障碍的深度和世界坐标；稀疏噪点不算挡路。"""
    depth = frame.depth
    intrinsics = frame.camera_intrinsics
    if depth is None or not depth or intrinsics is None or intrinsics.fx <= 0.0:
        return None
    height = len(depth)
    width = len(depth[0]) if height else 0
    if width < 1:
        return None
    first_row, last_row = int(height * ROI_TOP), max(1, int(height * ROI_BOTTOM))
    first_col, last_col = int(width * ROI_LEFT), max(1, int(width * ROI_RIGHT))
    row_step = max(1, (last_row - first_row) // 40)
    col_step = max(1, (last_col - first_col) // 60)
    near_by_grid: dict[Tuple[int, int], Tuple[float, int]] = {}
    for sample_row, row in enumerate(range(first_row, last_row, row_step)):
        for sample_col, col in enumerate(range(first_col, last_col, col_step)):
            value = depth[row][col]
            if value is None or not math.isfinite(value) or value <= 0.0:
                continue
            value = float(value)
            if MIN_DEPTH_M <= value <= maximum_depth_m:
                near_by_grid[(sample_row, sample_col)] = (value, col)
    near_cluster = _largest_near_cluster(near_by_grid)
    if len(near_cluster) < MIN_NEAR_CLUSTER_SAMPLE_COUNT:
        return None

    depth_m = statistics.median(item[0] for item in near_cluster)
    column = statistics.median(item[1] for item in near_cluster)
    return depth_m, _depth_point_in_world(frame, depth_m, column)


def _largest_near_cluster(
    near_by_grid: dict[Tuple[int, int], Tuple[float, int]],
) -> list[Tuple[float, int]]:
    """返回采样网格中深度连续的最大八邻接簇，滤掉零散近深度噪点。"""
    remaining = set(near_by_grid)
    largest: list[Tuple[float, int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        cluster = [near_by_grid[start]]
        while stack:
            row, col = stack.pop()
            depth_m = near_by_grid[(row, col)][0]
            for row_offset in (-1, 0, 1):
                for col_offset in (-1, 0, 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor = (row + row_offset, col + col_offset)
                    if neighbor not in remaining:
                        continue
                    neighbor_depth = near_by_grid[neighbor][0]
                    if abs(neighbor_depth - depth_m) > MAX_NEIGHBOR_DEPTH_DELTA_M:
                        continue
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    cluster.append(near_by_grid[neighbor])
        if len(cluster) > len(largest):
            largest = cluster
    return largest


def _depth_point_in_world(
    frame: NavigationFrame,
    depth_m: float,
    column: float,
) -> Tuple[float, float]:
    """把深度点按当前摆头角度投到世界平面，供跨帧确认是同一片障碍。"""
    intrinsics = frame.camera_intrinsics
    assert intrinsics is not None
    extrinsics = frame.camera_extrinsics_in_robot
    robot_cosine = math.cos(frame.pose.yaw_rad)
    robot_sine = math.sin(frame.pose.yaw_rad)
    camera_x = (
        frame.pose.x_m
        + extrinsics.forward_m * robot_cosine
        - extrinsics.left_m * robot_sine
    )
    camera_y = (
        frame.pose.y_m
        + extrinsics.forward_m * robot_sine
        + extrinsics.left_m * robot_cosine
    )
    camera_forward = depth_m * math.cos(extrinsics.pitch_down_rad)
    camera_left = (float(intrinsics.cx) - column) * depth_m / float(intrinsics.fx)
    camera_heading = frame.pose.yaw_rad + extrinsics.yaw_rad
    heading_cosine = math.cos(camera_heading)
    heading_sine = math.sin(camera_heading)
    return (
        camera_x + camera_forward * heading_cosine - camera_left * heading_sine,
        camera_y + camera_forward * heading_sine + camera_left * heading_cosine,
    )


def _obstruction_from_window(
    samples: list[_NearObservation],
    duration_s: float,
    sample_count: int,
    anchor_pose: Pose2D,
    target_world_xy: Optional[Tuple[float, float]],
) -> FrontObstruction:
    depth_m = None
    if samples:
        obstacle_x = statistics.median(item.obstacle_world_xy[0] for item in samples)
        obstacle_y = statistics.median(item.obstacle_world_xy[1] for item in samples)
        heading = math.atan2(
            obstacle_y - anchor_pose.y_m,
            obstacle_x - anchor_pose.x_m,
        )
        obstacle_distance = math.hypot(
            obstacle_x - anchor_pose.x_m,
            obstacle_y - anchor_pose.y_m,
        )
        wall_distance = max(
            MIN_WALL_DISTANCE_FROM_ROBOT_M,
            obstacle_distance - WALL_BEFORE_OBSTACLE_M,
        )
        depth_m = statistics.median(item.depth_m for item in samples)
    else:
        target_x, target_y = target_world_xy or (
            anchor_pose.x_m + math.cos(anchor_pose.yaw_rad),
            anchor_pose.y_m + math.sin(anchor_pose.yaw_rad),
        )
        if math.hypot(target_x - anchor_pose.x_m, target_y - anchor_pose.y_m) > 1e-6:
            heading = math.atan2(
                target_y - anchor_pose.y_m,
                target_x - anchor_pose.x_m,
            )
        else:
            heading = anchor_pose.yaw_rad
        wall_distance = MIN_WALL_DISTANCE_FROM_ROBOT_M
    return FrontObstruction(
        center_world_xy=(
            anchor_pose.x_m + wall_distance * math.cos(heading),
            anchor_pose.y_m + wall_distance * math.sin(heading),
        ),
        heading_world_rad=heading,
        depth_m=depth_m,
        duration_s=duration_s,
        sample_count=sample_count,
    )


__all__ = [
    "FrontObstruction",
    "FrontObstructionDetector",
    "FrontObstructionProgress",
]
