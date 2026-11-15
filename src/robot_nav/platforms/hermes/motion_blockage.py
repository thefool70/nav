"""用 MoveToAction 期间的底盘位姿判断阻塞，并用深度细化人工墙位置。"""

from __future__ import annotations

import math
import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

from robot_nav.interfaces.geometry import wrap_angle
from robot_nav.interfaces.types import NavigationFrame, Pose2D


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
# 人工墙膨胀半径为 0.36 m；墙中心至少留在机器人当前位置前方 0.60 m。
MIN_WALL_DISTANCE_FROM_ROBOT_M = 0.60
WALL_BEFORE_OBSTACLE_M = 0.15
RETRY_BLOCKAGE_DURATION_S = 3.0


@dataclass(frozen=True)
class FrontObstruction:
    """持续近距离障碍的位置；墙与机器人到障碍的方向垂直。"""

    center_world_xy: Tuple[float, float]
    heading_world_rad: float
    depth_m: Optional[float]
    duration_s: float
    sample_count: int
    confirmed_after_retry: bool


@dataclass(frozen=True)
class FrontObstructionProgress:
    """最近一次滑动窗口比较，供动作日志显示实际时长、位移和角差。"""

    timestamp_s: float
    retrying: bool
    duration_s: float
    depth_m: Optional[float]
    displacement_m: float
    heading_change_rad: float
    sample_count: int


@dataclass(frozen=True)
class _NearObservation:
    timestamp_s: float
    depth_m: float
    obstacle_world_xy: Tuple[float, float]
    robot_pose: Pose2D


class FrontObstructionDetector:
    """比较当前与一个时间窗口前的位姿，判断平移动作是否受阻。"""

    def __init__(self, *, maximum_depth_m: float, duration_s: float, movement_radius_m: float) -> None:
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
        # 首次停滞后的原目标重发正在复查；有效平移或动作结束即清除。
        self._retrying = False
        # observe 更新；只保留窗口内位姿及窗口边界之前最近的一帧，动作结束清空。
        self._poses: deque[Tuple[float, Pose2D]] = deque()
        self._samples: list[_NearObservation] = []
        self._target_world_xy: Optional[Tuple[float, float]] = None

    def start_translation(self, target_world_xy: Optional[Tuple[float, float]], *, retrying: bool = False) -> None:
        """新平移动作从首帧积累检测窗口；首次与重试均无额外起步宽限。"""
        with self._lock:
            self._active = True
            self._triggered = False
            self._retrying = retrying
            self._poses.clear()
            self._samples.clear()
            self._target_world_xy = target_world_xy

    def stop_translation(self) -> None:
        with self._lock:
            self._active = False
            self._retrying = False
            self._poses.clear()
            self._samples.clear()
            self._target_world_xy = None

    def observe(self, frame: NavigationFrame) -> Optional[FrontObstruction]:
        """只按窗口净位移判停滞；转向仅记录，不作为脱困依据。"""
        with self._lock:
            if not self._active or self._triggered:
                return None
            timestamp_s = float(frame.timestamp_s)
            if self._poses and timestamp_s <= self._poses[-1][0]:
                return None
            self._poses.append((timestamp_s, frame.pose))
            window_s = RETRY_BLOCKAGE_DURATION_S if self._retrying else self._duration_s
            cutoff_s = timestamp_s - window_s
            # 没有恰好窗口边界时刻的帧时取该时刻之前最近的帧，不插值猜测历史位姿。
            while len(self._poses) > 1 and self._poses[1][0] <= cutoff_s:
                self._poses.popleft()
            self._samples = [item for item in self._samples if item.timestamp_s >= cutoff_s]
            near = _front_observation(frame, self._maximum_depth_m)
            self._record_near_observation(timestamp_s, frame.pose, near)
            comparison = self._window_comparison()
            moved = comparison.displacement_m >= self._movement_radius_m
            if self._retrying and moved:
                # 已产生有效平移，这次停滞结束；新位置重新积累普通窗口。
                self._retrying = False
                self._poses.clear()
                self._poses.append((timestamp_s, frame.pose))
                self._samples.clear()
                self._record_near_observation(timestamp_s, frame.pose, near)
                return None
            if comparison.duration_s < window_s or moved:
                return None
            self._triggered = True
            return _obstruction_from_window(
                self._samples, comparison.duration_s, comparison.sample_count,
                frame.pose, self._target_world_xy, self._retrying,
            )

    def progress(self) -> Optional[FrontObstructionProgress]:
        """返回窗口两端的比较结果；不额外维护一份位姿或计时状态。"""
        with self._lock:
            if not self._active or self._triggered or not self._poses:
                return None
            return self._window_comparison()

    def wall_for_unknown_path(self, stopped_pose: Pose2D) -> FrontObstruction:
        """路径超限已取消确认后，复用当前深度样本和原目标生成墙；不要求停滞。"""
        with self._lock:
            duration_s = self._poses[-1][0] - self._poses[0][0] if self._poses else 0.0
            return _obstruction_from_window(
                self._samples, duration_s, len(self._samples),
                stopped_pose, self._target_world_xy, False,
            )

    def _window_comparison(self) -> FrontObstructionProgress:
        """调用方持锁；同一计算同时用于受阻判断和日志。"""
        started_s, previous = self._poses[0]
        timestamp_s, current = self._poses[-1]
        return FrontObstructionProgress(
            timestamp_s=timestamp_s, retrying=self._retrying,
            duration_s=timestamp_s - started_s,
            depth_m=self._samples[-1].depth_m if self._samples else None,
            displacement_m=math.hypot(current.x_m - previous.x_m, current.y_m - previous.y_m),
            heading_change_rad=abs(wrap_angle(current.yaw_rad - previous.yaw_rad)),
            sample_count=len(self._samples),
        )

    def _record_near_observation(self, timestamp_s, pose, near) -> None:
        """只保留窗口内同一片近障碍的有效样本；深度不决定是否触发。"""
        if near is None:
            return
        depth_m, obstacle_world_xy = near
        if self._samples:
            first_xy = self._samples[0].obstacle_world_xy
            if math.hypot(obstacle_world_xy[0] - first_xy[0],
                          obstacle_world_xy[1] - first_xy[1]) > MAX_OBSTACLE_SPREAD_M:
                return
        self._samples.append(_NearObservation(timestamp_s, depth_m, obstacle_world_xy, pose))


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
    robot_pose: Pose2D,
    target_world_xy: Optional[Tuple[float, float]],
    confirmed_after_retry: bool,
) -> FrontObstruction:
    depth_m = None
    if samples:
        obstacle_x = statistics.median(item.obstacle_world_xy[0] for item in samples)
        obstacle_y = statistics.median(item.obstacle_world_xy[1] for item in samples)
        heading = math.atan2(
            obstacle_y - robot_pose.y_m,
            obstacle_x - robot_pose.x_m,
        )
        obstacle_distance = math.hypot(
            obstacle_x - robot_pose.x_m,
            obstacle_y - robot_pose.y_m,
        )
        wall_distance = max(
            MIN_WALL_DISTANCE_FROM_ROBOT_M,
            obstacle_distance - WALL_BEFORE_OBSTACLE_M,
        )
        depth_m = statistics.median(item.depth_m for item in samples)
    else:
        target_x, target_y = target_world_xy or (
            robot_pose.x_m + math.cos(robot_pose.yaw_rad),
            robot_pose.y_m + math.sin(robot_pose.yaw_rad),
        )
        if math.hypot(target_x - robot_pose.x_m, target_y - robot_pose.y_m) > 1e-6:
            heading = math.atan2(
                target_y - robot_pose.y_m,
                target_x - robot_pose.x_m,
            )
        else:
            heading = robot_pose.yaw_rad
        wall_distance = MIN_WALL_DISTANCE_FROM_ROBOT_M
    return FrontObstruction(
        center_world_xy=(
            robot_pose.x_m + wall_distance * math.cos(heading),
            robot_pose.y_m + wall_distance * math.sin(heading),
        ),
        heading_world_rad=heading,
        depth_m=depth_m,
        duration_s=duration_s,
        sample_count=sample_count,
        confirmed_after_retry=confirmed_after_retry,
    )


__all__ = [
    "FrontObstruction",
    "FrontObstructionDetector",
    "FrontObstructionProgress",
]
