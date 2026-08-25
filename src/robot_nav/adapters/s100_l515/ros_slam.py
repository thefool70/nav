"""把 ROS 2 SLAM 话题转换为项目统一的 NavigationFrame。"""

from __future__ import annotations

import importlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from ...core.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    NavigationFrame,
    ObstacleMap,
    Pose2D,
)
from .mapping import CameraMount


@dataclass(frozen=True)
class RosSlamConfig:
    """SLAM 话题、帧名和数据有效性配置。"""

    map_topic: str = "/map"
    pose_topic: str = "/pose"
    rgb_topic: str = "/camera/camera/color/image_raw"
    depth_topic: str = "/camera/camera/aligned_depth_to_color/image_raw"
    camera_info_topic: str = "/camera/camera/color/camera_info"
    map_frame: str = "map"
    odom_frame: str = "odom"
    base_frame: str = "base_link"
    camera_frame: str = "camera_color_frame"
    frame_timeout_s: float = 15.0
    stale_after_s: float = 5.0
    map_stale_after_s: float = 10.0
    max_rgb_depth_skew_s: float = 0.08
    odom_publish_hz: float = 20.0
    depth_unit_m: float = 0.00025
    free_occupancy_max: int = 20
    occupied_occupancy_min: int = 65


@dataclass(frozen=True)
class _ReceivedMessage:
    message: Any
    received_at: float


class RosSlamSource:
    """订阅 slam_toolbox 与 RealSense，并持续发布轮速里程计 TF。"""

    def __init__(
        self,
        config: RosSlamConfig,
        camera_mount: CameraMount,
        odometry_pose: Callable[[], Pose2D],
    ) -> None:
        _validate_config(config)
        if not isinstance(camera_mount, CameraMount):
            raise ValueError("camera_mount 必须为 CameraMount")
        if not callable(odometry_pose):
            raise ValueError("odometry_pose 必须可调用")

        self.config = config
        self._camera_mount = camera_mount
        self._odometry_pose = odometry_pose
        self._condition = threading.Condition()
        self._map: Optional[_ReceivedMessage] = None
        self._pose: Optional[_ReceivedMessage] = None
        self._rgb: Optional[_ReceivedMessage] = None
        self._depth: Optional[_ReceivedMessage] = None
        self._camera_info: Optional[_ReceivedMessage] = None
        self._last_returned_pose_at = 0.0
        self._background_error: Optional[BaseException] = None
        self._closed = False

        try:
            self._load_ros_modules()
            self._context = self._rclpy.Context()
            self._rclpy.init(args=[], context=self._context)
            self._node = self._rclpy.create_node(
                "robot_nav_s100_l515_bridge",
                context=self._context,
            )
            self._bridge = self._CvBridge()
            self._subscriptions = []
            self._create_subscriptions()
            self._tf = self._TransformBroadcaster(self._node)
            self._static_tf = self._StaticTransformBroadcaster(self._node)
            self._publish_camera_mount()
            self._timer = self._node.create_timer(
                1.0 / config.odom_publish_hz,
                self._publish_odometry,
            )
            self._executor = self._SingleThreadedExecutor(
                context=self._context
            )
            self._executor.add_node(self._node)
            self._thread = threading.Thread(
                target=self._spin,
                name="robot-nav-ros-slam",
                daemon=True,
            )
            self._thread.start()
        except Exception:
            self.close()
            raise

    def read_frame(self) -> NavigationFrame:
        """等待一组新的 SLAM 位姿、地图与同步 RGB-D，并返回统一导航帧。"""
        deadline = time.monotonic() + self.config.frame_timeout_s
        with self._condition:
            while True:
                if self._background_error is not None:
                    raise RuntimeError(
                        f"ROS SLAM 后台线程失败：{self._background_error}"
                    ) from self._background_error
                snapshot = self._ready_snapshot(time.monotonic())
                if snapshot is not None:
                    self._last_returned_pose_at = snapshot[1].received_at
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "等待 ROS SLAM 数据超时：" + self._missing_data_reason()
                    )
                self._condition.wait(remaining)

        map_data, pose, rgb, depth, camera_info = snapshot
        return self._build_frame(
            map_data.message,
            pose.message,
            rgb.message,
            depth.message,
            camera_info.message,
        )

    def close(self) -> None:
        """停止 ROS executor，并释放本类创建的独立 ROS context。"""
        if self._closed:
            return
        self._closed = True
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        thread = getattr(self, "_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        node = getattr(self, "_node", None)
        if node is not None:
            node.destroy_node()
        context = getattr(self, "_context", None)
        if context is not None:
            context.try_shutdown()

    def _load_ros_modules(self) -> None:
        """延迟导入 ROS，使普通模式无需安装 ROS 2。"""
        try:
            self._rclpy = importlib.import_module("rclpy")
            qos = importlib.import_module("rclpy.qos")
            executors = importlib.import_module("rclpy.executors")
            geometry_msgs = importlib.import_module("geometry_msgs.msg")
            nav_msgs = importlib.import_module("nav_msgs.msg")
            sensor_msgs = importlib.import_module("sensor_msgs.msg")
            tf2_ros = importlib.import_module("tf2_ros")
            cv_bridge = importlib.import_module("cv_bridge")
            self._np = importlib.import_module("numpy")
        except ImportError as exc:
            raise ImportError(
                "ROS SLAM 模式需要 robot-nav-slam 环境"
            ) from exc

        self._QoSProfile = qos.QoSProfile
        self._ReliabilityPolicy = qos.ReliabilityPolicy
        self._DurabilityPolicy = qos.DurabilityPolicy
        self._SingleThreadedExecutor = executors.SingleThreadedExecutor
        self._TransformStamped = geometry_msgs.TransformStamped
        self._OccupancyGrid = nav_msgs.OccupancyGrid
        self._PoseWithCovarianceStamped = (
            geometry_msgs.PoseWithCovarianceStamped
        )
        self._Image = sensor_msgs.Image
        self._CameraInfo = sensor_msgs.CameraInfo
        self._TransformBroadcaster = tf2_ros.TransformBroadcaster
        self._StaticTransformBroadcaster = tf2_ros.StaticTransformBroadcaster
        self._CvBridge = cv_bridge.CvBridge

    def _create_subscriptions(self) -> None:
        sensor_qos = self._QoSProfile(
            depth=1,
            reliability=self._ReliabilityPolicy.BEST_EFFORT,
            durability=self._DurabilityPolicy.VOLATILE,
        )
        reliable_qos = self._QoSProfile(depth=5)
        map_qos = self._QoSProfile(
            depth=1,
            reliability=self._ReliabilityPolicy.RELIABLE,
            durability=self._DurabilityPolicy.TRANSIENT_LOCAL,
        )
        specifications = (
            (self._OccupancyGrid, self.config.map_topic, self._on_map, map_qos),
            (
                self._PoseWithCovarianceStamped,
                self.config.pose_topic,
                self._on_pose,
                reliable_qos,
            ),
            (self._Image, self.config.rgb_topic, self._on_rgb, sensor_qos),
            (self._Image, self.config.depth_topic, self._on_depth, sensor_qos),
            (
                self._CameraInfo,
                self.config.camera_info_topic,
                self._on_camera_info,
                sensor_qos,
            ),
        )
        for message_type, topic, callback, qos in specifications:
            self._subscriptions.append(
                self._node.create_subscription(
                    message_type,
                    topic,
                    callback,
                    qos,
                )
            )

    def _spin(self) -> None:
        try:
            self._executor.spin()
        except BaseException as exc:
            with self._condition:
                self._background_error = exc
                self._condition.notify_all()

    def _on_map(self, message: Any) -> None:
        self._store("_map", message)

    def _on_pose(self, message: Any) -> None:
        self._store("_pose", message)

    def _on_rgb(self, message: Any) -> None:
        self._store("_rgb", message)

    def _on_depth(self, message: Any) -> None:
        self._store("_depth", message)

    def _on_camera_info(self, message: Any) -> None:
        self._store("_camera_info", message)

    def _store(self, attribute: str, message: Any) -> None:
        with self._condition:
            setattr(
                self,
                attribute,
                _ReceivedMessage(message, time.monotonic()),
            )
            self._condition.notify_all()

    def _ready_snapshot(
        self,
        now: float,
    ) -> Optional[Tuple[_ReceivedMessage, ...]]:
        values = (
            self._map,
            self._pose,
            self._rgb,
            self._depth,
            self._camera_info,
        )
        if any(value is None for value in values):
            return None
        map_data, pose, rgb, depth, camera_info = values
        assert all(value is not None for value in values)
        if pose.received_at <= self._last_returned_pose_at:
            return None
        if now - map_data.received_at > self.config.map_stale_after_s:
            return None
        for value in (pose, rgb, depth, camera_info):
            if now - value.received_at > self.config.stale_after_s:
                return None
        if abs(
            _message_stamp_s(rgb.message) - _message_stamp_s(depth.message)
        ) > self.config.max_rgb_depth_skew_s:
            return None
        return map_data, pose, rgb, depth, camera_info

    def _missing_data_reason(self) -> str:
        missing = [
            topic
            for topic, value in (
                (self.config.map_topic, self._map),
                (self.config.pose_topic, self._pose),
                (self.config.rgb_topic, self._rgb),
                (self.config.depth_topic, self._depth),
                (self.config.camera_info_topic, self._camera_info),
            )
            if value is None
        ]
        if missing:
            return "尚未收到 " + ", ".join(missing)
        return "数据过期、RGB-D 不同步，或尚无新的 SLAM 位姿"

    def _publish_camera_mount(self) -> None:
        transform = self._TransformStamped()
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.header.frame_id = self.config.base_frame
        transform.child_frame_id = self.config.camera_frame
        transform.transform.translation.x = self._camera_mount.forward_m
        transform.transform.translation.y = self._camera_mount.left_m
        transform.transform.translation.z = self._camera_mount.height_m
        quaternion = _mount_quaternion(
            self._camera_mount.yaw_rad,
            self._camera_mount.pitch_down_rad,
            self._camera_mount.roll_rad,
        )
        _set_quaternion(transform.transform.rotation, quaternion)
        self._static_tf.sendTransform(transform)

    def _publish_odometry(self) -> None:
        pose = self._odometry_pose()
        transform = self._TransformStamped()
        transform.header.stamp = self._node.get_clock().now().to_msg()
        transform.header.frame_id = self.config.odom_frame
        transform.child_frame_id = self.config.base_frame
        transform.transform.translation.x = pose.x_m
        transform.transform.translation.y = pose.y_m
        transform.transform.translation.z = 0.0
        _set_quaternion(
            transform.transform.rotation,
            (
                0.0,
                0.0,
                math.sin(pose.yaw_rad / 2.0),
                math.cos(pose.yaw_rad / 2.0),
            ),
        )
        self._tf.sendTransform(transform)

    def _build_frame(
        self,
        map_message: Any,
        pose_message: Any,
        rgb_message: Any,
        depth_message: Any,
        camera_info_message: Any,
    ) -> NavigationFrame:
        rgb = self._bridge.imgmsg_to_cv2(rgb_message, desired_encoding="rgb8")
        raw_depth = self._bridge.imgmsg_to_cv2(
            depth_message,
            desired_encoding="passthrough",
        )
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError("ROS RGB 话题不是 H×W×3 图像")
        if raw_depth.ndim != 2 or raw_depth.shape != rgb.shape[:2]:
            raise RuntimeError("ROS 对齐深度与 RGB 尺寸不一致")
        depth_m = raw_depth.astype(self._np.float32) * self.config.depth_unit_m
        intrinsics = _camera_intrinsics(camera_info_message)
        return NavigationFrame(
            timestamp_s=time.monotonic(),
            pose=_slam_pose(pose_message, self.config.map_frame),
            obstacle_map=_obstacle_map(map_message, self.config),
            depth=_freeze_depth(depth_m),
            rgb=_freeze_rgb(rgb),
            camera_intrinsics=intrinsics,
            camera_extrinsics_in_robot=CameraExtrinsics(
                forward_m=self._camera_mount.forward_m,
                left_m=self._camera_mount.left_m,
                height_m=self._camera_mount.height_m,
                yaw_rad=self._camera_mount.yaw_rad,
                pitch_down_rad=self._camera_mount.pitch_down_rad,
                roll_rad=self._camera_mount.roll_rad,
            ),
        )


def _slam_pose(message: Any, expected_frame: str) -> Pose2D:
    frame_id = str(message.header.frame_id).strip()
    if frame_id != expected_frame:
        raise RuntimeError(
            f"SLAM 位姿坐标系为 {frame_id!r}，预期 {expected_frame!r}"
        )
    pose = message.pose.pose
    return Pose2D(
        x_m=float(pose.position.x),
        y_m=float(pose.position.y),
        yaw_rad=_quaternion_yaw(pose.orientation),
    )


def _obstacle_map(message: Any, config: RosSlamConfig) -> ObstacleMap:
    width = int(message.info.width)
    height = int(message.info.height)
    data = tuple(int(value) for value in message.data)
    if width <= 0 or height <= 0 or len(data) != width * height:
        raise RuntimeError("ROS OccupancyGrid 尺寸或数据长度无效")
    frame_id = str(message.header.frame_id).strip()
    if frame_id != config.map_frame:
        raise RuntimeError(
            f"SLAM 地图坐标系为 {frame_id!r}，预期 {config.map_frame!r}"
        )
    rows = []
    for row_index in range(height):
        row = data[row_index * width : (row_index + 1) * width]
        rows.append(
            tuple(
                _occupancy_value(value, config)
                for value in row
            )
        )
    origin = _cell_center_origin(message.info)
    return ObstacleMap(
        occupancy=tuple(rows),
        resolution_m=float(message.info.resolution),
        origin=origin,
        frame_id=frame_id,
    )


def _cell_center_origin(map_info: Any) -> Pose2D:
    """把 ROS 的左下角原点转换为项目约定的 (0, 0) 栅格中心。"""
    resolution_m = float(map_info.resolution)
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise RuntimeError("ROS OccupancyGrid 分辨率无效")
    corner = map_info.origin
    yaw_rad = _quaternion_yaw(corner.orientation)
    half_cell = resolution_m * 0.5
    return Pose2D(
        x_m=float(corner.position.x)
        + half_cell * (math.cos(yaw_rad) - math.sin(yaw_rad)),
        y_m=float(corner.position.y)
        + half_cell * (math.sin(yaw_rad) + math.cos(yaw_rad)),
        yaw_rad=yaw_rad,
    )


def _occupancy_value(value: int, config: RosSlamConfig) -> Optional[float]:
    if value < 0:
        return None
    if value <= config.free_occupancy_max:
        return 0.0
    if value >= config.occupied_occupancy_min:
        return 1.0
    return None


def _camera_intrinsics(message: Any) -> CameraIntrinsics:
    matrix = tuple(float(value) for value in message.k)
    if len(matrix) != 9 or matrix[0] <= 0.0 or matrix[4] <= 0.0:
        raise RuntimeError("ROS CameraInfo 内参无效")
    return CameraIntrinsics(
        fx=matrix[0],
        fy=matrix[4],
        cx=matrix[2],
        cy=matrix[5],
    )


def _freeze_rgb(image: Any):
    return tuple(
        tuple((int(pixel[0]), int(pixel[1]), int(pixel[2])) for pixel in row)
        for row in image.tolist()
    )


def _freeze_depth(image: Any):
    return tuple(
        tuple(
            float(value) if math.isfinite(float(value)) and value > 0.0 else None
            for value in row
        )
        for row in image.tolist()
    )


def _message_stamp_s(message: Any) -> float:
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _quaternion_yaw(quaternion: Any) -> float:
    sin_yaw = 2.0 * (
        float(quaternion.w) * float(quaternion.z)
        + float(quaternion.x) * float(quaternion.y)
    )
    cos_yaw = 1.0 - 2.0 * (
        float(quaternion.y) ** 2 + float(quaternion.z) ** 2
    )
    return math.atan2(sin_yaw, cos_yaw)


def _mount_quaternion(
    yaw_rad: float,
    pitch_down_rad: float,
    roll_rad: float,
):
    """返回 Rz(yaw)·Ry(pitch-down)·Rx(-roll) 的 xyzw 四元数。"""
    half_yaw = yaw_rad / 2.0
    half_pitch = pitch_down_rad / 2.0
    half_roll = -roll_rad / 2.0
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    return (
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
        cy * cp * cr + sy * sp * sr,
    )


def _set_quaternion(target: Any, values: Tuple[float, float, float, float]) -> None:
    target.x, target.y, target.z, target.w = values


def _validate_config(config: RosSlamConfig) -> None:
    if not isinstance(config, RosSlamConfig):
        raise ValueError("config 必须为 RosSlamConfig")
    for name in (
        "map_topic",
        "pose_topic",
        "rgb_topic",
        "depth_topic",
        "camera_info_topic",
        "map_frame",
        "odom_frame",
        "base_frame",
        "camera_frame",
    ):
        value = getattr(config, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} 必须为非空字符串")
    for name in (
        "frame_timeout_s",
        "stale_after_s",
        "map_stale_after_s",
        "max_rgb_depth_skew_s",
        "odom_publish_hz",
        "depth_unit_m",
    ):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} 必须为正有限数")
    free_max = config.free_occupancy_max
    occupied_min = config.occupied_occupancy_min
    if (
        isinstance(free_max, bool)
        or not isinstance(free_max, int)
        or isinstance(occupied_min, bool)
        or not isinstance(occupied_min, int)
        or not 0 <= free_max < occupied_min <= 100
    ):
        raise ValueError("占用阈值必须满足 0 <= free < occupied <= 100")


__all__ = ["RosSlamConfig", "RosSlamSource"]
