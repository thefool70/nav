"""内存快照：固定同帧 RGB-D、位姿、标定和 YOLOE 结果；队列负责持有与释放。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from itertools import chain
from typing import Optional, Tuple

import numpy as np

from ..core.models import (CameraExtrinsics, CameraIntrinsics, NavigationFrame, Pose2D,
                           RgbImage, ObservationView, TargetObservation)
from .analyzer import VlmInputImage


@dataclass(frozen=True)
class BufferedScanImage:
    """不携带地图的紧凑扫描帧，避免把大图放入感知缓冲。"""

    width_px: int
    height_px: int
    rgb_bytes: bytes
    pose: Pose2D
    intrinsics: CameraIntrinsics
    camera_yaw_rad: float
    camera_extrinsics_in_robot: CameraExtrinsics = field(default_factory=CameraExtrinsics)


@dataclass(frozen=True)
class CapturedView:
    """一张固定画面及覆盖；RGB-D 和同帧检测留在内存，不携带地图。"""

    image: BufferedScanImage
    coverage: ObservationView
    map_frame_id: str
    depth: Optional[np.ndarray] = None
    source: str = "scan"
    yolo_observations: Tuple[TargetObservation, ...] = ()


def buffer_scan_image(
    frame: NavigationFrame,
) -> BufferedScanImage:
    """保留原始分辨率与同帧标定，让归一化框直接对应历史 RGB-D。"""
    if frame.rgb is None or frame.camera_intrinsics is None:
        raise ValueError("Frontier 评分需要 RGB 和相机内参")
    packed = pack_rgb_image(frame.rgb)
    return BufferedScanImage(
        width_px=packed.width_px,
        height_px=packed.height_px,
        rgb_bytes=packed.rgb_bytes,
        pose=frame.pose,
        intrinsics=frame.camera_intrinsics,
        camera_yaw_rad=frame.camera_extrinsics_in_robot.yaw_rad,
        camera_extrinsics_in_robot=frame.camera_extrinsics_in_robot,
    )


def pack_rgb_image(image: RgbImage) -> VlmInputImage:
    """按行打包已由 Adapter 规范化的 RGB，避免逐通道 Python 调用与赋值。"""
    pixels = chain.from_iterable(image)
    return VlmInputImage(
        width_px=len(image[0]),
        height_px=len(image),
        rgb_bytes=bytes(chain.from_iterable(pixels)),
    )


def freeze_depth(depth, width: int, height: int) -> Optional[np.ndarray]:
    """独立保存米制 float32 深度；缺测为 NaN，只读数组可在线程间共享。"""
    if depth is None:
        return None
    frozen = np.array(depth, dtype=np.float32, copy=True)
    if frozen.shape != (height, width):
        raise ValueError("快照深度必须与 RGB 对齐且尺寸相同")
    frozen.flags.writeable = False
    return frozen


def snapshot_bytes(view: CapturedView) -> int:
    """供队列限制图像内存；不包含小型元数据和 Python 对象开销。"""
    return (len(view.image.rgb_bytes) + (view.depth.nbytes if view.depth is not None else 0)
            + sum(item.target_mask.nbytes for item in view.yolo_observations
                  if item.target_mask is not None))


def clue_frame(view: CapturedView, current: NavigationFrame) -> NavigationFrame:
    """历史图像、位姿和标定配当前同坐标系地图，供深度和障碍射线定位。"""
    image = view.image
    if view.map_frame_id != current.obstacle_map.frame_id:
        raise ValueError("历史图像与当前地图坐标系不同")
    rgb = np.frombuffer(image.rgb_bytes, dtype=np.uint8).reshape(image.height_px, image.width_px, 3)
    return replace(current, rgb=rgb, depth=view.depth, pose=image.pose,
                   timestamp_s=view.coverage.timestamp_s, camera_intrinsics=image.intrinsics,
                   camera_extrinsics_in_robot=image.camera_extrinsics_in_robot)


def view_trace(view: CapturedView):
    """画面编号、拍摄时间与机器人位姿用于关联 VLM 请求，不携带覆盖点大数组。"""
    return {"view_id": 1, "map_frame_id": view.map_frame_id,
            "timestamp_s": view.coverage.timestamp_s, "pose": asdict(view.coverage.pose),
            "heading_world_rad": view.coverage.camera_heading_world_rad}
