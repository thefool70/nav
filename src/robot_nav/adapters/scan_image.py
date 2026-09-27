"""冻结单帧 RGB 和相机标定；模型输入不叠加编号、前沿点或页眉。"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain

from ..core.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    NavigationFrame,
    Pose2D,
    RgbImage,
)
from .perception import VlmInputImage


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
