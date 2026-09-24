"""rgbd_pose v2：接收 RGB-D、采集时间和同步底盘位姿。"""

import math
from dataclasses import dataclass

from ....core.models import CameraIntrinsics, Pose2D
from ...realsense import D435iCapture

MAX_FRAME_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class RgbdPoseCapture(D435iCapture):
    """一包同步观测；timestamp_s 是随车端 RGB 采集时刻的 Unix 秒数。"""

    pose: Pose2D


def decode_capture(parts, msgpack, np):
    """校验发布协议，使用彩色内参、米制深度和同一采集时刻的位姿。"""
    if len(parts) != 4 or sum(map(len, parts)) > MAX_FRAME_BYTES:
        raise ValueError("相机消息必须为四段且总大小不超过 32 MiB")
    _, metadata, rgb_bytes, depth_bytes = parts
    if len(metadata) > 64 * 1024:
        raise ValueError("相机元数据超过 64 KiB")
    meta = msgpack.unpackb(metadata, raw=False)
    if not isinstance(meta, dict):
        raise ValueError("相机元数据必须为字典")
    if meta["version"] != 2 or meta["type"] != "rgbd_pose":
        raise ValueError("相机消息必须使用 rgbd_pose v2 协议")
    if meta["depth_aligned_to"] != "color" or meta["depth_frame"] != "color_optical":
        raise ValueError("导航需要对齐到彩色光学坐标系的深度图")
    if meta["rgb_dtype"] != "uint8" or meta["depth_dtype"] != "uint16":
        raise ValueError("相机编码必须为 RGB8 和 uint16 深度")
    sync = meta["sync"]
    if not isinstance(sync, dict):
        raise ValueError("sync 必须为同步诊断字典")
    if meta["pose_valid"] is not True or sync["valid"] is not True:
        raise ValueError(f"发布器未提供有效同步位姿：{sync.get('reason', 'unknown')}")
    timestamp_ns = meta["timestamp_ns"]
    if type(timestamp_ns) is not int or timestamp_ns <= 0:
        raise ValueError("timestamp_ns 必须为正整数 Unix 纳秒")
    # 协议要求发布器保持 depth→color 对齐；随车 x86 的 uint16 为小端。
    # v2 的 depth_intr 描述对齐后深度；共用 color_intr 进行 RGB-D 投影。
    width, height = meta["width"], meta["height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("相机尺寸必须为正整数")
    if (meta["depth_width"], meta["depth_height"]) != (width, height):
        raise ValueError("对齐后的深度尺寸必须与彩色图一致")
    pixels = width * height
    if len(rgb_bytes) != pixels * 3 or len(depth_bytes) != pixels * 2:
        raise ValueError("RGB/深度字节数与对齐尺寸不一致")
    scale = float(meta["depth_scale_m"])
    k = meta["color_intr"]
    intrinsics = CameraIntrinsics(*(float(k[name]) for name in ("fx", "fy", "ppx", "ppy")))
    if not all(math.isfinite(v) for v in (scale, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy)):
        raise ValueError("深度单位和内参必须为有限数")
    if min(scale, intrinsics.fx, intrinsics.fy) <= 0:
        raise ValueError("深度单位和焦距必须为正数")
    map_base = read_transform(meta["T_map_base"], "T_map_base", np)
    pose = Pose2D(float(map_base[0, 3]), float(map_base[1, 3]),
                  math.atan2(map_base[1, 0], map_base[0, 0]))
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
    depth = np.frombuffer(depth_bytes, dtype="<u2").reshape(height, width).astype(np.float32) * scale
    return RgbdPoseCapture(
        timestamp_s=timestamp_ns / 1e9, rgb=rgb, depth_m=depth,
        camera_intrinsics=intrinsics, pose=pose,
    )


def read_transform(value, name, np):
    """外部矩阵采用 p_map = T_map_sensor @ p_sensor，长度单位为米。"""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须为有限的 4×4 矩阵")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} 的末行必须为 [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4)):
        raise ValueError(f"{name} 必须包含有效旋转矩阵")
    return matrix
