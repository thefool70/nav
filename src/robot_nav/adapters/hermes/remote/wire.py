"""随车发布器四段消息：topic / msgpack 元数据 / RGB8 / 小端 uint16 深度。"""

import math

from ....core.models import CameraIntrinsics
from ...realsense import D435iCapture

MAX_FRAME_BYTES = 32 * 1024 * 1024


def decode_capture(parts, msgpack, np, timestamp_s):
    """按原发布器默认对齐配置解码；校验尺寸，转换米制深度，忽略占位位姿。"""
    if len(parts) != 4 or sum(map(len, parts)) > MAX_FRAME_BYTES:
        raise ValueError("相机消息必须为四段且总大小不超过 32 MiB")
    _, metadata, rgb_bytes, depth_bytes = parts
    if len(metadata) > 64 * 1024:
        raise ValueError("相机元数据超过 64 KiB")
    meta = msgpack.unpackb(metadata, raw=False)
    if not isinstance(meta, dict):
        raise ValueError("相机元数据必须为字典")
    # 原发布器不声明对齐/编码：约定默认开启 depth→color、RGB8，
    # 随车 x86 主机的 uint16 为小端。仅凭相同尺寸无法确认已对齐。
    width, height = meta["width"], meta["height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("相机尺寸必须为正整数")
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
    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
    depth = np.frombuffer(depth_bytes, dtype="<u2").reshape(height, width).astype(np.float32) * scale
    # 使用调用方提供的本机接收时刻，避免把随车端单调时钟当成本机时钟。
    return D435iCapture(timestamp_s, rgb, depth, intrinsics)
