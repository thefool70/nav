"""D435i 传输格式：小型 JSON 头、RGB8 和小端 float32 米制深度，整体 gzip。"""

from dataclasses import asdict
import gzip
import io
import json
import math
import struct

from ...core.models import CameraIntrinsics
from ..realsense.l515_camera import L515Capture


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 32 * 1024 * 1024
MAX_HEADER_BYTES = 16 * 1024


def encode_capture(capture):
    """只编码相机数据，不携带地图、导航状态或运动命令。"""
    rgb = capture.rgb.astype("uint8", copy=False)
    depth = capture.depth_m.astype("<f4", copy=False)
    height, width = depth.shape
    if rgb.shape != (height, width, 3):
        raise ValueError("RGB 与深度尺寸不一致")
    header = json.dumps({
        "version": PROTOCOL_VERSION, "width": width, "height": height,
        "capture_timestamp_s": capture.timestamp_s,
        "intrinsics": asdict(capture.camera_intrinsics),
        "rgb_format": "rgb8", "depth_format": "float32_le_m",
    }, separators=(",", ":"), allow_nan=False).encode("utf-8")
    raw = struct.pack("<I", len(header)) + header + rgb.tobytes(order="C") + depth.tobytes(order="C")
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("相机帧超过 32 MiB 传输上限")
    return gzip.compress(raw, compresslevel=1)


def decode_capture(payload, np, timestamp_s):
    """还原 RGB-D；timestamp_s 由调用方提供开发机本地时间，不混用两台机器时钟。"""
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("相机响应超过 32 MiB 传输上限")
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        raw = stream.read(MAX_FRAME_BYTES + 1)
    if not 4 <= len(raw) <= MAX_FRAME_BYTES:
        raise ValueError("相机响应长度无效")
    header_size = struct.unpack_from("<I", raw)[0]
    if not 0 < header_size <= MAX_HEADER_BYTES or len(raw) < 4 + header_size:
        raise ValueError("相机响应头长度无效")
    header = json.loads(raw[4:4 + header_size])
    if (header["version"] != PROTOCOL_VERSION or header["rgb_format"] != "rgb8"
            or header["depth_format"] != "float32_le_m"):
        raise ValueError("香橙派相机协议不匹配，请同步两端代码")
    width, height = header["width"], header["height"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("相机尺寸必须为正整数")
    pixels = width * height
    offset = 4 + header_size
    if len(raw) != offset + pixels * 7:
        raise ValueError("相机数组长度与尺寸不匹配")
    k = header["intrinsics"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in k.values()):
        raise ValueError("相机内参必须为有限数")
    intrinsics = CameraIntrinsics(**k)
    if min(intrinsics.fx, intrinsics.fy) <= 0:
        raise ValueError("相机焦距必须为正值")
    rgb = np.frombuffer(raw, dtype=np.uint8, count=pixels * 3, offset=offset).reshape(height, width, 3).copy()
    depth = np.frombuffer(raw, dtype="<f4", count=pixels, offset=offset + pixels * 3).reshape(height, width).copy()
    return L515Capture(timestamp_s, rgb, depth, intrinsics)
