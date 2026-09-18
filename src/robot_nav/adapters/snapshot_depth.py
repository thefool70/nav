"""视觉快照的无损深度编码：gzip 压缩的小端 float64，长度单位为米。"""

import gzip
import math
import struct
import sys
from array import array
from typing import Optional

from ..core.models import DepthImage


_HEADER = struct.Struct("<8sII")
_MAGIC = b"RNDEPTH1"


def encode_depth(depth: Optional[DepthImage], width: int, height: int) -> Optional[bytes]:
    """复制同帧深度为二进制；None 用零表示，避免逐像素 JSON 浮点格式化。"""
    if depth is None:
        return None
    if len(depth) != height or any(len(row) != width for row in depth):
        raise ValueError("快照深度必须与 RGB 对齐且尺寸相同")
    values = array("d", (0.0 if value is None else value for row in depth for value in row))
    if values.itemsize != 8:
        raise ValueError("当前平台不支持 8 字节 double 深度快照")
    if sys.byteorder != "little":
        values.byteswap()
    return gzip.compress(_HEADER.pack(_MAGIC, width, height) + values.tobytes(), compresslevel=1, mtime=0)


def decode_depth(payload: bytes, width: int, height: int) -> DepthImage:
    """检查格式、尺寸和长度，恢复米制深度；非正数与非有限数恢复为 None。"""
    raw = gzip.decompress(payload)
    if len(raw) != _HEADER.size + width * height * 8:
        raise ValueError("历史深度数据长度错误")
    if _HEADER.unpack_from(raw) != (_MAGIC, width, height):
        raise ValueError("历史深度格式或尺寸不匹配")
    values = array("d")
    values.frombytes(raw[_HEADER.size:])
    if values.itemsize != 8:
        raise ValueError("当前平台不支持 8 字节 double 深度快照")
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(
        tuple(value if math.isfinite(value) and value > 0.0 else None for value in values[start:start + width])
        for start in range(0, width * height, width)
    )
