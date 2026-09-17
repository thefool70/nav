"""开发机的远程 RealSense 采集端；通过 SSH 转发后的 HTTP 地址获取新帧。"""

import importlib
import json
import math
import time
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .wire import MAX_FRAME_BYTES, decode_capture


class RemoteD435iCamera:
    """提供 Hermes Adapter 所需的 capture/close；不在开发机打开 USB 相机。"""

    def __init__(self, base_url, *, timeout_s=5.0, max_roundtrip_s=3.0):
        parts = urlsplit(base_url)
        if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("远程相机应使用 SSH 隧道的本地 HTTP 地址")
        if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("相机地址不能包含账号、路径或查询参数")
        for value in (timeout_s, max_roundtrip_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("相机超时与最大往返时间必须为正有限秒数")
        self._np = importlib.import_module("numpy")
        self._url = base_url.rstrip("/") + "/frame"
        self._prepare_url = base_url.rstrip("/") + "/prepare"
        self._imu_url = base_url.rstrip("/") + "/imu"
        self._timeout_s = timeout_s
        self._max_roundtrip_s = max_roundtrip_s
        self._opener = build_opener(ProxyHandler({}))
        self._closed = False

    def capture(self):
        if self._closed:
            raise RuntimeError("远程 D435i 采集端已关闭")
        try:
            # 每次取帧前确认相机已唤醒；SDK 冷启动单独允许至少 15 秒。
            with self._opener.open(Request(self._prepare_url, data=b"", method="POST"),
                                   timeout=max(15.0, self._timeout_s)) as response:
                if response.status != 204:
                    raise RuntimeError("香橙派未确认相机已唤醒")
            started = time.monotonic()
            with self._opener.open(Request(self._url, method="GET"), timeout=self._timeout_s) as response:
                payload = response.read(MAX_FRAME_BYTES + 1)
            received = time.monotonic()
            if received - started > self._max_roundtrip_s:
                raise RuntimeError(
                    f"D435i 帧往返耗时 {received - started:.2f}s，超过 {self._max_roundtrip_s:.2f}s；拒绝滞后图像"
                )
            return decode_capture(payload, self._np, received)
        except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError, EOFError, zlib.error) as exc:
            raise RuntimeError(f"香橙派 D435i 数据读取失败：{exc}") from exc

    def close(self):
        self._closed = True

    def read_imu_samples(self):
        """读取一批原始 IMU 与 depth→color 旋转；其时间戳不与开发机时钟直接比较。"""
        if self._closed:
            raise RuntimeError("远程 D435i 采集端已关闭")
        try:
            with self._opener.open(Request(self._imu_url, data=b"", method="POST"),
                                   timeout=max(30.0, self._timeout_s)) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise ValueError("IMU 响应超过大小上限")
            return json.loads(raw)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            raise RuntimeError(f"香橙派 D435i IMU 读取失败：{exc}") from exc
