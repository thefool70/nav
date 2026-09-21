"""按需订阅随车端 ZMQ RGB-D；每次重新订阅，避免导航暂停期间积压旧帧。"""

import importlib
import math
import threading
import time

from .wire import MAX_FRAME_BYTES, decode_capture


class RemoteD435iCamera:
    """capture 可由不同采样线程串行调用；ZMQ socket 不跨线程共享。

    时间戳使用开发机接收时刻。等待上限不是跨机传感器到接收端的帧龄保证。
    """

    def __init__(self, endpoint, *, topic="ngd.frame", timeout_s=3.0):
        if not endpoint.startswith("ipc:///"):
            raise ValueError("相机地址必须为绝对路径 ZMQ ipc:/// 地址")
        if not topic or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("相机 topic 不能为空，等待时间必须为正有限秒数")
        try:
            self._zmq = importlib.import_module("zmq")
            self._msgpack = importlib.import_module("msgpack")
            self._np = importlib.import_module("numpy")
        except ImportError as exc:
            raise RuntimeError("ZMQ 相机需要安装项目的 [remote-camera] 可选依赖") from exc
        self._endpoint = endpoint
        self._topic = topic.encode("utf-8")
        self._timeout_s = timeout_s
        self._closed = False
        self._lock = threading.Lock()

    def capture(self):
        """新建订阅并等待新消息；断流超时报错，不复用上一次画面。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("远程 D435i 采集端已关闭")
            zmq = self._zmq
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            started = time.monotonic()
            try:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVHWM, 1)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_FRAME_BYTES)
                socket.setsockopt(zmq.SUBSCRIBE, self._topic)
                socket.connect(self._endpoint)
                # SUB 的 topic 是前缀匹配；按剩余期限跳过不完全匹配的消息。
                while True:
                    remaining = self._timeout_s - (time.monotonic() - started)
                    if remaining <= 0 or not socket.poll(max(1, int(remaining * 1000)), zmq.POLLIN):
                        raise RuntimeError("等待 ZMQ RGB-D 超时，请检查发布器与 SSH 隧道")
                    parts = socket.recv_multipart()
                    received = time.monotonic()
                    if received - started > self._timeout_s:
                        raise RuntimeError("ZMQ RGB-D 读取超过等待上限")
                    if parts and parts[0] == self._topic:
                        return decode_capture(parts, self._msgpack, self._np, received)
            except (zmq.ZMQError, ValueError, KeyError, TypeError, OverflowError) as exc:
                raise RuntimeError(f"随车端 ZMQ RGB-D 数据读取失败：{exc}") from exc
            finally:
                socket.close(0)
                context.term()

    def close(self):
        """等待正在进行的有限时读取结束；后续调用拒绝读取。"""
        with self._lock:
            self._closed = True
