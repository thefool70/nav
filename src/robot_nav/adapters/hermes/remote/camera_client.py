"""持续接收随车端 RGB-D + 同步位姿，只保留最新完整包。"""

import importlib
import math
import threading
import time

from .wire import MAX_FRAME_BYTES, decode_capture


class RemoteD435iCamera:
    """接收线程独占 ZMQ socket；调用方只读取完整快照。

    采集时间保留随车端 Unix 秒数，不与本机单调计时器混算。
    等待上限不是跨机传感器到接收端的帧龄保证。
    """

    def __init__(self, endpoint, *, topic="rgbd.pose", timeout_s=3.0):
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
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._received_s = 0.0
        self._error = None
        self._thread = threading.Thread(target=self._receive_loop, name="rgbd-receiver", daemon=True)
        self._thread.start()

    def capture(self, *, after_s=0.0):
        """读取最新包；启动或动作结束后只等下一包，不重新连接。after_s 为本机单调秒。"""
        with self._condition:
            ready = self._condition.wait_for(lambda: (
                self._error is not None or self._stop.is_set()
                or (self._latest is not None and self._received_s >= after_s)
            ), timeout=self._timeout_s)
            if self._error is not None:
                raise RuntimeError(f"随车相机接收失败：{self._error}") from self._error
            if self._stop.is_set():
                raise RuntimeError("远程 D435i 采集端已关闭")
            if not ready or time.monotonic() - self._received_s > self._timeout_s:
                raise RuntimeError("等待新 RGB-D/位姿超时，请检查发布器与 SSH 隧道")
            return self._latest

    def _receive_loop(self):
        """持续排空消息，覆盖旧包；socket 在此线程创建和关闭。"""
        context = None
        socket = None
        try:
            zmq = self._zmq
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVHWM, 1)
            socket.setsockopt(zmq.MAXMSGSIZE, MAX_FRAME_BYTES)
            socket.setsockopt(zmq.SUBSCRIBE, self._topic)
            socket.connect(self._endpoint)
            last_received = time.monotonic()
            while not self._stop.is_set():
                if not socket.poll(100, zmq.POLLIN):
                    if time.monotonic() - last_received > self._timeout_s:
                        raise RuntimeError("ZMQ RGB-D/位姿断流超时")
                    continue
                parts = socket.recv_multipart()
                if not parts or parts[0] != self._topic:
                    if time.monotonic() - last_received > self._timeout_s:
                        raise RuntimeError("ZMQ 未收到匹配 topic 的数据")
                    continue
                received = time.monotonic()
                capture = decode_capture(parts, self._msgpack, self._np)
                with self._condition:
                    self._latest = capture
                    self._received_s = received
                    self._condition.notify_all()
                last_received = received
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if socket is not None:
                socket.close(0)
            if context is not None:
                context.term()

    def close(self):
        """唤醒读取者并等待接收线程释放 socket。"""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join()
