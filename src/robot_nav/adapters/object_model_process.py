"""YOLOE 常驻进程；RGB 管道、阶段进度与超时收尾。"""

from __future__ import annotations

from dataclasses import dataclass
import sys
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import threading
import time


@dataclass(frozen=True)
class ObjectModelConfig:
    """YOLOE 的运行环境与视频采样频率；扫描图另外按需检测一次。"""

    python_executable: str = sys.executable
    class_text: str = ""
    device: str = "cuda"
    yolo_model: Path = Path("data/models/yoloe/yoloe-26s-seg.pt")
    timeout_s: float = 120.0
    frequency_hz: float = 10.0


class ObjectModelProcess:
    """持有一个 YOLOE 进程，扫描和视频请求通过互斥锁串行执行。"""

    def __init__(self, python: str, directory: Path, timeout_s: float):
        self.python = python
        self.directory = directory
        self.timeout_s = timeout_s
        self._process = None
        self._messages = Queue()
        self._request_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._closed = threading.Event()

    def request(self, payload, on_progress=lambda *_: None, *, rgb_bytes=None):
        """串行请求模型；RGB 和掩码走管道，磁盘仅保留模型错误日志。"""
        started = time.monotonic()
        stage = "starting"

        def failed(reason):
            result = {"error": reason}
            on_progress("yoloe", f"failed: {reason}", time.monotonic() - started)
            return result

        if not self._request_lock.acquire(timeout=self.timeout_s):
            return failed(f"等待 YOLOE 上一帧超时")
        try:
            if self._closed.is_set():
                return failed(f"YOLOE 进程已关闭")
            request = dict(payload)
            if rgb_bytes is not None:
                request["rgb_length"] = len(rgb_bytes)
            process, messages = self._ensure_started()
            process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            if rgb_bytes is not None:
                process.stdin.write(rgb_bytes)
            process.stdin.flush()
            last_report = 0.0
            while not self._closed.is_set():
                elapsed = time.monotonic() - started
                if elapsed >= self.timeout_s:
                    self._stop()
                    return failed(f"YOLOE 在 {stage} 超时（{elapsed:.1f}s）")
                try:
                    message = messages.get(timeout=0.2)
                except Empty:
                    if elapsed - last_report >= 5.0:
                        on_progress("yoloe", stage, elapsed)
                        last_report = elapsed
                    continue
                if message is None:
                    code = process.poll()
                    self._stop()
                    return failed(f"YOLOE 进程在 {stage} 退出或输出中断（code={code}）")
                if message["event"] == "result":
                    if message["result"].get("error"):
                        return failed(message["result"]["error"])
                    return message["result"]
                stage = message["stage"]
                on_progress("yoloe", stage, elapsed)
                last_report = elapsed
            return failed(f"YOLOE 进程已关闭")
        except (OSError, ValueError, RuntimeError) as exc:
            self._stop()
            return failed(f"YOLOE 启动或通信失败：{exc}")
        finally:
            self._request_lock.release()

    def close(self):
        self._closed.set()
        self._stop()

    def _ensure_started(self):
        with self._process_lock:
            if self._closed.is_set():
                raise RuntimeError("模型进程已关闭")
            return self._start_process()

    def _start_process(self):
        """复用存活模型进程，否则以指定 Python 启动；stdout 只传消息，stderr 写模型日志。"""
        if self._process is not None and self._process.poll() is None:
            return self._process, self._messages
        if self._process is not None:
            self._process.stdin.close()
        self.directory.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        for name in ("LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"):
            environment.pop(name, None)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        messages = Queue()
        self._messages = messages
        with (self.directory / "yoloe.log").open("a") as log:
            self._process = subprocess.Popen(
                [self.python, "-u", "-m", "robot_nav.adapters.object_detection_worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                env=environment,
            )
        threading.Thread(
            target=_read_messages, args=(self._process.stdout, messages), daemon=True,
            name="yoloe-output",
        ).start()
        return self._process, messages

    def _stop(self):
        """先终止、超时再强制结束模型进程，并释放请求管道。"""
        with self._process_lock:
            process = self._process
            self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        try:
            process.stdin.close()
        except OSError:
            pass


def _read_messages(stream, messages):
    """把逐行 JSON 协议转入线程队列；流结束或损坏时用 None 通知请求方。"""
    try:
        for line in stream:
            message = json.loads(line)
            if not isinstance(message, dict) or message.get("event") not in ("progress", "result"):
                raise ValueError("invalid model message")
            messages.put(message)
    except (OSError, ValueError):
        pass
    finally:
        stream.close()
        messages.put(None)
