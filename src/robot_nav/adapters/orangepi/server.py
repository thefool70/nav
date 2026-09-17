"""香橙派上的 D435i 数据服务：只采集 RGB-D，无底盘控制、地图处理和模型。"""

import argparse
from dataclasses import replace
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import threading
import time

from ..realsense.d435i_camera import D435iCamera, D435iConfig
from ..realsense.d435i_motion import read_d435i_imu_samples
from .wire import encode_capture


class CameraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port, camera_config, idle_timeout_s):
        if not math.isfinite(idle_timeout_s) or idle_timeout_s <= 0:
            raise ValueError("相机空闲超时必须为正有限秒数")
        self.camera_config = camera_config
        self.camera = None
        self.idle_timeout_s = idle_timeout_s
        self.last_used_s = 0.0
        self.stopping = False
        self.capture_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), CameraRequestHandler)

    def _prepare_locked(self):
        """持锁唤醒相机；后台服务启动本身不打开 USB 数据流。"""
        if self.stopping:
            raise RuntimeError("相机服务正在关闭")
        if self.camera is None:
            self.camera = D435iCamera(self.camera_config)
            self.camera_config = replace(self.camera_config, serial_number=self.camera.serial_number)
            print("D435i 收到请求，已开启数据流。", flush=True)
        self.last_used_s = time.monotonic()

    def prepare_camera(self):
        with self.capture_lock:
            self._prepare_locked()

    def capture_frame(self):
        with self.capture_lock:
            self._prepare_locked()
            try:
                capture = self.camera.capture()
            except Exception:
                self._close_camera_locked()
                raise
            self.last_used_s = time.monotonic()
            return capture

    def read_imu_samples(self):
        """固定设备编号后切换到短时 IMU 采集，结束即释放；下一帧重新打开 RGB-D。"""
        with self.capture_lock:
            self._prepare_locked()
            serial = self.camera.serial_number
            rotation = self.camera.depth_to_color_rotation
            self._close_camera_locked()
            samples = read_d435i_imu_samples(serial)
            samples["depth_to_color_rotation"] = rotation
            return samples

    def _close_camera_locked(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None
            print("D435i 数据流已关闭，等待下次请求。", flush=True)

    def service_actions(self):
        """HTTP 主循环定期回收空闲相机，不与取帧或唤醒并发关闭。"""
        if not self.capture_lock.acquire(blocking=False):
            return
        try:
            if self.camera is not None and time.monotonic() - self.last_used_s >= self.idle_timeout_s:
                self._close_camera_locked()
        finally:
            self.capture_lock.release()

    def server_close(self):
        with self.capture_lock:
            self.stopping = True
            self._close_camera_locked()
        super().server_close()


class CameraRequestHandler(BaseHTTPRequestHandler):
    """只支持读取新帧，HTTP 回环监听由 SSH 提供认证与加密。"""

    def setup(self):
        super().setup()
        self.connection.settimeout(5.0)

    def do_GET(self):
        if self.path != "/frame":
            self.send_error(404, "Only /frame is available")
            return
        try:
            capture = self.server.capture_frame()
            payload = encode_capture(capture)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print(f"D435i 数据传输失败：{exc}", flush=True)
            try:
                self.send_error(503, "D435i capture unavailable")
            except OSError:
                pass

    def do_POST(self):
        # 唤醒独立于取帧，冷启动耗时不占用开发机的图像往返时间门槛。
        if self.path not in ("/prepare", "/imu"):
            self.send_error(404, "Only /prepare and /imu are available")
            return
        try:
            if self.path == "/prepare":
                self.server.prepare_camera()
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                payload = json.dumps(self.server.read_imu_samples(), allow_nan=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            print(f"D435i {self.path} 请求失败：{exc}", flush=True)
            try:
                self.send_error(503, "D435i request unavailable")
            except OSError:
                pass

    def log_message(self, format, *args):
        pass


def serve(camera_config, *, port=8765, idle_timeout_s=900.0):
    with CameraHTTPServer(port, camera_config, idle_timeout_s) as server:
        print(f"D435i 数据服务：127.0.0.1:{port}；按需开启相机，空闲 {idle_timeout_s:g}s 关闭数据流。", flush=True)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="香橙派 D435i RGB-D 数据转发服务（不控制底盘）")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera-serial", help="指定 D435i 序列号")
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--idle-timeout-s", type=float, default=900.0, help="无取帧请求后关闭相机的秒数，默认 900（15 分钟）")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port 必须位于 1–65535")
    if not math.isfinite(args.idle_timeout_s) or args.idle_timeout_s <= 0:
        parser.error("idle-timeout-s 必须为正有限秒数")
    config = D435iConfig(
        color_width=args.color_width, color_height=args.color_height,
        depth_width=args.depth_width, depth_height=args.depth_height,
        fps=args.fps, serial_number=args.camera_serial,
    )
    serve(config, port=args.port, idle_timeout_s=args.idle_timeout_s)


if __name__ == "__main__":
    main()
