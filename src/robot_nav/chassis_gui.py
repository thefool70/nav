"""本地浏览器底盘操作面板：python -m robot_nav.chassis_gui。"""

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

from .adapters.hermes.rest_client import HermesRestClient


class ChassisControls:
    """显式步进操作；查询不控制运动，取消会使尚未提交的操作失效。"""

    def __init__(self, base_url):
        self.base_url = base_url
        self._commands = threading.Lock()
        self._submission = threading.Lock()
        self._generation = 0

    def client(self):
        # 每个请求独立客户端，避免状态查询与取消共用网络对象。
        return HermesRestClient(self.base_url, request_timeout_s=3.0)

    def status(self):
        client = self.client()
        return {"base_url": self.base_url, "pose": asdict(client.get_pose()),
                "slam": asdict(client.get_slam_state()),
                "health": asdict(client.get_robot_health()),
                "power": client.get_power_status(), "action": client.get_current_action()}

    def stop(self):
        # 与最后一次提交串行：取消之后不能再发出之前仍在检查的动作。
        with self._submission:
            self._generation += 1
            self.client().abort_current_action()
        return {"message": "取消请求已受理，请观察任务状态与实际底盘是否停稳。"}

    def move(self, request):
        if request.get("enabled") is not True:
            raise ValueError("请先勾选允许运动")
        kind = request.get("kind")
        if kind not in ("forward", "turn", "dock"):
            raise ValueError("未知操作")
        value = 0.0 if kind == "dock" else float(request.get("value", 0))
        limit = 0.5 if kind == "forward" else 90.0
        if kind != "dock" and (not math.isfinite(value) or not 0 < abs(value) <= limit):
            raise ValueError(f"步长必须非零且绝对值不超过 {limit}")
        if not self._commands.acquire(blocking=False):
            raise ValueError("上一操作仍在提交，请等待")
        try:
            with self._submission:
                generation = self._generation
            client = self.client()
            self._require_ready(client)
            names = client.get_action_names()
            suffix = {"forward": "MoveToAction", "turn": "RotateToAction", "dock": "GoHomeAction"}[kind]
            name = next((n for n in names if n.endswith("." + suffix)), None)
            if name is None:
                raise ValueError(f"当前固件未提供 {suffix}")
            options = self._options(client, kind, value)
            with self._submission:
                if generation != self._generation:
                    raise ValueError("操作已被停止按钮取消")
                if kind == "dock":
                    result = client.go_home()
                    action_id = result.get("action_id")
                else:
                    action_id = client.create_action(name, options)
            return {"message": f"任务已提交，编号 {action_id}；完成情况以状态和实际底盘为准。"}
        finally:
            self._commands.release()

    @staticmethod
    def _require_ready(client):
        health = client.get_robot_health()
        if health.has_error or health.has_fatal:
            raise ValueError("底盘健康异常，拒绝运动")
        slam = client.get_slam_state()
        if slam.mode == "odometry" or (slam.mode == "localization" and slam.localization_quality < 1):
            raise ValueError("底盘未可靠定位，拒绝运动")
        action = client.get_current_action()
        if action and action.get("state", {}).get("status") != 4:
            raise ValueError("底盘已有任务，请等待结束或先取消；不要与导航程序同时控制")

    @staticmethod
    def _options(client, kind, value):
        if kind == "dock":
            return {}
        pose = client.get_pose()
        if kind == "turn":
            angle = pose.yaw_rad + math.radians(value)
            return {"angle": math.atan2(math.sin(angle), math.cos(angle))}
        return {"target": {"x": pose.x_m + value * math.cos(pose.yaw_rad),
                           "y": pose.y_m + value * math.sin(pose.yaw_rad), "z": 0.0}}


def main():
    parser = argparse.ArgumentParser(description="Hermes 本地浏览器操作面板")
    parser.add_argument("--base-url", default="http://127.0.0.1:11448")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    controls = ChassisControls(args.base_url)
    controls.client()  # 仅校验地址，不连接底盘。
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(controls, secrets.token_hex(24)))
    print(f"底盘面板：http://127.0.0.1:{server.server_port}  →  {args.base_url}", flush=True)
    print("关闭页面或服务不会取消底盘任务；退出前请点击停止并确认停稳。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def handler_for(controls, token):
    """只允许本地页面访问固定操作；不提供任意底盘 API 代理。"""
    page = Path(__file__).with_name("chassis_gui.html").read_text().replace("__TOKEN__", token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, code, payload, html=False):
            raw = (payload if html else json.dumps(payload, ensure_ascii=False)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(raw)

        def allowed(self):
            host = urlsplit("http://" + self.headers.get("Host", "")).hostname
            return host in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self.allowed():
                return self.respond(403, {"error": "仅允许本机访问"})
            if self.path == "/":
                return self.respond(200, page, html=True)
            if self.path != "/status" or self.headers.get("X-Control-Token") != token:
                return self.respond(404, {"error": "未找到"})
            try:
                self.respond(200, controls.status())
            except (RuntimeError, ValueError) as exc:
                self.respond(502, {"error": str(exc)})

        def do_POST(self):
            if not self.allowed() or self.headers.get("X-Control-Token") != token:
                return self.respond(403, {"error": "请使用本机操作页面"})
            if self.path not in ("/move", "/stop"):
                return self.respond(404, {"error": "未找到"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2048:
                    raise ValueError("请求长度无效")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("请求格式无效")
                result = controls.stop() if self.path == "/stop" else controls.move(request)
                self.respond(200, result)
            except (ValueError, TypeError) as exc:
                self.respond(400, {"error": str(exc)})
            except RuntimeError as exc:
                self.respond(502, {"error": f"{exc}；若发送时通信失败，任务可能已受理，请查询状态，不要重复点击。"})

    return Handler


if __name__ == "__main__":
    main()
